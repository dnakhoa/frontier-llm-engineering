# %% [markdown]
# # Lab 22 — KV cache, continuous batching, and paged allocation
#
# Companion to [Chapter 22](../book/part-4-infra/22-inference-serving.md).
#
# Serving throughput is decided by two things that have nothing to do with the
# model: **when you schedule** and **how you allocate**. This lab implements
# both, and the numbers that come out are the ones that separate a serving
# stack from `model.generate()`.
#
# You will:
#
# 1. Model a realistic request stream, where output lengths are heavy-tailed.
# 2. Implement static and continuous batching and measure the throughput ratio.
# 3. Implement a naive KV allocator and measure its fragmentation.
# 4. Implement paged allocation and measure it again.
# 5. Add prefix sharing and see what a shared system prompt is worth.
# 6. Sweep the block size and find the trade the block table imposes.
#
# **What this is and is not.** The allocators and the scheduler here are real
# implementations doing real bookkeeping — block tables, refcounts, eviction.
# The *timing* is a model: one decode iteration is one unit of time regardless
# of batch size, which is roughly true for the memory-bound decode phase and
# not true for prefill. Treat throughput ratios as informative and absolute
# latencies as fiction. Same disclaimer as
# [`lab06`](lab06_parallelism_memory_model.py) and
# [`lab07`](lab07_collective_bandwidth.py).
#
# Pure standard library. Runs instantly.
#
# **Predict before you run.** Eight requests share one slot budget; one
# generates 2,000 tokens and seven generate 50. What fraction of the static
# batch's compute is wasted? And what fraction of a naive KV reservation goes
# unused? Write both down.

# %%
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
SEED = 0
print(f"smoke_test={SMOKE}")

# Llama-3-70B, GQA 8:1, BF16 -- the configuration worked through in §22.3.
LAYERS, KV_HEADS, HEAD_DIM, DTYPE_BYTES = 80, 8, 128, 2
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES
GB = 1e9

print(f"\nKV cache: {BYTES_PER_TOKEN:,} bytes/token "
      f"({BYTES_PER_TOKEN/1e3:.0f} kB), matching §22.3")

# %% [markdown]
# ## 1. A request stream that looks like real traffic
#
# The single fact that makes serving hard: **output lengths are heavy-tailed.**
# Most replies are short, a few are enormous, and you cannot tell which is
# which until it stops.

# %%
MAX_LEN = 8192


@dataclass
class Request:
    rid: int
    prompt_len: int
    output_len: int          # not known to the scheduler in advance
    generated: int = 0

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len


def make_requests(n: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        prompt = int(min(rng.lognormvariate(6.0, 0.7), 4000)) + 16
        if rng.random() < 0.10:                    # the tail: long generations
            out = rng.randint(800, 2500)
        else:
            out = rng.randint(20, 200)
        out = min(out, MAX_LEN - prompt)
        reqs.append(Request(i, prompt, max(out, 1)))
    return reqs


N_REQ = 120 if SMOKE else 600
REQUESTS = make_requests(N_REQ)
outs = sorted(r.output_len for r in REQUESTS)

print(f"\n{N_REQ} requests")
print("-" * 74)
print(f"{'median output':<26}: {outs[len(outs)//2]:>7,} tokens")
print(f"{'mean output':<26}: {sum(outs)/len(outs):>7,.0f} tokens")
print(f"{'p99 output':<26}: {outs[int(0.99*len(outs))]:>7,} tokens")
print(f"{'longest output':<26}: {outs[-1]:>7,} tokens")
print(f"{'ratio p99 / median':<26}: {outs[int(0.99*len(outs))]/max(outs[len(outs)//2],1):>7.1f}x")
print("\nThat ratio is the whole problem. A batch runs until its slowest member")
print("finishes, and the slowest is many times the median.")

# %% [markdown]
# ## 2. Static versus continuous batching
#
# **Static:** take $B$ requests, run until all are done, then take the next
# $B$. A finished sequence's slot sits idle until the batch completes.
#
# **Continuous** (iteration-level scheduling, from Orca): make the decision
# every *token*. The moment a sequence finishes, a waiting request takes its
# slot.

# %%
BATCH = 32


def run_static(requests: list, batch: int) -> dict:
    """Iterations, and how many slot-steps did useful work."""
    iterations = 0
    useful = 0
    for start in range(0, len(requests), batch):
        group = requests[start:start + batch]
        longest = max(r.output_len for r in group)
        iterations += longest
        useful += sum(r.output_len for r in group)
    return {"iterations": iterations, "useful_slot_steps": useful,
            "slot_steps": iterations * batch}


def run_continuous(requests: list, batch: int) -> dict:
    """One decode iteration advances every active sequence by one token."""
    queue = list(requests)
    active: list = []
    iterations = 0
    useful = 0
    while queue or active:
        while queue and len(active) < batch:
            active.append(queue.pop(0))
        iterations += 1
        useful += len(active)
        for r in active:
            r.generated += 1
        active = [r for r in active if not r.done]
    return {"iterations": iterations, "useful_slot_steps": useful,
            "slot_steps": iterations * batch}


fresh = lambda: make_requests(N_REQ)          # noqa: E731
static = run_static(fresh(), BATCH)
cont = run_continuous(fresh(), BATCH)

print(f"\nServing {N_REQ} requests with {BATCH} slots")
print("-" * 74)
print(f"{'scheduler':<22} {'iterations':>12} {'utilisation':>13} {'throughput':>13}")
for name, r in (("static", static), ("continuous", cont)):
    util = 100 * r["useful_slot_steps"] / r["slot_steps"]
    tput = r["useful_slot_steps"] / r["iterations"]
    print(f"{name:<22} {r['iterations']:>12,} {util:>12.1f}% "
          f"{tput:>10.1f} tok/it")

ratio = static["iterations"] / cont["iterations"]
print(f"\nspeedup from continuous batching: {ratio:.2f}x")
print("\nNothing about the model changed. The same GPU, the same weights, the")
print("same requests -- only the moment at which the scheduler is allowed to")
print("make a decision. Chapter 22 quotes 2-4x on realistic traffic, and the")
print("size of the win here is set entirely by how heavy the tail is.")

# %% [markdown]
# ## 3. Naive KV allocation, and where the memory goes
#
# Continuous batching creates a memory problem. A sequence's cache grows as it
# generates and you do not know the final length, so the naive allocator
# reserves the maximum a sequence *could* reach.

# %%
def naive_fragmentation(requests: list, batch: int) -> dict:
    """Reserve MAX_LEN per slot; measure how much is actually occupied."""
    queue = list(requests)
    active: list = []
    reserved_steps = 0
    used_steps = 0
    iterations = 0
    while queue or active:
        while queue and len(active) < batch:
            active.append(queue.pop(0))
        iterations += 1
        reserved_steps += len(active) * MAX_LEN
        used_steps += sum(r.total_len for r in active)
        for r in active:
            r.generated += 1
        active = [r for r in active if not r.done]
    return {"reserved": reserved_steps, "used": used_steps,
            "iterations": iterations}


naive = naive_fragmentation(fresh(), BATCH)
waste = 1 - naive["used"] / naive["reserved"]

print(f"\nNaive allocator: {MAX_LEN:,} tokens reserved per slot")
print("-" * 74)
print(f"{'peak reservation':<34}: {BATCH*MAX_LEN*BYTES_PER_TOKEN/GB:>8.1f} GB")
print(f"{'mean tokens reserved per slot':<34}: {MAX_LEN:>8,}")
print(f"{'mean tokens actually held':<34}: "
      f"{naive['used']/naive['iterations']/BATCH:>8,.0f}")
print(f"{'internal fragmentation':<34}: {100*waste:>7.1f}%")
print("\nThe reservation has to cover the longest sequence a request MIGHT")
print("produce, and almost none of them do. Most of the most expensive memory")
print("on the machine is reserved for tokens that will never exist.")

# %% [markdown]
# ## 4. Paged allocation
#
# PagedAttention's idea, borrowed from virtual memory: split the cache into
# fixed-size blocks and give each sequence a **block table** mapping logical
# positions to physical blocks. Allocate a block only when the previous one
# fills. Waste is then bounded by one partial block per sequence, no matter how
# long the sequence turns out to be.

# %%
@dataclass
class BlockAllocator:
    block_size: int
    total_blocks: int
    free: list = field(default_factory=list)
    refcount: dict = field(default_factory=dict)

    def __post_init__(self):
        self.free = list(range(self.total_blocks))

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("out of KV blocks")
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def share(self, block: int) -> int:
        self.refcount[block] += 1
        return block

    def release(self, block: int) -> None:
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            del self.refcount[block]
            self.free.append(block)

    @property
    def in_use(self) -> int:
        return self.total_blocks - len(self.free)


def paged_fragmentation(requests: list, batch: int, block_size: int) -> dict:
    queue = list(requests)
    active: list = []
    tables: dict = {}
    alloc = BlockAllocator(block_size, total_blocks=10 ** 7)
    reserved_steps = used_steps = iterations = 0
    peak_blocks = 0

    def blocks_needed(n_tokens: int) -> int:
        return max(1, math.ceil(n_tokens / block_size))

    while queue or active:
        while queue and len(active) < batch:
            r = queue.pop(0)
            active.append(r)
            tables[r.rid] = [alloc.allocate() for _ in range(blocks_needed(r.total_len))]
        iterations += 1
        for r in active:
            need = blocks_needed(r.total_len)
            while len(tables[r.rid]) < need:
                tables[r.rid].append(alloc.allocate())
        reserved_steps += sum(len(tables[r.rid]) * block_size for r in active)
        used_steps += sum(r.total_len for r in active)
        peak_blocks = max(peak_blocks, alloc.in_use)
        for r in active:
            r.generated += 1
        still = []
        for r in active:
            if r.done:
                for b in tables.pop(r.rid):
                    alloc.release(b)
            else:
                still.append(r)
        active = still
    return {"reserved": reserved_steps, "used": used_steps,
            "iterations": iterations, "peak_blocks": peak_blocks}


paged = paged_fragmentation(fresh(), BATCH, block_size=16)
paged_waste = 1 - paged["used"] / paged["reserved"]

print("\nPaged allocator, 16-token blocks")
print("-" * 74)
print(f"{'allocator':<24} {'fragmentation':>15} {'peak KV memory':>17}")
print(f"{'naive (reserve max)':<24} {100*waste:>14.1f}% "
      f"{BATCH*MAX_LEN*BYTES_PER_TOKEN/GB:>15.1f} GB")
print(f"{'paged (16-token)':<24} {100*paged_waste:>14.1f}% "
      f"{paged['peak_blocks']*16*BYTES_PER_TOKEN/GB:>15.1f} GB")
print(f"\nmemory reduction: "
      f"{(BATCH*MAX_LEN)/(paged['peak_blocks']*16):.1f}x")
print("\nFragmentation falls to roughly half a block per sequence, and it stays")
print("there regardless of how long sequences turn out to be. The memory you")
print("get back is what pays for a larger batch, which is what pays for")
print("throughput -- paging is a throughput optimisation wearing a memory")
print("optimisation's clothes.")

# %% [markdown]
# ## 5. Prefix sharing
#
# Sequences sampled from the same prompt hold identical cache for the whole
# prompt. With a block table you can point them at the *same physical blocks*
# and refcount them, copying only when they diverge.
#
# Exercise 22.8 item 4: 8 requests, one shared 1,200-token prompt.

# %%
SHARED_PROMPT = 1200
N_SAMPLES = 8
GEN = 200


def shared_prefix_cost(block_size: int, share: bool) -> dict:
    alloc = BlockAllocator(block_size, total_blocks=10 ** 7)
    prompt_blocks = [alloc.allocate()
                     for _ in range(math.ceil(SHARED_PROMPT / block_size))]
    tables = []
    for _ in range(N_SAMPLES):
        if share:
            table = [alloc.share(b) for b in prompt_blocks]
        else:
            table = [alloc.allocate() for _ in prompt_blocks]
        tables.append(table)
    for table in tables:                       # each sample generates its own
        for _ in range(math.ceil(GEN / block_size)):
            table.append(alloc.allocate())
    return {"blocks": alloc.in_use,
            "bytes": alloc.in_use * block_size * BYTES_PER_TOKEN}


no_share = shared_prefix_cost(16, share=False)
with_share = shared_prefix_cost(16, share=True)

print(f"\n{N_SAMPLES} samples of one {SHARED_PROMPT:,}-token prompt, "
      f"{GEN} generated tokens each")
print("-" * 74)
print(f"{'scheme':<28} {'blocks':>10} {'KV memory':>13} {'saving':>9}")
print(f"{'copy the prompt per sample':<28} {no_share['blocks']:>10,} "
      f"{no_share['bytes']/GB:>11.2f} GB {'-':>9}")
print(f"{'share prompt blocks':<28} {with_share['blocks']:>10,} "
      f"{with_share['bytes']/GB:>11.2f} GB "
      f"{no_share['bytes']/with_share['bytes']:>8.2f}x")

print("\nThe saving grows with the number of samples and with prompt length,")
print("which is exactly the shape of two workloads that matter: a production")
print("chat service with a fixed system prompt, and a GRPO rollout sampling")
print("G completions from one problem (§15.9). A group of 32 rollouts pays for")
print("one copy of the prompt cache instead of 32.")

# %% [markdown]
# ## 6. Block size: the trade nobody mentions
#
# Small blocks waste less on the final partial block. But every block needs an
# entry in a block table, and that table is itself memory the scheduler must
# walk on every step. Exercise 22.8 item 5 asks what a 1-token block does.

# %%
POINTER_BYTES = 4

print("\nBlock size sweep")
print("-" * 74)
print(f"{'block':>8} {'fragmentation':>15} {'KV memory':>12} "
      f"{'table entries':>15} {'table bytes':>13}")
for bs in (1, 4, 16, 64, 256, 1024):
    p = paged_fragmentation(fresh(), BATCH, block_size=bs)
    frag = 1 - p["used"] / p["reserved"]
    entries = p["peak_blocks"]
    print(f"{bs:>8} {100*frag:>14.2f}% {p['peak_blocks']*bs*BYTES_PER_TOKEN/GB:>10.2f} GB "
          f"{entries:>15,} {entries*POINTER_BYTES/1e6:>11.2f} MB")

print("\nOne end of that table behaves exactly as advertised: a 1,024-token block")
print("throws away a third of the cache and rebuilds the problem paging was")
print("invented to solve.")
print("\nThe other end does not. I built this sweep expecting the block table to")
print("punish tiny blocks, and it does not -- the largest table here is a")
print("fraction of a megabyte against gigabytes of cache. On this evidence a")
print("1-token block looks strictly better than 16, and every production engine")
print("disagrees.")
print("\nSo the simulation is missing the cost, and it is worth naming rather")
print("than papering over. The real price of tiny blocks is paid in the")
print("attention KERNEL: a block table of 25,000 entries means gathering keys")
print("and values from 25,000 scattered locations, and GPU memory systems are")
print("built to move contiguous runs. Fewer, longer blocks make each gather a")
print("coalesced read. That cost is invisible to a Python simulation of")
print("bookkeeping, and it is the one that sets the default.")
print("\n16 is where fragmentation has already collapsed below 1% and the")
print("blocks are still long enough to read efficiently. The arithmetic here")
print("gets you one side of that trade; the other side needs a profiler.")

# %% [markdown]
# ## 7. Things to try
#
# **1. Make every output length identical, then rerun section 2.**
# *Common prediction:* continuous batching still wins.
# *What happens:* the advantage vanishes almost entirely. Continuous batching
# does not make decoding faster; it recovers the waste caused by *variance*. On
# uniform workloads static batching is fine, which is why benchmarks with fixed
# generation length make serving engines look interchangeable.
#
# **2. Set `BATCH = 4` and then `BATCH = 128` in section 2.**
# *Common prediction:* bigger batch, bigger continuous-batching win.
# *What happens:* the win grows and then flattens, because with enough slots
# the tail requests no longer block anyone. The interesting consequence is the
# other direction: paging is what lets you afford the larger batch, so sections
# 3–4 are the precondition for section 2's speedup.
#
# **3. Give the naive allocator a `MAX_LEN` of 2,048 instead of 8,192.**
# *Common prediction:* fragmentation improves fourfold.
# *What happens:* it does — and any request wanting more than 2,048 tokens now
# fails. That is the actual pre-paging trade: fragmentation against a hard cap
# on what you can serve.
#
# **4. Share a prompt of 50 tokens instead of 1,200 in section 5.**
# *Common prediction:* proportionally smaller saving.
# *What happens:* nearly no saving, because the shared portion is smaller than
# the generated portion. Prefix sharing pays exactly in proportion to how much
# of the cache is common, which is why system prompts and RL rollouts are its
# best cases and short chat turns are not.
#
# **5. Add preemption: when the allocator runs out of blocks, evict the
# newest sequence and requeue it.**
# This is the piece the lab leaves out, and it is where real schedulers get
# complicated — evicting means either recomputing the prefill or swapping the
# blocks to host memory, and choosing between those is a live design question.

# %% [markdown]
# ## What to take away
#
# 1. **Continuous batching is a scheduling change, not a kernel change**, and
#    it is the largest single win in serving. Its size is set by the variance
#    of your output lengths.
# 2. **The naive allocator reserves for the sequence a request might become.**
#    Most of the most expensive memory on the machine ends up holding tokens
#    that never exist.
# 3. **Paging bounds waste at one partial block per sequence**, independent of
#    sequence length — and the memory it returns is what funds the bigger batch
#    that makes continuous batching pay.
# 4. **Prefix sharing is refcounting.** It pays in proportion to the shared
#    fraction, which makes fixed system prompts and $G$-sample RL rollouts its
#    ideal workloads.
# 5. **Block size is a trade this lab can only half measure.** Large blocks
#    reintroduce fragmentation, which the arithmetic shows plainly. Small
#    blocks are punished by *kernel* efficiency — thousands of scattered
#    gathers instead of coalesced reads — which a bookkeeping simulation cannot
#    see at all. When a lab can only measure one side of a trade-off, the
#    honest move is to say which side, rather than to conclude from the half
#    you happen to have.
