# %% [markdown]
# # Lab 7 — Collective communication, and why topology decides your MFU
#
# Companion to [Chapter 7](../book/part-2-pretraining/07-cluster-reality.md).
#
# The same model, the same code, and the same 1,024 GPUs can run at 40% MFU or
# 25% depending on which nodes the scheduler handed you. This lab builds the
# cost model that explains why.
#
# You will:
#
# 1. Model ring all-reduce and discover per-rank traffic is nearly independent
#    of rank count — but latency is not.
# 2. Compare flat against hierarchical all-reduce and find the crossover.
# 3. Model all-to-all and see why it degrades under oversubscription in a way
#    all-reduce does not.
# 4. Measure the straggler tax: one slow rank costs the whole cluster.
# 5. Model bucketing and find the message size where you stop being
#    latency-bound.
#
# Pure arithmetic. No GPU, no PyTorch, runs instantly.
#
# **Predict before you run.** Section 1 asks how per-rank traffic scales from 8
# to 1,024 ranks. Write it down.

# %%
from __future__ import annotations

import math
import os
from dataclasses import dataclass

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"

GB = 1e9

# %% [markdown]
# ## 1. A link model
#
# Three numbers describe any interconnect for our purposes: **bandwidth** (how
# fast, once flowing), **latency** (fixed cost per message), and
# **oversubscription** (how much of the aggregate bandwidth actually exists at
# a given tier).
#
# The constants below are **H100-class** figures (NVLink 4, NDR InfiniBand).
# The book's anchor cluster is different: DeepSeek-V3 trained on H800s, whose
# cut-down NVLink gives about 3.2× InfiniBand, not 11×
# ([fact sheet](../book/appendix/fact-sheets/deepseek-v3.md#cluster-and-interconnect)).


# %%
@dataclass
class Link:
    name: str
    gbps: float             # achieved unidirectional bandwidth, GB/s per rank
    latency_us: float       # per-hop latency
    oversubscription: float = 1.0     # 2.0 means half the bandwidth is real

    @property
    def effective_gbps(self) -> float:
        return self.gbps / self.oversubscription

    def time_s(self, bytes_moved: float, hops: int = 1) -> float:
        transfer = bytes_moved / (self.effective_gbps * GB)
        return transfer + hops * self.latency_us * 1e-6


NVLINK = Link("NVLink (intra-node)", 450.0, 2.0)
IB_400 = Link("InfiniBand 400G", 40.0, 5.0)
IB_400_OS2 = Link("InfiniBand 400G, 2:1 oversubscribed", 40.0, 5.0, oversubscription=2.0)
ETH_100 = Link("Ethernet 100G (RoCE)", 10.0, 15.0)

LINKS = [NVLINK, IB_400, IB_400_OS2, ETH_100]

print("Interconnect model")
print("-" * 74)
print(f"{'link':<38} {'GB/s':>8} {'eff GB/s':>10} {'latency':>10}")
for link in LINKS:
    print(
        f"{link.name:<38} {link.gbps:>8.0f} {link.effective_gbps:>10.1f} "
        f"{link.latency_us:>8.0f} us"
    )
print(f"\nNVLink / InfiniBand bandwidth ratio (H100-class links): {NVLINK.gbps / IB_400.gbps:.0f}x")
print("That ratio is the key number in Chapter 7's H100 discussion. It is an H100 figure:")
print("on the book's anchor cluster, DeepSeek-V3's H800s, the report gives 160 vs 50 GB/s,")
print("about 3.2x (V3 section 3.2.2; see the DeepSeek-V3 fact sheet).")

# %% [markdown]
# ## 2. Ring all-reduce
#
# A ring all-reduce is a reduce-scatter followed by an all-gather. Each of the
# two phases moves $(N-1)/N$ of the payload per rank, in $N-1$ steps.
#
# So per-rank traffic is $2(N-1)/N \times$ payload — which **approaches 2× and
# never exceeds it**, no matter how many ranks. Latency, however, is $2(N-1)$
# sequential hops and grows without bound.


# %%
def ring_allreduce(payload_bytes: float, n_ranks: int, link: Link) -> dict:
    """Ring all-reduce: reduce-scatter then all-gather."""
    if n_ranks <= 1:
        return {"bytes_per_rank": 0.0, "hops": 0, "time_s": 0.0}
    frac = 2 * (n_ranks - 1) / n_ranks
    bytes_per_rank = payload_bytes * frac
    hops = 2 * (n_ranks - 1)
    # Bandwidth term uses total bytes; latency term uses hop count.
    transfer = bytes_per_rank / (link.effective_gbps * GB)
    latency = hops * link.latency_us * 1e-6
    return {
        "bytes_per_rank": bytes_per_rank,
        "hops": hops,
        "transfer_s": transfer,
        "latency_s": latency,
        "time_s": transfer + latency,
    }


PAYLOAD = 70e9 * 2          # 70B model, BF16 gradients = 140 GB

print("\nRing all-reduce of 140 GB (70B model gradients, BF16)")
print("-" * 74)
print(f"{'ranks':>7} {'bytes/rank':>13} {'multiplier':>11} {'hops':>7} {'transfer':>10} {'latency':>10}")
for n in (2, 8, 64, 256, 1024):
    r = ring_allreduce(PAYLOAD, n, IB_400)
    print(
        f"{n:>7} {r['bytes_per_rank']/GB:>11.1f} GB {r['bytes_per_rank']/PAYLOAD:>10.2f}x "
        f"{r['hops']:>7} {r['transfer_s']:>9.2f}s {r['latency_s']*1000:>8.1f}ms"
    )

print("\nPer-rank traffic is flat at ~2x. Latency grows linearly and is")
print("negligible for a 140 GB payload -- but hold that thought for section 5.")

# %% [markdown]
# ## 3. Flat versus hierarchical all-reduce
#
# A flat ring across 1,024 GPUs sends most of its traffic over the slow
# inter-node link. A hierarchical all-reduce does:
#
# 1. **Reduce-scatter within each node** over NVLink (fast).
# 2. **All-reduce the per-node partials across nodes** — but only $1/G$ of the
#    payload, where $G$ is GPUs per node.
# 3. **All-gather within each node** over NVLink.
#
# Inter-node volume drops by the intra-node group size.


# %%
def hierarchical_allreduce(
    payload_bytes: float, n_ranks: int, gpus_per_node: int,
    intra: Link, inter: Link,
) -> dict:
    n_nodes = max(1, n_ranks // gpus_per_node)

    # Phase 1: reduce-scatter within the node.
    rs = ring_allreduce(payload_bytes, gpus_per_node, intra)
    phase1 = rs["time_s"] / 2 if gpus_per_node > 1 else 0.0   # scatter half only

    # Phase 2: all-reduce 1/G of the payload across nodes.
    shard = payload_bytes / gpus_per_node
    phase2 = ring_allreduce(shard, n_nodes, inter)["time_s"] if n_nodes > 1 else 0.0

    # Phase 3: all-gather within the node.
    phase3 = phase1

    return {
        "phase1_s": phase1,
        "phase2_s": phase2,
        "phase3_s": phase3,
        "time_s": phase1 + phase2 + phase3,
        "inter_node_bytes": shard * 2 * (n_nodes - 1) / max(n_nodes, 1),
    }


print("\nFlat ring vs hierarchical, 140 GB payload, 8 GPUs/node")
print("-" * 74)
print(f"{'ranks':>7} {'flat':>10} {'hierarchical':>14} {'speedup':>10} {'inter-node bytes':>19}")
for n in (16, 64, 256, 1024):
    flat = ring_allreduce(PAYLOAD, n, IB_400)
    hier = hierarchical_allreduce(PAYLOAD, n, 8, NVLINK, IB_400)
    print(
        f"{n:>7} {flat['time_s']:>9.2f}s {hier['time_s']:>13.2f}s "
        f"{flat['time_s']/hier['time_s']:>9.2f}x {hier['inter_node_bytes']/GB:>17.1f} GB"
    )

print("\nHierarchical wins at every node count above 1, because the flat ring")
print("moves ~2x payload over the SLOW link while hierarchical moves 2/8 of it.")
print("This is why every production NCCL config uses hierarchical algorithms")
print("and the flat ring is mostly a pedagogical object.")

# %% [markdown]
# ## 4. All-to-all, and why oversubscription hurts it more
#
# An all-reduce ring can be **ordered** so that most hops stay under a leaf
# switch. An all-to-all cannot: every rank sends to every other rank, so a fixed
# fraction of traffic *must* cross the spine.


# %%
def all_to_all(bytes_per_rank_out: float, n_ranks: int, link: Link) -> dict:
    """Each rank sends bytes_per_rank_out total, split across all peers."""
    if n_ranks <= 1:
        return {"time_s": 0.0, "message_bytes": 0.0}
    # (N-1)/N of the payload leaves this rank.
    leaving = bytes_per_rank_out * (n_ranks - 1) / n_ranks
    message = bytes_per_rank_out / n_ranks           # one message per peer
    # All-to-all is a scatter of many small messages: latency is paid per peer.
    transfer = leaving / (link.effective_gbps * GB)
    latency = (n_ranks - 1) * link.latency_us * 1e-6
    return {"time_s": transfer + latency, "message_bytes": message,
            "transfer_s": transfer, "latency_s": latency}


# MoE dispatch: tokens_per_rank * top_k * d_model * 2 bytes, dispatch + combine.
MOE_BYTES = 2 * (65536 * 8 * 7168 * 2)      # ~15 GB per rank per layer

print("\nSame collective, with and without 2:1 spine oversubscription")
print("-" * 74)
print(f"{'collective':<22} {'clean':>10} {'2:1 oversub':>13} {'degradation':>13}")

ar_clean = ring_allreduce(PAYLOAD, 256, IB_400)["time_s"]
ar_os = ring_allreduce(PAYLOAD, 256, IB_400_OS2)["time_s"]
a2a_clean = all_to_all(MOE_BYTES, 64, IB_400)["time_s"]
a2a_os = all_to_all(MOE_BYTES, 64, IB_400_OS2)["time_s"]

print(f"{'all-reduce (256 rk)':<22} {ar_clean:>9.2f}s {ar_os:>12.2f}s {ar_os/ar_clean:>12.2f}x")
print(f"{'all-to-all (64 rk)':<22} {a2a_clean:>9.2f}s {a2a_os:>12.2f}s {a2a_os/a2a_clean:>12.2f}x")

print("\nBoth degrade ~2x in this model, which is the honest arithmetic. The")
print("real difference is that a TOPOLOGY-AWARE ring can be ordered to keep")
print("most hops under a leaf switch, recovering much of the loss; an")
print("all-to-all is all-pairs and has no such ordering. Chapter 7 section 7.2.")

print("\nAll-to-all placement: intra-node vs spanning nodes")
print("-" * 74)
print(f"{'EP degree':>10} {'link':<34} {'time':>10}")
for ep, link in ((8, NVLINK), (8, IB_400), (64, IB_400)):
    t = all_to_all(MOE_BYTES, ep, link)["time_s"]
    print(f"{ep:>10} {link.name:<34} {t:>9.2f}s")
print("\nEP=8 on NVLink vs EP=8 on InfiniBand is the default placement decision")
print("Chapter 6 section 6.11 describes, quantified. DeepSeek-V3 ran the EP=64 row")
print("anyway, and paid for it with overlap and a 4-node routing limit (section 6.10).")

# %% [markdown]
# ## 5. Latency, bucketing, and small messages
#
# The latency term was negligible for a 140 GB payload. It is not negligible
# for the hundreds of small per-parameter gradient tensors a real model
# produces, which is why gradient **bucketing** exists.


# %%
def bucketed_allreduce(
    total_bytes: float, n_tensors: int, bucket_bytes: float,
    n_ranks: int, link: Link,
) -> dict:
    """Fuse tensors into buckets, then all-reduce each bucket."""
    n_buckets = max(1, math.ceil(total_bytes / bucket_bytes))
    n_buckets = min(n_buckets, n_tensors)          # cannot have more than tensors
    per_bucket = total_bytes / n_buckets
    r = ring_allreduce(per_bucket, n_ranks, link)
    return {
        "n_buckets": n_buckets,
        "time_s": n_buckets * r["time_s"],
        "transfer_s": n_buckets * r["transfer_s"],
        "latency_s": n_buckets * r["latency_s"],
    }


N_TENSORS = 1200                     # a 70B model has ~1200 parameter tensors

print("\nBucket size sweep: 140 GB across 1,200 tensors, 256 ranks")
print("-" * 74)
print(f"{'bucket':>10} {'buckets':>9} {'transfer':>11} {'latency':>11} {'total':>10} {'latency %':>11}")
for mb in (0.1, 1, 10, 25, 100, 500):
    r = bucketed_allreduce(PAYLOAD, N_TENSORS, mb * 1e6, 256, IB_400)
    frac = 100 * r["latency_s"] / r["time_s"]
    print(
        f"{mb:>8.1f}MB {r['n_buckets']:>9} {r['transfer_s']:>10.2f}s "
        f"{r['latency_s']:>10.2f}s {r['time_s']:>9.2f}s {frac:>10.1f}%"
    )

print("\nBelow ~10 MB the latency term dominates: you are paying 2*(N-1) hops")
print("per bucket and the transfer is trivial. Above it, you are bandwidth-")
print("bound and bucketing further buys nothing. That knee is why NCCL's")
print("default bucket size is in the tens of megabytes.")

# %% [markdown]
# ## 6. The straggler tax
#
# Every synchronous collective completes at the pace of its slowest
# participant. This is the single most counterintuitive fact in Chapter 7.
#
# **Predict:** on 256 ranks, one rank running 20% slow — what is the aggregate
# throughput loss?


# %%
def straggler_impact(n_ranks: int, n_slow: int, slowdown: float) -> dict:
    """Aggregate throughput under a synchronous collective with stragglers."""
    # Everyone waits for the slowest, so the step time is set by it.
    step_time_multiplier = slowdown
    # Naive intuition: the loss is proportional to the fraction that is slow.
    naive = 1 - (n_slow / n_ranks) * (1 - 1 / slowdown)
    actual = 1 / step_time_multiplier
    return {"naive_throughput": naive, "actual_throughput": actual}


print("\nStraggler tax on a synchronous collective")
print("-" * 74)
print(f"{'ranks':>7} {'slow':>6} {'slowdown':>10} {'naive guess':>13} {'actual':>10}")
for n_ranks, n_slow, slow in ((256, 1, 1.20), (256, 1, 2.00), (1024, 1, 1.20), (1024, 8, 1.20)):
    r = straggler_impact(n_ranks, n_slow, slow)
    print(
        f"{n_ranks:>7} {n_slow:>6} {slow:>9.2f}x {r['naive_throughput']*100:>12.1f}% "
        f"{r['actual_throughput']*100:>9.1f}%"
    )

print("\nOne rank at 80% speed costs the ENTIRE cluster 17% of its throughput,")
print("whether the cluster is 256 GPUs or 1,024. The naive 'it is only 1/256")
print("of the machine' intuition is wrong by two orders of magnitude.")
print("\nThis is why Chapter 7 section 7.4's first diagnostic for a cluster-wide")
print("slowdown is a PER-RANK step-time histogram: a 15% aggregate drop is far")
print("more likely to be one rank at 85% than all ranks at 85%.")

# %% [markdown]
# ## 7. Putting it together: does the communication hide?
#
# The DP all-reduce overlaps with the backward pass. TP and EP collectives sit
# on the critical path. Whether a configuration works comes down to whether the
# overlappable part fits inside the compute.

# %%
print("\nCan the DP all-reduce hide behind the backward pass?")
print("-" * 74)
print(f"{'scenario':<40} {'compute':>9} {'comm':>9} {'exposed':>9}")
scenarios = [
    ("1024 ranks, InfiniBand, 8s compute", 8.0, ring_allreduce(PAYLOAD, 1024, IB_400)["time_s"]),
    ("1024 ranks, IB 2:1 oversub, 8s", 8.0, ring_allreduce(PAYLOAD, 1024, IB_400_OS2)["time_s"]),
    ("1024 ranks, 100G Ethernet, 8s", 8.0, ring_allreduce(PAYLOAD, 1024, ETH_100)["time_s"]),
    ("hierarchical, InfiniBand, 8s", 8.0, hierarchical_allreduce(PAYLOAD, 1024, 8, NVLINK, IB_400)["time_s"]),
]
step_s = {}
for label, compute, comm in scenarios:
    exposed = max(0.0, comm - compute)
    step_s[label] = compute + exposed
    print(f"{label:<40} {compute:>8.1f}s {comm:>8.2f}s {exposed:>8.2f}s")

# The headline multiplier is computed from the rows above, never typed in.
IB_ROW, ETH_ROW = scenarios[0][0], scenarios[2][0]
eth_vs_ib = step_s[ETH_ROW] / step_s[IB_ROW]
headline = f"{eth_vs_ib:.1f}x"
assert float(headline[:-1]) == round(step_s[ETH_ROW] / step_s[IB_ROW], 1)

print("\nThe Ethernet row is the one to sit with: the same model, the same code,")
print("the same GPU count, and communication no longer hides. Step time goes from")
print(f"{step_s[IB_ROW]:.1f}s to {step_s[ETH_ROW]:.1f}s: a {headline} "
      "throughput difference decided entirely by the interconnect.")

# %% [markdown]
# ## 8. Things to try
#
# **1. Set `oversubscription=4.0` and rerun section 4.**
# *Common prediction:* everything degrades 4x.
# *What happens:* it does, in this model — and in reality the all-reduce
# recovers much of it through topology-aware ring ordering while the all-to-all
# cannot. This model deliberately does not simulate ring ordering, which is a
# limitation worth knowing: it *understates* the gap between the two.
#
# **2. Make `gpus_per_node=4` in section 3.**
# *Common prediction:* hierarchical still wins by the same margin.
# *What happens:* the win shrinks, because the inter-node volume reduction is
# by the intra-node group size. With 4 GPUs/node you divide by 4 rather than 8.
# Node size is a topology decision that directly sets your collective cost.
#
# **3. Set the bucket size to 1 KB.**
# *Common prediction:* somewhat slower.
# *What happens:* catastrophically slower — you pay `2*(N-1)` hops of latency
# for every one of 1,200 buckets. Latency becomes ~100% of the time. This is
# what an unbucketed gradient all-reduce actually looks like.
#
# **4. Model 8 slow ranks out of 1,024 at 1.2x in section 6.**
# *Common prediction:* eight times worse than one slow rank.
# *What happens:* **identical**. The step time is set by the slowest, and eight
# ranks at the same slowdown are no slower than one. Straggler cost depends on
# the *worst* rank, not on how many are bad — which is why finding and evicting
# the single worst node is worth more than improving the average.
#
# **5. Compute the payload for a 671B MoE and rerun section 2.**
# The all-reduce is over sharded gradients, so it is smaller than you expect
# under EP. Work out what actually gets all-reduced before you compute it —
# that question is Exercise 6.4 part 5.

# %% [markdown]
# ## What to take away
#
# 1. **Ring all-reduce per-rank traffic approaches 2× payload and stops there.**
#    It does not grow with rank count. Latency does.
# 2. **Hierarchical all-reduce wins at any node count above 1**, because it
#    moves only $1/G$ of the payload over the slow link.
# 3. **All-to-all has no good ordering.** Every rank talks to every rank, so a
#    fixed fraction crosses the spine no matter what — which is why EP defaults
#    to living inside a node, and why running it across nodes (as DeepSeek-V3
#    did) takes overlap and routing limits to pay for.
# 4. **Below ~10 MB you are latency-bound.** Bucketing exists to get you above
#    that knee.
# 5. **One rank at 80% speed costs the whole cluster 17%.** Synchronous
#    collectives complete at the pace of the slowest participant, and the cost
#    does not shrink with cluster size.
# 6. **The interconnect decides the parallelism strategy more than the model
#    does.** The same configuration on InfiniBand and on Ethernet are different
#    runs.
