# %% [markdown]
# # Lab 21 — Checkpoint and resume: what "state" actually means
#
# Companion to [Chapter 21](../book/part-4-infra/21-checkpointing-resumption.md).
#
# A two-week training run will be interrupted. The only question is whether
# resuming puts you back exactly where you were, or somewhere that merely looks
# similar. The second one is far more common, and much harder to notice: the
# loss curve after a bad resume looks *fine*. It is slightly worse, forever,
# and nothing in your dashboard says so.
#
# This lab makes the difference measurable. We define resume correctness as
# **bit-exactness**: steps 51–100 of a resumed run must produce byte-identical
# losses to steps 51–100 of a run that was never interrupted. Anything less is
# a silent bug.
#
# You will:
#
# 1. Build a training loop and enumerate everything that counts as state.
# 2. Checkpoint four ways — weights, then optimiser, then RNG, then data order
#    — and find exactly which step each one diverges on.
# 3. Reshard a checkpoint from 4 shards to 2 and watch a physical layout break
#    where a logical one survives.
# 4. Kill a process mid-write and see what the loader does, with and without an
#    atomic rename.
#
# Runs in seconds on CPU. Writes only to a temporary directory, which it
# removes at the end.
#
# **Predict before you run.** A checkpoint saves model weights and optimiser
# state — the two things everyone remembers. On which step does the resumed run
# first differ from the uninterrupted one? Write down a step number.

# %%
from __future__ import annotations

import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# %%
import os
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
SEED = 0
WORKDIR = tempfile.mkdtemp(prefix="fle_lab21_")
print(f"smoke_test={SMOKE}  torch={torch.__version__}")
print(f"checkpoints in {WORKDIR}")

TOTAL_STEPS = 40 if SMOKE else 100
RESUME_AT = 20 if SMOKE else 50

# %% [markdown]
# ## 1. Everything that is "state"
#
# The list is longer than people expect, and every item on it is something a
# real run has lost at least once:
#
# | State | Lost if you forget it |
# |---|---|
# | Model parameters | Obvious; nobody forgets this |
# | Optimiser moments | Adam's $m$ and $v$ — the run re-warms them from zero |
# | LR schedule position | You restart the schedule and re-heat a converged model |
# | Global RNG | Different dropout masks, different augmentation |
# | Data-loader position | You re-train on data you have already seen |
# | Step counter | Everything above silently disagrees about "when" it is |
#
# Our toy has all six. The model uses dropout, so it consumes global RNG; the
# loader draws batch indices from its **own** generator, so the two can be
# ablated independently.

# %%
class Net(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.drop = nn.Dropout(0.2)          # consumes global RNG every step
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class Loader:
    """Shuffled sampler with explicit, checkpointable position."""

    def __init__(self, n_samples: int, batch: int, seed: int = SEED):
        self.n, self.batch = n_samples, batch
        self.gen = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(self.n, generator=self.gen)
        self.pos = 0
        self.epoch = 0

    def next_ids(self) -> torch.Tensor:
        if self.pos + self.batch > self.n:
            self.epoch += 1
            self.order = torch.randperm(self.n, generator=self.gen)
            self.pos = 0
        ids = self.order[self.pos:self.pos + self.batch]
        self.pos += self.batch
        return ids

    def state_dict(self) -> dict:
        return {"pos": self.pos, "epoch": self.epoch,
                "order": self.order.clone(), "gen": self.gen.get_state()}

    def load_state_dict(self, sd: dict) -> None:
        self.pos, self.epoch = sd["pos"], sd["epoch"]
        self.order = sd["order"].clone()
        self.gen.set_state(sd["gen"])


torch.manual_seed(SEED)
N_SAMPLES, D = 512, 64
DATA_X = torch.randn(N_SAMPLES, D)
DATA_Y = torch.randint(0, 8, (N_SAMPLES,))


def build(seed: int = SEED) -> tuple:
    torch.manual_seed(seed)
    model = Net(D)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_STEPS)
    loader = Loader(N_SAMPLES, batch=32, seed=seed)
    return model, opt, sched, loader


def train_steps(model, opt, sched, loader, n_steps: int) -> list:
    """Run n_steps and return (loss, first_sample_id) for each."""
    trace = []
    model.train()
    for _ in range(n_steps):
        ids = loader.next_ids()
        logits = model(DATA_X[ids])
        loss = F.cross_entropy(logits, DATA_Y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        trace.append((float(loss), int(ids[0])))
    return trace


# %% [markdown]
# ## 2. The reference run
#
# One uninterrupted run of every step. Every experiment below is compared
# against this, exactly.

# %%
model, opt, sched, loader = build()
REFERENCE = train_steps(model, opt, sched, loader, TOTAL_STEPS)

print(f"\nReference: {TOTAL_STEPS} uninterrupted steps")
print("-" * 74)
print(f"  first loss {REFERENCE[0][0]:.6f}   last loss {REFERENCE[-1][0]:.6f}")
print(f"  sample IDs at steps 1-5: {[t[1] for t in REFERENCE[:5]]}")

# %% [markdown]
# ## 3. Four checkpoint recipes
#
# Each recipe saves at step `RESUME_AT`, builds a fresh process-equivalent, and
# restores only what the recipe includes. We then compare the remaining steps
# to the reference and report **the first step that differs at all**.

# %%
RECIPES = [
    ("weights only", {"model"}),
    ("+ optimiser & schedule", {"model", "opt", "sched"}),
    ("+ global RNG", {"model", "opt", "sched", "rng"}),
    ("+ data-loader position", {"model", "opt", "sched", "rng", "loader"}),
]


def save_checkpoint(path: str, model, opt, sched, loader, step: int) -> None:
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "rng": torch.get_rng_state(),
        "loader": loader.state_dict(),
        "step": step,
    }, path)


def resume_with(parts: set, path: str) -> list:
    """Restore only `parts`, then run the remaining steps."""
    ckpt = torch.load(path, weights_only=False)
    # A fresh start, as a restarted process would be -- note the DIFFERENT seed,
    # because a restarted job does not magically re-roll the same dice.
    model, opt, sched, loader = build(seed=SEED + 1)
    if "model" in parts:
        model.load_state_dict(ckpt["model"])
    if "opt" in parts:
        opt.load_state_dict(ckpt["opt"])
    if "sched" in parts:
        sched.load_state_dict(ckpt["sched"])
    if "rng" in parts:
        torch.set_rng_state(ckpt["rng"])
    if "loader" in parts:
        loader.load_state_dict(ckpt["loader"])
    return train_steps(model, opt, sched, loader, TOTAL_STEPS - ckpt["step"])


# Produce the checkpoint by replaying the reference to RESUME_AT.
model, opt, sched, loader = build()
train_steps(model, opt, sched, loader, RESUME_AT)
CKPT = os.path.join(WORKDIR, "step.pt")
save_checkpoint(CKPT, model, opt, sched, loader, RESUME_AT)

tail_ref = REFERENCE[RESUME_AT:]

print(f"\nResume at step {RESUME_AT}, compare steps "
      f"{RESUME_AT+1}-{TOTAL_STEPS} against the reference")
print("-" * 74)
print(f"{'checkpoint contains':<26} {'1st differing step':>19} {'max loss delta':>16} "
      f"{'IDs match':>10}")
for label, parts in RECIPES:
    tail = resume_with(parts, CKPT)
    first_diff = None
    max_delta = 0.0
    ids_match = True
    for i, (got, want) in enumerate(zip(tail, tail_ref), start=RESUME_AT + 1):
        if got[1] != want[1]:
            ids_match = False
        delta = abs(got[0] - want[0])
        max_delta = max(max_delta, delta)
        if first_diff is None and (delta > 0 or got[1] != want[1]):
            first_diff = i
    shown = "none (exact)" if first_diff is None else str(first_diff)
    print(f"{label:<26} {shown:>19} {max_delta:>16.2e} {str(ids_match):>10}")

_ref_scale = sum(t[0] for t in tail_ref) / len(tail_ref)
print(f"\nEvery incomplete recipe diverges on the FIRST step after the resume --")
print("not gradually, immediately.")
print(f"\nThe deltas are a few percent of a typical loss ({_ref_scale:.3f} here), so")
print("nothing crashes and nothing spikes. That combination is the problem: the")
print("run takes a slightly wrong step, then another, and the curve looks")
print("entirely normal while being permanently worse than the run you believed")
print("you were continuing. There is no alert for this.")
print("\nOnly the complete recipe is bit-exact. That is the acceptance test:")
print("not 'the loss looks reasonable' but 'the losses are identical'.")

# %% [markdown]
# ## 4. Resharding: logical layout versus physical
#
# Checkpoints are written in shards, one per rank. Restart on a different
# number of GPUs — because that is what the scheduler gave you — and the
# checkpoint has to be re-split.
#
# Whether that works depends entirely on what the format stored. A **physical**
# layout records "rank 2's slice of the flattened buffer". A **logical** layout
# records "the tensor named `fc1.weight`, and which part of it this is". Only
# one of those survives a change in world size.

# %%
def shard_physical(sd: dict, world: int) -> list:
    """Flatten every parameter into one buffer; rank r writes chunk r.

    This is the naive format: each shard file is an anonymous run of floats.
    Nothing in it records which tensor those floats belong to, or how many
    shards there were -- the loader is expected to know from the world size.
    """
    flat = torch.cat([v.flatten() for v in sd.values()])
    return list(torch.chunk(flat, world))


def load_physical(shard_files: list, template: dict, new_world: int) -> dict:
    """Each of `new_world` ranks reads its own shard and assumes it holds
    1/new_world of the buffer -- because that is all the format tells it."""
    picked = shard_files[:new_world]
    flat = torch.cat(picked)
    expected = sum(v.numel() for v in template.values())
    if flat.numel() != expected:
        raise ValueError(
            f"expected {expected:,} elements, got {flat.numel():,}")
    out, off = {}, 0
    for k, v in template.items():
        out[k] = flat[off:off + v.numel()].view_as(v)
        off += v.numel()
    return out


def shard_logical(sd: dict, world: int) -> list:
    """Each shard is a dict: tensor name -> (slice index, total slices, data)."""
    shards = [dict() for _ in range(world)]
    for k, v in sd.items():
        for i, piece in enumerate(torch.chunk(v.flatten(), world)):
            shards[i][k] = (i, world, piece.clone())
    return shards


def load_logical(shards: list, template: dict, new_world: int) -> dict:
    """Reassemble each tensor from its named slices, then re-split for the new
    world size. Works for ANY new_world, because names and order are stored."""
    full = {}
    for k, v in template.items():
        pieces = [sh[k][2] for sh in sorted(shards, key=lambda sh: sh[k][0])]
        full[k] = torch.cat(pieces).view_as(v)
    # Re-split and reassemble, i.e. what the new ranks would each own.
    out = {}
    for k, v in full.items():
        out[k] = torch.cat(list(torch.chunk(v.flatten(), new_world))).view_as(v)
    return out


reference_sd = {k: v.clone() for k, v in model.state_dict().items()}
n_elems = sum(v.numel() for v in reference_sd.values())

print(f"\nCheckpoint saved from 4 ranks ({n_elems:,} parameters), restarted on 2")
print("-" * 74)

phys4 = shard_physical(reference_sd, 4)
log4 = shard_logical(reference_sd, 4)

for label, world in (("same world size (4)", 4), ("resharded to 2", 2)):
    try:
        got = load_physical(phys4, reference_sd, world)
        ok = torch.allclose(got["fc1.weight"], reference_sd["fc1.weight"])
        print(f"{'physical, ' + label:<40}: {'exact' if ok else 'WRONG VALUES'}")
    except Exception as exc:
        print(f"{'physical, ' + label:<40}: FAILS -- {exc}")

for label, world in (("same world size (4)", 4), ("resharded to 2", 2)):
    got = load_logical(log4, reference_sd, world)
    ok = all(torch.equal(got[k], reference_sd[k]) for k in reference_sd)
    print(f"{'logical, ' + label:<40}: {'exact' if ok else 'WRONG VALUES'}")

print("\nThe physical format cannot reshard, and the reason is that it never")
print("stored enough to try. A shard is an anonymous run of floats; the mapping")
print("from bytes to tensors lives in the loader, derived from the world size.")
print("Change the world size and the derivation is wrong -- here it is caught by")
print("a size check, and in formats without one it silently loads shifted data,")
print("which is the same shape as your model and complete nonsense.")
print("\nThe logical format stores the tensor NAME and the slice index with")
print("every piece, so it can always be reassembled and re-split to any world")
print("size. That is why production formats -- PyTorch Distributed Checkpoint,")
print("safetensors index files, Megatron's dist-ckpt -- all store names and")
print("shapes rather than offsets.")
print("\nThis matters because resharding is not exotic. It happens whenever the")
print("scheduler gives you a different number of nodes than you had yesterday.")

# %% [markdown]
# ## 5. The torn checkpoint
#
# Writes are not atomic. If the process dies partway through `torch.save`, the
# file exists, has a plausible size, and is garbage. The next start reads it.

# %%
GOOD = os.path.join(WORKDIR, "good.pt")
save_checkpoint(GOOD, model, opt, sched, loader, RESUME_AT)
size = os.path.getsize(GOOD)

TORN = os.path.join(WORKDIR, "torn.pt")
with open(GOOD, "rb") as src, open(TORN, "wb") as dst:
    dst.write(src.read(int(size * 0.6)))       # killed 60% of the way through

print(f"\nCheckpoint is {size:,} bytes; the torn one is "
      f"{os.path.getsize(TORN):,} bytes")
print("-" * 74)
for label, path in (("complete checkpoint", GOOD), ("torn checkpoint", TORN)):
    try:
        torch.load(path, weights_only=False)
        print(f"{label:<26}: loads")
    except Exception as exc:
        print(f"{label:<26}: FAILS -- {type(exc).__name__}")

# The fix: write to a temporary name, fsync, then rename. POSIX rename within a
# filesystem is atomic, so the destination is either the old file or the new
# one and never a half-written mixture.
FINAL = os.path.join(WORKDIR, "atomic.pt")
tmp = FINAL + ".tmp"
save_checkpoint(tmp, model, opt, sched, loader, RESUME_AT)
with open(tmp, "rb") as fh:
    os.fsync(fh.fileno())
os.replace(tmp, FINAL)
print(f"{'atomic write + rename':<26}: loads, and {FINAL.split('/')[-1]} never")
print(f"{'':<26}  existed in a partial state")

print("\nThis lab's torn file fails loudly because torch.save writes a zip with")
print("a trailing directory. That is luck, not design. A raw tensor dump")
print("truncated at 60% loads happily and gives you a model with a corrupted")
print("final layer -- which is the version that reaches production.")
print("\nWrite to a temporary name, fsync, rename. Keep the previous checkpoint")
print("until the new one is verified. Chapter 21 section 21.4.")

# %%
shutil.rmtree(WORKDIR, ignore_errors=True)
print(f"\ncleaned up {WORKDIR}")

# %% [markdown]
# ## 6. Things to try
#
# **1. Remove `sched` from the third recipe but keep `opt`.**
# *Common prediction:* a small learning-rate difference, easily absorbed.
# *What happens:* the schedule restarts from step 0, so a cosine-decayed run
# jumps back to peak learning rate. On a real run this is the classic "loss
# spiked after a restart and then never recovered" incident.
#
# **2. Make the model deterministic by setting `Dropout(0.0)`, then rerun
# section 3.**
# *Common prediction:* the RNG recipe becomes unnecessary.
# *What happens:* it does, here — and that is a trap worth feeling. Removing
# the only RNG consumer makes an incomplete checkpoint look correct. Add data
# augmentation, dropout, or stochastic depth back and the bug returns, having
# passed all your tests.
#
# **3. Add a parameter to `Net` so the flattened size is not divisible by 4,
# then rerun section 4.**
# *Common prediction:* padding handles it.
# *What happens:* the physical reshard produces misaligned tensors and the
# `allclose` check fails, while the logical one still passes. This is the
# failure mode the section warns about, made concrete.
#
# **4. Truncate the checkpoint at 99.9% instead of 60%.**
# *Common prediction:* it fails the same way.
# *What happens:* it may still fail on the zip directory — but try truncating a
# `state_dict` saved with `torch.save(..., _use_new_zipfile_serialization=False)`
# and see how much closer to silent the failure gets.
#
# **5. Resume twice: checkpoint at 25, resume, checkpoint at 50, resume.**
# *Common prediction:* errors compound.
# *What happens:* with the complete recipe it is still bit-exact, which is the
# property that makes checkpointing composable. A resume that is only
# *approximately* correct degrades every time it is used, and a two-week run
# resumes many times.

# %% [markdown]
# ## What to take away
#
# 1. **"Resumed correctly" means bit-exact, and nothing else is testable.**
#    "The loss looks fine" is not an acceptance criterion, because a subtly
#    wrong resume produces a loss curve that looks fine.
# 2. **An incomplete checkpoint diverges on the very first step after resume**,
#    and by a small amount. Small and immediate is the worst combination: too
#    small to alarm anyone, early enough to affect everything after it.
# 3. **State is six things, not two.** Weights, optimiser moments, schedule
#    position, global RNG, data-loader position, step counter.
# 4. **Store logical layout, not physical bytes.** Names and shapes survive a
#    change in world size; flattened offsets do not, and they fail in a way
#    that works on the easy case first.
# 5. **Rename is the only atomic operation you get.** Write to a temporary
#    file, fsync, rename, and keep the previous checkpoint until the new one
#    has been read back.
