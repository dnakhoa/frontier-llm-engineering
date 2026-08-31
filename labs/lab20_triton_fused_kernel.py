# %% [markdown]
# # Lab 20 — Kernel fusion: the arithmetic, the measurement, and the kernel
#
# Companion to [Chapter 20](../book/part-4-infra/20-custom-kernels.md).
#
# **Read this first, because this lab has an honesty problem the others do not.**
# Triton compiles to GPU code. This book's labs are CPU-first, and CI has no
# GPU, so a lab that *requires* one would be a lab most readers cannot run.
#
# So this lab is split by what can actually be established where:
#
# | Section | Needs a GPU? | Status here |
# |---|---|---|
# | 1. Why fusion wins: memory-traffic arithmetic | No | exact |
# | 2. Fused vs unfused, measured | No | real measurement on CPU |
# | 3. `torch.compile` as the baseline you must beat | No | real measurement |
# | 4. Numerical equivalence and FP32 accumulation | No | exact, real dtypes |
# | 5. The Triton kernel | **Yes** | source included, runs if a GPU exists |
# | 6. What only a profiler can tell you | **Yes** | discussed, not measured |
#
# The arithmetic in section 1 is the part that actually transfers. A fused
# kernel is not faster because it is clever; it is faster because it moves
# fewer bytes, and you can predict how much before writing any code.
#
# **Predict before you run.** Section 1 computes the memory traffic of RMSNorm
# written with explicit intermediates versus a single fused pass. Guess the
# ratio, then guess how much of it you will actually see on a real machine.

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
import time

import torch

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

torch.manual_seed(0)
print(f"torch={torch.__version__}  cuda={HAS_CUDA}  triton={HAS_TRITON}  "
      f"smoke_test={SMOKE}")

# %% [markdown]
# ## 1. Why fusion wins, in bytes
#
# The operation is **RMSNorm**, the normalisation used by essentially every
# current LLM:
#
# $$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot w$$
#
# Written naively, each step is a separate kernel launch that reads its input
# from HBM and writes its output back. Written fused, a tile of $x$ is loaded
# once into registers and everything happens there.
#
# Count the traffic. This is the whole argument for custom kernels, and it is
# arithmetic you can do in your head before committing to a week of work.

# %%
def traffic_unfused(rows: int, d: int, bytes_per: int = 2) -> dict:
    """Every intermediate materialised, one kernel per step."""
    n = rows * d
    steps = [
        ("sq = x * x", n, n),                    # read x, write sq
        ("m = sq.mean(-1)", n, rows),            # read sq, write m
        ("r = rsqrt(m + eps)", rows, rows),      # trivial
        ("xn = x * r", n + rows, n),             # read x again, write xn
        ("y = xn * w", n + d, n),                # read xn and w, write y
    ]
    reads = sum(r for _, r, _ in steps)
    writes = sum(w for _, _, w in steps)
    return {"steps": steps, "reads": reads, "writes": writes,
            "bytes": (reads + writes) * bytes_per}


def traffic_fused(rows: int, d: int, bytes_per: int = 2) -> dict:
    """One pass: load a tile of x, reduce, normalise, scale, store."""
    n = rows * d
    # x is read once for the reduction and once for the normalisation. A tile
    # that fits in registers or SRAM is only fetched from HBM once.
    reads = n + d
    writes = n
    return {"reads": reads, "writes": writes,
            "bytes": (reads + writes) * bytes_per}


ROWS, D = 4096, 4096
un = traffic_unfused(ROWS, D)
fu = traffic_fused(ROWS, D)

print(f"\nRMSNorm on a {ROWS} x {D} activation, BF16")
print("-" * 74)
print(f"{'step':<22} {'elements read':>15} {'elements written':>18}")
for name, r, w in un["steps"]:
    print(f"{name:<22} {r:>15,} {w:>18,}")
print("-" * 74)
print(f"{'unfused total':<22} {un['reads']:>15,} {un['writes']:>18,}")
print(f"{'fused total':<22} {fu['reads']:>15,} {fu['writes']:>18,}")
print()
print(f"{'unfused HBM traffic':<28}: {un['bytes']/1e9:>8.3f} GB")
print(f"{'fused HBM traffic':<28}: {fu['bytes']/1e9:>8.3f} GB")
print(f"{'predicted speedup if memory-bound':<28}: "
      f"{un['bytes']/fu['bytes']:>8.2f}x")

# Arithmetic intensity: FLOPs per byte moved.
flops = ROWS * D * 4          # square, add, multiply, multiply -- roughly
print(f"\n{'FLOPs':<28}: {flops/1e9:>8.3f} GFLOP")
print(f"{'arithmetic intensity, unfused':<28}: "
      f"{flops/un['bytes']:>8.3f} FLOP/byte")
print(f"{'arithmetic intensity, fused':<28}: "
      f"{flops/fu['bytes']:>8.3f} FLOP/byte")
print("\nAn H100 does roughly 990 TFLOP/s in BF16 against about 3.35 TB/s of")
print("memory bandwidth -- a ratio near 295 FLOP/byte. Both numbers above are")
print("far below that, so this operation is memory-bound by a wide margin and")
print("its runtime is set by bytes moved, not by arithmetic.")
print("\nThat is the entire case for fusing it, and it is also the reason not to")
print("bother hand-writing a fused matmul: a matmul's arithmetic intensity is")
print("already above the ratio, so it is compute-bound and cuBLAS has spent a")
print("decade on it. Fuse the memory-bound things around the matmul instead.")

# %% [markdown]
# ## 2. Fused versus unfused, measured
#
# The arithmetic predicts a ratio. Reality gives you part of it, and the gap is
# where caches, launch overhead and vectorisation live.

# %%
def rms_unfused(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sq = x * x
    m = sq.mean(-1, keepdim=True)
    r = torch.rsqrt(m + eps)
    xn = x * r
    return xn * w


def rms_chained(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def bench(fn, *args, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn(*args)
    if HAS_CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if HAS_CUDA:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


DIMS = (512, 2048) if SMOKE else (512, 2048, 8192)
ROWS_BENCH = 512 if SMOKE else 1024
ITERS = 5 if SMOKE else 20

print(f"\nWall clock, {ROWS_BENCH} rows, FP32 on {('cuda' if HAS_CUDA else 'cpu')}")
print("-" * 74)
print(f"{'d':>7} {'unfused (ms)':>14} {'chained (ms)':>14} {'speedup':>10} "
      f"{'predicted':>11}")
for d in DIMS:
    x = torch.randn(ROWS_BENCH, d, device="cuda" if HAS_CUDA else "cpu")
    w = torch.randn(d, device=x.device)
    t_un = bench(rms_unfused, x, w, iters=ITERS)
    t_ch = bench(rms_chained, x, w, iters=ITERS)
    pred = (traffic_unfused(ROWS_BENCH, d, 4)["bytes"]
            / traffic_fused(ROWS_BENCH, d, 4)["bytes"])
    print(f"{d:>7} {1000*t_un:>14.3f} {1000*t_ch:>14.3f} "
          f"{t_un/t_ch:>9.2f}x {pred:>10.2f}x")

print("\nThat column of roughly 1.00x is the lesson, and it is not the one I")
print("expected to write. Rewriting the operation with fewer named temporaries")
print("bought essentially nothing.")
print("\nThe reason is that `rms_chained` is not fused. Python-level chaining")
print("changes how the code READS; it does not change what the runtime does,")
print("which is still to allocate a tensor for each intermediate and write it")
print("to memory. The byte count in section 1 is unchanged, so the time is")
print("unchanged.")
print("\nFusion is a property of the generated code, not of the source. To get")
print("any of that predicted 3.5x, the intermediates have to never reach memory")
print("-- which needs either a compiler that rewrites the whole chain, or a")
print("kernel that keeps the tile in registers. Both are next.")

# %% [markdown]
# ## 3. `torch.compile` is the baseline you have to beat
#
# Exercise 20.8 item 2. Before writing a kernel, find out what the compiler
# already does — it fuses elementwise chains automatically, and a hand-written
# kernel that merely matches it has cost you a week.

# %%
if SMOKE:
    print("\n(skipped in smoke mode: torch.compile spends several seconds warming"
          " up)")
else:
    print("\nEager versus torch.compile")
    print("-" * 74)
    print(f"{'d':>7} {'eager (ms)':>13} {'compiled (ms)':>15} {'speedup':>10}")
    for d in DIMS:
        x = torch.randn(ROWS_BENCH, d, device="cuda" if HAS_CUDA else "cpu")
        w = torch.randn(d, device=x.device)
        try:
            compiled = torch.compile(rms_chained)
            compiled(x, w)                     # trigger compilation
            t_eager = bench(rms_chained, x, w, iters=ITERS)
            t_comp = bench(compiled, x, w, iters=ITERS)
            print(f"{d:>7} {1000*t_eager:>13.3f} {1000*t_comp:>15.3f} "
                  f"{t_eager/t_comp:>9.2f}x")
        except Exception as exc:               # pragma: no cover
            print(f"{d:>7}  torch.compile unavailable: {type(exc).__name__}")

    print("\nThere is the fusion that section 2 failed to get by hand, and note")
    print("where it appears: the gain grows with d, and is largest at the size")
    print("where the tensors stop fitting in cache and the byte count starts to")
    print("govern. At small d the whole working set is cached, memory traffic is")
    print("nearly free, and there is nothing for fusion to win.")
    print("\nSo the compiler recovers a substantial share of section 1's ceiling")
    print("for the cost of one decorator. THAT is the number a custom kernel has")
    print("to beat -- not the eager number, which is the comparison that makes")
    print("hand-written kernels look better than they are.")
    print("\nChapter 20's advice is deliberately deflating: measure torch.compile")
    print("first, and write Triton only where you can name the reason the")
    print("compiler cannot do the job -- an unusual reduction, a fused backward,")
    print("a memory layout it will not choose, an op with no elementwise")
    print("decomposition at all.")

# %% [markdown]
# ## 4. Numerical equivalence, and where FP32 accumulation matters
#
# Exercise 20.8 items 1 and 5. A fused kernel is only correct if it agrees with
# the reference, and the usual cause of disagreement is the accumulator.
#
# RMSNorm contains a reduction over $d$ elements. Do it in the storage dtype
# and the error grows with $d$; do it in FP32 and it does not. This is the
# same mechanism as
# [`lab08_precision_and_stability`](lab08_precision_and_stability.py) section 4,
# arriving in a place where it silently changes a layer's output.

# %%
def sum_squares(row: torch.Tensor, acc_dtype: torch.dtype) -> float:
    """Sequential sum of squares with a REAL accumulator of the given dtype.

    torch.mean on a bfloat16 tensor already accumulates internally in FP32, so
    calling it twice with different input dtypes does not test the accumulator
    at all -- that was my first version of this section and it showed a 1.3x
    difference that had nothing to do with accumulation. To measure the thing
    the kernel actually controls, the running total has to be held in the dtype
    under test.
    """
    acc = torch.zeros((), dtype=acc_dtype)
    for v in row.to(acc_dtype):
        acc = acc + v * v
    return float(acc)


print("\nSum of squares over d elements: what dtype is the RUNNING TOTAL?")
print("(inputs are identical BF16 values in every column)")
print("-" * 74)
print(f"{'d':>7} {'BF16 accumulator':>20} {'FP32 accumulator':>20} {'ratio':>9}")
DIMS_ACC = (512, 2048) if SMOKE else (512, 2048, 8192)
for d in DIMS_ACC:
    torch.manual_seed(0)
    row = (torch.randn(d) * 0.5).to(torch.bfloat16)
    truth = float((row.double() ** 2).sum())
    e_low = abs(sum_squares(row, torch.bfloat16) - truth) / truth
    e_hi = abs(sum_squares(row, torch.float32) - truth) / truth
    print(f"{d:>7} {e_low:>20.3e} {e_hi:>20.3e} "
          f"{e_low/max(e_hi, 1e-30):>8.0f}x")

print("\nThe inputs are the same BF16 numbers in both columns. The only")
print("difference is the dtype of the running total, and the gap widens with d")
print("because each addition rounds a total that is growing relative to the")
print("next term.")
print("\nThis is why `tl.sum` accumulates in FP32 even when the tile was loaded")
print("as BF16, and why 'my Triton kernel does not match the reference' is")
print("almost always the accumulator. Exercise 20.8 item 1 says to check it")
print("first, and that is the right order.")
print("\nIt is also a trap in the other direction: because PyTorch quietly")
print("upcasts reductions for you, a naive comparison in PyTorch will NOT")
print("reproduce this, and you can convince yourself the accumulator does not")
print("matter right up until you write the kernel yourself.")

# %% [markdown]
# ## 5. The kernel
#
# Here is the real thing. It runs one program per row: load the row, reduce in
# FP32, normalise, scale, store. One read of $x$, one write of $y$, and no
# intermediates in memory at all.
#
# It executes below **only if Triton and a CUDA device are present**. There is
# no CPU fallback for Triton, and pretending otherwise would be worse than
# saying so.

# %%
TRITON_SOURCE = '''
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd(
    x_ptr, w_ptr, y_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    y_row = y_ptr + row * stride_row

    # --- pass 1: sum of squares, accumulated in FP32 --------------------
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += vals * vals
    mean_sq = tl.sum(acc, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    # --- pass 2: normalise and scale, still no intermediate in HBM ------
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        wts = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + cols, (vals * inv_rms * wts).to(y_ptr.dtype.element_ty),
                 mask=mask)


def rmsnorm_triton(x, w, eps=1e-6):
    y = torch.empty_like(x)
    rows, cols = x.shape
    BLOCK = min(triton.next_power_of_2(cols), 1024)
    num_warps = 4 if cols <= 2048 else 8
    rmsnorm_fwd[(rows,)](
        x, w, y, x.stride(0), cols, eps,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return y
'''

print("\nThe kernel source:")
print("-" * 74)
print(TRITON_SOURCE.strip())

print("\n" + "-" * 74)
if HAS_TRITON and HAS_CUDA:
    ns: dict = {"torch": torch}
    exec(TRITON_SOURCE, ns)                     # noqa: S102 - lab code, by design
    rmsnorm_triton = ns["rmsnorm_triton"]
    print("Triton and CUDA present -- running the kernel.")
    for d in DIMS:
        x = torch.randn(ROWS_BENCH, d, device="cuda", dtype=torch.float32)
        w = torch.randn(d, device="cuda", dtype=torch.float32)
        y_t = rmsnorm_triton(x, w)
        y_r = rms_chained(x, w)
        ok = torch.allclose(y_t, y_r, atol=1e-4, rtol=1e-4)
        t_t = bench(rmsnorm_triton, x, w, iters=ITERS)
        t_c = bench(rms_chained, x, w, iters=ITERS)
        print(f"  d={d:>6}  matches reference: {ok}   "
              f"triton {1000*t_t:.3f} ms vs chained {1000*t_c:.3f} ms "
              f"({t_c/t_t:.2f}x)")
else:
    missing = []
    if not HAS_TRITON:
        missing.append("triton is not installed")
    if not HAS_CUDA:
        missing.append("no CUDA device is available")
    print("NOT RUN: " + "; ".join(missing) + ".")
    print()
    print("To run it: open this lab in Colab with a GPU runtime, where triton")
    print("ships with torch, and rerun this cell. Everything above ran here.")
    print()
    print("What the kernel would show, from section 1's arithmetic: one read of")
    print("x, one read of w, one write of y, against the unfused version's seven")
    print("passes -- so a speedup approaching the predicted ratio rather than")
    print("the fraction of it measured in section 2, because the intermediates")
    print("genuinely never reach memory.")

# %% [markdown]
# ## 6. What only a GPU and a profiler can tell you
#
# Three of Exercise 20.8's items cannot be answered on a CPU, and it is worth
# being precise about *why* rather than leaving them as homework.
#
# **`num_warps` (item 4).** A warp is 32 threads scheduled together. `num_warps`
# sets how many of them run one program instance, which sets how much of the
# row each thread handles and how many registers each gets. At small `d`, more
# warps means most threads idle and the launch dominates; at large `d`, too few
# warps means each thread loops over too much data and cannot hide memory
# latency. The knee is hardware-specific — it depends on registers per SM and
# on occupancy — and it is why Triton has an autotuner rather than a default.
#
# **Where a hand-written kernel loses (item 3).** Sweeping $d$ from 512 to
# 16384, expect to lose at both ends. Small $d$: launch overhead dominates and
# a fused-away kernel is better. Large $d$: the row stops fitting in SRAM, the
# two passes in the kernel above start re-reading from HBM, and the traffic
# advantage evaporates — at which point you need a single-pass algorithm
# (Welford, or the online softmax trick) rather than a two-pass one.
#
# **Occupancy and bandwidth utilisation.** The only honest way to know whether a
# memory-bound kernel is done is to compare achieved bandwidth against the
# device's peak. Section 1 gives you the byte count; `ncu` or
# `torch.profiler` gives you the time; the ratio tells you whether there is
# anything left to win. Without that ratio you are guessing.
#
# Section 1's arithmetic is what makes all three of those *predictions* rather
# than experiments, and that is the part worth carrying to a machine that has a
# GPU on it.

# %% [markdown]
# ## 7. Things to try
#
# **1. Change `traffic_unfused` to assume the intermediates stay in cache.**
# *Common prediction:* the predicted speedup collapses to 1.
# *What happens:* it drops a great deal but not to 1, because $x$ is still read
# twice and $y$ written once. This is why fusion helps less on CPUs with large
# caches than on GPUs, and why section 2's measured ratio is well below its
# prediction.
#
# **2. Run section 2 with BF16 inputs instead of FP32.**
# *Common prediction:* twice as fast, being half the bytes.
# *What happens:* on CPU it is often *slower*, because BF16 elementwise work
# gets converted to FP32 to compute and back. Halving the bytes only helps when
# you were bandwidth-bound, and a laptop CPU on a small tensor frequently is
# not.
#
# **3. Add a fused backward pass to the Triton source.**
# The forward pass is the easy half. The backward needs the reduction's
# gradient, which couples every element of the row — this is where hand-written
# kernels earn their keep, because `torch.compile` fuses forward chains far
# better than it fuses backward ones.
#
# **4. Compute the arithmetic intensity of a matmul at these shapes and compare
# with section 1.**
# *Common prediction:* somewhat higher.
# *What happens:* orders of magnitude higher, above the hardware's FLOP/byte
# ratio, so it is compute-bound. That single comparison explains why nobody
# hand-writes matmuls and everybody hand-writes normalisations.
#
# **5. Rewrite the kernel's two passes as one using Welford's algorithm.**
# *Common prediction:* the same speed, cleaner code.
# *What happens:* it matters exactly when the row does not fit in SRAM, which is
# the large-$d$ regime where the two-pass version starts losing. Predict the
# crossover from your cache size before measuring it.

# %% [markdown]
# ## What to take away
#
# 1. **Fusion is a bytes argument, not a cleverness argument.** Count the
#    traffic first; the ratio you compute is the ceiling on what any
#    implementation can win.
# 2. **Check arithmetic intensity against your hardware's FLOP/byte ratio.**
#    Below it you are memory-bound and fusion pays; above it you are
#    compute-bound and a vendor library has already won.
# 3. **Fusion is a property of the generated code, not of the source.**
#    Rewriting the chain with fewer temporaries changed nothing; the compiler
#    fusing it recovered a large share of the predicted ceiling, and most of
#    that gain appeared only at the size where the data stopped fitting in
#    cache. `torch.compile` is the baseline a custom kernel must beat, and you
#    should be able to say in one sentence why the compiler cannot do what you
#    are about to do by hand.
# 4. **When a kernel disagrees with the reference, suspect the accumulator.**
#    Holding the running total in BF16 rather than FP32 is five orders of
#    magnitude of error by $d = 8192$, and nothing crashes. PyTorch hides this
#    by upcasting reductions for you, so you cannot reproduce it in PyTorch and
#    will meet it for the first time inside your own kernel.
# 5. **Two-pass kernels stop working when the row stops fitting.** The fix is a
#    single-pass algorithm, and you can predict the crossover from cache size
#    rather than discovering it.
# 6. **Some questions need the hardware.** `num_warps`, occupancy and achieved
#    bandwidth are not derivable from arithmetic, and a lab that pretended
#    otherwise would be teaching you something false.
