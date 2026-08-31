# %% [markdown]
# # Lab 8b — Precision: what FP8 costs you and where the bits actually go
#
# Companion to [Chapter 8](../book/part-2-pretraining/08-optimization.md).
#
# Low precision is where training runs go wrong quietly. Nothing crashes. The
# loss curve looks fine for a while. Then it stops improving, or diverges at
# step 40,000, and you are trying to work out which of a dozen tensors lost its
# information three hours ago.
#
# This lab uses **real** low-precision types — `torch.float8_e4m3fn`,
# `float8_e5m2`, `bfloat16`, `float16` — not simulated rounding. Every number
# below comes from an actual cast.
#
# You will:
#
# 1. Measure the range and precision of five formats, and see why the choice
#    between BF16 and FP16 is about *range*, not accuracy.
# 2. Reproduce gradient underflow in FP16 and fix it with loss scaling.
# 3. Quantize a tensor to FP8 with per-tensor scaling and watch one outlier
#    destroy every small value in it.
# 4. Fix that with per-block scaling, which is what frontier FP8 recipes
#    actually do.
# 5. Turn off FP32 accumulation in a long reduction and measure the error at
#    8,192 elements.
#
# Pure arithmetic and casting. Runs in seconds on CPU.
#
# **Predict before you run.** A tensor of values around $10^{-2}$ contains one
# value of 1000. You quantize it to FP8 with a single per-tensor scale. What
# fraction of the small values survive? Write down a number.

# %%
from __future__ import annotations

import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# %%
import math
import os

import torch

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
SEED = 0
torch.manual_seed(SEED)
print(f"smoke_test={SMOKE}  torch={torch.__version__}")

FORMATS = [
    ("FP32", torch.float32),
    ("BF16", torch.bfloat16),
    ("FP16", torch.float16),
    ("FP8 E4M3", torch.float8_e4m3fn),
    ("FP8 E5M2", torch.float8_e5m2),
]

# %% [markdown]
# ## 1. Five formats, and what each one gives up
#
# A floating-point format splits its bits between **exponent** (how big and how
# small a number it can express — its *range*) and **mantissa** (how finely it
# can distinguish nearby numbers — its *precision*).
#
# BF16 and FP16 are both 16 bits. BF16 spends more of them on the exponent,
# which is the entire reason it displaced FP16 for training.

# %%
print("Format comparison")
print("-" * 78)
print(f"{'format':<10} {'bits':>5} {'max':>12} {'min normal':>13} "
      f"{'eps':>10} {'decimal digits':>15}")
for name, dt in FORMATS:
    fi = torch.finfo(dt)
    digits = -math.log10(fi.eps)
    print(f"{name:<10} {fi.bits:>5} {fi.max:>12.3g} {fi.tiny:>13.3g} "
          f"{fi.eps:>10.3g} {digits:>15.1f}")

print("\nBF16 and FP16 are both 16 bits and are not interchangeable:")
bf, fp = torch.finfo(torch.bfloat16), torch.finfo(torch.float16)
print(f"  BF16 reaches down to {bf.tiny:.2e} but resolves only {-math.log10(bf.eps):.1f} digits")
print(f"  FP16 reaches down to {fp.tiny:.2e} and resolves {-math.log10(fp.eps):.1f} digits")
print(f"  BF16's usable range is {bf.max/bf.tiny:.1e} wide; FP16's is {fp.max/fp.tiny:.1e}")
print("\nBF16 has the same exponent width as FP32, so anything that fits in FP32")
print("fits in BF16 -- you lose accuracy, never magnitude. That property is why")
print("BF16 needs no loss scaling and FP16 does. Section 2 shows what happens")
print("when you forget.")

# %% [markdown]
# ## 2. Gradient underflow, and loss scaling
#
# Gradients late in training are small. In FP16 anything below $6\times10^{-8}$
# is not merely inaccurate — it is exactly zero, and a zero gradient is
# indistinguishable from a converged parameter.
#
# Loss scaling is the fix: multiply the loss by a large constant $S$ before the
# backward pass, which multiplies every gradient by $S$, then divide the
# gradients by $S$ before the optimiser step. The arithmetic is unchanged; the
# gradients just spend the backward pass somewhere FP16 can represent them.

# %%
torch.manual_seed(SEED)
N = 100_000
# A realistic late-training gradient distribution: log-uniform across scales.
exponents = torch.empty(N).uniform_(-9.0, -3.0)
grads = (10.0 ** exponents) * torch.where(torch.rand(N) < 0.5, -1.0, 1.0)

def survival(x: torch.Tensor, dt: torch.dtype) -> tuple:
    q = x.to(dt).to(torch.float32)
    zeroed = int((q == 0).sum())
    nonzero = x != 0
    rel = ((q[nonzero] - x[nonzero]).abs() / x[nonzero].abs())
    return zeroed, float(rel[torch.isfinite(rel)].mean())

print("\n100,000 gradients spread over 1e-9 .. 1e-3")
print("-" * 78)
print(f"{'format':<22} {'flushed to zero':>17} {'mean rel error':>16}")
for label, dt in (("BF16", torch.bfloat16), ("FP16", torch.float16)):
    z, r = survival(grads, dt)
    print(f"{label:<22} {z:>10,} ({100*z/N:>4.1f}%) {r:>15.2%}")

LOSS_SCALE = 2.0 ** 15
scaled = grads * LOSS_SCALE
q = (scaled.to(torch.float16).to(torch.float32)) / LOSS_SCALE
z = int((q == 0).sum())
rel = ((q - grads).abs() / grads.abs())
print(f"{'FP16 + loss scale 2^15':<22} {z:>10,} ({100*z/N:>4.1f}%) "
      f"{float(rel[torch.isfinite(rel)].mean()):>15.2%}")

print("\nFP16 silently deletes a large share of the gradient signal. Loss scaling")
print("recovers nearly all of it without changing any mathematics -- it just")
print("moves the numbers into the part of the format that exists. BF16 never")
print("had the problem, which is why modern recipes use it and treat FP16 as a")
print("legacy inference format.")

# %% [markdown]
# ## 3. Per-tensor FP8 scaling, and the one value that ruins it
#
# FP8 E4M3 tops out at 448, so a tensor has to be scaled before it can be cast:
#
# $$s = \frac{\max|x|}{448}, \qquad x_q = \mathrm{fp8}(x / s), \qquad
#   \hat{x} = x_q \cdot s$$
#
# The scale is set by the **largest** element. Everything else has to live in
# whatever resolution is left over.

# %%
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def fp8_per_tensor(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max() / FP8_MAX
    q = (x / scale).to(torch.float8_e4m3fn).to(torch.float32)
    return q * scale


def fp8_per_block(x: torch.Tensor, block: int = 128) -> torch.Tensor:
    """One scale per block of `block` elements -- the frontier FP8 recipe."""
    flat = x.flatten()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block)
    scales = blocks.abs().amax(dim=1, keepdim=True) / FP8_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = (blocks / scales).to(torch.float8_e4m3fn).to(torch.float32) * scales
    return q.flatten()[:x.numel()].view_as(x)


torch.manual_seed(SEED)
SIZE = 4096
activations = torch.randn(SIZE) * 0.01          # typical activation scale
clean = activations.clone()
outlier_idx = 7
activations[outlier_idx] = 1000.0               # one massive value


def report(name: str, original: torch.Tensor, recovered: torch.Tensor,
           skip: int) -> None:
    mask = torch.ones_like(original, dtype=torch.bool)
    mask[skip] = False
    small_o, small_r = original[mask], recovered[mask]
    zeroed = int((small_r == 0).sum())
    rel = ((small_r - small_o).abs() / small_o.abs().clamp_min(1e-30))
    print(f"{name:<28} {zeroed:>8,} ({100*zeroed/mask.sum():>5.1f}%) "
          f"{float(rel.median()):>16.1%}")


print(f"\n{SIZE:,} activations ~N(0, 0.01), one value set to 1000")
print("-" * 78)
print(f"{'scheme':<28} {'small values zeroed':>17} {'median rel error':>16}")
report("per-tensor scale", activations, fp8_per_tensor(activations), outlier_idx)
report("per-block scale (128)", activations, fp8_per_block(activations, 128), outlier_idx)
report("per-tensor, no outlier", clean, fp8_per_tensor(clean), outlier_idx)

scale = float(activations.abs().max() / FP8_MAX)
fi8 = torch.finfo(torch.float8_e4m3fn)
smallest_sub = 2.0 ** -9            # E4M3 subnormal spacing
typical = 0.01 / scale
print(f"\nThe per-tensor scale is amax/448 = {scale:.4f}. A typical activation of")
print(f"~0.01 becomes {typical:.2e} after scaling. E4M3's smallest NORMAL value is")
print(f"{fi8.tiny:.4g}, so that lands in the subnormal range, where the only")
print(f"representable values are multiples of {smallest_sub:.6f}.")
print(f"An activation of 0.01 is therefore stored as one of two nearby steps")
print(f"({smallest_sub*2:.6f} or {smallest_sub*3:.6f}), and anything below")
print(f"~{smallest_sub/2:.1e} rounds to exactly zero.")
print("\nThat is the failure: not a degraded tensor but a truncated one, with a")
print("perfectly reasonable-looking maximum.")
print("\nPer-block scaling gives the block containing the outlier its own scale")
print("and leaves the other 31 blocks unharmed. This is why DeepSeek-V3's FP8")
print("recipe scales per 128-element tile rather than per tensor, and why")
print("'we trained in FP8' is a claim about the SCALING GRANULARITY, not about")
print("the format. Chapter 8 section 8.3.")

# %% [markdown]
# ## 4. Accumulation: the other place precision hides
#
# Exercise 8.9 item 4. A dot product of length $K$ is $K$ multiplies and $K-1$
# additions. The multiplies can be low precision; the **running sum** is a
# different question, because error compounds across the reduction.
#
# Every tensor core accumulates into FP32 for this reason. Here we do it both
# ways and measure.

# %%
def sequential_sum(x: torch.Tensor, acc_dtype: torch.dtype) -> float:
    acc = torch.zeros((), dtype=acc_dtype)
    for v in x.to(acc_dtype):
        acc = acc + v
    return float(acc)


def pairwise_sum(x: torch.Tensor, acc_dtype: torch.dtype) -> float:
    """Tree reduction: what a real kernel does. Error grows as log K, not K."""
    cur = x.to(acc_dtype)
    while cur.numel() > 1:
        n = cur.numel()
        if n % 2:
            cur = torch.cat([cur, torch.zeros(1, dtype=acc_dtype)])
            n += 1
        cur = (cur[0::2] + cur[1::2])
    return float(cur[0])


torch.manual_seed(SEED)
LENGTHS = (256, 1024, 8192) if SMOKE else (256, 1024, 4096, 8192, 32768)

print("\nSumming K positive values ~U(0, 0.02); FP64 is ground truth")
print("-" * 78)
print(f"{'K':>8} {'FP32 acc':>12} {'FP16 seq acc':>14} {'FP16 pairwise':>15} "
      f"{'seq/pairwise':>13}")
for K in LENGTHS:
    x = torch.rand(K) * 0.02
    truth = float(x.double().sum())
    e32 = abs(sequential_sum(x, torch.float32) - truth) / truth
    e16 = abs(sequential_sum(x, torch.float16) - truth) / truth
    e16p = abs(pairwise_sum(x, torch.float16) - truth) / truth
    ratio = e16 / e16p if e16p > 0 else float("inf")
    print(f"{K:>8,} {e32:>12.2e} {e16:>14.2e} {e16p:>15.2e} {ratio:>12.0f}x")

print("\nThree things in that table.")
print("  1. FP32 accumulation stays near machine precision at every length.")
print("  2. FP16 sequential accumulation degrades with K, because each addition")
print("     rounds a running total that is large relative to the next term --")
print("     eventually the addition changes nothing at all.")
print("  3. The reduction ORDER matters as much as the format: pairwise")
print("     summation in the same FP16 is far more accurate, because error")
print("     grows as log K rather than K.")
print("\nSo 'we compute in FP8' and 'we accumulate in FP8' are completely")
print("different claims, and only the first one is common.")

# %% [markdown]
# ## 5. Does it survive into the matmul?
#
# Section 3 measured damage to a *tensor*. Whether that reaches the operation
# you care about is a separate question, and I assumed the answer was obviously
# yes. It is not obvious, and working out when it is `yes` took several tries.
#
# The construction below is the realistic one. A large **outlier feature** — a
# single channel far above the rest, which is a documented property of real
# transformer activations — multiplied by weights that are near zero in that
# channel. The outlier therefore contributes almost nothing to the output while
# still setting the scale for the entire tensor.

# %%
torch.manual_seed(SEED)
M, K, Nn = 64, 1024, 64

print(f"\nMatmul ({M}x{K}) @ ({K}x{Nn}), FP32 accumulation, mean relative error")
print("-" * 78)
print(f"{'outlier feature':<20} {'spread':>10} {'BF16':>11} {'FP8 tensor':>12} "
      f"{'FP8 block':>11} {'gain':>7}")
for mag in (0.0, 60.0, 1000.0):
    torch.manual_seed(SEED)
    A = torch.randn(M, K) * 0.01
    B = torch.randn(K, Nn) * 0.01
    if mag:
        A[:, 0] = mag
        B[0, :] = 0.0                 # the outlier channel is multiplied by ~0
    ref = (A.double() @ B.double()).float()

    def rel(x: torch.Tensor) -> float:
        return float((x - ref).abs().mean() / ref.abs().mean())

    e_bf = rel(A.to(torch.bfloat16).float() @ B.to(torch.bfloat16).float())
    e_t = rel(fp8_per_tensor(A) @ fp8_per_tensor(B))
    e_b = rel(fp8_per_block(A, 128) @ fp8_per_block(B, 128))
    label = "none" if not mag else f"{mag:g}"
    spread = "-" if not mag else f"{mag/0.01:.0e}x"
    print(f"{label:<20} {spread:>10} {e_bf:>11.3e} {e_t:>12.3e} {e_b:>11.3e} "
          f"{e_t/e_b:>6.1f}x")

print("\nRead the rows in order, because the middle one is the surprise.")
print("\nAt 6e+03x the two FP8 schemes are indistinguishable. The outlier is")
print("large but E4M3 still spans it: min normal 0.0156 up to 448 is a range of")
print("~29,000x, and a 6,000x spread fits inside it with room to spare. Nothing")
print("flushes, so there is nothing for finer scaling to rescue.")
print("\nAt 1e+05x the spread finally exceeds the format's dynamic range, small")
print("values start hitting subnormals and zero, and per-block scaling buys a")
print("real improvement -- roughly 2x here, not the orders of magnitude I")
print("expected before running it.")
print("\nSo the rule is not 'per-block scaling is better'. It is: scaling")
print("granularity matters exactly when the dynamic range WITHIN a scaling")
print("group exceeds what the format can hold. That is a property of your")
print("activations, and it is why FP8 recipes are reported alongside outlier")
print("statistics rather than on their own. Chapter 8 section 8.3.")

# %% [markdown]
# ## 6. Things to try
#
# **1. Sweep the outlier magnitude in section 3 from 1 to 10,000.**
# *Common prediction:* damage rises smoothly with the outlier.
# *What happens:* nothing happens at all until the ratio to the typical value
# approaches E4M3's dynamic range, and then it collapses quickly. Precision
# failures are threshold effects, not gradual ones, which is why they arrive as
# a cliff at step 40,000 rather than as a slow drift you could have watched.
#
# **2. Set `LOSS_SCALE = 2**24` in section 2.**
# *Common prediction:* even fewer gradients are lost.
# *What happens:* the small gradients are saved and the *large* ones overflow to
# infinity, because FP16 tops out at 65,504. This is why real implementations
# use dynamic loss scaling — raise the scale until you see an inf, then back
# off and skip that step.
#
# **3. Use `float8_e5m2` instead of `e4m3fn` in section 3.**
# *Common prediction:* worse, since it has fewer mantissa bits.
# *What happens:* the small values survive better — E5M2's extra exponent bit
# buys the range that the outlier stole. This is the actual reason mixed FP8
# recipes use E4M3 for weights and activations but E5M2 for gradients, which
# have a much wider dynamic range.
#
# **4. Change `pairwise_sum` to accumulate in `bfloat16`.**
# *Common prediction:* similar to FP16, since both are 16 bits.
# *What happens:* noticeably worse. BF16 has three fewer mantissa bits, and
# accumulation is precision-bound, not range-bound. BF16 is the better format
# for storing values and the worse one for adding them up.
#
# **5. Sum values of mixed magnitude — `torch.rand(K) * 0.02` plus one value of
# 1e4 — and compare sorted against unsorted.**
# *Common prediction:* order does not matter.
# *What happens:* summing largest-first is markedly worse than smallest-first.
# Numerical libraries sort or use compensated summation for exactly this
# reason.

# %% [markdown]
# ## What to take away
#
# 1. **BF16 versus FP16 is a range decision, not an accuracy one.** BF16 keeps
#    FP32's exponent, so it cannot underflow where FP32 would not. FP16 has
#    more mantissa and a range narrow enough to delete real gradients.
# 2. **Loss scaling is bookkeeping, not mathematics.** It moves gradients into
#    the representable band and moves them back. Set it too high and you
#    overflow instead, which is why it is made dynamic.
# 3. **A per-tensor FP8 scale is set by the worst element in the tensor.** One
#    outlier and every small value is flushed to zero, with no error, no
#    warning, and a healthy-looking maximum.
# 4. **Scaling granularity matters exactly when the spread inside a scaling
#    group exceeds the format's dynamic range — and not before.** An outlier
#    6,000× above typical changed nothing here, because E4M3 spans ~29,000×.
#    At 100,000× per-block scaling bought ~2×. "We trained in FP8" is a claim
#    about granularity *and* about your activation statistics; neither is
#    interpretable alone.
# 5. **Compute precision and accumulate precision are different decisions.**
#    Accumulate in FP32. And when you cannot, reduce pairwise — the order of a
#    reduction matters as much as its format.
