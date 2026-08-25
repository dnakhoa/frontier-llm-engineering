# Chapter 20: Custom kernels — Triton, CUDA, fused ops

> Reading time: ~45 minutes. This is a densely technical chapter; skim it on a first pass if you need to. By the end you should be able to compute arithmetic intensity and predict whether an operation is compute-bound or memory-bound, explain what FlashAttention actually does and why it is not an approximation, write a fused kernel in Triton, and benchmark honestly — including reporting the cases where your kernel loses.

## 20.1 Why anyone writes kernels

A frontier pre-training run costs millions of dollars and occupies thousands of GPUs for weeks. A 20% speedup on the training step is 20% of that bill and 20% of the calendar. One engineer spending a month to get it is an obviously good trade, and that arithmetic is why kernel engineers exist as a distinct role at every frontier lab.

The other reason is that the alternative is often *impossible* rather than merely slow. FlashAttention [\[19\]](../appendix/b-references.md#19-flashattention) did not make attention 2× faster; it made long-context attention *fit in memory*, by never materializing the $N \times N$ attention matrix at all. Training at 128K context is not a speed optimization over training at 128K context slowly — without the kernel there is no 128K context.

This chapter is about the mental model and the mechanics. It will not make you a kernel engineer; it will let you read a kernel, know whether one is worth writing, and hold a conversation with the person who writes them.

## 20.2 The GPU model you actually need

Not the full architecture. Four facts.

**1. There is a memory hierarchy, and the gaps are enormous.**

For an H100, approximately:

| Level | Size | Bandwidth | Latency |
|---|---|---|---|
| Registers | ~256 KB per SM | ~100 TB/s | ~1 cycle |
| Shared memory / L1 | 228 KB per SM | ~30 TB/s | ~30 cycles |
| L2 cache | 50 MB | ~10 TB/s | ~200 cycles |
| HBM3 | 80 GB | ~3.35 TB/s | ~500 cycles |
| Host DRAM (over PCIe) | — | ~64 GB/s | ~microseconds |

The ratio between shared memory and HBM bandwidth is roughly 10×, and between registers and HBM roughly 30×. **Almost all kernel optimization is about moving work up this hierarchy and keeping it there.**

**2. Compute is much faster than memory.**

An H100 does roughly 990 TFLOP/s of dense BF16 matrix math on its tensor cores, against 3.35 TB/s of HBM bandwidth. Dividing:

$$\text{ridge point} = \frac{990 \times 10^{12} \text{ FLOP/s}}{3.35 \times 10^{12} \text{ byte/s}} \approx 295 \text{ FLOP per byte}$$

**To saturate the compute units, you must do about 295 floating-point operations for every byte you read from HBM.** Fewer than that and you are waiting on memory, with the tensor cores idle.

That number is the single most useful thing in this chapter. Most operations in a transformer do not come close to it.

**3. Execution is in warps of 32 threads, in lockstep.**

Threads are grouped into warps that execute the same instruction together. A branch where some threads go one way and some the other is *divergent*: both paths execute serially, with the inactive threads masked. Divergence within a warp costs you directly, which is why GPU code avoids data-dependent branching.

**4. Kernel launches cost real time.**

Roughly 5–10 microseconds of overhead per launch. For a large matrix multiply this is nothing. For an element-wise multiply on a small tensor, the launch costs more than the work — and a transformer layer contains dozens of such operations.

## 20.3 The three regimes

Every operation is limited by one of three things, and knowing which determines what to do about it [\[86\]](../appendix/b-references.md#86-making-deep-learning-go-brrrr).

**Compute-bound.** Doing arithmetic, tensor cores saturated. Large matrix multiplies. The only fix is fewer FLOPs or lower precision.

**Memory-bound.** Waiting for data. Element-wise operations, normalizations, activations, and — crucially — *all* of autoregressive decoding. The fix is fusion: touch the data fewer times.

**Overhead-bound.** Waiting for the CPU to launch the next kernel, or for Python. Small operations in sequence. The fix is fusion or CUDA graphs.

**Arithmetic intensity** tells you which regime you are in:

$$I = \frac{\text{FLOPs performed}}{\text{bytes moved from HBM}}$$

Compare $I$ against the ridge point. Worked examples for a transformer with hidden size $d = 8192$, in BF16:

**A big matmul.** $(B \cdot T, d) \times (d, 4d)$ with $B \cdot T = 8192$:

- FLOPs: $2 \times 8192 \times 8192 \times 32768 = 4.4 \times 10^{12}$
- Bytes: $(8192 \cdot 8192 + 8192 \cdot 32768 + 8192 \cdot 32768) \times 2 \approx 1.21 \times 10^9$
- $I \approx 3{,}600$ FLOP/byte → **compute-bound**, by more than an order of magnitude. Good.

**RMSNorm.** Read $(8192, 8192)$, do a handful of operations per element, write it back:

- FLOPs: roughly $4 \times 8192 \times 8192 = 2.7 \times 10^8$
- Bytes: read + write $= 2 \times 8192 \times 8192 \times 2 = 2.7 \times 10^8$
- $I \approx 1$ FLOP/byte → **memory-bound**, by a factor of ~295.

RMSNorm uses about 0.3% of the machine's compute capability. It is pure memory traffic. And a transformer layer has two of them, plus a SwiGLU activation, plus residual adds, plus RoPE — all in the same regime.

**This is where fusion pays.** Each of those operations, run separately, reads the whole tensor from HBM and writes it back. Fuse them and you read once, do all the work in registers, and write once.

## 20.4 Fusion

The core idea, in one comparison.

```python
# Unfused: four kernels, four round trips to HBM.
#   Each line reads a (B, T, d) tensor from HBM and writes one back.
h = rms_norm(x, w1)          # read x,      write h
h = h @ w_gate               # read h,      write h
h = silu(h)                  # read h,      write h
out = h * (x @ w_up)         # read h and x, write out
```

If the tensor is 128 MB, that is roughly 1 GB of HBM traffic. At 3.35 TB/s, about 300 µs — for work whose actual arithmetic would take under 10 µs.

```python
# Fused: one kernel.
#   Load a tile of x into shared memory once. Do the normalization, the
#   projection, the activation, and the gating without ever writing an
#   intermediate back to HBM. Write only the result.
out = fused_rmsnorm_swiglu(x, w1, w_gate, w_up)
```

Roughly 250 MB of traffic instead of 1 GB. Close to a 4× speedup on an operation that was doing no useful arithmetic anyway.

Fusion also eliminates three kernel launches (§20.2's fact 4) and the memory allocations for the intermediates. In a training loop where the same layer runs 80 times per forward pass, these add up.

## 20.5 FlashAttention

The canonical kernel, and the one worth understanding in detail because it demonstrates the whole discipline.

**The problem.** Standard attention computes $S = QK^\top$, an $N \times N$ matrix, then $P = \mathrm{softmax}(S)$, then $O = PV$. For $N = 8192$ and 32 heads in BF16, $S$ alone is:

$$32 \times 8192 \times 8192 \times 2 \text{ bytes} = 4.3 \text{ GB}$$

Per layer, per sequence in the batch. At 128K context it is 1,100 GB. This is why long context was impossible before 2022, and it is a *memory* problem before it is a speed problem.

**The insight.** You never need the full matrix. Softmax appears to require the global maximum and the global sum, which appear to require all of $S$ — but the **online softmax** trick lets you compute it incrementally, rescaling as you go.

Process $K$ and $V$ in blocks. Maintain a running maximum $m$, a running sum $\ell$, and a running output accumulator $O$. When a new block arrives with a larger maximum, rescale the accumulators by $e^{m_{\text{old}} - m_{\text{new}}}$ and continue.

```python
# Conceptual FlashAttention forward, for one query block.
# The real kernel does this in registers and shared memory; nothing below
# ever touches HBM except the loads of K and V blocks.

m = float("-inf")          # running max, for numerical stability
l = 0.0                    # running sum of exponentials
O = zeros(block_q, d_head) # running weighted-value accumulator

for K_block, V_block in blocks_of(K, V):
    S = Q_block @ K_block.T * scale          # (block_q, block_k) -- small!

    m_new = maximum(m, S.max(axis=-1))
    correction = exp(m - m_new)              # rescale factor for old state

    P = exp(S - m_new)                       # unnormalized probabilities
    l = l * correction + P.sum(axis=-1)
    O = O * correction + P @ V_block

    m = m_new

O = O / l                                     # normalize once, at the end
```

**What this achieves.** Memory goes from $O(N^2)$ to $O(N)$ — you only ever hold a block. And it is *faster*, despite doing slightly more arithmetic (the rescaling), because it is memory-bound and you removed almost all the memory traffic.

**Two things people get wrong about it.**

**It is not an approximation.** The output is numerically equivalent to standard attention, up to floating-point associativity. This is not sparse attention or a linear-attention approximation; it computes exact attention with a better memory schedule.

**The backward pass recomputes $S$ rather than storing it.** Storing $S$ for the backward pass would reintroduce the $O(N^2)$ memory. So the backward recomputes it from $Q$, $K$, $V$. This is activation recomputation (§6.12) applied inside a single operation, and it is why FlashAttention's backward is relatively more expensive than its forward.

**The lineage.** FlashAttention-2 [\[83\]](../appendix/b-references.md#83-flashattention-2) improved work partitioning across warps and reduced non-matmul FLOPs, roughly doubling throughput. FlashAttention-3 [\[84\]](../appendix/b-references.md#84-flashattention-3) targets Hopper specifically, using asynchronous tensor cores and the TMA engine to overlap data movement with computation, plus FP8 support. Each generation is the same algorithm mapped better onto newer hardware — which is itself the lesson: **kernels are hardware-specific, and they age.**

## 20.6 Triton

Writing CUDA is slow, error-prone, and requires managing shared memory, thread indices, and synchronization by hand. Triton [\[85\]](../appendix/b-references.md#85-triton) raises the abstraction to *blocks*: you write code that operates on tiles, and the compiler handles the intra-block parallelism, the memory coalescing, and much of the scheduling.

The result is typically 80–95% of hand-written CUDA performance at perhaps 20% of the effort, in Python. For most fusion work, that trade is correct, and it is why Triton is now the default for kernel work outside the top few percent of hot loops.

```python
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,   # compile-time constant -> unrolled
):
    """Fused RMSNorm: one HBM read, one HBM write, everything else in SRAM.

    Each program instance handles one row. tl.program_id gives the row index;
    the whole row is loaded into fast memory as a single tile.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Single read from HBM. Everything after this stays in registers/SRAM.
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)   # accumulate in FP32 even when the data is BF16

    # rms = sqrt(mean(x^2) + eps)
    mean_square = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_square + eps)

    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    # Single write back to HBM.
    tl.store(out_ptr + row * n_cols + offsets, y.to(tl.bfloat16), mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    """Launch one program per row."""
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](
        x, weight, out, n_cols, eps,
        BLOCK_SIZE=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return out
```

What to notice, because these are the recurring patterns:

- **`tl.constexpr`** marks compile-time constants. Triton specializes the kernel per value, so loop bounds are known and unrolled.
- **Masking** handles rows whose length is not a power of two. `other=0.0` supplies a safe value for out-of-range lanes.
- **FP32 accumulation** for the reduction, even with BF16 inputs and outputs. §20.9.
- **`num_warps`** is a tuning knob controlling parallelism per program. Triton's autotuner can search it.

Lab `lab20_triton_fused_kernel` builds this kernel, proves numerical equivalence against PyTorch, and benchmarks it — including the sizes where it loses.

## 20.7 When to write CUDA instead

Triton is the default. CUDA still wins when:

- **You need a hardware feature Triton does not expose.** Warp-level primitives, specific tensor-core instructions, the TMA engine, asynchronous copies. FlashAttention-3's Hopper-specific work is in CUDA for this reason.
- **The access pattern is irregular.** MoE dispatch, sparse operations, and anything with data-dependent indexing map poorly onto Triton's tile model.
- **You are in the top 1% of hot loops** and the last 15% of performance is worth weeks of effort. For a kernel running billions of times in a frontier training run, it is.
- **You need explicit control of shared memory layout** to avoid bank conflicts, or precise warp scheduling.

A reasonable escalation: PyTorch → `torch.compile` → Triton → CUDA. Move up only when you have measured that the current level is the bottleneck.

## 20.8 Numerics

Kernels are where numerical bugs live, and they are quiet.

**Accumulate in higher precision than you store.** Summing 8,192 BF16 values in BF16 loses precision badly — BF16 has 7 mantissa bits, so once the accumulator is large, small addends vanish entirely. Accumulate in FP32, store in BF16. Every kernel above does this.

**Softmax needs the max subtracted.** $e^{x}$ overflows for $x > 88$ in FP32 and $x > 11$ in FP16. Subtracting the row maximum before exponentiating is mandatory, and it is exactly what FlashAttention's running $m$ is for.

**Reduction order changes results.** Floating-point addition is not associative, so a parallel tree reduction gives a different answer than a sequential one. Both are "correct"; they differ in the last bits. This means **your kernel will not match PyTorch bit-for-bit**, and testing for exact equality will fail. Test with a tolerance appropriate to the precision — for BF16, relative error around $10^{-2}$ is normal, and anything much larger indicates a real bug.

**It also means kernels break determinism.** If bit-exact reproducibility matters (Chapter 21 §21.7), a nondeterministic reduction order will defeat it, and atomics are the usual culprit.

**Test against a reference, at scale, in the real dtype.** A kernel correct on a $128 \times 128$ FP32 tensor can be wrong on an $8192 \times 8192$ BF16 one, because that is where accumulation error and edge-case masking appear.

## 20.9 Benchmarking honestly

Most reported kernel speedups are wrong. The standard mistakes:

**Not synchronizing.** CUDA kernels launch asynchronously. Timing without `torch.cuda.synchronize()` measures the launch, not the work, and produces spectacular fake numbers.

**Not warming up.** The first call includes JIT compilation, autotuning, and memory allocation. Discard it.

**Measuring one shape.** A kernel tuned for $d = 4096$ may lose at $d = 5120$ where the tile size no longer divides evenly. Report a sweep.

**Ignoring cache effects.** Running the same small tensor in a loop leaves it in L2, which does not happen in a real model. Use realistic sizes, or rotate through several buffers.

**Comparing against the wrong baseline.** Beating a naive PyTorch loop is not interesting. Beat `torch.compile`, and beat the vendor library if one exists.

```python
import torch


def benchmark(fn, *args, warmup=25, iters=100):
    """Time a GPU function correctly.

    Uses CUDA events rather than wall clock, so the measurement is on the GPU
    timeline and does not include Python overhead on the host.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # milliseconds per call
```

And report **achieved bandwidth or achieved FLOP/s**, not just time. For a memory-bound kernel, bytes moved divided by time, compared against the 3.35 TB/s peak, tells you how much headroom remains. "My kernel is 2× faster" is a comparison; "my kernel achieves 82% of peak bandwidth" is a measurement, and it tells you when to stop.

## 20.10 torch.compile, and when hand-writing loses

`torch.compile` traces your model, fuses element-wise operations, and generates Triton kernels automatically. For a large fraction of fusion opportunities it does a good job with zero effort.

**When it wins:** element-wise chains, simple reductions, and anything where the win is "stop round-tripping to HBM between four pointwise ops." This is most of the low-hanging fruit, and hand-writing those kernels is now usually wasted effort.

**When it loses:** operations requiring a genuinely different algorithm. `torch.compile` will not invent FlashAttention — it fuses the operations you wrote, and the online-softmax restructuring is a different computation. It also struggles with data-dependent control flow, and it introduces graph breaks that undo fusion in ways that are hard to predict.

**The practical workflow:** run `torch.compile` first, profile, and hand-write only what remains hot. Most teams skip the profiling step and hand-write kernels the compiler would have handled, which is the most common waste of kernel-engineering time.

## 20.11 What frontier kernel work actually looks like

Not, mostly, writing new attention algorithms.

**Porting to new hardware.** Every GPU generation changes the optimal tiling, the available instructions, and the memory hierarchy sizes. A kernel tuned for A100 leaves substantial performance on an H100. This is steady, unglamorous, high-value work.

**MoE kernels.** The dispatch and combine operations (§6.8) are irregular, data-dependent gather/scatter — exactly what standard libraries handle badly. Grouped GEMM for expert computation, where each expert gets a different number of tokens, is a genuinely hard kernel problem and a major focus at labs training MoE models.

**Quantization kernels.** FP8 and INT4 matmuls with per-block scaling, dequantization fused into the GEMM prologue. DeepSeek-V3's FP8 training (§8.3) depended on kernels like this.

**Communication overlap.** Fusing a collective with the computation that produces or consumes its data, so the network transfer hides behind arithmetic. DualPipe [\[50\]](../appendix/b-references.md#50-dualpipe-deepseek-github) is this idea at the schedule level; there is a kernel-level version too.

**Decoding kernels.** Autoregressive decoding is entirely memory-bound (§20.3), so serving performance is a kernel problem. Paged attention, speculative decoding verification, and fused sampling all live here. Chapter 22.

## 20.12 JD, decoded

**Kernel Engineer / GPU Performance Engineer.** Writes and tunes the kernels. Requires CUDA or Triton, a working model of the memory hierarchy, and — the actually scarce skill — the profiling discipline to know *which* kernel to write. A JD mentioning "Triton," "CUDA," "fused kernels," or "GPU performance optimization" is this role. It is among the highest-paid and most consistently under-supplied roles in the field.

**ML Systems Engineer.** Overlaps heavily. Owns whether the training step is bound by compute, memory, communication, or the data loader, and hands the kernel engineer a specific target.

**Inference Engineer.** Kernel work aimed at decoding. Different constraints entirely — batch size 1 to 256 rather than thousands, latency rather than throughput, and the KV cache dominating memory.

**What to learn if you want this job**, in order: the roofline model and arithmetic intensity; how to read a profiler (Nsight Compute); Triton; then CUDA. Write a fused RMSNorm, benchmark it honestly against `torch.compile`, and understand why you did or did not win. That exercise teaches more than any course, and it is exactly what Lab 20 does.

## 20.13 What you should take from this chapter

1. **A 20% training speedup on a frontier run is worth an engineer-month**, and sometimes the kernel is the difference between possible and impossible rather than fast and slow.
2. **The H100 ridge point is roughly 295 FLOP per byte.** Below that you are memory-bound and the tensor cores are idle.
3. **Normalizations, activations, and residual adds have arithmetic intensity around 1.** They use under 1% of the machine's compute and are pure memory traffic — which is exactly why fusing them pays.
4. **Fusion means touching HBM once.** Load a tile, do all the work in registers and shared memory, write once.
5. **FlashAttention is exact, not approximate.** Online softmax lets you compute attention block by block, taking memory from $O(N^2)$ to $O(N)$ and making long context possible at all.
6. **Its backward pass recomputes the attention matrix** rather than storing it — activation recomputation inside a single operation.
7. **Triton gets 80–95% of CUDA performance for ~20% of the effort.** Escalate PyTorch → `torch.compile` → Triton → CUDA, and only after measuring.
8. **Accumulate in FP32, store in BF16**, and always subtract the max before exponentiating. Reduction order changes the last bits, so test with a tolerance and expect kernels to break bit-exact determinism.
9. **Benchmark with CUDA events, after warmup, across a shape sweep, against `torch.compile`** — and report achieved bandwidth, not just a speedup ratio, so you know when to stop.
10. **Most frontier kernel work is porting, MoE dispatch, quantization, and communication overlap** — not inventing new attention algorithms.

The next chapter is checkpointing and resumption: what happens when a run that has been going for eleven days loses a node at 3am, and how you make that cost twenty minutes instead of eleven days.

---

**Exercises:** [Chapter 20 problem set](../../exercises/ch20.md) — includes arithmetic-intensity calculations for every operation in a transformer layer, a fusion-opportunity audit, and a benchmarking-methodology critique.
**Lab:** [`lab20_triton_fused_kernel`](../../labs/lab20_triton_fused_kernel.py) — write a fused RMSNorm in Triton, prove numerical equivalence, benchmark it honestly against `torch.compile`, and find the shapes where it loses. Falls back to a numerics-only check on CPU.

---

**References for this chapter**

- [\[19\] FlashAttention](../appendix/b-references.md#19-flashattention) — Dao et al., 2022. The IO-aware exact attention algorithm.
- [\[50\] DualPipe](../appendix/b-references.md#50-dualpipe-deepseek-github) — DeepSeek-AI. Communication/computation overlap at the schedule level.
- [\[83\] FlashAttention-2](../appendix/b-references.md#83-flashattention-2) — Dao, July 2023. Better work partitioning; roughly 2× over the original.
- [\[84\] FlashAttention-3](../appendix/b-references.md#84-flashattention-3) — Shah et al., July 2024. Hopper-specific asynchrony and FP8.
- [\[85\] Triton](../appendix/b-references.md#85-triton) — Tillet et al., 2019. The tile-based language and compiler.
- [\[86\] Making Deep Learning Go Brrrr From First Principles](../appendix/b-references.md#86-making-deep-learning-go-brrrr) — Horace He, 2022. The compute/memory/overhead framing.
- [See full reference list](../appendix/b-references.md)
