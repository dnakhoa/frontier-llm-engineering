# Solution — prerequisites self-check

The problem, from [the prerequisites page](../../book/prerequisites.md):

> A 7B-parameter dense transformer is trained in BF16 with the AdamW optimizer, keeping FP32 master weights.
>
> 1. How many bytes per parameter are held for the weights, the gradients, and the optimizer states combined?
> 2. What is the total in gigabytes for the 7B model, before any activations?
> 3. Will that fit on one 80 GB H100? What is the first thing you would do if it did not?

---

## 1. Bytes per parameter

Count every tensor that exists per parameter in the standard mixed-precision AdamW setup:

| Tensor | Precision | Bytes |
|---|---|---|
| BF16 weights (used in forward/backward) | BF16 | 2 |
| BF16 gradients | BF16 | 2 |
| FP32 master weights | FP32 | 4 |
| AdamW first moment, $m$ | FP32 | 4 |
| AdamW second moment, $v$ | FP32 | 4 |
| **Total** | | **16** |

**16 bytes per parameter.**

This number is worth memorizing. It is the single most-used figure in pre-training capacity planning, and it appears again in [Chapter 6 §6.1](../../book/part-2-pretraining/06-distributed-training.md).

Two things people get wrong here:

- **Forgetting the FP32 master weights.** Mixed-precision training keeps a full-precision copy of the weights, because applying a small update to a BF16 value repeatedly loses the update to rounding — BF16 has only 7 mantissa bits. The optimizer step happens in FP32 and the result is cast down.
- **Counting gradients in FP32.** Gradients are produced by the backward pass in the compute precision, which is BF16. They are cast to FP32 only inside the optimizer step, transiently.

Some implementations differ. Keeping gradients in FP32 gives 18 bytes/param; some setups skip the separate BF16 weight copy and cast on the fly, giving 14. If you answered 14 or 18 with the right reasoning, you understand the mechanism, which is the point.

## 2. Total for 7B

$$7 \times 10^9 \text{ params} \times 16 \text{ bytes/param} = 112 \times 10^9 \text{ bytes}$$

In the decimal convention that GPU vendors use, that is **112 GB**.

In binary units it is $112 \times 10^9 / 2^{30} \approx 104$ GiB. The distinction matters when you are within a few percent of a memory limit, and here we are not — so 112 GB is the number to carry.

Note what this total excludes: **activations**. Those depend on batch size, sequence length, and how much recomputation you do, and for a real training configuration they are frequently comparable to the static memory or larger. See [Chapter 6 §6.12](../../book/part-2-pretraining/06-distributed-training.md).

## 3. Does it fit on one 80 GB H100?

**No.** 112 GB of static state against 80 GB of HBM, and that is before a single activation, before the CUDA context, and before the communication buffers.

This is the result that motivates all of Chapter 6: **a 7B model — small by frontier standards — already does not fit on the largest GPU you can buy.** Every parallelism strategy in the book exists because of this arithmetic.

### What you would do first

The honest ranked answer, roughly in the order a practitioner reaches for them:

1. **ZeRO / FSDP sharding of the optimizer state.** The 12 bytes of FP32 master weights and Adam moments are 75% of the total, they are only touched during the optimizer step, and they shard cleanly across data-parallel ranks with no change to the model code. ZeRO stage 1 shards the optimizer state; stage 2 adds gradients; stage 3 adds parameters. On 8 GPUs, stage 1 alone takes the per-GPU static cost from 112 GB to $7\times10^9 \times (2 + 2 + 12/8) = 38.5$ GB. This is the standard first move and it is why FSDP is the default for models in this size class. See [§6.3–6.4](../../book/part-2-pretraining/06-distributed-training.md).

2. **Activation recomputation**, if activations rather than static state are what pushes you over. Trades roughly 30% more compute for a large reduction in activation memory. Selective recomputation (Korthikanti et al. [\[39\]](../../book/appendix/b-references.md#39-korthikanti-2022-activation-recomputation)) gets most of the saving for about a third of the cost. See [§6.12](../../book/part-2-pretraining/06-distributed-training.md).

3. **Tensor parallelism**, if you need the *parameters* themselves split — for example because a single layer's weights do not fit. Costs an all-reduce per layer, so it wants NVLink and effectively confines you to within a node. See [§6.5](../../book/part-2-pretraining/06-distributed-training.md).

4. **Pipeline parallelism**, if you have more layers than one node can hold. Cheap on communication, expensive in scheduling complexity and pipeline bubbles. See [§6.6](../../book/part-2-pretraining/06-distributed-training.md).

An answer of "use more GPUs" is not wrong, but it is incomplete — *how* you use them is the entire subject of Chapter 6, and the four options above are not interchangeable.

---

## Scoring yourself

- **Got all three, including the 16 bytes/param breakdown:** you are ready. Start at [Chapter 1](../../book/part-1-frontier/01-what-a-frontier-run-looks-like.md).
- **Got the structure but fumbled a factor:** you are ready. The book re-derives this in Chapter 6.
- **Did not know the optimizer kept multiple states per parameter:** read "Transformer Math 101" (EleutherAI) before Chapter 6. An afternoon well spent.
- **The question felt unanswerable:** work through [the prerequisites list](../../book/prerequisites.md) first. Nothing here is hard, but it does not come for free either.
