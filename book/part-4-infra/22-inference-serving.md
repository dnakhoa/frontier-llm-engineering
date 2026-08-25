# Chapter 22: Inference and serving

> Reading time: ~40 minutes. By the end of this chapter you should be able to compute a KV cache's size and see immediately why GQA and MLA exist, explain why prefill and decode are opposite workloads, describe what continuous batching and paged attention each fix, and reason about the latency-throughput trade-off that every serving decision sits on. If you are here from Part III: this is the chapter that explains where 60–80% of your RLHF wall-clock went.

## 22.1 A different problem entirely

Training and serving look like the same computation and they are not.

Training is a batch job. You control the input, the batch size, and the schedule; you optimize throughput; a 5% slowdown costs 5% more money. Failures are expensive but tolerable (Chapter 21).

Serving is an online system. Requests arrive when they arrive, with wildly varying prompt and output lengths. You optimize a *trade-off* between latency and throughput. A 200ms regression is a product problem, and every request is a live one.

The transition surprises people, and it surprises post-training teams in particular — Chapter 13 §13.9 established that a production RLHF system is an inference system with a trainer attached, and teams that treat rollout generation as "training with `model.generate()`" leave factors of 5–20× on the floor.

## 22.2 Prefill and decode are opposite workloads

The single most important structural fact about LLM inference.

**Prefill.** Process the whole prompt at once. Every token is computed in parallel, exactly like a training forward pass. Large matrix multiplies, high arithmetic intensity, **compute-bound** (§20.3).

**Decode.** Generate one token. Load the entire model's weights from HBM to produce a single token per sequence. Almost no arithmetic per byte loaded, **memory-bandwidth-bound**, and severely so.

Put numbers on it. A 70B model in BF16 is 140 GB of weights. On an H100 at 3.35 TB/s, reading all the weights once takes about 42 ms. That is the floor on time-per-token at batch size 1 — regardless of how fast the tensor cores are, because they are idle.

$$\text{arithmetic intensity of decode at batch } B \approx 2B \text{ FLOP/byte}$$

Against the ridge point of ~295 (§20.2), you need a batch of roughly 150 sequences before decode becomes compute-bound. Below that, you are paying for memory bandwidth and getting the compute for free.

**Everything in serving follows from this.** Batching is not an optimization; it is *the* optimization, because it amortizes one weight read across many sequences. Quantization helps decode because it shrinks the bytes read, not because it speeds up arithmetic. Speculative decoding works because it converts sequential decode steps into a parallel verification step.

## 22.3 The KV cache

Without a cache, generating token $n$ would recompute the keys and values for all $n-1$ preceding tokens. The cache stores them.

$$\text{bytes per token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per element}$$

The 2 is for K and V.

**Llama-3-70B**: 80 layers, 8 KV heads (GQA), head dimension 128, BF16.

$$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} = 320 \text{ KB per token}$$

At 8,192 tokens that is **2.6 GB per sequence**. On an 80 GB H100 holding 140 GB of weights across two GPUs, the cache is what limits your concurrency, not the weights.

**The same model with plain multi-head attention** (64 KV heads instead of 8):

$$2 \times 80 \times 64 \times 128 \times 2 = 2.6 \text{ MB per token} \;\Rightarrow\; 21 \text{ GB per sequence at 8K}$$

Eight times larger. **This is why GQA exists** [\[20\]](../appendix/b-references.md#20-gqa) — it is a serving decision made at architecture time, and Chapter 5 covers the quality trade-off. Three sequences of MHA cache fill an entire H100.

**DeepSeek's MLA** [\[2\]](../appendix/b-references.md#2-deepseek-v2) goes further: instead of storing K and V per head, store a low-rank latent vector per token and reconstruct at attention time. DeepSeek-V2 reports a KV cache reduction of roughly 93% against its MHA baseline. For V3's configuration the cache is on the order of tens of kilobytes per token rather than hundreds — an order of magnitude below GQA.

The lesson worth carrying: **attention-variant choices are serving-economics choices.** A team that picks MHA for a model it intends to serve at long context has made a decision that will cost far more than the quality difference is worth.

## 22.4 Batching

**Static batching** is the naive approach: collect $B$ requests, run them together, return when all finish.

The problem is that outputs have very different lengths. If one request generates 2,000 tokens and seven generate 50, the seven finish and their slots sit idle for the remaining 1,950 steps. Utilization collapses.

```
Static batching:
  req A ████████████████████████████████  2000 tokens
  req B ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  50 tokens, then 1950 steps of waste
  req C ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  req D ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

**Continuous batching** — introduced as iteration-level scheduling in Orca [\[90\]](../appendix/b-references.md#90-orca) — makes the scheduling decision at every *token*, not every batch. When a sequence finishes, a waiting request takes its slot immediately.

```
Continuous batching:
  slot 0 ████████████████████████████████  req A
  slot 1 ██▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░░░████████████  B, then E, then F, then I
  slot 2 ██▓▓▓▓▒▒▒▒▒▒████████████▓▓▓▓▓▓▓▓  C, then G, then J, ...
  slot 3 ███▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒████████░░░░░  D, then H, ...
```

This is typically a **2–4× throughput improvement** on realistic traffic and it is the single largest win in serving. Every modern engine does it; `model.generate()` does not.

## 22.5 PagedAttention

Continuous batching creates a memory problem. Each sequence's KV cache grows as it generates, and you do not know the final length in advance.

The naive approach reserves the maximum: allocate a contiguous 2.6 GB buffer for every slot in case the sequence runs to 8K. A request generating 200 tokens uses 2.5% of its reservation. Measured internal fragmentation in pre-vLLM systems was substantial — most of the KV memory was reserved and unused.

**PagedAttention** [\[23\]](../appendix/b-references.md#23-vllm) applies virtual memory's central idea. Split the KV cache into fixed-size blocks (typically 16 tokens). A sequence gets a *block table* mapping logical positions to physical blocks, which need not be contiguous. Allocate a new block only when the current one fills.

```
Sequence A (37 tokens):  block table -> [ 4, 11, 2 ]
Sequence B (20 tokens):  block table -> [ 7, 9 ]

Physical KV blocks (16 tokens each), non-contiguous:
  [0][1][2*][3][4*][5][6][7*][8][9*][10][11*][12]...
                                        ^ free blocks reusable immediately
```

What this buys:

- **Near-zero fragmentation.** Waste is bounded by one partial block per sequence.
- **Copy-on-write sharing.** Several sequences sampled from the same prompt share the prompt's physical blocks until they diverge. For best-of-$n$ and GRPO group sampling (§15.9) this is a large saving — $G = 32$ samples of one prompt share one copy of the prompt cache.
- **Preemption.** A sequence can be evicted and its blocks reclaimed, then recomputed or swapped back later.

The copy-on-write property is why this chapter is on the post-training learning path. A GRPO rollout batch is *exactly* the shared-prefix workload paged attention is best at.

**RadixAttention** [\[91\]](../appendix/b-references.md#91-sglang) generalizes the sharing: keep a radix tree of cached prefixes across *requests*, so a common system prompt is computed once for all users. On workloads with a fixed system prompt — most production chat — this eliminates a large fraction of prefill.

## 22.6 Quantization

Decode is memory-bound (§22.2), so halving the bytes read roughly halves decode time. Quantization is therefore a *latency* optimization for decode, which is the opposite of the intuition that it is a memory-capacity optimization.

| Format | Bits | Typical use | Quality cost |
|---|---|---|---|
| BF16 | 16 | Baseline | — |
| FP8 (E4M3) | 8 | Weights and activations on Hopper+ | Very small with per-block scaling |
| INT8 | 8 | Weights, sometimes activations | Small with good calibration |
| INT4 (AWQ, GPTQ) | 4 | Weights only | Noticeable; depends heavily on method |

Three things practitioners get wrong:

**Weight-only quantization does not speed up prefill.** Prefill is compute-bound, and if you dequantize to BF16 to do the matmul, you have added work. Weight-only INT4 can make prefill *slower* while making decode much faster.

**The KV cache can be quantized too**, and at long context it dominates memory. FP8 KV cache roughly doubles your concurrency. Quality impact is small in practice; it is under-used.

**Per-tensor scaling is not enough at low precision.** A single scale factor for a whole tensor cannot cover its dynamic range. Per-channel or per-block scaling is what makes FP8 and INT8 work, and it is the same lesson as FP8 *training* (§8.3).

## 22.7 Speculative decoding

The cleverest idea in serving, and it follows directly from §22.2.

Decode at batch size 1 wastes almost all the compute — you read 140 GB of weights to do a trivial amount of arithmetic. So: use a small, fast draft model to guess the next $k$ tokens, then verify all $k$ **in a single forward pass of the large model**, which costs the same as generating one token.

```python
def speculative_step(target, draft, tokens, k=4):
    """Draft k tokens cheaply, verify them in one expensive pass.

    Verification is exact: the accepted tokens are distributed identically to
    what the target model would have produced on its own. This is not an
    approximation -- it is a scheduling trick.
    """
    # 1. Cheap sequential drafting with the small model.
    drafted = draft.generate(tokens, max_new_tokens=k)          # k fast steps

    # 2. One expensive parallel pass scores all k positions at once.
    target_logprobs = target.forward(tokens + drafted)          # 1 slow step

    # 3. Accept the longest prefix consistent with the target's distribution;
    #    resample the first rejected position from the corrected distribution.
    return rejection_sample(drafted, target_logprobs)
```

**The key property: it is exact.** The rejection-sampling scheme of Leviathan et al. [\[89\]](../appendix/b-references.md#89-speculative-decoding) guarantees the output distribution is identical to sampling from the target model directly. You are not trading quality for speed; you are trading *wasted compute* for speed.

Typical speedups are 2–3× at low batch size. Variants replace the separate draft model with extra prediction heads on the target (Medusa), with n-gram lookup from the prompt, or — as in DeepSeek-V3 — with a multi-token-prediction module trained alongside the main model.

**When it stops helping:** at large batch sizes. Once you are compute-bound (batch above ~150), there is no wasted compute to reclaim and the draft model's cost is pure overhead. Speculative decoding is a *low-batch, latency-sensitive* technique — exactly the regime of a single interactive user, and exactly not the regime of bulk rollout generation.

## 22.8 Parallelism for inference

Different constraints from training (Chapter 6).

**Tensor parallelism** is the workhorse. Splits weights across GPUs, so each reads fewer bytes — which directly reduces decode latency, since decode is bandwidth-bound. Costs an all-reduce per layer, so it wants NVLink and effectively confines you within a node.

**Pipeline parallelism** works for throughput and hurts latency: a request traverses every stage sequentially. Use it when the model does not fit within one NVLink domain, not to reduce latency.

**Expert parallelism** for MoE, and it is genuinely hard at serving time. The routing is data-dependent, so load across experts varies per batch, and the all-to-all sits on the critical path of every token. Serving a large MoE well is one of the harder open problems in the field.

**No data parallelism inside a replica** — that is just running more replicas.

A useful rule: **use the smallest tensor-parallel degree that fits the model and the KV cache**, because every additional rank adds communication to every token. TP=8 across a node when TP=4 would fit is a latency regression you are paying for nothing.

## 22.9 The metrics, and the trade-off

Four numbers, and they conflict.

| Metric | What it is | Who cares |
|---|---|---|
| **TTFT** — time to first token | Prompt arrival → first token out | Interactive users. Dominated by prefill. |
| **TPOT** — time per output token | Steady-state inter-token latency | Perceived speed. Dominated by memory bandwidth. |
| **Throughput** | Total tokens/sec across all requests | Cost per token. What the bill depends on. |
| **Goodput** | Throughput *within* an SLO | The only one that means anything in production |

**The central trade-off:** larger batches raise throughput and raise TPOT. Each token now waits for a bigger matmul, and more sequences contend for bandwidth. You cannot maximize both.

```
throughput
    ^
    |                    ..........  (compute-bound plateau)
    |              .....
    |         ....
    |      ...
    |   ..
    | .
    +----------------------------------> batch size
      ^                    ^
   low latency,      high throughput,
   poor utilization   bad TPOT
```

**Goodput is the metric that resolves it.** "Throughput subject to p95 TTFT under 500 ms and TPOT under 50 ms" is a number you can optimize; raw throughput is a number you can game by making every user wait.

A second trade-off worth naming: **prefill and decode compete.** A long prompt arriving mid-batch stalls every decoding sequence while it is processed. **Chunked prefill** — splitting a long prefill into pieces interleaved with decode steps — bounds that stall, at some cost to TTFT for the prefilling request.

## 22.10 Serving reasoning models

Chapter 15's models break several serving assumptions, and this is now a first-order concern.

**Outputs are very long.** A reasoning trace can be tens of thousands of tokens. KV cache per sequence grows accordingly, concurrency drops, and a single request can occupy a slot for minutes.

**Output length is unpredictable.** The model decides how long to think. You cannot size a batch on expected output length because the variance is enormous — which makes continuous batching and preemption more valuable, not less.

**Test-time compute changes the unit of work.** Best-of-$n$ (§16.6) means one logical request is $n$ generations sharing a prefix. Paged attention's copy-on-write is exactly the right mechanism, and an engine without prefix sharing serves this workload badly.

**Most tokens are never shown.** The reasoning trace is usually hidden. You are paying full price for tokens the user never sees, which shifts the economics: throughput per *useful* token, not per token.

**Disaggregated prefill and decode** has become the standard structural response. Run prefill on one pool of GPUs and decode on another, transferring the KV cache between them. The two workloads have opposite resource profiles (§22.2), so giving each its own pool lets you size and tune them independently — different batch sizes, different parallelism, different hardware. The cost is a KV-cache transfer over the network, which is why it needs fast interconnect to pay off.

## 22.11 What goes wrong

| Symptom | Cause |
|---|---|
| Throughput far below expectation | Static batching, or `model.generate()` in a loop |
| Out-of-memory at moderate concurrency | KV cache sized by max length, no paging |
| TTFT spikes under load | Long prefills blocking decode; no chunked prefill |
| Quantization made prefill slower | Weight-only quant on a compute-bound phase |
| Speculative decoding made things slower | Large batch — already compute-bound |
| Quality differs from training | Chat template mismatch (§11.6) |
| p99 latency terrible, p50 fine | No admission control; long requests starving short ones |
| MoE throughput collapses on some batches | Expert load imbalance in the all-to-all |

The first row is the one that matters for readers of Part III. If your GRPO rollouts run at a few thousand tokens per second on a node that should do tens of thousands, this is why.

## 22.12 JD, decoded

**Inference / Serving Engineer.** Owns the engine, the scheduler, the KV cache manager, and the latency-throughput trade-off. Requires systems engineering plus enough kernel knowledge (Chapter 20) to know why decode is slow. A JD mentioning "vLLM," "SGLang," "TensorRT-LLM," "continuous batching," or "KV cache" is this role.

**Performance / Kernel Engineer (inference).** Decode kernels specifically: paged attention, fused sampling, quantized matmuls, speculative verification. Overlaps heavily with Chapter 20 and has different constraints — small batch, latency, memory-bound.

**ML Systems Engineer (post-training).** The reason serving appears in post-training postings. Building the rollout-generation infrastructure in §13.10 is a serving problem, and the people who know both are scarce.

**Capacity / Reliability Engineer.** Autoscaling on a workload where request cost varies by three orders of magnitude, admission control, and defending p99.

## 22.13 What you should take from this chapter

1. **Prefill is compute-bound; decode is memory-bandwidth-bound.** Nearly every serving technique follows from that one asymmetry.
2. **At batch size 1, time-per-token is bounded below by the time to read the weights** — about 42 ms for a 70B model in BF16 on one H100, with the tensor cores idle.
3. **You need a batch of roughly 150 sequences before decode is compute-bound.** Batching is not an optimization, it is *the* optimization.
4. **KV cache per token is $2 \times L \times H_{kv} \times d_{head} \times$ bytes.** For Llama-3-70B that is 320 KB/token, or 2.6 GB for an 8K sequence — the cache, not the weights, is what limits concurrency.
5. **GQA and MLA are serving decisions made at architecture time.** MHA at long context costs 8× the cache of GQA; MLA cuts it by another order of magnitude.
6. **Continuous batching schedules per token, not per batch**, and is worth 2–4× on realistic traffic. `model.generate()` gives you none of it.
7. **PagedAttention removes fragmentation and enables prefix sharing.** For best-of-$n$ and GRPO group sampling, copy-on-write on the shared prompt is a large win.
8. **Speculative decoding is exact**, not an approximation — and it stops helping once you are compute-bound at large batch.
9. **Optimize goodput, not throughput.** Throughput without an SLO is a number you can improve by making users wait.
10. **Reasoning models break the assumptions**: very long, unpredictable outputs, most tokens never shown, and test-time compute turning one request into $n$ prefix-sharing generations. Disaggregated prefill and decode is the standard structural answer.

The next chapter is evaluation — how you know any of this worked, why most reported benchmark numbers are less meaningful than they look, and how contamination quietly invalidates the comparisons everyone is making.

---

**Exercises:** [Chapter 22 problem set](../../exercises/ch22.md) — includes KV-cache sizing across attention variants, a concurrency budget, and a batching-policy design problem under an SLO.
**Lab:** [`lab22_kv_cache_and_batching`](../../labs/lab22_kv_cache_and_batching.py) — implement a KV cache, compare static against continuous batching on a realistic length distribution, and build a paged block allocator with prefix sharing.

---

**References for this chapter**

- [\[2\] DeepSeek-V2](../appendix/b-references.md#2-deepseek-v2) — DeepSeek-AI, 2024. Multi-head latent attention and the KV cache reduction it buys.
- [\[20\] GQA](../appendix/b-references.md#20-gqa) — Ainslie et al., 2023. Grouped-query attention.
- [\[23\] vLLM / PagedAttention](../appendix/b-references.md#23-vllm) — Kwon et al., 2023. Paged KV cache and copy-on-write prefix sharing.
- [\[89\] Speculative decoding](../appendix/b-references.md#89-speculative-decoding) — Leviathan et al., 2022. Exact-distribution draft-and-verify.
- [\[90\] Orca](../appendix/b-references.md#90-orca) — Yu et al., OSDI 2022. Iteration-level scheduling, i.e. continuous batching.
- [\[91\] SGLang / RadixAttention](../appendix/b-references.md#91-sglang) — Zheng et al., 2023. Cross-request prefix caching in a radix tree.
- [See full reference list](../appendix/b-references.md)
