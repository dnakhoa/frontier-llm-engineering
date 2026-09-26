# DeepSeek-V4 fact sheet

Every fact the book relies on about DeepSeek-V4 lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md). V4 builds on V3 and says that "all other unspecified details follow the settings established in DeepSeek-V3", so read this sheet alongside the [DeepSeek-V3 fact sheet](deepseek-v3.md).

**Sources.** `[V4]` is DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348**v1**, submitted 2026-04-26 according to arXiv's submission history. The pairing is odd: an arXiv id beginning `2606` normally means a June 2026 submission, yet arXiv's own record gives 26 April 2026. We cite arXiv's record as it stands. The report describes **preview** versions of both models. Section, table and figure numbers refer to v1, the only version.

Checked against the source: 2026-09-26.

## Models and context

| Fact | Value | Source |
|---|---|---|
| The two models | DeepSeek-V4-Pro, 1.6T total parameters with 49B activated; DeepSeek-V4-Flash, 284B total with 13B activated. Both are "preview versions" | [V4] §1, §4.2.1 |
| Context length | One million tokens, supported natively after pre-training | [V4] §1 |
| How long context was reached | The training sequence length grows inside pre-training, 4K → 16K → 64K → 1M. The report does not mention YaRN or any other RoPE-scaling method | [V4] §4.2.2 |
| Efficiency at 1M tokens vs V3.2 | V4-Pro: 27% of V3.2's single-token FLOPs (in equivalent FP8 FLOPs) and 10% of its KV cache. V4-Flash: 10% of the FLOPs and 7% of the KV cache | [V4] §1, Figure 1 |
| KV cache vs a GQA baseline | About 2% of a BF16 GQA8 baseline with head dimension 128, at 1M context | [V4] §2.3.4 |
| What stayed from V3 | The DeepSeekMoE framework "with only minor adjustments", and MTP with a configuration "identical to that of DeepSeek-V3" | [V4] §2, §2.1 |

## Architecture

| Fact | Value | Source |
|---|---|---|
| Layers and hidden size | Flash: 43 layers, d = 4096. Pro: 61 layers, d = 7168 | [V4] §4.2.1 |
| Attention layout | CSA and HCA interleaved. The first two layers are pure sliding-window attention in Flash and HCA in Pro | [V4] §4.2.1 |
| CSA (Compressed Sparse Attention) | Compresses the KV cache of every m tokens into one entry, then applies DeepSeek Sparse Attention: a lightning indexer picks the top-k compressed entries for each query | [V4] §2.3, §2.3.1 |
| HCA (Heavily Compressed Attention) | Compresses every m′ (≫ m) tokens into one entry, without overlap, and keeps dense attention over the compressed entries | [V4] §2.3, §2.3.2 |
| CSA and HCA settings | CSA m = 4, 64 indexer query heads, indexer head dimension 128; top-k 512 (Flash) and 1024 (Pro). HCA m′ = 128 | [V4] §4.2.1 |
| Attention heads | 64 query heads (Flash) and 128 (Pro), head dimension 512. Core attention is MQA-style: each compressed KV entry is both key and value | [V4] §2.3.1, §4.2.1 |
| Sliding-window branch | Both CSA and HCA also attend to n_win = 128 uncompressed recent tokens, because a query cannot see tokens inside its own compressed block | [V4] §2.3.3, §4.2.1 |
| Normalisation and position | RMSNorm on each query head and on the KV entry before core attention. RoPE on the last 64 dimensions only, also applied with position −i to the attention outputs. Attention sinks with learnable sink logits | [V4] §2.3.3 |
| MoE | 1 shared expert plus 256 routed experts of intermediate size 2048 (Flash), or 384 of size 3072 (Pro); 6 routed experts active per token in both | [V4] §4.2.1 |
| MoE changes from V3 | Affinity activation changed from Sigmoid to Sqrt(Softplus). The first 3 MoE layers use hash routing on the token ID, replacing V3's dense FFN layers. The cap on routing target nodes is removed | [V4] §2.1, §4.2.1 |
| Load balancing | Auxiliary-loss-free (bias update speed 0.001) plus a sequence-wise balance loss of weight 0.0001 | [V4] §2.1, §4.2.2 |
| MTP | Depth 1. Loss weight 0.3 for most of training, 0.1 once learning-rate decay starts | [V4] §4.2.1, §4.2.2 |
| mHC (Manifold-Constrained Hyper-Connections) | Replaces the plain residual. The residual stream widens by a factor n_hc, and the residual mapping is projected onto doubly stochastic matrices with the Sinkhorn-Knopp algorithm | [V4] §2.2 |
| mHC settings | n_hc = 4; t_max = 20 Sinkhorn-Knopp iterations | [V4] §4.2.1 |

## Optimizer

| Fact | Value | Source |
|---|---|---|
| Which parameters use Muon | Most modules. AdamW is kept for the embedding, the prediction head and all RMSNorm weights; §2.4 also lists mHC's static biases and gating factors | [V4] §2.4, §4.2.2 |
| Muon settings | Momentum 0.95, weight decay 0.1, Nesterov trick, update RMS rescaled to 0.18 so the AdamW learning rate can be reused | [V4] §2.4, §4.2.2 |
| Orthogonalisation | Hybrid Newton-Schulz: 10 iterations, 8 with coefficients (3.4445, −4.7750, 2.0315), then 2 with (2, −1.5, 0.5) | [V4] §2.4 |
| QK-Clip | Not used. RMSNorm on queries and KV entries already keeps attention logits from exploding | [V4] §2.4 |
| AdamW settings | β₁ = 0.9, β₂ = 0.95, ε = 10⁻²⁰, weight decay 0.1 | [V4] §4.2.2 |

## Pre-training

| Fact | Value | Source |
|---|---|---|
| Corpus | "More than 32T tokens": maths, code, web pages, long documents and other categories. The report gives no mixture proportions | [V4] §4.1 |
| Tokens trained | Flash 32T; Pro 33T | [V4] §1, §4.2.2 |
| Batch size | Ramped up to 75.5M tokens (Flash) or a maximum of 94.4M tokens (Pro) | [V4] §4.2.2 |
| Learning-rate schedule | Linear warmup over 2,000 steps, held at the peak "for most of the training", then cosine decay near the end. Peak/end: 2.7×10⁻⁴ / 2.7×10⁻⁵ (Flash) and 2.0×10⁻⁴ / 2.0×10⁻⁵ (Pro) | [V4] §4.2.2 |
| When sparse attention starts | Flash trains with dense attention for the first 1T tokens and switches to sparse attention at sequence length 64K, after a short lightning-indexer warmup. Pro starts with a longer dense stage | [V4] §4.2.2 |
| Packing and masking | Documents from different sources are packed to minimise truncation, with sample-level attention masking. This differs from V3 | [V4] §4.1 |
| Instability | "Notable instability challenges": loss spikes tied to outliers in the MoE layers. Rollbacks alone did not stop them from recurring | [V4] §4.2.3 |
| Anticipatory routing | Routing indices at step t are computed with the parameters from step t−Δt. About 20% extra wall-clock time while active, and it switches on automatically after a loss spike, following a short rollback | [V4] §4.2.3 |
| SwiGLU clamping | Linear component clamped to [−10, 10]; gate component capped at 10 | [V4] §4.2.3 |
| Compute and cost | **Not published.** The report gives no GPU count, GPU-hours or dollar cost. It says only that the fine-grained EP scheme was validated on both NVIDIA GPUs and Huawei Ascend NPUs | [V4] §3.1 |

## Precision

| Fact | Value | Source |
|---|---|---|
| Routed expert weights | FP4. FP4 × FP8 has the same peak FLOPs as FP8 × FP8 on current hardware; the report says future hardware could make it about 1/3 more efficient | [V4] §1 |
| FP4 quantization-aware training | Used in **post-training**: MXFP4 for MoE expert weights and for the indexer's QK path | [V4] §1, §5.2.1 |
| How the QAT works | FP32 master weights are quantized to FP4, then dequantized to FP8 (E4M3) for compute. The report calls this dequantization lossless when the scale-factor ratio stays within a threshold, so the existing FP8 training framework is reused unchanged. Gradients flow to the FP32 master weights, which is equivalent to a straight-through estimator | [V4] §5.2.1 |
| FP4 at inference | Rollout and inference use native FP4 weights instead of simulated quantization | [V4] §5.2.1 |
| KV-cache storage | RoPE dimensions in BF16 and the rest in FP8, which almost halves the cache compared with pure BF16 | [V4] §2.3.4 |
| Indexer attention | Computed in FP4 | [V4] §2.3.4 |

## Determinism and kernels

| Fact | Value | Source |
|---|---|---|
| Goal | "End-to-end, bitwise batch-invariant, and deterministic kernels", so pre-training, post-training and inference stay bitwise aligned | [V4] §3.3 |
| Batch-invariant attention | Split-KV is not allowed. A dual-kernel decode, one SM per sequence plus a multi-SM kernel with the same accumulation order, recovers the throughput lost to wave quantization | [V4] §3.3 |
| Batch-invariant matmul | cuBLAS is replaced end to end with DeepGEMM, and split-k is dropped in most cases | [V4] §3.3 |
| Deterministic backward | Per-SM accumulation buffers plus a deterministic global sum for sparse-attention backward. Token-order pre-processing and buffer isolation for MoE backward. A deterministic split-k reduction for mHC's small matmul, whose output dimension is 24 | [V4] §3.3 |

## Tokenizer

| Fact | Value | Source |
|---|---|---|
| Tokenizer | The DeepSeek-V3 tokenizer plus "a few special tokens for context construction"; vocabulary still 128K. Token-splitting and Fill-in-Middle are inherited from V3 | [V4] §4.1 |
| Tool-call format | A new XML-style schema built around a special `\|DSML\|` token, which the report says reduces escaping failures and tool-call errors | [V4] §5.1.1, Table 4 |

## Post-training

| Fact | Value | Source |
|---|---|---|
| Pipeline shape | Two stages. First, a separate specialist is trained for each domain (for example maths, coding, agent and instruction following) by SFT and then GRPO. Second, the specialists are merged into one model by on-policy distillation | [V4] §1, §5.1 |
| What was replaced | The pipeline "largely mirrored" V3.2's, but V3.2's mixed RL stage "was entirely replaced by On-Policy Distillation" | [V4] §5.1 |
| RL algorithm | GRPO, with hyperparameters "closely aligned with our prior research". The values are not restated | [V4] §5.1.1 |
| Reasoning modes | Three modes, Non-think, Think High and Think Max, each trained with its own length penalties and context windows. Think Max adds an injected system-prompt instruction | [V4] §5.1.1, Table 2, Table 3 |
| Reward models | Scalar reward models are dropped. Hard-to-verify tasks use rubric-guided RL data judged by a generative reward model (GRM), and RL is applied to the GRM itself: the actor also acts as the judge | [V4] §5.1.1 |
| On-policy distillation | Multi-teacher, reverse KL, trained on the student's own trajectories, with more than ten teacher models. It uses full-vocabulary logit distillation instead of a per-token KL estimate, which the report says had high gradient variance | [V4] §5.1.2 |
| What is not published | RL prompt counts, SFT data sizes and post-training compute. §5 gives none of them; the report's only scale figure here is that a sandbox cluster manages "hundreds of thousands of concurrent sandbox instances" | [V4] §5.1, §5.2.5 |
