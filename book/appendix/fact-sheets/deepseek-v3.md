# DeepSeek-V3 fact sheet

Every fact the book relies on about DeepSeek-V3 lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md).

**Sources.** `[V3]` is DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437**v2** (the revision these section numbers refer to). See [reference 1](../b-references.md#1-deepseek-v3).

Checked against the source: 2026-09-26.

## Cluster and interconnect

| Fact | Value | Source |
|---|---|---|
| Training cluster | 2,048 NVIDIA H800 GPUs | [V3] §3.1 |
| Node | 8 GPUs, connected by NVLink and NVSwitch; nodes connected by InfiniBand | [V3] §3.1 |
| NVLink bandwidth (H800, as used) | 160 GB/s | [V3] §3.2.2 |
| InfiniBand bandwidth (as used) | 50 GB/s | [V3] §3.2.2 |
| NVLink : InfiniBand ratio | ~3.2× | [V3] §3.2.2 |
| Node-limited routing | Each token is dispatched to at most 4 nodes, to cap InfiniBand traffic | [V3] §3.2.2, §4.2 |

## Parallelism layout

| Fact | Value | Source |
|---|---|---|
| Pipeline parallelism | 16-way (PP16), scheduled with DualPipe | [V3] §3.2, §3.2.1 |
| Expert parallelism | 64-way (EP64), spanning 8 nodes; each layer's routed experts are spread evenly over 64 GPUs | [V3] §3.2, §4.2 |
| Data parallelism | ZeRO-1 | [V3] §3.2 |
| Tensor parallelism in training | None; memory optimisations make TP unnecessary | [V3] §3.2, §3.2.3 |
| Implied data-parallel degree | 2,048 / 16 = 128 ranks per pipeline stage. This is our arithmetic; the report does not state the DP degree | [V3] §3.1, §3.2 |
| Inference layout (prefill), for contrast | Attention TP4 + SP with DP8; MoE EP32; minimum unit 4 nodes / 32 GPUs | [V3] §3.4.1 |

## Architecture

| Fact | Value | Source |
|---|---|---|
| Transformer layers | 61 | [V3] §4.2 |
| Hidden dimension | 7,168 | [V3] §4.2 |
| Dense layers | The first 3; every other FFN is MoE (so 58 MoE layers) | [V3] §4.2 |
| Attention | MLA, 128 heads, per-head dimension 128 | [V3] §2.1.1, §4.2 |
| MLA KV compression dimension $d_c$ | 512 | [V3] §4.2 |
| MLA query compression dimension $d_c'$ | 1,536 | [V3] §4.2 |
| MLA decoupled RoPE key dimension $d_h^R$ | 64 | [V3] §4.2 |
| Experts per MoE layer | 1 shared + 256 routed | [V3] §4.2 |
| Expert intermediate dimension | 2,048 | [V3] §4.2 |
| Routed experts per token | 8 | [V3] §4.2 |
| MLA values cached per token per layer | 576: the joint KV latent $c^{KV}$ (512) plus the decoupled RoPE key $k^R$ (64), shared across all 128 heads. There is no KV-head grouping | [V3] §2.1.1, §4.2 |
| Uncompressed equivalent, for scale | 2 × 128 heads × 128 = 32,768 values per token per layer, so MLA caches ~1.8% of it (our arithmetic) | [V3] §4.2 |
| Gating | Sigmoid affinity scores; the per-expert balancing bias is added **only to choose the top-8**; gating weights are the original scores, normalized over the 8 selected | [V3] §2.1.2 |
| Balance losses | Auxiliary-loss-free bias balancing, plus a complementary sequence-wise balance loss with a very small weight (α = 0.0001) | [V3] §2.1.2, §4.2 |
| Bias update speed γ | 0.001 for the first 14.3T tokens, 0.0 for the last 500B | [V3] §4.2 |
| Multi-token prediction depth | 1 (one additional token) | [V3] §2.2, §4.2 |
| MTP module structure | Sequential: shared embedding, shared output head, one full Transformer block, and a projection that combines the main model's state at $t$ with the embedding of token $t+1$ | [V3] §2.2 |
| MTP module size | 14B parameters (checkpoint 685B = 671B main model + 14B MTP) | `README.md` @e815299 |
| MTP loss weight λ | 0.3 for the first 10T tokens, then 0.1 for the remaining 4.8T | [V3] §4.2 |
| MTP at inference | Can be discarded; or repurposed for speculative decoding: second-token acceptance 85–90%, 1.8× tokens per second | [V3] §2.2, §5.4.3 |
| MTP origin | "Inspired by Gloeckle et al. (2024)"; V3 did not introduce MTP | [V3] §2.2 |
| Published config | `num_hidden_layers` 61, `first_k_dense_replace` 3, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `num_nextn_predict_layers` 1, dense `intermediate_size` 18,432 | `config.json` @e815299 |
| Parameters | 671B total, 37B activated per token | [V3] §4.2 |

## Precision

| Fact | Value | Source |
|---|---|---|
| FP8 format | E4M3 on all tensors (not the E4M3-forward / E5M2-backward hybrid), made feasible by tile- and block-wise scaling | [V3] §3.3.2 |
| FP8 scaling granularity | Activations: 1×128 tiles (per token per 128 channels). Weights: 128×128 blocks | [V3] §3.3.2 |
| FP8 GEMMs | Fprop, Dgrad and Wgrad of every Linear run in FP8 | [V3] §3.3.1 |
| Kept in higher precision (BF16/FP32) | Embedding, output head, MoE gating, normalization, attention operators | [V3] §3.3.1 |
| FP8 accumulation | Partial sums promoted to FP32 on CUDA cores every $N_C = 128$ elements | [V3] §3.3.2 |
| AdamW moments | BF16 (not FP32), "without incurring observable performance degradation" | [V3] §3.3.3 |
| Master weights and gradients | FP32 | [V3] §3.3.3 |

## Optimizer and schedule

| Fact | Value | Source |
|---|---|---|
| Optimizer | AdamW, β₁ = 0.9, β₂ = 0.95, weight decay 0.1 | [V3] §4.2 |
| Pre-training tokens | 14.8T | [V3] §4.2 |
| Pre-training sequence length | 4K | [V3] §4.2 |
| Learning-rate warmup | Linear from 0 to 2.2×10⁻⁴ over the first 2K steps | [V3] §4.2 |
| Learning-rate schedule | Constant 2.2×10⁻⁴ until 10T tokens; cosine decay to 2.2×10⁻⁵ over the next 4.3T; then constant 2.2×10⁻⁵ for 333B tokens and 7.3×10⁻⁶ for the final 167B (10T + 4.3T + 0.5T = 14.8T) | [V3] §4.2 |
| Batch size | Ramped from 3,072 to 15,360 over the first 469B tokens, then 15,360 | [V3] §4.2 |
| Gradient clipping | Norm 1.0 | [V3] §4.2 |
| Stability | No irrecoverable loss spikes and no rollbacks in the whole run | [V3] §1 |

## Context extension, post-training and cost

| Fact | Value | Source |
|---|---|---|
| Context extension | Two stages: 4K → 32K → 128K | [V3] §1, §4.3 |
| Post-training RL algorithm | GRPO | [V3] §5.2.2 |
| Distillation direction | Reasoning capability distilled **from** DeepSeek-R1-series models **into** V3 | [V3] §1, §5.4.1 |
| GPU-hours: pre-training | 2,664K H800 GPU-hours | [V3] Table 1 |
| GPU-hours: context extension | 119K | [V3] Table 1 |
| GPU-hours: post-training | 5K | [V3] Table 1 |
| GPU-hours: total | 2,788K, covering all three stages, at an assumed $2/GPU-hour = $5.576M; excludes prior research and ablations | [V3] Table 1, §1 |
