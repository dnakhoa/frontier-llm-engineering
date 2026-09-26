# Qwen3 fact sheet

Every fact the book relies on about Qwen3 lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md). Tokenizer facts are on the [tokenizer fact sheet](tokenizers.md#qwen3).

**Sources.**
- `[Qwen3]` is the Qwen Team, *Qwen3 Technical Report*, arXiv:2505.09388**v1** (the revision these section numbers refer to). See [reference 6](../b-references.md#6-qwen3).
- `config.json` is the published file in `Qwen/Qwen3-235B-A22B` @8efa617 (commit `8efa61729e24bd65b1d152b5ab5409052aa80e65`), the same commit the tokenizer sheet uses.

Checked against the sources: 2026-09-26.

## Pre-training data

| Fact | Value | Source |
|---|---|---|
| Pre-training tokens | About 36 trillion. This is the pre-training corpus, used across the three pre-training stages below; it does not include post-training data | [Qwen3] §1, §3.1, §3.2 |
| Languages | 119 languages and dialects, up from 29 in Qwen2.5 | [Qwen3] §1, §3.1 |
| Language split | Not published. The report gives no English/Chinese (or any other) ratio | [Qwen3] §3.1 |
| Data annotation | A multilingual annotation system labels over 30T tokens by educational value, field, domain and safety; the mixture is optimized at the instance level using these labels | [Qwen3] §3.1 |
| PDF text | Qwen2.5-VL is fine-tuned to extract text from PDF-like documents, then Qwen2.5 refines it; "trillions" of tokens in total | [Qwen3] §1, §3.1 |
| Synthetic data | Qwen2.5, Qwen2.5-Math and Qwen2.5-Coder synthesize "trillions of text tokens" (textbooks, QA, instructions, code) | [Qwen3] §3.1 |

## Pre-training stages

| Fact | Value | Source |
|---|---|---|
| S1, General Stage | Over 30T tokens at a sequence length of 4,096, covering 119 languages and dialects | [Qwen3] §3.2 |
| S2, Reasoning Stage | About 5T higher-quality tokens at 4,096, with a larger share of STEM, coding, reasoning and synthetic data, and accelerated learning-rate decay | [Qwen3] §3.2 |
| S3, Long Context Stage | Hundreds of billions of tokens at 32,768; 75% of the text is 16,384–32,768 tokens long, 25% is 4,096–16,384 | [Qwen3] §3.2 |
| Context during pre-training | 4,096 in S1 and S2, raised to 32,768 in S3. The model is never pre-trained at 128K | [Qwen3] §1, §3.2 |
| RoPE base | Raised from 10,000 to 1,000,000 with ABF during S3 | [Qwen3] §3.2 |
| Context beyond 32K | YaRN plus Dual Chunk Attention give "a four-fold increase in sequence length capacity during inference" | [Qwen3] §3.2 |
| Advertised context, Qwen3-235B-A22B | 128K | [Qwen3] Table 2 |
| Learning rate and batch size | Predicted per model from scaling laws over the three stages; the values are not published | [Qwen3] §3.2 |
| Optimizer and other training settings | Not published. The report names no optimizer and gives no warmup, LR floor, weight decay, gradient-clipping threshold or training precision. The only schedule detail is that S2 "accelerate[s] the learning rate decay" | [Qwen3] §3.2 |
| Training hardware and cluster | Not published. The report describes no training GPUs, cluster, site or parallelism layout; "GPU" appears only as GPU-hours, comparing distillation with RL | [Qwen3] §3; §4, §4.7, Table 21 |

## Post-training

| Fact | Value | Source |
|---|---|---|
| Distillation direction | Strong-to-weak: knowledge from larger teacher models is distilled **into** the lightweight models, 5 dense (0.6B, 1.7B, 4B, 8B, 14B) and Qwen3-30B-A3B, off-policy then on-policy | [Qwen3] §4, §4.5 |

## Architecture (Qwen3-235B-A22B)

| Fact | Value | Source |
|---|---|---|
| Parameters | 235B total, 22B activated per token | [Qwen3] §1 |
| Layers | 94 | [Qwen3] Table 2; `config.json` `num_hidden_layers` @8efa617 |
| Hidden size | 4,096 | `config.json` `hidden_size` @8efa617 |
| Attention | GQA: 64 query heads, 4 KV heads, head dimension 128 | [Qwen3] Table 2; `config.json` `num_attention_heads`, `num_key_value_heads`, `head_dim` @8efa617 |
| Experts | 128 total, 8 activated per token, from fine-grained expert segmentation | [Qwen3] §2, Table 2 |
| Shared experts | None. "Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts" | [Qwen3] §2 |
| Expert width | `moe_intermediate_size` = **1,536** | `config.json` @8efa617 |
| The `intermediate_size` field | 12,288. It sizes a dense FFN layer, and no layer is dense (next row), so it is not an expert width | `config.json` @8efa617 |
| MoE layers | All 94: `mlp_only_layers` is empty and `decoder_sparse_step` is 1. There are no leading dense layers | `config.json` @8efa617 |
| Load balancing | A global-batch load-balancing loss (Qiu et al., 2025), not DeepSeek-style bias-based balancing | [Qwen3] §2 |
| RoPE $\theta$ in the released config | 1,000,000, with `rope_scaling` null (a single base) | `config.json` `rope_theta` @8efa617 |
| Active and total parameters, recomputed | Per layer: experts $8 \times 3 \times 4096 \times 1536 = 151\text{M}$ active; attention $71.3\text{M}$. Over 94 layers plus untied embeddings ($2 \times 151{,}936 \times 4096$): **22.1B active, 235.0B total**, matching the report. With 12,288 per expert the active count would be about 121B. This is our arithmetic | `config.json` @8efa617; [Qwen3] §1 |
