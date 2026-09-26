# Llama 3 fact sheet

Every fact the book relies on about Llama 3 lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md). Tokenizer facts are on the [tokenizer fact sheet](tokenizers.md#llama-3).

**Sources.**
- `[Llama3]` is Llama Team, AI @ Meta, *The Llama 3 Herd of Models*, arXiv:2407.21783**v3** (the revision these section, table and figure numbers refer to). See [reference 5](../b-references.md#5-llama-3). The report calls its 8B, 70B and 405B models "Llama 3". Where it gives a value for the 405B flagship only, the row says so.
- `[L31-card]` is Meta's model card for the same models under their release name, Llama 3.1: `meta-llama/llama-models` `models/llama3_1/MODEL_CARD.md` @e2e67cd.

Checked against the sources: 2026-09-26.

## Pre-training data

| Fact | Value | Source |
|---|---|---|
| Corpus size | "About 15T multilingual tokens", against 1.8T for Llama 2 | [Llama3] §1 |
| Flagship training tokens | The 405B model was pre-trained on 15.6T tokens at an 8K context window | [Llama3] §1, §2 |
| Final data mix | Roughly 50% general knowledge, 25% mathematical and reasoning, 17% code, 8% multilingual | [Llama3] §3.1.2 |
| How the mix was chosen | A knowledge classifier to downsample over-represented web categories, plus scaling-law experiments that train small models on candidate mixes to predict a large model's performance | [Llama3] §3.1.2 |
| De-duplication | Three levels: URL-level across the whole dataset; document-level with global MinHash; aggressive line-level, similar to ccNet. The section names no other method (no suffix arrays) | [Llama3] §3.1.1 |
| Quality classifiers | A fastText classifier trained to recognise text that would be referenced by Wikipedia, and RoBERTa-based classifiers trained on Llama 2 predictions; DistilRoberta produces the per-document quality scores | [Llama3] §3.1.1 |
| Code and reasoning classifiers | DistilRoberta models trained on web data annotated by Llama 2 | [Llama3] §3.1.1 |
| Language identification | A fastText-based model, 176 languages | [Llama3] §3.1.1 |
| Supported languages | 8: English, German, French, Italian, Portuguese, Hindi, Spanish and Thai. The base model saw a broader set | [Llama3] §5.2.4 |
| Contamination analysis | 8-gram overlap, following Singh et al. (2024), with a per-benchmark threshold. Table 15 reports 21 benchmarks | [Llama3] §5.1.4, Table 15 |

## Architecture

| Fact | Value | Source |
|---|---|---|
| Layers | 32 (8B), 80 (70B), 126 (405B) | [Llama3] Table 3 |
| Model dimension | 4,096 (8B), 8,192 (70B), 16,384 (405B) | [Llama3] Table 3 |
| FFN dimension | 14,336 (8B), 28,672 (70B), 53,248 (405B). That is 3.5×, 3.5× and 3.25× the model dimension; the ratios are our arithmetic | [Llama3] Table 3 |
| Attention | GQA; 32, 64 and 128 query heads with 8 key/value heads at every size | [Llama3] §3.2, Table 3 |
| Peak learning rate | 3×10⁻⁴ (8B), 1.5×10⁻⁴ (70B), 8×10⁻⁵ (405B) | [Llama3] Table 3 |
| Activation and position | SwiGLU; RoPE with base θ = 500,000 | [Llama3] §3.2, Table 3 |
| Document masking | An attention mask prevents self-attention across documents packed in one sequence | [Llama3] §3.2 |
| 405B training compute | 3.8×10²⁵ FLOPs | [Llama3] §3.2 |

## Training recipe

| Fact | Value | Source |
|---|---|---|
| Stages | Three: initial pre-training, long-context pre-training, annealing. The report gives the 405B recipe and says the 8B and 70B use "similar recipes" | [Llama3] §3.4 |
| Optimizer and schedule (405B) | AdamW, peak LR 8×10⁻⁵, linear warmup over 8,000 steps, cosine decay to 8×10⁻⁷ over 1,200,000 steps | [Llama3] §3.4.1 |
| Batch-size ramp (405B) | 4M tokens at sequence length 4,096; doubled to 8M tokens at 8,192 after 252M tokens; doubled to 16M after 2.87T tokens | [Llama3] §3.4.1 |
| Stability (405B) | "Few loss spikes", and no interventions were needed to correct divergence | [Llama3] §3.4.1 |
| Not reported for the flagship run | Adam β₁, β₂ and ε, weight decay, and the gradient-clipping threshold. The only weight decay given (0.1 × the LR) is for the scaling-law runs | [Llama3] §3.2.1, §3.4.1 |
| Numerics | MFU is reported against BF16 peak; gradients are accumulated, and reduce-scattered in FSDP, in FP32. FP8 appears only as a quantization for 405B inference | [Llama3] §3.3.2, Table 4, §6.2 |

## Long context and annealing

| Fact | Value | Source |
|---|---|---|
| Context extension (405B) | Six stages, from 8K to 128K | [Llama3] §3.4.2 |
| Context-extension tokens | About 800B | [Llama3] §3.4.2 |
| When a stage advances | When short-context evaluations have fully recovered and the model solves needle-in-a-haystack at that length | [Llama3] §3.4.2 |
| Annealing | The final 40M tokens: LR annealed linearly to 0 at 128K context, with the mix adjusted to upsample very high-quality sources | [Llama3] §3.4.3 |
| Checkpoint averaging | The final pre-trained model averages checkpoints (Polyak averaging) taken during annealing | [Llama3] §3.4.3 |
| Share of the run | 40M / 15.6T ≈ 0.0003% for annealing; 800B / 15.6T ≈ 5% for context extension. This is our arithmetic; the report does not say whether the 800B is counted inside the 15.6T | [Llama3] §2, §3.4.2, §3.4.3 |
| Effect of annealing on code and math | Raised a pre-trained 8B model's GSM8k and MATH validation scores by 24.0% and 6.4%; negligible for the 405B | [Llama3] §3.1.3 |
| Annealing as a data probe | A separate experiment: a 50%-trained 8B model annealed linearly to 0 over 40B tokens, with 30% weight on the candidate dataset | [Llama3] §3.1.3 |

## Infrastructure and parallelism

| Fact | Value | Source |
|---|---|---|
| GPUs | Up to 16K H100s (700 W TDP, 80 GB HBM3) on Meta's Grand Teton servers | [Llama3] §3.3.1 |
| Network | RoCE, based on Arista 7800 and Minipack2 switches. The cluster has 24K GPUs in a three-layer Clos network of 3,072-GPU pods; Llama 3 used up to 16K of them | [Llama3] §3.3.1 |
| Collective library | NCCLX, Meta's fork of NCCL | [Llama3] §3.3.3 |
| Parallelism | 4D, in the order [TP, CP, PP, DP], where DP is FSDP | [Llama3] §3.3.2, Figure 5 |
| Layout at 8K context | 8,192 GPUs: TP 8, CP 1, PP 16, DP 64 (43% MFU). 16,384 GPUs: TP 8, CP 1, PP 16, DP 128 (41% MFU) | [Llama3] Table 4 |
| Layout at 128K context | 16,384 GPUs: TP 8, CP 16, PP 16, DP 8, sequence length 131,072 (38% MFU). Every configuration keeps 16M tokens per batch | [Llama3] Table 4 |
| Context parallelism | All-gather of K and V, not a ring; the sequence is cut into 2 × CP chunks for load balance | [Llama3] §3.3.2 |

## Reliability

| Fact | Value | Source |
|---|---|---|
| Interruptions | 466 in a 54-day snapshot of 405B pre-training: 47 planned, 419 unexpected | [Llama3] §3.3.4 |
| Hardware share | About 78% of unexpected interruptions attributed to confirmed or suspected hardware issues | [Llama3] §3.3.4, Table 5 |
| GPU share | GPU issues are the largest category: 58.7% of unexpected interruptions. (Table 5's GPU rows count 268 of 419, which is 64%; the gap is its "Faulty GPU" row, printed as 30.1% for 148 interruptions, which is 35.3%. The book quotes the report's 58.7%. The check is our arithmetic.) | [Llama3] §3.3.4, Table 5 |
| Next-largest causes | Software bugs 12.9% (54), network switch or cable 8.4% (35), unplanned host maintenance 7.6% (32). Every remaining non-GPU row is 1.7% or less | [Llama3] Table 5 |
| Silent data corruption | 6 interruptions, 1.4% | [Llama3] Table 5 |
| Effective training time | Above 90%; significant manual intervention was needed only three times | [Llama3] §3.3.4 |
| Interruption rate | One unexpected interruption every ~3.1 hours (54 × 24 / 419), or ~51,000 GPU-hours per unexpected interruption if all 16,384 GPUs ran throughout. This is our arithmetic | [Llama3] §3.3.4 |

## Compute

| Fact | Value | Source |
|---|---|---|
| Training GPU-hours | 30.84M H100-80GB GPU-hours for Llama 3.1 405B; 39.3M for the 8B, 70B and 405B together | [L31-card] "Training Energy Use" @e2e67cd |
| Dollar cost | Neither source gives one | [Llama3] §3.3; [L31-card] @e2e67cd |
