# Kimi K2 fact sheet

Every fact the book relies on about Kimi K2 lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md). K2's architecture "follows a similar design to DeepSeek-V3", so the [DeepSeek-V3 fact sheet](deepseek-v3.md) is the natural comparison.

**Sources.** `[K2]` is Kimi Team (Moonshot AI), *Kimi K2: Open Agentic Intelligence*, arXiv:2507.20534**v2** (v1 2025-07-28; v2 2026-02-03). We confirmed the id against arXiv's abstract page. Section, table and figure numbers refer to v2. v1 has the same section structure, and the numbers we spot-checked in it are identical: 15.5T tokens, 1.04T parameters, τ = 100, 67M-token batch, the 400B + 60B annealing tokens, YaRN, 3,000+ MCP tools, 20,000+ synthetic tools and 10,000+ sandboxes.

Checked against the source: 2026-09-26.

## Architecture

| Fact | Value | Source |
|---|---|---|
| Size | 1.04T total parameters, 32.6B activated (the abstract rounds these to 1T and 32B) | [K2] §2.3, Table 2 |
| Design lineage | "Similar design to DeepSeek-V3": MLA attention, hidden dimension 7168, MoE expert hidden dimension 2048 | [K2] §2.3 |
| Layers | 61, the same as V3 | [K2] Table 2 |
| Experts | 384 routed experts (V3 has 256), 8 active per token, 1 shared expert, no expert grouping | [K2] §2.3, Table 2 |
| Dense layers | 1 (V3 has 3) | [K2] Table 2 |
| Attention heads | 64, half of V3's 128. At 128K sequence length, going from 64 to 128 heads would add 83% to inference FLOPs, for a validation-loss gain of only 0.5–1.2% | [K2] §2.3, Figure 6 |
| Sparsity choice | Sparsity is total experts over active experts; K2 uses 48 (384 / 8). At a fixed validation loss of 1.5, sparsity 48 needs 1.69×, 1.39× and 1.15× fewer FLOPs than sparsity 8, 16 and 32 | [K2] §2.3, Figure 5 |

## MuonClip

| Fact | Value | Source |
|---|---|---|
| Optimizer | MuonClip: Muon with weight decay, consistent update-RMS matching, and QK-Clip | [K2] §2.1, Algorithm 1 |
| The problem it fixes | Exploding attention logits, which the report says happen more often with Muon than with AdamW. In a 9B-active / 53B-total MoE trained with plain Muon, maximum logits quickly exceeded 1000 | [K2] §2.1, Figure 2 |
| QK-Clip mechanism | After each update, rescale W_q and W_k for any head whose maximum logit S_max exceeds a threshold τ. The forward and backward pass of the current step is unchanged | [K2] §2.1 |
| Per-head clipping for MLA | γ_h = min(1, τ / S_max^h). The head-specific q^C and k^C are scaled by √γ_h and the head-specific rotary q^R by γ_h; the shared rotary key k^R is left alone | [K2] §2.1 |
| Threshold used for K2 | τ = 100. Logits sat at the cap early on and decayed to a normal range after about 30% of training steps, with no change to τ | [K2] §2.1, Figure 2 |
| Stability result | 15.5T tokens "without a single loss spike". The per-step loss curve is published unsmoothed | [K2] §1, Figure 3 |

## Pre-training

| Fact | Value | Source |
|---|---|---|
| Tokens | 15.5T, across four domains: web text, code, mathematics and knowledge | [K2] §2.2, §2.5 |
| Rephrasing | Knowledge text is rewritten by an LLM in varied styles, chunk by chunk, with fidelity checks; each corpus is rephrased at most twice. Maths is rewritten in a "learning-note" style, and some is translated into English | [K2] §2.2 |
| Rephrasing evidence | SimpleQA accuracy on an early checkpoint: 23.76 for raw data × 10 epochs; 27.39 for 1 rephrasing × 10 epochs; 28.94 for 10 rephrasings × 1 epoch | [K2] §2.2, Table 1 |
| Schedule | WSD at a 4,096-token context: 500-step warmup; constant 2e-4 for 10T tokens; cosine decay 2e-4 → 2e-5 over 5.5T | [K2] §2.5 |
| Weight decay and batch | Weight decay 0.1; global batch 67M tokens throughout | [K2] §2.5 |
| Annealing | Learning rate 2e-5 → 7e-6 over 400B tokens at 4K sequence length, then 60B tokens at 32K | [K2] §2.5 |
| Precision | Parameters in BF16 with FP32 gradient accumulation. Some activations are stored in FP8-E4M3, but FP8 is **not** used for computation | [K2] §2.4.2, §2.4.3 |

## Context length

| Fact | Value | Source |
|---|---|---|
| Context window | 128K, reached with YaRN. Training itself went only to 32K: pre-training at 4K, then the 60B-token annealing stage at 32K | [K2] §2.5 |

## Training infrastructure and compute

| Fact | Value | Source |
|---|---|---|
| Cluster | NVIDIA H800 GPUs; 8 per node with NVLink/NVSwitch and 2 TB RAM; 8×400 Gbps RoCE between nodes | [K2] §2.4.1 |
| Parallelism | 16-way PP with virtual stages, 16-way EP and ZeRO-1 DP. It runs on any multiple of 32 nodes. DualPipe was rejected because it doubles parameter and gradient memory | [K2] §2.4.2 |
| Compute and cost | **Not published.** The report gives no GPU count, GPU-hours or dollar cost | [K2] §2.4 |

## Post-training: agentic data synthesis

| Fact | Value | Source |
|---|---|---|
| SFT optimizer | Muon, which the report also recommends for anyone fine-tuning K2 | [K2] §3.1 |
| Pipeline stages | Tool-spec generation, then agent and task generation, then trajectory generation | [K2] §3.1.1, Figure 8 |
| Tools | More than 3,000 real MCP tools fetched from GitHub, plus more than 20,000 synthetic tools produced by hierarchical domain evolution | [K2] §3.1.1 |
| Agents and tasks | "Thousands" of agents, each a synthesised system prompt with a tool combination. Every task has an explicit rubric | [K2] §3.1.1 |
| Trajectories | Simulated users and a stateful tool simulator. An LLM judge keeps only the trajectories that meet the rubric. Real execution sandboxes are added for coding and software engineering | [K2] §3.1.1 |

## Post-training: RL

| Fact | Value | Source |
|---|---|---|
| Framework | A Gym-like framework combining verifiable rewards (RLVR) with a self-critique rubric reward for subjective tasks | [K2] §1, §3.2 |
| Verifiable domains | Maths, STEM and logic (problems of moderate difficulty, chosen by the SFT model's pass@k); complex instruction following (code checks plus an LLM judge, with a hack-check layer); faithfulness (a sentence-level judge); coding and SWE; safety | [K2] §3.2.1 |
| SWE sandboxes | Built from GitHub PRs and issues on Kubernetes, supporting "over 10,000 concurrent sandbox instances" | [K2] §3.2.1 |
| Self-Critique Rubric Reward | The K2 actor generates responses and the K2 critic ranks them pairwise against core rubrics, prescriptive rubrics (meant to stop reward hacking) and human-annotated rubrics. The critic ability is first initialised in SFT from preference data | [K2] §3.2.2, Appendix F |
| Critic refinement | The critic is updated continuously on on-policy rollouts from verifiable-reward prompts, which grounds its subjective judgments in verifiable signals | [K2] §3.2.2 |
| RL objective | K1.5's policy-optimisation algorithm: a squared loss on (r − r̄ − τ log π_θ/π_old) over K samples per prompt, minimised with Muon | [K2] §3.2.3 |
| Additions | Per-task maximum token budgets, with a penalty for truncated responses; a PTX loss on hand-picked high-quality data; a temperature decay from exploration to exploitation | [K2] §3.2.3 |
| What is not published | RL prompt counts, SFT dataset size and post-training compute | [K2] §3.1, §3.2 |
