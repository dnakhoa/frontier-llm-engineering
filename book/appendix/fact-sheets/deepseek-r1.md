# DeepSeek-R1 fact sheet

Every fact the book relies on about DeepSeek-R1 lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md). R1 is post-trained from DeepSeek-V3-Base; for the base model, see the [DeepSeek-V3 fact sheet](deepseek-v3.md).

**Sources.**
- `[Nature]` is Guo et al. (DeepSeek-AI), "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", *Nature* **645**, 633–638 (2025), doi:[10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z). It was received 2025-02-14, accepted 2025-07-17 and published 2025-09-17, and it is open access. This is the source of record. Its Methods subsections are unnumbered, so rows cite them by name, for example Methods ‘Reward design’.
- `[SI]` is that article's Supplementary Information (file `41586_2025_9422_MOESM1_ESM.pdf`). It has its own numbered sections and "Supplementary" tables and figures; rows cite them as §N, Table N, Fig. N.
- `[v1]` is the preprint, DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948**v1** (2025-01-22). See [reference 10](../b-references.md#10-deepseek-r1). arXiv also holds a v2 (2026-01-04), which this sheet does not use.
- `[V3]` is the DeepSeek-V3 report, as cited on the [DeepSeek-V3 fact sheet](deepseek-v3.md).

The **v1** column says what the preprint says where it differs from, or is silent on, the *Nature* version. A dash means the two agree, or that v1 says nothing relevant beyond what the row already states.

Checked against the sources: 2026-09-26.

## Pipelines

| Fact | Value | Source | v1 |
|---|---|---|---|
| R1-Zero | RL applied directly to DeepSeek-V3 Base, with no SFT first, using GRPO and a rule-based reward | [Nature] Main ‘DeepSeek-R1-Zero’; Methods ‘GRPO’ | — |
| R1-Zero template | The model must put its reasoning in `<think>…</think>` and its answer in `<answer>…</answer>`. Only this structure is imposed, with no content constraints | [Nature] Main ‘DeepSeek-R1-Zero’; [v1] Table 1 | Same template (Table 1) |
| Why R1 exists | R1-Zero has poor readability and mixes languages | [Nature] Main ‘DeepSeek-R1’ | — |
| R1 pipeline | Cold-start SFT on "thousands" of conversational long-CoT examples; a first RL stage; rejection sampling plus SFT on reasoning and non-reasoning data; a second RL stage for helpfulness and harmlessness | [Nature] Main ‘DeepSeek-R1’, Fig. 2 | Same four stages (§2.3.1–§2.3.4) |
| Named checkpoints | Dev1, Dev2 and Dev3 are intermediate checkpoints in the pipeline | [Nature] Fig. 2 | Not named |
| What the stages did | Dev1 improved instruction following but lost some reasoning, notably on AIME, because the cold-start set was small. Dev2 (after reasoning RL) improved reasoning benchmarks much more than general ones | [Nature] Table 2 and the text on it | Not reported |
| R1-Zero AIME 2024 | pass@1 rose from 15.6% to 77.9% during RL; self-consistency decoding reached 86.7% (Fig. 1a plots cons@16) | [Nature] Main ‘DeepSeek-R1-Zero’, Fig. 1 | 15.6% → **71.0%** pass@1; 86.7% with majority voting (§1, §2.2.4) |

## GRPO hyperparameters

| Fact | Value | Source | v1 |
|---|---|---|---|
| R1-Zero: optimiser settings | Learning rate 3×10⁻⁶, KL coefficient 0.001, rollout temperature 1 | [Nature] Methods ‘Training details of DeepSeek-R1-Zero’ | Not published |
| R1-Zero: sampling | 16 outputs per question. Maximum length 32,768 tokens before step 8.2k and 65,536 after, with a jump in both performance and length at step 8.2k | [Nature] Methods ‘Training details of DeepSeek-R1-Zero’ | Not published |
| R1-Zero: steps | 10,400 steps in total, 1.6 epochs | [Nature] Methods ‘Training details of DeepSeek-R1-Zero’ | Not published |
| R1-Zero: batch | 32 unique questions per step, so a batch of 512. Each rollout generates 8,192 outputs, split randomly into 16 minibatches and trained for one inner epoch | [Nature] Methods ‘Training details of DeepSeek-R1-Zero’ | Not published |
| Reference model | Replaced with the latest policy every 400 steps | [Nature] Methods ‘Training details of DeepSeek-R1-Zero’ | Not published |
| KL estimator | π_ref/π_θ − log(π_ref/π_θ) − 1, per output | [Nature] Methods ‘GRPO’, Eq. 2 | Same formula (§2.2.1) |
| R1 first RL stage | Learning rate 3×10⁻⁶, KL 0.001, **GRPO clip ratio ε = 10**, temperature 1, 16 outputs of up to 32,768 tokens, 32 questions per step (batch 512), reference reset every 400 steps, 8,192 outputs per rollout in 16 minibatches | [Nature] Methods ‘Training details of the first RL stage’ | Not published |
| Clip ratio | The paper states ε = 10 and says it "plays a crucial role": lower values truncate gradients for many tokens, higher ones destabilise training. The value is as printed; we have not seen it explained further | [Nature] Methods ‘Training details of the first RL stage’ | Not published |
| R1 second RL stage | Temperature lowered to 0.7 because higher temperatures gave incoherent generation. 1,700 steps; general instruction data and preference rewards are used only in the final 400 | [Nature] Methods ‘Training details of the second RL stage’ | Not published |
| R1-Zero-Qwen-32B (ablation) | Learning rate 2×10⁻⁶; otherwise the same settings as R1-Zero, with maximum length 32,768 | [SI] §2.4.1 | Not published |

## RL data

| Fact | Value | Source | v1 |
|---|---|---|---|
| RL prompts by domain | Math 26K; Code 17K; STEM 22K; Logic 15K; General 66K | [SI] §2.3.1, Table 1 | Not published |
| Total RL prompts | 146K. *Our arithmetic*: 26 + 17 + 22 + 15 + 66 | [SI] Table 1 | Not published |
| Domain details | Maths excludes proofs, reward 1/0 on the answer. Code is 17K competition problems "along with 8k bug fixing problems" from GitHub issues. STEM is multiple choice with 4 to 8 options. General includes 12,000 harmlessness questions. All questions are in Chinese or English | [SI] §2.3.1 | Not published |

## Rewards

| Fact | Value | Source | v1 |
|---|---|---|---|
| Rule-based reward | Reward_rule = Reward_acc + Reward_format, "combined with the same weight". Accuracy is checked by rules (a boxed answer; a compiler with test cases). The format reward requires `<think>` tags | [Nature] Methods ‘Reward design’, Eq. 4 | Same two rewards; no equation (§2.2.2) |
| No neural RM on reasoning | Neither outcome nor process neural reward models are used for reasoning tasks, because "neural reward models are susceptible to reward hacking during large-scale RL" and retraining them is costly | [Nature] Methods ‘Reward design’ | Same reasoning (§2.2.2) |
| Helpful reward model | 66,000 preference pairs, built by querying DeepSeek-V3 four times per pair with positions randomised, and keeping pairs whose score difference exceeds 1. Same architecture as R1 plus a scalar reward head. Batch 256, learning rate 6×10⁻⁶, one epoch, maximum length 8,192 tokens in training | [Nature] Methods ‘Helpful reward model’ | Not published |
| Safety reward model | 106,000 prompts with responses labelled safe or unsafe, trained pointwise with the helpful RM's hyperparameters | [Nature] Methods ‘Safety reward model’ | Not published |
| What each RM scores | Helpfulness: the final summary only. Harmlessness: the whole response, reasoning included | [Nature] Methods ‘Reward design’ | Same (§2.3.4) |
| Language-consistency reward | The proportion of target-language words in the CoT, added directly to the final reward | [Nature] Methods ‘Training details of the first RL stage’, Eq. 7 | Same definition, "directly summing" it with accuracy (§2.3.2) |
| Language-consistency ablation | On DeepSeek-R1-Distill-Qwen-7B: maths comparable, a slight degradation on coding | [SI] §2.6, Fig. 5 | Mentions a "slight degradation" without the ablation (§2.3.2) |
| Second-stage reward | Reward = Reward_reasoning + Reward_general + Reward_language, where Reward_reasoning = Reward_rule and Reward_general = Reward_reward_model + Reward_format | [Nature] Methods ‘Training details of the second RL stage’, Eqs 8–10 | Described in prose; no equation (§2.3.4) |

## SFT data

| Fact | Value | Source | v1 |
|---|---|---|---|
| Cold-start collection | R1-Zero samples at temperature 1.0 are kept only if the answer is correct and readable (sympy for maths; rules for repetition and language mixing). DeepSeek-V3 then refines them. Human annotators first rewrote traces into a conversational style used as examples | [SI] §2.3.2 | "Thousands" of examples. v1 lists the approaches explored (few-shot long-CoT prompting, prompting for reflection, readable R1-Zero outputs, human post-processing) but not this procedure (§2.3.1) |
| Second SFT set | About 600k reasoning samples from rejection sampling on the first-RL checkpoint (partly judged by DeepSeek-V3 as a generative RM), plus about 200k non-reasoning samples | [SI] §2.3.3 | Same 600k + 200k ≈ 800k (§2.3.3) |
| SFT samples by domain | Math 395,285; Code 211,129; STEM 10,124; Logic 10,395; General 177,812; total 804,745 | [SI] Table 2 | Not published |
| SFT hyperparameters | DeepSeek-V3-Base fine-tuned for 2–3 epochs; cosine decay from 5×10⁻⁵ to 5×10⁻⁶; maximum context 32,768; batch 128 | [SI] §2.4.2 | "Two epochs" (§2.3.3); no other values |

## Compute and cost

| Fact | Value | Source | v1 |
|---|---|---|---|
| Hardware | 64×8 H800 GPUs for both R1-Zero and R1. Preparatory experiments on a ~30B model used A100s | [SI] §2.4.4 | Not published |
| Wall-clock | R1-Zero about 198 hours. R1 "about 4 days, or roughly 80 hours", as printed; 4 days would be 96 hours | [SI] §2.4.4 | Not published |
| GPU-hours | R1-Zero 101K; SFT data creation 5K; R1 41K; total 147K H800 GPU-hours | [SI] Table 4 | Not published |
| Dollar cost | $202K + $10K + $82K = $294K, at an assumed $2 per H800 GPU-hour | [SI] Table 4 | Not published |
| Consistency check | 512 GPUs × 198 h ≈ 101K and 512 × 80 h ≈ 41K GPU-hours, matching Table 4. *Our arithmetic* | [SI] §2.4.4, Table 4 | — |
| Share of V3's budget | 147K is about 5% of V3's 2,788K H800 GPU-hours. *Our arithmetic*, using the [V3 sheet](deepseek-v3.md#context-extension-post-training-and-cost) | [SI] Table 4; [V3] Table 1 | — |

## Reward hacking and limitations

| Fact | Value | Source | v1 |
|---|---|---|---|
| The documented episode | With the helpful reward model, the reward score keeps rising while Codeforces pass@1 falls | [SI] §2.5, Fig. 4 | Not reported |
| What they did about it | More steps with the model-based preference reward "may lead to reward hacking", so preference rewards are used only in the last 400 of 1,700 second-stage steps | [Nature] Methods ‘Training details of the second RL stage’ | Not reported |
| Stated limitation | Where no reliable reward exists, R1 uses human-annotated supervised data "and only conducts RL for hundreds of steps"; scaling pure RL there "remains an open challenge" | [Nature] Main, ‘Reward hacking’ | §5 lists general capability, language mixing, prompting and software engineering; not reward hacking |
| Unsuccessful attempts | PRMs (hard to define steps, hard to label, and "inevitably" reward-hacked) and MCTS (a huge search space and a hard-to-train value model) | [SI] §7.2 | Same two attempts (§4.2) |

## Distillation

| Fact | Value | Source | v1 |
|---|---|---|---|
| Method | SFT of open base models on the ~800k R1-curated samples, for 2–3 epochs, cosine decay to one-tenth of the initial learning rate, maximum context 32,768, batch 64. No RL stage | [SI] §2.4.3 | SFT only on the 800k samples; no hyperparameters (§2.4) |
| Students and learning rates | Qwen2.5-Math-1.5B (1×10⁻⁴), Qwen2.5-Math-7B (8×10⁻⁵), Qwen2.5-14B (7×10⁻⁵), Qwen2.5-32B (6×10⁻⁵), Llama-3.1-8B (5×10⁻⁵), Llama-3.3-70B-Instruct (2×10⁻⁵) | [SI] Table 3 | Same six base models; no learning rates (§2.4) |
| Distillation vs RL at 32B | Qwen2.5-32B-Zero (more than 10K RL steps) roughly matches QwQ-32B-Preview; DeepSeek-R1-Distill-Qwen-32B beats it on every benchmark listed, for example AIME 2024 pass@1 of 72.6 vs 47.0 | [SI] §6.1, Table 13 | Same numbers; the RL model is called DeepSeek-R1-Zero-Qwen-32B (§4.1, Table 6) |
