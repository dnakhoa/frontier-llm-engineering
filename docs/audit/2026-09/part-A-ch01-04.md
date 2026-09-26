# Staleness audit, Part A: Chapters 1–4

Audit date: 2026-09-26. Read-only; no repo file was edited.

**Method.** I read each chapter in full and checked every load-bearing claim against primary sources: arXiv technical reports, official lab blogs, model cards, and the `config.json` / `tokenizer.json` files in official Hugging Face repos, which I parsed directly. I checked each 2025–26 anchor myself. Anything I could not confirm from a primary source is under "Unverified leads" at the end.

**One finding is not about staleness.** Several claims in Ch 1, 3 and 4 **were never true**: the DeepSeek-V3 parallelism layout, the Llama-3 data mix, the tokenizer JSON blocks, and the Qwen3 pre-tokenizer regex. The primary sources the book cites contradict them. They are tagged WRONG-NOW with the note *"(never accurate)"*, because a refresh has to fix them whatever it does about dates. These are the most urgent items in this report.

**Terms used below:**
- **Primary source** means the lab's own report, blog, model card or repo, not press coverage or a third-party blog.
- **Mirror** means an unofficial Hugging Face copy of a gated official repo. The `unsloth` repos used for Llama 4 and Gemma 3 are mirrors.
- **Vocab figures:** when a model's embedding table is padded, this report gives both the tokenizer's base vocabulary and the padded `vocab_size`.

---

## Ch 1 — What a real frontier training run looks like

**Verdict:** Needs rewrite of §1.5 (parallelism, which is factually wrong) and §1.6 (optimizer and precision). §1.2 needs a refresh ("the cleanest public example" is now DeepSeek-V4, or V3 kept explicitly as a historical spine). §1.4 needs a refresh for attention patterns. Light touch elsewhere.

### Findings (ranked most-severe first)

1. **WRONG-NOW (never accurate)** · `01-what-a-frontier-run-looks-like.md:145-152`
   - **Claim:** "the 60 transformer layers are split across 4 pipeline stages … 256 routed experts are split across the 8 GPUs within a node … Data parallelism: 64 replicas … Dense layers use small TP … 4 (PP) × 8 (EP) × 64 (DP) = 2,048"
   - **What the source says:** the DeepSeek-V3 report (§3.2) says: *"16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)."* It also says it trains *"without using costly Tensor Parallelism (TP)."* The official config has `num_hidden_layers: 61`, not 60.
   - **Knock-on:** the whole worked decomposition, and the cross-reference to §10.9, need redoing.
   - **Source:** DeepSeek-V3 Technical Report, DeepSeek-AI, Dec 2024, https://arxiv.org/abs/2412.19437 (§3.2), and https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
   - **Also:** `:152` says "the Qwen3 team used a similar 3D-parallel layout." The Qwen3 report (arXiv 2505.09388) describes no parallelism layout, so this is unsupported.

2. **WRONG-NOW** · `:14`, `:203`
   - **Claim:** "An optimizer (almost always AdamW or a variant)"; "The optimizer is almost always AdamW"
   - **What supersedes it:** Muon is now used for the bulk of parameters in several flagship open runs:
     - Kimi K2 used MuonClip, with QK-Clip, over 15.5T tokens "without a single loss spike".
     - GLM-4.5 used Muon for all parameters except embeddings, biases and RMSNorm weights.
     - DeepSeek-V4 uses Muon for most parameters, with AdamW kept for the embedding, prediction head and RMSNorm.
   - The honest 2026 statement is "AdamW or Muon (with AdamW kept for embeddings/norms)".
   - **Sources:**
     - Kimi K2: Open Agentic Intelligence, Moonshot AI, Jul 2025, https://arxiv.org/abs/2507.20534
     - GLM-4.5, Zhipu/Z.ai, Aug 2025, https://arxiv.org/abs/2508.06471 (§2.4)
     - DeepSeek-V4, DeepSeek-AI, Apr 2026, https://arxiv.org/abs/2606.19348 (§4.2.2)
     - Muon is Scalable for LLM Training, Moonshot, Feb 2025, https://arxiv.org/abs/2502.16982

3. **DATED-FRAMING** · `:22`, `:24-38`
   - **Claim:** "The cleanest public example is DeepSeek-V3"
   - **What supersedes it:** DeepSeek itself has shipped a successor, DeepSeek-V4 (report 26 Apr 2026). It has two models:
     - V4-Pro: 1.6T total, 49B active.
     - V4-Flash: 284B total, 13B active.
     - Both: 1M-token context, >32T pre-training tokens, a hybrid CSA/HCA attention stack, Manifold-Constrained Hyper-Connections (mHC), Muon, and FP4 QAT (FP4 quantization-aware training) for expert weights.
   - V3 can stay as the spine only if the chapter says it is a 2024 snapshot and adds a "what changed by V4" box.
   - The comparison "matches or exceeds Llama-3.1-405B-Instruct and GPT-4o" (`:36`) is also a 2024 frame.
   - **Source:** DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, DeepSeek-AI, 26 Apr 2026, https://arxiv.org/abs/2606.19348; model card https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

4. **DATED-FRAMING** · `:94`, `:103-105`, `:12`
   - **Claim:** "The pre-training architecture for a 2024–2025 frontier model is almost always a decoder-only transformer, with one of three attention patterns" (MHA / GQA / MLA)
   - **What supersedes it:** the 2025–26 frontier added at least four more families:
     - **Learned sparse attention on top of MLA:** DeepSeek Sparse Attention's lightning indexer in V3.2. V4's Compressed Sparse Attention (CSA) plus Heavily Compressed Attention (HCA) cut the KV cache to 10% of V3.2's at 1M context.
     - **Hybrid linear attention:**
       - Qwen3.5 uses a 3:1 Gated DeltaNet : gated-attention layout. It is 60 layers, with `layer_types` = 3×`linear_attention` then 1×`full_attention`.
       - MiniMax-M1 uses lightning attention.
     - **Sliding-window / local–global hybrids:** gpt-oss has `sliding_window: 128`. Gemma 3 has `sliding_window: 1024` with local and global layers.
     - **A counter-trend worth one sentence:** MiniMax-M2 went back to full attention in every layer, and MiniMax published why.
   - **Sources:**
     - DeepSeek-V3.2, Dec 2025, https://arxiv.org/abs/2512.02556
     - DeepSeek-V4 (above)
     - Qwen3.5-397B-A17B model card, Feb 2026, https://huggingface.co/Qwen/Qwen3.5-397B-A17B
     - MiniMax-M1, Jun 2025, https://arxiv.org/abs/2506.13585
     - gpt-oss model card, OpenAI, Aug 2025, https://arxiv.org/abs/2508.10925
     - "Why Did M2 End Up as a Full Attention Model?", MiniMax, 2025, https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model

5. **DATED-FRAMING** · `:211-215`
   - **Claim:** the precision timeline ends at "2024–2025: FP8 mixed-precision"
   - **What supersedes it:**
     - DeepSeek-V3.1 (Aug 2025) trains with the "UE8M0 FP8 scale data format" for compatibility with microscaling formats.
     - NVIDIA published full NVFP4 pre-training (Sep 2025).
     - DeepSeek-V4 applies MXFP4 QAT to MoE expert weights and to the CSA indexer's QK path. The released checkpoints are "FP4 + FP8 Mixed". Note that V4 uses FP4 in post-training QAT and serving, not in the pre-training matmuls.
   - **Sources:**
     - DeepSeek-V3.1 model card, Aug 2025, https://huggingface.co/deepseek-ai/DeepSeek-V3.1
     - Pretraining Large Language Models with NVFP4, NVIDIA, 29 Sep 2025, https://arxiv.org/abs/2509.25149
     - DeepSeek-V4 §5.2.1 (above)

6. **DATED-FRAMING** · `:11`, `:12`
   - **Claim:** "10–15 trillion training tokens"; "100B–1T+ parameters"
   - **What supersedes it:** 2025–26 pre-training corpora are larger:

     | Model | Pre-training tokens |
     |---|---|
     | Qwen3 | 36T |
     | Llama 4 | >30T mixture |
     | Llama 4 Scout | ~40T |
     | DeepSeek-V4 | >32T |
     | GLM-4.5 | 23T |
     | Kimi K2 | 15.5T |

   - The largest open MoE models now reach 1T (Kimi K2) and 1.6T (V4-Pro) total parameters.
   - **Sources:**
     - Qwen3 Technical Report, May 2025, https://arxiv.org/abs/2505.09388
     - "The Llama 4 herd", Meta, Apr 2025, https://ai.meta.com/blog/llama-4-multimodal-intelligence/
     - Llama 4 model card, https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md
     - DeepSeek-V4, Kimi K2 (above)

7. **DATED-FRAMING** · `:197`, `:208`
   - **Claim:** "cosine decay with a small floor … is the most common"
   - **What supersedes it:** constant-then-decay (WSD-style) schedules are now common in flagship reports:
     - Kimi K2: constant 2e-4 for 10T tokens, then cosine to 2e-5.
     - DeepSeek-V4-Flash: warmup, constant 2.7e-4 "for most of the training", cosine decay near the end.
     - SmolLM3: stable, stable, then decay stages.
   - GLM-4.5 is a documented counterexample. It chose cosine because WSD "perform[ed] worse … indicating underfitting". That disagreement is worth showing.
   - **Sources:** Kimi K2 §2; DeepSeek-V4 §4.2.2; GLM-4.5 §2.4; "SmolLM3", Hugging Face, 8 Jul 2025, https://huggingface.co/blog/smollm3

8. **MISSING-MAJOR** · §1.7 (`:239-259`)
   - **What's missing:** training-stability engineering has become a first-class, published topic:
     - Kimi K2's QK-Clip.
     - V4's "Anticipatory Routing", with a note that spikes are "consistently tied to outliers in the MoE layers".
     - V4's bitwise batch-invariant, deterministic kernels, used to debug spikes.
   - The chapter's failure list covers hardware but not these numerics-side failures.
   - **Source:** DeepSeek-V4 §3.3 and §4.2.3; Kimi K2 §2.

9. **MISSING-MINOR** · `:273`
   - **Claim:** capability evals are "MMLU, GSM8K, HumanEval, MMLU-Pro, GPQA"
   - **What's missing:** 2025–26 reports lead with agentic and long-horizon evals, such as SWE-bench Verified and Terminal-Bench (see the MiniMax M2 post and the V3.2/V4 reports). GSM8K and HumanEval are saturated for frontier models. This belongs to the eval chapter's auditor, but Ch 1 should not present them as the frontier eval set.

10. **MISSING-MINOR** · `:300`
    - **Claim:** post-training is "SFT, RLHF, DPO, GRPO"
    - **What's missing:** V4 describes a new standard shape: independently trained domain experts (SFT + GRPO), then **on-policy distillation with full-vocabulary logit KL** into one model. Worth one line here.
    - **Source:** DeepSeek-V4 §5.1.2.

### Keep as-is
- §1.1 pipeline-of-pipelines list, apart from the numbers in items 1, 2 and 4.
- The KV-bytes formula (`:99`).
- The MHA/GQA/MLA descriptions, as the base three.
- The 16-bytes-per-parameter memory arithmetic (`:217-235`).
- The recovery loop steps (`:249-255`).
- The monitoring signals (`:265-270`).
- The §1.9 JD table and §1.10 takeaways.

These are mechanism-level and still true.

### Lab opportunity (optional)
None for Ch 1 itself. A Muon-vs-AdamW lab belongs to Ch 8 (see the Ch 8 audit): Newton–Schulz orthogonalized updates on a small MLP or transformer on CPU, with QK logit growth measured with and without QK-Clip. It would support the rewritten §1.6.

### References to add
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* arXiv:2606.19348, 26 Apr 2026.
- DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* arXiv:2512.02556, 2 Dec 2025.
- Kimi Team (Moonshot AI). *Kimi K2: Open Agentic Intelligence.* arXiv:2507.20534, 28 Jul 2025.
- Liu et al. (Moonshot AI). *Muon is Scalable for LLM Training.* arXiv:2502.16982, 24 Feb 2025.
- GLM-4.5 Team (Zhipu AI / Z.ai). *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models.* arXiv:2508.06471, 8 Aug 2025.
- NVIDIA. *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149, 29 Sep 2025.
- Meta AI. *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation.* Blog, 5 Apr 2025, https://ai.meta.com/blog/llama-4-multimodal-intelligence/
  - It reports: FP8 on 32K GPUs at 390 TFLOPs/GPU for Behemoth, a >30T-token mixture, 200 languages, and MetaP.
  - **Do not cite arXiv:2601.11659 ("The Llama 4 Herd …").** It is withdrawn and is not a Meta publication.
- MiniMax. *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention.* arXiv:2506.13585, 16 Jun 2025.
- MiniMax. *Why Did M2 End Up as a Full Attention Model?* 2025, https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model
- Qwen Team. *Qwen3.5-397B-A17B model card.* Feb 2026, https://huggingface.co/Qwen/Qwen3.5-397B-A17B

---

## Ch 2 — The org chart: who actually does what

**Verdict:** Light touch, with one section refresh: §2.3.8, the Reasoning/Agents role. The chapter says openly that it has no primary literature, and most of it is role mechanics that age well. The dated parts are the lab roster, the role descriptions for RL and agents, and the "as of 2024–2025" headcount framing.

### Findings (ranked most-severe first)

1. **DATED-FRAMING** · `02-org-chart-and-roles.md:194-207`
   - **Claim:** the Reasoning/Agents work "is closer to classical AI (planning, search) than to standard RLHF", and the typical week includes "design a new MCTS-based rollout strategy"
   - **What supersedes it:** the reasoning-RL recipe that won in 2025 is outcome-reward RL on verifiable tasks, not tree search. DeepSeek-R1's paper lists MCTS under "Unsuccessful Attempts".
   - The agent work in 2025–26 reports is dominated by:
     - **Environment and task synthesis:**
       - Kimi K2: "large-scale agentic data synthesis pipeline".
       - V3.2: "Large-Scale Agentic Task Synthesis Pipeline".
       - Qwen3.5: "RL scaled across million-agent environments".
     - **Asynchronous RL infrastructure:** Qwen3.5's "asynchronous RL frameworks", and Kimi K2's checkpoint-engine weight broadcast.
   - The role description, skills list and JD should move toward "RL environments + verifiers + agent harnesses + async rollout infra".
   - **Sources:**
     - DeepSeek-R1, DeepSeek-AI, Nature 645:633–638, 17 Sep 2025, https://www.nature.com/articles/s41586-025-09422-z (and arXiv:2501.12948)
     - Kimi K2 (§3, App. G)
     - DeepSeek-V3.2 (abstract)
     - Qwen3.5 model card

2. **DATED-FRAMING** · `:176`, `:188`
   - **Claim:** the RL loop has "a policy model, a reference model, a reward model, a value model"
   - **What supersedes it:** GRPO-family methods, which the book teaches later, drop the value model. That is their point, and the book's own ref [15] covers it. By 2025–26, rule-based or verifier rewards and sandboxed environments often replace the learned reward model for reasoning and agent RL.
   - **What changed for this role:** rollout generation is now the whole problem. Examples are asynchronous rollouts and the H2D→broadcast→reload weight-sync pipeline in Kimi K2 App. G.
   - **Source:** Kimi K2 App. G; DeepSeek-R1 (Nature).

3. **DATED-FRAMING** · `:60`, `:347`, `:357-362`
   - **Claim:** the lab roster is "Alibaba Qwen, DeepSeek, Moonshot, Zhipu, ByteDance Doubao … Baichuan" and "Anthropic, OpenAI, Google DeepMind, Meta FAIR, xAI", with headcounts "As of 2024–2025"
   - **What's dated:**
     - The date stamp.
     - Meta FAIR as the Llama team. Llama was built by Meta's GenAI org, and the 2025 reorganisation matters. See Unverified leads; I could not find a Meta-published primary source.
     - The omission of MiniMax, which has published open reports such as M1.
     - Baichuan listed as a peer of the frontier labs, which is a 2024 frame.
     - The "DeepSeek ~200 people total" figure. The DeepSeek-V3.2 arXiv entry lists "DeepSeek-AI and 263 other authors". That is authors, not staff, but it is the kind of dated public data point the section already says it uses.
   - **Sources:**
     - DeepSeek-V3.2 arXiv abstract page, https://arxiv.org/abs/2512.02556
     - MiniMax-M1, arXiv:2506.13585
     - GLM-4.5, arXiv:2508.06471

4. **MISSING-MINOR** · §2.1 station list (`:11-20`) and §2.5 flow (`:293-322`)
   - **What's missing:**
     - Mid-training now routinely carries **agentic data** (V4 §4.1: "incorporating agentic data during the mid-training phase").
     - Post-training has a **specialist-experts → on-policy-distillation merge** stage (V4 §5.1).
     - The flow diagram's "14T tokens" and "60 days" are V3-era numbers. V4 uses >32T tokens.
   - **Source:** DeepSeek-V4.

5. **MISSING-MINOR** · `:228`, `:241`
   - **Claim:** the Evaluation Engineer runs "MMLU, GSM8K, HumanEval"
   - **What's dated:** the benchmark examples. Agentic evals (SWE-bench Verified, Terminal-Bench) are what 2025–26 releases headline. Defer details to the eval chapter's audit.

### Keep as-is
- §2.1's "org chart is the pipeline" model.
- The role taxonomy structure (§2.2).
- Roles 2.3.1–2.3.5 (data, pre-training research, large-scale training, kernel, cluster).
- The §2.4 blurry boundaries.
- §2.5's "dependency graph, not a flow chart".
- §2.6 JD-phrase decoder.
- The estimate caveat in §2.7.

These are role mechanics and remain accurate. Keep the §2.7 disclaimer; just re-date it.

### Lab opportunity (optional)
None. The chapter has no lab, and a CPU lab does not fit an org-chart chapter.

### References to add
- DeepSeek-AI. *DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.* Nature 645, 633–638, 17 Sep 2025. doi:10.1038/s41586-025-09422-z. This is the peer-reviewed version of the book's ref [10]. Update ref [10] to point at it as well.
- Kimi K2 (above), App. G for RL infrastructure.
- DeepSeek-V4 (above), §5 for the post-training pipeline.

---

## Ch 3 — Trillion-token data pipelines

**Verdict:** Needs rewrite of §3.3.3 (learned quality filtering), §3.5 (real numbers, where several figures are wrong), §3.2.6 and §3.3.7 (synthetic data and mixing). Refresh §3.3.1, the PDF extraction paragraph. The pipeline skeleton is still correct. Its model-based steps and all of its "real numbers" are either 2023 methods or factually wrong.

### Findings (ranked most-severe first)

1. **WRONG-NOW (never accurate)** · `03-data-pipelines.md:362-367`
   - **Claim:** Llama-3 used "15.6T training tokens (for the 8B model…)", a "Data mix: ~50% English web, ~25% code, ~10% multilingual, ~7.5% academic, ~7.5% other", and "Aggressive dedup with MinHash and suffix arrays"
   - **What the Llama 3 paper says:**
     - Data mix (§3.1.2): *"roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens."*
     - Dedup: URL-level, document-level (global MinHash) and line-level. It does not mention suffix arrays.
     - The ~15.6T figure is for the 405B flagship, not the 8B.
   - **Source:** The Llama 3 Herd of Models, Meta, Jul 2024, https://arxiv.org/abs/2407.21783 (§3.1.1, §3.1.2)

2. **WRONG-NOW (never accurate)** · `:32`, `:369-373`
   - **Claim:** Qwen3's "~36T tokens (across all stages including pre-training, mid-training, and post-training data)", with a "~50/50 English/Chinese" mix
   - **What the Qwen3 report says:**
     - 36T is the pre-training corpus: S1 >30T at 4K, S2 ~5T STEM/code/reasoning/synthetic, S3 hundreds of billions of tokens at 32K.
     - It covers 119 languages and dialects.
     - It gives no 50/50 EN/ZH split.
   - The same "50/50" claim appears at `:82`.
   - **Source:** Qwen3 Technical Report, 14 May 2025, https://arxiv.org/abs/2505.09388 (§3.1, §3.2)

3. **WRONG-NOW (never accurate)** · `:146-148`
   - **Claim:** "The Llama-2 paper introduced … a quality classifier on data labeled by Llama-2 itself"
   - **What the sources say:** the Llama 3 paper does this. It uses fastText plus DistilRoberta classifiers trained on Llama 2 annotations. The circularity argument stays valid; only the attribution changes.
   - `:423` repeats "The Llama-2/3 paper showed that aggressive dedup dramatically improves model quality". That result is Lee et al. 2022 (ref [8]), not Llama.
   - **Source:** Llama 3 §3.1.1.

4. **WRONG-NOW** · `:90`
   - **Claim:** synthetic data is "used heavily in mid-training and post-training; less so in pre-training (because the model needs to learn from ground truth…)"
   - **What supersedes it:** synthetic data is now a trillion-token pre-training input:
     - Qwen3 synthesized "trillions of text tokens" with Qwen2.5-Math and Qwen2.5-Coder.
     - Kimi K2's core pre-training data innovation is rephrasing. Knowledge rewrites and "learning-note" math rewrites raise token utility, with a SimpleQA table showing rephrasing beating multi-epoch repetition.
     - Nemotron-CC and Nemotron-CC-v2 ship LLM-rephrased Common Crawl, v2 using Qwen3-30B-A3B.
   - §3.2.6 and the takeaway list need rewriting around "synthetic rephrasing as token amplification".
   - **Sources:**
     - Qwen3 §3.1
     - Kimi K2 §2.2
     - Nemotron-CC, NVIDIA, Dec 2024, https://arxiv.org/abs/2412.02595
     - Nemotron-CC-v2 dataset card, NVIDIA, 2025, https://huggingface.co/datasets/nvidia/Nemotron-CC-v2

5. **DATED-FRAMING** · `:138-182`, `:86` (and Ch 1 `:86`)
   - **Claim:** the canonical learned filter is "XGBoost on hand-crafted features, or a 100M–1B transformer", trained on 50K hand-labelled docs, with a keep-top-40% threshold
   - **What supersedes it:** a hard percentile cut is no longer the state of the art. The current practice has three parts:
     - **LLM-annotated labels distilled into cheap classifiers.** The DCLM fastText and FineWeb-Edu lineage, and Llama 3's DistilRoberta.
     - **Classifier ensembles plus quality buckets.** Nemotron-CC; and GLM-4.5, which is "inspired by Nemotron-CC", divides the web into quality buckets and has its top bucket contribute >3.2 epochs.
     - **Quality-aware upsampling instead of hard thresholds.** Olmo 3's curve discards the bottom ~40% and *repeats* the top 5%. Its appendix shows this beats "naive quality-based filtering" in a data-constrained 4.51T simulation.
   - The XGBoost code block can stay as a teaching toy. It should not be labelled "a real (simplified) quality classifier" of 2026.
   - **Sources:**
     - DataComp-LM, Jun 2024, https://arxiv.org/abs/2406.11794
     - Nemotron-CC, arXiv:2412.02595
     - GLM-4.5 §2.2
     - Olmo 3, Ai2, 15 Dec 2025, https://arxiv.org/abs/2512.13961 (§3.4.4, App. A.2.5)

6. **MISSING-MAJOR** · §3.2 and §3.3.7 (`:279-288`)
   - **What's missing:** **multi-stage pre-training curricula** as the normal shape. The chapter treats the mix as one static ratio, adjusted mid-run.
   - Every 2025 report documents staged mixes:
     - Qwen3: S1 general, S2 reasoning-heavy with faster LR decay, S3 long-context.
     - SmolLM3: three stages, with web going from 85% to 75% to 63% and math/code upsampled at decay.
     - Olmo 3: pre-training on Dolma 3 Mix (6T), then Dolmino mid-training (100B), then long-context.
     - DeepSeek-V4: agentic data in mid-training.
   - Mixing tools have also moved past DoReMi (2023): Nemotron-CLIMB clustering-based mixture search, and Olmo 3's swarm-based "token-constrained mixing".
   - The claim "The DeepSeek-V3 paper mentions this kind of mid-run adjustment" (`:286`) is not supported by the V3 report as far as I could find.
   - **Sources:**
     - Qwen3 §3.2; SmolLM3 blog (above); Olmo 3 §3.4–3.5; DeepSeek-V4 §4.1
     - Nemotron-CLIMB, NVIDIA, Apr 2025, https://arxiv.org/abs/2504.13161

7. **DATED-FRAMING** · `:109-111`
   - **Claim:** PDF extraction uses "pdftotext, pdfplumber, grobid … Nougat, Marker"; "The DeepSeek-V3 paper mentions … a custom pipeline. The Qwen3 paper similarly references custom PDF handling."
   - **What supersedes it:** 2025 PDF extraction is VLM OCR at trillion-token scale:
     - Qwen3 used Qwen2.5-VL to OCR "a large volume of PDF-like documents", refined with Qwen2.5, yielding "trillions" of tokens.
     - Olmo 3's olmOCR science-PDF pool.
     - Hugging Face's FinePDFs: ~3T tokens, 475M docs, 1,733 languages, built with Docling plus RolmOCR.
     - DeepSeek-OCR (Oct 2025), which DeepSeek positions for generating LLM training data at "200k+ pages per day".
   - I could not find a PDF-pipeline statement in the DeepSeek-V3 report. Treat that attribution as unsupported. DeepSeek-V4 §4.1 does emphasise "long-document data curation, prioritizing scientific papers, technical reports".
   - **Sources:**
     - Qwen3 §3.1
     - olmOCR, Ai2, Feb 2025, https://arxiv.org/abs/2502.18443
     - FinePDFs dataset card, Hugging Face, 2025, https://huggingface.co/datasets/HuggingFaceFW/finepdfs
     - DeepSeek-OCR, 21 Oct 2025, https://arxiv.org/abs/2510.18234

8. **MISSING-MAJOR** · §3.2 (sources) and §3.5
   - **What's missing:** the 2025 open-data releases a reader would expect to be named. All are primary and all are usable as teaching anchors:
     - **FineWeb2** (Jun 2025): >1,000 languages, ~100 CC snapshots, dedup-informed "rehydration" upsampling.
     - **Nemotron-CC** (Dec 2024) and v2/v2.1 (2025).
     - **Common Pile v0.1** (Jun 2025): 8TB of public-domain and openly licensed text.
     - **Essential-Web v1.0** (Jun 2025): 24T tokens with taxonomy labels.
     - **Dolma 3** (Olmo 3, Dec 2025): 9.3T pool, 6T mix.
     - **FinePDFs** (2025).
   - The §3.5 "real numbers from real labs" table should also add at least three 2025–26 rows: DeepSeek-V4 (>32T), Kimi K2 (15.5T), and GLM-4.5 (23T). Replace or supplement the V3 and Llama-3 rows.
   - **Sources:**
     - https://arxiv.org/abs/2506.20920
     - https://arxiv.org/abs/2506.05209
     - https://arxiv.org/abs/2506.14111
     - https://arxiv.org/abs/2512.13961
     - https://arxiv.org/abs/2412.02595

9. **WRONG-NOW** · `:55`
   - **Claim:** LibGen-style book collections: "The legal and ethical status is debated."
   - **What supersedes it:** there is now a US district-court ruling. In *Bartz v. Anthropic* (N.D. Cal., order of 23 Jun 2025), Judge Alsup ruled on summary judgment that:
     - training on lawfully acquired books was fair use;
     - building a central library from pirated copies was not fair use.
   - The case then settled, with a reported $1.5B settlement in Sep 2025. The book should state this in one or two neutral sentences, not as legal advice.
   - **Source:** docket *Bartz v. Anthropic PBC*, 3:24-cv-05417 (N.D. Cal.), https://www.courtlistener.com/docket/69058235/bartz-v-anthropic-pbc/. Cite the 23 Jun 2025 summary-judgment order from the docket.

10. **MISSING-MINOR** · §3.3.4 (`:184-229`)
    - **What's missing:** **semantic (embedding) dedup**. GLM-4.5 found template-generated pages that "cannot be removed by MinHash deduplication" and applied SemDedup.
    - **Also missing:** trillion-token **global dedup tooling** (Olmo 3 lists it as a key novelty), and FineWeb2's "rehydration", which uses duplicate counts as an upsampling signal rather than only deleting duplicates.
    - **Also:** the claims "Llama-2 used a Jaccard threshold of 0.8; DeepSeek-V3 uses a similar range" (`:200`) and "The DeepSeek-V3 paper references a custom dedup pipeline" (`:198`) are not supported by those papers as far as I could find.
    - **Sources:** GLM-4.5 §2.2; Olmo 3 §3.4; FineWeb2 (above).

11. **MISSING-MINOR** · §3.3.6 (`:267-275`)
    - **What's missing:** **pre-training safety filtering** has a named, documented example. gpt-oss reused "the CBRN pre-training filters from GPT-4o".
    - **Also missing:** DeepSeek-V4 filters "batched auto-generated and templated content, thereby mitigating the risk of model collapse". This is a new failure mode for §3.6 (the web filling with LLM output).
    - **Sources:** gpt-oss model card §2.4; DeepSeek-V4 §4.1.

12. **WRONG-NOW (never accurate)** · `:306`
    - **Claim:** "The packing is done with attention masks that prevent cross-document attention"
    - **What the sources say:** this is not universal, and the book's own spine model does the opposite.
      - DeepSeek-V3 §4.1: "we implement the document packing method … but do not incorporate cross-sample attention masking."
      - Llama 3 does mask, and says the mask had "limited impact" in standard pre-training.
    - This makes a good "labs disagree" sentence.
    - **Source:** DeepSeek-V3 §4.1; Llama 3 §3.2.

13. **MISSING-MINOR** · `:117-121`
    - **Claim:** language ID uses "`fastText`… ~200 languages … runs at millions of documents per second on a single GPU"
    - **What's missing:** fastText LID is a CPU model, and Llama 3 used it for 176 languages. At the >1,000-language scale, FineWeb2 moved to GlotLID with per-language thresholds. See Unverified leads for the exact classifier name.
    - **Source:** Llama 3 §3.1.1; FineWeb2.

14. **DATED-FRAMING** · `:73`, `:82`
    - **Claim:** "Llama-3 supports 8 languages"; "DeepSeek-V3's reported mix is roughly 60% English, 30% Chinese, 10% other"
    - **What's dated or wrong:**
      - Multilingual coverage has jumped: Qwen3 119, Llama 4 200, Qwen3.5 201 languages.
      - The DeepSeek 60/30/10 split does not appear in the V3 report, which only says it expanded "multilingual coverage beyond English and Chinese". Remove it or mark it as an estimate.
    - **Sources:** Qwen3; Llama 4 blog; Qwen3.5 card; DeepSeek-V3 §4.1.

15. **UNSUPPORTED (flag for check)** · `:377-378`
    - **Claim:** R1 used "RL data: 600K reasoning prompts, plus preference data"
    - **What the source says:** the R1 report's 600K / 200K figures are for **SFT** data. I did not find a 600K RL-prompt figure. Re-check against the Nature version.

### Keep as-is
- §3.1's framing (data as compression; four implications), with its synthetic-data sentence fixed.
- §3.2.1 Common Crawl description.
- §3.2.2 code-source list.
- §3.3.1 HTML extraction challenges.
- §3.3.2 subtle point on LID errors.
- §3.3.3 heuristic filter list.
- The MinHash + LSH mechanics (`:188-196`).
- The suffix-array explanation and Lee et al. attribution (`:202-206`).
- §3.3.5 n-gram contamination mechanics and rephrasing caveat.
- §3.4 pipeline-as-a-system properties (idempotent, checkpointed, versioned, observable).
- §3.6 failure modes: add model collapse, keep the rest.
- §3.7 JD table.

The lab (`lab03_dedup_and_quality`) teaches MinHash/LSH and the negative-class lesson. Both are still correct and still surprising.

### Lab opportunity (optional)
**`lab03b_upsample_vs_filter`** (CPU, ~seconds). Under a *fixed token budget*, compare three data-selection policies using a toy quality score, on a synthetic corpus whose "quality" is known:
- a hard top-40% threshold;
- flat repetition;
- an Olmo-3-style quality-aware upsampling curve (drop the bottom 40%, repeat the top 5% k times).

Train a tiny n-gram or bigram LM per policy and measure held-out loss. The surprising, measurable result the 2025 reports claim: with scarce tokens, *repeating the best data beats filtering harder*, until repetition overfits. The lab can show the crossover epoch count. This teaches the 2025 shift (GLM-4.5's 3.2 epochs, Olmo 3, FineWeb2 rehydration) with a mechanism rather than a citation.

### References to add
- Penedo et al. (Hugging Face). *FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language.* arXiv:2506.20920, 26 Jun 2025.
- Su et al. (NVIDIA). *Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset.* arXiv:2412.02595, 3 Dec 2024 (ACL 2025).
- NVIDIA. *Nemotron-CC-v2* dataset card, 2025, https://huggingface.co/datasets/nvidia/Nemotron-CC-v2
- NVIDIA. *Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training.* arXiv:2504.13161, 17 Apr 2025.
- Li et al. *DataComp-LM: In search of the next generation of training sets for language models.* arXiv:2406.11794, 17 Jun 2024.
- Olmo Team (Ai2). *Olmo 3.* arXiv:2512.13961, 15 Dec 2025.
- Poznanski et al. (Ai2). *olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models.* arXiv:2502.18443, 25 Feb 2025.
- Hugging Face. *FinePDFs* dataset card, 2025, https://huggingface.co/datasets/HuggingFaceFW/finepdfs
- Kandpal et al. *The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text.* arXiv:2506.05209, 5 Jun 2025.
- Essential AI. *Essential-Web v1.0: 24T tokens of organized web data.* arXiv:2506.14111, 17 Jun 2025.
- Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner.* Blog, 8 Jul 2025, https://huggingface.co/blog/smollm3
- Wei et al. (DeepSeek-AI). *DeepSeek-OCR: Contexts Optical Compression.* arXiv:2510.18234, 21 Oct 2025.
- Qwen3 (already ref [6]): cite §3.1–3.2 correctly. Kimi K2, GLM-4.5, DeepSeek-V4 and gpt-oss as in Ch 1.
- *Bartz v. Anthropic PBC*, No. 3:24-cv-05417 (N.D. Cal.), order on summary judgment, 23 Jun 2025 (docket above).

---

## Ch 4 — Tokenization at scale

**Verdict:** Needs rewrite of §4.4 (the vocab-size table and the "settled at 128K–152K" claim), §4.8 (the Qwen3 regex) and §4.11 (all three "real configurations"). The JSON blocks do not match the published `tokenizer.json` files: wrong `byte_fallback`, wrong merge counts, wrong special-token names, and an invented CJK regex class for Qwen3. §4.1–4.3, §4.5 (mechanism), §4.9 and §4.10 age well.

I parsed the official files directly: `deepseek-ai/DeepSeek-V3`, `NousResearch/Meta-Llama-3-8B` (a byte-identical community mirror of the gated Meta repo), `Qwen/Qwen3-8B`, `Qwen/Qwen3.5-397B-A17B`, `openai/gpt-oss-120b`, `moonshotai/Kimi-K2-Instruct` and `deepseek-ai/DeepSeek-V4-Pro`.

### Findings (ranked most-severe first)

1. **WRONG-NOW** · `04-tokenization.md:128`, `:130-138`, `:534`
   - **Claim:** "The frontier has settled on large vocabularies — 128K to 152K"; "The next generation is likely to push further"
   - **What supersedes it:** the next generation did push further. Vocab sizes from official configs and tokenizers:

     | Model | Vocab | Source |
     |---|---|---|
     | Gemma 3 | 262K; `vocab_size` 262,208 | Gemma 3 report; mirror config |
     | Qwen3.5 | 248,044 base, 248,320 padded; "201 languages" | tokenizer.json, config |
     | Llama 4 | 202,048 | config via mirror |
     | gpt-oss | o200k_harmony, 201,088 total | model card §2.3 |
     | MiniMax-M2 | 200,064 | config |
     | Kimi K2 | 163,840 | config |

   - **The counterexample to teach:** DeepSeek-V4 *kept* the V3 tokenizer. It "still remain[s] the vocabulary size to be 128K" (`vocab_size` 129,280), adding only special tokens. That gives the chapter a clean contrast: Qwen changed its tokenizer at a generation boundary, DeepSeek did not.
   - **Sources:**
     - Gemma 3 Technical Report, Google, 25 Mar 2025, https://arxiv.org/abs/2503.19786
     - Qwen3.5-397B-A17B repo, Feb 2026
     - https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E (gated; values read from a mirror's `config.json`)
     - gpt-oss model card, arXiv:2508.10925
     - https://huggingface.co/MiniMaxAI/MiniMax-M2
     - https://huggingface.co/moonshotai/Kimi-K2-Instruct
     - DeepSeek-V4 §4.1

2. **WRONG-NOW (never accurate)** · `:270-284`, `:497`, `:509`
   - **Claim:** "Qwen3 uses a similar but Chinese-extended regex" with a `[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]+` class, plus `\p{N}{1,3}`
   - **What the file says:** the published `Qwen/Qwen3-8B/tokenizer.json` pre-tokenizer has **no CJK class**. It splits digits **one at a time** (`\p{N}`, not `\p{N}{1,3}`). Qwen3.5 keeps single-digit splitting and adds `\p{M}` (combining marks).
   - This invalidates the chapter's central causal story: that Qwen's Chinese efficiency comes from a CJK pre-tokenizer class.
   - The real DeepSeek-V3 file *does* isolate CJK, and more broadly than the book says: `[一-龥぀-ゟ゠-ヿ]+` covers Han *and* kana. It also isolates `\p{N}{1,3}` before the main regex.
   - `:248` says GPT-2's regex "splits on … CJK characters separately". The GPT-2 regex has no CJK class.
   - **Source:** `tokenizer.json` in https://huggingface.co/Qwen/Qwen3-8B, https://huggingface.co/Qwen/Qwen3.5-397B-A17B and https://huggingface.co/deepseek-ai/DeepSeek-V3

3. **WRONG-NOW (never accurate)** · §4.11 (`:423-511`) and `:87`, `:124`, `:132-134`, `:159-161`

   What the three published files actually contain (DeepSeek-V3, Llama-3, Qwen3):

   | Field | DeepSeek-V3 | Llama-3 | Qwen3 |
   |---|---|---|---|
   | `byte_fallback` | false | false | false |
   | vocab entries | 128,000 | 128,000 | 151,643 |
   | merges | 127,741 | 280,147 (HF file) | 151,387 |
   | `vocab_size` (embedding) | 129,280 | 128,256 | 151,936 |

   - **What the book gets wrong about the three models:**
     - All three are byte-level BPE with `byte_fallback: false`. `byte_fallback` is SentencePiece-style fallback, which byte-level BPE does not need.
     - DeepSeek-V3's embedding is 129,280, not 128,000. That changes the ~918M embedding figure.
     - The first added tokens in DeepSeek-V3 are `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<｜▁pad▁｜>` and 800+ placeholders. FIM tokens are named `<｜fim▁hole｜>`, `<｜fim▁begin｜>` and `<｜fim▁end｜>`, not at IDs 3–5. I found no `<|file_sep|>` among its non-placeholder added tokens.
     - The Qwen3 report states a vocab of 151,669 (151,643 BPE entries plus 26 added tokens). The book's 152,064 is Qwen2-era padding.
     - Qwen3's tool tokens are `<tool_call>` / `</tool_call>` (no pipes). It also has `<think>` / `</think>` and FIM tokens `<|fim_prefix|>` etc.
     - Llama 3's special range is 128,000–128,255, which is **256** reserved/special IDs, not 128.
     - Llama 3's vocabulary is "100K tokens from the tiktoken tokenizer with 28K additional tokens" for non-English, per Llama 3 §3.2. The book never says this, and it is the most instructive fact about the Llama 3 tokenizer.
   - **Fix:** replace the three JSON blocks with excerpts copied from the files, or a small script the reader runs to print them.
   - **Sources:** the repos above; Llama 3 §3.2; Qwen3 §2.

4. **WRONG-NOW (never accurate)** · `:140`
   - **Claim:** "Llama-3 uses *tied* embeddings"
   - **What the config says:** `Meta-Llama-3-8B` has `tie_word_embeddings: false`. Qwen3-8B and 235B and DeepSeek-V3 are also untied. Tying shows up in small models, e.g. SmolLM3-3B has `tie_word_embeddings: true`. This is a better statement of the tradeoff.
   - **Source:** the respective `config.json` files.

5. **MISSING-MAJOR** · §4.4 / new section
   - **What's missing:** 2025 research that reframes the vocab-size question. Scaling the *input* vocabulary separately from the output vocabulary has a log-linear effect on loss. This directly challenges §4.4's "diminishing returns" framing.
   - **Source:** Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling, ByteDance Seed, 28 Jan 2025 (ICML 2025), https://arxiv.org/abs/2501.16975

6. **MISSING-MAJOR** · §4.2 / §4.13 (no mention of tokenizer-free or beyond-BPE work)
   - **What's missing:** the 2024–26 research line a reader would expect after "BPE is the workhorse":
     - **Byte Latent Transformer:** dynamic byte patches, "patches scale better than tokens".
     - **H-Net:** dynamic chunking, end-to-end hierarchical modelling without a fixed tokenizer.
     - **SuperBPE:** "superword" tokens that cross whitespace.
     - **Modular per-language tokenizers** (Sep 2026).
   - None of these has displaced BPE in a flagship release I could verify. That is itself the point to make: frontier releases through V4 and Qwen3.5 still ship byte-level BPE.
   - **Sources:**
     - Pagnoni et al. (Meta), *Byte Latent Transformer*, 13 Dec 2024, https://arxiv.org/abs/2412.09871
     - Hwang, Wang, Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, 10 Jul 2025, https://arxiv.org/abs/2507.07955
     - Liu et al., *SuperBPE: Space Travel for Language Models*, 17 Mar 2025, https://arxiv.org/abs/2503.13423
     - *To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs*, 14 Sep 2026, https://arxiv.org/abs/2609.15528

7. **DATED-FRAMING** · §4.10 (`:389-411`), `:538`
   - **Claim:** "you cannot re-tokenize a model"; adding tokens later "requires a partial re-train"
   - **What supersedes it:** still true for *flagship* practice. DeepSeek-V4 inherited V3's tokenizer, and that is good evidence for the chapter's claim. But training-free tokenizer transplantation now exists: reconstructing new embeddings via Orthogonal Matching Pursuit, evaluated Llama→Mistral NeMo and Qwen→Llama.
   - The section should soften "cannot" to "cannot cheaply, and no flagship does".
   - **Source:** Goddard & Fernandes Neto, *Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit*, 7 Jun 2025, https://arxiv.org/abs/2506.06607

8. **MISSING-MINOR** · §4.5 special tokens (`:157-167`)
   - **What's missing:** the 2025 reasoning and agent special-token layer:
     - Qwen3's `<think>`/`</think>` plus `/think` and `/no_think` mode flags.
     - gpt-oss's harmony format tokens (`<|start|>`, `<|message|>`, `<|channel|>`, `<|call|>`, `<|return|>`, …).
     - DeepSeek's tool-call token family.
     - DeepSeek-V4 dropping the Jinja chat template for an `encoding/` Python package.
   - **Sources:** Qwen3 §4.3 (Table 9); gpt-oss model card §2.3; DeepSeek-V4-Pro README.

9. **DATED-FRAMING** · `:192`, `:216`, `:511`, `:535`
   - **Claim:** "Qwen3's 152K vocabulary is heavily Chinese-optimized"; "~24K more tokens … almost all of them Chinese-optimized"
   - **What's wrong:** no primary source supports the "almost all Chinese" attribution. Qwen3 is a 119-language model, and the real gap to DeepSeek-V3 is ~22.6K padded rows (151,936 − 129,280).
   - The 2026 story is better: Qwen3.5 grew the vocabulary to ~248K explicitly for 201-language coverage. The tokenizer can be re-expanded at a generation boundary.
   - The "Chinese is punished" mechanism in §4.6 is sound. Only the Qwen3-specific attribution is unsupported.
   - Gemma 3's report is a primary source for the tradeoff: its new tokenizer "significantly improves the encoding of Chinese, Japanese and Korean text, at the expense of a slight increase of the token counts for English and Code".
   - **Sources:** Qwen3 §2; Qwen3.5 card; Gemma 3 report §2.

10. **WRONG-NOW (never accurate)** · `:413`
    - **Claim:** "The DeepSeek-V3 team's choice of 128K is partly because they started from a 32K Llama-1-style tokenizer and grew it as the model grew"
    - **What the source says:** not supported by the V3 report. See Unverified leads for the DeepSeek LLM vocab. Remove it, or source it.

11. **MISSING-MINOR** · `:12`
    - **Claim:** a 30% token increase would "have cost roughly an extra month"
    - **What's dated:** the arithmetic is fine, but the anchor is V3. With V4's >32T tokens the same argument is stronger. Optional update.

### Keep as-is
- §4.1 (why tokenization matters; compression ratio and fertility definitions).
- §4.2 BPE algorithm and worked example.
- §4.3 Unigram/SentencePiece.
- §4.4's *arguments* for and against large vocabularies. Keep the arguments, replace the table.
- §4.5 byte-level vocabulary completeness and new-embedding initialisation options.
- §4.6's resource-allocation lesson.
- §4.7 code-tokenization issues.
- §4.8's key-points bullets on possessive quantifiers, `\p{N}{1,3}` (true for Llama-3, gpt-oss and DeepSeek-V3) and `\s+`.
- §4.9 training procedure and SentencePiece and HF snippets, as generic recipes.
- §4.10's A/B procedure.
- §4.12 JD table.
- `lab04_train_a_bpe_tokenizer`: the Vietnamese ~1.8× penalty and the regex-deletion result are mechanism-level and still correct.

### Lab opportunity (optional)
- **`lab04b_superwords_and_vocab_scaling`** (CPU, seconds). Extend the existing byte-level BPE lab two ways:
  - **(a)** After N ordinary merges, lift the whitespace pre-tokenization boundary and keep merging, SuperBPE-style. Measure bytes/token on held-out English and on a no-space language. The surprise: most of the gain comes from a small number of frequent multi-word merges, and it varies sharply by language.
  - **(b)** Sweep vocab 8K→256K on the lab corpus, and plot compression gain against embedding-row cost and against mean training count per token. This makes the 128K→262K industry move measurable.
- **Smaller, higher-value fix:** ship a `print_tokenizer_facts.py` that downloads the public `tokenizer.json` / `config.json` files and prints `byte_fallback`, merges, vocab, `vocab_size`, the pre-tokenizer regex and added tokens. §4.11 could then be generated, not hand-written, and would never drift again.

### References to add
- Gemma Team (Google). *Gemma 3 Technical Report.* arXiv:2503.19786, 25 Mar 2025.
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card.* arXiv:2508.10925, 5–8 Aug 2025 (o200k_harmony, 201,088 tokens).
- Qwen Team. *Qwen3.5-397B-A17B* model card and tokenizer, Feb 2026, https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Huang et al. (ByteDance Seed). *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling.* arXiv:2501.16975, 28 Jan 2025.
- Pagnoni et al. (Meta). *Byte Latent Transformer: Patches Scale Better Than Tokens.* arXiv:2412.09871, 13 Dec 2024.
- Hwang, Wang, Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* arXiv:2507.07955, 10 Jul 2025.
- Liu et al. *SuperBPE: Space Travel for Language Models.* arXiv:2503.13423, 17 Mar 2025.
- Goddard, Fernandes Neto. *Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit.* arXiv:2506.06607, 7 Jun 2025.
- *To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs.* arXiv:2609.15528, 14 Sep 2026.
- DeepSeek-V4 (§4.1, tokenizer retained at 128K), Kimi K2 (config: 163,840), Llama 4 (config: 202,048), as above.

---

## Unverified leads

Not confirmed from a primary source in this pass. Do not cite without checking.

- **Qwen3.5 "10–60%" encoding-efficiency gain** from the new ~250K vocab. The figure appears in search snippets attributed to the Qwen3.5 blog (https://qwen.ai/blog?id=qwen3.5). The page is JS-rendered, so I could not read it.
  - **Verified:** the tokenizer size (248,044 base entries; `vocab_size` 248,320) and "201 languages" (HF model card).
- **Qwen3.5 exact release date.** The HF repo was created 16 Feb 2026, but I did not confirm the announcement date.
- **Meta's 2025 reorganisation** into "Meta Superintelligence Labs", and whether FAIR or GenAI owned Llama. This is relevant to Ch 2 §2.2/§2.7. I found no Meta-published primary source in this pass.
- **FineWeb2's language-ID model.** I believe it is GlotLID, but the name is from memory and third-party summaries. Confirm in arXiv:2506.20920 §3.
- **SuperBPE's headline numbers** (tokens saved, downstream gain). I verified only the title and date.
- **DeepSeek LLM (Jan 2024) vocab size** (~100K?), which bears on the Ch 4 `:413` claim. Not checked.
- **"Llama-3 … detailed contamination check against 32 benchmarks"** (Ch 3 `:367`). Not located in the Llama 3 paper in this pass.
- **Anthropic settlement amount and approval status** in *Bartz*. Search results report $1.5B (Sep 2025). Confirm against the docket entry before printing a figure.
- **Kadrey v. Meta** (N.D. Cal., June 2025) as a second fair-use ruling on book training. Not checked against the docket.
- **DeepSeek-V4 pre-training GPU count, GPU-hours and cost.** I did not find these in the searched sections of arXiv:2606.19348. Ch 1's "$5.5M"-style framing has no V4 equivalent that I confirmed.
- **The Llama 4 vocab (202,048) came from an unofficial mirror** (`unsloth/Llama-4-Scout-17B-16E-Instruct`), because the official repo is gated. It matches the token-ID ranges in that config. Confirm against the gated meta-llama repo before printing.
- **arXiv:2601.11659, "The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes"**, is **withdrawn** and not a Meta publication. Do not cite it.
