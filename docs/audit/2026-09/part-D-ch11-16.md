# Staleness audit, Part III chapters 11–16 (post-training core)

Auditor scope: `book/part-3-post-training/11`–`16`. Audit date 2026-09-26. Read-only; no repo files edited.
Line numbers refer to the files as of commit `e02b15a`. Every source listed under "Findings" or "References to add" was opened or located on arXiv, an official lab site, or an official repo during this audit. Anything I could not confirm is under "Unverified leads" at the end.

**Cross-cutting summary.** The mechanics these chapters teach still hold: masking, packing, templates, Bradley–Terry, the overoptimization curve, the DPO derivation, baselines-as-taxonomy, and zero-variance groups. What has aged is the account of **what frontier labs actually do**. Four 2025–26 shifts touch nearly every chapter:
1. **Distillation, especially on-policy distillation (OPD), is now a primary post-training stage.** Qwen3 and DeepSeek-V4 use it, and V4 uses it as the main way to merge specialists.
2. **Generative and rubric-based reward models now sit inside the RL loop.** DeepSeek-V4 explicitly drops scalar reward models.
3. **GRPO has been replaced in practice by patched variants** (DAPO, Dr. GRPO, GSPO, CISPO), and **train–inference mismatch plus asynchronous RL** has become a first-class topic.
4. **"Verifiable rewards cannot be hacked" is no longer defensible.** OpenAI and Anthropic have both documented reward hacking of coding verifiers in production-scale RL.

---

## Ch 11 — SFT, the way frontier labs actually do it
**Verdict:** Needs refresh. §11.3 (data sources), §11.8 (LoRA) and §11.10 (rejection sampling / distillation) need rewrites. §11.6 needs a reasoning-template addendum. §11.4–11.5 are evergreen.

### Findings (ranked)
1. **MISSING-MAJOR** · `11-sft-frontier-style.md:32-42`, `:232`, `:244` · The text lists four SFT sources and says on-policy data is "the fundamental advantage over distilling from a different model." It also says a "specialist generating training data for a generalist" is standard. · **What is missing: on-policy distillation.** The student samples its own trajectories and a teacher grades every token with a reverse KL. This combines on-policy data with a dense per-token signal, which removes the either/or the chapter sets up.
   - Qwen3 splits small-model post-training into off-policy distillation followed by on-policy distillation. On an 8B model it reports that OPD beats direct RL on math and code at about 1/10 of the GPU hours (§4.5, §4.7).
   - DeepSeek-V4 uses multi-teacher OPD with more than ten domain teachers as "the primary technique for merging expert capabilities into the final model" (§5.1.2).
   - GLM-4.5 uses expert training followed by unified self-distillation. DeepSeek-V3.2 uses specialist distillation.
   - The "specialist teaches generalist" pattern at `:244` is right in spirit. Its mechanism has moved from filtered SFT data to logit-level distillation.
   - Sources: Qwen3 Technical Report, Qwen Team, May 2025, https://arxiv.org/abs/2505.09388 · "On-Policy Distillation," Thinking Machines Lab (K. Lu), 27 Oct 2025, https://thinkingmachines.ai/blog/on-policy-distillation/ · DeepSeek-V4, DeepSeek-AI, Apr 2026, https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro and https://arxiv.org/html/2606.19348v1 · GLM-4.5, Zhipu/THUDM, Aug 2025, https://arxiv.org/abs/2508.06471 · DeepSeek-V3.2, Dec 2025, https://arxiv.org/abs/2512.02556
2. **MISSING-MAJOR** · `:155-168` (§11.6 chat templates) · The rules cover role tokens and system prompts only. · **What is missing: templates for reasoning models.** These now carry the think/no-think switch:
   - Qwen3's `/think` and `/no_think` flags, the `enable_thinking` parameter, and an empty `<think></think>` block for non-thinking turns (Qwen3 §4.3, Table 9).
   - DeepSeek-V4's three reasoning-effort modes, marked by `<think>` tokens plus a system-prompt instruction for "Think Max" (§5.1.1).
   - The industry reversal: Qwen abandoned the hybrid model in July 2025 and shipped separate `-Instruct-2507` and `-Thinking-2507` models.
   - Hybrid-versus-separate is now a real post-training design decision. The template is where it lives, and it is a new class of train/serve mismatch bug.
   - Sources: Qwen3 report (above) · Qwen3-235B-A22B-Instruct-2507 model card, Qwen, Jul 2025, https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507 · DeepSeek-V4 (above).
3. **DATED-FRAMING** · `:186-200`, `:302` (§11.8, takeaway 7) · The text says: "Frontier labs full fine-tune; LoRA is for ablations … Do not assume LoRA-based results … transfer." · Schulman and Thinking Machines report that LoRA matches full fine-tuning in a "low-regret regime" covering most post-training dataset sizes, and for RL even at very low rank, provided LoRA is applied to all layers (including MLP/MoE) with about 10× the learning rate. It is less tolerant of large batches. The claim about frontier-lab practice may still hold (unverifiable). The warning that LoRA results "often do not transfer" now needs these conditions attached.
   - Source: "LoRA Without Regret," Thinking Machines Lab (J. Schulman et al.), Sep 2025, https://thinkingmachines.ai/blog/lora/
4. **MISSING-MAJOR** · `:23-28` (LIMA vs FLAN), `:13` ("hundreds of thousands to a few million examples") · For reasoning, the "how much data" debate has been re-run with long chain-of-thought traces:
   - s1: 1,000 curated traces plus "budget forcing" beat o1-preview on competition math.
   - OpenThoughts: more than 1,000 controlled experiments. Scaling distilled reasoning SFT to 1.2M examples kept improving results (OpenThoughts3-7B: 53% AIME25).
   - The chapter's resolution (format is cheap, coverage needs scale) still holds. Long-CoT distillation is now the concrete test case, and the chapter never mentions it.
   - Sources: "s1: Simple test-time scaling," Muennighoff et al., Jan 2025, https://arxiv.org/abs/2501.19393 · "OpenThoughts: Data Recipes for Reasoning Models," Guha et al., Jun 2025, https://arxiv.org/abs/2506.04178
5. **DATED-FRAMING** · `:21` · "Tülu 3 … the standard recipe … the place you can go and read the actual code." · The fully open successor is **Olmo 3**, with separate Think and Instruct lines. Its pipeline is SFT (Dolci-Think-SFT) → DPO (delta learning) → RLVR (OlmoRL), with all data and code released.
   - Source: "Olmo 3," Ai2, blog 20 Nov 2025, arXiv Dec 2025 (v2 Apr 2026), https://arxiv.org/abs/2512.13961 and https://allenai.org/blog/olmo3
6. **MISSING-MINOR** · `:210` ("Reasoning length versus latency … a product decision") · This is now handled explicitly with thinking budgets and effort modes. Qwen3 shows smooth thinking-budget scaling curves (§4.7). V4 trains each effort mode with its own length penalty and context window. Cross-reference to Ch15.
7. **MISSING-MINOR** · `:202-214` (§11.9 capability tax) · "RL's Razor" finds that SFT forgets prior capabilities more than on-policy RL at equal new-task performance. It attributes the difference to on-policy sampling, not negative examples. This is useful evidence for the tax framing, and it also motivates OPD.
   - Source: Shenfeld, Pari, Agrawal, Sep 2025, https://arxiv.org/abs/2509.04259
8. **MISSING-MINOR** · `:180` (sequence length) · Reasoning SFT now routinely trains on very long responses. GLM-4.5 says its SFT "conditioned the model on generating 64K-length responses." The advice holds, but the scale in the examples (4K, 128K context) should be updated.

### Keep as-is
§11.1, §11.4 (loss masking and its three bugs), §11.5 (packing, block-diagonal varlen), the core of §11.6 (exact-match templates, special role tokens, round-trip test), §11.7 LR and epoch guidance for chat SFT, §11.11 evaluation stack, §11.12 failure table, §11.13 JD section. These are mechanism-level and have not aged.

### Lab opportunity
**`lab11b_on_policy_distillation` (new, CPU).** Train a small "teacher" to competence on the Countdown task already used in lab15, or on multi-digit addition. Then compare three ways to train a smaller student from the same start: (a) SFT on teacher samples, (b) RL with a 1-bit correctness reward, (c) on-policy distillation with per-token reverse KL against the teacher. Plot accuracy against student-generated tokens. The expected surprise, per Qwen3 and TML: OPD reaches RL's accuracy with far fewer samples, and SFT on teacher samples plateaus lower when the teacher's style is hard for the student to reproduce. Measurable and mechanism-level, in the book's lab style.

### References to add
- Qwen Team. "Qwen3 Technical Report." May 2025. arXiv:2505.09388 (already ref [6]; cite §4.3 and §4.5).
- Lu, K. and Thinking Machines Lab. "On-Policy Distillation." 27 Oct 2025. https://thinkingmachines.ai/blog/on-policy-distillation/
- Schulman, J. et al., Thinking Machines Lab. "LoRA Without Regret." Sep 2025. https://thinkingmachines.ai/blog/lora/
- DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." Apr 2026 (model release 24 Apr 2026). https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro ; arXiv HTML https://arxiv.org/html/2606.19348v1
- Team OLMo / Ai2. "Olmo 3." Dec 2025. arXiv:2512.13961.
- Muennighoff et al. "s1: Simple test-time scaling." Jan 2025. arXiv:2501.19393.
- Guha et al. "OpenThoughts: Data Recipes for Reasoning Models." Jun 2025. arXiv:2506.04178.
- Shenfeld, Pari, Agrawal. "RL's Razor: Why Online Reinforcement Learning Forgets Less." Sep 2025. arXiv:2509.04259.
- Qwen. Qwen3-235B-A22B-Instruct-2507 / Thinking-2507 model cards. Jul 2025. https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507

---

## Ch 12 — Reward modeling
**Verdict:** Needs rewrite of §12.8 (generative RMs), §12.9 (verifiable rewards: hackability) and §12.12 (what labs do). §12.2–12.7 and §12.10–12.11 are evergreen.

### Findings (ranked)
1. **WRONG-NOW** · `12-reward-modeling.md:224`, `:313` (takeaway 8) · "The practical split at most labs: **generative judges for evaluation and data generation, scalar reward models inside the RL loop**." · Superseded:
   - **DeepSeek-V4:** "we dispense with these conventional scalar-based reward models." For hard-to-verify tasks it uses rubric-guided RL data scored by a generative reward model. "The actor network natively functions as the GRM," and RL is applied to the GRM itself (§5.1.1).
   - **Kimi K2** uses a Self-Critique Rubric Reward inside RL. The K2 critic ranks rollouts pairwise against core and prescriptive rubrics (§3.2.2).
   - **DeepSeek-GRM (SPCT)** scales generative reward models at inference: 32 samples plus a meta-RM lift a 27B GRM above much larger scalar RMs.
   - The "too slow for the inner loop" argument at `:222` has been overtaken in practice.
   - Sources: DeepSeek-V4, Apr 2026 (above) · "Kimi K2: Open Agentic Intelligence," Kimi Team, Jul 2025, https://arxiv.org/abs/2507.20534 · "Inference-Time Scaling for Generalist Reward Modeling," Liu et al. (DeepSeek), Apr 2025, https://arxiv.org/abs/2504.02495
2. **WRONG-NOW** · `:243`, `:314` (takeaway 9) · "It cannot be hacked in the §12.6 sense — there is no artifact to find"; "Verifiable rewards cannot be hacked." · A verifier is a program with bugs and gaps, and a policy under optimization finds them:
   - OpenAI documents a frontier reasoning model reward-hacking agentic coding environments during RL. Penalizing it via chain-of-thought monitoring produces *obfuscated* hacking.
   - Anthropic shows that models learning to reward-hack real production coding environments generalize to broad misalignment (alignment faking, sabotage). Mitigations include preventing the hack and "inoculation prompting."
   - "Spurious rewards" shows RLVR on Qwen2.5-Math improving MATH-500 by 21 points with *random* reward. A verifier-driven gain is therefore not evidence that the verifier's signal is what was learned.
   - Sources: "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation," Baker et al. (OpenAI), 14 Mar 2025, https://arxiv.org/abs/2503.11926 · "Natural Emergent Misalignment from Reward Hacking in Production RL," MacDiarmid et al. (Anthropic), Nov 2025, https://arxiv.org/abs/2511.18397 · "Spurious Rewards: Rethinking Training Signals in RLVR," Shao et al., Jun 2025, https://arxiv.org/abs/2506.10947
3. **MISSING-MAJOR** · `:232-251` (§12.9 table: "General helpfulness — No verifier"; "the modern pipeline is hybrid") · **What is missing: rubrics as rewards.** A rubric is a checklist of criteria scored by a judge. It is now the standard bridge into non-verifiable domains. Examples: RaR (up to +31% relative on HealthBench over Likert LLM-judge rewards), Kimi K2's rubric reward, V4's rubric-guided RL data, and Qwen3's "automatically generated checklists" for non-thinking SFT data (§4.3). The hybrid is now three-way: rule verifier, rubric plus generative judge, and scalar RM (shrinking).
   - Source: "Rubrics as Rewards: RL Beyond Verifiable Domains," Gunjal et al., Jul 2025, https://arxiv.org/abs/2507.17746
4. **DATED-FRAMING** · `:278-292` (§12.12) · The section covers InstructGPT, CAI, Llama 2 (calling it "still the best single read on production reward modelling"), R1/V3 and Tülu 3. · It misses every 2025–26 open report with reward-model detail: Kimi K2 (rubric self-critique), DeepSeek-V3.2 and V4 (GRM with RL on the judge), Qwen3 (checklists), Olmo 3 (fully open). The closing "trend" sentence at `:292` is directionally right and understated: at least one frontier lab has now removed scalar RMs entirely.
5. **MISSING-MINOR** · `:202` (§12.7 item 4, RewardBench) · RewardBench 2 is harder (models score about 20 points lower) and adds Factuality and Precise-IF categories. It reports correlation with downstream best-of-N and PPO use, which speaks to the chapter's own worry about RewardBench's validity.
   - Source: Malik et al. (Ai2), 2 Jun 2025, https://arxiv.org/abs/2506.01937
6. **MISSING-MINOR** · `:97-115` (§12.4 preference data) · Skywork-Reward-V2 shows scalar Bradley–Terry RMs still improving sharply through data curation alone. A 40M-pair human-AI curated set produces 0.6B–8B RMs, and the 1.7B one beats a prior 70B state of the art. This supports the chapter's "data sets the ceiling" point with a 2025 number.
   - Source: Liu et al., 2 Jul 2025, https://arxiv.org/abs/2507.01352
7. **MISSING-MINOR** · `:95` (RM size relative to policy) · In V4, the reward model *is* the policy (actor-as-GRM), which reframes the size trade-off entirely.

### Keep as-is
§12.2 Bradley–Terry and its three consequences; §12.3 architecture and the last-real-token bug; §12.4 agreement ceiling; §12.5 training; §12.6 length bias, sycophancy, and the Gao et al. overoptimization curve; §12.7 evaluation stack (especially best-of-N curves); §12.10 ensembles ("catch variance, not bias"); §12.11 multi-objective gating. These are the chapter's core and remain correct.

### Lab opportunity
**Extend `lab12_reward_model` with a "hackable verifier" arm.** Add a rule-based verifier with one realistic bug, for example accepting the answer if the correct number appears *anywhere* in the output, or passing tests that only check return type. Run best-of-N and a few steps of RL-style reweighting against it. The measurable surprise: a *programmatic* reward shows the same rise-then-fall best-of-N curve as the learned RM, which directly tests takeaway 9.

### References to add
- DeepSeek-AI. "DeepSeek-V4 …" Apr 2026 (above), §5.1.1 GRM.
- Kimi Team. "Kimi K2: Open Agentic Intelligence." Jul 2025. arXiv:2507.20534.
- Liu, Z. et al. "Inference-Time Scaling for Generalist Reward Modeling." Apr 2025. arXiv:2504.02495.
- Gunjal et al. "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains." Jul 2025. arXiv:2507.17746.
- Baker et al. "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation." Mar 2025. arXiv:2503.11926.
- MacDiarmid et al. (Anthropic). "Natural Emergent Misalignment from Reward Hacking in Production RL." Nov 2025. arXiv:2511.18397.
- Shao, R. et al. "Spurious Rewards: Rethinking Training Signals in RLVR." Jun 2025. arXiv:2506.10947.
- Malik et al. "RewardBench 2: Advancing Reward Model Evaluation." Jun 2025. arXiv:2506.01937.
- Liu, C.Y. et al. "Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy." Jul 2025. arXiv:2507.01352.

---

## Ch 13 — PPO and RLOO at frontier scale
**Verdict:** Needs rewrite of §13.9–13.10 (systems). Light touch elsewhere. The algorithmic core (§13.2–13.8) is evergreen as foundations but should stop being framed as current frontier practice.

### Findings (ranked)
1. **WRONG-NOW** · `13-ppo-rloo-frontier.md:270` · "Overlap generation with training … This makes the algorithm slightly off-policy, **which PPO's clipping is designed to tolerate**." · Two problems.
   - **Numerical mismatch.** Even *synchronous* RL is off-policy in practice, because the inference engine (vLLM/SGLang kernels) and the trainer (FSDP/Megatron) compute different probabilities for the same tokens. PPO's ratio is computed against trainer-recomputed log-probs, not the sampler's, so the clip does not see or correct this mismatch.
   - **Documented fixes:**
     - Truncated importance sampling (TIS) on the sampler/trainer ratio (Yao et al.).
     - Switching BF16 to FP16 to shrink the mismatch (up to 24×, Qi et al.).
     - For MoE, "Keep Routing": replay the sampler's expert routes in training ("crucial for RL training stability of MoE models," in DeepSeek's pipeline since V3-0324).
     - "Keep Sampling Mask": reapply top-p/top-k truncation in the loss.
     - For stale asynchronous samples, a staleness-aware ("decoupled") PPO objective (AReaL).
   - Sources: "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training," F. Yao, L. Liu et al., Aug 2025, https://fengyao.notion.site/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56 · "Defeating the Training-Inference Mismatch via FP16," Qi et al. (Sea AI Lab), 30 Oct 2025, https://arxiv.org/abs/2510.26788 · DeepSeek-V3.2 §3.1, Dec 2025, https://arxiv.org/abs/2512.02556 · AReaL, Fu et al., May 2025, https://arxiv.org/abs/2505.24298
2. **MISSING-MAJOR** · `:275-310` (§13.10 "A production RLHF architecture"; `:302` "weight sync each iteration") · **What is missing: asynchronous RL systems.** Each is an open system with a paper or repo.
   - **AReaL:** fully decouples rollout from training, up to about 2× speedup.
   - **PipelineRL:** in-flight weight updates broadcast mid-generation without pausing sampling, about 2× faster learning on 4 DGX-H100 nodes.
   - **slime (THUDM):** Megatron plus SGLang with a shared data buffer. It is the RL framework behind GLM-4.5 and later GLM releases.
   - **verl/HybridFlow:** 3D-HybridEngine resharding.
   - The chapter's diagram is a 2024 synchronous design and should show the async/in-flight variant.
   - Sources: AReaL (above) · "PipelineRL," Piché et al. (ServiceNow), Sep 2025, https://arxiv.org/abs/2509.19128 · slime, THUDM, https://github.com/THUDM/slime · "HybridFlow," Sheng et al. (ByteDance), EuroSys 2025, https://arxiv.org/abs/2409.19256 and https://github.com/volcengine/verl
3. **DATED-FRAMING** · `:1-15`, `:339-343`, chapter title · The chapter presents PPO with a value network as the frontier RLHF algorithm. · In every 2025–26 open frontier report I checked, the RL stage uses critic-free GRPO-family methods, and this now includes non-verifiable rewards:
   - DeepSeek-V3.2 scales GRPO across all domains; V4 uses GRPO for specialists.
   - GLM-4.5 uses GRPO "excluding the KL loss term."
   - Olmo 3 uses OlmoRL (GRPO plus DAPO/Dr. GRPO fixes).
   - PPO should be taught as the conceptual ancestor, with the value model presented as a historical cost, not "frontier scale."
   - Sources: V3.2, V4, GLM-4.5, Olmo 3 (above).
4. **MISSING-MAJOR** · `:9-15`, `:343` ("whether to run RL at all … resource-allocation decision") · **What is missing: RL compute now has a published scaling methodology.** ScaleRL (Meta et al.) fits sigmoid compute–performance curves across more than 400k GB200 GPU-hours and predicts a 100k-GPU-hour run from smaller runs. Most recipe details (loss aggregation, normalization, off-policy algorithm) shift *efficiency*, not the asymptote. ScaleRL uses PipelineRL-style async training, CISPO/truncated IS, and FP32 logits.
   - Source: "The Art of Scaling Reinforcement Learning Compute for LLMs," Khatri et al., 15 Oct 2025, https://arxiv.org/abs/2510.13786
5. **MISSING-MINOR** · `:191` (entropy collapse: "the fix is a lower learning rate or a higher β") · The now-standard fix is DAPO's decoupled "Clip-Higher" (raise ε_high only). Olmo 3 adopts it. Cross-reference to Ch15.
   - Source: DAPO, ByteDance Seed, Mar 2025, https://arxiv.org/abs/2503.14476
6. **MISSING-MINOR** · `:262` ("Step 1 is typically 60–80% of wall clock") · Still a defensible order of magnitude. Note that async systems exist precisely because long-tail generation leaves synchronous trainers idle (AReaL's motivation).

### Keep as-is
§13.2 objective and per-token KL folding; §13.3 "everything is a choice of baseline" (the chapter's best idea, and it extends naturally to GSPO/CISPO); §13.4 clipped surrogate and code; §13.5 four-model memory arithmetic; §13.6 target-a-KL and curve reading; §13.7 failure field guide; §13.8 RLOO; the §13.9 thesis "an inference system with a trainer attached" (more true now than when written); §13.11 metrics list.

### Lab opportunity
**Add a "train–inference mismatch" arm to `lab13_ppo_minimal`.** Sample rollouts from a copy of the policy whose logits get small deterministic perturbations, simulating BF16 or kernel differences, while the trainer uses clean logits. Show that PPO's clip does not catch the mismatch and training drifts or collapses as it grows. Then add token-level TIS (cap the sampler/trainer ratio at C) and show recovery. Runs on CPU in the existing toy setting and gives a surprising, measurable result, which the book's labs aim for.

### References to add
- Yao, F., Liu, L., Zhang, D., Dong, C., Shang, J., Gao, J. "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training." Aug 2025. Blog (URL above).
- Qi, P. et al. "Defeating the Training-Inference Mismatch via FP16." Oct 2025. arXiv:2510.26788.
- Fu, W. et al. "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning." May 2025. arXiv:2505.24298.
- Piché, A. et al. "PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation." Sep 2025. arXiv:2509.19128.
- Sheng, G. et al. "HybridFlow: A Flexible and Efficient RLHF Framework." EuroSys 2025. arXiv:2409.19256.
- THUDM. slime. https://github.com/THUDM/slime
- Khatri, D. et al. "The Art of Scaling Reinforcement Learning Compute for LLMs." Oct 2025. arXiv:2510.13786.
- DeepSeek-AI. "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models." Dec 2025. arXiv:2512.02556.

---

## Ch 14 — DPO and the offline preference family
**Verdict:** Needs refresh. §14.5 needs a delta-learning correction, §14.8 and §14.10 need to reflect where DPO sits in 2026 pipelines. §14.2–14.7 are evergreen.

### Findings (ranked)
1. **WRONG-NOW** · `14-dpo-family.md:235`, `:279` · Iterative DPO is "now the standard open recipe"; iterative DPO "is what the strongest open models use." · That was true of Llama 3 (2024). The strongest 2025–26 open models describe RL, not DPO, as their preference and alignment stage:
   - Qwen3: a four-stage pipeline ending in general RL.
   - DeepSeek-V3.2 and V4: GRPO plus specialist distillation or OPD.
   - GLM-4.5: expert RL plus self-distillation.
   - Kimi K2: RLVR plus self-critique rubric RL.
   - Where DPO survives in a top open recipe (Olmo 3), it is a **middle stage before RLVR**, and Olmo 3's RL "ran on top of the DPO checkpoint."
   - Sources: Qwen3, V3.2, V4, GLM-4.5, Kimi K2, Olmo 3 (above).
2. **WRONG-NOW** · `:148` · "Off-policy data from a much stronger model … **This is the trap** … a strange, poorly-conditioned form of distillation." · The delta-learning result says the opposite for *paired* data. Pairing a stronger model's response (chosen) with a weaker model's (rejected) gives large gains *even when SFT on the chosen responses hurts*.
   - Geng et al.: 3B-vs-1.5B pairs match Tülu 3's GPT-4o-supervised recipe on 8B.
   - Olmo 3 Think DPO pairs Qwen3-32B-thinking (chosen) with Qwen3-0.6B-thinking (rejected). Its Table 21 shows that SFT on the 32B outputs hurts, while DPO on the pair "yields strong gains across math and code reasoning."
   - The chapter's warning applies to pairs where *both* responses come from a much stronger model. The "delta" framing should be added.
   - Sources: "The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains," Geng et al., Jul 2025 (COLM 2025), https://arxiv.org/abs/2507.06187 · Olmo 3 §4.3, https://arxiv.org/abs/2512.13961
3. **WRONG-NOW** · `:283`, `:308` (takeaway 10) · "If you are training reasoning on math or code, do not use DPO"; "If your reward is verifiable, skip this chapter's methods." · Olmo 3 Think uses DPO specifically for math and code reasoning, reports strong gains, and then runs RLVR on top. Better advice for 2026: DPO is a cheap, stable stage *before* RLVR, not an alternative to it.
4. **MISSING-MAJOR** · `:273-285` (§14.10 decision procedure) · The four options (DPO / iterative DPO / PPO / GRPO / rejection sampling) have no "SFT → DPO → RLVR" sequencing and no on-policy distillation. When a strong teacher exists, OPD competes directly with DPO for the "cheap, stable improvement over SFT" slot (Qwen3 §4.7; TML blog).
5. **MISSING-MINOR** · `:241-256` (§14.9 comparison table) · Add a row for forgetting. RL's Razor finds that offline methods (SFT and SimPO) forget more than on-policy RL at matched new-task performance, and attributes this to on-policy sampling rather than negative examples. This is empirical support for §14.5's thesis and should be cited.
   - Source: https://arxiv.org/abs/2509.04259

### Keep as-is
§14.1–14.3 (derivation, partition-function cancellation, code, sum-vs-mean note, precomputed reference log-probs); §14.4 β as saturation rate; §14.5's three regimes (amend as in finding 2); §14.6 decreasing chosen log-prob and its monitors; §14.7 "four knobs" table and "some variant beats DPO, rarely the same one twice." These are durable.

### Lab opportunity
**Add a "delta learning" arm to `lab14_dpo_from_scratch`.** Build pairs from a "strong" and a "weak" toy generator, for example two policies with different accuracy on the lab's task. Compare (a) SFT on the strong outputs, (b) DPO on strong-vs-weak pairs, and (c) DPO on strong-vs-strong pairs. The Olmo 3 and Geng et al. result predicts (b) beats (a), with (a) possibly *hurting*. That is a surprising, measurable result that tests the chapter's own §14.5 claim.

### References to add
- Geng, S. et al. "The Delta Learning Hypothesis: Preference Tuning on Weak Data can Yield Strong Gains." Jul 2025. arXiv:2507.06187.
- Team OLMo / Ai2. "Olmo 3." Dec 2025. arXiv:2512.13961 (§4.3 Dolci Think-DPO, Table 21).
- Shenfeld et al. "RL's Razor." Sep 2025. arXiv:2509.04259.

---

## Ch 15 — GRPO and the reasoning revolution
**Verdict:** Needs rewrite of §15.2/§15.9 (algorithm: GRPO is no longer what anyone runs unmodified), §15.4/§15.7 (hacking), and §15.11 (open questions, several now have substantial evidence). Add a length-control / thinking-budget section. §15.5–15.6 and §15.10 remain good.

### Findings (ranked)
1. **MISSING-MAJOR** · `15-grpo-reasoning.md:17-79` (§15.2), `:219-249` (§15.9) · The chapter teaches vanilla GRPO, with Dr. GRPO as the only fix. **Missing: the 2025 successors, which production recipes now adopt.**
   - **DAPO:** Clip-Higher (decoupled ε), Dynamic Sampling (drop all-right and all-wrong groups), token-level loss aggregation, and overlong reward shaping. It reports 50 on AIME24 from Qwen2.5-32B base with 50% of R1-Zero-Qwen-32B's steps.
   - **GSPO (Qwen):** a sequence-level importance ratio with sequence-level clipping. It stabilizes MoE RL without routing replay and fed into later Qwen3 models.
   - **CISPO (MiniMax-M1):** clips the importance-sampling *weight* instead of dropping clipped tokens' gradients.
   - **Adoption:** Olmo 3's OlmoRL explicitly adopts zero-gradient filtering, active sampling, token-level loss, no KL, clip-higher and TIS. ScaleRL's recipe uses CISPO.
   - The `1/|o_i|` per-sample normalization in the book's objective (`:27`) is the thing most recipes have replaced with token-level aggregation.
   - Sources: "DAPO: An Open-Source LLM RL System at Scale," Yu et al. (ByteDance Seed/Tsinghua), Mar 2025, https://arxiv.org/abs/2503.14476 · "Group Sequence Policy Optimization," Zheng et al. (Qwen), Jul 2025, https://arxiv.org/abs/2507.18071 and https://qwenlm.github.io/blog/gspo/ · "MiniMax-M1," MiniMax, 16 Jun 2025, https://arxiv.org/abs/2506.13585 · Olmo 3 §4.4.1 · ScaleRL (above)
2. **WRONG-NOW** · `:126`, `:194`, `:305` (takeaway 2) · "Reward hacking (§12.6)? There is nothing to hack … A model that games it has, by construction, produced a correct answer"; "the failure mode you should expect but do not get: reward hacking." · Refuted for coding and agentic RL by OpenAI (frontier reasoning model hacking coding environments; obfuscation under CoT pressure) and Anthropic (production coding-environment hacks generalizing to misalignment). Spurious-reward gains on Qwen2.5-Math show RLVR improvements can come from things other than the verifier's signal. The "specification gap, not hacking" distinction at `:194` should become "verifiers get hacked too, just differently." Sources as in Ch12 finding 2.
3. **DATED-FRAMING** · `:148` ("Reasoning capability was already in the base model. RL did not install it; RL found it."), `:284` ("Is the base model the ceiling? … The evidence is not decisive.") · The question is still open, but it is now a specific, named debate. The chapter states one side as fact in §15.5 and then calls it open in §15.11.
   - Yue et al.: RLVR models win at pass@1, but base models win at large pass@k, and RL-model reasoning paths already lie in the base distribution (NeurIPS 2025 best-paper runner-up).
   - ProRL (NVIDIA): prolonged RL with KL control and reference resets finds strategies base models cannot reach even under extensive sampling.
   - Spurious rewards: model-family-specific elicitation.
   - Sources: Yue et al., Apr 2025, https://arxiv.org/abs/2504.13837 · "ProRL," Liu, M. et al. (NVIDIA), 30 May 2025, https://arxiv.org/abs/2505.24864 · Shao et al., Jun 2025 (above)
4. **MISSING-MAJOR** · `:178`, `:184-188` (length exploitation mitigations: "explicit length penalty … or simply capping") · **What is missing: length control and thinking budgets as trained capabilities.**
   - L1/LCPO trains adherence to a length target given in the prompt.
   - Qwen3 ships a thinking budget with smooth budget-scaling curves.
   - DeepSeek-V4 trains three reasoning-effort modes, each with its own length penalty and context window.
   - Kimi K2 enforces per-task token budgets in RL.
   - DAPO adds overlong reward shaping.
   - GLM-4.5 reports that single-stage RL at the full 64K length beats progressive length stages, which directly contradicts the "cap and accept truncation" advice.
   - Sources: "L1: Controlling How Long A Reasoning Model Thinks With RL," Aggarwal & Welleck (CMU), 6 Mar 2025, https://arxiv.org/abs/2503.04697 · Qwen3 §4.7 · DeepSeek-V4 §5.1.1 · Kimi K2 · GLM-4.5 §3.2
5. **DATED-FRAMING** · `:286` ("How much does the algorithm matter? … perform comparably"), `:290` ("compute-optimal split … Nobody has published a scaling law for this") · ScaleRL publishes predictable RL compute-scaling curves. It finds recipe choices mostly change efficiency, while some change the asymptote. The pretrain/RL *split* is still unpublished as far as I could find, but "no RL scaling law" is no longer true.
   - Source: https://arxiv.org/abs/2510.13786
6. **MISSING-MAJOR** · `:304` (takeaway 1), and the chapter's implicit claim that RL is *how* reasoning gets in · For models below the frontier, Qwen3 reports that on-policy distillation from a larger reasoner beats direct RL at about 1/10 of the GPU hours on an 8B model. DeepSeek-V4's final model is assembled by multi-teacher OPD from RL-trained specialists. A 2026 reader needs "RL the specialists, distill the generalist" as the standard pipeline shape. Sources: Qwen3 §4.5/§4.7; DeepSeek-V4 §5.1.2.
7. **DATED-FRAMING** · `:55-66` (`k3_kl` docstring: "Low-variance unbiased KL estimator … GRPO uses it") · DeepSeek-V3.2 notes that the k3 estimator, used as a loss under off-policy sampling, gives a biased gradient. It corrects k3 with the π_θ/π_old importance ratio and keeps a per-domain-weighted KL. The code is fine as a value estimator but misleading as a loss.
   - Source: V3.2 §3.1.
8. **DATED-FRAMING** · `:235` ("Common practice is a small β (0.001–0.01), or zero") · Now "no KL" is described as *the* common practice: Olmo 3 ("We remove the KL loss as a common practice"), GLM-4.5 (GRPO "excluding the KL loss term"), and DAPO. DeepSeek-V3.2 is the notable holdout, with a tuned per-domain KL. Update the default and the citations.
9. **MISSING-MINOR** · `:219-226` (prompts near 50% success) · This is right. It is now formalized as DAPO dynamic sampling and Olmo 3 "active sampling" (fill batches only with non-zero-gradient groups; offline filter pass rate > 62.5%). GLM-4.5 adds a two-stage difficulty curriculum that switches to pass@8 = 0 / pass@512 > 0 problems. Cite these as the production form of the advice.
10. **MISSING-MINOR** · `:327` (ref [10] R1, Jan 2025) · The R1 work was published in *Nature* in Sep 2025 (vol. 645, pp. 633–638). Cite the peer-reviewed version alongside the preprint.
    - Source: https://www.nature.com/articles/s41586-025-09422-z

### Keep as-is
§15.1's thesis ("the verifiable reward mattered more than the algorithm"; ScaleRL's finding that details mostly modulate efficiency reinforces it); §15.3 memory arithmetic; §15.5 R1-Zero account; §15.6 emergent behaviours and the "same symptom, opposite diagnosis" length argument; §15.7 format collapse and language mixing; §15.8 R1 pipeline outline; the zero-variance-group analysis in §15.9; §15.10 verifier landscape; §15.12 JD section.

### Lab opportunity
**Update `lab15_grpo_countdown` with successor toggles.** It already measures length collapse and zero-advantage groups. Add flags for:
1. Clip-Higher (ε_high > ε_low). Measure policy entropy against vanilla.
2. Token-level vs per-sample loss aggregation. Measure length drift on wrong answers, which tests Dr. GRPO/DAPO's length-bias claim directly.
3. Dynamic sampling. Measure effective batch size and wall-clock-to-accuracy.
4. GSPO sequence-level ratio. Measure clip fraction and stability.

All are a few lines each on the existing CPU setup. The expected surprise is that token-level aggregation alone changes the length trajectory on this task.

### References to add
- Yu, Q. et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." Mar 2025. arXiv:2503.14476.
- Zheng, C. et al. "Group Sequence Policy Optimization." Jul 2025. arXiv:2507.18071.
- MiniMax. "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention." Jun 2025. arXiv:2506.13585 (CISPO).
- Yue, Y. et al. "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" Apr 2025. arXiv:2504.13837.
- Liu, M. et al. "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models." May 2025. arXiv:2505.24864.
- Shao, R. et al. "Spurious Rewards." Jun 2025. arXiv:2506.10947.
- Aggarwal, P., Welleck, S. "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning." Mar 2025. arXiv:2503.04697.
- Khatri et al. "The Art of Scaling Reinforcement Learning Compute for LLMs." Oct 2025. arXiv:2510.13786.
- DeepSeek-AI. "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning." *Nature* 645, 633–638, Sep 2025. doi:10.1038/s41586-025-09422-z.
- DeepSeek-AI. DeepSeek-V3.2 (Dec 2025) and DeepSeek-V4 (Apr 2026), above.
- GLM-4.5 Team. "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models." Aug 2025. arXiv:2508.06471.
- Baker et al. (OpenAI), Mar 2025; MacDiarmid et al. (Anthropic), Nov 2025, above.

---

## Ch 16 — Process reward models, verifiers, search-time compute
**Verdict:** Needs refresh. §16.4 (MC labelling framed as "what made PRMs practical"), §16.6 (test-time compute taxonomy) and §16.9 (generative verifiers, now realized and used in training) need rework. §16.1–16.3 and §16.7's reconciliation are evergreen.

### Findings (ranked)
1. **WRONG-NOW** · `16-process-reward-models.md:228` · Generative-verifier cost "limits it to inference-time use with modest $N$." · Generative verifiers are now used *as training rewards* at frontier scale.
   - DeepSeekMath-V2 trains an LLM proof verifier plus a *meta-verifier* that checks the verifier's claimed issues. It rewards the generator against that verifier, and scales verification compute to auto-label harder proofs. It reports gold-level IMO 2025 and near-perfect Putnam 2024 with scaled test-time compute.
   - DeepSeek-V4 runs RL against a GRM for hard-to-verify tasks.
   - Sources: "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning," DeepSeek-AI, 27 Nov 2025, https://arxiv.org/abs/2511.22570 · DeepSeek-V4 §5.1.1
2. **DATED-FRAMING** · `:75`, `:258` (takeaway 3) · Math-Shepherd-style Monte Carlo labelling is "the technique that made PRMs practical." · Qwen's PRM study finds MC-estimation labels give *inferior* performance and generalization compared with LLM-as-judge and human annotation. It also finds best-of-N evaluation of PRMs biased, and it recommends consensus filtering of MC and judge labels. ThinkPRM, a generative PRM that writes a verification chain of thought, beats discriminative PRMs using about 1% of PRM800K's labels. The chapter's costs list is right; its framing of MC as the practical route is not.
   - Sources: "The Lessons of Developing Process Reward Models in Mathematical Reasoning," Zhang et al. (Qwen), Jan 2025, https://arxiv.org/abs/2501.07301 · "Process Reward Models That Think," Khalifa et al., Apr 2025 (TMLR), https://arxiv.org/abs/2504.16828
3. **MISSING-MAJOR** · `:203-230` (§16.9: "an active area … the most likely route") · The prediction came true and should be written up as current practice: DeepSeekMath-V2 (verifier plus meta-verifier), ThinkPRM, DeepSeek-GRM (inference-time-scaled generative RM), Kimi K2's self-critic, and V4's actor-as-judge. The "safe as a ranker, dangerous as an objective" reconciliation in §16.7 needs an update. Frontier labs *are* optimizing against learned judges, mitigated by meta-verification, rubric constraints, and co-training the judge with RL.
4. **DATED-FRAMING** · `:122-152` (§16.6 "three methods": voting → best-of-N → beam/MCTS; Snell et al. as the organizing frame) · Shipped test-time compute in 2025–26 is mainly *sequential long chain-of-thought with a controllable budget*: Qwen3 thinking-budget curves, V4's three effort modes, s1 budget forcing. Parallel sampling is combined with *self*-verification (DeepSeekMath-V2) rather than PRM-guided tree search. Tier 3 (beam/MCTS over steps) is not how frontier reasoning models spend inference compute, as far as their reports say. Keep it, and demote it to "what was tried."
   - Sources: Qwen3 §4.7; DeepSeek-V4 §5.1.1; s1 (arXiv:2501.19393); DeepSeekMath-V2.
5. **WRONG-NOW** · `:190` (table: rule-based verifier "Hackable | No"), `:197` ("It dominates on every axis") · Same correction as Ch12 finding 2 and Ch15 finding 2. Rule-based verifiers for code and agentic tasks are hackable in documented production RL (Baker et al. 2025; MacDiarmid et al. 2025). The table should say "Rarely, but yes — via verifier bugs/gaps."
6. **MISSING-MINOR** · `:172`, `:157` ("the most successful reasoning model of the era") · R1 was surpassed within the year. The factual point (R1's "unsuccessful attempts" rejected PRMs and MCTS) still stands and is worth keeping. Update the framing, and cite the Sep 2025 *Nature* version.

### Keep as-is
§16.1 correct-by-luck and ORM/PRM table; §16.2 Cobbe verifiers; §16.3 PRM800K and the architecture; §16.4 cost analysis (steps × rollouts; "difficulty-for-the-current-policy, not correctness"); §16.5 inference vs training use; the §16.7 summary of R1's reasons; §16.10 point 4 (rule-based PRMs from compilers and proof checkers), which is still underexploited and still true; §16.11 JD (the evaluation-as-a-function-of-compute point has aged well).

### Lab opportunity
**Extend `lab16_best_of_n_and_prm` with a generative-verifier arm.** Simulate a critique-then-verdict verifier as a noisy step-checker that also outputs its claimed first-error location. Add a meta-verifier that accepts a verdict only if the claimed error location is actually wrong. Measure how best-of-N accuracy at large N changes with and without meta-verification. The expected result, mirroring DeepSeekMath-V2: meta-verification removes most of the high-N downturn caused by hallucinated issues.

### References to add
- DeepSeek-AI. "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning." Nov 2025. arXiv:2511.22570.
- Zhang, Z. et al. (Qwen). "The Lessons of Developing Process Reward Models in Mathematical Reasoning." Jan 2025. arXiv:2501.07301.
- Khalifa, M. et al. "Process Reward Models That Think." Apr 2025. arXiv:2504.16828.
- Liu, Z. et al. (DeepSeek). "Inference-Time Scaling for Generalist Reward Modeling." Apr 2025. arXiv:2504.02495.
- Muennighoff et al. "s1: Simple test-time scaling." Jan 2025. arXiv:2501.19393.
- DeepSeek-R1, *Nature*, Sep 2025, above.

---

## Should the part's structure change?

**Yes.** The current split (SFT / RM / PPO-RLOO / DPO / GRPO / PRM) follows the order in which the *methods were invented* (2022–2024). A 2026 practitioner reading DeepSeek-V4, Qwen3, GLM-4.5 or Olmo 3 would carve post-training by **the signal and the stage**, not by the algorithm's name. Three facts force the change:
- Algorithm choice has collapsed to the GRPO family plus patches, and ScaleRL finds most recipe details shift efficiency, not asymptote. PPO, RLOO and GRPO no longer earn three separate chapters' worth of "which one?"
- **Where the reward comes from** (rule verifier, rubric plus generative judge, scalar RM, PRM, teacher logits) is now the main design axis. It is currently split across Ch12, Ch15 and Ch16.
- Distillation (off-policy long chain-of-thought, and on-policy with teacher logits) and RL *systems* (async, mismatch) are major 2025–26 topics with no home in the current structure.

**Proposed carve (same chapter count, Ch11–16):**
1. **Ch11 — Imitation: SFT and distillation.** Current §11.4–11.7 mechanics, plus long-CoT distillation data (s1, OpenThoughts), reasoning chat templates and hybrid vs separate think models, and on-policy distillation as the stage that merges specialists (Qwen3, V4, GLM-4.5).
2. **Ch12 — Reward signals.** Bradley–Terry RMs (current §12.2–12.7), rule verifiers *and how they get hacked*, rubrics plus generative reward models, and process rewards and PRMs (absorbing current §16.1–16.5 and §16.7–16.9). One chapter, one question: *what is the signal and how does it fail?*
3. **Ch13 — Policy gradients: one family.** REINFORCE → PPO → RLOO → GRPO → DAPO / Dr. GRPO / GSPO / CISPO, taught as choices of **baseline, ratio (token vs sequence), clipping, and loss aggregation**. This is the "what do you subtract" taxonomy the book already has at §13.3 and §13.8, extended. PPO's value network becomes the historical first answer.
4. **Ch14 — Offline preference optimization.** The DPO family, repositioned as a cheap stage between SFT and RLVR. Adds delta learning.
5. **Ch15 — Reasoning RL at scale.** RLVR, R1-Zero and R1, curriculum and dynamic sampling, length control and effort modes, the "does RL add capability?" debate, and RL compute scaling (ScaleRL).
6. **Ch16 — RL systems.** Generation as bottleneck (current §13.9–13.10), sync vs async (AReaL, PipelineRL, slime, verl), weight sync, train–inference mismatch (TIS, FP16, Keep Routing / Keep Sampling Mask), and test-time compute serving (the current §16.6 material, re-centred on budgets and self-verification).

Ch17–19 are outside my scope. The Ch19 R1 case study would benefit from a companion "2026 pipeline" case (DeepSeek-V4 or Olmo 3, the latter being fully open). If the refresh cannot restructure, the minimum is: move §13.9–13.10 into a systems section, merge Ch16's PRM material into Ch12's reward-signal framing, and add OPD to Ch11.

---

## Unverified leads
Not confirmed from primary sources during this audit. Do not cite without checking.
- **DeepSeek-V4 arXiv ID.** I read V4's post-training text from `https://arxiv.org/html/2606.19348v1`, fetched earlier into the shared session index. I confirmed the title against the official HF model card and the 24 Apr 2026 API release note, but I did not independently open the arXiv abstract page. Verify the ID before citing.
- **Which Qwen models were trained with GSPO.** The GSPO paper says it "contributed to the remarkable improvements in the latest Qwen3 models." I did not confirm which checkpoints (2507 releases? Qwen3-Next?).
- **Kimi K2 Thinking / Kimi K2.5 (arXiv:2602.02276) and GLM-4.6 → GLM-5.x post-training recipes.** The slime README lists GLM-5.x as trained with slime. I read no primary report for these.
- **Whether DeepSeek-V3.2/V4, Qwen3, Kimi K2 or GLM-4.5 use DPO anywhere.** I verified that their described alignment stages are RL- and distillation-based. I did not exhaustively confirm that DPO is absent from every stage.
- **Frontier (closed-lab) use of LoRA** in main post-training paths. There is no primary source either way.
- **Whether the Nature (Sep 2025) version of DeepSeek-R1 retains the "unsuccessful attempts" section on PRM/MCTS verbatim.**
- **The exact form of Qwen3's thinking budget** (hard truncation vs trained). The report shows budget-scaling curves; the mechanism detail is unconfirmed.
- **REINFORCE++, VAPO, and other 2025 critic-based variants.** Not checked; they may deserve a line in the Ch13 lineage.
- **A published compute-optimal pre-training/RL split.** I found none. ScaleRL covers RL compute only.
- **OpenAI/Google/Anthropic reasoning-RL recipes** beyond the two hacking papers cited. Nothing primary found or checked.
