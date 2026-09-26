# Staleness audit — Frontier LLM Engineering (2026-09-26)

Read-only audit of all 26 chapters against primary sources published through Sept 2026.
Five research agents, one per chapter group; per-chapter detail, line numbers, and full
citations are in the part reports:

- [part-A-ch01-04.md](part-A-ch01-04.md) — frontier run, org chart, data, tokenization
- [part-B-ch05-09-10.md](part-B-ch05-09-10.md) — architecture, mid-training, DeepSeek-V3 case study
- [part-C-systems.md](part-C-systems.md) — distributed, cluster, optimization, kernels, checkpointing
- [part-D-ch11-16.md](part-D-ch11-16.md) — SFT, RM, PPO/RLOO, DPO, GRPO, PRM (+ structure proposal)
- [part-E-ch17-26.md](part-E-ch17-26.md) — CAI, agentic, R1, serving, eval, careers (+ ch26 prediction scorecard)
- [part-F-exercises-labs.md](part-F-exercises-labs.md) — exercises, solutions and lab code, checked for Tier 0 propagation only: 43 findings (16 change a worked answer, 6 wrong exercise premises, 2 change a lab result, 19 text-only). No lab's headline result changes; `lab07` prints ~2× where its own values give ~3.5×.

Tagged findings (approximate, counted by regex over the reports): **~63 WRONG-NOW, ~50 MISSING-MAJOR,
~49 MISSING-MINOR, ~55 DATED-FRAMING.** Plus ~60 "unverified leads" held back from the findings.

## The headline: two different problems

1. **Never-true claims (accuracy).** Claims about what a cited report says that the report does not say.
   These were wrong the day the book was published. They damage the book's core promise ("we read the
   reports for you") and should be fixed first, independent of any refresh.
2. **Staleness (currency).** True in 2024, superseded by 2025–26 work. This is what the refresh is for.

### Tier 0 — never true, confirmed by more than one agent or by me

| Claim | Where | What the source says |
|---|---|---|
| DeepSeek-V3 layout is PP4 × EP8 × DP64, 60 layers, small TP | ch01:147–149, ch06:409–411 & 521–523, ch07:99–101, ch10:332 | PP16, EP64 across 8 nodes, ZeRO-1 DP, **no TP**, 61 layers (V3 report). Found independently by agents A, B, C; grep confirms 4 chapters. |
| Block-quoted V3 report text: "900 GB/s NVLink bandwidth per GPU" | ch07:31 | Quote does not exist; H800 NVLink is ~160 GB/s vs IB 50 GB/s (3.2×, not the book's ~18×). 900 GB/s is the H100 figure. |
| FP32 optimizer state is "non-negotiable" | ch08:248 | V3 kept AdamW moments in BF16. |
| V3 LR schedule with spike-skipping / rollbacks | ch08 §8.4 | Constant LR to 10T tokens, then cosine; no rollbacks reported. |
| V3 MLA has 4 KV heads; first layer dense; RL is PPO; distillation V3→R1 | ch10 | No KV-head grouping; first 3 layers dense; GRPO; distillation R1→V3. |
| Tokenizer JSON blocks for Llama-3 / Qwen / etc. | ch04 §4.8 | Parsed from published `tokenizer.json`: `byte_fallback` false in all three, merge counts & special tokens wrong, Qwen3 splits single digits and has no CJK class, Llama-3 embeddings untied. |
| Llama-3 data mix, dedup; Qwen3 36T split | ch03 §3.5 | Llama-3 50/25/17/8, no suffix-array dedup; Qwen3 36T is pre-training only, 119 languages, no 50/50 EN/ZH. |
| Qwen3 / V3 / Mixtral configs | ch05 §5.14 | Qwen3 has no shared experts, uses global-batch balance loss, 4K→32K; Mixtral is GQA; V3 caches 576 values/token/layer. |
| Llama-3 context extension & annealing | ch09 | Six stages over ~800B tokens; annealing is the final 40M tokens, not 5–10%. |
| Failure taxonomy: network 40–50%, GPU 10–20% | ch07 | Llama-3: GPU issues 58.7% of unexpected interruptions (ch21 quotes this correctly — the book contradicts itself). |
| "Labs withhold the algorithm, not the data" | ch26:112 | Contradicts the book's own §19.8 ("the data, not the algorithm"). |

**Root cause, as far as the audit can tell:** earlier review passes checked the book against its
*labs*; nothing checked "the report says X" claims against the reports. One wrong fact (the V3 layout)
was restated in four chapters instead of linked from one.

### Tier 1 — now false (WRONG-NOW), highest reader impact

- **"V4 is expected"** (ch08:355, ch10). DeepSeek-V4 shipped: arXiv:2606.19348 (arXiv date 2026-04-26,
  verified directly): 1.6T/49B and 284B/13B MoE, 1M context, CSA+HCA hybrid attention, mHC, Muon, 32T+ tokens.
- **AdamW is universal** (ch01, ch08). Muon at frontier scale: Moonlight, Kimi K2 (MuonClip), GLM-4.5, V4.
- **Verifiable / final-state rewards can't be hacked** (ch12:243, 314; ch15:126, 194; ch18:118). Refuted by
  OpenAI (Mar 2025) and Anthropic (Nov 2025) reward-hacking work.
- **SWE-bench Verified is "the one to quote"** (ch18:207). Retired by OpenAI 2026-02-23; ch23 should use it
  as the contamination case study.
- **Claude's constitution is the 2023 principle list** (ch17:115). Replaced 2026-01-22.
- **R1 withholds X** (ch19 §19.8). The *Nature* version (2025-09-17) discloses most of it, including compute
  (~147K H800 GPU-hours).
- **Scalar RMs in the RL loop / iterative DPO is what strong open models use / generative verifiers are
  inference-only** (ch12, ch14, ch16). Current recipes: rubric generative RMs, RL + on-policy distillation.
- **"No fundamentally new operations", "RoPE has won"** (ch05). Sparse attention (V3.2, V4), linear/softmax
  hybrids (Qwen3-Next, Qwen3.5, MiniMax-M1), mHC, NoPE / partial RoPE (Llama 4, GLM-4.5, V4).
- **Vocabularies settled at 128K–152K** (ch04). Gemma 3 262K, Qwen3.5 248K, Llama 4 ~202K, gpt-oss 201K.
- **TP ≤ 8; 5D parallelism not standard** (ch06). NVL72 racks; Megatron Parallel Folding, TorchTitan.
- **Serving large MoE is an open problem** (ch22:168). DeepSeek published its EP + PD-disaggregated system (Mar 2025).

### Tier 2 — major gaps (MISSING-MAJOR)

On-policy distillation (ch11), GRPO successors DAPO / GSPO / CISPO (ch15), RL scaling laws (ScaleRL) and
the "does RL add capability" debate (ch15), async RL systems and train–inference mismatch fixes
(ch13: AReaL, PipelineRL, slime, verl; truncated IS), FP4/NVFP4/MXFP8 training (ch08, ch22), FlashAttention-4
and TileLang / CuTe DSL (ch20), synthetic pre-training data at scale (ch03), mid-training as its own
stage (ch09), agentic RL recipes, MCP, METR time horizons, Terminal-Bench 2.0, τ²-bench (ch18), evaluation
awareness and Leaderboard Illusion (ch23), 2025 open datasets FineWeb2 / Common Pile / Dolma 3 (ch03).

## Chapter verdicts

| Ch | Verdict | Tier 0 | Notes |
|---|---|---|---|
| 1 | Rewrite §1.5–1.6 | ✔ | V3 layout; V4 replaces V3 as "cleanest public example" |
| 2 | Light touch | | §2.3.8 reasoning role is MCTS-framed (R1 lists MCTS as a failed attempt) |
| 3 | Rewrite §3.2.6, 3.3.3, 3.5 | ✔ | Numbers; synthetic data; bucketed quality |
| 4 | Rewrite §4.4, 4.8, 4.11 | ✔ | Tokenizer JSON vs real files; vocab table |
| 5 | Rewrite many sections + new hybrid-attention section | ✔ | Configs wrong; new ops |
| 6 | Rewrite §6.10, 6.11, 6.13 | ✔ | V3 layout ×2 |
| 7 | Rewrite §7.1–7.4, 7.12, 7.15 | ✔ | Fabricated quote; failure taxonomy |
| 8 | Rewrite §8.3–8.4, 8.7–8.9, 8.14–8.15 | ✔ | Muon; BF16 moments; FP4; FP8 "unsigned" |
| 9 | Refresh | ✔ | Llama-3 recipe; YaRN no longer the default |
| 10 | Rewrite §10.4, 10.7–10.10, 10.12–10.14 | ✔ | Keep V3; add closing V3→V3.1→V3.2→V4 section |
| 11 | Refresh | | On-policy distillation; think/no-think templates; LoRA Without Regret |
| 12 | Rewrite §12.8, 12.9, 12.12 | | Generative/rubric RMs; reward hacking |
| 13 | Rewrite §13.9–13.10 | | Train–inference mismatch; async RL |
| 14 | Refresh | | Delta learning; DPO's role now a middle stage |
| 15 | Rewrite §15.2, 15.4/15.7, 15.9, 15.11 | | Successors; scaling law; thinking budgets |
| 16 | Refresh | | Generative verifiers in training; fold into RM chapter? |
| 17 | Rewrite §17.5 | | 2026 constitution; CoT monitorability |
| 18 | Rewrite §18.6, 18.9, 18.10 | | Most dated chapter |
| 19 | Rewrite §19.8 | | Move to *Nature* version; keep R1 as the case study |
| 20 | Light touch | | FA-4; TileLang/CuTe DSL; Triton has TMA now |
| 21 | Light touch | | Correct against Llama-3; add async ckpt, V4 determinism, rollout-resume bias |
| 22 | Light touch | | Physics evergreen; add EP serving, FP4, EAGLE-3, PD disaggregation |
| 23 | Refresh | | SWE-bench Verified retirement as case study |
| 24 | Light touch | | 2026 JD vocabulary (RL environments, verifier design, MCP) |
| 25 | Light touch | | Employer list; "environment engineer" is now established |
| 26 | Refresh | ✔ | Prediction scorecard in part-E; fix line 112 |

**Evergreen, per the agents — do not churn:** ring all-reduce / collective cost arithmetic, memory
accounting, bit-exact resume (ch21), KV-cache and batching physics (ch22), BPE mechanics, DPO derivation,
PPO/RLOO mechanics, the JD-reading method (ch24). Detail in each report's "Keep as-is".

## Lab opportunities (from the reports)

| Candidate | Chapter | Why a 2026 reader wants it |
|---|---|---|
| Toy multi-turn tool-use env where the agent learns to edit the tests | 18 (no lab today) | Demonstrates "final-state rewards are unhackable" failing, measurably |
| On-policy distillation vs SFT vs RL on the Countdown task | 11 / 15 | The 2025–26 post-training workhorse; reuses lab15 infra |
| DAPO / GSPO / CISPO variants in `lab15_grpo_countdown` | 15 | Shows what each fix actually changes |
| Muon vs AdamW at small scale | 8 | The optimizer shift of the period |
| FP4 (NVFP4 / MXFP4) block scaling in `lab08_precision_and_stability` | 8 | Extends the existing outlier-scaling result |
| Sparse / hybrid attention KV-cost lab | 5 / 22 | Extends lab05_attention_variants with V3.2/V4-style attention |

## Decisions for the grilling session (yours, not the audit's)

> **Resolved 2026-09-26.** Every decision below was settled in a grilling session; the outcome is
> recorded in the repo's `CONTEXT.md` (vocabulary), `docs/adr/` (ADRs 0001–0003) and the spec,
> [dnakhoa/frontier-llm-engineering#4](https://github.com/dnakhoa/frontier-llm-engineering/issues/4).
> The list is kept as the audit originally posed it.

1. **Accuracy first?** Ship a Tier 0 correction release before the refresh, or fold it in?
2. **Single source of truth for lab facts.** State V3/V4/Llama-3 facts once (table in ch10 / an appendix)
   and link, rather than restate in four chapters?
3. **Configs generated, not hand-written.** Generate the ch04/ch05/ch06 config blocks from published
   files (the ch04 audit did this and found every block wrong)?
4. **Part 3 structure.** Agent D proposes reorganising by signal/stage: imitation & distillation → reward
   signals (absorbs PRMs) → policy-gradient family → DPO as a middle stage → reasoning RL at scale → new
   RL-systems chapter. Minimum alternative: add on-policy distillation to ch11, fold ch16 into ch12,
   split ch13's systems sections out. Renumbering touches link permanence.
5. **Case studies.** Keep V3 + R1, each with a "what came next" section (agents' recommendation), or add V4 /
   Kimi K2 / Olmo 3 as new case studies?
6. **Which new labs,** from the table above.
7. **Repackaging** (separate from this audit): README leads with "what you'll build", zero-install
   Colab path, lab-first map — modelled on ai-engineering-from-scratch / TorchCode / how-to-train-your-gpt.

## Citation cautions

- DeepSeek-V4 arXiv:2606.19348 — **verified** on arXiv; cite by ID with arXiv's date (2026-04-26), despite the odd ID/date pairing.
- arXiv:2601.11659 "The Llama 4 Herd" — withdrawn and not from Meta. The book does not cite it; don't start.
- Llama 4 vocab figure came from an unofficial mirror (official repo is gated).
- Kimi K3 — model card only, no technical report found. Any architectural claim needs that caveat.
- Every report ends with "Unverified leads" — none of those are findings until checked.
