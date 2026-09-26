# v1.0.1 release sweep: every never-true finding, resolved

This is the release checklist for issue #14. Every finding in the six audit reports that was tagged as contradicting a primary source, or the book itself, is classified below as **FIXED** (with its errata entry), **STALE** (true when written; deferred to its refresh group), **UNSUPPORTED** (no source, but not contradicted; softened or sourced, recorded in the CHANGELOG) or **N/A** (an exercise premise, not a claim).

The sweep ran before the last corrections landed. The 20 items it listed as *still present* were then fixed in commits `2afa9e5` (items 1–5, 13–20) and `89ffb3b` (items 6–12), and entered on the errata page as E-018 to E-022 and E-025 to E-028; its unsupported list was handled in the same commits. Errata ids below are the final ones, E-001 to E-046.

---

# Release sweep for #14 (v1.0.1): every never-true finding in audit parts A–F

Swept 2026-09-26 against worktree `book-content-outdated-99be26` at HEAD `377c707`. The working tree also had an uncommitted `README.md` edit from another session.

Line numbers are current as of the sweep. The audit's line numbers are older.

**Errata IDs have shifted.** While this sweep ran, a new E-029 ("Lab descriptions that quoted smoke-test numbers") was inserted. Every entry from the old E-029 onward moved up by one, so the Llama 3 entries are now E-030–E-041, Qwen3 E-042–E-045, and Mixtral E-046. This sweep uses the new numbers.

**Primary sources checked:** the arXiv HTML of DeepSeek-V3 2412.19437v2, Qwen3 2505.09388v1, Mixtral 2401.04088v1, Llama 3 2407.21783v3, R1 2501.12948v1, DeepSeek-V2 2405.04434v5 and DeepSeek LLM 2401.02954v1. Wherever this sheet says "0 hits", it means a regex search of the full paper text found nothing, references excluded.

Classes: **FIXED** · **STILL-PRESENT-NEVER-TRUE** (abbreviated SP-NT) · **STALE** · **UNSUPPORTED**. A few rows are **N/A**: the finding was self-consistent under a premise the exercise itself sets, so it is not a claim about a source.

---

## Part A — Chapters 1–4

| Finding | Class | Evidence |
|---|---|---|
| Ch1 #1: the V3 layout PP4×EP8×DP64, 60 layers, small TP | FIXED | E-001 |
| Ch1 #1 "Also": "Qwen3 team used a similar 3D-parallel layout" | FIXED | This was an unsupported claim, not a never-true one. The text is gone, and no errata entry is needed. |
| Ch1 #2: "almost always AdamW" | STALE | It was true when written. Muon (Kimi K2, GLM-4.5, V4) arrived in 2025–26. |
| Ch3 #1: Llama 3 mix, "15.6T for the 8B", MinHash + suffix arrays | FIXED | E-030. Ch3 §3.5 now reads "50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual". |
| Ch3 #2: Qwen3 "36T across all stages", 50/50 EN/ZH | FIXED | E-045 |
| Ch3 #3: "Llama-2 paper introduced" the quality classifier; :423 "Llama-2/3 paper showed aggressive dedup" | FIXED | E-031 covers the classifier. The :423 dedup attribution is also gone, but no errata entry names it. |
| Ch3 #4: synthetic data "less so in pre-training" | STALE | A claim about the field, overtaken by trillion-token synthetic pre-training in 2025. |
| Ch3 #9: LibGen's legal status "is debated" | STALE | The *Bartz v. Anthropic* order of Jun 2025 came after it was written. |
| Ch3 #12: packing "done with attention masks that prevent cross-document attention" | **SP-NT** | `03-data-pipelines.md:308`. V3 §4.1: "do not incorporate cross-sample attention masking". The book's own §3.5 (:362) says the same, so the chapter contradicts itself. |
| Ch3 #14: V3 "60% English, 30% Chinese, 10% other" | FIXED | E-015 |
| Ch3 #15: R1 "RL data: 600K reasoning prompts, plus preference data" | UNSUPPORTED | `03-data-pipelines.md:381`. R1 v1 gives 600k/200k for SFT only (§2.3.3) and no count of RL prompts. |
| Ch4 #1: vocabularies "settled at 128K–152K" | STALE | Gemma 3, Llama 4 and Qwen3.5 came later. |
| Ch4 #2: Qwen3 regex with a CJK class and `\p{N}{1,3}` | FIXED | E-003 |
| Ch4 #3: tokenizer JSON blocks, byte_fallback, merge counts, special tokens, 152,064 | FIXED in §4.4/§4.5/§4.11 (E-002). **SP-NT at :89** | `04-tokenization.md:89` still says "Llama-3 used 256-byte base + 127,744 merges. DeepSeek-V3 used 256-byte base + 128,000 merges. Qwen3 used 256-byte base + 151,808 merges." |
| Ch4 #4: "Llama-3 uses tied embeddings" | FIXED | E-004. Solution 5.1's "ties or partially shares" is also gone. |
| Ch4 #10: V3's 128K "started from a 32K Llama-1-style tokenizer" | FIXED | E-005 |

## Part B — Chapters 5, 9, 10

| Finding | Class | Evidence |
|---|---|---|
| Ch5 #1: "no fundamentally new operations" | STALE | CSA/HCA, Gated DeltaNet and mHC appeared in 2025–26. |
| Ch5 #3: "RoPE has won", NoPE "not used at frontier" | STALE | Llama 4, GLM-4.5 and V4 came later. |
| Ch5 #5: Qwen3 config: shared experts, aux loss, "trained at 32K directly", dual-base RoPE, d_ff 12288 | FIXED | E-042, E-043, E-044 |
| Ch5 #6: V3 config: 60 layers, 2·d_c = 1024, 15 GB / 2×, E5M2 | FIXED | E-007, E-008, E-012 |
| Ch5 #7: ">1M only Gemini 1.5 Pro" | STALE | Open 1M-context models came later. |
| Ch5 #8: MTP "introduced in V3", "single linear layer", "training-only" | FIXED | E-009. The audit's note on how MTP spread in 2025–26 is STALE material. |
| Ch5 #10: Mixtral "MHA, not GQA" | FIXED | E-046 |
| Candidate 5: Mixtral "Switch-style aux loss" in §5.9 and §5.11 | UNSUPPORTED, **and it conflicts with E-046** | `05-architecture.md:18`, `:517` ("[37] … with the standard Switch-style auxiliary loss") and `:633`. The Mixtral paper has 0 hits for "auxiliary" or "Switch". E-046 says "The paper does not describe its load-balancing loss" and marks the claim fixed, but it survives in two sections that cite [37]. Listed as action item 16. |
| Ch9 #1: Qwen3 "32K → 128K", "trained at 8K" | FIXED | E-044. `09-mid-training.md:29`, `:124` and `:461` are now correct. |
| Ch9 #2: Llama 3 annealing "last 5–10%", stages | FIXED | E-032 (`:23`, `:67`) |
| Ch9 #2: "Llama-3 uses a YaRN-style rescaling for its 128K context" (candidate 7) | UNSUPPORTED | `09-mid-training.md:242`. The Llama 3 paper has 0 hits for YaRN. It gives only RoPE θ = 500,000 (Table 3) and six extension stages (§3.4.2). |
| Ch9 #4: "YaRN is the standard for frontier long-context" | STALE | The field split into several approaches in 2025–26. |
| Ch9 #9: `rescale_inv_freq_ntk` code bug | **SP-NT** (self-contradiction) | `09-mid-training.md:309`: `base_new = inv_freq.max() ** (dim / (dim - 2))`. `inv_freq.max()` is 1.0, which contradicts the formula at `:250`. |
| Candidate 3: Qwen3/V3 "hint at" annealing followed directly by SFT | **SP-NT** | `09-mid-training.md:128`. Qwen3 has 0 hits for "anneal". Its post-training (§4) is a separate four-stage pipeline applied to the base models. V3 §5.1 fine-tunes V3-Base for 2 epochs as a separate stage. |
| Ch10 #1: "V4 report (when it lands)" | STALE | V4 shipped in Apr 2026. |
| Ch10 #2: the V3 facts (attention, layers, dense layers, parallelism, FP8, hours, schedule, post-training, MTP, data, MTBF) | FIXED | E-001, E-007, E-008, E-009, E-012, E-014, E-015, E-016, E-017. None of the old phrases matches any more. |

## Part C — Chapters 6, 7, 8, 20

| Finding | Class | Evidence |
|---|---|---|
| Ch6 #1: V3 PP4×EP8×DP64 | FIXED | E-001 |
| Ch6 #2: "only the 37B active parameters are optimized" | FIXED, **no errata** | §6.10 was rewritten (`06:418–424`). No entry records the old claim. |
| Ch6 #4: "5D … not yet standard" | STALE | Megatron Core published it later. |
| Ch6 #7: Llama 3 `BACKWARD_PRE` per-block wrapping; "no CP needed" | FIXED | E-037. The generic FSDP text at `06:122–140` is fine. |
| Ch6 #9: "Blackwell-era node has 8 NDR links" | STALE | ConnectX-8 at 800 Gb/s came later. |
| Ch6 #10: GPipe attributed to "Google [3] Megatron-LM"; "Megatron-V3 (2021)" | **SP-NT** | `06-distributed-training.md:201` and `:215` |
| Ch7 #1: block quote of "900 GB/s … 400 Gb/s NIC" | FIXED | E-011 |
| Ch7 #2: V3 EP8/PP4/DP64 | FIXED | E-001 |
| Ch7 #3: failure taxonomy (network 40–50%) | FIXED | E-033 |
| Ch7 #4: V3 "few unexpected interruptions", NVLink errors, silent-corruption rollback, DualPipe/DeepEP checkpointing, "builds on" PersistentWorker | **SP-NT** (all five) | `07:248`, `:279`, `:304`, `:390` and `:427`. See action item 6. |
| Candidate 9: "DeepSeek team has stated publicly that they target 2–4 hour MTBF" | **SP-NT** | `07-cluster-reality.md:285`. V3 has 0 hits for MTBF, and no other DeepSeek report is cited. The only V3 statement on the subject is "did not experience any irrecoverable loss spikes or perform any rollbacks" (Abstract, §4.2). |
| Candidate 2: "Qwen3 cluster at Alibaba Panjin" | **SP-NT** | `07:15`, `:332` and `:589` cite [6] Qwen3 for a Panjin cluster. The Qwen3 report has 0 hits for "Panjin" or "cluster" and describes no hardware. |
| Ch7 #6: Colossus "100,000 H100s" | STALE | xAI later doubled it. |
| Ch7 #7: "$1.50/H800-hour … ~$5.5M" | FIXED | E-016. `07:524` now gives $2 and $5.576M. |
| Ch7 #8: `NCCL_MIN/MAX_NCHANNELS` | STALE | Deprecated in NCCL 2.17, after the chapter's source era. `07:120`, `:229`. |
| Ch8 #1: FP32 optimizer state "non-negotiable" / "essential" | FIXED in §8.7 (E-013). **SP-NT at :122, :292** | `08:122`: "the optimizer state (m, v) is held in **FP32** … This is essential." `08:292`: "For the optimizer state … it is not fine". Both contradict the chapter's own `08:251` ("a default, not a law. DeepSeek-V3 stored both … in BF16") and `:476`. |
| Ch8 #2: V3 "skip ~200 batches, halve LR" | FIXED | E-014 |
| Ch8 #3: V3 schedule and batch ramp | FIXED in §8.15 text (E-014). **SP-NT in both tables** | `08:171`: V3 warmup "0.2%", floor "10%". `08:511–518`: V3 warmup "0.1%", schedule "cosine to 10%", batch "**73M** (ramped)". `08:462` itself says "~63M". |
| Ch8 #6: FP8 "unsigned" | FIXED | E-012 |
| Ch8 #8: Llama 3 8B/70B LRs | FIXED (E-038). **The audit was wrong in part.** | Llama 3 Table 3 gives 3×10⁻⁴, 1.5×10⁻⁴ and 8×10⁻⁵, so `08:166–168` and `:174` are correct. |
| Ch8 #8 + candidate 1: Qwen3 7e-5 / 2e-4, 0.5% warmup, 16–32M batch, BF16, AdamW β/ε/wd, clip 1.0 | **SP-NT** | `08:170`, `:174`, `:215`, `:493–498` and `:510–519`. Qwen3 §3.2 says LR and batch size are *predicted* per model by scaling laws. It gives no values, and has 0 hits for clip, AdamW or β. The book's own Qwen3 fact sheet (row "Learning rate and batch size") says "the values are not published". |
| Ch8 #9: "B* grows slowly with model size" | STALE | CBS scaling work (Zhang et al.) came later. The Llama "~4× CBS" part is FIXED (E-038). |
| Ch8 #12: V3 "custom fused cross-entropy", embedding WD 0.2, per-block LR "consistent with µP" | **SP-NT** | `08:54`, `:120`, `:194` and `:409`. See action item 8. |
| Ch20 #3: Triton "does not expose" TMA | STALE | Triton exposed TMA later. `20:226` |

## Part D — Chapters 11–16 (every flagged finding is a claim about the field)

| Finding | Class | Evidence (the chapter's wording that decides it) |
|---|---|---|
| Ch12 #1: "scalar reward models inside the RL loop" | STALE | `12:226`: "The practical split at most labs". A description of practice in 2024, overtaken by V4 and Kimi K2's generative/rubric RMs. |
| Ch12 #2: "Verifiable rewards cannot be hacked" | STALE | `12:245`: "It cannot be hacked **in the §12.6 sense** — there is no artifact to find". The book's §12.6 sense means exploiting a learned proxy, and no source is cited or contradicted. The field refuted the claim in 2025 (Baker et al.; MacDiarmid et al.). **Note:** the unqualified takeaway `12:316` ("Verifiable rewards cannot be hacked") sits against Solution 18's "Exploit A — special-case the test" (`solutions/ch18.md:167`). Fix the wording in the refresh. |
| Ch13 #1: off-policy lag "which PPO's clipping is designed to tolerate" | STALE | `13:272`. A method claim. Train/inference mismatch fixes (TIS, Keep Routing) came in 2025. |
| Ch14 #1: iterative DPO is "now the standard open recipe" / "what the strongest open models use" | STALE | `14:237`, `:281`. True of Llama 3 in 2024. |
| Ch14 #2: stronger-model pairs: "This is the trap" | STALE | `14:150`. The delta-learning result (Geng et al.) is from Jul 2025. |
| Ch14 #3: "do not use DPO" for math/code; "skip this chapter" | STALE | `14:285`, `:310`. Olmo 3 (Dec 2025) came later. |
| Ch15 #2: "There is nothing to hack … by construction, produced a correct answer" | STALE | `15:128`, `:194`. The chapter scopes this to math answer-checking, where it holds. Coding/agentic hacking was documented in 2025. |
| Ch15 MISSING-MAJOR #4 (GLM-4.5 "contradicts" cap-length advice) | STALE | `15:178`, `:184–188`. Newer work versus an advice claim. |
| Ch16 #1: generative verifiers limited "to inference-time use" | STALE | `16:230`. DeepSeekMath-V2 and V4 came later. |
| Ch16 #5: rule-based verifier "Hackable: No" | STALE | `16:190`, `:197`. Same reasoning as Ch12 #2. |

## Part E — Chapters 17–26

| Finding | Class | Evidence |
|---|---|---|
| Ch17 #1: Claude's constitution = the 2023 principle list | STALE | `17:117`. Accurate to the May 2023 publication; replaced on 22 Jan 2026. |
| Ch18 #1: SWE-bench Verified "is the one to quote" | STALE | `18:209`. Retired by OpenAI on 23 Feb 2026. |
| Ch18 #2: final-state reward "unhackable in the section 12.6 sense" | STALE | `18:118`. Same framing as Ch12 #2. |
| Ch19 #1: §19.8 "What the report does not tell you" (compute "is nowhere", hyperparameters) | STALE | `19:152–170`. True of arXiv v1 (Jan 2025). The *Nature* version (17 Sep 2025) discloses most of it. |
| Ch19 #2: arXiv 2501.12948 is "the primary source for this entire chapter" | STALE | `19:243`. *Nature* is now the version of record. |
| Ch26 #1: labs withhold "the algorithm, not the data" (candidate 10) | **SP-NT** (self-contradiction) | `26-whats-next.md:114`, against `19-case-study-r1.md:170` ("**the data, not the algorithm.**") and `exercises/ch19.md:100` |
| Ch26 #2: "FP4 is being explored" | STALE | `26:68`. FP4 shipped in 2025–26. |

## Part F — exercises, solutions and labs

| File:line (audit) | Class | Evidence |
|---|---|---|
| ex ch01:23: "14.8T tokens, 2,788K H800 GPU-hours" | **SP-NT** | `exercises/ch01.md:23`. V3 Table 1: 2,664K pre-training + 119K + 5K = 2,788K. The 14.8T belong to the 2,664K. |
| ex ch05:32: MLA 2·d_c = 1024 | FIXED | E-007 (now 576) |
| ex ch05:43: V3 60 layers | FIXED | E-008 |
| ex ch05:56: Qwen3 d_ff 12288 | FIXED | E-042 |
| ex ch06:74–80: EP8 hypothetical, "Over 60 layers" | FIXED | E-001. The hypothetical stays; the 60-layer copy is gone. |
| ex ch08:96: "why two different ones?" | FIXED | E-012 |
| ex ch10:62: one head, λ 0.3 | FIXED | E-009 |
| sol ch01:13, 40–56: 56.7 days, 3.02M tok/s, 1,475, MFU 33% | **SP-NT** | `solutions/ch01.md:11–56`. Knock-on at `solutions/ch03.md:299` ("~3.02M tokens/s"). |
| sol ch01:76–83: 16 B/param, 10.7 TB | N/A | The exercise stipulates 16 B. A footnote is optional. |
| sol ch03:176: Llama mix with an "academic" bucket | FIXED | E-030 |
| sol ch03:184: classifier architecture undisclosed | FIXED | E-031 |
| sol ch04:217: Qwen "151,646" | FIXED | E-002 |
| sol ch05:40: "ties or partially shares" | FIXED | E-004 |
| sol ch05:54–93: 164 KB, 21 GB, 7 sequences | FIXED | E-007 (the correction note is at :95) |
| sol ch05:105–117: 676B, 23.8B | FIXED | E-008 |
| sol ch05:148–165: Qwen3 151M/expert, 1,209M | FIXED | E-042 |
| sol ch05:279: aux-free "without touching the loss at all … DeepSeek's report is explicit" | **SP-NT** | Now at `solutions/ch05.md:306`. V3 §2.1.2 keeps a "Complementary Sequence-Wise Auxiliary Loss". |
| sol ch05:295–296: 12,288 per expert, vocab 151,646 | FIXED | E-042 |
| sol ch06:30: 134 H100s at 16 B | N/A | Stipulated recipe |
| sol ch06:220–226: 60 × 0.94 GB | FIXED | E-008 (now `:222`, 58 layers). Small residue: `:360` still says "~900 GB per step"; the audit gives ~870. |
| sol ch06:230–232, 260, 297: EP "must" be intra-node | FIXED | E-001 (correction note at `:240`). `:268` "EP = 8 … within a node" is a design choice for a hypothetical 200B model and agrees with the revised §6.11. |
| sol ch06:348: "for 60 layers … 120 blocking collectives" | **SP-NT** | Now at `solutions/ch06.md:356`. This is V3's no-TP answer: 61 layers, so 122 (V3 §4.2). |
| sol ch07:51: "EP degree is usually 8 (one node)" | UNSUPPORTED | `solutions/ch07.md:51`. A claim about the field with no source. V3 itself uses EP64. |
| sol ch08:171: moments "already FP32" | FIXED | E-013 |
| sol ch08:291–295: E4M3/E5M2 split | FIXED | E-012 |
| sol ch09:258: Qwen3 32K → 128K | FIXED | E-044 |
| sol ch10:11: 60 layers, MTP head | FIXED | E-008, E-009 |
| sol ch10:17: E5M2, FP32 moments | FIXED | E-012, E-013 |
| sol ch10:23, 295–297: 2,788K ≈ 56.7 days, "Consistent" | FIXED | E-016 (`:299` now uses 2,664K) |
| sol ch10:65–67: DualPipe and H800 NVLink | FIXED, no errata found | The sentence is gone. |
| sol ch10:145: "loss remains pure" | FIXED (by context) | `:145` refers to the bias only. `:11` now states the sequence-wise loss. |
| sol ch10:181: λ = 0.3 | FIXED | E-009 |
| sol ch10:185–206: MTP 0.6B | FIXED | E-009 |
| sol ch10:287–291: 676B overshoot | FIXED | E-008 |
| sol ch21:24–44: 8,052 GB, 80/107 s | FIXED | E-013. The V3 BF16-moment rows were added. |
| lab05:271–291: the 2·d_c "convention" | FIXED | E-007. The row is now labelled "wrong 2*d_c convention". |
| lab06:419: NVLink 200 / IB 25 | FIXED | E-023 |
| lab06:533: V3 searched with FP32 moments | FIXED | E-023 (`DEEPSEEK_V3_RECIPE`) |
| lab06:601–605: "both Adam moments stay FP32" | FIXED | E-023 |
| lab06:558, 622: "TP and EP belong inside a node" | FIXED | E-023 ("EP by default") |
| lab07:63, 78–79: 11×, "single most important number" | N/A | These are H100 constants and the lab never says H800. Label them "on H100" (UNSUPPORTED list). |
| lab07:233–234, 384–386: "EP belongs inside a node" | FIXED | E-024 / E-001 (`:233–235` now names V3's EP64) |

## Other candidates the task listed

| Candidate | Class | Evidence |
|---|---|---|
| 4. Qwen3 "Aggressive dedup." | UNSUPPORTED | `03-data-pipelines.md:377`. The Qwen3 report has 0 hits for dedup. |
| 6. Solution 4: DeepSeek and Qwen "trained on Chinese-heavy corpora" | UNSUPPORTED | `solutions/ch04.md:222`. Neither report gives a language split (E-045). V3 says only "multilingual coverage beyond English and Chinese" (§4.1). |
| 8. XGBoost "for the Llama pipeline" | UNSUPPORTED (not attributed to Llama) | No XGBoost in Ch10. `01:88` and `03:140/144` say "often/typically an XGBoost" generically. Llama 3 uses fastText and DistilRoberta (§3.1.1). |
| 11. README "real, or real-shaped" | FIXED (uncommitted) | The working-tree `README.md:148` now states the real/illustrative rule. That edit is uncommitted, from another session, and must land in the release. Residue: `book/how-to-use-this-book.md:11` ("real configurations from real technical reports") and `book/appendix/style-guide.md:79` ("real or real-shaped code"). |

## Candidate 12: spot-check of other misattributions (new; not in the audit)

| Location | Class | Evidence |
|---|---|---|
| `01-what-a-frontier-run-looks-like.md:34`: V3 used "FP8 (E4M3 and E5M2 formats)" | **SP-NT** | V3 §3.3.2: "we adopt the E4M3 format on all tensors". This contradicts `01:218` and the fact sheet. E-012's Ch1 fix missed §1.2. |
| `01:262`: "2,788K H800 GPU-hours of *useful* training … over 2 months"; `01:28`: "approximately 2 months (2,788K …)" | **SP-NT** | V3 Table 1: 2,788K is the total including context extension and post-training. "Less than two months" is pre-training (2,664K). This is the E-016 claim, surviving in Ch1. |
| `08:389`: small models "distilling it into a larger model … Qwen3, Llama 3, and DeepSeek-V3 all use" | **SP-NT** | Qwen3 §4.5 is "Strong-to-Weak Distillation": large into small. Llama 3 §1 says distillation is "an alternative path" and chose to over-train instead. V3 released no smaller variants. |
| `05-architecture.md:209`: "the DeepSeek team reports a 93.3% reduction vs MHA on a 128K context" | **SP-NT** | DeepSeek-V2 abstract: "Compared with DeepSeek 67B … reduces the KV cache by 93.3%". DeepSeek LLM §2.1: 67B "uses Grouped-Query Attention (GQA) instead of … MHA". |
| `22-inference-serving.md:55`: "DeepSeek-V2 reports a KV cache reduction of roughly 93% against its MHA baseline" | **SP-NT** | Same source: the baseline is DeepSeek 67B, a GQA model. |
| `23-evaluation.md:112`: "Llama 3 and DeepSeek [1] both discuss decontamination in their reports"; ref note `:223` "Decontamination reporting" | **SP-NT** (the V3 half) | The V3 report has 0 hits for "contaminat" or "decontam" outside its reference list. Llama 3 does (§5.1.4). |
| Chs 11, 12, 14–16, 19, 25, 26: other V3/R1/Llama/Qwen claims (1.5M SFT, R1→V3 distillation, 600k+200k, Qwen2.5/Llama 1.5B–70B distills, PRM/MCTS unsuccessful attempts, rule-based + model RM) | FIXED / correct | Checked against V3 §5.1–5.2 and R1 §2.3–2.4 and §4.2. No contradiction. |

---

## STILL-PRESENT-NEVER-TRUE (action list)

1. **Ch 3 §3.3.9 (`book/part-2-pretraining/03-data-pipelines.md:308`).**
   - Replace: "The packing is done with attention masks that prevent cross-document attention (so the model does not learn to attend to the boundary between two documents)."
   - With: "Labs differ here. Llama 3 masks attention across document boundaries and reports the mask had limited impact in standard pre-training; DeepSeek-V3 packs documents 'but do[es] not incorporate cross-sample attention masking'."
   - Cite: DeepSeek-V3 arXiv:2412.19437 §4.1; Llama 3 arXiv:2407.21783 §3.2. This also removes the conflict with the chapter's own §3.5 (`:362`).

2. **Ch 4 §4.2 (`04-tokenization.md:89`).**
   - Replace: "Llama-3 used 256-byte base + 127,744 merges. DeepSeek-V3 used 256-byte base + 128,000 merges. Qwen3 used 256-byte base + 151,808 merges."
   - With: "The published files hold 127,741 merge rules for DeepSeek-V3 (128,000 entries), 280,147 for Llama 3 (128,000 entries), and 151,387 for Qwen3 (151,643 entries)."
   - Cite: `book/appendix/fact-sheets/tokenizers.md` rows "Merge rules" (`tokenizer.json` @e815299, @8cde5ca, @8efa617).
   - Add it to E-002's "Where".

3. **Ch 9 §9.13 (`09-mid-training.md:309`).**
   - Replace: `base_new = inv_freq.max() ** (dim / (dim - 2))  # crude but standard`
   - With: `base_new = base * (new_max_pos / original_max_pos) ** (dim / (dim - 2))`, which needs a `base` argument. This is the formula the chapter states at `:250`.
   - Why: `inv_freq.max()` is 1.0, so every frequency collapses to 1.
   - Cite: the chapter itself (`:250`), and Part B Ch9 #9. Check it against `labs/lab09_rope_extension.py`.

4. **Ch 9 §9.4 (`09-mid-training.md:128`).**
   - Replace: "The Qwen3 and DeepSeek-V3 papers hint at a pattern where the annealing phase is followed directly by SFT, without a separate SFT run."
   - With: delete the attribution, or relabel the whole pattern as a hypothetical. Both reports describe post-training as a separate stage on the base model: V3 "fine-tune[s] DeepSeek-V3-Base for two epochs" (§5.1), and Qwen3 runs a four-stage post-training pipeline (§4).
   - Cite: V3 §5.1; Qwen3 arXiv:2505.09388 §4 (0 hits for "anneal").

5. **Ch 6 §6.5 (`06-distributed-training.md:201`, `:215`).**
   - Replace: "The original schedule, from Google [3]" (a citation to Megatron-LM). Replace with: "from GPipe (Huang et al., 2019)", with a new reference entry.
   - Replace: "A refinement from Megatron-V3 (2021)". Replace with: "the interleaved schedule of Narayanan et al., 2021 ('Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM')".
   - Cite: GPipe arXiv:1811.06965; Narayanan et al. arXiv:2104.04473.

6. **Ch 7, DeepSeek-V3 reliability claims (`07-cluster-reality.md:248`, `:279`, `:285`, `:304`, `:390`, `:427`).** Delete or correct all six:
   - `:248`: "The DeepSeek-V3 paper reports that their run 'suffered only a few unexpected interruptions'". The phrase is not in the report.
   - `:279`: "The DeepSeek-V3 paper notes that NVLink errors are a non-trivial fraction of their failures". Not in the report.
   - `:285`: "the DeepSeek team has stated publicly that they target 2–4 hour MTBF and design the run around it". V3 has 0 hits for MTBF, and no other source is cited.
   - `:304`: "DeepSeek has stated publicly that they encountered silent corruption during the V3 run and recovered by rolling back to a clean checkpoint". This directly contradicts the report.
   - `:390`: "the DualPipe and DeepEP repositories include asynchronous checkpointing code". Per the audit, both repos were inspected: they are a scheduling library and a communication library.
   - `:427`: "DeepSeek-V3 builds on this pattern [1]". V3 uses its in-house HAI-LLM framework (§3.2). The `PersistentWorker` model itself is also unverified.
   - The only true statement to use: V3's "training process is remarkably stable … we did not experience any irrecoverable loss spikes or perform any rollbacks".
   - Cite: DeepSeek-V3 arXiv:2412.19437 Abstract and §4.2.

7. **Ch 7, the Qwen3/Panjin cluster (`07-cluster-reality.md:15`, `:332`, `:589`).**
   - Replace: "the Qwen3 / Panjin cluster for the Chinese-side view [6]"; "The Alibaba Panjin cluster (Qwen) is similarly liquid-cooled"; "[6] Qwen3 — the Chinese-side case study for the Qwen3 cluster at Alibaba Panjin".
   - With: remove all three. The Qwen3 report describes no hardware or cluster (0 hits for "cluster" or "Panjin"). If the claim about the Panjin site is kept, it needs its own primary source.
   - Cite: Qwen3 arXiv:2505.09388 (the whole text).

8. **Ch 8, invented DeepSeek-V3 details (`08-optimization.md:54`, `:120`, `:194`, `:409`).**
   - `:54`: "The DeepSeek-V3 paper specifically calls out a custom fused cross-entropy …". Delete. The only cross-entropy in V3 is the MTP loss definition (§2.2).
   - `:120`: "some labs (DeepSeek-V3 in particular) use higher values for the embedding". Replace with "DeepSeek-V3 uses a single weight_decay = 0.1".
   - `:194`: "Some labs (DeepSeek-V3, for example) split the weight decay: a higher value (e.g., 0.2) for the embedding …". Same fix as `:120`.
   - `:409`: "the per-block learning rate schedule they describe is consistent with the µP framework". Delete. V3 describes one global LR schedule (§4.2).
   - Cite: V3 §4.2 ("β1 = 0.9, β2 = 0.95, and weight_decay = 0.1").

9. **Ch 8 §8.3 and §8.7 (`08-optimization.md:122`, `:292`).**
   - Replace `:122`: "held in **FP32** even when the parameters are in BF16. This is essential. If the optimizer state is in BF16, … the model can fail to train."
   - With: "FP32 is the standard default; it is not a law — DeepSeek-V3 kept both AdamW moments in BF16 'without incurring observable performance degradation' (§8.7)."
   - Replace `:292`: "For the **optimizer state** … it is not fine". Soften it to match.
   - Cite: V3 §3.3.3; the chapter's own `:251` and `:476`. Add both lines to E-013's "Where".

10. **Ch 8, the DeepSeek-V3 rows of both tables (`08-optimization.md:171`, `:511–518`).**
    - Warmup: replace "0.2%" (`:171`) and "0.1%" (`:514`) with "2K steps". The report gives steps, not a fraction; by our arithmetic it is about 0.8% of about 240K steps.
    - Floor: replace "10%" / "cosine to 10%" (`:171`, `:515`) with "constant, then cosine to 2.2e-5 (10%), then 7.3e-6 (3.3%)".
    - Batch: replace "73M (ramped)" (`:517`) with "~63M (ramped from ~13M)". `:462` already says ~63M.
    - Cite: V3 §4.2; `fact-sheets/deepseek-v3.md` rows 89–91. Add to E-014's "Where".

11. **Ch 8, the Qwen3 optimization "recipe" (`08-optimization.md:170`, `:174`, `:215`, `:493–498`, `:510–519` Qwen3 column).**
    - Remove every Qwen3 value: 2e-4 for 32B, 7e-5 for 235B, 10% floor, 0.5% warmup, AdamW β1 0.9 / β2 0.95 / ε 1e-8 / wd 0.1, "~16M–32M tokens, held constant", BF16, clip 1.0, "cosine to 10%".
    - Replace with: "Qwen3 does not publish its learning rates, batch sizes or optimizer settings; it predicts them per model from scaling laws over its three pre-training stages."
    - Drop the Qwen3 column from the four-way table, or fill it with "not published".
    - Cite: Qwen3 §3.2 ("we set the predicted optimal learning rate and batch size strategy for each dense or MoE model"); `fact-sheets/qwen3.md` row "Learning rate and batch size".
    - This needs a new errata entry, "Qwen3's optimizer recipe".

12. **Ch 8 §8.11 (`08-optimization.md:389`).**
    - Replace: "training a **smaller model on more data** and *distilling* it into a larger model … the 'small models, big data' approach that Qwen3, Llama 3, and DeepSeek-V3 all use for some of their model variants."
    - With: "Qwen3 distils its flagship *into* its smaller models ('Strong-to-Weak Distillation'); Llama 3 names distillation as an alternative and instead over-trains its small models; DeepSeek-V3 has no smaller variants."
    - Cite: Qwen3 §4.5; Llama 3 §1 ("An alternative path is to distill larger models into smaller ones").

13. **Ch 26 (`26-whats-next.md:114`).**
    - Replace: "Chapter 19 §19.8's observation about what labs withhold: the algorithm, not the data."
    - With: "… what labs withhold: the data, not the algorithm."
    - Cite: `19-case-study-r1.md:170`; `exercises/ch19.md:100`. This is audit Tier 0 and a self-contradiction.

14. **Ch 1 §1.2 (`01-what-a-frontier-run-looks-like.md:34`).**
    - Replace: "to use FP8 (E4M3 and E5M2 formats) for the bulk of the matmuls".
    - With: "to use FP8 (E4M3 on all tensors, with fine-grained scaling) for the bulk of the matmuls".
    - Cite: V3 §3.3.2; the chapter's own `:218`. Add to E-012's "Where".

15. **Ch 1 (`01:28`, `:262`); Exercise 1.2 (`exercises/ch01.md:23`); Solutions 1.1–1.2 (`solutions/ch01.md:11–56`); knock-on `solutions/ch03.md:299`.**
    - Replace `:262`: "2,788K H800 GPU-hours of *useful* training, on a 2,048-GPU cluster over 2 months". Replace `:28`: "approximately 2 months (2,788K H800 GPU-hours)".
    - With: "2,788K H800 GPU-hours in total — 2,664K for pre-training, which took under two months, plus 119K for context extension and 5K for post-training".
    - Rework Ex 1.2 and Sol 1.1–1.2 on 2,664K hours: 54.2 days, 3.16M tok/s, 1,543 tok/s/GPU, MFU about 34.7% (Part F's recomputation).
    - Cite: V3 Table 1. Add to E-016's "Where", with an "If you worked this before v1.0.1" line.

16. **Ch 5 §5.9, §5.11 and the chapter summary (`05-architecture.md:18`, `:517`, `:633`): Mixtral's aux loss.**
    - Replace: "with the standard Switch-style auxiliary loss" (`:517`, `:633`) and "top-$k$ with auxiliary loss (Switch-style)" (`:18`, only if it names Mixtral).
    - With: "the paper does not describe its load-balancing loss". The claim is unsupported rather than contradicted, but E-046 already lists it as a fixed never-true claim, so leaving it makes the errata page false.
    - Cite: Mixtral arXiv:2401.04088 (0 hits for "auxiliary" or "Switch"); `fact-sheets/mixtral.md`. Add §5.9 and §5.11 to E-046's "Where".

17. **Solution 5.7 part 4 (`exercises/solutions/ch05.md:306`).**
    - Replace: "The bias approach achieves balance **without touching the loss at all.** … DeepSeek's report is explicit …"
    - With: "The bias does not touch the loss; V3 still keeps a complementary sequence-wise balance loss with a tiny weight (α = 0.0001) to avoid extreme imbalance within a single sequence."
    - Cite: V3 §2.1.2 ("Complementary Sequence-Wise Auxiliary Loss"). Add to E-010's "Where".

18. **Solution 6.9 part 2 (`exercises/solutions/ch06.md:356`).**
    - Replace: "for 60 layers.** That is 120 blocking collectives".
    - With: "for 61 layers.** That is 122 blocking collectives".
    - Cite: V3 §4.2 ("the number of Transformer layers to 61"). Add to E-008's "Where".
    - Optional: `:360` "~900 GB per step" becomes about 870 GB (58 MoE layers).

19. **Ch 5 §5.3 (`05-architecture.md:209`) and Ch 22 §22.3 (`22-inference-serving.md:55`).**
    - Replace: "the DeepSeek team reports a 93.3% reduction vs MHA on a 128K context" and "DeepSeek-V2 reports a KV cache reduction of roughly 93% against its MHA baseline".
    - With: "DeepSeek-V2 reports a 93.3% smaller KV cache than DeepSeek 67B (a GQA model)".
    - Cite: DeepSeek-V2 arXiv:2405.04434 Abstract; DeepSeek LLM arXiv:2401.02954 §2.1 ("the 67B model uses Grouped-Query Attention (GQA)").

20. **Ch 23 §23.6 (`23-evaluation.md:112`, ref note `:223`).**
    - Replace: "Llama 3 [5] and DeepSeek [1] both discuss decontamination in their reports."
    - With: "Llama 3 [5] reports a per-benchmark contamination analysis (§5.1.4); the DeepSeek-V3 report does not discuss decontamination." Drop "Decontamination reporting" from the [1] reference note.
    - Cite: V3 arXiv:2412.19437 (0 hits for "contamination" outside its references).

## UNSUPPORTED (label or source)

- `09-mid-training.md:242`, "Llama-3 uses a YaRN-style rescaling for its 128K context". The paper names no method. Either cite the Llama 3.1 `config.json` `rope_scaling` (`rope_type: "llama3"`, a frequency-banded interpolation) and call it "YaRN-like", or delete the sentence.
- `03-data-pipelines.md:377`, Qwen3 "Aggressive dedup." The Qwen3 report is silent (0 hits). Replace with "dedup method not published".
- `03-data-pipelines.md:381`, R1 "RL data: 600K reasoning prompts, plus preference data". R1 v1 gives no RL prompt count. Delete it, or source it to the *Nature* SI.
- `solutions/ch04.md:222`, DeepSeek and Qwen "trained on Chinese-heavy corpora". No split is published (E-045). Say "both put more Han tokens in their vocabularies (our count)", or label it an inference.
- `05-architecture.md:18`, `:517`, `:633`, Mixtral "Switch-style aux loss". Paper silent; see action item 16.
- `solutions/ch07.md:51`, "EP degree is usually 8 (one node)". No source, and V3 uses EP64. Label it as a rule of thumb.
- `01:88`, `03:140`, `:144`, quality classifier "often/typically an XGBoost". No source. Llama 3 uses fastText and DistilRoberta. Soften, or cite an example.
- `01:260`, "A well-engineered frontier run has an MTBF of 4–8 hours at 2,048-GPU scale". No source. Llama 3's figure is one interruption every ~3.1 h at 16K GPUs (our arithmetic, E-034).
- `09-mid-training.md:37`, "The DeepSeek-V3 paper describes exactly this kind of multi-stage approach" (8K→128K, upweighted code/math, LR to 10%, 5–10% of budget). V3 describes 4K→32K→128K YaRN extension (§4.3) and its LR schedule, but no mid-training mix shift. Weaken "exactly".
- `08-optimization.md:391`, "the annealing phase (the last 5–10% …) provides most of the downstream benchmark gains". No source, and it conflicts in spirit with Llama 3's 40M-token annealing (`09:67`). Source it, or soften.
- `labs/lab07_collective_bandwidth.py:78–79`, the 11× ratio as "the single most important number in Chapter 7". Label it "on H100". The book's anchor cluster (H800) has 3.2× (V3 §3.2.2).
- `book/how-to-use-this-book.md:11`, "real configurations from real technical reports", and `book/appendix/style-guide.md:79`, "real or real-shaped code". Align both with the real/illustrative rule that the uncommitted README edit now states.

## Counts

Across the 122 table rows:

| Class | Count |
|---|---|
| FIXED | 65 rows. 3 of them are fixed only in part, and their leftover lines are also counted under SP-NT: Ch4 #3, Ch8 #1 and Ch8 #3. |
| STILL-PRESENT-NEVER-TRUE | 20 rows, which become 20 action items |
| STALE | 31 rows |
| UNSUPPORTED | 6 rows, which become 12 items in the list above |
| N/A (stipulated premise) | 3 rows |
