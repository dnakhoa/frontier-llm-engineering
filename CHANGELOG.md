# Changelog

The frontier moves; this file records when the book moved with it.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0] — 2026-08-31

The book as first published, tagged `v1.0`. It is kept unchanged as the record that the
errata page quotes from: the 2026-09 audit found claims in it
that contradict their own sources, and v1.0.1 corrects them.

### Changed
- **Reading-time estimates made mutually consistent.** They implied anywhere from 76 to 251 words per minute, and Chapter 7 (8,791 words) claimed *less* reading time than Chapter 23 (3,054 words). Eight chapters were raised so that none implies faster than 150 wpm; nothing was lowered, so no chapter now under-warns a reader. The remaining 76–148 wpm spread is deliberate and tracks difficulty. The style guide records both rules.
- **The style guide's own Length section was wrong.** It gave long chapters as 3,000–5,000 words and listed Chapters 3, 6, 7, 8, 10, 13, 15, 19, 20, 22 as "in this range" — but Chapters 6, 7, 8 and 10 are 6,497–8,791 words, and Chapter 5 (7,378) was not listed at all. Its total target of ~80,000–100,000 words was also stale against an actual ~116,000. Replaced with measured bands and a rule for checking new chapters.
- **Chapter 2's headcount figures are now labelled as estimates.** §2.7 gave specific team sizes for twelve labs with no source and no caveat, which contradicts the book's own convention that where a lab has not published something, we say so instead of guessing. No lab publishes foundation-team headcount; the section now says so and explains what the ranges are assembled from.
- **Reference count corrected from 96 to 95.** Entry 29 ("PagedAttention and vLLM") was a heading with no body, duplicating entry 23 (Kwon et al., the vLLM paper), cited nowhere. It is now an explicit pointer to 23 rather than an empty stub; the number is retained so no other citation needs renumbering.
- **Every chapter now ends with an Exercises pointer, and every chapter with a lab links to it.** Ten chapters (all of Parts I and II) previously had no practice footer at all, which made the Part II labs unreachable from their own chapters.
- **Seven lab descriptions corrected to match what the labs measure.** Several promised results the finished labs do not produce: `lab23`'s blurb said paraphrase defeats both detectors (the fuzzy detector catches it — *translation* defeats both); `lab15`'s said you would watch response length grow (on a task where length is not instrumental it *collapses*, which is the sharper version of §15.6's point); `lab14`'s and `how-to-use-this-book`'s said the DPO loss is checked against `trl`, which is not available offline, so the lab checks it against the definition and an independently written implementation instead.
- **The chapter word count in `README.md` corrected from ~135,000 to ~116,000**, the measured figure. The ~125,000 for exercises and solutions was accurate.
- **The labs design rule relaxed from "under ~400 lines".** No lab has respected it since `lab04`; the rule now describes what it was actually protecting.
- **The style guide now documents the labs' first-person register**, which deliberately differs from the chapters, along with the process rule behind it: write the code, run it, then write the narration to match, and derive printed claims from measured values so they cannot drift.
- **The site is now built with mdBook** (the same toolchain and rust/navy theme as Everything Data Structures), replacing Honkit. All published URLs are unchanged. Math is rendered client-side with KaTeX using Pandoc-style `$` delimiters — inline math must hug its dollar signs, which is what keeps prose like "costs $2 and $15" from being parsed as math. Mermaid diagrams render client-side and follow the light/dark theme.

### Added
- **Per-chapter reference blocks for Chapters 2 and 3.** Chapter 3 cited eight sources inline and had no block at all; Chapter 2 had neither block nor citations. Chapter 8 was missing [39] and Chapter 10 was missing [5] and [8] from blocks that otherwise listed everything they cited. Every inline citation in all 26 chapters now appears in that chapter's own reference block.
- **Chapters 11-26** — Parts III (post-training), IV (infrastructure), and V (the job), written from scratch. ~90,000 words across 16 chapters.
- **All 26 exercise sets and 26 worked solution sets** (~125,000 words): arithmetic drills, design problems, paper-reading prompts, and lab experiments with stated predictions.
- **All 19 runnable labs**, CPU-first and every one of them run by CI on each commit. The first five: BPE tokenization and multilingual fertility (Ch 4), MoE routing and load imbalance (Ch 5), a distributed-training memory and throughput model (Ch 6), a collective-communication cost model (Ch 7), and SFT loss masking and sequence packing (Ch 11). The remaining fourteen:
  - `lab03_dedup_and_quality` — MinHash, LSH banding, and a quality classifier whose negative class turns out to *be* the corpus policy.
  - `lab05_attention_variants` — MHA/GQA/MQA/MLA as one module; Exercise 5.2's arithmetic checked against measured bytes; cached decode verified against a full forward pass.
  - `lab08_scaling_laws` — fit `L(N) = E + A·N^-α` against a source whose irreducible loss is computable, then check the extrapolation.
  - `lab08_precision_and_stability` — real FP8/BF16/FP16 casts; per-tensor versus per-block scaling; accumulator precision.
  - `lab09_rope_extension` — RoPE's wavelength table, then PI, NTK-aware and YaRN measured by position.
  - `lab12_reward_model` — a Bradley–Terry RM that scores 100% held-out while learning nothing about correctness.
  - `lab13_ppo_minimal` — PPO and RLOO with the guardrails removed one at a time.
  - `lab14_dpo_from_scratch` — the DPO loss checked against the definition and an independent implementation.
  - `lab15_grpo_countdown` — group-relative advantage, the zero-advantage problem, and reward shaping that destroys accuracy.
  - `lab16_best_of_n_and_prm` — majority voting versus best-of-N versus PRM aggregators.
  - `lab20_triton_fused_kernel` — fusion's memory-traffic ceiling, `torch.compile` as the real baseline, and the Triton kernel itself.
  - `lab21_checkpoint_resume` — resume correctness as bit-exactness, and logical versus physical shard layout.
  - `lab22_kv_cache_and_batching` — continuous batching, paged allocation, prefix sharing.
  - `lab23_contamination_check` — what an n-gram detector finds, and the three things it cannot.
- References 51-96, covering the post-training, infrastructure, and evaluation literature.
- Open-repository scaffolding: dual license (CC BY-SA 4.0 for prose, MIT for code), contribution guide, code of conduct.
- mdBook configuration (`book.toml`) plus a theme that renders KaTeX math and Mermaid diagrams client-side, so the repo builds into a book with no further setup.
- `tools/check_links.py` — validates every internal link *and* every heading anchor across the book.
- `tools/check_structure.py` — enforces that every chapter has exercises, every exercise set has solutions, and no lab is orphaned.
- `tools/build_notebooks.py` — generates Colab-ready `.ipynb` files from the `.py` labs.
- `tools/run_labs.py` — runs every lab; CI runs it on a CPU-only runner on each commit.
- CI workflow running all four checks.

### Known gaps
- **`lab20_triton_fused_kernel` needs a GPU for part of what it teaches.** Triton compiles to GPU code and CI has no GPU. The lab states which sections are exact, which are measured on CPU, and which require hardware, and includes the kernel source in full rather than substituting a fake.
- **Chapters 2 and 3 have no per-chapter reference block**, unlike the other 24 chapters. Their citations are in the full reference list.

### Fixed
- **232 broken reference links.** Every chapter linked to `appendix/b-references.md`, which resolves relative to the chapter's own directory and therefore pointed at nothing. Corrected to `../appendix/b-references.md`.
- **32 broken heading anchors**, including 23 instances of `#5-llama3` (the heading slug is `#5-llama-3`), plus `#50-dualpipe`, `#8-dedup`, and `#48-ntk-aware-scaling`.
- Reference 44 (`µTransfer`) and reference 29 (`PagedAttention and vLLM (see #23)`) had headings whose slugs contained a non-ASCII character and a `#`, making their anchors unlinkable. Renamed to ASCII-only headings.
- Removed a stray build report (`06-deliverable.md`) that was sitting inside the book tree and colliding with Chapter 6's number. It now lives in `.meta/build-reports/`.

## [0.1.0]

### Added
- Preface, Part I (Chapters 1–2), Part II (Chapters 3–10).
- Glossary and a 50-entry reference list.
- Style guide.
