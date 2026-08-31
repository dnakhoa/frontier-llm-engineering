# Completion plan

Working document. What is actually left before this book is finished, measured
rather than estimated, and in what order.

Last measured: 2026-08-31, at commit `1a65085`.

---

## 1. Where the book actually stands

Word counts below are measured from the repository, not quoted from the README.

| Component | Files | Words | Status |
|---|---:|---:|---|
| Chapters | 26 | 115,193 | **Complete** |
| Front matter | 4 | 5,426 | **Complete** |
| Appendix (glossary, references, style guide) | 3 | 8,697 | **Complete** |
| Exercise sets | 26 | 21,772 | **Complete** |
| Worked solution sets | 28 | 102,767 | **Complete** |
| **Total prose** | **87** | **253,855** | **Complete** |
| Labs | 8 of 19 | 9,125 (lab prose) | **In progress** |

**All prose is written.** The remaining work is labs, and the documentation
that describes them.

### A correction to make

`README.md` describes the chapters as "~135,000 words". The measured figure is
**115,193**. The companion claim of "~125,000 words" for exercises and
solutions is accurate (124,539). The chapter figure should be corrected to
~115,000 — a book that insists on honest numbers should start with its own.

---

## 2. What "half way" means

Eight of nineteen labs exist. Chapters and exercise sets name eleven more that
do not, and `labs/README.md` lists them by name so no reader hits a dead link.

Done: `lab03_dedup_and_quality` · `lab04_train_a_bpe_tokenizer` ·
`lab05_attention_variants` · `lab05_moe_routing` ·
`lab06_parallelism_memory_model` · `lab07_collective_bandwidth` ·
`lab08_scaling_laws` · `lab11_sft_packing`

Remaining, in book order:

| # | Lab | Ch | What it has to demonstrate | Risk |
|---|---|---|---|---|
| 1 | `lab08_precision_and_stability` | 8 | FP8 per-tensor scaling destroyed by one outlier; FP32 accumulation off at 8,192 elements | Low — pure numerics, exact |
| 2 | `lab09_rope_extension` | 9 | Perplexity vs position past the training length; PI, NTK-aware, YaRN; brief fine-tune after each | **High** — needs a real trained model and four extension schemes |
| 3 | `lab12_reward_model` | 12 | Bradley–Terry RM from preference pairs; length bias; reward hacking made visible | Medium |
| 4 | `lab13_ppo_minimal` | 13 | PPO clip objective, KL penalty, advantage estimation, and what each stops | **High** — RL is unstable at toy scale |
| 5 | `lab14_dpo_from_scratch` | 14 | DPO loss, the implicit reward, beta sweep, comparison against the RM above | Medium |
| 6 | `lab15_grpo_countdown` | 15 | GRPO group-relative advantage on a verifiable task | **High** — same RL instability |
| 7 | `lab16_best_of_n_and_prm` | 16 | Best-of-N scaling, ORM vs PRM, where search-time compute pays | Medium |
| 8 | `lab20_triton_fused_kernel` | 20 | Fusion arithmetic and memory traffic; a real fused op | **High** — no GPU in CI, needs an honest CPU design |
| 9 | `lab21_checkpoint_resume` | 21 | Bit-exact resume; the RNG/optimiser/dataloader state people forget | Low |
| 10 | `lab22_kv_cache_and_batching` | 22 | Continuous batching vs static; prefill/decode; throughput vs latency | Medium |
| 11 | `lab23_contamination_check` | 23 | N-gram contamination detection and its false-positive rate | Low |

---

## 3. Constraints every lab has to satisfy

These are enforced by CI (`tools/run_labs.py`, `tools/check_structure.py`,
`tools/build_notebooks.py --check`) and by the design rules in
`labs/README.md`.

- **Standard library plus torch.** No downloads, no network, no dataset fetch.
- **CPU is the target.** GPU may accelerate; it is never required.
- **`FLE_SMOKE_TEST=1` must shrink the lab to seconds** — this is what CI runs,
  so *the printed narration must be true in smoke mode too*.
- **Python 3.9 compatible** (`from __future__ import annotations` first, before
  any other statement).
- **Every lab must be referenced from a chapter or exercise set** or
  `check_structure.py` fails. All eleven already are.
- **Honest about what it demonstrates.** A lab that overclaims is worse than no
  lab.

### The process rule, learned the hard way

Write the **code first, run it, then write the prose to match the output.**

Writing prose first produced three confidently wrong claims in `lab03` and one
in `lab05`, each of which survived until the output was read carefully. Where a
claim depends on a measurement, the narration should be *derived from the
measured value* rather than hardcoded, so it cannot drift and cannot lie in
smoke mode.

Corollary: **negative results ship.** `lab05` section 5 documents an ablation
that cannot rank the variants it was built to compare, because recognising an
underpowered experiment is worth more than a table that ranks noise.

---

## 4. Sequencing

Ordered to put the cheap, certain labs first and to group the three RL labs
together, since they share machinery (a policy, a reference model, a sampler, a
verifiable task).

**Batch A — numerics and infrastructure (low risk).**
`lab08_precision_and_stability`, `lab21_checkpoint_resume`,
`lab23_contamination_check`. Exact, fast, no training-stability risk.

**Batch B — serving and search (medium).**
`lab22_kv_cache_and_batching`, `lab16_best_of_n_and_prm`. Both build on
`lab05_attention_variants`, which already has a verified KV cache.

**Batch C — preference learning (medium).**
`lab12_reward_model`, then `lab14_dpo_from_scratch`, which compares against it.
Building the RM first makes DPO's implicit reward a measurable claim.

**Batch D — RL (high risk, do last).**
`lab13_ppo_minimal`, `lab15_grpo_countdown`. Shared infrastructure, and the
most likely to need several attempts to become stable enough to teach from.

**Batch E — the two hard singletons.**
`lab09_rope_extension` (needs a genuinely trained long-context model) and
`lab20_triton_fused_kernel` (needs a design that is honest without a GPU).

**Batch F — documentation.** Once the labs land:

- `labs/README.md`: move eleven labs from "does not exist yet" into the table,
  with real measured runtimes; relax the "~400 lines" rule, which no lab has
  respected since `lab04`.
- `README.md`: "Five labs" → nineteen; fix the 135,000-word claim; update the
  "what is complete" section, which currently exists to admit the lab gap.
- `CHANGELOG.md`: one entry per batch.
- `book/appendix/style-guide.md`: record the labs-only first-person convention
  (done).
- `book/learning-paths.md`: verify every lab reference now resolves to a file.

---

## 5. Definition of done

The book is finished when all of the following pass on a clean checkout:

1. `python3 tools/check_links.py` — every internal link and anchor resolves.
2. `python3 tools/check_structure.py` — no orphaned labs, every chapter has
   exercises and solutions.
3. `python3 tools/build_notebooks.py --check` — committed notebooks match their
   sources.
4. `FLE_SMOKE_TEST=1 python3 tools/run_labs.py` — 19 of 19 labs run clean.
5. Every lab also runs correctly *without* `FLE_SMOKE_TEST`, with its printed
   narration true in both modes.
6. No document claims something the repository does not contain.

---

## 6. Open question

The instruction that prompted this document was to "finish the full 135k words
write". That number matches the README's (overstated) description of the
chapters, which are already complete — so the intent is ambiguous. Three
readings:

1. **Finish the book** — i.e. the eleven labs plus documentation. This plan
   assumes this reading.
2. **Retrofit the 26 chapters into the learning-journey voice.** Explicitly
   declined earlier in favour of "new labs only", and it would rewrite
   115,193 words of already-published prose.
3. **Write ~135,000 words of genuinely new prose**, which would mean new
   chapters, and would need a table of contents before anything else.

Confirmation would change the plan substantially; work continues on reading 1
until told otherwise.
