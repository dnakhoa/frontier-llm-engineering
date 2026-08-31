# Completion plan — final status

This began as a working document listing what was left. The list is now empty.
Kept as a record of what "finished" was defined to mean, and of the process
rules that turned out to matter.

Last measured: 2026-08-31, at commit `10353b9`.

---

## 1. Where the book stands

Measured from the repository, not quoted from the README.

| Component | Files | Words | Status |
|---|---:|---:|---|
| Chapters | 26 | 115,763 | **Complete** |
| Front matter | 4 | 5,451 | **Complete** |
| Appendix (glossary, references, style guide) | 3 | 8,939 | **Complete** |
| Exercise sets | 26 | 21,772 | **Complete** |
| Worked solution sets | 28 | 102,767 | **Complete** |
| **Total prose** | **87** | **254,692** | **Complete** |
| Labs | 19 of 19 | 20,880 words of lab prose, 10,061 lines | **Complete** |

Every lab the chapters name exists, runs, and is exercised by CI on each commit.

---

## 2. Definition of done

All of these pass on a clean checkout:

1. `python3 tools/check_links.py` — every internal link and anchor resolves.
2. `python3 tools/check_structure.py` — no orphaned labs; every chapter has
   exercises and solutions.
3. `python3 tools/build_notebooks.py --check` — committed notebooks match their
   sources.
4. `FLE_SMOKE_TEST=1 python3 tools/run_labs.py` — 19 of 19 labs run clean.
5. Every lab also runs correctly *without* `FLE_SMOKE_TEST`, with its printed
   narration true in both modes.
6. No document claims something the repository does not contain.

Item 6 was the expensive one. It is what forced the corrections in section 4.

---

## 3. What is deliberately not finished

Stated here rather than left for a reader to discover.

- **`lab20_triton_fused_kernel` needs a GPU for part of what it teaches.**
  Triton compiles to GPU code; CI has none. The lab opens with a table of which
  sections are exact arithmetic, which are measured on CPU, and which require
  hardware. The kernel source is included in full and executes where a GPU
  exists. `num_warps`, occupancy and achieved bandwidth are named as
  unmeasurable here, with enough reasoning to turn them into predictions.
- **Chapters 2 and 3 have no per-chapter reference block**, unlike the other 24.
  Adding one means choosing citations, which is authorial work rather than a
  consistency pass, so it was left rather than guessed at.
- **`lab05_attention_variants` section 5 is a negative result.** The ablation it
  was built to run cannot rank the attention variants — within-variant seed
  spread reached 72pp because the task is bimodal on whether an induction
  circuit forms. It ships saying so, and pointing at the GQA paper for the real
  evidence.
- **`lab15_grpo_countdown` section 6 is also a negative result.** Oversampling
  and dropping zero-advantage groups — the obvious fix for the problem section 5
  establishes — costs ~1.8x the generation to reach the same accuracy at every
  target tried. The lab explains what published dynamic-sampling schemes do
  differently instead of implying the naive version works.

---

## 4. The process rules that mattered

These were learned by getting it wrong, and they are now in
[the style guide](book/appendix/style-guide.md).

**Write the code, run it, then write the prose.** Writing prose first produced
confidently wrong claims in `lab03` (three of them), `lab05`, `lab08b`,
`lab13` (three), `lab14`, `lab15`, `lab16`, `lab20` and `lab22`. In each case
the text described what the mechanism *should* do and the output disagreed.

**Derive printed claims from measured values.** A hardcoded number in a
`print()` is a claim that will drift. CI runs labs under `FLE_SMOKE_TEST=1`, so
narration has to be true at both sizes — several sections now branch on what
was actually observed.

**The most productive bug was the quietest.** In `lab03`, `DOMAINS` held
strings that were never split into sentence lists, so `rng.sample()` sampled
individual *characters*. Every document was garbage like `"a t a o e o w"`. The
pipeline ran, printed plausible tables, and every number in them was noise.
Nothing failed.

**Check whether two guardrails are doing one job.** PPO's clip appeared to do
nothing until `clip_grad_norm_(1.0)` was removed from the loop — global
gradient clipping had been holding the trust region all along.

**Ship negative results.** Four labs report an experiment that did not work.
Recognising an underpowered ablation is worth more than a table that ranks
noise.

---

## 5. Corrections this pass made to existing prose

Found by building the labs the chapters described, then comparing.

- `README.md` described the chapters as ~135,000 words. Measured: 115,763.
- `lab23`'s chapter blurb said paraphrase defeats both detectors. The fuzzy
  detector catches paraphrase completely; *translation* defeats both, and so
  does a verbatim copy shorter than $n$.
- `lab15`'s blurb promised "watch response length grow on its own". On a task
  where longer reasoning is not instrumental, length *collapses* — which is the
  sharper form of what §15.6 already argues.
- `lab14`'s blurb and `how-to-use-this-book.md` promised a check against `trl`,
  which is not available offline. The lab checks against the definition and an
  independently written implementation.
- Ten chapters had no Exercises footer, leaving the Part II labs unreachable
  from their own chapters.
- Seven lab pointers were unlinked plain code spans.
