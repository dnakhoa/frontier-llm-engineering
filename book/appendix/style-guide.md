# Style guide

> This is the style guide for the rest of the book. Every chapter should follow it. If you are reading the book and a chapter deviates, that's a bug — please open an issue.

## Voice

- **Direct.** We say things. We do not hedge. We use "the model is" not "the model may be considered to be."
- **Specific.** Real numbers, real papers, real names. No "some researchers have suggested." If a lab has not published, we say so.
- **Casual but technical.** This is not a paper. It is a field guide written by someone who has done the work. We use contractions. We use "we" to refer to the field. We use humor sparingly and only when earned.
- **Honest about uncertainty.** Where the field does not know, we say so. Where a technique is hotly debated, we say so. Where a lab has not published, we say so.

## Voice in the labs (and only in the labs)

The chapters use the register above: "we" means the field, and the text is
written from settled understanding. The **labs are different on purpose.**

A lab is a record of someone finding something out, so labs may use the first
person singular and *should* keep the wrong turns visible:

- **Say what you expected before you say what happened.** Every lab opens with
  a "predict before you run" prompt, and sections that produced a surprise say
  so.
- **Leave the failed drafts in.** Where an earlier version of a lab measured
  the wrong thing, that belongs in a comment next to the fix. `lab03` documents
  three measurement artefacts; `lab05` documents an ablation that does not work.
- **Ship negative results.** A lab that honestly reports "this experiment
  cannot answer the question, and here is how I know" teaches more than a table
  that ranks noise. Point at the real evidence in the literature instead.
- **Never let the prose outrun the output.** Write the code, run it, then write
  the narration to match. Where a claim depends on a measured value, derive the
  printed sentence from that value so it cannot go stale — and remember CI runs
  labs under `FLE_SMOKE_TEST=1`, so the narration must be true in that mode too.

What does not change: no marketing language, no moralising, real numbers, and
explicit honesty about what a lab does *not* show.

## Structure

Every chapter has:

1. **A header block** with reading time and a one-line summary of what the reader will know after.
2. **A motivating example or case study** in the first section. Anchor on a real lab / real paper.
3. **The technical content** in numbered sections, each with a clear topic.
4. **A "JD, decoded" or "what this means in practice" section** near the end, mapping the chapter content to actual frontier-lab job roles.
5. **A "what you should take from this chapter" section** with 3–7 numbered takeaways.
6. **A "references for this chapter" block** with `[N]`-style links to the appendix references.

## Length

Measured, not aspirational — these are the current word counts, and a new
chapter should land inside the band for its kind rather than matching a number
someone wrote down once.

- **Long chapters: 6,497–8,791 words** (Chapters 5, 6, 7, 8, 10). The systems-heavy
  chapters, where the arithmetic is the content.
- **Medium chapters: 4,060–5,679 words** (Chapters 2, 3, 4, 9, 11, 12, 13, 15).
- **Short chapters: 2,379–3,771 words** (Chapters 1, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), including the
  role-specific chapters in Part V.

Total: **~116,000 words** across 26 chapters.

### Reading time

Every chapter opens with a `> Reading time: ~N minutes` line. Two rules keep
those numbers comparable to each other:

- **Never imply faster than 150 words per minute.** Nobody follows dense
  technical prose, tables and arithmetic faster than that while actually
  understanding it. A chapter of 7,500 words cannot honestly claim 35 minutes.
- **Slower than 150 wpm is a deliberate signal**, and the book uses it. Chapters
  where the reader is expected to work through derivations sit near 80–110 wpm;
  narrative chapters sit near 130–150. The current range is 76–148 wpm, and the
  slow end is Part III/IV on purpose.

Check a new chapter with `wc -w` before writing the header line.

## Code and configuration

- Use code blocks for any real or real-shaped code: configuration files, model definitions, training scripts, kernel code.
- Comment the code. The reader is a serious engineer, not a beginner.
- Prefer real configurations from public papers and blog posts over invented ones.
- For pseudocode, use Python-like syntax with type hints where helpful.

## Citations

- Cite real papers with `[N]` notation, where N is the index in the [References](b-references.md).
- Use the form `[\[N\]](../appendix/b-references.md#N-slug)` for clickable links.
- Cite real blog posts and technical reports inline with a URL and the lab name.
- Where a technique is widely used but the original source is unclear, cite 2–3 representative papers and say so.
- Where a lab has not published, say so explicitly: "The exact [X] has not been published; the discussion below is based on [secondary evidence]."

## Math

- Use LaTeX in markdown: `$inline$` and `$$display$$`.
- Show the math for the key operations, but do not over-derive. The reader is a practitioner, not a mathematician.
- Include the standard formulations: attention, AdamW, cross-entropy, PPO clip, etc.

## Diagrams

- Where a diagram would clarify (e.g., a flow chart of the data pipeline, a diagram of 3D parallelism), use a Mermaid diagram or a description in a code block.
- Do not require external images; the book should render in any GitBook-compatible viewer.

## Case studies

The book is anchored on two main case studies:

- **DeepSeek-V3 / R1** — for the pre-training and post-training technical depth, because DeepSeek has published the most detail.
- **Qwen3** — for the multilingual / Chinese-side perspective.

Use additional case studies where they are the clearest example of a specific technique (e.g., the Llama-3 data card for data filtering, the Megatron paper for 3D parallelism).

## What not to do

- **No marketing language.** No "revolutionary," "groundbreaking," "next-generation" unless the lab itself used the word in a primary source.
- **No "this is just like X but for LLMs" comparisons** to other fields unless the comparison is illuminating.
- **No moralizing about AI safety or AI progress.** The book describes the work; the reader can form their own opinions.
- **No tutorial-style step-by-step instructions for "build your own LLM."** This is not that book.
