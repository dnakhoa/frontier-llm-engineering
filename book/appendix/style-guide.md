# Style guide

> This is the style guide for the rest of the book. Every chapter should follow it. If you are reading the book and a chapter deviates, that's a bug — please open an issue.

## Voice

- **Direct.** We say things. We do not hedge. We use "the model is" not "the model may be considered to be."
- **Specific.** Real numbers, real papers, real names. No "some researchers have suggested." If a lab has not published, we say so.
- **Casual but technical.** This is not a paper. It is a field guide written by someone who has done the work. We use contractions. We use "we" to refer to the field. We use humor sparingly and only when earned.
- **Honest about uncertainty.** Where the field does not know, we say so. Where a technique is hotly debated, we say so. Where a lab has not published, we say so.

## Structure

Every chapter has:

1. **A header block** with reading time and a one-line summary of what the reader will know after.
2. **A motivating example or case study** in the first section. Anchor on a real lab / real paper.
3. **The technical content** in numbered sections, each with a clear topic.
4. **A "JD, decoded" or "what this means in practice" section** near the end, mapping the chapter content to actual frontier-lab job roles.
5. **A "what you should take from this chapter" section** with 3–7 numbered takeaways.
6. **A "references for this chapter" block** with `[N]`-style links to the appendix references.

## Length

- Long chapters: 3,000–5,000 words (Chapters 3, 6, 7, 8, 10, 13, 15, 19, 20, 22 are in this range).
- Medium chapters: 2,000–3,000 words.
- Short chapters: 1,500–2,000 words (the role-specific chapters in Part V).

Total target: ~80,000–100,000 words across 26 chapters.

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
