# How to use this book

> Reading time: ~10 minutes. By the end you should know which chapters you need, in what order, how the exercises and labs fit in, and how to tell whether a chapter actually landed.

This book is designed to be worked through alone, without a cohort, an instructor, or a Slack channel to ask questions in. That constraint shaped the structure. Every chapter has a way to check whether you understood it, and every mechanism that can be made concrete in fifty lines of Python has been.

## The three layers

Each chapter has three layers, and you can stop at any of them.

**Layer 1 — the chapter.** Prose, math, real configurations from real technical reports. Reading a chapter takes 20–45 minutes. If you only read the chapters, you will be able to follow a frontier-lab technical report and hold a conversation with someone who does this work. That is a real outcome and it is enough for many readers.

**Layer 2 — the exercises.** A problem set per chapter, in [`exercises/`](../exercises/README.md), in three flavours:

- **Arithmetic drills.** "A 70B model in BF16 with AdamW — how much memory before you have allocated a single activation?" These are the calculations a pre-training engineer does on a whiteboard, and they are the fastest way to find out whether you actually internalized a chapter or just recognized the words.
- **Design problems.** "You have 512 H100s and a 200B MoE. Choose TP, PP, EP, and DP. Justify each." There is rarely one right answer; the solutions walk through the reasoning and name the trade-offs.
- **Paper-reading prompts.** "Open the DeepSeek-V3 report to §3.2. What does the DualPipe schedule buy them that 1F1B does not?" These build the skill the book cannot give you directly: reading a frontier technical report and extracting the engineering from the marketing.

Every exercise has a worked solution in [`exercises/solutions/`](../exercises/solutions/README.md). Write your answer down *before* opening the solution — the gap between "I could have gotten that" and "I got that" is where learning lives.

**Layer 3 — the labs.** Runnable Python in [`labs/`](../labs/README.md). Not toy demos: each lab implements the actual mechanism a chapter describes, small enough to run on a free Colab T4 or a laptop CPU. You train a BPE tokenizer and measure its fertility on five languages. You implement top-k MoE routing and watch expert collapse happen. You write the DPO loss from scratch and check it against the definition to the last decimal place. You run GRPO on a task with a verifiable reward and find out what actually decides whether response length grows or collapses.

The labs are where the abstractions stop being abstractions.

## Reading the chapters

Every chapter opens with a header block giving an estimated reading time and a one-sentence statement of what you should be able to do afterwards. Take that sentence seriously — it is the exit criterion, and the exercises test exactly it.

Under the header is a **currency stamp**: *Current as of early 2025*, or a month and year. It gives the date the chapter's claims were last checked against the field, not the date it was last edited, so a typo fix or a correction leaves it alone. This field moves by the quarter, so chapters age at different rates. A chapter stamped "early 2025" may be accurate about everything it says and still be missing a year of work. The book's own mistakes are listed on the [errata page](appendix/d-errata.md), and the numbers it relies on are collected, with sources, in the [fact sheets](appendix/c-fact-sheets.md).

The chapter body is numbered by section (§6.4, §11.2) so the exercises and solutions can point at specific arguments. When a solution says "see §8.7," it means it.

Every chapter ends with:

- **"JD, decoded"** or **"what this means in practice"** — mapping the content to the actual job roles that do this work. If you are reading to get hired, these sections are the spine of the book.
- **Takeaways** — 3 to 10 numbered claims. If you cannot explain each one to someone else, reread the section it came from.
- **References for this chapter** — the primary sources. The book is a map; the papers are the territory.

## Which chapters you actually need

You almost certainly do not need all 26 chapters, and reading them in order is not the fastest route to what you want. [Learning paths](learning-paths.md) lays out several routes — a pre-training track, a post-training track, an inference track, a "get through a frontier interview" track, and a shortest-path skim.

Two rules that hold regardless of the path you pick:

**Chapters 1 and 2 come first, always.** They are the shortest chapters in the book and they define the vocabulary everything else uses. Chapter 1 in particular walks one real training run end-to-end, so every later chapter has a place to attach.

**Chapters 6, 8, and 20 are the hard ones.** Distributed training, optimization, and custom kernels. They are dense on purpose, and they are the chapters most people bounce off. Skim them on a first pass and come back. Nothing later in the book breaks if you have only a rough model of 1F1B pipeline scheduling on your first read.

## How to tell whether a chapter landed

The honest test is not "did I finish reading it." Three better tests, in ascending order of rigour:

1. **Explain the takeaways out loud** without looking. Most failures show up here — you will find yourself saying "and then the gradients get... synchronized somehow" and that "somehow" is the thing you did not learn.
2. **Do the arithmetic drills** from the exercise set. These are unforgiving in a useful way: either the number comes out or it does not.
3. **Run the lab and break it.** Change the number of experts, turn off the auxiliary loss, set the KL coefficient to zero. Predict what will happen before you run it. When your prediction is wrong, you have found the edge of your model of the system, which is exactly the thing worth finding.

## Pacing

Some realistic numbers, based on the chapter lengths:

- **Skim pass over the whole book:** 12–15 hours. You get the map, not the territory.
- **Chapters + exercises, one part at a time:** 8–12 hours per part.
- **Chapters + exercises + labs, all 26:** 100–150 hours. This is a semester-length commitment, and it is what the book is designed for.

The [learning paths](learning-paths.md) page turns these into week-by-week schedules if you want the structure.

## If you get stuck

- **The [glossary](appendix/a-glossary.md)** covers every acronym the book uses. EP, CP, MLA, GQA, PRM, RLOO, GRPO, muP — all of it. Frontier writing is dense with jargon and there is no shame in looking things up constantly; everyone does.
- **The [references](appendix/b-references.md)** are the primary sources. If a chapter's explanation does not click, the underlying paper often will — the book compresses, and compression sometimes loses the thing you needed.
- **Prerequisites.** If several chapters in a row feel like they are written in a foreign language, the problem may be upstream. [Prerequisites and self-assessment](prerequisites.md) has an honest checklist and pointers to the standard material for each gap.
- **Open an issue.** If a paragraph is genuinely unclear, that is a bug in the book, not in you. Say which paragraph and what you thought it meant.

## A note on what this book cannot give you

You cannot learn to run a 2,000-GPU training job by reading about it, any more than you can learn to land a plane by reading about it. What you can get from this book is the model: what the pieces are, why they are shaped the way they are, what breaks, and what the people who do this for a living are actually worried about on a given Tuesday.

That model is most of the gap between "I have fine-tuned a model" and "I could contribute to a frontier run." The rest of the gap closes on the job, with a cluster you did not pay for.

---

Next: [Prerequisites and self-assessment](prerequisites.md) to check your footing, or straight to [Chapter 1](part-1-frontier/01-what-a-frontier-run-looks-like.md) if you would rather find the gaps by hitting them.
