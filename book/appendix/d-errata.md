# Errata

This page lists every **never-true claim** the book has published: statements about what a primary source says that the source contradicts, and places where the book contradicted itself. Each was wrong on the day it was written. They are listed here in full, with what the source actually says and the version that fixed them. A book that asks you to read reports critically should show you its own mistakes.

Stale claims are different: those were accurate when written and have since been overtaken by newer work. They are not listed here. They go in the [changelog](../../CHANGELOG.md), and each chapter's currency stamp tells you how recently it was checked.

The text as first published is tagged `v1.0` in the repository, so every "the book said" below can be checked.

**If you worked an exercise before v1.0.1**, look for an *If you worked this before v1.0.1* line. Some corrections change a final answer, and a few reverse the conclusion.

---

### E-001 — The DeepSeek-V3 parallelism layout

- **The book said:** "4 (PP) × 8 (EP, which is also the node size) × 64 (DP)", over "60 transformer layers" split into 4 stages of 15, with "small TP" for dense layers. It also said V3 kept expert parallelism inside a node, on NVLink, and that this was "why DeepSeek's EP choice is exactly the node size".
- **The source says:** 16-way pipeline parallelism, 64-way expert parallelism spanning 8 nodes, ZeRO-1 data parallelism, and no tensor parallelism at all, over 61 layers. Expert parallelism deliberately crosses node boundaries. That is affordable because DualPipe overlaps the all-to-all with compute, and because each token is routed to at most 4 nodes. See the [fact sheet](fact-sheets/deepseek-v3.md#parallelism-layout).
- **Where:** Chapter 1 §1.5; Chapter 6 §6.10, §6.11, §6.13 and takeaways 8–9; Chapter 7 §7.3 and §7.4; Chapter 10 §10.7, its parallelism block, and takeaway 4; Exercise 6.6 and its solution. Solutions 6.9 and 10.1 already gave the correct layout, contradicting the chapters; they now link to the fact sheet.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Exercise 6.6 part 4 now uses 58 MoE layers, so the answer is ~55 GB per micro-batch rather than 56 (~870 GB per step rather than ~900). Part 5's conclusion changes more. "EP belongs inside a node" is a sound default, but it is not what V3 did, and V3 is the book's anchor example. V3's design point is cross-node EP made cheap, and it is the more interesting lesson.
