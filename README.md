# Frontier LLM Engineering

> A free, open field guide to how large language models are actually built at the labs that train their own — from data pipelines at petabyte scale, through trillion-token pre-training, to the post-training that turned base models into reasoning systems.

[![CI](https://github.com/dnakhoa/frontier-llm-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/dnakhoa/frontier-llm-engineering/actions/workflows/ci.yml)
[![Prose: CC BY-SA 4.0](https://img.shields.io/badge/prose-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE-CODE)

**📖 Read it online: [dnakhoa.github.io/frontier-llm-engineering](https://dnakhoa.github.io/frontier-llm-engineering/)** — full book with rendered math, diagrams, and search, republished automatically on every commit.

**26 chapters. 19 runnable labs, CPU-first, every one tested in CI. Exercises with worked solutions for every chapter. Free, with no paywall, application or cohort.**

---

## What you'll build

Each lab is one Python file that runs in Colab or on a laptop CPU in seconds to minutes. Each one builds a piece of a frontier training stack from scratch and measures something about it that people usually get wrong. These are the numbers the labs print at full size, on a laptop CPU:

| Build this | and find out that… | Open |
|---|---|---|
| **GRPO** on a countdown-arithmetic task | a 0.05-per-token "thinking" bonus takes accuracy from **0.996 to exactly 0.000** — the policy pads its reasoning to collect the bonus | [lab15](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab15_grpo_countdown.ipynb) |
| A **Bradley–Terry reward model** | it scores **100%** on held-out pairs and **48.5%** — below chance — once response length stops giving the answer away | [lab12](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab12_reward_model.ipynb) |
| A **byte-level BPE tokenizer** | training its merges on English alone makes Vietnamese cost **~2×** as many tokens | [lab04](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab04_train_a_bpe_tokenizer.ipynb) |
| **MinHash + LSH** deduplication | exact hashing finds only **~48%** of the duplicates you planted | [lab03](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab03_dedup_and_quality.ipynb) |
| An **MoE router**, then DeepSeek's auxiliary-loss-free balancer | an unbalanced router leaves the busiest expert **3.4×** the mean load, and the whole step waits for it | [lab05](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab05_moe_routing.ipynb) |
| **Continuous batching** and a **paged KV allocator** | continuous batching is **5.8×** static on a heavy-tailed request stream; paging cuts fragmentation from **86.5% to 0.7%** | [lab22](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab22_kv_cache_and_batching.ipynb) |
| A **parallelism memory model** for any TP/PP/EP/DP/ZeRO layout | why DeepSeek-V3 ran expert parallelism across 8 nodes, when a naive cost model says that should be ~26× slower | [lab06](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab06_parallelism_memory_model.ipynb) |
| **Checkpoint resume** tested for bit-exactness | every incomplete resume recipe diverges on the **first** step after the restart, silently | [lab21](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab21_checkpoint_resume.ipynb) |
| An **n-gram contamination detector** | it catches **100%** of verbatim copies and **0%** of paraphrases | [lab23](https://colab.research.google.com/github/dnakhoa/frontier-llm-engineering/blob/main/labs/lab23_contamination_check.ipynb) |

**Zero install:** click any lab above. It opens in Colab, installs what it needs and runs top to bottom. The [labs index](labs/README.md) has all 19 and the chapter each one belongs to.

**On accuracy and currency.** The book corrects itself in public. Every claim it once made that its own sources contradict is listed, with the fix, on the [errata page](book/appendix/d-errata.md). The numbers it relies on are collected with pinpoint citations in the [fact sheets](book/appendix/c-fact-sheets.md). Each chapter carries a *Current as of* stamp; most still read "early 2025", and the post-training chapters are being refreshed first.

## Why this exists

How a frontier model gets built is not secret. It is spread across a few dozen technical reports, a few hundred papers, some very good blog posts, and a lot of tacit knowledge that never gets written down. Assembling it is the hard part, and people pay thousands of dollars for courses that do the assembly. This repository does it in the open, for free: star it, fork it, translate it, teach from it. The license allows all of that, including commercially.

## Who this is for

- **ML/AI engineers** who know transformers and PyTorch and want to understand what happens when the same ideas scale to 10,000 GPUs, 14 trillion tokens, and a six-figure daily compute bill.
- **Graduate students** moving from coursework into frontier-lab internships, looking for the unstated context papers leave out.
- **Self-learners** with a laptop and a Colab tab who want a structured path instead of a reading list.
- **Researchers** who want to understand how systems-level choices in modern training runs shape the scientific questions they can ask.

It is *not* a fine-tuning tutorial, a survey of LLM applications, or a distributed-systems textbook with transformers stapled on. The labs teach mechanisms, not recipes.

## Start here

| If you are… | Start with |
|---|---|
| Brand new to the book | [How to use this book](book/how-to-use-this-book.md) |
| Not sure you have the background | [Prerequisites and self-assessment](book/prerequisites.md) |
| Looking for a structured route | [Learning paths](book/learning-paths.md) |
| Here for one specific topic | [The full table of contents](SUMMARY.md) |
| Here to run code | [The labs](labs/README.md) |

New readers: read the [Preface](book/preface.md), then [Chapter 1](book/part-1-frontier/01-what-a-frontier-run-looks-like.md). Chapter 1 walks a real training run end-to-end and tells you which later chapters you actually need.

## What's in the box

```
book/          26 chapters across 5 parts, plus glossary, 97 references,
               fact sheets and the errata page
exercises/     26 problem sets — arithmetic drills, design problems,
               paper-reading prompts — each with a worked solution set
labs/          19 runnable labs you can paste straight into Colab
tools/         link and structure checkers, the real-config generator,
               notebook builder, lab runner, and their tests
docs/          the 2026-09 audit, the glossary's decision records (ADRs)
```

### What is complete

All 26 chapters (~119,000 words), all 26 exercise sets and all 26 worked solution
sets (~129,000 words), the glossary, 97 references, and **all 19 labs** — every
lab the chapters name now exists, runs, and is checked by CI on every commit.

Two caveats worth stating plainly, because the book asks the same of the papers
it cites:

- **[`lab20_triton_fused_kernel`](labs/lab20_triton_fused_kernel.py) needs a GPU
  for part of what it teaches.** Triton compiles to GPU code. The lab opens with
  a table of which sections are exact arithmetic, which are measured on CPU, and
  which require hardware; the kernel source is included in full and runs where a
  GPU exists. Nothing is faked.
- **"Complete" does not mean "current".** A primary-source audit in September
  2026 found claims the book's own sources contradict. Those are fixed in v1.0.1
  and listed on the [errata page](book/appendix/d-errata.md). It also found that
  most chapters' sources stop in early 2025. Refreshing them is under way, and
  each chapter's currency stamp says where it stands.

Corrections and better labs are still very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

### The five parts

1. **The frontier** — what a real training run looks like, who does what.
2. **Pre-training** — data, tokenization, architecture, distributed training, cluster reality, optimization, mid-training.
3. **Post-training** — SFT, reward modeling, PPO/RLOO, DPO, GRPO, process rewards, Constitutional AI, tool use.
4. **Infra** — kernels, checkpointing, serving, evaluation.
5. **The job** — translating the JDs, career paths, what's next.

Chapters 6 (distributed training), 8 (optimization), and 20 (kernels) are densely technical and can be skimmed on a first pass.

## Running the labs

Most labs finish on a laptop CPU in seconds; the slowest takes a couple of minutes, and `FLE_SMOKE_TEST=1` shrinks every one of them to seconds (that is what CI runs). None require a download, and only `lab20`'s Triton section wants a GPU. Every lab is a plain `.py` file that is also published as a `.ipynb`. You have three options, in increasing order of setup:

**1. Copy-paste into Colab.** Open any lab's `.py` file, copy the whole thing into a Colab cell, run it. The labs are written to work this way — dependencies self-install, no local files are read, and everything sizes itself to the hardware it finds.

**2. Click the Colab badge.** Every `.ipynb` in `labs/` opens directly in Colab with an "Open In Colab" badge at the top, split into cells.

**3. Run locally.**

```bash
pip install -r requirements.txt
python3 tools/run_labs.py
```

Labs detect a CPU-only machine and shrink their workload automatically. Setting `FLE_SMOKE_TEST=1` shrinks them further — that is what CI uses to prove, on every commit, that the labs still run.

## Reading it as a book

**The hosted book lives at [dnakhoa.github.io/frontier-llm-engineering](https://dnakhoa.github.io/frontier-llm-engineering/)** — 92 pages with KaTeX-rendered math, Mermaid diagrams, and full-text search, rebuilt by CI on every push to `main`. Built with [mdBook](https://github.com/rust-lang/mdBook), the same toolchain as [Everything Data Structures](https://github.com/dnakhoa/everything-data-structures), so the two sites share one look.

To preview locally:

```bash
brew install mdbook   # or: cargo install mdbook
mdbook serve
```

The Markdown also renders correctly on GitHub as-is.

## Contributing

The frontier moves fast; chapters go stale. Corrections, new references, better exercises, and labs that run faster are all welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

The highest-value contributions are: **a citation we got wrong**, **a number that has been superseded by a newer technical report**, and **a lab that fails on current library versions**.

## License

Prose, diagrams, and exercises: [CC BY-SA 4.0](LICENSE) — share and adapt, with attribution, under the same license.
Code in `labs/`, `tools/`, and the chapters: [MIT](LICENSE-CODE) — lift it into anything, including proprietary work.

## Conventions used throughout

- A config block is either **real** — generated from the published file at a pinned commit, and labelled with its origin — or labelled **illustrative**. Nothing in between.
- Citations are inline as `[N]` with full entries in [the reference list](book/appendix/b-references.md).
- "We" refers to the field, not a specific lab, unless a section anchors on one.
- "Frontier lab" means a lab that trains its own foundation model from scratch — Anthropic, OpenAI, Google DeepMind, Meta, xAI, Microsoft, Alibaba (Qwen), DeepSeek, Moonshot, ByteDance, Zhipu, Mistral, and similar.
- Where a lab has not published something, we say so instead of guessing. Facts about a specific model live once, with a citation, in its [fact sheet](book/appendix/c-fact-sheets.md).

## Status

A living document. The frontier moves fast; expect chapters to be revised as new papers and technical reports drop. See [the changelog](CHANGELOG.md) for what moved recently.
