# Frontier LLM Engineering

> A free, open field guide to how large language models are actually built at the labs that train their own — from data pipelines at petabyte scale, through trillion-token pre-training, to the post-training that turned base models into reasoning systems.

[![CI](https://github.com/dnakhoa/frontier-llm-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/dnakhoa/frontier-llm-engineering/actions/workflows/ci.yml)
[![Prose: CC BY-SA 4.0](https://img.shields.io/badge/prose-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE-CODE)

**📖 Read it online: [dnakhoa.github.io/frontier-llm-engineering](https://dnakhoa.github.io/frontier-llm-engineering/)** — full book with rendered math, diagrams, and search, republished automatically on every commit.

**26 chapters. 19 runnable labs. Exercises with worked solutions for every chapter. No paywall, no application, no cohort, no waitlist.**

---

## Why this exists

The knowledge of how a frontier model gets built is not actually secret. It is spread across a few dozen technical reports, a few hundred papers, some very good blog posts, and a lot of tacit knowledge that never gets written down. Assembling it is the hard part, and that assembly work is what people currently pay thousands of dollars and compete for limited slots to have done for them.

That gap does not need to exist. This repository is the assembly, done in the open, given away.

If it helps you get the job, ship the run, or just understand what the DeepSeek-V3 report is actually saying — that is the entire point. Star it, fork it, translate it, teach from it. The license explicitly allows all of that, including commercially.

## Who this is for

- **ML/AI engineers** who know transformers and PyTorch and want to understand what happens when the same ideas scale to 10,000 GPUs, 14 trillion tokens, and a six-figure daily compute bill.
- **Graduate students** moving from coursework into frontier-lab internships, looking for the unstated context papers leave out.
- **Self-learners** with a laptop and a Colab tab who want a structured path instead of a reading list.
- **Researchers** who want to understand how systems-level choices in modern training runs shape the scientific questions they can ask.

## Who this is *not* for

- People looking for a tutorial on fine-tuning a 7B model in Colab. (Though the labs *do* run in Colab — they teach the mechanisms, not the recipe.)
- People looking for a survey of "top 100 LLM applications."
- People looking for a generic distributed-systems textbook with transformers stapled on at the end.

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
book/          26 chapters across 5 parts, plus glossary and 96 references
exercises/     26 problem sets — arithmetic drills, design problems,
               paper-reading prompts — each with a worked solution set
labs/          19 runnable labs you can paste straight into Colab
tools/         link checker, structure checker, notebook builder, lab runner
```

### What is complete

All 26 chapters (~115,000 words), all 26 exercise sets and all 26 worked solution
sets (~125,000 words), the glossary, 96 references, and **all 19 labs** — every
lab the chapters name now exists, runs, and is checked by CI on every commit.

Two caveats worth stating plainly, because the book asks the same of the papers
it cites:

- **[`lab20_triton_fused_kernel`](labs/lab20_triton_fused_kernel.py) needs a GPU
  for part of what it teaches.** Triton compiles to GPU code. The lab opens with
  a table of which sections are exact arithmetic, which are measured on CPU, and
  which require hardware; the kernel source is included in full and runs where a
  GPU exists. Nothing is faked.
- **Chapters 2 and 3 have no per-chapter reference block**, unlike the other 24.
  Their citations live in [the full reference list](book/appendix/b-references.md).

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

- Code blocks are real, or real-shaped, configurations from public papers and blog posts.
- Citations are inline as `[N]` with full entries in [the reference list](book/appendix/b-references.md).
- "We" refers to the field, not a specific lab, unless a section anchors on one.
- "Frontier lab" means a lab that trains its own foundation model from scratch — Anthropic, OpenAI, Google DeepMind, Meta, xAI, Microsoft, Alibaba (Qwen), DeepSeek, Moonshot, ByteDance, Zhipu, Mistral, and similar.
- Where a lab has not published something, we say so instead of guessing.

## Status

A living document. The frontier moves fast; expect chapters to be revised as new papers and technical reports drop. See [the changelog](CHANGELOG.md) for what moved recently.
