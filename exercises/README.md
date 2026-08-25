# Exercises

One problem set per chapter. Every set has a [worked solution](solutions/README.md).

## How to use these

Write your answer down before you open the solution. The gap between "I could have gotten that" and "I got that" is where the learning is, and it is invisible unless you commit to an answer first.

Each set mixes three kinds of problem, tagged so you can pick:

**🔢 Drill** — arithmetic you should be able to do on a whiteboard. Memory budgets, FLOP counts, bandwidth, KV-cache sizes, token economics. These are the questions frontier interviews actually ask, and they are unforgiving in a useful way: the number either comes out or it does not.

**🏗️ Design** — open-ended problems with no single right answer. "You have 512 H100s and a 200B MoE — choose the parallelism and justify it." The solution walks the reasoning and names the trade-offs rather than asserting an answer. Your answer can differ from the solution and still be correct; what matters is whether you considered the same constraints.

**📄 Paper** — read a specific section of a real paper and answer a question about it. These build the skill the book cannot hand you: extracting engineering from a technical report. All the papers are free and linked from [the reference list](../book/appendix/b-references.md).

Some sets also have:

**🧪 Lab** — a pointer into [`labs/`](../labs/README.md) with a specific thing to change and a prediction to make before you run it.

## Difficulty

Problems are marked ★ (follows directly from the chapter), ★★ (requires combining two or more sections), or ★★★ (requires going beyond the chapter — a paper, a lab, or genuine design judgement).

If you are short on time, do the ★★ problems. The ★ problems check that you read; the ★★ problems check that you understood.

## The sets

### Part I — The Frontier
- [Chapter 1 — What a frontier run looks like](ch01.md)
- [Chapter 2 — The org chart](ch02.md)

### Part II — Pre-training
- [Chapter 3 — Data pipelines](ch03.md)
- [Chapter 4 — Tokenization](ch04.md)
- [Chapter 5 — Architecture](ch05.md)
- [Chapter 6 — Distributed training](ch06.md)
- [Chapter 7 — Cluster reality](ch07.md)
- [Chapter 8 — Optimization](ch08.md)
- [Chapter 9 — Mid-training](ch09.md)
- [Chapter 10 — DeepSeek-V3 case study](ch10.md)

### Part III — Post-training
- [Chapter 11 — SFT](ch11.md)
- [Chapter 12 — Reward modeling](ch12.md)
- [Chapter 13 — PPO and RLOO](ch13.md)
- [Chapter 14 — DPO](ch14.md)
- [Chapter 15 — GRPO](ch15.md)
- [Chapter 16 — Process rewards](ch16.md)
- [Chapter 17 — Constitutional AI](ch17.md)
- [Chapter 18 — Tool use](ch18.md)
- [Chapter 19 — DeepSeek-R1 case study](ch19.md)

### Part IV — Infra
- [Chapter 20 — Custom kernels](ch20.md)
- [Chapter 21 — Checkpointing](ch21.md)
- [Chapter 22 — Inference and serving](ch22.md)
- [Chapter 23 — Evaluation](ch23.md)

### Part V — The job
- [Chapter 24 — Translating JDs](ch24.md)
- [Chapter 25 — Career paths](ch25.md)
- [Chapter 26 — What's next](ch26.md)

### Before you start
- [Prerequisites self-check solution](solutions/prerequisites.md) — the entry problem from [the prerequisites page](../book/prerequisites.md).

## A note on the numbers

Drill problems use round numbers that are close to, but not always identical to, the real configurations in the chapters. This is deliberate: it stops you from pattern-matching a memorized figure and forces the calculation. Where a problem uses a real published configuration, it says so and cites it.
