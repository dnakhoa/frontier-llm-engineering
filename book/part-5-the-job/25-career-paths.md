# Chapter 25: Career paths at frontier labs

> Reading time: ~25 minutes. By the end of this chapter you should have a realistic map of the roles that exist, an honest answer to the PhD question, a sense of which entry points are actually open, and a view on what compounds in a field that reinvents its techniques every eighteen months.

*Current as of early 2025.*

## 25.1 The role map

Chapter 2 introduced the org chart. Here it is again as a career map, with the honest annotations.

| Role | Owns | Gated on a PhD? | Supply vs demand |
|---|---|---|---|
| **Pre-training Researcher** | Architecture, recipe, scaling decisions | Usually yes | Very few roles, very many applicants |
| **Pre-training / Large-Scale Training Engineer** | Running the runs, parallelism, throughput | No | Scarce |
| **Data Engineer (pre-training)** | Crawls, dedup, filtering, mixtures | No | Very scarce |
| **Post-training Researcher** | SFT mixtures, RL techniques, ablations | Sometimes | Many applicants |
| **Post-training / RL Engineer** | The RL loop, rollout infra, stability | No | Scarce |
| **Data Engineer (post-training)** | Synthetic data, preference pipelines, verifiers | No | Very scarce |
| **ML Systems / Performance Engineer** | MFU, profiling, communication | No | Very scarce |
| **Kernel Engineer** | Triton, CUDA, fused ops | No | Extremely scarce |
| **Inference / Serving Engineer** | Latency, throughput, serving stack | No | Scarce |
| **Evaluation Engineer** | Benchmarks, contamination, held-out sets | No | Scarce, and undervalued |
| **Environment Engineer (agentic)** | Sandboxes, task supply, reproducibility | No | Very scarce, very new |
| **Alignment / Safety Researcher** | Constitutions, safety training, red teaming | Often | Many applicants |
| **Research Manager / TL** | People, priorities, run allocation | No | — |

Two patterns in the right-hand columns worth taking seriously.

**The roles people want are the ones with the most competition.** "Pre-training Researcher" and "Alignment Researcher" are the titles that attract applications, and they have the fewest seats.

**The roles that are hard to fill are the engineering ones.** Data, systems, kernels, inference, evaluation, environments. These have open headcount that stays open, and they are the ones this book spends most of its pages on — Chapters 3, 6, 7, 20, 21, 22, 23 are all in that column.

If you are optimizing for getting in, this table is the most actionable thing in the chapter.

## 25.2 The PhD question

Asked constantly, and the honest answer is more differentiated than either "you need one" or "credentials don't matter."

**Where a PhD genuinely helps:**

- **Pre-training research** — architecture and recipe decisions on runs that cost millions. Labs want people who have demonstrated they can design an experiment, and a PhD is the standard evidence.
- **Alignment research** — the questions are open-ended and the field is close to academia.
- **Anywhere the job is "propose what to try next"** rather than "make this work."

The mechanism is not the credential. It is that these roles require designing experiments whose results are ambiguous, on budgets where being wrong is expensive, and a PhD is four years of supervised practice at exactly that. If you have that skill from elsewhere, it counts — but you have to be able to show it.

**Where it does not matter:**

- Training infrastructure, systems, kernels, inference, data engineering, evaluation, environments.

These are engineering roles with deep domain requirements. A strong systems engineer who learns the domain is more valuable than a PhD who has not shipped production infrastructure, and labs know this. The scarcity column in §25.1 is the reason.

**The honest picture across a frontier lab:** a large fraction of the people making a frontier run happen do not have research degrees. The published papers over-represent the research roles, which distorts the perception.

**If you are considering a PhD for this purpose specifically:** it is four to six years in a field whose techniques turn over every eighteen months. The techniques you specialize in will likely be superseded. What will not be superseded is the experimental design skill and the collaborators — and if those are what you want, it is a reasonable choice. If you want to work on frontier models and you are already a strong engineer, the shorter path is Chapter 24 §24.6.

## 25.3 Entry points

Ranked by how often they actually work.

**1. Lateral from adjacent engineering.** Distributed systems, HPC, GPU programming, database internals, high-performance networking. This is the most common successful path into the scarce roles and it is underused because people assume they need an ML background. They mostly need the ML background this book contains — a few months of reading — on top of systems skills that take years.

**2. Through the open-source infrastructure.** vLLM, SGLang, `torchtitan`, `trl`, Megatron, FlashAttention. Substantial contributions get you known by the people hiring, and the work is directly comparable to the job. Per §24.6, the highest-leverage thing on the list.

**3. Internships and residencies.** Real but narrow. Most are aimed at PhD students, and the residency programs that accept non-traditional backgrounds are heavily oversubscribed.

**4. From a smaller lab or an applied team.** Work at a company doing serious fine-tuning or serving, get real experience with real scale, move. Slower and reliable.

**5. Publishing something useful.** A reproduction, a benchmark, a tool people adopt. Works, and it is high-variance.

**What does not work:** cold applications to the research roles in §25.1's crowded rows, with a résumé of API-wrapper projects. The competition is people with directly relevant published work.

## 25.4 Moving between roles

More common than the titles suggest, and worth planning for.

**Easy transitions:**

- Systems → kernels. Same mental model, narrower focus.
- Kernels → inference. Decode kernels are the bridge.
- Pre-training engineering → post-training engineering. The RL loop is a training loop plus a generation cluster.
- Inference → post-training infrastructure. §13.9 — post-training *is* a serving problem, and people who know both are scarce enough to name their role.
- Data engineering → post-training research. The mixture is the research.

**Harder transitions:**

- Engineering → research. The bar is demonstrated experimental design, not permission. The usual route is doing research work inside an engineering role until the title catches up.
- Research → management. As everywhere.
- Any role → pre-training research. The narrowest door in the building.

**The transition worth engineering deliberately:** get into a lab in a scarce role, then move. Internal moves are dramatically easier than external ones, because your colleagues have direct evidence of your judgement. This is the single most reliable strategy for reaching a crowded role, and it is why §25.1's scarcity column matters even if the scarce roles are not what you ultimately want.

## 25.5 What changes with seniority

The technical work stays; the surface changes.

**Junior.** You are given a task. Success is doing it well and learning the system. Most valuable behaviour: asking questions early rather than being stuck silently — on a run costing $200,000 a day, a day of silent confusion is expensive.

**Mid.** You are given a problem. Success is choosing an approach and executing it. You start owning a subsystem.

**Senior.** You are given an area. Success is knowing what *should* be worked on and why the other things should not. The scarce skill stops being implementation and becomes judgement — specifically, the ability to say "this will not work, here is what will" and be right often enough to be trusted.

**Staff and beyond.** You are responsible for outcomes that require other people. Technical depth is still the basis of your credibility, and most of your leverage comes from decisions and from unblocking others. On a frontier run this often looks like owning a cross-cutting concern — stability, throughput, evaluation integrity — rather than a component.

The thing that surprises people moving up: **the interesting problems get less technical and more organizational.** Deciding what mixture to train on is a technical question that is settled by a social process, and at senior levels you spend more time on the process than the question.

## 25.6 Where to work

**Large frontier labs** (Anthropic, OpenAI, Google DeepMind, Meta, xAI, Microsoft; Alibaba/Qwen, DeepSeek, Moonshot, ByteDance, Zhipu). Compute at a scale nothing else matches, colleagues who have done it, and a narrow slice of the problem each. You will not see the whole pipeline.

**Smaller labs and well-funded startups** (Mistral, Cohere, AI2, and the frequently-changing set of new entrants). Less compute, much more surface area per person, and a real chance to own something end to end. AI2 is worth calling out separately: it publishes fully open recipes (Tülu 3 [\[53\]](../appendix/b-references.md#53-tulu-3)), which means the work is visible and citable in a way most lab work is not.

**Applied teams at large companies.** Fine-tuning and serving rather than pre-training. Real scale on the inference side, and often the best place to actually learn Chapter 22.

**Open source and independent research.** No compute, maximum visibility. The reproductions of R1 (§19.9) came out of this world within weeks, and several of the people who did them were hired shortly after.

**Cloud providers and hardware vendors.** AWS, Google Cloud, NVIDIA, AMD. Systems and kernel work at enormous scale, often on the infrastructure the labs run on. Underrated entry point to the scarce roles, and they hire more people than the labs do.

## 25.7 The geography that is actually one field

A structural note that matters for anyone planning a career here.

The most detailed public technical reports of the last two years — DeepSeek-V3 and R1 [\[1\]](../appendix/b-references.md#1-deepseek-v3)[\[10\]](../appendix/b-references.md#10-deepseek-r1), Qwen3 [\[6\]](../appendix/b-references.md#6-qwen3), Kimi k1.5 [\[69\]](../appendix/b-references.md#69-kimi-k15) — came from Chinese labs. This book is anchored on them for exactly that reason: they published the most.

Two practical consequences.

**Read them.** A meaningful fraction of the field's published technical detail is coming from these teams, and any engineer who filters it out is working with less information than their peers.

**The open-weight ecosystem is substantially built on them.** Qwen and DeepSeek derivatives dominate the open fine-tuning world, which means that if you work in open models, you are already working with their artifacts.

The techniques are the same techniques. Compute access, publication norms, and hiring practices differ; the arithmetic in Chapter 6 does not.

## 25.8 What compounds

The field reinvents its techniques every eighteen months. Between the first edition of this book's chapters and now, GRPO went from a paper to standard practice, DPO went from a revelation to one option among six, and verifiable rewards went from a niche idea to the organizing principle of a model generation.

Given that turnover, what is worth investing in?

**The arithmetic compounds.** Bytes, FLOPs, bandwidth, memory hierarchies. The specific numbers change with each GPU generation; the skill of decomposing a system and counting does not. It has been the same skill since the 1990s and it will outlast every technique in this book.

**The failure modes compound.** Reward hacking, overoptimization, silent data bugs, distribution shift between training and serving. These recur in new clothes. Someone who recognized length exploitation in RLHF recognized it immediately in reasoning RL, because it is the same shape.

**Debugging methodology compounds.** Forming hypotheses, ordering them by cost to test, and looking at the data before the hyperparameters. This is the most transferable thing in the book and it is what §24.5's interview questions are actually probing.

**Judgement about what to try compounds**, slowly, and mostly through being wrong in public and noticing.

**Specific frameworks do not compound.** Megatron expertise is worth something and it is not worth what its holders think. Neither will vLLM expertise be, in five years.

**Specific algorithms compound less than you would hope.** PPO expertise transferred to GRPO — but through the *shared structure* (§13.3's "everything is a choice of baseline"), not the implementation details. The people who learned PPO as a recipe struggled; the people who learned it as a policy gradient with a particular baseline did not.

That distinction is the single most useful career advice in this chapter: **learn the structure, not the recipe.** This book is organized to make that possible, and it is why Chapter 13 spends a section establishing that PPO, RLOO, and GRPO are one algorithm with three answers to one question.

## 25.9 What you should take from this chapter

1. **The roles people want are crowded; the roles labs cannot fill are engineering ones.** Data, systems, kernels, inference, evaluation, environments have open headcount that stays open.
2. **A PhD is genuinely gating for pre-training research and alignment research**, and genuinely not for infrastructure, systems, data, inference, and evaluation. A large fraction of the people making a frontier run happen do not have one.
3. **The mechanism behind the PhD requirement is experimental design under expensive ambiguity.** If you have that from elsewhere, it counts — but you must be able to demonstrate it.
4. **Lateral entry from distributed systems, HPC, or GPU programming is the most common successful path**, and it is underused because people assume they need an ML background first.
5. **Contributing to open infrastructure is the highest-leverage evidence** you can build from outside. It is directly comparable to the job and reviewable by the person hiring.
6. **Get in through a scarce role, then move internally.** Internal transitions are dramatically easier than external ones because colleagues have direct evidence of your judgement.
7. **Seniority moves you from tasks to problems to areas**, and the interesting problems become progressively more organizational.
8. **Cloud providers and hardware vendors are an underrated entry point** to systems and kernel work, and they hire far more people than the labs.
9. **Read the Chinese labs' technical reports.** A meaningful fraction of the field's published detail is there, and filtering it out means working with less information than your peers.
10. **Learn the structure, not the recipe.** Arithmetic, failure modes, and debugging methodology compound across technique generations. Framework expertise and algorithm recipes do not.

The last chapter is about where the field is going — multimodal, agent RL, synthetic data, and the compute wall — with an explicit note on how much to trust any of it.

---

**Exercises:** [Chapter 25 problem set](../../exercises/ch25.md) — includes a self-assessment against the role map, a transition plan from your current background, and an honest evaluation of the PhD question for your situation.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024.
- [\[6\] Qwen3 Technical Report](../appendix/b-references.md#6-qwen3) — Qwen Team, 2025.
- [\[10\] DeepSeek-R1](../appendix/b-references.md#10-deepseek-r1) — DeepSeek-AI, January 2025.
- [\[53\] Tülu 3](../appendix/b-references.md#53-tulu-3) — Lambert et al., November 2024. The fully open recipe from AI2.
- [\[69\] Kimi k1.5](../appendix/b-references.md#69-kimi-k15) — Kimi Team, January 2025.
- [See full reference list](../appendix/b-references.md)
