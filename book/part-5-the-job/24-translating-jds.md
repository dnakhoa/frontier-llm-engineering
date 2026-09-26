# Chapter 24: Translating frontier JDs into a skill stack

> Reading time: ~25 minutes. By the end of this chapter you should be able to read a frontier-lab job description and know which chapters of this book it is describing, what the team is actually struggling with, what they will test in an interview, and how to build evidence you can do the work without access to a cluster.

*Current as of early 2025.*

## 24.1 Why job descriptions are written the way they are

A frontier-lab JD is usually written by an engineer on the team, lightly edited by a recruiter, and it is optimizing for something other than clarity. Two forces distort it.

**It cannot reveal what the team is doing.** "Improve the reward-model pipeline for the next reasoning model" becomes "work on advanced alignment techniques." The specificity is stripped deliberately.

**It is a wish list, not a requirement list.** Written by imagining the ideal candidate, then adding everything anyone on the team can do. Nobody has all of it. The unwritten convention is that meeting roughly 60% of a frontier JD is a reasonable application, and this is more true at frontier labs than elsewhere because the roles are new enough that almost nobody has the full stack.

What a JD *does* reliably tell you: which subsystem the role touches, and — if you read carefully — what is currently painful. A JD that dwells on fault tolerance is written by a team having outages.

## 24.2 The decoder ring

Phrases that appear constantly, and what they mean.

| Phrase | What it means | Chapter |
|---|---|---|
| "Large-scale distributed training" | TP/PP/DP/EP, FSDP or Megatron, thousands of GPUs | 6, 7 |
| "Training efficiency" / "MFU" | Kernel work, communication overlap, profiling | 20, 7 |
| "Model architecture research" | MoE, attention variants, ablations at small scale | 5 |
| "Data pipelines at scale" | Petabyte crawls, dedup, filtering, mixture design | 3 |
| "Tokenization" | BPE training, vocabulary size, multilingual fertility | 4 |
| "Training stability" | Loss spikes, precision, LR schedules, muP | 8 |
| "Long-context" | RoPE extension, ring attention, varlen kernels | 9, 20 |
| "Post-training" | SFT, preference optimization, RL — the whole of Part III | 11–19 |
| "RLHF" | PPO/DPO/GRPO plus the reward model | 12–15 |
| "Alignment" | Could be Constitutional AI, could be safety evaluation. Ask. | 17, 23 |
| "Verifiable rewards" / "RLVR" | Reasoning RL — this team read the R1 paper | 15, 19 |
| "Agentic" / "tool use" | Environments, trajectories, long-horizon RL | 18 |
| "Kernel development" | Triton and CUDA, fused ops | 20 |
| "Inference optimization" | Batching, KV cache, quantization, speculative decoding | 22 |
| "Evaluation" | Benchmarks, contamination, judges, held-out sets | 23 |
| "Fault tolerance" / "job resiliency" | Checkpointing, restart automation | 21 |
| "Synthetic data" | Rejection sampling and filtering pipelines | 11 §11.10 |
| "Data-centric" | The mixture is the product; expect ablation work | 3, 11 |

And some phrases that carry information beyond their literal content:

- **"Comfortable with ambiguity"** — the team does not know what it is doing yet, which is often a good sign at the research end and a bad one at the infrastructure end.
- **"Full-stack"** on an ML posting — small team, you will own more than the title suggests.
- **"Scale our training infrastructure to the next order of magnitude"** — they are hitting the limits of the current system and want someone who has seen the next size up.
- **"Work closely with researchers"** — you are the engineer who makes researchers' ideas run. Rewarding, and the JD is telling you where the authority sits.
- **No mention of evaluation anywhere** — a warning sign, per Chapter 23.

## 24.3 Four archetypes, decoded

Composite JDs assembled from patterns across published frontier postings. Not quotes from any specific company.

---

### Archetype A — Pre-training Engineer

> *"You will train large language models at scale, working on data pipelines, model architecture, and distributed training infrastructure. Experience with Megatron, DeepSpeed, or FSDP required. Familiarity with MoE architectures and mixed-precision training preferred. You will own end-to-end training runs on thousands of accelerators."*

**Chapters:** 3, 4, 5, 6, 7, 8, 9, 21.

**What the job is:** you own runs. You decide the parallelism configuration, you watch the loss, you diagnose the spike at 3am, you decide whether to roll back.

**What "own end-to-end training runs" is really telling you:** on-call. A run that costs $200,000 a day does not pause for the weekend.

**Interview will test:** the memory arithmetic (§6.1) on a whiteboard, parallelism selection for a given model and cluster (§6.13), and a debugging scenario — "throughput dropped 30% overnight, what do you check?"

**The differentiator:** everyone can describe TP and PP. Far fewer can *size a run* — take a model and a cluster and produce a configuration with the arithmetic shown. That is Exercise 6.7 and it is what the interview is actually for.

---

### Archetype B — Post-training Researcher

> *"You will develop and improve techniques for aligning language models with human intent, including supervised fine-tuning, reward modeling, and reinforcement learning from human feedback. You will design experiments, analyze results, and work with data annotation teams. Publications in top venues are a plus."*

**Chapters:** 11, 12, 13, 14, 15, 16, 17.

**What the job is:** experimental design and data judgement, far more than algorithm implementation. "Work with data annotation teams" is doing a lot of work in that paragraph — a large fraction of the job is deciding what data to collect and whether it worked.

**"Publications are a plus"** — a plus, not a requirement. Note the phrasing; it is usually accurate.

**Interview will test:** derive the DPO loss (§14.2). Explain PPO versus GRPO and when each is right. Design an experiment to test whether a mixture change helped. Diagnose reward hacking from a description of the curves.

**The differentiator:** knowing that the data question dominates the algorithm question, and being able to say why with a specific example. A candidate who reaches for "we'd try GRPO" before asking about the reward signal has revealed something.

---

### Archetype C — ML Systems / Performance Engineer

> *"You will optimize the performance of large-scale training and inference workloads. Responsibilities include profiling, writing custom CUDA/Triton kernels, optimizing collective communication, and improving hardware utilization. Deep understanding of GPU architecture required."*

**Chapters:** 6, 7, 20, 22.

**What the job is:** find the bottleneck, remove it, prove it with a measurement.

**Interview will test:** roofline analysis — given an operation, is it compute- or memory-bound, and what is the arithmetic intensity (§20.3)? Why is FlashAttention faster? Write or read a Triton kernel. Explain what a profiler trace is showing you.

**The differentiator:** benchmarking discipline (§20.9). A candidate who reports a speedup without warmup, synchronization, and a shape sweep has told you they have not shipped a kernel. Conversely, a candidate who volunteers the cases where their kernel *lost* is immediately credible.

---

### Archetype D — Inference / Serving Engineer

> *"You will build and optimize the systems that serve our models to millions of users. Experience with vLLM, TensorRT-LLM, or similar inference frameworks. Understanding of batching strategies, KV cache management, and quantization. You will own latency and throughput targets."*

**Chapters:** 5 (attention variants), 20, 22.

**What the job is:** the latency-throughput trade-off, in production, under a real SLO.

**Interview will test:** the KV cache calculation (§22.3). Why is decode memory-bound? What does continuous batching fix, and what does paged attention fix — they are different problems. When does speculative decoding stop helping?

**The differentiator:** thinking in goodput rather than throughput (§22.9), and knowing that "own latency and throughput targets" means owning a conflict rather than two goals.

---

## 24.4 The stacks

What to actually learn, per target role, in order.

**Pre-training:** memory and FLOP arithmetic → parallelism strategies → mixed precision and stability → data pipelines → MoE → cluster failure modes. Chapters 6 and 8 are the load-bearing ones; Chapter 10 shows them assembled.

**Post-training:** SFT mechanics including masking and packing → Bradley-Terry and reward-model failure modes → policy gradients and baselines → DPO derivation → GRPO and verifiable rewards → evaluation methodology. Chapter 12 is more load-bearing than people expect, and Chapter 23 is what separates a researcher from someone who runs experiments.

**Systems/kernels:** roofline model → profiling → Triton → collective communication → CUDA. Chapter 20's lab is the single highest-signal thing you can do here.

**Inference:** prefill/decode asymmetry → KV cache economics → batching → quantization → speculative decoding. Chapter 22, then read vLLM's source.

**Across all four:** the arithmetic. Every one of these interviews contains a back-of-envelope calculation, and being fluent with bytes, FLOPs, and bandwidth is the most transferable skill in this book.

## 24.5 What interviews actually test

Aggregating across the archetypes, four things recur.

**1. Arithmetic under mild pressure.** "How much memory does a 70B model need to train?" is the canonical opener. It is not testing whether you memorized 16 bytes per parameter; it is testing whether you can decompose a system into components and count. Practise until it is boring.

**2. A design question with an incomplete specification.** "Train a 30B MoE on 512 H100s." The evaluated behaviour is whether you ask what you need to know — context length, token budget, interconnect topology — before answering. Candidates who answer immediately do worse than candidates who ask three questions first.

**3. A debugging scenario.** "Loss spiked at step 40,000." "Throughput dropped 30%." "The model works in eval and worse in production." They are testing whether you have a *procedure*: form hypotheses, order them by cheapness-to-test, and say what you would look at first. Chapter 11 §11.12, Chapter 13 §13.7, and Chapter 21 §21.11 are essentially interview answer keys.

**4. Depth on one thing you claim to know.** Whatever you put on your résumé, they will go three questions deep. This is why "familiar with" is dangerous phrasing and why it is better to have one genuinely deep area than five shallow ones.

What is tested much less than candidates expect: LeetCode-style algorithms, ML theory, and knowledge of specific papers beyond the handful that matter.

## 24.6 Building evidence without a cluster

The central problem for anyone outside a lab: you cannot demonstrate 1,000-GPU experience without 1,000 GPUs. The good news is that nobody expects you to, and there are things that do transfer.

**Reproduce something small and write it up honestly.** Train a 100M model. Implement GRPO on a verifiable toy task. Write a fused kernel. The value is not the artifact — it is the write-up, and specifically the part where you say what did not work and why. Every lab engineer recognizes an honest debugging narrative and nobody can fake one.

**Contribute to the open infrastructure.** vLLM, SGLang, `trl`, `torchtitan`, Megatron, FlashAttention, `lm-evaluation-harness`. A merged PR in one of these is direct evidence of exactly the skills in the JD, and it is reviewable by the person reading your application. This is the highest-leverage thing on the list.

**Do the arithmetic publicly.** A blog post working out the memory budget, the parallelism configuration, and the cost of a hypothetical training run demonstrates the skill from §24.5 point 1 better than any credential.

**Read a technical report properly and write the missing half.** Take the R1 paper, reconstruct the pipeline, and write down what it does not tell you and what you would need to reproduce it (§19.8). That exercise is a direct simulation of the job.

**Use the free tiers.** A Colab T4 is enough for every lab in this book. Scale is not required to demonstrate understanding of scale — the arithmetic is.

**What does not work:** listing frameworks on a résumé, generic "LLM projects" that are API wrappers, and certificates. None of them survive question three of §24.5 point 4.

## 24.7 What matters less than you think

- **A PhD.** Required for some research roles, not for most engineering ones. Chapter 25 §25.2 goes through which is which, and the answer surprises people.
- **Having trained a large model.** Almost nobody has before their first job at a place that has one. That is the entire premise of Chapters 6 and 7.
- **Publications.** "A plus" in the archetype B posting, and that phrasing is honest.
- **Knowing every paper.** This book cites around 96. You need maybe a dozen deeply.
- **Framework brand loyalty.** Megatron versus FSDP versus JAX is a detail; the concepts transfer, and JDs that name one are usually describing their current stack rather than a requirement.

## 24.8 What you should take from this chapter

1. **A JD is a wish list written by an engineer and stripped of specifics.** Roughly 60% is a reasonable application, and the parts it dwells on tell you what the team is currently struggling with.
2. **The decoder ring in §24.2 maps phrases to chapters.** Most JDs are describing three or four chapters of this book.
3. **Every archetype's interview contains arithmetic.** Memory budgets, KV cache size, arithmetic intensity, GPU-hours. Practise until it is dull.
4. **In design questions, asking what you need to know beats answering fast.** The incompleteness is deliberate.
5. **Debugging scenarios test procedure, not recall.** Have an ordered set of hypotheses and say what you would check first and why.
6. **Depth beats breadth.** They will go three questions deep on anything you claim. One deep area beats five shallow ones.
7. **The highest-leverage evidence you can build without a cluster is a merged PR** to vLLM, `trl`, `torchtitan`, or a similar project — direct, reviewable proof of exactly the JD's skills.
8. **Write up your failures.** An honest debugging narrative is the thing that cannot be faked and the thing every lab engineer recognizes.
9. **A Colab T4 is enough** to demonstrate understanding of every mechanism in this book. Scale is demonstrated through arithmetic, not access.
10. **Nobody has trained a frontier model before their first frontier job.** The gap you are being assessed on is the model in your head, not the runs on your résumé.

The next chapter is what happens after you get in: the role map, whether the PhD question actually matters, how people move between these roles, and what compounds over a career in a field that reinvents itself every eighteen months.

---

**Exercises:** [Chapter 24 problem set](../../exercises/ch24.md) — includes decoding three real JDs against the chapter map, a self-assessment against a target role, and designing a portfolio project for a stated gap.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — the reading-a-report exercise in §24.6.
- [\[10\] DeepSeek-R1](../appendix/b-references.md#10-deepseek-r1) — likewise; see [Chapter 19 §19.8](../part-3-post-training/19-case-study-r1.md).
- [\[23\] vLLM](../appendix/b-references.md#23-vllm) — one of the open projects worth contributing to.
- [\[53\] Tülu 3](../appendix/b-references.md#53-tulu-3) — the open post-training recipe to reproduce.
- [See full reference list](../appendix/b-references.md)
