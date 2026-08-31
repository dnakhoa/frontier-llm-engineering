# Chapter 2: The org chart — who actually does what

> Reading time: ~20 minutes. By the end of this chapter you should be able to read a frontier-lab JD and tell what the person actually does all day.

## 2.1 Why the org chart is the org chart

Frontier LLM labs are organized around the training pipeline. The roles exist because the pipeline has components, and the components require different skills. The split is not arbitrary: data engineers build data pipelines, kernel engineers write GPU kernels, post-training researchers design RL algorithms. But the *boundaries* between roles are blurry in practice, and that blur is part of the "weird" feeling of frontier JDs.

A useful mental model: a frontier lab is a factory. The factory has stations, each with a specific function. A single LLM moves down the line, getting worked on at each station. The stations, in order, are roughly:

1. **Data acquisition and curation** — raw crawl, dedup, filtering, mixing.
2. **Pre-training** — architecture, optimizer, distributed training.
3. **Mid-training / continued pre-training** — long context, domain annealing.
4. **Supervised fine-tuning (SFT)** — instruction following, format following.
5. **Reward modeling and preference learning** — RLHF, DPO, etc.
6. **Reinforcement learning** — PPO, GRPO, reasoning RL.
7. **Safety / alignment** — refusal, honesty, red-teaming.
8. **Evaluation** — capability, contamination, safety.
9. **Inference / serving** — quantization, speculative decoding, kernel work.
10. **Product / API surface** — token limits, function calling, system prompts.

Each of these stations has a team. The teams share infrastructure (the cluster, the data lake, the eval pipeline) but have distinct expertise and distinct hiring profiles.

This chapter walks through each role: what they do, what skills they need, what a real JD looks like, and what a typical day looks like.

## 2.2 The role taxonomy

There is no canonical taxonomy, but a useful breakdown of the 12–15 distinct roles at a frontier lab:

### Pre-training side

1. **Pre-training Data Engineer** — owns the data pipeline.
2. **Pre-training Researcher** — designs the architecture, the optimizer, the recipe.
3. **Large-Scale Training Engineer** — owns the distributed training system.
4. **Kernel Engineer** — writes custom GPU kernels.
5. **Cluster / Infrastructure Engineer** — owns the GPU cluster, the network, the storage.

### Post-training side

6. **Post-training Researcher** — designs the SFT, RLHF, GRPO algorithms.
7. **RL Infrastructure Engineer** — owns the RL training loop, the reward model serving, the rollout system.
8. **Reasoning / Agents Researcher** — works on agentic post-training, tool use, long-horizon reasoning.
9. **Alignment / Safety Researcher** — refusal, honesty, red-teaming, interpretability.

### Evaluation and analysis

10. **Evaluation Engineer** — owns the eval pipeline, the contamination detection, the leaderboards.
11. **Capability Researcher / Benchmark Lead** — designs new evaluations, studies capability emergence.

### Serving and product

12. **Inference Engineer** — quantization, kernel work for serving, scheduling.
13. **Applied Engineer / Solutions Engineer** — works with the API surface, function calling, latency.

### Cross-cutting

14. **Research Engineer** — sits between researcher and engineer, prototyping fast.
15. **Tech Lead / Staff Engineer** — owns a major component end-to-end, coordinates across teams.

The Chinese labs (Alibaba Qwen, DeepSeek, Moonshot, Zhipu, ByteDance Doubao) and the Western labs (Anthropic, OpenAI, Google DeepMind, Meta FAIR, xAI) have variations on this. Some roles are merged (e.g., xAI tends to have broader roles per person; Anthropic has more dedicated alignment people; DeepSeek has fewer people doing more). The above is the union.

## 2.3 Each role in detail

For each role, I will give: the actual job, the skills it requires, a real-shaped JD, and a typical week.

### 2.3.1 Pre-training Data Engineer

**What they do.** Build and maintain the pipeline that turns raw web crawls, code repositories, books, and papers into trillions of training tokens. The pipeline runs continuously on a Ray or Spark cluster, processing terabytes per hour. Dedup happens at multiple granularities. Quality is filtered by a learned classifier. Contamination against evaluation benchmarks is removed. The data mix is tuned based on validation-loss curves.

**Skills required.**
- Strong Python, fluent in at least one of: Ray, Spark, Beam, Dask.
- Familiarity with text processing at scale: tokenization, n-gram overlap, MinHash, suffix arrays.
- Some ML background (knows what a transformer is, can read a paper and reproduce it).
- Comfort with petabyte-scale data: knows when to use columnar formats (Parquet, Arrow), when to use raw text, when to use a database.

**A real-shaped JD (composite, anonymized).**

> **Senior Pre-training Data Engineer**
>
> You will own a major component of our pre-training data pipeline — the deduplication, quality filtering, and contamination removal stages that process ~10T tokens of training data per training run. You will work closely with pre-training researchers to tune the data mix, with infrastructure engineers to keep the pipeline running at scale, and with evaluation engineers to ensure the data is not contaminated. We expect experience with one of: Ray, Spark, or Dask at >100-node scale; comfort with petabyte-scale data; and a track record of building data systems that did not break in production.

**A typical week.** Monday: investigate a 3% drop in validation loss on code that appeared after a data mix change. Tuesday: write a MinHash dedup at higher precision to remove a class of near-duplicates we missed. Wednesday: 1:1 with the pre-training researcher to align on whether to upweight math. Thursday: code review on a colleague's PR for the new quality classifier. Friday: investigate why a contamination check missed a benchmark — it turns out the benchmark was reformatted to Markdown, the n-gram overlap didn't catch it.

### 2.3.2 Pre-training Researcher

**What they do.** Design the model architecture (or pick which existing architecture to scale), the optimizer, the training recipe (LR schedule, batch size ramp, data mix). Run ablation studies on smaller models to inform choices for the big run. Own the validation-loss curve.

**Skills required.**
- PhD-level ML background, comfortable with transformer internals.
- Knows how to design an ablation, how to read a loss curve, how to detect when a result is signal vs noise.
- Comfortable writing PyTorch / JAX at the level of "I will prototype a new attention variant this week."
- Reads papers, attends conferences, has opinions about scaling laws.

**Real-shaped JD.**

> **Pre-training Research Scientist**
>
> Design and execute the next generation of pre-training architectures and training recipes. You will own ablations on 1B–70B parameter models that inform the architecture and recipe for our 1T+ parameter runs. We expect a strong publication record at top ML venues (NeurIPS, ICML, ICLR), deep familiarity with transformer internals, and the ability to turn an architectural hypothesis into a clean ablation within a week.

**A typical week.** Monday: review the loss curve from the weekend's run, decide whether to keep the new attention variant or revert. Tuesday: design an ablation to test whether the auxiliary loss is helping or hurting. Wednesday: paper reading — three new MoE papers dropped on arXiv. Thursday: present the ablation results to the team. Friday: spec out the next architecture change for the 1T run.

### 2.3.3 Large-Scale Training Engineer / Distributed Systems Engineer

**What they do.** Own the distributed training system. Tune the 3D parallelism configuration. Debug NCCL hangs. Write the checkpointing and resumption logic. Profile the training loop to find bottlenecks. Make the run finish faster, or use less memory, or both.

**Skills required.**
- Strong systems background, knows how a GPU works at the SM and HBM level.
- Familiarity with at least one of: Megatron-LM, DeepSpeed, FSDP, JAX pjit.
- Comfort with NCCL, collective communication, network topology.
- Has debugged a multi-GPU training job before and lived to tell the tale.

**Real-shaped JD.**

> **Large-Scale Training Engineer**
>
> Push the boundary of efficient training at 10k+ GPU scale. You will own the distributed training stack — 3D parallelism, ZeRO, FSDP, expert parallelism — and the failure-recovery loop. You will profile the training step, identify bottlenecks, and ship optimizations that make the run faster. We expect experience with multi-node training at >1k GPU scale, comfort with NCCL, and a track record of shipping optimizations to a production training system.

**A typical week.** Monday: NCCL hang on the 4k-GPU run, dig into the collective log, find a slow all-reduce on a particular link. Tuesday: write a faster checkpoint save that uses async I/O, shaves 30s off each save (which happens every hour). Wednesday: 1:1 with the pre-training researcher about a new architecture that needs a different parallelism strategy. Thursday: code review on a PR that adds FP8 all-reduce support. Friday: profile the new attention kernel, find a 15% speedup, land it.

### 2.3.4 Kernel Engineer

**What they do.** Write custom GPU kernels for the parts of the training and inference loop that the standard libraries cannot do fast enough. The attention kernel. The MoE routing and all-to-all. The fused cross-entropy. The paged KV cache for inference. The speculative decoding verification.

**Skills required.**
- Deep CUDA knowledge. Knows about memory coalescing, shared memory banks, warp-level primitives, tensor cores, async copy.
- Fluent in Triton (the Python-like DSL for GPU kernels).
- Has read the PTX or SASS for a kernel they care about.
- Comfortable with low-level performance work: nsight-compute, nsight-systems, profiling.

**Real-shaped JD.**

> **GPU Kernel Engineer, Foundation Models**
>
> Write the GPU kernels that make our training and inference fast. You will own the attention kernel, the MoE all-to-all, and the fused cross-entropy. You will profile, optimize, and ship. We expect deep CUDA / Triton experience, a track record of writing kernels that beat the standard libraries, and comfort with low-level performance work.

**A typical week.** Monday: profile the new MLA attention kernel, find a 20% memory-bandwidth bottleneck. Tuesday: rewrite the kernel in Triton with a better memory access pattern. Wednesday: benchmark the new kernel against the old, ship it. Thursday: design a fused MoE routing + dispatch kernel to avoid an intermediate buffer. Friday: write a CUDA kernel for a new operation that doesn't exist in cuBLAS.

### 2.3.5 Cluster / Infrastructure Engineer

**What they do.** Own the GPU cluster: provisioning, scheduling, monitoring, networking, storage. The cluster is the asset. Without it, nothing else works.

**Skills required.**
- Strong systems / SRE background.
- Familiarity with one of: Slurm, Kubernetes, custom schedulers.
- Network knowledge: InfiniBand, RoCE, NCCL topology.
- Storage: parallel filesystems (Lustre, WekaFS, GPFS), object storage.

**Real-shaped JD.**

> **ML Infrastructure Engineer**
>
> Own the GPU cluster. Provision nodes, schedule jobs, monitor the network, debug hardware failures, keep the storage full. You will work closely with training engineers to make the run finish and with finance to manage the budget. We expect experience with HPC clusters at >1k node scale, comfort with network tuning, and a track record of keeping a complex system running.

**A typical week.** Monday: 3 nodes down, RMA them, swap parts. Tuesday: tune the NCCL topology-aware mapping. Wednesday: capacity planning for the next training run. Thursday: write a better monitoring dashboard. Friday: investigate a slow filesystem, find a misconfigured stripe size.

### 2.3.6 Post-training Researcher

**What they do.** Design the post-training pipeline: SFT data curation, reward model architecture, RL algorithm choice (PPO, DPO, GRPO, etc.), RL training recipe. Own the post-training loss curves, the eval scores, the model release.

**Skills required.**
- Strong ML background. PhD preferred, but industry experience works.
- Deep familiarity with RL (in the deep-learning sense): PPO, DPO, GRPO, reward modeling.
- Knows how to design an ablation in RL (which is much harder than in supervised learning because the reward signal is noisy).
- Comfortable reading and writing the relevant papers.

**Real-shaped JD.**

> **Post-training Research Scientist**
>
> Design and ship the post-training pipeline. You will own SFT data curation, reward modeling, and the RL algorithm (PPO, DPO, or GRPO) that takes a base model to a frontier chat model. We expect deep familiarity with RLHF and its variants, a strong publication record, and the ability to ship a model within a quarter.

**A typical week.** Monday: review the GRPO run that finished over the weekend, the math score is up 5 points. Tuesday: design a new reward model architecture. Wednesday: paper reading. Thursday: ablate the SFT data mix. Friday: ship a candidate model to internal evaluation.

### 2.3.7 RL Infrastructure Engineer

**What they do.** Own the RL training loop. The RL loop has different scaling properties than pre-training: you have a policy model, a reference model, a reward model, a value model, and rollouts being generated in parallel. The infra must keep all of these in sync. Rollout generation is a significant fraction of the wall-clock cost.

**Skills required.**
- Strong systems background.
- Familiarity with distributed training (the policy is fine-tuned, similar to pre-training infra).
- Familiarity with inference (the rollouts are generated by the current policy).
- Has built a distributed system before.

**Real-shaped JD.**

> **RL Infrastructure Engineer**
>
> Own the RL training loop. The system runs a policy, a reference model, a reward model, a value model, and a rollout generator across hundreds of GPUs. You will make this faster, more reliable, and more scalable. We expect experience with distributed training and inference, and a track record of building systems that handle the messy reality of RL.

**A typical week.** Monday: rollout generation is the bottleneck, profile it, find that the reward model is the slow part. Tuesday: shard the reward model across more GPUs. Wednesday: 1:1 with the post-training researcher about a new algorithm that needs a different rollout strategy. Thursday: write a faster KL penalty computation. Friday: debug a divergence in the RL run.

### 2.3.8 Reasoning / Agents Researcher

**What they do.** Work on agentic post-training: tool use, long-horizon reasoning, multi-turn RL. This role is newer than the others and has grown rapidly since 2024. The work is closer to classical AI (planning, search) than to standard RLHF.

**Skills required.**
- Strong ML background with focus on RL.
- Familiarity with classical AI: planning, search, tree search, Monte Carlo methods.
- Comfort with multi-turn settings and partial observability.

**Real-shaped JD.**

> **Research Scientist, Reasoning and Agents**
>
> Push the state of the art in reasoning and agentic capabilities. You will work on long-horizon reasoning (math, code, multi-step problems), tool use, and the RL algorithms that train for these. We expect a strong RL background, familiarity with classical AI methods, and a track record of building reasoning systems.

**A typical week.** Monday: design a new MCTS-based rollout strategy. Tuesday: read three new reasoning papers. Wednesday: ablate the verifier. Thursday: integrate a new tool into the agent. Friday: design the next reasoning benchmark.

### 2.3.9 Alignment / Safety Researcher

**What they do.** Work on the model's behavior: refusal of harmful requests, honesty (does the model know what it doesn't know?), sycophancy reduction, deception detection. Some alignment researchers do mechanistic interpretability (reverse-engineering what the model is doing internally). Some do evals and red-teaming.

**Skills required.**
- Strong ML background.
- Familiarity with the safety literature: Constitutional AI, RLHF, debate, interpretability.
- Often a philosophy / ethics background is useful but not required.

**Real-shaped JD.**

> **Research Scientist, Alignment**
>
> Push the state of the art in model alignment. You will work on refusal, honesty, sycophancy, and interpretability. We expect a strong ML background, familiarity with the alignment literature, and a track record of asking good questions about what models are doing.

**A typical week.** Monday: design a new honesty eval. Tuesday: investigate a case where the model is sycophantic. Wednesday: read the latest interpretability paper. Thursday: ablate a Constitutional AI approach. Friday: red-team the latest model checkpoint.

### 2.3.10 Evaluation Engineer

**What they do.** Own the eval pipeline. Run capability benchmarks (MMLU, GSM8K, HumanEval, etc.) on every model checkpoint. Run contamination detection. Build the leaderboards. Make sure the eval scores are real.

**Skills required.**
- Strong engineering background, comfortable with Python at scale.
- Familiarity with the major benchmarks.
- Comfort with statistics: confidence intervals, paired tests, multiple-comparison corrections.

**Real-shaped JD.**

> **Evaluation Engineer**
>
> Own the eval pipeline. Run capability benchmarks on every checkpoint. Detect contamination. Build the leaderboards. We expect strong engineering skills, familiarity with the major LLM benchmarks, and a track record of building reliable eval systems.

**A typical week.** Monday: a new MMLU-Pro checkpoint came in, run the eval, log the results. Tuesday: investigate a 3-point drop on HumanEval that might be noise. Wednesday: write a better contamination detector. Thursday: 1:1 with the pre-training researcher about a new benchmark they want to add. Friday: write a paired test to compare two checkpoints.

### 2.3.11 Inference Engineer

**What they do.** Make the model serve fast and cheap. Quantize weights to INT8/INT4/FP8. Write custom kernels for the attention, the MoE routing, the paged KV cache. Implement speculative decoding. Tune the serving scheduler. Hit latency SLOs.

**Skills required.**
- Strong kernel background (CUDA / Triton).
- Familiarity with at least one serving framework: vLLM, SGLang, TGI, TensorRT-LLM.
- Comfort with the inference-specific trade-offs: throughput vs latency, batch size, KV cache memory.

**Real-shaped JD.**

> **Inference Engineer, Foundation Models**
>
> Make our models serve at the lowest possible cost and latency. You will quantize, write kernels, tune the scheduler, and ship optimizations to production. We expect deep kernel experience, familiarity with vLLM / SGLang / TensorRT-LLM, and a track record of shipping inference optimizations.

**A typical week.** Monday: profile the inference path, find that the MoE routing is slow. Tuesday: write a faster routing kernel. Wednesday: integrate speculative decoding. Thursday: A/B test the new kernel in production. Friday: tune the KV cache memory budget.

### 2.3.12 Applied Engineer / Solutions Engineer

**What they do.** Build the API surface, the function-calling layer, the system prompt, the rate limiter, the developer experience. The role is closer to a standard backend engineer than to an ML engineer, but the constraints (latency, cost) are ML-shaped.

**Skills required.**
- Strong backend engineering.
- Familiarity with at least one model API (OpenAI, Anthropic, etc.).
- Comfort with prompt engineering and function calling.

**Real-shaped JD.**

> **Applied Engineer, Foundation Models**
>
> Build the API surface and the developer experience. Function calling, system prompts, rate limits, latency. We expect strong backend skills and comfort with LLM APIs.

**A typical week.** Mostly a normal backend week: build features, fix bugs, write docs. Maybe 10% of the time spent on prompt engineering or function-calling schema design.

## 2.4 The blurry boundaries

The roles above are idealized. In practice, the boundaries are blurry, and the blurriness is part of the "weird" feeling of frontier JDs.

- A **research engineer** might be half researcher (designs ablations) and half engineer (implements the infrastructure for the ablations).
- A **post-training researcher** at a small lab might also do the SFT data curation, the eval pipeline, and the RL infrastructure, because the lab doesn't have separate people for each.
- A **kernel engineer** might be doing inference kernels and training kernels in the same week.
- A **large-scale training engineer** at Anthropic might also be doing kernel work, because the boundary between "systems" and "kernels" is thin.
- A **pre-training researcher** at DeepSeek might own the entire data pipeline, because the team is small.

The right way to read a JD is not "this person does X." It is "this person has Y as their primary skill and is expected to be effective at X, with help on the adjacent skills."

## 2.5 How the teams interact

A simplified flow of how a model moves from data to deployment:

```
Pre-training Data Engineer
        │ (data pipeline → 14T tokens)
        ▼
Pre-training Researcher + Large-Scale Training Engineer
        │ (pre-training run → 60 days)
        ▼
Pre-trained base model
        │
        ├──► Kernel Engineer / Inference Engineer (in parallel: inference optimization)
        │
        ▼
Post-training Researcher
        │ (SFT data + RLHF/DPO/GRPO)
        ▼
Aligned chat model
        │
        ├──► Alignment / Safety Researcher (in parallel: red-teaming, alignment evals)
        │
        ├──► Evaluation Engineer (in parallel: capability + safety evals)
        │
        ▼
Released model
        │
        ▼
Applied Engineer (API surface, function calling)
        │
        ▼
Users
```

The arrows are not strict. In practice, the teams are deeply interleaved. The post-training researcher might ask the pre-training researcher to re-train with a different data mix. The inference engineer might discover a numerical issue in the training that requires a precision change. The alignment researcher might find a problem in the post-training data that requires a new SFT pass.

A useful way to think about it: the org chart is a dependency graph, not a flow chart.

## 2.6 The JDs, decoded

A few common JD phrases, decoded:

| Phrase | What it actually means |
|---|---|
| "Train large-scale language models" | You will own a stage of a frontier pre-training or post-training pipeline. |
| "Distributed training at scale" | You will debug 3D-parallel configs and NCCL hangs. |
| "Experience with FSDP / DeepSpeed / Megatron-LM" | You have used one of these in anger, not just read the README. |
| "Strong publication record" | You can design and execute an ablation and write it up. |
| "Familiarity with PyTorch internals" | You have read `torch.distributed` source, not just the docs. |
| "CUDA / Triton experience" | You have written a kernel that was faster than the standard library, ideally by a lot. |
| "Experience with RLHF / DPO / GRPO" | You have run one of these end-to-end, not just read the paper. |
| "Familiarity with NCCL" | You have debugged an NCCL hang and know what to look at. |
| "Strong systems background" | You have shipped a system that runs continuously, not a notebook. |
| "Petabyte-scale data" | You have worked on data where the unit is the terabyte and the storage is parallel. |

## 2.7 The team size and shape

Different labs have different team sizes. As of 2024–2025:

- **OpenAI, Anthropic, Google DeepMind**: ~100–500 people working on foundation models total, with significant overlap between teams.
- **Meta FAIR**: ~100–200 on Llama, plus the broader FAIR org.
- **xAI**: smaller, ~50–100, broader roles per person.
- **DeepSeek**: ~200 people total, very strong technical density.
- **Alibaba Qwen, ByteDance Doubao, Moonshot, Zhipu, Baichuan**: ~50–200 on the foundation model team each, with different ratios of researchers to engineers.
- **Mistral**: ~50 people, very engineering-heavy.

The trend across the industry is toward more specialization: a frontier lab is too big a project for generalists alone. The Chinese labs and DeepSeek are notable for being more concentrated on the engineering side (their researchers also write code), while some Western labs have a stronger separation between researchers and engineers.

## 2.8 What you should take from this chapter

1. **The roles are specialized but the boundaries are blurry.** "Pre-training Engineer" is at least five different jobs.
2. **The JD phrases are not mysterious.** "Distributed training at scale" means you will debug 3D-parallel configs and NCCL hangs.
3. **The org chart is a dependency graph.** Teams are deeply interleaved; a model moves down the line, getting worked on by all of them.
4. **The team shape varies by lab.** DeepSeek is concentrated and engineering-heavy; OpenAI is larger with more separation between roles.

The next chapter starts the deep technical work, with the pre-training data pipeline.

---

**Exercises:** [Chapter 2 problem set](../../exercises/ch02.md) — includes the role-mapping exercise and the on-call escalation problem.
