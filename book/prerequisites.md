# Prerequisites and self-assessment

> Reading time: ~12 minutes. By the end you should know honestly whether you can read this book today, and if not, exactly what to go learn first and where.

This book assumes a specific starting point, and being vague about it wastes your time. This page is the honest version: a checklist you can actually score yourself against, and a named resource for every gap.

None of the gaps are permanent. All of them have good free material.

## The three-question triage

Answer these before anything else.

**1. Can you explain what a key/value cache is and why it exists?**
If yes, your transformer background is probably fine. If you know what attention is but have never thought about the KV cache, you are close — Chapter 5 will fill it in. If "key/value" means nothing, start with the transformer material below.

**2. Have you written a training loop yourself — forward, loss, `backward()`, `step()` — not just called `Trainer.train()`?**
If yes, the PyTorch assumption holds. If you have only ever used high-level wrappers, most of the book still reads fine, but the labs will be rough.

**3. Do you know roughly what your GPU's memory is spent on during training?**
If you can say "parameters, gradients, optimizer states, activations" without looking it up, Chapter 6 will feel like a natural extension. If not, that is fine — Chapter 6 §6.1 derives it from scratch. It is the single most load-bearing piece of arithmetic in the book.

Two out of three yes: start reading. Zero or one: spend a few weeks on the gaps below, then come back. The book will still be here, and it will be much less frustrating.

## The full checklist

Score yourself: **✅ solid**, **🟡 shaky**, **❌ gap**. Anything marked ❌ in the "required" section is worth closing first.

### Required

| # | You can… | Where it bites if you cannot |
|---|---|---|
| 1 | Explain self-attention including Q, K, V, and why it is $O(N^2)$ | Ch 5, 6, 9, 20 |
| 2 | Explain what a transformer block contains and why there is a residual stream | Ch 5, 8 |
| 3 | Explain backpropagation well enough to know why activations must be stored | Ch 6 (recomputation), 21 |
| 4 | Write a PyTorch training loop from scratch | Every lab |
| 5 | Explain what an optimizer state is and why AdamW needs two of them per parameter | Ch 6, 8 |
| 6 | Read Python with type hints and follow tensor-shape comments | Every lab |
| 7 | Do back-of-envelope arithmetic with bytes, FLOPs, and bandwidth without panicking | Ch 6, 7, 8, 22 |
| 8 | Explain cross-entropy loss and perplexity, and convert between them | Ch 3, 4, 8, 9 |

### Strongly recommended

| # | You can… | Where it bites if you cannot |
|---|---|---|
| 9 | Explain what a GPU does differently from a CPU — SMs, warps, memory hierarchy | Ch 7, 20, 22 |
| 10 | Explain what "memory bandwidth bound" means and compute arithmetic intensity | Ch 9, 20, 22 |
| 11 | Describe a collective operation — all-reduce, all-gather, all-to-all | Ch 6, 7 |
| 12 | Read a `.yaml` training config and guess what each field does | Ch 6, 8, 10 |
| 13 | Explain the difference between supervised learning and reinforcement learning | All of Part III |
| 14 | Explain a policy gradient at the level of REINFORCE | Ch 13, 15 |

### Helpful, not required

| # | You can… | Where it shows up |
|---|---|---|
| 15 | Read CUDA or Triton | Ch 20 (the chapter teaches enough to follow along) |
| 16 | Explain KL divergence and why it appears in RLHF objectives | Ch 13, 14, 17 |
| 17 | Have opinions about tokenizers | Ch 4 (you will after) |
| 18 | Have run a multi-GPU job of any kind | Ch 6, 7 |

## Closing the gaps

Every resource below is free.

**Transformers, from the ground up.**
Andrej Karpathy's *Neural Networks: Zero to Hero* series, in particular "Let's build GPT: from scratch, in code, spelled out." Two hours of video that will do more for items 1–4 than any textbook. If you prefer reading: Jay Alammar's "The Illustrated Transformer," then the original *Attention Is All You Need* paper — which is genuinely readable and only eight pages.

**PyTorch mechanics.**
The official PyTorch 60-minute blitz for the basics, then write a character-level language model from scratch without copying. The exercise that matters is implementing `nn.Module`, the optimizer step, and the training loop yourself at least once.

**GPU mental model.**
The first three chapters of the *Programming Massively Parallel Processors* book (Kirk & Hwu) if you want depth. For a fast version: Horace He's "Making Deep Learning Go Brrrr From First Principles" is the single best short piece on compute-bound vs memory-bound vs overhead-bound, and Chapter 20 assumes something like its mental model.

**Distributed training basics.**
The PyTorch DDP tutorial, then the ZeRO paper ([\[17\]](appendix/b-references.md#17-zero)). The ZeRO paper is unusually well-written for a systems paper and it sets up Chapter 6 almost perfectly.

**Reinforcement learning, enough for Part III.**
You do not need a full RL course. You need REINFORCE, the idea of a baseline and why it reduces variance, and the shape of the PPO clipped objective. Spinning Up in Deep RL (OpenAI) covers all three, and its PPO page is the standard reference. Skip the continuous-control material; none of it transfers.

**The arithmetic.**
"Transformer Inference Arithmetic" (Kipply) and "Transformer Math 101" (EleutherAI) between them cover most of item 7. Both are short. Doing their worked examples by hand is worth more than reading them twice.

## What you explicitly do *not* need

Stated because these come up constantly and cost people months:

- **A PhD.** A significant fraction of frontier pre-training and infra work is done by people with engineering backgrounds and no research degree. Chapter 25 goes through which roles actually gate on a PhD (fewer than you would think) and which do not.
- **Experience with a large cluster.** Nobody has this before their first job at a place that has one. That is the whole point of Chapters 6 and 7 — to give you the model so the first day is not a total shock.
- **To have read every paper.** The book cites about 50. You need maybe a dozen of them in depth, and the chapters tell you which.
- **Distributed-systems theory.** No Paxos, no Raft, no CAP theorem. Frontier training is a synchronous, batch-scheduled, gang-allocated workload — its failure model is nothing like a web backend's, and the classical theory transfers less than you would expect. Chapter 7 covers what actually matters.
- **Kubernetes.** Genuinely not used for the training jobs at most frontier labs. Slurm is.
- **Fluency in JAX or TensorFlow.** Some labs use JAX; the concepts transfer directly and the book stays framework-agnostic where it can.

## Self-check: the entry problem

If you want one concrete test of whether you are ready, do this before Chapter 1. It is the same style as the arithmetic drills throughout the book.

> A 7B-parameter dense transformer is trained in BF16 with the AdamW optimizer, keeping FP32 master weights.
>
> 1. How many bytes per parameter are held for the weights, the gradients, and the optimizer states combined?
> 2. What is the total in gigabytes for the 7B model, before any activations?
> 3. Will that fit on one 80 GB H100? What is the first thing you would do if it did not?

Work it out on paper. The answer is in the [solutions to the prerequisites check](../exercises/solutions/prerequisites.md).

If you got it, or if you got it after one hint, you are ready for Chapter 1. If the question felt unanswerable, work through "Transformer Math 101" first — it will take an afternoon and it will change how the rest of the book reads.

---

Next: [Learning paths](learning-paths.md) to pick a route, or [Chapter 1](part-1-frontier/01-what-a-frontier-run-looks-like.md) to begin.
