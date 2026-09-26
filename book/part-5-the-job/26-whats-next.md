# Chapter 26: What's next — multi-modal, agent RL, synthetic data 2.0

> Reading time: ~25 minutes. This is the chapter most likely to be wrong. By the end you should have a map of the directions the field is actively pushing on, an explicit sense of which parts of this book are durable and which are dated, and a set of open questions worth watching.

*Current as of early 2025.*

## 26.1 How to read this chapter

Everything before this point describes work that has been done and published. This chapter describes work in progress, which means it is speculation with citations.

The calibration, stated up front: **the specific techniques below will look dated within two years, and the constraints will not.** Memory hierarchies, bandwidth, the cost of communication, the difference between verification and judgement, the fact that RL optimizes exactly what you specify — those are structural. Which optimizer, which parallelism configuration, which preference-optimization variant — those are contingent.

When this chapter is wrong, it will most likely be wrong in the direction of *underestimating how fast the boring engineering problems get solved and overestimating how fast the conceptual ones do.* That has been the pattern for the last several years.

## 26.2 Multi-modal

Text-only models are already the minority of frontier releases. The direction is toward models that natively handle images, audio, and video in and out, rather than bolting an encoder onto a text model.

**What is settled.** Vision-language input works. The dominant pattern — encode images into tokens, interleave them with text, train jointly — is stable and the engineering is understood.

**What is not.** Native generation across modalities in one model, video at length (an hour of video is an enormous number of tokens), and audio with the latency required for real conversation.

**What changes for the engineering in this book:**

- **Data pipelines get much harder.** Chapter 3's deduplication and quality filtering assume text. Near-duplicate detection for video, quality classification for audio, and licensing for images are each their own discipline.
- **Tokenization becomes a research problem again.** Chapter 4 is about text BPE. How to tokenize an image or a second of audio is contested, and the choice has the same architectural consequences BPE choices have.
- **Sequence lengths explode**, which makes Chapter 9's long-context work and Chapter 22's KV cache economics more load-bearing, not less.
- **Evaluation gets worse.** Chapter 23's problems, plus no equivalent of MMLU for most modality combinations.

**What does not change:** the parallelism arithmetic, the failure modes, the RL structure. A multi-modal training run is still bounded by memory and bandwidth in the same way.

## 26.3 Agent RL and the environment economy

The clearest direction, and the one Chapter 18 is a snapshot of.

The argument in one line: **verifiable rewards were the unlock (§15.4), agentic tasks are where verifiable rewards are most abundant, and environments are the constraint.**

What follows from that:

**Environments become the contested resource.** If you can turn a domain into a reproducible, sandboxed, resettable environment with programmatic success criteria, you can generate unbounded training signal in it. Expect heavy investment, expect it to be less published than model architecture, and expect environment engineering to be one of the fastest-growing job categories (§25.1).

**Computer use as the general interface.** Rather than an API per tool, operate a screen and keyboard. Vastly more general, vastly harder — pixel observations, unconstrained action space, longer horizons, worse credit assignment.

**Reliability rather than capability.** §18.9's pass^k gap is the honest summary: a model that solves a task 70% of the time solves it eight times running about 6% of the time. Closing that is a different problem from raising the single-attempt number, and it is what deployment requires.

**Multi-agent structures** — planner and executors, proposer and critic — are currently more prompting pattern than training target. Training the roles jointly is active and unsettled.

## 26.4 Synthetic data, second generation

Chapter 11 §11.3 covered where SFT data comes from. The trajectory is toward more of it being generated and less of it being collected, and the interesting question is where that stops.

**What works now:** rejection sampling against a verifier (§11.10), specialist models generating data for generalists (§19.6's distillation result), and difficulty escalation on prompts (§11.3's Evol-Instruct).

**Where it is going:**

- **Verifier-generated data at scale** in every domain where a verifier can be constructed. This is the same push as §26.3 seen from the data side — they are the same bet.
- **Self-improvement loops** where a model generates, verifies, filters, and trains on its own output across rounds. Works where verification is reliable. Where it is not, it is a known route to distribution collapse.
- **Curriculum generation** — models proposing problems at the right difficulty for the current policy. §15.9 established that prompts near 50% pass rate carry the signal; automating the search for them is high-leverage and underexplored.

**The constraint that has not moved:** synthetic data cannot create knowledge the system does not have access to. It can surface, recombine, and amplify what is latent (§19.4), and it can distil a strong model into a weak one. Whether it can produce genuinely new capability without external grounding is §15.11's open question, and a verifier *is* external grounding — which is precisely why verifiable domains are where synthetic data works.

**The failure mode to watch:** contamination laundering (§23.5). Data generated by a model trained on a benchmark carries the contamination with no textual overlap. As synthetic data grows as a fraction of training data, this gets harder to trace, not easier.

## 26.5 Efficiency and the compute wall

Frontier training runs cannot grow their compute at the rate of the last five years indefinitely. Power, fab capacity, and capital all bind. Several responses are visible.

**Lower precision.** FP8 training is production (§8.3, DeepSeek-V3). FP4 is being explored. Each halving buys roughly a doubling of effective compute and costs numerical headroom, and the per-block scaling lesson (§22.6) applies harder at each step.

**Sparsity.** MoE is sparsity at the layer level and it is standard. Finer-grained sparsity — within a matmul, or dynamic per-token depth — is researched and not yet routine.

**Better data rather than more.** If you cannot scale compute, scale quality. This is why Chapter 3 is the longest pre-training chapter, and why data engineering (§25.1) is the scarcest role.

**Inference compute as a substitute for training compute.** §16.6's result: at some budgets, spending compute at inference beats spending it on a larger model. This is the reasoning-model bet, and it changes the economics of the whole stack — it moves cost from a fixed training bill to a variable per-query one.

**Hardware diversity.** TPUs, Trainium, MI300, and a long tail of accelerators. The practical consequence for engineers is that framework-agnostic understanding (§25.8) gets more valuable and CUDA-specific expertise gets slightly less portable.

## 26.6 Long context, and whether it is the right frame

Context windows went from 2K to 1M in about three years. Two honest observations.

**Advertised length exceeds effective length.** RULER [\[49\]](../appendix/b-references.md#49-ruler) established that models degrade well before their stated limit, and §9.11 covers this. Reporting both numbers is the professional standard and it is not universal.

**Attention is still quadratic.** FlashAttention (§20.5) made it memory-efficient, not sub-quadratic. At 1M tokens the compute is genuinely enormous, and the alternatives — linear attention, state-space models, hybrid architectures — trade quality for scaling in ways that have not clearly won.

**The reframe worth considering:** long context may be the wrong abstraction for what people want it for. Retrieval, explicit memory, and structured state are different answers to "the model should know about this large corpus," and they have better scaling properties. The field currently does both and the boundary is unsettled.

## 26.7 Interpretability entering the training loop

Interpretability has been largely a post-hoc discipline: train a model, then investigate it. The direction worth watching is interpretability that *feeds back*.

Concretely: using interpretability tools to detect a failure mode during training rather than after, to build evaluations that measure internal properties rather than outputs, or to intervene on representations directly.

This is early and it is the direction that would change the most about how the work in this book is done. Chapter 23's evaluation problem — you can only measure outputs — is a large constraint, and anything that relaxes it is significant.

## 26.8 What will not change

The durable content, which is most of why this book was written as it was.

**The arithmetic.** Memory per parameter, FLOPs per token, bytes per second. Chapter 6 §6.1's calculation has been true for every transformer ever trained and will be true for the next generation.

**The memory hierarchy.** Registers, SRAM, HBM, network. The numbers change; the ratios and the consequence — fusion, locality, minimizing data movement — do not.

**Communication is expensive.** Every parallelism decision in Chapter 6 is a trade between memory and communication, and that trade does not go away.

**Failure at scale is normal.** Chapter 21's arithmetic is a property of large numbers of components, not of any particular hardware generation.

**RL optimizes exactly what you specify.** §15.7's lesson, and the single most portable statement in Part III. Every property you care about but did not encode is free to degrade.

**Verification beats judgement where it is available.** §12.9's structural result. It has not been superseded and there is no reason to expect it to be.

**Evaluation is the bottleneck.** §23.1, and getting more true as generating candidates gets cheaper.

**Data is the product.** Chapter 19 §19.8's observation about what labs withhold: the algorithm, not the data. That pattern has held across every technical report cited in this book.

## 26.9 The questions actually open

Stated as open because confident answers to them are currently ahead of the evidence.

**How far does reasoning transfer?** Training on math and code improves math and code. Whether it improves reasoning far from the training domain is contested and the evaluations are not clean (§15.11).

**Is the base model the ceiling?** If RL only amplifies what pre-training made likely, reasoning is bounded by pre-training and RL is an extraction procedure. If RL composes new capability, the ceiling is elsewhere. §19.4 leans toward the first; the evidence is not decisive.

**What is the compute-optimal split** between pre-training, mid-training, and RL? No published scaling law. Probably the most valuable unpublished number in the field (§15.11).

**Does synthetic data have a ceiling** without external grounding? §26.4.

**Can agentic reliability be trained**, or does it require architectural change? The pass^k gap is large and it is not obviously a data problem.

**How do you evaluate a system you cannot out-think?** Chapter 23's problem, extended. Currently answered with verification where possible and human judgement where not, and both have limits.

## 26.10 What you should take from this chapter, and from the book

1. **This chapter will date; the constraints will not.** Memory, bandwidth, communication cost, and the verification-versus-judgement distinction are structural.
2. **Environments are the next contested resource.** Verifiable reward was the unlock, agentic tasks are where it is abundant, and building environments is the bottleneck.
3. **Reliability, not capability, is what agentic deployment needs.** The pass^k gap is the honest measure and it is not closing on its own.
4. **Synthetic data works where verification works.** A verifier is external grounding, which is why verifiable domains are the ones where self-improvement loops do not collapse.
5. **Watch contamination laundering.** As synthetic data grows, benchmark contamination becomes untraceable rather than merely undetected.
6. **Inference compute substitutes for training compute** at some budgets, which moves cost from a fixed bill to a variable one and changes the economics of the whole stack.
7. **Advertised context length is not effective context length**, and attention is still quadratic. Retrieval and structured memory are live alternatives, not obviously worse.
8. **The arithmetic compounds; the recipes do not.** This is §25.8's advice and it is the reason this book spends so many pages on back-of-envelope calculations.
9. **The most important open question is whether the base model is the ceiling.** Almost everything else about where the field goes next depends on the answer.
10. **The gap this book set out to close was never secret knowledge.** It was assembly — a few dozen technical reports, a few hundred papers, and the tacit context that connects them. That assembly is now public, free, and yours to extend.

---

## A closing note

You have reached the end of a book whose premise is that this material should not cost thousands of dollars or require competing for a slot.

Three things worth doing from here.

**Go read the primary sources.** This book is a map. The DeepSeek-V3 and R1 reports, the Llama 3 paper, the Tülu 3 recipe, the FlashAttention papers — read them directly. They will read very differently now than they would have before Chapter 1, and that difference is the thing the book was for.

**Build the evidence.** Chapter 24 §24.6. A merged PR, an honest write-up of something that did not work, a reproduction at small scale. The arithmetic is demonstrable on a free Colab tier; scale is not required to show that you understand scale.

**Fix what is wrong here.** The frontier moves and this book will go stale. A citation we got wrong, a number a newer report superseded, a lab that no longer runs — [these are the most valuable contributions](../../CONTRIBUTING.md), and the repository exists so that anyone can make them.

The knowledge was always assemblable. Now it is assembled, and it is free.

---

**Exercises:** [Chapter 26 problem set](../../exercises/ch26.md) — includes a calibration exercise on this chapter's predictions, and a design problem on converting a judgement domain into a verifiable one.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. FP8 training in production.
- [\[10\] DeepSeek-R1](../appendix/b-references.md#10-deepseek-r1) — DeepSeek-AI, January 2025. The verifiable-reward result this chapter extrapolates from.
- [\[49\] RULER](../appendix/b-references.md#49-ruler) — Hsieh et al., 2024. Effective versus advertised context length.
- [\[53\] Tülu 3](../appendix/b-references.md#53-tulu-3) — Lambert et al., November 2024. The open recipe worth reproducing.
- [\[74\] Scaling LLM Test-Time Compute](../appendix/b-references.md#74-test-time-compute) — Snell et al., August 2024. Inference compute as a substitute for parameters.
- [\[81\] τ-bench](../appendix/b-references.md#81-tau-bench) — Yao et al., June 2024. The pass^k reliability gap.
- [See full reference list](../appendix/b-references.md)
