# Preface

If you have ever read a job description for a "Pre-training Engineer" at a frontier lab — say, the Qwen team at Alibaba, the Grok team at xAI, the Gemini team at Google DeepMind, or the foundation model team at a major Chinese tech company — and felt that the role description was written in a foreign language, you are not alone.

Most of the LLM content available today is written for a different audience. It targets people who want to *use* a model — fine-tune a 7B parameter Llama on their company's Slack history, build a RAG system over their PDFs, ship a chat product on top of an OpenAI or Anthropic API. That audience is real, important, and well-served. This book is not for them.

This book is for the people who want to *build* the model.

## What this book is

A field guide to the actual engineering and research work that goes into training a frontier large language model from scratch in 2024–2026. We cover pre-training, mid-training, and post-training end-to-end, with the same depth that operating-systems textbooks give to virtual memory and distributed-systems textbooks give to consensus.

The case studies are anchored on **DeepSeek-V3 / R1** and **Qwen3** for a specific reason: these are the frontier runs whose teams have published the most honest, most complete technical detail. Where the DeepSeek team has written a technical report, we cite it. Where the Qwen team has published a blog post, we cite it. Where Anthropic or OpenAI has hinted at a technique in a system card, we say so. Where a lab has not published, we say so too, and we explain what is *known* from inference behavior, from the open-weight artifacts, and from the secondary literature.

## What this book is not

- **Not a beginner's introduction to transformers.** We assume you have read "Attention Is All You Need," you know what query/key/value are, you have trained a small model, and you understand backpropagation. If you do not, the standard references (the fast.ai course, Karpathy's "Zero to Hero" series, "Dive into Deep Learning") will get you there.
- **Not a survey of LLM applications.** We do not cover RAG, agents-in-production, prompt engineering, or application-layer design. Those are different books.
- **Not a hype piece.** Frontier training is expensive, fragile, and full of trade-offs that nobody talks about publicly. We cover those.
- **Not a tutorial on a specific framework.** Megatron, DeepSpeed, FSDP, JAX, Triton, vLLM, SGLang, and the rest are mentioned where they matter, but this is not their documentation.

## How this book is organized

The book is split into five parts.

**Part I — The Frontier.** Two chapters. The first walks through what a real frontier training run actually looks like, end-to-end, using DeepSeek-V3 as the spine. The second breaks down the org chart — who the "Pre-training Engineer," the "Distributed Systems Engineer," the "Post-training Researcher," and the "Kernel Engineer" actually are, and how they interact. By the end of Part I you should be able to read a frontier-lab JD and not be mystified.

**Part II — Pre-training.** Eight chapters covering data, tokenization, architecture, distributed training, cluster reality, optimization, mid-training, and a case study. This is the longest part. Pre-training is where the bulk of the compute goes, where most of the engineering effort goes, and where the largest gap exists between the public understanding and the reality.

**Part III — Post-training.** Nine chapters covering SFT, reward modeling, PPO/RLOO, DPO, GRPO and the reasoning revolution, process reward models, Constitutional AI, tool use, and a case study on DeepSeek-R1. Post-training is where the model becomes the model you actually use. It is also where the most rapid evolution is happening right now — the techniques that worked in 2023 are mostly obsolete.

**Part IV — Infra.** Four chapters on custom kernels, checkpointing, inference/serving, and evaluation. These are the supporting systems that make everything else possible. They are also where most of the "weird" JDs come from — the kernel work, the distributed-systems work, the serving work.

**Part V — The job.** Three chapters on translating JDs into skill stacks, career paths, and where the field is going next. This is the part to read if you are deciding whether to apply, what to learn, and where to position yourself.

## A note on timing

The frontier is moving fast. Between the time this book is published and the time you read it, several of the techniques described here will have been superseded. We have tried to anchor on the *principles* (why FP8 works, what GRPO is fundamentally doing, how MoE routing learns) rather than the *specific configuration* (a particular learning rate, a particular GPU count). The configurations get stale; the principles do not.

## A note on honesty

Frontier labs do not publish everything. Some techniques are trade secrets; some are simply too in-progress to write up; some would be embarrassing if described candidly. Where we know something from public sources, we cite them. Where we are inferring, we say so. Where we genuinely do not know, we say that too. The frontier has enough hype; this book tries not to add to it.

---

Let us begin.
