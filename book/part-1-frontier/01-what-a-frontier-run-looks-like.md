# Chapter 1: What a real frontier training run looks like

> Reading time: ~25 minutes. By the end of this chapter you should be able to read a frontier-lab technical report and recognize every component in it.

## 1.1 The mental model

Most of the public discussion of "training an LLM" is a cartoon. You have some data. You have a model. You do forward pass, backward pass, update weights, repeat. The cartoon is not wrong, but it is missing almost everything that actually matters at frontier scale.

A real frontier training run, in 2024–2026, looks like this:

1. **A data pipeline** that ingests tens of petabytes of raw web text, code, books, papers, and synthetic data, deduplicates, filters, classifies by quality, removes contamination against evaluation benchmarks, and tokenizes into a stream of 10–15 trillion training tokens.
2. **A model architecture** — typically a decoder-only transformer with 100B–1T+ parameters, often a Mixture-of-Experts (MoE) variant, with attention patterns like Grouped Query Attention or Multi-head Latent Attention.
3. **A distributed training system** that splits the model across 2,000–50,000 GPUs using some combination of tensor parallelism, pipeline parallelism, data parallelism, context parallelism, and expert parallelism, communicating via NCCL or custom collectives.
4. **An optimizer** (almost always AdamW or a variant) with a learning-rate schedule that warms up over the first 1% of training and decays over the remainder, with precision in BF16, FP8, or a mix.
5. **A monitoring system** that watches loss curves, gradient norms, attention entropy, expert load balance, and dozens of other signals, alerting on a loss spike within minutes.
6. **A checkpointing system** that saves the full state of training (model weights, optimizer state, dataloader position, RNG state) every few hours to a distributed filesystem, and can resume seamlessly from any checkpoint.
7. **A failure-recovery system** that detects GPU hangs, NCCL timeouts, disk-full errors, OOM kills, and resumes training from the latest checkpoint, often masking 10+ failures per day in a large cluster.
8. **An evaluation system** that runs capability and safety benchmarks on a held-out set every few thousand steps, gating decisions about learning rate, data mix, and stop time.

That is the pre-training loop. Post-training is a different beast on top of it, which we cover starting in Chapter 11.

To make this concrete, let us walk through one real run end-to-end. The cleanest public example is **DeepSeek-V3** [\[1\]](../appendix/b-references.md#1-deepseek-v3). I will use it as the spine for this chapter and refer back to it throughout the book.

## 1.2 The case: DeepSeek-V3 in one page

DeepSeek-V3 is a 671B-parameter Mixture-of-Experts language model with 37B parameters active per token [\[1\]](../appendix/b-references.md#1-deepseek-v3). It was trained on 14.8 trillion tokens of multilingual data, on a cluster of 2,048 NVIDIA H800 GPUs, in approximately 2 months (2,788K H800 GPU-hours). The team reports a training cost of approximately $5.5M USD, which is roughly an order of magnitude cheaper than comparable Western frontier runs of the same period.

The architectural choices that made DeepSeek-V3 unusual:

- **Multi-head Latent Attention (MLA)** — a compressed Key-Value representation that reduces KV cache memory by 93% compared to standard Multi-Head Attention, without quality loss. This is what makes serving a 671B model economically viable.
- **DeepSeekMoE** — a fine-grained MoE architecture with 256 routed experts and 1 shared expert, using an auxiliary-loss-*free* load balancing scheme. This is one of the first large-scale demonstrations that you do not need the standard auxiliary loss to keep expert loads balanced.
- **FP8 mixed-precision training** — one of the first production-scale runs to use FP8 (E4M3 and E5M2 formats) for the bulk of the matmuls, with BF16 retained for certain numerically sensitive ops. This halves the memory and roughly doubles the throughput relative to BF16.
- **DualPipe** — a custom pipeline-parallelism schedule that overlaps attention and MoE communication with computation, hiding the network cost of the all-to-all collective.
- **Multi-token prediction (MTP)** — a training objective where the model predicts not just the next token but several future tokens at each position, providing a denser training signal.

The training-time data mix was approximately 14.8T tokens, with a customized ratio of English, Chinese, code, math, and multilingual content. The team reports that, by their internal evaluations, DeepSeek-V3 matches or exceeds Llama-3.1-405B-Instruct and GPT-4o on most benchmarks, while training for a fraction of the cost.

That is the one-paragraph version. The rest of this chapter unpacks each piece.

## 1.3 The data pipeline

The data pipeline is the single largest determinant of model quality. As the saying goes, often attributed to multiple labs: *"your model is your data."* This is not quite true — architecture and post-training matter — but it is much closer to true than most people realize.

A frontier pre-training data pipeline, in the order data flows through it:

```
Raw crawl (Common Crawl, web, code repos, books, papers, ...)
        │
        ▼
[Extraction]  HTML/JSON/PDF → plain text, with structure preserved
        │
        ▼
[Language ID]  fastText classifier, drop non-target languages
        │
        ▼
[Quality filtering]  heuristic (line length, symbol-to-word ratio, etc.)
                      + learned classifier (trained on "good" vs "bad" examples)
        │
        ▼
[Deduplication]  MinHash / SimHash across documents, suffix-array dedup within documents
        │
        ▼
[Contamination removal]  n-gram overlap against evaluation benchmark sources
        │
        ▼
[PII / harmful content filtering]  regex + classifier
        │
        ▼
[Domain mixing]  upweight / downweight by source domain (math, code, web, ...)
        │
        ▼
[Tokenization]  BPE / Unigram tokenizer, often with byte-fallback for code
        │
        ▼
[Shuffling and packing]  stream shuffled, packed into fixed-length sequences (e.g. 4096 tokens)
        │
        ▼
Pre-training data loader
```

For DeepSeek-V3 specifically, the 14.8T token corpus was built from a Chinese-common-crawl snapshot, English-common-crawl, code (GitHub), math (including synthetic math problems), and multilingual sources. The exact ratios are partially published; the team has stated that the data mix is iterated and refined based on validation-loss curves on a held-out set.

What makes this "weird" compared to what you might have seen in a tutorial:

- **Deduplication happens at multiple granularities.** Document-level (MinHash) catches exact and near-duplicate web pages. Paragraph-level catches sections copied across documents. Token-level (suffix array) catches repeated boilerplate inside a document. Each is a separate pipeline, often built on different infrastructure (Spark, Ray, custom C++).
- **Quality filtering is a model itself.** Frontier labs train a small classifier (often an XGBoost on simple features, or a 100M–1B parameter transformer) on a small labeled set of "high quality" vs "low quality" documents. The classifier is then used to filter the corpus. The classifier is itself trained on outputs from a strong model. This is circular and it is fine, because the strong model was trained on previous data of similar character.
- **Contamination removal is non-negotiable.** If your evaluation benchmark is in your training data, the eval is meaningless. Labs run n-gram overlap (typically 8-gram or 13-gram with a threshold) against every benchmark they care about (MMLU, GSM8K, HumanEval, etc.) and drop contaminated documents. This is the reason that, when a new benchmark drops, you cannot just trust a model's leaderboard score until the lab has re-run the contamination check.
- **Synthetic data is in the mix.** DeepSeek-V3 explicitly includes synthetic math and code in its training mix. The synthetic data is generated by previous DeepSeek models (including DeepSeek-Coder and earlier versions) and filtered for correctness.

The whole pipeline, end-to-end, takes weeks and is itself one of the hardest engineering challenges in the company. We cover data in detail in Chapter 3.

## 1.4 The architecture

The pre-training architecture for a 2024–2025 frontier model is almost always a decoder-only transformer, with one of three attention patterns:

- **Multi-Head Attention (MHA)** — the original. Q, K, V all have the same number of heads, the same head dimension. Memory cost: 2 × (num_layers × num_heads × head_dim × seq_len × num_kv_heads) for the KV cache. This is what GPT-3 used. It is rarely used for inference at frontier scale anymore because the KV cache is too big.
- **Grouped-Query Attention (GQA)** — multiple Q heads share a single K/V head. Memory cost: 2 × (num_layers × num_kv_heads × head_dim × seq_len). Used in Llama 2/3, Mistral, and most post-2023 dense models. The number of KV heads is typically 1/4 or 1/8 of the number of Q heads.
- **Multi-head Latent Attention (MLA)** — the Q, K, V are all compressed to a low-dimensional latent vector, then expanded back. Memory cost: 2 × (num_layers × latent_dim × seq_len), where latent_dim is much smaller than num_kv_heads × head_dim. Used in DeepSeek-V2 and V3. This gives 93% KV cache reduction compared to MHA at the same quality, which is what makes a 671B model servable.

For the FFN / MoE block, three patterns are common:

- **Dense FFN** — two linear layers with a SwiGLU or GeLU activation in between. Standard. Used in all dense models.
- **Standard MoE** — N "experts," each a dense FFN. Each token is routed to top-k experts (typically k=2 or k=4). Used in Mixtral, GPT-4 (rumored), and many others. Suffers from load imbalance: popular experts get more tokens, become more popular, get even more tokens. The classic fix is an auxiliary load-balancing loss, but this hurts model quality.
- **DeepSeekMoE** — fine-grained MoE with many small experts (256 in DeepSeek-V3) and a shared expert that all tokens pass through, plus an auxiliary-loss-*free* load balancing scheme based on bias terms on the routing scores. This gives finer routing, better quality per FLOP, and avoids the quality hit of the auxiliary loss.

A simplified block from the DeepSeek-V3 architecture, in PyTorch-ish pseudocode:

```python
class DecoderLayer(nn.Module):
    def __init__(self, config):
        self.attn = MLA(
            d_model=config.d_model,
            n_heads=config.n_heads,
            kv_latent_dim=config.kv_latent_dim,  # 512 in V3
            q_latent_dim=config.q_latent_dim,    # 1536 in V3
        )
        self.mlp = MoE(
            n_routed_experts=config.n_routed_experts,  # 256
            n_shared_experts=config.n_shared_experts,  # 1
            expert_dim=config.expert_dim,
            top_k=config.top_k,                        # 8
        )
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

The actual MLA module is more involved because the K and V projections share a latent vector, and the attention computation is rewritten in terms of that latent. We cover this in Chapter 5.

## 1.5 The distributed training system

DeepSeek-V3 was trained on 2,048 H800 GPUs connected via NVLink within a node (8 GPUs) and InfiniBand across nodes. The parallelism strategy:

- **Pipeline parallelism (PP)**: the 60 transformer layers are split across 4 pipeline stages (so each stage has 15 layers). DualPipe overlaps forward and backward of one micro-batch with the all-to-all of another.
- **Expert parallelism (EP)**: the 256 routed experts are split across the 8 GPUs within a node. The all-to-all collective is used to route tokens to their assigned experts.
- **Data parallelism (DP)**: the 2,048 GPUs are split into 256 DP groups of 8 GPUs each (since each node has 8 GPUs).
- **No tensor parallelism** within the MoE block, because the experts are already sharded. Dense layers use small TP for the attention heads.

The total parallelism is 4 (PP) × 8 (EP, which is also the node size) × 64 (DP, in terms of pipeline replicas) = 2,048 GPUs. This is not a unique configuration — the Qwen3 team used a similar 3D-parallel layout, as did Llama-3.

A real Megatron-LM-style configuration file for a 70B-scale model on 1,024 H100s, showing the 3D-parallel layout:

```yaml
# model
num_layers: 80
hidden_size: 8192
num_attention_heads: 64
num_query_groups: 8            # GQA
ffn_hidden_size: 28672
seq_length: 8192
# parallelism
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 4
num_layers_per_virtual_pipeline_stage: 10
context_parallel_size: 1
# data
micro_batch_size: 1
global_batch_size: 1024
# precision
bf16: true
fp8: true
fp8_e4m3: true
fp8_e5m2: false
# optimizer
optimizer: adam
adam_beta1: 0.9
adam_beta2: 0.95
adam_eps: 1e-8
weight_decay: 0.1
# schedule
lr: 3.0e-4
min_lr: 3.0e-5
lr_decay_style: cosine
lr_warmup_iters: 2000
train_iters: 1200000
```

The key fields a frontier job tunes:

- `tensor_model_parallel_size` and `pipeline_model_parallel_size` — set so that one model replica fits in one node (TP) and the pipeline depth is small enough to keep bubble time low.
- `num_query_groups` < `num_attention_heads` — this is GQA; set the ratio to 1:4 or 1:8.
- `micro_batch_size` × `pipeline_micro_batch_size` × `global_batch_size / dp_size` — keeps memory and pipeline bubble in balance.
- `fp8: true` + `fp8_e4m3: true` — uses FP8 for the bulk of the matmuls.
- `lr_warmup_iters` and `lr_decay_style` — cosine decay with a small floor (`min_lr`) is the most common.

We cover the entire 3D-parallelism machinery in Chapter 6 and the cluster / NCCL / failure-recovery machinery in Chapter 7.

## 1.6 The optimizer and precision

The optimizer is almost always AdamW, with a few frontier-specific tweaks:

- `beta1=0.9, beta2=0.95` (or 0.98 in some configs), `eps=1e-8`.
- Weight decay of 0.1 applied only to weight matrices, not to biases or layer norms.
- Gradient clipping to a global norm (typically 1.0) before the optimizer step.
- Learning rate warmup over 0.1%–1% of training, then cosine decay to a small floor (10% of peak LR).
- For very large batch sizes, sometimes a `lr = lr_base * sqrt(batch_size / baseline_batch_size)` rescaling.

Precision is where the recent frontier has moved:

- **2018–2022**: FP32 training, then FP16 mixed-precision (FP16 weights, FP32 master copy, FP32 optimizer state).
- **2022–2023**: BF16 mixed-precision. BF16 has the same dynamic range as FP32 but only 8 bits of mantissa; for transformers, the loss of mantissa is usually fine and the lack of overflow risk is a big win.
- **2024–2025**: FP8 mixed-precision. The dominant format is E4M3 for forward and weight gradients, E5M2 for activation gradients (because activation gradients have a wider dynamic range). The DeepSeek team was one of the first to publish an FP8-at-scale run.

Memory cost per parameter under BF16 + AdamW, for a 70B model:
- Weights in BF16: 70B × 2 = 140 GB
- Gradients in BF16: 140 GB
- Optimizer state (m and v in FP32): 70B × 4 × 2 = 560 GB
- Activations: depends on batch and sequence, typically 50–200 GB at moderate batch
- **Total: ~900 GB–1 TB just for the model**

This is why a 70B model needs at least 8 H100s (640 GB HBM) for inference and 64+ H100s for training. For a 671B model like DeepSeek-V3 with 37B active, the memory and bandwidth are even more extreme, which is why they use MoE + FP8 + lots of nodes.

## 1.7 The failure modes

The single least-discussed part of frontier training is what happens when things break. In a 2,000-GPU run, things break constantly:

- **GPU hardware failures**: HBM errors, NVLink errors, PCIe errors. Typically 1–2 per week across a 2k-GPU cluster.
- **NCCL hangs and timeouts**: collective communication operations that should take 100ms occasionally take 60s. Usually a transient network issue, sometimes a buggy collective.
- **Silent data corruption**: a single bit flip in a weight or gradient that produces a wrong number but does not crash. Frontier labs run periodic checksum verification on weights to catch this.
- **Loss spikes**: a sudden jump in training loss, sometimes recoverable (just skip the batch), sometimes a sign of a deeper problem.
- **OOM kills**: out-of-memory, typically when an unusual sequence length or an imbalance in MoE routing pushes memory over the limit.
- **Filesystem full**: the dataloader or the checkpointing system fills the parallel filesystem.
- **Node death**: an entire node crashes, taking 8 GPUs out. The training job must detect this and resume on the remaining 7/8 of the cluster.

The recovery loop looks like:
1. NCCL watchdog fires a timeout.
2. The training script catches the exception.
3. The latest checkpoint is loaded (from, say, 30 minutes ago).
4. The dataloader position is restored to match the checkpoint.
5. The RNG state is restored to match the checkpoint.
6. Training resumes.

A well-engineered frontier run has a **Mean Time Between Failures (MTBF)** of 4–8 hours at 2,048-GPU scale. That is, the run crashes, on average, 3–6 times a day. The goal of the training infrastructure is to make each recovery fast enough that the wall-clock cost of failure is small.

The DeepSeek team reported 2,788K H800 GPU-hours of *useful* training, on a 2,048-GPU cluster over 2 months, implying the cluster was utilized at a high rate. The failure-recovery time must have been small relative to the total.

## 1.8 The monitoring and evaluation loop

Throughout the run, the team watches:

- **Loss curve**: training loss, validation loss, validation loss on different sub-domains (math, code, web).
- **Gradient norm**: should stay roughly constant, modulo warmup and decay.
- **Attention entropy**: per-layer, per-head. A sudden collapse in attention entropy is a known failure mode.
- **Expert load balance**: in MoE, the fraction of tokens routed to each expert. Should be roughly uniform; a sudden skew means a routing problem.
- **Throughput**: tokens per second per GPU. Should stay roughly constant; a drop signals a problem.
- **GPU utilization**: SM occupancy, memory bandwidth utilization, NVLink utilization.

Evaluation runs are scheduled every few thousand steps:
- **Capability evals**: MMLU, GSM8K, HumanEval, MMLU-Pro, GPQA, etc. The model under evaluation is the current checkpoint, run with a small inference harness.
- **Held-out validation**: a held-out set of pre-training data, used to compute validation loss.
- **Safety evals**: refusal benchmarks, toxicity classifiers, etc. Typically only run on post-trained checkpoints.

A signal that often shows up around mid-training is *capability inversion*: a model gets better at some tasks (e.g., math) while plateauing on others (e.g., general knowledge). This is when the data mix is revisited, sometimes mid-run.

## 1.9 What the JD actually means

When a frontier lab posts a JD for a "Pre-training Engineer" or "Large Model Training Engineer," here is what the role actually involves, mapped to the components of this chapter:

| Component of the run | What the JD actually means |
|---|---|
| Data pipeline | Building or maintaining one of the pipeline stages in §1.3, or the orchestration that runs it on a Ray/Spark cluster |
| Architecture | Implementing a new attention variant, a new MoE design, or porting an architecture to a new framework |
| Distributed training | Adding a new parallelism strategy, debugging a ZeRO-3 memory cliff, or writing a custom collective |
| Optimizer & precision | Tuning the LR schedule, integrating FP8, implementing a custom optimizer |
| Failure recovery | Building the checkpoint / resume loop, the failure detector, the watchdog |
| Monitoring | Building the dashboard, the alerting, the eval pipeline |
| Cluster | Tuning NCCL, the network topology, the GPU placement |

A single engineer rarely owns all of this. The split is typically:

- **Pre-training Data Engineer** — owns §1.3.
- **Pre-training Researcher** — owns the architecture (§1.4), the optimizer (§1.6), the training-recipe tuning.
- **Large-Scale Training Engineer / Distributed Systems Engineer** — owns §1.5 and §1.7.
- **Kernel Engineer** — owns custom GPU kernels (Triton/CUDA) for the attention, the MoE all-to-all, the fused cross-entropy, etc. (Covered in Chapter 20.)
- **Inference Engineer** — owns serving the trained model efficiently (Chapter 22).
- **Post-training Researcher / Engineer** — owns SFT, RLHF, DPO, GRPO. (Part III.)

The rest of this book is structured around these roles.

## 1.10 What you should take from this chapter

If you remember nothing else:

1. **Frontier training is a pipeline of pipelines.** Data, architecture, distributed system, optimizer, failure recovery, evaluation — each is a substantial engineering project, and the run depends on all of them working.
2. **The data is the model.** More so than the architecture, more so than the optimizer, the data mix and quality determine capability.
3. **Failures are the steady state.** A 2,000-GPU run crashes multiple times a day. The infrastructure exists to make recovery cheap.
4. **The roles are specialized.** "Pre-training Engineer" can mean five different jobs depending on the lab. We will keep being specific.

The next chapter breaks down the org chart and the actual JDs at frontier labs.

---

**Exercises:** [Chapter 1 problem set](../../exercises/ch01.md) — includes the run-budget arithmetic and the incident-triage drill.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. The most detailed public pre-training report of 2024.
- [\[2\] DeepSeek-V2](../appendix/b-references.md#2-deepseek-v2) — Earlier paper, where MLA and DeepSeekMoE were introduced.
- [\[3\] Megatron-LM](../appendix/b-references.md#3-megatron-lm) — The original 3D-parallelism paper, still the reference for distributed transformer training.
- [See full reference list](../appendix/b-references.md)
