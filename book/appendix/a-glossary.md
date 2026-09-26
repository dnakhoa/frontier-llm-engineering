# Glossary

A working glossary of the terms used throughout the book. Where a term is local to one chapter, see that chapter for the deeper definition.

## A

**AdamW.** The de-facto optimizer for transformer training. Adam with decoupled weight decay. See [\[25\]](b-references.md#25-adamw).

**All-reduce.** A collective communication operation that sums (or averages) a tensor across all GPUs and returns the result to all GPUs. The standard way to do data-parallel gradient synchronization.

**All-to-all.** A collective communication operation where each GPU sends a chunk of its data to every other GPU. Used heavily in MoE training and inference for routing tokens to their assigned experts.

**Attention.** The core operation of a transformer: for each query, compute a weighted average of the values, where the weights are determined by the query-key similarity. Standard formulation: `softmax(QK^T / sqrt(d)) V`.

**Auxiliary loss.** A loss term added to the main loss to encourage a desired behavior (e.g., load balance across MoE experts) without being part of the primary task objective. The DeepSeekMoE design notably removes the auxiliary loss and uses bias-based load balancing instead.

## B

**BF16.** Brain Float 16: a 16-bit floating-point format with 8 bits of exponent and 7 bits of mantissa. Same dynamic range as FP32, half the storage of FP16. The default training precision for modern transformers.

**BPE (Byte-Pair Encoding).** A subword tokenization algorithm that greedily merges the most frequent adjacent pair of tokens. The basis of GPT, Llama, and most modern tokenizers.

## C

**Chinchilla scaling.** The compute-optimal scaling law from [\[21\]](b-references.md#21-scaling-laws), stating that for a given compute budget, model size and data should be scaled roughly equally. The reference for "you need ~20 tokens per parameter" (the 1:20 rule of thumb).

**Cluster.** The physical hardware (GPUs, networking, storage) that a training run runs on. A frontier cluster is 2,000–100,000 GPUs connected by NVLink within a node and InfiniBand across nodes.

**Common Crawl.** A non-profit that crawls the web and releases snapshots every 1–2 months. The single largest source of pre-training data.

**Constitutional AI.** The Anthropic technique [\[13\]](b-references.md#13-constitutional-ai) of using a written constitution and model-generated critiques to produce preference data, replacing human labels.

**Context parallelism (CP).** Splitting a long sequence across multiple GPUs so that each GPU holds a different slice of the sequence. Used for training on long contexts (>32K tokens).

## D

**Data parallelism (DP).** Replicating the model across multiple GPUs, each processing a different shard of the data, and synchronizing gradients via all-reduce.

**DPO (Direct Preference Optimization).** The technique from [\[14\]](b-references.md#14-dpo) that re-derives the RLHF objective as a supervised loss on preference data, removing the need for a separate reward model and RL loop.

**DeepSeekMoE.** The MoE architecture from DeepSeek [\[1\]](b-references.md#1-deepseek-v3) [\[2\]](b-references.md#2-deepseek-v2). Fine-grained experts (256 in V3), one shared expert, auxiliary-loss-free load balancing via bias terms.

## E

**E4M3 / E5M2.** Two 8-bit floating-point formats used in FP8 training. E4M3 has 4 exponent bits and 3 mantissa bits (used for forward pass and weight gradients). E5M2 has 5 exponent bits and 2 mantissa bits (used for activation gradients, where the dynamic range matters more).

**Expert parallelism (EP).** Splitting the experts of an MoE model across multiple GPUs. Tokens are routed to their assigned expert via an all-to-all collective.

## F

**FlashAttention.** The IO-aware exact attention algorithm from [\[19\]](b-references.md#19-flashattention). Tiles the attention computation to fit in SRAM and avoids materializing the full N×N attention matrix, reducing memory from O(N^2) to O(N).

**FP8.** 8-bit floating-point. Used in frontier training (DeepSeek-V3 onwards) for the bulk of the matmuls, with BF16 retained for numerically sensitive operations. Llama 3.1 used FP8 for inference only ([fact sheet](fact-sheets/llama-3.md#training-recipe)).

**FSDP (Fully Sharded Data Parallel).** The PyTorch-native equivalent of ZeRO-3 [\[18\]](b-references.md#18-fsdp). Shards model parameters, gradients, and optimizer state across data-parallel workers, with all-gather on demand.

**FT (Fine-Tuning).** Continuing training of a pre-trained model on a smaller, task-specific dataset. SFT, RLHF, DPO, and GRPO are all forms of fine-tuning.

## G

**GQA (Grouped-Query Attention).** The attention pattern from [\[20\]](b-references.md#20-gqa) where multiple Q heads share a single K/V head. Reduces KV cache memory by 4–8x compared to MHA at minimal quality cost.

**GRPO (Group Relative Policy Optimization).** The RL algorithm from [\[15\]](b-references.md#15-grpo) that replaces the value model with group-relative advantages. Used in DeepSeek-R1 and most reasoning-focused post-training.

## H

**H100 / H800 / A100 / B200.** NVIDIA GPU generations. H100 is the Hopper generation, standard for frontier training as of 2023–2025. H800 is a China-export-compliant variant with reduced interconnect bandwidth. B200 is the Blackwell generation, 2025+.

## I

**Inference.** Running a trained model to generate outputs. Distinct from training in that the model weights are fixed and only forward passes are run. Optimized differently (continuous batching, paged KV cache, speculative decoding).

## K

**Kernel.** A custom GPU function, typically written in CUDA or Triton, that implements a specific operation (e.g., attention, MoE routing) faster than the standard library.

**KV cache.** The cached Key and Value tensors from previous tokens in a sequence, used during inference to avoid recomputing them. The dominant memory cost in inference.

## L

**LLM (Large Language Model).** A transformer-based language model with >1B parameters, trained on >100B tokens. The subject of this book.

**LoRA (Low-Rank Adaptation).** The parameter-efficient fine-tuning technique from [\[22\]](b-references.md#22-lora) that adds low-rank updates to weight matrices, training only a small fraction of the parameters.

**LR (Learning Rate).** The step size for the optimizer. Typically scheduled with a warmup (linear increase from 0 to peak over the first 0.1%–1% of training) and a cosine decay to a small floor.

## M

**Megatron-LM.** The 3D-parallelism framework from [\[3\]](b-references.md#3-megatron-lm). Combines tensor, pipeline, and data parallelism. The reference implementation for distributed transformer training.

**MHA (Multi-Head Attention).** The original attention pattern. Q, K, V all have the same number of heads. Memory-intensive for inference.

**MLA (Multi-head Latent Attention).** The attention pattern from [\[2\]](b-references.md#2-deepseek-v2). Q, K, V are compressed to a low-dimensional latent vector, then expanded. 93% KV cache reduction vs MHA at the same quality.

**MinHash.** A probabilistic technique for estimating the Jaccard similarity between two sets. The basis of document-level deduplication.

**MoE (Mixture of Experts).** A model architecture where each FFN block is replaced with N "expert" FFNs, and each token is routed to a subset of them (typically top-k). Reduces inference compute by having only a fraction of the parameters active per token.

**MTP (Multi-Token Prediction).** A training objective where the model predicts not just the next token but several future tokens at each position. Provides a denser training signal.

## N

**NCCL.** NVIDIA Collective Communication Library. The standard library for collective communication (all-reduce, all-gather, all-to-all) on NVIDIA GPUs.

**NCCL hang.** When an NCCL collective operation times out, usually due to a network or hardware issue. A common failure mode in large-scale training.

## O

**Optimizer state.** The state maintained by the optimizer (e.g., m and v for Adam). Typically 2x the model size in FP32. The dominant memory cost in training.

## P

**Pipeline parallelism (PP).** Splitting the model layers across multiple GPUs, with each GPU processing a subset of layers and passing activations to the next. Reduces memory per GPU at the cost of pipeline bubble time.

**PPO (Proximal Policy Optimization).** The RL algorithm from [\[16\]](b-references.md#16-ppo) that is the default for RLHF. Uses a clipped surrogate objective to stabilize training.

**Pre-training.** The initial training of a language model on a large corpus of text, typically with a next-token-prediction objective. The "P" in GPT.

**PRM (Process Reward Model).** A reward model that scores each step of a reasoning chain, not just the final answer. The reference is [\[26\]](b-references.md#26-process-reward-models).

**Post-training.** All the training that happens after pre-training: SFT, RLHF, DPO, GRPO, etc. The phase that turns a base model into a chat / assistant / reasoning model.

## Q

**QLoRA.** The technique from [\[28\]](b-references.md#28-nf4--qlora) that combines 4-bit quantization (NF4) with LoRA for parameter-efficient fine-tuning of quantized models.

## R

**Reward hacking.** When a model learns to game the reward model without actually improving on the task. A central concern in RLHF and RL post-training.

**Reward model (RM).** A model trained to predict human preferences, used as the reward signal in RLHF.

**RLHF (Reinforcement Learning from Human Feedback).** The framework from [\[11\]](b-references.md#11-rlhf-christiano-et-al) and [\[12\]](b-references.md#12-instructgpt) of training a reward model on human preferences, then using RL (typically PPO) to optimize the policy against the reward model.

**RLAIF (RL from AI Feedback).** A variant of RLHF where the preference labels are generated by an AI (typically the model itself or a stronger model) rather than by humans. The basis of Constitutional AI.

**RLOO (REINFORCE Leave-One-Out).** A simpler RL algorithm than PPO that uses leave-one-out baselines. Used in some frontier post-training pipelines as a lighter-weight alternative to PPO.

## S

**Scaling laws.** Empirical relationships between model size, data size, and compute, and the resulting model quality. The reference is [\[21\]](b-references.md#21-scaling-laws).

**SFT (Supervised Fine-Tuning).** Continuing training of a pre-trained model on a curated dataset of (input, output) pairs. The first stage of post-training.

**Speculative decoding.** An inference technique where a small draft model generates tokens quickly and a larger model verifies them in parallel. Trades off extra draft-model compute for lower latency.

**SwiGLU.** A variant of the GLU activation used in most modern transformers. The FFN block is `down(swish(gate(x)) * up(x))` instead of the original `down(relu(up(x)))`.

## T

**Tensor parallelism (TP).** Splitting individual weight matrices across multiple GPUs, with each GPU computing a partial result and the results combined via all-reduce. Used within a node (typically 8 GPUs) due to NVLink bandwidth.

**Tokenization.** The process of converting text into a sequence of integer token IDs. The choice of tokenizer affects the effective sequence length and the model's efficiency on different languages and code.

**Triton.** A Python-like DSL for writing custom GPU kernels. Easier to use than CUDA for most cases, with comparable performance for typical operations.

## U

**Unigram.** A subword tokenization algorithm that starts with a large vocabulary and prunes to a target size. Used by SentencePiece for multilingual models.

## V

**vLLM.** The open-source inference engine from [\[23\]](b-references.md#23-vllm). Introduces paged attention and continuous batching.

## Z

**ZeRO (Zero Redundancy Optimizer).** The memory optimization from [\[17\]](b-references.md#17-zero) that shards optimizer state, gradients, and parameters across data-parallel workers. The basis of DeepSpeed.
