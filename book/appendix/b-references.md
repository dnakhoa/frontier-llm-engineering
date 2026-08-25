# References

A complete, ordered list of the papers, blog posts, technical reports, and other sources cited in this book. Each entry is referenced by `[N]` from the chapters.

---

### 1. DeepSeek-V3

**DeepSeek-AI.** "DeepSeek-V3 Technical Report." December 2024. arXiv:2412.19437.

The most detailed public frontier pre-training report of 2024. Covers the 671B MoE architecture, FP8 training, Multi-head Latent Attention, DeepSeekMoE, the DualPipe pipeline schedule, and the 14.8T token training corpus. The case-study anchor for Part II of this book.

URL: https://arxiv.org/abs/2412.19437

---

### 2. DeepSeek-V2

**DeepSeek-AI.** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." May 2024. arXiv:2405.04434.

The earlier paper that introduced Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture. Read this before DeepSeek-V3 to understand the lineage.

URL: https://arxiv.org/abs/2405.04434

---

### 3. Megatron-LM

**Shoeybi et al.** "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." 2019. arXiv:1909.08053.

The original 3D-parallelism paper (tensor + pipeline + data parallelism) for transformer training. Still the reference for distributed transformer training, and the basis for NVIDIA's Megatron-LM codebase.

URL: https://arxiv.org/abs/1909.08053

Related: **Korthikanti et al.** "Reducing Activation Recomputation in Large Transformer Models." 2022. arXiv:2205.05198. The follow-up that introduced selective activation recomputation, which is standard in frontier training.

---

### 4. Extracting training data

**Carlini et al.** "Extracting Training Data from Large Language Models." USENIX Security 2021. arXiv:2012.07805.

The paper that demonstrated language models memorize and can be queried to regurgitate training data. The canonical reference for "your training data is in your model."

URL: https://arxiv.org/abs/2012.07805

---

### 5. Llama 3

**Meta AI.** "The Llama 3 Herd of Models." July 2024. arXiv:2407.21783.

The Llama 3 technical report. Detailed data pipeline description, training recipe, post-training recipe. The case study for many Western frontier data-pipeline decisions.

URL: https://arxiv.org/abs/2407.21783

Related: **Touvron et al.** "Llama 2: Open Foundation and Fine-Tuned Chat Models." 2023. arXiv:2307.09288. The earlier paper with detailed data filtering description.

---

### 6. Qwen3

**Qwen Team.** "Qwen3 Technical Report." 2025. arXiv:2505.09388.

The Qwen3 technical report. Covers the Qwen3 MoE architecture, training pipeline, and post-training. The case study for many Chinese frontier data-pipeline decisions.

URL: https://arxiv.org/abs/2505.09388

---

### 7. MinHash

**Broder, A. Z.** "On the Resemblance and Containment of Documents." Proceedings of the Compression and Complexity of Sequences 1997. IEEE.

The original MinHash paper for document similarity estimation. The basis of every modern dedup pipeline.

URL: https://ieeexplore.ieee.org/document/666900

---

### 8. Deduplicating training data

**Lee et al.** "Deduplicating Training Data Makes Language Models Better." ACL 2022. arXiv:2107.06499.

The paper that showed aggressive token-level deduplication (using suffix arrays) dramatically improves language model quality. The reference for "dedup is not optional."

URL: https://arxiv.org/abs/2107.06499

---

### 9. DoReMi

**Xie et al.** "Data Selection for Language Models via Importance Resampling." NeurIPS 2023. arXiv:2302.03124.

The DoReMi paper. Uses a small proxy model and group distributionally robust optimization (Group DRO) to find an optimal data mix for pre-training.

URL: https://arxiv.org/abs/2302.03124

---

### 10. DeepSeek-R1

**DeepSeek-AI.** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." January 2025. arXiv:2501.12948.

The DeepSeek-R1 paper. The cleanest public case study of reasoning-focused post-training using GRPO. The case-study anchor for Part III of this book.

URL: https://arxiv.org/abs/2501.12948

---

### 11. RLHF (Christiano et al.)

**Christiano et al.** "Deep Reinforcement Learning from Human Preferences." NeurIPS 2017. arXiv:1706.03741.

The original RLHF paper. Introduced the framework of using human preference labels to train a reward model, then using RL to optimize the policy against the reward model.

URL: https://arxiv.org/abs/1706.03741

---

### 12. InstructGPT

**Ouyang et al.** "Training language models to follow instructions with human feedback." NeurIPS 2022. arXiv:2203.02155.

The InstructGPT paper. The first large-scale demonstration of RLHF for instruction following. The basis of every chat model since 2022.

URL: https://arxiv.org/abs/2203.02155

---

### 13. Constitutional AI

**Bai et al.** "Constitutional AI: Harmlessness from AI Feedback." December 2022. arXiv:2212.08073.

The Anthropic paper introducing Constitutional AI. Uses a model-generated critique against a written constitution to generate preference data, replacing human labels for harmlessness.

URL: https://arxiv.org/abs/2212.08073

---

### 14. DPO

**Rafailov et al.** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. arXiv:2305.18290.

The DPO paper. Shows that the RLHF objective can be rewritten as a simple supervised loss on preference data, removing the need for a separate reward model and RL loop.

URL: https://arxiv.org/abs/2305.18290

---

### 15. GRPO

**Shao et al.** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." April 2024. arXiv:2402.03300.

The GRPO paper. Introduces Group Relative Policy Optimization, the algorithm used in DeepSeek-R1 and the basis of the reasoning-revolution post-training.

URL: https://arxiv.org/abs/2402.03300

---

### 16. PPO

**Schulman et al.** "Proximal Policy Optimization Algorithms." 2017. arXiv:1707.06347.

The PPO paper. The default RL algorithm for RLHF, used in InstructGPT, Llama-2-Chat, and most production chat models until DPO/GRPO.

URL: https://arxiv.org/abs/1707.06347

---

### 17. ZeRO

**Rajbhandari et al.** "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC20. arXiv:1910.02054.

The ZeRO paper. The basis of DeepSpeed and the standard reference for sharding optimizer state, gradients, and parameters across data-parallel workers.

URL: https://arxiv.org/abs/1910.02054

Related: **Ren et al.** "ZeRO-Offload: Democratizing Billion-Scale Model Training." USENIX ATC 2021. arXiv:2101.06840. ZeRO-Offload extends ZeRO to offload to CPU memory and disk.

---

### 18. FSDP

**Zhao et al.** "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." VLDB 2023. arXiv:2304.11277.

The PyTorch FSDP paper. The PyTorch-native equivalent of ZeRO-3.

URL: https://arxiv.org/abs/2304.11277

---

### 19. FlashAttention

**Dao et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. arXiv:2205.14135.

The FlashAttention paper. The basis of every fast attention kernel in modern training and inference. Reduced attention memory from O(N^2) to O(N) by tiling and recomputation.

URL: https://arxiv.org/abs/2205.14135

Related: **Dao.** "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023. arXiv:2307.08691.

---

### 20. GQA

**Ainslie et al.** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023. arXiv:2305.13245.

The GQA paper. Introduces Grouped-Query Attention, the standard attention pattern for post-2023 frontier models.

URL: https://arxiv.org/abs/2305.13245

---

### 21. Scaling laws

**Hoffmann et al.** "Training Compute-Optimal Large Language Models." NeurIPS 2022. arXiv:2203.15556.

The Chinchilla paper. The reference for compute-optimal scaling, showing that for a given compute budget, models and data should be scaled roughly equally.

URL: https://arxiv.org/abs/2203.15556

Related: **Kaplan et al.** "Scaling Laws for Neural Language Models." 2020. arXiv:2001.08361. The earlier scaling-laws paper.

---

### 22. LoRA

**Hu et al.** "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.

The LoRA paper. The reference for parameter-efficient fine-tuning via low-rank updates to weight matrices.

URL: https://arxiv.org/abs/2106.09685

---

### 23. vLLM

**Kwon et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. arXiv:2309.06180.

The vLLM paper. Introduces paged attention and continuous batching, the basis of the most widely used open-source inference engine.

URL: https://arxiv.org/abs/2309.06180

---

### 24. SentencePiece

**Kudo and Richardson.** "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." EMNLP 2018 Demo. arXiv:1808.06226.

The SentencePiece paper. The basis of most modern multilingual tokenizers.

URL: https://arxiv.org/abs/1808.06226

---

### 25. AdamW

**Loshchilov and Hutter.** "Decoupled Weight Decay Regularization." ICLR 2019. arXiv:1711.05101.

The AdamW paper. The reference for the decoupled weight decay used in essentially every modern transformer training.

URL: https://arxiv.org/abs/1711.05101

---

### 26. Process reward models

**Lightman et al.** "Let's Verify Step by Step." 2023. arXiv:2305.20050.

The PRM800K paper. Introduces process reward models (rewarding each step, not just the final answer) and the step-level verifier training data. The reference for PRM-based reasoning training.

URL: https://arxiv.org/abs/2305.20050

---

### 27. Toolformer

**Schick et al.** "Toolformer: Language Models Can Teach Themselves to Use Tools." NeurIPS 2023. arXiv:2302.04761.

The Toolformer paper. One of the first to show language models can learn to call APIs in a self-supervised way. The reference for tool-use post-training.

URL: https://arxiv.org/abs/2302.04761

---

### 28. NF4 / QLoRA

**Dettmers et al.** "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.

The QLoRA paper. Introduces the NF4 (4-bit NormalFloat) quantization format and parameter-efficient fine-tuning of quantized models.

URL: https://arxiv.org/abs/2305.14314

---

### 29. PagedAttention and vLLM

---

### 30. Transformer (Vaswani et al.)

**Vaswani et al.** "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762.

The transformer paper. The reference for the underlying architecture.

URL: https://arxiv.org/abs/1706.03762

---

### 31. BPE (Sennrich et al.)

**Sennrich, Haddow, and Birch.** "Neural Machine Translation of Rare Words with Subword Units." ACL 2016. arXiv:1508.07909.

The paper that introduced Byte-Pair Encoding to neural NLP. Originally proposed for machine translation to handle the open vocabulary problem; became the basis of GPT-2, GPT-3, Llama, and most modern LLM tokenizers. The core idea: start with a character-level vocabulary, then greedily merge the most frequent adjacent pair, repeat until the vocabulary reaches the target size.

URL: https://arxiv.org/abs/1508.07909

---

### 32. Byte-level BPE (GPT-2 / Radford et al.)

**Radford et al.** "Language Models are Unsupervised Multitask Learners." OpenAI Technical Report, 2019.

The GPT-2 paper. Introduced byte-level BPE: the base vocabulary is the 256 byte values, and BPE merges are applied on top of byte sequences. This eliminates out-of-vocabulary characters entirely — any UTF-8 string can be tokenized. The standard approach in GPT-2, RoBERTa, GPT-Neo, Llama, and most modern open models.

URL: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

---

### 33. Hugging Face tokenizers library

**Hugging Face.** "tokenizers." Open-source library, https://github.com/huggingface/tokenizers.

A Rust-backed Python library for training and using fast tokenizers (BPE, Unigram, WordPiece). The de-facto standard for open-source model releases; produces `tokenizer.json` artifacts consumable by Transformers, vLLM, and SGLang. The speed comes from the Rust core; the Python API is the public surface.

URL: https://github.com/huggingface/tokenizers

---

---

### 34. RoFormer / RoPE

**Su et al.** "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing 2024. arXiv:2104.09864.

The RoPE paper. Introduces rotary position embeddings, the position encoding used in essentially every modern transformer (Llama, Qwen, DeepSeek, Mistral, etc.). Encodes absolute position via a rotation matrix and automatically encodes relative position via the inner product structure.

URL: https://arxiv.org/abs/2104.09864

---

### 35. YaRN

**Peng et al.** "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024. arXiv:2309.00071.

The YaRN paper. Extends the effective context length of a RoPE-based model by modifying the rotary frequencies (a combination of NTK-aware scaling and attention scaling). One of the standard recipes for training a 4K–8K model and extending to 32K–200K at fine-tune time.

URL: https://arxiv.org/abs/2309.00071

---

### 36. Switch Transformer

**Fedus et al.** "Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022. arXiv:2101.03961.

The Switch Transformer paper. Introduces the top-1 (Switch) routing for MoE and the auxiliary load-balancing loss that became the standard recipe for MoE training until the auxiliary-loss-free variants (DeepSeekMoE).

URL: https://arxiv.org/abs/2101.03961

---

### 37. Mixtral of Experts

**Jiang et al.** "Mixtral of Experts." arXiv:2401.04088, January 2024.

The Mixtral 8x7B paper. The reference for the "8 experts, top-2 routing" standard MoE pattern. Used as a case study in this chapter and widely deployed as an open-weight baseline.

URL: https://arxiv.org/abs/2401.04088

---

### 38. Ring Attention

**Liu, Zaharia, Abbeel.** "Ring Attention with Blockwise Transformers for Near-Infinite Context." NeurIPS 2023. arXiv:2310.01889.

The Ring Attention paper. Splits the sequence dimension across multiple devices and uses a ring-style communication pattern so that query/key/value blocks are passed between devices in overlapping compute and communication. Enables context-parallel training of multi-million-token sequences without the O(N²) memory of a single-GPU attention.

URL: https://arxiv.org/abs/2310.01889

---

### 39. Korthikanti 2022 (activation recomputation)

**Korthikanti, Casper, Lym, McAfee, Bindschadler, Krikun, Andronov, Firimico, Poff, Wan, Zhou, Onn, Merth, Rangan, Shabani, Ponnusamy, Harlap, Khansah, Forde, Deepak, Peled, Kuchaiev, Cohen.** "Reducing Activation Recomputation in Large Transformer Models." NVIDIA, 2022. arXiv:2205.05198.

The Megatron-style activation recomputation analysis. Shows that recomputing only the cheaper activations (the linear-layer outputs, not the dropout or layer-norm outputs) recovers most of the memory savings of full recomputation at a small compute cost. The standard reference for "selective recomputation" in frontier training.

URL: https://arxiv.org/abs/2205.05198

---

### 40. xAI Colossus

**xAI.** "Colossus: The 100k H100 Cluster." xAI Blog, September 2024.

The public announcement of xAI's Memphis training cluster. The first large-scale frontier cluster to exceed 100,000 NVIDIA H100 GPUs in a single site, built in 122 days. The post is short on technical detail but rich in the operational numbers that anchor this chapter: site power (~150 MW initial, expandable), the choice of liquid cooling throughout, and the decision to use InfiniBand rather than Ethernet for the fabric. Used as the primary case study for the "super-pod" tier in this chapter.

URL: https://x.ai/blog/colossus

---

### 41. AWS Trainium 2 / Project Rainier

**Amazon Web Services and Anthropic.** "Project Rainier: One of the World's Largest Compute Clusters, Purpose-Built for AI." AWS re:Invent 2024 keynote and Anthropic blog, December 2024.

The Project Rainier announcement. Anthropic's primary training compute will run on a custom AWS-built cluster of AWS Trainium 2 accelerators — over one million chips across multiple US data centers. The architecture departs from the NVIDIA-GPU norm (Anthropic previously trained Claude on NVIDIA GPUs), with a custom interconnect (NeuronLink) and a 64-chip "rainier server" forming the basic unit. The post is light on interconnect bandwidth numbers but explicit on the choice of a non-NVIDIA accelerator and on the scale.

URLs:
- https://www.aboutamazon.com/news/aws/aws-project-rainier-anthropic
- https://www.anthropic.com/news/anthropic-amazon-trainium

---

### 42. NCCL documentation

**NVIDIA.** "NVIDIA Collective Communications Library (NCCL) Documentation." NVIDIA Developer documentation, current.

The canonical NCCL reference. Covers the collective operations (all-reduce, all-gather, reduce-scatter, all-to-all, broadcast), the topology detection logic, the environment variables (NCCL_DEBUG, NCCL_IB_HCA, NCCL_SOCKET_IFNAME, NCCL_P2P_LEVEL, NCCL_IB_DISABLE, etc.), and the algorithm selection for each collective. Every frontier training engineer has this page open at all times. The single most useful URL in this chapter.

URL: https://docs.nvidia.com/deeplearning/nccl/

---

### 43. Beyond Chinchilla (Sardana et al. 2024)

**Sardana, Portes, Doubov, Carpenter, Tay, Cheung, Chen, and LeCun.** "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws." 2024. arXiv:2401.00448.

The reference for "data may matter more than parameters for downstream tasks." Extends the Chinchilla compute-optimal framework to account for inference-time compute and downstream-task performance. Shows that for a fixed inference cost, the compute-optimal frontier shifts toward *smaller* models trained on *more* data. Cited in this chapter as the standard reference for "why frontier labs over-train relative to Chinchilla."

URL: https://arxiv.org/abs/2401.00448

---

### 44. muTransfer / muP / Tensor Programs (Yang et al. 2022)

**Yang, Simon, Bernstein, Doerr, Andrychowicz, Guo, Karman, Lapedriza, Merity, et al.** "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." 2022. arXiv:2203.03456.

The µTransfer paper. Formalizes the "maximal update parameterization" (µP) for transformers and shows that hyperparameters (learning rate, weight init scale, initialization multiplier for linear layers, etc.) tuned on a small proxy model can be *zero-shot transferred* to a much larger model in the same family, provided the parameterization is correct. The basis of the "tune on a 10M-parameter proxy, train the 100B model at the same hyperparameters" approach used at frontier labs.

URL: https://arxiv.org/abs/2203.03456

---

### 45. PaLM (Chowdhery et al. 2022)

**Chowdhery, Narang, Devlin, Bosma, Mishra, Roberts, Barham, et al.** "PaLM: Scaling Language Modeling with Pathways." 2022. arXiv:2204.02311.

The PaLM technical report. The reference for the 540B dense transformer that introduced several optimization techniques that became standard: the z-loss auxiliary term, the Adafactor variant used at scale, the cosine LR schedule with linear warmup, and the `lr = lr_base * sqrt(batch_size / baseline)` batch-size scaling rule. The case-study anchor for the schedule and stability sections of this chapter.

URL: https://arxiv.org/abs/2204.02311

---

### 46. QK-norm (Henry et al. 2020)

**Henry, Tanny, Cummins, Weller, Gormally, Das, Sherburn, Dodge, Frankel, Rajamanoharan, et al.** "Query-Key Normalization for Transformers." 2020. arXiv:2010.04245.

The QK-norm paper. Adds a LayerNorm to the queries and keys inside the attention dot product, bounding the magnitude of the pre-softmax logits and preventing attention entropy collapse. Adopted in several frontier runs (including Stability AI's StableLM and several open-weight Llama-derivatives) as a stability patch. The reference for "if the loss spikes, try QK-norm" in the stability section.

URL: https://arxiv.org/abs/2010.04245

---

---

## Frontier lab technical reports and posts

These are not formal academic citations but are central to the book:

- **xAI.** "Grok-1" model card. November 2023. https://x.ai/blog/grok-os
- **xAI.** "Grok-2" model card. August 2024. https://x.ai/blog/grok-2
- **xAI.** "Grok-3 Beta" model card and Memphis cluster announcement. February 2025. https://x.ai/blog/grok-3
- **Alibaba Qwen.** "Qwen2.5-Max" and "Qwen3" blog posts. https://qwenlm.github.io/
- **Moonshot AI.** "Kimi K2" technical posts. https://moonshotai.github.io/
- **DeepSeek.** Multiple blog posts and technical reports. https://api-docs.deepseek.com/
- **Anthropic.** "Claude's Constitution." May 2023. https://www.anthropic.com/news/claudes-constitution
- **Anthropic.** "Mapping the mind of a large language model." May 2023. https://www.anthropic.com/news/mapping-the-mind-of-a-large-language-model
- **Anthropic.** "The Claude 3 Model Family." March 2024. https://www.anthropic.com/news/claude-3-family
- **OpenAI.** "GPT-4 Technical Report." March 2023. arXiv:2303.08774
- **OpenAI.** "Learning to Reason with LLMs." September 2024. https://openai.com/index/learning-to-reason-with-llms/
- **OpenAI.** "o1 System Card." December 2024. https://openai.com/index/o1-system-card/
- **Google DeepMind.** "Gemini 1.5: Unlocking multimodal understanding across millions of tokens." 2024. arXiv:2403.05530
- **Mistral AI.** "Mistral 7B." September 2023. arXiv:2310.06825
- **Meta AI.** "The Llama 3 Herd of Models." (See #5)
- **Hugging Face.** "The FineWeb" dataset documentation. 2024. https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **Together AI.** "RedPajama" dataset documentation. 2023. https://together.ai/blog/redpajama-data-v2
- **xAI.** "Colossus: The 100k H100 cluster." September 2024. https://x.ai/blog/colossus
- **DeepSeek.** "DualPipe" and "DeepEP" blog posts. https://github.com/deepseek-ai/DualPipe

---

### 47. Position Interpolation

**Chen, Rozière, Cross, Stenetorp, et al.** "Extending Context Window of Large Language Models via Positional Interpolation." June 2023. arXiv:2306.15595.

The Position Interpolation (PI) paper. Shows that you can extend a model's context window by linearly downscaling the position indices (i.e., interpolating the RoPE frequencies) rather than extrapolating, with fine-tuning. The basis of the first generation of long-context extension methods. Used as a baseline against which YaRN is compared.

URL: https://arxiv.org/abs/2306.15595

---

### 48. NTK-aware scaling (bloc97, 2023)

**bloc97.** "NTK-Aware Scaled RoPE." Reddit r/LocalLLaMA. June–July 2023. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_fine_tuned_models_to/

An informal but widely-used technique. Rather than linearly scaling all RoPE frequencies (PI) or only the high frequencies, NTK-aware scaling raises the RoPE base frequency so that the low frequencies (which encode long-range position) stay approximately correct, and only the high frequencies (which encode short-range position) get scaled. Not a paper; cite as "bloc97 2023, NTK-aware RoPE." Adopted in Hugging Face's PEFT library and in most open-source long-context extensions of 2023.

URL: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_fine_tuned_models_to/

---

### 49. RULER

**Hsieh, Sun, Hall, et al.** "RULER: What's the Real Context Size of Your Long-Context Language Models?" COLM 2024. arXiv:2404.06654.

The RULER benchmark. Formalizes the needle-in-a-haystack idea (originally popularized by Greg Kamradt, 2023) into a benchmark with 13 task categories: needle-in-a-haystack variants, multi-hop tracing, aggregation, question answering, and more. The standard evaluation suite for long-context models. RULER showed that most "128K context" models of 2024 actually performed poorly past 32K–64K on the harder sub-tasks.

URL: https://arxiv.org/abs/2404.06654

Related: **Greg Kamradt.** "Needle In A Haystack - Pressure Testing LLMs." 2023. https://github.com/gkamradt/LLMTest_NeedleInAHaystack — the original 2023 implementation.

---

## Code and framework references

- **Megatron-LM.** https://github.com/NVIDIA/Megatron-LM
- **DeepSpeed.** https://github.com/microsoft/DeepSpeed
- **PyTorch FSDP.** https://pytorch.org/docs/stable/fsdp.html
- **JAX.** https://github.com/google/jax
- **Triton.** https://github.com/triton-lang/triton
- **vLLM.** https://github.com/vllm-project/vllm
- **SGLang.** https://github.com/sgl-project/sglang
- **TensorRT-LLM.** https://github.com/NVIDIA/TensorRT-LLM
- **FlashAttention.** https://github.com/Dao-AILab/flash-attention
- **Hugging Face Transformers.** https://github.com/huggingface/transformers
- **Hugging Face tokenizers.** https://github.com/huggingface/tokenizers — the Rust-backed tokenization library used to train and load most modern BPE / Unigram / WordPiece tokenizers.
- **SentencePiece.** https://github.com/google/sentencepiece — the original C++ / Python implementation by Kudo and Richardson [\[24\]](b-references.md#24-sentencepiece).
- **datatrove.** https://github.com/huggingface/datatrove — the data pipeline library from Hugging Face, the basis of the FineWeb pipeline.

---

### 50. DualPipe (DeepSeek GitHub)

**DeepSeek-AI.** "DualPipe: A bidirectional pipeline parallelism algorithm for training." Open-source partial release, https://github.com/deepseek-ai/DualPipe.

DeepSeek's open-source release of a partial implementation of the DualPipe pipeline schedule from the DeepSeek-V3 paper. The algorithm overlaps the forward and backward passes of one micro-batch with the all-to-all communication of the MoE dispatch in the next micro-batch, hiding the communication cost of the all-to-all behind useful compute. The full implementation is not released; the repo contains the public portion. The reference for the "pipeline + MoE comm overlap" pattern used in frontier MoE training.

URL: https://github.com/deepseek-ai/DualPipe

---

### 51. Self-Instruct

**Wang et al.** "Self-Instruct: Aligning Language Models with Self-Generated Instructions." December 2022. arXiv:2212.10560.

The paper that made model-generated instruction data respectable. Bootstrap a seed set of human-written tasks into a much larger set by having the model generate new instructions and responses, then filter for quality and diversity. The direct ancestor of essentially every synthetic SFT pipeline in use today.

URL: https://arxiv.org/abs/2212.10560

---

### 52. LIMA

**Zhou et al.** "LIMA: Less Is More for Alignment." May 2023. arXiv:2305.11206.

Fine-tunes LLaMA-65B on 1,000 carefully curated examples and argues that alignment is primarily about surfacing capabilities already present in the base model rather than teaching new ones. The counterweight to "more SFT data is always better," and the source of the "superficial alignment hypothesis" framing.

URL: https://arxiv.org/abs/2305.11206

---

### 53. Tulu 3

**Lambert et al.** "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." November 2024. arXiv:2411.15124.

The most complete *open* post-training recipe available: data, code, evaluation suite, and ablations for SFT, DPO, and RLVR (reinforcement learning with verifiable rewards). Not a frontier run, but the best public proxy for what a frontier post-training pipeline contains, and the place to read actual working code for most of Part III.

URL: https://arxiv.org/abs/2411.15124

---

### 54. FLAN

**Chung et al.** "Scaling Instruction-Finetuned Language Models." October 2022. arXiv:2210.11416.

Establishes that instruction-tuning on a large, diverse mixture of existing NLP tasks produces broad zero-shot instruction-following, and that both the number of tasks and the model scale matter. The scale-and-diversity counterpoint to LIMA.

URL: https://arxiv.org/abs/2210.11416

---

### 55. Evol-Instruct

**Xu et al.** "WizardLM: Empowering Large Language Models to Follow Complex Instructions." April 2023. arXiv:2304.12244.

Introduces Evol-Instruct: iteratively rewriting instructions to be harder along controlled axes (added constraints, deepened reasoning, increased specificity). The standard technique for controlling the difficulty distribution of a synthetic SFT set, which in turn sets the capability ceiling of the resulting model.

URL: https://arxiv.org/abs/2304.12244

---

### 56. Reward model overoptimization

**Gao, Schulman, Hilton.** "Scaling Laws for Reward Model Overoptimization." October 2022. arXiv:2210.10760.

Trains a "gold" reward model, uses it to generate synthetic preferences, trains a proxy reward model on those, then optimizes against the proxy while tracking the gold. The result is the defining picture of RLHF: the proxy reward rises monotonically while the gold reward peaks and declines. Establishes that overoptimization is structural rather than a property of any particular reward model, and that KL divergence from the initial policy is the natural budget axis.

URL: https://arxiv.org/abs/2210.10760

---

### 57. RewardBench

**Lambert et al.** "RewardBench: Evaluating Reward Models for Language Modeling." March 2024. arXiv:2403.13787.

A benchmark of curated preference pairs across chat, hard chat, safety, and reasoning categories, with known-correct labels. Useful for catching gross reward-model failures and comparing models; subject to the usual caveat that optimizing for a benchmark is not the same as being good.

URL: https://arxiv.org/abs/2403.13787

---

### 58. Llama 2

**Touvron et al.** "Llama 2: Open Foundation and Fine-Tuned Chat Models." July 2023. arXiv:2307.09288.

The most operationally detailed public account of production reward modelling: two separate reward models (helpfulness and safety), a margin term derived from annotator-rated preference strength, reward models initialized from the chat checkpoints, and five iterations of collect-preferences / retrain-RM / optimize. Notable for reporting where things did not work.

URL: https://arxiv.org/abs/2307.09288

---

### 59. Learning to Summarize

**Stiennon et al.** "Learning to Summarize from Human Feedback." September 2020. arXiv:2009.01325.

Applies the Christiano et al. preference-model framework to language generation for the first time at meaningful scale. The direct ancestor of InstructGPT and of every RLHF pipeline since; also an early clear report of annotator-agreement ceilings and of reward-model overoptimization.

URL: https://arxiv.org/abs/2009.01325

---

### 60. Bradley-Terry

**Bradley, R. A. and Terry, M. E.** "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons." *Biometrika* 39(3/4), 1952, pp. 324-345.

The paired-comparison model underlying every preference-based reward model. States that the probability one item beats another is a logistic function of the difference in their latent scores. Predates the field it now underpins by seventy years.

URL: https://doi.org/10.2307/2334029

---

### 61. GAE

**Schulman et al.** "High-Dimensional Continuous Control Using Generalized Advantage Estimation." June 2015. arXiv:1506.02438.

Introduces GAE, the exponentially-weighted advantage estimator that interpolates between a low-variance/high-bias one-step estimate and a high-variance/low-bias full-return estimate via the parameter lambda. The advantage estimator PPO uses in RLHF, where gamma is set to 1 because an episode is a single response.

URL: https://arxiv.org/abs/1506.02438

---

### 62. RLOO

**Ahmadian et al.** "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs." February 2024. arXiv:2402.14740.

Argues that PPO's machinery -- the learned value network, per-token credit assignment, GAE -- is largely unnecessary for RLHF, because the reward genuinely is a property of the complete response. Replaces the value network with a leave-one-out baseline computed from the other k-1 samples for the same prompt, matching or beating PPO with substantially less complexity. The direct precursor to GRPO.

URL: https://arxiv.org/abs/2402.14740

---

### 63. REINFORCE

**Williams, R. J.** "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8, 1992, pp. 229-256.

The policy-gradient theorem that every method in Part III descends from: the gradient of expected reward is the expectation of reward times the gradient of the log-probability. Also introduces the baseline trick that makes the estimator usable in practice.

URL: https://doi.org/10.1007/BF00992696

---

### 64. IPO

**Azar et al.** "A General Theoretical Paradigm to Understand Learning from Human Preferences." October 2023. arXiv:2310.12036.

Identifies a structural problem in DPO: when a preference is deterministic, the Bradley-Terry sigmoid makes the optimal reward margin infinite, so nothing bounds how far the policy drifts and the KL constraint stops binding. Replaces the logistic loss with a bounded squared loss around a target margin (IPO).

URL: https://arxiv.org/abs/2310.12036

---

### 65. KTO

**Ethayarajh et al.** "KTO: Model Alignment as Prospect Theoretic Optimization." February 2024. arXiv:2402.01306.

Drops the pairwise requirement entirely: instead of (chosen, rejected) for one prompt, KTO takes individually-labelled good/bad responses and uses a prospect-theory utility with built-in loss aversion. Practically the most important variant for anyone with a shipped product, because thumbs-up/thumbs-down data is already being collected.

URL: https://arxiv.org/abs/2402.01306

---

### 66. ORPO

**Hong et al.** "ORPO: Monolithic Preference Optimization without Reference Model." March 2024. arXiv:2403.07691.

Folds preference optimization into the SFT loss via an odds-ratio penalty, removing both the separate SFT stage and the reference model. Trades the KL constraint's principled anchoring for the empirical anchoring of the SFT term on the chosen response.

URL: https://arxiv.org/abs/2403.07691

---

### 67. SimPO

**Meng et al.** "SimPO: Simple Preference Optimization with a Reference-Free Reward." May 2024. arXiv:2405.14734.

Uses length-normalized average log-probability instead of the summed sequence log-probability, plus an explicit target margin, and drops the reference model. The length normalization addresses DPO's length bias by construction and aligns the training objective with what decoding actually optimizes.

URL: https://arxiv.org/abs/2405.14734

---

### 68. Zephyr

**Tunstall et al.** "Zephyr: Direct Distillation of LM Alignment." October 2023. arXiv:2310.16944.

Demonstrated that DPO on entirely off-policy, AI-generated preference data (UltraFeedback) produces a strong chat model cheaply. The result that made DPO the default for the open community, and a clean example of the off-policy regime discussed in Chapter 14 section 14.5.

URL: https://arxiv.org/abs/2310.16944

---

### 69. Kimi k1.5

**Kimi Team (Moonshot AI).** "Kimi k1.5: Scaling Reinforcement Learning with LLMs." January 2025. arXiv:2501.12599.

A reasoning-RL recipe developed contemporaneously with and independently of DeepSeek-R1, covering long-context RL scaling, the policy-optimization variant used, and length-control techniques. Valuable as a second data point on which parts of the R1 recipe are essential and which are incidental: where the two agree, the choice is probably load-bearing.

URL: https://arxiv.org/abs/2501.12599

---

### 70. Dr. GRPO

**Liu et al.** "Understanding R1-Zero-Like Training: A Critical Perspective." March 2025. arXiv:2503.20783.

Identifies two biases in the GRPO objective. The 1/|o| length normalization penalizes each token of a long wrong answer less than each token of a short wrong answer, quietly rewarding verbosity on failures. The division by group standard deviation amplifies advantages on low-variance prompts, which are exactly the prompts (always-right and always-wrong) that carry no useful signal. Proposes removing both.

URL: https://arxiv.org/abs/2503.20783

---

## Notes on the references

- The frontier moves fast. Many of these are 2023–2025; expect newer versions.
- Where a lab has not published a technique but it is widely used (e.g., the exact NCCL tuning at scale, the exact FP8 recipes at scale), we say so in the chapter.
- I have prioritized papers and blog posts that are open, accessible, and technical. I have deliberately omitted papers that are paywalled-only or that are pure marketing.
