# Errata

This page lists every **never-true claim** the book has published: statements about what a primary source says that the source contradicts, and places where the book contradicted itself. Each was wrong on the day it was written. They are listed here in full, with what the source actually says and the version that fixed them. A book that asks you to read reports critically should show you its own mistakes.

Stale claims are different: those were accurate when written and have since been overtaken by newer work. They are not listed here. They go in the [changelog](../../CHANGELOG.md), and each chapter's currency stamp tells you how recently it was checked.

The text as first published is tagged `v1.0` in the repository, so every "the book said" below can be checked.

**If you worked an exercise before v1.0.1**, look for an *If you worked this before v1.0.1* line. Some corrections change a final answer, and a few reverse the conclusion.

---

### E-001 — The DeepSeek-V3 parallelism layout

- **The book said:** "4 (PP) × 8 (EP, which is also the node size) × 64 (DP)", over "60 transformer layers" split into 4 stages of 15, with "small TP" for dense layers. It also said V3 kept expert parallelism inside a node, on NVLink, and that this was "why DeepSeek's EP choice is exactly the node size".
- **The source says:** 16-way pipeline parallelism, 64-way expert parallelism spanning 8 nodes, ZeRO-1 data parallelism, and no tensor parallelism at all, over 61 layers. Expert parallelism deliberately crosses node boundaries. That is affordable because DualPipe overlaps the all-to-all with compute, and because each token is routed to at most 4 nodes. See the [fact sheet](fact-sheets/deepseek-v3.md#parallelism-layout).
- **Where:** Chapter 1 §1.5; Chapter 6 §6.10, §6.11, §6.13 and takeaways 8–9; Chapter 7 §7.3 and §7.4; Chapter 10 §10.7, its parallelism block, and takeaway 4; Exercise 6.6 and its solution. Solutions 6.9 and 10.1 already gave the correct layout, contradicting the chapters; they now link to the fact sheet.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Exercise 6.6 part 4 now uses 58 MoE layers, so the answer is ~55 GB per micro-batch rather than 56 (~870 GB per step rather than ~900). Part 5's conclusion changes more. "EP belongs inside a node" is a sound default, but it is not what V3 did, and V3 is the book's anchor example. V3's design point is cross-node EP made cheap, and it is the more interesting lesson.

### E-002 — The Chapter 4 tokenizer configurations

- **The book said:** three JSON blocks presented as the published tokenizer files for DeepSeek-V3, Llama-3 and Qwen3. All three had `"byte_fallback": true`, merge counts of 127,744 or 151,808, invented special tokens (`<|fim_begin|>`, `<|tool_call|>`), and a Qwen3 vocabulary of 152,064. §4.5 said Llama-3 "reserves 128 special tokens".
- **The source says:** `byte_fallback` is `false` in all three published `tokenizer.json` files. Merges are 127,741 (DeepSeek-V3), 280,147 (Llama-3) and 151,387 (Qwen3). Llama-3 has 256 special tokens. Qwen3 has 151,643 BPE entries plus 26 added (151,669, as its report says), padded to 151,936 embedding rows. The blocks are now generated from the hash-verified files; see the [tokenizer fact sheet](fact-sheets/tokenizers.md).
- **Where:** Chapter 4 §4.4 (vocabulary table), §4.5 (special tokens), §4.11 (all three blocks); Solution 4.7 part 1.
- **Fixed in:** v1.0.1

### E-003 — Which tokenizer isolates CJK, and which is "Chinese-optimized"

- **The book said:** Qwen3's pre-tokenizer adds a `[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]+` class, "which is what lets the tokenizer build a vocabulary" of Chinese tokens, and its "152K vocabulary is heavily Chinese-optimized, with a large fraction of merges" producing Chinese characters. It also said Qwen3 and Llama-3 split digits into 1–3 digit chunks, and showed possessive quantifiers in both regexes.
- **The source says:** Qwen3's published pre-tokenizer has **no CJK class**. It splits digits **one at a time**, and it differs from Llama-3's only in that. The tokenizer that isolates CJK runs is **DeepSeek-V3**'s. Counting the vocabularies, DeepSeek-V3 has 35,334 entries containing Han, Qwen3 25,511 and Llama-3 4,387, and the Qwen3 report says nothing about Chinese optimization. The published regexes have no possessive quantifiers. See the [tokenizer fact sheet](fact-sheets/tokenizers.md).
- **Where:** Chapter 4 §4.6, §4.8, §4.11 and takeaway 4.
- **Fixed in:** v1.0.1
- **If you read this before v1.0.1:** the chapter's causal story ran the wrong way. Qwen's efficiency on Chinese, whatever its size, is not explained by a CJK pre-tokenizer rule. The lesson that survives is the chapter's own: pre-tokenization matters, and you check it by opening the file.

### E-004 — Llama-3's embeddings and width

- **The book said:** "Llama-3 uses *tied* embeddings". Its vocabulary table gave `d_model = 4,096` for "Llama-3 (8B/70B/405B)".
- **The source says:** `tie_word_embeddings` is `false` in the Llama-3 config, and in DeepSeek-V3's and Qwen3's too. 4,096 is the 8B model's width only; the 70B and 405B models are wider. See the [tokenizer fact sheet](fact-sheets/tokenizers.md#llama-3).
- **Where:** Chapter 4 §4.4; Solution 5.1, which offered tied embeddings as an explanation.
- **Fixed in:** v1.0.1

### E-005 — A history attributed to the DeepSeek-V3 report

- **The book said:** DeepSeek-V3's 128K vocabulary was chosen "partly because they started from a 32K Llama-1-style tokenizer and grew it", citing the V3 report.
- **The source says:** only that the tokenizer is byte-level BPE "with an extended vocabulary of 128K tokens", with its pre-tokenizer and training data "modified to optimize multilingual compression efficiency" ([V3] §4.1). The history does not appear in the report.
- **Where:** Chapter 4 §4.10.
- **Fixed in:** v1.0.1

### E-006 — An invented config presented as real

- **The book said:** "A real Megatron-LM-style configuration file for a 70B-scale model on 1,024 H100s".
- **The source says:** it is not any lab's file; the book wrote it to show the shape of a 3D-parallel config. It is now labelled illustrative, as are the similar blocks in Chapters 6 and 9. The book's rule is now that a real config is generated from the published file, and anything else says so.
- **Where:** Chapter 1 §1.5; Chapter 6 §6.9; Chapter 9 §9.6.
- **Fixed in:** v1.0.1

### E-007 — DeepSeek-V3's attention: "4 KV heads" and a 1,024-number cache

- **The book said:** V3's MLA has "128 Q heads, 4 KV heads (GQA-like inside MLA)". It showed MLA code that repeats 4 KV heads 32×, and it said MLA caches $2 d_c = 1024$ numbers per token per layer (Chapter 5 §5.14.1; Exercise 5.2).
- **The source says:** MLA has no KV-head grouping. All 128 heads' K and V are reconstructed from one 512-d latent, and position is carried by a separate 64-d RoPE key shared across heads. Only those two are cached: **576** numbers per token per layer ([fact sheet](fact-sheets/deepseek-v3.md#architecture)).
- **Where:** Chapter 10 §10.4 and §10.4.1 (the code and its notes); Chapter 5 §5.14.1; Exercise 5.2 and its solution; the lab05 row label.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Exercise 5.2's MLA answers change from 164 KB/token, 21 GB per 128K sequence and 7 sequences in 10 GB, to **92 KB, 12.1 GB and 13**. MLA is 3.6× smaller than GQA 8:1, not 2×. Chapter 5's V3 cache comparison changes from 15 GB vs 30 GB to 9.2 GB vs 33 GB.

### E-008 — DeepSeek-V3 has 61 layers, the first three dense

- **The book said:** "60 transformer blocks"; "the first transformer layer is dense … The exact number of dense layers is not published, but later presentations suggest 1"; the MoE FFN is in "every layer except the first". It added that the pattern "is now used in Qwen3".
- **The source says:** 61 layers. "We substitute all FFNs except for the first three layers with MoE layers", so 58 MoE layers; the config's `first_k_dense_replace` is 3 ([fact sheet](fact-sheets/deepseek-v3.md#architecture)). The report gives no rationale for the dense layers. The Qwen3 claim had no source and is removed.
- **Where:** Chapter 10 §10.4, §10.4.2, §10.4.3 and takeaway 2; Chapter 5 §5.14.1; Exercise 5.3; Solutions 5.3, 10.1 and 10.8.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Exercise 5.3's routed-expert total changes from 676B ("essentially 100%") to **654B (~97%)**, leaving ~17B for attention, dense FFNs, shared experts and embeddings. Active parameters reconcile at ≈37.4B. Solution 10.8's "676B > 671B overshoot" no longer exists; the architecture adds up exactly.

### E-009 — DeepSeek-V3's multi-token prediction

- **The book said:** MTP was "introduced in DeepSeek-V3". The extra heads are "a single linear layer", "cheap", with the MTP "head" costing "0.6B parameters". It said MTP gives "1–2% on most evals" and "1.5–2x" speculative speedup, and uses a weight of 0.3 throughout. Chapter 10 added that MTP outputs provide a "denser preference signal for DPO" and that R1's reasoning traces are "an MTP-style rollout", reused by "the R1 distillation pipeline".
- **The source says:** V3's MTP is "inspired by Gloeckle et al. (2024)". It is a sequential module: one full Transformer block that combines the main model's state with the next token's embedding. The released checkpoint is 685B = 671B + **14B MTP**. The loss weight is 0.3 for the first 10T tokens, then 0.1. As a speculative draft, second-token acceptance is 85–90%, giving **1.8×** tokens per second. The "1–2%" figure and the DPO and R1 connections appear in neither the V3 nor the R1 report ([fact sheet](fact-sheets/deepseek-v3.md#architecture)).
- **Where:** Chapter 5 §5.12, §5.14.1 and takeaway 6; Chapter 10 §10.5 and §10.13; Exercise 10.5 and its solution; Solution 10.1.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Solution 10.5 part 2's overhead estimate changes from 0.6B (<0.1%) to **14B (~2%)**. The $12d^2$ rule is for dense blocks, and V3's MTP block is an MoE block. Compute overhead stays a few percent.

### E-010 — DeepSeek-V3's MoE gating and "removing" the auxiliary loss

- **The book said:** the Chapter 10 DeepSeekMoE code applied the balancing bias to the scores and then took the sigmoid of the *biased* scores as gating weights, unnormalized. The text said V3 "remove[s] the auxiliary loss".
- **The source says:** "the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score." Gating weights are the sigmoid affinities of the selected experts, normalized over them. V3 also keeps a complementary sequence-wise balance loss with weight 0.0001 ([fact sheet](fact-sheets/deepseek-v3.md#architecture)).
- **Where:** Chapter 10 §10.4.2 (code and text) and §10.6.
- **Fixed in:** v1.0.1

### E-020 — A block quote the DeepSeek-V3 report does not contain

- **The book said:** Chapter 7 block-quoted the V3 report: *"… with a total of 900 GB/s NVLink bandwidth per GPU. The 256 nodes are connected via InfiniBand, with each GPU having a 400 Gb/s NIC,"* and called it "the entire physical description of the cluster in the paper".
- **The source says:** no such sentence. §3.1 says only that each node's 8 GPUs are connected by NVLink and NVSwitch, and nodes by InfiniBand. The report's bandwidth figures, in §3.2.2, are **NVLink 160 GB/s and InfiniBand 50 GB/s**, about 3.2×. 900 GB/s is the H100's figure, not the H800's ([fact sheet](fact-sheets/deepseek-v3.md#cluster-and-interconnect)).
- **Where:** Chapter 7 §7.2.
- **Fixed in:** v1.0.1

### E-021 — DeepSeek-V3's FP8 formats, and "unsigned" FP8

- **The book said:** V3 used E4M3 for the forward pass and weight gradients and **E5M2 for activation gradients**, with a per-matmul table said to come from "the report's appendix". It said the master copy of weights was BF16 and the moments FP32. Chapter 8 said FP8 formats "are unsigned (no sign bit)".
- **The source says:** "we adopt the E4M3 format on all tensors for higher precision", explicitly instead of the E4M3/E5M2 hybrid; the fine-grained 1×128 / 128×128 scaling is what makes that feasible. Master weights and gradients are FP32 and the AdamW moments BF16 (§3.3.2–3.3.3). E4M3 and E5M2 both have a sign bit; Chapter 10 itself said so ([fact sheet](fact-sheets/deepseek-v3.md#precision)).
- **Where:** Chapter 1 §1.6; Chapter 5 §5.14.1; Chapter 8 §8.9, §8.15 and takeaway 4; Chapter 10 §10.8.1, §10.8.3, §10.13 and takeaway 3; Exercise 8.8 and its solution; Solution 10.1.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Exercise 8.8 part 1 used to ask "why two different ones?" and the answer explained the hybrid. V3 uses one. The better question, now asked, is why fine-grained scaling lets one format do both jobs.

### E-022 — FP32 optimizer state is "non-negotiable"

- **The book said:** "Optimizer state in FP32 … This is non-negotiable." Solution 8.4 gave V3's high-precision components as "the optimizer state and the master weights. Already FP32".
- **The source says:** V3 stored AdamW's first and second moments in **BF16** "without incurring observable performance degradation", keeping only master weights and gradients in FP32 (§3.3.3) ([fact sheet](fact-sheets/deepseek-v3.md#precision)).
- **Where:** Chapter 8 §8.7; Solution 8.4; Solution 21.1, whose V3 checkpoint size assumed FP32 moments.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Solution 21.1 part 3 still gives 8 TB under the exercise's 12 B/param recipe, and now adds V3's actual **5.4 TB (54 s)** at 8 B/param.

### E-023 — DeepSeek-V3's schedule, batch ramp and "spike handling"

- **The book said:** the schedule was warmup then "cosine decay to 10% of peak over the remainder of training"; the batch ramped "from ~2M tokens to ~19M tokens" (or 2,304 to 14,400 sequences) "in the first 1.3% of training"; there were "roughly 18,000 steps at global batch 8,192"; ε was 1e-8; and "DeepSeek-V3 uses this pattern: skip a window of ~200 batches after a spike, reduce the LR by 50%".
- **The source says:** the LR is held **constant** at 2.2×10⁻⁴ until 10T tokens, then decays by cosine to 2.2×10⁻⁵ over 4.3T, then steps down over the last 500B. The batch ramps from **3,072 to 15,360** sequences over the first **469B** tokens (~3%). That makes ~235K steps (our arithmetic). The report gives no ε and no spike policy; it reports no irrecoverable spikes and no rollbacks (§1, §4.2) ([fact sheet](fact-sheets/deepseek-v3.md#optimizer-and-schedule)).
- **Where:** Chapter 8 §8.13, §8.14, §8.15; Chapter 10 §10.9.
- **Fixed in:** v1.0.1

### E-024 — DeepSeek-V3's data, tokenizer, and training dynamics: quotations and figures not in the report

- **The book said:** Chapter 10 quoted the V3 report five times on its data ("common crawl, code, math, multilingual", "code-related math", "high-quality data", "lessons learned", "based on validation loss curves on a held-out set"). It gave a mix of "approximately 60% English, 30% Chinese, 10% other", described the tokenizer as "bilingual rather than multilingual", with a special-token list "paraphrased from the released config", and described the main run's loss curve (an early "bump"), an MTBF of "several hours", and the team's explanation of the run's stability.
- **The source says:** none of the five quotations appears in the report, and neither do the mix, the loss-curve description, the MTBF or the explanation. The report says the corpus expands multilingual coverage **beyond** English and Chinese, gives no ratios, and publishes no main-run loss curve. The special tokens in the released file are different; the chapter now shows the file itself ([fact sheet](fact-sheets/deepseek-v3.md#data-and-tokenizer)).
- **Where:** Chapter 10 §10.1, §10.2, §10.3, §10.10 and §10.14; Chapter 3 §3.2, §3.3 and §3.5, which repeated the mix and described a V3 PDF pipeline, quality classifier, dedup threshold and mid-run mix changes that the report also does not mention.
- **Fixed in:** v1.0.1

### E-025 — DeepSeek-V3's cost and wall-clock

- **The book said:** the $5.576M covered "the pre-training of V3 … for 2,788K GPU-hours" and "excludes post-training"; the run took "about 2 months … including all failures"; 2,788K GPU-hours implied 50–60% utilization. Chapter 7 priced it at "$1.50/H800-hour". Chapter 10 described Llama-3.1-405B's compute as "~16,000 H100-hours × 30M H100-hours".
- **The source says:** 2,788K is the **full** training: 2,664K pre-training, 119K context extension and **5K post-training**, at an assumed **$2** per GPU-hour. The "less than two months" refers to pre-training, whose GPU-hours match its stated rate of 180K per trillion tokens exactly (Table 1, §1). Llama 3's report gives 3.8×10²⁵ FLOPs on up to 16K H100s and no GPU-hour cost ([fact sheet](fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)).
- **Where:** Chapter 7 §7.15; Chapter 10 §10.1, §10.10 and §10.11; Solutions 10.1 and 10.8.
- **Fixed in:** v1.0.1

### E-026 — DeepSeek-V3's post-training

- **The book said:** "~2M" SFT examples; a reward model on "a smaller variant of V3"; "RLHF (PPO)"; and "Distillation into DeepSeek-R1. The V3 reasoning capabilities are distilled into the DeepSeek-R1 series." Chapter 10 also said later frontier runs from "Llama-3.1 onward" adopted V3's FP8 recipe, and that V3's bias balancer is "now used in Qwen3".
- **The source says:** **1.5M** SFT instances; a model-based reward model trained from the V3 **SFT checkpoints**, alongside rule-based rewards; RL with **GRPO**; and the distillation runs the other way, **from R1 into V3** (§5.1, §5.2, §5.4.1). Llama-3.1 predates V3 and trained in BF16. Qwen3 uses a global-batch balancing loss, not V3's bias ([fact sheet](fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)).
- **Where:** Chapter 10 §10.12 and §10.13.
- **Fixed in:** v1.0.1
