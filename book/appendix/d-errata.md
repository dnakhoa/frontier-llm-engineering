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

### E-050 — Qwen3's expert width and Exercise 5.4

- **The book said:** each of Qwen3-235B's experts has "$d_\text{ff} = 12288$". Solution 5.4 built on it: "151M per expert", "**1,209M**" active expert parameters per layer. Solution 5.8's config table repeated "12,288 per expert".
- **The source says:** the expert width is `moe_intermediate_size` = **1,536**. 12,288 is the config's `intermediate_size`, which sizes a dense FFN layer, and this model has none: all 94 layers are MoE. Only 1,536 reproduces the report's 22B active parameters; 12,288 would give about 121B. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#architecture-qwen3-235b-a22b).
- **Where:** Chapter 5 §5.14.3; Exercise 5.4; Solution 5.4 parts 1–3; Solution 5.8 part 1.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** the exercise's conclusion reverses. *Old conclusion:* Qwen3 has the largest experts of the three (151M each, 1,209M active per layer, three times V3's), so it sits with Mixtral on the coarse side. *New conclusion:* each Qwen3 expert is 18.9M and 151M are active per layer, the smallest of the three. Its expert width is 0.375 of the model width, close to V3's 0.29 and far from Mixtral's 3.5, so Qwen3 is a fine-grained MoE. Its eight active experts add up to one ordinary dense FFN ($8 \times 1536 = 12{,}288 = 3d$), which is what fine-grained segmentation is. *Why:* the exercise read the wrong field of the config. Exercise 5.4 now makes choosing the field part of the problem: you check each candidate against the published 22B.

### E-051 — Qwen3's MoE design: shared experts, balancing and dense layers

- **The book said:** the Qwen3 MoE models use "fine-grained experts and shared experts"; they use the "standard auxiliary loss with low weight", with bias-based balancing "on their roadmap". Chapter 10 said V3's auxiliary-loss-free balancing "is now used in Qwen3", and that V3's first-K-dense pattern "is now used in Qwen3".
- **The source says:** "the Qwen3-MoE design excludes shared experts", and it adopts "the global-batch load balancing loss" ([Qwen3] §2). The report says nothing of a roadmap. The published config has no dense layers: `mlp_only_layers` is empty and `decoder_sparse_step` is 1. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#architecture-qwen3-235b-a22b).
- **Where:** Chapter 5 §5.11 and §5.14.3; Chapter 10 §10.4.3 and §10.13.
- **Fixed in:** v1.0.1

### E-052 — Qwen3's context length and RoPE recipe

- **The book said:** Qwen3 was "trained at 32K directly, no extension needed", "ships a 128K variant directly trained at that context length", and uses "RoPE with dual base". Chapter 9 said Qwen3 "describes a 32K → 128K long-context extension phase", was "trained at 8K context", followed a two-stage "32K → 128K" schedule, and extended context with YaRN like Llama-3. Solution 9.7 repeated "(32K → 128K)".
- **The source says:** three pre-training stages. S1 is over 30T tokens at 4,096; S2 is about 5T reasoning-heavy tokens at 4,096, with accelerated learning-rate decay; S3 is hundreds of billions of tokens at 32,768. During S3 the RoPE base is raised from 10,000 to 1,000,000 with ABF. 128K comes from YaRN and Dual Chunk Attention "during inference" ([Qwen3] §3.2). The released config has a single `rope_theta`. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#pre-training-stages).
- **Where:** Chapter 5 §5.13.3 and §5.14.3; Chapter 9 §9.1, §9.2, §9.6, §9.10 and its reference note; Solution 9.7 parts 1 and 2.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Solution 9.7 part 2 said all three reports put a small single-digit share of tokens into their final phases. Qwen3 does not. Its long-context stage is about 1% of ~36T, but its reasoning stage, the one with the faster decay and the reweighted mix, is about 5T, roughly 14%.

### E-053 — Qwen3's pre-training corpus

- **The book said:** Qwen3's "~36T tokens (across all stages including pre-training, mid-training, and post-training data)", with a mix "closer to 50/50 English/Chinese".
- **The source says:** the 36T tokens are the pre-training corpus, "covering up to 119 languages and dialects" ([Qwen3] §1, §3.1). The report gives no English/Chinese split. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#pre-training-data).
- **Where:** Chapter 3 §3.2.1, §3.2.5, §3.5 and its reference note.
- **Fixed in:** v1.0.1

### E-054 — Mixtral's attention, context and parameter ratio

- **The book said:** Mixtral uses "Standard MHA … *not* GQA"; its context is "32K (extended from initial 8K training via RoPE scaling)"; and, under "From the Mixtral paper", it uses a "Standard Switch-style auxiliary loss". §5.8 said its "total parameters are $8 \times$ the active parameters".
- **The source says:** Table 1 gives `n_heads` 32 and `n_kv_heads` 8, so Mixtral uses GQA. The model "was trained with a context size of 32k tokens", and no shorter first stage is described ([Mixtral] Abstract, §2). The paper does not describe its load-balancing loss. With 8 experts and top-2, the expert parameters are 4× the active ones; overall the model is 47B total against 13B active. See the [Mixtral fact sheet](fact-sheets/mixtral.md).
- **Where:** Chapter 5 §5.8 and §5.14.4.
- **Fixed in:** v1.0.1
