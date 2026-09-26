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

### E-040 — The Llama 3 data mix, corpus and dedup

- **The book said:** Llama-3's mix was "~50% English web, ~25% code, ~10% multilingual, ~7.5% academic, ~7.5% 'other'"; its 15.6T tokens were "for the 8B model"; it used "aggressive dedup with MinHash and suffix arrays"; and it checked contamination "against 32 benchmarks".
- **The source says:** roughly 50% general knowledge, 25% mathematical and reasoning, 17% code and 8% multilingual, with no academic bucket. 15.6T is the 405B flagship's count; the corpus is "about 15T". Dedup is URL-level, document-level (global MinHash) and line-level; suffix arrays are not mentioned. The contamination table covers 21 benchmarks. See the [fact sheet](fact-sheets/llama-3.md#pre-training-data).
- **Where:** Chapter 3 §3.5 and takeaway 4; Solution 3.6 parts 1 and 3.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** in Exercise 3.6 part 1, the mix is 50 / 25 / 17 / 8 by knowledge domain, not the old 50 / 25 / 10 / 7.5 by source type. Code and reasoning are separate buckets, and math-and-reasoning is the larger one. For part 3, the answer is 21 benchmarks, not 32.

### E-041 — Three things Chapter 3 attributed to the Llama papers

- **The book said:** the Llama-3 paper reports books from "a larger, more carefully curated corpus"; "the Llama-2 paper introduced" training a quality classifier on labels from Llama-2; and the Llama-3 paper discusses "how they re-checked for contamination after every major data update". Solution 3.6 listed the classifier's architecture as undisclosed.
- **The source says:** the Llama 3 report does not describe a books source, and the quoted phrase is not in it. The classifier trained on Llama 2's labels is Llama 3's (fastText and DistilRoberta, [Llama3] §3.1.1). Its contamination section is a single 8-gram overlap analysis, with no re-check after data updates ([Llama3] §5.1.4). See the [fact sheet](fact-sheets/llama-3.md#pre-training-data).
- **Where:** Chapter 3 §3.2.3, §3.3.3 and §3.3.5; Solution 3.6 part 4.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** in Exercise 3.6 part 4, "the classifier's architecture" is no longer a valid answer, because the paper gives it. What the paper withholds is the classifier's weights, labelled training set and threshold.

### E-042 — Llama 3's long-context and annealing recipe

- **The book said:** Llama-3's annealing was "the last few percent of pre-training", on a mix that "downsamples web" and adds "more books, more papers", decayed "to a small floor", with the model "evaluated frequently". Llama-3 was the example of a *two*-stage schedule. The illustrative annealing block's bulk column repeated the wrong mix of E-040. Chapter 5 said Llama-3 reached 128K "via RoPE scaling + fine-tune".
- **The source says:** Llama 3 405B has three stages. Context goes from 8K to 128K in six stages over about 800B tokens of continued pre-training. Annealing is the final 40M tokens (about 0.0003% of 15.6T), at 128K, with the LR annealed linearly to 0, very high-quality sources upsampled, and the base model an average of checkpoints from that phase. The report gives no annealing ratios, and mentions neither books nor downsampling web there. See the [fact sheet](fact-sheets/llama-3.md#long-context-and-annealing).
- **Where:** Chapter 9 §9.1, §9.4 (text and the annealing-mix block), §9.6; Chapter 5 §5.14.2.
- **Fixed in:** v1.0.1

### E-043 — Chapter 7's failure taxonomy contradicted Llama 3 and Chapter 21

- **The book said:** "GPU failures are not the dominant cause of run interruptions", ranking network events first at 40–50%, filesystem events at 20–30%, and GPU hardware at 10–20%. Chapter 21 gave Llama 3's numbers correctly, so the book contradicted itself.
- **The source says:** of 419 unexpected interruptions in Llama 3's 54-day snapshot, GPU issues were the largest category at 58.7%, and about 78% were confirmed or suspected hardware. Software bugs were 12.9% and network switches and cables 8.4%. Filesystems and cooling are not categories at all ([Llama3] §3.3.4, Table 5). Chapter 7 now gives the same breakdown as Chapter 21; see the [fact sheet](fact-sheets/llama-3.md#reliability).
- **Where:** Chapter 7 §7.12.
- **Fixed in:** v1.0.1

### E-044 — Chapter 7's MTBF figures

- **The book said:** "The Llama-3 paper reports an MTBF of 'a few hours'", and a per-GPU MTBF at frontier load is "closer to 8–24 hours". §7.12's table assumed "a per-GPU MTBF of 16 hours". The Llama-3 paper was said to discuss silent data corruption "but does not give numbers", and to mention in-memory optimizer replication on hot-spare nodes.
- **The source says:** the report gives no MTBF and does not contain that phrase. It gives 419 unexpected interruptions in 54 days: one every ~3.1 hours, or about 51,000 GPU-hours per interruption from all causes (our arithmetic). A per-GPU MTBF of 8–24 hours would mean a failure every few seconds on 16K GPUs. It also contradicted §7.12's own table, which works out at about 16,000 hours per GPU, and Chapter 21's 50,000. Silent data corruption caused 6 interruptions (1.4%, Table 5). The report does not describe hot spares. See the [fact sheet](fact-sheets/llama-3.md#reliability).
- **Where:** Chapter 7 §7.7.1, §7.8, §7.11 and §7.12.
- **Fixed in:** v1.0.1

### E-045 — Llama 3's network and collective library

- **The book said:** the Llama-3 cluster sat in Meta's "two-by-two" data-center fabric, on "a customized Arista 7800R3 fabric", and Meta used "a custom NCCL plugin ('nccl-core' plugins)".
- **The source says:** neither quoted name is in the report. It describes a 24K-GPU RoCE cluster, a three-layer Clos network of 3,072-GPU pods built from Arista 7800 and Minipack2 switches, of which Llama 3 used up to 16K GPUs. The collective library is NCCLX, Meta's fork of NCCL ([Llama3] §3.3.1, §3.3.3). See the [fact sheet](fact-sheets/llama-3.md#infrastructure-and-parallelism).
- **Where:** Chapter 7 §7.2, §7.3.3 and §7.5.2.
- **Fixed in:** v1.0.1

### E-046 — Llama 3's training cost

- **The book said:** the Llama-3 405B cost "is reported at 'tens of millions of dollars'", and its cost example took 19.2M H100-hours as consistent with "the reported number". Chapter 10 gave "~16,000 H100-hours × 30M H100-hours" and said "Meta has not published a precise number".
- **The source says:** neither the report nor the model card gives a dollar figure, so there was no reported number to be consistent with. Meta's Llama 3.1 model card does publish the compute: 30.84M H100 GPU-hours for the 405B ([L31-card] @e2e67cd). The Chapter 10 product multiplied two quantities in the same unit. Both chapters now start from 30.84M GPU-hours, with the dollar price labelled as our assumption. See the [fact sheet](fact-sheets/llama-3.md#compute).
- **Where:** Chapter 7 §7.15; Chapter 10 §10.11.
- **Fixed in:** v1.0.1

### E-047 — Llama 3's FSDP settings and context parallelism

- **The book said:** "The Meta team reported a per-block wrapping policy with `BACKWARD_PRE` prefetching". §6.13 gave Llama-3's layout with "sequence length up to 8K, so no CP needed".
- **The source says:** the report gives no wrapping policy or prefetch setting. It uses 4D parallelism in the order [TP, CP, PP, DP], with DP as FSDP. Table 4 has a 128K-context row on the same 16,384 GPUs with CP = 16 and DP = 8 ([Llama3] §3.3.2, Table 4). Context parallelism is part of the Llama 3 layout. See the [fact sheet](fact-sheets/llama-3.md#infrastructure-and-parallelism).
- **Where:** Chapter 6 §6.4, §6.13 and takeaway 8.
- **Fixed in:** v1.0.1

### E-048 — Llama 3's optimizer recipe

- **The book said:** a "Llama 3 (70B)" recipe of AdamW with β₂ = 0.95, ε = 10⁻⁸ and weight decay 0.1, cosine decay "to 10% of peak", a batch of "~16M tokens, held constant", gradient clipping 1.0, and a "critical batch size ~4M tokens … trained at ~4× CBS". Elsewhere: Llama 3 "uses" rollback "for catastrophic spikes", used "a constant global batch size of ~16M tokens" for the 8B, and trained the 8B and 70B on 15.6T tokens.
- **The source says:** the report publishes the recipe for the **405B** only, and says the 8B and 70B use "similar recipes". For the 405B: peak LR 8×10⁻⁵, 8,000 warmup steps, cosine decay to 8×10⁻⁷ (1% of peak), and a batch ramped 4M → 8M → 16M tokens. It reports no β, ε, weight decay, clipping threshold or critical batch size. The run saw "few loss spikes" and needed no interventions. The corpus is "about 15T" tokens; 15.6T is the 405B's count ([Llama3] §1, §3.4.1). The peak LRs for the 8B and 70B (Table 3) were right, and stay. See the [fact sheet](fact-sheets/llama-3.md#training-recipe).
- **Where:** Chapter 8 §8.4 (the LR table), §8.6, §8.10, §8.11, §8.13, §8.14 and §8.15 (the Llama 3 recipe and the four-way comparison table).
- **Fixed in:** v1.0.1

### E-049 — FP8 attributed to Llama 3.1 training

- **The book said:** FP8 training was adopted by "subsequent frontier runs (Llama-3.1 onward, …)", and the glossary said FP8 is used in frontier training by "DeepSeek-V3, Llama-3.1 onwards". Chapter 5 said FP8 was "added in some Llama-3.1 variants".
- **The source says:** the Llama 3 report uses FP8 only as a quantization for 405B inference ([Llama3] §6.2); training MFU is reported against BF16 ([Llama3] Table 4). Llama 3.1 was also released before DeepSeek-V3, so it cannot be a subsequent run. See the [fact sheet](fact-sheets/llama-3.md#training-recipe).
- **Where:** Chapter 10 §10.13; Chapter 5 §5.14.2; the glossary entry for FP8.
- **Fixed in:** v1.0.1

### E-050 — The SwiGLU width rule and Llama 3

- **The book said:** the rule $d_\text{ff} = \frac{8}{3} d_\text{model}$ "gives $d_\text{ff} \approx 28672$ (the published value)" for Llama-3-70B, and §5.14.2 wrote $28672$ as $8/3 \cdot 8192$, rounded.
- **The source says:** 8/3 × 8,192 is about 21,845. Llama 3's published FFN width is 28,672 (Table 3), which is 3.5 × 8,192: wider than the rule, at every size. The arithmetic is ours; the values are in the [fact sheet](fact-sheets/llama-3.md#architecture).
- **Where:** Chapter 5 §5.7 and §5.14.2.
- **Fixed in:** v1.0.1

### E-051 — Llama 3's tokenizer and code

- **The book said:** "Llama-3 specifically increased the number of merges dedicated to whitespace tokens", citing the Llama 3 report. It gave "a real tokenizer behavior" for a line of Python as 16 tokens, with a 4-space indent token and a bare `'return'`. It said Meta grew the vocabulary "citing multilingual and code improvements".
- **The source says:** the report says nothing about whitespace, and gives English compression (3.17 → 3.94 characters per token) and non-English support as its reasons, not code ([Llama3] §3.2). In the published `tokenizer.json`, every whitespace-run token has an ID below 100,000, so none is among the 28K tokens Meta added. The published tokenizer encodes the example line as 17 tokens: `'   '` then `' return'`, and `' '` then `'0'`. See the [tokenizer fact sheet](fact-sheets/tokenizers.md#llama-3).
- **Where:** Chapter 4 §4.7 and §4.10, and the chapter's reference note for [5].
- **Fixed in:** v1.0.1
