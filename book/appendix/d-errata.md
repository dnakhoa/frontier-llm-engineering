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

### E-011 — A block quote the DeepSeek-V3 report does not contain

- **The book said:** Chapter 7 block-quoted the V3 report: *"… with a total of 900 GB/s NVLink bandwidth per GPU. The 256 nodes are connected via InfiniBand, with each GPU having a 400 Gb/s NIC,"* and called it "the entire physical description of the cluster in the paper".
- **The source says:** no such sentence. §3.1 says only that each node's 8 GPUs are connected by NVLink and NVSwitch, and nodes by InfiniBand. The report's bandwidth figures, in §3.2.2, are **NVLink 160 GB/s and InfiniBand 50 GB/s**, about 3.2×. 900 GB/s is the H100's figure, not the H800's ([fact sheet](fact-sheets/deepseek-v3.md#cluster-and-interconnect)).
- **Where:** Chapter 7 §7.2.
- **Fixed in:** v1.0.1

### E-012 — DeepSeek-V3's FP8 formats, and "unsigned" FP8

- **The book said:** V3 used E4M3 for the forward pass and weight gradients and **E5M2 for activation gradients**, with a per-matmul table said to come from "the report's appendix". It said the master copy of weights was BF16 and the moments FP32. Chapter 8 said FP8 formats "are unsigned (no sign bit)".
- **The source says:** "we adopt the E4M3 format on all tensors for higher precision", explicitly instead of the E4M3/E5M2 hybrid; the fine-grained 1×128 / 128×128 scaling is what makes that feasible. Master weights and gradients are FP32 and the AdamW moments BF16 (§3.3.2–3.3.3). E4M3 and E5M2 both have a sign bit; Chapter 10 itself said so ([fact sheet](fact-sheets/deepseek-v3.md#precision)).
- **Where:** Chapter 1 §1.6; Chapter 5 §5.14.1; Chapter 8 §8.9, §8.15 and takeaway 4; Chapter 10 §10.8.1, §10.8.3, §10.13 and takeaway 3; Exercise 8.8 and its solution; Solution 10.1.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Exercise 8.8 part 1 used to ask "why two different ones?" and the answer explained the hybrid. V3 uses one. The better question, now asked, is why fine-grained scaling lets one format do both jobs.

### E-013 — FP32 optimizer state is "non-negotiable"

- **The book said:** "Optimizer state in FP32 … This is non-negotiable." Solution 8.4 gave V3's high-precision components as "the optimizer state and the master weights. Already FP32".
- **The source says:** V3 stored AdamW's first and second moments in **BF16** "without incurring observable performance degradation", keeping only master weights and gradients in FP32 (§3.3.3) ([fact sheet](fact-sheets/deepseek-v3.md#precision)).
- **Where:** Chapter 8 §8.7; Solution 8.4; Solution 21.1, whose V3 checkpoint size assumed FP32 moments.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Solution 21.1 part 3 still gives 8 TB under the exercise's 12 B/param recipe, and now adds V3's actual **5.4 TB (54 s)** at 8 B/param.

### E-014 — DeepSeek-V3's schedule, batch ramp and "spike handling"

- **The book said:** the schedule was warmup then "cosine decay to 10% of peak over the remainder of training"; the batch ramped "from ~2M tokens to ~19M tokens" (or 2,304 to 14,400 sequences) "in the first 1.3% of training"; there were "roughly 18,000 steps at global batch 8,192"; ε was 1e-8; and "DeepSeek-V3 uses this pattern: skip a window of ~200 batches after a spike, reduce the LR by 50%".
- **The source says:** the LR is held **constant** at 2.2×10⁻⁴ until 10T tokens, then decays by cosine to 2.2×10⁻⁵ over 4.3T, then steps down over the last 500B. The batch ramps from **3,072 to 15,360** sequences over the first **469B** tokens (~3%). That makes ~235K steps (our arithmetic). The report gives no ε and no spike policy; it reports no irrecoverable spikes and no rollbacks (§1, §4.2) ([fact sheet](fact-sheets/deepseek-v3.md#optimizer-and-schedule)).
- **Where:** Chapter 8 §8.13, §8.14, §8.15; Chapter 10 §10.9.
- **Fixed in:** v1.0.1

### E-015 — DeepSeek-V3's data, tokenizer, and training dynamics: quotations and figures not in the report

- **The book said:** Chapter 10 quoted the V3 report five times on its data ("common crawl, code, math, multilingual", "code-related math", "high-quality data", "lessons learned", "based on validation loss curves on a held-out set"). It gave a mix of "approximately 60% English, 30% Chinese, 10% other", described the tokenizer as "bilingual rather than multilingual", with a special-token list "paraphrased from the released config", and described the main run's loss curve (an early "bump"), an MTBF of "several hours", and the team's explanation of the run's stability.
- **The source says:** none of the five quotations appears in the report, and neither do the mix, the loss-curve description, the MTBF or the explanation. The report says the corpus expands multilingual coverage **beyond** English and Chinese, gives no ratios, and publishes no main-run loss curve. The special tokens in the released file are different; the chapter now shows the file itself ([fact sheet](fact-sheets/deepseek-v3.md#data-and-tokenizer)).
- **Where:** Chapter 10 §10.1, §10.2, §10.3, §10.10 and §10.14; Chapter 3 §3.2, §3.3 and §3.5, which repeated the mix and described a V3 PDF pipeline, quality classifier, dedup threshold and mid-run mix changes that the report also does not mention.
- **Fixed in:** v1.0.1

### E-016 — DeepSeek-V3's cost and wall-clock

- **The book said:** the $5.576M covered "the pre-training of V3 … for 2,788K GPU-hours" and "excludes post-training"; the run took "about 2 months … including all failures"; 2,788K GPU-hours implied 50–60% utilization. Chapter 7 priced it at "$1.50/H800-hour". Chapter 10 described Llama-3.1-405B's compute as "~16,000 H100-hours × 30M H100-hours".
- **The source says:** 2,788K is the **full** training: 2,664K pre-training, 119K context extension and **5K post-training**, at an assumed **$2** per GPU-hour. The "less than two months" refers to pre-training, whose GPU-hours match its stated rate of 180K per trillion tokens exactly (Table 1, §1). Llama 3's report gives 3.8×10²⁵ FLOPs on up to 16K H100s and no GPU-hour cost ([fact sheet](fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)).
- **Where:** Chapter 7 §7.15; Chapter 10 §10.1, §10.10 and §10.11; Solutions 10.1 and 10.8.
- **Fixed in:** v1.0.1

### E-017 — DeepSeek-V3's post-training

- **The book said:** "~2M" SFT examples; a reward model on "a smaller variant of V3"; "RLHF (PPO)"; and "Distillation into DeepSeek-R1. The V3 reasoning capabilities are distilled into the DeepSeek-R1 series." Chapter 10 also said later frontier runs from "Llama-3.1 onward" adopted V3's FP8 recipe, and that V3's bias balancer is "now used in Qwen3".
- **The source says:** **1.5M** SFT instances; a model-based reward model trained from the V3 **SFT checkpoints**, alongside rule-based rewards; RL with **GRPO**; and the distillation runs the other way, **from R1 into V3** (§5.1, §5.2, §5.4.1). Llama-3.1 predates V3 and trained in BF16. Qwen3 uses a global-batch balancing loss, not V3's bias ([fact sheet](fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)).
- **Where:** Chapter 10 §10.12 and §10.13.
- **Fixed in:** v1.0.1

### E-018 — lab06's H800 bandwidths, and the layout it recommended for DeepSeek-V3

- **The book said:** `lab06` modelled the 2,048-H800 cluster with NVLink 200 GB/s and InfiniBand 25 GB/s (8×), searched it with FP32 Adam moments and a 4M-token batch, and concluded that "TP and EP belong inside a node".
- **The source says:** DeepSeek report NVLink **160 GB/s** and InfiniBand **50 GB/s** (3.2×) for this cluster, BF16 moments, and a batch of 15,360 × 4K ≈ 63M tokens. They ran EP=64 across 8 nodes ([fact sheet](fact-sheets/deepseek-v3.md#cluster-and-interconnect)). The lab now uses those values. It also prints DeepSeek's real layout next to its own pick: with the all-to-all charged as exposed, the real layout looks ~26× slower, but once the all-to-all is overlapped, which is DualPipe's purpose, it matches the pick. The gap measures what the calculator leaves out.
- **Where:** `lab06_parallelism_memory_model` sections 6, 8, 9, "Things to try" 4 and takeaway 4.
- **Fixed in:** v1.0.1

### E-019 — lab07's headline multiplier

- **The book said:** `lab07` printed that Ethernet versus InfiniBand is "a ~2x throughput difference", and the labs README and Chapter 7 repeated "~2×".
- **The source says:** the lab's own rows give 8.0 s per step on InfiniBand and 28.0 s on 100G Ethernet, a **3.5×** difference. The printed figure is now computed from those rows and asserted, as the style guide requires of every printed claim.
- **Where:** `lab07_collective_bandwidth` section 7; `labs/README.md`; Chapter 7's lab note.
- **Fixed in:** v1.0.1

### E-020 — The Llama 3 data mix, corpus and dedup

- **The book said:** Llama-3's mix was "~50% English web, ~25% code, ~10% multilingual, ~7.5% academic, ~7.5% 'other'"; its 15.6T tokens were "for the 8B model"; it used "aggressive dedup with MinHash and suffix arrays"; and it checked contamination "against 32 benchmarks".
- **The source says:** roughly 50% general knowledge, 25% mathematical and reasoning, 17% code and 8% multilingual, with no academic bucket. 15.6T is the 405B flagship's count; the corpus is "about 15T". Dedup is URL-level, document-level (global MinHash) and line-level; suffix arrays are not mentioned. The contamination table covers 21 benchmarks. See the [fact sheet](fact-sheets/llama-3.md#pre-training-data).
- **Where:** Chapter 3 §3.5 and takeaway 4; Solution 3.6 parts 1 and 3.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** in Exercise 3.6 part 1, the mix is 50 / 25 / 17 / 8 by knowledge domain, not the old 50 / 25 / 10 / 7.5 by source type. Code and reasoning are separate buckets, and math-and-reasoning is the larger one. For part 3, the answer is 21 benchmarks, not 32.

### E-021 — Three things Chapter 3 attributed to the Llama papers

- **The book said:** the Llama-3 paper reports books from "a larger, more carefully curated corpus"; "the Llama-2 paper introduced" training a quality classifier on labels from Llama-2; and the Llama-3 paper discusses "how they re-checked for contamination after every major data update". Solution 3.6 listed the classifier's architecture as undisclosed.
- **The source says:** the Llama 3 report does not describe a books source, and the quoted phrase is not in it. The classifier trained on Llama 2's labels is Llama 3's (fastText and DistilRoberta, [Llama3] §3.1.1). Its contamination section is a single 8-gram overlap analysis, with no re-check after data updates ([Llama3] §5.1.4). See the [fact sheet](fact-sheets/llama-3.md#pre-training-data).
- **Where:** Chapter 3 §3.2.3, §3.3.3 and §3.3.5; Solution 3.6 part 4.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** in Exercise 3.6 part 4, "the classifier's architecture" is no longer a valid answer, because the paper gives it. What the paper withholds is the classifier's weights, labelled training set and threshold.

### E-022 — Llama 3's long-context and annealing recipe

- **The book said:** Llama-3's annealing was "the last few percent of pre-training", on a mix that "downsamples web" and adds "more books, more papers", decayed "to a small floor", with the model "evaluated frequently". Llama-3 was the example of a *two*-stage schedule. The illustrative annealing block's bulk column repeated the wrong mix of E-020. Chapter 5 said Llama-3 reached 128K "via RoPE scaling + fine-tune".
- **The source says:** Llama 3 405B has three stages. Context goes from 8K to 128K in six stages over about 800B tokens of continued pre-training. Annealing is the final 40M tokens (about 0.0003% of 15.6T), at 128K, with the LR annealed linearly to 0, very high-quality sources upsampled, and the base model an average of checkpoints from that phase. The report gives no annealing ratios, and mentions neither books nor downsampling web there. See the [fact sheet](fact-sheets/llama-3.md#long-context-and-annealing).
- **Where:** Chapter 9 §9.1, §9.4 (text and the annealing-mix block), §9.6; Chapter 5 §5.14.2.
- **Fixed in:** v1.0.1

### E-023 — Chapter 7's failure taxonomy contradicted Llama 3 and Chapter 21

- **The book said:** "GPU failures are not the dominant cause of run interruptions", ranking network events first at 40–50%, filesystem events at 20–30%, and GPU hardware at 10–20%. Chapter 21 gave Llama 3's numbers correctly, so the book contradicted itself.
- **The source says:** of 419 unexpected interruptions in Llama 3's 54-day snapshot, GPU issues were the largest category at 58.7%, and about 78% were confirmed or suspected hardware. Software bugs were 12.9% and network switches and cables 8.4%. Filesystems and cooling are not categories at all ([Llama3] §3.3.4, Table 5). Chapter 7 now gives the same breakdown as Chapter 21; see the [fact sheet](fact-sheets/llama-3.md#reliability).
- **Where:** Chapter 7 §7.12.
- **Fixed in:** v1.0.1

### E-024 — Chapter 7's MTBF figures

- **The book said:** "The Llama-3 paper reports an MTBF of 'a few hours'", and a per-GPU MTBF at frontier load is "closer to 8–24 hours". §7.12's table assumed "a per-GPU MTBF of 16 hours". The Llama-3 paper was said to discuss silent data corruption "but does not give numbers", and to mention in-memory optimizer replication on hot-spare nodes.
- **The source says:** the report gives no MTBF and does not contain that phrase. It gives 419 unexpected interruptions in 54 days: one every ~3.1 hours, or about 51,000 GPU-hours per interruption from all causes (our arithmetic). A per-GPU MTBF of 8–24 hours would mean a failure every few seconds on 16K GPUs. It also contradicted §7.12's own table, which works out at about 16,000 hours per GPU, and Chapter 21's 50,000. Silent data corruption caused 6 interruptions (1.4%, Table 5). The report does not describe hot spares. See the [fact sheet](fact-sheets/llama-3.md#reliability).
- **Where:** Chapter 7 §7.7.1, §7.8, §7.11 and §7.12.
- **Fixed in:** v1.0.1

### E-025 — Llama 3's network and collective library

- **The book said:** the Llama-3 cluster sat in Meta's "two-by-two" data-center fabric, on "a customized Arista 7800R3 fabric", and Meta used "a custom NCCL plugin ('nccl-core' plugins)".
- **The source says:** neither quoted name is in the report. It describes a 24K-GPU RoCE cluster, a three-layer Clos network of 3,072-GPU pods built from Arista 7800 and Minipack2 switches, of which Llama 3 used up to 16K GPUs. The collective library is NCCLX, Meta's fork of NCCL ([Llama3] §3.3.1, §3.3.3). See the [fact sheet](fact-sheets/llama-3.md#infrastructure-and-parallelism).
- **Where:** Chapter 7 §7.2, §7.3.3 and §7.5.2.
- **Fixed in:** v1.0.1

### E-026 — Llama 3's training cost

- **The book said:** the Llama-3 405B cost "is reported at 'tens of millions of dollars'", and its cost example took 19.2M H100-hours as consistent with "the reported number". Chapter 10 gave "~16,000 H100-hours × 30M H100-hours" and said "Meta has not published a precise number".
- **The source says:** neither the report nor the model card gives a dollar figure, so there was no reported number to be consistent with. Meta's Llama 3.1 model card does publish the compute: 30.84M H100 GPU-hours for the 405B ([L31-card] @e2e67cd). The Chapter 10 product multiplied two quantities in the same unit. Both chapters now start from 30.84M GPU-hours, with the dollar price labelled as our assumption. See the [fact sheet](fact-sheets/llama-3.md#compute).
- **Where:** Chapter 7 §7.15; Chapter 10 §10.11.
- **Fixed in:** v1.0.1

### E-027 — Llama 3's FSDP settings and context parallelism

- **The book said:** "The Meta team reported a per-block wrapping policy with `BACKWARD_PRE` prefetching". §6.13 gave Llama-3's layout with "sequence length up to 8K, so no CP needed".
- **The source says:** the report gives no wrapping policy or prefetch setting. It uses 4D parallelism in the order [TP, CP, PP, DP], with DP as FSDP. Table 4 has a 128K-context row on the same 16,384 GPUs with CP = 16 and DP = 8 ([Llama3] §3.3.2, Table 4). Context parallelism is part of the Llama 3 layout. See the [fact sheet](fact-sheets/llama-3.md#infrastructure-and-parallelism).
- **Where:** Chapter 6 §6.4, §6.13 and takeaway 8.
- **Fixed in:** v1.0.1

### E-028 — Llama 3's optimizer recipe

- **The book said:** a "Llama 3 (70B)" recipe of AdamW with β₂ = 0.95, ε = 10⁻⁸ and weight decay 0.1, cosine decay "to 10% of peak", a batch of "~16M tokens, held constant", gradient clipping 1.0, and a "critical batch size ~4M tokens … trained at ~4× CBS". Elsewhere: Llama 3 "uses" rollback "for catastrophic spikes", used "a constant global batch size of ~16M tokens" for the 8B, and trained the 8B and 70B on 15.6T tokens.
- **The source says:** the report publishes the recipe for the **405B** only, and says the 8B and 70B use "similar recipes". For the 405B: peak LR 8×10⁻⁵, 8,000 warmup steps, cosine decay to 8×10⁻⁷ (1% of peak), and a batch ramped 4M → 8M → 16M tokens. It reports no β, ε, weight decay, clipping threshold or critical batch size. The run saw "few loss spikes" and needed no interventions. The corpus is "about 15T" tokens; 15.6T is the 405B's count ([Llama3] §1, §3.4.1). The peak LRs for the 8B and 70B (Table 3) were right, and stay. See the [fact sheet](fact-sheets/llama-3.md#training-recipe).
- **Where:** Chapter 8 §8.4 (the LR table), §8.6, §8.10, §8.11, §8.13, §8.14 and §8.15 (the Llama 3 recipe and the four-way comparison table).
- **Fixed in:** v1.0.1

### E-029 — FP8 attributed to Llama 3.1 training

- **The book said:** FP8 training was adopted by "subsequent frontier runs (Llama-3.1 onward, …)", and the glossary said FP8 is used in frontier training by "DeepSeek-V3, Llama-3.1 onwards". Chapter 5 said FP8 was "added in some Llama-3.1 variants".
- **The source says:** the Llama 3 report uses FP8 only as a quantization for 405B inference ([Llama3] §6.2); training MFU is reported against BF16 ([Llama3] Table 4). Llama 3.1 was also released before DeepSeek-V3, so it cannot be a subsequent run. See the [fact sheet](fact-sheets/llama-3.md#training-recipe).
- **Where:** Chapter 10 §10.13; Chapter 5 §5.14.2; the glossary entry for FP8.
- **Fixed in:** v1.0.1

### E-030 — The SwiGLU width rule and Llama 3

- **The book said:** the rule $d_\text{ff} = \frac{8}{3} d_\text{model}$ "gives $d_\text{ff} \approx 28672$ (the published value)" for Llama-3-70B, and §5.14.2 wrote $28672$ as $8/3 \cdot 8192$, rounded.
- **The source says:** 8/3 × 8,192 is about 21,845. Llama 3's published FFN width is 28,672 (Table 3), which is 3.5 × 8,192: wider than the rule, at every size. The arithmetic is ours; the values are in the [fact sheet](fact-sheets/llama-3.md#architecture).
- **Where:** Chapter 5 §5.7 and §5.14.2.
- **Fixed in:** v1.0.1

### E-031 — Llama 3's tokenizer and code

- **The book said:** "Llama-3 specifically increased the number of merges dedicated to whitespace tokens", citing the Llama 3 report. It gave "a real tokenizer behavior" for a line of Python as 16 tokens, with a 4-space indent token and a bare `'return'`. It said Meta grew the vocabulary "citing multilingual and code improvements".
- **The source says:** the report says nothing about whitespace, and gives English compression (3.17 → 3.94 characters per token) and non-English support as its reasons, not code ([Llama3] §3.2). In the published `tokenizer.json`, every whitespace-run token has an ID below 100,000, so none is among the 28K tokens Meta added. The published tokenizer encodes the example line as 17 tokens: `'   '` then `' return'`, and `' '` then `'0'`. See the [tokenizer fact sheet](fact-sheets/tokenizers.md#llama-3).
- **Where:** Chapter 4 §4.7 and §4.10, and the chapter's reference note for [5].
- **Fixed in:** v1.0.1

### E-032 — Qwen3's expert width and Exercise 5.4

- **The book said:** each of Qwen3-235B's experts has "$d_\text{ff} = 12288$". Solution 5.4 built on it: "151M per expert", "**1,209M**" active expert parameters per layer. Solution 5.8's config table repeated "12,288 per expert".
- **The source says:** the expert width is `moe_intermediate_size` = **1,536**. 12,288 is the config's `intermediate_size`, which sizes a dense FFN layer, and this model has none: all 94 layers are MoE. Only 1,536 reproduces the report's 22B active parameters; 12,288 would give about 121B. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#architecture-qwen3-235b-a22b).
- **Where:** Chapter 5 §5.14.3; Exercise 5.4; Solution 5.4 parts 1–3; Solution 5.8 part 1.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** the exercise's conclusion reverses. *Old conclusion:* Qwen3 has the largest experts of the three (151M each, 1,209M active per layer, three times V3's), so it sits with Mixtral on the coarse side. *New conclusion:* each Qwen3 expert is 18.9M and 151M are active per layer, the smallest of the three. Its expert width is 0.375 of the model width, close to V3's 0.29 and far from Mixtral's 3.5, so Qwen3 is a fine-grained MoE. Its eight active experts add up to one ordinary dense FFN ($8 \times 1536 = 12{,}288 = 3d$), which is what fine-grained segmentation is. *Why:* the exercise read the wrong field of the config. Exercise 5.4 now makes choosing the field part of the problem: you check each candidate against the published 22B.

### E-033 — Qwen3's MoE design: shared experts, balancing and dense layers

- **The book said:** the Qwen3 MoE models use "fine-grained experts and shared experts"; they use the "standard auxiliary loss with low weight", with bias-based balancing "on their roadmap". Chapter 10 said V3's auxiliary-loss-free balancing "is now used in Qwen3", and that V3's first-K-dense pattern "is now used in Qwen3".
- **The source says:** "the Qwen3-MoE design excludes shared experts", and it adopts "the global-batch load balancing loss" ([Qwen3] §2). The report says nothing of a roadmap. The published config has no dense layers: `mlp_only_layers` is empty and `decoder_sparse_step` is 1. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#architecture-qwen3-235b-a22b).
- **Where:** Chapter 5 §5.11 and §5.14.3; Chapter 10 §10.4.3 and §10.13.
- **Fixed in:** v1.0.1

### E-034 — Qwen3's context length and RoPE recipe

- **The book said:** Qwen3 was "trained at 32K directly, no extension needed", "ships a 128K variant directly trained at that context length", and uses "RoPE with dual base". Chapter 9 said Qwen3 "describes a 32K → 128K long-context extension phase", was "trained at 8K context", followed a two-stage "32K → 128K" schedule, and extended context with YaRN like Llama-3. Solution 9.7 repeated "(32K → 128K)".
- **The source says:** three pre-training stages. S1 is over 30T tokens at 4,096; S2 is about 5T reasoning-heavy tokens at 4,096, with accelerated learning-rate decay; S3 is hundreds of billions of tokens at 32,768. During S3 the RoPE base is raised from 10,000 to 1,000,000 with ABF. 128K comes from YaRN and Dual Chunk Attention "during inference" ([Qwen3] §3.2). The released config has a single `rope_theta`. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#pre-training-stages).
- **Where:** Chapter 5 §5.13.3 and §5.14.3; Chapter 9 §9.1, §9.2, §9.6, §9.10 and its reference note; Solution 9.7 parts 1 and 2.
- **Fixed in:** v1.0.1
- **If you worked this before v1.0.1:** Solution 9.7 part 2 said all three reports put a small single-digit share of tokens into their final phases. Qwen3 does not. Its long-context stage is about 1% of ~36T, but its reasoning stage, the one with the faster decay and the reweighted mix, is about 5T, roughly 14%.

### E-035 — Qwen3's pre-training corpus

- **The book said:** Qwen3's "~36T tokens (across all stages including pre-training, mid-training, and post-training data)", with a mix "closer to 50/50 English/Chinese".
- **The source says:** the 36T tokens are the pre-training corpus, "covering up to 119 languages and dialects" ([Qwen3] §1, §3.1). The report gives no English/Chinese split. See the [Qwen3 fact sheet](fact-sheets/qwen3.md#pre-training-data).
- **Where:** Chapter 3 §3.2.1, §3.2.5, §3.5 and its reference note.
- **Fixed in:** v1.0.1

### E-036 — Mixtral's attention, context and parameter ratio

- **The book said:** Mixtral uses "Standard MHA … *not* GQA"; its context is "32K (extended from initial 8K training via RoPE scaling)"; and, under "From the Mixtral paper", it uses a "Standard Switch-style auxiliary loss". §5.8 said its "total parameters are $8 \times$ the active parameters".
- **The source says:** Table 1 gives `n_heads` 32 and `n_kv_heads` 8, so Mixtral uses GQA. The model "was trained with a context size of 32k tokens", and no shorter first stage is described ([Mixtral] Abstract, §2). The paper does not describe its load-balancing loss. With 8 experts and top-2, the expert parameters are 4× the active ones; overall the model is 47B total against 13B active. See the [Mixtral fact sheet](fact-sheets/mixtral.md).
- **Where:** Chapter 5 §5.8 and §5.14.4.
- **Fixed in:** v1.0.1
