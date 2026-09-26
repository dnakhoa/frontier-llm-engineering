# Part F — Tier 0 propagation into exercises/, exercises/solutions/ and labs/*.py

Read-only audit, 2026-09-26. Scope: places where a Tier 0 error (AUDIT.md), or another claim that contradicts the correct values in part reports A–E, is repeated or built on. General staleness is out of scope. `.ipynb` files were skipped.

**Method.**
- Grepped all 26 problem sets, all 26 solution sets, `prerequisites.md` and every `labs/*.py` for the Tier 0 facts: layout, layer count, NVLink, FP32 moments, E5M2, LR schedule, MLA/KV heads, dense layers, PPO/GRPO, distillation direction, tokenizer fields, Llama-3 mix, annealing, failure taxonomy, and the Qwen3, Mixtral and V3 configs.
- Read ch01, ch05–ch10 and ch21, and the solutions for ch03, ch04 and ch11, in full around every hit.
- Ran `lab06` and `lab07`, and re-ran `lab06` in a scratch copy with the V3-reported H800 bandwidths.
- Checked these primary sources directly because the part reports left them open:
  - `Qwen/Qwen3-235B-A22B/config.json`: `moe_intermediate_size` 1536, `hidden_size` 4096, 94 layers, 4 KV heads.
  - `deepseek-ai/DeepSeek-V3/config.json`: 61 layers, `first_k_dense_replace` 3, `kv_lora_rank` 512, `qk_rope_head_dim` 64, dense `intermediate_size` 18432.
  - The DeepSeek-V3 HF README: "685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights".

**Impact legend.**
- **text-only**: prose is wrong but no computed answer changes.
- **changes-answer**: a solution's arithmetic or its reference answer changes.
- **changes-lab-result**: a lab's constant or printed output changes.
- **exercise-premise-wrong**: the exercise gives the reader a wrong fact to derive from or confirm.

**Counts (43 findings):**

| Impact | Count |
|---|---|
| exercise-premise-wrong | 6 |
| changes-answer | 16 |
| changes-lab-result | 2 |
| text-only | 19 |

**Lab headline results.**
- `lab07`'s "~2× throughput" headline does **not** depend on the NVLink figure. That row compares a flat ring over IB (40 GB/s) with 100G Ethernet (10 GB/s); NVLink enters only the hierarchical row. `lab07` also uses 450 GB/s, which is an H100 number, not 900, and it makes no H800 claim.
- No lab's headline conclusion changes.
- `lab06`'s DeepSeek-V3 search does change: its top recommendation moves from EP=8 to EP=4 once the H800 constants are corrected. The lab prints no conclusion that depends on this.

**Good news worth keeping:**
- `solutions/ch06.md:342` and `solutions/ch10.md:15` state the **correct** V3 layout (PP16, EP64 across 8 nodes, ZeRO-1, no TP). The chapters are wrong and the solutions are right.
- `solutions/ch07.md:239` states the Llama 3 failure taxonomy correctly (78% hardware, GPU the largest category). It contradicts ch07 correctly.
- `solutions/ch03.md:178` gives Llama 3's dedup correctly (URL, document MinHash, line-level; no suffix arrays).
- `solutions/ch10.md:21` gives V3's context path correctly (4K, then 32K, then 128K via YaRN).
- `lab06`'s V3 model config is correct (61 layers, 3 dense layers, MLA 512+64, vocabulary 129,280).
- `lab05_attention_variants` prints the true 576-value MLA row.

---

## exercises/ch01.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch01.md:23 | "Same run: 14.8T tokens, 2,788K H800 GPU-hours" (used to derive tokens/s and MFU) | 2,788K is the total: pre-training 2,664K + context extension 119K + post-training 5K. The 14.8T tokens belong to the 2,664K pre-training hours. (V3 Table 1; part-C ch07 #7, part-B ch10 "Hours") | exercise-premise-wrong |

## exercises/ch05.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch05.md:32 | "MLA \| latent $d_c = 512$, so $2 d_c$ = 1024 numbers per layer per token" | MLA caches the single joint latent (512) plus the decoupled RoPE key (64): **576** values per layer per token. (V3 §2.1; part-B ch05 #6; V3 config `kv_lora_rank` 512, `qk_rope_head_dim` 64) | exercise-premise-wrong |
| ch05.md:43 | "DeepSeek-V3: 671B total, 37B active, **60 layers**…" (no mention of dense layers) | **61** layers, of which the first **3 are dense**, so 58 MoE layers. (V3 §4.2, "all FFNs except for the first three layers"; config `first_k_dense_replace: 3`) | exercise-premise-wrong |
| ch05.md:56 | "Qwen3-235B uses 128 experts of $d_{\text{ff}} = 12288$" | Expert FFN width is **1,536** (`moe_intermediate_size`). 12288 is the unused dense `intermediate_size` field. So Qwen3 is fine-grained too, not "coarse". (Qwen3-235B-A22B config.json; part-B unverified lead, now confirmed) | exercise-premise-wrong |

## exercises/ch06.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch06.md:74–80 | A V3-shaped MoE (256 experts, top-8, d = 7168) with "8-way expert parallelism", "Over 60 layers", then "what does that imply about where EP ranks should be placed?" | This is a hypothetical, but it copies the wrong V3 layout (EP8, 60 layers) and steers the reader to "EP inside a node". V3 runs EP64 across 8 nodes and has 61 layers (58 MoE). (V3 §3.2) | text-only |

## exercises/ch08.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch08.md:96 | 8.8.1 "Which formats do they use where, and **why two different ones?**" | V3 "adopt[s] the E4M3 format on all tensors", explicitly instead of the E4M3/E5M2 hybrid. There is one format, not two. (V3 §3.3.2; part-B ch05 #6, ch10 "FP8 formats") | exercise-premise-wrong |

## exercises/ch10.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch10.md:62 | "V3 uses one extra prediction **head** with loss weight 0.3" | MTP depth 1, but the module is a full sequential transformer block (14B parameters with its MoE FFN), not a head. λ = 0.3 for the first 10T tokens, then **0.1** for the remaining 4.8T. (V3 §2.2, §4.2; HF README; part-B ch05 #8) | exercise-premise-wrong |

## exercises/solutions/ch01.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch01.md:13, 40–56 | Divides 14.8T tokens by all 2,788K hours: 56.7 days, 3.02M tok/s, 1,475 tok/s/GPU, MFU 33% | Pre-training is 2,664K hours (V3 Table 1). Recomputed: **54.2 days, 3.16M tok/s, 1,543 tok/s/GPU, MFU ≈ 34.7%** (BF16 peak). The qualitative lesson stands. | changes-answer (small, ~5%) |
| ch01.md:76–83 | Full-model static state at 16 B/param gives 10.7 TB (6.5% of HBM) | The exercise stipulates 16 B, so the arithmetic is self-consistent. But V3 itself kept AdamW moments in **BF16** (V3 §3.3.3), 4 B/param less, about 8.1 TB. Worth a footnote, since this is V3's own recipe. | text-only |

## exercises/solutions/ch03.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch03.md:176 | Llama 3 mix: "Roughly 50% general web/knowledge, **25% code and reasoning, ~10% multilingual, ~7.5% academic**" | "roughly 50% … general knowledge, **25% mathematical and reasoning**, **17% code**, and **8% multilingual**". There is no academic bucket. (Llama 3 §3.1.2; part-A ch03 #1) | changes-answer |
| ch03.md:184 | "The quality classifier itself — its training data, its architecture … the single most load-bearing undisclosed component" | Llama 3 describes its classifiers: fastText and RoBERTa-based (DistilRoberta) classifiers trained on Llama 2 annotations. The weights and thresholds are withheld; the architecture and training-label source are not. (Llama 3 §3.1.1; part-A ch03 #3) | text-only |

## exercises/solutions/ch04.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch04.md:217 | "Qwen's recent models use around 151,646" | Qwen3 report: **151,669** (151,643 BPE + 26 added). The embedding `vocab_size` is **151,936**. (Qwen3 §2; part-A ch04 #3) | text-only |

## exercises/solutions/ch05.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch05.md:40 | "The small excess suggests the real model **ties or partially shares embeddings**" | Llama-3 is **untied** (`tie_word_embeddings: false`), as the exercise itself states. The published count is about 70.6B, so the excess is just rounding of "70B". (part-A ch04 #4; Llama-3 config) | text-only |
| ch05.md:54, 63, 72, 83, 93 | MLA "stores $2 d_c$", giving **164 KB/token**, **21 GB** at 128K, **7** sequences at 8K in 10 GB, "164 KB versus 328 KB" | With 576 values: 80 × 576 × 2 B = **92 KB/token**, **12.1 GB** at 128K, **13** sequences at 8K. MLA is 3.6× smaller than GQA-8, not 2×. (V3 §2.1; part-B ch05 #6). `solutions/ch05.md:382` later concedes the RoPE part but keeps the table. | changes-answer |
| ch05.md:105–117 | Routed experts "$256 \times 44.0\text{M} \times 60 = 676$B … essentially 100%"; active "$9 \times 44.0\text{M} \times 60 = 23.8$B" | 58 MoE layers: 256 × 44.04M × 58 = **654B (≈97%)**, leaving ~17B for attention, embeddings, shared and dense FFNs. Active routed+shared = **23.0B**, plus 3 dense FFNs (3 × 7168 × 18432 × 3 = 1.19B). (V3 §4.2; config) | changes-answer |
| ch05.md:148, 155–165 | Qwen3-235B: "12,288 … 151M per expert … **1,209M** active per layer"; the fine-vs-coarse argument uses Qwen3 and Mixtral as the coarse side | 3 × 4096 × 1536 = **18.9M per expert**; 8 × 18.9 = **151M active per layer**. That is the smallest of the three, not the largest. Qwen3 is a fine-grained MoE. (Qwen3-235B-A22B config.json) | changes-answer |
| ch05.md:279 | Aux-free bias "achieves balance **without touching the loss at all** … DeepSeek's report is explicit" | V3 also keeps a "complementary sequence-wise balance loss" with a tiny weight. (V3 §2.1.2; part-B ch05 #9) | text-only |
| ch05.md:295–296 | Config diff table: "$d_{\text{ff}}$ … **12,288 per expert**"; vocabulary "~151,646" | 1,536 per expert (config). Vocabulary is 151,936 in the config (151,669 in the report). | changes-answer |

## exercises/solutions/ch06.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch06.md:30 | "DeepSeek-V3 … 10,736 GB … **134** H100s" at the 16 B FP32-moment recipe | This applies the exercise's stated recipe, so it is self-consistent. V3's own recipe used BF16 moments (§3.3.3): about 8,052 GB, about 101 GPUs. Footnote it. | text-only |
| ch06.md:220–226 | "Over 60 layers: $60 \times 0.94$ GB ≈ 56 GB … ~900 GB per step" | For the V3 shape: 58 MoE layers, so **~54.5 GB** per micro-batch and **~870 GB** per step at 16 micro-batches. The "all-to-all dominates ~3×" conclusion holds. | changes-answer (small) |
| ch06.md:230–232, 260, 297 | "expert-parallel ranks must be … within a node over NVLink"; "EP = 8. Within a node … the most important placement decision"; "EP must stay strictly intra-node"; DualPipe exists because "even on NVLink" the all-to-all is large | The book's own anchor contradicts this. V3 runs **EP64 across 8 nodes** over IB with node-limited routing (≤4 nodes/token); DualPipe exists to hide *cross-node* all-to-all. (V3 §3.2, §3.2.2; part-C ch06 #1) | text-only |
| ch06.md:348 | "two all-reduces per layer … for **60 layers**. That is **120** blocking collectives" | 61 layers, so 122. (V3 §4.2) | text-only |

## exercises/solutions/ch07.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch07.md:51 | "Confine EP within a node … This is why **EP degree is usually 8 (one node)**" | V3, the book's anchor, uses EP64 across 8 nodes. (V3 §3.2; part-C ch06 #1). The H100 bandwidth numbers at `:11–14` (≈450 vs 40 GB/s, 11×) are fine for H100. | text-only |

## exercises/solutions/ch08.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch08.md:171 | Answer to "DeepSeek-V3 kept certain operations in BF16": "**The optimizer state and the master weights. Already FP32** in the standard recipe" | V3 kept AdamW **moments in BF16**. Only master weights and gradients stayed FP32. V3's high-precision list is: embedding, output head, **MoE gating**, normalization, attention operators. (V3 §3.3.1, §3.3.3; part-C ch08 #1) | changes-answer |
| ch08.md:291–295 | "**E4M3** … for the forward pass and weight gradients; **E5M2** … for activation gradients. Why two: …" | "We adopt the E4M3 format on all tensors", in explicit contrast to the E4M3/E5M2 hybrid. The whole "why two" rationale is answering a question the report does not pose. (V3 §3.3.2; part-B ch05 #6, ch10) | changes-answer |

## exercises/solutions/ch09.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch09.md:258 | "Qwen3: … a **long-context extension** stage (**32K → 128K**)" | Qwen3 S3 goes **4,096 → 32,768**; 128K comes from YaRN + DCA **at inference**; RoPE base 10K → 1M via ABF. (Qwen3 §3.2; part-B ch09 #1) | changes-answer |

## exercises/solutions/ch10.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch10.md:11 | Capstone reference: "**60 layers** … **MTP**: one additional prediction head, loss weight 0.3". No dense layers and no sequence-wise balance loss. | 61 layers; first 3 dense; MTP module is a transformer block; λ 0.3 then 0.1; complementary sequence-wise balance loss. (V3 §2.1–2.2, §4.2) | changes-answer |
| ch10.md:17 | "E4M3 for forward and weight gradients, **E5M2 for activation gradients** … **FP32** master weights, **optimizer moments**" | E4M3 on all tensors; moments in **BF16**; master weights and gradients FP32. (V3 §3.3.2–3.3.3) | changes-answer |
| ch10.md:23, 295–297 | "2,788K H800 GPU-hours, ~56.7 days"; "Against the stated 'approximately 2 months.' **Consistent.**" | V3's "less than two months" is for pre-training: 2,664K hours ≈ 54.2 days. 2,788K includes context extension and post-training. (V3 Table 1; part-C ch07 #7) | text-only |
| ch10.md:65–67 | DualPipe "shaped by the H800's reduced inter-GPU bandwidth … On hardware with full NVLink bandwidth, the all-to-all is less of a bottleneck" | V3 frames the bottleneck as **cross-node IB** (NVLink 160 GB/s vs IB 50 GB/s, "3.2×"); node-limited routing exists to cut IB traffic. IB per GPU is the same as on H100 clusters, so full NVLink would not remove it. (V3 §3.2.2) | text-only |
| ch10.md:145 | "the language-modelling loss remains pure" under aux-free balancing | V3 adds a small complementary sequence-wise balance loss. (V3 §2.1.2) | text-only |
| ch10.md:181 | "with $\lambda = 0.3$ in V3" | 0.3 for the first 10T tokens, 0.1 for the last 4.8T. (V3 §4.2) | text-only |
| ch10.md:185–187, 206 | MTP overhead "≈ 6×10⁸ = **0.6B** parameters — under 0.1% of 671B"; "it costs **0.6B** parameters rather than a whole second model" | The V3 MTP module is **14B** parameters (~2% of 671B), because its block has the full MoE FFN. (DeepSeek-V3 HF README: "14B of the Multi-Token Prediction (MTP) Module weights"). Compute overhead is about one layer of 61. | changes-answer |
| ch10.md:287–291 | "$256 \times 44.0\text{M} \times 60 = 676$B … **676B > 671B**, so the naive calculation slightly overshoots. Benign explanations: …" | With 61 layers / 3 dense: **654B routed**, an *undershoot* that leaves ~17B for attention, embeddings, the shared expert and the dense FFNs, which reconciles. The "overshoot" exists only because of the wrong layer count. (V3 §4.2; config) | changes-answer |

## exercises/solutions/ch21.md

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| ch21.md:24–26, 33, 43–44 | V3 checkpoint "$671\times10^9 \times 12 = 8{,}052$ GB"; +EMA 10.7 TB; write 80 s / 107 s | V3 keeps moments in BF16: 4 (FP32 master) + 2 + 2 = 8 B/param, so **≈5.4 TB, 54 s**; +FP32 EMA ≈ 8.1 TB, 81 s. (V3 §3.3.3). The "async is mandatory at MoE scale" conclusion survives. | changes-answer |

## labs/lab05_attention_variants.py

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| lab05_attention_variants.py:271–275, 291 | "The exercise uses the **convention** $2 d_c$ … 1024 numbers per layer per token", printed as the row "MLA (2*d_c, d_c=512)" | $2 d_c$ is not an MLA convention: MLA caches the latent once (512) plus the RoPE key (64). The lab prints the correct 576 row alongside, so the output is right, but the framing legitimizes the Ex 5.2 error. (V3 §2.1) | text-only |

## labs/lab06_parallelism_memory_model.py

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| lab06_parallelism_memory_model.py:419 | `H800_2048 = Cluster("2048x H800", …, 200.0, 25.0)`: NVLink 200 GB/s, IB 25 GB/s (8×) | V3 §3.2.2: NVLink **160 GB/s**, IB **50 GB/s** (3.2×). Re-running with 160/50 changes the V3 top recommendation from `TP=1 PP=16 EP=8 DP=16` (2.55 s, 20.1% MFU) to `TP=1 PP=16 EP=4 DP=32` (2.41 s, 21.2% MFU). No printed text depends on it. | changes-lab-result |
| lab06_parallelism_memory_model.py:533 | V3 searched with `MIXED_BF16` (FP32 m and v, 16 B/param) | V3's recipe used BF16 moments (§3.3.3). The memory column for V3 is overstated by 4 B/param × sharded share. Step-time ranking is unaffected. | changes-lab-result |
| lab06_parallelism_memory_model.py:601–605 | Try-this 4: "the FP32 master weights and **both Adam moments stay FP32** … FP8's win is … not optimizer state" | True of the lab's `MIXED_FP8` preset, but V3's FP8 framework moved both moments to BF16 "without incurring observable performance degradation" (§3.3.3). Presenting FP32 moments as a law repeats the ch08 "non-negotiable" error. | text-only |
| lab06_parallelism_memory_model.py:558, 622 | "the reason **TP and EP belong inside a node**"; takeaway 4 "TP and EP belong inside a node" | The lab's own line 575 notes V3's EP64 spans 8 nodes. V3's design point is cross-node EP made cheap by DualPipe and node-limited routing. (V3 §3.2) | text-only |

## labs/lab07_collective_bandwidth.py

| file:line | wrong claim | correct value + source | impact |
|---|---|---|---|
| lab07_collective_bandwidth.py:63, 78–79 | `NVLINK = Link(…, 450.0, …)`; "NVLink / InfiniBand bandwidth ratio: 11x. That ratio is **the single most important number in Chapter 7**" | 450 GB/s unidirectional is a fair H100 figure, and the lab never says H800, so the constant itself is not wrong. But for the book's anchor cluster (H800) the ratio is **3.2×** (V3 §3.2.2). Chapter 7's "18× cliff" is the Tier 0 error; the lab's 11× does not repeat it. The prose should say "on H100". | text-only |
| lab07_collective_bandwidth.py:233–234, 384–386 | "EP=8 on NVLink vs EP=8 on InfiniBand is the placement decision that Chapter 6 section 6.7 makes"; takeaway 3 "…which is why **EP belongs inside a node**" | V3 runs EP64 across 8 nodes. (V3 §3.2; part-C ch06 #1) | text-only |

**lab07 headline check.**
- The "~2× throughput difference decided entirely by the interconnect" (`:339–341`) comes from `ring_allreduce(PAYLOAD, 1024, ETH_100)` against IB_400. Neither uses NVLink.
- Changing NVLink to 160 or 900 GB/s moves only the "hierarchical" row, which stays hidden behind the 8 s of compute either way. The headline does not depend on the NVLink figure.
- *(Outside the audit's scope, noted for the fixer:)* the lab's own printed numbers give 8 s against 28 s step time, about **3.5×**, not "~2×". That is an internal arithmetic inconsistency, not a source contradiction.

---

## Checked, no contradiction found

- `lab08_precision_and_stability.py:215` (V3 tile-wise scaling) is fine.
- `:354` ("mixed FP8 recipes use … E5M2 for gradients") is generic, not attributed to V3.
- `lab03`, `lab04`, `lab09` and `lab11`–`lab23` contain no Tier 0 facts. `lab09` teaches YaRN as a method only, and `lab22` uses a correct Llama-3-70B GQA 8:1 configuration.
- ch11 exercise/solution 11.7 (1.5M SFT instances, R1 → V3 distillation direction) matches the report.
- No exercise or solution says V3 used PPO, or that V3 was distilled into R1.
- `solutions/ch19.md:254–258` and `solutions/ch26.md` keep the correct "data, not the algorithm" order. The ch26:112 inversion is not repeated.
- `solutions/ch01.md:212–223` and `solutions/ch07.md:239` use Llama 3's 419 unexpected / 466 total interruptions and "78% hardware, GPU largest" correctly.
- No exercise or solution repeats the Llama-3 "last 5–10%" annealing, the tokenizer JSON blocks, `byte_fallback`, the Qwen3 CJK regex, V3's "spike-skip/rollback", the V3 batch ramp, the $1.50/h cost, or Mixtral-is-MHA.
