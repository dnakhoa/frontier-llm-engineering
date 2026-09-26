# Staleness audit, Part C: systems and optimization (Ch 6, 7, 8, 20, 21)

Audit date: 2026-09-26. Read-only; no repo files were edited.
All sources below were fetched and read during this audit unless listed under "Unverified leads".

**A cross-cutting finding comes first, because it affects the refresh plan.** Several of the most serious problems are not staleness. They are claims about DeepSeek-V3 and Llama 3 that the primary reports contradict, and they were already wrong when the book was written. The book's anchor case study, DeepSeek-V3, is misdescribed in Ch 6, 7 and 8: its parallelism layout, its interconnect bandwidth, its optimizer-state precision, its LR schedule, its batch ramp, and its loss-spike handling are all wrong. These are tagged WRONG-NOW with the note "(never true)". Fix them before any 2026 content is layered on top.

Second cross-cutting finding: **DeepSeek-V4 exists** (arXiv:2606.19348). It uses Muon, TileLang kernels, bitwise batch-invariant and deterministic kernels, MXFP4 QAT and a new fine-grained EP overlap scheme. Ch 8 still says "DeepSeek-V4 etc. are expected to push FP8 even further". V4 should probably become the book's second anchor next to V3.

---

## Ch 6 — Distributed training (TP, PP, DP, CP, EP)
**Verdict:** Needs rewrite of §6.10 (the DeepSeek-V3 subsection), §6.13 (the DeepSeek-V3 bullets) and §6.11 (the network tiers). Light touch elsewhere.

### Findings (ranked most-severe first)

1. **WRONG-NOW (never true)** · 06-distributed-training.md:405–415, 521–525, 565 · "DeepSeek-V3: 4-way PP × 8-way EP × 64-way DP … EP = 8 … co-located with the 8-GPU node, so the all-to-all runs over NVLink" · The report states the opposite: "16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)". It avoided TP entirely. The routed experts are "uniformly deployed on 64 GPUs belonging to 8 nodes". The all-to-all is deliberately cross-node, over IB then NVLink, with node-limited routing of at most 4 nodes per token. The whole "EP = node size" lesson in lines 415, 452 and 566 is inverted. DeepSeek's contribution was making *cross-node* EP cheap with DualPipe and custom all-to-all kernels. · Source: DeepSeek-V3 Technical Report, DeepSeek-AI, Dec 2024, §3.2 and §4.2, https://arxiv.org/abs/2412.19437
2. **WRONG-NOW (never true)** · 06:421 · "only the 37B active parameters are optimized (DeepSeek-V3 does not update the unselected experts' parameters per step)" · Every routed expert receives gradients from some tokens in a 15,360-sequence batch, so the whole model has optimizer state. The memory arithmetic in 419–425 should be redone from the real layout (PP16 × EP64, ZeRO-1). Note too that V3 keeps its AdamW moments in BF16 (see Ch 8, finding 1). · Same source, §3.3.3.
3. **DATED-FRAMING** · 06:189, 504, 513, 561 · "TP stays within a node (8 GPUs) … TP is at most 8" and "TP is limited to within a node" · The NVLink domain is no longer 8 GPUs. GB200 NVL72 and GB300 NVL72 put 72 GPUs in one NVLink domain, with 130 TB/s aggregate and 3.6 TB/s per superchip (1.8 TB/s per GPU). The TP and EP "fast domain" is now a rack. Rubin NVL72 (NVLink 6) is announced for H2 2026. · NVIDIA GB200 NVL72 product page, https://www.nvidia.com/en-us/data-center/gb200-nvl72/ ; NVIDIA GB300 NVL72 page, https://www.nvidia.com/en-us/data-center/gb300-nvl72/
4. **WRONG-NOW** · 06:403 · "5D parallelism … Used by some frontier labs but not yet standard in the public literature" · Megatron Core publishes a five-dimensional hybrid parallelism (TP, EP, CP, DP, PP) with "MoE Parallel Folding", which decouples attention-layer and MoE-layer mappings. Its 2026 MoE report gives 1,233 TFLOPS/GPU for DeepSeek-V3 on GB300. TorchTitan composes FSDP2, TP, PP, CP and EP in open source. · "MoE Parallel Folding", NVIDIA, Apr 2025, https://arxiv.org/abs/2504.14960 ; "Scalable Training of Mixture-of-Experts Models with Megatron Core", NVIDIA, Mar 2026, https://arxiv.org/abs/2603.07685 ; TorchTitan, Meta, Oct 2024 (ICLR 2025), https://arxiv.org/abs/2410.06511
5. **MISSING-MAJOR** · §6.8, 06:274, 313 · DualPipe is described only as the paper's schedule, and EP gets only pseudocode · Missing are the open-source releases a 2026 reader would expect. DeepEP (Feb 2025) provides FP8-dispatch/BF16-combine all-to-all kernels using NVLink and RDMA; V2 is up to 1.3× faster with up to 4× fewer SMs, and there are Hybrid-EP and NVFP4 branches. DualPipe and DualPipeV are open-sourced, with DualPipeV halving the device count. Ch 6 should also cover DeepSeek-V4's wave-based fused EP mega-kernel (MegaMoE), reported at 1.50–1.73× over non-fused baselines. · https://github.com/deepseek-ai/DeepEP ; https://github.com/deepseek-ai/DualPipe ; DeepSeek-V4, DeepSeek-AI, 2026, §3.1, https://arxiv.org/abs/2606.19348
6. **DATED-FRAMING** · 06:88–138 · the FSDP section uses the FSDP1 `FullyShardedDataParallel` wrapper with `ShardingStrategy`, `BackwardPrefetch` and `auto_wrap_policy` · Current PyTorch docs (2.14) centre on FSDP2 `fully_shard`, per-parameter DTensor sharding, which is what TorchTitan uses. The code block should be rewritten for FSDP2. (Separate bug: `ShardingStrategy` is used but never imported.) · https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html ; TorchTitan paper above.
7. **WRONG-NOW (never true)** · 06:140, 533, 560 · "Llama-3 used FSDP … The Meta team reported a per-block wrapping policy with `BACKWARD_PRE` prefetching" · Llama 3 reports 4D parallelism in the order [TP, CP, PP, DP], where "DP stands for FSDP". It does not report a `BACKWARD_PRE` wrapping policy. The TP=8 × PP=16 × DP=128 figure (06:527–533) is consistent with Llama 3 Table 4. Keep that figure and drop the invented detail. · The Llama 3 Herd of Models, Meta, Jul 2024, §3.3.2, https://arxiv.org/abs/2407.21783
8. **DATED-FRAMING** · 06:250–262, 563 · CP means "Ring Attention (Liu et al. 2023)" · Llama 3's production CP is explicitly *not* ring-based: "our CP implementation adopts an all-gather based method", with 2×CP chunking for load balance. DeepSeek-V4 trains up to 1M-token sequences using sparse attention. Present ring attention and all-gather CP as the two variants, and note that the dominant 2026 long-context lever has moved toward sparse or compressed attention. · Llama 3 §3.3.2 ; DeepSeek-V4 §4.2.2.
9. **WRONG-NOW** · 06:433 · "InfiniBand NDR … A typical Blackwell-era node has 8 such links" · Blackwell Ultra systems ship ConnectX-8 at 800 Gb/s per GPU (Quantum-X800 IB or Spectrum-X Ethernet). · GB300 NVL72 page (ConnectX-8 section).
10. **WRONG-NOW (never true)** · 06:199 · GPipe is attributed to "Google [3] Megatron-LM" · The citation is wrong: GPipe is Huang et al. 2019, not Megatron. Also 06:213 "Megatron-V3 (2021)" is not a real name; the paper is Narayanan et al. 2021. · (Citation fix; no new source needed.)
11. **MISSING-MINOR** · §6.6 · Missing zero-bubble schedules (ZB1P) and DualPipe's bubble table, which the DualPipe repo gives as (PP/2−1)(F&B+B−3W) versus 1F1B's (PP−1)(F+B). · https://github.com/deepseek-ai/DualPipe
12. **MISSING-MINOR** · §6.1 / §6.3 memory math · The 16-bytes-per-parameter AdamW budget is presented as fixed. Two 2025–26 changes alter it. V3 uses BF16 moments. Muon keeps one momentum buffer instead of AdamW's two, but needs whole-matrix access, which breaks element-wise ZeRO sharding; DeepSeek-V4 §3.4.1 describes a knapsack-based "hybrid ZeRO bucket assignment" to deal with this. That is a new parallelism fact worth one paragraph. · DeepSeek-V4 §3.4.1 ; Muon is Scalable, Moonshot, Feb 2025, https://arxiv.org/abs/2502.16982 (open-sources a "memory optimal and communication efficient" distributed Muon).

### Keep as-is
- §6.1 memory-wall arithmetic for dense 70B (lines 9–25), apart from the optimizer-precision caveat.
- §6.2–6.3 DP, ZeRO-1/2/3 mechanics and arithmetic.
- §6.5 column/row-parallel TP derivation and the diagram.
- §6.6 bubble formula (P−1)/(M+P−1), 1F1B memory argument, interleaving.
- §6.12 activation recomputation (Korthikanti) and the `checkpoint(use_reentrant=False)` pattern.
- The 3D config for 70B on 1,024 H100s (§6.9) as a *pedagogical* example.

### Lab opportunity
Extend `lab06_parallelism_memory_model` with (a) an NVL72 preset where TP/EP ≤ 72 in the fast domain, (b) the real DeepSeek-V3 layout (PP16/EP64/ZeRO-1, BF16 moments), and (c) Muon versus AdamW optimizer-state bytes, plus a "Muon cannot element-shard" constraint. The surprising, measurable result: the configuration the book currently prescribes for V3 does not fit the report's own constraints, and the real one does.

### References to add
- DeepEP, DeepSeek-AI, GitHub, Feb 2025–, https://github.com/deepseek-ai/DeepEP
- DualPipe / DualPipeV, DeepSeek-AI, GitHub, Feb 2025–, https://github.com/deepseek-ai/DualPipe
- Liang et al., "TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training", Oct 2024 (ICLR 2025), arXiv:2410.06511
- Liu et al., "MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core", NVIDIA, Apr 2025, arXiv:2504.14960
- Yan et al., "Scalable Training of Mixture-of-Experts Models with Megatron Core", NVIDIA, Mar 2026, arXiv:2603.07685
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", 2026, arXiv:2606.19348
- NVIDIA GB200 NVL72 and GB300 NVL72 product pages (vendor docs, accessed 2026-09-26)
- PyTorch `fully_shard` (FSDP2) docs, PyTorch 2.14

---

## Ch 7 — Cluster reality (topology, NCCL, fault tolerance)
**Verdict:** Needs rewrite of §7.1–7.3 (hardware hierarchy and the fabricated DeepSeek quote), §7.4 (DeepSeek placement), §7.7.1 and §7.12 (the failure taxonomy), and §7.15 (costs). The NCCL and debugging craft sections are a light touch.

### Findings (ranked most-severe first)

1. **WRONG-NOW (never true)** · 07-cluster-reality.md:29–33 · A quoted "DeepSeek-V3 hardware block": *"…with a total of 900 GB/s NVLink bandwidth per GPU. The 256 nodes are connected via InfiniBand, with each GPU having a 400 Gb/s NIC."* · This quotation does not appear in the report. The actual §3.1 text is: "trained on a cluster equipped with 2048 NVIDIA H800 GPUs. Each node in the H800 cluster contains 8 GPUs connected by NVLink and NVSwitch within nodes. Across different nodes, InfiniBand (IB) interconnects are utilized". §3.2.2 gives "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)". The H800 is export-limited, so it does not have 900 GB/s. The "18× cliff" framing built on this (07:47–62, 561) is therefore wrong for the book's own anchor cluster; the ratio is 3.2×. · DeepSeek-V3 report §3.1, §3.2.2, https://arxiv.org/abs/2412.19437
2. **WRONG-NOW (never true)** · 07:96–103 · The DeepSeek-V3 placement repeats EP=8/PP=4/DP=64 · Same error as Ch 6, finding 1: the real layout is PP16, EP64 across 8 nodes, ZeRO-1 DP. · Same source, §3.2.
3. **WRONG-NOW (contradicted by primary data)** · 07:407–413 · "dominant causes: Network events 40–50% … GPU hardware failures 10–20%" · Llama 3's 54-day root-cause table says the opposite: "~78% of unexpected interruptions are attributed to confirmed hardware issues … GPU issues are the largest category, accounting for 58.7%". Ch 21 (lines 15–17) quotes this correctly, so the book contradicts itself. Also 07:283: "The Llama-3 paper reports an MTBF of 'a few hours'" and "the DeepSeek team … target 2–4 hour MTBF". Neither appears in either primary source. Llama 3 gives 466 interruptions in 54 days. · Llama 3 §3.3.4, https://arxiv.org/abs/2407.21783
4. **WRONG-NOW (never true)** · 07:246, 277, 302, 388, 392, 421 · DeepSeek-V3 "suffered only a few unexpected interruptions"; "notes that NVLink errors are a non-trivial fraction of their failures"; "stated publicly that they encountered silent corruption … recovered by rolling back"; "DualPipe and DeepEP repositories include asynchronous checkpointing code"; "DeepSeek-V3 builds on this [Megatron PersistentWorker] pattern" · None of these are in the report. V3 states: "we did not experience any irrecoverable loss spikes or perform any rollbacks", and that it used its in-house HAI-LLM framework. The DualPipe and DeepEP repos are a scheduling library and a communication library; neither contains checkpointing code. Megatron's "PersistentWorker" (07:421) also looks invented (see Unverified leads). · DeepSeek-V3 abstract and §3.2 ; DeepEP and DualPipe repos.
5. **DATED-FRAMING** · 07:19, 21, 53, 66 · "For NVIDIA-based clusters in 2024–2025, a node is one DGX H100 or DGX H200"; "The rack holds 4 to 8 nodes"; "ConnectX-8 for NDR" · The 2025–26 frontier unit is the rack-scale NVL72: 72 Blackwell or Blackwell Ultra GPUs, a 130 TB/s NVLink Switch domain, fully liquid-cooled, and ConnectX-8 at 800 Gb/s per GPU (that is XDR-class, not NDR). Vera Rubin NVL72 follows in H2 2026. The node → rack → pod hierarchy in §7.2 needs a rack-as-NVLink-domain tier. · GB200 and GB300 NVL72 pages.
6. **WRONG-NOW** · 07:11, 25, 328, 405, 516, 539 · "xAI Colossus reached 100,000 H100s"; Rainier "hundreds of thousands of accelerators" (future tense) · xAI's own page: "Built in 122 days … Then we doubled it in 92 days to 200k GPUs". Project Rainier is live with "nearly half a million Trainium2 chips", and Anthropic is "on more than 1 million Trainium2 chips … by the end of the year". The xAI page also now carries "New: Anthropic partners with Colossus". · https://x.ai/colossus ; AWS, "AWS activates Project Rainier", https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster
7. **WRONG-NOW (never true)** · 07:518 · "At $1.50/H800-hour … that is ~$5.5M" · The report assumes $2/GPU-hour: 2,788K hours gives $5.576M. At $1.50 the figure would be about $4.2M. Also, 2,664K of those hours are pre-training, not "2 months" for the whole 2.788M. · DeepSeek-V3 Table 1.
8. **DATED-FRAMING / WRONG-NOW** · 07:118, 212, 226–229 · NCCL vars · In the current NCCL (2.32) docs, `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` are "deprecated in 2.17" in favour of `NCCL_MIN_CTAS` and `NCCL_MAX_CTAS`. `NCCL_ALGO` now includes NVLS and NVLSTree (NVLink SHARP) and PAT, which the list at 07:212 omits. `NCCL_TIMEOUT` (07:228) does not appear in NCCL's environment-variable docs; the watchdog timeout is a PyTorch process-group setting. `python -m torch.distributed.launch` (07:133) should be `torchrun`. · NCCL Environment Variables, NCCL 2.32.3 docs, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
9. **MISSING-MAJOR** · §7.8 silent data corruption · The section cites a non-existent reference ("[42 - 'An Epidemic of Single Event Upsets', not in references]", 07:291) and an invented $10^{-12}$ per bit-hour rate. Two real primary sources should replace them. Ma et al. (Feb 2025) compare training on healthy nodes against real SDC-afflicted nodes, which fleet management had swept out of production, and show SDCs can cause loss spikes and divergence to different optima. Llama 3 lists SDC inside its 78% hardware bucket. · "Understanding Silent Data Corruption in LLM Training", Feb 2025, https://arxiv.org/abs/2502.12340 ; Llama 3 §3.3.4
10. **MISSING-MAJOR** · §7.13 · The chapter's frameworks are Megatron, FSDP1 and DeepSpeed only. Missing: TorchTitan, which does production elastic training and checkpointing in PyTorch; torchft, which adds per-step fault tolerance across HSDP replica groups, with a "lighthouse" quorum and live peer recovery "at the training step granularity … avoiding stopping the world"; and DeepSeek's 3FS filesystem, with 6.6 TiB/s aggregate read on 180 nodes. 3FS belongs in §7.9 next to Lustre and Weka. · https://github.com/pytorch/torchft ; https://github.com/deepseek-ai/3FS ; TorchTitan arXiv:2410.06511
11. **MISSING-MINOR** · §7.3.3 · RoCE is framed as "less mature" · Llama 3 documents running a 24K-GPU RoCE cluster with E-ECMP "without traditional congestion control methods such as DCQCN". Kimi K2 trained on H800s with "8×400 Gbps RoCE" per node. Ethernet is now a first-class frontier fabric. · Llama 3 §3.3.1 ; Kimi K2 §2.4.1, https://arxiv.org/abs/2507.20534
12. **MISSING-MINOR** · §7.10 · Llama 3 reports a "diurnal 1-2% throughput variation" from temperature-driven DVFS, and site-level power swings when tens of thousands of GPUs idle together at a checkpoint or collective. These are concrete, citable replacements for the unsourced power claims. · Llama 3 §3.3.4.

### Keep as-is
- §7.5.1 ring, tree and double-binary-tree explanations.
- §7.6 hang taxonomy and §7.6.1–7.6.2 debugging workflow ("rank 17 is slow"), though the examples should drop the DeepSeek attributions.
- §7.9 filesystem failure modes (add 3FS; do not churn the rest).
- §7.11 checkpoint/dataloader/RNG pieces, which are consistent with Ch 21.
- §7.14 on-call structure, generic but harmless.

### Lab opportunity
Update `lab07_collective_bandwidth` with an "NVLink domain size" knob covering 8 (H100 or H800), 72 (NVL72) and the H800's 160 GB/s. Run the same EP all-to-all under three topologies: V3's real two-hop IB→NVLink with node-limited routing (M=4), naive flat IB, and a single NVL72 domain. The measurable surprise is that node-limited routing on a 3.2×-ratio cluster beats flat routing on a much faster fabric at the same expert count.

### References to add
- NVIDIA GB200 NVL72 and GB300 NVL72 product pages (vendor, accessed 2026-09-26)
- xAI, "Colossus", https://x.ai/colossus (accessed 2026-09-26)
- AWS/Amazon, "AWS activates Project Rainier", https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster (2025)
- NVIDIA, NCCL 2.32 Environment Variables docs
- Ma, Pei, Lausen, Karypis, "Understanding Silent Data Corruption in LLM Training", Feb 2025, arXiv:2502.12340
- Meta, torchft, GitHub, https://github.com/pytorch/torchft
- DeepSeek-AI, 3FS (Fire-Flyer File System), GitHub, Feb 2025, https://github.com/deepseek-ai/3FS
- Kimi Team, "Kimi K2: Open Agentic Intelligence", Jul 2025, arXiv:2507.20534 (cluster section)

---

## Ch 8 — Optimization (precision, schedules, scaling laws)
**Verdict:** Needs rewrite of §8.3 (optimizer choice), §8.4 (schedule), §8.7–8.9 (precision), §8.14 (batch size) and §8.15 (frontier configs). This is the most-stale chapter in Part C. §8.2, §8.5, §8.6, §8.10 and §8.12 are mostly fine.

### Findings (ranked most-severe first)

1. **WRONG-NOW (never true)** · 08-optimization.md:120, 248, 289, 527 · "the optimizer state (m, v) is held in FP32 … This is essential"; "This is non-negotiable" · The book's own anchor case study did the opposite: "We adopt the BF16 data format instead of FP32 to track the first and second moments in the AdamW optimizer, without incurring observable performance degradation" (master weights and gradients stayed FP32). · DeepSeek-V3 §3.3.3.
2. **WRONG-NOW (never true)** · 08:442, 475, 513 · "DeepSeek-V3 uses this pattern: skip a window of ~200 batches after a spike, reduce the LR by 50%" · The report says: "we did not experience any irrecoverable loss spikes or perform any rollbacks." The skip-200/halve-LR policy is not in the report. · DeepSeek-V3 abstract.
3. **WRONG-NOW (never true)** · 08:154–159, 470–472, 507 · DeepSeek-V3 schedule "warmup … then cosine decay to 10% of peak"; batch "ramps from 2,304 sequences (12M tokens) to 14,400 sequences (73M tokens) in the first 1.3%" · Actual schedule: warmup to 2.2e-4 over 2K steps, **constant until 10T tokens**, cosine to 2.2e-5 over 4.3T, then constant 2.2e-5 for 333B tokens and 7.3e-6 for the last 167B. Actual batch ramp: 3,072 to 15,360 sequences over the first 469B tokens, which at 4K length is about 12.6M to 62.9M tokens. · DeepSeek-V3 §4.2.
4. **MISSING-MAJOR** · §8.3, 08:56–58, 539 · "AdamW, the de-facto optimizer. Almost every frontier pre-training run uses it"; "the optimizer is AdamW … universal" · By 2025–26, Muon is a production frontier optimizer:
   - Moonshot's "Muon is Scalable" adds weight decay and per-parameter update-RMS matching and reports "~2× computational efficiency compared to AdamW with compute optimal training" (Moonlight 16B-A3B, 5.7T tokens).
   - Kimi K2 (1T total / 32B active) was pre-trained on 15.5T tokens with **MuonClip** (Muon + QK-Clip) "with zero loss spike".
   - GLM-4.5 uses Muon for all parameters except embeddings, biases and RMSNorm (NS=5 steps, momentum 0.95, update RMS 0.2).
   - DeepSeek-V4 uses Muon "for the majority of modules" and keeps AdamW for the embedding, prediction head and RMSNorm weights.

   The chapter needs a Muon section: the Newton–Schulz orthogonalisation, the hybrid AdamW-for-embeddings/norms split, QK-Clip, and the distributed/ZeRO complication. · Jordan, "Muon: An optimizer for hidden layers in neural networks", blog, https://kellerjordan.github.io/posts/muon/ ; Liu et al., Moonshot AI, Feb 2025, https://arxiv.org/abs/2502.16982 ; Kimi K2, Jul 2025, https://arxiv.org/abs/2507.20534 ; GLM-4.5, Zhipu/Z.ai, Aug 2025, https://arxiv.org/abs/2508.06471 ; DeepSeek-V4 §2.4, https://arxiv.org/abs/2606.19348
5. **MISSING-MAJOR** · §8.9, 08:293, 355, 542 · "FP8: the new frontier … Frontier runs on Blackwell (Llama 4, DeepSeek-V4, etc.) are expected to push FP8 even further" · Several things have since shipped. Llama 4 Behemoth was pre-trained in FP8 on 32K GPUs at 390 TFLOPs/GPU. Blackwell adds native MXFP8/MXFP6/MXFP4 and NVFP4 tensor-core formats. NVIDIA pre-trained a 12B model on 10T tokens in **NVFP4**, matching an FP8 baseline, using random Hadamard transforms, 2D quantization, stochastic rounding and some high-precision layers. TorchTitan with MXFP8 and DeepEP reports +41% throughput for DeepSeek-V3 671B on 256 B200s, with MXFP8 convergence equal to BF16 on 16B. Megatron Core ships FP8 and NVFP4 training. DeepSeek-V4 applies MXFP4 QAT to expert weights and the CSA indexer QK path in post-training. · Meta, "The Llama 4 herd", Apr 2025, https://ai.meta.com/blog/llama-4-multimodal-intelligence/ ; NVIDIA, "Pretraining Large Language Models with NVFP4", Sep 2025, https://arxiv.org/abs/2509.25149 ; PyTorch/Nebius blog, https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/ ; Megatron Core MoE, arXiv:2603.07685 ; DeepSeek-V4 §5.2.1
6. **WRONG-NOW (never true)** · 08:300 · "Both formats [E4M3, E5M2] are unsigned (no sign bit …)" · False: both OCP FP8 formats have a sign bit (1+4+3, 1+5+2). The book's own `lab08_precision_and_stability` casts real `float8_e4m3fn` values, which are signed. Also 08:355 "Blackwell … native FP8 tensor cores at 2× the throughput of BF16" presents a Hopper-era ratio as new; the Blackwell novelty is FP4 at 2× FP8 (GB200: FP8 720 PF, NVFP4 1,440 PF sparse). · GB200 NVL72 spec table.
7. **DATED-FRAMING** · 08:124–133, 507–514, 540 · "the schedule shape is universal (warmup + cosine to 10% floor)" · Warmup-stable-decay (WSD) and constant-then-cooldown schedules are now common at the frontier. Kimi K2 used WSD: constant 2e-4 for 10T tokens, then cosine over 5.5T. DeepSeek-V3 and V4 hold a constant LR for most of training. Hägele et al. show constant LR plus cooldown "scales predictably and reliably similar to cosine" and makes scaling studies cheaper. This is not universal: GLM-4.5 tested WSD and chose cosine because WSD underfit on general benchmarks. The book should present the choice as a trade-off. · Hägele et al., May 2024 (NeurIPS 2024 spotlight), https://arxiv.org/abs/2405.18392 ; Kimi K2 §2.5 ; GLM-4.5 §2.4
8. **WRONG-NOW (never true)** · 08:165–171, 479–490 · Llama 3 70B "Peak LR 1.5e-4 … 16M constant"; 8B "3e-4"; Qwen3 "7e-5 … 2e-4 for 32B … warmup 0.5%" · Llama 3 publishes the recipe for **405B only**: "peak learning rate of 8×10⁻⁵, a linear warm up of 8,000 steps", with the batch ramped 4M → 8M → 16M tokens (at 252M and 2.87T tokens). It is not constant. The Qwen3 report's pre-training section gives three stages (30T + ~5T + long-context), not per-model peak LRs. The table's Qwen3 and Llama-3-70B/8B rows look invented and should be removed or sourced. · Llama 3 §3.4.1 ; Qwen3 Technical Report, May 2025, §3.2, https://arxiv.org/abs/2505.09388
9. **DATED-FRAMING / WRONG-NOW** · 08:453, 483, 546 · "B* grows slowly with model size … For a 70B model, B* is roughly 4M–8M tokens"; "Llama 3 trained at ~4× CBS" · Zhang et al. find "CBS scales primarily with data size rather than model size". The Llama 3 CBS figure is not in the Llama 3 paper. Frontier batches have grown to 60–95M tokens: Kimi K2 used 67M, and DeepSeek-V4 ramps to 75.5M (Flash) and 94.4M (Pro). · Zhang et al., "How Does Critical Batch Size Scale in Pre-training?", Oct 2024 (ICLR 2025), https://arxiv.org/abs/2410.21676 ; Kimi K2 §2.5 ; DeepSeek-V4 §4.2.2
10. **DATED-FRAMING** · 08:392–406 · µP presented as the hyperparameter-transfer method, with "Megatron-LM, Transformer Engine provide µP parameterizations" · Llama 4 introduced "MetaP", which sets per-layer LRs and init scales with transfer "across … batch size, model width, depth, and training tokens". Muon's update-RMS matching (Moonlight, K2, V4) is another route to re-using AdamW LRs. The Megatron/TE µP claim is unverified. · Llama 4 blog ; Muon is Scalable.
11. **DATED-FRAMING** · §8.13, 08:418–429 · QK-norm as "cheap insurance … used in StableLM" · Kimi K2 notes QK-Norm "is not applicable to multi-head latent attention (MLA)" and introduces QK-Clip, which rescales W_q and W_k after each update to cap max logits at τ=100, because Muon made logit explosion (>1000) worse. This is the 2025 stability story. · Kimi K2 §2.1
12. **WRONG-NOW (never true)** · 08:52, 118, 191, 406 · "DeepSeek-V3 … custom fused cross-entropy"; "DeepSeek-V3 in particular use higher [weight decay] values for the embedding … 0.2"; "per-block learning rate schedule … consistent with µP" · The report gives a single weight_decay=0.1 and does not describe a per-block LR schedule or a fused cross-entropy optimisation. · DeepSeek-V3 §4.2.
13. **MISSING-MINOR** · 08:461 · "the optimizer state scales linearly with batch size" · False; optimizer state is independent of batch size. · (Correctness fix.)

### Keep as-is
- §8.2 cross-entropy, perplexity, why not MSE.
- §8.3 AdamW reference code and the decoupled weight decay explanation (reframe it as "the baseline", not "universal").
- §8.5 weight-decay exclusions; §8.6 global-norm clipping at 1.0. This is confirmed by DeepSeek-V3 ("gradient clipping norm is set to 1.0") and remains standard.
- §8.8 BF16 vs FP16 range/precision table and the loss-scaling explanation.
- §8.10 Kaplan/Chinchilla framing (fix "Kaplan: roughly in proportion", since Kaplan tilts strongly to N), and §8.11 inference-aware scaling.
- DeepSeek-V3's fine-grained FP8 description at 08:307 (1×128 activation tiles, 128×128 weight blocks, FP32 accumulation) is broadly right; keep it and add V3's note that the H800's FP8 accumulator keeps only about 14 bits, which is why they promote to CUDA cores.

### Lab opportunity
1. **New `lab08_muon_vs_adamw` (CPU):** a small MLP or char-level transformer trained with AdamW and with Muon (5-step Newton–Schulz, update-RMS matched to AdamW). Measure (a) steps-to-loss and (b) max attention logit with and without QK-Clip. Surprising, measurable results: Muon reaches the target in fewer steps, and without clipping its max logit grows much faster than AdamW's, the mechanism the K2 report describes.
2. **Update `lab08_precision_and_stability`:** add an MXFP4 (E2M1 + E8M0 per-32 scale) and NVFP4 (E2M1 + E4M3 per-16 scale + FP32 tensor scale) *simulation* next to the real FP8 casts. Show (a) that a single outlier destroys a per-tensor FP4 block, (b) that a random Hadamard rotation before quantization recovers most of it, and (c) that stochastic rounding makes the mean of quantized gradients unbiased. All three mechanisms come from the NVFP4 paper.
3. Add a BF16-moments ablation to the same lab to check V3's "no observable degradation" claim on a toy problem.

### References to add
- Jordan, "Muon: An optimizer for hidden layers in neural networks", blog (2024), https://kellerjordan.github.io/posts/muon/
- Liu et al. (Moonshot AI), "Muon is Scalable for LLM Training", 24 Feb 2025, arXiv:2502.16982
- Kimi Team, "Kimi K2: Open Agentic Intelligence", 28 Jul 2025 (rev. Feb 2026), arXiv:2507.20534
- GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", 8 Aug 2025, arXiv:2508.06471
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", 2026, arXiv:2606.19348
- NVIDIA, "Pretraining Large Language Models with NVFP4", 29 Sep 2025 (rev. 2026), arXiv:2509.25149
- Meta, "The Llama 4 herd", blog, 5 Apr 2025, https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- PyTorch & Nebius, "Enabling Up to 41% Faster Pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan", PyTorch blog (2026, see Unverified leads)
- Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", May 2024 (NeurIPS 2024), arXiv:2405.18392
- Zhang et al., "How Does Critical Batch Size Scale in Pre-training?", Oct 2024 (ICLR 2025), arXiv:2410.21676
- Qwen Team, "Qwen3 Technical Report", 14 May 2025, arXiv:2505.09388 (to replace the reference [6] details)

---

## Ch 20 — Custom kernels (Triton, CUDA, fused ops)
**Verdict:** Light touch for the mechanism sections. Needs refresh of §20.5 (the lineage paragraph), §20.6–20.7 (the tooling landscape) and §20.11 (frontier kernel work). The pedagogy is the book's strongest and ages well.

### Findings (ranked most-severe first)

1. **MISSING-MAJOR** · 20-custom-kernels.md:154 · FlashAttention lineage ends at FA-3 (Hopper) · FlashAttention-4 (Mar 2026, MLSys 2026) targets Blackwell's "asymmetric hardware scaling": tensor-core throughput doubles while shared-memory bandwidth and exponential units do not. FA-4 therefore uses fully async MMA, software-emulated exp, conditional softmax rescaling, tensor memory (TMEM) and 2-CTA MMA. It reaches "up to 1613 TFLOPs/s (71% utilization)" on B200 in BF16, 1.3× over cuDNN 9.13 and 2.7× over Triton, and is "implemented … entirely in CuTe-DSL", with 20–30× faster compiles. This extends the chapter's thesis that "kernels are hardware-specific, and they age" and should be the next paragraph. · Zadouri et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", 5 Mar 2026, https://arxiv.org/abs/2603.05451 ; repo https://github.com/Dao-AILab/flash-attention (`pip install flash-attn-4`)
2. **MISSING-MAJOR** · §20.6–20.7, 20:160, 229 · The escalation ladder is "PyTorch → torch.compile → Triton → CUDA" · The middle of that ladder has filled in:
   - **TileLang** is a tile DSL that decouples dataflow from scheduling, with backends from SM70 to SM120 plus AMD, Metal and Ascend. DeepSeek-V4 used it to replace "the vast majority" of "hundreds of fine-grained Torch ATen operators" with fused kernels.
   - **CuTe DSL** (CUTLASS 4.x Python) gives explicit control of layouts, copy and MMA atoms, and pipelines from Python, with the aim of matching CUTLASS C++. FA-4 is built on it.
   - **ThunderKittens** offers warp-level 16×16 tiles and matches cuBLAS and FA-3 on H100.

   The book should present these as a new rung between Triton and raw CUDA. · Wang et al., "TileLang: A Composable Tiled Programming Model for AI Systems", Apr 2025, https://arxiv.org/abs/2504.17577 and https://github.com/tile-ai/tilelang ; NVIDIA CUTLASS Python/CuTe DSL overview, https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html ; Spector et al., "ThunderKittens", Oct 2024 (ICLR 2025), https://arxiv.org/abs/2410.20399 ; DeepSeek-V4 §3.2
3. **WRONG-NOW** · 20:224 · "a hardware feature Triton does not expose … the TMA engine, asynchronous copies" · Current Triton exposes TMA through `tl.make_tensor_descriptor` and host `TensorDescriptor`, has warp specialization, and its official tutorial covers TMA-persistent and CLC matmuls. The reasons for dropping to CUDA/CuTe have narrowed to TMEM/2-CTA MMA, fine layout control and irregular MoE comms. · Triton "Persistent Matmul" tutorial, https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
4. **DATED-FRAMING** · 20:19–39, 324 · "For an H100 … ridge point ≈ 295 FLOP/byte" is the only machine model · Keep the H100 derivation, but add a B200/GB200 row. From the NVIDIA table, a GB200 superchip has 10 PF dense-equivalent BF16 (sparse 10 PF, so 5 PF dense, for 2 GPUs) and 16 TB/s HBM3E. That is roughly 2.5 PF and 8 TB/s per GPU, a ridge point of about 310 FLOP/byte for BF16 and about 625 for FP8. The ridge climbs with every precision step, which is FA-4's "asymmetric scaling" point in roofline form. (Recompute from the datasheet before printing.) · GB200 NVL72 spec table.
5. **MISSING-MAJOR** · §20.11, 20:303–307 · "MoE kernels … Quantization kernels … DeepSeek-V3's FP8 training (§8.3)" · The chapter has no concrete open-source examples. It should add DeepEP (dispatch/combine), DeepSeek-V4's MegaMoE fused EP mega-kernel, TorchAO MXFP8 grouped GEMMs (+41% combined with DeepEP on V3-671B/B200) and Megatron Core's Grouped GEMM with FP8/NVFP4. Cross-reference fix: FP8 is §8.9, not §8.3. · DeepEP repo ; DeepSeek-V4 §3.1 ; PyTorch/Nebius MXFP8 blog ; Megatron Core MoE arXiv:2603.07685
6. **MISSING-MINOR** · §20.8, 20:241 · "kernels break determinism … atomics are the usual culprit" · DeepSeek-V4 ships "end-to-end, bitwise batch-invariant, and deterministic kernels with minimal performance overhead", aimed at "bitwise alignment among pre-training, post-training, and inference". A frontier lab now treats determinism as achievable at production cost; this should be a paragraph here and in Ch 21 §21.8. · DeepSeek-V4 §3.3

### Keep as-is
- §20.2 four GPU facts and the H100 memory-hierarchy table (label it as H100).
- §20.3 regimes and arithmetic-intensity worked examples (matmul ≈ 3,600, RMSNorm ≈ 1).
- §20.4 fusion; §20.5 online-softmax derivation, "not an approximation", backward recompute.
- §20.6 Triton RMSNorm kernel and the recurring patterns.
- §20.8 numerics; §20.9 benchmarking honestly; §20.10 torch.compile first. These are evergreen and are the chapter's best material.

### Lab opportunity
Extend `lab20_triton_fused_kernel`'s CPU half with a "ridge point by precision and generation" section. Compute roofline positions for RMSNorm, SwiGLU, attention and GEMM on H100 BF16/FP8 and on B200 BF16/FP8/FP4 from datasheet numbers. The surprising, measurable result: the same attention kernel moves from compute-bound toward exp/softmax-bound as precision drops, which motivates FA-4's software exp. Optionally, add a CPU-runnable online-softmax variant with FA-4-style *conditional* rescaling (rescale only when the max grows by more than a threshold) and count rescale operations saved while checking that the result is still exact.

### References to add
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, "FlashAttention-4 …", 5 Mar 2026, arXiv:2603.05451
- Wang et al., "TileLang: A Composable Tiled Programming Model for AI Systems", 24 Apr 2025, arXiv:2504.17577
- Spector et al., "ThunderKittens: Simple, Fast, and Adorable AI Kernels", 27 Oct 2024, arXiv:2410.20399
- NVIDIA, CUTLASS 4.x Python / CuTe DSL documentation
- Triton "Persistent Matmul" tutorial (TMA / tensor descriptors)
- DeepEP (GitHub); DeepSeek-V4 §3.1–3.3

---

## Ch 21 — Checkpointing, resumption, the two-week-run problem
**Verdict:** Light touch. This chapter is correct against its primary source (Llama 3's 466/419/78%/>90% numbers match §3.3.4) and is mostly evergreen. It needs additions, not rewrites.

### Findings (ranked most-severe first)

1. **MISSING-MAJOR** · §21.8, 21:204–219, 284 · "The pragmatic position most labs take: do not require bit-exactness … Keep a deterministic mode available for debugging on small configurations" · DeepSeek-V4 reports production-scale "end-to-end, bitwise batch-invariant, and deterministic kernels with minimal performance overhead" to align pre-training, post-training and inference. The "most labs" generalisation should be softened, and the book should present V4 as the counter-example. V4 also shows why this matters for resumption: its RL rollout service uses a token-granular write-ahead log, because regenerating interrupted requests from scratch "introduces length bias". That is a new, surprising resumption bug class and fits §21.6/§21.11 perfectly. · DeepSeek-V4 §3.3, §5.2.3, https://arxiv.org/abs/2606.19348
2. **MISSING-MAJOR** · §21.5, §21.9 · Async checkpointing is described generically, and elastic training is dismissed as less common · Real systems to name: PyTorch DCP `async_save`, reported as going from under 5 minutes to under 30 seconds of stall and "over 10x" faster, with the next step being "zero overhead checkpointing" by streaming weights during backward. ByteCheckpoint gives parallelism-agnostic load-time resharding, a 54.20× average stall reduction and up to 9.96×/8.80× faster save/load. torchft gives per-step fault tolerance across replica groups with live recovery from healthy peers, a checkpoint-free path §21.9 does not mention. · PyTorch blog, "Reducing Model Checkpointing Times by Over 10x with PyTorch Distributed Asynchronous Checkpointing", https://pytorch.org/blog/reducing-checkpointing-times/ ; Wan et al., "ByteCheckpoint", Jul 2024 (rev.), https://arxiv.org/abs/2407.20143 ; https://github.com/pytorch/torchft
3. **DATED-FRAMING** · §21.2–21.3, 21:29–39, 51–61 · Checkpoint sizing assumes FP32 Adam m and v (840 GB for 70B) · Two changes shrink or reshape this. V3-style BF16 moments give 70B × (4 + 2 + 2) = 560 GB. Muon keeps one momentum buffer and no v. The checkpoint must also record *which optimizer owns which parameter*, since V4 and K2 use AdamW for embeddings and norms and Muon elsewhere. Add a row for "optimizer type / per-parameter optimizer map". FP8 or MXFP8 scaling state is correctly flagged already (21:37). · DeepSeek-V3 §3.3.3 ; DeepSeek-V4 §2.4
4. **MISSING-MINOR** · §21.10 · The training→inference conversion list omits FP4/MXFP4 QAT outputs · V4 notes its FP4→FP8 dequantization is lossless when the scale ratio condition holds; this is a new conversion path to check. · DeepSeek-V4 §5.2.1
5. **MISSING-MINOR** · §21.7 · Young/Daly worked example · Add Llama 3's observed power-swing effect: tens of thousands of GPUs idling together during checkpoint or collectives cause site-level power fluctuation. This is a checkpoint-cadence constraint the formula does not capture. · Llama 3 §3.3.4

### Keep as-is
- §21.1 failure framing and the Llama 3 statistics, which are correct.
- §21.2 state table; §21.3 sizing arithmetic (add the BF16-moment and Muon variants); §21.4 sharded/resharding DCP code.
- §21.5 async mechanics, atomic rename and completion marker.
- §21.6 data-loader bug and the sample-ID test, the single best section in Part C.
- §21.7 Young/Daly formula and worked example; §21.10 safetensors-not-pickle; §21.11 symptom table.

### Lab opportunity
Add a section to `lab21_checkpoint_resume`: an "interrupted-generation length-bias" demo. Simulate RL rollouts with a variable length distribution and random preemption. Compare regenerating from scratch with WAL-resume. The mean completed length measurably shrinks under regenerate-from-scratch, the DeepSeek-V4 finding, with nothing erroring. This matches the lab's existing style: a silent, measurable divergence. A second, smaller addition: an optimizer-mixture checkpoint (Muon on matrices, AdamW on norms) where forgetting the per-parameter optimizer map passes shape checks but diverges on step 1.

### References to add
- PyTorch, "Reducing Model Checkpointing Times by Over 10x with PyTorch Distributed Asynchronous Checkpointing", PyTorch blog (2024), https://pytorch.org/blog/reducing-checkpointing-times/
- Wan et al., "ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development", 29 Jul 2024 (rev.), arXiv:2407.20143
- Meta, torchft, https://github.com/pytorch/torchft
- DeepSeek-AI, DeepSeek-V4, 2026, arXiv:2606.19348 (§3.3 determinism, §5.2.3 WAL rollout)

---

## Unverified leads
These were not confirmed from a primary source during this audit. Do not cite them without checking.

- **DeepSeek-V4 dates.** The arXiv abstract page shows "Submitted on 26 Apr 2026" under ID 2606.19348, which is odd (a 2606 ID implies June). Confirm the date before citing. I also did not confirm V4's *pre-training* matmul precision (FP8 vs MXFP8) or its training hardware; only the post-training MXFP4 QAT is confirmed.
- **Rubin.** Vera Rubin NVL72 (formerly "NVL144") and NVLink 6 at 3.6 TB/s per GPU, shipping H2 2026. This comes from search snippets of NVIDIA newsroom and third-party coverage; the fetched newsroom page did not surface the specs in text.
- **FA-4 award.** FlashAttention-4 is said to be an MLSys 2026 Best Paper Honorable Mention (search snippet only).
- **MXFP8 on Llama 4 Scout/GB200.** The TorchTitan MXFP8 result ("1.2–1.3× vs BF16") appeared only in search snippets. The DeepSeek-V3 +41% B200 result is confirmed. The blog's publication date was not captured.
- **Muon blog date.** Keller Jordan's Muon blog is believed to date from Dec 2024; the date was not captured from the page.
- **PyTorch env-var renames.** `TORCH_NCCL_ASYNC_ERROR_HANDLING` replacing `NCCL_ASYNC_ERROR_HANDLING`, and `torchrun` replacing `torch.distributed.launch`, are well known, but I did not fetch the PyTorch docs page this session.
- **Megatron "PersistentWorker" (07:421).** I found no evidence that it exists. It is likely invented; check Megatron-LM docs before keeping.
- **"Alibaba Panjin cluster" for Qwen3 (07:13, 330).** Not found in the Qwen3 technical report sections I read. Possibly invented.
- **Meta "1.3M H100-equivalent supercluster" (07:11, 25, 539).** No primary source was fetched.
- **"Megatron-LM, Transformer Engine provide µP parameterizations" (08:406).** Unverified.
- **Newer SDC work.** 2026 arXiv papers "TrainSDC" (2608.30769), "The Anatomy of Silent Data Corruption" (2605.04213) and "LLM-PRISM" (2604.10390) appeared in search results. I did not read them.
- **Other leads not investigated:** Helion (PyTorch kernel DSL), Triton Gluon, DeepGEMM and FlashMLA (DeepSeek open-infra week, Feb 2025), DeepSeek-V3.1 UE8M0 FP8 scale format, OCP Microscaling spec v1.0 (Rouhani et al. 2023, cited inside V3 and V4), the Llama 3 context-parallel paper (Chu et al. 2025), and Olmo 3 (arXiv 2512.13961, HSDP + CP on H100s) as an open reference recipe.
