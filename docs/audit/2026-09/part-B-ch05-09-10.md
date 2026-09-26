# Staleness audit, Part B: Chapters 5, 9, 10

Audit date: 2026-09-26. Read-only; no repo files were edited.
Method: I read each chapter in full, then checked its load-bearing claims against primary sources (arXiv reports, official model cards on Hugging Face, lab blogs). I downloaded each source and grepped it; every number quoted below was read from that source. Where a claim in the book was **never** true against the source it cites, I say so and still rank it here, because a refresh has to fix it anyway. Those findings carry the tag "never matched source".

Line numbers refer to the files at commit `e02b15a`.

---

## Ch 5 — Architecture: dense, MoE, attention variants

**Verdict:** Needs a rewrite of §5.1, §5.3.2, §5.4–5.5, §5.12, §5.13.3, §5.14.1, §5.14.3, §5.14.4, §5.15 and §5.16. Needs a new section on efficient and hybrid attention. The mechanism sections (§5.2, §5.3.1, §5.6–5.10) can stay, apart from the small fixes listed below.

### Findings (ranked most-severe first)

1. **WRONG-NOW** · `05-architecture.md:815`, `:7` · "There are no fundamentally new operations in any 2024–2025 frontier model." / "The frontier work of 2024–2025 is not about new high-level ideas". In 2025–26, frontier open models shipped operations that are genuinely new at scale:
   - linear-attention/softmax hybrids: Gated DeltaNet in Qwen3-Next and Qwen3.5; KDA in Kimi Linear and Kimi K3; lightning attention in MiniMax-M1;
   - learned sparse attention: DSA in DeepSeek-V3.2 and GLM-5; CSA/HCA in DeepSeek-V4; MiniMax Sparse Attention;
   - new residual topologies: mHC in DeepSeek-V4; Attention Residuals in Kimi K3.

   Sources:
   - Qwen3-Next-80B-A3B model card, Qwen, 2025-09-09. Layout "12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))". https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
   - Qwen3.5-397B-A17B model card, Qwen, 2026-02-16. 60 layers, "15 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))". https://huggingface.co/Qwen/Qwen3.5-397B-A17B
   - Kimi Linear, Moonshot, arXiv:2510.26692, 2025-10-30. https://arxiv.org/abs/2510.26692
   - Kimi K3 model card, Moonshot, July 2026. 93 layers, "69 KDA + 24 Gated MLA", 896 experts with 16 selected, 1M context. https://huggingface.co/moonshotai/Kimi-K3
   - MiniMax-M1, arXiv:2506.13585, 2025-06-16. One softmax block after every seven lightning-attention blocks. https://arxiv.org/abs/2506.13585
   - DeepSeek-V3.2, arXiv:2512.02556, 2025-12-02. https://arxiv.org/abs/2512.02556
   - DeepSeek-V4, arXiv:2606.19348 (see the note on its date under "References to add"). https://arxiv.org/abs/2606.19348
   - mHC, arXiv:2512.24880, 2025-12-31. https://arxiv.org/abs/2512.24880

2. **MISSING-MAJOR** · §5.3 (`:177–319`) and §5.13.2 (`:712–716`) · The attention taxonomy stops at MHA/MQA/GQA/MLA, plus a two-sentence note on sliding windows. A 2026 reader needs a section on sub-quadratic attention. The key cases:
   - **DSA** (V3.2). A "lightning indexer" plus top-k token selection; top-k = 2048 KV tokens in the sparse stage. It was added to V3.1-Terminus by continued pre-training: a dense warm-up, then a 943.7B-token sparse stage at 128K.
   - **CSA + HCA** (V4). Compressed Sparse Attention (compression m=4, top-k 512 in Flash and 1024 in Pro) interleaved with Heavily Compressed Attention (m′=128), plus a 128-token sliding-window branch. V4-Pro needs "27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2" at 1M tokens. Against a BF16 GQA8 baseline, V4's KV cache is about 2%.
   - **Linear/softmax hybrids** at a 3:1 ratio (Qwen3-Next, Qwen3.5, Kimi K3) and at 7:1 (MiniMax-M1).
   - **Local:global SWA hybrids.** Gemma 3 uses 5:1 with a 1024-token window. gpt-oss alternates a 128-token banded window with dense layers. Gemma 4 uses 512 or 1024-token windows and makes the final layer global.
   - **The counter-example.** MiniMax-M2 went back to full attention and published why (numerical precision, prefix caching, immature infrastructure). A balanced chapter should include this.

   Sources:
   - DeepSeek-V3.2 §2.1, arXiv:2512.02556.
   - DeepSeek-V4 §2.3 and §4.2.1, arXiv:2606.19348.
   - Gemma 3 Technical Report, arXiv:2503.19786, 2025-03-25.
   - gpt-oss-120b & gpt-oss-20b Model Card, OpenAI, arXiv:2508.10925, 2025-08-08.
   - Gemma 4 model card, Google, 2026. https://ai.google.dev/gemma/docs/core/model_card_4
   - "Why Did MiniMax M2 End Up as a Full Attention Model?", MiniMax, 2025-10-30. https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model
   - Native Sparse Attention (NSA), DeepSeek/PKU, arXiv:2502.11089, 2025-02-16 (the precursor).

3. **WRONG-NOW** · `:326`, `:353`, `:355`, `:817` · "NoPE ... Not used at frontier" / "RoPE has won. Llama, Qwen, DeepSeek, ... all use RoPE" / "RoPE has won the position-encoding war."
   - **Llama 4** uses "interleaved attention layers without positional embeddings" (iRoPE) plus inference-time attention temperature scaling. Source: "The Llama 4 herd", Meta AI blog, 2025-04-05. https://ai.meta.com/blog/llama-4-multimodal-intelligence/
   - **SmolLM3** uses NoPE layers. Source: "SmolLM3: smol, multilingual, long-context reasoner", Hugging Face blog, 2025. https://huggingface.co/blog/smollm3
   - **Partial RoPE** is now common:
     - GLM-4.5: "Grouped-Query Attention with partial RoPE". arXiv:2508.06471, 2025-08-08.
     - Qwen3-Next and Qwen3.5: rotary dimension 64 of head dimension 256 (model cards).
     - DeepSeek-V4: RoPE on the last 64 dimensions only, and also applied with position −i to the attention *outputs*.
     - Gemma 4: "Proportional RoPE (p-RoPE)" on its global layers.

     Stating "RoPE has won" as absolute is now false. The true statement is "RoPE is the default, applied partially or skipped on some layers."

4. **MISSING-MAJOR** · §5.2 (`:116–122`), §5.6, §5.15 · The block-level stabilizers that became standard in 2025 are missing.
   - **QK-Norm.** Used in Qwen3 (which also removes QKV-bias), GLM-4.5, and V4 (an RMSNorm on each query head and on the KV entry). Gemma 3 replaced Gemma 2's soft-capping with QK-norm.
   - **Output-gated attention.** A sigmoid gate after SDPA (Qwen, "Gated Attention for LLMs", arXiv:2505.06708, 2025-05-10). Deployed as "Gated Attention" in Qwen3-Next and Qwen3.5 and as "Gated MLA" in Kimi K3.
   - **Attention sinks.** gpt-oss adds "a learned bias in the denominator of the softmax" per head.
   - **mHC.** Doubly-stochastic hyper-connections replace the plain residual in V4 (Sinkhorn-Knopp, 20 iterations, expansion factor 4). The residual-stream diagram at `:124–173` is no longer universal.

   Sources: Qwen3 arXiv:2505.09388; Gemma 3 arXiv:2503.19786; GLM-4.5 arXiv:2508.06471; gpt-oss arXiv:2508.10925; DeepSeek-V4 §2.2 and §2.3.3.

5. **WRONG-NOW (never matched source)** · §5.14.3 `:762–776`, `:639`, `:725`, `:770` · The Qwen3-235B configuration is wrong in several places. The Qwen3 report says:
   - "excludes shared experts" and adopts a "global-batch load balancing loss". It is neither "the standard auxiliary loss" nor "bias-based ... on their roadmap". §5.11 `:639` also wrongly says Qwen3-MoE uses "shared experts".
   - Pre-training is 4,096 tokens (S1, over 30T tokens) and then 4,096→32,768 (S3). RoPE base goes 10,000→1,000,000 via ABF, and YaRN + DCA give a 4× extension *at inference*. So "trained at 32K directly, no extension needed" and "Qwen3 ships a 128K variant directly trained at that context length" (`:725`) are both false.
   - The report does not describe a "dual base" RoPE.
   - The chapter's expert d_ff = 12288 is not in the report. I could not confirm the correct value from the paper, so this is listed under unverified leads.

   Source: Qwen3 Technical Report, Qwen Team, arXiv:2505.09388, 2025-05-14. https://arxiv.org/abs/2505.09388

6. **WRONG-NOW (never matched source)** · §5.14.1 `:736–744` · The DeepSeek-V3 configuration is wrong in four places.
   - **Layers.** The report says "number of Transformer layers to 61", not 60.
   - **KV cache per token per layer.** MLA caches the latent (d_c = 512) plus the decoupled RoPE key (d_h^R = 64), which is 576 numbers. The chapter says "2 · d_c = 1024".
   - **The comparison.** Redone with the right numbers: 61 × 576 × 2 B × 131,072 ≈ **9.2 GB** for MLA versus ≈ 32.7 GB for GQA-8 at 128K, a ratio of about 3.6×. The chapter says "≈15 GB" and "2×".
   - **FP8.** V3 "adopt[s] the E4M3 format on all tensors". There is no E5M2 for activation gradients. Chapter 10 repeats this error.

   Source: DeepSeek-V3 Technical Report, arXiv:2412.19437 (v1 2024-12-27), §3.3 and §4.2.

7. **WRONG-NOW** · §5.13.3 `:718–725`, §5.16 point 7 `:821` · ">1M: only a few models (Gemini 1.5 Pro) have demonstrated this" / "Long context is solved with YaRN, mostly." At least these open-weight models now support 1M-class context:
   - MiniMax-M1: native 1M;
   - Qwen2.5-1M (Jan 2025);
   - Llama 4 Scout: 10M;
   - Qwen3-Next and Qwen3.5: 262,144 native, "extensible up to 1,010,000";
   - DeepSeek-V4: 1M, trained up to 1M without YaRN;
   - Kimi K3: 1,048,576.

   At the 1M scale, the recipe is an attention-architecture change plus native long-sequence training, not YaRN. Source: Qwen2.5-1M Technical Report, arXiv:2501.15383, 2025-01-26, plus the other sources above.

8. **WRONG-NOW** · §5.12 `:647`, `:652–655`, `:698`, `:808`, `:820` · "MTP, introduced in DeepSeek-V3" / "The additional heads are cheap — each is a single linear layer" / "At inference ... the rest are discarded. So MTP is a training-time-only change."
   - MTP predates V3. Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction", Meta, arXiv:2404.19737, 2024-04-30; V3 and V4 both cite it.
   - V3's MTP modules are sequential transformer blocks, not linear heads.
   - V3 says the modules "can also [be] repurpose[d] ... for speculative decoding", so MTP is not training-only.
   - MTP spread in 2025–26, mostly as a speculative-decoding draft:
     - GLM-4.5: "an MoE layer as the MTP layer to support speculative decoding";
     - Qwen3-Next: MTP "accelerates inference";
     - Qwen3.5: "MTP: trained with multi-steps";
     - V4: MTP depth 1, loss weight 0.3 → 0.1 at LR decay.
   - "+1–2% on most evals" (`:655`, `:808`) was not found in V3. Treat it as unverified.

9. **DATED-FRAMING** · §5.10–5.11, §5.15 `:807`, §5.16 point 5 · MoE is still framed as "DeepSeekMoE vs Mixtral". The 2025–26 axis is **sparsity**:
   - **Kimi K2.** A "sparsity scaling law" motivated 384 experts, versus 256 in V3. K2 also cut attention heads to 64 and uses 1 dense layer versus V3's 3.
   - **Very high sparsity** in later models: Qwen3-Next with 512 experts and 10 active; V4-Pro with 384 experts and 6 active; Kimi K3 with 896 experts and 16 active.
   - **Hash routing** replaces the first dense layers in V4.
   - **Gating function.** V4 changes affinity from Sigmoid to Sqrt(Softplus).
   - **Load balancing is aux-free plus a tiny sequence-wise loss.** V3 already had this "complementary sequence-wise balance loss", which the chapter never mentions. GLM-4.5 uses a 0.0001 weight. V4 also uses 0.0001.
   - **MXFP4 expert weights** are now shipped: gpt-oss post-trained MoE weights in MXFP4; V4 uses FP4 QAT for experts; K3 uses MXFP4/MXFP8 QAT.

   Sources:
   - Kimi K2, arXiv:2507.20534, 2025-07-28, §2.3 and Table 2.
   - DeepSeek-V4 §2.1 and §4.2.1.
   - GLM-4.5 §2.4.
   - gpt-oss model card §2.

10. **WRONG-NOW (never matched source)** · §5.14.4 `:785` · "Mixtral uses the original MHA ... *not* GQA". The Mixtral paper's Table 1 lists `n_heads 32, n_kv_heads 8`, so it is GQA. Source: Mixtral of Experts, arXiv:2401.04088.

11. **DATED-FRAMING** · §5.3.2 `:319`, §5.15 `:811` · "The frontier has converged on GQA (dense) and MLA (large MoE)" / "converging on MLA or GQA ... YaRN for context extension". The 2026 picture has no convergence:
   - DeepSeek left dense MLA for CSA/HCA (V4);
   - Moonshot kept MLA but made 3 of every 4 layers KDA (K3);
   - Qwen moved to Gated DeltaNet hybrids;
   - MiniMax went hybrid (M1), then back to full attention (M2);
   - GLM-4.5 uses GQA with 96 heads at hidden size 5120, then added DSA in GLM-5 (arXiv:2602.15763, 2026-02-17).

12. **MISSING-MINOR** · §5.13.2 `:716` · "Hybrid attention ... Used by Gemini 1.5 and Claude." Neither lab has published this. Replace it with the published examples above (Gemma 3/4, gpt-oss, Llama 4).

13. **MISSING-MINOR** · §5.14 overall · The four case-study configurations (V3, Llama-3-70B, Qwen3, Mixtral) are all from 2023 to mid-2025. I suggest replacing Mixtral with one hybrid-attention configuration (Qwen3.5-397B-A17B or Qwen3-Next-80B-A3B, both fully specified in their model cards). I also suggest adding DeepSeek-V4-Flash, fully specified in §4.2.1 of its report.

### Keep as-is
- §5.2 standard block and code (except `:122`; see the unverified leads).
- §5.3.1 MHA/MQA/GQA/MLA mechanics and the `MLAAttention` sketch.
- §5.4 RoPE math.
- §5.6 RMSNorm and §5.7 SwiGLU.
- §5.8 the MoE compute/parameter table and the `MoEBlock` code.
- §5.9 Switch aux loss and capacity.
- §5.10 DeepSeekMoE mechanism and bias controller. This is still current: V4 keeps DeepSeekMoE "with only minor adjustments" and the bias update speed is still 0.001.
- §5.14.2 Llama-3-70B as the canonical dense GQA configuration, framed as history.

### Lab opportunity
**`lab05_hybrid_attention`** (CPU, a few minutes). Train four tiny models on multi-query associative recall (MQAR):
1. full softmax attention;
2. a linear attention with the gated delta rule (the Gated DeltaNet/KDA family);
3. sliding-window attention;
4. a 3:1 linear:full hybrid.

Sweep the number of key-value pairs and measure recall against state or KV bytes per token. The expected surprise: the fixed-size linear state fails at a sharp capacity cliff, and a single full-attention layer in four restores near-full recall at about a quarter of the KV cost. That is exactly the bet Qwen3.5 and Kimi K3 made. An optional section 2 would add a gpt-oss-style learned sink logit and measure the attention mass on token 0 with and without it. This fits the style of the existing labs: a measurable number that shows a mechanism, not a quality ranking.

### References to add
- Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction", Meta FAIR, arXiv:2404.19737, 2024-04-30.
- Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388, 2025-05-14 (already [6]; fix the text that cites it).
- Gemma Team, "Gemma 3 Technical Report", arXiv:2503.19786, 2025-03-25.
- Gemma Team, "Gemma 4 Technical Report", arXiv:2607.02770, 2026-07-02; Gemma 4 model card, https://ai.google.dev/gemma/docs/core/model_card_4.
- OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card", arXiv:2508.10925, 2025-08-08.
- Meta AI, "The Llama 4 herd", blog, 2025-04-05, https://ai.meta.com/blog/llama-4-multimodal-intelligence/.
- Qiu et al. (Qwen), "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", arXiv:2505.06708, 2025-05-10.
- Yuan et al. (DeepSeek), "Native Sparse Attention", arXiv:2502.11089, 2025-02-16.
- DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open LLMs", arXiv:2512.02556, 2025-12-02.
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348 (page says submitted 2026-04-26); model card https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro.
- Xie et al. (DeepSeek), "mHC: Manifold-Constrained Hyper-Connections", arXiv:2512.24880, 2025-12-31.
- Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv:2507.20534, 2025-07-28.
- Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv:2510.26692, 2025-10-30.
- Moonshot AI, Kimi K3 model card, 2026-07, https://huggingface.co/moonshotai/Kimi-K3.
- Qwen, Qwen3-Next-80B-A3B model card, 2025-09-09; Qwen3.5-397B-A17B model card, 2026-02-16.
- MiniMax, "MiniMax-M1", arXiv:2506.13585, 2025-06-16; MiniMax, "Why Did M2 End Up as a Full Attention Model?", 2025-10-30; MiniMax, "MiniMax Sparse Attention", arXiv:2606.13392, 2026-06-11.
- Zhipu/Z.ai, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", arXiv:2508.06471, 2025-08-08; "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763, 2026-02-17.
- NVIDIA, "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models", arXiv:2504.03624, 2025-04-04 (the hybrid SSM example).

---

## Ch 9 — Mid-training: long context, domain annealing

**Verdict:** Needs a refresh. The rewrites are §9.3 (progressive recipe), §9.6 (schedules), §9.10's "frontier labs use YaRN" framing, §9.11's "compute is unavoidable" framing, and §9.15. There should also be a new section on mid-training as a *capability* stage (reasoning, agentic, repo-level code). The mechanics sections (§9.8–9.10 math, §9.9) age well.

### Findings (ranked most-severe first)

1. **WRONG-NOW (never matched source)** · `09-mid-training.md:21`, `:27`, `:123`, `:460` · "Qwen3 describes a 32K → 128K long-context extension phase" / "Qwen3 (32K → 128K)" / "typically trained at 8K context (Llama-3, Qwen3)". Qwen3's pre-training was 4,096 tokens (S1), then a reasoning stage (S2), then long context 4,096 → 32,768 (S3). 128K comes from YaRN + DCA at inference, and the RoPE base went 10,000 → 1,000,000 via ABF. Source: Qwen3 Technical Report, arXiv:2505.09388, §3.2.

2. **WRONG-NOW (never matched source)** · `:21`, `:29`, `:65`, `:131`, `:241` · Llama 3's recipe as described does not match its paper.
   - The paper extends context "gradually in six stages" from 8K to 128K using about 800B tokens.
   - Annealing is "the final 40M tokens" at 128K, with LR linearly annealed to 0 and high-quality sources upsampled. That is about 0.0003% of 15.6T, not "the last 5–10% of tokens".
   - The paper never mentions YaRN (0 hits), so "Llama-3 uses a YaRN-style rescaling for its 128K context" (`:241`) is unsupported.

   Source: "The Llama 3 Herd of Models", Meta, arXiv:2407.21783, §3.4.2–3.4.3.

3. **MISSING-MAJOR** · §9.1–9.2, §9.12 · Mid-training is now a named, capability-targeted stage with its own data (synthetic reasoning, instruction data, agentic trajectories, repo-level code), not just "same data, shifted mix". The chapter's premise "the data is still the same general pre-training data" (`:19`) is now DATED-FRAMING. Primary examples:
   - **GLM-4.5 §2.3.** "Mid-Training: Boost Reasoning & Agentic Capacity": repo-level code, then synthetic reasoning, then long-context & agent training. It uses "medium-size domain-specific datasets, including instruction data" and takes sequence length 4K → 32K → 128K. arXiv:2508.06471.
   - **Qwen3 S2 "Reasoning Stage"** upweights STEM, code and reasoning before the long-context stage.
   - **Llama 4.** Meta explicitly calls it "mid-training", with long-context extension "using specialized datasets", and it unlocked 10M context for Scout. Blog, 2025-04-05.
   - **OLMo 2.** The Dolmino Mix 1124 is "introduced via late-stage curriculum training (i.e. specialized data during the annealing phase)". Fully open data and checkpoints; the best public teaching example. arXiv:2501.00656.
   - **SmolLM3.** "We call the long context adaptation and reasoning adaptation 'mid-training'." It has a published three-stage mixture. Hugging Face blog, 2025.
   - **Kimi K2.** An annealing phase of 400B tokens at 4K, then 60B at 32K, then YaRN to 128K, with LR 2e-5 → 7e-6. It also uses rephrased synthetic data to raise "token utility". arXiv:2507.20534 §2.2 and §2.5.

4. **WRONG-NOW** · `:241`, `:442` (takeaway 4), `:147–152` (YAML `rope_rescaling: yarn`) · "YaRN ... is the standard for frontier long-context extension" / "Frontier labs use YaRN." In 2025–26 the field split into several recipes:
   - **ABF / raise the RoPE base** during a long-context stage. Qwen3 and GLM-4.5 both go 10,000 → 1,000,000, GLM-4.5 at the 32K switch.
   - **YaRN** is still used (V3; K2 for 128K; gpt-oss's dense layers to 131,072), often only as a final or inference-time step.
   - **Native progressive training to the target length** with a changed attention architecture. V4 goes 4K → 16K → 64K → 1M *inside* pre-training, with no YaRN. Sparse attention is introduced at 64K after a 1T-token dense warm-up.
   - **Partial or No-PE layers plus attention temperature scaling** (Llama 4 iRoPE; SmolLM3 NoPE + YaRN).
   - **LongRoPE2**, an evolutionary per-dimension rescale plus mixed-window training (Microsoft, arXiv:2502.20082, 2025-02-27).

   PI/NTK/YaRN are still the right *teaching* trio, but the chapter should present ABF as a fourth first-class method, and stop calling YaRN the frontier default.

5. **DATED-FRAMING** · §9.11 `:259`, `:267` · "The compute is unavoidable — ... there is no approximation that reduces this without changing the model." / "Frontier labs ... (c) use a more efficient attention." This is true as stated, but the 2025–26 frontier *did* change the model. That is now the main long-context story, not a footnote:
   - **DSA** was retrofitted by continued pre-training: a dense warm-up (indexer only), then a sparse stage of 15,000 steps × 480 × 128K = 943.7B tokens with top-k 2048 (V3.2 §2.1.1).
   - **CSA/HCA** from the start of long training (V4).
   - **Hybrid linear attention** (Qwen3-Next/3.5, MiniMax-M1, Kimi Linear/K3).

   Long-context mid-training is now also where an attention retrofit happens. That belongs in this chapter.

6. **MISSING-MAJOR** · §9.3 `:43–51`, §9.6 · The size of long-context phases grew by an order of magnitude, and the chapter's example schedules (5–20B tokens at 128K) understate it.
   - DeepSeek-V3: two YaRN phases of 1,000 steps each, 4K → 32K → 128K (V3 §4.3).
   - DeepSeek-V3.1: the 32K phase grew 10× to 630B tokens, and the 128K phase grew 3.3× to 209B. Source: DeepSeek-V3.1 model card, 2025-08-21, https://huggingface.co/deepseek-ai/DeepSeek-V3.1
   - Llama 3: about 800B tokens.
   - Present the V3 → V3.1 jump as the lesson: long-context quality is data-bound.

7. **MISSING-MINOR** · §9.4, §9.6 · Annealing now usually happens inside a **WSD** (warmup-stable-decay) schedule rather than at the tail of a cosine.
   - Kimi K2 uses WSD: a constant 2e-4 for 10T tokens, then decay.
   - V3 and V4 are effectively WSD-shaped: V3 holds 2.2e-4 until 10T tokens, then decays with a cosine; V4 holds 2.7e-4 "for most of the training".
   - GLM-4.5 is a useful counterpoint: it tested WSD and chose cosine.

   The chapter's "peak LR, full decay" bulk phase (`:15`) and "cosine" YAML (`:142`) misdescribe these runs.

8. **DATED-FRAMING** · §9.7 `:173` · "RULER is now the standard." Recent reports use newer long-context evals: V4 reports LongBench-V2 for base models, and V3.2 cites AA-LCR and Fiction.liveBench. RULER is still a good teaching benchmark, but it should not be called "the standard". The 2024 framing at `:55` and `:171` ("most 128K models of 2024 ...") can stay as history. Sources: V4 §4.3; V3.2 §2.2.

9. **WRONG-NOW (code bug, not staleness)** · §9.13 `:306–310` · `rescale_inv_freq_ntk` computes `base_new = inv_freq.max() ** (dim/(dim-2))`. `inv_freq.max()` is 1.0 (the i=0 term), so `base_new` = 1 and every frequency collapses to 1. It should use `base * (new/orig) ** (dim/(dim-2))`, as the text at `:250` states. The lab `lab09_rope_extension.py` may already implement this correctly; I did not check it against this snippet.

### Keep as-is
- §9.2 motivations 1–4.
- §9.5 domain continued pre-training (DeepSeek-Coder, DeepSeekMath, multilingual).
- §9.8 ring attention and context parallelism.
- §9.9 FlashAttention varlen.
- §9.10 PI/NTK/YaRN mechanics and the inv_freq math (add ABF alongside).
- §9.7's definition of effective versus advertised context.
- §9.14 roles.
- §9.12 as a bridge to reasoning. Its thesis is *strengthened* by GLM-4.5 and Qwen3 S2.

### Lab opportunity
Extend **`lab09_rope_extension`** with two sections.
1. **ABF.** Raise the base 10k → 1M, as Qwen3 and GLM-4.5 do, and measure it by position against PI, NTK and YaRN at the same 4× extension. The expected result is that ABF behaves like NTK-by-construction, and it needs the short fine-tune the lab already runs.
2. **One NoPE layer in four** (the Llama 4 and SmolLM3 idea). Measure length extrapolation with and without it.

Both run on CPU in seconds, and both fit the lab's "wavelength table predicts what breaks" framing.

A separate small lab would also fit: **`lab09_annealing_mix`**. Train a tiny LM with WSD, then compare decay phases on (a) the same mix and (b) a mix upweighted on a held-out "domain". It would measure how much of the domain gain comes from the decay alone versus the mix, the question behind OLMo 2's Dolmino and GLM-4.5's mid-training.

### References to add
- Groeneveld/Walsh et al. (AI2), "2 OLMo 2 Furious", arXiv:2501.00656, 2024-12-31 (rev. 2025-10-08).
- Qwen Team, "Qwen2.5-1M Technical Report", arXiv:2501.15383, 2025-01-26.
- Shang et al. (Microsoft), "LongRoPE2: Near-Lossless LLM Context Window Scaling", arXiv:2502.20082, 2025-02-27.
- Zhipu/Z.ai, "GLM-4.5", arXiv:2508.06471, 2025-08-08 (§2.3 mid-training).
- Kimi Team, "Kimi K2", arXiv:2507.20534, 2025-07-28 (§2.5 annealing and long-context activation).
- Meta AI, "The Llama 4 herd", 2025-04-05.
- Hugging Face, "SmolLM3: smol, multilingual, long-context reasoner", blog, 2025, https://huggingface.co/blog/smollm3.
- DeepSeek-AI, DeepSeek-V3.1 model card, 2025-08-21, https://huggingface.co/deepseek-ai/DeepSeek-V3.1.
- DeepSeek-AI, "DeepSeek-V3.2", arXiv:2512.02556 (§2.1.1 continued pre-training for DSA).
- DeepSeek-AI, "DeepSeek-V4", arXiv:2606.19348 (§4.2.2: 4K→16K→64K→1M).
- Xiong et al. (Meta), "Effective Long-Context Scaling of Foundation Models" (the ABF paper, cited by Qwen3 as Xiong et al. 2023). I did not fetch this one separately; confirm its arXiv ID before adding.

---

## Ch 10 — Case study: DeepSeek-V3 end-to-end

**Verdict:** Needs a rewrite of §10.4 (architecture numbers), §10.7 (parallelism), §10.8.1 (FP8 formats), §10.9 (schedule), §10.10 (hours), §10.12 (post-training), §10.13 and §10.14. It also needs a new closing section on the V3 → V3.1 → V3.2 → V4 lineage.

**Recommendation on scope: keep V3 as the case study.** Reasons:
- V3 is still the most readable end-to-end report.
- Chapter 19 (R1) is built on V3-Base.
- V4 itself says "All other unspecified details follow the settings established in DeepSeek-V3", so V3 is the prerequisite for reading V4.

Add §10.16, "What changed by V4". Don't add a second whole-chapter case study. Kimi K2 would largely repeat V3's skeleton (K2 is explicitly "a similar design to DeepSeek-V3": 61 layers, MLA, hidden 7168, expert dimension 2048). Its unique lessons are MuonClip/QK-Clip, the sparsity scaling law and rephrasing, which fit better as a boxed contrast in §10.16 or in Ch 8 (optimizer).

### Findings (ranked most-severe first)

1. **WRONG-NOW** · `10-case-study-deepseek-v3.md:573` · "We expect the V4 report (when it lands) to fill in some of these gaps" and "as of early 2026".
   - DeepSeek-V4 Preview shipped with open weights and a technical report on 2026-04-24: V4-Pro at 1.6T total / 49B active and V4-Flash at 284B / 13B, both with 1M context, pre-trained on "more than 32T" tokens.
   - The report does *not* publish the data mix, the classifier or the kernels either. The chapter's prediction was half right: V4 open-sources a mega-kernel for fine-grained EP, but not the pre-training data pipeline.

   Sources: DeepSeek API news "DeepSeek V4 Preview Release", 2026-04-24, https://api-docs.deepseek.com/news/news260424/ ; model card https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro ; arXiv:2606.19348.

2. **WRONG-NOW (never matched source)** · §10.4, §10.7, §10.8.1, §10.9, §10.10, §10.12 · The chapter's V3 facts contradict the V3 report it cites. Each item below was checked against arXiv:2412.19437.
   - **Attention.** `:88`, `:117–155`, `:161`, `:346` say "128 Q heads, 4 KV heads (GQA-like inside MLA)". MLA has no KV-head grouping. The report sets n_h = 128 and d_h = 128, and caches the 512-d latent plus the 64-d decoupled RoPE key. The `MultiHeadLatentAttention` code with `n_kv_heads=4` and `repeat_interleave` teaches GQA, not MLA, and has no decoupled RoPE path. It also contradicts Ch 5's correct `MLAAttention`.
   - **Layers.** `:86`, `:332`, `:343` say 60. The report says "number of Transformer layers to 61". The 15-layers-per-stage arithmetic at `:332` is therefore also wrong.
   - **Dense layers.** `:253`, `:353`, `:571` say "the first transformer layer is dense ... The exact number ... not published". The report says "We substitute all FFNs except for the first three layers with MoE layers". Kimi K2's Table 2 also lists V3 with 3 dense layers.
   - **Parallelism.** `:328–335`, `:356–361`, `:375`, `:580` give "4-way PP × 8-way EP × 64-way DP" with EP inside one node. The report says "16-way Pipeline Parallelism, 64-way Expert Parallelism spanning 8 nodes, and ZeRO-1 Data Parallelism". The chapter's claim that the "all-to-all stay[s] within the node" (`:333`, `:392`) is the opposite of V3's design: node-limited routing across IB, with NVLink forwarding.
   - **FP8 formats.** `:367–369`, `:400–409`, `:549`, `:579` split E4M3 and E5M2 by pass. The report "adopt[s] the E4M3 format on all tensors", specifically in contrast to the E4M3/E5M2 hybrid.
   - **Hours.** `:496` says "2,788K H800 GPU-hours of useful training". 2,788K is the *total*: pre-training 2,664K + context extension 119K + post-training 5K (report Table 1). The utilization inference drawn from it is invalid.
   - **Schedule.** `:479–483`, `:362–364` say cosine from peak, global batch 8,192 and about 18,000 steps. The report: constant 2.2e-4 until 10T tokens; cosine to 2.2e-5 over 4.3T tokens; 2.2e-5 for 333B tokens; 7.3e-6 for the final 167B. Batch size ramps from 3,072 to 15,360 sequences over the first 469B tokens.
   - **Post-training.** `:536–539` say "~2M" SFT examples, "RLHF (PPO)" and "Distillation into DeepSeek-R1". The report says 1.5M SFT instances; RL is **GRPO** (§5.2.2); and §5.4.1 is "Distillation *from* DeepSeek-R1" into V3. The direction is reversed in the chapter.
   - **MTP.** `:265` says "discarded at inference" (the report adds that the modules can be repurposed for speculative decoding). `:273` says "The DPO signal ... R1's reasoning traces are, in effect, an MTP-style rollout". I found nothing in the V3 or R1 reports supporting this; delete it. `:553` repeats it.
   - **Data.** `:563` gives "approximately 60% English, 30% Chinese, 10% other". I found no such figure in the V3 report (0 hits for "60%").
   - **Reliability.** `:494`, `:565` quote "MTBF ... 'around several hours'". I found no MTBF statement in the report. What the report does say is "did not experience any irrecoverable loss spikes or perform any rollbacks". `:504` attributes stability to FP8 and aux-free balancing; that attribution is not in the report.

3. **MISSING-MAJOR** · new §10.16 · The V3 lineage, which is the natural 2026 continuation of this chapter:
   - **V3.1** (2025-08-21). Hybrid thinking/non-thinking in one model. The long-context phases grew to 630B tokens at 32K and 209B at 128K. Trained with the UE8M0 FP8 scale format. Source: model card.
   - **V3.2-Exp and V3.2** (2025-12-02). The only architectural change from V3.1-Terminus is DSA, retrofitted by continued pre-training. Also scaled GRPO ("Keep Routing", "Off-Policy Sequence Masking"), specialist distillation, and V3.2-Speciale. Source: arXiv:2512.02556.
   - **V4** (2026-04-24).
     - CSA + HCA replace dense MLA attention, still with a shared-KV/latent flavor (single-head KV entries of dimension 512, partial RoPE).
     - mHC residuals.
     - **Muon** for most parameters, with AdamW for embeddings, the head, RMSNorm and the mHC gates.
     - Hash-routed MoE replaces the dense first layers. Affinity changes to Sqrt(Softplus). "Anticipatory routing" computes routing from θ_{t−Δt} to kill loss spikes. V4 openly reports "notable instability challenges", a contrast with V3's clean run.
     - 32T+ tokens; sequence length 4K→16K→64K→1M during pre-training.
     - FP4 (MXFP4) QAT for experts and for the indexer QK path.
     - Post-training by specialist SFT + GRPO, then on-policy distillation.
     - Self-assessed as trailing "state-of-the-art frontier models by approximately 3 to 6 months".

   This section also resolves several items on the §10.14 "not published" list and updates §10.13's "what propagated".

4. **DATED-FRAMING** · §10.13 `:549–557`, §10.15 point 8 · "V3 techniques are now standard" needs checking one by one.
   - **Aux-loss-free balancing.** Adopted by GLM-4.5 ("loss-free balance routing", bias update rate 0.001) and by V4. But `:551` says "now used in Qwen3". Wrong: Qwen3 uses a global-batch balancing loss. "The Mistral large model" is unverified.
   - **FP8 training.** Adopted for Llama 4 Behemoth ("FP8 and 32K GPUs ... 390 TFLOPs/GPU"). K2 uses FP8 *storage* for insensitive activations. "Llama-3.1 onward ... adopted FP8" (`:549`) is unverified: the Llama 3 paper does not say pre-training used FP8.
   - **MTP.** Adopted by GLM-4.5, Qwen3-Next/3.5 and V4, mostly for speculative decoding.
   - **MLA.** Adopted by Kimi K2, Kimi Linear and Kimi K3 ("Gated MLA"). DeepSeek itself moved past dense MLA in V4.
   - **Missing entirely:** the biggest post-V3 shift in DeepSeek's own stack, AdamW → Muon.

5. **DATED-FRAMING** · §10.1 `:13`, `:15` · "the most detailed public frontier report to date" / "behind the closed frontier ... in the same tier as 400B-class dense Western models". Write this in the past tense (true as of Dec 2024). As of 2026, detailed open reports include V4, Kimi K2, GLM-4.5 and Qwen3. The comparison set in `:9` (Llama-3.1-405B, Qwen-2.5-72B, Claude-3.5-Sonnet) should be labeled "at release".

6. **DATED-FRAMING** · §10.4.3 `:253` · "This 'first-K-dense, rest-MoE' pattern is now used in Qwen3 and other MoE models". The Qwen3 report does not state this; I did not confirm it. The pattern has since been *reversed* by V4, which uses hash-routed MoE instead of dense first layers. Kimi K2 uses 1 dense layer.

7. **MISSING-MINOR** · §10.6 `:285–324` · The auxiliary-loss-free section omits V3's "complementary sequence-wise balance loss" (a tiny α). V4 keeps it at weight 0.0001. The claim at `:285` that the aux loss is removed entirely is therefore overstated.

8. **MISSING-MINOR** · §10.11 `:526` · "~16,000 H100-hours × 30M H100-hours of compute" is garbled; it multiplies two numbers with the same unit. Rewrite it or drop it. I did not verify a Llama-3.1-405B GPU-hour figure from a primary source.

### Keep as-is
- §10.2's honest "what is not published" framing.
- §10.3 tokenizer (byte-level BPE, 128K vocab; matches the report).
- §10.5's intuition for why MTP helps.
- §10.6 bias-controller mechanism and pseudocode. It is still current: V4 keeps bias update speed 0.001.
- §10.7.1 DualPipe concept.
- §10.8.2 fine-grained 128-element block quantization (matches the report's tile/block-wise scaling).
- §10.11 cost-claim caveats. They are well done; only the "useful training" line in §10.10 is wrong.
- §10.15's structure.

### Lab opportunity
There is currently no Ch 10 lab. I propose **`lab10_kv_cache_lineage`** (pure arithmetic plus tiny tensors, CPU, seconds). It would build the per-token KV-cache and attention-FLOP accounting for:
- Llama-3-70B GQA-8;
- V3 MLA (576 numbers per token per layer);
- V3.2 DSA (MLA cache plus the indexer keys, and top-k 2048 attended);
- V4 CSA/HCA (m=4 / m′=128, FP8 non-RoPE dimensions, BF16 RoPE dimensions).

Plot them from 4K to 1M. The lab would verify V4's claims ("~2% of BF16 GQA8", "10% of V3.2's KV cache") from the configurations alone. As a side effect, the lab corrects the chapter's own 15 GB figure: the right number is about 9.2 GB. The surprising result: at 1M tokens, MLA alone does not make long context cheap. The attended-token count (top-k) is what bends the curve.

### References to add
- DeepSeek-AI, DeepSeek-V3.1 model card, 2025-08-21, https://huggingface.co/deepseek-ai/DeepSeek-V3.1.
- DeepSeek-AI, DeepSeek-V3.2-Exp model card, https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp; "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", arXiv:2512.02556, 2025-12-02.
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348. The arXiv page reads "Submitted on 26 Apr 2026", although a 2606 ID normally means June; cite the model card and the 2026-04-24 API news as the release anchor.
- Xie et al., "mHC: Manifold-Constrained Hyper-Connections", arXiv:2512.24880, 2025-12-31.
- Liu et al. (Moonshot), "Muon is Scalable for LLM Training", arXiv:2502.16982, 2025-02-24.
- Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv:2507.20534, 2025-07-28 (MuonClip/QK-Clip, for the contrast box).
- Roller et al., "Hash Layers for Large Sparse Models", 2021 (cited by V4 for hash routing; I did not fetch it, so confirm the ID).

---

## Unverified leads

These came up during research, but I could not confirm them from a primary source. Verify each one before using it.

- **Qwen3-235B-A22B hidden size and expert intermediate dimension** (the book says 4096 and 12288). Not in the report's Table 2, which lists only layers, heads, experts and context. Check the official `config.json`; I believe the expert intermediate size is much smaller than 12288, but I have not confirmed it.
- **Ch 5 `:122` "Most frontier models tie the input and output embeddings."** I did not verify this for V3, Llama 3 or Qwen3-235B. It is likely false for the large models (tying is typical only for small ones). Check the configs.
- **Ch 5 `:380` "The one paper that pushed RMSNorm hard is the one from the Gemma team."** RMSNorm is Zhang & Sennrich 2019. I found no support for the Gemma attribution.
- **Ch 9 `:237` "NTK-aware ... is the default in Hugging Face's PEFT library."** Not verified, and probably wrong (RoPE scaling lives in `transformers`' `rope_scaling`, not PEFT).
- **Ch 9 `:127` "Annealing + SFT in one run" hinted at by Qwen3 and V3.** I did not find this in either report. GLM-4.5's mid-training does mix instruction data, which is the nearest verified example.
- **Llama 3.1's actual RoPE scaling** (believed to be a custom frequency-dependent "llama3" rope type in the released configs, not YaRN). The paper does not describe it, and the configs are gated, so this is unverified.
- **Ch 10 `:551` "aux-free balancing ... used in the Mistral large model."** Not verified.
- **Ch 10 `:383` "20–30% wall-clock improvement" from DualPipe.** Not found in the V3 report.
- **Gemma 4 details beyond the model card** (the p-RoPE definition, the global:local ratio, "unified Keys and Values"). The model card confirms the terms; I did not read the Gemma 4 report (arXiv:2607.02770) itself.
- **Kimi K3 technical report.** I confirmed only the Hugging Face model card and the vLLM blog (2026-07-22). I did not find an arXiv report. Its "Attention Residuals", "Stable LatentMoE" and whether it used MuonClip are unverified beyond the card's one-line descriptions.
- **MiniMax-M2 / M2.1 / M2.5 architecture details** beyond the official "why full attention" post, and whether a formal M2 technical report exists. A secondary source (Raschka, 2026) refers to one; I did not find the primary.
- **GLM-5 architecture numbers** (744B total / ~40B active) come from secondary coverage. The arXiv abstract confirms only that GLM-5 "adopts DSA".
- **"Llama 4 Scout 10M context" in practice.** Meta's blog claims it; I found no independent primary long-context evaluation.
- **Hybrid SSM beyond Nemotron-H** (Jamba, Falcon-H1, IBM Granite 4.0). I did not fetch primary sources for these.
