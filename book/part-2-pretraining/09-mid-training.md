# Chapter 9: Mid-training — long context, domain annealing

> Reading time: ~40 minutes. By the end of this chapter you should understand the training stages that sit between full pre-training and SFT — long-context extension, domain annealing, and the multi-stage schedules frontier labs actually use. You should be able to read the "annealing" section of a frontier-lab technical report and not be mystified.

*Current as of early 2025.*

## 9.1 What is mid-training

A frontier pre-training run does not go from "cold start" to "ready for post-training" in one monotonic shot. Between the bulk of the pre-training and the start of SFT, there is a distinct, often undocumented phase that the field has settled on calling **mid-training**, and which papers variously refer to as *continued pre-training*, *annealing*, *the second stage*, or simply *the long-context phase*. Different labs draw the boundaries differently, but the *work* being done in that phase is recognizable across all of them.

The broad shape:

```
[Pre-training bulk]  →  [Mid-training]  →  [SFT / post-training]
   90–95% of tokens       5–10% of tokens      separate data, separate loop
   standard mix           shifted mix          much smaller data
   peak LR, full decay    lower LR, annealing  LR warmup + small SFT LR
   8K context (typical)   32K → 128K           8K–32K (varies)
```

The data is still the same general pre-training data — web, code, math, books, papers — but the *mix* has been deliberately shifted, the *context length* has often been extended, and the *learning rate* is in its annealing phase. The model is the same architecture; the training is just being tuned for the capabilities that post-training will need to inherit.

This phase is rarely announced as a separate "mid-training" stage. Llama-3 [\[5\]](../appendix/b-references.md#5-llama-3) splits it in two: a *long-context pre-training* stage and a short *annealing* stage, the final 40M tokens, where the learning rate decays to zero on a high-quality data mix ([fact sheet](../appendix/fact-sheets/llama-3.md#long-context-and-annealing)). DeepSeek-V3 [\[1\]](../appendix/b-references.md#1-deepseek-v3) talks about a "context length extension" stage separately from the bulk pre-training and the annealing. Qwen3 [\[6\]](../appendix/b-references.md#6-qwen3) describes a 32K → 128K long-context extension phase. The names differ; the work is the same.

## 9.2 Why mid-training exists

Four motivations, all of which apply in practice, and most of which apply in the same run.

**1. Long-context extension.** A frontier base model is typically trained at 8K context (Llama-3, Qwen3) or 4K. The market wants 128K, 256K, 1M. You cannot just *evaluate* a model at 128K if it was trained at 8K; the position embeddings have never seen those positions, and the attention patterns have never seen those distances. So you need a phase where the model trains on long sequences with the position embedding extended. This is the *RoPE extension* phase.

**2. Domain upweighting.** Pre-training is dominated by web text because that is the largest source. But the last few percent of pre-training is the highest-leverage compute you have: it is the data the model is "freshest" on, the data that biases the final loss. Frontier labs use this last chunk to upweight high-quality domains — books, papers, code, math — to bias the final model toward capabilities that the bulk mix underweights.

**3. Capability calibration.** A model straight out of pre-training is not the same as a model that has been SFT'd. The annealing phase lets the lab push the model toward the capabilities the post-training pipeline will need: reasoning, instruction-following shapes, long-context retrieval. Some of this is data, some of it is the lower learning rate, some of it is the "settling" that happens at the bottom of the cosine.

**4. Decoupling from the big run.** Pre-training a 70B+ model is a 60-day, $5M+ commitment. You do not want to stop the bulk run because someone decided they want 128K context. Instead, you *finish* the bulk run at 8K, then run a separate mid-training pass on the side. The big run is a known quantity; the long-context extension is a shorter, lower-stakes run that can be iterated.

These four motivations stack. A typical mid-training pass might: extend the context from 8K to 128K, upweight code and math, decay the learning rate to 10% of peak, and run for 5–10% of the original token budget. The DeepSeek-V3 paper describes exactly this kind of multi-stage approach.

## 9.3 Long-context extension

The core problem: a model trained at 8K context has only ever seen positions 0 through 8191. If you try to evaluate it at 32K or 128K, the RoPE frequencies at the new positions are out-of-distribution. The model has not learned to attend at those distances, the position embeddings are extrapolating rather than interpolating, and quality collapses.

The fix is to *continue training* at the longer context, with the position embedding modified so that the new positions look similar to the old ones. The three standard modifications — Position Interpolation, NTK-aware, and YaRN — are covered in §9.10. For now, the operational shape of the long-context extension phase.

**Progressive context length training.** The standard recipe is *progressive*: extend the context in stages rather than jumping from 8K to 128K in one step.

```
Stage 1: 8K context,  ~50B tokens, normal RoPE
Stage 2: 32K context, ~20B tokens, RoPE rescaled
Stage 3: 128K context, ~5B tokens, RoPE rescaled
```

Each stage costs less than the previous (the data is the same, the model is the same, only the sequence length and the position embedding change). The reasoning is that training at 128K directly from 8K is unstable — the model has to learn a huge new distribution in one step. Progressive training gives the model a smoother path.

In practice, the stages are not as cleanly separated as the diagram suggests. The data is a single continuous stream, and the context length is just a hyperparameter of the data loader. The "stages" are just points in the run where the hyperparameter changes. From the outside, the run is one training job; from the inside, it is three or four configurations.

**The quality vs length trade-off.** Extending context is not free. Even with a good RoPE rescaling, the perplexity at the new long context is typically slightly higher than at the original short context. The trade-off is not "0 cost, 128K for free" — it is "10% perplexity hit at 8K, full quality at 32K, slight degradation at 128K." Llama-3-8B, extended to 128K, shows a small but real perplexity increase at the original 8K context, in exchange for working (sort of) at 128K. RULER [\[49\]](../appendix/b-references.md#49-ruler) showed that most "128K context" models of 2024 actually degrade past 32K on the harder sub-tasks.

The honest framing: long-context extension buys you a *usable* long context at some cost in short-context quality. For long-document summarization, code-repo understanding, multi-file reasoning, the trade is worth it. For tasks that are already short-context (most chat, most Q&A), the extension is irrelevant.

## 9.4 The data mix change — annealing

The other big thing mid-training does is change the data mix. This is the part Llama-3 calls *annealing*.

The motivation: pre-training is dominated by web text because that is the largest source. But the *last* few percent of pre-training is disproportionately influential — the model is "fresh" on this data, and the loss at the end of training is biased toward whatever the final mix was. Frontier labs use this last chunk to upweight high-quality, capability-relevant domains.

The Llama-3 paper [\[5\]](../appendix/b-references.md#5-llama-3) reports that its annealing phase adjusts the mix to upsample data sources of very high quality, and separately that annealing on small amounts of high-quality code and math boosts benchmark scores for the 8B model, though negligibly for the 405B. It does not publish the annealing ratios. The learning rate is decayed linearly to zero over this phase, which is the literal "annealing." Llama 3's annealing is also much shorter than "the last few percent": 40M tokens, about 0.0003% of the 405B model's 15.6T. The part of its run that is a few percent of the budget is the long-context stage, about 800B tokens ([fact sheet](../appendix/fact-sheets/llama-3.md#long-context-and-annealing)).

A representative annealing mix. The left column is Llama-3's published bulk mix ([fact sheet](../appendix/fact-sheets/llama-3.md#pre-training-data)); the right column is illustrative, because Llama-3 has not published its annealing ratios:

```
# ILLUSTRATIVE (right column only): not any lab's annealing mix
Bulk pre-training (Llama 3):   Annealing phase (illustrative):
  50% general knowledge          30% general knowledge
  25% math and reasoning         40% math and reasoning
  17% code                       22% code
   8% multilingual                8% multilingual
                                 + highest-quality sources upsampled within each domain
```

The reason annealing matters is that the *final* loss surface is what the model ships with. A model that ends training on a mix biased toward code will be better at code than one that ended on a uniform mix. A model that ends on math-rich data will be better at math. The same total compute, with the same architecture, can produce meaningfully different downstream capabilities depending on the last 5% of training data.

Frontier labs iterate on the data mix during the run, not just at the start. A capability inversion — a model improving on math but plateauing on web — is a signal to upweight math in the remaining budget. By the annealing phase, the budget is small (5–10% of total) but the leverage is high.

## 9.5 Domain-specific continued pre-training

The same "continue pre-training on a focused corpus" technique is used to specialize a general model. Three worked examples.

**DeepSeek-Coder.** DeepSeek released DeepSeek-Coder-V2 (and the original DeepSeek-Coder) as a code-specialized model. The recipe: take a general base model, continue pre-training on a large, high-quality code corpus (filtered GitHub + synthetic code), then SFT. The result is dramatically better at code than the general base, at the cost of some general capability degradation. This is the "specialist model" path.

**DeepSeekMath.** The DeepSeekMath paper (which also introduced GRPO [\[15\]](../appendix/b-references.md#15-grpo)) reports continuing pre-training of a base model on a math-heavy corpus — competition math, proof-based content, synthetic math problems — with the explicit goal of improving mathematical reasoning.

**Multilingual continued pre-training.** A general model that under-serves a target language (Vietnamese, Arabic, etc.) can be improved by continuing pre-training on a large corpus of that language. The vocabulary and general capabilities stay, but the model is given more in-domain data. This is the standard recipe for producing a strong regional model from a general base.

A worked example, in pseudocode:

```python
# Domain-specific continued pre-training
# Take a general base model, continue on a focused corpus

base = load_pretrained("deepseek-v3-base")
domain_corpus = load_corpus("github-code-filtered")  # 500B tokens
optimizer = AdamW(base.parameters(), lr=2e-5, weight_decay=0.1)
lr_schedule = cosine_with_warmup(optimizer, warmup=200, total=50_000)

for step, batch in enumerate(domain_corpus):
    input_ids, labels = batch
    loss = forward(base, input_ids, labels)
    loss.backward()
    clip_grad_norm_(base.parameters(), 1.0)
    optimizer.step()
    lr_schedule.step()
    if step % 1000 == 0:
        eval_on_humaneval(base)  # watch code capability
        eval_on_mbpp(base)
```

The hyperparameters that matter: the learning rate is much lower than pre-training (1/10 to 1/20 of peak), the data is single-domain, the evaluation is targeted. The model degrades slightly on the original distribution; the new distribution improves significantly.

## 9.6 Multi-stage training schedules

The frontier labs do not all run the same schedule. Three patterns are common.

**Two-stage.** Bulk pre-training at the original context length, then a long-context extension phase. Used by Qwen3 (32K → 128K). The mid-training is the long-context phase.

**Three-stage.** Bulk pre-training, then a long-context extension, then an annealing phase. Used by some labs for the largest runs. Llama-3 405B is the documented example: 8K → 128K in six stages over about 800B tokens, then annealing on the final 40M tokens ([fact sheet](../appendix/fact-sheets/llama-3.md#long-context-and-annealing)). Each stage has its own data mix, learning rate, and context length. The boundaries are operational checkpoints.

**Annealing + SFT in one run.** The Qwen3 and DeepSeek-V3 papers hint at a pattern where the annealing phase is followed directly by SFT, without a separate SFT run. The advantage is operational: the model does not have to be re-loaded, the data mix transition is smooth, the learning rate transitions naturally. The disadvantage is that SFT data is much smaller and the mix shift is large, so the "one run" pattern requires careful handling of the data loader.

The DeepSeek-V3 paper [\[1\]](../appendix/b-references.md#1-deepseek-v3) describes a multi-stage training run with explicit context-length extensions (the 14.8T-token training included separate phases at different context lengths) and a final annealing phase. The exact boundaries are not all published, but the shape is clear.

The Llama-3 paper [\[5\]](../appendix/b-references.md#5-llama-3) describes the annealing phase briefly but concretely: over the final 40M tokens, at 128K context, the data mix is shifted toward very high-quality sources, the learning rate is decayed linearly (not cosine) to zero, and the released base model is an average of checkpoints taken during annealing ([fact sheet](../appendix/fact-sheets/llama-3.md#long-context-and-annealing)). It also uses short annealing runs as a cheap way to measure the value of a new dataset. The Llama-3 paper's annealing is one of the more public descriptions of the technique.

A typical schedule, in YAML:

```yaml
# ILLUSTRATIVE: a representative three-stage schedule, not from a specific lab
stages:
  - name: bulk_pretraining
    tokens: 14_000_000_000_000   # 14T
    context_length: 8192
    learning_rate: 3.0e-4
    lr_schedule: cosine
    warmup_steps: 2000
    data_mix: standard_pre_training
    
  - name: long_context_extension
    tokens: 200_000_000_000      # 200B
    context_length: 32768        # then 131072 in a sub-stage
    learning_rate: 3.0e-5         # 10% of peak
    lr_schedule: linear_decay
    data_mix: standard_with_more_long_docs
    rope_rescaling: yarn
    
  - name: annealing
    tokens: 500_000_000_000      # 500B
    context_length: 131072
    learning_rate: 3.0e-6         # 1% of peak
    lr_schedule: linear_to_zero
    data_mix: high_quality_books_papers_code_math
    eval_frequency: 1000
```

This is roughly the shape that frontier labs run. The exact numbers vary; the structure is consistent.

## 9.7 The long-context evaluation

You cannot improve what you cannot measure. Long-context evaluation is its own subfield.

**Needle-in-a-haystack (NIAH).** The original test, popularized by Greg Kamradt in 2023, and formalized in RULER [\[49\]](../appendix/b-references.md#49-ruler). The setup: insert a single "needle" sentence (a random fact) into a long context at a specific position, then ask the model to recall it. Measure accuracy as a function of (1) the context length and (2) the needle position. The result is a heatmap: a 2D plot with context length on one axis and needle position on the other, color-coded by accuracy.

A good long-context model has a uniformly green heatmap. A bad one has green only at the bottom-left (short context, beginning of the document). Most "128K context" models of 2024 have a heatmap that is mostly green up to 32K, mixed from 32K to 64K, and red past 64K.

**RULER.** The RULER benchmark extends NIAH to 13 task categories: NIAH variants (single needle, multi-needle, needle with distractors), multi-hop tracing, aggregation, question answering, and more. RULER showed that NIAH is the easiest of the long-context tasks — a model that passes NIAH at 128K might still fail the harder RULER sub-tasks at 32K. RULER is now the standard.

**LongBench.** A benchmark for long-context understanding in Chinese and English, with tasks like multi-document QA, code understanding, and summarization. Used heavily by Chinese labs (Qwen, DeepSeek, Moonshot).

**LEval.** A long-context benchmark focused on "real" long documents (academic papers, legal contracts, books) rather than synthetic needle insertion. Harder than NIAH, more representative of production use cases.

The metrics that matter:

- **NIAH score at 128K** — does the model find a needle at any position in a 128K context?
- **RULER composite** — weighted average across the 13 sub-tasks.
- **Effective context length** — the longest context at which the model achieves ≥95% of its short-context performance on a held-out task. This is the honest number; "128K context" usually means "32K effective context."

A real NIAH eval script is in §9.13.

## 9.8 Ring attention for long context

Training at long context (≥32K, definitely ≥128K) hits a memory wall: the attention matrix is O(N^2), and for a 128K context with reasonable batch size, the activations blow past GPU memory. FlashAttention (next section) reduces the *memory* to O(N), but the *compute* is still O(N^2), and the *sequence* still has to fit in some GPU's memory.

The fix is **context parallelism**: split the sequence across multiple GPUs, with each GPU holding a different slice. The challenge is that attention requires every query position to attend to every key position, so the GPUs need to exchange keys and values. The standard implementation is **ring attention** [\[38\]](../appendix/b-references.md#38-ring-attention), introduced by Liu et al. in 2023.

The idea: arrange the GPUs in a ring. Each GPU holds 1/P of the sequence. The K and V tensors are passed around the ring; each GPU computes its local Q × K^T, runs softmax (over the local K/V plus the received K/V), then accumulates the result. After P steps, every GPU has computed its full attention output.

The bandwidth cost is O(P) communications per step, each of size O(N × d / P) per GPU, so the total communication is O(N × d) per step, independent of P. The compute is still O(N^2 / P) per GPU, but the *peak memory* per GPU is O(N / P) for the activations. For a 128K context split across 8 GPUs, each GPU holds 16K positions, and the attention memory drops by 8x.

Ring attention composes naturally with tensor and pipeline parallelism. A 4D parallelism layout might be PP × TP × CP × DP, with CP being context parallelism for the long-context phase. The same run that does bulk pre-training at 8K can switch to long-context training at 128K by turning on CP and increasing the sequence length.

The standard implementation lives in the `ring_flash_attn` library (a fork of FlashAttention that adds the ring communication primitive). The kernel work is non-trivial: the ring communication must overlap with the attention compute, and the softmax is over a *partial* set of keys, which means the algorithm is a "streaming softmax" that maintains running max and sum statistics.

## 9.9 FlashAttention for long context

FlashAttention [\[19\]](../appendix/b-references.md#19-flashattention) is the IO-aware exact attention algorithm that tiles the attention computation to fit in SRAM, avoiding the O(N^2) memory cost of materializing the full attention matrix. For long context, FlashAttention is the prerequisite for any further optimization: without it, a 128K context does not fit in GPU memory at all.

The variant that matters for mid-training is **FlashAttention's varlen mode** (`flash_attn_varlen_func`). In a standard training run, all sequences in a batch are padded to the same length (e.g., 8K). The padding is wasted compute. In a long-context training run, the documents are of very different lengths, and padding to 128K would be ruinous — most documents are 1K–4K, padding them to 128K wastes 32x–128x compute.

The fix is `varlen`: concatenate all documents in a batch into one long sequence, pass the sequence along with `cu_seqlens` (cumulative sequence lengths) to FlashAttention, and let the kernel handle the boundaries via the attention mask internally. No padding, no wasted compute, exact same attention outputs as the padded version.

A worked example:

```python
# Without varlen: pad every doc to max_len in batch
padded_batch = [pad(doc, max_len=131072) for doc in docs]  # terrible
attn_out = flash_attn(padded_batch)  # 128x wasted compute on a 1K doc

# With varlen: concatenate, pass cu_seqlens
flat_input = torch.cat(docs)            # shape: [total_tokens, hidden]
cu_seqlens = torch.tensor([0, len(docs[0]), len(docs[0])+len(docs[1]), ...])
attn_out = flash_attn_varlen_func(flat_input, cu_seqlens=cu_seqlens, max_seqlen=131072)
# no padding, exact attention
```

Varlen is not optional for long-context training. Without it, the training loop is dominated by padding compute. With it, the training loop is bounded by the *actual* token count, which is typically 5–10x smaller than the padded count.

## 9.10 Position Interpolation vs YaRN vs NTK-aware

The three main RoPE extension methods, in chronological order of practical adoption.

**Position Interpolation (PI).** The Chen et al. 2023 paper [\[47\]](../appendix/b-references.md#47-position-interpolation). The simplest method: linearly downscale the position indices. If your model was trained at 8K and you want 128K, you multiply every position by 8K/128K = 1/16, so position 128K in the new model looks like position 8K in the old one. The RoPE frequencies are unchanged, but the position values are interpolated, not extrapolated. Fine-tune for a few thousand steps, and the model learns the new positions.

PI is simple and works, but it has a quality cost: the *resolution* of the position embedding is reduced by 16x. Near positions that were 1 apart in the original are now 1/16 apart, so the model has trouble distinguishing them. For tasks that need fine position discrimination (code with indentation, character-level tasks), PI hurts.

**NTK-aware scaling.** A 2023 technique from the Reddit / LocalLLaMA community, attributed to bloc97 [\[48\]](../appendix/b-references.md#48-ntk-aware-scaling-bloc97-2023). The observation: in RoPE, the *low-frequency* components encode long-range position and the *high-frequency* components encode short-range position. If you only need to fix long-range position, you should not change the high frequencies.

NTK-aware scaling raises the RoPE base frequency (`base` in the standard formulation) so that the *highest* frequency in the new model equals the *highest* frequency in the old model. The result: low frequencies are preserved (good for long-range), high frequencies are scaled (the model can still distinguish nearby positions, but with reduced resolution). Fine-tune, and you get long context with less quality loss than PI.

NTK-aware is widely used because it is a single-line change to the RoPE code and works well in practice. It is the default in Hugging Face's PEFT library for long-context extension.

**YaRN.** The Peng et al. 2023 paper [\[35\]](../appendix/b-references.md#35-yarn). The most sophisticated of the three. YaRN observes that different RoPE frequencies need different treatment: very low frequencies (long-range) need to be interpolated (like PI), very high frequencies (short-range) need to be left alone, and the middle range needs a smooth transition. YaRN also rescales the attention logits by a factor of `1/t` where `t` is the length scaling, to compensate for the change in attention distribution.

In practice, YaRN gives the best quality-vs-length trade-off of the three methods, and is the standard for frontier long-context extension. Llama-3 uses a YaRN-style rescaling for its 128K context. Qwen3 uses a similar approach.

The RoPE math, briefly. Standard RoPE computes the inverse frequencies as

$$\text{inv\_freq}_i = \frac{1}{\text{base}^{2i / d}}$$

for $i = 0, 1, \ldots, d/2 - 1$, with `base = 10000` (the original transformer) or `base = 500000` (Llama-3, larger for better resolution). For long-context extension:

- **PI** rescales positions: $m' = m \cdot (L_\text{orig} / L_\text{new})$.
- **NTK-aware** rescales the base: $\text{base}' = \text{base} \cdot (L_\text{new} / L_\text{orig})^{d / (d - 2)}$.
- **YaRN** rescales the base (like NTK-aware), adds a wavelength-dependent interpolation ramp, and rescales attention logits.

A PyTorch implementation of the inv_freq rescaling for PI and YaRN is in §9.13.

## 9.11 The cost of long context

The cost analysis is the part that gets glossed over in marketing. Long context is *expensive*.

**Compute.** Attention is O(N^2) in the sequence length N, for both memory and compute. The compute is unavoidable — every query attends to every key, and there is no approximation that reduces this without changing the model. FlashAttention reduces the *memory* to O(N) (by tiling, no full attention matrix materialized), but the *compute* is still O(N^2) FLOPs per attention layer.

For a single attention layer with sequence length N, hidden dim d, and h heads, the FLOPs are roughly $4 \cdot h \cdot N \cdot d$ for the projections plus $4 \cdot h \cdot N^2$ for the attention itself. The attention FLOPs dominate for $N > d$. For a 7B model with d=4096 and N=128K, the attention FLOPs are 4 × 32 × 128K^2 ≈ 2.1 × 10^{12} per token, compared to 4 × 32 × 128K × 4096 ≈ 6.7 × 10^{10} for the projections — the attention is 32x more expensive than the projections at 128K context.

**Memory.** The KV cache (for inference) is $2 \cdot N \cdot d$ per layer per token. For a 70B model with 80 layers, d=8192 (combined heads), and N=128K, the KV cache is 2 × 80 × 128K × 8192 × 2 bytes = 336 GB. This is why long-context inference is a separate engineering problem from short-context inference, and why KV cache compression (GQA, MLA) is so important.

**Arithmetic intensity.** The attention operation has a low arithmetic intensity (few FLOPs per byte of memory traffic) compared to matmul. FlashAttention's contribution is to fuse the attention with the softmax and the value multiplication, so the memory traffic is dominated by the SRAM-tile loads rather than HBM. This is what makes long-context attention tractable at all. Without FlashAttention, a 128K context on a single H100 would OOM.

**The practical cost.** A 70B model at 8K context processes a batch of, say, 4M tokens at a certain throughput. At 128K, the same batch is 16x longer, the attention compute is 256x larger (16^2), and the throughput per token drops by roughly 8–16x. Long-context training is not 16x more expensive; it is *more* than 16x more expensive, because the attention dominates. Frontier labs that train at long context either (a) train much smaller batches, (b) use much more compute, or (c) use a more efficient attention.

## 9.12 Mid-training for reasoning

The last topic, and the one that ties this chapter to Part III. A base model straight out of pre-training is a next-token predictor. The mid-training phase is where it starts to look like a model that can *reason*. The reasoning itself is the work of post-training — SFT on reasoning traces, GRPO on verifiable problems, the subject of Chapter 15 and Chapter 19 — but the *foundation* is laid in mid-training.

What mid-training does for reasoning:

- **Upweights math and code.** Reasoning is heavily correlated with math/code capability, and the annealing phase is the highest-leverage place to push this.
- **Extends context for long reasoning chains.** A 4-step reasoning chain fits in 1K tokens; a 50-step chain with self-critique fits in 8K; a 200-step chain with R1-style "aha moments" needs 16K+. Mid-training at long context is the prerequisite for long reasoning.
- **Teaches the model to use long context well.** A model that has only ever seen 8K context has never *learned* to use 32K context for anything. The long-context phase is where the model learns to attend across long distances, follow references in the middle of a long document, and ignore distractor information. This generalizes to reasoning.

The R1 case study in Chapter 19 is the cleanest public example. DeepSeek-R1-Zero is built by taking DeepSeek-V3-Base, doing *no SFT*, and running GRPO directly. The "aha moment" emergence happens at scale, and it depends on the model already being a strong reasoner from pre-training and mid-training. Without the mid-training phase that biased V3-Base toward math and code, the GRPO loop would have nothing to amplify.

The preview is short on purpose. Chapter 15 (GRPO) and Chapter 19 (R1) are the deep dives. The point of this section is: mid-training is the bridge. The model is shaped here for the post-training that comes next.

## 9.13 Code

Three pieces of code: a RoPE rescaling function, a progressive context-length training loop, and a needle-in-a-haystack eval.

**RoPE inv_freq rescaling (PI and YaRN-style).**

```python
import torch
import math

def get_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Standard RoPE inverse frequencies."""
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rescale_inv_freq_pi(inv_freq, original_max_pos, new_max_pos):
    """Position Interpolation: no change to inv_freq; positions are scaled at apply time.
    Returns the same inv_freq — the rescaling happens in the position ids.
    """
    scale = original_max_pos / new_max_pos
    return inv_freq, scale


def rescale_inv_freq_ntk(inv_freq, original_max_pos, new_max_pos, dim):
    """NTK-aware: raise the base so high frequencies are preserved."""
    base_new = inv_freq.max() ** (dim / (dim - 2))  # crude but standard
    # Re-derive inv_freq with the new base
    return 1.0 / (base_new ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rescale_inv_freq_yarn(inv_freq, original_max_pos, new_max_pos, dim, beta_fast=32, beta_slow=1):
    """YaRN: piecewise rescaling — leave high frequencies alone, scale low frequencies by 1/t."""
    t = new_max_pos / original_max_pos
    # Wavelength of each frequency component
    wavelengths = 2 * math.pi / inv_freq  # in tokens
    # Ramp: 1.0 for short wavelengths (no scaling), 1/t for long wavelengths (PI-like)
    ramp = torch.clamp(
        (wavelengths - beta_fast) / (beta_slow - beta_fast),
        min=0.0, max=1.0
    )
    # Apply ramp: high freq (small ramp) unchanged, low freq (large ramp) scaled by 1/t
    inv_freq_new = inv_freq / (1.0 - ramp + ramp / t)
    return inv_freq_new


def apply_rope_extention(model, method="yarn", original_max_pos=8192, new_max_pos=131072):
    """Apply RoPE extension to every attention layer in a model."""
    for layer in model.layers:
        if hasattr(layer.attn, "inv_freq"):
            d = layer.attn.head_dim
            inv_freq_orig = get_inv_freq(d, base=layer.attn.rope_base)
            if method == "pi":
                inv_freq_new, scale = rescale_inv_freq_pi(inv_freq_orig, original_max_pos, new_max_pos)
                layer.attn.rope_scale = scale
            elif method == "ntk":
                inv_freq_new = rescale_inv_freq_ntk(inv_freq_orig, original_max_pos, new_max_pos, d)
            elif method == "yarn":
                inv_freq_new = rescale_inv_freq_yarn(inv_freq_orig, original_max_pos, new_max_pos, d)
            else:
                raise ValueError(method)
            layer.attn.inv_freq = inv_freq_new.to(layer.attn.inv_freq.device)
    return model
```

**Progressive context-length training loop.**

```python
import torch
from torch.utils.data import DataLoader

# Schedule: short context first, then long
context_schedule = [
    {"seq_length": 8192,   "tokens": 50_000_000_000,  "lr": 3.0e-5},
    {"seq_length": 32768,  "tokens": 20_000_000_000,  "lr": 1.0e-5},
    {"seq_length": 131072, "tokens": 5_000_000_000,   "lr": 3.0e-6},
]

model = load_pretrained("base-model-8k")
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
global_step = 0

for stage in context_schedule:
    print(f"=== Stage: seq_length={stage['seq_length']}, tokens={stage['tokens']}, lr={stage['lr']}")
    # Update the model's max position embeddings
    model.config.max_position_embeddings = stage["seq_length"]
    # Rebuild the dataloader with the new sequence length
    dataloader = build_dataloader(stage["seq_length"], use_varlen=True)
    for batch in dataloader:
        loss = forward(model, batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        # Linear LR schedule within the stage
        for pg in optimizer.param_groups:
            pg["lr"] = stage["lr"] * (1 - global_step / total_steps)
        optimizer.step()
        global_step += 1
        if global_step % 1000 == 0:
            run_long_context_eval(model)  # NIAH, RULER, etc.
```

**Needle-in-a-haystack eval.**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def make_niah_prompt(needle: str, context_length: int, position: float = 0.5) -> tuple[str, str]:
    """Build a NIAH prompt: filler text with a needle at a given fractional position."""
    # The "haystack" is repeated generic paragraphs (Paul Graham essays, in the original)
    filler_para = "The grass is green. The sky is blue. The sun is yellow. " * 20
    filler_para += "\n"  # paragraph break
    n_filler_paras = context_length // len(filler_para.split())
    haystack_paras = [filler_para] * n_filler_paras
    insert_idx = int(position * len(haystack_paras))
    haystack_paras.insert(insert_idx, f"\n\n{needle}\n\n")
    question = "\n\nWhat is the magic number? Answer with just the number."
    return "".join(haystack_paras), question


def niah_eval(model, tokenizer, needle="The magic number is 42.", lengths=[8*1024, 16*1024, 32*1024, 64*1024, 128*1024], positions=[0.0, 0.1, 0.2, ..., 1.0]):
    """Run a NIAH eval and return a heatmap of (length, position) -> accuracy."""
    results = {}
    for length in lengths:
        for position in positions:
            prompt, question = make_niah_prompt(needle, length, position)
            input_ids = tokenizer(prompt + question, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
            answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            correct = "42" in answer  # crude: real eval is more careful
            results[(length, position)] = correct
    return results

# Usage
model = AutoModelForCausalLM.from_pretrained("my-extended-model", torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("my-extended-model")
heatmap = niah_eval(model, tokenizer)
# Plot with matplotlib; green = correct, red = wrong
```

The full heatmap visualization is one `matplotlib.imshow` call away; we have left it as an exercise because no frontier lab ships a model without one.

## 9.14 What this means in practice — the JD, decoded

The work in this chapter is done by people with three different role tags.

- **Pre-training Researcher** owns the *recipe* — when to switch context length, what to upweight in the annealing mix, what RoPE rescaling to use. This is the recipe-tuning work, closely related to Chapter 8.
- **Large-Scale Training Engineer** owns the *infrastructure* — making the long-context training run actually work at 128K with ring attention and FlashAttention varlen. This is the distributed-systems work from Chapter 6 plus the kernel work from Chapter 20.
- **Evaluation Engineer** owns the *measurement* — running RULER, NIAH, LongBench, LEval on every checkpoint, building the leaderboards, detecting when the long-context extension has actually worked vs. when the model is just passing the easy NIAH but failing the harder sub-tasks.

The role of *post-training researcher* is downstream: they receive the mid-trained model and run SFT, RLHF, DPO, GRPO on it. If the mid-training was done well, the post-training has a much easier job. If the mid-training was done poorly, no amount of post-training can recover long-context capability.

A useful framing: mid-training is where the lab decides *what kind of model* the post-training will have to work with. Long context, math/code upweighting, the annealing mix — these are choices that the post-training inherits.

## 9.15 What you should take from this chapter

1. **Mid-training is the stage between bulk pre-training and SFT.** It is the annealing phase, the long-context extension phase, and the domain-upweighting phase, often combined.
2. **Long-context extension is the most concrete part.** Going from 8K to 128K requires modifying the RoPE frequencies (PI, NTK-aware, or YaRN), training progressively (8K → 32K → 128K), and accepting some quality cost at the original short context.
3. **The annealing phase is disproportionately influential.** The last 5–10% of pre-training tokens, with the learning rate decayed and the data mix upweighted to high-quality sources, is where the final loss surface is set.
4. **The three RoPE extension methods are PI, NTK-aware, and YaRN.** PI is the simplest; NTK-aware is a single-line hack; YaRN gives the best quality but is more complex. Frontier labs use YaRN.
5. **Long context is expensive.** Attention is O(N^2) compute and (with FlashAttention) O(N) memory. The arithmetic intensity of attention is low, so long-context training is bound by memory bandwidth and kernel efficiency. Ring attention + FlashAttention varlen is the standard.
6. **Long-context evaluation is harder than it looks.** NIAH is the easy test; RULER is the standard; the *effective* context length is usually 1/4 to 1/2 of the *advertised* context length. Frontier labs report both numbers.
7. **Mid-training is the bridge to reasoning.** The math/code upweighting, the long-context extension, the annealing on high-quality data — these set up the post-training (SFT, GRPO) that produces a reasoning model. Chapter 15 (GRPO) and Chapter 19 (R1) are where this pays off.

The next chapter is the case study — DeepSeek-V3 end-to-end — where the multi-stage training schedule and the long-context extension come together in one of the most detailed public pre-training reports of 2024.

---

**Exercises:** [Chapter 9 problem set](../../exercises/ch09.md) — includes the RoPE-extension comparison and the annealing-mixture design problem.
**Lab:** [`lab09_rope_extension`](../../labs/lab09_rope_extension.py) — read RoPE's wavelength table to predict what will break, then measure Position Interpolation destroying short-range accuracy that NTK-aware scaling and YaRN preserve.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. Source for the multi-stage training description and the FP8 + long-context combination.
- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — Meta AI, July 2024. Source for the annealing phase description.
- [\[6\] Qwen3 Technical Report](../appendix/b-references.md#6-qwen3) — Qwen Team, 2025. Source for the long-context extension at 32K → 128K.
- [\[15\] DeepSeekMath / GRPO](../appendix/b-references.md#15-grpo) — Shao et al., April 2024. Source for the math-domain continued pre-training and the GRPO algorithm used in post-training.
- [\[19\] FlashAttention](../appendix/b-references.md#19-flashattention) — Dao et al., 2022. Source for the IO-aware attention algorithm and the varlen mode used in long-context training.
- [\[35\] YaRN](../appendix/b-references.md#35-yarn) — Peng et al., 2023. Source for the YaRN RoPE extension.
- [\[38\] Ring Attention](../appendix/b-references.md#38-ring-attention) — Liu et al., 2023. Source for the context-parallelism approach to long-context training.
- [\[47\] Position Interpolation](../appendix/b-references.md#47-position-interpolation) — Chen et al., June 2023. Source for the PI method.
- [\[48\] NTK-aware scaling](../appendix/b-references.md#48-ntk-aware-scaling-bloc97-2023) — bloc97, 2023. The widely-used informal technique.
- [\[49\] RULER](../appendix/b-references.md#49-ruler) — Hsieh et al., 2024. Source for the formal long-context benchmark.
- [See full reference list](../appendix/b-references.md)
