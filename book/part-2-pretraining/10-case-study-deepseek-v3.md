# Chapter 10: Case study — DeepSeek-V3 end-to-end

> Reading time: ~50 minutes. This is the first case study of the book — a full walkthrough of the pre-training (and the publicly visible post-training) of DeepSeek-V3 [\[1\]](../appendix/b-references.md#1-deepseek-v3), from the team's stated goal to the training dynamics to the $5.5M cost claim and the things the report does not actually tell us. Chapter 1 used DeepSeek-V3 as a one-paragraph spine; this is the deep dive.

*Current as of early 2025.*

## 10.1 The team and the goal

DeepSeek-V3 was released by **DeepSeek-AI** on **December 26, 2024**, with a 60-page technical report and a permissive open-weight license [\[1\]](../appendix/b-references.md#1-deepseek-v3). DeepSeek-AI is a Chinese AI research company founded in 2023, spun out of High-Flyer (the quantitative hedge fund) and based in Hangzhou. The report credits roughly 200 named authors; the core pre-training team was much smaller. The model is fully open-weight; the training data is not released, and most data-pipeline internals remain unpublished.

The stated goal: build a frontier-quality open model at a fraction of the cost that comparable Western labs were reporting. The 671B total / 37B active model is positioned against Llama-3.1-405B, Qwen-2.5-72B, and Claude-3.5-Sonnet on capability, and against any frontier model on *training cost per capability*. The cost number — $5.576M of H800 compute at an assumed rental rate, for the full training (pre-training, context extension and post-training) — is the most-discussed line in the report; we devote §10.11 to it.

Three things make V3 a useful case study:

1. **It is the most detailed public frontier report to date.** The PDF reads more like an engineering spec than a research paper. The team publishes configuration numbers, the FP8 recipe, the load-balancing mechanism, the training schedule, the parallelism layout, the GPU-hours, and the cost. Western labs publish less.
2. **It introduces four techniques that have propagated through the field.** FP8 training at scale, auxiliary-loss-free MoE load balancing, multi-token prediction, and the DualPipe pipeline schedule. Each is now standard in some form.
3. **It is not the most capable model in the world** — it is in the same tier as 400B-class dense Western models, and behind the closed frontier. The fact that an order-of-magnitude-cheaper open run lands there is the whole point. We use V3 not because it is strongest, but because it is the most *transparent*.

The rest of this chapter walks through every layer of the run, in roughly the order data and compute flow through the system. We reference Chapter 3 for the data pipeline, Chapter 4 for tokenization, Chapter 5 for architecture, Chapter 6 for distributed training, and Chapter 8 for the optimizer — and we focus on the *specific choices* the DeepSeek team made.

## 10.2 The data

The full pre-training corpus is **14.8 trillion tokens** [\[1\]](../appendix/b-references.md#1-deepseek-v3). What the report says about it is short enough to give in full ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#data-and-tokenizer)):

- Compared with DeepSeek-V2, the corpus **raises the ratio of math and programming samples** and **expands multilingual coverage beyond English and Chinese**.
- The processing pipeline was "refined to minimize redundancy while maintaining corpus diversity".
- Documents are **packed** without cross-sample attention masking.
- **Fill-in-the-middle** training, in the Prefix-Suffix-Middle format, is applied to 10% of documents, carried over from DeepSeek-Coder-V2.

That is all. The report gives no mix ratios, no quality-classifier design, no deduplication method and no filter thresholds. Anything more specific you read about V3's data, including in earlier editions of this chapter, is inference, not reporting. Chapter 3 covers what the standard pipeline stages look like at other labs that did publish them.

The honest summary: V3's data is the least-documented part of an otherwise unusually detailed report. The 14.8T-token count is high but not unprecedented; Llama-3's flagship used 15.6T.

## 10.3 The tokenizer

DeepSeek-V3 uses a **byte-level BPE tokenizer** with a 128K vocabulary. Its pre-tokenizer and training data were "modified to optimize multilingual compression efficiency" [\[1\]](../appendix/b-references.md#1-deepseek-v3). The one design change the report describes: the new pre-tokenizer introduces tokens that **combine punctuation and line breaks**. That creates a *token-boundary bias* on multi-line prompts without a trailing newline, common in few-shot evaluation. DeepSeek mitigate it by randomly splitting a proportion of those combined tokens during training.

The published tokenizer file, generated from the hash-verified snapshot (Chapter 4 §4.11 compares it with Llama-3's and Qwen3's):

<!-- real-config: deepseek-v3-tokenizer -->
```json
// tokenizer.json (excerpt)
{
  "model": {
    "type": "BPE",
    "byte_fallback": false,
    "vocab": "<128,000 entries>",
    "merges": "<127,741 rules>"
  },
  "added_tokens": "<818 entries, 804 of them special>",
  "added_tokens (first special, by id)": {
    "0": "<｜begin▁of▁sentence｜>",
    "1": "<｜end▁of▁sentence｜>",
    "2": "<｜▁pad▁｜>",
    "128000": "<｜place▁holder▁no▁0｜>",
    "128001": "<｜place▁holder▁no▁1｜>",
    "128002": "<｜place▁holder▁no▁2｜>"
  },
  "normalizer": {
    "type": "Sequence",
    "normalizers": []
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "\\p{N}{1,3}"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "Split",
        "pattern": {
          "Regex": "[一-龥぀-ゟ゠-ヿ]+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "Split",
        "pattern": {
          "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
      }
    ]
  },
  "decoder": {
    "type": "ByteLevel"
  }
}
// config.json (excerpt)
{
  "vocab_size": 129280,
  "tie_word_embeddings": false
}
```

*Real config. Generated from `deepseek-ai/DeepSeek-V3` at commit `e815299` (retrieved 2026-09-26) by `tools/build_configs.py`; do not edit by hand.*
<!-- /real-config -->

Three things worth reading out of it. First, the pre-tokenizer **isolates CJK runs** (the basic Han block plus Hiragana and Katakana) before BPE, so no merge straddles a Chinese–Latin boundary. Second, there is **no byte fallback** (`byte_fallback: false`). Every string still encodes, because the base alphabet is the 256 bytes, so there is never an unknown token. Third, the 818 added tokens sit after the 128,000 BPE entries, and the embedding table is padded to 129,280 rows ([tokenizer fact sheet](../appendix/fact-sheets/tokenizers.md#deepseek-v3)). At $d = 7168$ that table is ~927M parameters, ~1.9 GB in BF16.

The tokenizer is *frozen* before pre-training — the team could not iterate on it during the run, because the model has been trained on the existing token IDs. A bad tokenizer choice is permanent; this is one of the reasons frontier labs A/B test tokenizers on small proxy models before committing.

## 10.4 The architecture

The architecture is a **decoder-only transformer** with two structural innovations: **Multi-head Latent Attention (MLA)** and **DeepSeekMoE**. The headline numbers [\[1\]](../appendix/b-references.md#1-deepseek-v3):

- **Total parameters:** 671B.
- **Active parameters per token:** 37B.
- **Layers:** 61 transformer blocks. The first 3 have dense FFNs; the other 58 are MoE.
- **Hidden dim:** 7168.
- **Attention:** MLA with 128 heads of dimension 128. There is **no KV-head grouping**: every head's K and V are reconstructed from one shared latent.
- **MLA latents:** KV latent 512, query latent 1536, plus a 64-dimensional decoupled RoPE key.
- **MoE:** 256 routed experts + 1 shared expert, top-8 routing.

Every number is a row on the [fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture).
- **MoE expert dim:** 2048 (each expert is a small SwiGLU FFN).

Chapter 5 covers the architecture in detail; this section focuses on the DeepSeek-specific choices.

### 10.4.1 Multi-head Latent Attention (MLA)

MLA was introduced in DeepSeek-V2 [\[2\]](../appendix/b-references.md#2-deepseek-v2) and reused, with minor changes, in V3. The core idea: instead of caching per-head K and V tensors for inference, compress them into a single low-dimensional latent vector per token, and reconstruct K and V on the fly during attention.

The forward path of an MLA block, abstracted. It follows the report's §2.1.1 equations; the dimensions are V3's.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Minimal RoPE on the last dimension of x: (B, T, ..., d), d even. pos: (T,)."""
    d = x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device) / d))
    ang = pos[:, None].float() * freqs[None, :]            # (T, d/2)
    ang = ang.view(1, x.shape[1], *([1] * (x.dim() - 3)), d // 2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA), DeepSeek-V2/V3.

    K and V for all heads are reconstructed from ONE latent per token (d_c = 512).
    RoPE cannot pass through that compression, so position is carried by a separate
    "decoupled" key of d_rope = 64 dims, shared by all heads. At inference only
    the latent and the decoupled key are cached: 512 + 64 = 576 values per token
    per layer. There is no KV-head grouping; every head gets its own K and V.
    """
    def __init__(self, d_model=7168, n_heads=128, head_dim=128,
                 kv_latent_dim=512, q_latent_dim=1536, rope_dim=64):
        super().__init__()
        self.h, self.dh, self.dr = n_heads, head_dim, rope_dim
        # KV path: one down-projection to the latent (this is what is cached) ...
        self.kv_down = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_latent_dim)
        # ... and per-head up-projections from it, for the position-free part of K, and for V.
        self.k_up = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        # Decoupled RoPE key: computed from the hidden state, shared across heads (also cached).
        self.k_rope = nn.Linear(d_model, rope_dim, bias=False)
        # Q path: compressed too, to save activation memory in training (not cached).
        self.q_down = nn.Linear(d_model, q_latent_dim, bias=False)
        self.q_norm = nn.RMSNorm(q_latent_dim)
        self.q_up = nn.Linear(q_latent_dim, n_heads * (head_dim + rope_dim), bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        c_kv = self.kv_norm(self.kv_down(x))                              # (B, T, 512)  <- cached
        k_r = rotate(self.k_rope(x), pos)                                 # (B, T, 64)   <- cached
        k_c = self.k_up(c_kv).view(B, T, self.h, self.dh)
        v = self.v_up(c_kv).view(B, T, self.h, self.dh)
        q = self.q_up(self.q_norm(self.q_down(x))).view(B, T, self.h, self.dh + self.dr)
        q_c, q_r = q.split([self.dh, self.dr], dim=-1)
        q = torch.cat([q_c, rotate(q_r, pos)], dim=-1)
        k = torch.cat([k_c, k_r[:, :, None, :].expand(B, T, self.h, self.dr)], dim=-1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                  # (B, h, T, ·)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # scale 1/sqrt(dh + dr)
        return self.o_proj(attn.transpose(1, 2).reshape(B, T, self.h * self.dh))
```

Two things to note in this code:

1. **The KV cache is the latent plus the decoupled key: 576 values per token per layer.** Standard MHA with the same 128 heads of dimension 128 would cache $2 \times 128 \times 128 = 32{,}768$, so MLA caches about 1.8% of it (our arithmetic; [fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture)). This is what makes serving a 671B model economically tractable. The decoupled key exists because RoPE's position-dependent rotation would otherwise sit between the query and the up-projection $W^{UK}$, blocking the trick below.
2. **The Q path also goes through a compression latent** (1536). It is not cached; the report motivates it as reducing activation memory during training.

In an optimized V3 implementation, the K and V up-projections can be **absorbed** into the query and output projections, so attention runs directly against the cached latent and the per-head K and V are never materialized. That absorption is possible for the position-free part of K precisely because position lives in the separate RoPE key.

### 10.4.2 DeepSeekMoE

The FFN block of every layer except the first three is a **Mixture-of-Experts** layer using the DeepSeekMoE design from V2 [\[2\]](../appendix/b-references.md#2-deepseek-v2):

- **256 routed experts**, each a small SwiGLU FFN (`up`, `gate`, `down` projections of size `2048` intermediate, `7168` hidden).
- **1 shared expert**, also a SwiGLU FFN, that every token passes through.
- **Top-8 routing**: each token is routed to its 8 highest-scoring experts.
- **Sigmoid gating**: affinity scores are sigmoids, and the gating weights are the scores of the 8 chosen experts, normalized to sum to 1.
- **Auxiliary-loss-free load balancing** via per-expert bias terms that affect only *which* experts are chosen (see §10.6).

A minimal DeepSeekMoE block:

```python
class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE — 256 routed experts + 1 shared expert, top-k=8 routing,
    auxiliary-loss-free load balancing via per-expert bias terms.
    """
    def __init__(
        self,
        d_model: int = 7168,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        expert_intermediate_dim: int = 2048,
        top_k: int = 8,
    ):
        super().__init__()
        self.top_k = top_k
        self.n_routed = n_routed_experts
        
        # Router: hidden -> n_routed_experts.
        self.gate = nn.Linear(d_model, n_routed_experts, bias=False)
        
        # Per-expert bias for auxiliary-loss-free load balancing. Added to the
        # routing scores BEFORE top-k. NOT trained by gradient descent; updated
        # by a control loop based on observed expert load.
        self.expert_bias = nn.Parameter(
            torch.zeros(n_routed_experts), requires_grad=False
        )
        
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, expert_intermediate_dim)
            for _ in range(n_routed_experts)
        ])
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(d_model, expert_intermediate_dim)
            for _ in range(n_shared_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.gate(x))        # (B, T, n_routed) affinities, sigmoid not softmax
        # CRITICAL: the bias changes WHICH experts are chosen ...
        _, topk_indices = (scores + self.expert_bias).topk(self.top_k, dim=-1)
        # ... but not HOW MUCH they are weighted: gating uses the original scores,
        # normalized over the chosen k.
        topk_scores = scores.gather(-1, topk_indices)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        
        # Naive per-expert dispatch. In real V3, this is an all-to-all + grouped GEMM.
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_indices[..., k]            # (B, T)
            weight = topk_weights[..., k].unsqueeze(-1)  # (B, T, 1)
            for e in range(self.n_routed):
                mask = (expert_idx == e)
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e](x[mask])
        for shared_expert in self.shared_experts:
            out = out + shared_expert(x)
        return out


class SwiGLUExpert(nn.Module):
    """One SwiGLU FFN — used for both routed and shared experts."""
    def __init__(self, d_model: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

The critical lines are the two around the bias. The bias is **added to the scores only for the top-k selection**, so it steers which experts are picked. The report is explicit that the gating value multiplied into the expert output "is still derived from the original affinity score" ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture)). If the bias leaked into the weights, the balancer would distort every token's output, not just its routing. The bias is *not* trained by gradient descent; it is updated by a separate rule based on running expert-load statistics. We explain the mechanism in §10.6.

In the real V3, the per-expert FFNs are stored as **block-sparse 3D tensors** of shape `(n_experts, hidden_dim, intermediate_dim)`, and the dispatch is a **grouped GEMM** that takes a sorted list of `(token, expert)` pairs and runs all expert computations in one kernel. We cover this in Chapter 6 and Chapter 20.

### 10.4.3 First-K-dense, MoE-later

The **first three layers are dense**; the report says it substitutes "all FFNs except for the first three layers with MoE layers", and the config's `first_k_dense_replace` is 3 ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture)). Their dense FFNs are wide (intermediate size 18,432). The report does not give a reason. A common interpretation, which is ours and not DeepSeek's, is that routing is least useful in the earliest layers, where representations are closest to the token embeddings and the router has little to specialize on.

## 10.5 Multi-token prediction (MTP)

Standard next-token prediction trains the model to predict $p(x_{t+1} \mid x_{\le t})$. DeepSeek-V3 additionally predicts **one more token** at each position (MTP depth $D = 1$) [\[1\]](../appendix/b-references.md#1-deepseek-v3). The report credits the idea to Gloeckle et al. (2024) and changes how it is done ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture)).

The mechanism, in words:

- The main model produces a hidden state $h_t$ at each position and predicts $x_{t+1}$ as usual.
- The **MTP module** takes $h_t$, combines it with the *embedding of the true next token* $x_{t+1}$ through a projection, runs the result through **one full Transformer block**, and predicts $x_{t+2}$ with the shared output head. The embedding and output head are shared with the main model.
- Because the module sees $x_{t+1}$ before predicting $x_{t+2}$, the prediction keeps the full causal chain. Gloeckle et al. instead predict all future tokens independently from the same state, with parallel heads.
- The MTP loss is added with weight $\lambda$: **0.3 for the first 10T tokens, then 0.1** for the remaining 4.8T.
- The module is not small. It holds **14B parameters**; the released checkpoint is 685B = 671B main model + 14B MTP. Most of that is its block's MoE FFN.
- At inference the module can simply be **discarded**, and the main model runs unchanged. It can also be **repurposed for speculative decoding**: DeepSeek report 85–90% acceptance for the second token and 1.8× tokens per second.

Why MTP helps, per the report and as commonly understood:

1. **Denser signal per position.** Each position contributes to two loss terms instead of one.
2. **Pre-planning.** To predict $x_{t+2}$, the representation at $t$ must encode something about what follows the next token. The report's phrase is that MTP "may enable the model to pre-plan its representations".
3. **A free draft model.** The speculative-decoding use comes at no extra training cost, because the module was trained alongside the main model.

## 10.6 Auxiliary-loss-free load balancing

Standard MoE training adds an **auxiliary loss** to the main cross-entropy loss, of the form:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{e=1}^{N} f_e \cdot p_e
$$

where $f_e$ is the fraction of tokens routed to expert $e$ in the current batch, $p_e$ is the average routing probability for expert $e$, and $\alpha$ is a coefficient (typically 0.01). The auxiliary loss is minimized when the experts are uniformly loaded, but it directly *hurts* the main loss — the model is paying a price for balance.

DeepSeek-V3's contribution is to **replace the auxiliary loss as the main balancer** with a **bias term** on the routing scores [\[1\]](../appendix/b-references.md#1-deepseek-v3). A complementary *sequence-wise* balance loss remains, with a very small weight (0.0001), only "to avoid extreme imbalance within any single sequence" ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture)). The mechanism:

- For each expert $e$, maintain a bias $b_e$ (initialized to 0).
- During the forward pass, the routing score for expert $e$ is `s_e + b_e`, where `s_e` is the router's output.
- During the backward pass, observe which experts received more or fewer tokens than the uniform target.
- **If expert $e$ is overloaded** (received > 1/N fraction of tokens), decrement $b_e$ by a small step $\gamma$. This makes the expert less likely to be picked next time.
- **If expert $e$ is underloaded**, increment $b_e$. This makes the expert more attractive.
- The bias is **not** a learnable parameter — it is updated by a rule, not by gradient descent.

Pseudocode for the bias update:

```python
# Inside the training loop, after the forward pass
def update_expert_bias(expert_bias, token_counts, target_count, gamma=0.001):
    """
    Adjust per-expert bias based on observed load.
    expert_bias: (n_experts,) current bias values
    token_counts: (n_experts,) how many tokens were routed to each expert
    target_count: scalar — uniform target = total_tokens / n_experts
    gamma: bias update step size
    """
    # Overloaded experts: decrement
    overloaded = token_counts > target_count
    expert_bias[overloaded] -= gamma
    
    # Underloaded experts: increment
    underloaded = token_counts < target_count
    expert_bias[underloaded] += gamma
    
    # Clamp to keep biases in a reasonable range
    expert_bias.clamp_(-0.5, 0.5)
```

Why this beats the auxiliary loss:

- **The auxiliary loss directly competes with the main loss.** The gradient from $\mathcal{L}_{\text{aux}}$ pushes the router to make the *routing probabilities* more uniform, which can hurt model quality. The bias mechanism instead pushes the *post-bias scores* to be uniform, while leaving the underlying `s_e` (which the main loss trains) untouched.
- **The bias is decoupled from the gradient.** There is no gradient flowing through $b_e$ into the router. The router trains on the main loss only; the bias is a separate, simple control loop.
- **The hyperparameter is just one scalar** (the update rate $\gamma$) instead of the auxiliary-loss coefficient $\alpha$, which has to be tuned jointly with the main loss.

The DeepSeek team reports that with auxiliary-loss-free balancing, the expert loads stay roughly uniform throughout training and the model quality is higher than the auxiliary-loss baseline at matched load uniformity.

## 10.7 The distributed training system

V3 was trained on **2,048 NVIDIA H800 GPUs** connected via NVLink within a node and InfiniBand across nodes [\[1\]](../appendix/b-references.md#1-deepseek-v3). The H800 is the China-export-compliant variant of the H100, with reduced interconnect bandwidth: DeepSeek report 160 GB/s NVLink against 50 GB/s InfiniBand. The layout is **16-way pipeline parallelism, 64-way expert parallelism spanning 8 nodes, and ZeRO-1 data parallelism, with no tensor parallelism** ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#parallelism-layout)).

Chapter 6 covers each parallelism strategy in depth; here are the V3-specific choices.

- **Pipeline parallelism (PP) = 16.** The 61 transformer layers are split across 16 stages and scheduled with DualPipe (§10.7.1). The report's Table 2 gives DualPipe's bubble as $(PP/2-1)(F\&B+B-3W)$ against 1F1B's $(PP-1)(F+B)$, paid for with twice the parameter memory.
- **Expert parallelism (EP) = 64, spanning 8 nodes.** Each layer's 256 routed experts are spread evenly over 64 GPUs, 4 per GPU. The all-to-all therefore crosses InfiniBand, which is the expensive choice on a cluster where InfiniBand is about 3.2× slower than NVLink. Two things make it pay. The all-to-all is overlapped with compute. **Node-limited routing** sends each token to at most 4 nodes: it crosses InfiniBand once per target node, to the same-index GPU, and fans out over NVLink from there. In exchange, every expert sees a large batch, and the model needs no TP.
- **Data parallelism: ZeRO-1**, sharding the optimizer state across data-parallel ranks. The report does not give the DP degree. With 16 stages on 2,048 GPUs there are 128 ranks per stage (our arithmetic).
- **No tensor parallelism.** DeepSeek say they optimized memory enough to avoid "costly Tensor Parallelism" altogether. They list three techniques: recomputing RMSNorm and the MLA up-projections in the backward pass, keeping the parameters' exponential moving average in CPU memory, and sharing the embedding and output head between the main model and the MTP module.

The reported parallelism config, in the Megatron-DeepSpeed style:

```yaml
# ILLUSTRATIVE: not a DeepSeek file. A summary of values from the report, in the
# Megatron-DeepSpeed style. Every value is a row on the V3 fact sheet.
# Total GPUs: 2048
model:
  num_layers: 61
  hidden_size: 7168
  n_attention_heads: 128    # MLA: no KV-head grouping; K and V come from the latent
  kv_latent_dim: 512
  q_latent_dim: 1536
  n_routed_experts: 256
  n_shared_experts: 1
  expert_intermediate_size: 2048
  top_k: 8
  first_k_dense: 3          # the first 3 layers are dense; the other 58 are MoE
  mtp_depth: 1              # one extra predicted token

parallelism:
  tensor_model_parallel_size: 1   # no TP anywhere
  pipeline_model_parallel_size: 16
  expert_model_parallel_size: 64  # spans 8 nodes; each token to at most 4 nodes
  zero_stage: 1                   # ZeRO-1 across the data-parallel ranks
  global_batch_size: 15360        # as reported; ramped from 3072 over the first 469B tokens
  sequence_length: 4096

precision:
  gemm_format: fp8_e4m3           # E4M3 for Fprop, Dgrad and Wgrad alike
  activation_scaling: 1x128       # per token per 128 channels
  weight_scaling: 128x128         # per 128 x 128 block
  accumulation: fp32_every_128    # promoted to FP32 on CUDA cores every 128 elements
  master_weights: fp32
  gradients: fp32
  adamw_moments: bf16
```

The combination is V3-specific: **wide cross-node EP, no TP**. Llama-3, a dense model, uses TP=8 inside each node instead. V3 shows the usual rule, keep the all-to-all on NVLink, being broken on purpose. The breach is paid for with a pipeline schedule, a routing constraint and custom kernels designed together.

### 10.7.1 DualPipe

The custom pipeline schedule, **DualPipe**, is one of V3's headline contributions [\[1\]](../appendix/b-references.md#1-deepseek-v3) [\[50\]](../appendix/b-references.md#50-dualpipe-deepseek-github). The motivation: in a standard 1F1B pipeline schedule, the all-to-all collective for MoE routing runs in the forward pass and again in the backward pass. Both of these collective operations compete with the matmul compute for the network, and on a 2,048-GPU cluster with hundreds of gigabytes per second of all-to-all traffic, the network becomes the bottleneck.

DualPipe's idea is to **overlap the MoE all-to-all of one micro-batch with the attention compute of the next micro-batch** in the pipeline. The pipeline is split into two "directions" of execution that run in parallel on the same GPUs, so that the network and compute are interleaved at fine granularity. The result is that the MoE communication is hidden behind the attention compute, and the per-step time is roughly the larger of (attention compute, MoE compute, MoE communication) instead of the sum.

A real implementation of DualPipe is published at the DeepSeek GitHub [\[50\]](../appendix/b-references.md#50-dualpipe-deepseek-github); the open-source release includes the schedule logic but not the full distributed runtime. Most teams that have studied the design report that the gains are substantial (on the order of 20–30% wall-clock improvement at the V3 scale) but the implementation is non-trivial — it requires a custom pipeline-parallel runtime rather than off-the-shelf Megatron or DeepSpeed.

### 10.7.2 Communication cost

A 2,048-GPU V3 run moves a lot of data:

- **All-reduce of gradients** at each step: roughly $2 \times 32\text{GB} = 64\text{GB}$ per DP group, all-reduced across 32 GPUs in the group. At ~400 GB/s effective InfiniBand bandwidth, each all-reduce takes ~0.2s per step. With 8K-token global batch, the all-reduce is dominated by other costs.
- **All-to-all for MoE routing** at each layer: each token's 8 expert assignments are sent to the 8 GPUs in the node. Total traffic per layer is roughly `B × T × top_k × hidden_dim × 2 bytes` per direction, where B × T is the per-node micro-batch tokens. At 4096 tokens per micro-batch and 8 top-k, this is ~50 MB per direction per layer, all-to-all'd across 8 GPUs in <1ms at NVLink bandwidth.

The critical insight is that **all-to-all within a node is essentially free at NVLink speeds**, and V3's design keeps the all-to-all within the node. The cross-node traffic is limited to the all-reduce of gradients, which is dominated by the data-parallel group.

## 10.8 FP8 training

V3 validated **FP8 mixed-precision training** "on an extremely large-scale model" for the first time, per its report [\[1\]](../appendix/b-references.md#1-deepseek-v3). Chapter 8 covers precision in general; this section focuses on the V3 specifics.

### 10.8.1 The format, and what stays in higher precision

The common FP8 recipe, used by NVIDIA's Transformer Engine among others, is a **hybrid**: E4M3 (4 exponent bits, 3 mantissa bits) for the forward pass, and E5M2 (5 exponent, 2 mantissa), with its wider range, for the backward pass. **V3 does not do this.** It uses **E4M3 on all tensors**, forward and backward, and says so in contrast to the hybrid ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#precision)). What makes the narrower range survivable in the backward pass is the fine-grained scaling in §10.8.2: each small group gets its own scale factor, so no single outlier has to fit in the same range as everything else.

Which operations run in FP8, and which do not:

- **In FP8:** all three GEMMs of every linear layer: `Fprop` (Y = X W), `Dgrad` (dX) and `Wgrad` (dW). Outputs are BF16 or FP32.
- **Kept in BF16/FP32:** the embedding, the output head, MoE gating, normalization and the attention operators.
- **Stored in FP32:** the master weights and the gradients used for accumulation.
- **Stored in BF16:** AdamW's first and second moments, "without incurring observable performance degradation". This is a departure from the standard FP32 optimizer state; see Chapter 8 §8.7.

### 10.8.2 Fine-grained per-block quantization

The naive way to do FP8 is to scale the entire tensor to fit in the E4M3 range. This loses precision for small-magnitude elements, which are common in transformer activations (the "outlier" problem: a few activations are very large, the rest are near zero, and scaling the whole tensor to fit the outliers crushes the small values).

V3's contribution is **per-block scaling**: each tensor is divided into blocks of 128 elements along the contraction dimension, and each block has its own FP32 scale factor. The contraction-dimension grouping ensures that every inner product (the multiply-and-accumulate of two blocks) is done in matched scale, with no precision loss from inter-block scale mismatch.

Pseudocode:

```python
def quantize_blockwise(x: torch.Tensor, block_size: int = 128, fmt=torch.float8_e4m3fn):
    """
    Quantize x to FP8 with per-block scaling.
    For an activation tensor of shape (..., d_model), the contraction
    dimension is d_model, and we scale per-block along that dim.
    """
    *batch_shape, d = x.shape
    assert d % block_size == 0
    
    # Reshape to (..., d // block_size, block_size)
    x_blocks = x.reshape(*batch_shape, d // block_size, block_size)
    
    # Per-block max-abs scale
    block_max = x_blocks.abs().amax(dim=-1, keepdim=True)  # (..., d/block, 1)
    block_max = block_max.clamp(min=1e-9)
    
    # E4M3 max representable value
    e4m3_max = 448.0  # E4M3 finite max
    
    # Scale and quantize
    scale = block_max / e4m3_max
    x_quant = (x_blocks / scale).to(fmt)
    
    return x_quant, scale  # scale in FP32

def fp8_matmul(a_quant, a_scale, b_quant, b_scale, out_dtype=torch.bfloat16):
    """
    FP8 matmul with per-block scaling. The grouped GEMM kernel applies
    the per-block scale factors at the end of each inner-product block.
    """
    # The actual contraction is a standard FP8 GEMM; the scale factors are
    # applied per-block during the epilogue. In real V3, this is a
    # CUTLASS / Triton kernel; the pseudocode below is the math.
    out = torch._scaled_mm(a_quant, b_quant, a_scale, b_scale, out_dtype=out_dtype)
    return out
```

The per-block scaling is the key engineering contribution. With 128-element blocks, a 7168-dim activation tensor has 56 scale factors per token — small enough to keep in registers, large enough to give most elements a useful dynamic range.

### 10.8.3 The cost / benefit

Running the GEMMs in FP8 "theoretically doubles the computational speed compared with the original BF16 method", in the report's words, and caching activations in FP8 roughly halves their memory. The report does not publish an end-to-end speedup over a BF16 run. Its quality claim is narrower and tested: at two smaller scales trained for ~1T tokens, the FP8 model's loss stayed within **0.25%** (relative) of a BF16 baseline ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule)).

The cost is engineering complexity: the FP8 kernels are custom, the per-block scaling adds branching to the matmul epilogues, and the choice of what stays in higher precision has to be made operator by operator. The DeepSeek team wrote custom CUTLASS / Triton kernels for the FP8 matmuls with the per-block scaling. The kernels are not fully open-sourced; the report describes the design but the implementation is internal.

## 10.9 The optimizer and schedule

The optimizer is **AdamW** [\[25\]](../appendix/b-references.md#25-adamw) with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1 and gradient clipping at norm 1.0 [\[1\]](../appendix/b-references.md#1-deepseek-v3) ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule)). Its moments are kept in BF16 (§10.8.1).

The learning-rate schedule is **not** warmup-then-cosine. It is closer to a warmup-stable-decay shape:

- **Warmup:** linear from 0 to $2.2 \times 10^{-4}$ over the first 2K steps.
- **Constant** at $2.2 \times 10^{-4}$ until **10T tokens**, about two-thirds of the run.
- **Cosine decay** to $2.2 \times 10^{-5}$ over the next **4.3T** tokens.
- **Two final constant phases** over the last 500B tokens: $2.2 \times 10^{-5}$ for 333B, then $7.3 \times 10^{-6}$ for 167B.

The batch size ramps from **3,072 to 15,360** sequences over the first 469B tokens, then stays at 15,360. At sequence length 4K that is ~63M tokens per step, so the run is roughly **235K optimizer steps** (our arithmetic: 14.8T / (15,360 × 4,096), ignoring the ramp). Two other schedules run alongside: the MTP loss weight (0.3, then 0.1 after 10T tokens) and the balancing bias's update speed (0.001, then 0 for the last 500B).

The long constant phase matters for readers of Chapter 8. It means most of the run happened at peak learning rate, and the decay, which is where much of the final loss improvement comes from, was concentrated at the end. It is also the phase where the data mix can be changed without restarting a cosine schedule.

## 10.10 The training dynamics

The report says less about the run's dynamics than about its design, and it is worth being exact about what it does say [\[1\]](../appendix/b-references.md#1-deepseek-v3) ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule)):

- **Stability.** "Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks." That is the whole statement. It does not publish a loss curve for the main run, a failure count or an MTBF, and it does not explain the stability.
- **Wall-clock.** Pre-training took "less than two months". The report gives the rate as 180K H800 GPU-hours per trillion tokens, which is 3.7 days per trillion on 2,048 GPUs; 14.8T tokens is therefore about 55 days.
- **GPU-hours.** 2,664K for pre-training, 119K for context extension and 5K for post-training: **2,788K** in total. The pre-training figure matches the rate: 14.8 × 180K ≈ 2,664K.

Earlier editions of this chapter described a loss curve with an early "bump", an MTBF of "several hours", and the team's explanation for the lack of spikes. None of that is in the report; see the [errata](../appendix/d-errata.md).

### 10.10.1 What the dynamics tell us

What can be concluded from so little:

1. **No rollbacks at this scale is itself informative.** Chapter 8 §8.13 describes the skip-and-roll-back playbook other labs have needed. V3 reports needing none. The report does not say why. FP8 with fine-grained scaling, the auxiliary-loss-free balancer, BF16 moments and the long constant learning rate are all plausible contributors. Attributing the stability to any one of them is speculation.
2. **The GPU-hour arithmetic is internally consistent.** The rate, the token count, the cluster size and the wall-clock all agree. That is worth checking in any report (Exercise 10.8).
3. **What is missing is the operational story.** How often hardware failed, how long recovery took, and how much time was lost are exactly the numbers Chapter 7 says labs rarely publish, and V3 does not either.

## 10.11 The cost claim

The $5.5M cost is the single most-discussed number in the report, and the single most-misquoted. The report's exact statement: V3's *full training* took 2,788K H800 GPU-hours (2,664K pre-training, 119K context extension, 5K post-training), which at an *assumed rental* rate of $2 per GPU-hour is $5.576M, of which pre-training is $5.328M [\[1\]](../appendix/b-references.md#1-deepseek-v3) ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)).

The caveats, which the report includes and which most secondary coverage omits:

- **Excludes prior research.** The MLA architecture, the DeepSeekMoE design, the FP8 kernels, the DualPipe schedule, the load-balancing mechanism — all were developed on earlier runs (DeepSeek-V2, DeepSeek-Coder, DeepSeekMath). The development cost of those techniques is not in the $5.5M.
- **Excludes salaries.** The team is ~200 named contributors; the labor cost during the V3 pre-training run alone is several million dollars at industry rates.
- **Excludes data preparation.** The 14.8T-token corpus took weeks of pipeline time on a separate cluster. The data-pipeline engineering and compute is not in the $5.5M.
- **Excludes failed runs.** The report does not say how many pre-training runs were started, partially trained, and abandoned. The $5.5M is for the *successful* run only.
- **Post-training is included, and small.** The 2,788K covers V3's own post-training: 5K GPU-hours, under 0.2% of the total. It does not cover training the internal R1 models whose outputs V3's SFT data was distilled from.
- **Excludes evaluation, safety testing, red-teaming.** The infrastructure to evaluate, ablate, and audit a 671B model is itself a meaningful cost.
- **Uses rental rate, not purchase cost.** The H800 list price is roughly $30,000 per GPU; 2,048 H800s is ~$60M of hardware. The $5.5M is the rental cost, not the cost to own the cluster.

A reasonable total-cost-of-ownership estimate, including all of the above, is **2–5× the headline number**, putting the realistic V3 cost in the $10M–$30M range. This is still dramatically cheaper than comparable Western runs of the period — Llama-3.1-405B is widely estimated at $100M+, GPT-4-class training is in the hundreds of millions — but it is not the "$5.5M for a frontier model" headline that gets repeated.

The comparison to Western frontier runs, with the caveats:

- **Llama-3-405B [\[5\]](../appendix/b-references.md#5-llama-3).** 405B dense, 15.6T tokens, $3.8 \times 10^{25}$ training FLOPs on up to 16K H100s. Meta did not publish a dollar figure; estimates in secondary coverage vary widely.
- **GPT-4.** Estimated >$100M training cost (OpenAI has not published; industry estimates are in this range).
- **Claude 3.5 Sonnet.** Anthropic has not published training cost estimates.

The DeepSeek cost claim is correct, narrow, and impressive. It is not the cost to *reproduce* the V3 result from scratch; it is the rental-rate compute of *one successful training run*, with all prior work thrown in for free.

## 10.12 The post-training overview

The V3 report describes a short post-training pipeline [\[1\]](../appendix/b-references.md#1-deepseek-v3) ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)):

1. **Supervised fine-tuning (SFT) on 1.5M instances.** Reasoning data (math, competition code, logic) is generated with an internal **DeepSeek-R1** model. Domain "expert" models trained with SFT and RL produce responses that keep R1's accuracy but are shorter and better formatted, and rejection sampling curates the final set. Non-reasoning data (creative writing, role-play, simple QA) is generated by DeepSeek-V2.5 and verified by human annotators. SFT runs for 2 epochs.
2. **Reward models.** A **rule-based** reward wherever the answer can be checked (a boxed math answer, test cases for code), which the report prefers as "resistant to manipulation or exploitation". Otherwise a **model-based** reward model trained from the V3 SFT checkpoints, with chain-of-thought in its preference data.
3. **Reinforcement learning with GRPO**, the critic-free algorithm from DeepSeekMath (Chapter 15), not PPO.

The direction of the R1 relationship is the reverse of a common misreading, and of earlier editions of this chapter: **reasoning is distilled *from* R1 *into* V3** (the report's §5.4.1 is "Distillation from DeepSeek-R1"). R1 itself is trained on top of V3-Base; Chapter 19 covers it.

The post-training is described in less detail than the pre-training, which is the same asymmetry as most Western reports.

The headline capability numbers — V3 matching or exceeding Llama-3.1-405B-Instruct and GPT-4o on MMLU, GSM8K, HumanEval, etc. — are measured on the **post-trained** model, not the base. The base model is competitive but not at the same level; post-training is what makes V3 a "frontier" model in the public-facing sense.

## 10.13 What was novel

Four specific contributions, each of which has propagated through the field [\[1\]](../appendix/b-references.md#1-deepseek-v3) [\[2\]](../appendix/b-references.md#2-deepseek-v2):

1. **FP8 training at scale.** The report claims the first validation of FP8 training "on an extremely large-scale model". The recipe is E4M3 on all tensors, tile- and block-wise scaling, and FP32 promotion of partial sums every 128 elements, with BF16 optimizer moments. The recipe is the contribution; the kernels are the engineering substrate. (Llama-3.1, released five months earlier, trained in BF16.)

2. **Auxiliary-loss-free MoE load balancing.** The bias-based mechanism is a clean alternative to the standard auxiliary loss. The argument — that the auxiliary loss directly competes with the main loss, while the bias-based mechanism does not — is theoretically and empirically supported. V3 pairs it with a tiny sequence-wise balance loss rather than removing auxiliary losses entirely. Other labs chose differently: Qwen3, for example, uses a global-batch balancing *loss* instead (see its [fact sheet](../appendix/fact-sheets/qwen3.md)).

3. **Multi-token prediction (MTP).** A sequential MTP module (14B parameters, one Transformer block) adds a second-token objective during training, and the same module doubles as a speculative-decoding draft at inference (1.8× tokens per second in the report).

4. **DualPipe.** The pipeline schedule that overlaps MoE all-to-all with attention compute. The open-source release is partial, but the design has been studied and partially adopted by other labs running large MoE models.

The cumulative effect is that V3 is a *demonstration* that frontier-quality models can be trained at a fraction of the cost that Western labs were reporting, with novel techniques that are now standard. The 671B / 37B-active MoE architecture with MLA is itself a contribution — it is the design that most subsequent Chinese MoE models have adopted or built on.

## 10.14 Limitations and open questions

V3 is unusually transparent for a frontier model, but it is not fully transparent. The things that are *not* published, and that the community is still figuring out:

- **The data mix.** The report publishes no ratios at all, by language or by category, and no record of changes to the mix during the run.
- **The exact quality classifier.** The features, the training labels, the threshold — all unpublished. The community has reverse-engineered guesses based on the V3 model's behavior, but the actual pipeline is not public.
- **The failure modes.** The report gives no MTBF, failure count or breakdown. Which component fails most often? NCCL hangs, HBM errors, FP8 numerical issues, MoE routing imbalance? Unpublished.
- **The exact filter thresholds.** The MinHash Jaccard threshold, the n-gram contamination threshold, the line-length and symbol-ratio cutoffs — all unpublished.
- **The exact DualPipe implementation.** The GitHub release is partial. The full distributed runtime, the schedule details, the overlap of forward and backward passes — not fully open.
- **The exact FP8 kernel implementation.** The report describes the design; the kernels are internal. This is the biggest open question for labs trying to reproduce the FP8-at-scale claim.
- **The exact RLHF preference data and reward model.** The SFT data is described at a high level; the preference data is not described at all.
- **The exact cost of the failed runs.** The $5.5M is the successful run only. The total cost including the failed runs (which the team surely had) is not published.
- **The exact "first-K-dense" choice.** The report mentions that the first few layers are dense but does not specify the number or justify the choice.

A reader who is trying to *reproduce* V3 has, as of early 2026, a clear architecture and a clear training schedule, but a fuzzy data pipeline and an opaque set of engineering details. The report is the most-detailed public frontier pre-training document, but it is not a complete engineering spec. We expect the V4 report (when it lands) to fill in some of these gaps; we do not expect any frontier lab to publish everything.

## 10.15 What you should take from this chapter

1. **V3 is a coherent end-to-end demonstration, not a single trick.** FP8 + auxiliary-loss-free MoE + MTP + DualPipe + 14.8T tokens of bilingual data, all composed. Each piece is reproducible in isolation; the composition is what is hard.
2. **The architecture is the MLA + DeepSeekMoE + MTP stack.** The numbers — 671B / 37B active, 61 layers (3 dense), 256 routed + 1 shared expert, kv_latent=512, q_latent=1536, a 64-d decoupled RoPE key, hidden=7168 — are the headline ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#architecture)). MLA caches 576 values per token per layer instead of per-head K and V; the MoE sharding keeps the per-GPU compute bounded; the MTP provides a denser training signal.
3. **The FP8 is the practical contribution.** E4M3 on all tensors, tile- and block-wise scaling, FP32 promotion of partial sums, BF16 optimizer moments. This is the engineering substrate that makes the rest affordable.
4. **The distributed system breaks a rule on purpose.** 16-way PP, 64-way EP across 8 nodes, ZeRO-1 DP, no TP. The cross-node all-to-all is paid for by DualPipe overlap, a 4-node routing limit and custom kernels ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#parallelism-layout)). The pieces are standard; the composition, and the co-design that makes it work, are V3's.
5. **The cost claim is narrow and correct.** $5.5M for one successful pre-training run at rental rates. Not the cost to reproduce; not the cost to own the cluster; not the total engineering cost.
6. **The post-training is closer to a black art than the pre-training.** SFT, RLHF, R1 distillation — all described at a high level, with the actual recipe unpublished.
7. **What is *not* published is significant.** The data-mix ratios, the quality classifier, the failure-mode breakdown, the FP8 kernel implementation, the DualPipe runtime — all are partial. A team trying to reproduce V3 has a clear architecture but a fuzzy pipeline.
8. **The V3 techniques are now standard.** FP8 with per-block scaling, auxiliary-loss-free MoE, MTP, and pipeline-overlap schedules have been adopted across the field. The report is not just a description of V3; it is a description of techniques that subsequent models use.

Chapter 19 will dive into DeepSeek-R1 — the reasoning-focused post-training built on top of V3-Base, with GRPO at the center. The R1 story is a separate case study, and a separate set of techniques.

---

**Exercises:** [Chapter 10 problem set](../../exercises/ch10.md) — includes the full recipe-reconstruction problem.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. The central reference for this chapter. arXiv:2412.19437.
- [\[2\] DeepSeek-V2](../appendix/b-references.md#2-deepseek-v2) — Earlier paper introducing MLA and DeepSeekMoE. arXiv:2405.04434.
- [\[3\] Megatron-LM](../appendix/b-references.md#3-megatron-lm) — Shoeybi et al. 2019. The 3D-parallelism reference.
- [\[10\] DeepSeek-R1](../appendix/b-references.md#10-deepseek-r1) — Preview of Chapter 19. arXiv:2501.12948.
- [\[21\] Scaling laws (Chinchilla)](../appendix/b-references.md#21-scaling-laws) — Hoffmann et al. 2022. The compute-optimal scaling reference.
- [\[25\] AdamW](../appendix/b-references.md#25-adamw) — Loshchilov and Hutter 2019. The optimizer.
- [\[50\] DualPipe (DeepSeek GitHub)](../appendix/b-references.md#50-dualpipe-deepseek-github) — The open-source partial release of the DualPipe pipeline schedule. https://github.com/deepseek-ai/DualPipe
- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — Meta AI 2024. The Llama-3.1-405B comparison figures.
- [\[8\] Deduplicating Training Data Makes Language Models Better](../appendix/b-references.md#8-deduplicating-training-data) — Lee et al. 2022. The suffix-array token-level dedup referenced in the data section.
- [See full reference list](../appendix/b-references.md)
