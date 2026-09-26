# Chapter 8: Optimization — precision, schedules, scaling laws

> Reading time: ~45 minutes. By the end of this chapter you should be able to read the optimization section of any frontier-lab technical report, know what each number means, and know which choice is a default and which is a deliberate, lab-specific decision.

*Current as of early 2025.*

## 8.1 The layer nobody talks about at the right altitude

The optimizer is the second-most-imitated part of frontier training (the first is the data mix). Almost every public reproduction of a frontier model starts with "use AdamW, cosine schedule, BF16, global-norm clip at 1.0." And the published *numbers* — peak LR, warmup steps, weight decay — are often close across labs. The differences show up in three places: how aggressively they push precision (BF16 → FP8 → FP4), how they handle the rare-but-catastrophic loss spike, and how they scale the batch. The art is in those.

This chapter walks through the entire optimization layer of a frontier pre-training run, anchored on four public case studies: **DeepSeek-V3** [\[1\]](../appendix/b-references.md#1-deepseek-v3) (FP8, fine-grained schedule), **Llama 3** [\[5\]](../appendix/b-references.md#5-llama-3) (BF16, the canonical modern schedule), **Qwen3** [\[6\]](../appendix/b-references.md#6-qwen3) (MoE schedule on a dense backbone), and **PaLM** [\[45\]](../appendix/b-references.md#45-palm-chowdhery-et-al-2022) (z-loss, the original 540B run). Where Anthropic or OpenAI have hinted at techniques in system cards, we say so. Where the lab has not published, we say that too.

The chapter is long, but most of the topics are short. The structure:

- **8.2** — the cross-entropy loss and why it is the right objective.
- **8.3** — AdamW, with PyTorch code.
- **8.4** — the learning-rate schedule: warmup, cosine, floor. Worked example.
- **8.5** — weight decay, decoupled.
- **8.6** — gradient clipping, global norm.
- **8.7** — mixed precision: FP32 master weights, BF16/FP16 forward, FP32 optimizer state.
- **8.8** — BF16 and why it works for transformers.
- **8.9** — FP8: E4M3, E5M2, scaling, what stays in BF16.
- **8.10** — scaling laws: Kaplan, Chinchilla.
- **8.11** — beyond Chinchilla: inference-aware and downstream-aware scaling.
- **8.12** — µTransfer: tune on a small model, transfer to a big one.
- **8.13** — stability tricks: z-loss, QK-norm, gradient-norm monitoring, loss-spike handling.
- **8.14** — batch size, critical batch size, and the LR↔batch-size coupling.
- **8.15** — the actual frontier configurations: DeepSeek-V3, Llama-3, Qwen3, PaLM.
- **8.16** — what this means in practice.
- **8.17** — takeaways.

## 8.2 The cross-entropy loss

The loss for next-token prediction is the **cross-entropy** between the model's predicted distribution and the one-hot target token, averaged over positions in the sequence:

$$ \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_{i+1} \mid x_1, \ldots, x_i) $$

where $p_\theta(x_{i+1} \mid x_1, \ldots, x_i)$ is the softmax of the model's logits at position $i$ over the vocabulary, evaluated at the true next token. In code:

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (B*T, V)
    targets.view(-1),              # (B*T,)
    reduction="mean",
)
```

Three things are worth knowing about this loss.

**Maximum likelihood.** Cross-entropy minimization *is* MLE on the joint distribution of the sequence under the autoregressive factorization — there is no cleaner information-theoretic justification. The model is trained to assign the highest possible probability to the true next token, minimizing the KL divergence between the model's distribution and the data distribution.

**Perplexity.** Perplexity is $\exp(\mathcal{L})$ — the geometric mean of "1 over probability of the true token" at each position. A frontier 70B model hits a held-out perplexity of roughly 6–8 on web text (a cross-entropy of about 1.8–2.1 nats per token). Lower on code, higher on long-tail web.

**Numerical issues.** The cross-entropy is computed in a fused kernel that combines the gather of the target logit with the log-softmax, avoiding materializing the full V-dim softmax (a memory cliff for large vocabularies). PyTorch's `F.cross_entropy` does this; Megatron-LM, Transformer Engine, and torchao do it faster. The DeepSeek-V3 paper specifically calls out a custom fused cross-entropy as one of the optimizations that lets them train at FP8 without losing accuracy on the loss term.

Why cross-entropy and not, say, MSE on the one-hot, or Hinge loss? Three reasons: the softmax probabilities are empirically well-calibrated (matters for rejection sampling, best-of-N, RLHF); the gradient with respect to the logits is bounded and well-scaled (composes cleanly with LR scheduling); and the gradient vanishes when the model assigns near-1.0 to the true token (the correct behavior — nothing more to learn from that example).

## 8.3 AdamW, the de-facto optimizer

AdamW [\[25\]](../appendix/b-references.md#25-adamw) is Adam with the weight-decay term decoupled from the gradient. Almost every frontier pre-training run uses it. A reference implementation:

```python
import torch
import math

class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay. Reference: Loshchilov & Hutter 2019."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps, wd = group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)        # m
                    state["exp_avg_sq"] = torch.zeros_like(p)     # v

                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # Biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step

                # Adam update
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)

                # Decoupled weight decay — applied directly to p
                if wd != 0:
                    p.mul_(1 - lr * wd)

        return loss
```

The two features that distinguish AdamW from "Adam + L2 regularization":

- **Decoupled weight decay.** $w \leftarrow w \cdot (1 - \eta \lambda)$ is applied directly to the parameters, *after* the Adam update. In practice, decoupling makes the effective regularization strength independent of the gradient magnitude, which gives much more predictable behavior than the coupled form $w \leftarrow w - \eta(\nabla \mathcal{L} + \lambda w)$.
- **Bias correction.** The first few steps of Adam have biased moment estimates because $m$ and $v$ are initialized to zero. The bias correction $\hat m = m / (1 - \beta_1^t)$, $\hat v = v / (1 - \beta_2^t)$ removes this. Without it, the first updates are too small and the optimizer takes longer to warm up.

The standard frontier hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.95$ (or 0.98 in some configs) instead of the original $\beta_2 = 0.999$, because the lower $\beta_2$ responds faster to recent gradient magnitudes over the millions of steps of a frontier run. $\epsilon = 1 \times 10^{-8}$. Weight decay = 0.1 for the body; some labs (DeepSeek-V3 in particular) use higher values for the embedding.

A small subtlety: the optimizer state ($m$, $v$) is held in **FP32** even when the parameters are in BF16. This is essential. If the optimizer state is in BF16, the $1/\sqrt{v}$ term loses precision in the low-magnitude regime and the model can fail to train. The implementation keeps `m` and `v` in FP32 by allocating them as `torch.zeros_like(p)` where `p` is the FP32 master copy, not the BF16 shadow copy. We cover this in §8.7.

## 8.4 The learning-rate schedule

The standard schedule is **linear warmup, then cosine decay to a floor**. The parameters:

- **Warmup:** linear increase from 0 to peak LR over $T_{\text{warmup}}$ steps, where $T_{\text{warmup}}$ is 0.1%–1% of the total training steps.
- **Decay:** cosine from peak LR to a floor LR over the remaining steps.
- **Floor:** typically 10% of the peak LR, sometimes 0% (full decay to zero) for the final 1–2% of training.

Why warmup? At step 0, the model has random weights. The gradient is correspondingly large and noisy. A peak LR applied immediately causes the weights to swing wildly in the first steps, sometimes diverging immediately. Warmup lets the optimizer stabilize.

Why cosine? It is smooth, has a continuous derivative, and the slow decay at the end lets the model "settle" into a low-loss region. The alternative — linear decay — works almost as well. The exact shape of the decay matters less than the total area under the curve (which is the cumulative learning, related to the integral of LR over training).

The schedule in code:

```python
import math

def get_lr(step, warmup_steps, peak_lr, total_steps, min_lr):
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup_steps:
        # Linear warmup from 0 -> peak_lr
        return peak_lr * (step / max(1, warmup_steps))

    if step > total_steps:
        return min_lr

    # Cosine decay from peak_lr -> min_lr over (total_steps - warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * coeff

# Worked example: DeepSeek-V3-style schedule
peak_lr      = 2.2e-4    # peak learning rate
min_lr       = 2.2e-5    # floor (10% of peak)
warmup_steps = 2000      # ~0.1% of total
total_steps  = 1_000_000  # ~14.8T tokens at 4M tokens / step
```

The numerical values vary by lab:

| Model | Peak LR | Floor (fraction) | Warmup (fraction) | Total tokens |
|---|---|---|---|---|
| Llama 3 8B | $3 \times 10^{-4}$ | not published | not published | ~15T |
| Llama 3 70B | $1.5 \times 10^{-4}$ | not published | not published | ~15T |
| Llama 3 405B | $8 \times 10^{-5}$ | 1% | 8,000 steps (0.7%) | 15.6T |
| Qwen3 32B (dense) | $2 \times 10^{-4}$ | 10% | 0.5% | ~36T (multi-stage) |
| DeepSeek-V3 (MoE) | $2.2 \times 10^{-4}$ | 10% | 0.2% | 14.8T |
| PaLM 540B | $1 \times 10^{-2}$ (Adafactor scale) | 10% | 0.1% | 0.78T |

The interesting pattern: larger models use *smaller* peak LRs. The intuition is that larger models have smaller optimal per-parameter step sizes. Llama 3 8B uses $3 \times 10^{-4}$; Llama 3 70B uses $1.5 \times 10^{-4}$; Llama 3 405B uses $8 \times 10^{-5}$ ([fact sheet](../appendix/fact-sheets/llama-3.md#architecture)). The report gives the floor and warmup only for the 405B. Qwen3 235B (MoE) uses $7 \times 10^{-5}$. The µTransfer framework formalizes this in §8.12.

A small but important detail: the "fraction of total" warmup numbers above assume the run goes for the full token budget. Many real runs adjust the warmup to be ~2000–5000 steps absolute, because the first few thousand steps are particularly delicate and a longer warmup is cheap insurance.

## 8.5 Weight decay, decoupled

Weight decay in AdamW is applied only to **weight matrices**, not to biases, layer norms, or embeddings. This is a long-standing best practice that traces back to the original AdamW paper and has been re-verified at every frontier scale.

The reason: weight decay shrinks parameters toward zero, which is a useful regularizer for weight matrices (which are over-parameterized and prone to large magnitudes) but counterproductive for biases (which have a different role) and for norms (which are constrained to have unit scale by definition, so shrinking them is wasted effort).

The decoupled form (from §8.3):

```python
# Inside the optimizer step
if wd != 0:
    p.mul_(1 - lr * wd)
```

This is equivalent to $w \leftarrow w \cdot (1 - \eta \lambda)$ per step, which accumulates to an exponential decay of $w$ over training. The effective $\ell_2$ regularization strength on the weights is $\lambda$, but the gradient updates are *not* contaminated by the weight-decay term (which is the "decoupled" part).

The standard value is $\lambda = 0.1$. Some labs (DeepSeek-V3, for example) split the weight decay: a higher value (e.g., 0.2) for the embedding and the LM head, and 0.1 for the transformer body. Some labs apply **no** weight decay to the embedding. The lab-specific choices are often unpublished.

## 8.6 Gradient clipping

The standard recipe is **global-norm gradient clipping** to a threshold of 1.0 (sometimes 0.5, sometimes 5.0 — varies by lab). The pseudocode:

```python
# Inside the training loop, after loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    norm_type=2.0,  # L2 norm
)
```

The global norm is the L2 norm of the concatenation of all parameter gradients. If the norm exceeds the threshold, all gradients are scaled by `threshold / norm` so that the new norm equals the threshold.

Why global-norm and not per-parameter? Because the gradients of different parameters have different typical scales (embeddings have small gradients, attention output projections have large ones), and a per-parameter threshold would have to be tuned per layer. The global norm is a single number that captures the overall "step size" of the update.

Why clip at all? Because gradient spikes — sudden, transient increases in gradient magnitude — can destabilize training. They are caused by rare training examples (very long documents, adversarial web text, MoE routing collapse), by numerical issues, or by hardware glitches. Clipping bounds the worst-case update.

The threshold is the second-most-tuned optimization hyperparameter after the peak LR. Llama 3 does not report its threshold ([fact sheet](../appendix/fact-sheets/llama-3.md#training-recipe)). PaLM uses 1.0. Qwen3 uses 1.0. DeepSeek-V3 uses 1.0. The 1.0 default is so universal that any deviation is worth noting.

The clipping happens **before** the optimizer step. Some implementations also log the unclipped norm for monitoring — a sudden increase in the unclipped norm (with no corresponding loss spike) is an early warning sign of a problem.

## 8.7 Mixed precision: FP32 master weights, BF16/FP16 forward, FP32 optimizer state

Mixed-precision training is the single largest performance optimization in modern transformer training, and the place where the precision formats are most carefully chosen. The full picture:

```
   FP32 master weights (1 copy of all parameters)
        │
        │ cast to BF16
        ▼
   BF16 forward pass
        │
        │ BF16 loss
        ▼
   BF16 backward pass (gradients in BF16)
        │
        │ cast to FP32
        ▼
   FP32 optimizer state (m, v in AdamW)
        │
        │ apply AdamW step in FP32
        ▼
   FP32 master weights updated
        │
        │ cast to BF16 for next forward
        ▼
   BF16 forward pass (next step)
```

The key choices:

- **Master weights in FP32.** The "true" parameters live in FP32. A BF16 copy is made at the start of each forward pass. The AdamW step is computed in FP32 using the FP32 weights, the FP32 optimizer state, and the FP32-cast gradients.
- **Forward and backward in BF16 (or FP16, see §8.8).** Matmuls are ~8× faster on Tensor Cores in BF16 than in FP32, and activations are 2× smaller, which means less memory traffic and larger effective batch sizes.
- **Optimizer state in FP32.** The $m$ and $v$ tensors are FP32 in the standard recipe. The usual argument is that $1/\sqrt{v}$ is sensitive and BF16's 7-bit mantissa is coarse in the low-magnitude regime. It is a default, not a law. DeepSeek-V3 stored both AdamW moments in **BF16** for its whole 14.8T-token run, "without incurring observable performance degradation", while keeping the master weights and gradients in FP32 ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#precision)). It saves 4 bytes per parameter of optimizer state.

The PyTorch pattern (using `torch.cuda.amp`):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # no-op for BF16, used for FP16

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.bfloat16):
        logits = model(batch.input_ids)
        loss = F.cross_entropy(logits.view(-1, V), batch.targets.view(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)   # skips update + reduces scale on Inf/NaN
    scaler.update()
```

For BF16, the `GradScaler` is a no-op (scales by 1.0). It is kept so the same code path works if the user switches to FP16. **Loss scaling** (for FP16) multiplies the loss by a large factor before the backward pass, preventing the small FP16 gradients from underflowing to zero. The `unscale_` reverses it before the optimizer step. If the scaled loss overflows to Inf/NaN, the scaler skips the update and reduces the scale factor for next time. This is the standard "skip the bad batch" pattern.

## 8.8 BF16: why it works for transformers

BF16 (Brain Float 16) is a 16-bit floating-point format with **8 bits of exponent** and **7 bits of mantissa**. FP16 has 5 bits of exponent and 10 bits of mantissa. The trade:

| Format | Bits | Exponent | Mantissa | Dynamic range | Precision |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | high |
| BF16 | 16 | 8 | 7 | same as FP32 | lower |
| FP16 | 16 | 5 | 10 | $\pm 65504$ | medium |

BF16 has the **same dynamic range as FP32** (because the 8-bit exponent is the same), but only 7 bits of mantissa. This is the right trade-off for transformers:

- **Activations, weights, and gradients have wide dynamic ranges.** The magnitudes of activations in a 70-layer transformer can span 6+ orders of magnitude across layers. FP16 would underflow on the small magnitudes and overflow on the large ones. BF16 handles the range without scaling.
- **The 7-bit mantissa is enough precision for the matmul accumulation.** Modern Tensor Cores accumulate BF16 matmuls in FP32, so the per-element precision loss in BF16 does not propagate to the final result. The intermediate precision (FP32 accumulator) is what determines the final accuracy.
- **No loss scaling needed.** Because BF16 does not underflow on small values, the entire `GradScaler` machinery from §8.7 is unnecessary. This simplifies the code and removes a class of bugs.

The 7-bit mantissa caveat: the precision per value is about 1 part in 128. For the parameters and gradients, this is fine. For the **optimizer state** (m, v in Adam), it is not fine — see §8.7. For **layer-norm statistics** (mean, variance, reciprocal standard deviation), it is borderline and most frontier implementations cast the layer-norm inputs to FP32 internally. For the **final cross-entropy**, it is generally fine if a fused kernel does the log-softmax in FP32. DeepSeek-V3 specifically keeps the cross-entropy in FP32 for this reason.

BF16 has been the default training precision for frontier transformer training since GPT-3 (2020) and Stable Diffusion (2022). Llama, Qwen, DeepSeek (V1/V2) all use it. The frontier has moved to FP8 starting in 2024 — see §8.9.

## 8.9 FP8: the new frontier

FP8 is the next step down in precision. There are two formats:

- **E4M3:** 1 sign bit, 4 exponent bits, 3 mantissa bits. Smaller dynamic range (max finite value 448), better precision per value.
- **E5M2:** 1 sign bit, 5 exponent bits, 2 mantissa bits. Wider range (max 57,344), worse precision.

Both formats are **signed**: 1 + 4 + 3 and 1 + 5 + 2 bits. The common hybrid recipe (Transformer Engine's default) uses E4M3 for the forward pass and E5M2 for gradients in the backward pass, trading precision for range where gradients need it. DeepSeek-V3 is the notable exception, using E4M3 everywhere (below).

The numerical issues are severe. E4M3 has only 3 bits of mantissa, which is about 1 part in 8 precision. A matmul of two E4M3 matrices would, with no protection, produce essentially random output. The fix is **per-tensor** or **per-block scaling factors** that rescale the values to use the available mantissa range. There are two standard scaling strategies:

- **Per-tensor scaling.** Compute the maximum absolute value of the tensor, then scale so the maximum maps to the max representable value. The scale factor is a single FP32 number per tensor.
- **Per-block / per-tile scaling.** Divide the tensor into blocks of 32 or 128 elements, compute a scale factor per block, and apply per-block. This gives much better dynamic range at the cost of more bookkeeping.

The DeepSeek-V3 paper [\[1\]](../appendix/b-references.md#1-deepseek-v3) uses a finer-grained scheme: **1×128 tiles** (per token per 128 channels) for activations and **128×128 blocks** for weights. Partial sums are promoted to FP32 on CUDA cores every 128 elements, because the H800's tensor-core FP8 accumulation keeps too few bits. With scaling that fine, V3 uses **E4M3 on all tensors**, including the backward pass, rather than the E4M3/E5M2 hybrid. Validated at two smaller scales over ~1T tokens, FP8's loss stayed within 0.25% (relative) of BF16 ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#precision)).

A minimal FP8 training loop sketch (using Transformer Engine, which is the most widely used library for FP8 in PyTorch):

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Create a recipe: E4M3 for forward, E5M2 for backward, FP32 accumulation
fp8_recipe = recipe.DelayedScaling(
    margin=0,                  # no extra margin
    interval=1,                # recompute scale every step
    fp8_format=recipe.Format.E4M3,
    amax_history_len=1024,
    amax_compute_algo="max",
)

class FP8Block(nn.Module):
    """Linear layer with FP8 forward and backward."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = te.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # te.Linear uses the FP8 recipe automatically
        return self.linear(x)

# In the training loop
model = MyTransformer(...).cuda()  # uses te.Linear in place of nn.Linear
for batch in dataloader:
    with autocast(enabled=False):  # TE handles its own casting
        # TE's autocast wrapper applies the recipe
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            logits = model(batch.input_ids)
            loss = F.cross_entropy(logits.view(-1, V), batch.targets.view(-1))
    loss.backward()
    optimizer.step()
```

The torchao library (from PyTorch) provides a similar API. The key invariant: **what stays in BF16 (or FP32)** is:

- The **embedding lookup** (the gradient w.r.t. the embedding table is very sparse and FP8 cannot represent it without per-element scaling).
- The **final layer norm** (statistics are sensitive to precision).
- The **LM head** and the **cross-entropy loss** (numerical issues in the log-softmax are amplified by BF16/FP8).
- The **router** in MoE models (the routing logits need full precision to keep the load balance stable).

The DeepSeek-V3 paper has a table of which operations are FP8 and which are BF16, and it is non-trivial. The lesson: FP8 is not a drop-in replacement for BF16. It is a careful migration where ~80% of the matmul FLOPs go to FP8 and ~20% stay in BF16/FP32.

The Blackwell-generation GPUs (B200, B100) have native FP8 tensor cores at 2× the throughput of BF16. Frontier runs on Blackwell (Llama 4, DeepSeek-V4, etc.) are expected to push FP8 even further.

## 8.10 Scaling laws: Kaplan 2020, Chinchilla 2022

The scaling laws formalize the relationship between model size $N$ (number of parameters, excluding embeddings), dataset size $D$ (tokens), and compute $C$ (FLOPs). The two foundational papers:

- **Kaplan et al. 2020** [\[21\]](../appendix/b-references.md#21-scaling-laws) — the first scaling-laws paper, from OpenAI. Found that loss decreases as a power law in $N$, $D$, and $C$ independently, with different exponents. Recommended scaling $N$ and $D$ roughly in proportion to $C$ (i.e., bigger models on bigger datasets, but with a slight tilt toward bigger models).
- **Chinchilla 2022** [\[21\]](../appendix/b-references.md#21-scaling-laws) — from DeepMind, the famous rebuttal. Showed that Kaplan's analysis was confounded — when you correctly account for the fact that bigger models need proportionally bigger datasets, the optimal scaling is $N$ and $D$ scaled **equally** with $C$. The Chinchilla rule of thumb is **~20 tokens per parameter** for compute-optimal training.

The compute-optimal frontier in math:

For a compute budget of $C$ FLOPs, the optimal model size and dataset size are:

$$ N^*(C) \approx \left(\frac{C}{a}\right)^{0.5}, \quad D^*(C) \approx \left(\frac{C}{b}\right)^{0.5} $$

with constants $a, b$ such that $D^* / N^* \approx 20$ across several orders of magnitude.

The practical implication: for a 70B model, you should train on at least 1.4T tokens. For a 7B model, at least 140B tokens. The Llama 1 paper trained the 65B model on 1.4T tokens, which was exactly on the Chinchilla line. The Llama 2 paper trained the 70B on 2T tokens, slightly over-Chinchilla. The Llama 3 paper trained on a corpus of about 15T tokens ([fact sheet](../appendix/fact-sheets/llama-3.md#pre-training-data)), **massively** over-Chinchilla for the 70B — and the result was a much better model. The reason is the topic of the next section.

## 8.11 Beyond Chinchilla: data may matter more than parameters

The Chinchilla rule is for *pre-training loss*, not for *downstream task performance*. Sardana et al. 2024 [\[43\]](../appendix/b-references.md#43-beyond-chinchilla-sardana-et-al-2024) extended the framework to include **inference cost** and **downstream evaluation performance**. The finding:

- For a fixed inference budget, the compute-optimal model is **smaller** than Chinchilla's prediction, because a smaller model is cheaper to run.
- For a fixed downstream benchmark, the model can be **smaller** and trained on **more data**, because downstream tasks benefit more from data than from parameters.
- Frontier labs care about *inference cost*, not just *training cost*, and a model that is 3× cheaper to run can serve 3× more users.

This is why Llama 3 8B is trained on about 15T tokens (roughly 1,900 tokens per parameter, way over Chinchilla) and is a much better model than a Chinchilla-optimal 8B trained on 160B tokens would be. Same for Qwen3, DeepSeek-V3, and the rest of the post-2023 frontier. The "20 tokens per parameter" rule is **wrong** for modern frontier runs, and the labs know it.

The rule of thumb that has replaced Chinchilla in practice is **"train until downstream benchmarks plateau"**, which for an 8B model turns out to be ~10–20T tokens, and for a 70B model ~15–30T tokens. The exact number is lab-specific and not always published.

A second, related finding: for a fixed compute budget, training a **smaller model on more data** and *distilling* it into a larger model can match or exceed the larger model trained directly. This is the basis of the "small models, big data" approach that Qwen3, Llama 3, and DeepSeek-V3 all use for some of their model variants.

A third finding: the **annealing** phase (the last 5–10% of training on a higher-quality data subset) provides most of the downstream benchmark gains, not the bulk training. This is the basis of mid-training (Chapter 9).

## 8.12 µTransfer: tune on a small model, transfer to a large one

The Tensor Programs line of work (Yang et al. 2022 [\[44\]](../appendix/b-references.md#44-mutransfer--mup--tensor-programs-yang-et-al-2022)) introduces a **maximal update parameterization (µP)** of neural networks in which the optimal hyperparameters (peak learning rate, weight initialization scale, Adam epsilon, etc.) **do not change with model width**. The practical implication:

1. Train a small proxy model (e.g., 10M parameters) with µP.
2. Tune the hyperparameters on the small model (which is cheap — you can run hundreds of trials).
3. Train the big model (e.g., 100B parameters) with the same hyperparameters in the µP parameterization.

The hyperparameters transfer zero-shot. No re-tuning needed.

This is a huge deal for frontier labs, because tuning a 100B model is prohibitively expensive (each trial takes weeks and millions of dollars). The µTransfer framework makes the LR schedule, the weight init scale, and the Adam epsilon free to tune.

In practice, frontier labs use a hybrid: tune LR, weight init, and Adam epsilon with µTransfer on a small proxy, then tune other hyperparameters (warmup, weight decay, batch size) on the big model. The µTransfer-tuned portion is the high-dimensional part that would otherwise be infeasible.

The framework also predicts that the optimal LR scales as $1 / \sqrt{N}$ for the embedding and $1/N$ for the hidden weights (where $N$ is the width). This is consistent with the empirical observation that larger models use smaller peak LRs.

A subtlety: the parameterization matters. If the model is not in µP form, the zero-shot transfer does not work. Most frameworks (Megatron-LM, Transformer Engine) provide µP parameterizations of the standard transformer block. The DeepSeek-V3 paper does not explicitly use µTransfer but the per-block learning rate schedule they describe is consistent with the µP framework.

## 8.13 Stability tricks: z-loss, QK-norm, monitoring, loss-spike handling

Frontier runs are not smooth. They have **loss spikes** — sudden jumps in the training loss that recover in a few hundred steps, sometimes after a checkpoint rollback. The standard toolkit:

**Z-loss (PaLM, Chowdhery et al. 2022 [\[45\]](../appendix/b-references.md#45-palm-chowdhery-et-al-2022)).** Add an auxiliary term that penalizes the log of the softmax partition function:

$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \log^2 Z $$

where $Z = \sum_j \exp(\ell_j)$ is the softmax denominator. This penalizes logits becoming very large (which would make the softmax very peaked and the gradient very large for the negative classes). $\alpha$ is typically $1 \times 10^{-4}$. Z-loss is used in PaLM, PaLM-2, and several subsequent frontier runs.

**QK-norm (Henry et al. 2020 [\[46\]](../appendix/b-references.md#46-qk-norm-henry-et-al-2020)).** Add a LayerNorm to the queries and keys inside the attention dot product:

```python
# Inside the attention block
q = self.q_proj(x)   # (B, H, T, D)
k = self.k_proj(x)
q = self.q_norm(q)   # layer norm per head
k = self.k_norm(k)
attn = (q @ k.transpose(-2, -1)) / sqrt(D)
```

This bounds the magnitude of the pre-softmax logits, preventing attention entropy collapse and the resulting gradient spikes. Used in StableLM and several open-weight Llama derivatives as a "cheap insurance" patch.

**Gradient norm monitoring.** Track the global gradient norm (before clipping) at every step. A sudden increase — say, 5× over the moving average — is an early warning sign. The standard pattern:

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
if grad_norm > 5 * grad_norm_ema:
    log.warning(f"Gradient spike: {grad_norm:.2f} (EMA: {grad_norm_ema:.2f})")
```

**Loss-spike handling.** When a loss spike is detected, the standard playbook is one of:

1. **Skip the batch** and continue at the current LR (cheapest, but doesn't prevent the next spike).
2. **Skip + reduce LR** by a factor (e.g., 0.5–0.8) for the next window of steps.
3. **Roll back to a pre-spike checkpoint** and reduce the LR. Exact thresholds are rarely published. Llama 3 reports not needing any of these: its 405B run saw "few loss spikes" and needed no interventions to correct divergence ([fact sheet](../appendix/fact-sheets/llama-3.md#training-recipe)).

DeepSeek-V3 is the counterpoint: it reports no irrecoverable spikes and no rollbacks in its whole run, and publishes no spike policy ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule)).

**Activation recomputation.** Not strictly a stability trick, but tightly coupled: re-computing activations during the backward pass trades compute for memory, enabling larger batch sizes. The Megatron analysis (Korthikanti et al. [\[39\]](../appendix/b-references.md#39-korthikanti-2022-activation-recomputation)) showed that **selective** recomputation (only the linear-layer outputs, not the layer norms or dropouts) recovers most of the memory savings of full recomputation at a small compute cost.

## 8.14 Batch size: the critical batch size and the LR↔batch coupling

The global batch size is one of the most consequential hyperparameters, and the one most tightly coupled to the learning rate. The relationship is captured by the **critical batch size** $B^*$:

- For batch sizes $B \ll B^*$, the gradient is noisy and a larger batch gives nearly-linear speedup in steps-to-converge. Doubling $B$ halves the steps, so wall-clock time to a given loss is roughly constant. **Linear scaling rule:** $\text{LR} \propto B$.
- For batch sizes $B \gg B^*$, the gradient becomes redundant (each minibatch gives almost the same gradient as the previous one), so larger batches give no speedup. **Square-root scaling rule:** $\text{LR} \propto \sqrt{B}$.
- $B^*$ itself grows slowly with model size and data size. For a 70B model, $B^*$ is roughly 4M–8M tokens. For an 8B model, $B^*$ is roughly 1M–2M tokens.

The PaLM paper [\[45\]](../appendix/b-references.md#45-palm-chowdhery-et-al-2022) uses the square-root scaling rule: $\text{LR} = \text{LR}_{\text{base}} \cdot \sqrt{B / B_{\text{base}}}$. The original GPT-3 paper used linear scaling. Modern frontier runs use a mix: linear scaling for the warmup phase, then a fixed LR for the main training.

The DeepSeek-V3 paper uses a constant LR for the first 10T of its 14.8T tokens, with the batch size ramped from 3,072 to 15,360 sequences (~13M to ~63M tokens at 4K) over the first 469B tokens, about 3% of training, then held constant. The Llama 3 paper ramps the 405B batch: 4M tokens at sequence length 4,096, then 8M tokens at 8,192 after 252M tokens, then 16M after 2.87T tokens ([fact sheet](../appendix/fact-sheets/llama-3.md#training-recipe)). It does not give the 8B and 70B batch sizes. The Qwen3 paper gives no batch size.

The critical batch size is a function of the model, the data, and the loss. Frontier labs often use **batch size ramping** — starting at a smaller batch (where the gradient is noisier and the LR scaling is more forgiving) and ramping up to the critical batch over the first few percent of training. This is what DeepSeek-V3 does.

A second consideration: **the batch size is also constrained by memory.** The activations scale linearly with batch size × sequence length × hidden dimension, and the optimizer state scales linearly with batch size. At frontier scale, the memory cost of a 16M-token batch is non-trivial. The activation-recomputation trade-off (§8.13) is the main lever.

## 8.15 The actual frontier configurations

The published (or strongly inferred) optimization configurations for the four case-study models.

### DeepSeek-V3

Every value is a row on the [fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule).

- **Optimizer:** AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay = 0.1. Moments in BF16; master weights and gradients in FP32.
- **Schedule:** linear warmup over 2K steps to $2.2 \times 10^{-4}$; **constant** until 10T tokens; cosine decay to $2.2 \times 10^{-5}$ over 4.3T; then $2.2 \times 10^{-5}$ for 333B tokens and $7.3 \times 10^{-6}$ for the final 167B.
- **Batch size:** ramps from 3,072 to 15,360 sequences over the first 469B tokens, then constant. The **batch-size ramp** pattern.
- **Precision:** FP8 **E4M3 on all tensors** for the linear-layer GEMMs, with 1×128 / 128×128 scaling; BF16/FP32 kept for the embedding, output head, MoE gating, normalization and attention.
- **Gradient clipping:** global norm, threshold 1.0.
- **Total tokens:** 14.8T. **Stability:** no irrecoverable spikes and no rollbacks reported; no spike policy published.

### Llama 3 (405B)

The report publishes the recipe for the 405B and says the 8B and 70B use "similar recipes" ([fact sheet](../appendix/fact-sheets/llama-3.md#training-recipe)).

- **Optimizer:** AdamW. $\beta_1$, $\beta_2$, $\epsilon$ and weight decay are not reported for this run.
- **Schedule:** linear warmup over 8,000 steps (0.7% of 1.2M), then cosine decay to $8 \times 10^{-7}$ (1% of peak) over 1,200,000 steps. The last 40M tokens anneal linearly to zero (§9.4).
- **Peak LR:** $8 \times 10^{-5}$. **Batch size:** ramped 4M → 8M → 16M tokens, at 252M and 2.87T tokens.
- **Precision:** BF16, with FP32 gradient accumulation. **Gradient clipping:** not reported. **Total tokens:** 15.6T.
- **Stability:** "few loss spikes", and no interventions to correct divergence.

### Qwen3 (235B MoE)

- **Optimizer:** AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 1 \times 10^{-8}$, weight decay = 0.1.
- **Schedule:** warmup, then cosine. Multi-stage: pre-training on ~30T tokens, then mid-training on higher-quality data with a smaller LR.
- **Peak LR:** $7 \times 10^{-5}$ (for the 235B MoE). Smaller dense variants use larger LRs (e.g., $2 \times 10^{-4}$ for 32B).
- **Batch size:** ~16M–32M tokens, held constant. **Precision:** BF16. **Total tokens:** ~36T.

### PaLM (540B)

- **Optimizer:** Adafactor (a memory-efficient alternative to AdamW) with $\beta_1 = 0.9$, $\beta_2 = 0.99$.
- **Schedule:** warmup over 0.1% of steps, then cosine decay to 10% of peak.
- **Peak LR:** $1 \times 10^{-2}$ (Adafactor scale; the equivalent AdamW LR is ~$1 \times 10^{-4}$).
- **Batch size:** ramps from 1M to 4M tokens over the first 7,500 steps, then constant.
- **Precision:** BF16. **Stability:** z-loss with $\alpha = 1 \times 10^{-4}$. **Total tokens:** 0.78T (much less than modern runs).

Across the four:

| | DeepSeek-V3 | Llama 3 405B | Qwen3 235B | PaLM 540B |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW | Adafactor |
| Peak LR | 2.2e-4 | 8e-5 | 7e-5 | 1e-2 (Adafactor) |
| Warmup | 0.1% | 0.7% | 0.5% | 0.1% |
| Schedule | cosine to 10% | cosine to 1% | cosine to 10% | cosine to 10% |
| Weight decay | 0.1 | not reported | 0.1 | 0.1 |
| Batch size | 73M (ramped) | 16M (ramped) | ~32M (constant) | 4M (ramped) |
| Precision | FP8 | BF16 | BF16 | BF16 |
| Total tokens | 14.8T | 15.6T | ~36T | 0.78T |
| Stability | custom | few spikes, no interventions | standard | z-loss |

The takeaway: the schedule shape is universal (warmup + cosine to 10% floor), the weight decay is universal (0.1), the global-norm clip is universal (1.0). The differences are in the peak LR (smaller for bigger models), the batch size (larger for bigger models, but with a critical-batch ceiling), the precision (FP8 for DeepSeek-V3, BF16 for the rest), and the stability tricks (z-loss for PaLM, custom spike handling for DeepSeek-V3).

## 8.16 What this means in practice

Mapping the chapter to frontier-lab job roles:

| What you read in this chapter | What the engineer does |
|---|---|
| Cross-entropy loss | Implements the fused cross-entropy kernel, debugs loss anomalies |
| AdamW | Tunes $\beta_1, \beta_2, \epsilon$ for the specific run; implements custom optimizer variants (Adafactor, Lion, Sophia) |
| LR schedule | Tunes peak LR, warmup, floor; implements custom schedule shapes (WSD, trapezoidal) |
| Weight decay | Decides which parameters get decayed (and at what rate) |
| Gradient clipping | Monitors the unclipped norm, builds alerting |
| Mixed precision | Implements the FP32 master / BF16 copy machinery, fuses the cast operations |
| BF16 | Verifies the numerical correctness of the BF16 path, handles edge cases (NaN losses) |
| FP8 | Implements the per-block scaling, the FP8 linear layer, the E4M3/E5M2 selection per op |
| Scaling laws | Runs the small-model scaling-law sweeps, extrapolates to the big run |
| µTransfer | Implements the µP parameterization, runs the small-model proxy tuning |
| Stability tricks | Monitors for loss spikes, implements the spike-handling policy, integrates z-loss and QK-norm |
| Batch size | Tunes the batch size, the batch ramp schedule, the LR↔batch coupling |

A "Pre-training Researcher" at a frontier lab owns the optimizer, the schedule, the precision, and the scaling-law analysis. A "Large-Scale Training Engineer" owns the FP8 implementation, the distributed AdamW, the fused kernels, and the batch-size tuning. A "Stability / Run Manager" owns the loss-spike handling, the checkpoint rollback, and the alerting.

## 8.17 What you should take from this chapter

1. **The loss is cross-entropy, the optimizer is AdamW, the precision is BF16 or FP8.** These are universal across frontier pre-training. The interesting choices are the FP8-vs-BF16 boundary, the stability tricks, and the scaling-law analysis.
2. **The schedule is warmup + cosine to a 10% floor.** Peak LR is $10^{-4}$ to $10^{-3}$, decreasing with model size. Warmup is 0.1%–1% of training. Weight decay is 0.1, decoupled, applied only to weight matrices.
3. **Gradient clipping is global-norm, threshold 1.0.** Almost universal.
4. **Mixed precision is FP32 master / BF16 forward / FP32 optimizer state.** FP8 is the new frontier (DeepSeek-V3, Llama 4+). The common recipe is E4M3 forward and E5M2 backward. DeepSeek-V3 showed that E4M3 everywhere works with fine enough scaling, and that BF16 optimizer moments can replace FP32. The embedding, norms and output head stay in higher precision.
5. **Chinchilla's "20 tokens per parameter" is wrong for downstream tasks.** Frontier labs train 8B models on 15T tokens (1:2000) and 70B models on 15T tokens (1:200). Data matters more than parameters for downstream quality.
6. **µTransfer lets you tune on a small model and transfer to a large one.** The framework formalizes why "smaller LR for larger models" works.
7. **Stability is not free.** z-loss, QK-norm, and loss-spike handling are all part of the standard toolkit. The exact thresholds and the rollback policy are the unpublished parts of any frontier run.
8. **The critical batch size is the ceiling on linear LR scaling.** Beyond it, you must use the square-root rule. Most frontier runs use a constant batch size, deliberately over the critical batch size, to improve stability at the cost of some compute efficiency.

The next chapter covers mid-training — the domain annealing, long-context, and reasoning-data phases that happen between pre-training and post-training. The optimization story continues there, but with different constraints.

---

**Exercises:** [Chapter 8 problem set](../../exercises/ch08.md) — includes the scaling-law extrapolation problem and the precision drills.
**Labs:** [`lab08_scaling_laws`](../../labs/lab08_scaling_laws.py) — fit $L(N) = E + AN^{-\alpha}$ against a source whose irreducible loss is *computable*, so you can mark the fitted asymptote and watch the extrapolation fail in a measurable direction. [`lab08_precision_and_stability`](../../labs/lab08_precision_and_stability.py) — real FP8 casts: watch one outlier delete every small value in a tensor, and find the threshold where scaling granularity starts to matter.

---

**References for this chapter**

- [\[1\] DeepSeek-V3](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. The FP8 case study.
- [\[5\] Llama 3](../appendix/b-references.md#5-llama-3) — Meta AI, July 2024. The BF16 case study.
- [\[6\] Qwen3](../appendix/b-references.md#6-qwen3) — Qwen Team, 2025. The MoE-on-dense case study.
- [\[21\] Scaling laws / Chinchilla](../appendix/b-references.md#21-scaling-laws) — Hoffmann et al., 2022. The compute-optimal scaling reference.
- [\[25\] AdamW](../appendix/b-references.md#25-adamw) — Loshchilov and Hutter, 2019. The decoupled weight decay reference.
- [\[43\] Beyond Chinchilla (Sardana et al. 2024)](../appendix/b-references.md#43-beyond-chinchilla-sardana-et-al-2024) — arXiv:2401.00448. The inference-aware scaling reference.
- [\[44\] µTransfer / Tensor Programs (Yang et al. 2022)](../appendix/b-references.md#44-mutransfer--mup--tensor-programs-yang-et-al-2022) — arXiv:2203.03456. The zero-shot hyperparameter transfer reference.
- [\[45\] PaLM (Chowdhery et al. 2022)](../appendix/b-references.md#45-palm-chowdhery-et-al-2022) — arXiv:2204.02311. The z-loss and Adafactor reference.
- [\[46\] QK-norm (Henry et al. 2020)](../appendix/b-references.md#46-qk-norm-henry-et-al-2020) — arXiv:2010.04245. The attention-stability reference.
- [\[39\] Korthikanti et al. 2022 (activation recomputation)](../appendix/b-references.md#39-korthikanti-2022-activation-recomputation) — the selective-recomputation analysis cited in §8.4.
- [See full reference list](../appendix/b-references.md)
