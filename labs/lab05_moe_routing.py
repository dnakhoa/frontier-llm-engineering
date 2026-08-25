# %% [markdown]
# # Lab 5 — MoE routing, load imbalance, and two ways to fix it
#
# Companion to [Chapter 5](../book/part-2-pretraining/05-architecture.md).
#
# Mixture-of-Experts buys you the quality of a large model at the compute of a
# small one, and charges you in memory, communication, and one failure mode that
# will eat your throughput if you let it: **expert load imbalance**.
#
# You will:
#
# 1. Build a top-k router and a real MoE layer.
# 2. Turn off load balancing and watch substantial load imbalance develop --
#    a ~3x straggler tax, which is exactly the Chapter 5 section 5.7 scenario.
# 3. Fix it with the standard auxiliary loss, and measure what it costs.
# 4. Fix it with DeepSeek's **auxiliary-loss-free** bias controller, and see why
#    keeping balance out of the objective is the better trade.
# 5. Vary the expert count and find the counterintuitive result about load
#    variance that explains why frontier models use many small experts.
#
# CPU-only, runs in about a minute.
#
# **An honest note on scale.** This lab reproduces *load imbalance* reliably.
# It does not reach the fully-collapsed state -- experts receiving zero tokens
# and never recovering -- which is reported at production scale over far more
# steps than a CPU lab can run. What you will see is the mechanism that leads
# there, measured, and that is the part worth understanding.
#
# **Predict before you run.** Section 2 asks how skewed the load gets. Write a
# max/mean ratio down.

# %%
import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# %%
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
DEVICE = torch.device(
    os.environ.get("FLE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
)
SEED = 0
torch.manual_seed(SEED)

print(f"device={DEVICE}  smoke_test={SMOKE}  torch={torch.__version__}")

# %% [markdown]
# ## 1. A real MoE layer
#
# The router scores each token against every expert, takes the top-k, and
# dispatches. Two details matter and are easy to get wrong:
#
# - **Gating weights come from the softmax over the selected experts**, and they
#   must be renormalized after selection so the outputs are a convex combination.
# - **The bias affects *selection* but not the *weights*.** That separation is
#   what makes the auxiliary-loss-free scheme work (section 4), and merging them
#   quietly puts balance back into the gradient.


# %%
class Expert(nn.Module):
    """One SwiGLU FFN. Three matrices, not two."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoELayer(nn.Module):
    """Top-k routed MoE with three selectable balancing strategies."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k: int = 2,
        balance: str = "none",          # "none" | "aux_loss" | "bias"
        aux_weight: float = 0.01,
        bias_update_rate: float = 2e-2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance = balance
        self.aux_weight = aux_weight
        self.bias_update_rate = bias_update_rate

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))

        # The bias is a CONTROL variable, not a parameter. It is deliberately
        # not in the gradient path -- see section 4.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("token_counts", torch.zeros(n_experts))

        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)                       # (N, d)
        n_tokens = flat.shape[0]

        logits = self.router(flat)                    # (N, E)
        probs = F.softmax(logits, dim=-1)

        # Selection uses the biased score; weighting uses the unbiased one.
        select_on = logits + self.expert_bias if self.balance == "bias" else logits
        _, topk_idx = torch.topk(select_on, self.top_k, dim=-1)         # (N, k)

        topk_probs = probs.gather(-1, topk_idx)                        # (N, k)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)     # renormalize

        # Dispatch. A real implementation does this with an all-to-all; here we
        # loop, which is clear and slow and fine at this scale.
        out = torch.zeros_like(flat)
        counts = torch.zeros(self.n_experts, device=flat.device)
        for e in range(self.n_experts):
            hit = topk_idx == e                                        # (N, k)
            tok_idx, slot = hit.nonzero(as_tuple=True)
            counts[e] = tok_idx.numel()
            if tok_idx.numel() == 0:
                continue
            weights = topk_probs[tok_idx, slot].unsqueeze(-1)
            out[tok_idx] += weights * self.experts[e](flat[tok_idx])

        self.token_counts = counts.detach()

        # Standard auxiliary loss: E * sum_i f_i * P_i, minimized at uniform.
        # f_i = fraction of tokens routed to i; P_i = mean router probability.
        if self.balance == "aux_loss":
            f = counts / max(1, n_tokens * self.top_k)
            P = probs.mean(dim=0)
            self.aux_loss = self.aux_weight * self.n_experts * (f * P).sum()
        else:
            self.aux_loss = torch.tensor(0.0, device=flat.device)

        return out.reshape(b, t, d)

    @torch.no_grad()
    def update_bias(self):
        """Bias-based balancing, applied AFTER the optimizer step.

        Over-subscribed experts get their bias decreased so they are selected
        less; under-subscribed experts get it increased. The language-modelling
        loss never sees this -- which is the whole point (section 4).
        """
        if self.balance != "bias":
            return
        total = self.token_counts.sum()
        if total == 0:
            return
        share = self.token_counts / total
        target = 1.0 / self.n_experts
        # Proportional to the error, not just its sign -- converges far faster
        # at this scale. Production implementations often use sign() with a
        # smaller rate over many more steps.
        self.expert_bias += self.bias_update_rate * (target - share) * self.n_experts


# %% [markdown]
# ## 2. Watch the imbalance develop
#
# A tiny model on a synthetic task. What matters is not the task but the routing
# dynamics, which are the same at any scale.
#
# The feedback loop: an expert that receives slightly more tokens gets more
# gradient, becomes slightly better, so the router prefers it more, so it
# receives more. Nothing in the objective opposes this.


# %%
class TinyMoEModel(nn.Module):
    def __init__(self, vocab, d_model, n_experts, top_k, balance, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, 4 * d_model, n_experts, top_k, balance, **kw)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        h = self.emb(ids)
        h = h + self.moe(self.norm(h))
        return self.head(h)


def make_batch(
    batch: int, seq: int, vocab: int, gen: torch.Generator,
    n_topics: int = 4, skew: float = 3.0,
):
    """Synthetic next-token task with structure, so routing has something to
    specialize on: several 'topics', each with its own token subrange.

    `skew` controls how unevenly topics occur. Real corpora are heavily skewed,
    and a skewed data distribution is what turns mild routing drift into the
    straggler problem -- with uniform topics, balanced routing is simply
    optimal and no imbalance develops.
    """
    w = torch.tensor([skew ** -i for i in range(n_topics)])
    w = w / w.sum()
    topic = torch.multinomial(w, batch, replacement=True, generator=gen)
    span = vocab // n_topics
    lo = (topic * span).unsqueeze(1)
    ids = lo + torch.randint(0, span, (batch, seq + 1), generator=gen)
    return ids[:, :-1].to(DEVICE), ids[:, 1:].to(DEVICE)


def gini(counts: torch.Tensor) -> float:
    """0 = perfectly balanced, approaching 1 = one expert takes everything."""
    x = counts.float().sort().values
    n = x.numel()
    if x.sum() == 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float, device=x.device)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def train(balance: str, steps: int, n_experts: int = 32, top_k: int = 2, **kw):
    torch.manual_seed(SEED)                # identical init across strategies
    gen = torch.Generator().manual_seed(SEED)

    vocab, d_model = 256, 64
    model = TinyMoEModel(vocab, d_model, n_experts, top_k, balance, **kw).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    history = []
    for step in range(steps):
        ids, targets = make_batch(16, 24, vocab, gen)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        total = loss + model.moe.aux_loss

        opt.zero_grad()
        total.backward()
        opt.step()
        model.moe.update_bias()            # no-op unless balance == "bias"

        counts = model.moe.token_counts
        history.append(
            {
                "step": step,
                "loss": loss.item(),
                "gini": gini(counts),
                "max_over_mean": float(counts.max() / counts.float().mean()),
                "dead": int((counts == 0).sum()),
            }
        )
    return model, history


STEPS = 60 if SMOKE else 400

print("\nTraining three balancing strategies from identical initialization...")
runs = {}
for balance in ("none", "aux_loss", "bias"):
    print(f"  {balance} ...", flush=True)
    runs[balance] = train(balance, STEPS)

# %%
print("\nLoad balance over training (32 experts, top-2, skewed topics)")
print("-" * 78)
print(f"{'step':>6}" + "".join(f"{b:>24}" for b in runs))
print(f"{'':>6}" + "".join(f"{'gini  max/mean  dead':>24}" for _ in runs))
marks = [0, STEPS // 4, STEPS // 2, 3 * STEPS // 4, STEPS - 1]
for s in marks:
    row = f"{s:>6}"
    for balance, (_, hist) in runs.items():
        h = hist[s]
        row += f"{h['gini']:>9.3f}{h['max_over_mean']:>9.2f}{h['dead']:>6d}"
    print(row)

print("\nFinal loss:", {b: round(h[-1]["loss"], 4) for b, (_, h) in runs.items()})

# %% [markdown]
# Read the `none` column. Gini climbs and max/mean reaches roughly 3x — one
# expert absorbing three times its fair share of tokens.
#
# That number is the whole point. Because the MoE all-to-all is a **synchronous
# collective**, every GPU waits for the one holding the busiest expert. A 3x
# straggler does not cost you that expert's throughput; it costs the entire
# cluster's throughput, multiplied across every layer and every step
# (Chapter 5 section 5.7).
#
# Note what you do *not* see here: experts at exactly zero. Full collapse is an
# absorbing state that takes far more steps and far more experts than a CPU lab
# reaches. What you see is the mechanism -- a self-reinforcing feedback loop
# where more tokens means more gradient means more router preference -- running
# for as long as this lab can afford.

# %% [markdown]
# ## 3. Final expert distribution

# %%
print("\nFinal token counts per expert")
print("-" * 78)
for balance, (model, _) in runs.items():
    counts = model.moe.token_counts.cpu().float()
    srt = counts.sort(descending=True).values
    share = 100 * srt / srt.sum()
    top = " ".join(f"{s:>5.1f}%" for s in share[:6])
    bot = " ".join(f"{s:>5.1f}%" for s in share[-3:])
    print(
        f"{balance:>10}: busiest 6 = {top}  ...  quietest 3 = {bot}\n"
        f"{'':>10}  gini={gini(counts):.3f}  max/mean={float(srt[0]/srt.mean()):.2f}x"
    )

# %% [markdown]
# ## 4. Why the bias controller is the better fix
#
# Both `aux_loss` and `bias` achieve balance. They differ in **where the
# constraint lives**.
#
# The auxiliary loss adds a term to the objective that is *not what you want the
# model to learn*. The router wants to send a token to the expert that handles
# it best; the auxiliary loss pushes it elsewhere for balance. Those fight, and
# the exchange rate (`aux_weight`) has no principled value.
#
# The bias controller achieves balance **without touching the loss at all**. The
# gradient signal stays purely "route to the expert that lowers the loss," and
# balance is enforced by a separate feedback loop on *selection*.
#
# The general principle, and it recurs in Chapter 12's argument for gating
# safety rather than summing it: **when a constraint conflicts with your
# objective, enforce it outside the objective rather than adding a competing
# term.**

# %%
print("\nThe cost of putting balance in the objective")
print("-" * 78)
print(f"{'aux_weight':>12} {'final loss':>12} {'gini':>8} {'max/mean':>10}")
weights = [0.0, 0.01, 0.1] if SMOKE else [0.0, 0.001, 0.01, 0.05, 0.2]
for w in weights:
    _, hist = train("aux_loss", STEPS, aux_weight=w)
    h = hist[-1]
    print(f"{w:>12.3f} {h['loss']:>12.4f} {h['gini']:>8.3f} {h['max_over_mean']:>10.2f}")

_, bias_hist = train("bias", STEPS)
hb = bias_hist[-1]
print(f"{'bias ctrl':>12} {hb['loss']:>12.4f} {hb['gini']:>8.3f} {hb['max_over_mean']:>10.2f}")
print("\nCompare the bias row against the aux_weight row with similar gini.")
print("That loss difference is the tax you pay for putting balance in the")
print("objective -- and it is what DeepSeek's scheme avoids (Chapter 5 s5.10).")

# %% [markdown]
# ## 5. Expert count and the counterintuitive variance result
#
# **Predict first:** with active parameters held roughly fixed, does going from
# 8 experts to 64 make load balance better or worse?

# %%
print("\nLoad statistics vs expert count (no balancing, so the effect is visible)")
print("-" * 78)
print(f"{'experts':>9} {'top_k':>6} {'gini':>8} {'max/mean':>10} {'dead':>6} {'straggler tax':>15}")
configs = [(8, 2), (16, 2), (32, 2)] if SMOKE else [(8, 2), (16, 2), (32, 2), (64, 4)]
for n_e, k in configs:
    model, hist = train("none", STEPS, n_experts=n_e, top_k=k)
    counts = model.moe.token_counts
    h = hist[-1]
    # With EP, the slowest device sets the pace. The straggler tax is how much
    # longer the busiest expert takes than the average one.
    tax = h["max_over_mean"]
    print(
        f"{n_e:>9} {k:>6} {h['gini']:>8.3f} {h['max_over_mean']:>10.2f} "
        f"{h['dead']:>6d} {tax:>14.2f}x"
    )

print("\nThe gini rises (relative variance is worse with fewer tokens per")
print("expert) while the SHARE any single hot expert absorbs falls. With 8")
print("experts one expert at 3x its share takes 37% of all tokens; with 64,")
print("only 4.7%. Since the all-to-all is synchronous, that bound on the")
print("straggler is what matters -- which is why frontier MoE trends toward")
print("more, smaller experts (Chapter 5 s5.4).")

# %% [markdown]
# ## 6. Things to try
#
# **1. Set `skew=1.0` in `make_batch` (uniform topics).**
# *Common prediction:* imbalance still develops, just more slowly.
# *What happens:* it largely does not. With uniform data, balanced routing is
# optimal and the router finds it. **Imbalance is driven by the data
# distribution, not by the routing mechanism alone** -- which is why real
# corpora, which are heavily skewed, need balancing and toy uniform benchmarks
# do not.
#
# **2. Set `top_k=1` with no balancing.**
# *Common prediction:* balance improves, since there are fewer assignments.
# *What happens:* balance gets **worse** and collapse is faster. With top-2, a
# token going to a collapsing expert still gives gradient to a second expert,
# which keeps alternatives alive. Top-1 is winner-take-all per token, which is
# exactly the feedback loop.
#
# **3. Raise `bias_update_rate` by 10x.**
# *Common prediction:* balance converges faster.
# *What happens:* the load histogram **oscillates**. The bias update is a
# proportional controller with no damping, so too high a rate overshoots. Plot
# `max_over_mean` per step and you can see the ringing.
#
# **4. Set `aux_weight=1.0`.**
# *Common prediction:* perfect balance, slightly worse loss.
# *What happens:* near-perfect balance and a **substantially** worse loss — the
# router is now optimizing balance more than it is optimizing prediction. This
# is the exchange-rate problem, at its unfavourable end.
#
# **5. Raise `skew` to 10.0.**
# *Common prediction:* the expert handling it gets more tokens; balancing fixes it.
# *What happens:* balancing forces uniform load even though the *data* is not
# uniform, so tokens from the common topic get pushed to experts that are worse
# at them. Balance and specialization genuinely conflict, and the auxiliary loss
# resolves it in favour of balance whether you wanted that or not.
#
# **6. Remove `topk_probs` renormalization.**
# *Common prediction:* small quality change.
# *What happens:* outputs are no longer a convex combination, so their magnitude
# varies with how confident the router was. The residual stream picks up a
# scale that depends on routing confidence, which destabilizes training.

# %% [markdown]
# ## What to take away
#
# 1. **Load imbalance is self-reinforcing.** More tokens means more gradient
#    means a better expert means more router preference. Nothing in the loss
#    opposes it, and this lab measures roughly a 3x straggler after a few
#    hundred steps.
# 2. **Imbalance is driven by skewed data**, not by routing alone — and every
#    real corpus is skewed.
# 3. **The auxiliary loss works and taxes quality**, because it adds a term that
#    competes with the objective and its weight has no principled value.
# 4. **The bias controller achieves balance outside the objective**, which is
#    why DeepSeek-V3's auxiliary-loss-free scheme is a genuine contribution.
# 5. **More, smaller experts are noisier per expert and more robust in
#    aggregate**, because the synchronous all-to-all makes the *worst* expert's
#    share the thing that matters.
# 6. **Balance and specialization conflict.** Balancing is necessary and it is
#    not free, and being explicit about which you are buying is the job.
