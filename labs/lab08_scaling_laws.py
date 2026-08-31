# %% [markdown]
# # Lab 8a — Scaling laws: fitting one, and finding out how far it carries
#
# Companion to [Chapter 8](../book/part-2-pretraining/08-optimization.md).
#
# A scaling law is a bet. You train a handful of models you can afford, fit a
# curve, and extrapolate to a model you cannot afford but are about to buy. The
# budget of a frontier run rests on that extrapolation being roughly right.
#
# The trouble with learning this from papers is that you never see the
# extrapolation checked, because checking it means training the big model
# anyway — at which point you did not need the law. Here we can check it,
# because the "big model" costs ten seconds.
#
# Two things make this lab honest:
#
# - The data comes from a source whose **irreducible loss $E$ is computable
#   exactly**, so we can ask whether the fit recovers it instead of taking the
#   fitted value on faith.
# - The data is **effectively infinite** — every batch is freshly sampled — so
#   nothing here is overfitting, and the curve measures capacity alone.
#
# You will:
#
# 1. Build a source that produces a genuine power law, and see why its shape
#    matters.
# 2. Train a ladder of models and watch loss fall smoothly with $N$.
# 3. Fit $L(N) = E + A N^{-\alpha}$ and compare the fitted $E$ to the true one.
# 4. Extrapolate from small models, then train the large one and measure the
#    error *and its direction*.
# 5. Refit from two points and watch a perfect-looking fit predict badly.
#
# Runs on CPU in about a minute.
#
# **Predict before you run.** Fit on small models, extrapolate up: is the
# predicted loss too optimistic or too pessimistic? Commit to a direction now.

# %%
from __future__ import annotations

import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# %%
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
DEVICE = torch.device(
    os.environ.get("FLE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
)

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
print(f"device={DEVICE}  smoke_test={SMOKE}  torch={torch.__version__}")

# %% [markdown]
# ## 1. A source that actually produces a power law
#
# **This is the part I got wrong twice, and it turned out to be the most
# interesting thing in the lab.**
#
# My first source was an order-3 Markov chain. Models could not learn it at all
# — 4,096 contexts and not enough data — so every model sat at the loss of a
# uniform predictor and "loss" was flat in $N$. My second was an order-2 chain,
# which models *could* learn, and the curve looked like this:
#
# ```
#   d=32   gap above E = 0.77
#   d=48   gap above E = 0.27      <- cliff
#   d=64   gap above E = 0.20
# ```
#
# A cliff, not a power law. And that is not a bug — it is what a finite-context
# Markov source *should* do. There are 256 conditionals to memorise; below some
# capacity you have almost none of them, above it you have essentially all of
# them, and there is very little in between.
#
# Power laws need structure at **many scales**, so that a model with twice the
# capacity always finds something new and slightly less valuable to learn. The
# standard toy model for this — and the one behind most theory of neural
# scaling laws — is a **Zipf-distributed** set of skills: context frequencies
# follow $p(\text{rank}) \propto 1/\text{rank}^{s}$, so a model learns the
# common contexts first and pays a slowly-shrinking price on the long tail.
#
# So: 4,096 contexts drawn Zipfian, each with its own peaked conditional over
# the vocabulary. The model sees a context and predicts the next symbol.

# %%
VOCAB = 16
ORDER = 3                 # context is rendered as 3 base-16 symbols
N_CONTEXTS = 4096
ZIPF_S = 1.0              # exponent of the context frequency distribution
CONCENTRATION = 0.3       # Dirichlet alpha: low = peaked conditionals


class ZipfContextSource:
    """Zipf-frequent contexts, each with a fixed random conditional."""

    def __init__(self, n_ctx: int, vocab: int, order: int,
                 zipf_s: float, alpha: float, seed: int = SEED):
        self.n_ctx, self.vocab, self.order = n_ctx, vocab, order
        rank = torch.arange(1, n_ctx + 1, dtype=torch.float)
        self.ctx_p = (1.0 / rank ** zipf_s)
        self.ctx_p /= self.ctx_p.sum()

        gamma = torch.distributions.Gamma(
            torch.full((n_ctx, vocab), alpha), torch.ones(n_ctx, vocab)
        )
        weights = gamma.sample()
        self.cond = weights / weights.sum(-1, keepdim=True)

        ids = torch.arange(n_ctx)
        self.ctx_tokens = torch.stack(
            [(ids // (vocab ** (order - 1 - i))) % vocab for i in range(order)], dim=1
        )

    def true_E(self) -> float:
        """Exact irreducible loss: frequency-weighted entropy of the conditionals."""
        H = -(self.cond * self.cond.clamp_min(1e-12).log()).sum(-1)
        return float((self.ctx_p * H).sum())

    def head_mass(self, k: int) -> float:
        """What fraction of samples come from the k most common contexts?"""
        return float(self.ctx_p[:k].sum())

    def batch(self, bs: int, gen: torch.Generator) -> tuple:
        c = torch.multinomial(self.ctx_p, bs, replacement=True, generator=gen)
        y = torch.multinomial(self.cond[c], 1, generator=gen).squeeze(1)
        return self.ctx_tokens[c], y


SOURCE = ZipfContextSource(N_CONTEXTS, VOCAB, ORDER, ZIPF_S, CONCENTRATION)
E_TRUE = SOURCE.true_E()
UNIFORM = math.log(VOCAB)

print(f"\nSource: {N_CONTEXTS:,} Zipf(s={ZIPF_S}) contexts over {VOCAB} symbols")
print("-" * 74)
print(f"{'loss of a uniform predictor':<36}: {UNIFORM:.4f} nats")
print(f"{'irreducible loss E (computed exactly)':<36}: {E_TRUE:.4f} nats")
print(f"{'headroom available to a model':<36}: {UNIFORM - E_TRUE:.4f} nats")
print()
for k in (10, 100, 1000):
    print(f"  top {k:>5,} contexts cover {100*SOURCE.head_mass(k):>5.1f}% of samples")
print("\nThat concentration is the whole reason a power law appears. Capacity")
print("spent on the first 10 contexts buys a lot; the next 990 buy less each;")
print("and there is always more tail. E is not fitted here -- it is computed")
print("from the generating distribution, so section 3 can be marked.")

# %% [markdown]
# ## 2. A ladder of models
#
# We scale width, count **non-embedding** parameters (the Chinchilla
# convention), and scale the learning rate as $1/\sqrt{d}$.
#
# That last detail is not cosmetic. Holding the learning rate fixed across the
# ladder made my larger models come out *worse* than my medium ones, which
# turns a scaling law into noise. A "scaling law" measured with an untuned
# optimiser is measuring your optimiser, and Chapter 8 section 8.5 is about
# exactly this.

# %%
class ContextNet(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_layers: int, order: int,
                 n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(order, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(d_model),
                nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                nn.LayerNorm(d_model),
                nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                              nn.Linear(4 * d_model, d_model)),
            ]))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def non_embedding_params(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        return total - self.embed.weight.numel() - self.pos.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(t, device=x.device))
        for ln1, attn, ln2, mlp in self.blocks:
            a = ln1(h)
            h = h + attn(a, a, a, need_weights=False)[0]
            h = h + mlp(ln2(h))
        return self.head(self.norm(h))[:, -1]


def train_model(d_model: int, n_layers: int, steps: int, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    model = ContextNet(VOCAB, d_model, n_layers, ORDER).to(DEVICE)
    lr = 3e-3 * math.sqrt(32.0 / d_model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x, y = SOURCE.batch(256, gen)
        loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    eval_gen = torch.Generator().manual_seed(12345)
    total, n = 0.0, 0
    with torch.no_grad():
        for _ in range(20):
            x, y = SOURCE.batch(512, eval_gen)
            total += float(F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))) * y.numel()
            n += y.numel()
    return {"N": model.non_embedding_params(), "loss": total / n,
            "d_model": d_model, "n_layers": n_layers, "lr": lr}


STEPS = 250 if SMOKE else 1000
LADDER = ([(16, 1), (32, 2), (64, 2), (128, 2)] if SMOKE
          else [(16, 1), (24, 1), (32, 2), (48, 2), (64, 2), (96, 2), (160, 2)])

print(f"\nTraining {len(LADDER)} models for {STEPS} steps each, on fresh data")
print("-" * 74)
print(f"{'d_model':>8} {'layers':>7} {'lr':>8} {'N (non-emb)':>13} {'eval loss':>11} {'above E':>9}")
runs = []
for d, L in LADDER:
    r = train_model(d, L, STEPS)
    runs.append(r)
    print(f"{d:>8} {L:>7} {r['lr']:>8.4f} {r['N']:>13,} {r['loss']:>11.4f} "
          f"{r['loss']-E_TRUE:>9.4f}")

monotone = all(runs[i]["loss"] >= runs[i + 1]["loss"] for i in range(len(runs) - 1))
print(f"\nmonotonically improving with N: {monotone}")
if not monotone:
    print("A non-monotone ladder means some model was under-trained, not that")
    print("scaling broke. Fix the optimiser before you fit anything to this.")

# %% [markdown]
# ## 3. Fitting $L(N) = E + A N^{-\alpha}$
#
# Three parameters, one of whose true value we happen to know. We fit all three
# as if we did not, then mark the answer.
#
# No scipy needed: grid-search $E$, and for each candidate the rest is linear in
# log space, since $\log(L - E) = \log A - \alpha \log N$.

# %%
def linreg(xs: list, ys: list) -> tuple:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx


def fit_scaling(rs: list, grid: int = 3000) -> dict:
    Ns = [r["N"] for r in rs]
    Ls = [r["loss"] for r in rs]
    best = None
    for i in range(grid):
        E = min(Ls) * i / grid
        resid = [L - E for L in Ls]
        if min(resid) <= 1e-9:
            continue
        slope, intercept = linreg([math.log(N) for N in Ns],
                                  [math.log(r) for r in resid])
        A, alpha = math.exp(intercept), -slope
        sse = sum((E + A * N ** (-alpha) - L) ** 2 for N, L in zip(Ns, Ls))
        if best is None or sse < best["sse"]:
            best = {"sse": sse, "E": E, "A": A, "alpha": alpha}
    return best


def predict(fit: dict, N: int) -> float:
    return fit["E"] + fit["A"] * N ** (-fit["alpha"])


full_fit = fit_scaling(runs)
print("\nFit using every model in the ladder")
print("-" * 74)
print(f"{'fitted E':<16}: {full_fit['E']:.4f}   true E = {E_TRUE:.4f}   "
      f"error {full_fit['E']-E_TRUE:+.4f}")
print(f"{'fitted alpha':<16}: {full_fit['alpha']:.4f}")
print(f"{'residual SSE':<16}: {full_fit['sse']:.2e}")
print()
print(f"{'N':>13} {'actual':>10} {'fitted':>10} {'error':>9}")
for r in runs:
    p = predict(full_fit, r["N"])
    print(f"{r['N']:>13,} {r['loss']:>10.4f} {p:>10.4f} {p-r['loss']:>+9.4f}")

print(f"\nThe curve fits well (SSE {full_fit['sse']:.1e}) and the fitted E is")
print(f"off by {full_fit['E']-E_TRUE:+.3f} nats. Sit with that: the functional")
print("form is right, the data is clean, there is no noise from overfitting,")
print("and the asymptote is still wrong. E and alpha trade off against each")
print("other, so a fit can track every point you HAVE and still misplace the")
print("floor you have not reached.")

# %% [markdown]
# ## 4. The extrapolation, actually checked
#
# Now the part a paper cannot show you. Fit on the small models only, predict
# the large ones, then look at what the large ones really did.

# %%
n_hold = 2 if len(runs) > 4 else 1
small_runs, held = runs[:-n_hold], runs[-n_hold:]
small_fit = fit_scaling(small_runs)
largest_fitted = max(r["N"] for r in small_runs)

print(f"\nFit on the {len(small_runs)} smallest, extrapolate to the largest {n_hold}")
print("-" * 74)
print(f"fitted E = {small_fit['E']:.4f}   alpha = {small_fit['alpha']:.4f}   "
      f"(true E = {E_TRUE:.4f})")
print()
print(f"{'N':>13} {'predicted':>11} {'actual':>10} {'error':>9} {'beyond fit':>12}")
for r in held:
    p = predict(small_fit, r["N"])
    print(f"{r['N']:>13,} {p:>11.4f} {r['loss']:>10.4f} {p-r['loss']:>+9.4f} "
          f"{r['N']/largest_fitted:>11.1f}x")

errs = [predict(small_fit, r["N"]) - r["loss"] for r in held]
if errs[-1] < 0:
    print("\nThe law was OPTIMISTIC: it promised a lower loss than the model")
    print("actually delivered.")
    if len(errs) > 1 and abs(errs[-1]) > abs(errs[0]):
        print(f"And the error grew with reach: {errs[0]:+.4f} at "
              f"{held[0]['N']/largest_fitted:.1f}x, {errs[-1]:+.4f} at "
              f"{held[-1]['N']/largest_fitted:.1f}x.")
    print("\nThat direction is the one that costs money. An optimistic law says")
    print("a given loss is reachable with fewer parameters or fewer tokens than")
    print("it really needs, so the run is scoped too small and either misses its")
    print("target or overruns its budget. Every published law is fitted in the")
    print("regime where models are cheap and applied in the regime where they")
    print("are not.")
else:
    print("\nThe law was PESSIMISTIC on this run: the real model beat the")
    print("prediction. That happens too, and the honest summary is that the")
    print("SIGN of the error is not reliable -- only its growth with distance is.")

# %% [markdown]
# ## 5. Two points are not a scaling law
#
# Exercise 8.9 item 2. Two points determine a line, so a two-point fit in log
# space is exact by construction: zero residual, perfect-looking, and carrying
# no information whatsoever about curvature — which is the only thing that
# matters when extrapolating.

# %%
target = runs[-1]
print("\nHow many models did you fit on?")
print("-" * 74)
print(f"{'fit on':<13} {'fitted E':>10} {'alpha':>8} {'SSE':>10} "
      f"{'pred @ top':>12} {'actual':>9} {'error':>9}")
for k in range(2, len(runs)):
    f = fit_scaling(runs[:k])
    p = predict(f, target["N"])
    print(f"{f'{k} smallest':<13} {f['E']:>10.4f} {f['alpha']:>8.4f} {f['sse']:>10.1e} "
          f"{p:>12.4f} {target['loss']:>9.4f} {p-target['loss']:>+9.4f}")

print("\nRead the SSE column and the E column together. The two-point fit has")
print("the best residual of any row and one of the worst answers -- because with")
print("three free parameters and two constraints it is not fitting, it is")
print("interpolating. Goodness of fit tells you nothing about extrapolation")
print("when the model has as many knobs as you have points.")
print("\nThe practical rule: the asymptote is the expensive parameter. Alpha is")
print("usually recoverable from a few cheap models; E needs models close enough")
print("to the floor to feel it, which is exactly the regime you were trying to")
print("avoid paying for.")

# %% [markdown]
# ## 6. Things to try
#
# **1. Set `ZIPF_S = 2.0` and rerun.**
# *Common prediction:* a steeper power law.
# *What happens:* the head gets much heavier — the top 10 contexts cover most
# of the data — so even tiny models capture nearly everything and the curve
# flattens early. A steeper Zipf gives you a *shallower* loss curve, because
# there is less tail left to buy. The exponent of your data distribution sets
# the exponent of your scaling law.
#
# **2. Set `CONCENTRATION = 3.0` and rerun.**
# *Common prediction:* the models get worse.
# *What happens:* the conditionals flatten toward uniform, $E$ climbs toward
# $\log 16$, and the headroom nearly vanishes. Every model lands in the same
# place and $\alpha$ becomes meaningless. Scaling laws describe how fast you
# close a gap; with no gap there is no law.
#
# **3. Remove the `1/sqrt(d)` learning-rate scaling.**
# *Common prediction:* slightly noisier points.
# *What happens:* the ladder goes non-monotone — larger models come out worse —
# and the `monotone` check in section 2 flips to `False`. This is worth doing
# once, because it is what a broken scaling study looks like from the inside.
#
# **4. Fit on the 3 LARGEST models instead of the smallest, and extrapolate
# down.**
# *Common prediction:* symmetric error.
# *What happens:* it is far more accurate. Extrapolating toward the asymptote
# you have already approached is easy; extrapolating away from it is the hard
# direction, and it is the only direction anyone actually needs.
#
# **5. Train each rung with three seeds and add error bars.**
# *Common prediction:* noise is negligible next to the trend.
# *What happens:* at the small end the seed spread is a real fraction of the
# gap between adjacent rungs — the same lesson
# [`lab05_attention_variants`](lab05_attention_variants.py) ran into, in a
# different guise.

# %% [markdown]
# ## What to take away
#
# 1. **Power laws come from the data, not from neural networks.** A source with
#    structure at one scale gives a cliff; a Zipf-distributed source gives a
#    power law. The exponent you fit is a property of your corpus.
# 2. **A scaling law has three parameters and they are not equally knowable.**
#    $\alpha$ is cheap; $E$ is expensive, because pinning an asymptote needs
#    models near it. Even a clean fit over seven models misplaced $E$ here.
# 3. **Extrapolation error grows with distance, and the direction matters.**
#    An optimistic law under-scopes a run — the expensive failure.
# 4. **A perfect residual can mean you have no evidence.** The two-point fit has
#    the lowest SSE and among the worst predictions.
# 5. **Tune the optimiser before you fit the law.** An untuned learning rate
#    makes big models look worse than small ones, and what you then measure is
#    your optimiser, not your architecture.
