# %% [markdown]
# # Lab 9 — Extending RoPE: PI, NTK-aware, and YaRN
#
# Companion to [Chapter 9](../book/part-2-pretraining/09-mid-training.md).
#
# A model trained at 2K context does not work at 8K. The usual explanation —
# "it has not seen those positions" — is half right and hides the mechanism.
# RoPE encodes position as a **rotation**, and every extension method is a
# different answer to one question: which rotation frequencies are you willing
# to distort?
#
# You will:
#
# 1. Look at RoPE's geometry directly and find which dimensions have even
#    completed a full rotation inside the training window.
# 2. Train a model on a task that genuinely needs long-range lookup, so
#    breaking long-range attention actually shows up.
# 3. Evaluate 4× beyond the training length with no modification.
# 4. Apply Position Interpolation and watch it break something that was
#    working.
# 5. Apply NTK-aware scaling and YaRN, and see what each preserves.
# 6. Fine-tune briefly at the extended length and find which method needed it.
#
# Runs on CPU in a couple of minutes.
#
# **Predict before you run.** Position Interpolation divides every position
# index by the extension factor. What does that do to *short* positions, which
# were working perfectly well before? Write it down.

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

TRAIN_LEN = 128
EVAL_LEN = 384 if SMOKE else 512
SCALE = EVAL_LEN / TRAIN_LEN

D_MODEL, N_HEADS = 64, 2
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 2
BASE = 10000.0

# %% [markdown]
# ## 1. The geometry, before any model
#
# RoPE rotates each pair of dimensions by an angle proportional to position.
# Dimension pair $i$ rotates at frequency
# $\theta_i = \text{base}^{-2i/d}$, so its **wavelength** — the number of
# positions needed for one full rotation — is $2\pi/\theta_i$.
#
# That wavelength is the whole story. A dimension whose wavelength fits inside
# the training window has been seen through every phase it can take. A
# dimension whose wavelength is longer than the training window has only ever
# been observed through part of one rotation, and positions past the training
# length take it somewhere genuinely new.

# %%
def inv_freq(head_dim: int = HEAD_DIM, base: float = BASE) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


freqs = inv_freq()
wavelengths = 2 * math.pi / freqs

print(f"\nRoPE frequencies for head_dim={HEAD_DIM}, base={BASE:.0f}")
print("-" * 74)
print(f"{'dim pair':>9} {'theta':>12} {'wavelength':>13} {'rotations in 128':>18}")
for i in range(len(freqs)):
    rot = TRAIN_LEN / float(wavelengths[i])
    tag = "" if rot >= 1 else "   <- never completes one"
    print(f"{i:>9} {float(freqs[i]):>12.5f} {float(wavelengths[i]):>13.1f} "
          f"{rot:>18.2f}{tag}")

n_complete = int((TRAIN_LEN / wavelengths >= 1).sum())
print(f"\n{n_complete} of {len(freqs)} dimension pairs complete at least one full")
print(f"rotation within the {TRAIN_LEN}-token training window.")
print("\nThe fast dimensions encode fine local offsets and have seen every phase.")
print("The slow ones encode coarse 'how far back roughly' and have only ever")
print("been seen through a fraction of a turn. Extending context is dangerous")
print("for the slow dimensions and unnecessary for the fast ones -- which is")
print("exactly the distinction the three methods below treat differently.")

# %% [markdown]
# ## 2. Four ways to place a position
#
# - **none** — use the position as-is. Beyond the training length the slow
#   dimensions rotate into angles never seen.
# - **Position Interpolation (PI)** — divide every position by $s$. All angles
#   land back in the trained range, *including the fast ones*, which now have to
#   resolve $s$ times finer differences than they were trained to.
# - **NTK-aware** — raise the base instead, so slow dimensions are stretched and
#   fast ones are left almost untouched.
# - **YaRN** — do both, selectively: interpolate dimensions whose wavelength
#   exceeds the training window, leave the short-wavelength ones alone, and ramp
#   between.

# %%
def rope_tables(seq_len: int, mode: str = "none", scale: float = 1.0,
                head_dim: int = HEAD_DIM, base: float = BASE,
                train_len: int = TRAIN_LEN) -> tuple:
    pos = torch.arange(seq_len, dtype=torch.float)
    inv = inv_freq(head_dim, base)

    if mode == "pi":
        pos = pos / scale
    elif mode == "ntk":
        # Raise the base so the SLOWEST dimension is interpolated by ~scale
        # while the fastest is essentially unchanged.
        inv = inv_freq(head_dim, base * (scale ** (head_dim / (head_dim - 2))))
    elif mode == "yarn":
        wavelen = 2 * math.pi / inv
        # ratio > 1 means the wavelength fits inside the training window.
        ratio = train_len / wavelen
        ramp = ((ratio - 1.0) / (32.0 - 1.0)).clamp(0, 1)
        inv = (inv / scale) * (1 - ramp) + inv * ramp
    elif mode != "none":
        raise ValueError(mode)

    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


print("\nEffective wavelength per dimension pair under each method (scale=4)")
print("-" * 74)
print(f"{'dim pair':>9} {'native':>11} {'PI':>11} {'NTK':>11} {'YaRN':>11}")


def effective_wavelength(mode: str, scale: float) -> torch.Tensor:
    cos, sin = rope_tables(4, mode, scale)
    # Recover the per-dim angle step from the table.
    step = torch.atan2(sin[1], cos[1]).abs().clamp_min(1e-9)
    return 2 * math.pi / step


native = effective_wavelength("none", 1.0)
pi_w = effective_wavelength("pi", SCALE)
ntk_w = effective_wavelength("ntk", SCALE)
yarn_w = effective_wavelength("yarn", SCALE)
for i in range(len(native)):
    print(f"{i:>9} {float(native[i]):>11.1f} {float(pi_w[i]):>11.1f} "
          f"{float(ntk_w[i]):>11.1f} {float(yarn_w[i]):>11.1f}")

print("\nRead the PI column: every wavelength is stretched by the same factor,")
print("including dimension 0, whose job was to distinguish adjacent tokens.")
print("NTK barely touches the fast dimensions and stretches the slow ones.")
print("YaRN is the explicit version of the same policy.")

# %% [markdown]
# ## 3. A task that actually needs long-range attention
#
# This part took me two attempts. My first task was a local one — an order-3
# Markov source — and extending the context changed *nothing*, because RoPE is
# **relative**: an attention score depends on $m - n$, not on $m$ and $n$
# separately. If your model only ever looks back three tokens, it does not care
# how large the absolute positions are.
#
# To break long-range attention you need a model that uses it. So: a block of
# key–value definitions, then a long run of queries. Answering a query means
# attending back to its definition, and the further into the sequence a query
# sits, the longer that reach.

# %%
N_KEYS, N_VALS = 8, 8
KEY0, VAL0 = 0, N_KEYS
VOCAB = N_KEYS + N_VALS


def make_batch(n: int, seq_len: int, gen: torch.Generator) -> tuple:
    """[defs...][queries...]; loss is measured only on the value after a query."""
    n_pairs = min(seq_len // 4, N_KEYS)
    x = torch.zeros(n, seq_len, dtype=torch.long)
    mask = torch.zeros(n, seq_len, dtype=torch.bool)
    for b in range(n):
        keys = torch.randperm(N_KEYS, generator=gen)[:n_pairs]
        vals = torch.randint(0, N_VALS, (n_pairs,), generator=gen)
        seq, targets = [], []
        for i in range(n_pairs):
            seq += [KEY0 + int(keys[i]), VAL0 + int(vals[i])]
        while len(seq) < seq_len:
            j = int(torch.randint(0, n_pairs, (1,), generator=gen))
            targets.append(len(seq) + 1)
            seq += [KEY0 + int(keys[j]), VAL0 + int(vals[j])]
        x[b] = torch.tensor(seq[:seq_len])
        for p in targets:
            if p < seq_len:
                mask[b, p] = True
    return x, mask


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, h, cos, sin, temp: float = 1.0):
        b, t, _ = h.shape
        x = self.n1(h)
        q = self.q(x).view(b, t, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(b, t, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(b, t, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos[None, None], sin[None, None])
        k = apply_rope(k, cos[None, None], sin[None, None])
        att = (q @ k.transpose(-2, -1)) / (math.sqrt(HEAD_DIM) * temp)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), 1)
        att = att.masked_fill(causal, float("-inf"))
        out = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(b, t, D_MODEL)
        h = h + self.o(out)
        return h + self.mlp(self.n2(h))


class RopeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x, mode: str = "none", scale: float = 1.0,
                temp: float = 1.0):
        cos, sin = rope_tables(x.shape[1], mode, scale)
        cos, sin = cos.to(x.device), sin.to(x.device)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, cos, sin, temp)
        return self.head(self.norm(h))


def masked_loss(logits, targets, mask):
    per = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                          reduction="none").view(targets.shape)
    return (per * mask).sum() / mask.sum().clamp_min(1)


def train(model, steps: int, seq_len: int, lr: float = 2e-3, batch: int = 24,
          mode: str = "none", scale: float = 1.0, seed: int = SEED):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x, m = make_batch(batch, seq_len, gen)
        x, m = x.to(DEVICE), m.to(DEVICE)
        loss = masked_loss(model(x[:, :-1], mode, scale), x[:, 1:], m[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


TRAIN_STEPS = 600 if SMOKE else 1500
model = RopeLM().to(DEVICE)
final = train(model, TRAIN_STEPS, TRAIN_LEN)
chance = math.log(N_VALS)
print(f"\ntrained {TRAIN_STEPS} steps at length {TRAIN_LEN}")
print(f"  final query loss: {final:.4f}   (chance = {chance:.3f})")
print("\nThe model has learned the lookup, not a shortcut -- values are random")
print("per sequence, so the only way to answer is to attend to the definition.")

# %% [markdown]
# ## 4. Four times the context
#
# Now evaluate the same weights at $4\times$ the training length, binned by
# position, under each method.

# %%
eval_gen = torch.Generator().manual_seed(SEED + 99)
EVAL_X, EVAL_M = make_batch(16, EVAL_LEN, eval_gen)
EVAL_X, EVAL_M = EVAL_X.to(DEVICE), EVAL_M.to(DEVICE)


@torch.no_grad()
def binned_loss(model, mode: str, scale: float = 1.0, temp: float = 1.0,
                bins: int = 8) -> list:
    logits = model(EVAL_X[:, :-1], mode, scale, temp)
    targets, mask = EVAL_X[:, 1:], EVAL_M[:, 1:]
    per = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                          reduction="none").view(targets.shape)
    width = per.shape[1] // bins
    out = []
    for i in range(bins):
        sl = slice(i * width, (i + 1) * width)
        mk = mask[:, sl]
        out.append(float((per[:, sl] * mk).sum() / mk.sum()) if mk.sum() > 0
                   else float("nan"))
    return out


BINS = 8
width = (EVAL_LEN - 1) // BINS
METHODS = [("none", "none", 1.0), ("PI", "pi", SCALE),
           ("NTK-aware", "ntk", SCALE), ("YaRN", "yarn", SCALE)]

print(f"\nQuery loss by position, evaluated at {EVAL_LEN} "
      f"(trained at {TRAIN_LEN}, scale {SCALE:.0f}x)")
print("-" * 74)
print(f"{'method':<11} " + " ".join(f"{i*width:>7}" for i in range(BINS)))
results = {}
for label, mode, sc in METHODS:
    row = binned_loss(model, mode, sc)
    results[label] = row
    print(f"{label:<11} " + " ".join(
        ("    n/a" if v != v else f"{v:>7.2f}") for v in row))
print(f"{'(chance)':<11} " + " ".join(f"{chance:>7.2f}" for _ in range(BINS)))

def finite(row):
    return [v for v in row if v == v]

print("\nStart with the PI row, because it is the one that should feel wrong.")
pi_row = finite(results["PI"])
none_row = finite(results["none"])
print(f"  PI at the earliest positions:   {pi_row[0]:.2f}")
print(f"  no modification, same positions:{none_row[0]:.2f}")
print("\nPI made the model worse at positions it already handled perfectly.")
if pi_row[0] >= chance:
    print(f"Not a subtle regression: at the earliest positions it is now WORSE")
    print(f"than chance ({chance:.2f}) -- the rescaled positions are actively")
    print("misleading it, on a lookup that cost it nothing before.")
else:
    print(f"At the earliest positions it has gone from essentially free to")
    print(f"{pi_row[0]:.2f}, against a chance level of {chance:.2f}.")
print(f"\nThe reason is in section 1's table. PI divides every position by "
      f"{SCALE:.0f},")
print(f"so the fastest dimension -- the one that distinguished token m from")
print(f"token m+1 -- now has to resolve differences {SCALE:.0f}x finer than")
print("anything it saw in training. Interpolation buys long-range validity by")
print("spending short-range resolution, and short range is where almost all of")
print("the model's work happens.")
print("\nThis is why every published PI result is paired with fine-tuning, and")
print("why NTK-aware scaling was proposed as the training-free alternative.")

# %% [markdown]
# ## 5. What NTK and YaRN preserve

# %%
print("\nSame table, read as degradation against the model's own training regime")
print("-" * 74)
print(f"{'method':<11} {'mean loss':>11} {'first bin':>11} {'last bin':>11} "
      f"{'worst bin':>11}")
for label in ("none", "PI", "NTK-aware", "YaRN"):
    row = finite(results[label])
    print(f"{label:<11} {sum(row)/len(row):>11.3f} {row[0]:>11.3f} "
          f"{row[-1]:>11.3f} {max(row):>11.3f}")

print("\nNTK-aware and YaRN both hold the short positions, because both leave")
print("the fast dimensions alone and stretch only the slow ones -- the")
print("dimensions that were extrapolating anyway.")
print("\nAnd note how well 'none' does here. A 2-layer model on a clean")
print("synthetic lookup extrapolates further than a real language model on")
print("natural text, so treat the size of the no-modification degradation as a")
print("lower bound rather than a measurement of anything. What transfers is the")
print("ORDERING and the mechanism, not the magnitudes.")

# %% [markdown]
# ## 6. Fine-tuning at the extended length
#
# Exercise 9.8 item 5. Each method starts from the same weights, then gets a
# short fine-tune at the extended length under its own position scheme.

# %%
FT_STEPS = 40 if SMOKE else 120
FT_BATCH = 4

print(f"\nFine-tuning {FT_STEPS} steps at length {EVAL_LEN}")
print("-" * 74)
print(f"{'method':<11} {'before':>10} {'after':>10} {'improvement':>13}")
import copy
for label, mode, sc in METHODS:
    before = sum(finite(results[label])) / len(finite(results[label]))
    ft = copy.deepcopy(model)
    train(ft, FT_STEPS, EVAL_LEN, lr=5e-4, batch=FT_BATCH, mode=mode, scale=sc,
          seed=SEED + 7)
    after_row = finite(binned_loss(ft, mode, sc))
    after = sum(after_row) / len(after_row)
    print(f"{label:<11} {before:>10.3f} {after:>10.3f} {before-after:>13.3f}")

print("\nThe method with the most to gain from fine-tuning is the one that")
print("broke the most, which is PI -- its positions are all in-distribution")
print("after rescaling, so a short fine-tune can re-teach the fast dimensions")
print("their new resolution. That is exactly the recipe the PI paper used.")
print("\nMethods that were already close to their un-extended behaviour have")
print("less to recover, and a fine-tune mostly just costs compute.")

# %% [markdown]
# ## 7. Things to try
#
# **1. Set the YaRN ramp bounds from (1, 32) to (1, 2).**
# *Common prediction:* a small change.
# *What happens:* YaRN collapses toward PI, because almost every dimension is
# now treated as "long wavelength" and interpolated. The ramp bounds *are* the
# method — YaRN is PI and NTK with a switch, and the switch position is the
# hyperparameter.
#
# **2. Evaluate at 16× instead of 4×.**
# *Common prediction:* everything degrades proportionally.
# *What happens:* the no-modification row collapses much faster than the
# others, and NTK starts to lose its advantage, because at large scale even the
# fast dimensions have to be stretched. Training-free extension has a range,
# and past it fine-tuning stops being optional.
#
# **3. Set `BASE = 500000` before training and repeat the whole lab.**
# *Common prediction:* nothing much; it is just a constant.
# *What happens:* far more dimensions have wavelengths longer than the training
# window (check section 1's table), and the model extrapolates better without
# any modification at all. This is why modern models are pre-trained with a
# large RoPE base — it is context extension paid for in advance.
#
# **4. Add YaRN's attention temperature: pass `temp=0.9` in `binned_loss`.**
# *Common prediction:* it is a detail.
# *What happens:* it shifts the whole curve slightly, because longer contexts
# spread attention over more keys and the softmax gets flatter. YaRN's
# temperature term compensates for that, and it is a separate mechanism from
# the frequency ramp — worth being able to name independently.
#
# **5. Train at `TRAIN_LEN = 256` and extend 2× instead of 4×.**
# *Common prediction:* the same picture, smaller effects.
# *What happens:* PI's damage falls sharply, because the resolution it spends
# is proportional to the scale factor. Most of the difficulty in context
# extension is in the *ratio*, not the absolute length.

# %% [markdown]
# ## What to take away
#
# 1. **RoPE is relative, so extending context only breaks models that actually
#    use long range.** A model attending three tokens back does not care about
#    absolute position, which is why a local task shows nothing at all.
# 2. **The wavelength table tells you what will break.** Dimensions whose
#    wavelength fits inside the training window have seen every phase; the
#    slower ones have not, and those are the ones extrapolating.
# 3. **Position Interpolation trades short-range resolution for long-range
#    validity.** It damages positions that were already working — which is why
#    it is always paired with fine-tuning.
# 4. **NTK-aware scaling stretches slow dimensions and leaves fast ones alone**,
#    which is why it works without fine-tuning at moderate extension factors.
# 5. **YaRN makes that policy explicit** with a wavelength ramp, plus an
#    attention temperature for the flatter softmax that longer contexts
#    produce.
# 6. **The method that gains most from fine-tuning is the one that broke most.**
#    That is not a virtue of the method; it is a measure of the damage it did.
