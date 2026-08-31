# %% [markdown]
# # Lab 15 — GRPO: group-relative advantage, and where the gradient goes
#
# Companion to
# [Chapter 15](../book/part-3-post-training/15-grpo-reasoning.md).
#
# GRPO removes the value network. For each prompt it samples a **group** of $G$
# completions and uses the group itself as the baseline:
#
# $$A_i = \frac{R_i - \mathrm{mean}(R_{1..G})}{\mathrm{std}(R_{1..G})}$$
#
# That is the whole idea, and two consequences follow immediately that this lab
# is built to make visible.
#
# The first: **if every completion in a group gets the same reward, the
# advantage is exactly zero and that prompt contributes nothing.** Not a small
# gradient — none. A prompt your model always solves and a prompt it never
# solves are equally worthless.
#
# The second: the policy optimises the reward you wrote, including the parts
# you added for convenience.
#
# You will:
#
# 1. Implement GRPO and train a policy that must both answer correctly and
#    follow a format.
# 2. Track mean response length and find out which way it moves.
# 3. Add a per-token "thinking" bonus and watch length become a target.
# 4. Turn off the standard-deviation normalisation and measure which prompts
#    take over the gradient.
# 5. Sweep $G$ and measure the fraction of rollouts that produce no signal.
# 6. Filter prompts by pass rate and measure the sample-efficiency gain.
#
# Real policy, real sampling, real rewards. Runs on CPU in about a minute.
#
# **Predict before you run.** Over training, does mean response length go up or
# down? R1's famously went up. Commit before section 3.

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

# %% [markdown]
# ## 1. A task with a format and an answer
#
# The prompt is a digit. A completion is $T$ tokens, and the policy chooses how
# to spend them: `THINK` tokens, then an `ANS` marker, then its answer.
#
# Two rewards, exactly as an RLVR recipe would have:
#
# - **accuracy** (1.0): the token after `ANS` is the right answer.
# - **format** (0.2): exactly one `ANS`, and something after it.
#
# Nothing rewards thinking. Remember that when you read section 3.

# %%
THINK, ANS = 0, 1
D0 = 2
N_DIGITS = 10
VOCAB = D0 + N_DIGITS
T_GEN = 6
N_PROMPTS = 10


def target_of(d: int) -> int:
    return (d + 3) % N_DIGITS


class Policy(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(T_GEN + 1, d_model)
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
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(t, device=x.device))
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        for ln1, attn, ln2, mlp in self.blocks:
            a = ln1(h)
            h = h + attn(a, a, a, attn_mask=mask, need_weights=False)[0]
            h = h + mlp(ln2(h))
        return self.head(self.norm(h))


@torch.no_grad()
def rollout(policy: Policy, prompts: torch.Tensor) -> torch.Tensor:
    seq = (D0 + prompts).unsqueeze(1)
    for _ in range(T_GEN):
        logp = torch.log_softmax(policy(seq)[:, -1], dim=-1)
        action = torch.multinomial(logp.exp(), 1)
        seq = torch.cat([seq, action], dim=1)
    return seq


def score(seq: torch.Tensor, format_w: float = 0.2,
          think_bonus: float = 0.0) -> dict:
    """Accuracy, format, an optional per-THINK-token bonus, and length."""
    batch = seq.shape[0]
    prompt = (seq[:, 0] - D0).tolist()
    body = seq[:, 1:].tolist()
    acc = torch.zeros(batch)
    fmt = torch.zeros(batch)
    length = torch.zeros(batch)
    n_think = torch.zeros(batch)
    for b in range(batch):
        toks = body[b]
        marks = [i for i, t in enumerate(toks) if t == ANS]
        n_think[b] = sum(1 for t in toks if t == THINK)
        if len(marks) == 1 and marks[0] < T_GEN - 1:
            fmt[b] = 1.0
            i = marks[0]
            length[b] = i                      # tokens spent before answering
            nxt = toks[i + 1]
            if nxt >= D0 and (nxt - D0) == target_of(prompt[b]):
                acc[b] = 1.0
        else:
            length[b] = float(T_GEN)           # never answered
    reward = acc + format_w * fmt + think_bonus * n_think
    return {"reward": reward, "acc": acc, "fmt": fmt, "length": length,
            "n_think": n_think}


# %% [markdown]
# ## 2. GRPO
#
# No value network, no clipping, no GAE. Sample $G$ per prompt, standardise
# within the group, and use that as the advantage.

# %%
def train_grpo(steps: int, G: int = 8, n_prompts: int = 32, lr: float = 1e-3,
               format_w: float = 0.2, think_bonus: float = 0.0,
               std_norm: bool = True, filter_pass_rate: bool = False,
               seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    policy = Policy().to(DEVICE)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    history = []
    trace = []                      # (cumulative rollouts, accuracy) every step
    rollouts_used = 0
    for step in range(steps):
        n_sample = n_prompts * 2 if filter_pass_rate else n_prompts
        prompts = torch.randint(0, N_PROMPTS, (n_sample,), generator=gen)
        prompts = prompts.repeat_interleave(G).to(DEVICE)
        seq = rollout(policy, prompts)
        s = score(seq, format_w, think_bonus)
        rollouts_used += seq.shape[0]

        grouped = s["reward"].view(n_sample, G)
        spread = grouped.std(dim=1)
        keep = torch.ones(n_sample, dtype=torch.bool)
        if filter_pass_rate:
            # Keep only groups that disagree with themselves. A group whose
            # members all score the same has an advantage of exactly zero.
            keep = spread > 1e-6
            if keep.sum() < 2:
                continue
            keep_idx = torch.nonzero(keep).squeeze(1)[:n_prompts]
            keep = torch.zeros(n_sample, dtype=torch.bool)
            keep[keep_idx] = True

        adv = grouped - grouped.mean(dim=1, keepdim=True)
        if std_norm:
            adv = adv / (grouped.std(dim=1, keepdim=True) + 1e-4)
        adv = adv[keep].flatten().detach().to(DEVICE)

        sub = seq.view(n_sample, G, -1)[keep].reshape(-1, seq.shape[1])
        logits = policy(sub[:, :-1])
        logp = torch.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, sub[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = -(token_lp.sum(1) * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        trace.append((rollouts_used, float(s["acc"].mean())))
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            history.append({
                "step": step,
                "reward": float(s["reward"].mean()),
                "acc": float(s["acc"].mean()),
                "fmt": float(s["fmt"].mean()),
                "length": float(s["length"].mean()),
                "zero_adv": float((spread < 1e-6).float().mean()),
                "rollouts": rollouts_used,
            })
    return {"history": history, "policy": policy, "rollouts": rollouts_used,
            "trace": trace}


def rollouts_to_reach(run: dict, target: float, window: int = 3) -> int:
    """Cumulative rollouts before accuracy first holds at `target`."""
    hits = 0
    for used, acc in run["trace"]:
        hits = hits + 1 if acc >= target else 0
        if hits >= window:
            return used
    return -1


STEPS = 120 if SMOKE else 250


def show(label: str, hist: list) -> None:
    print(f"\n{label}")
    print("-" * 74)
    print(f"{'step':>6} {'reward':>9} {'accuracy':>10} {'format':>9} "
          f"{'mean length':>13} {'zero-adv groups':>17}")
    for h in hist:
        print(f"{h['step']:>6} {h['reward']:>9.3f} {h['acc']:>10.3f} "
              f"{h['fmt']:>9.3f} {h['length']:>13.2f} {100*h['zero_adv']:>16.0f}%")


base = train_grpo(STEPS)
show("GRPO baseline (G=8, format bonus 0.2, no thinking bonus)", base["history"])

print("\nAccuracy and format both go to 1. Two other columns are the lesson.")

# %% [markdown]
# ## 3. Length is whatever you paid for
#
# Exercise 15.9 item 1 asks for the shape of the length curve. R1's response
# length famously grew over training, and it is easy to read that as something
# GRPO does. Look at the length column above before continuing.

# %%
print("\nMean response length, baseline run")
print("-" * 74)
for h in base["history"]:
    bar = "#" * int(round(h["length"] * 8))
    print(f"  step {h['step']:>4}  length {h['length']:>5.2f}  {bar}")

print("\nIt collapses. The policy learns to emit ANS immediately, because")
print("nothing in the reward pays for a THINK token and answering sooner is")
print("never punished. GRPO did not make responses longer -- it made them")
print("whatever the reward function paid for, and ours pays for brevity by")
print("omission.")
print("\nSo now add a small per-token bonus for thinking and run it again.")

think = train_grpo(STEPS, think_bonus=0.05)
show("GRPO with a 0.05 bonus per THINK token", think["history"])

print(f"\nfinal length  -- baseline: {base['history'][-1]['length']:.2f}   "
      f"with think bonus: {think['history'][-1]['length']:.2f}")
print(f"final accuracy-- baseline: {base['history'][-1]['acc']:.3f}   "
      f"with think bonus: {think['history'][-1]['acc']:.3f}")

print("\nSame algorithm, same task, opposite length dynamics. A per-token bonus")
print("is unbounded in a way the accuracy reward is not -- there is always one")
print("more token to add -- so the policy takes the free reward.")
print("\nThe honest reading of R1's growing response length is therefore not")
print("'GRPO produces longer reasoning'. It is that on problems hard enough")
print("that longer reasoning raises the pass rate, length grows because it is")
print("INSTRUMENTAL to accuracy. Where it is not instrumental, as here, it")
print("collapses -- and if you pay for it directly you get length whether or")
print("not it helps. Chapter 15 section 15.5.")

# %% [markdown]
# ## 4. What the standard deviation is doing
#
# Exercise 15.9 item 3. Dividing by the group's standard deviation makes every
# prompt contribute a comparably-sized advantage. Remove it and the prompts
# with the widest reward spread speak loudest.
#
# The training outcome barely changes, so measure the gradient directly rather
# than reading the reward curve.

# %%
@torch.no_grad()
def advantage_concentration(policy: Policy, G: int = 8, n_prompts: int = 64,
                            std_norm: bool = True, seed: int = SEED) -> dict:
    gen = torch.Generator().manual_seed(seed)
    prompts = torch.randint(0, N_PROMPTS, (n_prompts,), generator=gen)
    seq = rollout(policy, prompts.repeat_interleave(G).to(DEVICE))
    grouped = score(seq)["reward"].view(n_prompts, G)
    adv = grouped - grouped.mean(dim=1, keepdim=True)
    if std_norm:
        adv = adv / (grouped.std(dim=1, keepdim=True) + 1e-4)
    mass = adv.abs().sum(dim=1)
    total = mass.sum().clamp_min(1e-9)
    share = (mass / total).sort(descending=True).values
    spread = grouped.std(dim=1)
    # Correlation between a group's reward spread and its share of the gradient.
    x, y = spread, mass
    xm, ym = x.mean(), y.mean()
    denom = (((x - xm) ** 2).sum().sqrt() * ((y - ym) ** 2).sum().sqrt()).clamp_min(1e-9)
    corr = float(((x - xm) * (y - ym)).sum() / denom)
    return {"top_10pct_share": float(share[:max(1, n_prompts // 10)].sum()),
            "corr_spread_vs_mass": corr}


probe = train_grpo(max(20, STEPS // 4))["policy"]
print("\nHow concentrated is the gradient across prompts?")
print("-" * 74)
print(f"{'advantage':<22} {'top 10% of prompts hold':>26} "
      f"{'corr(spread, mass)':>21}")
for norm in (True, False):
    c = advantage_concentration(probe, std_norm=norm)
    label = "standardised" if norm else "mean-centred only"
    print(f"{label:<22} {100*c['top_10pct_share']:>25.1f}% "
          f"{c['corr_spread_vs_mass']:>21.3f}")

print("\nWithout the division, a group's influence is proportional to how much")
print("its rewards disagree, so the gradient concentrates on prompts near a 50%")
print("pass rate and thins out over prompts the model nearly always or nearly")
print("never solves. Standardising flattens that: every group that disagrees at")
print("all gets the same total say, regardless of by how much.")
print("\nThe effect is real and it is not dramatic -- read the top-10% column")
print("as a tendency rather than a cliff. It compounds over thousands of steps,")
print("which is where it starts to matter.")
print("\nWhich you want is a real design question rather than a bug: the")
print("unnormalised version naturally focuses on the frontier of what the model")
print("can do, and the normalised version treats a prompt with a marginal")
print("disagreement as equal to one with a decisive one. Several later variants")
print("(Dr. GRPO among them) revisit exactly this term.")

# %% [markdown]
# ## 5. $G$, and the rollouts that produce nothing
#
# Exercise 15.9 item 5. A group whose members all get the same reward has zero
# advantage and contributes no gradient. Small $G$ makes that far more likely.

# %%
print("\nSweeping G (rollout budget per step held roughly constant)")
print("-" * 74)
print(f"{'G':>4} {'prompts':>9} {'zero-adv at start':>19} {'zero-adv at end':>17} "
      f"{'final accuracy':>16}")
GS = (2, 8) if SMOKE else (2, 4, 8, 16)
for g in GS:
    n_p = max(4, 256 // g)
    r = train_grpo(STEPS, G=g, n_prompts=n_p)
    h0, h1 = r["history"][0], r["history"][-1]
    print(f"{g:>4} {n_p:>9} {100*h0['zero_adv']:>18.0f}% "
          f"{100*h1['zero_adv']:>16.0f}% {h1['acc']:>16.3f}")

print("\nAt G=2 a group agrees with itself very often even at the start, so a")
print("large share of the generation budget is spent producing no gradient at")
print("all. Larger G makes disagreement more likely and the advantage estimate")
print("less noisy -- which is why GRPO recipes use G of 8 to 64 and why the")
print("method is generation-heavy.")
print("\nNow read the end column for every G. As the policy improves, groups")
print("stop disagreeing, and by the end of training almost every rollout is")
print("wasted. GRPO gets less sample-efficient exactly as it succeeds.")

# %% [markdown]
# ## 6. Filtering by pass rate
#
# Exercise 15.9 item 4. If groups that agree with themselves contribute
# nothing, do not train on them: sample a larger pool, keep the groups that
# disagree, and spend the gradient step on those.
#
# The right metric is **generation spent to reach a given accuracy**, not final
# accuracy — filtering costs extra rollouts up front, so comparing endpoints
# hides the whole question.

# %%
TARGET = 0.90
print(f"\nGeneration spent to reach and hold {TARGET:.0%} accuracy")
print("-" * 74)
print(f"{'configuration':<34} {'rollouts to target':>20} {'final acc':>12}")

configs = [
    ("G=8, all groups", dict(G=8, n_prompts=32)),
    ("G=8, filtered", dict(G=8, n_prompts=32, filter_pass_rate=True)),
    ("G=2, all groups", dict(G=2, n_prompts=128)),
    ("G=2, filtered", dict(G=2, n_prompts=128, filter_pass_rate=True)),
]
results = {}
for label, kw in configs:
    r = train_grpo(STEPS, **kw)
    n = rollouts_to_reach(r, TARGET)
    results[label] = (n, r["history"][-1]["acc"])
    shown = f"{n:,}" if n > 0 else "not reached"
    print(f"{label:<34} {shown:>20} {r['history'][-1]['acc']:>12.3f}")

print("\nFiltering loses. In both G settings, and at every accuracy target I")
print("tried (90%, 97%, 99%), it needs roughly 1.8x the generation to get to")
print("the same place. I expected the G=2 row to vindicate it, since 62% of")
print("those groups agree with themselves from the start, and it does not.")
print("\nThe arithmetic is unforgiving once you look at it. Generation is the")
print("expensive part, and this implementation generates twice as much and")
print("throws half away. Meanwhile a zero-advantage group in an unfiltered")
print("batch is not harmful -- its advantage is exactly zero, so it contributes")
print("no wrong direction. It only dilutes the mean, and Adam's per-parameter")
print("normalisation absorbs most of a constant scale factor anyway.")
print("\nSo 'skip the groups that agree' is intuitive, true as far as it goes,")
print("and not by itself a saving. Zero-advantage groups waste generation")
print("whether you filter them or not -- filtering does not un-spend it.")
print("\nWhat the real dynamic-sampling schemes (DAPO and relatives) do")
print("differently is the part this lab does not implement: they resample")
print("ADAPTIVELY until a batch is full of usable groups, rather than paying a")
print("fixed 2x oversample every step. That converts the cost from a constant")
print("tax into one you pay only when the batch is actually degenerate -- and")
print("it matters most in the regime this task never reaches, where a model is")
print("near-saturated on nearly every prompt and an unfiltered step would carry")
print("almost no signal at all.")
print("\nThe honest summary: the zero-advantage problem in section 5 is real,")
print("and the naive fix for it measured here is not a fix. That gap is why the")
print("published variants are more complicated than 'drop the flat groups'.")

# %% [markdown]
# ## 7. Things to try
#
# **1. Set `format_w = 1.0` so format pays as much as accuracy.**
# *Common prediction:* the model games the format and ignores the answer.
# *What happens:* it learns both. Format saturates almost immediately, and once
# it does there is no more reward available there, so every remaining gradient
# goes to accuracy. A bounded shaped reward cannot be farmed. Compare with the
# *per-token* bonus in section 3, which is unbounded and is farmed instantly —
# the distinction that matters is boundedness, not weight.
#
# **2. Set `think_bonus = 0.2` and watch accuracy.**
# *Common prediction:* longer responses, similar accuracy.
# *What happens:* length saturates at the maximum and accuracy suffers, because
# tokens spent on THINK are tokens not spent answering within the budget. A
# shaped reward that competes with the real objective for a fixed resource does
# not just add noise, it substitutes.
#
# **3. Make the task unsolvable — change `target_of` to return a random digit
# per call.**
# *Common prediction:* accuracy stays at chance, everything else is fine.
# *What happens:* format is still learned perfectly. Given a reward with a
# reachable component and an unreachable one, the policy takes the reachable
# one and looks like it is making progress. This is what reward hacking looks
# like from the dashboard.
#
# **4. Run with `G=1`.**
# *Common prediction:* it degenerates to REINFORCE without a baseline.
# *What happens:* every group has zero spread, so every advantage is zero and
# nothing trains at all. The group IS the baseline in GRPO — with one sample
# there is no comparison to make.
#
# **5. Make the filter adaptive instead of a fixed 2x oversample.**
# Keep sampling groups until you have collected `n_prompts` with non-zero
# spread, and count the rollouts you actually spent.
# *Common prediction:* the same result as section 6, since you still pay for
# the discarded groups.
# *What happens:* it changes the cost profile — you pay nothing extra while
# groups disagree and pay heavily only once they stop, instead of a flat 2x
# tax on every step. Implementing this is the most instructive thing in the
# lab, because it turns section 6's negative result into the design that the
# published variants actually use.

# %% [markdown]
# ## What to take away
#
# 1. **A group that agrees with itself contributes exactly zero.** Not a small
#    gradient — none. Prompts that are always solved and never solved are
#    equally useless.
# 2. **GRPO gets less sample-efficient as it succeeds**, because success means
#    groups stop disagreeing. By the end of training most rollouts produce no
#    signal at all. The obvious fix — oversample and drop the flat groups —
#    measured *worse* here at every accuracy target, because it doubles
#    generation to remove samples that were contributing zero rather than
#    something harmful. Real dynamic sampling resamples adaptively instead, and
#    that difference is the whole trick.
# 3. **Response length is a property of the reward, not of the algorithm.**
#    With nothing paying for thinking, length collapses. R1's length grew
#    because longer reasoning raised the pass rate on hard problems; pay for
#    tokens directly and you get tokens whether or not they help.
# 4. **Bounded shaped rewards saturate; unbounded ones get farmed.** A format
#    bonus weighted equal to accuracy is harmless. A per-token bonus a
#    twentieth of its size is not.
# 5. **The standard deviation in the advantage decides which prompts matter.**
#    Divide by it and every disagreeing group speaks equally; leave it out and
#    the model's frontier dominates the gradient. Both are defensible; only one
#    is what you meant.
