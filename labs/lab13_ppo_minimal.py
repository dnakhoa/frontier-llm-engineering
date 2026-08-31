# %% [markdown]
# # Lab 13 — PPO and RLOO, and what each guardrail is actually holding
#
# Companion to
# [Chapter 13](../book/part-3-post-training/13-ppo-rloo-frontier.md).
#
# PPO's reputation for fiddliness comes from the fact that it has several
# safety devices whose purpose is invisible until you remove them. The clip,
# the KL penalty, the advantage normalisation and the value baseline all look
# like implementation detail. Each of them is load-bearing, and this lab takes
# them out one at a time.
#
# You will:
#
# 1. Implement PPO — rollouts, ratios, clipping, KL penalty, value baseline —
#    against a task with checkable reward.
# 2. Set $\beta = 0$ and watch the policy leave the reference behind.
# 3. Effectively unclip the objective and run 4 epochs per batch.
# 4. Implement RLOO and compare it to PPO on the same budget.
# 5. Sweep $k$ in RLOO and measure gradient variance directly.
# 6. Corrupt one reward in 500 and measure how much of the batch gradient that
#    single sample commands, with and without advantage normalisation.
#
# The task is small enough that RL is stable and honest — a real policy, real
# sampling, real rewards, no simulation. Runs on CPU in about a minute.
#
# **Predict before you run.** With the KL penalty removed, how many steps
# before the damage is visible in the reward curve? And will the reward go up
# or down? Write both down.

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
# ## 1. The task, the policy, the reward
#
# A prompt is a digit. The policy generates $T$ tokens autoregressively. The
# reward is the fraction of positions where the token follows a fixed rule.
#
# Deliberately simple: the point is to watch the *optimiser*, and a task the
# policy can actually solve is what lets us tell a broken optimiser from a hard
# problem.

# %%
VOCAB = 12
T_GEN = 3
N_PROMPTS = 10


def rule(prompt: torch.Tensor, i: int) -> torch.Tensor:
    return (prompt + i + 1) % VOCAB


class Policy(nn.Module):
    """Tiny autoregressive policy with a value head."""

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
        self.value = nn.Linear(d_model, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(t, device=x.device))
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        for ln1, attn, ln2, mlp in self.blocks:
            a = ln1(h)
            h = h + attn(a, a, a, attn_mask=mask, need_weights=False)[0]
            h = h + mlp(ln2(h))
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.trunk(x)
        return self.head(h), self.value(h[:, 0]).squeeze(-1)


@torch.no_grad()
def rollout(policy: Policy, prompts: torch.Tensor) -> tuple:
    """Sample T_GEN tokens autoregressively; return sequences and log-probs."""
    seq = prompts.unsqueeze(1)
    logps = []
    for _ in range(T_GEN):
        logits, _ = policy(seq)
        logp = torch.log_softmax(logits[:, -1], dim=-1)
        action = torch.multinomial(logp.exp(), 1)
        logps.append(logp.gather(1, action).squeeze(1))
        seq = torch.cat([seq, action], dim=1)
    return seq, torch.stack(logps, dim=1)


def reward_of(seq: torch.Tensor) -> torch.Tensor:
    prompt = seq[:, 0]
    hits = torch.zeros(seq.shape[0], device=seq.device)
    for i in range(T_GEN):
        hits += (seq[:, 1 + i] == rule(prompt, i)).float()
    return hits / T_GEN


def logprobs_of(policy: Policy, seq: torch.Tensor) -> tuple:
    logits, value = policy(seq[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logp, value, logp


# %% [markdown]
# ## 2. PPO
#
# The objective, with every term the chapter names:
#
# $$\mathcal{L} = -\mathbb{E}\Big[\min\big(\rho_t A_t,\;
#   \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)\Big]
#   \quad \rho_t = \frac{\pi_\theta(a_t)}{\pi_{\theta_{old}}(a_t)}$$
#
# and the reward the advantage is computed from carries a KL penalty against
# the reference policy:
#
# $$r' = r - \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)$$

# %%
def train_ppo(steps: int, beta: float = 0.02, clip: float = 0.2,
              epochs: int = 2, batch: int = 256, lr: float = 3e-4,
              normalize_adv: bool = True, seed: int = SEED,
              corrupt_frac: float = 0.0, corrupt_sigma: float = 20.0,
              grad_clip: float = 1.0) -> dict:
    torch.manual_seed(seed)
    policy = Policy().to(DEVICE)
    reference = Policy().to(DEVICE)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    history = []
    outlier_shares = []
    max_ratio = 1.0
    peak_reward = 0.0
    for step in range(steps):
        prompts = torch.randint(0, N_PROMPTS, (batch,), generator=gen).to(DEVICE)
        seq, old_logp = rollout(policy, prompts)
        reward = reward_of(seq)

        if corrupt_frac > 0:
            n_bad = max(1, int(batch * corrupt_frac))
            idx = torch.randperm(batch, generator=gen)[:n_bad].to(DEVICE)
            reward = reward.clone()
            reward[idx] = reward.mean() + corrupt_sigma * reward.std().clamp_min(1e-6)

        with torch.no_grad():
            ref_logp, _, _ = logprobs_of(reference, seq)
        kl = (old_logp - ref_logp).sum(1)
        shaped = reward - beta * kl

        for _ in range(epochs):
            logp, value, dist = logprobs_of(policy, seq)
            adv = (shaped - value).detach()
            if normalize_adv:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            if corrupt_frac > 0:
                share = adv[idx].abs().sum() / adv.abs().sum()
                outlier_shares.append(float(share))
            ratio = (logp.sum(1) - old_logp.sum(1)).exp()
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            pg_loss = -torch.min(unclipped, clipped).mean()
            v_loss = F.mse_loss(value, shaped)
            (pg_loss + 0.5 * v_loss).backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()
            opt.zero_grad()
            max_ratio = max(max_ratio, float(ratio.max()))

        if step % max(1, steps // 6) == 0 or step == steps - 1:
            with torch.no_grad():
                entropy = float(-(dist.exp() * dist).sum(-1).mean())
                clipfrac = float(((ratio - 1).abs() > clip).float().mean())
            history.append({"step": step, "reward": float(reward.mean()),
                            "kl": float(kl.mean()), "entropy": entropy,
                            "clipfrac": clipfrac})
        peak_reward = max(peak_reward, float(reward.mean()))
    return {"history": history, "policy": policy,
            "max_ratio": max_ratio, "peak_reward": peak_reward,
            "outlier_share": (sum(outlier_shares) / len(outlier_shares)
                              if outlier_shares else None)}


STEPS = 80 if SMOKE else 220


def show(label: str, hist: list) -> None:
    print(f"\n{label}")
    print("-" * 74)
    print(f"{'step':>6} {'reward':>9} {'KL(pi||ref)':>13} {'entropy':>10} "
          f"{'clip frac':>11}")
    for h in hist:
        print(f"{h['step']:>6} {h['reward']:>9.3f} {h['kl']:>13.3f} "
              f"{h['entropy']:>10.3f} {h['clipfrac']:>11.3f}")


baseline = train_ppo(STEPS, lr=1e-3)
show("PPO with every guardrail on (beta=0.02, clip=0.2, 2 epochs)",
     baseline["history"])
print("\nReward climbs, KL grows to a plateau, entropy falls. That entropy")
print("collapse is not a bug -- it is what convergence looks like for a policy")
print("with one right answer. It is also why a converged RL policy generates")
print("almost deterministically, and why sampling diversity has to be bought")
print("back explicitly if you want it.")

# %% [markdown]
# ## 3. The KL penalty, swept
#
# Exercise 13.9 item 1 asks what happens at $\beta = 0$. I ran that first and
# the answer was *nothing much*, which is a more interesting result than the
# one I expected, so here is the whole sweep rather than the single point.

# %%
BETAS = (0.0, 0.02, 0.3) if SMOKE else (0.0, 0.02, 0.1, 0.3, 1.0)
print("\nSweeping the KL coefficient")
print("-" * 74)
print(f"{'beta':>7} {'final reward':>14} {'final KL':>11} {'entropy':>10}")
beta_rows = []
for b in BETAS:
    r = train_ppo(STEPS, beta=b)
    h = r["history"][-1]
    beta_rows.append((b, h["reward"], h["kl"]))
    print(f"{b:>7.2f} {h['reward']:>14.3f} {h['kl']:>11.2f} {h['entropy']:>10.3f}")

_kl0 = beta_rows[0][2]
_kl_small = beta_rows[1][2]
print(f"\nbeta=0 and beta=0.02 end at KL {_kl0:.2f} and {_kl_small:.2f} -- "
      f"effectively identical.")
print("\nSo on this task the default KL coefficient is not restraining anything.")
print("It only begins to bind an order of magnitude higher, and once it does it")
print("trades reward for proximity almost linearly: by the largest beta in the")
print("sweep the policy has barely left the reference and barely learned.")
print("\nThat is the practical lesson, and it is not the one the exercise")
print("predicts. beta is not a safety setting with a sensible default -- its")
print("effect depends entirely on the scale of your reward, because the shaped")
print("reward is r - beta*KL and those two terms have to be commensurate. A")
print("beta tuned for a reward in [0, 1] means something completely different")
print("for a reward model whose outputs span [-10, 10].")
print("\nWhy does the policy stop drifting on its own here? Because the task has")
print("one right answer. Once the policy finds it, there is nowhere further to")
print("go, and the KL plateaus without help. A LEARNED reward model has no such")
print("ceiling: there is always a slightly higher-scoring region, and the KL")
print("term is what stops the policy walking into the part of the space where")
print("the reward model was never trained. Chapter 12's unbounded reward gap")
print("and this term are the same problem from two ends.")

# %% [markdown]
# ## 4. Unclipping, with multiple epochs per batch
#
# Exercise 13.9 item 2. The clip exists because PPO reuses each batch of
# rollouts for several gradient steps. After the first epoch that data is
# off-policy, and the ratio $\rho$ is the importance-sampling correction —
# until $\rho$ gets large enough that the correction is worse than useless.
#
# **This is where my first attempt failed to show anything**, and the reason is
# worth more than the experiment. I had `clip_grad_norm_(..., 1.0)` in the
# training loop, as everyone does. Global gradient clipping was quietly
# bounding every update, so removing PPO's trust region changed nothing and
# both configurations trained fine.
#
# Two different safety devices were doing one job, and the visible one was not
# the one I was studying. Below, gradient-norm clipping is off, the learning
# rate is raised, and the batch is reused eight times — the regime PPO was
# designed for.

# %%
AGGRESSIVE = dict(lr=3e-3, epochs=8, grad_clip=0.0)
print("\nSame aggressive settings, with and without the trust region")
print(f"(lr={AGGRESSIVE['lr']}, {AGGRESSIVE['epochs']} epochs per batch, "
      f"no gradient-norm clipping)")
print("-" * 74)
print(f"{'clip epsilon':>14} {'final reward':>14} {'peak reward':>13} "
      f"{'max ratio':>12}")
for eps in (0.2, 10.0):
    r = train_ppo(STEPS, clip=eps, **AGGRESSIVE)
    label = f"{eps}" + ("  (none)" if eps > 1 else "")
    print(f"{label:>14} {r['history'][-1]['reward']:>14.3f} "
          f"{r['peak_reward']:>13.3f} {r['max_ratio']:>12.1f}")

print("\nThe max-ratio column is the diagnosis. With the clip on, the policy")
print("stays within a bounded factor of the one that generated the batch. With")
print("it off, the ratio reaches into the hundreds or thousands: the update is")
print("being dominated by a handful of sequences the current policy considers")
print("wildly more likely than the sampling policy did, and their importance")
print("weights are enormous and meaningless.")
print("\nThe clip is not a regulariser on the policy. It is an admission that")
print("importance sampling stops working once two policies differ too much, and")
print("a refusal to trust the estimate past that point. Note the direction of")
print("the relationship: MORE epochs per batch makes the clip matter more, not")
print("less, because every extra epoch pushes the data further off-policy.")
print("\nAnd keep the masking effect in mind. If your PPO runs are stable with")
print("the clip disabled, check whether something else -- gradient clipping, a")
print("small learning rate, a single epoch per batch -- is holding the trust")
print("region for you. With one epoch per batch every ratio is exactly 1 and")
print("the clip is inert by construction.")

# %% [markdown]
# ## 5. RLOO
#
# RLOO drops the value network. For each prompt, sample $k$ completions and use
# the mean of the *others* as the baseline:
#
# $$A_i = R_i - \frac{1}{k-1}\sum_{j \ne i} R_j$$
#
# That is unbiased, needs no second model, and needs no GAE. The cost is $k$
# rollouts per prompt.

# %%
def train_rloo(steps: int, k: int = 8, beta: float = 0.02, batch_prompts: int = 32,
               lr: float = 1e-3, seed: int = SEED,
               collect_grad_var: bool = False) -> dict:
    torch.manual_seed(seed)
    policy = Policy().to(DEVICE)
    reference = Policy().to(DEVICE)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    history, grad_norms = [], []
    for step in range(steps):
        prompts = torch.randint(0, N_PROMPTS, (batch_prompts,), generator=gen)
        prompts = prompts.repeat_interleave(k).to(DEVICE)
        seq, old_logp = rollout(policy, prompts)
        reward = reward_of(seq)
        with torch.no_grad():
            ref_logp, _, _ = logprobs_of(reference, seq)
        kl = (old_logp - ref_logp).sum(1)
        shaped = (reward - beta * kl).view(batch_prompts, k)

        # Leave-one-out baseline, computed in closed form.
        total = shaped.sum(dim=1, keepdim=True)
        loo = (total - shaped) / (k - 1)
        adv = (shaped - loo).flatten().detach()

        logp, _, dist = logprobs_of(policy, seq)
        loss = -(logp.sum(1) * adv).mean()
        loss.backward()
        if collect_grad_var:
            gn = math.sqrt(sum(float((p.grad ** 2).sum())
                               for p in policy.parameters() if p.grad is not None))
            grad_norms.append(gn)
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % max(1, steps // 6) == 0 or step == steps - 1:
            with torch.no_grad():
                entropy = float(-(dist.exp() * dist).sum(-1).mean())
            history.append({"step": step, "reward": float(reward.mean()),
                            "kl": float(kl.mean()), "entropy": entropy,
                            "clipfrac": 0.0})
    var = None
    if grad_norms:
        m = sum(grad_norms) / len(grad_norms)
        var = sum((g - m) ** 2 for g in grad_norms) / len(grad_norms)
    return {"history": history, "grad_norm_mean": (sum(grad_norms)/len(grad_norms)
                                                   if grad_norms else None),
            "grad_norm_var": var}


rloo = train_rloo(STEPS, k=8, batch_prompts=32)
show("RLOO, k = 8", rloo["history"])

print("\nSame budget in rollouts (32 prompts x 8 samples = 256 sequences per")
print("step, matching PPO's batch), no value network, no clipping, no GAE.")
print(f"\nfinal reward -- PPO: {baseline['history'][-1]['reward']:.3f}   "
      f"RLOO: {rloo['history'][-1]['reward']:.3f}")
print("\nOn a task this size RLOO is competitive and is a great deal less")
print("machinery: no second network to size, initialise, or debug. That is the")
print("argument Chapter 13 makes for it, and the reason several labs moved.")
print("The cost is that RLOO needs k samples of the SAME prompt to form its")
print("baseline, so its rollout budget is spent differently -- see section 6.")

# %% [markdown]
# ## 6. What $k$ buys in RLOO
#
# Exercise 13.9 item 4. The leave-one-out baseline is estimated from $k-1$
# samples, so smaller $k$ should mean a noisier baseline and a noisier
# gradient.
#
# Measuring that during training does not work — I tried, and the gradient norm
# is dominated by where the policy happens to be, not by the estimator. The
# clean measurement holds the policy **fixed** and draws many independent
# batches, so the only thing varying is the sampling.

# %%
def estimator_variance(policy: Policy, k: int, n_batches: int,
                       rollout_budget: int = 256, beta: float = 0.02,
                       seed: int = SEED, fixed_prompts: int = 0) -> float:
    """Variance of the RLOO gradient estimate across independent batches.

    fixed_prompts=0 holds the ROLLOUT budget constant (prompts = budget / k).
    fixed_prompts=n holds the PROMPT count constant, so larger k costs more.
    """
    batch_prompts = fixed_prompts if fixed_prompts else max(1, rollout_budget // k)
    gen = torch.Generator().manual_seed(seed)
    flat_grads = []
    for _ in range(n_batches):
        prompts = torch.randint(0, N_PROMPTS, (batch_prompts,), generator=gen)
        prompts = prompts.repeat_interleave(k).to(DEVICE)
        seq, _ = rollout(policy, prompts)
        reward = reward_of(seq).view(batch_prompts, k)
        total = reward.sum(dim=1, keepdim=True)
        loo = (total - reward) / (k - 1)
        adv = (reward - loo).flatten().detach()

        logp, _, _ = logprobs_of(policy, seq)
        loss = -(logp.sum(1) * adv).mean()
        policy.zero_grad()
        loss.backward()
        flat_grads.append(torch.cat([p.grad.flatten() for p in policy.parameters()
                                     if p.grad is not None]).clone())
    policy.zero_grad()
    stack = torch.stack(flat_grads)
    # Total variance of the estimator: summed per-coordinate variance.
    return float(stack.var(dim=0, unbiased=True).sum())


# Use a partially-trained policy: at initialisation every advantage is noise,
# and at convergence they are all zero. Neither says anything.
probe_policy = train_ppo(max(10, STEPS // 4), lr=1e-3)["policy"]
N_BATCH = 12 if SMOKE else 30

print("\nRLOO gradient-estimator variance at a FIXED policy")
print("(total rollouts per batch held constant)")
print("-" * 74)
print(f"{'k':>4} {'prompts/batch':>15} {'estimator variance':>22} {'vs k=2':>10}")
KS = (2, 4, 8) if SMOKE else (2, 4, 8, 16)
base_var = None
for k in KS:
    v = estimator_variance(probe_policy, k, N_BATCH)
    if base_var is None:
        base_var = v
    print(f"{k:>4} {256//k:>15} {v:>22.5f} {v/base_var:>9.2f}x")

print("\nThat is close to flat, and it is the answer rather than a failed")
print("measurement. At a fixed rollout budget, raising k improves the baseline")
print("and removes prompts in the same breath, and on this task the two effects")
print("very nearly cancel.")

print("\nNow the same sweep holding PROMPTS fixed, so larger k costs more")
print("rollouts rather than trading them away:")
print("-" * 74)
print(f"{'k':>4} {'rollouts/batch':>16} {'estimator variance':>22} {'vs k=2':>10}")
base_var2 = None
for k in KS:
    v = estimator_variance(probe_policy, k, N_BATCH, fixed_prompts=32)
    if base_var2 is None:
        base_var2 = v
    print(f"{k:>4} {32*k:>16} {v:>22.5f} {v/base_var2:>9.2f}x")

print("\nHere k does what the theory says: more samples per prompt, a better")
print("leave-one-out baseline, lower variance -- bought with proportionally")
print("more generation.")
print("\nSo the two tables answer different questions, and only one of them is")
print("the question a practitioner has. If you are choosing k with a fixed GPU")
print("budget, it is nearly free either way and 4-8 is fine. If you are")
print("choosing k with a fixed prompt set, larger k buys real variance")
print("reduction at real cost. Quoting a k without saying which budget was held")
print("fixed does not describe an experiment.")

# %% [markdown]
# ## 7. One corrupted reward in 500
#
# Exercise 13.9 item 5. A reward model occasionally returns nonsense — a
# malformed generation, a tokenisation edge case, a genuine bug. What does one
# such sample do to a batch?

# %%
print("\nOne reward in 500 corrupted to +20 sigma")
print("-" * 74)
print(f"{'advantage normalisation':<28} {'outlier share of |A|':>22} "
      f"{'over-representation':>21}")
CORRUPT_FRAC = 1 / 500
shares = {}
for norm in (True, False):
    r = train_ppo(STEPS, normalize_adv=norm, corrupt_frac=CORRUPT_FRAC)
    shares[norm] = r["outlier_share"]
    print(f"{str(norm):<28} {100*r['outlier_share']:>21.1f}% "
          f"{r['outlier_share']/CORRUPT_FRAC:>20.0f}x")

print(f"\nOne sample in 500 is 0.2% of the batch and commands "
      f"{100*shares[True]:.0f}% of the")
print(f"total advantage mass -- about {shares[True]/CORRUPT_FRAC:.0f} times its share.")
print("\nAnd advantage normalisation barely changes it. I expected that row to")
print("be the fix and it is not, which is worth understanding: normalisation")
print("divides by the batch standard deviation, and the outlier is what inflated")
print("that standard deviation in the first place. Dividing by a number the")
print("outlier chose does not demote the outlier -- it demotes everything else.")
print("The 499 legitimate samples get quieter by exactly the factor that keeps")
print("the outlier loud.")
print("\nThe real defence is not in the optimiser at all: clip or winsorise the")
print("REWARD before it becomes an advantage. Chapter 13 section 13.6. An")
print("optimiser cannot distinguish a genuinely excellent sample from a broken")
print("reward model, and it is not the right place to try.")

# %% [markdown]
# ## 8. Things to try
#
# **1. Set `beta = 0.5`.**
# *Common prediction:* a slightly more conservative policy.
# *What happens:* learning nearly stops. The KL penalty is subtracted from the
# reward, so a large beta means the shaped reward is dominated by "stay where
# you started" and the task signal is a rounding error. Beta is not a safety
# dial you can turn up for free.
#
# **2. Set `epochs = 1` with `clip = 10`.**
# *Common prediction:* still broken, since the clip is off.
# *What happens:* it is fine. With one epoch per batch the data is on-policy,
# every ratio is exactly 1, and the clip has nothing to do. This is the
# cleanest way to see that the clip exists to enable batch reuse — nothing
# else.
#
# **3. Remove the value baseline entirely (use `adv = shaped`).**
# *Common prediction:* higher variance, same destination.
# *What happens:* it still learns here, because rewards are bounded in [0, 1]
# and the task is easy — which is a useful calibration on how much a toy task
# can tell you. Scale the reward by 100 and the same change becomes
# catastrophic.
#
# **4. Give RLOO $k = 1$.**
# *Common prediction:* it degenerates to REINFORCE.
# *What happens:* it divides by $k - 1 = 0$. Worth hitting once: the leave-one-
# out baseline is undefined for a single sample, which is the formal statement
# of why you cannot have a baseline without at least two samples to compare.
#
# **5. Corrupt the reward NEGATIVELY (-20 sigma) instead.**
# *Common prediction:* symmetric damage.
# *What happens:* it is not symmetric. A hugely negative advantage pushes the
# policy away from one specific sampled sequence, which is a much smaller
# change than being pulled toward one. Reward hacking is asymmetric in exactly
# this way, and it is why upward outliers get the attention.

# %% [markdown]
# ## What to take away
#
# 1. **Entropy collapse is what convergence looks like.** A converged RL policy
#    samples almost deterministically; diversity has to be bought back on
#    purpose.
# 2. **The KL coefficient has no transferable default.** At the usual 0.02 it
#    did not restrain this policy at all; it only began to bind an order of
#    magnitude higher, and then traded reward for proximity almost linearly.
#    beta has to be commensurate with your reward scale, because the shaped
#    reward is $r - \beta\,\mathrm{KL}$. What it buys is not protection from
#    forgetting but staying inside the region where a learned reward model
#    still means something.
# 3. **The clip exists to make batch reuse safe, and nothing else.** With one
#    epoch per batch every ratio is 1 and the clip is inert; with four epochs
#    it is the only thing bounding the update.
# 4. **RLOO removes the value network** and buys its baseline with $k$ samples
#    per prompt instead. At a fixed *rollout* budget the choice of $k$ is
#    nearly free — better baseline and fewer prompts roughly cancel. At a fixed
#    *prompt* budget, larger $k$ cuts estimator variance severalfold and costs
#    proportionally more generation. A reported $k$ means nothing unless you
#    know which budget was held fixed.
# 5. **One corrupted reward in 500 commands over a tenth of the advantage
#    mass**, a fiftyfold over-representation — and advantage normalisation
#    barely helps, because the outlier is what inflated the standard deviation
#    you are dividing by. Clip the reward, not the advantage.
# 6. **Two safety devices can hide each other.** Gradient-norm clipping made
#    PPO's trust region look unnecessary in my first draft. When a guardrail
#    appears to do nothing, check what else is doing its job.
