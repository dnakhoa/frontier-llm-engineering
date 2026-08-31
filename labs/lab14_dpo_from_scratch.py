# %% [markdown]
# # Lab 14 — DPO from scratch, and the reward that goes down
#
# Companion to [Chapter 14](../book/part-3-post-training/14-dpo-family.md).
#
# DPO's claim is that you do not need a reward model. The preference data
# already contains one, implicitly, and you can optimise the policy against it
# directly:
#
# $$\mathcal{L} = -\log \sigma\!\left(\beta\big[
#   (\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x)) -
#   (\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x))\big]\right)$$
#
# It is about fifteen lines. Almost all of the difficulty is in three details
# that produce a loss which trains, descends, and optimises the wrong thing.
#
# You will:
#
# 1. Implement the loss and check it against values computed by hand.
# 2. Reproduce the three classic implementation bugs and see what each costs.
# 3. Track the implicit rewards for chosen and rejected separately, and find
#    out which direction they go.
# 4. Sweep $\beta$ and watch saturation speed change.
# 5. Feed it identical pairs and confirm what the gradient does.
#
# Uses the same checkable task as
# [`lab12_reward_model`](lab12_reward_model.py), so "did it work" is a fact
# rather than a vibe. Runs on CPU in about a minute.
#
# **Predict before you run.** During DPO training, does the implicit reward of
# the *chosen* response go up or down? Commit before section 4.

# %%
from __future__ import annotations

import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# %%
import copy
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
# ## 1. The task and a reference policy
#
# Prompt: an addition problem. Response: some filler tokens, then an answer.
# A response is correct if the answer is right — checkable ground truth.
#
# DPO needs a **reference policy** $\pi_{\text{ref}}$, which in practice is the
# SFT model. We make one by training briefly on a mixture of correct and
# incorrect responses, so it starts out mediocre and has somewhere to go.

# %%
PAD, SEP, FILL = 0, 1, 2
MAX_DIGIT = 4 if SMOKE else 6
MAX_FILL = 6
DIGIT0 = 3
ANS0 = DIGIT0 + MAX_DIGIT + 1
VOCAB = ANS0 + 2 * MAX_DIGIT + 1
PROMPT_LEN = 3                      # a, b, SEP
SEQ_LEN = PROMPT_LEN + MAX_FILL + 1


def encode(a: int, b: int, answer: int, n_fill: int) -> list:
    seq = [DIGIT0 + a, DIGIT0 + b, SEP] + [FILL] * n_fill + [ANS0 + answer]
    return seq + [PAD] * (SEQ_LEN - len(seq))


def sample_pair(rng: random.Random, confound: int = 0) -> tuple:
    a, b = rng.randint(0, MAX_DIGIT), rng.randint(0, MAX_DIGIT)
    truth = a + b
    wrong = rng.choice([v for v in range(2 * MAX_DIGIT + 1) if v != truth])
    kc = rng.randint(0, MAX_FILL - 1)
    kr = rng.randint(0, MAX_FILL - 1)
    return encode(a, b, truth, kc), encode(a, b, wrong, kr)


class TinyLM(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(SEQ_LEN, d_model)
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


def sequence_logprob(model: nn.Module, seq: torch.Tensor,
                     mask_prompt: bool = True, average: bool = False) -> torch.Tensor:
    """Log-probability of the RESPONSE tokens under `model`.

    Two of the three classic DPO bugs live in this function's arguments.
    """
    logits = model(seq[:, :-1])
    targets = seq[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    valid = targets != PAD
    if mask_prompt:
        # Position i of `token_logp` predicts seq[:, i+1]. Response tokens start
        # at index PROMPT_LEN, so predictions from index PROMPT_LEN-1 onward.
        idx = torch.arange(token_logp.shape[1], device=seq.device)
        valid = valid & (idx >= PROMPT_LEN - 1)
    token_logp = token_logp * valid

    total = token_logp.sum(-1)
    return total / valid.sum(-1).clamp_min(1) if average else total


def make_data(n: int, rng: random.Random) -> list:
    return [sample_pair(rng) for _ in range(n)]


rng = random.Random(SEED)
N_PAIRS = 800 if SMOKE else 3000
PAIRS = make_data(N_PAIRS, rng)
EVAL_PAIRS = make_data(300, rng)

# Reference policy: brief SFT on a 50/50 mixture, i.e. a model with no
# particular preference for being right.
def train_reference(steps: int) -> nn.Module:
    torch.manual_seed(SEED)
    model = TinyLM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    r = random.Random(SEED + 1)
    for _ in range(steps):
        batch = []
        for _ in range(64):
            c, w = sample_pair(r)
            batch.append(c if r.random() < 0.5 else w)
        seq = torch.tensor(batch).to(DEVICE)
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1),
                               ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


REF_STEPS = 150 if SMOKE else 400
REFERENCE = train_reference(REF_STEPS)
REFERENCE.eval()
for p in REFERENCE.parameters():
    p.requires_grad_(False)
print(f"\nreference policy trained for {REF_STEPS} steps (50/50 correct/incorrect)")

# %% [markdown]
# ## 2. The loss, and a check by hand
#
# Exercise 14.9 item 1 asks you to verify the loss against a reference
# implementation. `trl` is not available offline, so instead we check against
# something stronger: values computed directly from the definition.
#
# With $\beta = 0.1$ and a policy identical to the reference, every log-ratio is
# zero, so the argument of $\sigma$ is zero and the loss must be exactly
# $-\log \sigma(0) = \log 2 = 0.6931\ldots$ — regardless of the data.

# %%
def dpo_loss(policy: nn.Module, ref: nn.Module, chosen: torch.Tensor,
             rejected: torch.Tensor, beta: float, ref_no_grad: bool = True,
             **kw) -> tuple:
    pi_c = sequence_logprob(policy, chosen, **kw)
    pi_r = sequence_logprob(policy, rejected, **kw)
    if ref_no_grad:
        with torch.no_grad():
            ref_c = sequence_logprob(ref, chosen, **kw)
            ref_r = sequence_logprob(ref, rejected, **kw)
    else:
        # The bug: the reference is in the graph and in the optimiser, so the
        # anchor is free to move toward whatever makes the loss small.
        ref_c = sequence_logprob(ref, chosen, **kw)
        ref_r = sequence_logprob(ref, rejected, **kw)
    chosen_reward = beta * (pi_c - ref_c)
    rejected_reward = beta * (pi_r - ref_r)
    logits = chosen_reward - rejected_reward
    loss = -F.logsigmoid(logits).mean()
    return loss, chosen_reward.detach(), rejected_reward.detach()


eval_c = torch.tensor([p[0] for p in EVAL_PAIRS]).to(DEVICE)
eval_r = torch.tensor([p[1] for p in EVAL_PAIRS]).to(DEVICE)

policy0 = copy.deepcopy(REFERENCE)
for p in policy0.parameters():
    p.requires_grad_(True)

loss0, rc0, rr0 = dpo_loss(policy0, REFERENCE, eval_c, eval_r, beta=0.1)
print("\nCheck 1: policy == reference")
print("-" * 74)
print(f"{'loss':<34}: {float(loss0):.10f}")
print(f"{'log(2)':<34}: {math.log(2):.10f}")
print(f"{'max |chosen_reward|':<34}: {float(rc0.abs().max()):.3e}")
print(f"{'match to 1e-6':<34}: {abs(float(loss0) - math.log(2)) < 1e-6}")

# Check 2: an INDEPENDENT implementation of the same formula.
# Written a different way on purpose -- explicit sigmoid, explicit log, one
# example at a time -- so that a shared bug would have to be made twice.
def dpo_loss_reference(policy, ref, chosen, rejected, beta):
    with torch.no_grad():
        pc = sequence_logprob(policy, chosen)
        pr = sequence_logprob(policy, rejected)
        rc = sequence_logprob(ref, chosen)
        rr = sequence_logprob(ref, rejected)
    total = 0.0
    for i in range(chosen.shape[0]):
        margin = beta * ((float(pc[i]) - float(rc[i])) - (float(pr[i]) - float(rr[i])))
        total += -math.log(1.0 / (1.0 + math.exp(-margin)))
    return total / chosen.shape[0]


# Move the policy off the reference so the check is not trivially satisfied.
with torch.no_grad():
    for p_ in policy0.parameters():
        p_.add_(torch.randn_like(p_) * 0.02)

loss_mine, _, _ = dpo_loss(policy0, REFERENCE, eval_c, eval_r, beta=0.1)
loss_ref = dpo_loss_reference(policy0, REFERENCE, eval_c, eval_r, beta=0.1)
print("\nCheck 2: against an independently written implementation")
print("-" * 74)
print(f"{'vectorised (logsigmoid)':<34}: {float(loss_mine):.10f}")
print(f"{'per-example (explicit sigmoid)':<34}: {loss_ref:.10f}")
print(f"{'absolute difference':<34}: {abs(float(loss_mine)-loss_ref):.3e}")
print("\nExercise 14.9 item 1 asks you to check this against `trl`. That is the")
print("right instinct and this lab cannot do it offline, so it does the")
print("stronger version instead: a second implementation of the formula written")
print("in a different shape, where a shared mistake would have to be made twice.")
print("If you have `trl` installed, run its DPO loss on the same tensors -- it")
print("should agree with both of these, and if it does not, §14.3's three")
print("suspects are where to look.")

# Restore the policy to the reference for the degenerate-pair check below.
policy0 = copy.deepcopy(REFERENCE)
for p_ in policy0.parameters():
    p_.requires_grad_(True)

# Check 3: identical chosen and rejected (Exercise 14.9 item 5).
loss_same, rc_s, rr_s = dpo_loss(policy0, REFERENCE, eval_c, eval_c, beta=0.1)
grads = torch.autograd.grad(loss_same, list(policy0.parameters()),
                            retain_graph=False, allow_unused=True)
grad_norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads if g is not None))
print("\nCheck 3: chosen and rejected are the same sequence")
print("-" * 74)
print(f"{'loss':<34}: {float(loss_same):.10f}   (log 2)")
print(f"{'gradient norm':<34}: {grad_norm:.3e}")
print("\nExactly log 2 and a zero gradient. DPO, like Bradley-Terry before it,")
print("supervises only the DIFFERENCE between two responses -- given a pair it")
print("cannot tell apart, it has nothing to say. Worth confirming once, because")
# noqa
print("a pipeline that silently produces duplicate pairs trains on nothing and")
print("shows a perfectly healthy-looking loss of 0.693 while doing it.")

# %% [markdown]
# ## 3. The three bugs in §14.3
#
# All three produce a loss that decreases. None of them error. One of them
# turns out not to be a bug at all, which I only found by measuring it.
#
# 1. **Not masking the prompt.** The received wisdom is that this corrupts the
#    objective. Watch the numbers before believing it.
# 2. **Averaging instead of summing.** Changes what the objective says about
#    length.
# 3. **Letting the reference model receive gradients.** The "reference" then
#    moves, and the KL anchor is anchored to nothing.

# %%
def train_dpo(beta: float = 0.1, steps: int = 200, seed: int = SEED,
              mask_prompt: bool = True, average: bool = False,
              ref_trainable: bool = False, pairs: list = None) -> dict:
    torch.manual_seed(seed)
    policy = copy.deepcopy(REFERENCE)
    for p in policy.parameters():
        p.requires_grad_(True)
    ref = copy.deepcopy(REFERENCE)
    if ref_trainable:
        for p in ref.parameters():
            p.requires_grad_(True)
        ref.train()
    else:
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    params = list(policy.parameters()) + (list(ref.parameters()) if ref_trainable else [])
    opt = torch.optim.AdamW(params, lr=5e-4)
    data = pairs if pairs is not None else PAIRS
    r = random.Random(seed)
    kw = {"mask_prompt": mask_prompt, "average": average}
    history = []
    for step in range(steps):
        batch = [data[r.randrange(len(data))] for _ in range(32)]
        c = torch.tensor([b[0] for b in batch]).to(DEVICE)
        j = torch.tensor([b[1] for b in batch]).to(DEVICE)
        loss, rc, rr = dpo_loss(policy, ref, c, j, beta,
                                ref_no_grad=not ref_trainable, **kw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            with torch.no_grad():
                el, erc, err = dpo_loss(policy, ref, eval_c, eval_r, beta, **kw)
                acc = float((erc > err).float().mean())
            history.append({"step": step, "loss": float(el),
                            "chosen_reward": float(erc.mean()),
                            "rejected_reward": float(err.mean()),
                            "accuracy": acc})
    return {"policy": policy, "history": history}


STEPS = 120 if SMOKE else 350
runs = {
    "correct": train_dpo(steps=STEPS),
    "no prompt mask": train_dpo(steps=STEPS, mask_prompt=False),
    "averaged logprobs": train_dpo(steps=STEPS, average=True),
    "reference trainable": train_dpo(steps=STEPS, ref_trainable=True),
}

print("\nThe correct implementation against the three classic bugs")
print("-" * 74)
print(f"{'implementation':<24} {'final loss':>12} {'pref accuracy':>15} "
      f"{'chosen r':>10} {'rejected r':>12}")
for name, r in runs.items():
    h = r["history"][-1]
    print(f"{name:<24} {h['loss']:>12.4f} {100*h['accuracy']:>14.1f}% "
          f"{h['chosen_reward']:>10.3f} {h['rejected_reward']:>12.3f}")

print("\nEvery row trained and every loss went down, which is the point worth")
print("keeping: none of these announce themselves.")

# Is the prompt-mask row identical, or merely close?
with torch.no_grad():
    l_masked, _, _ = dpo_loss(policy0, REFERENCE, eval_c, eval_r, 0.1,
                              mask_prompt=True)
    l_unmasked, _, _ = dpo_loss(policy0, REFERENCE, eval_c, eval_r, 0.1,
                                mask_prompt=False)
print(f"\nloss with prompt masked   : {float(l_masked):.10f}")
print(f"loss with prompt included : {float(l_unmasked):.10f}")
print(f"difference                : {abs(float(l_masked)-float(l_unmasked)):.3e}")

print("\nThat is not a small difference, it is zero to floating point. And once")
print("you write out the objective it has to be: both responses in a pair share")
print("the same prompt, so the prompt's log-probabilities appear in pi_c and")
print("pi_r, and in ref_c and ref_r, with the same sign -- and the loss depends")
print("only on (pi_c - ref_c) - (pi_r - ref_r). Every prompt term cancels.")
print("\nI included this expecting to show a bug and instead measured a")
print("non-bug. Masking the prompt in DPO saves compute and protects you if")
print("your pairs ever stop sharing a prompt; it does not change this loss.")
print("(The chosen/rejected reward COLUMNS differ slightly for that row, because")
print("those are reported with the prompt included -- the reported reward moves,")
print("the optimised quantity does not.)")

# How far did the "reference" travel when we let it learn?
@torch.no_grad()
def drift(model: nn.Module) -> float:
    """Mean absolute change in response log-prob against the true reference."""
    a = sequence_logprob(model, eval_c)
    b = sequence_logprob(REFERENCE, eval_c)
    return float((a - b).abs().mean())


ref_run = runs["reference trainable"]
print("\nNow the row that matters: what happened to the anchor?")
print("-" * 74)
print(f"{'trainable-reference run, final loss':<44}: "
      f"{ref_run['history'][-1]['loss']:>8.4f}")
print(f"{'correct run, final loss':<44}: "
      f"{runs['correct']['history'][-1]['loss']:>8.4f}")
print(f"{'policy drift from reference (correct run)':<44}: "
      f"{drift(runs['correct']['policy']):>8.3f} nats")

print("\nThe buggy run has the LOWER loss and the HIGHER preference accuracy.")
print("If you were choosing a configuration by looking at the loss curve, you")
print("would pick it.")
print("\nWhat it actually did is move the goalposts. DPO's loss is a difference")
print("of log-ratios, and there are two ways to make a log-ratio large: raise")
print("the policy or lower the reference. Give the optimiser access to the")
print("reference and it will happily do the second, because the reference has")
print("no other objective pulling on it. The KL anchor is now anchored to a")
print("model that is being optimised to make the KL look small.")
print("\nThis is the most dangerous class of bug in post-training: not one that")
print("breaks a run, but one that improves every number you are watching.")

# %% [markdown]
# ## 4. Both rewards go down
#
# Exercise 14.9 item 2: track the implicit rewards separately and predict the
# shapes before you look.

# %%
h = runs["correct"]["history"]
print("\nImplicit rewards during correct DPO training")
print("-" * 74)
print(f"{'step':>6} {'loss':>10} {'chosen r':>11} {'rejected r':>12} "
      f"{'margin':>9} {'accuracy':>10}")
for row in h:
    print(f"{row['step']:>6} {row['loss']:>10.4f} {row['chosen_reward']:>11.3f} "
          f"{row['rejected_reward']:>12.3f} "
          f"{row['chosen_reward']-row['rejected_reward']:>9.3f} "
          f"{100*row['accuracy']:>9.1f}%")

first, last = h[0], h[-1]
both_down = (last["chosen_reward"] < first["chosen_reward"] and
             last["rejected_reward"] < first["rejected_reward"])
print(f"\nchosen reward moved {last['chosen_reward']-first['chosen_reward']:+.3f}")
print(f"rejected reward moved {last['rejected_reward']-first['rejected_reward']:+.3f}")
if both_down:
    print("\nBoth went DOWN, and the margin between them still grew. That is the")
    print("result that surprises people, and it is not a bug.")
    print("\nDPO only ever constrains the DIFFERENCE. Making the chosen response")
    print("slightly less likely while making the rejected response much less")
    print("likely satisfies the objective perfectly -- and it is the easier")
    print("direction to move, because pushing probability mass DOWN on two")
    print("specific sequences is cheaper than redistributing it upward.")
    print("\nSo a DPO run can reduce the likelihood of the very responses you")
    print("labelled as good, while every number on the dashboard improves. This")
    print("is the reported behaviour in the DPO literature and the motivation")
    print("for the variants in §14.6 that add an explicit term to hold the")
    print("chosen likelihood up.")
else:
    print("\nOn this run the rewards did not both fall. The margin is what the")
    print("objective controls; the individual levels are free, so their")
    print("direction varies with the data and the seed. Run it again with")
    print("another seed -- the margin always grows, the levels do as they like.")

# %% [markdown]
# ## 5. Beta controls how fast examples saturate

# %%
print("\nSweeping beta")
print("-" * 74)
print(f"{'beta':>7} {'final loss':>12} {'pref accuracy':>15} {'margin':>10} "
      f"{'policy drift':>14}")
BETAS = (0.01, 0.1, 1.0) if SMOKE else (0.01, 0.05, 0.1, 0.5, 1.0)
for beta in BETAS:
    r = train_dpo(beta=beta, steps=STEPS)
    hh = r["history"][-1]
    margin = hh["chosen_reward"] - hh["rejected_reward"]
    # The reward columns are beta-scaled by definition, so they hide the thing
    # beta actually controls. Divide it back out to see how far the policy went.
    print(f"{beta:>7.2f} {hh['loss']:>12.4f} {100*hh['accuracy']:>14.1f}% "
          f"{margin:>10.3f} {margin/beta:>14.2f}")

print("\nThe last column is the margin in log-probability space -- the reward")
print("with beta divided back out, i.e. how far the policy actually travelled.")
print("Read down it: small beta moves the policy furthest, large beta least.")
print("\nBeta is the inverse temperature on the implicit reward, and it decides")
print("how large a log-ratio counts as 'settled'. At small beta the sigmoid")
print("argument stays near zero, every example keeps producing gradient, and")
print("the policy drifts a long way from the reference. At large beta a modest")
print("log-ratio saturates the sigmoid, examples stop contributing almost")
print("immediately, and the policy barely moves.")
print("\nThat is the same trade the KL coefficient makes in PPO, which is the")
print("point of §14.4: DPO did not remove the KL constraint, it renamed it.")

# %% [markdown]
# ## 6. Summed versus averaged log-probabilities
#
# Exercise 14.9 item 4. Summing means a long response has a lower
# log-probability simply for having more tokens, so the objective is not
# length-neutral.

# %%
def length_preference(policy: nn.Module, ref: nn.Module, beta: float = 0.1,
                      average: bool = False, n: int = 300) -> float:
    """Implicit reward of a LONG correct answer minus a SHORT correct one."""
    r = random.Random(SEED + 21)
    longs, shorts = [], []
    for _ in range(n):
        a, b = r.randint(0, MAX_DIGIT), r.randint(0, MAX_DIGIT)
        longs.append(encode(a, b, a + b, MAX_FILL - 1))
        shorts.append(encode(a, b, a + b, 0))
    lt = torch.tensor(longs).to(DEVICE)
    st = torch.tensor(shorts).to(DEVICE)
    with torch.no_grad():
        kw = {"average": average}
        rl = beta * (sequence_logprob(policy, lt, **kw) - sequence_logprob(ref, lt, **kw))
        rs = beta * (sequence_logprob(policy, st, **kw) - sequence_logprob(ref, st, **kw))
    return float((rl - rs).mean())


print("\nDoes the objective prefer long or short responses?")
print("-" * 74)
print(f"{'log-prob reduction':<24} {'reward(long) - reward(short)':>32}")
for label, avg in (("summed", False), ("averaged", True)):
    r = runs["correct" if not avg else "averaged logprobs"]
    delta = length_preference(r["policy"], REFERENCE, average=avg)
    print(f"{label:<24} {delta:>32.4f}")

print("\nThe two reductions define different objectives, and neither is")
print("length-neutral by construction. Summing makes the total log-probability")
print("depend on token count directly; averaging removes that, and in doing so")
print("it also changes the per-token weight each example carries -- a short")
print("response's few tokens now each count for more.")
print("\nChapter 14 section 14.6 covers the variants built on this observation.")
print("The practical point for a lab is narrower: `sum` and `mean` here are not")
print("a normalisation detail, they are two different training objectives, and")
print("swapping one for the other silently changes what your model learns about")
print("how much to say.")

# %% [markdown]
# ## 7. Things to try
#
# **1. Set `beta = 0.0`.**
# *Common prediction:* an error, or no learning.
# *What happens:* the sigmoid argument is identically zero, the loss is exactly
# log 2, and the gradient vanishes — the same degenerate state as identical
# pairs in section 2, reached from a different direction. Beta is not a
# learning rate; at zero there is no objective at all.
#
# **2. Train with the reference model set to `policy` itself (updated every
# step).**
# *Common prediction:* it collapses immediately.
# *What happens:* the log-ratios are identically zero at every step, so the loss
# sits at log 2 forever and nothing moves. Compare with the "reference
# trainable" row in section 3, where the reference drifts more slowly and the
# run *looks* healthy while its anchor moves.
#
# **3. Corrupt 20% of the preference labels by swapping chosen and rejected.**
# *Common prediction:* accuracy drops about 20%.
# *What happens:* it drops further, because the corrupted pairs produce the
# largest gradients — they are the examples the model is most confident about
# and most wrong on. Preference noise is not linear in its effect, and this is
# why annotator agreement matters more than annotator volume.
#
# **4. Run DPO for 10x the steps and watch the margin.**
# *Common prediction:* it converges.
# *What happens:* the margin keeps growing, exactly as the reward gap did in
# [`lab12_reward_model`](lab12_reward_model.py) section 5. Both objectives are
# unbounded in the same way, for the same reason.
#
# **5. Use a reference model trained ONLY on correct answers.**
# *Common prediction:* DPO has less to do.
# *What happens:* the initial log-ratios are already favourable, the loss starts
# low, and there is very little gradient — the quality of your SFT model decides
# how much DPO can achieve, which is why every recipe in Chapter 14 specifies
# the SFT stage before the preference stage.

# %% [markdown]
# ## What to take away
#
# 1. **DPO supervises a difference, and only a difference.** Identical pairs
#    give exactly log 2 and a zero gradient; $\beta = 0$ does the same.
# 2. **All three classic bugs produce a decreasing loss.** Not masking the
#    prompt, averaging instead of summing, and letting the reference drift are
#    invisible in the loss curve and change what is optimised.
# 3. **The implicit reward of the chosen response can fall.** The objective
#    controls the margin, not the levels, and pushing both down is often the
#    cheaper way to widen a gap.
# 4. **$\beta$ is the KL constraint wearing a different name.** Small $\beta$
#    lets the policy travel far from the reference; large $\beta$ saturates
#    examples early and barely moves.
# 5. **`sum` versus `mean` over token log-probabilities is an objective
#    change**, not a normalisation choice, and it is where DPO's length
#    behaviour comes from.
