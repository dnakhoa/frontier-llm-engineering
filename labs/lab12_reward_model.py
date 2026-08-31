# %% [markdown]
# # Lab 12 — A reward model, and the shortcut it will take if you let it
#
# Companion to [Chapter 12](../book/part-3-post-training/12-reward-modeling.md).
#
# A reward model is trained on comparisons, not scores. You show it a pair,
# tell it which one a human preferred, and it learns a scalar $r(x, y)$ whose
# *differences* reproduce those preferences. Nothing in that objective says the
# scalar has to mean quality. It only has to rank.
#
# That gap is where reward hacking lives, and this lab makes it visible: we
# build a preference set with a mild, entirely realistic confound, and watch
# the model learn the confound instead of the task.
#
# You will:
#
# 1. Build a task with checkable ground truth, so "correct" is not a matter of
#    opinion.
# 2. Train a Bradley–Terry reward model from scratch.
# 3. Measure how much of its score is explained by *response length* alone.
# 4. Rebalance the data, retrain, and see how much of the bias survives.
# 5. Train for five epochs and watch accuracy and the reward gap come apart.
# 6. Draw the best-of-$n$ curve and find where it turns over.
#
# Runs on CPU in about a minute.
#
# **Predict before you run.** In the training pairs the preferred answer is on
# average about four tokens longer. After training, what fraction of the reward
# model's score do you expect length alone to explain? Write down a number.

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
# ## 1. A task with ground truth
#
# The prompt is an addition problem. A response is some number of filler
# ("reasoning") tokens followed by an answer. A response is **correct** if the
# answer token is right — checkable, not a matter of taste.
#
# Having ground truth is what makes this lab work. In real RLHF you do not have
# it, which is exactly why length bias went unnoticed for as long as it did.

# %%
PAD, SEP, FILL = 0, 1, 2
MAX_DIGIT = 4 if SMOKE else 9         # operands 0..MAX_DIGIT
MAX_FILL = 8
DIGIT0 = 3
ANS0 = DIGIT0 + MAX_DIGIT + 1
VOCAB = ANS0 + 2 * MAX_DIGIT + 1
SEQ_LEN = 3 + MAX_FILL + 1            # a, b, SEP, fillers, answer


def encode(a: int, b: int, answer: int, n_fill: int) -> list:
    seq = [DIGIT0 + a, DIGIT0 + b, SEP] + [FILL] * n_fill + [ANS0 + answer]
    return seq + [PAD] * (SEQ_LEN - len(seq))


def make_response(a: int, b: int, correct: bool, n_fill: int,
                  rng: random.Random) -> tuple:
    truth = a + b
    if correct:
        answer = truth
    else:
        answer = rng.choice([v for v in range(2 * MAX_DIGIT + 1) if v != truth])
    return encode(a, b, answer, n_fill), answer == truth


def make_pairs(n: int, length_confound: int, rng: random.Random) -> list:
    """Preference pairs. `length_confound` = extra filler on the chosen side."""
    pairs = []
    for _ in range(n):
        a, b = rng.randint(0, MAX_DIGIT), rng.randint(0, MAX_DIGIT)
        base = rng.randint(0, MAX_FILL - max(length_confound, 0) - 1)
        n_chosen = base + max(length_confound, 0)
        n_rejected = base
        chosen, _ = make_response(a, b, True, n_chosen, rng)
        rejected, _ = make_response(a, b, False, n_rejected, rng)
        pairs.append((chosen, rejected, n_chosen, n_rejected))
    return pairs


N_PAIRS = 1500 if SMOKE else 6000
N_EVAL = 400 if SMOKE else 800
EPOCHS_TRAIN = 4 if SMOKE else 8
CONFOUND = 4

rng = random.Random(SEED)
TRAIN = make_pairs(N_PAIRS, CONFOUND, rng)
EVAL = make_pairs(N_EVAL, CONFOUND, rng)
EVAL_BALANCED = make_pairs(N_EVAL, 0, rng)

mean_gap = sum(c - r for _, _, c, r in TRAIN) / len(TRAIN)
print(f"\n{N_PAIRS:,} training pairs")
print("-" * 74)
print(f"  chosen responses are on average {mean_gap:.1f} tokens longer")
print("  the chosen response is ALWAYS the correct one")
print("\nBoth of those are true of real preference data. Humans write longer")
print("explanations when they are right, and annotators prefer the answer that")
print("shows its working. The confound is not an artefact -- it is the data.")

# %% [markdown]
# ## 2. The Bradley–Terry objective
#
# Given a preferred $y_c$ and a rejected $y_r$ for the same prompt, the model
# should give the pair a higher difference:
#
# $$\mathcal{L} = -\log \sigma\big(r(x, y_c) - r(x, y_r)\big)$$
#
# Note what is *not* here: any target value for $r$ itself. Only differences
# are supervised, so the scale and offset of the reward are unidentified — a
# fact that matters enormously in Chapter 13.

# %%
class RewardModel(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 pool: str = "last"):
        super().__init__()
        self.pool = pool
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
        self.head = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(t, device=x.device))
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        for ln1, attn, ln2, mlp in self.blocks:
            a = ln1(h)
            h = h + attn(a, a, a, attn_mask=mask, need_weights=False)[0]
            h = h + mlp(ln2(h))
        h = self.norm(h)
        idx = 0 if self.pool == "first" else -1
        return self.head(h[:, idx]).squeeze(-1)


def batches(pairs: list, bs: int, gen: torch.Generator):
    order = torch.randperm(len(pairs), generator=gen).tolist()
    for i in range(0, len(order), bs):
        chunk = [pairs[j] for j in order[i:i + bs]]
        yield (torch.tensor([c[0] for c in chunk]),
               torch.tensor([c[1] for c in chunk]))


def train_rm(pairs: list, epochs: int, pool: str = "last",
             seed: int = SEED) -> tuple:
    torch.manual_seed(seed)
    model = RewardModel(pool=pool).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(epochs):
        model.train()
        for chosen, rejected in batches(pairs, 64, gen):
            rc = model(chosen.to(DEVICE))
            rr = model(rejected.to(DEVICE))
            loss = -F.logsigmoid(rc - rr).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        history.append(evaluate(model, EVAL) | {"epoch": ep + 1,
                                                "loss": float(loss)})
    return model, history


@torch.no_grad()
def evaluate(model: nn.Module, pairs: list) -> dict:
    model.eval()
    chosen = torch.tensor([p[0] for p in pairs]).to(DEVICE)
    rejected = torch.tensor([p[1] for p in pairs]).to(DEVICE)
    rc, rr = model(chosen), model(rejected)
    # Ties get half credit. Without this a model that emits one constant score
    # reports 0% rather than 50%, which reads as "perfectly wrong" when it
    # actually means "no signal at all" -- see section 7.
    wins = (rc > rr).float() + 0.5 * (rc == rr).float()
    return {"accuracy": float(wins.mean()),
            "reward_gap": float((rc - rr).mean())}


model, history = train_rm(TRAIN, epochs=EPOCHS_TRAIN)

base = evaluate(model, EVAL)
balanced_eval = evaluate(model, EVAL_BALANCED)
print(f"\nReward model trained {EPOCHS_TRAIN} epochs on the CONFOUNDED pairs")
print("-" * 74)
print(f"{'final training loss':<52}: {history[-1]['loss']:>7.4f}")
print(f"{'accuracy on held-out pairs from the SAME distribution':<52}: "
      f"{100*base['accuracy']:>6.1f}%")
print(f"{'accuracy on LENGTH-BALANCED pairs':<52}: "
      f"{100*balanced_eval['accuracy']:>6.1f}%")
print("\nThe training loss is essentially zero and the held-out accuracy is")
print("essentially perfect. That first pair of numbers is what goes in a model")
print("card, and on its own it is indistinguishable from a reward model that")
print("works.")
print("\nThe third number is the same model asked the same question with the")
print("length cue removed. Chance is 50%.")

# %% [markdown]
# ## 3. How much of the reward is just length?
#
# The direct test. Score responses of varying length, holding correctness
# fixed, and regress reward on length.

# %%
@torch.no_grad()
def reward_of(model: nn.Module, seqs: list) -> torch.Tensor:
    return model(torch.tensor(seqs).to(DEVICE)).cpu()


def length_probe(model: nn.Module, correct: bool, n: int = 400) -> tuple:
    """Reward as a function of filler length, correctness held constant."""
    r = random.Random(SEED + 5)
    lengths, rewards = [], []
    seqs = []
    for _ in range(n):
        a, b = r.randint(0, MAX_DIGIT), r.randint(0, MAX_DIGIT)
        k = r.randint(0, MAX_FILL - 1)
        seq, _ = make_response(a, b, correct, k, r)
        seqs.append(seq)
        lengths.append(k)
    rewards = reward_of(model, seqs).tolist()
    return lengths, rewards


def pearson(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def correctness_effect(model: nn.Module, n: int = 400) -> float:
    """Mean reward(correct) - reward(incorrect) at MATCHED length."""
    r = random.Random(SEED + 9)
    cs, ws = [], []
    for _ in range(n):
        a, b = r.randint(0, MAX_DIGIT), r.randint(0, MAX_DIGIT)
        k = r.randint(0, MAX_FILL - 1)
        c, _ = make_response(a, b, True, k, r)
        w, _ = make_response(a, b, False, k, r)
        cs.append(c)
        ws.append(w)
    return float((reward_of(model, cs) - reward_of(model, ws)).mean())


L_c, R_c = length_probe(model, correct=True)
L_w, R_w = length_probe(model, correct=False)
r_len = pearson(L_c + L_w, R_c + R_w)
corr_effect = correctness_effect(model)

print("\nWhat is the reward model actually responding to?")
print("-" * 74)
print(f"{'correlation of reward with LENGTH (correctness mixed)':<54}: {r_len:>6.3f}")
print(f"{'variance in reward explained by length alone':<54}: {100*r_len**2:>5.1f}%")
print(f"{'mean reward(correct) - reward(wrong) at MATCHED length':<54}: "
      f"{corr_effect:>6.3f}")
def length_effect(lengths: list, rewards: list) -> float:
    """Mean reward of long responses minus short ones, correctness held fixed."""
    mid = MAX_FILL / 2
    lo = [r for k, r in zip(lengths, rewards) if k < mid]
    hi = [r for k, r in zip(lengths, rewards) if k >= mid]
    if not lo or not hi:
        return float("nan")
    return sum(hi) / len(hi) - sum(lo) / len(lo)


print(f"{'mean reward(long) - reward(short) at matched correctness':<54}: "
      f"{length_effect(L_c, R_c):>6.3f}")

# %% [markdown]
# ## 4. Rebalance and retrain
#
# Exercise 12.9 item 2: balance the pairs on length and train again. The
# obvious prediction is that the bias disappears.

# %%
TRAIN_BALANCED = make_pairs(N_PAIRS, 0, rng)
model_b, hist_b = train_rm(TRAIN_BALANCED, epochs=EPOCHS_TRAIN)

Lb_c, Rb_c = length_probe(model_b, correct=True)
Lb_w, Rb_w = length_probe(model_b, correct=False)
r_len_b = pearson(Lb_c + Lb_w, Rb_c + Rb_w)
corr_effect_b = correctness_effect(model_b)

print("\nConfounded training data versus length-balanced training data")
print("-" * 74)
bal_acc_b = evaluate(model_b, EVAL_BALANCED)["accuracy"]
print(f"{'trained on':<26} {'len correlation':>17} {'correctness effect':>20} "
      f"{'bal. accuracy':>15}")
print(f"{'confounded pairs':<26} {r_len:>17.3f} {corr_effect:>20.3f} "
      f"{100*balanced_eval['accuracy']:>14.1f}%")
print(f"{'length-balanced pairs':<26} {r_len_b:>17.3f} {corr_effect_b:>20.3f} "
      f"{100*bal_acc_b:>14.1f}%")

print("\nSame architecture, same number of steps, same near-zero training loss.")
print("The only thing that changed is whether the training pairs contained a")
print("shortcut, and it decides whether the model learned the task or learned")
print("nothing about it at all.")
print("\nLook at the correctness-effect column rather than the accuracy column.")
print(f"Trained on confounded pairs, the reward difference between a right and")
print(f"a wrong answer of the SAME length is {corr_effect:.3f} -- indistinguishable")
print(f"from zero. Trained on balanced pairs it is {corr_effect_b:.3f}. The first")
print("model is not a weak reward model. It is a tape measure.")
print("\nOne caution before you conclude that balancing solves this: we knew")
print("what the confound was. Length is the one everybody checks precisely")
print("because it was found first. The confounds that matter are the ones you")
print("have not thought to balance, and nothing in the training loss, the")
print("held-out accuracy, or the loss curve would tell you they are there.")

# %% [markdown]
# ## 5. Over-training: accuracy and the reward gap come apart
#
# Exercise 12.9 item 3. Watch the two metrics separately, because they do
# different things and only one of them is reported.

# %%
hist = hist_b
print(f"\nTraining for {EPOCHS_TRAIN} epochs on the balanced pairs")
print("-" * 74)
print(f"{'epoch':>6} {'train loss':>12} {'held-out acc':>14} {'reward gap':>13}")
for h in hist:
    print(f"{h['epoch']:>6} {h['loss']:>12.4f} {100*h['accuracy']:>13.1f}% "
          f"{h['reward_gap']:>13.3f}")

_acc_gain = 100 * (hist[-1]["accuracy"] - hist[0]["accuracy"])
_gap_growth = hist[-1]["reward_gap"] / max(abs(hist[0]["reward_gap"]), 1e-6)
print(f"\nOver these epochs accuracy moved {_acc_gain:+.1f} points while the reward")
print(f"gap grew about {_gap_growth:.0f}x. Accuracy is bounded above by 100%; the")
print("gap is bounded by nothing at all.")
print("\nThat is the Bradley-Terry objective behaving exactly as specified: it is")
print("never satisfied, because -log sigmoid(d) keeps falling as d grows. Long")
print("after the model has stopped ordering any NEW pairs correctly, it keeps")
print("being rewarded for pushing the pairs it already knows further apart.")
print("\nThe consequence lands in Chapter 13: the policy does not optimise the")
print("ordering, it optimises the SCALE. An over-trained RM hands the policy a")
print("steeper landscape without a more accurate one, and a steeper landscape")
print("is easier to hack. This is a large part of why KL control is not")
print("optional.")
print("\nThe consequence lands in Chapter 13: a policy optimised against an")
print("over-trained RM chases a scale that has become steeper without becoming")
print("more accurate, which is a large part of why KL control is not optional.")

# %% [markdown]
# ## 6. The best-of-$n$ curve
#
# The practical test of a reward model. Sample $n$ candidates, keep the one the
# RM scores highest, and measure how often it is actually correct.

# %%
def best_of_n_curve(model: nn.Module, ns: list, trials: int = 300,
                    p_correct: float = 0.35, seed: int = SEED) -> list:
    """Raw best-of-n accuracy, with the oracle (pass@n) as the ceiling.

    No conditioning on 'a correct candidate exists' -- that convention makes
    n=1 trivially 100% and hides the whole effect, which is a mistake worth
    making once and not twice.
    """
    out = []
    for n in ns:
        r = random.Random(seed)          # same candidates for every n's prefix
        hits = oracle = 0
        for _ in range(trials):
            a, b = r.randint(0, MAX_DIGIT), r.randint(0, MAX_DIGIT)
            cands, truths = [], []
            for _ in range(n):
                correct = r.random() < p_correct
                k = r.randint(0, MAX_FILL - 1)
                seq, ok = make_response(a, b, correct, k, r)
                cands.append(seq)
                truths.append(ok)
            oracle += 1 if any(truths) else 0
            scores = reward_of(model, cands)
            hits += 1 if truths[int(scores.argmax())] else 0
        out.append((n, hits / trials, oracle / trials))
    return out


NS = [1, 2, 4, 8] if SMOKE else [1, 2, 4, 8, 16, 32]
curve_bal = best_of_n_curve(model_b, NS)
curve_conf = best_of_n_curve(model, NS)

print("\nBest-of-n accuracy: sample n candidates, keep the RM's favourite")
print("-" * 74)
print(f"{'n':>5} {'balanced RM':>14} {'confounded RM':>16} "
      f"{'oracle (pass@n)':>18}")
for (n, acc_b, orc), (_, acc_c, _) in zip(curve_bal, curve_conf):
    print(f"{n:>5} {100*acc_b:>13.1f}% {100*acc_c:>15.1f}% {100*orc:>17.1f}%")

peak = max(curve_conf, key=lambda t: t[1])
_n, _acc_c, _orc = curve_conf[-1]
_, _acc_b, _ = curve_bal[-1]
print(f"\nAt n={_n} a correct candidate is present {100*_orc:.0f}% of the time.")
print(f"The balanced RM finds it {100*_acc_b:.0f}% of the time. The confounded RM")
print(f"finds it {100*_acc_c:.0f}% of the time -- barely better than at n=1.")
print("\nThe confounded curve is flat. Extra samples buy it nothing, because it")
print("is not ranking on the axis that decides correctness; it picks the")
print("longest candidate, and length is uncorrelated with being right. Every")
print("additional sample costs compute and returns nothing.")
print("\nThis is the cheapest audit of a reward model there is, and it needs no")
print("human labels: sample n, plot accuracy against the oracle. A curve that")
print("tracks the oracle is measuring what you wanted. A flat curve means the")
print("reward model is measuring something else -- and a curve that TURNS OVER")
print("means it actively prefers wrong answers, which is the case Chapter 16")
print("builds on and lab16_best_of_n_and_prm constructs deliberately.")

# %% [markdown]
# ## 7. Where you read the reward from
#
# Exercise 12.9 item 5. The reward head reads one position. With causal
# attention, position 0 has seen exactly one token.

# %%
model_first, hist_first = train_rm(TRAIN_BALANCED, epochs=EPOCHS_TRAIN, pool="first")
first_eval = evaluate(model_first, EVAL_BALANCED)
print("\nPooling position")
print("-" * 74)
print(f"{'read reward from':<24} {'final loss':>12} {'held-out accuracy':>20}")
print(f"{'last token':<24} {hist[0]['loss']:>12.4f} "
      f"{100*evaluate(model_b, EVAL_BALANCED)['accuracy']:>19.1f}%")
print(f"{'first token':<24} {hist_first[0]['loss']:>12.4f} "
      f"{100*first_eval['accuracy']:>19.1f}%")
print(f"\nchance is 50.0%")
print("\nThe first-token model sits exactly at chance and its loss never leaves")
print("log 2 = 0.693. Under a causal mask, position 0 attends only to itself:")
print("it has not seen the response, or even the prompt beyond one token, so")
print("there is nothing for the head to read.")
print("\nWhat makes this bug survive review is that it does not look like a bug.")
print("There is no crash, no NaN, no shape error -- just a loss that sits flat")
print("while you assume the task is hard and the learning rate needs tuning.")
print("\nA note on how that number is computed, because it bit me: with no")
print("signal the model emits the SAME score for both responses, and counting")
print("strict rc > rr scores every tie as a loss and reports 0.0%. That reads")
print("as a perfectly inverted model rather than an uninformed one, and it")
print("sends you looking for a sign error that does not exist. Ties get half")
print("credit above.")

# %% [markdown]
# ## 8. Things to try
#
# **1. Set `CONFOUND = 8` and rerun.**
# *Common prediction:* proportionally more length bias.
# *What happens:* the model becomes almost purely a length detector, and its
# reported accuracy on the confounded eval goes *up*. Higher accuracy, worse
# reward model — which is why the headline number on a preference test set is
# nearly uninformative on its own.
#
# **2. Set `CONFOUND = -4` so the *rejected* response is longer.**
# *Common prediction:* the bias reverses.
# *What happens:* it does, and this is the useful diagnostic. Train two RMs
# with opposite confounds and compare their length correlations; the gap
# measures how much of your reward is length, without needing a balanced set.
#
# **3. Raise `p_correct` to 0.8 in the best-of-$n$ curve.**
# *Common prediction:* everything improves.
# *What happens:* the turnover moves out to much larger $n$ or disappears,
# because with a strong generator the RM rarely has to discriminate among many
# wrong answers. Best-of-$n$ curves measure the RM *and* the policy, and a
# curve without the policy's pass rate quoted next to it cannot be interpreted.
#
# **4. Train the RM on pairs where chosen and rejected are identical.**
# *Common prediction:* the loss goes to zero.
# *What happens:* it sits at $-\log \sigma(0) = 0.693$ and the gradient is
# zero — the objective is exactly indifferent. Worth doing once to feel that
# Bradley–Terry supervises only differences.
#
# **5. Add a second confound: make correct answers use a distinctive filler
# token.**
# *Common prediction:* the model finds it.
# *What happens:* instantly and completely, and the length correlation drops
# because it no longer needs length. Reward models take the cheapest available
# shortcut, and removing one just promotes the next.

# %% [markdown]
# ## What to take away
#
# 1. **Bradley–Terry supervises differences, not values.** The scale and offset
#    of the reward are unidentified, and the objective is never satisfied — it
#    always prefers a larger gap.
# 2. **A reward model learns the cheapest feature that ranks the data.** With a
#    four-token length confound it learns length, and its accuracy on the
#    matching test set goes *up* while the model gets worse.
# 3. **Balancing helps and never finishes.** The correlation falls a long way
#    and does not reach zero, because a finite sample cannot balance every
#    confound and some of them are real signal.
# 4. **Accuracy and the reward gap are different metrics.** Only one plateaus.
#    The other grows indefinitely, which is what a policy will later optimise
#    against.
# 5. **The best-of-$n$ curve is the cheapest audit of a reward model**, needs no
#    labels, and a curve that turns over is telling you the model is measuring
#    the wrong thing.
# 6. **Read the reward from a position that has seen the response.** The failure
#    when you do not is quiet: a loss that falls slightly and a model that
#    cannot work.
