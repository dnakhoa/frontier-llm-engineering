# %% [markdown]
# # Lab 16 — Best-of-$n$, majority voting, and what a PRM aggregator decides
#
# Companion to
# [Chapter 16](../book/part-3-post-training/16-process-reward-models.md).
#
# Search-time compute is the cheapest capability gain available: sample $n$
# solutions instead of one and pick a good one. Everything then depends on
# *pick*, and the three standard answers — majority voting, an outcome reward
# model, a process reward model — behave very differently as $n$ grows.
#
# You will:
#
# 1. Build multi-step solutions where step correctness is known, including the
#    case that motivates PRMs: a right answer reached through a wrong step.
# 2. Compare majority voting against best-of-$n$ at $n = 4, 16, 64$.
# 3. Degrade the verifier with noise and find where majority voting overtakes.
# 4. Build a verifier that quietly prefers longer solutions and find the
#    turnover in its best-of-$n$ curve.
# 5. Compare min, product and mean as PRM aggregators.
#
# **What is real here and what is not.** The selection strategies, the
# aggregators and the statistics are real implementations. The *generator* is a
# model of a language model — solutions are sampled from a specified per-step
# error process rather than decoded from a network. That keeps the lab instant
# and, more importantly, gives us ground-truth step labels, which is exactly
# what makes PRM behaviour measurable. Read the mechanisms as real and the
# absolute accuracies as parameters we chose. Same contract as
# [`lab07`](lab07_collective_bandwidth.py).
#
# Pure standard library. Runs in seconds.
#
# **Predict before you run.** At $n = 64$, does majority voting or best-of-$n$
# with a good verifier win? And is the gap between them growing or shrinking
# with $n$? Write both down.

# %%
from __future__ import annotations

import math
import os
import random
from collections import Counter
from dataclasses import dataclass

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
SEED = 0
print(f"smoke_test={SMOKE}")

# %% [markdown]
# ## 1. Problems, solutions, and the case that motivates PRMs
#
# A problem is a chain of arithmetic operations. A solution is the sequence of
# intermediate values a model would write down. Each step is produced correctly
# with probability $p$; otherwise the value is perturbed, and the error
# propagates into every later step.
#
# Because the perturbations are small integers, a later error sometimes cancels
# an earlier one — the solution reaches the **right answer through wrong
# reasoning**. That case is the entire reason process supervision exists, and
# it is invisible to any verifier that reads only the final answer.

# %%
N_STEPS = 5
P_STEP = 0.82                # per-step probability of doing the operation right


@dataclass
class Solution:
    steps: list                # the intermediate values written down
    step_correct: list         # ground truth per step
    final: int
    correct: bool              # final answer matches the truth
    length: int                # "tokens" of working shown

    @property
    def all_steps_correct(self) -> bool:
        return all(self.step_correct)


def make_problem(rng: random.Random) -> tuple:
    """Return (ops, true intermediate values)."""
    value = rng.randint(1, 9)
    ops, truth = [], []
    for _ in range(N_STEPS):
        op = rng.choice(["+", "-", "*"])
        operand = rng.randint(1, 5)
        ops.append((op, operand))
        if op == "+":
            value += operand
        elif op == "-":
            value -= operand
        else:
            value *= operand
        truth.append(value)
    return ops, truth


def sample_solution(ops: list, truth: list, rng: random.Random,
                    p_step: float = P_STEP) -> Solution:
    start = truth[0]
    # Recompute forward from a running value so errors PROPAGATE.
    value = None
    steps, correct_flags = [], []
    for i, (op, operand) in enumerate(ops):
        if i == 0:
            base = truth[0]
            if op == "+":
                prev = base - operand
            elif op == "-":
                prev = base + operand
            else:
                prev = base // operand if operand else base
            value = prev
        if op == "+":
            value = value + operand
        elif op == "-":
            value = value - operand
        else:
            value = value * operand
        if rng.random() >= p_step:                 # the model slipped
            value += rng.choice([-2, -1, 1, 2])
        steps.append(value)
        correct_flags.append(value == truth[i])
    # Length of working shown: a model that rambles writes more.
    length = sum(rng.randint(4, 12) for _ in range(N_STEPS))
    return Solution(steps, correct_flags, steps[-1], steps[-1] == truth[-1], length)


rng = random.Random(SEED)
N_PROBLEMS = 300 if SMOKE else 1200
MAX_N = 16 if SMOKE else 64

PROBLEMS = []
for _ in range(N_PROBLEMS):
    ops, truth = make_problem(rng)
    sols = [sample_solution(ops, truth, rng) for _ in range(MAX_N)]
    PROBLEMS.append({"ops": ops, "truth": truth, "solutions": sols})

all_sols = [s for p in PROBLEMS for s in p["solutions"]]
n_correct = sum(s.correct for s in all_sols)
n_lucky = sum(1 for s in all_sols if s.correct and not s.all_steps_correct)

print(f"\n{N_PROBLEMS:,} problems x {MAX_N} sampled solutions")
print("-" * 74)
print(f"{'per-step accuracy':<44}: {P_STEP:.2f}")
print(f"{'solutions with the correct final answer':<44}: "
      f"{100*n_correct/len(all_sols):>5.1f}%")
print(f"{'  of those, ones with a WRONG step':<44}: "
      f"{100*n_lucky/max(n_correct,1):>5.1f}%")
print(f"{'solutions with every step correct':<44}: "
      f"{100*sum(s.all_steps_correct for s in all_sols)/len(all_sols):>5.1f}%")
print("\nThat second number is the PRM's whole argument. A meaningful share of")
print("correct answers are reached through reasoning that contains an error,")
print("and an outcome-only verifier marks every one of them as a good example")
print("to imitate.")

# %% [markdown]
# ## 2. Majority voting versus best-of-$n$
#
# **Majority voting** (self-consistency) needs no verifier at all: sample $n$
# solutions, return the most common final answer.
#
# **Best-of-$n$** needs a verifier: score each solution, return the best.

# %%
def majority_vote(sols: list) -> int:
    counts = Counter(s.final for s in sols)
    top = max(counts.values())
    # Deterministic tie-break, so the result does not depend on dict order.
    return min(v for v, c in counts.items() if c == top)


def best_of_n(sols: list, scorer) -> Solution:
    return max(sols, key=scorer)


def orm_scorer(noise: float = 0.0, seed: int = SEED):
    """Outcome reward model: reads only the final answer, with optional noise."""
    r = random.Random(seed)

    def score(s: Solution) -> float:
        return (1.0 if s.correct else 0.0) + r.gauss(0.0, noise)
    return score


def evaluate_strategy(ns: list, strategy) -> list:
    out = []
    for n in ns:
        hits = 0
        for p in PROBLEMS:
            sols = p["solutions"][:n]
            hits += 1 if strategy(sols, p["truth"][-1]) else 0
        out.append((n, hits / len(PROBLEMS)))
    return out


NS = [1, 4, 16] if SMOKE else [1, 4, 16, 64]

maj = evaluate_strategy(NS, lambda sols, t: majority_vote(sols) == t)
bon = evaluate_strategy(NS, lambda sols, t: best_of_n(sols, orm_scorer(0.0)).correct)
oracle = evaluate_strategy(NS, lambda sols, t: any(s.correct for s in sols))

print("\nSelection strategies as n grows")
print("-" * 74)
print(f"{'n':>5} {'majority vote':>15} {'best-of-n (clean ORM)':>24} "
      f"{'oracle pass@n':>16}")
for (n, m), (_, b), (_, o) in zip(maj, bon, oracle):
    print(f"{n:>5} {100*m:>14.1f}% {100*b:>23.1f}% {100*o:>15.1f}%")

print("\nA noiseless verifier is equivalent to the oracle -- it finds a correct")
print("solution whenever one exists -- so that column is an upper bound, not a")
print("realistic system. The interesting comparison is majority voting, which")
print("uses no verifier at all and still climbs steeply, then flattens.")
print("\nMajority voting saturates because it is limited by whether the MODE is")
print("correct. Once enough samples exist to identify the mode, more samples")
print("add nothing, and if the model's most common answer is wrong then no")
print("amount of sampling fixes it. Best-of-n has no such ceiling, provided the")
print("verifier can tell the difference -- which is the assumption section 3")
print("takes away.")

# %% [markdown]
# ## 3. A noisy verifier, and where majority voting overtakes
#
# Real verifiers are wrong sometimes. Add Gaussian noise to the ORM's score and
# sweep it. Exercise 16.8 item 2 asks where the crossover is.

# %%
NOISES = (0.0, 0.5, 1.0, 2.0) if SMOKE else (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
N_FIXED = NS[-1]

print(f"\nVerifier noise sweep at n={N_FIXED}")
print("-" * 74)
print(f"{'noise sigma':>13} {'best-of-n':>12} {'majority':>11} {'winner':>12}")
maj_at_n = dict(maj)[N_FIXED]
crossover = None
for sigma in NOISES:
    acc = 0
    for i, p in enumerate(PROBLEMS):
        sols = p["solutions"][:N_FIXED]
        acc += 1 if best_of_n(sols, orm_scorer(sigma, seed=SEED + i)).correct else 0
    acc /= len(PROBLEMS)
    winner = "best-of-n" if acc > maj_at_n else "majority"
    if crossover is None and acc <= maj_at_n:
        crossover = sigma
    print(f"{sigma:>13.2f} {100*acc:>11.1f}% {100*maj_at_n:>10.1f}% {winner:>12}")

if crossover is not None:
    print(f"\nMajority voting overtakes at sigma ~= {crossover:.2f}, i.e. once the")
    print("verifier's noise is comparable to the gap it is trying to resolve")
    print("(the signal here is 1.0: correct scores 1, incorrect scores 0).")
else:
    print("\nBest-of-n stayed ahead across this whole sweep. Widen NOISES to")
    print("find the crossover -- it exists, it is just further out.")

print("\nThe practical reading: a verifier is only worth using while it is more")
print("reliable than the model's own consensus. Below that, self-consistency is")
print("both cheaper and better, because it needs no verifier to train, host, or")
print("keep calibrated.")

# %% [markdown]
# ## 4. A verifier with a taste for long solutions
#
# Exercise 16.8 item 3. Nobody builds this deliberately. You get it by training
# a reward model on data where good answers happened to be longer — exactly the
# confound built in [`lab12_reward_model`](lab12_reward_model.py).

# %%
def length_biased_scorer(weight: float):
    def score(s: Solution) -> float:
        return (1.0 if s.correct else 0.0) * (1.0 - weight) + weight * (s.length / 60.0)
    return score


print("\nBest-of-n with a verifier that partly rewards length")
print("-" * 74)
print(f"{'n':>5} " + " ".join(f"{'w=' + str(w):>10}" for w in (0.0, 0.5, 0.9, 1.0)))
curves = {w: [] for w in (0.0, 0.5, 0.9, 1.0)}
for n in NS:
    row = [f"{n:>5} "]
    for w in curves:
        hits = 0
        for p in PROBLEMS:
            sols = p["solutions"][:n]
            hits += 1 if best_of_n(sols, length_biased_scorer(w)).correct else 0
        acc = hits / len(PROBLEMS)
        curves[w].append((n, acc))
        row.append(f"{100*acc:>9.1f}%")
    print(" ".join(row))

for w, curve in curves.items():
    peak_n, peak_acc = max(curve, key=lambda t: t[1])
    last_n, last_acc = curve[-1]
    turned = peak_n < last_n
    print(f"  w={w}: peaks at n={peak_n} ({100*peak_acc:.1f}%), "
          f"ends at {100*last_acc:.1f}%  -> "
          f"{'TURNS OVER' if turned else 'still rising'}")

print("\nA fully length-driven verifier (w=1.0) gets WORSE with more samples,")
print("and that is the signature to recognise. More samples means more chances")
print("to find a correct solution and more chances to find a long wrong one; if")
print("the verifier ranks on length, the second effect wins and search actively")
print("hurts.")
print("\nThis is why the best-of-n curve is the standard audit for a reward")
print("model. A curve that rises is a verifier tracking correctness; a flat one")
print("is a verifier tracking nothing; a falling one is a verifier tracking")
print("something you did not intend, and scaling compute makes it worse.")

# %% [markdown]
# ## 5. PRM aggregators: min, product, mean
#
# A process reward model scores every step. Turning $k$ step scores into one
# number is a choice, and Exercise 16.8 item 4 asks which choice is right.
#
# The question to hold onto: what has to be true for a solution to be correct?

# %%
def prm_step_scores(s: Solution, noise: float, rng: random.Random) -> list:
    return [(1.0 if ok else 0.0) + rng.gauss(0.0, noise) for ok in s.step_correct]


AGGREGATORS = {
    "min": min,
    "product": lambda xs: math.prod(max(x, 1e-6) for x in xs),
    "mean": lambda xs: sum(xs) / len(xs),
    "last step only": lambda xs: xs[-1],
}

PRM_NOISES = (0.3, 1.0, 2.0) if SMOKE else (0.3, 0.75, 1.5, 3.0)


def aggregator_accuracy(agg, noise: float) -> tuple:
    hits = clean = 0
    for i, prob in enumerate(PROBLEMS):
        r = random.Random(SEED + i)
        sols = prob["solutions"][:N_FIXED]
        scored = [(agg(prm_step_scores(s, noise, r)), s) for s in sols]
        pick = max(scored, key=lambda t: t[0])[1]
        hits += 1 if pick.correct else 0
        clean += 1 if pick.all_steps_correct else 0
    return hits / len(PROBLEMS), clean / len(PROBLEMS)


print(f"\nPRM aggregators at n={N_FIXED}: FULLY-CORRECT selection rate")
print("(did the chosen solution reach the answer with every step right?)")
print("-" * 74)
print(f"{'aggregator':<18} " + " ".join(f"{'sigma=' + str(v):>12}"
                                        for v in PRM_NOISES))
results = {}
for name, agg in AGGREGATORS.items():
    row = []
    for noise in PRM_NOISES:
        _, clean = aggregator_accuracy(agg, noise)
        row.append(clean)
    results[name] = row
    print(f"{name:<18} " + " ".join(f"{100*v:>11.1f}%" for v in row))

print(f"\nSame table, but scoring only the final answer")
print("-" * 74)
print(f"{'aggregator':<18} " + " ".join(f"{'sigma=' + str(v):>12}"
                                        for v in PRM_NOISES))
for name, agg in AGGREGATORS.items():
    row = [aggregator_accuracy(agg, noise)[0] for noise in PRM_NOISES]
    print(f"{name:<18} " + " ".join(f"{100*v:>11.1f}%" for v in row))

print("\nAt low noise every aggregator looks the same, which is worth noticing")
print("on its own: a clean PRM makes the aggregation choice irrelevant, so any")
print("comparison run at low noise will tell you these are interchangeable.")
print("They are not. Read across the rows as sigma grows.")

_lo, _hi = PRM_NOISES[0], PRM_NOISES[-1]
_best_lo = max(results, key=lambda k: results[k][0])
_best_hi = max(results, key=lambda k: results[k][-1])
print(f"\nbest aggregator at sigma={_lo}: {_best_lo}")
print(f"best aggregator at sigma={_hi}: {_best_hi}")

print("\nI expected 'min' to win everywhere and it does not. It wins cleanly")
print("when the PRM is accurate and it degrades fastest as the PRM gets noisy")
print("-- at the high-noise end it is the worst of the four.")
print("\nThe reason is in the word. A minimum over k noisy estimates is decided")
print("by the unluckiest of the k draws, so it does no averaging at all and")
print("inherits the worst error in the solution's score vector. One perfectly")
print("good step that happens to draw a bad sample sinks the whole candidate.")
print("\nThe aggregators that POOL evidence across steps -- product and mean --")
print("degrade more gracefully, because independent step noise partly cancels")
print("before the comparison is made. Which of the two comes out on top at the")
print("noisy end is not stable across runs; that they both overtake min is.")
print("\nSo the ranking is not a fact about aggregators, it is a fact about your")
print("PRM's calibration. Take min if you trust the step scores, and something")
print("that pools them if you do not. Anyone comparing these on a clean PRM")
print("will conclude they are interchangeable and ship whichever they wrote")
print("first.")
print("\nNow read the two tables against each other. 'last step only' is close")
print("to the best on FINAL-ANSWER accuracy and clearly the worst on")
print("FULLY-CORRECT selection -- it is an outcome reward model wearing a PRM's")
print("interface. That gap is the entire argument for process supervision, and")
print("it is invisible if you only measure whether the answer was right.")
print("\nIt matters when the selected solutions become training data --")
print("rejection sampling, expert iteration, STaR. Selecting answers that are")
print("right for the wrong reason teaches the model the wrong reason.")
print("Chapter 16 section 16.5.")

# %% [markdown]
# ## 6. Things to try
#
# **1. Raise `P_STEP` to 0.95 and rerun section 2.**
# *Common prediction:* every strategy improves by a similar amount.
# *What happens:* majority voting improves most and the strategies converge,
# because a strong generator makes the mode almost always right and there is
# little left for a verifier to add. Search-time compute pays most on models
# that are *nearly* capable, which is why it appeared as a headline technique
# exactly when base models got good.
#
# **2. Lower `P_STEP` to 0.5.**
# *Common prediction:* everything degrades gracefully.
# *What happens:* majority voting collapses faster than best-of-$n$, because
# the mode becomes wrong rather than merely uncertain. Voting fails
# catastrophically rather than gradually, and it fails silently — the votes are
# still confident.
#
# **3. Add a soft minimum — e.g. $-\log \sum_i e^{-s_i}$ — as a fifth
# aggregator.**
# *Common prediction:* it lands between min and mean.
# *What happens:* it tracks min at low noise and product at high noise, which
# is why practical systems reach for it. Having watched min lose the high-noise
# column above, the motivation should now be obvious rather than a formula
# someone hands you.
#
# **4. Combine strategies: majority-vote over the top-$k$ by verifier score.**
# *Common prediction:* somewhere between the two.
# *What happens:* it usually beats both, because filtering removes the
# confidently-wrong solutions that poison the vote while the vote absorbs the
# verifier's noise. This is roughly what weighted self-consistency does.
#
# **5. Count how often `min` and `product` disagree.**
# *Common prediction:* rarely, since they encode the same claim.
# *What happens:* they disagree more than expected once noise is present,
# because product accumulates evidence from every step while min listens to
# exactly one. Which you want depends on whether your step scores are
# well-calibrated.

# %% [markdown]
# ## What to take away
#
# 1. **A meaningful share of correct answers contain a wrong step.** Outcome
#    supervision cannot see the difference, and treats every one as a model to
#    imitate.
# 2. **Majority voting saturates; best-of-$n$ need not.** Voting is bounded by
#    whether the mode is right, and no amount of sampling moves the mode.
# 3. **A verifier is worth using only while it beats the model's own
#    consensus.** Past a noise level comparable to the signal it is resolving,
#    self-consistency wins and costs less.
# 4. **A best-of-$n$ curve that falls means your verifier is ranking on
#    something else.** Rising, flat and falling curves are three different
#    diagnoses, and the falling one means more compute makes the system worse.
# 5. **The right PRM aggregator depends on how noisy your PRM is.** min
#    encodes "every step must hold" exactly and wins when step scores are
#    reliable; it is a minimum over noisy estimates, so it degrades fastest and
#    is among the worst once they are not. Aggregators that pool evidence
#    across steps degrade more gracefully, because independent step noise
#    partly cancels. Comparing aggregators on a clean PRM shows no difference
#    at all, which is how the wrong one gets shipped.
# 6. **Selecting right answers is not the same as selecting right reasoning**,
#    and the difference only matters — enormously — when the selected solutions
#    become training data.
