# Chapter 16: Process reward models, verifiers, search-time compute

> Reading time: ~35 minutes. By the end of this chapter you should understand the difference between outcome and process supervision, how PRMs get their step labels without armies of human annotators, the three ways to spend compute at inference time and how each scales, and why the most successful reasoning model of the era explicitly tried process reward models and abandoned them. You should be able to decide whether a PRM is worth building for your problem.

## 16.1 The right answer for the wrong reasons

Chapter 15's reward function asks one question: is the final answer correct?

Consider what that misses. A model solves a probability problem in six steps. Step 3 contains a sign error. Step 4 contains a second sign error that cancels the first. The final answer is correct, and the model receives full reward for a derivation that is wrong in the middle.

Under GRPO, that trace gets a positive advantage and every token in it — including both errors — gets its probability increased. You have trained the model to make compensating mistakes.

This is the **credit assignment problem**, and it is the motivation for process supervision: instead of one reward at the end, score each step.

| | Outcome supervision (ORM) | Process supervision (PRM) |
|---|---|---|
| What is scored | The final answer | Each reasoning step |
| Label cost | Free with a verifier | Expensive |
| Credit assignment | None — all tokens share one signal | Per-step |
| Rewards correct-by-luck | Yes | No |
| Usable as a search signal | Only at the end | At every step |

The theoretical case for process supervision is strong. The empirical case is more complicated, and §16.8 is where this chapter earns its keep.

## 16.2 Verifiers, before PRMs

The lineage starts with Cobbe et al. [\[71\]](../appendix/b-references.md#71-gsm8k-verifiers), the GSM8K paper, in 2021. The idea: instead of only training a model to *generate* solutions, also train one to *judge* them. Then at inference, sample many solutions and let the judge pick.

This is a **verifier** — a learned model that scores complete solutions, trained on (solution, correct?) pairs. It is a reward model (Chapter 12) with a binary ground-truth label instead of a human preference, which makes it much easier to train and much harder to hack.

The result that mattered: sampling 100 solutions and selecting with a verifier beat greedy decoding by a wide margin, and beat a much larger model decoding greedily. **You could buy accuracy with inference compute.** That observation sat relatively quietly for three years before becoming the organizing principle of an entire model generation.

## 16.3 Process reward models

Lightman et al. [\[26\]](../appendix/b-references.md#26-process-reward-models) — "Let's Verify Step by Step" — took the next step: score each reasoning step rather than the whole solution.

They collected PRM800K: roughly 800,000 step-level human annotations on MATH solutions, each step labelled positive, negative, or neutral. Then trained a model to predict those labels.

```python
import torch
import torch.nn as nn


class ProcessRewardModel(nn.Module):
    """Predicts per-step correctness.

    Steps are delimited by a special token during training. The score is read at
    each delimiter position, so one forward pass produces a score for every step
    -- which is what makes PRM-guided search affordable.
    """

    def __init__(self, backbone, step_token_id: int):
        super().__init__()
        self.backbone = backbone
        self.step_token_id = step_token_id
        self.head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        scores = self.head(hidden).squeeze(-1)         # (B, T)
        is_step_end = input_ids == self.step_token_id  # (B, T)
        return scores, is_step_end                     # read scores where True
```

The headline result: process supervision beat outcome supervision on MATH, and the gap widened as you sampled more solutions. The explanation is exactly §16.1 — the PRM rejects solutions that reached the right answer by a wrong route, and at large sample counts those become an increasing fraction of what outcome supervision accepts.

They also released PRM800K, which is why this line of work is reproducible at all.

**The obvious problem: 800,000 human step annotations.** That is a serious annotation project for one domain (competition math) in one language. It does not generalize, and it does not scale.

## 16.4 Getting step labels without humans

The technique that made PRMs practical, from Math-Shepherd [\[73\]](../appendix/b-references.md#73-math-shepherd) and closely related work.

The insight: **a step is good if you can finish from it.**

For a prefix of a solution ending at step $k$, sample $N$ completions from that prefix. The fraction that reach the correct final answer is an estimate of the step's value:

$$V(\text{prefix}_k) \approx \frac{1}{N}\sum_{j=1}^{N} \mathbb{1}\big[\text{completion}_j \text{ is correct}\big]$$

That is Monte Carlo rollout value estimation, and it converts a problem you *can* verify (final answers) into labels for a problem you cannot (intermediate steps).

```python
def estimate_step_values(model, problem, steps, answer_key, n_rollouts=16):
    """Label each step by how often you can finish correctly from it.

    Cost: len(steps) * n_rollouts generations per problem. This is the dominant
    expense of building a PRM without human annotation -- typically far more
    generation than the RL run it will support.
    """
    values = []
    for k in range(1, len(steps) + 1):
        prefix = "\n".join(steps[:k])
        completions = model.sample(problem + prefix, n=n_rollouts)
        correct = sum(
            check_answer(extract_answer(c), answer_key) for c in completions
        )
        values.append(correct / n_rollouts)
    return values  # e.g. [0.9, 0.85, 0.2, 0.15, ...] -- step 3 broke it
```

Read that example output. A sharp drop between consecutive steps localizes the error: steps 1–2 were fine, step 3 was where it went wrong. That is exactly the signal §16.1 wanted, obtained with no human involvement.

**The costs, which are not small:**

- **Generation.** With 8 steps and 16 rollouts, that is 128 generations *per problem*, before any training. For 100,000 problems, 12.8 million generations. This is frequently more compute than the RL run the PRM is meant to improve.
- **It inherits the policy's ability.** The value estimate is "how often can *this model* finish from here." A step that a stronger model could complete is scored as bad. The PRM measures difficulty-for-the-current-policy, not correctness, and those diverge.
- **It is noisy at both ends.** Steps early in a solution have high value regardless (many routes remain); steps immediately before a correct answer have value 1. The informative region is the middle.

## 16.5 Two ways to use a PRM

Once you have one, there are two distinct applications, and conflating them causes confusion.

**Search-time (inference).** Use the PRM to guide generation. The model is fixed; you spend inference compute to get better answers. Covered in §16.6.

**Training-time.** Use per-step scores as a dense reward signal in RL, replacing GRPO's single terminal reward with per-token or per-step advantages. Theoretically appealing — dense reward is the classic fix for sparse-reward RL.

Practically, training-time use is where PRMs have disappointed, and §16.8 covers why.

## 16.6 Spending compute at inference

Three methods, in increasing order of sophistication and *decreasing* order of how reliably they help.

**1. Majority voting (self-consistency).** Sample $N$ solutions, take the most common final answer [\[72\]](../appendix/b-references.md#72-self-consistency). No verifier needed at all.

```python
from collections import Counter

def majority_vote(model, problem, n=64):
    answers = [extract_answer(model.sample(problem)) for _ in range(n)]
    answers = [a for a in answers if a is not None]
    return Counter(answers).most_common(1)[0][0] if answers else None
```

Absurdly simple and remarkably strong. It works because errors are diverse while correct reasoning converges — there are many ways to be wrong and usually one way to be right. It requires a discrete, comparable answer, so it does not apply to open-ended generation.

**2. Best-of-$N$ with a verifier or PRM.** Sample $N$, score each, take the highest. With an ORM the score is for the whole solution; with a PRM it is typically the minimum or product of step scores — the *minimum* is the better aggregator, because one bad step should condemn the solution regardless of how good the others are.

Better than majority voting when the verifier is good, and worse when it is not. And per §12.7, best-of-$N$ curves are also the diagnostic that reveals a verifier's exploitable artifacts: if quality falls at large $N$, you have found them.

**3. Guided search — beam search or MCTS over steps.** Use the PRM to prune during generation: expand promising prefixes, discard bad ones, never finish a solution that went wrong at step 2.

More compute-efficient in principle, since you stop wasting generation on doomed branches. Substantially more complex, and it requires the PRM to be accurate on *partial* solutions, which is harder than being accurate on complete ones.

**How they scale.** Snell et al. [\[74\]](../appendix/b-references.md#74-test-time-compute) studied this systematically, and the two findings that matter:

- **Test-time compute has a scaling curve.** Accuracy improves predictably with inference compute, in the same way it improves with training compute. This is what makes reasoning models commercially distinctive.
- **The optimal strategy depends on problem difficulty.** On easy problems, sampling a few and voting is best — the model mostly gets them right and you just need to filter noise. On hard problems, guided search wins, because the model needs help finding *any* correct route. A fixed strategy leaves a lot on the table; adapting it to estimated difficulty does better than either.

And the finding people remember: for some budgets, **spending compute at inference beats spending it on a larger model.** A smaller model with search can beat a larger model decoding greedily. This is the economic argument for the entire reasoning-model architecture.

## 16.7 What R1 did instead, and why

Here is the part of this chapter that matters most, because it cuts against everything above.

DeepSeek-R1 [\[10\]](../appendix/b-references.md#10-deepseek-r1) — the model that made reasoning work at scale — **tried process reward models and abandoned them.** The paper has a section on unsuccessful attempts, and PRMs are in it, alongside MCTS.

Their stated reasons, which are worth taking seriously because they come from a team that got the rest of it right:

**Defining a step is hard.** In competition math with numbered derivations, steps are natural. In general reasoning — a trace that goes "hmm, let me think about this differently" — there is no clean decomposition. The PRM's basic unit is ill-defined outside narrow domains.

**Determining whether an intermediate step is correct is hard.** Even with human annotators, agreement on "is this step right" is lower than on "is this answer right." Automated labelling (§16.4) inherits the policy's limitations. Both routes are noisy.

**A learned PRM gets hacked.** This is the decisive one. A PRM is a *learned* reward model, and Chapter 12 §12.6 applies to it in full: optimize hard against it and the policy will find inputs it scores highly and a human would not. R1's whole advantage was a reward that could not be hacked (§15.4). Putting a neural network back in the reward path gives that up.

**The retraining burden.** As the policy improves, the PRM goes stale (§12.7), so it needs periodic retraining — which means redoing the §16.4 rollouts.

They also tried MCTS and found that the branching factor in language generation — every token is a possible action — makes the search space unmanageable compared to games like Go where MCTS succeeded.

**The uncomfortable conclusion:** the most successful reasoning model of the era got there with a sparse, one-bit, outcome-only reward, having explicitly evaluated and rejected the dense-reward approach that theory favours.

**How to reconcile this with §16.3's positive results?** The most consistent reading:

- **PRMs are genuinely useful at inference time**, where they only need to *rank* candidates rather than serve as an optimization target. Ranking a fixed set is a much weaker demand than being an objective a policy optimizes against for thousands of steps. There is no adversary at inference time.
- **PRMs are dangerous at training time**, for exactly the §12.6 reason. The policy *is* an adversary, and it is an efficient one.
- **The sparse reward turned out to be sufficient**, which nobody expected. §15.5's surprise: one bit at the end of two thousand tokens was enough.

That distinction — safe as a ranker, dangerous as an objective — is the practical takeaway, and it generalizes beyond PRMs to every learned scorer in this book.

## 16.8 Verifiers versus PRMs

Worth keeping crisp, because the terms get used loosely.

| | Verifier (rule-based) | ORM (learned) | PRM (learned) |
|---|---|---|---|
| Scores | Final answer | Final answer | Each step |
| Ground truth | Yes, by construction | Learned from labels | Learned from labels |
| Hackable | No | Yes | Yes, and more surface area |
| Cost to build | Low, where applicable | Medium | High |
| Cost to run | Near zero | One forward pass | One forward pass |
| Works on | Math, code, structured output | Anything with labels | Anything with step labels |
| Safe as an RL objective | Yes | With a KL budget | Risky |
| Safe as an inference ranker | Yes | Yes | Yes |

**Use a rule-based verifier wherever one exists.** It dominates on every axis. This is §12.9's advice and it does not change.

**Use a PRM at inference**, for best-of-$N$ or guided search on hard problems, where its ranking ability is what you need.

**Be cautious using a PRM as an RL objective.** If you do, keep a KL budget and measure with something independent, exactly as in Chapter 13.

## 16.9 Generative verifiers

A development that partially resolves the tension. Instead of a scalar head predicting correctness, use a language model that *writes out* a critique and then judges [\[75\]](../appendix/b-references.md#75-generative-verifiers).

```
Problem: [...]
Proposed solution: [...]

Critique: Step 1 correctly applies the chain rule. Step 2 differentiates
the inner function but drops a factor of 2 from the exponent. This makes
step 3 and everything after it incorrect.

Verdict: INCORRECT (first error: step 2)
```

Read the probability of the verdict token as the score.

Advantages over a scalar PRM:

- **It reasons before judging**, which improves accuracy for the same reason chain-of-thought improves anything.
- **It uses the full pre-trained model** rather than a thin head on top.
- **It can be trained with the same verifiable rewards** as the policy — you know whether the solution was actually correct, so you know whether the verdict was right. This is a rare case of a learned scorer with a ground-truth training signal.
- **It produces an explanation**, which makes its failures diagnosable. A scalar that says 0.3 tells you nothing; a critique that says "step 2 drops a factor of 2" is checkable.
- **It localizes the error**, giving you PRM-like step information from an ORM-like training setup.

The cost is inference: writing 200 tokens of critique per candidate is far more expensive than a forward pass, which limits it to inference-time use with modest $N$ — the same trade-off as generative reward models in §12.8.

This is an active area, and it is the most likely route to process-level signal that does not carry §16.7's hacking risk.

## 16.10 What this means for your problem

A decision procedure:

**Do you have a rule-based verifier?** Use it. Do not build a PRM. (§12.9)

**Do you need better accuracy from a fixed model?** Spend inference compute. Try majority voting first — it is three lines and often most of the benefit. Then best-of-$N$ with a verifier. Then guided search, if the problems are hard enough to justify it.

**Do you want to improve the model itself?** Outcome-supervised RL with a verifier (Chapter 15) is the higher-expected-value path. Consider a PRM only if you have evidence that correct-by-luck solutions are a significant fraction of your positive rewards — which you can measure directly, by sampling correct solutions and checking their steps.

**Do you have step-level supervision available cheaply?** Unusual, but it happens: compiler errors localize to a line, proof assistants localize to a tactic, test failures localize to a case. In those domains you have a *rule-based* PRM, which has none of §16.7's problems. This is the best case and it is underexploited.

## 16.11 JD, decoded

**Reasoning Researcher.** Owns the outcome-versus-process decision and the inference-time strategy. The valuable judgement is knowing when a PRM is worth its cost, which per §16.7 is less often than the literature suggests.

**Verifier Engineer.** Building the programs that score outputs: sandboxed execution, symbolic comparison, proof checking. Increasingly a named role. The hard part is robustness against a policy actively probing for edge cases.

**Inference / Serving Engineer.** Best-of-$N$ at $N = 64$ is 64× the generation cost, and guided search has an irregular, branching request pattern that ordinary continuous batching handles poorly. Making search-time compute affordable is a serving problem — Chapter 22.

**Evaluation Engineer.** Test-time compute breaks the usual reporting convention: "the model scores X on MATH" is meaningless without stating the inference budget. Building evaluation that reports accuracy *as a function of compute* is now part of the job.

## 16.12 What you should take from this chapter

1. **Outcome supervision rewards correct-by-luck solutions.** A trace with two cancelling errors gets full reward, and every token in it gets reinforced.
2. **PRMs score each step**, and beat outcome supervision on MATH — with the gap widening as you sample more, because correct-by-luck grows as a fraction of what ORMs accept.
3. **Step labels can be generated automatically:** a step is good if you can finish correctly from it. Monte Carlo rollouts convert final-answer verification into step labels.
4. **That automatic labelling is expensive and biased.** Steps × rollouts generations per problem, often more compute than the RL run it supports, and it measures difficulty-for-the-current-policy rather than correctness.
5. **Majority voting is three lines and is often most of the benefit.** Try it before anything more complex.
6. **Test-time compute has a scaling curve**, and the optimal strategy depends on problem difficulty — voting on easy problems, guided search on hard ones.
7. **R1 tried PRMs and MCTS and abandoned both.** Ill-defined steps, noisy labels, and — decisively — a learned PRM is hackable, which gives up the exact property that made verifiable rewards work.
8. **The reconciliation: safe as a ranker, dangerous as an objective.** At inference there is no adversary; during RL the policy is one. This generalizes to every learned scorer in this book.
9. **Generative verifiers may resolve the tension** — they reason before judging, can be trained against ground truth, and localize errors. The cost is inference-time only.
10. **Where step-level ground truth exists — compilers, proof assistants, test harnesses — you have a rule-based PRM** with none of the hacking risk. This is the best case and it is underexploited.

The next chapter turns from capability to values: Constitutional AI, RLAIF, and deliberative alignment — where the criteria the model is optimized against are written down in English rather than implied by annotator behaviour.

---

**Exercises:** [Chapter 16 problem set](../../exercises/ch16.md) — includes the PRM labelling cost calculation, a best-of-$N$ scaling analysis, and a build-or-skip decision problem.
**Lab:** [`lab16_best_of_n_and_prm`](../../labs/lab16_best_of_n_and_prm.py) — compare majority voting, best-of-$N$ and PRM-guided selection, find the $N$ at which a flawed verifier starts to hurt, and discover that the best PRM aggregator depends on how noisy the PRM is: `min` wins when step scores are clean and is among the worst when they are not.

---

**References for this chapter**

- [\[10\] DeepSeek-R1](../appendix/b-references.md#10-deepseek-r1) — DeepSeek-AI, January 2025. The unsuccessful-attempts section on PRMs and MCTS.
- [\[26\] Let's Verify Step by Step](../appendix/b-references.md#26-process-reward-models) — Lightman et al., 2023. PRM800K and the process-beats-outcome result.
- [\[71\] Training Verifiers to Solve Math Word Problems](../appendix/b-references.md#71-gsm8k-verifiers) — Cobbe et al., October 2021. GSM8K, and the original sample-many-and-verify result.
- [\[72\] Self-Consistency](../appendix/b-references.md#72-self-consistency) — Wang et al., March 2022. Majority voting over sampled reasoning paths.
- [\[73\] Math-Shepherd](../appendix/b-references.md#73-math-shepherd) — Wang et al., 2024. Automatic step labels from Monte Carlo rollouts.
- [\[74\] Scaling LLM Test-Time Compute](../appendix/b-references.md#74-test-time-compute) — Snell et al., August 2024. Test-time compute scaling and difficulty-dependent strategy selection.
- [\[75\] Generative Verifiers](../appendix/b-references.md#75-generative-verifiers) — Zhang et al., August 2024. Critique-then-judge verification.
- [See full reference list](../appendix/b-references.md)
