# Chapter 14: DPO and the offline preference family

> Reading time: ~35 minutes. By the end of this chapter you should be able to derive the DPO loss from the RLHF objective, explain why the partition function cancels, understand the counterintuitive fact that DPO usually *decreases* the probability of the preferred response, and know when DPO is the right choice versus PPO or GRPO. You should be able to look at the alphabet soup — IPO, KTO, ORPO, SimPO, CPO — and see which knob each one is turning.

*Current as of early 2025.*

## 14.1 Your language model is secretly a reward model

Chapter 13 built an elaborate machine: a reward model, a value model, a reference model, a policy, a generation cluster, a KL controller, and a set of diagnostics for the six ways it destabilizes.

DPO [\[14\]](../appendix/b-references.md#14-dpo) asks whether all of that is necessary, and the answer turns out to be no — at least not always.

The insight is that the RLHF objective has a *closed-form* optimal policy. You do not need to search for it with RL; you can write it down. And once you write it down, you can invert the relationship: instead of learning a reward model and then finding the policy that maximizes it, you can express the reward in terms of the policy and optimize the policy directly against the preference data.

No reward model. No value model. No rollouts. No KL controller. Just a classification loss on pairs, trained exactly like SFT.

That is a large claim and it is substantially true. It is also more qualified than the initial enthusiasm suggested, and the qualifications are what this chapter is actually about.

## 14.2 The derivation

Worth working through, because the two places where something is given up are both visible in the algebra.

**Step 1 — the optimal policy has a closed form.**

The KL-constrained objective from §13.2:

$$\max_{\pi} \; \mathbb{E}_{x, y \sim \pi}\big[ r(x,y) \big] - \beta \, \mathbb{D}_{\mathrm{KL}}\big[\pi(y|x) \,\|\, \pi_{\mathrm{ref}}(y|x)\big]$$

This is a standard constrained-optimization problem with a known solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\mathrm{ref}}(y|x) \, \exp\!\left(\frac{1}{\beta} r(x,y)\right)$$

where $Z(x) = \sum_y \pi_{\mathrm{ref}}(y|x) \exp(r(x,y)/\beta)$ is the partition function.

The intuition: the optimal policy is the reference policy *reweighted* by the exponentiated reward. High-reward responses get their probability multiplied up, low-reward responses down, and $\beta$ controls how aggressively.

This solution is exact and it is also useless computationally — $Z(x)$ sums over every possible response, which is an infinite set.

**Step 2 — invert it.**

Take logs and rearrange for $r$:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)$$

This says something remarkable: **any policy implicitly defines a reward function.** The reward is the log-ratio of the policy to the reference, scaled by $\beta$, plus a term that depends only on the prompt.

**Step 3 — substitute into Bradley-Terry, and watch $Z$ vanish.**

Recall (§12.2) that Bradley-Terry only depends on reward *differences*:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

Substituting the expression for $r$:

$$r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} - \beta \log Z(x)$$

**The $\beta \log Z(x)$ terms cancel.** Both responses share the same prompt, so they share the same partition function. The intractable term disappears — not by approximation, but exactly.

**Step 4 — the loss.**

$$\mathcal{L}_{\mathrm{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) \right]$$

Compare to the reward-model loss from §12.2:

$$\mathcal{L}_{\mathrm{RM}} = -\,\mathbb{E}\Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]$$

It is the same loss. DPO replaces the reward model's scalar output with the policy's log-ratio against the reference. **The language model is the reward model**, which is the paper's title claim and it is literal rather than metaphorical.

## 14.3 The loss in code

```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,     # (B,) sum of log p over chosen tokens
    policy_rejected_logps: torch.Tensor,   # (B,)
    ref_chosen_logps: torch.Tensor,        # (B,) frozen reference, no grad
    ref_rejected_logps: torch.Tensor,      # (B,)
    beta: float = 0.1,
):
    """Direct Preference Optimization.

    Each *_logps is the SUM of token log-probabilities over the response only --
    prompt tokens must be masked out exactly as in SFT (Chapter 11 section 11.4).
    Getting that mask wrong is the most common DPO bug and it is silent.
    """
    # How far the policy has moved from the reference, per response.
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps

    # The implicit reward margin.
    logits = beta * (chosen_logratio - rejected_logratio)

    loss = -F.logsigmoid(logits).mean()

    # Diagnostics. These are the numbers that tell you what is happening;
    # the loss alone does not.
    chosen_reward = beta * chosen_logratio.detach()
    rejected_reward = beta * rejected_logratio.detach()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    margin = (chosen_reward - rejected_reward).mean()

    return loss, {
        "chosen_reward": chosen_reward.mean(),
        "rejected_reward": rejected_reward.mean(),
        "accuracy": accuracy,
        "margin": margin,
    }
```

Two implementation notes that account for most DPO bugs in the wild:

**Sum, do not average, the token log-probabilities.** The derivation is in terms of $\log \pi(y|x)$, the log-probability of the complete sequence, which is the *sum* over tokens. Averaging instead normalizes by length, which changes the objective — that variant is SimPO (§14.7), and it behaves differently. Both are defensible; silently doing one while believing you did the other is not.

**You do not need to keep the reference model in memory.** The reference log-probabilities are constant — the reference model is frozen. Precompute them once over your preference dataset and store them alongside the pairs. This removes a whole model from the training job, which for a 70B policy is 140 GB. Nearly every production DPO implementation does this and it is worth doing on day one.

## 14.4 What $\beta$ does here

In PPO, $\beta$ priced KL divergence and you adjusted it to hold a target (§13.6). In DPO it is baked into the loss, and it plays a different role.

Look at the gradient. Differentiating the loss with respect to the implicit margin gives a factor of

$$\sigma\!\left( -\beta \big(\text{margin}\big) \right)$$

which is a weight on each example. When the policy already gets a pair right by a large margin, this factor is near zero and the example contributes almost nothing. When the policy gets a pair wrong, the factor is near one and the example contributes fully.

So **$\beta$ controls how quickly examples stop mattering once they are learned.**

- **Small $\beta$ (0.01):** margins must grow large before an example is "done." The policy is pushed far from the reference. More movement, more risk of degradation.
- **Large $\beta$ (0.5):** examples saturate quickly. The policy stays close to the reference. Safe, and possibly too conservative to change anything.

Typical values are 0.1 to 0.5, with 0.1 the most common default.

Unlike PPO's $\beta$, this one **cannot be adapted to hold a KL target**, because there is no rollout from which to measure the actual KL. You are choosing a fixed price for drift and finding out afterwards how much drift you bought. This is a real disadvantage and it is not usually stated.

## 14.5 The data problem

Here is where DPO's simplicity has a cost, and it is the most important limitation in the chapter.

DPO trains on a fixed dataset of preference pairs. Those pairs were generated by *some* model — call it $\pi_{\mathrm{gen}}$. The derivation assumes the preference data is drawn from the policy being optimized, but in offline DPO it is not: the policy changes with every step, while the data does not.

Concretely, three regimes:

**On-policy data** (pairs sampled from the current SFT model). The derivation's assumptions approximately hold. DPO works well.

**Off-policy data from a related model** (pairs from your previous generation, or from a public preference dataset like UltraFeedback). Common in practice, and it works — this is what Zephyr [\[68\]](../appendix/b-references.md#68-zephyr) demonstrated, and it is why DPO became the default for the open community. But the further the generating distribution is from your policy, the more you are optimizing a preference model of *someone else's* output distribution.

**Off-policy data from a much stronger model** (pairs where both responses came from GPT-4). This is the trap. DPO will push your model's probability up on the chosen GPT-4 response and down on the rejected one — but your model would never have produced either. You are not optimizing a preference; you are doing a strange, poorly-conditioned form of distillation.

Compare with PPO, where every sample is generated by the current policy by construction. **The generation cost that made PPO expensive (§13.9) is exactly what keeps it on-policy.** DPO's saving and DPO's weakness are the same fact.

This is why iterative DPO (§14.8) exists, and why the gap between DPO and PPO in the literature narrows substantially once the DPO data is made on-policy.

## 14.6 What DPO actually does to the probabilities

An empirical result that surprises nearly everyone and is worth internalizing, because it explains several downstream pathologies.

**During DPO training, the log-probability of the chosen response usually goes *down*.**

Not up. Down. The rejected response's log-probability goes down faster, so the *margin* increases and the loss decreases exactly as intended — but the model is becoming less likely to produce the response you told it was good.

Why this happens: the DPO loss constrains only the *difference*. Nothing in it requires the chosen response's absolute probability to increase. And it is easier for gradient descent to push down on the rejected response — a specific, concentrated target — than to push up on the chosen one, because probability mass has to come from somewhere. The easiest place to move mass is *away from both responses*, toward the rest of the distribution, as long as it leaves the rejected response faster.

The consequences:

- **Probability mass leaks to unrelated outputs.** Both responses become less likely; the model becomes more likely to produce something that was in neither. In the worst case this is degenerate text.
- **It explains DPO's characteristic verbosity failure.** If mass leaks toward longer, more generic continuations, responses get longer without getting better.
- **It explains why DPO can degrade a model while its accuracy metric improves.** The DPO "accuracy" diagnostic (fraction of pairs where the implicit reward is ordered correctly) can go to 95% while the model itself gets worse.

**What to monitor**, and this is not optional:

- `chosen_reward` and `rejected_reward` *separately*, not just the margin. If both are strongly negative and diverging, mass is leaking.
- The absolute log-probability of the chosen responses over training.
- Actual generations, at every checkpoint. Same rule as §13.7.

Several of the variants in §14.7 exist specifically to address this.

## 14.7 The family

DPO spawned a large literature. Rather than cataloguing it, here is what each variant actually changes.

**IPO** — *Identity Preference Optimization* [\[64\]](../appendix/b-references.md#64-ipo). Azar et al. observed that DPO's use of the Bradley-Terry sigmoid means that when a preference is deterministic (the annotator always prefers $y_w$), the optimal margin is *infinite* — nothing bounds how far the policy pushes, so it overfits and the KL constraint stops binding. IPO replaces the logistic loss with a squared loss around a target margin, which is bounded by construction.

$$\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}\left[\left( \log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} - \frac{1}{2\beta} \right)^2\right]$$

*What it fixes:* overfitting to deterministic preferences. *What it costs:* the target margin is another hyperparameter, and the squared loss keeps pushing on examples that are already correct.

**KTO** — *Kahneman-Tversky Optimization* [\[65\]](../appendix/b-references.md#65-kto). Drops the requirement for *pairs* entirely. Instead of (chosen, rejected) for the same prompt, KTO takes individually-labelled responses — this one was good, that one was bad — and uses a utility function inspired by prospect theory, with loss aversion built in (bad examples weigh more than good ones).

*What it fixes:* the data requirement. Pairwise preference data is expensive; thumbs-up/thumbs-down data is abundant, and every production system already collects it. *What it costs:* a weaker signal per example, and a sensitivity to the ratio of positive to negative examples in your data.

This is the most practically important variant for anyone with a shipped product, because the data it needs is data you already have.

**ORPO** — *Odds Ratio Preference Optimization* [\[66\]](../appendix/b-references.md#66-orpo). Removes the reference model altogether and folds preference optimization into the SFT loss:

$$\mathcal{L}_{\mathrm{ORPO}} = \mathcal{L}_{\mathrm{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\mathrm{OR}}(y_w, y_l)$$

where the odds-ratio term penalizes the rejected response relative to the chosen one. One stage instead of two: no separate SFT run, no reference model, no precomputed reference log-probabilities.

*What it fixes:* pipeline complexity and memory. *What it costs:* the KL constraint's anchoring is gone — nothing holds the model near a known-good reference — and the SFT term on $y_w$ is what keeps the chosen response's probability from collapsing (§14.6). It trades a principled constraint for an empirical one.

**SimPO** — *Simple Preference Optimization* [\[67\]](../appendix/b-references.md#67-simpo). Two changes: drop the reference model, and use **length-normalized** average log-probability instead of the sum, plus a target margin $\gamma$:

$$\mathcal{L}_{\mathrm{SimPO}} = -\log \sigma\left( \frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma \right)$$

*What it fixes:* length bias, directly and by construction. Under summed log-probabilities, longer sequences have systematically lower total log-probability, which interacts badly with preference data where the chosen response is usually longer. Normalizing removes the interaction. It also aligns the training objective with the *generation* objective, since decoding effectively works with average per-token likelihood.

*What it costs:* the reference model's anchoring, again, and two hyperparameters instead of one.

**The pattern.** Every variant is turning one of four knobs:

| Knob | Turned by |
|---|---|
| Bounded vs unbounded margin | IPO (bounded), SimPO (target margin) |
| Pairs vs individual labels | KTO |
| Keep vs drop the reference model | ORPO, SimPO (drop) |
| Sum vs length-normalized log-probs | SimPO (normalize) |

The honest summary of the empirical literature: **on any given benchmark, some variant beats DPO, and it is rarely the same one twice.** The differences are real but smaller than the difference between good and bad preference data, and much smaller than the difference between on-policy and off-policy data (§14.5). Choose based on what your data looks like, not on a leaderboard.

## 14.8 Iterative DPO

The fix for §14.5, and the thing that closes most of the gap with PPO.

Instead of one pass over a fixed dataset:

1. Sample $k$ responses per prompt **from the current policy**.
2. Rank them — with a reward model, a judge, or a verifier.
3. Form preference pairs from the best and worst.
4. Run DPO for one round.
5. Go back to step 1 with the updated policy.

This is on-policy again. Each round's data comes from the model being trained, so the derivation's assumptions approximately hold, and the training distribution tracks the policy.

Llama 3 [\[5\]](../appendix/b-references.md#5-llama-3) used exactly this pattern — multiple rounds of rejection sampling and DPO — and it is now the standard open recipe.

The trade-off worth stating plainly: **iterative DPO reintroduces the generation cost that made PPO expensive.** You are back to sampling $k$ responses per prompt per round. What you keep is the simplicity — no value network, no clipping, no KL controller, no rollout buffer, no four-model memory footprint. What you give up is PPO's continuous, fine-grained policy updates in exchange for a few discrete rounds.

For most teams that is an excellent trade. It is also why the "DPO is free, PPO is expensive" framing is misleading: the good version of DPO is not free.

## 14.9 DPO versus PPO, honestly

The comparison has been contentious. Here is the state of the evidence.

| | DPO | PPO |
|---|---|---|
| Models in memory | 1 (2 if reference not precomputed) | 4 |
| Generation in the loop | No (offline) / Yes (iterative) | Yes, always |
| Wall-clock per round | Fast — it is SFT | Slow — generation is 60–80% (§13.9) |
| Hyperparameters | $\beta$ | $\beta$, LR, $\epsilon$, $\lambda$, epochs, batch, value LR |
| Stability | High | Low |
| Explicit KL control | No — fixed price, unknown drift | Yes — targetable |
| Uses non-verifiable rewards | Yes | Yes |
| Uses verifiable rewards | Awkwardly | Naturally |
| Can exceed the data distribution | No | Yes |
| Implementation risk | Low | High |

**Where PPO genuinely wins:**

- **On-policy by construction.** No §14.5 problem.
- **Explicit KL targeting.** You can decide how far to move and hold it.
- **Verifiable rewards.** A unit-test pass/fail is a reward, not a preference. To use it in DPO you must synthesize pairs from it, which discards information — the reward's *magnitude*, and the case where all $k$ samples fail.
- **It can find responses the model rarely produces.** DPO can only reweight what is in the data.

**Where DPO genuinely wins:**

- **Everything operational.** One model, one loss, no generation cluster, no weight sync, no rollout buffer.
- **You can actually run it.** A team of two can run DPO on a 70B model. Running PPO on a 70B model is a project.
- **It does not reward-hack in the §12.6 sense**, because there is no reward model to hack — though it has its own pathology in §14.6, which is arguably worse for being less well known.

**What the literature says.** Careful comparisons find PPO ahead when both are well-tuned and the data is on-policy, particularly on reasoning and code. But the gap is much smaller than the effort difference, and it nearly vanishes for DPO when the data is on-policy (§14.8). Meanwhile many published DPO-beats-PPO results are comparing well-tuned DPO to under-tuned PPO, which given PPO's seven hyperparameters is easy to do accidentally.

## 14.10 Choosing

A decision procedure that will not embarrass you:

**Use DPO if:** you have preference pairs and limited engineering capacity; you want a safe improvement over SFT; your task is general helpfulness, tone, or style; you cannot staff a generation cluster.

**Use iterative DPO if:** you have DPO working and want most of PPO's benefit for a fraction of its complexity. This is the right default for most teams, and it is what the strongest open models use.

**Use PPO if:** you have the infrastructure and the people; you need explicit KL control; you are pushing hard enough that overoptimization is a live concern; you have well-tuned everything else and are chasing the last few points.

**Use GRPO (Chapter 15) if:** your reward is *verifiable*. This is the case where the whole analysis changes — with a rule-based reward there is nothing to hack, so the KL machinery matters far less, and GRPO's simplicity dominates. If you are training reasoning on math or code, do not use DPO.

**Use rejection sampling (§11.10) if:** you want 70% of the benefit for 10% of the complexity, and you have not yet exhausted it. Many teams reach for DPO before they have finished extracting the value from rejection sampling, which is a mistake of sequencing.

## 14.11 JD, decoded

**Post-training Researcher.** Owns the choice among these methods and the preference data feeding them. The distinguishing skill is knowing that the data question (§14.5) dominates the algorithm question, and being able to say why iterative beats offline without reciting a benchmark.

**Post-training Engineer.** Implements the loss and, more importantly, the diagnostics from §14.6. A DPO implementation that logs only the loss will not tell you the model is degrading. This role also owns the reference-log-probability precomputation and the pair-construction pipeline.

**Data Operations.** Owns preference collection. Under KTO this changes shape substantially — thumbs-up/down from production is suddenly training data, which makes the feedback pipeline a first-class system.

A JD mentioning "DPO, IPO, KTO, ORPO" as a list is usually signalling breadth rather than depth. One mentioning "on-policy preference data" or "iterative DPO" was written by someone who has run this.

## 14.12 What you should take from this chapter

1. **DPO derives a closed-form optimal policy for the RLHF objective, then inverts it** so the policy expresses the reward. The partition function cancels exactly because Bradley-Terry only sees reward differences.
2. **The DPO loss is the reward-model loss with the policy's log-ratio substituted for the scalar reward.** Same loss, different parameterization.
3. **Sum the token log-probabilities, not the average**, unless you intend SimPO. Mask the prompt exactly as in SFT.
4. **Precompute the reference log-probabilities.** The reference model is frozen; keeping it in memory is 140 GB you do not need to spend.
5. **$\beta$ controls how fast examples saturate**, and unlike PPO's $\beta$ it cannot be adapted to hold a KL target. You buy an unknown amount of drift at a fixed price.
6. **Offline DPO's weakness is off-policy data**, and it is the same fact as its strength. PPO's expensive generation is what keeps PPO on-policy.
7. **DPO usually decreases the chosen response's log-probability.** Only the margin is constrained; mass leaks away from both responses. Monitor `chosen_reward` and `rejected_reward` separately, not just the margin.
8. **The variants each turn one knob:** bounded margin (IPO), no pairs needed (KTO), no reference model (ORPO, SimPO), length normalization (SimPO). Choose by what your data looks like.
9. **Iterative DPO closes most of the gap with PPO** by restoring on-policy data, at the cost of reintroducing generation. It is the right default for most teams.
10. **If your reward is verifiable, skip this chapter's methods.** Chapter 15 is where you should be.

The next chapter is GRPO and the reasoning revolution — where dropping the value network, dropping the learned reward model, and dropping most of the machinery in Chapters 12 through 14 produced the largest capability jump of the post-training era.

---

**Exercises:** [Chapter 14 problem set](../../exercises/ch14.md) — includes deriving the loss yourself, the decreasing-logprob diagnosis, and a method-selection design problem.
**Lab:** [`lab14_dpo_from_scratch`](../../labs/lab14_dpo_from_scratch.py) — implement the DPO loss and verify it against the definition and against an independently written implementation, reproduce the decreasing-chosen-logprob phenomenon, and check which of §14.3's three suspects actually change the objective. One of them turns out not to.

---

**References for this chapter**

- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — Meta AI, July 2024. Iterative rejection sampling plus DPO in production.
- [\[14\] DPO](../appendix/b-references.md#14-dpo) — Rafailov et al., May 2023. The derivation in §14.2.
- [\[64\] IPO](../appendix/b-references.md#64-ipo) — Azar et al., October 2023. Bounded margins for deterministic preferences.
- [\[65\] KTO](../appendix/b-references.md#65-kto) — Ethayarajh et al., February 2024. Unpaired preference optimization.
- [\[66\] ORPO](../appendix/b-references.md#66-orpo) — Hong et al., March 2024. Reference-free, single-stage.
- [\[67\] SimPO](../appendix/b-references.md#67-simpo) — Meng et al., May 2024. Length-normalized, reference-free.
- [\[68\] Zephyr](../appendix/b-references.md#68-zephyr) — Tunstall et al., October 2023. Distilled DPO on off-policy preference data; what made DPO the open-community default.
- [See full reference list](../appendix/b-references.md)
