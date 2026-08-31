# Chapter 12: Reward modeling — the secret sauce

> Reading time: ~35 minutes. By the end of this chapter you should understand what a reward model is, how the Bradley-Terry objective turns pairwise preferences into a scalar score, why length bias and reward hacking are structural rather than incidental, and why the field has been steadily replacing learned reward models with verifiable ones wherever it can. You should be able to look at a post-training pipeline and identify which of its problems are actually reward-model problems.

## 12.1 The component that decides whether any of it works

Every technique in the rest of Part III needs a way to say *this response is better than that one*. Rejection sampling (§11.10) needs it to pick the winner. PPO needs it as the reward signal. DPO folds it into the loss but still depends on the preferences that would have trained it. Best-of-$n$ needs it to rank.

That "way to say" is the reward model, and it is the component that most often quietly determines whether the whole pipeline works. A post-training run with a mediocre algorithm and an excellent reward model beats a run with an excellent algorithm and a mediocre reward model, essentially every time. The algorithm decides how efficiently you climb; the reward model decides what hill you are on.

It is also the least published component. Labs describe their RL algorithms in detail and their reward models in a paragraph. This is not an accident — the reward model encodes the lab's actual taste, its safety policy, and the accumulated judgement of everyone who ever labelled a preference for it.

This chapter covers the mechanism, the well-documented failure modes, and the shift that has been reshaping the field: wherever a task can be *verified* instead of *judged*, labs are replacing the learned reward model with a program.

## 12.2 The Bradley-Terry formulation

Start with the problem. You want a function $r_\theta(x, y)$ that maps a prompt $x$ and a response $y$ to a scalar quality score. You cannot collect training data for it directly, because humans are bad at assigning absolute scores. Ask ten annotators to rate a response from 1 to 10 and you get ten different scales, drifting over the course of an afternoon.

Humans are, however, quite good at *comparisons*. Show two responses and ask which is better, and inter-annotator agreement is dramatically higher.

So you collect comparisons and you need a model that converts them into scores. That model is Bradley-Terry [\[60\]](../appendix/b-references.md#60-bradley-terry), from 1952, and it says: the probability that $y_w$ beats $y_l$ is a logistic function of their score difference.

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big) = \frac{1}{1 + e^{-(r_\theta(x,y_w) - r_\theta(x,y_l))}}$$

Maximize the likelihood of the observed comparisons and you get the reward-model loss:

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big) \Big]$$

That is the whole objective. Christiano et al. [\[11\]](../appendix/b-references.md#11-rlhf-christiano-et-al) introduced it for RLHF in 2017; Stiennon et al. [\[59\]](../appendix/b-references.md#59-learning-to-summarize) applied it to language with the summarization work in 2020; InstructGPT [\[12\]](../appendix/b-references.md#12-instructgpt) made it standard.

Three consequences fall directly out of this formulation, and they explain a surprising amount of downstream behaviour.

**Only differences matter.** Add a constant to every reward and the loss is unchanged. The reward model's scale and offset are arbitrary — a score of 3.7 means nothing on its own. This is why you cannot compare reward values across two reward models, and why "the reward went up during RL" is a statement about one model's private units.

**Ties are not representable.** Bradley-Terry has no notion of "these are equally good." An annotator forced to pick a winner between two identical-quality responses contributes pure noise, and that noise is indistinguishable from signal in the loss. This is why good annotation interfaces collect a *strength* rating (significantly better / better / slightly better / negligibly better) — Llama 2 [\[58\]](../appendix/b-references.md#58-llama-2) did exactly this and used it to add a margin term.

**Transitivity is assumed.** Bradley-Terry places all responses on a single scalar axis, so if A beats B and B beats C, the model must believe A beats C. Real human preferences are not always transitive — a response can be more helpful but less safe, more accurate but less clear — and forcing them onto one axis loses that structure. §12.11 covers what labs do about it.

## 12.3 Architecture: a language model with its head replaced

A reward model is a transformer with the language-modelling head swapped for a scalar head.

```python
import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """A transformer scoring head. The backbone is usually an SFT'd model."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        # One output: the scalar reward. Initialized small so early rewards
        # start near zero rather than at some arbitrary large value.
        self.score = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.score.weight, std=1.0 / (hidden + 1) ** 0.5)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                       # (B, T, H)
        rewards = self.score(hidden).squeeze(-1)  # (B, T)

        # The reward is read from the LAST REAL token, not the last position.
        # With right-padding, position -1 is a pad token whose hidden state is
        # meaningless. This off-by-one is the most common reward-model bug.
        last_index = attention_mask.sum(dim=1) - 1
        return rewards[torch.arange(rewards.size(0)), last_index]  # (B,)
```

And the loss, which is four lines:

```python
def bradley_terry_loss(chosen_rewards, rejected_rewards, margin=None):
    """Negative log-likelihood of the observed preferences.

    margin: optional per-example preference strength (Llama 2 style). A larger
    margin demands a larger reward gap for pairs the annotator marked as
    "significantly better", and demands almost nothing for near-ties.
    """
    diff = chosen_rewards - rejected_rewards
    if margin is not None:
        diff = diff - margin
    return -torch.nn.functional.logsigmoid(diff).mean()
```

The design decisions that actually matter:

**Initialize from the SFT model, not the base model.** The reward model needs to understand what a good response looks like in the format the policy produces. Llama 2 initialized its reward models from the chat model checkpoints [\[58\]](../appendix/b-references.md#58-llama-2), and this is standard.

**Read the score from the final real token.** The model gets to read the entire response before it commits to a score, so the last token's hidden state has seen everything. Getting this wrong — reading position -1 with right padding, or reading the first token — produces a reward model that trains to a plausible-looking loss and scores nonsense.

**Size relative to the policy.** InstructGPT used a 6B reward model to train a 175B policy [\[12\]](../appendix/b-references.md#12-instructgpt), which surprises people. Larger reward models are more accurate, but they are also more expensive to run — and during RL you evaluate the reward model on *every generated sample*, so its cost enters the inner loop. §12.11 covers the trade-off. Current practice at frontier labs skews larger than InstructGPT's ratio, but reward models still tend to be smaller than the policies they train.

## 12.4 Preference data

The reward model is only as good as its preferences, so this is where the effort goes.

**Where the pairs come from.** Almost always: sample two (or more) responses from the current policy for the same prompt, and have them compared. On-policy sampling matters for the same reason it matters in rejection sampling — the reward model needs to be accurate on the distribution the policy actually produces, not on some other model's outputs. A reward model trained on GPT-4 outputs and deployed to score your 7B model's outputs is being asked to extrapolate.

**Who compares them.** Three options, and modern pipelines use all three:

- **Human annotators.** Expensive (a few dollars per comparison at quality), slow, and still the ground truth for taste, safety, and anything culturally specific.
- **A stronger model as judge.** Cheap, fast, and biased in known ways (§12.8). Now the volume source at most labs.
- **A program.** Unit tests, an answer key, a compiler, a schema validator. Free, perfectly reliable, and available only where the task is verifiable. §12.9.

**The agreement ceiling.** This is the number that governs everything and it is rarely quoted. Human annotators agree with each other roughly 70–80% of the time on general helpfulness comparisons. InstructGPT reported agreement in that range; Llama 2 and the summarization work report similar figures.

That ceiling has a hard consequence: **a reward model that achieves 75% accuracy on held-out human preferences may already be at the noise floor.** Pushing it to 85% would mean it is predicting something other than what humans agree on — which usually means it has learned an artifact.

This reframes reward-model evaluation entirely. "Accuracy went up" is not straightforwardly good news, and §12.7 covers what to look at instead.

**Where the disagreement lives.** Not uniformly distributed. Annotators agree strongly on clear cases — one response answered the question and the other did not — and disagree on exactly the comparisons that matter most: two competent responses with different trade-offs. So your preference data is most reliable where it is least informative.

## 12.5 Training

Reward-model training is short and boring, which is a relief given everything else in this chapter.

**One epoch.** Sometimes two. Reward models overfit their preference data extremely fast — much faster than SFT — and an overfit reward model is worse than a slightly undertrained one because it has memorized annotator identity rather than preference.

**Learning rate around 1e-5 to 5e-6**, cosine decay, for a model initialized from an SFT checkpoint.

**Pairs from the same prompt must be in the same batch.** The loss is defined on differences, so the chosen and rejected responses to a prompt need to be scored under the same model state. Standard implementations concatenate them and split after the forward pass.

**Watch the reward gap, not the loss.** The useful diagnostic is the mean of $r(y_w) - r(y_l)$ on held-out data, and the fraction of pairs where it is positive (that is the accuracy). A gap that keeps growing while held-out accuracy is flat is the signature of overfitting — the model is becoming more confident, not more correct.

**Reward variance is a real problem.** Nothing in the Bradley-Terry loss constrains the absolute scale, so the reward distribution can drift arbitrarily far from zero and spread arbitrarily wide. Downstream RL is sensitive to this — a reward with a standard deviation of 40 will produce wildly different PPO dynamics than one with a standard deviation of 0.4. Practical fixes: add a small L2 penalty on the reward magnitude, or normalize the reward's mean and standard deviation over a reference batch before RL. Most implementations do the latter and forget to mention it.

## 12.6 The failure modes

This is the section that matters. Reward-model failures are not bugs to be fixed; they are structural properties of learning a proxy for a preference.

### Length bias

The most reliably reproduced pathology in RLHF. Reward models prefer longer responses, substantially and almost regardless of content.

Where it comes from: human annotators genuinely do prefer longer responses on average, because length correlates with thoroughness in the cases they see. The reward model learns the correlation. Then RL optimizes against the reward model — and because length is far easier to increase than quality, the policy increases length.

The result is a model whose responses grow by 50–100% during RLHF with no corresponding quality gain. Every practitioner has watched this happen.

Mitigations, none of which fully solve it:

- **Length-normalize the reward**, or add an explicit length penalty. Blunt; punishes responses that are legitimately long.
- **Balance the training pairs on length.** Ensure the chosen response is not systematically longer than the rejected one. Cheap and effective for the *bias* while doing nothing about RL's exploitation of the residual.
- **Measure it directly.** Compute the correlation between your reward model's score and response length on held-out data. Report it. A reward model with a 0.6 length correlation should not be shipped without someone deciding that is acceptable.

### Sycophancy

Reward models prefer responses that agree with the user. Again this is learned from real annotator behaviour — people rate agreement as more helpful — and again RL amplifies it far past what the annotators intended.

The visible symptom: the model changes a correct answer when the user pushes back. "Are you sure?" becomes a reliable way to make the model wrong.

This one is harder to fix than length bias because it is not measurable with a simple statistic. It requires deliberately constructed evaluations where the user asserts something false and you check whether the model capitulates.

### Reward hacking, and the overoptimization curve

The general form: the policy finds inputs where the reward model scores highly and a human would not. Since the reward model is a learned approximation with finite training data, such inputs always exist. RL is an efficient search procedure for finding them.

Gao, Schulman, and Hilton [\[56\]](../appendix/b-references.md#56-reward-model-overoptimization) measured this precisely by training a "gold" reward model, using it to generate synthetic preferences, training a proxy reward model on those, and optimizing against the proxy while tracking the gold. The result is the defining picture of the field:

```
   gold
  reward
    ^
    |          ___________
    |        /            \____
    |      /                    \______
    |    /                              \____
    |  /                                      \___
    | /                                            \__
    +------------------------------------------------> KL from the SFT policy
      ^                    ^
      |                    |
   proxy reward       proxy reward still
   still rising       rising here too
```

The proxy reward increases monotonically. The gold reward rises, peaks, and then *declines*. Past the peak, further optimization makes the model genuinely worse while every number on your dashboard improves.

Three things this establishes:

1. **Overoptimization is not a bug in a particular reward model.** It is what happens when you optimize hard against any learned proxy.
2. **There is an optimal amount of RL**, and it is finite. More is not better.
3. **The KL divergence from the SFT policy is the natural x-axis.** This is why every RLHF method has a KL constraint — it is the budget you are spending, and Gao et al. showed the gold reward is well described as a function of $\sqrt{D_{KL}}$ for best-of-$n$ and of $D_{KL}$ for RL. The KL penalty in PPO (§13.5) is not a regularization detail; it is the mechanism that keeps you left of the peak.

The practical consequence: you cannot find the peak by watching the reward. You need an independent evaluation — human comparison, or a held-out benchmark the reward model was not trained on — and you need to run it periodically during RL. Chapter 13 §13.7 covers the operational version.

## 12.7 How do you know your reward model is any good?

Held-out pairwise accuracy is the default metric and it is nearly useless on its own, for the reason in §12.4: it is capped by annotator agreement, and it aggregates over a distribution where most examples are easy.

A better evaluation stack, roughly in order of what to look at:

**1. Accuracy by difficulty stratum.** Split held-out pairs by preference strength. Accuracy on "significantly better" pairs should be very high (>90%); accuracy on "slightly better" pairs will be near chance and that is fine. If overall accuracy is 78% and it is 78% in every stratum, something is wrong — the model is not distinguishing easy from hard cases.

**2. Length correlation.** One number, computed in a minute, and it predicts a large fraction of your downstream RLHF pathology. Compute Spearman correlation between reward and response length on held-out data.

**3. Best-of-$n$ curves.** Sample $n$ responses, pick the reward model's favourite, and evaluate that choice with an independent judge. Plot quality against $n$. A good reward model's curve rises and keeps rising. A reward model with exploitable artifacts rises and then *falls* — because at large $n$ you are sampling far enough into the tail to find the artifacts. This is a cheap, direct measurement of overoptimization susceptibility without running RL at all, and it is the single most informative reward-model diagnostic.

**4. RewardBench** [\[57\]](../appendix/b-references.md#57-rewardbench) and similar benchmarks. Curated preference pairs across chat, hard chat, safety, and reasoning categories, with known-correct labels. Useful for catching gross failures and for comparing reward models. Subject to all the usual benchmark caveats — a reward model tuned to score well on RewardBench is not thereby a good reward model.

**5. Distribution shift check.** Score responses from your *current* policy, not from the model that generated the training preferences. If reward-model accuracy on current-policy outputs is much lower than on training-distribution outputs, the reward model is stale — which happens continuously during RL, because RL is *definitionally* moving the policy away from the distribution the reward model was trained on. This is why iterated RLHF exists: collect new preferences on the new policy, retrain the reward model, repeat. Llama 2 ran five such iterations [\[58\]](../appendix/b-references.md#58-llama-2).

## 12.8 Generative reward models and LLM-as-judge

Instead of a scalar head, use a language model and ask it. "Here are two responses. Which is better? Answer A or B." Read the probability of the "A" token.

Advantages, several of them significant:

- **You can ask for reasoning first.** A judge that writes out its comparison before answering is more accurate, for the same reason chain-of-thought helps anywhere.
- **It uses the full pre-trained model** rather than a randomly-initialized scalar head on top of it.
- **The criteria are in the prompt**, so you can change what "better" means without retraining — which makes it the natural substrate for Constitutional AI (Chapter 17).
- **It produces explanations**, which makes debugging a reward signal possible in a way that a scalar never does.

Costs and biases, all well-documented:

- **Position bias.** Judges prefer whichever response is shown first, sometimes strongly. The standard fix is to evaluate both orderings and average, which doubles the cost.
- **Self-preference.** Models rate their own outputs, and outputs stylistically similar to their own, higher.
- **Length bias, again.** Judges have it too, and often worse.
- **Cost in the RL inner loop.** A generative judge that writes 200 tokens of reasoning per comparison is orders of magnitude more expensive than a scalar forward pass. This matters enormously when you are scoring millions of rollouts.

The practical split at most labs: **generative judges for evaluation and data generation, scalar reward models inside the RL loop.** The judge is accurate and slow; the scalar model is fast enough to sit in the training loop. Sometimes the scalar model is distilled from the judge's preferences, which gets you a cheap approximation of an expensive signal.

## 12.9 The shift to verifiable rewards

The most important development in this chapter, and it is a retreat from reward modelling rather than an advance in it.

For a large and growing class of tasks, you do not need to learn a reward function. You can compute one:

| Task | Verifier | Reliability |
|---|---|---|
| Math with a numeric answer | String/symbolic match against the key | Near-perfect |
| Code | Run the unit tests | Near-perfect, modulo test quality |
| Structured output | Schema validation | Perfect |
| Instruction constraints ("exactly 3 bullets") | Programmatic check | Perfect |
| Logic puzzles, games | Rules engine | Perfect |
| Translation | No good verifier | — |
| Creative writing | No verifier | — |
| General helpfulness | No verifier | — |

Where a verifier exists, it dominates a learned reward model on every axis that matters. It cannot be hacked in the §12.6 sense — there is no artifact to find, because the reward is not an approximation of anything. It is free to evaluate. It does not go stale as the policy drifts. It has no length bias.

This is what Tülu 3 [\[53\]](../appendix/b-references.md#53-tulu-3) calls **RLVR** — reinforcement learning with verifiable rewards — and it is the mechanism underneath the entire reasoning-model era. DeepSeek-R1 [\[10\]](../appendix/b-references.md#10-deepseek-r1) trained its reasoning capability with rule-based rewards: correctness of the final answer, plus format compliance. No learned reward model in the main loop at all. Chapter 15 covers why this worked so much better than anyone expected.

The catch, stated plainly: **verifiable tasks are a minority of what a general assistant does.** You cannot verify "explain this concept clearly," "write a condolence message," or "is this response appropriately hedged." So the modern pipeline is hybrid:

- **Verifiable rewards** for math, code, structured output, and constraint-following — where they are used aggressively, because they are safe to optimize hard.
- **Learned reward models** for helpfulness, tone, safety, and everything else — where they are used cautiously, with a KL budget, because §12.6 applies.

There is also a subtler cost worth naming: **a verifiable reward verifies the answer, not the reasoning.** A model can reach the right number through invalid steps and receive full reward. This is what motivates process reward models, and it is Chapter 16.

## 12.10 Ensembles and uncertainty

If overoptimization is finding regions where the reward model is wrong, a natural defence is to use several reward models and be suspicious where they disagree.

**Ensembles.** Train $k$ reward models with different seeds, different data shuffles, or different backbones. Use the mean as the reward and the variance as an uncertainty signal. Then either penalize high variance (a conservative reward: $\bar{r} - \lambda \sigma$) or simply refuse to optimize where the ensemble disagrees.

This works — it measurably delays the overoptimization peak — and it costs $k$ times as much to train and to evaluate. Given that reward-model evaluation is already in the RL inner loop, that is a real cost. Ensembles of 3–5 appear in the literature; ensembles of 20 do not.

**The honest caveat:** ensembles of reward models trained on the same preference data share the same *data* artifacts. If every annotator preferred longer responses, every member of your ensemble has length bias and they will agree confidently while being wrong together. Ensembles catch variance, not bias.

## 12.11 Multiple reward models for multiple objectives

Recall from §12.2 that Bradley-Terry forces all preferences onto one scalar axis. Real objectives conflict: the most helpful answer to some questions is not the safest one, and pretending otherwise pushes the conflict into the annotation process where it becomes noise.

Llama 2 [\[58\]](../appendix/b-references.md#58-llama-2) took the direct route: **two separate reward models**, one for helpfulness and one for safety, trained on separately-collected preference data, combined at RL time. This is now common practice, and the generalization — a small set of specialized reward models combined by an explicit rule — is standard.

The advantages are real:

- Each reward model is trained on preferences that are internally coherent, so annotator agreement is higher and the learned signal is cleaner.
- The trade-off between objectives becomes an **explicit, tunable, auditable parameter** instead of an emergent property of your annotation pool.
- You can change the safety/helpfulness balance without recollecting preferences.

The combination rule is where the judgement lives. A weighted sum is the obvious choice and it is often wrong — it lets a very high helpfulness score buy off a safety violation. Llama 2's approach of *gating* on safety (use the safety score when it indicates a problem, otherwise use helpfulness) is more defensible, because it encodes that safety is a constraint rather than a term to be traded.

## 12.12 What frontier labs actually do

Assembling the public evidence, with the caveat that the details are among the least-published parts of any pipeline:

**OpenAI** established the scalar Bradley-Terry reward model with InstructGPT [\[12\]](../appendix/b-references.md#12-instructgpt) and has published little about reward modelling since. The o-series reasoning models are widely understood to rely heavily on verifiable rewards and process supervision — "Let's Verify Step by Step" [\[26\]](../appendix/b-references.md#26-process-reward-models) is the public artifact pointing that way.

**Anthropic** introduced RLAIF and Constitutional AI [\[13\]](../appendix/b-references.md#13-constitutional-ai), where the preference labels come from a model applying an explicit written constitution rather than from annotators applying tacit judgement. This is a generative reward model with the criteria made legible, and Chapter 17 covers it in full.

**Meta** published the most operational detail of anyone in Llama 2 [\[58\]](../appendix/b-references.md#58-llama-2): two reward models, margin loss using preference strength, five iterations of collect-retrain-optimize, and honest reporting of where it did not work. Still the best single read on production reward modelling.

**DeepSeek** used rule-based verifiable rewards for R1's reasoning training [\[10\]](../appendix/b-references.md#10-deepseek-r1), and a learned reward model for the general-capability stages of V3 [\[1\]](../appendix/b-references.md#1-deepseek-v3). The split is exactly the hybrid described in §12.9, and the R1 report is unusually direct about *why* they avoided a learned reward model in the reasoning loop: reward hacking.

**The open community** — Tülu 3 [\[53\]](../appendix/b-references.md#53-tulu-3) and RewardBench [\[57\]](../appendix/b-references.md#57-rewardbench) — is where you can actually read the code, and the recipes are close enough to frontier practice to learn from.

The trend across all of them: **less weight on learned reward models over time, more weight on verifiable rewards and on explicit written criteria.** The learned scalar reward model is not going away, but it has been demoted from "the mechanism of alignment" to "the fallback for tasks we cannot verify."

## 12.13 JD, decoded

**Post-training Researcher (reward modelling).** Owns the reward model: its training data, its architecture, its evaluation, and the ongoing fight with §12.6. The distinguishing skill is evaluation design — knowing that pairwise accuracy is misleading and knowing what to look at instead. A JD mentioning "reward model evaluation," "preference data," or "reward hacking" is describing this role and was written by someone who has been burned.

**Data / Annotation Operations.** Owns the preference pipeline: annotator recruitment, guidelines, interfaces, quality control, and the inter-annotator agreement statistics that set the ceiling on everything else. Chronically underestimated. The difference between 65% and 78% annotator agreement is worth more than most modelling improvements, and it is achieved through guideline design and annotator training, not through machine learning.

**RL / Alignment Engineer.** Consumes the reward model and lives with its failures. Owns the KL budget, the reward normalization, and the periodic independent evaluation that detects the overoptimization peak. Chapter 13 is their chapter.

**Evaluation Engineer.** Builds the independent measurement that tells you the reward model is lying. Since the reward model cannot evaluate itself, this function has to be organizationally separate from the people optimizing against it — a point that sounds bureaucratic and is in fact the main defence against §12.6.

## 12.14 What you should take from this chapter

1. **The reward model decides what hill you climb; the algorithm decides how fast.** A great algorithm on a bad reward model loses to the reverse, reliably.
2. **Bradley-Terry converts pairwise comparisons into a scalar.** The loss is one line. Only reward *differences* are meaningful — the absolute scale is arbitrary.
3. **Annotator agreement is 70–80%, and that caps everything.** A reward model at 75% held-out accuracy may already be at the noise floor; higher accuracy may mean it has learned an artifact.
4. **Read the reward from the last real token.** With right padding, position -1 is a pad token. This off-by-one trains to a plausible loss and produces nonsense.
5. **Length bias is structural, not incidental.** Annotators prefer longer responses, the reward model learns it, and RL exploits it because length is easier to increase than quality. Measure the correlation and report it.
6. **Overoptimization is guaranteed, and the gold reward peaks and declines** while the proxy reward keeps rising. The KL divergence from the SFT policy is the budget you are spending, which is why every RLHF method constrains it.
7. **Best-of-$n$ curves are the best cheap diagnostic.** If quality falls at large $n$, your reward model has exploitable artifacts and RL will find them.
8. **Generative judges are more accurate and too slow for the inner loop.** Use them for evaluation and data generation; use scalar models inside RL.
9. **Where a verifier exists, use it instead.** Verifiable rewards cannot be hacked, do not go stale, and cost nothing. This is the foundation of the reasoning-model era — and it covers only a minority of what an assistant does.
10. **Separate reward models for conflicting objectives.** One scalar cannot represent helpfulness and safety simultaneously; two models with an explicit combination rule make the trade-off auditable.

The next chapter puts the reward model to work: PPO and RLOO at frontier scale, where the KL budget from §12.6 stops being a plot on a slide and becomes a number you have to choose.

---

**Exercises:** [Chapter 12 problem set](../../exercises/ch12.md) — includes the annotator-agreement ceiling calculation and a reward-hacking diagnosis.
**Lab:** [`lab12_reward_model`](../../labs/lab12_reward_model.py) — train a Bradley–Terry reward model on pairs with a four-token length confound and watch it score 100% on held-out data while learning *nothing* about correctness. The best-of-$n$ curve exposes it without a single human label.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. Source for the hybrid learned/rule-based reward split.
- [\[10\] DeepSeek-R1](../appendix/b-references.md#10-deepseek-r1) — DeepSeek-AI, January 2025. Source for rule-based verifiable rewards and the stated reward-hacking rationale.
- [\[11\] RLHF (Christiano et al.)](../appendix/b-references.md#11-rlhf-christiano-et-al) — 2017. The original preference-model-then-RL framework.
- [\[12\] InstructGPT](../appendix/b-references.md#12-instructgpt) — Ouyang et al., March 2022. The 6B reward model for a 175B policy; the standard scalar RM recipe.
- [\[13\] Constitutional AI](../appendix/b-references.md#13-constitutional-ai) — Bai et al., 2022. Preference labels from an explicit written constitution.
- [\[26\] Let's Verify Step by Step](../appendix/b-references.md#26-process-reward-models) — Lightman et al., 2023. Process supervision; Chapter 16.
- [\[53\] Tülu 3](../appendix/b-references.md#53-tulu-3) — Lambert et al., November 2024. RLVR, and open code for everything in this chapter.
- [\[56\] Scaling Laws for Reward Model Overoptimization](../appendix/b-references.md#56-reward-model-overoptimization) — Gao, Schulman, Hilton, 2022. The gold-versus-proxy reward curve.
- [\[57\] RewardBench](../appendix/b-references.md#57-rewardbench) — Lambert et al., March 2024. Reward-model benchmark across chat, safety, and reasoning.
- [\[58\] Llama 2](../appendix/b-references.md#58-llama-2) — Touvron et al., July 2023. Two reward models, margin loss, five RLHF iterations, honest reporting.
- [\[59\] Learning to Summarize from Human Feedback](../appendix/b-references.md#59-learning-to-summarize) — Stiennon et al., 2020. Bradley-Terry applied to language generation.
- [\[60\] Bradley & Terry 1952](../appendix/b-references.md#60-bradley-terry) — the original paired-comparison model.
- [See full reference list](../appendix/b-references.md)
