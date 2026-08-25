# Chapter 23: Evaluation — capability, contamination, safety

> Reading time: ~40 minutes. By the end of this chapter you should understand why evaluation is the real bottleneck in modern model development, how much of a reported benchmark score is prompt-formatting artifact, how contamination is detected and why it is nearly impossible to rule out, what LLM-as-judge measures and what it does not, and how to build an internal evaluation suite you can actually make decisions with.

## 23.1 The actual bottleneck

Every chapter in this book has described something you can do to a model. This chapter is about knowing whether it helped, and it is harder than any of them.

The asymmetry has grown steadily worse. Chapter 11 §11.2's exercise established that an SFT run costs about 0.017% of pre-training compute — you can run forty mixture ablations for a rounding error. Chapter 17 §17.6 established that AI feedback made preference labels roughly four orders of magnitude cheaper. Chapter 15 established that RL against a verifier can just be *run*.

Generating candidate models is now cheap. **Knowing which one is better is not.** Evaluation is what gates the loop, and at most labs it is the thing that determines how fast the model improves.

The uncomfortable version: a post-training team that can run 40 experiments a week and evaluate 3 of them properly is a team running 3 experiments a week.

## 23.2 Four kinds of evaluation

They answer different questions and get conflated constantly.

| Kind | Question | Method | Failure mode |
|---|---|---|---|
| **Capability** | Can it do the task? | Benchmarks with known answers | Contamination, prompt sensitivity |
| **Preference** | Do people like it better? | Pairwise comparison, human or model | Length bias, style over substance |
| **Safety** | Does it refuse what it should, and only that? | Adversarial probes, red teaming | Coverage; measures the red team |
| **Regression** | Did we break something? | Fixed suite, run every checkpoint | Only catches what it covers |

The mistake that costs the most is using capability benchmarks to answer preference questions. MMLU tells you whether SFT damaged the model's knowledge (§11.11's first check). It tells you nothing about whether the model is pleasant to use, and a team optimizing MMLU while users complain is a team measuring the wrong thing well.

## 23.3 Static benchmarks and what they are worth

**MMLU** [\[92\]](../appendix/b-references.md#92-mmlu) — 57 subjects of multiple-choice knowledge questions. Became the default headline number, which is more than it can carry: it is multiple choice, so it measures recognition rather than generation; it is saturated at the top; and it has been in every crawl for years (§23.6).

**GPQA** [\[96\]](../appendix/b-references.md#96-gpqa) — graduate-level science questions written to be "Google-proof," where domain experts score well and skilled non-experts with web access do not. Built specifically to resist the saturation and contamination that ruined MMLU as a discriminator.

**GSM8K and MATH** — grade-school and competition math. Verifiable, which makes them useful for RL (§15.4) as well as evaluation. Heavily contaminated.

**HumanEval and MBPP** — code, scored by running tests. Small (164 problems for HumanEval), so the confidence interval on a reported score is wider than most reported differences.

**HELM** [\[93\]](../appendix/b-references.md#93-helm) — not a benchmark but a *methodology*: evaluate many models on many scenarios along many metrics, with the protocol fixed and published. Its contribution is standardization, which is what makes numbers comparable at all.

**What a static benchmark is genuinely good for:** regression detection. Run MMLU on every checkpoint; if it drops four points, something broke. This is a real and important use and it does not require the benchmark to be a good measure of intelligence.

**What it is not good for:** ranking frontier models against each other, once every one of them is above 85%.

## 23.4 Prompt sensitivity, and the reproducibility problem

The finding that should change how you read leaderboards.

**The same model on the same benchmark scores differently depending on formatting choices that are not part of the task.** Whether options are labelled `A)` or `(A)` or `1.`, whether there is a space after the colon, the order of few-shot examples, whether you score by generation or by comparing option log-probabilities — each moves the number, and together they can move it by several points.

Several points is larger than most differences reported between models.

Consequences:

**Cross-paper comparisons are usually invalid.** Model A's paper reports 78.3 and Model B's reports 76.1. If they used different harnesses, different few-shot counts, or different scoring methods, the comparison says nothing. This is extremely common and rarely disclosed.

**Report the harness.** "MMLU 5-shot, `lm-evaluation-harness` v0.4.2, log-probability scoring" is a reproducible claim. "MMLU: 78.3" is not.

**Evaluate every model in your comparison yourself, in your harness.** Never mix your own numbers with numbers copied from another paper's table. This is the single most common methodological error in the field and it is entirely avoidable.

**Some of the reported gains are harness tuning.** A team that tries several prompt formats and reports the best has measured their prompt search, not their model. The defence is fixing the protocol before you run.

## 23.5 Contamination

The benchmark is in the training data. Given a 15-trillion-token web crawl and benchmarks that have been public for years, the prior should be that it is, not that it is not.

### Detecting it

**N-gram overlap.** Check whether benchmark questions appear verbatim in the training corpus — typically 8- to 13-gram matching. Catches exact copies, misses paraphrases, translations, and reformattings. This is the standard check and it is a floor, not a ceiling.

**Perplexity comparison.** A model shows anomalously low perplexity on data it memorized. Compare perplexity on the benchmark against similar held-out text; a large gap is suggestive.

**Canary strings.** Some benchmarks embed a unique random string in their files. If the model can reproduce the canary, it saw the file. Clean signal, and it only works if the benchmark included one and nobody stripped it.

**Ordering sensitivity.** A contaminated model often shows a suspicious pattern: it does much better when multiple-choice options are in their original order than when shuffled. A memorized answer is tied to a position.

**Train/test perplexity gap on the benchmark's own splits.** If the model's perplexity on the test split is much lower than on the train split of the same dataset, something is wrong — the test split is the one more likely to have been copied into web pages.

```python
def ngram_contamination(benchmark_texts, corpus_index, n=13):
    """Fraction of benchmark items with an exact n-gram match in the corpus.

    A floor, not a ceiling: this catches verbatim copies and misses every
    paraphrase, translation, and reformatting. A 0% result means "no verbatim
    copies found", never "not contaminated".
    """
    hits = 0
    for text in benchmark_texts:
        tokens = text.split()
        grams = {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
        if any(g in corpus_index for g in grams):
            hits += 1
    return hits / len(benchmark_texts)
```

### Why you cannot rule it out

**Paraphrase is invisible to n-grams.** A tutorial explaining a GSM8K problem in different words teaches the model the problem without a single matching n-gram.

**Translations are contamination.** A benchmark discussed on a Chinese forum contaminates a multilingual model, and no English n-gram check will see it.

**Synthetic data launders it.** If you generate SFT data with a model that was itself trained on the benchmark, the contamination transfers with no textual overlap at all (§11.3). This is the mechanism that makes contamination essentially untraceable in a modern pipeline.

**The web absorbs benchmarks continuously.** Every benchmark gets discussed, tutorialized, and reproduced in blog posts and repositories. A benchmark released today is contaminating crawls within months.

### What to actually do

- **Held-out private sets.** The only real defence: evaluation data that has never been public. Expensive, and it is what every serious lab maintains internally.
- **Time-based splits.** Evaluate on material created after the training cutoff. Clean by construction, and it expires.
- **Freshly generated variants.** Same problem structure, new instances — new numbers in a math problem, new names in a reasoning problem. Cheap and effective for templated tasks.
- **Report the contamination check.** Llama 3 [\[5\]](../appendix/b-references.md#5-llama-3) and DeepSeek [\[1\]](../appendix/b-references.md#1-deepseek-v3) both discuss decontamination in their reports. Doing the check and reporting it is the professional standard, even though it cannot be conclusive.

## 23.6 LLM-as-judge

Use a strong model to compare two responses [\[95\]](../appendix/b-references.md#95-mt-bench). Cheap, fast, and correlates reasonably with human preference — reported agreement with human raters is in the neighbourhood of human-human agreement itself, which is 70–80% (§12.4).

That last point is the one that gets misread. **A judge agreeing with humans 80% of the time is at the ceiling, not below it.** The remaining 20% is largely genuine human disagreement, not judge error.

**The biases, all measurable and all worth correcting:**

- **Position bias.** Judges prefer the first-presented response. Always evaluate both orderings and average. Skipping this is the most common flaw in published judge-based evaluations.
- **Length bias.** Judges prefer longer responses, often more strongly than humans do. Report the length distribution of both models alongside the win rate; a win rate that moves with a length difference is not a quality result.
- **Self-preference.** Models rate their own outputs and stylistically similar outputs higher. Do not use your own model to judge itself against a competitor.
- **Style over substance.** Confident, well-formatted, wrong answers score well. This is §11.9's formatting-versus-substance tax appearing in the measurement layer, where it is more dangerous.

**Where judges are genuinely unreliable:** anything requiring expertise the judge lacks. A judge cannot tell you which of two proofs is correct if it cannot verify proofs. For those domains, use a verifier (§12.9) — and if no verifier exists, use experts and accept the cost.

## 23.7 Arena-style human preference

**Chatbot Arena** [\[94\]](../appendix/b-references.md#94-chatbot-arena) collects pairwise human preferences on real user prompts, anonymized and randomized, and fits a Bradley-Terry model (§12.2) to produce ratings.

What makes it valuable: real prompts from real users, at scale, with the models blinded. That combination is very hard to reproduce internally.

What it measures, precisely: **which response a user prefers on first read, without verification.** That is a real and important property. It is not accuracy, it is not reliability, and it is not usefulness over a long task. A model that is confidently wrong in an appealing way does well.

Known effects: length and formatting help; the prompt distribution skews toward what people bring to a free public demo; and once a rating is a public metric, optimizing for it becomes worthwhile, which changes what it measures (Goodhart, in the ordinary way).

Use it as one input. A model that is top-5 on Arena and fails your internal regression suite has a problem, and Arena will not tell you what it is.

## 23.8 Evaluating reasoning models

Chapter 15's models break the reporting conventions, and the field has not fully adjusted.

**A score without a compute budget is meaningless.** "86.7% on AIME" (§19.3) was with majority voting at 64 samples. Pass@1 was 71.0%. Both are true and they are different claims. Any reasoning-model score must state the inference budget.

**Report the curve, not the point.** Accuracy as a function of samples, or of token budget. That curve is the actual result (§16.6); a single number is a slice through it at an undisclosed position.

**Pass@k and pass^k are opposite metrics.** Pass@k — solved on *at least one* of $k$ attempts — measures capability and flatters the model. Pass^k — solved on *all* $k$ attempts — measures reliability and is brutal (§18.9). Deployment cares about the second. Papers report the first.

**Latency and cost belong in the table.** A model that reasons for 20,000 tokens to gain two points is a different product from one that answers immediately, and an accuracy-only comparison hides the trade.

## 23.9 Building a suite you can decide with

The practical core of the chapter. A working internal evaluation suite has five layers, and they run at different frequencies.

**1. Smoke tests — every checkpoint, seconds.** Does it generate? Does it stop at EOS? Does it follow a format instruction? Does it emit spurious role markers? These catch the Chapter 11 bugs and cost nothing.

**2. Regression suite — every checkpoint, minutes.** A fixed set of capability benchmarks compared against the *base model* and the previous release. Absolute values matter less than deltas. This is where you catch "SFT ate four points of MMLU."

**3. Capability evaluation — every candidate, hours.** The benchmarks you actually care about, plus held-out private sets, plus freshly generated variants. Run every model in the comparison yourself, in your harness (§23.4).

**4. Preference evaluation — every candidate, hours to days.** Pairwise against the previous model on a fixed prompt set that mirrors your capability mixture (§11.3). Judge-based for volume, human for the decisions that matter, both orderings, with length reported.

**5. Safety evaluation — every release.** Harmful compliance and over-refusal measured *separately*, because they move in opposite directions (§17.8). Plus red teaming, which is a process rather than a suite.

**The properties that make it useful:**

- **A fixed protocol, decided before the run.** Changing the harness between candidates invalidates the comparison.
- **Confidence intervals.** HumanEval has 164 problems; a 2-point difference is noise. Most reported deltas in the field are within noise and are reported as if they were not.
- **A held-out set nobody optimizes against.** The moment an evaluation is a target, it stops measuring. Keep one set that is never used for iteration.
- **Organizational separation.** §12.13 and §13.12 both made this point: the people building the evaluation should not be the people optimizing against it. This sounds bureaucratic and it is the main structural defence against overoptimization.

## 23.10 Red teaming

Adversarial evaluation: people, or models, deliberately trying to elicit failures.

**What it finds:** jailbreaks, prompt-injection paths (§18.8), capability failures on edge cases, and behaviours the training distribution never covered.

**What it cannot tell you:** that the model is safe. §17.9's result — that safety training modifies behaviour on the distribution it trained on — means a passed red-team evaluation is evidence about the red team's coverage. Adversarial training against a specific probe teaches the model to handle that probe.

**What makes it work in practice:**

- **Diversity of red-teamers** matters more than their number. Homogeneous teams find homogeneous failures.
- **Automated red teaming** — a model generating adversarial prompts — scales the volume and shares blind spots with the target (§17.8's correlated-errors problem). Use both.
- **Rotate.** A fixed red-team set becomes a training target and stops finding anything.
- **Multilingual and cross-cultural coverage.** Safety training done in English generalizes poorly, and this is one of the most consistent gaps in shipped models.

## 23.11 JD, decoded

**Evaluation Engineer / Research Scientist (evaluation).** Builds and runs the suite. Undervalued relative to its leverage: per §23.1, evaluation throughput gates experiment throughput, which gates model improvement. The distinguishing skill is methodological rigour — knowing that a 2-point difference on HumanEval is noise, and saying so when someone wants to ship on it.

**Data Engineer (evaluation).** Held-out sets, contamination detection, freshly generated variants, and keeping evaluation data out of training. §23.5 makes this a permanent adversarial job against your own pipeline.

**Red Team.** A permanent function, not a pre-launch gate. Often part-adversarial-ML, part-domain-expert, part-creative.

**Every other role in this book consumes evaluation.** A team without trustworthy evaluation is doing all the work in Chapters 3 through 22 blind, and it will not know it.

## 23.12 What you should take from this chapter

1. **Evaluation is the bottleneck.** Generating candidates is cheap now; knowing which is better is not, and evaluation throughput sets experiment throughput.
2. **Capability, preference, safety, and regression are four different questions.** Using MMLU to answer a preference question is the most common and most expensive conflation.
3. **Prompt formatting moves benchmark scores by several points** — more than most reported differences between models. Report your harness; evaluate every model in the comparison yourself.
4. **Never mix your numbers with numbers copied from another paper.** It is the field's most common methodological error and it is entirely avoidable.
5. **Assume contamination.** Given 15T tokens and years-old public benchmarks, the prior should be that the benchmark is in there.
6. **N-gram checks are a floor.** They miss paraphrase, translation, and — decisively — synthetic data generated by a contaminated model, which launders contamination with no textual overlap.
7. **Held-out private sets are the only real defence**, plus time-based splits and freshly generated variants.
8. **A judge agreeing with humans 80% of the time is at the ceiling.** Correct for position bias by evaluating both orderings, and always report the length distributions.
9. **Reasoning-model scores are meaningless without an inference budget.** Report the accuracy-versus-compute curve, and report pass^k when the claim is about reliability.
10. **Keep one evaluation set nobody optimizes against, and separate the people who build evaluation from the people optimizing against it.** This is the main structural defence against everything Chapter 12 warned about.

This closes Part IV. The next part is about the job: reading a frontier JD and knowing what it means, the paths through these roles, and where the field is heading.

---

**Exercises:** [Chapter 23 problem set](../../exercises/ch23.md) — includes a confidence-interval calculation that invalidates a real-looking result, a contamination audit design, and a judge-bias correction.
**Lab:** `lab23_contamination_check` — build n-gram and embedding-based contamination detectors, then demonstrate the paraphrase case that defeats both.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. Decontamination reporting.
- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — Meta AI, July 2024. Contamination analysis at scale.
- [\[92\] MMLU](../appendix/b-references.md#92-mmlu) — Hendrycks et al., 2020. The default headline benchmark, and its limits.
- [\[93\] HELM](../appendix/b-references.md#93-helm) — Liang et al., 2022. Evaluation as a standardized methodology.
- [\[94\] Chatbot Arena](../appendix/b-references.md#94-chatbot-arena) — Chiang et al., 2024. Blinded pairwise human preference at scale.
- [\[95\] MT-Bench / LLM-as-a-Judge](../appendix/b-references.md#95-mt-bench) — Zheng et al., 2023. Judge agreement rates and the catalogue of judge biases.
- [\[96\] GPQA](../appendix/b-references.md#96-gpqa) — Rein et al., 2023. Google-proof graduate-level questions.
- [See full reference list](../appendix/b-references.md)
