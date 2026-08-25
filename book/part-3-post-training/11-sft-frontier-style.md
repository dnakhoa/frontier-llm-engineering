# Chapter 11: SFT, the way frontier labs actually do it

> Reading time: ~35 minutes. By the end of this chapter you should understand what supervised fine-tuning actually consists of at a frontier lab — where the data comes from, why loss masking and sequence packing are where most bugs live, and why SFT is simultaneously the least glamorous and highest-leverage stage of post-training. You should be able to read the SFT section of a technical report and know which of its unstated choices matter.

## 11.1 The stage everybody underestimates

Pre-training is where the compute goes. Reinforcement learning is where the papers go. Supervised fine-tuning is where the *model* comes from.

A base model out of pre-training is not usable. Give it "What is the capital of France?" and a well-trained base model will happily continue with "What is the capital of Germany? What is the capital of Italy?" — because it has correctly inferred that it is looking at a list of quiz questions. It has the knowledge. It has no idea that you wanted it to answer.

SFT is the stage that fixes this: train on (prompt, response) pairs with the loss computed only on the response, and the model learns the *shape* of being an assistant. Everything downstream — reward modeling, PPO, DPO, GRPO — operates on the SFT'd model and inherits its habits. A bad SFT stage cannot be recovered by good RL. This is one of the few things in post-training that essentially everyone agrees on.

The uncomfortable part: SFT is mostly a data problem wearing an engineering costume. The algorithm is next-token prediction, exactly as in pre-training. The learning rate is small. The dataset is tiny by pre-training standards — hundreds of thousands to a few million examples, which is to say a rounding error against 14 trillion tokens. And yet the gap between a good SFT dataset and a mediocre one is larger than the gap between most architectural choices.

This chapter covers what actually goes into that, including the three implementation details — loss masking, packing, and chat templates — that account for a wildly disproportionate share of post-training bugs.

## 11.2 Where the field landed: the honest case study

Frontier labs publish less about SFT than about almost anything else, and for an understandable reason: the SFT dataset *is* much of the product. DeepSeek-V3 [\[1\]](../appendix/b-references.md#1-deepseek-v3) describes its SFT set as roughly 1.5 million instances spanning multiple domains, and describes the *generation* process in useful detail while saying little about the exact composition. Qwen3 [\[6\]](../appendix/b-references.md#6-qwen3) is similar. OpenAI's InstructGPT paper [\[12\]](../appendix/b-references.md#12-instructgpt) — now old enough to be a historical document — remains one of the most detailed public accounts, and it describes a pipeline (human contractors writing demonstrations) that essentially no frontier lab uses in that form any more.

The most useful public artifact is Tülu 3 [\[53\]](../appendix/b-references.md#53-tulu-3), from AI2. It is not a frontier run, but it is a *fully open* post-training recipe — data, code, evaluations, and the ablations — at a scale where the conclusions transfer. When this chapter says "the standard recipe," Tülu 3 is usually the place you can go and read the actual code.

Two older results anchor the debate about how much data you need:

- **FLAN** [\[54\]](../appendix/b-references.md#54-flan) established that instruction-tuning on a large, diverse mixture of *existing* NLP tasks produces broad zero-shot instruction-following. Scale and task diversity both mattered.
- **LIMA** [\[52\]](../appendix/b-references.md#52-lima) argued the opposite extreme: 1,000 carefully curated examples were enough to produce a competent assistant from a strong base model, and the authors framed this as evidence that alignment is mostly about *surfacing* capabilities that pre-training already installed, not teaching new ones.

Both are partly right, and the resolution is the thing worth internalizing. LIMA is right that a strong base model needs very little data to learn the *format* of being an assistant. FLAN is right that breadth of *capability* coverage — code, math, multilingual, long-context, tool calls, refusals — needs scale. Frontier SFT sets are large not because format-learning needs a million examples, but because there are a hundred distinct behaviours to cover and each needs its own few thousand.

## 11.3 Where SFT data actually comes from

Four sources, in rough order of how much volume they contribute at a modern lab.

**1. Model-generated, filtered.** The dominant source since roughly 2023. You take a strong existing model — often your own previous generation, sometimes a larger internal model — generate candidate responses to prompts, and filter aggressively. Self-Instruct [\[51\]](../appendix/b-references.md#51-self-instruct) was the paper that made this respectable: bootstrap instructions and responses from the model itself, filter for quality and diversity, train on the survivors. Evol-Instruct [\[55\]](../appendix/b-references.md#55-evol-instruct) added the idea of *evolving* prompts to be harder along controlled axes (more constraints, more reasoning steps, more specificity), which turned out to matter a lot: the difficulty distribution of your prompts sets the ceiling of the resulting model.

The critical word is *filtered*. Unfiltered self-generated data degrades the model. The filters are the intellectual property.

**2. Rejection sampling from your own model.** A special case important enough to name separately. Generate $k$ responses to a prompt, score them (with a reward model, a verifier, or an execution check), keep the best one, train on it. This is covered in §11.10 and it is the bridge between SFT and RL — DeepSeek-V3 used exactly this pattern, generating SFT data for the V3 model using the R1 reasoning model as the generator [\[1\]](../appendix/b-references.md#1-deepseek-v3).

**3. Human-written demonstrations.** Still used, now targeted rather than bulk. Nobody pays contractors to write 50,000 general-purpose responses any more; the model writes better ones. Humans write the cases where the model is systematically wrong — hard refusals, sensitive-domain responses, specific tone requirements, and the seed examples that anchor a new capability.

**4. Converted existing datasets.** Academic benchmarks, code repositories with tests, math datasets with solutions, translated corpora — reformatted into instruction/response shape. Cheap, high-quality for narrow capabilities, and the reason your model can do named-entity recognition when nobody ever set out to teach it.

### The composition problem

A real SFT mixture is not a pile of examples; it is a budget allocation across capabilities. A rough shape of what a modern general-purpose mixture covers, with the caveat that the exact ratios are unpublished at every frontier lab:

```
General chat / instruction-following   ~25-35%
Code (generation, editing, explanation) ~15-25%
Math and quantitative reasoning         ~10-20%
Multilingual                            ~10-20%
Long-context (summarization, retrieval)  ~5-10%
Tool / function calling                  ~5-10%
Safety, refusals, and edge cases        ~3-8%
Formatting, structured output (JSON)     ~2-5%
```

Two things about this table. First, it is illustrative — assembled from what open recipes like Tülu 3 publish and what technical reports imply, not from any single lab's disclosure. Second, every one of those percentages is contested internally at every lab, continuously, because they trade off against each other. This is what a post-training data meeting is: people arguing about these numbers with evaluation results as ammunition.

## 11.4 Loss masking: the bug everybody ships once

Here is the single most common implementation error in post-training, and it is worth spending a section on because it is silent.

In SFT you have sequences that look like:

```
<|user|>What is the capital of France?<|assistant|>Paris is the capital of France.<|end|>
```

You want the model to learn to produce the assistant turn. You do **not** want it to learn to produce user turns — that is not its job, and training on user text teaches it to hallucinate the user's side of a conversation, which shows up later as the model writing both halves of a dialogue.

So the loss is computed only on the response tokens. The prompt tokens are still *fed in* — the model attends to them — but they contribute nothing to the gradient.

```python
import torch

IGNORE_INDEX = -100  # PyTorch's cross_entropy ignores this label by convention

def build_example(tokenizer, prompt: str, response: str, max_len: int):
    """Tokenize one (prompt, response) pair with the prompt masked out of the loss."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    eos = [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids + eos
    # Labels are the same sequence, with the prompt positions blanked out.
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids + eos

    return {
        "input_ids": input_ids[:max_len],
        "labels": labels[:max_len],
    }
```

Three ways this goes wrong in practice, all of which have shipped in public codebases:

**Off-by-one on the boundary.** The `<|assistant|>` marker itself: is it part of the prompt (masked) or the response (learned)? It must be masked — the marker is inserted by the serving harness at inference time, so if the model learns to generate it, it will emit a spurious `<|assistant|>` at the start of every reply. Get this wrong by one token and you get a model that either duplicates its role marker or, if you mask one token too many, never learns to produce its first real token confidently.

**Forgetting the EOS.** If the end-of-sequence token is not in the labels, the model never learns to stop. It will generate until it hits the max-token limit, every time. This is trivially diagnosable — your model rambles forever — and it is embarrassingly common.

**Masking the wrong turn in multi-turn data.** In a five-turn conversation, *all* assistant turns should be learned and *all* user turns masked. A naive implementation that treats "everything before the last assistant turn" as prompt throws away 80% of your training signal on multi-turn data. This one is silent: the model still works, it is just trained on a quarter of the data you paid for.

**How to catch all three:** decode your labels. Before any training run, take ten examples from the batch, replace `IGNORE_INDEX` with a visible marker, and print them.

```python
def show_masking(tokenizer, example):
    """Print an example with masked (non-learned) regions bracketed."""
    out = []
    for tok, label in zip(example["input_ids"], example["labels"]):
        piece = tokenizer.decode([tok])
        out.append(piece if label != IGNORE_INDEX else f"[{piece}]")
    print("".join(out))
```

Everything in brackets is context; everything outside is what the model is being taught to say. Ten seconds of looking at this output prevents a class of bug that otherwise costs a full training run. Do it every time the data format changes.

Lab `lab11_sft_packing` implements masking and packing from scratch and includes the failure cases as runnable examples.

## 11.5 Sequence packing and cross-contamination

SFT examples vary wildly in length: a one-line refusal is 20 tokens, a code-generation example with a long file is 8,000. If you pad every sequence to the maximum length, you waste most of your compute on padding tokens. At a 4,096-token context with a median example of 400 tokens, you are burning 90% of your FLOPs on nothing.

The fix is **packing**: concatenate multiple examples into one sequence up to the context length.

```
[example A][example B][example C][example D...]
|<------------ 4096 tokens, no padding ------------>|
```

This is a large efficiency win — commonly 3–8× on a realistic SFT mixture — and it introduces a subtle correctness problem.

**Cross-contamination.** With naive packing, the attention mask is the standard causal mask, so tokens in example B can attend to tokens in example A. The model learns from a context that will never occur at inference time, and worse, it learns spurious dependencies between unrelated examples. On a mixture where similar examples cluster (because your data loader did not shuffle well), this can be materially harmful.

There are three responses, in increasing order of correctness and cost:

**1. Ignore it.** Widely done, and defensible for large diverse mixtures with good shuffling: the model learns to treat the EOS token as a hard reset, and the contamination is noise. Most published SFT recipes do this, usually without mentioning it.

**2. Reset position IDs at document boundaries.** Cheap and helps: each packed example restarts its positional encoding at 0, so at least the *positions* are in-distribution.

**3. Block-diagonal attention masks.** The correct fix. Each example attends only within itself. FlashAttention's variable-length mode [\[19\]](../appendix/b-references.md#19-flashattention) implements exactly this efficiently — you pass cumulative sequence-length offsets and it computes the attention per segment without materializing the mask. This is what the varlen API is *for*, and it costs essentially nothing over naive packing.

```python
# Conceptual shape of the varlen interface. `cu_seqlens` holds the cumulative
# boundaries of each packed example, so attention never crosses them.
#
#   packed:     [A A A A][B B][C C C C C C]
#   cu_seqlens: [0,       4,   6,          12]
#
# flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen=...)
```

If you are packing and not using block-diagonal masks, you should at minimum know that you made that choice. The number of codebases that pack accidentally, with no masking and no position reset, is high.

## 11.6 Chat templates: the boring thing that breaks everything

A chat template is the function that turns a structured conversation into a token sequence. It defines the role markers, where the system prompt goes, how tool calls are represented, and what the generation prompt looks like.

It is the most boring component in post-training and it causes an outsized share of production incidents, for one reason: **the template used at training must exactly match the template used at inference**. Not "basically match." Exactly — same tokens, same whitespace, same newlines.

A model trained with `<|assistant|>\n` and served with `<|assistant|> ` (space instead of newline) is being asked to generate from a context it has never seen. The failure mode is not a crash; it is a model that is measurably but inexplicably worse. Quality drops by a few points on every benchmark and nobody can find the cause because the diff is one whitespace character in a Jinja template.

Practical rules:

- **Ship the template with the weights.** The `chat_template` field in the tokenizer config exists for this. A model artifact without its template is incomplete.
- **Use special tokens for role markers, not plain text.** If `<|user|>` is a single token in the vocabulary, no user input can forge it. If it is the plain string `"User:"`, a user can type `"User:"` and inject a turn. This is the tokenizer-level defence against a whole class of prompt injection, and it is why every serious chat model has dedicated role tokens.
- **Decide what happens with no system prompt.** Empty system block, omitted entirely, or a default string? Whatever you choose, do it identically in training and serving. Models are sensitive to this in ways that surprise people.
- **Test round-trip.** Render a conversation with your template, tokenize, decode, and diff against the original render. It should be byte-identical.

## 11.7 Hyperparameters

SFT hyperparameters are boring and mostly settled. The published values cluster tightly.

**Learning rate: 1e-5 to 2e-5** for full fine-tuning of a model in the 7B–70B range, with cosine decay to roughly 10% of peak and a short warmup (3% of steps, or a few dozen steps). This is one to two orders of magnitude below pre-training peak LR, and the reason is straightforward: you are adjusting a model that already works, not building one. Too high and you get catastrophic forgetting — the model becomes a good assistant and a worse reasoner.

**Epochs: 2 to 3.** SFT overfits fast because the dataset is small. Three epochs is the common upper bound; beyond that, benchmark scores keep improving while actual output quality degrades, which is one of the more treacherous dynamics in post-training. The model memorizes response *patterns* — it learns that answers start with "Certainly!" and contain exactly three bullet points — and this looks like improvement on automatic metrics.

**Batch size: 64 to 512 sequences**, usually expressed in tokens (0.5M–4M tokens per batch after packing). Larger than you might expect, because the gradient signal from a small SFT set is noisy.

**Sequence length: whatever the model supports**, with the long-context examples in the mixture actually exercising it. If your model has 128K context and your SFT set has nothing over 4K, you have quietly untrained the long-context capability that Chapter 9 paid for.

**Weight decay: 0 to 0.1**, and it genuinely does not matter much at this scale of update.

The one hyperparameter that *does* need tuning is the **data mixture**, which is not a hyperparameter in the config file. That is the real search space, and it is why post-training teams run dozens of SFT runs — not to tune the learning rate, but to test mixture ablations.

## 11.8 Full fine-tuning versus LoRA

LoRA [\[22\]](../appendix/b-references.md#22-lora) freezes the base weights and trains low-rank adapters, cutting memory dramatically. QLoRA [\[28\]](../appendix/b-references.md#28-nf4--qlora) adds 4-bit quantization of the frozen base, making 70B fine-tuning fit on a single 48 GB card.

For the open-source community this is transformative. For frontier labs it is mostly irrelevant, and it is worth being clear about why, because the internet's enthusiasm for LoRA does not reflect frontier practice.

**Frontier labs do full fine-tuning for the main post-training path.** They have the GPUs. The whole point of SFT is to move the model meaningfully, and low-rank updates constrain how much it can move. When you are producing the flagship model, "cheaper" is not a consideration that outweighs "better."

Where parameter-efficient methods *do* appear at frontier labs:

- **Ablations and mixture search.** If you are testing 40 data mixtures, running them as LoRA fine-tunes is much cheaper and the *relative* ordering usually transfers, even if absolute quality does not.
- **Per-customer or per-domain specialization** on top of a shipped model, where you need hundreds of variants and cannot store hundreds of full checkpoints.
- **Fast experimentation** by researchers who do not have a cluster allocation this week.

The trap to avoid: reading LoRA-based results from the open community and assuming they transfer to full fine-tuning. They often do not. LoRA's implicit regularization changes what mixtures look good — a mixture that a rank-16 adapter cannot overfit may be one a full fine-tune memorizes immediately.

## 11.9 The capability tax

Every SFT mixture is a set of trade-offs, and the trade-offs are real rather than merely theoretical. Some of the well-documented ones:

**Safety versus helpfulness.** Adding refusal data reduces harmful outputs and increases over-refusal — the model starts declining benign requests that superficially resemble the refused ones. "How do I kill a Python process?" is the canonical example. The fix is not less safety data; it is *paired* data that includes benign near-misses with helpful answers.

**Formatting versus substance.** Train heavily on well-formatted responses and the model learns formatting as a proxy for quality. You get beautifully structured, confidently wrong answers with excellent bullet points. Human raters reward this, which means your reward model will too (Chapter 12 §12.6).

**Reasoning length versus latency.** Training on long chain-of-thought responses makes the model reason better and makes every response slower and more expensive. This is a product decision disguised as a data decision, and Chapter 15 shows what happens when you let RL push it without a constraint.

**English versus multilingual.** Capacity is finite. Adding multilingual SFT data measurably costs English performance at fixed model size. Qwen3 [\[6\]](../appendix/b-references.md#6-qwen3) is notably explicit about the multilingual investment; the trade-off is inherent, not a failure of technique.

The general shape: SFT is capacity allocation. The model has a fixed budget of "behaviours it can be good at," and the mixture decides how it is spent. This framing is more useful than thinking of SFT as "teaching the model things."

## 11.10 Rejection sampling: SFT that is secretly RL

The single most important SFT technique of the current era, and the bridge to Part III's second half.

The procedure:

1. Take a prompt.
2. Sample $k$ responses from the current model (typically $k = 4$ to $64$).
3. Score each response — with a reward model, a verifier, unit tests, or a math answer-checker.
4. Keep the best one (or the top few).
5. Add it to the SFT set. Retrain. Repeat.

This is variously called rejection sampling fine-tuning, best-of-$n$ distillation, RAFT, or STaR depending on the details. Llama-3 [\[5\]](../appendix/b-references.md#5-llama-3) describes rejection sampling as a core part of its post-training loop, iterated over multiple rounds.

Why it works so well:

- **The data is on-policy.** The responses come from the model's own distribution, so it is not being asked to imitate a style it cannot naturally produce. This is the fundamental advantage over distilling from a different model.
- **It converts a scorer into a teacher.** If you have a verifier — unit tests for code, an answer key for math — you can generate unlimited correct training data without a single human label. This is the engine behind the reasoning models.
- **It is stable.** No KL penalty to tune, no value function, no reward hacking through the RL loop. It is just SFT on filtered data, with all of SFT's operational simplicity.

And where it stops:

- **It only reinforces what the model already does sometimes.** If the model never produces a correct solution in $k$ samples, rejection sampling has nothing to keep. Genuinely novel capability does not come from here.
- **It is expensive at inference.** Generating 32 samples per prompt across 100,000 prompts is 3.2 million generations. This is why Chapter 22 (serving) is on the post-training learning path.
- **It inherits the scorer's flaws.** If your reward model prefers verbose answers, rejection sampling will make your model verbose, faithfully and rapidly.

The relationship to RL is worth stating precisely. Rejection sampling fine-tuning is approximately a single, very conservative policy-gradient step: sample from the policy, weight by reward (in the crudest way — keep the best, discard the rest), and update. PPO and GRPO do the same thing with a proper gradient estimator, more sample efficiency, and considerably more machinery to go wrong. Rejection sampling gets a large fraction of the benefit for a small fraction of the complexity, which is why nearly every lab does it before, and alongside, actual RL.

DeepSeek-V3's SFT set was constructed substantially this way, using DeepSeek-R1 to generate reasoning-heavy responses that were then filtered and used to train V3 [\[1\]](../appendix/b-references.md#1-deepseek-v3). The reasoning model taught the general model, through SFT data. That pattern — a specialist generating training data for a generalist — is now standard.

## 11.11 Evaluating an SFT model

The core problem: SFT quality is not well captured by any automatic metric, and the metrics that exist are actively misleading.

**Validation loss is nearly useless.** It measures how well the model predicts *held-out responses from your own data distribution*. A model that has memorized your annotators' verbal tics scores well. Loss going down after epoch 2 while quality goes down is routine.

**Benchmark suites measure capability, not assistant quality.** MMLU, GSM8K, HumanEval measure whether the knowledge survived SFT — a real and important question, since aggressive SFT can damage them. But they say nothing about whether the model is pleasant, appropriately concise, or good at following a format instruction.

**Head-to-head comparison is the actual metric.** Generate responses from two models on the same prompts and ask which is better. With human raters this is expensive and slow; with a strong model as judge it is cheap and biased in known ways (toward length, toward its own style, toward the first-presented option). Chapter 23 covers the methodology and the failure modes in depth.

The practical evaluation stack for an SFT run, in the order you would look at it:

1. **Did capability survive?** MMLU, GSM8K, HumanEval versus the base model. A drop of more than a point or two means the LR or epoch count is too high.
2. **Did the format take?** Does it stop at EOS? Does it follow explicit format instructions? Does it emit spurious role markers? These are checkable programmatically and they catch the §11.4 and §11.6 bugs.
3. **Is it better than the previous model?** Pairwise, on a fixed prompt set that covers your capability mixture.
4. **Did safety behaviour change in either direction?** Both harmful compliance and over-refusal, measured separately, because they move in opposite directions.

## 11.12 What actually goes wrong

A field guide to SFT failures, roughly in order of how often they occur.

| Symptom | Usual cause | Where to look |
|---|---|---|
| Model never stops generating | EOS not in labels | §11.4 |
| Model emits `<\|assistant\|>` in its output | Role marker not masked | §11.4 |
| Model writes both sides of the conversation | User turns not masked | §11.4 |
| Quality mysteriously worse in production than in eval | Template mismatch between training and serving | §11.6 |
| Benchmarks improved, humans say it got worse | Overfitting to format; too many epochs | §11.7, §11.11 |
| Model refuses benign requests | Unpaired safety data | §11.9 |
| Base capabilities dropped several points | LR too high, or mixture lacks capability-preserving data | §11.7 |
| Long-context capability vanished | No long examples in the SFT mixture | §11.7 |
| Training is 5× slower than expected | No packing | §11.5 |
| Model has strange cross-example associations | Packing without block-diagonal masks | §11.5 |

The pattern worth noticing: almost none of these are algorithmic. They are data-plumbing bugs and mixture decisions. This is what people mean when they say post-training is a data discipline.

## 11.13 JD, decoded

SFT work sits with three roles, and the boundaries between them are blurrier here than anywhere else in the book.

**Post-training Researcher / Research Engineer.** Owns the mixture. Decides what capabilities go in, in what proportion, runs the ablations, and reads the evaluation results. When a JD says "experience with instruction tuning and data curation at scale," this is the job. The daily work is much closer to experimental design and data analysis than to modelling.

**Data Engineer (post-training flavour).** Owns the pipeline that produces SFT data: the generation infrastructure for model-generated data, the filtering, the deduplication, the annotation tooling, the quality scoring. At a frontier lab this is a substantial engineering organization, and it is chronically underestimated by candidates who think post-training is about the RL algorithm.

**Infrastructure / Training Engineer.** Owns the training loop itself: packing, masking, distributed training for the fine-tuning job, checkpoint management, and the generation infrastructure that rejection sampling depends on. The generation side is increasingly the harder half — a rejection-sampling round is an enormous batch inference job, which is why serving expertise (Chapter 22) shows up in post-training JDs.

If a job description mentions "chat templates," "data mixtures," or "rejection sampling" explicitly, it was written by someone who has done this work, and it is a good sign. If it describes post-training purely in terms of RLHF algorithms, the team may be earlier in its journey than it realizes.

## 11.14 What you should take from this chapter

1. **SFT turns a text predictor into an assistant, and everything downstream inherits its habits.** A bad SFT stage cannot be repaired by good RL.
2. **The algorithm is trivial; the data is everything.** Next-token prediction with a small learning rate. All the difficulty is in what you train on and in what proportion.
3. **Loss masking is where the bugs live.** Mask the prompt, mask user turns, keep the EOS, and decode your labels before every run. Three lines of debugging output prevent a class of expensive silent failures.
4. **Packing is a 3–8× efficiency win with a correctness footgun.** Use block-diagonal attention masks — FlashAttention's varlen mode does this for free — or at least know that you chose not to.
5. **The chat template must match exactly between training and serving.** One whitespace character is enough to degrade every benchmark by a few points with no visible cause.
6. **2–3 epochs, LR around 1e-5, and stop.** SFT overfits fast, and it overfits into *format memorization* that automatic metrics reward.
7. **Frontier labs full fine-tune; LoRA is for ablations and specialization.** Do not assume LoRA-based results from the open community transfer to full fine-tuning.
8. **Every mixture decision is a capacity trade-off.** Safety costs helpfulness, multilingual costs English, formatting costs substance. Budget deliberately instead of discovering it in evaluation.
9. **Rejection sampling is the highest-leverage technique in modern SFT.** Sample $k$, score, keep the best, retrain. It is approximately a very conservative RL step with none of RL's operational risk, and it is how verifiable domains generate unlimited training data.
10. **Validation loss does not measure what you care about.** Pairwise comparison against the previous model, plus capability-preservation checks, is the real evaluation.

The next chapter is reward modeling — how you build the scorer that rejection sampling and every RL method depend on, and why it is the component that most often quietly determines whether the whole post-training pipeline works.

---

**Exercises:** [Chapter 11 problem set](../../exercises/ch11.md) — includes the masking-bug diagnosis drill and an SFT mixture design problem.
**Lab:** [`lab11_sft_packing`](../../labs/lab11_sft_packing.py) — implement masking and packing from scratch, reproduce all three masking bugs, and measure the packing speedup.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — DeepSeek-AI, December 2024. Source for the 1.5M-instance SFT set and the R1-generates-data-for-V3 pattern.
- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — Meta AI, July 2024. Source for iterated rejection sampling in the post-training loop.
- [\[6\] Qwen3 Technical Report](../appendix/b-references.md#6-qwen3) — Qwen Team, 2025. Source for the multilingual post-training investment.
- [\[12\] InstructGPT](../appendix/b-references.md#12-instructgpt) — Ouyang et al., March 2022. The original detailed public account of the SFT-then-RLHF pipeline.
- [\[19\] FlashAttention](../appendix/b-references.md#19-flashattention) — Dao et al., 2022. Source for the variable-length mode used for block-diagonal packing.
- [\[22\] LoRA](../appendix/b-references.md#22-lora) — Hu et al., 2021. Low-rank adaptation.
- [\[28\] QLoRA](../appendix/b-references.md#28-nf4--qlora) — Dettmers et al., 2023. 4-bit quantized base with LoRA adapters.
- [\[51\] Self-Instruct](../appendix/b-references.md#51-self-instruct) — Wang et al., December 2022. Bootstrapping instruction data from the model itself.
- [\[52\] LIMA](../appendix/b-references.md#52-lima) — Zhou et al., May 2023. The 1,000-example alignment argument.
- [\[53\] Tülu 3](../appendix/b-references.md#53-tulu-3) — Lambert et al., November 2024. The fully open post-training recipe.
- [\[54\] FLAN / Scaling Instruction-Finetuned Language Models](../appendix/b-references.md#54-flan) — Chung et al., October 2022. Instruction tuning at task-mixture scale.
- [\[55\] WizardLM / Evol-Instruct](../appendix/b-references.md#55-evol-instruct) — Xu et al., April 2023. Evolving prompts to control difficulty.
- [See full reference list](../appendix/b-references.md)
