# Labs

Runnable code for the mechanisms the chapters describe. Each lab implements the
real thing rather than a demonstration of it, and each is small enough to run on
a laptop CPU.

**All 19 labs the chapters name now exist**, and CI runs every one of them on
every commit.

## The labs

| Lab | Chapter | What you build | Runtime (CPU) |
|---|---|---|---|
| [`lab03_dedup_and_quality`](lab03_dedup_and_quality.py) | [3 — Data pipelines](../book/part-2-pretraining/03-data-pipelines.md) | MinHash and LSH banding from scratch against a corpus with a known duplicate structure. Exact-hash dedup recalls ~26% of it. Then a quality classifier that scores 100% held-out while its behaviour is decided entirely by what you swept into the negative class. | ~9 s |
| [`lab04_train_a_bpe_tokenizer`](lab04_train_a_bpe_tokenizer.py) | [4 — Tokenization](../book/part-2-pretraining/04-tokenization.md) | Byte-level BPE from scratch, then fertility measured across five languages on held-out text. Shows the ~1.8× penalty an English-only merge corpus imposes on Vietnamese, and what deleting the pre-tokenizer regex does to the learned merges. | ~4 s |
| [`lab05_attention_variants`](lab05_attention_variants.py) | [5 — Architecture](../book/part-2-pretraining/05-architecture.md) | MHA, GQA, MQA and MLA as one module with one knob. Verifies cached incremental decoding against a full forward pass, and checks Exercise 5.2's arithmetic against measured tensor bytes — they agree exactly. Section 5 is a negative result, on purpose. | ~2.4 min |
| [`lab05_moe_routing`](lab05_moe_routing.py) | [5 — Architecture](../book/part-2-pretraining/05-architecture.md) | Top-k router and a real MoE layer. Watch load imbalance reach a ~3× straggler tax, then fix it two ways — the standard auxiliary loss, and DeepSeek's auxiliary-loss-free bias controller — and measure what each costs the loss. | ~56 s |
| [`lab06_parallelism_memory_model`](lab06_parallelism_memory_model.py) | [6 — Distributed training](../book/part-2-pretraining/06-distributed-training.md) | A memory-and-throughput calculator for dense and MoE models under any TP/PP/EP/DP/ZeRO configuration. Searches the configuration space and tells you what fits and roughly how fast. Pure arithmetic, no GPU. | < 1 s |
| [`lab07_collective_bandwidth`](lab07_collective_bandwidth.py) | [7 — Cluster reality](../book/part-2-pretraining/07-cluster-reality.md) | A cost model for ring and hierarchical all-reduce, all-to-all under oversubscription, the straggler tax, and gradient bucketing. Shows the same job running 3.5× slower on Ethernet than on InfiniBand, decided by one interconnect decision. | < 1 s |
| [`lab08_scaling_laws`](lab08_scaling_laws.py) | [8 — Optimization](../book/part-2-pretraining/08-optimization.md) | Fit L(N) = E + A·N^-α against a source whose irreducible loss is *computable*, so the fitted asymptote can be marked rather than trusted. A clean 7-point fit still misplaces E by +0.47 nats, and the extrapolation is optimistic in a way that grows with reach. | ~32 s |
| [`lab08_precision_and_stability`](lab08_precision_and_stability.py) | [8 — Optimization](../book/part-2-pretraining/08-optimization.md) | Real `float8_e4m3fn` / `float8_e5m2` / BF16 / FP16 casts, not simulated rounding. FP16 flushes 24% of a realistic gradient distribution to zero; one outlier under per-tensor FP8 scaling zeroes 18% of a tensor; FP16 sequential accumulation reaches 80% error at 32K elements. | ~1 s |
| [`lab09_rope_extension`](lab09_rope_extension.py) | [9 — Mid-training](../book/part-2-pretraining/09-mid-training.md) | RoPE's wavelength table predicts what will break; then PI, NTK-aware and YaRN measured by position at 4× extension. Position Interpolation drives the *earliest* positions worse than chance, and a 120-step fine-tune recovers it completely. | ~43 s |
| [`lab11_sft_packing`](lab11_sft_packing.py) | [11 — SFT](../book/part-3-post-training/11-sft-frontier-style.md) | Loss masking and sequence packing from scratch. Reproduces all three classic masking bugs and shows what each does to a trained model, then measures packing cross-contamination — which drops to *exactly* zero under block-diagonal attention. | ~28 s |
| [`lab12_reward_model`](lab12_reward_model.py) | [12 — Reward modeling](../book/part-3-post-training/12-reward-modeling.md) | A Bradley–Terry reward model trained on pairs with a four-token length confound. Loss 0.0000, held-out accuracy 100%, and 48.5% on length-balanced pairs — chance is 50%. It learned nothing about correctness. The best-of-n curve exposes it with no labels. | ~24 s |
| [`lab13_ppo_minimal`](lab13_ppo_minimal.py) | [13 — PPO and RLOO](../book/part-3-post-training/13-ppo-rloo-frontier.md) | PPO and RLOO with the guardrails removed one at a time. The usual KL coefficient restrains nothing; gradient clipping can silently do the trust region's job; and one corrupted reward in 500 commands ~16% of the batch gradient, which advantage normalisation barely changes. | ~50 s |
| [`lab14_dpo_from_scratch`](lab14_dpo_from_scratch.py) | [14 — DPO](../book/part-3-post-training/14-dpo-family.md) | The DPO loss checked against the definition and an independent implementation. Both implicit rewards fall while the margin grows. Prompt masking turns out to be a *non*-bug (it cancels exactly); a trainable reference is a real one, and it reports a better loss. | ~28 s |
| [`lab15_grpo_countdown`](lab15_grpo_countdown.py) | [15 — GRPO](../book/part-3-post-training/15-grpo-reasoning.md) | Group-relative advantage, and reward shaping that destroys the task: a 0.05 per-token thinking bonus takes accuracy from 0.996 to exactly 0.000. Also measures the zero-advantage problem, and reports that the obvious fix for it makes things worse. | ~51 s |
| [`lab16_best_of_n_and_prm`](lab16_best_of_n_and_prm.py) | [16 — Process rewards](../book/part-3-post-training/16-process-reward-models.md) | Majority voting versus best-of-N versus PRM aggregators. Majority voting overtakes a noisy verifier at a measurable point; a length-biased verifier turns over; and the best aggregator depends on PRM calibration — `min` wins clean and loses noisy. | ~8 s |
| [`lab20_triton_fused_kernel`](lab20_triton_fused_kernel.py) | [20 — Custom kernels](../book/part-4-infra/20-custom-kernels.md) | Fusion's memory-traffic ceiling (3.50×), then the measurement that rewriting the chain by hand wins nothing while `torch.compile` reaches 2.07×. **Part of this lab needs a GPU** and says so up front; the Triton kernel is included in full. | ~4 s |
| [`lab21_checkpoint_resume`](lab21_checkpoint_resume.py) | [21 — Checkpointing](../book/part-4-infra/21-checkpointing-resumption.md) | Resume correctness defined as bit-exactness and tested. Every incomplete recipe diverges on the *first* step after the resume, by a few percent of the loss, with nothing to alert you. Also: why a physical shard layout cannot reshard 4 → 2. | ~1 s |
| [`lab22_kv_cache_and_batching`](lab22_kv_cache_and_batching.py) | [22 — Inference and serving](../book/part-4-infra/22-inference-serving.md) | Continuous batching is 3.3× static on a heavy-tailed request stream. A naive KV allocator runs at 86.9% fragmentation against 0.7% for 16-token paging. Prefix sharing across 8 samples of one prompt saves 4.3×. | ~2 s |
| [`lab23_contamination_check`](lab23_contamination_check.py) | [23 — Evaluation](../book/part-4-infra/23-evaluation.md) | A 13-gram detector gets 100% on verbatim copies, 0% on paraphrase, and 0% on verbatim copies *shorter than n* — which is most of MMLU. A fuzzy detector recovers the paraphrases; translation defeats both. A contamination percentage describes the detector, not the corpus. | < 1 s |

Runtimes are wall clock on a laptop CPU with no `FLE_SMOKE_TEST`. Under
`FLE_SMOKE_TEST=1` — which is what CI uses — every lab finishes in seconds.

## One lab needs a GPU, and says so

[`lab20_triton_fused_kernel`](lab20_triton_fused_kernel.py) is the exception to
"CPU is a first-class target". Triton compiles to GPU code, so the lab opens with
a table of which sections are exact arithmetic, which are measured on CPU, and
which require hardware. The kernel source is included in full and executes where
a GPU is present. Nothing is faked and nothing silently degrades.

## Three ways to run them

**1. Copy-paste into Colab.** Open a lab's `.py` file, copy the whole thing into
one Colab cell, run it. Every lab is written to work this way: dependencies
self-install, nothing is read from disk, and the workload sizes itself to
whatever hardware it finds.

**2. Open the notebook.** Every lab has a generated `.ipynb` beside it with an
"Open In Colab" badge, split into cells so you can poke at intermediate state.

**3. Run locally.**

```bash
pip install -r ../requirements.txt
python3 lab04_train_a_bpe_tokenizer.py
```

## Two environment variables

| Variable | Effect |
|---|---|
| `FLE_SMOKE_TEST=1` | Shrink every loop, dataset, and model to the smallest size that still exercises the code path. Finishes in seconds. CI sets this on every commit, which is how we know the labs still run. |
| `FLE_DEVICE=cpu` / `cuda` | Force a device. Labs auto-detect when unset. |

## How to get value out of a lab

Reading a lab teaches you almost nothing. The value is in breaking it.

Before you run anything, **write down a prediction**. Then change one thing and
see whether you were right:

- Train the BPE tokenizer on 90% Python and predict what happens to English
  fertility. (It is asymmetric, and the direction surprises people.)
- Set `skew=1.0` in the MoE lab and predict whether imbalance still develops.
  (It largely does not — imbalance is driven by the *data* distribution, not the
  routing mechanism.)
- Remove `RAW_CRAWL_SAMPLES` from the dedup lab's negative class and predict
  what the quality classifier does. (Every out-of-domain probe flips to KEEP,
  and the finding disappears.)
- Set `flash_attention=False` in the memory model at 32K and predict how much
  activation memory grows. (Most people say 4×. It is 16×.)
- Add a `0.05` per-token thinking bonus in the GRPO lab and predict what happens
  to accuracy. (It goes to exactly zero.)

Every lab ends with a "things to try" section listing changes worth making, each
with the wrong prediction most people make and why it is wrong. The surprise is
the point — being wrong about a system is how you find the edge of your model of
it.

## Four labs report a negative result

This is deliberate, and it is worth knowing before you read them:

- **`lab05_attention_variants` §5** cannot rank the attention variants.
  Within-variant seed spread reaches 72pp because the task is bimodal on whether
  an induction circuit forms. Recognising an underpowered ablation is the lesson.
- **`lab15_grpo_countdown` §6** finds that oversampling and dropping
  zero-advantage groups costs ~1.8× the generation to reach the same accuracy.
  The naive fix for §5's real problem is not a fix.
- **`lab14_dpo_from_scratch` §3** finds that one of the three "classic DPO bugs"
  is not a bug: prompt log-probabilities cancel exactly.
- **`lab22_kv_cache_and_batching` §6** cannot explain why 1-token blocks are bad,
  because block tables are negligible at every size measured. The real cost is in
  the attention kernel, which a bookkeeping simulation cannot see.

A lab that reports what it could not establish is more useful than one that
rounds a result up.

## Design rules

Every lab here follows these, and CI enforces what it can:

- **No downloads required.** Corpora and datasets are embedded or generated. A
  lab that only works online is a lab that will not work for someone.
- **CPU is a first-class target.** GPU makes labs faster; it is never required,
  with the single stated exception of `lab20`'s Triton section.
- **Small enough to reason about.** Roughly 400–900 lines including the prose
  cells. (An earlier version of this file said "under ~400 lines"; no lab has
  respected that since `lab04`, and the rule was really protecting *readability*,
  which is what it now says.)
- **Honest about what it demonstrates.** `lab05_moe_routing` says explicitly that
  it does not reach full expert collapse, because a CPU lab cannot. `lab06`
  prints a disclaimer that its configuration ranking is a model rather than a
  measurement. `lab16` says its generator is a model of a language model. A lab
  that overclaims is worse than no lab.
- **Real, not simulated,** wherever it can be. The MoE lab really routes tokens.
  The BPE lab really learns merges. `lab08_precision_and_stability` uses real
  FP8 dtypes. Where something *is* modelled rather than measured, the lab says so
  in its header.
- **Write the code, run it, then write the prose.** Narration is derived from
  measured values so it cannot drift, and it has to be true under
  `FLE_SMOKE_TEST=1` as well, because that is what CI runs.

## Regenerating the notebooks

The `.py` files are the source of truth. After editing one:

```bash
python3 ../tools/build_notebooks.py
```

Commit both the `.py` and the regenerated `.ipynb`.
