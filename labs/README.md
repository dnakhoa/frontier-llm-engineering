# Labs

Runnable code for the mechanisms the chapters describe. Each lab implements the real thing rather than a demonstration of it, and each is small enough to run on a laptop CPU.

## What exists today

**Four labs, all tested and all passing CI on every commit.**

| Lab | Chapter | What you build | Runtime (CPU) |
|---|---|---|---|
| [`lab04_train_a_bpe_tokenizer`](lab04_train_a_bpe_tokenizer.py) | [4 — Tokenization](../book/part-2-pretraining/04-tokenization.md) | Byte-level BPE from scratch, then fertility measured across five languages on held-out text. Shows the ~1.8× penalty an English-only merge corpus imposes on Vietnamese, and what deleting the pre-tokenizer regex does to the learned merges. | ~4 s |
| [`lab05_moe_routing`](lab05_moe_routing.py) | [5 — Architecture](../book/part-2-pretraining/05-architecture.md) | Top-k router and a real MoE layer. Watch load imbalance reach a ~3× straggler tax, then fix it two ways — the standard auxiliary loss, and DeepSeek's auxiliary-loss-free bias controller — and measure what each costs the loss. | ~50 s |
| [`lab06_parallelism_memory_model`](lab06_parallelism_memory_model.py) | [6 — Distributed training](../book/part-2-pretraining/06-distributed-training.md) | A memory-and-throughput calculator for dense and MoE models under any TP/PP/EP/DP/ZeRO configuration. Searches the configuration space and tells you what fits and roughly how fast. Pure arithmetic, no GPU. | ~2 s |
| [`lab11_sft_packing`](lab11_sft_packing.py) | [11 — SFT](../book/part-3-post-training/11-sft-frontier-style.md) | Loss masking and sequence packing from scratch. Reproduces all three classic masking bugs and shows what each does to a trained model, then measures packing cross-contamination — which drops to *exactly* zero under block-diagonal attention. | ~30 s |

## What does not exist yet

**The other labs referenced in the chapters have not been written.** Chapters 3, 7, 8, 9, 12, 13, 15, 16, 20, 21, 22, and 23 each name a lab in their closing section and in their exercise set; those names are placeholders describing what the lab *would* do, not links to code.

Named but not written:

`lab03_dedup_and_quality` · `lab05_attention_variants` · `lab07_collective_bandwidth` · `lab08_precision_and_stability` · `lab08_scaling_laws` · `lab09_rope_extension` · `lab12_reward_model` · `lab13_ppo_minimal` · `lab14_dpo_from_scratch` · `lab15_grpo_countdown` · `lab16_best_of_n_and_prm` · `lab20_triton_fused_kernel` · `lab21_checkpoint_resume` · `lab22_kv_cache_and_batching` · `lab23_contamination_check`

The exercise sets contain "🧪 break the lab" problems for several of these. Read them as specifications: they describe an experiment worth running and state what the common wrong prediction is. If you build one, [the contribution is very welcome](../CONTRIBUTING.md) — the design rules are below and CI will check them.

**Everything else in this book is complete.** All 26 chapters, all 26 exercise sets, and all 26 worked solution sets are written. The labs are the one place where the repository promises more than it currently delivers, and this section exists so nobody discovers that by clicking a dead link.

## Three ways to run them

**1. Copy-paste into Colab.** Open a lab's `.py` file, copy the whole thing into one Colab cell, run it. Every lab is written to work this way: dependencies self-install, nothing is read from disk, and the workload sizes itself to whatever hardware it finds.

**2. Open the notebook.** Every lab has a generated `.ipynb` beside it with an "Open In Colab" badge, split into cells so you can poke at intermediate state.

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

Before you run anything, **write down a prediction**. Then change one thing and see whether you were right:

- Train the BPE tokenizer on 90% Python and predict what happens to English fertility. (It is asymmetric, and the direction surprises people.)
- Set `skew=1.0` in the MoE lab and predict whether imbalance still develops. (It largely does not — imbalance is driven by the *data* distribution, not the routing mechanism.)
- Remove the block-diagonal mask in the packing lab and predict whether you can measure the contamination. (You can, and it is exactly zero with the mask on.)
- Set `flash_attention=False` in the memory model at 32K and predict how much activation memory grows. (Most people say 4×. It is 16×.)

Every lab ends with a "things to try" section listing changes worth making, each with the wrong prediction most people make and why it is wrong. The surprise is the point — being wrong about a system is how you find the edge of your model of it.

## Design rules

Every lab here follows these, and CI enforces them:

- **No downloads required.** Corpora and datasets are embedded. A lab that only works online is a lab that will not work for someone.
- **CPU is a first-class target.** GPU makes labs faster; it is never required.
- **Small enough to reason about.** Under ~400 lines.
- **Honest about what it demonstrates.** `lab05` reproduces load imbalance and says explicitly that it does not reach full expert collapse, because a CPU lab cannot. `lab06` prints a disclaimer that its configuration ranking is a model rather than a measurement. A lab that overclaims is worse than no lab.
- **Real, not simulated.** The MoE lab really routes tokens. The BPE lab really learns merges. Nothing here is a mock-up of a mechanism.

## Regenerating the notebooks

The `.py` files are the source of truth. After editing one:

```bash
python3 ../tools/build_notebooks.py
```

Commit both the `.py` and the regenerated `.ipynb`.
