# Labs

Runnable code for the mechanisms the chapters describe. Each lab is small enough to run on a free Colab T4 — most run on a laptop CPU — and implements the real thing rather than a demonstration of it.

## Three ways to run them

**1. Copy-paste into Colab.** Open a lab's `.py` file, copy the whole thing, paste into one Colab cell, run. Every lab is written to work this way: dependencies self-install, nothing is read from disk, and the workload sizes itself to whatever hardware it finds.

**2. Open the notebook.** Every lab has a generated `.ipynb` beside it with an "Open In Colab" badge at the top, split into cells so you can run it piece by piece and poke at the intermediate state. This is the better option if you want to experiment rather than just watch it run.

**3. Run locally.**

```bash
pip install -r ../requirements.txt
python3 lab11_sft_packing.py
```

## Two environment variables

| Variable | Effect |
|---|---|
| `FLE_SMOKE_TEST=1` | Shrink every loop, dataset, and model to the smallest size that still exercises the code path. Finishes in under a minute. CI sets this on every commit, which is how we know the labs still run. |
| `FLE_DEVICE=cpu` / `cuda` | Force a device. Labs auto-detect when this is unset. |

## How to get value out of a lab

Reading a lab teaches you almost nothing. The value is in breaking it.

Before you run anything, **write down a prediction**. Then change one thing and see whether you were right:

- Turn off the auxiliary loss in the MoE lab. How long until experts collapse?
- Set the KL coefficient to zero in the PPO lab. What does the policy do?
- Remove the block-diagonal mask in the packing lab. Can you measure the contamination?
- Halve the group size in the GRPO lab. What happens to gradient variance?

Every lab ends with a "things to try" section listing changes worth making, each with the wrong prediction most people make and why it is wrong. The surprise is the point — being wrong about a system is how you find the edge of your model of it.

## The labs

### Part II — Pre-training

| Lab | Chapter | What you build | Runs on |
|---|---|---|---|
| [`lab03_dedup_and_quality`](lab03_dedup_and_quality.py) | [3](../book/part-2-pretraining/03-data-pipelines.md) | MinHash-LSH near-duplicate detection and a quality classifier over a real web corpus sample | CPU |
| [`lab04_train_a_bpe_tokenizer`](lab04_train_a_bpe_tokenizer.py) | [4](../book/part-2-pretraining/04-tokenization.md) | BPE from scratch, then fertility and compression measured across five languages | CPU |
| [`lab05_attention_variants`](lab05_attention_variants.py) | [5](../book/part-2-pretraining/05-architecture.md) | MHA, MQA, GQA, and MLA side by side, with the KV-cache arithmetic verified empirically | CPU / T4 |
| [`lab05_moe_routing`](lab05_moe_routing.py) | [5](../book/part-2-pretraining/05-architecture.md) | Top-k routing, expert collapse, auxiliary-loss balancing vs the bias-based scheme | CPU / T4 |
| [`lab06_parallelism_memory_model`](lab06_parallelism_memory_model.py) | [6](../book/part-2-pretraining/06-distributed-training.md) | A memory model for DP/ZeRO/TP/PP/EP that tells you what fits on what | CPU |
| [`lab07_collective_bandwidth`](lab07_collective_bandwidth.py) | [7](../book/part-2-pretraining/07-cluster-reality.md) | Collective-communication cost model, and why a rack-split job slows down | CPU |
| [`lab08_precision_and_stability`](lab08_precision_and_stability.py) | [8](../book/part-2-pretraining/08-optimization.md) | FP8/BF16/FP16 numerics, overflow, and why per-block scaling is necessary | CPU / T4 |
| [`lab08_scaling_laws`](lab08_scaling_laws.py) | [8](../book/part-2-pretraining/08-optimization.md) | Fit a Chinchilla-style scaling law on models you train yourself, then extrapolate | CPU / T4 |
| [`lab09_rope_extension`](lab09_rope_extension.py) | [9](../book/part-2-pretraining/09-mid-training.md) | Position Interpolation, NTK-aware, and YaRN, with the quality cost measured | CPU / T4 |

### Part III — Post-training

| Lab | Chapter | What you build | Runs on |
|---|---|---|---|
| [`lab11_sft_packing`](lab11_sft_packing.py) | [11](../book/part-3-post-training/11-sft-frontier-style.md) | Loss masking, all three masking bugs, packing, and block-diagonal attention | CPU |
| [`lab12_reward_model`](lab12_reward_model.py) | [12](../book/part-3-post-training/12-reward-modeling.md) | A Bradley-Terry reward model, plus the length-bias measurement | CPU / T4 |
| [`lab13_ppo_minimal`](lab13_ppo_minimal.py) | [13](../book/part-3-post-training/13-ppo-rloo-frontier.md) | PPO and RLOO on a small language task, with the KL budget made visible | CPU / T4 |
| [`lab14_dpo_from_scratch`](lab14_dpo_from_scratch.py) | [14](../book/part-3-post-training/14-dpo-family.md) | The DPO loss from scratch, checked against `trl` to numerical precision | CPU / T4 |
| [`lab15_grpo_countdown`](lab15_grpo_countdown.py) | [15](../book/part-3-post-training/15-grpo-reasoning.md) | GRPO on a verifiable task — watch response length grow on its own | T4 |
| [`lab16_best_of_n_and_prm`](lab16_best_of_n_and_prm.py) | [16](../book/part-3-post-training/16-process-reward-models.md) | Best-of-n, majority voting, and a process reward model compared honestly | CPU / T4 |

### Part IV — Infra

| Lab | Chapter | What you build | Runs on |
|---|---|---|---|
| [`lab20_triton_fused_kernel`](lab20_triton_fused_kernel.py) | [20](../book/part-4-infra/20-custom-kernels.md) | A fused RMSNorm kernel in Triton, benchmarked against PyTorch — including where it loses | T4 (CPU falls back to a numerics-only check) |
| [`lab21_checkpoint_resume`](lab21_checkpoint_resume.py) | [21](../book/part-4-infra/21-checkpointing-resumption.md) | Sharded checkpointing, and a bit-exact resume test that catches the usual bugs | CPU |
| [`lab22_kv_cache_and_batching`](lab22_kv_cache_and_batching.py) | [22](../book/part-4-infra/22-inference-serving.md) | KV cache, static vs continuous batching, and paged-attention block management | CPU / T4 |
| [`lab23_contamination_check`](lab23_contamination_check.py) | [23](../book/part-4-infra/23-evaluation.md) | N-gram and embedding-based benchmark contamination detection | CPU |

## Design rules

Every lab in this directory follows these, and CI enforces them:

- **No downloads required.** Labs that benefit from a real tokenizer or dataset try to fetch one and fall back to a self-contained substitute if the network is unavailable. A lab that only works online is a lab that will not work for someone.
- **CPU is a first-class target.** GPU makes labs faster; it is never required except where the lab is *about* the GPU (Lab 20), and even then there is a CPU path that checks correctness.
- **Small enough to reason about.** Under ~400 lines. If a mechanism cannot be shown in 400 lines, the lab shows the core of it and the chapter explains the rest.
- **Real, not simulated.** The MoE lab really routes tokens. The DPO lab really computes the DPO loss and really matches `trl`. Nothing here is a mock-up of a mechanism.

## Regenerating the notebooks

The `.py` files are the source of truth. After editing one:

```bash
python3 ../tools/build_notebooks.py
```

Commit both the `.py` and the regenerated `.ipynb`.
