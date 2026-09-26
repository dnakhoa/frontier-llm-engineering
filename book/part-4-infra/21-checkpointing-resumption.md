# Chapter 21: Checkpointing, resumption, the two-week-run problem

> Reading time: ~35 minutes. By the end of this chapter you should be able to compute a checkpoint's size and the optimal interval between checkpoints, explain why the data loader is the most commonly forgotten piece of training state, know what makes bit-exact resumption hard and when it is worth the cost, and diagnose the characteristic failure where a run resumes successfully and silently trains on the wrong data.

*Current as of early 2025.*

## 21.1 Failure is the normal case

A frontier pre-training run occupies thousands of GPUs for weeks. Individual components are reliable; the aggregate is not.

Take a per-GPU mean time between failures of 50,000 hours — optimistic for hardware under sustained full load. On 16,000 GPUs:

$$\text{MTBF}_{\text{cluster}} \approx \frac{50{,}000 \text{ hours}}{16{,}000} \approx 3.1 \text{ hours}$$

And that counts only GPUs. Add NICs, optical transceivers, cables, switches, PSUs, cooling, host memory, and the storage layer, and the practical figure is worse.

Meta's Llama 3 paper [\[5\]](../appendix/b-references.md#5-llama-3) publishes the real numbers, which is rare and valuable. Over a 54-day pre-training snapshot on 16,384 GPUs they recorded **466 job interruptions**: 47 planned, 419 unexpected. Roughly 78% of the unexpected ones were attributed to confirmed hardware issues, with GPU problems the largest single category. They still achieved over 90% effective training time.

Read those two facts together. **A failure roughly every three hours, and over 90% of wall-clock spent training.** That gap is what this chapter is about. It is not achieved by making failures rare; it is achieved by making them cheap.

The governing quantity is simple:

$$\text{cost of a failure} = \text{time since last checkpoint} + \text{time to detect} + \text{time to reschedule} + \text{time to reload}$$

Every term is an engineering target.

## 21.2 What "state" means

The most common resumption bug is an incomplete definition of state. A full checkpoint contains:

| Component | Size (70B, mixed precision) | Forgotten? |
|---|---|---|
| Model parameters (FP32 master) | 280 GB | Never |
| Optimizer state — Adam $m$ and $v$ | 560 GB | Rarely |
| Learning-rate scheduler position | bytes | Sometimes |
| RNG state — per rank, per device | KB | **Often** |
| Data loader position | KB to MB | **Very often** |
| Gradient accumulation buffers | 0 mid-step | Only matters if you checkpoint mid-step |
| Loss-scale / FP8 scaling factors | KB | **Often** |
| Training step count and token count | bytes | Rarely |
| EMA weights, if used | 280 GB | Sometimes |

The bottom half of that table is kilobytes against terabytes, and it is where the bugs are — precisely *because* it is small enough to overlook.

**RNG state.** Dropout masks, data shuffling, and any stochastic augmentation depend on it. Resume without restoring it and the run continues with a different random stream. This is usually harmless statistically and it is fatal to reproducibility, so a run you cannot reproduce is a run you cannot debug.

**Loss scale and FP8 scaling factors.** Dynamic loss scaling adapts over training; reset it to the initial value on every resume and you get a burst of skipped steps and possibly instability, at every resume, on a run that resumes 419 times.

**The data loader position.** §21.6. This is the important one.

## 21.3 Checkpoint size and time

The arithmetic, for a 70B dense model in mixed precision with AdamW:

```
FP32 master weights:      70e9 × 4 =  280 GB
Adam first moment (m):    70e9 × 4 =  280 GB
Adam second moment (v):   70e9 × 4 =  280 GB
                                     -------
                                      840 GB per checkpoint
```

(You do not save the BF16 working copy — it is derived from the master weights. Saving it is a common and pointless 140 GB.)

Now the write. If your storage fabric delivers 100 GB/s aggregate:

$$\frac{840 \text{ GB}}{100 \text{ GB/s}} = 8.4 \text{ seconds}$$

That is fine. But two things make it much worse in practice:

**Naive gathering.** If every rank sends its shard to rank 0, which assembles the full state and writes it, you have serialized 840 GB through one node's NIC and one node's memory. At 25 GB/s that is 34 seconds of *all* GPUs idle, and the assembling node may not even have 840 GB of host RAM.

**Storage contention.** 2,048 ranks writing simultaneously to a shared filesystem is a workload most filesystems handle badly. Metadata operations — creating thousands of files — frequently dominate the actual data transfer.

For a 671B MoE like DeepSeek-V3, the same arithmetic gives roughly 8 TB per checkpoint. At that size, naive approaches stop being slow and start being impossible.

## 21.4 Sharded checkpointing

The fix, and it is standard now.

**Each rank writes only what it owns, in parallel, to separate files.**

Under FSDP or ZeRO the parameters and optimizer state are already sharded across ranks (§6.3). A sharded checkpoint preserves that layout: rank $i$ writes its shard, all ranks write concurrently, and the aggregate bandwidth is the fabric's rather than one node's.

`torch.distributed.checkpoint` [\[87\]](../appendix/b-references.md#87-pytorch-distributed-checkpoint) implements this, and the property that makes it genuinely useful is **resharding**: the checkpoint records the logical structure of each tensor, not just the bytes one rank happened to hold. That means you can save on 512 GPUs and load on 256, or change your TP degree between runs.

That capability matters more than it first appears:

- A node fails and the replacement has fewer working GPUs.
- You want to continue a pre-training run at a different parallelism configuration for the annealing phase.
- You want to load a training checkpoint for inference at a completely different TP degree.
- You want to debug on 8 GPUs a problem that appeared on 1,024.

A checkpoint format that hardcodes the parallelism layout makes all four of these impossible, and teams discover this at the worst moment.

```python
import torch.distributed.checkpoint as dcp

def save(model, optimizer, dataloader, step, path):
    """Sharded save. Every rank writes its own shard, concurrently.

    The small-but-critical state -- RNG, data loader position, scheduler --
    travels in the same checkpoint. Splitting it across two mechanisms is how
    it gets forgotten (section 21.2).
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        # The kilobytes that cause the outages.
        "dataloader": dataloader.state_dict(),
        "rng": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
    }
    dcp.save(state, checkpoint_id=path)
```

## 21.5 Asynchronous checkpointing

Even at 8 seconds, checkpointing every 30 minutes costs 0.4% of your run. Cheap. But you want to checkpoint *more* often than that (§21.7), and the cost is linear in frequency.

The fix: **copy the state to host memory synchronously, then write to storage in the background while training continues.**

```
GPU timeline:  [ train ][copy][ train train train train train ][copy][ train ...
Host timeline:              [------ write 840 GB to storage ------]
```

The GPU stall is now the device-to-host copy — typically a fraction of a second over PCIe or NVLink-C2C — instead of the storage write.

Three things this requires:

**Host memory for the staging buffer.** 840 GB of it, across the nodes. Usually fine, occasionally the binding constraint.

**A guarantee that the buffer is not mutated while writing.** The optimizer must not overwrite the staged state. Either copy into a dedicated buffer, or block the next optimizer step until the copy completes.

**Handling a failure mid-write.** A checkpoint that is half-written must never be mistaken for a complete one. Write to a temporary path and rename atomically on success — the same pattern as any durable write. A checkpoint directory without a completion marker is garbage and should be treated as such by the loader.

## 21.6 The data loader is the piece everyone forgets

The most consequential item in §21.2's table, and the one that produces a bug you may never notice.

**What goes wrong.** You resume from step 40,000. The model weights and optimizer state are restored perfectly. The data loader restarts **from the beginning of the dataset**.

The run continues. The loss looks fine — slightly *better*, actually, because the model is now seeing data it has already learned from. Nothing errors. Nothing alerts.

You have quietly trained a second epoch on the first 40,000 steps' worth of data while believing you did a single pass. On a run with 419 interruptions, you may have seen the first 5% of your corpus dozens of times and the last 20% never.

**Why it happens.** Data loading is usually a separate subsystem from the model, written by different people, and `state_dict()` on a `DataLoader` does not capture position in any obvious way. Sharded, shuffled, multi-worker, infinitely-streaming pipelines make "where was I" genuinely non-trivial.

**What the state actually is:**

```python
class ResumableDataLoader:
    """A data loader whose position survives a restart.

    The invariant that matters: given the same state_dict, every rank must
    resume at exactly the sample it would have consumed next -- not
    approximately, and not at a shard boundary.
    """

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "shuffle_seed": self.seed,          # the permutation must be reproducible
            "samples_consumed": self.consumed,  # global, not per-rank
            "shard_order": self.shard_order,    # which files, in which order
            "within_shard_offset": self.offset, # position inside the current file
            "world_size": self.world_size,      # to detect a reshard on resume
        }
```

**Two acceptable designs.** Either make the loader genuinely resumable, as above — the right answer for a long pre-training run. Or make the data stream a deterministic function of the global step, so that step $n$ always yields the same batch and "resuming" means recomputing an index. The second is simpler and constrains your shuffling; the first is more flexible and needs testing.

**How to test it, in a way that actually catches the bug:** run 100 steps, checkpoint at step 50, resume, and assert that steps 51–100 consume the identical sample IDs in both the uninterrupted and the resumed run. Lab 21 does exactly this, and it is the single highest-value test in this chapter.

## 21.7 How often to checkpoint

There is a right answer, and it is old. The Young/Daly result [\[88\]](../appendix/b-references.md#88-young-daly) gives the optimal interval as

$$\tau_{\text{opt}} \approx \sqrt{2 \cdot C \cdot \text{MTBF}}$$

where $C$ is the cost of writing one checkpoint and MTBF is the mean time between failures.

The intuition: checkpoint too often and you pay $C$ constantly; too rarely and each failure costs half an interval of lost work in expectation. The optimum balances them.

**Worked example.** Cluster MTBF of 3 hours (10,800 s), asynchronous checkpoint cost of 10 s:

$$\tau_{\text{opt}} = \sqrt{2 \times 10 \times 10{,}800} \approx 465 \text{ s} \approx 8 \text{ minutes}$$

Eight minutes, which is far more frequent than most people's instinct. And it is exactly why asynchronous checkpointing (§21.5) matters: with a synchronous 60-second checkpoint the optimum moves to 20 minutes, and each failure costs 10 minutes of lost work instead of 4.

Reducing $C$ improves the outcome twice over — cheaper checkpoints *and* a shorter optimal interval.

**Two adjustments the formula does not include:**

- **Detection and rescheduling time.** If it takes 5 minutes to notice a hang and 10 to get replacement nodes, that dominates the 4 minutes of lost work and is where your engineering effort should go instead.
- **Checkpoints you keep for reasons other than restart.** Evaluation checkpoints, annealing branch points, and the ability to roll back past a loss spike (§8.6). These have their own cadence, and they need retention policy — 8 TB every 8 minutes fills a filesystem quickly.

A common policy: frequent lightweight checkpoints for restart with a short retention window (keep the last 3), plus infrequent permanent checkpoints (every few thousand steps) kept forever.

## 21.8 Bit-exact resumption

Should a resumed run produce numerically identical results to an uninterrupted one?

**The case for yes.** Debugging. If a loss spike appears at step 47,000 you want to reproduce it exactly. Without determinism you cannot bisect, and you are reduced to arguing about whether a rerun that looked different was a different bug.

**Why it is hard:**

- **Floating-point addition is not associative** (§20.9). All-reduce over 512 ranks can complete in different orders on different runs, giving different last bits.
- **Some kernels are nondeterministic** by design, using atomics for speed.
- **cuDNN autotuning** may select a different algorithm on a different run.
- **Resharding changes the reduction tree**, so bit-exactness across a parallelism change is essentially impossible.

**The cost of forcing it:** deterministic algorithms are slower, sometimes substantially. `torch.use_deterministic_algorithms(True)` will also raise on operations with no deterministic implementation.

**The pragmatic position most labs take:** do not require bit-exactness for the main run. Do require **exact data and state resumption** — the same samples, the same RNG stream, the same scheduler position — so the run is *statistically* identical and the data-loader bug in §21.6 is impossible. Keep a deterministic mode available for debugging on small configurations.

That distinction is worth being precise about, because "reproducible" gets used for both and they are very different engineering commitments.

## 21.9 Elastic training

The next level: rather than waiting for a replacement node, continue with fewer.

Requires resharding (§21.4), a data loader that handles a changed world size, and a decision about the global batch size — do you keep it constant by increasing gradient accumulation, or let it shrink and change the effective learning rate?

Keeping the global batch constant is almost always right, because changing it mid-run changes the optimization trajectory in ways that interact badly with the LR schedule (§8.5).

Elastic training is real and it is less common at frontier labs than the literature suggests, for a straightforward reason: **frontier training jobs are gang-scheduled on dedicated clusters where you own all the nodes.** There is no other tenant to yield to. If a node dies you want a replacement from the hot spare pool in minutes, not a degraded configuration. Elasticity matters much more for spot-instance and shared-cluster training.

## 21.10 Formats, and the conversion nobody plans for

A training checkpoint and an inference checkpoint are different objects.

**Training:** sharded across ranks, FP32 master weights, optimizer state, in whatever layout the trainer uses.

**Inference:** consolidated or resharded to the serving TP degree, weights only, BF16 or FP8, often with fused QKV and gate/up projections, in `safetensors`.

The conversion is a real pipeline step and it is a reliable source of bugs:

- **Tensor-parallel resharding** — a weight split across 8 training ranks must be reassembled and re-split across 4 serving ranks, along the correct axis. Getting the axis wrong for one weight type produces a model that runs and generates garbage.
- **Fusing projections** for serving efficiency changes the tensor layout.
- **Quantization** to FP8 or INT8, with the scaling factors computed and stored.
- **The tokenizer and chat template** must ship with it (§11.6).

Use `safetensors` for anything you distribute. It is memory-mappable, it is fast to load, and — the important part — it does not execute code on load, unlike `torch.save`'s pickle format. Loading an untrusted `.pt` file is arbitrary code execution.

## 21.11 What goes wrong

| Symptom | Cause | Where |
|---|---|---|
| Loss improves suspiciously after a resume | Data loader restarted from the beginning | §21.6 |
| Loss spikes on every resume | Loss scale or FP8 scaling factors reset | §21.2 |
| Cannot reproduce a run | RNG state not saved | §21.2, §21.8 |
| Resume fails after replacing a node | Checkpoint format hardcodes world size | §21.4 |
| Checkpoint write takes minutes | Gathering to rank 0 | §21.3 |
| Filesystem full at 2am | No retention policy | §21.7 |
| Corrupt checkpoint after a crash | Non-atomic write, no completion marker | §21.5 |
| Converted model generates garbage | TP resharding along the wrong axis | §21.10 |
| Model loads but quality is subtly worse | Loaded BF16 working weights instead of FP32 master | §21.3 |

The first row is the one to internalize. It is silent, it looks like good news, and it corrupts your data distribution for the remainder of the run.

## 21.12 JD, decoded

**Training Infrastructure Engineer.** Owns checkpointing, resumption, failure detection, and the automatic-restart loop. This is a distributed-systems job with a specific and unusual failure model: synchronous, gang-scheduled, all-or-nothing. Classical web-backend intuitions transfer poorly.

**ML Systems Engineer.** Overlaps. Also owns the storage layer's interaction with training — a shared filesystem that handles 2,048 concurrent writers badly is a training problem.

**Research Engineer (pre-training).** Consumes it. Cares about resume correctness, retention policy, and the ability to branch a run from an earlier checkpoint for the annealing phase (§9.4).

A JD mentioning "fault-tolerant training at scale," "distributed checkpointing," or "job resiliency" is this work. It is unglamorous and it is the difference between 90% and 60% effective training time on a run costing millions.

## 21.13 What you should take from this chapter

1. **At 16,000 GPUs a failure every few hours is normal.** Llama 3 recorded 466 interruptions in 54 days and still achieved over 90% effective training time — by making failures cheap, not rare.
2. **Checkpoint state is more than weights and optimizer.** RNG, scheduler, loss scale, and the data loader position are kilobytes against terabytes, which is exactly why they get forgotten.
3. **A 70B mixed-precision checkpoint is about 840 GB** — FP32 master plus both Adam moments. Do not save the BF16 working copy; it is derived.
4. **Shard the checkpoint and write in parallel.** Gathering to rank 0 serializes terabytes through one NIC.
5. **Save the logical tensor structure, not the physical layout**, so you can reshard. Saving on 512 ranks and loading on 256 is a routine need, and a format that forbids it fails at the worst moment.
6. **Checkpoint asynchronously:** synchronous device-to-host copy, background write. The GPU stall becomes a fraction of a second instead of the full storage write.
7. **The optimal interval is $\sqrt{2 \cdot C \cdot \text{MTBF}}$** — about 8 minutes for a 10-second checkpoint at a 3-hour MTBF. Far more often than instinct suggests, and reducing $C$ helps twice.
8. **The data loader bug is silent and it looks like good news.** Test it by asserting that a resumed run consumes identical sample IDs to an uninterrupted one.
9. **Require exact state resumption; do not require bit-exactness.** They are very different commitments, and only the first is worth its cost on a production run.
10. **Training and inference checkpoints are different objects**, and the conversion — resharding, fusing, quantizing — is a real pipeline step with real bugs. Distribute `safetensors`, never pickle.

The next chapter is inference and serving: where the model finally meets a user, where decoding is entirely memory-bound, and where — per Chapter 13 — most of your post-training wall-clock has been hiding all along.

---

**Exercises:** [Chapter 21 problem set](../../exercises/ch21.md) — includes checkpoint sizing for an MoE, the optimal-interval calculation under several failure regimes, and designing a resumable sharded data loader.
**Lab:** [`lab21_checkpoint_resume`](../../labs/lab21_checkpoint_resume.py) — define resume correctness as bit-exactness and test it. Every incomplete recipe diverges on the *first* step after the resume, by a few percent of the loss, with nothing to alert you.

---

**References for this chapter**

- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — Meta AI, July 2024. The published failure statistics: 466 interruptions in 54 days on 16,384 GPUs.
- [\[3\] Megatron-LM](../appendix/b-references.md#3-megatron-lm) — Shoeybi et al., 2019. Distributed training layout that checkpoints must preserve.
- [\[87\] PyTorch Distributed Checkpoint](../appendix/b-references.md#87-pytorch-distributed-checkpoint) — the sharded, reshardable checkpoint API.
- [\[88\] Young/Daly optimal checkpoint interval](../appendix/b-references.md#88-young-daly) — Daly, 2006. The $\sqrt{2 C \cdot \text{MTBF}}$ result.
- [See full reference list](../appendix/b-references.md)
