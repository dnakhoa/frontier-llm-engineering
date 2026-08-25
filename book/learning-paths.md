# Learning paths

> Reading time: ~15 minutes. By the end you should have picked a route through the book, with a week-by-week schedule and a concrete artifact to produce at the end of it.

Twenty-six chapters is too many to read front-to-back before you get value out of them. This page gives six routes. Pick one, follow it, and treat every other chapter as reference material you can dip into.

Every path assumes you have cleared [the prerequisites](prerequisites.md).

## Choosing

| If your goal is… | Take |
|---|---|
| Understand how a frontier model gets built, broadly | [The complete path](#the-complete-path) |
| Work on pre-training or large-scale training | [The pre-training track](#the-pre-training-track) |
| Work on RLHF, reasoning, or alignment | [The post-training track](#the-post-training-track) |
| Work on kernels, serving, or training infra | [The infrastructure track](#the-infrastructure-track) |
| Prepare for frontier-lab interviews in the next month | [The interview sprint](#the-interview-sprint) |
| Just read the papers better | [The weekend skim](#the-weekend-skim) |

---

## The complete path

**16 weeks, ~8 hours/week. Chapters + exercises + labs, everything.**

This is the book as designed. At roughly 8 hours a week it lands in a semester.

| Week | Chapters | Labs | Milestone |
|---|---|---|---|
| 1 | Preface, 1, 2 | — | You can read a frontier JD and name every acronym in it |
| 2 | 3 | `lab03_dedup_and_quality` | You have deduplicated a real corpus with MinHash-LSH |
| 3 | 4 | `lab04_train_a_bpe_tokenizer` | You have trained a BPE tokenizer and measured fertility across 5 languages |
| 4 | 5 (§5.1–5.8) | `lab05_attention_variants` | You can derive the KV-cache size for MHA, MQA, GQA, and MLA |
| 5 | 5 (§5.9–end) | `lab05_moe_routing` | You have watched expert collapse happen and fixed it |
| 6 | 6 (§6.1–6.8) | `lab06_parallelism_memory_model` | You can size a run: given a model and a cluster, choose the parallelism |
| 7 | 6 (§6.9–end), 7 | `lab07_collective_bandwidth` | You can explain why a job slows down when the scheduler splits it across racks |
| 8 | 8 (§8.1–8.8) | `lab08_precision_and_stability` | You can explain why FP8 training needs per-block scaling |
| 9 | 8 (§8.9–end) | `lab08_scaling_laws` | You have fit a scaling law and used it to pick a model size |
| 10 | 9 | `lab09_rope_extension` | You have extended a model's context with YaRN and measured the cost |
| 11 | 10 | — | You can reconstruct DeepSeek-V3's full training recipe from memory |
| 12 | 11, 12 | `lab11_sft_packing`, `lab12_reward_model` | You have trained a Bradley-Terry reward model on real preferences |
| 13 | 13, 14 | `lab13_ppo_minimal`, `lab14_dpo_from_scratch` | Your from-scratch DPO loss matches `trl` to numerical precision |
| 14 | 15, 16 | `lab15_grpo_countdown`, `lab16_best_of_n_and_prm` | You have run GRPO and watched response length grow on its own |
| 15 | 17, 18, 19 | — | You can explain how R1-Zero got reasoning with no SFT at all |
| 16 | 20–23 skim, 24–26 | `lab22_kv_cache_and_batching` | You have a concrete skill-gap list and a plan |

**Final artifact.** Write a 2,000-word technical post reconstructing a frontier training run of your choice from its public report — the full recipe, the choices you would have made differently, and the parts the report does not tell you. If you can write that convincingly, you have the model this book exists to give you.

---

## The pre-training track

**8 weeks. For "Pre-training Engineer," "Large-Scale Training Engineer," "Foundation Model Engineer" roles.**

Core: **1, 2, 3, 4, 5, 6, 7, 8, 9, 10**, plus **21** (checkpointing) from Part IV.

| Week | Focus | Chapters | Labs |
|---|---|---|---|
| 1 | Orientation and data | 1, 2, 3 | `lab03_dedup_and_quality` |
| 2 | Tokenization and architecture | 4, 5 (§5.1–5.8) | `lab04_train_a_bpe_tokenizer` |
| 3 | MoE | 5 (§5.9–end) | `lab05_moe_routing` |
| 4 | Parallelism | 6 | `lab06_parallelism_memory_model` |
| 5 | The cluster | 7, 21 | `lab07_collective_bandwidth`, `lab21_checkpoint_resume` |
| 6 | Precision and stability | 8 (§8.1–8.8) | `lab08_precision_and_stability` |
| 7 | Scaling laws and mid-training | 8 (§8.9–end), 9 | `lab08_scaling_laws`, `lab09_rope_extension` |
| 8 | Synthesis | 10, 24 | — |

**Skip for now:** all of Part III. You will absorb what you need about post-training from Chapter 1 and from Chapter 10's final sections.

**Final artifact.** A written training plan for a 30B MoE on 256 H100s: data mix with token counts, tokenizer decision, full architecture config, parallelism strategy with the memory arithmetic shown, precision choice, LR schedule, checkpoint cadence, and the three failure modes you consider most likely with your mitigation for each. Exercise 6.7 and Exercise 8.9 are deliberately scaffolded toward this.

---

## The post-training track

**8 weeks. For "Post-training Researcher," "RLHF Engineer," "Alignment Researcher" roles.**

Core: **1, 2, 5 (skim), 11, 12, 13, 14, 15, 16, 17, 18, 19**, plus **22** (serving — post-training is bottlenecked on generation throughput, and most people learn this the hard way).

| Week | Focus | Chapters | Labs |
|---|---|---|---|
| 1 | Orientation | 1, 2, 5 (skim §5.1–5.8) | — |
| 2 | SFT | 11 | `lab11_sft_packing` |
| 3 | Reward modeling | 12 | `lab12_reward_model` |
| 4 | Online RL | 13 | `lab13_ppo_minimal` |
| 5 | Offline preference optimization | 14 | `lab14_dpo_from_scratch` |
| 6 | GRPO and reasoning | 15 | `lab15_grpo_countdown` |
| 7 | Verifiers and search-time compute | 16, 22 | `lab16_best_of_n_and_prm` |
| 8 | Alignment, tools, synthesis | 17, 18, 19 | — |

**Why Chapter 22 is in a post-training track:** an RL loop spends most of its wall-clock generating rollouts. If you cannot reason about batching, KV cache, and throughput, your GRPO run is three times slower than it needs to be and you will not know why. This surprises almost everyone the first time.

**Final artifact.** Implement a full preference-optimization pipeline end to end on a small model: SFT, then a reward model, then both PPO and DPO from the same starting checkpoint, then evaluate. Write up where they diverged and why. Labs 11 through 14 are the components; assembling them is the exercise.

---

## The infrastructure track

**6 weeks. For "ML Systems Engineer," "Kernel Engineer," "Inference Engineer," "Training Infrastructure" roles.**

Core: **1, 2, 5 (§5.1–5.8), 6, 7, 20, 21, 22, 23**.

| Week | Focus | Chapters | Labs |
|---|---|---|---|
| 1 | Orientation and architecture shapes | 1, 2, 5 (§5.1–5.8) | `lab05_attention_variants` |
| 2 | Parallelism | 6 | `lab06_parallelism_memory_model` |
| 3 | Cluster and failure | 7, 21 | `lab07_collective_bandwidth`, `lab21_checkpoint_resume` |
| 4 | Kernels | 20 | `lab20_triton_fused_kernel` |
| 5 | Serving | 22 | `lab22_kv_cache_and_batching` |
| 6 | Evaluation infra, synthesis | 23, 24 | `lab23_contamination_check` |

**Note on Chapter 20.** The kernel lab wants a GPU. Colab's free T4 is enough for the Triton exercises. Without any GPU the lab still runs — it falls back to a PyTorch reference implementation and compares numerics rather than speed, which teaches the correctness half but not the performance half.

**Final artifact.** Take one operation from a real model — RMSNorm, SwiGLU, or the RoPE application — write a fused Triton kernel for it, prove numerical equivalence against the PyTorch reference, and benchmark it honestly (including the cases where it loses).

---

## The interview sprint

**4 weeks, ~10 hours/week. For a frontier-lab interview loop you already have scheduled.**

This path optimizes for the questions actually asked, which are narrower than the book.

| Week | Read | Drill |
|---|---|---|
| 1 | 1, 2, 24 in full. 5 in full. | Every arithmetic drill in exercises 1, 5. Memorize the memory-per-parameter table cold. |
| 2 | 6 in full. 7 §7.1–7.6. 8 §8.1–8.8. | Exercises 6 and 8. You must be able to size a run on a whiteboard without notes. |
| 3 | 11, 12, 13, 14, 15. 19. | Exercises 12–15. Be able to write the DPO loss and the GRPO advantage from memory. |
| 4 | 3, 9, 22, 23, 25. Reread 24. | Mock questions below. Read the two case-study chapters (10, 19) again. |

**The questions that actually come up**, and the section that answers each:

- "How much memory does a 70B model need to train?" → §6.1
- "Walk me through how you would train a 30B MoE on 512 GPUs." → §6.9, §6.13
- "Why does DeepSeek use MLA instead of GQA?" → §5.7
- "What is the difference between PPO and GRPO, and why did GRPO win for reasoning?" → §13.4, §15.3
- "Your loss spikes at step 40,000. What do you do?" → §8.6
- "How would you detect benchmark contamination?" → §23.6
- "Why is FP8 training hard and how did DeepSeek make it work?" → §8.3
- "What breaks when you go from 1,000 to 10,000 GPUs?" → §7.8
- "How do you know your reward model is any good?" → §12.7

**A warning about this path.** It optimizes for a specific outcome and it will leave you with gaps you cannot see. If you get the job, come back and do the full track — the first six months are much easier with the whole model in your head.

---

## The weekend skim

**~12 hours. For reading frontier technical reports with comprehension.**

Read, in order: **Preface, 1, 2**, then **5**, then **6 §6.1–6.2 and §6.9–6.11**, then **8 §8.1–8.5**, then **15**, then **10 and 19**.

Skip every lab. Do only the paper-reading prompts in the exercise sets — they are the ones that pay off directly for this goal.

At the end, open the DeepSeek-V3 technical report ([\[1\]](appendix/b-references.md#1-deepseek-v3)) and read it straight through. If Chapters 5, 6, 8, and 10 did their job, it should read as a series of familiar decisions rather than a wall of jargon. That is the test.

---

## Working without a cohort

The hardest part of self-study is not the material. It is that nothing external forces you to continue. Four things that measurably help:

**Write down predictions before you run anything.** Before a lab, write what you expect. Before an exercise, write your answer. The prediction, not the result, is where the learning is — you cannot be surprised by something you never committed to.

**Keep a "did not understand" list.** One line per confusion, with the section number. Do not stop to resolve them mid-chapter. Revisit the list at the end of each part; a surprising fraction resolve themselves, and the ones that do not are your real syllabus.

**Ship the artifact.** Every path above ends with something you produce. Publish it — a blog post, a gist, a repo. The completion rate for self-study with a public artifact is dramatically higher than without one, and the artifact is also the thing you show in an interview.

**Timebox the hard chapters.** Chapters 6, 8, and 20 are where people stall. Give each a fixed budget on the first pass — three hours, say — and move on when it runs out, gaps and all. Come back after the next case-study chapter. The second read is always easier because Chapter 10 shows you why any of it mattered.

---

Next: [Chapter 1 — What a real frontier training run looks like](part-1-frontier/01-what-a-frontier-run-looks-like.md).
