# %% [markdown]
# # Lab 6 — A memory model for distributed training
#
# Companion to [Chapter 6](../book/part-2-pretraining/06-distributed-training.md).
#
# The single most useful skill in pre-training is being able to answer "will this
# fit, and how fast will it go" before spending a cluster-week finding out. This
# lab builds that calculator.
#
# You will:
#
# 1. Compute static memory (weights, gradients, optimizer state) for dense and
#    MoE models, and see why MoE memory scales with *total* parameters while its
#    compute scales with *active* ones.
# 2. Add activation memory, and discover it usually dominates.
# 3. Model ZeRO stages, tensor/pipeline/expert parallelism, and recomputation.
# 4. Search the configuration space for a model + cluster and find what fits.
# 5. Estimate communication volume per step and check whether it can overlap.
#
# Pure arithmetic — no GPU, no PyTorch, runs in about a second. That is the
# point: this calculation is supposed to be cheap enough to do before you write
# any code.
#
# **Predict before you run.** Before section 4, write down what configuration you
# think a 70B model needs on 64 GPUs.

# %%
from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"

GB = 1024**3
TB = 1024**4

# %% [markdown]
# ## 1. The model and the cluster
#
# Two dataclasses. Everything downstream is a function of these.
#
# Note `n_experts=1` means dense: the "expert" is just the FFN.


# %%
@dataclass
class Model:
    """A transformer configuration, dense or MoE."""

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int              # == n_heads for MHA, 1 for MQA, else GQA
    d_ff: int                    # per expert
    vocab_size: int
    n_experts: int = 1           # 1 == dense
    top_k: int = 1               # experts routed per token
    n_shared_experts: int = 0
    tied_embeddings: bool = False
    n_dense_layers: int = 0      # leading layers that use a dense FFN (DeepSeek does this)
    kv_latent_dim: int = 0       # MLA: cache a latent instead of per-head K/V
    kv_rope_dim: int = 0         # MLA: the decoupled RoPE part, which cannot be compressed

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def attention_params(self) -> int:
        """Q, K, V, O projections. GQA shrinks K and V."""
        d, h, kv, hd = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim
        q = d * (h * hd)
        k = d * (kv * hd)
        v = d * (kv * hd)
        o = (h * hd) * d
        return q + k + v + o

    def expert_params(self) -> int:
        """One SwiGLU FFN: THREE matrices, not two."""
        return 3 * self.d_model * self.d_ff

    def embedding_params(self) -> int:
        n = self.vocab_size * self.d_model
        return n if self.tied_embeddings else 2 * n

    @property
    def n_moe_layers(self) -> int:
        return max(0, self.n_layers - self.n_dense_layers)

    def routed_expert_params(self) -> int:
        """Total parameters in routed experts. MoE layers only."""
        return self.n_moe_layers * self.n_experts * self.expert_params()

    def total_params(self) -> int:
        attn = self.n_layers * self.attention_params()
        dense_ffn = self.n_dense_layers * self.expert_params()
        shared = self.n_moe_layers * self.n_shared_experts * self.expert_params()
        return attn + dense_ffn + shared + self.routed_expert_params() + self.embedding_params()

    def active_params(self) -> int:
        """Parameters touched by a single token. Drives FLOPs, not memory."""
        attn = self.n_layers * self.attention_params()
        dense_ffn = self.n_dense_layers * self.expert_params()
        routed = self.n_moe_layers * (self.top_k + self.n_shared_experts) * self.expert_params()
        return attn + dense_ffn + routed + self.embedding_params()

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """MLA caches a latent plus an uncompressible RoPE part; GQA/MHA cache K and V."""
        if self.kv_latent_dim:
            per_layer = self.kv_latent_dim + self.kv_rope_dim
        else:
            per_layer = 2 * self.n_kv_heads * self.head_dim
        return self.n_layers * per_layer * dtype_bytes


@dataclass
class Cluster:
    name: str
    n_gpus: int
    hbm_per_gpu_gb: float
    peak_tflops: float           # dense, at the training precision
    intra_node_gbs: float        # achieved collective bandwidth, GB/s
    inter_node_gbs: float
    gpus_per_node: int = 8


# %% [markdown]
# ## 2. Static memory
#
# The 16-bytes-per-parameter figure, decomposed. This is the arithmetic from
# Chapter 6 §6.1 that appears in essentially every frontier interview.


# %%
@dataclass
class Precision:
    """Bytes per parameter for each tensor the optimizer keeps."""

    name: str
    weights: int = 2             # BF16 working copy
    grads: int = 2               # BF16 gradients
    master: int = 4              # FP32 master weights
    optim_m: int = 4             # Adam first moment
    optim_v: int = 4             # Adam second moment

    @property
    def bytes_per_param(self) -> int:
        return self.weights + self.grads + self.master + self.optim_m + self.optim_v


MIXED_BF16 = Precision("bf16-mixed")
MIXED_FP8 = Precision("fp8-mixed", weights=1, grads=1)     # master/optim stay FP32
# DeepSeek-V3's recipe (V3 report section 3.3.3): AdamW moments in BF16, master
# weights and gradients in FP32. The 2-byte working copy is our assumption; the
# report does not say how the FP8 casts are staged.
DEEPSEEK_V3_RECIPE = Precision("deepseek-v3", weights=2, grads=4, master=4, optim_m=2, optim_v=2)
PURE_BF16 = Precision("bf16-pure", master=0, optim_m=2, optim_v=2)


def static_memory_gb(model: Model, prec: Precision) -> dict[str, float]:
    """Unsharded static state. Uses TOTAL parameters -- every expert has
    optimizer state whether or not it was routed to this step."""
    n = model.total_params()
    return {
        "weights": n * prec.weights / GB,
        "grads": n * prec.grads / GB,
        "master": n * prec.master / GB,
        "optim": n * (prec.optim_m + prec.optim_v) / GB,
        "total": n * prec.bytes_per_param / GB,
    }


# %% [markdown]
# ## 3. Activation memory
#
# The term people forget. It scales with batch x sequence x hidden x layers, it
# does *not* shard across data-parallel ranks, and at realistic sequence lengths
# it usually exceeds the static state.


# %%
# Multiples of d_model retained per token per layer, for the backward pass.
# Chapter 6 §6.3 walks through where these come from.
ACT_MULTIPLES_NO_RECOMPUTE = 16.0
ACT_MULTIPLES_SELECTIVE = 5.0        # keep GEMM outputs, recompute attention internals
ACT_MULTIPLES_FULL = 1.0             # keep only the layer input


RECOMPUTE = {
    "none": (ACT_MULTIPLES_NO_RECOMPUTE, 1.00),
    "selective": (ACT_MULTIPLES_SELECTIVE, 1.10),
    "full": (ACT_MULTIPLES_FULL, 1.33),
}


def activation_memory_gb(
    model: Model,
    micro_batch: int,
    seq_len: int,
    recompute: str = "selective",
    dtype_bytes: int = 2,
    flash_attention: bool = True,
) -> float:
    """Activation memory for ONE micro-batch on ONE pipeline stage's layers.

    Without FlashAttention the O(s^2) attention matrix is materialized, which is
    what made long context impossible before 2022.
    """
    multiples, _ = RECOMPUTE[recompute]
    tokens = micro_batch * seq_len
    per_layer = multiples * model.d_model * tokens * dtype_bytes

    if not flash_attention:
        # The N x N score matrix, per head, per layer.
        per_layer += model.n_heads * micro_batch * seq_len**2 * dtype_bytes

    return model.n_layers * per_layer / GB


# %% [markdown]
# ## 4. Parallelism
#
# Each axis divides a different part of the footprint. Getting this table right
# is most of what Chapter 6 is about.


# %%
@dataclass
class Parallelism:
    tp: int = 1        # tensor: splits weights AND activations within a layer
    pp: int = 1        # pipeline: splits layers across devices
    ep: int = 1        # expert: splits experts (MoE only)
    dp: int = 1        # data: replicates; ZeRO shards optimizer/grad/param state
    zero_stage: int = 1
    recompute: str = "selective"

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.ep * self.dp

    def label(self) -> str:
        return (
            f"TP={self.tp} PP={self.pp} EP={self.ep} DP={self.dp} "
            f"ZeRO-{self.zero_stage} recompute={self.recompute}"
        )


def per_gpu_memory_gb(
    model: Model,
    cluster: Cluster,
    par: Parallelism,
    prec: Precision,
    micro_batch: int,
    seq_len: int,
    flash_attention: bool = True,
    overhead_gb: float = 6.0,      # CUDA context, NCCL buffers, fragmentation
) -> dict[str, float]:
    """Per-GPU memory under a given parallelism configuration."""
    n_total = model.total_params()

    # Expert parameters shard across EP; everything else shards across TP.
    expert_p = model.routed_expert_params()
    other_p = n_total - expert_p
    sharded_params = expert_p / par.ep + other_p / par.tp
    sharded_params /= par.pp        # pipeline splits layers

    # ZeRO shards the data-parallel replicas of each component.
    zero_div_optim = par.dp if par.zero_stage >= 1 else 1
    zero_div_grads = par.dp if par.zero_stage >= 2 else 1
    zero_div_weights = par.dp if par.zero_stage >= 3 else 1

    weights = sharded_params * prec.weights / zero_div_weights
    grads = sharded_params * prec.grads / zero_div_grads
    optim = sharded_params * (prec.master + prec.optim_m + prec.optim_v) / zero_div_optim

    # Activations: only this stage's layers, and TP splits the hidden dimension.
    stage = Model(**{**model.__dict__, "n_layers": max(1, model.n_layers // par.pp)})
    act = activation_memory_gb(
        stage, micro_batch, seq_len, par.recompute, prec.weights, flash_attention
    ) / par.tp

    # 1F1B keeps activations for up to PP in-flight micro-batches.
    act *= min(par.pp, 4) if par.pp > 1 else 1

    return {
        "weights": weights / GB,
        "grads": grads / GB,
        "optim": optim / GB,
        "activations": act,
        "overhead": overhead_gb,
        "total": weights / GB + grads / GB + optim / GB + act + overhead_gb,
        "capacity": cluster.hbm_per_gpu_gb,
        "fits": (weights / GB + grads / GB + optim / GB + act + overhead_gb)
        <= cluster.hbm_per_gpu_gb,
    }


# %% [markdown]
# ## 5. Throughput and communication
#
# Memory tells you whether it fits. This tells you whether it is fast.


# %%
def step_time_s(
    model: Model,
    cluster: Cluster,
    par: Parallelism,
    global_batch_tokens: int,
    mfu: float = 0.40,
    micro_batch: int = 1,
    seq_len: int = 8192,
) -> dict[str, float]:
    """Estimate compute time and communication volume for one optimizer step."""
    _, recompute_cost = RECOMPUTE[par.recompute]

    # Splitting a model very finely makes every GEMM small, and small GEMMs do
    # not saturate tensor cores. Without this term the search happily recommends
    # TP=8 x PP=32 for a 30B model, which no practitioner would run. Chapter 6
    # section 6.8 makes the same point about micro-batch size.
    width_per_rank = model.d_model / par.tp
    layers_per_rank = model.n_layers / par.pp
    width_eff = min(1.0, (width_per_rank / 2048) ** 0.5)     # want >= 2048 wide
    depth_eff = min(1.0, (layers_per_rank / 4) ** 0.25)      # want >= 4 layers/stage
    effective_mfu = mfu * width_eff * depth_eff

    flops = 6 * model.active_params() * global_batch_tokens * recompute_cost
    aggregate = cluster.n_gpus * cluster.peak_tflops * 1e12 * effective_mfu
    compute_s = flops / aggregate

    # --- The distinction that makes or breaks this model -------------------
    # A global batch is split across DP replicas, and each replica's tokens are
    # further split across pipeline stages. Activation-carrying collectives (TP,
    # EP) move only the tokens ONE rank sees, not the whole global batch.
    # Getting this wrong overstates EP cost by the DP degree.
    tokens_per_rank = global_batch_tokens / max(par.dp, 1)
    layers_per_stage = model.n_layers / max(par.pp, 1)
    moe_layers_per_stage = model.n_moe_layers / max(par.pp, 1)

    # DP all-reduce: gradients for the shard this rank owns, ring cost ~2x.
    sharded_params = (
        model.routed_expert_params() / par.ep
        + (model.total_params() - model.routed_expert_params()) / par.tp
    ) / par.pp
    dp_bytes = sharded_params * 2 * 2 * (par.dp - 1) / max(par.dp, 1)
    dp_bw = cluster.inter_node_gbs if par.dp > cluster.gpus_per_node else cluster.intra_node_gbs
    dp_comm_s = dp_bytes / (dp_bw * 1e9) if par.dp > 1 else 0.0

    # TP all-reduce: two per layer, on this stage's layers, over this rank's
    # tokens. On the critical path -- nothing to overlap it with.
    tp_comm_s = 0.0
    if par.tp > 1:
        per_allreduce = tokens_per_rank * model.d_model * 2
        tp_bytes = per_allreduce * 2 * layers_per_stage * 2 * (par.tp - 1) / par.tp
        tp_bw = cluster.intra_node_gbs if par.tp <= cluster.gpus_per_node else cluster.inter_node_gbs
        tp_comm_s = tp_bytes / (tp_bw * 1e9)

    # EP all-to-all: dispatch + combine, per MoE layer, over this rank's tokens.
    ep_comm_s = 0.0
    if par.ep > 1 and model.n_experts > 1:
        copies = tokens_per_rank * model.top_k
        ep_bytes = 2 * copies * model.d_model * 2 * moe_layers_per_stage * (par.ep - 1) / par.ep
        ep_bw = cluster.intra_node_gbs if par.ep <= cluster.gpus_per_node else cluster.inter_node_gbs
        ep_comm_s = ep_bytes / (ep_bw * 1e9)

    # Pipeline bubble: (P-1)/(M+P-1), where M is micro-batches per DP replica.
    # Nothing else in this model penalizes deep pipelines, and without this term
    # the search happily recommends PP=32 for a 30B model.
    bubble_frac = 0.0
    if par.pp > 1:
        m = max(1, int(tokens_per_rank / (micro_batch * seq_len)))
        bubble_frac = (par.pp - 1) / (m + par.pp - 1)
    bubble_s = compute_s * bubble_frac / max(1e-9, 1 - bubble_frac)

    # DP overlaps with the backward pass; TP, EP, and the bubble do not.
    exposed = max(0.0, dp_comm_s - compute_s) + tp_comm_s + ep_comm_s + bubble_s

    return {
        "compute_s": compute_s,
        "dp_comm_s": dp_comm_s,
        "tp_comm_s": tp_comm_s,
        "ep_comm_s": ep_comm_s,
        "bubble_s": bubble_s,
        "bubble_frac": bubble_frac,
        "exposed_comm_s": exposed,
        "step_s": compute_s + exposed,
        "kernel_eff": width_eff * depth_eff,
        "effective_mfu": effective_mfu * compute_s / (compute_s + exposed),
    }


# %% [markdown]
# ## 6. Reference configurations

# %%
LLAMA3_70B = Model(
    name="Llama-3-70B",
    n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8,
    d_ff=28672, vocab_size=128256,
)

DEEPSEEK_V3 = Model(
    name="DeepSeek-V3",
    n_layers=61, d_model=7168, n_heads=128, n_kv_heads=128,
    d_ff=2048, vocab_size=129280,
    n_experts=256, top_k=8, n_shared_experts=1,
    n_dense_layers=3,            # the first layers use a dense FFN
    kv_latent_dim=512, kv_rope_dim=64,   # MLA, not GQA
)

MOE_30B = Model(
    name="30B-A3.5B MoE",
    n_layers=40, d_model=2560, n_heads=20, n_kv_heads=4,
    d_ff=1408, vocab_size=131072,
    n_experts=64, top_k=6, n_shared_experts=1,
)

H100_64 = Cluster("64x H100", 64, 80.0, 990.0, 450.0, 40.0)
H100_1024 = Cluster("1024x H100", 1024, 80.0, 990.0, 450.0, 40.0)
# The H800's NVLink is cut down relative to the H100's. These are the numbers
# DeepSeek report for their own cluster (V3 report section 3.2.2): NVLink
# 160 GB/s, InfiniBand 50 GB/s. See book/appendix/fact-sheets/deepseek-v3.md.
H800_2048 = Cluster("2048x H800", 2048, 80.0, 990.0, 160.0, 50.0)

print("Model parameter counts")
print("-" * 74)
print(f"{'model':<20} {'total':>14} {'active':>14} {'sparsity':>10} {'KV/token':>12}")
for m in (LLAMA3_70B, DEEPSEEK_V3, MOE_30B):
    print(
        f"{m.name:<20} {m.total_params()/1e9:>12.1f}B {m.active_params()/1e9:>12.1f}B "
        f"{m.total_params()/m.active_params():>9.1f}x {m.kv_bytes_per_token()/1024:>10.0f} KB"
    )

# %% [markdown]
# Check these against Chapter 5 and Chapter 10. Llama-3-70B should come out near
# 70B; DeepSeek-V3's routed experts should account for essentially all of its
# 671B, which is the arithmetic behind "MoE memory scales with total parameters."

# %%
print("\nStatic memory, unsharded (bf16 mixed precision, 16 B/param)")
print("-" * 74)
for m in (LLAMA3_70B, DEEPSEEK_V3, MOE_30B):
    s = static_memory_gb(m, MIXED_BF16)
    print(
        f"{m.name:<20} weights {s['weights']:>7.0f}  grads {s['grads']:>7.0f}  "
        f"optim+master {s['master']+s['optim']:>7.0f}  TOTAL {s['total']:>8.0f} GB"
    )
    print(f"{'':<20} = {s['total']/80:.0f} H100s for parameters alone")

# %% [markdown]
# ## 7. Activations dominate
#
# The result that surprises people.

# %%
print("\nStatic vs activation memory, Llama-3-70B, micro-batch 1")
print("-" * 74)
print(f"{'seq len':>10} {'static (ZeRO-3/64)':>20} {'activations':>14} {'ratio':>8}")
static_sharded = static_memory_gb(LLAMA3_70B, MIXED_BF16)["total"] / 64
for seq in (2048, 8192, 32768, 131072):
    act = activation_memory_gb(LLAMA3_70B, 1, seq, "selective")
    print(f"{seq:>10} {static_sharded:>18.1f} GB {act:>12.1f} GB {act/static_sharded:>7.1f}x")

print("\nWithout FlashAttention (the O(s^2) term is materialized):")
for seq in (8192, 32768):
    act = activation_memory_gb(LLAMA3_70B, 1, seq, "selective", flash_attention=False)
    print(f"{seq:>10} {'':>18}    {act:>12.0f} GB")
print("\nThat second table is why FlashAttention made long context possible at all.")

# %% [markdown]
# ## 8. Search the configuration space
#
# Given a model and a cluster, find every parallelism configuration that fits,
# and rank by estimated step time.
#
# **Predict first:** what TP/PP/EP/DP do you expect for Llama-3-70B on 64 H100s
# at 8K context?


# %%
def search(
    model: Model,
    cluster: Cluster,
    prec: Precision,
    seq_len: int,
    micro_batch: int = 1,
    global_batch_tokens: int = 4_194_304,
    max_results: int = 8,
) -> list[tuple[Parallelism, dict, dict]]:
    """Enumerate valid configurations, keep those that fit, rank by step time."""
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    valid = []

    for tp, pp, ep in itertools.product(powers, repeat=3):
        if tp > cluster.gpus_per_node:
            continue                                  # TP must stay on NVLink
        if ep > 1 and model.n_experts == 1:
            continue                                  # EP is meaningless for dense
        if ep > 1 and model.n_experts % ep != 0:
            continue
        if pp > model.n_layers:
            continue
        head = tp * pp * ep
        if head > cluster.n_gpus or cluster.n_gpus % head != 0:
            continue

        dp = cluster.n_gpus // head
        for zero in (1, 2, 3):
            for recompute in ("selective", "full"):
                par = Parallelism(tp, pp, ep, dp, zero, recompute)
                mem = per_gpu_memory_gb(model, cluster, par, prec, micro_batch, seq_len)
                if not mem["fits"]:
                    continue
                perf = step_time_s(
                    model, cluster, par, global_batch_tokens,
                    micro_batch=micro_batch, seq_len=seq_len,
                )
                valid.append((par, mem, perf))

    # Rank by step time, then break ties toward the SIMPLEST configuration:
    # fewer parallelism axes and a lower ZeRO stage are easier to debug and to
    # reason about, and at equal predicted throughput that is what you want.
    valid.sort(
        key=lambda x: (
            round(x[2]["step_s"], 3),
            x[0].tp * x[0].pp * x[0].ep,      # prefer fewer non-DP axes
            x[0].zero_stage,
            x[0].recompute != "selective",
        )
    )
    return valid[:max_results]


# DeepSeek-V3's own global batch: 15,360 sequences of 4K (V3 report section 4.2).
V3_BATCH_TOKENS = 15_360 * 4096

for model, cluster, seq, prec, batch in [
    (LLAMA3_70B, H100_64, 8192, MIXED_BF16, 4_194_304),
    (MOE_30B, H100_1024, 8192, MIXED_BF16, 4_194_304),
    (DEEPSEEK_V3, H800_2048, 4096, DEEPSEEK_V3_RECIPE, V3_BATCH_TOKENS),
]:
    print(f"\n{'='*74}\n{model.name} on {cluster.name}, seq={seq}, {prec.name}, "
          f"batch={batch/1e6:.1f}M tokens\n{'='*74}")
    results = search(model, cluster, prec, seq, global_batch_tokens=batch,
                     max_results=3 if SMOKE else 6)
    if not results:
        print("  NOTHING FITS. Try a shorter sequence, more GPUs, or full recompute.")
        continue
    print(f"{'configuration':<52} {'mem GB':>8} {'step s':>8} {'MFU':>6}")
    for par, mem, perf in results:
        print(
            f"{par.label():<52} {mem['total']:>7.1f} {perf['step_s']:>8.2f} "
            f"{perf['effective_mfu']*100:>5.1f}%"
        )
    print(
        "  (This ranking is a model, not a measurement. It captures memory,\n"
        "   communication volume, the pipeline bubble, and a crude GEMM-efficiency\n"
        "   penalty. It does NOT capture collective latency, expert load imbalance,\n"
        "   kernel quality, or scheduling overhead -- so treat a recommendation of\n"
        "   extreme TP or PP with suspicion and measure before committing.)"
    )

# %% [markdown]
# ### 8b. What DeepSeek actually ran, and why the calculator disagrees
#
# DeepSeek-V3 trained with PP=16, EP=64 spanning 8 nodes, ZeRO-1 data
# parallelism and no TP (V3 report section 3.2; the book's fact sheet). The
# calculator above does not pick that. Compare the two, and read the gap as a
# statement about what this model leaves out, not about DeepSeek.

# %%
V3_ACTUAL = Parallelism(tp=1, pp=16, ep=64, dp=H800_2048.n_gpus // (16 * 64), zero_stage=1)
v3_mem = per_gpu_memory_gb(DEEPSEEK_V3, H800_2048, V3_ACTUAL, DEEPSEEK_V3_RECIPE, 1, 4096)
v3_perf = step_time_s(DEEPSEEK_V3, H800_2048, V3_ACTUAL, V3_BATCH_TOKENS, seq_len=4096)
best_par, best_mem, best_perf = search(DEEPSEEK_V3, H800_2048, DEEPSEEK_V3_RECIPE, 4096,
                                      global_batch_tokens=V3_BATCH_TOKENS, max_results=1)[0]

# The model charges the EP all-to-all as fully exposed. DualPipe's purpose is to
# overlap it with compute (V3 report section 3.2.1), so bound the effect by
# hiding it entirely.
v3_hidden_step = v3_perf["step_s"] - v3_perf["ep_comm_s"]
best_hidden_step = best_perf["step_s"] - best_perf["ep_comm_s"]

print(f"{'configuration':<44} {'fits':>5} {'mem GB':>7} {'step s':>7} {'EP comm s':>9} {'EP hidden':>9}")
for label, par, mem, perf, hidden in [
    ("calculator's pick: " + f"TP={best_par.tp} PP={best_par.pp} EP={best_par.ep}", best_par, best_mem, best_perf, best_hidden_step),
    ("DeepSeek's layout: TP=1 PP=16 EP=64", V3_ACTUAL, v3_mem, v3_perf, v3_hidden_step),
]:
    print(f"{label:<44} {str(mem['fits']):>5} {mem['total']:>7.1f} {perf['step_s']:>7.2f} "
          f"{perf['ep_comm_s']:>9.2f} {hidden:>9.2f}")

ratio_exposed = v3_perf["step_s"] / best_perf["step_s"]
ratio_hidden = v3_hidden_step / best_hidden_step
print(f"\nWith EP charged as exposed, DeepSeek's layout is {ratio_exposed:.2f}x the pick's step time.")
print(f"With EP overlapped (DualPipe's goal), the ratio is {ratio_hidden:.2f}x.")
print("What this calculator omits, each of which V3 relies on:")
print("  - overlap of the all-to-all with compute (DualPipe, report 3.2.1);")
print("  - node-limited routing: each token reaches at most 4 nodes over InfiniBand")
print("    and fans out over NVLink (report 3.2.2); here every token pays the slow link;")

# %% [markdown]
# ## 9. Where the communication goes
#
# The exposed-communication column is what separates a 40% MFU run from a 20%
# one. It is why EP *defaults* to living inside a node, and why DeepSeek, which
# ran it across 8 nodes, had to build machinery to hide it.

# %%
print("\nCommunication breakdown, MoE-30B on 1024x H100, 4.2M-token global batch")
print("-" * 74)
print(f"{'configuration':<40} {'compute':>9} {'DP':>8} {'TP':>8} {'EP':>8} {'exposed':>9}")
for tp, ep in [(1, 8), (2, 8), (8, 1), (1, 64)]:
    head = tp * ep
    if H100_1024.n_gpus % head:
        continue
    par = Parallelism(tp=tp, pp=4, ep=ep, dp=H100_1024.n_gpus // (head * 4))
    perf = step_time_s(MOE_30B, H100_1024, par, 4_194_304, seq_len=8192)
    print(
        f"{par.label()[:39]:<40} {perf['compute_s']:>8.2f}s {perf['dp_comm_s']:>7.2f}s "
        f"{perf['tp_comm_s']:>7.2f}s {perf['ep_comm_s']:>7.2f}s {perf['exposed_comm_s']:>8.2f}s"
    )

print("\nNote EP=64 spans 8 nodes, so its all-to-all runs at inter-node bandwidth.")
print("DeepSeek-V3 ran exactly that, and built DualPipe to hide it (Chapter 6 section 6.10).")

# %% [markdown]
# ## 10. Things to try
#
# Each has a common wrong prediction.
#
# **1. Set `flash_attention=False` and re-run section 7 at 32K.**
# *Common prediction:* activation memory grows 4x, matching the sequence length.
# *What happens:* the linear term grows 4x and the O(s^2) term grows 16x, and the
# second dominates completely. FlashAttention is not a speed optimization here;
# it is the difference between possible and impossible.
#
# **2. Search Llama-3-70B on 8 GPUs instead of 64.**
# *Common prediction:* ZeRO-3 handles it.
# *What happens:* ZeRO-3 across 8 ranks gives 1,120/8 = 140 GB per GPU — still
# over 80. ZeRO's benefit scales with the DP degree, so it is weakest exactly
# where you have few GPUs.
#
# **3. Set `ep=1` for DeepSeek-V3 and search again.**
# *Common prediction:* ZeRO-3 substitutes for expert parallelism.
# *What happens:* nothing fits, or what fits is unusably slow. With EP the tokens
# move to the experts (~470 MB/layer); without it, ZeRO-3 must all-gather the
# full 1.3 TB parameter set per layer. For MoE, move the data to the parameters.
#
# **4. Change `MIXED_BF16` to `MIXED_FP8` and re-run section 6.**
# *Common prediction:* memory halves.
# *What happens:* it drops by 2/16 = 12.5%, not 50% — because in this preset the
# FP32 master weights and both Adam moments stay FP32, and they are 12 of the 16
# bytes. FP8's win is throughput and activation memory. DeepSeek-V3 went further
# on the optimizer side: `DEEPSEEK_V3_RECIPE` stores both moments in BF16, which
# the report found cost nothing measurable. Try it on Llama-3-70B.
#
# **5. Raise `micro_batch` from 1 to 4 and re-search.**
# *Common prediction:* memory rises 4x across the board.
# *What happens:* only the activation term scales; static memory is unchanged.
# Whether that breaks the configuration depends entirely on which term dominated,
# which is section 7's question.

# %% [markdown]
# ## What to take away
#
# 1. **MoE memory scales with total parameters, compute with active ones.** Every
#    expert has optimizer state whether or not it was routed to.
# 2. **16 bytes per parameter** — 2 weights + 2 grads + 4 master + 4 + 4 Adam.
#    Memorize the decomposition, not the number.
# 3. **Activations usually dominate**, and they do not shard across data-parallel
#    ranks. ZeRO helps the smaller term.
# 4. **TP belongs inside a node, and so does EP by default**; DP and PP tolerate
#    the slow network. The exposed-communication column shows why, and section
#    8b shows what DeepSeek had to add to run EP across nodes anyway.
# 5. **This calculation costs a second.** Do it before writing code, not after a
#    cluster-week.
#
# Next: [`lab07_collective_bandwidth`](lab07_collective_bandwidth.py), which
# models the communication side in more detail.
