# Chapter 7: Cluster reality — topology, NCCL, fault tolerance

> Reading time: ~60 minutes. By the end of this chapter you should understand the physical substrate of a frontier training run, the software that drives it, and the failure modes that define daily life for the engineers who keep it running.

*Current as of early 2025.*

## 7.1 The world between the silicon and the script

Chapter 6 covered the *logical* structure of a distributed training run: tensor parallelism, pipeline parallelism, data parallelism, expert parallelism, and the collectives that tie them together. That chapter assumed the network and the nodes were there. This chapter is about what is actually there.

A frontier training run is, at its physical core, 2,000 to 100,000 GPUs, the wires between them, the switches that route those wires, the racks that hold the switches, the building that holds the racks, the substation that feeds the building, the cooling plant that takes the heat away, the parallel filesystem that streams the data, and the monitoring system that watches all of it. The training script that "just runs PyTorch" is a thin layer over a much more complicated world, and most frontier training jobs spend the majority of their wall-clock time dealing with that world, not with the math.

The DeepSeek-V3 pre-training run [\[1\]](../appendix/b-references.md#1-deepseek-v3) used 2,048 NVIDIA H800 GPUs connected in 256 nodes of 8. The Llama-3 405B pre-training run [\[5\]](../appendix/b-references.md#5-llama-3) used roughly 16,000 H100s. The xAI Colossus cluster [\[40\]](../appendix/b-references.md#40-xai-colossus) reached 100,000 H100s in a single site in late 2024. Meta has announced plans for a 1.3M-H100-equivalent cluster. Anthropic's Project Rainier [\[41\]](../appendix/b-references.md#41-aws-trainium-2--project-rainier) moves away from NVIDIA entirely, onto AWS Trainium 2, with the cluster itself numbering in the hundreds of thousands of accelerators. We are well past the era when "a frontier run" fits in a single rack, or a single room, or even a single building. It is a piece of civil infrastructure, and the engineering practice around it has more in common with operating a power plant than with writing a script.

This chapter walks that infrastructure, top to bottom, with the failures and the recovery loop that the public papers mostly leave out. We anchor on DeepSeek-V3's 2,048-GPU layout [\[1\]](../appendix/b-references.md#1-deepseek-v3), the xAI Colossus numbers [\[40\]](../appendix/b-references.md#40-xai-colossus), and Llama-3's training infrastructure [\[5\]](../appendix/b-references.md#5-llama-3). The Qwen3 report describes no hardware at all ([fact sheet](../appendix/fact-sheets/qwen3.md#pre-training-stages)), so Qwen3 does not appear here.

## 7.2 The cluster as a system

A frontier cluster is a hierarchy, and at every level of the hierarchy the bandwidth and the failure mode look different. The terminology is not entirely standard across vendors, but the structure is consistent.

**The node.** The basic unit. For NVIDIA-based clusters in 2024–2025, a node is one DGX H100 or DGX H200 box: 8 GPUs, connected to each other by NVLink and NVSwitch, with 4 or 8 InfiniBand or Ethernet network cards (HCAs) facing outward. Each H100 has 80 GB of HBM3 and delivers roughly 3.35 TB/s of HBM bandwidth to its local SMs. The 8 GPUs inside the box are connected to each other by 18 NVLink links, totaling 900 GB/s of all-to-all bandwidth per GPU. This is the highest-bandwidth link in the entire system, and it is the only link where every GPU can talk to every other GPU at full speed. Inside a node, the topology is "all GPUs fully connected by NVSwitch."

**The rack.** A rack holds 4 to 8 nodes (so 32 to 64 GPUs). The nodes in a rack share one or two leaf InfiniBand switches. Within a rack, every GPU can reach every other GPU at InfiniBand speed (typically 400 Gb/s NDR per port for H100-era clusters). Between GPUs in the same rack but different nodes, the bandwidth is lower than intra-node NVLink — typically by a factor of 5 to 10 — but it is still direct, with a single switch hop. The rack is the level at which a tensor-parallel group typically lives: one TP group = one node (8 GPUs), because the all-reduce inside a tensor-parallel layer is bandwidth-bound and you want it on the fastest available links.

**The super-pod / pod.** A super-pod is a collection of racks that share a non-blocking (or near-non-blocking) network fabric. The xAI Memphis cluster is a single super-pod of 100,000 H100s, but it is built from many smaller sub-pods internally [\[40\]](../appendix/b-references.md#40-xai-colossus). The classic InfiniBand "7-rail topology" — which we cover in §7.3 — is the canonical super-pod design.

**The cluster.** The full site. For DeepSeek-V3, the cluster is 256 nodes = 2,048 GPUs in a single InfiniBand fabric [\[1\]](../appendix/b-references.md#1-deepseek-v3). For Llama-3 405B, the run used up to 16,000 H100s of a 24,000-GPU RoCE cluster, built as a three-layer Clos network of 3,072-GPU pods [\[5\]](../appendix/b-references.md#5-llama-3) ([fact sheet](../appendix/fact-sheets/llama-3.md#infrastructure-and-parallelism)). For Colossus, the cluster is 100,000 H100s in a single 150 MW building [\[40\]](../appendix/b-references.md#40-xai-colossus). For the announced Meta "1.3M H100 equivalent" supercluster, the cluster is multiple buildings, multiple substations, and multiple gigawatts.

**The data center.** A cluster typically lives in a single data center, with its own substation, its own cooling plant, and its own network connection to the rest of the world. Larger labs have multiple data centers and stitch them together with long-haul fiber, but most pre-training runs are run inside a single data center because the latency between data centers is too high for synchronous collective communication.

The DeepSeek-V3 paper's hardware description is short, and typical of what frontier runs publish [\[1\]](../appendix/b-references.md#1-deepseek-v3):

> *"DeepSeek-V3 is trained on a cluster equipped with 2048 NVIDIA H800 GPUs. Each node in the H800 cluster contains 8 GPUs connected by NVLink and NVSwitch within nodes."* (§3.1)

The report's one other hardware number, in §3.2.2, is the one that matters: on this cluster, NVLink offers **160 GB/s**, about **3.2×** InfiniBand's 50 GB/s ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#cluster-and-interconnect)). The H800 is an export-compliant H100 with cut-down NVLink, so the intra/inter-node gap is far smaller than the ~18× of the H100 figures below. Everything else in this chapter is the part that does not get published.

## 7.3 The network topology

The bandwidth hierarchy inside a frontier cluster is brutal and is the single most important fact about the system. Numbers for an H100 cluster with NDR InfiniBand, in approximate round-trip bandwidths per GPU:

| Link | Bandwidth per GPU | Latency |
|---|---|---|
| GPU HBM (local) | ~3,350 GB/s | ~0.5 µs |
| NVLink (intra-node, to peer GPU) | 900 GB/s (bidirectional aggregate per GPU) | ~1–2 µs |
| InfiniBand NDR (intra-rack, to peer GPU) | 400 Gb/s ≈ 50 GB/s | ~2–5 µs |
| InfiniBand NDR (inter-rack, same pod) | 400 Gb/s ≈ 50 GB/s | ~3–8 µs |
| InfiniBand NDR (inter-pod, same site) | 50 GB/s (with oversubscription) | ~5–15 µs |

The intra-node NVLink is roughly 18× faster than the inter-node InfiniBand per GPU. This is the number every parallelism choice has to respect. It is why tensor parallelism lives inside a node (the all-reduce inside a TP layer is bandwidth-bound and you do not want to put it on a 18×-slower link), and why pipeline and data parallelism live across nodes (they are less bandwidth-bound and more latency-tolerant).

The ratio is also why the MoE all-to-all *usually* stays inside a node: it is large and bandwidth-hungry, and you want it on the fastest links. DeepSeek-V3 is the instructive exception [\[1\]](../appendix/b-references.md#1-deepseek-v3). Its expert parallelism spans 8 nodes, on a cluster whose NVLink-to-InfiniBand gap is about 3.2×, not 18×, because the H800's NVLink is cut down ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#cluster-and-interconnect)). DeepSeek hid the cross-node all-to-all behind compute and capped each token at 4 nodes. It is the reason Llama-3 orders its parallelism [TP, CP, PP, DP], with the innermost, most bandwidth-hungry dimension "usually constrained to within the same server" [\[5\]](../appendix/b-references.md#5-llama-3) ([fact sheet](../appendix/fact-sheets/llama-3.md#infrastructure-and-parallelism)). It is the reason "NVLink-only" research prototypes (e.g., a 16-GPU workstation with two NVLink-connected DGX nodes) often do not predict real cluster behavior — the 18× cliff is invisible until you try to go across it.

### 7.3.1 NVLink and NVSwitch

NVLink is a NVIDIA-proprietary high-bandwidth point-to-point link between GPUs (and, on newer chips, between GPUs and the BlueField DPUs or Grace CPUs). On Hopper-generation H100 / H200, each GPU has 18 NVLink 4.0 links, each at 50 GB/s unidirectional (so 100 GB/s bidirectional). The total per-GPU NVLink bandwidth is therefore 900 GB/s, all going through the on-board NVSwitch. The NVSwitch is an 8-port switch chip; a DGX H100 has 4 NVSwitches on the baseboard, fully connecting the 8 GPUs.

The bandwidth hierarchy within a single DGX H100 node, measured in real workloads, is roughly:

- **HBM to SMs**: 3.35 TB/s per GPU (the local-compute ceiling)
- **NVLink peer-to-peer**: 900 GB/s per GPU, fully connected (all 8 GPUs)
- **PCIe Gen5 to CPU/host**: 128 GB/s per GPU, but shared across the node's 8 GPUs in practice
- **InfiniBand NDR (out of node)**: 4 × 400 Gb/s = 200 GB/s per GPU (4 HCAs), 400 Gb/s effective

The number that matters is that the NVLink between two GPUs in the same node is roughly 18× faster than the InfiniBand between two GPUs in different nodes. This ratio drives the parallelism strategy: anything that does a lot of GPU-to-GPU communication wants to live on the NVLink.

### 7.3.2 InfiniBand NDR and the 7-rail topology

Inter-node communication in a frontier cluster is almost always InfiniBand NDR (Next Data Rate, 400 Gb/s per port, sometimes 200 Gb/s called "HDR" on older clusters). The HCAs on the GPU are typically ConnectX-7 (or, on the latest generation, ConnectX-8 for NDR). The HCAs and the GPUs are on the same PCIe root complex, and the GPUDirect RDMA path lets the HCA read and write GPU memory directly without going through host RAM, which is essential for NCCL performance.

The classic super-pod topology for InfiniBand is the **rail-optimized** or **"7-rail"** topology. The idea: 8 GPUs in a node each have their own dedicated HCA (so 8 HCAs per node), and the 8 HCAs are each connected to a different leaf switch. Within a rack, all the HCA-1s from all the nodes connect to switch 1, all the HCA-2s to switch 2, and so on. The result is that two GPUs in the same rail (same HCA index) across different nodes are 1 hop away, while two GPUs in different rails are at most 2 hops away. This guarantees that the most common communication pattern — collective ops with regular strides — is a single switch hop.

For a 256-node pod (like DeepSeek-V3's [\[1\]](../appendix/b-references.md#1-deepseek-v3)), the 7-rail topology looks like:

```
        ┌─────── leaf 0 ───────┐
node 0: GPU0 ─┐                ├─ all node i's GPU0
node 1: GPU0 ─┤                ├─ ... 
...           ├────────────────┤
node 255: GPU0┘                │
        └─────────────────────┘
        (similarly for leaves 1..7 — total 8 leaves for an 8-GPU node)
```

For larger super-pods, leaves are aggregated into a spine layer (a 2-tier fat-tree), with the spine typically sized to give 1:1 oversubscription within a super-pod. The xAI Colossus announcement mentions "InfiniBand fabric" without specifying the exact topology [\[40\]](../appendix/b-references.md#40-xai-colossus), but public talks and the standard NVIDIA reference designs for 100k-GPU clusters use a 2- or 3-tier fat-tree with rail-optimized leaves. We say this is the design because the alternative — a Clos network with no rail alignment — is dramatically slower for the typical stride patterns of NCCL collectives.

### 7.3.3 RoCE on Ethernet

The alternative to InfiniBand is RDMA over Converged Ethernet (RoCE), most commonly using Spectrum-X switches from NVIDIA. RoCE v2 runs RDMA over a standard Ethernet L3 network and is significantly cheaper to deploy at scale than InfiniBand. The bandwidth and latency are comparable at the link level, but RoCE is more sensitive to network configuration (PFC, ECN, congestion control) and the operational practice is less mature.

The Llama-3 paper [\[5\]](../appendix/b-references.md#5-llama-3) reports that Meta's 16k-H100 cluster uses Ethernet (RoCE), not InfiniBand, on a fabric built from Arista 7800 and Minipack2 switches ([fact sheet](../appendix/fact-sheets/llama-3.md#infrastructure-and-parallelism)). This is a deliberate trade: Ethernet is cheaper and easier to source, at the cost of more tuning work to get the same lossless RDMA behavior. Most hyperscaler-scale clusters are now Ethernet/RoCE; most "research cluster" purchases for labs that can afford it are still InfiniBand because the operational risk is lower. Anthropic's Project Rainier uses a custom interconnect (NeuronLink) because the accelerator itself is non-NVIDIA [\[41\]](../appendix/b-references.md#41-aws-trainium-2--project-rainier).

## 7.4 Topology-aware placement

Once the physical topology is fixed, the placement of the logical parallelism groups onto the physical topology is the next decision. Done right, it can yield 10–30% better collective throughput; done wrong, it can yield NCCL hangs and silent slowdowns that look like model issues.

NCCL's topology detection is the heart of the system. When NCCL initializes, it queries the system's PCIe topology (via `libibverbs` and the kernel's NUMA / sysfs tree), enumerates the GPUs and HCAs, and builds a graph of the available communication paths. For a 2-GPU all-reduce, it picks the fastest path (usually NVLink for intra-node, IB for inter-node). For an N-GPU all-reduce, it picks an algorithm (ring, tree, double-binary-tree) and a path through that algorithm that minimizes total time. The detection is automatic, but it is sensitive to the environment: an HCA that is misconfigured (wrong `NCCL_IB_HCA` setting, wrong subnet) is invisible to NCCL and degrades performance silently.

The DeepSeek-V3 parallelism layout [\[1\]](../appendix/b-references.md#1-deepseek-v3) shows placement *co-designed* with the network rather than simply fitted to it ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#parallelism-layout)):

- **No TP** anywhere.
- **EP = 64, spanning 8 nodes**, so the MoE all-to-all crosses InfiniBand.
- **PP = 16**, over the 61 transformer layers.
- **ZeRO-1 data parallelism** over the remaining ranks.

The rail-aware part is in how the all-to-all moves. A token bound for experts on another node first crosses InfiniBand to the GPU with the **same in-node index** on the target node, which is the same rail. Then it is forwarded over NVLink to the GPU that holds its expert. Each token may target at most 4 nodes, so InfiniBand traffic is bounded while NVLink absorbs the fan-out. DeepSeek report that this lets the two links run fully overlapped. The layout only works because the kernel, the routing rule and the topology were designed together. The report does not publish its physical rank-to-node placement; anything more specific than the above would be a guess.

A real cluster bring-up script for a frontier run looks like this (highly simplified from public Megatron / DeepSpeed launch scripts):

```bash
# /opt/cluster/bin/launch_frontier_run.sh
# Set NCCL environment for an H100 / NDR cluster
export NCCL_DEBUG=WARN                # INFO in production for the first hour, then WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5           # GPUDirect RDMA, NVLink/NDR
export NCCL_P2P_LEVEL=NVL             # prefer NVLink peer-to-peer
export NCCL_BUFFSIZE=8388608          # 8 MiB chunk size for collectives
export NCCL_NTHREADS=512              # NCCL threads per process
export NCCL_MIN_NCHANNELS=112         # NDR has 16 lanes, use them all
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_CQ=1
export NCCL_IB_USE_INLINE=0

# Force NUMA binding — critical for cross-socket H100 nodes
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# For rail-optimized all-reduce, the data-parallel group should align
# with the IB rail index. PyTorch DDP / FSDP need this hint:
export NCCL_TOPO_FILE=/opt/cluster/etc/topo.xml

# Launch
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=256 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    /opt/run/train.py --config /opt/run/configs/v3_2048.yaml
```

The `NCCL_IB_HCA` line in particular is something every frontier engineer learns to set by hand. The default ("all HCAs") is wrong for most clusters because the node has HCAs on different subnets, some of which are not on the cluster's high-performance fabric, and NCCL will silently include them. Setting it explicitly to the right list of HCAs (matching the rail index) is the standard way to keep NCCL on the fast path.

A small topology-detection script that engineers use during cluster bring-up:

```python
#!/usr/bin/env python3
"""
cluster_topo_dump.py — print the NCCL-visible topology of this node.
Run on every node of a cluster; diff the outputs to find misconfigurations.
"""
import os
import subprocess
import json

def detect_hcas():
    """Enumerate InfiniBand HCAs and their state."""
    out = subprocess.check_output(["ibstat"], text=True)
    hcas = []
    current = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("CA "):
            if current:
                hcas.append(current)
            current = {"name": line.split("'")[1]}
        elif ":" in line:
            k, v = line.split(":", 1)
            current[k.strip()] = v.strip()
    if current:
        hcas.append(current)
    return hcas

def detect_gpus():
    """Enumerate GPUs and their NVLink peers."""
    out = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"], text=True
    )
    return out

def detect_numa():
    """NUMA topology of the host."""
    out = subprocess.check_output(["lscpu"], text=True)
    return out

if __name__ == "__main__":
    info = {
        "node": os.uname().nodename,
        "hcas": detect_hcas(),
        "gpu_topo": detect_gpus(),
        "numa": detect_numa(),
    }
    print(json.dumps(info, indent=2))
```

The script is run on every node during cluster bring-up. If the GPU topo matrix shows "NVLink" everywhere within the node and "SYS" (system, slow) for cross-node, that is the expected layout. If the matrix shows "PIX" or "NODE" where you expected NVLink, you have a bad cable or a bad NVSwitch port and you should RMA the node before you start the run.

## 7.5 NCCL internals

NCCL (NVIDIA Collective Communication Library) [\[42\]](../appendix/b-references.md#42-nccl-documentation) is the standard library for GPU collective communication on NVIDIA hardware. Every front-end framework (PyTorch DDP, FSDP, Megatron-LM, DeepSpeed, JAX with `pjit`) eventually calls NCCL for its all-reduce, all-gather, reduce-scatter, all-to-all, and broadcast operations. When a frontier training job is "fast" or "slow," nine times out of ten the difference is in how NCCL is being driven.

### 7.5.1 Collective algorithms

NCCL implements each collective with a choice of algorithms, and the choice depends on the tensor size, the number of GPUs, and the topology. The three algorithms you should know:

- **Ring.** All GPUs form a logical ring. Each GPU sends chunks of the tensor to the next GPU in the ring while receiving chunks from the previous GPU. Total data per GPU: $N-1$ sends and $N-1$ receives, each of size $S/N$ for a tensor of size $S$ across $N$ GPUs. Bandwidth-optimal (uses the full link bandwidth) but latency scales linearly with $N$ and is sensitive to the slowest link in the ring.
- **Tree.** All GPUs form a binomial or binary tree. The reduce or broadcast proceeds from the leaves to the root (or vice versa). Lower latency than ring for moderate $N$ (because the depth is $\log N$) but uses more links and can be more sensitive to link quality.
- **Double-binary-tree.** Two binary trees used in parallel (one for even-indexed chunks, one for odd-indexed chunks), with a small "split" at the end. This is the algorithm NCCL prefers for large all-reduce on a fat-tree InfiniBand fabric, because it gets the latency of tree and the bandwidth of ring. The cost is more complex scheduling.

For an all-reduce of a 1 GB tensor across 2,048 GPUs, the per-GPU data volume is ~1 GB, and at 50 GB/s effective inter-node bandwidth, the all-reduce should take ~20 ms. In practice, a well-tuned run sees 25–40 ms; a poorly tuned run sees 200+ ms, and that is the gap that separates a healthy run from a stuck run.

NCCL picks the algorithm automatically based on heuristics (tensor size, GPU count, link count, NVLink availability). You can override the choice with `NCCL_ALGO` (`=ring`, `=tree`, `=collnet` for some operations). Most frontier engineers do not touch the algorithm choice; the heuristics are right 95% of the time, and forcing an algorithm is a debugging move, not a production move.

### 7.5.2 Environment variables

The environment variables in §7.4 are the ones every frontier engineer has memorized. A short annotated list:

- `NCCL_DEBUG=INFO` — turn on verbose logging. Every collective, every algorithm choice, every link is logged. **Always set this for the first hour of a new run**, then demote to `WARN`. The log files are massive.
- `NCCL_DEBUG_SUBSYS=ALL` or `=COLL,NET,GRAPH` — finer-grained debug. Useful when you suspect a specific subsystem.
- `NCCL_IB_HCA=mlx5_0,mlx5_1,...` — pin NCCL to a specific list of HCAs. **Critical for rail-optimized placement.**
- `NCCL_SOCKET_IFNAME=eth0` — pin NCCL's bootstrap sockets to a specific network interface. Default (`^lo,docker`) is sometimes wrong in containerized clusters.
- `NCCL_IB_DISABLE=0|1` — turn InfiniBand on or off. `1` is a debugging move: forces NCCL onto sockets over Ethernet, which is very slow but eliminates IB as a variable.
- `NCCL_P2P_LEVEL=SYS|NVL|PHB` — peer-to-peer level. `NVL` means use NVLink; `SYS` means use the system (PCIe + IB).
- `NCCL_NET_GDR_LEVEL=5` — GPUDirect RDMA level. `5` is full peer-to-peer, the fastest.
- `NCCL_BUFFSIZE=8388608` — buffer size for collective chunks. Larger is generally better for big tensors.
- `NCCL_NTHREADS=512` — number of NCCL internal threads per process. Higher helps on large messages.
- `NCCL_MIN_NCHANNELS=112` — minimum number of NetCHannels. Higher uses more links in parallel.
- `NCCL_TIMEOUT` — watchdog timeout in seconds. Default 1800 (30 min); can be lowered for faster failure detection.
- `NCCL_ASYNC_ERROR_HANDLING=1` — turn on async error handling. Without it, an NCCL error in one rank is reported asynchronously and can be very hard to attribute to a node.

The DeepSeek-V3 paper does not publish its NCCL environment [\[1\]](../appendix/b-references.md#1-deepseek-v3), and neither of DeepSeek's public repositories fills the gap: `DualPipe` is a pipeline-scheduling library and `DeepEP` a communication library, and neither sets any of the variables above (`deepseek-ai/DualPipe` @030ce43, `deepseek-ai/DeepEP` @a56d615). The Llama-3 paper [\[5\]](../appendix/b-references.md#5-llama-3) explicitly discusses the cluster tuning: Meta runs NCCLX, its own fork of NCCL, tuned for the higher latency of its RoCE fabric ([fact sheet](../appendix/fact-sheets/llama-3.md#infrastructure-and-parallelism)).

## 7.6 Common NCCL failures

The phrase "NCCL hang" is in every frontier training engineer's vocabulary. It refers to a collective operation that has started but never finished. The job is stuck; no progress is being made; the cluster is wasting power and money.

The most common causes, in rough order of frequency at frontier scale:

1. **Transient network blip.** A switch port flaps, a fiber is jostled, an HCA firmware bug, a congestion event. Usually resolves in 30–60 seconds; the run either continues or times out. Most common cause overall.
2. **Misconfigured HCA.** A new node is added to the cluster with a different HCA firmware or a wrong subnet. NCCL silently falls back to a slower path; the run is 2×–10× slower than expected but does not hang. Diagnosed by comparing per-collective timings across nodes.
3. **Bad NVLink / PCIe link.** A GPU or NVSwitch port is failing intermittently. Collectives that cross the bad link take 100× longer. Diagnosed by `nvidia-smi topo -m` and `nvidia-smi -q -d ECC` (ECC errors).
4. **Out-of-memory on a peer rank.** One rank in a pipeline-parallel group runs out of memory, the host kernel kills it, but the other 7 ranks are blocked in a `recv` from the dead rank. The watchdog eventually times out.
5. **CPU-side bug in the training loop.** A rank enters a state where it is not posting NCCL calls (e.g., stuck in a Python loop). The other ranks block on a `recv` from it.
6. **Filesystem stall.** A checkpointing operation blocks for minutes, during which the training loop is not posting collectives. Other ranks hang in the next all-reduce.

The recovery for a hang is automatic in a well-engineered run: a watchdog timer (`NCCL_TIMEOUT` plus a Python-level watchdog) fires, the run is killed, the latest checkpoint is loaded, and the run resumes. Public statements of interruption rates for frontier runs are rare. The DeepSeek-V3 report gives none; the Llama-3 report is the exception (§7.7.1). The engineering practice, though, is well understood.

### 7.6.1 Debugging NCCL

The first thing every frontier engineer does on an NCCL hang is to set `NCCL_DEBUG=INFO` (or `=TRACE` for the truly stuck) and look at the log. A few patterns to recognize:

- **All ranks posting the same op at the same time** — normal, the run is healthy.
- **One rank "stuck" on a `ncclBcast` while others wait on `ncclAllReduce`** — the rank is in a different collective than the others, almost always a bug in the training loop.
- **A rank's "rx total" not increasing for 30+ seconds** — that rank is not receiving; the sender is the suspect, or the link is bad.
- **A rank is missing from the rank table** — that rank crashed and was not restarted.

The second thing is to look at the network. `ibstatus`, `ibnetdiscover`, `perfquery` are the standard InfiniBand diagnostic tools. `pingpong` and `ib_write_bw` from the perftest suite measure raw link bandwidth; a healthy NDR link should show 12.5 GB/s unidirectional per port (and a port should show 50 GB/s for 4 ports). If a link shows 1/10 the expected bandwidth, the port is bad and the node needs to be drained.

The third thing is to profile. NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) are the standard tools. `nsys profile -o trace.nsys-rep python train.py` gives a timeline of the training step; the timeline shows the collectives, their durations, and the gaps between them. A "gap" between collectives (the time between one collective finishing and the next starting) is the metric to watch: large gaps mean the compute or the dataloader is the bottleneck, not the network.

### 7.6.2 The "rank 17 is slow" problem

The classic mystery of frontier training. All ranks are doing the same work, on the same hardware, on the same network. The training step should take the same time on every rank. But rank 17's all-reduce takes 3× longer than everyone else's, and the run is gated by rank 17's wall-clock per step.

The cause is almost always a slow link in rank 17's communication path. Either a marginal InfiniBand cable, a hot NVSwitch port, a peer GPU on the same PCIe switch that is doing extra work (e.g., a dataloader worker on the host CPU), or a NIC that is sharing bandwidth with a non-RDMA workload on the same host. The fix is to find the bad link and replace it, or to rebalance the placement to avoid the slow path.

Diagnosing this in production: every rank logs its collective durations; a dashboard highlights ranks whose 99th percentile is more than 1.5× the median. The on-call engineer looks at those ranks, inspects their NCCL log for the slow collective, identifies the link, and schedules a maintenance window.

## 7.7 GPU hardware failures

GPUs fail. The failure modes are well known, and the recovery is the same for every frontier run: detect, drain, resume.

The three main categories:

**HBM errors.** HBM3 memory has hardware ECC. Single-bit errors are corrected transparently; double-bit errors are detected and reported as ECC errors. The rate of single-bit errors is high enough that a 1,000-GPU cluster sees a few per day. A double-bit error in GPU memory is the failure that takes the GPU out of the run. NVIDIA's `nvidia-smi -q -d ECC` reports the running counts; the convention is to set a threshold (e.g., "any double-bit error in the last 24 hours, drain the GPU").

**NVLink errors.** Similar to HBM: a link can have a correctable error (CRC retry, transparent) or an uncorrectable error (the link goes down and the GPU is effectively cut off from its peers). Neither the DeepSeek-V3 report nor the Llama-3 report counts NVLink errors separately. The Llama-3 report does describe how they show up: NVLink failures "often manifest as stalled load/store operations" inside CUDA kernels, with no clear error code, which its collective library catches by timing out (§3.3.4) [\[5\]](../appendix/b-references.md#5-llama-3).

**PCIe errors.** The most catastrophic. A PCIe error can take the entire host down, dropping all 8 GPUs at once. PCIe errors are often the result of bad host hardware (a failing riser, a marginal CPU) and require a full node reboot to recover.

### 7.7.1 The MTBF math

Mean Time Between Failures for a GPU is published by NVIDIA in the range of 50,000–100,000 hours of useful life for H100. For a 2,048-GPU cluster at MTBF 100,000 hours, the expected time to the next GPU failure is $100{,}000 / 2{,}048 = 49$ hours, or roughly 2 days. In practice, what matters is the interruption rate of the whole job, which adds every non-GPU component on top. The Llama-3 paper [\[5\]](../appendix/b-references.md#5-llama-3) gives the best public measurement: on 16K H100s, 419 unexpected interruptions in 54 days, one every ~3.1 hours, or about 51,000 GPU-hours per interruption from all causes ([fact sheet](../appendix/fact-sheets/llama-3.md#reliability)). The DeepSeek-V3 report gives no interruption count or MTBF ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule)).

The 5-failures-per-day reality: a 2,000-GPU cluster at 8-hour MTBF loses on average 1 GPU every 8 hours. Across 60 days, that is 180 GPU-hours of failure, plus the recovery time. With 8 GPUs per node and 1 GPU's worth of node-level disruption per failure (the other 7 GPUs of the node are blocked), the effective loss is closer to 1,440 GPU-hours over 60 days. On a 2,048-GPU run, that is ~1.2% of the total compute. Not catastrophic, but non-trivial, and the cost of the failure is multiplied by the recovery time (loading a 1.2 TB checkpoint, restarting NCCL, re-establishing the dataloader position — typically 5–15 minutes).

## 7.8 Silent data corruption

The most feared failure mode. A bit flip in a weight, a gradient, an activation, or an optimizer state produces a wrong number, but no crash, no error message, no signal except that the validation loss curve takes a small but real hit.

The phenomenon is real. Google has published detailed studies [\[42 - "An Epidemic of Single Event Upsets", not in references\]](../appendix/b-references.md#42-nccl-documentation) showing that large-scale compute systems see soft errors at non-negligible rates. NVIDIA's H100 includes SRAM ECC, HBM ECC, and PCIe ECC, but the path from one bit of HBM to the final gradient is long, and a flip anywhere in the chain can corrupt the output.

The mechanism at frontier scale is dominated by cosmic-ray-induced upsets and by marginal hardware that is operating near its voltage/frequency corner. The rate scales with the number of bits in flight and the time those bits spend in vulnerable state (HBM, register file, on-die SRAM). A 2,000-GPU cluster running for 2 months processes roughly $10^{22}$ bit-operations; at an error rate of $10^{-12}$ per bit-hour, the expected number of errors is in the thousands.

The mitigation:

- **Periodic checksums.** Every N steps, the run computes a checksum of the weights (sum of a few random subsets, or a full SHA-256 of the gradient buffer) and compares across ranks. A mismatch between ranks in the data-parallel all-reduce is the early signal of a bit flip.
- **Weight snapshotting.** Keep multiple recent checkpoints. If a corruption is detected, roll back to the last clean checkpoint.
- **Hardware ECC.** HBM ECC catches single-bit errors transparently and reports double-bit errors. NVIDIA's `nvidia-smi -q -d ECC` reports counts.
- **Process isolation.** Run training under a process that can detect memory corruption (e.g., with `cuda-memcheck` periodically).

The DeepSeek-V3 report does not mention silent corruption, and it says the opposite of a rollback: "we did not experience any irrecoverable loss spikes or perform any rollbacks" [\[1\]](../appendix/b-references.md#1-deepseek-v3) ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#optimizer-and-schedule)). The Llama-3 paper [\[5\]](../appendix/b-references.md#5-llama-3) counts it: silent data corruption caused 6 of its 419 unexpected interruptions (1.4%) ([fact sheet](../appendix/fact-sheets/llama-3.md#reliability)). Frontier labs treat silent corruption as a first-class failure mode and run periodic checksum checks on the gradient buffer as a default.

## 7.9 Filesystem failures

A frontier training run reads tens of terabytes of training data and writes hundreds of gigabytes of checkpoints over its lifetime. The parallel filesystem is in the hot path of every step.

The standard filesystems at frontier scale:

- **Lustre.** The most common in HPC and in many Chinese labs. A parallel filesystem with a metadata server (MDS) and many object storage servers (OSS). Mature, well-understood, with good performance for large sequential reads.
- **WekaFS.** A high-performance parallel filesystem with a distributed metadata architecture. Used at Meta, xAI, and several other Western labs. Faster than Lustre for small-file workloads, more expensive.
- **GPFS / IBM Spectrum Scale.** Used at some hyperscalers and in research institutions. Mature, scales well, more operationally complex than Lustre.
- **S3-like object stores.** Used for cold storage (final checkpoints, dataset releases) and increasingly for the dataloader via a FUSE mount. Slower than a parallel filesystem but operationally simpler and effectively infinite capacity.

The common failure modes:

- **Filesystem full.** The most common. Checkpoints are large (hundreds of GB at frontier scale); if the retention policy is not aggressive about deleting old checkpoints, the filesystem fills. The dataloader or the next checkpoint write fails, the run crashes.
- **Metadata server down.** The MDS is a single point of failure for Lustre; if it goes down, all metadata operations (open, stat, mkdir) fail, and the dataloader stalls. Recovery requires restarting the MDS, which can take minutes to hours.
- **OSS failure.** An object storage server fails; the data it held is served by replicas on other OSSs but at reduced throughput. The training step slows down.
- **Stripe misconfiguration.** A file is written with a stripe size or stripe count that does not match the access pattern. The I/O is much slower than expected.

The mitigation: monitoring on filesystem capacity (`df` on the mount), aggressive checkpoint rotation (keep only the last 3–5 checkpoints), and a "pre-flight" check before every run that confirms the filesystem is healthy and has enough space for the expected run duration.

## 7.10 Power and cooling

The physical plant is a constraint that has become a primary design parameter.

**Power.** An H100 SXM is rated at 700 W TDP. An H200 is similar. A DGX H100 node with 8 H100s and the rest of the system draws approximately 10–12 kW under sustained training load. A rack of 4 nodes draws 40–50 kW. The xAI Colossus announcement [\[40\]](../appendix/b-references.md#40-xai-colossus) reports a 150 MW initial buildout for the 100k H100 cluster — at 700 W per H100, 100k H100s alone are 70 MW, and the rest of the infrastructure (CPU, memory, storage, networking, cooling overhead) brings the total to 150 MW. A "1 MW per rack" design is the modern standard: 4–8 DGX nodes, fully populated, with InfiniBand switches and storage, and the cooling plant to match.

**Cooling.** At 700 W per H100, air cooling is not enough. The standard frontier data center uses **direct-to-chip liquid cooling**: a cold plate on the GPU die, with a coolant loop that takes the heat to a heat exchanger on the building exterior. The non-GPU parts (CPU, memory, NVSwitch, HCAs) can be air-cooled, but the GPU block needs liquid. The DeepSeek-V3 paper does not discuss cooling [\[1\]](../appendix/b-references.md#1-deepseek-v3); the xAI Memphis site is liquid-cooled throughout, and the public renderings show cold-plate loops on every node [\[40\]](../appendix/b-references.md#40-xai-colossus). Air cooling is a legacy approach and is no longer built for new frontier sites.

**Capacity planning.** A frontier run is now gated by megawatts. The lead time for new substation capacity in the US is 2–5 years. This is why xAI built in Memphis (where the Tennessee Valley Authority had spare capacity), why Anthropic is partnering with AWS for new data centers, and why the next round of frontier clusters is being sited in regions with available power (Pacific Northwest, Iceland, Norway, the Middle East). We cover this in §7.14.

## 7.11 Checkpointing and resumption

The failure-recovery loop is the most operationally important piece of the training infrastructure. Without it, every failure costs the full run; with it, every failure costs a few minutes of recovery time and the latest checkpoint's worth of training.

The loop, in pseudocode:

```python
def train_with_checkpointing():
    state = load_latest_checkpoint() or initial_state()
    
    while state.step < state.total_steps:
        # Watchdog: kill the run if a single step takes too long
        with watchdog(timeout=600):
            batch = next_batch(state.dataloader)
            loss = model(batch)
            loss.backward()
            optimizer.step()
            state.step += 1
        
        # Save checkpoint every N steps
        if state.step % state.checkpoint_interval == 0:
            async_save_checkpoint(state)  # async, returns immediately
        
        # Periodic health checks
        if state.step % 100 == 0:
            run_health_check(state)  # gradient checksum, loss sanity, etc.
    
    return state

def main_with_recovery():
    while True:
        try:
            state = train_with_checkpointing()
            break  # success
        except NCCLTimeoutError:
            log("NCCL timeout, restarting from checkpoint")
            continue
        except OOMError:
            log("OOM, reducing microbatch and restarting")
            reduce_microbatch()
            continue
        except (GPULostError, NodeDownError):
            log("Hardware failure, draining affected GPUs and restarting")
            drain_failed_gpus()
            continue
        except SilentCorruptionError:
            log("Checksum mismatch, rolling back to last clean checkpoint")
            state.rollback_to_last_clean()
            continue
```

The pieces:

- **Synchronous checkpoint every N steps.** A full save of model weights, optimizer state, dataloader position, RNG state, and step count. For a 671B model in BF16 with FP32 optimizer state, the checkpoint is ~5 TB. Writing 5 TB to a parallel filesystem at 10 GB/s takes ~500 seconds (~8 minutes). This is too slow to do every step, so the typical cadence is every 100–1,000 steps (every 30 minutes to a few hours of training).
- **Asynchronous checkpointing.** The save runs in a background thread or process while the main training loop continues. The trick: keep a copy of the model and optimizer state in pinned CPU memory, snapshot it asynchronously, and write the snapshot to disk. The main loop is gated on a much smaller "shadow" save (every few minutes), and the full save happens out-of-band. The DeepSeek-V3 paper does not describe its checkpointing system, and neither the `DualPipe` nor the `DeepEP` repository contains checkpointing code (same commits as §7.5.2). The Megatron-LM `dist_checkpointing` and the PyTorch FSDP `SHARDED_STATE_DICT` mechanisms are the standard building blocks.
- **Dataloader resumption.** The dataloader is stateful (it has a current position in the data, an RNG state for shuffling, and a current sequence index). The checkpoint must include the dataloader state, or the resumed run will re-process the data it just trained on. This is a common bug: the run resumes, the loss curve looks normal, but the data is duplicated and the effective number of tokens trained is less than the step counter says.
- **RNG state.** AdamW is sensitive to the exact order of random operations (dropout, weight init, dataloader shuffling). If the RNG state is not restored, the resumed run is not bit-identical to the un-failed run, which makes the loss curve discontinuous. This is not catastrophic but it is operationally annoying.

The two-week run that died at hour 200 problem: a 14-day run that crashed 6 hours before completion has lost 13 days of training and 6 hours of compute. The mitigation is aggressive checkpointing (every 30 minutes) and, increasingly, **in-memory replication** of the optimizer state on a small set of "hot spare" nodes that can take over if a node dies. The DeepSeek-V3 paper does not describe this, and neither does the Llama-3 paper, which reports shorter job-startup and checkpointing times and fast diagnosis tools instead [\[5\]](../appendix/b-references.md#5-llama-3) ([fact sheet](../appendix/fact-sheets/llama-3.md#reliability)).

## 7.12 The MTBF reality

Let's do the math for a few common cluster sizes, assuming a per-GPU MTBF of about 16,000 hours of sustained training (16,384, to keep the arithmetic round; below the datasheet's spec):

| Cluster size | MTBF (one failure) | Failures per day | Failures per 60-day run |
|---|---|---|---|
| 256 GPUs | 64 hours | 0.4 | 22 |
| 2,048 GPUs | 8 hours | 3.0 | 180 |
| 16,000 GPUs | 1 hour | 24 | 1,440 |
| 100,000 GPUs | 10 minutes | 144 | 8,640 |

The 100k-GPU row is why frontier training is, at the limit, a problem of operational engineering. Colossus at 100k H100s is expected to lose ~6 GPUs per hour, and the failure-recovery loop must complete in under a minute to keep the cluster utilization above 80%.

The assumption is pessimistic. Llama-3 measured about 8 unexpected interruptions a day on 16K H100s, from all causes, against the table's 24 ([fact sheet](../appendix/fact-sheets/llama-3.md#reliability)).

In practice, GPU failures are the dominant cause of run interruptions. The one large public breakdown is Llama-3's: 419 unexpected interruptions in a 54-day snapshot of its 405B run [\[5\]](../appendix/b-references.md#5-llama-3). In order ([fact sheet](../appendix/fact-sheets/llama-3.md#reliability)):

1. **GPU issues** (faulty GPUs, HBM3 and SRAM memory, the GPU system processor, thermal interfaces, silent data corruption). 58.7% of unexpected interruptions.
2. **Software bugs.** 12.9%.
3. **Network switches and cables.** 8.4%.
4. **Unplanned host maintenance.** 7.6%.
5. **Everything else** (NICs, NCCL watchdog timeouts, SSDs, power supplies, CPUs, host memory). Each 1.7% or less.

About 78% of the unexpected interruptions were confirmed or suspected hardware issues. Chapter 21 works from the same numbers. Filesystem and cooling events do not appear as categories at all.

A well-engineered frontier run keeps the failure-recovery loop under 5 minutes for 95% of interruptions. The 5% that take longer (a metadata server restart, a switch firmware update) cost more in lost time but are rare.

## 7.13 Failure detection and recovery in production frameworks

The three main frameworks each handle this differently:

- **Megatron-LM** [\[3\]](../appendix/b-references.md#3-megatron-lm). The reference implementation. It writes checkpoints asynchronously through a persistent background worker process (`init_persistent_async_worker` in `megatron/training/async_utils.py`), and it can optionally restart training inside the same processes after a fault, through NVIDIA's resiliency extension (`megatron/training/inprocess_restart.py`); both files are in `NVIDIA/Megatron-LM` @d113016. DeepSeek-V3 did not use Megatron-LM: it was trained on HAI-LLM, DeepSeek's in-house framework, whose recovery system the report does not describe [\[1\]](../appendix/b-references.md#1-deepseek-v3) ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#parallelism-layout)).

- **FSDP** [\[18\]](../appendix/b-references.md#18-fsdp). The PyTorch-native fully sharded data parallel. FSDP's `BACKWARD_PRE` and `BACKWARD_POST` hooks handle the all-gather and reduce-scatter; on failure, FSDP's `StateDictType.SHARDED_STATE_DICT` allows a partial reload from the last sharded checkpoint. The recovery is at the rank level.

- **DeepSpeed ZeRO** [\[17\]](../appendix/b-references.md#17-zero). Similar to FSDP. DeepSpeed adds a `torchelastic`-style launcher that respawns failed ranks and reloads the ZeRO-partitioned state.

The on-call engineer's role in the recovery is to: (1) get paged when the watchdog fires; (2) check the rank that failed; (3) decide whether to drain the rank and continue, or to fail the whole job and restart; (4) if restart, wait for the checkpoint load and the new run to start; (5) verify the resumed run is making forward progress (loss curve, throughput, etc.). The "is the resumed run healthy?" check is the part that takes the most judgment. A run that resumes and immediately crashes is easy; a run that resumes and silently trains on corrupted data is the bad case.

A simple health-check script, the kind that lives in the on-call's toolbox:

```python
#!/usr/bin/env python3
"""
training_health_check.py — called every 100 steps from the training loop.
Detects the common failure modes and raises an explicit exception for each.
"""
import torch
import hashlib
import time

def health_check(state, model, optimizer, dataloader):
    # 1. NaN / Inf in the loss
    if not torch.isfinite(state.last_loss):
        raise NonFiniteLossError(f"loss is {state.last_loss} at step {state.step}")
    
    # 2. Gradient norm collapse or explosion
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None])
    )
    if total_norm < 1e-6:
        raise GradientCollapseError(f"grad norm {total_norm} at step {state.step}")
    if total_norm > 1e4:
        raise GradientExplosionError(f"grad norm {total_norm} at step {state.step}")
    
    # 3. Cross-rank gradient checksum — catches silent corruption
    local_grad_hash = hashlib.sha256(
        torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu().numpy().tobytes()
    ).hexdigest()
    all_hashes = [None] * state.world_size
    torch.distributed.all_gather_object(all_hashes, local_grad_hash)
    if len(set(all_hashes)) > 1:
        raise SilentCorruptionError(
            f"Gradient hash mismatch at step {state.step}: {set(all_hashes)}"
        )
    
    # 4. Throughput check — has the step time regressed?
    if state.recent_step_times and state.recent_step_times[-1] > 1.5 * state.median_step_time:
        raise ThroughputRegressionError(
            f"Step time {state.recent_step_times[-1]:.2f}s is >1.5x median {state.median_step_time:.2f}s"
        )
    
    # 5. Stale checkpoint — is the last checkpoint older than the policy allows?
    checkpoint_age = time.time() - state.last_checkpoint_time
    if checkpoint_age > state.max_checkpoint_age:
        raise CheckpointStaleError(
            f"Last checkpoint is {checkpoint_age/3600:.1f}h old, max is {state.max_checkpoint_age/3600:.1f}h"
        )
```

The checks are deliberately redundant with the framework's own internal checks. The reason: a failure that the framework misses (because the framework is the thing that crashed) is exactly the failure the on-call needs to catch.

## 7.14 The on-call rotation

A frontier training job is a 24/7/365 asset. The cluster is paid for whether the job is running or not. The job's effective utilization is the most important operational metric. The on-call engineer's job is to maximize utilization by minimizing time-to-recovery.

A typical on-call rotation at a frontier lab:

- **Rotation length**: 1 week per engineer. 5–8 engineers in the rotation, so each engineer is on-call roughly once every two months.
- **Coverage**: primary (the on-call) and secondary (the backup). The primary gets paged first; if they do not acknowledge within 5 minutes, the secondary is paged.
- **Paging tool**: PagerDuty, Opsgenie, or an in-house equivalent. Pages are sent on: job death, GPU failure rate above threshold, checkpoint save failure, gradient checksum mismatch, throughput regression above threshold.
- **Severity levels**:
  - **P1** (page immediately): job is down or has been silent for >10 minutes. Must acknowledge within 5 min, restore within 30 min.
  - **P2** (page within 30 min): job is running but unhealthy (throughput down 30%, health check failing). Restore within 4 hours.
  - **P3** (next business day): job is running but with a known issue. Investigate in the next workday.

The runbook. A frontier training job has a runbook that documents every known failure mode and the response. A simplified version:

- **NCCL hang**: set `NCCL_DEBUG=INFO`, dump the last 1000 lines, identify the stuck rank, drain the rank, restart. If the same rank hangs again, RMA the node.
- **GPU ECC error**: check the error count with `nvidia-smi -q -d ECC`. If double-bit >0, drain the GPU and restart on the remaining 7 of the node.
- **Filesystem full**: check `df` on the mount, identify the largest checkpoint, delete the oldest, restart. If filesystem is not the issue, check the OSS logs.
- **Loss spike**: skip the batch (continue training). If the spike persists, check the data for contamination. If persistent, roll back to the last clean checkpoint.
- **Throughput regression**: check the cluster for hot GPUs, check the InfiniBand fabric for errors, check the dataloader for I/O wait. Most throughput regressions are dataloader-bound.

The 3am page. The on-call gets paged at 3am because the watchdog fired. The job is down. The on-call has 5 minutes to acknowledge and 30 minutes to either restore the job or escalate. The first 5 minutes are triage: is the job actually down? Is the watchdog signal real? (Sometimes the watchdog itself has a bug.) The next 10 minutes are diagnosis: which rank failed, what was the error, was the last checkpoint good? The next 15 minutes are recovery: drain the bad rank, restart from the latest checkpoint, verify the resumed run is healthy. Total time: 30 minutes. The on-call has done this 50 times before; it is muscle memory.

The cost of a 3am page: $500 in on-call pay, $20,000–$100,000 in lost compute (depending on the cluster size and the time-to-recovery). Frontier labs pay the on-call engineer well because the alternative is much more expensive.

## 7.15 Cost economics

The dollar number is the constraint that bounds everything else.

**Per-GPU-hour cost.** The economic unit of a frontier run is the GPU-hour. Numbers are approximate and depend on contract:

- AWS p5 instance (8x H100): ~$98/hour. Per-GPU-hour: ~$12.
- Direct purchase of H100: ~$30,000–$40,000 per GPU. At a 3-year amortization and 80% utilization, per-GPU-hour: ~$1.40.
- xAI Colossus: not publicly priced, but the $6B funding round and the 100k H100 build suggest a per-GPU-hour in the $1.50–$2.50 range for the compute itself, plus power, cooling, and operations.

The DeepSeek-V3 paper reports 2,788K H800 GPU-hours for its full training, of which 2,664K was pre-training that took "less than two months" [\[1\]](../appendix/b-references.md#1-deepseek-v3). At the report's assumed $2 per GPU-hour, that is $5.576M ([fact sheet](../appendix/fact-sheets/deepseek-v3.md#context-extension-post-training-and-cost)). Meta gives no dollar figure for Llama-3 405B. Its model card gives 30.84M H100 GPU-hours ([fact sheet](../appendix/fact-sheets/llama-3.md#compute)), which at $1–$3 per GPU-hour is roughly $30M–$90M. Anthropic's Project Rainier is reportedly a >$10B multi-year commitment, spread over hundreds of thousands of Trainium 2 chips [\[41\]](../appendix/b-references.md#41-aws-trainium-2--project-rainier).

**Cluster utilization.** The fraction of the time the cluster is doing useful work, as opposed to being broken, idle, or running overhead. Frontier numbers:

- A well-engineered frontier run: 70–85% utilization.
- A run with a known issue: 50–70%.
- A run with a serious problem: <50%.

The non-utilized time is spent on: failure recovery (5–15%), checkpointing (1–3%), evaluation (1–2%), debugging (variable), and operator overhead (job startup, dataloader warmup, etc.).

**Cost of failure.** A single NCCL hang that takes 10 minutes to recover, on a 16,000-H100 cluster, at $2/H100-hour, costs $5,300. A run with 3 hangs per day, over 50 days, loses 25 hours of compute and ~$800K. A serious issue (a misconfigured HCA that silently halves collective throughput) can cost 50% of the run's effective compute. The on-call rotation exists to keep this number low.

**Cost of idle time.** A 100k-H100 cluster at $2/H100-hour costs $200K per hour, $4.8M per day, $35M per week. Idle time is the most expensive cost category. This is why frontier labs run the cluster as close to continuously as possible: a job that ends on Friday afternoon is replaced by a new job on Friday afternoon, and the new job is warm-started from the previous job's checkpoint (or is a different pre-training run, or is an eval job, or is a post-training run).

A cost calculation example. The Llama-3 405B pre-training run, in approximate dollars:

- 30.84M H100-hours, from Meta's model card ([fact sheet](../appendix/fact-sheets/llama-3.md#compute))
- At $2/H100-hour (rough estimate for owned hardware): ~$62M
- At AWS p5 pricing ($12/GPU-hour): ~$370M
- Meta reports no dollar figure. These prices are our assumptions, not Meta's.

**Cost of frontier lab buildouts.** The xAI Colossus announcement [\[40\]](../appendix/b-references.md#40-xai-colossus) is the most cited number: a 100k H100 cluster built in 122 days, at an estimated cost of $3B–$5B for the compute and another $1B–$2B for the site, power, and cooling. The Anthropic Project Rainier announcement [\[41\]](../appendix/b-references.md#41-aws-trainium-2--project-rainier) implies a similar multi-billion-dollar multi-year commitment, but on a non-NVIDIA platform. Meta's announced 1.3M-H100-equivalent supercluster is a multi-year, multi-site build that, at $30K–$40K per H100-equivalent, implies a $40B–$50B capex. These are the numbers that bound the next 2–3 years of the field.

## 7.16 What the JD actually means

Mapping the chapter to frontier-lab roles:

| Cluster / NCCL / fault-tolerance topic | Role that owns it |
|---|---|
| Cluster hardware and topology | **Cluster / Infrastructure Engineer** — owns the GPU boxes, the network, the storage, the cooling. |
| NCCL tuning and debugging | **Large-Scale Training Engineer** — owns the distributed training system, the parallelism config, the NCCL environment. |
| Network topology and placement | **Cluster Engineer + Large-Scale Training Engineer** (jointly) — the topology is owned by the cluster team, the placement by the training team. |
| Failure detection and recovery loop | **Large-Scale Training Engineer** — owns the watchdog, the checkpoint, the resume. |
| MTBF and silent corruption | **Large-Scale Training Engineer + Cluster Engineer** — hardware failures are cluster; corruption detection is training. |
| Filesystem and storage | **Cluster Engineer** — owns Lustre/WekaFS, capacity planning, the I/O path. |
| Power and cooling | **Cluster Engineer + facilities** — out of scope for ML, but the constraint is theirs. |
| Cost economics | **Tech Lead + finance partner** — owned at the program level, not the engineer level. |
| On-call rotation | **All of the above, in rotation** — typically 5–8 engineers across the training and cluster teams. |

A useful frame: every frontier-lab team has at least one Cluster Engineer and at least one Large-Scale Training Engineer, and these two roles share the on-call rotation for the training job. The Cluster Engineer gets paged for hardware events; the Training Engineer gets paged for software events. The boundary is fuzzy: an NCCL hang can be either, and the on-call escalates as needed.

## 7.17 What you should take from this chapter

1. **A frontier cluster is a hierarchy, and the bandwidth hierarchy is the single most important fact.** NVLink within a node is ~18× faster than InfiniBand between nodes. Every parallelism choice respects this.
2. **The 7-rail topology and rail-optimized placement are the standard.** NCCL's automatic topology detection is good but not perfect; setting `NCCL_IB_HCA` explicitly is the most common tuning move.
3. **NCCL hangs are the daily reality.** The recovery is automatic: watchdog fires, run is killed, latest checkpoint is loaded, run resumes. A well-engineered frontier run keeps the recovery time under 5 minutes.
4. **The MTBF math is unforgiving.** A 2,000-GPU run sees ~3 failures per day; a 100k-GPU run sees ~144 per day. The failure-recovery loop is the difference between a useful run and a wasted one.
5. **Silent corruption is real and first-class.** Frontier labs run periodic checksum checks on the gradient buffer and keep multiple recent checkpoints for rollback.
6. **Power and cooling are now primary design constraints.** A 100k H100 cluster is a 150 MW facility. The next round of frontier buildouts is gated by megawatts.
7. **The on-call rotation is what keeps the cluster running.** 5-minute acknowledgment, 30-minute restore. The runbook is muscle memory. The 3am page is part of the job.

The next chapter is the optimization deep dive: the optimizer internals, the LR schedule, the precision choices, the techniques (gradient clipping, weight decay, ZeRO-1/2/3) that turn a working run into a fast one.

---

**Exercises:** [Chapter 7 problem set](../../exercises/ch07.md) — includes the straggler-tax arithmetic and the topology-diagnosis drill.
**Lab:** [`lab07_collective_bandwidth`](../../labs/lab07_collective_bandwidth.py) — model ring and hierarchical all-reduce, all-to-all under oversubscription, and the straggler tax, and watch the same job run 3.5× slower because of one interconnect decision.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — primary case study for the 2,048-H800 cluster layout and the parallelism strategy.
- [\[3\] Megatron-LM](../appendix/b-references.md#3-megatron-lm) — the reference 3D-parallelism implementation; the basis of the topology-aware placement discussion.
- [\[5\] Llama 3](../appendix/b-references.md#5-llama-3) — the Meta training infrastructure section, including the RoCE fabric and the interruption statistics from the 54-day snapshot.
- [\[17\] ZeRO](../appendix/b-references.md#17-zero) — DeepSpeed ZeRO and the FSDP/ZeRO-3 equivalence for the checkpoint and recovery discussion.
- [\[18\] FSDP](../appendix/b-references.md#18-fsdp) — PyTorch FSDP for the failure-recovery and sharded-checkpoint discussion.
- [\[40\] xAI Colossus](../appendix/b-references.md#40-xai-colossus) — the 100k H100 cluster announcement; the case study for the super-pod tier and the power/cooling numbers.
- [\[41\] AWS Trainium 2 / Project Rainier](../appendix/b-references.md#41-aws-trainium-2--project-rainier) — the non-NVIDIA case study for the cluster-reality chapter.
- [\[42\] NCCL documentation](../appendix/b-references.md#42-nccl-documentation) — the canonical NCCL reference, opened at all times by every frontier training engineer.

[See full reference list](../appendix/b-references.md)
