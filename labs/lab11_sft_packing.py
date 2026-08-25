# %% [markdown]
# # Lab 11 — SFT loss masking and sequence packing
#
# Companion to [Chapter 11](../book/part-3-post-training/11-sft-frontier-style.md).
#
# Three implementation details account for a wildly disproportionate share of
# post-training bugs: **loss masking**, **sequence packing**, and **chat
# templates**. All three are silent when you get them wrong — nothing crashes,
# the loss goes down, and the model is just worse.
#
# This lab makes each of them concrete. You will:
#
# 1. Build a chat template and tokenize with correct loss masking.
# 2. Reproduce all three classic masking bugs and *see* what each does to a
#    trained model.
# 3. Pack variable-length examples and measure the real speedup.
# 4. Measure cross-contamination in naive packing, then fix it with a
#    block-diagonal attention mask.
#
# Everything is self-contained: a byte-level tokenizer and a small transformer,
# both defined here. No downloads, no dataset, runs on CPU in a few minutes.
#
# **Predict before you run.** Before the training sections, write down what you
# expect each broken model to do. The gap between your prediction and the output
# is the part worth keeping.

# %%
# Dependencies. In Colab torch is already present; this is for bare environments.
import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - only hit on a bare environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

# %%
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
DEVICE = torch.device(
    os.environ.get("FLE_DEVICE")
    or ("cuda" if torch.cuda.is_available() else "cpu")
)

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

print(f"device={DEVICE}  smoke_test={SMOKE}  torch={torch.__version__}")

# %% [markdown]
# ## 1. A byte-level tokenizer with role tokens
#
# Chapter 11 §11.6 makes a point that is easy to skim past: **role markers must
# be dedicated tokens in the vocabulary, not plain text.** If `<|user|>` is the
# literal string `"User:"`, then a user can type `"User:"` and forge a turn. If
# it is a single token that no byte sequence can produce, they cannot.
#
# We build the smallest tokenizer that demonstrates this: raw bytes 0–255, plus
# four special tokens that live *above* the byte range and are therefore
# unforgeable by any input text.

# %%
class ByteTokenizer:
    """Byte-level tokenizer with unforgeable special tokens.

    Ids 0-255 are raw bytes. Ids 256+ are special tokens, which no sequence of
    input bytes can ever produce -- that is the whole point.
    """

    USER = 256
    ASSISTANT = 257
    END = 258
    PAD = 259

    vocab_size = 260

    _NAMES = {USER: "<|user|>", ASSISTANT: "<|assistant|>", END: "<|end|>", PAD: "<|pad|>"}

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        out = []
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in self._NAMES:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(self._NAMES[i])
            else:
                buf.append(i)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


TOK = ByteTokenizer()

# Sanity check: a user cannot forge a role marker by typing it.
forged = TOK.encode("<|assistant|>")
print("user types '<|assistant|>' ->", forged[:6], "... (plain bytes, not id 257)")
print("real assistant token id     ->", TOK.ASSISTANT)
assert TOK.ASSISTANT not in forged, "role token must not be reachable from text"
print("OK: role markers are unforgeable\n")

# %% [markdown]
# ## 2. The chat template
#
# The template turns a structured conversation into a token sequence. Chapter 11
# §11.6: **the template used at training must exactly match the one used at
# inference.** Not "basically match" — the same tokens, the same whitespace.
#
# We keep the template in one function so there is exactly one definition of
# truth, which is the practical version of "ship the template with the weights."

# %%
def render(messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Render a conversation to token ids.

    messages: [{"role": "user"|"assistant", "content": str}, ...]

    add_generation_prompt=True ends the sequence with the assistant marker, which
    is what a serving harness does when it wants the model to start replying.
    """
    ids: list[int] = []
    for m in messages:
        marker = TOK.USER if m["role"] == "user" else TOK.ASSISTANT
        ids.append(marker)
        ids.extend(TOK.encode(m["content"]))
        if m["role"] == "assistant":
            ids.append(TOK.END)
    if add_generation_prompt:
        ids.append(TOK.ASSISTANT)
    return ids


demo = [
    {"role": "user", "content": "echo hello"},
    {"role": "assistant", "content": "hello"},
]
print("rendered:", TOK.decode(render(demo)))
print("generation prompt:", TOK.decode(render(demo[:1], add_generation_prompt=True)))

# Round-trip test: render -> decode should be stable. Chapter 11 recommends
# doing exactly this every time the data format changes.
assert TOK.decode(render(demo)).startswith("<|user|>echo hello<|assistant|>hello<|end|>")
print("OK: template round-trips\n")

# %% [markdown]
# ## 3. Loss masking — the correct version, and three ways to break it
#
# The rule (§11.4): feed the prompt in, but compute loss **only** on assistant
# content plus its terminating `<|end|>`.
#
# Note carefully what is masked: the `<|assistant|>` marker itself is *context*,
# not something to learn. The serving harness emits it. If the model learns to
# produce it, every reply starts with a spurious role marker.

# %%
IGNORE = -100  # F.cross_entropy ignores this label index by convention


def build_example(messages: list[dict], mode: str = "correct") -> dict:
    """Tokenize a conversation into input_ids + labels.

    mode="correct"      -- mask everything except assistant content and <|end|>
    mode="no_eos"       -- BUG: <|end|> excluded from the loss
    mode="no_mask"      -- BUG: everything is learned, including user turns
    mode="last_turn"    -- BUG: only the final assistant turn is learned
    """
    input_ids: list[int] = []
    labels: list[int] = []

    assistant_turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    last_assistant = assistant_turns[-1] if assistant_turns else -1

    for idx, m in enumerate(messages):
        marker = TOK.USER if m["role"] == "user" else TOK.ASSISTANT
        content = TOK.encode(m["content"])

        if m["role"] == "user":
            input_ids.append(marker)
            input_ids.extend(content)
            if mode == "no_mask":
                # BUG: the model is taught to generate the user's side too.
                labels.append(marker)
                labels.extend(content)
            else:
                labels.extend([IGNORE] * (1 + len(content)))
        else:
            learn = mode != "last_turn" or idx == last_assistant

            input_ids.append(marker)
            # The marker is always context, never a target -- except in no_mask,
            # which is the bug that makes models emit "<|assistant|>" out loud.
            labels.append(marker if mode == "no_mask" else IGNORE)

            input_ids.extend(content)
            labels.extend(content if learn else [IGNORE] * len(content))

            input_ids.append(TOK.END)
            if mode == "no_eos":
                # BUG: the model never learns to stop.
                labels.append(IGNORE)
            else:
                labels.append(TOK.END if learn else IGNORE)

    assert len(input_ids) == len(labels)
    return {"input_ids": input_ids, "labels": labels}


def show_masking(example: dict) -> str:
    """Bracket everything the model is NOT being taught to produce.

    Ten seconds of looking at this output prevents a class of bug that otherwise
    costs a full training run. Chapter 11 §11.4.
    """
    out = []
    for tok, label in zip(example["input_ids"], example["labels"]):
        piece = TOK.decode([tok])
        out.append(piece if label != IGNORE else f"[{piece}]")
    return "".join(out)


multi_turn = [
    {"role": "user", "content": "echo cat"},
    {"role": "assistant", "content": "cat"},
    {"role": "user", "content": "echo dog"},
    {"role": "assistant", "content": "dog"},
]

print("Bracketed = context (not learned). Unbracketed = training target.\n")
for mode in ("correct", "no_eos", "no_mask", "last_turn"):
    ex = build_example(multi_turn, mode)
    learned = sum(1 for x in ex["labels"] if x != IGNORE)
    print(f"{mode:>10}: {show_masking(ex)}")
    print(f"{'':>10}  {learned}/{len(ex['labels'])} tokens contribute to the loss\n")

# %% [markdown]
# Read those four lines carefully before continuing.
#
# - `correct` learns `cat<|end|>` and `dog<|end|>` — both assistant turns.
# - `no_eos` learns the content but never the stop token.
# - `no_mask` learns *everything*, including `<|user|>echo dog`. This model will
#   happily write your side of the conversation.
# - `last_turn` learns only `dog<|end|>`. On this two-turn example it throws away
#   half the signal; on a five-turn conversation it throws away 80%. Nothing
#   about the loss curve tells you this is happening.

# %% [markdown]
# ## 4. A small transformer that accepts an explicit attention mask
#
# We need an explicit mask (rather than `is_causal=True`) because §6 of this lab
# swaps in a block-diagonal mask. Position ids are also explicit, so packed
# examples can restart their positions — the cheap half-fix from §11.5.

# %%
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        # (B, T, D) -> (B, n_heads, T, head_dim)
        q, k, v = (
            t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v)
        )
        # attn_mask is (B, 1, T, T) boolean; True means "may attend".
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d_model=128, n_layers=4, n_heads=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.max_len = max_len

    def forward(self, input_ids, attn_mask=None, position_ids=None):
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).expand(B, T)
        if attn_mask is None:
            attn_mask = causal_mask(B, T, input_ids.device)

        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.head(self.ln_f(x))


def causal_mask(B: int, T: int, device) -> torch.Tensor:
    """Standard lower-triangular mask, shaped (B, 1, T, T)."""
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return m.view(1, 1, T, T).expand(B, 1, T, T)


# %% [markdown]
# ## 5. Train four models — one correct, three broken
#
# The task is deliberately trivial (`echo WORD` -> `WORD`) so a 4-layer model
# learns it on CPU in a couple of minutes. Triviality is the point: any
# difference in behaviour between these four models is caused by the *masking*,
# not by the difficulty of the task.

# %%
WORDS = ["cat", "dog", "bird", "fish", "tree", "rock", "star", "moon", "leaf", "wave"]
MAX_LEN = 128
STEPS = 40 if SMOKE else 600
BATCH = 8 if SMOKE else 16
D_MODEL = 64 if SMOKE else 128
N_LAYERS = 2 if SMOKE else 4


def sample_conversation(rng: random.Random, n_turns: int = 2) -> list[dict]:
    msgs = []
    for _ in range(n_turns):
        w = rng.choice(WORDS)
        msgs.append({"role": "user", "content": f"echo {w}"})
        msgs.append({"role": "assistant", "content": w})
    return msgs


def make_batch(mode: str, batch_size: int, rng: random.Random):
    """Build a padded batch of examples under the given masking mode."""
    examples = [
        build_example(sample_conversation(rng), mode) for _ in range(batch_size)
    ]
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((batch_size, width), TOK.PAD, dtype=torch.long)
    labels = torch.full((batch_size, width), IGNORE, dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
    return ids.to(DEVICE), labels.to(DEVICE)


def train(mode: str) -> TinyLM:
    torch.manual_seed(SEED)  # identical init for every mode -- only masking differs
    model = TinyLM(TOK.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, max_len=MAX_LEN)
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = random.Random(SEED)

    for step in range(STEPS):
        ids, labels = make_batch(mode, BATCH, rng)
        logits = model(ids)
        # Standard next-token shift: predict position t+1 from position t.
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, TOK.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, STEPS // 4) == 0:
            print(f"  [{mode}] step {step:4d}  loss {loss.item():.4f}")
    return model.eval()


@torch.no_grad()
def generate(model: TinyLM, prompt: str, max_new: int = 40, stop_at_end: bool = True) -> str:
    """Greedy generation.

    stop_at_end=True mimics a serving harness, which halts on <|end|>.
    stop_at_end=False keeps going, which is how you see what the model *would*
    have said -- the probe that exposes the no_mask bug.
    """
    ids = render([{"role": "user", "content": prompt}], add_generation_prompt=True)
    ids = torch.tensor([ids], device=DEVICE)
    produced = []
    for _ in range(max_new):
        if ids.shape[1] >= model.max_len:
            break
        nxt = model(ids)[:, -1].argmax(-1)
        produced.append(int(nxt))
        if stop_at_end and int(nxt) == TOK.END:
            break
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return TOK.decode(produced)


models = {}
for mode in ("correct", "no_eos", "no_mask", "last_turn"):
    print(f"training '{mode}' ...")
    models[mode] = train(mode)
print()

# %% [markdown]
# ### What each broken model does
#
# Look at the generations below against your predictions.

# %%
print("prompt: 'echo star'   (expected correct output: 'star<|end|>')\n")
for mode, model in models.items():
    out = generate(model, "echo star")
    stopped = "<|end|>" in out
    print(f"{mode:>10}: {out!r}")
    print(f"{'':>10}  stops on its own: {stopped}")
print()

print("Diagnosis:")
print("  correct   -- emits the word, then <|end|>. Stops. This is the target.")
print("  no_eos    -- emits the word, then babbles. Never learned to stop.")
print("  no_mask   -- looks fine here! The damage is hidden; see the probe below.")
print("  last_turn -- trained on ~31% of the tokens, so at equal steps it is")
print("               undertrained. On this trivial task it often degenerates to")
print("               emitting <|end|> immediately.")

# %% [markdown]
# ### Probing the hidden bug
#
# `no_mask` passed the test above, which is exactly what makes it dangerous. The
# serving harness stops at `<|end|>`, so the damage is invisible in normal use.
#
# Keep generating past the stop token and it surfaces: a model trained on user
# turns will happily write the *next user message*.

# %%
print("Ignoring <|end|> and generating 60 tokens anyway:\n")
for mode in ("correct", "no_mask"):
    out = generate(models[mode], "echo star", max_new=60, stop_at_end=False)
    forged = out.count("<|user|>")
    print(f"{mode:>10}: {out!r}")
    print(f"{'':>10}  user turns hallucinated: {forged}")
print()
print("The 'no_mask' model was explicitly trained to produce '<|user|>echo ...',")
print("so it does. In a multi-turn product this shows up as the model answering")
print("questions the user never asked.")

# %% [markdown]
# The `last_turn` bug is the dangerous one. The other two announce themselves the
# first time you look at a generation. `last_turn` produces a model that works —
# it just silently learned from a quarter of the data you paid to produce.
#
# The only way to catch it is the bracketed printout from §3. Which is why you
# print it every time the data format changes.

# %%
# Quantify the silent one: how much training signal does each bug destroy?
rng = random.Random(123)
convo = sample_conversation(rng, n_turns=3)
print("\nTraining signal per 3-turn conversation:")
baseline = sum(1 for x in build_example(convo, "correct")["labels"] if x != IGNORE)
for mode in ("correct", "no_eos", "no_mask", "last_turn"):
    n = sum(1 for x in build_example(convo, mode)["labels"] if x != IGNORE)
    print(f"  {mode:>10}: {n:3d} supervised tokens  ({n / baseline:5.1%} of correct)")

# %% [markdown]
# ## 6. Packing, and the contamination it introduces
#
# SFT examples vary wildly in length. Padding every sequence to the longest one
# wastes most of your compute on nothing. Packing concatenates examples up to the
# context length instead.
#
# First, measure the waste.

# %%
def pack(examples: list[dict], max_len: int):
    """Greedily concatenate examples into fixed-width sequences.

    Returns (input_ids, labels, segment_ids). segment_ids records which example
    each token came from -- that is what makes the block-diagonal mask possible.
    """
    packs, cur_ids, cur_labels, cur_seg = [], [], [], []
    seg = 0
    for e in examples:
        n = len(e["input_ids"])
        if n > max_len:
            continue  # a single example longer than the context; skip for clarity
        if len(cur_ids) + n > max_len:
            packs.append((cur_ids, cur_labels, cur_seg))
            cur_ids, cur_labels, cur_seg, seg = [], [], [], 0
        cur_ids.extend(e["input_ids"])
        cur_labels.extend(e["labels"])
        cur_seg.extend([seg] * n)
        seg += 1
    if cur_ids:
        packs.append((cur_ids, cur_labels, cur_seg))
    return packs


# A realistic SFT length distribution: mostly short, with a long tail.
rng = random.Random(7)
n_examples = 200 if SMOKE else 2000
corpus = []
for _ in range(n_examples):
    turns = rng.choices([1, 1, 1, 2, 2, 3, 6], k=1)[0]
    corpus.append(build_example(sample_conversation(rng, turns), "correct"))

lengths = [len(e["input_ids"]) for e in corpus]
longest = max(lengths)
padded_tokens = len(corpus) * longest
real_tokens = sum(lengths)

packs = pack(corpus, MAX_LEN)
packed_tokens = sum(len(p[0]) for p in packs)

print(f"examples:            {len(corpus)}")
print(f"length: median {sorted(lengths)[len(lengths)//2]}, max {longest}")
print()
print(f"padded to max:       {padded_tokens:,} tokens across {len(corpus)} sequences")
print(f"actual content:      {real_tokens:,} tokens  ({real_tokens/padded_tokens:.1%} useful)")
print(f"packed to {MAX_LEN}:       {packed_tokens:,} tokens across {len(packs)} sequences")
print()
print(f"speedup from packing: {padded_tokens / packed_tokens:.1f}x fewer tokens processed")

# %% [markdown]
# That multiplier is why everyone packs. Now the correctness problem.
#
# With a plain causal mask, tokens in example B attend to tokens in example A.
# The model is learning from a context that will never occur at inference time.
#
# We can measure this directly: take a packed sequence, compute the logits for
# example B's tokens, then change *only example A* and recompute. If example B's
# predictions move, contamination is real and quantifiable.

# %%
def block_diagonal_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Causal mask AND same-segment mask, shaped (B, 1, T, T).

    This is what FlashAttention's varlen mode computes efficiently from
    cumulative sequence lengths. Chapter 11 §11.5.
    """
    B, T = segment_ids.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=segment_ids.device))
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    return (causal.view(1, T, T) & same_segment).unsqueeze(1)


def positions_within_segment(segment_ids: torch.Tensor) -> torch.Tensor:
    """Restart position ids at every segment boundary (the cheap half-fix).

    position = absolute index - index of the first token of this segment, so
    every packed example sees positions 0..len-1 exactly as it would standalone.
    """
    B, T = segment_ids.shape
    idx = torch.arange(T, device=segment_ids.device).expand(B, T)
    is_start = torch.ones_like(segment_ids, dtype=torch.bool)
    is_start[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    # Running maximum of "index, but only at segment starts" = current segment start.
    segment_start = torch.cummax(torch.where(is_start, idx, torch.zeros_like(idx)), dim=1).values
    return idx - segment_start


model = models["correct"]

# Build two packs that share example B but differ in example A.
ex_a1 = build_example([{"role": "user", "content": "echo moon"},
                       {"role": "assistant", "content": "moon"}], "correct")
ex_a2 = build_example([{"role": "user", "content": "echo tree"},
                       {"role": "assistant", "content": "tree"}], "correct")
ex_b = build_example([{"role": "user", "content": "echo star"},
                      {"role": "assistant", "content": "star"}], "correct")


def assemble(prefix, target):
    ids = prefix["input_ids"] + target["input_ids"]
    seg = [0] * len(prefix["input_ids"]) + [1] * len(target["input_ids"])
    return (
        torch.tensor([ids], device=DEVICE),
        torch.tensor([seg], device=DEVICE),
        len(prefix["input_ids"]),
    )


ids1, seg1, off1 = assemble(ex_a1, ex_b)
ids2, seg2, off2 = assemble(ex_a2, ex_b)


@torch.no_grad()
def logits_for_target(ids, seg, offset, use_block_mask: bool):
    B, T = ids.shape
    mask = block_diagonal_mask(seg) if use_block_mask else causal_mask(B, T, ids.device)
    pos = positions_within_segment(seg) if use_block_mask else None
    out = model(ids, attn_mask=mask, position_ids=pos)
    return F.log_softmax(out[0, offset:], dim=-1)


for label, use_block in (("naive causal packing", False), ("block-diagonal packing", True)):
    lp1 = logits_for_target(ids1, seg1, off1, use_block)
    lp2 = logits_for_target(ids2, seg2, off2, use_block)
    n = min(lp1.shape[0], lp2.shape[0])
    kl = F.kl_div(lp1[:n], lp2[:n], log_target=True, reduction="batchmean")
    max_abs = (lp1[:n] - lp2[:n]).abs().max()
    print(f"{label:>24}: KL={kl.item():.6f}  max |Δlogprob|={max_abs.item():.6f}")

print()
print("Example B is byte-identical in both packs. Only the *preceding* example")
print("changed. Under naive causal packing its predictions move anyway -- that is")
print("cross-contamination, measured. Under block-diagonal masking the divergence")
print("is exactly zero, because example B cannot see example A at all.")

# %% [markdown]
# ## 7. Things to try
#
# Each of these has a common wrong prediction. Write yours down first.
#
# **1. Train the `no_mask` model for 5× longer.**
# *Common prediction:* it eventually learns to stop generating the user's turn.
# *What happens:* it gets better at generating user turns. Nothing in the
# objective ever told it not to. More compute makes a wrong objective worse, not
# better — a theme that recurs through all of Part III.
#
# **2. Set `MAX_LEN` to 512 and re-run the packing measurement.**
# *Common prediction:* packing efficiency improves.
# *What happens:* the padded baseline gets much worse (it pads to the longest
# example) while packed tokens stay nearly constant, so the *ratio* jumps. The
# speedup number is a property of your length distribution, not of packing.
#
# **3. Use block-diagonal masks but leave position ids running across segments.**
# *Common prediction:* it barely matters since attention is already blocked.
# *What happens:* example B at offset 60 sees positional embeddings it would
# never see standalone. With learned positional embeddings the effect is small;
# with RoPE and long contexts it is not. This is why §11.5 lists position reset
# as a separate fix.
#
# **4. Make one word in `WORDS` appear 10× more often, then check `no_eos`.**
# *Common prediction:* only that word's generations get longer.
# *What happens:* the model's runaway continuations become dominated by that
# word. A missing EOS turns your frequency distribution into a babble
# distribution — which is roughly what an unaligned base model does, and a decent
# intuition for why SFT is needed at all.
#
# **5. Delete the `<|end|>` from `render()` but keep it in the labels.**
# *Common prediction:* an assertion fires.
# *What happens:* `input_ids` and `labels` go out of sync by one token per turn,
# and every label is shifted onto the wrong input. The loss still decreases. This
# is the ugliest bug in this lab and the reason `build_example` asserts on length.

# %% [markdown]
# ## What to take away
#
# 1. **Decode your labels before every run.** The bracketed printout in §3 costs
#    ten seconds and catches three of the four bugs in this lab.
# 2. **The dangerous masking bug is the silent one.** `last_turn` produces a
#    working model trained on a fraction of your data, and no metric shows it.
# 3. **Packing is a large, easy win** — the exact multiplier depends on your
#    length distribution, so measure it rather than quoting someone else's.
# 4. **Cross-contamination is real and measurable**, and block-diagonal masking
#    removes it exactly. FlashAttention's varlen mode does this for free, so
#    there is little reason to accept the contamination.
# 5. **Role markers must be vocabulary tokens.** Anything a user can type, a user
#    can forge.
#
# Next: [Chapter 12 — Reward modeling](../book/part-3-post-training/12-reward-modeling.md),
# where you build the scorer that rejection sampling and every RL method depend
# on, and [`lab12_reward_model`](lab12_reward_model.py).
