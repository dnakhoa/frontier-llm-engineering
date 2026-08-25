# %% [markdown]
# # Lab 4 — Train a BPE tokenizer, and measure what it costs you
#
# Companion to [Chapter 4](../book/part-2-pretraining/04-tokenization.md).
#
# Tokenization looks like preprocessing and behaves like an architecture
# decision: you cannot change it after training, it sets your compute bill per
# language, and it silently decides how much context each of your users gets.
#
# You will:
#
# 1. Implement byte-level BPE from scratch — the merge loop is about 30 lines.
# 2. Measure **fertility** (tokens per word) across five languages and see the
#    2-4x spread that decides who pays more.
# 3. Watch what happens when the merge corpus is English-only.
# 4. Vary vocabulary size and find where the returns stop.
# 5. Delete the pre-tokenizer regex and look at the wreckage.
#
# Pure Python and no downloads — the multilingual corpus is embedded. Runs in
# about a minute on any machine.
#
# **Predict before you run.** Section 3 asks you to guess Vietnamese fertility
# under an English-only tokenizer. Write the number down first.

# %%
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"

# %% [markdown]
# ## 1. A multilingual corpus, embedded
#
# Five scripts with genuinely different byte-per-character behaviour in UTF-8:
#
# | Language | Script | Bytes per character |
# |---|---|---|
# | English | Latin, ASCII | 1 |
# | Vietnamese | Latin + diacritics | 1-3 |
# | Japanese | Kanji + kana | 3 |
# | Chinese | Han | 3 |
# | Python | ASCII + punctuation | 1 |
#
# That table is most of the story. BPE operates on bytes, so a script whose
# characters cost 3 bytes starts three times further from a useful token.

# %%
CORPUS = {
    "english": """
The training run began on a Tuesday and was expected to take sixty days.
Every morning the team checked the loss curve, the gradient norm, and the
throughput of each rank. A single slow node would hold back the entire cluster,
because the all-reduce completes at the pace of its slowest participant.
Data quality mattered more than anyone expected. The engineers spent months on
deduplication and filtering before a single gradient step was taken. When the
model finally converged, its behaviour was determined as much by what had been
removed from the corpus as by what remained in it.
Memory was the binding constraint. Parameters, gradients, and optimizer states
together consumed sixteen bytes for every parameter in the model, and the
activations frequently exceeded all three combined.
""",
    "vietnamese": """
Quá trình huấn luyện bắt đầu vào một ngày thứ Ba và dự kiến kéo dài sáu mươi ngày.
Mỗi buổi sáng, nhóm kỹ sư kiểm tra đường cong mất mát, chuẩn gradient và thông
lượng của từng tiến trình. Một nút chậm duy nhất có thể làm chậm toàn bộ cụm máy,
bởi vì phép all-reduce chỉ hoàn tất khi thành phần chậm nhất hoàn tất.
Chất lượng dữ liệu quan trọng hơn nhiều so với dự đoán. Các kỹ sư đã dành nhiều
tháng để loại bỏ trùng lặp và lọc dữ liệu trước khi thực hiện bước huấn luyện
đầu tiên. Khi mô hình hội tụ, hành vi của nó được quyết định bởi những gì đã bị
loại khỏi kho ngữ liệu nhiều như bởi những gì còn lại.
Bộ nhớ là ràng buộc chính. Tham số, gradient và trạng thái của bộ tối ưu hóa
cùng nhau tiêu tốn mười sáu byte cho mỗi tham số của mô hình.
""",
    "japanese": """
学習の実行は火曜日に始まり、六十日かかると見込まれていた。
毎朝、チームは損失曲線と勾配ノルム、そして各ランクのスループットを確認した。
一つの遅いノードがクラスタ全体を遅らせることがある。なぜなら、集団通信は
最も遅い参加者の速度で完了するからである。
データの品質は誰もが予想した以上に重要だった。技術者たちは最初の勾配ステップ
を実行する前に、重複除去とフィルタリングに何ヶ月も費やした。モデルが収束した
とき、その振る舞いはコーパスに残されたものと同じくらい、取り除かれたものに
よって決定されていた。
メモリが制約であった。パラメータ、勾配、最適化器の状態は合わせて一つの
パラメータあたり十六バイトを消費した。
""",
    "chinese": """
训练运行从周二开始，预计需要六十天完成。
每天早上，团队都会检查损失曲线、梯度范数以及每个进程的吞吐量。
单个缓慢的节点会拖慢整个集群，因为集合通信只有在最慢的参与者完成后才会结束。
数据质量比任何人预期的都更重要。工程师们在执行第一次梯度步骤之前，
花了数月时间进行去重和过滤。当模型最终收敛时，它的行为在很大程度上
既取决于语料库中保留的内容，也取决于被移除的内容。
内存是主要的限制因素。参数、梯度和优化器状态加起来，
每个参数需要消耗十六个字节，而激活值往往超过这三者的总和。
""",
    "python": """
def build_example(tokenizer, prompt: str, response: str, max_len: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids[:max_len], "labels": labels[:max_len]}

class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(backbone.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids).last_hidden_state
        rewards = self.score(hidden).squeeze(-1)
        last = attention_mask.sum(dim=1) - 1
        return rewards[torch.arange(rewards.size(0)), last]
""",
}

# Held-out text, in the same languages and register but NOT in the training
# corpus. Measuring fertility on the training text lets BPE memorize whole
# sentences and produces fertility below 1.0, which is meaningless.
EVAL = {
    "english": (
        "The scheduler allocated two hundred and fifty six nodes to the job. "
        "Throughput held steady for eleven hours and then dropped by a third, "
        "which the on-call engineer traced to a single throttling accelerator."
    ),
    "vietnamese": (
        "Bộ lập lịch đã cấp hai trăm năm mươi sáu nút cho công việc này. "
        "Thông lượng ổn định trong mười một giờ rồi giảm một phần ba, "
        "và kỹ sư trực đã tìm ra nguyên nhân là một thiết bị bị giảm xung nhịp."
    ),
    "japanese": (
        "スケジューラはこのジョブに二百五十六ノードを割り当てた。"
        "スループットは十一時間安定していたが、その後三分の一まで低下し、"
        "当番の技術者は原因が一つの加速器の周波数低下であることを突き止めた。"
    ),
    "chinese": (
        "调度器为该任务分配了二百五十六个节点。"
        "吞吐量稳定了十一个小时，随后下降了三分之一，"
        "值班工程师最终查明原因是一个加速器出现了降频。"
    ),
    "python": (
        "def rloo_advantages(rewards):\n"
        "    k = rewards.shape[1]\n"
        "    total = rewards.sum(dim=1, keepdim=True)\n"
        "    baseline = (total - rewards) / (k - 1)\n"
        "    return rewards - baseline\n"
    ),
}

print("Corpus character and byte counts")
print("-" * 66)
print(f"{'language':<12} {'chars':>8} {'bytes':>8} {'bytes/char':>12}")
for lang, text in CORPUS.items():
    text = text.strip()
    b = len(text.encode("utf-8"))
    print(f"{lang:<12} {len(text):>8} {b:>8} {b/len(text):>12.2f}")

# %% [markdown]
# Note the bytes-per-character column. Japanese and Chinese are around 3;
# English and Python are 1. Vietnamese sits in between because its base letters
# are ASCII and its diacritics are not.
#
# BPE starts from bytes, so a CJK character begins life as three separate
# tokens. Everything downstream follows from that.

# %% [markdown]
# ## 2. Byte-level BPE from scratch
#
# The algorithm: start with raw bytes, repeatedly find the most frequent
# adjacent pair, and merge it into a new token. That is the whole thing.
#
# The **pre-tokenizer** is the part people skip. It splits text into chunks
# before counting pairs, so merges can never span a word boundary or mix
# letters with digits. Section 6 shows what happens without it.

# %%
# The GPT-4 pattern is written with Unicode property escapes:
#
#   '(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++...
#
# Reading it left to right: contractions, letter runs with an optional leading
# space, digit runs CAPPED AT 3, punctuation runs, then whitespace. The digit
# cap is the interesting part -- it is why "2024" is not a single token, and
# consistent digit chunking is what makes arithmetic representable at all.
#
# Python's standard `re` has no \p{L}; the third-party `regex` module does.
# The approximation below uses \w, which is close enough for this lab and
# differs mainly in how it treats underscores and some punctuation.
SIMPLE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def pretokenize(text: str, use_pattern: bool = True) -> list[str]:
    """Split text into chunks that merges may not cross.

    With use_pattern=False we still split on newlines. Without that, the entire
    document is one chunk and BPE merges whole paragraphs -- a degenerate case
    that says nothing about what the regex is actually for. Splitting on lines
    isolates the effect we want to see: merges spanning word boundaries.
    """
    if not use_pattern:
        return [ln for ln in text.splitlines(keepends=True) if ln]
    return [m for m in SIMPLE_PATTERN.findall(text) if m]


class BPE:
    """Byte-level BPE. Vocabulary is 256 byte values plus learned merges."""

    def __init__(self, use_pretokenizer: bool = True):
        self.merges: dict[tuple[int, int], int] = {}
        self.use_pretokenizer = use_pretokenizer
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _word_counts(self, text: str) -> Counter:
        """Count pre-token chunks, so identical words are processed once."""
        return Counter(pretokenize(text, self.use_pretokenizer))

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        n_merges = vocab_size - 256
        if n_merges <= 0:
            return

        # Represent each distinct chunk as a tuple of byte ids, with its count.
        words = {
            tuple(w.encode("utf-8")): c for w, c in self._word_counts(text).items()
        }

        for i in range(n_merges):
            # Count adjacent pairs, weighted by how often each chunk occurs.
            pairs: Counter = Counter()
            for word, count in words.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += count
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break                          # nothing left worth merging

            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Apply the merge everywhere.
            words = {_merge(w, best, new_id): c for w, c in words.items()}

            if verbose and i % 200 == 0:
                shown = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {i:5d}: {shown!r:<20} (count {pairs[best]})")

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text, self.use_pretokenizer):
            ids = list(chunk.encode("utf-8"))
            # Repeatedly apply the earliest-learned applicable merge.
            while len(ids) >= 2:
                pairs = set(zip(ids, ids[1:]))
                candidate = min(
                    (p for p in pairs if p in self.merges),
                    key=lambda p: self.merges[p],
                    default=None,
                )
                if candidate is None:
                    break
                ids = list(_merge(tuple(ids), candidate, self.merges[candidate]))
            out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace every occurrence of `pair` in `word` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


# %% [markdown]
# ## 3. Fertility
#
# **Fertility** = tokens per word. It is the number that decides your compute
# bill per language, your users' effective context length, and their API bill.
#
# Word counting differs by script: Japanese and Chinese do not use spaces, so
# we count characters and apply the conventional ~1.5 characters-per-word
# adjustment. That is approximate, and it is approximate in the same way for
# every tokenizer we compare, so the comparison holds.


# %%
def count_words(text: str, lang: str) -> float:
    """Approximate word count. CJK has no spaces, so count characters."""
    if lang in ("japanese", "chinese"):
        cjk = sum(
            1
            for ch in text
            if unicodedata.category(ch).startswith("L") and ord(ch) > 0x2E80
        )
        return max(1.0, cjk / 1.5)
    return max(1.0, len(text.split()))


def fertility(tok: BPE, text: str, lang: str) -> float:
    return len(tok.encode(text)) / count_words(text, lang)


def report_fertility(tok: BPE, label: str) -> dict[str, float]:
    """Fertility on HELD-OUT text. Measuring on the training corpus lets BPE
    memorize whole sentences and reports numbers below 1.0."""
    print(f"\n{label}  (vocab {tok.vocab_size})")
    print("-" * 66)
    print(f"{'language':<12} {'tokens':>8} {'words':>8} {'fertility':>11} {'vs English':>12}")
    results = {}
    base = None
    for lang, text in EVAL.items():
        text = text.strip()
        f = fertility(tok, text, lang)
        results[lang] = f
        if lang == "english":
            base = f
        print(
            f"{lang:<12} {len(tok.encode(text)):>8} {count_words(text, lang):>8.0f} "
            f"{f:>11.2f} {f/base:>11.2f}x"
        )
    return results


# %%
VOCAB = 1024 if SMOKE else 4096
ALL_TEXT = "\n".join(t.strip() for t in CORPUS.values())

print(f"\nTraining balanced BPE (vocab {VOCAB}) on all five languages...")
balanced = BPE()
balanced.train(ALL_TEXT * (2 if SMOKE else 6), VOCAB, verbose=not SMOKE)
balanced_f = report_fertility(balanced, "BALANCED tokenizer")

# %% [markdown]
# ## 4. The English-only tokenizer
#
# Now the experiment that matters. Train on English alone, then measure the
# other four languages.
#
# **Predict first.** Vietnamese uses the Latin alphabet. What fertility do you
# expect relative to the balanced tokenizer — 1.2x worse? 2x? More?

# %%
print(f"\nTraining ENGLISH-ONLY BPE (vocab {VOCAB})...")
english_only = BPE()
english_only.train(CORPUS["english"].strip() * (20 if SMOKE else 60), VOCAB)
english_f = report_fertility(english_only, "ENGLISH-ONLY tokenizer")

print("\nPenalty from training the tokenizer on English alone")
print("-" * 66)
print(f"{'language':<12} {'balanced':>10} {'english-only':>14} {'penalty':>10}")
for lang in CORPUS:
    print(
        f"{lang:<12} {balanced_f[lang]:>10.2f} {english_f[lang]:>14.2f} "
        f"{english_f[lang]/balanced_f[lang]:>9.2f}x"
    )

# %% [markdown]
# The Latin-alphabet intuition is the trap. What matters is not the script but
# whether the **byte sequences** appeared in the merge-training corpus.
# Vietnamese diacritics are multi-byte in UTF-8 and appear in no English merge,
# so they fragment toward one token per byte.
#
# Chapter 4 §4.2's arithmetic turns that multiplier into money: a language at
# 2.45 fertility instead of 1.60 costs you roughly 10% of your entire training
# compute, and it costs every user of that language the same fraction of their
# context window and their API bill — permanently, because §4.5 means you
# cannot re-tokenize a trained model.

# %% [markdown]
# ## 5. Vocabulary size and diminishing returns
#
# BPE adds the highest-value merges first, so fertility improves quickly and
# then flattens. Where it flattens is the argument against very large
# vocabularies (Chapter 4 §4.9).

# %%
sizes = [512, 1024, 2048] if SMOKE else [512, 1024, 2048, 4096, 8192]
print("\nFertility vs vocabulary size (balanced corpus)")
print("-" * 66)
header = f"{'vocab':>8}" + "".join(f"{lang[:8]:>10}" for lang in EVAL)
print(header)

prev = None
for size in sizes:
    tok = BPE()
    tok.train(ALL_TEXT * (2 if SMOKE else 6), size)
    row = {lang: fertility(tok, text.strip(), lang) for lang, text in EVAL.items()}
    line = f"{tok.vocab_size:>8}" + "".join(f"{row[l]:>10.2f}" for l in EVAL)
    if prev is not None:
        gain = sum(prev[l] - row[l] for l in EVAL) / len(EVAL)
        line += f"   (mean gain {gain:+.3f})"
    print(line)
    prev = row

print("\nThe gain column shrinks with each doubling. Chapter 4 section 4.9:")
print("the marginal vocabulary tokens are trainable and not worth their")
print("softmax cost -- which is a cost argument, not an undertraining one.")

# %% [markdown]
# ## 6. Delete the pre-tokenizer
#
# The regex prevents merges from spanning word boundaries, whitespace, and
# digit runs. Remove it and look at what BPE learns instead.

# %%
print("\nTraining WITHOUT a pre-tokenizer...")
no_pretok = BPE(use_pretokenizer=False)
no_pretok.train(CORPUS["english"].strip() * (20 if SMOKE else 60), 512)

print("\nLast 12 merges learned, with and without the pre-tokenizer:")
print("-" * 66)
with_pretok = BPE()
with_pretok.train(CORPUS["english"].strip() * (20 if SMOKE else 60), 512)

def last_merges(tok: BPE, n: int = 12) -> list[str]:
    ids = sorted(tok.merges.values())[-n:]
    out = []
    for i in ids:
        s = tok.vocab[i].decode("utf-8", errors="replace")
        out.append(s if len(s) <= 24 else s[:24] + "...")
    return out

print(f"{'with pre-tokenizer':<30} {'without':<30}")
for a, b in zip(last_merges(with_pretok), last_merges(no_pretok)):
    print(f"{a!r:<30} {b!r:<30}")

print("\nWithout the regex, merges span word boundaries and whitespace.")
print("Tokens like ' the ' and 'ing the' are wasted vocabulary, and digit runs")
print("merge inconsistently -- which is why arithmetic becomes unrepresentable.")

# %% [markdown]
# ## 7. Round-trip and sanity checks

# %%
print("\nRound-trip check on every language:")
for lang, text in CORPUS.items():
    text = text.strip()
    ok = balanced.decode(balanced.encode(text)) == text
    print(f"  {lang:<12} {'OK' if ok else 'FAILED'}")
    assert ok, f"round-trip failed for {lang}"

sample = "Bộ nhớ là ràng buộc chính."
print(f"\nToken-by-token, Vietnamese sample: {sample!r}")
for name, tok in (("balanced", balanced), ("english-only", english_only)):
    ids = tok.encode(sample)
    pieces = [tok.vocab[i].decode("utf-8", errors="replace") for i in ids]
    print(f"  {name:<14} {len(ids):>3} tokens: {pieces}")

# %% [markdown]
# Look at the two token lists. The balanced tokenizer keeps syllables intact;
# the English-only one shreds the diacritics into individual bytes, which is
# what the fertility multiplier in section 4 actually consists of.

# %% [markdown]
# ## 8. Things to try
#
# **1. Train on 90% Python and measure English fertility.**
# *Common prediction:* English gets somewhat worse, symmetrically.
# *What happens:* English degrades noticeably, and the effect is **asymmetric**.
# A code-heavy tokenizer handles English badly; an English-heavy tokenizer
# handles code tolerably — because code contains a great deal of English in its
# identifiers, keywords, and comments. Code needs explicit vocabulary
# allocation; English partly comes along for free.
#
# **2. Double the vocabulary and check which language improves most.**
# *Common prediction:* everything improves proportionally.
# *What happens:* the language with the **worst** starting fertility improves
# most — BPE greedily buys the largest savings first. But only if that language
# is in the merge corpus at all. Try it on the English-only tokenizer and
# Vietnamese barely moves, which is section 4's lesson restated.
#
# **3. Add a sixth language the tokenizer has never seen.**
# Paste in some Korean or Arabic and measure its fertility under the balanced
# tokenizer. Expect roughly byte-level performance. This is what a user in an
# unsupported language experiences.
#
# **4. Compare against a real tokenizer.**
# If you have `transformers` installed, load Llama-3's or Qwen's tokenizer and
# run `report_fertility` logic against these same texts. Chapter 4 §4.7 asks you
# to do exactly this, and the differences tell you each team's target market.

# %% [markdown]
# ## What to take away
#
# 1. **Fertility is the number that matters.** It sets your training compute per
#    language, your users' effective context, and their bill.
# 2. **BPE learns byte sequences, not scripts.** A Latin-alphabet language whose
#    diacritics were absent from the merge corpus fragments to bytes.
# 3. **The penalty is permanent.** You cannot re-tokenize a trained model
#    (Chapter 4 §4.5), so this is a decision made once per model family.
# 4. **Vocabulary returns diminish quickly**, and the marginal tokens cost
#    softmax compute on every forward pass forever.
# 5. **The pre-tokenizer regex is load-bearing.** Without it, merges span words
#    and digits, and arithmetic stops being representable.
# 6. **Measure fertility on your target languages before training anything.** It
#    takes an afternoon and it is the highest-return check in the pipeline.
