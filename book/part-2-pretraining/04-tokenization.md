# Chapter 4: Tokenization at scale

> Reading time: ~40 minutes. By the end of this chapter, you should be able to read the tokenizer config of any frontier model, predict its behavior on a piece of text, and explain why a bad tokenizer is the most expensive mistake a pre-training team can make.

*Current as of early 2025.*

## 4.1 Why tokenization matters

The data-pipeline chapter (Chapter 3) treated tokenization as one stage in a 10-stage pipeline. This chapter makes the case that it deserves its own chapter: a bad tokenizer choice is a problem that cannot be fixed after a model is trained, and the cost of getting it wrong compounds across every training token, every inference call, and every downstream user.

Three things the tokenizer choice determines:

- **Effective sequence length.** A sequence of $N$ characters becomes a sequence of $T$ tokens. The model has a context window measured in *tokens*, not characters. If your tokenizer produces more tokens per character, the same text eats more of the context window, and you process fewer "real" characters per training step.
- **Training cost.** Training cost is roughly proportional to tokens processed, not characters. A tokenizer that requires 1.5x more tokens for the same text trains 1.5x slower, or 1.5x more expensively. At DeepSeek-V3's 14.8T tokens, a 30% increase in tokens-per-character would have cost roughly an extra month of training on the 2,048-GPU cluster [\[1\]](../appendix/b-references.md#1-deepseek-v3).
- **Quality on different languages and code.** A tokenizer trained mostly on English web text will produce many tokens for a Chinese character, an Arabic word, or a line of Python. The model still learns these tokens, but with less data per token, and with more context window eaten for the same content. This is the "Chinese is punished" problem; we cover it in §4.6.

Two metrics capture the first two effects:

- **Compression ratio** = characters per token. Higher is better.
- **Fertility score** = tokens per word. Lower is better. The standard definition is the average number of tokens a tokenizer produces per word in a reference corpus. A fertility of 1.2 means each word is split into ~1.2 tokens on average; 3.0 means each word splits into 3, a strong signal of mismatch.

A worked comparison on the same English sentence across three real tokenizers:

```
Text: "The quick brown fox jumps over the lazy dog."

GPT-2 (50K vocab, byte-level BPE):
  ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]
  → 10 tokens, ~4.5 chars/token, fertility ≈ 1.1

Llama-3 (128K, byte-level BPE):
  → 10 tokens, similar

DeepSeek-V3 (128K, bilingual BPE):
  → 10 tokens, similar
```

For English, all three land near 1 token per word. The story is different in other languages.

## 4.2 BPE: Byte-Pair Encoding

Byte-Pair Encoding was originally a data-compression algorithm (Gage, 1994) that iteratively replaced the most frequent adjacent pair of symbols with a new symbol. Sennrich, Haddow, and Birch [\[31\]](../appendix/b-references.md#31-bpe-sennrich-et-al) applied it to neural machine translation in 2016, and the rest is history: GPT-2, GPT-3, Llama, Mistral, Qwen, DeepSeek, and most modern open models use some variant of BPE.

The algorithm, given a training corpus and a target vocabulary size $V$:

1. Initialize the vocabulary with the base symbols. In a *character-level* BPE this is the set of characters; in a *byte-level* BPE this is the 256 byte values.
2. Count the frequency of every adjacent pair in the corpus.
3. Merge the most frequent pair into a new symbol. Add it to the vocabulary.
4. Repeat from step 2 until the vocabulary reaches size $V$.

The output is a *merge list* — a sequence of pair-merge rules in priority order — plus the final vocabulary. Tokenizing a new text means splitting it into base symbols, then applying the merges in priority order, greedily, until no more merges apply.

A worked example. Take the corpus:

```
low low low low low
lower lower
newest newest newest newest newest newest newest newest
widest widest widest
```

Lowercase, split on whitespace:

```
low low low low low lower lower newest newest newest newest newest newest newest widest widest
```

**Count adjacent pairs.** Top pairs: `(l, o)` 13, `(w, e)` 12, `(e, s)` 11, `(s, t)` 11, `(e, r)` 4, `(i, d)` 3, `(n, e)` 8.

**Merge the most frequent pair.** `(l, o) → lo`. Recount; `(lo, w) → low` is now most frequent. After a few more iterations the merge list looks like:

```
1. (l, o)         → lo
2. (lo, w)        → low
3. (e, r)         → er
4. (e, s)         → es
5. (es, t)        → est
6. (est, _)       → est_
7. (n, e)         → ne
8. (ne, w)        → new
9. (new, es)      → newes
10. (newes, t)    → newest
11. (i, d)        → id
12. (wid, est)    → widest
```

After these merges, the corpus tokenizes as `low low low ... er er newest newest ... widest widest`. The common sub-pieces `low`, `er`, `est` are merged into single tokens, while rare words decompose into pieces the model can still handle.

In real BPE training, the base vocabulary is bytes (for byte-level BPE) or characters (for character-level BPE, used by some SentencePiece BPE configs). The base vocabulary size is either 256 (bytes) or the size of the character set in the training corpus. The rest of the vocabulary is filled by merges until the target size is reached. GPT-2 used 256-byte base + 50,000 merges. Llama-3 used 256-byte base + 127,744 merges. DeepSeek-V3 used 256-byte base + 128,000 merges. Qwen3 used 256-byte base + 151,808 merges.

The choice of base matters: byte-level BPE is *vocabulary-complete*. Any UTF-8 string can be tokenized without an unknown character, because the base alphabet covers all possible bytes. Character-level BPE can encounter characters it has never seen; older models handled this with an `<unk>` token, which is a quality hit because the model loses information about what character it is looking at.

## 4.3 Unigram and SentencePiece

BPE is greedy: each merge is locally optimal, but the result is not necessarily globally optimal. The alternative is *Unigram* [\[24\]](../appendix/b-references.md#24-sentencepiece), the algorithm behind most SentencePiece tokenizers.

The Unigram model assumes a probabilistic model over segmentations. Given a vocabulary of subword pieces and a probability for each piece, the probability of a particular segmentation of a text is the product of the piece probabilities. Tokenization is the search for the most probable segmentation, computed with the Viterbi algorithm in linear time.

Training a Unigram model:

1. Start with a *very* large vocabulary — typically all substrings up to length 16 found in the corpus, often tens of millions of pieces.
2. Initialize piece probabilities from their corpus frequency, with smoothing.
3. For each piece, compute the expected loss if it were removed (EM's E-step).
4. Remove the bottom 10–20% (those whose removal increases the loss least).
5. Repeat from step 3 until the vocabulary reaches the target size.

The output is a vocabulary of pieces with associated probabilities. Tokenization is Viterbi decoding.

Unigram has two advantages over BPE: probabilistic tokenization (sampling segmentations during training acts as a form of data augmentation, useful for multilingual models where the "right" segmentation is ambiguous) and principled access to multiple segmentation candidates (useful for T5-style span corruption and tokenizer analysis).

The main disadvantage is complexity: BPE is a few lines of code; Unigram is an EM loop. In practice, Unigram is most useful when you have a large multilingual corpus and want a tokenizer that is *balanced* across languages, rather than overfit to whichever language has the most training data.

SentencePiece is the canonical implementation [\[24\]](../appendix/b-references.md#24-sentencepiece). It can train either BPE or Unigram, and operates directly on raw text (treating input as a byte sequence with a configurable character set), which makes it language-agnostic. This is why SentencePiece is the tool of choice for any tokenizer that needs to handle CJK, Thai, or other non-whitespace-segmented languages well.

## 4.4 Vocabulary size tradeoffs

The vocabulary size $V$ is the most consequential single hyperparameter of the tokenizer. The tradeoffs:

**Arguments for a larger vocabulary:**

- Lower fertility on the training distribution. More merges means more common sub-pieces become single tokens.
- Fewer tokens per sequence. On the training corpus, a larger vocab means shorter tokenized sequences, which means more text fits in a context window and more samples per training step.

**Arguments for a smaller vocabulary:**

- Smaller embedding tables. The embedding matrix is $V \times d_{\text{model}}$. For DeepSeek-V3 ($d = 7168$, $V = 128{,}000$) the input embedding table is ~918M parameters (~1.8 GB in BF16); for Llama-3 ($d = 4096$, $V = 128{,}000$) it is ~524M (~1.0 GB); for Qwen3 ($d = 4096$, $V = 152{,}064$) it is ~623M (~1.2 GB). Small fractions of the total model, but not free, and they grow linearly with $V$.
- Fewer rare tokens. A 256K-vocab tokenizer on a 1T-token corpus has, on average, 4,000 training examples per token; a 32K-vocab tokenizer has 31,000. The difference is real.
- Faster softmax. The output projection is $O(V \cdot d_{\text{model}})$ per token. For a 256K vocab this dominates per-token compute on the output side.

The frontier has settled on large vocabularies — 128K to 152K — because training corpora are now large enough that even 128K tokens each get 100M+ training examples, and the savings in sequence length outweigh the embedding cost. What frontier labs actually pick:

| Model | Vocab size | $d_{\text{model}}$ | Embedding params | Source |
|---|---|---|---|---|
| DeepSeek-V3 | 128,000 | 7,168 | ~918M | [\[1\]](../appendix/b-references.md#1-deepseek-v3) |
| Llama-3 (8B/70B/405B) | 128,000 | 4,096 | ~524M | [\[5\]](../appendix/b-references.md#5-llama-3) |
| Qwen3 | 152,064 | 4,096 | ~623M | [\[6\]](../appendix/b-references.md#6-qwen3) |
| GPT-2 | 50,257 | 768 | ~39M | [\[32\]](../appendix/b-references.md#32-byte-level-bpe-gpt-2--radford-et-al) |
| Llama-1 | 32,000 | 4,096 | ~131M | (Meta) |

The trend is clear: as training corpora grew from hundreds of billions to tens of trillions of tokens, vocabulary size grew from 32K to 128K–152K. The next generation is likely to push further, though with diminishing returns.

A subtle point: the embedding table is not really $V \times d_{\text{model}}$ in modern implementations. Llama-3 uses *tied* embeddings — input embedding and output projection share weights. DeepSeek-V3 does not (separate input and output embeddings). The decision is partly about training stability (tied embeddings regularize) and partly about the fact that the output side has a different distribution from the input side.

## 4.5 Byte fallback and special tokens

Byte-level BPE was introduced in GPT-2 [\[32\]](../appendix/b-references.md#32-byte-level-bpe-gpt-2--radford-et-al) and is now standard. The base vocabulary is the 256 byte values, and merges are applied on top of byte sequences. The key property: *any* UTF-8 string can be tokenized without an `<unk>` token, because the base alphabet covers all possible bytes. A character like '字' (U+5B57) is encoded as three UTF-8 bytes and tokenized as a sequence of base byte tokens (or short merges on top of them).

This is not a small detail. Before byte-level BPE, models used character-level BPE with a learned character vocabulary. Any character not in the vocabulary (a rare CJK character, an emoji, a typo) became `<unk>`, and the model lost all information about that character. The 2018-era "fast vs slow" tokenizer wars were largely about handling out-of-vocabulary characters; byte-level BPE ended the wars by making the problem go away.

The cost is that very rare characters cost many tokens. On a tokenizer with a small CJK vocabulary, '字' might decompose into 3 base byte tokens + several merge tokens, for 6–8 tokens total. On a tokenizer trained heavily on Chinese (Qwen3), it is 1 token.

**Special tokens** are reserved IDs that do not correspond to any text. Standard ones:

- `<bos>` / `<s>` — beginning of sequence, prepended to every input.
- `<eos>` / `</s>` — end of sequence, also used in training to separate documents.
- `<pad>` — padding in batches.
- `<unk>` — mostly vestigial in byte-level BPE.

Frontier models add *domain-specific* special tokens:

- **Llama-3** reserves 128 special tokens at IDs 128002–128255 for function calling, code, and chat format, with specific tokens like `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` already defined and the rest reserved for future expansion.
- **DeepSeek-V3** has FIM tokens (`<|fim_begin|>`, `<|fim_hole|>`, `<|fim_end|>`) for code fill-in-the-middle, plus `<|file_sep|>` to mark document boundaries in code corpora.
- **Qwen3** has chat tokens (`<|im_start|>`, `<|im_end|>`) and `<|tool_call|>` for function calling.

Adding special tokens is non-trivial. The new tokens are *added* to the vocabulary, which means the embedding table grows. The new token embeddings are randomly initialized, which means the model is initially poor at predicting them. Three common approaches:

1. **Continue pre-training.** Initialize new embeddings (often to the mean of existing embeddings) and run a small amount of additional pre-training to let them stabilize.
2. **Average existing embeddings.** Initialize the new token's embedding to the average of similar existing tokens (e.g., initialize `<|fim_begin|>` to the average of the embeddings of "begin" and "code").
3. **Just add and hope.** Sometimes the new tokens are rare enough (they only appear at chat-format boundaries) that their embeddings do not need to be well-trained. This is more common than you would expect.

## 4.6 Multilingual tokenization and the "Chinese is punished" problem

The well-known failure mode: a tokenizer trained mostly on English web text produces many more tokens per word in other languages, particularly CJK. The numbers are stark.

A fertility comparison on reference text in each language, using the published DeepSeek-V3 tokenizer (128K, bilingual optimized) and a hypothetical "English-only" tokenizer (32K, trained on English web text):

| Language | Fertility (DeepSeek-V3) | Fertility (English-only) | Multiplier |
|---|---|---|---|
| English | 1.3 | 1.2 | 1.08x |
| Chinese | 1.5 | 3.0 | 2.0x |
| Japanese | 1.7 | 3.5 | 2.1x |
| Korean | 1.8 | 3.8 | 2.1x |
| Arabic | 1.6 | 3.2 | 2.0x |
| Russian | 1.4 | 2.0 | 1.4x |
| Code (Python) | 1.3 | 1.4 | 1.1x |

*Illustrative numbers based on the published tokenizer configurations; the exact fertility depends on the test text.*

The "English-only" tokenizer punishes every non-English language by 2x or more. A Chinese sentence that would be 200 tokens in a Chinese-optimized tokenizer becomes 400+ tokens in an English-only one. This has two consequences:

1. **Higher cost.** A Chinese-language user pays 2x more tokens for the same content. For an API provider, this is a 2x higher inference cost per Chinese request.
2. **Worse quality.** The model spends more of its context window on the same text, leaving less room for reasoning. And because the Chinese tokens are rare in the training distribution, the model has seen less data per Chinese token, so its Chinese capabilities are weaker.

This is what the field means by "Chinese is punished." The fix is to *upweight* Chinese in the tokenizer training corpus. Qwen3's 152K vocabulary is heavily Chinese-optimized, with a large fraction of merges that produce common Chinese characters and bigrams as single tokens [\[6\]](../appendix/b-references.md#6-qwen3). DeepSeek-V3 similarly optimized for English + Chinese [\[1\]](../appendix/b-references.md#1-deepseek-v3).

A fertility computation in Python using the HuggingFace `tokenizers` library [\[33\]](../appendix/b-references.md#33-hugging-face-tokenizers-library):

```python
from tokenizers import Tokenizer

def fertility(tokenizer, text):
    """Tokens per whitespace-separated word."""
    return len(tokenizer.encode(text).ids) / max(1, len(text.split()))

ds = Tokenizer.from_file("deepseek-v3-tokenizer.json")
qwen = Tokenizer.from_file("qwen3-tokenizer.json")

en = "The quick brown fox jumps over the lazy dog. " * 100
zh = "快速的棕色狐狸跳过了懒惰的狗。" * 100
code = "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n" * 100

for label, text in [("English", en), ("Chinese", zh), ("Code", code)]:
    print(f"{label:8s}  DeepSeek-V3: {fertility(ds, text):.2f}  Qwen3: {fertility(qwen, text):.2f}")
```

On a proper multilingual evaluation set, the conclusion is robust: Qwen3 has lower fertility on Chinese than DeepSeek-V3, and lower fertility on English than some Chinese-heavy tokenizers. Each tokenizer is a compromise.

The deeper lesson: tokenizer design is a *resource allocation* problem. Every merge in the vocabulary is a bet that the corresponding sub-piece will be frequent enough in the training corpus to be worth its embedding slot. If you allocate more slots to Chinese, you have fewer for English (or code, or other languages). The right allocation depends on the target use case — which is why Qwen and DeepSeek made different choices, and why Llama-3's more balanced tokenizer is worse than either of them on Chinese specifically but better on average across many languages.

## 4.7 Code tokenization

Code has its own tokenization challenges. Three issues dominate:

- **Whitespace and indentation are syntactically significant.** In Python, `def f():\n    return 1` and `def f():\n        return 1` are different. The tokenizer must preserve indentation, ideally as a single token per indent level.
- **Common keywords and operators should be single tokens.** `def`, `class`, `return`, `if`, `else`, `==`, `!=`, `->` are all good candidates.
- **Identifiers and string literals are highly variable.** A good tokenizer handles them as sub-pieces the model can compose.

Llama-3 specifically increased the number of merges dedicated to whitespace tokens, encoding runs of spaces (one, two, three, four, tab) as separate tokens [\[5\]](../appendix/b-references.md#5-llama-3). A 4-space indent costs 1 token instead of 4. DeepSeek-Coder, the predecessor to DeepSeek-V3's code abilities, similarly added a large set of whitespace tokens.

The trade-off: every whitespace token in the vocabulary is a slot that could have been used for a common word. For a code-heavy model the trade-off is worth it; for a general-purpose chat model it is less clear.

A second concern is the handling of long identifiers. In a real codebase, identifiers like `process_user_input_with_validation` are common. A good tokenizer will split this as `process`, `_user`, `_input`, `_with`, `_validation`, each of which appears in many codebases. A bad tokenizer will split it as individual characters, costing 30+ tokens for one identifier.

A real tokenizer behavior on a line of Python:

```python
code = "    return [x*2 for x in items if x > 0]"
# Llama-3 (128K, code-aware): 16 tokens
#   ['<|begin_of_text|>', '    ', 'return', ' [', 'x', '*', '2', ' for',
#    ' x', ' in', ' items', ' if', ' x', ' >', ' 0', ']']
# A naive 32K English-only BPE might split 'return' as 'ret','urn', hurting the model
```

The difference is more pronounced on longer identifiers and on languages with unusual syntax (Rust lifetimes, Haskell type signatures, deeply nested YAML). Frontier models that care about code (DeepSeek-Coder, Code Llama, Qwen2.5-Coder, Llama-3) have tokenizer configurations that visibly favor code patterns.

## 4.8 The pre-tokenizer and regex splitting

Before BPE merges are applied, the text is *pre-tokenized* — split into chunks that are then tokenized independently. This is where most of the language-specific behavior lives.

A character-level BPE pre-tokenizer might just split on whitespace. A byte-level BPE pre-tokenizer typically uses a regex to define how to split. GPT-2's pre-tokenizer is the canonical example; it splits on whitespace, punctuation, and CJK characters separately. Llama-3 extends this with explicit handling of numbers, underscores, and code-style identifiers.

The Llama-3 pre-tokenizer regex (from the published tokenizer config [\[5\]](../appendix/b-references.md#5-llama-3)):

```python
# Llama-3 pre-tokenizer (simplified)
LLAMA3_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"          # English contractions
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"            # Letters (with optional prefix punct)
    r"|\p{N}{1,3}"                            # Numbers (1-3 digit chunks)
    r"| ?[^\s\p{L}\p{N}]++"                   # Punctuation
    r"|\s+"                                   # Whitespace (preserved!)
)
```

Key points:

- The possessive `+` (`?+`, `++`) prevents regex backtracking.
- `\p{N}{1,3}` is the trick that handles numbers as small chunks: `1234567` becomes `123`, `456`, `7` (or `1`, `234`, `567`, depending on position).
- ` ?[^\s\p{L}\p{N}]++` matches punctuation, optionally with a leading space — so ` .` and `.` are different tokens.
- `\s+` matches runs of whitespace as a single token. This is what makes indentation cheap: 4 spaces of indent = 1 token.

Qwen3 uses a similar but Chinese-extended regex. The Chinese-specific part handles Han characters as a separate class:

```python
# Qwen3 pre-tokenizer (simplified)
QWEN3_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|[^\r\n\p{L}\p{N}]?+[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]+"  # CJK
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++"
    r"|\s+"
)
```

The added `[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]` class matches CJK characters as their own pre-tokens, so they are tokenized independently of surrounding Latin text. This is what lets the tokenizer build a vocabulary that includes common Chinese characters and bigrams as single tokens.

Pre-tokenization is one of the most under-discussed parts of tokenizer design. A pre-tokenizer that does not split on whitespace, or does not handle CJK, will produce dramatically worse tokenization for the relevant languages, and the BPE merges built on top cannot fully fix it.

## 4.9 Training a tokenizer

A tokenizer is trained on a representative sample of the pre-training corpus. The standard procedure:

1. **Sample the corpus.** A few hundred million to a few billion characters is enough. The sample must be representative of the data mix that will be used for pre-training: a Chinese-heavy training corpus needs a Chinese-heavy tokenizer sample.
2. **Pick the vocabulary size.** Based on model size, training corpus size, and target languages (see §4.4).
3. **Pick the algorithm.** BPE for most modern frontier models, Unigram if you specifically want the multilingual benefits [\[24\]](../appendix/b-references.md#24-sentencepiece).
4. **Pick the pre-tokenizer.** Regex-based, as in §4.8.
5. **Train.** Run the algorithm on the sample, with a few quality-validation metrics tracked during training.
6. **Validate.** Tokenize a held-out set in each target language, compute fertility, and check that no language is wildly worse than the others.

### 4.9.1 SentencePiece

The classic command-line invocation. SentencePiece [\[24\]](../appendix/b-references.md#24-sentencepiece) takes a plain-text corpus and produces a `tokenizer.model` and a `tokenizer.vocab` file:

```bash
# Train a BPE tokenizer with 128K vocab, byte fallback
spm_train \
    --input=corpus.txt \
    --model_prefix=my_tokenizer \
    --vocab_size=128000 \
    --model_type=bpe \
    --character_coverage=0.9999 \
    --byte_fallback=true \
    --split_digits=true \
    --normalization_rule_name=identity \
    --input_sentence_size=5000000 \
    --shuffle_input_sentence=true
```

Key flags:

- `--model_type=bpe` or `unigram`. BPE for most frontier models, Unigram for multilingual balance.
- `--character_coverage=0.9999` — fraction of corpus characters that must be covered by the base character vocabulary. Lower (0.9995) keeps the base vocab small at the cost of more rare-character subwords; higher (0.99999) inflates the base vocab.
- `--byte_fallback=true` — when a character is not in the base vocabulary, fall back to its UTF-8 bytes. Makes the tokenizer vocabulary-complete.
- `--split_digits=true` — split numbers into 1–3 digit chunks.

### 4.9.2 HuggingFace tokenizers

The HuggingFace `tokenizers` library [\[33\]](../appendix/b-references.md#33-hugging-face-tokenizers-library) is Rust-backed, fast, and the standard tool in the open-source ecosystem. It produces a `tokenizer.json` artifact consumable by Transformers, vLLM, and SGLang:

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# Pre-tokenizer: regex split, preserving CJK as its own class
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)
                  |[^\r\n\p{L}\p{N}]?+\p{L}+
                  |[^\r\n\p{L}\p{N}]?+[\p{Han}]+
                  |\p{N}{1,3}
                  | ?[^\s\p{L}\p{N}]++
                  |\s+(?!\S)|\s+""",
        behavior="isolated",
    ),
])
tokenizer.decoder = decoders.ByteFallback()

trainer = trainers.BpeTrainer(
    vocab_size=128000,
    min_frequency=2,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>",
                    "<|fim_begin|>", "<|fim_hole|>", "<|fim_end|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    max_token_length=16,
)

tokenizer.train(["corpus_en.txt", "corpus_zh.txt", "corpus_code.txt"], trainer)
tokenizer.save("my_tokenizer.json")

# Use
loaded = Tokenizer.from_file("my_tokenizer.json")
print(loaded.encode("The quick brown fox.").tokens)
```

### 4.9.3 Quality validation

After training, tokenize a representative sample of *each* language and code in the training mix and verify the compression ratio. A tokenizer that compresses English at 4.0 chars/token but Chinese at 1.0 chars/token is misconfigured for a bilingual model.

A useful diagnostic:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("my_tokenizer.json")

samples = {
    "english": "The quick brown fox jumps over the lazy dog. " * 1000,
    "chinese": "快速的棕色狐狸跳过了懒惰的狗。" * 1000,
    "code":    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n" * 1000,
}

for lang, text in samples.items():
    ratio = len(text) / len(tokenizer.encode(text).ids)
    print(f"{lang:10s}  {ratio:.2f} chars/token")
```

A typical good result for a balanced multilingual model: 3.5–4.5 chars/token for English, 1.5–2.5 for Chinese, 3.0–4.0 for code. If any is off by more than ~30%, the tokenizer is misconfigured and should be retrained.

## 4.10 The "you cannot re-tokenize a model" problem

The hard constraint: a model's vocabulary is part of its identity. The embedding table has $V$ rows, and each row is tied to a specific token ID. If you want to switch to a different tokenizer, you have to either:

- **Re-initialize the embeddings for new tokens** and re-train, which is expensive (a partial pre-training run from scratch).
- **Map old tokens to new tokens** in some lossy way, which still requires significant additional training.

This is why tokenizer choice is treated as effectively permanent for a model release. Anthropic, OpenAI, Meta, DeepSeek, and Qwen all pick their tokenizer once, at the start of a major pre-training run, and live with it for the lifetime of the model.

The downstream consequences:

- **Every user pays the cost.** A user who only writes Chinese pays for an English-heavy tokenizer. A user who only writes code pays for a general-purpose tokenizer.
- **Fine-tuning cannot fix it.** Continued pre-training on Chinese data will improve the model's *Chinese capabilities*, but it will not change the number of tokens per Chinese character. The context window still gets eaten 2x faster.
- **Serving cost is locked in.** Cost per request is proportional to output tokens. A tokenizer that produces 2x more tokens for the same content doubles the inference cost for that content.

This is why the tokenizer choice gets A/B tested before a major run. The typical procedure:

1. **Train candidate tokenizers.** 2–4 candidates, varying vocab size (e.g., 64K vs 128K vs 192K) and pre-tokenizer regex.
2. **Tokenize a held-out set in each target language and code.** Compute fertility and compression ratio.
3. **Downstream eval (sometimes).** Train a small (1B–7B) model with each candidate tokenizer on a small data slice, and run a small benchmark suite. Expensive and not always done, but the gold standard.
4. **Pick the winner.** Commit to a tokenizer configuration and train the full-scale model on it.

The cost of the A/B test is small relative to the cost of the full run. The cost of a bad choice is enormous. This asymmetry drives a lot of the pre-training team's attention to tokenization.

A real example: the Llama-3 team explicitly increased vocab size from 32K (Llama-2) to 128K (Llama-3), citing multilingual and code improvements [\[5\]](../appendix/b-references.md#5-llama-3). The Qwen3 team's vocabulary distribution reflects a similar iteration, even when the size stays constant at 152K [\[6\]](../appendix/b-references.md#6-qwen3). The DeepSeek-V3 team's choice of 128K is partly because they started from a 32K Llama-1-style tokenizer and grew it as the model grew [\[1\]](../appendix/b-references.md#1-deepseek-v3). The vocab size grows with the model, not the other way around.

## 4.11 Real configurations

What the actual tokenizer files look like for the three case-study models. All three publish their tokenizers in the open-weight release; the `tokenizer.json` or `tokenizer.model` is in the model repo.

### 4.11.1 DeepSeek-V3

From the DeepSeek-V3 model release [\[1\]](../appendix/b-references.md#1-deepseek-v3):

```json
{
  "added_tokens_decoder": {
    "0": {"content": "<unk>",  "special": true},
    "1": {"content": "<s>",    "special": true},
    "2": {"content": "</s>",   "special": true},
    "3": {"content": "<|fim_begin|>", "special": true},
    "4": {"content": "<|fim_hole|>",  "special": true},
    "5": {"content": "<|fim_end|>",   "special": true},
    "6": {"content": "<|file_sep|>",  "special": true}
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "[\\p{Han}]+"},       "behavior": "Isolated"},
      {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"},       "behavior": "Isolated"},
      {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
    ]
  },
  "model": {
    "type": "BPE", "byte_fallback": true,
    "vocab": { "...": "... 128000 entries ..." },
    "merges": [ "...": "... 127744 merge rules ..." ]
  }
}
```

Key properties: 128K vocab, byte fallback, BPE, byte-level base, **Han character pre-tokenization** (Chinese characters isolated as their own pre-tokens before BPE, so merges built on top are CJK-friendly), number splitting, FIM and file-separator special tokens.

### 4.11.2 Llama-3

From the Llama-3 model release [\[5\]](../appendix/b-references.md#5-llama-3):

```json
{
  "added_tokens_decoder": {
    "128000": {"content": "<|begin_of_text|>", "special": true},
    "128001": {"content": "<|end_of_text|>",   "special": true},
    "128007": {"content": "<|start_header_id|>", "special": true},
    "128008": {"content": "<|end_header_id|>",   "special": true},
    "128009": {"content": "<|eot_id|>",          "special": true}
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++|\\s+"}, "behavior": "Isolated"},
      {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
    ]
  },
  "model": {
    "type": "BPE", "byte_fallback": true,
    "vocab": { "...": "... 128000 entries ..." },
    "merges": [ "...": "... 127744 merge rules ..." ]
  }
}
```

Key properties: 128K vocab, byte fallback, BPE, byte-level base, **whitespace and number pre-tokenization** (runs of whitespace are a single token, numbers split into 1–3 digit chunks), **128 reserved special tokens** for function calling, chat template markers (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`), **no explicit CJK pre-tokenization** (Han characters get tokenized by byte-level base + BPE merges, but no Han-isolated pre-tokens — this is why Llama-3 has higher Chinese fertility than Qwen3).

### 4.11.3 Qwen3

From the Qwen3 model release [\[6\]](../appendix/b-references.md#6-qwen3):

```json
{
  "added_tokens_decoder": {
    "151643": {"content": "<|endoftext|>", "special": true},
    "151644": {"content": "<|im_start|>",  "special": true},
    "151645": {"content": "<|im_end|>",    "special": true},
    "151670": {"content": "<|tool_call|>", "special": true}
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|[^\\r\\n\\p{L}\\p{N}]?+[\\p{Han}\\p{Katakana}\\p{Hiragana}\\p{Hangul}]+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++|\\s+"}, "behavior": "Isolated"},
      {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
    ]
  },
  "model": {
    "type": "BPE", "byte_fallback": true,
    "vocab": { "...": "... 152064 entries ..." },
    "merges": [ "...": "... 151808 merge rules ..." ]
  }
}
```

Key properties: **152,064-token vocabulary** — the largest of the three, with heavy Chinese optimization. BPE, byte-level base, **CJK pre-tokenization** (`[\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}]+` isolated as a separate pre-token class so CJK text gets its own merge treatment), chat format tokens (`<|im_start|>`, `<|im_end|>`) following the Qwen2-style template, tool-use token (`<|tool_call|>`).

The contrast is sharp: Qwen3 has ~24K more tokens than DeepSeek-V3 or Llama-3, almost all of them Chinese-optimized. This is the most direct evidence of the "Chinese is punished" problem being taken seriously: the Qwen team decided that the additional embedding cost was worth it for the reduction in Chinese-language fertility.

## 4.12 The JD, decoded

A "Tokenizer Engineer" or "Tokenization Researcher" is not a standard role at most frontier labs, but tokenization work is owned by a small number of people in the pre-training data team. What that work looks like:

| Component of tokenization | What the engineer does |
|---|---|
| Pre-tokenizer design | Iterate on the regex to handle new languages, code patterns, or domain-specific content |
| Vocabulary size | Pick the size that balances embedding cost, fertility, and rare-token risk for the target use cases |
| Training corpus sampling | Make sure the tokenizer is trained on a representative slice of the pre-training data, with target languages appropriately weighted |
| Multilingual balancing | Tune the language mix of the tokenizer training data so that no language is grossly over- or under-represented in the vocabulary |
| Quality validation | Compute fertility and compression ratios on held-out sets in every target language; flag regressions |
| A/B testing | Train candidate tokenizers, optionally run small downstream evals, pick the winner before the full-scale run |
| Special token integration | Design and add tokens for new domains (function calling, FIM, chat format) and continue-train embeddings to stabilize them |
| Tooling | Maintain the SentencePiece / tokenizers pipeline that the rest of the team uses |

This work is done by 1–3 people at a typical frontier lab, and it is *front-loaded* — most decisions are made before the pre-training run, and the rest of the team lives with them. A bad call here propagates to every training step, every inference call, and every downstream user.

## 4.13 What you should take from this chapter

1. **Tokenization is permanent.** A model's vocabulary is locked in at pre-training time. Every user pays the cost of the choice. A/B testing tokenizers before a major run is one of the highest-leverage activities a pre-training team can do.
2. **BPE is the workhorse; Unigram is the multilingual alternative.** Frontier models use byte-level BPE almost universally. Unigram / SentencePiece is the right choice when multilingual balance matters more than peak English compression.
3. **Vocabulary size is a resource allocation problem.** More tokens = better compression, smaller sequences, higher embedding cost, more rare tokens. Frontier labs have settled on 128K–152K. The allocation across languages is where the real choice lives.
4. **Multilingual tokenization is the "Chinese is punished" problem.** A tokenizer trained mostly on English will cost 2x more tokens for Chinese, Japanese, Korean. Qwen3's 152K vocab is heavily Chinese-optimized to fix this.
5. **Pre-tokenization is the most under-discussed part of tokenizer design.** The regex that splits text before BPE determines whether CJK is handled well, whether numbers are efficient, and whether indentation is preserved. Most tokenizer regressions are pre-tokenizer bugs.
6. **Byte-level BPE eliminated the unknown-token problem.** With GPT-2's byte-level approach [\[32\]](../appendix/b-references.md#32-byte-level-bpe-gpt-2--radford-et-al), any UTF-8 string tokenizes without `<unk>`. The cost is that very rare characters cost many tokens, which is part of why Chinese-optimized tokenizers matter.
7. **Code and chat tokens are added at design time.** Function-calling tokens, FIM tokens, chat template markers — these are added to the vocabulary before pre-training. Adding them later requires a partial re-train.

The next chapter covers the model architecture: the choices in attention, MoE, and the rest of the transformer block that the tokenizer's output feeds into.

---

**Exercises:** [Chapter 4 problem set](../../exercises/ch04.md) — includes the fertility arithmetic and the vocabulary-sizing problem.
**Lab:** [`lab04_train_a_bpe_tokenizer`](../../labs/lab04_train_a_bpe_tokenizer.py) — train a byte-level BPE tokenizer, measure fertility across five languages, and see what deleting the pre-tokenizer regex does to the learned merges.

---

**References for this chapter**

- [\[1\] DeepSeek-V3 Technical Report](../appendix/b-references.md#1-deepseek-v3) — for the DeepSeek-V3 tokenizer description (128K BPE, bilingual EN/ZH).
- [\[5\] Llama 3 Herd of Models](../appendix/b-references.md#5-llama-3) — for the Llama-3 tokenizer description (128K BPE, byte-level, code-aware whitespace).
- [\[6\] Qwen3 Technical Report](../appendix/b-references.md#6-qwen3) — for the Qwen3 tokenizer description (152K BPE, Chinese-heavy).
- [\[24\] SentencePiece](../appendix/b-references.md#24-sentencepiece) — Kudo and Richardson, the Unigram / BPE implementation.
- [\[31\] BPE (Sennrich et al., 2016)](../appendix/b-references.md#31-bpe-sennrich-et-al) — the original Byte-Pair Encoding for neural NLP.
- [\[32\] Byte-level BPE (GPT-2 / Radford et al., 2019)](../appendix/b-references.md#32-byte-level-bpe-gpt-2--radford-et-al) — the byte-level BPE used in GPT-2, Llama, and most modern open models.
- [\[33\] Hugging Face tokenizers](../appendix/b-references.md#33-hugging-face-tokenizers-library) — the Rust-backed tokenization library, standard in open-source releases.
- [See full reference list](../appendix/b-references.md)
