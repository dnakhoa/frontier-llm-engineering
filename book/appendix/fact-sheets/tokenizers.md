# Tokenizer fact sheet

Facts about the three case-study tokenizers, read from the published files rather than from descriptions of them. Every file was downloaded at a pinned commit and verified against the official repository's hash by `tools/fetch_config_snapshots.py`; the snapshots are in `tools/config_snapshots/`. See [how fact sheets work](../c-fact-sheets.md).

**Sources.**
- `[DS]` `deepseek-ai/DeepSeek-V3` `tokenizer.json` and `config.json` @e815299
- `[L3]` `meta-llama/Meta-Llama-3-8B` `tokenizer.json` and `config.json` @8cde5ca. The official repo is gated, so we downloaded from `NousResearch/Meta-Llama-3-8B` @315b200, whose bytes match the official files' git blob ids.
- `[Q3]` `Qwen/Qwen3-235B-A22B` `tokenizer.json` and `config.json` @8efa617
- `[V3]` the DeepSeek-V3 report v2, `[Llama3]` the Llama 3 report v3 (arXiv:2407.21783v3), `[Qwen3]` the Qwen3 report v1 (arXiv:2505.09388v1)

Checked against the files: 2026-09-26.

## DeepSeek-V3

| Fact | Value | Source |
|---|---|---|
| Algorithm | Byte-level BPE, "extended vocabulary of 128K tokens" | [V3] §4.1 |
| BPE vocabulary entries | 128,000 | `tokenizer.json` model.vocab @e815299 |
| Merge rules | 127,741 | `tokenizer.json` model.merges @e815299 |
| Added tokens | 818 (804 special), including FIM, chat-role, tool-call and ~800 placeholder tokens | `tokenizer.json` added_tokens @e815299 |
| `vocab_size` (embedding rows) | 129,280 | `config.json` @e815299 |
| `byte_fallback` | false | `tokenizer.json` model @e815299 |
| Normalizer | Empty sequence (none) | `tokenizer.json` @e815299 |
| Pre-tokenizer | 1. digits `\p{N}{1,3}`; 2. CJK runs `[一-龥぀-ゟ゠-ヿ]+` isolated; 3. a punctuation/letter/newline regex; 4. ByteLevel | `tokenizer.json` pre_tokenizer @e815299 |
| Tied embeddings | No (`tie_word_embeddings: false`) | `config.json` @e815299 |

## Llama-3

| Fact | Value | Source |
|---|---|---|
| Composition | "100K tokens from the tiktoken tokenizer with 28K additional tokens to better support non-English languages" | [Llama3] §3.2 |
| BPE vocabulary entries | 128,000 | `tokenizer.json` model.vocab @8cde5ca |
| Merge rules | 280,147 | `tokenizer.json` model.merges @8cde5ca |
| Added tokens | 256, all special, IDs 128000–128255; named ones include `<\|begin_of_text\|>`, `<\|end_of_text\|>`, `<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>` | `tokenizer.json` added_tokens @8cde5ca |
| `vocab_size` (embedding rows) | 128,256 | `config.json` @8cde5ca |
| `byte_fallback` | false | `tokenizer.json` model @8cde5ca |
| Normalizer | None | `tokenizer.json` @8cde5ca |
| Pre-tokenizer | tiktoken-style regex with `\p{N}{1,3}`, no CJK class; then ByteLevel | `tokenizer.json` pre_tokenizer @8cde5ca |
| Tied embeddings | No (`tie_word_embeddings: false`) | `config.json` @8cde5ca |

## Qwen3

| Fact | Value | Source |
|---|---|---|
| Algorithm and size | Byte-level BPE, vocabulary size 151,669 | [Qwen3] §2 |
| BPE vocabulary entries | 151,643 | `tokenizer.json` model.vocab @8efa617 |
| Merge rules | 151,387 | `tokenizer.json` model.merges @8efa617 |
| Added tokens | 26 (14 special); `<tool_call>`, `<think>` and FIM tokens are added but not special | `tokenizer.json` added_tokens @8efa617 |
| `vocab_size` (embedding rows) | 151,936 | `config.json` @8efa617 |
| `byte_fallback` | false | `tokenizer.json` model @8efa617 |
| Normalizer | NFC | `tokenizer.json` @8efa617 |
| Pre-tokenizer | Llama-3's regex with `\p{N}` (single digits) in place of `\p{N}{1,3}`; no CJK class; then ByteLevel | `tokenizer.json` pre_tokenizer @8efa617 |
| Tied embeddings | No (`tie_word_embeddings: false`) | `config.json` @8efa617 |

## Measured Han coverage

Our measurement, not a figure any lab publishes. We decoded each BPE vocabulary entry from its byte-level form to UTF-8 (skipping entries that are partial characters) and counted those containing at least one character in U+4E00–U+9FFF.

| Fact | Value | Source |
|---|---|---|
| DeepSeek-V3 entries containing Han | 35,334 of 128,000 (27.6%) | our count over `tokenizer.json` @e815299 |
| Qwen3 entries containing Han | 25,511 of 151,643 (16.8%) | our count over `tokenizer.json` @8efa617 |
| Llama-3 entries containing Han | 4,387 of 128,000 (3.4%) | our count over `tokenizer.json` @8cde5ca |

```python
# Reproduce the count for any byte-level BPE tokenizer.json.
import json, re

def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def han_entries(path: str) -> int:
    dec, han = byte_decoder(), re.compile(r"[一-鿿]")
    vocab = json.load(open(path))["model"]["vocab"]
    count = 0
    for token in vocab:
        try:
            text = bytes(dec[ch] for ch in token).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue  # a fragment of a multi-byte character
        count += bool(han.search(text))
    return count
```
