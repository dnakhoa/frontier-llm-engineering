#!/usr/bin/env python3
"""The book's hand-split transcriptions of real published patterns must match them.

Chapter 4 splits the Llama-3 and Qwen3 pre-tokenizer regexes onto several
lines so they can be annotated. The blocks are labelled illustrative layouts,
and this test is what makes that honest: joined back together, each must equal
the pattern in the hash-verified tokenizer snapshot byte for byte.

    python3 -m unittest discover -s tools -p "test_*.py"
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTER = ROOT / "book/part-2-pretraining/04-tokenization.md"
TRANSCRIPTIONS = {"LLAMA3_PATTERN": "llama-3-tokenizer", "QWEN3_PATTERN": "qwen3-tokenizer"}


def transcribed(name: str) -> str:
    block = re.search(name + r" = \((.*?)\n\)", CHAPTER.read_text(), re.S)
    assert block, f"{name} not found in {CHAPTER.name}"
    return "".join(re.findall(r'r"((?:[^"\\]|\\.)*)"', block.group(1)))


def published(snapshot_id: str) -> str:
    snap = json.loads((ROOT / "tools/config_snapshots" / f"{snapshot_id}.json").read_text())
    return snap["tokenizer"]["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]


class Transcriptions(unittest.TestCase):
    def test_each_transcribed_regex_equals_the_published_pattern(self) -> None:
        for name, snapshot_id in TRANSCRIPTIONS.items():
            with self.subTest(name=name):
                self.assertEqual(transcribed(name), published(snapshot_id))


if __name__ == "__main__":
    unittest.main()
