#!/usr/bin/env python3
"""Tests for build_configs.py: a real-config block can't drift from its snapshot.

    python3 -m unittest discover -s tools -p "test_*.py"
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_configs  # noqa: E402

SNAPSHOT = {
    "id": "toy-tokenizer",
    "provenance": {
        "repo": "example/toy",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "downloaded_from": "example/toy@0123456789abcdef0123456789abcdef01234567",
        "retrieved": "2026-09-26",
        "files": {"tokenizer": {"path": "tokenizer.json"}, "config": {"path": "config.json"}},
    },
    "tokenizer": {
        "model_type": "BPE", "byte_fallback": False, "vocab_entries": 100, "merges": 90,
        "added_tokens": 2, "special_tokens": 2,
        "special_tokens_head": [{"id": 100, "content": "<s>"}, {"id": 101, "content": "</s>"}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "decoder_type": "ByteLevel",
    },
    "config": {"vocab_size": 128, "tie_word_embeddings": True},
}

CHAPTER = """# Chapter

Before.

<!-- real-config: toy-tokenizer -->
stale hand-written text
<!-- /real-config -->

After.
"""


class BuildConfigs(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools/config_snapshots").mkdir(parents=True)
        (self.root / "tools/config_snapshots/toy-tokenizer.json").write_text(json.dumps(SNAPSHOT))
        (self.root / "book").mkdir()
        self.chapter = self.root / "book/ch.md"
        self.chapter.write_text(CHAPTER)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_tool(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = build_configs.main(list(args), root=self.root)
        return code, out.getvalue()

    def test_check_fails_on_a_stale_block(self) -> None:
        code, out = self.run_tool("--check")
        self.assertEqual(code, 1)
        self.assertIn("toy-tokenizer", out)

    def test_build_then_check_passes(self) -> None:
        self.assertEqual(self.run_tool()[0], 0)
        code, out = self.run_tool("--check")
        self.assertEqual(code, 0, out)

    def test_generated_block_shows_values_and_origin(self) -> None:
        self.run_tool()
        text = self.chapter.read_text()
        self.assertIn('"byte_fallback": false', text)
        self.assertIn("example/toy", text)
        self.assertIn("0123456", text)
        self.assertTrue(text.startswith("# Chapter\n\nBefore."))
        self.assertTrue(text.endswith("After.\n"))

    def test_hand_edit_to_a_generated_block_fails_check(self) -> None:
        self.run_tool()
        self.chapter.write_text(self.chapter.read_text().replace('"byte_fallback": false', '"byte_fallback": true'))
        code, out = self.run_tool("--check")
        self.assertEqual(code, 1)
        self.assertIn("toy-tokenizer", out)

    def test_marker_naming_an_unknown_snapshot_fails(self) -> None:
        self.chapter.write_text(CHAPTER.replace("toy-tokenizer", "no-such-model"))
        code, out = self.run_tool("--check")
        self.assertEqual(code, 1)
        self.assertIn("no-such-model", out)


if __name__ == "__main__":
    unittest.main()
