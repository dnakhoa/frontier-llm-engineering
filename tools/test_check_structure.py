#!/usr/bin/env python3
"""Tests for check_structure.py, run against small fixture books.

Each test builds a minimal book in a temporary directory, breaks exactly one
rule, and asserts the checker reports it. The checker is only ever observed
through its exit code and printed report, never through its internals.

    python3 -m unittest discover -s tools -p "test_*.py"
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_structure  # noqa: E402

CHAPTER = """# Chapter 1: A chapter

*Current as of early 2025.*

Body text.
"""

ERRATA_OK = """# Errata

### E-001 — A wrong layout

- **The book said:** "PP = 4"
- **The source says:** PP16 ([fact](fact-sheets/deepseek-v3.md#parallelism-layout))
- **Where:** Chapter 1 §1.5
- **Fixed in:** v1.0.1
"""

FACTS_OK = """# DeepSeek-V3 fact sheet

## Parallelism layout

| Fact | Value | Source |
|---|---|---|
| Pipeline parallelism | 16-way | [V3] §3.2 |
| Tokenizer vocab | 129,280 | `tokenizer_config.json` @1a2b3c4 |
"""


def make_book(root: Path, *, chapter: str = CHAPTER, errata: str | None = ERRATA_OK,
              facts: str | None = FACTS_OK) -> None:
    (root / "book/part-1-frontier").mkdir(parents=True)
    (root / "book/part-1-frontier/01-a-chapter.md").write_text(chapter)
    (root / "exercises/solutions").mkdir(parents=True)
    (root / "exercises/ch01.md").write_text("# Exercises 1\n")
    (root / "exercises/solutions/ch01.md").write_text("# Solutions 1\n")
    (root / "SUMMARY.md").write_text("* [Ch 1](book/part-1-frontier/01-a-chapter.md)\n")
    appendix = root / "book/appendix"
    appendix.mkdir(parents=True)
    if errata is not None:
        (appendix / "d-errata.md").write_text(errata)
    if facts is not None:
        (appendix / "fact-sheets").mkdir()
        (appendix / "fact-sheets/deepseek-v3.md").write_text(facts)


def run(root: Path) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = check_structure.main(root)
    return code, out.getvalue()


class StructureChecks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_well_formed_book_passes(self) -> None:
        make_book(self.root)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)

    # --- errata ---------------------------------------------------------------

    def test_errata_entry_missing_a_field_fails(self) -> None:
        make_book(self.root, errata=ERRATA_OK.replace("- **Fixed in:** v1.0.1\n", ""))
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("E-001", out)
        self.assertIn("Fixed in", out)

    def test_errata_entry_with_empty_field_fails(self) -> None:
        make_book(self.root, errata=ERRATA_OK.replace("**Where:** Chapter 1 §1.5", "**Where:**"))
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("Where", out)

    # --- fact sheets ------------------------------------------------------------

    def test_fact_row_without_pinpoint_citation_fails(self) -> None:
        make_book(self.root, facts=FACTS_OK.replace("[V3] §3.2", "[V3]"))
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("Pipeline parallelism", out)

    def test_fact_row_with_empty_source_fails(self) -> None:
        make_book(self.root, facts=FACTS_OK.replace("| [V3] §3.2 |", "|  |"))
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("Pipeline parallelism", out)

    def test_escaped_pipe_inside_a_fact_value_is_not_a_cell_boundary(self) -> None:
        row = "| Chat marker | `<\\|eot_id\\|>` | `tokenizer.json` @1a2b3c4 |\n"
        make_book(self.root, facts=FACTS_OK + row)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)

    # --- currency stamps ----------------------------------------------------------

    def test_chapter_without_currency_stamp_fails(self) -> None:
        make_book(self.root, chapter=CHAPTER.replace("*Current as of early 2025.*\n", ""))
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("currency stamp", out)

    def test_malformed_currency_stamp_fails(self) -> None:
        make_book(self.root, chapter=CHAPTER.replace("early 2025", "sometime in 2025"))
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("currency stamp", out)

    def test_month_currency_stamp_passes(self) -> None:
        make_book(self.root, chapter=CHAPTER.replace("early 2025", "September 2026"))
        code, out = run(self.root)
        self.assertEqual(code, 0, out)


if __name__ == "__main__":
    unittest.main()
