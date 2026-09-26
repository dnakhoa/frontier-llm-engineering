#!/usr/bin/env python3
"""Verify the book, the exercises, and the labs stay in sync.

The failure mode this guards against: a chapter gets written but its exercise
set never does, and a self-learner hits a dead "Exercises" link at the end of
the chapter. Or an exercise set exists with no solutions, which is worse than
having no exercises at all.

Rules:

  1. Every `book/part-*/NN-*.md` chapter has `exercises/chNN.md`.
  2. Every `exercises/chNN.md` has `exercises/solutions/chNN.md`.
  3. Every chapter listed in SUMMARY.md exists on disk (also covered by
     check_links.py, repeated here so this script is useful standalone).
  4. Every lab in `labs/` is referenced from at least one chapter or exercise
     set, so no lab is orphaned.
  5. Every chapter carries a currency stamp in the fixed format, near its top:
     `*Current as of <Month YYYY | early/mid/late YYYY>.*`
  6. Every errata entry (`### E-NNN`) has all four fields filled in: what the
     book said, what the source says, where it appeared, and the fixed-in
     version.
  7. Every fact-sheet row (a table headed `| Fact | Value | Source |`) has a
     pinpoint citation: a section (§), a table or figure number, a named
     subsection of a Nature article's unnumbered Main or Methods
     (`Methods ‘GRPO’`), or
     `@commit` for a published file.

Rules 6 and 7 check that the fields exist, not that they are true. Whether
a fact row matches its source is the citation check, which a person or agent
does against the source itself; see CONTRIBUTING.md.

    python3 tools/check_structure.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTER_RE = re.compile(r"^(\d{2})-")


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
STAMP_RE = re.compile(rf"^\*Current as of (?:{MONTHS}|early|mid|late) 20\d\d\.\*$", re.M)
STAMP_WINDOW = 15  # the stamp must sit within this many lines of the top
ERRATA_FIELDS = ("The book said", "The source says", "Where", "Fixed in")
# A Nature article's Main and Methods have unnumbered, named subsections, so
# `Methods ‘Reward design’` (the name in quotes) is a pinpoint; bare "Methods" is not.
PINPOINT_RE = re.compile(
    r"§\s*\d|\bTable\s+\d|\bFig(?:ure|\.)?\s*\d|@[0-9a-f]{7,}"
    r"|\b(?:Main|Methods),?\s+[‘\'\"“]\w"
)


def chapter_numbers(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for md in sorted(root.glob("book/part-*/*.md")):
        m = CHAPTER_RE.match(md.name)
        if m:
            found[m.group(1)] = md
    return found


def check_currency_stamps(root: Path, chapters: dict[str, Path]) -> list[str]:
    errors = []
    for num, path in sorted(chapters.items()):
        head = "\n".join(path.read_text().splitlines()[:STAMP_WINDOW])
        if not STAMP_RE.search(head):
            errors.append(
                f"chapter {num} ({path.name}) has no well-formed currency stamp in its first "
                f"{STAMP_WINDOW} lines (expected '*Current as of <Month YYYY | early/mid/late YYYY>.*')"
            )
    return errors


def check_errata(root: Path) -> list[str]:
    errata = root / "book/appendix/d-errata.md"
    if not errata.exists():
        return []
    errors = []
    entries = re.split(r"^### (?=E-\d+)", errata.read_text(), flags=re.M)[1:]
    for entry in entries:
        entry_id = entry.split()[0]
        for field in ERRATA_FIELDS:
            m = re.search(rf"^\s*-\s+\*\*{re.escape(field)}:\*\*(.*)$", entry, re.M)
            if not m or not m.group(1).strip():
                errors.append(f"errata {entry_id} is missing '{field}'")
    return errors


def check_fact_sheets(root: Path) -> list[str]:
    errors = []
    for sheet in sorted((root / "book/appendix/fact-sheets").glob("*.md")):
        in_table = False
        for line in sheet.read_text().splitlines():
            # a `\|` inside a cell is an escaped pipe, not a column boundary
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
            if line.startswith("| Fact | Value | Source"):
                in_table = True
                continue
            if not in_table or not line.startswith("|"):
                in_table = in_table and line.startswith("|")
                continue
            if set(line.replace("|", "").strip()) <= set("-: "):
                continue  # the separator row
            if len(cells) < 3 or not PINPOINT_RE.search(cells[2]):
                errors.append(
                    f"{sheet.name}: fact '{cells[0]}' has no pinpoint citation "
                    f"(need §N, Table N, Figure N, or @commit)"
                )
    return errors


def main(root: Path = ROOT) -> int:
    errors: list[str] = []
    chapters = chapter_numbers(root)

    if not chapters:
        print("no chapters found — is the book/ directory intact?")
        return 1

    exercises_dir = root / "exercises"
    solutions_dir = exercises_dir / "solutions"

    # --- 1. chapter -> exercises ------------------------------------------
    for num, path in sorted(chapters.items()):
        ex = exercises_dir / f"ch{num}.md"
        if not ex.exists():
            errors.append(f"chapter {num} ({path.name}) has no exercises/ch{num}.md")

    # --- 2. exercises -> solutions ----------------------------------------
    if exercises_dir.exists():
        for ex in sorted(exercises_dir.glob("ch*.md")):
            sol = solutions_dir / ex.name
            if not sol.exists():
                errors.append(f"{ex.name} has no solutions/{ex.name}")

    # --- 3. SUMMARY entries exist -----------------------------------------
    summary = (root / "SUMMARY.md").read_text()
    for target in re.findall(r"\]\((book/[^)#]+\.md)\)", summary):
        if not (root / target).exists():
            errors.append(f"SUMMARY.md lists a chapter that does not exist: {target}")

    # --- 4. no orphaned labs ----------------------------------------------
    labs_dir = root / "labs"
    if labs_dir.exists():
        corpus = "\n".join(
            p.read_text()
            for p in list(root.glob("book/**/*.md")) + list(exercises_dir.glob("**/*.md"))
        )
        for lab in sorted(labs_dir.rglob("*.py")):
            if lab.name.startswith("_"):
                continue
            stem = lab.stem
            if stem not in corpus:
                errors.append(f"lab {lab.relative_to(root)} is not referenced from any chapter or exercise")

    # --- 5-7. currency stamps, errata, fact sheets ---------------------------
    errors += check_currency_stamps(root, chapters)
    errors += check_errata(root)
    errors += check_fact_sheets(root)

    if errors:
        print(f"FAIL: {len(errors)} structural problem(s)\n")
        for e in errors:
            print(f"  {e}")
        return 1

    print(
        f"OK: {len(chapters)} chapters, each with exercises and solutions; "
        f"all SUMMARY entries resolve; no orphaned labs; currency stamps, "
        f"errata entries and fact-sheet citations well-formed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
