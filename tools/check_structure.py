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

    python3 tools/check_structure.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTER_RE = re.compile(r"^(\d{2})-")


def chapter_numbers() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for md in sorted(ROOT.glob("book/part-*/*.md")):
        m = CHAPTER_RE.match(md.name)
        if m:
            found[m.group(1)] = md
    return found


def main() -> int:
    errors: list[str] = []
    chapters = chapter_numbers()

    if not chapters:
        print("no chapters found — is the book/ directory intact?")
        return 1

    exercises_dir = ROOT / "exercises"
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
    summary = (ROOT / "SUMMARY.md").read_text()
    for target in re.findall(r"\]\((book/[^)#]+\.md)\)", summary):
        if not (ROOT / target).exists():
            errors.append(f"SUMMARY.md lists a chapter that does not exist: {target}")

    # --- 4. no orphaned labs ----------------------------------------------
    labs_dir = ROOT / "labs"
    if labs_dir.exists():
        corpus = "\n".join(
            p.read_text()
            for p in list(ROOT.glob("book/**/*.md")) + list(exercises_dir.glob("**/*.md"))
        )
        for lab in sorted(labs_dir.rglob("*.py")):
            if lab.name.startswith("_"):
                continue
            stem = lab.stem
            if stem not in corpus:
                errors.append(f"lab {lab.relative_to(ROOT)} is not referenced from any chapter or exercise")

    if errors:
        print(f"FAIL: {len(errors)} structural problem(s)\n")
        for e in errors:
            print(f"  {e}")
        return 1

    print(
        f"OK: {len(chapters)} chapters, each with exercises and solutions; "
        f"all SUMMARY entries resolve; no orphaned labs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
