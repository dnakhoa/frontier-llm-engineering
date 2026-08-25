#!/usr/bin/env python3
"""Validate every internal Markdown link in the book.

Checks three things, because all three have broken in this repo before:

  1. Every file referenced by SUMMARY.md exists (a missing chapter renders as a
     dead entry in the GitBook sidebar).
  2. Every relative link in every Markdown file points at a file that exists.
     Relative links resolve from the *linking file's* directory, which is the
     rule that `appendix/b-references.md` from inside `book/part-2-pretraining/`
     silently violated for 232 links.
  3. Every `#anchor` on an internal link matches a heading in the target file,
     using GitHub's slug algorithm.

Exit code is non-zero if anything is broken, so CI fails loudly.

    python3 tools/check_links.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Matches [text](target) but not ![image](target), and skips external links.
LINK_RE = re.compile(r"(?<!!)\[(?:[^\]]*)\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)
FENCE_RE = re.compile(r"^(```|~~~)")


def slugify(heading: str) -> str:
    """Reproduce GitHub's heading -> anchor transformation."""
    # Strip Markdown inline formatting and links, keeping the visible text.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").replace("*", "").replace("_", "")
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.strip().lower().replace(" ", "-")


def strip_code_fences(text: str) -> str:
    """Blank out fenced code blocks so example links inside them aren't checked."""
    out, in_fence = [], False
    for line in text.splitlines():
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


def anchors_for(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path not in cache:
        found: set[str] = set()
        seen: Counter[str] = Counter()
        for _, heading in HEADING_RE.findall(strip_code_fences(path.read_text())):
            base = slugify(heading)
            if not base:
                continue
            # GitHub disambiguates duplicate headings with -1, -2, ...
            found.add(base if seen[base] == 0 else f"{base}-{seen[base]}")
            seen[base] += 1
        cache[path] = found
    return cache[path]


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "SUMMARY.md"]
    for sub in ("book", "exercises", "labs"):
        files.extend(sorted((ROOT / sub).rglob("*.md")))
    for extra in ("CONTRIBUTING.md", "CODE_OF_CONDUCT.md"):
        if (ROOT / extra).exists():
            files.append(ROOT / extra)
    return [f for f in files if f.exists()]


def check() -> list[str]:
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    # --- 1. SUMMARY.md targets exist -------------------------------------
    summary = ROOT / "SUMMARY.md"
    if summary.exists():
        for target in LINK_RE.findall(strip_code_fences(summary.read_text())):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if not (ROOT / target.split("#")[0]).exists():
                errors.append(f"SUMMARY.md -> missing file: {target}")

    # --- 2 & 3. relative links and anchors --------------------------------
    for md in markdown_files():
        body = strip_code_fences(md.read_text())
        rel = md.relative_to(ROOT)
        for target in LINK_RE.findall(body):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            file_part, _, anchor = target.partition("#")
            if not file_part:
                continue
            resolved = (md.parent / file_part).resolve()
            if not resolved.exists():
                errors.append(f"{rel} -> missing file: {target}")
                continue
            if anchor and resolved.suffix == ".md":
                if anchor not in anchors_for(resolved, anchor_cache):
                    errors.append(f"{rel} -> missing anchor: {target}")

    # SUMMARY.md is walked twice (once for rule 1, once for rules 2-3); the two
    # passes agree on missing files, so collapse duplicates before reporting.
    return list(dict.fromkeys(errors))


def main() -> int:
    errors = check()
    checked = len(markdown_files())
    if errors:
        print(f"FAIL: {len(errors)} broken link(s) across {checked} Markdown files\n")
        for e in errors:
            print(f"  {e}")
        return 1
    print(f"OK: all internal links resolve across {checked} Markdown files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
