#!/usr/bin/env python3
"""Convert the `.py` labs into Colab-ready `.ipynb` notebooks.

Labs are authored as plain Python so that they stay diffable, greppable, and
copy-pasteable straight into a Colab cell. This script turns them into real
notebooks so readers can also click "Open in Colab" and run them cell by cell.

Cell format (the jupytext / VS Code convention):

    # %% [markdown]
    # This becomes a Markdown cell. The leading "# " is stripped.

    # %%
    # This becomes a code cell.
    print("hello")

Usage:

    python3 tools/build_notebooks.py           # rebuild every notebook
    python3 tools/build_notebooks.py --check   # fail if any is out of date (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "labs"
GITHUB_REPO = "dnakhoa/frontier-llm-engineering"
BRANCH = "main"


def split_cells(source: str) -> list[tuple[str, str]]:
    """Split lab source into (cell_type, body) pairs."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))
        buf.clear()

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        buf.append(line)
    flush()
    return cells


def demote_markdown(body: str) -> str:
    """Strip the leading comment marker from a Markdown cell's lines."""
    out = []
    for line in body.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def to_notebook(py_path: Path) -> dict:
    rel = py_path.relative_to(ROOT).as_posix()
    colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{BRANCH}/{rel[:-3]}.ipynb"
    badge = (
        f'<a href="{colab_url}" target="_blank">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        f'alt="Open In Colab"/></a>'
    )

    cells: list[dict] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (badge + "\n"),
        }
    ]
    for kind, body in split_cells(py_path.read_text()):
        if kind == "markdown":
            cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": demote_markdown(body) + "\n"}
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": body + "\n",
                }
            )

    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify, don't write")
    args = parser.parse_args()

    if not LABS.exists():
        print("no labs/ directory yet — nothing to build")
        return 0

    stale: list[str] = []
    built = 0
    for py in sorted(LABS.rglob("*.py")):
        if py.name.startswith("_"):
            continue
        nb_path = py.with_suffix(".ipynb")
        rendered = json.dumps(to_notebook(py), indent=1, ensure_ascii=False) + "\n"
        if args.check:
            if not nb_path.exists() or nb_path.read_text() != rendered:
                stale.append(str(nb_path.relative_to(ROOT)))
        else:
            nb_path.write_text(rendered)
        built += 1

    if args.check and stale:
        print(f"FAIL: {len(stale)} notebook(s) out of date. Run tools/build_notebooks.py:\n")
        for s in stale:
            print(f"  {s}")
        return 1

    verb = "verified" if args.check else "wrote"
    print(f"OK: {verb} {built} notebook(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
