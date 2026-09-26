#!/usr/bin/env python3
"""Render the book's real-config blocks from committed snapshots.

A block presented as a real model's configuration is never typed by hand
(docs/adr/0003). In the Markdown it sits between two markers:

    <!-- real-config: qwen3-tokenizer -->
    ... everything in here is generated; do not edit ...
    <!-- /real-config -->

This script fills each marked region from tools/config_snapshots/<id>.json,
which tools/fetch_config_snapshots.py wrote from the published file at a
pinned commit, after verifying its hash. Rendering is offline and
deterministic, so CI can run it.

    python3 tools/build_configs.py           # regenerate every block
    python3 tools/build_configs.py --check   # fail if any block is stale (CI)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK_RE = re.compile(r"(<!-- real-config: ([\w.-]+) -->\n)(.*?)(<!-- /real-config -->)", re.S)


def render_tokenizer(snap: dict) -> str:
    tok, cfg, prov = snap["tokenizer"], snap.get("config", {}), snap["provenance"]
    excerpt = {
        "model": {
            "type": tok["model_type"],
            "byte_fallback": tok["byte_fallback"],
            "vocab": f"<{tok['vocab_entries']:,} entries>",
            "merges": f"<{tok['merges']:,} rules>",
        },
        "added_tokens": f"<{tok['added_tokens']:,} entries, {tok['special_tokens']:,} of them special>",
        "added_tokens (first special, by id)": {str(t["id"]): t["content"] for t in tok["special_tokens_head"]},
        "normalizer": tok["normalizer"],
        "pre_tokenizer": tok["pre_tokenizer"],
        "decoder": {"type": tok["decoder_type"]},
    }
    config = {k: v for k, v in cfg.items() if k in ("vocab_size", "tie_word_embeddings")}
    files = prov["files"]
    commit = prov["revision"][:7]
    via = "" if prov["downloaded_from"].startswith(prov["repo"] + "@") else (
        f" Downloaded from the ungated mirror `{prov['downloaded_from'].split('@')[0]}`, whose bytes match the official file's hash."
    )
    return (
        f"```json\n// {files['tokenizer']['path']} (excerpt)\n{json.dumps(excerpt, indent=2, ensure_ascii=False)}\n"
        f"// {files['config']['path']} (excerpt)\n{json.dumps(config, indent=2)}\n```\n\n"
        f"*Real config. Generated from `{prov['repo']}` at commit `{commit}` "
        f"(retrieved {prov['retrieved']}) by `tools/build_configs.py`; do not edit by hand.{via}*\n"
    )


RENDERERS = {"tokenizer": render_tokenizer}


def render(snapshot_id: str, root: Path) -> str:
    path = root / "tools/config_snapshots" / f"{snapshot_id}.json"
    if not path.exists():
        raise KeyError(snapshot_id)
    snap = json.loads(path.read_text())
    kind = next(k for k in RENDERERS if k in snap)
    return RENDERERS[kind](snap)


def main(argv: list[str], root: Path = ROOT) -> int:
    check = "--check" in argv
    problems: list[str] = []
    rendered = 0
    for md in sorted(root.glob("book/**/*.md")) + sorted(root.glob("exercises/**/*.md")):
        text = md.read_text()
        if "<!-- real-config:" not in text:
            continue

        def fill(m: re.Match) -> str:
            nonlocal rendered
            try:
                body = render(m.group(2), root)
            except KeyError:
                problems.append(f"{md.relative_to(root)}: no snapshot named '{m.group(2)}' in tools/config_snapshots/")
                return m.group(0)
            rendered += 1
            if check and m.group(3) != body:
                problems.append(f"{md.relative_to(root)}: real-config block '{m.group(2)}' does not match its snapshot")
            return m.group(1) + body + m.group(4)

        new = BLOCK_RE.sub(fill, text)
        if not check and new != text:
            md.write_text(new)
    if problems:
        print(f"FAIL: {len(problems)} real-config problem(s)\n")
        for p in problems:
            print(f"  {p}")
        if check:
            print("\nRun: python3 tools/build_configs.py")
        return 1
    print(f"OK: {rendered} real-config block(s) {'match their snapshots' if check else 'rendered'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
