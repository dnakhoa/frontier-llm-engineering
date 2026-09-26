#!/usr/bin/env python3
"""Fetch the published files behind the book's real-config blocks.

Real-config blocks in the book are never typed by hand (docs/adr/0003). This
script downloads each file listed in tools/config_sources.json at its pinned
commit, verifies its bytes, and writes a small snapshot of the fields the book
shows to tools/config_snapshots/<id>.json. tools/build_configs.py then renders
the blocks from those snapshots, offline, so CI never needs the network or a
token.

Verification: every downloaded file must match the hash the *official* repo
reports for that path at the pinned commit: the Git LFS sha256 for LFS files,
the git blob id for the rest. That is what makes
a mirror acceptable. For a gated repo such as meta-llama/*, the Hub still
publishes the tree, so we can download from an ungated mirror and prove the
bytes are Meta's. A mismatch is a hard failure, never a warning.

    python3 tools/fetch_config_snapshots.py            # refresh every snapshot
    python3 tools/fetch_config_snapshots.py qwen3      # sources matching a substring

Needs the network; run by hand when a source or its pinned commit changes,
then commit the snapshots and run build_configs.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "tools/config_sources.json"
SNAPSHOTS = ROOT / "tools/config_snapshots"
HUB = "https://huggingface.co"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "frontier-llm-engineering/fetch_config_snapshots"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def official_hashes(repo: str, revision: str) -> dict[str, tuple[str, str]]:
    """path -> ("sha256", lfs oid) for LFS files, ("git_blob", blob id) otherwise."""
    tree = json.loads(get(f"{HUB}/api/models/{repo}/tree/{revision}"))
    out = {}
    for entry in tree:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs")
        out[entry["path"]] = ("sha256", lfs["oid"]) if lfs else ("git_blob", entry["oid"])
    return out


def extract_tokenizer(tok: dict) -> dict:
    """The fields Chapter 4 shows, taken verbatim from tokenizer.json."""
    model = tok["model"]
    special = [t for t in tok.get("added_tokens", []) if t.get("special")]
    return {
        "model_type": model.get("type"),
        "byte_fallback": model.get("byte_fallback"),
        "vocab_entries": len(model.get("vocab", {})),
        "merges": len(model.get("merges", [])),
        "added_tokens": len(tok.get("added_tokens", [])),
        "special_tokens": len(special),
        # the first few special tokens, by id, as the file lists them
        "special_tokens_head": [{"id": t["id"], "content": t["content"]} for t in sorted(special, key=lambda t: t["id"])[:6]],
        "normalizer": tok.get("normalizer"),
        "pre_tokenizer": tok.get("pre_tokenizer"),
        "decoder_type": (tok.get("decoder") or {}).get("type"),
    }


def extract_config(cfg: dict) -> dict:
    keys = ("vocab_size", "tie_word_embeddings", "bos_token_id", "eos_token_id")
    return {k: cfg.get(k) for k in keys}


EXTRACTORS = {"tokenizer": extract_tokenizer, "config": extract_config}


def fetch(source: dict) -> dict:
    repo, rev = source["repo"], source["revision"]
    official = official_hashes(repo, rev)
    dl_repo, dl_rev = (source["via"]["repo"], source["via"]["revision"]) if "via" in source else (repo, rev)
    snapshot: dict = {
        "id": source["id"],
        "provenance": {
            "repo": repo,
            "revision": rev,
            "downloaded_from": f"{dl_repo}@{dl_rev}",
            "retrieved": dt.date.today().isoformat(),
            "files": {},
        },
    }
    for kind, path in source["files"].items():
        data = get(f"{HUB}/{dl_repo}/resolve/{dl_rev}/{path}")
        hashes = {"git_blob": git_blob_id(data), "sha256": hashlib.sha256(data).hexdigest()}
        if path not in official:
            raise SystemExit(f"{source['id']}: {repo}@{rev} has no {path}")
        algo, expected = official[path]
        if hashes[algo] != expected:
            raise SystemExit(
                f"{source['id']}: {path} from {dl_repo}@{dl_rev} has {algo} {hashes[algo]}, but {repo}@{rev} "
                f"lists {expected}. Refusing to snapshot a file we cannot prove is the official one."
            )
        snapshot["provenance"]["files"][kind] = {"path": path, "verified_by": algo, **hashes}
        snapshot[kind] = EXTRACTORS[kind](json.loads(data))
    return snapshot


def main(argv: list[str]) -> int:
    sources = json.loads(SOURCES.read_text())["sources"]
    wanted = [s for s in sources if not argv or any(a in s["id"] for a in argv)]
    SNAPSHOTS.mkdir(exist_ok=True)
    for source in wanted:
        snap = fetch(source)
        out = SNAPSHOTS / f"{source['id']}.json"
        out.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out.relative_to(ROOT)}  ({snap['provenance']['downloaded_from']}, blobs verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
