#!/usr/bin/env python3
"""Run every lab and report which ones fail.

Labs are the part of this book most likely to rot: a library changes an API, a
dataset moves, a default flips. Running them all in CI on a plain CPU runner is
the only way to know the book still works for someone opening Colab today.

Every lab honours two environment variables:

  FLE_SMOKE_TEST=1   shrink every loop, dataset, and model to the smallest size
                     that still exercises the code path. CI sets this.
  FLE_DEVICE         force "cpu" or "cuda". Labs auto-detect when unset.

    python3 tools/run_labs.py              # run all
    python3 tools/run_labs.py lab04        # run labs matching a substring
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABS = ROOT / "labs"
TIMEOUT_SECONDS = 900


def main() -> int:
    if not LABS.exists():
        print("no labs/ directory yet — nothing to run")
        return 0

    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    labs = [p for p in sorted(LABS.rglob("*.py")) if not p.name.startswith("_")]
    labs = [p for p in labs if needle in str(p)]

    if not labs:
        print(f"no labs matched {needle!r}")
        return 1

    env = {**os.environ, "FLE_SMOKE_TEST": os.environ.get("FLE_SMOKE_TEST", "1"), "MPLBACKEND": "Agg"}
    failures: list[tuple[str, str]] = []

    for lab in labs:
        rel = lab.relative_to(ROOT)
        print(f"--> {rel}", flush=True)
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(lab)],
                cwd=lab.parent,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            failures.append((str(rel), f"timed out after {TIMEOUT_SECONDS}s"))
            print(f"    TIMEOUT after {TIMEOUT_SECONDS}s")
            continue

        elapsed = time.time() - start
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-20:]
            failures.append((str(rel), "\n".join(tail)))
            print(f"    FAIL ({elapsed:.1f}s)")
        else:
            print(f"    ok ({elapsed:.1f}s)")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{len(labs)} lab(s) failed\n")
        for name, detail in failures:
            print(f"--- {name} ---")
            print(detail)
            print()
        return 1

    print(f"OK: {len(labs)}/{len(labs)} labs ran clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
