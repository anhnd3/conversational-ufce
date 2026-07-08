#!/usr/bin/env python3
"""Canonical Route 1 step 03: UFCE post-hoc valid-only aggregation.

Post-hoc is an aggregate/evaluation view over the raw selected UFCE outputs.
It must not regenerate candidates through a separate core, otherwise the audit
can silently become UFCE-FF-like or drift from the raw reproduction path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final.part1 import author_raw_posthoc_replay as runner


def _ensure_arg(flag: str, value: str) -> None:
    if any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
        return
    sys.argv[1:1] = [flag, value]


def main() -> int:
    _ensure_arg("--core-package", "auto")
    _ensure_arg("--bundle-mode", "table7_author_public")
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
