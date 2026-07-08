#!/usr/bin/env python3
"""Canonical Route 1 step 01: raw author-code UFCE reproduction."""

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
    _ensure_arg("--core-package", "ufce.core_author")
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
