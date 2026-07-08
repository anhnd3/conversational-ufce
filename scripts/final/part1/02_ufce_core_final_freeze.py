#!/usr/bin/env python3
"""Canonical Route 1 step 02: thesis UFCE core final-freeze reproduction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final.part1 import ufce_only_reproduction as runner


def _ensure_arg(flag: str, value: str) -> None:
    if any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
        return
    sys.argv[1:1] = [flag, value]


def _normalize_aliases() -> None:
    aliases = {
        "--runtime-profile": "--runtime_profile",
        "--max-folds": "--max_folds",
        "--fold-file": "--fold_file",
        "--no-cf": "--no_cf",
    }
    for index, arg in enumerate(sys.argv):
        if arg in aliases:
            sys.argv[index] = aliases[arg]
            continue
        for src, dst in aliases.items():
            prefix = src + "="
            if arg.startswith(prefix):
                sys.argv[index] = dst + "=" + arg[len(prefix):]
                break


def main() -> int:
    _normalize_aliases()
    _ensure_arg("--runtime_profile", "final_freeze")
    _ensure_arg("--bundle-mode", "table7_author_public")
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
