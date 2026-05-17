#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FINAL_ROOT = Path(__file__).resolve().parents[1]  # Insert scripts/final for _common.py import
if str(FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_ROOT))

from _common import ROOT, make_run_dir, run_command, summarize_wrapper_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final wrapper for black-box/model bundle regression checks.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "final" / "part1" / "blackbox_regression")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id, run_root = make_run_dir(args.out_dir, "part1_blackbox_regression")
    command = [sys.executable, "-m", "pytest", "ufce/tests/test_lr_bundle.py", "-q"]
    result = run_command(command, log_dir=run_root / "logs", log_prefix="pytest_lr_bundle")
    result["name"] = "ufce_lr_bundle_tests"
    summarize_wrapper_run(
        title="Part I Black-Box Regression Audit",
        run_id=run_id,
        run_root=run_root,
        results=[result],
        input_paths=[ROOT / "ufce" / "tests" / "test_lr_bundle.py", ROOT / "ufce" / "model_bundles" / "lr_bundle.py"],
        extra={"dataset": args.dataset, "model_validation_policy": "pytest_existing_lr_bundle_contracts"},
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

