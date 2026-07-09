#!/usr/bin/env python3
"""Run the four canonical Part I thesis UFCE pipelines in order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = ROOT / "scripts" / "final" / "part1"

STEPS = [
    ("01_author_raw", SCRIPT_DIR / "01_author_raw_reproduction.py"),
    ("02_ufce_core", SCRIPT_DIR / "02_ufce_core_final_freeze.py"),
    ("03_post_hoc", SCRIPT_DIR / "03_ufce_post_hoc.py"),
    ("04_ufce_ff", SCRIPT_DIR / "04_ufce_ff.py"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical Part I final UFCE bundle.")
    parser.add_argument("--dataset", default="all", help="{bank,bupa,grad,wine,movie,all} or comma-list")
    parser.add_argument("--runtime-profile", "--runtime_profile", dest="runtime_profile", default="final_freeze")
    parser.add_argument("--bundle-mode", "--bundle_mode", dest="bundle_mode", default="table7_author_public")
    parser.add_argument("--max-folds", "--max_folds", dest="max_folds", type=int, default=0)
    parser.add_argument("--fold-file", "--fold_file", dest="fold_file", default=None)
    parser.add_argument("--mi-k", dest="mi_k", default="5")
    parser.add_argument("--no-cf", "--no_cf", dest="no_cf", type=int, default=10)
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/final/part1/final_ufce_bundle")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def step_args(args: argparse.Namespace, step_name: str, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(dict(STEPS)[step_name]),
        "--dataset",
        str(args.dataset),
        "--bundle-mode",
        str(args.bundle_mode),
        "--max-folds",
        str(args.max_folds),
        "--no-cf",
        str(args.no_cf),
        "--out-dir",
        str(out_dir / step_name),
    ]
    if step_name in {"02_ufce_core", "04_ufce_ff"}:
        cmd.extend(["--runtime_profile", str(args.runtime_profile)])
    if step_name in {"01_author_raw", "03_post_hoc"}:
        cmd.extend(["--mi-k", str(args.mi_k)])
    if args.fold_file:
        cmd.extend(["--fold-file", str(args.fold_file)])
    if args.no_progress and step_name in {"01_author_raw", "03_post_hoc"}:
        cmd.append("--no-progress")
    return cmd


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "status": "running",
        "run_started_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(out_dir),
        "dataset": str(args.dataset),
        "runtime_profile": str(args.runtime_profile),
        "bundle_mode": str(args.bundle_mode),
        "steps": [],
    }

    for step_name, _script_path in STEPS:
        cmd = step_args(args, step_name, out_dir)
        step_record = {"step": step_name, "command": cmd, "status": "running"}
        manifest["steps"].append(step_record)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        step_record["returncode"] = int(completed.returncode)
        step_record["status"] = "ok" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_step"] = step_name
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return int(completed.returncode)

    manifest["status"] = "ok"
    manifest["run_finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["table_source_map"] = {
        "author_raw": "01_author_raw",
        "ufce_core_final_freeze": "02_ufce_core",
        "post_hoc_valid_only": "03_post_hoc",
        "ufce_ff": "04_ufce_ff",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Part I final UFCE bundle written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
