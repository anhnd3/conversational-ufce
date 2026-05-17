#!/usr/bin/env python3
"""
Final force-flip audit wrapper for Part I thesis evidence.

Distinguishes raw UFCE candidates from strict valid flip recourse by running
the trace harness with --flip-filter 1 --strict flags, then aggregates counts
into required summary JSON fields.

Current source: wraps scripts/run_ufce_trace_harness.py with specific flags
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_force_flip_audit(*, dataset: str, out_dir: Path, args_extra: list[str] = None) -> Dict[str, Any]:
    """Run trace harness with flip-filter and strict mode enabled."""
    
    if args_extra is None:
        args_extra = []
        
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_ufce_trace_harness.py"),
        "--dataset", dataset,
        "--flip-filter", "1",
        "--strict",
        "--out-dir", str(out_dir),
        *args_extra,
    ]
    
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    
    return {
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": [line.rstrip() for line in completed.stdout.splitlines()[-40:]],
        "stderr_tail": [line.rstrip() for line in completed.stderr.splitlines()[-40:]],
    }


def aggregate_force_flip_summary(out_dir: Path) -> Dict[str, Any]:
    """Parse trace harness outputs to extract raw vs valid flip counts."""
    
    summary = {
        "raw_candidate_count": None,
        "flip_valid_candidate_count": None, 
        "invalid_or_nonflip_count": None,
        "dataset_breakdown": {},
        "warnings": [],
    }
    
    # Look for run_meta.json or method_stats in output
    meta_path = out_dir / "run_meta.json"
    if not meta_path.exists():
        summary["warnings"].append("No run_meta.json found; could not extract flip counts")
        return summary
        
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # Extract from method_stats if present
        stats = meta.get("method_stats", {})
        for method, info in stats.items():
            if isinstance(info, dict):
                raw = info.get("raw_candidate_count") or info.get("total_candidates")
                valid = info.get("valid_recourse_count") or info.get("flip_valid_candidate_count")
                
                summary["dataset_breakdown"][method] = {
                    "raw": raw,
                    "valid_flip": valid,
                }
                
        if not summary["dataset_breakdown"]:
            summary["warnings"].append("No candidate counts extracted from run_meta.json; fields remain null")
            
    except Exception as e:
        summary["warnings"].append(f"Error parsing trace outputs: {e}")
        
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force-flip audit wrapper with strict validation.")
    parser.add_argument("--dataset", required=True, choices=["grad", "bank", "wine", "bupa", "movie", "all"], help="Dataset")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dataset == "all":
        datasets = ["grad", "bank", "wine", "bupa", "movie"]
        failures = 0
        for ds in datasets:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset",
                ds,
            ]
            if args.out_dir is not None:
                cmd.extend(["--out-dir", str(args.out_dir)])
            completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
            if completed.returncode != 0:
                failures += 1
        return 0 if failures == 0 else 1
    
    if args.out_dir is None:
        args.out_dir = (ROOT / "outputs" / "final" / "part1" / "force_flip" / 
                       f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        args.out_dir = args.out_dir / args.dataset
        
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the harness with flip-filter enabled and strict mode
    result = run_force_flip_audit(
        dataset=args.dataset,
        out_dir=args.out_dir,
    )
    
    # Aggregate summary with required fields
    summary = aggregate_force_flip_summary(args.out_dir)
    
    # Write provenance outputs
    provenance = {
        "title": f"Force-Flip Audit: {args.dataset}",
        "run_id": args.out_dir.name,
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), 
                                     text=True, stdout=subprocess.PIPE).stdout.strip(),
        "python_executable": sys.executable,
        "command_line": " ".join(sys.argv),
        "dataset": args.dataset,
    }
    
    # Write summary.json with required fields
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    
    # Write command.txt
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    
    print(json.dumps({"status": "passed" if result["passed"] else "failed", 
                       "exit_code": result["exit_code"],
                       "summary_path": str(args.out_dir / "summary.json")}, indent=2))
                       
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
