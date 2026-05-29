#!/usr/bin/env python3
"""
Checkpointed runner for public-config MI pool expansion.

Each MI setting is executed as a separate 07 UFCE-FF child run.
This gives us a durable checkpoint after every full-dataset MI setting, so a
power cut only loses the currently running MI stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUNNER_07 = ROOT / "scripts" / "final" / "part1" / "07_ufce_ff_experiment.py"
DEFAULT_OUT_PARENT = ROOT / "outputs" / "final" / "part1"
DEFAULT_MI_K_LIST = ["5", "10", "15", "20", "all"]


def now_tag() -> str:
    return datetime.now().strftime("09_pool_expansion_checkpoints_%Y%m%d_%H%M%S")


def parse_mi_k_list(value: str) -> List[str]:
    tokens: List[str] = []
    for raw in str(value).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token != "all":
            number = int(token)
            if number <= 0:
                raise ValueError(f"MI-k must be positive, got {token}")
            token = str(number)
        tokens.append(token)
    if not tokens:
        raise ValueError("At least one MI-k token is required.")
    return tokens


def mi_stage_name(mi_k: str) -> str:
    return f"mi_{str(mi_k).replace(':', '_').replace('/', '_')}"


def mode_for_mi(mi_k: str) -> str:
    return "pool_expansion_forceflip"


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def stage_is_complete(checkpoint_path: Path, stage_dir: Path) -> bool:
    checkpoint = read_json(checkpoint_path)
    if not checkpoint or checkpoint.get("status") != "completed":
        return False
    required = [
        stage_dir / "ff_selector_comparison.csv",
        stage_dir / "ff_failure_taxonomy.csv",
        stage_dir / "summary.json",
        stage_dir / "summary.md",
    ]
    return all(path.exists() for path in required)


def refresh_combined_outputs(output_root: Path, mi_k_list: List[str]) -> None:
    frames: List[pd.DataFrame] = []
    checkpoint_rows: List[Dict] = []
    for mi_k in mi_k_list:
        stage = mi_stage_name(mi_k)
        stage_dir = output_root / stage
        checkpoint_path = output_root / "checkpoints" / f"{stage}.json"
        checkpoint = read_json(checkpoint_path) or {}
        checkpoint_rows.append(
            {
                "mi_k": mi_k,
                "stage": stage,
                "status": checkpoint.get("status", "not_started"),
                "output_dir": str(stage_dir),
                "started_at": checkpoint.get("started_at"),
                "completed_at": checkpoint.get("completed_at"),
                "duration_sec": checkpoint.get("duration_sec"),
                "exit_code": checkpoint.get("exit_code"),
            }
        )
        summary_csv = stage_dir / "ff_selector_comparison.csv"
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            df.insert(0, "checkpoint_stage", stage)
            df.insert(1, "checkpoint_mi_k", mi_k)
            frames.append(df)

    pd.DataFrame(checkpoint_rows).to_csv(output_root / "checkpoint_status.csv", index=False)

    lines = [
        "# Pool Expansion Checkpoint Status",
        "",
        "| MI-k | Stage | Status | Duration sec | Output dir |",
        "|---|---|---|---:|---|",
    ]
    for row in checkpoint_rows:
        duration = "" if row["duration_sec"] is None else f"{float(row['duration_sec']):.1f}"
        lines.append(
            f"| {row['mi_k']} | {row['stage']} | {row['status']} | {duration} | `{row['output_dir']}` |"
        )
    (output_root / "checkpoint_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_root / "combined_ff_selector_comparison.csv", index=False)
        wanted_modes = {"public_forceflip", "mi_expansion_diagnostic"}
        combined.loc[combined["mode"].isin(wanted_modes)].to_csv(
            output_root / "combined_pool_expansion_summary.csv",
            index=False,
        )


def build_child_command(args: argparse.Namespace, mi_k: str, stage_dir: Path) -> List[str]:
    mode = mode_for_mi(mi_k)
    cmd = [
        sys.executable,
        str(RUNNER_07),
        "--dataset",
        str(args.dataset),
        "--mode",
        mode,
        "--mi-k-list",
        str(mi_k),
        "--config-profile",
        str(args.config_profile),
        "--bundle-mode",
        str(args.bundle_mode),
        "--no_cf",
        str(args.no_cf),
        "--max_folds",
        str(args.max_folds),
        "--contprox_metric",
        str(args.contprox_metric),
        "--mi-feature-scope",
        str(args.mi_feature_scope),
        "--out-dir",
        str(stage_dir),
    ]
    if args.fold_file:
        cmd.extend(["--fold_file", str(args.fold_file)])
    if args.no_progress:
        cmd.append("--no-progress")
    return cmd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run checkpointed MI pool expansion diagnostics.")
    parser.add_argument("--dataset", default="all", help="Dataset name, comma-separated list, or all.")
    parser.add_argument("--mi-k-list", default=",".join(DEFAULT_MI_K_LIST), help="MI-k list, e.g. 5,10,15,20,all.")
    parser.add_argument(
        "--config-profile",
        default="public_github_source",
        choices=["public_github_source", "final_freeze", "new_best_params"],
        help="Runtime config source passed to 07.",
    )
    parser.add_argument(
        "--bundle-mode",
        default="table7_author_public",
        choices=["author_public", "table7_author_public", "domain_actionable"],
        help="UF/f2change/step bundle passed to 07.",
    )
    parser.add_argument("--no_cf", type=int, default=10)
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--fold_file", default=None)
    parser.add_argument("--contprox_metric", default="euclidean")
    parser.add_argument(
        "--mi-feature-scope",
        default="configured_actionable",
        choices=["configured_actionable", "uf_only", "all_features"],
        help="Feature-pair scope passed to 07. configured_actionable requires uf, step, and f2change.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rerun-completed", action="store_true", help="Rerun MI stages even when completed checkpoints exist.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars in child runs.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    mi_k_list = parse_mi_k_list(args.mi_k_list)
    output_root = Path(args.out_dir).resolve() if args.out_dir else (DEFAULT_OUT_PARENT / now_tag()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "runner_07": str(RUNNER_07),
        "dataset": str(args.dataset),
        "mi_k_list": mi_k_list,
        "config_profile": str(args.config_profile),
        "bundle_mode": str(args.bundle_mode),
        "no_cf": int(args.no_cf),
        "max_folds": int(args.max_folds),
        "fold_file": args.fold_file,
        "contprox_metric": str(args.contprox_metric),
        "mi_feature_scope": str(args.mi_feature_scope),
        "output_root": str(output_root),
        "checkpoint_policy": "one completed checkpoint after each full dataset sweep for one MI-k value",
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    refresh_combined_outputs(output_root, mi_k_list)

    for index, mi_k in enumerate(mi_k_list, start=1):
        stage = mi_stage_name(mi_k)
        stage_dir = output_root / stage
        checkpoint_path = output_root / "checkpoints" / f"{stage}.json"
        if not args.rerun_completed and stage_is_complete(checkpoint_path, stage_dir):
            print(f"[{index}/{len(mi_k_list)}] skip completed {stage}: {stage_dir}", flush=True)
            continue

        cmd = build_child_command(args, mi_k, stage_dir)
        started = time.perf_counter()
        started_at = datetime.now().isoformat(timespec="seconds")
        write_json(
            checkpoint_path,
            {
                "status": "running",
                "mi_k": mi_k,
                "stage": stage,
                "started_at": started_at,
                "command": cmd,
                "output_dir": str(stage_dir),
            },
        )
        refresh_combined_outputs(output_root, mi_k_list)

        print(f"[{index}/{len(mi_k_list)}] running {stage}", flush=True)
        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        duration = float(time.perf_counter() - started)
        completed_at = datetime.now().isoformat(timespec="seconds")

        if result.returncode != 0:
            write_json(
                checkpoint_path,
                {
                    "status": "failed",
                    "mi_k": mi_k,
                    "stage": stage,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_sec": duration,
                    "exit_code": int(result.returncode),
                    "command": cmd,
                    "output_dir": str(stage_dir),
                },
            )
            refresh_combined_outputs(output_root, mi_k_list)
            print(f"[{index}/{len(mi_k_list)}] failed {stage} exit_code={result.returncode}", flush=True)
            return int(result.returncode)

        write_json(
            checkpoint_path,
            {
                "status": "completed",
                "mi_k": mi_k,
                "stage": stage,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_sec": duration,
                "exit_code": int(result.returncode),
                "command": cmd,
                "output_dir": str(stage_dir),
            },
        )
        refresh_combined_outputs(output_root, mi_k_list)
        print(f"[{index}/{len(mi_k_list)}] completed {stage} in {duration:.1f}s", flush=True)

    refresh_combined_outputs(output_root, mi_k_list)
    print(f"done: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
