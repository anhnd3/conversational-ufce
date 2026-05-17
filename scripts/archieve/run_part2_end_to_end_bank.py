#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import (
    add_summary_output_args,
    build_runner_command,
    lm_studio_preflight,
    progress_iter,
    write_optional_summary_outputs,
)
from llm.src.part2_eval.corpora import G5_AGENT_PORTABILITY_CORPUS_PATH, TIER_C_CORPUS_PATH
from llm.src.product.config import ProductConfig
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_closeout"
RUNNER_SCOPE = "part2_bank_automation_closeout"
PYTEST_TARGETS = [
    "llm/tests/test_part2_eval_corpora.py",
    "llm/tests/test_part2_eval_common.py",
    "llm/tests/test_part2_thesis_metrics_report.py",
    "llm/tests/test_part2_refinement_metrics_report.py",
    "llm/tests/test_part2_backend_comparison_report.py",
    "llm/tests/test_part2_agent_portability_report.py",
    "llm/tests/test_part2_replay_robustness_report.py",
    "llm/tests/test_part2_end_to_end_bank.py",
]


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Run the Part II serverless automation closeout.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-catalog", type=Path, default=ROOT / "docs" / "validation" / "catalogs" / "phase3_2_validation_catalog_v1.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml")
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--tier-a-corpus", type=Path, default=ROOT / "docs" / "validation" / "corpora" / "part2_tier_a_bank_annotations_v1.json")
    parser.add_argument("--tier-b-corpus", type=Path, default=ROOT / "docs" / "validation" / "corpora" / "part2_tier_b_bank_sessions_v1.json")
    parser.add_argument("--tier-c-corpus", type=Path, default=TIER_C_CORPUS_PATH)
    parser.add_argument("--tier-d-corpus", type=Path, default=ROOT / "docs" / "validation" / "corpora" / "part2_tier_d_bank_replay_v1.json")
    parser.add_argument("--g5-corpus", type=Path, default=G5_AGENT_PORTABILITY_CORPUS_PATH)
    parser.add_argument("--g5-backends", default="ufce,dice,ar")
    parser.add_argument("--g5-attempts-per-case", type=int, default=3)
    parser.add_argument("--g5-case-limit", type=int, default=None)
    parser.add_argument("--backend-seed-id", default="TIERC-001")
    parser.add_argument("--backend-seed-index", type=int, default=None)
    parser.add_argument("--full-tier-c", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_closeout(args=args, command=command)
    markdown = render_markdown(summary)
    write_optional_summary_outputs(
        summary=summary,
        summary_json_path=args.summary_json,
        summary_markdown_path=args.summary_md,
        markdown_text=markdown,
    )
    if args.summary_json is None:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print(f"summary_json_path={Path(args.summary_json).resolve()}")
    return 0 if summary["closeout_passed"] else 1


def run_closeout(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    run_id = "part2_closeout_" + local_now_compact()
    run_root = args.out_dir / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    stage_steps = [
        (
            "targeted_pytest",
            lambda: run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *PYTEST_TARGETS,
                    "-q",
                ],
                name="targeted_pytest",
            ),
        ),
        (
            "lm_studio_preflight",
            lambda: lm_studio_preflight(api_base=args.lm_studio_api_base, model_alias=args.model_alias),
        ),
        (
            "thesis_metrics_report",
            lambda: run_script_step(
                name="thesis_metrics_report",
                script=ROOT / "scripts" / "run_part2_thesis_metrics_report.py",
                out_dir=run_root / "thesis_metrics",
                args=args,
            ),
        ),
        (
            "refinement_metrics_report",
            lambda: run_script_step(
                name="refinement_metrics_report",
                script=ROOT / "scripts" / "run_part2_refinement_metrics_report.py",
                out_dir=run_root / "refinement_metrics",
                args=args,
            ),
        ),
        (
            "backend_comparison_report",
            lambda: run_script_step(
                name="backend_comparison_report",
                script=ROOT / "scripts" / "run_part2_backend_comparison_report.py",
                out_dir=run_root / "backend_comparison",
                args=args,
            ),
        ),
        (
            "agent_portability_report",
            lambda: run_script_step(
                name="agent_portability_report",
                script=ROOT / "scripts" / "run_part2_agent_portability_report.py",
                out_dir=run_root / "agent_portability",
                args=args,
            ),
        ),
        (
            "replay_robustness_report",
            lambda: run_script_step(
                name="replay_robustness_report",
                script=ROOT / "scripts" / "run_part2_replay_robustness_report.py",
                out_dir=run_root / "replay_robustness",
                args=args,
            ),
        ),
    ]
    child_runs: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    for stage_name, action in progress_iter(
        stage_steps,
        enabled=not args.no_progress,
        desc="Part II closeout",
        unit="stage",
        total=len(stage_steps),
    ):
        result = action()
        if stage_name == "lm_studio_preflight":
            preflight = result
            if not preflight["ok"]:
                break
        else:
            child_runs.append(result)
            if not result["passed"]:
                break
    if preflight is None:
        preflight = {
            "ok": False,
            "detail": "not_run_due_to_prior_stage_failure",
            "api_base": args.lm_studio_api_base,
            "model_alias": args.model_alias,
            "model_alias_present": None,
            "available_models": [],
        }

    report_runs = [item for item in child_runs if item.get("payload") is not None]
    report_validations = [
        {
            "name": item["name"],
            "aggregate_validation_ok": ((item["payload"].get("aggregate_validation") or {}).get("ok") is True),
            "corpus_path": item["payload"].get("corpus_path"),
            "report_json_path": item["payload"].get("report_json_path"),
            "report_markdown_path": item["payload"].get("report_markdown_path"),
            "corpus_version": item["payload"].get("corpus_version"),
            "corpus_sha256": item["payload"].get("corpus_sha256"),
            "loaded_corpora": item["payload"].get("loaded_corpora"),
        }
        for item in report_runs
    ]
    closeout_passed = preflight["ok"] and all(item["passed"] for item in child_runs) and all(
        item["aggregate_validation_ok"] for item in report_validations
    )
    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "closeout_root": str(run_root.resolve()),
        "lm_studio_preflight": preflight,
        "child_runs": child_runs,
        "report_validations": report_validations,
        "closeout_passed": closeout_passed,
    }
    write_json(run_root / "part2_closeout_summary.json", summary)
    (run_root / "part2_closeout_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def run_script_step(*, name: str, script: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_json_path = out_dir.parent / f"{name}_summary.json"
    summary_markdown_path = out_dir.parent / f"{name}_summary.md"
    command = [
        sys.executable,
        str(script),
        "--out-dir",
        str(out_dir),
        "--baseline-catalog",
        str(args.baseline_catalog),
        "--benchmark",
        str(args.benchmark),
        "--lm-studio-api-base",
        args.lm_studio_api_base,
        "--model-alias",
        args.model_alias,
        "--product-mode",
        args.product_mode,
        "--api-version",
        args.api_version,
        "--app-version",
        args.app_version,
        "--summary-json",
        str(summary_json_path),
        "--summary-md",
        str(summary_markdown_path),
    ]
    if args.no_progress:
        command.append("--no-progress")
    if script.name == "run_part2_thesis_metrics_report.py":
        command.extend(["--tier-a-corpus", str(args.tier_a_corpus), "--tier-b-corpus", str(args.tier_b_corpus)])
    if script.name == "run_part2_refinement_metrics_report.py":
        command.extend(["--tier-a-corpus", str(args.tier_a_corpus), "--tier-b-corpus", str(args.tier_b_corpus)])
    if script.name == "run_part2_backend_comparison_report.py":
        command = [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--baseline-catalog",
            str(args.baseline_catalog),
            "--tier-c-corpus",
            str(args.tier_c_corpus),
            "--summary-json",
            str(summary_json_path),
            "--summary-md",
            str(summary_markdown_path),
        ]
        if not args.full_tier_c:
            command.extend(["--single-case"])
            if args.backend_seed_index is not None:
                command.extend(["--seed-index", str(args.backend_seed_index)])
            else:
                command.extend(["--seed-id", args.backend_seed_id])
        if args.no_progress:
            command.append("--no-progress")
    if script.name == "run_part2_replay_robustness_report.py":
        command.extend(["--tier-d-corpus", str(args.tier_d_corpus)])
    if script.name == "run_part2_agent_portability_report.py":
        command.extend(
            [
                "--g5-corpus",
                str(args.g5_corpus),
                "--backends",
                args.g5_backends,
                "--attempts-per-case",
                str(args.g5_attempts_per_case),
            ]
        )
        if args.g5_case_limit is not None:
            command.extend(["--case-limit", str(args.g5_case_limit)])
    result = run_command(
        command,
        name=name,
        stream_stderr=True,
        summary_json_path=summary_json_path,
    )
    result["summary_json_path"] = str(summary_json_path)
    result["summary_markdown_path"] = str(summary_markdown_path)
    return result


def run_command(
    command: list[str],
    *,
    name: str | None = None,
    stream_stderr: bool = False,
    summary_json_path: Path | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None if stream_stderr else subprocess.PIPE,
    )
    payload = None
    parse_error = None
    if summary_json_path is not None and completed.returncode == 0:
        try:
            payload = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    passed = completed.returncode == 0 and (summary_json_path is None or payload is not None)
    return {
        "name": name or Path(command[0]).name,
        "command": " ".join(shlex.quote(item) for item in command),
        "exit_code": completed.returncode,
        "passed": passed,
        "payload": payload,
        "parse_error": parse_error,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": [] if stream_stderr else tail_text(completed.stderr or ""),
    }


def tail_text(text: str, *, line_limit: int = 20) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-line_limit:]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Closeout Summary",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- closeout_root: `{summary['closeout_root']}`",
        "",
        "## LM Studio Preflight",
        "",
        f"- ok: `{summary['lm_studio_preflight']['ok']}`",
        f"- detail: `{summary['lm_studio_preflight']['detail']}`",
        f"- model_alias_present: `{summary['lm_studio_preflight']['model_alias_present']}`",
        "",
        "## Child Runs",
        "",
    ]
    for item in summary["child_runs"]:
        lines.extend(
            [
                f"- {item['name']}: passed=`{item['passed']}` exit_code=`{item['exit_code']}`",
                f"  command: `{item['command']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Report Validation",
            "",
        ]
    )
    for item in summary["report_validations"]:
        lines.append(
            f"- {item['name']}: aggregate_validation_ok=`{item['aggregate_validation_ok']}` corpus_version=`{item['corpus_version']}`"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- closeout_passed: `{summary['closeout_passed']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
