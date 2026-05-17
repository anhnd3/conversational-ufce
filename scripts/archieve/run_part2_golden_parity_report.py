#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import (
    add_summary_output_args,
    build_runner_command,
    counter_dict,
    progress_iter,
    write_optional_summary_outputs,
)
from llm.src.product.config import try_get_git_commit
from llm.src.runtime.datasets import BankDatasetPackage
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_golden_parity"
RUNNER_SCOPE = "part2_bank_golden_parity"
SCORER_VERSION = "part2_golden_parity_report_v1"
EXPECTED_VERIFIER_DELTA = "expected_verifier_delta"
EXPECTED_RANKING_DELTA = "expected_ranking_delta"
UNEXPECTED_REGRESSION = "unexpected_regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Part II bank golden parity report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", default="ufce")
    parser.add_argument("--dataset", default="bank")
    parser.add_argument("--golden-corpus", type=Path, default=BankDatasetPackage().golden_parity_corpus_path())
    parser.add_argument("--unexpected-regression-waiver", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_golden_parity_report(args=args, command=command)
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
    return 0 if summary["parity_ok"] else 1


def run_golden_parity_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    if str(args.dataset).strip().lower() != "bank":
        raise ValueError("Only the bank golden parity corpus is currently supported.")

    corpus_path = Path(args.golden_corpus).resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    runtime = RuntimeOrchestrator(counterfactual_backend_name=str(args.backend).strip().lower())
    cases = [
        evaluate_case(
            runtime=runtime,
            case=case,
            backend_id=str(args.backend).strip().lower(),
        )
        for case in progress_iter(
            corpus["cases"],
            enabled=not args.no_progress,
            desc="Golden parity",
            unit="case",
            total=len(corpus["cases"]),
        )
    ]

    unexpected_regressions = [case for case in cases if case["delta_classification"] == UNEXPECTED_REGRESSION]
    waiver_path = None if args.unexpected_regression_waiver is None else Path(args.unexpected_regression_waiver).resolve()
    waiver_text = None
    waiver_applied = False
    if waiver_path is not None:
        waiver_text = waiver_path.read_text(encoding="utf-8").strip()
        waiver_applied = bool(waiver_text) and bool(unexpected_regressions)

    run_id = "part2_golden_parity_" + local_now_compact()
    run_root = Path(args.out_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    classification_counts = counter_dict(
        case["delta_classification"] for case in cases if case["delta_classification"] is not None
    )
    blocked = bool(unexpected_regressions) and not waiver_applied
    differences = [
        {
            "path": f"cases[{index}].delta_classification",
            "expected": "not_unexpected_regression",
            "actual": case["delta_classification"],
            "case_id": case["case_id"],
            "detail": case["delta_reason"],
        }
        for index, case in enumerate(cases)
        if case["delta_classification"] == UNEXPECTED_REGRESSION
    ]
    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "git_commit": try_get_git_commit(ROOT),
        "dataset": corpus.get("dataset"),
        "backend_id": str(args.backend).strip().lower(),
        "corpus_path": str(corpus_path),
        "corpus_version": corpus["corpus_version"],
        "corpus_sha256": sha256_file(corpus_path),
        "report_json_path": str((run_root / "golden_parity_report.json").resolve()),
        "report_markdown_path": str((run_root / "golden_parity_report.md").resolve()),
        "case_count": len(cases),
        "delta_classification_counts": classification_counts,
        "unexpected_regression_count": len(unexpected_regressions),
        "unexpected_regression_case_ids": [case["case_id"] for case in unexpected_regressions],
        "unexpected_regression_waiver": {
            "path": None if waiver_path is None else str(waiver_path),
            "applied": waiver_applied,
            "written_rationale": waiver_text,
        },
        "parity_ok": not blocked,
        "boundary_corpus_policy": {
            "included_in_closeout_metrics": False,
            "note": "Boundary corpus is diagnostic only and excluded from final closeout metrics.",
        },
        "cases": cases,
        "aggregate_validation": {
            "ok": not blocked,
            "difference_count": len(differences),
            "differences": differences,
        },
    }
    write_json(run_root / "golden_parity_report.json", summary)
    (run_root / "golden_parity_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def evaluate_case(*, runtime: RuntimeOrchestrator, case: dict[str, Any], backend_id: str) -> dict[str, Any]:
    try:
        result = runtime.handle(dict(case["runtime_request"]), include_debug_trace=True)
        payload = result.to_dict(include_debug_trace=True)
    except Exception as exc:  # pragma: no cover - defensive for live runs
        payload = {
            "controller_state": "EXECUTION_ERROR",
            "reason_codes": [],
            "counterfactual": None,
            "canonical_candidates": [],
            "verification_results": [],
            "backend_id": backend_id,
            "debug_trace": {"execution_error": f"{type(exc).__name__}: {exc}"},
        }

    actual_kind = classify_actual_kind(payload)
    delta_classification, delta_reason = classify_case_delta(
        expected_kind=str(case["kind"]),
        actual_kind=actual_kind,
        payload=payload,
        backend_id=backend_id,
    )
    selected_candidate_id = build_selected_candidate_id(payload=payload, backend_id=backend_id)
    verification_results = list(payload.get("verification_results") or [])
    invalid_candidate_count = sum(1 for item in verification_results if item.get("is_valid") is False)
    verification_reason_counts = counter_dict(
        reason_code
        for item in verification_results
        for reason_code in item.get("reason_codes") or []
    )
    return {
        "case_id": case["case_id"],
        "expected_kind": case["kind"],
        "actual_kind": actual_kind,
        "delta_classification": delta_classification,
        "delta_reason": delta_reason,
        "controller_state": payload.get("controller_state"),
        "reason_codes": list(payload.get("reason_codes") or []),
        "backend_id": payload.get("backend_id"),
        "reason_code_version": payload.get("reason_code_version"),
        "selected_candidate_id": selected_candidate_id,
        "canonical_candidate_count": len(payload.get("canonical_candidates") or []),
        "invalid_candidate_count": invalid_candidate_count,
        "verification_reason_code_counts": verification_reason_counts,
        "terminal_counterfactual_feasible": None
        if payload.get("counterfactual") is None
        else bool((payload.get("counterfactual") or {}).get("feasible")),
        "runtime_result": payload,
    }


def classify_actual_kind(payload: dict[str, Any]) -> str:
    controller_state = payload.get("controller_state")
    reason_codes = set(str(item) for item in payload.get("reason_codes") or [])
    counterfactual = payload.get("counterfactual") or {}
    candidates = list(counterfactual.get("candidates") or [])
    feasible = bool(counterfactual.get("feasible"))
    if controller_state == "TERMINAL_SUCCESS" and not candidates and "NO_RECOURSE_NEEDED" in reason_codes:
        return "runtime_success_no_recourse"
    if controller_state == "TERMINAL_SUCCESS" and feasible and candidates:
        return "runtime_success_counterfactual"
    if controller_state == "TERMINAL_REJECT" and ("NO_FEASIBLE_CF_FOUND" in reason_codes or not feasible):
        return "runtime_reject_infeasible"
    return f"unexpected:{controller_state}"


def classify_case_delta(
    *,
    expected_kind: str,
    actual_kind: str,
    payload: dict[str, Any],
    backend_id: str,
) -> tuple[str | None, str]:
    if actual_kind != expected_kind:
        return (
            UNEXPECTED_REGRESSION,
            f"expected kind {expected_kind} but observed {actual_kind}",
        )
    verification_results = list(payload.get("verification_results") or [])
    if any(item.get("is_valid") is False for item in verification_results):
        return (EXPECTED_VERIFIER_DELTA, "verifier filtered at least one generated candidate before final output")
    selected_candidate_id = build_selected_candidate_id(payload=payload, backend_id=backend_id)
    first_candidate_id = None
    canonical_candidates = list(payload.get("canonical_candidates") or [])
    if canonical_candidates:
        first_candidate_id = canonical_candidates[0].get("candidate_id")
    if selected_candidate_id is not None and first_candidate_id is not None and selected_candidate_id != first_candidate_id:
        return (EXPECTED_RANKING_DELTA, "final selected candidate differed from the first generated canonical candidate")
    return (None, "semantic parity preserved")


def build_selected_candidate_id(*, payload: dict[str, Any], backend_id: str) -> str | None:
    counterfactual = payload.get("counterfactual") or {}
    candidates = list(counterfactual.get("candidates") or [])
    if not candidates:
        return None
    winner = candidates[0]
    method = winner.get("method")
    rank = winner.get("rank")
    if method is None or rank is None:
        return None
    backend_name = str(payload.get("backend_id") or backend_id)
    return f"{backend_name}:{method}:{rank}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Golden Parity Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- scorer_version: `{summary['scorer_version']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- dataset: `{summary['dataset']}`",
        f"- backend_id: `{summary['backend_id']}`",
        f"- corpus_version: `{summary['corpus_version']}`",
        f"- corpus_sha256: `{summary['corpus_sha256']}`",
        "",
        "## Delta Summary",
        "",
        f"- delta_classification_counts: `{summary['delta_classification_counts']}`",
        f"- unexpected_regression_count: `{summary['unexpected_regression_count']}`",
        f"- unexpected_regression_case_ids: `{summary['unexpected_regression_case_ids']}`",
        f"- waiver_applied: `{summary['unexpected_regression_waiver']['applied']}`",
        "",
        "## Aggregate Validation",
        "",
        f"- ok: `{summary['aggregate_validation']['ok']}`",
        f"- difference_count: `{summary['aggregate_validation']['difference_count']}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary["cases"]:
        lines.extend(
            [
                f"- {case['case_id']}: expected=`{case['expected_kind']}` actual=`{case['actual_kind']}` classification=`{case['delta_classification']}`",
                f"  reason: `{case['delta_reason']}`",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
