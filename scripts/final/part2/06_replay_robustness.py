#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import (
    add_summary_output_args,
    build_in_process_service,
    build_runner_command,
    build_session_detail_payload,
    call_with_legacy_stdout_redirect,
    counter_dict,
    get_dataset_entry,
    prepare_run_layout,
    progress_iter,
    recompute_and_validate_aggregates,
    safe_mean,
    summarize_latency_ms,
    write_optional_summary_outputs,
)
from llm.src.part2_eval.corpora import TIER_D_CORPUS_PATH, load_tier_d_bank_replay_corpus
from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.product.config import ProductConfig
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_replay_robustness"
RUNNER_SCOPE = "part2_tier_d_replay_robustness"
SCORER_VERSION = "part2_replay_robustness_report_v1"
REPRODUCIBILITY_SAMPLE_SIZE = 25
REPRODUCIBILITY_REPEATS = 3
PERSISTENCE_SAMPLE_SIZE = 10


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Generate the Part II Tier D replay robustness report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--benchmark", type=Path, default=Path(ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml"))
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--tier-d-corpus", type=Path, default=TIER_D_CORPUS_PATH)
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_replay_robustness_report(args=args, command=command)
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
    return 0


def run_replay_robustness_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    baseline_catalog = load_catalog(args.baseline_catalog)
    model_registry = ModelRegistry()
    policy_registry = PolicyRegistry(model_registry)
    model_registry.get_bundle("bank")
    policy_registry.get_policy("bank")
    corpus = load_tier_d_bank_replay_corpus(args.tier_d_corpus)

    run_id = "part2_replay_robustness_" + local_now_compact()
    layout = prepare_run_layout(out_dir=args.out_dir, run_id=run_id)
    handle = build_in_process_service(
        layout=layout,
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
        product_mode=args.product_mode,
        api_version=args.api_version,
        app_version=args.app_version,
        benchmark_path=args.benchmark,
    )
    version_payload = call_with_legacy_stdout_redirect(handle.service.version)
    dataset_entry = get_dataset_entry(call_with_legacy_stdout_redirect(handle.service.list_dataset_catalog), dataset_key="bank")
    replay_results = collect_replay_results(
        handle=handle,
        replay_requests=corpus["replay_requests"],
        progress_enabled=not args.no_progress,
    )
    reproducibility_checks = collect_reproducibility_checks(
        handle=handle,
        replay_requests=corpus["replay_requests"][:REPRODUCIBILITY_SAMPLE_SIZE],
        progress_enabled=not args.no_progress,
    )
    persistence_checks = collect_persistence_checks(
        layout=layout,
        args=args,
        replay_requests=corpus["replay_requests"][:PERSISTENCE_SAMPLE_SIZE],
        progress_enabled=not args.no_progress,
    )
    summary = build_replay_summary(
        run_id=run_id,
        command=command,
        run_root=layout["run_root"],
        handle=handle,
        baseline_catalog=baseline_catalog,
        version_payload=version_payload,
        dataset_entry=dataset_entry,
        corpus=corpus,
        tier_d_corpus_path=args.tier_d_corpus,
        replay_results=replay_results,
        reproducibility_checks=reproducibility_checks,
        persistence_checks=persistence_checks,
        benchmark_path=args.benchmark,
    )
    write_json(layout["run_root"] / "replay_robustness_report.json", summary)
    (layout["run_root"] / "replay_robustness_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def collect_replay_results(*, handle, replay_requests: list[dict[str, Any]], progress_enabled: bool) -> list[dict[str, Any]]:
    return [
        execute_replay_request(handle=handle, replay_request=item)
        for item in progress_iter(
            replay_requests,
            enabled=progress_enabled,
            desc="Replay requests",
            unit="request",
        )
    ]


def execute_replay_request(*, handle, replay_request: dict[str, Any]) -> dict[str, Any]:
    created = handle.service.create_session()
    session_id = str(created.session_id)
    try:
        stored_turn = call_with_legacy_stdout_redirect(handle.service.submit_message, session_id, replay_request["user_input"])
        payload = handle.service.build_turn_response(stored_turn)
        session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
        debug_summary = payload.get("debug_summary") or {}
        timing_metrics = debug_summary.get("timing_metrics") or {}
        explanation_payload = payload.get("explanation_payload") or {}
        return {
            "replay_id": replay_request["replay_id"],
            "source_case_id": replay_request["source_case_id"],
            "source_group": replay_request["source_group"],
            "source_session_shape": replay_request["source_session_shape"],
            "source_turn_index": replay_request["source_turn_index"],
            "cycle_index": replay_request["cycle_index"],
            "session_id": session_id,
            "final_public_state": payload["public_state"],
            "case_completion_reason": payload.get("case_completion_reason"),
            "summary_type": explanation_payload.get("summary_type"),
            "invariant_validation_status": debug_summary.get("invariant_validation_status"),
            "restart_required": bool(payload.get("restart_required")),
            "is_case_complete": bool(payload.get("is_case_complete")),
            "active_constraint_spec": session_detail.get("active_constraint_spec") or {},
            "final_latency_ms": timing_metrics.get("end_to_end_latency_ms"),
            "runtime_latency_ms": timing_metrics.get("runtime_latency_ms"),
            "error_class": None,
            "error_detail": None,
        }
    except Exception as exc:  # pragma: no cover - exercised in live runs
        return {
            "replay_id": replay_request["replay_id"],
            "source_case_id": replay_request["source_case_id"],
            "source_group": replay_request["source_group"],
            "source_session_shape": replay_request["source_session_shape"],
            "source_turn_index": replay_request["source_turn_index"],
            "cycle_index": replay_request["cycle_index"],
            "session_id": session_id,
            "final_public_state": "EXECUTION_ERROR",
            "case_completion_reason": None,
            "summary_type": None,
            "invariant_validation_status": None,
            "restart_required": False,
            "is_case_complete": False,
            "active_constraint_spec": {},
            "final_latency_ms": None,
            "runtime_latency_ms": None,
            "error_class": type(exc).__name__,
            "error_detail": str(exc),
        }


def collect_reproducibility_checks(
    *,
    handle,
    replay_requests: list[dict[str, Any]],
    progress_enabled: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for replay_request in progress_iter(
        replay_requests,
        enabled=progress_enabled,
        desc="Replay reproducibility",
        unit="request",
    ):
        attempts = [execute_replay_request(handle=handle, replay_request=replay_request) for _ in range(REPRODUCIBILITY_REPEATS)]
        surfaces = [
            {
                "final_public_state": attempt["final_public_state"],
                "summary_type": attempt["summary_type"],
                "invariant_validation_status": attempt["invariant_validation_status"],
                "case_completion_reason": attempt["case_completion_reason"],
            }
            for attempt in attempts
        ]
        stable = all(surface == surfaces[0] for surface in surfaces[1:]) if surfaces else False
        checks.append(
            {
                "replay_id": replay_request["replay_id"],
                "attempt_count": len(attempts),
                "stable": stable,
                "surfaces": surfaces,
            }
        )
    return checks


def collect_persistence_checks(
    *,
    layout: dict[str, Path],
    args: argparse.Namespace,
    replay_requests: list[dict[str, Any]],
    progress_enabled: bool,
) -> list[dict[str, Any]]:
    handle_one = build_in_process_service(
        layout=layout,
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
        product_mode=args.product_mode,
        api_version=args.api_version,
        app_version=args.app_version,
        benchmark_path=args.benchmark,
    )
    created_sessions: list[dict[str, Any]] = []
    for replay_request in progress_iter(
        replay_requests,
        enabled=progress_enabled,
        desc="Replay persistence",
        unit="request",
    ):
        created = handle_one.service.create_session()
        session_id = str(created.session_id)
        stored_turn = call_with_legacy_stdout_redirect(
            handle_one.service.submit_message,
            session_id,
            replay_request["user_input"],
        )
        created_sessions.append(
            {
                "replay_id": replay_request["replay_id"],
                "session_id": session_id,
                "latest_turn_id": stored_turn.turn_id,
                "final_public_state": stored_turn.public_state,
            }
        )
    handle_two = build_in_process_service(
        layout=layout,
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
        product_mode=args.product_mode,
        api_version=args.api_version,
        app_version=args.app_version,
        benchmark_path=args.benchmark,
    )
    checks: list[dict[str, Any]] = []
    for item in created_sessions:
        restored_session = handle_two.repository.get_session(item["session_id"])
        restored_turns = handle_two.repository.list_turns(item["session_id"], order="asc")
        restored = (
            restored_session.latest_turn_id == item["latest_turn_id"]
            and restored_session.current_public_state == item["final_public_state"]
            and len(restored_turns) >= 1
        )
        checks.append(
            {
                "replay_id": item["replay_id"],
                "session_id": item["session_id"],
                "restored": restored,
                "restored_turn_count": len(restored_turns),
                "restored_latest_turn_id": restored_session.latest_turn_id,
                "restored_public_state": restored_session.current_public_state,
            }
        )
    return checks


def compute_replay_aggregate_blocks(
    replay_results: list[dict[str, Any]],
    reproducibility_checks: list[dict[str, Any]],
    persistence_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    total_requests = len(replay_results)
    error_count = sum(1 for row in replay_results if row["error_class"] is not None)
    stable_count = sum(1 for row in reproducibility_checks if row["stable"])
    restored_count = sum(1 for row in persistence_checks if row["restored"])
    return {
        "robustness_metrics": {
            "request_count": total_requests,
            "public_state_counts": counter_dict(row["final_public_state"] for row in replay_results),
            "case_completion_reason_counts": counter_dict(
                row["case_completion_reason"] for row in replay_results if row["case_completion_reason"] is not None
            ),
            "error_rate": {
                "numerator": error_count,
                "denominator": total_requests,
                "mean": safe_mean(error_count, total_requests),
            },
            "restart_required_rate": {
                "numerator": sum(1 for row in replay_results if row["restart_required"]),
                "denominator": total_requests,
                "mean": safe_mean(sum(1 for row in replay_results if row["restart_required"]), total_requests),
            },
            "request_latency_ms": summarize_latency_ms([row["final_latency_ms"] for row in replay_results]),
            "runtime_latency_ms": summarize_latency_ms([row["runtime_latency_ms"] for row in replay_results]),
            "reproducibility_stability": {
                "numerator": stable_count,
                "denominator": len(reproducibility_checks),
                "mean": safe_mean(stable_count, len(reproducibility_checks)),
            },
            "persistence_restoration": {
                "numerator": restored_count,
                "denominator": len(persistence_checks),
                "mean": safe_mean(restored_count, len(persistence_checks)),
            },
            "excluded_from_semantic_tables": True,
        }
    }


def build_replay_summary(
    *,
    run_id: str,
    command: str,
    run_root: Path,
    handle,
    baseline_catalog,
    version_payload: dict[str, Any],
    dataset_entry: dict[str, Any],
    corpus: dict[str, Any],
    tier_d_corpus_path: Path,
    replay_results: list[dict[str, Any]],
    reproducibility_checks: list[dict[str, Any]],
    persistence_checks: list[dict[str, Any]],
    benchmark_path: Path,
) -> dict[str, Any]:
    baseline_catalog_sha256 = sha256_file(baseline_catalog.source_path)
    aggregate_blocks = compute_replay_aggregate_blocks(replay_results, reproducibility_checks, persistence_checks)
    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "execution_mode": handle.execution_mode,
        "catalog_version": baseline_catalog.catalog_version,
        "catalog_path": str(baseline_catalog.source_path),
        "catalog_sha256": baseline_catalog_sha256,
        "git_commit": version_payload.get("git_commit"),
        "corpus_path": str(Path(tier_d_corpus_path).resolve()),
        "corpus_version": corpus["corpus_version"],
        "corpus_sha256": corpus["corpus_sha256"],
        "report_json_path": str((run_root / "replay_robustness_report.json").resolve()),
        "report_markdown_path": str((run_root / "replay_robustness_report.md").resolve()),
        "loaded_corpora": {
            "tier_d": {
                "corpus_path": str(Path(tier_d_corpus_path).resolve()),
                "corpus_version": corpus["corpus_version"],
                "corpus_sha256": corpus["corpus_sha256"],
            }
        },
        "provenance": {
            "runner_scope": RUNNER_SCOPE,
            "scorer_version": SCORER_VERSION,
            "timestamp_local": local_now_iso(),
            "timezone": "UTC+07:00",
            "command": command,
            "execution_mode": handle.execution_mode,
            "corpus_path": str(Path(tier_d_corpus_path).resolve()),
            "catalog_version": baseline_catalog.catalog_version,
            "catalog_sha256": baseline_catalog_sha256,
            "dataset_key": dataset_entry["dataset_key"],
            "api_version": version_payload.get("api_version"),
            "app_version": version_payload.get("app_version"),
            "model_alias": version_payload.get("model_alias"),
            "runtime_mode": version_payload.get("runtime_mode"),
            "git_commit": version_payload.get("git_commit"),
            "lm_studio_api_base": handle.config.lm_studio_api_base,
            "benchmark_path": str(Path(benchmark_path).resolve()),
        },
        "frozen_inputs": {
            "replay_request_count": corpus["replay_request_count"],
            "source_request_count": corpus["source_request_count"],
            "source_case_count": corpus["source_case_count"],
        },
        "robustness_metrics": aggregate_blocks["robustness_metrics"],
        "replay_results": replay_results,
        "reproducibility_checks": reproducibility_checks,
        "persistence_checks": persistence_checks,
    }
    summary["aggregate_validation"] = recompute_and_validate_aggregates(
        expected_blocks={"robustness_metrics": summary["robustness_metrics"]},
        recomputed_blocks=aggregate_blocks,
    )
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Replay Robustness Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- scorer_version: `{summary['scorer_version']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- execution_mode: `{summary['execution_mode']}`",
        f"- corpus_version: `{summary['corpus_version']}`",
        f"- corpus_sha256: `{summary['corpus_sha256']}`",
        "",
        "## Robustness Metrics",
        "",
        f"- request_count: `{summary['robustness_metrics']['request_count']}`",
        f"- public_state_counts: `{summary['robustness_metrics']['public_state_counts']}`",
        f"- case_completion_reason_counts: `{summary['robustness_metrics']['case_completion_reason_counts']}`",
        f"- error_rate: `{summary['robustness_metrics']['error_rate']['mean']}`",
        f"- request_latency_ms: `{summary['robustness_metrics']['request_latency_ms']}`",
        f"- runtime_latency_ms: `{summary['robustness_metrics']['runtime_latency_ms']}`",
        f"- reproducibility_stability: `{summary['robustness_metrics']['reproducibility_stability']['mean']}`",
        f"- persistence_restoration: `{summary['robustness_metrics']['persistence_restoration']['mean']}`",
        "",
        "## Scope Boundary",
        "",
        "- Tier D is reported for latency, robustness, reproducibility, and persistence only.",
        "- Tier D is excluded from semantic thesis tables and semantic performance aggregates.",
        "",
        "## Aggregate Validation",
        "",
        f"- ok: `{summary['aggregate_validation']['ok']}`",
        f"- difference_count: `{summary['aggregate_validation']['difference_count']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
