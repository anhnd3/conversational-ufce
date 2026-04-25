#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, Phase32ValidationCatalog, load_catalog
from llm.src.product.config import ProductConfig
from llm.src.runtime.constraint_spec import effective_blocked_fields
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase3_2_metrics"
RUNNER_SCOPE = "product_facing_g1_g2"
SERVICE_SCRIPT = ROOT / "scripts" / "run_phase3_2_demo.py"
SCOPE_NOTE = (
    "This runner is the authoritative G1/G2 product-facing scorer, not the full parser-layer thesis metrics harness. "
    "Lower-level metrics such as JSON validity, schema compliance, and repair rate remain separate artifact-level concerns."
)
REQUEST_TIMEOUT_S = 30.0


@dataclass
class IsolatedServiceHandle:
    process: subprocess.Popen[str]
    base_url: str
    service_command: str
    sqlite_path: Path
    artifact_root: Path
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Generate the product-facing Phase 3.2 G1/G2 metrics report.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--service-script", type=Path, default=SERVICE_SCRIPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command()
    summary = run_metrics_report(args=args, command=command)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def run_metrics_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    run_id = "phase3_2_metrics_" + local_now_compact()
    layout = prepare_run_layout(out_dir=args.out_dir, run_id=run_id)
    service = launch_isolated_service(layout=layout, args=args)
    try:
        version_payload, dataset_entry, case_results = collect_metrics_from_api(
            base_url=service.base_url,
            api_version=args.api_version,
            catalog=catalog,
        )
        summary = build_metrics_summary(
            run_id=run_id,
            command=command,
            run_root=layout["run_root"],
            service=service,
            catalog=catalog,
            version_payload=version_payload,
            dataset_entry=dataset_entry,
            case_results=case_results,
        )
        write_json(layout["run_root"] / "phase3_2_metrics_report.json", summary)
        (layout["run_root"] / "phase3_2_metrics_report.md").write_text(
            render_markdown(summary),
            encoding="utf-8",
        )
        return summary
    finally:
        stop_isolated_service(service)


def prepare_run_layout(*, out_dir: Path, run_id: str) -> dict[str, Path]:
    run_root = Path(out_dir) / run_id
    artifact_root = run_root / "isolated_product_artifacts"
    sqlite_path = run_root / "isolated_sessions.sqlite3"
    stdout_path = run_root / "service_stdout.log"
    stderr_path = run_root / "service_stderr.log"
    run_root.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=False)
    return {
        "run_root": run_root.resolve(),
        "artifact_root": artifact_root.resolve(),
        "sqlite_path": sqlite_path.resolve(),
        "stdout_path": stdout_path.resolve(),
        "stderr_path": stderr_path.resolve(),
    }


def launch_isolated_service(*, layout: dict[str, Path], args: argparse.Namespace) -> IsolatedServiceHandle:
    host = "127.0.0.1"
    port = pick_free_port(host=host)
    base_url = f"http://{host}:{port}"
    env_overrides = {
        "LM_STUDIO_API_BASE": args.lm_studio_api_base.rstrip("/"),
        "MODEL_ALIAS": args.model_alias,
        "PRODUCT_MODE": args.product_mode,
        "ARTIFACT_ROOT": str(layout["artifact_root"]),
        "SQLITE_PATH": str(layout["sqlite_path"]),
        "API_VERSION": args.api_version,
        "APP_VERSION": args.app_version,
        "HOST": host,
        "PORT": str(port),
    }
    env = os.environ.copy()
    env.update(env_overrides)
    service_command = render_env_prefixed_command(
        env_overrides=env_overrides,
        command=[sys.executable, str(Path(args.service_script).resolve())],
    )
    stdout_handle = layout["stdout_path"].open("w", encoding="utf-8")
    stderr_handle = layout["stderr_path"].open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(Path(args.service_script).resolve())],
        cwd=ROOT,
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    service = IsolatedServiceHandle(
        process=process,
        base_url=base_url,
        service_command=service_command,
        sqlite_path=layout["sqlite_path"],
        artifact_root=layout["artifact_root"],
        stdout_path=layout["stdout_path"],
        stderr_path=layout["stderr_path"],
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )
    try:
        wait_for_service(
            base_url=base_url,
            api_version=args.api_version,
            process=process,
            stdout_path=layout["stdout_path"],
            stderr_path=layout["stderr_path"],
            timeout_s=float(args.startup_timeout_s),
        )
        return service
    except Exception:
        stop_isolated_service(service)
        raise


def stop_isolated_service(service: IsolatedServiceHandle) -> None:
    try:
        if service.process.poll() is None:
            service.process.terminate()
            try:
                service.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                service.process.kill()
                service.process.wait(timeout=5.0)
    finally:
        service.stdout_handle.close()
        service.stderr_handle.close()


def wait_for_service(
    *,
    base_url: str,
    api_version: str,
    process: subprocess.Popen[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "Isolated product service exited before becoming healthy. "
                f"exit_code={return_code}; stdout_tail={tail_lines(stdout_path)}; stderr_tail={tail_lines(stderr_path)}"
            )
        try:
            response = requests.get(f"{base_url}/api/{api_version}/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"http_{response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(
        "Timed out waiting for isolated product service startup. "
        f"last_error={last_error}; stdout_tail={tail_lines(stdout_path)}; stderr_tail={tail_lines(stderr_path)}"
    )


def collect_metrics_from_api(
    *,
    base_url: str,
    api_version: str,
    catalog: Phase32ValidationCatalog,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    api_prefix = f"/api/{api_version}"
    session = requests.Session()
    version_payload = request_json(session, "GET", f"{base_url}{api_prefix}/version")
    dataset_catalog = request_json(session, "GET", f"{base_url}{api_prefix}/catalog/datasets")
    dataset_entry = get_dataset_entry(dataset_catalog, dataset_key="bank")

    case_results: list[dict[str, Any]] = []
    for scenario in catalog.scenarios:
        session_payload = request_json(session, "POST", f"{base_url}{api_prefix}/sessions")
        session_id = str(session_payload["session_id"])
        final_turn = None
        for turn_text in scenario.turns:
            final_turn = request_json(
                session,
                "POST",
                f"{base_url}{api_prefix}/sessions/{session_id}/messages",
                json={"user_input": turn_text},
            )
        if final_turn is None:
            raise RuntimeError(f"Scenario {scenario.scenario_id} produced no turns.")
        session_detail = request_json(session, "GET", f"{base_url}{api_prefix}/sessions/{session_id}")
        case_results.append(
            build_case_result(
                scenario=scenario,
                session_id=session_id,
                final_turn=final_turn,
                session_detail=session_detail,
                dataset_entry=dataset_entry,
            )
        )
    return version_payload, dataset_entry, case_results


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.request(method, url, json=json, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def get_dataset_entry(payload: list[dict[str, Any]], *, dataset_key: str) -> dict[str, Any]:
    for item in payload:
        if item.get("dataset_key") == dataset_key:
            return dict(item)
    raise KeyError(f"Missing dataset catalog entry: {dataset_key}")


def build_case_result(
    *,
    scenario,
    session_id: str,
    final_turn: dict[str, Any],
    session_detail: dict[str, Any],
    dataset_entry: dict[str, Any],
) -> dict[str, Any]:
    public_state = str(final_turn["public_state"])
    case_completion_reason = final_turn.get("case_completion_reason")
    explanation_payload = final_turn.get("explanation_payload") or {}
    debug_summary = final_turn.get("debug_summary") or {}
    runtime_summary = debug_summary.get("runtime_summary") or {}
    summary_type = explanation_payload.get("summary_type")
    reject_class = classify_reject_class(
        public_state=public_state,
        case_completion_reason=case_completion_reason,
        explanation_payload=explanation_payload,
        debug_summary=debug_summary,
    )
    g2_scores = compute_g2_scores(
        public_state=public_state,
        explanation_payload=explanation_payload,
        active_constraint_spec=session_detail.get("active_constraint_spec") or {},
        debug_summary=debug_summary,
        dataset_entry=dataset_entry,
    )
    reproducibility_surface = {
        "final_public_state": public_state,
        "summary_type": summary_type,
        "reject_class": reject_class,
        "invariant_validation_status": debug_summary.get("invariant_validation_status"),
    }
    return {
        "scenario_id": scenario.scenario_id,
        "slug": scenario.slug,
        "expected_final_state": scenario.expected_final_state,
        "expected_final_state_match": public_state == scenario.expected_final_state,
        "session_id": session_id,
        "turn_id": final_turn["turn_id"],
        "final_public_state": public_state,
        "is_case_complete": bool(final_turn["is_case_complete"]),
        "case_completion_reason": case_completion_reason,
        "summary_type": summary_type,
        "reject_class": reject_class,
        "invariant_validation_status": debug_summary.get("invariant_validation_status"),
        "runtime_executed": bool(runtime_summary.get("executed")),
        "runtime_controller_state": runtime_summary.get("controller_state"),
        "reproducibility_surface": reproducibility_surface,
        "g2_applicable": g2_scores["applicable"],
        "g2_not_applicable_reason": g2_scores["not_applicable_reason"],
        "M11_actionability": g2_scores["M11_actionability"],
        "M12_plausibility": g2_scores["M12_plausibility"],
        "M13_feasibility": g2_scores["M13_feasibility"],
    }


def classify_reject_class(
    *,
    public_state: str,
    case_completion_reason: str | None,
    explanation_payload: dict[str, Any],
    debug_summary: dict[str, Any],
) -> str | None:
    if public_state == "RUNTIME_REJECT":
        runtime_summary = debug_summary.get("runtime_summary") or {}
        reason_codes = list(explanation_payload.get("reason_codes") or runtime_summary.get("reason_codes") or [])
        if "REQUEST_CONSTRAINTS_BLOCKED" in reason_codes:
            return "request_constraints_blocked"
        if "NO_FEASIBLE_CF_FOUND" in reason_codes:
            return "no_feasible_cf"
        if "INVALID_COUNTERFACTUAL_BLOCKED" in reason_codes:
            return "invariant_blocked"
        return "system_error"
    if case_completion_reason == "clarification_limit_reached":
        return "clarification_limit_reached"
    if public_state == "CONFLICT" or case_completion_reason == "conflict":
        return "conflict"
    if public_state == "UNSUPPORTED_REQUEST" or case_completion_reason == "unsupported_request":
        return "unsupported_request"
    if public_state == "PARSER_FAILURE" or case_completion_reason == "parser_failure":
        return "parser_failure"
    return None


def compute_g2_scores(
    *,
    public_state: str,
    explanation_payload: dict[str, Any],
    active_constraint_spec: dict[str, Any],
    debug_summary: dict[str, Any],
    dataset_entry: dict[str, Any],
) -> dict[str, Any]:
    summary_type = explanation_payload.get("summary_type")
    if public_state != "RUNTIME_SUCCESS" or summary_type != "counterfactual_found":
        return {
            "applicable": False,
            "not_applicable_reason": f"{public_state}:{summary_type}",
            "M11_actionability": None,
            "M12_plausibility": None,
            "M13_feasibility": None,
        }
    counterfactual_summary = explanation_payload.get("counterfactual_summary") or {}
    feature_order = [str(item) for item in dataset_entry.get("full_feature_list", []) if isinstance(item, str)]
    if not feature_order:
        feature_order = [str(item) for item in (counterfactual_summary.get("profile") or {}).keys()]
    policy_f2change = [str(item) for item in dataset_entry.get("f2change", []) if isinstance(item, str)]
    return {
        "applicable": True,
        "not_applicable_reason": None,
        "M11_actionability": compute_actionability(
            counterfactual_summary=counterfactual_summary,
            active_constraint_spec=active_constraint_spec,
            policy_f2change=policy_f2change,
            feature_order=feature_order,
        ),
        "M12_plausibility": compute_plausibility(debug_summary=debug_summary),
        "M13_feasibility": compute_feasibility(
            public_state=public_state,
            summary_type=summary_type,
            debug_summary=debug_summary,
        ),
    }


def compute_actionability(
    *,
    counterfactual_summary: dict[str, Any],
    active_constraint_spec: dict[str, Any],
    policy_f2change: list[str],
    feature_order: list[str],
) -> int:
    changed_fields = [str(item) for item in counterfactual_summary.get("changed_fields", []) if isinstance(item, str)]
    if any(field not in policy_f2change for field in changed_fields):
        return 0
    blocked_fields = set(
        effective_blocked_fields(
            active_constraint_spec if isinstance(active_constraint_spec, dict) else {},
            feature_order=feature_order,
        )
    )
    if any(field in blocked_fields for field in changed_fields):
        return 0
    max_changed_features = None
    if isinstance(active_constraint_spec, dict):
        max_changed_features = active_constraint_spec.get("max_changed_features")
    if isinstance(max_changed_features, int) and len(changed_fields) > max_changed_features:
        return 0
    numeric_bounds = {}
    if isinstance(active_constraint_spec, dict):
        numeric_bounds = active_constraint_spec.get("numeric_bounds") or {}
    profile = counterfactual_summary.get("profile") or {}
    for field_name, bounds in numeric_bounds.items():
        if field_name not in profile:
            return 0
        value = profile.get(field_name)
        if not isinstance(value, (int, float)):
            return 0
        if "min" in bounds and float(value) < float(bounds["min"]):
            return 0
        if "max" in bounds and float(value) > float(bounds["max"]):
            return 0
    return 1


def compute_plausibility(*, debug_summary: dict[str, Any]) -> int:
    return 1 if debug_summary.get("invariant_validation_status") == "passed" else 0


def compute_feasibility(*, public_state: str, summary_type: str | None, debug_summary: dict[str, Any]) -> int:
    runtime_summary = debug_summary.get("runtime_summary") or {}
    return int(
        public_state == "RUNTIME_SUCCESS"
        and summary_type == "counterfactual_found"
        and runtime_summary.get("executed") is True
        and runtime_summary.get("controller_state") == "TERMINAL_SUCCESS"
    )


def build_metrics_summary(
    *,
    run_id: str,
    command: str,
    run_root: Path,
    service: IsolatedServiceHandle,
    catalog: Phase32ValidationCatalog,
    version_payload: dict[str, Any],
    dataset_entry: dict[str, Any],
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_cases = len(case_results)
    completed_cases = sum(1 for item in case_results if item["is_case_complete"])
    case_completion_reason_counts = counter_dict(
        item["case_completion_reason"] for item in case_results if item["case_completion_reason"] is not None
    )
    public_state_counts = counter_dict(item["final_public_state"] for item in case_results)
    runtime_success_summary_type_counts = counter_dict(
        item["summary_type"]
        for item in case_results
        if item["final_public_state"] == "RUNTIME_SUCCESS" and item["summary_type"] is not None
    )
    reject_class_counts = counter_dict(item["reject_class"] for item in case_results if item["reject_class"] is not None)
    reproducibility_signature_counts = counter_dict(
        json.dumps(item["reproducibility_surface"], sort_keys=True) for item in case_results
    )
    applicable_cases = [item for item in case_results if item["g2_applicable"]]
    m11_numerator = sum(int(item["M11_actionability"]) for item in applicable_cases)
    m12_numerator = sum(int(item["M12_plausibility"]) for item in applicable_cases)
    m13_numerator = sum(int(item["M13_feasibility"]) for item in applicable_cases)
    denominator = len(applicable_cases)
    catalog_sha256 = sha256_file(catalog.source_path)

    provenance = {
        "runner_scope": RUNNER_SCOPE,
        "scope_note": SCOPE_NOTE,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "service_command": service.service_command,
        "service_base_url": service.base_url,
        "service_stdout_path": str(service.stdout_path),
        "service_stderr_path": str(service.stderr_path),
        "catalog_version": catalog.catalog_version,
        "catalog_path": str(catalog.source_path),
        "catalog_sha256": catalog_sha256,
        "catalog_created_timestamp_utc": catalog.created_timestamp_utc,
        "prompt_template_version": catalog.prompt_template_version,
        "dataset_key": dataset_entry["dataset_key"],
        "isolated_run": True,
        "sqlite_path": str(service.sqlite_path),
        "artifact_root": str(service.artifact_root),
        "api_version": version_payload.get("api_version"),
        "app_version": version_payload.get("app_version"),
        "model_alias": version_payload.get("model_alias"),
        "runtime_mode": version_payload.get("runtime_mode"),
        "git_commit": version_payload.get("git_commit"),
    }
    return {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "timestamp_local": provenance["timestamp_local"],
        "timezone": provenance["timezone"],
        "command": command,
        "service_command": service.service_command,
        "catalog_version": catalog.catalog_version,
        "catalog_path": str(catalog.source_path),
        "catalog_sha256": catalog_sha256,
        "isolated_run": True,
        "sqlite_path": str(service.sqlite_path),
        "artifact_root": str(service.artifact_root),
        "output_root": str(run_root),
        "report_json_path": str((run_root / "phase3_2_metrics_report.json").resolve()),
        "report_markdown_path": str((run_root / "phase3_2_metrics_report.md").resolve()),
        "corpus_counts": {
            "total_cases": total_cases,
            "completed_cases": completed_cases,
            "expected_final_state_matches": sum(1 for item in case_results if item["expected_final_state_match"]),
            "runtime_success_cases": public_state_counts.get("RUNTIME_SUCCESS", 0),
            "runtime_reject_cases": public_state_counts.get("RUNTIME_REJECT", 0),
            "g2_applicable_cases": denominator,
            "g2_not_applicable_cases": total_cases - denominator,
        },
        "g1_metrics": {
            "completion": {
                "numerator": completed_cases,
                "denominator": total_cases,
                "mean": safe_mean(completed_cases, total_cases),
            },
            "case_completion_reason_counts": case_completion_reason_counts,
            "public_state_counts": public_state_counts,
            "runtime_success_summary_type_counts": runtime_success_summary_type_counts,
            "reject_class_counts": reject_class_counts,
            "reproducibility_signature_counts": reproducibility_signature_counts,
        },
        "g2_metrics": {
            "applicable_cases": denominator,
            "not_applicable_cases": total_cases - denominator,
            "M11_actionability": build_binary_metric("Actionability", m11_numerator, denominator),
            "M12_plausibility": build_binary_metric("Plausibility", m12_numerator, denominator),
            "M13_feasibility": build_binary_metric("Feasibility", m13_numerator, denominator),
        },
        "per_case_results": case_results,
        "provenance": provenance,
    }


def build_binary_metric(name: str, numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "name": name,
        "numerator": numerator,
        "denominator": denominator,
        "mean": safe_mean(numerator, denominator),
    }


def counter_dict(items) -> dict[str, int]:
    counts = Counter(str(item) for item in items)
    return dict(sorted(counts.items()))


def safe_mean(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3.2 Metrics Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- command: `{summary['command']}`",
        f"- service_command: `{summary['service_command']}`",
        f"- catalog_version: `{summary['catalog_version']}`",
        f"- catalog_path: `{summary['catalog_path']}`",
        f"- catalog_sha256: `{summary['catalog_sha256']}`",
        f"- isolated_run: `{summary['isolated_run']}`",
        f"- sqlite_path: `{summary['sqlite_path']}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        "",
        "## Scope",
        "",
        SCOPE_NOTE,
        "",
        "## Corpus",
        "",
        f"- total_cases: `{summary['corpus_counts']['total_cases']}`",
        f"- completed_cases: `{summary['corpus_counts']['completed_cases']}`",
        f"- expected_final_state_matches: `{summary['corpus_counts']['expected_final_state_matches']}`",
        f"- g2_applicable_cases: `{summary['corpus_counts']['g2_applicable_cases']}`",
        f"- g2_not_applicable_cases: `{summary['corpus_counts']['g2_not_applicable_cases']}`",
        "",
        "## G1 Metrics",
        "",
        f"- completion_rate: `{summary['g1_metrics']['completion']['mean']}`",
        f"- case_completion_reason_counts: `{summary['g1_metrics']['case_completion_reason_counts']}`",
        f"- public_state_counts: `{summary['g1_metrics']['public_state_counts']}`",
        f"- runtime_success_summary_type_counts: `{summary['g1_metrics']['runtime_success_summary_type_counts']}`",
        f"- reject_class_counts: `{summary['g1_metrics']['reject_class_counts']}`",
        "",
        "## G2 Metrics",
        "",
        "- `M11 = Actionability`: final exposed recommendation respects request-level change permissions and active constraint rules.",
        "- `M12 = Plausibility`: final exposed recommendation passes the exposed invariant plausibility gate.",
        "- `M13 = Feasibility`: final exposed recommendation is exposed by the system as a valid runtime-backed counterfactual after all gates.",
        "",
        "| Metric | Numerator | Denominator | Mean |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("M11_actionability", "M12_plausibility", "M13_feasibility"):
        metric = summary["g2_metrics"][key]
        lines.append(
            f"| `{key}` | `{metric['numerator']}` | `{metric['denominator']}` | `{metric['mean']}` |"
        )
    lines.extend(
        [
            "",
            "## Case Breakdown",
            "",
            "| Scenario | Expected | Actual | Summary Type | Reject Class | M11 | M12 | M13 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in summary["per_case_results"]:
        lines.append(
            "| `{scenario_id}` | `{expected}` | `{actual}` | `{summary_type}` | `{reject_class}` | `{m11}` | `{m12}` | `{m13}` |".format(
                scenario_id=item["scenario_id"],
                expected=item["expected_final_state"],
                actual=item["final_public_state"],
                summary_type=item["summary_type"],
                reject_class=item["reject_class"],
                m11=render_metric_value(item["M11_actionability"]),
                m12=render_metric_value(item["M12_plausibility"]),
                m13=render_metric_value(item["M13_feasibility"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_metric_value(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def build_runner_command() -> str:
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    return " ".join(shlex.quote(item) for item in command)


def render_env_prefixed_command(*, env_overrides: dict[str, str], command: list[str]) -> str:
    rendered_env = [f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())]
    rendered_command = [shlex.quote(item) for item in command]
    return " ".join(rendered_env + rendered_command)


def pick_free_port(*, host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def tail_lines(path: Path, *, limit: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-limit:]


if __name__ == "__main__":
    raise SystemExit(main())
