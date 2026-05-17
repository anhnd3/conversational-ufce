#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH
from llm.src.product.config import ProductConfig, try_get_git_commit
from llm.src.runtime.constraint_spec import (
    apply_constraint_spec_to_candidates,
    validate_and_normalize_constraint_spec,
)
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.reason_codes import REQUEST_CONSTRAINTS_BLOCKED
from llm.src.runtime.reproducibility import sort_counterfactual_candidates
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase3_2_acceptance"
CLAIM_MAP_PATH = ROOT / "docs" / "thesis" / "part2" / "claim_to_evidence_map.md"
THESIS_ALIGNMENT_PATH = ROOT / "docs" / "thesis" / "part2" / "final_thesis_alignment_notes.md"


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Run the combined Phase 3.2 acceptance report.")
    parser.add_argument("--base-url", default=f"http://{product_config.host}:{product_config.port}")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sqlite-path", type=Path, default=product_config.sqlite_path)
    parser.add_argument("--smoke-out-dir", type=Path, default=ROOT / "outputs" / "phase3_2_product_smoke")
    parser.add_argument("--repro-out-dir", type=Path, default=ROOT / "outputs" / "phase3_2_reproducibility")
    parser.add_argument("--metrics-out-dir", type=Path, default=ROOT / "outputs" / "phase3_2_metrics")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--manual-session-id", default="")
    parser.add_argument("--manual-evidence-note", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = "phase3_2_acceptance_" + local_now_compact()
    run_root = args.out_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    smoke_suite = run_json_script(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_phase3_2_product_smoke.py"),
            "--base-url",
            args.base_url,
            "--catalog",
            str(args.catalog),
            "--out-dir",
            str(args.smoke_out_dir),
        ]
    )
    repro_suite = run_json_script(
        [
            sys.executable,
            str(ROOT / "scripts" / "probe_phase3_2_reproducibility.py"),
            "--catalog",
            str(args.catalog),
            "--out-dir",
            str(args.repro_out_dir),
            "--repeats",
            str(args.repeats),
        ]
    )
    metrics_suite = run_json_script(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_phase3_2_metrics_report.py"),
            "--catalog",
            str(args.catalog),
            "--out-dir",
            str(args.metrics_out_dir),
        ]
    )
    product_checks = run_product_acceptance_checks(
        base_url=args.base_url,
        sqlite_path=args.sqlite_path,
    )
    constraint_checks = run_constraint_acceptance_checks()
    docs_checks = {
        "claim_to_evidence_map_exists": CLAIM_MAP_PATH.exists(),
        "thesis_alignment_notes_exists": THESIS_ALIGNMENT_PATH.exists(),
    }
    manual_evidence = build_manual_evidence_summary(
        sqlite_path=args.sqlite_path,
        manual_session_id=args.manual_session_id.strip(),
        manual_evidence_note=args.manual_evidence_note.strip(),
    )

    summary = build_acceptance_summary(
        run_id=run_id,
        base_url=args.base_url,
        sqlite_path=args.sqlite_path,
        catalog_path=args.catalog,
        smoke_suite=smoke_suite,
        repro_suite=repro_suite,
        metrics_suite=metrics_suite,
        product_checks=product_checks,
        constraint_checks=constraint_checks,
        docs_checks=docs_checks,
        manual_evidence=manual_evidence,
    )
    write_json(run_root / "phase3_2_acceptance_report.json", summary)
    (run_root / "phase3_2_acceptance_report.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if summary["acceptance_passed"] else 1


def run_json_script(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = parse_json_payload(completed.stdout)
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "summary": payload,
        "stdout_tail": tail_lines(completed.stdout),
        "stderr_tail": tail_lines(completed.stderr),
    }


def parse_json_payload(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"(?m)^\{", text):
        candidate = text[match.start():].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def tail_lines(text: str, *, limit: int = 8) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def run_product_acceptance_checks(*, base_url: str, sqlite_path: Path) -> dict[str, Any]:
    session = requests.Session()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    home = request_json(session, "GET", f"{base_url}/", expect_json=False)
    health = request_json(session, "GET", f"{base_url}/api/v1/health")
    version = request_json(session, "GET", f"{base_url}/api/v1/version")
    catalog = request_json(session, "GET", f"{base_url}/api/v1/catalog/datasets")
    sessions_before = request_json(session, "GET", f"{base_url}/api/v1/sessions")

    checks["home_page_ok"] = home["ok"]
    checks["health_endpoint_ok"] = health["ok"] and health["payload"]["status"] in {"healthy", "unhealthy"}
    checks["version_endpoint_ok"] = version["ok"] and version["payload"]["parser_schema_version"] == "parser_schema_v2"
    checks["catalog_endpoint_ok"] = catalog["ok"] and any(
        item["dataset_key"] == "bank" and item["availability_status"] == "active"
        for item in catalog["payload"]
    )
    checks["sessions_endpoint_ok"] = sessions_before["ok"] and isinstance(sessions_before["payload"], list)

    terminal_case = run_terminal_case_probe(session=session, base_url=base_url)
    clarification_case = run_clarification_limit_probe(session=session, base_url=base_url)
    sqlite_summary = read_sqlite_summary(sqlite_path)

    checks.update(terminal_case["checks"])
    checks.update(clarification_case["checks"])
    checks["sqlite_schema_ok"] = sqlite_summary["ok"]

    details["health"] = health["payload"] if health["ok"] else health["error"]
    details["version"] = version["payload"] if version["ok"] else version["error"]
    details["terminal_case"] = terminal_case["details"]
    details["clarification_case"] = clarification_case["details"]
    details["sqlite_summary"] = sqlite_summary

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
    }


def run_terminal_case_probe(*, session: requests.Session, base_url: str) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    created = request_json(session, "POST", f"{base_url}/api/v1/sessions")
    if not created["ok"]:
        return {"checks": {"terminal_case_session_created": False}, "details": {"create_error": created["error"]}}
    session_id = created["payload"]["session_id"]
    message = request_json(
        session,
        "POST",
        f"{base_url}/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
                "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
            )
        },
    )
    if not message["ok"]:
        return {
            "checks": {"terminal_case_session_created": True, "terminal_case_message_ok": False},
            "details": {"session_id": session_id, "message_error": message["error"]},
        }

    turn_payload = message["payload"]
    session_detail = request_json(session, "GET", f"{base_url}/api/v1/sessions/{session_id}")
    messages = request_json(session, "GET", f"{base_url}/api/v1/sessions/{session_id}/messages")
    artifacts = request_json(session, "GET", f"{base_url}/api/v1/sessions/{session_id}/artifacts")

    preview_ok = False
    preview_blocked = False
    blocked_followup = False
    preview_payload = None
    blocked_payload = None
    traversal_payload = None
    if artifacts["ok"] and artifacts["payload"]:
        first_bundle = artifacts["payload"][0]
        preview_url = first_bundle["preview_urls"].get("artifact_manifest.json")
        if preview_url:
            preview = request_json(session, "GET", f"{base_url}{preview_url}")
            preview_ok = preview["ok"] and preview["payload"]["filename"] == "artifact_manifest.json"
            preview_payload = preview["payload"] if preview["ok"] else preview["error"]
        traversal = request_json(
            session,
            "GET",
            f"{base_url}/api/v1/sessions/{session_id}/artifacts/{turn_payload['turn_id']}/../secret.txt/preview",
        )
        preview_blocked = not traversal["ok"] and traversal["status_code"] == 404
        traversal_payload = traversal["error"]

    blocked = request_json(
        session,
        "POST",
        f"{base_url}/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 120."},
    )
    blocked_followup = (
        not blocked["ok"]
        and blocked["status_code"] == 409
        and blocked["error"].get("error_code") == "case_complete"
        and blocked["error"].get("case_completion_reason") == "runtime_success"
        and blocked["error"].get("restart_required") is True
    )
    blocked_payload = blocked["error"]

    checks["terminal_case_session_created"] = True
    checks["terminal_case_message_ok"] = True
    checks["terminal_case_marked_complete"] = (
        turn_payload["is_case_complete"] is True
        and turn_payload["case_completion_reason"] == "runtime_success"
        and turn_payload["restart_required"] is True
    )
    checks["terminal_case_session_summary_complete"] = (
        session_detail["ok"]
        and session_detail["payload"]["is_case_complete"] is True
        and session_detail["payload"]["case_completion_reason"] == "runtime_success"
    )
    checks["terminal_case_messages_available"] = messages["ok"] and len(messages["payload"]) >= 1
    checks["terminal_case_artifacts_available"] = artifacts["ok"] and len(artifacts["payload"]) >= 1
    checks["artifact_preview_ok"] = preview_ok
    checks["artifact_preview_traversal_blocked"] = preview_blocked
    checks["terminal_case_followup_blocked_409"] = blocked_followup

    details["session_id"] = session_id
    details["turn_id"] = turn_payload["turn_id"]
    details["turn_payload"] = turn_payload
    details["preview_payload"] = preview_payload
    details["traversal_response"] = traversal_payload
    details["blocked_followup_response"] = blocked_payload
    return {"checks": checks, "details": details}


def run_clarification_limit_probe(*, session: requests.Session, base_url: str) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    created = request_json(session, "POST", f"{base_url}/api/v1/sessions")
    if not created["ok"]:
        return {"checks": {"clarification_session_created": False}, "details": {"create_error": created["error"]}}
    session_id = created["payload"]["session_id"]
    turn_texts = [
        "Income 40 and CCAvg 1.5.",
        "Family 3 and Education 2.",
        "Mortgage 80.",
    ]
    turn_payloads: list[dict[str, Any]] = []
    for turn_text in turn_texts:
        response = request_json(
            session,
            "POST",
            f"{base_url}/api/v1/sessions/{session_id}/messages",
            json={"user_input": turn_text},
        )
        if not response["ok"]:
            return {
                "checks": {"clarification_session_created": True, "clarification_turn_sequence_ok": False},
                "details": {"session_id": session_id, "turn_error": response["error"]},
            }
        turn_payloads.append(response["payload"])

    exhausted = turn_payloads[-1]
    blocked = request_json(
        session,
        "POST",
        f"{base_url}/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Online yes."},
    )

    checks["clarification_session_created"] = True
    checks["clarification_turn_sequence_ok"] = len(turn_payloads) == 3
    checks["clarification_limit_reached_payload"] = (
        exhausted["public_state"] == "NEEDS_CLARIFICATION"
        and exhausted["is_case_complete"] is True
        and exhausted["case_completion_reason"] == "clarification_limit_reached"
        and exhausted["restart_required"] is True
        and exhausted["clarification_payload"]["clarification_type"] == "clarification_limit_reached"
        and exhausted["clarification_payload"]["remaining_rounds"] == 0
        and exhausted["clarification_payload"]["restart_required"] is True
    )
    checks["clarification_followup_blocked_409"] = (
        not blocked["ok"]
        and blocked["status_code"] == 409
        and blocked["error"].get("error_code") == "case_complete"
        and blocked["error"].get("case_completion_reason") == "clarification_limit_reached"
        and blocked["error"].get("restart_required") is True
    )
    details["session_id"] = session_id
    details["turn_payloads"] = turn_payloads
    details["blocked_response"] = blocked["error"]
    return {"checks": checks, "details": details}


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    expect_json: bool = True,
    **kwargs,
) -> dict[str, Any]:
    try:
        response = session.request(method, url, timeout=180.0, **kwargs)
    except Exception as exc:
        return {"ok": False, "status_code": None, "error": {"detail": str(exc)}}
    if expect_json:
        try:
            payload = response.json()
        except Exception:
            payload = None
    else:
        payload = response.text
    if response.status_code >= 400:
        return {"ok": False, "status_code": response.status_code, "error": payload or {"detail": response.text}}
    return {"ok": True, "status_code": response.status_code, "payload": payload}


def read_sqlite_summary(sqlite_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {"ok": False, "detail": f"sqlite file not found: {sqlite_path}"}
    try:
        with sqlite3.connect(str(sqlite_path)) as connection:
            connection.row_factory = sqlite3.Row
            metadata = connection.execute("SELECT * FROM app_metadata WHERE singleton = 1").fetchone()
            session_count = int(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
            turn_count = int(connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0])
            completion_rows = connection.execute(
                """
                SELECT COALESCE(case_completion_reason, 'null') AS case_completion_reason, COUNT(*) AS count
                FROM sessions
                GROUP BY COALESCE(case_completion_reason, 'null')
                ORDER BY case_completion_reason
                """
            ).fetchall()
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}

    return {
        "ok": metadata is not None and int(metadata["db_schema_version"]) >= 3,
        "db_schema_version": None if metadata is None else int(metadata["db_schema_version"]),
        "app_version": None if metadata is None else metadata["app_version"],
        "session_count": session_count,
        "turn_count": turn_count,
        "case_completion_counts": {
            str(row["case_completion_reason"]): int(row["count"]) for row in completion_rows
        },
    }


def run_constraint_acceptance_checks() -> dict[str, Any]:
    orchestrator = RuntimeOrchestrator()
    feature_order = [
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    legacy_profile = {
        "Income": 100,
        "Family": 1,
        "CCAvg": 2.7,
        "Education": 2,
        "Mortgage": 0,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 0,
        "CreditCard": 0,
    }
    extended_profile = {
        "Income": 72,
        "Family": 1,
        "CCAvg": 4.8,
        "Education": 2,
        "Mortgage": 200,
        "SecuritiesAccount": 1,
        "CDAccount": 1,
        "Online": 0,
        "CreditCard": 0,
    }

    normalized_spec, normalize_errors = validate_and_normalize_constraint_spec(
        {
            "immutable": ["CreditCard", "Income", "CreditCard"],
            "disallowed_changes": ["Mortgage", "Income", "Mortgage"],
        },
        feature_order=feature_order,
    )
    legacy_result = orchestrator.handle({"dataset": "bank", "profile": legacy_profile}, include_debug_trace=True)
    blocked_result = orchestrator.handle(
        {
            "dataset": "bank",
            "profile": legacy_profile,
            "constraint_spec": {"disallowed_changes": ["CDAccount"]},
        },
        include_debug_trace=True,
    )
    max_changed_result = orchestrator.handle(
        {
            "dataset": "bank",
            "profile": extended_profile,
            "constraint_spec": {"max_changed_features": 1},
        },
        include_debug_trace=True,
    )
    numeric_bounds_result = orchestrator.handle(
        {
            "dataset": "bank",
            "profile": extended_profile,
            "constraint_spec": {"numeric_bounds": {"Income": {"min": 90}}},
        },
        include_debug_trace=True,
    )
    prefer_default, prefer_fewer = build_prefer_fewer_probe()

    checks = {
        "legacy_no_constraint_unchanged": (
            legacy_result.controller_state == "TERMINAL_SUCCESS"
            and legacy_result.reason_codes == []
            and legacy_result.counterfactual is not None
            and legacy_result.counterfactual.feasible is True
            and legacy_result.counterfactual.candidates[0].changed_features == ["CDAccount"]
        ),
        "constraint_ordering_normalized": (
            normalize_errors == []
            and normalized_spec == {
                "immutable": ["Income", "CreditCard"],
                "disallowed_changes": ["Income", "Mortgage"],
            }
        ),
        "request_constraints_blocked_distinct": (
            blocked_result.controller_state == "TERMINAL_REJECT"
            and blocked_result.reason_codes == [REQUEST_CONSTRAINTS_BLOCKED]
            and blocked_result.debug_trace is not None
            and blocked_result.debug_trace.constraint_filter is not None
        ),
        "max_changed_features_enforced": (
            max_changed_result.controller_state == "TERMINAL_SUCCESS"
            and max_changed_result.counterfactual is not None
            and all(len(candidate.changed_features) <= 1 for candidate in max_changed_result.counterfactual.candidates)
        ),
        "numeric_bounds_apply_to_final_candidates": (
            numeric_bounds_result.controller_state == "TERMINAL_SUCCESS"
            and numeric_bounds_result.counterfactual is not None
            and all(candidate.profile["Income"] >= 90 for candidate in numeric_bounds_result.counterfactual.candidates)
        ),
        "prefer_fewer_changes_reorders_candidates": (
            [candidate.method for candidate in prefer_default] == ["sfexp", "tfexp"]
            and [candidate.method for candidate in prefer_fewer] == ["tfexp", "sfexp"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "legacy_result": legacy_result.to_dict(include_debug_trace=True),
            "blocked_result": blocked_result.to_dict(include_debug_trace=True),
            "max_changed_result": max_changed_result.to_dict(include_debug_trace=True),
            "numeric_bounds_result": numeric_bounds_result.to_dict(include_debug_trace=True),
            "normalized_constraint_spec": normalized_spec,
            "prefer_default": [candidate.to_dict() for candidate in prefer_default],
            "prefer_fewer": [candidate.to_dict() for candidate in prefer_fewer],
        },
    }


def build_prefer_fewer_probe() -> tuple[list[CounterfactualCandidate], list[CounterfactualCandidate]]:
    candidates = [
        CounterfactualCandidate(
            method="sfexp",
            rank=1,
            profile={"Income": 88.0, "CCAvg": 5.1},
            changed_features=["Income", "CCAvg"],
        ),
        CounterfactualCandidate(
            method="tfexp",
            rank=1,
            profile={"Income": 90.0, "CCAvg": 4.8},
            changed_features=["Income"],
        ),
    ]
    default_order = sort_counterfactual_candidates(
        candidates=candidates,
        feature_order=["Income", "CCAvg"],
    )
    prefer_fewer_order = sort_counterfactual_candidates(
        candidates=candidates,
        feature_order=["Income", "CCAvg"],
        prefer_fewer_changes=True,
    )
    filtered_result, _ = apply_constraint_spec_to_candidates(
        result=CounterfactualResult(feasible=True, candidates=candidates, reason_codes=[]),
        constraint_spec={"prefer_fewer_changes": True},
        feature_order=["Income", "CCAvg"],
        sort_candidates=sort_counterfactual_candidates,
        request_constraints_blocked_code=REQUEST_CONSTRAINTS_BLOCKED,
    )
    if filtered_result.candidates != prefer_fewer_order:
        raise RuntimeError("prefer_fewer_changes probe did not preserve deterministic helper ordering.")
    return default_order, prefer_fewer_order


def build_manual_evidence_summary(
    *,
    sqlite_path: Path,
    manual_session_id: str,
    manual_evidence_note: str,
) -> dict[str, Any]:
    exists = False
    if manual_session_id and sqlite_path.exists():
        with sqlite3.connect(str(sqlite_path)) as connection:
            row = connection.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (manual_session_id,),
            ).fetchone()
            exists = row is not None
    return {
        "manual_session_id": manual_session_id or None,
        "manual_session_present_in_db": exists if manual_session_id else False,
        "manual_evidence_note": manual_evidence_note or None,
        "manual_evidence_recorded": bool(manual_session_id and exists),
    }


def build_acceptance_summary(
    *,
    run_id: str,
    base_url: str,
    sqlite_path: Path,
    catalog_path: Path,
    smoke_suite: dict[str, Any],
    repro_suite: dict[str, Any],
    metrics_suite: dict[str, Any],
    product_checks: dict[str, Any],
    constraint_checks: dict[str, Any],
    docs_checks: dict[str, bool],
    manual_evidence: dict[str, Any],
) -> dict[str, Any]:
    acceptance_gates = {
        "phase3_2_acceptance_report_formalized": True,
        "product_smoke_passed": smoke_suite["passed"],
        "reproducibility_probe_passed": repro_suite["passed"],
        "product_facing_metrics_report_passed": metrics_suite["passed"],
        "product_routes_and_persistence_verified": product_checks["passed"],
        "constraint_spec_v1_verified": constraint_checks["passed"],
        "claim_to_evidence_map_exists": docs_checks["claim_to_evidence_map_exists"],
        "thesis_alignment_notes_exist": docs_checks["thesis_alignment_notes_exists"],
    }
    failed_gates = [name for name, passed in acceptance_gates.items() if not passed]
    return {
        "run_id": run_id,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "base_url": base_url,
        "sqlite_path": str(sqlite_path),
        "catalog_path": str(catalog_path),
        "catalog_version": None if smoke_suite["summary"] is None else smoke_suite["summary"].get("catalog_version"),
        "git_commit": try_get_git_commit(ROOT),
        "smoke_suite": smoke_suite,
        "repro_suite": repro_suite,
        "metrics_suite": metrics_suite,
        "product_checks": product_checks,
        "constraint_checks": constraint_checks,
        "docs_checks": docs_checks,
        "manual_evidence": manual_evidence,
        "manual_evidence_policy": "informational_only",
        "acceptance_gates": acceptance_gates,
        "acceptance_passed": not failed_gates,
        "failed_gates": failed_gates,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3.2 Acceptance Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- base_url: `{summary['base_url']}`",
        f"- sqlite_path: `{summary['sqlite_path']}`",
        f"- catalog_path: `{summary['catalog_path']}`",
        f"- catalog_version: `{summary['catalog_version']}`",
        f"- git_commit: `{summary['git_commit']}`",
        "",
        "## Acceptance Gates",
        "",
    ]
    for key, passed in summary["acceptance_gates"].items():
        lines.append(f"- {key}: `{passed}`")
    lines.extend(
        [
            "",
            "## Product Smoke",
            "",
            f"- command: `{summary['smoke_suite']['command']}`",
            f"- passed: `{summary['smoke_suite']['passed']}`",
            f"- exit_code: `{summary['smoke_suite']['exit_code']}`",
            "",
            "## Reproducibility Probe",
            "",
            f"- command: `{summary['repro_suite']['command']}`",
            f"- passed: `{summary['repro_suite']['passed']}`",
            f"- exit_code: `{summary['repro_suite']['exit_code']}`",
            "",
            "## Product Metrics",
            "",
            f"- command: `{summary['metrics_suite']['command']}`",
            f"- passed: `{summary['metrics_suite']['passed']}`",
            f"- exit_code: `{summary['metrics_suite']['exit_code']}`",
            f"- report_json_path: `{None if summary['metrics_suite']['summary'] is None else summary['metrics_suite']['summary'].get('report_json_path')}`",
            f"- report_markdown_path: `{None if summary['metrics_suite']['summary'] is None else summary['metrics_suite']['summary'].get('report_markdown_path')}`",
            "",
            "## Product Verification",
            "",
        ]
    )
    for key, passed in summary["product_checks"]["checks"].items():
        lines.append(f"- {key}: `{passed}`")
    lines.extend(
        [
            "",
            "## Constraint Verification",
            "",
        ]
    )
    for key, passed in summary["constraint_checks"]["checks"].items():
        lines.append(f"- {key}: `{passed}`")
    lines.extend(
        [
            "",
            "## Thesis Alignment Files",
            "",
            f"- claim_to_evidence_map_exists: `{summary['docs_checks']['claim_to_evidence_map_exists']}`",
            f"- thesis_alignment_notes_exists: `{summary['docs_checks']['thesis_alignment_notes_exists']}`",
            "",
            "## Manual Evidence",
            "",
            f"- manual_evidence_policy: `{summary['manual_evidence_policy']}`",
            f"- manual_session_id: `{summary['manual_evidence']['manual_session_id']}`",
            f"- manual_session_present_in_db: `{summary['manual_evidence']['manual_session_present_in_db']}`",
            f"- manual_evidence_recorded: `{summary['manual_evidence']['manual_evidence_recorded']}`",
            "",
            "## Acceptance Verdict",
            "",
            f"- acceptance_passed: `{summary['acceptance_passed']}`",
            f"- failed_gates: `{summary['failed_gates']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
