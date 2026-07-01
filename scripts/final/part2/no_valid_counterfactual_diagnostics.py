#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import call_with_legacy_stdout_redirect, progress_iter
from llm.src.part2_eval.corpora import load_tier_b_bank_corpus
from llm.src.runtime.constraint_spec import effective_blocked_fields
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.reason_codes import NO_FEASIBLE_CF_FOUND, NO_RECOURSE_NEEDED, REQUEST_CONSTRAINTS_BLOCKED
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso
from ufce.core import cfmethods


DEFAULT_SOURCE_REPORT = (
    ROOT
    / "outputs"
    / "part2_closeout_bundle_live_1"
    / "part2_closeout_bundle_20260404_203633_542433"
    / "v1_thesis_metrics"
    / "part2_thesis_metrics_20260404_203759_456731"
    / "thesis_metrics_report.json"
)
DEFAULT_TIER_B_CORPUS = ROOT / "docs" / "validation" / "corpora" / "part2_tier_b_bank_sessions_v1.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "part2_no_valid_cf_diagnostics"
PRIMARY_RUN_ID = "part2_thesis_metrics_20260404_203759_456731"
EXPECTED_SOURCE_REPORT_SHA256 = "410bf30207d65418865e1716fba8b4dc999f63fe02d519ce5ce096e2ca89dfcd"
EXPECTED_CORPUS_SHA256 = "6dbd89e11be870a1a5d8a0d7747fa58968fc62b2fba27ea8062e811930000f78"
EXPECTED_EMPTY_NO_VALID_CF = {"G1": 69, "G2": 71}
EXPECTED_POST_FILTER_BLOCKED = 1
RUNNER_SCOPE = "part2_no_valid_cf_diagnostics"
SCORER_VERSION = "part2_no_valid_cf_diagnostics_v2_witness_logging"

RUNTIME_CONFIGS = {
    "B_C0": {"radius": 500, "n_neighbors": 1000},
    "B_C1": {"radius": 500, "n_neighbors": 2000},
    "B_C2": {"radius": 1000, "n_neighbors": 1000},
    "B_C3": {"radius": 1000, "n_neighbors": 2000},
}
UFCE_FIXED_CONFIG = {
    "contprox_metric": "euclidean",
    "min_act": 3,
    "min_feas": 2,
    "atol": 1e-5,
}
ORACLE_ATOL = float(UFCE_FIXED_CONFIG["atol"])
EXPECTED_FULL_FINAL_SUBTYPE_COUNTS = {
    "oracle_feasible_but_ufce_missed": 127,
    "likely_infeasible_under_formalized_constraints": 13,
}
BANK_CANONICAL_DISCRETE_DOMAINS = {
    "Family": [1, 2, 3, 4],
    "Education": [1, 2, 3],
    "SecuritiesAccount": [0, 1],
    "CDAccount": [0, 1],
    "Online": [0, 1],
    "CreditCard": [0, 1],
}
PRIMARY_OUTPUTS = [
    "runtime_semantics_audit.md",
    "diagnostic_report.json",
    "diagnostic_report.md",
    "experiment_a_post_filter_constraint_sanity.csv",
    "experiment_b_runtime_config_sensitivity.csv",
    "experiment_c_feasibility_upper_bound.csv",
    "final_no_valid_cf_subtypes.csv",
    "post_filter_constraint_blocked_sanity_case.csv",
    "table4x_constraint_sanity.csv",
    "table4y_runtime_config_sensitivity.csv",
    "table4z_feasibility_upper_bound.csv",
    "oracle_witnesses.jsonl",
    "oracle_infeasible_best_attempts.jsonl",
    "oracle_constraint_audit.csv",
    "oracle_representative_cases.md",
    "provenance.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part II no-valid-CF diagnostic phases D0/C0/A/B/C/F.")
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--tier-b-corpus", type=Path, default=DEFAULT_TIER_B_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--case-limit", type=int, default=None, help="Limit each diagnostic cohort for smoke runs.")
    parser.add_argument(
        "--config-ids",
        default=",".join(RUNTIME_CONFIGS),
        help="Comma-separated config IDs from B_C0,B_C1,B_C2,B_C3.",
    )
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_config_ids = parse_config_ids(args.config_ids)
    report = load_json(args.source_report)
    corpus = load_tier_b_bank_corpus(args.tier_b_corpus)
    source_report_sha256 = sha256_file(args.source_report)
    parity = build_parity_gate(
        report=report,
        corpus=corpus,
        source_report_path=args.source_report,
        source_report_sha256=source_report_sha256,
    )

    run_id = "part2_no_valid_cf_diagnostics_" + local_now_compact()
    run_root = args.out_dir.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    runtime = RuntimeOrchestrator()
    model_registry = runtime.model_registry if isinstance(runtime.model_registry, ModelRegistry) else ModelRegistry()
    policy_registry = runtime.policy_registry if isinstance(runtime.policy_registry, PolicyRegistry) else PolicyRegistry(model_registry)
    context = policy_registry.get_runtime_context("bank")
    feature_ranges = training_feature_ranges(context)
    feature_domains = build_feature_domains(context)
    model_input_audit = detect_oracle_model_input_transform(context)

    cohorts = build_cohorts(report=report, corpus=corpus, case_limit=args.case_limit)
    semantics = build_runtime_semantics_audit(run_root=run_root)
    phase_a_rows, post_filter_rows = run_phase_a(
        runtime=runtime,
        g2_empty_cases=cohorts["g2_empty_cases"],
        post_filter_cases=cohorts["post_filter_cases"],
        feature_ranges=feature_ranges,
        progress_enabled=not args.no_progress,
    )
    phase_b_rows = run_phase_b(
        runtime=runtime,
        empty_cases=cohorts["empty_cases"],
        config_ids=selected_config_ids,
        feature_ranges=feature_ranges,
        progress_enabled=not args.no_progress,
    )
    phase_c_rows = []
    if not args.skip_oracle:
        phase_c_rows = run_phase_c(
            empty_cases=cohorts["empty_cases"],
            context=context,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            model_input_audit=model_input_audit,
            progress_enabled=not args.no_progress,
        )
    final_rows = build_final_subtypes(
        empty_cases=cohorts["empty_cases"],
        phase_b_rows=phase_b_rows,
        phase_c_rows=phase_c_rows,
        oracle_skipped=args.skip_oracle,
    )
    tables = build_thesis_tables(
        phase_a_rows=phase_a_rows,
        phase_b_rows=phase_b_rows,
        phase_c_rows=phase_c_rows,
        final_rows=final_rows,
        post_filter_rows=post_filter_rows,
    )
    provenance = build_provenance(
        args=args,
        run_id=run_id,
        source_report_sha256=source_report_sha256,
        parity=parity,
        selected_config_ids=selected_config_ids,
        context=context,
        model_input_audit=model_input_audit,
    )
    oracle_witness_audit = build_oracle_witness_audit_summary(
        phase_c_rows=phase_c_rows,
        final_rows=final_rows,
        oracle_skipped=args.skip_oracle,
    )
    validate_oracle_audit(
        phase_c_rows=phase_c_rows,
        final_rows=final_rows,
        oracle_witness_audit=oracle_witness_audit,
        full_run=args.case_limit is None and not args.skip_oracle and set(selected_config_ids) == set(RUNTIME_CONFIGS),
    )
    summary = build_summary(
        run_id=run_id,
        parity=parity,
        semantics=semantics,
        phase_a_rows=phase_a_rows,
        phase_b_rows=phase_b_rows,
        phase_c_rows=phase_c_rows,
        final_rows=final_rows,
        post_filter_rows=post_filter_rows,
        provenance=provenance,
        oracle_witness_audit=oracle_witness_audit,
        oracle_skipped=args.skip_oracle,
    )
    write_outputs(
        run_root=run_root,
        summary=summary,
        provenance=provenance,
        phase_a_rows=phase_a_rows,
        post_filter_rows=post_filter_rows,
        phase_b_rows=phase_b_rows,
        phase_c_rows=phase_c_rows,
        final_rows=final_rows,
        tables=tables,
        oracle_witness_audit=oracle_witness_audit,
    )
    print(json.dumps({"run_id": run_id, "run_root": str(run_root), "outputs": PRIMARY_OUTPUTS}, indent=2, sort_keys=True))
    return 0


def parse_config_ids(raw: str) -> list[str]:
    config_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = [item for item in config_ids if item not in RUNTIME_CONFIGS]
    if unknown:
        raise ValueError("Unknown config IDs: " + ", ".join(unknown))
    if not config_ids:
        raise ValueError("At least one config ID is required.")
    return config_ids


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def try_get_git_commit() -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return output.strip() or None


def build_parity_gate(
    *,
    report: dict[str, Any],
    corpus: dict[str, Any],
    source_report_path: Path,
    source_report_sha256: str,
) -> dict[str, Any]:
    rows = list(report.get("per_case_results") or [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("source report per_case_results must be a list of objects")
    counts = {
        "G1_no_valid_cf": sum(1 for row in rows if row.get("group") == "G1" and row.get("reject_class") == "no_feasible_cf"),
        "G2_no_valid_cf": sum(1 for row in rows if row.get("group") == "G2" and row.get("reject_class") == "no_feasible_cf"),
        "constraint_blocked_post_filter": sum(
            1 for row in rows if row.get("group") == "G2" and row.get("reject_class") == "request_constraints_blocked"
        ),
    }
    expected = {
        "G1_no_valid_cf": EXPECTED_EMPTY_NO_VALID_CF["G1"],
        "G2_no_valid_cf": EXPECTED_EMPTY_NO_VALID_CF["G2"],
        "constraint_blocked_post_filter": EXPECTED_POST_FILTER_BLOCKED,
    }
    checks = {
        "source_report_path": str(Path(source_report_path).resolve()),
        "source_report_sha256": source_report_sha256,
        "expected_source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_report_sha256_match": source_report_sha256 == EXPECTED_SOURCE_REPORT_SHA256,
        "run_id": report.get("run_id"),
        "primary_run_id_match": report.get("run_id") == PRIMARY_RUN_ID,
        "aggregate_validation_ok": (report.get("aggregate_validation") or {}).get("ok") is True,
        "corpus_sha256": report.get("corpus_sha256"),
        "corpus_sha256_from_corpus": corpus.get("corpus_sha256"),
        "expected_corpus_sha256": EXPECTED_CORPUS_SHA256,
        "corpus_sha256_match": report.get("corpus_sha256") == corpus.get("corpus_sha256") == EXPECTED_CORPUS_SHA256,
        "counts": counts,
        "expected_counts": expected,
        "counts_match": counts == expected,
    }
    failed = [key for key, value in checks.items() if key.endswith("_match") and value is not True]
    if checks["aggregate_validation_ok"] is not True:
        failed.append("aggregate_validation_ok")
    if failed:
        raise ValueError("C0 parity gate failed: " + ", ".join(failed))
    checks["ok"] = True
    return checks


def build_cohorts(*, report: dict[str, Any], corpus: dict[str, Any], case_limit: int | None) -> dict[str, list[dict[str, Any]]]:
    cases_by_id = {case["case_id"]: case for case in corpus.get("cases", []) if isinstance(case, dict)}
    rows = list(report.get("per_case_results") or [])
    empty_rows = [
        row
        for row in rows
        if row.get("group") in {"G1", "G2"}
        and row.get("summary_type") == "runtime_reject"
        and row.get("reject_class") == "no_feasible_cf"
    ]
    post_filter_rows = [
        row
        for row in rows
        if row.get("group") == "G2"
        and row.get("summary_type") == "runtime_reject"
        and row.get("reject_class") == "request_constraints_blocked"
    ]
    empty_cases = [merge_case_row(row, cases_by_id) for row in empty_rows]
    g1_empty_cases = [case for case in empty_cases if case["group"] == "G1"]
    g2_empty_cases = [case for case in empty_cases if case["group"] == "G2"]
    post_filter_cases = [merge_case_row(row, cases_by_id) for row in post_filter_rows]
    if case_limit is not None:
        limit = max(0, int(case_limit))
        g1_empty_cases = g1_empty_cases[:limit]
        g2_empty_cases = g2_empty_cases[:limit]
        empty_cases = g1_empty_cases + g2_empty_cases
        post_filter_cases = post_filter_cases[: min(limit, len(post_filter_cases))]
    return {
        "empty_cases": empty_cases,
        "g2_empty_cases": g2_empty_cases,
        "post_filter_cases": post_filter_cases,
    }


def merge_case_row(row: dict[str, Any], cases_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = str(row["case_id"])
    case = cases_by_id.get(case_id)
    if case is None:
        raise ValueError(f"Case {case_id} exists in source report but not in Tier-B corpus.")
    merged = dict(case)
    merged["source_report_row"] = dict(row)
    merged["session_id"] = row.get("session_id")
    merged["original_summary_type"] = row.get("summary_type")
    merged["original_reject_class"] = row.get("reject_class")
    merged["original_constraint_spec"] = case.get("active_constraint_spec_expected")
    merged["constraint_group"] = constraint_group(case.get("active_constraint_spec_expected"))
    return merged


def constraint_group(spec: Any) -> str:
    if not isinstance(spec, dict) or not spec:
        return "none"
    if "numeric_bounds" in spec:
        return "numeric_bound"
    has_blocked = bool(spec.get("immutable") or spec.get("disallowed_changes"))
    has_max = isinstance(spec.get("max_changed_features"), int)
    has_prefer = "prefer_fewer_changes" in spec
    if has_blocked and has_max:
        return "hard_constraint"
    if has_blocked and has_prefer:
        return "constraint_sensitive"
    if has_blocked:
        return "blocked_feature"
    if has_max:
        return "hard_constraint"
    if has_prefer:
        return "constraint_sensitive"
    return "other_constraint"


def build_runtime_semantics_audit(*, run_root: Path) -> dict[str, Any]:
    source_paths = {
        "ufce_request_builder": ROOT / "llm" / "src" / "runtime" / "ufce_request_builder.py",
        "ufce_mapper": ROOT / "llm" / "src" / "runtime" / "backend_packages" / "ufce" / "mapper.py",
        "orchestrator": ROOT / "llm" / "src" / "runtime" / "orchestrator.py",
        "constraint_spec": ROOT / "llm" / "src" / "runtime" / "constraint_spec.py",
        "verification_checks": ROOT / "llm" / "src" / "runtime" / "verification" / "checks.py",
    }
    texts = {name: path.read_text(encoding="utf-8") for name, path in source_paths.items()}
    backend_pos = texts["orchestrator"].find("_generate_backend_result(")
    constraint_pos = texts["orchestrator"].find("_apply_request_constraints(")
    audit = {
        "expected_finding": "constraint_spec is applied after UFCE candidate generation",
        "ufce_request_builder_mentions_constraint_spec": "constraint_spec" in texts["ufce_request_builder"],
        "ufce_mapper_mentions_constraint_spec": "constraint_spec" in texts["ufce_mapper"],
        "orchestrator_applies_constraints_after_backend_generation": backend_pos >= 0 and constraint_pos > backend_pos,
        "post_generation_filter_function_present": "apply_constraint_spec_to_candidates" in texts["constraint_spec"],
        "post_generation_hard_constraint_verifier_present": "class HardConstraintCheck" in texts["verification_checks"],
        "source_files": {key: str(value) for key, value in source_paths.items()},
    }
    audit["semantics_ok"] = (
        not audit["ufce_request_builder_mentions_constraint_spec"]
        and not audit["ufce_mapper_mentions_constraint_spec"]
        and audit["orchestrator_applies_constraints_after_backend_generation"]
        and audit["post_generation_filter_function_present"]
        and audit["post_generation_hard_constraint_verifier_present"]
    )
    markdown = render_runtime_semantics_audit(audit)
    (run_root / "runtime_semantics_audit.md").write_text(markdown, encoding="utf-8")
    return audit


def render_runtime_semantics_audit(audit: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Runtime Semantics Audit",
            "",
            f"- expected_finding: `{audit['expected_finding']}`",
            f"- semantics_ok: `{audit['semantics_ok']}`",
            f"- UFCERequestBuilder mentions constraint_spec: `{audit['ufce_request_builder_mentions_constraint_spec']}`",
            f"- UFCERequestMapper mentions constraint_spec: `{audit['ufce_mapper_mentions_constraint_spec']}`",
            f"- constraints applied after backend generation: `{audit['orchestrator_applies_constraints_after_backend_generation']}`",
            f"- post-generation filter function present: `{audit['post_generation_filter_function_present']}`",
            f"- hard constraint verifier present: `{audit['post_generation_hard_constraint_verifier_present']}`",
            "",
            "Interpretation: under the current runtime, request constraints can block or verify generated candidates after UFCE runs. "
            "They are not passed into UFCE candidate generation through the request builder/mapper.",
            "",
        ]
    )


def run_phase_a(
    *,
    runtime: RuntimeOrchestrator,
    g2_empty_cases: list[dict[str, Any]],
    post_filter_cases: list[dict[str, Any]],
    feature_ranges: dict[str, tuple[float, float]],
    progress_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for case in progress_iter(g2_empty_cases, enabled=progress_enabled, desc="Phase A empty G2", unit="case"):
        original = evaluate_runtime_case(runtime=runtime, case=case, constraint_spec=case.get("original_constraint_spec"), config_id="B_C0", feature_ranges=feature_ranges)
        relaxed = evaluate_runtime_case(runtime=runtime, case=case, constraint_spec=None, config_id="B_C0", feature_ranges=feature_ranges)
        invariant_status = phase_a_invariant_status(original, relaxed)
        for mode, result in (("original", original), ("relaxed", relaxed)):
            rows.append(
                {
                    **base_case_columns(case),
                    "mode": mode,
                    "outcome": result["outcome"],
                    "generated_candidate_count": result["generated_candidate_count"],
                    "valid_cf_count": result["valid_cf_count"],
                    "reason_codes": json.dumps(result["reason_codes"], ensure_ascii=True),
                    "invariant_status": invariant_status,
                    "finding": "hidden_coupling_finding" if invariant_status != "expected_invariant_holds" else "expected_invariant_holds",
                }
            )
    post_filter_rows: list[dict[str, Any]] = []
    for case in post_filter_cases:
        original = evaluate_runtime_case(runtime=runtime, case=case, constraint_spec=case.get("original_constraint_spec"), config_id="B_C0", feature_ranges=feature_ranges)
        relaxed = evaluate_runtime_case(runtime=runtime, case=case, constraint_spec=None, config_id="B_C0", feature_ranges=feature_ranges)
        for mode, result in (("original", original), ("relaxed", relaxed)):
            post_filter_rows.append(
                {
                    **base_case_columns(case),
                    "mode": mode,
                    "outcome": result["outcome"],
                    "generated_candidate_count": result["generated_candidate_count"],
                    "valid_cf_count": result["valid_cf_count"],
                    "changed_features": json.dumps(result["changed_features"], ensure_ascii=True),
                    "reason_codes": json.dumps(result["reason_codes"], ensure_ascii=True),
                    "relaxed_allows_publication": mode == "relaxed" and result["outcome"] == "counterfactual_found",
                }
            )
    return rows, post_filter_rows


def phase_a_invariant_status(original: dict[str, Any], relaxed: dict[str, Any]) -> str:
    if (
        original["outcome"] == relaxed["outcome"] == "no_valid_cf"
        and int(original["generated_candidate_count"]) == int(relaxed["generated_candidate_count"])
    ):
        return "expected_invariant_holds"
    return "runtime_semantics_violation_or_hidden_coupling"


def run_phase_b(
    *,
    runtime: RuntimeOrchestrator,
    empty_cases: list[dict[str, Any]],
    config_ids: list[str],
    feature_ranges: dict[str, tuple[float, float]],
    progress_enabled: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    units = [(case, config_id) for case in empty_cases for config_id in config_ids]
    for case, config_id in progress_iter(units, enabled=progress_enabled, desc="Phase B runtime configs", unit="run"):
        result = evaluate_runtime_case(
            runtime=runtime,
            case=case,
            constraint_spec=case.get("original_constraint_spec"),
            config_id=config_id,
            feature_ranges=feature_ranges,
        )
        config = RUNTIME_CONFIGS[config_id]
        rows.append(
            {
                **base_case_columns(case),
                "config_id": config_id,
                "radius": config["radius"],
                "n_neighbors": config["n_neighbors"],
                "outcome": result["outcome"],
                "generated_candidate_count": result["generated_candidate_count"],
                "valid_cf_count": result["valid_cf_count"],
                "changed_features": json.dumps(result["changed_features"], ensure_ascii=True),
                "prox_euc": result["prox_euc"],
                "sparsity": result["sparsity"],
                "runtime_seconds": result["runtime_seconds"],
                "reason_codes": json.dumps(result["reason_codes"], ensure_ascii=True),
                "service_error_count": result["service_error_count"],
            }
        )
    return rows


def evaluate_runtime_case(
    *,
    runtime: RuntimeOrchestrator,
    case: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    config_id: str,
    feature_ranges: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    config = RUNTIME_CONFIGS[config_id]
    call_with_legacy_stdout_redirect(
        cfmethods.initUFCE,
        radius=int(config["radius"]),
        n_neighbors=int(config["n_neighbors"]),
        contprox_metric=UFCE_FIXED_CONFIG["contprox_metric"],
        min_act=int(UFCE_FIXED_CONFIG["min_act"]),
        min_feas=int(UFCE_FIXED_CONFIG["min_feas"]),
        atol=float(UFCE_FIXED_CONFIG["atol"]),
    )
    request = {"dataset": "bank", "profile": dict(case["seed_profile"])}
    if isinstance(constraint_spec, dict) and constraint_spec:
        request["constraint_spec"] = dict(constraint_spec)
    start = time.perf_counter()
    result = runtime.handle(request, include_debug_trace=True)
    elapsed = time.perf_counter() - start
    payload = result.to_dict(include_debug_trace=True)
    debug = payload.get("debug_trace") or {}
    counterfactual = payload.get("counterfactual") or {}
    candidates = list(counterfactual.get("candidates") or [])
    first_candidate = candidates[0] if candidates else {}
    profile = first_candidate.get("profile") if isinstance(first_candidate, dict) else None
    changed = list(first_candidate.get("changed_features") or []) if isinstance(first_candidate, dict) else []
    reason_codes = list(payload.get("reason_codes") or counterfactual.get("reason_codes") or [])
    generated = int((debug.get("generation_stats") or {}).get("generated_candidate_count") or len(payload.get("canonical_candidates") or []))
    valid_cf_count = len(candidates) if counterfactual.get("feasible") else 0
    return {
        "outcome": classify_runtime_outcome(payload),
        "generated_candidate_count": generated,
        "valid_cf_count": valid_cf_count,
        "changed_features": changed,
        "prox_euc": normalized_proximity(case["seed_profile"], profile, feature_ranges) if isinstance(profile, dict) else None,
        "sparsity": len(changed) if changed else None,
        "runtime_seconds": round(elapsed, 6),
        "reason_codes": reason_codes,
        "service_error_count": len(debug.get("service_errors") or []),
    }


def classify_runtime_outcome(payload: dict[str, Any]) -> str:
    counterfactual = payload.get("counterfactual") or {}
    reason_codes = set(payload.get("reason_codes") or counterfactual.get("reason_codes") or [])
    if bool(counterfactual.get("feasible")) and counterfactual.get("candidates"):
        return "counterfactual_found"
    if NO_RECOURSE_NEEDED in reason_codes:
        return "no_recourse_needed"
    if REQUEST_CONSTRAINTS_BLOCKED in reason_codes:
        return "constraint_blocked_post_filter"
    if NO_FEASIBLE_CF_FOUND in reason_codes:
        return "no_valid_cf"
    if reason_codes:
        return "runtime_reject_other"
    return "other"


def base_case_columns(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "session_id": case.get("session_id") or "",
        "group": case["group"],
        "constraint_group": case.get("constraint_group") or "none",
        "constraint_spec": json.dumps(case.get("original_constraint_spec") or {}, ensure_ascii=True, sort_keys=True),
    }


def training_feature_ranges(context) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for feature_name in context.bundle.feature_order:
        series = context.bundle.Xtrain[feature_name]
        ranges[feature_name] = (float(series.min()), float(series.max()))
    return ranges


def normalized_proximity(
    factual_profile: dict[str, Any],
    candidate_profile: dict[str, Any] | None,
    feature_ranges: dict[str, tuple[float, float]],
) -> float | None:
    if not isinstance(candidate_profile, dict):
        return None
    squared = 0.0
    for feature_name, (minimum, maximum) in feature_ranges.items():
        span = maximum - minimum
        if span <= 0:
            span = 1.0
        delta = (float(candidate_profile[feature_name]) - float(factual_profile[feature_name])) / span
        squared += delta * delta
    return round(math.sqrt(squared), 6)


def build_feature_domains(context) -> dict[str, list[Any]]:
    domains: dict[str, list[Any]] = {}
    for feature_name in context.bundle.feature_order:
        feature_type = context.policy.feature_type_map[feature_name]
        if feature_type == "float":
            continue
        if feature_name in BANK_CANONICAL_DISCRETE_DOMAINS:
            domains[feature_name] = list(BANK_CANONICAL_DISCRETE_DOMAINS[feature_name])
            continue
        values = sorted({int(value) for value in context.bundle.Xtrain[feature_name].dropna().tolist()})
        domains[feature_name] = values
    return domains


def run_phase_c(
    *,
    empty_cases: list[dict[str, Any]],
    context,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    model_input_audit: dict[str, Any],
    progress_enabled: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in progress_iter(empty_cases, enabled=progress_enabled, desc="Phase C oracle", unit="case"):
        original_profile = ordered_profile(case["seed_profile"], context)
        oracle = lr_feasibility_oracle(
            factual_profile=original_profile,
            constraint_spec=case.get("original_constraint_spec"),
            context=context,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            model_input_audit=model_input_audit,
        )
        empirical = empirical_witness(
            factual_profile=original_profile,
            constraint_spec=case.get("original_constraint_spec"),
            context=context,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            model_input_audit=model_input_audit,
        )
        witness = oracle.get("witness")
        best_scored = oracle.get("best_scored_profile")
        witness_score = oracle.get("witness_score") or {}
        best_score = oracle.get("best_scored_score") or {}
        audit = oracle.get("constraint_audit") or {}
        rows.append(
            {
                **base_case_columns(case),
                "original_profile_json": json_dumps_compact(original_profile),
                "oracle_status": oracle["status"],
                "oracle_feasible": oracle["feasible"],
                "oracle_best_margin": oracle["best_margin"],
                "oracle_changed_features": json_dumps_compact(oracle["changed_features"]),
                "oracle_error": oracle["error"],
                "oracle_witness_persisted": witness is not None,
                "oracle_witness_profile_json": json_dumps_compact(witness) if witness is not None else "",
                "oracle_witness_model_input_json": json_dumps_compact(oracle.get("witness_model_input")) if witness is not None else "",
                "oracle_witness_lr_margin": witness_score.get("margin"),
                "oracle_witness_lr_probability": witness_score.get("probability"),
                "oracle_witness_predicted_label": witness_score.get("predicted_label"),
                "desired_label": oracle.get("desired_label"),
                "oracle_label_satisfied": oracle.get("label_satisfied"),
                "oracle_atol": ORACLE_ATOL,
                "oracle_changed_feature_atol": ORACLE_ATOL,
                "oracle_numeric_equality_atol": ORACLE_ATOL,
                "oracle_witness_changed_features_json": json_dumps_compact(oracle.get("witness_changed_features") or []),
                "oracle_witness_changed_values_json": json_dumps_compact(oracle.get("witness_changed_values") or {}),
                "oracle_witness_change_count": oracle.get("witness_change_count"),
                "oracle_max_changed_features": oracle.get("max_changed_features"),
                "oracle_max_change_satisfied": audit.get("max_change_satisfied"),
                "oracle_within_continuous_bounds": audit.get("within_continuous_bounds"),
                "oracle_categorical_domain_satisfied": audit.get("categorical_domain_satisfied"),
                "oracle_binary_domain_satisfied": audit.get("binary_domain_satisfied"),
                "oracle_blocked_fields_satisfied": audit.get("blocked_fields_satisfied"),
                "oracle_numeric_bounds_satisfied": audit.get("numeric_bounds_satisfied"),
                "oracle_actionable_features_satisfied": audit.get("actionable_features_satisfied"),
                "oracle_all_constraints_satisfied": audit.get("all_constraints_satisfied"),
                "oracle_violated_constraints_json": json_dumps_compact(audit.get("violated_constraints") or []),
                "oracle_allowed_changeable_features_json": json_dumps_compact(oracle.get("allowed_changeable_features") or []),
                "oracle_blocked_features_json": json_dumps_compact(oracle.get("blocked_features") or []),
                "oracle_effective_changeable_features_json": json_dumps_compact(oracle.get("effective_changeable_features") or []),
                "oracle_subset_count_evaluated": oracle.get("subset_count_evaluated"),
                "oracle_candidate_count_scored": oracle.get("candidate_count_scored"),
                "oracle_selected_subset_json": json_dumps_compact(oracle.get("selected_subset") or []),
                "oracle_selected_subset_size": len(oracle.get("selected_subset") or []),
                "oracle_selected_subset_satisfies_max_change": oracle.get("selected_subset_satisfies_max_change"),
                "oracle_best_scored_profile_json": json_dumps_compact(best_scored) if best_scored is not None else "",
                "oracle_best_scored_model_input_json": json_dumps_compact(oracle.get("best_scored_model_input")) if best_scored is not None else "",
                "oracle_best_scored_margin": best_score.get("margin"),
                "oracle_best_scored_probability": best_score.get("probability"),
                "oracle_best_scored_predicted_label": best_score.get("predicted_label"),
                "oracle_best_scored_changed_features_json": json_dumps_compact(oracle.get("best_scored_changed_features") or []),
                "oracle_best_scored_change_count": oracle.get("best_scored_change_count"),
                "oracle_best_scored_constraints_satisfied": oracle.get("best_scored_constraints_satisfied"),
                "oracle_best_scored_reason_for_infeasible": oracle.get("best_scored_reason_for_infeasible"),
                "oracle_search_completed": oracle.get("search_completed"),
                "oracle_best_scored_is_valid_witness": oracle.get("best_scored_is_valid_witness"),
                "is_valid_oracle_witness": oracle.get("is_valid_oracle_witness"),
                "oracle_reloaded_witness_score_matches": oracle.get("reloaded_witness_score_matches"),
                "oracle_reloaded_witness_audit_matches": oracle.get("reloaded_witness_audit_matches"),
                "empirical_witness_found": empirical["found"],
                "empirical_witness_row_id": empirical["row_id"],
                "empirical_witness_profile_json": json_dumps_compact(empirical["profile"]) if empirical["profile"] is not None else "",
                "empirical_witness_distance": empirical["distance"],
                "empirical_witness_changed_features": json_dumps_compact(empirical["changed_features"]),
                "empirical_witness_predicted_label": empirical["predicted_label"],
                "empirical_witness_constraints_satisfied": empirical["constraints_satisfied"],
            }
        )
    return rows


def lr_feasibility_oracle(
    *,
    factual_profile: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    context,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    model_input_audit: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _lr_feasibility_oracle(
            factual_profile=factual_profile,
            constraint_spec=constraint_spec,
            context=context,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            model_input_audit=model_input_audit,
        )
    except Exception as exc:
        return {
            "status": "error",
            "feasible": None,
            "best_margin": None,
            "changed_features": [],
            "witness": None,
            "best_scored_profile": None,
            "search_completed": False,
            "error": str(exc),
        }


def _lr_feasibility_oracle(
    *,
    factual_profile: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    context,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    model_input_audit: dict[str, Any],
) -> dict[str, Any]:
    feature_order = list(context.bundle.feature_order)
    desired = int(context.policy.desired_outcome)
    lr = context.bundle.lr
    classes = [int(item) for item in list(lr.classes_)]
    positive_class = int(classes[-1])
    target_positive = desired == positive_class
    coef = np.asarray(lr.coef_, dtype=float).reshape(1, -1)[0]
    coef_by_feature = {feature: float(coef[index]) for index, feature in enumerate(feature_order)}
    blocked = set(effective_blocked_fields(constraint_spec, feature_order=feature_order))
    changeable = set(context.policy.f2change)
    fixed = {feature for feature in feature_order if feature not in changeable} | blocked
    max_changes_raw = constraint_spec.get("max_changed_features") if isinstance(constraint_spec, dict) else None
    max_changes = int(max_changes_raw) if isinstance(max_changes_raw, int) else len(feature_order)
    numeric_bounds = constraint_spec.get("numeric_bounds") if isinstance(constraint_spec, dict) and isinstance(constraint_spec.get("numeric_bounds"), dict) else {}
    allowed_changeable = [feature for feature in feature_order if feature in changeable]
    blocked_features = [feature for feature in feature_order if feature in blocked]
    effective_changeable = [feature for feature in allowed_changeable if feature not in blocked]

    continuous = [feature for feature in feature_order if context.policy.feature_type_map[feature] == "float"]
    discrete = [feature for feature in feature_order if feature not in continuous]
    discrete_domains: list[list[Any]] = []
    for feature in discrete:
        domain = [coerce_feature_value(feature, factual_profile[feature], context)] if feature in fixed else list(feature_domains[feature])
        domain = [value for value in domain if satisfies_bound(feature, value, numeric_bounds, feature_ranges)]
        if not domain:
            return oracle_result(
                feasible=False,
                margin=None,
                changed=[],
                witness=None,
                best_scored_profile=None,
                best_scored_score=None,
                best_scored_audit=None,
                selected_subset=[],
                evaluated_subsets=set(),
                candidate_count_scored=0,
                context=context,
                factual_profile=factual_profile,
                constraint_spec=constraint_spec,
                feature_domains=feature_domains,
                feature_ranges=feature_ranges,
                max_changed_features=max_changes_raw,
                allowed_changeable_features=allowed_changeable,
                blocked_features=blocked_features,
                effective_changeable_features=effective_changeable,
                desired_label=desired,
                status="infeasible",
                error=f"empty domain for {feature}",
            )
        discrete_domains.append(domain)

    best_score = -math.inf if target_positive else math.inf
    best_profile = None
    best_score_payload: dict[str, Any] | None = None
    best_audit: dict[str, Any] | None = None
    best_subset: list[str] = []
    evaluated_subsets: set[tuple[str, ...]] = set()
    candidate_count_scored = 0
    for values in product(*discrete_domains):
        profile = {feature: value for feature, value in zip(discrete, values)}
        changed_discrete = [
            feature
            for feature in discrete
            if not same_value(profile[feature], factual_profile[feature], context.policy.feature_type_map[feature], atol=ORACLE_ATOL)
        ]
        if len(changed_discrete) > max_changes:
            continue
        remaining_changes = max_changes - len(changed_discrete)
        continuous_options = continuous_profiles(
            continuous=continuous,
            fixed=fixed,
            factual_profile=factual_profile,
            numeric_bounds=numeric_bounds,
            feature_ranges=feature_ranges,
            coef_by_feature=coef_by_feature,
            target_positive=target_positive,
            remaining_changes=remaining_changes,
            context=context,
        )
        for cont_option in continuous_options:
            cont_profile = cont_option["profile"]
            full_profile = ordered_profile({**profile, **cont_profile}, context)
            audit = audit_oracle_candidate(
                factual_profile=factual_profile,
                candidate_profile=full_profile,
                constraint_spec=constraint_spec,
                context=context,
                feature_domains=feature_domains,
                feature_ranges=feature_ranges,
                atol=ORACLE_ATOL,
            )
            changed = audit["changed_features"]
            if len(changed) > max_changes:
                continue
            selected_subset = [feature for feature in feature_order if feature in set(changed)]
            evaluated_subsets.add(tuple(selected_subset))
            score_payload = score_lr_profile(full_profile, context, model_input_audit=model_input_audit)
            score = float(score_payload["margin"])
            candidate_count_scored += 1
            if (target_positive and score > best_score) or ((not target_positive) and score < best_score):
                best_score = score
                best_profile = full_profile
                best_score_payload = score_payload
                best_audit = audit
                best_subset = selected_subset
    if best_profile is None:
        return oracle_result(
            feasible=False,
            margin=None,
            changed=[],
            witness=None,
            best_scored_profile=None,
            best_scored_score=None,
            best_scored_audit=None,
            selected_subset=[],
            evaluated_subsets=evaluated_subsets,
            candidate_count_scored=candidate_count_scored,
            context=context,
            factual_profile=factual_profile,
            constraint_spec=constraint_spec,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            max_changed_features=max_changes_raw,
            allowed_changeable_features=allowed_changeable,
            blocked_features=blocked_features,
            effective_changeable_features=effective_changeable,
            desired_label=desired,
            status="infeasible",
            error="no scored profile survived oracle domain and constraint pruning",
        )
    feasible = best_score >= 0.0 if target_positive else best_score < 0.0
    changed = best_audit["changed_features"] if best_audit else []
    witness = best_profile if feasible else None
    witness_score = best_score_payload if feasible else None
    witness_audit = best_audit if feasible else None
    reload_check = {"score_matches": False, "audit_matches": False}
    if witness is not None:
        reload_check = validate_witness_json_reload(
            witness_profile=witness,
            persisted_score=witness_score or {},
            persisted_audit=witness_audit or {},
            factual_profile=factual_profile,
            constraint_spec=constraint_spec,
            context=context,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            model_input_audit=model_input_audit,
            atol=ORACLE_ATOL,
        )
    return oracle_result(
        feasible=feasible,
        margin=round_float(best_score),
        changed=changed,
        witness=witness,
        best_scored_profile=best_profile,
        best_scored_score=best_score_payload,
        best_scored_audit=best_audit,
        selected_subset=best_subset,
        evaluated_subsets=evaluated_subsets,
        candidate_count_scored=candidate_count_scored,
        context=context,
        factual_profile=factual_profile,
        constraint_spec=constraint_spec,
        feature_domains=feature_domains,
        feature_ranges=feature_ranges,
        max_changed_features=max_changes_raw,
        allowed_changeable_features=allowed_changeable,
        blocked_features=blocked_features,
        effective_changeable_features=effective_changeable,
        desired_label=desired,
        status="feasible" if feasible else "infeasible",
        error="",
        reload_check=reload_check,
    )


def continuous_profiles(
    *,
    continuous: list[str],
    fixed: set[str],
    factual_profile: dict[str, Any],
    numeric_bounds: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
    coef_by_feature: dict[str, float],
    target_positive: bool,
    remaining_changes: int,
    context,
) -> list[dict[str, Any]]:
    changeable_cont = [feature for feature in continuous if feature not in fixed]
    options: list[dict[str, Any]] = []
    for count in range(0, min(len(changeable_cont), max(0, int(remaining_changes))) + 1):
        for subset_tuple in combinations(changeable_cont, count):
            subset = set(subset_tuple)
            profile: dict[str, Any] = {}
            valid = True
            for feature in continuous:
                if feature in fixed or feature not in subset:
                    value = float(factual_profile[feature])
                else:
                    low, high = constrained_numeric_range(feature, numeric_bounds, feature_ranges)
                    if low > high:
                        valid = False
                        break
                    coef = coef_by_feature[feature]
                    value = high if (coef >= 0.0) == target_positive else low
                if not satisfies_bound(feature, value, numeric_bounds, feature_ranges):
                    valid = False
                    break
                profile[feature] = coerce_feature_value(feature, value, context)
            if valid:
                options.append({"profile": profile, "subset": [feature for feature in continuous if feature in subset]})
    return options


def detect_oracle_model_input_transform(context) -> dict[str, Any]:
    feature_order = list(context.bundle.feature_order)
    lr = context.bundle.lr
    coef_width = int(np.asarray(lr.coef_, dtype=float).reshape(1, -1).shape[1])
    if coef_width != len(feature_order):
        raise ValueError(f"Oracle LR coefficient width {coef_width} does not match Bank feature_order length {len(feature_order)}.")
    known_transform_attrs = [
        "scaler",
        "transformer",
        "preprocessor",
        "pipeline",
        "model_pipeline",
        "feature_transformer",
        "x_scaler",
        "X_scaler",
    ]
    non_identity_objects = {
        name: type(getattr(context.bundle, name)).__name__
        for name in known_transform_attrs
        if getattr(context.bundle, name, None) is not None
    }
    if bool(getattr(context.bundle, "has_scaler", False)):
        non_identity_objects["has_scaler"] = "True"
    if non_identity_objects:
        raise ValueError(f"Non-identity oracle model-input transform detected: {non_identity_objects}")
    return {
        "oracle_model_input_transform": "identity",
        "model_input_feature_order": feature_order,
        "lr_coef_width": coef_width,
        "checked_transform_attrs": known_transform_attrs + ["has_scaler"],
    }


def ordered_profile(profile: dict[str, Any], context) -> dict[str, Any]:
    return {feature: coerce_feature_value(feature, profile[feature], context) for feature in context.bundle.feature_order}


def score_lr_profile(
    profile: dict[str, Any],
    context,
    *,
    model_input_audit: dict[str, Any],
) -> dict[str, Any]:
    feature_order = list(context.bundle.feature_order)
    if model_input_audit.get("oracle_model_input_transform") != "identity":
        raise ValueError("Oracle LR scoring requires identity model input transform.")
    if list(model_input_audit.get("model_input_feature_order") or []) != feature_order:
        raise ValueError("Oracle model input feature order does not match Bank feature_order.")
    model_input = ordered_profile(profile, context)
    lr = context.bundle.lr
    coef = np.asarray(lr.coef_, dtype=float).reshape(1, -1)[0]
    intercept = float(np.asarray(lr.intercept_, dtype=float).reshape(-1)[0])
    margin = intercept + sum(float(coef[index]) * float(model_input[feature]) for index, feature in enumerate(feature_order))
    probability = sigmoid(float(margin))
    classes = [int(item) for item in list(lr.classes_)]
    predicted_label = int(classes[-1] if probability >= 0.5 else classes[0])
    return {
        "margin": round_float(margin),
        "probability": round_float(probability),
        "predicted_label": predicted_label,
        "model_input_profile": model_input,
    }


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def round_float(value: Any, digits: int = 12) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isinf(number) or math.isnan(number):
        return None
    return round(number, digits)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def json_dumps_compact(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def audit_oracle_candidate(
    *,
    factual_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    context,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    atol: float,
) -> dict[str, Any]:
    feature_order = list(context.bundle.feature_order)
    feature_type_map = context.policy.feature_type_map
    changed = changed_features(factual_profile, candidate_profile, feature_type_map, atol=atol)
    changed_set = set(changed)
    changed_values = {
        feature: {"from": factual_profile[feature], "to": candidate_profile[feature]}
        for feature in changed
    }
    numeric_bounds = constraint_spec.get("numeric_bounds") if isinstance(constraint_spec, dict) and isinstance(constraint_spec.get("numeric_bounds"), dict) else {}
    blocked = set(effective_blocked_fields(constraint_spec, feature_order=feature_order))
    max_changes = constraint_spec.get("max_changed_features") if isinstance(constraint_spec, dict) else None
    max_changes = int(max_changes) if isinstance(max_changes, int) else None
    violations: list[dict[str, Any]] = []

    within_continuous_bounds = True
    categorical_domain_satisfied = True
    binary_domain_satisfied = True
    numeric_bounds_satisfied = True

    for feature in feature_order:
        value = candidate_profile[feature]
        feature_type = feature_type_map[feature]
        if feature_type == "float" and feature in feature_ranges:
            low, high = feature_ranges[feature]
            if float(value) < low - atol or float(value) > high + atol:
                within_continuous_bounds = False
                violations.append({"type": "continuous_bound", "feature": feature, "value": value, "bound": f"[{low},{high}]"})
        if feature_type == "int":
            domain = list(feature_domains.get(feature, []))
            if int(value) not in {int(item) for item in domain}:
                categorical_domain_satisfied = False
                violations.append({"type": "categorical_domain", "feature": feature, "value": value, "domain": domain})
        if feature_type == "binary":
            if int(value) not in {0, 1}:
                binary_domain_satisfied = False
                violations.append({"type": "binary_domain", "feature": feature, "value": value, "domain": [0, 1]})
        bounds = numeric_bounds.get(feature)
        if isinstance(bounds, dict):
            if "min" in bounds and float(value) < float(bounds["min"]) - atol:
                numeric_bounds_satisfied = False
                violations.append({"type": "numeric_bound", "feature": feature, "value": value, "bound": f">={float(bounds['min'])}"})
            if "max" in bounds and float(value) > float(bounds["max"]) + atol:
                numeric_bounds_satisfied = False
                violations.append({"type": "numeric_bound", "feature": feature, "value": value, "bound": f"<={float(bounds['max'])}"})

    blocked_fields_satisfied = not any(feature in blocked for feature in changed_set)
    if not blocked_fields_satisfied:
        for feature in changed:
            if feature in blocked:
                violations.append({"type": "blocked_field", "feature": feature, "from": factual_profile[feature], "to": candidate_profile[feature]})

    actionable_features_satisfied = not any(feature not in set(context.policy.f2change) for feature in changed_set)
    if not actionable_features_satisfied:
        for feature in changed:
            if feature not in set(context.policy.f2change):
                violations.append({"type": "actionability", "feature": feature, "from": factual_profile[feature], "to": candidate_profile[feature]})

    max_change_satisfied = True if max_changes is None else len(changed) <= max_changes
    if not max_change_satisfied:
        violations.append({"type": "max_changed_features", "value": len(changed), "bound": f"<={max_changes}"})

    all_constraints_satisfied = (
        within_continuous_bounds
        and categorical_domain_satisfied
        and binary_domain_satisfied
        and blocked_fields_satisfied
        and numeric_bounds_satisfied
        and max_change_satisfied
        and actionable_features_satisfied
    )
    return {
        "within_continuous_bounds": within_continuous_bounds,
        "categorical_domain_satisfied": categorical_domain_satisfied,
        "binary_domain_satisfied": binary_domain_satisfied,
        "blocked_fields_satisfied": blocked_fields_satisfied,
        "numeric_bounds_satisfied": numeric_bounds_satisfied,
        "max_change_satisfied": max_change_satisfied,
        "actionable_features_satisfied": actionable_features_satisfied,
        "all_constraints_satisfied": all_constraints_satisfied,
        "violated_constraints": violations,
        "changed_features": changed,
        "changed_values": changed_values,
        "change_count": len(changed),
    }


def validate_witness_json_reload(
    *,
    witness_profile: dict[str, Any],
    persisted_score: dict[str, Any],
    persisted_audit: dict[str, Any],
    factual_profile: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    context,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    model_input_audit: dict[str, Any],
    atol: float,
) -> dict[str, bool]:
    reloaded = ordered_profile(json.loads(json_dumps_compact(witness_profile)), context)
    rescored = score_lr_profile(reloaded, context, model_input_audit=model_input_audit)
    reaudit = audit_oracle_candidate(
        factual_profile=factual_profile,
        candidate_profile=reloaded,
        constraint_spec=constraint_spec,
        context=context,
        feature_domains=feature_domains,
        feature_ranges=feature_ranges,
        atol=atol,
    )
    score_matches = (
        abs(float(rescored["margin"]) - float(persisted_score["margin"])) <= atol
        and abs(float(rescored["probability"]) - float(persisted_score["probability"])) <= atol
        and int(rescored["predicted_label"]) == int(persisted_score["predicted_label"])
    )
    audit_keys = [
        "within_continuous_bounds",
        "categorical_domain_satisfied",
        "binary_domain_satisfied",
        "blocked_fields_satisfied",
        "numeric_bounds_satisfied",
        "max_change_satisfied",
        "actionable_features_satisfied",
        "all_constraints_satisfied",
    ]
    audit_matches = all(bool(reaudit[key]) == bool(persisted_audit[key]) for key in audit_keys)
    return {"score_matches": score_matches, "audit_matches": audit_matches}


def oracle_label_satisfied(score_payload: dict[str, Any] | None, desired_label: int, context, *, atol: float) -> bool:
    if not score_payload:
        return False
    classes = [int(item) for item in list(context.bundle.lr.classes_)]
    positive_class = int(classes[-1])
    margin = float(score_payload["margin"])
    margin_satisfied = margin >= 0.0 - atol if desired_label == positive_class else margin <= 0.0 + atol
    return int(score_payload["predicted_label"]) == int(desired_label) and margin_satisfied


def oracle_result(
    *,
    feasible: bool | None,
    margin: float | None,
    changed: list[str],
    witness: dict[str, Any] | None,
    best_scored_profile: dict[str, Any] | None,
    best_scored_score: dict[str, Any] | None,
    best_scored_audit: dict[str, Any] | None,
    selected_subset: list[str],
    evaluated_subsets: set[tuple[str, ...]],
    candidate_count_scored: int,
    context,
    factual_profile: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    max_changed_features: Any,
    allowed_changeable_features: list[str],
    blocked_features: list[str],
    effective_changeable_features: list[str],
    desired_label: int,
    status: str,
    error: str,
    reload_check: dict[str, bool] | None = None,
) -> dict[str, Any]:
    best_scored_score = best_scored_score or {}
    best_scored_audit = best_scored_audit or {}
    witness_score = best_scored_score if witness is not None else None
    witness_audit = best_scored_audit if witness is not None else {}
    label_satisfied = oracle_label_satisfied(witness_score, desired_label, context, atol=ORACLE_ATOL) if witness is not None else False
    best_scored_label_satisfied = oracle_label_satisfied(best_scored_score, desired_label, context, atol=ORACLE_ATOL) if best_scored_score else False
    all_constraints_satisfied = bool(witness_audit.get("all_constraints_satisfied")) if witness is not None else False
    is_valid_witness = bool(witness is not None and feasible is True and label_satisfied and all_constraints_satisfied)
    max_change_satisfied = bool(best_scored_audit.get("max_change_satisfied")) if best_scored_audit else None
    max_changed = int(max_changed_features) if isinstance(max_changed_features, int) else None
    selected_subset_satisfies_max_change = True if max_changed is None else len(selected_subset) <= max_changed
    reason_for_infeasible = ""
    if feasible is False:
        if best_scored_profile is None:
            reason_for_infeasible = error or "no scored profile"
        elif not best_scored_label_satisfied:
            reason_for_infeasible = "best scored profile did not reach desired LR label"
        elif not bool(best_scored_audit.get("all_constraints_satisfied")):
            reason_for_infeasible = "best scored profile failed encoded constraints"
        else:
            reason_for_infeasible = "oracle status infeasible"
    return {
        "status": status,
        "feasible": feasible,
        "best_margin": margin,
        "changed_features": changed,
        "witness": witness,
        "witness_model_input": (witness_score or {}).get("model_input_profile"),
        "witness_score": witness_score,
        "witness_changed_features": list(witness_audit.get("changed_features") or []),
        "witness_changed_values": dict(witness_audit.get("changed_values") or {}),
        "witness_change_count": witness_audit.get("change_count"),
        "best_scored_profile": best_scored_profile,
        "best_scored_model_input": best_scored_score.get("model_input_profile"),
        "best_scored_score": best_scored_score,
        "best_scored_changed_features": list(best_scored_audit.get("changed_features") or []),
        "best_scored_change_count": best_scored_audit.get("change_count"),
        "best_scored_constraints_satisfied": best_scored_audit.get("all_constraints_satisfied"),
        "best_scored_reason_for_infeasible": reason_for_infeasible,
        "best_scored_is_valid_witness": is_valid_witness if feasible is True else False,
        "is_valid_oracle_witness": is_valid_witness,
        "label_satisfied": label_satisfied,
        "constraint_audit": witness_audit if witness is not None else best_scored_audit,
        "max_changed_features": max_changed,
        "max_change_satisfied": max_change_satisfied,
        "allowed_changeable_features": allowed_changeable_features,
        "blocked_features": blocked_features,
        "effective_changeable_features": effective_changeable_features,
        "subset_count_evaluated": len(evaluated_subsets),
        "candidate_count_scored": candidate_count_scored,
        "selected_subset": selected_subset,
        "selected_subset_satisfies_max_change": selected_subset_satisfies_max_change,
        "desired_label": desired_label,
        "search_completed": True,
        "reloaded_witness_score_matches": False if reload_check is None else reload_check["score_matches"],
        "reloaded_witness_audit_matches": False if reload_check is None else reload_check["audit_matches"],
        "error": error,
    }


def constrained_numeric_range(
    feature: str,
    numeric_bounds: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    low, high = feature_ranges[feature]
    bounds = numeric_bounds.get(feature)
    if isinstance(bounds, dict):
        if "min" in bounds:
            low = max(low, float(bounds["min"]))
        if "max" in bounds:
            high = min(high, float(bounds["max"]))
    return float(low), float(high)


def satisfies_bound(
    feature: str,
    value: Any,
    numeric_bounds: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
) -> bool:
    if feature in feature_ranges:
        low, high = constrained_numeric_range(feature, numeric_bounds, feature_ranges)
        return float(value) >= low - 1e-9 and float(value) <= high + 1e-9
    return True


def coerce_feature_value(feature: str, value: Any, context) -> Any:
    feature_type = context.policy.feature_type_map[feature]
    if feature_type == "float":
        return float(value)
    return int(value)


def same_value(left: Any, right: Any, feature_type: str, *, atol: float = 1e-9) -> bool:
    if feature_type == "float":
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    return int(left) == int(right)


def changed_features(
    factual: dict[str, Any],
    candidate: dict[str, Any],
    feature_type_map: dict[str, str],
    *,
    atol: float = 1e-9,
) -> list[str]:
    return [feature for feature in feature_type_map if not same_value(candidate[feature], factual[feature], feature_type_map[feature], atol=atol)]


def empirical_witness(
    *,
    factual_profile: dict[str, Any],
    constraint_spec: dict[str, Any] | None,
    context,
    feature_domains: dict[str, list[Any]],
    feature_ranges: dict[str, tuple[float, float]],
    model_input_audit: dict[str, Any],
) -> dict[str, Any]:
    desired = int(context.policy.desired_outcome)
    train_y = context.bundle.y.iloc[context.bundle.train_indices].reset_index(drop=True)
    train_x = context.bundle.Xtrain.reset_index(drop=True)
    train_indices = list(context.bundle.train_indices)
    pool = train_x.loc[train_y == desired, list(context.bundle.feature_order)]
    best_distance = None
    best_changed: list[str] = []
    best_row_id = None
    best_profile = None
    best_predicted_label = None
    best_constraints_satisfied = False
    for row_index, row in pool.iterrows():
        candidate = {feature: coerce_feature_value(feature, row[feature], context) for feature in context.bundle.feature_order}
        audit = audit_oracle_candidate(
            factual_profile=factual_profile,
            candidate_profile=candidate,
            constraint_spec=constraint_spec,
            context=context,
            feature_domains=feature_domains,
            feature_ranges=feature_ranges,
            atol=ORACLE_ATOL,
        )
        changed = audit["changed_features"]
        if not candidate_satisfies_policy_and_constraints(
            factual_profile=factual_profile,
            candidate=candidate,
            changed=changed,
            constraint_spec=constraint_spec,
            context=context,
            feature_ranges=feature_ranges,
        ):
            continue
        distance = normalized_proximity(factual_profile, candidate, feature_ranges)
        if distance is not None and (best_distance is None or distance < best_distance):
            best_distance = distance
            best_changed = changed
            best_row_id = int(train_indices[int(row_index)]) if int(row_index) < len(train_indices) else int(row_index)
            best_profile = candidate
            best_predicted_label = score_lr_profile(candidate, context, model_input_audit=model_input_audit)["predicted_label"]
            best_constraints_satisfied = bool(audit["all_constraints_satisfied"])
    return {
        "found": best_distance is not None,
        "row_id": best_row_id,
        "profile": best_profile,
        "distance": best_distance,
        "changed_features": best_changed,
        "predicted_label": best_predicted_label,
        "constraints_satisfied": best_constraints_satisfied,
    }


def candidate_satisfies_policy_and_constraints(
    *,
    factual_profile: dict[str, Any],
    candidate: dict[str, Any],
    changed: list[str],
    constraint_spec: dict[str, Any] | None,
    context,
    feature_ranges: dict[str, tuple[float, float]],
) -> bool:
    if any(feature not in set(context.policy.f2change) for feature in changed):
        return False
    feature_order = list(context.bundle.feature_order)
    blocked = set(effective_blocked_fields(constraint_spec, feature_order=feature_order))
    if any(feature in blocked for feature in changed):
        return False
    if isinstance(constraint_spec, dict) and isinstance(constraint_spec.get("max_changed_features"), int):
        if len(changed) > int(constraint_spec["max_changed_features"]):
            return False
    numeric_bounds = constraint_spec.get("numeric_bounds") if isinstance(constraint_spec, dict) and isinstance(constraint_spec.get("numeric_bounds"), dict) else {}
    for feature, value in candidate.items():
        if feature in feature_ranges and not satisfies_bound(feature, value, numeric_bounds, feature_ranges):
            return False
    return True


def build_final_subtypes(
    *,
    empty_cases: list[dict[str, Any]],
    phase_b_rows: list[dict[str, Any]],
    phase_c_rows: list[dict[str, Any]],
    oracle_skipped: bool,
) -> list[dict[str, Any]]:
    b_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in phase_b_rows:
        b_by_case.setdefault(str(row["case_id"]), {})[str(row["config_id"])] = row
    c_by_case = {str(row["case_id"]): row for row in phase_c_rows}
    final_rows: list[dict[str, Any]] = []
    for case in empty_cases:
        case_id = str(case["case_id"])
        configs = b_by_case.get(case_id, {})
        found_configs = [config_id for config_id in ("B_C1", "B_C2", "B_C3") if (configs.get(config_id) or {}).get("outcome") == "counterfactual_found"]
        config_detail = classify_config_limited(found_configs)
        oracle = c_by_case.get(case_id)
        if found_configs:
            final_subtype = "config_limited"
        elif oracle_skipped or oracle is None or oracle.get("oracle_status") == "error":
            final_subtype = "inconclusive"
        elif as_bool(oracle.get("oracle_feasible")):
            final_subtype = "oracle_feasible_but_ufce_missed"
        elif oracle.get("oracle_feasible") is False:
            final_subtype = "likely_infeasible_under_formalized_constraints"
        else:
            final_subtype = "inconclusive"
        final_rows.append(
            {
                **base_case_columns(case),
                "final_subtype": final_subtype,
                "config_limited_detail": config_detail,
                "found_config_ids": ",".join(found_configs),
                "oracle_status": "" if oracle is None else oracle.get("oracle_status"),
                "oracle_feasible": "" if oracle is None else oracle.get("oracle_feasible"),
                "oracle_witness_persisted": "" if oracle is None else oracle.get("oracle_witness_persisted"),
                "oracle_witness_lr_margin": "" if oracle is None else oracle.get("oracle_witness_lr_margin"),
                "oracle_witness_lr_probability": "" if oracle is None else oracle.get("oracle_witness_lr_probability"),
                "oracle_witness_predicted_label": "" if oracle is None else oracle.get("oracle_witness_predicted_label"),
                "oracle_witness_change_count": "" if oracle is None else oracle.get("oracle_witness_change_count"),
                "oracle_max_changed_features": "" if oracle is None else oracle.get("oracle_max_changed_features"),
                "oracle_max_change_satisfied": "" if oracle is None else oracle.get("oracle_max_change_satisfied"),
                "oracle_all_constraints_satisfied": "" if oracle is None else oracle.get("oracle_all_constraints_satisfied"),
                "oracle_witness_changed_features_json": "" if oracle is None else oracle.get("oracle_witness_changed_features_json"),
                "oracle_witness_profile_json": "" if oracle is None else oracle.get("oracle_witness_profile_json"),
                "oracle_best_margin": "" if oracle is None else oracle.get("oracle_best_margin"),
                "oracle_best_scored_margin": "" if oracle is None else oracle.get("oracle_best_scored_margin"),
                "oracle_subset_count_evaluated": "" if oracle is None else oracle.get("oracle_subset_count_evaluated"),
                "empirical_witness_found": "" if oracle is None else oracle.get("empirical_witness_found"),
                "empirical_witness_distance": "" if oracle is None else oracle.get("empirical_witness_distance"),
            }
        )
    return final_rows


def classify_config_limited(found_configs: list[str]) -> str:
    found = set(found_configs)
    if "B_C2" in found:
        return "radius_limited"
    if "B_C1" in found:
        return "neighbor_filter_sensitive"
    if "B_C3" in found:
        return "combined_config_limited"
    return ""


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def build_thesis_tables(
    *,
    phase_a_rows: list[dict[str, Any]],
    phase_b_rows: list[dict[str, Any]],
    phase_c_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    post_filter_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    table4x = summarize_phase_a(phase_a_rows, post_filter_rows)
    table4y = summarize_phase_b(phase_b_rows)
    table4z = summarize_phase_c(phase_c_rows, final_rows)
    return {
        "table4x_constraint_sanity.csv": table4x,
        "table4y_runtime_config_sensitivity.csv": table4y,
        "table4z_feasibility_upper_bound.csv": table4z,
    }


def summarize_phase_a(phase_a_rows: list[dict[str, Any]], post_filter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({row["constraint_group"] for row in phase_a_rows})
    rows = []
    for group in groups:
        group_rows = [row for row in phase_a_rows if row["constraint_group"] == group and row["mode"] == "relaxed"]
        rows.append(
            {
                "constraint_group": group,
                "g2_empty_candidate_cases": len(group_rows),
                "relaxed_counterfactual_found": sum(1 for row in group_rows if row["outcome"] == "counterfactual_found"),
                "relaxed_still_no_valid_cf": sum(1 for row in group_rows if row["outcome"] == "no_valid_cf"),
                "runtime_semantics_violation_or_hidden_coupling": sum(
                    1 for row in group_rows if row["invariant_status"] != "expected_invariant_holds"
                ),
            }
        )
    rows.append(
        {
            "constraint_group": "post_filter_constraint_blocked_sanity_case",
            "g2_empty_candidate_cases": 0,
            "relaxed_counterfactual_found": sum(1 for row in post_filter_rows if row["mode"] == "relaxed" and row["outcome"] == "counterfactual_found"),
            "relaxed_still_no_valid_cf": sum(1 for row in post_filter_rows if row["mode"] == "relaxed" and row["outcome"] == "no_valid_cf"),
            "runtime_semantics_violation_or_hidden_coupling": 0,
        }
    )
    return rows


def summarize_phase_b(phase_b_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in sorted({row["group"] for row in phase_b_rows}):
        group_rows = [row for row in phase_b_rows if row["group"] == group]
        baseline_cases = {row["case_id"] for row in group_rows if row["config_id"] == "B_C0"}
        for config_id in sorted({row["config_id"] for row in group_rows}):
            config_rows = [row for row in group_rows if row["config_id"] == config_id]
            rows.append(
                {
                    "group": group,
                    "config_id": config_id,
                    "baseline_no_valid_cf_cases": len(baseline_cases),
                    "counterfactual_found": sum(1 for row in config_rows if row["outcome"] == "counterfactual_found"),
                    "still_no_valid_cf": sum(1 for row in config_rows if row["outcome"] == "no_valid_cf"),
                    "runtime_reject_other": sum(1 for row in config_rows if row["outcome"] not in {"counterfactual_found", "no_valid_cf"}),
                }
            )
    return rows


def summarize_phase_c(phase_c_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in sorted({row["group"] for row in final_rows}):
        group_c = [row for row in phase_c_rows if row["group"] == group]
        group_final = [row for row in final_rows if row["group"] == group]
        rows.append(
            {
                "group": group,
                "empty_candidate_no_valid_cf": len(group_final),
                "oracle_feasible": sum(1 for row in group_c if as_bool(row.get("oracle_feasible"))),
                "oracle_infeasible": sum(1 for row in group_c if row.get("oracle_feasible") is False),
                "empirical_witness_found": sum(1 for row in group_c if as_bool(row.get("empirical_witness_found"))),
                "config_limited": sum(1 for row in group_final if row["final_subtype"] == "config_limited"),
                "oracle_feasible_but_ufce_missed": sum(1 for row in group_final if row["final_subtype"] == "oracle_feasible_but_ufce_missed"),
                "likely_infeasible_under_formalized_constraints": sum(
                    1 for row in group_final if row["final_subtype"] == "likely_infeasible_under_formalized_constraints"
                ),
                "inconclusive": sum(1 for row in group_final if row["final_subtype"] == "inconclusive"),
            }
        )
    return rows


def build_provenance(
    *,
    args: argparse.Namespace,
    run_id: str,
    source_report_sha256: str,
    parity: dict[str, Any],
    selected_config_ids: list[str],
    context,
    model_input_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "git_commit": try_get_git_commit(),
        "primary_source_report": str(Path(args.source_report).resolve()),
        "primary_source_report_sha256": source_report_sha256,
        "primary_source_report_expected_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "tier_b_corpus": str(Path(args.tier_b_corpus).resolve()),
        "tier_b_corpus_sha256": parity["corpus_sha256"],
        "selected_runtime_config_ids": list(selected_config_ids),
        "runtime_configs": {config_id: dict(RUNTIME_CONFIGS[config_id]) for config_id in selected_config_ids},
        "ufce_fixed_config": dict(UFCE_FIXED_CONFIG),
        "bank_feature_bounds_source": "training-set feature bounds under frozen split",
        "categorical_domain_source": "canonical Bank schema domains: Family=1..4, Education=1..3, binary fields=0/1",
        "continuous_box_definition": "closed interval [min(Xtrain[feature]), max(Xtrain[feature])] intersected with active numeric_bounds",
        "policy_actionability_source": "BankDatasetPackage policy f2change from legacy get_bank_user_constraints",
        "oracle_bounds_source": "training-set feature bounds under frozen split",
        "oracle_domain_source": "canonical Bank schema domains: Family=1..4, Education=1..3, binary fields=0/1",
        "oracle_actionability_source": "BankDatasetPackage policy f2change from legacy get_bank_user_constraints",
        "oracle_constraint_source": "active_constraint_spec_expected from Tier-B corpus joined to authoritative 203759 metric rows",
        "oracle_max_change_semantics": "count changed raw features after applying atol comparison",
        "oracle_numeric_bound_semantics": "candidate raw feature values must satisfy active numeric_bounds after intersecting oracle search boxes with training-set bounds",
        "oracle_blocked_field_semantics": "effective blocked fields are computed by effective_blocked_fields and may not appear in changed raw features",
        "oracle_changed_feature_atol": ORACLE_ATOL,
        "oracle_numeric_equality_atol": ORACLE_ATOL,
        "oracle_model_input_transform": model_input_audit["oracle_model_input_transform"],
        "oracle_model_input_feature_order": list(model_input_audit["model_input_feature_order"]),
        "oracle_lr_coef_width": model_input_audit["lr_coef_width"],
        "desired_label": int(context.policy.desired_outcome),
        "feature_order": list(context.bundle.feature_order),
        "changeable_features": list(context.policy.f2change),
        "case_limit": args.case_limit,
        "oracle_skipped": bool(args.skip_oracle),
    }


def build_oracle_witness_audit_summary(
    *,
    phase_c_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    oracle_skipped: bool,
) -> dict[str, Any]:
    if oracle_skipped:
        return {"skipped": True}
    feasible_rows = [row for row in phase_c_rows if as_bool(row.get("oracle_feasible"))]
    infeasible_rows = [row for row in phase_c_rows if row.get("oracle_feasible") is False]
    max_rows = [row for row in phase_c_rows if not is_blank(row.get("oracle_max_changed_features"))]
    feasible_max_rows = [row for row in feasible_rows if not is_blank(row.get("oracle_max_changed_features"))]
    final_counts = Counter(row["final_subtype"] for row in final_rows)
    return {
        "skipped": False,
        "oracle_feasible_cases": len(feasible_rows),
        "oracle_infeasible_cases": len(infeasible_rows),
        "feasible_witness_persisted": sum(1 for row in feasible_rows if as_bool(row.get("oracle_witness_persisted"))),
        "feasible_label_satisfied": sum(1 for row in feasible_rows if as_bool(row.get("oracle_label_satisfied"))),
        "feasible_bounds_domain_constraints_satisfied": sum(1 for row in feasible_rows if as_bool(row.get("oracle_all_constraints_satisfied"))),
        "feasible_max_change_applicable": len(feasible_max_rows),
        "feasible_max_change_satisfied": sum(1 for row in feasible_max_rows if as_bool(row.get("oracle_max_change_satisfied"))),
        "infeasible_best_scored_trace_persisted": sum(1 for row in infeasible_rows if not is_blank(row.get("oracle_best_scored_profile_json"))),
        "feasible_reloaded_score_matches": sum(1 for row in feasible_rows if as_bool(row.get("oracle_reloaded_witness_score_matches"))),
        "feasible_reloaded_audit_matches": sum(1 for row in feasible_rows if as_bool(row.get("oracle_reloaded_witness_audit_matches"))),
        "max_changed_features_cases": len(max_rows),
        "max_changed_features_oracle_feasible": sum(1 for row in max_rows if as_bool(row.get("oracle_feasible"))),
        "max_changed_features_oracle_infeasible": sum(1 for row in max_rows if row.get("oracle_feasible") is False),
        "final_subtype_counts": dict(sorted(final_counts.items())),
        "infeasible_best_scored_note": "Best scored profile for infeasible cases is diagnostic only, not a valid counterfactual witness.",
    }


def validate_oracle_audit(
    *,
    phase_c_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    oracle_witness_audit: dict[str, Any],
    full_run: bool,
) -> None:
    if oracle_witness_audit.get("skipped"):
        return
    feasible_rows = [row for row in phase_c_rows if as_bool(row.get("oracle_feasible"))]
    infeasible_rows = [row for row in phase_c_rows if row.get("oracle_feasible") is False]
    errors: list[str] = []
    for row in feasible_rows:
        case_id = row["case_id"]
        if not as_bool(row.get("oracle_witness_persisted")) or is_blank(row.get("oracle_witness_profile_json")):
            errors.append(f"{case_id}: feasible oracle row missing persisted witness")
        if not as_bool(row.get("oracle_label_satisfied")):
            errors.append(f"{case_id}: feasible witness does not satisfy desired LR label")
        if not as_bool(row.get("oracle_all_constraints_satisfied")):
            errors.append(f"{case_id}: feasible witness does not satisfy encoded constraints")
        if not is_blank(row.get("oracle_max_changed_features")) and not as_bool(row.get("oracle_max_change_satisfied")):
            errors.append(f"{case_id}: feasible witness violates max_changed_features")
        if not as_bool(row.get("oracle_reloaded_witness_score_matches")):
            errors.append(f"{case_id}: feasible witness JSON reload changed LR score")
        if not as_bool(row.get("oracle_reloaded_witness_audit_matches")):
            errors.append(f"{case_id}: feasible witness JSON reload changed constraint audit")
    for row in infeasible_rows:
        case_id = row["case_id"]
        if not as_bool(row.get("oracle_search_completed")):
            errors.append(f"{case_id}: infeasible oracle row did not complete search")
        if is_blank(row.get("oracle_best_scored_margin")):
            errors.append(f"{case_id}: infeasible oracle row missing best scored margin")
        if is_blank(row.get("oracle_best_scored_profile_json")):
            errors.append(f"{case_id}: infeasible oracle row missing best scored trace")
        if as_bool(row.get("is_valid_oracle_witness")):
            errors.append(f"{case_id}: infeasible oracle row marked as valid witness")
    if oracle_witness_audit["feasible_witness_persisted"] != oracle_witness_audit["oracle_feasible_cases"]:
        errors.append("feasible witness JSONL count would not match oracle feasible count")
    if oracle_witness_audit["infeasible_best_scored_trace_persisted"] != oracle_witness_audit["oracle_infeasible_cases"]:
        errors.append("infeasible best-scored trace count would not match oracle infeasible count")
    if full_run:
        final_counts = Counter(row["final_subtype"] for row in final_rows)
        for subtype, expected in EXPECTED_FULL_FINAL_SUBTYPE_COUNTS.items():
            if final_counts.get(subtype, 0) != expected:
                errors.append(f"full-run subtype count changed for {subtype}: expected {expected}, got {final_counts.get(subtype, 0)}")
    if errors:
        raise RuntimeError("Oracle witness audit validation failed: " + "; ".join(errors))


def is_blank(value: Any) -> bool:
    return value is None or value == ""


def build_summary(
    *,
    run_id: str,
    parity: dict[str, Any],
    semantics: dict[str, Any],
    phase_a_rows: list[dict[str, Any]],
    phase_b_rows: list[dict[str, Any]],
    phase_c_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    post_filter_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    oracle_witness_audit: dict[str, Any],
    oracle_skipped: bool,
) -> dict[str, Any]:
    final_counts = Counter(row["final_subtype"] for row in final_rows)
    return {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "parity_gate": parity,
        "runtime_semantics_audit": semantics,
        "phase_a": {
            "empty_candidate_rows": len(phase_a_rows),
            "hidden_coupling_findings": sum(1 for row in phase_a_rows if row["finding"] == "hidden_coupling_finding"),
            "post_filter_sanity_rows": len(post_filter_rows),
        },
        "phase_b": {
            "rows": len(phase_b_rows),
            "counterfactual_found": sum(1 for row in phase_b_rows if row["outcome"] == "counterfactual_found"),
        },
        "phase_c": {
            "skipped": bool(oracle_skipped),
            "rows": len(phase_c_rows),
            "oracle_feasible": sum(1 for row in phase_c_rows if as_bool(row.get("oracle_feasible"))),
            "oracle_infeasible": sum(1 for row in phase_c_rows if row.get("oracle_feasible") is False),
        },
        "final_subtype_counts": dict(sorted(final_counts.items())),
        "oracle_witness_audit": oracle_witness_audit,
        "claim_boundary": (
            "The single post-filter constraint-blocked case is reported separately from the 140 empty-candidate "
            "no_valid_cf cohort. Constraint relaxation changing empty-candidate outcomes is treated as hidden coupling, "
            "not direct constraint-induced failure."
        ),
        "provenance": provenance,
    }


def write_outputs(
    *,
    run_root: Path,
    summary: dict[str, Any],
    provenance: dict[str, Any],
    phase_a_rows: list[dict[str, Any]],
    post_filter_rows: list[dict[str, Any]],
    phase_b_rows: list[dict[str, Any]],
    phase_c_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
    oracle_witness_audit: dict[str, Any],
) -> None:
    write_json(run_root / "diagnostic_report.json", summary)
    write_json(run_root / "provenance.json", provenance)
    (run_root / "diagnostic_report.md").write_text(render_diagnostic_markdown(summary), encoding="utf-8")
    write_csv(run_root / "experiment_a_post_filter_constraint_sanity.csv", phase_a_rows)
    write_csv(run_root / "post_filter_constraint_blocked_sanity_case.csv", post_filter_rows)
    write_csv(run_root / "experiment_b_runtime_config_sensitivity.csv", phase_b_rows)
    write_csv(run_root / "experiment_c_feasibility_upper_bound.csv", phase_c_rows)
    write_csv(run_root / "final_no_valid_cf_subtypes.csv", final_rows)
    write_jsonl(run_root / "oracle_witnesses.jsonl", build_oracle_witness_jsonl_rows(phase_c_rows, final_rows))
    write_jsonl(run_root / "oracle_infeasible_best_attempts.jsonl", build_oracle_infeasible_jsonl_rows(phase_c_rows, final_rows))
    write_csv(run_root / "oracle_constraint_audit.csv", build_oracle_constraint_audit_rows(phase_c_rows, final_rows))
    (run_root / "oracle_representative_cases.md").write_text(
        render_oracle_representative_cases(phase_c_rows=phase_c_rows, phase_b_rows=phase_b_rows, final_rows=final_rows),
        encoding="utf-8",
    )
    for filename, rows in tables.items():
        write_csv(run_root / filename, rows)
    missing = [name for name in PRIMARY_OUTPUTS if not (run_root / name).exists()]
    if missing:
        raise RuntimeError("Required outputs missing: " + ", ".join(missing))


def build_oracle_witness_jsonl_rows(phase_c_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_case = {row["case_id"]: row for row in final_rows}
    rows = []
    for row in phase_c_rows:
        if not as_bool(row.get("oracle_feasible")):
            continue
        rows.append(
            {
                "case_id": row["case_id"],
                "session_id": row["session_id"],
                "group": row["group"],
                "constraint_group": row["constraint_group"],
                "final_subtype": (final_by_case.get(row["case_id"]) or {}).get("final_subtype", ""),
                "original_profile_json": parse_json_field(row.get("original_profile_json")),
                "oracle_witness_profile_json": parse_json_field(row.get("oracle_witness_profile_json")),
                "oracle_witness_model_input_json": parse_json_field(row.get("oracle_witness_model_input_json")),
                "oracle_witness_changed_features_json": parse_json_field(row.get("oracle_witness_changed_features_json")),
                "oracle_witness_changed_values_json": parse_json_field(row.get("oracle_witness_changed_values_json")),
                "oracle_witness_change_count": row.get("oracle_witness_change_count"),
                "oracle_witness_lr_margin": row.get("oracle_witness_lr_margin"),
                "oracle_witness_lr_probability": row.get("oracle_witness_lr_probability"),
                "oracle_witness_predicted_label": row.get("oracle_witness_predicted_label"),
                "desired_label": row.get("desired_label"),
                "oracle_label_satisfied": row.get("oracle_label_satisfied"),
                "oracle_atol": row.get("oracle_atol"),
                "oracle_all_constraints_satisfied": row.get("oracle_all_constraints_satisfied"),
                "oracle_violated_constraints_json": parse_json_field(row.get("oracle_violated_constraints_json")),
                "oracle_max_changed_features": row.get("oracle_max_changed_features"),
                "oracle_max_change_satisfied": row.get("oracle_max_change_satisfied"),
                "oracle_allowed_changeable_features_json": parse_json_field(row.get("oracle_allowed_changeable_features_json")),
                "oracle_blocked_features_json": parse_json_field(row.get("oracle_blocked_features_json")),
                "oracle_effective_changeable_features_json": parse_json_field(row.get("oracle_effective_changeable_features_json")),
                "oracle_subset_count_evaluated": row.get("oracle_subset_count_evaluated"),
                "oracle_candidate_count_scored": row.get("oracle_candidate_count_scored"),
                "oracle_selected_subset_json": parse_json_field(row.get("oracle_selected_subset_json")),
                "oracle_selected_subset_size": row.get("oracle_selected_subset_size"),
                "oracle_selected_subset_satisfies_max_change": row.get("oracle_selected_subset_satisfies_max_change"),
                "oracle_reloaded_witness_score_matches": row.get("oracle_reloaded_witness_score_matches"),
                "oracle_reloaded_witness_audit_matches": row.get("oracle_reloaded_witness_audit_matches"),
                "is_valid_oracle_witness": row.get("is_valid_oracle_witness"),
            }
        )
    return rows


def build_oracle_infeasible_jsonl_rows(phase_c_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_case = {row["case_id"]: row for row in final_rows}
    rows = []
    for row in phase_c_rows:
        if row.get("oracle_feasible") is not False:
            continue
        rows.append(
            {
                "case_id": row["case_id"],
                "session_id": row["session_id"],
                "group": row["group"],
                "constraint_group": row["constraint_group"],
                "final_subtype": (final_by_case.get(row["case_id"]) or {}).get("final_subtype", ""),
                "oracle_status": row.get("oracle_status"),
                "oracle_best_scored_profile_json": parse_json_field(row.get("oracle_best_scored_profile_json")),
                "oracle_best_scored_margin": row.get("oracle_best_scored_margin"),
                "oracle_best_scored_probability": row.get("oracle_best_scored_probability"),
                "oracle_best_scored_predicted_label": row.get("oracle_best_scored_predicted_label"),
                "oracle_best_scored_changed_features_json": parse_json_field(row.get("oracle_best_scored_changed_features_json")),
                "oracle_best_scored_change_count": row.get("oracle_best_scored_change_count"),
                "oracle_best_scored_constraints_satisfied": row.get("oracle_best_scored_constraints_satisfied"),
                "oracle_best_scored_reason_for_infeasible": row.get("oracle_best_scored_reason_for_infeasible"),
                "oracle_subset_count_evaluated": row.get("oracle_subset_count_evaluated"),
                "oracle_candidate_count_scored": row.get("oracle_candidate_count_scored"),
                "oracle_search_completed": row.get("oracle_search_completed"),
                "oracle_best_scored_is_valid_witness": False,
                "is_valid_oracle_witness": False,
                "note": "Best scored profile for infeasible cases is diagnostic only, not a valid counterfactual witness.",
            }
        )
    return rows


def build_oracle_constraint_audit_rows(phase_c_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_case = {row["case_id"]: row for row in final_rows}
    fields = [
        "oracle_within_continuous_bounds",
        "oracle_categorical_domain_satisfied",
        "oracle_binary_domain_satisfied",
        "oracle_blocked_fields_satisfied",
        "oracle_numeric_bounds_satisfied",
        "oracle_max_change_satisfied",
        "oracle_actionable_features_satisfied",
        "oracle_all_constraints_satisfied",
    ]
    rows = []
    for row in phase_c_rows:
        rows.append(
            {
                "case_id": row["case_id"],
                "session_id": row["session_id"],
                "group": row["group"],
                "constraint_group": row["constraint_group"],
                "final_subtype": (final_by_case.get(row["case_id"]) or {}).get("final_subtype", ""),
                "oracle_feasible": row.get("oracle_feasible"),
                "oracle_status": row.get("oracle_status"),
                "oracle_witness_persisted": row.get("oracle_witness_persisted"),
                **{field: row.get(field) for field in fields},
                "oracle_violated_constraints_json": row.get("oracle_violated_constraints_json"),
                "oracle_witness_change_count": row.get("oracle_witness_change_count"),
                "oracle_max_changed_features": row.get("oracle_max_changed_features"),
                "oracle_selected_subset_json": row.get("oracle_selected_subset_json"),
                "oracle_subset_count_evaluated": row.get("oracle_subset_count_evaluated"),
                "oracle_candidate_count_scored": row.get("oracle_candidate_count_scored"),
            }
        )
    return rows


def parse_json_field(value: Any) -> Any:
    if is_blank(value):
        return None
    return json.loads(str(value))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps_compact(row) + "\n")


def render_oracle_representative_cases(
    *,
    phase_c_rows: list[dict[str, Any]],
    phase_b_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> str:
    final_by_case = {row["case_id"]: row for row in final_rows}
    b0_by_case = {row["case_id"]: row for row in phase_b_rows if row.get("config_id") == "B_C0"}
    categories = [
        ("G1 oracle-feasible but UFCE-missed", lambda row: row["group"] == "G1" and as_bool(row.get("oracle_feasible"))),
        (
            "G2 oracle-feasible with max_changed_features",
            lambda row: row["group"] == "G2" and as_bool(row.get("oracle_feasible")) and not is_blank(row.get("oracle_max_changed_features")),
        ),
        (
            "G2 oracle-feasible with blocked feature",
            lambda row: row["group"] == "G2" and as_bool(row.get("oracle_feasible")) and parse_json_field(row.get("oracle_blocked_features_json")),
        ),
        ("G2 likely infeasible", lambda row: row["group"] == "G2" and row.get("oracle_feasible") is False),
    ]
    lines = [
        "# Oracle Representative Cases",
        "",
        "Best scored profile for infeasible cases is diagnostic only, not a valid counterfactual witness.",
        "",
    ]
    used: set[str] = set()
    for title, predicate in categories:
        row = next((candidate for candidate in phase_c_rows if candidate["case_id"] not in used and predicate(candidate)), None)
        if row is None:
            lines.extend([f"## {title}", "", "Not available in this run.", ""])
            continue
        used.add(row["case_id"])
        final = final_by_case.get(row["case_id"]) or {}
        b0 = b0_by_case.get(row["case_id"]) or {}
        witness_or_best = row.get("oracle_witness_profile_json") or row.get("oracle_best_scored_profile_json")
        lines.extend(
            [
                f"## Case: {row['case_id']}",
                "",
                f"Purpose: {title}",
                "",
                "Original profile:",
                fenced_json(parse_json_field(row.get("original_profile_json"))),
                "",
                "Constraints:",
                fenced_json(parse_json_field(row.get("constraint_spec")) or {}),
                "",
                f"UFCE runtime result: {b0.get('outcome', 'no_valid_cf')} / empty_candidate, generated_candidate_count = {b0.get('generated_candidate_count', 0)}",
                "",
                f"Final subtype: {final.get('final_subtype', '')}",
                f"Oracle result: {row.get('oracle_status')}",
                "",
                "Oracle witness/best-scored profile:",
                fenced_json(parse_json_field(witness_or_best)),
                "",
                f"Changed fields: `{row.get('oracle_witness_changed_features_json') or row.get('oracle_best_scored_changed_features_json')}`",
                f"LR margin/probability/predicted label: `{row.get('oracle_witness_lr_margin') or row.get('oracle_best_scored_margin')}` / `{row.get('oracle_witness_lr_probability') or row.get('oracle_best_scored_probability')}` / `{row.get('oracle_witness_predicted_label') or row.get('oracle_best_scored_predicted_label')}`",
                "",
                "Constraint audit:",
                f"- bounds: {'pass' if as_bool(row.get('oracle_within_continuous_bounds')) else 'fail'}",
                f"- categorical/binary domain: {'pass' if as_bool(row.get('oracle_categorical_domain_satisfied')) and as_bool(row.get('oracle_binary_domain_satisfied')) else 'fail'}",
                f"- blocked fields: {'pass' if as_bool(row.get('oracle_blocked_fields_satisfied')) else 'fail'}",
                f"- numeric bounds: {'pass' if as_bool(row.get('oracle_numeric_bounds_satisfied')) else 'fail'}",
                f"- max_changed_features: {'pass' if as_bool(row.get('oracle_max_change_satisfied')) else 'not applicable/fail'}",
                "",
                "Interpretation: oracle evidence is recorded independently from UFCE candidate generation for this empty-candidate no_valid_cf case.",
                "",
            ]
        )
    return "\n".join(lines)


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n```"


def render_diagnostic_markdown(summary: dict[str, Any]) -> str:
    counts = summary["final_subtype_counts"]
    oracle_audit = summary.get("oracle_witness_audit") or {}
    lines = [
        "# No-Valid-CF Diagnostics",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- parity_gate_ok: `{summary['parity_gate']['ok']}`",
        f"- runtime_semantics_ok: `{summary['runtime_semantics_audit']['semantics_ok']}`",
        f"- Phase A hidden coupling findings: `{summary['phase_a']['hidden_coupling_findings']}`",
        f"- Phase B counterfactual_found rows: `{summary['phase_b']['counterfactual_found']}`",
        f"- Phase C oracle skipped: `{summary['phase_c']['skipped']}`",
        "",
        "## Final Subtypes",
    ]
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]}")
    if not oracle_audit.get("skipped"):
        feasible = int(oracle_audit.get("oracle_feasible_cases") or 0)
        infeasible = int(oracle_audit.get("oracle_infeasible_cases") or 0)
        max_applicable = int(oracle_audit.get("feasible_max_change_applicable") or 0)
        max_pass = int(oracle_audit.get("feasible_max_change_satisfied") or 0)
        lines.extend(
            [
                "",
                "## Oracle Witness Audit Summary",
                "",
                f"- Oracle feasible cases: `{feasible}`",
                f"- Feasible cases with persisted witness profile: `{oracle_audit.get('feasible_witness_persisted')}/{feasible}`",
                f"- Feasible cases satisfying LR desired label: `{oracle_audit.get('feasible_label_satisfied')}/{feasible}`",
                f"- Feasible cases satisfying bounds/domain constraints: `{oracle_audit.get('feasible_bounds_domain_constraints_satisfied')}/{feasible}`",
                f"- Feasible cases satisfying max_changed_features where applicable: `{max_pass}/{max_applicable}`",
                f"- Infeasible cases with persisted best-scored trace: `{oracle_audit.get('infeasible_best_scored_trace_persisted')}/{infeasible}`",
                f"- Cases with max_changed_features constraint: `{oracle_audit.get('max_changed_features_cases')}`",
                f"- Oracle-feasible among max_changed_features cases: `{oracle_audit.get('max_changed_features_oracle_feasible')}`",
                f"- Oracle-infeasible among max_changed_features cases: `{oracle_audit.get('max_changed_features_oracle_infeasible')}`",
                "",
                oracle_audit.get("infeasible_best_scored_note", ""),
            ]
        )
    lines.extend(["", "## Claim Boundary", summary["claim_boundary"], ""])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
