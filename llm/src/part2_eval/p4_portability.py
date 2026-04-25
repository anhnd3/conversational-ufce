from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.src.part2_eval.common import (
    build_in_process_service,
    build_session_detail_payload,
    call_with_legacy_stdout_redirect,
    prepare_run_layout,
)
from llm.src.product.config import ProductConfig
from llm.src.product.service import (
    ProductSessionService,
    RefinementLimitReachedError,
    RefinementNotAllowedError,
    SessionArchivedError,
)
from llm.src.refinement.types import (
    REFINEMENT_STATUS_APPLIED,
    REFINEMENT_STATUS_CLARIFICATION_REQUIRED,
    REFINEMENT_STATUS_LIMIT_REACHED,
    REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK,
)
from llm.src.runtime.backend_packages.registry_defaults import build_default_backends
from llm.src.runtime.contracts import CanonicalProfile
from llm.src.runtime.datasets import BankDatasetPackage, GradDatasetPackage
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.registries.backend_registry import BackendRegistry
from llm.src.runtime.registries.dataset_registry import DatasetRegistry
from llm.src.utils.time import local_now_compact, local_now_iso


P4A_REQUIRED_MATRIX = (("bank", "ufce"), ("grad", "ufce"), ("bank", "dice"))
P4A_OPTIONAL_MATRIX = (("grad", "dice"),)
P4B_PRIMARY_MATRIX = tuple((dataset_id, backend_id) for dataset_id in ("bank", "grad") for backend_id in ("ufce", "dice", "ar"))
P4B_REFINEMENT_MATRIX = P4B_PRIMARY_MATRIX
PRIMARY_FAILURE_STAGES = (
    "session_binding",
    "parser",
    "canonical_validation",
    "backend_execution",
    "verification",
    "ranking",
    "explanation",
    "product_flow",
)
REFINEMENT_FAILURE_STAGES = PRIMARY_FAILURE_STAGES + (
    "refinement_parser",
    "refinement_application",
)
RUNTIME_BACKED_STATES = {"RUNTIME_SUCCESS", "RUNTIME_REJECT"}

PRIMARY_SMOKE_CASES = {
    "bank": (
        {
            "case_id": "complete_request",
            "turns": (
                "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
                "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no."
            ,),
            "expectation": "runtime_backed",
        },
        {
            "case_id": "clarification_flow",
            "turns": (
                "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32.",
                "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no.",
            ),
            "expectation": "runtime_backed",
        },
        {
            "case_id": "cross_dataset_lock",
            "turns": ("Switch this case to the graduate admission dataset and use GRE 320 instead.",),
            "expectation": "unsupported_request",
        },
    ),
    "grad": (
        {
            "case_id": "complete_request",
            "turns": (
                "GRE Score 320, TOEFL Score 110, University Rating 4, SOP 4.5, "
                "LOR 4.0, CGPA 8.9, Research yes."
            ,),
            "expectation": "runtime_backed",
        },
        {
            "case_id": "clarification_flow",
            "turns": (
                "GRE Score 320, TOEFL Score 110, University Rating 4.",
                "SOP 4.5, LOR 4.0, CGPA 8.9, Research yes.",
            ),
            "expectation": "runtime_backed",
        },
        {
            "case_id": "cross_dataset_lock",
            "turns": ("Switch this case to the bank dataset and use Income 140 instead.",),
            "expectation": "unsupported_request",
        },
    ),
}

REFINEMENT_SMOKE_CASES = {
    "bank": (
        {
            "case_id": "applied_refinement",
            "seed_turns": PRIMARY_SMOKE_CASES["bank"][0]["turns"],
            "feedbacks": ("Keep Income above 100 and limit changes to two features.",),
            "expected_refinement_status": REFINEMENT_STATUS_APPLIED,
        },
        {
            "case_id": "clarification_required_refinement",
            "seed_turns": PRIMARY_SMOKE_CASES["bank"][0]["turns"],
            "feedbacks": ("Do not change Income, actually Income can change.",),
            "expected_refinement_status": REFINEMENT_STATUS_CLARIFICATION_REQUIRED,
        },
        {
            "case_id": "unsupported_refinement",
            "seed_turns": PRIMARY_SMOKE_CASES["bank"][0]["turns"],
            "feedbacks": ("Write a poem about this bank customer instead.",),
            "expected_refinement_status": REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK,
        },
        {
            "case_id": "round_limit",
            "seed_turns": PRIMARY_SMOKE_CASES["bank"][0]["turns"],
            "feedbacks": (
                "Make it better.",
                "Still make it better.",
                "Make it even better.",
                "One more improvement please.",
            ),
            "expected_refinement_status": REFINEMENT_STATUS_LIMIT_REACHED,
        },
    ),
    "grad": (
        {
            "case_id": "applied_refinement",
            "seed_turns": PRIMARY_SMOKE_CASES["grad"][0]["turns"],
            "feedbacks": ("Keep CGPA above 8.5 and limit changes to one feature.",),
            "expected_refinement_status": REFINEMENT_STATUS_APPLIED,
        },
        {
            "case_id": "clarification_required_refinement",
            "seed_turns": PRIMARY_SMOKE_CASES["grad"][0]["turns"],
            "feedbacks": ("Do not change CGPA, actually CGPA can change.",),
            "expected_refinement_status": REFINEMENT_STATUS_CLARIFICATION_REQUIRED,
        },
        {
            "case_id": "unsupported_refinement",
            "seed_turns": PRIMARY_SMOKE_CASES["grad"][0]["turns"],
            "feedbacks": ("Write a poem about this graduate admission applicant instead.",),
            "expected_refinement_status": REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK,
        },
        {
            "case_id": "round_limit",
            "seed_turns": PRIMARY_SMOKE_CASES["grad"][0]["turns"],
            "feedbacks": (
                "Make it better.",
                "Still make it better.",
                "Make it even better.",
                "One more improvement please.",
            ),
            "expected_refinement_status": REFINEMENT_STATUS_LIMIT_REACHED,
        },
    ),
}


def build_dataset_registry(model_registry: ModelRegistry | None = None) -> DatasetRegistry:
    active_model_registry = model_registry or ModelRegistry()
    return DatasetRegistry(
        {
            "bank": BankDatasetPackage(active_model_registry),
            "grad": GradDatasetPackage(active_model_registry),
        }
    )


def build_backend_registry() -> BackendRegistry:
    return BackendRegistry(backends=build_default_backends())


def summarize_dataset_conformance(
    dataset_ids: list[str] | tuple[str, ...],
    *,
    dataset_registry: DatasetRegistry | None = None,
) -> dict[str, Any]:
    registry = dataset_registry or build_dataset_registry()
    summary: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        normalized = str(dataset_id).strip().lower()
        if not registry.has(normalized):
            summary[normalized] = {
                "dataset_id": normalized,
                "pass": False,
                "errors": [f"Unsupported dataset package: {dataset_id}"],
                "checks": {},
            }
            continue
        package = registry.get(normalized)
        feature_order = list(package.profile_schema()["field_order"])
        full_profile = _build_sample_profile(package)
        missing_profile = dict(full_profile)
        if feature_order:
            missing_profile.pop(feature_order[-1], None)
        missing_result = package.validate_profile(
            CanonicalProfile(dataset_id=normalized, values=missing_profile),
        )
        numeric_bound_fields = list(package.numeric_bound_fields())
        hard_constraints = (
            {"numeric_bounds": {numeric_bound_fields[0]: {"min": float(full_profile[numeric_bound_fields[0]])}}}
            if numeric_bound_fields
            else {}
        )
        hard_constraint_result = package.validate_profile(
            CanonicalProfile(dataset_id=normalized, values=full_profile),
            hard_constraints=hard_constraints,
        )
        explanation_templates = package.explanation_templates()
        checks = {
            "feature_schema_present": bool(package.feature_schema()),
            "aliases_present": bool(package.aliases()),
            "field_order_present": bool(feature_order),
            "full_profile_validation": bool(
                package.validate_profile(CanonicalProfile(dataset_id=normalized, values=full_profile)).ok
            ),
            "missing_field_behavior": bool((not missing_result.ok) and missing_result.missing_fields),
            "hard_constraint_behavior": bool(hard_constraint_result.ok),
            "explanation_templates_present": all(
                key in explanation_templates
                for key in ("summary_no_recourse", "summary_counterfactual", "summary_reject")
            ),
            "numeric_bound_fields_present": bool(numeric_bound_fields),
            "live_runtime_enabled": bool(package.compatibility_manifest().live_runtime_enabled),
        }
        summary[normalized] = {
            "dataset_id": normalized,
            "pass": all(checks.values()),
            "errors": [],
            "supported_backends": list(package.compatibility_manifest().supported_backends),
            "checks": checks,
        }
    return summary


def summarize_backend_conformance(
    backend_ids: list[str] | tuple[str, ...],
    *,
    backend_registry: BackendRegistry | None = None,
) -> dict[str, Any]:
    registry = backend_registry or build_backend_registry()
    summary: dict[str, Any] = {}
    for backend_id in backend_ids:
        normalized = str(backend_id).strip().lower()
        if not registry.has(normalized):
            manifest = None
            try:
                manifest = registry.manifest(normalized)
            except KeyError:
                manifest = None
            summary[normalized] = {
                "backend_id": normalized,
                "pass": False,
                "errors": [f"Unsupported or disabled backend: {backend_id}"],
                "checks": {
                    "manifest_present": manifest is not None,
                    "enabled": False,
                },
            }
            continue
        backend = registry.get(normalized)
        manifest = registry.manifest(normalized)
        checks = {
            "manifest_present": manifest is not None,
            "enabled": bool(manifest.enabled),
            "request_contract_v1": manifest.request_contract_version == "canonical_request_v1",
            "candidate_contract_v1": manifest.candidate_contract_version == "canonical_candidate_v1",
            "backend_id_match": getattr(backend, "backend_id", None) == normalized,
            "generate_callable": callable(getattr(backend, "generate", None)),
        }
        summary[normalized] = {
            "backend_id": normalized,
            "pass": all(checks.values()),
            "errors": [],
            "manifest": manifest.to_dict(),
            "checks": checks,
        }
    return summary


def compute_enabled_combinations(
    dataset_ids: list[str] | tuple[str, ...],
    backend_ids: list[str] | tuple[str, ...],
    *,
    dataset_registry: DatasetRegistry | None = None,
    backend_registry: BackendRegistry | None = None,
) -> list[dict[str, Any]]:
    datasets = dataset_registry or build_dataset_registry()
    backends = backend_registry or build_backend_registry()
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        normalized_dataset = str(dataset_id).strip().lower()
        if not datasets.has(normalized_dataset):
            continue
        package = datasets.get(normalized_dataset)
        manifest = package.compatibility_manifest()
        supported_backends = set(manifest.supported_backends)
        for backend_id in backend_ids:
            normalized_backend = str(backend_id).strip().lower()
            backend_enabled = backends.has(normalized_backend)
            dataset_enabled = bool(manifest.live_runtime_enabled)
            supported = normalized_backend in supported_backends
            rows.append(
                {
                    "combination": f"{normalized_dataset}+{normalized_backend}",
                    "dataset_id": normalized_dataset,
                    "backend_id": normalized_backend,
                    "dataset_runtime_enabled": dataset_enabled,
                    "backend_enabled": backend_enabled,
                    "supported_by_dataset_manifest": supported,
                    "enabled": bool(dataset_enabled and backend_enabled and supported),
                }
            )
    return rows


def classify_primary_failure_stage(
    *,
    public_state: str | None,
    builder_status: str | None = None,
    runtime_executed: bool = False,
) -> str:
    if public_state == "PARSER_FAILURE":
        return "parser"
    if public_state in {"NEEDS_CLARIFICATION", "CONFLICT"} or builder_status in {"NEEDS_CLARIFICATION", "CONFLICT"}:
        return "canonical_validation"
    if builder_status == "READY_FOR_RUNTIME" and not runtime_executed:
        return "backend_execution"
    if public_state == "UNSUPPORTED_REQUEST":
        return "product_flow"
    if public_state in RUNTIME_BACKED_STATES:
        return "explanation"
    return "product_flow"


def classify_refinement_failure_stage(
    *,
    public_state: str | None,
    refinement_status: str | None,
    runtime_executed: bool = False,
    limit_reached: bool = False,
) -> str:
    if limit_reached or refinement_status == REFINEMENT_STATUS_LIMIT_REACHED:
        return "refinement_application"
    if refinement_status in {REFINEMENT_STATUS_CLARIFICATION_REQUIRED, REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK}:
        return "refinement_parser"
    if public_state == "PARSER_FAILURE":
        return "refinement_parser"
    if refinement_status == REFINEMENT_STATUS_APPLIED and not runtime_executed:
        return "backend_execution"
    if public_state in RUNTIME_BACKED_STATES:
        return "explanation"
    return "refinement_application"


def execute_primary_matrix(
    *,
    run_root: Path,
    dataset_ids: list[str],
    backend_ids: list[str],
    benchmark_path: Path,
    lm_studio_api_base: str,
    model_alias: str,
    product_mode: str,
    api_version: str,
    app_version: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for backend_id in backend_ids:
        layout = prepare_run_layout(out_dir=run_root / "backend_runs", run_id=backend_id)
        handle = build_in_process_service(
            layout=layout,
            lm_studio_api_base=lm_studio_api_base,
            model_alias=model_alias,
            product_mode=product_mode,
            api_version=api_version,
            app_version=app_version,
            benchmark_path=benchmark_path,
            counterfactual_backend_name=backend_id,
        )
        for dataset_id in dataset_ids:
            combo_key = f"{dataset_id}+{backend_id}"
            case_rows = []
            for case in PRIMARY_SMOKE_CASES[dataset_id]:
                case_rows.append(execute_primary_case(handle=handle, dataset_id=dataset_id, backend_id=backend_id, case=case))
            results[combo_key] = {
                "dataset_id": dataset_id,
                "backend_id": backend_id,
                "pass": all(row["pass"] for row in case_rows),
                "cases": case_rows,
                "artifact_root": str(handle.artifact_root),
                "sqlite_path": str(handle.sqlite_path),
            }
    return results


def execute_refinement_matrix(
    *,
    run_root: Path,
    dataset_ids: list[str],
    backend_ids: list[str],
    benchmark_path: Path,
    lm_studio_api_base: str,
    model_alias: str,
    product_mode: str,
    api_version: str,
    app_version: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for backend_id in backend_ids:
        layout = prepare_run_layout(out_dir=run_root / "backend_runs", run_id=f"{backend_id}_refinement")
        handle = build_in_process_service(
            layout=layout,
            lm_studio_api_base=lm_studio_api_base,
            model_alias=model_alias,
            product_mode=product_mode,
            api_version=api_version,
            app_version=app_version,
            benchmark_path=benchmark_path,
            counterfactual_backend_name=backend_id,
        )
        for dataset_id in dataset_ids:
            combo_key = f"{dataset_id}+{backend_id}"
            case_rows = []
            for case in REFINEMENT_SMOKE_CASES[dataset_id]:
                case_rows.append(
                    execute_refinement_case(handle=handle, dataset_id=dataset_id, backend_id=backend_id, case=case)
                )
            results[combo_key] = {
                "dataset_id": dataset_id,
                "backend_id": backend_id,
                "pass": all(row["pass"] for row in case_rows),
                "cases": case_rows,
                "artifact_root": str(handle.artifact_root),
                "sqlite_path": str(handle.sqlite_path),
            }
    return results


def execute_primary_case(*, handle, dataset_id: str, backend_id: str, case: dict[str, Any]) -> dict[str, Any]:
    session_id = None
    turn_payloads: list[dict[str, Any]] = []
    try:
        created = call_with_legacy_stdout_redirect(handle.service.create_session, dataset_key=dataset_id)
        session_id = str(created.session_id)
        for turn_text in case["turns"]:
            stored_turn = call_with_legacy_stdout_redirect(handle.service.submit_message, session_id, turn_text)
            turn_payloads.append(handle.service.build_turn_response(stored_turn))
        session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
    except Exception as exc:
        session_detail = (
            build_session_detail_payload(handle.repository.get_session(session_id))
            if session_id is not None
            else None
        )
        return _failed_case_result(
            dataset_id=dataset_id,
            backend_id=backend_id,
            case_id=case["case_id"],
            flow="primary",
            failure_stage="session_binding" if session_id is None else "product_flow",
            session_id=session_id,
            turn_payloads=turn_payloads,
            session_detail=session_detail,
            error=str(exc),
        )

    final_turn = turn_payloads[-1] if turn_payloads else None
    if _primary_expectation_met(case["expectation"], final_turn):
        return {
            "dataset_id": dataset_id,
            "backend_id": backend_id,
            "case_id": case["case_id"],
            "flow": "primary",
            "pass": True,
            "failure_stage": None,
            "session_id": session_id,
            "turn_count": len(turn_payloads),
            "final_public_state": None if final_turn is None else final_turn.get("public_state"),
            "turn_payloads": turn_payloads,
            "session_detail": session_detail,
            "error": None,
        }

    debug_summary = {} if final_turn is None else dict(final_turn.get("debug_summary") or {})
    failure_stage = classify_primary_failure_stage(
        public_state=None if final_turn is None else final_turn.get("public_state"),
        builder_status=debug_summary.get("builder_status"),
        runtime_executed=bool(((debug_summary.get("runtime_summary") or {}).get("executed"))),
    )
    return _failed_case_result(
        dataset_id=dataset_id,
        backend_id=backend_id,
        case_id=case["case_id"],
        flow="primary",
        failure_stage=failure_stage,
        session_id=session_id,
        turn_payloads=turn_payloads,
        session_detail=session_detail,
        error=f"Unexpected final public_state for expectation={case['expectation']}",
    )


def execute_refinement_case(*, handle, dataset_id: str, backend_id: str, case: dict[str, Any]) -> dict[str, Any]:
    session_id = None
    turn_payloads: list[dict[str, Any]] = []
    refinement_payloads: list[dict[str, Any]] = []
    try:
        created = call_with_legacy_stdout_redirect(handle.service.create_session, dataset_key=dataset_id)
        session_id = str(created.session_id)
        for turn_text in case["seed_turns"]:
            stored_turn = call_with_legacy_stdout_redirect(handle.service.submit_message, session_id, turn_text)
            turn_payloads.append(handle.service.build_turn_response(stored_turn))
        if not turn_payloads or turn_payloads[-1].get("public_state") not in RUNTIME_BACKED_STATES:
            session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
            return _failed_case_result(
                dataset_id=dataset_id,
                backend_id=backend_id,
                case_id=case["case_id"],
                flow="refinement",
                failure_stage="backend_execution",
                session_id=session_id,
                turn_payloads=turn_payloads,
                session_detail=session_detail,
                error="Primary seed did not reach a runtime-backed state required for refinement.",
            )
        for feedback in case["feedbacks"]:
            stored_turn = call_with_legacy_stdout_redirect(handle.service.submit_refinement, session_id, feedback)
            refinement_payloads.append(handle.service.build_turn_response(stored_turn))
        session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
    except RefinementLimitReachedError as exc:
        session_detail = (
            build_session_detail_payload(handle.repository.get_session(session_id))
            if session_id is not None
            else None
        )
        expected = case.get("expected_refinement_status")
        passed = expected == REFINEMENT_STATUS_LIMIT_REACHED
        return {
            "dataset_id": dataset_id,
            "backend_id": backend_id,
            "case_id": case["case_id"],
            "flow": "refinement",
            "pass": passed,
            "failure_stage": None if passed else "refinement_application",
            "session_id": session_id,
            "turn_count": len(turn_payloads) + len(refinement_payloads),
            "final_public_state": None if not refinement_payloads else refinement_payloads[-1].get("public_state"),
            "turn_payloads": turn_payloads,
            "refinement_payloads": refinement_payloads,
            "session_detail": session_detail,
            "error": None if passed else str(exc),
            "refinement_status": REFINEMENT_STATUS_LIMIT_REACHED,
        }
    except (RefinementNotAllowedError, SessionArchivedError) as exc:
        session_detail = (
            build_session_detail_payload(handle.repository.get_session(session_id))
            if session_id is not None
            else None
        )
        return _failed_case_result(
            dataset_id=dataset_id,
            backend_id=backend_id,
            case_id=case["case_id"],
            flow="refinement",
            failure_stage="refinement_application",
            session_id=session_id,
            turn_payloads=turn_payloads,
            session_detail=session_detail,
            error=str(exc),
            refinement_payloads=refinement_payloads,
        )
    except Exception as exc:
        session_detail = (
            build_session_detail_payload(handle.repository.get_session(session_id))
            if session_id is not None
            else None
        )
        return _failed_case_result(
            dataset_id=dataset_id,
            backend_id=backend_id,
            case_id=case["case_id"],
            flow="refinement",
            failure_stage="session_binding" if session_id is None else "product_flow",
            session_id=session_id,
            turn_payloads=turn_payloads,
            session_detail=session_detail,
            error=str(exc),
            refinement_payloads=refinement_payloads,
        )

    final_turn = refinement_payloads[-1] if refinement_payloads else None
    expected_status = case["expected_refinement_status"]
    actual_status = None if final_turn is None else final_turn.get("refinement_status")
    if actual_status == expected_status:
        return {
            "dataset_id": dataset_id,
            "backend_id": backend_id,
            "case_id": case["case_id"],
            "flow": "refinement",
            "pass": True,
            "failure_stage": None,
            "session_id": session_id,
            "turn_count": len(turn_payloads) + len(refinement_payloads),
            "final_public_state": None if final_turn is None else final_turn.get("public_state"),
            "turn_payloads": turn_payloads,
            "refinement_payloads": refinement_payloads,
            "session_detail": session_detail,
            "error": None,
            "refinement_status": actual_status,
        }

    debug_summary = {} if final_turn is None else dict(final_turn.get("debug_summary") or {})
    failure_stage = classify_refinement_failure_stage(
        public_state=None if final_turn is None else final_turn.get("public_state"),
        refinement_status=actual_status,
        runtime_executed=bool(((debug_summary.get("runtime_summary") or {}).get("executed"))),
    )
    return _failed_case_result(
        dataset_id=dataset_id,
        backend_id=backend_id,
        case_id=case["case_id"],
        flow="refinement",
        failure_stage=failure_stage,
        session_id=session_id,
        turn_payloads=turn_payloads,
        session_detail=session_detail,
        error=f"Unexpected refinement_status={actual_status!r}; expected={expected_status!r}",
        refinement_payloads=refinement_payloads,
    )


def build_portability_summary(
    *,
    milestone: str,
    run_root: Path,
    command: str,
    datasets: list[str],
    backends: list[str],
    required_primary_matrix: list[tuple[str, str]],
    optional_primary_matrix: list[tuple[str, str]] | None = None,
    required_refinement_matrix: list[tuple[str, str]] | None = None,
    dataset_conformance: dict[str, Any],
    backend_conformance: dict[str, Any],
    enabled_combinations: list[dict[str, Any]],
    primary_results: dict[str, Any],
    refinement_results: dict[str, Any] | None,
) -> dict[str, Any]:
    primary_required_ok = all(primary_results.get(f"{dataset}+{backend}", {}).get("pass") for dataset, backend in required_primary_matrix)
    refinement_required_ok = None
    if required_refinement_matrix is not None and refinement_results is not None:
        refinement_required_ok = all(
            refinement_results.get(f"{dataset}+{backend}", {}).get("pass")
            for dataset, backend in required_refinement_matrix
        )
    return {
        "runner_scope": f"part2_{milestone.lower()}_portability",
        "report_version": "part2_p4_portability_report_v1",
        "milestone": milestone,
        "generated_at": local_now_iso(),
        "run_id": f"{milestone.lower()}_portability_{local_now_compact()}",
        "run_root": str(Path(run_root).resolve()),
        "command": command,
        "datasets": list(datasets),
        "backends": list(backends),
        "required_primary_matrix": [f"{dataset}+{backend}" for dataset, backend in required_primary_matrix],
        "optional_primary_matrix": []
        if optional_primary_matrix is None
        else [f"{dataset}+{backend}" for dataset, backend in optional_primary_matrix],
        "required_refinement_matrix": []
        if required_refinement_matrix is None
        else [f"{dataset}+{backend}" for dataset, backend in required_refinement_matrix],
        "dataset_conformance": dataset_conformance,
        "backend_conformance": backend_conformance,
        "enabled_combinations": enabled_combinations,
        "primary_results": primary_results,
        "refinement_results": refinement_results,
        "baseline_status": {
            "protected_combination": "bank+ufce",
            "bank_validation_rerun": "not_run",
            "aggregate_validation_ok": None,
            "script_mismatch_count": None,
        },
        "failure_taxonomy": list(REFINEMENT_FAILURE_STAGES if refinement_results is not None else PRIMARY_FAILURE_STAGES),
        "all_dataset_conformance_passed": all(item.get("pass") for item in dataset_conformance.values()),
        "all_backend_conformance_passed": all(item.get("pass") for item in backend_conformance.values()),
        "all_required_primary_passed": primary_required_ok,
        "all_required_refinement_passed": refinement_required_ok,
    }


def render_portability_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['milestone']} Portability Report",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Datasets: `{', '.join(summary['datasets'])}`",
        f"- Backends: `{', '.join(summary['backends'])}`",
        f"- Required primary matrix passed: `{summary['all_required_primary_passed']}`",
    ]
    if summary.get("all_required_refinement_passed") is not None:
        lines.append(f"- Required refinement matrix passed: `{summary['all_required_refinement_passed']}`")
    lines.extend(
        [
            "",
            "## Dataset Conformance",
            "",
            "| Dataset | Pass | Supported Backends |",
            "| --- | --- | --- |",
        ]
    )
    for dataset_id, row in summary["dataset_conformance"].items():
        lines.append(
            f"| {dataset_id} | {row['pass']} | {', '.join(row.get('supported_backends', [])) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Backend Conformance",
            "",
            "| Backend | Pass | Request Contract | Candidate Contract |",
            "| --- | --- | --- | --- |",
        ]
    )
    for backend_id, row in summary["backend_conformance"].items():
        manifest = row.get("manifest") or {}
        lines.append(
            f"| {backend_id} | {row['pass']} | {manifest.get('request_contract_version', '-')} | "
            f"{manifest.get('candidate_contract_version', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Enabled Combinations",
            "",
            "| Combination | Enabled |",
            "| --- | --- |",
        ]
    )
    for row in summary["enabled_combinations"]:
        lines.append(f"| {row['combination']} | {row['enabled']} |")
    lines.extend(_render_result_block("Primary Smoke", summary["primary_results"]))
    if summary.get("refinement_results") is not None:
        lines.extend(_render_result_block("Refinement Smoke", summary["refinement_results"]))
    return "\n".join(lines) + "\n"


def parse_csv_list(raw_value: str) -> list[str]:
    items = [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def default_benchmark_path() -> Path:
    return Path(__file__).resolve().parents[3] / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml"


def default_product_config() -> ProductConfig:
    return ProductConfig.load()


def _build_sample_profile(package) -> dict[str, Any]:
    feature_order = list(package.profile_schema()["field_order"])
    feature_schema = package.feature_schema()
    row = package.load_model_bundle().dataset_df.iloc[0].to_dict()
    profile: dict[str, Any] = {}
    for field_name in feature_order:
        value = row[field_name]
        field_type = str(feature_schema[field_name]["type"])
        if field_type == "float":
            profile[field_name] = float(value)
        else:
            profile[field_name] = int(value)
    return profile


def _primary_expectation_met(expectation: str, final_turn: dict[str, Any] | None) -> bool:
    if final_turn is None:
        return False
    public_state = final_turn.get("public_state")
    if expectation == "runtime_backed":
        return public_state in RUNTIME_BACKED_STATES
    if expectation == "unsupported_request":
        return public_state == "UNSUPPORTED_REQUEST"
    return False


def _failed_case_result(
    *,
    dataset_id: str,
    backend_id: str,
    case_id: str,
    flow: str,
    failure_stage: str,
    session_id: str | None,
    turn_payloads: list[dict[str, Any]],
    session_detail: dict[str, Any] | None,
    error: str,
    refinement_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "backend_id": backend_id,
        "case_id": case_id,
        "flow": flow,
        "pass": False,
        "failure_stage": failure_stage,
        "session_id": session_id,
        "turn_count": len(turn_payloads) + len(refinement_payloads or []),
        "final_public_state": None
        if not (refinement_payloads or turn_payloads)
        else (refinement_payloads or turn_payloads)[-1].get("public_state"),
        "turn_payloads": turn_payloads,
        "refinement_payloads": refinement_payloads,
        "session_detail": session_detail,
        "error": error,
        "refinement_status": None
        if not refinement_payloads
        else refinement_payloads[-1].get("refinement_status"),
    }


def _render_result_block(title: str, results: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Combination | Pass | Failing Cases |",
        "| --- | --- | --- |",
    ]
    for combo_key, row in results.items():
        failing_cases = [
            f"{case['case_id']}:{case['failure_stage']}"
            for case in row.get("cases", [])
            if not case.get("pass")
        ]
        lines.append(f"| {combo_key} | {row['pass']} | {', '.join(failing_cases) or '-'} |")
    return lines
