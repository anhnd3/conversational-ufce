from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.bank_synth_corpus import load_bank_source_analysis
from llm.src.runtime.counterfactual_service import CounterfactualService, UFCECoreConfig
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.reason_codes import (
    CONSTRAINT_BLOCKED,
    GENERATION_ERROR,
    NO_CANDIDATE_GENERATED,
    NO_FEASIBLE_CF_FOUND,
    NO_VALID_FLIP,
    REQUEST_CONSTRAINTS_BLOCKED,
    UFCE_EXECUTION_ERROR,
)
from llm.src.runtime.ufce_request_builder import UFCERequestBuilder
from llm.src.utils.hashing import sha256_file


DEFAULT_OUT_DIR = ROOT / "outputs" / "policy_control"
SELECTION_SOURCE = "llm.src.part2_eval.bank_synth_corpus.load_bank_source_analysis.source_profiles"
FULL_RUN_DIFFICULTY_QUOTA = {"easy": 3, "medium": 4, "hard": 3}
S5_SCOPE_NOTE = "step_override_is_ufce1_scoped; UFCE2/UFCE3 do not consume step in the current core."
STANDARD_ALIGNMENT = [
    {
        "standard": "ISO/IEC/IEEE 29119-4:2021",
        "source": "https://www.iso.org/standard/79430.html",
        "use": "Scenario coverage, equivalence classes, and boundary/constraint-driven test design.",
        "claim_scope": "Aligned with test design principles; not a certification claim.",
    },
    {
        "standard": "NIST AI RMF 1.0",
        "source": "https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10",
        "use": "Valid/reliable, safe, accountable/transparent, and interpretable AI-system evidence goals.",
        "claim_scope": "Aligned with trustworthiness characteristics; not a user-study substitute.",
    },
]
REQUIRED_RUN_FIELDS = [
    "scenario_id",
    "mode",
    "profile_id",
    "base_policy",
    "policy_override",
    "effective_policy",
    "effective_mi_feature_pairs",
    "constraint_spec",
    "core_status",
    "presentation_status",
    "reason_code",
    "prediction_before",
    "prediction_after",
    "changed_fields",
    "policy_violation",
    "constraint_violation",
    "force_flip_invariant",
    "trace_summary",
    "difficulty_label",
    "expectation_id",
    "expectation_result",
    "mode_a_audit_only",
    "selection_source",
    "profile_signature",
]
REQUIRED_ARTIFACTS = [
    "policy_control_manifest.json",
    "policy_control_profiles.csv",
    "policy_control_expectations.json",
    "policy_control_runs.jsonl",
    "policy_control_summary.csv",
    "policy_control_by_scenario.csv",
    "policy_control_reason_codes.csv",
    "policy_control_changed_fields.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bank policy-control A/B/C evaluation.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--profiles-per-scenario", type=int, default=10)
    args = parser.parse_args()

    payload = run_policy_control_evaluation(
        out_dir=args.out_dir,
        profiles_per_scenario=args.profiles_per_scenario,
    )
    print(json.dumps(payload["manifest"], indent=2, sort_keys=True))


def run_policy_control_evaluation(*, out_dir: Path, profiles_per_scenario: int = 10) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    context = PolicyRegistry(registry).get_runtime_context("bank")
    orchestrator = RuntimeOrchestrator(model_registry=registry)
    scenarios = build_scenarios(feature_order=list(context.bundle.feature_order))
    calibration_records = calibrate_profile_pool(
        context=context,
        orchestrator=orchestrator,
        target_per_scenario=profiles_per_scenario,
    )
    selected_profiles = select_profiles_by_scenario(
        calibration_records=calibration_records,
        scenarios=scenarios,
        profiles_per_scenario=profiles_per_scenario,
    )
    expectations = build_expectations()

    runs: list[dict[str, Any]] = []
    scenario_by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}
    for selected in selected_profiles:
        scenario = scenario_by_id[selected["scenario_id"]]
        for mode in scenario["modes"]:
            runs.append(
                run_single_mode(
                    orchestrator=orchestrator,
                    context=context,
                    scenario=scenario,
                    mode=mode,
                    profile=dict(selected["profile"]),
                    profile_id=str(selected["profile_id"]),
                    profile_metadata=selected,
                )
            )

    expectation_payload = {
        "registered_expectations": expectations,
        "observed_expectation_summary": expectation_summary(runs),
        "standards_alignment": list(STANDARD_ALIGNMENT),
    }
    manifest = build_manifest(
        scenarios=scenarios,
        selected_profiles=selected_profiles,
        runs=runs,
        profiles_per_scenario=profiles_per_scenario,
        expectation_payload=expectation_payload,
    )
    validate_artifacts_payload(
        manifest=manifest,
        selected_profiles=selected_profiles,
        expectations=expectation_payload,
        runs=runs,
    )
    write_outputs(
        out_dir=out_dir,
        manifest=manifest,
        selected_profiles=selected_profiles,
        expectations=expectation_payload,
        runs=runs,
    )
    return {"manifest": manifest, "profiles": selected_profiles, "expectations": expectation_payload, "runs": runs}


def build_scenarios(*, feature_order: list[str]) -> list[dict[str, Any]]:
    immutable_except_income = [field for field in feature_order if field != "Income"]
    immutable_except_numeric = [field for field in feature_order if field not in {"Income", "CCAvg", "Mortgage"}]
    return [
        {
            "scenario_id": "S0",
            "description": "Baseline reference",
            "modes": ["A", "B", "C"],
            "constraint_spec": None,
            "policy_override": None,
            "mode_b_constraint_spec": None,
            "scope_note": "baseline_reference_not_counted_as_policy_scenario",
        },
        {
            "scenario_id": "S1",
            "description": "Mortgage immutable",
            "modes": ["A", "B", "C"],
            "constraint_spec": {"immutable": ["Mortgage"]},
            "policy_override": {"remove_from_f2change": ["Mortgage"]},
            "mode_b_constraint_spec": {"immutable": ["Mortgage"]},
            "scope_note": "immutable_field_policy_control",
        },
        {
            "scenario_id": "S2",
            "description": "Income max increase +15",
            "modes": ["A", "B", "C"],
            "constraint_spec": {"numeric_bounds_delta": {"Income": {"max_increase": 15}}},
            "policy_override": {"uf": {"Income": 15}},
            "mode_b_constraint_spec": {"numeric_bounds_delta": {"Income": {"max_increase": 15}}},
            "scope_note": "uf_generation_bound_control",
        },
        {
            "scenario_id": "S3",
            "description": "Only Income changeable",
            "modes": ["A", "B", "C"],
            "constraint_spec": {"max_changed_features": 1},
            "policy_override": {"f2change": ["Income"]},
            "mode_b_constraint_spec": {"immutable": immutable_except_income, "max_changed_features": 1},
            "scope_note": "narrow_f2change_policy_control",
        },
        {
            "scenario_id": "S4",
            "description": "Income + CCAvg + Mortgage changeable",
            "modes": ["A", "B", "C"],
            "constraint_spec": None,
            "policy_override": {"f2change": ["Income", "CCAvg", "Mortgage"]},
            "mode_b_constraint_spec": {"immutable": immutable_except_numeric},
            "scope_note": "wider_f2change_policy_control",
        },
        {
            "scenario_id": "S5",
            "description": "CCAvg step control",
            "modes": ["A", "C"],
            "constraint_spec": None,
            "policy_override": {"step": {"CCAvg": 0.5}},
            "mode_b_constraint_spec": None,
            "scope_note": S5_SCOPE_NOTE,
        },
        {
            "scenario_id": "S6",
            "description": "Tight combined constraint",
            "modes": ["A", "B", "C"],
            "constraint_spec": {
                "immutable": ["Mortgage", "CDAccount"],
                "numeric_bounds_delta": {"Income": {"max_increase": 15}},
            },
            "policy_override": {
                "remove_from_f2change": ["Mortgage", "CDAccount"],
                "uf": {"Income": 15},
            },
            "mode_b_constraint_spec": {
                "immutable": ["Mortgage", "CDAccount"],
                "numeric_bounds_delta": {"Income": {"max_increase": 15}},
            },
            "scope_note": "tight_combined_constraint_safe_stop",
        },
    ]


def calibrate_profile_pool(
    *,
    context,
    orchestrator: RuntimeOrchestrator,
    target_per_scenario: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    quota = difficulty_quota(target_per_scenario)
    scan_limit = max(target_per_scenario * 4, sum(quota.values()) * 3, 24)
    for source_index, profile in enumerate(load_bank_source_analysis()["source_profiles"]):
        if len(records) >= scan_limit:
            break
        prediction_before = predict_label(context=context, profile=profile)
        if prediction_before == int(context.policy.desired_outcome):
            continue
        result = orchestrator.handle({"dataset": "bank", "profile": dict(profile)}, include_debug_trace=True)
        candidate = first_candidate(result)
        changed_fields = list(candidate.changed_features) if candidate is not None else []
        candidate_profile = dict(candidate.profile) if candidate is not None else None
        core_status = result.debug_trace.core_status if result.debug_trace is not None else None
        difficulty_label = classify_difficulty(
            feasible=bool(result.counterfactual and result.counterfactual.feasible),
            changed_fields=changed_fields,
            factual_profile=profile,
            candidate_profile=candidate_profile,
            core_status=core_status,
        )
        records.append(
            {
                "source_profile_id": f"bank_source_{source_index:03d}",
                "source_index": source_index,
                "profile": dict(profile),
                "profile_signature": profile_signature(profile),
                "prediction_before": prediction_before,
                "baseline_core_status": core_status,
                "baseline_presentation_status": "presented"
                if result.counterfactual is not None and result.counterfactual.feasible
                else "rejected",
                "baseline_changed_fields": changed_fields,
                "baseline_candidate_profile": candidate_profile,
                "difficulty_label": difficulty_label,
            }
        )
    if not records:
        raise ValueError("No rejected Bank profiles were available for policy-control evaluation.")
    return records


def quota_satisfied(records: list[dict[str, Any]], quota: dict[str, int]) -> bool:
    counts = Counter(str(record["difficulty_label"]) for record in records)
    return all(counts.get(difficulty, 0) >= count for difficulty, count in quota.items())


def classify_difficulty(
    *,
    feasible: bool,
    changed_fields: list[str],
    factual_profile: dict[str, Any],
    candidate_profile: dict[str, Any] | None,
    core_status: str | None,
) -> str:
    if not feasible or candidate_profile is None or core_status != "valid_counterfactual_found":
        return "hard"
    income_delta = float(candidate_profile.get("Income", 0.0)) - float(factual_profile.get("Income", 0.0))
    if (
        "Mortgage" in changed_fields
        or income_delta > 15.0
        or len(changed_fields) > 1
    ):
        return "medium"
    return "easy"


def select_profiles_by_scenario(
    *,
    calibration_records: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    profiles_per_scenario: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    quota = difficulty_quota(profiles_per_scenario)

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        scenario_records: list[dict[str, Any]] = []
        used_source_ids: set[str] = set()
        scenario_calibration_records = []
        by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in calibration_records:
            scenario_record = dict(record)
            scenario_record["difficulty_label"] = scenario_difficulty_label(record, scenario)
            scenario_calibration_records.append(scenario_record)
            by_difficulty[str(scenario_record["difficulty_label"])].append(scenario_record)
        for difficulty, count in quota.items():
            candidates = by_difficulty.get(difficulty, [])
            offset = scenario_offset(scenario_id, len(candidates))
            for record in candidates[offset:] + candidates[:offset]:
                if record["source_profile_id"] in used_source_ids:
                    continue
                scenario_records.append(record)
                used_source_ids.add(record["source_profile_id"])
                if sum(1 for item in scenario_records if item["difficulty_label"] == difficulty) >= count:
                    break
        if len(scenario_records) < profiles_per_scenario:
            offset = scenario_offset(scenario_id, len(scenario_calibration_records))
            for record in scenario_calibration_records[offset:] + scenario_calibration_records[:offset]:
                if record["source_profile_id"] in used_source_ids:
                    continue
                scenario_records.append(record)
                used_source_ids.add(record["source_profile_id"])
                if len(scenario_records) >= profiles_per_scenario:
                    break
        if len(scenario_records) < profiles_per_scenario:
            raise ValueError(
                f"Scenario {scenario_id} expected {profiles_per_scenario} profiles, found {len(scenario_records)}."
            )
        scenario_records = scenario_records[:profiles_per_scenario]
        for index, record in enumerate(scenario_records):
            item = dict(record)
            item["scenario_id"] = scenario_id
            item["profile_id"] = f"{scenario_id}_profile_{index:02d}"
            item["selection_source"] = SELECTION_SOURCE
            selected.append(item)
    return selected


def scenario_difficulty_label(record: dict[str, Any], scenario: dict[str, Any]) -> str:
    if record.get("baseline_presentation_status") != "presented":
        return "hard"
    changed_fields = list(record.get("baseline_changed_fields") or [])
    candidate_profile = record.get("baseline_candidate_profile")
    if not isinstance(candidate_profile, dict):
        return "hard"
    if scenario_baseline_would_be_blocked(
        scenario=scenario,
        changed_fields=changed_fields,
        factual_profile=dict(record["profile"]),
        candidate_profile=candidate_profile,
    ):
        return "medium"
    return "easy"


def scenario_baseline_would_be_blocked(
    *,
    scenario: dict[str, Any],
    changed_fields: list[str],
    factual_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
) -> bool:
    if scenario["scenario_id"] == "S0":
        return False
    constraint_spec = scenario.get("mode_b_constraint_spec") or scenario.get("constraint_spec") or {}
    blocked_fields = set(constraint_spec.get("immutable") or [])
    if any(field in blocked_fields for field in changed_fields):
        return True
    max_changed_features = constraint_spec.get("max_changed_features")
    if isinstance(max_changed_features, int) and len(changed_fields) > max_changed_features:
        return True
    numeric_bounds_delta = constraint_spec.get("numeric_bounds_delta")
    if isinstance(numeric_bounds_delta, dict):
        for field_name, bounds in numeric_bounds_delta.items():
            if not isinstance(bounds, dict) or "max_increase" not in bounds:
                continue
            delta = float(candidate_profile.get(field_name, 0.0)) - float(factual_profile.get(field_name, 0.0))
            if delta > float(bounds["max_increase"]):
                return True
    policy_override = scenario.get("policy_override")
    if isinstance(policy_override, dict) and "f2change" in policy_override:
        allowed = set(policy_override.get("f2change") or [])
        if any(field not in allowed for field in changed_fields):
            return True
    if isinstance(policy_override, dict) and "remove_from_f2change" in policy_override:
        removed = set(policy_override.get("remove_from_f2change") or [])
        if any(field in removed for field in changed_fields):
            return True
    return False


def difficulty_quota(profiles_per_scenario: int) -> dict[str, int]:
    if profiles_per_scenario == 10:
        return dict(FULL_RUN_DIFFICULTY_QUOTA)
    base = profiles_per_scenario // 3
    remainder = profiles_per_scenario - (base * 3)
    return {
        "easy": base + (1 if remainder > 0 else 0),
        "medium": base + (1 if remainder > 1 else 0),
        "hard": base,
    }


def scenario_offset(scenario_id: str, length: int) -> int:
    if length <= 0:
        return 0
    return sum(ord(char) for char in scenario_id) % length


def run_single_mode(
    *,
    orchestrator: RuntimeOrchestrator,
    context,
    scenario: dict[str, Any],
    mode: str,
    profile: dict[str, Any],
    profile_id: str,
    profile_metadata: dict[str, Any],
) -> dict[str, Any]:
    base_policy = context.policy.to_dict()
    prediction_before = predict_label(context=context, profile=profile)
    policy_override = scenario["policy_override"] if mode == "C" else None
    constraint_spec = scenario["mode_b_constraint_spec"] if mode == "B" else scenario["constraint_spec"]
    if mode == "A":
        return run_legacy_audit(
            context=context,
            scenario=scenario,
            profile=profile,
            profile_id=profile_id,
            profile_metadata=profile_metadata,
            prediction_before=prediction_before,
        )

    request: dict[str, Any] = {"dataset": "bank", "profile": dict(profile)}
    if constraint_spec:
        request["constraint_spec"] = dict(constraint_spec)
    if policy_override:
        request["policy_override"] = dict(policy_override)
    result = orchestrator.handle(request, include_debug_trace=True)
    debug_trace = result.debug_trace.to_dict() if result.debug_trace is not None else {}
    candidate = first_candidate(result)
    prediction_after = None
    changed_fields: list[str] = []
    if candidate is not None:
        prediction_after = predict_label(context=context, profile=candidate.profile)
        changed_fields = list(candidate.changed_features)

    reason_code = normalize_reason_code(
        result.reason_codes[0] if result.reason_codes else None,
        trace_summary=debug_trace.get("trace_summary") or {},
    )
    effective_policy = debug_trace.get("effective_policy") or base_policy
    policy_violation = bool(
        mode == "C"
        and changed_fields
        and any(field not in set(effective_policy.get("f2change", [])) for field in changed_fields)
    )
    force_flip_invariant = bool(
        candidate is None
        or prediction_after == int(context.policy.desired_outcome)
    )
    constraint_violation = reason_code == CONSTRAINT_BLOCKED
    return normalize_run_record(
        scenario_id=scenario["scenario_id"],
        mode=mode,
        profile_id=profile_id,
        base_policy=base_policy,
        policy_override=policy_override,
        effective_policy=effective_policy,
        effective_mi_feature_pairs=debug_trace.get("effective_mi_feature_pairs", []),
        constraint_spec=constraint_spec,
        core_status=debug_trace.get("core_status"),
        presentation_status="presented" if result.counterfactual is not None and result.counterfactual.feasible else "rejected",
        reason_code=reason_code,
        prediction_before=prediction_before,
        prediction_after=prediction_after,
        changed_fields=changed_fields,
        policy_violation=policy_violation,
        constraint_violation=constraint_violation,
        force_flip_invariant=force_flip_invariant,
        trace_summary=debug_trace.get("trace_summary") or {},
        difficulty_label=profile_metadata["difficulty_label"],
        expectation_id=expectation_id_for_run(mode=mode, scenario_id=str(scenario["scenario_id"])),
        expectation_result=expectation_result_for_run(
            mode=mode,
            scenario_id=str(scenario["scenario_id"]),
            policy_violation=policy_violation,
            force_flip_invariant=force_flip_invariant,
        ),
        mode_a_audit_only=False,
        selection_source=profile_metadata["selection_source"],
        profile_signature=profile_metadata["profile_signature"],
    )


def run_legacy_audit(
    *,
    context,
    scenario: dict[str, Any],
    profile: dict[str, Any],
    profile_id: str,
    profile_metadata: dict[str, Any],
    prediction_before: int,
) -> dict[str, Any]:
    service = CounterfactualService(UFCECoreConfig(force_flip=False, allow_legacy_non_flipping_output=True))
    request = UFCERequestBuilder().build(
        "bank",
        pd.DataFrame([profile], columns=context.bundle.feature_order),
        context,
    )
    debug_trace = __import__("llm.src.runtime.types", fromlist=["RuntimeDebugTrace"]).RuntimeDebugTrace()
    result = service.generate(request, debug_trace=debug_trace)
    candidate = result.candidates[0] if result.candidates else None
    prediction_after = None
    changed_fields: list[str] = []
    if candidate is not None:
        prediction_after = predict_label(context=context, profile=candidate.profile)
        changed_fields = list(candidate.changed_features)
    return normalize_run_record(
        scenario_id=scenario["scenario_id"],
        mode="A",
        profile_id=profile_id,
        base_policy=context.policy.to_dict(),
        policy_override=None,
        effective_policy=context.policy.to_dict(),
        effective_mi_feature_pairs=[list(pair) for pair in context.mi_feature_pairs],
        constraint_spec=scenario.get("constraint_spec"),
        core_status="legacy_audit_raw_output" if candidate is not None else "legacy_audit_no_output",
        presentation_status="audit_only",
        reason_code=normalize_reason_code(
            result.reason_codes[0] if result.reason_codes else None,
            trace_summary={} if debug_trace.trace_summary is None else dict(debug_trace.trace_summary),
        ),
        prediction_before=prediction_before,
        prediction_after=prediction_after,
        changed_fields=changed_fields,
        policy_violation=False,
        constraint_violation=False,
        force_flip_invariant=True,
        trace_summary={} if debug_trace.trace_summary is None else dict(debug_trace.trace_summary),
        difficulty_label=profile_metadata["difficulty_label"],
        expectation_id="legacy_audit_reference",
        expectation_result="audit_only",
        mode_a_audit_only=True,
        selection_source=profile_metadata["selection_source"],
        profile_signature=profile_metadata["profile_signature"],
    )


def build_expectations() -> list[dict[str, Any]]:
    return [
        {
            "expectation_id": "mode_c_policy_compliance",
            "description": "Mode C candidates must not change fields outside effective f2change.",
            "applies_to": {"mode": "C"},
            "required_result": "pass",
        },
        {
            "expectation_id": "verified_runtime_force_flip",
            "description": "Presented B/C candidates must predict the desired outcome.",
            "applies_to": {"mode": ["B", "C"]},
            "required_result": "pass",
        },
        {
            "expectation_id": "s5_step_scope_note",
            "description": S5_SCOPE_NOTE,
            "applies_to": {"scenario_id": "S5"},
            "required_result": "scope_note",
        },
        {
            "expectation_id": "legacy_audit_reference",
            "description": "Mode A is a raw legacy reference and is not a production-equivalent runtime.",
            "applies_to": {"mode": "A"},
            "required_result": "audit_only",
        },
    ]


def expectation_id_for_run(*, mode: str, scenario_id: str) -> str:
    if mode == "A":
        return "legacy_audit_reference"
    if scenario_id == "S5":
        return "s5_step_scope_note"
    if mode == "C":
        return "mode_c_policy_compliance"
    return "verified_runtime_force_flip"


def expectation_result_for_run(
    *,
    mode: str,
    scenario_id: str,
    policy_violation: bool,
    force_flip_invariant: bool,
) -> str:
    if mode == "A":
        return "audit_only"
    if scenario_id == "S5":
        return "scope_note" if force_flip_invariant and not policy_violation else "fail"
    if mode == "C":
        return "pass" if force_flip_invariant and not policy_violation else "fail"
    return "pass" if force_flip_invariant else "fail"


def expectation_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((run["expectation_id"], run["expectation_result"]) for run in runs)
    return {
        expectation_id: {result: count for (_expectation_id, result), count in counts.items() if _expectation_id == expectation_id}
        for expectation_id in sorted({run["expectation_id"] for run in runs})
    }


def build_manifest(
    *,
    scenarios: list[dict[str, Any]],
    selected_profiles: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    profiles_per_scenario: int,
    expectation_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evaluation_name": "bank_policy_control_v1",
        "dataset": "bank",
        "profiles_per_policy_scenario": profiles_per_scenario,
        "scenario_count": len([scenario for scenario in scenarios if scenario["scenario_id"] != "S0"]),
        "run_count": len(runs),
        "expected_run_count": expected_run_count(scenarios=scenarios, profiles_per_scenario=profiles_per_scenario),
        "modes": ["A", "B", "C"],
        "mode_a_note": "Audit reference only; not production-equivalent runtime behavior.",
        "main_comparison": "B_vs_C",
        "required_run_fields": list(REQUIRED_RUN_FIELDS),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "scenario_definitions": serialize_scenarios(scenarios),
        "selected_profile_ids": [
            {
                "scenario_id": item["scenario_id"],
                "profile_id": item["profile_id"],
                "source_profile_id": item["source_profile_id"],
                "difficulty_label": item["difficulty_label"],
                "profile_signature": item["profile_signature"],
            }
            for item in selected_profiles
        ],
        "difficulty_distribution": difficulty_distribution(selected_profiles),
        "expectation_summary": expectation_payload["observed_expectation_summary"],
        "standards_alignment": list(STANDARD_ALIGNMENT),
        "standards_claim_scope": (
            "The evaluation is aligned with scenario coverage and AI trustworthiness principles; "
            "it does not claim certification or user-study evidence."
        ),
        "selection_source": SELECTION_SOURCE,
        "selection_strategy": "deterministic_stratified_scenario_selection",
        "random_seed": None,
        "policy_semantics": {
            "uf": "generation bound for UFCE1/UFCE2/UFCE3 intervals",
            "f2change": "allowed generated changed fields for UFCE1 and MI-pair filtering for UFCE2/UFCE3",
            "step": "UFCE1 step control only in the current core",
        },
        "code_version_config": build_code_version_config(),
    }


def validate_artifacts_payload(
    *,
    manifest: dict[str, Any],
    selected_profiles: list[dict[str, Any]],
    expectations: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    del expectations
    if manifest["run_count"] != len(runs):
        raise ValueError("Manifest run_count does not match generated runs.")
    if manifest["expected_run_count"] != len(runs):
        raise ValueError(
            f"Expected {manifest['expected_run_count']} mode runs, generated {len(runs)}."
        )
    missing_fields = [
        field
        for run in runs
        for field in REQUIRED_RUN_FIELDS
        if field not in run
    ]
    if missing_fields:
        raise ValueError(f"Run records are missing required fields: {sorted(set(missing_fields))}")
    if not selected_profiles:
        raise ValueError("No selected profiles were recorded.")


def write_outputs(
    *,
    out_dir: Path,
    manifest: dict[str, Any],
    selected_profiles: list[dict[str, Any]],
    expectations: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    write_json(out_dir / "policy_control_manifest.json", manifest)
    write_json(out_dir / "policy_control_expectations.json", expectations)
    write_profiles_csv(out_dir / "policy_control_profiles.csv", selected_profiles)
    with (out_dir / "policy_control_runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    write_summary_csv(out_dir / "policy_control_summary.csv", runs)
    write_by_scenario_csv(out_dir / "policy_control_by_scenario.csv", runs)
    write_reason_codes_csv(out_dir / "policy_control_reason_codes.csv", runs)
    write_changed_fields_csv(out_dir / "policy_control_changed_fields.csv", runs)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_profiles_csv(path: Path, selected_profiles: list[dict[str, Any]]) -> None:
    fields = [
        "scenario_id",
        "profile_id",
        "source_profile_id",
        "difficulty_label",
        "prediction_before",
        "baseline_core_status",
        "baseline_presentation_status",
        "baseline_changed_fields",
        "profile_signature",
    ]
    rows = []
    for item in selected_profiles:
        rows.append(
            {
                **{field: item.get(field) for field in fields},
                "baseline_changed_fields": json.dumps(item.get("baseline_changed_fields", []), sort_keys=True),
            }
        )
    write_csv(path, rows, fields)


def write_summary_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = aggregate_rows(runs, key_fields=["mode"])
    write_csv(
        path,
        rows,
        [
            "mode",
            "n",
            "core_verified_yield",
            "presented_yield",
            "constraint_blocked_rate",
            "no_valid_cf_rate",
            "policy_violation_rate",
            "force_flip_invariant_rate",
            "avg_sparsity",
        ],
    )


def write_by_scenario_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = aggregate_rows(runs, key_fields=["scenario_id", "mode"])
    write_csv(
        path,
        rows,
        [
            "scenario_id",
            "mode",
            "n",
            "core_verified_yield",
            "presented_yield",
            "constraint_blocked_rate",
            "no_valid_cf_rate",
            "policy_violation_rate",
            "force_flip_invariant_rate",
            "avg_sparsity",
            "main_changed_fields",
        ],
    )


def aggregate_rows(runs: list[dict[str, Any]], *, key_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[tuple(run[field] for field in key_fields)].append(run)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        row = {field: value for field, value in zip(key_fields, key)}
        n = len(items)
        field_counts = Counter(field for item in items for field in (item["changed_fields"] or []))
        row.update(
            {
                "n": n,
                "core_verified_yield": _rate(items, lambda item: item["core_status"] == "valid_counterfactual_found"),
                "presented_yield": _rate(items, lambda item: item["presentation_status"] == "presented"),
                "constraint_blocked_rate": _rate(items, lambda item: bool(item["constraint_violation"])),
                "no_valid_cf_rate": _rate(items, lambda item: item["core_status"] == "no_valid_counterfactual"),
                "policy_violation_rate": _rate(items, lambda item: bool(item["policy_violation"])),
                "force_flip_invariant_rate": _rate(items, lambda item: bool(item["force_flip_invariant"])),
                "avg_sparsity": round(sum(len(item["changed_fields"] or []) for item in items) / n, 4) if n else 0.0,
                "main_changed_fields": ";".join(field for field, _count in field_counts.most_common(3)) or "NONE",
            }
        )
        rows.append(row)
    return rows


def write_reason_codes_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    counts = Counter((run["scenario_id"], run["mode"], run["reason_code"] or "NONE") for run in runs)
    rows = [
        {"scenario_id": scenario_id, "mode": mode, "reason_code": reason_code, "count": count}
        for (scenario_id, mode, reason_code), count in sorted(counts.items())
    ]
    write_csv(path, rows, ["scenario_id", "mode", "reason_code", "count"])


def write_changed_fields_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    counts: Counter[tuple[str, str, str]] = Counter()
    for run in runs:
        fields = run["changed_fields"] or []
        if not fields:
            counts[(run["scenario_id"], run["mode"], "NONE")] += 1
        for field in fields:
            counts[(run["scenario_id"], run["mode"], field)] += 1
    rows = [
        {"scenario_id": scenario_id, "mode": mode, "field": field, "count": count}
        for (scenario_id, mode, field), count in sorted(counts.items())
    ]
    write_csv(path, rows, ["scenario_id", "mode", "field", "count"])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def expected_run_count(*, scenarios: list[dict[str, Any]], profiles_per_scenario: int) -> int:
    return sum(len(scenario["modes"]) * profiles_per_scenario for scenario in scenarios)


def serialize_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario["scenario_id"],
            "description": scenario["description"],
            "modes": list(scenario["modes"]),
            "constraint_spec": scenario["constraint_spec"],
            "policy_override": scenario["policy_override"],
            "mode_b_constraint_spec": scenario["mode_b_constraint_spec"],
            "scope_note": scenario["scope_note"],
        }
        for scenario in scenarios
    ]


def difficulty_distribution(selected_profiles: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = Counter((item["scenario_id"], item["difficulty_label"]) for item in selected_profiles)
    distribution: dict[str, dict[str, int]] = defaultdict(dict)
    for (scenario_id, difficulty), count in sorted(counts.items()):
        distribution[scenario_id][difficulty] = count
    return dict(distribution)


def build_code_version_config() -> dict[str, Any]:
    return {
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short"]),
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }


def git_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def normalize_run_record(**kwargs) -> dict[str, Any]:
    return {field: kwargs.get(field) for field in REQUIRED_RUN_FIELDS}


def normalize_reason_code(reason_code: str | None, *, trace_summary: dict[str, Any]) -> str | None:
    if reason_code == REQUEST_CONSTRAINTS_BLOCKED:
        return CONSTRAINT_BLOCKED
    if reason_code == UFCE_EXECUTION_ERROR:
        return GENERATION_ERROR
    if reason_code == NO_FEASIBLE_CF_FOUND:
        if int(trace_summary.get("raw_outputs_seen") or 0) == 0:
            return NO_CANDIDATE_GENERATED
        return NO_VALID_FLIP
    return reason_code


def first_candidate(result) -> Any | None:
    if result.counterfactual is None or not result.counterfactual.candidates:
        return None
    return result.counterfactual.candidates[0]


def predict_label(*, context, profile: dict[str, Any]) -> int:
    return int(context.bundle.lr.predict(pd.DataFrame([profile], columns=context.bundle.feature_order))[0])


def profile_signature(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=True, sort_keys=True)


def _rate(items: list[dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if predicate(item)) / len(items), 6)


if __name__ == "__main__":
    main()
