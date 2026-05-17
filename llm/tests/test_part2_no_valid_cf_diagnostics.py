from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "final" / "part2" / "07_no_valid_cf_diagnostics.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("part2_no_valid_cf_diagnostics", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_row(case_id: str, group: str, reject_class: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "session_id": f"session_{case_id}",
        "group": group,
        "summary_type": "runtime_reject",
        "reject_class": reject_class,
    }


def primary_parity_report(module) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    rows.extend(metric_row(f"G1-{index:03d}", "G1", "no_feasible_cf") for index in range(69))
    rows.extend(metric_row(f"G2-{index:03d}", "G2", "no_feasible_cf") for index in range(71))
    rows.append(metric_row("G2-BLOCK-001", "G2", "request_constraints_blocked"))
    return {
        "run_id": module.PRIMARY_RUN_ID,
        "aggregate_validation": {"ok": True},
        "corpus_sha256": module.EXPECTED_CORPUS_SHA256,
        "per_case_results": rows,
    }


def toy_context() -> SimpleNamespace:
    return SimpleNamespace(
        bundle=SimpleNamespace(
            feature_order=["Income"],
            lr=SimpleNamespace(
                classes_=np.asarray([0, 1]),
                coef_=np.asarray([[1.0]]),
                intercept_=np.asarray([-5.0]),
            ),
        ),
        policy=SimpleNamespace(
            desired_outcome=1,
            f2change=["Income"],
            feature_type_map={"Income": "float"},
        ),
    )


def test_d0_runtime_semantics_audit_writes_expected_findings(tmp_path: Path):
    module = load_runner()

    audit = module.build_runtime_semantics_audit(run_root=tmp_path)

    assert audit["semantics_ok"] is True
    assert audit["ufce_request_builder_mentions_constraint_spec"] is False
    assert audit["ufce_mapper_mentions_constraint_spec"] is False
    assert (tmp_path / "runtime_semantics_audit.md").exists()


def test_c0_parity_accepts_authoritative_203759_counts(tmp_path: Path):
    module = load_runner()
    report = primary_parity_report(module)
    corpus = {"corpus_sha256": module.EXPECTED_CORPUS_SHA256}

    parity = module.build_parity_gate(
        report=report,
        corpus=corpus,
        source_report_path=tmp_path / "thesis_metrics_report.json",
        source_report_sha256=module.EXPECTED_SOURCE_REPORT_SHA256,
    )

    assert parity["ok"] is True
    assert parity["counts"] == {
        "G1_no_valid_cf": 69,
        "G2_no_valid_cf": 71,
        "constraint_blocked_post_filter": 1,
    }


def test_c0_parity_rejects_non_full_or_wrong_count_run(tmp_path: Path):
    module = load_runner()
    report = primary_parity_report(module)
    report["per_case_results"] = report["per_case_results"][:-1]

    with pytest.raises(ValueError, match="C0 parity gate failed"):
        module.build_parity_gate(
            report=report,
            corpus={"corpus_sha256": module.EXPECTED_CORPUS_SHA256},
            source_report_path=tmp_path / "thesis_metrics_report.json",
            source_report_sha256=module.EXPECTED_SOURCE_REPORT_SHA256,
        )


def test_phase_a_expected_invariant_and_hidden_coupling_statuses():
    module = load_runner()

    assert (
        module.phase_a_invariant_status(
            {"outcome": "no_valid_cf", "generated_candidate_count": 0},
            {"outcome": "no_valid_cf", "generated_candidate_count": 0},
        )
        == "expected_invariant_holds"
    )
    assert (
        module.phase_a_invariant_status(
            {"outcome": "no_valid_cf", "generated_candidate_count": 0},
            {"outcome": "counterfactual_found", "generated_candidate_count": 1},
        )
        == "runtime_semantics_violation_or_hidden_coupling"
    )


def test_runtime_config_limited_detail_labels():
    module = load_runner()

    assert module.classify_config_limited(["B_C2"]) == "radius_limited"
    assert module.classify_config_limited(["B_C1"]) == "neighbor_filter_sensitive"
    assert module.classify_config_limited(["B_C3"]) == "combined_config_limited"


def test_lr_oracle_toy_feasible_and_infeasible_numeric_bound():
    module = load_runner()
    context = toy_context()
    factual = {"Income": 1.0}
    feature_ranges = {"Income": (0.0, 10.0)}
    model_input_audit = module.detect_oracle_model_input_transform(context)

    feasible = module.lr_feasibility_oracle(
        factual_profile=factual,
        constraint_spec=None,
        context=context,
        feature_domains={},
        feature_ranges=feature_ranges,
        model_input_audit=model_input_audit,
    )
    infeasible = module.lr_feasibility_oracle(
        factual_profile=factual,
        constraint_spec={"numeric_bounds": {"Income": {"max": 4.0}}},
        context=context,
        feature_domains={},
        feature_ranges=feature_ranges,
        model_input_audit=model_input_audit,
    )

    assert feasible["status"] == "feasible"
    assert feasible["feasible"] is True
    assert feasible["changed_features"] == ["Income"]
    assert feasible["witness"] == {"Income": 10.0}
    assert feasible["witness_score"]["predicted_label"] == 1
    assert feasible["label_satisfied"] is True
    assert feasible["constraint_audit"]["all_constraints_satisfied"] is True
    assert feasible["reloaded_witness_score_matches"] is True
    assert feasible["reloaded_witness_audit_matches"] is True
    assert infeasible["status"] == "infeasible"
    assert infeasible["feasible"] is False
    assert infeasible["witness"] is None
    assert infeasible["best_scored_profile"] == {"Income": 4.0}
    assert infeasible["best_scored_score"]["margin"] == -1.0
    assert infeasible["best_scored_is_valid_witness"] is False
    assert infeasible["is_valid_oracle_witness"] is False


def test_witness_json_reload_rescores_stably():
    module = load_runner()
    context = toy_context()
    model_input_audit = module.detect_oracle_model_input_transform(context)
    witness = {"Income": 10.0}
    score = module.score_lr_profile(witness, context, model_input_audit=model_input_audit)
    audit = module.audit_oracle_candidate(
        factual_profile={"Income": 1.0},
        candidate_profile=witness,
        constraint_spec=None,
        context=context,
        feature_domains={},
        feature_ranges={"Income": (0.0, 10.0)},
        atol=module.ORACLE_ATOL,
    )

    reload_check = module.validate_witness_json_reload(
        witness_profile=witness,
        persisted_score=score,
        persisted_audit=audit,
        factual_profile={"Income": 1.0},
        constraint_spec=None,
        context=context,
        feature_domains={},
        feature_ranges={"Income": (0.0, 10.0)},
        model_input_audit=model_input_audit,
        atol=module.ORACLE_ATOL,
    )

    assert reload_check == {"score_matches": True, "audit_matches": True}


def test_changed_feature_count_uses_numeric_atol():
    module = load_runner()
    feature_type_map = {"Income": "float", "CDAccount": "binary"}
    factual = {"Income": 10.0, "CDAccount": 0}

    within_atol = {"Income": 10.0 + module.ORACLE_ATOL / 2.0, "CDAccount": 0}
    beyond_atol = {"Income": 10.0 + module.ORACLE_ATOL * 2.0, "CDAccount": 0}
    categorical = {"Income": 10.0, "CDAccount": 1}

    assert module.changed_features(factual, within_atol, feature_type_map, atol=module.ORACLE_ATOL) == []
    assert module.changed_features(factual, beyond_atol, feature_type_map, atol=module.ORACLE_ATOL) == ["Income"]
    assert module.changed_features(factual, categorical, feature_type_map, atol=module.ORACLE_ATOL) == ["CDAccount"]


def test_blocked_feature_negative_audit_and_validation_failure():
    module = load_runner()
    context = toy_context()
    audit = module.audit_oracle_candidate(
        factual_profile={"Income": 1.0},
        candidate_profile={"Income": 10.0},
        constraint_spec={"disallowed_changes": ["Income"]},
        context=context,
        feature_domains={},
        feature_ranges={"Income": (0.0, 10.0)},
        atol=module.ORACLE_ATOL,
    )

    assert audit["blocked_fields_satisfied"] is False
    assert audit["all_constraints_satisfied"] is False
    with pytest.raises(RuntimeError, match="feasible witness does not satisfy encoded constraints"):
        module.validate_oracle_audit(
            phase_c_rows=[
                {
                    "case_id": "toy",
                    "oracle_feasible": True,
                    "oracle_witness_persisted": True,
                    "oracle_witness_profile_json": '{"Income":10.0}',
                    "oracle_label_satisfied": True,
                    "oracle_all_constraints_satisfied": False,
                    "oracle_max_changed_features": "",
                    "oracle_reloaded_witness_score_matches": True,
                    "oracle_reloaded_witness_audit_matches": True,
                }
            ],
            final_rows=[],
            oracle_witness_audit={
                "skipped": False,
                "feasible_witness_persisted": 1,
                "oracle_feasible_cases": 1,
                "infeasible_best_scored_trace_persisted": 0,
                "oracle_infeasible_cases": 0,
            },
            full_run=False,
        )
