from __future__ import annotations

from types import SimpleNamespace

from llm.src.runtime.types import CounterfactualCandidate
from scripts.archieve.run_part2_backend_comparison_report import (
    G4_BASELINE_UNCONSTRAINED,
    G4_CONSTRAINT_AWARE,
    METHODS,
    build_failure_breakdown,
    build_single_case_results,
    build_method_table,
    classify_failure_bucket,
    flatten_seed_results,
    render_markdown,
    select_seed_batches,
    select_final_candidate,
)


class PassingInvariantValidator:
    def validate(self, *, result, current_profile, context):
        del result
        del current_profile
        del context
        return SimpleNamespace(status="passed")


def test_select_final_candidate_uses_shared_deterministic_sort_and_constraint_filter():
    context = SimpleNamespace(
        bundle=SimpleNamespace(feature_order=["Income", "Mortgage"]),
    )
    runtime = SimpleNamespace(
        invariant_validator=PassingInvariantValidator(),
        runtime_mode="stable_demo",
    )
    candidate_income = CounterfactualCandidate(
        method="dice",
        rank=1,
        profile={"Income": 90.0, "Mortgage": 100.0},
        changed_features=["Income"],
    )
    candidate_mortgage = CounterfactualCandidate(
        method="dice",
        rank=2,
        profile={"Income": 80.0, "Mortgage": 120.0},
        changed_features=["Mortgage"],
    )

    unconstrained = select_final_candidate(
        runtime=runtime,
        context=context,
        factual_prediction=None,
        factual_profile={"Income": 80.0, "Mortgage": 100.0},
        raw_candidates=[candidate_mortgage, candidate_income],
        active_constraint_spec=None,
    )
    constrained = select_final_candidate(
        runtime=runtime,
        context=context,
        factual_prediction=None,
        factual_profile={"Income": 80.0, "Mortgage": 100.0},
        raw_candidates=[candidate_mortgage, candidate_income],
        active_constraint_spec={"disallowed_changes": ["Income"]},
    )

    assert unconstrained["selected_candidate"] == candidate_income
    assert constrained["selected_candidate"] == candidate_mortgage


def test_build_method_table_aggregates_locked_backend_columns():
    seed_results = [
        {
            "methods": {
                "UFCE1": {
                    "success": 1,
                    "failure": 0,
                    "raw_validity": 1,
                    "raw_label_flip_count": 1,
                    "actionable_label_flip_count": 1,
                    "invariant_valid_candidate_count": 1,
                    "failure_bucket": None,
                    "failure_reason_codes": [],
                    "feasibility": 1,
                    "actionability": 1,
                    "plausibility": 1,
                    "proximity": 0.2,
                    "sparsity": 1,
                    "latency_ms": 10.0,
                },
                "UFCE2": {
                    "success": 1,
                    "failure": 0,
                    "raw_validity": 1,
                    "raw_label_flip_count": 1,
                    "actionable_label_flip_count": 1,
                    "invariant_valid_candidate_count": 1,
                    "failure_bucket": None,
                    "failure_reason_codes": [],
                    "feasibility": 1,
                    "actionability": 1,
                    "plausibility": 1,
                    "proximity": 0.3,
                    "sparsity": 2,
                    "latency_ms": 11.0,
                },
                "UFCE3": {
                    "success": 1,
                    "failure": 0,
                    "raw_validity": 1,
                    "raw_label_flip_count": 1,
                    "actionable_label_flip_count": 1,
                    "invariant_valid_candidate_count": 1,
                    "failure_bucket": None,
                    "failure_reason_codes": [],
                    "feasibility": 1,
                    "actionability": 1,
                    "plausibility": 1,
                    "proximity": 0.4,
                    "sparsity": 3,
                    "latency_ms": 12.0,
                },
                "DiCE": {
                    "success": 0,
                    "failure": 1,
                    "raw_validity": 1,
                    "raw_label_flip_count": 1,
                    "actionable_label_flip_count": 0,
                    "invariant_valid_candidate_count": 0,
                    "failure_bucket": "invariant_fail",
                    "failure_reason_codes": ["REQUEST_CONSTRAINTS_BLOCKED"],
                    "feasibility": 0,
                    "actionability": 0,
                    "plausibility": 0,
                    "proximity": None,
                    "sparsity": None,
                    "latency_ms": 20.0,
                },
                "AR": {
                    "success": 1,
                    "failure": 0,
                    "raw_validity": 1,
                    "raw_label_flip_count": 1,
                    "actionable_label_flip_count": 1,
                    "invariant_valid_candidate_count": 1,
                    "failure_bucket": None,
                    "failure_reason_codes": [],
                    "feasibility": 1,
                    "actionability": 1,
                    "plausibility": 1,
                    "proximity": 0.4,
                    "sparsity": 2,
                    "latency_ms": 30.0,
                },
            }
        }
    ]

    table = build_method_table(seed_results)

    assert table["UFCE1"]["validity"]["mean"] == 1.0
    assert table["UFCE2"]["failure_rate"]["mean"] == 0.0
    assert table["DiCE"]["failure_rate"]["mean"] == 1.0
    assert table["DiCE"]["diagnostics"]["failure_reason_counts"] == {"REQUEST_CONSTRAINTS_BLOCKED": 1}
    assert table["DiCE"]["diagnostics"]["failure_bucket_counts"]["invariant_fail"] == 1
    assert table["UFCE3"]["diagnostics"]["raw_flip_rate"]["mean"] == 1.0
    assert table["UFCE3"]["plausibility"]["mean"] == 1.0
    assert table["AR"]["actionability"]["mean"] == 1.0


def test_flatten_seed_results_emits_per_seed_per_method_rows():
    rows = flatten_seed_results(
        [
            {
                "seed_id": "seed-1",
                "dataset": "bank",
                "active_constraint_spec": None,
                "methods": {
                    "UFCE1": {
                        "success": 1,
                        "failure": 0,
                        "raw_validity": 1,
                        "raw_flip_count": 1,
                        "invariant_pass_count": 1,
                        "final_selection_success": 1,
                        "final_failure_reason_codes": [],
                        "first_failed_invariant_violations": None,
                        "feasibility": 1,
                        "actionability": 1,
                        "proximity": 0.2,
                        "sparsity": 1,
                        "latency_ms": 12.0,
                        "selected_candidate": {"changed_features": ["Income"]},
                        "selection_reason_codes": [],
                        "raw_error": None,
                    },
                    "UFCE2": {
                        "success": 1,
                        "failure": 0,
                        "raw_validity": 1,
                        "raw_flip_count": 1,
                        "invariant_pass_count": 1,
                        "final_selection_success": 1,
                        "final_failure_reason_codes": [],
                        "first_failed_invariant_violations": None,
                        "feasibility": 1,
                        "actionability": 1,
                        "proximity": 0.3,
                        "sparsity": 2,
                        "latency_ms": 13.0,
                        "selected_candidate": {"changed_features": ["Income", "CCAvg"]},
                        "selection_reason_codes": [],
                        "raw_error": None,
                    },
                    "UFCE3": {
                        "success": 1,
                        "failure": 0,
                        "raw_validity": 1,
                        "raw_flip_count": 1,
                        "invariant_pass_count": 1,
                        "final_selection_success": 1,
                        "final_failure_reason_codes": [],
                        "first_failed_invariant_violations": None,
                        "feasibility": 1,
                        "actionability": 1,
                        "proximity": 0.4,
                        "sparsity": 3,
                        "latency_ms": 14.0,
                        "selected_candidate": {"changed_features": ["Income", "CCAvg", "Mortgage"]},
                        "selection_reason_codes": [],
                        "raw_error": None,
                    },
                    "DiCE": {
                        "success": 0,
                        "failure": 1,
                        "raw_validity": 0,
                        "raw_flip_count": 0,
                        "invariant_pass_count": 0,
                        "final_selection_success": 0,
                        "final_failure_reason_codes": ["REQUEST_CONSTRAINTS_BLOCKED"],
                        "first_failed_invariant_violations": ["candidate_does_not_flip"],
                        "feasibility": 0,
                        "actionability": 0,
                        "proximity": None,
                        "sparsity": None,
                        "latency_ms": 14.0,
                        "selected_candidate": None,
                        "selection_reason_codes": ["REQUEST_CONSTRAINTS_BLOCKED"],
                        "raw_error": "no_cf",
                    },
                },
            }
        ]
    )

    assert rows[0]["seed_id"] == "seed-1"
    assert rows[0]["method"] == "UFCE1"
    assert rows[0]["success"] == 1
    assert rows[3]["method"] == "DiCE"
    assert rows[3]["raw_error"] == "no_cf"


def test_build_failure_breakdown_counts_exclusive_failure_buckets():
    seed_results = [
        {
            "methods": {
                method: {
                    "failure_bucket": "constraint_blocked" if method == "DiCE" else "no_candidate" if method == "AR" else None
                }
                for method in METHODS
            }
        }
    ]

    breakdown = build_failure_breakdown(seed_results)

    assert breakdown["DiCE"]["constraint_blocked"] == 1
    assert breakdown["AR"]["no_candidate"] == 1
    assert breakdown["UFCE1"]["timeout_or_error"] == 0


def test_classify_failure_bucket_applies_locked_priority_order():
    assert (
        classify_failure_bucket(
            raw_error="timeout",
            raw_candidate_count=0,
            raw_label_flip_count=0,
            invariant_valid_candidate_count=0,
            constraint_compliant_candidate_count=0,
            selected_candidate_emitted=0,
            active_constraint_spec={"disallowed_changes": ["Income"]},
        )
        == "timeout_or_error"
    )
    assert (
        classify_failure_bucket(
            raw_error=None,
            raw_candidate_count=0,
            raw_label_flip_count=0,
            invariant_valid_candidate_count=0,
            constraint_compliant_candidate_count=0,
            selected_candidate_emitted=0,
            active_constraint_spec=None,
        )
        == "no_candidate"
    )
    assert (
        classify_failure_bucket(
            raw_error=None,
            raw_candidate_count=2,
            raw_label_flip_count=0,
            invariant_valid_candidate_count=0,
            constraint_compliant_candidate_count=0,
            selected_candidate_emitted=0,
            active_constraint_spec=None,
        )
        == "no_flip"
    )
    assert (
        classify_failure_bucket(
            raw_error=None,
            raw_candidate_count=2,
            raw_label_flip_count=1,
            invariant_valid_candidate_count=0,
            constraint_compliant_candidate_count=0,
            selected_candidate_emitted=0,
            active_constraint_spec=None,
        )
        == "invariant_fail"
    )
    assert (
        classify_failure_bucket(
            raw_error=None,
            raw_candidate_count=2,
            raw_label_flip_count=1,
            invariant_valid_candidate_count=1,
            constraint_compliant_candidate_count=0,
            selected_candidate_emitted=0,
            active_constraint_spec={"max_changed_features": 1},
        )
        == "constraint_blocked"
    )
    assert (
        classify_failure_bucket(
            raw_error=None,
            raw_candidate_count=2,
            raw_label_flip_count=1,
            invariant_valid_candidate_count=1,
            constraint_compliant_candidate_count=1,
            selected_candidate_emitted=1,
            active_constraint_spec=None,
        )
        is None
    )


def test_select_seed_batches_defaults_to_first_single_case_and_matches_constrained_profile():
    corpus = {
        "seeds": [
            {"seed_id": "TIERC-001", "dataset": "bank", "profile": {"Income": 49.0}},
            {"seed_id": "TIERC-002", "dataset": "bank", "profile": {"Income": 34.0}},
        ],
        "constrained_subset": [
            {
                "seed_id": "TIERC-C-001",
                "dataset": "bank",
                "profile": {"Income": 49.0},
                "constraint_spec": {"disallowed_changes": ["Income"]},
            }
        ],
    }

    main, constrained, meta = select_seed_batches(
        corpus=corpus,
        single_case=True,
        seed_id=None,
        seed_index=None,
    )

    assert [item["seed_id"] for item in main] == ["TIERC-001"]
    assert [item["seed_id"] for item in constrained] == ["TIERC-C-001"]
    assert meta["mode"] == "single_case"
    assert meta["seed_id"] == "TIERC-001"


def test_build_single_case_results_preserves_selected_method_rows():
    payload = build_single_case_results(
        main_seed_result={
            "seed_id": "TIERC-001",
            "dataset": "bank",
            "profile": {"Income": 49.0},
            "methods": {method: {"method": method, "success": 1} for method in METHODS},
        },
        constrained_seed_results=[],
        selection_metadata={"mode": "single_case", "seed_id": "TIERC-001"},
    )

    assert payload["selected_seed"]["seed_id"] == "TIERC-001"
    assert payload["main_unconstrained"]["methods"]["UFCE3"]["method"] == "UFCE3"
    assert payload["matched_constrained_subset"] is None


def test_render_markdown_includes_fairness_contract_and_git_commit():
    summary = {
        "run_id": "demo",
        "runner_scope": "part2_g4_backend_comparison",
        "scorer_version": "v1",
        "timestamp_local": "2026-03-25T12:00:00+07:00",
        "timezone": "UTC+07:00",
        "corpus_version": "tier_c_demo",
        "corpus_sha256": "abc",
        "catalog_version": "phase3_2_validation_catalog_v1",
        "catalog_sha256": "def",
        "git_commit": "deadbeef",
        "method_list": list(METHODS),
        "single_case_mode": True,
        "fairness_contract": {
            "seed_selection_rule": "predictor_rejected_factual_seeds",
            "selection_space": "raw_feature_space",
            "rejected_label": 0,
            "rules": ["same factual seeds", "same predictor"],
            "preprocessing_and_normalization": {"shared": True},
        },
        G4_BASELINE_UNCONSTRAINED: {
            method: {
                "validity": {"mean": 1.0},
                "sparsity": {"mean": 1.0},
                "proximity": {"mean": 0.1},
                "feasibility": {"mean": 1.0},
                "actionability": {"mean": 1.0},
                "plausibility": {"mean": 1.0},
                "runtime_latency_ms": {"mean": 10.0, "p50": 10.0, "p95": 10.0, "max": 10.0},
                "failure_rate": {"mean": 0.0},
                "diagnostics": {
                    "raw_flip_rate": {"mean": 1.0},
                    "invariant_pass_rate": {"mean": 1.0},
                    "final_selection_success_rate": {"mean": 1.0},
                    "failure_bucket_counts": {
                        "no_candidate": 0,
                        "no_flip": 0,
                        "invariant_fail": 0,
                        "constraint_blocked": 0,
                        "timeout_or_error": 0,
                    },
                    "failure_reason_counts": {},
                },
            }
            for method in METHODS
        },
        G4_CONSTRAINT_AWARE: {
            method: {
                "validity": {"mean": 1.0},
                "sparsity": {"mean": 1.0},
                "proximity": {"mean": 0.1},
                "feasibility": {"mean": 1.0},
                "actionability": {"mean": 1.0},
                "plausibility": {"mean": 1.0},
                "runtime_latency_ms": {"mean": 10.0, "p50": 10.0, "p95": 10.0, "max": 10.0},
                "failure_rate": {"mean": 0.0},
                "diagnostics": {
                    "raw_flip_rate": {"mean": 1.0},
                    "invariant_pass_rate": {"mean": 1.0},
                    "final_selection_success_rate": {"mean": 1.0},
                    "failure_bucket_counts": {
                        "no_candidate": 0,
                        "no_flip": 0,
                        "invariant_fail": 0,
                        "constraint_blocked": 0,
                        "timeout_or_error": 0,
                    },
                    "failure_reason_counts": {},
                },
            }
            for method in METHODS
        },
        "seed_counts": {"constrained_subset": 20},
        "failure_breakdown": {
            G4_BASELINE_UNCONSTRAINED: {
                method: {
                    "no_candidate": 0,
                    "no_flip": 0,
                    "invariant_fail": 0,
                    "constraint_blocked": 0,
                    "timeout_or_error": 0,
                }
                for method in METHODS
            },
            G4_CONSTRAINT_AWARE: {
                method: {
                    "no_candidate": 0,
                    "no_flip": 0,
                    "invariant_fail": 0,
                    "constraint_blocked": 0,
                    "timeout_or_error": 0,
                }
                for method in METHODS
            },
        },
        "single_case_results": {
            "selected_seed": {"seed_id": "TIERC-001", "dataset": "bank", "profile": {"Income": 49.0}},
            "main_unconstrained": {
                "methods": {
                    method: {
                        "success": 1,
                        "raw_candidate_count": 1,
                        "raw_label_flip_count": 1,
                        "actionable_label_flip_count": 1,
                        "invariant_valid_candidate_count": 1,
                        "selected_candidate_emitted": 1,
                        "failure_bucket": None,
                        "sparsity": 1,
                        "proximity": 0.1,
                        "actionability": 1,
                        "feasibility": 1,
                        "latency_ms": 10.0,
                        "selection_reason_codes": [],
                        "first_failed_invariant_violations": None,
                    }
                    for method in METHODS
                }
            },
            "matched_constrained_subset": None,
        },
        "aggregate_validation": {"ok": True, "difference_count": 0},
    }

    markdown = render_markdown(summary)

    assert "deadbeef" in markdown
    assert "same factual seeds" in markdown
    assert "predictor_rejected_factual_seeds" in markdown
    assert "G4_baseline_unconstrained" in markdown
    assert "G4_constraint_aware" in markdown
    assert "constrained_subset_seed_count" in markdown
    assert "Failure Breakdown" in markdown
    assert "Single Case" in markdown
    assert "UFCE3" in markdown
