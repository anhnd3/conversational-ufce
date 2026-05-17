from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
MPL_DIR = ROOT / ".pytest_cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))


def load_runner():
    path = ROOT / "scripts" / "final" / "part1" / "01b_reproduce_ufce_only.py"
    spec = importlib.util.spec_from_file_location("table7_ufce_only_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_delta_rows_mark_missing_author_values_unrankable() -> None:
    runner = load_runner()
    mean_df = pd.DataFrame(
        {
            "UFCE1": [0.3, 1.0, 1.0, 2.0, 3.0, 4.0],
            "UFCE2": [0.2, 1.0, 1.0, 2.0, 3.0, 4.0],
            "UFCE3": [0.1, 1.0, 1.0, 2.0, 3.0, 4.0],
        },
        index=runner.METRICS,
    )

    rows = runner.build_table7_delta_rows("wine", mean_df)
    prox_jac = [row for row in rows if row["metric_name"] == "Prox-Jac"]

    assert len(prox_jac) == 3
    assert all(row["metric_available"] is False for row in prox_jac)
    assert all(row["relative_delta"] is None for row in prox_jac)
    assert all(row["selection_score"] is None for row in prox_jac)
    assert all(row["missing_reason"] == "author_value_missing_or_not_applicable" for row in prox_jac)


def test_cli_defaults_to_clean_table7_bundle() -> None:
    runner = load_runner()

    args = runner.build_arg_parser().parse_args([])

    assert args.runtime_profile == "final_freeze"
    assert args.bundle_mode == "table7_author_public"
    assert args.ufce_flip_filter == 0


def test_bundle_mode_accepts_hyphen_and_underscore_aliases() -> None:
    runner = load_runner()

    hyphen_args = runner.build_arg_parser().parse_args(["--bundle-mode", "table7_author_public"])
    underscore_args = runner.build_arg_parser().parse_args(["--bundle_mode", "table7_author_public"])
    legacy_args = runner.build_arg_parser().parse_args(["--bundle_mode", "author_public"])

    assert hyphen_args.bundle_mode == "table7_author_public"
    assert underscore_args.bundle_mode == "table7_author_public"
    assert runner._canonical_bundle_mode(legacy_args.bundle_mode) == "table7_author_public"


def test_table7_bundle_resolves_author_public_sources_for_all_datasets() -> None:
    runner = load_runner()
    args = runner.build_arg_parser().parse_args([])

    for dataset in runner.ALL_DATASETS:
        resolution = runner.resolve_bundle_config(
            dataset=dataset,
            args=args,
            author_uf={"feature": 1},
            author_f2change=[],
        )
        assert resolution.effective_bundle_mode == "table7_author_public"
        assert resolution.bundle_cfg["uf_mode"] == "author_public"
        assert resolution.bundle_cfg["f2change_mode"] == "author_public"
        assert resolution.bundle_cfg["step_mode"] == "author_public"
        assert resolution.fallback_used is False
        assert resolution.not_main_table7 is False
        assert resolution.effective_uf_source
        assert resolution.effective_f2change_source
        assert resolution.effective_step_source


def test_movie_author_public_step_includes_experiment_layer_budget() -> None:
    runner = load_runner()
    args = runner.build_arg_parser().parse_args([])

    resolution = runner.resolve_bundle_config(
        dataset="movie",
        args=args,
        author_uf={"Budget": 3000},
        author_f2change=["Budget"],
    )

    assert resolution.step["Budget"] == 3000
    assert "experiment-layer" in resolution.effective_step_source
    assert resolution.fallback_used is False


def test_main_table7_guard_rejects_blindspot_uf_modes() -> None:
    runner = load_runner()

    with pytest.raises(ValueError, match="scaled_up_150"):
        runner.validate_main_table7_bundle(
            "bank",
            "table7_author_public",
            {"uf_mode": "scaled_up_150", "step_mode": "author_public", "f2change_mode": "author_public"},
        )
    with pytest.raises(ValueError, match="neutral_all_1"):
        runner.validate_main_table7_bundle(
            "grad",
            "table7_author_public",
            {"uf_mode": "neutral_all_1", "step_mode": "author_public", "f2change_mode": "author_public"},
        )
    with pytest.raises(ValueError, match="final_blindspot_best"):
        runner.validate_main_table7_bundle(
            "bank",
            "final_blindspot_best",
            {"uf_mode": "author_public", "step_mode": "author_public", "f2change_mode": "author_public"},
        )


def test_blindspot_bundle_resolves_as_not_main_table7() -> None:
    runner = load_runner()
    args = runner.build_arg_parser().parse_args(["--bundle-mode", "final_blindspot_best"])

    resolution = runner.resolve_bundle_config(
        dataset="bank",
        args=args,
        author_uf={"Income": 40},
        author_f2change=[],
    )

    assert resolution.effective_bundle_mode == "final_blindspot_best"
    assert resolution.bundle_cfg["uf_mode"] == "scaled_up_150"
    assert resolution.not_main_table7 is True
    assert resolution.effective_uf_source.startswith("blindspot_diagnostic:")


def test_target_selection_uses_normalized_score_over_raw_abs_delta() -> None:
    runner = load_runner()
    rows = [
        {
            "dataset": "toy",
            "ufce_variant": "UFCE1",
            "metric_name": "Prox-Euc",
            "abs_delta": 100.0,
            "relative_delta": 0.01,
            "selection_score": 0.01,
            "metric_available": True,
        },
        {
            "dataset": "toy",
            "ufce_variant": "UFCE1",
            "metric_name": "Actionability",
            "abs_delta": 1.0,
            "relative_delta": 1.0,
            "selection_score": 1.0,
            "metric_available": True,
        },
    ]

    targets = runner.select_top_metric_targets(rows, 2)

    assert targets[0]["metric_family"] == "APF contract"
    assert targets[0]["primary_metric"] == "Actionability"
    assert targets[0]["abs_delta"] == 1.0
    assert targets[0]["relative_delta"] == 1.0
    assert targets[0]["selection_score"] > targets[1]["selection_score"]


def test_mi_feature_pair_normalization_uses_dataset_feature_order() -> None:
    runner = load_runner()

    normalized = runner.normalize_mi_feature_pairs(
        [["CCAvg", "Income"], {"CDAccount", "CCAvg"}],
        ["Income", "CCAvg", "CDAccount"],
    )

    assert normalized == [["Income", "CCAvg"], ["CCAvg", "CDAccount"]]


def test_delta_normalization_handles_zero_and_missing_author_values() -> None:
    runner = load_runner()

    assert runner._relative_delta(0.0, 2.0) == 2.0
    assert runner._relative_delta(float("nan"), 2.0) is None


def test_apf_denominator_schema_requires_explicit_contract_fields() -> None:
    runner = load_runner()
    rows = [
        {
            "apf_metric_name": "Actionability",
            "apf_pass": True,
            "apf_denominator_type": "queries_with_selected_candidate",
            "apf_eligible_query": True,
            "apf_has_candidate": True,
            "apf_has_selected_candidate": True,
            "apf_counted_in_metric": True,
            "apf_fold_numerator": 1,
            "apf_fold_denominator": 1,
            "apf_fail_reason": "",
        }
    ]

    assert runner.validate_apf_denominator_schema(rows) == []
    assert runner.validate_apf_denominator_schema([{"apf_metric_name": "Actionability"}])


def test_reconstruction_validation_rebuilds_fold_mean_contract() -> None:
    runner = load_runner()
    delta_rows = [
        {
            "dataset": "bank",
            "ufce_variant": "UFCE1",
            "metric_name": "Prox-Euc",
            "reproduced_value": 2.0,
        },
        {
            "dataset": "bank",
            "ufce_variant": "UFCE1",
            "metric_name": "Actionability",
            "reproduced_value": 1.5,
        },
    ]
    metric_rows = [
        {"dataset": "bank", "ufce_variant": "UFCE1", "fold_id": "f0", "prox_euc_final_value": 1.0},
        {"dataset": "bank", "ufce_variant": "UFCE1", "fold_id": "f0", "prox_euc_final_value": 3.0},
        {"dataset": "bank", "ufce_variant": "UFCE1", "fold_id": "f1", "prox_euc_final_value": 2.0},
    ]
    apf_rows = [
        {"dataset": "bank", "ufce_variant": "UFCE1", "fold_id": "f0", "apf_metric_name": "Actionability", "apf_fold_numerator": 1},
        {"dataset": "bank", "ufce_variant": "UFCE1", "fold_id": "f1", "apf_metric_name": "Actionability", "apf_fold_numerator": 2},
    ]

    validation = runner.reconstruct_metrics_from_trace(
        delta_rows=delta_rows,
        metric_trace_rows=metric_rows,
        apf_trace_rows=apf_rows,
    )

    assert validation["validation_ok"].tolist() == [True, True]


def test_diagnostic_validation_requires_provenance_fields_and_candidate_presence() -> None:
    runner = load_runner()
    errors = runner.validate_diagnostic_artifacts(
        reconstruction_df=pd.DataFrame([{"validation_ok": True}]),
        top_targets=[{"dataset": "bank", "rank": 1}],
        representative_cases=[{"dataset": "bank", "target_rank": 1}],
        candidate_generation_rows=[{"selected_candidate_by_ufce_id": "candidate-a"}],
        candidate_selection_rows=[
            {
                "selected_candidate_by_ufce": "candidate-a",
                "selected_candidate_by_ufce_id": "candidate-a",
                "selected_candidate_used_for_metric": "candidate-a",
                "selected_candidate_used_for_force_flip": "candidate-a",
                "metric_candidate_type": "raw_candidate",
                "metric_candidate_id": "candidate-a",
                "metric_candidate_selection_stage": "ufce_returned_output",
                "metric_candidate_differs_from_ufce_selected": True,
                "metric_candidate_explanation": "documented",
            }
        ],
        apf_component_rows=[
            {
                "apf_metric_name": "Actionability",
                "apf_pass": True,
                "apf_denominator_type": "queries_with_selected_candidate",
                "apf_eligible_query": True,
                "apf_has_candidate": True,
                "apf_has_selected_candidate": True,
                "apf_counted_in_metric": True,
                "apf_fold_numerator": 1,
                "apf_fold_denominator": 1,
                "apf_fail_reason": "",
            }
        ],
        provenance={
            "locked_config_source": "source",
            "hyper_tuning_source": "source",
            "hyper_tuning_run_id": "run",
            "hyper_tuning_selection_criterion": "criterion",
            "locked_config_values": {"bank": {}},
            "locked_config_claim_boundary": "boundary",
            "effective_config_by_dataset": {"bank": {}},
            "effective_radius": {"bank": 500},
            "effective_n_neighbors": {"bank": 1000},
            "effective_min_act": {"bank": 0},
            "effective_min_feas": {"bank": 0},
            "effective_uf_source": {"bank": "uf-source"},
            "effective_f2change_source": {"bank": "f2change-source"},
            "effective_step_source": {"bank": "step-source"},
            "effective_bundle_mode": {"bank": "table7_author_public"},
            "fallback_used": {"bank": False},
            "fallback_reason": {"bank": ""},
        },
        fail_on_reconstruction_mismatch=True,
    )

    assert errors == []
