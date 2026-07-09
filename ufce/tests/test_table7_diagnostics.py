from __future__ import annotations

import importlib.util
import math
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
    path = ROOT / "scripts" / "final" / "part1" / "ufce_only_reproduction.py"
    spec = importlib.util.spec_from_file_location("table7_ufce_only_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_tuner():
    path = ROOT / "scripts" / "final" / "part1" / "ufce_parameter_tuning.py"
    spec = importlib.util.spec_from_file_location("table7_ufce_tuner", path)
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
    assert args.ufce_flip_filter is None


def test_diagnostics_accepts_force_flip_override() -> None:
    runner = load_runner()

    args = runner.build_arg_parser().parse_args(["--diagnostics", "--ufce_flip_filter", "1"])
    args.run_id = "unit-test-run"
    runner.validate_diagnostic_args(args)

    assert args.runtime_profile == "final_freeze"
    assert args.ufce_flip_filter == 1
    assert args.out_dir.endswith("unit-test-run")


def test_new_best_params_runtime_profile_resolves_known_dataset_configs() -> None:
    runner = load_runner()

    expected = {
        "bank": {
            "radius": 90,
            "n_neighbors": 150,
            "min_act": 1,
            "min_feas": 0,
        },
        "bupa": {
            "radius": 70,
            "n_neighbors": 200,
            "min_act": 1,
            "min_feas": 1,
        },
        "grad": {
            "radius": 16,
            "n_neighbors": 400,
            "min_act": 0,
            "min_feas": 0,
        },
        "movie": {
            "radius": 160,
            "n_neighbors": 50,
            "min_act": 1,
            "min_feas": 1,
        },
        "wine": {
            "radius": 15,
            "n_neighbors": 1000,
            "min_act": 0,
            "min_feas": 0,
        },
    }

    for dataset, cfg_expected in expected.items():
        args = runner.build_arg_parser().parse_args(["--runtime_profile", "new_best_params", "--dataset", dataset])
        cfg = runner.resolve_effective_cfg(dataset, args)
        assert {key: cfg[key] for key in ["radius", "n_neighbors", "min_act", "min_feas"]} == cfg_expected
        assert cfg["ufce_flip_filter"] == 0
        assert cfg["core_variant"] == "ufce_core"


def test_bundle_mode_accepts_hyphen_and_underscore_aliases() -> None:
    runner = load_runner()

    hyphen_args = runner.build_arg_parser().parse_args(["--bundle-mode", "table7_author_public"])
    underscore_args = runner.build_arg_parser().parse_args(["--bundle_mode", "table7_author_public"])
    legacy_args = runner.build_arg_parser().parse_args(["--bundle_mode", "author_public"])
    domain_args = runner.build_arg_parser().parse_args(["--bundle_mode", "domain_actionable"])

    assert hyphen_args.bundle_mode == "table7_author_public"
    assert underscore_args.bundle_mode == "table7_author_public"
    assert runner._canonical_bundle_mode(legacy_args.bundle_mode) == "table7_author_public"
    assert domain_args.bundle_mode == "domain_actionable"


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


def test_domain_actionable_bundle_resolves_expected_constraints() -> None:
    runner = load_runner()
    args = runner.build_arg_parser().parse_args(["--bundle-mode", "domain_actionable"])

    expected = {
        "bank": {
            "f2change": ["Income", "CCAvg", "Mortgage", "CDAccount", "Online", "CreditCard", "SecuritiesAccount"],
            "uf": {},
            "step": {"CreditCard": 1, "SecuritiesAccount": 1},
        },
        "grad": {
            "f2change": ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"],
            "uf": {},
            "step": {},
        },
        "bupa": {
            "f2change": ["Sgpt", "Sgot", "Gammagt", "Drinks"],
            "uf": {},
            "step": {"Drinks": 1},
        },
        "wine": {
            "f2change": [
                "fixed acidity",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "pH",
                "alcohol",
                "density",
                "volatile acidity",
                "citric acid",
                "residual sugar",
            ],
            "uf": {},
            "step": {},
        },
        "movie": {
            "f2change": [
                "Production_expense",
                "Multiplex_coverage",
                "Num_multiplex",
                "Movie_length",
                "Lead_Actor_Rating",
                "Lead_Actress_rating",
                "Director_rating",
                "Producer_rating",
                "Genre",
                "Collection",
                "Budget",
            ],
            "uf": {},
            "step": {},
        },
    }

    for dataset, expected_cfg in expected.items():
        dataset_df = pd.read_csv(ROOT / "ufce" / "data" / f"{dataset}.csv")
        (
            _features,
            _catf,
            _numf,
            uf,
            f2change,
            *_rest,
        ) = runner.get_dataset_constraints(dataset, dataset_df)

        resolution = runner.resolve_bundle_config(
            dataset=dataset,
            args=args,
            author_uf=uf,
            author_f2change=f2change,
        )

        assert resolution.effective_bundle_mode == "domain_actionable"
        assert resolution.not_main_table7 is True
        assert resolution.bundle_cfg["uf_mode"] == "domain_actionable"
        assert resolution.bundle_cfg["f2change_mode"] == "domain_actionable"
        assert resolution.bundle_cfg["step_mode"] == "domain_actionable"
        assert resolution.f2change == expected_cfg["f2change"]
        assert set(f2change).issubset(set(resolution.f2change))
        for feature, value in expected_cfg["uf"].items():
            assert resolution.uf[feature] == value
        for feature, value in expected_cfg["step"].items():
            assert resolution.step[feature] == value
        assert resolution.effective_uf_source.startswith(f"domain_actionable:{dataset}:")
        assert resolution.effective_f2change_source.startswith(f"domain_actionable:{dataset}:")
        assert resolution.effective_step_source.startswith(f"domain_actionable:{dataset}:")


def test_domain_actionable_rejects_unknown_datasets() -> None:
    runner = load_runner()
    args = runner.build_arg_parser().parse_args(["--bundle-mode", "domain_actionable"])

    with pytest.raises(ValueError, match="domain_actionable is only supported"):
        runner.resolve_bundle_config(
            dataset="adult",
            args=args,
            author_uf={"hours-per-week": 5},
            author_f2change=["hours-per-week"],
        )


def test_tuner_domain_actionable_profile_avoids_magic_uf_step() -> None:
    tuner = load_tuner()

    wine_df = pd.read_csv(ROOT / "ufce" / "data" / "wine.csv")
    *_prefix, wine_uf, wine_f2change, _outcome, _desired, _nbr, _protect, _lab0, _lab1 = tuner.get_dataset_constraints("wine", wine_df)
    wine_step = tuner.get_step_config("wine")
    resolved_uf, resolved_f2change, resolved_step = tuner.apply_constraint_profile(
        dataset="wine",
        uf=wine_uf,
        f2change=wine_f2change,
        step=wine_step,
        profile=tuner.DOMAIN_ACTIONABLE_CONSTRAINT_PROFILE,
    )

    assert "citric acid" in resolved_f2change
    assert "residual sugar" in resolved_f2change
    assert "sulphates" not in resolved_f2change
    assert "sulphates" not in resolved_uf
    assert "sulphates" not in resolved_step

    movie_df = pd.read_csv(ROOT / "ufce" / "data" / "movie.csv")
    *_prefix, movie_uf, movie_f2change, _outcome, _desired, _nbr, _protect, _lab0, _lab1 = tuner.get_dataset_constraints("movie", movie_df)
    movie_step = tuner.get_step_config("movie")
    resolved_uf, resolved_f2change, resolved_step = tuner.apply_constraint_profile(
        dataset="movie",
        uf=movie_uf,
        f2change=movie_f2change,
        step=movie_step,
        profile=tuner.DOMAIN_ACTIONABLE_CONSTRAINT_PROFILE,
    )

    assert "Marketing_expense" not in resolved_f2change
    assert "3D_available" not in resolved_f2change
    assert "Marketing_expense" not in resolved_uf
    assert "3D_available" not in resolved_uf
    assert "Marketing_expense" not in resolved_step
    assert "3D_available" not in resolved_step


def test_tuner_eval_mode_and_constraint_profile_are_part_of_config_key() -> None:
    tuner = load_tuner()
    base = {
        "radius": 7,
        "n_neighbors": 1000,
        "contprox_metric": "euclidean",
        "min_act": 1,
        "min_feas": 1,
        "ufce_flip_filter": 1,
        "constraint_profile": tuner.DOMAIN_ACTIONABLE_CONSTRAINT_PROFILE,
        "eval_mode": tuner.STRICT_EVAL_MODE,
    }
    relaxed = dict(base)
    relaxed["eval_mode"] = tuner.RELAXED_EVAL_MODE

    assert tuner.eval_mode_to_ufce_method(tuner.STRICT_EVAL_MODE) == "other"
    assert tuner.eval_mode_to_ufce_method(tuner.RELAXED_EVAL_MODE) == "ufc"
    assert tuner.config_key("wine", 123, base) != tuner.config_key("wine", 123, relaxed)


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


def test_movie_prox_euc_contract_uses_rms_minmax_0_100() -> None:
    runner = load_runner()
    scaler = {
        "kind": "movie_minmax_0_100",
        "scale_cols": ["a", "b"],
        "medians": pd.Series({"a": 0.0, "b": 0.0}),
        "mads": pd.Series({"a": 1.0, "b": 1.0}),
        "constant_cols": [],
    }
    factual = pd.DataFrame([{"a": 0.0, "b": 0.0}])

    one_axis, _contrib, cols, normalizer = runner._feature_contributions(
        factual,
        pd.DataFrame([{"a": 100.0, "b": 0.0}]),
        ["a", "b"],
        scaler,
    )
    all_axes, _contrib, _cols, _normalizer = runner._feature_contributions(
        factual,
        pd.DataFrame([{"a": 100.0, "b": 100.0}]),
        ["a", "b"],
        scaler,
    )
    clipped, _contrib, _cols, _normalizer = runner._feature_contributions(
        factual,
        pd.DataFrame([{"a": 200.0, "b": 200.0}]),
        ["a", "b"],
        scaler,
    )
    no_shared, _contrib, no_cols, no_normalizer = runner._feature_contributions(
        factual,
        pd.DataFrame([{"c": 100.0}]),
        ["c"],
        scaler,
    )

    assert cols == ["a", "b"]
    assert normalizer == "movie_minmax_0_100_rms"
    assert one_axis == pytest.approx(100.0 / math.sqrt(2.0))
    assert all_axes == pytest.approx(100.0)
    assert clipped == pytest.approx(100.0)
    assert math.isnan(no_shared)
    assert no_cols == []
    assert no_normalizer == "none"


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
