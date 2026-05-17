from __future__ import annotations

import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ufce" / "data"
FOLD_DIR = DATA_DIR / "folds"
MPL_DIR = ROOT / ".pytest_cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from scripts.archieve.reproduce_full_table7_result import run_one_fold
from scripts.archieve.ufce_hypothesis_ablation import (
    ANCHOR_PUBLIC_NAME,
    DEFAULT_DATASETS,
    build_anchor_specs,
    build_dataset_context,
    build_f2change_variants,
    extract_provenance_records,
    main,
    run_variant_fold,
    select_strict_variant_names,
    standardized_feature_ranking,
    validate_variant_step_coverage,
)


@pytest.fixture(scope="session")
def provenance_records() -> list[dict]:
    return extract_provenance_records(DATA_DIR, DEFAULT_DATASETS)


@pytest.fixture(scope="session")
def provenance_map(provenance_records: list[dict]) -> dict[str, dict]:
    return {str(row["dataset"]): row for row in provenance_records}


@pytest.fixture(scope="session")
def grad_context(provenance_map: dict[str, dict]):
    return build_dataset_context(
        "grad",
        DATA_DIR,
        provenance_map,
        contprox_metric="euclidean",
    )


@pytest.fixture(scope="session")
def movie_context(provenance_map: dict[str, dict]):
    return build_dataset_context(
        "movie",
        DATA_DIR,
        provenance_map,
        contprox_metric="euclidean",
    )


@pytest.fixture(scope="session")
def grad_smoke_outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("uf_ablation_smoke")
    main(
        [
            "--datasets",
            "grad",
            "--out_dir",
            str(out_dir),
            "--max_folds",
            "1",
            "--stages",
            "single",
            "--variant_filter",
            "author_public,neutral_all_1",
            "--data_dir",
            str(DATA_DIR),
            "--fold_dir",
            str(FOLD_DIR),
            "--no_cf",
            "50",
            "--seed",
            "123",
        ]
    )
    return out_dir


@pytest.fixture(scope="session")
def movie_smoke_outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("uf_ablation_movie_smoke")
    main(
        [
            "--datasets",
            "movie",
            "--out_dir",
            str(out_dir),
            "--max_folds",
            "1",
            "--stages",
            "all",
            "--data_dir",
            str(DATA_DIR),
            "--fold_dir",
            str(FOLD_DIR),
            "--no_cf",
            "50",
            "--seed",
            "123",
        ]
    )
    return out_dir


def test_extract_provenance_records_includes_expected_role_scan(provenance_records: list[dict]) -> None:
    provenance_by_dataset = {row["dataset"]: row for row in provenance_records}

    assert sorted(provenance_by_dataset.keys()) == sorted(DEFAULT_DATASETS)

    grad_row = provenance_by_dataset["grad"]
    assert grad_row["layer_b_available"] is True
    assert grad_row["exact_match"] is True
    assert grad_row["normalized_match"] is True
    assert grad_row["ufce1_generation_uses_step"] is True
    assert grad_row["ufce1_generation_uses_f2change"] is True
    assert grad_row["ufce2_generation_uses_f2change"] is False
    assert grad_row["ufce3_generation_uses_f2change"] is True
    assert grad_row["actionability_uses_uf"] is True
    assert grad_row["actionability_uses_f2change"] is True
    assert grad_row["feasibility_uses_uf"] is True
    assert grad_row["feasibility_uses_f2change"] is True

    movie_row = provenance_by_dataset["movie"]
    assert json.loads(movie_row["layer_a_step_missing_for_f2change"]) == ["Budget"]
    assert movie_row["layer_a_step_runnable_for_f2change"] is False
    assert json.loads(movie_row["layer_c_step_missing_for_f2change"]) == []
    assert movie_row["layer_c_step_runnable_for_f2change"] is True


def test_visible_experiment_available_only_for_grad(provenance_records: list[dict]) -> None:
    provenance_by_dataset = {row["dataset"]: row for row in provenance_records}

    assert provenance_by_dataset["grad"]["layer_b_available"] is True
    for dataset in ["bank", "wine", "bupa", "movie"]:
        assert provenance_by_dataset[dataset]["layer_b_available"] is False
        assert "feature_mismatch" in provenance_by_dataset[dataset]["notes"]


def test_f2change_variants_use_standardized_coefficient_ranking(grad_context) -> None:
    ranking = standardized_feature_ranking(grad_context, grad_context.author_f2change)
    variants = build_f2change_variants(grad_context)

    expected_minus_top_1 = [feature for feature in grad_context.author_f2change if feature != ranking[0]]
    expected_minus_top_2 = [
        feature
        for feature in grad_context.author_f2change
        if feature not in set(ranking[:2])
    ]

    assert ranking[:2] == ["CGPA", "GRE Score"]
    assert variants["minus_top_1"] == expected_minus_top_1
    assert variants["minus_top_2"] == expected_minus_top_2
    assert variants["categorical_only"] == ["Research"]


def test_movie_author_preset_variant_is_detected_as_invalid(movie_context) -> None:
    variant = next(spec for spec in build_anchor_specs(movie_context) if spec.variant_name == "author_preset_bundle")
    missing = validate_variant_step_coverage(movie_context, variant)

    assert missing == ["Budget"]


def test_select_strict_variant_names_keeps_anchor_and_top_two() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "dataset": "grad",
                "method": "UFCE1",
                "variant_name": ANCHOR_PUBLIC_NAME,
                "variant_family": "anchor",
                "flip_filter": 0,
                "delta_valid_rate_vs_public_baseline": 0.0,
                "delta_invalid_rate_vs_public_baseline": 0.0,
                "delta_empty_rate_vs_public_baseline": 0.0,
                "feature_path_score_vs_public_baseline": 0.0,
            },
            {
                "dataset": "grad",
                "method": "UFCE1",
                "variant_name": "uf__neutral_all_1",
                "variant_family": "uf",
                "flip_filter": 0,
                "delta_valid_rate_vs_public_baseline": 0.30,
                "delta_invalid_rate_vs_public_baseline": 0.10,
                "delta_empty_rate_vs_public_baseline": -0.20,
                "feature_path_score_vs_public_baseline": 0.40,
            },
            {
                "dataset": "grad",
                "method": "UFCE2",
                "variant_name": "step__coarser_double",
                "variant_family": "step",
                "flip_filter": 0,
                "delta_valid_rate_vs_public_baseline": -0.25,
                "delta_invalid_rate_vs_public_baseline": 0.05,
                "delta_empty_rate_vs_public_baseline": 0.10,
                "feature_path_score_vs_public_baseline": 0.60,
            },
            {
                "dataset": "grad",
                "method": "UFCE3",
                "variant_name": "f2change__minus_top_1",
                "variant_family": "f2change",
                "flip_filter": 0,
                "delta_valid_rate_vs_public_baseline": 0.05,
                "delta_invalid_rate_vs_public_baseline": 0.02,
                "delta_empty_rate_vs_public_baseline": -0.01,
                "feature_path_score_vs_public_baseline": 0.20,
            },
        ]
    )

    selected = select_strict_variant_names(summary_df)

    assert selected == {
        "grad": [
            ANCHOR_PUBLIC_NAME,
            "uf__neutral_all_1",
            "step__coarser_double",
        ]
    }


def test_smoke_run_writes_expected_artifacts(grad_smoke_outputs: Path) -> None:
    provenance_path = grad_smoke_outputs / "provenance.csv"
    master_summary_path = grad_smoke_outputs / "master_summary.csv"
    master_summary_partial_path = grad_smoke_outputs / "master_summary_partial.csv"
    dataset_dir = grad_smoke_outputs / "grad"
    single_factor_path = dataset_dir / "single_factor.csv"
    single_factor_partial_path = dataset_dir / "single_factor_partial.csv"
    query_trace_path = dataset_dir / "query_trace.jsonl"
    query_trace_partial_path = dataset_dir / "query_trace_partial.jsonl"

    assert provenance_path.exists()
    assert master_summary_path.exists()
    assert master_summary_partial_path.exists()
    assert single_factor_path.exists()
    assert single_factor_partial_path.exists()
    assert query_trace_path.exists()
    assert query_trace_partial_path.exists()
    assert query_trace_path.read_text(encoding="utf-8").strip() != ""
    assert query_trace_partial_path.read_text(encoding="utf-8").strip() != ""

    summary_df = pd.read_csv(master_summary_path)
    single_factor_df = pd.read_csv(single_factor_path)

    assert set(summary_df["variant_name"]) >= {
        ANCHOR_PUBLIC_NAME,
        "author_preset_bundle",
        "uf__author_public",
        "uf__neutral_all_1",
    }
    assert set(single_factor_df["variant_family"]) == {"uf"}
    assert set(single_factor_df["variant_name"]) == {"uf__author_public", "uf__neutral_all_1"}


def test_movie_smoke_run_skips_invalid_step_variants_and_writes_partials(movie_smoke_outputs: Path) -> None:
    provenance_path = movie_smoke_outputs / "provenance.csv"
    master_summary_path = movie_smoke_outputs / "master_summary.csv"
    master_summary_partial_path = movie_smoke_outputs / "master_summary_partial.csv"
    master_sidecar_path = movie_smoke_outputs / "master_summary_sidecar.json"
    skipped_variants_path = movie_smoke_outputs / "skipped_variants.csv"
    dataset_dir = movie_smoke_outputs / "movie"
    single_factor_partial_path = dataset_dir / "single_factor_partial.csv"
    joint_ablation_partial_path = dataset_dir / "joint_ablation_partial.csv"
    feature_change_partial_path = dataset_dir / "feature_change_profile_partial.csv"
    fold_breakdown_partial_path = dataset_dir / "fold_breakdown_partial.csv"
    query_trace_partial_path = dataset_dir / "query_trace_partial.jsonl"

    assert provenance_path.exists()
    assert master_summary_path.exists()
    assert master_summary_partial_path.exists()
    assert master_sidecar_path.exists()
    assert skipped_variants_path.exists()
    assert single_factor_partial_path.exists()
    assert joint_ablation_partial_path.exists()
    assert feature_change_partial_path.exists()
    assert fold_breakdown_partial_path.exists()
    assert query_trace_partial_path.exists()

    skipped_df = pd.read_csv(skipped_variants_path)
    assert not skipped_df.empty
    assert set(skipped_df["variant_name"]) >= {"author_preset_bundle", "step__author_preset"}
    assert "Budget" in set(skipped_df["missing_step_keys"])

    summary_df = pd.read_csv(master_summary_path)
    assert "author_preset_bundle" not in set(summary_df["variant_name"])
    assert "step__author_preset" not in set(summary_df["variant_name"])

    sidecar = json.loads(master_sidecar_path.read_text(encoding="utf-8"))
    skipped_names = {row["variant_name"] for row in sidecar["skipped_variants"]}
    assert {"author_preset_bundle", "step__author_preset"} <= skipped_names


def test_public_bundle_baseline_matches_frozen_reproduction_fold(grad_context) -> None:
    variant = build_anchor_specs(grad_context)[0]
    fold_path = FOLD_DIR / "grad" / "totest" / "testfold_0_pred_0.csv"

    random.seed(123)
    np.random.seed(123)
    harness_result = run_variant_fold(
        grad_context,
        variant,
        fold_path,
        no_cf=50,
    )

    random.seed(123)
    np.random.seed(123)
    fold_result = run_one_fold(
        dataset="grad",
        fold_idx=0,
        fold_path=str(fold_path),
        datasetdf=grad_context.datasetdf,
        features=grad_context.features,
        catf=grad_context.catf,
        numf=grad_context.numf,
        uf=grad_context.author_uf,
        f2change=grad_context.author_f2change,
        protectf=grad_context.protectf,
        outcome_label=grad_context.outcome_label,
        desired_outcome=grad_context.desired_outcome,
        step=variant.step,
        MI_FP=grad_context.mi_pairs_top5,
        lr=grad_context.lr,
        X=grad_context.X,
        Xtest=grad_context.Xtest,
        Xtrain=grad_context.Xtrain,
        data_lab1=grad_context.data_lab1,
        no_cf=50,
        scaler_ar=grad_context.scaler_ar,
        ufce_mad_scaler=grad_context.ufce_mad_scaler,
        movie_distance_scaler=grad_context.movie_distance_scaler,
        flip_filter_enabled=False,
        debug_enabled=False,
        trace_positions=[],
        total_folds=1,
        expected_pred1_rate=None,
    )
    baseline_metrics, _baseline_stds, _times, _pred_debug = fold_result

    baseline_rows = {
        row["method"]: row
        for row in harness_result["fold_rows"]
    }

    method_index = {"UFCE1": 0, "UFCE2": 1, "UFCE3": 2}
    metric_name_map = {
        "prox_jac": "Prox-Jac",
        "prox_euc": "Prox-Euc",
        "sparsity": "Sparsity",
        "actionability": "Actionability",
        "plausibility": "Plausibility",
        "feasibility": "Feasibility",
    }

    for method, idx in method_index.items():
        row = baseline_rows[method]
        for summary_metric, baseline_metric in metric_name_map.items():
            assert float(row[summary_metric]) == pytest.approx(
                float(baseline_metrics[baseline_metric][idx]),
                abs=1e-12,
            )


def test_master_summary_counts_match_query_totals(grad_smoke_outputs: Path) -> None:
    summary_df = pd.read_csv(grad_smoke_outputs / "master_summary.csv")

    assert not summary_df.empty
    counts = summary_df["valid_count"] + summary_df["invalid_count"] + summary_df["empty_count"]
    pd.testing.assert_series_equal(
        counts.reset_index(drop=True),
        summary_df["n_queries"].reset_index(drop=True),
        check_names=False,
    )
