from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ufce" / "data"
MPL_DIR = ROOT / ".pytest_cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from scripts.archieve.reproduce_results_v3 import build_movie_distance_scaler as reproduce_v3_movie_distance_scaler
from ufce.core.data_processing import classify_dataset_getModel, get_movie_user_constraints
from ufce.model_bundles.lr_bundle import (
    DATASET_DROP_COLUMNS,
    DATASET_LABELS,
    DEFAULT_DATASETS,
    DEFAULT_MOVIE_SCALER_LOGIC_VERSION,
    DEFAULT_NON_MOVIE_SCALER_LOGIC_VERSION,
    DatasetModelBundle,
    build_movie_scalers,
    bundle_to_legacy_tuple,
    load_dataset_model_bundle,
    prepare_dataset_for_training,
    save_dataset_model_bundle,
    select_best_split,
    train_dataset_model_bundle,
    validate_bundle_against_metadata,
)


ABS_TOL = 1e-12


def _dataset_path(dataset_name: str) -> Path:
    return DATA_DIR / "{0}.csv".format(dataset_name)


def _load_raw_df(dataset_name: str) -> pd.DataFrame:
    return pd.read_csv(_dataset_path(dataset_name))


def _legacy_classify(dataset_name: str):
    out = classify_dataset_getModel(_load_raw_df(dataset_name), data_name=dataset_name)
    if len(out) == 8:
        lr, lr_mean, lr_std, Xtest, Xtrain, X, y, dataset_df = out
        scaler = None
    elif len(out) == 9:
        lr, lr_mean, lr_std, Xtest, Xtrain, X, y, dataset_df, scaler = out
    else:
        raise AssertionError("Unexpected legacy output length: {0}".format(len(out)))
    return lr, lr_mean, lr_std, Xtest, Xtrain, X, y, dataset_df, scaler


def _train_bundle(dataset_name: str) -> DatasetModelBundle:
    raw_df = _load_raw_df(dataset_name)
    cleaned_dataset_df, X, y, label_col, dropped_columns = prepare_dataset_for_training(
        raw_df,
        dataset_name,
    )
    return train_dataset_model_bundle(
        cleaned_dataset_df=cleaned_dataset_df,
        X=X,
        y=y,
        data_name=dataset_name,
        label_col=label_col,
        dropped_columns=dropped_columns,
        source_data_path=_dataset_path(dataset_name),
    )


@pytest.mark.parametrize("dataset_name", DEFAULT_DATASETS)
def test_prepare_dataset_for_training_matches_legacy(dataset_name: str) -> None:
    cleaned_dataset_df, X, y, label_col, dropped_columns = prepare_dataset_for_training(
        _load_raw_df(dataset_name),
        dataset_name,
    )
    _, _, _, _, _, legacy_X, legacy_y, legacy_dataset_df, _ = _legacy_classify(dataset_name)

    assert label_col == DATASET_LABELS[dataset_name]
    assert dropped_columns == DATASET_DROP_COLUMNS[dataset_name]
    pd.testing.assert_frame_equal(cleaned_dataset_df, legacy_dataset_df)
    pd.testing.assert_frame_equal(X, legacy_X)
    pd.testing.assert_series_equal(y, legacy_y)


@pytest.mark.parametrize("dataset_name", DEFAULT_DATASETS)
def test_select_best_split_matches_legacy_train_test_indices(dataset_name: str) -> None:
    _, X, y, _, _ = prepare_dataset_for_training(_load_raw_df(dataset_name), dataset_name)
    _, train_indices, test_indices, split_diagnostics = select_best_split(X, y)
    _, _, _, legacy_Xtest, legacy_Xtrain, _, _, _, _ = _legacy_classify(dataset_name)

    assert train_indices == legacy_Xtrain.index.tolist()
    assert test_indices == legacy_Xtest.index.tolist()
    assert len(split_diagnostics) == 10


@pytest.mark.parametrize("dataset_name", DEFAULT_DATASETS)
def test_train_dataset_model_bundle_matches_legacy_classifier_behavior(dataset_name: str) -> None:
    bundle = _train_bundle(dataset_name)
    legacy_lr, legacy_mean, legacy_std, legacy_Xtest, legacy_Xtrain, legacy_X, legacy_y, legacy_df, legacy_scaler = (
        _legacy_classify(dataset_name)
    )

    validate_bundle_against_metadata(bundle)
    assert bundle.label_col == DATASET_LABELS[dataset_name]
    assert bundle.dropped_columns == DATASET_DROP_COLUMNS[dataset_name]
    expected_movie_scaler_logic_version = (
        DEFAULT_MOVIE_SCALER_LOGIC_VERSION
        if dataset_name == "movie"
        else DEFAULT_NON_MOVIE_SCALER_LOGIC_VERSION
    )
    assert bundle.movie_scaler_logic_version == expected_movie_scaler_logic_version
    assert bundle.feature_order == list(legacy_X.columns)
    assert bundle.train_indices == legacy_Xtrain.index.tolist()
    assert bundle.test_indices == legacy_Xtest.index.tolist()
    assert math.isclose(bundle.lr_mean, legacy_mean, rel_tol=0.0, abs_tol=ABS_TOL)
    assert math.isclose(bundle.lr_std, legacy_std, rel_tol=0.0, abs_tol=ABS_TOL)
    assert list(bundle.X.columns) == list(legacy_X.columns)
    pd.testing.assert_frame_equal(bundle.X, legacy_X)
    pd.testing.assert_series_equal(bundle.y, legacy_y)
    pd.testing.assert_frame_equal(bundle.dataset_df, legacy_df)
    pd.testing.assert_frame_equal(bundle.Xtrain, legacy_Xtrain)
    pd.testing.assert_frame_equal(bundle.Xtest, legacy_Xtest)
    assert bundle.scaler == legacy_scaler
    assert bundle.lr.predict(bundle.Xtest).tolist() == legacy_lr.predict(legacy_Xtest).tolist()


@pytest.mark.parametrize("dataset_name", DEFAULT_DATASETS)
def test_save_and_load_round_trip_without_retraining(dataset_name: str, tmp_path: Path) -> None:
    bundle = _train_bundle(dataset_name)
    artifact_root = tmp_path / "models"
    save_dataset_model_bundle(bundle, artifact_root / dataset_name)

    loaded_bundle = load_dataset_model_bundle(dataset_name, _dataset_path(dataset_name), artifact_root)
    validate_bundle_against_metadata(loaded_bundle)

    expected_movie_scaler_logic_version = (
        DEFAULT_MOVIE_SCALER_LOGIC_VERSION
        if dataset_name == "movie"
        else DEFAULT_NON_MOVIE_SCALER_LOGIC_VERSION
    )
    assert loaded_bundle.movie_scaler_logic_version == expected_movie_scaler_logic_version
    assert loaded_bundle.feature_order == bundle.feature_order
    assert math.isclose(loaded_bundle.lr_mean, bundle.lr_mean, rel_tol=0.0, abs_tol=ABS_TOL)
    assert math.isclose(loaded_bundle.lr_std, bundle.lr_std, rel_tol=0.0, abs_tol=ABS_TOL)
    assert loaded_bundle.lr.predict(loaded_bundle.Xtest).tolist() == bundle.lr.predict(bundle.Xtest).tolist()
    assert len(bundle_to_legacy_tuple(loaded_bundle)) == 9


def test_movie_distance_scaler_matches_reproduction_helper() -> None:
    bundle = _train_bundle("movie")
    (
        features,
        _catf,
        numf,
        _uf,
        _f2change,
        outcome_label,
        _desired_outcome,
        _nbr_features,
        _protectf,
        _data_lab0,
        _data_lab1,
    ) = get_movie_user_constraints(bundle.dataset_df.copy())
    _, expected_scaler = build_movie_scalers(
        dataset_df=bundle.dataset_df,
        Xtrain=bundle.Xtrain,
        X=bundle.X,
        features=features,
        numf=numf,
        outcome_label=outcome_label,
    )
    reproduction_scaler = reproduce_v3_movie_distance_scaler(
        datasetdf=bundle.dataset_df,
        features=features,
        numf=numf,
        outcome_label=outcome_label,
    )

    assert bundle.movie_distance_scaler["kind"] == "movie_minmax_0_100"
    assert bundle.movie_distance_scaler["scale_cols"] == reproduction_scaler["scale_cols"]
    assert bundle.movie_distance_scaler["scale_cols"] == expected_scaler["scale_cols"]
    assert bundle.movie_distance_scaler["constant_cols"] == reproduction_scaler["constant_cols"]
    pd.testing.assert_series_equal(bundle.movie_distance_scaler["medians"], reproduction_scaler["medians"])
    pd.testing.assert_series_equal(bundle.movie_distance_scaler["mads"], reproduction_scaler["mads"])
    pd.testing.assert_series_equal(bundle.movie_distance_scaler["mins"], reproduction_scaler["mins"])
    pd.testing.assert_series_equal(bundle.movie_distance_scaler["maxs"], reproduction_scaler["maxs"])


def test_load_rejects_hash_mismatch(tmp_path: Path) -> None:
    bundle = _train_bundle("bank")
    artifact_root = tmp_path / "models"
    save_dataset_model_bundle(bundle, artifact_root / "bank")

    modified_data_path = tmp_path / "bank_modified.csv"
    original_bytes = _dataset_path("bank").read_bytes()
    modified_data_path.write_bytes(original_bytes + b"\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_dataset_model_bundle("bank", modified_data_path, artifact_root)


def test_load_rejects_feature_order_mismatch(tmp_path: Path) -> None:
    bundle = _train_bundle("bank")
    artifact_root = tmp_path / "models"
    dataset_dir = artifact_root / "bank"
    save_dataset_model_bundle(bundle, dataset_dir)

    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_order"] = list(reversed(metadata["feature_order"]))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Feature order mismatch"):
        load_dataset_model_bundle("bank", _dataset_path("bank"), artifact_root)


def test_load_rejects_bad_indices(tmp_path: Path) -> None:
    bundle = _train_bundle("bank")
    artifact_root = tmp_path / "models"
    dataset_dir = artifact_root / "bank"
    save_dataset_model_bundle(bundle, dataset_dir)

    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_indices"] = metadata["train_indices"][:]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="disjoint"):
        load_dataset_model_bundle("bank", _dataset_path("bank"), artifact_root)


def test_validate_bundle_against_metadata_checks_movie_scaler_flag() -> None:
    bundle = _train_bundle("movie")
    invalid_bundle = replace(bundle, has_movie_distance_scaler=False)

    with pytest.raises(ValueError, match="has_movie_distance_scaler"):
        validate_bundle_against_metadata(invalid_bundle)
