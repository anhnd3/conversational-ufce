from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "final" / "part1" / "author_pool_selector_audit.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("author_pool_selector_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThresholdModel:
    def predict(self, frame):
        values = np.asarray(frame["x"], dtype=float)
        return (values >= 5.0).astype(int)


class DistanceModule:
    @staticmethod
    def find_best_row(df, test_instance, continuous_features):
        out = df.copy()
        query = test_instance.loc[:, continuous_features].iloc[0].to_numpy(dtype=float)
        out["proximity"] = out.loc[:, continuous_features].apply(
            lambda row: float(np.linalg.norm(row.to_numpy(dtype=float) - query)),
            axis=1,
        )
        return out.loc[out["proximity"].idxmin()]


class NearestAuthorUFC:
    @staticmethod
    def NNkdtree(data_lab1, test_inst, radius):
        values = data_lab1.loc[:, ["x", "y"]].to_numpy(dtype=float)
        query = test_inst.loc[:, ["x", "y"]].iloc[0].to_numpy(dtype=float)
        distances = np.linalg.norm(values - query, axis=1)
        idx = [int(np.argmin(distances))]
        return data_lab1.iloc[idx].reset_index(drop=True), idx


def movie_scaler():
    return {
        "kind": "movie_minmax_0_100",
        "scale_cols": ["x", "y"],
        "medians": pd.Series({"x": 0.0, "y": 0.0}),
        "mads": pd.Series({"x": 1.0, "y": 100.0}),
        "constant_cols": [],
    }


def test_forceflip_selects_valid_from_same_mixed_pool() -> None:
    runner = load_runner()
    pool = pd.DataFrame({"x": [1.0, 6.0]})
    factual = pd.DataFrame({"x": [0.0]})
    raw, _ = runner.select_author_raw(
        DistanceModule, pool, pd.DataFrame(columns=["x"]), factual, ["x"],
        ThresholdModel(), 1, ["x"], "UFCE2",
    )
    forceflip, _ = runner.select_forceflip(
        DistanceModule, pool, pd.DataFrame(columns=["x"]), factual, ["x"],
        ThresholdModel(), 1, ["x"],
    )
    assert float(raw.iloc[0]["x"]) == 1.0
    assert float(forceflip.iloc[0]["x"]) == 6.0
    assert runner.candidate_record(raw, ThresholdModel(), 1, ["x"])["strict_valid_selected"] == 0
    assert runner.candidate_record(forceflip, ThresholdModel(), 1, ["x"])["strict_valid_selected"] == 1


def test_select_best_uses_movie_scaled_distance_when_scaler_is_supplied() -> None:
    runner = load_runner()
    pool = pd.DataFrame({"x": [1.0, 10.0], "y": [100.0, 0.0]})
    factual = pd.DataFrame({"x": [0.0], "y": [0.0]})

    raw_selected = runner.select_best(DistanceModule, pool, factual, ["x", "y"])
    scaled_selected = runner.select_best(DistanceModule, pool, factual, ["x", "y"], movie_scaler())

    assert float(raw_selected.iloc[0]["x"]) == 10.0
    assert float(scaled_selected.iloc[0]["x"]) == 1.0


def test_movie_neighbor_search_uses_scaled_space_but_returns_raw_rows() -> None:
    runner = load_runner()
    context = {
        "dataset": "movie",
        "features": ["x", "y"],
        "data_lab1": pd.DataFrame({"x": [1.0, 10.0], "y": [100.0, 0.0]}),
        "movie_distance_scaler": movie_scaler(),
    }
    factual = pd.DataFrame({"x": [0.0], "y": [0.0]})

    nn, idx = runner.nearest_author_neighbors(
        NearestAuthorUFC,
        context,
        factual,
        radius=80,
    )

    assert idx == [0]
    assert float(nn.iloc[0]["x"]) == 1.0
    assert float(nn.iloc[0]["y"]) == 100.0


def test_pair_rows_for_scope_emits_aligned_factual_and_counterfactual_values() -> None:
    runner = load_runner()
    record = {
        "dataset": "wine",
        "fold_id": "testfold_0_pred_0.csv",
        "query_pos": 7,
        "method": "UFCE2",
        "config_profile": "final_freeze",
        "bundle_mode": "table7_author_public",
        "effective_radius": 7,
        "effective_n_neighbors": 1000,
        "mi_k": 5,
        "distance_space": "raw",
        "raw_selected_source": "primary_raw",
        "ff_selected_source": "primary_forceflip",
        "raw_pred_label": 0,
        "ff_pred_label": 1,
        "raw_valid": 0,
        "ff_valid": 1,
        "factual": pd.DataFrame({"x": [1.0], "y": [2.0]}),
        "raw_selected": pd.DataFrame({"x": [1.0], "y": [5.0]}),
        "ff_selected": pd.DataFrame({"x": [6.0], "y": [2.0]}),
    }

    raw_rows = runner.pair_rows_for_scope([record], "author_raw", ["x", "y"])
    posthoc_rows = runner.pair_rows_for_scope([record], "author_posthoc_valid_only", ["x", "y"])
    ff_rows = runner.pair_rows_for_scope([record], "author_pool_forceflip", ["x", "y"])

    assert posthoc_rows == []
    assert raw_rows[0]["output_set"] == "author_raw"
    assert raw_rows[0]["changed_features"] == "y"
    assert float(raw_rows[0]["x::y"]) == 2.0
    assert float(raw_rows[0]["cf::y"]) == 5.0
    assert ff_rows[0]["output_set"] == "author_pool_forceflip"
    assert ff_rows[0]["strict_valid_selected"] == 1
    assert ff_rows[0]["changed_features"] == "x"
