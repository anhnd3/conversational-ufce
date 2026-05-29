from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MPL_DIR = ROOT / ".pytest_cache" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

from ufce.core_author_fixed import cfmethods
from ufce.core_author_fixed.ufce import UFCE


class SumThresholdModel:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def predict(self, X):
        frame = pd.DataFrame(X)
        return (frame.sum(axis=1).to_numpy(dtype=float) > self.threshold).astype(int)


class AlwaysDesiredModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_decimal_candidate_grid_is_not_truncated() -> None:
    ufc = UFCE()

    grid = ufc._candidate_grid(0.2, 0.8, "density", step={"density": 0.2})

    assert grid == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert all(isinstance(value, float) for value in grid)


def test_strict_filter_rejects_nonflipping_candidates() -> None:
    ufc = UFCE()
    factual = pd.DataFrame([{"a": 0.0, "b": 0.0}])
    candidates = pd.DataFrame([{"a": 0.1, "b": 0.0}, {"a": 2.0, "b": 0.0}])

    strict = ufc._strict_candidates(
        candidates,
        factual,
        ["a", "b"],
        ["a"],
        [],
        SumThresholdModel(threshold=1.0),
        1,
        {"a": [0.0, 3.0]},
        order=["a", "b"],
        intervals={"a": [0.0, 3.0]},
        plausibility_data=None,
        require_plausible=False,
    )

    assert strict.to_dict("records") == [{"a": 2.0, "b": 0.0}]


def test_strict_filter_rejects_non_actionable_and_protected_changes() -> None:
    ufc = UFCE()
    factual = pd.DataFrame([{"a": 0.0, "b": 0.0, "protected": 0.0}])
    candidates = pd.DataFrame(
        [
            {"a": 1.0, "b": 1.0, "protected": 0.0},
            {"a": 1.0, "b": 0.0, "protected": 1.0},
            {"a": 1.0, "b": 0.0, "protected": 0.0},
        ]
    )

    strict = ufc._strict_candidates(
        candidates,
        factual,
        ["a", "b", "protected"],
        ["a"],
        ["protected"],
        AlwaysDesiredModel(),
        1,
        {"a": [0.0, 2.0]},
        order=["a", "b", "protected"],
        intervals={"a": [0.0, 2.0]},
        plausibility_data=None,
        require_plausible=False,
    )

    assert strict.to_dict("records") == [{"a": 1.0, "b": 0.0, "protected": 0.0}]


def test_strict_filter_rejects_lof_outlier_even_when_it_flips() -> None:
    ufc = UFCE(n_neighbors=3)
    factual = pd.DataFrame([{"a": 0.0, "b": 0.0}])
    plausible_train = pd.DataFrame(
        [{"a": float(v), "b": 0.0} for v in [-0.1, -0.05, 0.0, 0.05, 0.1, 0.12, -0.12]]
    )
    candidates = pd.DataFrame([{"a": 0.08, "b": 0.0}, {"a": 100.0, "b": 0.0}])

    strict = ufc._strict_candidates(
        candidates,
        factual,
        ["a", "b"],
        ["a"],
        [],
        AlwaysDesiredModel(),
        1,
        {"a": [0.0, 200.0]},
        order=["a", "b"],
        intervals={"a": [0.0, 200.0]},
        plausibility_data=plausible_train,
        require_plausible=True,
    )

    assert strict.to_dict("records") == [{"a": 0.08, "b": 0.0}]


def test_double_and_triple_do_not_change_features_outside_f2change() -> None:
    ufc = UFCE()
    df = pd.DataFrame(
        [
            {"Sgpt": 0.0, "Sgot": 0.0, "Gammagt": 0.0, "Drinks": 0.0},
            {"Sgpt": 1.0, "Sgot": 1.0, "Gammagt": 1.0, "Drinks": 1.0},
            {"Sgpt": 2.0, "Sgot": 2.0, "Gammagt": 2.0, "Drinks": 2.0},
            {"Sgpt": 3.0, "Sgot": 3.0, "Gammagt": 3.0, "Drinks": 3.0},
        ]
    )
    factual = pd.DataFrame([{"Sgpt": 0.0, "Sgot": 0.0, "Gammagt": 0.0, "Drinks": 0.0}])
    intervals = {"Sgpt": [0.0, 3.0], "Sgot": [0.0, 3.0], "Gammagt": [0.0, 3.0], "Drinks": [0.0, 3.0]}
    uf = dict(intervals)
    f2change = ["Sgpt", "Sgot", "Gammagt"]
    pairs = [["Gammagt", "Drinks"], ["Sgpt", "Sgot"]]
    order = ["Sgpt", "Sgot", "Gammagt", "Drinks"]
    model = SumThresholdModel(threshold=0.5)

    double_cf, _ = ufc.Double_F(
        df,
        factual,
        [],
        pairs,
        [],
        order,
        intervals,
        f2change,
        model,
        1,
        order,
        5,
        step={"Sgpt": 1.0, "Sgot": 1.0, "Gammagt": 1.0},
        changeable_features=f2change,
        uf=uf,
        plausibility_data=None,
    )
    triple_cf, _ = ufc.Triple_F(
        df,
        factual,
        [],
        pairs,
        [],
        order,
        intervals,
        f2change,
        model,
        1,
        order,
        5,
        step={"Sgpt": 1.0, "Sgot": 1.0, "Gammagt": 1.0},
        uf=uf,
        plausibility_data=None,
    )

    assert not double_cf.empty
    assert not triple_cf.empty
    assert set(double_cf["Drinks"].unique()) == {0.0}
    assert set(triple_cf["Drinks"].unique()) == {0.0}


def test_mixed_selector_uses_jaccard_and_normalized_mad_euclidean() -> None:
    candidates = pd.DataFrame(
        [
            {"num": 9.0, "cat": 0.0},
            {"num": 1.0, "cat": 1.0},
        ]
    )
    factual = pd.DataFrame([{"num": 0.0, "cat": 0.0}])
    scaler = {
        "scale_cols": ["num"],
        "medians": pd.Series({"num": 0.0}),
        "mads": pd.Series({"num": 1.0}),
        "constant_cols": [],
    }

    best = cfmethods.find_best_row(
        candidates,
        factual,
        continuous_features=["num"],
        categorical_features=["cat"],
        distance_scaler=scaler,
        lambda_weight=2.0,
    )

    assert best["num"] == 1.0
    assert best["cat"] == 1.0
