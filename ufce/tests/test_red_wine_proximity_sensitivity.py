from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "final" / "part1" / "red_wine_proximity_sensitivity.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("red_wine_prox_euc_sensitivity_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pair_frame(features):
    data = {}
    for feature in features:
        data[f"x::{feature}"] = [1.0]
    for feature in features:
        data[f"cf::{feature}"] = [2.0]
    return pd.DataFrame(data)


def test_euclidean_distance() -> None:
    runner = load_runner()
    assert runner.euclidean([0.0, 0.0], [3.0, 4.0]) == 5.0


def test_validate_feature_alignment_rejects_order_mismatch() -> None:
    runner = load_runner()
    frame = pair_frame(["b", "a"])
    with pytest.raises(ValueError, match="x feature order mismatch"):
        runner.validate_feature_alignment(frame, ["a", "b"])


def test_validate_feature_alignment_rejects_label_like_columns() -> None:
    runner = load_runner()
    frame = pair_frame(["label"])
    with pytest.raises(ValueError, match="Forbidden"):
        runner.validate_feature_alignment(frame, ["label"])


def test_minmax_transform_handles_constant_columns() -> None:
    runner = load_runner()
    train = pd.DataFrame({"a": [0.0, 10.0], "b": [5.0, 5.0]})
    spec = runner.fit_minmax_0_1(train, ["a", "b"])
    transformed = runner.transform_minmax_0_1(pd.DataFrame({"a": [5.0], "b": [5.0]}), spec, ["a", "b"])
    clipped = runner.transform_minmax_0_1(pd.DataFrame({"a": [15.0], "b": [5.0]}), spec, ["a", "b"])
    assert np.isclose(float(transformed.iloc[0]["a"]), 0.5)
    assert float(transformed.iloc[0]["b"]) == 0.0
    assert float(clipped.iloc[0]["a"]) == 1.0


def test_changed_feature_distance_uses_only_changed_original_features() -> None:
    runner = load_runner()
    spec = runner.fit_minmax_0_1(pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0]}), ["a", "b"])
    x = pd.Series({"a": 0.0, "b": 10.0})
    cf = pd.Series({"a": 3.0, "b": 10.0})

    distance, changed = runner.distance_for_space(x, cf, ["a", "b"], "changed_features_original", spec)

    assert distance == 3.0
    assert changed == ["a"]


def test_common_subset_detail_keeps_only_queries_present_in_all_output_sets() -> None:
    runner = load_runner()
    detail = pd.DataFrame(
        [
            {"variant": "UFCE1", "output_set": "author_raw", "fold_id": "f0", "query_pos": 1, "space": "original_raw", "distance": 1.0},
            {"variant": "UFCE1", "output_set": "author_raw", "fold_id": "f0", "query_pos": 2, "space": "original_raw", "distance": 2.0},
            {"variant": "UFCE1", "output_set": "author_posthoc_valid_only", "fold_id": "f0", "query_pos": 2, "space": "original_raw", "distance": 3.0},
            {"variant": "UFCE1", "output_set": "author_pool_forceflip", "fold_id": "f0", "query_pos": 2, "space": "original_raw", "distance": 4.0},
            {"variant": "UFCE1", "output_set": "author_pool_forceflip", "fold_id": "f0", "query_pos": 3, "space": "original_raw", "distance": 5.0},
        ]
    )

    common = runner.common_subset_detail(detail)
    summary = runner.summarize_distances(common, "common_sample_subset")

    assert set(common["query_pos"]) == {2}
    assert set(summary["n"]) == {1}
