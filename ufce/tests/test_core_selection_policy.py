from __future__ import annotations

import inspect

import pandas as pd

from llm.src.runtime.counterfactual_service import CounterfactualService
from scripts.final.part1.author_raw_posthoc_replay import MethodOutput, aligned_cf_and_test
from ufce.core import cfmethods as core_cfmethods
from ufce.core.ufce import UFCE as CoreUFCE
from ufce.post_hoc import cfmethods as post_hoc_cfmethods
from ufce.post_hoc.ufce import UFCE as PostHocUFCE
from ufce.ufce_ff import cfmethods as ff_cfmethods


FEATURES = ["x", "flip"]


class FakeModel:
    def predict(self, df):
        return pd.Series(df["flip"]).astype(int).to_numpy()


class FakeUFC:
    radius = 500
    n_neighbors = 100

    def __init__(self, primary=None, explore=None, single=None):
        self.primary = primary if primary is not None else pd.DataFrame(columns=FEATURES)
        self.explore = explore if explore is not None else pd.DataFrame(columns=FEATURES)
        self.single = single if single is not None else pd.DataFrame(columns=FEATURES)

    def NNkdtree(self, data, query, return_meta=False):
        nn = pd.DataFrame([{"x": 0.0, "flip": 0}])
        idx = [0]
        if return_meta:
            return nn, idx, {"within_radius_count": 1, "neighbor_idx_head": [0]}
        return nn, idx

    def make_uf_nn_interval(self, *args, **kwargs):
        return {}

    def make_intervals(self, *args, **kwargs):
        return {}

    def Single_F(self, *args, **kwargs):
        return self.single.copy()

    def Double_F(self, *args, **kwargs):
        return self.primary.copy(), self.explore.copy()

    def Triple_F(self, *args, **kwargs):
        return self.primary.copy(), self.explore.copy()


class ConstantRegression:
    def predict(self, values):
        return pd.Series([0.0] * len(values)).to_numpy()


def test_core_exposes_raw_contract_without_selection_policy_helpers() -> None:
    assert not hasattr(core_cfmethods, "resolve_selection_policy")
    assert "selection_policy" not in inspect.signature(core_cfmethods.sfexp).parameters
    assert "selection_policy" not in inspect.signature(core_cfmethods.dfexp).parameters
    assert "selection_policy" not in inspect.signature(core_cfmethods.tfexp).parameters


def test_core_ufce3_keeps_author_raw_behavior_without_explore_fallback(monkeypatch) -> None:
    explore = pd.DataFrame([{"x": 1.0, "flip": 1}])
    monkeypatch.setattr(core_cfmethods, "ufc", FakeUFC(primary=pd.DataFrame(columns=FEATURES), explore=explore))

    cfs, _time, idx, trace = core_cfmethods.tfexp(
        pd.DataFrame(columns=FEATURES),
        pd.DataFrame(columns=FEATURES),
        pd.DataFrame([{"x": 0.0, "flip": 0}]),
        {},
        [["x", "flip"]],
        ["x"],
        [],
        FEATURES,
        [],
        FakeModel(),
        1,
        10,
        FEATURES,
        return_trace=True,
    )

    assert cfs.empty
    assert idx == []
    assert trace[0]["author_compat_fallback"] == "ufce3_explore_fallback_disabled_to_match_author"


def test_post_hoc_generation_matches_core_raw_generation(monkeypatch) -> None:
    single = pd.DataFrame([{"x": 0.1, "flip": 0}, {"x": 1.0, "flip": 1}])
    factual = pd.DataFrame([{"x": 0.0, "flip": 0}])
    fake = FakeUFC(single=single)
    monkeypatch.setattr(core_cfmethods, "ufc", fake)
    monkeypatch.setattr(post_hoc_cfmethods, "ufc", fake)

    core_cfs, *_ = core_cfmethods.sfexp(
        pd.DataFrame(columns=FEATURES),
        pd.DataFrame(columns=FEATURES),
        factual,
        {},
        {},
        FEATURES,
        ["x"],
        [],
        FakeModel(),
        1,
        10,
        FEATURES,
    )
    post_hoc_cfs, *_ = post_hoc_cfmethods.sfexp(
        pd.DataFrame(columns=FEATURES),
        pd.DataFrame(columns=FEATURES),
        factual,
        {},
        {},
        FEATURES,
        ["x"],
        [],
        FakeModel(),
        1,
        10,
        FEATURES,
    )

    pd.testing.assert_frame_equal(core_cfs.reset_index(drop=True), post_hoc_cfs.reset_index(drop=True))


def test_core_and_post_hoc_double_feature_do_not_pre_filter_invalid_candidates(monkeypatch) -> None:
    def fake_regression_model(self, df, f1, f2):
        return ConstantRegression(), 0.0, 0.0

    for cls in (CoreUFCE, PostHocUFCE):
        monkeypatch.setattr(cls, "regressionModel", fake_regression_model)
        ufc = cls()
        cfdf, _explore = ufc.Double_F(
            pd.DataFrame([{"x": 0.0, "flip": 0.0}, {"x": 1.0, "flip": 1.0}]),
            pd.DataFrame([{"x": 0.0, "flip": 0.0}]),
            [],
            [["x", "flip"]],
            [],
            FEATURES,
            {"x": [0, 2], "flip": [0, 0]},
            FEATURES,
            FakeModel(),
            1,
            FEATURES,
            1,
        )

        assert len(cfdf) > 0
        assert set(int(v) for v in FakeModel().predict(cfdf[FEATURES])) == {0}


def test_post_hoc_valid_only_aggregation_drops_non_flipping_outputs() -> None:
    method_output = MethodOutput(
        cfdf=pd.DataFrame([{"x": 0.1, "flip": 0}, {"x": 1.0, "flip": 1}]),
        found_idx=[0, 1],
        runtime_ms=0.0,
    )
    fold_df = pd.DataFrame([{"x": 0.0, "flip": 0}, {"x": 0.2, "flip": 0}])

    cfdf, testdf, selected_count, valid_count = aligned_cf_and_test(
        method_output=method_output,
        fold_df=fold_df,
        features=FEATURES,
        bb_model=FakeModel(),
        desired_outcome=1,
        valid_only=True,
    )

    assert selected_count == 2
    assert valid_count == 1
    assert len(cfdf) == 1
    assert len(testdf) == 1
    assert int(cfdf.iloc[0]["flip"]) == 1


def test_ufce_ff_ufce3_filters_before_nearest_selection(monkeypatch) -> None:
    explore = pd.DataFrame(
        [
            {"x": 10.0, "flip": 1},
            {"x": 1.0, "flip": 1},
            {"x": 0.1, "flip": 0},
        ]
    )
    monkeypatch.setattr(ff_cfmethods, "ufc", FakeUFC(primary=pd.DataFrame(columns=FEATURES), explore=explore))

    cfs, _time, idx, trace = ff_cfmethods.tfexp(
        pd.DataFrame(columns=FEATURES),
        pd.DataFrame(columns=FEATURES),
        pd.DataFrame([{"x": 0.0, "flip": 0}]),
        {},
        [["x", "flip"]],
        ["x"],
        [],
        FEATURES,
        [],
        FakeModel(),
        1,
        10,
        FEATURES,
        return_trace=True,
    )

    assert idx == [0]
    assert float(cfs.iloc[0]["x"]) == 1.0
    assert int(cfs.iloc[0]["flip"]) == 1
    assert trace[0]["core_variant"] == "ufce_ff"
    assert trace[0]["validity_gate_stage"] == "pre_find_best_row"
    assert trace[0]["effective_validity_gate"] == 1


def test_route2_counterfactual_service_uses_ufce_ff_core() -> None:
    service = CounterfactualService()
    modules = [runner.__module__ for _name, runner in service._ordered_runners()]
    assert modules == [
        "ufce.ufce_ff.cfmethods",
        "ufce.ufce_ff.cfmethods",
        "ufce.ufce_ff.cfmethods",
    ]
