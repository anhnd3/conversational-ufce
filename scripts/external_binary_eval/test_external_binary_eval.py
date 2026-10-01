import numpy as np
import pandas as pd

from scripts.external_binary_eval.config import CATEGORICAL_FEATURES, REGIMES, TERMINAL_STATUSES
from scripts.external_binary_eval.constraints import build_constraint_policy, policy_for_query
from scripts.external_binary_eval.metrics import status_from_results, wilson_interval
from scripts.external_binary_eval.model import fit_model
from scripts.external_binary_eval.verification import verify_candidate


def _schema():
    families = [
        "pft_events", "pft_days_logged", "pft_visits", "pft_assignment_submissions",
        "pft_test_submissions", "pft_total_minutes", "n_wifi_days", "resource_events",
        "n_resource_days",
    ]
    monthly = ["%s_%s" % (family, month) for family in families for month in ("m09", "m10", "m11", "m12")]
    base = list(CATEGORICAL_FEATURES) + ["base_%02d" % index for index in range(39)]
    names = base + monthly
    return {
        "feature_names": names,
        "numeric_features": [name for name in names if name not in CATEGORICAL_FEATURES],
        "categorical_features": list(CATEGORICAL_FEATURES),
    }


def _frame(schema, n=80):
    rng = np.random.RandomState(42)
    frame = pd.DataFrame(index=range(n))
    for index, name in enumerate(schema["feature_names"]):
        if name in schema["categorical_features"]:
            frame[name] = np.where(np.arange(n) % 2, "a", "b")
        else:
            frame[name] = rng.poisson(3, n).astype(float)
    frame["base_00"] = np.linspace(0, 1, n)
    return frame


def test_regime_counts_and_raw_policy():
    schema = _schema()
    train = _frame(schema)
    policies = build_constraint_policy(train, schema)
    assert [policies[name]["n_actionable_features"] for name in ("STRICT", "MODERATE", "FLEXIBLE")] == [16, 28, 36]
    query = train.iloc[[0]].copy()
    policy = policy_for_query(policies["STRICT"], query, train, schema)
    assert all(policy["features"][name]["direction"] == "immutable" for name in schema["feature_names"] if name not in policies["STRICT"]["actionable_features"])


def test_label_mapping_is_explicit_and_model_verifier_is_raw_space():
    schema = _schema()
    train = _frame(schema, 100)
    y = (train["base_00"] > 0.5).astype(int).to_numpy()
    bundle, warnings = fit_model(train, schema, y)
    assert not warnings
    policies = build_constraint_policy(train, schema)
    query = train.iloc[[0]].copy()
    policy = policy_for_query(policies["FLEXIBLE"], query, train, schema)
    candidate = query.copy()
    candidate["pft_events_m09"] = candidate["pft_events_m09"] + 1
    result = verify_candidate(query, candidate, bundle, schema, policy)
    assert result.raw_reconstruction_ok
    assert result.immutable_valid
    assert result.bounds_valid
    assert isinstance(result.constraint_feasible, bool)


def test_verifier_detects_immutable_bound_direction_domain_and_schema_violations():
    schema = _schema()
    train = _frame(schema, 100)
    y = (train["base_00"] > 0.5).astype(int).to_numpy()
    bundle, _ = fit_model(train, schema, y)
    policies = build_constraint_policy(train, schema)
    query = train.iloc[[0]].copy()
    policy = policy_for_query(policies["STRICT"], query, train, schema)

    immutable = query.copy()
    immutable["base_01"] = immutable["base_01"] + 99
    result = verify_candidate(query, immutable, bundle, schema, policy)
    assert not result.immutable_valid
    assert any(item.startswith("immutable_violation") for item in result.violations)

    actionable = policies["STRICT"]["actionable_features"][0]
    bounded = query.copy()
    bounded[actionable] = bounded[actionable] + 999999
    result = verify_candidate(query, bounded, bundle, schema, policy)
    assert not result.bounds_valid

    reversed_direction = query.copy()
    reversed_direction[actionable] = float(reversed_direction[actionable].iloc[0]) - 1
    result = verify_candidate(query, reversed_direction, bundle, schema, policy)
    assert not result.direction_valid or not result.bounds_valid

    invalid_domain = query.copy()
    invalid_domain["campus_hash"] = "not-a-domain-value"
    result = verify_candidate(query, invalid_domain, bundle, schema, policy)
    assert not result.categorical_domain_valid

    missing = query.drop(columns=[schema["feature_names"][0]])
    result = verify_candidate(query, missing, bundle, schema, policy)
    assert not result.raw_reconstruction_ok
    assert "schema_violation" in result.violations


def test_terminal_status_precedence_and_ci_contract():
    assert status_from_results({"runtime_error": True}, []) == "RUNTIME_ERROR"
    assert status_from_results({"timeout": True}, []) == "TIMEOUT"
    assert status_from_results({"unsupported": True}, []) == "UNSUPPORTED"
    assert status_from_results({}, []) == "NO_RETURNED_CF"
    assert set(TERMINAL_STATUSES) == {
        "RUNTIME_ERROR", "TIMEOUT", "UNSUPPORTED", "NO_RETURNED_CF",
        "RETURNED_NON_FLIPPING_CF", "SUCCESS_VALID_CONSTRAINT_FAILED", "SUCCESS_VALID_FEASIBLE",
    }
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(5, 10)
    assert 0 < low < 0.5 < high < 1



def test_dedicacion_round_trip_raw_constraints_and_prediction_parity():
    from scripts.external_binary_eval.adapters import EncodedModel, EncodedSpace

    schema = _schema()
    train = _frame(schema, 100)
    train["dedicacion"] = pd.Series(np.where(np.arange(len(train)) % 2, "TP", "TC"), dtype="string")
    y = (train["dedicacion"] == "TP").astype(int).to_numpy()
    bundle, _ = fit_model(train, schema, y)
    space = EncodedSpace(train, schema)
    raw = train.iloc[:8].copy()
    round_trip = space.decode(space.encode(raw))
    assert round_trip["dedicacion"].astype(str).tolist() == raw["dedicacion"].astype(str).tolist()
    encoded_model = EncodedModel(bundle, space)
    assert np.array_equal(encoded_model.predict(space.encode(raw)), bundle.predict_raw(raw))

    policies = build_constraint_policy(train, schema)
    query = train.iloc[[0]].copy()
    raw_policy = policy_for_query(policies["MODERATE"], query, train, schema)
    raw_policy["features"]["dedicacion"] = {
        "immutable": False,
        "allowed_values": ["TC", "TP"],
        "direction": "categorical_transition",
    }
    candidate = query.copy()
    candidate["dedicacion"] = "TP"
    class CategoryTarget:
        def predict_raw(self, frame):
            return (frame["dedicacion"].astype(str) == "TP").astype(int).to_numpy()

    target_bundle = CategoryTarget()
    verified = verify_candidate(query, candidate, target_bundle, schema, raw_policy)
    assert verified.raw_reconstruction_ok
    assert verified.categorical_domain_valid
    assert verified.immutable_valid
    assert verified.target_valid
    assert verified.constraint_feasible

    reverse_query = train.iloc[[1]].copy()
    reverse_policy = policy_for_query(policies["MODERATE"], reverse_query, train, schema)
    reverse_policy["features"]["dedicacion"] = {
        "immutable": False,
        "allowed_values": ["TC", "TP"],
        "direction": "categorical_transition",
    }
    reverse_candidate = reverse_query.copy()
    reverse_candidate["dedicacion"] = "TC"
    reverse_verified = verify_candidate(reverse_query, reverse_candidate, target_bundle, schema, reverse_policy)
    assert reverse_verified.categorical_domain_valid
    assert reverse_verified.immutable_valid

    missing_query = query.copy()
    missing_query["dedicacion"] = pd.NA
    missing_policy = policy_for_query(policies["MODERATE"], missing_query, train, schema)
    missing_candidate = missing_query.copy()
    missing_candidate["dedicacion"] = "TP"
    missing_verified = verify_candidate(missing_query, missing_candidate, target_bundle, schema, missing_policy)
    assert not missing_verified.immutable_valid
    assert any(item.startswith("immutable_violation:dedicacion") for item in missing_verified.violations)

    invalid = candidate.copy()
    invalid["dedicacion"] = "not-a-category"
    invalid_verified = verify_candidate(query, invalid, target_bundle, schema, raw_policy)
    assert not invalid_verified.categorical_domain_valid

    immutable_category = query.copy()
    immutable_category["campus_hash"] = "a" if str(query.iloc[0]["campus_hash"]) != "a" else "b"
    immutable_verified = verify_candidate(query, immutable_category, target_bundle, schema, raw_policy)
    assert not immutable_verified.immutable_valid


def test_ufce_ff1_ff2_ff3_try_the_other_category_in_both_directions():
    from ufce.ufce_ff.ufce import UFCE

    class CategoryFlipModel:
        def predict(self, frame):
            values = frame["dedicacion"].to_numpy(dtype=float)
            anchor = frame["anchor"].to_numpy(dtype=float)
            return (values != anchor).astype(int)

    class PairFlipModel:
        def predict(self, frame):
            changed = frame["dedicacion"].to_numpy(dtype=float) != frame["anchor"].to_numpy(dtype=float)
            return (changed & (frame["digital"].to_numpy(dtype=float) >= 1)).astype(int)

    class TripleFlipModel:
        def predict(self, frame):
            changed = frame["dedicacion"].to_numpy(dtype=float) != frame["anchor"].to_numpy(dtype=float)
            return (
                changed
                & (frame["digital"].to_numpy(dtype=float) >= 1)
                & (frame["extra"].to_numpy(dtype=float) >= 1)
            ).astype(int)

    uf = UFCE(radius=100.0)
    order = ["dedicacion", "digital", "extra", "anchor"]
    intervals = {"dedicacion": [0, 1], "digital": [0, 2], "extra": [0, 2]}
    for factual_code, alternate_code in ((0.0, 1.0), (1.0, 0.0)):
        query = pd.DataFrame([{
            "dedicacion": factual_code,
            "digital": 0.0,
            "extra": 0.0,
            "anchor": factual_code,
        }])
        single = uf.Single_F(
            query, ["dedicacion"], {"dedicacion": [0, 1]},
            CategoryFlipModel(), 1, {},
        )
        assert len(single) == 1
        assert float(single.iloc[0]["dedicacion"]) == alternate_code

        double, _ = uf.Double_F(
            query, query, [], [["dedicacion", "digital"]],
            ["dedicacion"], order, intervals, order, PairFlipModel(), 1, order, 5,
        )
        assert len(double) >= 1
        assert float(double.iloc[0]["dedicacion"]) == alternate_code
        assert float(double.iloc[0]["digital"]) >= 1

        triple, _ = uf.Triple_F(
            query, query, [], [["dedicacion", "digital"]],
            ["dedicacion"], order, intervals,
            ["dedicacion", "digital", "extra"], TripleFlipModel(), 1, order, 5,
        )
        assert len(triple) >= 1
        assert float(triple.iloc[0]["dedicacion"]) == alternate_code
        assert float(triple.iloc[0]["digital"]) >= 1
        assert float(triple.iloc[0]["extra"]) >= 1


def test_full_mi_ranking_and_paired_bootstrap_join_are_deterministic():
    from scripts.external_binary_eval.adapters import rank_mi_pairs
    from scripts.external_binary_eval.author_alignment import _paired_row

    reference = pd.DataFrame({
        "a": [0, 1, 2, 3, 4, 5, 6, 7],
        "b": [0, 2, 4, 6, 8, 10, 12, 14],
        "c": [7, 1, 4, 2, 6, 3, 5, 0],
    })
    ranked, summary = rank_mi_pairs(reference, ["a", "b", "c"], seed=0)
    assert len(ranked) == 3
    assert summary["mi_estimator_calls"] == 3
    assert ranked[0]["left_feature"] == "a"
    assert ranked[0]["right_feature"] == "b"

    left = pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "returned_any_valid_cf": [True, False, True],
        "returned_any_feasible_cf": [True, False, False],
        "runtime_ms": [10.0, 20.0, 30.0],
    })
    right = pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "returned_any_valid_cf": [False, False, True],
        "returned_any_feasible_cf": [False, False, False],
        "runtime_ms": [15.0, 25.0, 35.0],
    })
    paired = _paired_row(left, right, "unit", "left", "right")
    assert paired["n_queries"] == 3
    assert np.isclose(paired["delta_valid_availability_percentage_points"], 100.0 / 3.0)
    assert paired["mcnemar"]["b_left_only"] == 1


def test_full_reference_radius_fast_path_returns_every_row_in_source_order():
    from ufce.ufce_ff.ufce import UFCE

    n_rows = 10001
    reference = pd.DataFrame({
        "a": np.arange(n_rows, dtype=float),
        "b": np.arange(n_rows - 1, -1, -1, dtype=float),
    }, index=np.arange(n_rows, dtype=np.int64) * 2)
    query = reference.iloc[[0]].copy()
    index = UFCE(radius=1e9)
    assert index.cache_reference_bounds(reference)

    neighbors, indices, metadata = index.NNkdtree(reference, query, return_meta=True)

    assert neighbors.reset_index(drop=True).equals(reference.reset_index(drop=True))
    assert np.array_equal(indices, np.arange(n_rows))
    assert metadata["backend"] == "exact_radius_superset_scan"
    assert metadata["upper_bound_max_distance"] < metadata["radius"]
    assert index._kdtree_cache == {}

def test_dice_proxies_only_unseen_immutable_categories_then_restores_raw_values():
    from scripts.external_binary_eval.adapters import DiceAdapter

    schema = _schema()
    raw = _frame(schema, 8).iloc[[0]].copy()
    raw["dedicacion"] = "TC"
    raw["campus_hash"] = "outside_10k"
    raw["base_00"] = np.nan
    actionable = "pft_events_m09"

    class FakeExplainer:
        def generate_counterfactuals(self, query, **kwargs):
            self.query = query.copy()
            self.kwargs = kwargs
            candidate = query.copy()
            candidate.loc[candidate.index[0], actionable] = float(candidate.iloc[0][actionable]) + 1.0
            candidate.loc[candidate.index[0], "dedicacion"] = "TP"
            class Case:
                final_cfs_df = candidate
            class Explanation:
                cf_examples_list = [Case()]
            return Explanation()

    adapter = object.__new__(DiceAdapter)
    adapter.dice_ml = object()
    adapter.feature_names = list(schema["feature_names"])
    adapter.feature_schema = schema
    adapter.actionable = [actionable]
    adapter.categorical_actionable = ["dedicacion"]
    adapter.numeric_imputations = {name: 7.0 for name in schema["numeric_features"]}
    adapter.categorical_imputations = {name: "b" for name in schema["categorical_features"]}
    adapter.reference_categorical_domains = {
        name: {str(raw.iloc[0][name])} for name in schema["categorical_features"]
    }
    adapter.reference_categorical_domains["campus_hash"] = {"only_in_reference"}
    adapter.explainer = FakeExplainer()

    features = {name: {"immutable": True} for name in schema["feature_names"]}
    features[actionable] = {"immutable": False, "lower": 0.0, "upper": 20.0}
    features["dedicacion"] = {
        "immutable": False, "allowed_values": ["TC", "TP"],
    }
    policy = {"features": features}
    result = adapter.generate(raw, raw_constraint_policy=policy, max_candidates=5, seed=0)

    assert result.candidate_space == "raw"
    assert len(result.candidates) == 1
    assert adapter.explainer.query.iloc[0]["campus_hash"] == "only_in_reference"
    assert result.candidates.iloc[0]["campus_hash"] == "outside_10k"
    assert pd.isna(result.candidates.iloc[0]["base_00"])
    assert result.candidates.iloc[0]["dedicacion"] == "TP"
    assert actionable in adapter.explainer.kwargs["features_to_vary"]
    assert adapter.explainer.kwargs["permitted_range"]["dedicacion"] == ["TC", "TP"]
def test_ar_raw_binary_dedication_equation_matches_frozen_pipeline_both_directions():
    from scripts.external_binary_eval.adapters import ARAdapter

    schema = {
        "feature_names": ["x", "dedicacion"],
        "numeric_features": ["x"],
        "categorical_features": ["dedicacion"],
    }
    train = pd.DataFrame({
        "x": np.linspace(0.0, 1.0, 100),
        "dedicacion": np.where(np.arange(100) % 2, "TP", "TC"),
    })
    y = ((train["x"] > 0.70) | (train["dedicacion"] == "TP")).astype(int).to_numpy()
    bundle, warnings = fit_model(train, schema, y)
    assert not warnings

    adapter = object.__new__(ARAdapter)
    adapter.bundle = bundle
    adapter.feature_names = list(schema["feature_names"])
    adapter.numeric_features = list(schema["numeric_features"])
    adapter.category_feature = "dedicacion"

    for factual, alternate in (("TC", "TP"), ("TP", "TC")):
        query = pd.DataFrame({"x": [0.25], "dedicacion": [factual]})
        coefficients, intercept, vector = adapter._raw_linear_model(query, ["dedicacion"])
        factual_score = float(bundle.model.decision_function(bundle._prepare(query))[0])
        changed = query.copy()
        changed.loc[0, "dedicacion"] = alternate
        alternate_score = float(bundle.model.decision_function(bundle._prepare(changed))[0])
        assert np.isclose(intercept + float(np.dot(coefficients, vector)), factual_score, atol=1e-10)
        assert np.isclose(intercept + coefficients[0] * (1.0 - vector[0]), alternate_score, atol=1e-10)
