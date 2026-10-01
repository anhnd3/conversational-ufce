import numpy as np
import pandas as pd

from ufce.ufce_ff.cfmethods import _filter_flipping_candidates
from ufce.ufce_ff.ufce import UFCE

from .config import ACTIONABLE_FEATURES, TRANSITIONS
from .dataset import split_dataset
from .metrics import summarize_transition
from .model import fit_model
from .policy import audit_invariants
from .space import EncodedSpace
from .verification import verify_candidate


class GradeModel:
    def predict_raw(self, frame):
        return np.rint(pd.to_numeric(frame[ACTIONABLE_FEATURES[2]])).astype(int).to_numpy()

    predict = predict_raw


def _schema_and_policy():
    schema = {
        "feature_names": list(ACTIONABLE_FEATURES) + ["fixed_flag"],
        "numeric_features": list(ACTIONABLE_FEATURES),
        "categorical_features": ["fixed_flag"],
    }
    policy = {
        "features": {
            name: {"lower": 0.0, "upper": 2.0, "integer": True}
            for name in ACTIONABLE_FEATURES
        },
        "immutable_features": ["fixed_flag"],
        "categorical_domains": {"fixed_flag": ["a", "b"]},
        "invariants_enforced": [],
    }
    return schema, policy


def _factual(class_id):
    row = {name: 1.0 for name in ACTIONABLE_FEATURES}
    row[ACTIONABLE_FEATURES[2]] = float(class_id)
    row["fixed_flag"] = "a"
    return pd.DataFrame([row])


def _candidate(class_id):
    row = _factual(0).iloc[0].to_dict()
    row[ACTIONABLE_FEATURES[2]] = float(class_id)
    return pd.DataFrame([row])


def test_native_three_class_verifier_covers_all_six_transitions():
    schema, policy = _schema_and_policy()
    model = GradeModel()
    for source in range(3):
        for target in range(3):
            if source == target:
                continue
            result = verify_candidate(
                _factual(source), _candidate(target), model, schema, policy, source, target
            )
            assert result.schema_valid
            assert result.prediction == target
            assert result.target_valid
            assert result.constraint_feasible


def test_off_target_and_immutable_or_invariant_violations_are_detected():
    schema, policy = _schema_and_policy()
    model = GradeModel()
    off_target = verify_candidate(
        _factual(0), _candidate(2), model, schema, policy, 0, 1
    )
    assert not off_target.target_valid
    assert "target_class_violation" in off_target.violations

    changed_immutable = _candidate(1)
    changed_immutable.loc[0, "fixed_flag"] = "b"
    result = verify_candidate(
        _factual(0), changed_immutable, model, schema, policy, 0, 1
    )
    assert not result.immutable_valid

    constrained_schema = dict(schema)
    constrained_schema["feature_names"] = list(schema["feature_names"]) + [
        "Curricular units 1st sem (enrolled)"
    ]
    constrained_schema["numeric_features"] = list(constrained_schema["feature_names"][:-1])
    constrained_schema["categorical_features"] = ["fixed_flag"]
    constraint_policy = dict(policy)
    constraint_policy["immutable_features"] = ["fixed_flag", "Curricular units 1st sem (enrolled)"]
    constraint_policy["invariants_enforced"] = ["approved_plus_without_eval_le_enrolled_1st_sem"]
    factual = _factual(0)
    factual["Curricular units 1st sem (enrolled)"] = 5.0
    candidate = _candidate(1)
    candidate["Curricular units 1st sem (enrolled)"] = 5.0
    candidate.loc[0, "Curricular units 1st sem (approved)"] = 4.0
    candidate.loc[0, "Curricular units 1st sem (without evaluations)"] = 2.0
    result = verify_candidate(
        factual, candidate, model, constrained_schema, constraint_policy, 0, 1
    )
    assert not result.invariant_valid
    assert any(item.startswith("invariant_violation:") for item in result.violations)


def test_ufce_ff_candidate_gate_accepts_only_explicit_target_and_validator():
    schema, _policy = _schema_and_policy()
    candidates = pd.concat([_candidate(0), _candidate(1), _candidate(2)], ignore_index=True)
    selected = _filter_flipping_candidates(
        candidates,
        GradeModel(),
        desired_outcome=2,
        order=schema["feature_names"],
        candidate_validator=lambda frame: frame.iloc[1:],
    )
    # The target row is kept by the prediction gate; the supplied hard gate
    # then rejects it. This checks ordering and fail-closed behavior.
    assert selected.empty

    selected = _filter_flipping_candidates(
        candidates, GradeModel(), desired_outcome=2, order=schema["feature_names"]
    )
    assert selected[ACTIONABLE_FEATURES[2]].tolist() == [2.0]


def test_signed_bounds_are_supported_by_ufce_interval_builders():
    ufce = UFCE()
    neighbors = pd.DataFrame({"score": [1, 2, 3], "other": [2, 3, 4]})
    factual = pd.DataFrame({"score": [2], "other": [3]})
    intervals = ufce.make_intervals(
        neighbors, {"score": (0, 3)}, ["score"], factual
    )
    assert intervals["score"] == [1, 3]
    pair_intervals = ufce.make_uf_nn_interval(
        neighbors, {"score": (0, 3), "other": (1, 5)}, [["score", "other"]], factual
    )
    assert pair_intervals == {"score": [1, 3], "other": [2, 4]}


def test_split_is_disjoint_stratified_70_15_15():
    features = pd.DataFrame({"x": np.arange(300)})
    labels = np.repeat([0, 1, 2], 100)
    splits = split_dataset(features, labels, seed=42)
    assert [len(splits[key]) for key in ("train", "dev", "test")] == [210, 45, 45]
    assert all(len(set(labels[index] for index in splits[key])) == 3 for key in splits)
    assert not (set(splits["train"]) & set(splits["dev"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["dev"]) & set(splits["test"]))


def test_only_train_supported_semantic_invariants_are_enforced():
    data = {}
    for semester in ("1st sem", "2nd sem"):
        data["Curricular units %s (approved)" % semester] = [1, 2, 3]
        data["Curricular units %s (evaluations)" % semester] = [1, 2, 3]
        data["Curricular units %s (without evaluations)" % semester] = [1, 0, 1]
        data["Curricular units %s (enrolled)" % semester] = [3, 3, 5]
    policy_audit = audit_invariants(pd.DataFrame(data))
    assert len(policy_audit) == 8
    assert all(item["enforced"] for item in policy_audit)

    data["Curricular units 2nd sem (approved)"] = [1, 6, 3]
    policy_audit = audit_invariants(pd.DataFrame(data))
    failed = [item for item in policy_audit if not item["enforced"]]
    assert failed
    assert all(item["train_violations"] > 0 for item in failed)


def test_unseen_immutable_category_roundtrips_and_model_scores_it():
    schema = {
        "feature_names": ["score", "sector"],
        "numeric_features": ["score"],
        "categorical_features": ["sector"],
    }
    train = pd.DataFrame({
        "score": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
        "sector": ["A", "A", "A", "B", "B", "B"],
    })
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    bundle, _warnings = fit_model(train, labels, schema)
    space = EncodedSpace(train, schema)

    unseen = pd.DataFrame({"score": [1.5], "sector": ["never-seen-in-train"]})
    encoded = space.encode(unseen)
    decoded = space.decode(encoded)

    assert encoded["sector"].notna().all()
    assert decoded.loc[0, "sector"] == "never-seen-in-train"
    assert bundle.predict_raw(decoded).shape == (1,)

    malformed = encoded.copy()
    malformed.loc[0, "sector"] = 99.5
    assert space.decode(malformed).loc[0, "sector"] == "__UFCE_INVALID_CATEGORY__"


def test_exposed_validity_and_constraint_rates_use_all_exposed_cfs_and_empty_is_na():
    transition = TRANSITIONS[0].key
    queries = pd.DataFrame({
        "method": ["UFCE-FF1", "UFCE-FF1"],
        "transition": [transition, transition],
        "query_id": ["q1", "q2"],
        "target_valid_availability": [True, False],
        "constraint_feasible_availability": [True, False],
        "runtime_ms": [100.0, 200.0],
        "setup_ms": [10.0, 10.0],
    })
    candidates = pd.DataFrame({
        "method": ["UFCE-FF1", "UFCE-FF1"],
        "transition": [transition, transition],
        "candidate_count": [3, 0],
        "target_count": [1, 0],
        "off_target_count": [1, 0],
        "feasible_count": [1, 0],
    })
    exposed = pd.DataFrame({
        "method": ["UFCE-FF1", "UFCE-FF1"],
        "transition": [transition, transition],
        "target_valid": [True, False],
        "constraint_feasible": [True, False],
    })
    summary = summarize_transition(queries, candidates, exposed, "UFCE-FF1", transition)
    assert summary["exposed_cf_target_validity_numerator"] == 1
    assert summary["exposed_cf_target_validity_denominator"] == 2
    assert summary["exposed_cf_constraint_satisfaction_numerator"] == 1
    assert summary["exposed_cf_constraint_satisfaction_denominator"] == 2

    empty = summarize_transition(
        queries, candidates, exposed.iloc[0:0], "UFCE-FF1", transition
    )
    assert empty["exposed_cf_target_validity"] is None
    assert empty["exposed_cf_constraint_satisfaction"] is None
