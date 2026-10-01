import numpy as np
import pandas as pd

from .dice_variants import CostRecordingModel, ReferenceAwareLedger
from .space import EncodedSpace
from .test_native_multiclass import GradeModel, _candidate, _factual, _schema_and_policy
from .timeout import QueryTimeout


class _Bundle:
    def predict_raw(self, frame):
        return GradeModel().predict_raw(frame)


class _EncodedGradeModel:
    classes_ = np.asarray([0, 1, 2])

    def predict(self, frame):
        return GradeModel().predict_raw(frame)

    def predict_proba(self, frame):
        labels = self.predict(frame)
        result = np.zeros((len(labels), 3), dtype=float)
        result[np.arange(len(labels)), labels] = 1.0
        return result


def _ledger_fixture():
    schema, train_policy = _schema_and_policy()
    train_raw = pd.concat([_candidate(0), _candidate(1), _candidate(2)], ignore_index=True)
    space = EncodedSpace(train_raw, schema)
    train_encoded = space.encode(train_raw)
    factual = _factual(0)
    factual_encoded = space.encode(factual)
    query_policy = {
        "features": {
            name: {"lower": 0.0, "upper": 2.0, "integer": True}
            for name in schema["numeric_features"]
        },
        "immutable_features": ["fixed_flag"],
        "categorical_domains": {"fixed_flag": ["a", "b"]},
        "invariants_enforced": [],
    }
    ledger = ReferenceAwareLedger(
        space, _Bundle(), schema, factual, factual_encoded, query_policy, 0, 1,
        train_encoded=train_encoded, train_predictions=np.asarray([0, 1, 2]),
    )
    recorder = CostRecordingModel(_EncodedGradeModel(), schema["feature_names"], ledger)
    return schema, train_encoded, ledger, recorder


def test_full_reference_prediction_is_costed_but_not_counted_as_cf_proposals():
    _schema, train_encoded, ledger, recorder = _ledger_fixture()
    predictions = recorder.predict(train_encoded.astype(np.float32))
    assert predictions.tolist() == [0, 1, 2]
    assert recorder.predict_calls == 1
    assert recorder.prediction_rows == 3
    assert ledger.counts["candidate_count"] == 0
    # DiCE 0.7.2 probes the model on a single training row while constructing
    # the KD-tree explainer; float32 should still be recognized as reference.
    recorder.predict(train_encoded.iloc[[1]].astype(np.float32))
    assert ledger.counts["candidate_count"] == 0


def test_kdtree_pool_rows_are_counted_when_they_are_considered_as_cf_proposals():
    _schema, train_encoded, ledger, _recorder = _ledger_fixture()
    ledger.observe_reference_proposal(train_encoded.iloc[1].astype(np.float32))
    assert ledger.counts["candidate_count"] == 1
    assert ledger.counts["target_count"] == 1
    assert ledger.counts["off_target_count"] == 0


def test_off_target_proposals_and_ledger_survive_query_timeout():
    schema, _train_encoded, ledger, recorder = _ledger_fixture()
    candidate = _candidate(2)
    candidate.loc[0, schema["numeric_features"][0]] = 2.0
    try:
        recorder.predict(candidate)
        raise QueryTimeout()
    except QueryTimeout:
        pass
    assert recorder.predict_calls == 1
    assert ledger.counts["candidate_count"] == 1
    assert ledger.counts["off_target_count"] == 1
