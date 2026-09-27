"""UFCE-FF and explicit-target DiCE adapters for the same frozen model."""

import itertools
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from ufce.ufce_ff import cfmethods
from ufce.ufce_ff.ufce import UFCE

from .config import (
    ACTIONABLE_FEATURES,
    DICE_SAMPLE_SIZE,
    MAX_CANDIDATES,
    MI_TOP_K,
    TIMEOUT_SECONDS,
    UFCE_NEIGHBORS,
    UFCE_RADIUS,
)
from .space import EncodedSpace
from .timeout import QueryTimeout
from .verification import verify_candidate


@dataclass
class GenerationResult:
    exposed_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_summary: dict = field(default_factory=dict)
    generation_seconds: float = 0.0
    error_type: str = ""
    error_detail: str = ""
    schema_errors: list = field(default_factory=list)


class EncodedModel:
    def __init__(self, bundle, space):
        self.bundle = bundle
        self.space = space
        self.classes_ = np.asarray(bundle.model.named_steps["model"].classes_)

    def _raw(self, frame):
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame, columns=self.space.feature_names)
        return self.space.decode(frame.loc[:, self.space.feature_names])

    def predict(self, frame):
        return self.bundle.predict_raw(self._raw(frame))

    def predict_proba(self, frame):
        return self.bundle.predict_proba_raw(self._raw(frame))


class CandidateLedger:
    def __init__(self, space, bundle, schema, factual_raw, factual_encoded, query_policy, source_class, target_class):
        self.space = space
        self.bundle = bundle
        self.schema = schema
        self.factual_raw = factual_raw
        self.factual_encoded = factual_encoded
        self.query_policy = query_policy
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.factual_key = tuple(factual_encoded.iloc[0][name] for name in space.feature_names)
        self.seen = set()
        self.counts = {"candidate_count": 0, "target_count": 0, "source_count": 0, "off_target_count": 0, "feasible_count": 0}
        self.schema_errors = []

    def observe(self, row, prediction):
        key = tuple(row[name] for name in self.space.feature_names)
        if key == self.factual_key or key in self.seen:
            return
        self.seen.add(key)
        self.counts["candidate_count"] += 1
        if int(prediction) == self.target_class:
            self.counts["target_count"] += 1
            raw = self.space.decode(pd.DataFrame([row], columns=self.space.feature_names))
            verification = verify_candidate(
                self.factual_raw, raw, self.bundle, self.schema,
                self.query_policy, self.source_class, self.target_class,
            )
            if verification.constraint_feasible:
                self.counts["feasible_count"] += 1
        elif int(prediction) == self.source_class:
            self.counts["source_count"] += 1
        else:
            self.counts["off_target_count"] += 1


class RecordingModel:
    """Transparent prediction proxy that records proposals before UFCE-FF gating."""

    def __init__(self, model, feature_names, ledger):
        self.model = model
        self.feature_names = list(feature_names)
        self.classes_ = model.classes_
        self.ledger = ledger
        self.schema_errors = self.ledger.schema_errors

    def _frame(self, value):
        if isinstance(value, pd.DataFrame):
            candidate = value.copy()
            if all(name in candidate.columns for name in self.feature_names):
                candidate = candidate.loc[:, self.feature_names]
            elif len(candidate.columns) != len(self.feature_names):
                self.schema_errors.append("candidate_prediction_schema_mismatch")
                return None
            else:
                candidate.columns = self.feature_names
            return candidate.reset_index(drop=True)
        array = np.asarray(value)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            self.schema_errors.append("candidate_prediction_schema_mismatch")
            return None
        return pd.DataFrame(array, columns=self.feature_names)

    def _record(self, frame, predictions):
        if frame is None:
            return
        values = np.asarray(predictions).reshape(-1)
        if len(values) != len(frame):
            self.schema_errors.append("candidate_prediction_length_mismatch")
            return
        for index, prediction in enumerate(values):
            self.ledger.observe(frame.iloc[index], int(prediction))

    def predict(self, value):
        frame = self._frame(value)
        predictions = self.model.predict(value)
        self._record(frame, predictions)
        return predictions

    def predict_proba(self, value):
        frame = self._frame(value)
        probabilities = self.model.predict_proba(value)
        if frame is not None:
            predictions = self.classes_[np.argmax(np.asarray(probabilities), axis=1)]
            self._record(frame, predictions)
        return probabilities

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def candidate_summary(self):
        return dict(self.ledger.counts)


def compute_mi_pairs(train_encoded, seed=42):
    rows = []
    for left, right in itertools.combinations(ACTIONABLE_FEATURES, 2):
        try:
            score = float(mutual_info_regression(
                train_encoded[[left]], train_encoded[right], random_state=seed
            )[0])
        except Exception:
            score = 0.0
        rows.append((score, left, right))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [[left, right] for _score, left, right in rows[:MI_TOP_K]]


class BaseAdapter:
    def __init__(self, train_raw, train_encoded, bundle, schema, policy, mi_pairs=None):
        self.train_raw = train_raw.reset_index(drop=True)
        self.train_encoded = train_encoded.reset_index(drop=True)
        self.bundle = bundle
        self.schema = schema
        self.policy = policy
        self.space = EncodedSpace(train_raw, schema)
        self.feature_names = list(schema["feature_names"])
        self.actionable = list(ACTIONABLE_FEATURES)
        self.mi_pairs = list(mi_pairs or [])
        self.model = EncodedModel(bundle, self.space)
        self.setup_seconds = 0.0

    def _candidate_validator(self, factual_raw, query_policy, source_class, target_class):
        def validate(encoded_candidates):
            if not isinstance(encoded_candidates, pd.DataFrame) or encoded_candidates.empty:
                return pd.DataFrame(columns=self.feature_names)
            raw = self.space.decode(encoded_candidates.loc[:, self.feature_names])
            keep = []
            for position in range(len(raw)):
                result = verify_candidate(
                    factual_raw, raw.iloc[[position]], self.bundle, self.schema,
                    query_policy, source_class, target_class,
                )
                if result.constraint_feasible:
                    keep.append(position)
            return encoded_candidates.iloc[keep].reset_index(drop=True)
        return validate


class UFCEAdapter(BaseAdapter):
    def __init__(self, method, train_raw, train_encoded, train_predictions, bundle, schema, policy, mi_pairs):
        started = time.perf_counter()
        super().__init__(train_raw, train_encoded, bundle, schema, policy, mi_pairs)
        self.method = method
        self.desired_space = {
            class_id: train_encoded.loc[np.asarray(train_predictions) == int(class_id)].reset_index(drop=True)
            for class_id in range(3)
        }
        self.desired_space_positions = {
            class_id: np.flatnonzero(np.asarray(train_predictions) == int(class_id))
            for class_id in range(3)
        }
        self.center = train_encoded[self.actionable].mean(axis=0)
        self.scale = train_encoded[self.actionable].std(axis=0, ddof=0).replace(0.0, 1.0)
        self.distance_space = (train_encoded[self.actionable] - self.center) / self.scale
        cfmethods.initUFCE(radius=UFCE_RADIUS, n_neighbors=min(UFCE_NEIGHBORS, max(1, len(train_encoded) - 1)))
        self.setup_seconds = time.perf_counter() - started

    def generate(
        self, factual_raw, source_class, target_class, query_policy,
        max_candidates=MAX_CANDIDATES, seed=0,
    ):
        started = time.perf_counter()
        factual_raw = factual_raw.loc[:, self.feature_names].reset_index(drop=True)
        factual_encoded = self.space.encode(factual_raw)
        target_reference = self.desired_space[int(target_class)]
        if target_reference.empty:
            return GenerationResult(generation_seconds=time.perf_counter() - started)

        bounds = {
            feature: (int(rule["lower"]), int(rule["upper"]))
            for feature, rule in query_policy["features"].items()
        }
        steps = {feature: 1 for feature in self.actionable}
        target_positions = self.desired_space_positions[int(target_class)]
        target_distance = self.distance_space.iloc[target_positions].reset_index(drop=True)
        query_distance = ((factual_encoded[self.actionable] - self.center) / self.scale).reset_index(drop=True)
        if len(target_distance) != len(target_reference):
            raise ValueError("desired_space target rows do not align with train predictions")

        ledger = CandidateLedger(
            self.space, self.bundle, self.schema, factual_raw, factual_encoded,
            query_policy, source_class, target_class,
        )
        recorder = RecordingModel(self.model, self.feature_names, ledger)
        validator = self._candidate_validator(factual_raw, query_policy, source_class, target_class)
        protected = [name for name in self.feature_names if name not in self.actionable]
        random.seed(seed)
        np.random.seed(seed)
        try:
            if self.method == "UFCE-FF1":
                result = cfmethods.sfexp(
                    self.train_encoded, target_reference, factual_encoded, bounds, steps,
                    self.actionable, self.actionable, [], recorder, int(target_class),
                    max_candidates, self.feature_names, return_stats=True,
                    distance_data_lab1=target_distance, distance_X_test=query_distance,
                    candidate_validator=validator,
                )
            elif self.method == "UFCE-FF2":
                result = cfmethods.dfexp(
                    self.train_encoded, target_reference, factual_encoded, bounds, self.mi_pairs,
                    self.actionable, [], self.actionable, protected, recorder,
                    int(target_class), max_candidates, self.feature_names,
                    return_stats=True, distance_data_lab1=target_distance,
                    distance_X_test=query_distance, candidate_validator=validator,
                )
            else:
                result = cfmethods.tfexp(
                    self.train_encoded, target_reference, factual_encoded, bounds, self.mi_pairs,
                    self.actionable, [], self.actionable, protected, recorder,
                    int(target_class), max_candidates, self.feature_names,
                    return_stats=True, distance_data_lab1=target_distance,
                    distance_X_test=query_distance, candidate_validator=validator,
                )
            selected = result[0] if isinstance(result, tuple) else pd.DataFrame()
            if not isinstance(selected, pd.DataFrame):
                selected = pd.DataFrame()
            if not selected.empty and all(name in selected.columns for name in self.feature_names):
                selected = validator(selected.loc[:, self.feature_names])
            else:
                selected = pd.DataFrame(columns=self.feature_names)
            return GenerationResult(
                exposed_candidates=selected.loc[:, self.feature_names].reset_index(drop=True),
                candidate_summary=recorder.candidate_summary,
                generation_seconds=time.perf_counter() - started,
                schema_errors=list(recorder.schema_errors),
            )
        except QueryTimeout:
            raise
        except Exception as exc:
            return GenerationResult(
                candidate_summary=recorder.candidate_summary,
                generation_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                error_detail=str(exc),
                schema_errors=list(recorder.schema_errors),
            )


class DiCEAdapter(BaseAdapter):
    def __init__(self, train_raw, train_encoded, train_predictions, bundle, schema, policy, mi_pairs=None):
        started = time.perf_counter()
        super().__init__(train_raw, train_encoded, bundle, schema, policy, mi_pairs)
        try:
            import dice_ml
            self.dice_ml = dice_ml
            train_for_dice = train_encoded.copy()
            train_for_dice["__target__"] = np.asarray(train_predictions, dtype=int)
            data = dice_ml.Data(
                dataframe=train_for_dice,
                # Every categorical input is immutable in this phase. Expose
                # the numeric surrogate codes as fixed continuous columns to
                # DiCE so a query category absent from train stays fixed and
                # does not create an out-of-vocabulary one-hot column.
                continuous_features=list(schema["feature_names"]),
                categorical_features=[],
                outcome_name="__target__",
            )
            model = dice_ml.Model(model=self.model, backend="sklearn")
            self.explainer = dice_ml.Dice(data, model, method="random")
        except QueryTimeout:
            raise
        except Exception as exc:
            self.explainer = None
            self.setup_error = "%s: %s" % (type(exc).__name__, exc)
        self.setup_seconds = time.perf_counter() - started

    def generate(
        self, factual_raw, source_class, target_class, query_policy,
        max_candidates=MAX_CANDIDATES, seed=0,
    ):
        started = time.perf_counter()
        if self.explainer is None:
            return GenerationResult(
                generation_seconds=time.perf_counter() - started,
                error_type="dice_setup_error",
                error_detail=getattr(self, "setup_error", "DiCE setup failed"),
            )
        factual_raw = factual_raw.loc[:, self.feature_names].reset_index(drop=True)
        factual_encoded = self.space.encode(factual_raw)
        permitted = {
            feature: [float(rule["lower"]), float(rule["upper"])]
            for feature, rule in query_policy["features"].items()
        }
        ledger = CandidateLedger(
            self.space, self.bundle, self.schema, factual_raw, factual_encoded,
            query_policy, source_class, target_class,
        )
        recorder = RecordingModel(self.model, self.feature_names, ledger)
        # Replace the wrapped model for this query so the prediction ledger
        # observes the full multiclass DiCE search while preserving the same
        # classifier and preprocessing.
        self.explainer.model.model = recorder
        random.seed(seed)
        np.random.seed(seed)
        try:
            explanation = self.explainer.generate_counterfactuals(
                factual_encoded,
                total_CFs=int(max_candidates),
                desired_class=int(target_class),
                features_to_vary=list(self.actionable),
                permitted_range=permitted,
                sample_size=int(DICE_SAMPLE_SIZE),
                random_seed=int(seed),
            )
            candidates = pd.DataFrame(columns=self.feature_names)
            if explanation is not None and explanation.cf_examples_list:
                final = explanation.cf_examples_list[0].final_cfs_df
                if final is not None and not final.empty:
                    final = final.drop(columns=["__target__"], errors="ignore")
                    if all(name in final.columns for name in self.feature_names):
                        candidates = final.loc[:, self.feature_names].head(max_candidates).reset_index(drop=True)
                    else:
                        recorder.schema_errors.append("dice_candidate_schema_mismatch")
            validated = []
            for position in range(len(candidates)):
                raw = self.space.decode(candidates.iloc[[position]])
                verification = verify_candidate(
                    factual_raw, raw, self.bundle, self.schema,
                    query_policy, source_class, target_class,
                )
                if verification.constraint_feasible:
                    validated.append(candidates.iloc[position])
            exposed = pd.DataFrame(validated, columns=self.feature_names).reset_index(drop=True)
            return GenerationResult(
                exposed_candidates=exposed,
                candidate_summary=recorder.candidate_summary,
                generation_seconds=time.perf_counter() - started,
                schema_errors=list(recorder.schema_errors),
            )
        except Exception as exc:
            return GenerationResult(
                candidate_summary=recorder.candidate_summary,
                generation_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                error_detail=str(exc),
                schema_errors=list(recorder.schema_errors),
            )
