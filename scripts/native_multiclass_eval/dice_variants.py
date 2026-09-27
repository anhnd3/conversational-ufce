"""Post-hoc full-test comparison of DiCE genetic and KD-tree backends.

This runner reuses the frozen native-multiclass model/split and writes into a
new output directory. It never edits the canonical UFCE/DiCE-random results.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .adapters import CandidateLedger, EncodedModel
from .config import (
    ACTIONABLE_FEATURES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CHECKPOINT_INTERVAL_QUERIES,
    CLASS_NAMES,
    MAX_CANDIDATES,
    PILOT_COUNTS,
    PILOT_SEED,
    SPLIT_SEED,
    TIMEOUT_SECONDS,
    TRANSITIONS,
)
from .dataset import load_dataset, sha256_file, split_dataset, write_json
from .metrics import paired_bootstrap, wilson
from .policy import build_train_policy, policy_for_query
from .runner import _select_final_queries, _select_pilot_queries
from .space import EncodedSpace
from .timeout import QueryTimeout, query_timeout
from .verification import verify_candidate


METHODS = ("DiCE-genetic", "DiCE-kdtree")
BASELINE_METHODS = ("UFCE-FF1", "UFCE-FF2", "UFCE-FF3", "DiCE")
BASELINE_DIR = Path("outputs/native_multiclass_eval/student_outcomes_20260924_resume1")
DEFAULT_OUT_DIR = Path("outputs/native_multiclass_eval/student_outcomes_dice_variants_20260925_retry2")
GENETIC_OPTIONS = {"initialization": "kdtree", "maxiterations": 500}


def _canonical_hash(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def _key(row, names):
    return tuple(row[name] for name in names)


def _canonical_key(row, names):
    """Normalize numeric dtypes so DiCE float32 rows match float64 train rows."""
    normalized = []
    for name in names:
        value = row[name]
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            normalized.append(round(number, 6))
        else:
            normalized.append(str(value))
    return tuple(normalized)


class ReferenceAwareLedger(CandidateLedger):
    """Exclude DiCE's full reference-label pass from candidate proposals."""

    def __init__(self, *args, train_encoded, train_predictions, **kwargs):
        super().__init__(*args, **kwargs)
        reference_frame = train_encoded.loc[:, self.space.feature_names].reset_index(drop=True)
        self.reference_matrix = reference_frame.to_numpy(dtype=float)
        self.reference_keys = {
            _canonical_key(row, self.space.feature_names)
            for row in reference_frame.to_dict("records")
        }
        self.reference_predictions = {
            _canonical_key(row, self.space.feature_names): int(prediction)
            for row, prediction in zip(
                train_encoded.loc[:, self.space.feature_names].to_dict("records"),
                np.asarray(train_predictions, dtype=int),
            )
        }

    def observe(self, row, prediction):
        if _canonical_key(row, self.space.feature_names) in self.reference_keys:
            return
        super().observe(row, prediction)

    def observe_reference_proposal(self, row):
        """Count a train row only when the KD-tree actually considers it."""
        prediction = self.reference_predictions.get(_canonical_key(row, self.space.feature_names))
        if prediction is not None:
            super().observe(row, prediction)

    def is_full_reference_batch(self, frame):
        if len(frame) != len(self.reference_keys):
            return False
        try:
            values = frame.loc[:, self.space.feature_names].to_numpy(dtype=float)
        except (TypeError, ValueError):
            return False
        # DiCE 0.7.2 casts continuous inputs to float32 before querying the
        # model, whereas EncodedSpace stores float64. Compare ordered rows with
        # a tolerance so reference-label inference is not counted as proposals.
        return bool(np.allclose(values, self.reference_matrix, rtol=1e-6, atol=1e-6))


class CostRecordingModel:
    """Count classifier calls/rows and record non-reference proposals."""

    def __init__(self, model, feature_names, ledger):
        self.model = model
        self.feature_names = list(feature_names)
        self.ledger = ledger
        self.classes_ = model.classes_
        self.predict_calls = 0
        self.predict_proba_calls = 0
        self.prediction_rows = 0
        self.schema_errors = ledger.schema_errors

    def _frame(self, value):
        if isinstance(value, pd.DataFrame):
            frame = value.copy()
            if all(name in frame.columns for name in self.feature_names):
                frame = frame.loc[:, self.feature_names]
            elif len(frame.columns) == len(self.feature_names):
                frame.columns = self.feature_names
            else:
                self.schema_errors.append("candidate_prediction_schema_mismatch")
                return None
            return frame.reset_index(drop=True)
        values = np.asarray(value)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            self.schema_errors.append("candidate_prediction_schema_mismatch")
            return None
        return pd.DataFrame(values, columns=self.feature_names)

    def _record(self, frame, predictions):
        if frame is None or self.ledger.is_full_reference_batch(frame):
            return
        labels = np.asarray(predictions).reshape(-1)
        if len(labels) != len(frame):
            self.schema_errors.append("candidate_prediction_length_mismatch")
            return
        for position, prediction in enumerate(labels):
            self.ledger.observe(frame.iloc[position], int(prediction))

    def predict(self, value):
        frame = self._frame(value)
        self.predict_calls += 1
        self.prediction_rows += 0 if frame is None else len(frame)
        result = self.model.predict(value)
        self._record(frame, result)
        return result

    def predict_proba(self, value):
        frame = self._frame(value)
        self.predict_proba_calls += 1
        self.prediction_rows += 0 if frame is None else len(frame)
        result = self.model.predict_proba(value)
        if frame is not None and not self.ledger.is_full_reference_batch(frame):
            labels = self.classes_[np.argmax(np.asarray(result), axis=1)]
            self._record(frame, labels)
        return result

    def __getattr__(self, name):
        return getattr(self.model, name)


class DiceVariantAdapter:
    def __init__(self, method, train_raw, train_encoded, train_predictions, bundle, schema, policy):
        if method not in METHODS:
            raise ValueError("Unsupported supplementary method: %s" % method)
        self.method = method
        self.train_raw = train_raw.reset_index(drop=True)
        self.train_encoded = train_encoded.reset_index(drop=True)
        self.train_predictions = np.asarray(train_predictions, dtype=int)
        self.bundle = bundle
        self.schema = schema
        self.policy = policy
        self.feature_names = list(schema["feature_names"])
        self.space = EncodedSpace(train_raw, schema)
        self.base_model = EncodedModel(bundle, self.space)
        self.setup_seconds = 0.0
        started = time.perf_counter()
        import dice_ml

        self.dice_ml = dice_ml
        dice_frame = self.train_encoded.loc[:, self.feature_names].copy()
        dice_frame["__target__"] = self.train_predictions
        self.data = dice_ml.Data(
            dataframe=dice_frame,
            continuous_features=self.feature_names,
            categorical_features=[],
            outcome_name="__target__",
        )
        self.setup_seconds = time.perf_counter() - started

    def _record_kdtree_pool(self, explainer, query_instance, total_cfs, ledger):
        """Record the exact nearest-neighbour pool inspected by DiCE KD."""
        original = explainer.vary_valid

        def wrapped(kd_query_instance, requested, features_to_vary, permitted_range,
                    factual, sparsity_weight):
            tree = getattr(explainer, "KD_tree", None)
            reference = getattr(explainer, "dataset_with_predictions", None)
            if tree is not None and reference is not None and len(reference):
                pool_size = min(len(reference), int(requested) * 10)
                indexes = tree.query(kd_query_instance, pool_size)[1][0]
                for index in indexes:
                    row = reference.iloc[int(index)].loc[self.feature_names]
                    ledger.observe_reference_proposal(row)
            return original(
                kd_query_instance, requested, features_to_vary, permitted_range,
                factual, sparsity_weight,
            )

        explainer.vary_valid = wrapped

    def generate(self, factual_raw, source_class, target_class, query_policy, seed=42):
        factual_raw = factual_raw.loc[:, self.feature_names].reset_index(drop=True)
        factual_encoded = self.space.encode(factual_raw)
        ledger = ReferenceAwareLedger(
            self.space, self.bundle, self.schema, factual_raw, factual_encoded,
            query_policy, source_class, target_class,
            train_encoded=self.train_encoded, train_predictions=self.train_predictions,
        )
        recorder = CostRecordingModel(self.base_model, self.feature_names, ledger)
        permitted = {
            name: [float(rule["lower"]), float(rule["upper"])]
            for name, rule in query_policy["features"].items()
        }
        generation_status = "OK"
        error_type = ""
        error_detail = ""
        exposed = []
        explainer = None
        random.seed(int(seed))
        np.random.seed(int(seed))
        started = time.perf_counter()
        try:
            with query_timeout(TIMEOUT_SECONDS):
                model = self.dice_ml.Model(model=recorder, backend="sklearn")
                method_name = self.method.split("-", 1)[1]
                explainer = self.dice_ml.Dice(self.data, model, method=method_name)
                if method_name == "kdtree":
                    self._record_kdtree_pool(explainer, factual_encoded, MAX_CANDIDATES, ledger)
                kwargs = {
                    "desired_class": int(target_class),
                    "features_to_vary": list(ACTIONABLE_FEATURES),
                    "permitted_range": permitted,
                    "stopping_threshold": 0.5,
                    "posthoc_sparsity_param": 0.1,
                    "posthoc_sparsity_algorithm": "linear",
                    "verbose": False,
                }
                if method_name == "genetic":
                    kwargs.update(GENETIC_OPTIONS)
                explanation = explainer.generate_counterfactuals(
                    factual_encoded,
                    total_CFs=MAX_CANDIDATES,
                    **kwargs
                )
                candidates = pd.DataFrame(columns=self.feature_names)
                if explanation is not None and explanation.cf_examples_list:
                    final = explanation.cf_examples_list[0].final_cfs_df
                    if final is not None and not final.empty:
                        final = final.drop(columns=["__target__", "__target___pred"], errors="ignore")
                        if all(name in final.columns for name in self.feature_names):
                            candidates = final.loc[:, self.feature_names].head(MAX_CANDIDATES)
                        else:
                            ledger.schema_errors.append("dice_candidate_schema_mismatch")
                for position in range(len(candidates)):
                    encoded = candidates.iloc[[position]].reset_index(drop=True)
                    raw = self.space.decode(encoded)
                    prediction = int(self.bundle.predict_raw(raw)[0])
                    ledger.observe_reference_proposal(encoded.iloc[0])
                    # Non-reference proposals were already counted by the model recorder.
                    if _canonical_key(encoded.iloc[0], self.feature_names) not in ledger.reference_keys:
                        ledger.observe(encoded.iloc[0], prediction)
                    verification = verify_candidate(
                        factual_raw, raw, self.bundle, self.schema, query_policy,
                        int(source_class), int(target_class),
                    )
                    if verification.target_valid and verification.constraint_feasible:
                        exposed.append(encoded.iloc[0])
        except QueryTimeout:
            generation_status = "TIMEOUT"
        except Exception as exc:
            generation_status = "RUNTIME_ERROR"
            error_type = type(exc).__name__
            error_detail = str(exc)
        if ledger.schema_errors and generation_status == "OK":
            generation_status = "SCHEMA_ERROR"
        exposed_frame = pd.DataFrame(exposed, columns=self.feature_names).reset_index(drop=True)
        return {
            "status": generation_status,
            "error_type": "QueryTimeout" if generation_status == "TIMEOUT" else error_type,
            "error_detail": error_detail,
            "exposed_candidates": exposed_frame,
            "candidate_summary": dict(ledger.counts),
            "schema_errors": list(ledger.schema_errors),
            "prediction_calls": int(recorder.predict_calls + recorder.predict_proba_calls),
            "prediction_rows": int(recorder.prediction_rows),
            "predict_calls": int(recorder.predict_calls),
            "predict_proba_calls": int(recorder.predict_proba_calls),
            "runtime_ms": float((time.perf_counter() - started) * 1000.0),
        }


def _load_frozen_context(root, baseline_dir):
    baseline_dir = Path(baseline_dir)
    if not baseline_dir.is_absolute():
        baseline_dir = root / baseline_dir
    if not baseline_dir.is_dir():
        raise FileNotFoundError("Frozen baseline directory missing: %s" % baseline_dir)
    run_status = json.loads((baseline_dir / "run_status.json").read_text(encoding="utf-8"))
    if run_status.get("status") != "COMPLETE":
        raise ValueError("Canonical native multiclass run must be COMPLETE before supplementary evaluation")
    freeze = json.loads((baseline_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    frozen = freeze["configuration"]
    data_path = root / "data/native_multiclass_eval/uci_students_697.zip"
    features, labels, schema, dataset_summary = load_dataset(data_path, auto_download=False)
    if dataset_summary["source_sha256"] != frozen["dataset"]["source_sha256"]:
        raise ValueError("Dataset checksum differs from frozen baseline")
    model_path = baseline_dir / "model_bundle.joblib"
    if sha256_file(model_path) != frozen["model_bundle_sha256"]:
        raise ValueError("Model bundle checksum differs from frozen baseline")
    saved_schema = json.loads((baseline_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if saved_schema != schema:
        raise ValueError("Feature schema differs from frozen baseline")
    splits = split_dataset(features, labels, SPLIT_SEED)
    expected_split = pd.DataFrame([
        {"row_id": int(row_id), "partition": part, "target_class": int(labels[row_id])}
        for part, row_ids in splits.items() for row_id in row_ids
    ]).sort_values("row_id").reset_index(drop=True)
    saved_split = pd.read_csv(baseline_dir / "split_manifest.csv").sort_values("row_id").reset_index(drop=True)
    if not expected_split.equals(saved_split):
        raise ValueError("Reconstructed split differs from frozen baseline")
    bundle = joblib.load(model_path)
    train_rows, dev_rows, test_rows = splits["train"], splits["dev"], splits["test"]
    train_raw = features.iloc[train_rows].reset_index(drop=True)
    train_policy = build_train_policy(train_raw, schema)
    saved_policy = json.loads((baseline_dir / "constraint_policy.json").read_text(encoding="utf-8"))
    if _canonical_hash(train_policy) != _canonical_hash(saved_policy):
        raise ValueError("Reconstructed constraint policy differs from frozen baseline")
    train_predictions = bundle.predict_raw(train_raw)
    space = EncodedSpace(train_raw, schema)
    train_encoded = space.encode(train_raw)
    dev_predictions = bundle.predict_raw(features.iloc[dev_rows].reset_index(drop=True))
    test_features = features.iloc[test_rows].reset_index(drop=True)
    test_predictions = bundle.predict_raw(test_features)
    expected_queries = _select_final_queries(test_rows, test_predictions)
    saved_queries = pd.read_csv(baseline_dir / "final_test_queries.csv")
    key_columns = ["query_id", "row_id", "source_class", "target_class", "transition"]
    if not expected_queries[key_columns].equals(saved_queries[key_columns]):
        raise ValueError("Reconstructed final test queries differ from frozen manifest")
    baseline_results = pd.read_csv(baseline_dir / "final_test_query_results.csv")
    expected_keys = set(saved_queries["query_id"].astype(str))
    if set(baseline_results["method"].unique()) != set(BASELINE_METHODS):
        raise ValueError("Canonical result is missing one or more frozen baseline methods")
    if len(baseline_results) != len(saved_queries) * len(BASELINE_METHODS):
        raise ValueError("Canonical result row count does not match frozen query manifest")
    for method in BASELINE_METHODS:
        rows = baseline_results[baseline_results["method"] == method]
        if set(rows["query_id"].astype(str)) != expected_keys or rows["query_id"].duplicated().any():
            raise ValueError("Baseline method coverage is incomplete: %s" % method)
    if baseline_results["status"].isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).any():
        raise ValueError("Canonical baseline contains runtime or schema errors")
    dice_version = importlib.metadata.version("dice-ml")
    frozen_dice_version = frozen.get("versions", {}).get("dice-ml")
    if dice_version != frozen_dice_version:
        raise ValueError("Installed dice-ml version differs from frozen baseline: %s != %s" % (dice_version, frozen_dice_version))
    pilot_queries = _select_pilot_queries(dev_rows, dev_predictions)
    return {
        "baseline_dir": baseline_dir,
        "features": features,
        "labels": labels,
        "schema": schema,
        "dataset_summary": dataset_summary,
        "bundle": bundle,
        "splits": splits,
        "train_raw": train_raw,
        "train_encoded": train_encoded,
        "train_predictions": train_predictions,
        "train_policy": train_policy,
        "dev_predictions": dev_predictions,
        "test_predictions": test_predictions,
        "test_queries": saved_queries,
        "pilot_queries": pilot_queries,
        "baseline_results": baseline_results,
        "freeze": freeze,
    }


def _query_policy_dict(train_policy, factual, features, schema, query):
    return policy_for_query(train_policy, factual)


def _run_method_queries(method, queries, context, stage, setup_seconds, adapter=None, setup_error=None):
    features = context["features"]
    schema = context["schema"]
    query_rows = []
    candidate_rows = []
    exposed_rows = []
    if adapter is None and setup_error is None:
        try:
            adapter = DiceVariantAdapter(
                method, context["train_raw"], context["train_encoded"],
                context["train_predictions"], context["bundle"], schema,
                context["train_policy"],
            )
            setup_seconds += adapter.setup_seconds
        except Exception as exc:
            setup_error = exc
    for query in queries.to_dict("records"):
        factual = features.iloc[[int(query["row_id"])]].loc[:, schema["feature_names"]].reset_index(drop=True)
        query_policy = _query_policy_dict(context["train_policy"], factual, features, schema, query)
        try:
            if adapter is None:
                raise RuntimeError("DiCE adapter setup failed: %s" % setup_error)
            result = adapter.generate(
                factual, int(query["source_class"]), int(query["target_class"]),
                query_policy, seed=SPLIT_SEED,
            )
        except Exception as exc:
            result = {
                "status": "RUNTIME_ERROR", "error_type": type(exc).__name__,
                "error_detail": str(exc), "exposed_candidates": pd.DataFrame(columns=schema["feature_names"]),
                "candidate_summary": {"candidate_count": 0, "target_count": 0, "source_count": 0,
                                      "off_target_count": 0, "feasible_count": 0},
                "schema_errors": [], "prediction_calls": 0, "prediction_rows": 0,
                "predict_calls": 0, "predict_proba_calls": 0, "runtime_ms": 0.0,
            }
        summary = result["candidate_summary"]
        candidates = result["exposed_candidates"]
        query_id = str(query["query_id"])
        candidate_rows.append({
            "stage": stage, "query_id": query_id, "method": method,
            "transition": query["transition"],
            "candidate_count": int(summary["candidate_count"]),
            "target_count": int(summary["target_count"]),
            "source_count": int(summary["source_count"]),
            "off_target_count": int(summary["off_target_count"]),
            "feasible_count": int(summary["feasible_count"]),
            "proposal_count": int(summary["candidate_count"]),
            "model_prediction_calls": int(result["prediction_calls"]),
            "model_prediction_rows": int(result["prediction_rows"]),
            "predict_calls": int(result["predict_calls"]),
            "predict_proba_calls": int(result["predict_proba_calls"]),
            "schema_errors": ";".join(result["schema_errors"]),
        })
        verification_rows = []
        for rank in range(len(candidates)):
            encoded = candidates.iloc[[rank]].loc[:, schema["feature_names"]]
            raw = adapter.space.decode(encoded)
            verified = verify_candidate(
                factual, raw, context["bundle"], schema, query_policy,
                int(query["source_class"]), int(query["target_class"]),
            )
            verification_rows.append({
                "stage": stage, "query_id": query_id, "method": method,
                "transition": query["transition"], "exposed_rank": int(rank),
                "target_valid": bool(verified.target_valid),
                "constraint_feasible": bool(verified.constraint_feasible),
                "prediction": verified.prediction,
                "violations": ";".join(verified.violations),
                **{"feature_%s" % name: value for name, value in verified.raw_candidate.items()},
            })
        exposed_rows.extend(verification_rows)
        query_rows.append({
            "stage": stage, "query_id": query_id, "row_id": int(query["row_id"]),
            "method": method, "transition": query["transition"],
            "source_class": int(query["source_class"]), "target_class": int(query["target_class"]),
            "target_valid_availability": bool(summary["target_count"] > 0),
            "constraint_feasible_availability": bool(summary["feasible_count"] > 0),
            "returned_any_cf": bool(verification_rows),
            "exposed_target_valid": bool(verification_rows and all(row["target_valid"] for row in verification_rows)),
            "exposed_constraint_satisfied": bool(verification_rows and all(row["constraint_feasible"] for row in verification_rows)),
            "candidate_count": int(summary["candidate_count"]),
            "target_candidate_count": int(summary["target_count"]),
            "off_target_candidate_count": int(summary["off_target_count"]),
            "feasible_candidate_count": int(summary["feasible_count"]),
            "exposed_cf_count": int(len(verification_rows)),
            "runtime_ms": float(result["runtime_ms"]),
            "setup_ms": float(setup_seconds * 1000.0),
            "model_prediction_calls": int(result["prediction_calls"]),
            "model_prediction_rows": int(result["prediction_rows"]),
            "status": result["status"], "error_type": result["error_type"],
            "error_detail": result["error_detail"],
            "schema_error_count": int(len(result["schema_errors"])),
        })
    exposed_columns = [
        "stage", "query_id", "method", "transition", "exposed_rank",
        "target_valid", "constraint_feasible", "prediction", "violations",
    ] + ["feature_%s" % name for name in schema["feature_names"]]
    return (pd.DataFrame(query_rows), pd.DataFrame(candidate_rows),
            pd.DataFrame(exposed_rows, columns=exposed_columns))


def _summarize_transition(query_results, candidate_summaries, exposed_results, method, transition):
    queries = query_results[(query_results.method == method) & (query_results.transition == transition)]
    candidates = candidate_summaries[(candidate_summaries.method == method) & (candidate_summaries.transition == transition)]
    exposed = exposed_results[(exposed_results.method == method) & (exposed_results.transition == transition)]
    n = len(queries)
    candidate_total = int(candidates.candidate_count.sum()) if len(candidates) else 0
    off_target = int(candidates.off_target_count.sum()) if len(candidates) else 0
    target_valid = int(queries.target_valid_availability.fillna(False).sum()) if n else 0
    feasible = int(queries.constraint_feasible_availability.fillna(False).sum()) if n else 0
    exposed_n = len(exposed)
    exposed_valid = int(exposed.target_valid.fillna(False).sum()) if exposed_n else 0
    exposed_feasible = int(exposed.constraint_feasible.fillna(False).sum()) if exposed_n else 0
    runtime = pd.to_numeric(queries.runtime_ms, errors="coerce").fillna(0.0)
    setup_ms = float(queries.setup_ms.iloc[0]) if n else 0.0
    return {
        "method": method, "transition": transition, "query_count": int(n),
        "target_valid_availability": wilson(target_valid, n)["estimate"],
        "target_valid_availability_numerator": target_valid,
        "target_valid_availability_denominator": n,
        "target_valid_availability_ci95_low": wilson(target_valid, n)["ci_low"],
        "target_valid_availability_ci95_high": wilson(target_valid, n)["ci_high"],
        "constraint_feasible_availability": wilson(feasible, n)["estimate"],
        "constraint_feasible_availability_numerator": feasible,
        "constraint_feasible_availability_denominator": n,
        "constraint_feasible_availability_ci95_low": wilson(feasible, n)["ci_low"],
        "constraint_feasible_availability_ci95_high": wilson(feasible, n)["ci_high"],
        "off_target_rate": wilson(off_target, candidate_total)["estimate"],
        "off_target_numerator": off_target, "off_target_denominator": candidate_total,
        "exposed_cf_target_validity": wilson(exposed_valid, exposed_n)["estimate"],
        "exposed_cf_target_validity_numerator": exposed_valid,
        "exposed_cf_target_validity_denominator": exposed_n,
        "exposed_cf_constraint_satisfaction": wilson(exposed_feasible, exposed_n)["estimate"],
        "exposed_cf_constraint_satisfaction_numerator": exposed_feasible,
        "exposed_cf_constraint_satisfaction_denominator": exposed_n,
        "median_latency_ms": float(runtime.median()) if n else None,
        "p95_latency_ms": float(runtime.quantile(0.95)) if n else None,
        "timeout_count": int((queries.status == "TIMEOUT").sum()) if n else 0,
        "setup_ms": setup_ms,
        "effective_compute_cost_ms_per_feasible_cf": (
            (float(runtime.sum()) + setup_ms) / exposed_feasible if exposed_feasible else None
        ),
        "model_prediction_calls": int(queries.model_prediction_calls.sum()) if n else 0,
        "model_prediction_rows": int(queries.model_prediction_rows.sum()) if n else 0,
    }


def _paired_reports(combined):
    rows = []
    methods = list(METHODS)
    pairs = [(base, variant) for base in BASELINE_METHODS[:3] for variant in methods]
    pairs.extend([("DiCE", variant) for variant in methods])
    pairs.append((METHODS[0], METHODS[1]))
    for transition in [item.key for item in TRANSITIONS]:
        for left, right in pairs:
            for metric in ("target_valid_availability", "constraint_feasible_availability"):
                result = paired_bootstrap(combined, transition, left, right, metric)
                rows.append(result)
    return pd.DataFrame(rows)


def _pilot_gate(queries, candidates, exposed, pilot_manifest):
    expected_ids = set(pilot_manifest.query_id.astype(str))
    errors = int(queries.status.isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).sum())
    schema_errors = int(queries.schema_error_count.fillna(0).sum())
    completeness = len(queries) == len(expected_ids) * len(METHODS)
    coverage = all(
        len(queries[(queries.method == method) & (queries.transition == transition.key)])
        == int(PILOT_COUNTS[transition.key])
        for method in METHODS for transition in TRANSITIONS
    )
    ledger_complete = len(candidates) == len(queries) and {
        "proposal_count", "off_target_count", "model_prediction_calls"
    }.issubset(candidates.columns)
    verifier_present = "constraint_feasible" in exposed.columns
    return {
        "passed": bool(completeness and coverage and ledger_complete and verifier_present and errors == 0 and schema_errors == 0),
        "checks": {
            "all_six_transitions_and_expected_denominators": bool(coverage),
            "40_method_query_rows": bool(completeness),
            "proposal_and_model_call_ledgers_present": bool(ledger_complete),
            "verifier_ran": bool(verifier_present),
            "no_runtime_or_schema_errors": errors == 0,
            "no_silent_schema_errors": schema_errors == 0,
            "timeouts_are_recorded_results": bool(queries.status.isin(["OK", "TIMEOUT"]).all()),
            "availability_denominators_present": bool(
                queries.target_valid_availability.notna().all()
                and queries.constraint_feasible_availability.notna().all()
            ),
        },
        "query_rows": int(len(queries)), "candidate_rows": int(len(candidates)),
        "exposed_rows": int(len(exposed)), "runtime_error_count": errors,
        "schema_error_count": schema_errors,
    }


def _write_stage(out_dir, stage, queries, candidates, exposed):
    _atomic_csv(queries, out_dir / (stage + "_query_results.csv"))
    _atomic_csv(candidates, out_dir / (stage + "_candidate_summaries.csv"))
    _atomic_csv(exposed, out_dir / (stage + "_exposed_candidates.csv"))
    summary = pd.DataFrame([
        _summarize_transition(queries, candidates, exposed, method, transition.key)
        for transition in TRANSITIONS for method in METHODS
    ])
    _atomic_csv(summary, out_dir / (stage + "_transition_summary.csv"))
    return summary


def _run_stage(stage, manifest, context, out_dir):
    query_path = out_dir / (stage + "_query_results.partial.csv")
    candidate_path = out_dir / (stage + "_candidate_summaries.partial.csv")
    exposed_path = out_dir / (stage + "_exposed_candidates.partial.csv")
    if query_path.exists():
        all_queries = pd.read_csv(query_path)
        all_candidates = pd.read_csv(candidate_path)
        all_exposed = pd.read_csv(exposed_path)
    else:
        all_queries = pd.DataFrame()
        all_candidates = pd.DataFrame()
        all_exposed = pd.DataFrame()
    expected_keys = {(method, str(qid)) for method in METHODS for qid in manifest.query_id.astype(str)}
    completed = set(zip(all_queries.get("method", []), all_queries.get("query_id", []).astype(str) if len(all_queries) else []))
    if not completed.issubset(expected_keys) or all_queries.duplicated(["method", "query_id"]).any():
        raise ValueError("Partial checkpoint contains invalid or duplicate method/query keys")
    if len(all_queries):
        candidate_keys = set(zip(all_candidates.method.astype(str), all_candidates.query_id.astype(str)))
        query_keys = set(zip(all_queries.method.astype(str), all_queries.query_id.astype(str)))
        if candidate_keys != query_keys or len(all_candidates) != len(all_queries):
            raise ValueError("Partial candidate ledger does not match completed query rows")
        if all_exposed.get("target_valid", pd.Series(dtype=bool)).eq(False).any() or all_exposed.get("constraint_feasible", pd.Series(dtype=bool)).eq(False).any():
            raise ValueError("Partial output contains a target-invalid or infeasible exposed CF")
    total = len(manifest) * len(METHODS)
    for transition in TRANSITIONS:
        block = manifest[manifest.transition == transition.key].reset_index(drop=True)
        for method in METHODS:
            done = set(
                all_queries.loc[(all_queries.method == method) & (all_queries.transition == transition.key), "query_id"].astype(str)
            ) if len(all_queries) else set()
            pending = block.loc[~block.query_id.astype(str).isin(done)].reset_index(drop=True)
            if pending.empty:
                continue
            setup_error = None
            try:
                adapter = DiceVariantAdapter(
                    method, context["train_raw"], context["train_encoded"],
                    context["train_predictions"], context["bundle"], context["schema"],
                    context["train_policy"],
                )
                setup_seconds = adapter.setup_seconds
            except Exception as exc:
                adapter = None
                setup_seconds = 0.0
                setup_error = exc
            # Reuse the transition-level data preparation, but each query gets
            # a fresh DiCE explainer inside adapter.generate.
            for query in pending.to_dict("records"):
                one, cand, exp = _run_method_queries(
                    method, pd.DataFrame([query]), context, stage, setup_seconds,
                    adapter=adapter, setup_error=setup_error
                )
                # Each call creates a fresh DiCE explainer while reusing the
                # read-only Data interface for this transition/method block.
                one.loc[:, "setup_ms"] = setup_seconds * 1000.0
                all_queries = pd.concat([all_queries, one], ignore_index=True)
                all_candidates = pd.concat([all_candidates, cand], ignore_index=True)
                all_exposed = pd.concat([all_exposed, exp], ignore_index=True)
                completed_rows = len(all_queries)
                if completed_rows % CHECKPOINT_INTERVAL_QUERIES == 0 or completed_rows == total:
                    _atomic_csv(all_queries, query_path)
                    _atomic_csv(all_candidates, candidate_path)
                    _atomic_csv(all_exposed, exposed_path)
                    write_json(out_dir / (stage + "_progress.json"), {
                        "stage": stage, "transition": query["transition"],
                        "method": method, "last_query_id": query["query_id"],
                        "completed_method_query_rows": int(completed_rows),
                        "expected_method_query_rows": int(total),
                        "checkpoint_interval_queries": CHECKPOINT_INTERVAL_QUERIES,
                        "updated_at_unix": time.time(),
                    })
                    print("[progress] %s %s %s %d/%d" % (
                        stage, method, transition.key, completed_rows, total
                    ), flush=True)
    _atomic_csv(all_queries, query_path)
    _atomic_csv(all_candidates, candidate_path)
    _atomic_csv(all_exposed, exposed_path)
    summary = _write_stage(out_dir, stage, all_queries, all_candidates, all_exposed)
    return all_queries, all_candidates, all_exposed, summary


def run_supplement(baseline_dir=BASELINE_DIR, out_dir=DEFAULT_OUT_DIR):
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    context = _load_frozen_context(root, baseline_dir)
    config = {
        "kind": "posthoc_dice_multiclass_backend_supplement",
        "baseline_dir": str(context["baseline_dir"]),
        "baseline_final_results_sha256": sha256_file(context["baseline_dir"] / "final_test_query_results.csv"),
        "dataset_sha256": context["dataset_summary"]["source_sha256"],
        "model_sha256": sha256_file(context["baseline_dir"] / "model_bundle.joblib"),
        "freeze_configuration_sha256": context["freeze"]["configuration_sha256"],
        "methods": list(METHODS), "dice_ml_version": importlib.metadata.version("dice-ml"),
        "max_exposed_cfs": MAX_CANDIDATES, "timeout_seconds": TIMEOUT_SECONDS,
        "seed": SPLIT_SEED, "genetic_options": GENETIC_OPTIONS,
        "default_stopping_threshold": 0.5, "posthoc_sparsity_param": 0.1,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "pilot_counts": PILOT_COUNTS, "final_query_target_pairs": int(len(context["test_queries"])),
        "posthoc": True,
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [
                Path(__file__), Path(__file__).with_name("adapters.py"),
                Path(__file__).with_name("config.py"), Path(__file__).with_name("verification.py"),
                Path(__file__).with_name("policy.py"), Path(__file__).with_name("space.py"),
            ]
        },
    }
    config["configuration_sha256"] = _canonical_hash(config)
    config_path = out_dir / "supplement_manifest.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != config["configuration_sha256"]:
            raise ValueError("Output directory belongs to a different supplementary configuration")
    else:
        write_json(config_path, config)

    pilot_queries, pilot_candidates, pilot_exposed, pilot_summary = _run_stage(
        "pilot", context["pilot_queries"], context, out_dir
    )
    gate = _pilot_gate(pilot_queries, pilot_candidates, pilot_exposed, context["pilot_queries"])
    write_json(out_dir / "pilot_gate.json", gate)
    if not gate["passed"]:
        status = {"status": "STOP_PILOT_GATE", "finished_at_unix": time.time()}
        write_json(out_dir / "run_status.json", status)
        return status
    write_json(out_dir / "supplement_freeze.json", {
        "frozen": True, "frozen_at_unix": time.time(),
        "configuration": config,
        "pilot_gate": gate,
        "pilot_query_results_sha256": sha256_file(out_dir / "pilot_query_results.csv"),
    })

    final_queries, final_candidates, final_exposed, final_summary = _run_stage(
        "final_test", context["test_queries"], context, out_dir
    )
    combined = pd.concat([context["baseline_results"], final_queries], ignore_index=True)
    comparisons = _paired_reports(combined)
    _atomic_csv(comparisons, out_dir / "final_test_paired_comparisons.csv")
    status_counts = final_queries.groupby(["method", "transition", "status"]).size().reset_index(name="count")
    exposed_failures = int((~final_exposed.target_valid.fillna(False)).sum()) + int((~final_exposed.constraint_feasible.fillna(False)).sum()) if len(final_exposed) else 0
    errors = int(final_queries.status.isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).sum())
    timeout_count = int((final_queries.status == "TIMEOUT").sum())
    run_status = "COMPLETE" if errors == 0 and exposed_failures == 0 else "FAILED_EXPOSURE_OR_RUNTIME_GATE"
    if run_status == "COMPLETE" and timeout_count:
        run_status = "COMPLETE_WITH_TIMEOUTS"
    report = {
        "status": run_status, "posthoc_supplement": True,
        "baseline_dir": str(context["baseline_dir"]),
        "baseline_final_results_sha256": config["baseline_final_results_sha256"],
        "final_test_query_target_pairs": int(len(context["test_queries"])),
        "new_method_query_rows": int(len(final_queries)),
        "transition_summary": final_summary.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
        "query_status_counts": status_counts.to_dict(orient="records"),
        "runtime_or_schema_error_count": errors, "timeout_count": timeout_count,
        "exposed_target_or_constraint_failures": exposed_failures,
        "search_budget_note": (
            "All methods return at most five CFs and share a 60-second per-query timeout. "
            "DiCE random uses a 1000-draw sample_size; genetic uses up to 500 iterations "
            "with a 50-member population; kdtree searches the training reference. These "
            "internal search budgets are not equivalent."
        ),
    }
    write_json(out_dir / "final_report.json", report)
    write_json(out_dir / "run_status.json", {"status": run_status, "finished_at_unix": time.time()})
    return {"status": run_status, "out_dir": str(out_dir), "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    result = run_supplement(args.baseline_dir, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    status = result.get("status", "")
    if isinstance(status, dict):
        status = status.get("status", "")
    return 0 if str(status).startswith("COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
