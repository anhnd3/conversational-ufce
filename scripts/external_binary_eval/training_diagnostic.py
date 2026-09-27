"""Continue the saved UPV LogisticRegression snapshot with validation logging.

This is a diagnostic run only. It never changes the frozen primary model
artifacts. sklearn 0.24 exposes LogisticRegression coefficients but does not
persist SAGA's per-sample gradient memory, so each warm-started chunk resumes
from the prior coefficients while initializing fresh SAGA optimizer memory.
"""

import argparse
import copy
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from .config import RANDOM_SEED, STUDENT_ID
from .dataset import group_split, load_upv, write_json
from .model import ModelBundle


DEFAULT_CHUNK_SIZE = 100
DEFAULT_PATIENCE = 3
DEFAULT_MIN_DELTA = 1e-5
DEFAULT_MAX_ADDITIONAL_ITERATIONS = 2000


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(estimator, x_train, y_train, x_validation, y_validation):
    train_probability = estimator.predict_proba(x_train)
    validation_probability = estimator.predict_proba(x_validation)
    train_weights = compute_sample_weight("balanced", y_train)
    validation_risk = 1 - y_validation
    validation_risk_score = validation_probability[:, 0]
    return {
        "weighted_train_log_loss": float(log_loss(
            y_train, train_probability, sample_weight=train_weights, labels=[0, 1]
        )),
        "validation_log_loss": float(log_loss(
            y_validation, validation_probability, labels=[0, 1]
        )),
        "validation_roc_auc_dropout_risk": float(roc_auc_score(
            validation_risk, validation_risk_score
        )),
        "validation_pr_auc_dropout_risk": float(average_precision_score(
            validation_risk, validation_risk_score
        )),
    }


def _append_trace(trace_path, rows):
    pd.DataFrame(rows).to_csv(trace_path, index=False)


def _atomic_joblib_dump(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def run_diagnostic(data_dir, out_dir, chunk_size=DEFAULT_CHUNK_SIZE,
                   patience=DEFAULT_PATIENCE, min_delta=DEFAULT_MIN_DELTA,
                   max_additional_iterations=DEFAULT_MAX_ADDITIONAL_ITERATIONS,
                   no_download=True):
    out_dir = Path(out_dir).resolve()
    source_path = out_dir / "model_bundle_checkpoint.joblib"
    if not source_path.is_file():
        raise FileNotFoundError("Missing fitted source checkpoint: %s" % source_path)
    if chunk_size < 1 or patience < 1 or max_additional_iterations < 1:
        raise ValueError("chunk_size, patience and max_additional_iterations must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    source_sha256 = _sha256(source_path)
    source_pipeline = joblib.load(source_path)
    if "preprocessor" not in source_pipeline.named_steps or "model" not in source_pipeline.named_steps:
        raise ValueError("Source checkpoint must contain preprocessor and model pipeline steps")

    frame, schema, dataset_summary = load_upv(data_dir, auto_download=not no_download)
    frame["student_id"] = frame[STUDENT_ID].astype(str)
    partition, _ = group_split(frame, RANDOM_SEED)
    frame = frame.copy()
    frame["partition"] = partition.to_numpy()
    train = frame[frame.partition == "train"].copy()
    validation = frame[frame.partition == "validation"].copy()
    feature_names = list(schema["feature_names"])
    source_bundle = ModelBundle(
        feature_names=feature_names,
        numeric_features=list(schema["numeric_features"]),
        categorical_features=list(schema["categorical_features"]),
        preprocessor=source_pipeline.named_steps["preprocessor"],
        model=source_pipeline,
    )
    x_train = source_bundle.transform_raw(train)
    x_validation = source_bundle.transform_raw(validation)
    y_train = train.y.to_numpy(dtype=int)
    y_validation = validation.y.to_numpy(dtype=int)
    if len(np.unique(y_validation)) != 2:
        raise ValueError("Validation split must contain both classes for early stopping")

    config = {
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "chunk_size": int(chunk_size),
        "patience": int(patience),
        "min_delta_validation_log_loss": float(min_delta),
        "max_additional_iterations": int(max_additional_iterations),
        "validation_monitor": "unweighted validation_log_loss",
        "primary_model_modified": False,
        "saga_optimizer_state_restored": False,
        "continuation_note": (
            "Each chunk warm-starts from checkpoint coefficients; sklearn 0.24 "
            "does not persist SAGA per-sample gradient memory between fit calls."
        ),
    }
    config_path = out_dir / "training_diagnostic_config.json"
    state_path = out_dir / "training_diagnostic_checkpoint.joblib"
    trace_path = out_dir / "training_trace.csv"
    summary_path = out_dir / "training_diagnostic_summary.json"
    best_model_path = out_dir / "model_training_diagnostic_best.joblib"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            prior_summary = json.load(handle)
        if prior_summary.get("stopping_reason") in ("EARLY_STOPPING", "MAX_ADDITIONAL_ITERATIONS"):
            raise RuntimeError("Diagnostic already completed: %s" % prior_summary["stopping_reason"])

    if state_path.exists():
        state = joblib.load(state_path)
        if state.get("config") != config:
            raise ValueError("Diagnostic checkpoint settings/source do not match this invocation")
        estimator = state["estimator"]
        best_estimator = state["best_estimator"]
        trace_rows = state["trace_rows"]
        additional_iterations = int(state["additional_iterations"])
        best_loss = float(state["best_validation_log_loss"])
        best_iteration = int(state["best_total_iteration"])
        stale_checks = int(state["stale_checks"])
    else:
        estimator = copy.deepcopy(source_pipeline.named_steps["model"])
        estimator.set_params(warm_start=True, max_iter=chunk_size)
        best_estimator = copy.deepcopy(estimator)
        trace_rows = []
        additional_iterations = 0
        stale_checks = 0
        start_metrics = _metrics(estimator, x_train, y_train, x_validation, y_validation)
        initial_total_iteration = int(np.max(getattr(estimator, "n_iter_", [0])))
        trace_rows.append({
            "stage": "checkpoint",
            "chunk": 0,
            "additional_iterations": 0,
            "total_iteration_estimate": initial_total_iteration,
            "chunk_actual_iterations": 0,
            "chunk_seconds": 0.0,
            "convergence_warning": False,
            **start_metrics,
        })
        best_loss = start_metrics["validation_log_loss"]
        best_iteration = initial_total_iteration
        state = None

    write_json(config_path, config)
    _append_trace(trace_path, trace_rows)
    chunk_number = sum(1 for row in trace_rows if row.get("stage") == "continuation")
    stop_reason = None
    while additional_iterations < max_additional_iterations:
        remaining = max_additional_iterations - additional_iterations
        current_chunk_size = min(chunk_size, remaining)
        estimator.set_params(max_iter=current_chunk_size)
        started = time.perf_counter()
        caught_messages = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(x_train, y_train)
            caught_messages = [
                str(item.message) for item in caught
                if "converg" in str(item.message).lower()
            ]
        chunk_seconds = time.perf_counter() - started
        chunk_actual_iterations = int(np.max(estimator.n_iter_))
        additional_iterations += chunk_actual_iterations
        chunk_number += 1
        current_metrics = _metrics(estimator, x_train, y_train, x_validation, y_validation)
        current_loss = current_metrics["validation_log_loss"]
        improved = current_loss < best_loss - min_delta
        if improved:
            best_loss = current_loss
            best_iteration = int(np.max(getattr(source_pipeline.named_steps["model"], "n_iter_", [0]))) + additional_iterations
            best_estimator = copy.deepcopy(estimator)
            stale_checks = 0
            best_pipeline = copy.deepcopy(source_pipeline)
            best_pipeline.named_steps["model"] = copy.deepcopy(best_estimator)
            _atomic_joblib_dump(best_pipeline, best_model_path)
        else:
            stale_checks += 1

        total_iteration_estimate = int(np.max(getattr(source_pipeline.named_steps["model"], "n_iter_", [0]))) + additional_iterations
        trace_rows.append({
            "stage": "continuation",
            "chunk": chunk_number,
            "additional_iterations": additional_iterations,
            "total_iteration_estimate": total_iteration_estimate,
            "chunk_actual_iterations": chunk_actual_iterations,
            "chunk_seconds": float(chunk_seconds),
            "convergence_warning": bool(caught_messages),
            "convergence_warning_messages": " | ".join(caught_messages),
            "validation_improved": bool(improved),
            "stale_checks": stale_checks,
            "best_validation_log_loss": best_loss,
            **current_metrics,
        })
        _append_trace(trace_path, trace_rows)

        state = {
            "config": config,
            "estimator": estimator,
            "best_estimator": best_estimator,
            "trace_rows": trace_rows,
            "additional_iterations": additional_iterations,
            "best_validation_log_loss": best_loss,
            "best_total_iteration": best_iteration,
            "stale_checks": stale_checks,
        }
        _atomic_joblib_dump(state, state_path)
        print(
            "chunk=%d total_iter~%d val_log_loss=%.7f val_roc_auc=%.6f val_pr_auc=%.6f stale=%d/%d"
            % (chunk_number, total_iteration_estimate, current_loss,
               current_metrics["validation_roc_auc_dropout_risk"],
               current_metrics["validation_pr_auc_dropout_risk"], stale_checks, patience),
            flush=True,
        )
        if stale_checks >= patience:
            stop_reason = "EARLY_STOPPING"
            break

    if stop_reason is None:
        stop_reason = "MAX_ADDITIONAL_ITERATIONS"
    if not best_model_path.exists():
        best_pipeline = copy.deepcopy(source_pipeline)
        best_pipeline.named_steps["model"] = copy.deepcopy(best_estimator)
        _atomic_joblib_dump(best_pipeline, best_model_path)
    total_iteration_estimate = int(np.max(getattr(source_pipeline.named_steps["model"], "n_iter_", [0]))) + additional_iterations
    final_summary = {
        **config,
        "stopping_reason": stop_reason,
        "chunks_completed": int(chunk_number),
        "additional_iterations_completed": int(additional_iterations),
        "total_iteration_estimate_at_stop": total_iteration_estimate,
        "best_total_iteration": int(best_iteration),
        "best_validation_log_loss": float(best_loss),
        "stale_checks_at_stop": int(stale_checks),
        "validation_rows": int(len(validation)),
        "train_rows": int(len(train)),
        "trace_csv": str(trace_path),
        "best_model_snapshot": str(best_model_path),
    }
    write_json(summary_path, final_summary)
    return final_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/upv_2025")
    parser.add_argument("--out-dir", default="outputs/external_binary_eval/upv_2025")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--max-additional-iterations", type=int, default=DEFAULT_MAX_ADDITIONAL_ITERATIONS)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    summary = run_diagnostic(
        args.data_dir,
        args.out_dir,
        chunk_size=args.chunk_size,
        patience=args.patience,
        min_delta=args.min_delta,
        max_additional_iterations=args.max_additional_iterations,
        no_download=not args.allow_download,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
