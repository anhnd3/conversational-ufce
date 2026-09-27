"""Frozen preprocessing/model bundle and the pre-CF quality gate."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn 0.24/1.0 compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


@dataclass
class ModelBundle:
    feature_names: list
    numeric_features: list
    categorical_features: list
    preprocessor: ColumnTransformer
    model: Pipeline

    def _prepare(self, frame):
        out = frame.loc[:, self.feature_names].copy()
        for name in self.categorical_features:
            # sklearn 0.24's SimpleImputer does not understand pandas
            # StringDtype; object is the canonical compatibility type.
            out[name] = out[name].astype(object).where(~out[name].isna(), np.nan)
        return out

    def predict_raw(self, frame):
        frame = self._prepare(frame)
        return np.asarray(self.model.predict(frame), dtype=int).reshape(-1)

    def predict_proba_raw(self, frame):
        frame = self._prepare(frame)
        return np.asarray(self.model.predict_proba(frame))[:, 1]

    def transform_raw(self, frame):
        return self.preprocessor.transform(self._prepare(frame))


def fit_model(train_frame, feature_schema, labels):
    numeric = list(feature_schema["numeric_features"])
    categorical = list(feature_schema["categorical_features"])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _one_hot_encoder()),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    estimator = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        n_jobs=1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    convergence_warnings = []
    model_frame = train_frame.loc[:, feature_schema["feature_names"]].copy()
    for name in categorical:
        model_frame[name] = model_frame[name].astype(object).where(~model_frame[name].isna(), np.nan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(model_frame, np.asarray(labels, dtype=int))
        convergence_warnings = [str(item.message) for item in caught if "converg" in str(item.message).lower()]
    return ModelBundle(
        feature_names=list(feature_schema["feature_names"]),
        numeric_features=numeric,
        categorical_features=categorical,
        preprocessor=preprocessor,
        model=pipeline,
    ), convergence_warnings


def quality_metrics(bundle, test_frame, test_labels, eligible_query_count):
    y_true = np.asarray(test_labels, dtype=int)
    y_pred = bundle.predict_raw(test_frame)
    y_score = bundle.predict_proba_raw(test_frame)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc_dropout_positive_risk": float(average_precision_score(1 - y_true, 1 - y_score)) if len(np.unique(y_true)) == 2 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_dropout_positive_risk": float(precision_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "recall_dropout_positive_risk": float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "f1_dropout_positive_risk": float(f1_score(1 - y_true, 1 - y_pred, zero_division=0)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
        "predicted_0": int((y_pred == 0).sum()),
        "predicted_1": int((y_pred == 1).sum()),
        "test_rows": int(len(y_true)),
        "eligible_unique_student_queries": int(eligible_query_count),
    }
    return metrics


def quality_gate(metrics, convergence_warnings):
    failures = []
    if convergence_warnings:
        failures.append("CONVERGENCE_WARNING")
    for name, value in metrics.items():
        if isinstance(value, (float, int)) and not np.isfinite(value):
            failures.append("NONFINITE_METRIC:%s" % name)
    if metrics.get("predicted_0", 0) == 0 or metrics.get("predicted_1", 0) == 0:
        failures.append("MODEL_CLASS_COLLAPSE")
    if metrics.get("eligible_unique_student_queries", 0) < 20:
        failures.append("TOO_FEW_ELIGIBLE_QUERIES")
    total = max(1, metrics.get("predicted_0", 0) + metrics.get("predicted_1", 0))
    minority = min(metrics.get("predicted_0", 0), metrics.get("predicted_1", 0)) / total
    review_class_collapse = minority < 0.01
    return {
        "passed": not failures and not review_class_collapse,
        "status": "PASS" if not failures and not review_class_collapse else ("REVIEW_CLASS_COLLAPSE" if not failures else "STOP"),
        "failures": failures,
        "review_class_collapse": bool(review_class_collapse),
        "predicted_minority_fraction": float(minority),
        "convergence_warnings": list(convergence_warnings),
    }


def save_model_metrics(path, metrics, gate):
    payload = {"metrics": metrics, "quality_gate": gate}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
