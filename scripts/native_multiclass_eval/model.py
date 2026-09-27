"""Train-only preprocessing and a frozen native multinomial classifier."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLASS_NAMES


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


@dataclass
class MulticlassBundle:
    feature_names: list
    numeric_features: list
    categorical_features: list
    preprocessor: object
    model: object

    def prepare_raw(self, frame):
        result = frame.loc[:, self.feature_names].copy()
        for name in self.categorical_features:
            result[name] = result[name].astype(object)
        return result

    def predict_raw(self, frame):
        return np.asarray(self.model.predict(self.prepare_raw(frame)), dtype=int).reshape(-1)

    def predict_proba_raw(self, frame):
        return np.asarray(self.model.predict_proba(self.prepare_raw(frame)))


def fit_model(train_frame, train_labels, schema):
    numeric = list(schema["numeric_features"])
    categorical = list(schema["categorical_features"])
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
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        C=1.0,
        max_iter=3000,
        random_state=42,
        n_jobs=1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    train = train_frame.loc[:, schema["feature_names"]].copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(train, np.asarray(train_labels, dtype=int))
    convergence = [str(item.message) for item in caught if "converg" in str(item.message).lower()]
    bundle = MulticlassBundle(
        feature_names=list(schema["feature_names"]),
        numeric_features=numeric,
        categorical_features=categorical,
        preprocessor=preprocessor,
        model=pipeline,
    )
    return bundle, convergence


def prediction_summary(bundle, features, labels):
    truth = np.asarray(labels, dtype=int)
    predicted = bundle.predict_raw(features)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    return {
        "rows": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion_matrix_labels_0_1_2": matrix.tolist(),
        "true_support": {CLASS_NAMES[i]: int((truth == i).sum()) for i in range(3)},
        "predicted_support": {CLASS_NAMES[i]: int((predicted == i).sum()) for i in range(3)},
        "classification_report": classification_report(
            truth, predicted, labels=[0, 1, 2], target_names=[CLASS_NAMES[i] for i in range(3)],
            output_dict=True, zero_division=0,
        ),
    }
