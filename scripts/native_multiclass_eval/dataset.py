"""Strict UCI dataset loading and leakage-safe train/dev/test split."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    CLASS_LABELS,
    DATASET_URL,
    EXPECTED_FEATURES,
    EXPECTED_ROWS,
    SPLIT_SEED,
    TARGET,
)


FEATURE_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
    "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment",
    "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_if_missing(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "UFCE-multiclass-eval/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)


def load_dataset(path, auto_download=True):
    path = Path(path)
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(str(path))
        _download_if_missing(path)
    if zipfile.is_zipfile(str(path)):
        with zipfile.ZipFile(str(path), "r") as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith("data.csv")]
            if len(csv_names) != 1:
                raise ValueError("UCI archive must contain exactly one data.csv; found %d" % len(csv_names))
            with archive.open(csv_names[0], "r") as source:
                frame = pd.read_csv(source, sep=";")
    else:
        frame = pd.read_csv(str(path), sep=";")

    frame.columns = [str(column).strip() for column in frame.columns]
    expected = FEATURE_COLUMNS + [TARGET]
    if list(frame.columns) != expected:
        missing = sorted(set(expected) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(expected))
        raise ValueError("UCI schema mismatch: missing=%r unexpected=%r" % (missing, unexpected))
    if frame.columns.duplicated().any():
        raise ValueError("UCI schema has duplicate columns")
    if len(frame) != EXPECTED_ROWS or len(FEATURE_COLUMNS) != EXPECTED_FEATURES:
        raise ValueError("UCI dataset size mismatch: rows=%d features=%d" % (len(frame), len(FEATURE_COLUMNS)))
    if frame.isna().any().any():
        raise ValueError("UCI dataset contains missing values; expected the published complete dataset")
    unknown_labels = sorted(set(frame[TARGET].astype(str)) - set(CLASS_LABELS))
    if unknown_labels:
        raise ValueError("Unexpected target labels: %r" % unknown_labels)

    frame = frame.copy().reset_index(drop=True)
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].astype(str)
    numeric = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_FEATURES]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    labels = frame[TARGET].astype(str).map(CLASS_LABELS).astype(int)
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    schema = {
        "feature_names": list(FEATURE_COLUMNS),
        "numeric_features": numeric,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_name": TARGET,
        "class_names": {str(key): value for key, value in sorted(CLASS_LABELS.items())},
    }
    summary = {
        "dataset_id": 697,
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "n_rows": int(len(frame)),
        "n_features": int(len(FEATURE_COLUMNS)),
        "target_counts": {CLASS_NAMES[key]: int((labels == key).sum()) for key in sorted(CLASS_NAMES)},
    }
    return features, labels.to_numpy(dtype=int), schema, summary


def split_dataset(features, labels, seed=SPLIT_SEED):
    labels = np.asarray(labels, dtype=int)
    all_rows = np.arange(len(labels))
    train_rows, rest_rows = train_test_split(
        all_rows, test_size=0.30, random_state=seed, stratify=labels
    )
    dev_rows, test_rows = train_test_split(
        rest_rows, test_size=0.50, random_state=seed, stratify=labels[rest_rows]
    )
    splits = {
        "train": np.asarray(train_rows, dtype=int),
        "dev": np.asarray(dev_rows, dtype=int),
        "test": np.asarray(test_rows, dtype=int),
    }
    if set(splits["train"]) & set(splits["dev"]) or set(splits["train"]) & set(splits["test"]) or set(splits["dev"]) & set(splits["test"]):
        raise AssertionError("Train/dev/test rows overlap")
    if sum(len(rows) for rows in splits.values()) != len(features):
        raise AssertionError("Train/dev/test split lost rows")
    return splits


def write_split_manifest(path, splits, labels):
    rows = []
    for partition, indexes in splits.items():
        rows.extend(
            {"row_id": int(index), "partition": partition, "target_class": int(labels[index])}
            for index in indexes
        )
    pd.DataFrame(rows).sort_values("row_id").to_csv(path, index=False)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)
