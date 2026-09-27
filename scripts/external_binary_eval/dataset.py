"""UPV-2025 download, normalization, schema audit and group splitting."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    DROP_FIELDS,
    EXPECTED_CATEGORICAL_COUNT,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_FILES,
    EXPECTED_NUMERIC_COUNT,
    EXPECTED_STUDENTS,
    EXPECTED_TOTAL_ROWS,
    MONTHLY_FAMILIES,
    MONTHS,
    RANDOM_SEED,
    SOURCE_CLASS,
    TARGET,
    STUDENT_ID,
    ZENODO_API,
)


MONTH_RE = re.compile(r"^(?P<family>.+)_20\d{2}_(?P<month>\d{1,2})$")


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zenodo_files():
    response = requests.get(ZENODO_API, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload.get("files", [])


def download_upv(data_dir, force=False, session=None):
    """Download the three source archives, validating the published MD5 hashes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = _zenodo_files()
    session = session or requests.Session()
    downloaded = []
    for year, expected in EXPECTED_FILES.items():
        candidates = [
            item for item in files
            if year in str(item.get("key", "")) and str(item.get("key", "")).lower().endswith((".zip", ".csv"))
        ]
        if not candidates:
            raise RuntimeError("Zenodo record has no file matching year %s" % year)
        item = sorted(candidates, key=lambda x: str(x.get("key")))[0]
        name = Path(str(item["key"])).name
        target = data_dir / name
        if force or not target.exists() or md5_file(target) != expected["md5"]:
            download_url = item.get("links", {}).get("content") or item.get("links", {}).get("self")
            with session.get(download_url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(target, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        actual = md5_file(target)
        if actual != expected["md5"]:
            raise RuntimeError("MD5 mismatch for %s: %s != %s" % (target, actual, expected["md5"]))
        downloaded.append(str(target))
    return downloaded


def _read_source(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Expected exactly one CSV in %s" % path)
            with archive.open(names[0]) as handle:
                return pd.read_csv(handle, sep=";", low_memory=False)
    return pd.read_csv(path, sep=";", low_memory=False)


def _normalize_columns(frame):
    """Rename September--December columns to a cohort-independent m09--m12 schema."""
    rename = {}
    keep = []
    for column in frame.columns:
        match = MONTH_RE.match(str(column))
        if not match:
            keep.append(column)
            continue
        month = int(match.group("month"))
        family = match.group("family")
        if month in (9, 10, 11, 12) and family in MONTHLY_FAMILIES:
            new_name = "%s_m%02d" % (family, month)
            rename[column] = new_name
            keep.append(column)
        # Other months are after the fixed December cutoff and are dropped.
    out = frame.loc[:, keep].rename(columns=rename)
    duplicate_columns = out.columns[out.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError("Duplicate normalized monthly columns: %s" % duplicate_columns)
    return out


def _coerce_labels(frame):
    labels = frame[TARGET].astype(str).str.strip().str.upper()
    unexpected = sorted(set(labels) - {"A", "B"})
    if unexpected:
        raise ValueError("Unexpected abandono_hash labels: %s" % unexpected)
    out = frame.copy()
    out["y"] = labels.map({"A": SOURCE_CLASS, "B": 1}).astype(int)
    return out


def load_upv(data_dir, auto_download=True, force_download=False):
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*"))
    usable = [path for path in paths if path.suffix.lower() in (".zip", ".csv") and any(year in path.name for year in EXPECTED_FILES)]
    if len(usable) < 3 and auto_download:
        usable = [Path(path) for path in download_upv(data_dir, force=force_download)]
    if len(usable) < 3:
        raise FileNotFoundError("Need UPV 2018/2021/2022 archives in %s" % data_dir)

    frames = []
    source_manifest = []
    for year in sorted(EXPECTED_FILES):
        matches = [path for path in usable if year in path.name]
        if not matches:
            raise FileNotFoundError("Missing UPV %s source" % year)
        path = sorted(matches)[0]
        digest = md5_file(path)
        expected = EXPECTED_FILES[year]
        if digest != expected["md5"]:
            raise ValueError("Checksum mismatch for %s: %s != %s" % (path, digest, expected["md5"]))
        source = _normalize_columns(_read_source(path))
        if len(source) != expected["rows"]:
            raise ValueError("Row count mismatch for %s: %s != %s" % (path, len(source), expected["rows"]))
        source["source_year"] = int(year)
        source = _coerce_labels(source)
        frames.append(source)
        source_manifest.append({"year": year, "path": str(path.resolve()), "md5": digest, "rows": int(len(source)), "sha256": sha256_file(path)})

    frame = pd.concat(frames, ignore_index=True, sort=False)
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValueError("Total row count mismatch: %s != %s" % (len(frame), EXPECTED_TOTAL_ROWS))
    if frame[STUDENT_ID].nunique(dropna=False) != EXPECTED_STUDENTS:
        raise ValueError("Student count mismatch: %s != %s" % (frame[STUDENT_ID].nunique(dropna=False), EXPECTED_STUDENTS))

    # Compute the expected schema from the union of non-monthly source fields,
    # while preserving source order.  All 36 monthly fields are explicit.
    monthly = ["%s_%s" % (family, month) for family in MONTHLY_FAMILIES for month in MONTHS]
    present_drop = set(DROP_FIELDS)
    base_features = []
    for column in frame.columns:
        if column in {STUDENT_ID, TARGET, "y", "source_year"} or column in present_drop or column in monthly:
            continue
        if column not in base_features:
            base_features.append(column)
    for column in monthly:
        if column not in frame.columns:
            frame[column] = np.nan
    feature_names = base_features + monthly
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError("Expected 81 raw predictors after December cutoff, got %s: %s" % (len(feature_names), feature_names))
    frame = frame[[STUDENT_ID, "y", "source_year"] + feature_names].copy()
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_names]
    numeric = [column for column in feature_names if column not in categorical]
    if len(categorical) != EXPECTED_CATEGORICAL_COUNT or len(numeric) != EXPECTED_NUMERIC_COUNT:
        raise ValueError("Expected 75 numeric + 6 categorical, got %s + %s" % (len(numeric), len(categorical)))
    frame[STUDENT_ID] = frame[STUDENT_ID].astype(str)
    for column in categorical:
        frame[column] = frame[column].astype("string")
    for column in numeric:
        # UPV CSVs use the semicolon as delimiter and a decimal comma in
        # numeric cells (for example ``7,0``).  Normalize that representation
        # before the frozen sklearn imputer sees the raw values.
        values = frame[column]
        if values.dtype == object:
            values = values.astype(str).str.replace(",", ".", regex=False)
        frame[column] = pd.to_numeric(values, errors="coerce")

    schema = {
        "feature_names": feature_names,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "n_raw_features": len(feature_names),
        "monthly_features": monthly,
        "cutoff": "end_of_december",
        "label_mapping": {"A": 0, "B": 1},
    }
    summary = {
        "n_rows": int(len(frame)),
        "n_students": int(frame[STUDENT_ID].nunique()),
        "dropout_rows": int((frame["y"] == 0).sum()),
        "non_dropout_rows": int((frame["y"] == 1).sum()),
        "dropout_rate": float((frame["y"] == 0).mean()),
        "source_manifest": source_manifest,
        "schema": schema,
    }
    return frame, schema, summary


def group_split(frame, seed=RANDOM_SEED):
    """Make a deterministic 70/15/15 student-group split stratified by ever_dropout."""
    group_labels = frame.groupby(STUDENT_ID)["y"].apply(lambda values: int((values == 0).any()))
    students = group_labels.index.to_numpy()
    labels = group_labels.to_numpy()
    train_students, heldout_students = train_test_split(students, test_size=0.30, random_state=seed, stratify=labels)
    heldout_labels = group_labels.loc[heldout_students].to_numpy()
    valid_students, test_students = train_test_split(heldout_students, test_size=0.50, random_state=seed, stratify=heldout_labels)
    assignments = {}
    assignments.update({student: "train" for student in train_students})
    assignments.update({student: "validation" for student in valid_students})
    assignments.update({student: "test" for student in test_students})
    partition = frame[STUDENT_ID].map(assignments)
    if partition.isna().any() or len(assignments) != frame[STUDENT_ID].nunique():
        raise AssertionError("Every student must map to exactly one partition")
    if any(set(frame.loc[partition == left, STUDENT_ID]) & set(frame.loc[partition == right, STUDENT_ID]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Student overlap between partitions")
    manifest = pd.DataFrame({"student_id": list(assignments), "partition": list(assignments.values()), "ever_dropout": [int(group_labels.loc[s]) for s in assignments]})
    return partition, manifest


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
