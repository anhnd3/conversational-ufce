#!/usr/bin/env python3
"""
P7 UFCE hyper-tuning driver.

Run 1: tune Category A gates only (min_act/min_feas).
Run 2: fix gates and sweep radius + n_neighbors.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ---- PYTHONPATH bootstrap ----
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------

from ufce import UFCE
from ufce.core import cfmethods
from ufce.core import evaluations as eval_module
from ufce.core.evaluations import Catproximity, Contproximity, Sparsity, Actionability, Plausibility, Feasibility
from ufce.core.data_processing import classify_dataset_getModel
from ufce.core.data_processing import (
    get_bank_user_constraints,
    get_grad_user_constraints,
    get_wine_user_constraints,
    get_bupa_user_constraints,
    get_movie_user_constraints,
)
from scripts.archieve.reproduce_results_v3 import AUTHOR_TABLE7


METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
EXPECTED_COMPARISON_ROWS_PER_CONFIG = len(METHODS) * len(METRICS)

# Ranking robustness defaults (degenerate guard).
MIN_REQUIRED_FINITE = 4
MIN_REQUIRED_RATIO = 0.8
COVERAGE_EPS = 1e-12

DEBUG_MOVIE_DIAG = False

SNAPSHOT_FLOAT_ATOL = 1e-12
SNAPSHOT_MAX_DIFFS = 10
DIAG_LOF_SWEEP_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4]

MOVIE_DISTANCE_SCALER_MODE_DEFAULT = "none"
MOVIE_LOF_SPACE_MODE_DEFAULT = "raw"
MOVIE_LOF_STANDARDIZE_MODE_DEFAULT = "on"
MOVIE_LOF_CONTAMINATION_DEFAULT = "auto"
MOVIE_APF_UNIT_MODE_DEFAULT = "count"
MOVIE_APF_DENOM_MODE_DEFAULT = "explained"
MOVIE_N_EXPLAINED_TARGET_DEFAULT = 20
PROX_SPACE_MODE_DEFAULT = "raw"
FOLDS_SOURCE_DEFAULT = "auto"

EFFECTIVE_CONFIG_FIELDS = [
    "dataset",
    "seed",
    "radius",
    "n_neighbors",
    "contprox_metric",
    "min_act",
    "min_feas",
    "ufce_flip_filter",
    "folds_selected",
    "n_folds_selected",
    "folds_source",
    "N_explained_target",
    "movie_distance_scaler_mode",
    "movie_lof_space_mode",
    "movie_lof_standardize_mode",
    "movie_lof_contamination",
    "movie_apf_unit_mode",
    "movie_apf_denom_mode",
    "prox_space_mode",
]
RUN_SORT_FIELDS = ["run_started_at_utc", "run_id", "run_seq_in_file"]


def parse_int_grid(grid_text: str | None, default_values: List[int]) -> List[int]:
    if grid_text is None:
        return list(default_values)
    values = []
    for token in grid_text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Parsed grid is empty.")
    return values


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def get_ufce_flip_filter_from_record(record, default: int = 1) -> int:
    """
    Backward-compatible reader for old/new flip-filter field names.
    """
    if "ufce_flip_filter" in record and pd.notna(record["ufce_flip_filter"]):
        return int(record["ufce_flip_filter"])
    if "ufce3_flip_filter" in record and pd.notna(record["ufce3_flip_filter"]):
        return int(record["ufce3_flip_filter"])
    return int(default)


def parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return bool(default)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value) != 0)
    return bool(default)


def get_run_started_at_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def build_run_id(run_started_at_utc: str, seed: int) -> str:
    entropy = f"{time.time_ns()}_{os.getpid()}_{int(seed)}"
    suffix = hashlib.sha1(entropy.encode("utf-8")).hexdigest()[:8]
    return f"{run_started_at_utc}_{suffix}"


def extract_fold_id_from_name(fold_name: str) -> int:
    match = re.search(r"testfold_(\d+)", str(fold_name))
    if match:
        return int(match.group(1))
    return -1


def build_folds_selected_signature(testfold_paths: List[str]) -> str:
    fold_ids = sorted({extract_fold_id_from_name(os.path.basename(p)) for p in testfold_paths if extract_fold_id_from_name(os.path.basename(p)) >= 0})
    return ",".join(str(i) for i in fold_ids) if fold_ids else "unknown"


def sanitize_mode_string(value, default_value: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else str(default_value)


def ensure_effective_config_fields_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "folds_selected": "unknown",
        "n_folds_selected": 0,
        "folds_source": FOLDS_SOURCE_DEFAULT,
        "N_explained_target": 0,
        "movie_distance_scaler_mode": MOVIE_DISTANCE_SCALER_MODE_DEFAULT,
        "movie_lof_space_mode": MOVIE_LOF_SPACE_MODE_DEFAULT,
        "movie_lof_standardize_mode": MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
        "movie_lof_contamination": MOVIE_LOF_CONTAMINATION_DEFAULT,
        "movie_apf_unit_mode": MOVIE_APF_UNIT_MODE_DEFAULT,
        "movie_apf_denom_mode": MOVIE_APF_DENOM_MODE_DEFAULT,
        "prox_space_mode": PROX_SPACE_MODE_DEFAULT,
        "run_started_at_utc": "00000000T000000Z",
        "run_id": "00000000T000000Z_00000000",
        "run_seq_in_file": 0,
    }
    for col, default_value in defaults.items():
        if col not in out.columns:
            out[col] = default_value
    out["run_started_at_utc"] = out["run_started_at_utc"].apply(lambda v: sanitize_mode_string(v, "00000000T000000Z"))
    out["run_id"] = out["run_id"].apply(lambda v: sanitize_mode_string(v, "00000000T000000Z_00000000"))
    out["run_seq_in_file"] = pd.to_numeric(out["run_seq_in_file"], errors="coerce").fillna(0).astype(int)
    out["folds_selected"] = out["folds_selected"].apply(lambda v: sanitize_mode_string(v, "unknown"))
    out["n_folds_selected"] = pd.to_numeric(out["n_folds_selected"], errors="coerce").fillna(0).astype(int)
    out["folds_source"] = out["folds_source"].apply(lambda v: sanitize_mode_string(v, FOLDS_SOURCE_DEFAULT))
    out["N_explained_target"] = pd.to_numeric(out["N_explained_target"], errors="coerce").fillna(0).astype(int)
    out["movie_distance_scaler_mode"] = out["movie_distance_scaler_mode"].apply(
        lambda v: sanitize_mode_string(v, MOVIE_DISTANCE_SCALER_MODE_DEFAULT)
    )
    out["movie_lof_space_mode"] = out["movie_lof_space_mode"].apply(lambda v: sanitize_mode_string(v, MOVIE_LOF_SPACE_MODE_DEFAULT))
    out["movie_lof_standardize_mode"] = out["movie_lof_standardize_mode"].apply(
        lambda v: sanitize_mode_string(v, MOVIE_LOF_STANDARDIZE_MODE_DEFAULT)
    )
    out["movie_lof_contamination"] = out["movie_lof_contamination"].apply(
        lambda v: sanitize_mode_string(v, MOVIE_LOF_CONTAMINATION_DEFAULT)
    )
    out["movie_apf_unit_mode"] = out["movie_apf_unit_mode"].apply(lambda v: sanitize_mode_string(v, MOVIE_APF_UNIT_MODE_DEFAULT))
    out["movie_apf_denom_mode"] = out["movie_apf_denom_mode"].apply(lambda v: sanitize_mode_string(v, MOVIE_APF_DENOM_MODE_DEFAULT))
    out["prox_space_mode"] = out["prox_space_mode"].apply(lambda v: sanitize_mode_string(v, PROX_SPACE_MODE_DEFAULT))
    return out


def ensure_effective_config_fields_record(record) -> Dict[str, object]:
    defaults = {
        "folds_selected": "unknown",
        "n_folds_selected": 0,
        "folds_source": FOLDS_SOURCE_DEFAULT,
        "N_explained_target": 0,
        "movie_distance_scaler_mode": MOVIE_DISTANCE_SCALER_MODE_DEFAULT,
        "movie_lof_space_mode": MOVIE_LOF_SPACE_MODE_DEFAULT,
        "movie_lof_standardize_mode": MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
        "movie_lof_contamination": MOVIE_LOF_CONTAMINATION_DEFAULT,
        "movie_apf_unit_mode": MOVIE_APF_UNIT_MODE_DEFAULT,
        "movie_apf_denom_mode": MOVIE_APF_DENOM_MODE_DEFAULT,
        "prox_space_mode": PROX_SPACE_MODE_DEFAULT,
    }
    out = dict(record)
    for col, default_value in defaults.items():
        if col in {"n_folds_selected", "N_explained_target"}:
            val = pd.to_numeric(out.get(col, default_value), errors="coerce")
            out[col] = int(default_value if pd.isna(val) else int(val))
        else:
            out[col] = sanitize_mode_string(out.get(col, default_value), default_value)
    return out


def get_effective_key_from_record(record) -> Tuple:
    rec = ensure_effective_config_fields_record(record)
    return tuple(rec.get(col) for col in EFFECTIVE_CONFIG_FIELDS)


def infer_raw_fold_rows(dataset: str, fold_name: str, fallback_rows: int) -> int:
    if str(dataset).strip().lower() != "movie":
        return int(fallback_rows)
    raw_name = str(fold_name).replace("_pred_0", "")
    raw_path = os.path.join(ROOT, "ufce", "data", "folds", "movie", raw_name)
    if os.path.exists(raw_path):
        try:
            return int(len(pd.read_csv(raw_path)))
        except Exception:
            return int(fallback_rows)
    return int(fallback_rows)


def safe_rate(numerator: int, denominator: int) -> float:
    denom = int(denominator)
    if denom <= 0:
        return float("nan")
    return float(numerator / denom)


def dataframe_first_row_hash(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "NA"
    payload = "|".join(str(v) for v in df.iloc[0].tolist())
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12] if payload else "NA"


def ensure_method_detail_template(detail: Dict[str, object] | None, n_pairs_input: int) -> Dict[str, object]:
    d = {} if detail is None else dict(detail)
    passed_idx = d.get("passed_idx", [])
    if not isinstance(passed_idx, (list, tuple, np.ndarray)):
        passed_idx = []
    passed_idx = [int(i) for i in passed_idx]
    n_input = int(d.get("n_pairs_input", n_pairs_input))
    n_valid = int(d.get("n_pairs_valid", n_input))
    count = int(d.get("count", len(passed_idx)))
    return {
        "count": count,
        "passed_idx": passed_idx,
        "n_pairs_input": n_input,
        "n_pairs_valid": n_valid,
        "pair_details": list(d.get("pair_details", [])),
        "fail_reason_counts": dict(d.get("fail_reason_counts", {})),
    }


def canonical_float(value) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if np.isnan(x):
        return "nan"
    if np.isposinf(x):
        return "inf"
    if np.isneginf(x):
        return "-inf"
    if x == 0.0:
        x = 0.0
    return np.format_float_positional(x, precision=12, unique=False, trim="k")


def _stable_text_hash(parts: List[str]) -> str:
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def hash_index(index_obj) -> str:
    if index_obj is None:
        return "NA"
    parts = [str(v) for v in list(index_obj)]
    return _stable_text_hash(parts) if len(parts) > 0 else "NA"


def hash_table_content(df: pd.DataFrame, cols: List[str]) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "NA"
    use_cols = [c for c in cols if c in df.columns]
    if len(use_cols) == 0:
        return "NA"
    parts: List[str] = []
    for _, row in df.loc[:, use_cols].iterrows():
        row_parts = []
        for col in use_cols:
            v = row[col]
            if isinstance(v, (float, np.floating, int, np.integer)):
                row_parts.append(canonical_float(v))
            else:
                row_parts.append(str(v))
        parts.append("::".join(row_parts))
    return _stable_text_hash(parts)


def _ordered_method_metric_pairs(df: pd.DataFrame) -> List[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if "method" not in df.columns or "metric" not in df.columns:
        return []
    return [f"{str(m)}::{str(metric)}" for m, metric in zip(df["method"].tolist(), df["metric"].tolist())]


def _prod_config_key_from_record(record: Dict[str, object]) -> Tuple:
    rec = ensure_effective_config_fields_record(record)
    return (
        str(rec.get("dataset", "")),
        int(pd.to_numeric(rec.get("seed", 0), errors="coerce")),
        int(pd.to_numeric(rec.get("radius", 0), errors="coerce")),
        int(pd.to_numeric(rec.get("n_neighbors", 0), errors="coerce")),
        str(rec.get("contprox_metric", "")),
        int(pd.to_numeric(rec.get("min_act", 0), errors="coerce")),
        int(pd.to_numeric(rec.get("min_feas", 0), errors="coerce")),
        int(pd.to_numeric(rec.get("ufce_flip_filter", 1), errors="coerce")),
        sanitize_mode_string(rec.get("folds_selected", "unknown"), "unknown"),
        sanitize_mode_string(rec.get("folds_source", FOLDS_SOURCE_DEFAULT), FOLDS_SOURCE_DEFAULT),
        sanitize_mode_string(
            rec.get("movie_distance_scaler_mode", MOVIE_DISTANCE_SCALER_MODE_DEFAULT),
            MOVIE_DISTANCE_SCALER_MODE_DEFAULT,
        ),
        sanitize_mode_string(rec.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT), MOVIE_LOF_SPACE_MODE_DEFAULT),
        sanitize_mode_string(
            rec.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT),
            MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
        ),
        sanitize_mode_string(rec.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT), MOVIE_APF_UNIT_MODE_DEFAULT),
        sanitize_mode_string(rec.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT), MOVIE_APF_DENOM_MODE_DEFAULT),
    )


def _prod_row_hash(row: Dict[str, object]) -> str:
    parts = []
    for key in sorted(row.keys()):
        if str(key).startswith("diag_"):
            continue
        val = row[key]
        if isinstance(val, (float, np.floating, int, np.integer)):
            sval = canonical_float(val)
        else:
            sval = str(val)
        parts.append(f"{key}={sval}")
    return _stable_text_hash(parts)


def _extract_topk_prod_keys(
    leaderboard_df: pd.DataFrame,
    k: int = 10,
    prioritize_ufce3: bool = False,
) -> Tuple[str, List[str]]:
    if not isinstance(leaderboard_df, pd.DataFrame) or leaderboard_df.empty:
        return "NA", []
    lb = drop_diag_columns_for_prod(leaderboard_df.copy())
    lb = ensure_effective_config_fields_df(lb)
    lb = sort_leaderboard_configs(lb, prioritize_ufce3=prioritize_ufce3)
    keys = []
    for _, r in lb.head(k).iterrows():
        keys.append(repr(_prod_config_key_from_record(r.to_dict())))
    best_key = keys[0] if len(keys) > 0 else "NA"
    return best_key, keys


def build_fold_snapshot(
    comparison_df_fold: pd.DataFrame,
    method_stats_fold: Dict[str, Dict[str, float]],
    fold_file: str,
    fold_id: int,
    folds_selected: str,
    prod_config_key: str,
) -> Dict[str, object]:
    comp = drop_diag_columns_for_prod(comparison_df_fold.copy())
    pair_order = _ordered_method_metric_pairs(comp)
    dtype_parts = [f"{c}:{str(t)}" for c, t in zip(comp.columns, comp.dtypes)] if isinstance(comp, pd.DataFrame) else []
    out = {
        "layer": "fold",
        "prod_config_key": str(prod_config_key),
        "fold_file": str(fold_file),
        "fold_id": int(fold_id),
        "folds_selected": str(folds_selected),
        "comp_n_rows": int(len(comp)),
        "comp_n_cols": int(len(comp.columns)) if isinstance(comp, pd.DataFrame) else 0,
        "pair_order_hash": _stable_text_hash(pair_order) if len(pair_order) > 0 else "NA",
        "columns_hash": _stable_text_hash([str(c) for c in comp.columns]) if isinstance(comp, pd.DataFrame) and len(comp.columns) > 0 else "NA",
        "dtypes_hash": _stable_text_hash(dtype_parts) if len(dtype_parts) > 0 else "NA",
        "index_len": int(len(comp.index)) if isinstance(comp, pd.DataFrame) else 0,
        "index_hash": hash_index(comp.index) if isinstance(comp, pd.DataFrame) else "NA",
        "comparison_hash": hash_table_content(comp, ["method", "metric", "ours", "author", "norm_error"]),
        "comparison_prod_columns_hash": hash_table_content(
            comp,
            ["method", "metric", "ours", "author", "delta_ours_minus_author", "norm_error"],
        ),
    }
    for method in METHODS:
        mk = method.lower()
        ms = method_stats_fold.get(method, {}) if isinstance(method_stats_fold, dict) else {}
        out[f"{mk}_coverage"] = float(ms.get("coverage", 0.0))
        out[f"{mk}_n_instances"] = int(ms.get("n_instances", 0))
        out[f"{mk}_n_selected"] = int(ms.get("n_instances_with_selected_usable_cf", 0))
    return out


def build_config_snapshot(
    comparison_df_config: pd.DataFrame,
    row: Dict[str, object],
    prod_config_key: str,
    leaderboard_preview: pd.DataFrame | None = None,
    prioritize_ufce3: bool = False,
) -> Dict[str, object]:
    comp = drop_diag_columns_for_prod(comparison_df_config.copy())
    pair_order = _ordered_method_metric_pairs(comp)
    dtype_parts = [f"{c}:{str(t)}" for c, t in zip(comp.columns, comp.dtypes)] if isinstance(comp, pd.DataFrame) else []
    best_key, top_keys = (
        _extract_topk_prod_keys(leaderboard_preview, k=10, prioritize_ufce3=prioritize_ufce3)
        if leaderboard_preview is not None
        else ("NA", [])
    )
    out = {
        "layer": "config",
        "prod_config_key": str(prod_config_key),
        "valid_config": parse_bool(row.get("valid_config", True), default=True),
        "score_ufce1": float(row.get("score_ufce1", float("nan"))),
        "score_ufce2": float(row.get("score_ufce2", float("nan"))),
        "score_ufce3": float(row.get("score_ufce3", float("nan"))),
        "score_max": float(row.get("score_max", float("nan"))),
        "score_mean": float(row.get("score_mean", float("nan"))),
        "score_max_penalized": float(row.get("score_max_penalized", float("nan"))),
        "min_coverage": float(row.get("min_coverage", float("nan"))),
        "best_config_key": str(best_key),
        "sorted_leaderboard_top_k_keys": " || ".join(top_keys),
        "comp_n_rows": int(len(comp)),
        "comp_n_cols": int(len(comp.columns)) if isinstance(comp, pd.DataFrame) else 0,
        "pair_order_hash": _stable_text_hash(pair_order) if len(pair_order) > 0 else "NA",
        "columns_hash": _stable_text_hash([str(c) for c in comp.columns]) if isinstance(comp, pd.DataFrame) and len(comp.columns) > 0 else "NA",
        "dtypes_hash": _stable_text_hash(dtype_parts) if len(dtype_parts) > 0 else "NA",
        "index_len": int(len(comp.index)) if isinstance(comp, pd.DataFrame) else 0,
        "index_hash": hash_index(comp.index) if isinstance(comp, pd.DataFrame) else "NA",
        "comparison_hash": hash_table_content(comp, ["method", "metric", "ours", "author", "norm_error"]),
        "comparison_prod_columns_hash": hash_table_content(
            comp,
            ["method", "metric", "ours", "author", "delta_ours_minus_author", "norm_error"],
        ),
        "leaderboard_row_hash": _prod_row_hash(row),
    }
    return out


def _snapshot_values_equal(left, right) -> bool:
    if isinstance(left, (float, np.floating, int, np.integer)) and isinstance(right, (float, np.floating, int, np.integer)):
        lval = float(left)
        rval = float(right)
        if np.isnan(lval) and np.isnan(rval):
            return True
        return bool(np.isclose(lval, rval, atol=SNAPSHOT_FLOAT_ATOL, rtol=0.0, equal_nan=True))
    return left == right


def validate_snapshot_unchanged(
    before: Dict[str, object],
    after: Dict[str, object],
    layer: str,
    prod_config_key: str,
    fold_id: int | None = None,
    fold_file: str | None = None,
) -> None:
    diffs = []
    keys = sorted(set(before.keys()) | set(after.keys()))
    for key in keys:
        b = before.get(key, "__missing__")
        a = after.get(key, "__missing__")
        if not _snapshot_values_equal(b, a):
            diffs.append((key, b, a))
            if len(diffs) >= SNAPSHOT_MAX_DIFFS:
                break
    if len(diffs) == 0:
        return
    print(
        f"[ERR][DIAG-LEAK] layer={layer}, prod_config_key={prod_config_key}, "
        f"fold_id={fold_id}, fold_file={fold_file}"
    )
    for key, old_v, new_v in diffs:
        print(f"[ERR][DIAG-LEAK] diff key={key}: old={old_v} -> new={new_v}")
    raise RuntimeError("Diagnostic path mutated production snapshot.")


def drop_diag_columns_for_prod(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    diag_cols = [c for c in df.columns if str(c).startswith("diag_")]
    if len(diag_cols) == 0:
        return df
    return df.drop(columns=diag_cols, errors="ignore")


@contextmanager
def preserve_rng_state():
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@contextmanager
def temporary_single_thread_env():
    keys = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    previous = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "1"
        yield
    finally:
        for k, val in previous.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def _contains_frame_or_array(obj) -> bool:
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
        return True
    if isinstance(obj, dict):
        return any(_contains_frame_or_array(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_frame_or_array(v) for v in obj)
    return False


def sanitize_diag_payload(obj):
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        sample = obj.head(3).to_dict(orient="records")
        return {
            "_type": "DataFrameSummary",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns": [str(c) for c in obj.columns.tolist()],
            "first_row_hash": dataframe_first_row_hash(obj),
            "sample": sample,
        }
    if isinstance(obj, pd.Series):
        return {
            "_type": "SeriesSummary",
            "len": int(len(obj)),
            "name": str(obj.name),
            "sample": [sanitize_diag_payload(v) for v in obj.head(5).tolist()],
        }
    if isinstance(obj, np.ndarray):
        return {
            "_type": "ndarraySummary",
            "shape": [int(x) for x in obj.shape],
            "sample": [sanitize_diag_payload(v) for v in obj.reshape(-1)[:8].tolist()],
        }
    if isinstance(obj, dict):
        return {str(k): sanitize_diag_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_diag_payload(v) for v in obj]
    return str(obj)


def make_diag_key_hash(diag_key: str) -> str:
    return hashlib.sha1(str(diag_key).encode("utf-8")).hexdigest()[:12]


def write_diag_sidecar_json(
    out_dir: str,
    run_id: str,
    diag_key: str,
    fold_id: int,
    payload: Dict[str, object],
) -> str:
    diag_hash = make_diag_key_hash(diag_key)
    diag_dir = os.path.join(str(out_dir), "diag", str(run_id), str(diag_hash))
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, f"fold_{int(fold_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def build_movie_distance_scaler(
    datasetdf: pd.DataFrame,
    features: List[str],
    numf: List[str],
    outcome_label: str,
) -> Dict[str, object]:
    """
    Build a global (full-dataset) movie-only affine scaler so that
    transformed numeric columns are in [0, 100]:
      scaled = (raw - min) / ((max-min)/100)
    Constant columns are pinned to 0 in scaled space.
    """
    cols = [c for c in numf if c in features and c in datasetdf.columns and c != outcome_label]
    if len(cols) == 0:
        raise ValueError("Movie distance scaler could not find numeric columns to scale.")

    base = datasetdf.loc[:, features].copy()
    mins = base[cols].min()
    maxs = base[cols].max()
    ranges = maxs - mins
    constant_cols = [c for c in cols if float(ranges[c]) == 0.0]

    # Reuse legacy affine fields used by existing code paths:
    # transformed = (x - medians) / mads
    # Here: medians -> mins, mads -> (range / 100)
    mads = (ranges / 100.0).replace(0.0, 1.0)
    medians = mins

    return {
        "kind": "movie_minmax_0_100",
        "scale_cols": cols,
        "medians": medians.astype(float),
        "mads": mads.astype(float),
        "mins": mins.astype(float),
        "maxs": maxs.astype(float),
        "constant_cols": constant_cols,
    }


def apply_distance_scaler(df: pd.DataFrame, scaler: Dict | None) -> pd.DataFrame:
    if scaler is None:
        return df.copy()
    out = df.copy()
    scale_cols = scaler.get("scale_cols", None)
    if scale_cols is None:
        scale_cols = list(getattr(scaler.get("medians", pd.Series(dtype=float)), "index", []))
    cols = [c for c in scale_cols if c in out.columns]
    if len(cols) == 0:
        return out
    out.loc[:, cols] = (out.loc[:, cols] - scaler["medians"].reindex(cols)) / scaler["mads"].reindex(cols)
    constant_cols = [c for c in scaler.get("constant_cols", []) if c in cols]
    if len(constant_cols) != 0:
        out.loc[:, constant_cols] = 0.0
    return out


def inverse_distance_scaler(df: pd.DataFrame, scaler: Dict | None) -> pd.DataFrame:
    if scaler is None:
        return df.copy()
    out = df.copy()
    scale_cols = scaler.get("scale_cols", None)
    if scale_cols is None:
        scale_cols = list(getattr(scaler.get("medians", pd.Series(dtype=float)), "index", []))
    cols = [c for c in scale_cols if c in out.columns]
    if len(cols) == 0:
        return out
    out.loc[:, cols] = (out.loc[:, cols] * scaler["mads"].reindex(cols)) + scaler["medians"].reindex(cols)
    constant_cols = [c for c in scaler.get("constant_cols", []) if c in cols]
    if len(constant_cols) != 0:
        out.loc[:, constant_cols] = scaler["medians"].reindex(constant_cols).values
    return out


def log_movie_distance_sanity(
    X_distance: pd.DataFrame,
    features: List[str],
    radius_values: List[int],
    seed: int,
) -> None:
    """
    One-time movie diagnostic to verify scaled geometry:
    - sample pairwise L2 stats in distance space
    - median/mean neighbor counts for each radius in the active grid
    """
    from scipy.spatial import KDTree

    if X_distance.empty:
        print("[MOVIE][DistanceSpace] Sanity skipped: empty distance matrix.")
        return

    arr = X_distance.loc[:, features].to_numpy(dtype=float, copy=False)
    n = arr.shape[0]
    rng = np.random.default_rng(int(seed))

    if n < 2:
        print("[MOVIE][DistanceSpace] Sanity skipped: fewer than 2 rows.")
        return

    sample_pairs = min(512, max(32, n * 2))
    pair_i = rng.integers(0, n, size=sample_pairs)
    pair_j = rng.integers(0, n, size=sample_pairs)
    same = pair_i == pair_j
    if np.any(same):
        pair_j[same] = (pair_j[same] + 1) % n
    pair_d = np.linalg.norm(arr[pair_i] - arr[pair_j], axis=1)
    print(
        "[MOVIE][DistanceSpace] pairwise_l2(sample="
        f"{len(pair_d)}): mean={float(np.mean(pair_d)):.3f}, "
        f"std={float(np.std(pair_d)):.3f}, min={float(np.min(pair_d)):.3f}, max={float(np.max(pair_d)):.3f}"
    )

    tree = KDTree(arr)
    query_n = min(n, 300)
    q_idx = rng.choice(n, size=query_n, replace=False)
    for radius in sorted(set(int(r) for r in radius_values)):
        counts = []
        for qi in q_idx:
            ids = tree.query_ball_point(arr[qi], r=radius)
            counts.append(max(0, len(ids) - 1))  # exclude self
        counts_arr = np.asarray(counts, dtype=float)
        print(
            f"[MOVIE][DistanceSpace] radius={radius}: "
            f"median_neighbors={float(np.median(counts_arr)):.2f}, "
            f"mean_neighbors={float(np.mean(counts_arr)):.2f}"
        )


def extract_method_coverage(record, method: str, default: float = 0.0) -> float:
    """
    Read coverage from either new or legacy leaderboard column names.
    """
    method_key = method.lower()
    candidates = [f"coverage_{method_key}", f"{method_key}_coverage"]
    for col in candidates:
        if col in record and pd.notna(record[col]):
            try:
                return float(record[col])
            except Exception:
                pass
    return float(default)


def sort_leaderboard_configs(leaderboard: pd.DataFrame, prioritize_ufce3: bool = False) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard

    df = drop_diag_columns_for_prod(leaderboard.copy())
    if "valid_config" not in df.columns:
        df["valid_config"] = True
    df["valid_config"] = df["valid_config"].apply(lambda v: parse_bool(v, default=True))
    df["_valid_rank"] = np.where(df["valid_config"], 0, 1)

    numeric_cols = [
        "score_total",
        "score_ufce1",
        "score_ufce2",
        "score_ufce3",
        "score_max",
        "score_mean",
        "score_max_penalized",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    primary_rank_col = "score_max_penalized"
    if primary_rank_col not in df.columns:
        primary_rank_col = "score_max" if "score_max" in df.columns else "score_total"
    if primary_rank_col not in df.columns:
        df[primary_rank_col] = float("nan")

    if "score_mean" not in df.columns:
        if "score_total" in df.columns:
            df["score_mean"] = pd.to_numeric(df["score_total"], errors="coerce")
        else:
            df["score_mean"] = float("nan")

    sort_cols = ["_valid_rank", primary_rank_col, "score_mean"]
    ascending = [True, True, True]
    if prioritize_ufce3 and "score_ufce3" in df.columns:
        sort_cols.append("score_ufce3")
        ascending.append(True)

    for col in ["score_total", "dataset", "seed", "radius", "n_neighbors", "min_act", "min_feas", "ufce_flip_filter"]:
        if col in df.columns and col not in sort_cols:
            sort_cols.append(col)
            ascending.append(True)

    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort", na_position="last").reset_index(drop=True)
    return df.drop(columns=["_valid_rank"], errors="ignore")


def backup_inconsistent_csv(csv_path: str) -> str | None:
    """
    If a CSV has inconsistent column counts across rows, move it to a backup file.
    """
    if not os.path.exists(csv_path):
        return None

    expected_fields = None
    mismatches: List[Tuple[int, int]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            field_count = len(row)
            if expected_fields is None:
                expected_fields = field_count
                continue
            if field_count != expected_fields:
                mismatches.append((line_no, field_count))
                if len(mismatches) >= 5:
                    break

    if not mismatches:
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{csv_path}.backup_{ts}"
    os.replace(csv_path, backup_path)

    mismatch_preview = ", ".join([f"line {ln}={fc}" for ln, fc in mismatches])
    print(
        f"[WARN] Inconsistent CSV detected: {csv_path} "
        f"(expected {expected_fields} fields; {mismatch_preview})."
    )
    print(f"[WARN] Moved to backup: {backup_path}")
    return backup_path


def get_dataset_constraints(dataset: str, datasetdf: pd.DataFrame):
    if dataset == "bank":
        return get_bank_user_constraints(datasetdf)
    if dataset == "grad":
        return get_grad_user_constraints(datasetdf)
    if dataset == "wine":
        return get_wine_user_constraints(datasetdf)
    if dataset == "bupa":
        return get_bupa_user_constraints(datasetdf)
    if dataset == "movie":
        return get_movie_user_constraints(datasetdf)
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_step_config(dataset: str) -> Dict[str, float]:
    if dataset == "bank":
        return {
            "Income": 1,
            "Family": 1,
            "CCAvg": 0.1,
            "Education": 1,
            "Mortgage": 1,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 1,
            "CreditCard": 1,
        }
    if dataset == "grad":
        return {
            "GRE Score": 1,
            "TOEFL Score": 1,
            "University Rating": 1,
            "SOP": 1,
            "LOR": 1,
            "CGPA": 0.1,
            "Research": 1,
        }
    if dataset == "wine":
        return {
            "fixed acidity": 0.5,
            "volatile acidity": 0.10,
            "citric acid": 0.1,
            "residual sugar": 0.5,
            "free sulfur dioxide": 1.0,
            "total sulfur dioxide": 1.0,
            "density": 0.1,
            "pH": 0.5,
            "alcohol": 0.5,
        }
    if dataset == "bupa":
        return {
            "Mcv": 1,
            "Alkphos": 1,
            "Sgpt": 1,
            "Sgot": 1,
            "Gammagt": 1,
            "Drinks": 1,
        }
    if dataset == "movie":
        return {
            "Production_expense": 3,
            "Num_multiplex": 3,
            "Multiplex_coverage": 0.2,
            "Movie_length": 5,
            "Lead_Actor_Rating": 1.0,
            "Lead_Actress_rating": 1.0,
            "Director_rating": 1.0,
            "Producer_rating": 1.0,
            "Genre": 1,
            "Collection": 500,
            "Budget": 3000,
        }
    return {}


def init_ufce_global(
    radius: int,
    n_neighbors: int,
    contprox_metric: str,
    min_act: int,
    min_feas: int,
    atol: float,
) -> None:
    cfmethods.initUFCE(
        radius=radius,
        n_neighbors=n_neighbors,
        contprox_metric=contprox_metric,
        min_act=min_act,
        min_feas=min_feas,
        atol=atol,
    )
    # Keep evaluation helpers aligned with generation parameters.
    eval_module.ufc = cfmethods.ufc


def run_one_fold(
    dataset: str,
    fold_name: str,
    fold_df: pd.DataFrame,
    X: pd.DataFrame,
    Xtest: pd.DataFrame,
    Xtrain: pd.DataFrame,
    data_lab1: pd.DataFrame,
    features: List[str],
    catf: List[str],
    numf: List[str],
    uf: Dict[str, float],
    f2change: List[str],
    protectf: List[str],
    bb_model,
    desired_outcome: float,
    mi_fp: List,
    no_cf: int,
    step: Dict[str, float],
    scaler: Dict | None,
    radius: int,
    n_neighbors: int,
    contprox_metric: str,
    min_act: int,
    min_feas: int,
    atol: float,
    ufce_flip_filter: int = 1,
    raw_fold_rows: int | None = None,
    n_explained_target: int = 0,
    movie_apf_unit_mode: str = MOVIE_APF_UNIT_MODE_DEFAULT,
    movie_apf_denom_mode: str = MOVIE_APF_DENOM_MODE_DEFAULT,
    movie_lof_space_mode: str = MOVIE_LOF_SPACE_MODE_DEFAULT,
    movie_lof_standardize_mode: str = MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
    movie_lof_contamination: str = MOVIE_LOF_CONTAMINATION_DEFAULT,
    prox_space_mode: str = PROX_SPACE_MODE_DEFAULT,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, object]]:
    use_movie_distance_space = bool(
        scaler is not None and str(scaler.get("kind", "")).lower() == "movie_minmax_0_100"
    )

    if use_movie_distance_space:
        # Keep UF constraints/model actions in raw space; use scaled geometry for Euclidean-only paths.
        fold_df_gen = fold_df[features].copy()
        data_lab1_gen = data_lab1[features].copy()
        X_gen = X[features].copy() if all(c in X.columns for c in features) else X.copy()
        fold_df_dist = apply_distance_scaler(fold_df[features], scaler)
        data_lab1_dist = apply_distance_scaler(data_lab1[features], scaler)
    else:
        # Backward-compatible path (non-movie behavior unchanged).
        if scaler is not None:
            fold_df_gen = apply_distance_scaler(fold_df[features], scaler)
            data_lab1_gen = apply_distance_scaler(data_lab1[features], scaler)
            X_gen = apply_distance_scaler(
                X[features] if all(c in X.columns for c in features) else X,
                scaler,
            )
        else:
            fold_df_gen = fold_df[features]
            data_lab1_gen = data_lab1[features]
            X_gen = X[features] if all(c in X.columns for c in features) else X
        fold_df_dist = None
        data_lab1_dist = None

    n_raw_test = int(raw_fold_rows) if raw_fold_rows is not None else int(len(fold_df))
    n_pred0_file = int(len(fold_df_gen))
    n_explained = int(len(fold_df_gen))
    debug_ctx = None

    init_ufce_global(
        radius=radius,
        n_neighbors=n_neighbors,
        contprox_metric=contprox_metric,
        min_act=min_act,
        min_feas=min_feas,
        atol=atol,
    )
    if getattr(cfmethods, "ufc", None) is not None:
        cfmethods.ufc.debug_ctx = debug_ctx

    onecfs, t1, idx1, s1 = cfmethods.sfexp(
        X_gen,
        data_lab1_gen,
        fold_df_gen[:],
        uf,
        step,
        f2change,
        numf,
        catf,
        bb_model,
        desired_outcome,
        no_cf,
        features,
        return_stats=True,
        flip_filter_enabled=bool(int(ufce_flip_filter)),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=fold_df_dist,
        distance_scaler=scaler if use_movie_distance_space else None,
        debug_ctx=None,
    )
    twocfs, t2, idx2, s2 = cfmethods.dfexp(
        X_gen,
        data_lab1_gen,
        fold_df_gen[:],
        uf,
        mi_fp[:5],
        numf,
        catf,
        f2change,
        protectf,
        bb_model,
        desired_outcome,
        no_cf,
        features,
        return_stats=True,
        flip_filter_enabled=bool(int(ufce_flip_filter)),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=fold_df_dist,
        distance_scaler=scaler if use_movie_distance_space else None,
    )
    threecfs, t3, idx3, s3 = cfmethods.tfexp(
        X_gen,
        data_lab1_gen,
        fold_df_gen[:],
        uf,
        mi_fp[:5],
        numf,
        catf,
        f2change,
        protectf,
        bb_model,
        desired_outcome,
        no_cf,
        features,
        return_stats=True,
        flip_filter_enabled=bool(int(ufce_flip_filter)),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=fold_df_dist,
        distance_scaler=scaler if use_movie_distance_space else None,
    )

    times = {"UFCE1": float(t1), "UFCE2": float(t2), "UFCE3": float(t3)}
    method_stats = {"UFCE1": s1, "UFCE2": s2, "UFCE3": s3}
    coverage_mode = "flip" if int(ufce_flip_filter) == 1 else "non_empty"

    for method in METHODS:
        ms = method_stats[method]
        n_instances = int(ms.get("n_instances", 0))
        flip_count = int(ms.get("n_instances_with_flip_cf", 0))
        non_empty_count_raw = ms.get("n_instances_with_non_empty_cf", ms.get("n_instances_with_usable_cf", None))
        if non_empty_count_raw is None or pd.isna(non_empty_count_raw):
            non_empty_count = flip_count
        else:
            non_empty_count = int(non_empty_count_raw)

        usable_count = flip_count if coverage_mode == "flip" else non_empty_count
        usable_coverage = float(usable_count / n_instances) if n_instances > 0 else 0.0
        ms["n_instances_with_flip_cf"] = flip_count
        ms["n_instances_with_usable_cf"] = usable_count
        ms["n_instances_with_non_empty_cf"] = non_empty_count
        ms["coverage"] = usable_coverage
        ms["coverage_mode"] = coverage_mode

        # print(
        #     f"    [Fold={fold_name}][{method}] "
        #     f"n_instances={n_instances}, "
        #     f"raw_total={int(ms['n_candidates_raw_total'])}, "
        #     f"flip_total={int(ms['n_candidates_flip_total'])}, "
        #     f"with_flip_cf={flip_count}, "
        #     f"with_usable_cf={usable_count}, "
        #     f"usable_coverage={usable_coverage:.4f}({coverage_mode}), "
        #     f"empty_after_filter={int(ms['n_empty_after_filter'])}"
        # )

    if scaler is not None and not use_movie_distance_space:
        for cf_df in [onecfs, twocfs, threecfs]:
            if not cf_df.empty:
                restored = inverse_distance_scaler(cf_df, scaler)
                cf_df.loc[:, restored.columns] = restored.loc[:, cf_df.columns]

    onetest = fold_df.loc[idx1].reset_index(drop=True)
    twotest = fold_df.loc[idx2].reset_index(drop=True)
    threetest = fold_df.loc[idx3].reset_index(drop=True)

    dummy = pd.DataFrame(columns=features)

    cat_means, _ = Catproximity(onecfs, onetest, twocfs, twotest, threecfs, threetest, dummy, dummy, dummy, dummy, Xtest, catf)
    if use_movie_distance_space:
        onecfs_dist = apply_distance_scaler(onecfs, scaler)
        twocfs_dist = apply_distance_scaler(twocfs, scaler)
        threecfs_dist = apply_distance_scaler(threecfs, scaler)
        onetest_dist = fold_df_dist.loc[idx1].reset_index(drop=True)
        twotest_dist = fold_df_dist.loc[idx2].reset_index(drop=True)
        threetest_dist = fold_df_dist.loc[idx3].reset_index(drop=True)
        Xtest_dist = apply_distance_scaler(Xtest, scaler)
        cont_means, _ = Contproximity(
            onecfs_dist,
            onetest_dist,
            twocfs_dist,
            twotest_dist,
            threecfs_dist,
            threetest_dist,
            dummy,
            dummy,
            dummy,
            dummy,
            Xtest_dist,
            numf,
        )
    else:
        cont_means, _ = Contproximity(onecfs, onetest, twocfs, twotest, threecfs, threetest, dummy, dummy, dummy, dummy, Xtest, numf)
    spar_means, _ = Sparsity(onecfs, onetest, twocfs, twotest, threecfs, threetest, dummy, dummy, dummy, dummy, Xtest, numf)
    act_means, _ = Actionability(onecfs, onetest, twocfs, twotest, threecfs, threetest, dummy, dummy, dummy, dummy, Xtest, features, f2change, uf)
    plaus_means, _ = Plausibility(onecfs, onetest, twocfs, twotest, threecfs, threetest, dummy, dummy, dummy, dummy, Xtest, Xtrain)
    feas_means, _ = Feasibility(
        onecfs,
        onetest,
        twocfs,
        twotest,
        threecfs,
        threetest,
        dummy,
        dummy,
        dummy,
        dummy,
        Xtest,
        Xtrain,
        features,
        f2change,
        bb_model,
        desired_outcome,
        uf,
    )

    def take3(arr):
        return list(arr[:3])

    per_metric = {
        "Prox-Jac": take3(cat_means),
        "Prox-Euc": take3(cont_means),
        "Sparsity": take3(spar_means),
        "Actionability": take3(act_means),
        "Plausibility": take3(plaus_means),
        "Feasibility": take3(feas_means),
    }

    means: Dict[str, Dict[str, float]] = {m: {} for m in METHODS}
    for mi, method in enumerate(METHODS):
        for metric in METRICS:
            means[method][metric] = float(per_metric[metric][mi])

    fold_artifacts: Dict[str, object] = {
        "fold_name": str(fold_name),
        "n_raw_test": int(n_raw_test),
        "n_pred0_file": int(n_pred0_file),
        "n_explained": int(n_explained),
        "features": list(features),
        "numf": list(numf),
        "catf": list(catf),
        "f2change": list(f2change),
        "uf": copy.deepcopy(uf),
        "Xtrain": Xtrain.copy(deep=True),
        "Xtest": Xtest.copy(deep=True),
        "onecfs": onecfs.copy(deep=True),
        "twocfs": twocfs.copy(deep=True),
        "threecfs": threecfs.copy(deep=True),
        "onetest": onetest.copy(deep=True),
        "twotest": twotest.copy(deep=True),
        "threetest": threetest.copy(deep=True),
        "idx_map": {
            "UFCE1": [int(i) for i in idx1],
            "UFCE2": [int(i) for i in idx2],
            "UFCE3": [int(i) for i in idx3],
        },
        "ufce_params": {
            "radius": int(radius),
            "n_neighbors": int(n_neighbors),
            "contprox_metric": str(contprox_metric),
            "min_act": int(min_act),
            "min_feas": int(min_feas),
            "atol": float(atol),
        },
        "distance_scaler": copy.deepcopy(scaler) if isinstance(scaler, dict) else None,
        "use_movie_distance_space": bool(use_movie_distance_space),
        "movie_lof_space_mode": str(movie_lof_space_mode),
        "movie_lof_standardize_mode": str(movie_lof_standardize_mode),
        "movie_lof_contamination": str(movie_lof_contamination),
        "movie_apf_unit_mode": str(movie_apf_unit_mode),
        "movie_apf_denom_mode": str(movie_apf_denom_mode),
        "prox_space_mode": str(prox_space_mode),
    }

    return means, times, method_stats, fold_artifacts


def _clone_diag_inputs(diag_inputs: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in diag_inputs.items():
        if isinstance(value, pd.DataFrame):
            out[key] = value.copy(deep=True)
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            try:
                out[key] = copy.deepcopy(value)
            except Exception:
                out[key] = value
    return out


def _build_local_ufce(ufce_params: Dict[str, object]) -> UFCE:
    return UFCE(
        radius=int(ufce_params.get("radius", 500)),
        n_neighbors=int(ufce_params.get("n_neighbors", 1000)),
        contprox_metric=str(ufce_params.get("contprox_metric", "euclidean")),
        min_actionable_other=int(ufce_params.get("min_act", 1)),
        min_actionable_feasible_other=int(ufce_params.get("min_feas", 1)),
        atol=float(ufce_params.get("atol", 1e-5)),
    )


def _mapped_flags_from_local_idx(local_passed_idx: List[int], idx_map: List[int], n_explained: int) -> np.ndarray:
    flags = np.zeros(int(n_explained), dtype=bool)
    for local_idx in local_passed_idx:
        if 0 <= int(local_idx) < len(idx_map):
            global_idx = int(idx_map[int(local_idx)])
            if 0 <= global_idx < int(n_explained):
                flags[global_idx] = True
    return flags


def _decision_stats(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"min": float("nan"), "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _build_lof_sweep_for_method(
    X_train: pd.DataFrame,
    test_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    n_neighbors: int,
    contamination_values: List[float],
) -> Dict[str, object]:
    if len(test_df) == 0 or len(cf_df) == 0 or len(X_train) == 0:
        return {
            str(c): {"count": 0, "rate_explained": float("nan"), "decision_stats": _decision_stats([])}
            for c in contamination_values
        }
    n_pairs = int(min(len(test_df), len(cf_df)))
    X_train_np = X_train.to_numpy(dtype=float, copy=True)
    test_np = test_df.iloc[:n_pairs].to_numpy(dtype=float, copy=True)
    cf_np = cf_df.iloc[:n_pairs].to_numpy(dtype=float, copy=True)

    scaler = StandardScaler()
    scaler.fit(X_train_np)
    X_train_scaled = scaler.transform(X_train_np)
    test_scaled = scaler.transform(test_np)
    cf_scaled = scaler.transform(cf_np)

    results: Dict[str, object] = {}
    for contamination in contamination_values:
        labels = []
        decisions = []
        for i in range(n_pairs):
            x_fit = np.vstack([test_scaled[i : i + 1], X_train_scaled])
            n_neighbors_eff = max(2, min(int(n_neighbors), int(x_fit.shape[0] - 1)))
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors_eff,
                novelty=True,
                contamination=float(contamination),
            )
            lof.fit(x_fit)
            label = int(np.asarray(lof.predict(cf_scaled[i : i + 1])).reshape(-1)[0])
            labels.append(label)
            try:
                decision = float(np.asarray(lof.decision_function(cf_scaled[i : i + 1])).reshape(-1)[0])
            except Exception:
                decision = float("nan")
            decisions.append(decision)
        inlier_count = int(sum(1 for x in labels if int(x) == 1))
        results[str(contamination)] = {
            "count": int(inlier_count),
            "rate_explained": safe_rate(inlier_count, n_pairs),
            "decision_stats": _decision_stats([d for d in decisions if np.isfinite(d)]),
        }
    return results


def run_movie_diag_sidecar(diag_inputs: Dict[str, object]) -> Dict[str, object]:
    dataset = str(diag_inputs.get("dataset", "")).strip().lower()
    if dataset != "movie":
        return {"dataset": dataset, "diag_skipped": True}

    local_ufc = _build_local_ufce(diag_inputs.get("ufce_params", {}))
    method_data = copy.deepcopy(diag_inputs.get("method_data", {}))
    features = list(diag_inputs.get("features", []))
    numf = list(diag_inputs.get("numf", []))
    f2change = list(diag_inputs.get("f2change", []))
    uf = copy.deepcopy(diag_inputs.get("uf", {}))
    desired_outcome = diag_inputs.get("desired_outcome", 1)
    bb_model = diag_inputs.get("bb_model")
    distance_scaler = copy.deepcopy(diag_inputs.get("distance_scaler", None))
    use_movie_distance_space = bool(diag_inputs.get("use_movie_distance_space", False))
    n_raw_test = int(diag_inputs.get("n_raw_test", 0))
    n_pred0_file = int(diag_inputs.get("n_pred0_file", 0))
    n_explained = int(diag_inputs.get("n_explained", 0))
    shadow_n = int(min(20, max(0, n_explained)))

    X_train = diag_inputs.get("Xtrain", pd.DataFrame()).copy(deep=True)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    out: Dict[str, object] = {
        "base_counts": {
            "N_raw_test": int(n_raw_test),
            "N_pred0_file": int(n_pred0_file),
            "N_explained": int(n_explained),
        },
        "apf_legend": {
            "A": "passes actionability constraints",
            "P": "LOF inlier",
            "F": "passes feasibility constraints",
        },
        "actionability": {},
        "lof_dual": {},
        "lof_sweep": {},
        "prox_dual": {},
        "apf_views": {},
        "shadow20": {"n_shadow": int(shadow_n), "methods": {}},
    }

    for method in METHODS:
        entry = method_data.get(method, {})
        test_df = entry.get("test", pd.DataFrame()).copy(deep=True)
        cf_df = entry.get("cf", pd.DataFrame()).copy(deep=True)
        idx_map = [int(x) for x in entry.get("idx_map", [])]
        n_pairs = int(min(len(test_df), len(cf_df), len(idx_map)))
        if n_pairs <= 0:
            out["actionability"][method] = {"count": 0, "fail_reason_counts": {"no_pairs": 0}, "n_pairs_input": 0}
            out["apf_views"][method] = {}
            out["lof_dual"][method] = {"raw_lof": {}, "distance_lof": {}, "n_pairs_valid_common": 0}
            out["lof_sweep"][method] = {}
            out["prox_dual"][method] = {"prox_euc_raw_diag": float("nan"), "prox_euc_distance_diag": float("nan")}
            out["shadow20"]["methods"][method] = {"A_count": 0, "P_count": 0, "F_count": 0}
            continue

        test_df = test_df.iloc[:n_pairs].reset_index(drop=True).copy(deep=True)
        cf_df = cf_df.iloc[:n_pairs].reset_index(drop=True).copy(deep=True)
        idx_map = idx_map[:n_pairs]

        act_cfs, _, act_local_idx, _ = local_ufc.actionability(
            cf_df.copy(deep=True),
            test_df.copy(deep=True),
            list(features),
            list(f2change),
            0,
            copy.deepcopy(uf),
            method="other",
        )
        _ = act_cfs  # kept for parity/debug potential
        a_flags = _mapped_flags_from_local_idx([int(i) for i in act_local_idx], idx_map, n_explained)
        out["actionability"][method] = {
            "count": int(int(np.sum(a_flags))),
            "n_pairs_input": int(n_pairs),
            "n_pairs_failed": int(max(0, n_pairs - len(act_local_idx))),
            "fail_reason_counts": {"fails_actionability_constraints": int(max(0, n_pairs - len(act_local_idx)))},
        }

        plaus_raw = local_ufc.implausibility(
            cf_df.copy(deep=True),
            test_df.copy(deep=True),
            X_train.copy(deep=True),
            len(cf_df),
            0,
            method_name=method,
            return_details=True,
            use_standard_scaler=True,
            lof_space_label="raw_lof",
        )
        p_flags = _mapped_flags_from_local_idx([int(i) for i in plaus_raw.get("passed_idx", [])], idx_map, n_explained)

        _, _, feas_raw = local_ufc.feasibility(
            test_df.copy(deep=True),
            cf_df.copy(deep=True),
            X_train.copy(deep=True),
            list(features),
            list(f2change),
            bb_model,
            desired_outcome,
            copy.deepcopy(uf),
            0,
            method="other",
            return_details=True,
            use_standard_scaler=True,
            lof_space_label="raw_lof",
        )
        f_flags = _mapped_flags_from_local_idx([int(i) for i in feas_raw.get("passed_idx", [])], idx_map, n_explained)

        ap = np.logical_and(a_flags, p_flags)
        af = np.logical_and(a_flags, f_flags)
        pf = np.logical_and(p_flags, f_flags)
        apf = np.logical_and(ap, f_flags)

        counts = {
            "A": int(np.sum(a_flags)),
            "P": int(np.sum(p_flags)),
            "F": int(np.sum(f_flags)),
            "AP": int(np.sum(ap)),
            "AF": int(np.sum(af)),
            "PF": int(np.sum(pf)),
            "APF": int(np.sum(apf)),
        }
        for key, value in counts.items():
            if int(value) > int(n_explained):
                raise RuntimeError(f"Diagnostic APF count overflow: method={method}, key={key}, value={value}, N_explained={n_explained}")

        out["apf_views"][method] = {
            "counts": counts,
            "rate_explained": {k: safe_rate(v, n_explained) for k, v in counts.items()},
            "rate_pred0": {k: safe_rate(v, n_pred0_file) for k, v in counts.items()},
            "rate_raw": {k: safe_rate(v, n_raw_test) for k, v in counts.items()},
            "A_flags": [bool(x) for x in a_flags.tolist()],
            "P_flags": [bool(x) for x in p_flags.tolist()],
            "F_flags": [bool(x) for x in f_flags.tolist()],
        }

        shadow_slice = slice(0, shadow_n)
        out["shadow20"]["methods"][method] = {
            "A_count": int(np.sum(a_flags[shadow_slice])),
            "P_count": int(np.sum(p_flags[shadow_slice])),
            "F_count": int(np.sum(f_flags[shadow_slice])),
            "APF_count": int(np.sum(apf[shadow_slice])),
        }

        method_cols = [c for c in numf if c in test_df.columns and c in cf_df.columns and c in X_train.columns]
        prox_raw_vals: List[float] = []
        prox_dist_vals: List[float] = []
        if len(method_cols) > 0:
            for i in range(n_pairs):
                try:
                    prox_raw_vals.append(
                        float(local_ufc.continuous_distance(test_df[i : i + 1], cf_df[i : i + 1], method_cols, metric="euclidean", agg=None))
                    )
                except Exception:
                    continue
            if use_movie_distance_space and distance_scaler is not None:
                t_dist = apply_distance_scaler(test_df.loc[:, method_cols], distance_scaler)
                c_dist = apply_distance_scaler(cf_df.loc[:, method_cols], distance_scaler)
                for i in range(len(t_dist)):
                    prox_dist_vals.append(
                        float(
                            np.linalg.norm(
                                c_dist.iloc[i].to_numpy(dtype=float) - t_dist.iloc[i].to_numpy(dtype=float)
                            )
                        )
                    )
        out["prox_dual"][method] = {
            "prox_euc_raw_diag": float(np.mean(prox_raw_vals)) if len(prox_raw_vals) > 0 else float("nan"),
            "prox_euc_distance_diag": float(np.mean(prox_dist_vals)) if len(prox_dist_vals) > 0 else float("nan"),
            "numf_count": int(len(method_cols)),
            "numf_hash": _stable_text_hash([str(c) for c in method_cols]) if len(method_cols) > 0 else "NA",
        }

        raw_pairs = {int(d.get("pair_pos", -1)): d for d in plaus_raw.get("pair_details", []) if isinstance(d, dict)}
        raw_valid = {k for k, v in raw_pairs.items() if bool(v.get("valid", False))}
        raw_input_df = pd.concat([test_df.loc[:, method_cols], cf_df.loc[:, method_cols]], ignore_index=True) if len(method_cols) > 0 else pd.DataFrame()
        lof_method_payload = {
            "raw_lof": {
                "lof_space_label": "raw_lof",
                "lof_input_shape": [int(raw_input_df.shape[0]), int(raw_input_df.shape[1])],
                "lof_input_numf_hash": _stable_text_hash(method_cols) if len(method_cols) > 0 else "NA",
                "lof_input_first_row_hash": dataframe_first_row_hash(raw_input_df),
                "n_pairs_input": int(plaus_raw.get("n_pairs_input", n_pairs)),
                "n_pairs_valid": int(plaus_raw.get("n_pairs_valid", len(raw_valid))),
            },
            "distance_lof": {
                "lof_space_label": "distance_lof",
                "lof_input_shape": [0, 0],
                "lof_input_numf_hash": "NA",
                "lof_input_first_row_hash": "NA",
                "n_pairs_input": 0,
                "n_pairs_valid": 0,
            },
            "n_pairs_valid_common": 0,
            "raw_inlier_common": 0,
            "distance_inlier_common": 0,
        }

        if use_movie_distance_space and distance_scaler is not None and len(method_cols) > 0:
            test_dist = apply_distance_scaler(test_df.loc[:, method_cols], distance_scaler)
            cf_dist = apply_distance_scaler(cf_df.loc[:, method_cols], distance_scaler)
            xtrain_dist = apply_distance_scaler(X_train.loc[:, method_cols], distance_scaler)
            plaus_dist = local_ufc.implausibility(
                cf_dist.copy(deep=True),
                test_dist.copy(deep=True),
                xtrain_dist.copy(deep=True),
                len(cf_dist),
                0,
                method_name=method,
                return_details=True,
                use_standard_scaler=False,
                lof_space_label="distance_lof",
            )
            dist_pairs = {int(d.get("pair_pos", -1)): d for d in plaus_dist.get("pair_details", []) if isinstance(d, dict)}
            dist_valid = {k for k, v in dist_pairs.items() if bool(v.get("valid", False))}
            common_valid = raw_valid & dist_valid
            raw_inlier_common = sum(1 for k in common_valid if int(raw_pairs[k].get("label", 0)) == 1)
            dist_inlier_common = sum(1 for k in common_valid if int(dist_pairs[k].get("label", 0)) == 1)
            dist_input_df = pd.concat([test_dist, cf_dist], ignore_index=True)

            lof_method_payload["distance_lof"] = {
                "lof_space_label": "distance_lof",
                "lof_input_shape": [int(dist_input_df.shape[0]), int(dist_input_df.shape[1])],
                "lof_input_numf_hash": _stable_text_hash(method_cols),
                "lof_input_first_row_hash": dataframe_first_row_hash(dist_input_df),
                "n_pairs_input": int(plaus_dist.get("n_pairs_input", n_pairs)),
                "n_pairs_valid": int(plaus_dist.get("n_pairs_valid", len(dist_valid))),
            }
            lof_method_payload["n_pairs_valid_common"] = int(len(common_valid))
            lof_method_payload["raw_inlier_common"] = int(raw_inlier_common)
            lof_method_payload["distance_inlier_common"] = int(dist_inlier_common)

        out["lof_dual"][method] = lof_method_payload

        if len(method_cols) > 0:
            out["lof_sweep"][method] = _build_lof_sweep_for_method(
                X_train=X_train.loc[:, method_cols].copy(deep=True),
                test_df=test_df.loc[:, method_cols].copy(deep=True),
                cf_df=cf_df.loc[:, method_cols].copy(deep=True),
                n_neighbors=int(local_ufc.n_neighbors),
                contamination_values=list(DIAG_LOF_SWEEP_VALUES),
            )
        else:
            out["lof_sweep"][method] = {}

    return out


def aggregate_results(
    folds: List[Tuple[str, Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, float]]]],
    coverage_mode: str = "flip",
):
    rows = []
    time_rows = []
    mode = str(coverage_mode).strip().lower()
    mode = "flip" if mode == "flip" else "non_empty"
    method_stats_total = {
        m: {
            "n_instances": 0,
            "n_candidates_raw_total": 0,
            "n_candidates_flip_total": 0,
            "n_instances_with_flip_cf": 0,
            "n_instances_with_usable_cf": 0,
            "n_instances_with_non_empty_cf": 0,
            "n_instances_with_selected_usable_cf": 0,
            "n_empty_after_filter": 0,
            "coverage": 0.0,
            "coverage_mode": mode,
        }
        for m in METHODS
    }

    for fold_name, means, times, method_stats in folds:
        for method in METHODS:
            row = {"fold": fold_name, "method": method}
            row.update(means[method])
            rows.append(row)
            ms = method_stats.get(method, {})
            method_stats_total[method]["n_instances"] += int(ms.get("n_instances", 0))
            method_stats_total[method]["n_candidates_raw_total"] += int(ms.get("n_candidates_raw_total", 0))
            method_stats_total[method]["n_candidates_flip_total"] += int(ms.get("n_candidates_flip_total", 0))
            flip_count = int(ms.get("n_instances_with_flip_cf", 0))
            usable_count = int(ms.get("n_instances_with_usable_cf", flip_count))
            non_empty_count = int(ms.get("n_instances_with_non_empty_cf", usable_count))
            method_stats_total[method]["n_instances_with_flip_cf"] += flip_count
            method_stats_total[method]["n_instances_with_usable_cf"] += usable_count
            method_stats_total[method]["n_instances_with_non_empty_cf"] += non_empty_count
            method_stats_total[method]["n_empty_after_filter"] += int(ms.get("n_empty_after_filter", 0))
        time_rows.append(times)

    for method in METHODS:
        denom = int(method_stats_total[method]["n_instances"])
        if mode == "flip":
            selected_usable = int(method_stats_total[method]["n_instances_with_flip_cf"])
        else:
            selected_usable = int(method_stats_total[method]["n_instances_with_usable_cf"])
        method_stats_total[method]["n_instances_with_selected_usable_cf"] = selected_usable
        method_stats_total[method]["coverage"] = float(selected_usable / denom) if denom > 0 else 0.0

    df = pd.DataFrame(rows)
    time_df = pd.DataFrame(time_rows)
    mean_df = df.groupby("method")[METRICS].mean().T
    std_df = df.groupby("method")[METRICS].std(ddof=1).T
    return mean_df[METHODS], std_df[METHODS], time_df[METHODS].mean(), method_stats_total


def compute_score(dataset: str, mean_df: pd.DataFrame, eps: float) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    targets = AUTHOR_TABLE7.get(dataset)
    if targets is None:
        raise ValueError(f"No AUTHOR_TABLE7 values registered for dataset='{dataset}'.")

    all_errors: List[float] = []
    method_scores: Dict[str, float] = {}
    cell_errors: Dict[str, float] = {}

    for method in METHODS:
        method_errors: List[float] = []
        for metric in METRICS:
            target = float(targets[method][metric])
            ours = float(mean_df.loc[metric, method])
            if np.isnan(target) or np.isnan(ours):
                continue
            delta = ours - target
            if abs(target) <= eps:
                err = abs(delta)
            else:
                err = abs(delta) / max(eps, abs(target))
            method_errors.append(float(err))
            all_errors.append(float(err))
            cell_errors[f"err_{slugify(method)}_{slugify(metric)}"] = float(err)
        method_scores[f"score_{method.lower()}"] = float(np.mean(method_errors)) if method_errors else float("nan")

    score_total = float(np.mean(all_errors)) if all_errors else float("inf")
    return score_total, method_scores, cell_errors


def build_comparison_table(dataset: str, mean_df: pd.DataFrame, eps: float) -> pd.DataFrame:
    targets = AUTHOR_TABLE7.get(dataset)
    if targets is None:
        raise ValueError(f"No AUTHOR_TABLE7 values registered for dataset='{dataset}'.")

    rows = []
    for method in METHODS:
        for metric in METRICS:
            author = float(targets[method][metric])
            ours = float(mean_df.loc[metric, method])

            if np.isnan(author) or np.isnan(ours):
                delta = float("nan")
                norm_err = float("nan")
            else:
                delta = ours - author
                if abs(author) <= eps:
                    norm_err = abs(delta)
                else:
                    norm_err = abs(delta) / max(eps, abs(author))

            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "ours": ours,
                    "author": author,
                    "delta_ours_minus_author": delta,
                    "norm_error": norm_err,
                }
            )
    return pd.DataFrame(rows)


def compute_method_level_ranking(
    comparison_df: pd.DataFrame,
    method_stats_total: Dict[str, Dict[str, float]],
    hard_min_coverage: float,
    coverage_penalty_lambda: float,
) -> Dict[str, object]:
    """
    Reproduction-mode ranking:
    - per-method mean normalized errors (UFCE1/UFCE2/UFCE3)
    - minimax score over methods (L-infinity)
    - optional coverage penalty
    - validity flag for degenerate outputs
    """
    out: Dict[str, object] = {}
    comparison_df = drop_diag_columns_for_prod(comparison_df.copy())
    method_scores: List[float] = []
    method_coverages: List[float] = []
    hard_threshold = float(hard_min_coverage)
    hard_filter_active = hard_threshold > 0.0
    valid_config = True
    any_unrankable = False

    for method in METHODS:
        method_key = method.lower()
        method_df = comparison_df.loc[comparison_df["method"] == method].copy()
        method_df["author"] = pd.to_numeric(method_df.get("author"), errors="coerce")
        method_df["norm_error"] = pd.to_numeric(method_df["norm_error"], errors="coerce")
        required_mask = np.isfinite(method_df["author"].to_numpy(dtype=float, copy=False))
        required_metric_count = int(np.sum(required_mask))
        finite_mask_required = required_mask & np.isfinite(method_df["norm_error"].to_numpy(dtype=float, copy=False))
        finite_count = int(np.sum(finite_mask_required))
        finite_ratio = float(finite_count / required_metric_count) if required_metric_count > 0 else 0.0

        coverage = float(method_stats_total.get(method, {}).get("coverage", 0.0))
        coverage_zero = abs(coverage) < COVERAGE_EPS

        reason_parts = []
        if coverage_zero:
            reason_parts.append("coverage_zero")
        if required_metric_count == 0:
            reason_parts.append("metrics_no_reference")
        if finite_count < int(MIN_REQUIRED_FINITE):
            reason_parts.append("metrics_too_few")
        if finite_ratio < float(MIN_REQUIRED_RATIO):
            reason_parts.append("metrics_insufficient_ratio")

        is_rankable = (
            (not coverage_zero)
            and (required_metric_count > 0)
            and (finite_count >= int(MIN_REQUIRED_FINITE))
            and (finite_ratio >= float(MIN_REQUIRED_RATIO))
        )

        if is_rankable:
            method_score = float(method_df.loc[finite_mask_required, "norm_error"].mean())
        else:
            method_score = float("inf")
            any_unrankable = True
            valid_config = False

        reason = "ok" if len(reason_parts) == 0 else "+".join(reason_parts)
        out[f"invalid_reason_{method_key}"] = reason
        out[f"score_{method_key}_required_n"] = int(required_metric_count)
        out[f"score_{method_key}_rankable"] = bool(is_rankable)
        out[f"score_{method_key}"] = method_score
        method_scores.append(method_score)
        out[f"score_{method_key}_n"] = int(finite_count)

        out[f"coverage_{method_key}"] = coverage
        method_coverages.append(coverage)
        out[f"{method_key}_n_instances_with_selected_usable_cf"] = int(
            method_stats_total.get(method, {}).get("n_instances_with_selected_usable_cf", 0)
        )

    min_coverage = float(min(method_coverages)) if method_coverages else 0.0
    out["min_coverage"] = min_coverage
    min_coverage_zero = abs(min_coverage) < COVERAGE_EPS
    if min_coverage_zero:
        valid_config = False
    if min_coverage < hard_threshold:
        valid_config = False
    if hard_filter_active and min_coverage_zero:
        valid_config = False

    degenerate_all_methods = bool(all(abs(c) < COVERAGE_EPS for c in method_coverages)) if method_coverages else True
    out["degenerate_all_methods"] = degenerate_all_methods
    if degenerate_all_methods:
        valid_config = False

    summary_parts = []
    for method in METHODS:
        mk = method.lower()
        reason = str(out.get(f"invalid_reason_{mk}", "ok"))
        if reason != "ok":
            summary_parts.append(f"{method}:{reason}")
    out["invalid_reason_summary"] = "ok" if len(summary_parts) == 0 else ", ".join(summary_parts)
    out["any_method_unrankable"] = bool(any_unrankable)

    if any(np.isnan(v) for v in method_scores):
        score_max = float("inf")
        score_mean = float("inf")
    else:
        score_max = float(np.max(method_scores))
        score_mean = float(np.mean(method_scores))

    coverage_penalty = float(coverage_penalty_lambda) * (1.0 - min_coverage)
    score_max_penalized = float(score_max + coverage_penalty) if np.isfinite(score_max) else float("inf")

    out["score_max"] = score_max
    out["score_mean"] = score_mean
    out["score_max_penalized"] = score_max_penalized
    out["valid_config"] = bool(valid_config)
    return out


def append_dataframe(csv_path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    if not exists:
        df.to_csv(csv_path, mode="w", header=True, index=False)
        return

    try:
        existing = pd.read_csv(csv_path, nrows=0)
        existing_cols = list(existing.columns)
    except Exception:
        backup_inconsistent_csv(csv_path)
        df.to_csv(csv_path, mode="w", header=True, index=False)
        return

    incoming_cols = list(df.columns)
    if existing_cols != incoming_cols:
        try:
            prev = pd.read_csv(csv_path)
            merged = pd.concat([prev, df], ignore_index=True, sort=False)
            merged.to_csv(csv_path, index=False)
        except Exception:
            backup_inconsistent_csv(csv_path)
            df.to_csv(csv_path, mode="w", header=True, index=False)
        return

    df.to_csv(csv_path, mode="a", header=False, index=False)


def rebuild_comparison_from_leaderboard(
    leaderboard_csv: str,
    comparison_csv: str,
    best_comparison_csv: str,
    run_stage: str,
    eps: float,
    prioritize_ufce3: bool = False,
) -> None:
    if not os.path.exists(leaderboard_csv):
        return

    try:
        lb = pd.read_csv(leaderboard_csv)
    except pd.errors.ParserError:
        backup_path = backup_inconsistent_csv(leaderboard_csv)
        if backup_path is not None:
            print(f"[WARN] Skipping comparison rebuild because leaderboard was malformed: {leaderboard_csv}")
        return
    if lb.empty:
        return
    lb = drop_diag_columns_for_prod(lb)
    if "ufce_flip_filter" not in lb.columns:
        lb["ufce_flip_filter"] = lb.apply(lambda r: get_ufce_flip_filter_from_record(r, default=1), axis=1)
    lb = ensure_effective_config_fields_df(lb)
    lb = drop_diag_columns_for_prod(lb)
    lb = lb.sort_values(RUN_SORT_FIELDS, ascending=True, kind="mergesort", na_position="last")
    lb_latest = lb.groupby(EFFECTIVE_CONFIG_FIELDS, sort=False, dropna=False).tail(1).reset_index(drop=True)

    rows = []
    for _, cfg in lb_latest.iterrows():
        dataset = str(cfg.get("dataset", ""))
        targets = AUTHOR_TABLE7.get(dataset)
        if targets is None:
            continue

        for method in METHODS:
            for metric in METRICS:
                mean_col = f"mean_{slugify(method)}_{slugify(metric)}"
                ours = float(cfg[mean_col]) if mean_col in lb.columns and pd.notna(cfg[mean_col]) else float("nan")
                author = float(targets[method][metric])
                cov1 = extract_method_coverage(cfg, "UFCE1", default=float("nan"))
                cov2 = extract_method_coverage(cfg, "UFCE2", default=float("nan"))
                cov3 = extract_method_coverage(cfg, "UFCE3", default=float("nan"))
                coverage_mode = str(cfg.get("coverage_mode", "")).strip().lower()
                if coverage_mode not in {"flip", "non_empty"}:
                    coverage_mode = "flip" if get_ufce_flip_filter_from_record(cfg, default=1) == 1 else "non_empty"
                derived_min_cov = (
                    float(np.nanmin([cov1, cov2, cov3])) if not np.isnan(cov1) or not np.isnan(cov2) or not np.isnan(cov3) else float("nan")
                )

                if np.isnan(author) or np.isnan(ours):
                    delta = float("nan")
                    norm_err = float("nan")
                else:
                    delta = ours - author
                    if abs(author) <= eps:
                        norm_err = abs(delta)
                    else:
                        norm_err = abs(delta) / max(eps, abs(author))

                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(cfg.get("seed", 0)),
                        "run_stage": run_stage,
                        "run_started_at_utc": sanitize_mode_string(cfg.get("run_started_at_utc", "00000000T000000Z"), "00000000T000000Z"),
                        "run_id": sanitize_mode_string(cfg.get("run_id", "00000000T000000Z_00000000"), "00000000T000000Z_00000000"),
                        "run_seq_in_file": int(pd.to_numeric(cfg.get("run_seq_in_file", 0), errors="coerce")),
                        "radius": int(cfg.get("radius", 0)),
                        "n_neighbors": int(cfg.get("n_neighbors", 0)),
                        "contprox_metric": str(cfg.get("contprox_metric", "")),
                        "min_act": int(cfg.get("min_act", 0)),
                        "min_feas": int(cfg.get("min_feas", 0)),
                        "ufce_flip_filter": get_ufce_flip_filter_from_record(cfg, default=1),
                        "folds_selected": sanitize_mode_string(cfg.get("folds_selected", "unknown"), "unknown"),
                        "n_folds_selected": (
                            0
                            if pd.isna(pd.to_numeric(cfg.get("n_folds_selected", 0), errors="coerce"))
                            else int(pd.to_numeric(cfg.get("n_folds_selected", 0), errors="coerce"))
                        ),
                        "folds_source": sanitize_mode_string(cfg.get("folds_source", FOLDS_SOURCE_DEFAULT), FOLDS_SOURCE_DEFAULT),
                        "N_explained_target": (
                            0
                            if pd.isna(pd.to_numeric(cfg.get("N_explained_target", 0), errors="coerce"))
                            else int(pd.to_numeric(cfg.get("N_explained_target", 0), errors="coerce"))
                        ),
                        "movie_distance_scaler_mode": sanitize_mode_string(
                            cfg.get("movie_distance_scaler_mode", MOVIE_DISTANCE_SCALER_MODE_DEFAULT),
                            MOVIE_DISTANCE_SCALER_MODE_DEFAULT,
                        ),
                        "movie_lof_space_mode": sanitize_mode_string(
                            cfg.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT),
                            MOVIE_LOF_SPACE_MODE_DEFAULT,
                        ),
                        "movie_lof_standardize_mode": sanitize_mode_string(
                            cfg.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT),
                            MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
                        ),
                        "movie_lof_contamination": sanitize_mode_string(
                            cfg.get("movie_lof_contamination", MOVIE_LOF_CONTAMINATION_DEFAULT),
                            MOVIE_LOF_CONTAMINATION_DEFAULT,
                        ),
                        "movie_apf_unit_mode": sanitize_mode_string(
                            cfg.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT),
                            MOVIE_APF_UNIT_MODE_DEFAULT,
                        ),
                        "movie_apf_denom_mode": sanitize_mode_string(
                            cfg.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT),
                            MOVIE_APF_DENOM_MODE_DEFAULT,
                        ),
                        "prox_space_mode": sanitize_mode_string(
                            cfg.get("prox_space_mode", PROX_SPACE_MODE_DEFAULT),
                            PROX_SPACE_MODE_DEFAULT,
                        ),
                        "score_total": float(cfg.get("score_total", float("nan"))),
                        "score_ufce1": float(cfg.get("score_ufce1", float("nan"))),
                        "score_ufce2": float(cfg.get("score_ufce2", float("nan"))),
                        "score_ufce3": float(cfg.get("score_ufce3", float("nan"))),
                        "score_max": float(cfg.get("score_max", cfg.get("score_total", float("nan")))),
                        "score_mean": float(cfg.get("score_mean", cfg.get("score_total", float("nan")))),
                        "score_max_penalized": float(
                            cfg.get("score_max_penalized", cfg.get("score_max", cfg.get("score_total", float("nan"))))
                        ),
                        "coverage_mode": coverage_mode,
                        "coverage_ufce1": cov1,
                        "coverage_ufce2": cov2,
                        "coverage_ufce3": cov3,
                        "min_coverage": float(cfg.get("min_coverage", derived_min_cov)),
                        "valid_config": parse_bool(cfg.get("valid_config", True), default=True),
                        "method": method,
                        "metric": metric,
                        "ours": ours,
                        "author": author,
                        "delta_ours_minus_author": delta,
                        "norm_error": norm_err,
                    }
                )

    if not rows:
        return

    comp = pd.DataFrame(rows)
    comp = drop_diag_columns_for_prod(comp)
    comp = ensure_effective_config_fields_df(comp)
    if "valid_config" not in comp.columns:
        comp["valid_config"] = True
    comp["valid_config"] = comp["valid_config"].apply(lambda v: parse_bool(v, default=True))
    comp["_valid_rank"] = np.where(comp["valid_config"], 0, 1)

    for col in ["score_total", "score_ufce1", "score_ufce2", "score_ufce3", "score_max", "score_mean", "score_max_penalized"]:
        if col in comp.columns:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")

    primary_rank_col = "score_max_penalized" if "score_max_penalized" in comp.columns else "score_max"
    if primary_rank_col not in comp.columns:
        primary_rank_col = "score_total"
    if primary_rank_col not in comp.columns:
        comp[primary_rank_col] = float("nan")

    uniqueness_key = EFFECTIVE_CONFIG_FIELDS + ["method", "metric"]
    dup_counts = comp.groupby(uniqueness_key, dropna=False).size().reset_index(name="n")
    bad = dup_counts.loc[dup_counts["n"] != 1].copy()
    if not bad.empty:
        raise RuntimeError(
            "Comparison uniqueness assertion failed for (effective_config, method, metric). "
            f"Examples:\n{bad.head(10).to_string(index=False)}"
        )

    per_cfg_counts = comp.groupby(EFFECTIVE_CONFIG_FIELDS, dropna=False).size().reset_index(name="n")
    bad_rows = per_cfg_counts.loc[per_cfg_counts["n"] != EXPECTED_COMPARISON_ROWS_PER_CONFIG].copy()
    if not bad_rows.empty:
        raise RuntimeError(
            f"Comparison row-count assertion failed (expected {EXPECTED_COMPARISON_ROWS_PER_CONFIG}). "
            f"Examples:\n{bad_rows.head(10).to_string(index=False)}"
        )

    sort_cols = ["_valid_rank", primary_rank_col, "score_mean"]
    if prioritize_ufce3 and "score_ufce3" in comp.columns:
        sort_cols.append("score_ufce3")
    sort_cols.extend(
        [
            "dataset",
            "seed",
            "radius",
            "n_neighbors",
            "min_act",
            "min_feas",
            "ufce_flip_filter",
            "method",
            "metric",
        ]
    )
    comp = comp.sort_values(sort_cols, ascending=True, kind="mergesort", na_position="last").reset_index(drop=True)
    comp = comp.drop(columns=["_valid_rank"], errors="ignore")
    os.makedirs(os.path.dirname(comparison_csv), exist_ok=True)
    comp.to_csv(comparison_csv, index=False)

    lb_sorted = sort_leaderboard_configs(lb_latest, prioritize_ufce3=prioritize_ufce3)
    best_row = lb_sorted.iloc[0]
    best_record = ensure_effective_config_fields_record(best_row.to_dict())
    best_ufce_flip_filter = get_ufce_flip_filter_from_record(best_record, default=1)
    mask = (
        (comp["dataset"].astype(str) == str(best_record["dataset"]))
        & (comp["seed"].astype(int) == int(best_record["seed"]))
        & (comp["radius"].astype(int) == int(best_record["radius"]))
        & (comp["n_neighbors"].astype(int) == int(best_record["n_neighbors"]))
        & (comp["contprox_metric"].astype(str) == str(best_record["contprox_metric"]))
        & (comp["min_act"].astype(int) == int(best_record["min_act"]))
        & (comp["min_feas"].astype(int) == int(best_record["min_feas"]))
        & (comp["ufce_flip_filter"].astype(int) == best_ufce_flip_filter)
        & (comp["folds_selected"].astype(str) == str(best_record.get("folds_selected", "unknown")))
        & (comp["n_folds_selected"].astype(int) == int(best_record.get("n_folds_selected", 0)))
        & (comp["folds_source"].astype(str) == str(best_record.get("folds_source", FOLDS_SOURCE_DEFAULT)))
        & (comp["N_explained_target"].astype(int) == int(best_record.get("N_explained_target", 0)))
        & (
            comp["movie_distance_scaler_mode"].astype(str)
            == str(best_record.get("movie_distance_scaler_mode", MOVIE_DISTANCE_SCALER_MODE_DEFAULT))
        )
        & (comp["movie_lof_space_mode"].astype(str) == str(best_record.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT)))
        & (
            comp["movie_lof_standardize_mode"].astype(str)
            == str(best_record.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT))
        )
        & (
            comp["movie_lof_contamination"].astype(str)
            == str(best_record.get("movie_lof_contamination", MOVIE_LOF_CONTAMINATION_DEFAULT))
        )
        & (comp["movie_apf_unit_mode"].astype(str) == str(best_record.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT)))
        & (comp["movie_apf_denom_mode"].astype(str) == str(best_record.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT)))
        & (comp["prox_space_mode"].astype(str) == str(best_record.get("prox_space_mode", PROX_SPACE_MODE_DEFAULT)))
    )
    comp.loc[mask].to_csv(best_comparison_csv, index=False)


def print_topk_comparison_tables(
    leaderboard_csv: str,
    comparison_csv: str,
    top_k: int = 3,
    prioritize_ufce3: bool = False,
) -> None:
    if top_k <= 0:
        return
    if not os.path.exists(leaderboard_csv):
        print(f"[WARN] Cannot print top-{top_k} comparison tables: missing {leaderboard_csv}")
        return
    if not os.path.exists(comparison_csv):
        print(f"[WARN] Cannot print top-{top_k} comparison tables: missing {comparison_csv}")
        return

    try:
        lb = pd.read_csv(leaderboard_csv)
    except pd.errors.ParserError:
        backup_path = backup_inconsistent_csv(leaderboard_csv)
        if backup_path is not None:
            print(f"[WARN] Cannot print top-{top_k} comparison tables: malformed leaderboard was backed up.")
        return
    try:
        comp = pd.read_csv(comparison_csv)
    except pd.errors.ParserError:
        backup_path = backup_inconsistent_csv(comparison_csv)
        if backup_path is not None:
            print(f"[WARN] Cannot print top-{top_k} comparison tables: malformed comparison CSV was backed up.")
        return
    if lb.empty or comp.empty:
        print(f"[WARN] Cannot print top-{top_k} comparison tables: leaderboard/comparison is empty.")
        return
    lb = drop_diag_columns_for_prod(lb)
    comp = drop_diag_columns_for_prod(comp)
    if "ufce_flip_filter" not in lb.columns:
        lb["ufce_flip_filter"] = lb.apply(lambda r: get_ufce_flip_filter_from_record(r, default=1), axis=1)
    if "ufce_flip_filter" not in comp.columns:
        comp["ufce_flip_filter"] = comp.apply(lambda r: get_ufce_flip_filter_from_record(r, default=1), axis=1)
    lb = ensure_effective_config_fields_df(lb)
    comp = ensure_effective_config_fields_df(comp)
    lb = lb.sort_values(RUN_SORT_FIELDS, ascending=True, kind="mergesort", na_position="last")
    lb_latest = lb.groupby(EFFECTIVE_CONFIG_FIELDS, sort=False, dropna=False).tail(1).reset_index(drop=True)

    lb_sorted = sort_leaderboard_configs(lb_latest, prioritize_ufce3=prioritize_ufce3)
    top_df = lb_sorted.head(top_k)
    print(f"\n=== Top-{len(top_df)} Full Comparison Tables ===")

    def fmt_float(value) -> str:
        try:
            val = float(value)
        except Exception:
            return "nan"
        if np.isnan(val):
            return "nan"
        if np.isinf(val):
            return "inf"
        return f"{val:.6f}"

    for rank, row in top_df.iterrows():
        row_rec = ensure_effective_config_fields_record(row.to_dict())
        dataset = str(row_rec["dataset"])
        seed = int(row_rec["seed"])
        radius = int(row_rec["radius"])
        n_neighbors = int(row_rec["n_neighbors"])
        contprox_metric = str(row_rec["contprox_metric"])
        min_act = int(row_rec["min_act"])
        min_feas = int(row_rec["min_feas"])
        ufce_flip_filter = int(row_rec["ufce_flip_filter"])
        folds_selected = str(row_rec.get("folds_selected", "unknown"))
        n_folds_selected = int(row_rec.get("n_folds_selected", 0))
        folds_source = str(row_rec.get("folds_source", FOLDS_SOURCE_DEFAULT))
        n_explained_target = int(row_rec.get("N_explained_target", 0))
        movie_distance_scaler_mode = str(row_rec.get("movie_distance_scaler_mode", MOVIE_DISTANCE_SCALER_MODE_DEFAULT))
        movie_lof_space_mode = str(row_rec.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT))
        movie_lof_standardize_mode = str(row_rec.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT))
        movie_lof_contamination = str(row_rec.get("movie_lof_contamination", MOVIE_LOF_CONTAMINATION_DEFAULT))
        movie_apf_unit_mode = str(row_rec.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT))
        movie_apf_denom_mode = str(row_rec.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT))
        prox_space_mode = str(row_rec.get("prox_space_mode", PROX_SPACE_MODE_DEFAULT))
        score_total = float(row.get("score_total", float("nan")))
        score_ufce1 = float(row.get("score_ufce1", float("nan")))
        score_ufce2 = float(row.get("score_ufce2", float("nan")))
        score_ufce3 = float(row.get("score_ufce3", float("nan")))
        score_max = float(row.get("score_max", float("nan")))
        score_mean = float(row.get("score_mean", float("nan")))
        score_max_penalized = float(row.get("score_max_penalized", score_max))
        coverage_mode = str(row.get("coverage_mode", "")).strip().lower()
        if coverage_mode not in {"flip", "non_empty"}:
            coverage_mode = "flip" if ufce_flip_filter == 1 else "non_empty"
        coverage_ufce1 = extract_method_coverage(row, "UFCE1", default=float("nan"))
        coverage_ufce2 = extract_method_coverage(row, "UFCE2", default=float("nan"))
        coverage_ufce3 = extract_method_coverage(row, "UFCE3", default=float("nan"))
        if "min_coverage" in row and pd.notna(row["min_coverage"]):
            min_coverage = float(row["min_coverage"])
        else:
            cov_vals = [coverage_ufce1, coverage_ufce2, coverage_ufce3]
            min_coverage = float(np.nanmin(cov_vals)) if not all(np.isnan(v) for v in cov_vals) else float("nan")
        valid_config = parse_bool(row.get("valid_config", True), default=True)
        invalid_reason_summary = str(row.get("invalid_reason_summary", "ok"))
        if "degenerate_all_methods" in row:
            degenerate_all_methods = parse_bool(row.get("degenerate_all_methods", False), default=False)
        else:
            degenerate_all_methods = bool(
                (abs(coverage_ufce1) < COVERAGE_EPS)
                and (abs(coverage_ufce2) < COVERAGE_EPS)
                and (abs(coverage_ufce3) < COVERAGE_EPS)
            )

        mask = (
            (comp["dataset"].astype(str) == dataset)
            & (comp["seed"].astype(int) == seed)
            & (comp["radius"].astype(int) == radius)
            & (comp["n_neighbors"].astype(int) == n_neighbors)
            & (comp["contprox_metric"].astype(str) == contprox_metric)
            & (comp["min_act"].astype(int) == min_act)
            & (comp["min_feas"].astype(int) == min_feas)
            & (comp["ufce_flip_filter"].astype(int) == ufce_flip_filter)
            & (comp["folds_selected"].astype(str) == folds_selected)
            & (comp["n_folds_selected"].astype(int) == n_folds_selected)
            & (comp["folds_source"].astype(str) == folds_source)
            & (comp["N_explained_target"].astype(int) == n_explained_target)
            & (comp["movie_distance_scaler_mode"].astype(str) == movie_distance_scaler_mode)
            & (comp["movie_lof_space_mode"].astype(str) == movie_lof_space_mode)
            & (comp["movie_lof_standardize_mode"].astype(str) == movie_lof_standardize_mode)
            & (comp["movie_lof_contamination"].astype(str) == movie_lof_contamination)
            & (comp["movie_apf_unit_mode"].astype(str) == movie_apf_unit_mode)
            & (comp["movie_apf_denom_mode"].astype(str) == movie_apf_denom_mode)
            & (comp["prox_space_mode"].astype(str) == prox_space_mode)
        )
        cfg_comp = comp.loc[mask].copy()
        if cfg_comp.empty:
            print(
                f"\n--- Rank {rank + 1} | valid_config={valid_config} | "
                f"score_ufce1={fmt_float(score_ufce1)}, score_ufce2={fmt_float(score_ufce2)}, score_ufce3={fmt_float(score_ufce3)}, "
                f"score_max={fmt_float(score_max)}, score_mean={fmt_float(score_mean)}, "
                f"score_max_penalized={fmt_float(score_max_penalized)}, min_coverage={fmt_float(min_coverage)} | "
                f"degenerate_all_methods={degenerate_all_methods}, invalid_reason_summary={invalid_reason_summary} | "
                f"coverage_mode={coverage_mode}, "
                f"radius={radius}, n_neighbors={n_neighbors}, min_act={min_act}, min_feas={min_feas}, "
                f"ufce_flip_filter={ufce_flip_filter} ---"
            )
            print("[WARN] No comparison rows found for this config.")
            continue

        if len(cfg_comp) != EXPECTED_COMPARISON_ROWS_PER_CONFIG:
            pair_counts = (
                cfg_comp.groupby(["method", "metric"], dropna=False)
                .size()
                .reset_index(name="count")
                .loc[lambda d: d["count"] > 1]
            )
            print(
                f"[WARN] Broken comparison row set for rank={rank + 1}: "
                f"expected_rows={EXPECTED_COMPARISON_ROWS_PER_CONFIG}, actual_rows={len(cfg_comp)}, "
                f"effective_key=({dataset},{seed},{radius},{n_neighbors},{min_act},{min_feas},"
                f"{ufce_flip_filter},{folds_selected},{n_folds_selected},{folds_source},{n_explained_target},"
                f"{movie_distance_scaler_mode},{movie_lof_space_mode},{movie_lof_standardize_mode},{movie_lof_contamination},"
                f"{movie_apf_unit_mode},{movie_apf_denom_mode},{prox_space_mode})"
            )
            if not pair_counts.empty:
                print("[WARN] duplicated method/metric pairs:")
                print(pair_counts.to_string(index=False))
            continue

        cfg_comp["method"] = pd.Categorical(cfg_comp["method"], categories=METHODS, ordered=True)
        cfg_comp["metric"] = pd.Categorical(cfg_comp["metric"], categories=METRICS, ordered=True)
        cfg_comp = cfg_comp.sort_values(["method", "metric"], kind="mergesort")
        display_cols = ["method", "metric", "ours", "author", "delta_ours_minus_author", "norm_error"]

        print(
            f"\n--- Rank {rank + 1} | valid_config={valid_config} | "
            f"score_ufce1={fmt_float(score_ufce1)}, score_ufce2={fmt_float(score_ufce2)}, score_ufce3={fmt_float(score_ufce3)}, "
            f"score_max={fmt_float(score_max)}, score_mean={fmt_float(score_mean)}, "
            f"score_max_penalized={fmt_float(score_max_penalized)}, score_total={fmt_float(score_total)}, "
            f"coverage_ufce1={fmt_float(coverage_ufce1)}, coverage_ufce2={fmt_float(coverage_ufce2)}, coverage_ufce3={fmt_float(coverage_ufce3)}, "
            f"min_coverage={fmt_float(min_coverage)}, degenerate_all_methods={degenerate_all_methods}, "
            f"invalid_reason_summary={invalid_reason_summary}, coverage_mode={coverage_mode} | "
            f"radius={radius}, n_neighbors={n_neighbors}, min_act={min_act}, min_feas={min_feas}, "
            f"ufce_flip_filter={ufce_flip_filter} ---"
        )
        print(cfg_comp[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        top_contrib = (
            cfg_comp.assign(norm_error=pd.to_numeric(cfg_comp["norm_error"], errors="coerce"))
            .dropna(subset=["norm_error"])
            .sort_values("norm_error", ascending=False, kind="mergesort")
            .head(3)
        )
        if not top_contrib.empty:
            contrib_parts = []
            for _, contrib_row in top_contrib.iterrows():
                contrib_parts.append(
                    f"{str(contrib_row['method'])}/{str(contrib_row['metric'])}={float(contrib_row['norm_error']):.6f}"
                )
            print(f"top_contributors(norm_error): {', '.join(contrib_parts)}")


def prefer_existing_path(primary_path: str, legacy_path: str) -> str:
    if os.path.exists(primary_path):
        return primary_path
    if os.path.exists(legacy_path):
        return legacy_path
    return primary_path


def append_and_sort_leaderboard(csv_path: str, row: Dict[str, float], prioritize_ufce3: bool = False) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        try:
            leaderboard = pd.read_csv(csv_path)
        except pd.errors.ParserError:
            backup_inconsistent_csv(csv_path)
            leaderboard = pd.DataFrame(columns=list(row.keys()))
    else:
        leaderboard = pd.DataFrame(columns=list(row.keys()))
    leaderboard = drop_diag_columns_for_prod(leaderboard)
    leaderboard = pd.concat([leaderboard, pd.DataFrame([row])], ignore_index=True, sort=False)
    leaderboard = drop_diag_columns_for_prod(leaderboard)
    leaderboard = sort_leaderboard_configs(leaderboard, prioritize_ufce3=prioritize_ufce3)
    leaderboard.to_csv(csv_path, index=False)
    return leaderboard


def write_best_json(json_path: str, leaderboard_df: pd.DataFrame, run_stage: str) -> None:
    if leaderboard_df.empty:
        return
    best = leaderboard_df.iloc[0]
    best_ufce_flip_filter = get_ufce_flip_filter_from_record(best, default=1)
    min_cov = float(best.get("min_coverage", float("nan")))
    score_max = float(best.get("score_max", float("nan")))
    score_mean = float(best.get("score_mean", float("nan")))
    score_max_penalized = float(best.get("score_max_penalized", score_max))
    coverage_mode = str(best.get("coverage_mode", "")).strip().lower()
    if coverage_mode not in {"flip", "non_empty"}:
        coverage_mode = "flip" if best_ufce_flip_filter == 1 else "non_empty"
    valid_config = parse_bool(best.get("valid_config", True), default=True)
    if run_stage == "run1":
        payload = {
            "min_act": int(best["min_act"]),
            "min_feas": int(best["min_feas"]),
            "ufce_flip_filter": int(best_ufce_flip_filter),
            "coverage_mode": coverage_mode,
            "score_max_penalized": score_max_penalized,
            "score_max": score_max,
            "score_mean": score_mean,
            "score_total": float(best["score_total"]),
            "min_coverage": min_cov,
            "valid_config": bool(valid_config),
        }
    else:
        payload = {
            "radius": int(best["radius"]),
            "n_neighbors": int(best["n_neighbors"]),
            "min_act": int(best["min_act"]),
            "min_feas": int(best["min_feas"]),
            "ufce_flip_filter": int(best_ufce_flip_filter),
            "coverage_mode": coverage_mode,
            "score_max_penalized": score_max_penalized,
            "score_max": score_max,
            "score_mean": score_mean,
            "score_total": float(best["score_total"]),
            "min_coverage": min_cov,
            "valid_config": bool(valid_config),
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def config_key(dataset: str, seed: int, cfg: Dict[str, int | str]) -> Tuple:
    return (
        dataset,
        int(seed),
        int(cfg["radius"]),
        int(cfg["n_neighbors"]),
        str(cfg["contprox_metric"]),
        int(cfg["min_act"]),
        int(cfg["min_feas"]),
        int(cfg.get("ufce_flip_filter", cfg.get("ufce3_flip_filter", 1))),
        sanitize_mode_string(cfg.get("folds_selected", "unknown"), "unknown"),
        sanitize_mode_string(cfg.get("folds_source", FOLDS_SOURCE_DEFAULT), FOLDS_SOURCE_DEFAULT),
        sanitize_mode_string(
            cfg.get("movie_distance_scaler_mode", MOVIE_DISTANCE_SCALER_MODE_DEFAULT),
            MOVIE_DISTANCE_SCALER_MODE_DEFAULT,
        ),
        sanitize_mode_string(cfg.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT), MOVIE_LOF_SPACE_MODE_DEFAULT),
        sanitize_mode_string(
            cfg.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT),
            MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
        ),
        sanitize_mode_string(cfg.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT), MOVIE_APF_UNIT_MODE_DEFAULT),
        sanitize_mode_string(cfg.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT), MOVIE_APF_DENOM_MODE_DEFAULT),
    )


def load_existing_keys(csv_path: str) -> set:
    keys = set()
    if not os.path.exists(csv_path):
        return keys
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        backup_inconsistent_csv(csv_path)
        return keys
    required = {"dataset", "seed", "radius", "n_neighbors", "contprox_metric", "min_act", "min_feas"}
    if not required.issubset(set(df.columns)):
        return keys
    df = drop_diag_columns_for_prod(df)
    df = ensure_effective_config_fields_df(df)
    for _, row in df.iterrows():
        keys.add(
            (
                str(row["dataset"]),
                int(row["seed"]),
                int(row["radius"]),
                int(row["n_neighbors"]),
                str(row["contprox_metric"]),
                int(row["min_act"]),
                int(row["min_feas"]),
                get_ufce_flip_filter_from_record(row, default=1),
                sanitize_mode_string(row.get("folds_selected", "unknown"), "unknown"),
                sanitize_mode_string(row.get("folds_source", FOLDS_SOURCE_DEFAULT), FOLDS_SOURCE_DEFAULT),
                sanitize_mode_string(
                    row.get("movie_distance_scaler_mode", MOVIE_DISTANCE_SCALER_MODE_DEFAULT),
                    MOVIE_DISTANCE_SCALER_MODE_DEFAULT,
                ),
                sanitize_mode_string(row.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT), MOVIE_LOF_SPACE_MODE_DEFAULT),
                sanitize_mode_string(
                    row.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT),
                    MOVIE_LOF_STANDARDIZE_MODE_DEFAULT,
                ),
                sanitize_mode_string(row.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT), MOVIE_APF_UNIT_MODE_DEFAULT),
                sanitize_mode_string(row.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT), MOVIE_APF_DENOM_MODE_DEFAULT),
            )
        )
    return keys


def resolve_stage(args) -> str:
    if args.run_stage != "auto":
        return args.run_stage
    if args.radius_grid is not None or args.n_neighbors_grid is not None:
        return "run2"
    return "run1"


def build_pred0_testfolds(
    model,
    data_lab0: pd.DataFrame,
    outcome_label: str,
    features: List[str],
    out_dir: str,
    seed: int,
    desired_outcome: float | int | str | None = None,
    max_rows: int = 0,
    n_folds: int = 1,
) -> List[str]:
    """
    Build pred0 testfold CSVs from data_lab0 using the trained LR model.
    """
    if outcome_label not in data_lab0.columns:
        raise ValueError(f"Outcome label '{outcome_label}' not found in data_lab0 columns.")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")

    os.makedirs(out_dir, exist_ok=True)

    data_lab0 = data_lab0.reset_index(drop=True).copy()
    data_lab0.to_csv(os.path.join(out_dir, "data_lab0_raw.csv"), index=False)

    if all(col in data_lab0.columns for col in features):
        x_lab0 = data_lab0[features].copy()
    else:
        x_lab0 = data_lab0.drop(columns=[outcome_label]).copy()

    y_lab0 = data_lab0[outcome_label].copy()
    preds = pd.Series(np.asarray(model.predict(x_lab0)).reshape(-1), index=data_lab0.index)

    # Infer the class representing "pred0" from the provided data_lab0 slice.
    unique_labels = list(pd.Series(y_lab0).dropna().unique())
    if not unique_labels:
        raise RuntimeError("data_lab0 has no valid outcome labels.")

    def _equal(a, b) -> bool:
        try:
            return float(a) == float(b)
        except Exception:
            return str(a) == str(b)

    if len(unique_labels) == 1:
        pred0_label = unique_labels[0]
    else:
        pred0_label = unique_labels[0]
        if desired_outcome is not None:
            non_desired = [v for v in unique_labels if not _equal(v, desired_outcome)]
            if non_desired:
                pred0_label = non_desired[0]

    correct_mask = preds == y_lab0
    pred0_mask = y_lab0.apply(lambda v: _equal(v, pred0_label))
    pred0_df = data_lab0.loc[correct_mask & pred0_mask].copy()

    if pred0_df.empty:
        raise RuntimeError(
            f"No correctly classified pred0 rows found from data_lab0 for label={pred0_label}."
        )

    pred0_features = pred0_df.drop(columns=[outcome_label]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_rows > 0:
        pred0_features = pred0_features.iloc[:max_rows].reset_index(drop=True)

    if pred0_features.empty:
        raise RuntimeError("Pred0 dataset became empty after applying max_rows.")

    fold_paths: List[str] = []
    splits = np.array_split(pred0_features, n_folds)
    for i, split_df in enumerate(splits):
        split_df = split_df.reset_index(drop=True)
        if split_df.empty:
            continue
        fp = os.path.join(out_dir, f"testfold_{i}_pred_0.csv")
        split_df.to_csv(fp, index=False)
        fold_paths.append(fp)

    if not fold_paths:
        raise RuntimeError("No non-empty pred0 folds were written.")

    stats = {
        "total_data_lab0_rows": int(len(data_lab0)),
        "correctly_classified_pred0_rows": int(len(pred0_df)),
        "written_pred0_rows": int(len(pred0_features)),
        "pred0_label": str(pred0_label),
        "desired_outcome": str(desired_outcome),
        "n_folds_requested": int(n_folds),
        "n_folds_written": int(len(fold_paths)),
        "max_rows": int(max_rows),
    }
    with open(os.path.join(out_dir, "pred0_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        "Generated pred0 testfolds: "
        f"pred0_label={pred0_label}, desired_outcome={desired_outcome}, "
        f"lab0={len(data_lab0)}, correct_pred0={len(pred0_df)}, "
        f"written_rows={len(pred0_features)}, folds={len(fold_paths)}"
    )
    return fold_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="grad", choices=["bank", "grad", "wine", "bupa", "movie"])
    parser.add_argument("--data_dir", type=str, default=os.path.join("ufce", "data"))
    parser.add_argument("--folds_dir", type=str, default=os.path.join("ufce", "data", "folds"))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("archive", "part1_old_runs", "hypertune_out", "grad_run1"),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_cf", type=int, default=10)
    parser.add_argument("--max_folds", type=int, default=0, help="0=all folds")

    parser.add_argument("--run_stage", type=str, default="auto", choices=["auto", "run1", "run2"])
    parser.add_argument("--contprox_metric", type=str, default="euclidean")
    parser.add_argument("--radius", type=int, default=500)
    parser.add_argument("--n_neighbors", type=int, default=1000)
    parser.add_argument("--radius_grid", type=str, default=None)
    parser.add_argument("--n_neighbors_grid", type=str, default=None)

    parser.add_argument("--min_act", type=int, default=None)
    parser.add_argument("--min_feas", type=int, default=None)
    parser.add_argument("--min_act_grid", type=str, default="0,1,2,3")
    parser.add_argument("--min_feas_grid", type=str, default="0,1,2")
    parser.add_argument("--enforce_min_act_ge_min_feas", type=int, default=1)
    parser.add_argument("--ufce_flip_filter", type=int, default=1, help="1=apply flipping-candidate filter in UFCE1/UFCE2/UFCE3, 0=disable for testing")
    parser.add_argument("--ufce3_flip_filter", dest="ufce_flip_filter", type=int, help=argparse.SUPPRESS)

    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--coverage_penalty_lambda",
        type=float,
        default=0.2,
        help="Penalty lambda in score_max_penalized = score_max + lambda*(1-min_coverage).",
    )
    parser.add_argument(
        "--hard_min_coverage",
        type=float,
        default=0.0,
        help="Mark config invalid when min_coverage < this threshold.",
    )
    parser.add_argument(
        "--rank_tiebreak_ufce3",
        type=int,
        default=0,
        help="1=add score_ufce3 as an extra tie-break after score_mean.",
    )
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--print_top_k", type=int, default=3, help="Print full comparison tables for top-K leaderboard configs")
    parser.add_argument("--rebuild_comparison_only", type=int, default=0, help="1=rebuild comparison CSVs from existing leaderboard and exit")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_stage = resolve_stage(args)
    diag_enabled = bool(str(args.dataset).strip().lower() == "movie" and bool(DEBUG_MOVIE_DIAG))
    os.makedirs(args.out_dir, exist_ok=True)

    leaderboard_name = "leaderboard_run1.csv" if run_stage == "run1" else "leaderboard_run2.csv"
    best_name = "best_run1.json" if run_stage == "run1" else "best_run2.json"
    leaderboard_primary = os.path.join(args.out_dir, leaderboard_name)
    best_json_primary = os.path.join(args.out_dir, best_name)
    comparison_primary = os.path.join(args.out_dir, f"comparison_{run_stage}_all_configs.csv")
    best_comparison_primary = os.path.join(args.out_dir, f"comparison_{run_stage}_best.csv")

    # Backward-compatible file names from previous script revisions.
    leaderboard_legacy = os.path.join(args.out_dir, f"{args.dataset}_leaderboard_{run_stage}.csv")
    best_json_legacy = os.path.join(args.out_dir, f"{args.dataset}_best_{run_stage}.json")
    comparison_legacy = os.path.join(args.out_dir, f"{args.dataset}_comparison_{run_stage}_all_configs.csv")
    best_comparison_legacy = os.path.join(args.out_dir, f"{args.dataset}_comparison_{run_stage}_best.csv")

    leaderboard_csv = prefer_existing_path(leaderboard_primary, leaderboard_legacy)
    best_json = prefer_existing_path(best_json_primary, best_json_legacy)
    comparison_csv = prefer_existing_path(comparison_primary, comparison_legacy)
    best_comparison_csv = prefer_existing_path(best_comparison_primary, best_comparison_legacy)

    # Keep logic simple: malformed old files are moved aside and regenerated.
    backup_inconsistent_csv(leaderboard_csv)
    backup_inconsistent_csv(comparison_csv)
    backup_inconsistent_csv(best_comparison_csv)

    if int(args.rebuild_comparison_only) == 1:
        rebuild_comparison_from_leaderboard(
            leaderboard_csv=leaderboard_csv,
            comparison_csv=comparison_csv,
            best_comparison_csv=best_comparison_csv,
            run_stage=run_stage,
            eps=float(args.eps),
            prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
        )
        print_topk_comparison_tables(
            leaderboard_csv=leaderboard_csv,
            comparison_csv=comparison_csv,
            top_k=int(args.print_top_k),
            prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
        )
        print("Rebuilt comparison tables from leaderboard.")
        print(f"Leaderboard CSV:   {leaderboard_csv}")
        print(f"Compare (all):     {comparison_csv}")
        print(f"Compare (best):    {best_comparison_csv}")
        return

    if run_stage == "run1":
        min_act_grid = parse_int_grid(args.min_act_grid, default_values=[0, 1, 2, 3])
        min_feas_grid = parse_int_grid(args.min_feas_grid, default_values=[0, 1, 2])
        configs = []
        for min_act, min_feas in itertools.product(min_act_grid, min_feas_grid):
            if int(args.enforce_min_act_ge_min_feas) == 1 and min_act < min_feas:
                continue
            configs.append(
                {
                    "radius": args.radius,
                    "n_neighbors": args.n_neighbors,
                    "contprox_metric": args.contprox_metric,
                    "min_act": min_act,
                    "min_feas": min_feas,
                    "ufce_flip_filter": int(args.ufce_flip_filter),
                }
            )
    else:
        if args.min_act is None or args.min_feas is None:
            raise ValueError("Run 2 requires --min_act and --min_feas.")
        if args.radius_grid is None or args.n_neighbors_grid is None:
            raise ValueError("Run 2 requires both --radius_grid and --n_neighbors_grid.")
        radius_grid = parse_int_grid(args.radius_grid, default_values=[])
        n_neighbors_grid = parse_int_grid(args.n_neighbors_grid, default_values=[])
        configs = [
            {
                "radius": radius,
                "n_neighbors": n_neighbors,
                "contprox_metric": args.contprox_metric,
                "min_act": args.min_act,
                "min_feas": args.min_feas,
                "ufce_flip_filter": int(args.ufce_flip_filter),
            }
            for radius, n_neighbors in itertools.product(radius_grid, n_neighbors_grid)
        ]

    if not configs:
        raise ValueError("No configurations to run after applying constraints.")

    run_started_at_utc = get_run_started_at_utc()
    run_id = build_run_id(run_started_at_utc=run_started_at_utc, seed=int(args.seed))
    run_seq_counter = 0

    print(f"Run stage: {run_stage}")
    print(f"Total configurations: {len(configs)}")
    print(f"Leaderboard: {leaderboard_csv}")
    print(
        "Ranking mode: valid_config(True first), score_max_penalized, score_mean"
        + (", score_ufce3" if bool(int(args.rank_tiebreak_ufce3)) else "")
    )
    print(
        f"Coverage policy: hard_min_coverage={float(args.hard_min_coverage):.4f}, "
        f"coverage_penalty_lambda={float(args.coverage_penalty_lambda):.4f}"
    )
    print(f"Movie diagnostics side-car: {'ON' if diag_enabled else 'OFF'}")

    data_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
    datasetdf = pd.read_csv(data_path)
    lr, lr_mean, lr_std, Xtest, Xtrain, X, Y, df, scaler = classify_dataset_getModel(datasetdf, data_name=args.dataset)
    del Y, df

    (
        features,
        catf,
        numf,
        uf,
        f2change,
        _outcome_label,
        desired_outcome,
        _nbr_features,
        protectf,
        _data_lab0,
        data_lab1,
    ) = get_dataset_constraints(args.dataset, datasetdf)
    step = get_step_config(args.dataset)
    mi_fp = UFCE().get_top_MI_features(X, features)

    # Movie-only distance geometry fix:
    # use full-dataset global min-max scaling on UFCE numeric distance columns to [0,100].
    if args.dataset == "movie":
        scaler = build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=_outcome_label,
        )
        print(
            "[MOVIE][DistanceSpace] matrix=datasetdf[features], "
            f"scaled_numeric={len(scaler['scale_cols'])}, "
            f"unchanged_non_numeric={len(features) - len(scaler['scale_cols'])}"
        )
        print(
            "[MOVIE][DistanceSpace] scaler=global min-max on full dataset "
            "(constant numeric columns pinned to 0)."
        )
        X_distance = apply_distance_scaler(datasetdf.loc[:, features], scaler)
        radius_values = [int(cfg["radius"]) for cfg in configs]
        log_movie_distance_sanity(
            X_distance=X_distance,
            features=features,
            radius_values=radius_values,
            seed=int(args.seed),
        )
    # Non-movie datasets keep the original scaler returned by
    # classify_dataset_getModel(...) to preserve prior behavior.

    print(f"Dataset: {args.dataset}")
    print(f"CV accuracy: {lr_mean:.6f} +/- {lr_std:.6f}")

    testfold_path = os.path.join(args.folds_dir, args.dataset, "totest")
    testfolds = sorted(glob.glob(os.path.join(testfold_path, "*.csv")))
    folds_source = "max_folds" if args.max_folds > 0 else "auto"
    if args.max_folds > 0:
        testfolds = testfolds[: args.max_folds]
    if not testfolds:
        raise FileNotFoundError(f"No test folds found in {testfold_path}")
    print(f"Folds selected: {len(testfolds)}")
    folds_selected = build_folds_selected_signature(testfolds)

    movie_distance_scaler_mode = "minmax100_global" if args.dataset == "movie" else MOVIE_DISTANCE_SCALER_MODE_DEFAULT
    movie_lof_space_mode = MOVIE_LOF_SPACE_MODE_DEFAULT
    movie_lof_standardize_mode = MOVIE_LOF_STANDARDIZE_MODE_DEFAULT
    movie_lof_contamination = MOVIE_LOF_CONTAMINATION_DEFAULT
    movie_apf_unit_mode = MOVIE_APF_UNIT_MODE_DEFAULT
    movie_apf_denom_mode = MOVIE_APF_DENOM_MODE_DEFAULT
    prox_space_mode = PROX_SPACE_MODE_DEFAULT
    n_explained_target = 0
    for cfg in configs:
        cfg["folds_selected"] = folds_selected
        cfg["n_folds_selected"] = int(len(testfolds))
        cfg["folds_source"] = folds_source
        cfg["N_explained_target"] = int(n_explained_target)
        cfg["movie_distance_scaler_mode"] = movie_distance_scaler_mode
        cfg["movie_lof_space_mode"] = movie_lof_space_mode
        cfg["movie_lof_standardize_mode"] = movie_lof_standardize_mode
        cfg["movie_lof_contamination"] = movie_lof_contamination
        cfg["movie_apf_unit_mode"] = movie_apf_unit_mode
        cfg["movie_apf_denom_mode"] = movie_apf_denom_mode
        cfg["prox_space_mode"] = prox_space_mode

    existing = load_existing_keys(leaderboard_csv) if int(args.skip_existing) == 1 else set()
    if existing:
        print(f"Loaded {len(existing)} existing configurations from leaderboard.")

    t0 = time.time()
    completed = 0
    skipped = 0

    for idx, cfg in enumerate(configs, start=1):
        key = config_key(args.dataset, args.seed, cfg)
        prod_config_key_text = repr(key)
        if key in existing:
            skipped += 1
            print(f"[{idx}/{len(configs)}] Skip existing config: {cfg}")
            continue

        print(f"[{idx}/{len(configs)}] Running config: {cfg}")
        cfg_t0 = time.time()
        fold_results = []

        for fold_i, fp in enumerate(testfolds, start=1):
            fold_name = os.path.basename(fp)
            print(f"  - Fold {fold_i}/{len(testfolds)}: {fold_name}")
            fold_df = pd.read_csv(fp)
            raw_fold_rows = infer_raw_fold_rows(args.dataset, fold_name, fallback_rows=len(fold_df))
            fold_id = int(extract_fold_id_from_name(fold_name))

            means, times, method_stats, fold_artifacts = run_one_fold(
                dataset=args.dataset,
                fold_name=fold_name,
                fold_df=fold_df,
                X=X,
                Xtest=Xtest,
                Xtrain=Xtrain,
                data_lab1=data_lab1,
                features=features,
                catf=catf,
                numf=numf,
                uf=uf,
                f2change=f2change,
                protectf=protectf,
                bb_model=lr,
                desired_outcome=desired_outcome,
                mi_fp=mi_fp,
                no_cf=args.no_cf,
                step=step,
                scaler=scaler,
                radius=int(cfg["radius"]),
                n_neighbors=int(cfg["n_neighbors"]),
                contprox_metric=str(cfg["contprox_metric"]),
                min_act=int(cfg["min_act"]),
                min_feas=int(cfg["min_feas"]),
                atol=float(args.atol),
                ufce_flip_filter=int(cfg.get("ufce_flip_filter", cfg.get("ufce3_flip_filter", 1))),
                raw_fold_rows=raw_fold_rows,
                n_explained_target=int(cfg.get("N_explained_target", 0)),
                movie_apf_unit_mode=str(cfg.get("movie_apf_unit_mode", MOVIE_APF_UNIT_MODE_DEFAULT)),
                movie_apf_denom_mode=str(cfg.get("movie_apf_denom_mode", MOVIE_APF_DENOM_MODE_DEFAULT)),
                movie_lof_space_mode=str(cfg.get("movie_lof_space_mode", MOVIE_LOF_SPACE_MODE_DEFAULT)),
                movie_lof_standardize_mode=str(cfg.get("movie_lof_standardize_mode", MOVIE_LOF_STANDARDIZE_MODE_DEFAULT)),
                movie_lof_contamination=str(cfg.get("movie_lof_contamination", MOVIE_LOF_CONTAMINATION_DEFAULT)),
                prox_space_mode=str(cfg.get("prox_space_mode", PROX_SPACE_MODE_DEFAULT)),
            )
            fold_results.append((fold_name, means, times, method_stats))

            fold_comp_df = build_comparison_table(
                args.dataset,
                pd.DataFrame(
                    {m: [float(means[m][metric]) for metric in METRICS] for m in METHODS},
                    index=METRICS,
                ),
                eps=float(args.eps),
            )
            fold_snapshot_before = build_fold_snapshot(
                comparison_df_fold=fold_comp_df,
                method_stats_fold=method_stats,
                fold_file=fold_name,
                fold_id=fold_id,
                folds_selected=folds_selected,
                prod_config_key=prod_config_key_text,
            )

            if diag_enabled:
                raw_diag_inputs = {
                    "dataset": str(args.dataset),
                    "fold_name": str(fold_name),
                    "fold_id": int(fold_id),
                    "features": fold_artifacts["features"],
                    "numf": fold_artifacts["numf"],
                    "catf": fold_artifacts["catf"],
                    "f2change": fold_artifacts["f2change"],
                    "uf": fold_artifacts["uf"],
                    "Xtrain": fold_artifacts["Xtrain"],
                    "Xtest": fold_artifacts["Xtest"],
                    "bb_model": lr,
                    "desired_outcome": desired_outcome,
                    "ufce_params": fold_artifacts["ufce_params"],
                    "distance_scaler": fold_artifacts["distance_scaler"],
                    "use_movie_distance_space": fold_artifacts["use_movie_distance_space"],
                    "n_raw_test": fold_artifacts["n_raw_test"],
                    "n_pred0_file": fold_artifacts["n_pred0_file"],
                    "n_explained": fold_artifacts["n_explained"],
                    "method_data": {
                        "UFCE1": {
                            "test": fold_artifacts["onetest"],
                            "cf": fold_artifacts["onecfs"],
                            "idx_map": fold_artifacts["idx_map"]["UFCE1"],
                        },
                        "UFCE2": {
                            "test": fold_artifacts["twotest"],
                            "cf": fold_artifacts["twocfs"],
                            "idx_map": fold_artifacts["idx_map"]["UFCE2"],
                        },
                        "UFCE3": {
                            "test": fold_artifacts["threetest"],
                            "cf": fold_artifacts["threecfs"],
                            "idx_map": fold_artifacts["idx_map"]["UFCE3"],
                        },
                    },
                }
                diag_inputs = _clone_diag_inputs(raw_diag_inputs)
                assert id(diag_inputs["method_data"]["UFCE1"]["test"]) != id(fold_artifacts["onetest"])
                assert id(diag_inputs["method_data"]["UFCE1"]["cf"]) != id(fold_artifacts["onecfs"])
                with preserve_rng_state():
                    with temporary_single_thread_env():
                        diag = run_movie_diag_sidecar(diag_inputs)

                diag_payload = {
                    "run_id": str(run_id),
                    "diag_key": f"{run_id}|{prod_config_key_text}|{fold_id}",
                    "prod_config_key": str(prod_config_key_text),
                    "fold_id": int(fold_id),
                    "fold_name": str(fold_name),
                    "prod_snapshot_hash": str(fold_snapshot_before.get("comparison_prod_columns_hash", "NA")),
                    "diag": diag,
                }
                diag_payload = sanitize_diag_payload(diag_payload)
                if _contains_frame_or_array(diag_payload):
                    raise RuntimeError("Diagnostic payload contains DataFrame/ndarray objects after sanitization.")
                sidecar_path = write_diag_sidecar_json(
                    out_dir=args.out_dir,
                    run_id=run_id,
                    diag_key=str(diag_payload["diag_key"]),
                    fold_id=fold_id,
                    payload=diag_payload,
                )
                print(
                    "[DBG][MOVIE-DIAG] "
                    f"saved_sidecar={sidecar_path}, fold={fold_name}, prod_snapshot_hash={diag_payload['prod_snapshot_hash']}"
                )

                fold_snapshot_after = build_fold_snapshot(
                    comparison_df_fold=fold_comp_df,
                    method_stats_fold=method_stats,
                    fold_file=fold_name,
                    fold_id=fold_id,
                    folds_selected=folds_selected,
                    prod_config_key=prod_config_key_text,
                )
                validate_snapshot_unchanged(
                    fold_snapshot_before,
                    fold_snapshot_after,
                    layer="fold",
                    prod_config_key=prod_config_key_text,
                    fold_id=fold_id,
                    fold_file=fold_name,
                )

        cfg_coverage_mode = "flip" if int(cfg.get("ufce_flip_filter", cfg.get("ufce3_flip_filter", 1))) == 1 else "non_empty"
        mean_df, std_df, time_mean, method_stats_total = aggregate_results(fold_results, coverage_mode=cfg_coverage_mode)
        score_total, per_method_scores, cell_errors = compute_score(args.dataset, mean_df, eps=float(args.eps))
        comparison_df = drop_diag_columns_for_prod(build_comparison_table(args.dataset, mean_df, eps=float(args.eps)))
        ranking_stats = compute_method_level_ranking(
            comparison_df=comparison_df,
            method_stats_total=method_stats_total,
            hard_min_coverage=float(args.hard_min_coverage),
            coverage_penalty_lambda=float(args.coverage_penalty_lambda),
        )

        run_seq_counter += 1
        row = {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "run_started_at_utc": run_started_at_utc,
            "run_id": run_id,
            "run_seq_in_file": int(run_seq_counter),
            "radius": int(cfg["radius"]),
            "n_neighbors": int(cfg["n_neighbors"]),
            "contprox_metric": str(cfg["contprox_metric"]),
            "min_act": int(cfg["min_act"]),
            "min_feas": int(cfg["min_feas"]),
            "ufce_flip_filter": int(cfg.get("ufce_flip_filter", cfg.get("ufce3_flip_filter", 1))),
            "folds_selected": str(cfg.get("folds_selected", folds_selected)),
            "n_folds_selected": int(cfg.get("n_folds_selected", len(testfolds))),
            "folds_source": str(cfg.get("folds_source", folds_source)),
            "N_explained_target": int(cfg.get("N_explained_target", 0)),
            "movie_distance_scaler_mode": str(cfg.get("movie_distance_scaler_mode", movie_distance_scaler_mode)),
            "movie_lof_space_mode": str(cfg.get("movie_lof_space_mode", movie_lof_space_mode)),
            "movie_lof_standardize_mode": str(cfg.get("movie_lof_standardize_mode", movie_lof_standardize_mode)),
            "movie_lof_contamination": str(cfg.get("movie_lof_contamination", movie_lof_contamination)),
            "movie_apf_unit_mode": str(cfg.get("movie_apf_unit_mode", movie_apf_unit_mode)),
            "movie_apf_denom_mode": str(cfg.get("movie_apf_denom_mode", movie_apf_denom_mode)),
            "prox_space_mode": str(cfg.get("prox_space_mode", prox_space_mode)),
            "score_total": float(score_total),
            "time_ufce1_mean_sec": float(time_mean["UFCE1"]),
            "time_ufce2_mean_sec": float(time_mean["UFCE2"]),
            "time_ufce3_mean_sec": float(time_mean["UFCE3"]),
            "runtime_sec": float(time.time() - cfg_t0),
        }
        row.update(per_method_scores)
        row.update(cell_errors)
        for method in METHODS:
            ms = method_stats_total.get(method, {})
            mp = method.lower()
            row[f"{mp}_n_instances"] = int(ms.get("n_instances", 0))
            row[f"{mp}_n_candidates_raw_total"] = int(ms.get("n_candidates_raw_total", 0))
            row[f"{mp}_n_candidates_flip_total"] = int(ms.get("n_candidates_flip_total", 0))
            row[f"{mp}_n_instances_with_flip_cf"] = int(ms.get("n_instances_with_flip_cf", 0))
            row[f"{mp}_n_instances_with_usable_cf"] = int(ms.get("n_instances_with_usable_cf", ms.get("n_instances_with_flip_cf", 0)))
            row[f"{mp}_n_instances_with_non_empty_cf"] = int(ms.get("n_instances_with_non_empty_cf", ms.get("n_instances_with_usable_cf", 0)))
            row[f"{mp}_n_instances_with_selected_usable_cf"] = int(
                ms.get("n_instances_with_selected_usable_cf", ms.get("n_instances_with_usable_cf", 0))
            )
            row[f"{mp}_coverage"] = float(ms.get("coverage", 0.0))
            row[f"coverage_{mp}"] = float(ms.get("coverage", 0.0))
            row[f"{mp}_n_empty_after_filter"] = int(ms.get("n_empty_after_filter", 0))

        row.update(ranking_stats)
        row["coverage_mode"] = cfg_coverage_mode
        row["coverage_penalty_lambda"] = float(args.coverage_penalty_lambda)
        row["hard_min_coverage"] = float(args.hard_min_coverage)

        for metric in METRICS:
            for method in METHODS:
                col = f"mean_{slugify(method)}_{slugify(metric)}"
                row[col] = float(mean_df.loc[metric, method])

        leaderboard_preview = pd.DataFrame()
        if os.path.exists(leaderboard_csv):
            try:
                leaderboard_preview = pd.read_csv(leaderboard_csv)
            except Exception:
                leaderboard_preview = pd.DataFrame()
        leaderboard_preview = drop_diag_columns_for_prod(leaderboard_preview)
        leaderboard_preview = pd.concat([leaderboard_preview, pd.DataFrame([row])], ignore_index=True, sort=False)
        leaderboard_preview = drop_diag_columns_for_prod(leaderboard_preview)
        config_snapshot_before = build_config_snapshot(
            comparison_df_config=comparison_df,
            row=row,
            prod_config_key=prod_config_key_text,
            leaderboard_preview=leaderboard_preview,
            prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
        )

        if diag_enabled:
            config_snapshot_after = build_config_snapshot(
                comparison_df_config=comparison_df,
                row=row,
                prod_config_key=prod_config_key_text,
                leaderboard_preview=leaderboard_preview,
                prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
            )
            validate_snapshot_unchanged(
                config_snapshot_before,
                config_snapshot_after,
                layer="config",
                prod_config_key=prod_config_key_text,
                fold_id=None,
                fold_file=None,
            )

        leaderboard_df = append_and_sort_leaderboard(
            leaderboard_csv,
            row,
            prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
        )
        write_best_json(best_json, leaderboard_df, run_stage=run_stage)

        existing.add(key)
        completed += 1

        print(
            "  "
            f"score_total={float(score_total):.6f}, "
            f"score_max={float(row.get('score_max', float('nan'))):.6f}, "
            f"score_mean={float(row.get('score_mean', float('nan'))):.6f}, "
            f"score_max_penalized={float(row.get('score_max_penalized', float('nan'))):.6f}, "
            f"min_coverage={float(row.get('min_coverage', float('nan'))):.6f}, "
            f"valid_config={bool(row.get('valid_config', False))}, "
            f"elapsed={time.time() - cfg_t0:.1f}s"
        )
        if not leaderboard_df.empty:
            best = leaderboard_df.iloc[0]
            print(
                "  current_best: "
                f"valid_config={parse_bool(best.get('valid_config', True), default=True)}, "
                f"score_max_penalized={float(best.get('score_max_penalized', best.get('score_max', best.get('score_total', float('nan'))))):.6f}, "
                f"score_mean={float(best.get('score_mean', best.get('score_total', float('nan')))):.6f}, "
                f"score_total={float(best.get('score_total', float('nan'))):.6f}, "
                f"min_coverage={float(best.get('min_coverage', float('nan'))):.6f}, "
                f"coverage_mode={str(best.get('coverage_mode', cfg_coverage_mode))}, "
                f"radius={int(best['radius'])}, "
                f"n_neighbors={int(best['n_neighbors'])}, "
                f"min_act={int(best['min_act'])}, "
                f"min_feas={int(best['min_feas'])}, "
                f"ufce_flip_filter={get_ufce_flip_filter_from_record(best, default=1)}"
            )

    total_runtime = time.time() - t0
    rebuild_comparison_from_leaderboard(
        leaderboard_csv=leaderboard_csv,
        comparison_csv=comparison_csv,
        best_comparison_csv=best_comparison_csv,
        run_stage=run_stage,
        eps=float(args.eps),
        prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
    )
    print_topk_comparison_tables(
        leaderboard_csv=leaderboard_csv,
        comparison_csv=comparison_csv,
        top_k=int(args.print_top_k),
        prioritize_ufce3=bool(int(args.rank_tiebreak_ufce3)),
    )

    print("\nDone.")
    print(f"Completed configs: {completed}")
    print(f"Skipped configs:   {skipped}")
    print(f"Total runtime:     {total_runtime:.1f}s")
    print(f"Leaderboard CSV:   {leaderboard_csv}")
    print(f"Best config JSON:  {best_json}")
    print(f"Compare (all):     {comparison_csv}")
    print(f"Compare (best):    {best_comparison_csv}")


if __name__ == "__main__":
    main()
