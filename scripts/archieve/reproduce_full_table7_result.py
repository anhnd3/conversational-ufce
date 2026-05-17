#!/usr/bin/env python
# coding: utf-8

import os
from pyexpat import features
import sys
import glob
import time
import re
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# ---- Correct PYTHONPATH bootstrap (your version) ----
ROOT = os.path.abspath(os.getcwd())

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------------

import ufce
from ufce import UFCE
from ufce.core import cfmethods
from ufce.core.cfmethods import sfexp, dfexp, tfexp, dice_cfexp, dice_cfexp_in, ar_cfexp
from ufce.core import evaluations as eval_module
from ufce.core.evaluations import Catproximity, Contproximity, Sparsity, Actionability, Plausibility, Feasibility
from ufce.core.data_processing import (
    classify_dataset_getModel,
    get_bank_user_constraints,
    get_grad_user_constraints,
    get_wine_user_constraints,
    get_bupa_user_constraints,
    get_movie_user_constraints,
)

ufc = UFCE()

METHODS = ["UFCE1", "UFCE2", "UFCE3", "DiCE", "DiCE-UF", "AR"]
MNAMES  = ["ufce1", "ufce2", "ufce3", "dice", "dice-uf", "ar"]
METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]

# ---------- Author Table 7 (your corrected values) ----------
AUTHOR_TABLE7: Dict[str, Dict[str, Dict[str, float]]] = {
    "bank": {
        "DiCE":    {"prox_jac": 0.70, "prox_euc": 121.45, "sparsity": 5.20, "actionability":  8.00, "plausibility": 39.00, "feasibility":  8.00},
        "DiCE-UF": {"prox_jac": 0.65, "prox_euc":  26.12, "sparsity": 3.20, "actionability": 34.00, "plausibility": 20.00, "feasibility": 20.00},
        "AR":      {"prox_jac": 0.62, "prox_euc": 129.56, "sparsity": 5.30, "actionability":  9.00, "plausibility": 45.00, "feasibility":  9.00},
        "UFCE1":   {"prox_jac": 0.60, "prox_euc":  10.00, "sparsity": 1.00, "actionability": 14.00, "plausibility": 14.00, "feasibility": 14.00},
        "UFCE2":   {"prox_jac": 0.00, "prox_euc":  23.10, "sparsity": 2.00, "actionability": 30.00, "plausibility": 30.00, "feasibility": 30.00},
        "UFCE3":   {"prox_jac": 0.00, "prox_euc":  40.12, "sparsity": 3.00, "actionability": 44.00, "plausibility": 43.00, "feasibility": 43.00},
    },
    "grad": {
        "DiCE":    {"prox_jac": 0.40, "prox_euc": 22.12, "sparsity": 4.40, "actionability":  3.00, "plausibility":  9.00, "feasibility":  3.00},
        "DiCE-UF": {"prox_jac": 0.30, "prox_euc":  6.32, "sparsity": 3.40, "actionability":  6.00, "plausibility": 10.00, "feasibility":  6.00},
        "AR":      {"prox_jac": 0.40, "prox_euc": 13.34, "sparsity": 4.15, "actionability":  5.00, "plausibility":  9.00, "feasibility":  5.00},
        "UFCE1":   {"prox_jac": 0.00, "prox_euc":  2.34, "sparsity": 1.00, "actionability":  8.00, "plausibility":  8.00, "feasibility":  8.00},
        "UFCE2":   {"prox_jac": 0.00, "prox_euc":  4.85, "sparsity": 2.00, "actionability": 13.00, "plausibility": 13.00, "feasibility": 13.00},
        "UFCE3":   {"prox_jac": 0.00, "prox_euc":  6.32, "sparsity": 2.80, "actionability": 13.00, "plausibility": 13.00, "feasibility": 13.00},
    },
    "wine": {
        "DiCE":    {"prox_jac": np.nan, "prox_euc": 43.10, "sparsity": 7.00, "actionability": 10.00, "plausibility": 33.00, "feasibility": 10.00},
        "DiCE-UF": {"prox_jac": np.nan, "prox_euc": 28.15, "sparsity": 3.00, "actionability": 50.00, "plausibility": 25.00, "feasibility": 25.00},
        "AR":      {"prox_jac": np.nan, "prox_euc": 38.25, "sparsity": 7.20, "actionability": 11.00, "plausibility": 42.00, "feasibility": 11.00},
        "UFCE1":   {"prox_jac": np.nan, "prox_euc": 14.90, "sparsity": 1.00, "actionability": 43.00, "plausibility": 28.00, "feasibility": 28.00},
        "UFCE2":   {"prox_jac": np.nan, "prox_euc":  8.45, "sparsity": 2.00, "actionability": 50.00, "plausibility": 41.00, "feasibility": 41.00},
        "UFCE3":   {"prox_jac": np.nan, "prox_euc": 21.95, "sparsity": 3.00, "actionability": 50.00, "plausibility": 42.00, "feasibility": 42.00},
    },
    "bupa": {
        "DiCE":    {"prox_jac": np.nan, "prox_euc": 51.45, "sparsity": 4.50, "actionability":  1.00, "plausibility":  1.00, "feasibility":  1.00},
        "DiCE-UF": {"prox_jac": np.nan, "prox_euc": 20.34, "sparsity": 2.90, "actionability":  5.00, "plausibility": 12.00, "feasibility":  5.00},
        "AR":      {"prox_jac": np.nan, "prox_euc": 40.65, "sparsity": 4.20, "actionability":  9.00, "plausibility": 15.00, "feasibility":  9.00},
        "UFCE1":   {"prox_jac": np.nan, "prox_euc": 10.00, "sparsity": 1.00, "actionability": 17.00, "plausibility": 15.00, "feasibility": 15.00},
        "UFCE2":   {"prox_jac": np.nan, "prox_euc":  9.00, "sparsity": 2.00, "actionability": 15.00, "plausibility": 15.00, "feasibility": 15.00},
        "UFCE3":   {"prox_jac": np.nan, "prox_euc": 17.10, "sparsity": 2.90, "actionability": 13.00, "plausibility": 13.00, "feasibility": 13.00},
    },
    "movie": {
        "DiCE":    {"prox_jac": 0.55, "prox_euc": 78.00, "sparsity": 10.00, "actionability":  5.00, "plausibility": 19.00, "feasibility":  5.00},
        "DiCE-UF": {"prox_jac": 0.00, "prox_euc": 56.00, "sparsity":  4.00, "actionability": 20.00, "plausibility": 17.00, "feasibility": 17.00},
        "AR":      {"prox_jac": 0.45, "prox_euc": 80.00, "sparsity":  9.00, "actionability":  8.00, "plausibility": 19.00, "feasibility":  8.00},
        "UFCE1":   {"prox_jac": 0.00, "prox_euc": 20.00, "sparsity":  1.00, "actionability": 20.00, "plausibility":  8.00, "feasibility":  8.00},
        "UFCE2":   {"prox_jac": 0.00, "prox_euc": 32.00, "sparsity":  2.00, "actionability": 14.00, "plausibility": 14.00, "feasibility": 14.00},
        "UFCE3":   {"prox_jac": 0.00, "prox_euc": 43.00, "sparsity":  3.00, "actionability": 18.00, "plausibility": 18.00, "feasibility": 18.00},
    },
}

TUNED_RUN2: Dict[str, Dict[str, int]] = {
    "bank": {"radius": 500, "n_neighbors": 1000, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "bupa": {"radius": 70, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "grad": {"radius": 500, "n_neighbors": 400, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "wine": {"radius": 7, "n_neighbors": 1000, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "movie": {"radius": 160, "n_neighbors": 100, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
}
ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]

def _values_equal(a, b) -> bool:
    try:
        return float(a) == float(b)
    except Exception:
        return str(a) == str(b)

def _vector_equal(values, target) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return np.array([_values_equal(v, target) for v in arr], dtype=bool)

def _pairwise_equal(values_a, values_b) -> np.ndarray:
    a = np.asarray(values_a).reshape(-1)
    b = np.asarray(values_b).reshape(-1)
    if len(a) != len(b):
        raise ValueError("Length mismatch while comparing prediction arrays.")
    return np.array([_values_equal(x, y) for x, y in zip(a, b)], dtype=bool)

def _normalize_col_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

def _resolve_column(df: pd.DataFrame, target: str) -> Optional[str]:
    if target in df.columns:
        return target
    tkey = _normalize_col_key(target)
    for col in df.columns:
        if _normalize_col_key(col) == tkey:
            return col
    return None

def _align_feature_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    resolved_cols: List[str] = []
    missing: List[str] = []
    for feat in features:
        col = _resolve_column(df, feat)
        if col is None:
            missing.append(feat)
        else:
            resolved_cols.append(col)
    if missing:
        raise KeyError(f"Missing model features in raw fold: {missing}")
    out = df[resolved_cols].copy()
    out.columns = features
    return out

def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)

def log_fold_pred_distribution_debug(
    *,
    dataset: str,
    fold_idx: int,
    total_folds: int,
    fold_path: str,
    selected_fold_df: pd.DataFrame,
    features: List[str],
    outcome_label: str,
    desired_outcome: float,
    lr,
    expected_pred1_rate: Optional[float],
) -> Dict[str, float]:
    fold_tag = f"{fold_idx + 1}/{total_folds}" if total_folds > 0 else str(fold_idx + 1)
    fold_file = os.path.basename(fold_path)
    n_test = int(len(selected_fold_df))
    print(
        f"[DBG][PRED-DIST] dataset={dataset} fold={fold_tag} file={fold_file} "
        f"source=fold_dir_selected_pred0 n_test={n_test}"
    )
    if n_test == 0:
        print("[DBG][PRED-DIST] status=selected_fold_empty")
        return {"pred1_rate": np.nan, "all_pred0": np.nan}

    try:
        selected_x = _align_feature_frame(selected_fold_df, features)
    except KeyError as exc:
        print(f"[DBG][PRED-DIST] status=selected_missing_feature_cols detail={exc}")
        return {"pred1_rate": np.nan, "all_pred0": np.nan}

    y_pred = np.asarray(lr.predict(selected_x)).reshape(-1)
    if len(y_pred) != n_test:
        print(
            f"[DBG][PRED-DIST] status=selected_prediction_length_mismatch "
            f"n_test={n_test} n_pred={len(y_pred)}"
        )
        return {"pred1_rate": np.nan, "all_pred0": np.nan}

    # Mirror author fold filtering rule from data_processing.predict_X_test_folds():
    # for Selector (bupa) pred0 label is 1; otherwise pred0 label is 0.
    pred0_label = 1 if outcome_label == "Selector" else 0
    y_pred_is_pred0 = _vector_equal(y_pred, pred0_label)
    n_pred0 = int(np.sum(y_pred_is_pred0))
    n_pred_not0 = int(n_test - n_pred0)
    non_pred0_rate = float(n_pred_not0 / n_test)
    all_pred0 = n_pred_not0 == 0

    # Keep desired-outcome binary view for optional perf/proba stats.
    y_pred_desired_bin = _vector_equal(y_pred, desired_outcome).astype(int)

    se = float(np.sqrt(non_pred0_rate * (1.0 - non_pred0_rate) / n_test))
    delta95 = float(1.96 * se)
    delta_ok = float(max(0.03, 2.0 * se))

    print(
        f"[DBG][PRED-DIST] selected_hard pred0_label={pred0_label} desired_outcome={desired_outcome} "
        f"n_pred0={n_pred0} pct_pred0={_pct(n_pred0, n_test):.2f}% "
        f"n_pred_not_pred0={n_pred_not0} pct_pred_not_pred0={_pct(n_pred_not0, n_test):.2f}% "
        f"all_pred0={all_pred0}"
    )
    print(
        f"[DBG][PRED-DIST] selected_delta not_pred0_rate={non_pred0_rate*100.0:.2f}% "
        f"SE={se*100.0:.2f}% delta95={delta95*100.0:.2f}% delta_ok={delta_ok*100.0:.2f}%"
    )

    target_not_pred0_rate = 0.0 if expected_pred1_rate is None else float(expected_pred1_rate)
    delta_vs_target = abs(non_pred0_rate - target_not_pred0_rate)
    status = "OK" if delta_vs_target <= delta_ok else "WARN"
    print(
        f"[DBG][PRED-DIST] selected_expected expected_not_pred0_rate={target_not_pred0_rate*100.0:.2f}% "
        f"delta={delta_vs_target*100.0:.2f}% status={status}"
    )

    if hasattr(lr, "predict_proba"):
        try:
            proba = np.asarray(lr.predict_proba(selected_x))
            if proba.ndim == 2 and proba.shape[0] == n_test and proba.shape[1] > 0:
                class_vals = list(getattr(lr, "classes_", []))
                if class_vals:
                    p1_idx = 0
                    for ci, c in enumerate(class_vals):
                        if _values_equal(c, desired_outcome):
                            p1_idx = ci
                            break
                else:
                    p1_idx = min(1, proba.shape[1] - 1)
                p1 = proba[:, p1_idx]
                q10, q50, q90 = np.quantile(p1, [0.10, 0.50, 0.90])
                print(
                    f"[DBG][PRED-DIST] selected_proba y1_label={desired_outcome} "
                    f"mean_p1={float(np.mean(p1)):.4f} p10={float(q10):.4f} "
                    f"p50={float(q50):.4f} p90={float(q90):.4f}"
                )
        except Exception as exc:
            print(f"[DBG][PRED-DIST] selected_proba status=unavailable detail={type(exc).__name__}")

    outcome_col = _resolve_column(selected_fold_df, outcome_label)
    if outcome_col is None:
        print("[DBG][PRED-DIST] selected_perf status=skipped_no_true_labels")
    else:
        y_true = selected_fold_df[outcome_col].to_numpy().reshape(-1)
        if len(y_true) != n_test:
            print(
                f"[DBG][PRED-DIST] selected_perf status=length_mismatch "
                f"n_true={len(y_true)} n_test={n_test}"
            )
        else:
            y_true_bin = _vector_equal(y_true, desired_outcome).astype(int)
            accuracy = float(np.mean(_pairwise_equal(y_true, y_pred)))
            tn = int(np.sum((y_true_bin == 0) & (y_pred_desired_bin == 0)))
            fp = int(np.sum((y_true_bin == 0) & (y_pred_desired_bin == 1)))
            fn = int(np.sum((y_true_bin == 1) & (y_pred_desired_bin == 0)))
            tp = int(np.sum((y_true_bin == 1) & (y_pred_desired_bin == 1)))
            n_true1 = int(np.sum(y_true_bin == 1))
            print(
                f"[DBG][PRED-DIST] selected_perf accuracy={accuracy:.4f} "
                f"confusion=(TN={tn},FP={fp},FN={fn},TP={tp}) "
                f"n_true1={n_true1} base_rate_y1={_pct(n_true1, n_test):.2f}%"
            )

    print(
        f"[DBG][PRED-DIST] post_filter strategy=loaded_fold_file "
        f"n_selected={n_test} pct_selected=100.00% note=input_is_already_filtered_pred0_fold"
    )

    return {
        "non_pred0_rate": non_pred0_rate,
        "all_pred0": float(all_pred0),
    }

def get_step_config(dataset: str) -> Dict[str, float]:
    # Same mapping you used in v3 (keep naming EXACTLY as dataset CSV columns in UFCE repo)
    if dataset == "bank":
        return {
            "Income": 1, "Family": 1, "CCAvg": 0.1, "Education": 1, "Mortgage": 1,
            "SecuritiesAccount": 1, "CDAccount": 1, "Online": 1, "CreditCard": 1
        }
    if dataset == "grad":
        return {"GRE Score": 1, "TOEFL Score": 1, "University Rating": 1, "SOP": 1, "LOR": 1, "CGPA": 0.1, "Research": 1}
    if dataset == "wine":
        return {
            "fixed acidity": 0.5, "volatile acidity": 0.10, "citric acid": 0.1,
            "residual sugar": 0.5, "free sulfur dioxide": 1.0, "total sulfur dioxide": 1.0,
            "density": 0.1, "pH": 0.5, "alcohol": 0.5
        }
    if dataset == "bupa":
        return {"Mcv": 1, "Alkphos": 1, "Sgpt": 1, "Sgot": 1, "Gammagt": 1, "Drinks": 1}
    if dataset == "movie":
        return {
            "Production_expense": 3, "Num_multiplex": 3, "Multiplex_coverage": 0.2, "Movie_length": 5,
            "Lead_Actor_Rating": 1.0, "Lead_Actress_rating": 1.0,
            "Director_rating": 1.0, "Producer_rating": 1.0, "Genre": 1,
            "Collection": 500, "Budget": 3000
        }
    return {}

def get_constraints(dataset: str, datasetdf: pd.DataFrame):
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
    raise ValueError(f"Unknown dataset: {dataset}")

def init_ufce_global(
    *,
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

def parse_trace_positions_arg(trace_positions_raw: str, debug_flag: int) -> List[int]:
    trace_positions: List[int] = []
    text = str(trace_positions_raw).strip()
    if text:
        for token in text.split(","):
            cleaned = token.strip()
            if cleaned == "":
                continue
            try:
                trace_positions.append(int(cleaned))
            except Exception as exc:
                raise ValueError(f"Invalid --trace_positions token '{cleaned}'. Use comma-separated integers.") from exc
    if len(trace_positions) > 0 and int(debug_flag) != 1:
        raise ValueError("Set --debug 1 or remove --trace_positions.")
    return trace_positions

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

def apply_distance_scaler(df: pd.DataFrame, scaler: Optional[Dict]) -> pd.DataFrame:
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

def log_movie_scaler_sanity(
    *,
    datasetdf: pd.DataFrame,
    features: List[str],
    scaler: Optional[Dict],
    debug_enabled: bool,
) -> None:
    if not bool(debug_enabled) or scaler is None:
        return
    scale_cols = list(scaler.get("scale_cols", []))
    print(f"[DBG][MOVIE-SCALER] scale_cols={scale_cols}")
    if len(scale_cols) == 0:
        return
    raw = datasetdf.loc[:, features].copy()
    scaled = apply_distance_scaler(raw, scaler)
    for col in scale_cols[:2]:
        if col in raw.columns and col in scaled.columns:
            print(
                "[DBG][MOVIE-SCALER] "
                f"col={col} raw[min,max]=({float(raw[col].min()):.6f},{float(raw[col].max()):.6f}) "
                f"scaled[min,max]=({float(scaled[col].min()):.6f},{float(scaled[col].max()):.6f})"
            )

def movie_distance_schema_status(
    *,
    frame: object,
    features: List[str],
    numf: List[str],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if frame is None:
        return False, ["is_none"]
    if not isinstance(frame, pd.DataFrame):
        return False, [f"type={type(frame).__name__}"]
    missing_features = [c for c in features if c not in frame.columns]
    if len(missing_features) > 0:
        reasons.append(f"missing_features={missing_features[:5]}")
    numf_present = [c for c in numf if c in frame.columns]
    if not (set(numf).issubset(set(frame.columns)) or len(numf_present) > 0):
        reasons.append("numeric_scaling_not_ready")
    try:
        _ = frame.loc[:, features]
    except Exception as exc:
        reasons.append(f"projection_error={type(exc).__name__}")
    return len(reasons) == 0, reasons

def mad_fit(df: pd.DataFrame, numf: List[str]) -> Dict[str, pd.Series]:
    med = df[numf].median()
    mad = (df[numf] - med).abs().median()
    mad = mad.replace(0, 1.0)
    return {"medians": med, "mads": mad}

def mad_transform(df: pd.DataFrame, scaler: Dict[str, pd.Series], numf: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[numf] = (out[numf] - scaler["medians"][numf]) / scaler["mads"][numf]
    return out

def mad_inverse_inplace(df: pd.DataFrame, scaler: Dict[str, pd.Series], numf: List[str]) -> None:
    df[numf] = (df[numf] * scaler["mads"][numf]) + scaler["medians"][numf]

def drop_dice_cols(cfdf: pd.DataFrame, outcome_label: str) -> pd.DataFrame:
    if cfdf is None or cfdf.empty:
        return cfdf
    drop_cols = []
    for c in [outcome_label, "proximity"]:
        if c in cfdf.columns:
            drop_cols.append(c)
    if drop_cols:
        cfdf = cfdf.drop(drop_cols, axis=1)
    return cfdf

def fold_mean_std_table(metric_name: str, mmeans: List[float], mstds: List[float]) -> None:
    # helpful compact printing
    print(f"[FOLD] {metric_name} means: " + ", ".join([f"{MNAMES[i]}={mmeans[i]:.2f}" for i in range(len(MNAMES))]))
    print(f"[FOLD] {metric_name} stds : " + ", ".join([f"{MNAMES[i]}={mstds[i]:.2f}" for i in range(len(MNAMES))]))

def run_one_fold(
    *,
    dataset: str,
    fold_idx: int,
    fold_path: str,
    datasetdf: pd.DataFrame,
    features: List[str],
    catf: List[str],
    numf: List[str],
    uf: Dict,
    f2change: List[str],
    protectf: List[str],
    outcome_label: str,
    desired_outcome: float,
    step: Dict[str, float],
    MI_FP: List[List[str]],
    lr,
    X: pd.DataFrame,
    Xtest: pd.DataFrame,
    Xtrain: pd.DataFrame,
    data_lab1: pd.DataFrame,
    no_cf: int,
    scaler_ar,
    ufce_mad_scaler: Optional[Dict[str, pd.Series]],  # <-- FIX: only non-None when provided by author pipeline (movie)
    movie_distance_scaler: Optional[Dict[str, object]],
    flip_filter_enabled: bool,
    debug_enabled: bool,
    trace_positions: List[int],
    total_folds: int,
    expected_pred1_rate: Optional[float],
):
    """
    Runs one fold:
    - prepares UFCE search space (MAD-scaled ONLY when ufce_mad_scaler is provided)
    - generates CFs for 6 methods (UFCE1/2/3, DiCE, DiCE-UF, AR)
    - evaluates using evaluations.py
    Returns:
      fold_metrics: dict(metric -> list[6] means)
      fold_stds:    dict(metric -> list[6] stds)
      times:        dict(method -> float seconds)
    """
    print(f"\n[INFO] Fold {fold_idx}: {os.path.basename(fold_path)}")
    testset = pd.read_csv(fold_path)
    if bool(debug_enabled):
        pred_dist_debug = log_fold_pred_distribution_debug(
            dataset=dataset,
            fold_idx=fold_idx,
            total_folds=total_folds,
            fold_path=fold_path,
            selected_fold_df=testset,
            features=features,
            outcome_label=outcome_label,
            desired_outcome=desired_outcome,
            lr=lr,
            expected_pred1_rate=expected_pred1_rate,
        )
    else:
        pred_dist_debug = {"non_pred0_rate": np.nan, "all_pred0": np.nan}

    # -------------------------------
    # A) UFCE inputs (FIXED scaler)
    # -------------------------------
    # IMPORTANT:
    # - author pipeline provides MAD scaler only for movie; for other datasets it is None
    # - DO NOT compute MAD scaler for grad/bank/wine/bupa (that breaks UFCE intervals/step semantics)
    if dataset == "movie" and movie_distance_scaler is not None:
        # Movie: keep UF constraints/model actions in raw space and use distance-space only
        # for neighbor search / Euclidean operations.
        testset_ufce = testset[features].copy()
        data_lab1_ufce = data_lab1[features].copy()
        testset_dist = apply_distance_scaler(testset_ufce, movie_distance_scaler)
        data_lab1_dist = apply_distance_scaler(data_lab1_ufce, movie_distance_scaler)
    elif ufce_mad_scaler is not None:
        # Preserve existing non-movie MAD behavior.
        testset_ufce = mad_transform(testset[features], ufce_mad_scaler, numf)
        data_lab1_ufce = mad_transform(data_lab1[features], ufce_mad_scaler, numf)
        testset_dist = None
        data_lab1_dist = None
    else:
        # raw UFCE search space
        testset_ufce = testset[features].copy()
        data_lab1_ufce = data_lab1[features].copy()
        testset_dist = None
        data_lab1_dist = None

    debug_ctx = None
    if bool(debug_enabled):
        trace_positions_set = {int(v) for v in trace_positions}
        debug_ctx = {
            "enabled": True,
            "trace_positions": sorted(trace_positions_set),
            "trace_positions_set": trace_positions_set,
        }

    distance_scaler = movie_distance_scaler if (dataset == "movie" and movie_distance_scaler is not None) else None

    # -------------------------------
    # B) Generate CFs (author cfmethods.py)
    # -------------------------------
    onecfs, t_ufce1, idx1 = sfexp(
        X, data_lab1_ufce, testset_ufce[:],
        uf, step, f2change, numf, catf, lr, desired_outcome, no_cf, features,
        flip_filter_enabled=bool(flip_filter_enabled),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=testset_dist,
        distance_scaler=distance_scaler,
        debug_ctx=debug_ctx,
    )
    twocfs, t_ufce2, idx2 = dfexp(
        X, data_lab1_ufce, testset_ufce[:],
        uf, MI_FP[:5], numf, catf, f2change, protectf, lr, desired_outcome, no_cf, features,
        flip_filter_enabled=bool(flip_filter_enabled),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=testset_dist,
        distance_scaler=distance_scaler,
    )
    threecfs, t_ufce3, idx3 = tfexp(
        X, data_lab1_ufce, testset_ufce[:],
        uf, MI_FP[:5], numf, catf, f2change, protectf, lr, desired_outcome, no_cf, features,
        flip_filter_enabled=bool(flip_filter_enabled),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=testset_dist,
        distance_scaler=distance_scaler,
    )

    # Inverse-transform UFCE CFs back to RAW for evaluation
    if dataset != "movie" and ufce_mad_scaler is not None:
        for cf_df in (onecfs, twocfs, threecfs):
            if cf_df is not None and not cf_df.empty:
                mad_inverse_inplace(cf_df, ufce_mad_scaler, numf)

    onetestdata = testset.loc[idx1].reset_index(drop=True)
    twotestdata = testset.loc[idx2].reset_index(drop=True)
    threetestdata = testset.loc[idx3].reset_index(drop=True)

    # ---------------- DiCE ----------------
    dicecfs, idx_dice, t_dice, _flag = dice_cfexp(
        datasetdf, testset[:], numf, f2change, no_cf, lr, uf, outcome_label
    )

    no_dice = (
        dicecfs is None
        or (hasattr(dicecfs, "empty") and dicecfs.empty)
        or idx_dice is None
        or (hasattr(idx_dice, "__len__") and len(idx_dice) == 0)
    )

    if no_dice:
        dicecfs = pd.DataFrame(columns=features)
        dicetestdata = pd.DataFrame(columns=features)
    else:
        dicecfs = drop_dice_cols(dicecfs, outcome_label).reset_index(drop=True)
        dicetestdata = testset.loc[idx_dice].reset_index(drop=True)

    # ---------------- DiCE-UF ----------------
    dicecfs_in, idx_in, t_diceuf, _flag2 = dice_cfexp_in(
        datasetdf, testset[:], numf, f2change, no_cf, lr, uf, outcome_label
    )

    no_diceuf = (
        dicecfs_in is None
        or (hasattr(dicecfs_in, "empty") and dicecfs_in.empty)
        or idx_in is None
        or (hasattr(idx_in, "__len__") and len(idx_in) == 0)
    )

    if no_diceuf:
        dicecfs_in = pd.DataFrame(columns=features)
        dicetestdata_in = pd.DataFrame(columns=features)
    else:
        dicecfs_in = drop_dice_cols(dicecfs_in, outcome_label).reset_index(drop=True)
        dicetestdata_in = testset.loc[idx_in].reset_index(drop=True)


    # AR in RAW space + scaler (author)
    arcfs, t_ar, idx_ar = ar_cfexp(X, numf, lr, testset[:], uf, scaler_ar, Xtrain, f2change)
    artestdata = testset.loc[idx_ar].reset_index(drop=True)
    if arcfs is not None and not arcfs.empty:
        arcfs = arcfs.reset_index(drop=True)

    print(
        f"[TIME] UFCE1={t_ufce1:.6f} | UFCE2={t_ufce2:.6f} | UFCE3={t_ufce3:.6f} | "
        f"DiCE={t_dice:.6f} | DiCE-UF={t_diceuf:.6f} | AR={t_ar:.6f}"
    )

    # -------------------------------
    # C) Evaluate using author evaluations.py
    # -------------------------------
    def _eval_stage_start(stage_name: str) -> float:
        t_stage = time.time()
        print(f"[EVAL][Fold {fold_idx}] {stage_name} ...")
        return t_stage

    def _eval_stage_done(stage_name: str, t_stage: float) -> None:
        print(f"[EVAL][Fold {fold_idx}] {stage_name} done in {time.time() - t_stage:.2f}s")

    t_stage = _eval_stage_start("Catproximity")
    cat_means, cat_stds = Catproximity(
        onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
        dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, catf
    )
    _eval_stage_done("Catproximity", t_stage)
    # Prox-Jac is undefined when there are no categorical features.
    # Use NaN (not 0) so outputs align with author-style reporting (e.g., wine/bupa).
    if len(catf) == 0:
        cat_means = [np.nan] * len(METHODS)
        cat_stds = [np.nan] * len(METHODS)

    raw_cont_frames = {
        "onecfs": onecfs,
        "onetestdata": onetestdata,
        "twocfs": twocfs,
        "twotestdata": twotestdata,
        "threecfs": threecfs,
        "threetestdata": threetestdata,
        "dicecfs": dicecfs,
        "dicecfs_in": dicecfs_in,
        "dicetestdata_in": dicetestdata_in,
        "arcfs": arcfs,
        "Xtest": Xtest,
    }
    t_stage = _eval_stage_start("Contproximity")
    if dataset == "movie" and movie_distance_scaler is not None:
        all_safe = True
        projected_frames: Dict[str, pd.DataFrame] = {}
        unsafe_msgs: List[str] = []
        for frame_name in [
            "onecfs",
            "onetestdata",
            "twocfs",
            "twotestdata",
            "threecfs",
            "threetestdata",
            "dicecfs",
            "dicecfs_in",
            "dicetestdata_in",
            "arcfs",
            "Xtest",
        ]:
            frame = raw_cont_frames[frame_name]
            safe, reasons = movie_distance_schema_status(
                frame=frame,
                features=features,
                numf=numf,
            )
            if not safe:
                all_safe = False
                unsafe_msgs.append(
                    f"{frame_name}(type={type(frame).__name__}, reasons={';'.join(reasons)})"
                )
                continue
            projected_frames[frame_name] = frame.loc[:, features].copy()

        if all_safe:
            scaled_frames = {
                name: apply_distance_scaler(df, movie_distance_scaler)
                for name, df in projected_frames.items()
            }
            cont_means, cont_stds = Contproximity(
                scaled_frames["onecfs"],
                scaled_frames["onetestdata"],
                scaled_frames["twocfs"],
                scaled_frames["twotestdata"],
                scaled_frames["threecfs"],
                scaled_frames["threetestdata"],
                scaled_frames["dicecfs"],
                scaled_frames["dicecfs_in"],
                scaled_frames["dicetestdata_in"],
                scaled_frames["arcfs"],
                scaled_frames["Xtest"],
                numf,
            )
        else:
            if bool(debug_enabled):
                print(
                    "[DBG][MOVIE-PROX-EUC] Falling back to raw Contproximity due to unsafe frame schema: "
                    + ", ".join(unsafe_msgs)
                )
            cont_means, cont_stds = Contproximity(
                onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
                dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, numf
            )
    else:
        cont_means, cont_stds = Contproximity(
            onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
            dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, numf
        )
    _eval_stage_done("Contproximity", t_stage)

    t_stage = _eval_stage_start("Sparsity")
    spar_means, spar_stds = Sparsity(
        onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
        dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, numf
    )
    _eval_stage_done("Sparsity", t_stage)

    t_stage = _eval_stage_start("Actionability")
    act_means, act_stds = Actionability(
        onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
        dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, features, f2change, uf
    )
    _eval_stage_done("Actionability", t_stage)

    t_stage = _eval_stage_start("Plausibility")
    plaus_means, plaus_stds = Plausibility(
        onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
        dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, Xtrain
    )
    _eval_stage_done("Plausibility", t_stage)

    t_stage = _eval_stage_start("Feasibility")
    feas_means, feas_stds = Feasibility(
        onecfs, onetestdata, twocfs, twotestdata, threecfs, threetestdata,
        dicecfs, dicecfs_in, dicetestdata_in, arcfs, Xtest, Xtrain,
        features, f2change, lr, desired_outcome, uf
    )
    _eval_stage_done("Feasibility", t_stage)

    fold_metrics = {
        "Prox-Jac": cat_means,
        "Prox-Euc": cont_means,
        "Sparsity": spar_means,
        "Actionability": act_means,
        "Plausibility": plaus_means,
        "Feasibility": feas_means,
    }
    fold_stds = {
        "Prox-Jac": cat_stds,
        "Prox-Euc": cont_stds,
        "Sparsity": spar_stds,
        "Actionability": act_stds,
        "Plausibility": plaus_stds,
        "Feasibility": feas_stds,
    }
    times = {
        "UFCE1": t_ufce1, "UFCE2": t_ufce2, "UFCE3": t_ufce3,
        "DiCE": t_dice, "DiCE-UF": t_diceuf, "AR": t_ar
    }

    # Optional compact per-fold print
    fold_mean_std_table("Prox-Jac", cat_means, cat_stds)
    fold_mean_std_table("Prox-Euc", cont_means, cont_stds)
    fold_mean_std_table("Sparsity", spar_means, spar_stds)
    fold_mean_std_table("Actionability", act_means, act_stds)
    fold_mean_std_table("Plausibility", plaus_means, plaus_stds)
    fold_mean_std_table("Feasibility", feas_means, feas_stds)

    return fold_metrics, fold_stds, times, pred_dist_debug


def run_for_dataset(
    dataset: str,
    args,
    *,
    debug_enabled: bool,
    trace_positions: List[int],
    expected_pred1_rate: Optional[float],
) -> Dict[str, object]:
    t0 = time.time()
    allowed_tuned = sorted(TUNED_RUN2.keys())
    if dataset not in TUNED_RUN2:
        raise ValueError(f"Dataset '{dataset}' missing from TUNED_RUN2. Allowed: {allowed_tuned}")
    cfg = TUNED_RUN2[dataset]
    args.radius = int(cfg["radius"])
    args.n_neighbors = int(cfg["n_neighbors"])
    flip_filter_enabled = bool(int(cfg["ufce_flip_filter"]))
    print(
        "[CFG] "
        f"dataset={dataset} tuned radius={args.radius} n_neighbors={args.n_neighbors} "
        f"min_act={int(cfg['min_act'])} min_feas={int(cfg['min_feas'])} flip={int(cfg['ufce_flip_filter'])}"
    )

    datafile = os.path.join(args.data_dir, f"{dataset}.csv")
    datasetdf = pd.read_csv(datafile)

    out = classify_dataset_getModel(datasetdf, data_name=dataset)

    # author pipeline may return scaler for movie only
    if len(out) == 8:
        lr, lr_mean, lr_std, Xtest, Xtrain, X, Y, df = out
        scaler_mad_from_func = None
    else:
        lr, lr_mean, lr_std, Xtest, Xtrain, X, Y, df, scaler_mad_from_func = out

    print(f"[INFO] Dataset={dataset}")
    print(f"[INFO] LR CV acc: {lr_mean:.2f} +/- {lr_std:.2f}")

    (features, catf, numf, uf, f2change, outcome_label,
     desired_outcome, nbr_features, protectf, data_lab0, data_lab1) = get_constraints(dataset, datasetdf)

    step = get_step_config(dataset)
    MI_FP = ufc.get_top_MI_features(X, features)
    print(f"[INFO] Top-5 MI feature pairs: {MI_FP[:5]}")
    print(f"[INFO] outcome_label={outcome_label} | desired_outcome={desired_outcome}")
    print(f"[INFO] UFCE radius={args.radius} | n_neighbors={args.n_neighbors} | contprox_metric={args.contprox_metric} | no_cf={args.no_cf}")
    # This script consumes pre-existing fold CSV files; pred0_n_folds is not used here.

    # AR scaler (author style)
    scaler_ar = StandardScaler().fit(Xtrain[:])

    # -------------------------------
    # FIX: UFCE MAD scaling policy
    # -------------------------------
    # Use MAD scaler ONLY if returned by classify_dataset_getModel (author pipeline => movie).
    ufce_mad_scaler = scaler_mad_from_func
    if ufce_mad_scaler is None:
        print("[INFO] UFCE MAD scaling: OFF (author pipeline provides no scaler for this dataset)")
    else:
        print("[INFO] UFCE MAD scaling: ON (using scaler from classify_dataset_getModel())")

    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )
        log_movie_scaler_sanity(
            datasetdf=datasetdf,
            features=features,
            scaler=movie_distance_scaler,
            debug_enabled=debug_enabled,
        )

    # folds directory
    fold_dir = os.path.join(args.fold_dir, dataset, "totest")
    fold_files = sorted(glob.glob(os.path.join(fold_dir, "*.csv")))
    if not fold_files:
        raise FileNotFoundError(f"No folds found in {fold_dir}")

    init_ufce_global(
        radius=args.radius,
        n_neighbors=args.n_neighbors,
        contprox_metric=args.contprox_metric,
        min_act=int(cfg["min_act"]),
        min_feas=int(cfg["min_feas"]),
        atol=1e-5,
    )

    fold_records = []
    pred_dist_records = []
    fold_iter = fold_files
    if tqdm is not None:
        fold_iter = tqdm(fold_files, desc=f"{dataset} folds", unit="fold", ascii=True)
    for fi, fp in enumerate(fold_iter):
        fold_metrics, fold_stds, fold_times, pred_dist_debug = run_one_fold(
            dataset=dataset,
            fold_idx=fi,
            fold_path=fp,
            datasetdf=datasetdf,
            features=features,
            catf=catf,
            numf=numf,
            uf=uf,
            f2change=f2change,
            protectf=protectf,
            outcome_label=outcome_label,
            desired_outcome=desired_outcome,
            step=step,
            MI_FP=MI_FP,
            lr=lr,
            X=X,
            Xtest=Xtest,
            Xtrain=Xtrain,
            data_lab1=data_lab1,
            no_cf=args.no_cf,
            scaler_ar=scaler_ar,
            ufce_mad_scaler=ufce_mad_scaler,  # <-- FIX applied here
            movie_distance_scaler=movie_distance_scaler,
            flip_filter_enabled=flip_filter_enabled,
            debug_enabled=debug_enabled,
            trace_positions=trace_positions,
            total_folds=len(fold_files),
            expected_pred1_rate=expected_pred1_rate,
        )

        fold_records.append(fold_metrics)
        pred_dist_records.append(pred_dist_debug)

    # ---- Aggregate across folds ----
    mean_table = pd.DataFrame(index=METRICS, columns=MNAMES, dtype=float)
    std_table  = pd.DataFrame(index=METRICS, columns=MNAMES, dtype=float)

    for metric in METRICS:
        mat = np.array([fr[metric] for fr in fold_records], dtype=float)  # folds x methods
        mean_table.loc[metric, :] = np.nanmean(mat, axis=0)
        std_table.loc[metric, :]  = np.nanstd(mat, axis=0, ddof=1)

    print("\n==================== OUR RESULTS (mean) ====================")
    print(mean_table.to_string(float_format=lambda x: f"{x:.2f}" if np.isfinite(x) else "nan"))

    print("\n==================== OUR RESULTS (std) =====================")
    print(std_table.to_string(float_format=lambda x: f"{x:.2f}" if np.isfinite(x) else "nan"))

    # ---- Delta vs author (ours - author) ----
    metric_map = {
        "Prox-Jac": "prox_jac",
        "Prox-Euc": "prox_euc",
        "Sparsity": "sparsity",
        "Actionability": "actionability",
        "Plausibility": "plausibility",
        "Feasibility": "feasibility",
    }

    print("\n==================== DELTA vs AUTHOR (ours - author) ====================")
    rows = []
    for mi, method in enumerate(METHODS):
        col = MNAMES[mi]
        for metric in METRICS:
            ours_v = float(mean_table.loc[metric, col])
            author_v = AUTHOR_TABLE7[dataset].get(method, {}).get(metric_map[metric], np.nan)
            delta = ours_v - author_v if (np.isfinite(ours_v) and np.isfinite(author_v)) else np.nan
            rows.append([method, metric, author_v, ours_v, delta])

    df_delta = pd.DataFrame(rows, columns=["method", "metric", "author", "ours_mean", "delta"])
    print(df_delta.to_string(index=False, float_format=lambda x: f"{x:.2f}" if np.isfinite(x) else "nan"))

    valid_non_pred0 = np.array([r.get("non_pred0_rate", np.nan) for r in pred_dist_records], dtype=float)
    valid_non_pred0 = valid_non_pred0[np.isfinite(valid_non_pred0)]
    valid_all_pred0 = np.array([r.get("all_pred0", np.nan) for r in pred_dist_records], dtype=float)
    valid_all_pred0 = valid_all_pred0[np.isfinite(valid_all_pred0)]

    if bool(debug_enabled) and len(valid_non_pred0) > 0:
        non_pred0_mean = float(np.mean(valid_non_pred0))
        non_pred0_std = float(np.std(valid_non_pred0, ddof=1)) if len(valid_non_pred0) > 1 else 0.0
        non_pred0_min = float(np.min(valid_non_pred0))
        non_pred0_max = float(np.max(valid_non_pred0))
        print(
            "\n[DBG][PRED-DIST][SUMMARY] "
            f"dataset={dataset} not_pred0_rate_mean={non_pred0_mean*100.0:.2f}% "
            f"not_pred0_rate_std={non_pred0_std*100.0:.2f}% not_pred0_rate_min={non_pred0_min*100.0:.2f}% "
            f"not_pred0_rate_max={non_pred0_max*100.0:.2f}% folds={len(valid_non_pred0)}"
        )
    if bool(debug_enabled) and len(valid_all_pred0) > 0:
        all_pred0_folds = int(np.sum(valid_all_pred0))
        print(
            "[DBG][PRED-DIST][SUMMARY] "
            f"dataset={dataset} all_pred0_folds={all_pred0_folds}/{len(valid_all_pred0)}"
        )
        if len(valid_non_pred0) > 0:
            print(
                "[DBG][PRED-DIST][SUMMARY] "
                f"loaded pred0 folds predict non-pred0 at {non_pred0_mean*100.0:.2f}% +/- {non_pred0_std*100.0:.2f}% "
                f"across {len(valid_non_pred0)} folds; rows are all predicted as pred0 label in "
                f"{all_pred0_folds}/{len(valid_all_pred0)} folds."
            )

    runtime_sec = float(time.time() - t0)
    print(f"\n[DONE] Total runtime: {runtime_sec:.2f}s")
    return {
        "dataset": dataset,
        "status": "ok",
        "runtime_sec": runtime_sec,
        "error": "",
        "n_folds": int(len(fold_files)),
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bank", "grad", "wine", "bupa", "movie", "all"])
    ap.add_argument("--no_cf", type=int, default=50)
    ap.add_argument("--radius", type=int, default=500)
    ap.add_argument("--n_neighbors", type=int, default=1000)
    ap.add_argument("--contprox_metric", type=str, default="euclidean")
    ap.add_argument("--data_dir", type=str, default=os.path.join("ufce", "data"), help="Repo-relative path to data folder")
    ap.add_argument("--fold_dir", type=str, default=os.path.join("ufce", "data", "folds"), help="Repo-relative path to folds folder")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--trace_positions", type=str, default="")
    ap.add_argument(
        "--expected_pred1_rate",
        type=float,
        default=None,
        help="Optional expected class-1 prediction rate for debug checks. Use 0-1 (or 0-100, auto-normalized).",
    )
    args = ap.parse_args()
    if int(args.debug) not in (0, 1):
        raise ValueError("--debug must be 0 or 1.")
    trace_positions = parse_trace_positions_arg(args.trace_positions, int(args.debug))
    debug_enabled = bool(int(args.debug) == 1)

    expected_pred1_rate = args.expected_pred1_rate
    if expected_pred1_rate is not None and expected_pred1_rate > 1.0:
        expected_pred1_rate = expected_pred1_rate / 100.0
    if expected_pred1_rate is not None:
        expected_pred1_rate = float(expected_pred1_rate)

    if args.dataset != "all":
        run_for_dataset(
            args.dataset,
            args,
            debug_enabled=debug_enabled,
            trace_positions=trace_positions,
            expected_pred1_rate=expected_pred1_rate,
        )
        return

    batch_t0 = time.time()
    batch_records: List[Dict[str, object]] = []
    for dataset in ALL_DATASETS:
        ds_t0 = time.time()
        print(f"\n==================== BATCH DATASET: {dataset} ====================")
        try:
            rec = run_for_dataset(
                dataset,
                args,
                debug_enabled=debug_enabled,
                trace_positions=trace_positions,
                expected_pred1_rate=expected_pred1_rate,
            )
        except Exception as exc:
            rec = {
                "dataset": dataset,
                "status": "failed",
                "runtime_sec": float(time.time() - ds_t0),
                "error": f"{type(exc).__name__}: {exc}",
                "n_folds": 0,
            }
            print(f"[ERROR] dataset={dataset} failed with {type(exc).__name__}: {exc}")
        batch_records.append(rec)

    summary_df = pd.DataFrame(batch_records, columns=["dataset", "status", "runtime_sec", "n_folds", "error"])
    print("\n==================== BATCH SUMMARY ====================")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{float(x):.2f}" if np.isfinite(float(x)) else "nan"))
    ok_count = int(np.sum(summary_df["status"].values == "ok"))
    failed_count = int(np.sum(summary_df["status"].values == "failed"))
    total_runtime = float(time.time() - batch_t0)
    print(
        "[DONE][BATCH] "
        f"ok={ok_count} failed={failed_count} total_runtime={total_runtime:.2f}s"
    )
    if failed_count > 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
