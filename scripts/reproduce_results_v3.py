#!/usr/bin/env python3
"""
UFCE Core Module P3 — UFCE-only optimized reproduction runner.

- Optimizes only UFCE1/UFCE2/UFCE3.
- Hard-disables DiCE, DiCE-UF, and AR paths.
- Uses dataset-tuned UFCE defaults with CLI override support.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---- PYTHONPATH bootstrap (Python-native) ----
ROOT = os.path.abspath(os.getcwd())

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------

warnings.filterwarnings("ignore")
if plt is not None:
    plt.style.use("seaborn-whitegrid")
pd.set_option("display.max_columns", None)

import ufce
from ufce import UFCE
from ufce.core import cfmethods
from ufce.core import evaluations as eval_module
from ufce.core.data_processing import (
    classify_dataset_getModel,
    get_bank_user_constraints,
    get_grad_user_constraints,
    get_wine_user_constraints,
    get_bupa_user_constraints,
    get_movie_user_constraints,
)

ufc = UFCE()

# ----------------------------
# Author reference values (Table 7) — UFCE only
# ----------------------------
AUTHOR_TABLE7: Dict[str, Dict[str, Dict[str, float]]] = {
    "bank": {
        "UFCE1": {
            "Prox-Jac": 0.60,
            "Prox-Euc": 10.00,
            "Sparsity": 1.00,
            "Actionability": 14.00,
            "Plausibility": 14.00,
            "Feasibility": 14.00,
        },
        "UFCE2": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 23.10,
            "Sparsity": 2.00,
            "Actionability": 30.00,
            "Plausibility": 30.00,
            "Feasibility": 30.00,
        },
        "UFCE3": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 40.12,
            "Sparsity": 3.00,
            "Actionability": 44.00,
            "Plausibility": 43.00,
            "Feasibility": 43.00,
        },
    },
    "grad": {
        "UFCE1": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 2.34,
            "Sparsity": 1.00,
            "Actionability": 8.00,
            "Plausibility": 8.00,
            "Feasibility": 8.00,
        },
        "UFCE2": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 4.85,
            "Sparsity": 2.00,
            "Actionability": 13.00,
            "Plausibility": 13.00,
            "Feasibility": 13.00,
        },
        "UFCE3": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 6.32,
            "Sparsity": 2.80,
            "Actionability": 13.00,
            "Plausibility": 13.00,
            "Feasibility": 13.00,
        },
    },
    "wine": {
        "UFCE1": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 14.90,
            "Sparsity": 1.00,
            "Actionability": 43.00,
            "Plausibility": 28.00,
            "Feasibility": 28.00,
        },
        "UFCE2": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 8.45,
            "Sparsity": 2.00,
            "Actionability": 50.00,
            "Plausibility": 41.00,
            "Feasibility": 41.00,
        },
        "UFCE3": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 21.95,
            "Sparsity": 3.00,
            "Actionability": 50.00,
            "Plausibility": 42.00,
            "Feasibility": 42.00,
        },
    },
    "bupa": {
        "UFCE1": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 10.00,
            "Sparsity": 1.00,
            "Actionability": 17.00,
            "Plausibility": 15.00,
            "Feasibility": 15.00,
        },
        "UFCE2": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 9.00,
            "Sparsity": 2.00,
            "Actionability": 15.00,
            "Plausibility": 15.00,
            "Feasibility": 15.00,
        },
        "UFCE3": {
            "Prox-Jac": float("nan"),
            "Prox-Euc": 17.10,
            "Sparsity": 2.90,
            "Actionability": 13.00,
            "Plausibility": 13.00,
            "Feasibility": 13.00,
        },
    },
    "movie": {
        "UFCE1": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 20.00,
            "Sparsity": 1.00,
            "Actionability": 20.00,
            "Plausibility": 8.00,
            "Feasibility": 8.00,
        },
        "UFCE2": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 32.00,
            "Sparsity": 2.00,
            "Actionability": 14.00,
            "Plausibility": 14.00,
            "Feasibility": 14.00,
        },
        "UFCE3": {
            "Prox-Jac": 0.00,
            "Prox-Euc": 43.00,
            "Sparsity": 3.00,
            "Actionability": 18.00,
            "Plausibility": 18.00,
            "Feasibility": 18.00,
        },
    },
}

TUNED_RUN2: Dict[str, Dict[str, int]] = {
    "bank": {"radius": 60, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "bupa": {"radius": 70, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "grad": {"radius": 500, "n_neighbors": 400, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "wine": {"radius": 7, "n_neighbors": 1000, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "movie": {"radius": 160, "n_neighbors": 100, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
}

FINAL_RUNTIME_CONFIG: Dict[str, Dict[str, int]] = {
    "bank": {"radius": 500, "n_neighbors": 1000, "min_act": 0, "min_feas": 0, "ufce_flip_filter": 0},
    "bupa": {"radius": 70, "n_neighbors": 200, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "grad": {"radius": 500, "n_neighbors": 400, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "wine": {"radius": 7, "n_neighbors": 1000, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
    "movie": {"radius": 80, "n_neighbors": 50, "min_act": 1, "min_feas": 1, "ufce_flip_filter": 0},
}

FINAL_BLINDSPOT_BUNDLE = {
    "bank": {
        "uf_mode": "scaled_up_150",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "bupa": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "grad": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "wine": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
    "movie": {
        "uf_mode": "neutral_all_1",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    },
}

ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]

METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]
METHODS = ["UFCE1", "UFCE2", "UFCE3"]


@dataclass
class FoldResult:
    fold_name: str
    means: Dict[str, Dict[str, float]]
    times: Dict[str, float]


def safe_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg_name)
    except Exception:
        return "unknown"


def print_env_versions() -> None:
    pkgs = [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("numpy", safe_version("numpy")),
        ("pandas", safe_version("pandas")),
        ("scipy", safe_version("scipy")),
        ("scikit-learn", safe_version("scikit-learn")),
        ("matplotlib", safe_version("matplotlib")),
        ("ufce", getattr(ufce, "__version__", "unknown")),
    ]
    print("\n=== Environment / Library Versions ===")
    for k, v in pkgs:
        print(f"- {k:12s}: {v}")
    print("======================================\n")


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
        return {"Mcv": 1, "Alkphos": 1, "Sgpt": 1, "Sgot": 1, "Gammagt": 1, "Drinks": 1}
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


def map_nested_numeric(obj, fn):
    if isinstance(obj, dict):
        return {k: map_nested_numeric(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_nested_numeric(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_nested_numeric(v, fn) for v in obj)
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return fn(float(obj))
    return obj


def apply_uf_mode(author_uf: Dict, mode: str) -> Dict:
    if mode == "author_public":
        return copy.deepcopy(author_uf)
    if mode == "neutral_all_1":
        return map_nested_numeric(author_uf, lambda _x: 1.0)
    if mode == "scaled_up_150":
        return map_nested_numeric(author_uf, lambda x: x * 1.5)
    raise ValueError(f"Unsupported uf_mode: {mode}")


def apply_f2change_mode(author_f2change: List[str], mode: str) -> List[str]:
    if mode == "author_public":
        return list(author_f2change)
    raise ValueError(f"Unsupported f2change_mode for final reproduction: {mode}")


def print_feature_relations(X: pd.DataFrame, features: List[str], mi_pairs: List) -> None:
    print("=== Feature Relations ===")
    print(f"- Top-5 MI feature pairs: {mi_pairs[:5]}")
    Xf = X[features].copy()
    Xnum = Xf.select_dtypes(include=[np.number])
    if Xnum.shape[1] == 0:
        print("- No numeric columns found for Pearson correlation.")
        print("=========================\n")
        return

    corr = Xnum.corr()
    abs_corr = corr.abs()
    np.fill_diagonal(abs_corr.values, 0.0)
    stacked = abs_corr.stack().sort_values(ascending=False)
    top = stacked.head(10)

    print("- Top-10 |Pearson corr| pairs (numeric-only):")
    printed = 0
    for (a, b), v in top.items():
        if a < b:
            print(f"  {a} <-> {b}: {v:.2f}")
            printed += 1
        if printed >= 10:
            break
    print("=========================\n")


def resolve_effective_cfg(dataset: str, args) -> Dict[str, int]:
    config_source = FINAL_RUNTIME_CONFIG if args.runtime_profile == "final_freeze" else TUNED_RUN2

    if dataset not in config_source:
        allowed = ", ".join(sorted(config_source.keys()))
        raise ValueError(f"Dataset '{dataset}' missing from runtime profile. Allowed: {allowed}")

    tuned = config_source[dataset]
    return {
        "radius": int(args.radius) if args.radius is not None else int(tuned["radius"]),
        "n_neighbors": int(args.n_neighbors) if args.n_neighbors is not None else int(tuned["n_neighbors"]),
        "min_act": int(args.min_act) if args.min_act is not None else int(tuned["min_act"]),
        "min_feas": int(args.min_feas) if args.min_feas is not None else int(tuned["min_feas"]),
        "ufce_flip_filter": (
            int(args.ufce_flip_filter) if args.ufce_flip_filter is not None else int(tuned["ufce_flip_filter"])
        ),
    }


def _has_mad_scaler(scaler: Optional[Dict]) -> bool:
    return isinstance(scaler, dict) and "medians" in scaler and "mads" in scaler


def _mad_transform(df: pd.DataFrame, scaler: Dict, numf: List[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in numf if c in out.columns]
    if len(cols) == 0:
        return out
    med = scaler["medians"].reindex(cols)
    mad = scaler["mads"].reindex(cols).replace(0, 1.0)
    out.loc[:, cols] = (out.loc[:, cols] - med) / mad
    return out


def _mad_inverse_inplace(df: pd.DataFrame, scaler: Dict, numf: List[str]) -> None:
    cols = [c for c in numf if c in df.columns]
    if len(cols) == 0:
        return
    med = scaler["medians"].reindex(cols)
    mad = scaler["mads"].reindex(cols).replace(0, 1.0)
    df.loc[:, cols] = (df.loc[:, cols] * mad) + med


def build_movie_distance_scaler(
    datasetdf: pd.DataFrame,
    features: List[str],
    numf: List[str],
    outcome_label: str,
) -> Dict[str, object]:
    """
    Build movie-only affine scaler:
      scaled = (raw - min) / ((max - min) / 100)
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


def apply_distance_scaler(df: pd.DataFrame, scaler: Optional[Dict[str, object]]) -> pd.DataFrame:
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


def _active_ufce_instance() -> UFCE:
    if getattr(cfmethods, "ufc", None) is not None:
        return cfmethods.ufc
    if getattr(eval_module, "ufc", None) is not None:
        return eval_module.ufc
    return ufc


def _scalar(v) -> float:
    arr = np.asarray(v).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _mean_or_nan(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _shared_numeric_features(numf: List[str], *dfs: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in numf:
        if all(df is not None and c in df.columns for df in dfs):
            cols.append(c)
    return cols


def _row_l2_distance(lhs: pd.DataFrame, rhs: pd.DataFrame, cols: List[str]) -> float:
    if lhs is None or rhs is None or lhs.empty or rhs.empty or len(cols) == 0:
        return float("nan")
    left = lhs.iloc[0][cols].to_numpy(dtype=float)
    right = rhs.iloc[0][cols].to_numpy(dtype=float)
    return float(np.linalg.norm(left - right))


def _fmt_dbg_float(val: float) -> str:
    try:
        fv = float(val)
    except Exception:
        return "nan"
    if not np.isfinite(fv):
        return "nan"
    return f"{fv:.6f}"


def _safe_implausibility_count(
    active_ufc: UFCE,
    method_name: str,
    cfdf: pd.DataFrame,
    testdf: pd.DataFrame,
    xtrain: pd.DataFrame,
) -> float:
    try:
        val = active_ufc.implausibility(
            cfdf.copy(),
            testdf.copy(),
            xtrain.copy(),
            len(cfdf),
            0,
            method_name=method_name,
        )
        if isinstance(val, dict):
            return float(int(val.get("count", 0)))
        return float(val)
    except Exception:
        return 0.0


def _safe_feasibility_count(
    active_ufc: UFCE,
    cfdf: pd.DataFrame,
    testdf: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: List[str],
    f2change: List[str],
    bb_model,
    desired_outcome: float,
    uf: Dict[str, float],
) -> float:
    try:
        out = active_ufc.feasibility(
            testdf.copy(),
            cfdf.copy(),
            xtrain.copy(),
            features,
            f2change,
            bb_model,
            desired_outcome,
            uf,
            0,
            method="other",
        )
        if out is None:
            return 0.0
        if isinstance(out, (int, float)):
            return float(out)
        if isinstance(out, tuple) and len(out) >= 1:
            return float(out[0])
        return 0.0
    except Exception:
        return 0.0


def evaluate_ufce_only_metrics(
    *,
    onecfs: pd.DataFrame,
    onetest: pd.DataFrame,
    twocfs: pd.DataFrame,
    twotest: pd.DataFrame,
    threecfs: pd.DataFrame,
    threetest: pd.DataFrame,
    catf: List[str],
    numf: List[str],
    features: List[str],
    f2change: List[str],
    uf: Dict[str, float],
    xtrain: pd.DataFrame,
    bb_model,
    desired_outcome: float,
    contprox_distance_scaler: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, float]]:
    active_ufc = _active_ufce_instance()
    method_triplets = [
        ("UFCE1", onecfs, onetest),
        ("UFCE2", twocfs, twotest),
        ("UFCE3", threecfs, threetest),
    ]
    means: Dict[str, Dict[str, float]] = {m: {} for m in METHODS}

    for method, cfdf, testdf in method_triplets:
        # Prox-Jac
        if len(catf) == 0:
            means[method]["Prox-Jac"] = float("nan")
        else:
            vals: List[float] = []
            n = min(len(cfdf), len(testdf))
            for i in range(n):
                dist = active_ufc.categorical_distance(testdf[i : i + 1], cfdf[i : i + 1], catf, metric="jaccard", agg=None)
                vals.append(_scalar(dist))
            means[method]["Prox-Jac"] = _mean_or_nan(vals)

        # Prox-Euc
        vals = []
        n = min(len(cfdf), len(testdf))
        cf_dist_df = apply_distance_scaler(cfdf, contprox_distance_scaler) if contprox_distance_scaler is not None else cfdf
        test_dist_df = apply_distance_scaler(testdf, contprox_distance_scaler) if contprox_distance_scaler is not None else testdf
        for i in range(n):
            dist = active_ufc.continuous_distance(
                test_dist_df[i : i + 1],
                cf_dist_df[i : i + 1],
                numf,
                metric="euclidean",
                agg=None,
            )
            vals.append(_scalar(dist))
        means[method]["Prox-Euc"] = _mean_or_nan(vals)

        # Sparsity
        sparsity_d, _ = active_ufc.sparsity_count(cfdf.copy(), testdf.copy(), numf, numf)
        sv = np.asarray(list(sparsity_d.values()), dtype=float) if len(sparsity_d) > 0 else np.asarray([])
        means[method]["Sparsity"] = float(np.mean(sv)) if sv.size > 0 else float("nan")

        # Actionability
        try:
            act_cfs, _flag, _ids, _temp = active_ufc.actionability(
                cfdf.copy(),
                testdf.copy(),
                features,
                f2change,
                0,
                uf,
                method="other",
            )
            means[method]["Actionability"] = float(len(act_cfs))
        except Exception:
            means[method]["Actionability"] = 0.0

        means[method]["Plausibility"] = _safe_implausibility_count(active_ufc, method, cfdf, testdf, xtrain)
        means[method]["Feasibility"] = _safe_feasibility_count(
            active_ufc,
            cfdf,
            testdf,
            xtrain,
            features,
            f2change,
            bb_model,
            desired_outcome,
            uf,
        )

    return means


def run_one_fold(
    *,
    dataset: str,
    fold_df: pd.DataFrame,
    x_all: pd.DataFrame,
    xtest: pd.DataFrame,
    xtrain: pd.DataFrame,
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
    fold_name: str,
    scaler: Optional[Dict],
    flip_filter_enabled: bool,
    movie_distance_scaler: Optional[Dict[str, object]],
    fold_index: int,
    debug: int,
    prox_euc_contract_debug: int,
    contract_debug_fold: int,
    contract_debug_pos: int,
    contract_debug_method: str,
) -> FoldResult:
    use_mad = _has_mad_scaler(scaler)

    # Movie: keep UFCE logic in raw space, but use scaled distance space for neighbor search.
    if dataset == "movie" and movie_distance_scaler is not None:
        fold_df_ufce = fold_df[features].copy()
        data_lab1_ufce = data_lab1[features].copy()
        fold_df_dist = apply_distance_scaler(fold_df_ufce, movie_distance_scaler)
        data_lab1_dist = apply_distance_scaler(data_lab1_ufce, movie_distance_scaler)
        distance_scaler = movie_distance_scaler
    elif use_mad:
        # Non-movie MAD path when scaler is provided by pipeline.
        fold_df_ufce = _mad_transform(fold_df[features].copy(), scaler, numf)
        data_lab1_ufce = _mad_transform(data_lab1[features].copy(), scaler, numf)
        fold_df_dist = None
        data_lab1_dist = None
        distance_scaler = None
    else:
        fold_df_ufce = fold_df[features].copy()
        data_lab1_ufce = data_lab1[features].copy()
        fold_df_dist = None
        data_lab1_dist = None
        distance_scaler = None

    dbg_fold = fold_df_dist if fold_df_dist is not None else fold_df_ufce
    dbg_lab1 = data_lab1_dist if data_lab1_dist is not None else data_lab1_ufce
    if not dbg_fold.empty and not dbg_lab1.empty:
        x0 = dbg_fold[features].iloc[0:1].values
        lab1 = dbg_lab1[features].values
        dists = np.linalg.norm(lab1 - x0, axis=1)
        print(f"[DBG] Scaled Distances: min={dists.min():.2f}, mean={dists.mean():.2f}, max={dists.max():.2f}")

    onecfs, t1, idx1 = cfmethods.sfexp(
        x_all,
        data_lab1_ufce,
        fold_df_ufce[:],
        uf,
        step,
        f2change,
        numf,
        catf,
        bb_model,
        desired_outcome,
        no_cf,
        features,
        flip_filter_enabled=bool(flip_filter_enabled),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=fold_df_dist,
        distance_scaler=distance_scaler,
    )
    twocfs, t2, idx2 = cfmethods.dfexp(
        x_all,
        data_lab1_ufce,
        fold_df_ufce[:],
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
        flip_filter_enabled=bool(flip_filter_enabled),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=fold_df_dist,
        distance_scaler=distance_scaler,
    )
    threecfs, t3, idx3 = cfmethods.tfexp(
        x_all,
        data_lab1_ufce,
        fold_df_ufce[:],
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
        flip_filter_enabled=bool(flip_filter_enabled),
        distance_data_lab1=data_lab1_dist,
        distance_X_test=fold_df_dist,
        distance_scaler=distance_scaler,
    )

    if use_mad and dataset != "movie":
        for cf_df in [onecfs, twocfs, threecfs]:
            if cf_df is not None and not cf_df.empty:
                _mad_inverse_inplace(cf_df, scaler, numf)

    if int(debug) == 1:
        n_test = int(len(fold_df))
        one_total, two_total, three_total = int(len(onecfs)), int(len(twocfs)), int(len(threecfs))
        one_solved, two_solved, three_solved = int(len(idx1)), int(len(idx2)), int(len(idx3))
        one_mean = float(one_total / max(one_solved, 1))
        two_mean = float(two_total / max(two_solved, 1))
        three_mean = float(three_total / max(three_solved, 1))
        print(
            "[DBG][COVERAGE] "
            f"fold={fold_index} n_test={n_test} "
            f"UFCE1 solved={one_solved} mean_cf={one_mean:.3f} total_cf_rows={one_total} | "
            f"UFCE2 solved={two_solved} mean_cf={two_mean:.3f} total_cf_rows={two_total} | "
            f"UFCE3 solved={three_solved} mean_cf={three_mean:.3f} total_cf_rows={three_total}"
        )

    if int(prox_euc_contract_debug) == 1 and int(fold_index) == int(contract_debug_fold):
        method_to_payload = {
            "UFCE1": (onecfs.reset_index(drop=True), [int(v) for v in idx1]),
            "UFCE2": (twocfs.reset_index(drop=True), [int(v) for v in idx2]),
            "UFCE3": (threecfs.reset_index(drop=True), [int(v) for v in idx3]),
        }
        method = str(contract_debug_method)
        debug_pos = int(contract_debug_pos)
        cfdf_method, idx_method = method_to_payload.get(method, (pd.DataFrame(), []))

        status = "ok"
        raw_l2 = float("nan")
        z_l2 = float("nan")
        mm_l2 = float("nan")

        if debug_pos < 0 or debug_pos >= len(fold_df):
            status = "no_cf_for_instance"
        else:
            factual = fold_df.iloc[[debug_pos]][features].reset_index(drop=True)
            pair_pos = next((i for i, src_pos in enumerate(idx_method) if int(src_pos) == debug_pos), None)
            if pair_pos is None or pair_pos >= len(cfdf_method):
                status = "no_cf_for_instance"
            else:
                cf_row = cfdf_method.iloc[[pair_pos]].reset_index(drop=True)
                dist_cols = _shared_numeric_features(numf, xtrain, factual, cf_row)
                raw_l2 = _row_l2_distance(factual, cf_row, dist_cols)
                if len(dist_cols) > 0:
                    try:
                        z_scaler = StandardScaler().fit(xtrain.loc[:, dist_cols])
                        factual_z = pd.DataFrame(z_scaler.transform(factual.loc[:, dist_cols]), columns=dist_cols)
                        cf_z = pd.DataFrame(z_scaler.transform(cf_row.loc[:, dist_cols]), columns=dist_cols)
                        z_l2 = _row_l2_distance(factual_z, cf_z, dist_cols)
                    except Exception:
                        z_l2 = float("nan")
                    try:
                        mm_scaler = MinMaxScaler().fit(xtrain.loc[:, dist_cols])
                        factual_mm = pd.DataFrame(mm_scaler.transform(factual.loc[:, dist_cols]), columns=dist_cols)
                        cf_mm = pd.DataFrame(mm_scaler.transform(cf_row.loc[:, dist_cols]), columns=dist_cols)
                        mm_l2 = _row_l2_distance(factual_mm, cf_mm, dist_cols)
                    except Exception:
                        mm_l2 = float("nan")

        print(
            "[DBG][PROX-EUC-CONTRACT] "
            f"dataset={dataset} fold={fold_index} pos={debug_pos} "
            f"method={method} fit_scope=train status={status}"
        )
        print(
            f"raw_l2={_fmt_dbg_float(raw_l2)} "
            f"zscore_l2={_fmt_dbg_float(z_l2)} "
            f"minmax01_l2={_fmt_dbg_float(mm_l2)}"
        )

    onetest = fold_df.iloc[idx1][features].reset_index(drop=True)
    twotest = fold_df.iloc[idx2][features].reset_index(drop=True)
    threetest = fold_df.iloc[idx3][features].reset_index(drop=True)

    means = evaluate_ufce_only_metrics(
        onecfs=onecfs.reset_index(drop=True),
        onetest=onetest,
        twocfs=twocfs.reset_index(drop=True),
        twotest=twotest,
        threecfs=threecfs.reset_index(drop=True),
        threetest=threetest,
        catf=catf,
        numf=numf,
        features=features,
        f2change=f2change,
        uf=uf,
        xtrain=xtrain,
        bb_model=bb_model,
        desired_outcome=desired_outcome,
        contprox_distance_scaler=movie_distance_scaler if dataset == "movie" else None,
    )
    times = {"UFCE1": float(t1), "UFCE2": float(t2), "UFCE3": float(t3)}
    return FoldResult(fold_name=fold_name, means=means, times=times)


def aggregate_results(folds: List[FoldResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rows = []
    time_rows = []
    for fr in folds:
        for method in METHODS:
            row = {"fold": fr.fold_name, "method": method}
            row.update(fr.means[method])
            rows.append(row)
        time_rows.append(fr.times)
    df = pd.DataFrame(rows)
    time_df = pd.DataFrame(time_rows)
    mean_df = df.groupby("method")[METRICS].mean().T
    std_df = df.groupby("method")[METRICS].std(ddof=1).T
    return mean_df[METHODS], std_df[METHODS], time_df[METHODS].mean()


def print_ours_table(mean_df: pd.DataFrame, std_df: pd.DataFrame, time_mean: pd.Series) -> None:
    print("=== Ours (Fold-mean aggregated) ===")
    print("Mean (across folds):")
    print(mean_df.to_string(float_format=lambda x: f"{x:0.6f}"))
    print("\nStd  (across folds):")
    print(std_df.to_string(float_format=lambda x: f"{x:0.6f}"))
    print("\nMean time (sec) across folds:")
    for m in METHODS:
        print(f"- {m}: {time_mean[m]:0.2f}s")
    print("==================================\n")


def print_author_delta(dataset: str, mean_df: pd.DataFrame) -> None:
    author = AUTHOR_TABLE7.get(dataset, None)
    if author is None:
        print(f"[WARN] No AUTHOR_TABLE7 values registered for dataset='{dataset}'. Skipping delta table.\n")
        return

    print("=== Authors vs Ours (Delta = Ours - Authors) ===")
    out_rows = []
    for method in METHODS:
        a = author.get(method, None)
        if a is None:
            continue
        for metric in METRICS:
            ours = float(mean_df.loc[metric, method])
            auth = float(a[metric])
            out_rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Authors": auth,
                    "Ours": ours,
                    "Delta(Ours-Authors)": ours - auth,
                }
            )
    out = pd.DataFrame(out_rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print("==============================================\n")


def plot_2x3_metrics(mean_df: pd.DataFrame, std_df: pd.DataFrame, out_path: str, title: str) -> None:
    if plt is None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("matplotlib is not available in this environment; plot was skipped.\n")
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    x = np.arange(len(METHODS))
    for ax, metric in zip(axes, METRICS):
        y = mean_df.loc[metric, METHODS].values
        e = std_df.loc[metric, METHODS].fillna(0.0).values
        ax.bar(x, y, yerr=e, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_title(metric)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_for_dataset(dataset: str, args, run_id: str) -> Dict[str, object]:
    t0 = time.time()
    cfg = resolve_effective_cfg(dataset, args)
    print(
        "[CFG] "
        f"dataset={dataset} radius={cfg['radius']} n_neighbors={cfg['n_neighbors']} "
        f"min_act={cfg['min_act']} min_feas={cfg['min_feas']} "
        f"flip={cfg['ufce_flip_filter']} contprox_metric={args.contprox_metric} no_cf={args.no_cf}"
    )
    if int(args.debug) == 1:
        print(
            "[CFG][METRIC-CONTRACT] "
            f"Action/Plaus/Feas are COUNTS over generated CFs; max=no_cf={args.no_cf}"
        )

    data_path = os.path.join(args.data_dir, f"{dataset}.csv")
    datasetdf = pd.read_csv(data_path)
    out = classify_dataset_getModel(datasetdf, data_name=dataset)
    scaler = None
    if len(out) == 8:
        lr, lr_mean, lr_std, xtest, xtrain, x_all, _y, datasetdf = out
    elif len(out) == 9:
        lr, lr_mean, lr_std, xtest, xtrain, x_all, _y, datasetdf, scaler = out
    else:
        raise ValueError(f"Unexpected classify_dataset_getModel return length: {len(out)}")

    (
        features,
        catf,
        numf,
        uf,
        f2change,
        outcome_label,
        desired_outcome,
        _nbr_features,
        protectf,
        _data_lab0,
        data_lab1,
    ) = get_dataset_constraints(dataset, datasetdf)
    
    bundle_cfg = {
        "uf_mode": "author_public",
        "step_mode": "local_reproduction",
        "f2change_mode": "author_public",
    }

    if args.bundle_mode == "final_blindspot_best":
        bundle_cfg = FINAL_BLINDSPOT_BUNDLE[dataset]

    uf = apply_uf_mode(uf, bundle_cfg["uf_mode"])
    f2change = apply_f2change_mode(f2change, bundle_cfg["f2change_mode"])
    step = get_step_config(dataset)

    print(
        "[BUNDLE] "
        f"dataset={dataset} bundle_mode={args.bundle_mode} "
        f"uf_mode={bundle_cfg['uf_mode']} "
        f"step_mode={bundle_cfg['step_mode']} "
        f"f2change_mode={bundle_cfg['f2change_mode']}"
    )
    mi_fp = ufc.get_top_MI_features(x_all, features)
    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )

    print(f"Dataset: {dataset}")
    print("Black-box: LogisticRegression (author pipeline)")
    print(f"CV accuracy: {lr_mean:.6f} +/- {lr_std:.6f}")
    print_feature_relations(x_all, features, mi_fp)

    cfmethods.initUFCE(
        radius=cfg["radius"],
        n_neighbors=cfg["n_neighbors"],
        contprox_metric=args.contprox_metric,
        min_act=cfg["min_act"],
        min_feas=cfg["min_feas"],
        atol=1e-5,
    )
    eval_module.ufc = cfmethods.ufc

    testfold_path = os.path.join(args.folds_dir, dataset, "totest")
    testfolds = sorted(glob.glob(os.path.join(testfold_path, "*.csv")))
    if not testfolds:
        raise FileNotFoundError(f"No test folds found in {testfold_path}")
    if args.fold_file:
        target_name = os.path.basename(str(args.fold_file))
        selected = [fp for fp in testfolds if os.path.basename(fp) == target_name]
        if not selected:
            raise FileNotFoundError(
                f"Requested fold_file='{target_name}' was not found in {testfold_path}"
            )
        testfolds = selected
    if args.max_folds > 0:
        testfolds = testfolds[: args.max_folds]

    print(f"Found {len(testfolds)} folds. Processing...")
    folds_results: List[FoldResult] = []

    for fold_idx, fp in enumerate(testfolds):
        fold_name = os.path.basename(fp)
        print(f"\n--- Fold {fold_idx}: {fold_name} ---")
        fold_df = pd.read_csv(fp)
        fr = run_one_fold(
            dataset=dataset,
            fold_df=fold_df,
            x_all=x_all,
            xtest=xtest,
            xtrain=xtrain,
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
            fold_name=fold_name,
            scaler=scaler,
            flip_filter_enabled=bool(cfg["ufce_flip_filter"]),
            movie_distance_scaler=movie_distance_scaler,
            fold_index=fold_idx,
            debug=args.debug,
            prox_euc_contract_debug=args.prox_euc_contract_debug,
            contract_debug_fold=args.contract_debug_fold,
            contract_debug_pos=args.contract_debug_pos,
            contract_debug_method=args.contract_debug_method,
        )
        print("Times (sec): " + ", ".join([f"{m}={fr.times[m]:0.2f}" for m in METHODS]))
        folds_results.append(fr)

    mean_df, std_df, time_mean = aggregate_results(folds_results)
    print_ours_table(mean_df, std_df, time_mean)
    print_author_delta(dataset, mean_df)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, f"summary_{dataset}_{run_id}.csv")
    mean_df.to_csv(summary_csv)

    plot_path = os.path.join(args.out_dir, f"metrics_2x3_{dataset}_{run_id}.png")
    plot_2x3_metrics(
        mean_df=mean_df,
        std_df=std_df,
        out_path=plot_path,
        title=f"UFCE v3 Reproduction — {dataset.capitalize()}",
    )

    runtime_sec = float(time.time() - t0)
    print("\nExecution Complete.")
    print(f"- Summary CSV saved: {summary_csv}")
    print(f"- Metrics Plot saved: {plot_path}")
    print(f"Total runtime: {runtime_sec:0.2f}s")

    return {
        "dataset": dataset,
        "status": "ok",
        "runtime_sec": runtime_sec,
        "n_folds": int(len(testfolds)),
        "error": "",
        "summary_csv": summary_csv,
        "plot_path": plot_path,
    }


def write_run_manifest(args, run_id: str, out_dir: str, extra: Optional[Dict[str, object]] = None) -> str:
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/reproduce_results_v3.py",
        "dataset": args.dataset,
        "runtime_profile": args.runtime_profile,
        "bundle_mode": args.bundle_mode,
        "no_cf": args.no_cf,
        "max_folds": args.max_folds,
        "fold_file": args.fold_file,
        "contprox_metric": args.contprox_metric,
        "ufce_flip_filter_override": args.ufce_flip_filter,
        "data_dir": args.data_dir,
        "folds_dir": args.folds_dir,
        "out_dir": out_dir,
        "final_runtime_config": FINAL_RUNTIME_CONFIG,
        "final_blindspot_bundle": FINAL_BLINDSPOT_BUNDLE,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": safe_version("numpy"),
            "pandas": safe_version("pandas"),
            "scipy": safe_version("scipy"),
            "scikit-learn": safe_version("scikit-learn"),
            "matplotlib": safe_version("matplotlib"),
            "ufce": getattr(ufce, "__version__", "unknown"),
        },
        "extra": extra or {},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_manifest_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime_profile",
        type=str,
        default="tuned_run2",
        choices=["tuned_run2", "final_freeze"],
        help="Runtime config profile to use.",
    )
    parser.add_argument(
        "--bundle_mode",
        type=str,
        default="author_public",
        choices=["author_public", "final_blindspot_best"],
        help="UF/f2change/step bundle mode.",
    )
    parser.add_argument("--dataset", type=str, default="bank", choices=["bank", "grad", "wine", "bupa", "movie", "all"])
    parser.add_argument("--data_dir", type=str, default=os.path.join("ufce", "data"), help="Repo-relative path to data folder")
    parser.add_argument("--folds_dir", type=str, default=os.path.join("ufce", "data", "folds"), help="Repo-relative path to folds folder")
    parser.add_argument("--no_cf", type=int, default=10, help="Number of CFs requested per instance")
    parser.add_argument("--max_folds", type=int, default=0, help="0=all folds; otherwise limit folds")
    parser.add_argument(
        "--fold_file",
        type=str,
        default=None,
        help="Optional exact fold basename to run (e.g., testfold_0_pred_0.csv)",
    )
    parser.add_argument("--n_neighbors", type=int, default=None, help="Override tuned n_neighbors when provided")
    parser.add_argument("--radius", type=int, default=None, help="Override tuned radius when provided")
    parser.add_argument("--min_act", type=int, default=None, help="Override tuned min_act when provided")
    parser.add_argument("--min_feas", type=int, default=None, help="Override tuned min_feas when provided")
    parser.add_argument(
        "--ufce_flip_filter",
        type=int,
        default=None,
        choices=[0, 1],
        help="Override tuned UFCE flipping filter (0/1) when provided",
    )
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1], help="Enable trust debug logs (0/1)")
    parser.add_argument(
        "--prox_euc_contract_debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable single-instance Prox-Euc contract debug block (0/1)",
    )
    parser.add_argument(
        "--contract_debug_fold",
        type=int,
        default=0,
        help="Fold index for Prox-Euc contract debug",
    )
    parser.add_argument(
        "--contract_debug_pos",
        type=int,
        default=0,
        help="Index position inside fold test list for Prox-Euc contract debug",
    )
    parser.add_argument(
        "--contract_debug_method",
        type=str,
        default="UFCE1",
        choices=METHODS,
        help="UFCE method for Prox-Euc contract debug (UFCE1/UFCE2/UFCE3)",
    )
    parser.add_argument("--contprox_metric", type=str, default="euclidean", help="ContProximity metric to use in UFCE")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("archive", "part1_old_runs", "repro_v3_out"),
        help="Output directory for plots/tables",
    )
    args = parser.parse_args()

    print_env_versions()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest_path = write_run_manifest(args, run_id, args.out_dir)
    print(f"- Run manifest saved: {manifest_path}")

    if args.dataset != "all":
        run_for_dataset(args.dataset, args, run_id)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    batch_records: List[Dict[str, object]] = []
    batch_t0 = time.time()
    for dataset in ALL_DATASETS:
        ds_t0 = time.time()
        print(f"\n==================== BATCH DATASET: {dataset} ====================")
        try:
            rec = run_for_dataset(dataset, args, run_id)
        except Exception as exc:
            rec = {
                "dataset": dataset,
                "status": "failed",
                "runtime_sec": float(time.time() - ds_t0),
                "n_folds": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "summary_csv": "",
                "plot_path": "",
            }
            print(f"[ERROR] dataset={dataset} failed with {type(exc).__name__}: {exc}")
        batch_records.append(rec)

    batch_df = pd.DataFrame(
        batch_records,
        columns=["dataset", "status", "runtime_sec", "n_folds", "error", "summary_csv", "plot_path"],
    )
    batch_csv = os.path.join(args.out_dir, f"batch_summary_{run_id}.csv")
    batch_df.to_csv(batch_csv, index=False)

    print("\n==================== BATCH SUMMARY ====================")
    print(batch_df.to_string(index=False))
    print(f"- Batch summary CSV saved: {batch_csv}")
    print(f"- Batch total runtime: {time.time() - batch_t0:0.2f}s")

    failed_count = int(np.sum(batch_df["status"].values == "failed"))
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
