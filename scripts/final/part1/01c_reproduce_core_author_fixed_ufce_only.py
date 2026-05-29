#!/usr/bin/env python3
"""
UFCE core_author_fixed Table 7 reproduction runner.

This script is the fixed-core counterpart of 01b_reproduce_ufce_only.py:
- UFCE1/UFCE2/UFCE3 only.
- Uses ufce.core_author_fixed.
- Compares reproduced metrics with the public UFCE-only Table 7 values.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path(os.getcwd()).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
if plt is not None:
    plt.style.use("seaborn-whitegrid")
pd.set_option("display.max_columns", None)

from ufce.core_author_fixed import UFCE
from ufce.core_author_fixed import cfmethods
from ufce.core_author_fixed import evaluations as eval_module
from ufce.core_author_fixed.constraint_profiles import (
    AUTHOR_PUBLIC,
    CONSTRAINT_PROFILES,
    DOMAIN_ACTIONABLE,
    DOMAIN_ACTIONABLE_CHANGED_DATASETS,
    resolve_constraint_profile,
)
from ufce.core_author_fixed.data_processing import (
    classify_dataset_getModel,
    get_bank_user_constraints,
    get_bupa_user_constraints,
    get_grad_user_constraints,
    get_movie_user_constraints,
    get_wine_user_constraints,
)


METHODS = ["UFCE1", "UFCE2", "UFCE3"]
METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility", "Validity"]
ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]

AUTHOR_TABLE7: Dict[str, Dict[str, Dict[str, float]]] = {
    "bank": {
        "UFCE1": {"Prox-Jac": 0.60, "Prox-Euc": 10.00, "Sparsity": 1.00, "Actionability": 14.00, "Plausibility": 14.00, "Feasibility": 14.00},
        "UFCE2": {"Prox-Jac": 0.00, "Prox-Euc": 23.10, "Sparsity": 2.00, "Actionability": 30.00, "Plausibility": 30.00, "Feasibility": 30.00},
        "UFCE3": {"Prox-Jac": 0.00, "Prox-Euc": 40.12, "Sparsity": 3.00, "Actionability": 44.00, "Plausibility": 43.00, "Feasibility": 43.00},
    },
    "grad": {
        "UFCE1": {"Prox-Jac": 0.00, "Prox-Euc": 2.34, "Sparsity": 1.00, "Actionability": 8.00, "Plausibility": 8.00, "Feasibility": 8.00},
        "UFCE2": {"Prox-Jac": 0.00, "Prox-Euc": 4.85, "Sparsity": 2.00, "Actionability": 13.00, "Plausibility": 13.00, "Feasibility": 13.00},
        "UFCE3": {"Prox-Jac": 0.00, "Prox-Euc": 6.32, "Sparsity": 2.80, "Actionability": 13.00, "Plausibility": 13.00, "Feasibility": 13.00},
    },
    "wine": {
        "UFCE1": {"Prox-Jac": float("nan"), "Prox-Euc": 14.90, "Sparsity": 1.00, "Actionability": 43.00, "Plausibility": 28.00, "Feasibility": 28.00},
        "UFCE2": {"Prox-Jac": float("nan"), "Prox-Euc": 8.45, "Sparsity": 2.00, "Actionability": 50.00, "Plausibility": 41.00, "Feasibility": 41.00},
        "UFCE3": {"Prox-Jac": float("nan"), "Prox-Euc": 21.95, "Sparsity": 3.00, "Actionability": 50.00, "Plausibility": 42.00, "Feasibility": 42.00},
    },
    "bupa": {
        "UFCE1": {"Prox-Jac": float("nan"), "Prox-Euc": 10.00, "Sparsity": 1.00, "Actionability": 17.00, "Plausibility": 15.00, "Feasibility": 15.00},
        "UFCE2": {"Prox-Jac": float("nan"), "Prox-Euc": 9.00, "Sparsity": 2.00, "Actionability": 15.00, "Plausibility": 15.00, "Feasibility": 15.00},
        "UFCE3": {"Prox-Jac": float("nan"), "Prox-Euc": 17.10, "Sparsity": 2.90, "Actionability": 13.00, "Plausibility": 13.00, "Feasibility": 13.00},
    },
    "movie": {
        "UFCE1": {"Prox-Jac": 0.00, "Prox-Euc": 20.00, "Sparsity": 1.00, "Actionability": 20.00, "Plausibility": 8.00, "Feasibility": 8.00},
        "UFCE2": {"Prox-Jac": 0.00, "Prox-Euc": 32.00, "Sparsity": 2.00, "Actionability": 14.00, "Plausibility": 14.00, "Feasibility": 14.00},
        "UFCE3": {"Prox-Jac": 0.00, "Prox-Euc": 43.00, "Sparsity": 3.00, "Actionability": 18.00, "Plausibility": 18.00, "Feasibility": 18.00},
    },
}

DEFAULT_RUNTIME: Dict[str, Dict[str, int]] = {
    "bank": {"radius": 500, "n_neighbors": 1000},
    "bupa": {"radius": 70, "n_neighbors": 200},
    "grad": {"radius": 500, "n_neighbors": 400},
    "wine": {"radius": 7, "n_neighbors": 1000},
    "movie": {"radius": 80, "n_neighbors": 50},
}

STEP_CONFIG: Dict[str, Dict[str, float]] = {
    "bank": {"Income": 1, "Family": 1, "CCAvg": 0.1, "Education": 1, "Mortgage": 1, "SecuritiesAccount": 1, "CDAccount": 1, "Online": 1, "CreditCard": 1},
    "bupa": {"Mcv": 1, "Alkphos": 1, "Sgpt": 1, "Sgot": 1, "Gammagt": 1, "Drinks": 1},
    "grad": {"GRE Score": 1, "TOEFL Score": 1, "University Rating": 1, "SOP": 1, "LOR": 1, "CGPA": 0.1, "Research": 1},
    "wine": {"fixed acidity": 0.5, "volatile acidity": 0.1, "citric acid": 0.1, "residual sugar": 0.5, "free sulfur dioxide": 1.0, "total sulfur dioxide": 1.0, "density": 0.1, "pH": 0.5, "alcohol": 0.5},
    "movie": {"Production_expense": 3, "Num_multiplex": 3, "Multiplex_coverage": 0.2, "Movie_length": 5, "Lead_Actor_Rating": 1.0, "Lead_Actress_rating": 1.0, "Director_rating": 1.0, "Producer_rating": 1.0, "Genre": 1, "Collection": 500, "Budget": 3000},
}


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
    print("=== Environment / Library Versions ===")
    for key, value in [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("numpy", safe_version("numpy")),
        ("pandas", safe_version("pandas")),
        ("scipy", safe_version("scipy")),
        ("scikit-learn", safe_version("scikit-learn")),
    ]:
        print(f"- {key:12s}: {value}")
    print("======================================")


def get_dataset_constraints(dataset: str, datasetdf: pd.DataFrame):
    if dataset == "bank":
        return get_bank_user_constraints(datasetdf)
    if dataset == "bupa":
        return get_bupa_user_constraints(datasetdf)
    if dataset == "grad":
        return get_grad_user_constraints(datasetdf)
    if dataset == "wine":
        return get_wine_user_constraints(datasetdf)
    if dataset == "movie":
        return get_movie_user_constraints(datasetdf)
    raise ValueError(f"Unsupported dataset: {dataset}")


def normalize_mi_feature_pairs(mi_pairs: Sequence[Sequence[Any]], feature_order: Sequence[str]) -> List[List[str]]:
    positions = {str(feature): idx for idx, feature in enumerate(feature_order)}
    fallback = len(positions)
    out: List[List[str]] = []
    for pair in mi_pairs:
        items = [str(feature) for feature in list(pair)]
        out.append(sorted(items, key=lambda feature: (positions.get(feature, fallback), feature)))
    return out


def build_mad_distance_scaler(reference_df: pd.DataFrame, features: List[str], numf: List[str]) -> Dict[str, object]:
    cols = [col for col in numf if col in features and col in reference_df.columns]
    base = reference_df.loc[:, features].copy()
    medians = base[cols].median()
    mads = (base[cols] - medians).abs().median().replace(0.0, 1.0)
    return {
        "kind": "mad",
        "scale_cols": cols,
        "medians": medians.astype(float),
        "mads": mads.astype(float),
        "constant_cols": [col for col in cols if float(base[col].nunique()) <= 1],
    }


def apply_distance_scaler(df: pd.DataFrame, scaler: Optional[Dict[str, object]]) -> pd.DataFrame:
    if scaler is None or df is None or df.empty:
        return df.copy()
    out = df.copy()
    cols = [col for col in scaler.get("scale_cols", []) if col in out.columns]
    if not cols:
        return out
    out.loc[:, cols] = (out.loc[:, cols] - scaler["medians"].reindex(cols)) / scaler["mads"].reindex(cols)
    for col in scaler.get("constant_cols", []):
        if col in out.columns:
            out.loc[:, col] = 0.0
    return out


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def mean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def scalar(value: Any) -> float:
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def evaluate_metrics(
    *,
    active_ufc: UFCE,
    method_payloads: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    catf: List[str],
    numf: List[str],
    features: List[str],
    f2change: List[str],
    uf: Dict[str, float],
    xtrain: pd.DataFrame,
    model: Any,
    desired_outcome: float,
    distance_scaler: Optional[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {method: {} for method in METHODS}
    for method, (cfdf_raw, testdf_raw) in method_payloads.items():
        cfdf = cfdf_raw.reset_index(drop=True)
        testdf = testdf_raw.reset_index(drop=True)
        n = min(len(cfdf), len(testdf))

        if len(catf) == 0:
            out[method]["Prox-Jac"] = float("nan")
        else:
            vals = [
                scalar(active_ufc.categorical_distance(testdf[i : i + 1], cfdf[i : i + 1], catf, metric="jaccard", agg=None))
                for i in range(n)
            ]
            out[method]["Prox-Jac"] = mean_or_nan(vals)

        cf_dist = apply_distance_scaler(cfdf, distance_scaler)
        test_dist = apply_distance_scaler(testdf, distance_scaler)
        vals = [
            scalar(active_ufc.continuous_distance(test_dist[i : i + 1], cf_dist[i : i + 1], numf, metric="euclidean", agg=None))
            for i in range(n)
        ]
        out[method]["Prox-Euc"] = mean_or_nan(vals)

        sparsity_d, _ = active_ufc.sparsity_count(cfdf.copy(), testdf.copy(), numf, numf)
        out[method]["Sparsity"] = mean_or_nan(list(sparsity_d.values()))

        try:
            act_cfs, _flag, _ids, _tmp = active_ufc.actionability(
                cfdf.copy(), testdf.copy(), features, f2change, 0, uf, method="other"
            )
            out[method]["Actionability"] = float(len(act_cfs))
        except Exception:
            out[method]["Actionability"] = 0.0

        try:
            out[method]["Plausibility"] = float(
                active_ufc.implausibility(cfdf.copy(), testdf.copy(), xtrain.copy(), len(cfdf), 0, method_name=method)
            )
        except Exception:
            out[method]["Plausibility"] = 0.0

        try:
            feas, _feas_df = active_ufc.feasibility(
                testdf.copy(),
                cfdf.copy(),
                xtrain.copy(),
                features,
                f2change,
                model,
                desired_outcome,
                uf,
                0,
                method="other",
            )
            out[method]["Feasibility"] = float(feas)
        except Exception:
            out[method]["Feasibility"] = 0.0

        try:
            if len(cfdf) == 0:
                out[method]["Validity"] = 0.0
            else:
                pred_input = cfdf.loc[:, features] if all(col in cfdf.columns for col in features) else cfdf
                preds = np.asarray(model.predict(pred_input)).reshape(-1)
                out[method]["Validity"] = float(np.sum(preds == int(desired_outcome)))
        except Exception:
            out[method]["Validity"] = 0.0
    return out


def run_one_fold(
    *,
    dataset: str,
    fold_path: str,
    x_all: pd.DataFrame,
    xtrain: pd.DataFrame,
    data_lab1: pd.DataFrame,
    features: List[str],
    catf: List[str],
    numf: List[str],
    uf: Dict[str, float],
    f2change: List[str],
    protectf: List[str],
    model: Any,
    desired_outcome: float,
    mi_fp: List[List[str]],
    no_cf: int,
    step: Dict[str, float],
    distance_scaler: Optional[Dict[str, object]],
    max_rows: Optional[int],
) -> FoldResult:
    fold_name = os.path.basename(fold_path)
    seed = int(hashlib.sha256(f"{dataset}:{fold_name}:core_author_fixed".encode("utf-8")).hexdigest()[:8], 16)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    fold_df = pd.read_csv(fold_path).loc[:, features].copy()
    if max_rows is not None and max_rows > 0:
        fold_df = fold_df.head(int(max_rows)).copy()
    if distance_scaler is not None:
        distance_X_test = apply_distance_scaler(fold_df, distance_scaler)
        distance_data_lab1 = apply_distance_scaler(data_lab1.loc[:, features].copy(), distance_scaler)
    else:
        distance_X_test = None
        distance_data_lab1 = None

    onecfs, t1, idx1 = cfmethods.sfexp(
        x_all,
        data_lab1.loc[:, features],
        fold_df,
        uf,
        step,
        f2change,
        numf,
        catf,
        model,
        desired_outcome,
        no_cf,
        features,
        distance_data_lab1=distance_data_lab1,
        distance_X_test=distance_X_test,
        distance_scaler=distance_scaler,
        plausibility_data=xtrain.loc[:, features],
    )
    twocfs, t2, idx2 = cfmethods.dfexp(
        x_all,
        data_lab1.loc[:, features],
        fold_df,
        uf,
        mi_fp[:5],
        numf,
        catf,
        f2change,
        protectf,
        model,
        desired_outcome,
        no_cf,
        features,
        distance_data_lab1=distance_data_lab1,
        distance_X_test=distance_X_test,
        distance_scaler=distance_scaler,
        plausibility_data=xtrain.loc[:, features],
        step=step,
    )
    threecfs, t3, idx3 = cfmethods.tfexp(
        x_all,
        data_lab1.loc[:, features],
        fold_df,
        uf,
        mi_fp[:5],
        numf,
        catf,
        f2change,
        protectf,
        model,
        desired_outcome,
        no_cf,
        features,
        distance_data_lab1=distance_data_lab1,
        distance_X_test=distance_X_test,
        distance_scaler=distance_scaler,
        plausibility_data=xtrain.loc[:, features],
        step=step,
    )

    payloads = {
        "UFCE1": (onecfs.reset_index(drop=True), fold_df.iloc[idx1].reset_index(drop=True)),
        "UFCE2": (twocfs.reset_index(drop=True), fold_df.iloc[idx2].reset_index(drop=True)),
        "UFCE3": (threecfs.reset_index(drop=True), fold_df.iloc[idx3].reset_index(drop=True)),
    }
    means = evaluate_metrics(
        active_ufc=cfmethods.ufc,
        method_payloads=payloads,
        catf=catf,
        numf=numf,
        features=features,
        f2change=f2change,
        uf=uf,
        xtrain=xtrain.loc[:, features],
        model=model,
        desired_outcome=desired_outcome,
        distance_scaler=distance_scaler,
    )
    return FoldResult(fold_name=fold_name, means=means, times={"UFCE1": t1, "UFCE2": t2, "UFCE3": t3})


def aggregate_results(results: Sequence[FoldResult]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    mean_df = pd.DataFrame(index=METRICS, columns=METHODS, dtype=float)
    std_df = pd.DataFrame(index=METRICS, columns=METHODS, dtype=float)
    for metric in METRICS:
        for method in METHODS:
            vals = [result.means[method].get(metric, float("nan")) for result in results]
            arr = np.asarray(vals, dtype=float)
            mean_df.loc[metric, method] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")
            std_df.loc[metric, method] = float(np.nanstd(arr)) if np.any(np.isfinite(arr)) else float("nan")
    time_mean = {
        method: float(np.mean([result.times.get(method, 0.0) for result in results])) if results else 0.0
        for method in METHODS
    }
    return mean_df, std_df, time_mean


def build_delta_rows(dataset: str, mean_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in METHODS:
        for metric in METRICS:
            author_value = AUTHOR_TABLE7[dataset][method].get(metric, float("nan"))
            reproduced_value = mean_df.loc[metric, method]
            author_f = finite_float(author_value)
            reproduced_f = finite_float(reproduced_value)
            if author_f is None or reproduced_f is None:
                delta = abs_delta = rel_delta = None
            else:
                delta = float(reproduced_f - author_f)
                abs_delta = abs(delta)
                rel_delta = abs_delta / max(abs(author_f), 1.0)
            rows.append(
                {
                    "dataset": dataset,
                    "ufce_variant": method,
                    "metric_name": metric,
                    "author_value": author_f,
                    "reproduced_value": reproduced_f,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "relative_delta": rel_delta,
                }
            )
    return rows


def plot_metrics(mean_df: pd.DataFrame, out_path: str, dataset: str) -> None:
    if plt is None:
        with open(out_path + ".skipped.txt", "w", encoding="utf-8") as handle:
            handle.write("matplotlib is unavailable; plot skipped.\n")
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, metric in zip(axes.ravel(), METRICS):
        vals = [mean_df.loc[metric, method] for method in METHODS]
        ax.bar(METHODS, vals, color=["#4477aa", "#66c2a5", "#fc8d62"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"core_author_fixed UFCE Table 7 comparison - {dataset}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_for_dataset(dataset: str, args: argparse.Namespace, run_id: str) -> Dict[str, object]:
    cfg = DEFAULT_RUNTIME[dataset]
    radius = int(args.radius if args.radius is not None else cfg["radius"])
    n_neighbors = int(args.n_neighbors if args.n_neighbors is not None else cfg["n_neighbors"])
    print(
        f"[CFG] dataset={dataset} radius={radius} n_neighbors={n_neighbors} "
        f"actionability_threshold={args.actionability_threshold} no_cf={args.no_cf} "
        f"constraint_profile={args.constraint_profile}"
    )

    datasetdf = pd.read_csv(os.path.join(args.data_dir, f"{dataset}.csv"))
    out = classify_dataset_getModel(datasetdf, data_name=dataset)
    if len(out) == 8:
        lr, lr_mean, lr_std, _xtest, xtrain, x_all, _y, datasetdf = out
    elif len(out) == 9:
        lr, lr_mean, lr_std, _xtest, xtrain, x_all, _y, datasetdf, _scaler = out
    else:
        raise ValueError(f"Unexpected classify_dataset_getModel return length: {len(out)}")

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
    ) = get_dataset_constraints(dataset, datasetdf)
    constraint_profile = resolve_constraint_profile(
        dataset,
        features,
        uf,
        f2change,
        STEP_CONFIG[dataset],
        args.constraint_profile,
    )
    uf = constraint_profile.uf
    f2change = constraint_profile.f2change
    step = constraint_profile.step

    cfmethods.initUFCE(
        radius=radius,
        n_neighbors=n_neighbors,
        contprox_metric=args.contprox_metric,
        actionability_threshold=args.actionability_threshold,
        atol=args.atol,
    )
    eval_module.ufc = cfmethods.ufc

    mi_fp = normalize_mi_feature_pairs(cfmethods.ufc.get_top_MI_features(x_all, features), features)
    distance_scaler = build_mad_distance_scaler(xtrain.loc[:, features], features, numf)

    fold_root = os.path.join(args.folds_dir, dataset, "totest")
    fold_paths = sorted(glob.glob(os.path.join(fold_root, "*.csv")))
    if args.fold_file:
        target = os.path.basename(args.fold_file)
        fold_paths = [path for path in fold_paths if os.path.basename(path) == target]
    if args.max_folds > 0:
        fold_paths = fold_paths[: args.max_folds]
    if not fold_paths:
        raise FileNotFoundError(f"No fold files found in {fold_root}")

    print(f"[INFO] dataset={dataset} folds={len(fold_paths)} cv={lr_mean:.4f}+/-{lr_std:.4f}")
    fold_results: List[FoldResult] = []
    start = time.perf_counter()
    for i, fold_path in enumerate(fold_paths):
        print(f"  - Fold {i}: {os.path.basename(fold_path)}")
        fold_results.append(
            run_one_fold(
                dataset=dataset,
                fold_path=fold_path,
                x_all=x_all.loc[:, features],
                xtrain=xtrain.loc[:, features],
                data_lab1=data_lab1.loc[:, features],
                features=features,
                catf=catf,
                numf=numf,
                uf=uf,
                f2change=f2change,
                protectf=protectf,
                model=lr,
                desired_outcome=desired_outcome,
                mi_fp=mi_fp,
                no_cf=args.no_cf,
                step=step,
                distance_scaler=distance_scaler,
                max_rows=args.max_rows,
            )
        )

    mean_df, std_df, time_mean = aggregate_results(fold_results)
    delta_rows = build_delta_rows(dataset, mean_df)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, f"summary_{dataset}_{run_id}.csv")
    std_csv = os.path.join(args.out_dir, f"summary_std_{dataset}_{run_id}.csv")
    delta_csv = os.path.join(args.out_dir, f"table7_delta_{dataset}_{run_id}.csv")
    plot_path = os.path.join(args.out_dir, f"metrics_2x3_{dataset}_{run_id}.png")
    mean_df.to_csv(summary_csv)
    std_df.to_csv(std_csv)
    pd.DataFrame(delta_rows).to_csv(delta_csv, index=False)
    plot_metrics(mean_df, plot_path, dataset)

    print("\n[RESULT] reproduced means")
    print(mean_df.to_string())
    print("\n[RESULT] Table 7 deltas")
    print(pd.DataFrame(delta_rows).to_string(index=False))

    return {
        "dataset": dataset,
        "status": "ok",
        "runtime_sec": float(time.perf_counter() - start),
        "n_folds": int(len(fold_paths)),
        "error": "",
        "summary_csv": summary_csv,
        "std_csv": std_csv,
        "delta_csv": delta_csv,
        "plot_path": plot_path,
        "time_mean": json.dumps(time_mean, sort_keys=True),
        "constraint_profile": args.constraint_profile,
        "not_main_table7": bool(constraint_profile.not_main_table7),
        "effective_uf_keys": json.dumps(list(uf.keys())),
        "effective_f2change": json.dumps(f2change),
        "inactive_features": json.dumps(constraint_profile.inactive_features),
    }


def write_manifest(args: argparse.Namespace, run_id: str) -> str:
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"run_manifest_{run_id}.json")
    payload = {
        "script": "scripts/final/part1/01c_reproduce_core_author_fixed_ufce_only.py",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "default_runtime": DEFAULT_RUNTIME,
        "core_package": "ufce.core_author_fixed",
        "strict_paper": True,
        "constraint_profile": args.constraint_profile,
        "not_main_table7": args.constraint_profile != AUTHOR_PUBLIC,
        "distance_scaler": "mad",
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bank", choices=[*ALL_DATASETS, "all"])
    parser.add_argument("--data_dir", default=os.path.join("ufce", "data"))
    parser.add_argument("--folds_dir", default=os.path.join("ufce", "data", "folds"))
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", default=os.path.join("outputs", "final", "part1", "core_author_fixed_ufce_only"))
    parser.add_argument("--no_cf", type=int, default=10)
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0, help="Optional row limit per fold for smoke tests.")
    parser.add_argument("--fold_file", default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--n_neighbors", type=int, default=None)
    parser.add_argument("--actionability_threshold", type=float, default=0.30)
    parser.add_argument("--contprox_metric", default="mad")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--constraint_profile", default=AUTHOR_PUBLIC, choices=CONSTRAINT_PROFILES)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print_env_versions()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = write_manifest(args, run_id)
    print(f"[INFO] manifest={manifest}")

    if args.dataset == "all" and args.constraint_profile == DOMAIN_ACTIONABLE:
        datasets = list(DOMAIN_ACTIONABLE_CHANGED_DATASETS)
    else:
        datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    records: List[Dict[str, object]] = []
    for dataset in datasets:
        print(f"\n==================== DATASET: {dataset} ====================")
        try:
            records.append(run_for_dataset(dataset, args, run_id))
        except Exception as exc:
            record = {
                "dataset": dataset,
                "status": "failed",
                "runtime_sec": 0.0,
                "n_folds": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "summary_csv": "",
                "std_csv": "",
                "delta_csv": "",
                "plot_path": "",
                "time_mean": "{}",
            }
            records.append(record)
            print(f"[ERROR] dataset={dataset} failed: {record['error']}")

    batch_csv = os.path.join(args.out_dir, f"batch_summary_{run_id}.csv")
    pd.DataFrame(records).to_csv(batch_csv, index=False)
    print("\n==================== BATCH SUMMARY ====================")
    print(pd.DataFrame(records).to_string(index=False))
    print(f"[INFO] batch_summary={batch_csv}")

    if any(record["status"] != "ok" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
