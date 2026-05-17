#!/usr/bin/env python3
"""
Author-aligned UFCE-only reproduction runner.

- Keeps only UFCE1/UFCE2/UFCE3.
- Uses the current ufce/core generation API.
- Uses ufce/core/evaluations.py directly for metric computation.
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- PYTHONPATH bootstrap ----
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CORE_ROOT = os.path.join(ROOT, "ufce", "core")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if CORE_ROOT not in sys.path:
    sys.path.append(CORE_ROOT)
# ------------------------------

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

import ufce
from ufce import UFCE
from ufce.core import cfmethods
from ufce.core import evaluations as eval_module
from ufce.core.data_processing import (
    classify_dataset_getModel,
    get_bank_user_constraints,
    get_bupa_user_constraints,
    get_grad_user_constraints,
    get_movie_user_constraints,
    get_wine_user_constraints,
)

ufc = UFCE()
cfmethods.ufc = ufc
eval_module.ufc = ufc

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

ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]


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
    for name, value in pkgs:
        print(f"- {name:12s}: {value}")
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
            "Production expense": 3,
            "Num_multiplex": 3,
            "Multiplex coverage": 0.2,
            "Movie_length": 5,
            "Lead_ Actor_Rating": 1.0,
            "Lead_Actress_rating": 1.0,
            "Director_rating": 1.0,
            "Producer_rating": 1.0,
            "Genre": 1,
            "Collection": 500,
            "Budget": 3000,
        }
    return {}


def _normalize_col_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _resolve_column(df: pd.DataFrame, target: str) -> Optional[str]:
    if target in df.columns:
        return target
    target_key = _normalize_col_key(target)
    for col in df.columns:
        if _normalize_col_key(col) == target_key:
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
        raise KeyError(f"Missing model features: {missing}")
    out = df[resolved_cols].copy()
    out.columns = features
    return out


def _empty_frame(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


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
    top = abs_corr.stack().sort_values(ascending=False).head(10)

    print("- Top-10 |Pearson corr| pairs (numeric-only):")
    printed = 0
    for (left, right), value in top.items():
        if left < right:
            print(f"  {left} <-> {right}: {value:.2f}")
            printed += 1
        if printed >= 10:
            break
    print("=========================\n")


def evaluate_ufce_only_metrics(
    *,
    onecfs: pd.DataFrame,
    onetest: pd.DataFrame,
    twocfs: pd.DataFrame,
    twotest: pd.DataFrame,
    threecfs: pd.DataFrame,
    threetest: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: List[str],
    catf: List[str],
    numf: List[str],
    f2change: List[str],
    uf: Dict[str, float],
    bb_model,
    desired_outcome: float,
    contprox_method: str,
) -> Dict[str, Dict[str, float]]:
    empty = _empty_frame(features)
    means: Dict[str, Dict[str, float]] = {method: {} for method in METHODS}

    if len(catf) == 0:
        cat_means = [float("nan")] * 6
    else:
        cat_means, _ = eval_module.Catproximity(
            onecfs.copy(),
            onetest.copy(),
            twocfs.copy(),
            twotest.copy(),
            threecfs.copy(),
            threetest.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            catf,
        )

    cont_means, _ = eval_module.Contproximity(
        onecfs.copy(),
        onetest.copy(),
        twocfs.copy(),
        twotest.copy(),
        threecfs.copy(),
        threetest.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        numf,
        method=contprox_method,
        X=xtrain.copy(),
    )
    sparsity_means, _ = eval_module.Sparsity(
        onecfs.copy(),
        onetest.copy(),
        twocfs.copy(),
        twotest.copy(),
        threecfs.copy(),
        threetest.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        numf,
    )
    action_means, _ = eval_module.Actionability(
        onecfs.copy(),
        onetest.copy(),
        twocfs.copy(),
        twotest.copy(),
        threecfs.copy(),
        threetest.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        features,
        f2change,
        uf,
    )
    plaus_means, _ = eval_module.Plausibility(
        onecfs.copy(),
        onetest.copy(),
        twocfs.copy(),
        twotest.copy(),
        threecfs.copy(),
        threetest.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        xtrain.copy(),
    )
    feas_means, _ = eval_module.Feasibility(
        onecfs.copy(),
        onetest.copy(),
        twocfs.copy(),
        twotest.copy(),
        threecfs.copy(),
        threetest.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        empty.copy(),
        xtrain.copy(),
        features,
        f2change,
        bb_model,
        desired_outcome,
        uf,
    )

    metric_values = {
        "Prox-Jac": cat_means,
        "Prox-Euc": cont_means,
        "Sparsity": sparsity_means,
        "Actionability": action_means,
        "Plausibility": plaus_means,
        "Feasibility": feas_means,
    }
    for method_idx, method in enumerate(METHODS):
        for metric_name in METRICS:
            means[method][metric_name] = float(metric_values[metric_name][method_idx])
    return means


def run_one_fold(
    *,
    fold_df: pd.DataFrame,
    x_all: pd.DataFrame,
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
    contprox_method: str,
) -> FoldResult:
    fold_features = _align_feature_frame(fold_df, features)
    onecfs, t1, idx1 = cfmethods.sfexp(
        x_all.copy(),
        data_lab1.copy(),
        fold_features.copy(),
        uf,
        step,
        f2change,
        numf,
        catf,
        bb_model,
        desired_outcome,
        no_cf,
        features,
    )
    twocfs, t2, idx2 = cfmethods.dfexp(
        x_all.copy(),
        data_lab1.copy(),
        fold_features.copy(),
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
    )
    threecfs, t3, idx3 = cfmethods.tfexp(
        x_all.copy(),
        data_lab1.copy(),
        fold_features.copy(),
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
    )

    onetest = fold_features.iloc[idx1].reset_index(drop=True)
    twotest = fold_features.iloc[idx2].reset_index(drop=True)
    threetest = fold_features.iloc[idx3].reset_index(drop=True)

    means = evaluate_ufce_only_metrics(
        onecfs=onecfs.reset_index(drop=True),
        onetest=onetest,
        twocfs=twocfs.reset_index(drop=True),
        twotest=twotest,
        threecfs=threecfs.reset_index(drop=True),
        threetest=threetest,
        xtrain=xtrain,
        features=features,
        catf=catf,
        numf=numf,
        f2change=f2change,
        uf=uf,
        bb_model=bb_model,
        desired_outcome=desired_outcome,
        contprox_method=contprox_method,
    )
    times = {"UFCE1": float(t1), "UFCE2": float(t2), "UFCE3": float(t3)}
    return FoldResult(fold_name=fold_name, means=means, times=times)


def aggregate_results(folds: List[FoldResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rows = []
    time_rows = []
    for fold in folds:
        for method in METHODS:
            row = {"fold": fold.fold_name, "method": method}
            row.update(fold.means[method])
            rows.append(row)
        time_rows.append(fold.times)
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
    for method in METHODS:
        print(f"- {method}: {time_mean[method]:0.2f}s")
    print("==================================\n")


def build_author_table(dataset: str) -> Optional[pd.DataFrame]:
    author = AUTHOR_TABLE7.get(dataset)
    if author is None:
        return None
    return pd.DataFrame({method: {metric: author[method][metric] for metric in METRICS} for method in METHODS})


def print_author_table(dataset: str) -> Optional[pd.DataFrame]:
    author_df = build_author_table(dataset)
    if author_df is None:
        print(f"[WARN] No AUTHOR_TABLE7 values registered for dataset='{dataset}'. Skipping author table.\n")
        return None

    print("=== Authors (Table 7 Reference) ===")
    print(author_df.to_string(float_format=lambda x: f"{x:0.6f}"))
    print("===================================\n")
    return author_df


def print_full_comparison_table(dataset: str, mean_df: pd.DataFrame, std_df: pd.DataFrame) -> None:
    author_df = build_author_table(dataset)
    if author_df is None:
        print(f"[WARN] No AUTHOR_TABLE7 values registered for dataset='{dataset}'. Skipping comparison table.\n")
        return

    print("=== Full Comparison: Authors vs Ours ===")
    rows = []
    for method in METHODS:
        for metric in METRICS:
            authors_value = float(author_df.loc[metric, method])
            ours_value = float(mean_df.loc[metric, method])
            ours_std = float(std_df.loc[metric, method]) if pd.notna(std_df.loc[metric, method]) else float("nan")
            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Authors": authors_value,
                    "Ours Mean": ours_value,
                    "Ours Std": ours_std,
                    "Delta(Ours-Authors)": ours_value - authors_value,
                }
            )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print("==============================================\n")


def run_for_dataset(dataset: str, args) -> Dict[str, object]:
    t0 = time.time()
    data_path = os.path.join(args.data_dir, f"{dataset}.csv")
    datasetdf = pd.read_csv(data_path)
    lr, lr_mean, lr_std, _xtest, xtrain, x_all, _y, datasetdf = classify_dataset_getModel(datasetdf, data_name=dataset)

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

    x_all = _align_feature_frame(x_all, features)
    xtrain = _align_feature_frame(xtrain, features)
    data_lab1 = _align_feature_frame(data_lab1, features)
    step = get_step_config(dataset)
    mi_fp = ufc.get_top_MI_features(x_all, features)

    print(f"Dataset: {dataset}")
    print("Black-box: LogisticRegression (author pipeline)")
    print(f"CV accuracy: {lr_mean:.6f} +/- {lr_std:.6f}")
    print(f"Cont-Prox method: {args.contprox_method}")
    print_feature_relations(x_all, features, mi_fp)

    testfold_path = os.path.join(args.folds_dir, dataset, "totest")
    testfolds = sorted(glob.glob(os.path.join(testfold_path, "*.csv")))
    if not testfolds:
        raise FileNotFoundError(f"No test folds found in {testfold_path}")
    if args.fold_file:
        target_name = os.path.basename(str(args.fold_file))
        testfolds = [path for path in testfolds if os.path.basename(path) == target_name]
        if not testfolds:
            raise FileNotFoundError(f"Requested fold_file='{target_name}' was not found in {testfold_path}")
    if args.max_folds > 0:
        testfolds = testfolds[: args.max_folds]

    print(f"Found {len(testfolds)} folds. Processing...")
    fold_results: List[FoldResult] = []
    for fold_idx, fold_path in enumerate(testfolds):
        fold_name = os.path.basename(fold_path)
        print(f"\n--- Fold {fold_idx}: {fold_name} ---")
        fold_df = pd.read_csv(fold_path)
        fold_result = run_one_fold(
            fold_df=fold_df,
            x_all=x_all,
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
            contprox_method=args.contprox_method,
        )
        print("Times (sec): " + ", ".join([f"{method}={fold_result.times[method]:0.2f}" for method in METHODS]))
        fold_results.append(fold_result)

    mean_df, std_df, time_mean = aggregate_results(fold_results)
    print_ours_table(mean_df, std_df, time_mean)
    print_author_table(dataset)
    print_full_comparison_table(dataset, mean_df, std_df)

    runtime_sec = float(time.time() - t0)
    print("\nExecution Complete.")
    print(f"Total runtime: {runtime_sec:0.2f}s")

    return {
        "dataset": dataset,
        "status": "ok",
        "runtime_sec": runtime_sec,
        "n_folds": int(len(testfolds)),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bank", choices=["bank", "grad", "wine", "bupa", "movie", "all"])
    parser.add_argument("--data_dir", type=str, default=os.path.join("ufce", "data"))
    parser.add_argument("--folds_dir", type=str, default=os.path.join("ufce", "data", "folds"))
    parser.add_argument("--no_cf", type=int, default=10, help="Number of CFs requested per instance")
    parser.add_argument("--max_folds", type=int, default=0, help="0=all folds; otherwise limit folds")
    parser.add_argument(
        "--contprox_method",
        type=str,
        default="mad",
        choices=["eucl", "mad"],
        help="Continuous proximity method for evaluation",
    )
    parser.add_argument(
        "--fold_file",
        type=str,
        default=None,
        help="Optional exact fold basename to run (e.g., testfold_0_pred_0.csv)",
    )
    args = parser.parse_args()

    print_env_versions()

    if args.dataset != "all":
        run_for_dataset(args.dataset, args)
        return

    batch_records: List[Dict[str, object]] = []
    batch_t0 = time.time()
    for dataset in ALL_DATASETS:
        ds_t0 = time.time()
        print(f"\n==================== BATCH DATASET: {dataset} ====================")
        try:
            record = run_for_dataset(dataset, args)
        except Exception as exc:
            record = {
                "dataset": dataset,
                "status": "failed",
                "runtime_sec": float(time.time() - ds_t0),
                "n_folds": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[ERROR] dataset={dataset} failed with {type(exc).__name__}: {exc}")
        batch_records.append(record)

    batch_df = pd.DataFrame(
        batch_records,
        columns=["dataset", "status", "runtime_sec", "n_folds", "error"],
    )

    print("\n==================== BATCH SUMMARY ====================")
    print(batch_df.to_string(index=False))
    print(f"- Batch total runtime: {time.time() - batch_t0:0.2f}s")

    failed_count = int(np.sum(batch_df["status"].values == "failed"))
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
