#!/usr/bin/env python3
"""
Run the author/GitHub UFCE core as a raw post-hoc baseline.

This runner intentionally does not use the patched `ufce.core` generation path.
It loads `ufce/core_author/ufce.py` and `ufce/core_author/cfmethods.py`,
executes UFCE1/2/3 on the existing thesis folds, then checks validity only after
the selected counterfactual has been returned.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


ROOT = Path(__file__).resolve().parents[3]
RUNNER_01B = ROOT / "scripts" / "final" / "part1" / "ufce_only_reproduction.py"
AUTHOR_DIR = ROOT / "ufce" / "core_author"
OUT_PARENT = ROOT / "outputs" / "final" / "part1"
ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]
METRIC_DIRECTIONS = {
    "Prox-Jac": "lower_is_better",
    "Prox-Euc": "lower_is_better",
    "Sparsity": "lower_is_better",
    "Actionability": "higher_is_better",
    "Plausibility": "higher_is_better",
    "Feasibility": "higher_is_better",
}


class _DummyTqdm:
    def __init__(self, iterable=None, **_kwargs):
        self._iterable = iterable

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def update(self, _n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def tqdm_wrap(iterable=None, *, total=None, desc=None, leave=False, disable=False):
    if _tqdm is None:
        return _DummyTqdm(iterable=iterable)
    return _tqdm(iterable=iterable, total=total, desc=desc, leave=leave, disable=disable)


@dataclass
class MethodOutput:
    cfdf: pd.DataFrame
    found_idx: List[int]
    runtime_ms: float


def now_tag() -> str:
    return datetime.now().strftime("author_raw_posthoc_replay_%Y%m%d_%H%M%S")


def parse_dataset_arg(value: str) -> List[str]:
    text = str(value).strip().lower()
    if text == "all":
        return list(ALL_DATASETS)
    datasets = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = [item for item in datasets if item not in ALL_DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    if not datasets:
        raise ValueError("At least one dataset is required.")
    return datasets


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def install_optional_import_stubs() -> None:
    """Author files import optional packages in paths we do not exercise here."""
    module_names = [
        "dice_ml",
        "dice_ml.utils",
        "dice_ml.utils.helpers",
        "recourse",
        "simplenlg",
        "simplenlg.framework",
        "simplenlg.lexicon",
        "simplenlg.realiser",
        "simplenlg.realiser.english",
        "simplenlg.phrasespec",
        "simplenlg.features",
    ]
    for name in module_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["dice_ml.utils"].helpers = sys.modules["dice_ml.utils.helpers"]


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runner_01b():
    install_optional_import_stubs()
    return load_module_from_path("ufce_runner_01b_for_author_raw", RUNNER_01B)


def load_author_modules():
    install_optional_import_stubs()
    author_path = str(AUTHOR_DIR)
    if author_path not in sys.path:
        sys.path.insert(0, author_path)

    author_ufce = load_module_from_path("ufce_author_raw_ufce", AUTHOR_DIR / "ufce.py")

    original_ufce_module = sys.modules.get("ufce")
    shim = types.ModuleType("ufce")
    shim.UFCE = author_ufce.UFCE
    sys.modules["ufce"] = shim
    try:
        author_cfmethods = load_module_from_path("ufce_author_raw_cfmethods", AUTHOR_DIR / "cfmethods.py")
    finally:
        if original_ufce_module is not None:
            sys.modules["ufce"] = original_ufce_module
        else:
            sys.modules.pop("ufce", None)

    author_cfmethods.ufc = author_ufce.UFCE()
    return SimpleNamespace(ufce=author_ufce, cfmethods=author_cfmethods)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_07_dir() -> Optional[Path]:
    candidates = sorted(
        OUT_PARENT.glob("07_ufce_ff_*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def predict_label(model, row: pd.DataFrame, features: Sequence[str]) -> Optional[int]:
    if row is None or row.empty:
        return None
    pred_input = row.loc[:, [c for c in features if c in row.columns]]
    if pred_input.empty:
        return None
    pred = np.asarray(model.predict(pred_input)).reshape(-1)
    if pred.size == 0:
        return None
    return int(pred[0])


def scalar(value: Any) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def mean_or_nan(values: List[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan")
    return float(np.mean(np.asarray(clean, dtype=float)))


def empty_feature_frame(features: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(features))


def aligned_cf_and_test(
    *,
    method_output: MethodOutput,
    fold_df: pd.DataFrame,
    features: Sequence[str],
    bb_model,
    desired_outcome: float,
    valid_only: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    cfdf = method_output.cfdf.reset_index(drop=True) if isinstance(method_output.cfdf, pd.DataFrame) else pd.DataFrame()
    found_idx = [int(v) for v in method_output.found_idx]
    usable = min(len(found_idx), len(cfdf))
    selected_count = int(usable)
    valid_count = 0
    cf_rows: List[pd.DataFrame] = []
    test_rows: List[pd.DataFrame] = []

    for out_pos in range(usable):
        query_pos = int(found_idx[out_pos])
        if query_pos < 0 or query_pos >= len(fold_df):
            continue
        cf_row = cfdf.iloc[[out_pos]].loc[:, [c for c in features if c in cfdf.columns]].reset_index(drop=True)
        pred_label = predict_label(bb_model, cf_row, features)
        is_valid = bool(pred_label == int(desired_outcome)) if pred_label is not None else False
        valid_count += int(is_valid)
        if valid_only and not is_valid:
            continue
        cf_rows.append(cf_row)
        test_rows.append(fold_df.iloc[[query_pos]].loc[:, list(features)].reset_index(drop=True))

    if cf_rows:
        cf_out = pd.concat(cf_rows, ignore_index=True, axis=0)
        test_out = pd.concat(test_rows, ignore_index=True, axis=0)
    else:
        cf_out = empty_feature_frame(features)
        test_out = empty_feature_frame(features)
    return cf_out, test_out, selected_count, valid_count


def safe_implausibility_count(author_ufc, cfdf: pd.DataFrame, testdf: pd.DataFrame, xtrain: pd.DataFrame) -> float:
    try:
        return float(author_ufc.implausibility(cfdf.copy(), testdf.copy(), xtrain.copy(), 0, 0))
    except Exception:
        return 0.0


def safe_feasibility_count(
    author_ufc,
    cfdf: pd.DataFrame,
    testdf: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: List[str],
    f2change: List[str],
    bb_model,
    desired_outcome: float,
    uf: Dict[str, float],
    eval_method: str,
) -> float:
    try:
        out = author_ufc.feasibility(
            testdf.copy(),
            cfdf.copy(),
            xtrain.copy(),
            features,
            f2change,
            bb_model,
            desired_outcome,
            uf,
            0,
            method=str(eval_method),
        )
        if isinstance(out, tuple) and len(out) >= 1:
            return float(out[0])
        if isinstance(out, (int, float)):
            return float(out)
    except Exception:
        return 0.0
    return 0.0


def compute_six_metrics(
    *,
    author_ufc,
    mod01b,
    method_name: str,
    dataset: str,
    cfdf: pd.DataFrame,
    testdf: pd.DataFrame,
    catf: List[str],
    numf: List[str],
    features: List[str],
    f2change: List[str],
    uf: Dict[str, float],
    xtrain: pd.DataFrame,
    bb_model,
    desired_outcome: float,
    movie_distance_scaler: Optional[Dict[str, object]],
) -> Dict[str, float]:
    cfdf = cfdf.reset_index(drop=True).loc[:, [c for c in features if c in cfdf.columns]].copy()
    testdf = testdf.reset_index(drop=True).loc[:, [c for c in features if c in testdf.columns]].copy()
    n = min(len(cfdf), len(testdf))
    if n == 0:
        return {
            "Prox-Jac": float("nan"),
            "Prox-Euc": float("nan"),
            "Sparsity": float("nan"),
            "Actionability": 0.0,
            "Plausibility": 0.0,
            "Feasibility": 0.0,
        }
    cfdf = cfdf.iloc[:n].reset_index(drop=True)
    testdf = testdf.iloc[:n].reset_index(drop=True)
    eval_method = "ufc" if str(method_name).upper().startswith("UFCE") else "other"

    if len(catf) == 0:
        prox_jac = float("nan")
    else:
        vals = [
            scalar(author_ufc.categorical_distance(testdf[i : i + 1], cfdf[i : i + 1], catf, metric="jaccard", agg=None))
            for i in range(n)
        ]
        prox_jac = mean_or_nan(vals)

    if dataset == "movie" and movie_distance_scaler is not None:
        prox_vals = mod01b.pairwise_normalized_l2_0_100_values(testdf, cfdf, numf, movie_distance_scaler)
        prox_euc = mean_or_nan([float(v) for v in prox_vals])
    else:
        vals = [
            scalar(author_ufc.continuous_distance(testdf[i : i + 1], cfdf[i : i + 1], numf, metric="euclidean", agg=None))
            for i in range(n)
        ]
        prox_euc = mean_or_nan(vals)

    try:
        sparsity_d, _ = author_ufc.sparsity_count(cfdf.copy(), testdf.copy(), numf, numf)
        sparsity_values = list(sparsity_d.values()) if isinstance(sparsity_d, dict) else []
        sparsity = mean_or_nan([float(v) for v in sparsity_values])
    except Exception:
        sparsity = float("nan")

    try:
        action_cfs, _flag, _idx, _temp = author_ufc.actionability(
            cfdf.copy(),
            testdf.copy(),
            features,
            f2change,
            0,
            uf,
            method=eval_method,
        )
        actionability = float(len(action_cfs))
    except Exception:
        actionability = 0.0

    plausibility = safe_implausibility_count(author_ufc, cfdf, testdf, xtrain)
    feasibility = safe_feasibility_count(
        author_ufc,
        cfdf,
        testdf,
        xtrain,
        features,
        f2change,
        bb_model,
        desired_outcome,
        uf,
        eval_method,
    )
    return {
        "Prox-Jac": prox_jac,
        "Prox-Euc": prox_euc,
        "Sparsity": sparsity,
        "Actionability": actionability,
        "Plausibility": plausibility,
        "Feasibility": feasibility,
    }


def selected_rows_by_query(
    *,
    method_output: MethodOutput,
    fold_df: pd.DataFrame,
    features: Sequence[str],
    bb_model,
    desired_outcome: float,
) -> List[Dict[str, Any]]:
    cfdf = method_output.cfdf.reset_index(drop=True) if isinstance(method_output.cfdf, pd.DataFrame) else pd.DataFrame()
    found_idx = [int(v) for v in method_output.found_idx]
    selected_by_pos: Dict[int, Dict[str, Any]] = {}
    usable = min(len(found_idx), len(cfdf))

    for out_pos in range(usable):
        query_pos = int(found_idx[out_pos])
        cf_row = cfdf.iloc[[out_pos]].reset_index(drop=True)
        pred_label = predict_label(bb_model, cf_row, features)
        selected_by_pos[query_pos] = {
            "selected_exists": 1,
            "selected_output_pos": int(out_pos),
            "pred_label": pred_label,
            "strict_valid_selected": int(pred_label == int(desired_outcome)) if pred_label is not None else 0,
        }

    rows: List[Dict[str, Any]] = []
    for query_pos in range(len(fold_df)):
        record = selected_by_pos.get(
            int(query_pos),
            {
                "selected_exists": 0,
                "selected_output_pos": None,
                "pred_label": None,
                "strict_valid_selected": 0,
            },
        )
        rows.append(
            {
                "query_pos": int(query_pos),
                "selected_exists": int(record["selected_exists"]),
                "selected_output_pos": record["selected_output_pos"],
                "pred_label": record["pred_label"],
                "desired_label": int(desired_outcome),
                "strict_valid_selected": int(record["strict_valid_selected"]),
            }
        )
    return rows


def summarize_query_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["dataset", "fold_id", "method"]
    fold_rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        dataset, fold_id, method = key
        query_count = int(len(group))
        selected_count = int(pd.to_numeric(group["selected_exists"], errors="coerce").fillna(0).sum())
        valid_count = int(pd.to_numeric(group["strict_valid_selected"], errors="coerce").fillna(0).sum())
        fold_rows.append(
            {
                "dataset": dataset,
                "fold_id": fold_id,
                "method": method,
                "query_count": query_count,
                "selected_count": selected_count,
                "selected_valid_count": valid_count,
                "selected_posthoc_fail_count": int(selected_count - valid_count),
                "missing_count": int(query_count - selected_count),
                "selected_return_rate": float(selected_count / query_count) if query_count else 0.0,
                "strict_valid_rate_all_queries": float(valid_count / query_count) if query_count else 0.0,
                "strict_valid_rate_among_selected": float(valid_count / selected_count) if selected_count else 0.0,
            }
        )
    fold_summary = pd.DataFrame(fold_rows)

    summary_rows = []
    for key, group in fold_summary.groupby(["dataset", "method"], dropna=False):
        dataset, method = key
        query_count = int(group["query_count"].sum())
        selected_count = int(group["selected_count"].sum())
        valid_count = int(group["selected_valid_count"].sum())
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "query_count": query_count,
                "selected_count": selected_count,
                "selected_valid_count": valid_count,
                "selected_posthoc_fail_count": int(selected_count - valid_count),
                "missing_count": int(query_count - selected_count),
                "selected_return_rate": float(selected_count / query_count) if query_count else 0.0,
                "strict_valid_rate_all_queries": float(valid_count / query_count) if query_count else 0.0,
                "strict_valid_rate_among_selected": float(valid_count / selected_count) if selected_count else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    return fold_summary, summary


def attach_runtime(fold_summary: pd.DataFrame, runtime_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if fold_summary.empty:
        return fold_summary
    runtime_df = pd.DataFrame(runtime_rows)
    if runtime_df.empty:
        fold_summary["runtime_ms_per_query"] = None
        return fold_summary
    return fold_summary.merge(runtime_df, on=["dataset", "fold_id", "method"], how="left")


def aggregate_runtime(summary: pd.DataFrame, fold_summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or fold_summary.empty or "runtime_ms_per_query" not in fold_summary.columns:
        return summary
    runtime = (
        fold_summary.groupby(["dataset", "method"], dropna=False)["runtime_ms_per_query"]
        .mean()
        .reset_index()
        .rename(columns={"runtime_ms_per_query": "mean_runtime_ms_per_query"})
    )
    return summary.merge(runtime, on=["dataset", "method"], how="left")


def summarize_metric_rows(metric_fold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric_fold_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_cols = [c for c in METRICS if c in metric_fold_df.columns]
    group_cols = ["dataset", "method", "metric_scope"]
    summary = (
        metric_fold_df.groupby(group_cols, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    count_cols = [
        "query_count",
        "selected_count",
        "selected_valid_count",
        "selected_posthoc_fail_count",
        "missing_count",
        "metric_candidate_count",
    ]
    counts = (
        metric_fold_df.groupby(group_cols, dropna=False)[[c for c in count_cols if c in metric_fold_df.columns]]
        .sum(numeric_only=True)
        .reset_index()
    )
    summary = counts.merge(summary, on=group_cols, how="left")
    if "query_count" in summary.columns:
        summary["strict_valid_rate_all_queries"] = summary.apply(
            lambda row: float(row["selected_valid_count"] / row["query_count"]) if row["query_count"] else 0.0,
            axis=1,
        )
    if "selected_count" in summary.columns:
        summary["strict_valid_rate_among_selected"] = summary.apply(
            lambda row: float(row["selected_valid_count"] / row["selected_count"]) if row["selected_count"] else 0.0,
            axis=1,
        )

    long_rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        for metric in metric_cols:
            long_rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "metric_scope": row["metric_scope"],
                    "metric": metric,
                    "metric_direction": METRIC_DIRECTIONS.get(metric, "unknown"),
                    "value": finite_float(row.get(metric)),
                    "query_count": int(row.get("query_count", 0)),
                    "selected_count": int(row.get("selected_count", 0)),
                    "selected_valid_count": int(row.get("selected_valid_count", 0)),
                    "selected_posthoc_fail_count": int(row.get("selected_posthoc_fail_count", 0)),
                    "missing_count": int(row.get("missing_count", 0)),
                    "metric_candidate_count": int(row.get("metric_candidate_count", 0)),
                    "strict_valid_rate_all_queries": finite_float(row.get("strict_valid_rate_all_queries")),
                    "strict_valid_rate_among_selected": finite_float(row.get("strict_valid_rate_among_selected")),
                }
            )
    return summary, pd.DataFrame(long_rows)


def build_raw_vs_posthoc_metric_comparison(metric_summary_long: pd.DataFrame) -> pd.DataFrame:
    if metric_summary_long.empty:
        return pd.DataFrame()
    raw = metric_summary_long.loc[metric_summary_long["metric_scope"] == "author_raw"].copy()
    posthoc = metric_summary_long.loc[metric_summary_long["metric_scope"] == "author_posthoc_valid_only"].copy()
    if raw.empty or posthoc.empty:
        return pd.DataFrame()
    raw = raw.rename(
        columns={
            "value": "author_raw_value",
            "metric_candidate_count": "author_raw_metric_candidate_count",
            "selected_count": "author_raw_selected_count",
            "selected_valid_count": "author_raw_selected_valid_count",
            "selected_posthoc_fail_count": "author_raw_selected_posthoc_fail_count",
            "strict_valid_rate_all_queries": "author_raw_strict_valid_rate_all_queries",
            "strict_valid_rate_among_selected": "author_raw_strict_valid_rate_among_selected",
        }
    )
    posthoc = posthoc.rename(
        columns={
            "value": "posthoc_valid_value",
            "metric_candidate_count": "posthoc_valid_metric_candidate_count",
            "selected_count": "posthoc_selected_count",
            "selected_valid_count": "posthoc_selected_valid_count",
            "selected_posthoc_fail_count": "posthoc_selected_posthoc_fail_count",
            "strict_valid_rate_all_queries": "posthoc_strict_valid_rate_all_queries",
            "strict_valid_rate_among_selected": "posthoc_strict_valid_rate_among_selected",
        }
    )
    keep_raw = [
        "dataset",
        "method",
        "metric",
        "metric_direction",
        "author_raw_value",
        "author_raw_metric_candidate_count",
        "author_raw_selected_count",
        "author_raw_selected_valid_count",
        "author_raw_selected_posthoc_fail_count",
        "author_raw_strict_valid_rate_all_queries",
        "author_raw_strict_valid_rate_among_selected",
    ]
    keep_posthoc = [
        "dataset",
        "method",
        "metric",
        "posthoc_valid_value",
        "posthoc_valid_metric_candidate_count",
        "posthoc_selected_count",
        "posthoc_selected_valid_count",
        "posthoc_selected_posthoc_fail_count",
        "posthoc_strict_valid_rate_all_queries",
        "posthoc_strict_valid_rate_among_selected",
    ]
    out = raw.loc[:, [c for c in keep_raw if c in raw.columns]].merge(
        posthoc.loc[:, [c for c in keep_posthoc if c in posthoc.columns]],
        on=["dataset", "method", "metric"],
        how="outer",
    )
    out["raw_to_posthoc_raw_delta"] = out.apply(
        lambda row: float(row["posthoc_valid_value"] - row["author_raw_value"])
        if finite_float(row.get("author_raw_value")) is not None and finite_float(row.get("posthoc_valid_value")) is not None
        else None,
        axis=1,
    )

    def directional_delta(row: pd.Series) -> Optional[float]:
        raw_value = finite_float(row.get("author_raw_value"))
        post_value = finite_float(row.get("posthoc_valid_value"))
        if raw_value is None or post_value is None:
            return None
        if row.get("metric_direction") == "lower_is_better":
            return float(raw_value - post_value)
        if row.get("metric_direction") == "higher_is_better":
            return float(post_value - raw_value)
        return None

    out["raw_to_posthoc_directional_delta"] = out.apply(directional_delta, axis=1)
    out["raw_to_posthoc_verdict"] = out["raw_to_posthoc_directional_delta"].apply(
        lambda value: "na"
        if finite_float(value) is None
        else ("improved" if float(value) > 1e-12 else ("regressed" if float(value) < -1e-12 else "same"))
    )
    return out


def build_forceflip_comparison(author_summary: pd.DataFrame, compare_dir: Optional[Path]) -> pd.DataFrame:
    if compare_dir is None:
        return pd.DataFrame()
    selector_path = compare_dir / "ff_selector_comparison.csv"
    if not selector_path.exists() or author_summary.empty:
        return pd.DataFrame()

    ff = pd.read_csv(selector_path)
    ff = ff.loc[ff["mode"] == "public_forceflip"].copy()
    if ff.empty:
        return pd.DataFrame()
    ff = ff.rename(
        columns={
            "query_count": "forceflip_query_count",
            "public_selected_flip_count": "forceflip_valid_count",
            "strict_valid_rate": "forceflip_strict_valid_rate",
            "raw_candidate_count": "forceflip_raw_candidate_count",
            "flip_candidate_count": "forceflip_flip_candidate_count",
        }
    )
    keep = [
        "dataset",
        "method",
        "forceflip_query_count",
        "forceflip_valid_count",
        "forceflip_strict_valid_rate",
        "forceflip_raw_candidate_count",
        "forceflip_flip_candidate_count",
    ]
    ff = ff.loc[:, [c for c in keep if c in ff.columns]]

    auth = author_summary.rename(
        columns={
            "query_count": "author_query_count",
            "selected_count": "author_selected_count",
            "selected_valid_count": "author_valid_count",
            "selected_posthoc_fail_count": "author_posthoc_fail_count",
            "strict_valid_rate_all_queries": "author_strict_valid_rate",
            "strict_valid_rate_among_selected": "author_valid_rate_among_selected",
        }
    )
    out = auth.merge(ff, on=["dataset", "method"], how="left")
    out["forceflip_vs_author_valid_rate_delta"] = out.apply(
        lambda row: float(row["forceflip_strict_valid_rate"] - row["author_strict_valid_rate"])
        if finite_float(row.get("forceflip_strict_valid_rate")) is not None
        else None,
        axis=1,
    )
    out["forceflip_vs_author_valid_count_delta"] = out.apply(
        lambda row: int(row["forceflip_valid_count"] - row["author_valid_count"])
        if finite_float(row.get("forceflip_valid_count")) is not None
        else None,
        axis=1,
    )
    return out


def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def prepare_dataset_context(mod01b, dataset: str, bundle_mode: str) -> Dict[str, Any]:
    data_path = ROOT / "ufce" / "data" / f"{dataset}.csv"
    datasetdf = pd.read_csv(data_path)
    out = mod01b.classify_dataset_getModel(datasetdf, data_name=dataset)
    scaler = None
    if len(out) == 8:
        lr, _lr_mean, _lr_std, _xtest, xtrain, x_all, _y, datasetdf = out
    elif len(out) == 9:
        lr, _lr_mean, _lr_std, _xtest, xtrain, x_all, _y, datasetdf, scaler = out
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
    ) = mod01b.get_dataset_constraints(dataset, datasetdf)

    bundle = mod01b.resolve_bundle_config(
        dataset=dataset,
        args=SimpleNamespace(bundle_mode=str(bundle_mode)),
        author_uf=uf,
        author_f2change=f2change,
    )
    mi_fp = mod01b.normalize_mi_feature_pairs(mod01b.ufc.get_top_MI_features(x_all, features), features)
    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = mod01b.build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )

    return {
        "dataset": dataset,
        "datasetdf": datasetdf,
        "lr": lr,
        "xtrain": xtrain,
        "x_all": x_all,
        "scaler": scaler,
        "features": list(features),
        "catf": list(catf),
        "numf": list(numf),
        "uf": dict(bundle.uf),
        "f2change": list(bundle.f2change),
        "step": dict(bundle.step),
        "outcome_label": outcome_label,
        "desired_outcome": float(desired_outcome),
        "protectf": list(protectf),
        "data_lab1": data_lab1,
        "mi_fp": mi_fp,
        "movie_distance_scaler": movie_distance_scaler,
        "bundle": bundle,
    }


def run_author_methods(
    *,
    author_modules,
    context: Dict[str, Any],
    fold_df: pd.DataFrame,
    fold_name: str,
    no_cf: int,
    mi_top_k: Optional[int],
) -> Tuple[Dict[str, MethodOutput], List[Dict[str, Any]]]:
    dataset = str(context["dataset"])
    deterministic_seed = int(hashlib.sha256(f"{dataset}:{fold_name}:42".encode("utf-8")).hexdigest()[:8], 16)
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed % (2**32 - 1))

    features = context["features"]
    mi_fp = context["mi_fp"] if mi_top_k is None or int(mi_top_k) <= 0 else context["mi_fp"][: int(mi_top_k)]
    fold_features = fold_df.loc[:, features].copy()

    outputs: Dict[str, MethodOutput] = {}
    runtime_rows: List[Dict[str, Any]] = []

    onecfs, t1, idx1 = author_modules.cfmethods.sfexp(
        context["x_all"],
        context["data_lab1"],
        fold_features[:],
        context["uf"],
        context["step"],
        context["f2change"],
        context["numf"],
        context["catf"],
        context["lr"],
        context["desired_outcome"],
        no_cf,
        features,
    )
    outputs["UFCE1"] = MethodOutput(onecfs.reset_index(drop=True), [int(v) for v in idx1], float(t1) * 1000.0)

    twocfs, t2, idx2 = author_modules.cfmethods.dfexp(
        context["x_all"],
        context["data_lab1"],
        fold_features[:],
        context["uf"],
        mi_fp,
        context["numf"],
        context["catf"],
        context["f2change"],
        context["protectf"],
        context["lr"],
        context["desired_outcome"],
        no_cf,
        features,
    )
    outputs["UFCE2"] = MethodOutput(twocfs.reset_index(drop=True), [int(v) for v in idx2], float(t2) * 1000.0)

    threecfs, t3, idx3 = author_modules.cfmethods.tfexp(
        context["x_all"],
        context["data_lab1"],
        fold_features[:],
        context["uf"],
        mi_fp,
        context["numf"],
        context["catf"],
        context["f2change"],
        context["protectf"],
        context["lr"],
        context["desired_outcome"],
        no_cf,
        features,
    )
    outputs["UFCE3"] = MethodOutput(threecfs.reset_index(drop=True), [int(v) for v in idx3], float(t3) * 1000.0)

    for method, method_output in outputs.items():
        runtime_rows.append(
            {
                "dataset": dataset,
                "fold_id": fold_name,
                "method": method,
                "runtime_ms_per_query": float(method_output.runtime_ms),
            }
        )
    return outputs, runtime_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run raw author UFCE and post-hoc LR validity audit.")
    parser.add_argument("--dataset", default="all", help="{bank,bupa,grad,wine,movie,all} or comma-list")
    parser.add_argument("--bundle-mode", default="table7_author_public")
    parser.add_argument("--mi-k", default="5", help="Feature-pair count for UFCE2/3; use 'all' for all pairs.")
    parser.add_argument("--no-cf", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--fold-file", default=None)
    parser.add_argument("--compare-dir", default=None, help="Optional 07_ufce_ff output folder for comparison.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    datasets = parse_dataset_arg(args.dataset)
    mi_top_k = None if str(args.mi_k).strip().lower() == "all" else int(args.mi_k)
    progress = not bool(args.no_progress)
    output_root = Path(args.out_dir).resolve() if args.out_dir else (OUT_PARENT / now_tag()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mod01b = load_runner_01b()
    author_modules = load_author_modules()

    query_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    metric_fold_rows: List[Dict[str, Any]] = []

    dataset_iter = tqdm_wrap(datasets, total=len(datasets), desc="Datasets", leave=True, disable=not progress)
    for dataset in dataset_iter:
        context = prepare_dataset_context(mod01b, dataset, args.bundle_mode)
        testfold_path = ROOT / "ufce" / "data" / "folds" / dataset / "totest"
        testfolds = sorted(testfold_path.glob("*.csv"))
        if not testfolds:
            raise FileNotFoundError(f"No test folds found in {testfold_path}")
        if args.fold_file:
            target = os.path.basename(str(args.fold_file))
            testfolds = [fp for fp in testfolds if fp.name == target]
            if not testfolds:
                raise FileNotFoundError(f"Requested fold_file='{target}' was not found in {testfold_path}")
        if int(args.max_folds) > 0:
            testfolds = testfolds[: int(args.max_folds)]

        fold_iter = tqdm_wrap(
            testfolds,
            total=len(testfolds),
            desc=f"[{dataset}] author_raw_posthoc",
            leave=False,
            disable=not progress,
        )
        for fold_path in fold_iter:
            fold_df = pd.read_csv(fold_path)
            outputs, fold_runtime = run_author_methods(
                author_modules=author_modules,
                context=context,
                fold_df=fold_df,
                fold_name=fold_path.name,
                no_cf=int(args.no_cf),
                mi_top_k=mi_top_k,
            )
            runtime_rows.extend(fold_runtime)
            for method, method_output in outputs.items():
                raw_cfdf, raw_testdf, selected_count, valid_count = aligned_cf_and_test(
                    method_output=method_output,
                    fold_df=fold_df,
                    features=context["features"],
                    bb_model=context["lr"],
                    desired_outcome=context["desired_outcome"],
                    valid_only=False,
                )
                valid_cfdf, valid_testdf, _selected_count, _valid_count = aligned_cf_and_test(
                    method_output=method_output,
                    fold_df=fold_df,
                    features=context["features"],
                    bb_model=context["lr"],
                    desired_outcome=context["desired_outcome"],
                    valid_only=True,
                )
                for metric_scope, cf_for_metrics, test_for_metrics in [
                    ("author_raw", raw_cfdf, raw_testdf),
                    ("author_posthoc_valid_only", valid_cfdf, valid_testdf),
                ]:
                    metric_values = compute_six_metrics(
                        author_ufc=author_modules.cfmethods.ufc,
                        mod01b=mod01b,
                        method_name=method,
                        dataset=dataset,
                        cfdf=cf_for_metrics,
                        testdf=test_for_metrics,
                        catf=context["catf"],
                        numf=context["numf"],
                        features=context["features"],
                        f2change=context["f2change"],
                        uf=context["uf"],
                        xtrain=context["xtrain"],
                        bb_model=context["lr"],
                        desired_outcome=context["desired_outcome"],
                        movie_distance_scaler=context["movie_distance_scaler"],
                    )
                    row = {
                        "dataset": dataset,
                        "fold_id": fold_path.name,
                        "method": method,
                        "metric_scope": metric_scope,
                        "source_core": "ufce/core_author",
                        "bundle_mode": str(context["bundle"].effective_bundle_mode),
                        "mi_k": "all" if mi_top_k is None else str(mi_top_k),
                        "query_count": int(len(fold_df)),
                        "selected_count": int(selected_count),
                        "selected_valid_count": int(valid_count),
                        "selected_posthoc_fail_count": int(selected_count - valid_count),
                        "missing_count": int(len(fold_df) - selected_count),
                        "metric_candidate_count": int(len(cf_for_metrics)),
                    }
                    row.update(metric_values)
                    metric_fold_rows.append(row)
                method_rows = selected_rows_by_query(
                    method_output=method_output,
                    fold_df=fold_df,
                    features=context["features"],
                    bb_model=context["lr"],
                    desired_outcome=context["desired_outcome"],
                )
                for row in method_rows:
                    row.update(
                        {
                            "dataset": dataset,
                            "fold_id": fold_path.name,
                            "method": method,
                            "mode": "author_raw_posthoc",
                            "source_core": "ufce/core_author",
                            "bundle_mode": str(context["bundle"].effective_bundle_mode),
                            "mi_k": "all" if mi_top_k is None else str(mi_top_k),
                        }
                    )
                    query_rows.append(row)

    query_df = pd.DataFrame(query_rows)
    fold_summary, summary = summarize_query_rows(query_df)
    fold_summary = attach_runtime(fold_summary, runtime_rows)
    summary = aggregate_runtime(summary, fold_summary)
    metric_fold_df = pd.DataFrame(metric_fold_rows)
    metric_summary, metric_summary_long = summarize_metric_rows(metric_fold_df)
    metric_comparison = build_raw_vs_posthoc_metric_comparison(metric_summary_long)

    compare_dir = Path(args.compare_dir).resolve() if args.compare_dir else latest_07_dir()
    comparison = build_forceflip_comparison(summary, compare_dir)

    query_csv = output_root / "author_raw_posthoc_query.csv"
    fold_csv = output_root / "author_raw_posthoc_fold_summary.csv"
    summary_csv = output_root / "author_raw_posthoc_summary.csv"
    metric_fold_csv = output_root / "author_raw_posthoc_metric_fold_summary.csv"
    metric_summary_csv = output_root / "author_raw_posthoc_metric_summary.csv"
    metric_summary_long_csv = output_root / "author_raw_posthoc_metric_summary_long.csv"
    metric_comparison_csv = output_root / "author_raw_vs_posthoc_metric_comparison.csv"
    comparison_csv = output_root / "author_raw_vs_forceflip_comparison.csv"
    manifest_json = output_root / "manifest.json"
    summary_md = output_root / "summary.md"

    query_df.to_csv(query_csv, index=False)
    fold_summary.to_csv(fold_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    metric_fold_df.to_csv(metric_fold_csv, index=False)
    metric_summary.to_csv(metric_summary_csv, index=False)
    metric_summary_long.to_csv(metric_summary_long_csv, index=False)
    metric_comparison.to_csv(metric_comparison_csv, index=False)
    comparison.to_csv(comparison_csv, index=False)

    manifest = {
        "status": "ok",
        "output_root": str(output_root),
        "mode": "author_raw_posthoc",
        "datasets": datasets,
        "bundle_mode": str(args.bundle_mode),
        "mi_k": "all" if mi_top_k is None else str(mi_top_k),
        "no_cf": int(args.no_cf),
        "max_folds": int(args.max_folds),
        "fold_file": args.fold_file,
        "compare_dir": str(compare_dir) if compare_dir else None,
        "source": {
            "local_author_dir": str(AUTHOR_DIR),
            "github_repo": "https://github.com/msnizami/UFCE",
            "github_ufce_py": "https://github.com/msnizami/UFCE/blob/main/ufce.py",
            "github_cfmethods_py": "https://github.com/msnizami/UFCE/blob/main/cfmethods.py",
            "local_ufce_py_sha256": file_sha256(AUTHOR_DIR / "ufce.py"),
            "local_cfmethods_py_sha256": file_sha256(AUTHOR_DIR / "cfmethods.py"),
            "behavior_note": (
                "core_author mirrors the GitHub root files and is loaded with an author UFCE shim; "
                "validity is checked only after selected UFCE outputs are returned."
            ),
        },
        "outputs": {
            "query_csv": str(query_csv),
            "fold_summary_csv": str(fold_csv),
            "summary_csv": str(summary_csv),
            "metric_fold_csv": str(metric_fold_csv),
            "metric_summary_csv": str(metric_summary_csv),
            "metric_summary_long_csv": str(metric_summary_long_csv),
            "metric_comparison_csv": str(metric_comparison_csv),
            "comparison_csv": str(comparison_csv),
            "summary_md": str(summary_md),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Author Raw Post-hoc UFCE Summary",
        "",
        f"- output_root: `{output_root}`",
        f"- source_core: `ufce/core_author`",
        f"- github_repo: `https://github.com/msnizami/UFCE`",
        f"- compare_dir: `{compare_dir}`" if compare_dir else "- compare_dir: `None`",
        "",
        "## Aggregate Post-hoc Validity",
        "",
    ]
    if summary.empty:
        lines.append("_No rows produced._")
    else:
        lines.append(df_to_markdown(summary))
    if not comparison.empty:
        lines.extend(["", "## Raw Author vs Force-flip", ""])
        preview_cols = [
            "dataset",
            "method",
            "author_selected_count",
            "author_valid_count",
            "author_posthoc_fail_count",
            "author_strict_valid_rate",
            "forceflip_valid_count",
            "forceflip_strict_valid_rate",
            "forceflip_vs_author_valid_rate_delta",
        ]
        lines.append(df_to_markdown(comparison.loc[:, [c for c in preview_cols if c in comparison.columns]]))
    if not metric_comparison.empty:
        lines.extend(["", "## Raw Author vs Post-hoc Valid-only Metrics", ""])
        preview_cols = [
            "dataset",
            "method",
            "metric",
            "metric_direction",
            "author_raw_value",
            "posthoc_valid_value",
            "raw_to_posthoc_directional_delta",
            "raw_to_posthoc_verdict",
            "author_raw_metric_candidate_count",
            "posthoc_valid_metric_candidate_count",
        ]
        lines.append(df_to_markdown(metric_comparison.loc[:, [c for c in preview_cols if c in metric_comparison.columns]]))
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Author raw posthoc outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
