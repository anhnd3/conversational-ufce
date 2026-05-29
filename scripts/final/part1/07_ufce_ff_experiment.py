#!/usr/bin/env python3
"""
UFCE-FF experiment runner.

Primary path:
- public post-hoc audit: run the author-style selector, then check selected rows.
- force-flip pool: filter generated candidates by desired label before selecting.
- pool expansion diagnostic: repeat force-flip with wider MI feature-pair pools.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    _tqdm = None


ROOT = Path(__file__).resolve().parents[3]
RUNNER_01B = ROOT / "scripts" / "final" / "part1" / "01b_reproduce_ufce_only.py"

ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
DEFAULT_OUT_PARENT = ROOT / "outputs" / "final" / "part1"
PUBLIC_GITHUB_SOURCE_CONFIG = {
    "radius": 500,
    "n_neighbors": 100,
    "min_act": 3,
    "min_feas": 2,
    "ufce_flip_filter": 0,
}


class _DummyTqdm:
    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def update(self, _n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def tqdm_wrap(iterable=None, *, total=None, desc=None, leave=False, disable=False):
    if _tqdm is None:
        return _DummyTqdm(iterable=iterable)
    return _tqdm(iterable=iterable, total=total, desc=desc, leave=leave, disable=disable)


@dataclass
class QueryEval:
    dataset: str
    fold_id: str
    query_pos: int
    method: str
    mode: str
    config_profile: str
    bundle_mode: str
    mi_k: str
    top_n: int
    effective_radius: int
    effective_n_neighbors: int
    effective_min_act: int
    effective_min_feas: int
    effective_force_flip: int
    raw_candidate_count: int
    unique_candidate_count: int
    flip_candidate_count: int
    predict_call_count: int
    batch_predict_row_count: int
    dedup_reduction_ratio: float
    public_selected_flip: int
    verified_selected_flip: int
    strict_valid_selected: int
    failure_code: str
    failure_detail: str
    selected_source: str
    selected_prox_jac: Optional[float]
    selected_prox_euc: Optional[float]
    selected_sparsity: Optional[float]
    selected_apf_score: Optional[float]
    selected_actionability_pass: Optional[float]
    selected_plausibility_pass: Optional[float]
    selected_feasibility_pass: Optional[float]
    batch_predict_ms: float
    method_runtime_ms: float
    runtime_ms: float


def load_runner_01b():
    spec = importlib.util.spec_from_file_location("ufce_runner_01b", str(RUNNER_01B))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {RUNNER_01B}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def now_tag() -> str:
    return datetime.now().strftime("07_ufce_ff_%Y%m%d_%H%M%S")


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def resolve_trace_config(mod01b, dataset: str, config_profile: str, flip_filter_enabled: bool) -> Dict[str, int]:
    profile = str(config_profile).strip().lower()
    if profile == "public_github_source":
        cfg = dict(PUBLIC_GITHUB_SOURCE_CONFIG)
    elif profile == "final_freeze":
        cfg = dict(mod01b.FINAL_RUNTIME_CONFIG[dataset])
    elif profile == "new_best_params":
        cfg = dict(mod01b.NEW_BEST_PARAMS[dataset])
    else:
        raise ValueError(
            "Unsupported config_profile "
            f"'{config_profile}'. Allowed: public_github_source, final_freeze, new_best_params."
        )
    cfg["ufce_flip_filter"] = int(1 if flip_filter_enabled else 0)
    return {
        "radius": int(cfg["radius"]),
        "n_neighbors": int(cfg["n_neighbors"]),
        "min_act": int(cfg["min_act"]),
        "min_feas": int(cfg["min_feas"]),
        "ufce_flip_filter": int(cfg["ufce_flip_filter"]),
    }


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


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for item in str(text).split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer in list.")
    return values


def parse_mi_k_list(text: str) -> List[str]:
    out: List[str] = []
    for item in str(text).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "all":
            out.append("all")
            continue
        int(token)  # validate
        out.append(token)
    if not out:
        raise ValueError("Expected at least one MI-k value.")
    return out


def filter_mi_pairs_by_scope(
    mi_pairs: Sequence[Sequence[str]],
    uf: Dict[str, float],
    step: Dict[str, float],
    f2change: Sequence[str],
    scope: str,
) -> List[List[str]]:
    normalized_scope = str(scope).strip().lower()
    pairs = [list(pair) for pair in mi_pairs]
    if normalized_scope == "all_features":
        return pairs
    if normalized_scope not in {"uf_only", "configured_actionable"}:
        raise ValueError(f"Unsupported mi_feature_scope: {scope}")
    uf_keys = {str(key) for key in uf.keys()}
    if normalized_scope == "uf_only":
        return [pair for pair in pairs if all(str(feature) in uf_keys for feature in pair)]
    step_keys = {str(key) for key in step.keys()}
    f2change_keys = {str(key) for key in f2change}
    return [
        pair
        for pair in pairs
        if all(str(feature) in uf_keys and str(feature) in step_keys and str(feature) in f2change_keys for feature in pair)
    ]


def mode_list_from_arg(mode: str) -> List[str]:
    value = str(mode).strip().lower()
    if value == "all":
        return ["public_posthoc", "public_forceflip", "pool_expansion_forceflip"]
    if value == "final_comparison":
        return ["public_posthoc", "public_forceflip"]
    if value == "pool_expansion_comparison":
        return ["pool_expansion_forceflip"]
    supported = {
        "public_posthoc",
        "public_forceflip",
        "pool_expansion_forceflip",
        "pool_expansion_comparison",
        "final_comparison",
    }
    if value not in supported:
        raise ValueError(f"Unsupported mode: {value}")
    return [value]


def predict_label(model, candidate_df: pd.DataFrame, feature_order: Sequence[str]) -> Optional[int]:
    if not isinstance(candidate_df, pd.DataFrame) or candidate_df.empty:
        return None
    cols = [c for c in feature_order if c in candidate_df.columns]
    if len(cols) != len(feature_order):
        return None
    pred = np.asarray(model.predict(candidate_df.loc[:, cols])).reshape(-1)
    if pred.size == 0:
        return None
    return int(pred[0])


def audit_flipping_pool(
    model,
    candidate_df: pd.DataFrame,
    desired_outcome: float,
    feature_order: Sequence[str],
) -> Tuple[pd.DataFrame, int, int, float]:
    if not isinstance(candidate_df, pd.DataFrame) or candidate_df.empty:
        return pd.DataFrame(columns=list(feature_order)), 0, 0, 0.0
    pool = candidate_df.loc[:, [c for c in feature_order if c in candidate_df.columns]].copy().reset_index(drop=True)
    if len(pool.columns) != len(feature_order):
        return pd.DataFrame(columns=list(feature_order)), 0, 0, 0.0
    t0 = time.perf_counter()
    preds = np.asarray(model.predict(pool.loc[:, list(feature_order)])).reshape(-1)
    batch_predict_ms = float((time.perf_counter() - t0) * 1000.0)
    flip_pool = pool.loc[preds == int(desired_outcome)].reset_index(drop=True)
    return flip_pool, 1, int(len(pool)), batch_predict_ms


def _empty_metrics() -> Dict[str, Optional[float]]:
    return {
        "prox_jac": None,
        "prox_euc": None,
        "sparsity": None,
        "apf_score": None,
        "actionability_pass": None,
        "plausibility_pass": None,
        "feasibility_pass": None,
    }


def compute_candidate_metrics(
    *,
    mod01b,
    active_ufc,
    factual_row: pd.DataFrame,
    candidate_row: pd.DataFrame,
    xtrain: pd.DataFrame,
    features: List[str],
    numf: List[str],
    catf: List[str],
    f2change: List[str],
    uf: Dict[str, float],
    bb_model,
    desired_outcome: float,
    movie_distance_scaler: Optional[Dict[str, object]],
) -> Dict[str, Optional[float]]:
    if not isinstance(candidate_row, pd.DataFrame) or candidate_row.empty:
        return _empty_metrics()

    prox_jac = None
    try:
        if len(catf) > 0:
            prox_jac = mod01b._scalar(
                active_ufc.categorical_distance(
                    factual_row.copy(),
                    candidate_row.copy(),
                    catf,
                    metric="jaccard",
                    agg=None,
                )
            )
    except Exception:
        prox_jac = None

    sparsity = None
    try:
        sparsity_d, _ = active_ufc.sparsity_count(candidate_row.copy(), factual_row.copy(), numf, numf)
        vals = list(sparsity_d.values()) if isinstance(sparsity_d, dict) else []
        sparsity = float(vals[0]) if vals else None
    except Exception:
        sparsity = None

    prox_euc = None
    try:
        prox_euc, _contrib, _dist_cols, _normalizer = mod01b._feature_contributions(
            factual_row,
            candidate_row,
            numf,
            movie_distance_scaler,
        )
    except Exception:
        prox_euc = None

    action_pass = False
    plaus_pass = False
    feas_pass = False
    try:
        action_pass, _reason, _num, _den, _a_changed, _na_changed = mod01b._safe_actionability_pair(
            active_ufc,
            factual_row.copy(),
            candidate_row.copy(),
            features,
            f2change,
            uf,
        )
    except Exception:
        action_pass = False
    try:
        plaus_pass, _reason, _details = mod01b._safe_plausibility_pair(
            active_ufc,
            "other",
            factual_row.copy(),
            candidate_row.copy(),
            xtrain.copy(),
        )
    except Exception:
        plaus_pass = False
    try:
        feas_pass, _reason, _details = mod01b._safe_feasibility_pair(
            active_ufc,
            factual_row.copy(),
            candidate_row.copy(),
            xtrain.copy(),
            features,
            f2change,
            bb_model,
            desired_outcome,
            uf,
        )
    except Exception:
        feas_pass = False

    apf_score = int(bool(action_pass)) + int(bool(plaus_pass)) + int(bool(feas_pass))
    return {
        "prox_jac": finite_float(prox_jac),
        "prox_euc": finite_float(prox_euc),
        "sparsity": finite_float(sparsity),
        "apf_score": float(apf_score),
        "actionability_pass": float(int(bool(action_pass))),
        "plausibility_pass": float(int(bool(plaus_pass))),
        "feasibility_pass": float(int(bool(feas_pass))),
    }


def classify_failure(
    *,
    raw_count: int,
    flip_count: int,
    public_selected_exists: bool,
    public_selected_flip: bool,
    flip_rank_rows: pd.DataFrame,
) -> Tuple[str, str]:
    if raw_count == 0:
        return "F0", "no_raw_candidate_generated"
    if flip_count == 0:
        return "F1", "F5_not_observable_label_only"
    if public_selected_exists and not public_selected_flip:
        return "F2", "flip_exists_but_public_selected_non_flip"
    if public_selected_exists and public_selected_flip and flip_rank_rows.empty:
        return "OK", "public_selected_valid_no_rank_gate"

    if flip_rank_rows.empty:
        return "F1", "no_flip_rows_after_merge"
    any_action = bool((flip_rank_rows.get("actionability_pass", pd.Series(dtype=float)) > 0).any())
    any_plaus = bool((flip_rank_rows.get("plausibility_pass", pd.Series(dtype=float)) > 0).any())
    any_feas = bool((flip_rank_rows.get("feasibility_pass", pd.Series(dtype=float)) > 0).any())
    if not any_action:
        return "F3", "no_flip_candidate_passed_actionability_gate"
    if not (any_plaus and any_feas):
        return "F4", "flip_candidates_fail_downstream_apf_filters"
    return "OK", "has_valid_flip_selection_path"


def summarize_queries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "mode",
                "config_profile",
                "bundle_mode",
                "mi_k",
                "top_n",
                "effective_radius",
                "effective_n_neighbors",
                "effective_min_act",
                "effective_min_feas",
                "effective_force_flip",
                "query_count",
                "raw_candidate_count",
                "unique_candidate_count",
                "flip_candidate_count",
                "public_selected_flip_count",
                "verified_selected_flip_count",
                "strict_valid_rate",
                "main_failure_reduced",
                "mean_prox_jac_valid",
                "mean_prox_euc_valid",
                "mean_sparsity_valid",
                "mean_actionability_valid",
                "mean_plausibility_valid",
                "mean_feasibility_valid",
                "mean_apf_valid",
                "predict_call_count",
                "batch_predict_row_count",
                "mean_batch_predict_ms",
                "mean_runtime_ms",
                "runtime_ratio",
            ]
        )

    rows: List[Dict[str, Any]] = []
    group_cols = [
        "dataset",
        "method",
        "mode",
        "config_profile",
        "bundle_mode",
        "mi_k",
        "top_n",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
        "effective_force_flip",
    ]
    for key, group in df.groupby(group_cols, dropna=False):
        (
            dataset,
            method,
            mode,
            config_profile,
            bundle_mode,
            mi_k,
            top_n,
            effective_radius,
            effective_n_neighbors,
            effective_min_act,
            effective_min_feas,
            effective_force_flip,
        ) = key
        query_count = int(len(group))
        strict_valid = (
            int(group["public_selected_flip"].sum())
            if str(mode).startswith("public_")
            else int(group["verified_selected_flip"].sum())
        )
        valid_rows = group[group["strict_valid_selected"] == 1]

        def _fold_mean_sum(column: str) -> Optional[float]:
            if column not in group.columns:
                return None
            fold_ids = list(group["fold_id"].drop_duplicates())
            if not fold_ids:
                return None
            values: List[float] = []
            for fold_id in fold_ids:
                fold_valid = valid_rows[valid_rows["fold_id"] == fold_id]
                values.append(float(fold_valid[column].sum()) if not fold_valid.empty else 0.0)
            return finite_float(float(np.mean(values)))

        rows.append(
            {
                "dataset": str(dataset),
                "method": str(method),
                "mode": str(mode),
                "config_profile": str(config_profile),
                "bundle_mode": str(bundle_mode),
                "mi_k": str(mi_k),
                "top_n": int(top_n),
                "effective_radius": int(effective_radius),
                "effective_n_neighbors": int(effective_n_neighbors),
                "effective_min_act": int(effective_min_act),
                "effective_min_feas": int(effective_min_feas),
                "effective_force_flip": int(effective_force_flip),
                "query_count": query_count,
                "raw_candidate_count": int(group["raw_candidate_count"].sum()),
                "unique_candidate_count": int(group["unique_candidate_count"].sum()),
                "flip_candidate_count": int(group["flip_candidate_count"].sum()),
                "public_selected_flip_count": int(group["public_selected_flip"].sum()),
                "verified_selected_flip_count": int(group["verified_selected_flip"].sum()),
                "strict_valid_rate": float(strict_valid / query_count) if query_count > 0 else 0.0,
                "main_failure_reduced": str(group["failure_code"].value_counts().idxmax()) if not group.empty else "",
                "mean_prox_jac_valid": finite_float(valid_rows["selected_prox_jac"].mean()) if not valid_rows.empty else None,
                "mean_prox_euc_valid": finite_float(valid_rows["selected_prox_euc"].mean()) if not valid_rows.empty else None,
                "mean_sparsity_valid": finite_float(valid_rows["selected_sparsity"].mean()) if not valid_rows.empty else None,
                "mean_actionability_valid": _fold_mean_sum("selected_actionability_pass"),
                "mean_plausibility_valid": _fold_mean_sum("selected_plausibility_pass"),
                "mean_feasibility_valid": _fold_mean_sum("selected_feasibility_pass"),
                "mean_apf_valid": finite_float(valid_rows["selected_apf_score"].mean()) if not valid_rows.empty else None,
                "predict_call_count": int(group["predict_call_count"].sum()),
                "batch_predict_row_count": int(group["batch_predict_row_count"].sum()),
                "mean_batch_predict_ms": finite_float(group["batch_predict_ms"].mean()),
                "mean_runtime_ms": finite_float(group["runtime_ms"].mean()),
                "runtime_ratio": None,
            }
        )
    out = pd.DataFrame(rows)
    baseline_join_cols = [
        "dataset",
        "method",
        "config_profile",
        "bundle_mode",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
    ]
    baseline = out.loc[out["mode"] == "public_forceflip", baseline_join_cols + ["mean_runtime_ms"]].copy()
    baseline = baseline.rename(columns={"mean_runtime_ms": "baseline_runtime_ms"})
    out = out.merge(baseline, on=baseline_join_cols, how="left")
    out["runtime_ratio"] = out.apply(
        lambda row: float(row["mean_runtime_ms"] / row["baseline_runtime_ms"])
        if finite_float(row["mean_runtime_ms"]) is not None and finite_float(row["baseline_runtime_ms"]) not in (None, 0.0)
        else None,
        axis=1,
    )
    out = out.drop(columns=["baseline_runtime_ms"])
    return out


def run_dataset_traces(
    *,
    mod01b,
    dataset: str,
    mi_k: Optional[int],
    config_profile: str,
    bundle_mode: str,
    flip_filter_enabled: bool,
    no_cf: int,
    max_folds: int,
    fold_file: Optional[str],
    contprox_metric: str,
    mi_feature_scope: str,
    progress: bool = True,
) -> Dict[str, Any]:
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
        _outcome_label,
        desired_outcome,
        _nbr_features,
        protectf,
        _data_lab0,
        data_lab1,
    ) = mod01b.get_dataset_constraints(dataset, datasetdf)

    bundle_args = SimpleNamespace(bundle_mode=str(bundle_mode))
    bundle = mod01b.resolve_bundle_config(dataset=dataset, args=bundle_args, author_uf=uf, author_f2change=f2change)
    uf = bundle.uf
    f2change = bundle.f2change
    step = bundle.step

    cfg = resolve_trace_config(
        mod01b=mod01b,
        dataset=dataset,
        config_profile=config_profile,
        flip_filter_enabled=flip_filter_enabled,
    )

    mi_fp_raw = mod01b.normalize_mi_feature_pairs(mod01b.ufc.get_top_MI_features(x_all, features), features)
    mi_fp = filter_mi_pairs_by_scope(mi_fp_raw, uf, step, f2change, mi_feature_scope)
    if len(mi_fp) == 0:
        raise ValueError(
            f"No MI feature pairs remain for dataset={dataset} with mi_feature_scope={mi_feature_scope}. "
            "Use --mi-feature-scope all_features only if every expanded feature has a valid UF bound."
        )
    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = mod01b.build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=_outcome_label,
        )

    mod01b.cfmethods.initUFCE(
        radius=cfg["radius"],
        n_neighbors=cfg["n_neighbors"],
        contprox_metric=contprox_metric,
        min_act=cfg["min_act"],
        min_feas=cfg["min_feas"],
        atol=1e-5,
    )
    mod01b.eval_module.ufc = mod01b.cfmethods.ufc
    active_ufc = mod01b._active_ufce_instance()

    testfold_path = ROOT / "ufce" / "data" / "folds" / dataset / "totest"
    testfolds = sorted(testfold_path.glob("*.csv"))
    if not testfolds:
        raise FileNotFoundError(f"No test folds found in {testfold_path}")
    if fold_file:
        target = os.path.basename(str(fold_file))
        selected = [fp for fp in testfolds if fp.name == target]
        if not selected:
            raise FileNotFoundError(f"Requested fold_file='{target}' was not found in {testfold_path}")
        testfolds = selected
    if max_folds > 0:
        testfolds = testfolds[: int(max_folds)]

    folds: List[Dict[str, Any]] = []
    fold_iter = tqdm_wrap(
        enumerate(testfolds),
        total=len(testfolds),
        desc=f"[{dataset}] traces mi={mi_k if mi_k is not None else 'all'} flip={int(bool(flip_filter_enabled))}",
        leave=False,
        disable=not progress,
    )
    for fold_idx, fold_path in fold_iter:
        fold_df = pd.read_csv(fold_path)
        fr = mod01b.run_one_fold(
            dataset=dataset,
            fold_df=fold_df,
            x_all=x_all,
            xtest=fold_df.copy(),
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
            no_cf=no_cf,
            step=step,
            fold_name=fold_path.name,
            scaler=scaler,
            flip_filter_enabled=bool(cfg["ufce_flip_filter"]),
            movie_distance_scaler=movie_distance_scaler,
            fold_index=fold_idx,
            debug=0,
            prox_euc_contract_debug=0,
            contract_debug_fold=0,
            contract_debug_pos=0,
            contract_debug_method="UFCE2",
            diagnostics_enabled=True,
            diagnostics_top_k=max(100, no_cf),
            mi_top_k=mi_k,
            cfg=cfg,
            bundle_cfg=bundle.bundle_cfg,
            bundle_meta=mod01b.effective_config_record(dataset=dataset, cfg=cfg, bundle=bundle, runtime_profile=str(config_profile)),
        )
        folds.append(
            {
                "fold_name": fold_path.name,
                "fold_df": fold_df,
                "trace_payload": fr.trace_payload or {},
                "times": fr.times,
            }
        )

    return {
        "dataset": dataset,
        "config_profile": str(config_profile),
        "bundle_mode": str(bundle.effective_bundle_mode),
        "effective_cfg": dict(cfg),
        "features": list(features),
        "numf": list(numf),
        "catf": list(catf),
        "mi_feature_scope": str(mi_feature_scope),
        "mi_pair_count_raw": int(len(mi_fp_raw)),
        "mi_pair_count_effective": int(len(mi_fp)),
        "mi_pairs_effective": [list(pair) for pair in mi_fp],
        "f2change": list(f2change),
        "uf": dict(uf),
        "desired_outcome": float(desired_outcome),
        "bb_model": lr,
        "xtrain": xtrain,
        "movie_distance_scaler": movie_distance_scaler,
        "active_ufc": active_ufc,
        "folds": folds,
    }


def evaluate_mode_from_cache(
    *,
    mod01b,
    cache_obj: Dict[str, Any],
    mode: str,
    mi_k_label: str,
    top_n: int,
    progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[QueryEval]:
    dataset = str(cache_obj["dataset"])
    config_profile = str(cache_obj.get("config_profile", "unknown"))
    bundle_mode = str(cache_obj.get("bundle_mode", "unknown"))
    effective_cfg = dict(cache_obj.get("effective_cfg", {}))
    features = list(cache_obj["features"])
    numf = list(cache_obj["numf"])
    catf = list(cache_obj["catf"])
    f2change = list(cache_obj["f2change"])
    uf = dict(cache_obj["uf"])
    desired_outcome = float(cache_obj["desired_outcome"])
    bb_model = cache_obj["bb_model"]
    xtrain = cache_obj["xtrain"]
    active_ufc = cache_obj["active_ufc"]
    movie_distance_scaler = cache_obj["movie_distance_scaler"]
    rows: List[QueryEval] = []
    total_queries = 0
    for fold_bundle in cache_obj["folds"]:
        trace_payload = fold_bundle.get("trace_payload", {})
        for method in METHODS:
            total_queries += int(len(trace_payload.get(method, [])))
    pbar = tqdm_wrap(
        total=total_queries,
        desc=progress_desc or f"[{dataset}] {mode} mi={mi_k_label}",
        leave=False,
        disable=not progress,
    )

    for fold_bundle in cache_obj["folds"]:
        fold_name = str(fold_bundle["fold_name"])
        fold_df = fold_bundle["fold_df"]
        trace_payload = fold_bundle.get("trace_payload", {})
        times = fold_bundle.get("times", {})
        for method in METHODS:
            method_traces = list(trace_payload.get(method, []))
            method_runtime_ms = float(finite_float(times.get(method, 0.0)) or 0.0) * 1000.0
            for trace_row in method_traces:
                query_pos = int(trace_row.get("instance_pos", 0))
                factual_row = fold_df.iloc[[query_pos]][features].reset_index(drop=True)
                raw_pool_df = trace_row.get("generated_candidates_df")
                if not isinstance(raw_pool_df, pd.DataFrame):
                    raw_pool_df = pd.DataFrame(columns=features)
                else:
                    raw_pool_df = raw_pool_df.loc[:, [c for c in features if c in raw_pool_df.columns]].copy().reset_index(drop=True)
                public_selected_df = trace_row.get("selected_candidates_df")
                if not isinstance(public_selected_df, pd.DataFrame):
                    public_selected_df = pd.DataFrame(columns=features)
                else:
                    public_selected_df = public_selected_df.loc[:, [c for c in features if c in public_selected_df.columns]].copy().reset_index(drop=True)

                raw_count = int(len(raw_pool_df)) if isinstance(raw_pool_df, pd.DataFrame) else 0
                unique_count = int(len(raw_pool_df.drop_duplicates())) if raw_count > 0 else 0
                dedup_reduction_ratio = float((raw_count - unique_count) / raw_count) if raw_count > 0 else 0.0
                flip_candidates_df, predict_call_count, batch_predict_row_count, batch_predict_ms = audit_flipping_pool(
                    bb_model,
                    raw_pool_df,
                    desired_outcome,
                    features,
                )
                flip_count = int(len(flip_candidates_df))
                public_selected_exists = not public_selected_df.empty
                public_pred = predict_label(bb_model, public_selected_df.iloc[[0]] if public_selected_exists else public_selected_df, features)
                public_flip = bool(public_pred == int(desired_outcome)) if public_selected_exists else False

                selected_source = "public"
                strict_valid_selected = int(public_flip)
                selected_metrics = _empty_metrics()
                if public_selected_exists and public_flip:
                    selected_metrics = compute_candidate_metrics(
                        mod01b=mod01b,
                        active_ufc=active_ufc,
                        factual_row=factual_row,
                        candidate_row=public_selected_df.iloc[[0]],
                        xtrain=xtrain,
                        features=features,
                        numf=numf,
                        catf=catf,
                        f2change=f2change,
                        uf=uf,
                        bb_model=bb_model,
                        desired_outcome=desired_outcome,
                        movie_distance_scaler=movie_distance_scaler,
                    )

                failure_code, failure_detail = classify_failure(
                    raw_count=raw_count,
                    flip_count=flip_count,
                    public_selected_exists=public_selected_exists,
                    public_selected_flip=public_flip,
                    flip_rank_rows=pd.DataFrame(),
                )
                runtime_ms = float(method_runtime_ms)
                rows.append(
                    QueryEval(
                        dataset=dataset,
                        fold_id=fold_name,
                        query_pos=query_pos,
                        method=method,
                        mode=mode,
                        config_profile=config_profile,
                        bundle_mode=bundle_mode,
                        mi_k=str(mi_k_label),
                        top_n=int(top_n),
                        effective_radius=int(effective_cfg.get("radius", 0)),
                        effective_n_neighbors=int(effective_cfg.get("n_neighbors", 0)),
                        effective_min_act=int(effective_cfg.get("min_act", 0)),
                        effective_min_feas=int(effective_cfg.get("min_feas", 0)),
                        effective_force_flip=int(effective_cfg.get("ufce_flip_filter", 0)),
                        raw_candidate_count=raw_count,
                        unique_candidate_count=unique_count,
                        flip_candidate_count=flip_count,
                        predict_call_count=int(predict_call_count),
                        batch_predict_row_count=int(batch_predict_row_count),
                        dedup_reduction_ratio=dedup_reduction_ratio,
                        public_selected_flip=int(public_flip),
                        verified_selected_flip=int(public_flip),
                        strict_valid_selected=int(strict_valid_selected),
                        failure_code=failure_code,
                        failure_detail=failure_detail,
                        selected_source=selected_source,
                        selected_prox_jac=finite_float(selected_metrics.get("prox_jac")),
                        selected_prox_euc=finite_float(selected_metrics.get("prox_euc")),
                        selected_sparsity=finite_float(selected_metrics.get("sparsity")),
                        selected_apf_score=finite_float(selected_metrics.get("apf_score")),
                        selected_actionability_pass=finite_float(selected_metrics.get("actionability_pass")),
                        selected_plausibility_pass=finite_float(selected_metrics.get("plausibility_pass")),
                        selected_feasibility_pass=finite_float(selected_metrics.get("feasibility_pass")),
                        batch_predict_ms=batch_predict_ms,
                        method_runtime_ms=method_runtime_ms,
                        runtime_ms=runtime_ms,
                    )
                )
                pbar.update(1)
    pbar.close()
    return rows


def build_claim_check(summary_df: pd.DataFrame) -> Dict[str, Any]:
    if summary_df.empty:
        return {"status": "failed", "reason": "empty_summary"}

    posthoc_rows = summary_df[summary_df["mode"] == "public_posthoc"]
    forceflip_rows = summary_df[summary_df["mode"] == "public_forceflip"]
    merged = forceflip_rows.merge(
        posthoc_rows[["dataset", "method", "strict_valid_rate"]].rename(columns={"strict_valid_rate": "posthoc_rate"}),
        on=["dataset", "method"],
        how="left",
    )
    merged["delta_rate"] = merged["strict_valid_rate"] - merged["posthoc_rate"]
    improved_pairs = merged[merged["delta_rate"] >= 0.05]
    return {
        "status": "passed" if len(improved_pairs) >= 1 else "warning",
        "forceflip_vs_posthoc_pairs_ge_5pp": int(len(improved_pairs)),
        "evaluated_rows": int(len(merged)),
    }


def _safe_delta(new_value: Any, old_value: Any) -> Optional[float]:
    new_v = finite_float(new_value)
    old_v = finite_float(old_value)
    if new_v is None or old_v is None:
        return None
    return float(new_v - old_v)


def _safe_relative_pct(new_value: Any, old_value: Any) -> Optional[float]:
    delta = _safe_delta(new_value, old_value)
    old_v = finite_float(old_value)
    if delta is None or old_v in (None, 0.0):
        return None
    return float((delta / old_v) * 100.0)


def _verdict(delta: Optional[float], positive: str, negative: str, tol: float = 1e-12) -> str:
    if delta is None:
        return "na"
    if delta > tol:
        return positive
    if delta < -tol:
        return negative
    return "same"


def build_final_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "method",
        "config_profile",
        "bundle_mode",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
        "baseline_mi_k",
        "candidate_mi_k",
        "baseline_strict_valid_rate",
        "candidate_strict_valid_rate",
        "strict_valid_rate_delta",
        "strict_valid_rate_relative_delta_pct",
        "strict_valid_verdict",
        "baseline_mean_runtime_ms",
        "candidate_mean_runtime_ms",
        "runtime_ratio_candidate_vs_baseline",
        "runtime_verdict",
        "baseline_mean_prox_euc_valid",
        "candidate_mean_prox_euc_valid",
        "prox_euc_delta",
        "prox_euc_verdict",
        "baseline_mean_sparsity_valid",
        "candidate_mean_sparsity_valid",
        "sparsity_delta",
        "sparsity_verdict",
        "baseline_mean_apf_valid",
        "candidate_mean_apf_valid",
        "apf_delta",
        "apf_verdict",
        "baseline_main_failure",
        "candidate_main_failure",
    ]
    if summary_df.empty:
        return pd.DataFrame(columns=columns)

    baseline = summary_df.loc[
        (summary_df["mode"] == "public_forceflip") & (summary_df["mi_k"].astype(str) == "5")
    ].copy()
    if baseline.empty:
        return pd.DataFrame(columns=columns)
    keep_cols = [
        "dataset",
        "method",
        "config_profile",
        "bundle_mode",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
        "mi_k",
        "strict_valid_rate",
        "mean_runtime_ms",
        "mean_prox_euc_valid",
        "mean_sparsity_valid",
        "mean_apf_valid",
        "main_failure_reduced",
    ]
    baseline = baseline.loc[:, keep_cols].rename(
        columns={
            "mi_k": "baseline_mi_k",
            "strict_valid_rate": "baseline_strict_valid_rate",
            "mean_runtime_ms": "baseline_mean_runtime_ms",
            "mean_prox_euc_valid": "baseline_mean_prox_euc_valid",
            "mean_sparsity_valid": "baseline_mean_sparsity_valid",
            "mean_apf_valid": "baseline_mean_apf_valid",
            "main_failure_reduced": "baseline_main_failure",
        }
    )
    final_join_cols = [
        "dataset",
        "method",
        "config_profile",
        "bundle_mode",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
    ]
    baseline = baseline.drop_duplicates(subset=final_join_cols, keep="first")

    compare_rows = summary_df.loc[
        (summary_df["mode"] == "public_forceflip") & (summary_df["mi_k"].astype(str) != "5")
    ].copy()
    if compare_rows.empty:
        return pd.DataFrame(columns=columns)
    compare_rows = compare_rows.loc[:, keep_cols].rename(
        columns={
            "mi_k": "candidate_mi_k",
            "strict_valid_rate": "candidate_strict_valid_rate",
            "mean_runtime_ms": "candidate_mean_runtime_ms",
            "mean_prox_euc_valid": "candidate_mean_prox_euc_valid",
            "mean_sparsity_valid": "candidate_mean_sparsity_valid",
            "mean_apf_valid": "candidate_mean_apf_valid",
            "main_failure_reduced": "candidate_main_failure",
        }
    )

    merged = compare_rows.merge(baseline, on=final_join_cols, how="left")
    merged["strict_valid_rate_delta"] = merged.apply(
        lambda row: _safe_delta(row.get("candidate_strict_valid_rate"), row.get("baseline_strict_valid_rate")),
        axis=1,
    )
    merged["strict_valid_rate_relative_delta_pct"] = merged.apply(
        lambda row: _safe_relative_pct(row.get("candidate_strict_valid_rate"), row.get("baseline_strict_valid_rate")),
        axis=1,
    )
    merged["strict_valid_verdict"] = merged["strict_valid_rate_delta"].map(
        lambda d: _verdict(d, positive="improved", negative="regressed")
    )

    merged["runtime_ratio_candidate_vs_baseline"] = merged.apply(
        lambda row: float(row["candidate_mean_runtime_ms"] / row["baseline_mean_runtime_ms"])
        if finite_float(row.get("candidate_mean_runtime_ms")) is not None and finite_float(row.get("baseline_mean_runtime_ms")) not in (None, 0.0)
        else None,
        axis=1,
    )
    merged["runtime_verdict"] = merged["runtime_ratio_candidate_vs_baseline"].map(
        lambda r: "faster" if finite_float(r) is not None and float(r) < 1.0 else ("slower" if finite_float(r) is not None and float(r) > 1.0 else ("same" if finite_float(r) is not None else "na"))
    )

    merged["prox_euc_delta"] = merged.apply(
        lambda row: _safe_delta(row.get("candidate_mean_prox_euc_valid"), row.get("baseline_mean_prox_euc_valid")),
        axis=1,
    )
    merged["prox_euc_verdict"] = merged["prox_euc_delta"].map(
        lambda d: _verdict(-d if d is not None else None, positive="improved", negative="regressed")
    )

    merged["sparsity_delta"] = merged.apply(
        lambda row: _safe_delta(row.get("candidate_mean_sparsity_valid"), row.get("baseline_mean_sparsity_valid")),
        axis=1,
    )
    merged["sparsity_verdict"] = merged["sparsity_delta"].map(
        lambda d: _verdict(-d if d is not None else None, positive="improved", negative="regressed")
    )

    merged["apf_delta"] = merged.apply(
        lambda row: _safe_delta(row.get("candidate_mean_apf_valid"), row.get("baseline_mean_apf_valid")),
        axis=1,
    )
    merged["apf_verdict"] = merged["apf_delta"].map(
        lambda d: _verdict(d, positive="improved", negative="regressed")
    )

    final_df = merged.loc[:, [col for col in columns if col in merged.columns]].copy()
    final_df = final_df.sort_values(by=["dataset", "method", "candidate_mi_k"], kind="mergesort").reset_index(drop=True)
    return final_df


def build_stage_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "dataset",
        "method",
        "config_profile",
        "bundle_mode",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
    ]
    columns = id_cols + [
        "posthoc_strict_valid_rate",
        "forceflip_strict_valid_rate",
        "forceflip_vs_posthoc_delta",
        "forceflip_vs_posthoc_verdict",
        "posthoc_mean_runtime_ms",
        "forceflip_mean_runtime_ms",
        "forceflip_vs_posthoc_runtime_ratio",
        "posthoc_mean_prox_euc_valid",
        "forceflip_mean_prox_euc_valid",
        "posthoc_mean_sparsity_valid",
        "forceflip_mean_sparsity_valid",
        "posthoc_main_failure",
        "forceflip_main_failure",
    ]
    if summary_df.empty:
        return pd.DataFrame(columns=columns)

    source_cols = id_cols + [
        "mode",
        "strict_valid_rate",
        "mean_runtime_ms",
        "mean_prox_euc_valid",
        "mean_sparsity_valid",
        "main_failure_reduced",
    ]
    available = [col for col in source_cols if col in summary_df.columns]
    df = summary_df.loc[summary_df["mode"].isin(["public_posthoc", "public_forceflip"]), available].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    def _mode_frame(mode: str, prefix: str) -> pd.DataFrame:
        part = df.loc[df["mode"] == mode].copy()
        if part.empty:
            return pd.DataFrame(columns=id_cols)
        part = part.drop_duplicates(subset=id_cols, keep="first")
        return part.loc[
            :,
            id_cols + ["strict_valid_rate", "mean_runtime_ms", "mean_prox_euc_valid", "mean_sparsity_valid", "main_failure_reduced"],
        ].rename(
            columns={
                "strict_valid_rate": f"{prefix}_strict_valid_rate",
                "mean_runtime_ms": f"{prefix}_mean_runtime_ms",
                "mean_prox_euc_valid": f"{prefix}_mean_prox_euc_valid",
                "mean_sparsity_valid": f"{prefix}_mean_sparsity_valid",
                "main_failure_reduced": f"{prefix}_main_failure",
            }
        )

    posthoc = _mode_frame("public_posthoc", "posthoc")
    forceflip = _mode_frame("public_forceflip", "forceflip")
    merged = posthoc.merge(forceflip, on=id_cols, how="outer")

    merged["forceflip_vs_posthoc_delta"] = merged.apply(
        lambda row: _safe_delta(row.get("forceflip_strict_valid_rate"), row.get("posthoc_strict_valid_rate")),
        axis=1,
    )
    merged["forceflip_vs_posthoc_verdict"] = merged["forceflip_vs_posthoc_delta"].map(
        lambda d: _verdict(d, positive="improved", negative="regressed")
    )
    merged["forceflip_vs_posthoc_runtime_ratio"] = merged.apply(
        lambda row: float(row["forceflip_mean_runtime_ms"] / row["posthoc_mean_runtime_ms"])
        if finite_float(row.get("forceflip_mean_runtime_ms")) is not None and finite_float(row.get("posthoc_mean_runtime_ms")) not in (None, 0.0)
        else None,
        axis=1,
    )
    return merged.loc[:, [col for col in columns if col in merged.columns]].sort_values(by=["dataset", "method"], kind="mergesort").reset_index(drop=True)


def render_summary_md(
    output_root: Path,
    query_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    final_comparison_df: pd.DataFrame,
    stage_comparison_df: pd.DataFrame,
    claim_check: Dict[str, Any],
) -> str:
    lines = [
        "# UFCE-FF Experiment Summary",
        "",
        f"- output_root: `{output_root}`",
        f"- query_rows: `{len(query_df)}`",
        f"- summary_rows: `{len(summary_df)}`",
        f"- claim_status: `{claim_check.get('status', 'unknown')}`",
        f"- forceflip_vs_posthoc_pairs_ge_5pp: `{claim_check.get('forceflip_vs_posthoc_pairs_ge_5pp', 0)}`",
        "",
        "## Summary Preview",
        "",
    ]
    preview_cols = [
        "dataset",
        "method",
        "mode",
        "config_profile",
        "effective_radius",
        "effective_n_neighbors",
        "effective_min_act",
        "effective_min_feas",
        "effective_force_flip",
        "mi_k",
        "top_n",
        "strict_valid_rate",
        "mean_runtime_ms",
        "runtime_ratio",
        "mean_prox_euc_valid",
        "mean_sparsity_valid",
        "mean_apf_valid",
    ]
    preview = summary_df.loc[:, [c for c in preview_cols if c in summary_df.columns]].copy()
    lines.append(preview.head(60).to_markdown(index=False))
    if len(preview) > 60:
        lines.append("\n... (truncated)")
    lines.extend(
        [
            "",
            "## Final Comparison Preview",
            "",
        ]
    )
    if final_comparison_df.empty:
        lines.append("_No merged MI comparison table was generated (need MI[:5] and a wider MI force-flip run)._")
    else:
        fc_cols = [
            "dataset",
            "method",
            "config_profile",
            "baseline_mi_k",
            "candidate_mi_k",
            "effective_radius",
            "effective_n_neighbors",
            "effective_min_act",
            "effective_min_feas",
            "baseline_strict_valid_rate",
            "candidate_strict_valid_rate",
            "strict_valid_rate_delta",
            "strict_valid_verdict",
            "runtime_ratio_candidate_vs_baseline",
            "runtime_verdict",
        ]
        lines.append(final_comparison_df.loc[:, [c for c in fc_cols if c in final_comparison_df.columns]].head(60).to_markdown(index=False))
        if len(final_comparison_df) > 60:
            lines.append("\n... (truncated)")
    lines.extend(
        [
            "",
            "## Stage Comparison Preview",
            "",
        ]
    )
    if stage_comparison_df.empty:
        lines.append("_No posthoc/forceflip stage comparison table was generated._")
    else:
        stage_cols = [
            "dataset",
            "method",
            "posthoc_strict_valid_rate",
            "forceflip_strict_valid_rate",
            "forceflip_vs_posthoc_delta",
            "forceflip_vs_posthoc_verdict",
        ]
        lines.append(stage_comparison_df.loc[:, [c for c in stage_cols if c in stage_comparison_df.columns]].head(60).to_markdown(index=False))
        if len(stage_comparison_df) > 60:
            lines.append("\n... (truncated)")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UFCE-FF experiments with MI[:5] primary and MI expansion diagnostics.")
    parser.add_argument("--dataset", default="all", help="Dataset name, comma-separated list, or all.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "public_posthoc",
            "public_forceflip",
            "pool_expansion_forceflip",
            "pool_expansion_comparison",
            "final_comparison",
            "all",
        ],
        help="Execution mode.",
    )
    parser.add_argument("--mi-k-list", default="5,10,20,all", help="MI-k list for mi expansion diagnostics.")
    parser.add_argument(
        "--config-profile",
        default="public_github_source",
        choices=["public_github_source", "final_freeze", "new_best_params"],
        help="Runtime config source used by candidate generation.",
    )
    parser.add_argument(
        "--bundle-mode",
        default="table7_author_public",
        choices=["author_public", "table7_author_public", "domain_actionable"],
        help="UF/f2change/step bundle source.",
    )
    parser.add_argument("--no_cf", type=int, default=10)
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--fold_file", default=None)
    parser.add_argument("--contprox_metric", default="euclidean")
    parser.add_argument(
        "--mi-feature-scope",
        default="configured_actionable",
        choices=["configured_actionable", "uf_only", "all_features"],
        help="Feature-pair scope for MI expansion. configured_actionable requires uf, step, and f2change.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    datasets = parse_dataset_arg(args.dataset)
    modes = mode_list_from_arg(args.mode)
    mi_k_tokens = parse_mi_k_list(args.mi_k_list)
    progress_enabled = not bool(getattr(args, "no_progress", False))
    config_profile = str(args.config_profile)
    bundle_mode = str(args.bundle_mode)
    output_root = Path(args.out_dir).resolve() if args.out_dir else (DEFAULT_OUT_PARENT / now_tag()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mod01b = load_runner_01b()
    query_rows: List[QueryEval] = []
    run_cache: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}

    dataset_iter = tqdm_wrap(
        datasets,
        total=len(datasets),
        desc="Datasets",
        leave=True,
        disable=not progress_enabled,
    )
    for dataset in dataset_iter:
        if "public_posthoc" in modes:
            key = (dataset, config_profile, bundle_mode, "5", 0)
            if key not in run_cache:
                run_cache[key] = run_dataset_traces(
                    mod01b=mod01b,
                    dataset=dataset,
                    mi_k=5,
                    config_profile=config_profile,
                    bundle_mode=bundle_mode,
                    flip_filter_enabled=False,
                    no_cf=int(args.no_cf),
                    max_folds=int(args.max_folds),
                    fold_file=args.fold_file,
                    contprox_metric=args.contprox_metric,
                    mi_feature_scope=args.mi_feature_scope,
                    progress=progress_enabled,
                )
            query_rows.extend(
                evaluate_mode_from_cache(
                    mod01b=mod01b,
                    cache_obj=run_cache[key],
                    mode="public_posthoc",
                    mi_k_label="5",
                    top_n=0,
                    progress=progress_enabled,
                    progress_desc=f"[{dataset}] public_posthoc",
                )
            )

        if "public_forceflip" in modes:
            key = (dataset, config_profile, bundle_mode, "5", 1)
            if key not in run_cache:
                run_cache[key] = run_dataset_traces(
                    mod01b=mod01b,
                    dataset=dataset,
                    mi_k=5,
                    config_profile=config_profile,
                    bundle_mode=bundle_mode,
                    flip_filter_enabled=True,
                    no_cf=int(args.no_cf),
                    max_folds=int(args.max_folds),
                    fold_file=args.fold_file,
                    contprox_metric=args.contprox_metric,
                    mi_feature_scope=args.mi_feature_scope,
                    progress=progress_enabled,
                )
            query_rows.extend(
                evaluate_mode_from_cache(
                    mod01b=mod01b,
                    cache_obj=run_cache[key],
                    mode="public_forceflip",
                    mi_k_label="5",
                    top_n=0,
                    progress=progress_enabled,
                    progress_desc=f"[{dataset}] public_forceflip",
                )
            )

        if "pool_expansion_forceflip" in modes:
            for token in mi_k_tokens:
                mi_k = None if token == "all" else int(token)
                key = (dataset, config_profile, bundle_mode, str(token), 1)
                if key not in run_cache:
                    run_cache[key] = run_dataset_traces(
                        mod01b=mod01b,
                        dataset=dataset,
                        mi_k=mi_k,
                        config_profile=config_profile,
                        bundle_mode=bundle_mode,
                        flip_filter_enabled=True,
                        no_cf=int(args.no_cf),
                        max_folds=int(args.max_folds),
                        fold_file=args.fold_file,
                        contprox_metric=args.contprox_metric,
                        mi_feature_scope=args.mi_feature_scope,
                        progress=progress_enabled,
                    )
                query_rows.extend(
                    evaluate_mode_from_cache(
                        mod01b=mod01b,
                        cache_obj=run_cache[key],
                        mode="public_forceflip",
                        mi_k_label=str(token),
                        top_n=0,
                        progress=progress_enabled,
                        progress_desc=f"[{dataset}] pool_forceflip@{token}",
                    )
                )

    dataset_iter.close()

    query_df = pd.DataFrame([row.__dict__ for row in query_rows])
    summary_df = summarize_queries(query_df)
    final_comparison_df = build_final_comparison_table(summary_df)
    stage_comparison_df = build_stage_comparison_table(summary_df)
    claim_check = build_claim_check(summary_df)

    failure_df = query_df.loc[
        :,
        [
            "dataset",
            "method",
            "mode",
            "config_profile",
            "bundle_mode",
            "mi_k",
            "top_n",
            "effective_radius",
            "effective_n_neighbors",
            "effective_min_act",
            "effective_min_feas",
            "effective_force_flip",
            "fold_id",
            "query_pos",
            "raw_candidate_count",
            "unique_candidate_count",
            "flip_candidate_count",
            "public_selected_flip",
            "verified_selected_flip",
            "strict_valid_selected",
            "failure_code",
            "failure_detail",
            "predict_call_count",
            "batch_predict_row_count",
            "dedup_reduction_ratio",
            "batch_predict_ms",
            "runtime_ms",
        ],
    ].copy() if not query_df.empty else pd.DataFrame()
    runtime_df = summary_df.loc[
        :,
        [
            "dataset",
            "method",
            "mode",
            "config_profile",
            "bundle_mode",
            "mi_k",
            "top_n",
            "effective_radius",
            "effective_n_neighbors",
            "effective_min_act",
            "effective_min_feas",
            "effective_force_flip",
            "query_count",
            "mean_runtime_ms",
            "runtime_ratio",
            "mean_batch_predict_ms",
            "predict_call_count",
            "batch_predict_row_count",
        ],
    ].copy() if not summary_df.empty else pd.DataFrame()
    pool_diag_df = summary_df.loc[
        (summary_df["mode"] == "public_forceflip") & (summary_df["mi_k"].astype(str) != "5")
    ].copy() if not summary_df.empty else pd.DataFrame()

    failure_df.to_csv(output_root / "ff_failure_taxonomy.csv", index=False)
    summary_df.to_csv(output_root / "ff_selector_comparison.csv", index=False)
    runtime_df.to_csv(output_root / "ff_runtime_summary.csv", index=False)
    pool_diag_df.to_csv(output_root / "pool_expansion_diagnostic.csv", index=False)
    final_comparison_df.to_csv(output_root / "ff_final_comparison.csv", index=False)
    stage_comparison_df.to_csv(output_root / "ff_stage_comparison.csv", index=False)
    (output_root / "ff_claim_check.json").write_text(json.dumps(claim_check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(
        render_summary_md(output_root, query_df, summary_df, final_comparison_df, stage_comparison_df, claim_check),
        encoding="utf-8",
    )

    summary_json = {
        "status": "ok",
        "output_root": str(output_root),
        "config_profile": config_profile,
        "bundle_mode": bundle_mode,
        "mi_feature_scope": str(args.mi_feature_scope),
        "public_github_source_config": dict(PUBLIC_GITHUB_SOURCE_CONFIG),
        "ff_failure_taxonomy_csv": str(output_root / "ff_failure_taxonomy.csv"),
        "ff_selector_comparison_csv": str(output_root / "ff_selector_comparison.csv"),
        "ff_runtime_summary_csv": str(output_root / "ff_runtime_summary.csv"),
        "pool_expansion_diagnostic_csv": str(output_root / "pool_expansion_diagnostic.csv"),
        "ff_final_comparison_csv": str(output_root / "ff_final_comparison.csv"),
        "ff_stage_comparison_csv": str(output_root / "ff_stage_comparison.csv"),
        "ff_claim_check_json": str(output_root / "ff_claim_check.json"),
        "summary_md": str(output_root / "summary.md"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_json, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
