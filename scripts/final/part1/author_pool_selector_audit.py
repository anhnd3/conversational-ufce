#!/usr/bin/env python3
"""Paired selector audit on the original UFCE author candidate pools.

The generator is loaded from ``ufce/core_author``.  For every query we generate
one author pool and apply two selectors to that exact pool:

* ``author_raw``: the author nearest-row selector, without a label-validity gate;
* ``author_pool_forceflip``: filter the same pool by ``desired_outcome`` and then
  run the same nearest-row selector.

``author_posthoc_valid_only`` is an evaluation view of ``author_raw``; it does
not alter selection. Runtime/search parameters come from the requested
UFCE-only reproduction profile, while UF/f2change/step come from the requested
bundle.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE_08 = ROOT / "scripts" / "final" / "part1" / "author_raw_posthoc_replay.py"
OUT_PARENT = ROOT / "outputs" / "final" / "part1"
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
SCOPES = ["author_raw", "author_posthoc_valid_only", "author_pool_forceflip"]


def load_base08():
    spec = importlib.util.spec_from_file_location("ufce_author_raw_posthoc_base", str(BASE_08))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {BASE_08}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_runtime_config(mod01b, dataset: str, profile: str) -> Dict[str, int]:
    key = str(profile).strip().lower()
    profiles = {
        "final_freeze": mod01b.FINAL_RUNTIME_CONFIG,
        "new_best_params": mod01b.NEW_BEST_PARAMS,
        "tuned_run2": mod01b.TUNED_RUN2,
    }
    if key not in profiles:
        raise ValueError(f"Unsupported config profile: {profile}")
    cfg = dict(profiles[key][dataset])
    return {
        "radius": int(cfg["radius"]),
        "n_neighbors": int(cfg["n_neighbors"]),
        "min_act": int(cfg["min_act"]),
        "min_feas": int(cfg["min_feas"]),
        "ufce_flip_filter": 0,
    }


def _clean_pool(pool: Any, features: Sequence[str]) -> pd.DataFrame:
    if not isinstance(pool, pd.DataFrame) or pool.empty:
        return pd.DataFrame(columns=list(features))
    if not all(feature in pool.columns for feature in features):
        return pd.DataFrame(columns=list(features))
    return pool.loc[:, list(features)].copy().reset_index(drop=True)


def filter_desired(pool: pd.DataFrame, model, desired_outcome: float, features: Sequence[str]) -> pd.DataFrame:
    pool = _clean_pool(pool, features)
    if pool.empty:
        return pool
    preds = np.asarray(model.predict(pool.loc[:, list(features)])).reshape(-1)
    return pool.loc[preds == int(desired_outcome)].reset_index(drop=True)


def _apply_distance_scaler_no_clip(df: pd.DataFrame, distance_scaler: Optional[Dict[str, object]]) -> pd.DataFrame:
    """Match 01b.apply_distance_scaler for Movie search/selector space.

    The old Movie raw-reproduction path scales numeric distance columns with
    ``(value - min) / ((max - min) / 100)`` for KDTree/search and nearest-row
    selection, but it does not clip values to [0, 100] there.  The clipping
    remains part of the Prox-Euc metric helper, not this selector/search path.
    """
    if distance_scaler is None:
        return df.copy()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.copy()

    out = df.copy()
    scale_cols = distance_scaler.get("scale_cols", None)
    if scale_cols is None:
        scale_cols = list(getattr(distance_scaler.get("medians", pd.Series(dtype=float)), "index", []))
    cols = [col for col in scale_cols if col in out.columns]
    if not cols:
        return out

    out.loc[:, cols] = (
        out.loc[:, cols] - distance_scaler["medians"].reindex(cols)
    ) / distance_scaler["mads"].reindex(cols)
    constant_cols = [col for col in distance_scaler.get("constant_cols", []) if col in cols]
    if constant_cols:
        out.loc[:, constant_cols] = 0.0
    return out


def _selector_distance_scaler(context: Dict[str, Any]) -> Optional[Dict[str, object]]:
    if str(context.get("dataset", "")).lower() == "movie":
        scaler = context.get("movie_distance_scaler")
        if isinstance(scaler, dict):
            return scaler
    return None


def _select_best_scaled(
    pool: pd.DataFrame,
    factual: pd.DataFrame,
    numf: Sequence[str],
    distance_scaler: Dict[str, object],
) -> pd.Series:
    dist_pool = _apply_distance_scaler_no_clip(pool.copy(), distance_scaler)
    dist_factual = _apply_distance_scaler_no_clip(factual.copy(), distance_scaler)
    cols = [col for col in numf if col in dist_pool.columns and col in dist_factual.columns]
    if not cols:
        return pool.iloc[0].copy()
    factual_values = dist_factual.iloc[0].loc[cols].to_numpy(dtype=float)
    distances = dist_pool.loc[:, cols].apply(
        lambda row: float(np.linalg.norm(row.to_numpy(dtype=float) - factual_values)),
        axis=1,
    )
    best_idx = distances.idxmin()
    best_row = pool.loc[best_idx].copy()
    best_row["proximity"] = float(distances.loc[best_idx])
    return best_row


def nearest_author_neighbors(
    author_ufc,
    context: Dict[str, Any],
    factual: pd.DataFrame,
    radius: int,
) -> Tuple[pd.DataFrame, List[int]]:
    """Run author KDTree, using 01b Movie minmax distance space when needed.

    For Movie, this mirrors the old raw reproduction contract: search in scaled
    distance space, then map the returned indices back to raw rows before UFCE
    interval construction and candidate generation.
    """
    features = context["features"]
    distance_scaler = _selector_distance_scaler(context)
    if distance_scaler is None:
        nn, idx = author_ufc.NNkdtree(context["data_lab1"], factual, int(radius))
        return nn, [int(i) for i in idx]

    raw_data_lab1 = context["data_lab1"].loc[:, list(features)].copy().reset_index(drop=True)
    raw_factual = factual.loc[:, list(features)].copy().reset_index(drop=True)
    distance_data_lab1 = _apply_distance_scaler_no_clip(raw_data_lab1, distance_scaler)
    distance_factual = _apply_distance_scaler_no_clip(raw_factual, distance_scaler)
    _nn_dist, idx = author_ufc.NNkdtree(distance_data_lab1, distance_factual, int(radius))
    idx_list = [int(i) for i in idx]
    nn = raw_data_lab1.iloc[idx_list].reset_index(drop=True)
    return nn, idx_list


def select_best(
    author_cfmethods,
    pool: pd.DataFrame,
    factual: pd.DataFrame,
    numf: Sequence[str],
    distance_scaler: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    if not isinstance(pool, pd.DataFrame) or pool.empty:
        return pd.DataFrame(columns=[c for c in factual.columns])
    if len(pool) == 1:
        return pool.iloc[[0]].copy().reset_index(drop=True)
    if distance_scaler is None:
        best = author_cfmethods.find_best_row(pool.copy(), factual.copy(), list(numf))
    else:
        best = _select_best_scaled(pool.copy(), factual.copy(), list(numf), distance_scaler)
    selected = best.to_frame().T
    if "proximity" in selected.columns:
        selected = selected.drop(columns=["proximity"])
    return selected.loc[:, [c for c in factual.columns if c in selected.columns]].reset_index(drop=True)


def select_author_raw(
    author_cfmethods,
    primary_pool: pd.DataFrame,
    explore_pool: pd.DataFrame,
    factual: pd.DataFrame,
    numf: Sequence[str],
    model,
    desired_outcome: float,
    features: Sequence[str],
    method: str,
    distance_scaler: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Mirror the author selector: primary nearest row, then UFCE2 fallback."""
    primary_pool = _clean_pool(primary_pool, features)
    explore_pool = _clean_pool(explore_pool, features)
    if not primary_pool.empty:
        return select_best(author_cfmethods, primary_pool, factual, numf, distance_scaler), "primary_raw"
    if method == "UFCE2" and not explore_pool.empty:
        # This is the original dfexp fallback at core_author/cfmethods.py:267-272.
        fallback = filter_desired(explore_pool, model, desired_outcome, features)
        if not fallback.empty:
            return fallback.iloc[[0]].reset_index(drop=True), "explore_author_fallback"
    return pd.DataFrame(columns=list(features)), "none"


def select_forceflip(
    author_cfmethods,
    primary_pool: pd.DataFrame,
    explore_pool: pd.DataFrame,
    factual: pd.DataFrame,
    numf: Sequence[str],
    model,
    desired_outcome: float,
    features: Sequence[str],
    distance_scaler: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Apply validity immediately after generation, before nearest-row selection."""
    valid_primary = filter_desired(primary_pool, model, desired_outcome, features)
    if not valid_primary.empty:
        return select_best(author_cfmethods, valid_primary, factual, numf, distance_scaler), "primary_forceflip"
    valid_explore = filter_desired(explore_pool, model, desired_outcome, features)
    if not valid_explore.empty:
        return select_best(author_cfmethods, valid_explore, factual, numf, distance_scaler), "explore_forceflip"
    return pd.DataFrame(columns=list(features)), "none"


def candidate_record(selected: pd.DataFrame, model, desired_outcome: float, features: Sequence[str]) -> Dict[str, Any]:
    exists = isinstance(selected, pd.DataFrame) and not selected.empty
    pred: Optional[int] = None
    if exists:
        values = np.asarray(model.predict(selected.loc[:, list(features)])).reshape(-1)
        if values.size:
            pred = int(values[0])
    return {
        "selected_exists": int(exists),
        "pred_label": pred,
        "strict_valid_selected": int(exists and pred == int(desired_outcome)),
    }


def generate_author_pool(
    *,
    author_ufc,
    method: str,
    context: Dict[str, Any],
    factual: pd.DataFrame,
    cfg: Dict[str, int],
    mi_pairs: Sequence[Sequence[str]],
    no_cf: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    features = context["features"]
    nn, idx = nearest_author_neighbors(author_ufc, context, factual, int(cfg["radius"]))
    if not isinstance(nn, pd.DataFrame) or nn.empty:
        return pd.DataFrame(columns=features), pd.DataFrame(columns=features), int(len(idx))

    if method == "UFCE1":
        intervals = author_ufc.make_intervals(nn, context["uf"], context["f2change"], factual)
        primary = author_ufc.Single_F(
            factual,
            context["catf"],
            intervals,
            context["lr"],
            context["desired_outcome"],
            context["step"],
        )
        return _clean_pool(primary, features), pd.DataFrame(columns=features), int(len(idx))

    intervals = author_ufc.make_uf_nn_interval(nn, context["uf"], list(mi_pairs), factual)
    if method == "UFCE2":
        primary, explore = author_ufc.Double_F(
            context["x_all"], factual, context["protectf"], list(mi_pairs),
            context["catf"], context["numf"], intervals, context["f2change"],
            context["lr"], context["desired_outcome"], features, int(no_cf),
        )
    elif method == "UFCE3":
        primary, explore = author_ufc.Triple_F(
            context["x_all"], factual, context["protectf"], list(mi_pairs),
            context["catf"], context["numf"], intervals, context["f2change"],
            context["lr"], context["desired_outcome"], features, int(no_cf),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return _clean_pool(primary, features), _clean_pool(explore, features), int(len(idx))


def metric_frame(records: List[Dict[str, Any]], scope: str, features: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: List[pd.DataFrame] = []
    factual_rows: List[pd.DataFrame] = []
    for record in records:
        selected = record["raw_selected"] if scope != "author_pool_forceflip" else record["ff_selected"]
        if not isinstance(selected, pd.DataFrame) or selected.empty:
            continue
        if scope == "author_posthoc_valid_only" and int(record["raw_valid"]) != 1:
            continue
        selected_rows.append(selected.loc[:, list(features)].copy())
        factual_rows.append(record["factual"].loc[:, list(features)].copy())
    cfdf = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=list(features))
    testdf = pd.concat(factual_rows, ignore_index=True) if factual_rows else pd.DataFrame(columns=list(features))
    return cfdf, testdf


def selected_for_scope(record: Dict[str, Any], scope: str) -> Tuple[pd.DataFrame, str, Optional[int], int]:
    if scope == "author_pool_forceflip":
        return (
            record["ff_selected"],
            str(record["ff_selected_source"]),
            record["ff_pred_label"],
            int(record["ff_valid"]),
        )
    return (
        record["raw_selected"],
        str(record["raw_selected_source"]),
        record["raw_pred_label"],
        int(record["raw_valid"]),
    )


def changed_features(factual: pd.DataFrame, selected: pd.DataFrame, features: Sequence[str]) -> List[str]:
    changed: List[str] = []
    for feature in features:
        before = factual.iloc[0][feature]
        after = selected.iloc[0][feature]
        try:
            is_changed = not bool(np.isclose(float(before), float(after), rtol=1e-09, atol=1e-12))
        except (TypeError, ValueError):
            is_changed = bool(before != after)
        if is_changed:
            changed.append(str(feature))
    return changed


def pair_rows_for_scope(records: List[Dict[str, Any]], scope: str, features: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        selected, selected_source, pred_label, strict_valid = selected_for_scope(record, scope)
        if not isinstance(selected, pd.DataFrame) or selected.empty:
            continue
        if scope == "author_posthoc_valid_only" and strict_valid != 1:
            continue
        factual = record["factual"]
        changed = changed_features(factual, selected, features)
        row: Dict[str, Any] = {
            "dataset": record["dataset"],
            "fold_id": record["fold_id"],
            "query_pos": int(record["query_pos"]),
            "method": record["method"],
            "output_set": scope,
            "config_profile": record["config_profile"],
            "bundle_mode": record["bundle_mode"],
            "effective_radius": int(record["effective_radius"]),
            "effective_n_neighbors": int(record["effective_n_neighbors"]),
            "mi_k": int(record["mi_k"]),
            "distance_space": record["distance_space"],
            "selected_source": selected_source,
            "pred_label": pred_label,
            "strict_valid_selected": int(strict_valid),
            "changed_feature_count": int(len(changed)),
            "changed_features": "|".join(changed),
        }
        for feature in features:
            row[f"x::{feature}"] = factual.iloc[0][feature]
            row[f"cf::{feature}"] = selected.iloc[0][feature]
        rows.append(row)
    return rows


def summarize_queries(query_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (dataset, method), group in query_df.groupby(["dataset", "method"], dropna=False):
        n = int(len(group))
        raw_selected = int(group["raw_selected_exists"].sum())
        raw_valid = int(group["raw_strict_valid_selected"].sum())
        ff_selected = int(group["ff_selected_exists"].sum())
        ff_valid = int(group["ff_strict_valid_selected"].sum())
        rows.append({
            "dataset": dataset,
            "method": method,
            "query_count": n,
            "raw_selected_count": raw_selected,
            "raw_valid_count": raw_valid,
            "raw_invalid_selected_count": raw_selected - raw_valid,
            "raw_strict_valid_rate": float(raw_valid / n) if n else 0.0,
            "forceflip_selected_count": ff_selected,
            "forceflip_valid_count": ff_valid,
            "forceflip_invalid_selected_count": ff_selected - ff_valid,
            "forceflip_strict_valid_rate": float(ff_valid / n) if n else 0.0,
            "forceflip_valid_rate_delta": float((ff_valid - raw_valid) / n) if n else 0.0,
            "invalid_outputs_removed": int((raw_selected - raw_valid) - (ff_selected - ff_valid)),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and force-flip selectors on original author pools.")
    parser.add_argument("--dataset", default="bank,bupa,grad,wine")
    parser.add_argument("--config-profile", default="final_freeze", choices=["final_freeze", "new_best_params", "tuned_run2"])
    parser.add_argument("--bundle-mode", default="table7_author_public")
    parser.add_argument("--mi-k", type=int, default=5)
    parser.add_argument("--no-cf", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0, help="Debug-only cap per fold; 0 runs every query.")
    parser.add_argument("--fold-file", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--emit-pairs", action="store_true", help="Write row-level factual/CF pairs for diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_base08()
    datasets = base.parse_dataset_arg(args.dataset)
    output_root = Path(args.out_dir).resolve() if args.out_dir else (OUT_PARENT / "author_pool_selector_audit").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mod01b = base.load_runner_01b()
    author_modules = base.load_author_modules()
    author_ufc = author_modules.cfmethods.ufc
    query_rows: List[Dict[str, Any]] = []
    metric_fold_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        context = base.prepare_dataset_context(mod01b, dataset, args.bundle_mode)
        cfg = resolve_runtime_config(mod01b, dataset, args.config_profile)
        mi_pairs = context["mi_fp"][: int(args.mi_k)]
        fold_dir = ROOT / "ufce" / "data" / "folds" / dataset / "totest"
        folds = sorted(fold_dir.glob("*.csv"))
        if args.fold_file:
            folds = [p for p in folds if p.name == os.path.basename(str(args.fold_file))]
        if int(args.max_folds) > 0:
            folds = folds[: int(args.max_folds)]
        if not folds:
            raise FileNotFoundError(f"No folds selected for {dataset}: {fold_dir}")

        for fold_path in folds:
            seed = int(hashlib.sha256(f"{dataset}:{fold_path.name}:42".encode("utf-8")).hexdigest()[:8], 16)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            fold_df = pd.read_csv(fold_path)
            fold_features = fold_df.loc[:, context["features"]].reset_index(drop=True)
            if int(args.max_queries) > 0:
                fold_features = fold_features.iloc[: int(args.max_queries)].reset_index(drop=True)
            per_method_records: Dict[str, List[Dict[str, Any]]] = {method: [] for method in METHODS}
            selector_distance_scaler = _selector_distance_scaler(context)
            distance_space = "movie_minmax_0_100_no_clip_search_selector" if selector_distance_scaler is not None else "raw"

            for method in METHODS:
                method_start = time.perf_counter()
                for query_pos in range(len(fold_features)):
                    factual = fold_features.iloc[[query_pos]].reset_index(drop=True)
                    primary, explore, neighbor_count = generate_author_pool(
                        author_ufc=author_ufc,
                        method=method,
                        context=context,
                        factual=factual,
                        cfg=cfg,
                        mi_pairs=mi_pairs,
                        no_cf=int(args.no_cf),
                    )
                    raw_selected, raw_source = select_author_raw(
                        author_modules.cfmethods, primary, explore, factual, context["numf"],
                        context["lr"], context["desired_outcome"], context["features"], method,
                        selector_distance_scaler,
                    )
                    ff_selected, ff_source = select_forceflip(
                        author_modules.cfmethods, primary, explore, factual, context["numf"],
                        context["lr"], context["desired_outcome"], context["features"],
                        selector_distance_scaler,
                    )
                    raw_rec = candidate_record(raw_selected, context["lr"], context["desired_outcome"], context["features"])
                    ff_rec = candidate_record(ff_selected, context["lr"], context["desired_outcome"], context["features"])
                    flip_primary = filter_desired(primary, context["lr"], context["desired_outcome"], context["features"])
                    flip_explore = filter_desired(explore, context["lr"], context["desired_outcome"], context["features"])
                    record = {
                        "dataset": dataset,
                        "fold_id": fold_path.name,
                        "query_pos": int(query_pos),
                        "method": method,
                        "config_profile": str(args.config_profile),
                        "bundle_mode": str(context["bundle"].effective_bundle_mode),
                        "effective_radius": int(cfg["radius"]),
                        "effective_n_neighbors": int(cfg["n_neighbors"]),
                        "mi_k": int(args.mi_k),
                        "distance_space": distance_space,
                        "neighbor_count": int(neighbor_count),
                        "primary_candidate_count": int(len(primary)),
                        "explore_candidate_count": int(len(explore)),
                        "primary_flip_count": int(len(flip_primary)),
                        "explore_flip_count": int(len(flip_explore)),
                        "raw_selected_exists": int(raw_rec["selected_exists"]),
                        "raw_pred_label": raw_rec["pred_label"],
                        "raw_strict_valid_selected": int(raw_rec["strict_valid_selected"]),
                        "raw_selected_source": raw_source,
                        "ff_selected_exists": int(ff_rec["selected_exists"]),
                        "ff_pred_label": ff_rec["pred_label"],
                        "ff_strict_valid_selected": int(ff_rec["strict_valid_selected"]),
                        "ff_selected_source": ff_source,
                    }
                    query_rows.append(record)
                    per_method_records[method].append({
                        **record,
                        "factual": factual,
                        "raw_selected": raw_selected,
                        "ff_selected": ff_selected,
                        "raw_valid": int(raw_rec["strict_valid_selected"]),
                        "ff_valid": int(ff_rec["strict_valid_selected"]),
                        "raw_pred_label": raw_rec["pred_label"],
                        "ff_pred_label": ff_rec["pred_label"],
                        "raw_selected_source": raw_source,
                        "ff_selected_source": ff_source,
                    })
                runtime_ms = float((time.perf_counter() - method_start) * 1000.0 / max(1, len(fold_features)))

                records = per_method_records[method]
                raw_selected_count = sum(int(not r["raw_selected"].empty) for r in records)
                raw_valid_count = sum(int(r["raw_valid"]) for r in records)
                ff_selected_count = sum(int(not r["ff_selected"].empty) for r in records)
                ff_valid_count = sum(int(r["ff_valid"]) for r in records)
                for scope in SCOPES:
                    cfdf, testdf = metric_frame(records, scope, context["features"])
                    if bool(args.emit_pairs):
                        pair_rows.extend(pair_rows_for_scope(records, scope, context["features"]))
                    metrics = base.compute_six_metrics(
                        author_ufc=author_ufc,
                        mod01b=mod01b,
                        method_name=method,
                        dataset=dataset,
                        cfdf=cfdf,
                        testdf=testdf,
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
                    if scope == "author_pool_forceflip":
                        selected_count, valid_count = ff_selected_count, ff_valid_count
                    else:
                        selected_count, valid_count = raw_selected_count, raw_valid_count
                    metric_count = int(len(cfdf))
                    metric_fold_rows.append({
                        "dataset": dataset,
                        "fold_id": fold_path.name,
                        "method": method,
                        "metric_scope": scope,
                        "source_core": "ufce/core_author",
                        "config_profile": str(args.config_profile),
                        "bundle_mode": str(context["bundle"].effective_bundle_mode),
                        "effective_radius": int(cfg["radius"]),
                        "effective_n_neighbors": int(cfg["n_neighbors"]),
                        "distance_space": distance_space,
                        "query_count": int(len(fold_features)),
                        "selected_count": int(selected_count),
                        "selected_valid_count": int(valid_count),
                        "selected_posthoc_fail_count": int(selected_count - valid_count),
                        "missing_count": int(len(fold_features) - selected_count),
                        "metric_candidate_count": metric_count,
                        "runtime_ms_per_query": runtime_ms,
                        **metrics,
                    })

    query_df = pd.DataFrame(query_rows)
    summary_df = summarize_queries(query_df)
    metric_fold_df = pd.DataFrame(metric_fold_rows)
    metric_summary_df, metric_long_df = base.summarize_metric_rows(metric_fold_df)

    query_df.to_csv(output_root / "author_pool_query.csv", index=False)
    summary_df.to_csv(output_root / "author_pool_selector_summary.csv", index=False)
    metric_fold_df.to_csv(output_root / "author_pool_metric_fold.csv", index=False)
    metric_summary_df.to_csv(output_root / "author_pool_metric_summary.csv", index=False)
    metric_long_df.to_csv(output_root / "author_pool_metric_summary_long.csv", index=False)
    if bool(args.emit_pairs):
        pd.DataFrame(pair_rows).to_csv(output_root / "author_pool_pairs.csv", index=False)

    metadata = {
        "status": "ok",
        "source_core": "ufce/core_author",
        "selector_contract": "same_author_pool_raw_vs_filter_desired_then_find_best_row",
        "config_profile": str(args.config_profile),
        "bundle_mode": str(args.bundle_mode),
        "datasets": datasets,
        "mi_k": int(args.mi_k),
        "no_cf": int(args.no_cf),
        "max_queries": int(args.max_queries),
        "emit_pairs": bool(args.emit_pairs),
        "pair_artifact": "author_pool_pairs.csv" if bool(args.emit_pairs) else None,
        "note": "n_neighbors is recorded from final_freeze but original author KDTree is radius-only.",
        "movie_distance_contract": (
            "For dataset=movie, KDTree/search and nearest-row selectors use the same no-clip "
            "minmax-0-100 distance space as UFCE-only raw Movie reproduction; CF generation remains raw. "
            "Prox-Euc metrics still use the RMS movie_minmax_0_100 metric helper."
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if summary_df.empty:
        preview = "_No rows._"
    else:
        try:
            preview = summary_df.to_markdown(index=False)
        except ImportError:
            preview = summary_df.to_string(index=False)
    (output_root / "summary.md").write_text(
        "# Original-author pool selector experiment\n\n"
        f"- source_core: `ufce/core_author`\n"
        f"- config_profile: `{args.config_profile}`\n"
        f"- bundle_mode: `{args.bundle_mode}`\n"
        "- contract: generate once; raw nearest-row vs desired-label filter then nearest-row\n\n"
        "## Validity summary\n\n" + preview + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Original-author paired selector outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
