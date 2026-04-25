#!/usr/bin/env python3
"""
P7-A style UFCE trace harness for investigating suspicious points in the paper
and reproduction pipeline.

Default behavior uses the author-style `totest/testfold_*_pred_0.csv` inputs so
the harness can be pointed at the same screened queries used in reproduction.
It can also screen the full dataset directly.

Outputs:
- ENV.md
- run_meta.json
- traces/grouped_by_query.json
- traces/trace_rows.jsonl
- traces/trace_<dataset>.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ufce
from ufce import UFCE
from ufce.core import cfmethods
from ufce.core import evaluations as eval_module
from ufce.core.cfmethods import dfexp, sfexp, tfexp
from ufce.core.data_processing import (
    classify_dataset_getModel,
    get_bank_user_constraints,
    get_bupa_user_constraints,
    get_grad_user_constraints,
    get_movie_user_constraints,
    get_wine_user_constraints,
)


DATASET_FILES = {
    "bank": "ufce/data/bank.csv",
    "grad": "ufce/data/grad.csv",
    "wine": "ufce/data/wine.csv",
    "bupa": "ufce/data/bupa.csv",
    "movie": "ufce/data/movie.csv",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "unknown"


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def value_changed(a, b, atol: float = 1e-9) -> bool:
    try:
        return not np.isclose(float(a), float(b), atol=atol, rtol=0.0)
    except Exception:
        return str(a) != str(b)


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


def build_movie_distance_scaler(
    datasetdf: pd.DataFrame,
    features: List[str],
    numf: List[str],
    outcome_label: str,
) -> Dict[str, object]:
    cols = [c for c in numf if c in features and c in datasetdf.columns and c != outcome_label]
    if len(cols) == 0:
        raise ValueError("Movie distance scaler could not find numeric columns to scale.")
    base = datasetdf.loc[:, features].copy()
    mins = base[cols].min()
    maxs = base[cols].max()
    ranges = maxs - mins
    constant_cols = [c for c in cols if float(ranges[c]) == 0.0]
    return {
        "kind": "movie_minmax_0_100",
        "scale_cols": cols,
        "medians": mins.astype(float),
        "mads": (ranges / 100.0).replace(0.0, 1.0).astype(float),
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
    if constant_cols:
        out.loc[:, constant_cols] = 0.0
    return out


def resolve_column(df: pd.DataFrame, target: str) -> Optional[str]:
    if target in df.columns:
        return target
    norm_target = "".join(ch for ch in str(target).lower() if ch.isalnum())
    for col in df.columns:
        norm_col = "".join(ch for ch in str(col).lower() if ch.isalnum())
        if norm_col == norm_target:
            return col
    return None


def align_feature_frame(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    cols: List[str] = []
    missing: List[str] = []
    for feat in features:
        col = resolve_column(df, feat)
        if col is None:
            missing.append(feat)
        else:
            cols.append(col)
    if missing:
        raise KeyError(f"Missing features: {missing}")
    out = df[cols].copy()
    out.columns = features
    return out


def init_runtime(radius: int, n_neighbors: int, contprox_metric: str, min_act: int, min_feas: int, atol: float) -> None:
    cfmethods.initUFCE(
        radius=radius,
        n_neighbors=n_neighbors,
        contprox_metric=contprox_metric,
        min_act=min_act,
        min_feas=min_feas,
        atol=atol,
    )
    eval_module.ufc = cfmethods.ufc


def log_env(out_dir: str) -> None:
    lines = [
        "# Environment",
        "",
        f"- timestamp_utc: {now_iso()}",
        f"- platform: {platform.platform()}",
        f"- python: {sys.version.split()[0]}",
        f"- numpy: {safe_version('numpy')}",
        f"- pandas: {safe_version('pandas')}",
        f"- scipy: {safe_version('scipy')}",
        f"- scikit-learn: {safe_version('scikit-learn')}",
        f"- ufce: {getattr(ufce, '__version__', 'unknown')}",
        "",
    ]
    with open(os.path.join(out_dir, "ENV.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def load_dataset_context(dataset: str, data_dir: str) -> Dict[str, object]:
    data_path = os.path.join(data_dir, f"{dataset}.csv")
    datasetdf = pd.read_csv(data_path)
    out = classify_dataset_getModel(datasetdf, data_name=dataset)
    scaler = None
    if len(out) == 8:
        lr, lr_mean, lr_std, _xtest, _xtrain, x_all, _y, datasetdf = out
    elif len(out) == 9:
        lr, lr_mean, lr_std, _xtest, _xtrain, x_all, _y, datasetdf, scaler = out
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
        data_lab0,
        data_lab1,
    ) = get_dataset_constraints(dataset, datasetdf)

    movie_distance_scaler = None
    if dataset == "movie":
        movie_distance_scaler = build_movie_distance_scaler(
            datasetdf=datasetdf,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )

    ufc = UFCE()
    mi_pairs = ufc.get_top_MI_features(x_all, features)

    return {
        "dataset": dataset,
        "datasetdf": datasetdf,
        "lr": lr,
        "lr_mean": lr_mean,
        "lr_std": lr_std,
        "x_all": x_all,
        "features": features,
        "catf": catf,
        "numf": numf,
        "uf": uf,
        "f2change": f2change,
        "outcome_label": outcome_label,
        "desired_outcome": desired_outcome,
        "protectf": protectf,
        "data_lab0": data_lab0,
        "data_lab1": data_lab1,
        "step": get_step_config(dataset),
        "mi_pairs_top5": mi_pairs[:5],
        "pipeline_scaler": scaler,
        "movie_distance_scaler": movie_distance_scaler,
    }


def load_queries_from_totest(dataset: str, folds_dir: str, features: List[str]) -> Tuple[pd.DataFrame, List[dict]]:
    pattern = os.path.join(folds_dir, dataset, "totest", "testfold_*_pred_0.csv")
    rows: List[pd.DataFrame] = []
    meta: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        raw = pd.read_csv(path)
        aligned = align_feature_frame(raw, features)
        for row_idx in range(len(aligned)):
            rows.append(aligned.iloc[[row_idx]].reset_index(drop=True))
            meta.append(
                {
                    "query_id": f"{os.path.basename(path)}:{row_idx}",
                    "source_file": os.path.basename(path),
                    "source_row_in_file": int(row_idx),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No totest pred_0 folds found for dataset='{dataset}' in {pattern}")
    return pd.concat(rows, ignore_index=True), meta


def load_queries_from_full_dataset(
    datasetdf: pd.DataFrame,
    lr,
    features: List[str],
    outcome_label: str,
) -> Tuple[pd.DataFrame, List[dict]]:
    feature_df = datasetdf.loc[:, features].copy()
    preds = np.asarray(lr.predict(feature_df)).reshape(-1)
    author_pred0_label = 1 if outcome_label == "Selector" else 0
    if outcome_label in datasetdf.columns:
        labels = np.asarray(datasetdf[outcome_label]).reshape(-1)
        mask = np.logical_and(preds == author_pred0_label, labels == author_pred0_label)
    else:
        mask = preds == author_pred0_label
    selected = feature_df.loc[mask].reset_index(drop=True)
    source_indices = datasetdf.index[mask].tolist()
    meta = [
        {"query_id": str(src_idx), "source_dataset_index": int(src_idx)}
        for src_idx in source_indices
    ]
    return selected, meta


def filter_queries(query_df: pd.DataFrame, query_meta: List[dict], query_ids: List[str], query_limit: Optional[int]) -> Tuple[pd.DataFrame, List[dict]]:
    if query_ids:
        wanted = set(str(v) for v in query_ids)
        keep_positions = [idx for idx, meta in enumerate(query_meta) if str(meta["query_id"]) in wanted]
        if not keep_positions:
            raise ValueError(f"Requested query ids were not found: {sorted(wanted)}")
        query_df = query_df.iloc[keep_positions].reset_index(drop=True)
        query_meta = [query_meta[idx] for idx in keep_positions]
    if query_limit is not None:
        query_df = query_df.iloc[:query_limit].reset_index(drop=True)
        query_meta = query_meta[:query_limit]
    return query_df, query_meta


def group_events_by_instance(events: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for event in events:
        payload = event.get("payload", {})
        instance_pos = payload.get("instance_pos", payload.get("t", None))
        if instance_pos is None:
            continue
        grouped[int(instance_pos)].append(event)
    return grouped


def candidate_records(
    df: pd.DataFrame,
    factual_row: pd.DataFrame,
    features: List[str],
    model,
    desired_outcome: float,
    cap: int,
) -> List[dict]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    rows: List[dict] = []
    factual = factual_row.iloc[0]
    pred_input = df.loc[:, [c for c in features if c in df.columns]]
    preds = np.asarray(model.predict(pred_input)).reshape(-1)
    for rank, (_, row) in enumerate(df.head(cap).iterrows(), start=1):
        changed = [feat for feat in features if feat in row.index and value_changed(row[feat], factual[feat])]
        rec = {
            "rank": int(rank),
            "profile": {feat: to_jsonable(row.get(feat, None)) for feat in features if feat in row.index},
            "prediction": int(preds[rank - 1]) if len(preds) >= rank else None,
            "flips_label": bool(len(preds) >= rank and int(preds[rank - 1]) == int(desired_outcome)),
            "changed_features": changed,
            "changed_feature_count": int(len(changed)),
            "proximity": to_jsonable(row.get("proximity", None)),
        }
        rows.append(rec)
    return rows


def selected_prediction(records: List[dict]) -> Optional[int]:
    if not records:
        return None
    pred = records[0].get("prediction", None)
    return int(pred) if pred is not None else None


def invariant_violations(
    method: str,
    selected_records: List[dict],
    factual_pred: int,
    desired_outcome: float,
    allowed_features: List[str],
    flip_filter_enabled: bool,
) -> List[str]:
    limits = {"UFCE1": 1, "UFCE2": 2, "UFCE3": 3}
    violations: List[str] = []
    if factual_pred == int(desired_outcome):
        violations.append("factual_already_in_desired_class")
    if not selected_records:
        return violations
    selected = selected_records[0]
    if flip_filter_enabled and selected.get("prediction") != int(desired_outcome):
        violations.append("selected_candidate_does_not_flip_under_flip_filter")
    if selected.get("changed_feature_count", 0) > limits[method]:
        violations.append(f"changed_feature_count_exceeds_{limits[method]}")
    invalid_changed = [feat for feat in selected.get("changed_features", []) if feat not in allowed_features]
    if invalid_changed:
        violations.append(f"changed_disallowed_features={invalid_changed}")
    return violations


def run_methods(
    *,
    context: Dict[str, object],
    query_df: pd.DataFrame,
    query_meta: List[dict],
    radius: int,
    n_neighbors: int,
    min_act: int,
    min_feas: int,
    contprox_metric: str,
    atol: float,
    flip_filter_enabled: bool,
    candidate_cap: int,
    ufce1_debug: bool,
) -> Tuple[dict, List[dict], Dict[str, dict]]:
    init_runtime(radius, n_neighbors, contprox_metric, min_act, min_feas, atol)
    features = context["features"]
    numf = context["numf"]
    catf = context["catf"]
    uf = context["uf"]
    f2change = context["f2change"]
    desired_outcome = context["desired_outcome"]
    protectf = context["protectf"]
    data_lab1 = context["data_lab1"]
    x_all = context["x_all"]
    lr = context["lr"]
    step = context["step"]
    mi_pairs_top5 = context["mi_pairs_top5"]
    movie_distance_scaler = context["movie_distance_scaler"]

    distance_query = None
    distance_data_lab1 = None
    if context["dataset"] == "movie" and movie_distance_scaler is not None:
        distance_query = apply_distance_scaler(query_df, movie_distance_scaler)
        distance_data_lab1 = apply_distance_scaler(data_lab1, movie_distance_scaler)

    debug_ctx = None
    if ufce1_debug:
        scale_cols = list(movie_distance_scaler.get("scale_cols", [])) if movie_distance_scaler else []
        debug_ctx = {
            "enabled": True,
            "structured_only": True,
            "events": [],
            "trace_positions": list(range(len(query_df))),
            "trace_positions_set": set(range(len(query_df))),
            "numf_set": set(numf),
            "scale_cols_set": set(scale_cols),
            "scaled_bounds": {col: (0.0, 100.0) for col in scale_cols},
            "radius_probe": [],
        }

    method_calls = {
        "UFCE1": lambda: sfexp(
            x_all,
            data_lab1,
            query_df,
            uf,
            step,
            f2change,
            numf,
            catf,
            lr,
            desired_outcome,
            1,
            features,
            return_stats=True,
            flip_filter_enabled=flip_filter_enabled,
            distance_data_lab1=distance_data_lab1,
            distance_X_test=distance_query,
            distance_scaler=movie_distance_scaler,
            debug_ctx=debug_ctx,
            return_trace=True,
        ),
        "UFCE2": lambda: dfexp(
            x_all,
            data_lab1,
            query_df,
            uf,
            mi_pairs_top5,
            numf,
            catf,
            features,
            protectf,
            lr,
            desired_outcome,
            1,
            features,
            return_stats=True,
            flip_filter_enabled=flip_filter_enabled,
            distance_data_lab1=distance_data_lab1,
            distance_X_test=distance_query,
            distance_scaler=movie_distance_scaler,
            return_trace=True,
        ),
        "UFCE3": lambda: tfexp(
            x_all,
            data_lab1,
            query_df,
            uf,
            mi_pairs_top5,
            numf,
            catf,
            f2change,
            protectf,
            lr,
            desired_outcome,
            1,
            features,
            return_stats=True,
            flip_filter_enabled=flip_filter_enabled,
            distance_data_lab1=distance_data_lab1,
            distance_X_test=distance_query,
            distance_scaler=movie_distance_scaler,
            return_trace=True,
        ),
    }

    grouped: dict = {}
    trace_rows_out: List[dict] = []
    method_stats: Dict[str, dict] = {}
    ufce1_events = group_events_by_instance(debug_ctx.get("events", [])) if debug_ctx else {}

    for position, meta in enumerate(query_meta):
        factual_row = query_df.iloc[[position]].reset_index(drop=True)
        pred_before = int(np.asarray(lr.predict(factual_row)).reshape(-1)[0])
        grouped[str(meta["query_id"])] = {
            "query_id": str(meta["query_id"]),
            "source": meta,
            "pred_before": pred_before,
            "factual": factual_row.iloc[0].to_dict(),
            "methods": {},
        }

    for method, runner in method_calls.items():
        _cfs, _elapsed, _found_idx, stats, trace_rows = runner()
        method_stats[method] = stats
        for trace_row in trace_rows:
            pos = int(trace_row["instance_pos"])
            meta = query_meta[pos]
            factual_row = query_df.iloc[[pos]].reset_index(drop=True)
            generated = candidate_records(
                trace_row.get("generated_candidates_df", pd.DataFrame()),
                factual_row,
                features,
                lr,
                desired_outcome,
                candidate_cap,
            )
            label_flip = candidate_records(
                trace_row.get("label_flip_candidates_df", pd.DataFrame()),
                factual_row,
                features,
                lr,
                desired_outcome,
                candidate_cap,
            )
            selected = candidate_records(
                trace_row.get("selected_candidates_df", pd.DataFrame()),
                factual_row,
                features,
                lr,
                desired_outcome,
                1,
            )
            pred_before = grouped[str(meta["query_id"])]["pred_before"]
            violations = invariant_violations(
                method=method,
                selected_records=selected,
                factual_pred=pred_before,
                desired_outcome=desired_outcome,
                allowed_features=f2change,
                flip_filter_enabled=flip_filter_enabled,
            )
            detail = {
                "method": method,
                "query_id": str(meta["query_id"]),
                "pred_before": pred_before,
                "num_candidates": int(len(generated)),
                "trace_written": int(min(candidate_cap, len(generated))),
                "flip_ok_count_in_cap": int(sum(1 for rec in generated if rec["flips_label"])),
                "cap": int(candidate_cap),
                "selected_count": int(len(selected)),
                "selected_prediction": selected_prediction(selected),
                "search_meta": trace_row.get("search_meta", {}),
                "search_parameters": trace_row.get("search_parameters", {}),
                "source_path": trace_row.get("source_path", None),
                "raw_primary_count": int(trace_row.get("raw_primary_count", 0)),
                "raw_explore_count": int(trace_row.get("raw_explore_count", 0)),
                "generated_candidates": generated,
                "label_flip_candidates": label_flip,
                "selected_candidates": selected,
                "invariant_violations": violations,
            }
            if method == "UFCE1":
                detail["internal_debug_events"] = ufce1_events.get(pos, [])
            grouped[str(meta["query_id"])]["methods"][method] = detail
            trace_rows_out.append(detail)

    return grouped, trace_rows_out, method_stats


def build_markdown_summary(dataset: str, desired_outcome: float, grouped: dict) -> str:
    lines = [
        "# UFCE Trace Summary (P7-A) - Batch",
        "",
        f"- timestamp_utc: {now_iso()}",
        f"- dataset: {dataset}",
        f"- desired_outcome: {int(desired_outcome)}",
        f"- query_count: {len(grouped)}",
        "",
        "## Per-query method summaries",
        "",
    ]
    for query_id, payload in grouped.items():
        lines.append(f"### query_id={query_id}")
        lines.append(f"- pred_before: {payload['pred_before']}")
        for method in ("UFCE1", "UFCE2", "UFCE3"):
            detail = payload["methods"].get(method, {})
            lines.append(
                f"- {method}: num_candidates={detail.get('num_candidates', 0)}, "
                f"trace_written={detail.get('trace_written', 0)}, "
                f"flip_ok_count_in_cap={detail.get('flip_ok_count_in_cap', 0)}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace-first UFCE harness for audit/debug work.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES.keys()))
    parser.add_argument("--data-dir", default="ufce/data")
    parser.add_argument("--folds-dir", default="ufce/data/folds")
    parser.add_argument("--input-mode", choices=["totest_pred0", "full_rejected"], default="totest_pred0")
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--radius", type=int, default=500)
    parser.add_argument("--n-neighbors", type=int, default=1000)
    parser.add_argument("--min-act", type=int, default=1)
    parser.add_argument("--min-feas", type=int, default=1)
    parser.add_argument("--contprox-metric", default="euclidean")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--flip-filter", type=int, choices=[0, 1], default=1)
    parser.add_argument("--candidate-cap", type=int, default=200)
    parser.add_argument("--no-ufce1-debug", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join("outputs", "ufce_trace_harness", f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "traces"))
    log_env(out_dir)

    context = load_dataset_context(args.dataset, args.data_dir)
    if args.input_mode == "totest_pred0":
        query_df, query_meta = load_queries_from_totest(args.dataset, args.folds_dir, context["features"])
    else:
        query_df, query_meta = load_queries_from_full_dataset(
            context["datasetdf"],
            context["lr"],
            context["features"],
            context["outcome_label"],
        )
    query_df, query_meta = filter_queries(query_df, query_meta, args.query_id, args.query_limit)
    if query_df.empty:
        raise ValueError("No queries selected for tracing.")

    grouped, trace_rows, method_stats = run_methods(
        context=context,
        query_df=query_df,
        query_meta=query_meta,
        radius=args.radius,
        n_neighbors=args.n_neighbors,
        min_act=args.min_act,
        min_feas=args.min_feas,
        contprox_metric=args.contprox_metric,
        atol=args.atol,
        flip_filter_enabled=bool(args.flip_filter),
        candidate_cap=args.candidate_cap,
        ufce1_debug=not args.no_ufce1_debug,
    )

    run_meta = {
        "timestamp_utc": now_iso(),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "input_mode": args.input_mode,
        "query_ids": [str(meta["query_id"]) for meta in query_meta],
        "query_count": int(len(query_meta)),
        "desired_outcome": int(context["desired_outcome"]),
        "features": context["features"],
        "numf": context["numf"],
        "catf": context["catf"],
        "f2change": context["f2change"],
        "protectf": context["protectf"],
        "ufce_init": {
            "radius": int(args.radius),
            "n_neighbors": int(args.n_neighbors),
            "contprox_metric": args.contprox_metric,
            "min_act": int(args.min_act),
            "min_feas": int(args.min_feas),
            "atol": float(args.atol),
            "flip_filter_enabled": bool(args.flip_filter),
        },
        "prediction": {
            "mode": "movie_distance_space" if args.dataset == "movie" else "raw",
            "desired_outcome": int(context["desired_outcome"]),
        },
        "MI_FP_top5": [list(pair) for pair in context["mi_pairs_top5"]],
        "method_stats": method_stats,
        "summaries": trace_rows,
    }
    grouped_payload = {
        "timestamp_utc": now_iso(),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "desired_outcome": int(context["desired_outcome"]),
        "prediction_mode": "movie_distance_space" if args.dataset == "movie" else "raw",
        "query_count": int(len(query_meta)),
        "query_ids": [str(meta["query_id"]) for meta in query_meta],
        "queries": grouped,
    }

    write_json(os.path.join(out_dir, "run_meta.json"), run_meta)
    write_json(os.path.join(out_dir, "traces", "grouped_by_query.json"), grouped_payload)
    write_jsonl(os.path.join(out_dir, "traces", "trace_rows.jsonl"), trace_rows)

    markdown = build_markdown_summary(args.dataset, context["desired_outcome"], grouped)
    with open(os.path.join(out_dir, "traces", f"trace_{args.dataset}.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)

    violations = []
    for row in trace_rows:
        for violation in row.get("invariant_violations", []):
            violations.append({"query_id": row["query_id"], "method": row["method"], "violation": violation})
    if args.strict and violations:
        raise AssertionError(f"Trace harness detected invariant violations: {violations[:10]}")

    print(f"[UFCE-TRACE] dataset={args.dataset} queries={len(query_meta)} out_dir={out_dir}")
    if violations:
        print(f"[UFCE-TRACE] invariant_warnings={len(violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
