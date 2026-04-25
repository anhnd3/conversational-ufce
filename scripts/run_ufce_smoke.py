#!/usr/bin/env python3
"""
Small UFCE smoke runner for a single query.

This is intentionally kept as a fast inspection tool for suspicious algorithm
behaviors, while `run_ufce_trace_harness.py` remains the fuller trace-first
audit path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_ufce_trace_harness as harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-query UFCE smoke run.")
    parser.add_argument("--dataset", default="bank", choices=sorted(harness.DATASET_FILES.keys()))
    parser.add_argument("--data-dir", default="ufce/data")
    parser.add_argument("--folds-dir", default="ufce/data/folds")
    parser.add_argument("--input-mode", choices=["totest_pred0", "full_rejected"], default="totest_pred0")
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--radius", type=int, default=500)
    parser.add_argument("--n-neighbors", type=int, default=1000)
    parser.add_argument("--min-act", type=int, default=1)
    parser.add_argument("--min-feas", type=int, default=1)
    parser.add_argument("--contprox-metric", default="euclidean")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--flip-filter", type=int, choices=[0, 1], default=1)
    parser.add_argument("--candidate-cap", type=int, default=20)
    parser.add_argument("--no-ufce1-debug", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def select_query(query_df, query_meta, query_id: Optional[str]):
    if query_id is None:
        return query_df.iloc[:1].reset_index(drop=True), query_meta[:1]
    for idx, meta in enumerate(query_meta):
        if str(meta["query_id"]) == str(query_id):
            return query_df.iloc[idx : idx + 1].reset_index(drop=True), [meta]
    raise ValueError(f"Query id '{query_id}' was not found.")


def main() -> int:
    args = parse_args()
    harness.random.seed(args.seed)
    harness.np.random.seed(args.seed)

    context = harness.load_dataset_context(args.dataset, args.data_dir)
    if args.input_mode == "totest_pred0":
        query_df, query_meta = harness.load_queries_from_totest(args.dataset, args.folds_dir, context["features"])
    else:
        query_df, query_meta = harness.load_queries_from_full_dataset(
            context["datasetdf"],
            context["lr"],
            context["features"],
            context["outcome_label"],
        )
    query_df, query_meta = select_query(query_df, query_meta, args.query_id)

    grouped, trace_rows, _method_stats = harness.run_methods(
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

    query_id = str(query_meta[0]["query_id"])
    record = grouped[query_id]
    print(f"[UFCE-SMOKE] dataset={args.dataset} query_id={query_id} pred_before={record['pred_before']}")
    for method in ("UFCE1", "UFCE2", "UFCE3"):
        detail = record["methods"][method]
        print(
            f"[UFCE-SMOKE] {method} generated={detail['num_candidates']} "
            f"selected={detail['selected_count']} selected_prediction={detail['selected_prediction']} "
            f"flip_ok_in_cap={detail['flip_ok_count_in_cap']}"
        )
        if detail.get("invariant_violations"):
            print(f"[UFCE-SMOKE] {method} invariant_violations={detail['invariant_violations']}")

    if args.out_json:
        payload = {
            "dataset": args.dataset,
            "query_id": query_id,
            "input_mode": args.input_mode,
            "config": {
                "radius": args.radius,
                "n_neighbors": args.n_neighbors,
                "min_act": args.min_act,
                "min_feas": args.min_feas,
                "contprox_metric": args.contprox_metric,
                "atol": args.atol,
                "flip_filter_enabled": bool(args.flip_filter),
            },
            "record": record,
            "trace_rows": trace_rows,
        }
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(harness.to_jsonable(payload), handle, indent=2, ensure_ascii=False)
        print(f"[UFCE-SMOKE] wrote_json={args.out_json}")

    if args.strict:
        violations = []
        for row in trace_rows:
            for violation in row.get("invariant_violations", []):
                violations.append((row["method"], violation))
        if violations:
            raise AssertionError(f"Invariant violations detected: {violations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
