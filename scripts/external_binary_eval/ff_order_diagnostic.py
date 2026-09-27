"""Cold-process UFCE-FF2/FF3 order-reversal timing diagnostic.

This deliberately writes outside the frozen thesis experiment outputs. MI is
loaded from the saved author-alignment artifact, not recomputed or charged to
either method's per-query latency.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import make_adapter
from .author_alignment import CATEGORY_FEATURE, REGIME, _new_raw_policy, _reference_state
from .dataset import write_json
from .run_ten_percent_eval import _append_csv, _student_reference_subset, prepare_data
from .runner import _run_one_query, _sha256


SOURCE = Path("outputs/external_binary_eval/upv_2025_author_alignment_v1")
OUTPUT = Path("outputs/external_binary_eval/upv_2025_ff_order_check_v1")
METHODS = {"FF2": "UFCE-FF2", "FF3": "UFCE-FF3"}


def _check_saved_inputs(queries, reference):
    saved_queries = pd.read_csv(SOURCE / "query_ids.csv")
    columns = ["query_id", "student_id", "source_year", "fold_id", "y", "original_prediction"]
    left = queries.loc[:, columns].astype(str).reset_index(drop=True)
    right = saved_queries.loc[:, columns].astype(str).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Regenerated 207-query cohort differs from saved author-alignment cohort")
    saved_reference = pd.read_csv(SOURCE / "reference_ids_10k.csv")
    if not np.array_equal(reference.index.to_numpy(dtype=np.int64), saved_reference.reference_row_index.to_numpy(dtype=np.int64)):
        raise ValueError("The ordered 10k reference row IDs differ from the saved experiment")
    if not np.array_equal(reference.student_id.astype(str).to_numpy(), saved_reference.student_id.astype(str).to_numpy()):
        raise ValueError("The ordered 10k reference student IDs differ from the saved experiment")


def _install_cache_probe(ufc, counters, active):
    original = ufc.regressionModel

    def observed_regression(df, independent, dependent):
        method = active["method"]
        if method:
            key = (id(df), str(dependent))
            counters[method]["regression_calls"] += 1
            if key in ufc._regression_model_cache:
                counters[method]["regression_cache_hits"] += 1
            else:
                counters[method]["regression_cache_misses"] += 1
        return original(df, independent, dependent)

    ufc.regressionModel = observed_regression


def run(order, run_id, out_dir):
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "query_timing.csv"
    setup_path = out_dir / "method_setup.csv"
    if query_path.exists() or setup_path.exists():
        raise FileExistsError("Diagnostic run ID already has results; use a fresh --run-id")

    preparation_started = time.perf_counter()
    _, train, _, queries, schema, _, bundle, policies, model_path = prepare_data(
        "data/upv_2025", out_dir / "input_audit"
    )
    reference = _student_reference_subset(train, 10000, seed=42)
    _check_saved_inputs(queries, reference)
    raw_train = train.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy()
    reference = reference.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy()
    state = _reference_state(raw_train, reference, bundle, schema)
    with open(SOURCE / "mi_pair_sets.json", "r", encoding="utf-8") as handle:
        saved_mi = json.load(handle)["DIGITAL_PAIRS_PLUS_CATEGORY"]
    pairs = saved_mi["pairs"]
    preparation_seconds = time.perf_counter() - preparation_started

    counters = {method: {"regression_calls": 0, "regression_cache_hits": 0, "regression_cache_misses": 0}
                for method in METHODS.values()}
    active = {"method": None}
    probe_owner = None
    method_summaries = []
    for position, short_name in enumerate(order.split(","), start=1):
        method = METHODS[short_name]
        setup_started = time.perf_counter()
        adapter = make_adapter(
            method, raw_train, reference, bundle, schema, policies, REGIME,
            seed=0, method_name=method, categorical_actionable=[CATEGORY_FEATURE],
            mi_pairs_override=pairs, mi_summary_override=saved_mi,
            shared_reference_state=state,
        )
        from ufce.ufce_ff import cfmethods
        ufc = cfmethods.ufc
        if probe_owner is None:
            _install_cache_probe(ufc, counters, active)
            probe_owner = ufc
        elif ufc is not probe_owner:
            raise RuntimeError("UFCE cache object changed between methods; this run cannot test shared warm cache")
        setup_seconds = time.perf_counter() - setup_started
        _append_csv({
            "run_id": run_id, "order": order, "position": position, "method": method,
            "setup_seconds": setup_seconds, "adapter_setup_seconds": adapter.setup_seconds,
            "regression_cache_entries_before": len(ufc._regression_model_cache),
            "kdtree_cache_entries_before": len(ufc._kdtree_cache),
            "radius_bounds_cache_entries_before": len(ufc._radius_bounds_cache),
        }, setup_path)
        active["method"] = method
        method_started = time.perf_counter()
        latencies = []
        statuses = {}
        for query_position, (_, query) in enumerate(queries.iterrows(), start=1):
            raw_query = query.loc[schema["feature_names"]].to_frame().T
            raw_policy = _new_raw_policy(policies[REGIME], raw_query, train, schema)
            cache_before = len(ufc._regression_model_cache)
            hits_before = counters[method]["regression_cache_hits"]
            misses_before = counters[method]["regression_cache_misses"]
            wall_started = time.perf_counter()
            result, candidates = _run_one_query(
                adapter, raw_query, str(query.query_id), raw_policy, bundle, schema,
                method, REGIME, 0, 60.0,
            )
            measured_ms = (time.perf_counter() - wall_started) * 1000.0
            latency_ms = max(60000.0, measured_ms) if result["final_status"] == "TIMEOUT" else measured_ms
            latencies.append(latency_ms)
            statuses[result["final_status"]] = statuses.get(result["final_status"], 0) + 1
            _append_csv({
                "run_id": run_id, "order": order, "position": position, "method": method,
                "query_position": query_position, "query_id": query.query_id,
                "fold_id": int(query.fold_id), "status": result["final_status"],
                "latency_ms": latency_ms,
                "generation_ms": result["generation_runtime_ms"],
                "verification_ms": result["verification_runtime_ms"],
                "returned_any_feasible_cf": bool(result["returned_any_feasible_cf"]),
                "returned_candidate_count": int(len(candidates)),
                "regression_cache_entries_before": cache_before,
                "regression_cache_entries_after": len(ufc._regression_model_cache),
                "regression_cache_hits_query": counters[method]["regression_cache_hits"] - hits_before,
                "regression_cache_misses_query": counters[method]["regression_cache_misses"] - misses_before,
            }, query_path)
            if query_position % 25 == 0 or query_position == len(queries):
                print("%s %s %d/%d elapsed=%.1fs" % (
                    run_id, method, query_position, len(queries), time.perf_counter() - method_started
                ), flush=True)
        active["method"] = None
        method_summaries.append({
            "method": method, "position": position, "query_seconds": time.perf_counter() - method_started,
            "median_ms": float(np.median(latencies)), "p95_ms": float(np.percentile(latencies, 95)),
            "total_latency_seconds": float(np.sum(latencies) / 1000.0),
            "status_counts": statuses,
            "regression_cache_entries_after": len(ufc._regression_model_cache),
            "kdtree_cache_entries_after": len(ufc._kdtree_cache),
            "radius_bounds_cache_entries_after": len(ufc._radius_bounds_cache),
            **counters[method],
        })

    manifest = {
        "purpose": "FF2/FF3 run-order and cache timing diagnostic; not a replacement for thesis results",
        "run_id": run_id, "order": order, "reference_rows": len(reference), "query_count": len(queries),
        "model_sha256": _sha256(model_path),
        "saved_query_ids_sha256": _sha256(SOURCE / "query_ids.csv"),
        "saved_reference_ids_sha256": _sha256(SOURCE / "reference_ids_10k.csv"),
        "saved_mi_sets_sha256": _sha256(SOURCE / "mi_pair_sets.json"),
        "mi_pair_sha256": saved_mi["pair_sha256"],
        "mi_seconds_saved": saved_mi["mi_seconds"],
        "preparation_seconds": preparation_seconds,
        "methods": method_summaries,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "code_sha256": {
            name: _sha256(Path(name)) for name in (
                "scripts/external_binary_eval/ff_order_diagnostic.py",
                "scripts/external_binary_eval/adapters.py",
                "scripts/external_binary_eval/author_alignment.py",
                "ufce/ufce_ff/ufce.py",
                "ufce/ufce_ff/cfmethods.py",
            )
        },
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(method_summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", required=True, choices=("FF2,FF3", "FF3,FF2"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=str(OUTPUT))
    arguments = parser.parse_args()
    run(arguments.order, arguments.run_id, arguments.out_dir)
