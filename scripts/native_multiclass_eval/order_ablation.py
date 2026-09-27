"""Post-hoc FF3-before-FF2 timing check on the frozen student test queries.

This reads the completed experiment's model, split, policy, and MI ranking. It
writes only to a fresh output directory and leaves the original result intact.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .adapters import UFCEAdapter
from .config import SPLIT_SEED
from .dataset import load_dataset, sha256_file, split_dataset, write_json
from .runner import _run_queries, _write_csv_atomic
from .space import EncodedSpace


TRANSITIONS = (
    "dropout_to_enrolled",
    "dropout_to_graduate",
    "enrolled_to_graduate",
)
METHODS = ("UFCE-FF3", "UFCE-FF2")


def run(baseline_dir, out_dir, methods=METHODS):
    if set(methods) != {"UFCE-FF2", "UFCE-FF3"} or len(methods) != 2:
        raise ValueError("The order must contain FF2 and FF3 exactly once")
    baseline_dir, out_dir = Path(baseline_dir).resolve(), Path(out_dir).resolve()
    if baseline_dir == out_dir or out_dir.exists():
        raise ValueError("Output must be a new directory distinct from baseline")
    frozen = json.loads((baseline_dir / "freeze_manifest.json").read_text())["configuration"]
    dataset_path = Path(frozen["dataset"]["source_path"])
    if sha256_file(dataset_path) != frozen["dataset"]["source_sha256"]:
        raise ValueError("Dataset checksum differs from baseline")
    model_path = baseline_dir / "model_bundle.joblib"
    if sha256_file(model_path) != frozen["model_bundle_sha256"]:
        raise ValueError("Frozen model checksum differs from baseline")
    features, labels, schema, _ = load_dataset(dataset_path, auto_download=False)
    saved_schema = json.loads((baseline_dir / "feature_schema.json").read_text())
    policy = json.loads((baseline_dir / "constraint_policy.json").read_text())
    if schema != saved_schema or policy != frozen["policy"]:
        raise ValueError("Schema or policy differs from baseline")
    split = pd.read_csv(baseline_dir / "split_manifest.csv")
    if len(split) != len(features) or not np.array_equal(split.row_id.to_numpy(), np.arange(len(features))):
        raise ValueError("Split manifest row IDs differ from dataset")
    if not np.array_equal(split.target_class.to_numpy(), labels):
        raise ValueError("Split manifest target labels differ from dataset")
    recreated_split = split_dataset(features, labels, SPLIT_SEED)
    for partition, row_ids in recreated_split.items():
        saved = split.loc[split.partition.eq(partition), "row_id"].to_numpy(dtype=int)
        if not np.array_equal(np.sort(row_ids), np.sort(saved)):
            raise ValueError("Recreated %s split differs from baseline" % partition)
    # The CSV is sorted by row ID; preserve train_test_split reference order.
    train_rows = recreated_split["train"]
    bundle = joblib.load(model_path)
    train_raw = features.iloc[train_rows].reset_index(drop=True)
    train_encoded = EncodedSpace(train_raw, schema).encode(train_raw)
    train_predictions = bundle.predict_raw(train_raw)
    queries = pd.read_csv(baseline_dir / "final_test_queries.csv")
    queries = queries.loc[queries.transition.isin(TRANSITIONS)].copy()
    if len(queries) != 546 or queries.query_id.duplicated().any():
        raise ValueError("Expected exactly 546 distinct thesis-transition queries")
    if not set(queries.row_id).issubset(set(split.loc[split.partition.eq("test"), "row_id"])):
        raise ValueError("Query outside frozen final test")
    if not np.array_equal(bundle.predict_raw(features.iloc[queries.row_id.to_numpy()]), queries.source_class.to_numpy()):
        raise ValueError("Model predictions differ from query source class")
    if frozen["split_seed"] != SPLIT_SEED:
        raise ValueError("Split seed differs from current adapter configuration")

    out_dir.mkdir(parents=True)
    source_changes = {}
    for name, old_hash in frozen["source_sha256"].items():
        path = Path(name)
        current = sha256_file(path) if path.is_file() else None
        if current != old_hash:
            source_changes[name] = {"frozen_sha256": old_hash, "current_sha256": current}
    write_json(out_dir / "provenance.json", {
        "baseline_dir": str(baseline_dir),
        "baseline_freeze_sha256": sha256_file(baseline_dir / "freeze_manifest.json"),
        "dataset_sha256": frozen["dataset"]["source_sha256"],
        "model_sha256": frozen["model_bundle_sha256"],
        "order_within_transition": list(methods),
        "transitions": list(TRANSITIONS),
        "query_count": len(queries),
        "post_hoc": True,
        "source_changes_since_baseline": source_changes,
        "started_at_unix": time.time(),
    })
    all_queries, all_candidates, all_exposed = [], [], []
    for transition in TRANSITIONS:
        current = queries.loc[queries.transition.eq(transition)].reset_index(drop=True)
        for method in methods:
            adapter = UFCEAdapter(
                method, train_raw, train_encoded, train_predictions, bundle,
                schema, policy, frozen["mi_pairs"],
            )

            def checkpoint(query_rows, candidate_rows, exposed_rows, last_query):
                for name, prior, new in (
                    ("query_results", all_queries, query_rows),
                    ("candidate_summaries", all_candidates, candidate_rows),
                    ("exposed_candidates", all_exposed, exposed_rows),
                ):
                    frames = prior + [pd.DataFrame(new)]
                    _write_csv_atomic(pd.concat(frames, ignore_index=True), out_dir / (name + ".partial.csv"))
                write_json(out_dir / "progress.json", {
                    "transition": transition,
                    "method": method,
                    "last_query_id": last_query["query_id"],
                    "completed_rows": sum(len(frame) for frame in all_queries) + len(query_rows),
                    "expected_rows": 2 * len(queries),
                    "updated_at_unix": time.time(),
                })

            query_rows, candidate_rows, exposed_rows = _run_queries(
                current, features, train_raw, adapter, bundle, schema, policy,
                method, "order_ablation", progress_callback=checkpoint,
            )
            all_queries.append(query_rows)
            all_candidates.append(candidate_rows)
            all_exposed.append(exposed_rows)
            print("[order-check] %s %s complete %d/%d" % (
                transition, method, sum(map(len, all_queries)), 2 * len(queries)), flush=True)

    result = pd.concat(all_queries, ignore_index=True)
    result.to_csv(out_dir / "query_results.csv", index=False)
    pd.concat(all_candidates, ignore_index=True).to_csv(out_dir / "candidate_summaries.csv", index=False)
    pd.concat(all_exposed, ignore_index=True).to_csv(out_dir / "exposed_candidates.csv", index=False)
    original = pd.read_csv(baseline_dir / "final_test_query_results.csv")
    original = original.loc[original.transition.isin(TRANSITIONS) & original.method.isin(methods)]
    merged = result.merge(original, on=["query_id", "method"], how="outer", validate="one_to_one",
                          suffixes=("_rerun", "_original"), indicator=True)
    if len(merged) != 2 * len(queries) or not merged._merge.eq("both").all():
        raise ValueError("Original and rerun result query IDs do not pair exactly")
    fields = ["target_valid_availability", "constraint_feasible_availability", "returned_any_cf",
              "candidate_count", "target_candidate_count", "off_target_candidate_count",
              "feasible_candidate_count", "exposed_cf_count", "status", "schema_error_count"]
    mismatches = {field: int((merged[field + "_rerun"] != merged[field + "_original"]).sum())
                  for field in fields}
    summary = {}
    for method, group in merged.groupby("method"):
        new = group.runtime_ms_rerun.to_numpy(dtype=float) / 1000
        old = group.runtime_ms_original.to_numpy(dtype=float) / 1000
        summary[method] = {
            "query_count": len(group),
            "original_total_seconds": float(old.sum()),
            "rerun_total_seconds": float(new.sum()),
            "original_median_seconds": float(np.median(old)),
            "rerun_median_seconds": float(np.median(new)),
            "original_p95_seconds": float(np.percentile(old, 95)),
            "rerun_p95_seconds": float(np.percentile(new, 95)),
            "median_paired_delta_seconds": float(np.median(new - old)),
            "original_feasible_queries": int(group.constraint_feasible_availability_original.sum()),
            "rerun_feasible_queries": int(group.constraint_feasible_availability_rerun.sum()),
        }
    write_json(out_dir / "comparison.json", {
        "completed_at_unix": time.time(),
        "query_method_rows": len(result),
        "mismatch_counts_by_field": mismatches,
        "summary": summary,
        "source_changes_since_baseline": source_changes,
    })
    print(json.dumps(summary, indent=2), flush=True)
    print("mismatches", mismatches, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--order", choices=("ff3-first", "ff2-first"), default="ff3-first")
    args = parser.parse_args()
    methods = ("UFCE-FF3", "UFCE-FF2") if args.order == "ff3-first" else ("UFCE-FF2", "UFCE-FF3")
    run(args.baseline_dir, args.out_dir, methods)


if __name__ == "__main__":
    main()
