"""Gated end-to-end native multiclass evaluation runner."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .adapters import DiCEAdapter, UFCEAdapter
from .config import (
    ACTIONABLE_FEATURES,
    CLASS_NAMES,
    CHECKPOINT_INTERVAL_QUERIES,
    DICE_SAMPLE_SIZE,
    MAX_CANDIDATES,
    METHODS,
    PILOT_COUNTS,
    PILOT_SEED,
    SPLIT_SEED,
    TIMEOUT_SECONDS,
    TRANSITIONS,
)
from .dataset import load_dataset, sha256_file, split_dataset, write_json, write_split_manifest
from .metrics import paired_comparisons, summarize_all
from .mi_cache import load_mi_cache, prepare_mi_cache
from .model import fit_model, prediction_summary
from .policy import build_train_policy, policy_for_query
from .space import EncodedSpace
from .timeout import QueryTimeout, query_timeout
from .verification import verify_candidate


DEFAULT_DATA_PATH = Path("data/native_multiclass_eval/uci_students_697.zip")
DEFAULT_OUTPUT_ROOT = Path("outputs/native_multiclass_eval")
CORE_PATHS = [
    Path("ufce/ufce_ff/ufce.py"),
    Path("ufce/ufce_ff/cfmethods.py"),
]


def _safe_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _canonical_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_csv_atomic(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(str(temporary), str(path))


def _source_fingerprints(root):
    source_files = sorted((root / "scripts/native_multiclass_eval").glob("*.py"))
    source_files.extend(root / path for path in CORE_PATHS)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in source_files if path.is_file()
    }


def _select_pilot_queries(dev_rows, dev_predictions):
    rows = []
    for transition in TRANSITIONS:
        count = int(PILOT_COUNTS[transition.key])
        eligible = np.flatnonzero(np.asarray(dev_predictions) == int(transition.source))
        if len(eligible) < count:
            raise ValueError(
                "Pilot has only %d dev rows predicted %s; needs %d for %s"
                % (len(eligible), CLASS_NAMES[transition.source], count, transition.key)
            )
        rng = np.random.RandomState(PILOT_SEED + transition.source * 10 + transition.target)
        selected = np.sort(rng.choice(eligible, size=count, replace=False))
        for dev_position in selected:
            original_row_id = int(dev_rows[int(dev_position)])
            rows.append({
                "query_id": "dev_row_%05d__%s" % (original_row_id, transition.key),
                "row_id": original_row_id,
                "source_class": int(transition.source),
                "target_class": int(transition.target),
                "transition": transition.key,
                "split": "dev",
            })
    return pd.DataFrame(rows)


def _select_final_queries(test_rows, test_predictions):
    rows = []
    test_predictions = np.asarray(test_predictions, dtype=int)
    for position, source_class in enumerate(test_predictions):
        row_id = int(test_rows[position])
        for target_class in range(3):
            if target_class == int(source_class):
                continue
            transition = "%s_to_%s" % (
                CLASS_NAMES[int(source_class)].lower(), CLASS_NAMES[target_class].lower()
            )
            rows.append({
                "query_id": "test_row_%05d__%s" % (row_id, transition),
                "row_id": row_id,
                "source_class": int(source_class),
                "target_class": int(target_class),
                "transition": transition,
                "split": "test",
            })
    return pd.DataFrame(rows)


def _make_adapter(method, train_raw, train_encoded, train_predictions, bundle, schema, policy, mi_pairs):
    if method.startswith("UFCE-FF"):
        return UFCEAdapter(
            method, train_raw, train_encoded, train_predictions, bundle, schema, policy, mi_pairs
        )
    if method == "DiCE":
        return DiCEAdapter(
            train_raw, train_encoded, train_predictions, bundle, schema, policy, mi_pairs
        )
    raise ValueError("Unsupported method: %s" % method)


def _run_queries(query_manifest, features, train_raw, adapter, bundle, schema, train_policy, method, stage, progress_callback=None):
    query_rows = []
    candidate_rows = []
    exposed_rows = []
    # Use the adapter's reversible feature mapping so categories first seen in
    # dev/test remain decodable for the factual and every returned candidate.
    space = adapter.space
    for query in query_manifest.to_dict("records"):
        factual = features.iloc[[int(query["row_id"])]].loc[:, schema["feature_names"]].reset_index(drop=True)
        query_policy = policy_for_query(train_policy, factual)
        started = time.perf_counter()
        generation = None
        status = "OK"
        error_type = ""
        error_detail = ""
        try:
            with query_timeout(TIMEOUT_SECONDS):
                generation = adapter.generate(
                    factual,
                    int(query["source_class"]),
                    int(query["target_class"]),
                    query_policy,
                    max_candidates=MAX_CANDIDATES,
                    seed=SPLIT_SEED,
                )
        except QueryTimeout:
            status = "TIMEOUT"
        except Exception as exc:
            status = "RUNTIME_ERROR"
            error_type = type(exc).__name__
            error_detail = str(exc)
        runtime_ms = float((time.perf_counter() - started) * 1000.0)
        if generation is not None and generation.error_type:
            status = "RUNTIME_ERROR"
        if generation is not None and generation.schema_errors:
            status = "SCHEMA_ERROR"
        if generation is None:
            candidate_summary = {
                "candidate_count": 0,
                "target_count": 0,
                "source_count": 0,
                "off_target_count": 0,
                "feasible_count": 0,
            }
            exposed_candidates = pd.DataFrame(columns=schema["feature_names"])
            schema_errors = []
            error_type = "QueryTimeout" if status == "TIMEOUT" else error_type
        else:
            candidate_summary = dict(generation.candidate_summary)
            exposed_candidates = generation.exposed_candidates
            schema_errors = list(generation.schema_errors)
            error_type = generation.error_type
            error_detail = generation.error_detail
        required_candidate_fields = {
            "candidate_count", "target_count", "source_count", "off_target_count", "feasible_count"
        }
        if set(candidate_summary) != required_candidate_fields:
            status = "SCHEMA_ERROR"
            candidate_summary = {name: int(candidate_summary.get(name, 0)) for name in required_candidate_fields}

        verification_rows = []
        for index in range(len(exposed_candidates)):
            candidate = exposed_candidates.iloc[[index]].loc[:, schema["feature_names"]]
            raw_candidate = space.decode(candidate)
            verification = verify_candidate(
                factual, raw_candidate, bundle, schema, query_policy,
                int(query["source_class"]), int(query["target_class"]),
            )
            record = {
                "stage": stage,
                "query_id": query["query_id"],
                "method": method,
                "transition": query["transition"],
                "exposed_rank": int(index),
                "target_valid": bool(verification.target_valid),
                "constraint_feasible": bool(verification.constraint_feasible),
                "prediction": verification.prediction,
                "violations": ";".join(verification.violations),
            }
            record.update({"feature_%s" % name: value for name, value in verification.raw_candidate.items()})
            verification_rows.append(record)
        exposed_rows.extend(verification_rows)
        candidate_rows.append({
            "stage": stage,
            "query_id": query["query_id"],
            "method": method,
            "transition": query["transition"],
            "source_class": int(query["source_class"]),
            "target_class": int(query["target_class"]),
            "candidate_count": int(candidate_summary.get("candidate_count", 0)),
            "target_count": int(candidate_summary.get("target_count", 0)),
            "source_count": int(candidate_summary.get("source_count", 0)),
            "off_target_count": int(candidate_summary.get("off_target_count", 0)),
            "feasible_count": int(candidate_summary.get("feasible_count", 0)),
            "off_target_rate": (
                float(candidate_summary.get("off_target_count", 0)) / candidate_summary["candidate_count"]
                if candidate_summary.get("candidate_count", 0) else None
            ),
            "schema_errors": ";".join(schema_errors),
        })
        query_rows.append({
            "stage": stage,
            "query_id": query["query_id"],
            "row_id": int(query["row_id"]),
            "method": method,
            "transition": query["transition"],
            "source_class": int(query["source_class"]),
            "target_class": int(query["target_class"]),
            "target_valid_availability": bool(candidate_summary.get("target_count", 0) > 0),
            "constraint_feasible_availability": bool(candidate_summary.get("feasible_count", 0) > 0),
            "returned_any_cf": bool(len(verification_rows)),
            "exposed_target_valid": bool(verification_rows and all(row["target_valid"] for row in verification_rows)),
            "exposed_constraint_satisfied": bool(verification_rows and all(row["constraint_feasible"] for row in verification_rows)),
            "candidate_count": int(candidate_summary.get("candidate_count", 0)),
            "target_candidate_count": int(candidate_summary.get("target_count", 0)),
            "off_target_candidate_count": int(candidate_summary.get("off_target_count", 0)),
            "feasible_candidate_count": int(candidate_summary.get("feasible_count", 0)),
            "exposed_cf_count": int(len(verification_rows)),
            "runtime_ms": runtime_ms,
            "setup_ms": float(adapter.setup_seconds * 1000.0),
            "status": status,
            "error_type": error_type,
            "error_detail": error_detail,
            "schema_error_count": int(len(schema_errors)),
        })
        if progress_callback is not None and (
            len(query_rows) % CHECKPOINT_INTERVAL_QUERIES == 0
            or len(query_rows) == len(query_manifest)
        ):
            progress_callback(query_rows, candidate_rows, exposed_rows, query)
    exposed_columns = [
        "stage", "query_id", "method", "transition", "exposed_rank", "target_valid",
        "constraint_feasible", "prediction", "violations",
    ] + ["feature_%s" % name for name in schema["feature_names"]]
    return pd.DataFrame(query_rows), pd.DataFrame(candidate_rows), pd.DataFrame(exposed_rows, columns=exposed_columns)


def _write_stage(out_dir, name, query_rows, candidate_rows, exposed_rows):
    query_rows.to_csv(out_dir / (name + "_query_results.csv"), index=False)
    candidate_rows.to_csv(out_dir / (name + "_candidate_summaries.csv"), index=False)
    exposed_rows.to_csv(out_dir / (name + "_exposed_candidates.csv"), index=False)
    summary = summarize_all(query_rows, candidate_rows, exposed_rows)
    paired = paired_comparisons(query_rows)
    summary.to_csv(out_dir / (name + "_transition_summary.csv"), index=False)
    paired.to_csv(out_dir / (name + "_paired_comparisons.csv"), index=False)
    return summary, paired


def _pilot_gate(queries, query_rows, candidate_rows, exposed_rows):
    required_pairs = set(queries["transition"].unique())
    observed_pairs = set(query_rows["transition"].unique())
    expected_methods = set(METHODS)
    observed_methods = set(query_rows["method"].unique())
    completeness = all(
        len(query_rows[(query_rows["transition"] == transition) & (query_rows["method"] == method)])
        == len(queries[queries["transition"] == transition])
        for transition in required_pairs for method in expected_methods
    )
    schema_errors = int(query_rows.get("schema_error_count", pd.Series(dtype=int)).fillna(0).sum())
    runtime_errors = int(query_rows.get("status", pd.Series(dtype=str)).isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).sum())
    candidate_status_fields = {"target_count", "source_count", "off_target_count", "feasible_count"}
    ledger_present = candidate_status_fields.issubset(candidate_rows.columns)
    verifier_ran = "constraint_feasible" in exposed_rows.columns or "feasible_count" in candidate_rows.columns
    checks = {
        "all_six_transitions_present": required_pairs == observed_pairs and len(required_pairs) == 6,
        "all_methods_present": observed_methods == expected_methods,
        "all_query_method_denominators_complete": bool(completeness),
        "off_target_ledger_present": bool(ledger_present),
        "verifier_results_present": bool(verifier_ran),
        "no_silent_schema_errors": schema_errors == 0,
        "no_runtime_or_schema_errors": runtime_errors == 0,
        "availability_denominators_complete": bool(
            query_rows["target_valid_availability"].notna().all()
            and query_rows["constraint_feasible_availability"].notna().all()
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "schema_error_count": schema_errors, "runtime_error_count": runtime_errors}


def _run_stage(
    query_manifest, features, train_raw, train_encoded, train_predictions,
    bundle, schema, policy, mi_pairs, stage, out_dir, resume_data=None,
):
    all_queries = []
    all_candidates = []
    all_exposed = []
    if resume_data is not None:
        resume_queries, resume_candidates, resume_exposed = resume_data
        if len(resume_queries):
            all_queries.append(resume_queries)
        if len(resume_candidates):
            all_candidates.append(resume_candidates)
        if len(resume_exposed):
            all_exposed.append(resume_exposed)
    for transition in [item.key for item in TRANSITIONS]:
        current = query_manifest.loc[query_manifest["transition"] == transition].reset_index(drop=True)
        if current.empty:
            continue
        for method in METHODS:
            block_total = len(current)
            if resume_data is not None and all_queries:
                completed = pd.concat(all_queries, ignore_index=True)
                completed_ids = set(
                    completed.loc[
                        (completed["method"] == method) & (completed["transition"] == transition),
                        "query_id",
                    ].astype(str)
                )
            else:
                completed_ids = set()
            method_queries = current.loc[
                ~current["query_id"].astype(str).isin(completed_ids)
            ].reset_index(drop=True)
            already_complete = len(completed_ids)
            if method_queries.empty:
                if resume_data is not None:
                    print(
                        "[resume] %s %s %s already complete=%d/%d"
                        % (stage, method, transition, already_complete, block_total),
                        flush=True,
                    )
                continue
            adapter = _make_adapter(
                method, train_raw, train_encoded, train_predictions, bundle, schema, policy, mi_pairs
            )

            def persist_progress(current_queries, current_candidates, current_exposed, last_query):
                query_frames = all_queries + [pd.DataFrame(current_queries)]
                candidate_frames = all_candidates + [pd.DataFrame(current_candidates)]
                exposed_columns = [
                    "stage", "query_id", "method", "transition", "exposed_rank", "target_valid",
                    "constraint_feasible", "prediction", "violations",
                ] + ["feature_%s" % name for name in schema["feature_names"]]
                exposed_frames = all_exposed + [pd.DataFrame(current_exposed, columns=exposed_columns)]
                _write_csv_atomic(
                    pd.concat(query_frames, ignore_index=True),
                    out_dir / (stage + "_query_results.partial.csv"),
                )
                _write_csv_atomic(
                    pd.concat(candidate_frames, ignore_index=True),
                    out_dir / (stage + "_candidate_summaries.partial.csv"),
                )
                _write_csv_atomic(
                    pd.concat(exposed_frames, ignore_index=True),
                    out_dir / (stage + "_exposed_candidates.partial.csv"),
                )
                completed_rows = sum(len(frame) for frame in all_queries) + len(current_queries)
                write_json(out_dir / (stage + "_progress.json"), {
                    "stage": stage,
                    "transition": last_query["transition"],
                    "method": method,
                    "last_query_id": last_query["query_id"],
                    "completed_in_block": int(already_complete + len(current_queries)),
                    "block_query_count": int(block_total),
                    "completed_method_query_rows": int(completed_rows),
                    "expected_method_query_rows": int(len(query_manifest) * len(METHODS)),
                    "checkpoint_interval_queries": CHECKPOINT_INTERVAL_QUERIES,
                    "updated_at_unix": time.time(),
                })
                print(
                    "[progress] %s %s %s block=%d/%d total=%d/%d"
                    % (stage, method, last_query["transition"], already_complete + len(current_queries),
                       block_total, completed_rows, len(query_manifest) * len(METHODS)),
                    flush=True,
                )

            query_rows, candidate_rows, exposed_rows = _run_queries(
                method_queries, features, train_raw, adapter, bundle, schema, policy, method, stage,
                progress_callback=persist_progress,
            )
            all_queries.append(query_rows)
            all_candidates.append(candidate_rows)
            all_exposed.append(exposed_rows)
            _write_csv_atomic(pd.concat(all_queries, ignore_index=True), out_dir / (stage + "_query_results.partial.csv"))
            _write_csv_atomic(pd.concat(all_candidates, ignore_index=True), out_dir / (stage + "_candidate_summaries.partial.csv"))
            _write_csv_atomic(pd.concat(all_exposed, ignore_index=True), out_dir / (stage + "_exposed_candidates.partial.csv"))
    query_rows = pd.concat(all_queries, ignore_index=True) if all_queries else pd.DataFrame()
    candidate_rows = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    exposed_rows = pd.concat(all_exposed, ignore_index=True) if all_exposed else pd.DataFrame()
    _write_csv_atomic(query_rows, out_dir / (stage + "_query_results.csv"))
    _write_csv_atomic(candidate_rows, out_dir / (stage + "_candidate_summaries.csv"))
    _write_csv_atomic(exposed_rows, out_dir / (stage + "_exposed_candidates.csv"))
    return query_rows, candidate_rows, exposed_rows

def _resume_experiment(root, out_dir, checkpoint_dir, features, labels, schema, dataset_summary, splits):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError("Resume checkpoint does not exist: %s" % checkpoint_dir)
    if out_dir.resolve() == checkpoint_dir:
        raise ValueError("Resume output must be a new directory; keep the source checkpoint immutable")
    unexpected_output = [
        path.name for path in out_dir.iterdir() if path.name != "resume_console.log"
    ]
    if unexpected_output:
        raise ValueError("Resume output directory contains unexpected files: %s" % unexpected_output)

    manifest_path = checkpoint_dir / "freeze_manifest.json"
    checkpoint_path = checkpoint_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("Checkpoint is missing its freeze or checkpoint manifest")
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    frozen = freeze.get("configuration", {})
    if not freeze.get("frozen") or not checkpoint.get("process_stopped"):
        raise ValueError("Resume source is not a frozen, stopped checkpoint")

    # The only source changes allowed since this checkpoint are operational:
    # progress persistence and resume orchestration. Generation, model,
    # verification, policy, metric and adapter source must remain byte-identical.
    current_hashes = _source_fingerprints(root)
    previous_hashes = frozen.get("source_sha256", {})
    operational_paths = {
        "scripts/native_multiclass_eval/config.py",
        "scripts/native_multiclass_eval/runner.py",
    }
    # Tests cannot affect generation; older checkpoints already freeze their
    # MI pair ranking and do not depend on this newly added cache module.
    operational_paths.update(
        name for name in set(current_hashes) | set(previous_hashes)
        if Path(name).name.startswith("test_")
    )
    if "mi_cache_sha256" not in frozen:
        operational_paths.add("scripts/native_multiclass_eval/mi_cache.py")
    mismatches = sorted(
        name for name in (set(current_hashes) | set(previous_hashes))
        if name not in operational_paths and current_hashes.get(name) != previous_hashes.get(name)
    )
    if mismatches:
        raise ValueError("Non-operational experiment source changed since checkpoint: %s" % mismatches)

    if dataset_summary.get("source_sha256") != frozen.get("dataset", {}).get("source_sha256"):
        raise ValueError("Dataset checksum differs from the frozen checkpoint")
    if frozen.get("split_seed") != SPLIT_SEED or frozen.get("pilot_seed") != PILOT_SEED:
        raise ValueError("Split or pilot seed differs from the frozen checkpoint")
    if frozen.get("target_mapping") != {str(key): value for key, value in CLASS_NAMES.items()}:
        raise ValueError("Target mapping differs from the frozen checkpoint")
    if frozen.get("actionable_features") != ACTIONABLE_FEATURES:
        raise ValueError("Actionable feature policy differs from the frozen checkpoint")
    if frozen.get("max_candidates") != MAX_CANDIDATES or frozen.get("dice_sample_size") != DICE_SAMPLE_SIZE:
        raise ValueError("Candidate budget differs from the frozen checkpoint")
    if float(frozen.get("timeout_seconds", -1)) != float(TIMEOUT_SECONDS):
        raise ValueError("Query timeout differs from the frozen checkpoint")
    if list(frozen.get("methods", [])) != list(METHODS):
        raise ValueError("Method set differs from the frozen checkpoint")

    train_rows, dev_rows, test_rows = splits["train"], splits["dev"], splits["test"]
    expected_manifest = pd.DataFrame([
        {"row_id": int(row_id), "partition": partition, "target_class": int(labels[row_id])}
        for partition, row_ids in splits.items() for row_id in row_ids
    ]).sort_values("row_id").reset_index(drop=True)
    saved_manifest = pd.read_csv(checkpoint_dir / "split_manifest.csv").sort_values("row_id").reset_index(drop=True)
    if not expected_manifest.equals(saved_manifest):
        raise ValueError("Deterministic train/dev/test split differs from the checkpoint")

    model_path = checkpoint_dir / "model_bundle.joblib"
    if sha256_file(model_path) != frozen.get("model_bundle_sha256"):
        raise ValueError("Saved model checksum differs from the frozen checkpoint")
    bundle = joblib.load(model_path)
    saved_schema = json.loads((checkpoint_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if saved_schema != schema:
        raise ValueError("Feature schema differs from the frozen checkpoint")
    train_raw = features.iloc[train_rows].reset_index(drop=True)
    train_policy = build_train_policy(train_raw, schema)
    saved_policy = json.loads((checkpoint_dir / "constraint_policy.json").read_text(encoding="utf-8"))
    if _canonical_hash(train_policy) != _canonical_hash(saved_policy):
        raise ValueError("Constraint policy differs from the frozen checkpoint")
    if _canonical_hash(train_policy) != _canonical_hash(frozen.get("policy")):
        raise ValueError("Frozen constraint policy does not match the training data")

    static_files = [
        "split_manifest.csv", "model_bundle.joblib", "feature_schema.json", "constraint_policy.json",
        "model_sanity.json", "pilot_queries.csv", "pilot_query_results.csv",
        "pilot_candidate_summaries.csv", "pilot_exposed_candidates.csv",
        "pilot_transition_summary.csv", "pilot_paired_comparisons.csv", "pilot_gate.json",
        "freeze_manifest.json", "final_test_model_summary.json", "final_test_queries.csv",
        "checkpoint_manifest.json", "final_test_query_results.partial.csv",
        "final_test_candidate_summaries.partial.csv", "final_test_exposed_candidates.partial.csv",
    ]
    if frozen.get("mi_cache_sha256"):
        static_files.append("mi_cache.json")
    for filename in static_files:
        source = checkpoint_dir / filename
        if not source.is_file():
            raise ValueError("Checkpoint is missing required artifact: %s" % filename)
        shutil.copy2(str(source), str(out_dir / filename))

    train_space = EncodedSpace(train_raw, schema)
    train_encoded = train_space.encode(train_raw)
    train_predictions = bundle.predict_raw(train_raw)
    mi_pairs = frozen.get("mi_pairs")
    if not isinstance(mi_pairs, list):
        raise ValueError("Frozen mutual-information feature pairs are missing")
    if frozen.get("mi_cache_sha256"):
        cache_path = out_dir / "mi_cache.json"
        if sha256_file(cache_path) != frozen["mi_cache_sha256"]:
            raise ValueError("Frozen MI cache checksum differs from checkpoint")
        cached = load_mi_cache(
            cache_path, train_encoded, train_rows, dataset_summary["source_sha256"]
        )
        if cached["mi_pairs"] != mi_pairs:
            raise ValueError("Frozen MI cache pairs differ from checkpoint")

    final_features = features.iloc[test_rows].reset_index(drop=True)
    final_predictions = bundle.predict_raw(final_features)
    expected_queries = _select_final_queries(test_rows, final_predictions)
    saved_queries = pd.read_csv(checkpoint_dir / "final_test_queries.csv")
    key_columns = ["query_id", "row_id", "source_class", "target_class", "transition"]
    if len(expected_queries) != len(saved_queries) or not expected_queries[key_columns].equals(saved_queries[key_columns]):
        raise ValueError("Frozen final-test query manifest does not match the saved model and split")

    partial_queries = pd.read_csv(checkpoint_dir / "final_test_query_results.partial.csv")
    partial_candidates = pd.read_csv(checkpoint_dir / "final_test_candidate_summaries.partial.csv")
    partial_exposed = pd.read_csv(checkpoint_dir / "final_test_exposed_candidates.partial.csv")
    expected_methods = set(METHODS)
    if partial_queries.duplicated(["method", "query_id"]).any():
        raise ValueError("Checkpoint has duplicate completed method/query rows")
    if not set(partial_queries["method"]).issubset(expected_methods):
        raise ValueError("Checkpoint contains an unknown method")
    if not set(partial_queries["query_id"]).issubset(set(saved_queries["query_id"])):
        raise ValueError("Checkpoint contains query IDs outside the frozen manifest")
    if partial_queries["status"].isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).any():
        raise ValueError("Checkpoint includes a failed implementation or schema query")
    expected_keys = set(zip(partial_queries["method"], partial_queries["query_id"]))
    candidate_keys = set(zip(partial_candidates["method"], partial_candidates["query_id"]))
    if len(partial_candidates) != len(partial_queries) or candidate_keys != expected_keys:
        raise ValueError("Candidate ledger does not cover every completed checkpoint query exactly once")
    if partial_exposed[["target_valid", "constraint_feasible"]].eq(False).any().any():
        raise ValueError("Checkpoint exposes a target-invalid or constraint-infeasible candidate")
    exposed_counts = partial_exposed.groupby(["method", "query_id"]).size().to_dict()
    for row in partial_queries.to_dict("records"):
        actual = int(exposed_counts.get((row["method"], row["query_id"]), 0))
        if actual != int(row["exposed_cf_count"]):
            raise ValueError("Checkpoint exposed-candidate count does not match its query ledger")
    for col, summary_col in [
        ("target_valid_availability", "target_count"),
        ("constraint_feasible_availability", "feasible_count"),
    ]:
        expected = partial_candidates[summary_col].astype(int).gt(0).to_numpy()
        actual = partial_queries[col].astype(bool).to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError("Checkpoint availability does not match candidate ledger: %s" % col)

    block_counts = {
        "%s|%s" % (transition, method): int(count)
        for (transition, method), count in partial_queries.groupby(["transition", "method"]).size().items()
    }
    if block_counts != checkpoint.get("completed_blocks"):
        raise ValueError("Checkpoint block manifest does not match completed query rows")
    if int(checkpoint.get("query_method_rows", -1)) != len(partial_queries):
        raise ValueError("Checkpoint row count does not match its manifest")

    write_json(out_dir / "resume_manifest.json", {
        "resumed_from": str(checkpoint_dir),
        "resumed_at_unix": time.time(),
        "checkpoint_query_method_rows": int(len(partial_queries)),
        "expected_query_method_rows": int(len(saved_queries) * len(METHODS)),
        "checkpoint_source_run": checkpoint.get("source_run"),
        "checkpoint_source_sha256": frozen.get("source_sha256"),
        "continuation_source_sha256": current_hashes,
        "allowed_source_differences": sorted(operational_paths),
        "checkpoint_compatibility_passed": True,
        "pilot_gate_reused": True,
        "model_and_split_reused": True,
    })
    write_json(out_dir / "run_status.json", {
        "status": "RESUMING_FROM_CHECKPOINT",
        "started_at_unix": time.time(),
        "completed_query_method_rows": int(len(partial_queries)),
        "expected_query_method_rows": int(len(saved_queries) * len(METHODS)),
    })
    print(
        "[resume] validated checkpoint %s rows=%d/%d; continuing only missing method/query pairs"
        % (checkpoint_dir, len(partial_queries), len(saved_queries) * len(METHODS)),
        flush=True,
    )

    final_query_rows, final_candidate_rows, final_exposed_rows = _run_stage(
        saved_queries, features, train_raw, train_encoded, train_predictions, bundle, schema,
        train_policy, mi_pairs, "final_test", out_dir,
        resume_data=(partial_queries, partial_candidates, partial_exposed),
    )
    final_summary, final_paired = _write_stage(
        out_dir, "final_test", final_query_rows, final_candidate_rows, final_exposed_rows
    )
    exposed_failures = int((~final_exposed_rows["target_valid"].fillna(False)).sum()) if len(final_exposed_rows) else 0
    constraint_failures = int((~final_exposed_rows["constraint_feasible"].fillna(False)).sum()) if len(final_exposed_rows) else 0
    status_counts = (
        final_query_rows.groupby(["transition", "method", "status"]).size().reset_index(name="count")
        if len(final_query_rows) else pd.DataFrame(columns=["transition", "method", "status", "count"])
    )
    runtime_or_schema_errors = int(final_query_rows["status"].isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).sum())
    timeout_count = int((final_query_rows["status"] == "TIMEOUT").sum())
    run_status = (
        "COMPLETE"
        if exposed_failures == 0 and constraint_failures == 0 and runtime_or_schema_errors == 0
        else "FAILED_EXPOSURE_OR_RUNTIME_GATE"
    )
    if run_status == "COMPLETE" and timeout_count:
        run_status = "COMPLETE_WITH_TIMEOUTS"
    report = {
        "status": run_status,
        "dataset": dataset_summary,
        "resumed_from_checkpoint": str(checkpoint_dir),
        "model_sanity_passed": True,
        "pilot_gate_passed": True,
        "final_test_query_target_pairs": int(len(saved_queries)),
        "final_test_rows": int(len(splits["test"])),
        "final_test_model_summary": json.loads((out_dir / "final_test_model_summary.json").read_text(encoding="utf-8")),
        "full_transition_rows": final_summary.to_dict(orient="records"),
        "paired_comparisons": final_paired.to_dict(orient="records"),
        "query_status_counts": status_counts.to_dict(orient="records"),
        "runtime_or_schema_error_count": runtime_or_schema_errors,
        "timeout_count": timeout_count,
        "exposed_target_validity_failures": exposed_failures,
        "exposed_constraint_failures": constraint_failures,
        "freeze_configuration_sha256": freeze["configuration_sha256"],
        "checkpoint_rows_reused": int(len(partial_queries)),
    }
    write_json(out_dir / "final_report.json", report)
    write_json(out_dir / "run_status.json", {"status": run_status, "finished_at_unix": time.time()})
    return {"status": run_status, "out_dir": str(out_dir), "report": report}

def run_experiment(data_path=DEFAULT_DATA_PATH, out_dir=None, no_download=False,
                   resume_from=None, mi_cache_path=None):
    root = Path(__file__).resolve().parents[2]
    if out_dir is None:
        out_dir = DEFAULT_OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = root / data_path

    features, labels, schema, dataset_summary = load_dataset(data_path, auto_download=not no_download)
    splits = split_dataset(features, labels, SPLIT_SEED)
    if resume_from is not None:
        if mi_cache_path is not None:
            raise ValueError("A resumed run must reuse the MI pairs frozen in its checkpoint")
        checkpoint_dir = Path(resume_from)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = root / checkpoint_dir
        return _resume_experiment(
            root, out_dir, checkpoint_dir, features, labels, schema, dataset_summary, splits
        )
    write_split_manifest(out_dir / "split_manifest.csv", splits, labels)
    train_rows = splits["train"]
    dev_rows = splits["dev"]
    test_rows = splits["test"]
    train_raw = features.iloc[train_rows].reset_index(drop=True)
    bundle, convergence_warnings = fit_model(train_raw, labels[train_rows], schema)

    split_summaries = {}
    split_predictions = {}
    model_gate_checks = {"no_convergence_warnings": not bool(convergence_warnings)}
    for name in ("train", "dev"):
        row_indexes = splits[name]
        split_features = features.iloc[row_indexes].reset_index(drop=True)
        summary = prediction_summary(bundle, split_features, labels[row_indexes])
        split_summaries[name] = summary
        split_predictions[name] = bundle.predict_raw(split_features)
        model_gate_checks["%s_all_predicted_classes_supported" % name] = all(
            summary["predicted_support"][CLASS_NAMES[class_id]] > 0 for class_id in range(3)
        )
    model_sanity = {
        "passed": bool(all(model_gate_checks.values())),
        "checks": model_gate_checks,
        "convergence_warnings": convergence_warnings,
        "split_metrics": split_summaries,
        "class_names": CLASS_NAMES,
    }
    write_json(out_dir / "model_sanity.json", model_sanity)
    joblib.dump(bundle, out_dir / "model_bundle.joblib")
    write_json(out_dir / "feature_schema.json", schema)

    train_policy = build_train_policy(train_raw, schema)
    write_json(out_dir / "constraint_policy.json", train_policy)
    if not model_sanity["passed"]:
        write_json(out_dir / "pilot_gate.json", {"passed": False, "blocked_at": "model_sanity", "model_sanity": model_sanity})
        return {"status": "STOP_MODEL_SANITY", "out_dir": str(out_dir)}

    space = EncodedSpace(train_raw, schema)
    train_encoded = space.encode(train_raw)
    train_predictions = bundle.predict_raw(train_raw)
    local_cache = out_dir / "mi_cache.json"
    if mi_cache_path is None:
        mi_artifact, mi_cache_reused = prepare_mi_cache(
            local_cache, train_encoded, train_rows, dataset_summary["source_sha256"]
        )
    else:
        external_cache = Path(mi_cache_path)
        if not external_cache.is_absolute():
            external_cache = root / external_cache
        mi_artifact = load_mi_cache(
            external_cache, train_encoded, train_rows, dataset_summary["source_sha256"]
        )
        if external_cache.resolve() != local_cache.resolve():
            shutil.copy2(str(external_cache), str(local_cache))
        mi_cache_reused = True
    mi_pairs = mi_artifact["mi_pairs"]
    dev_predictions = split_predictions["dev"]
    pilot_queries = _select_pilot_queries(dev_rows, dev_predictions)
    pilot_queries["source_class_name"] = pilot_queries["source_class"].map(CLASS_NAMES)
    pilot_queries["target_class_name"] = pilot_queries["target_class"].map(CLASS_NAMES)
    pilot_queries.to_csv(out_dir / "pilot_queries.csv", index=False)
    pilot_query_rows, pilot_candidate_rows, pilot_exposed_rows = _run_stage(
        pilot_queries, features, train_raw, train_encoded, train_predictions, bundle, schema,
        train_policy, mi_pairs, "pilot", out_dir,
    )
    pilot_summary, pilot_paired = _write_stage(
        out_dir, "pilot", pilot_query_rows, pilot_candidate_rows, pilot_exposed_rows
    )
    gate = _pilot_gate(pilot_queries, pilot_query_rows, pilot_candidate_rows, pilot_exposed_rows)
    gate["query_pairs"] = int(len(pilot_queries))
    gate["transition_summary_rows"] = int(len(pilot_summary))
    gate["paired_comparison_rows"] = int(len(pilot_paired))
    write_json(out_dir / "pilot_gate.json", gate)
    if not gate["passed"]:
        return {"status": "STOP_PILOT_GATE", "out_dir": str(out_dir), "pilot_gate": gate}

    freeze_payload = {
        "dataset": dataset_summary,
        "split_seed": SPLIT_SEED,
        "pilot_seed": PILOT_SEED,
        "target_mapping": CLASS_NAMES,
        "actionable_features": ACTIONABLE_FEATURES,
        "policy": train_policy,
        "max_candidates": MAX_CANDIDATES,
        "dice_sample_size": DICE_SAMPLE_SIZE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "checkpoint_interval_queries": CHECKPOINT_INTERVAL_QUERIES,
        "methods": METHODS,
        "mi_pairs": mi_pairs,
        "mi_cache_sha256": sha256_file(local_cache),
        "mi_precompute_seconds": mi_artifact["precompute_seconds"],
        "mi_cache_reused": mi_cache_reused,
        "source_sha256": _source_fingerprints(root),
        "model_bundle_sha256": sha256_file(out_dir / "model_bundle.joblib"),
        "versions": {
            "python": platform.python_version(),
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "scikit-learn": _safe_version("scikit-learn"),
            "dice-ml": _safe_version("dice-ml"),
            "scipy": _safe_version("scipy"),
        },
    }
    freeze_manifest = {
        "frozen": True,
        "frozen_at_unix": time.time(),
        "configuration_sha256": _canonical_hash(freeze_payload),
        "configuration": freeze_payload,
    }
    write_json(out_dir / "freeze_manifest.json", freeze_manifest)

    # The final test partition is first predicted only after the pilot passed
    # and the implementation/model/configuration have been frozen.
    final_test_features = features.iloc[test_rows].reset_index(drop=True)
    final_test_predictions = bundle.predict_raw(final_test_features)
    final_test_model_summary = prediction_summary(bundle, final_test_features, labels[test_rows])
    write_json(out_dir / "final_test_model_summary.json", final_test_model_summary)
    final_queries = _select_final_queries(test_rows, final_test_predictions)
    final_queries["source_class_name"] = final_queries["source_class"].map(CLASS_NAMES)
    final_queries["target_class_name"] = final_queries["target_class"].map(CLASS_NAMES)
    final_queries.to_csv(out_dir / "final_test_queries.csv", index=False)
    final_query_rows, final_candidate_rows, final_exposed_rows = _run_stage(
        final_queries, features, train_raw, train_encoded, train_predictions, bundle, schema,
        train_policy, mi_pairs, "final_test", out_dir,
    )
    final_summary, final_paired = _write_stage(
        out_dir, "final_test", final_query_rows, final_candidate_rows, final_exposed_rows
    )
    exposed_failures = int((~final_exposed_rows["target_valid"].fillna(False)).sum()) if len(final_exposed_rows) else 0
    constraint_failures = int((~final_exposed_rows["constraint_feasible"].fillna(False)).sum()) if len(final_exposed_rows) else 0
    status_counts = (
        final_query_rows.groupby(["transition", "method", "status"]).size().reset_index(name="count")
        if len(final_query_rows) else pd.DataFrame(columns=["transition", "method", "status", "count"])
    )
    runtime_or_schema_errors = int(final_query_rows["status"].isin(["RUNTIME_ERROR", "SCHEMA_ERROR"]).sum())
    timeout_count = int((final_query_rows["status"] == "TIMEOUT").sum())
    run_status = (
        "COMPLETE"
        if exposed_failures == 0 and constraint_failures == 0 and runtime_or_schema_errors == 0
        else "FAILED_EXPOSURE_OR_RUNTIME_GATE"
    )
    if run_status == "COMPLETE" and timeout_count:
        run_status = "COMPLETE_WITH_TIMEOUTS"
    report = {
        "status": run_status,
        "dataset": dataset_summary,
        "model_sanity_passed": bool(model_sanity["passed"]),
        "pilot_gate_passed": bool(gate["passed"]),
        "final_test_query_target_pairs": int(len(final_queries)),
        "final_test_rows": int(len(test_rows)),
        "final_test_model_summary": final_test_model_summary,
        "full_transition_rows": final_summary.to_dict(orient="records"),
        "paired_comparisons": final_paired.to_dict(orient="records"),
        "query_status_counts": status_counts.to_dict(orient="records"),
        "runtime_or_schema_error_count": runtime_or_schema_errors,
        "timeout_count": timeout_count,
        "exposed_target_validity_failures": exposed_failures,
        "exposed_constraint_failures": constraint_failures,
        "freeze_configuration_sha256": freeze_manifest["configuration_sha256"],
    }
    write_json(out_dir / "final_report.json", report)
    write_json(out_dir / "run_status.json", {"status": run_status, "finished_at_unix": time.time()})
    return {"status": run_status, "out_dir": str(out_dir), "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--resume-from", default=None, help="Continue missing query-target pairs from a validated checkpoint")
    parser.add_argument("--mi-cache", default=None, help="Reuse a precomputed train-only MI cache")
    args = parser.parse_args(argv)
    result = run_experiment(
        args.data_path, args.out_dir, args.no_download, args.resume_from, args.mi_cache
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
