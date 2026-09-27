"""Command-line runner for the frozen UPV-2025 external binary evaluation."""

import argparse
import json
import hashlib
import importlib.metadata
import os
import platform
import signal
import subprocess
import traceback
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .adapters import GenerationResult, make_adapter
from .config import (
    DESIRED_CLASS,
    METHODS,
    PRIMARY_SEED,
    RANDOM_SEED,
    REFERENCE_SIZES,
    REGIMES,
    ROBUSTNESS_SEEDS,
    SOURCE_CLASS,
    STUDENT_ID,
    TERMINAL_STATUSES,
    TIMEOUT_SECONDS,
)
from .constraints import build_constraint_policy, policy_for_query, write_constraint_policy
from .dataset import group_split, load_upv, write_json
from .metrics import mcnemar_table, paired_bootstrap_difference, status_from_results, summarize_queries
from .model import ModelBundle, fit_model, quality_gate, quality_metrics, save_model_metrics
from .verification import CandidateVerification, verify_candidate


class QueryTimeout(Exception):
    pass


@contextmanager
def _timeout(seconds):
    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)

    def handler(_signum, _frame):
        raise QueryTimeout()

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions():
    names = ("numpy", "pandas", "scikit-learn", "scipy", "dice-ml", "actionable-recourse", "joblib")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def select_queries(test_frame, bundle, max_queries=1000, seed=RANDOM_SEED):
    predictions = bundle.predict_raw(test_frame)
    eligible = test_frame.loc[predictions == SOURCE_CLASS].copy()
    eligible["_prediction"] = SOURCE_CLASS
    eligible = eligible.sort_values(["student_id", "source_year"] if "student_id" in eligible else list(eligible.columns[:1]))
    eligible = eligible.drop_duplicates("student_id", keep="first")
    if len(eligible) > max_queries:
        eligible = eligible.sample(n=max_queries, random_state=seed)
    eligible = eligible.sort_values("student_id").reset_index(drop=True)
    eligible["query_id"] = ["upv_q%05d" % index for index in range(len(eligible))]
    return eligible


def _distance(query, candidate, schema):
    numeric = schema["numeric_features"]
    query_values = query.iloc[0].loc[numeric] if isinstance(query, pd.DataFrame) else query.loc[numeric]
    candidate_values = candidate.iloc[0].loc[numeric] if isinstance(candidate, pd.DataFrame) else candidate.loc[numeric]
    left = pd.to_numeric(query_values, errors="coerce").fillna(0).to_numpy(dtype=float)
    right = pd.to_numeric(candidate_values, errors="coerce").fillna(0).to_numpy(dtype=float)
    return float(np.linalg.norm(right - left))


def _run_one_query(adapter, raw_query, query_id, regime_policy, bundle, feature_schema, method, regime_name, seed, timeout_seconds, max_candidates=5):
    generation = {"runtime_error": False, "timeout": False, "unsupported": False}
    started = time.perf_counter()
    generation_result = None
    try:
        with _timeout(timeout_seconds):
            generation_result = adapter.generate(
                raw_query.loc[:, feature_schema["feature_names"]],
                desired_class=DESIRED_CLASS,
                raw_constraint_policy=regime_policy,
                max_candidates=max_candidates,
                seed=seed,
                timeout_seconds=timeout_seconds,
            )
    except QueryTimeout:
        generation["timeout"] = True
        generation_result = GenerationResult()
    except Exception as exc:
        generation["runtime_error"] = True
        generation_result = GenerationResult(error_type=type(exc).__name__, unsupported_reason=str(exc), error_traceback=traceback.format_exc())
    runtime_ms = (time.perf_counter() - started) * 1000.0
    if runtime_ms < timeout_seconds * 1000.0 and generation["timeout"]:
        runtime_ms = timeout_seconds * 1000.0
    if generation_result is None:
        generation_result = GenerationResult()
    if generation_result.error_type == "QueryTimeout":
        generation["timeout"] = True
        generation_result.error_type = ""
    if generation_result.error_type:
        generation["runtime_error"] = True
    if generation_result.unsupported_reason and not generation_result.error_type:
        generation["unsupported"] = True
    candidates = generation_result.candidates if isinstance(generation_result.candidates, pd.DataFrame) else pd.DataFrame()
    verifications = []
    candidate_rows = []
    verification_started = time.perf_counter()
    for index in range(min(max_candidates, len(candidates))):
        returned_candidate = candidates.iloc[[index]].copy()
        if getattr(generation_result, "candidate_space", "encoded") == "raw":
            raw_candidate = returned_candidate
        else:
            try:
                raw_candidate = adapter.space.decode(returned_candidate)
            except Exception:
                raw_candidate = returned_candidate
        # Numeric imputation is many-to-one: the model-space median also
        # represents a raw missing value. Restore missing immutable factual
        # values during canonical reconstruction so imputation is never
        # mistaken for an action. The frozen preprocessor maps NaN back to the
        # same imputed model value during independent target verification.
        if isinstance(raw_candidate, pd.DataFrame) and len(raw_candidate):
            factual = raw_query.iloc[0]
            for feature, rule in regime_policy.get("features", {}).items():
                if rule.get("immutable", True) and pd.isna(factual.get(feature)):
                    raw_candidate.loc[raw_candidate.index[0], feature] = np.nan
        verification = verify_candidate(raw_query, raw_candidate, bundle, feature_schema, regime_policy)
        verifications.append(verification)
        row = {
            "query_id": query_id,
            "method": method,
            "constraint_regime": regime_name,
            "seed": int(seed),
            "candidate_id": int(index),
            "raw_reconstruction_ok": bool(verification.raw_reconstruction_ok),
            "categorical_domain_valid": bool(verification.categorical_domain_valid),
            "immutable_valid": bool(verification.immutable_valid),
            "bounds_valid": bool(verification.bounds_valid),
            "direction_valid": bool(verification.direction_valid),
            "target_valid": bool(verification.target_valid),
            "valid": bool(verification.valid),
            "constraint_feasible": bool(verification.constraint_feasible),
            "violations": ";".join(verification.violations),
            "numeric_distance": _distance(raw_query, pd.DataFrame([verification.raw_candidate]), feature_schema) if verification.raw_candidate else None,
        }
        row.update({"feature_%s" % key: value for key, value in verification.raw_candidate.items()})
        candidate_rows.append(row)
    verification_runtime_ms = (time.perf_counter() - verification_started) * 1000.0
    total_runtime_ms = runtime_ms + verification_runtime_ms
    status = status_from_results(generation, verifications)
    query_row = {
        "query_id": query_id,
        "method": method,
        "constraint_regime": regime_name,
        "seed": int(seed),
        "original_prediction": SOURCE_CLASS,
        "desired_class": DESIRED_CLASS,
        "returned_any_candidate": bool(verifications),
        "returned_any_valid_cf": bool(any(item.valid for item in verifications)),
        "returned_any_feasible_cf": bool(any(item.constraint_feasible for item in verifications)),
        "final_status": status,
        "runtime_ms": float(max(total_runtime_ms, timeout_seconds * 1000.0) if generation["timeout"] else total_runtime_ms),
        "generation_runtime_ms": float(max(runtime_ms, timeout_seconds * 1000.0) if generation["timeout"] else runtime_ms),
        "verification_runtime_ms": float(verification_runtime_ms),
        "candidate_count": int(len(verifications)),
        "valid_candidate_count": int(sum(item.valid for item in verifications)),
        "feasible_candidate_count": int(sum(item.constraint_feasible for item in verifications)),
        "changed_feature_count": int(min([
            sum(1 for feature in feature_schema["feature_names"]
                if str(item.raw_candidate.get(feature)) != str(raw_query.iloc[0].get(feature)))
            for item in verifications if item.constraint_feasible
        ] or [0])),
        "numeric_distance": float(min([row["numeric_distance"] for row in candidate_rows if row["numeric_distance"] is not None], default=np.nan)),
        "categorical_distance": 0.0,
        "immutable_violation": bool(any("immutable_violation" in violation for item in verifications for violation in item.violations)),
        "bound_violation": bool(any("numeric_bound_violation" in violation for item in verifications for violation in item.violations)),
        "domain_violation": bool(any("categorical_domain_violation" in violation or "schema_violation" in violation for item in verifications for violation in item.violations)),
        "direction_violation": bool(any("direction_violation" in violation for item in verifications for violation in item.violations)),
        "generated_candidate_count": int(generation_result.generated_candidate_count),
        "evaluated_candidate_count": int(generation_result.evaluated_candidate_count),
        "internally_rejected_count": int(generation_result.internally_rejected_count),
        "error_type": generation_result.error_type,
        "unsupported_reason": generation_result.unsupported_reason,
        "error_traceback": generation_result.error_traceback,
    }
    return query_row, candidate_rows


def run_matrix(raw_train, query_frame, bundle, feature_schema, policies, reference_frame, methods, regimes, seeds, timeout_seconds=TIMEOUT_SECONDS, max_queries=None):
    query_rows = []
    candidate_rows = []
    mi_cache = {}
    for regime_name in regimes:
        for seed in seeds:
            for method in methods:
                if method == "AR" and seed != PRIMARY_SEED:
                    continue
                cache_key = (regime_name, seed, len(reference_frame))
                shared_mi = mi_cache.get(cache_key) if method == "UFCE-FF3" else None
                adapter = make_adapter(
                    method, raw_train, reference_frame, bundle, feature_schema, policies,
                    regime_name, seed=seed, shared_mi=shared_mi,
                )
                if method == "UFCE-FF2":
                    mi_cache[cache_key] = (adapter.mi_pairs, adapter.mi_summary)
                for _, query in query_frame.iterrows():
                    query_id = query["query_id"]
                    policy = policy_for_query(policies[regime_name], query.to_frame().T, raw_train, feature_schema)
                    qrow, crows = _run_one_query(
                        adapter, query.to_frame().T, query_id, policy, bundle, feature_schema,
                        method, regime_name, seed, timeout_seconds,
                    )
                    qrow["setup_seconds"] = float(adapter.setup_seconds)
                    qrow["mi_estimator_calls"] = int(adapter.mi_summary.get("mi_estimator_calls", 0))
                    qrow["retained_mi_pairs"] = int(adapter.mi_summary.get("retained_pairs", 0))
                    query_rows.append(qrow)
                    candidate_rows.extend(crows)
    return pd.DataFrame(query_rows), pd.DataFrame(candidate_rows)


def pairwise_summary(query_results, methods, regimes, seed=PRIMARY_SEED):
    rows = []
    for regime in regimes:
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1:]:
                left_rows = query_results[(query_results.method == left) & (query_results.constraint_regime == regime) & (query_results.seed == seed)].sort_values("query_id").reset_index(drop=True)
                right_rows = query_results[(query_results.method == right) & (query_results.constraint_regime == regime) & (query_results.seed == seed)].sort_values("query_id").reset_index(drop=True)
                if len(left_rows) != len(right_rows) or not left_rows.query_id.equals(right_rows.query_id):
                    continue
                row = {"left_method": left, "right_method": right, "constraint_regime": regime, "seed": seed}
                for metric, column in (("valid_cf", "returned_any_valid_cf"), ("feasible_cf", "returned_any_feasible_cf")):
                    observed, ci = paired_bootstrap_difference(left_rows[column].astype(float).to_numpy(), right_rows[column].astype(float).to_numpy())
                    row["delta_%s_percentage_points" % metric] = observed * 100.0 if observed is not None else None
                    row["delta_%s_ci_low_percentage_points" % metric] = ci[0] * 100.0 if ci else None
                    row["delta_%s_ci_high_percentage_points" % metric] = ci[1] * 100.0 if ci else None
                observed, ci = paired_bootstrap_difference(left_rows.runtime_ms.to_numpy(), right_rows.runtime_ms.to_numpy(), statistic="median")
                row["median_latency_difference_ms"] = observed
                row["median_latency_ci_low_ms"] = ci[0] if ci else None
                row["median_latency_ci_high_ms"] = ci[1] if ci else None
                row["mcnemar_feasible"] = mcnemar_table(left_rows.returned_any_feasible_cf, right_rows.returned_any_feasible_cf)
                rows.append(row)
    return pd.DataFrame(rows)


def _nested_reference_pools(train_frame, sizes, seed=RANDOM_SEED):
    students = train_frame["student_id"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    pools = {}
    selected = []
    for size in sizes:
        if size == "full_train":
            pools[size] = train_frame.copy()
            continue
        target = int(size)
        current_rows = 0
        selected = []
        for student in students:
            selected.append(student)
            current_rows += int((train_frame.student_id == student).sum())
            if current_rows >= target:
                break
        pools[size] = train_frame[train_frame.student_id.isin(selected)].copy()
    return pools


def run_scaling(train_frame, queries, bundle, feature_schema, policies, out_dir, timeout_seconds=TIMEOUT_SECONDS):
    scale_queries = queries.head(250).copy()
    pools = _nested_reference_pools(train_frame, REFERENCE_SIZES)
    rows = []
    mi_rows = []
    for reference_size in REFERENCE_SIZES:
        reference = pools[reference_size]
        mi_cache = {}
        for method in METHODS:
            shared_mi = mi_cache.get(reference_size) if method == "UFCE-FF3" else None
            adapter = make_adapter(
                method, train_frame, reference, bundle, feature_schema, policies,
                "MODERATE", seed=0, shared_mi=shared_mi,
            )
            if method == "UFCE-FF2":
                mi_cache[reference_size] = (adapter.mi_pairs, adapter.mi_summary)
            started = time.perf_counter()
            scale_query_rows = []
            for _, query in scale_queries.iterrows():
                policy = policy_for_query(policies["MODERATE"], query.to_frame().T, train_frame, feature_schema)
                qrow, _ = _run_one_query(
                    adapter, query.to_frame().T, query["query_id"], policy, bundle,
                    feature_schema, method, "MODERATE", 0, timeout_seconds,
                )
                scale_query_rows.append(qrow)
            qdf = pd.DataFrame(scale_query_rows)
            elapsed = time.perf_counter() - started
            valid = int(qdf.returned_any_valid_cf.sum()) if len(qdf) else 0
            feasible = int(qdf.returned_any_feasible_cf.sum()) if len(qdf) else 0
            latency = pd.to_numeric(qdf.runtime_ms, errors="coerce") if len(qdf) else pd.Series(dtype=float)
            rows.append({
                "method": method,
                "constraint_regime": "MODERATE",
                "seed": 0,
                "reference_size_requested": reference_size,
                "reference_rows_actual": int(len(reference)),
                "reference_students_actual": int(reference.student_id.nunique()),
                "setup_seconds": float(adapter.setup_seconds),
                "total_query_seconds": float(elapsed),
                "n_queries": int(len(qdf)),
                "valid_availability": float(valid / len(qdf)) if len(qdf) else None,
                "feasible_availability": float(feasible / len(qdf)) if len(qdf) else None,
                "median_latency_ms": float(latency.median()) if len(latency) else None,
                "p95_latency_ms": float(latency.quantile(0.95)) if len(latency) else None,
                "median_candidate_count": float(qdf.candidate_count.median()) if len(qdf) else None,
                "peak_rss_mb": None,
            })
            mi_rows.append({
                "method": method,
                "reference_size_requested": reference_size,
                "reference_rows_actual": int(len(reference)),
                "mi_estimator_calls": int(adapter.mi_summary.get("mi_estimator_calls", 0)),
                "retained_pairs": int(adapter.mi_summary.get("retained_pairs", 0)),
                "mi_seconds": float(adapter.mi_summary.get("mi_seconds", 0.0)),
            })
    pd.DataFrame(rows).to_csv(out_dir / "scaling_summary.csv", index=False)
    pd.DataFrame(mi_rows).to_csv(out_dir / "mi_scaling_summary.csv", index=False)


def write_report(out_dir, summaries, gate):
    lines = ["# UFCE External Binary Evaluation", "", "Frozen Phase-1 UPV-2025 binary contract.", "", "## Quality gate", "", "- Status: `%s`" % gate["status"]]
    if gate.get("failures"):
        lines.append("- Failures: `%s`" % ", ".join(gate["failures"]))
    if summaries:
        lines.extend(["", "## Primary summaries", "", "| Method | Regime | Valid availability | Feasible availability | Median ms | P95 ms | ECCF ms |", "|---|---|---:|---:|---:|---:|---:|"])
        for row in summaries:
            lines.append("| {method} | {constraint_regime} | {valid_cf_availability_rate} | {feasible_cf_availability_rate} | {median_latency_ms} | {p95_latency_ms} | {effective_compute_cost_per_feasible_query_ms} |".format(**row))
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, schema, dataset_summary = load_upv(args.data_dir, auto_download=not args.no_download, force_download=args.force_download)
    # Keep the source identifier (`dni_hash`) in the audited schema while using
    # a stable metadata alias throughout the query/reference pipeline.
    frame["student_id"] = frame[STUDENT_ID].astype(str)
    partition, split_manifest = group_split(frame, RANDOM_SEED)
    frame = frame.copy()
    frame["partition"] = partition.to_numpy()
    train = frame[frame.partition == "train"].copy()
    validation = frame[frame.partition == "validation"].copy()
    test = frame[frame.partition == "test"].copy()
    checkpoint_path = out_dir / "model_bundle_checkpoint.joblib"
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        pipeline = joblib.load(checkpoint_path)
        bundle = ModelBundle(
            feature_names=list(schema["feature_names"]),
            numeric_features=list(schema["numeric_features"]),
            categorical_features=list(schema["categorical_features"]),
            preprocessor=pipeline.named_steps["preprocessor"],
            model=pipeline,
        )
        prior_gate_path = out_dir / "model_quality_gate.json"
        if prior_gate_path.exists():
            with open(prior_gate_path, "r", encoding="utf-8") as handle:
                convergence_warnings = list(json.load(handle).get("convergence_warnings", []))
        else:
            convergence_warnings = []
    else:
        bundle, convergence_warnings = fit_model(train, schema, train.y)
        # Persist the expensive fitted pipeline before query selection/CF work
        # so a later resume can audit the model artifact even if a downstream
        # stage fails.
        joblib.dump(bundle.model, checkpoint_path)
    queries = select_queries(test, bundle, args.max_queries, RANDOM_SEED)
    metrics = quality_metrics(bundle, test.loc[:, schema["feature_names"]], test.y, queries.student_id.nunique())
    gate = quality_gate(metrics, convergence_warnings)
    write_json(out_dir / "dataset_summary.json", dataset_summary)
    write_json(out_dir / "feature_schema.json", schema)
    split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    queries[["query_id", "student_id", "source_year"]].to_csv(out_dir / "query_ids.csv", index=False)
    save_model_metrics(out_dir / "model_metrics.json", metrics, gate)
    write_json(out_dir / "model_quality_gate.json", gate)
    model_path = out_dir / "model_bundle.joblib"
    joblib.dump(bundle.model, model_path)
    policies = build_constraint_policy(train, schema)
    write_constraint_policy(out_dir / "constraint_policy.json", policies)
    config = {
        "dataset": "UPV-2025",
        "phase": 1,
        "label_mapping": {"A": 0, "B": 1},
        "source_class": SOURCE_CLASS,
        "desired_class": DESIRED_CLASS,
        "methods": list(METHODS),
        "regimes": [regime.name for regime in REGIMES],
        "primary_seed": PRIMARY_SEED,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_candidates": 5,
        "max_queries": args.max_queries,
        "reference_sizes": list(REFERENCE_SIZES),
        "statistical_contract": {"primary_seed_only": True, "wilson": True, "paired_bootstrap_resamples": 2000, "paired_bootstrap_seed": 42},
    }
    write_json(out_dir / "experiment_config.json", config)
    manifest = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "dataset_dir": str(Path(args.data_dir).resolve()),
        "out_dir": str(out_dir),
        "dataset_checksums": dataset_summary.get("source_manifest", []),
        "model_artifact": str(model_path.resolve()),
        "model_artifact_sha256": _sha256(model_path),
        "feature_counts": {
            "raw": len(schema["feature_names"]),
            "numeric": len(schema["numeric_features"]),
            "categorical": len(schema["categorical_features"]),
            "encoded": int(bundle.transform_raw(train.iloc[:1]).shape[1]),
        },
        "seeds": {"split": RANDOM_SEED, "primary": PRIMARY_SEED, "robustness": list(ROBUSTNESS_SEEDS)},
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    if not gate["passed"]:
        write_report(out_dir, [], gate)
        raise RuntimeError("Predictive-model quality gate did not pass: %s" % gate["status"])
    if args.stage == "audit":
        write_report(out_dir, [], gate)
        return out_dir

    raw_train = train.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy()
    query_frame = queries.copy()
    if args.stage in ("pilot", "primary", "robustness", "all"):
        pilot_queries = query_frame.head(20) if args.stage == "pilot" else query_frame
        seeds = (PRIMARY_SEED,) if args.stage in ("pilot", "primary") else (PRIMARY_SEED,) + tuple(ROBUSTNESS_SEEDS)
        methods = METHODS
        regimes = ("MODERATE",) if args.stage == "pilot" else tuple(regime.name for regime in REGIMES)
        qdf, cdf = run_matrix(raw_train, pilot_queries, bundle, schema, policies, raw_train, methods, regimes, seeds, TIMEOUT_SECONDS)
        qdf.to_csv(out_dir / "query_results.csv", index=False)
        cdf.to_csv(out_dir / "candidate_results.csv", index=False)
        summaries = []
        for regime in regimes:
            for method in methods:
                summaries.append(summarize_queries(qdf, cdf, method, regime, PRIMARY_SEED))
        pd.DataFrame(summaries).to_csv(out_dir / "method_summary.csv", index=False)
        pairwise_summary(qdf, methods, regimes, PRIMARY_SEED).to_csv(out_dir / "pairwise_comparison.csv", index=False)
        pd.DataFrame([{"constraint_regime": name, "n_actionable_features": policies[name]["n_actionable_features"], "actionable_features": ";".join(policies[name]["actionable_features"])} for name in policies]).to_csv(out_dir / "constraint_regime_summary.csv", index=False)
        if len(cdf) and "violations" in cdf.columns:
            cdf.loc[cdf.violations.astype(str).str.len() > 0].to_csv(out_dir / "violations.csv", index=False)
        else:
            pd.DataFrame().to_csv(out_dir / "violations.csv", index=False)
        qdf.loc[qdf.final_status.isin(["RUNTIME_ERROR", "TIMEOUT", "UNSUPPORTED", "RETURNED_NON_FLIPPING_CF"])].to_csv(out_dir / "failures.csv", index=False)
        write_report(out_dir, summaries, gate)
    if args.stage in ("scaling", "all"):
        run_scaling(raw_train, query_frame, bundle, schema, policies, out_dir, TIMEOUT_SECONDS)
    return out_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "pilot", "primary", "robustness", "scaling", "all"), default="audit")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
