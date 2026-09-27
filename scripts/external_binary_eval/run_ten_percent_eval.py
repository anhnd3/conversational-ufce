"""Run a resumable 10%-query UPV external evaluation from the frozen checkpoint."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from scipy import sparse

from .adapters import make_adapter
from .config import METHODS, PRIMARY_SEED, RANDOM_SEED, REGIMES, SOURCE_CLASS, STUDENT_ID, TIMEOUT_SECONDS
from .constraints import build_constraint_policy, policy_for_query, write_constraint_policy
from .dataset import group_split, load_upv, write_json
from .metrics import summarize_queries
from .model import ModelBundle
from .runner import _run_one_query, pairwise_summary


DEFAULT_OUT_DIR = "outputs/external_binary_eval/upv_2025_10pct_5fold"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _append_csv(row_or_rows, path):
    path = Path(path)
    rows = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _student_reference_subset(train, target_rows=10000, seed=RANDOM_SEED):
    counts = train.groupby("student_id", sort=False).size()
    shuffled_students = counts.sample(frac=1.0, random_state=seed)
    chosen = shuffled_students.cumsum()
    selected_students = chosen.index[chosen.shift(fill_value=0) < target_rows].tolist()
    if not selected_students:
        selected_students = [shuffled_students.index[0]]
    return train.loc[train.student_id.isin(selected_students)].copy()


def prepare_data(data_dir, out_dir):
    frame, schema, dataset_summary = load_upv(data_dir, auto_download=False)
    frame["student_id"] = frame[STUDENT_ID].astype(str)
    partition, split_manifest = group_split(frame, RANDOM_SEED)
    frame = frame.copy()
    frame["partition"] = partition.to_numpy()
    train = frame.loc[frame.partition == "train"].copy()
    test = frame.loc[frame.partition == "test"].copy()
    checkpoint_path = Path("outputs/external_binary_eval/upv_2025/model_bundle_checkpoint.joblib")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Missing accepted model snapshot: %s" % checkpoint_path)
    pipeline = joblib.load(checkpoint_path)
    bundle = ModelBundle(
        feature_names=list(schema["feature_names"]),
        numeric_features=list(schema["numeric_features"]),
        categorical_features=list(schema["categorical_features"]),
        preprocessor=pipeline.named_steps["preprocessor"],
        model=pipeline,
    )
    test["original_prediction"] = bundle.predict_raw(test)
    eligible = test.loc[test.original_prediction == SOURCE_CLASS].copy()
    eligible = eligible.sort_values(["student_id", "source_year"]).drop_duplicates("student_id", keep="first")
    sample_size = max(1, int(math.ceil(0.10 * len(eligible))))
    queries = eligible.sample(n=sample_size, random_state=RANDOM_SEED).sort_values("student_id").reset_index(drop=True)
    if int(queries.y.nunique()) < 2 or int(queries.y.value_counts().min()) < 5:
        raise ValueError("Cannot stratify the requested query sample into five folds")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_ids = np.zeros(len(queries), dtype=int)
    for fold_index, (_, fold_test_indices) in enumerate(splitter.split(np.zeros(len(queries)), queries.y), start=1):
        fold_ids[fold_test_indices] = fold_index
    queries["fold_id"] = fold_ids
    queries["query_id"] = ["upv10_q%05d" % index for index in range(len(queries))]
    policies = build_constraint_policy(train, schema)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "dataset_summary.json", dataset_summary)
    write_json(out_dir / "feature_schema.json", schema)
    write_constraint_policy(out_dir / "constraint_policy.json", policies)
    _atomic_csv(split_manifest, out_dir / "split_manifest.csv")
    _atomic_csv(queries[["query_id", "student_id", "source_year", "fold_id", "y", "original_prediction"] + schema["feature_names"]], out_dir / "query_ids.csv")
    model_metrics_path = Path("outputs/external_binary_eval/upv_2025/model_metrics.json")
    model_gate_path = Path("outputs/external_binary_eval/upv_2025/model_quality_gate.json")
    if model_metrics_path.exists():
        with open(model_metrics_path, "r", encoding="utf-8") as handle:
            write_json(out_dir / "model_metrics.json", json.load(handle))
    if model_gate_path.exists():
        with open(model_gate_path, "r", encoding="utf-8") as handle:
            write_json(out_dir / "model_quality_gate.json", json.load(handle))
    return frame, train, test, queries, schema, dataset_summary, bundle, policies, checkpoint_path


def _run_stage(stage, data_dir, out_dir, timeout_seconds=TIMEOUT_SECONDS, methods=METHODS,
               pilot_reference_size=10000, primary_reference_size=10000):
    out_dir = Path(out_dir).resolve()
    (frame, train, test, queries, schema, dataset_summary,
     bundle, policies, checkpoint_path) = prepare_data(data_dir, out_dir)
    if stage == "pilot":
        reference = train.copy() if pilot_reference_size == "full_train" else _student_reference_subset(
            train, int(pilot_reference_size), seed=RANDOM_SEED
        )
        stage_queries = queries.head(20).copy()
        regimes = ("MODERATE",)
    else:
        reference = train.copy() if primary_reference_size == "full_train" else _student_reference_subset(
            train, int(primary_reference_size), seed=RANDOM_SEED
        )
        stage_queries = queries.copy()
        regimes = tuple(regime.name for regime in REGIMES)

    config = {
        "dataset": "UPV-2025",
        "stage": stage,
        "query_sampling": "ceil(10% of unique test students predicted source class 0)",
        "eligible_unique_students_predicted_0": int(test.loc[test.original_prediction == SOURCE_CLASS, "student_id"].nunique()),
        "sampled_queries": int(len(queries)),
        "stage_queries": int(len(stage_queries)),
        "folds": 5,
        "fold_definition": "StratifiedKFold over sampled unique students, stratified by observed data label; each query is evaluated once.",
        "source_class": SOURCE_CLASS,
        "desired_class": 1,
        "methods": list(methods),
        "regimes": list(regimes),
        "seed": PRIMARY_SEED,
        "reference_rows": int(len(reference)),
        "reference_students": int(reference.student_id.nunique()),
        "pilot_reference_size_requested": pilot_reference_size if stage == "pilot" else None,
        "primary_reference_size_requested": primary_reference_size if stage == "primary" else None,
        "model_fit_train_rows": int(len(train)),
        "max_candidates": 5,
        "timeout_seconds": float(timeout_seconds),
        "predictive_quality_gate_override": "USER_ACCEPTED_EXISTING_LOGISTIC_REGRESSION_SNAPSHOT_DESPITE_CONVERGENCE_WARNING",
        "primary_model_checkpoint": str(checkpoint_path.resolve()),
        "primary_model_sha256": _sha256(checkpoint_path),
        "constraint_policy": "frozen training-derived STRICT/MODERATE/FLEXIBLE from v1.2",
        "shared_mi_setup": "one MI computation shared by UFCE-FF2 and UFCE-FF3 for each regime/reference/seed",
        "ufce_n_neighbors": 1000,
        "note": "The automated v1.2 gate remains STOP/CONVERGENCE_WARNING; this run follows the user's decision to evaluate the saved snapshot.",
    }
    write_json(out_dir / "experiment_config.json", config)
    write_json(out_dir / "run_manifest.json", {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_checksums": dataset_summary.get("source_manifest", []),
        "model_checkpoint": str(checkpoint_path.resolve()),
        "model_checkpoint_sha256": _sha256(checkpoint_path),
        "query_count": int(len(stage_queries)),
        "reference_rows": int(len(reference)),
        "raw_feature_count": int(len(schema["feature_names"])),
        "encoded_feature_count": int(bundle.transform_raw(train.iloc[:1]).shape[1]),
        "stage": stage,
    })

    raw_train = train.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy()
    query_columns = ["query_id", "fold_id", "student_id", "y", "source_year"] + schema["feature_names"]
    existing_queries_path = out_dir / "query_results.csv"
    completed = set()
    if existing_queries_path.exists():
        prior = pd.read_csv(existing_queries_path)
        completed = set(zip(prior.query_id.astype(str), prior.method.astype(str), prior.constraint_regime.astype(str), prior.seed.astype(int)))
    setup_path = out_dir / "adapter_setup.csv"
    setup_keys = set()
    if setup_path.exists():
        prior_setup = pd.read_csv(setup_path)
        setup_keys = set(zip(prior_setup.method.astype(str), prior_setup.constraint_regime.astype(str)))

    for regime_name in regimes:
        mi_cache = None
        for method in methods:
            shared_mi = mi_cache if method == "UFCE-FF3" else None
            adapter = make_adapter(
                method, raw_train, reference, bundle, schema, policies,
                regime_name, seed=PRIMARY_SEED, shared_mi=shared_mi,
            )
            if method == "UFCE-FF2":
                mi_cache = (adapter.mi_pairs, adapter.mi_summary)
            setup_row = {
                "method": method,
                "constraint_regime": regime_name,
                "reference_rows": int(len(reference)),
                "reference_students": int(reference.student_id.nunique()),
                "setup_seconds": float(adapter.setup_seconds),
                "mi_seconds": float(adapter.mi_summary.get("mi_seconds", 0.0)) if method == "UFCE-FF2" else 0.0,
                "mi_estimator_calls": int(adapter.mi_summary.get("mi_estimator_calls", 0)) if method == "UFCE-FF2" else 0,
                "retained_mi_pairs": int(adapter.mi_summary.get("retained_pairs", 0)),
                "mi_reused_from": "UFCE-FF2" if method == "UFCE-FF3" else "",
                "setup_error": getattr(adapter, "setup_error", ""),
            }
            if (method, regime_name) not in setup_keys:
                _append_csv(setup_row, setup_path)
                setup_keys.add((method, regime_name))

            for _, query in stage_queries.iterrows():
                key = (str(query.query_id), method, regime_name, PRIMARY_SEED)
                if key in completed:
                    continue
                raw_query = query.loc[query_columns].to_frame().T
                policy = policy_for_query(policies[regime_name], raw_query, train, schema)
                query_row, candidate_rows = _run_one_query(
                    adapter, raw_query, query.query_id, policy, bundle, schema,
                    method, regime_name, PRIMARY_SEED, timeout_seconds,
                )
                query_row["fold_id"] = int(query.fold_id)
                query_row["observed_label"] = int(query.y)
                query_row["setup_seconds"] = float(adapter.setup_seconds)
                query_row["mi_estimator_calls"] = int(setup_row["mi_estimator_calls"])
                query_row["retained_mi_pairs"] = int(setup_row["retained_mi_pairs"])
                _append_csv(query_row, existing_queries_path)
                for candidate in candidate_rows:
                    candidate["fold_id"] = int(query.fold_id)
                _append_csv(candidate_rows, out_dir / "candidate_results.csv")
                completed.add(key)
                if len(completed) % 10 == 0:
                    print("stage=%s method=%s regime=%s done=%d/%d" % (
                        stage, method, regime_name, len(completed),
                        len(stage_queries) * len(methods) * len(regimes),
                    ), flush=True)

    _finalize(out_dir, methods=methods, regimes=regimes, seed=PRIMARY_SEED,
              train=train, schema=schema, bundle=bundle, policies=policies,
              reference=reference)
    return out_dir


def _six_metric_characterization(out_dir, train, schema, bundle, policies, reference):
    candidate_path = Path(out_dir) / "candidate_results.csv"
    query_path = Path(out_dir) / "query_ids.csv"
    if not candidate_path.exists() or not query_path.exists():
        return
    candidates = pd.read_csv(candidate_path)
    queries = pd.read_csv(query_path)
    report_regimes = ("MODERATE",) if Path(out_dir).name == "pilot" else tuple(item.name for item in REGIMES)

    def write_empty_summaries(note):
        rows = [{
            "method": method,
            "constraint_regime": regime,
            "valid_returned_candidates": 0,
            "prox_jac_mean": None,
            "prox_euc_mean": None,
            "sparsity_mean": None,
            "actionability_pass_rate": None,
            "plausibility_pass_rate": None,
            "feasibility_pass_rate": None,
            "metric_contract": note,
        } for method in METHODS for regime in report_regimes]
        _atomic_csv(pd.DataFrame(), Path(out_dir) / "six_metrics_candidate.csv")
        _atomic_csv(pd.DataFrame(rows), Path(out_dir) / "six_metric_summary.csv")

    if candidates.empty:
        write_empty_summaries("no candidates returned")
        return
    candidates = candidates[candidates.target_valid.fillna(False)].copy()
    if candidates.empty:
        write_empty_summaries("no independently target-valid candidates")
        return

    feature_names = list(schema["feature_names"])
    numeric = list(schema["numeric_features"])
    categorical = list(schema["categorical_features"])
    query_by_id = queries.drop_duplicates("query_id").set_index("query_id")
    reference_sample = _student_reference_subset(train, 10000, seed=RANDOM_SEED)
    reference_x = bundle.transform_raw(reference_sample)
    lof_neighbors = min(100, max(1, len(reference_sample) - 1))
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
    lof_started = time.perf_counter()
    lof.fit(reference_x)
    lof_fit_seconds = float(time.perf_counter() - lof_started)

    raw_candidates = []
    raw_queries = []
    for _, row in candidates.iterrows():
        raw_candidates.append({name: row.get("feature_" + name, np.nan) for name in feature_names})
        original = query_by_id.loc[row.query_id]
        raw_queries.append({name: original[name] for name in feature_names})
    candidate_frame = pd.DataFrame(raw_candidates, columns=feature_names)
    query_frame = pd.DataFrame(raw_queries, columns=feature_names)
    candidate_x = bundle.transform_raw(candidate_frame)
    candidate_plausible = lof.predict(candidate_x) == 1
    result_rows = []
    for index, (_, candidate) in enumerate(candidates.iterrows()):
        query = query_frame.iloc[index]
        cf = candidate_frame.iloc[index]
        regime = candidate.constraint_regime
        changed = []
        for feature in feature_names:
            left, right = query[feature], cf[feature]
            left_missing, right_missing = bool(pd.isna(left)), bool(pd.isna(right))
            if left_missing and right_missing:
                different = False
            elif left_missing or right_missing:
                different = True
            elif feature in categorical:
                different = str(left) != str(right)
            else:
                try:
                    different = not np.isclose(float(left), float(right), atol=1e-5, rtol=0)
                except Exception:
                    different = str(left) != str(right)
            if different:
                changed.append(feature)
        changed_categories = [feature for feature in categorical if feature in changed]
        cat_jaccard = (2.0 * len(changed_categories) / (len(categorical) + len(changed_categories))) if categorical else 0.0
        query_x = bundle.transform_raw(pd.DataFrame([query.to_dict()], columns=feature_names))
        cf_x = candidate_x[index]
        delta_x = query_x - cf_x
        if sparse.issparse(delta_x):
            normalized_euclidean = float(np.sqrt(delta_x.multiply(delta_x).sum()))
        else:
            normalized_euclidean = float(np.linalg.norm(delta_x))
        allowed = set(policies[regime]["actionable_features"])
        actionability_pass = all(feature in allowed for feature in changed)
        plausible = bool(candidate_plausible[index])
        constraint_feasible = bool(candidate.constraint_feasible)
        feasible_style_pass = bool(constraint_feasible and actionability_pass and plausible and len(changed) >= 2)
        result_rows.append({
            "query_id": candidate.query_id,
            "fold_id": int(candidate.fold_id),
            "method": candidate.method,
            "constraint_regime": regime,
            "candidate_id": int(candidate.candidate_id),
            "target_valid": True,
            "prox_jac_nominal": float(cat_jaccard),
            "prox_euc_preprocessed_l2": normalized_euclidean,
            "sparsity_raw_changed_features": int(len(changed)),
            "actionability_pass": bool(actionability_pass),
            "plausibility_lof_10k_reference": plausible,
            "feasibility_style_pass": feasible_style_pass,
            "hard_constraint_feasible": constraint_feasible,
            "n_changed_actionable_features": int(sum(feature in allowed for feature in changed)),
            "lof_reference_rows": int(len(reference_sample)),
            "lof_n_neighbors": int(lof_neighbors),
            "lof_setup_seconds": lof_fit_seconds,
        })
    candidate_metrics = pd.DataFrame(result_rows)
    _atomic_csv(candidate_metrics, Path(out_dir) / "six_metrics_candidate.csv")
    rows = []
    candidate_groups = {
        key: group for key, group in candidate_metrics.groupby(["method", "constraint_regime"], sort=False)
    }
    for method in METHODS:
        for regime in report_regimes:
            group = candidate_groups.get((method, regime))
            if group is None or group.empty:
                rows.append({
                    "method": method,
                    "constraint_regime": regime,
                    "valid_returned_candidates": 0,
                    "prox_jac_mean": None,
                    "prox_euc_mean": None,
                    "sparsity_mean": None,
                    "actionability_pass_rate": None,
                    "plausibility_pass_rate": None,
                    "feasibility_pass_rate": None,
                    "metric_contract": "no independently valid returned candidates",
                })
                continue
            rows.append({
                "method": method,
                "constraint_regime": regime,
                "valid_returned_candidates": int(len(group)),
                "prox_jac_mean": float(group.prox_jac_nominal.mean()),
                "prox_euc_mean": float(group.prox_euc_preprocessed_l2.mean()),
                "sparsity_mean": float(group.sparsity_raw_changed_features.mean()),
                "actionability_pass_rate": float(group.actionability_pass.mean()),
                "plausibility_pass_rate": float(group.plausibility_lof_10k_reference.mean()),
                "feasibility_pass_rate": float(group.feasibility_style_pass.mean()),
                "metric_contract": "adapted descriptive definitions in report.md; no comparison to Table 7 author values",
            })
    _atomic_csv(pd.DataFrame(rows), Path(out_dir) / "six_metric_summary.csv")


def _finalize(out_dir, methods, regimes, seed, train, schema, bundle, policies, reference):
    out_dir = Path(out_dir)
    queries = pd.read_csv(out_dir / "query_results.csv") if (out_dir / "query_results.csv").exists() else pd.DataFrame()
    candidates = pd.read_csv(out_dir / "candidate_results.csv") if (out_dir / "candidate_results.csv").exists() else pd.DataFrame()
    if queries.empty:
        return
    method_rows = []
    fold_rows = []
    for regime in regimes:
        for method in methods:
            method_rows.append(summarize_queries(queries, candidates, method, regime, seed))
            for fold_id in sorted(queries.fold_id.unique()):
                qf = queries[queries.fold_id == fold_id]
                cf = candidates[candidates.fold_id == fold_id] if len(candidates) else candidates
                summary = summarize_queries(qf, cf, method, regime, seed)
                summary["fold_id"] = int(fold_id)
                fold_rows.append(summary)
    method_summary = pd.DataFrame(method_rows)
    _atomic_csv(method_summary, out_dir / "method_summary.csv")
    _atomic_csv(pd.DataFrame(fold_rows), out_dir / "fold_summary.csv")
    pairwise = pairwise_summary(queries, list(methods), list(regimes), seed)
    _atomic_csv(pairwise, out_dir / "pairwise_comparison.csv")
    # Report a separate summary alias for convenient filtering by regime.
    _atomic_csv(method_summary, out_dir / "constraint_regime_summary.csv")
    _atomic_csv(queries[queries.final_status != "SUCCESS_VALID_FEASIBLE"], out_dir / "failures.csv")
    if len(candidates):
        violation_rows = []
        for _, row in candidates.iterrows():
            if row.get("violations"):
                violation_rows.append(row.to_dict())
        _atomic_csv(pd.DataFrame(violation_rows), out_dir / "violations.csv")
    _six_metric_characterization(out_dir, train, schema, bundle, policies, reference)

    setup = pd.read_csv(out_dir / "adapter_setup.csv") if (out_dir / "adapter_setup.csv").exists() else pd.DataFrame()
    if len(setup):
        setup_summary = setup.rename(columns={"reference_rows": "reference_rows_actual", "reference_students": "reference_students_actual"}).copy()
        for _, row in method_summary.iterrows():
            mask = (setup_summary.method == row.method) & (setup_summary.constraint_regime == row.constraint_regime)
            setup_summary.loc[mask, "n_queries"] = row.n_queries
            setup_summary.loc[mask, "valid_availability"] = row.valid_cf_availability_rate
            setup_summary.loc[mask, "feasible_availability"] = row.feasible_cf_availability_rate
            setup_summary.loc[mask, "median_latency_ms"] = row.median_latency_ms
            setup_summary.loc[mask, "p95_latency_ms"] = row.p95_latency_ms
        _atomic_csv(setup_summary, out_dir / "scaling_summary.csv")
        _atomic_csv(setup[["method", "reference_rows", "mi_seconds", "mi_estimator_calls", "retained_mi_pairs"]], out_dir / "mi_scaling_summary.csv")

    lines = [
        "# UPV-2025 external binary evaluation: 10% query sample",
        "",
        "The saved v1.2 LogisticRegression checkpoint was used after the user accepted its predictive performance for this external CF evaluation. The automated convergence gate remains recorded as STOP/CONVERGENCE_WARNING.",
        "",
        "The sample contains 10% of the unique held-out students predicted as source class 0; each student appears once. Five stratified folds split those fixed query IDs and each query is evaluated once. The model fit uses the full training partition; the counterfactual reference pool is a student-group sample from it, with size recorded in `experiment_config.json`.",
        "",
        "## Primary metrics (pooled query estimates)",
        "",
        "| Method | Regime | Valid availability | Constraint-feasible availability | Exposed validity | Constraint compliance | Median ms | P95 ms | ECCF ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in method_summary.iterrows():
        def fmt(key):
            value = row.get(key)
            return "NA" if pd.isna(value) else ("%.3f" % float(value) if "rate" in key else "%.1f" % float(value))
        lines.append("| %s | %s | %s (%s/%s) | %s (%s/%s) | %s (%s/%s) | %s (%s/%s) | %s | %s | %s |" % (
            row.method, row.constraint_regime,
            fmt("valid_cf_availability_rate"), fmt("valid_cf_availability_numerator"), fmt("valid_cf_availability_denominator"),
            fmt("feasible_cf_availability_rate"), fmt("feasible_cf_availability_numerator"), fmt("feasible_cf_availability_denominator"),
            fmt("exposed_cf_validity_rate"), fmt("exposed_cf_validity_numerator"), fmt("exposed_cf_validity_denominator"),
            fmt("constraint_satisfaction_rate"), fmt("constraint_satisfaction_numerator"), fmt("constraint_satisfaction_denominator"),
            fmt("median_latency_ms"), fmt("p95_latency_ms"), fmt("effective_compute_cost_per_feasible_query_ms"),
        ))
    lines.extend([
        "",
        "## Secondary six-metric characterization",
        "",
        "See `six_metric_summary.csv`. These are explicitly adapted descriptive metrics: Prox-Jac is nominal one-hot Jaccard distance over raw categories; Prox-Euc is L2 in the frozen preprocessor output; sparsity counts changed raw features; actionability checks that every changed field belongs to the regime's actionable families; plausibility uses novelty LOF fit once on a fixed 10k training reference sample in preprocessed space (k<=100); feasibility-style pass requires independent hard-constraint feasibility, actionability, LOF inlier status and at least two actionable changes. These values are not numerically comparable to Table 7 author values.",
        "",
        "DiCE reached the 60-second per-query timeout on its first compatibility-pilot query with only 10k reference rows, so the full matrix was not run. AR is fail-closed as UNSUPPORTED because the installed ActionSet adapter cannot preserve the same raw 81-feature constraints and frozen one-hot preprocessing. These are reported as compatibility/runtime outcomes, not failed counterfactuals.",
        "",
        "## Runtime and MI setup",
        "",
        "See `scaling_summary.csv` and `mi_scaling_summary.csv` for setup, MI and per-query times. Shared MI setup is computed once per regime for UFCE-FF2 and reused by UFCE-FF3. The separate full-reference MI pilot was stopped after more than 12 minutes without finishing its 378 estimator calls; see `full_reference_mi_limit.json`.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "primary"), required=True)
    parser.add_argument("--data-dir", default="data/upv_2025")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--pilot-reference-size", choices=("10000", "50000", "100000", "full_train"), default="10000")
    parser.add_argument("--primary-reference-size", choices=("10000", "50000", "100000", "full_train"), default="10000")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir) / args.stage
    result = _run_stage(
        args.stage, args.data_dir, out_dir, args.timeout_seconds,
        tuple(args.methods), args.pilot_reference_size, args.primary_reference_size,
    )
    print("Completed %s: %s" % (args.stage, result), flush=True)


if __name__ == "__main__":
    main()
