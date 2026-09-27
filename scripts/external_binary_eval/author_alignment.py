"""Resumable categorical author-alignment evaluation for UPV-2025."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import LocalOutlierFactor

from .adapters import EncodedModel, EncodedSpace, make_adapter, rank_mi_pairs
from .config import DESIRED_CLASS, PRIMARY_SEED, RANDOM_SEED, TIMEOUT_SECONDS
from .constraints import policy_for_query, write_constraint_policy
from .dataset import write_json
from .metrics import mcnemar_table, paired_bootstrap_difference, summarize_queries
from .run_ten_percent_eval import (
    _atomic_csv,
    _append_csv,
    _student_reference_subset,
    prepare_data,
)
from .runner import _run_one_query, _sha256, _library_versions
from .verification import verify_candidate


DEFAULT_OUT_DIR = "outputs/external_binary_eval/upv_2025_author_alignment_v1"
DEFAULT_BASELINE_DIR = "outputs/external_binary_eval/upv_2025_10pct_5fold/primary"
CATEGORY_FEATURE = "dedicacion"
REGIME = "MODERATE"
METHODS = ("UFCE-FF1", "UFCE-FF2", "UFCE-FF3")
STRATEGIES = ("DIGITAL_PAIRS_PLUS_CATEGORY", "ALL_FEATURE_TOP5")
MUTABLE_CATEGORY_DOMAIN = ("TC", "TP")


def _json_hash(value):
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path, payload):
    write_json(path, payload)


def _reference_id_manifest(reference, label, out_dir):
    ids = pd.DataFrame({
        "reference_row_index": reference.index.to_numpy(dtype=np.int64),
        "student_id": reference["student_id"].astype(str).to_numpy(),
        "source_year": reference["source_year"].to_numpy(),
    })
    path = Path(out_dir) / ("reference_ids_%s.csv" % label)
    _atomic_csv(ids, path)
    digest = hashlib.sha256(ids.to_csv(index=False).replace("\r\n", "\n").encode("utf-8")).hexdigest()
    return {
        "label": label,
        "rows": int(len(reference)),
        "students": int(reference.student_id.nunique()),
        "ordered_reference_ids_path": str(path.resolve()),
        "ordered_reference_ids_sha256": digest,
    }


def _reference_state(raw_train, reference, bundle, schema):
    space = EncodedSpace(raw_train, schema)
    encoded = space.encode(reference)
    model = EncodedModel(bundle, space)
    desired = encoded.loc[model.predict(encoded) == int(DESIRED_CLASS)].reset_index(drop=True)
    return {
        "space": space,
        "reference_encoded": encoded,
        "model": model,
        "desired_reference": desired,
    }


def _mi_artifacts(out_dir, encoded_10k, schema, numeric_actionable):
    out_dir = Path(out_dir)
    sets_path = out_dir / "mi_pair_sets.json"
    rankings_path = out_dir / "mi_pair_rankings.csv"
    reference_hash = hashlib.sha256(
        encoded_10k.to_csv(index=False).replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()
    if sets_path.exists() and rankings_path.exists():
        with open(sets_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("mi_reference_sha256") == reference_hash:
            return saved, pd.read_csv(rankings_path)

    digital_and_category = [
        name for name in schema["feature_names"]
        if name in set(numeric_actionable) or name == CATEGORY_FEATURE
    ]
    digital_category_ranking, digital_category_stats = rank_mi_pairs(
        encoded_10k, digital_and_category, seed=PRIMARY_SEED
    )
    numeric_set = set(numeric_actionable)
    digital_pairs = [
        [row["left_feature"], row["right_feature"]]
        for row in digital_category_ranking
        if row["left_feature"] in numeric_set and row["right_feature"] in numeric_set
    ][:100]
    dedication_pairs = [
        [row["left_feature"], row["right_feature"]]
        for row in digital_category_ranking
        if CATEGORY_FEATURE in (row["left_feature"], row["right_feature"])
        and (row["left_feature"] in numeric_set or row["right_feature"] in numeric_set)
    ][:5]
    if len(digital_pairs) != 100:
        raise ValueError("Expected 100 reconstructed digital numeric MI pairs, got %d" % len(digital_pairs))
    if len(dedication_pairs) != 5:
        raise ValueError("Expected 5 dedication-numeric MI pairs, got %d" % len(dedication_pairs))
    digital_plus = digital_pairs + dedication_pairs

    all_ranking, all_stats = rank_mi_pairs(
        encoded_10k, schema["feature_names"], seed=PRIMARY_SEED
    )
    mutable = numeric_set | {CATEGORY_FEATURE}
    global_top5 = all_ranking[:5]
    eligible_all_ranking = [
        row for row in all_ranking
        if row["left_feature"] in mutable and row["right_feature"] in mutable
    ]
    all_top5_selected = [
        [row["left_feature"], row["right_feature"]]
        for row in eligible_all_ranking[:5]
    ]
    if len(all_top5_selected) != 5:
        raise ValueError("Could not select 5 mutable pairs from all-feature MI ranking")

    ranking_rows = []
    for row in digital_category_ranking:
        ranking_rows.append({"mi_ranking": "DIGITAL_PLUS_CATEGORY_SOURCE", **row})
    for row in all_ranking:
        ranking_rows.append({"mi_ranking": "ALL_FEATURES_81", **row})
    _atomic_csv(pd.DataFrame(ranking_rows), rankings_path)
    pair_sets = {
        "mi_reference_rows": int(len(encoded_10k)),
        "mi_reference_sha256": reference_hash,
        "seed": int(PRIMARY_SEED),
        "categorical_encoding": "ordinal codes sorted by raw string value; dedicacion TC=0, TP=1 when both are present",
        "DIGITAL_PAIRS_PLUS_CATEGORY": {
            "pairs": digital_plus,
            "digital_numeric_pairs": digital_pairs,
            "dedicacion_numeric_pairs": dedication_pairs,
            "pair_sha256": _json_hash(digital_plus),
            "digital_pair_sha256": _json_hash(digital_pairs),
            "mi_estimator_calls": int(digital_category_stats["mi_estimator_calls"]),
            "mi_seconds": float(digital_category_stats["mi_seconds"]),
            "ranked_candidates": int(len(digital_category_ranking)),
        },
        "ALL_FEATURE_TOP5": {
            "pairs": all_top5_selected,
            "global_top5": [
                [row["left_feature"], row["right_feature"]]
                for row in global_top5
            ],
            "global_top5_with_scores": global_top5,
            "pair_sha256": _json_hash(all_top5_selected),
            "mi_estimator_calls": int(all_stats["mi_estimator_calls"]),
            "mi_seconds": float(all_stats["mi_seconds"]),
            "ranked_candidates": int(len(all_ranking)),
            "mutable_feature_filter": sorted(mutable),
        },
    }
    _write_json(sets_path, pair_sets)
    return pair_sets, pd.DataFrame(ranking_rows)


def _new_raw_policy(base_policy, raw_query, train, schema):
    policy = policy_for_query(base_policy, raw_query, train, schema)
    domain = set(policy["categorical_domains"].get(CATEGORY_FEATURE, []))
    allowed = [value for value in MUTABLE_CATEGORY_DOMAIN if value in domain]
    query_value = raw_query.iloc[0][CATEGORY_FEATURE]
    if len(allowed) == 2 and not pd.isna(query_value) and str(query_value) in allowed:
        policy["features"][CATEGORY_FEATURE] = {
            "immutable": False,
            "missing_query": False,
            "direction": "categorical_transition",
            "allowed_values": allowed,
        }
    else:
        policy["features"][CATEGORY_FEATURE] = {
            "immutable": True,
            "missing_query": bool(pd.isna(query_value)),
            "direction": "immutable",
        }
    policy["categorical_actionable_features"] = (
        [CATEGORY_FEATURE] if not policy["features"][CATEGORY_FEATURE]["immutable"] else []
    )
    return policy


def _append_rows(rows, path):
    if rows:
        _append_csv(rows, path)


def _annotate_result(query_row, candidate_rows, query, cohort, reference_label, strategy, method, mi_hash):
    factual_category = query.get(CATEGORY_FEATURE)
    changed_valid = []
    changed_feasible = []
    for row in candidate_rows:
        candidate_category = row.get("feature_" + CATEGORY_FEATURE)
        changed = (
            not pd.isna(factual_category)
            and not pd.isna(candidate_category)
            and str(factual_category) != str(candidate_category)
        )
        row["dedicacion_original"] = factual_category
        row["dedicacion_changed"] = bool(changed)
        row["dedicacion_transition"] = (
            "%s->%s" % (factual_category, candidate_category) if changed else ""
        )
        if changed and bool(row.get("target_valid")):
            changed_valid.append(candidate_category)
        if changed and bool(row.get("constraint_feasible")):
            changed_feasible.append(candidate_category)
        row["cohort"] = cohort
        row["reference_size"] = reference_label
        row["mi_strategy"] = strategy
        row["mi_pair_sha256"] = mi_hash
    query_row["cohort"] = cohort
    query_row["reference_size"] = reference_label
    query_row["mi_strategy"] = strategy
    query_row["mi_pair_sha256"] = mi_hash
    query_row["dedicacion_original"] = factual_category
    query_row["categorical_change_any_valid_cf"] = bool(changed_valid)
    query_row["categorical_change_any_feasible_cf"] = bool(changed_feasible)
    query_row["tc_to_tp_valid_candidate_count"] = int(
        sum(str(value) == "TP" for value in changed_valid) if str(factual_category) == "TC" else 0
    )
    query_row["tp_to_tc_valid_candidate_count"] = int(
        sum(str(value) == "TC" for value in changed_valid) if str(factual_category) == "TP" else 0
    )
    return query_row, candidate_rows


def _setup_row(adapter, method, stage, cohort, reference_label, reference, strategy, pair_meta):
    summary = adapter.mi_summary
    return {
        "stage": stage,
        "cohort": cohort,
        "method": method,
        "constraint_regime": REGIME,
        "seed": int(PRIMARY_SEED),
        "reference_size": reference_label,
        "reference_rows": int(len(reference)),
        "reference_students": int(reference.student_id.nunique()),
        "candidate_generation_reference_rows": int(len(reference)),
        "desired_class_search_rows": int(len(getattr(adapter, "desired_reference", []))),
        "neighbor_backend_policy": getattr(adapter, "neighbor_backend_policy", ""),
        "mi_strategy": strategy,
        "mi_pair_sha256": pair_meta.get("pair_sha256", ""),
        "mi_estimator_calls": int(summary.get("mi_estimator_calls", 0)),
        "mi_seconds": float(summary.get("mi_seconds", 0.0)),
        "retained_mi_pairs": int(summary.get("retained_pairs", 0)),
        "setup_seconds": float(adapter.setup_seconds),
        "setup_error": getattr(adapter, "setup_error", ""),
        "mi_shared_between_ff2_ff3": bool(method in ("UFCE-FF2", "UFCE-FF3")),
    }


def _run_cell(stage, cohort, reference_label, strategy, queries, train, raw_train, reference,
              bundle, schema, policies, state, out_dir, pair_meta, pairs,
              methods, timeout_seconds=TIMEOUT_SECONDS):
    cell_dir = Path(out_dir) / stage / reference_label / strategy.lower()
    cell_dir.mkdir(parents=True, exist_ok=True)
    query_path = cell_dir / "query_results.csv"
    candidate_path = cell_dir / "candidate_results.csv"
    setup_path = cell_dir / "adapter_setup.csv"
    existing = pd.read_csv(query_path) if query_path.exists() else pd.DataFrame()
    completed = set()
    if len(existing):
        completed = set(zip(existing.query_id.astype(str), existing.method.astype(str)))
    setup_existing = pd.read_csv(setup_path) if setup_path.exists() else pd.DataFrame()
    setup_done = set(setup_existing.method.astype(str)) if len(setup_existing) else set()

    for method in methods:
        adapter = make_adapter(
            method,
            raw_train,
            reference,
            bundle,
            schema,
            policies,
            REGIME,
            seed=PRIMARY_SEED,
            method_name=method,
            categorical_actionable=[CATEGORY_FEATURE],
            mi_pairs_override=(pairs if method != "UFCE-FF1" else None),
            mi_summary_override=(pair_meta if method != "UFCE-FF1" else None),
            shared_reference_state=state,
        )
        if method not in setup_done:
            _append_rows(_setup_row(
                adapter, method, stage, cohort, reference_label, reference, strategy, pair_meta
            ), setup_path)
            setup_done.add(method)
        for _, query in queries.iterrows():
            key = (str(query.query_id), method)
            if key in completed:
                continue
            raw_query = query.loc[schema["feature_names"]].to_frame().T
            raw_policy = _new_raw_policy(policies[REGIME], raw_query, train, schema)
            wall_started = time.perf_counter()
            query_row, candidate_rows = _run_one_query(
                adapter,
                raw_query,
                str(query.query_id),
                raw_policy,
                bundle,
                schema,
                method,
                REGIME,
                PRIMARY_SEED,
                timeout_seconds,
            )
            total_wall_ms = (time.perf_counter() - wall_started) * 1000.0
            generation_ms = float(query_row.get("runtime_ms", 0.0))
            reported_latency = (
                float(timeout_seconds) * 1000.0
                if query_row.get("final_status") == "TIMEOUT"
                else total_wall_ms
            )
            query_row["generation_runtime_ms"] = generation_ms
            query_row["verification_runtime_ms"] = max(0.0, total_wall_ms - generation_ms)
            query_row["runtime_ms"] = reported_latency
            query_row["fold_id"] = query.get("fold_id", np.nan)
            query_row["observed_label"] = int(query.get("y", -1))
            query_row["setup_seconds"] = float(adapter.setup_seconds)
            query_row["mi_estimator_calls"] = int(adapter.mi_summary.get("mi_estimator_calls", 0))
            query_row["retained_mi_pairs"] = int(adapter.mi_summary.get("retained_pairs", 0))
            query_row, candidate_rows = _annotate_result(
                query_row, candidate_rows, query, cohort, reference_label, strategy, method,
                pair_meta.get("pair_sha256", ""),
            )
            for row in candidate_rows:
                row["fold_id"] = query.get("fold_id", np.nan)
            # Candidates are appended before the query row, so an interrupted
            # resume may deduplicate candidates but cannot mark an unfinished
            # query as complete.
            _append_rows(candidate_rows, candidate_path)
            _append_rows(query_row, query_path)
            completed.add(key)
            if len(completed) % 10 == 0:
                print(
                    "stage=%s cohort=%s ref=%s arm=%s method=%s done=%d/%d"
                    % (stage, cohort, reference_label, strategy, method, len(completed),
                       len(queries) * len(methods)),
                    flush=True,
                )

    qdf = pd.read_csv(query_path) if query_path.exists() else pd.DataFrame()
    cdf = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()
    if len(qdf):
        qdf = qdf.drop_duplicates(["query_id", "method"], keep="last")
        _atomic_csv(qdf, query_path)
    if len(cdf):
        cdf = cdf.drop_duplicates(["query_id", "method", "candidate_id"], keep="last")
        _atomic_csv(cdf, candidate_path)
    return cell_dir


def _read_cell_files(stage_dir):
    query_frames, candidate_frames, setup_frames = [], [], []
    stage_dir = Path(stage_dir)
    for path in stage_dir.rglob("query_results.csv"):
        frame = pd.read_csv(path)
        if len(frame):
            query_frames.append(frame)
    for path in stage_dir.rglob("candidate_results.csv"):
        frame = pd.read_csv(path)
        if len(frame):
            candidate_frames.append(frame)
    for path in stage_dir.rglob("adapter_setup.csv"):
        frame = pd.read_csv(path)
        if len(frame):
            setup_frames.append(frame)
    def combine(frames, keys):
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, sort=False)
        return out.drop_duplicates(keys, keep="last")
    return (
        combine(query_frames, ["query_id", "method", "reference_size", "mi_strategy", "cohort"]),
        combine(candidate_frames, ["query_id", "method", "candidate_id", "reference_size", "mi_strategy", "cohort"]),
        combine(setup_frames, ["method", "reference_size", "mi_strategy", "cohort"]),
    )


def _paired_row(left, right, comparison, left_name, right_name):
    left = left.sort_values("query_id").reset_index(drop=True)
    right = right.sort_values("query_id").reset_index(drop=True)
    if len(left) != len(right) or not left.query_id.equals(right.query_id):
        raise ValueError("Paired query IDs differ for %s vs %s" % (left_name, right_name))
    row = {
        "comparison": comparison,
        "left": left_name,
        "right": right_name,
        "n_queries": int(len(left)),
        "paired_query_ids_sha256": _json_hash(left.query_id.astype(str).tolist()),
    }
    for name, column in (
        ("valid_availability", "returned_any_valid_cf"),
        ("feasible_availability", "returned_any_feasible_cf"),
    ):
        observed, interval = paired_bootstrap_difference(
            left[column].astype(float).to_numpy(), right[column].astype(float).to_numpy()
        )
        row["delta_%s_percentage_points" % name] = 100.0 * observed if observed is not None else None
        row["delta_%s_ci_low_percentage_points" % name] = 100.0 * interval[0] if interval else None
        row["delta_%s_ci_high_percentage_points" % name] = 100.0 * interval[1] if interval else None
        if name == "feasible_availability":
            row["mcnemar"] = mcnemar_table(left[column], right[column])
    latency, latency_ci = paired_bootstrap_difference(
        left["runtime_ms"].to_numpy(), right["runtime_ms"].to_numpy(), statistic="median"
    )
    row["median_latency_difference_ms"] = latency
    row["median_latency_ci_low_ms"] = latency_ci[0] if latency_ci else None
    row["median_latency_ci_high_ms"] = latency_ci[1] if latency_ci else None
    return row


def _summaries(query_results, candidate_results):
    rows = []
    if query_results.empty:
        return pd.DataFrame()
    groups = ["cohort", "reference_size", "mi_strategy", "method"]
    for key, group in query_results.groupby(groups, sort=False):
        cohort, ref_size, strategy, method = key
        candidate_group = candidate_results
        if len(candidate_group):
            for column, value in zip(groups, key):
                candidate_group = candidate_group[candidate_group[column] == value]
        else:
            candidate_group = pd.DataFrame()
        row = summarize_queries(group, candidate_group, method, REGIME, PRIMARY_SEED)
        row.update({
            "cohort": cohort,
            "reference_size": ref_size,
            "mi_strategy": strategy,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _transition_summary(query_results, candidate_results):
    if query_results.empty:
        return pd.DataFrame()
    records = []
    groups = ["cohort", "reference_size", "mi_strategy", "method"]
    for key, group in query_results.groupby(groups, sort=False):
        cohort, ref_size, strategy, method = key
        candidates = candidate_results
        if len(candidates):
            for column, value in zip(groups, key):
                candidates = candidates[candidates[column] == value]
        if len(candidates) and "dedicacion_changed" in candidates:
            changed = candidates[candidates["dedicacion_changed"].fillna(False)]
        else:
            changed = pd.DataFrame(columns=candidate_results.columns)
        valid_changed = changed[changed.target_valid.fillna(False)] if len(changed) else changed
        feasible_changed = changed[changed.constraint_feasible.fillna(False)] if len(changed) else changed
        records.append({
            "cohort": cohort,
            "reference_size": ref_size,
            "mi_strategy": strategy,
            "method": method,
            "eligible_queries": int(len(group)),
            "queries_with_valid_category_change": int(valid_changed.query_id.nunique()) if len(valid_changed) else 0,
            "valid_category_change_availability": float(valid_changed.query_id.nunique() / len(group)) if len(group) else None,
            "queries_with_feasible_category_change": int(feasible_changed.query_id.nunique()) if len(feasible_changed) else 0,
            "feasible_category_change_availability": float(feasible_changed.query_id.nunique() / len(group)) if len(group) else None,
            "tc_to_tp_valid_candidates": int((valid_changed.dedicacion_transition == "TC->TP").sum()) if len(valid_changed) else 0,
            "tp_to_tc_valid_candidates": int((valid_changed.dedicacion_transition == "TP->TC").sum()) if len(valid_changed) else 0,
            "tc_to_tp_feasible_candidates": int((feasible_changed.dedicacion_transition == "TC->TP").sum()) if len(feasible_changed) else 0,
            "tp_to_tc_feasible_candidates": int((feasible_changed.dedicacion_transition == "TP->TC").sum()) if len(feasible_changed) else 0,
        })
    return pd.DataFrame(records)


def _paired_comparisons(query_results, baseline_dir):
    rows = []
    if query_results.empty:
        return pd.DataFrame()
    baseline_path = Path(baseline_dir) / "query_results.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[
        (baseline.constraint_regime == REGIME)
        & (baseline.seed == PRIMARY_SEED)
    ]
    primary = query_results[query_results.cohort == "paired_207"]
    for method in METHODS:
        old = baseline[baseline.method == method]
        for ref_size, strategy in (
            ("10k", "SHARED_FF1"),
            ("10k", "DIGITAL_PAIRS_PLUS_CATEGORY"),
            ("10k", "ALL_FEATURE_TOP5"),
        ):
            new = primary[(primary.method == method) & (primary.reference_size == ref_size) & (primary.mi_strategy == strategy)]
            if len(old) == 207 and len(new) == 207:
                rows.append(_paired_row(
                    new, old, "categorical_10k_vs_saved_DIGITAL_ONLY",
                    "%s/%s/%s" % (method, ref_size, strategy),
                    "%s/DIGITAL_ONLY/10k" % method,
                ))
        for strategy in (("SHARED_FF1",) if method == "UFCE-FF1" else STRATEGIES):
            small = primary[
                (primary.method == method)
                & (primary.reference_size == "10k")
                & (primary.mi_strategy == strategy)
            ]
            full = primary[
                (primary.method == method)
                & (primary.reference_size == "full_train")
                & (primary.mi_strategy == strategy)
            ]
            if len(small) == 207 and len(full) == 207:
                rows.append(_paired_row(
                    full, small, "full_reference_vs_10k_same_MI_pairs",
                    "%s/full_train/%s" % (method, strategy),
                    "%s/10k/%s" % (method, strategy),
                ))
    return pd.DataFrame(rows)


def _table7_descriptive(query_results, candidate_results, train, reference_10k, bundle, schema, policies, query_instances=None):
    if query_results.empty or candidate_results.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw_features = list(schema["feature_names"])
    categorical = list(schema["categorical_features"])
    mutable = set(policies[REGIME]["actionable_features"]) | {CATEGORY_FEATURE}
    ref_x = bundle.transform_raw(reference_10k)
    lof_neighbors = min(100, max(1, len(reference_10k) - 1))
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
    setup_started = time.perf_counter()
    lof.fit(ref_x)
    lof_setup_seconds = time.perf_counter() - setup_started
    lookup_source = query_instances if query_instances is not None and len(query_instances) else query_results
    query_lookup = lookup_source.drop_duplicates(["query_id", "cohort"]).set_index(["cohort", "query_id"])
    base = candidate_results[candidate_results.target_valid.fillna(False)].copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_raw = pd.DataFrame(
        [{name: row.get("feature_" + name, np.nan) for name in raw_features} for _, row in base.iterrows()],
        columns=raw_features,
    )
    candidate_x = bundle.transform_raw(candidate_raw)
    plausibility = lof.predict(candidate_x) == 1
    candidate_rows = []
    for index, (_, item) in enumerate(base.iterrows()):
        query_key = (item.cohort, item.query_id)
        if query_key not in query_lookup.index:
            continue
        query_row = query_lookup.loc[query_key]
        query_raw = pd.DataFrame([{name: query_row[name] for name in raw_features}], columns=raw_features)
        cf_raw = candidate_raw.iloc[[index]].copy()
        changed = []
        for feature in raw_features:
            left, right = query_raw.iloc[0][feature], cf_raw.iloc[0][feature]
            if pd.isna(left) and pd.isna(right):
                continue
            if pd.isna(left) or pd.isna(right):
                changed.append(feature)
            elif feature in categorical:
                if str(left) != str(right):
                    changed.append(feature)
            else:
                try:
                    if not np.isclose(float(left), float(right), atol=1e-5, rtol=0):
                        changed.append(feature)
                except Exception:
                    if str(left) != str(right):
                        changed.append(feature)
        changed_categories = [feature for feature in changed if feature in categorical]
        prox_jac = (
            2.0 * len(changed_categories) / (len(categorical) + len(changed_categories))
            if categorical else 0.0
        )
        query_x = bundle.transform_raw(query_raw)
        delta = query_x - candidate_x[index]
        prox_euc = float(np.sqrt(delta.multiply(delta).sum())) if sparse.issparse(delta) else float(np.linalg.norm(delta))
        actionability_pass = all(feature in mutable for feature in changed)
        plausible = bool(plausibility[index])
        changed_actionable = sum(feature in mutable for feature in changed)
        feasible_style = bool(item.constraint_feasible and actionability_pass and plausible and changed_actionable >= 2)
        candidate_rows.append({
            "cohort": item.cohort,
            "reference_size": item.reference_size,
            "mi_strategy": item.mi_strategy,
            "method": item.method,
            "query_id": item.query_id,
            "candidate_id": int(item.candidate_id),
            "prox_jac_nominal": float(prox_jac),
            "prox_euc_preprocessed_l2": prox_euc,
            "sparsity_raw_changed_features": int(len(changed)),
            "actionability_pass": bool(actionability_pass),
            "plausibility_lof_10k_reference": plausible,
            "feasibility_style_pass": feasible_style,
            "hard_constraint_feasible": bool(item.constraint_feasible),
            "n_changed_actionable_features": int(changed_actionable),
            "lof_reference_rows": int(len(reference_10k)),
            "lof_n_neighbors": int(lof_neighbors),
            "lof_setup_seconds": float(lof_setup_seconds),
        })
    detail = pd.DataFrame(candidate_rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(["cohort", "reference_size", "mi_strategy", "method"], as_index=False).agg(
        valid_returned_candidates=("candidate_id", "count"),
        prox_jac_mean=("prox_jac_nominal", "mean"),
        prox_euc_mean=("prox_euc_preprocessed_l2", "mean"),
        sparsity_mean=("sparsity_raw_changed_features", "mean"),
        actionability_pass_rate=("actionability_pass", "mean"),
        plausibility_pass_rate=("plausibility_lof_10k_reference", "mean"),
        feasibility_pass_rate=("feasibility_style_pass", "mean"),
    )
    summary["metric_contract"] = "adapted descriptive definitions; not a Table 7 magnitude benchmark"
    return detail, summary


def _raw_query_instances(out_dir):
    frames = []
    for filename, cohort in (("query_ids.csv", "paired_207"), ("tp_diagnostic_query_ids.csv", "supplemental_tp")):
        path = Path(out_dir) / filename
        if path.exists():
            frame = pd.read_csv(path)
            frame["cohort"] = cohort
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()



def _collect_and_finalize(out_dir, baseline_dir, query_frames, candidate_frames, setup_frames,
                          train, ref_10k, bundle, schema, policies):
    all_queries = [frame for frame in query_frames if len(frame)]
    all_candidates = [frame for frame in candidate_frames if len(frame)]
    all_setup = [frame for frame in setup_frames if len(frame)]
    qdf = pd.concat(all_queries, ignore_index=True, sort=False) if all_queries else pd.DataFrame()
    cdf = pd.concat(all_candidates, ignore_index=True, sort=False) if all_candidates else pd.DataFrame()
    sdf = pd.concat(all_setup, ignore_index=True, sort=False) if all_setup else pd.DataFrame()
    if len(qdf):
        qdf = qdf.drop_duplicates(["query_id", "method", "reference_size", "mi_strategy", "cohort"], keep="last")
    if len(cdf):
        cdf = cdf.drop_duplicates(["query_id", "method", "candidate_id", "reference_size", "mi_strategy", "cohort"], keep="last")
    _atomic_csv(qdf, Path(out_dir) / "query_results.csv")
    _atomic_csv(cdf, Path(out_dir) / "candidate_results.csv")
    _atomic_csv(sdf.drop_duplicates(["method", "reference_size", "mi_strategy", "cohort"], keep="last") if len(sdf) else sdf, Path(out_dir) / "adapter_setup.csv")
    summaries = _summaries(qdf, cdf)
    _atomic_csv(summaries, Path(out_dir) / "method_summary.csv")
    transitions = _transition_summary(qdf, cdf)
    _atomic_csv(transitions, Path(out_dir) / "categorical_transition_summary.csv")
    pairs = _paired_comparisons(qdf, baseline_dir)
    _atomic_csv(pairs, Path(out_dir) / "pairwise_comparison.csv")
    if len(qdf):
        _atomic_csv(qdf[qdf.final_status != "SUCCESS_VALID_FEASIBLE"], Path(out_dir) / "failures.csv")
    violations = cdf[cdf.violations.fillna("").astype(str).str.len() > 0] if len(cdf) else cdf
    _atomic_csv(violations, Path(out_dir) / "violations.csv")
    raw_queries = _raw_query_instances(out_dir)
    detail, table7 = _table7_descriptive(qdf, cdf, train, ref_10k, bundle, schema, policies, raw_queries)
    _atomic_csv(detail, Path(out_dir) / "table7_descriptive_candidates.csv")
    _atomic_csv(table7, Path(out_dir) / "table7_descriptive_summary.csv")
    scaling = []
    if len(sdf):
        for key, group in qdf.groupby(["cohort", "reference_size", "mi_strategy", "method"], sort=False):
            cohort, ref_size, strategy, method = key
            setups = sdf[
                (sdf.cohort == cohort) & (sdf.reference_size == ref_size)
                & (sdf.mi_strategy == strategy) & (sdf.method == method)
            ]
            summary = summaries[
                (summaries.cohort == cohort) & (summaries.reference_size == ref_size)
                & (summaries.mi_strategy == strategy) & (summaries.method == method)
            ]
            setup_seconds = float(setups.setup_seconds.max()) if len(setups) else 0.0
            mi_seconds = float(setups.mi_seconds.max()) if len(setups) else 0.0
            scaling.append({
                "cohort": cohort,
                "reference_size": ref_size,
                "mi_strategy": strategy,
                "method": method,
                "reference_rows": int(setups.reference_rows.max()) if len(setups) else None,
                "method_setup_seconds": setup_seconds,
                "mi_seconds_shared": mi_seconds,
                "median_query_latency_ms": float(group.runtime_ms.median()),
                "p95_query_latency_ms": float(group.runtime_ms.quantile(0.95)),
                "valid_availability": float(group.returned_any_valid_cf.mean()),
                "feasible_availability": float(group.returned_any_feasible_cf.mean()),
                "candidate_count_median": float(group.candidate_count.median()),
                "eccf_ms": (
                    float(summary.iloc[0].effective_compute_cost_per_feasible_query_ms)
                    if len(summary) and pd.notna(summary.iloc[0].effective_compute_cost_per_feasible_query_ms)
                    else None
                ),
                "peak_rss_mb": None,
            })
    _atomic_csv(pd.DataFrame(scaling), Path(out_dir) / "scaling_summary.csv")

    mi_sets_path = Path(out_dir) / "mi_pair_sets.json"
    if mi_sets_path.exists():
        with open(mi_sets_path, "r", encoding="utf-8") as handle:
            pair_sets = json.load(handle)
        mi_rows = []
        for strategy in STRATEGIES:
            data = pair_sets[strategy]
            mi_rows.append({
                "mi_strategy": strategy,
                "mi_reference_rows": pair_sets["mi_reference_rows"],
                "reference_sha256": pair_sets["mi_reference_sha256"],
                "mi_estimator_calls": data["mi_estimator_calls"],
                "mi_seconds": data["mi_seconds"],
                "retained_pairs": len(data["pairs"]),
                "pair_sha256": data["pair_sha256"],
            })
        _atomic_csv(pd.DataFrame(mi_rows), Path(out_dir) / "mi_scaling_summary.csv")

    lines = [
        "# UPV-2025 UFCE-FF categorical author-alignment evaluation",
        "",
        "The run uses the accepted frozen LogisticRegression snapshot and the saved 207-query, five-fold test cohort. The model and split are reused. The 10,007-row train reference is compared with the full 325,483-row train reference; MI pair rankings for both arms are computed only from the 10,007-row train sample.",
        "",
        "Only the raw categorical predictor dedicacion may switch between TC and TP when present. The other five categorical predictors remain immutable. The category switch is an algorithmic capability test, not a validated educational intervention.",
        "",
        "## Primary query results",
        "",
        "| Reference | MI arm | Method | Valid availability | Constraint-feasible availability | Exposed validity | Constraint compliance | Median ms | P95 ms | ECCF ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    headline = summaries[summaries.cohort == "paired_207"] if len(summaries) else summaries
    for _, row in headline.iterrows():
        lines.append(
            "| {reference_size} | {mi_strategy} | {method} | {valid_cf_availability_rate} ({valid_cf_availability_numerator}/{valid_cf_availability_denominator}) | {feasible_cf_availability_rate} ({feasible_cf_availability_numerator}/{feasible_cf_availability_denominator}) | {exposed_cf_validity_rate} ({exposed_cf_validity_numerator}/{exposed_cf_validity_denominator}) | {constraint_satisfaction_rate} ({constraint_satisfaction_numerator}/{constraint_satisfaction_denominator}) | {median_latency_ms} | {p95_latency_ms} | {effective_compute_cost_per_feasible_query_ms} |".format(
                **{name: ("NA" if pd.isna(value) else ("%.3f" % value if isinstance(value, (float, np.floating)) else value)) for name, value in row.items()}
            )
        )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "See pairwise_comparison.csv for paired percentage-point differences, 2,000-resample bootstrap intervals with seed 42, McNemar results, and paired median latency differences. Categorical 10k versus saved DIGITAL_ONLY estimates the policy-arm difference. Full versus 10k within an MI arm estimates the reference-size difference.",
        "",
        "## Categorical transitions",
        "",
        "See categorical_transition_summary.csv. Valid/feasible categorical evidence requires an actual TC-to-TP or TP-to-TC change that passes raw-domain, constraint and frozen-target checks. The supplemental TP cohort is shown separately from the paired 207-query estimates.",
        "",
        "## Compatibility",
        "",
        "DiCE and AR statuses are preserved in adapter_compatibility.csv. They are not included in the new branch when the existing compatibility pilot recorded a timeout or an unsupported raw-constraint adapter.",
        "",
        "## Six Table 7 metrics",
        "",
        "table7_descriptive_summary.csv contains adapted descriptive Prox-Jac, Prox-Euc, sparsity, actionability, LOF plausibility and feasibility-style values. They characterize this dataset and are not magnitude benchmarks against the published Table 7 values.",
        "",
        "Setup, MI, query and verification wall time are separated in adapter_setup.csv and query_results.csv. Full-reference rows are never capped: a conservative feature-box distance bound returns all radius neighbors exactly when it proves the configured radius contains the entire desired-class pool; otherwise the KD-tree path is used. Repeatable auxiliary model fits are cached.",
    ])
    (Path(out_dir) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compatibility_status(out_dir, baseline_dir):
    rows = []
    prior_paths = {
        "DiCE": Path("outputs/external_binary_eval/upv_2025_10pct_5fold/pilot/query_results.csv"),
        "AR": Path(baseline_dir) / "query_results.csv",
    }
    for method in ("DiCE", "AR"):
        path = prior_paths[method]
        statuses = []
        if path.exists():
            prior = pd.read_csv(path)
            if "method" in prior:
                statuses = prior.loc[prior.method == method, "final_status"].astype(str).tolist()
        if method == "DiCE" and "TIMEOUT" in statuses:
            status = "PRIOR_PILOT_TIMEOUT"
            reason = "Existing 10k compatibility pilot timed out at 60 seconds/query."
        elif method == "DiCE" and "RUNTIME_ERROR" in statuses:
            status = "PRIOR_PILOT_RUNTIME_ERROR"
            reason = "Existing 10k compatibility pilot returned runtime errors for the sampled queries."
        elif method == "AR" and "UNSUPPORTED" in statuses:
            status = "UNSUPPORTED_RAW_CONSTRAINT_ADAPTER"
            reason = "Existing adapter cannot preserve raw 81-feature constraints with the frozen one-hot model."
        else:
            status = "NO_PRIOR_COMPATIBILITY_EVIDENCE"
            reason = "No reusable compatibility status was found."
        rows.append({
            "method": method,
            "new_branch_status": "SKIPPED_NOT_COMPATIBLE",
            "prior_status": status,
            "prior_artifact": str(path.resolve()),
            "prior_artifact_sha256": _sha256(path) if path.exists() else None,
            "reason": reason,
            "primary_run_included": False,
        })
    _atomic_csv(pd.DataFrame(rows), Path(out_dir) / "adapter_compatibility.csv")


def prepare_experiment(data_dir, out_dir, baseline_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, train, test, prepared_queries, schema, dataset_summary, bundle, policies, checkpoint_path = prepare_data(
        data_dir, out_dir
    )
    baseline_query_path = Path(baseline_dir) / "query_ids.csv"
    if not baseline_query_path.exists():
        raise FileNotFoundError("Saved DIGITAL_ONLY query cohort is missing: %s" % baseline_query_path)
    baseline_queries = pd.read_csv(baseline_query_path)
    generated_ids = prepared_queries[["query_id", "student_id"]].astype(str).sort_values("query_id").reset_index(drop=True)
    baseline_ids = baseline_queries[["query_id", "student_id"]].astype(str).sort_values("query_id").reset_index(drop=True)
    if not generated_ids.equals(baseline_ids):
        raise ValueError("Recomputed query cohort differs from the saved 207 query IDs")
    if len(prepared_queries) != 207:
        raise ValueError("Expected 207 paired queries, got %d" % len(prepared_queries))
    if prepared_queries.student_id.nunique() != 207:
        raise ValueError("The paired cohort must contain 207 unique students")

    raw_train = train.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy()
    ref_10k = _student_reference_subset(train, 10000, seed=RANDOM_SEED)
    full_reference = train.copy()
    refs = {
        "10k": ref_10k.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy(),
        "full_train": full_reference.loc[:, ["student_id", "y", "source_year"] + schema["feature_names"]].copy(),
    }
    reference_manifests = [
        _reference_id_manifest(refs[label], label, out_dir)
        for label in ("10k", "full_train")
    ]
    if len(refs["full_train"]) != 325483:
        raise ValueError("Expected 325,483 full training rows, got %d" % len(refs["full_train"]))

    for feature in schema["categorical_features"]:
        train_domain = sorted(train[feature].dropna().astype(str).unique().tolist())
        if feature == CATEGORY_FEATURE and not set(MUTABLE_CATEGORY_DOMAIN).issubset(set(train_domain)):
            raise ValueError("dedicacion must contain both TC and TP in training data: %s" % train_domain)

    supplementary = test.loc[test.original_prediction == 0].copy()
    supplementary = supplementary.sort_values(["student_id", "source_year"]).drop_duplicates("student_id", keep="first")
    main_student_ids = set(prepared_queries.student_id.astype(str))
    supplementary = supplementary[
        (supplementary[CATEGORY_FEATURE].astype("string") == "TP")
        & (~supplementary.student_id.astype(str).isin(main_student_ids))
    ].copy()
    if len(supplementary) > 100:
        supplementary = supplementary.sample(n=100, random_state=RANDOM_SEED)
    supplementary = supplementary.sort_values("student_id").reset_index(drop=True)
    supplementary["query_id"] = ["upv_tp_q%05d" % index for index in range(len(supplementary))]
    supplementary["fold_id"] = np.nan
    supplementary_ids_path = out_dir / "tp_diagnostic_query_ids.csv"
    _atomic_csv(
        supplementary[["query_id", "student_id", "source_year", "y", "original_prediction"] + schema["feature_names"]],
        supplementary_ids_path,
    )

    base_moderate_policy = policies[REGIME]
    _write_json(Path(out_dir) / "constraint_policy.json", {
        "numeric_regimes": policies,
        "categorical_extension": {
            "actionable_feature": CATEGORY_FEATURE,
            "allowed_values": list(MUTABLE_CATEGORY_DOMAIN),
            "transitions": ["TC->TP", "TP->TC"],
            "other_categorical_features": "immutable",
            "missing_query_value": "immutable",
            "semantic_status": "algorithmic capability test only; not a validated educational intervention",
        },
    })

    checkpoint_path = Path(checkpoint_path)
    baseline_files = {}
    for name in (
        "query_ids.csv", "query_results.csv", "candidate_results.csv",
        "method_summary.csv", "adapter_setup.csv", "run_manifest.json",
    ):
        path = Path(baseline_dir) / name
        baseline_files[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path) if path.exists() else None,
        }
    all_names = list(schema["feature_names"])
    numeric_actionable = list(base_moderate_policy["actionable_features"])
    state_10k = _reference_state(raw_train, refs["10k"], bundle, schema)
    mi_sets, rankings = _mi_artifacts(out_dir, state_10k["reference_encoded"], schema, numeric_actionable)
    _atomic_csv(rankings, Path(out_dir) / "mi_pair_rankings.csv")
    for strategy in STRATEGIES:
        branch_dir = Path(out_dir) / "mi_strategies" / strategy.lower()
        branch_dir.mkdir(parents=True, exist_ok=True)
        _write_json(branch_dir / "mi_pairs.json", mi_sets[strategy])

    # Exact query order and IDs are frozen from the saved primary artifact.
    _atomic_csv(prepared_queries[["query_id", "student_id", "source_year", "fold_id", "y", "original_prediction"] + schema["feature_names"]], Path(out_dir) / "query_ids.csv")
    _write_json(Path(out_dir) / "experiment_config.json", {
        "experiment": "UPV-2025 UFCE-FF Author-Alignment Experiment",
        "dataset": "UPV-2025",
        "query_contract": "saved 207 unique-student queries, model prediction 0, desired class 1",
        "model_checkpoint": str(checkpoint_path.resolve()),
        "model_sha256": _sha256(checkpoint_path),
        "split": "saved deterministic student-group 70/15/15 split",
        "regime": REGIME,
        "source_class": 0,
        "desired_class": DESIRED_CLASS,
        "primary_seed": PRIMARY_SEED,
        "max_candidates": 5,
        "timeout_seconds": TIMEOUT_SECONDS,
        "reference_sizes": {"10k": int(len(refs["10k"])), "full_train": int(len(refs["full_train"]))},
        "mi_rows": int(len(refs["10k"])),
        "mi_strategies": list(STRATEGIES),
        "methods": list(METHODS),
        "categorical_policy": {
            "actionable": CATEGORY_FEATURE,
            "allowed_values": list(MUTABLE_CATEGORY_DOMAIN),
            "other_categorical_features": "immutable",
        },
        "paired_query_count": int(len(prepared_queries)),
        "supplementary_tp_query_count": int(len(supplementary)),
        "supplementary_tp_seed": RANDOM_SEED,
        "paired_bootstrap": {"resamples": 2000, "seed": 42},
        "baseline_artifacts": baseline_files,
        "dataset_checksums": dataset_summary.get("source_manifest", []),
        "reference_id_manifests": reference_manifests,
        "mi_pair_hashes": {
            strategy: mi_sets[strategy]["pair_sha256"] for strategy in STRATEGIES
        },
    })
    _compatibility_status(out_dir, baseline_dir)
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_code_hashes": {
            str(path): _sha256(path)
            for path in (
                Path("scripts/external_binary_eval/adapters.py"),
                Path("scripts/external_binary_eval/author_alignment.py"),
                Path("scripts/external_binary_eval/verification.py"),
                Path("ufce/ufce_ff/ufce.py"),
            )
            if path.exists()
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": _library_versions(),
        "dataset_checksums": dataset_summary.get("source_manifest", []),
        "model_path": str(checkpoint_path.resolve()),
        "model_sha256": _sha256(checkpoint_path),
        "raw_feature_count": int(len(all_names)),
        "numeric_feature_count": int(len(schema["numeric_features"])),
        "categorical_feature_count": int(len(schema["categorical_features"])),
        "encoded_feature_count": int(bundle.transform_raw(train.iloc[:1]).shape[1]),
        "query_ids_sha256": _sha256(Path(out_dir) / "query_ids.csv"),
        "baseline_artifacts": baseline_files,
        "reference_id_manifests": reference_manifests,
        "mi_pair_hashes": {
            strategy: mi_sets[strategy]["pair_sha256"] for strategy in STRATEGIES
        },
        "mi_estimator_calls": {
            strategy: mi_sets[strategy]["mi_estimator_calls"] for strategy in STRATEGIES
        },
        "timeout_seconds": TIMEOUT_SECONDS,
        "primary_seed": PRIMARY_SEED,
        "status": "PREPARED",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(Path(out_dir) / "run_manifest.json", manifest)
    return {
        "frame": frame,
        "train": train,
        "test": test,
        "queries": prepared_queries,
        "supplementary": supplementary,
        "schema": schema,
        "dataset_summary": dataset_summary,
        "bundle": bundle,
        "policies": policies,
        "raw_train": raw_train,
        "references": refs,
        "state_10k": state_10k,
        "mi_sets": mi_sets,
        "baseline_files": baseline_files,
        "reference_manifests": reference_manifests,
        "checkpoint_path": checkpoint_path,
    }


def _run_stage(stage, prepared, out_dir, timeout_seconds):
    train = prepared["train"]
    raw_train = prepared["raw_train"]
    schema = prepared["schema"]
    bundle = prepared["bundle"]
    policies = prepared["policies"]
    references = prepared["references"]
    mi_sets = prepared["mi_sets"]
    if stage == "pilot":
        main_queries = prepared["queries"].head(20).copy()
        tp_queries = pd.DataFrame()
        cohort_name = "compatibility_pilot_20"
    elif stage == "primary":
        main_queries = prepared["queries"].copy()
        tp_queries = pd.DataFrame()
        cohort_name = "paired_207"
    elif stage == "supplemental":
        main_queries = pd.DataFrame()
        tp_queries = prepared["supplementary"].copy()
        cohort_name = "supplemental_tp"
    else:
        raise ValueError("Unknown stage %s" % stage)

    for reference_label in ("10k", "full_train"):
        reference = references[reference_label]
        if not main_queries.empty:
            if reference_label == "10k":
                state = prepared["state_10k"]
            else:
                state = _reference_state(raw_train, reference, bundle, schema)
            common_dir = Path(out_dir) / stage / reference_label / "shared_ff1"
            common_meta = {
                "pair_sha256": "no_mi_ff1",
                "mi_estimator_calls": 0,
                "mi_seconds": 0.0,
                "retained_pairs": 0,
            }
            _run_cell(
                stage, cohort_name, reference_label, "SHARED_FF1", main_queries,
                train, raw_train, reference, bundle, schema, policies, state,
                out_dir, common_meta, [], ("UFCE-FF1",), timeout_seconds,
            )
            for strategy in STRATEGIES:
                pair_meta = mi_sets[strategy]
                _run_cell(
                    stage, cohort_name, reference_label, strategy, main_queries,
                    train, raw_train, reference, bundle, schema, policies, state,
                    out_dir, pair_meta, pair_meta["pairs"], ("UFCE-FF2", "UFCE-FF3"),
                    timeout_seconds,
                )
        if not tp_queries.empty:
            if reference_label == "10k":
                state = prepared["state_10k"]
            else:
                state = _reference_state(raw_train, reference, bundle, schema)
            common_meta = {
                "pair_sha256": "no_mi_ff1",
                "mi_estimator_calls": 0,
                "mi_seconds": 0.0,
                "retained_pairs": 0,
            }
            _run_cell(
                "supplemental_tp", "supplemental_tp", reference_label, "SHARED_FF1",
                tp_queries, train, raw_train, reference, bundle, schema, policies,
                state, out_dir, common_meta, [], ("UFCE-FF1",), timeout_seconds,
            )
            for strategy in STRATEGIES:
                pair_meta = mi_sets[strategy]
                _run_cell(
                    "supplemental_tp", "supplemental_tp", reference_label, strategy,
                    tp_queries, train, raw_train, reference, bundle, schema, policies,
                    state, out_dir, pair_meta, pair_meta["pairs"], ("UFCE-FF2", "UFCE-FF3"),
                    timeout_seconds,
                )

    query_frames, candidate_frames, setup_frames = [], [], []
    for stage_dir in (Path(out_dir) / "primary", Path(out_dir) / "supplemental_tp"):
        if stage_dir.exists():
            qdf, cdf, sdf = _read_cell_files(stage_dir)
            query_frames.append(qdf)
            candidate_frames.append(cdf)
            setup_frames.append(sdf)
    # Pilot summaries are isolated and never included in the primary estimates.
    pilot_dir = Path(out_dir) / "pilot"
    if pilot_dir.exists():
        pilot_q, pilot_c, pilot_s = _read_cell_files(pilot_dir)
        _atomic_csv(_summaries(pilot_q, pilot_c), pilot_dir / "method_summary.csv")
        _atomic_csv(_transition_summary(pilot_q, pilot_c), pilot_dir / "categorical_transition_summary.csv")
        _atomic_csv(pilot_s, pilot_dir / "adapter_setup_summary.csv")
    _collect_and_finalize(
        out_dir, DEFAULT_BASELINE_DIR, query_frames, candidate_frames, setup_frames,
        train, references["10k"], bundle, schema, policies,
    )
    manifest_path = Path(out_dir) / "run_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["status"] = {
        "pilot": "PILOT_COMPLETE",
        "primary": "PRIMARY_COMPLETE",
        "supplemental": "SUPPLEMENTAL_COMPLETE",
    }[stage]
    manifest["last_stage"] = stage
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_query_rows"] = int(sum(len(frame) for frame in query_frames))
    _write_json(manifest_path, manifest)
    print("Completed %s stage; outputs in %s" % (stage, out_dir), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "primary", "supplemental"), required=True)
    parser.add_argument("--data-dir", default="data/upv_2025")
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    prepared = prepare_experiment(args.data_dir, args.out_dir, args.baseline_dir)
    _run_stage(args.stage, prepared, args.out_dir, args.timeout_seconds)


if __name__ == "__main__":
    main()
