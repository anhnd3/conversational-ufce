"""Transition-level Wilson intervals and paired bootstrap comparisons."""

import math

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, METHODS, TRANSITIONS


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return {"estimate": None, "numerator": int(successes), "denominator": 0, "ci_low": None, "ci_high": None}
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "estimate": p,
        "numerator": int(successes),
        "denominator": int(total),
        "ci_low": max(0.0, centre - margin),
        "ci_high": min(1.0, centre + margin),
    }


def _rate_fields(prefix, value):
    return {
        prefix: value["estimate"],
        prefix + "_numerator": value["numerator"],
        prefix + "_denominator": value["denominator"],
        prefix + "_ci95_low": value["ci_low"],
        prefix + "_ci95_high": value["ci_high"],
    }


def summarize_transition(query_results, candidate_summaries, exposed_results, method, transition):
    queries = query_results[
        (query_results["method"] == method)
        & (query_results["transition"] == transition)
    ].copy()
    candidates = candidate_summaries[
        (candidate_summaries["method"] == method)
        & (candidate_summaries["transition"] == transition)
    ].copy()
    exposed = exposed_results[
        (exposed_results["method"] == method)
        & (exposed_results["transition"] == transition)
    ].copy()
    n_queries = len(queries)
    raw_valid_queries = int(queries["target_valid_availability"].fillna(False).sum()) if n_queries else 0
    feasible_queries = int(queries["constraint_feasible_availability"].fillna(False).sum()) if n_queries else 0
    raw_candidates = int(candidates["candidate_count"].sum()) if len(candidates) else 0
    off_target_candidates = int(candidates["off_target_count"].sum()) if len(candidates) else 0
    target_candidates = int(candidates["target_count"].sum()) if len(candidates) else 0
    exposed_count = int(len(exposed))
    exposed_valid = int(exposed["target_valid"].fillna(False).sum()) if exposed_count else 0
    exposed_feasible = int(exposed["constraint_feasible"].fillna(False).sum()) if exposed_count else 0
    runtime = pd.to_numeric(queries.get("runtime_ms", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    setup_ms = float(queries["setup_ms"].dropna().iloc[0]) if n_queries and queries["setup_ms"].notna().any() else 0.0
    valid_availability = wilson(raw_valid_queries, n_queries)
    feasible_availability = wilson(feasible_queries, n_queries)
    off_target_rate = wilson(off_target_candidates, raw_candidates)
    exposed_validity = wilson(exposed_valid, exposed_count)
    constraint_satisfaction = wilson(exposed_feasible, exposed_count)
    feasible_cf_count = exposed_feasible
    effective_cost = (
        (float(runtime.sum()) + setup_ms) / feasible_cf_count
        if feasible_cf_count else None
    )
    summary = {
        "method": method,
        "transition": transition,
        "query_count": int(n_queries),
        "raw_candidate_count": raw_candidates,
        "off_target_candidate_count": off_target_candidates,
        "target_candidate_count": target_candidates,
        "exposed_cf_count": exposed_count,
        "feasible_exposed_cf_count": feasible_cf_count,
        "median_latency_ms": float(runtime.median()) if n_queries else None,
        "p95_latency_ms": float(runtime.quantile(0.95)) if n_queries else None,
        "setup_ms": setup_ms,
        "effective_compute_cost_ms_per_feasible_cf": effective_cost,
    }
    summary.update(_rate_fields("target_valid_availability", valid_availability))
    summary.update(_rate_fields("constraint_feasible_availability", feasible_availability))
    summary.update(_rate_fields("off_target_rate", off_target_rate))
    summary.update(_rate_fields("exposed_cf_target_validity", exposed_validity))
    summary.update(_rate_fields("exposed_cf_constraint_satisfaction", constraint_satisfaction))
    return summary


def paired_bootstrap(query_results, transition, left_method, right_method, metric):
    subset = query_results[query_results["transition"] == transition]
    left = subset[subset["method"] == left_method].set_index("query_id")[metric].sort_index()
    right = subset[subset["method"] == right_method].set_index("query_id")[metric].sort_index()
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
    if joined.empty or len(joined) != len(left) or len(joined) != len(right):
        return {
            "left_method": left_method,
            "right_method": right_method,
            "transition": transition,
            "metric": metric,
            "paired_query_count": int(len(joined)),
            "difference_percentage_points": None,
            "ci95_low_percentage_points": None,
            "ci95_high_percentage_points": None,
        }
    differences = joined["left"].astype(float).to_numpy() - joined["right"].astype(float).to_numpy()
    observed = float(np.mean(differences))
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        draw = rng.randint(0, len(differences), len(differences))
        samples[index] = np.mean(differences[draw])
    return {
        "left_method": left_method,
        "right_method": right_method,
        "transition": transition,
        "metric": metric,
        "paired_query_count": int(len(joined)),
        "difference_percentage_points": 100.0 * observed,
        "ci95_low_percentage_points": 100.0 * float(np.quantile(samples, 0.025)),
        "ci95_high_percentage_points": 100.0 * float(np.quantile(samples, 0.975)),
    }


def summarize_all(query_results, candidate_summaries, exposed_results):
    rows = []
    for transition in [item.key for item in TRANSITIONS]:
        for method in METHODS:
            rows.append(summarize_transition(
                query_results, candidate_summaries, exposed_results, method, transition
            ))
    return pd.DataFrame(rows)


def paired_comparisons(query_results):
    rows = []
    for transition in [item.key for item in TRANSITIONS]:
        for method in METHODS[:3]:
            for metric in ("target_valid_availability", "constraint_feasible_availability"):
                rows.append(paired_bootstrap(
                    query_results, transition, method, "DiCE", metric
                ))
    return pd.DataFrame(rows)
