"""Observable outcomes, confidence intervals and paired summaries."""

import math
from collections import Counter

import numpy as np
import pandas as pd

from .config import TERMINAL_STATUSES


def wilson_interval(successes, total, z=1.959963984540054):
    if not total:
        return None, None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def paired_bootstrap_difference(left, right, n_resamples=2000, seed=42, statistic="mean"):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) != len(right) or len(left) == 0:
        return None, None
    observed = float(np.mean(left - right)) if statistic == "mean" else float(np.median(left) - np.median(right))
    rng = np.random.RandomState(seed)
    values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.randint(0, len(left), len(left))
        if statistic == "mean":
            values[index] = np.mean(left[sample] - right[sample])
        else:
            values[index] = np.median(left[sample]) - np.median(right[sample])
    return observed, (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def mcnemar_table(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    b = int(np.sum(left & ~right))
    c = int(np.sum(~left & right))
    # Exact two-sided binomial test without depending on scipy's versioned API.
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2.0 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {"b_left_only": b, "c_right_only": c, "discordant": n, "exact_p_value": float(p_value)}


def status_from_results(generation, verifications):
    if generation.get("runtime_error"):
        return "RUNTIME_ERROR"
    if generation.get("timeout"):
        return "TIMEOUT"
    if generation.get("unsupported"):
        return "UNSUPPORTED"
    if not verifications:
        return "NO_RETURNED_CF"
    if not any(item.valid for item in verifications):
        return "RETURNED_NON_FLIPPING_CF"
    if not any(item.constraint_feasible for item in verifications):
        return "SUCCESS_VALID_CONSTRAINT_FAILED"
    return "SUCCESS_VALID_FEASIBLE"


def _rate(successes, total):
    low, high = wilson_interval(successes, total)
    return {
        "estimate": float(successes / total) if total else None,
        "numerator": int(successes),
        "denominator": int(total),
        "ci_low": low,
        "ci_high": high,
    }


def summarize_queries(query_results, candidate_results, method, regime, seed):
    queries = query_results[
        (query_results["method"] == method)
        & (query_results["constraint_regime"] == regime)
        & (query_results["seed"] == seed)
    ].copy()
    if len(candidate_results) and "method" in candidate_results.columns:
        candidates = candidate_results[
            (candidate_results["method"] == method)
            & (candidate_results["constraint_regime"] == regime)
            & (candidate_results["seed"] == seed)
        ].copy()
    else:
        candidates = pd.DataFrame()
    n_queries = len(queries)
    returned = len(candidates)
    valid = int(candidates.get("target_valid", pd.Series(dtype=bool)).fillna(False).sum()) if returned else 0
    feasible = int(candidates.get("constraint_feasible", pd.Series(dtype=bool)).fillna(False).sum()) if returned else 0
    valid_queries = int(queries.returned_any_valid_cf.fillna(False).sum()) if n_queries else 0
    feasible_queries = int(queries.returned_any_feasible_cf.fillna(False).sum()) if n_queries else 0
    runtime = pd.to_numeric(queries.runtime_ms, errors="coerce").fillna(0.0) if n_queries else pd.Series(dtype=float)
    statuses = Counter(queries.final_status.tolist())
    summary = {
        "method": method,
        "constraint_regime": regime,
        "seed": int(seed),
        "n_queries": int(n_queries),
        "exposed_cf_validity_rate": _rate(valid, returned),
        "constraint_satisfaction_rate": _rate(feasible, valid),
        "valid_cf_availability_rate": _rate(valid_queries, n_queries),
        "feasible_cf_availability_rate": _rate(feasible_queries, n_queries),
        "median_latency_ms": float(runtime.median()) if n_queries else None,
        "p95_latency_ms": float(runtime.quantile(0.95)) if n_queries else None,
        "mean_latency_ms": float(runtime.mean()) if n_queries else None,
        "effective_compute_cost_per_feasible_query_ms": float(runtime.sum() / feasible_queries) if feasible_queries else None,
        "eccf_reason": None if feasible_queries else "no_feasible_queries",
        "median_candidate_count": float(pd.to_numeric(queries.candidate_count, errors="coerce").median()) if n_queries else None,
        "p95_candidate_count": float(pd.to_numeric(queries.candidate_count, errors="coerce").quantile(0.95)) if n_queries else None,
        "status_counts": dict(statuses),
    }
    # Flatten rate objects for CSV while preserving the nested object in callers
    # that serialize JSON.
    for prefix in ("exposed_cf_validity", "constraint_satisfaction", "valid_cf_availability", "feasible_cf_availability"):
        rate = summary.pop(prefix + "_rate")
        summary[prefix + "_rate"] = rate["estimate"]
        summary[prefix + "_numerator"] = rate["numerator"]
        summary[prefix + "_denominator"] = rate["denominator"]
        summary[prefix + "_ci_low"] = rate["ci_low"]
        summary[prefix + "_ci_high"] = rate["ci_high"]
    for status in TERMINAL_STATUSES:
        summary[status.lower()] = float(statuses.get(status, 0) / n_queries) if n_queries else None
    summary["status_rate_sum"] = float(sum(summary[status.lower()] for status in TERMINAL_STATUSES)) if n_queries else None
    summary["timeout_rate"] = summary["timeout"]
    summary["runtime_error_rate"] = summary["runtime_error"]
    summary["constraint_blocked_rate"] = summary["success_valid_constraint_failed"]
    summary["no_returned_cf_rate"] = summary["no_returned_cf"]
    summary["no_valid_cf_rate"] = float(
        (statuses.get("NO_RETURNED_CF", 0) + statuses.get("RETURNED_NON_FLIPPING_CF", 0)) / n_queries
    ) if n_queries else None
    return summary
