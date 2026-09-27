
"""Summarize comparator-recovery extensions over the saved 207 query cohort."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .comparator_recovery import paired
from .metrics import summarize_queries, wilson_interval
from .run_ten_percent_eval import _atomic_csv


def postprocess(out_dir, baseline_dir):
    out = Path(out_dir)
    baseline = Path(baseline_dir)
    query_path = out / "query_results.csv"
    if not query_path.exists():
        raise FileNotFoundError(query_path)
    q = pd.read_csv(query_path)
    keys = ["method", "reference_size", "query_id", "candidate_request", "seed"]
    q = q.drop_duplicates(keys, keep="last")
    cand_path = out / "candidate_results.csv"
    candidates = pd.read_csv(cand_path) if cand_path.exists() else pd.DataFrame()
    if len(candidates):
        candidates = candidates.drop_duplicates(keys + ["candidate_id"], keep="last")
    ids = pd.read_csv(baseline / "query_ids.csv").query_id.astype(str).tolist()
    if len(ids) != 207 or len(set(ids)) != 207:
        raise ValueError("Expected the exact saved 207-query cohort.")
    eligible = q[
        (q.reference_size == "10k")
        & (q.candidate_request == 5)
        & q.query_id.astype(str).isin(set(ids))
        & q.cohort.isin(["main_30", "extended_100", "extended_207"])
    ].copy()
    summaries, slides, pairs = [], [], []
    base = pd.read_csv(baseline / "query_results.csv")
    strategies = {
        "UFCE-FF1": "SHARED_FF1",
        "UFCE-FF2": "DIGITAL_PAIRS_PLUS_CATEGORY",
        "UFCE-FF3": "DIGITAL_PAIRS_PLUS_CATEGORY",
    }
    for method in ("AR", "DiCE"):
        group = eligible[eligible.method == method].drop_duplicates("query_id", keep="last")
        cand = candidates[
            (candidates.method == method)
            & (candidates.reference_size == "10k")
            & (candidates.candidate_request == 5)
            & candidates.query_id.astype(str).isin(group.query_id.astype(str))
        ] if len(candidates) else pd.DataFrame()
        row = summarize_queries(group, cand, method, "MODERATE", 0)
        row.update({
            "reference_size": "10k",
            "reference_rows": 10007,
            "cohort": "saved_207_extension",
            "n_scheduled": len(ids),
            "n_attempted": int(group.query_id.nunique()),
            "coverage_fraction": float(group.query_id.nunique() / len(ids)),
            "interpretation": "Full saved 207-query cohort." if group.query_id.nunique() == len(ids) else "Denominator is attempted IDs; incomplete coverage is not a 207-query estimate.",
        })
        summaries.append(row)
        n = int(group.query_id.nunique())
        feasible = int(group.returned_any_feasible_cf.fillna(False).sum())
        low, high = wilson_interval(feasible, n)
        times = pd.to_numeric(group.runtime_ms, errors="coerce").dropna()
        states = group.final_status.value_counts(normalize=True).to_dict()
        setup_seconds = pd.to_numeric(group.setup_seconds, errors="coerce").max() if n else None
        slides.append({
            "method": method, "reference_size": "10k", "reference_rows": 10007,
            "n_scheduled": len(ids), "n_attempted": n, "coverage_fraction": n / len(ids),
            "feasible_queries": feasible, "feasible_availability": feasible / n if n else None,
            "feasible_ci_low": low, "feasible_ci_high": high,
            "timeout_rate": states.get("TIMEOUT", 0),
            "median_latency_ms": times.median() if n else None,
            "p95_latency_ms": times.quantile(.95) if n else None,
            "latency_is_lower_bound": states.get("TIMEOUT", 0) > 0,
            "setup_seconds": setup_seconds, "query_timeout_seconds": 60 if method == "AR" else 120,
            "note": ("DiCE gets 120s/query, twice the saved UFCE 60s/query; timed-out latency is censored." if method == "DiCE" else "AR is given 60s/query; no timeouts occurred."),
        })
        sampled = set(group.query_id.astype(str))
        left = group.sort_values("query_id").drop_duplicates("query_id", keep="last")
        for ufce, strategy in strategies.items():
            right = base[
                (base.method == ufce) & (base.reference_size == "10k")
                & (base.mi_strategy == strategy) & (base.constraint_regime == "MODERATE")
                & (base.seed == 0) & base.query_id.astype(str).isin(sampled)
            ]
            common = set(left.query_id.astype(str)) & set(right.query_id.astype(str))
            paired_row = paired(
                left[left.query_id.astype(str).isin(common)],
                right[right.query_id.astype(str).isin(common)],
                "%s expanded cohort vs %s/%s" % (method, ufce, strategy),
            )
            paired_row["n_scheduled"] = len(ids)
            paired_row["coverage_fraction"] = len(common) / len(ids)
            pairs.append(paired_row)
    transition_rows = []
    for (method, reference, phase, request), qgroup in q.groupby(
            ["method", "reference_size", "run_phase", "candidate_request"], dropna=False):
        candidate_group = candidates[
            (candidates.method == method) & (candidates.reference_size == reference)
            & (candidates.run_phase == phase) & (candidates.candidate_request == request)
        ] if len(candidates) else pd.DataFrame()
        if len(candidate_group):
            evidence = candidate_group[
                candidate_group.dedicacion_changed.fillna(False)
                & candidate_group.valid.fillna(False)
                & candidate_group.constraint_feasible.fillna(False)
            ]
        else:
            evidence = pd.DataFrame()
        transition_rows.append({
            "method": method, "reference_size": reference, "run_phase": phase,
            "candidate_request": request, "n_queries": int(qgroup.query_id.nunique()),
            "queries_with_feasible_categorical_change": int(evidence.query_id.nunique()) if len(evidence) else 0,
            "feasible_category_change_candidates": len(evidence),
            "TC_to_TP_candidates": int((evidence.dedicacion_transition == "TC->TP").sum()) if len(evidence) else 0,
            "TP_to_TC_candidates": int((evidence.dedicacion_transition == "TP->TC").sum()) if len(evidence) else 0,
        })
    transitions = pd.DataFrame(transition_rows)
    _atomic_csv(transitions, out / "categorical_transition_summary.csv")
    one_cf = q[q.run_phase == "one_cf_timeout_diagnostic"].copy()
    if len(one_cf):
        one_cf = one_cf[[
            "query_id", "method", "reference_size", "final_status", "runtime_ms",
            "generation_runtime_ms", "candidate_count", "valid_candidate_count",
            "feasible_candidate_count", "timeout_budget_seconds",
        ]]
    _atomic_csv(one_cf, out / "one_cf_timeout_diagnostic.csv")
    _atomic_csv(pd.DataFrame(summaries), out / "expanded_207_method_summary.csv")
    _atomic_csv(pd.DataFrame(slides), out / "expanded_207_slide_table.csv")
    _atomic_csv(pd.DataFrame(pairs), out / "expanded_207_pairwise.csv")
    lines = [
        "# Comparator recovery: expanded saved 207-query cohort", "",
        "This is a separate extension, not pooled into the prespecified n=30 primary cohort. Each method's rate denominator is the number of unique queries actually attempted; coverage is shown against all 207 saved IDs. If coverage is below 100%, this table is descriptive of the attempted, schedule-ordered subset and must not be described as a full-cohort estimate.", "",
        "DiCE requested five CFs with 120 seconds per query; AR used 60 seconds. Timeout latencies are censored at their budget and are lower bounds.", "",
        "## Slide-ready extension table", "", pd.DataFrame(slides).to_markdown(index=False),
        "", "## M1-M6 details", "", pd.DataFrame(summaries).to_markdown(index=False),
        "", "## Verified categorical transitions", "", transitions.to_markdown(index=False),
        "", "## One-CF timing diagnostic (separate from primary M1-M6)", "", one_cf.to_markdown(index=False),
        "", "Exact-ID paired bootstrap CIs (2,000 resamples, seed 42) and supplemental McNemar results are in expanded_207_pairwise.csv.",
        "", "Shared-host timing caveat: a separate CPU-heavy Python evaluation was observed during the extension, so wall-clock latency is not isolated.", "",
    ]
    timing_note = ("A separate CPU-heavy Python evaluation process was observed concurrently on the same host during the DiCE extension. "
                   "Wall-clock timing therefore reflects shared-host contention and is not an isolated-runtime benchmark.")
    main_report = out / "report.md"
    if main_report.exists():
        contents = main_report.read_text(encoding="utf-8")
        if timing_note not in contents:
            main_report.write_text(contents.rstrip() + "\n\n## Timing caveat\n\n" + timing_note + "\n", encoding="utf-8")
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["timing_environment_note"] = timing_note
        manifest["expanded_207_coverage"] = {
            row["method"]: {"attempted": int(row["n_attempted"]), "scheduled": int(row["n_scheduled"]), "feasible_queries": int(row["feasible_queries"])}
            for row in slides
        }
        manifest["status"] = "COMPLETE" if all(row["n_attempted"] == len(ids) for row in slides) else "PARTIAL_BUDGET"
        json_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        json_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        json_tmp.replace(manifest_path)
    (out / "expanded_207_report.md").write_text("\n".join(lines), encoding="utf-8")
    return pd.DataFrame(slides), pd.DataFrame(summaries), pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/external_binary_eval/upv_2025_comparator_recovery_v1")
    parser.add_argument("--baseline-dir", default="outputs/external_binary_eval/upv_2025_author_alignment_v1")
    args = parser.parse_args()
    postprocess(args.out_dir, args.baseline_dir)


if __name__ == "__main__":
    main()
