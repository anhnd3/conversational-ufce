"""Resumable two-hour AR/DiCE recovery evaluation for UPV-2025."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import make_adapter
from .author_alignment import _new_raw_policy
from .config import PRIMARY_SEED, RANDOM_SEED
from .metrics import mcnemar_table, paired_bootstrap_difference, summarize_queries, wilson_interval
from .run_ten_percent_eval import _append_csv, _atomic_csv, prepare_data
from .runner import _library_versions, _run_one_query, _sha256

OUT_DEFAULT = "outputs/external_binary_eval/upv_2025_comparator_recovery_v1"
BASELINE_DEFAULT = "outputs/external_binary_eval/upv_2025_author_alignment_v1"
METHODS = ("AR", "DiCE")
REGIME = "MODERATE"
CATEGORY = "dedicacion"
RESERVE = 300


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
    os.replace(str(temp), str(path))


def rss_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def ids_hash(values):
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def prepare(data_dir, baseline_dir, out_dir):
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame, train, test, generated, schema, summary, bundle, policies, checkpoint = prepare_data(data_dir, out / "metadata")
    baseline = Path(baseline_dir).resolve()
    queries = pd.read_csv(baseline / "query_ids.csv").sort_values("query_id").reset_index(drop=True)
    match_a = generated[["query_id", "student_id"]].astype(str).sort_values("query_id").reset_index(drop=True)
    match_b = queries[["query_id", "student_id"]].astype(str).sort_values("query_id").reset_index(drop=True)
    if len(queries) != 207 or queries.student_id.nunique() != 207 or not match_a.equals(match_b):
        raise ValueError("Saved 207 query IDs differ from the frozen split/checkpoint.")
    features = schema["feature_names"]
    references, reference_ids = {}, {}
    for name, manifest, expected in (("10k", "reference_ids_10k.csv", 10007), ("full_train", "reference_ids_full_train.csv", 325483)):
        ids = pd.read_csv(baseline / manifest)
        if len(ids) != expected:
            raise ValueError("%s reference manifest row count differs." % name)
        ref = train.loc[ids.reference_row_index.astype(int).tolist()].copy()
        if not ref[["student_id", "source_year"]].astype(str).reset_index(drop=True).equals(ids[["student_id", "source_year"]].astype(str).reset_index(drop=True)):
            raise ValueError("%s reference ID ordering differs." % name)
        references[name] = ref.loc[:, ["student_id", "source_year", "y"] + features].reset_index(drop=True)
        reference_ids[name] = ids
    if len(train) != 325483:
        raise ValueError("Expected 325,483 train reference rows.")
    return dict(out=out, baseline=baseline, train=train, test=test, queries=queries,
                schema=schema, dataset_summary=summary, bundle=bundle, policies=policies,
                checkpoint=Path(checkpoint).resolve(), references=references,
                reference_ids=reference_ids)


def cohorts(queries):
    main, pilot, full = [], [], []
    for _, fold in queries.groupby("fold_id", sort=True):
        if len(fold) < 6:
            raise ValueError("Every fold must contain at least six rows.")
        rows = fold.sample(6, random_state=RANDOM_SEED).sort_values("query_id")
        main.extend(rows.query_id.astype(str))
        pilot.extend(rows.sample(1, random_state=RANDOM_SEED).query_id.astype(str))
        full.extend(rows.sample(2, random_state=RANDOM_SEED).query_id.astype(str))
    return main, pilot, full


def verify_ar_equation(adapter, queries, train, schema, policies):
    checked, errors = 0, []
    numeric = set(schema["numeric_features"])
    for _, query in queries.iterrows():
        raw = query.loc[schema["feature_names"]].to_frame().T
        policy = _new_raw_policy(policies[REGIME], raw, train, schema)
        active = [f for f in policy["actionable_features"] if f in numeric and not policy["features"][f].get("immutable", True)
                  and policy["features"][f]["upper"] > policy["features"][f]["lower"]]
        if not policy["features"][CATEGORY].get("immutable", True):
            active.append(CATEGORY)
        if not active:
            continue
        w, b, x = adapter._raw_linear_model(raw, active)
        score = float(adapter.bundle.model.decision_function(adapter.bundle._prepare(raw))[0])
        errors.append(abs(score - (b + float(np.dot(w, x)))))
        checked += 1
        numeric_move = next((name for name in active if name != CATEGORY), None)
        if numeric_move:
            i = active.index(numeric_move)
            changed = raw.copy()
            changed.loc[changed.index[0], numeric_move] = policy["features"][numeric_move]["upper"]
            actual = float(adapter.bundle.model.decision_function(adapter.bundle._prepare(changed))[0])
            x2 = x.copy()
            x2[i] = float(changed.iloc[0][numeric_move])
            errors.append(abs(actual - (b + float(np.dot(w, x2)))))
            checked += 1
        if CATEGORY in active:
            i = active.index(CATEGORY)
            changed = raw.copy()
            old = str(changed.iloc[0][CATEGORY])
            changed.loc[changed.index[0], CATEGORY] = "TP" if old == "TC" else "TC"
            actual = float(adapter.bundle.model.decision_function(adapter.bundle._prepare(changed))[0])
            x2 = x.copy()
            x2[i] = 0.0 if old == "TP" else 1.0
            errors.append(abs(actual - (b + float(np.dot(w, x2)))))
            checked += 1
    maximum = max(errors) if errors else None
    return {"status": "PASS" if maximum is not None and maximum <= 1e-8 else "FAIL",
            "queries_checked": len(queries), "equation_checks": checked,
            "max_absolute_score_error": maximum, "tolerance": 1e-8}


def dice_parity(adapter, queries, bundle, schema):
    ok = 0
    for _, row in queries.iterrows():
        raw = row.loc[schema["feature_names"]].to_frame().T
        full = raw.copy()
        for f, v in adapter.numeric_imputations.items():
            full[f] = pd.to_numeric(full[f], errors="coerce").fillna(v)
        for f, v in adapter.categorical_imputations.items():
            full[f] = full[f].astype("string").fillna(v).astype(str)
        ok += int(bundle.predict_raw(raw)[0] == bundle.predict_raw(full)[0])
    return {"status": "PASS" if ok == len(queries) else "FAIL", "queries_checked": len(queries), "rows_passing": ok}


def init_outputs(ctx, budget, main, pilot, full):
    out = ctx["out"]
    chosen = ctx["queries"].loc[ctx["queries"].query_id.astype(str).isin(main)].copy()
    chosen["compatibility_pilot"] = chosen.query_id.astype(str).isin(pilot)
    chosen["full_reference_pilot"] = chosen.query_id.astype(str).isin(full)
    _atomic_csv(chosen[["query_id", "student_id", "fold_id", "compatibility_pilot", "full_reference_pilot"]].sort_values(["fold_id", "query_id"]), out / "selected_query_ids.csv")
    for name, ids in ctx["reference_ids"].items():
        _atomic_csv(ids, out / ("reference_ids_%s.csv" % name))
    write_json(out / "experiment_config.json", {
        "dataset": "UPV-2025", "model_sha256": _sha256(ctx["checkpoint"]),
        "methods": list(METHODS), "regime": REGIME, "seed": PRIMARY_SEED,
        "primary_n": 30, "queries_per_fold": 6, "full_reference_n": 10,
        "pilot_n_each_reference": 5, "selected_ids_sha256": ids_hash(main),
        "pilot_query_ids": pilot, "full_reference_query_ids": full,
        "dedicacion_policy": "TC<->TP; other categorical fields immutable",
        "AR": {"solver": "CPLEX", "numeric_raw_coef": "LR beta / scaler scale",
               "categorical_variable": "binary TC=0 TP=1 using exact score difference",
               "continuous_grid": "21 inclusive points", "integer_step": 1},
        "DiCE": {"method": "random", "sample_size": 1000, "seed": 0, "total_CFs": 5},
        "timeouts_seconds_per_query": {"AR": 60, "DiCE": 120},
        "reference_rows": {"10k": 10007, "full_train": 325483},
        "budget_minutes": budget, "session_budget_minutes": budget,
        "total_budget_minutes": 120, "reserve_before_new_query_seconds": RESERVE,
        "comparison_note": "DiCE has twice saved UFCE timeout budget.",
    })
    baseline_results = ctx["baseline"] / "query_results.csv"
    manifest_path = out / "run_manifest.json"
    session_started = datetime.now(timezone.utc)
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    sessions = list(previous_manifest.get("sessions", []))
    if previous_manifest.get("status") == "RUNNING" and previous_manifest.get("started_at_utc"):
        try:
            previous_started = datetime.fromisoformat(previous_manifest["started_at_utc"])
        except (TypeError, ValueError):
            previous_started = session_started
        sessions.append({
            "started_at_utc": previous_started.isoformat(),
            "ended_at_utc": session_started.isoformat(),
            "elapsed_seconds": max(0.0, (session_started - previous_started).total_seconds()),
            "status": "INTERRUPTED_FOR_ADAPTER_FIX",
        })
    sessions.append({"started_at_utc": session_started.isoformat(), "budget_minutes": budget, "status": "RUNNING"})
    write_json(out / "run_manifest.json", {
        "status": "RUNNING", "started_at_utc": session_started.isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": _library_versions(), "model_sha256": _sha256(ctx["checkpoint"]),
        "dataset_source_manifest": ctx["dataset_summary"].get("source_manifest", []),
        "baseline_query_results": str(baseline_results.resolve()),
        "baseline_query_results_sha256": _sha256(baseline_results),
        "reference_id_hashes": {k: hashlib.sha256(v.to_csv(index=False).encode()).hexdigest() for k, v in ctx["reference_ids"].items()},
        "peak_rss_mb_at_start": rss_mb(), "budget_minutes": budget,
        "budget_minutes_total": 120, "session_budget_minutes": budget, "sessions": sessions,
    })


def adapter_get(ctx, method, ref_name, cache, setup):
    key = (method, ref_name)
    if key in cache:
        return cache[key]
    adapter = make_adapter(method, ctx["train"], ctx["references"][ref_name], ctx["bundle"],
                           ctx["schema"], ctx["policies"], REGIME, seed=0,
                           categorical_actionable=[CATEGORY])
    setup.append({"method": method, "reference_size": ref_name,
                  "reference_rows": len(ctx["references"][ref_name]),
                  "setup_seconds": adapter.setup_seconds,
                  "setup_error": getattr(adapter, "setup_error", ""),
                  "setup_traceback": getattr(adapter, "setup_traceback", ""),
                  "peak_rss_mb_after_setup": rss_mb()})
    cache[key] = adapter
    return adapter


def completed(out):
    path = Path(out) / "query_results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    df = df[~df.final_status.isin(["RUNTIME_ERROR", "UNSUPPORTED"])]
    return set((str(r.method), str(r.reference_size), str(r.query_id), int(r.candidate_request), int(r.seed)) for r in df.itertuples())


def run_ids(ctx, method, ref_name, query_ids, phase, cohort, timeout, ncf, cache, setup, done, deadline):
    adapter = adapter_get(ctx, method, ref_name, cache, setup)
    qs = ctx["queries"].set_index(ctx["queries"].query_id.astype(str), drop=False)
    out = ctx["out"]
    for query_id in query_ids:
        key = (method, ref_name, str(query_id), int(ncf), int(PRIMARY_SEED))
        if key in done:
            continue
        remaining = deadline - time.monotonic()
        if remaining < timeout + RESERVE:
            print("budget stop before %s/%s/%s; %.0fs remain" % (method, ref_name, query_id, remaining), flush=True)
            break
        q = qs.loc[str(query_id)]
        raw = q.loc[ctx["schema"]["feature_names"]].to_frame().T
        policy = _new_raw_policy(ctx["policies"][REGIME], raw, ctx["train"], ctx["schema"])
        category = raw.iloc[0].get(CATEGORY)
        if pd.isna(category) or str(category) not in ("TC", "TP"):
            policy["features"][CATEGORY] = {"immutable": True, "missing_query": bool(pd.isna(category)), "direction": "immutable"}
            policy["categorical_actionable_features"] = []
        started = time.perf_counter()
        qr, cr = _run_one_query(adapter, raw, str(query_id), policy, ctx["bundle"], ctx["schema"],
                                method, REGIME, PRIMARY_SEED, timeout, max_candidates=ncf)
        wall = time.perf_counter() - started
        qr.update({"reference_size": ref_name, "reference_rows": len(ctx["references"][ref_name]),
                   "run_phase": phase, "cohort": cohort, "fold_id": int(q.fold_id),
                   "candidate_request": int(ncf), "timeout_budget_seconds": float(timeout),
                   "setup_seconds": adapter.setup_seconds, "wall_total_including_verification_seconds": wall,
                   "dedicacion_original": category})
        for candidate in cr:
            target = candidate.get("feature_dedicacion")
            changed = not pd.isna(category) and not pd.isna(target) and str(category) != str(target)
            candidate.update({"reference_size": ref_name, "reference_rows": len(ctx["references"][ref_name]),
                              "run_phase": phase, "cohort": cohort, "fold_id": int(q.fold_id),
                              "candidate_request": int(ncf), "dedicacion_changed": bool(changed),
                              "dedicacion_transition": "%s->%s" % (category, target) if changed else ""})
        _append_csv(qr, out / "query_results.csv")
        _append_csv(cr, out / "candidate_results.csv")
        if qr["final_status"] in ("RUNTIME_ERROR", "TIMEOUT", "UNSUPPORTED"):
            _append_csv({"query_id": query_id, "method": method, "reference_size": ref_name,
                         "run_phase": phase, "final_status": qr["final_status"],
                         "error_type": qr["error_type"], "unsupported_reason": qr["unsupported_reason"],
                         "traceback": qr.get("error_traceback", "")}, out / "failures.csv")
        for candidate in cr:
            if candidate.get("violations"):
                _append_csv(candidate, out / "violations.csv")
        done.add(key)
        print("%s %s %s %s %s generation=%.1fs verification=%.3fs" % (
            method, ref_name, phase, query_id, qr["final_status"],
            qr.get("generation_runtime_ms", 0) / 1000.0,
            qr.get("verification_runtime_ms", 0) / 1000.0), flush=True)
        if qr["final_status"] in ("RUNTIME_ERROR", "UNSUPPORTED"):
            return False
    return True


def paired(left, right, label):
    left, right = left.sort_values("query_id").reset_index(drop=True), right.sort_values("query_id").reset_index(drop=True)
    if not len(left) or len(left) != len(right) or not left.query_id.astype(str).equals(right.query_id.astype(str)):
        return {"comparison": label, "n_queries": 0, "pairing_status": "query_id_mismatch"}
    result = {"comparison": label, "n_queries": len(left), "paired_query_ids_sha256": ids_hash(left.query_id)}
    for metric, col in (("valid", "returned_any_valid_cf"), ("feasible", "returned_any_feasible_cf")):
        diff, ci = paired_bootstrap_difference(left[col].astype(float), right[col].astype(float), 2000, 42)
        result["delta_%s_pp" % metric] = 100 * diff if diff is not None else None
        result["delta_%s_ci_low_pp" % metric] = 100 * ci[0] if ci else None
        result["delta_%s_ci_high_pp" % metric] = 100 * ci[1] if ci else None
        if metric == "feasible":
            result["mcnemar"] = mcnemar_table(left[col], right[col])
    diff, ci = paired_bootstrap_difference(left.runtime_ms, right.runtime_ms, 2000, 42, statistic="median")
    result["paired_median_latency_difference_ms"] = diff
    result["paired_median_latency_ci_low_ms"] = ci[0] if ci else None
    result["paired_median_latency_ci_high_ms"] = ci[1] if ci else None
    return result


def summarize(ctx):
    out = ctx["out"]
    q = pd.read_csv(out / "query_results.csv") if (out / "query_results.csv").exists() else pd.DataFrame()
    keys = ["method", "reference_size", "query_id", "candidate_request", "seed"]
    if len(q):
        q = q.drop_duplicates(keys, keep="last")
        _atomic_csv(q, out / "query_results.csv")
    c = pd.read_csv(out / "candidate_results.csv") if (out / "candidate_results.csv").exists() else pd.DataFrame()
    if len(c):
        c = c.drop_duplicates(keys + ["candidate_id"], keep="last")
    if len(q):
        active_failures = q.final_status.isin(["RUNTIME_ERROR", "TIMEOUT", "UNSUPPORTED", "RETURNED_NON_FLIPPING_CF"])
        _atomic_csv(q.loc[active_failures].copy(), out / "failures.csv")
    if len(c) and "violations" in c.columns:
        violation_rows = c[c.violations.fillna("").astype(str).str.len() > 0]
        _atomic_csv(violation_rows, out / "violations.csv")
    if len(c):
        _atomic_csv(c, out / "candidate_results.csv")
    summaries = []
    for method in METHODS:
        for ref, expected in (("10k", 30), ("full_train", 10)):
            qr = q[(q.method == method) & (q.reference_size == ref) & (q.cohort == "main_30") & (q.candidate_request == 5)] if len(q) else pd.DataFrame()
            cr = c[(c.method == method) & (c.reference_size == ref) & (c.cohort == "main_30") & (c.candidate_request == 5)] if len(c) else pd.DataFrame()
            if len(qr):
                row = summarize_queries(qr, cr, method, REGIME, 0)
                row.update({"reference_size": ref, "reference_rows": 10007 if ref == "10k" else 325483,
                            "n_scheduled": expected, "n_completed": len(qr),
                            "n_unrun": max(0, expected - len(qr)), "cohort": "main_30"})
                summaries.append(row)
    summary = pd.DataFrame(summaries)
    _atomic_csv(summary, out / "method_summary.csv")
    return q, c, summary


def final_tables(ctx, q, c, summary, selected):
    out = ctx["out"]
    base = pd.read_csv(ctx["baseline"] / "query_results.csv")
    main_ids = set(selected)
    selection = pd.read_csv(out / "selected_query_ids.csv")
    full_ids = set(selection.loc[selection.full_reference_pilot.astype(bool), "query_id"].astype(str))
    strategies = {"UFCE-FF1": "SHARED_FF1", "UFCE-FF2": "DIGITAL_PAIRS_PLUS_CATEGORY", "UFCE-FF3": "DIGITAL_PAIRS_PLUS_CATEGORY"}
    pair_rows, slide_rows = [], []
    for method in METHODS:
        for ref, ids in (("10k", main_ids), ("full_train", full_ids)):
            new = q[(q.method == method) & (q.reference_size == ref) & (q.cohort == "main_30") & (q.candidate_request == 5) & q.query_id.astype(str).isin(ids)] if len(q) else pd.DataFrame()
            for old_method, strat in strategies.items():
                old = base[(base.method == old_method) & (base.reference_size == ref) & (base.mi_strategy == strat) & (base.constraint_regime == REGIME) & (base.seed == 0) & base.query_id.astype(str).isin(ids)]
                result = paired(new, old, "%s/%s vs %s/%s/%s" % (method, ref, old_method, ref, strat))
                result["timeout_budget_note"] = "DiCE 120s/query; UFCE/AR 60s/query" if method == "DiCE" else "60s/query"
                pair_rows.append(result)
            if len(new):
                n = len(new)
                success = int(new.returned_any_feasible_cf.fillna(False).sum())
                low, high = wilson_interval(success, n)
                runtime = pd.to_numeric(new.runtime_ms, errors="coerce").dropna()
                st = new.final_status.value_counts(normalize=True).to_dict()
                slide_rows.append({
                    "method": method, "reference_size": ref, "reference_rows": 10007 if ref == "10k" else 325483,
                    "n": n, "n_scheduled": len(ids), "feasible_queries": success,
                    "feasible_availability": success / n, "feasible_ci_low": low, "feasible_ci_high": high,
                    "timeout_rate": st.get("TIMEOUT", 0), "median_latency_ms": runtime.median() if len(runtime) else None,
                    "p95_latency_ms": runtime.quantile(.95) if len(runtime) else None,
                    "latency_is_lower_bound": st.get("TIMEOUT", 0) > 0,
                    "setup_seconds": new.setup_seconds.max(), "timeout_seconds": new.timeout_budget_seconds.max(),
                    "notes": "DiCE has twice the saved UFCE/AR query timeout." if method == "DiCE" else "Raw-space comparator; 60s/query.",
                })
        small = q[(q.method == method) & (q.reference_size == "10k") & (q.cohort == "main_30") & (q.candidate_request == 5) & q.query_id.astype(str).isin(full_ids)] if len(q) else pd.DataFrame()
        full = q[(q.method == method) & (q.reference_size == "full_train") & (q.cohort == "main_30") & (q.candidate_request == 5) & q.query_id.astype(str).isin(full_ids)] if len(q) else pd.DataFrame()
        pair_rows.append(paired(full, small, "%s full reference vs 10k" % method))
    for ref, ids in (("10k", main_ids), ("full_train", full_ids)):
        for method, strat in (("UFCE-FF1", "SHARED_FF1"), ("UFCE-FF2", "DIGITAL_PAIRS_PLUS_CATEGORY"), ("UFCE-FF3", "DIGITAL_PAIRS_PLUS_CATEGORY")):
            group = base[(base.method == method) & (base.reference_size == ref) & (base.mi_strategy == strat) & (base.constraint_regime == REGIME) & (base.seed == 0) & base.query_id.astype(str).isin(ids)]
            if len(group):
                successes = int(group.returned_any_feasible_cf.fillna(False).sum())
                low, high = wilson_interval(successes, len(group))
                times = pd.to_numeric(group.runtime_ms, errors="coerce").dropna()
                state = group.final_status.value_counts(normalize=True).to_dict()
                slide_rows.append({
                    "method": method + " " + strat, "reference_size": ref,
                    "reference_rows": 10007 if ref == "10k" else 325483, "n": len(group),
                    "n_scheduled": len(ids), "feasible_queries": successes, "feasible_availability": successes / len(group),
                    "feasible_ci_low": low, "feasible_ci_high": high, "timeout_rate": state.get("TIMEOUT", 0),
                    "median_latency_ms": times.median() if len(times) else None,
                    "p95_latency_ms": times.quantile(.95) if len(times) else None,
                    "latency_is_lower_bound": state.get("TIMEOUT", 0) > 0,
                    "setup_seconds": group.setup_seconds.max(), "timeout_seconds": 60,
                    "notes": "Saved UFCE-FF result, 60s/query.",
                })
    _atomic_csv(pd.DataFrame(pair_rows), out / "pairwise_comparison.csv")
    slides = pd.DataFrame(slide_rows)
    _atomic_csv(slides, out / "slide_table.csv")
    return slides


def finish(ctx, started, compatibility, parity, selected):
    out = ctx["out"]
    setup_file = out / "adapter_setup.csv"
    if setup_file.exists():
        setup = pd.read_csv(setup_file).drop_duplicates(["method", "reference_size"], keep="last")
        _atomic_csv(setup, setup_file)
    _atomic_csv(pd.DataFrame(compatibility), out / "compatibility_pilot.csv")
    write_json(out / "model_parity_checks.json", parity)
    q, c, sm = summarize(ctx)
    slides = final_tables(ctx, q, c, sm, selected) if len(q) else pd.DataFrame()
    if len(q):
        statuses = q.final_status.value_counts().to_dict()
        terminal = ["RUNTIME_ERROR", "TIMEOUT", "UNSUPPORTED", "NO_RETURNED_CF", "RETURNED_NON_FLIPPING_CF", "SUCCESS_VALID_CONSTRAINT_FAILED", "SUCCESS_VALID_FEASIBLE"]
        total = sum(statuses.get(x, 0) for x in terminal) / len(q)
        lines = [
            "# UPV-2025 Comparator Recovery", "",
            "AR uses ActionSet/CPLEX with the exact affine score of the frozen, preprocessed LR in raw actionable coordinates. DiCE sees 75 numeric and 6 categorical raw predictors and a wrapper around the frozen pipeline.",
            "",
            "Primary paired cohort: 30 fixed queries (6/fold); full reference subset: 10 of those queries (2/fold). DiCE requests five CFs with a 120-second total per-query timeout; AR and saved UFCE use 60 seconds/query. Timeout latency is censored at budget and is a lower bound.",
            "",
            "The historical DiCE runs are separate: first pilot = IndexError at roughly 50–89 ms; retry = 2 TIMEOUTs at roughly 60s. Both timeouts covered one query requesting up to five CFs, not each CF.",
            "", "## Slide-ready table", "", slides.to_markdown(index=False),
            "", "## Primary M1–M6 summary", "", sm.to_markdown(index=False),
            "", "## Terminal statuses", "", "Rows: %d; terminal-rate sum: %.3f" % (len(q), total),
            "", json.dumps(statuses, indent=2, sort_keys=True),
            "", "See pairwise_comparison.csv for exact-ID bootstrap intervals and McNemar tests. DiCE has twice the saved UFCE budget, so no winner claim follows from availability or latency alone.", "",
        ]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    finished_at = datetime.now(timezone.utc)
    session_elapsed = time.monotonic() - started
    manifest.update({"status": "COMPLETE_OR_BUDGET_EXHAUSTED", "finished_at_utc": finished_at.isoformat(),
                     "session_elapsed_seconds": session_elapsed, "peak_rss_mb": rss_mb(),
                     "compatibility": compatibility, "parity": parity})
    sessions = manifest.get("sessions", [])
    if sessions:
        sessions[-1].update({"ended_at_utc": finished_at.isoformat(), "elapsed_seconds": session_elapsed,
                             "status": "COMPLETE_OR_BUDGET_EXHAUSTED"})
        manifest["sessions"] = sessions
    manifest["elapsed_seconds"] = sum(float(x.get("elapsed_seconds", 0.0)) for x in sessions)
    manifest["budget_minutes_total"] = 120
    write_json(out / "run_manifest.json", manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/upv_2025")
    parser.add_argument("--baseline-dir", default=BASELINE_DEFAULT)
    parser.add_argument("--out-dir", default=OUT_DEFAULT)
    parser.add_argument("--budget-minutes", type=float, default=120)
    args = parser.parse_args(argv)
    start = time.monotonic()
    deadline = start + args.budget_minutes * 60
    ctx = prepare(args.data_dir, args.baseline_dir, args.out_dir)
    main_ids, pilot_ids, full_ids = cohorts(ctx["queries"])
    init_outputs(ctx, args.budget_minutes, main_ids, pilot_ids, full_ids)
    cache, setup, done = {}, [], completed(ctx["out"])
    parity, compatibility = {}, []
    ok_by_method = {name: True for name in METHODS}
    setup_path = ctx["out"] / "adapter_setup.csv"

    for method in METHODS:
        method_start = time.monotonic()
        print("compatibility pilot: %s, 5 queries per reference" % method, flush=True)
        a10 = adapter_get(ctx, method, "10k", cache, setup)
        if method == "AR":
            parity["AR_exact_raw_score"] = verify_ar_equation(a10, ctx["queries"], ctx["train"], ctx["schema"], ctx["policies"])
        else:
            parity["DiCE_factual_prediction"] = dice_parity(a10, ctx["queries"], ctx["bundle"], ctx["schema"])
        parity_key = "AR_exact_raw_score" if method == "AR" else "DiCE_factual_prediction"
        if parity[parity_key]["status"] != "PASS":
            ok_by_method[method] = False
            compatibility.append({"method": method, "status": "STOP_PARITY_FAILURE", **parity[parity_key]})
            continue
        for ref in ("10k", "full_train"):
            adapter_get(ctx, method, ref, cache, setup)
            adapter = cache[(method, ref)]
            if getattr(adapter, "setup_error", ""):
                ok_by_method[method] = False
                compatibility.append({"method": method, "reference_size": ref, "status": "STOP_SETUP_ERROR",
                                      "error": adapter.setup_error, "traceback": getattr(adapter, "setup_traceback", "")})
                break
            complete = run_ids(ctx, method, ref, pilot_ids, "compatibility_pilot", "main_30",
                               60 if method == "AR" else 120, 5, cache, setup, done, deadline)
            results = pd.read_csv(ctx["out"] / "query_results.csv") if (ctx["out"] / "query_results.csv").exists() else pd.DataFrame()
            part = results[(results.method == method) & (results.reference_size == ref) & results.query_id.astype(str).isin(pilot_ids) & (results.candidate_request == 5)]
            part = part.drop_duplicates("query_id", keep="last")
            bad = bool(len(part) and part.final_status.isin(["RUNTIME_ERROR", "UNSUPPORTED"]).any())
            compatibility.append({"method": method, "reference_size": ref, "pilot_n": len(part),
                                  "runtime_errors": int((part.final_status == "RUNTIME_ERROR").sum()) if len(part) else 0,
                                  "unsupported": int((part.final_status == "UNSUPPORTED").sum()) if len(part) else 0,
                                  "timeouts": int((part.final_status == "TIMEOUT").sum()) if len(part) else 0,
                                  "status": "PASS" if complete and not bad and len(part) == len(pilot_ids) else "STOP_OR_BUDGET_INCOMPLETE"})
            if setup:
                _append_csv(setup, setup_path)
                setup.clear()
            if bad or not complete:
                ok_by_method[method] = False
                break
        if not ok_by_method[method]:
            continue
        for ref, ids in (("10k", main_ids), ("full_train", full_ids)):
            run_ids(ctx, method, ref, ids, "primary", "main_30", 60 if method == "AR" else 120,
                    5, cache, setup, done, deadline)
            if setup:
                _append_csv(setup, setup_path)
                setup.clear()
        if method == "AR" and time.monotonic() - method_start <= 1200:
            run_ids(ctx, method, "10k", ctx["queries"].query_id.astype(str).tolist(),
                    "extension", "extended_207", 60, 5, cache, setup, done, deadline)
        if setup:
            _append_csv(setup, setup_path)
            setup.clear()

    results_path = ctx["out"] / "query_results.csv"
    if ok_by_method["DiCE"] and results_path.exists() and time.monotonic() < deadline - 1200:
        results = pd.read_csv(results_path)
        timed = results[(results.method == "DiCE") & (results.reference_size == "10k") &
                        (results.cohort == "main_30") & (results.candidate_request == 5) &
                        (results.final_status == "TIMEOUT")].query_id.astype(str).drop_duplicates().tolist()[:3]
        run_ids(ctx, "DiCE", "10k", timed, "one_cf_timeout_diagnostic",
                "one_cf_timeout_diagnostic", 300, 1, cache, setup, done, deadline)
        if setup:
            _append_csv(setup, setup_path)
            setup.clear()
    if ok_by_method["DiCE"] and time.monotonic() < deadline - 420:
        for limit, cohort in ((100, "extended_100"), (207, "extended_207")):
            run_ids(ctx, "DiCE", "10k", ctx["queries"].query_id.astype(str).tolist()[:limit],
                    "extension", cohort, 120, 5, cache, setup, done, deadline)
            if setup:
                _append_csv(setup, setup_path)
                setup.clear()
            if time.monotonic() >= deadline - 420:
                break
    if setup:
        _append_csv(setup, setup_path)
    finish(ctx, start, compatibility, parity, main_ids)
    print("outputs: %s" % ctx["out"], flush=True)


if __name__ == "__main__":
    main()

