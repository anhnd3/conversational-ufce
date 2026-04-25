#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import (
    add_summary_output_args,
    build_runner_command,
    call_with_legacy_stdout_redirect,
    progress_iter,
    recompute_and_validate_aggregates,
    safe_mean,
    summarize_latency_ms,
    summarize_numeric_values,
    write_optional_summary_outputs,
)
from llm.src.part2_eval.corpora import TIER_C_CORPUS_PATH, load_tier_c_bank_backend_corpus
from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.product.config import try_get_git_commit
from llm.src.runtime.constraint_spec import apply_constraint_spec_to_candidates
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.reproducibility import sort_counterfactual_candidates
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult, RuntimeResult
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso
from ufce.core import cfmethods
from ufce.core.cfmethods import ar_cfexp, dfexp, dice_cfexp, sfexp, tfexp


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_backend_comparison"
RUNNER_SCOPE = "part2_g4_backend_comparison"
SCORER_VERSION = "part2_backend_comparison_report_v2"
METHODS = ("UFCE1", "UFCE2", "UFCE3", "DiCE", "AR")
DEFAULT_SINGLE_CASE_SEED_ID = "TIERC-001"
REQUEST_CONSTRAINTS_BLOCKED_CODE = "REQUEST_CONSTRAINTS_BLOCKED"
G4_BASELINE_UNCONSTRAINED = "G4_baseline_unconstrained"
G4_CONSTRAINT_AWARE = "G4_constraint_aware"
FAILURE_BUCKETS = ("no_candidate", "no_flip", "invariant_fail", "constraint_blocked", "timeout_or_error")
AUTHOR_STYLE_UFCE_CONFIG = {
    "radius": 500,
    "n_neighbors": 1000,
    "contprox_metric": "euclidean",
    "min_act": 1,
    "min_feas": 1,
    "atol": 1e-5,
    "no_cf": 50,
    "flip_filter_enabled": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Part II G4 backend comparison report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--tier-c-corpus", type=Path, default=TIER_C_CORPUS_PATH)
    parser.add_argument("--single-case", action="store_true")
    seed_selector = parser.add_mutually_exclusive_group()
    seed_selector.add_argument("--seed-id", default=None)
    seed_selector.add_argument("--seed-index", type=int, default=None, help="0-based index into Tier C main seeds.")
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_backend_comparison_report(args=args, command=command)
    markdown = render_markdown(summary)
    write_optional_summary_outputs(
        summary=summary,
        summary_json_path=args.summary_json,
        summary_markdown_path=args.summary_md,
        markdown_text=markdown,
    )
    if args.summary_json is None:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print(f"summary_json_path={Path(args.summary_json).resolve()}")
    return 0


def run_backend_comparison_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    baseline_catalog = load_catalog(args.baseline_catalog)
    runtime = RuntimeOrchestrator()
    model_registry = runtime.model_registry if isinstance(runtime.model_registry, ModelRegistry) else ModelRegistry()
    policy_registry = runtime.policy_registry if isinstance(runtime.policy_registry, PolicyRegistry) else PolicyRegistry(model_registry)
    context = policy_registry.get_runtime_context("bank")
    bank_bundle = model_registry.get_bundle("bank")
    corpus = load_tier_c_bank_backend_corpus(args.tier_c_corpus)
    feature_ranges = build_feature_ranges(bank_bundle.dataset_df, list(context.bundle.feature_order))
    ar_scaler = StandardScaler().fit(bank_bundle.Xtrain.loc[:, context.bundle.feature_order].copy())
    initialize_author_style_ufce()
    single_case_requested = bool(args.single_case or args.seed_id is not None or args.seed_index is not None)
    selected_main_seeds, selected_constrained_subset, selection_metadata = select_seed_batches(
        corpus=corpus,
        single_case=single_case_requested,
        seed_id=args.seed_id,
        seed_index=args.seed_index,
    )

    run_id = "part2_backend_comparison_" + local_now_compact()
    run_root = args.out_dir / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    main_seed_results = [
        evaluate_seed(
            runtime=runtime,
            context=context,
            seed=seed,
            feature_ranges=feature_ranges,
            active_constraint_spec=None,
            ar_scaler=ar_scaler,
        )
        for seed in progress_iter(
            selected_main_seeds,
            enabled=not args.no_progress,
            desc="Backend seeds",
            unit="seed",
        )
    ]
    constrained_seed_results = [
        evaluate_seed(
            runtime=runtime,
            context=context,
            seed=seed,
            feature_ranges=feature_ranges,
            active_constraint_spec=seed["constraint_spec"],
            ar_scaler=ar_scaler,
        )
        for seed in progress_iter(
            selected_constrained_subset,
            enabled=not args.no_progress,
            desc="Backend constrained subset",
            unit="seed",
        )
    ]

    fairness_contract = {
        "dataset": "bank",
        "main_table_scope": "unconstrained",
        "secondary_table_scope": "matched constrained subset",
        "seed_selection_rule": corpus.get("selection_rule"),
        "selection_space": corpus.get("selection_space"),
        "rejected_label": corpus.get("rejected_label"),
        "rules": [
            "same factual seeds",
            "same predictor",
            "same desired class",
            "same preprocessing and normalization",
            "same invariant validator",
            "same effective post-generation constraint filter when applicable",
            "same deterministic final-candidate selection rule",
        ],
        "preprocessing_and_normalization": {
            "factual_profiles": "canonicalized to the shared bank feature order before method execution",
            "proximity_metric": "feature-range normalized Euclidean distance over the shared bank feature order",
        },
        "deterministic_final_candidate_selection_rule": (
            "Normalize each method output to the shared candidate schema, sort candidates with the shared deterministic "
            "ordering rule, retain invariant-valid candidates, apply the shared post-generation constraint filter when "
            "constraints are active, and select the first remaining candidate. If none remain, record failure."
        ),
    }
    baseline_catalog_sha256 = sha256_file(baseline_catalog.source_path)
    aggregate_blocks = {
        G4_BASELINE_UNCONSTRAINED: build_method_table(main_seed_results),
        G4_CONSTRAINT_AWARE: build_method_table(constrained_seed_results),
        "failure_breakdown": {
            G4_BASELINE_UNCONSTRAINED: build_failure_breakdown(main_seed_results),
            G4_CONSTRAINT_AWARE: build_failure_breakdown(constrained_seed_results),
        },
    }
    single_case_results = None
    if single_case_requested and main_seed_results:
        single_case_results = build_single_case_results(
            main_seed_result=main_seed_results[0],
            constrained_seed_results=constrained_seed_results,
            selection_metadata=selection_metadata,
        )
    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "catalog_version": baseline_catalog.catalog_version,
        "catalog_path": str(baseline_catalog.source_path),
        "catalog_sha256": baseline_catalog_sha256,
        "git_commit": try_get_git_commit(ROOT),
        "corpus_path": str(Path(args.tier_c_corpus).resolve()),
        "corpus_version": corpus["corpus_version"],
        "corpus_sha256": corpus["corpus_sha256"],
        "report_json_path": str((run_root / "backend_comparison_report.json").resolve()),
        "report_markdown_path": str((run_root / "backend_comparison_report.md").resolve()),
        "loaded_corpora": {
            "tier_c": {
                "corpus_path": str(Path(args.tier_c_corpus).resolve()),
                "corpus_version": corpus["corpus_version"],
                "corpus_sha256": corpus["corpus_sha256"],
            }
        },
        "fairness_contract": fairness_contract,
        "method_list": list(METHODS),
        "single_case_mode": single_case_requested,
        "seed_selection": selection_metadata,
        G4_BASELINE_UNCONSTRAINED: aggregate_blocks[G4_BASELINE_UNCONSTRAINED],
        G4_CONSTRAINT_AWARE: aggregate_blocks[G4_CONSTRAINT_AWARE],
        "failure_breakdown": aggregate_blocks["failure_breakdown"],
        "main_table": aggregate_blocks[G4_BASELINE_UNCONSTRAINED],
        "secondary_tables": {
            "constrained_subset": aggregate_blocks[G4_CONSTRAINT_AWARE],
        },
        "seed_counts": {
            "main_unconstrained": len(main_seed_results),
            "constrained_subset": len(constrained_seed_results),
        },
        "per_seed_results": {
            "main_unconstrained": main_seed_results,
            "constrained_subset": constrained_seed_results,
        },
        "per_seed_method_results": {
            "main_unconstrained": flatten_seed_results(main_seed_results),
            "constrained_subset": flatten_seed_results(constrained_seed_results),
        },
        "single_case_results": single_case_results,
    }
    summary["aggregate_validation"] = recompute_and_validate_aggregates(
        expected_blocks={
            G4_BASELINE_UNCONSTRAINED: summary[G4_BASELINE_UNCONSTRAINED],
            G4_CONSTRAINT_AWARE: summary[G4_CONSTRAINT_AWARE],
            "failure_breakdown": summary["failure_breakdown"],
        },
        recomputed_blocks=aggregate_blocks,
    )
    write_json(run_root / "backend_comparison_report.json", summary)
    (run_root / "backend_comparison_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def initialize_author_style_ufce() -> None:
    call_with_legacy_stdout_redirect(
        cfmethods.initUFCE,
        radius=int(AUTHOR_STYLE_UFCE_CONFIG["radius"]),
        n_neighbors=int(AUTHOR_STYLE_UFCE_CONFIG["n_neighbors"]),
        contprox_metric=str(AUTHOR_STYLE_UFCE_CONFIG["contprox_metric"]),
        min_act=int(AUTHOR_STYLE_UFCE_CONFIG["min_act"]),
        min_feas=int(AUTHOR_STYLE_UFCE_CONFIG["min_feas"]),
        atol=float(AUTHOR_STYLE_UFCE_CONFIG["atol"]),
    )


def select_seed_batches(
    *,
    corpus: dict[str, Any],
    single_case: bool,
    seed_id: str | None,
    seed_index: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    main_seeds = list(corpus["seeds"])
    constrained_subset = list(corpus["constrained_subset"])
    if not single_case:
        return (
            main_seeds,
            constrained_subset,
            {
                "mode": "full_corpus",
                "main_seed_count": len(main_seeds),
                "constrained_subset_count": len(constrained_subset),
            },
        )

    if seed_id is None and seed_index is None:
        seed_id = DEFAULT_SINGLE_CASE_SEED_ID
    selected_seed = resolve_main_seed(main_seeds=main_seeds, seed_id=seed_id, seed_index=seed_index)
    matched_constrained = [
        item
        for item in constrained_subset
        if item.get("profile") == selected_seed.get("profile")
    ][:1]
    return (
        [selected_seed],
        matched_constrained,
        {
            "mode": "single_case",
            "seed_id": selected_seed["seed_id"],
            "seed_index": next(index for index, item in enumerate(main_seeds) if item["seed_id"] == selected_seed["seed_id"]),
            "matched_constrained_seed_id": matched_constrained[0]["seed_id"] if matched_constrained else None,
            "main_seed_count": 1,
            "constrained_subset_count": len(matched_constrained),
        },
    )


def resolve_main_seed(
    *,
    main_seeds: list[dict[str, Any]],
    seed_id: str | None,
    seed_index: int | None,
) -> dict[str, Any]:
    if seed_id is not None:
        for item in main_seeds:
            if item["seed_id"] == seed_id:
                return item
        raise KeyError(f"Unknown Tier C main seed_id: {seed_id}")
    if seed_index is not None:
        if seed_index < 0 or seed_index >= len(main_seeds):
            raise IndexError(f"Tier C main seed index out of range: {seed_index}")
        return main_seeds[seed_index]
    return main_seeds[0]


def build_single_case_results(
    *,
    main_seed_result: dict[str, Any],
    constrained_seed_results: list[dict[str, Any]],
    selection_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed_selection": dict(selection_metadata),
        "selected_seed": {
            "seed_id": main_seed_result["seed_id"],
            "dataset": main_seed_result["dataset"],
            "profile": dict(main_seed_result["profile"]),
        },
        "main_unconstrained": {
            "seed_id": main_seed_result["seed_id"],
            "methods": dict(main_seed_result["methods"]),
        },
        "matched_constrained_subset": None
        if not constrained_seed_results
        else {
            "seed_id": constrained_seed_results[0]["seed_id"],
            "active_constraint_spec": constrained_seed_results[0]["active_constraint_spec"],
            "methods": dict(constrained_seed_results[0]["methods"]),
        },
    }


def evaluate_seed(
    *,
    runtime: RuntimeOrchestrator,
    context,
    seed: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
    active_constraint_spec: dict[str, Any] | None,
    ar_scaler,
) -> dict[str, Any]:
    canonical_profile = canonicalize_profile(runtime=runtime, context=context, profile=seed["profile"])
    canonical_profile_dict = canonical_profile.iloc[0].to_dict()
    factual_prediction = runtime.prediction_service.predict("bank", canonical_profile, context)
    per_method: dict[str, Any] = {}
    for method_name in METHODS:
        started = time.perf_counter()
        raw_candidates, raw_error = generate_raw_candidates(
            method_name=method_name,
            runtime=runtime,
            context=context,
            canonical_profile=canonical_profile,
            ar_scaler=ar_scaler,
        )
        label_flipping_candidates = [
            candidate
            for candidate in raw_candidates
            if candidate_flips_label(runtime=runtime, context=context, candidate=candidate)
        ]
        raw_flip_count = len(label_flipping_candidates)
        actionable_label_flip_count = sum(
            1
            for candidate in label_flipping_candidates
            if compute_candidate_actionability(
                candidate=candidate,
                active_constraint_spec=active_constraint_spec or {},
                context=context,
            )
            == 1
        )
        raw_validity = int(raw_flip_count > 0)
        final_selection = select_final_candidate(
            runtime=runtime,
            context=context,
            factual_prediction=factual_prediction,
            factual_profile=canonical_profile_dict,
            raw_candidates=raw_candidates,
            active_constraint_spec=active_constraint_spec,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        candidate = final_selection["selected_candidate"]
        selected_candidate_emitted = int(candidate is not None)
        failure_bucket = classify_failure_bucket(
            raw_error=raw_error,
            raw_candidate_count=len(raw_candidates),
            raw_label_flip_count=raw_flip_count,
            invariant_valid_candidate_count=int(final_selection["invariant_valid_candidate_count"]),
            constraint_compliant_candidate_count=int(final_selection["constraint_compliant_candidate_count"]),
            selected_candidate_emitted=selected_candidate_emitted,
            active_constraint_spec=active_constraint_spec,
        )
        failure_reason_codes = (
            ["TIMEOUT_OR_ERROR"] if raw_error is not None else list(final_selection["reason_codes"])
        )
        per_method[method_name] = {
            "method": method_name,
            "raw_candidate_count": len(raw_candidates),
            "raw_label_flip_count": raw_flip_count,
            "raw_flip_count": raw_flip_count,
            "raw_validity": raw_validity,
            "actionable_label_flip_count": actionable_label_flip_count,
            "invariant_valid_candidate_count": int(final_selection["invariant_valid_candidate_count"]),
            "constraint_compliant_candidate_count": int(final_selection["constraint_compliant_candidate_count"]),
            "invariant_pass_count": int(final_selection["invariant_valid_candidate_count"]),
            "selected_candidate_emitted": selected_candidate_emitted,
            "final_selection_success": selected_candidate_emitted,
            "failure_bucket": failure_bucket,
            "failure_reason_codes": failure_reason_codes,
            "final_failure_reason_codes": failure_reason_codes,
            "first_failed_invariant_violations": final_selection["first_failed_invariant_violations"],
            "selected_candidate": None if candidate is None else candidate.to_dict(),
            "success": selected_candidate_emitted,
            "failure": int(candidate is None),
            "plausibility": int(final_selection["invariant_valid_candidate_count"] > 0),
            "feasibility": selected_candidate_emitted,
            "actionability": int(actionable_label_flip_count > 0),
            "proximity": None
            if candidate is None
            else compute_normalized_proximity(
                factual_profile=canonical_profile_dict,
                counterfactual_profile=candidate.profile,
                feature_ranges=feature_ranges,
                feature_order=list(context.bundle.feature_order),
            ),
            "sparsity": None if candidate is None else len(candidate.changed_features),
            "latency_ms": elapsed_ms,
            "backend_error": raw_error,
            "raw_error": raw_error,
            "selection_reason_codes": list(final_selection["reason_codes"]),
        }
    return {
        "seed_id": seed["seed_id"],
        "dataset": seed["dataset"],
        "profile": dict(seed["profile"]),
        "active_constraint_spec": active_constraint_spec,
        "methods": per_method,
    }


def canonicalize_profile(*, runtime: RuntimeOrchestrator, context, profile: dict[str, Any]) -> pd.DataFrame:
    request = runtime.profile_service.parse_request(
        {"dataset": "bank", "profile": dict(profile)},
        "bank",
        feature_order=list(context.bundle.feature_order),
    )
    return runtime.profile_service.canonicalize(request, context, request.to_dict())


def generate_raw_candidates(
    *,
    method_name: str,
    runtime: RuntimeOrchestrator,
    context,
    canonical_profile: pd.DataFrame,
    ar_scaler,
) -> tuple[list[CounterfactualCandidate], str | None]:
    try:
        if method_name in {"UFCE1", "UFCE2", "UFCE3"}:
            raw_df = generate_author_style_ufce_candidates(
                method_name=method_name,
                context=context,
                canonical_profile=canonical_profile,
            )
            return normalize_candidate_frame(
                method_name=method_name,
                output_df=raw_df,
                canonical_profile=canonical_profile,
                context=context,
            ), None
        if method_name == "DiCE":
            raw_df, _idx, _time_s, _flag = call_with_legacy_stdout_redirect(
                dice_cfexp,
                context.bundle.dataset_df.copy(),
                canonical_profile.copy(),
                list(context.policy.numeric_features),
                list(context.policy.f2change),
                int(AUTHOR_STYLE_UFCE_CONFIG["no_cf"]),
                context.bundle.lr,
                dict(context.policy.uf),
                context.policy.label_col,
            )
            return normalize_candidate_frame(
                method_name=method_name,
                output_df=raw_df,
                canonical_profile=canonical_profile,
                context=context,
            ), None
        if method_name == "AR":
            raw_df, _time_s, _idx = call_with_legacy_stdout_redirect(
                ar_cfexp,
                context.bundle.X.loc[:, context.bundle.feature_order].copy(),
                list(context.policy.numeric_features),
                context.bundle.lr,
                canonical_profile.copy(),
                dict(context.policy.uf),
                ar_scaler,
                context.bundle.Xtrain.loc[:, context.bundle.feature_order].copy(),
                list(context.policy.f2change),
            )
            return normalize_candidate_frame(
                method_name=method_name,
                output_df=raw_df,
                canonical_profile=canonical_profile,
                context=context,
            ), None
        raise ValueError(f"Unsupported backend comparison method: {method_name}")
    except Exception as exc:  # pragma: no cover - exercised through live method execution or monkeypatch tests
        return [], f"{type(exc).__name__}: {exc}"


def generate_author_style_ufce_candidates(
    *,
    method_name: str,
    context,
    canonical_profile: pd.DataFrame,
) -> pd.DataFrame:
    feature_order = list(context.bundle.feature_order)
    positive_space = (
        context.bundle.dataset_df.loc[
            context.bundle.dataset_df[context.bundle.label_col] == context.policy.desired_outcome,
            feature_order,
        ]
        .copy()
        .reset_index(drop=True)
    )
    feature_matrix = context.bundle.X.loc[:, feature_order].copy()
    method_kwargs = {
        "flip_filter_enabled": bool(AUTHOR_STYLE_UFCE_CONFIG["flip_filter_enabled"]),
    }
    no_cf = int(AUTHOR_STYLE_UFCE_CONFIG["no_cf"])
    if method_name == "UFCE1":
        raw_df, *_ = call_with_legacy_stdout_redirect(
            sfexp,
            feature_matrix,
            positive_space,
            canonical_profile.copy(),
            dict(context.policy.uf),
            dict(context.policy.step),
            list(context.policy.f2change),
            list(context.policy.numeric_features),
            list(context.policy.categorical_features),
            context.bundle.lr,
            int(context.policy.desired_outcome),
            no_cf,
            feature_order,
            **method_kwargs,
        )
        return raw_df
    if method_name == "UFCE2":
        raw_df, *_ = call_with_legacy_stdout_redirect(
            dfexp,
            feature_matrix,
            positive_space,
            canonical_profile.copy(),
            dict(context.policy.uf),
            [list(item) for item in context.mi_feature_pairs[:5]],
            list(context.policy.numeric_features),
            list(context.policy.categorical_features),
            feature_order,
            list(context.policy.protected_features),
            context.bundle.lr,
            int(context.policy.desired_outcome),
            no_cf,
            feature_order,
            **method_kwargs,
        )
        return raw_df
    if method_name == "UFCE3":
        raw_df, *_ = call_with_legacy_stdout_redirect(
            tfexp,
            feature_matrix,
            positive_space,
            canonical_profile.copy(),
            dict(context.policy.uf),
            [list(item) for item in context.mi_feature_pairs[:5]],
            list(context.policy.numeric_features),
            list(context.policy.categorical_features),
            list(context.policy.f2change),
            list(context.policy.protected_features),
            context.bundle.lr,
            int(context.policy.desired_outcome),
            no_cf,
            feature_order,
            **method_kwargs,
        )
        return raw_df
    raise ValueError(f"Unsupported author-style UFCE method: {method_name}")


def normalize_candidate_frame(
    *,
    method_name: str,
    output_df: Any,
    canonical_profile: pd.DataFrame,
    context,
) -> list[CounterfactualCandidate]:
    if not isinstance(output_df, pd.DataFrame) or output_df.empty:
        return []
    feature_order = list(context.bundle.feature_order)
    if not set(feature_order).issubset(output_df.columns):
        return []
    factual_row = canonical_profile.loc[:, feature_order].iloc[0]
    normalized_df = output_df.loc[:, feature_order].reset_index(drop=True)
    candidates: list[CounterfactualCandidate] = []
    for index, row in normalized_df.iterrows():
        profile: dict[str, Any] = {}
        changed: list[str] = []
        for feature_name in feature_order:
            value = row[feature_name]
            feature_type = context.policy.feature_type_map[feature_name]
            if feature_type == "float":
                profile[feature_name] = float(value)
                factual_value = float(factual_row[feature_name])
                if abs(float(value) - factual_value) > 1e-9:
                    changed.append(feature_name)
            else:
                profile[feature_name] = int(value)
                factual_value = int(factual_row[feature_name])
                if int(value) != factual_value:
                    changed.append(feature_name)
        candidates.append(
            CounterfactualCandidate(
                method=method_name,
                rank=index + 1,
                profile=profile,
                changed_features=changed,
            )
        )
    return sort_counterfactual_candidates(candidates=candidates, feature_order=feature_order, prefer_fewer_changes=False)


def select_final_candidate(
    *,
    runtime: RuntimeOrchestrator,
    context,
    factual_prediction,
    factual_profile: dict[str, Any],
    raw_candidates: list[CounterfactualCandidate],
    active_constraint_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_candidates: list[CounterfactualCandidate] = []
    first_failed_invariant_violations: list[str] | None = None
    for candidate in raw_candidates:
        runtime_result = RuntimeResult(
            dataset="bank",
            controller_state="TERMINAL_SUCCESS",
            prediction=factual_prediction,
            counterfactual=CounterfactualResult(feasible=True, candidates=[candidate], reason_codes=[]),
            reason_codes=[],
            runtime_mode=runtime.runtime_mode,
            invariant_validation=None,
            debug_trace=None,
        )
        invariant = runtime.invariant_validator.validate(
            result=runtime_result,
            current_profile=factual_profile,
            context=context,
        )
        if invariant.status == "passed":
            valid_candidates.append(candidate)
        elif first_failed_invariant_violations is None:
            violations = invariant.details.get("violations") if isinstance(invariant.details, dict) else None
            if isinstance(violations, list) and violations:
                first_failed_invariant_violations = [str(item) for item in violations]
    if not valid_candidates:
        return {
            "selected_candidate": None,
            "reason_codes": ["NO_VALID_CANDIDATE_AFTER_INVARIANT_FILTER"],
            "invariant_valid_candidate_count": 0,
            "constraint_compliant_candidate_count": 0,
            "first_failed_invariant_violations": first_failed_invariant_violations,
        }

    ordered_candidates = sort_counterfactual_candidates(
        candidates=valid_candidates,
        feature_order=list(context.bundle.feature_order),
        prefer_fewer_changes=False,
    )
    constraint_compliant_candidate_count = len(ordered_candidates)
    if active_constraint_spec:
        filtered_result, _debug_summary = apply_constraint_spec_to_candidates(
            result=CounterfactualResult(feasible=True, candidates=ordered_candidates, reason_codes=[]),
            constraint_spec=active_constraint_spec,
            feature_order=list(context.bundle.feature_order),
            sort_candidates=sort_counterfactual_candidates,
            request_constraints_blocked_code=REQUEST_CONSTRAINTS_BLOCKED_CODE,
        )
        if not filtered_result.feasible or not filtered_result.candidates:
            return {
                "selected_candidate": None,
                "reason_codes": list(filtered_result.reason_codes),
                "invariant_valid_candidate_count": len(valid_candidates),
                "constraint_compliant_candidate_count": 0,
                "first_failed_invariant_violations": first_failed_invariant_violations,
            }
        ordered_candidates = list(filtered_result.candidates)
        constraint_compliant_candidate_count = len(ordered_candidates)

    return {
        "selected_candidate": ordered_candidates[0],
        "reason_codes": [],
        "invariant_valid_candidate_count": len(valid_candidates),
        "constraint_compliant_candidate_count": constraint_compliant_candidate_count,
        "first_failed_invariant_violations": first_failed_invariant_violations,
    }


def candidate_flips_label(*, runtime: RuntimeOrchestrator, context, candidate: CounterfactualCandidate) -> bool:
    candidate_frame = pd.DataFrame([candidate.profile], columns=context.bundle.feature_order)
    prediction = runtime.prediction_service.predict("bank", candidate_frame, context)
    return int(prediction.predicted_label) == int(context.policy.desired_outcome)


def compute_candidate_actionability(*, candidate: CounterfactualCandidate, active_constraint_spec: dict[str, Any], context) -> int:
    if any(feature not in context.policy.f2change for feature in candidate.changed_features):
        return 0
    filtered_result, _debug_summary = apply_constraint_spec_to_candidates(
        result=CounterfactualResult(feasible=True, candidates=[candidate], reason_codes=[]),
        constraint_spec=active_constraint_spec,
        feature_order=list(context.bundle.feature_order),
        sort_candidates=sort_counterfactual_candidates,
        request_constraints_blocked_code=REQUEST_CONSTRAINTS_BLOCKED_CODE,
    )
    return 1 if filtered_result.feasible and filtered_result.candidates else 0


def build_feature_ranges(dataset_df, feature_order: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for feature_name in feature_order:
        series = dataset_df[feature_name]
        ranges[feature_name] = (float(series.min()), float(series.max()))
    return ranges


def compute_normalized_proximity(
    *,
    factual_profile: dict[str, Any],
    counterfactual_profile: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
    feature_order: list[str],
) -> float:
    squared_sum = 0.0
    for feature_name in feature_order:
        factual_value = float(factual_profile[feature_name])
        counterfactual_value = float(counterfactual_profile[feature_name])
        minimum, maximum = feature_ranges[feature_name]
        span = maximum - minimum
        if span <= 0.0:
            span = 1.0
        scaled_delta = (counterfactual_value - factual_value) / span
        squared_sum += scaled_delta * scaled_delta
    return round(squared_sum ** 0.5, 6)


def build_method_table(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for method_name in METHODS:
        rows = [item["methods"][method_name] for item in seed_results]
        success_count = sum(int(row["success"]) for row in rows)
        raw_validity_count = sum(int(row["raw_validity"]) for row in rows)
        actionable_count = sum(1 for row in rows if int(row["actionable_label_flip_count"]) > 0)
        plausible_count = sum(1 for row in rows if int(row["invariant_valid_candidate_count"]) > 0)
        raw_flip_seed_count = sum(1 for row in rows if int(row["raw_label_flip_count"]) > 0)
        invariant_pass_seed_count = sum(1 for row in rows if int(row["invariant_valid_candidate_count"]) > 0)
        feasibility_count = sum(int(row["feasibility"]) for row in rows)
        failure_count = sum(int(row["failure"]) for row in rows)
        failure_reason_counts: dict[str, int] = {}
        failure_bucket_counts = {bucket: 0 for bucket in FAILURE_BUCKETS}
        for row in rows:
            if row["failure_bucket"] in failure_bucket_counts:
                failure_bucket_counts[row["failure_bucket"]] += 1
            for reason_code in row["failure_reason_codes"]:
                failure_reason_counts[reason_code] = failure_reason_counts.get(reason_code, 0) + 1
        failure_reason_counts = dict(sorted(failure_reason_counts.items()))
        successful_selection_count = success_count
        table[method_name] = {
            "validity": {
                "numerator": raw_validity_count,
                "denominator": len(rows),
                "mean": safe_mean(raw_validity_count, len(rows)),
            },
            "sparsity": {
                **summarize_numeric_values([row["sparsity"] for row in rows]),
                "denominator": successful_selection_count,
            },
            "proximity": {
                **summarize_numeric_values([row["proximity"] for row in rows]),
                "denominator": successful_selection_count,
            },
            "actionability": {
                "numerator": actionable_count,
                "denominator": len(rows),
                "mean": safe_mean(actionable_count, len(rows)),
            },
            "plausibility": {
                "numerator": plausible_count,
                "denominator": len(rows),
                "mean": safe_mean(plausible_count, len(rows)),
            },
            "feasibility": {
                "numerator": feasibility_count,
                "denominator": len(rows),
                "mean": safe_mean(feasibility_count, len(rows)),
            },
            "runtime_latency_ms": summarize_latency_ms([row["latency_ms"] for row in rows]),
            "success_rate": {
                "numerator": success_count,
                "denominator": len(rows),
                "mean": safe_mean(success_count, len(rows)),
            },
            "failure_rate": {
                "numerator": failure_count,
                "denominator": len(rows),
                "mean": safe_mean(failure_count, len(rows)),
            },
            "diagnostics": {
                "raw_flip_rate": {
                    "numerator": raw_flip_seed_count,
                    "denominator": len(rows),
                    "mean": safe_mean(raw_flip_seed_count, len(rows)),
                },
                "invariant_pass_rate": {
                    "numerator": invariant_pass_seed_count,
                    "denominator": len(rows),
                    "mean": safe_mean(invariant_pass_seed_count, len(rows)),
                },
                "final_selection_success_rate": {
                    "numerator": success_count,
                    "denominator": len(rows),
                    "mean": safe_mean(success_count, len(rows)),
                },
                "failure_bucket_counts": failure_bucket_counts,
                "failure_reason_counts": failure_reason_counts,
            },
        }
    return table


def build_failure_breakdown(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for method_name in METHODS:
        rows = [item["methods"][method_name] for item in seed_results]
        bucket_counts = {bucket: 0 for bucket in FAILURE_BUCKETS}
        for row in rows:
            bucket = row.get("failure_bucket")
            if bucket in bucket_counts:
                bucket_counts[bucket] += 1
        breakdown[method_name] = bucket_counts
    return breakdown


def classify_failure_bucket(
    *,
    raw_error: str | None,
    raw_candidate_count: int,
    raw_label_flip_count: int,
    invariant_valid_candidate_count: int,
    constraint_compliant_candidate_count: int,
    selected_candidate_emitted: int,
    active_constraint_spec: dict[str, Any] | None,
) -> str | None:
    del selected_candidate_emitted
    if raw_error is not None:
        return "timeout_or_error"
    if raw_candidate_count <= 0:
        return "no_candidate"
    if raw_label_flip_count <= 0:
        return "no_flip"
    if invariant_valid_candidate_count <= 0:
        return "invariant_fail"
    if isinstance(active_constraint_spec, dict) and active_constraint_spec and constraint_compliant_candidate_count <= 0:
        return "constraint_blocked"
    return None


def flatten_seed_results(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_result in seed_results:
        for method_name, method_payload in seed_result["methods"].items():
            rows.append(
                {
                    "seed_id": seed_result["seed_id"],
                    "dataset": seed_result["dataset"],
                    "active_constraint_spec": seed_result["active_constraint_spec"],
                    "method": method_name,
                    **method_payload,
                }
            )
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Backend Comparison Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- scorer_version: `{summary['scorer_version']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- corpus_version: `{summary['corpus_version']}`",
        f"- corpus_sha256: `{summary['corpus_sha256']}`",
        f"- catalog_version: `{summary['catalog_version']}`",
        f"- catalog_sha256: `{summary['catalog_sha256']}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- method_list: `{summary['method_list']}`",
        f"- single_case_mode: `{summary['single_case_mode']}`",
        "",
        "## Fairness Contract",
        "",
        f"- seed_selection_rule: `{summary['fairness_contract'].get('seed_selection_rule')}`",
        f"- selection_space: `{summary['fairness_contract'].get('selection_space')}`",
        f"- rejected_label: `{summary['fairness_contract'].get('rejected_label')}`",
        f"- rules: `{summary['fairness_contract']['rules']}`",
        f"- preprocessing_and_normalization: `{summary['fairness_contract']['preprocessing_and_normalization']}`",
        "",
        "## G4_baseline_unconstrained",
        "",
        "| Method | Validity | Actionability | Plausibility | Feasibility | Sparsity | Proximity | Runtime Latency | Failure Rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method_name in METHODS:
        row = summary[G4_BASELINE_UNCONSTRAINED][method_name]
        lines.append(
            "| `{method}` | `{validity}` | `{actionability}` | `{plausibility}` | `{feasibility}` | `{sparsity}` | `{proximity}` | `{latency}` | `{failure}` |".format(
                method=method_name,
                validity=row["validity"]["mean"],
                actionability=row["actionability"]["mean"],
                plausibility=row["plausibility"]["mean"],
                feasibility=row["feasibility"]["mean"],
                sparsity=row["sparsity"],
                proximity=row["proximity"],
                latency=row["runtime_latency_ms"],
                failure=row["failure_rate"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## G4_constraint_aware",
            "",
            f"- constrained_subset_seed_count: `{summary['seed_counts']['constrained_subset']}`",
            "",
            "| Method | Validity | Actionability | Plausibility | Feasibility | Sparsity | Proximity | Runtime Latency | Failure Rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method_name in METHODS:
        row = summary[G4_CONSTRAINT_AWARE][method_name]
        lines.append(
            "| `{method}` | `{validity}` | `{actionability}` | `{plausibility}` | `{feasibility}` | `{sparsity}` | `{proximity}` | `{latency}` | `{failure}` |".format(
                method=method_name,
                validity=row["validity"]["mean"],
                actionability=row["actionability"]["mean"],
                plausibility=row["plausibility"]["mean"],
                feasibility=row["feasibility"]["mean"],
                sparsity=row["sparsity"],
                proximity=row["proximity"],
                latency=row["runtime_latency_ms"],
                failure=row["failure_rate"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Failure Breakdown",
            "",
            "### G4_baseline_unconstrained",
            "",
            "| Method | no_candidate | no_flip | invariant_fail | constraint_blocked | timeout_or_error |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method_name in METHODS:
        row = summary["failure_breakdown"][G4_BASELINE_UNCONSTRAINED][method_name]
        lines.append(
            "| `{method}` | `{no_candidate}` | `{no_flip}` | `{invariant_fail}` | `{constraint_blocked}` | `{timeout_or_error}` |".format(
                method=method_name,
                no_candidate=row["no_candidate"],
                no_flip=row["no_flip"],
                invariant_fail=row["invariant_fail"],
                constraint_blocked=row["constraint_blocked"],
                timeout_or_error=row["timeout_or_error"],
            )
        )
    lines.extend(
        [
            "",
            "### G4_constraint_aware",
            "",
            "| Method | no_candidate | no_flip | invariant_fail | constraint_blocked | timeout_or_error |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method_name in METHODS:
        row = summary["failure_breakdown"][G4_CONSTRAINT_AWARE][method_name]
        lines.append(
            "| `{method}` | `{no_candidate}` | `{no_flip}` | `{invariant_fail}` | `{constraint_blocked}` | `{timeout_or_error}` |".format(
                method=method_name,
                no_candidate=row["no_candidate"],
                no_flip=row["no_flip"],
                invariant_fail=row["invariant_fail"],
                constraint_blocked=row["constraint_blocked"],
                timeout_or_error=row["timeout_or_error"],
            )
        )
    if summary.get("single_case_results") is not None:
        lines.extend(
            [
                "",
                "## Single Case",
                "",
                f"- selected_seed: `{summary['single_case_results']['selected_seed']}`",
                "| Method | Raw Candidates | Raw Flips | Actionable Flips | Invariant Valid | Selected Emitted | Failure Bucket | Sparsity | Proximity | Latency Ms | Reason Codes | First Failed Violations |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for method_name in METHODS:
            row = summary["single_case_results"]["main_unconstrained"]["methods"][method_name]
            lines.append(
                "| `{method}` | `{raw_count}` | `{raw_flips}` | `{actionable_flips}` | `{invariant_valid}` | `{selected}` | `{failure_bucket}` | `{sparsity}` | `{proximity}` | `{latency}` | `{reasons}` | `{violations}` |".format(
                    method=method_name,
                    raw_count=row["raw_candidate_count"],
                    raw_flips=row["raw_label_flip_count"],
                    actionable_flips=row["actionable_label_flip_count"],
                    invariant_valid=row["invariant_valid_candidate_count"],
                    selected=row["selected_candidate_emitted"],
                    failure_bucket=row["failure_bucket"],
                    sparsity=row["sparsity"],
                    proximity=row["proximity"],
                    latency=row["latency_ms"],
                    reasons=row["selection_reason_codes"],
                    violations=row["first_failed_invariant_violations"],
                )
            )
    lines.extend(
        [
            "",
            "## Aggregate Validation",
            "",
            f"- ok: `{summary['aggregate_validation']['ok']}`",
            f"- difference_count: `{summary['aggregate_validation']['difference_count']}`",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
