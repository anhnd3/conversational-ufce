#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import ufce_hypothesis_ablation as ablation


DEFAULT_AUDIT_OUT_DIR = "outputs/uf_final_audit"
DEFAULT_CONFIRM_DATASETS = ["bank", "grad", "movie"]
PHASE1_DIRNAME = "phase1_replay"
METHOD_SLICE_DIRNAME = "method_slice"
CONFIRM_DIRNAME = "confirm"
FINAL_REPORT_DIRNAME = "final_report"
MODE_CHOICES = ["phase1_replay", "method_slice", "confirm", "final_report", "all"]
PHASE1_REQUIRED_FILENAMES = [
    "master_summary.csv",
    "master_summary_sidecar.json",
    "master_ranked_findings.csv",
    "provenance.csv",
]
METHOD_SLICE_SCORE_COLUMNS = [
    "valid_success_score",
    "empty_rate_score",
    "feature_path_score",
    "metric_impact_score",
]
RANKING_FAMILIES = [
    ("valid_success", "valid_success_score", "top_valid_family", "top_valid_variant", "top_valid_score"),
    ("empty_rate", "empty_rate_score", "top_empty_family", "top_empty_variant", "top_empty_score"),
    ("feature_path", "feature_path_score", "top_feature_path_family", "top_feature_path_variant", "top_feature_path_score"),
    ("metric_impact", "metric_impact_score", "top_metric_family", "top_metric_variant", "top_metric_score"),
]
CONFIRM_STATUS_PRIORITY = {
    "matched_strict": 0,
    "matched_base": 1,
    "changed": 2,
    "unavailable": 3,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFCE final blind-spot audit wrapper.")
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--out_dir", default=DEFAULT_AUDIT_OUT_DIR)
    parser.add_argument("--source_out_dir", default=None)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--variant_filter", default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--data_dir", default=os.path.join("ufce", "data"))
    parser.add_argument("--fold_dir", default=os.path.join("ufce", "data", "folds"))
    parser.add_argument("--no_cf", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--contprox_metric", default="euclidean")
    return parser.parse_args(argv)


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_dataset_list(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    allowed = set(ablation.DEFAULT_DATASETS)
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Allowed: {sorted(allowed)}")
    return values


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_optional_csv(path: Path, *, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns) if columns is not None else None)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns) if columns is not None else None)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "flip_filter" in out.columns:
        out["flip_filter"] = pd.to_numeric(out["flip_filter"], errors="coerce").fillna(0).astype(int)
    return out


def _phase1_source_is_valid(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in PHASE1_REQUIRED_FILENAMES)


def _resolve_phase1_source(explicit_source: Optional[str], audit_root: Path) -> Path:
    if explicit_source:
        explicit_path = _resolve_repo_path(explicit_source)
        if not _phase1_source_is_valid(explicit_path):
            raise FileNotFoundError(
                f"Explicit --source_out_dir is not a valid Phase 1 artifact directory: {explicit_path}"
            )
        return explicit_path

    local_phase1 = audit_root / PHASE1_DIRNAME
    if _phase1_source_is_valid(local_phase1):
        return local_phase1

    outputs_root = ROOT / "outputs"
    candidates = []
    if outputs_root.exists():
        candidates = [
            path
            for path in outputs_root.glob("uf_ablation*")
            if _phase1_source_is_valid(path)
        ]
    if not candidates:
        raise FileNotFoundError(
            "Could not resolve a Phase 1 source directory. Checked --source_out_dir, "
            f"{local_phase1}, and outputs/uf_ablation*."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_datasets_from_phase1(source_dir: Path) -> list[str]:
    provenance_path = source_dir / "provenance.csv"
    if provenance_path.exists():
        provenance_df = _read_csv(provenance_path)
        if "dataset" in provenance_df.columns and not provenance_df.empty:
            return [str(value) for value in provenance_df["dataset"].tolist()]

    summary_df = _read_csv(source_dir / "master_summary.csv")
    if "dataset" in summary_df.columns and not summary_df.empty:
        return sorted({str(value) for value in summary_df["dataset"].tolist()})

    ranked_df = _read_csv(source_dir / "master_ranked_findings.csv")
    if "dataset" in ranked_df.columns and not ranked_df.empty:
        return sorted({str(value) for value in ranked_df["dataset"].tolist()})

    raise ValueError(f"Could not infer datasets from Phase 1 source: {source_dir}")


def _build_harness_argv(
    *,
    datasets: Sequence[str],
    out_dir: Path,
    stages: str,
    max_folds: Optional[int],
    variant_filter: Optional[str],
    data_dir: str,
    fold_dir: str,
    no_cf: int,
    seed: int,
    contprox_metric: str,
) -> list[str]:
    argv = [
        "--datasets",
        ",".join(datasets),
        "--out_dir",
        str(out_dir),
        "--stages",
        str(stages),
        "--data_dir",
        str(data_dir),
        "--fold_dir",
        str(fold_dir),
        "--no_cf",
        str(no_cf),
        "--seed",
        str(seed),
        "--contprox_metric",
        str(contprox_metric),
    ]
    if max_folds is not None:
        argv.extend(["--max_folds", str(max_folds)])
    if variant_filter:
        argv.extend(["--variant_filter", str(variant_filter)])
    return argv


def _run_phase1_replay(
    *,
    audit_root: Path,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    out_dir = _ensure_dir(audit_root / PHASE1_DIRNAME)
    argv = _build_harness_argv(
        datasets=datasets,
        out_dir=out_dir,
        stages="all",
        max_folds=args.max_folds,
        variant_filter=args.variant_filter,
        data_dir=args.data_dir,
        fold_dir=args.fold_dir,
        no_cf=args.no_cf,
        seed=args.seed,
        contprox_metric=args.contprox_metric,
    )
    print(
        "[UF-AUDIT][PHASE1-REPLAY] "
        f"datasets={list(datasets)} out_dir={out_dir}",
        flush=True,
    )
    ablation.main(argv)
    return out_dir


def _mean_abs(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(np.mean(np.abs(values.to_numpy(dtype=float))))


def _mean_value(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(np.mean(values.to_numpy(dtype=float)))


def _build_score_frame(
    summary_df: pd.DataFrame,
    *,
    group_keys: Sequence[str],
) -> pd.DataFrame:
    output_columns = list(group_keys) + METHOD_SLICE_SCORE_COLUMNS
    if summary_df.empty:
        return pd.DataFrame(columns=output_columns)

    working = summary_df.copy()
    grouped = (
        working.groupby(list(group_keys), dropna=False)
        .agg(
            valid_success_score=("delta_valid_rate_vs_public_baseline", _mean_abs),
            empty_rate_score=("delta_empty_rate_vs_public_baseline", _mean_abs),
            feature_path_score=("feature_path_score_vs_public_baseline", _mean_value),
            metric_impact_score=("delta_metric_score_vs_public_baseline", _mean_value),
        )
        .reset_index()
    )
    return grouped


def _group_key_debug_key(group_values: Any) -> str:
    if isinstance(group_values, tuple):
        return "|".join(str(value) for value in group_values)
    return str(group_values)


def _build_rankings_from_scores(
    score_df: pd.DataFrame,
    *,
    group_keys: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    base_columns = list(group_keys)
    ranking_columns = [
        "top_valid_family",
        "top_valid_variant",
        "top_valid_score",
        "top_empty_family",
        "top_empty_variant",
        "top_empty_score",
        "top_feature_path_family",
        "top_feature_path_variant",
        "top_feature_path_score",
        "top_metric_family",
        "top_metric_variant",
        "top_metric_score",
    ]
    if score_df.empty:
        return pd.DataFrame(columns=base_columns + ranking_columns), {}

    ranking_input_df = score_df[score_df["variant_family"].astype(str) != "anchor"].copy()
    if ranking_input_df.empty:
        return pd.DataFrame(columns=base_columns + ranking_columns), {}

    rows: list[dict[str, Any]] = []
    debug: dict[str, list[dict[str, Any]]] = {}
    grouped = ranking_input_df.groupby(list(group_keys), dropna=False)
    for group_value, group_df in grouped:
        if isinstance(group_value, tuple):
            out_row = {key: value for key, value in zip(group_keys, group_value)}
        else:
            out_row = {group_keys[0]: group_value}
        debug[_group_key_debug_key(group_value)] = group_df.to_dict(orient="records")

        for _, score_column, family_column, variant_column, score_output in RANKING_FAMILIES:
            ranked = group_df.sort_values(
                [score_column, "variant_family", "variant_name"],
                ascending=[False, True, True],
            ).reset_index(drop=True)
            top = ranked.iloc[0]
            out_row[family_column] = str(top["variant_family"])
            out_row[variant_column] = str(top["variant_name"])
            out_row[score_output] = float(top[score_column])
        rows.append(out_row)

    ranking_df = pd.DataFrame(rows)
    sort_keys = [key for key in group_keys if key in ranking_df.columns]
    if sort_keys:
        ranking_df = ranking_df.sort_values(sort_keys).reset_index(drop=True)
    return ranking_df, debug


def _nested_ranking_debug(
    score_df: pd.DataFrame,
    *,
    outer_keys: Sequence[str],
    inner_keys: Sequence[str],
) -> dict[str, Any]:
    debug: dict[str, Any] = defaultdict(dict)
    if score_df.empty:
        return {}
    for outer_value, outer_df in score_df.groupby(list(outer_keys), dropna=False):
        outer_key = _group_key_debug_key(outer_value)
        for inner_value, inner_df in outer_df.groupby(list(inner_keys), dropna=False):
            inner_key = _group_key_debug_key(inner_value)
            debug[outer_key][inner_key] = inner_df.to_dict(orient="records")
    return dict(debug)


def _load_phase1_artifacts(source_dir: Path) -> dict[str, Any]:
    return {
        "source_dir": str(source_dir),
        "master_summary_path": str(source_dir / "master_summary.csv"),
        "master_ranked_findings_path": str(source_dir / "master_ranked_findings.csv"),
        "master_summary_sidecar_path": str(source_dir / "master_summary_sidecar.json"),
        "provenance_path": str(source_dir / "provenance.csv"),
        "skipped_variants_path": str(source_dir / "skipped_variants.csv"),
    }


def _run_method_slice(
    *,
    source_dir: Path,
    audit_root: Path,
    datasets: Sequence[str],
) -> Path:
    out_dir = _ensure_dir(audit_root / METHOD_SLICE_DIRNAME)
    summary_df = _normalize_summary_df(_read_csv(source_dir / "master_summary.csv"))
    summary_df = summary_df[summary_df["dataset"].isin(list(datasets))].copy()
    score_input_df = summary_df[summary_df["flip_filter"] == 0].copy()

    method_slice_summary_df = _build_score_frame(
        score_input_df,
        group_keys=[
            "dataset",
            "method",
            "variant_name",
            "variant_family",
            "uf_mode",
            "step_mode",
            "f2change_mode",
        ],
    )
    method_slice_rankings_df, ranking_debug = _build_rankings_from_scores(
        method_slice_summary_df,
        group_keys=["dataset", "method"],
    )

    summary_path = out_dir / "method_slice_summary.csv"
    rankings_path = out_dir / "method_slice_rankings.csv"
    sidecar_path = out_dir / "method_slice_sidecar.json"
    method_slice_summary_df.to_csv(summary_path, index=False)
    method_slice_rankings_df.to_csv(rankings_path, index=False)

    skipped_variants_df = _read_optional_csv(
        source_dir / "skipped_variants.csv",
        columns=ablation.SKIPPED_VARIANT_COLUMNS,
    )
    skipped_variants_df = skipped_variants_df[
        skipped_variants_df["dataset"].isin(list(datasets))
    ].copy()

    sidecar = {
        "phase1_source": _load_phase1_artifacts(source_dir),
        "datasets": list(datasets),
        "summary_path": str(summary_path),
        "rankings_path": str(rankings_path),
        "source_sidecar": _read_json(source_dir / "master_summary_sidecar.json"),
        "skipped_variants": skipped_variants_df.to_dict(orient="records"),
        "ranking_debug": _nested_ranking_debug(
            method_slice_summary_df,
            outer_keys=["dataset"],
            inner_keys=["method"],
        ),
    }
    _write_json(sidecar_path, sidecar)
    print(
        "[UF-AUDIT][METHOD-SLICE] "
        f"datasets={list(datasets)} rows={len(method_slice_summary_df)} out_dir={out_dir}",
        flush=True,
    )
    return out_dir


def _parse_variant_filter(raw: Optional[str]) -> set[str]:
    if raw is None:
        return set()
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def _summary_row_passes_variant_filter(row: pd.Series, allowed: set[str]) -> bool:
    if not allowed:
        return True
    variant_family = str(row["variant_family"])
    if variant_family == "anchor":
        return True
    if str(row["variant_name"]) in allowed:
        return True
    for token in allowed:
        if token in ablation.UF_MODE_NAMES:
            if variant_family == "uf" and str(row["uf_mode"]) == token:
                return True
            continue
        if token in ablation.STEP_MODE_NAMES:
            if variant_family == "step" and str(row["step_mode"]) == token:
                return True
            continue
        if token in ablation.F2CHANGE_MODE_NAMES:
            if variant_family == "f2change" and str(row["f2change_mode"]) == token:
                return True
            continue
    return False


def _select_confirm_variants(
    *,
    phase1_summary_df: pd.DataFrame,
    datasets: Sequence[str],
    variant_filter: Optional[str],
) -> dict[str, dict[str, Any]]:
    allowed = _parse_variant_filter(variant_filter)
    working = phase1_summary_df.copy()
    working = working[
        (working["dataset"].isin(list(datasets)))
        & (working["flip_filter"] == 0)
        & (working["variant_family"] != "anchor")
    ].copy()
    if allowed:
        working = working[working.apply(lambda row: _summary_row_passes_variant_filter(row, allowed), axis=1)].copy()

    score_df = _build_score_frame(
        working,
        group_keys=[
            "dataset",
            "variant_name",
            "variant_family",
            "uf_mode",
            "step_mode",
            "f2change_mode",
        ],
    )
    selections: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        dataset_scores = score_df[score_df["dataset"] == dataset].copy()
        ranked = dataset_scores.sort_values(
            ["valid_success_score", "empty_rate_score", "feature_path_score", "variant_name"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        if len(ranked) < 2:
            raise ValueError(
                f"Confirm selection for dataset={dataset} requires at least two non-anchor variants after filtering. "
                f"Found {len(ranked)} candidate(s)."
            )
        top_two = ranked.head(2).copy()
        selections[dataset] = {
            "dataset": dataset,
            "anchors": [ablation.ANCHOR_PUBLIC_NAME, ablation.ANCHOR_AUTHOR_NAME],
            "selected_variant_names": top_two["variant_name"].tolist(),
            "candidate_rows": ranked.to_dict(orient="records"),
            "selected_rows": top_two.to_dict(orient="records"),
        }
    return selections


def _merge_confirm_query_traces(raw_root: Path, datasets: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        trace_path = raw_root / dataset / dataset / "query_trace.jsonl"
        if not trace_path.exists():
            continue
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
    return rows


def _run_confirm(
    *,
    source_dir: Path,
    audit_root: Path,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    out_dir = _ensure_dir(audit_root / CONFIRM_DIRNAME)
    raw_root = _ensure_dir(out_dir / "raw")
    phase1_summary_df = _normalize_summary_df(_read_csv(source_dir / "master_summary.csv"))
    selections = _select_confirm_variants(
        phase1_summary_df=phase1_summary_df,
        datasets=datasets,
        variant_filter=args.variant_filter,
    )

    raw_run_dirs: dict[str, str] = {}
    for dataset in datasets:
        raw_dataset_root = _ensure_dir(raw_root / dataset)
        selected_variant_filter = ",".join(selections[dataset]["selected_variant_names"])
        argv = _build_harness_argv(
            datasets=[dataset],
            out_dir=raw_dataset_root,
            stages="strict",
            max_folds=args.max_folds,
            variant_filter=selected_variant_filter,
            data_dir=args.data_dir,
            fold_dir=args.fold_dir,
            no_cf=args.no_cf,
            seed=args.seed,
            contprox_metric=args.contprox_metric,
        )
        print(
            "[UF-AUDIT][CONFIRM-DATASET] "
            f"dataset={dataset} selected={selections[dataset]['selected_variant_names']} out_dir={raw_dataset_root}",
            flush=True,
        )
        ablation.main(argv)
        raw_run_dirs[dataset] = str(raw_dataset_root)

    summary_frames = []
    skipped_frames = []
    for dataset in datasets:
        raw_dataset_root = raw_root / dataset
        summary_frames.append(_normalize_summary_df(_read_csv(raw_dataset_root / "master_summary.csv")))
        skipped_frames.append(
            _read_optional_csv(
                raw_dataset_root / "skipped_variants.csv",
                columns=ablation.SKIPPED_VARIANT_COLUMNS,
            )
        )

    confirm_summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    confirm_score_df = _build_score_frame(
        confirm_summary_df[confirm_summary_df["variant_family"] != "anchor"].copy(),
        group_keys=["dataset", "flip_filter", "variant_name", "variant_family"],
    )
    confirm_rankings_df, ranking_debug = _build_rankings_from_scores(
        confirm_score_df,
        group_keys=["dataset", "flip_filter"],
    )
    confirm_trace_rows = _merge_confirm_query_traces(raw_root, datasets)
    confirm_skipped_df = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(columns=ablation.SKIPPED_VARIANT_COLUMNS)

    summary_path = out_dir / "confirm_summary.csv"
    rankings_path = out_dir / "confirm_rankings.csv"
    trace_path = out_dir / "confirm_query_trace.jsonl"
    sidecar_path = out_dir / "confirm_sidecar.json"
    confirm_summary_df.to_csv(summary_path, index=False)
    confirm_rankings_df.to_csv(rankings_path, index=False)
    _write_jsonl(trace_path, confirm_trace_rows)

    sidecar = {
        "phase1_source": _load_phase1_artifacts(source_dir),
        "datasets": list(datasets),
        "variant_filter_tokens": sorted(_parse_variant_filter(args.variant_filter)),
        "selected_variants": selections,
        "raw_run_dirs": raw_run_dirs,
        "ranking_debug": ranking_debug,
        "skipped_variants": confirm_skipped_df.to_dict(orient="records"),
        "summary_path": str(summary_path),
        "rankings_path": str(rankings_path),
        "query_trace_path": str(trace_path),
    }
    _write_json(sidecar_path, sidecar)
    print(
        "[UF-AUDIT][CONFIRM] "
        f"datasets={list(datasets)} summary_rows={len(confirm_summary_df)} rankings_rows={len(confirm_rankings_df)}",
        flush=True,
    )
    return out_dir


def _find_row(df: pd.DataFrame, **criteria: Any) -> Optional[pd.Series]:
    if df.empty:
        return None
    mask = np.ones(len(df), dtype=bool)
    for key, value in criteria.items():
        if key not in df.columns:
            return None
        mask &= df[key].astype(object).values == value
    subset = df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _select_method_slice_top_rows(method_slice_rankings_df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if method_slice_rankings_df.empty:
        return {}
    for dataset, dataset_df in method_slice_rankings_df.groupby("dataset"):
        for finding_family, _, family_col, variant_col, score_col in RANKING_FAMILIES:
            ranked = dataset_df.sort_values(
                [score_col, "method", family_col, variant_col],
                ascending=[False, True, True, True],
            ).reset_index(drop=True)
            if ranked.empty:
                continue
            top = ranked.iloc[0]
            out[str(dataset)][finding_family] = {
                "method": str(top["method"]),
                "variant_family": str(top[family_col]),
                "variant_name": str(top[variant_col]),
                "score": float(top[score_col]),
            }
    return dict(out)


def _confirmation_status(
    *,
    phase1_family: str,
    phase1_variant: str,
    confirm_base: Optional[dict[str, Any]],
    confirm_strict: Optional[dict[str, Any]],
) -> str:
    if confirm_strict is not None:
        if (
            str(confirm_strict["variant_family"]) == phase1_family
            and str(confirm_strict["variant_name"]) == phase1_variant
        ):
            return "matched_strict"
    if confirm_base is not None:
        if (
            str(confirm_base["variant_family"]) == phase1_family
            and str(confirm_base["variant_name"]) == phase1_variant
        ):
            return "matched_base"
    if confirm_base is not None or confirm_strict is not None:
        return "changed"
    return "unavailable"


def _build_notes_tag(
    *,
    confirmation_status: str,
    phase1_variant: str,
    method_slice_choice: Optional[dict[str, Any]],
    has_phase1_skips: bool,
    has_confirm_skips: bool,
) -> str:
    tags = [confirmation_status]
    if method_slice_choice is not None and str(method_slice_choice["variant_name"]) != str(phase1_variant):
        tags.append("method_slice_differs")
    if has_phase1_skips:
        tags.append("phase1_skips")
    if has_confirm_skips:
        tags.append("confirm_skips")
    return ";".join(tags)


def _render_final_notes(
    *,
    audit_root: Path,
    phase1_source: Path,
    datasets: Sequence[str],
    final_summary_df: pd.DataFrame,
    phase1_rankings_df: pd.DataFrame,
    method_slice_top_rows: dict[str, dict[str, dict[str, Any]]],
    confirm_rankings_df: pd.DataFrame,
    phase1_skipped_df: pd.DataFrame,
    confirm_skipped_df: pd.DataFrame,
) -> str:
    lines = [
        "# UFCE Final Blind-Spot Audit",
        "",
        "## Provenance",
        "",
        f"- generated_at_local: `{datetime.now().astimezone().isoformat()}`",
        f"- audit_root: `{audit_root}`",
        f"- phase1_source_dir: `{phase1_source}`",
        f"- datasets: `{','.join(datasets)}`",
        "",
        "## Phase 1 Aggregate Findings",
        "",
    ]
    if phase1_rankings_df.empty:
        lines.append("- No Phase 1 aggregate findings were available.")
    else:
        for dataset in datasets:
            row = _find_row(phase1_rankings_df, dataset=dataset)
            if row is None:
                lines.append(f"- {dataset}: no Phase 1 ranking row found.")
                continue
            parts = []
            for finding_family, _, family_col, variant_col, score_col in RANKING_FAMILIES:
                parts.append(
                    f"{finding_family}=`{row[family_col]}/{row[variant_col]}` score=`{float(row[score_col]):.6f}`"
                )
            lines.append(f"- {dataset}: " + "; ".join(parts))
    lines.extend(["", "## Method-Sliced Clarification", ""])
    if not method_slice_top_rows:
        lines.append("- No method-sliced rankings were available.")
    else:
        for dataset in datasets:
            dataset_choices = method_slice_top_rows.get(dataset, {})
            if not dataset_choices:
                lines.append(f"- {dataset}: no method-sliced rows found.")
                continue
            parts = []
            for finding_family, *_rest in RANKING_FAMILIES:
                choice = dataset_choices.get(finding_family)
                if choice is None:
                    continue
                parts.append(
                    f"{finding_family}=`{choice['method']}` -> `{choice['variant_family']}/{choice['variant_name']}`"
                )
            lines.append(f"- {dataset}: " + "; ".join(parts))
    lines.extend(["", "## Confirmation on Bank / Grad / Movie", ""])
    if final_summary_df.empty:
        lines.append("- No final confirmation summary rows were generated.")
    else:
        for dataset in datasets:
            dataset_rows = final_summary_df[final_summary_df["dataset"] == dataset].copy()
            if dataset_rows.empty:
                lines.append(f"- {dataset}: no final summary rows found.")
                continue
            parts = []
            for row in dataset_rows.to_dict(orient="records"):
                parts.append(
                    f"{row['finding_family']}=`{row['confirmation_status']}` "
                    f"phase1=`{row['phase1_top_family']}/{row['phase1_top_variant']}` "
                    f"base=`{row['confirm_base_top_family']}/{row['confirm_base_top_variant']}` "
                    f"strict=`{row['confirm_strict_top_family']}/{row['confirm_strict_top_variant']}`"
                )
            lines.append(f"- {dataset}: " + "; ".join(parts))
    lines.extend(["", "## Caveats and Skips", ""])
    lines.append(f"- phase1_skipped_variant_count: `{int(len(phase1_skipped_df))}`")
    lines.append(f"- confirm_skipped_variant_count: `{int(len(confirm_skipped_df))}`")
    if not confirm_skipped_df.empty:
        movie_author_skip = confirm_skipped_df[
            (confirm_skipped_df["dataset"].astype(str) == "movie")
            & (confirm_skipped_df["variant_name"].astype(str) == ablation.ANCHOR_AUTHOR_NAME)
        ]
        if not movie_author_skip.empty:
            lines.append(
                "- `movie` keeps `author_preset_bundle` as a skip in confirm because the author preset step still "
                "omits `Budget` under the current harness semantics."
            )
    if phase1_skipped_df.empty and confirm_skipped_df.empty:
        lines.append("- No skipped variants were recorded.")
    return "\n".join(lines)


def _run_final_report(
    *,
    audit_root: Path,
    phase1_source: Path,
    datasets: Sequence[str],
) -> Path:
    out_dir = _ensure_dir(audit_root / FINAL_REPORT_DIRNAME)
    method_slice_dir = audit_root / METHOD_SLICE_DIRNAME
    confirm_dir = audit_root / CONFIRM_DIRNAME

    phase1_rankings_df = _read_csv(phase1_source / "master_ranked_findings.csv")
    phase1_rankings_df = phase1_rankings_df[phase1_rankings_df["dataset"].isin(list(datasets))].copy()
    method_slice_rankings_df = _read_csv(method_slice_dir / "method_slice_rankings.csv")
    method_slice_rankings_df = method_slice_rankings_df[method_slice_rankings_df["dataset"].isin(list(datasets))].copy()
    confirm_rankings_df = _normalize_summary_df(_read_csv(confirm_dir / "confirm_rankings.csv"))
    confirm_rankings_df = confirm_rankings_df[confirm_rankings_df["dataset"].isin(list(datasets))].copy()
    provenance_df = _read_csv(phase1_source / "provenance.csv")
    provenance_df = provenance_df[provenance_df["dataset"].isin(list(datasets))].copy()
    phase1_skipped_df = _read_optional_csv(
        phase1_source / "skipped_variants.csv",
        columns=ablation.SKIPPED_VARIANT_COLUMNS,
    )
    phase1_skipped_df = phase1_skipped_df[phase1_skipped_df["dataset"].isin(list(datasets))].copy()

    confirm_sidecar_path = confirm_dir / "confirm_sidecar.json"
    confirm_sidecar = _read_json(confirm_sidecar_path) if confirm_sidecar_path.exists() else {}
    confirm_skipped_df = pd.DataFrame(confirm_sidecar.get("skipped_variants", []))
    if not confirm_skipped_df.empty:
        confirm_skipped_df = confirm_skipped_df[confirm_skipped_df["dataset"].isin(list(datasets))].copy()

    method_slice_top_rows = _select_method_slice_top_rows(method_slice_rankings_df)
    final_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        phase1_row = _find_row(phase1_rankings_df, dataset=dataset)
        confirm_base_row = _find_row(confirm_rankings_df, dataset=dataset, flip_filter=0)
        confirm_strict_row = _find_row(confirm_rankings_df, dataset=dataset, flip_filter=1)
        dataset_has_phase1_skips = not phase1_skipped_df[phase1_skipped_df["dataset"] == dataset].empty
        dataset_has_confirm_skips = (
            not confirm_skipped_df.empty
            and not confirm_skipped_df[confirm_skipped_df["dataset"] == dataset].empty
        )
        for finding_family, _, family_col, variant_col, score_col in RANKING_FAMILIES:
            if phase1_row is None:
                continue
            phase1_family = str(phase1_row[family_col])
            phase1_variant = str(phase1_row[variant_col])
            phase1_score = float(phase1_row[score_col])
            method_slice_choice = method_slice_top_rows.get(dataset, {}).get(finding_family)

            confirm_base_choice = None
            if confirm_base_row is not None:
                confirm_base_choice = {
                    "variant_family": str(confirm_base_row[family_col]),
                    "variant_name": str(confirm_base_row[variant_col]),
                    "score": float(confirm_base_row[score_col]),
                }
            confirm_strict_choice = None
            if confirm_strict_row is not None:
                confirm_strict_choice = {
                    "variant_family": str(confirm_strict_row[family_col]),
                    "variant_name": str(confirm_strict_row[variant_col]),
                    "score": float(confirm_strict_row[score_col]),
                }

            confirmation_status = _confirmation_status(
                phase1_family=phase1_family,
                phase1_variant=phase1_variant,
                confirm_base=confirm_base_choice,
                confirm_strict=confirm_strict_choice,
            )
            notes_tag = _build_notes_tag(
                confirmation_status=confirmation_status,
                phase1_variant=phase1_variant,
                method_slice_choice=method_slice_choice,
                has_phase1_skips=dataset_has_phase1_skips,
                has_confirm_skips=dataset_has_confirm_skips,
            )
            final_rows.append(
                {
                    "dataset": dataset,
                    "finding_family": finding_family,
                    "phase1_top_family": phase1_family,
                    "phase1_top_variant": phase1_variant,
                    "phase1_top_score": phase1_score,
                    "method_slice_top_method": "" if method_slice_choice is None else str(method_slice_choice["method"]),
                    "method_slice_top_family": "" if method_slice_choice is None else str(method_slice_choice["variant_family"]),
                    "method_slice_top_variant": "" if method_slice_choice is None else str(method_slice_choice["variant_name"]),
                    "method_slice_top_score": np.nan if method_slice_choice is None else float(method_slice_choice["score"]),
                    "confirm_base_top_family": "" if confirm_base_choice is None else str(confirm_base_choice["variant_family"]),
                    "confirm_base_top_variant": "" if confirm_base_choice is None else str(confirm_base_choice["variant_name"]),
                    "confirm_base_top_score": np.nan if confirm_base_choice is None else float(confirm_base_choice["score"]),
                    "confirm_strict_top_family": "" if confirm_strict_choice is None else str(confirm_strict_choice["variant_family"]),
                    "confirm_strict_top_variant": "" if confirm_strict_choice is None else str(confirm_strict_choice["variant_name"]),
                    "confirm_strict_top_score": np.nan if confirm_strict_choice is None else float(confirm_strict_choice["score"]),
                    "confirmation_status": confirmation_status,
                    "notes_tag": notes_tag,
                }
            )

    final_summary_df = pd.DataFrame(final_rows)
    if not final_summary_df.empty:
        final_summary_df["status_order"] = final_summary_df["confirmation_status"].map(CONFIRM_STATUS_PRIORITY).fillna(999).astype(int)
        final_ranked_df = final_summary_df.sort_values(
            ["status_order", "phase1_top_score", "dataset", "finding_family"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
        final_ranked_df = final_ranked_df.drop(columns=["status_order"])
        final_summary_df = final_summary_df.drop(columns=["status_order"])
        final_summary_df = final_summary_df.sort_values(["dataset", "finding_family"]).reset_index(drop=True)
    else:
        final_ranked_df = final_summary_df.copy()

    summary_path = out_dir / "final_blindspot_summary.csv"
    ranked_path = out_dir / "final_blindspot_ranked_findings.csv"
    sidecar_path = out_dir / "final_blindspot_sidecar.json"
    notes_path = out_dir / "final_blindspot_notes.md"
    final_summary_df.to_csv(summary_path, index=False)
    final_ranked_df.to_csv(ranked_path, index=False)

    sidecar = {
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "audit_root": str(audit_root),
        "phase1_source": _load_phase1_artifacts(phase1_source),
        "method_slice_dir": str(method_slice_dir),
        "confirm_dir": str(confirm_dir),
        "datasets": list(datasets),
        "method_slice_top_rows": method_slice_top_rows,
        "confirm_selected_variants": confirm_sidecar.get("selected_variants", {}),
        "phase1_skipped_variants": phase1_skipped_df.to_dict(orient="records"),
        "confirm_skipped_variants": confirm_skipped_df.to_dict(orient="records"),
        "provenance": provenance_df.to_dict(orient="records"),
        "paths": {
            "summary_path": str(summary_path),
            "ranked_path": str(ranked_path),
            "notes_path": str(notes_path),
        },
    }
    _write_json(sidecar_path, sidecar)

    notes_text = _render_final_notes(
        audit_root=audit_root,
        phase1_source=phase1_source,
        datasets=datasets,
        final_summary_df=final_summary_df,
        phase1_rankings_df=phase1_rankings_df,
        method_slice_top_rows=method_slice_top_rows,
        confirm_rankings_df=confirm_rankings_df,
        phase1_skipped_df=phase1_skipped_df,
        confirm_skipped_df=confirm_skipped_df,
    )
    notes_path.write_text(notes_text, encoding="utf-8")
    print(
        "[UF-AUDIT][FINAL-REPORT] "
        f"datasets={list(datasets)} summary_rows={len(final_summary_df)} out_dir={out_dir}",
        flush=True,
    )
    return out_dir


def _resolve_mode_datasets(
    *,
    mode: str,
    args: argparse.Namespace,
    phase1_source: Optional[Path],
) -> list[str]:
    explicit = _parse_dataset_list(args.datasets)
    if explicit:
        return explicit
    if mode in {"phase1_replay", "all"}:
        return list(ablation.DEFAULT_DATASETS)
    if mode == "confirm":
        return list(DEFAULT_CONFIRM_DATASETS)
    if phase1_source is None:
        raise ValueError(f"Dataset inference for mode={mode} requires a resolved Phase 1 source.")
    return _infer_datasets_from_phase1(phase1_source)


def _run_all_modes(args: argparse.Namespace, audit_root: Path) -> None:
    phase1_datasets = _resolve_mode_datasets(mode="phase1_replay", args=args, phase1_source=None)
    phase1_source = _run_phase1_replay(audit_root=audit_root, datasets=phase1_datasets, args=args)
    method_slice_datasets = _resolve_mode_datasets(mode="method_slice", args=args, phase1_source=phase1_source)
    _run_method_slice(source_dir=phase1_source, audit_root=audit_root, datasets=method_slice_datasets)
    confirm_datasets = _resolve_mode_datasets(mode="confirm", args=args, phase1_source=phase1_source)
    _run_confirm(source_dir=phase1_source, audit_root=audit_root, datasets=confirm_datasets, args=args)
    final_datasets = _resolve_mode_datasets(mode="final_report", args=args, phase1_source=phase1_source)
    _run_final_report(audit_root=audit_root, phase1_source=phase1_source, datasets=final_datasets)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    audit_root = _ensure_dir(_resolve_repo_path(args.out_dir))

    if args.mode == "all":
        _run_all_modes(args, audit_root)
        return 0

    phase1_source: Optional[Path] = None
    if args.mode in {"method_slice", "confirm", "final_report"}:
        phase1_source = _resolve_phase1_source(args.source_out_dir, audit_root)

    if args.mode == "phase1_replay":
        datasets = _resolve_mode_datasets(mode="phase1_replay", args=args, phase1_source=None)
        _run_phase1_replay(audit_root=audit_root, datasets=datasets, args=args)
        return 0

    if args.mode == "method_slice":
        assert phase1_source is not None
        datasets = _resolve_mode_datasets(mode="method_slice", args=args, phase1_source=phase1_source)
        _run_method_slice(source_dir=phase1_source, audit_root=audit_root, datasets=datasets)
        return 0

    if args.mode == "confirm":
        assert phase1_source is not None
        datasets = _resolve_mode_datasets(mode="confirm", args=args, phase1_source=phase1_source)
        _run_confirm(source_dir=phase1_source, audit_root=audit_root, datasets=datasets, args=args)
        return 0

    if args.mode == "final_report":
        if args.rerun:
            if args.source_out_dir:
                phase1_source = _resolve_phase1_source(args.source_out_dir, audit_root)
            else:
                phase1_datasets = _resolve_mode_datasets(mode="phase1_replay", args=args, phase1_source=None)
                phase1_source = _run_phase1_replay(audit_root=audit_root, datasets=phase1_datasets, args=args)
            method_slice_datasets = _resolve_mode_datasets(mode="method_slice", args=args, phase1_source=phase1_source)
            _run_method_slice(source_dir=phase1_source, audit_root=audit_root, datasets=method_slice_datasets)
            confirm_datasets = _resolve_mode_datasets(mode="confirm", args=args, phase1_source=phase1_source)
            _run_confirm(source_dir=phase1_source, audit_root=audit_root, datasets=confirm_datasets, args=args)
        else:
            assert phase1_source is not None
            method_slice_dir = audit_root / METHOD_SLICE_DIRNAME
            confirm_dir = audit_root / CONFIRM_DIRNAME
            if not (method_slice_dir / "method_slice_rankings.csv").exists():
                raise FileNotFoundError(
                    f"Missing method-slice artifacts under {method_slice_dir}. Run --mode method_slice first or use --rerun."
                )
            if not (confirm_dir / "confirm_rankings.csv").exists():
                raise FileNotFoundError(
                    f"Missing confirm artifacts under {confirm_dir}. Run --mode confirm first or use --rerun."
                )
        assert phase1_source is not None
        datasets = _resolve_mode_datasets(mode="final_report", args=args, phase1_source=phase1_source)
        _run_final_report(audit_root=audit_root, phase1_source=phase1_source, datasets=datasets)
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
