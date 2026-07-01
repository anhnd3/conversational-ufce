#!/usr/bin/env python3
"""Red Wine Prox-Euc sensitivity diagnostic.

This script consumes row-level factual/counterfactual pairs from
``author_pool_selector_audit.py --emit-pairs`` and recomputes Red Wine
Euclidean proximity in several distance spaces.  If no pair CSV is provided, it
generates the thesis/slide Red Wine final-freeze pairs first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_PARENT = ROOT / "outputs" / "final" / "part1"
PAIR_SCRIPT = ROOT / "scripts" / "final" / "part1" / "author_pool_selector_audit.py"
BASELINE_DIR = OUT_PARENT / "author_pool_finalfreeze_20260624"
WINE_METADATA = ROOT / "llm" / "models" / "wine" / "metadata.json"
WINE_DATA = ROOT / "ufce" / "data" / "wine.csv"

METHODS = ["UFCE1", "UFCE2", "UFCE3"]
OUTPUT_SETS = ["author_raw", "author_posthoc_valid_only", "author_pool_forceflip"]
SPACES = ["model_raw", "original_raw", "minmax_0_1", "changed_features_original"]
FORBIDDEN_FEATURE_NAMES = {"label", "target", "id", "quality", "sample_id", "index", "unnamed: 0"}

PUBLISHED_ORIGINAL = {
    "UFCE1": 14.90,
    "UFCE2": 8.45,
    "UFCE3": 21.95,
}

FALLBACK_THESIS_BASELINE = {
    ("UFCE1", "author_raw"): 8.886122871497872,
    ("UFCE1", "author_posthoc_valid_only"): 8.886122871497872,
    ("UFCE1", "author_pool_forceflip"): 8.886122871497872,
    ("UFCE2", "author_raw"): 0.42507803779142295,
    ("UFCE2", "author_posthoc_valid_only"): 0.5215803156582155,
    ("UFCE2", "author_pool_forceflip"): 1.2389910494162726,
    ("UFCE3", "author_raw"): 1.0602902475305833,
    ("UFCE3", "author_posthoc_valid_only"): 0.9679451217384869,
    ("UFCE3", "author_pool_forceflip"): 2.072404449828204,
}


@dataclass(frozen=True)
class MinMaxSpec:
    minimum: pd.Series
    span: pd.Series
    constant_cols: Tuple[str, ...]


def now_tag() -> str:
    return datetime.now().strftime("red_wine_proximity_sensitivity_%Y%m%d_%H%M%S")


def finite_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    return float(np.linalg.norm(left - right))


def load_feature_order(metadata_path: Path = WINE_METADATA) -> Tuple[List[str], Dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_order = [str(feature) for feature in metadata["feature_order"]]
    return feature_order, metadata


def parse_pair_feature_columns(pair_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    x_features = [col.split("::", 1)[1] for col in pair_df.columns if str(col).startswith("x::")]
    cf_features = [col.split("::", 1)[1] for col in pair_df.columns if str(col).startswith("cf::")]
    return x_features, cf_features


def assert_no_forbidden_features(features: Iterable[str]) -> None:
    forbidden = [feature for feature in features if str(feature).strip().lower() in FORBIDDEN_FEATURE_NAMES]
    if forbidden:
        raise ValueError(f"Forbidden non-feature column(s) in feature set: {forbidden}")


def validate_feature_alignment(pair_df: pd.DataFrame, feature_order: Sequence[str]) -> None:
    expected = list(feature_order)
    x_features, cf_features = parse_pair_feature_columns(pair_df)
    assert_no_forbidden_features(expected)
    assert_no_forbidden_features(x_features)
    assert_no_forbidden_features(cf_features)
    if x_features != expected:
        raise ValueError(f"x feature order mismatch: expected={expected} actual={x_features}")
    if cf_features != expected:
        raise ValueError(f"cf feature order mismatch: expected={expected} actual={cf_features}")
    for prefix in ("x::", "cf::"):
        cols = [f"{prefix}{feature}" for feature in expected]
        for col in cols:
            pd.to_numeric(pair_df[col], errors="raise")


def load_training_frame(metadata: Mapping[str, Any], feature_order: Sequence[str]) -> pd.DataFrame:
    data = pd.read_csv(WINE_DATA)
    dropped = [col for col in metadata.get("dropped_columns", []) if col in data.columns]
    if dropped:
        data = data.drop(columns=dropped)
    train_indices = [int(idx) for idx in metadata.get("train_indices", [])]
    if not train_indices:
        raise ValueError(f"No train_indices found in {WINE_METADATA}")
    missing = [feature for feature in feature_order if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing Red Wine feature(s) in source data: {missing}")
    return data.iloc[train_indices].loc[:, list(feature_order)].reset_index(drop=True)


def fit_minmax_0_1(train_df: pd.DataFrame, feature_order: Sequence[str]) -> MinMaxSpec:
    base = train_df.loc[:, list(feature_order)].astype(float)
    minimum = base.min(axis=0)
    raw_span = base.max(axis=0) - minimum
    constant_cols = tuple(str(col) for col in raw_span.index[raw_span == 0.0])
    span = raw_span.mask(raw_span == 0.0, 1.0)
    return MinMaxSpec(minimum=minimum, span=span, constant_cols=constant_cols)


def transform_minmax_0_1(frame: pd.DataFrame, spec: MinMaxSpec, feature_order: Sequence[str]) -> pd.DataFrame:
    out = (frame.loc[:, list(feature_order)].astype(float) - spec.minimum.reindex(feature_order)) / spec.span.reindex(feature_order)
    out = out.clip(lower=0.0, upper=1.0)
    for col in spec.constant_cols:
        if col in out.columns:
            out.loc[:, col] = 0.0
    return out


def changed_feature_names(x: pd.Series, cf: pd.Series, feature_order: Sequence[str]) -> List[str]:
    changed: List[str] = []
    for feature in feature_order:
        if not np.isclose(float(x[feature]), float(cf[feature]), rtol=1e-09, atol=1e-12):
            changed.append(str(feature))
    return changed


def distance_for_space(
    x: pd.Series,
    cf: pd.Series,
    feature_order: Sequence[str],
    space: str,
    minmax_spec: MinMaxSpec,
) -> Tuple[float, List[str]]:
    changed = changed_feature_names(x, cf, feature_order)
    if space in {"model_raw", "original_raw"}:
        cols = list(feature_order)
        return euclidean(x.loc[cols].to_numpy(dtype=float), cf.loc[cols].to_numpy(dtype=float)), changed
    if space == "minmax_0_1":
        x_scaled = transform_minmax_0_1(x.to_frame().T, minmax_spec, feature_order).iloc[0]
        cf_scaled = transform_minmax_0_1(cf.to_frame().T, minmax_spec, feature_order).iloc[0]
        cols = list(feature_order)
        return euclidean(x_scaled.loc[cols].to_numpy(dtype=float), cf_scaled.loc[cols].to_numpy(dtype=float)), changed
    if space == "changed_features_original":
        if not changed:
            return 0.0, changed
        return euclidean(x.loc[changed].to_numpy(dtype=float), cf.loc[changed].to_numpy(dtype=float)), changed
    raise ValueError(f"Unsupported space: {space}")


def compute_distance_detail(
    pair_df: pd.DataFrame,
    feature_order: Sequence[str],
    minmax_spec: MinMaxSpec,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in pair_df.iterrows():
        x = pd.Series({feature: row[f"x::{feature}"] for feature in feature_order}, dtype=float)
        cf = pd.Series({feature: row[f"cf::{feature}"] for feature in feature_order}, dtype=float)
        for space in SPACES:
            distance, changed = distance_for_space(x, cf, feature_order, space, minmax_spec)
            rows.append({
                "dataset": row.get("dataset", "wine"),
                "variant": row["method"],
                "output_set": row["output_set"],
                "fold_id": row["fold_id"],
                "query_pos": int(row["query_pos"]),
                "space": space,
                "distance": distance,
                "strict_valid_selected": int(row.get("strict_valid_selected", 0)),
                "selected_source": row.get("selected_source", ""),
                "changed_feature_count": int(len(changed)),
                "changed_features": "|".join(changed),
            })
    return pd.DataFrame(rows)


def summarize_distances(detail_df: pd.DataFrame, subset_policy: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["variant", "output_set", "space"]
    for (variant, output_set, space), group in detail_df.groupby(group_cols, dropna=False):
        row_values = group["distance"].astype(float).to_numpy()
        fold_means = (
            group.groupby("fold_id", dropna=False)["distance"]
            .mean()
            .astype(float)
            .to_numpy()
        )
        values = fold_means if fold_means.size else row_values
        rows.append({
            "variant": variant,
            "output_set": output_set,
            "subset_policy": subset_policy,
            "space": space,
            "n": int(row_values.size),
            "fold_count": int(fold_means.size),
            "aggregation": "mean_of_fold_means",
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
            "row_mean": float(np.mean(row_values)) if row_values.size else float("nan"),
            "row_median": float(np.median(row_values)) if row_values.size else float("nan"),
            "row_std": float(np.std(row_values)) if row_values.size else float("nan"),
            "row_min": float(np.min(row_values)) if row_values.size else float("nan"),
            "row_max": float(np.max(row_values)) if row_values.size else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["variant", "output_set", "space"]).reset_index(drop=True)


def common_subset_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for variant, variant_df in detail_df.groupby("variant", dropna=False):
        key_sets = []
        for output_set in OUTPUT_SETS:
            output_df = variant_df.loc[variant_df["output_set"] == output_set]
            keys = set(zip(output_df["fold_id"], output_df["query_pos"]))
            key_sets.append(keys)
        common = set.intersection(*key_sets) if key_sets else set()
        if not common:
            continue
        mask = variant_df.apply(lambda row: (row["fold_id"], row["query_pos"]) in common, axis=1)
        frames.append(variant_df.loc[mask].copy())
    if not frames:
        return detail_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def load_thesis_baseline(baseline_dir: Path = BASELINE_DIR) -> Dict[Tuple[str, str], float]:
    path = baseline_dir / "author_pool_metric_summary_long.csv"
    if not path.exists():
        return dict(FALLBACK_THESIS_BASELINE)
    baseline: Dict[Tuple[str, str], float] = {}
    df = pd.read_csv(path)
    rows = df.loc[(df["dataset"] == "wine") & (df["metric"] == "Prox-Euc")]
    for _, row in rows.iterrows():
        value = finite_or_none(row["value"])
        if value is not None:
            baseline[(str(row["method"]), str(row["metric_scope"]))] = value
    return baseline or dict(FALLBACK_THESIS_BASELINE)


def attach_baselines(summary_df: pd.DataFrame, thesis_baseline: Mapping[Tuple[str, str], float]) -> pd.DataFrame:
    out = summary_df.copy()
    out["published_original"] = out["variant"].map(PUBLISHED_ORIGINAL).astype(float)
    out["thesis_baseline"] = [
        float(thesis_baseline.get((str(row.variant), str(row.output_set)), float("nan")))
        for row in out.itertuples(index=False)
    ]
    out["delta_to_published"] = out["mean"] - out["published_original"]
    out["delta_to_thesis"] = out["mean"] - out["thesis_baseline"]
    return out


def generate_pairs(output_root: Path) -> Path:
    mpl_config = output_root / "_mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(mpl_config))
    cmd = [
        sys.executable,
        str(PAIR_SCRIPT),
        "--dataset",
        "wine",
        "--config-profile",
        "final_freeze",
        "--bundle-mode",
        "table7_author_public",
        "--out-dir",
        str(output_root),
        "--emit-pairs",
        "--no-progress",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)
    pair_path = output_root / "author_pool_pairs.csv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Expected generated pair file: {pair_path}")
    return pair_path


def materialize_pairs(input_pairs: Optional[Path], output_root: Path) -> Path:
    target = output_root / "author_pool_pairs.csv"
    if input_pairs is None:
        return generate_pairs(output_root)
    source = input_pairs.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source != target.resolve():
        shutil.copyfile(source, target)
    return target


def assert_baseline_reproduction(summary_df: pd.DataFrame, thesis_baseline: Mapping[Tuple[str, str], float]) -> None:
    native_original = summary_df.loc[
        (summary_df["subset_policy"] == "native_output_set") & (summary_df["space"] == "original_raw")
    ]
    missing: List[str] = []
    mismatches: List[str] = []
    for method in METHODS:
        for output_set in OUTPUT_SETS:
            expected = thesis_baseline.get((method, output_set))
            row = native_original.loc[
                (native_original["variant"] == method) & (native_original["output_set"] == output_set)
            ]
            if expected is None or row.empty:
                missing.append(f"{method}/{output_set}")
                continue
            actual = float(row.iloc[0]["mean"])
            if round(actual, 2) != round(float(expected), 2):
                mismatches.append(f"{method}/{output_set}: actual={actual:.6f} expected={float(expected):.6f}")
    if missing or mismatches:
        raise AssertionError(
            "Baseline reproduction failed. "
            f"missing={missing} mismatches={mismatches}"
        )


def build_interpretation(
    summary_df: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    native = summary_df.loc[summary_df["subset_policy"] == "native_output_set"].copy()
    original = native.loc[native["space"] == "original_raw"].copy()
    minmax = native.loc[native["space"] == "minmax_0_1"].copy()
    closest = native.assign(abs_delta=native["delta_to_published"].abs()).sort_values("abs_delta").head(3)
    denominator_lines: List[str] = []
    for method in METHODS:
        vals = original.loc[original["variant"] == method].set_index("output_set")["mean"].to_dict()
        if not vals:
            continue
        denominator_lines.append(
            f"- {method}: raw={vals.get('author_raw', float('nan')):.2f}, "
            f"posthoc={vals.get('author_posthoc_valid_only', float('nan')):.2f}, "
            f"UFCE-FF={vals.get('author_pool_forceflip', float('nan')):.2f}"
        )
    model_note = (
        "Red Wine metadata reports `has_scaler=false`; therefore `model_raw` and "
        "`original_raw` are intentionally identical in this diagnostic."
        if not bool(metadata.get("has_scaler", False))
        else "A persisted Red Wine scaler was reported in metadata."
    )
    closest_lines = [
        f"- {row.variant}/{row.output_set}/{row.space}: mean={row.mean:.2f}, "
        f"published={row.published_original:.2f}, delta={row.delta_to_published:.2f}"
        for row in closest.itertuples(index=False)
    ]
    minmax_lines = [
        f"- {row.variant}/{row.output_set}: minmax mean={row.mean:.4f}"
        for row in minmax.sort_values(["variant", "output_set"]).itertuples(index=False)
    ]
    return "\n".join([
        "# Red Wine Prox-Euc Sensitivity Interpretation",
        "",
        model_note,
        "",
        "## Denominator Check",
        "",
        *denominator_lines,
        "",
        "## Scale Check",
        "",
        *minmax_lines,
        "",
        "## Closest Tested Variants To Published Original",
        "",
        *closest_lines,
        "",
        "## Thesis-ready Paragraph",
        "",
        (
            "Đối với Red Wine, luận văn kiểm tra bổ sung độ nhạy của Prox-Euc theo "
            "không gian đo khoảng cách và mẫu số đánh giá. Kết quả cho thấy việc "
            "chuyển sang min-max [0,1] không kéo Prox-Euc lại gần các giá trị công "
            "bố gốc; đồng thời các mẫu số `author_raw`, hậu kiểm đảo nhãn và UFCE-FF "
            "tạo ra phân phối khoảng cách khác nhau, đặc biệt ở UFCE2 và UFCE3. Vì "
            "vậy, Red Wine được đọc như trường hợp giới hạn của tái hiện số học: "
            "luận văn giữ kết quả này để minh bạch sai lệch, nhưng không dùng riêng "
            "Prox-Euc trên Red Wine để đưa ra kết luận tổng quát về ưu thế của phương pháp."
        ),
        "",
        "Prox-Jac is omitted for Red Wine because all Red Wine features are numeric.",
        "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Red Wine Prox-Euc sensitivity across distance spaces.")
    parser.add_argument("--pairs-csv", default=None, help="Optional author_pool_pairs.csv from script 09.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--skip-baseline-assert", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.out_dir).resolve() if args.out_dir else (OUT_PARENT / now_tag()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    input_pair_path = Path(args.pairs_csv).resolve() if args.pairs_csv else None
    pair_path = materialize_pairs(input_pair_path, output_root)
    pair_df = pd.read_csv(pair_path)
    feature_order, metadata = load_feature_order()
    validate_feature_alignment(pair_df, feature_order)

    train_df = load_training_frame(metadata, feature_order)
    minmax_spec = fit_minmax_0_1(train_df, feature_order)
    detail_df = compute_distance_detail(pair_df, feature_order, minmax_spec)
    native_summary = summarize_distances(detail_df, "native_output_set")
    common_detail = common_subset_detail(detail_df)
    common_summary = summarize_distances(common_detail, "common_sample_subset")

    thesis_baseline = load_thesis_baseline(Path(args.baseline_dir))
    summary_df = attach_baselines(pd.concat([native_summary, common_summary], ignore_index=True), thesis_baseline)
    native_summary_with_baseline = summary_df.loc[summary_df["subset_policy"] == "native_output_set"].reset_index(drop=True)
    common_summary_with_baseline = summary_df.loc[summary_df["subset_policy"] == "common_sample_subset"].reset_index(drop=True)
    baseline_comparison = native_summary_with_baseline.loc[
        native_summary_with_baseline["space"] == "original_raw"
    ].reset_index(drop=True)

    assert_reference_dir = output_root
    if input_pair_path is not None and (input_pair_path.parent / "author_pool_metric_summary_long.csv").exists():
        assert_reference_dir = input_pair_path.parent
    assert_reference = load_thesis_baseline(assert_reference_dir)
    if not bool(args.skip_baseline_assert):
        assert_baseline_reproduction(summary_df, assert_reference)

    detail_df.to_csv(output_root / "prox_euc_sensitivity_detail.csv", index=False)
    native_summary_with_baseline.to_csv(output_root / "prox_euc_sensitivity_summary.csv", index=False)
    common_summary_with_baseline.to_csv(output_root / "prox_euc_common_subset_summary.csv", index=False)
    baseline_comparison.to_csv(output_root / "baseline_comparison.csv", index=False)
    (output_root / "interpretation.md").write_text(
        build_interpretation(native_summary_with_baseline, metadata),
        encoding="utf-8",
    )
    summary_payload = {
        "status": "ok",
        "dataset": "wine",
        "source_pairs": str(pair_path),
        "baseline_dir": str(Path(args.baseline_dir).resolve()),
        "baseline_assert_reference_dir": str(assert_reference_dir.resolve()),
        "feature_order": feature_order,
        "has_scaler": bool(metadata.get("has_scaler", False)),
        "model_raw_equals_original_raw": not bool(metadata.get("has_scaler", False)),
        "prox_jac_omitted": True,
        "outputs": {
            "pairs": "author_pool_pairs.csv",
            "detail": "prox_euc_sensitivity_detail.csv",
            "summary": "prox_euc_sensitivity_summary.csv",
            "common_subset_summary": "prox_euc_common_subset_summary.csv",
            "baseline_comparison": "baseline_comparison.csv",
            "interpretation": "interpretation.md",
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[OK] Red Wine Prox-Euc sensitivity outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
