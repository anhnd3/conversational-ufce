#!/usr/bin/env python3
"""
Build a core-only Bank confirmation table from raw, post-hoc, and UFCE-FF runs.

This script is intentionally provenance-heavy: raw uses the core-only
reproduction summary, while post-hoc and UFCE-FF use the selector comparison
summary with explicit route-separation fields.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final.part1.ufce_only_reproduction import AUTHOR_TABLE7, METRICS, METHODS


FF_METRIC_COLUMNS = {
    "Prox-Jac": "mean_prox_jac_valid",
    "Prox-Euc": "mean_prox_euc_valid",
    "Sparsity": "mean_sparsity_valid",
    "Actionability": "mean_actionability_valid",
    "Plausibility": "mean_plausibility_valid",
    "Feasibility": "mean_feasibility_valid",
}


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def metric_value(series: pd.Series, metric: str) -> Optional[float]:
    col = FF_METRIC_COLUMNS[metric]
    if col not in series.index:
        return None
    return finite_float(series[col])


def load_raw_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    missing_metrics = [metric for metric in METRICS if metric not in df.index]
    missing_methods = [method for method in METHODS if method not in df.columns]
    if missing_metrics or missing_methods:
        raise ValueError(
            f"Raw summary is missing metrics={missing_metrics} methods={missing_methods}: {path}"
        )
    return df


def select_ff_row(ff_df: pd.DataFrame, *, dataset: str, method: str, mode: str) -> pd.Series:
    mask = (
        (ff_df["dataset"].astype(str) == dataset)
        & (ff_df["method"].astype(str) == method)
        & (ff_df["mode"].astype(str) == mode)
        & (ff_df["mi_k"].astype(str) == "5")
    )
    rows = ff_df.loc[mask].copy()
    if rows.empty:
        raise ValueError(f"Missing selector summary row for dataset={dataset} method={method} mode={mode}")
    rows = rows.sort_values(by=["top_n"], kind="mergesort")
    return rows.iloc[0]


def load_legacy_author_pool(path: Optional[Path], *, dataset: str) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = df[
        (df["dataset"].astype(str) == dataset)
        & (df["metric_scope"].astype(str) == "author_pool_forceflip")
    ]
    out: Dict[str, Dict[str, float]] = {}
    for row in rows.to_dict(orient="records"):
        method = str(row["method"])
        out[method] = {metric: finite_float(row.get(metric)) for metric in METRICS}
    return out


def build_table(
    *,
    dataset: str,
    raw_df: pd.DataFrame,
    ff_df: pd.DataFrame,
    legacy_author_pool: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in METHODS:
        posthoc = select_ff_row(ff_df, dataset=dataset, method=method, mode="public_posthoc")
        forceflip = select_ff_row(ff_df, dataset=dataset, method=method, mode="public_forceflip")
        for metric in METRICS:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "published_table7": finite_float(AUTHOR_TABLE7[dataset][method][metric]),
                    "raw_core_public_author": finite_float(raw_df.loc[metric, method]),
                    "posthoc_after_selection": metric_value(posthoc, metric),
                    "ufce_ff_pre_find_best_row": metric_value(forceflip, metric),
                    "legacy_author_pool_forceflip": legacy_author_pool.get(method, {}).get(metric),
                    "raw_selection_policy": "public_author",
                    "raw_validity_gate_stage": "none",
                    "posthoc_selection_policy": str(posthoc.get("selection_policy", "")),
                    "posthoc_validity_gate_stage": str(posthoc.get("validity_gate_stage", "")),
                    "ufce_ff_selection_policy": str(forceflip.get("selection_policy", "")),
                    "ufce_ff_validity_gate_stage": str(forceflip.get("validity_gate_stage", "")),
                    "posthoc_effective_force_flip": int(posthoc.get("effective_force_flip", 0)),
                    "ufce_ff_effective_force_flip": int(forceflip.get("effective_force_flip", 0)),
                }
            )
    return pd.DataFrame(rows)


def build_reconcile_note(table: pd.DataFrame, legacy_path: Optional[Path]) -> str:
    lines = [
        "# Core-Only Bank Confirmation Reconcile Note",
        "",
        "This table separates three routes by selection stage:",
        "",
        "- Raw core reproduction: public-author selection, no added validity gate.",
        "- Post-hoc: public-author selection first, validity audit after selection.",
        "- UFCE-FF: force-flip candidate filtering before find_best_row.",
        "",
    ]
    if legacy_path is not None:
        lines.append(f"Legacy author-pool reference, when present, was read from `{legacy_path}`.")
        lines.append("")
    disputed = table[
        (table["method"] == "UFCE1")
        & (table["metric"].isin(["Plausibility", "Feasibility"]))
    ]
    if not disputed.empty:
        lines.append("## UFCE1 Plausibility/Feasibility")
        lines.append("")
        for row in disputed.to_dict(orient="records"):
            lines.append(
                "- {metric}: raw_core_public_author={raw}, posthoc_after_selection={posthoc}, "
                "ufce_ff_pre_find_best_row={ff}, legacy_author_pool_forceflip={legacy}.".format(
                    metric=row["metric"],
                    raw=row["raw_core_public_author"],
                    posthoc=row["posthoc_after_selection"],
                    ff=row["ufce_ff_pre_find_best_row"],
                    legacy=row["legacy_author_pool_forceflip"],
                )
            )
        lines.append("")
        lines.append(
            "If the core-only cells differ from the legacy author-pool cells, treat this as a provenance "
            "difference. Do not mix these cells in the same thesis comparison without naming the runner family."
        )
    return "\n".join(lines) + "\n"


def render_md(table: pd.DataFrame) -> str:
    display_cols = [
        "method",
        "metric",
        "published_table7",
        "raw_core_public_author",
        "posthoc_after_selection",
        "ufce_ff_pre_find_best_row",
        "legacy_author_pool_forceflip",
    ]
    return "# Core-Only Bank Confirmation Table\n\n" + table.loc[:, display_cols].to_markdown(index=False) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build core-only Bank confirmation table.")
    parser.add_argument("--raw-summary-csv", required=True)
    parser.add_argument("--ff-selector-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="bank", choices=["bank"])
    parser.add_argument(
        "--legacy-author-pool-csv",
        default=os.path.join(
            "outputs",
            "final",
            "part1",
            "09_author_pool_finalfreeze_20260624",
            "author_pool_metric_summary.csv",
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_summary_csv = Path(args.raw_summary_csv)
    ff_selector_csv = Path(args.ff_selector_csv)
    legacy_path = Path(args.legacy_author_pool_csv) if args.legacy_author_pool_csv else None

    raw_df = load_raw_summary(raw_summary_csv)
    ff_df = pd.read_csv(ff_selector_csv)
    legacy = load_legacy_author_pool(legacy_path, dataset=args.dataset)
    table = build_table(dataset=args.dataset, raw_df=raw_df, ff_df=ff_df, legacy_author_pool=legacy)

    csv_path = out_dir / "core_only_bank_confirmation_table.csv"
    md_path = out_dir / "core_only_bank_confirmation_table.md"
    json_path = out_dir / "core_only_bank_confirmation_provenance.json"
    reconcile_path = out_dir / "core_only_vs_legacy_author_pool_reconcile.md"

    table.to_csv(csv_path, index=False)
    md_path.write_text(render_md(table), encoding="utf-8")
    reconcile_path.write_text(build_reconcile_note(table, legacy_path), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "dataset": args.dataset,
                "raw_summary_csv": str(raw_summary_csv),
                "ff_selector_csv": str(ff_selector_csv),
                "legacy_author_pool_csv": str(legacy_path) if legacy_path is not None else None,
                "outputs": {
                    "csv": str(csv_path),
                    "md": str(md_path),
                    "reconcile_md": str(reconcile_path),
                },
                "route_contract": {
                    "raw": {"selection_policy": "public_author", "validity_gate_stage": "none"},
                    "posthoc": {"selection_policy": "public_author", "validity_gate_stage": "posthoc_after_selection"},
                    "ufce_ff": {"selection_policy": "force_flip", "validity_gate_stage": "pre_find_best_row"},
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
