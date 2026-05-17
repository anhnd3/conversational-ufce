#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

from scripts.archieve.reproduce_full_table7_result import AUTHOR_TABLE7  # noqa: E402


DEFAULT_STRICT_INPUT = Path("outputs/final_table7_freeze/202625040849/final_table7_strict_validity_long.csv")
DEFAULT_RAW_INPUT = Path("outputs/final_table7_freeze/202625040849/final_table7_raw_long.csv")
DEFAULT_OUT_DIR = Path("chapter4_part1_tables")

DATASET_ORDER = ["bank", "grad", "wine", "bupa", "movie"]
DATASET_DISPLAY = {
    "bank": "Bank Loan",
    "grad": "Graduate Admission",
    "wine": "Red Wine",
    "bupa": "BUPA",
    "movie": "Movie",
}
METHOD_ORDER = ["UFCE1", "UFCE2", "UFCE3", "DiCE", "AR"]
METRIC_ORDER = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]
METRIC_AUTHOR_KEYS = {
    "Prox-Jac": ["Prox-Jac", "prox_jac", "prox-jac", "proxjac"],
    "Prox-Euc": ["Prox-Euc", "prox_euc", "prox-euc", "proxeuc"],
    "Sparsity": ["Sparsity", "sparsity"],
    "Actionability": ["Actionability", "actionability"],
    "Plausibility": ["Plausibility", "plausibility"],
    "Feasibility": ["Feasibility", "feasibility"],
}


def normalize_method(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper().replace("_", "-")
    if upper == "DICE":
        return "DiCE"
    if upper in {"DICE-UF", "DICEUF", "DICE UF"}:
        return "DiCE-UF"
    if upper.startswith("UFCE"):
        return upper
    if upper == "AR":
        return "AR"
    return text


def normalize_metric(value: Any) -> str:
    text = str(value).strip()
    key = text.lower().replace("_", "-").replace(" ", "-")
    mapping = {
        "prox-jac": "Prox-Jac",
        "proxjac": "Prox-Jac",
        "prox-euc": "Prox-Euc",
        "proxeuc": "Prox-Euc",
        "sparsity": "Sparsity",
        "actionability": "Actionability",
        "plausibility": "Plausibility",
        "feasibility": "Feasibility",
    }
    return mapping.get(key, text)


def normalize_dataset_id(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "bank": "bank",
        "bank_loan": "bank",
        "grad": "grad",
        "graduate": "grad",
        "graduate_admission": "grad",
        "wine": "wine",
        "red_wine": "wine",
        "bupa": "bupa",
        "movie": "movie",
    }
    return mapping.get(text, text)


def finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def author_value(dataset_id: str, method: str, metric: str) -> float:
    method_values = (AUTHOR_TABLE7.get(dataset_id) or {}).get(method) or {}
    for key in METRIC_AUTHOR_KEYS[metric]:
        if key in method_values:
            return finite_or_nan(method_values[key])
    return float("nan")


def load_ours(path: Path) -> dict[tuple[str, str, str], float]:
    df = pd.read_csv(path)
    required = {"dataset", "method", "metric", "ours"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    values: dict[tuple[str, str, str], float] = {}
    for row in df.to_dict("records"):
        dataset_id = normalize_dataset_id(row["dataset"])
        method = normalize_method(row["method"])
        metric = normalize_metric(row["metric"])
        if method == "DiCE-UF":
            continue
        values[(dataset_id, method, metric)] = finite_or_nan(row["ours"])
    return values


def build_long_table(input_path: Path) -> pd.DataFrame:
    ours_values = load_ours(input_path)
    rows = []
    for dataset_id in DATASET_ORDER:
        for method in METHOD_ORDER:
            for metric in METRIC_ORDER:
                ours = ours_values.get((dataset_id, method, metric), float("nan"))
                author = author_value(dataset_id, method, metric)
                delta = ours - author if math.isfinite(ours) and math.isfinite(author) else float("nan")
                rows.append(
                    {
                        "dataset": DATASET_DISPLAY[dataset_id],
                        "method": method,
                        "metric": metric,
                        "author": author,
                        "ours": ours,
                        "delta": delta,
                        "_dataset_id": dataset_id,
                    }
                )
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    public_df = df.drop(columns=["_dataset_id"])
    public_df.to_csv(out_dir / "part1_table7_main_long.csv", index=False)
    for dataset_id in DATASET_ORDER:
        sub = df[df["_dataset_id"].eq(dataset_id)].drop(columns=["_dataset_id"])
        sub.to_csv(out_dir / f"part1_table7_dataset_{dataset_id}.csv", index=False)


def build_combined_table(*, strict_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    strict_public = strict_df.drop(columns=["_dataset_id"]).copy()
    raw_public = raw_df.drop(columns=["_dataset_id"]).copy()
    strict_public.insert(0, "run_type", "strict_validity")
    raw_public.insert(0, "run_type", "raw")
    return pd.concat([strict_public, raw_public], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format Part I Table 7 into per-dataset Chapter 4 tables.")
    parser.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT_INPUT)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    strict_df = build_long_table(args.strict_input)
    raw_df = build_long_table(args.raw_input)

    write_tables(strict_df, args.out_dir)
    write_tables(strict_df, args.out_dir / "strict_validity")
    write_tables(raw_df, args.out_dir / "raw")
    combined_df = build_combined_table(strict_df=strict_df, raw_df=raw_df)
    combined_df.to_csv(args.out_dir / "part1_table7_main_long_both.csv", index=False)

    print(f"strict_input = {args.strict_input}")
    print(f"raw_input = {args.raw_input}")
    print(f"out_dir = {args.out_dir}")
    print("canonical rows =", len(strict_df))
    print("combined rows =", len(combined_df))
    print(
        strict_df.drop(columns=["_dataset_id"])
        .groupby(["dataset", "method"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
