#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("docs/validation/corpora/part2_tier_b_bank_sessions_v1.json")
DEFAULT_OUTPUT = Path("part2_expected_case_distribution.csv")
DEFAULT_DETAIL_OUTPUT = Path("part2_expected_cases_normalized.csv")
CORPUS_LABEL = "main_200"

GROUP_CATEGORY_ORDER = {
    "G1": [
        "complete_profile",
        "missing_information",
        "no_recourse_needed",
        "runtime_reject_expected",
    ],
    "G2": [
        "hard_constraint",
        "numeric_bound",
        "blocked_feature",
        "constraint_sensitive",
    ],
}

DESCRIPTIONS = {
    "complete_profile": "G1 single-turn bank profile with all required fields present before runtime.",
    "missing_information": "G1 profile intentionally split across clarification and follow-up turns.",
    "no_recourse_needed": "Outcome-only bucket; not encoded as a pre-run corpus design category in v1.",
    "runtime_reject_expected": "Outcome-only bucket; not encoded as a pre-run corpus design category in v1.",
    "hard_constraint": "G2 active constraint with max_changed_features, including combined disallowed-change cases.",
    "numeric_bound": "G2 active constraint with numeric final-value bounds.",
    "blocked_feature": "G2 active constraint with disallowed_changes only.",
    "constraint_sensitive": "G2 active constraint with prefer_fewer_changes, including combined disallowed-change cases.",
    "unknown": "Unmapped corpus design shape.",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.load(path.open("r", encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("cases", "sessions", "items", "data", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure: {path}")
    raise ValueError(f"Unsupported file type: {path}")


def infer_expected_category(case: dict[str, Any]) -> str:
    group = str(case.get("group") or "")
    if group == "G1":
        session_shape = str(case.get("session_shape") or "")
        if session_shape == "clarification_followup":
            return "missing_information"
        if session_shape == "single_turn":
            return "complete_profile"
        return "unknown"

    if group == "G2":
        spec = case.get("active_constraint_spec_expected") or {}
        if not isinstance(spec, dict):
            return "unknown"
        if "numeric_bounds" in spec:
            return "numeric_bound"
        if "max_changed_features" in spec:
            return "hard_constraint"
        if "prefer_fewer_changes" in spec:
            return "constraint_sensitive"
        if "disallowed_changes" in spec:
            return "blocked_feature"
        return "unknown"

    return "unknown"


def build_distribution(detail_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detail_df.groupby(["corpus", "group", "expected_category"], dropna=False)
        .size()
        .reset_index(name="n_sessions")
    )
    ordered_rows = []
    for group, categories in GROUP_CATEGORY_ORDER.items():
        for category in categories:
            matching = grouped[
                (grouped["group"] == group) & (grouped["expected_category"] == category)
            ]
            n_sessions = int(matching["n_sessions"].iloc[0]) if not matching.empty else 0
            ordered_rows.append(
                {
                    "corpus": CORPUS_LABEL,
                    "group": group,
                    "expected_category": category,
                    "n_sessions": n_sessions,
                }
            )

    extra = grouped[
        ~grouped.apply(
            lambda row: str(row["expected_category"]) in GROUP_CATEGORY_ORDER.get(str(row["group"]), []),
            axis=1,
        )
    ].copy()
    if not extra.empty:
        ordered_rows.extend(extra.to_dict("records"))

    dist = pd.DataFrame(ordered_rows)
    group_totals = detail_df.groupby("group").size().to_dict()
    total = len(detail_df)
    dist["pct_within_group"] = dist.apply(
        lambda row: (float(row["n_sessions"]) / float(group_totals.get(row["group"], 0)))
        if group_totals.get(row["group"], 0)
        else 0.0,
        axis=1,
    )
    dist["pct_total"] = dist["n_sessions"].astype(float) / float(total) if total else 0.0
    dist["description"] = dist["expected_category"].map(DESCRIPTIONS).fillna("")
    return dist[
        [
            "corpus",
            "group",
            "expected_category",
            "n_sessions",
            "pct_within_group",
            "pct_total",
            "description",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the pre-run expected design distribution for Part II G1/G2 bank sessions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_OUTPUT)
    args = parser.parse_args()

    records = load_records(args.input)
    rows = []
    for case in records:
        group = str(case.get("group") or "")
        if group not in {"G1", "G2"}:
            continue
        rows.append(
            {
                "corpus": CORPUS_LABEL,
                "case_id": str(case.get("case_id") or case.get("session_id") or case.get("id") or ""),
                "group": group,
                "session_shape": str(case.get("session_shape") or ""),
                "expected_category": infer_expected_category(case),
                "active_constraint_spec_expected": json.dumps(
                    case.get("active_constraint_spec_expected"), ensure_ascii=False, sort_keys=True
                )
                if case.get("active_constraint_spec_expected") is not None
                else "",
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError("No G1/G2 cases found in input corpus.")

    dist = build_distribution(detail_df)
    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.detail_output, index=False)
    dist.to_csv(args.output, index=False)

    print(f"input = {args.input}")
    print(f"detail_output = {args.detail_output}")
    print(f"output = {args.output}")
    print(f"total_records = {len(detail_df)}")
    print("group_counts =", detail_df.groupby("group").size().to_dict())
    print(dist.to_string(index=False))
    unknown = int((detail_df["expected_category"] == "unknown").sum())
    if unknown:
        print(f"WARNING: unknown categories found: {unknown}")


if __name__ == "__main__":
    main()
