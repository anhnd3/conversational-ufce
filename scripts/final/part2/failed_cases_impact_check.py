#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.utils.io import json_ready, write_json
from llm.src.utils.time import local_now_iso


METHODS = ["UFCE1", "UFCE2", "UFCE3"]
GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5"]
CONFLICT_FIELD_ORDER = [
    "Income",
    "Mortgage",
    "CCAvg",
    "Education",
    "Family",
    "SecuritiesAccount",
    "CDAccount",
    "Online",
    "CreditCard",
]
SUMMARY_FIELD_ORDER = ["Income", "Mortgage", "CCAvg", "Khác"]
FINAL_PARSE_RUN_ID = "nl_bank_bridge_20260704_200630"
FINAL_FULL250_RUN_ID = f"{FINAL_PARSE_RUN_ID}_latest_ufceff_all250_20260707"
FINAL_ACCEPTED_RUN_ID = f"{FINAL_PARSE_RUN_ID}_latest_ufceff_accepted_20260707"
DEFAULT_PARSE_OUT_DIR = ROOT / "outputs" / "final" / "part2" / FINAL_PARSE_RUN_ID
DEFAULT_FULL250_OUT_DIR = ROOT / "outputs" / "final" / "part2" / FINAL_FULL250_RUN_ID
DEFAULT_ACCEPTED_OUT_DIR = ROOT / "outputs" / "final" / "part2" / FINAL_ACCEPTED_RUN_ID
DEFAULT_OUT_DIR_NAME = "impact_check_7_failed_cases"
FINAL_VS_FULL250_BASENAME = "route2_final_vs_full250"
EXPECTED_FAILED_CASE_COUNT = 7
EXPECTED_TOTAL_CASE_COUNT = 250
EXPECTED_ACCEPTED_CASE_COUNT = 243
EXPECTED_FOLD_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the impact of the 7 failed Bank Route 2 parser cases on UFCE-FF coverage.")
    parser.add_argument("--parse-out-dir", type=Path, default=DEFAULT_PARSE_OUT_DIR)
    parser.add_argument("--full250-out-dir", type=Path, default=DEFAULT_FULL250_OUT_DIR)
    parser.add_argument("--accepted-out-dir", type=Path, default=DEFAULT_ACCEPTED_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(args.accepted_out_dir, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed_rows = load_jsonl_rows(args.parse_out_dir / "parse_errors.jsonl")
    if len(failed_rows) != EXPECTED_FAILED_CASE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FAILED_CASE_COUNT} failed cases, found {len(failed_rows)}.")
    failed_df = pd.DataFrame(failed_rows)

    query_df = pd.read_csv(args.parse_out_dir / "bank_250_queries.csv")
    selection_df = pd.read_csv(args.accepted_out_dir / "evaluation_case_selection.csv")
    included_case_ids = load_included_case_ids(selection_df)
    dropped_case_ids = load_dropped_case_ids(selection_df)
    failed_case_ids = {str(case_id) for case_id in failed_df["case_id"].tolist()}
    if dropped_case_ids != failed_case_ids:
        raise ValueError("Dropped accepted-gate case_id set does not match parse_errors.jsonl.")

    full250_direct_df = pd.read_csv(args.full250_out_dir / "direct_ufce_ff_results.csv")
    accepted_direct_df = pd.read_csv(args.accepted_out_dir / "direct_ufce_ff_results.csv")
    accepted_nl_first_df = pd.read_csv(args.accepted_out_dir / "nl_first_ufce_ff_results.csv")
    filtered_accepted_direct_df = filter_result_frame_to_selected_cases(
        result_df=accepted_direct_df,
        included_case_ids=included_case_ids,
        expected_case_count=EXPECTED_ACCEPTED_CASE_COUNT,
    )
    filtered_accepted_nl_first_df = filter_result_frame_to_selected_cases(
        result_df=accepted_nl_first_df,
        included_case_ids=included_case_ids,
        expected_case_count=EXPECTED_ACCEPTED_CASE_COUNT,
    )
    validate_result_frame(
        full250_direct_df,
        expected_case_ids={str(case_id) for case_id in query_df["case_id"].tolist()},
        label="full250 direct",
    )
    validate_result_frame(
        filtered_accepted_nl_first_df,
        expected_case_ids=included_case_ids,
        label="accepted243 nl_first",
    )
    validate_result_frame(
        filtered_accepted_direct_df,
        expected_case_ids=included_case_ids,
        label="accepted243 direct",
    )

    failed_cases_df = build_failed_cases_frame(
        failed_df=failed_df,
        query_df=query_df,
        selection_df=selection_df,
        full250_direct_df=full250_direct_df,
    )
    method_summary_df = build_method_summary_table(
        full250_direct_df=full250_direct_df,
        filtered_accepted_nl_first_df=filtered_accepted_nl_first_df,
        failed_cases_df=failed_cases_df,
    )
    group_summary_df = build_group_summary_table(failed_cases_df)
    fold_summary_df = build_fold_summary_table(failed_cases_df)
    field_summary_df = build_field_summary_table(failed_cases_df)
    family_summary_df = build_family_summary_table(failed_cases_df)
    final_vs_full250_df = build_final_vs_full250_table(
        full250_direct_df=full250_direct_df,
        filtered_accepted_nl_first_df=filtered_accepted_nl_first_df,
        source_case_count=EXPECTED_TOTAL_CASE_COUNT,
        accepted_case_count=EXPECTED_ACCEPTED_CASE_COUNT,
        dropped_case_count=EXPECTED_FAILED_CASE_COUNT,
        fold_count=EXPECTED_FOLD_COUNT,
    )
    accepted_direct_equals_nl_first = result_frames_match_on_outcomes(
        left_df=filtered_accepted_direct_df,
        right_df=filtered_accepted_nl_first_df,
    )
    insights = build_insights(
        failed_cases_df=failed_cases_df,
        method_summary_df=method_summary_df,
        fold_summary_df=fold_summary_df,
    )
    summary_payload = build_summary_payload(
        parse_out_dir=args.parse_out_dir,
        full250_out_dir=args.full250_out_dir,
        accepted_out_dir=args.accepted_out_dir,
        out_dir=out_dir,
        failed_cases_df=failed_cases_df,
        method_summary_df=method_summary_df,
        group_summary_df=group_summary_df,
        fold_summary_df=fold_summary_df,
        field_summary_df=field_summary_df,
        family_summary_df=family_summary_df,
        final_vs_full250_df=final_vs_full250_df,
        accepted_direct_equals_nl_first=accepted_direct_equals_nl_first,
        insights=insights,
    )

    failed_cases_df.to_csv(out_dir / "failed_cases_impact_cases.csv", index=False)
    method_summary_df.to_csv(out_dir / "failed_cases_impact_by_method.csv", index=False)
    group_summary_df.to_csv(out_dir / "failed_cases_impact_by_group.csv", index=False)
    fold_summary_df.to_csv(out_dir / "failed_cases_impact_by_fold.csv", index=False)
    field_summary_df.to_csv(out_dir / "failed_cases_impact_by_field.csv", index=False)
    family_summary_df.to_csv(out_dir / "failed_cases_impact_by_family.csv", index=False)
    final_vs_full250_df.to_csv(out_dir / f"{FINAL_VS_FULL250_BASENAME}.csv", index=False)
    write_json(out_dir / "failed_cases_impact_summary.json", summary_payload)
    write_json(
        out_dir / f"{FINAL_VS_FULL250_BASENAME}.json",
        {
            "generated_at": summary_payload["generated_at"],
            "source_case_count": EXPECTED_TOTAL_CASE_COUNT,
            "accepted_case_count": EXPECTED_ACCEPTED_CASE_COUNT,
            "dropped_case_count": EXPECTED_FAILED_CASE_COUNT,
            "accepted_subset_direct_equals_nl_first": accepted_direct_equals_nl_first,
            "rows": final_vs_full250_df.to_dict(orient="records"),
        },
    )
    (out_dir / "failed_cases_impact_summary.md").write_text(
        render_summary_md(
            failed_cases_df=failed_cases_df,
            method_summary_df=method_summary_df,
            group_summary_df=group_summary_df,
            fold_summary_df=fold_summary_df,
            field_summary_df=field_summary_df,
            insights=insights,
        ),
        encoding="utf-8",
    )
    (out_dir / f"{FINAL_VS_FULL250_BASENAME}.md").write_text(
        render_final_vs_full250_md(
            final_vs_full250_df=final_vs_full250_df,
            accepted_direct_equals_nl_first=accepted_direct_equals_nl_first,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            json_ready(
                {
                    "out_dir": out_dir,
                    "failed_case_count": int(len(failed_cases_df)),
                    "coverage_cost_label": insights["coverage_cost_label"],
                    "lost_released_cf_total": insights["lost_released_cf_total"],
                }
            ),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def resolve_out_dir(accepted_out_dir: Path, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir.resolve()
    return accepted_out_dir.resolve() / DEFAULT_OUT_DIR_NAME


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object rows in {path}.")
                rows.append(payload)
    return rows


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"Expected JSON list, found: {text}")


def parse_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Expected JSON object, found: {text}")


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def derive_final_status(row: dict[str, Any]) -> str:
    if str(row.get("final_stage") or "") == "CONFLICT":
        return "conflict"
    if to_bool(row.get("ready_for_runtime")) and not to_bool(row.get("exact_reconstruction")):
        return "reconstruction_mismatch"
    return "parser_failure"


def extract_semantic_mismatch_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = parse_json_list(row.get("field_semantic_mismatch_json"))
    return [record for record in records if isinstance(record, dict)]


def quote_alignment_error_present(row: dict[str, Any]) -> bool:
    text_parts: list[str] = []
    for column in [
        "retry_errors_json",
        "schema_errors_json",
        "verifier_errors_json",
        "field_semantic_mismatch_json",
    ]:
        text_parts.append(str(row.get(column) or ""))
    haystack = " ".join(text_parts).lower()
    return "not an exact substring" in haystack


def derive_failure_family(row: dict[str, Any]) -> tuple[str, str]:
    semantic_mismatches = extract_semantic_mismatch_records(row)
    semantic_families = ordered_unique(
        [str(record.get("failure_family") or "").strip() for record in semantic_mismatches if str(record.get("failure_family") or "").strip()]
    )
    if semantic_families:
        return ";".join(semantic_families), semantic_families[0]
    if str(row.get("final_stage") or "") == "CONFLICT":
        return "semantic_conflict", "semantic_conflict"
    if quote_alignment_error_present(row):
        return "quote_alignment_error", "quote_alignment_error"
    return "parser_failure", "parser_failure"


def find_conflict_fields(text: str) -> list[str]:
    haystack = str(text or "")
    matches: list[str] = []
    for field_name in CONFLICT_FIELD_ORDER:
        if field_name in haystack:
            matches.append(field_name)
    return matches


def derive_failed_field(row: dict[str, Any]) -> tuple[str, str | None]:
    semantic_mismatches = extract_semantic_mismatch_records(row)
    semantic_fields = ordered_unique(
        [str(record.get("field_name") or "").strip() for record in semantic_mismatches if str(record.get("field_name") or "").strip()]
    )
    if semantic_fields:
        return ";".join(semantic_fields), semantic_fields[0]

    conflict_fields = find_conflict_fields(" ".join(parse_json_list(row.get("confirmed_conflicts_json"))))
    if conflict_fields:
        return ";".join(conflict_fields), conflict_fields[0]

    reconstruction_fields = ordered_unique([str(item).strip() for item in parse_json_list(row.get("reconstruction_mismatch_fields_json")) if str(item).strip()])
    if len(reconstruction_fields) == 1:
        return reconstruction_fields[0], reconstruction_fields[0]
    return "MULTI_FIELD_OR_GLOBAL", None


def collapse_primary_failed_field(primary_failed_field: str | None) -> str:
    if primary_failed_field in {"Income", "Mortgage", "CCAvg"}:
        return str(primary_failed_field)
    return "Khác"


def display_primary_failed_field(primary_failed_field: str | None) -> str:
    if primary_failed_field:
        return str(primary_failed_field)
    return "Khác"


def load_included_case_ids(selection_df: pd.DataFrame) -> set[str]:
    included_mask = selection_df["eval_included"].map(to_bool)
    included_case_ids = {str(case_id) for case_id in selection_df.loc[included_mask, "case_id"].tolist()}
    dropped_case_ids = {str(case_id) for case_id in selection_df.loc[~included_mask, "case_id"].tolist()}
    if len(included_case_ids) != EXPECTED_ACCEPTED_CASE_COUNT:
        raise ValueError(f"Expected {EXPECTED_ACCEPTED_CASE_COUNT} accepted cases, found {len(included_case_ids)}.")
    if len(dropped_case_ids) != EXPECTED_FAILED_CASE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FAILED_CASE_COUNT} dropped cases, found {len(dropped_case_ids)}.")
    if len(included_case_ids | dropped_case_ids) != EXPECTED_TOTAL_CASE_COUNT:
        raise ValueError("Accepted + dropped case count does not equal 250.")
    return included_case_ids


def load_dropped_case_ids(selection_df: pd.DataFrame) -> set[str]:
    included_mask = selection_df["eval_included"].map(to_bool)
    return {str(case_id) for case_id in selection_df.loc[~included_mask, "case_id"].tolist()}


def filter_result_frame_to_selected_cases(
    *,
    result_df: pd.DataFrame,
    included_case_ids: set[str],
    expected_case_count: int,
) -> pd.DataFrame:
    filtered_df = result_df.loc[result_df["case_id"].astype(str).isin(included_case_ids)].copy()
    expected_rows = expected_case_count * len(METHODS)
    if len(filtered_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} accepted result rows, found {len(filtered_df)}.")
    return filtered_df


def validate_result_frame(result_df: pd.DataFrame, *, expected_case_ids: set[str], label: str) -> None:
    observed_case_ids = {str(case_id) for case_id in result_df["case_id"].tolist()}
    if observed_case_ids != expected_case_ids:
        raise ValueError(f"{label} case_id set does not match the expected case set.")
    expected_pairs = {(case_id, method) for case_id in expected_case_ids for method in METHODS}
    observed_pairs = {(str(case_id), str(method)) for case_id, method in zip(result_df["case_id"], result_df["method"])}
    if observed_pairs != expected_pairs:
        raise ValueError(f"{label} case/method matrix is incomplete or contains duplicates.")


def build_failed_cases_frame(
    *,
    failed_df: pd.DataFrame,
    query_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    full250_direct_df: pd.DataFrame,
) -> pd.DataFrame:
    query_index = query_df.set_index("case_id", drop=False)
    selection_index = selection_df.set_index("case_id", drop=False)
    direct_index = full250_direct_df.set_index(["case_id", "method"], drop=False)
    rows: list[dict[str, Any]] = []

    for failed_row in failed_df.to_dict(orient="records"):
        case_id = str(failed_row["case_id"])
        if case_id not in query_index.index:
            raise ValueError(f"Missing provenance row for failed case: {case_id}")
        query_row = query_index.loc[case_id].to_dict()
        selection_row = selection_index.loc[case_id].to_dict()
        failure_family, primary_failure_family = derive_failure_family(failed_row)
        failed_field, primary_failed_field = derive_failed_field(failed_row)
        released_cf_counts: dict[str, int] = {}
        direct_statuses: dict[str, str] = {}
        for method in METHODS:
            if (case_id, method) not in direct_index.index:
                raise ValueError(f"Missing full250 direct result row for {case_id} / {method}")
            direct_row = direct_index.loc[(case_id, method)].to_dict()
            status = str(direct_row.get("status") or "")
            direct_statuses[method] = status
            released_cf_counts[method] = 1 if status == "released_cf" else 0

        rows.append(
            {
                "case_id": case_id,
                "fold_index": int(query_row["fold_index"]),
                "fold_name": str(query_row["fold_name"]),
                "query_pos": int(query_row["query_pos"]),
                "language_group": str(query_row["language_group"]),
                "language_label": str(query_row["language_label"]),
                "benchmark_key": str(query_row["benchmark_key"]),
                "original_profile_signature": str(query_row["original_profile_signature"]),
                "eval_drop_reason": str(selection_row.get("eval_drop_reason") or ""),
                "parser_status": str(failed_row.get("parser_status") or ""),
                "final_stage": str(failed_row.get("final_stage") or ""),
                "failed_stage": str(failed_row.get("failed_stage") or ""),
                "final_status": derive_final_status(failed_row),
                "failed_field": failed_field,
                "primary_failed_field": primary_failed_field or "",
                "collapsed_primary_failed_field": collapse_primary_failed_field(primary_failed_field),
                "failure_family": failure_family,
                "primary_failure_family": primary_failure_family,
                "ready_for_runtime": to_bool(failed_row.get("ready_for_runtime")),
                "exact_reconstruction": to_bool(failed_row.get("exact_reconstruction")),
                "has_valid_cf_ufce1": bool(released_cf_counts["UFCE1"]),
                "has_valid_cf_ufce2": bool(released_cf_counts["UFCE2"]),
                "has_valid_cf_ufce3": bool(released_cf_counts["UFCE3"]),
                "released_cf_count_ufce1": int(released_cf_counts["UFCE1"]),
                "released_cf_count_ufce2": int(released_cf_counts["UFCE2"]),
                "released_cf_count_ufce3": int(released_cf_counts["UFCE3"]),
                "released_cf_count_by_method_json": json.dumps(released_cf_counts, ensure_ascii=False, sort_keys=True),
                "full250_direct_status_ufce1": direct_statuses["UFCE1"],
                "full250_direct_status_ufce2": direct_statuses["UFCE2"],
                "full250_direct_status_ufce3": direct_statuses["UFCE3"],
                "full250_direct_status_by_method_json": json.dumps(direct_statuses, ensure_ascii=False, sort_keys=True),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["fold_index", "query_pos"], kind="mergesort").reset_index(drop=True)


def build_method_summary_table(
    *,
    full250_direct_df: pd.DataFrame,
    filtered_accepted_nl_first_df: pd.DataFrame,
    failed_cases_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        full_count = int(
            full250_direct_df.loc[
                (full250_direct_df["method"] == method) & (full250_direct_df["status"] == "released_cf")
            ].shape[0]
        )
        accepted_count = int(
            filtered_accepted_nl_first_df.loc[
                (filtered_accepted_nl_first_df["method"] == method) & (filtered_accepted_nl_first_df["status"] == "released_cf")
            ].shape[0]
        )
        failed_valid_cf_count = int(failed_cases_df[f"has_valid_cf_{method.lower()}"].sum())
        rows.append(
            {
                "method": method,
                "full250_direct_released_cf_count": full_count,
                "accepted243_nl_first_released_cf_count": accepted_count,
                "lost_released_cf_count": full_count - accepted_count,
                "failed_cases_with_valid_cf": failed_valid_cf_count,
                "failed_cases_with_valid_cf_rate": failed_valid_cf_count / EXPECTED_FAILED_CASE_COUNT,
            }
        )
    return pd.DataFrame(rows)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def valid_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return finite_float(values.mean())


def fold_mean_sum(frame: pd.DataFrame, column: str, *, fold_count: int) -> float | None:
    if column not in frame.columns:
        return None
    if "fold_index" not in frame.columns:
        return finite_float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())
    values: list[float] = []
    for fold_index in range(fold_count):
        fold_rows = frame.loc[frame["fold_index"].astype(int) == fold_index]
        values.append(float(pd.to_numeric(fold_rows[column], errors="coerce").fillna(0.0).sum()))
    return finite_float(sum(values) / fold_count) if fold_count > 0 else None


def summarize_ufce_ff_metrics(
    *,
    frame: pd.DataFrame,
    setting: str,
    variant: str,
    source_case_count: int,
    eval_case_count: int,
    dropped_case_count: int,
    fold_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = frame.loc[frame["method"] == method].copy()
        released_rows = method_rows.loc[method_rows["status"] == "released_cf"].copy()
        released_count = int(len(released_rows))
        rows.append(
            {
                "setting": setting,
                "variant": variant,
                "method": method,
                "source_cases": int(source_case_count),
                "eval_cases": int(eval_case_count),
                "dropped": int(dropped_case_count),
                "released_cf": released_count,
                "release_per_eval": released_count / eval_case_count if eval_case_count else 0.0,
                "release_per_250": released_count / source_case_count if source_case_count else 0.0,
                "prox_jac": valid_mean(released_rows, "selected_prox_jac"),
                "prox_euc": valid_mean(released_rows, "selected_prox_euc"),
                "sparsity": valid_mean(released_rows, "selected_sparsity"),
                "actionability": fold_mean_sum(released_rows, "selected_actionability_pass", fold_count=fold_count),
                "plausibility": fold_mean_sum(released_rows, "selected_plausibility_pass", fold_count=fold_count),
                "feasibility": fold_mean_sum(released_rows, "selected_feasibility_pass", fold_count=fold_count),
            }
        )
    return rows


def build_final_vs_full250_table(
    *,
    full250_direct_df: pd.DataFrame,
    filtered_accepted_nl_first_df: pd.DataFrame,
    source_case_count: int,
    accepted_case_count: int,
    dropped_case_count: int,
    fold_count: int,
) -> pd.DataFrame:
    rows = summarize_ufce_ff_metrics(
        frame=full250_direct_df,
        setting="UFCE-FF direct comparator (full 250 source cases)",
        variant="direct_full250",
        source_case_count=source_case_count,
        eval_case_count=source_case_count,
        dropped_case_count=0,
        fold_count=fold_count,
    )
    rows.extend(
        summarize_ufce_ff_metrics(
            frame=filtered_accepted_nl_first_df,
            setting="Bank NLP -> UFCE-FF final result (accepted 243; nl_first)",
            variant="nl_first_accepted243",
            source_case_count=source_case_count,
            eval_case_count=accepted_case_count,
            dropped_case_count=dropped_case_count,
            fold_count=fold_count,
        )
    )
    return pd.DataFrame(rows)


def result_frames_match_on_outcomes(*, left_df: pd.DataFrame, right_df: pd.DataFrame) -> bool:
    columns = ["case_id", "method", "status", "selected_profile_signature"]
    left = left_df.loc[:, columns].sort_values(columns[:2], kind="mergesort").reset_index(drop=True)
    right = right_df.loc[:, columns].sort_values(columns[:2], kind="mergesort").reset_index(drop=True)
    return bool(left.equals(right))


def build_group_summary_table(failed_cases_df: pd.DataFrame) -> pd.DataFrame:
    group_counts = Counter(str(value) for value in failed_cases_df["language_group"].tolist())
    dominant_group = dominant_value(group_counts, GROUP_ORDER)
    rows: list[dict[str, Any]] = []
    for group_name in GROUP_ORDER:
        count = int(group_counts.get(group_name, 0))
        note = ""
        if count > 0 and group_name == dominant_group:
            group_slice = failed_cases_df.loc[failed_cases_df["language_group"] == group_name]
            note = "Nhóm lỗi nhiều nhất."
            if not group_slice.empty:
                top_field = dominant_value(
                    Counter(display_primary_failed_field(value or None) for value in group_slice["primary_failed_field"].tolist()),
                    CONFLICT_FIELD_ORDER + ["Khác"],
                )
                top_family = dominant_value(Counter(group_slice["primary_failure_family"].tolist()), [])
                if top_field and top_family:
                    note = f"Nhóm lỗi nhiều nhất; chủ yếu {top_field}/{top_family}."
        rows.append(
            {
                "group_id": group_name,
                "failed_case_count": count,
                "failed_case_rate": count / EXPECTED_FAILED_CASE_COUNT,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def build_fold_summary_table(failed_cases_df: pd.DataFrame) -> pd.DataFrame:
    fold_counts = Counter(int(value) for value in failed_cases_df["fold_index"].tolist())
    rows: list[dict[str, Any]] = []
    for fold_index in range(5):
        count = int(fold_counts.get(fold_index, 0))
        rows.append(
            {
                "fold_id": f"fold{fold_index}",
                "fold_index": fold_index,
                "failed_case_count": count,
                "failed_case_rate": count / EXPECTED_FAILED_CASE_COUNT,
            }
        )
    return pd.DataFrame(rows)


def build_field_summary_table(failed_cases_df: pd.DataFrame) -> pd.DataFrame:
    field_counts = Counter(str(value) for value in failed_cases_df["collapsed_primary_failed_field"].tolist())
    rows: list[dict[str, Any]] = []
    for field_name in SUMMARY_FIELD_ORDER:
        count = int(field_counts.get(field_name, 0))
        rows.append(
            {
                "failed_field": field_name,
                "failed_case_count": count,
                "failed_case_rate": count / EXPECTED_FAILED_CASE_COUNT,
            }
        )
    return pd.DataFrame(rows)


def build_family_summary_table(failed_cases_df: pd.DataFrame) -> pd.DataFrame:
    family_counts = Counter(str(value) for value in failed_cases_df["primary_failure_family"].tolist())
    ordered_families = sorted(family_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    rows = [
        {
            "failure_family": family_name,
            "failed_case_count": int(count),
            "failed_case_rate": int(count) / EXPECTED_FAILED_CASE_COUNT,
        }
        for family_name, count in ordered_families
    ]
    return pd.DataFrame(rows)


def dominant_value(counter: Counter[str], preferred_order: list[str]) -> str:
    if not counter:
        return ""
    order_index = {value: index for index, value in enumerate(preferred_order)}
    return sorted(
        counter.items(),
        key=lambda item: (-int(item[1]), order_index.get(str(item[0]), len(order_index)), str(item[0])),
    )[0][0]


def build_insights(
    *,
    failed_cases_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
) -> dict[str, Any]:
    lost_total = int(method_summary_df["lost_released_cf_count"].sum())
    coverage_cost_label = classify_coverage_cost(lost_total)

    dominant_method_row = method_summary_df.sort_values(
        by=["lost_released_cf_count", "method"],
        ascending=[False, True],
        key=lambda series: series.map({"UFCE1": 0, "UFCE2": 1, "UFCE3": 2}) if series.name == "method" else series,
    ).iloc[0]
    dominant_pattern_counts = Counter(
        (
            str(row["language_group"]),
            display_primary_failed_field(str(row["primary_failed_field"] or "").strip() or None),
            str(row["primary_failure_family"]),
        )
        for row in failed_cases_df.to_dict(orient="records")
    )
    dominant_pattern = sorted(
        dominant_pattern_counts.items(),
        key=lambda item: (
            -int(item[1]),
            GROUP_ORDER.index(item[0][0]) if item[0][0] in GROUP_ORDER else len(GROUP_ORDER),
            SUMMARY_FIELD_ORDER.index(item[0][1]) if item[0][1] in SUMMARY_FIELD_ORDER else len(SUMMARY_FIELD_ORDER),
            str(item[0][2]),
        ),
    )[0]
    fold_rows = fold_summary_df.to_dict(orient="records")
    max_fold_count = max(int(row["failed_case_count"]) for row in fold_rows) if fold_rows else 0
    dominant_folds = [str(row["fold_id"]) for row in fold_rows if int(row["failed_case_count"]) == max_fold_count and max_fold_count > 0]

    return {
        "lost_released_cf_total": lost_total,
        "coverage_cost_label": coverage_cost_label,
        "dominant_method": str(dominant_method_row["method"]),
        "dominant_method_lost_count": int(dominant_method_row["lost_released_cf_count"]),
        "dominant_pattern": {
            "language_group": dominant_pattern[0][0],
            "primary_failed_field": dominant_pattern[0][1],
            "failure_family": dominant_pattern[0][2],
            "count": int(dominant_pattern[1]),
        },
        "fold_concentration_abnormal": max_fold_count > 2,
        "dominant_folds": dominant_folds,
        "dominant_fold_count": max_fold_count,
    }


def classify_coverage_cost(lost_total: int) -> str:
    if lost_total <= 1:
        return "nhỏ"
    if lost_total <= 4:
        return "vừa"
    return "đáng kể"


def build_summary_payload(
    *,
    parse_out_dir: Path,
    full250_out_dir: Path,
    accepted_out_dir: Path,
    out_dir: Path,
    failed_cases_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
    field_summary_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    final_vs_full250_df: pd.DataFrame,
    accepted_direct_equals_nl_first: bool,
    insights: dict[str, Any],
) -> dict[str, Any]:
    thesis_sentence = render_thesis_sentence(method_summary_df, insights["coverage_cost_label"])
    return {
        "generated_at": local_now_iso(),
        "parse_out_dir": str(parse_out_dir.resolve()),
        "full250_out_dir": str(full250_out_dir.resolve()),
        "accepted_out_dir": str(accepted_out_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "overview": {
            "source_query_count": EXPECTED_TOTAL_CASE_COUNT,
            "accepted_query_count": EXPECTED_ACCEPTED_CASE_COUNT,
            "failed_query_count": EXPECTED_FAILED_CASE_COUNT,
            "failed_query_rate": EXPECTED_FAILED_CASE_COUNT / EXPECTED_TOTAL_CASE_COUNT,
        },
        "accepted_subset_direct_equals_nl_first": bool(accepted_direct_equals_nl_first),
        "insights": json_ready(insights),
        "thesis_sentence": thesis_sentence,
        "tables": {
            "failed_cases": failed_cases_df.to_dict(orient="records"),
            "by_method": method_summary_df.to_dict(orient="records"),
            "by_group": group_summary_df.to_dict(orient="records"),
            "by_fold": fold_summary_df.to_dict(orient="records"),
            "by_field": field_summary_df.to_dict(orient="records"),
            "by_family": family_summary_df.to_dict(orient="records"),
            "final_vs_full250": final_vs_full250_df.to_dict(orient="records"),
        },
        "artifacts": {
            "failed_cases_csv": str((out_dir / "failed_cases_impact_cases.csv").resolve()),
            "by_method_csv": str((out_dir / "failed_cases_impact_by_method.csv").resolve()),
            "by_group_csv": str((out_dir / "failed_cases_impact_by_group.csv").resolve()),
            "by_fold_csv": str((out_dir / "failed_cases_impact_by_fold.csv").resolve()),
            "by_field_csv": str((out_dir / "failed_cases_impact_by_field.csv").resolve()),
            "by_family_csv": str((out_dir / "failed_cases_impact_by_family.csv").resolve()),
            "final_vs_full250_csv": str((out_dir / f"{FINAL_VS_FULL250_BASENAME}.csv").resolve()),
            "final_vs_full250_json": str((out_dir / f"{FINAL_VS_FULL250_BASENAME}.json").resolve()),
            "final_vs_full250_md": str((out_dir / f"{FINAL_VS_FULL250_BASENAME}.md").resolve()),
            "summary_json": str((out_dir / "failed_cases_impact_summary.json").resolve()),
            "summary_md": str((out_dir / "failed_cases_impact_summary.md").resolve()),
        },
    }


def fmt_metric(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def render_final_vs_full250_md(
    *,
    final_vs_full250_df: pd.DataFrame,
    accepted_direct_equals_nl_first: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Route 2 Final Result vs Full-250 UFCE-FF Comparator")
    lines.append("")
    lines.append(
        f"- Final parser acceptance: `{EXPECTED_ACCEPTED_CASE_COUNT}/{EXPECTED_TOTAL_CASE_COUNT}` accepted; "
        f"`{EXPECTED_FAILED_CASE_COUNT}/{EXPECTED_TOTAL_CASE_COUNT}` dropped."
    )
    lines.append(
        f"- Accepted-subset direct equals nl_first: `{str(bool(accepted_direct_equals_nl_first)).lower()}`."
    )
    lines.append(
        "- `Release / eval` uses each row's evaluated-case denominator; `Release / 250` keeps the original 250-case denominator."
    )
    lines.append(
        "- `Actionability`, `Plausibility`, and `Feasibility` follow the fold-mean count convention used by the Part I runners."
    )
    lines.append("")
    lines.append(
        "| Setting | Method | Source cases | Eval cases | Dropped | Released CF | Release / eval | Release / 250 | Prox-Jac | Prox-Euc | Sparsity | Actionability | Plausibility | Feasibility |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in final_vs_full250_df.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["setting"]),
                    str(row["method"]),
                    str(int(row["source_cases"])),
                    str(int(row["eval_cases"])),
                    str(int(row["dropped"])),
                    str(int(row["released_cf"])),
                    fmt_metric(row["release_per_eval"]),
                    fmt_metric(row["release_per_250"]),
                    fmt_metric(row["prox_jac"]),
                    fmt_metric(row["prox_euc"]),
                    fmt_metric(row["sparsity"]),
                    fmt_metric(row["actionability"]),
                    fmt_metric(row["plausibility"]),
                    fmt_metric(row["feasibility"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_summary_md(
    *,
    failed_cases_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
    field_summary_df: pd.DataFrame,
    insights: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# 7 Failed Cases Impact Check — Final Summary")
    lines.append("")
    lines.append("## 1. Tổng quan")
    lines.append("")
    lines.append(f"- Tổng số truy vấn nguồn: {EXPECTED_TOTAL_CASE_COUNT}")
    lines.append(f"- Số truy vấn được chấp nhận: {EXPECTED_ACCEPTED_CASE_COUNT}/{EXPECTED_TOTAL_CASE_COUNT}")
    lines.append(f"- Số truy vấn không được chấp nhận: {EXPECTED_FAILED_CASE_COUNT}/{EXPECTED_TOTAL_CASE_COUNT}")
    lines.append(f"- Tỷ lệ không được chấp nhận: {EXPECTED_FAILED_CASE_COUNT / EXPECTED_TOTAL_CASE_COUNT:.1%}")
    lines.append("")
    lines.append("## 2. Phân bố 7 failed cases")
    lines.append("")
    lines.append("| Nhóm | Số failed cases | Ghi chú |")
    lines.append("|---|---:|---|")
    for row in group_summary_df.to_dict(orient="records"):
        lines.append(f"| {row['group_id']} | {int(row['failed_case_count'])} | {row['note']} |")
    lines.append("")
    lines.append("| Fold | Số failed cases |")
    lines.append("|---|---:|")
    for row in fold_summary_df.to_dict(orient="records"):
        lines.append(f"| {row['fold_id']} | {int(row['failed_case_count'])} |")
    lines.append("")
    lines.append("| Trường lỗi | Số lần xuất hiện |")
    lines.append("|---|---:|")
    for row in field_summary_df.to_dict(orient="records"):
        lines.append(f"| {row['failed_field']} | {int(row['failed_case_count'])} |")
    lines.append("")
    lines.append("## 3. Tác động đến UFCE-FF")
    lines.append("")
    lines.append("| Phương pháp | Full-250 direct | Sau NLP accepted 243 | Số nghiệm mất | Failed cases có valid CF |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in method_summary_df.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {int(row['full250_direct_released_cf_count'])} | {int(row['accepted243_nl_first_released_cf_count'])} | "
            f"{int(row['lost_released_cf_count'])} | {int(row['failed_cases_with_valid_cf'])}/{EXPECTED_FAILED_CASE_COUNT} |"
        )
    lines.append("")
    lines.append("## 4. Insight chính")
    lines.append("")
    lines.append(f"- 7 failed cases chỉ làm mất `{insights['lost_released_cf_total']}` nghiệm UFCE-FF trên toàn bộ 250 truy vấn.")
    lines.append(
        f"- Tác động độ phủ lớn nhất nằm ở `{insights['dominant_method']}`"
        + (
            f" với `{insights['dominant_method_lost_count']}` nghiệm bị mất."
            if int(insights["dominant_method_lost_count"]) > 0
            else ", nhưng không ghi nhận nghiệm bị mất thêm."
        )
    )
    dominant_pattern = insights["dominant_pattern"]
    lines.append(
        f"- Lỗi tập trung chủ yếu ở `{dominant_pattern['language_group']} / {dominant_pattern['primary_failed_field']} / {dominant_pattern['failure_family']}`."
    )
    if insights["fold_concentration_abnormal"]:
        fold_label = ", ".join(insights["dominant_folds"])
        lines.append(
            f"- Có quan sát thấy lỗi tập trung theo fold `{fold_label}` với `{insights['dominant_fold_count']}/{EXPECTED_FAILED_CASE_COUNT}` failed cases."
        )
    else:
        lines.append("- Không quan sát thấy lỗi tập trung bất thường theo fold.")
    lines.append(
        f"- Cổng NLP có chi phí độ phủ `{insights['coverage_cost_label']}`, nhưng giúp ngăn việc sinh nghiệm phản thực trên hồ sơ chưa được tái dựng chính xác."
    )
    lines.append("")
    lines.append("## 5. Câu có thể đưa vào luận văn")
    lines.append("")
    lines.append(render_thesis_sentence(method_summary_df, str(insights["coverage_cost_label"])))
    lines.append("")
    return "\n".join(lines)


def render_thesis_sentence(method_summary_df: pd.DataFrame, coverage_cost_label: str) -> str:
    rows = {str(row["method"]): row for row in method_summary_df.to_dict(orient="records")}
    ufce1 = rows["UFCE1"]
    ufce2 = rows["UFCE2"]
    ufce3 = rows["UFCE3"]
    return (
        "Trên bảy truy vấn không được chấp nhận, việc tra ngược tuyến kiểm chứng cho thấy cổng bàn giao chỉ làm mất "
        f"`{int(ufce1['lost_released_cf_count'])}` nghiệm phản thực hợp lệ ở UFCE-FF1, "
        f"`{int(ufce2['lost_released_cf_count'])}` nghiệm ở UFCE-FF2 và "
        f"`{int(ufce3['lost_released_cf_count'])}` nghiệm ở UFCE-FF3. "
        f"Do đó, chi phí độ phủ của cổng bàn giao trên bộ Bank Loan là `{coverage_cost_label}`. "
        "Kết quả này cho thấy lớp chuẩn hóa ngôn ngữ tự nhiên chủ yếu đóng vai trò bảo vệ đầu vào: "
        "các truy vấn chưa đủ điều kiện được dừng trước khi gọi UFCE-FF, trong khi hành vi tổng hợp của lõi phản thực "
        "trên phần truy vấn được chấp nhận vẫn gần với tuyến kiểm chứng."
    )


if __name__ == "__main__":
    raise SystemExit(main())
