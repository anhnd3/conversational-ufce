from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_case_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_alias",
        "case_id",
        "group",
        "repeat_id",
        "valid_json",
        "schema_valid",
        "exact_match",
        "field_accuracy",
        "status_correct",
        "missing_fields_correct",
        "conflicts_correct",
        "hallucination_count",
        "stability_case",
        "request_latency_ms",
        "ttft_ms",
        "total_time_ms",
        "tokens_per_second",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "http_status_code",
        "api_error",
        "parse_error",
        "validation_error_count",
        "validation_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
