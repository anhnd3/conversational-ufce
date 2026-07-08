from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm.src.utils.io import write_json, write_jsonl


def test_write_json_coerces_pandas_and_numpy_scalars(tmp_path: Path) -> None:
    out_path = tmp_path / "summary.json"
    payload = {
        "query_count": pd.Series([250], dtype="int64").iloc[0],
        "ready_rate": pd.Series([0.4], dtype="float64").iloc[0],
        "generated_at": pd.Timestamp("2026-07-02T14:00:00"),
        "artifacts": {
            "summary_json": out_path,
        },
    }

    write_json(out_path, payload)

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["query_count"] == 250
    assert abs(saved["ready_rate"] - 0.4) < 1e-12
    assert saved["generated_at"] == "2026-07-02T14:00:00"
    assert saved["artifacts"]["summary_json"] == str(out_path)


def test_write_jsonl_coerces_pandas_and_numpy_scalars(tmp_path: Path) -> None:
    out_path = tmp_path / "rows.jsonl"
    rows = [
        {
            "case_id": "c1",
            "ready_for_runtime": pd.Series([1], dtype="int64").iloc[0],
            "exact_reconstruction": pd.Series([True], dtype="bool").iloc[0],
        }
    ]

    write_jsonl(out_path, rows)

    saved = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert saved == [{"case_id": "c1", "ready_for_runtime": 1, "exact_reconstruction": True}]
