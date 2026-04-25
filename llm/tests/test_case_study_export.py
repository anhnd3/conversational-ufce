from __future__ import annotations

import json

from scripts.export_part2_case_studies import collect_case_studies, render_markdown


def write_case(root, name, *, manifest, turn_result):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "artifact_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (folder / "turn_result.json").write_text(json.dumps(turn_result) + "\n", encoding="utf-8")
    return folder


def test_collect_case_studies_uses_summary_type_and_session_metadata(tmp_path):
    write_case(
        tmp_path,
        "case_success",
        manifest={
            "turn_id": "run_001",
            "stage": "RUNTIME_SUCCESS",
            "model_alias": "qwen/qwen3-14b",
            "timestamp_utc": "2026-03-20T10:00:00Z",
            "command": "python demo.py",
            "session_id": "session_001",
            "turn_index": 1,
            "parent_turn_id": None,
            "merge_applied": False,
            "carried_fields": [],
        },
        turn_result={
            "explanation_payload": {
                "summary_type": "counterfactual_found",
            }
        },
    )

    records = collect_case_studies(tmp_path)

    assert len(records) == 1
    assert records[0]["case_label"] == "counterfactual_found"
    assert records[0]["summary_type"] == "counterfactual_found"
    assert records[0]["session_id"] == "session_001"
    assert records[0]["turn_index"] == 1
    markdown = render_markdown(records)
    assert "| Turn ID | Stage | Category | Model | Timestamp | Folder |" in markdown
    assert "counterfactual_found" in markdown


def test_collect_case_studies_marks_supplemental_followup(tmp_path):
    write_case(
        tmp_path / "supplemental" / "S-MERGE-01__demo",
        "turn2",
        manifest={
            "turn_id": "run_002",
            "stage": "RUNTIME_SUCCESS",
            "model_alias": "qwen/qwen3-14b",
            "timestamp_utc": "2026-03-20T10:05:00Z",
            "command": "python demo.py",
            "session_id": "supplemental_demo",
            "turn_index": 2,
            "parent_turn_id": "run_001",
            "merge_applied": True,
            "carried_fields": ["Income", "Family"],
        },
        turn_result={
            "explanation_payload": {
                "summary_type": "no_recourse_needed",
            }
        },
    )

    records = collect_case_studies(tmp_path)

    assert len(records) == 1
    assert records[0]["case_label"] == "supplemental_followup"
    assert records[0]["case_id"] == "S-MERGE-01"
    assert records[0]["merge_applied"] is True
    assert records[0]["carried_fields"] == ["Income", "Family"]
