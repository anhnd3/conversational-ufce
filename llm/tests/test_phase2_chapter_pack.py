from __future__ import annotations

import json
from pathlib import Path

from llm.src.conversation.parser_adapter import DEFAULT_BENCHMARK_PATH, DEFAULT_SCHEMA_PATH, DEFAULT_SYSTEM_PROMPT_PATH
from scripts.build_phase2_chapter_pack import main as build_chapter_pack_main


def write_turn(folder: Path, *, stage: str, summary_type: str | None, user_input: str, parser_status: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "user_input.txt").write_text(user_input + "\n", encoding="utf-8")
    (folder / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "turn_id": folder.name,
                "stage": stage,
                "model_alias": "qwen/qwen3-14b",
                "timestamp_utc": "2026-03-21T00:00:00Z",
                "command": "python demo.py",
                "session_id": None,
                "turn_index": 1,
                "parent_turn_id": None,
                "merge_applied": False,
                "carried_fields": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    turn_result = {
        "stage": stage,
        "normalized_parse": {"status": parser_status},
        "explanation_payload": None if summary_type is None else {"summary_type": summary_type},
    }
    (folder / "turn_result.json").write_text(json.dumps(turn_result) + "\n", encoding="utf-8")


def test_build_phase2_chapter_pack_generates_all_files(tmp_path):
    pack_root = tmp_path / "phase2_pack_demo"
    accepted_root = pack_root / "accepted"
    primary_root = accepted_root / "primary"
    supplemental_root = accepted_root / "supplemental"
    nr_case = primary_root / "P-NR-01__no_recourse"
    cf_case = primary_root / "P-CF-01__counterfactual"
    supplemental_case = supplemental_root / "S-MERGE-01__merge_to_success"
    write_turn(
        nr_case,
        stage="RUNTIME_SUCCESS",
        summary_type="no_recourse_needed",
        user_input="Income 140 ...",
        parser_status="complete",
    )
    write_turn(
        cf_case,
        stage="RUNTIME_SUCCESS",
        summary_type="counterfactual_found",
        user_input="Income 55 ...",
        parser_status="complete",
    )
    write_turn(
        supplemental_case / "turn1",
        stage="NEEDS_CLARIFICATION",
        summary_type=None,
        user_input="Turn1 text",
        parser_status="partial",
    )
    write_turn(
        supplemental_case / "turn2",
        stage="RUNTIME_SUCCESS",
        summary_type="counterfactual_found",
        user_input="Turn2 text",
        parser_status="complete",
    )
    turn2_manifest = json.loads((supplemental_case / "turn2" / "artifact_manifest.json").read_text(encoding="utf-8"))
    turn2_manifest["merge_applied"] = True
    turn2_manifest["parent_turn_id"] = "turn1"
    turn2_manifest["carried_fields"] = ["Income", "CCAvg", "Family", "Education", "Mortgage"]
    (supplemental_case / "turn2" / "artifact_manifest.json").write_text(
        json.dumps(turn2_manifest) + "\n",
        encoding="utf-8",
    )

    (pack_root / "pack_status.json").write_text(
        json.dumps({"pack_version": "phase2_pack_demo", "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    (pack_root / "attempt_summary.json").write_text(
        json.dumps({"stability_filter_results": [{"case_id": "P-NR-01", "accepted": True}]}) + "\n",
        encoding="utf-8",
    )
    (pack_root / "phase2_pack_manifest.json").write_text(
        json.dumps(
            {
                "pack_version": "phase2_pack_demo",
                "provenance": {
                    "pack_version": "phase2_pack_demo",
                    "scenario_catalog_version": "phase2_bank_catalog_v1",
                    "model_alias": "qwen/qwen3-14b",
                    "benchmark_path": str(DEFAULT_BENCHMARK_PATH.resolve()),
                    "system_prompt_path": str(DEFAULT_SYSTEM_PROMPT_PATH.resolve()),
                    "parser_schema_path": str(DEFAULT_SCHEMA_PATH.resolve()),
                    "parser_schema_version": "v1",
                    "prompt_template_version": "parser_system_prompt_v1",
                    "scenario_catalog_path": str((Path.cwd() / "docs" / "thesis" / "part2" / "catalogs" / "phase2_bank_catalog_v1.json").resolve()),
                    "accepted_root": str(accepted_root.resolve()),
                },
                "attempted_counts": {"primary_cases": 8, "primary_runs": 16, "supplemental_cases": 2},
                "accepted_counts": {
                    "primary_cases": 8,
                    "primary_by_label": {
                        "no_recourse_needed": 2,
                        "counterfactual_found": 2,
                        "runtime_reject": 2,
                        "clarification": 2
                    },
                    "supplemental_cases": 2
                },
                "primary_acceptance_target": {
                    "no_recourse_needed": 2,
                    "counterfactual_found": 2,
                    "runtime_reject": 2,
                    "clarification": 2
                },
                "accepted_primary_cases": [
                    {"case_id": "P-NR-01", "expected_label": "no_recourse_needed", "folder": str(nr_case.resolve())},
                    {"case_id": "P-CF-01", "expected_label": "counterfactual_found", "folder": str(cf_case.resolve())}
                ],
                "accepted_supplemental_demos": [
                    {
                        "case_id": "S-MERGE-01",
                        "expected_label": "supplemental_followup",
                        "supplemental_type": "supplemental_followup_merge_to_success",
                        "folder": str(supplemental_case.resolve())
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "generated"
    exit_code = build_chapter_pack_main(["--pack-root", str(pack_root), "--output-root", str(output_root)])

    assert exit_code == 0
    generated_root = output_root / "phase2_pack_demo"
    assert (generated_root / "phase2_chapter_pack.json").exists()
    assert (generated_root / "phase2_chapter_pack.md").exists()
    assert (generated_root / "part2_system_diagram.mmd").exists()
    assert (generated_root / "phase2_case_examples.md").exists()

    chapter_json = json.loads((generated_root / "phase2_chapter_pack.json").read_text(encoding="utf-8"))
    assert chapter_json["pack_version"] == "phase2_pack_demo"
    assert "contracts" in chapter_json
    assert "runtime_reason_codes" in chapter_json
    assert len(chapter_json["worked_examples"]) == 3
