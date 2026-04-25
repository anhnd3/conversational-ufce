#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.phase2.metadata import (  # noqa: E402
    build_artifact_schema_table,
    build_contract_metadata,
    build_conversation_stage_table,
    build_explanation_summary_type_table,
    build_limitations_block,
    build_runtime_reason_code_table,
    build_system_diagram,
)
from llm.src.phase2.taxonomy import MANDATORY_WORKED_EXAMPLE_BUCKETS  # noqa: E402
from llm.src.utils.io import write_json  # noqa: E402
from scripts.export_part2_case_studies import collect_case_studies  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate chapter-ready Phase 2 assets from a completed pack.")
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output-root", default="docs/thesis/part2/generated")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pack_root = Path(args.pack_root)
    manifest = load_complete_pack(pack_root)
    pack_version = str(manifest["pack_version"])
    output_root = Path(args.output_root) / pack_version
    output_root.mkdir(parents=True, exist_ok=True)

    chapter_pack = build_chapter_pack(pack_root=pack_root, manifest=manifest)
    write_json(output_root / "phase2_chapter_pack.json", chapter_pack)
    (output_root / "phase2_chapter_pack.md").write_text(render_chapter_markdown(chapter_pack), encoding="utf-8")
    (output_root / "part2_system_diagram.mmd").write_text(chapter_pack["system_diagram"] + "\n", encoding="utf-8")
    (output_root / "phase2_case_examples.md").write_text(
        render_case_examples_markdown(chapter_pack["worked_examples"]),
        encoding="utf-8",
    )

    print(json.dumps({"pack_version": pack_version, "output_root": str(output_root.resolve())}, ensure_ascii=True, indent=2))
    return 0


def load_complete_pack(pack_root: Path) -> dict[str, Any]:
    pack_status_path = pack_root / "pack_status.json"
    manifest_path = pack_root / "phase2_pack_manifest.json"
    if not pack_status_path.exists():
        raise ValueError(f"Missing pack_status.json in {pack_root}")
    if not manifest_path.exists():
        raise ValueError(f"Missing phase2_pack_manifest.json in {pack_root}")

    pack_status = json.loads(pack_status_path.read_text(encoding="utf-8"))
    if pack_status.get("status") != "complete":
        raise ValueError(f"Pack is not complete: {pack_status.get('status')!r}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_chapter_pack(*, pack_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    provenance = dict(manifest["provenance"])
    accepted_root = Path(provenance["accepted_root"])
    attempt_summary_path = pack_root / "attempt_summary.json"
    attempt_summary = json.loads(attempt_summary_path.read_text(encoding="utf-8")) if attempt_summary_path.exists() else {}
    accepted_records = collect_case_studies(accepted_root)
    contracts = build_contract_metadata(
        benchmark_path=Path(provenance["benchmark_path"]),
        parser_schema_path=Path(provenance["parser_schema_path"]),
    )
    worked_examples = select_worked_examples(manifest, accepted_records)
    return {
        "pack_version": manifest["pack_version"],
        "provenance": provenance,
        "system_diagram": build_system_diagram(),
        "contracts": contracts,
        "conversation_stages": build_conversation_stage_table(),
        "runtime_reason_codes": build_runtime_reason_code_table(),
        "explanation_summary_types": build_explanation_summary_type_table(),
        "artifact_schema": build_artifact_schema_table(),
        "evidence_metrics": build_evidence_metrics(manifest, attempt_summary),
        "accepted_primary_cases": list(manifest.get("accepted_primary_cases", [])),
        "supplemental_demos": list(manifest.get("accepted_supplemental_demos", [])),
        "worked_examples": worked_examples,
        "limitations": build_limitations_block(),
    }


def build_evidence_metrics(manifest: dict[str, Any], attempt_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempted_counts": dict(manifest.get("attempted_counts", {})),
        "accepted_counts": dict(manifest.get("accepted_counts", {})),
        "primary_acceptance_target": dict(manifest.get("primary_acceptance_target", {})),
        "stability_filter_results": list(attempt_summary.get("stability_filter_results", [])),
    }


def select_worked_examples(manifest: dict[str, Any], accepted_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_cases = list(manifest.get("accepted_primary_cases", []))
    supplemental_cases = list(manifest.get("accepted_supplemental_demos", []))
    examples: list[dict[str, Any]] = []
    required_buckets = set(MANDATORY_WORKED_EXAMPLE_BUCKETS)

    for case in primary_cases:
        folder = Path(case["folder"])
        turn_result = load_turn_result(folder)
        summary_type = extract_summary_type(turn_result)
        if summary_type not in required_buckets:
            continue
        if any(item["bucket"] == summary_type for item in examples):
            continue
        examples.append(
            {
                "bucket": summary_type,
                "case_id": case["case_id"],
                "user_input": load_user_input(folder),
                "parser_status": extract_parser_status(turn_result),
                "final_stage": str(turn_result.get("stage", "")),
                "explanation_summary_type": summary_type,
                "artifact_folder": str(folder.resolve()),
            }
        )

    for case in supplemental_cases:
        if case.get("supplemental_type") != "supplemental_followup_merge_to_success":
            continue
        bucket = "supplemental_followup_merge_to_success"
        if any(item["bucket"] == bucket for item in examples):
            continue
        case_root = Path(case["folder"])
        turn1 = case_root / "turn1"
        turn2 = case_root / "turn2"
        turn2_result = load_turn_result(turn2)
        examples.append(
            {
                "bucket": bucket,
                "case_id": case["case_id"],
                "user_input": "\n".join(
                    [
                        "Turn 1: " + load_user_input(turn1),
                        "Turn 2: " + load_user_input(turn2),
                    ]
                ),
                "parser_status": extract_parser_status(turn2_result),
                "final_stage": str(turn2_result.get("stage", "")),
                "explanation_summary_type": extract_summary_type(turn2_result),
                "artifact_folder": str(turn2.resolve()),
            }
        )

    found_buckets = {item["bucket"] for item in examples}
    missing = [bucket for bucket in MANDATORY_WORKED_EXAMPLE_BUCKETS if bucket not in found_buckets]
    if missing:
        raise ValueError(f"Missing mandatory worked example buckets: {missing}")

    examples.sort(key=lambda item: MANDATORY_WORKED_EXAMPLE_BUCKETS.index(item["bucket"]))
    return examples


def render_chapter_markdown(chapter_pack: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 Chapter Pack",
        "",
        f"- pack_version: {chapter_pack['pack_version']}",
        f"- scenario_catalog_version: {chapter_pack['provenance']['scenario_catalog_version']}",
        f"- model_alias: {chapter_pack['provenance']['model_alias']}",
        "",
        "## Implemented System Summary",
        "",
        "The implemented Part II system remains a bank-only conversational UFCE MVP.",
        "The local model is used as a structured parser only; validation, runtime execution, clarification, explanation, and artifact persistence remain deterministic backend responsibilities.",
        "",
        "## Frozen Contract Table",
        "",
        "| Field | Type | Description |",
        "| --- | --- | --- |",
    ]
    for field in chapter_pack["contracts"]["parser_output"]["target_cf_fields"]:
        lines.append(f"| {field['name']} | {field['type']} | {field['description']} |")

    lines.extend(
        [
            "",
            "## Conversation Stages",
            "",
            "| Stage | Layer |",
            "| --- | --- |",
        ]
    )
    for item in chapter_pack["conversation_stages"]:
        lines.append(f"| {item['stage']} | {item['layer']} |")

    lines.extend(
        [
            "",
            "## Runtime Reason Codes",
            "",
            "| Code | Layer |",
            "| --- | --- |",
        ]
    )
    for item in chapter_pack["runtime_reason_codes"]:
        lines.append(f"| {item['code']} | {item['layer']} |")

    lines.extend(
        [
            "",
            "## Explanation Summary Types",
            "",
            "| Summary Type | Layer |",
            "| --- | --- |",
        ]
    )
    for item in chapter_pack["explanation_summary_types"]:
        lines.append(f"| {item['summary_type']} | {item['layer']} |")

    lines.extend(
        [
            "",
            "## Artifact Schema",
            "",
            "| File | Presence | Description |",
            "| --- | --- | --- |",
        ]
    )
    for item in chapter_pack["artifact_schema"]:
        lines.append(f"| {item['file']} | {item['presence']} | {item['description']} |")

    metrics = chapter_pack["evidence_metrics"]
    lines.extend(
        [
            "",
            "## Evidence Metrics",
            "",
            f"- attempted primary cases: {metrics['attempted_counts'].get('primary_cases', 0)}",
            f"- attempted primary runs: {metrics['attempted_counts'].get('primary_runs', 0)}",
            f"- attempted supplemental cases: {metrics['attempted_counts'].get('supplemental_cases', 0)}",
            f"- accepted primary cases: {metrics['accepted_counts'].get('primary_cases', 0)}",
            f"- accepted supplemental demos: {metrics['accepted_counts'].get('supplemental_cases', 0)}",
            "",
            "Accepted primary distribution:",
            "",
        ]
    )
    for label, count in metrics["accepted_counts"].get("primary_by_label", {}).items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Accepted Primary Case Inventory",
            "",
        ]
    )
    for case in chapter_pack["accepted_primary_cases"]:
        lines.append(f"- {case['case_id']}: {case['expected_label']} -> `{case['folder']}`")

    lines.extend(
        [
            "",
            "## Supplemental Demo Inventory",
            "",
        ]
    )
    for case in chapter_pack["supplemental_demos"]:
        lines.append(
            f"- {case['case_id']}: {case['supplemental_type']} -> `{case['folder']}`"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for item in chapter_pack["limitations"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Worked Examples",
            "",
        ]
    )
    for item in chapter_pack["worked_examples"]:
        lines.append(
            f"- {item['bucket']} / {item['case_id']}: parser_status={item['parser_status']}, final_stage={item['final_stage']}, explanation_summary_type={item['explanation_summary_type']}, artifact_folder=`{item['artifact_folder']}`"
        )
    lines.append("")
    return "\n".join(lines)


def render_case_examples_markdown(examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2 Worked Case Examples",
        "",
    ]
    for example in examples:
        lines.extend(
            [
                f"## {example['bucket']} / {example['case_id']}",
                "",
                "Input:",
                "",
                "```text",
                example["user_input"],
                "```",
                "",
                f"- parser_status: {example['parser_status']}",
                f"- final_stage: {example['final_stage']}",
                f"- explanation_summary_type: {example['explanation_summary_type']}",
                f"- artifact_folder: `{example['artifact_folder']}`",
                "",
            ]
        )
    return "\n".join(lines)


def load_turn_result(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "turn_result.json").read_text(encoding="utf-8"))


def load_user_input(folder: Path) -> str:
    return (folder / "user_input.txt").read_text(encoding="utf-8").strip()


def extract_parser_status(turn_result: dict[str, Any]) -> str:
    normalized_parse = turn_result.get("normalized_parse")
    if isinstance(normalized_parse, dict):
        value = normalized_parse.get("status")
        if isinstance(value, str):
            return value
    return ""


def extract_summary_type(turn_result: dict[str, Any]) -> str | None:
    payload = turn_result.get("explanation_payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get("summary_type")
    return str(value) if isinstance(value, str) and value else None


if __name__ == "__main__":
    raise SystemExit(main())
