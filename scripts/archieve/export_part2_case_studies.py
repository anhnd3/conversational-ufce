#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.src.phase2.taxonomy import (
    CLARIFICATION,
    CONFLICT,
    NO_RECOURSE_NEEDED,
    COUNTERFACTUAL_FOUND,
    PARSER_FAILURE,
    RUNTIME_REJECT,
    SUPPLEMENTAL_FOLLOWUP,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a manifest-driven index of Part II case studies.")
    parser.add_argument("--input-root", default="outputs/conversations", help="Root folder containing conversation run directories.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-output", default="", help="Optional Markdown output path.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root = Path(args.input_root)
    records = collect_case_studies(input_root)
    json_output = Path(args.output) if args.output else input_root / "case_studies_index.json"
    markdown_output = (
        Path(args.markdown_output) if args.markdown_output else input_root / "case_studies_index.md"
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(records), encoding="utf-8")
    print(f"[OUT] {json_output}")
    print(f"[OUT] {markdown_output}")
    return 0


def collect_case_studies(input_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not input_root.exists():
        return records
    for artifact_path in sorted(input_root.rglob("artifact_manifest.json")):
        manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        folder = artifact_path.parent
        turn_result = load_turn_result(folder)
        case_label = classify_case_study(manifest, turn_result, folder)
        summary_type = extract_summary_type(turn_result)
        record = {
            "turn_id": str(manifest.get("turn_id", "")),
            "stage": str(manifest.get("stage", "")),
            "case_label": case_label,
            "summary_type": summary_type,
            "model_alias": str(manifest.get("model_alias", "")),
            "timestamp_utc": str(manifest.get("timestamp_utc", "")),
            "command": str(manifest.get("command", "")),
            "folder": str(folder.resolve()),
            "manifest_path": str(artifact_path.resolve()),
            "case_id": extract_case_id(folder),
            "session_id": manifest.get("session_id"),
            "turn_index": manifest.get("turn_index"),
            "parent_turn_id": manifest.get("parent_turn_id"),
            "merge_applied": bool(manifest.get("merge_applied", False)),
            "carried_fields": list(manifest.get("carried_fields", [])),
        }
        records.append(record)
    return records


def render_markdown(records: list[dict[str, object]]) -> str:
    lines = [
        "# Part II Case Studies",
        "",
        "| Turn ID | Stage | Category | Model | Timestamp | Folder |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | `{5}` |".format(
                record["turn_id"],
                record["stage"],
                record["case_label"],
                record["model_alias"],
                record["timestamp_utc"],
                record["folder"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def load_turn_result(folder: Path) -> dict[str, object]:
    turn_result_path = folder / "turn_result.json"
    if not turn_result_path.exists():
        return {}
    return json.loads(turn_result_path.read_text(encoding="utf-8"))


def classify_case_study(manifest: dict[str, object], turn_result: dict[str, object], folder: Path) -> str:
    if is_supplemental_followup(manifest, folder):
        return SUPPLEMENTAL_FOLLOWUP
    summary_type = extract_summary_type(turn_result)
    if summary_type in {NO_RECOURSE_NEEDED, COUNTERFACTUAL_FOUND, RUNTIME_REJECT}:
        return summary_type
    stage = str(manifest.get("stage", ""))
    if stage == "NEEDS_CLARIFICATION":
        return CLARIFICATION
    if stage == "CONFLICT":
        return CONFLICT
    if stage == "PARSER_FAILURE":
        return PARSER_FAILURE
    return stage.lower() if stage else "unknown"


def extract_summary_type(turn_result: dict[str, object]) -> str | None:
    payload = turn_result.get("explanation_payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get("summary_type")
    return str(value) if isinstance(value, str) and value else None


def is_supplemental_followup(manifest: dict[str, object], folder: Path) -> bool:
    if manifest.get("parent_turn_id"):
        return True
    if bool(manifest.get("merge_applied")):
        return True
    parts = {folder.name}
    if folder.parent != folder:
        parts.add(folder.parent.name)
    return any("supplemental_" in part or part.startswith("S-MERGE-") for part in parts)


def extract_case_id(folder: Path) -> str | None:
    for candidate in (folder.name, folder.parent.name):
        if "__" in candidate:
            prefix = candidate.split("__", 1)[0]
            if prefix:
                return prefix
    return None


if __name__ == "__main__":
    raise SystemExit(main())
