from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm.src.conversation.types import ArtifactRecord, ConversationTurnResult, serialize_normalized_parse_payload
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_folder_timestamp


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "conversations"


def save_conversation_artifacts(
    turn_result: ConversationTurnResult,
    *,
    output_root: Path | None = None,
    scenario_slug: str | None = None,
    command: str,
    config_snapshot: dict[str, Any],
    debug_trace_enabled: bool,
    session_trace: dict[str, Any] | None = None,
) -> ArtifactRecord:
    root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    folder_timestamp = local_now_folder_timestamp()
    folder_slug = normalize_scenario_slug(scenario_slug or turn_result.user_input)
    turn_dir = allocate_turn_dir(root, folder_timestamp, folder_slug)

    saved_files = build_saved_file_list(turn_result, debug_trace_enabled=debug_trace_enabled)
    write_text(turn_dir / "user_input.txt", turn_result.user_input)
    write_text(turn_dir / "parser_raw_output.txt", turn_result.parser_result.message_text)
    write_json(turn_dir / "parser_call.json", turn_result.parser_result.to_dict())
    write_json(
        turn_dir / "normalized_parse.json",
        serialize_normalized_parse_payload(
            turn_result.normalized_parse,
            turn_result.field_provenance,
            turn_result.parser_quality_metadata,
        ),
    )
    write_json(turn_dir / "schema_validation.json", turn_result.schema_validation)
    write_json(turn_dir / "canonical_validation.json", turn_result.canonical_validation)
    write_json(
        turn_dir / "builder_result.json",
        None if turn_result.builder_result is None else turn_result.builder_result.to_dict(),
    )
    write_json(
        turn_dir / "negotiation_transition.json",
        None if turn_result.negotiation_transition is None else turn_result.negotiation_transition.to_dict(),
    )
    write_json(turn_dir / "runtime_result.json", turn_result.runtime_result)
    if turn_result.runtime_debug_trace is not None:
        write_json(turn_dir / "runtime_debug_trace.json", turn_result.runtime_debug_trace)
    if turn_result.invariant_validation is not None:
        write_json(turn_dir / "invariant_validation.json", turn_result.invariant_validation)
    write_json(
        turn_dir / "clarification_payload.json",
        None if turn_result.clarification_payload is None else turn_result.clarification_payload.to_dict(),
    )
    write_json(
        turn_dir / "explanation_payload.json",
        None if turn_result.explanation_payload is None else turn_result.explanation_payload.to_dict(),
    )
    if isinstance(turn_result.runtime_result, dict):
        canonical_request = turn_result.runtime_result.get("canonical_request")
        canonical_candidates = turn_result.runtime_result.get("canonical_candidates")
        verification_results = turn_result.runtime_result.get("verification_results")
        backend_manifest = turn_result.runtime_result.get("backend_manifest")
        reason_code_version = turn_result.runtime_result.get("reason_code_version")
        if canonical_request is not None:
            write_json(turn_dir / "canonical_request.json", canonical_request)
        if canonical_candidates is not None:
            write_json(turn_dir / "canonical_candidates.json", canonical_candidates)
        if verification_results is not None:
            write_json(turn_dir / "verification_results.json", verification_results)
        if backend_manifest is not None or reason_code_version is not None:
            write_json(
                turn_dir / "backend_contract.json",
                {
                    "backend_id": turn_result.runtime_result.get("backend_id"),
                    "backend_manifest": backend_manifest,
                    "reason_code_version": reason_code_version,
                },
            )
    write_text(turn_dir / "response_text.txt", turn_result.response_text)
    if turn_result.timing_metrics is not None:
        write_json(turn_dir / "timing_metrics.json", turn_result.timing_metrics)

    if turn_result.repair_result is not None:
        write_text(turn_dir / "repair_raw_output.txt", turn_result.repair_result.message_text)
        write_json(turn_dir / "repair_call.json", turn_result.repair_result.to_dict())

    if turn_result.turn_kind == "refinement":
        write_json(turn_dir / "refinement_parser_call.json", turn_result.parser_result.to_dict())
        write_json(turn_dir / "refinement_delta.json", turn_result.constraint_feedback_delta)
        write_json(turn_dir / "refinement_state_before.json", turn_result.active_constraint_spec_before)
        write_json(turn_dir / "refinement_state_after.json", turn_result.active_constraint_spec)
        write_json(
            turn_dir / "refinement_result.json",
            {
                "refinement_status": turn_result.refinement_status,
                "refinement_revision_index": turn_result.refinement_revision_index,
                "parent_terminal_turn_id": turn_result.parent_terminal_turn_id,
                "parent_refinement_revision_index": turn_result.parent_refinement_revision_index,
                "refinement_rounds_used": turn_result.refinement_rounds_used,
                "refinement_round_limit": turn_result.refinement_round_limit,
                "public_state": None
                if turn_result.response_decision is None
                else turn_result.response_decision.final_public_state,
                "response_text": turn_result.response_text,
            },
        )

    if debug_trace_enabled and turn_result.runtime_debug_trace is not None:
        write_json(turn_dir / "debug_trace.json", turn_result.runtime_debug_trace)

    config_snapshot = dict(config_snapshot)
    config_snapshot["output_dir"] = str(turn_dir.resolve())
    config_snapshot["output_dir_sha256"] = sha256_file(turn_dir / "response_text.txt")
    write_json(turn_dir / "config_snapshot.json", config_snapshot)

    artifact_record = ArtifactRecord(
        turn_id=turn_result.turn_id,
        stage=turn_result.stage,
        output_dir=str(turn_dir.resolve()),
        saved_files=list(saved_files),
        model_alias=turn_result.model_alias,
        command=command,
        timestamp_utc=turn_result.timestamp_utc,
        debug_trace_enabled=debug_trace_enabled,
        repair_used=turn_result.repair_result is not None,
        session_id=extract_session_id(session_trace),
        turn_index=extract_turn_index(session_trace),
        parent_turn_id=extract_parent_turn_id(session_trace),
        merge_applied=extract_merge_applied(session_trace),
        carried_fields=extract_carried_fields(session_trace),
    )
    turn_result.artifact_record = artifact_record
    write_json(turn_dir / "turn_result.json", turn_result.to_dict())
    write_json(turn_dir / "artifact_manifest.json", artifact_record.to_dict())
    return artifact_record


def build_saved_file_list(
    turn_result: ConversationTurnResult,
    *,
    debug_trace_enabled: bool,
) -> list[str]:
    files = [
        "artifact_manifest.json",
        "builder_result.json",
        "canonical_validation.json",
        "clarification_payload.json",
        "config_snapshot.json",
        "explanation_payload.json",
        "negotiation_transition.json",
        "normalized_parse.json",
        "parser_call.json",
        "parser_raw_output.txt",
        "response_text.txt",
        "runtime_result.json",
        "schema_validation.json",
        "turn_result.json",
        "user_input.txt",
    ]
    if turn_result.invariant_validation is not None:
        files.append("invariant_validation.json")
    if turn_result.runtime_debug_trace is not None:
        files.append("runtime_debug_trace.json")
    if isinstance(turn_result.runtime_result, dict):
        if turn_result.runtime_result.get("canonical_request") is not None:
            files.append("canonical_request.json")
        if turn_result.runtime_result.get("canonical_candidates") is not None:
            files.append("canonical_candidates.json")
        if turn_result.runtime_result.get("verification_results") is not None:
            files.append("verification_results.json")
        if (
            turn_result.runtime_result.get("backend_manifest") is not None
            or turn_result.runtime_result.get("reason_code_version") is not None
        ):
            files.append("backend_contract.json")
    if turn_result.repair_result is not None:
        files.extend(["repair_call.json", "repair_raw_output.txt"])
    if debug_trace_enabled and turn_result.runtime_debug_trace is not None:
        files.append("debug_trace.json")
    if turn_result.timing_metrics is not None:
        files.append("timing_metrics.json")
    if turn_result.turn_kind == "refinement":
        files.extend(
            [
                "refinement_delta.json",
                "refinement_parser_call.json",
                "refinement_result.json",
                "refinement_state_after.json",
                "refinement_state_before.json",
            ]
        )
    return sorted(files)


def allocate_turn_dir(root: Path, folder_timestamp: str, folder_slug: str) -> Path:
    candidate = root / f"{folder_timestamp}_{folder_slug}"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    suffix = 1
    while True:
        retry = root / f"{folder_timestamp}_{folder_slug}_{suffix:02d}"
        if not retry.exists():
            retry.mkdir(parents=True, exist_ok=False)
            return retry
        suffix += 1


def normalize_scenario_slug(value: str, max_length: int = 48) -> str:
    collapsed = "_".join(value.lower().split())
    slug = re.sub(r"[^a-z0-9._-]+", "_", collapsed).strip("._-")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        return "bank_turn"
    return slug[:max_length].rstrip("._-") or "bank_turn"


def write_text(path: Path, text: str) -> None:
    path.write_text(text + ("" if text.endswith("\n") or not text else "\n"), encoding="utf-8")


def extract_session_id(session_trace: dict[str, Any] | None) -> str | None:
    if not isinstance(session_trace, dict):
        return None
    value = session_trace.get("session_id")
    return str(value) if isinstance(value, str) and value else None


def extract_turn_index(session_trace: dict[str, Any] | None) -> int | None:
    if not isinstance(session_trace, dict):
        return None
    value = session_trace.get("turn_index")
    return int(value) if isinstance(value, int) and value >= 1 else None


def extract_parent_turn_id(session_trace: dict[str, Any] | None) -> str | None:
    if not isinstance(session_trace, dict):
        return None
    value = session_trace.get("parent_turn_id")
    return str(value) if isinstance(value, str) and value else None


def extract_merge_applied(session_trace: dict[str, Any] | None) -> bool:
    if not isinstance(session_trace, dict):
        return False
    return bool(session_trace.get("merge_applied"))


def extract_carried_fields(session_trace: dict[str, Any] | None) -> list[str]:
    if not isinstance(session_trace, dict):
        return []
    value = session_trace.get("carried_fields")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
