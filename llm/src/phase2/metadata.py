from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from llm.src.conversation.canonical_validator import BankCanonicalValidator
from llm.src.conversation.types import ConversationStage
from llm.src.orchestration.clarification_flow import order_fields
from llm.src.phase2.taxonomy import EXPLANATION_SUMMARY_TYPES
from llm.src.runtime.reason_codes import ALL_REASON_CODES
from llm.src.utils.hashing import utc_now_iso


ROOT = Path(__file__).resolve().parents[3]
MODEL_TRANSPORT = "lmstudio_chat_completions_json_schema"
EVIDENCE_RUNNER_VERSION = "phase2_pack_runner_v1"
CHAPTER_GENERATOR_VERSION = "phase2_chapter_generator_v1"

ARTIFACT_SCHEMA = (
    {
        "file": "user_input.txt",
        "presence": "always_present",
        "description": "Raw user request text for the turn.",
    },
    {
        "file": "parser_raw_output.txt",
        "presence": "always_present",
        "description": "Parser text payload returned by the local model.",
    },
    {
        "file": "parser_call.json",
        "presence": "always_present",
        "description": "Structured parser transport request/response metadata.",
    },
    {
        "file": "normalized_parse.json",
        "presence": "always_present",
        "description": "Normalized bank parser candidate after parse/repair handling.",
    },
    {
        "file": "schema_validation.json",
        "presence": "always_present",
        "description": "Schema validation result against the frozen bank parser contract.",
    },
    {
        "file": "canonical_validation.json",
        "presence": "always_present",
        "description": "Canonical bank validation and runtime readiness result.",
    },
    {
        "file": "builder_result.json",
        "presence": "always_present",
        "description": "Conversation request-builder result with readiness, reasons, and canonical snapshot metadata.",
    },
    {
        "file": "negotiation_transition.json",
        "presence": "always_present",
        "description": "Conversation negotiation transition trace for the turn.",
    },
    {
        "file": "runtime_result.json",
        "presence": "always_present",
        "description": "Deterministic backend runtime output, including reason codes.",
    },
    {
        "file": "clarification_payload.json",
        "presence": "always_present",
        "description": "Clarification payload file; may contain null when the turn ended elsewhere.",
    },
    {
        "file": "explanation_payload.json",
        "presence": "always_present",
        "description": "Explanation payload file; may contain null when the turn did not reach runtime explanation.",
    },
    {
        "file": "response_text.txt",
        "presence": "always_present",
        "description": "Final user-facing response text for the turn.",
    },
    {
        "file": "config_snapshot.json",
        "presence": "always_present",
        "description": "Configuration snapshot captured at artifact write time.",
    },
    {
        "file": "artifact_manifest.json",
        "presence": "always_present",
        "description": "Artifact manifest including stage, session trace, and saved-file metadata.",
    },
    {
        "file": "turn_result.json",
        "presence": "always_present",
        "description": "Full serialized conversation turn result.",
    },
    {
        "file": "repair_raw_output.txt",
        "presence": "conditional",
        "description": "Repair model output written only when the one-step repair path is used.",
    },
    {
        "file": "repair_call.json",
        "presence": "conditional",
        "description": "Repair transport request/response metadata written only when repair is used.",
    },
    {
        "file": "debug_trace.json",
        "presence": "optional_debug",
        "description": "Runtime debug trace written only when debug tracing is enabled and available.",
    },
)

LIMITATIONS = (
    "Bank-only conversational runtime; no multi-dataset conversational execution path is enabled.",
    "Frozen bank parser contract and runtime input contract are preserved as implemented in Phase 1.",
    "Clarification carryover is CLI-scoped and limited to one pending missing-information turn.",
    "No broad multi-turn memory beyond one pending clarification object.",
    "No negotiation ladder or runtime relaxation loop is implemented.",
    "No post-generation invariant validator exists beyond the current deterministic validation and routing.",
    "No service, UI, or deployment layer is part of the implemented MVP.",
    "A policy registry exists internally, but the conversational runtime remains bank-only.",
    "Local model behavior remains prompt-sensitive and is bounded by deterministic backend checks and acceptance filtering.",
)


def get_bank_field_order() -> list[str]:
    return list(BankCanonicalValidator().required_fields)


def canonicalize_fields(fields: list[str], *, kind: str) -> list[str]:
    ordered = order_fields(list(fields), get_bank_field_order())
    if kind == "changed_fields":
        return ordered
    return ordered


def load_benchmark_contract(benchmark_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    target_fields = [
        {
            "name": str(item["name"]),
            "type": str(item["type"]),
            "description": str(item.get("description", "")),
        }
        for item in payload.get("target_cf_fields", [])
    ]
    output_contract = dict(payload.get("output_contract", {}))
    return {
        "benchmark_name": str(payload.get("benchmark_name", "")),
        "target_cf_fields": target_fields,
        "output_contract": output_contract,
    }


def build_contract_metadata(*, benchmark_path: Path, parser_schema_path: Path) -> dict[str, Any]:
    benchmark_contract = load_benchmark_contract(benchmark_path)
    return {
        "parser_output": {
            "task": benchmark_contract["output_contract"].get("task"),
            "status_enum": list(benchmark_contract["output_contract"].get("status_enum", [])),
            "target_cf_fields": list(benchmark_contract["target_cf_fields"]),
            "rules": list(benchmark_contract["output_contract"].get("rules", [])),
            "parser_schema_path": str(parser_schema_path.resolve()),
            "parser_schema_version": extract_version_token(parser_schema_path),
        },
        "runtime_input": {
            "dataset": "bank",
            "profile_field_order": get_bank_field_order(),
            "shape": {"dataset": "bank", "profile": "{...}"},
        },
    }


def build_conversation_stage_table() -> list[dict[str, str]]:
    return [
        {
            "stage": stage,
            "layer": "conversation",
        }
        for stage in (
            ConversationStage.READY_FOR_RUNTIME,
            ConversationStage.NEEDS_CLARIFICATION,
            ConversationStage.CONFLICT,
            ConversationStage.UNSUPPORTED_REQUEST,
            ConversationStage.RUNTIME_SUCCESS,
            ConversationStage.RUNTIME_REJECT,
            ConversationStage.PARSER_FAILURE,
        )
    ]


def build_runtime_reason_code_table() -> list[dict[str, str]]:
    return [{"code": code, "layer": "runtime"} for code in ALL_REASON_CODES]


def build_explanation_summary_type_table() -> list[dict[str, str]]:
    return [{"summary_type": summary_type, "layer": "explanation"} for summary_type in EXPLANATION_SUMMARY_TYPES]


def build_artifact_schema_table() -> list[dict[str, str]]:
    return [dict(item) for item in ARTIFACT_SCHEMA]


def build_limitations_block() -> list[str]:
    return list(LIMITATIONS)


def build_system_diagram() -> str:
    return "\n".join(
        [
            "flowchart LR",
            '  user["User / CLI"] --> parser["Live LM Studio Structured Parser"]',
            '  parser --> schema["Normalize + Schema Validate"]',
            '  schema -->|invalid output| repair["One-Step Repair"]',
            "  repair --> schema",
            '  schema --> canonical["Canonical Bank Validator"]',
            '  canonical -->|needs clarification or conflict| response["Deterministic Clarification / Explanation"]',
            '  canonical -->|ready_for_runtime: dataset=bank| runtime["Bank Runtime Orchestrator"]',
            '  runtime --> prediction["Prediction"]',
            '  prediction --> ufce["UFCE Candidate Search"]',
            "  ufce --> response",
            '  response --> artifacts["Artifact Writer"]',
        ]
    )


def collect_provenance(
    *,
    pack_version: str,
    model_alias: str,
    api_base: str,
    benchmark_path: Path,
    system_prompt_path: Path,
    parser_schema_path: Path,
    scenario_catalog_path: Path,
    prompt_template_version: str,
    scenario_catalog_version: str,
    attempt_root: Path,
    accepted_root: Path,
) -> dict[str, Any]:
    git_commit, git_available = _read_git_commit()
    return {
        "pack_version": pack_version,
        "git_commit": git_commit,
        "git_available": git_available,
        "model_alias": model_alias,
        "model_transport": MODEL_TRANSPORT,
        "api_base": api_base,
        "benchmark_path": str(benchmark_path.resolve()),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "parser_schema_path": str(parser_schema_path.resolve()),
        "parser_schema_version": extract_version_token(parser_schema_path),
        "prompt_template_version": prompt_template_version,
        "scenario_catalog_version": scenario_catalog_version,
        "evidence_runner_version": EVIDENCE_RUNNER_VERSION,
        "chapter_generator_version": CHAPTER_GENERATOR_VERSION,
        "generated_timestamp_utc": utc_now_iso(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "scenario_catalog_path": str(scenario_catalog_path.resolve()),
        "attempt_root": str(attempt_root.resolve()),
        "accepted_root": str(accepted_root.resolve()),
    }


def extract_version_token(path: Path) -> str:
    stem = path.stem.lower()
    parts = stem.split("_")
    for part in reversed(parts):
        if part.startswith("v") and len(part) > 1 and part[1:].isdigit():
            return part
    return stem


def _read_git_commit() -> tuple[str | None, bool]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None, False
    commit = completed.stdout.strip()
    if not commit:
        return None, False
    return commit, True
