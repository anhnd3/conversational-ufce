#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.conversation.parser_adapter import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_TIMEOUT_S,
    LiveLmStudioParserAdapter,
)
from llm.src.conversation.session import create_interactive_session_state, handle_session_turn  # noqa: E402
from llm.src.phase2.catalog import DEFAULT_CATALOG_PATH, CatalogCase, Phase2ScenarioCatalog, load_catalog  # noqa: E402
from llm.src.phase2.metadata import (  # noqa: E402
    EVIDENCE_RUNNER_VERSION,
    canonicalize_fields,
    collect_provenance,
)
from llm.src.phase2.taxonomy import (  # noqa: E402
    CLARIFICATION,
    COUNTERFACTUAL_FOUND,
    NO_RECOURSE_NEEDED,
    PARSER_FAILURE,
    PRIMARY_ACCEPTANCE_TARGET,
    RUNTIME_REJECT,
    SUPPLEMENTAL_FOLLOWUP,
    classify_turn_result,
)
from llm.src.utils.io import write_json  # noqa: E402
from scripts.archieve.export_part2_case_studies import collect_case_studies, render_markdown  # noqa: E402


@dataclass(frozen=True)
class AttemptRunOutcome:
    case_id: str
    slug: str
    run_name: str
    expected_label: str
    actual_label: str
    stage: str
    accepted: bool
    acceptance_errors: list[str]
    artifact_dir: str | None


@dataclass(frozen=True)
class PrimaryCaseOutcome:
    case_id: str
    slug: str
    expected_label: str
    accepted: bool
    accepted_case_dir: str | None
    runs: list[AttemptRunOutcome]
    rejection_reasons: list[str]


@dataclass(frozen=True)
class SupplementalCaseOutcome:
    case_id: str
    slug: str
    expected_label: str
    supplemental_type: str
    accepted: bool
    accepted_case_dir: str | None
    final_label: str
    final_stage: str
    merge_applied: bool
    parent_turn_id: str | None
    carried_fields: list[str]
    turn_dirs: list[str]
    rejection_reasons: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the locked Phase 2 bank evidence pack.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--out-dir", default="outputs/conversations_phase2_pack")
    parser.add_argument("--pack-version", default="")
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog)
    pack_version = args.pack_version or build_pack_version(catalog.catalog_version)
    pack_root = Path(args.out_dir) / pack_version
    if pack_root.exists():
        raise SystemExit(f"Pack root already exists: {pack_root}")

    layout = prepare_pack_layout(pack_root)
    write_pack_status(layout["pack_root"], pack_version=pack_version, status="in_progress", failure_reason=None)

    try:
        adapter = LiveLmStudioParserAdapter(
            model_alias=args.model_alias,
            api_base=args.api_base,
            timeout_s=args.timeout_s,
            benchmark_path=Path(args.benchmark),
            system_prompt_path=Path(args.system_prompt),
            schema_path=DEFAULT_SCHEMA_PATH,
        )
        orchestrator = BankConversationOrchestrator(
            parser_adapter=adapter,
            benchmark=adapter.load_benchmark(),
            benchmark_path=Path(args.benchmark),
            system_prompt_path=Path(args.system_prompt),
            output_root=layout["attempt_root"],
            model_alias=args.model_alias,
        )

        command = build_runner_command(args, pack_version=pack_version)
        primary_outcomes = [
            run_primary_case(
                orchestrator,
                case,
                attempt_primary_root=layout["attempt_primary_root"],
                accepted_primary_root=layout["accepted_primary_root"],
                command=command,
                debug_trace_enabled=args.debug_trace,
            )
            for case in catalog.primary_cases
        ]
        supplemental_outcomes = [
            run_supplemental_case(
                orchestrator,
                case,
                attempt_supplemental_root=layout["attempt_supplemental_root"],
                accepted_supplemental_root=layout["accepted_supplemental_root"],
                command=command,
                debug_trace_enabled=args.debug_trace,
            )
            for case in catalog.supplemental_cases
        ]

        write_case_study_indexes(layout["accepted_root"], layout["indexes_root"])
        provenance = collect_provenance(
            pack_version=pack_version,
            model_alias=args.model_alias,
            api_base=args.api_base,
            benchmark_path=Path(args.benchmark),
            system_prompt_path=Path(args.system_prompt),
            parser_schema_path=DEFAULT_SCHEMA_PATH,
            scenario_catalog_path=Path(args.catalog),
            prompt_template_version=catalog.prompt_template_version,
            scenario_catalog_version=catalog.catalog_version,
            attempt_root=layout["attempt_root"],
            accepted_root=layout["accepted_root"],
        )
        attempt_summary = build_attempt_summary(
            pack_version=pack_version,
            catalog=catalog,
            primary_outcomes=primary_outcomes,
            supplemental_outcomes=supplemental_outcomes,
            indexes_root=layout["indexes_root"],
        )
        write_attempt_summary(layout["pack_root"], attempt_summary)
        manifest = build_pack_manifest(
            pack_version=pack_version,
            catalog=catalog,
            provenance=provenance,
            primary_outcomes=primary_outcomes,
            supplemental_outcomes=supplemental_outcomes,
            indexes_root=layout["indexes_root"],
        )
        write_json(layout["pack_root"] / "phase2_pack_manifest.json", manifest)
        write_pack_status(layout["pack_root"], pack_version=pack_version, status="complete", failure_reason=None)
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        accepted_counts = manifest["accepted_counts"]["primary_by_label"]
        accepted_supplemental = manifest["accepted_counts"]["supplemental_cases"]
        return 0 if meets_primary_acceptance_target(accepted_counts) and accepted_supplemental == 2 else 1
    except Exception as exc:
        write_pack_status(
            layout["pack_root"],
            pack_version=pack_version,
            status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
        return 1


def build_pack_version(catalog_version: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"phase2_pack_{timestamp}__{catalog_version}"


def prepare_pack_layout(pack_root: Path) -> dict[str, Path]:
    attempt_root = pack_root / "attempts"
    accepted_root = pack_root / "accepted"
    layout = {
        "pack_root": pack_root,
        "attempt_root": attempt_root,
        "accepted_root": accepted_root,
        "attempt_primary_root": attempt_root / "primary",
        "attempt_supplemental_root": attempt_root / "supplemental",
        "accepted_primary_root": accepted_root / "primary",
        "accepted_supplemental_root": accepted_root / "supplemental",
        "indexes_root": pack_root / "indexes",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def run_primary_case(
    orchestrator: BankConversationOrchestrator,
    case: CatalogCase,
    *,
    attempt_primary_root: Path,
    accepted_primary_root: Path,
    command: str,
    debug_trace_enabled: bool,
) -> PrimaryCaseOutcome:
    case_attempt_root = attempt_primary_root / build_case_folder_name(case)
    case_attempt_root.mkdir(parents=True, exist_ok=True)
    run_outcomes: list[AttemptRunOutcome] = []

    for run_index in (1, 2):
        result = orchestrator.run_turn(
            user_input=case.turns[0],
            save_artifacts=True,
            scenario_slug=f"{case.case_id.lower()}__{case.slug}__run{run_index}",
            debug_trace_enabled=debug_trace_enabled,
            command=command,
        )
        relocated_dir = relocate_saved_artifacts(
            result.artifact_record.output_dir if result.artifact_record is not None else None,
            case_attempt_root / f"run{run_index}",
        )
        acceptance_errors = evaluate_primary_case_semantics(case, result)
        run_outcomes.append(
            AttemptRunOutcome(
                case_id=case.case_id,
                slug=case.slug,
                run_name=f"run{run_index}",
                expected_label=case.expected_label,
                actual_label=classify_turn_result(stage=result.stage, explanation_payload=result.explanation_payload),
                stage=result.stage,
                accepted=not acceptance_errors,
                acceptance_errors=acceptance_errors,
                artifact_dir=None if relocated_dir is None else str(relocated_dir.resolve()),
            )
        )

    rejection_reasons: list[str] = []
    accepted = all(outcome.accepted for outcome in run_outcomes)
    if not accepted:
        for outcome in run_outcomes:
            rejection_reasons.extend(outcome.acceptance_errors)
    accepted_case_dir = None
    if accepted:
        source_dir = case_attempt_root / "run2"
        accepted_case_dir = accepted_primary_root / build_case_folder_name(case)
        copy_artifact_tree(source_dir, accepted_case_dir)
    return PrimaryCaseOutcome(
        case_id=case.case_id,
        slug=case.slug,
        expected_label=case.expected_label,
        accepted=accepted,
        accepted_case_dir=None if accepted_case_dir is None else str(accepted_case_dir.resolve()),
        runs=run_outcomes,
        rejection_reasons=sorted(set(rejection_reasons)),
    )


def run_supplemental_case(
    orchestrator: BankConversationOrchestrator,
    case: CatalogCase,
    *,
    attempt_supplemental_root: Path,
    accepted_supplemental_root: Path,
    command: str,
    debug_trace_enabled: bool,
) -> SupplementalCaseOutcome:
    session_state = create_interactive_session_state(session_id=case.case_id)
    case_attempt_root = attempt_supplemental_root / build_case_folder_name(case)
    case_attempt_root.mkdir(parents=True, exist_ok=True)
    results = []
    turn_dirs: list[str] = []

    for index, turn_text in enumerate(case.turns, start=1):
        result = handle_session_turn(
            orchestrator,
            session_state,
            user_input=turn_text,
            save_artifacts=True,
            scenario_slug=f"{case.case_id.lower()}__{case.slug}__turn{index}",
            debug_trace_enabled=debug_trace_enabled,
            command=command,
        )
        relocated_dir = relocate_saved_artifacts(
            result.artifact_record.output_dir if result.artifact_record is not None else None,
            case_attempt_root / f"turn{index}",
        )
        results.append(result)
        if relocated_dir is not None:
            turn_dirs.append(str(relocated_dir.resolve()))

    final_result = results[-1]
    acceptance_errors = evaluate_supplemental_case_semantics(case, results)
    accepted = not acceptance_errors
    accepted_case_dir = None
    if accepted:
        accepted_case_dir = accepted_supplemental_root / build_case_folder_name(case)
        for index in range(1, len(results) + 1):
            copy_artifact_tree(case_attempt_root / f"turn{index}", accepted_case_dir / f"turn{index}")

    final_label = classify_turn_result(stage=final_result.stage, explanation_payload=final_result.explanation_payload)
    artifact = final_result.artifact_record
    return SupplementalCaseOutcome(
        case_id=case.case_id,
        slug=case.slug,
        expected_label=case.expected_label,
        supplemental_type=str(case.accept["supplemental_type"]),
        accepted=accepted,
        accepted_case_dir=None if accepted_case_dir is None else str(accepted_case_dir.resolve()),
        final_label=final_label,
        final_stage=final_result.stage,
        merge_applied=bool(artifact.merge_applied) if artifact is not None else False,
        parent_turn_id=None if artifact is None else artifact.parent_turn_id,
        carried_fields=[] if artifact is None else list(artifact.carried_fields),
        turn_dirs=turn_dirs,
        rejection_reasons=acceptance_errors,
    )


def evaluate_primary_case_semantics(case: CatalogCase, result) -> list[str]:
    actual_label = classify_turn_result(stage=result.stage, explanation_payload=result.explanation_payload)
    errors: list[str] = []
    if actual_label != case.expected_label:
        errors.append(f"expected_label={case.expected_label}, actual_label={actual_label}")

    if case.expected_label == NO_RECOURSE_NEEDED:
        payload = result.explanation_payload
        actual_reason_codes = [] if payload is None else list(payload.reason_codes)
        if result.stage != "RUNTIME_SUCCESS":
            errors.append(f"expected stage RUNTIME_SUCCESS, got {result.stage}")
        if payload is None or payload.summary_type != NO_RECOURSE_NEEDED:
            errors.append("expected explanation summary_type no_recourse_needed")
        if set(actual_reason_codes) != set(case.accept["reason_codes"]):
            errors.append("runtime reason code set mismatch")

    elif case.expected_label == COUNTERFACTUAL_FOUND:
        payload = result.explanation_payload
        actual_changed_fields = [] if payload is None else list(payload.changed_fields)
        expected_changed_fields = canonicalize_fields(list(case.accept["changed_fields"]), kind="changed_fields")
        if result.stage != "RUNTIME_SUCCESS":
            errors.append(f"expected stage RUNTIME_SUCCESS, got {result.stage}")
        if payload is None or payload.summary_type != COUNTERFACTUAL_FOUND:
            errors.append("expected explanation summary_type counterfactual_found")
        if payload is None or payload.counterfactual_summary is None:
            errors.append("missing first counterfactual summary")
        if canonicalize_fields(actual_changed_fields, kind="changed_fields") != expected_changed_fields:
            errors.append("changed field list mismatch")

    elif case.expected_label == RUNTIME_REJECT:
        payload = result.explanation_payload
        actual_reason_codes = [] if payload is None else list(payload.reason_codes)
        if result.stage != "RUNTIME_REJECT":
            errors.append(f"expected stage RUNTIME_REJECT, got {result.stage}")
        if payload is None or payload.summary_type != RUNTIME_REJECT:
            errors.append("expected explanation summary_type runtime_reject")
        if set(actual_reason_codes) != set(case.accept["reason_codes"]):
            errors.append("runtime reject reason code set mismatch")

    elif case.expected_label == CLARIFICATION:
        payload = result.clarification_payload
        expected_missing_fields = canonicalize_fields(list(case.accept["missing_fields"]), kind="missing_fields")
        actual_missing_fields = [] if payload is None else canonicalize_fields(list(payload.missing_fields), kind="missing_fields")
        expected_conflicts = list(case.accept.get("conflicts", []))
        actual_conflicts = [] if payload is None else list(payload.conflicts)
        if result.stage != "NEEDS_CLARIFICATION":
            errors.append(f"expected stage NEEDS_CLARIFICATION, got {result.stage}")
        if payload is None:
            errors.append("missing clarification payload")
        else:
            if payload.clarification_type != case.accept["clarification_type"]:
                errors.append("clarification type mismatch")
            if actual_missing_fields != expected_missing_fields:
                errors.append("missing_fields mismatch")
            if actual_conflicts != expected_conflicts:
                errors.append("conflict list mismatch")

    if actual_label == PARSER_FAILURE:
        errors.append("parser_failure is never accepted")
    return sorted(set(errors))


def evaluate_supplemental_case_semantics(case: CatalogCase, results: list[Any]) -> list[str]:
    errors: list[str] = []
    if len(results) != 2:
        return ["supplemental case must produce exactly 2 turns"]

    turn1, turn2 = results
    turn1_label = classify_turn_result(stage=turn1.stage, explanation_payload=turn1.explanation_payload)
    turn2_label = classify_turn_result(stage=turn2.stage, explanation_payload=turn2.explanation_payload)
    artifact = turn2.artifact_record
    expected_carried_fields = canonicalize_fields(list(case.accept["carried_fields"]), kind="carried_fields")
    actual_carried_fields = [] if artifact is None else canonicalize_fields(list(artifact.carried_fields), kind="carried_fields")

    if turn1.stage != case.accept["turn1_stage"]:
        errors.append(f"turn1_stage mismatch: expected {case.accept['turn1_stage']}, got {turn1.stage}")
    if artifact is None:
        errors.append("turn2 missing artifact record")
    else:
        if artifact.merge_applied is not True:
            errors.append("turn2 merge_applied must be true")
        if artifact.parent_turn_id != turn1.turn_id:
            errors.append("turn2 parent_turn_id must match turn1 turn_id")
        if actual_carried_fields != expected_carried_fields:
            errors.append("turn2 carried_fields mismatch")

    supplemental_type = case.accept["supplemental_type"]
    if supplemental_type == "supplemental_followup_merge_to_success":
        if turn2_label != case.accept["final_label"]:
            errors.append("turn2 final label mismatch")
    elif supplemental_type == "supplemental_followup_still_incomplete":
        payload = turn2.clarification_payload
        expected_missing_fields = canonicalize_fields(list(case.accept["missing_fields"]), kind="missing_fields")
        actual_missing_fields = [] if payload is None else canonicalize_fields(list(payload.missing_fields), kind="missing_fields")
        if turn2.stage != "NEEDS_CLARIFICATION":
            errors.append(f"turn2 stage mismatch: expected NEEDS_CLARIFICATION, got {turn2.stage}")
        if turn2_label != CLARIFICATION:
            errors.append("turn2 final label mismatch")
        if payload is None:
            errors.append("turn2 clarification payload missing")
        else:
            if payload.clarification_type != case.accept["clarification_type"]:
                errors.append("turn2 clarification_type mismatch")
            if actual_missing_fields != expected_missing_fields:
                errors.append("turn2 missing_fields mismatch")
    else:
        errors.append(f"unsupported supplemental_type: {supplemental_type}")
    return sorted(set(errors))


def relocate_saved_artifacts(source_dir: str | None, destination_dir: Path) -> Path | None:
    if not source_dir:
        return None
    source = Path(source_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.move(str(source), str(destination_dir))
    rewrite_output_dir_metadata(destination_dir)
    return destination_dir


def copy_artifact_tree(source_dir: Path, destination_dir: Path) -> Path:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    rewrite_output_dir_metadata(destination_dir)
    return destination_dir


def rewrite_output_dir_metadata(artifact_dir: Path) -> None:
    resolved = str(artifact_dir.resolve())
    manifest_path = artifact_dir / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_dir"] = resolved
        write_json(manifest_path, manifest)

    turn_result_path = artifact_dir / "turn_result.json"
    if turn_result_path.exists():
        turn_result = json.loads(turn_result_path.read_text(encoding="utf-8"))
        artifact_record = turn_result.get("artifact_record")
        if isinstance(artifact_record, dict):
            artifact_record["output_dir"] = resolved
        write_json(turn_result_path, turn_result)

    config_path = artifact_dir / "config_snapshot.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["output_dir"] = resolved
        write_json(config_path, config)


def build_case_folder_name(case: CatalogCase) -> str:
    return f"{case.case_id}__{case.slug}"


def build_attempt_summary(
    *,
    pack_version: str,
    catalog: Phase2ScenarioCatalog,
    primary_outcomes: list[PrimaryCaseOutcome],
    supplemental_outcomes: list[SupplementalCaseOutcome],
    indexes_root: Path,
) -> dict[str, Any]:
    accepted_primary = [outcome for outcome in primary_outcomes if outcome.accepted]
    accepted_primary_counts = Counter(outcome.expected_label for outcome in accepted_primary)
    return {
        "pack_version": pack_version,
        "scenario_catalog_version": catalog.catalog_version,
        "runner_version": EVIDENCE_RUNNER_VERSION,
        "primary_acceptance_target": dict(PRIMARY_ACCEPTANCE_TARGET),
        "attempted_counts": {
            "primary_cases": len(primary_outcomes),
            "primary_runs": sum(len(outcome.runs) for outcome in primary_outcomes),
            "supplemental_cases": len(supplemental_outcomes),
        },
        "accepted_counts": {
            "primary_cases": len(accepted_primary),
            "primary_by_label": dict(sorted(accepted_primary_counts.items())),
            "supplemental_cases": sum(1 for outcome in supplemental_outcomes if outcome.accepted),
        },
        "stability_filter_results": [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "stable": outcome.accepted,
                "run_labels": [run.actual_label for run in outcome.runs],
                "accepted": outcome.accepted,
                "rejection_reasons": list(outcome.rejection_reasons),
            }
            for outcome in primary_outcomes
        ],
        "primary_cases": [asdict(outcome) for outcome in primary_outcomes],
        "supplemental_cases": [asdict(outcome) for outcome in supplemental_outcomes],
        "indexes": {
            "case_studies_index_json": str((indexes_root / "case_studies_index.json").resolve()),
            "case_studies_index_md": str((indexes_root / "case_studies_index.md").resolve()),
        },
    }


def build_pack_manifest(
    *,
    pack_version: str,
    catalog: Phase2ScenarioCatalog,
    provenance: dict[str, Any],
    primary_outcomes: list[PrimaryCaseOutcome],
    supplemental_outcomes: list[SupplementalCaseOutcome],
    indexes_root: Path,
) -> dict[str, Any]:
    accepted_primary = [outcome for outcome in primary_outcomes if outcome.accepted]
    rejected_primary = [outcome for outcome in primary_outcomes if not outcome.accepted]
    accepted_supplemental = [outcome for outcome in supplemental_outcomes if outcome.accepted]
    rejected_supplemental = [outcome for outcome in supplemental_outcomes if not outcome.accepted]
    accepted_primary_counts = Counter(outcome.expected_label for outcome in accepted_primary)
    return {
        "pack_version": pack_version,
        "status": "complete",
        "provenance": dict(provenance),
        "scenario_catalog_version": catalog.catalog_version,
        "primary_acceptance_target": dict(PRIMARY_ACCEPTANCE_TARGET),
        "attempted_counts": {
            "primary_cases": len(primary_outcomes),
            "primary_runs": sum(len(outcome.runs) for outcome in primary_outcomes),
            "supplemental_cases": len(supplemental_outcomes),
        },
        "accepted_counts": {
            "primary_cases": len(accepted_primary),
            "primary_by_label": dict(sorted(accepted_primary_counts.items())),
            "supplemental_cases": len(accepted_supplemental),
        },
        "stability_filter_results": [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "accepted": outcome.accepted,
                "run_labels": [run.actual_label for run in outcome.runs],
                "rejection_reasons": list(outcome.rejection_reasons),
            }
            for outcome in primary_outcomes
        ],
        "accepted_primary_cases": [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "expected_label": outcome.expected_label,
                "folder": outcome.accepted_case_dir,
            }
            for outcome in accepted_primary
        ],
        "accepted_supplemental_demos": [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "expected_label": outcome.expected_label,
                "supplemental_type": outcome.supplemental_type,
                "folder": outcome.accepted_case_dir,
            }
            for outcome in accepted_supplemental
        ],
        "rejected_cases": [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "expected_label": outcome.expected_label,
                "rejection_reasons": list(outcome.rejection_reasons),
            }
            for outcome in rejected_primary
        ]
        + [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "expected_label": outcome.expected_label,
                "rejection_reasons": list(outcome.rejection_reasons),
            }
            for outcome in rejected_supplemental
        ],
        "indexes": {
            "case_studies_index_json": str((indexes_root / "case_studies_index.json").resolve()),
            "case_studies_index_md": str((indexes_root / "case_studies_index.md").resolve()),
        },
    }


def meets_primary_acceptance_target(counts: dict[str, int]) -> bool:
    return all(counts.get(label, 0) == target for label, target in PRIMARY_ACCEPTANCE_TARGET.items())


def write_pack_status(pack_root: Path, *, pack_version: str, status: str, failure_reason: str | None) -> None:
    payload: dict[str, Any] = {
        "pack_version": pack_version,
        "status": status,
    }
    if failure_reason:
        payload["failure_reason"] = failure_reason
    write_json(pack_root / "pack_status.json", payload)


def write_attempt_summary(pack_root: Path, summary: dict[str, Any]) -> None:
    write_json(pack_root / "attempt_summary.json", summary)
    lines = [
        "# Phase 2 Attempt Summary",
        "",
        f"- pack_version: {summary['pack_version']}",
        f"- scenario_catalog_version: {summary['scenario_catalog_version']}",
        f"- runner_version: {summary['runner_version']}",
        "",
        "## Attempted Counts",
        "",
    ]
    for key, value in summary["attempted_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Accepted Counts",
            "",
        ]
    )
    accepted_counts = summary["accepted_counts"]
    lines.append(f"- primary_cases: {accepted_counts['primary_cases']}")
    lines.append(f"- supplemental_cases: {accepted_counts['supplemental_cases']}")
    for label, count in accepted_counts["primary_by_label"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Primary Case Stability",
            "",
        ]
    )
    for case in summary["primary_cases"]:
        run_labels = ", ".join(run["actual_label"] for run in case["runs"])
        lines.append(
            f"- {case['case_id']} ({case['slug']}): accepted={case['accepted']}, run_labels=[{run_labels}], rejection_reasons={case['rejection_reasons']}"
        )
    lines.extend(
        [
            "",
            "## Supplemental Cases",
            "",
        ]
    )
    for case in summary["supplemental_cases"]:
        lines.append(
            f"- {case['case_id']} ({case['slug']}): accepted={case['accepted']}, final_label={case['final_label']}, rejection_reasons={case['rejection_reasons']}"
        )
    (pack_root / "attempt_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_study_indexes(accepted_root: Path, indexes_root: Path) -> None:
    indexes_root.mkdir(parents=True, exist_ok=True)
    records = collect_case_studies(accepted_root)
    write_json(indexes_root / "case_studies_index.json", records)
    (indexes_root / "case_studies_index.md").write_text(render_markdown(records), encoding="utf-8")


def build_runner_command(args, *, pack_version: str) -> str:
    return (
        "python scripts/run_part2_bank_evidence_pack.py "
        f"--model-alias {args.model_alias} "
        f"--api-base {args.api_base} "
        f"--timeout-s {args.timeout_s} "
        f"--benchmark {args.benchmark} "
        f"--system-prompt {args.system_prompt} "
        f"--catalog {args.catalog} "
        f"--out-dir {args.out_dir} "
        f"--pack-version {pack_version}"
        + (" --debug-trace" if args.debug_trace else "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
