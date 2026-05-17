#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.conversation.parser_adapter import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_TIMEOUT_S,
    LiveLmStudioParserAdapter,
)
from llm.src.conversation.session import create_interactive_session_state, handle_session_turn  # noqa: E402
from llm.src.phase2.catalog import DEFAULT_CATALOG_PATH, load_catalog  # noqa: E402
from llm.src.phase2.taxonomy import classify_turn_result  # noqa: E402
from scripts.archieve.export_part2_case_studies import collect_case_studies, render_markdown  # noqa: E402


@dataclass(frozen=True)
class SuiteScenario:
    slug: str
    expected_label: str
    turns: tuple[str, ...]
    expected_merge_applied: bool | None
    description: str
    expected_first_turn_stage: str | None = None


@dataclass(frozen=True)
class SuiteOutcome:
    slug: str
    description: str
    expected_label: str
    actual_label: str
    passed: bool
    expected_merge_applied: bool | None
    actual_merge_applied: bool | None
    stages: list[str]
    output_dirs: list[str]
    turn_count: int
    first_turn_stage: str | None = None
    expected_first_turn_stage: str | None = None
    actual_carried_fields: list[str] = field(default_factory=list)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 closeout smoke suite.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--out-dir", default="outputs/conversations_smoke")
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def build_default_scenarios(catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> list[SuiteScenario]:
    catalog = load_catalog(catalog_path)
    cases = {case.case_id: case for case in catalog.iter_all_cases()}
    return [
        SuiteScenario(
            slug="suite_no_recourse_success",
            expected_label=cases["P-NR-01"].expected_label,
            turns=cases["P-NR-01"].turns,
            expected_merge_applied=None,
            description=cases["P-NR-01"].description,
        ),
        SuiteScenario(
            slug="suite_counterfactual_success",
            expected_label=cases["P-CF-01"].expected_label,
            turns=cases["P-CF-01"].turns,
            expected_merge_applied=None,
            description=cases["P-CF-01"].description,
        ),
        SuiteScenario(
            slug="suite_runtime_reject",
            expected_label=cases["P-RJ-01"].expected_label,
            turns=cases["P-RJ-01"].turns,
            expected_merge_applied=None,
            description=cases["P-RJ-01"].description,
        ),
        SuiteScenario(
            slug="suite_clarification_missing_information",
            expected_label=cases["P-CL-01"].expected_label,
            turns=cases["P-CL-01"].turns,
            expected_merge_applied=None,
            description=cases["P-CL-01"].description,
        ),
        SuiteScenario(
            slug="suite_followup_merge_to_success",
            expected_label=str(cases["S-MERGE-01"].accept["final_label"]),
            turns=cases["S-MERGE-01"].turns,
            expected_merge_applied=True,
            description=cases["S-MERGE-01"].description,
            expected_first_turn_stage=str(cases["S-MERGE-01"].accept["turn1_stage"]),
        ),
        SuiteScenario(
            slug="suite_followup_still_incomplete",
            expected_label="clarification",
            turns=cases["S-MERGE-02"].turns,
            expected_merge_applied=True,
            description=cases["S-MERGE-02"].description,
            expected_first_turn_stage=str(cases["S-MERGE-02"].accept["turn1_stage"]),
        ),
        SuiteScenario(
            slug="suite_followup_reset_new_full_profile",
            expected_label=str(cases["SM-RESET-01"].accept["final_label"]),
            turns=cases["SM-RESET-01"].turns,
            expected_merge_applied=bool(cases["SM-RESET-01"].accept["expected_merge_applied"]),
            description=cases["SM-RESET-01"].description,
            expected_first_turn_stage=str(cases["SM-RESET-01"].accept["turn1_stage"]),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    adapter = LiveLmStudioParserAdapter(
        model_alias=args.model_alias,
        api_base=args.api_base,
        timeout_s=args.timeout_s,
        benchmark_path=Path(args.benchmark),
        system_prompt_path=Path(args.system_prompt),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=adapter.load_benchmark(),
        benchmark_path=Path(args.benchmark),
        system_prompt_path=Path(args.system_prompt),
        output_root=Path(args.out_dir),
        model_alias=args.model_alias,
    )
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    outcomes: list[SuiteOutcome] = []
    for scenario in build_default_scenarios(args.catalog):
        if len(scenario.turns) == 1:
            outcomes.append(run_one_shot_scenario(orchestrator, scenario, args=args))
        else:
            outcomes.append(run_followup_scenario(orchestrator, scenario, args=args))

    write_case_study_indexes(output_root)
    summary = build_suite_summary(outcomes)
    write_suite_summary(output_root, summary)
    write_standalone_report(output_root, summary)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if bool(summary["all_passed"]) else 1


def run_one_shot_scenario(orchestrator: BankConversationOrchestrator, scenario: SuiteScenario, *, args) -> SuiteOutcome:
    command = build_runner_command(args)
    result = orchestrator.run_turn(
        user_input=scenario.turns[0],
        save_artifacts=True,
        scenario_slug=scenario.slug,
        debug_trace_enabled=args.debug_trace,
        command=command,
    )
    actual_label = classify_turn_result(stage=result.stage, explanation_payload=result.explanation_payload)
    actual_merge_applied = None if result.artifact_record is None else result.artifact_record.merge_applied
    actual_carried_fields = [] if result.artifact_record is None else list(result.artifact_record.carried_fields)
    passed = actual_label == scenario.expected_label
    output_dirs = [] if result.artifact_record is None else [result.artifact_record.output_dir]
    return SuiteOutcome(
        slug=scenario.slug,
        description=scenario.description,
        expected_label=scenario.expected_label,
        actual_label=actual_label,
        passed=passed,
        expected_merge_applied=scenario.expected_merge_applied,
        actual_merge_applied=actual_merge_applied,
        stages=[result.stage],
        output_dirs=output_dirs,
        turn_count=1,
        first_turn_stage=result.stage,
        expected_first_turn_stage=scenario.expected_first_turn_stage,
        actual_carried_fields=actual_carried_fields,
    )


def run_followup_scenario(orchestrator: BankConversationOrchestrator, scenario: SuiteScenario, *, args) -> SuiteOutcome:
    command = build_runner_command(args)
    session_state = create_interactive_session_state(session_id=scenario.slug)
    results = []
    for index, turn_text in enumerate(scenario.turns, start=1):
        result = handle_session_turn(
            orchestrator,
            session_state,
            user_input=turn_text,
            save_artifacts=True,
            scenario_slug=f"{scenario.slug}_turn{index}",
            debug_trace_enabled=args.debug_trace,
            command=command,
        )
        results.append(result)

    final_result = results[-1]
    actual_label = classify_turn_result(stage=final_result.stage, explanation_payload=final_result.explanation_payload)
    actual_merge_applied = None if final_result.artifact_record is None else final_result.artifact_record.merge_applied
    actual_carried_fields = [] if final_result.artifact_record is None else list(final_result.artifact_record.carried_fields)
    first_turn_stage = results[0].stage if results else None
    passed = actual_label == scenario.expected_label
    if scenario.expected_merge_applied is not None:
        passed = passed and actual_merge_applied == scenario.expected_merge_applied
    if scenario.expected_first_turn_stage is not None:
        passed = passed and first_turn_stage == scenario.expected_first_turn_stage
    output_dirs = [
        result.artifact_record.output_dir
        for result in results
        if result.artifact_record is not None
    ]
    return SuiteOutcome(
        slug=scenario.slug,
        description=scenario.description,
        expected_label=scenario.expected_label,
        actual_label=actual_label,
        passed=passed,
        expected_merge_applied=scenario.expected_merge_applied,
        actual_merge_applied=actual_merge_applied,
        stages=[result.stage for result in results],
        output_dirs=output_dirs,
        turn_count=len(results),
        first_turn_stage=first_turn_stage,
        expected_first_turn_stage=scenario.expected_first_turn_stage,
        actual_carried_fields=actual_carried_fields,
    )

def build_suite_summary(outcomes: list[SuiteOutcome]) -> dict[str, object]:
    counts = Counter(outcome.actual_label for outcome in outcomes)
    passed_count = sum(1 for outcome in outcomes if outcome.passed)
    return {
        "scenario_count": len(outcomes),
        "passed_count": passed_count,
        "failed_count": len(outcomes) - passed_count,
        "all_passed": passed_count == len(outcomes),
        "actual_label_counts": dict(sorted(counts.items())),
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }


def write_suite_summary(output_root: Path, summary: dict[str, object]) -> None:
    json_path = output_root / "phase1_closeout_suite_summary.json"
    markdown_path = output_root / "phase1_closeout_suite_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 1 Closeout Suite Summary",
        "",
        "- scenario_count: {0}".format(summary["scenario_count"]),
        "- passed_count: {0}".format(summary["passed_count"]),
        "- failed_count: {0}".format(summary["failed_count"]),
        "- all_passed: {0}".format(summary["all_passed"]),
        "",
        "| Scenario | Expected | Actual | Passed | First Turn | Merge Applied | Carried Fields | Stages |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for outcome in summary.get("outcomes", []):
        stages_text = " -> ".join(outcome["stages"])
        carried_fields = ", ".join(outcome.get("actual_carried_fields", [])) or "-"
        lines.append(
            "| {slug} | {expected} | {actual} | {passed} | {first_turn} | {merge} | {carried_fields} | {stages} |".format(
                slug=outcome["slug"],
                expected=outcome["expected_label"],
                actual=outcome["actual_label"],
                passed=outcome["passed"],
                first_turn=outcome.get("first_turn_stage") or "-",
                merge=outcome["actual_merge_applied"],
                carried_fields=carried_fields,
                stages=stages_text,
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_standalone_report(output_root: Path, summary: dict[str, object]) -> None:
    report_path = output_root / "phase1_closeout_standalone_report.md"
    outcomes = summary.get("outcomes") or []
    saved_turns = sum(int(outcome.get("turn_count", 0)) for outcome in outcomes if isinstance(outcome, dict))
    lines = [
        "# Phase 1 Closeout Smoke Run Standalone Report",
        "",
        "## Scope",
        "",
        "This document is generated from the latest smoke-suite run in this output folder.",
        "",
        "- all 7 defined smoke scenarios",
        f"- all {saved_turns} saved turns from this run",
        "- expected outcome vs actual outcome",
        "- merge metadata and carried fields",
        "- whether turn 1 behaved as clarification or unexpectedly completed",
        "",
        "## Overall Result",
        "",
        f"- Scenario count: {summary['scenario_count']}",
        f"- Saved turns: {saved_turns}",
        f"- Passed scenarios: {summary['passed_count']}",
        f"- Failed scenarios: {summary['failed_count']}",
        f"- Overall status: {'PASSED' if summary['all_passed'] else 'FAILED'}",
        "",
        "Actual outcome distribution across scenarios:",
        "",
    ]
    counts = summary.get("actual_label_counts") or {}
    for label, count in counts.items():
        lines.append(f"- `{label}`: {count}")

    lines.extend(
        [
            "",
            "## Scenario Summary",
            "",
            "| Scenario | Expected | Actual | Passed | Expected Turn 1 | Actual Turn 1 | Merge Applied | Carried Fields |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        carried_fields = ", ".join(outcome.get("actual_carried_fields", [])) or "-"
        lines.append(
            "| {slug} | {expected} | {actual} | {passed} | {expected_turn1} | {actual_turn1} | {merge} | {carried_fields} |".format(
                slug=outcome["slug"],
                expected=outcome["expected_label"],
                actual=outcome["actual_label"],
                passed=outcome["passed"],
                expected_turn1=outcome.get("expected_first_turn_stage") or "-",
                actual_turn1=outcome.get("first_turn_stage") or "-",
                merge=outcome["actual_merge_applied"],
                carried_fields=carried_fields,
            )
        )

    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
        ]
    )
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        lines.extend(render_outcome_report(outcome))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_study_indexes(output_root: Path) -> None:
    records = collect_case_studies(output_root)
    (output_root / "case_studies_index.json").write_text(
        json.dumps(records, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "case_studies_index.md").write_text(render_markdown(records), encoding="utf-8")


def render_outcome_report(outcome: dict[str, object]) -> list[str]:
    lines = [
        "### `{0}`".format(outcome["slug"]),
        "",
        "Expected final label:",
        "",
        "- `{0}`".format(outcome["expected_label"]),
        "",
        "Actual final result:",
        "",
        "- Label: `{0}`".format(outcome["actual_label"]),
        "- Passed: `{0}`".format(outcome["passed"]),
        "- Expected turn 1 stage: `{0}`".format(outcome.get("expected_first_turn_stage") or "-"),
        "- Actual turn 1 stage: `{0}`".format(outcome.get("first_turn_stage") or "-"),
        "- Final merge applied: `{0}`".format(outcome["actual_merge_applied"]),
        "- Final carried fields: `{0}`".format(", ".join(outcome.get("actual_carried_fields", [])) or "-"),
        "",
    ]
    for index, output_dir in enumerate(outcome.get("output_dirs", []), start=1):
        lines.extend(render_turn_bundle(output_dir=Path(output_dir), turn_index=index))
    lines.extend(
        [
            "Interpretation:",
            "",
        ]
    )
    lines.extend(render_outcome_interpretation(outcome))
    lines.append("")
    return lines


def render_turn_bundle(*, output_dir: Path, turn_index: int) -> list[str]:
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    turn_result = json.loads((output_dir / "turn_result.json").read_text(encoding="utf-8"))
    user_input = (output_dir / "user_input.txt").read_text(encoding="utf-8").strip()
    response_text = (output_dir / "response_text.txt").read_text(encoding="utf-8").strip()
    normalized_parse = turn_result.get("normalized_parse") or {}
    explanation_payload = turn_result.get("explanation_payload")
    clarification_payload = turn_result.get("clarification_payload")

    lines = [
        "#### Turn {0}".format(turn_index),
        "",
        "- Turn ID: `{0}`".format(manifest["turn_id"]),
        "- Stage: `{0}`".format(manifest["stage"]),
        "- Session ID: `{0}`".format(manifest.get("session_id") or "-"),
        "- Turn Index: `{0}`".format(manifest.get("turn_index") or "-"),
        "- Merge Applied: `{0}`".format(manifest.get("merge_applied")),
        "- Parent Turn ID: `{0}`".format(manifest.get("parent_turn_id") or "-"),
        "- Carried Fields: `{0}`".format(", ".join(manifest.get("carried_fields", [])) or "-"),
        "",
        "Input:",
        "",
        "```text",
        user_input,
        "```",
        "",
        "Parsed profile:",
        "",
        "```json",
        json.dumps(normalized_parse.get("cf_request", {}), ensure_ascii=True, indent=2),
        "```",
        "",
        "- Parser status: `{0}`".format(normalized_parse.get("status")),
    ]
    if explanation_payload:
        lines.extend(
            [
                "- Explanation label: `{0}`".format(explanation_payload.get("summary_type")),
                "- Reason codes: `{0}`".format(", ".join(explanation_payload.get("reason_codes", [])) or "-"),
                "- Changed fields: `{0}`".format(", ".join(explanation_payload.get("changed_fields", [])) or "-"),
            ]
        )
    if clarification_payload:
        lines.extend(
            [
                "- Clarification type: `{0}`".format(clarification_payload.get("clarification_type")),
                "- Missing fields: `{0}`".format(", ".join(clarification_payload.get("missing_fields", [])) or "-"),
            ]
        )
    lines.extend(
        [
            "",
            "Response:",
            "",
            "```text",
            response_text,
            "```",
            "",
        ]
    )
    return lines


def render_outcome_interpretation(outcome: dict[str, object]) -> list[str]:
    if outcome["passed"]:
        if outcome.get("expected_merge_applied") is True:
            return [
                "- This scenario passed and exercised the merge seam as intended.",
            ]
        if outcome.get("expected_first_turn_stage") == "NEEDS_CLARIFICATION" and outcome.get("actual_merge_applied") is False:
            return [
                "- This scenario passed because the reset-to-new-turn behavior worked and merge remained disabled.",
            ]
        return [
            "- This scenario behaved as expected.",
        ]

    if outcome.get("expected_merge_applied") is True and outcome.get("actual_merge_applied") is False:
        if outcome.get("first_turn_stage") != "NEEDS_CLARIFICATION":
            return [
                "- This scenario failed because turn 1 completed unexpectedly, so no pending clarification state existed to merge on turn 2.",
            ]
        return [
            "- This scenario failed because the final turn did not record merge metadata even though merge was expected.",
        ]

    if outcome["expected_label"] == "runtime_reject" and outcome["actual_label"] == "counterfactual_found":
        return [
            "- This scenario failed because the selected reject prompt is not stable under the current runtime/model bundle and still yields a feasible counterfactual.",
        ]

    return [
        "- This scenario failed because the observed final label or stage flow did not match the expected smoke behavior.",
    ]


def build_runner_command(args) -> str:
    return (
        "python scripts/run_phase1_closeout_suite.py "
        f"--model-alias {args.model_alias} "
        f"--api-base {args.api_base} "
        f"--timeout-s {args.timeout_s} "
        f"--benchmark {args.benchmark} "
        f"--system-prompt {args.system_prompt} "
        f"--catalog {args.catalog} "
        f"--out-dir {args.out_dir}"
        + (" --debug-trace" if args.debug_trace else "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
