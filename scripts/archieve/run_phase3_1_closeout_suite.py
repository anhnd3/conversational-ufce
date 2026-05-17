#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.conversation.orchestrator import BankConversationOrchestrator  # noqa: E402
from llm.src.conversation.parser_adapter import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_TIMEOUT_S,
    LiveLmStudioParserAdapter,
)
from llm.src.conversation.session import create_interactive_session_state, handle_session_turn  # noqa: E402
from llm.src.phase3.catalog import DEFAULT_CATALOG_PATH, Phase31ValidationScenario, load_catalog  # noqa: E402
from scripts.archieve.export_part2_case_studies import collect_case_studies, render_markdown  # noqa: E402

PHASE32_HANDOFF_ITEMS = [
    "runtime reproducibility hardening pending",
    "invariant gate pending",
    "service/API pending",
    "UI pending",
    "persistence/observability pending",
    "E2E product smoke pending",
]


@dataclass(frozen=True)
class AutomatedSuiteResult:
    suite_name: str
    command: str
    exit_code: int
    passed: bool
    summary_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LiveScenarioOutcome:
    scenario_id: str
    slug: str
    description: str
    expected_final_state: str
    actual_final_state: str
    passed: bool
    turn_count: int
    stages: list[str]
    artifact_folders: list[str]
    key_payload_checks: dict[str, bool]
    expected_public_ready_hidden: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 3.1 closeout validation suite.")
    parser.add_argument("--mode", choices=("both", "tests", "live"), default="both")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--out-dir", default="outputs/phase3_1_closeout")
    parser.add_argument("--debug-trace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog)
    run_version = build_run_version()
    run_root = Path(args.out_dir) / run_version
    run_root.mkdir(parents=True, exist_ok=True)
    live_output_root = run_root / "live_outputs"
    live_output_root.mkdir(parents=True, exist_ok=True)

    test_results: list[AutomatedSuiteResult] = []
    live_results: list[LiveScenarioOutcome] = []

    if args.mode in {"both", "tests"}:
        test_results = run_automated_suites()

    if args.mode in {"both", "live"}:
        orchestrator = build_orchestrator(args=args, output_root=live_output_root)
        live_results = run_live_scenarios(orchestrator=orchestrator, catalog_path=args.catalog, args=args)
        write_case_study_indexes(live_output_root)

    summary = build_closeout_summary(
        args=args,
        catalog_version=catalog.catalog_version,
        run_version=run_version,
        run_root=run_root,
        live_output_root=live_output_root,
        mode=args.mode,
        automated_suites=test_results,
        live_scenarios=live_results,
    )
    write_closeout_summary(run_root, summary)
    write_standalone_report(run_root, summary)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if bool(summary["ready_to_start_phase3_2"]) else 1


def build_orchestrator(*, args, output_root: Path) -> BankConversationOrchestrator:
    adapter = LiveLmStudioParserAdapter(
        model_alias=args.model_alias,
        api_base=args.api_base,
        timeout_s=args.timeout_s,
        benchmark_path=Path(args.benchmark),
        system_prompt_path=Path(args.system_prompt),
    )
    return BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=adapter.load_benchmark(),
        benchmark_path=Path(args.benchmark),
        system_prompt_path=Path(args.system_prompt),
        output_root=output_root,
        model_alias=args.model_alias,
    )


def build_run_version(timestamp_utc: str | None = None) -> str:
    active = timestamp_utc or utc_now_token()
    return f"phase3_1_closeout_{active}"


def utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_automated_suites() -> list[AutomatedSuiteResult]:
    return [
        run_automated_suite("llm/tests", ["pytest", "-q", "llm/tests"]),
        run_automated_suite("llm_eval/tests", ["pytest", "-q", "llm_eval/tests"]),
    ]


def run_automated_suite(suite_name: str, command: list[str]) -> AutomatedSuiteResult:
    completed = subprocess.run(
        [sys.executable, "-m", *command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return AutomatedSuiteResult(
        suite_name=suite_name,
        command=" ".join(command),
        exit_code=completed.returncode,
        passed=completed.returncode == 0,
        summary_lines=extract_summary_lines(combined),
    )


def extract_summary_lines(output: str, *, max_lines: int = 4) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if any(token in line.lower() for token in ("passed", "failed", "error", "warning", "collected"))
    ]
    source = interesting or lines
    return source[-max_lines:]


def run_live_scenarios(
    *,
    orchestrator: BankConversationOrchestrator,
    catalog_path: str | Path,
    args,
) -> list[LiveScenarioOutcome]:
    catalog = load_catalog(catalog_path)
    outcomes: list[LiveScenarioOutcome] = []
    for scenario in catalog.scenarios:
        if len(scenario.turns) == 1:
            outcomes.append(run_live_one_shot_scenario(orchestrator=orchestrator, scenario=scenario, args=args))
        else:
            outcomes.append(run_live_followup_scenario(orchestrator=orchestrator, scenario=scenario, args=args))
    return outcomes


def run_live_one_shot_scenario(
    *,
    orchestrator: BankConversationOrchestrator,
    scenario: Phase31ValidationScenario,
    args,
) -> LiveScenarioOutcome:
    result = orchestrator.run_turn(
        user_input=scenario.turns[0],
        save_artifacts=True,
        scenario_slug=scenario.slug,
        debug_trace_enabled=args.debug_trace,
        command=build_runner_command(args),
    )
    return evaluate_live_outcome(scenario=scenario, results=[result])


def run_live_followup_scenario(
    *,
    orchestrator: BankConversationOrchestrator,
    scenario: Phase31ValidationScenario,
    args,
) -> LiveScenarioOutcome:
    session_state = create_interactive_session_state(session_id=scenario.scenario_id.lower())
    results = []
    for index, turn_text in enumerate(scenario.turns, start=1):
        result = handle_session_turn(
            orchestrator,
            session_state,
            user_input=turn_text,
            save_artifacts=True,
            scenario_slug=f"{scenario.slug}_turn{index}",
            debug_trace_enabled=args.debug_trace,
            command=build_runner_command(args),
        )
        results.append(result)
    return evaluate_live_outcome(scenario=scenario, results=results)


def evaluate_live_outcome(*, scenario: Phase31ValidationScenario, results) -> LiveScenarioOutcome:
    final_result = results[-1]
    actual_final_state = extract_public_state(final_result)
    checks = build_key_payload_checks(scenario=scenario, results=results)
    passed = actual_final_state == scenario.expected_final_state and all(checks.values())
    artifact_folders = [
        result.artifact_record.output_dir
        for result in results
        if result.artifact_record is not None
    ]
    return LiveScenarioOutcome(
        scenario_id=scenario.scenario_id,
        slug=scenario.slug,
        description=scenario.description,
        expected_final_state=scenario.expected_final_state,
        actual_final_state=actual_final_state,
        passed=passed,
        turn_count=len(results),
        stages=[extract_public_state(result) for result in results],
        artifact_folders=artifact_folders,
        key_payload_checks=checks,
        expected_public_ready_hidden=actual_final_state != "READY_FOR_RUNTIME",
    )


def build_key_payload_checks(*, scenario: Phase31ValidationScenario, results) -> dict[str, bool]:
    accept = scenario.accept
    kind = str(accept["kind"])
    final_result = results[-1]
    checks: dict[str, bool] = {
        "ready_for_runtime_not_final_public_state": extract_public_state(final_result) != "READY_FOR_RUNTIME",
    }
    if kind in {"no_recourse_needed", "counterfactual_found", "runtime_reject"}:
        checks["ready_for_runtime_trace_only"] = (
            "READY_FOR_RUNTIME" in list(getattr(final_result, "stage_trace", []))
            and extract_public_state(final_result) != "READY_FOR_RUNTIME"
        )
    if kind == "no_recourse_needed":
        checks["summary_type_match"] = extract_summary_type(final_result) == accept["summary_type"]
        checks["included_suggestion_types_match"] = extract_suggestion_types(final_result) == list(
            accept["included_suggestion_types"]
        )
        checks["runtime_result_present"] = final_result.runtime_result is not None
    elif kind == "counterfactual_found":
        checks["summary_type_match"] = extract_summary_type(final_result) == accept["summary_type"]
        checks["runtime_result_present"] = final_result.runtime_result is not None
    elif kind == "clarification_merge_success":
        checks["turn1_state_match"] = extract_public_state(results[0]) == accept["turn1_final_state"]
        checks["turn2_state_match"] = extract_public_state(results[1]) == accept["turn2_final_state"]
        checks["merge_applied_match"] = bool(results[1].artifact_record and results[1].artifact_record.merge_applied) == bool(
            accept["turn2_merge_applied"]
        )
        checks["parent_turn_link_match"] = bool(
            results[1].artifact_record
            and results[1].artifact_record.parent_turn_id == results[0].turn_id
        )
        checks["runtime_result_present"] = results[1].runtime_result is not None
        checks["ready_for_runtime_trace_only"] = (
            "READY_FOR_RUNTIME" in list(getattr(results[1], "stage_trace", []))
            and extract_public_state(results[1]) != "READY_FOR_RUNTIME"
        )
    elif kind == "clarification_still_incomplete":
        checks["turn1_state_match"] = extract_public_state(results[0]) == accept["turn1_final_state"]
        checks["turn2_state_match"] = extract_public_state(results[1]) == accept["turn2_final_state"]
        checks["merge_applied_match"] = bool(results[1].artifact_record and results[1].artifact_record.merge_applied) == bool(
            accept["turn2_merge_applied"]
        )
        checks["parent_turn_link_match"] = bool(
            results[1].artifact_record
            and results[1].artifact_record.parent_turn_id == results[0].turn_id
        )
    elif kind == "conflict":
        checks["runtime_result_absent"] = (final_result.runtime_result is None) == bool(accept["runtime_result_absent"])
    elif kind == "unsupported":
        checks["template_type_match"] = extract_template_type(final_result) == accept["template_type"]
        checks["runtime_result_absent"] = (final_result.runtime_result is None) == bool(accept["runtime_result_absent"])
    elif kind == "runtime_reject":
        checks["summary_type_match"] = extract_summary_type(final_result) == accept["summary_type"]
        checks["included_suggestion_types_match"] = extract_suggestion_types(final_result) == list(
            accept["included_suggestion_types"]
        )
        checks["runtime_result_present"] = final_result.runtime_result is not None
    elif kind == "reset_no_merge":
        turn1 = results[0]
        turn2 = results[1]
        checks["turn1_state_match"] = extract_public_state(turn1) == accept["turn1_final_state"]
        checks["turn2_state_match"] = extract_public_state(turn2) == accept["turn2_final_state"]
        checks["merge_applied_match"] = not bool(turn2.artifact_record and turn2.artifact_record.merge_applied)
        checks["parent_turn_link_absent"] = not bool(turn2.artifact_record and turn2.artifact_record.parent_turn_id)
        checks["carried_fields_empty"] = extract_carried_fields(turn2) == []
        checks["turn2_runtime_presence_match"] = (turn2.runtime_result is not None) == bool(
            accept["expected_turn2_runtime_presence"]
        )
        checks["no_stale_turn1_reuse"] = check_reset_no_stale_reuse(turn1=turn1, turn2=turn2, accept=accept)
    return checks


def extract_public_state(result) -> str:
    if result.response_decision is not None:
        return result.response_decision.final_public_state
    return result.stage


def extract_summary_type(result) -> str | None:
    payload = result.explanation_payload
    if payload is None:
        return None
    return getattr(payload, "summary_type", None)


def extract_suggestion_types(result) -> list[str]:
    if result.response_decision is not None:
        return list(result.response_decision.included_suggestion_types)
    payload = result.explanation_payload
    if payload is None:
        return []
    return list(getattr(payload, "included_suggestion_types", []))


def extract_template_type(result) -> str | None:
    if result.response_decision is None:
        return None
    return result.response_decision.template_type


def extract_carried_fields(result) -> list[str]:
    record = getattr(result, "artifact_record", None)
    if record is None:
        return []
    value = getattr(record, "carried_fields", [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def extract_builder_partial_snapshot(result) -> dict[str, object] | None:
    builder_result = getattr(result, "builder_result", None)
    if builder_result is None:
        return None
    value = getattr(builder_result, "partial_profile_snapshot", None)
    return dict(value) if isinstance(value, dict) else None


def extract_normalized_cf_request(result) -> dict[str, object]:
    normalized = getattr(result, "normalized_parse", None)
    if not isinstance(normalized, dict):
        return {}
    value = normalized.get("cf_request")
    return dict(value) if isinstance(value, dict) else {}


def check_reset_no_stale_reuse(*, turn1, turn2, accept: dict[str, object]) -> bool:
    turn2_snapshot = extract_builder_partial_snapshot(turn2) or {}
    turn2_request = extract_normalized_cf_request(turn2)

    expected_turn2_profile = accept.get("expected_turn2_profile")
    if isinstance(expected_turn2_profile, dict):
        return turn2_request == expected_turn2_profile and turn2_snapshot == expected_turn2_profile

    forbidden_fields = accept.get("forbidden_turn2_fields")
    if isinstance(forbidden_fields, list):
        turn1_snapshot = extract_builder_partial_snapshot(turn1) or {}
        forbidden = {str(field) for field in forbidden_fields if isinstance(field, str)}
        turn1_only_fields = forbidden.intersection(turn1_snapshot.keys())
        return not (turn1_only_fields & set(turn2_request.keys()) or turn1_only_fields & set(turn2_snapshot.keys()))

    return True


def build_closeout_summary(
    *,
    args,
    catalog_version: str,
    run_version: str,
    run_root: Path,
    live_output_root: Path,
    mode: str,
    automated_suites: list[AutomatedSuiteResult],
    live_scenarios: list[LiveScenarioOutcome],
) -> dict[str, object]:
    failed_gates: list[str] = []
    if mode != "both":
        failed_gates.append("full_closeout_requires_mode_both")
    if not automated_suites:
        failed_gates.append("automated_suites_not_run")
    elif any(not suite.passed for suite in automated_suites):
        failed_gates.append("automated_suites_failed")
    if not live_scenarios:
        failed_gates.append("live_scenarios_not_run")
    elif any(not scenario.passed for scenario in live_scenarios):
        failed_gates.append("live_scenarios_failed")

    ready = not failed_gates
    return {
        "run_version": run_version,
        "mode": mode,
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_alias": args.model_alias,
            "api_base": args.api_base,
            "benchmark_path": str(Path(args.benchmark)),
            "system_prompt_path": str(Path(args.system_prompt)),
            "scenario_catalog_version": catalog_version,
            "scenario_catalog_path": str(Path(args.catalog).resolve()),
            "output_root": str(run_root.resolve()),
            "live_output_root": str(live_output_root.resolve()),
        },
        "automated_suites": [asdict(suite) for suite in automated_suites],
        "live_scenarios": [asdict(scenario) for scenario in live_scenarios],
        "ready_to_start_phase3_2": ready,
        "failed_gates": failed_gates,
        "phase3_2_handoff": list(PHASE32_HANDOFF_ITEMS),
    }


def write_closeout_summary(run_root: Path, summary: dict[str, object]) -> None:
    json_path = run_root / "phase3_1_closeout_summary.json"
    markdown_path = run_root / "phase3_1_closeout_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 3.1 Closeout Summary",
        "",
        f"- run_version: {summary['run_version']}",
        f"- mode: {summary['mode']}",
        f"- ready_to_start_phase3_2: {summary['ready_to_start_phase3_2']}",
        f"- failed_gates: {', '.join(summary['failed_gates']) if summary['failed_gates'] else '-'}",
        "",
        "## Automated Suites",
        "",
        "| Suite | Exit Code | Passed | Summary |",
        "| --- | --- | --- | --- |",
    ]
    for suite in summary["automated_suites"]:
        lines.append(
            "| {suite_name} | {exit_code} | {passed} | {summary_line} |".format(
                suite_name=suite["suite_name"],
                exit_code=suite["exit_code"],
                passed=suite["passed"],
                summary_line=" / ".join(suite.get("summary_lines", [])) or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Live Scenarios",
            "",
            "| Scenario | Expected | Actual | Passed | Turns | Key Checks |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in summary["live_scenarios"]:
        passed_checks = sum(1 for value in scenario["key_payload_checks"].values() if value)
        total_checks = len(scenario["key_payload_checks"])
        lines.append(
            "| {scenario_id} | {expected} | {actual} | {passed} | {turn_count} | {checks} |".format(
                scenario_id=scenario["scenario_id"],
                expected=scenario["expected_final_state"],
                actual=scenario["actual_final_state"],
                passed=scenario["passed"],
                turn_count=scenario["turn_count"],
                checks=f"{passed_checks}/{total_checks}",
            )
        )
    lines.extend(
        [
            "",
            "## Phase 3.2 Handoff",
            "",
        ]
    )
    for item in summary["phase3_2_handoff"]:
        lines.append(f"- {item}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_standalone_report(run_root: Path, summary: dict[str, object]) -> None:
    report_path = run_root / "phase3_1_standalone_report.md"
    provenance = summary["provenance"]
    lines = [
        "# Phase 3.1 Closeout Standalone Report",
        "",
        "## Provenance",
        "",
        f"- timestamp_utc: {provenance['timestamp_utc']}",
        f"- model_alias: {provenance['model_alias']}",
        f"- api_base: {provenance['api_base']}",
        f"- benchmark_path: {provenance['benchmark_path']}",
        f"- system_prompt_path: {provenance['system_prompt_path']}",
        f"- scenario_catalog_version: {provenance['scenario_catalog_version']}",
        f"- output_root: {provenance['output_root']}",
        "",
        "## Automated Suites",
        "",
    ]
    for suite in summary["automated_suites"]:
        lines.extend(
            [
                f"### `{suite['suite_name']}`",
                "",
                f"- command: `{suite['command']}`",
                f"- exit_code: `{suite['exit_code']}`",
                f"- passed: `{suite['passed']}`",
                "- summary_lines:",
            ]
        )
        for line in suite.get("summary_lines", []):
            lines.append(f"  - {line}")
        lines.append("")

    lines.extend(
        [
            "## Live Scenario Results",
            "",
            "| Scenario | Expected | Actual | Passed | Turns | READY_FOR_RUNTIME Hidden |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in summary["live_scenarios"]:
        lines.append(
            "| {scenario_id} | {expected} | {actual} | {passed} | {turn_count} | {ready_hidden} |".format(
                scenario_id=scenario["scenario_id"],
                expected=scenario["expected_final_state"],
                actual=scenario["actual_final_state"],
                passed=scenario["passed"],
                turn_count=scenario["turn_count"],
                ready_hidden=scenario["expected_public_ready_hidden"],
            )
        )

    lines.extend(
        [
            "",
            "## Detailed Live Scenarios",
            "",
        ]
    )
    for scenario in summary["live_scenarios"]:
        lines.extend(render_live_scenario_report(scenario))

    lines.extend(
        [
            "## Readiness Verdict",
            "",
            f"- ready_to_start_phase3_2: `{summary['ready_to_start_phase3_2']}`",
        ]
    )
    if summary["failed_gates"]:
        lines.append("- failed_gates:")
        for gate in summary["failed_gates"]:
            lines.append(f"  - `{gate}`")
    else:
        lines.append("- failed_gates: none")

    lines.extend(
        [
            "",
            "## Phase 3.2 Handoff",
            "",
        ]
    )
    for item in summary["phase3_2_handoff"]:
        lines.append(f"- {item}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_live_scenario_report(scenario: dict[str, object]) -> list[str]:
    lines = [
        f"### `{scenario['scenario_id']}`",
        "",
        f"- description: {scenario['description']}",
        f"- expected_final_state: `{scenario['expected_final_state']}`",
        f"- actual_final_state: `{scenario['actual_final_state']}`",
        f"- passed: `{scenario['passed']}`",
        f"- turns: `{scenario['turn_count']}`",
        f"- stages: `{ ' -> '.join(scenario['stages']) }`",
        f"- READY_FOR_RUNTIME remained internal-only: `{scenario['expected_public_ready_hidden']}`",
        "- key_payload_checks:",
    ]
    for key, value in scenario["key_payload_checks"].items():
        lines.append(f"  - `{key}`: `{value}`")
    lines.append("- artifact_folders:")
    for folder in scenario["artifact_folders"]:
        lines.append(f"  - `{folder}`")
    lines.append("")
    return lines


def write_case_study_indexes(output_root: Path) -> None:
    records = collect_case_studies(output_root)
    (output_root / "case_studies_index.json").write_text(
        json.dumps(records, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "case_studies_index.md").write_text(render_markdown(records), encoding="utf-8")


def build_runner_command(args) -> str:
    command = (
        "python scripts/run_phase3_1_closeout_suite.py "
        f"--mode {args.mode} "
        f"--model-alias {args.model_alias} "
        f"--api-base {args.api_base} "
        f"--timeout-s {args.timeout_s} "
        f"--benchmark {args.benchmark} "
        f"--system-prompt {args.system_prompt} "
        f"--catalog {args.catalog} "
        f"--out-dir {args.out_dir}"
    )
    if args.debug_trace:
        command += " --debug-trace"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
