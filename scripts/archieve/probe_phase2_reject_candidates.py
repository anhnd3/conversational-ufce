#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
from llm.src.phase2.catalog import DEFAULT_CATALOG_PATH, load_catalog  # noqa: E402
from llm.src.phase2.taxonomy import classify_turn_result  # noqa: E402
from llm.src.utils.io import write_json  # noqa: E402


@dataclass(frozen=True)
class CandidatePrompt:
    case_id: str
    slug: str
    text: str


@dataclass(frozen=True)
class CandidateOutcome:
    case_id: str
    slug: str
    text: str
    run_labels: list[str]
    run_stages: list[str]
    reason_codes: list[list[str]]
    changed_fields: list[list[str]]
    stable_runtime_reject: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe replacement reject candidates under Phase 2 pack-order warmup.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--out-dir", default="outputs/reject_candidate_probes")
    return parser


def build_default_candidates() -> list[CandidatePrompt]:
    template = (
        "My target profile is Income {income}, CCAvg {ccavg}, Family 2, Education 1, "
        "Mortgage 0, CDAccount yes, Online yes, SecuritiesAccount no, and CreditCard yes."
    )
    return [
        CandidatePrompt(
            case_id="RJ-PROBE-01",
            slug="control_a04_income65_ccavg3_0",
            text=template.format(income="65", ccavg="3.0"),
        ),
        CandidatePrompt(
            case_id="RJ-PROBE-02",
            slug="income60_ccavg2_6",
            text=template.format(income="60", ccavg="2.6"),
        ),
        CandidatePrompt(
            case_id="RJ-PROBE-03",
            slug="income58_ccavg2_2",
            text=template.format(income="58", ccavg="2.2"),
        ),
        CandidatePrompt(
            case_id="RJ-PROBE-04",
            slug="income55_ccavg2_0",
            text=template.format(income="55", ccavg="2.0"),
        ),
        CandidatePrompt(
            case_id="RJ-PROBE-05",
            slug="income50_ccavg1_8",
            text=template.format(income="50", ccavg="1.8"),
        ),
        CandidatePrompt(
            case_id="RJ-PROBE-06",
            slug="income45_ccavg1_2",
            text=template.format(income="45", ccavg="1.2"),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    catalog = load_catalog(args.catalog)
    output_root = Path(args.out_dir)
    probe_root = output_root / build_probe_id()
    probe_root.mkdir(parents=True, exist_ok=False)

    warmup_cases = [
        catalog.get_case("P-NR-01"),
        catalog.get_case("P-NR-02"),
        catalog.get_case("P-CF-01"),
        catalog.get_case("P-CF-02"),
        catalog.get_case("P-RJ-01"),
    ]

    outcomes: list[CandidateOutcome] = []
    for candidate in build_default_candidates():
        orchestrator = build_orchestrator(args, output_root=probe_root)
        warmup_sequence(orchestrator, warmup_cases)
        outcomes.append(run_candidate(orchestrator, candidate))

    summary = {
        "probe_root": str(probe_root.resolve()),
        "catalog_version": catalog.catalog_version,
        "warmup_case_ids": [case.case_id for case in warmup_cases],
        "candidate_outcomes": [asdict(outcome) for outcome in outcomes],
        "stable_reject_candidates": [
            {
                "case_id": outcome.case_id,
                "slug": outcome.slug,
                "text": outcome.text,
            }
            for outcome in outcomes
            if outcome.stable_runtime_reject
        ],
    }
    write_json(probe_root / "reject_probe_summary.json", summary)
    (probe_root / "reject_probe_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def build_orchestrator(args, *, output_root: Path) -> BankConversationOrchestrator:
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


def warmup_sequence(orchestrator: BankConversationOrchestrator, cases) -> None:
    for case in cases:
        for run_index in (1, 2):
            orchestrator.run_turn(
                user_input=case.turns[0],
                save_artifacts=False,
                scenario_slug=f"warmup_{case.case_id.lower()}__run{run_index}",
            )


def run_candidate(orchestrator: BankConversationOrchestrator, candidate: CandidatePrompt) -> CandidateOutcome:
    run_labels: list[str] = []
    run_stages: list[str] = []
    reason_codes: list[list[str]] = []
    changed_fields: list[list[str]] = []
    for run_index in (1, 2):
        result = orchestrator.run_turn(
            user_input=candidate.text,
            save_artifacts=False,
            scenario_slug=f"{candidate.case_id.lower()}__{candidate.slug}__run{run_index}",
        )
        run_labels.append(classify_turn_result(stage=result.stage, explanation_payload=result.explanation_payload))
        run_stages.append(result.stage)
        payload = result.explanation_payload
        if payload is None:
            reason_codes.append([])
            changed_fields.append([])
        else:
            reason_codes.append(list(payload.reason_codes))
            changed_fields.append(list(payload.changed_fields))
    return CandidateOutcome(
        case_id=candidate.case_id,
        slug=candidate.slug,
        text=candidate.text,
        run_labels=run_labels,
        run_stages=run_stages,
        reason_codes=reason_codes,
        changed_fields=changed_fields,
        stable_runtime_reject=(
            run_labels == ["runtime_reject", "runtime_reject"]
            and run_stages == ["RUNTIME_REJECT", "RUNTIME_REJECT"]
            and reason_codes == [["NO_FEASIBLE_CF_FOUND"], ["NO_FEASIBLE_CF_FOUND"]]
        ),
    )


def build_probe_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"phase2_reject_probe_{timestamp}"


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 2 Reject Candidate Probe",
        "",
        f"- catalog_version: {summary['catalog_version']}",
        f"- probe_root: {summary['probe_root']}",
        "",
        "Warmup sequence:",
        "",
    ]
    for case_id in summary["warmup_case_ids"]:
        lines.append(f"- {case_id}")
    lines.extend(
        [
            "",
            "| Case ID | Slug | Stable Reject | Run Labels | Run Stages | Reason Codes | Changed Fields |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for outcome in summary["candidate_outcomes"]:
        labels = " / ".join(outcome["run_labels"])
        stages = " / ".join(outcome["run_stages"])
        reasons = " / ".join(",".join(item) or "-" for item in outcome["reason_codes"])
        changes = " / ".join(",".join(item) or "-" for item in outcome["changed_fields"])
        lines.append(
            f"| {outcome['case_id']} | {outcome['slug']} | {outcome['stable_runtime_reject']} | {labels} | {stages} | {reasons} | {changes} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
