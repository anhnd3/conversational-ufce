#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.conversation.parser_adapter import DEFAULT_API_BASE, DEFAULT_MODEL_ALIAS
from llm.src.conversation.session import create_interactive_session_state, handle_session_turn
from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.reproducibility import RUNTIME_MODE_STABLE_DEMO
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase3_2_reproducibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Phase 3.2 reproducibility across the accepted demo corpus.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    run_id = "phase3_2_repro_" + local_now_compact()
    output_root = args.out_dir / run_id
    live_output_root = output_root / "artifacts"
    live_output_root.mkdir(parents=True, exist_ok=True)

    orchestrator = BankConversationOrchestrator(
        model_alias=args.model_alias,
        output_root=live_output_root,
        runtime_orchestrator=RuntimeOrchestrator(runtime_mode=RUNTIME_MODE_STABLE_DEMO),
    )
    setattr(orchestrator.parser_adapter, "api_base", args.api_base.rstrip("/"))

    scenario_results: list[dict[str, Any]] = []
    all_passed = True
    for scenario in catalog.scenarios:
        signatures = []
        runs = []
        for repeat_index in range(1, args.repeats + 1):
            state = create_interactive_session_state(f"repro_{scenario.scenario_id}_{repeat_index}")
            turn_results = []
            for turn_index, turn_text in enumerate(scenario.turns, start=1):
                turn_results.append(
                    handle_session_turn(
                        orchestrator,
                        state,
                        user_input=turn_text,
                        save_artifacts=True,
                        scenario_slug=f"{scenario.slug}_repeat{repeat_index}_turn{turn_index}",
                        debug_trace_enabled=False,
                        command="probe_phase3_2_reproducibility",
                    )
                )
            final = turn_results[-1]
            signature = build_signature(final)
            signatures.append(signature)
            runs.append(
                {
                    "repeat_index": repeat_index,
                    "signature": signature,
                    "artifact_dirs": [
                        None if result.artifact_record is None else result.artifact_record.output_dir
                        for result in turn_results
                    ],
                }
            )
        stable = all(item == signatures[0] for item in signatures[1:])
        scenario_results.append(
            {
                "scenario_id": scenario.scenario_id,
                "slug": scenario.slug,
                "stable": stable,
                "runs": runs,
            }
        )
        all_passed = all_passed and stable

    summary = {
        "run_id": run_id,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "catalog_version": catalog.catalog_version,
        "runtime_mode": RUNTIME_MODE_STABLE_DEMO,
        "repeats": args.repeats,
        "all_passed": all_passed,
        "scenario_results": scenario_results,
    }
    write_json(output_root / "phase3_2_reproducibility_summary.json", summary)
    (output_root / "phase3_2_reproducibility_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_passed else 1


def build_signature(result) -> dict[str, Any]:
    explanation = result.explanation_payload
    reason_class = None
    if result.stage == "RUNTIME_REJECT":
        reason_codes = list((result.runtime_result or {}).get("reason_codes") or [])
        if "REQUEST_CONSTRAINTS_BLOCKED" in reason_codes:
            reason_class = "request_constraints_blocked"
        elif "NO_FEASIBLE_CF_FOUND" in reason_codes:
            reason_class = "no_feasible_cf"
        elif "INVALID_COUNTERFACTUAL_BLOCKED" in reason_codes:
            reason_class = "invariant_blocked"
        else:
            reason_class = "system_error"
    invariant_status = None
    if isinstance(result.invariant_validation, dict):
        invariant_status = result.invariant_validation.get("status")
    return {
        "final_public_state": result.response_decision.final_public_state if result.response_decision else result.stage,
        "summary_type": None if explanation is None else explanation.summary_type,
        "reject_reason_class": reason_class,
        "invariant_validation_status": invariant_status,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3.2 Reproducibility Probe",
        "",
        f"- catalog_version: `{summary['catalog_version']}`",
        f"- runtime_mode: `{summary['runtime_mode']}`",
        f"- repeats: `{summary['repeats']}`",
        f"- all_passed: `{summary['all_passed']}`",
        "",
        "| Scenario | Stable |",
        "| --- | --- |",
    ]
    for item in summary["scenario_results"]:
        lines.append(f"| `{item['scenario_id']}` | `{item['stable']}` |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
