#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso

DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase3_2_product_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3.2 product smoke suite against the local service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    catalog = load_catalog(args.catalog)
    run_id = "phase3_2_product_smoke_" + local_now_compact()
    output_root = args.out_dir / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    _assert_ok(session.get(f"{base_url}/"))

    results: list[dict[str, Any]] = []
    all_passed = True
    for scenario in catalog.scenarios:
        created = _assert_ok(session.post(f"{base_url}/api/v1/sessions")).json()
        session_id = created["session_id"]
        turn_payloads = []
        for turn_text in scenario.turns:
            response = _assert_ok(
                session.post(
                    f"{base_url}/api/v1/sessions/{session_id}/messages",
                    json={"user_input": turn_text},
                )
            )
            turn_payloads.append(response.json())
        _assert_ok(session.get(f"{base_url}/sessions/{session_id}"))
        artifact_listing = _assert_ok(session.get(f"{base_url}/api/v1/sessions/{session_id}/artifacts")).json()
        if artifact_listing:
            first_bundle = artifact_listing[-1]
            if first_bundle["files"]:
                first_file = first_bundle["files"][0]
                _assert_ok(
                    session.get(
                        f"{base_url}/api/v1/sessions/{session_id}/artifacts/{first_bundle['turn_id']}/{first_file}"
                    )
                )
        checks = evaluate_scenario(scenario.accept, turn_payloads)
        passed = (
            turn_payloads[-1]["public_state"] == scenario.expected_final_state
            and all(checks.values())
            and bool(artifact_listing)
        )
        results.append(
            {
                "scenario_id": scenario.scenario_id,
                "slug": scenario.slug,
                "expected_final_state": scenario.expected_final_state,
                "actual_final_state": turn_payloads[-1]["public_state"],
                "passed": passed,
                "checks": checks,
                "session_id": session_id,
            }
        )
        all_passed = all_passed and passed

    summary = {
        "run_id": run_id,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "base_url": base_url,
        "catalog_version": catalog.catalog_version,
        "all_passed": all_passed,
        "results": results,
    }
    write_json(output_root / "phase3_2_product_smoke_summary.json", summary)
    (output_root / "phase3_2_product_smoke_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_passed else 1


def evaluate_scenario(accept: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, bool]:
    kind = accept["kind"]
    final_turn = turns[-1]
    checks = {
        "ready_for_runtime_hidden": final_turn["public_state"] != "READY_FOR_RUNTIME",
    }
    if kind == "no_recourse_needed":
        checks["summary_type_match"] = final_turn["explanation_payload"]["summary_type"] == accept["summary_type"]
    elif kind == "counterfactual_found":
        checks["summary_type_match"] = final_turn["explanation_payload"]["summary_type"] == accept["summary_type"]
    elif kind == "clarification_merge_success":
        checks["turn1_state_match"] = turns[0]["public_state"] == accept["turn1_final_state"]
        checks["turn2_state_match"] = turns[1]["public_state"] == accept["turn2_final_state"]
        checks["merge_applied_match"] = turns[1]["debug_summary"]["merge_applied"] == accept["turn2_merge_applied"]
    elif kind == "clarification_still_incomplete":
        checks["turn1_state_match"] = turns[0]["public_state"] == accept["turn1_final_state"]
        checks["turn2_state_match"] = turns[1]["public_state"] == accept["turn2_final_state"]
        checks["merge_applied_match"] = turns[1]["debug_summary"]["merge_applied"] == accept["turn2_merge_applied"]
    elif kind == "conflict":
        checks["runtime_result_absent"] = turns[-1]["debug_summary"]["runtime_summary"]["executed"] is False
    elif kind == "unsupported":
        checks["runtime_result_absent"] = turns[-1]["debug_summary"]["runtime_summary"]["executed"] is False
    elif kind == "runtime_reject":
        checks["summary_type_match"] = final_turn["explanation_payload"]["summary_type"] == accept["summary_type"]
        checks["suggestions_match"] = (
            list(final_turn["explanation_payload"]["included_suggestion_types"]) == list(accept["included_suggestion_types"])
        )
    elif kind == "reset_no_merge":
        checks["turn1_state_match"] = turns[0]["public_state"] == accept["turn1_final_state"]
        checks["turn2_state_match"] = turns[1]["public_state"] == accept["turn2_final_state"]
        checks["merge_applied_false"] = turns[1]["debug_summary"]["merge_applied"] is False
    return checks


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3.2 Product Smoke",
        "",
        f"- base_url: `{summary['base_url']}`",
        f"- catalog_version: `{summary['catalog_version']}`",
        f"- all_passed: `{summary['all_passed']}`",
        "",
        "| Scenario | Expected | Actual | Passed |",
        "| --- | --- | --- | --- |",
    ]
    for item in summary["results"]:
        lines.append(
            f"| `{item['scenario_id']}` | `{item['expected_final_state']}` | `{item['actual_final_state']}` | `{item['passed']}` |"
        )
    return "\n".join(lines) + "\n"


def _assert_ok(response: requests.Response) -> requests.Response:
    response.raise_for_status()
    return response


if __name__ == "__main__":
    raise SystemExit(main())
