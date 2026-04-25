from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = ROOT / "docs" / "validation" / "catalogs" / "phase3_2_validation_catalog_v1.json"
RUNNER_COMPAT_VERSION = "phase3_2_acceptance_runner_v1"
SCENARIO_ID_RE = re.compile(r"^P32-[A-Z]{2}-\d{2}$")
CATALOG_VERSION_RE = re.compile(r"^[a-z0-9._-]+$")


@dataclass(frozen=True)
class Phase32ValidationScenario:
    scenario_id: str
    slug: str
    description: str
    turns: tuple[str, ...]
    expected_final_state: str
    accept: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "slug": self.slug,
            "description": self.description,
            "turns": list(self.turns),
            "expected_final_state": self.expected_final_state,
            "accept": dict(self.accept),
        }


@dataclass(frozen=True)
class Phase32ValidationCatalog:
    catalog_version: str
    runner_compat_version: str
    created_timestamp_utc: str
    prompt_template_version: str
    change_notes: list[str]
    scenarios: tuple[Phase32ValidationScenario, ...]
    source_path: Path

    def get_scenario(self, scenario_id: str) -> Phase32ValidationScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"Unknown Phase 3.2 validation scenario: {scenario_id}")


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> Phase32ValidationCatalog:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    validate_catalog_payload(payload)
    return Phase32ValidationCatalog(
        catalog_version=str(payload["catalog_version"]),
        runner_compat_version=str(payload["runner_compat_version"]),
        created_timestamp_utc=str(payload["created_timestamp_utc"]),
        prompt_template_version=str(payload["prompt_template_version"]),
        change_notes=[str(item) for item in payload.get("change_notes", [])],
        scenarios=tuple(_build_scenario(item) for item in payload["scenarios"]),
        source_path=catalog_path.resolve(),
    )


def validate_catalog_payload(payload: dict[str, Any]) -> None:
    required_top_level = (
        "catalog_version",
        "runner_compat_version",
        "created_timestamp_utc",
        "prompt_template_version",
        "change_notes",
        "scenarios",
    )
    for key in required_top_level:
        if key not in payload:
            raise ValueError(f"Phase 3.2 validation catalog missing required key: {key}")

    catalog_version = payload["catalog_version"]
    if not isinstance(catalog_version, str) or not CATALOG_VERSION_RE.match(catalog_version):
        raise ValueError("catalog_version must be a non-empty lowercase version token.")

    compat_version = payload["runner_compat_version"]
    if compat_version != RUNNER_COMPAT_VERSION:
        raise ValueError(
            f"Phase 3.2 validation catalog runner_compat_version must equal {RUNNER_COMPAT_VERSION}, got {compat_version!r}."
        )

    change_notes = payload["change_notes"]
    if not isinstance(change_notes, list):
        raise ValueError("change_notes must be a list.")

    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list.")

    seen_ids: set[str] = set()
    for item in scenarios:
        validate_scenario_payload(item, seen_ids=seen_ids)


def validate_scenario_payload(item: dict[str, Any], *, seen_ids: set[str]) -> None:
    required_keys = ("scenario_id", "slug", "description", "turns", "expected_final_state", "accept")
    for key in required_keys:
        if key not in item:
            raise ValueError(f"Phase 3.2 validation scenario missing required key: {key}")

    scenario_id = item["scenario_id"]
    if not isinstance(scenario_id, str) or not SCENARIO_ID_RE.match(scenario_id):
        raise ValueError(f"Invalid Phase 3.2 validation scenario_id: {scenario_id!r}")
    if scenario_id in seen_ids:
        raise ValueError(f"Duplicate Phase 3.2 validation scenario_id: {scenario_id}")
    seen_ids.add(scenario_id)

    turns = item["turns"]
    if not isinstance(turns, list) or not turns or any(not isinstance(turn, str) or not turn.strip() for turn in turns):
        raise ValueError(f"Scenario {scenario_id} must have a non-empty turns list of strings.")

    expected_final_state = item["expected_final_state"]
    if not isinstance(expected_final_state, str) or not expected_final_state:
        raise ValueError(f"Scenario {scenario_id} expected_final_state must be a non-empty string.")

    accept = item["accept"]
    if not isinstance(accept, dict):
        raise ValueError(f"Scenario {scenario_id} accept metadata must be an object.")

    _require_string(accept, scenario_id, "kind")
    kind = str(accept["kind"])
    if kind == "no_recourse_needed":
        _require_string(accept, scenario_id, "summary_type")
        _require_list(accept, scenario_id, "included_suggestion_types")
    elif kind == "counterfactual_found":
        _require_string(accept, scenario_id, "summary_type")
    elif kind in {"clarification_merge_success", "clarification_still_incomplete"}:
        _require_string(accept, scenario_id, "turn1_final_state")
        _require_string(accept, scenario_id, "turn2_final_state")
        _require_bool(accept, scenario_id, "turn2_merge_applied")
    elif kind == "conflict":
        _require_bool(accept, scenario_id, "runtime_result_absent")
    elif kind == "unsupported":
        _require_string(accept, scenario_id, "template_type")
        _require_bool(accept, scenario_id, "runtime_result_absent")
    elif kind == "runtime_reject":
        _require_string(accept, scenario_id, "summary_type")
        _require_list(accept, scenario_id, "included_suggestion_types")
    elif kind == "reset_no_merge":
        _require_string(accept, scenario_id, "turn1_final_state")
        _require_string(accept, scenario_id, "turn2_final_state")
        _require_bool(accept, scenario_id, "expected_turn2_runtime_presence")
        if accept.get("expected_turn2_runtime_presence") is True:
            _require_dict(accept, scenario_id, "expected_turn2_profile")
        else:
            _require_list(accept, scenario_id, "forbidden_turn2_fields")
    else:
        raise ValueError(f"Scenario {scenario_id} has unsupported kind: {kind!r}")


def _require_string(payload: dict[str, Any], scenario_id: str, key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Scenario {scenario_id} accept metadata requires non-empty string field: {key}")


def _require_list(payload: dict[str, Any], scenario_id: str, key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Scenario {scenario_id} accept metadata requires list field: {key}")


def _require_bool(payload: dict[str, Any], scenario_id: str, key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Scenario {scenario_id} accept metadata requires boolean field: {key}")


def _require_dict(payload: dict[str, Any], scenario_id: str, key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Scenario {scenario_id} accept metadata requires object field: {key}")


def _build_scenario(item: dict[str, Any]) -> Phase32ValidationScenario:
    return Phase32ValidationScenario(
        scenario_id=str(item["scenario_id"]),
        slug=str(item["slug"]),
        description=str(item["description"]),
        turns=tuple(str(turn) for turn in item["turns"]),
        expected_final_state=str(item["expected_final_state"]),
        accept=dict(item["accept"]),
    )
