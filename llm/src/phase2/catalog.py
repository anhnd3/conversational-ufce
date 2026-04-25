from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = ROOT / "docs" / "thesis" / "part2" / "catalogs" / "phase2_bank_catalog_v2.json"
RUNNER_COMPAT_VERSION = "phase2_pack_runner_v1"

PRIMARY_CASE_ID_RE = re.compile(r"^P-(NR|CF|RJ|CL)-\d{2}$")
SUPPLEMENTAL_CASE_ID_RE = re.compile(r"^S-MERGE-\d{2}$")
SMOKE_CASE_ID_RE = re.compile(r"^SM-[A-Z0-9-]+$")
CATALOG_VERSION_RE = re.compile(r"^[a-z0-9._-]+$")


@dataclass(frozen=True)
class CatalogCase:
    case_id: str
    slug: str
    description: str
    expected_label: str
    turns: tuple[str, ...]
    accept: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "slug": self.slug,
            "description": self.description,
            "expected_label": self.expected_label,
            "turns": list(self.turns),
            "accept": dict(self.accept),
        }


@dataclass(frozen=True)
class Phase2ScenarioCatalog:
    catalog_version: str
    runner_compat_version: str
    created_timestamp_utc: str
    prompt_template_version: str
    change_notes: list[str]
    primary_cases: tuple[CatalogCase, ...]
    supplemental_cases: tuple[CatalogCase, ...]
    smoke_only_cases: tuple[CatalogCase, ...]
    source_path: Path

    def iter_all_cases(self) -> tuple[CatalogCase, ...]:
        return self.primary_cases + self.supplemental_cases + self.smoke_only_cases

    def get_case(self, case_id: str) -> CatalogCase:
        for case in self.iter_all_cases():
            if case.case_id == case_id:
                return case
        raise KeyError(f"Unknown catalog case_id: {case_id}")


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> Phase2ScenarioCatalog:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    validate_catalog_payload(payload)
    return Phase2ScenarioCatalog(
        catalog_version=str(payload["catalog_version"]),
        runner_compat_version=str(payload["runner_compat_version"]),
        created_timestamp_utc=str(payload["created_timestamp_utc"]),
        prompt_template_version=str(payload["prompt_template_version"]),
        change_notes=[str(item) for item in payload.get("change_notes", [])],
        primary_cases=tuple(_build_case(item) for item in payload["primary_cases"]),
        supplemental_cases=tuple(_build_case(item) for item in payload["supplemental_cases"]),
        smoke_only_cases=tuple(_build_case(item) for item in payload["smoke_only_cases"]),
        source_path=catalog_path.resolve(),
    )


def validate_catalog_payload(payload: dict[str, Any]) -> None:
    required_top_level = (
        "catalog_version",
        "runner_compat_version",
        "created_timestamp_utc",
        "prompt_template_version",
        "change_notes",
        "primary_cases",
        "supplemental_cases",
        "smoke_only_cases",
    )
    for key in required_top_level:
        if key not in payload:
            raise ValueError(f"Scenario catalog missing required key: {key}")

    catalog_version = payload["catalog_version"]
    if not isinstance(catalog_version, str) or not CATALOG_VERSION_RE.match(catalog_version):
        raise ValueError("catalog_version must be a non-empty lowercase version token.")

    compat_version = payload["runner_compat_version"]
    if compat_version != RUNNER_COMPAT_VERSION:
        raise ValueError(
            f"Scenario catalog runner_compat_version must equal {RUNNER_COMPAT_VERSION}, got {compat_version!r}."
        )

    change_notes = payload["change_notes"]
    if not isinstance(change_notes, list):
        raise ValueError("change_notes must be a list.")

    seen_case_ids: set[str] = set()
    for key, id_re in (
        ("primary_cases", PRIMARY_CASE_ID_RE),
        ("supplemental_cases", SUPPLEMENTAL_CASE_ID_RE),
        ("smoke_only_cases", SMOKE_CASE_ID_RE),
    ):
        cases = payload[key]
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{key} must be a non-empty list.")
        for item in cases:
            validate_case_payload(item, id_re=id_re, seen_case_ids=seen_case_ids)


def validate_case_payload(
    item: dict[str, Any],
    *,
    id_re: re.Pattern[str],
    seen_case_ids: set[str],
) -> None:
    required_keys = ("case_id", "slug", "description", "expected_label", "turns", "accept")
    for key in required_keys:
        if key not in item:
            raise ValueError(f"Scenario entry missing required key: {key}")

    case_id = item["case_id"]
    if not isinstance(case_id, str) or not id_re.match(case_id):
        raise ValueError(f"Invalid case_id format: {case_id!r}")
    if case_id in seen_case_ids:
        raise ValueError(f"Duplicate case_id in scenario catalog: {case_id}")
    seen_case_ids.add(case_id)

    turns = item["turns"]
    if not isinstance(turns, list) or not turns or any(not isinstance(turn, str) or not turn.strip() for turn in turns):
        raise ValueError(f"Scenario {case_id} must have a non-empty turns list of strings.")

    accept = item["accept"]
    if not isinstance(accept, dict):
        raise ValueError(f"Scenario {case_id} accept metadata must be an object.")

    expected_label = item["expected_label"]
    if not isinstance(expected_label, str) or not expected_label:
        raise ValueError(f"Scenario {case_id} expected_label must be a non-empty string.")

    if case_id.startswith("P-NR-") or case_id.startswith("P-RJ-"):
        _require_list(accept, case_id, "reason_codes")
    elif case_id.startswith("P-CF-"):
        _require_list(accept, case_id, "changed_fields")
    elif case_id.startswith("P-CL-"):
        _require_string(accept, case_id, "clarification_type")
        _require_list(accept, case_id, "missing_fields")
    elif case_id.startswith("S-MERGE-"):
        _require_string(accept, case_id, "supplemental_type")
        _require_string(accept, case_id, "turn1_stage")
        _require_list(accept, case_id, "carried_fields")
        if accept.get("supplemental_type") == "supplemental_followup_still_incomplete":
            _require_string(accept, case_id, "clarification_type")
            _require_list(accept, case_id, "missing_fields")
        else:
            _require_string(accept, case_id, "final_label")
    elif case_id.startswith("SM-"):
        _require_string(accept, case_id, "final_label")
        _require_string(accept, case_id, "turn1_stage")


def _require_string(payload: dict[str, Any], case_id: str, key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Scenario {case_id} accept metadata requires non-empty string field: {key}")


def _require_list(payload: dict[str, Any], case_id: str, key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Scenario {case_id} accept metadata requires list field: {key}")


def _build_case(item: dict[str, Any]) -> CatalogCase:
    return CatalogCase(
        case_id=str(item["case_id"]),
        slug=str(item["slug"]),
        description=str(item["description"]),
        expected_label=str(item["expected_label"]),
        turns=tuple(str(turn) for turn in item["turns"]),
        accept=dict(item["accept"]),
    )
