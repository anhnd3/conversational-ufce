from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetField:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class OutputContract:
    task: str
    status_enum: tuple[str, ...]
    rules: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    group: str
    description: str
    input_text: str
    expected_output: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkDefinition:
    benchmark_name: str
    description: str
    target_cf_fields: tuple[TargetField, ...]
    output_contract: OutputContract
    cases: tuple[BenchmarkCase, ...]

    @property
    def allowed_field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.target_cf_fields)

    @property
    def field_type_map(self) -> dict[str, str]:
        return {field.name: field.type for field in self.target_cf_fields}

    @property
    def case_map(self) -> dict[str, BenchmarkCase]:
        return {case.case_id: case for case in self.cases}


@dataclass(frozen=True)
class EvalConfig:
    benchmark_path: Path
    model_alias: str
    out_dir: Path
    api_base: str
    repeats: int
    temperature: float
    top_p: float
    max_tokens: int
    timeout_s: float
    case_ids: tuple[str, ...]
    group: str | None
    limit: int | None
    system_prompt_path: Path

