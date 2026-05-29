from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec


REQUIRED_TOP_LEVEL_KEYS = ("task", "status", "cf_request", "missing_fields", "conflicts", "notes")
OPTIONAL_TOP_LEVEL_KEYS = ("constraint_spec", "field_evidence")
ALLOWED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS + OPTIONAL_TOP_LEVEL_KEYS
EVIDENCE_REQUIRED_KEYS = ("source_text", "evidence_kind", "normalized_from", "language")
EVIDENCE_KIND_ENUM = {
    "explicit_numeric",
    "explicit_boolean",
    "unit_converted",
    "taxonomy_mapped",
    "conflict",
}
EVIDENCE_LANGUAGE_ENUM = {"en", "vi", "mixed", "unknown"}


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...]
    unexpected_top_level_keys: tuple[str, ...]
    unexpected_cf_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "unexpected_top_level_keys": list(self.unexpected_top_level_keys),
            "unexpected_cf_fields": list(self.unexpected_cf_fields),
        }


def validate_prediction(
    candidate: dict[str, Any] | None,
    benchmark,
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
    require_field_evidence: bool | None = None,
) -> ValidationResult:
    errors: list[str] = []
    unexpected_top_level_keys: tuple[str, ...] = ()
    unexpected_cf_fields: tuple[str, ...] = ()

    if candidate is None:
        return ValidationResult(
            is_valid=False,
            errors=("No parsed JSON object available.",),
            unexpected_top_level_keys=(),
            unexpected_cf_fields=(),
        )

    if require_field_evidence is None:
        require_field_evidence = benchmark_requires_field_evidence(benchmark)

    missing_top_level = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in candidate]
    if require_field_evidence and "field_evidence" not in candidate:
        missing_top_level.append("field_evidence")
    unexpected_top_level_keys = tuple(
        sorted(key for key in candidate if key not in ALLOWED_TOP_LEVEL_KEYS)
    )
    if missing_top_level:
        errors.append(f"Missing top-level keys: {', '.join(missing_top_level)}")
    if unexpected_top_level_keys:
        errors.append(f"Unexpected top-level keys: {', '.join(unexpected_top_level_keys)}")

    if candidate.get("task") != benchmark.output_contract.task:
        errors.append(f"task must equal {benchmark.output_contract.task!r}")

    status = candidate.get("status")
    if status not in benchmark.output_contract.status_enum:
        errors.append(
            "status must be one of "
            + ", ".join(repr(value) for value in benchmark.output_contract.status_enum)
        )

    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        errors.append("cf_request must be an object.")
    else:
        unexpected_cf_fields = tuple(
            sorted(field for field in cf_request if field not in benchmark.allowed_field_names)
        )
        if unexpected_cf_fields:
            errors.append(f"Unexpected cf_request fields: {', '.join(unexpected_cf_fields)}")
        field_types = benchmark.field_type_map
        for field_name, value in cf_request.items():
            expected_type = field_types.get(field_name)
            if expected_type is None:
                continue
            error = validate_field_value(field_name, expected_type, value)
            if error:
                errors.append(error)

    field_evidence = candidate.get("field_evidence")
    validated_evidence: dict[str, dict[str, Any]] = {}
    if field_evidence is not None and not isinstance(field_evidence, dict):
        errors.append("field_evidence must be an object when provided.")
    elif isinstance(field_evidence, dict):
        for field_name, evidence in field_evidence.items():
            if field_name not in benchmark.allowed_field_names:
                errors.append(f"field_evidence contains unknown field name: {field_name}")
                continue
            evidence_error = validate_field_evidence(field_name=field_name, evidence=evidence)
            if evidence_error:
                errors.append(evidence_error)
                continue
            validated_evidence[field_name] = dict(evidence)
    if isinstance(cf_request, dict) and (require_field_evidence or isinstance(field_evidence, dict)):
        missing_evidence_fields = [
            field_name
            for field_name in cf_request
            if field_name not in validated_evidence
        ]
        if missing_evidence_fields:
            errors.append(
                "field_evidence must include entries for all cf_request fields: "
                + ", ".join(missing_evidence_fields)
            )

    missing_fields = candidate.get("missing_fields")
    if not is_string_list(missing_fields):
        errors.append("missing_fields must be an array of strings.")
    else:
        invalid_missing_fields = [
            field for field in missing_fields if field not in benchmark.allowed_field_names
        ]
        if invalid_missing_fields:
            errors.append(
                "missing_fields contains unknown field names: "
                + ", ".join(invalid_missing_fields)
            )

    conflicts = candidate.get("conflicts")
    if not is_string_list(conflicts):
        errors.append("conflicts must be an array of strings.")

    notes = candidate.get("notes")
    if not is_string_list(notes):
        errors.append("notes must be an array of strings.")

    constraint_spec = candidate.get("constraint_spec")
    _, constraint_errors = validate_and_normalize_constraint_spec(
        constraint_spec,
        feature_order=[field.name for field in benchmark.target_cf_fields],
        numeric_bound_fields=numeric_bound_fields,
    )
    errors.extend(constraint_errors)

    return ValidationResult(
        is_valid=not errors,
        errors=tuple(errors),
        unexpected_top_level_keys=unexpected_top_level_keys,
        unexpected_cf_fields=unexpected_cf_fields,
    )


def validate_field_value(field_name: str, expected_type: str, value: Any) -> str | None:
    if expected_type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{field_name} must be numeric."
        return None
    if expected_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{field_name} must be an integer."
        return None
    if expected_type == "binary":
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            return f"{field_name} must be binary 0 or 1."
        return None
    return None


def validate_field_evidence(*, field_name: str, evidence: Any) -> str | None:
    if not isinstance(evidence, dict):
        return f"field_evidence.{field_name} must be an object."
    missing_keys = [key for key in EVIDENCE_REQUIRED_KEYS if key not in evidence]
    extra_keys = [key for key in evidence if key not in EVIDENCE_REQUIRED_KEYS]
    if missing_keys:
        return (
            f"field_evidence.{field_name} missing required keys: "
            + ", ".join(missing_keys)
        )
    if extra_keys:
        return (
            f"field_evidence.{field_name} has unsupported keys: "
            + ", ".join(extra_keys)
        )
    if not all(isinstance(evidence.get(key), str) for key in EVIDENCE_REQUIRED_KEYS):
        return f"field_evidence.{field_name} values must all be strings."
    if evidence["evidence_kind"] not in EVIDENCE_KIND_ENUM:
        return (
            f"field_evidence.{field_name}.evidence_kind must be one of "
            + ", ".join(sorted(EVIDENCE_KIND_ENUM))
        )
    if evidence["language"] not in EVIDENCE_LANGUAGE_ENUM:
        return (
            f"field_evidence.{field_name}.language must be one of "
            + ", ".join(sorted(EVIDENCE_LANGUAGE_ENUM))
        )
    return None


def is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def benchmark_requires_field_evidence(benchmark) -> bool:
    benchmark_name = str(getattr(benchmark, "benchmark_name", "") or "").lower()
    if "v3" in benchmark_name:
        return True
    for case in getattr(benchmark, "cases", ()) or ():
        expected = getattr(case, "expected_output", None)
        if isinstance(expected, dict) and "field_evidence" in expected:
            return True
    return False
