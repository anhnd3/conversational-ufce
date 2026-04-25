from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec


REQUIRED_TOP_LEVEL_KEYS = ("task", "status", "cf_request", "missing_fields", "conflicts", "notes")
OPTIONAL_TOP_LEVEL_KEYS = ("constraint_spec",)
ALLOWED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS + OPTIONAL_TOP_LEVEL_KEYS


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

    missing_top_level = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in candidate]
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


def is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
