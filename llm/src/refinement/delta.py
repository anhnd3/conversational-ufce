from __future__ import annotations

from typing import Any

from llm.src.runtime.constraint_spec import (
    effective_blocked_fields,
    validate_and_normalize_constraint_spec,
)


REFINEMENT_DELTA_ALLOWED_KEYS = frozenset(
    {
        "add_blocked_fields",
        "remove_blocked_fields",
        "set_numeric_bounds",
        "clear_numeric_bounds",
        "set_max_changed_features",
        "clear_max_changed_features",
        "set_prefer_fewer_changes",
        "clear_prefer_fewer_changes",
    }
)
NUMERIC_BOUND_FIELD_ORDER = ("Income", "CCAvg", "Mortgage")
NUMERIC_BOUND_KEYS = ("min", "max")


def build_active_constraint_spec(
    constraint_spec: dict[str, Any] | None,
    *,
    feature_order: list[str],
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not isinstance(constraint_spec, dict):
        return {}
    active: dict[str, Any] = {}
    blocked = effective_blocked_fields(constraint_spec, feature_order=feature_order)
    if blocked:
        active["disallowed_changes"] = list(blocked)
    numeric_bounds = constraint_spec.get("numeric_bounds")
    derived_numeric_bound_fields = None
    if numeric_bound_fields is None and isinstance(numeric_bounds, dict):
        derived_numeric_bound_fields = list(numeric_bounds)
    if isinstance(numeric_bounds, dict) and numeric_bounds:
        active_numeric_bound_fields = _normalize_numeric_bound_fields(
            numeric_bound_fields if numeric_bound_fields is not None else derived_numeric_bound_fields
        )
        active["numeric_bounds"] = {
            field_name: {
                bound_key: float(bound_value)
                for bound_key, bound_value in bounds.items()
                if bound_key in NUMERIC_BOUND_KEYS and isinstance(bound_value, (int, float))
            }
            for field_name, bounds in numeric_bounds.items()
            if field_name in active_numeric_bound_fields and isinstance(bounds, dict)
        }
        active["numeric_bounds"] = {
            field_name: bounds for field_name, bounds in active["numeric_bounds"].items() if bounds
        }
        if not active["numeric_bounds"]:
            active.pop("numeric_bounds", None)
    max_changed_features = constraint_spec.get("max_changed_features")
    if isinstance(max_changed_features, int) and not isinstance(max_changed_features, bool):
        active["max_changed_features"] = int(max_changed_features)
    prefer_fewer_changes = constraint_spec.get("prefer_fewer_changes")
    if isinstance(prefer_fewer_changes, bool):
        active["prefer_fewer_changes"] = bool(prefer_fewer_changes)
    normalized, errors = validate_and_normalize_constraint_spec(
        active,
        feature_order=feature_order,
        numeric_bound_fields=_normalize_numeric_bound_fields(
            numeric_bound_fields if numeric_bound_fields is not None else derived_numeric_bound_fields
        ),
    )
    if errors:
        raise ValueError("; ".join(errors))
    return {} if normalized is None else dict(normalized)


def validate_and_normalize_refinement_delta(
    raw_delta: Any,
    *,
    feature_order: list[str],
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    if not isinstance(raw_delta, dict):
        return {}, ["constraint_feedback_delta must be an object."], []

    errors: list[str] = []
    clarification_reasons: list[str] = []
    normalized: dict[str, Any] = {}
    unknown_keys = sorted(key for key in raw_delta if key not in REFINEMENT_DELTA_ALLOWED_KEYS)
    if unknown_keys:
        errors.append("constraint_feedback_delta contains unknown keys: " + ", ".join(unknown_keys))

    if "add_blocked_fields" in raw_delta:
        value, value_errors = _normalize_feature_list(
            raw_delta.get("add_blocked_fields"),
            field_name="add_blocked_fields",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["add_blocked_fields"] = value

    if "remove_blocked_fields" in raw_delta:
        value, value_errors = _normalize_feature_list(
            raw_delta.get("remove_blocked_fields"),
            field_name="remove_blocked_fields",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["remove_blocked_fields"] = value

    if "set_numeric_bounds" in raw_delta:
        value, value_errors = _normalize_numeric_bounds(
            raw_delta.get("set_numeric_bounds"),
            field_name="set_numeric_bounds",
            numeric_bound_fields=_normalize_numeric_bound_fields(numeric_bound_fields),
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["set_numeric_bounds"] = value

    if "clear_numeric_bounds" in raw_delta:
        value, value_errors = _normalize_clear_numeric_bounds(
            raw_delta.get("clear_numeric_bounds"),
            numeric_bound_fields=_normalize_numeric_bound_fields(numeric_bound_fields),
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["clear_numeric_bounds"] = value

    if "set_max_changed_features" in raw_delta:
        value = raw_delta.get("set_max_changed_features")
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append("constraint_feedback_delta.set_max_changed_features must be an integer.")
        elif value not in {1, 2, 3}:
            errors.append("constraint_feedback_delta.set_max_changed_features must be one of 1, 2, 3.")
        else:
            normalized["set_max_changed_features"] = int(value)

    if "clear_max_changed_features" in raw_delta:
        value = raw_delta.get("clear_max_changed_features")
        if not isinstance(value, bool):
            errors.append("constraint_feedback_delta.clear_max_changed_features must be a boolean.")
        else:
            normalized["clear_max_changed_features"] = bool(value)

    if "set_prefer_fewer_changes" in raw_delta:
        value = raw_delta.get("set_prefer_fewer_changes")
        if not isinstance(value, bool):
            errors.append("constraint_feedback_delta.set_prefer_fewer_changes must be a boolean.")
        else:
            normalized["set_prefer_fewer_changes"] = bool(value)

    if "clear_prefer_fewer_changes" in raw_delta:
        value = raw_delta.get("clear_prefer_fewer_changes")
        if not isinstance(value, bool):
            errors.append("constraint_feedback_delta.clear_prefer_fewer_changes must be a boolean.")
        else:
            normalized["clear_prefer_fewer_changes"] = bool(value)

    clarification_reasons.extend(_detect_delta_conflicts(normalized))
    if errors:
        return {}, errors, clarification_reasons
    return normalized, [], clarification_reasons


def apply_refinement_delta_to_active_constraint_spec(
    active_constraint_spec: dict[str, Any] | None,
    delta: dict[str, Any],
    *,
    feature_order: list[str],
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    derived_numeric_bound_fields = None
    if numeric_bound_fields is None:
        current_bounds = {}
        if isinstance(active_constraint_spec, dict) and isinstance(active_constraint_spec.get("numeric_bounds"), dict):
            current_bounds.update(active_constraint_spec.get("numeric_bounds") or {})
        if isinstance(delta.get("set_numeric_bounds"), dict):
            current_bounds.update(delta.get("set_numeric_bounds") or {})
        if isinstance(delta.get("clear_numeric_bounds"), dict):
            current_bounds.update(delta.get("clear_numeric_bounds") or {})
        derived_numeric_bound_fields = list(current_bounds)
    active_numeric_bound_fields = _normalize_numeric_bound_fields(
        numeric_bound_fields if numeric_bound_fields is not None else derived_numeric_bound_fields
    )
    current = build_active_constraint_spec(
        active_constraint_spec,
        feature_order=feature_order,
        numeric_bound_fields=active_numeric_bound_fields,
    )
    blocked_fields = set(current.get("disallowed_changes", []))
    blocked_fields.update(delta.get("add_blocked_fields", []))
    blocked_fields.difference_update(delta.get("remove_blocked_fields", []))

    numeric_bounds = {
        field_name: dict(bounds)
        for field_name, bounds in (current.get("numeric_bounds") or {}).items()
        if isinstance(bounds, dict)
    }
    for field_name, bound_keys in (delta.get("clear_numeric_bounds") or {}).items():
        existing = numeric_bounds.get(field_name)
        if not isinstance(existing, dict):
            continue
        for bound_key in bound_keys:
            existing.pop(bound_key, None)
        if not existing:
            numeric_bounds.pop(field_name, None)
    for field_name, bounds in (delta.get("set_numeric_bounds") or {}).items():
        merged = dict(numeric_bounds.get(field_name) or {})
        merged.update(bounds)
        numeric_bounds[field_name] = merged

    if delta.get("clear_max_changed_features") is True:
        current.pop("max_changed_features", None)
    if "set_max_changed_features" in delta:
        current["max_changed_features"] = int(delta["set_max_changed_features"])

    if delta.get("clear_prefer_fewer_changes") is True:
        current.pop("prefer_fewer_changes", None)
    if "set_prefer_fewer_changes" in delta:
        current["prefer_fewer_changes"] = bool(delta["set_prefer_fewer_changes"])

    if blocked_fields:
        current["disallowed_changes"] = _ordered_feature_subset(list(blocked_fields), feature_order=feature_order)
    else:
        current.pop("disallowed_changes", None)
    if numeric_bounds:
        current["numeric_bounds"] = {
            field_name: numeric_bounds[field_name]
            for field_name in active_numeric_bound_fields
            if field_name in numeric_bounds
        }
    else:
        current.pop("numeric_bounds", None)

    normalized, errors = validate_and_normalize_constraint_spec(
        current,
        feature_order=feature_order,
        numeric_bound_fields=active_numeric_bound_fields,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return {} if normalized is None else dict(normalized)


def is_empty_refinement_delta(delta: dict[str, Any] | None) -> bool:
    if not isinstance(delta, dict):
        return True
    for value in delta.values():
        if isinstance(value, bool):
            if value:
                return False
            continue
        if isinstance(value, dict) and value:
            return False
        if isinstance(value, list) and value:
            return False
        if value is not None:
            return False
    return True


def _normalize_feature_list(
    raw_value: Any,
    *,
    field_name: str,
    feature_order: list[str],
) -> tuple[list[str], list[str]]:
    if not isinstance(raw_value, list) or any(not isinstance(item, str) for item in raw_value):
        return [], [f"constraint_feedback_delta.{field_name} must be an array of canonical field names."]
    invalid = sorted({item for item in raw_value if item not in feature_order})
    if invalid:
        return [], [f"constraint_feedback_delta.{field_name} contains unsupported fields: {', '.join(invalid)}"]
    return _ordered_feature_subset(raw_value, feature_order=feature_order), []


def _normalize_numeric_bounds(
    raw_value: Any,
    *,
    field_name: str,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    if not isinstance(raw_value, dict):
        return {}, [f"constraint_feedback_delta.{field_name} must be an object."]

    active_numeric_bound_fields = _normalize_numeric_bound_fields(numeric_bound_fields)
    errors: list[str] = []
    normalized: dict[str, dict[str, float]] = {}
    invalid_fields = sorted(key for key in raw_value if key not in active_numeric_bound_fields)
    if invalid_fields:
        errors.append(
            f"constraint_feedback_delta.{field_name} contains unsupported fields: " + ", ".join(invalid_fields)
        )

    for active_field in active_numeric_bound_fields:
        if active_field not in raw_value:
            continue
        raw_bounds = raw_value.get(active_field)
        if not isinstance(raw_bounds, dict):
            errors.append(f"constraint_feedback_delta.{field_name}.{active_field} must be an object.")
            continue
        unknown_bound_keys = sorted(key for key in raw_bounds if key not in NUMERIC_BOUND_KEYS)
        if unknown_bound_keys:
            errors.append(
                f"constraint_feedback_delta.{field_name}.{active_field} contains unknown keys: "
                + ", ".join(unknown_bound_keys)
            )
            continue
        if "min" not in raw_bounds and "max" not in raw_bounds:
            errors.append(f"constraint_feedback_delta.{field_name}.{active_field} must include min or max.")
            continue
        normalized_bounds: dict[str, float] = {}
        for bound_key in NUMERIC_BOUND_KEYS:
            if bound_key not in raw_bounds:
                continue
            bound_value = raw_bounds[bound_key]
            if isinstance(bound_value, bool) or not isinstance(bound_value, (int, float)):
                errors.append(
                    f"constraint_feedback_delta.{field_name}.{active_field}.{bound_key} must be numeric."
                )
                continue
            normalized_bounds[bound_key] = float(bound_value)
        if "min" in normalized_bounds and "max" in normalized_bounds:
            if normalized_bounds["min"] > normalized_bounds["max"]:
                errors.append(
                    f"constraint_feedback_delta.{field_name}.{active_field} must satisfy min <= max."
                )
                continue
        if normalized_bounds:
            normalized[active_field] = normalized_bounds

    return normalized, errors


def _normalize_clear_numeric_bounds(
    raw_value: Any,
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    if not isinstance(raw_value, dict):
        return {}, ["constraint_feedback_delta.clear_numeric_bounds must be an object."]
    active_numeric_bound_fields = _normalize_numeric_bound_fields(numeric_bound_fields)
    errors: list[str] = []
    normalized: dict[str, list[str]] = {}
    invalid_fields = sorted(key for key in raw_value if key not in active_numeric_bound_fields)
    if invalid_fields:
        errors.append(
            "constraint_feedback_delta.clear_numeric_bounds contains unsupported fields: " + ", ".join(invalid_fields)
        )
    for field_name in active_numeric_bound_fields:
        if field_name not in raw_value:
            continue
        raw_keys = raw_value.get(field_name)
        if not isinstance(raw_keys, list) or any(not isinstance(item, str) for item in raw_keys):
            errors.append(f"constraint_feedback_delta.clear_numeric_bounds.{field_name} must be an array.")
            continue
        invalid_keys = sorted({item for item in raw_keys if item not in NUMERIC_BOUND_KEYS})
        if invalid_keys:
            errors.append(
                f"constraint_feedback_delta.clear_numeric_bounds.{field_name} contains unsupported keys: "
                + ", ".join(invalid_keys)
            )
            continue
        normalized[field_name] = [bound_key for bound_key in NUMERIC_BOUND_KEYS if bound_key in set(raw_keys)]
    return normalized, errors


def _detect_delta_conflicts(normalized: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    add_blocked = set(normalized.get("add_blocked_fields", []))
    remove_blocked = set(normalized.get("remove_blocked_fields", []))
    overlap = sorted(add_blocked.intersection(remove_blocked))
    if overlap:
        reasons.append(
            "The same blocked fields were both added and removed in one refinement turn: " + ", ".join(overlap)
        )

    set_numeric_bounds = normalized.get("set_numeric_bounds") or {}
    clear_numeric_bounds = normalized.get("clear_numeric_bounds") or {}
    for field_name, bounds in set_numeric_bounds.items():
        cleared = set(clear_numeric_bounds.get(field_name, []))
        overlap_keys = [bound_key for bound_key in NUMERIC_BOUND_KEYS if bound_key in bounds and bound_key in cleared]
        if overlap_keys:
            reasons.append(
                f"The same numeric bound was both set and cleared for {field_name}: {', '.join(overlap_keys)}"
            )

    if "set_max_changed_features" in normalized and normalized.get("clear_max_changed_features") is True:
        reasons.append("max_changed_features was both set and cleared in one refinement turn.")
    if "set_prefer_fewer_changes" in normalized and normalized.get("clear_prefer_fewer_changes") is True:
        reasons.append("prefer_fewer_changes was both set and cleared in one refinement turn.")
    return reasons


def _ordered_feature_subset(fields: list[str], *, feature_order: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    field_set = {field for field in fields if isinstance(field, str)}
    for field in feature_order:
        if field in field_set and field not in seen:
            seen.add(field)
            ordered.append(field)
    return ordered


def _normalize_numeric_bound_fields(
    numeric_bound_fields: list[str] | tuple[str, ...] | None,
) -> list[str]:
    active = list(NUMERIC_BOUND_FIELD_ORDER) if numeric_bound_fields is None else list(numeric_bound_fields)
    seen: set[str] = set()
    normalized: list[str] = []
    for field_name in active:
        clean = str(field_name).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized
