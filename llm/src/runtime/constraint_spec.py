from __future__ import annotations

from typing import Any

from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult


CONSTRAINT_SPEC_ALLOWED_KEYS = frozenset(
    {
        "immutable",
        "disallowed_changes",
        "numeric_bounds",
        "max_changed_features",
        "prefer_fewer_changes",
    }
)
NUMERIC_BOUND_FIELD_ORDER = ("Income", "CCAvg", "Mortgage")


def validate_and_normalize_constraint_spec(
    raw_spec: Any,
    *,
    feature_order: list[str],
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any] | Any, list[str]]:
    if raw_spec is None:
        return None, []
    if not isinstance(raw_spec, dict):
        return raw_spec, ["constraint_spec must be an object."]

    errors: list[str] = []
    normalized: dict[str, Any] = {}
    unknown_keys = sorted(key for key in raw_spec if key not in CONSTRAINT_SPEC_ALLOWED_KEYS)
    if unknown_keys:
        errors.append("constraint_spec contains unknown keys: " + ", ".join(unknown_keys))

    if "immutable" in raw_spec:
        value, value_errors = _normalize_feature_list(
            raw_spec.get("immutable"),
            field_name="immutable",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["immutable"] = value

    if "disallowed_changes" in raw_spec:
        value, value_errors = _normalize_feature_list(
            raw_spec.get("disallowed_changes"),
            field_name="disallowed_changes",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["disallowed_changes"] = value

    if "numeric_bounds" in raw_spec:
        value, value_errors = _normalize_numeric_bounds(
            raw_spec.get("numeric_bounds"),
            numeric_bound_fields=_normalize_numeric_bound_fields(numeric_bound_fields),
        )
        errors.extend(value_errors)
        if not value_errors:
            normalized["numeric_bounds"] = value

    if "max_changed_features" in raw_spec:
        value = raw_spec.get("max_changed_features")
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append("constraint_spec.max_changed_features must be an integer.")
        elif value not in {1, 2, 3}:
            errors.append("constraint_spec.max_changed_features must be one of 1, 2, 3.")
        else:
            normalized["max_changed_features"] = int(value)

    if "prefer_fewer_changes" in raw_spec:
        value = raw_spec.get("prefer_fewer_changes")
        if not isinstance(value, bool):
            errors.append("constraint_spec.prefer_fewer_changes must be a boolean.")
        else:
            normalized["prefer_fewer_changes"] = bool(value)

    if errors:
        return raw_spec, errors
    return normalized, []


def effective_blocked_fields(
    constraint_spec: dict[str, Any] | None,
    *,
    feature_order: list[str],
) -> list[str]:
    if not isinstance(constraint_spec, dict):
        return []
    blocked = set()
    for key in ("immutable", "disallowed_changes"):
        value = constraint_spec.get(key)
        if isinstance(value, list):
            blocked.update(item for item in value if isinstance(item, str))
    return _ordered_feature_subset(list(blocked), feature_order=feature_order)


def apply_constraint_spec_to_candidates(
    *,
    result: CounterfactualResult,
    constraint_spec: dict[str, Any] | None,
    feature_order: list[str],
    sort_candidates,
    request_constraints_blocked_code: str,
) -> tuple[CounterfactualResult, dict[str, Any] | None]:
    if not isinstance(constraint_spec, dict) or not result.feasible or not result.candidates:
        return result, None

    blocked_fields = set(effective_blocked_fields(constraint_spec, feature_order=feature_order))
    numeric_bounds = constraint_spec.get("numeric_bounds") if isinstance(constraint_spec.get("numeric_bounds"), dict) else {}
    max_changed_features = constraint_spec.get("max_changed_features")
    prefer_fewer_changes = bool(constraint_spec.get("prefer_fewer_changes"))

    kept_candidates: list[CounterfactualCandidate] = []
    blocked_reason_counts: dict[str, int] = {}

    for candidate in result.candidates:
        reasons = _candidate_block_reasons(
            candidate=candidate,
            blocked_fields=blocked_fields,
            numeric_bounds=numeric_bounds,
            max_changed_features=max_changed_features,
        )
        if reasons:
            for reason in reasons:
                blocked_reason_counts[reason] = blocked_reason_counts.get(reason, 0) + 1
            continue
        kept_candidates.append(candidate)

    debug_summary = {
        "applied": True,
        "constraint_spec": dict(constraint_spec),
        "generated_candidate_count": len(result.candidates),
        "kept_candidate_count": len(kept_candidates),
        "blocked_candidate_count": len(result.candidates) - len(kept_candidates),
        "blocked_reason_counts": blocked_reason_counts,
        "prefer_fewer_changes": prefer_fewer_changes,
    }

    if not kept_candidates:
        return (
            CounterfactualResult(
                feasible=False,
                candidates=[],
                reason_codes=[request_constraints_blocked_code],
            ),
            debug_summary,
        )

    ordered = sort_candidates(
        candidates=kept_candidates,
        feature_order=feature_order,
        prefer_fewer_changes=prefer_fewer_changes,
    )
    return (
        CounterfactualResult(
            feasible=True,
            candidates=ordered,
            reason_codes=[],
        ),
        debug_summary,
    )


def _normalize_feature_list(
    raw_value: Any,
    *,
    field_name: str,
    feature_order: list[str],
) -> tuple[list[str], list[str]]:
    if not isinstance(raw_value, list) or any(not isinstance(item, str) for item in raw_value):
        return [], [f"constraint_spec.{field_name} must be an array of canonical field names."]
    invalid = sorted({item for item in raw_value if item not in feature_order})
    if invalid:
        return [], [f"constraint_spec.{field_name} contains unsupported fields: {', '.join(invalid)}"]
    return _ordered_feature_subset(raw_value, feature_order=feature_order), []


def _normalize_numeric_bounds_with_fields(
    raw_value: Any,
    *,
    numeric_bound_fields: list[str],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    if not isinstance(raw_value, dict):
        return {}, ["constraint_spec.numeric_bounds must be an object."]

    errors: list[str] = []
    normalized: dict[str, dict[str, float]] = {}
    invalid_fields = sorted(key for key in raw_value if key not in numeric_bound_fields)
    if invalid_fields:
        errors.append(
            "constraint_spec.numeric_bounds contains unsupported fields: " + ", ".join(invalid_fields)
        )

    for field_name in numeric_bound_fields:
        if field_name not in raw_value:
            continue
        raw_bounds = raw_value.get(field_name)
        if not isinstance(raw_bounds, dict):
            errors.append(f"constraint_spec.numeric_bounds.{field_name} must be an object.")
            continue
        unknown_bound_keys = sorted(key for key in raw_bounds if key not in {"min", "max"})
        if unknown_bound_keys:
            errors.append(
                f"constraint_spec.numeric_bounds.{field_name} contains unknown keys: {', '.join(unknown_bound_keys)}"
            )
            continue
        if "min" not in raw_bounds and "max" not in raw_bounds:
            errors.append(f"constraint_spec.numeric_bounds.{field_name} must include min or max.")
            continue

        normalized_bounds: dict[str, float] = {}
        for bound_key in ("min", "max"):
            if bound_key not in raw_bounds:
                continue
            bound_value = raw_bounds[bound_key]
            if isinstance(bound_value, bool) or not isinstance(bound_value, (int, float)):
                errors.append(
                    f"constraint_spec.numeric_bounds.{field_name}.{bound_key} must be numeric."
                )
                continue
            normalized_bounds[bound_key] = float(bound_value)
        if "min" in normalized_bounds and "max" in normalized_bounds:
            if normalized_bounds["min"] > normalized_bounds["max"]:
                errors.append(
                    f"constraint_spec.numeric_bounds.{field_name} must satisfy min <= max."
                )
                continue
        if normalized_bounds:
            normalized[field_name] = normalized_bounds

    return normalized, errors


def _normalize_numeric_bounds(
    raw_value: Any,
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    return _normalize_numeric_bounds_with_fields(
        raw_value,
        numeric_bound_fields=_normalize_numeric_bound_fields(numeric_bound_fields),
    )


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


def _candidate_block_reasons(
    *,
    candidate: CounterfactualCandidate,
    blocked_fields: set[str],
    numeric_bounds: dict[str, dict[str, float]],
    max_changed_features: Any,
) -> list[str]:
    reasons: list[str] = []
    changed = list(candidate.changed_features)
    if blocked_fields and any(field in blocked_fields for field in changed):
        reasons.append("blocked_change_field")
    if isinstance(max_changed_features, int) and len(changed) > max_changed_features:
        reasons.append("max_changed_features_exceeded")
    for field_name, bounds in numeric_bounds.items():
        value = candidate.profile.get(field_name)
        if value is None:
            continue
        if "min" in bounds and float(value) < float(bounds["min"]):
            reasons.append(f"numeric_bounds:{field_name}")
        elif "max" in bounds and float(value) > float(bounds["max"]):
            reasons.append(f"numeric_bounds:{field_name}")
    return reasons


def _ordered_feature_subset(fields: list[str], *, feature_order: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    field_set = {field for field in fields if isinstance(field, str)}
    for field in feature_order:
        if field in field_set and field not in seen:
            seen.add(field)
            ordered.append(field)
    for field in fields:
        if isinstance(field, str) and field not in seen:
            seen.add(field)
            ordered.append(field)
    return ordered
