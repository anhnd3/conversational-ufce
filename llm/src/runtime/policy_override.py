from __future__ import annotations

from dataclasses import replace
from typing import Any

from llm.src.runtime.types import DatasetPolicy


POLICY_OVERRIDE_ALLOWED_KEYS = frozenset({"f2change", "remove_from_f2change", "uf", "step"})
POLICY_OVERRIDE_DERIVATION_VERSION = "constraint_spec_to_policy_override_v1"


def validate_and_apply_policy_override(
    base_policy: DatasetPolicy,
    raw_override: Any,
    *,
    feature_order: list[str],
) -> tuple[DatasetPolicy, dict[str, Any] | None, list[str]]:
    if raw_override is None:
        return base_policy, None, []
    if not isinstance(raw_override, dict):
        return base_policy, raw_override, ["policy_override must be an object."]

    errors: list[str] = []
    normalized: dict[str, Any] = {}
    unknown_keys = sorted(key for key in raw_override if key not in POLICY_OVERRIDE_ALLOWED_KEYS)
    if unknown_keys:
        errors.append("policy_override contains unknown keys: " + ", ".join(unknown_keys))

    base_f2change = list(base_policy.f2change)
    effective_f2change = list(base_f2change)

    if "f2change" in raw_override:
        value, value_errors = _normalize_feature_list(
            raw_override.get("f2change"),
            field_name="f2change",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            effective_f2change = value
            normalized["f2change"] = list(value)

    if "remove_from_f2change" in raw_override:
        value, value_errors = _normalize_feature_list(
            raw_override.get("remove_from_f2change"),
            field_name="remove_from_f2change",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            remove = set(value)
            effective_f2change = [feature for feature in effective_f2change if feature not in remove]
            normalized["remove_from_f2change"] = list(value)

    if not effective_f2change:
        errors.append("policy_override results in empty effective f2change.")

    uf = dict(base_policy.uf)
    if "uf" in raw_override:
        value, value_errors = _normalize_numeric_map(
            raw_override.get("uf"),
            field_name="uf",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        if not value_errors:
            uf.update(value)
            normalized["uf"] = dict(value)

    step = dict(base_policy.step)
    if "step" in raw_override:
        value, value_errors = _normalize_numeric_map(
            raw_override.get("step"),
            field_name="step",
            feature_order=feature_order,
        )
        errors.extend(value_errors)
        for feature_name, step_value in value.items():
            if base_policy.feature_type_map.get(feature_name) == "binary" and float(step_value) != 1.0:
                errors.append(f"policy_override.step.{feature_name} must be 1 for binary fields.")
        if not value_errors:
            step.update(value)
            normalized["step"] = dict(value)

    if errors:
        return base_policy, raw_override, errors

    effective_policy = replace(
        base_policy,
        f2change=list(effective_f2change),
        uf=dict(uf),
        step=dict(step),
        policy_version=f"{base_policy.policy_version}+override",
    )
    return effective_policy, normalized, []


def apply_constraint_policy_override_to_runtime_request(
    runtime_request: dict[str, Any],
    *,
    base_policy: DatasetPolicy,
    feature_order: list[str],
) -> dict[str, Any]:
    """Attach a generation-time policy override inferred from request constraints.

    The original request-level constraint_spec is preserved as the public safety
    filter. This helper only adds the subset that can safely be represented by
    UFCE's generation knobs: f2change, uf, and step.
    """
    if not isinstance(runtime_request, dict):
        return runtime_request
    existing_override = runtime_request.get("policy_override")
    if existing_override is not None and not isinstance(existing_override, dict):
        return dict(runtime_request)

    derived_override = derive_policy_override_from_constraint_spec(
        base_policy=base_policy,
        constraint_spec=runtime_request.get("constraint_spec"),
        profile=runtime_request.get("profile"),
        feature_order=feature_order,
    )
    merged_override = merge_policy_overrides(existing_override, derived_override)
    if merged_override is None:
        return dict(runtime_request)

    updated = dict(runtime_request)
    updated["policy_override"] = merged_override
    return updated


def derive_policy_override_from_constraint_spec(
    *,
    base_policy: DatasetPolicy,
    constraint_spec: Any,
    profile: Any = None,
    feature_order: list[str],
) -> dict[str, Any] | None:
    if not isinstance(constraint_spec, dict):
        return None

    override: dict[str, Any] = {}
    base_f2change = _ordered_feature_subset(list(base_policy.f2change), feature_order=feature_order)
    effective_f2change = list(base_f2change)

    allowed_changed_features = constraint_spec.get("allowed_changed_features")
    if isinstance(allowed_changed_features, list) and all(isinstance(item, str) for item in allowed_changed_features):
        allowed = set(_ordered_feature_subset(allowed_changed_features, feature_order=feature_order))
        effective_f2change = [field_name for field_name in effective_f2change if field_name in allowed]

    blocked_fields = set(_constraint_blocked_fields(constraint_spec, feature_order=feature_order))
    if blocked_fields:
        effective_f2change = [field_name for field_name in effective_f2change if field_name not in blocked_fields]

    # Empty f2change is invalid for UFCE generation. Keep the post-generation
    # constraint_spec in force and let the runtime report REQUEST_CONSTRAINTS_BLOCKED.
    if effective_f2change and effective_f2change != base_f2change:
        override["f2change"] = list(effective_f2change)

    uf_update = _derive_uf_override_from_numeric_bounds(
        base_policy=base_policy,
        constraint_spec=constraint_spec,
        profile=profile,
        feature_order=feature_order,
    )
    if uf_update:
        override["uf"] = uf_update
        step_update = _derive_step_override_for_uf_bounds(
            base_policy=base_policy,
            uf_update=uf_update,
        )
        if step_update:
            override["step"] = step_update

    return override or None


def merge_policy_overrides(*overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for override in overrides:
        if not isinstance(override, dict):
            continue
        for list_key in ("f2change", "remove_from_f2change"):
            value = override.get(list_key)
            if isinstance(value, list):
                merged[list_key] = list(value)
        for map_key in ("uf", "step"):
            value = override.get(map_key)
            if not isinstance(value, dict):
                continue
            current = dict(merged.get(map_key) or {})
            current.update(value)
            if current:
                merged[map_key] = current
    return merged or None


def _normalize_feature_list(
    raw_value: Any,
    *,
    field_name: str,
    feature_order: list[str],
) -> tuple[list[str], list[str]]:
    if not isinstance(raw_value, list) or any(not isinstance(item, str) for item in raw_value):
        return [], [f"policy_override.{field_name} must be an array of canonical field names."]
    invalid = sorted({item for item in raw_value if item not in feature_order})
    if invalid:
        return [], [f"policy_override.{field_name} contains unsupported fields: {', '.join(invalid)}"]
    return _ordered_feature_subset(raw_value, feature_order=feature_order), []


def _normalize_numeric_map(
    raw_value: Any,
    *,
    field_name: str,
    feature_order: list[str],
) -> tuple[dict[str, float], list[str]]:
    if not isinstance(raw_value, dict):
        return {}, [f"policy_override.{field_name} must be an object."]
    errors: list[str] = []
    normalized: dict[str, float] = {}
    invalid = sorted(key for key in raw_value if key not in feature_order)
    if invalid:
        errors.append(f"policy_override.{field_name} contains unsupported fields: {', '.join(invalid)}")
    for feature_name in feature_order:
        if feature_name not in raw_value:
            continue
        value = raw_value.get(feature_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"policy_override.{field_name}.{feature_name} must be numeric.")
            continue
        numeric_value = float(value)
        if numeric_value <= 0:
            errors.append(f"policy_override.{field_name}.{feature_name} must be > 0.")
            continue
        normalized[feature_name] = numeric_value
    return normalized, errors


def _ordered_feature_subset(values: list[str], *, feature_order: list[str]) -> list[str]:
    requested = set(values)
    return [feature for feature in feature_order if feature in requested]


def _constraint_blocked_fields(
    constraint_spec: dict[str, Any],
    *,
    feature_order: list[str],
) -> list[str]:
    blocked: list[str] = []
    for key in ("immutable", "disallowed_changes"):
        value = constraint_spec.get(key)
        if isinstance(value, list):
            blocked.extend(item for item in value if isinstance(item, str))
    return _ordered_feature_subset(blocked, feature_order=feature_order)


def _derive_uf_override_from_numeric_bounds(
    *,
    base_policy: DatasetPolicy,
    constraint_spec: dict[str, Any],
    profile: Any,
    feature_order: list[str],
) -> dict[str, float]:
    profile_values = profile if isinstance(profile, dict) else {}
    uf_update: dict[str, float] = {}
    numeric_bounds = constraint_spec.get("numeric_bounds")
    if isinstance(numeric_bounds, dict):
        for field_name in feature_order:
            bounds = numeric_bounds.get(field_name)
            if not isinstance(bounds, dict) or "max" not in bounds:
                continue
            bound_max = bounds.get("max")
            factual_value = profile_values.get(field_name)
            if not _is_number(bound_max) or not _is_number(factual_value):
                continue
            allowed_increase = float(bound_max) - float(factual_value)
            _merge_uf_update(
                uf_update,
                base_policy=base_policy,
                field_name=field_name,
                candidate_value=allowed_increase,
            )

    numeric_bounds_delta = constraint_spec.get("numeric_bounds_delta")
    if isinstance(numeric_bounds_delta, dict):
        for field_name in feature_order:
            bounds = numeric_bounds_delta.get(field_name)
            if not isinstance(bounds, dict):
                continue
            max_increase = bounds.get("max_increase")
            if not _is_number(max_increase):
                continue
            _merge_uf_update(
                uf_update,
                base_policy=base_policy,
                field_name=field_name,
                candidate_value=float(max_increase),
            )
    return uf_update


def _derive_step_override_for_uf_bounds(
    *,
    base_policy: DatasetPolicy,
    uf_update: dict[str, float],
) -> dict[str, float]:
    step_update: dict[str, float] = {}
    for field_name, uf_value in uf_update.items():
        base_step = base_policy.step.get(field_name)
        if not _is_number(base_step):
            continue
        if str(base_policy.feature_type_map.get(field_name)) == "binary":
            continue
        numeric_step = float(base_step)
        numeric_uf = float(uf_value)
        if 0 < numeric_uf < numeric_step:
            step_update[field_name] = numeric_uf
    return step_update


def _merge_uf_update(
    uf_update: dict[str, float],
    *,
    base_policy: DatasetPolicy,
    field_name: str,
    candidate_value: float,
) -> None:
    if candidate_value <= 0:
        return
    base_value = base_policy.uf.get(field_name)
    if not _is_number(base_value):
        return
    bounded_value = min(float(base_value), float(candidate_value))
    if bounded_value >= float(base_value):
        return
    existing = uf_update.get(field_name)
    uf_update[field_name] = bounded_value if existing is None else min(float(existing), bounded_value)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
