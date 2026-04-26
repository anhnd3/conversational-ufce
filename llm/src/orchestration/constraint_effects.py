from __future__ import annotations

from typing import Any

from llm.src.orchestration.user_response_payload import ConstraintEffect


def describe_active_constraint_effects(
    *,
    active_constraint_spec: dict[str, Any] | None,
    policy=None,
) -> list[ConstraintEffect]:
    effects: list[ConstraintEffect] = []
    spec = dict(active_constraint_spec or {})

    disallowed_changes = spec.get("disallowed_changes")
    if isinstance(disallowed_changes, list):
        for field_name in disallowed_changes:
            if not isinstance(field_name, str) or not field_name.strip():
                continue
            effects.append(
                ConstraintEffect(
                    constraint_key=f"disallowed_changes:{field_name}",
                    title=f"{field_name} is fixed",
                    detail=f"You asked to keep {field_name} unchanged.",
                    affected_fields=[field_name],
                )
            )

    numeric_bounds = spec.get("numeric_bounds")
    if isinstance(numeric_bounds, dict):
        for field_name, bounds in numeric_bounds.items():
            if not isinstance(field_name, str) or not isinstance(bounds, dict):
                continue
            detail = _render_bounds_detail(field_name=field_name, bounds=bounds)
            if detail:
                effects.append(
                    ConstraintEffect(
                        constraint_key=f"numeric_bounds:{field_name}",
                        title=f"{field_name} bound",
                        detail=detail,
                        affected_fields=[field_name],
                    )
                )

    max_changed_features = spec.get("max_changed_features")
    if isinstance(max_changed_features, int) and not isinstance(max_changed_features, bool):
        plural = "" if int(max_changed_features) == 1 else "s"
        effects.append(
            ConstraintEffect(
                constraint_key="max_changed_features",
                title="Change limit",
                detail=f"At most {int(max_changed_features)} feature{plural} may change.",
                affected_fields=[],
            )
        )

    prefer_fewer_changes = spec.get("prefer_fewer_changes")
    if isinstance(prefer_fewer_changes, bool) and prefer_fewer_changes:
        effects.append(
            ConstraintEffect(
                constraint_key="prefer_fewer_changes",
                title="Fewer changes preferred",
                detail="The system prioritized options with fewer changed fields.",
                affected_fields=[],
            )
        )

    policy_effect = describe_allowed_change_policy(policy)
    if policy_effect is not None:
        effects.append(policy_effect)

    return effects


def describe_allowed_change_policy(policy) -> ConstraintEffect | None:
    if policy is None:
        return None
    f2change = list(getattr(policy, "f2change", []) or [])
    normalized = [field for field in f2change if isinstance(field, str) and field.strip()]
    if not normalized:
        return None
    dataset_name = str(getattr(policy, "dataset_name", "current dataset")).strip() or "current dataset"
    return ConstraintEffect(
        constraint_key="policy.f2change",
        title="Dataset change policy",
        detail=(
            f"The current dataset policy ({dataset_name}) only allows these fields to change: "
            + ", ".join(normalized)
            + "."
        ),
        affected_fields=list(normalized),
    )


def _render_bounds_detail(*, field_name: str, bounds: dict[str, Any]) -> str | None:
    has_min = "min" in bounds
    has_max = "max" in bounds
    min_value = bounds.get("min")
    max_value = bounds.get("max")
    if has_min and has_max:
        return f"{field_name} must stay between {min_value} and {max_value}."
    if has_min:
        return f"{field_name} must stay at or above {min_value}."
    if has_max:
        return f"{field_name} must stay at or below {max_value}."
    return None
