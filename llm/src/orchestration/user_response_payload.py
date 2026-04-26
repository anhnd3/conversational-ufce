from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ResponseKind = Literal[
    "no_recourse_needed",
    "counterfactual_found",
    "runtime_reject_no_feasible_cf",
    "runtime_reject_constraints_blocked",
    "runtime_reject_invalid_counterfactual_blocked",
    "clarification_required",
    "clarification_limit_reached",
    "conflict",
    "unsupported_request",
    "parser_failure",
    "refinement_clarification",
]

ResponseTone = Literal["success", "info", "warning", "danger"]


@dataclass(frozen=True)
class ChangeItem:
    field_name: str
    display_name: str
    before: Any
    after: Any
    unit: str | None = None
    direction: Literal["increase", "decrease", "change", "enable", "disable", "unknown"] = "unknown"
    user_facing_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "display_name": self.display_name,
            "before": self.before,
            "after": self.after,
            "unit": self.unit,
            "direction": self.direction,
            "user_facing_text": self.user_facing_text,
        }


@dataclass(frozen=True)
class BlockedReason:
    code: str
    title: str
    detail: str
    fields: list[str] = field(default_factory=list)
    source: Literal["runtime", "invariant", "constraint", "parser", "clarification"] = "runtime"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "fields": list(self.fields),
            "source": self.source,
        }


@dataclass(frozen=True)
class ConstraintEffect:
    constraint_key: str
    title: str
    detail: str
    affected_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_key": self.constraint_key,
            "title": self.title,
            "detail": self.detail,
            "affected_fields": list(self.affected_fields),
        }


@dataclass(frozen=True)
class NextAction:
    action_type: Literal[
        "none",
        "provide_missing_fields",
        "relax_constraints",
        "revise_profile",
        "start_new_case",
        "refine_recommendation",
        "check_technical_details",
    ]
    label: str
    detail: str
    fields: list[str] = field(default_factory=list)
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "label": self.label,
            "detail": self.detail,
            "fields": list(self.fields),
            "primary": bool(self.primary),
        }


@dataclass(frozen=True)
class UserResponsePayload:
    response_kind: ResponseKind
    tone: ResponseTone
    headline: str
    short_summary: str
    changed_items: list[ChangeItem] = field(default_factory=list)
    blocked_reasons: list[BlockedReason] = field(default_factory=list)
    constraint_effects: list[ConstraintEffect] = field(default_factory=list)
    next_actions: list[NextAction] = field(default_factory=list)
    technical_facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_kind": self.response_kind,
            "tone": self.tone,
            "headline": self.headline,
            "short_summary": self.short_summary,
            "changed_items": [item.to_dict() for item in self.changed_items],
            "blocked_reasons": [item.to_dict() for item in self.blocked_reasons],
            "constraint_effects": [item.to_dict() for item in self.constraint_effects],
            "next_actions": [item.to_dict() for item in self.next_actions],
            "technical_facts": dict(self.technical_facts),
        }
