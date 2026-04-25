from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REFINEMENT_STATUS_APPLIED = "applied"
REFINEMENT_STATUS_CLARIFICATION_REQUIRED = "clarification_required"
REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK = "unsupported_feedback"
REFINEMENT_STATUS_LIMIT_REACHED = "limit_reached"
REFINEMENT_PARSER_TASK = "extract_constraint_feedback"


@dataclass(frozen=True)
class RefinementValidationResult:
    is_valid: bool
    parser_status: str | None
    normalized_output: dict[str, Any] | None
    normalized_delta: dict[str, Any] | None
    errors: tuple[str, ...]
    clarification_reasons: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "parser_status": self.parser_status,
            "normalized_output": None if self.normalized_output is None else dict(self.normalized_output),
            "normalized_delta": None if self.normalized_delta is None else dict(self.normalized_delta),
            "errors": list(self.errors),
            "clarification_reasons": list(self.clarification_reasons),
            "unsupported_reasons": list(self.unsupported_reasons),
        }


@dataclass(frozen=True)
class PendingRefinementClarification:
    originating_turn_id: str
    ambiguities: list[str]
    next_required_input: str
    parent_terminal_turn_id: str
    parent_refinement_revision_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "originating_turn_id": self.originating_turn_id,
            "ambiguities": list(self.ambiguities),
            "next_required_input": self.next_required_input,
            "parent_terminal_turn_id": self.parent_terminal_turn_id,
            "parent_refinement_revision_index": self.parent_refinement_revision_index,
        }

