from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm.src.runtime.contracts.canonical_profile import CanonicalProfile


@dataclass(frozen=True)
class CanonicalRecourseRequest:
    dataset_id: str
    desired_outcome: int | str
    profile: CanonicalProfile
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    forbidden_features: list[str] = field(default_factory=list)
    max_changes: int | None = None
    explanation_level: str = "standard"
    session_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "desired_outcome": self.desired_outcome,
            "profile": self.profile.to_dict(),
            "hard_constraints": dict(self.hard_constraints),
            "soft_preferences": dict(self.soft_preferences),
            "forbidden_features": list(self.forbidden_features),
            "max_changes": self.max_changes,
            "explanation_level": self.explanation_level,
            "session_context": dict(self.session_context),
        }
