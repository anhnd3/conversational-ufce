from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionNegotiationState:
    session_id: str
    dataset_id: str
    backend_id: str | None = None
    profile_facts: dict[str, Any] = field(default_factory=dict)
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    rejected_candidate_ids: list[str] = field(default_factory=list)
    rejected_features: list[str] = field(default_factory=list)
    accepted_tradeoffs: list[dict[str, Any]] = field(default_factory=list)
    last_reason_codes: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    turn_count: int = 0
    canonical_mirror_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "backend_id": self.backend_id,
            "profile_facts": dict(self.profile_facts),
            "hard_constraints": dict(self.hard_constraints),
            "soft_preferences": dict(self.soft_preferences),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "rejected_features": list(self.rejected_features),
            "accepted_tradeoffs": [dict(item) for item in self.accepted_tradeoffs],
            "last_reason_codes": list(self.last_reason_codes),
            "terminal_status": self.terminal_status,
            "turn_count": int(self.turn_count),
            "canonical_mirror_ok": bool(self.canonical_mirror_ok),
        }
