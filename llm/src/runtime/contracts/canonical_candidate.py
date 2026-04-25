from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalCandidate:
    backend_id: str
    candidate_id: str
    changed_features: list[str]
    original_values: dict[str, Any]
    new_values: dict[str, Any]
    delta_summary: list[dict[str, Any]]
    predicted_outcome: int | str | None = None
    raw_backend_score: float | None = None
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "candidate_id": self.candidate_id,
            "changed_features": list(self.changed_features),
            "original_values": dict(self.original_values),
            "new_values": dict(self.new_values),
            "delta_summary": [dict(item) for item in self.delta_summary],
            "predicted_outcome": self.predicted_outcome,
            "raw_backend_score": self.raw_backend_score,
            "backend_metadata": dict(self.backend_metadata),
        }
