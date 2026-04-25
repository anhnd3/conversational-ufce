from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from llm.src.runtime.contracts import CanonicalCandidate, CanonicalRecourseRequest
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.types import RuntimeContext, RuntimeDebugTrace


@dataclass(frozen=True)
class BackendExecutionResult:
    backend_id: str
    candidates: list[CanonicalCandidate]
    reason_codes: list[str] = field(default_factory=list)
    failure_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason_codes": list(self.reason_codes),
            "failure_metadata": dict(self.failure_metadata),
        }


class RecourseBackend(Protocol):
    backend_id: str

    def generate(
        self,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
        *,
        deterministic_seed_value: int | None = None,
        debug_trace: RuntimeDebugTrace | None = None,
    ) -> BackendExecutionResult:
        ...
