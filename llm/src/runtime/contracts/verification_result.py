from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REASON_CODE_VERSION = "reason_codes_v1"


@dataclass(frozen=True)
class VerificationResult:
    is_valid: bool
    reason_codes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    reason_code_version: str = REASON_CODE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "reason_codes": list(self.reason_codes),
            "evidence": dict(self.evidence),
            "candidate_id": self.candidate_id,
            "reason_code_version": self.reason_code_version,
        }
