from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticValidationResult:
    is_valid: bool
    reasons: tuple[str, ...]


def validate_semantics(candidate: dict | None, benchmark) -> SemanticValidationResult:
    del candidate
    del benchmark
    return SemanticValidationResult(is_valid=True, reasons=())
