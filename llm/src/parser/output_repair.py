from __future__ import annotations

from llm.src.conversation.types import CanonicalValidationResult, ParserAdapterResult
from llm.src.parser.response_normalizer import NormalizedParseResult
from llm.src.validation.schema_validator import ValidationResult


def should_attempt_repair(*, raw_output: str, api_error: str | None, errors: list[str]) -> bool:
    if not errors:
        return False
    if api_error and not raw_output.strip():
        return False
    return bool(raw_output.strip())


def collect_repair_errors(
    *,
    parser_result: ParserAdapterResult,
    normalized: NormalizedParseResult,
    schema_validation: ValidationResult,
    canonical_validation: CanonicalValidationResult,
) -> list[str]:
    errors: list[str] = []
    if parser_result.api_error:
        errors.append(parser_result.api_error)
    if normalized.parse_error:
        errors.append(normalized.parse_error)
    errors.extend(schema_validation.errors)
    errors.extend(canonical_validation.errors)
    return dedupe_preserve_order(errors)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(str(item).split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered
