from __future__ import annotations

from llm.src.parser.parser_quality import run_parser_quality
from llm.src.parser.response_normalizer import NormalizedParseResult
from llm.src.validation.schema_validator import ValidationResult


def parse_then_validate(
    *,
    message_text: str,
    benchmark,
    user_text: str | None = None,
    api_error: str | None = None,
) -> tuple[NormalizedParseResult, ValidationResult]:
    quality_result = run_parser_quality(
        message_text=message_text,
        benchmark_spec=benchmark,
        user_text=user_text,
        api_error=api_error,
    )
    return quality_result.normalized, quality_result.schema_validation
