from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from llm.src.conversation.canonical_session_state import split_constraint_buckets
from llm.src.parser.response_normalizer import NormalizedParseResult, normalize_and_parse
from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from llm.src.runtime.datasets.bank.metadata import BANK_FEATURE_TYPES, BANK_REQUIRED_FIELD_ORDER
from llm.src.validation.schema_validator import ValidationResult, validate_prediction


BANK_PROFILE_PARSE_V2_TASK = "extract_bank_profile_v2"
BANK_PROFILE_PARSE_V2_EXTRACTION_SCHEMA_NAME = "bank_profile_parse_v2_evidence"
BANK_PROFILE_PARSE_V2_SCHEMA_NAME = "bank_profile_parse_v2"
BANK_PROFILE_PARSE_V2_VERIFIER_SCHEMA_NAME = "bank_profile_parse_v2_verifier"
BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE = "evidence_extraction"
BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE = "value_normalization"
BANK_PROFILE_PARSE_V2_VERIFIER_STAGE = "semantic_verification"
BANK_PROFILE_PARSE_V2_PROVENANCE = "llm_semantic_parse_v2"
MAX_EXTRACTION_RETRIES = 2
MAX_NORMALIZATION_RETRIES = 2
MAX_PARSE_RETRIES = 2
MAX_VERIFIER_CORRECTIONS = 1

_BANK_BOOLEAN_FIELDS = {
    field_name
    for field_name, field_type in BANK_FEATURE_TYPES.items()
    if field_type == "binary"
}
_BANK_INTEGER_DOMAINS = {
    "Family": {1, 2, 3, 4},
    "Education": {1, 2, 3},
}
_BANK_INTEGER_VALUED_FIELDS = {"Income", "Mortgage"}
_VALID_STATUS_VALUES = {"complete", "partial", "needs_clarification", "conflict"}


@dataclass(frozen=True)
class BankProfileParseV2QualityResult:
    normalized: NormalizedParseResult
    schema_validation: ValidationResult
    field_provenance: dict[str, str]
    metadata: dict[str, Any]
    v2_payload: dict[str, Any] | None


@dataclass(frozen=True)
class BankProfileParseV2DeterministicResult:
    value: int | float
    reason: str


def build_bank_profile_parse_v2_extraction_schema(benchmark) -> dict[str, Any]:
    field_properties = {
        field.name: _evidence_field_schema()
        for field in getattr(benchmark, "target_cf_fields", ())
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "stage", "status", "fields", "missing_fields", "conflicts", "notes"],
        "properties": {
            "task": {"type": "string", "const": BANK_PROFILE_PARSE_V2_TASK},
            "stage": {"type": "string", "const": BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE},
            "status": {"type": "string", "enum": sorted(_VALID_STATUS_VALUES)},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "properties": field_properties,
            },
            "missing_fields": {"type": "array", "items": {"type": "string"}},
            "conflicts": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_bank_profile_parse_v2_schema(benchmark) -> dict[str, Any]:
    field_properties = {
        field.name: _normalized_field_schema(str(field.name))
        for field in getattr(benchmark, "target_cf_fields", ())
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "stage", "status", "fields", "missing_fields", "conflicts", "notes"],
        "properties": {
            "task": {"type": "string", "const": BANK_PROFILE_PARSE_V2_TASK},
            "stage": {"type": "string", "const": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE},
            "status": {"type": "string", "enum": sorted(_VALID_STATUS_VALUES)},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "properties": field_properties,
            },
            "constraint_spec": _constraint_spec_schema(),
            "missing_fields": {"type": "array", "items": {"type": "string"}},
            "conflicts": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_bank_profile_parse_v2_verifier_schema(benchmark) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["stage", "verdict", "candidate", "errors", "notes"],
        "properties": {
            "stage": {"type": "string", "const": BANK_PROFILE_PARSE_V2_VERIFIER_STAGE},
            "verdict": {"type": "string", "enum": ["PASS", "CORRECTED", "FAIL"]},
            "candidate": build_bank_profile_parse_v2_schema(benchmark),
            "errors": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_bank_profile_parse_v2_extraction_prompt(
    benchmark,
    *,
    user_text: str,
    dataset_label: str = "bank profile",
    retry_errors: list[str] | tuple[str, ...] | None = None,
    previous_output: str | None = None,
    extraction_hints: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {"input": user_text}
    if retry_errors:
        payload["previous_output"] = previous_output or ""
        payload["validation_errors"] = list(retry_errors)
    if extraction_hints:
        payload["evidence_hints"] = extraction_hints
    instructions = {
        "contract": "BankProfileParseV2EvidenceExtraction",
        "field_order": _feature_order_for_benchmark(benchmark),
        "field_dictionary": _feature_dictionary(benchmark),
        "schema_reference": {
            "task": BANK_PROFILE_PARSE_V2_TASK,
            "stage": BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
            "status": sorted(_VALID_STATUS_VALUES),
            "fields": {
                "FieldName": {
                    "evidence_quote": "exact substring copied from the input",
                    "confidence": "number from 0.0 to 1.0",
                    "reason": "why this exact quote is the supporting evidence",
                }
            },
            "missing_fields": ["canonical field name"],
            "conflicts": ["brief string"],
            "notes": ["brief string"],
        },
    }
    retry_block = ""
    if retry_errors:
        retry_block = (
            "This is a targeted extraction retry. Correct the listed missing or invalid fields by copying exact spans from input. "
            "If validation_errors says a field is missing, search the original input for that field's label or alias and emit its evidence_quote. "
            "If Request payload includes evidence_hints, treat them only as exact-substring candidates to audit against the input; "
            "you may use a hinted quote when it is the best direct support for that field. "
            "Do not invent values, paraphrase quotes, or require the input fields to appear in canonical order."
        )
    return f"""Return exactly one JSON object that follows the BankProfileParseV2EvidenceExtraction contract.
The first character of your response must be '{{' and the last character must be '}}'.
Stop immediately after the closing '}}'. Do not include markdown fences or commentary.

Extract evidence spans for the {dataset_label}. This stage does not return normalized values.
Every evidence_quote must be copied exactly from the input text. Never paraphrase, trim away number words, or fix typos.
Return one field entry for every explicit Bank profile field present in the input, even when the value is written as words.
Do not skip a field because the number is in Vietnamese or English words; this stage only copies the supporting span.
Income examples include "thu nhap nam hai muoi chin" and "income ninety four per year".
CCAvg examples include "chi tieu the trung binh khong phay bon moi thang" and "average credit-card spending zero point eight each month".
Boolean examples include "hien tai khong co CD account", "hien tai khong co tai khoan chung khoan", "currently CD account no", and "currently securities account no".
Do not emit a field unless the evidence_quote is directly supported by the input.
Do not use overlapping or nested evidence spans for different fields.
{retry_block}

Instructions:
{json.dumps(instructions, ensure_ascii=True, indent=2)}

Behavioral few-shot examples:
{json.dumps(_bank_profile_parse_v2_extraction_exemplars(), ensure_ascii=True, indent=2)}

Request payload:
{json.dumps(payload, ensure_ascii=True, indent=2)}"""


def build_bank_profile_parse_v2_normalization_prompt(
    benchmark,
    *,
    user_text: str,
    extracted_evidence: dict[str, Any] | None,
    dataset_label: str = "bank profile",
    retry_errors: list[str] | tuple[str, ...] | None = None,
    previous_output: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "input": user_text,
        "extracted_fields": extracted_evidence or {},
    }
    if retry_errors:
        payload["previous_output"] = previous_output or ""
        payload["validation_errors"] = list(retry_errors)
    instructions = {
        "contract": "BankProfileParseV2",
        "field_order": _feature_order_for_benchmark(benchmark),
        "field_dictionary": _feature_dictionary(benchmark),
        "schema_reference": {
            "task": BANK_PROFILE_PARSE_V2_TASK,
            "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
            "status": sorted(_VALID_STATUS_VALUES),
            "fields": {
                "FieldName": {
                    "value": "normalized numeric or binary value",
                    "evidence_quote": "exact substring copied from the extracted evidence",
                    "confidence": "number from 0.0 to 1.0",
                    "normalization_note": "short audit note explaining how the quote maps to the value",
                }
            },
            "constraint_spec": {
                "disallowed_changes": ["canonical field name"],
                "numeric_bounds": {"Income": {"min": "number", "max": "number"}},
                "max_changed_features": "integer 1 to 3",
                "prefer_fewer_changes": "boolean",
            },
            "missing_fields": ["canonical field name"],
            "conflicts": ["brief string"],
            "notes": ["brief string"],
        },
        "hard_rules": [
            "Normalize only from extracted_fields evidence_quote values.",
            "Do not inspect other input text to infer a value.",
            "Income and Mortgage must be non-negative integers, never decimals.",
            "CCAvg must be non-negative and may use at most one decimal digit.",
            "Do not insert a decimal point unless the quote explicitly contains phay or point.",
            "Do not drop the last token in a number-word phrase.",
            "Do not convert khong phay X into X.0.",
        ],
    }
    retry_block = ""
    if retry_errors:
        retry_block = (
            "This is a targeted normalization retry. Correct only the listed field-level issues. "
            "Reuse the extracted evidence exactly as provided."
        )
    return f"""Return exactly one JSON object that follows the BankProfileParseV2 contract.
The first character of your response must be '{{' and the last character must be '}}'.
Stop immediately after the closing '}}'. Do not include markdown fences or commentary.

Normalize the extracted evidence for the {dataset_label}. This stage is not allowed to extract new evidence.
For every emitted field, value must be supported only by the paired evidence_quote from extracted_fields.
{retry_block}

Instructions:
{json.dumps(instructions, ensure_ascii=True, indent=2)}

Behavioral few-shot examples:
{json.dumps(_bank_profile_parse_v2_normalization_exemplars(), ensure_ascii=True, indent=2)}

Request payload:
{json.dumps(payload, ensure_ascii=True, indent=2)}"""


def build_bank_profile_parse_v2_prompt(
    benchmark,
    *,
    user_text: str,
    dataset_label: str = "bank profile",
    extracted_evidence: dict[str, Any] | None = None,
    retry_errors: list[str] | tuple[str, ...] | None = None,
    previous_output: str | None = None,
    extraction_hints: dict[str, Any] | None = None,
    stage: str = BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
) -> str:
    if stage == BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE:
        return build_bank_profile_parse_v2_extraction_prompt(
            benchmark,
            user_text=user_text,
            dataset_label=dataset_label,
            retry_errors=retry_errors,
            previous_output=previous_output,
            extraction_hints=extraction_hints,
        )
    return build_bank_profile_parse_v2_normalization_prompt(
        benchmark,
        user_text=user_text,
        extracted_evidence=extracted_evidence,
        dataset_label=dataset_label,
        retry_errors=retry_errors,
        previous_output=previous_output,
    )


def build_bank_profile_parse_v2_verifier_prompt(
    benchmark,
    *,
    user_text: str,
    candidate: dict[str, Any],
    validation_errors: list[str] | tuple[str, ...] | None = None,
) -> str:
    payload = {
        "input": user_text,
        "candidate": candidate,
        "validation_errors": list(validation_errors or []),
    }
    instructions = {
        "contract": "BankProfileParseV2Verifier",
        "field_order": _feature_order_for_benchmark(benchmark),
        "verdict": {
            "PASS": "candidate is fully entailed by the input and every evidence_quote is exact",
            "CORRECTED": "candidate needed correction and corrected candidate is returned",
            "FAIL": "candidate cannot be safely corrected from the input",
        },
        "audit_rules": [
            "Audit every numeric and binary field against its evidence_quote.",
            "Check quote-to-value semantics, not only substring presence.",
            "Use CORRECTED only when the corrected candidate can pass the full validator.",
            "Use FAIL when the quote does not support a safe normalized value.",
        ],
    }
    return f"""You are an independent semantic verifier for a BankProfileParseV2 candidate.
Return exactly one JSON object with stage, verdict, candidate, errors, and notes.
The first character of your response must be '{{' and the last character must be '}}'.
Stop immediately after the closing '}}'. Do not include markdown fences or commentary.

Audit every field against the input text. The candidate may normalize number words, but each evidence_quote must remain exact.
If a value is wrong, missing, or semantically unsupported, return verdict CORRECTED with a corrected candidate.
If the candidate is already correct, return verdict PASS and repeat the candidate unchanged.
If the input does not support a safe correction, return verdict FAIL and keep the safest candidate.
Use errors only for unresolved verifier failures; PASS and CORRECTED must return an empty errors array.

Verifier instructions:
{json.dumps(instructions, ensure_ascii=True, indent=2)}

Verification payload:
{json.dumps(payload, ensure_ascii=True, indent=2)}"""


def run_bank_profile_parse_v2_quality(
    *,
    message_text: str,
    benchmark_spec,
    user_text: str,
    api_error: str | None = None,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> BankProfileParseV2QualityResult:
    if api_error and not str(message_text or "").strip():
        normalized = NormalizedParseResult(
            normalized_text="",
            parsed_json=None,
            parse_error=None,
            used_brace_extraction=False,
        )
        validation = ValidationResult(
            is_valid=False,
            errors=(str(api_error),),
            unexpected_top_level_keys=(),
            unexpected_cf_fields=(),
        )
        return BankProfileParseV2QualityResult(
            normalized=normalized,
            schema_validation=validation,
            field_provenance={},
            metadata=_metadata_for_candidate(None, validation),
            v2_payload=None,
        )

    v2_normalized = normalize_and_parse(message_text)
    v2_payload = v2_normalized.parsed_json if isinstance(v2_normalized.parsed_json, dict) else None
    return run_bank_profile_parse_v2_quality_from_candidate(
        candidate=v2_payload,
        benchmark_spec=benchmark_spec,
        user_text=user_text,
        parse_error=v2_normalized.parse_error,
        used_brace_extraction=v2_normalized.used_brace_extraction,
        normalized_text=v2_normalized.normalized_text,
        numeric_bound_fields=numeric_bound_fields,
    )


def run_bank_profile_parse_v2_quality_from_candidate(
    *,
    candidate: dict[str, Any] | None,
    benchmark_spec,
    user_text: str,
    parse_error: str | None = None,
    used_brace_extraction: bool = False,
    normalized_text: str | None = None,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> BankProfileParseV2QualityResult:
    errors: list[str] = []
    canonical_candidate: dict[str, Any] | None = None
    deterministic_corrections: list[dict[str, Any]] = []
    if parse_error:
        errors.append(parse_error)
    if not isinstance(candidate, dict):
        errors.append("No BankProfileParseV2 JSON object available.")
    else:
        candidate, deterministic_corrections = apply_bank_profile_parse_v2_deterministic_guard(candidate)
        errors.extend(
            validate_bank_profile_parse_v2_payload(
                candidate,
                benchmark_spec=benchmark_spec,
                user_text=user_text,
                numeric_bound_fields=numeric_bound_fields,
            )
        )
        if not errors:
            canonical_candidate = bank_profile_parse_v2_to_canonical(
                candidate,
                benchmark_spec=benchmark_spec,
            )

    if canonical_candidate is None:
        validation = ValidationResult(
            is_valid=False,
            errors=tuple(_dedupe_strings(errors)),
            unexpected_top_level_keys=(),
            unexpected_cf_fields=(),
        )
        normalized = NormalizedParseResult(
            normalized_text=normalized_text or "",
            parsed_json=None,
            parse_error=parse_error,
            used_brace_extraction=bool(used_brace_extraction),
        )
        return BankProfileParseV2QualityResult(
            normalized=normalized,
            schema_validation=validation,
            field_provenance={},
            metadata=_metadata_for_candidate(
                candidate,
                validation,
                deterministic_corrections=deterministic_corrections,
            ),
            v2_payload=candidate,
        )

    schema_validation = validate_prediction(
        canonical_candidate,
        benchmark_spec,
        numeric_bound_fields=numeric_bound_fields,
        require_field_evidence=True,
    )
    if schema_validation.errors:
        schema_validation = _merge_validation_errors(schema_validation, errors)
    normalized = NormalizedParseResult(
        normalized_text=json.dumps(canonical_candidate, ensure_ascii=True, sort_keys=True),
        parsed_json=canonical_candidate,
        parse_error=None,
        used_brace_extraction=bool(used_brace_extraction),
    )
    field_provenance = {
        field_name: BANK_PROFILE_PARSE_V2_PROVENANCE
        for field_name in (canonical_candidate.get("cf_request") or {})
    }
    return BankProfileParseV2QualityResult(
        normalized=normalized,
        schema_validation=schema_validation,
        field_provenance=field_provenance,
        metadata=_metadata_for_candidate(
            canonical_candidate,
            schema_validation,
            deterministic_corrections=deterministic_corrections,
        ),
        v2_payload=candidate,
    )


def apply_bank_profile_parse_v2_deterministic_guard(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields = candidate.get("fields")
    if not isinstance(fields, dict):
        return candidate, []

    updated = deepcopy(candidate)
    updated_fields = updated.get("fields")
    if not isinstance(updated_fields, dict):
        return candidate, []

    corrections: list[dict[str, Any]] = []
    for field_name, field_payload in updated_fields.items():
        if not isinstance(field_payload, dict):
            continue
        if "value" not in field_payload:
            continue
        evidence_quote = field_payload.get("evidence_quote")
        if not isinstance(evidence_quote, str) or not evidence_quote.strip():
            continue
        deterministic = deterministic_bank_profile_parse_v2_value(str(field_name), evidence_quote)
        if deterministic is None:
            continue
        current_value = field_payload.get("value")
        if _bank_profile_parse_v2_values_equal(str(field_name), current_value, deterministic.value):
            continue
        field_payload["value"] = deterministic.value
        old_note = str(field_payload.get("normalization_note") or "").strip()
        correction_note = (
            f"deterministic evidence correction: {current_value!r} -> {deterministic.value!r} "
            f"({deterministic.reason})"
        )
        field_payload["normalization_note"] = (
            f"{old_note}; {correction_note}" if old_note else correction_note
        )
        corrections.append(
            {
                "field_name": str(field_name),
                "old_value": current_value,
                "new_value": deterministic.value,
                "evidence_quote": evidence_quote,
                "reason": deterministic.reason,
                "failure_family": _deterministic_failure_family(
                    str(field_name),
                    evidence_quote=evidence_quote,
                    old_value=current_value,
                    new_value=deterministic.value,
                ),
            }
        )
    return (updated, corrections) if corrections else (candidate, [])


def deterministic_bank_profile_parse_v2_value(
    field_name: str,
    evidence_quote: str,
) -> BankProfileParseV2DeterministicResult | None:
    if field_name in _BANK_BOOLEAN_FIELDS:
        boolean_value = _deterministic_boolean_value(evidence_quote)
        if boolean_value is None:
            return None
        return BankProfileParseV2DeterministicResult(
            value=boolean_value,
            reason="field-aware boolean evidence parse",
        )

    value_text = _strip_bank_profile_parse_v2_field_label(field_name, evidence_quote)
    if not value_text:
        return None

    digit_match = re.search(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", value_text)
    if digit_match:
        numeric_value = float(digit_match.group(0))
        if field_name == "CCAvg":
            return BankProfileParseV2DeterministicResult(
                value=round(numeric_value, 1),
                reason="field-aware digit evidence parse",
            )
        if numeric_value.is_integer():
            return BankProfileParseV2DeterministicResult(
                value=int(numeric_value),
                reason="field-aware digit evidence parse",
            )
        return None

    if field_name == "CCAvg":
        decimal_value = _parse_decimal_words(value_text)
        if decimal_value is None:
            return None
        return BankProfileParseV2DeterministicResult(
            value=decimal_value,
            reason="field-aware decimal word evidence parse",
        )

    integer_value = _parse_integer_words(value_text)
    if integer_value is None:
        return None
    return BankProfileParseV2DeterministicResult(
        value=integer_value,
        reason="field-aware integer word evidence parse",
    )


def _strip_bank_profile_parse_v2_field_label(field_name: str, evidence_quote: str) -> str:
    text = _normalize_evidence_text(evidence_quote)
    for pattern in _FIELD_LABEL_PATTERNS.get(field_name, ()):
        text = re.sub(rf"^\s*(?:{pattern})\s+", "", text, count=1)
    for suffix in _FIELD_VALUE_SUFFIX_PATTERNS.get(field_name, ()):
        text = re.sub(rf"\s+(?:{suffix})\s*$", "", text, count=1)
    return " ".join(text.split())


def _normalize_evidence_text(value: str) -> str:
    lowered = str(value).casefold().replace("-", " ")
    lowered = re.sub(r"[^a-z0-9.]+", " ", lowered)
    return " ".join(lowered.split())


def _parse_decimal_words(value_text: str) -> float | None:
    tokens = value_text.split()
    separator = None
    for candidate in ("phay", "point"):
        if candidate in tokens:
            separator = candidate
            break
    if separator is None:
        integer_value = _parse_integer_words(value_text)
        return float(integer_value) if integer_value is not None else None
    separator_index = tokens.index(separator)
    left_tokens = tokens[:separator_index] or ["khong"]
    right_tokens = tokens[separator_index + 1 :]
    if not right_tokens:
        return None
    integer_part = _parse_integer_words(" ".join(left_tokens))
    decimal_digits = [_single_digit_word(token) for token in right_tokens]
    if integer_part is None or any(value is None for value in decimal_digits):
        return None
    decimal_text = "".join(str(value) for value in decimal_digits if value is not None)
    if not decimal_text:
        return None
    return round(float(f"{integer_part}.{decimal_text}"), len(decimal_text))


def _parse_integer_words(value_text: str) -> int | None:
    tokens = [
        token
        for token in _normalize_evidence_text(value_text).split()
        if token not in _WORD_VALUE_FILLERS
    ]
    if not tokens:
        return None
    if all(token in _VI_NUMBER_WORDS or token in {"muoi", "tram"} for token in tokens):
        return _parse_vi_integer_tokens(tokens)
    if all(token in _EN_NUMBER_WORDS or token in _EN_TENS_WORDS or token == "hundred" for token in tokens):
        return _parse_en_integer_tokens(tokens)
    return None


def _parse_vi_integer_tokens(tokens: list[str]) -> int | None:
    if tokens == ["khong"]:
        return 0
    if "tram" in tokens:
        index = tokens.index("tram")
        if index == 0:
            return None
        hundreds = _single_digit_word(tokens[index - 1])
        if hundreds is None:
            return None
        remainder = tokens[index + 1 :]
        if not remainder:
            return hundreds * 100
        remainder_value = _parse_vi_under_100(remainder)
        return None if remainder_value is None else hundreds * 100 + remainder_value
    return _parse_vi_under_100(tokens)


def _parse_vi_under_100(tokens: list[str]) -> int | None:
    if not tokens:
        return 0
    if tokens[0] == "muoi":
        if len(tokens) == 1:
            return 10
        digit = _single_digit_word(tokens[1])
        return None if digit is None or len(tokens) > 2 else 10 + digit
    first = _single_digit_word(tokens[0])
    if first is None:
        return None
    if len(tokens) == 1:
        return first
    if tokens[1] == "muoi":
        if len(tokens) == 2:
            return first * 10
        digit = _single_digit_word(tokens[2])
        return None if digit is None or len(tokens) > 3 else first * 10 + digit
    return None


def _parse_en_integer_tokens(tokens: list[str]) -> int | None:
    if tokens == ["zero"]:
        return 0
    total = 0
    current = 0
    for token in tokens:
        if token in _EN_NUMBER_WORDS:
            current += _EN_NUMBER_WORDS[token]
            continue
        if token in _EN_TENS_WORDS:
            current += _EN_TENS_WORDS[token]
            continue
        if token == "hundred":
            if current == 0:
                return None
            current *= 100
            continue
        return None
    total += current
    return total if total >= 0 else None


def _single_digit_word(token: str) -> int | None:
    if token in _VI_NUMBER_WORDS:
        value = _VI_NUMBER_WORDS[token]
        return value if 0 <= value <= 9 else None
    if token in _EN_NUMBER_WORDS:
        value = _EN_NUMBER_WORDS[token]
        return value if 0 <= value <= 9 else None
    return None


def _deterministic_boolean_value(evidence_quote: str) -> int | None:
    tokens = _normalize_evidence_text(evidence_quote).split()
    token_text = " ".join(tokens)
    if not tokens:
        return None
    negative_phrases = (
        "khong co",
        "khong dung",
        "do not have",
        "do not use",
        "not have",
        "not use",
    )
    if any(phrase in token_text for phrase in negative_phrases):
        return 0
    if tokens[-1] in {"khong", "no"}:
        return 0
    positive_phrases = (
        "toi co",
        "hien tai co",
        "i have",
        "i use",
    )
    if any(phrase in token_text for phrase in positive_phrases):
        return 1
    if tokens[-1] in {"co", "yes"}:
        return 1
    return None


def _bank_profile_parse_v2_values_equal(field_name: str, left: Any, right: Any) -> bool:
    try:
        if field_name == "CCAvg":
            return abs(float(left) - float(right)) < 1e-9
        if field_name in _BANK_BOOLEAN_FIELDS or field_name in _BANK_INTEGER_VALUED_FIELDS:
            return int(left) == int(right)
        if BANK_FEATURE_TYPES.get(field_name) == "int":
            return int(left) == int(right)
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


def _deterministic_failure_family(
    field_name: str,
    *,
    evidence_quote: str,
    old_value: Any,
    new_value: Any,
) -> str:
    del new_value
    quote = _normalize_evidence_text(evidence_quote)
    if field_name == "Income" and quote.startswith("thu nhap nam "):
        try:
            if int(old_value) >= 50:
                return "field_aware_number_prefix_collision"
        except (TypeError, ValueError):
            return "field_aware_number_prefix_collision"
    return "deterministic_evidence_value_correction"


_FIELD_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "Income": (r"thu nhap nam", r"thu nhap", r"annual income", r"income"),
    "Family": (r"gia dinh", r"family size", r"family"),
    "CCAvg": (
        r"chi tieu the trung binh",
        r"chi tieu the",
        r"average credit card spending",
        r"average credit spending",
        r"average card spending",
        r"average card spend",
        r"monthly card spend",
        r"monthly ccavg",
        r"ccavg",
    ),
    "Education": (r"hoc van muc", r"hoc van", r"education level"),
    "Mortgage": (r"the chap", r"mortgage"),
}

_FIELD_VALUE_SUFFIX_PATTERNS: dict[str, tuple[str, ...]] = {
    "Income": (r"per year",),
    "Family": (r"nguoi",),
    "CCAvg": (r"moi thang", r"each month", r"per month"),
}

_WORD_VALUE_FILLERS = {"and", "a", "level", "muc"}

_VI_NUMBER_WORDS = {
    "khong": 0,
    "mot": 1,
    "hai": 2,
    "ba": 3,
    "bon": 4,
    "nam": 5,
    "sau": 6,
    "bay": 7,
    "tam": 8,
    "chin": 9,
}

_EN_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_EN_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def validate_bank_profile_parse_v2_extraction_payload(
    candidate: dict[str, Any],
    *,
    benchmark_spec,
    user_text: str,
) -> list[str]:
    errors: list[str] = []
    allowed_fields = set(_feature_order_for_benchmark(benchmark_spec))
    if candidate.get("task") != BANK_PROFILE_PARSE_V2_TASK:
        errors.append(f"task must equal {BANK_PROFILE_PARSE_V2_TASK!r}.")
    stage = candidate.get("stage")
    if stage not in {None, BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE}:
        errors.append(
            f"stage must be {BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE!r} for evidence extraction."
        )
    if candidate.get("status") not in _VALID_STATUS_VALUES:
        errors.append("status must be one of complete, conflict, needs_clarification, partial.")

    fields = candidate.get("fields")
    if not isinstance(fields, dict):
        errors.append("fields must be an object.")
        fields = {}
    else:
        unknown_fields = sorted(str(field_name) for field_name in fields if field_name not in allowed_fields)
        if unknown_fields:
            errors.append("fields contains unknown field names: " + ", ".join(unknown_fields))

    for field_name, field_payload in fields.items():
        if field_name not in allowed_fields:
            continue
        errors.extend(_validate_evidence_field_payload(field_name, field_payload, user_text=user_text))

    if candidate.get("status") == "complete" and isinstance(fields, dict):
        omitted_fields = [field_name for field_name in _feature_order_for_benchmark(benchmark_spec) if field_name not in fields]
        if omitted_fields:
            errors.append("complete extraction omitted field entries: " + ", ".join(omitted_fields))

    errors.extend(_validate_missing_and_conflict_lists(candidate, allowed_fields=allowed_fields))
    if isinstance(fields, dict):
        errors.extend(_validate_overlapping_quotes(fields, user_text=user_text))
    return _dedupe_strings(errors)


def validate_bank_profile_parse_v2_payload(
    candidate: dict[str, Any],
    *,
    benchmark_spec,
    user_text: str,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    allowed_fields = set(_feature_order_for_benchmark(benchmark_spec))
    if candidate.get("task") != BANK_PROFILE_PARSE_V2_TASK:
        errors.append(f"task must equal {BANK_PROFILE_PARSE_V2_TASK!r}.")
    stage = candidate.get("stage")
    if stage not in {None, BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE}:
        errors.append(
            f"stage must be {BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE!r} for normalized candidates."
        )
    if candidate.get("status") not in _VALID_STATUS_VALUES:
        errors.append("status must be one of complete, conflict, needs_clarification, partial.")

    fields = candidate.get("fields")
    if not isinstance(fields, dict):
        errors.append("fields must be an object.")
        fields = {}
    else:
        unknown_fields = sorted(str(field_name) for field_name in fields if field_name not in allowed_fields)
        if unknown_fields:
            errors.append("fields contains unknown field names: " + ", ".join(unknown_fields))

    for field_name, field_payload in fields.items():
        if field_name not in allowed_fields:
            continue
        errors.extend(_validate_v2_field_payload(field_name, field_payload, user_text=user_text))

    errors.extend(_validate_missing_and_conflict_lists(candidate, allowed_fields=allowed_fields))
    if isinstance(fields, dict):
        errors.extend(_validate_overlapping_quotes(fields, user_text=user_text))

    _, constraint_errors = validate_and_normalize_constraint_spec(
        candidate.get("constraint_spec"),
        feature_order=_feature_order_for_benchmark(benchmark_spec),
        numeric_bound_fields=list(numeric_bound_fields or ["Income", "CCAvg", "Mortgage"]),
    )
    errors.extend(constraint_errors)
    return _dedupe_strings(errors)


def bank_profile_parse_v2_to_canonical(
    candidate: dict[str, Any],
    *,
    benchmark_spec,
) -> dict[str, Any]:
    feature_order = _feature_order_for_benchmark(benchmark_spec)
    fields = candidate.get("fields")
    cf_request: dict[str, Any] = {}
    field_evidence: dict[str, dict[str, str]] = {}
    if isinstance(fields, dict):
        for field_name in feature_order:
            field_payload = fields.get(field_name)
            if not isinstance(field_payload, dict):
                continue
            cf_request[field_name] = field_payload.get("value")
            field_evidence[field_name] = {
                "source_text": str(field_payload.get("evidence_quote", "")),
                "evidence_kind": "explicit_boolean" if field_name in _BANK_BOOLEAN_FIELDS else "explicit_numeric",
                "normalized_from": str(field_payload.get("evidence_quote", "")),
                "language": "unknown",
            }
    canonical = {
        "task": "extract_cf_request",
        "status": candidate.get("status"),
        "cf_request": cf_request,
        "field_evidence": field_evidence,
        "missing_fields": list(candidate.get("missing_fields") or []),
        "conflicts": list(candidate.get("conflicts") or []),
        "notes": list(candidate.get("notes") or []),
    }
    constraint_spec = candidate.get("constraint_spec")
    if isinstance(constraint_spec, dict):
        canonical["constraint_spec"] = dict(constraint_spec)
    return canonical


def parse_bank_profile_parse_v2_extraction_payload(message_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    normalized = normalize_and_parse(message_text)
    errors: list[str] = []
    if normalized.parse_error:
        errors.append(normalized.parse_error)
    payload = normalized.parsed_json
    if not isinstance(payload, dict):
        errors.append("Evidence extractor did not return a JSON object.")
        return None, _dedupe_strings(errors)
    if payload.get("task") != BANK_PROFILE_PARSE_V2_TASK:
        errors.append(f"task must equal {BANK_PROFILE_PARSE_V2_TASK!r}.")
    stage = payload.get("stage")
    if stage not in {None, BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE}:
        errors.append("evidence extractor stage must be evidence_extraction.")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        errors.append("fields must be an object.")
    if not _is_string_list(payload.get("missing_fields")):
        errors.append("missing_fields must be an array of strings.")
    if not _is_string_list(payload.get("conflicts")):
        errors.append("conflicts must be an array of strings.")
    if not _is_string_list(payload.get("notes")):
        errors.append("notes must be an array of strings.")
    return payload, _dedupe_strings(errors)


def parse_bank_profile_parse_v2_verifier_payload(message_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    normalized = normalize_and_parse(message_text)
    errors: list[str] = []
    if normalized.parse_error:
        errors.append(normalized.parse_error)
    payload = normalized.parsed_json
    if not isinstance(payload, dict):
        errors.append("Verifier did not return a JSON object.")
        return None, _dedupe_strings(errors)
    stage = payload.get("stage")
    if stage not in {None, BANK_PROFILE_PARSE_V2_VERIFIER_STAGE}:
        errors.append("verifier stage must be semantic_verification.")
    if payload.get("verdict") not in {"PASS", "CORRECTED", "FAIL"}:
        errors.append("verifier verdict must be PASS, CORRECTED, or FAIL.")
    if not isinstance(payload.get("candidate"), dict):
        errors.append("verifier candidate must be a BankProfileParseV2 object.")
    if not _is_string_list(payload.get("errors")):
        errors.append("verifier errors must be an array of strings.")
    else:
        errors.extend(
            "verifier reported: " + str(error)
            for error in payload.get("errors")
            if str(error).strip()
        )
    if not _is_string_list(payload.get("notes")):
        errors.append("verifier notes must be an array of strings.")
    return payload, _dedupe_strings(errors)


def classify_bank_profile_v2_failed_stage(errors: list[str] | tuple[str, ...]) -> str | None:
    normalized_errors = [str(error).lower() for error in errors if str(error).strip()]
    if not normalized_errors:
        return None
    extraction_markers = (
        "evidence_quote",
        "exact substring",
        "overlap between",
        "missing required keys: evidence_quote",
        "reason must be a string",
    )
    normalization_markers = (
        "does not support",
        ".value must",
        "integer-valued",
        "decimal place",
        "outside the bank domain",
    )
    if any(marker in error for error in normalized_errors for marker in extraction_markers):
        return BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE
    if any(marker in error for error in normalized_errors for marker in normalization_markers):
        return BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE
    return BANK_PROFILE_PARSE_V2_VERIFIER_STAGE


def _validate_missing_and_conflict_lists(candidate: dict[str, Any], *, allowed_fields: set[str]) -> list[str]:
    errors: list[str] = []
    missing_fields = candidate.get("missing_fields")
    if not _is_string_list(missing_fields):
        errors.append("missing_fields must be an array of strings.")
    else:
        unknown_missing = sorted(field for field in missing_fields if field not in allowed_fields)
        if unknown_missing:
            errors.append("missing_fields contains unknown field names: " + ", ".join(unknown_missing))

    conflicts = candidate.get("conflicts")
    if not _is_string_list(conflicts):
        errors.append("conflicts must be an array of strings.")
    notes = candidate.get("notes")
    if not _is_string_list(notes):
        errors.append("notes must be an array of strings.")

    fields = candidate.get("fields")
    if isinstance(fields, dict) and isinstance(missing_fields, list):
        overlap = sorted(field_name for field_name in fields if field_name in set(missing_fields))
        if overlap:
            errors.append("fields and missing_fields overlap: " + ", ".join(overlap))
    return errors


def _validate_evidence_field_payload(field_name: str, field_payload: Any, *, user_text: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(field_payload, dict):
        return [f"fields.{field_name} must be an object."]
    required = ("evidence_quote", "confidence", "reason")
    missing = [key for key in required if key not in field_payload]
    extra = [key for key in field_payload if key not in required]
    if missing:
        errors.append(f"fields.{field_name} missing required keys: " + ", ".join(missing))
    if extra:
        errors.append(f"fields.{field_name} has unsupported keys: " + ", ".join(extra))

    evidence_quote = field_payload.get("evidence_quote")
    if not isinstance(evidence_quote, str) or not evidence_quote.strip():
        errors.append(f"fields.{field_name}.evidence_quote must be a non-empty string.")
    elif not _quote_present_in_input(evidence_quote, user_text):
        errors.append(f"fields.{field_name}.evidence_quote is not an exact substring of the input.")

    confidence = field_payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append(f"fields.{field_name}.confidence must be numeric.")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"fields.{field_name}.confidence must be between 0.0 and 1.0.")

    if not isinstance(field_payload.get("reason"), str):
        errors.append(f"fields.{field_name}.reason must be a string.")
    return errors


def _validate_v2_field_payload(field_name: str, field_payload: Any, *, user_text: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(field_payload, dict):
        return [f"fields.{field_name} must be an object."]
    required = ("value", "evidence_quote", "confidence", "normalization_note")
    missing = [key for key in required if key not in field_payload]
    extra = [key for key in field_payload if key not in required]
    if missing:
        errors.append(f"fields.{field_name} missing required keys: " + ", ".join(missing))
    if extra:
        errors.append(f"fields.{field_name} has unsupported keys: " + ", ".join(extra))

    value_error = _validate_bank_field_domain(field_name, field_payload.get("value"))
    if value_error:
        errors.append(value_error)

    evidence_quote = field_payload.get("evidence_quote")
    if not isinstance(evidence_quote, str) or not evidence_quote.strip():
        errors.append(f"fields.{field_name}.evidence_quote must be a non-empty string.")
    elif not _quote_present_in_input(evidence_quote, user_text):
        errors.append(f"fields.{field_name}.evidence_quote is not an exact substring of the input.")

    confidence = field_payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append(f"fields.{field_name}.confidence must be numeric.")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"fields.{field_name}.confidence must be between 0.0 and 1.0.")

    if not isinstance(field_payload.get("normalization_note"), str):
        errors.append(f"fields.{field_name}.normalization_note must be a string.")
    return errors


def _validate_bank_field_domain(field_name: str, value: Any) -> str | None:
    expected_type = BANK_FEATURE_TYPES.get(field_name)
    if expected_type == "binary":
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            return f"fields.{field_name}.value must be binary 0 or 1."
        return None

    if field_name in _BANK_INTEGER_VALUED_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"fields.{field_name}.value must be a non-negative integer."
        numeric_value = float(value)
        if numeric_value < 0:
            return f"fields.{field_name}.value must be non-negative."
        if isinstance(value, float) and not value.is_integer():
            return f"fields.{field_name}.value must be integer-valued, got {value!r}."
        if isinstance(value, int):
            return None
        if numeric_value.is_integer():
            return None
        return f"fields.{field_name}.value must be integer-valued, got {value!r}."

    if field_name == "CCAvg":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "fields.CCAvg.value must be numeric."
        if float(value) < 0:
            return "fields.CCAvg.value must be non-negative."
        if not _has_at_most_one_decimal_place(value):
            return "fields.CCAvg.value must have at most one decimal place."
        return None

    if expected_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"fields.{field_name}.value must be an integer."
        allowed = _BANK_INTEGER_DOMAINS.get(field_name)
        if allowed is not None and value not in allowed:
            return f"fields.{field_name}.value is outside the Bank domain."
        return None

    if expected_type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"fields.{field_name}.value must be numeric."
        if float(value) < 0:
            return f"fields.{field_name}.value must be non-negative."
        return None
    return None


def _validate_overlapping_quotes(fields: dict[str, Any], *, user_text: str) -> list[str]:
    spans: list[tuple[int, int, str]] = []
    for field_name, field_payload in fields.items():
        if not isinstance(field_payload, dict):
            continue
        evidence_quote = field_payload.get("evidence_quote")
        if not isinstance(evidence_quote, str) or not evidence_quote or evidence_quote not in user_text:
            continue
        start = user_text.find(evidence_quote)
        if start < 0:
            continue
        spans.append((start, start + len(evidence_quote), field_name))
    spans.sort()
    errors: list[str] = []
    for index in range(1, len(spans)):
        _, prev_end, prev_field = spans[index - 1]
        start, _, field_name = spans[index]
        if start < prev_end:
            errors.append(f"evidence_quote overlap between {prev_field} and {field_name}.")
    return errors


def _quote_present_in_input(quote: str, user_text: str) -> bool:
    if quote in user_text:
        return True
    normalized_quote = " ".join(quote.split())
    normalized_input = " ".join(user_text.split())
    return bool(normalized_quote) and normalized_quote in normalized_input


def _evidence_field_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_quote", "confidence", "reason"],
        "properties": {
            "evidence_quote": {"type": "string"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
    }


def _normalized_field_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "evidence_quote", "confidence", "normalization_note"],
        "properties": {
            "value": _field_value_schema(field_name),
            "evidence_quote": {"type": "string"},
            "confidence": {"type": "number"},
            "normalization_note": {"type": "string"},
        },
    }


def _field_value_schema(field_name: str) -> dict[str, Any]:
    if field_name in _BANK_BOOLEAN_FIELDS:
        return {"type": "integer", "enum": [0, 1]}
    if field_name in _BANK_INTEGER_VALUED_FIELDS:
        return {"type": "integer", "minimum": 0}
    if field_name == "CCAvg":
        return {"type": "number", "minimum": 0}
    if BANK_FEATURE_TYPES.get(field_name) == "int":
        return {"type": "integer"}
    return {"type": "number"}


def _constraint_spec_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "immutable": {"type": "array", "items": {"type": "string"}},
            "disallowed_changes": {"type": "array", "items": {"type": "string"}},
            "numeric_bounds": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "Income": _numeric_bound_schema(),
                    "CCAvg": _numeric_bound_schema(),
                    "Mortgage": _numeric_bound_schema(),
                },
            },
            "max_changed_features": {"type": "integer", "enum": [1, 2, 3]},
            "prefer_fewer_changes": {"type": "boolean"},
        },
    }


def _numeric_bound_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "min": {"type": "number"},
            "max": {"type": "number"},
        },
    }


def _feature_order_for_benchmark(benchmark) -> list[str]:
    benchmark_fields = [
        str(field.name)
        for field in getattr(benchmark, "target_cf_fields", ())
        if isinstance(getattr(field, "name", None), str)
    ]
    ordered = [field_name for field_name in BANK_REQUIRED_FIELD_ORDER if field_name in benchmark_fields]
    ordered.extend(field_name for field_name in benchmark_fields if field_name not in ordered)
    return ordered


def _feature_dictionary(benchmark) -> dict[str, dict[str, str]]:
    return {
        str(field.name): {
            "type": str(field.type),
            "description": str(field.description),
        }
        for field in getattr(benchmark, "target_cf_fields", ())
    }


def _bank_profile_parse_v2_extraction_exemplars() -> list[dict[str, Any]]:
    return [
        {
            "label": "vi_tens_teen_exact_quote",
            "input": "thu nhap tam muoi hai; gia dinh hai nguoi; the chap mot tram bon muoi tam.",
            "output": _v2_extraction_example_output(
                {
                    "Income": ("thu nhap tam muoi hai", "contains the full Vietnamese tens phrase"),
                    "Family": ("gia dinh hai nguoi", "contains the family size phrase"),
                    "Mortgage": ("the chap mot tram bon muoi tam", "contains the full hundreds phrase"),
                }
            ),
        },
        {
            "label": "vi_decimal_exact_quote",
            "input": "chi tieu the khong phay ba moi thang; online co.",
            "output": _v2_extraction_example_output(
                {
                    "CCAvg": ("chi tieu the khong phay ba", "contains the decimal phrase with phay"),
                    "Online": ("online co", "contains the explicit yes phrase"),
                }
            ),
        },
        {
            "label": "en_hundreds_exact_quote",
            "input": "annual income one hundred ninety three; mortgage one hundred forty eight; credit card no.",
            "output": _v2_extraction_example_output(
                {
                    "Income": ("annual income one hundred ninety three", "contains the full English hundreds phrase"),
                    "Mortgage": ("mortgage one hundred forty eight", "contains the full English hundreds phrase"),
                    "CreditCard": ("credit card no", "contains the explicit no phrase"),
                }
            ),
        },
        {
            "label": "vi_reordered_word_numbers_all_fields",
            "input": (
                "Ho so hien tai cua toi la: gia dinh bon nguoi; hoc van muc hai; "
                "thu nhap nam hai muoi chin; the chap khong; "
                "chi tieu the trung binh khong phay bon moi thang. "
                "hien tai co online banking, hien tai khong co the tin dung, "
                "hien tai khong co CD account, hien tai khong co tai khoan chung khoan."
            ),
            "output": _v2_extraction_example_output(
                {
                    "Income": ("thu nhap nam hai muoi chin", "contains the full Vietnamese income number words"),
                    "Family": ("gia dinh bon nguoi", "contains the family size phrase"),
                    "CCAvg": ("chi tieu the trung binh khong phay bon moi thang", "contains the full Vietnamese decimal phrase"),
                    "Education": ("hoc van muc hai", "contains the education level phrase"),
                    "Mortgage": ("the chap khong", "contains the mortgage zero phrase"),
                    "SecuritiesAccount": ("hien tai khong co tai khoan chung khoan", "contains the explicit no phrase"),
                    "CDAccount": ("hien tai khong co CD account", "contains the explicit no phrase"),
                    "Online": ("hien tai co online banking", "contains the explicit yes phrase"),
                    "CreditCard": ("hien tai khong co the tin dung", "contains the explicit no phrase"),
                }
            ),
        },
        {
            "label": "en_reordered_word_numbers_all_fields",
            "input": (
                "Current profile: family size one, education level three, income ninety four per year, "
                "mortgage two hundred twenty one, and average credit-card spending zero point eight each month. "
                "currently online banking no, currently credit card no, currently CD account no, "
                "currently securities account no."
            ),
            "output": _v2_extraction_example_output(
                {
                    "Income": ("income ninety four per year", "contains the full English income number words"),
                    "Family": ("family size one", "contains the family size phrase"),
                    "CCAvg": ("average credit-card spending zero point eight each month", "contains the full English decimal phrase"),
                    "Education": ("education level three", "contains the education level phrase"),
                    "Mortgage": ("mortgage two hundred twenty one", "contains the full English mortgage number words"),
                    "SecuritiesAccount": ("currently securities account no", "contains the explicit no phrase"),
                    "CDAccount": ("currently CD account no", "contains the explicit no phrase"),
                    "Online": ("currently online banking no", "contains the explicit no phrase"),
                    "CreditCard": ("currently credit card no", "contains the explicit no phrase"),
                }
            ),
        },
    ]


def _bank_profile_parse_v2_normalization_exemplars() -> list[dict[str, Any]]:
    return [
        {
            "label": "vi_tens_teen_normalization",
            "input": {
                "input": "thu nhap tam muoi hai; hoc van muc ba; online co.",
                "extracted_fields": {
                    "Income": _evidence_payload("thu nhap tam muoi hai", "contains the full Vietnamese tens phrase"),
                    "Education": _evidence_payload("hoc van muc ba", "contains the education phrase"),
                    "Online": _evidence_payload("online co", "contains the explicit yes phrase"),
                },
            },
            "output": _v2_example_output(
                {
                    "Income": (82, "thu nhap tam muoi hai", "tam muoi hai means 82"),
                    "Education": (3, "hoc van muc ba", "ba means education level 3"),
                    "Online": (1, "online co", "co means yes"),
                }
            ),
        },
        {
            "label": "vi_decimal_normalization",
            "input": {
                "input": "chi tieu the khong phay ba moi thang; the chap khong.",
                "extracted_fields": {
                    "CCAvg": _evidence_payload("chi tieu the khong phay ba", "contains the decimal phrase with phay"),
                    "Mortgage": _evidence_payload("the chap khong", "contains the explicit zero phrase"),
                },
            },
            "output": _v2_example_output(
                {
                    "CCAvg": (0.3, "chi tieu the khong phay ba", "khong phay ba means 0.3"),
                    "Mortgage": (0, "the chap khong", "khong means zero"),
                }
            ),
        },
        {
            "label": "en_hundreds_normalization",
            "input": {
                "input": "annual income one hundred ninety three; mortgage one hundred forty eight.",
                "extracted_fields": {
                    "Income": _evidence_payload("annual income one hundred ninety three", "contains the full English hundreds phrase"),
                    "Mortgage": _evidence_payload("mortgage one hundred forty eight", "contains the full English hundreds phrase"),
                },
            },
            "output": _v2_example_output(
                {
                    "Income": (193, "annual income one hundred ninety three", "one hundred ninety three means 193"),
                    "Mortgage": (148, "mortgage one hundred forty eight", "one hundred forty eight means 148"),
                }
            ),
        },
        {
            "label": "mixed_decimal_and_binary_normalization",
            "input": {
                "input": "Income 68; CCAvg zero point five; CDAccount khong.",
                "extracted_fields": {
                    "Income": _evidence_payload("Income 68", "contains the exact digit phrase"),
                    "CCAvg": _evidence_payload("CCAvg zero point five", "contains the decimal point phrase"),
                    "CDAccount": _evidence_payload("CDAccount khong", "contains the explicit no phrase"),
                },
            },
            "output": _v2_example_output(
                {
                    "Income": (68, "Income 68", "digit value 68"),
                    "CCAvg": (0.5, "CCAvg zero point five", "zero point five means 0.5"),
                    "CDAccount": (0, "CDAccount khong", "khong means no"),
                }
            ),
        },
    ]


def _v2_extraction_example_output(field_values: dict[str, tuple[str, str]]) -> dict[str, Any]:
    return {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
        "status": "partial" if len(field_values) < len(BANK_REQUIRED_FIELD_ORDER) else "complete",
        "fields": {
            field_name: _evidence_payload(evidence_quote, reason)
            for field_name, (evidence_quote, reason) in field_values.items()
        },
        "missing_fields": [
            field_name for field_name in BANK_REQUIRED_FIELD_ORDER if field_name not in field_values
        ],
        "conflicts": [],
        "notes": [],
    }


def _v2_example_output(field_values: dict[str, tuple[Any, str, str]]) -> dict[str, Any]:
    return {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        "status": "partial" if len(field_values) < len(BANK_REQUIRED_FIELD_ORDER) else "complete",
        "fields": {
            field_name: {
                "value": value,
                "evidence_quote": evidence_quote,
                "confidence": 1.0,
                "normalization_note": note,
            }
            for field_name, (value, evidence_quote, note) in field_values.items()
        },
        "missing_fields": [
            field_name for field_name in BANK_REQUIRED_FIELD_ORDER if field_name not in field_values
        ],
        "conflicts": [],
        "notes": [],
    }


def _evidence_payload(evidence_quote: str, reason: str) -> dict[str, Any]:
    return {
        "evidence_quote": evidence_quote,
        "confidence": 1.0,
        "reason": reason,
    }


def _metadata_for_candidate(
    candidate: dict[str, Any] | None,
    validation: ValidationResult,
    *,
    deterministic_corrections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    canonical_candidate = candidate if isinstance(candidate, dict) and candidate.get("cf_request") else None
    cf_request = canonical_candidate.get("cf_request") if isinstance(canonical_candidate, dict) else {}
    profile_facts = {
        field_name: cf_request[field_name]
        for field_name in BANK_REQUIRED_FIELD_ORDER
        if isinstance(cf_request, dict) and field_name in cf_request
    }
    hard_constraints: dict[str, Any] = {}
    soft_preferences: dict[str, Any] = {}
    if isinstance(canonical_candidate, dict):
        hard_constraints, soft_preferences = split_constraint_buckets(
            canonical_candidate.get("constraint_spec"),
            feature_order=list(BANK_REQUIRED_FIELD_ORDER),
        )
    reason_codes = [BANK_PROFILE_PARSE_V2_PROVENANCE]
    reason_codes.append("v2_schema_valid" if validation.is_valid else "v2_schema_or_evidence_invalid")
    correction_list = list(deterministic_corrections or [])
    if correction_list:
        reason_codes.append("v2_deterministic_evidence_correction")
    return {
        "reason_codes": reason_codes,
        "flags": {
            "deterministic_recovery_applied": bool(correction_list),
            "post_quality_schema_valid": bool(validation.is_valid),
            "canonical_pass_after_quality": False,
            "repair_invoked": False,
            "still_failed_after_quality": not bool(validation.is_valid),
            "constraint_extraction_absent": not bool(hard_constraints or soft_preferences),
        },
        "semantic_buckets": {
            "profile_facts": profile_facts,
            "hard_constraints": hard_constraints,
            "soft_preferences": soft_preferences,
        },
        "deterministic_corrections": correction_list,
    }


def _merge_validation_errors(validation: ValidationResult, errors: list[str]) -> ValidationResult:
    all_errors = _dedupe_strings(list(validation.errors) + list(errors))
    return ValidationResult(
        is_valid=not all_errors,
        errors=tuple(all_errors),
        unexpected_top_level_keys=validation.unexpected_top_level_keys,
        unexpected_cf_fields=validation.unexpected_cf_fields,
    )


def _has_at_most_one_decimal_place(value: Any) -> bool:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    exponent = decimal_value.normalize().as_tuple().exponent
    return exponent >= -1


def _dedupe_strings(values: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
