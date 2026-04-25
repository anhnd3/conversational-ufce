from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from llm.src.conversation.bank_completeness_guard import apply_bank_boolean_completeness_guard
from llm.src.conversation.bank_profile_extractor import (
    FIELD_PROVENANCE_CONFLICT,
    NUMBER_VALUE_PATTERN,
    build_alias_pattern,
    normalize_boolean_token,
    normalize_numeric_token,
    ordered_aliases_for_field,
    recover_explicit_labeled_bank_fields,
    recover_dense_bank_profile_candidate,
)
from llm.src.conversation.canonical_session_state import split_constraint_buckets
from llm.src.parser.response_normalizer import NormalizedParseResult, normalize_and_parse
from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from llm.src.runtime.datasets.bank.metadata import BANK_FEATURE_TYPES, BANK_REQUIRED_FIELD_ORDER
from llm.src.validation.schema_validator import ValidationResult, validate_prediction


ALIAS_KEY_NORMALIZED = "alias_key_normalized"
BINARY_STRING_COERCED = "binary_string_coerced"
NUMERIC_STRING_COERCED = "numeric_string_coerced"
PROFILE_FIELD_RECOVERED = "profile_field_recovered"
CONSTRAINT_SPEC_RECOVERED = "constraint_spec_recovered"
CONSTRAINT_PHRASE_RECOVERED = "constraint_phrase_recovered"
PREFERENCE_PHRASE_RECOVERED = "preference_phrase_recovered"
CONFLICTING_EXPLICIT_FIELD = "conflicting_explicit_field"
AMBIGUOUS_CONSTRAINT_PHRASE = "ambiguous_constraint_phrase"
AMBIGUOUS_PREFERENCE_PHRASE = "ambiguous_preference_phrase"
CONSTRAINT_SPEC_ABSENT = "constraint_spec_absent"

_CHANGE_WORD_TO_INT = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
_SEMANTIC_FLAG_KEYS = (
    "deterministic_recovery_applied",
    "post_quality_schema_valid",
    "canonical_pass_after_quality",
    "repair_invoked",
    "still_failed_after_quality",
    "constraint_extraction_absent",
)
_CONSTRAINT_VERB_RE = re.compile(r"\b(?:do\s+not|don't|dont)\s+change\b", re.IGNORECASE)
_BOUND_PHRASE_RE = re.compile(
    r"\b(?:must\s+stay\s+(?:under|below|over|above)|at\s+or\s+(?:below|above))\b",
    re.IGNORECASE,
)
_MAX_CHANGE_RE = re.compile(r"\bchange\s+at\s+most\b", re.IGNORECASE)
_PREFER_RE = re.compile(r"\bprefer\b", re.IGNORECASE)
_PREFER_FEWER_RE = re.compile(r"\bprefer\s+fewer\s+changes\b", re.IGNORECASE)
_PREFER_SMALLER_RE = re.compile(r"\bprefer\s+smaller\s+edits\b", re.IGNORECASE)


@dataclass(frozen=True)
class _BankPolicyAdapter:
    dataset_name: str = "bank"
    feature_type_map: dict[str, str] = field(default_factory=lambda: dict(BANK_FEATURE_TYPES))
    conversation_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {
            field_name: ordered_aliases_for_field(field_name=field_name)
            for field_name in BANK_REQUIRED_FIELD_ORDER
        }
    )


@dataclass(frozen=True)
class ParserQualityMetadata:
    reason_codes: tuple[str, ...] = ()
    deterministic_recovery_applied: bool = False
    post_quality_schema_valid: bool = False
    constraint_extraction_absent: bool = False
    semantic_buckets: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        *,
        canonical_pass_after_quality: bool = False,
        repair_invoked: bool = False,
    ) -> dict[str, Any]:
        return {
            "reason_codes": list(self.reason_codes),
            "flags": {
                "deterministic_recovery_applied": bool(self.deterministic_recovery_applied),
                "post_quality_schema_valid": bool(self.post_quality_schema_valid),
                "canonical_pass_after_quality": bool(canonical_pass_after_quality),
                "repair_invoked": bool(repair_invoked),
                "still_failed_after_quality": not bool(canonical_pass_after_quality),
                "constraint_extraction_absent": bool(self.constraint_extraction_absent),
            },
            "semantic_buckets": _stable_semantic_buckets(self.semantic_buckets),
        }


@dataclass(frozen=True)
class ParserQualityResult:
    normalized: NormalizedParseResult
    schema_validation: ValidationResult
    field_provenance: dict[str, str]
    metadata: ParserQualityMetadata


def run_parser_quality(
    message_text: str,
    benchmark_spec,
    user_text: str | None = None,
    api_error: str | None = None,
    dataset_id: str = "bank",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> ParserQualityResult:
    feature_order = _feature_order_for_benchmark(benchmark_spec)
    if should_skip_parse_for_api_error(api_error=api_error, message_text=message_text):
        normalized, validation = empty_parse_and_validation_result()
        metadata = ParserQualityMetadata(
            reason_codes=(),
            deterministic_recovery_applied=False,
            post_quality_schema_valid=False,
            constraint_extraction_absent=True,
            semantic_buckets=_empty_semantic_buckets(),
        )
        return ParserQualityResult(
            normalized=normalized,
            schema_validation=validation,
            field_provenance={},
            metadata=metadata,
        )

    normalized = normalize_and_parse(message_text)
    field_provenance: dict[str, str] = {}
    metadata_reasons: list[str] = []
    deterministic_recovery_applied = False
    candidate = normalized.parsed_json

    if isinstance(candidate, dict):
        candidate, alias_or_value_changed, alias_or_value_reasons = _normalize_parser_contract_fields(
            candidate,
            benchmark_spec=benchmark_spec,
            feature_order=feature_order,
        )
        metadata_reasons.extend(alias_or_value_reasons)
        deterministic_recovery_applied = deterministic_recovery_applied or alias_or_value_changed

        user_text_clean = str(user_text or "").strip()
        if user_text_clean and dataset_id == "bank":
            candidate, field_provenance, dense_recovery_changed = _recover_profile_fields_from_user_text(
                candidate=candidate,
                user_text=user_text_clean,
                required_fields=feature_order,
            )
            if dense_recovery_changed:
                metadata_reasons.append(PROFILE_FIELD_RECOVERED)
                deterministic_recovery_applied = True
            if any(value == FIELD_PROVENANCE_CONFLICT for value in field_provenance.values()):
                metadata_reasons.append(CONFLICTING_EXPLICIT_FIELD)

            candidate, constraint_reasons, constraint_changed = _recover_constraint_spec_from_user_text(
                candidate=candidate,
                user_text=user_text_clean,
                feature_order=feature_order,
            )
            metadata_reasons.extend(constraint_reasons)
            deterministic_recovery_applied = deterministic_recovery_applied or constraint_changed

            guard_result = apply_bank_boolean_completeness_guard(
                user_text=user_text_clean,
                candidate=candidate,
            )
            if guard_result.candidate != candidate:
                candidate = guard_result.candidate
                deterministic_recovery_applied = True

        normalized = replace(normalized, parsed_json=candidate)

    schema_validation = validate_prediction(
        normalized.parsed_json,
        benchmark_spec,
        numeric_bound_fields=numeric_bound_fields,
    )

    if schema_validation.is_valid and isinstance(normalized.parsed_json, dict):
        ordered_candidate = _post_schema_normalize_candidate(
            normalized.parsed_json,
            feature_order=feature_order,
        )
        if ordered_candidate != normalized.parsed_json:
            normalized = replace(normalized, parsed_json=ordered_candidate)
        semantic_buckets = _derive_semantic_buckets(
            ordered_candidate,
            feature_order=feature_order,
        )
        constraint_extraction_absent = _is_semantically_empty_constraint_spec(
            ordered_candidate.get("constraint_spec"),
            feature_order=feature_order,
            numeric_bound_fields=numeric_bound_fields,
        )
    else:
        semantic_buckets = _empty_semantic_buckets()
        constraint_extraction_absent = _is_semantically_empty_constraint_spec(
            None
            if not isinstance(normalized.parsed_json, dict)
            else normalized.parsed_json.get("constraint_spec"),
            feature_order=feature_order,
            numeric_bound_fields=numeric_bound_fields,
        )

    if constraint_extraction_absent:
        metadata_reasons.append(CONSTRAINT_SPEC_ABSENT)

    metadata = ParserQualityMetadata(
        reason_codes=tuple(_dedupe_strings(metadata_reasons)),
        deterministic_recovery_applied=bool(deterministic_recovery_applied),
        post_quality_schema_valid=bool(schema_validation.is_valid),
        constraint_extraction_absent=bool(constraint_extraction_absent),
        semantic_buckets=semantic_buckets,
    )
    return ParserQualityResult(
        normalized=normalized,
        schema_validation=schema_validation,
        field_provenance=_prune_field_provenance(
            candidate=normalized.parsed_json,
            field_provenance=field_provenance,
        ),
        metadata=metadata,
    )


def finalize_parser_quality_metadata(
    metadata: ParserQualityMetadata | dict[str, Any] | None,
    *,
    canonical_pass_after_quality: bool,
    repair_invoked: bool,
) -> dict[str, Any]:
    if isinstance(metadata, ParserQualityMetadata):
        return metadata.to_dict(
            canonical_pass_after_quality=canonical_pass_after_quality,
            repair_invoked=repair_invoked,
        )
    if isinstance(metadata, dict):
        payload = {
            "reason_codes": list(metadata.get("reason_codes", [])),
            "flags": {
                key: bool(dict(metadata.get("flags") or {}).get(key))
                for key in _SEMANTIC_FLAG_KEYS
            },
            "semantic_buckets": _stable_semantic_buckets(dict(metadata.get("semantic_buckets") or {})),
        }
        payload["flags"]["canonical_pass_after_quality"] = bool(canonical_pass_after_quality)
        payload["flags"]["repair_invoked"] = bool(repair_invoked)
        payload["flags"]["still_failed_after_quality"] = not bool(canonical_pass_after_quality)
        return payload
    return _empty_parser_quality_payload(
        canonical_pass_after_quality=canonical_pass_after_quality,
        repair_invoked=repair_invoked,
    )


def should_skip_parse_for_api_error(*, api_error: str | None, message_text: str) -> bool:
    return bool(api_error) and not message_text.strip()


def empty_parse_and_validation_result() -> tuple[NormalizedParseResult, ValidationResult]:
    return (
        NormalizedParseResult(
            normalized_text="",
            parsed_json=None,
            parse_error=None,
            used_brace_extraction=False,
        ),
        ValidationResult(
            is_valid=False,
            errors=(),
            unexpected_top_level_keys=(),
            unexpected_cf_fields=(),
        ),
    )


def _normalize_parser_contract_fields(
    candidate: dict[str, Any],
    *,
    benchmark_spec,
    feature_order: list[str],
) -> tuple[dict[str, Any], bool, list[str]]:
    updated = dict(candidate)
    changed = False
    reasons: list[str] = []
    field_types = dict(benchmark_spec.field_type_map)

    cf_request = updated.get("cf_request")
    if isinstance(cf_request, dict):
        normalized_cf_request, cf_changed, cf_reasons = _normalize_cf_request(
            cf_request,
            field_types=field_types,
        )
        if cf_changed:
            updated["cf_request"] = normalized_cf_request
            changed = True
        reasons.extend(cf_reasons)

    missing_fields = updated.get("missing_fields")
    if isinstance(missing_fields, list):
        normalized_missing_fields = _normalize_feature_name_list(missing_fields, dedupe=True)
        if normalized_missing_fields != missing_fields:
            updated["missing_fields"] = normalized_missing_fields
            changed = True
            reasons.append(ALIAS_KEY_NORMALIZED)

    raw_constraint_spec = updated.get("constraint_spec")
    if isinstance(raw_constraint_spec, dict):
        normalized_spec, spec_changed, spec_reasons = _normalize_constraint_spec_contract_fields(
            raw_constraint_spec,
            feature_order=feature_order,
        )
        if spec_changed:
            updated["constraint_spec"] = normalized_spec
            changed = True
        reasons.extend(spec_reasons)

    if changed and updated.get("status") not in {"conflict", "needs_clarification"}:
        updated = _sync_profile_shape(updated, feature_order=feature_order)

    return updated, changed, _dedupe_strings(reasons)


def _normalize_cf_request(
    cf_request: dict[str, Any],
    *,
    field_types: dict[str, str],
) -> tuple[dict[str, Any], bool, list[str]]:
    normalized = dict(cf_request)
    changed = False
    reasons: list[str] = []

    for original_key in list(cf_request):
        value = normalized.get(original_key)
        canonical_key = _canonical_field_name(original_key)
        effective_key = canonical_key or original_key
        coerced_value, value_changed, value_reason = _coerce_value_for_field(
            field_name=effective_key,
            value=value,
            field_types=field_types,
        )
        if value_changed:
            normalized[original_key] = coerced_value
            changed = True
            reasons.append(value_reason)
            value = coerced_value

        if not canonical_key or canonical_key == original_key:
            continue
        if canonical_key not in normalized:
            normalized[canonical_key] = value
            normalized.pop(original_key, None)
            changed = True
            reasons.append(ALIAS_KEY_NORMALIZED)
            continue
        if _values_equivalent(normalized[canonical_key], value):
            normalized.pop(original_key, None)
            changed = True
            reasons.append(ALIAS_KEY_NORMALIZED)

    return normalized, changed, _dedupe_strings(reasons)


def _normalize_constraint_spec_contract_fields(
    raw_spec: dict[str, Any],
    *,
    feature_order: list[str],
) -> tuple[dict[str, Any], bool, list[str]]:
    normalized = dict(raw_spec)
    changed = False
    reasons: list[str] = []

    for key in ("immutable", "disallowed_changes"):
        if key not in normalized:
            continue
        value = normalized.get(key)
        if isinstance(value, list):
            normalized_list = _normalize_feature_name_list(value, dedupe=True)
            if normalized_list != value:
                normalized[key] = normalized_list
                changed = True
                reasons.append(ALIAS_KEY_NORMALIZED)

    numeric_bounds = normalized.get("numeric_bounds")
    if isinstance(numeric_bounds, dict):
        updated_bounds: dict[str, Any] = {}
        bounds_changed = False
        for field_name, bound_payload in numeric_bounds.items():
            effective_field = _canonical_field_name(field_name) or field_name
            current_payload = bound_payload
            if isinstance(bound_payload, dict):
                updated_payload = dict(bound_payload)
                for bound_key in ("min", "max"):
                    if bound_key not in updated_payload:
                        continue
                    bound_value = updated_payload.get(bound_key)
                    coerced_value, value_changed, value_reason = _coerce_numeric_string(bound_value)
                    if value_changed:
                        updated_payload[bound_key] = coerced_value
                        reasons.append(value_reason)
                        bounds_changed = True
                current_payload = updated_payload
            if effective_field not in updated_bounds:
                updated_bounds[effective_field] = current_payload
            elif _values_equivalent(updated_bounds[effective_field], current_payload):
                bounds_changed = True
            else:
                updated_bounds[field_name] = current_payload
            if effective_field != field_name:
                bounds_changed = True
                reasons.append(ALIAS_KEY_NORMALIZED)
        if updated_bounds != numeric_bounds:
            normalized["numeric_bounds"] = updated_bounds
            changed = True
        changed = changed or bounds_changed

    if "max_changed_features" in normalized:
        coerced_value, value_changed, value_reason = _coerce_numeric_string(
            normalized.get("max_changed_features"),
            integer_only=True,
        )
        if value_changed:
            normalized["max_changed_features"] = coerced_value
            changed = True
            reasons.append(value_reason)

    if "prefer_fewer_changes" in normalized:
        coerced_value, value_changed, value_reason = _coerce_boolean_string(
            normalized.get("prefer_fewer_changes")
        )
        if value_changed:
            normalized["prefer_fewer_changes"] = coerced_value == 1
            changed = True
            reasons.append(value_reason)

    normalized_constraint_spec, errors = validate_and_normalize_constraint_spec(
        normalized,
        feature_order=feature_order,
    )
    if not errors and normalized_constraint_spec != normalized:
        normalized = dict(normalized_constraint_spec or {})
        changed = True

    return normalized, changed, _dedupe_strings(reasons)


def _recover_profile_fields_from_user_text(
    *,
    candidate: dict[str, Any],
    user_text: str,
    required_fields: list[str],
) -> tuple[dict[str, Any], dict[str, str], bool]:
    dense_recovery_result = recover_dense_bank_profile_candidate(
        user_input=user_text,
        candidate=candidate,
        policy=_BankPolicyAdapter(),
        required_fields=required_fields,
    )
    recovered_candidate = (
        dense_recovery_result.candidate
        if dense_recovery_result.candidate is not None
        else candidate
    )
    recovered_field_provenance = dict(dense_recovery_result.field_provenance)
    recovery_applied = bool(dense_recovery_result.recovery_applied)

    if _candidate_missing_profile_field(recovered_candidate, field_name="CCAvg"):
        explicit_recovery_result = recover_explicit_labeled_bank_fields(
            user_input=user_text,
            candidate=recovered_candidate,
            policy=_BankPolicyAdapter(),
            required_fields=required_fields,
            target_fields=("CCAvg",),
        )
        recovered_candidate = (
            explicit_recovery_result.candidate
            if explicit_recovery_result.candidate is not None
            else recovered_candidate
        )
        recovered_field_provenance = dict(explicit_recovery_result.field_provenance)
        recovery_applied = recovery_applied or bool(explicit_recovery_result.recovery_applied)

    return (
        recovered_candidate,
        recovered_field_provenance,
        recovery_applied,
    )


def _candidate_missing_profile_field(candidate: dict[str, Any] | None, *, field_name: str) -> bool:
    if not isinstance(candidate, dict):
        return True
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return True
    return field_name not in cf_request


def _recover_constraint_spec_from_user_text(
    *,
    candidate: dict[str, Any],
    user_text: str,
    feature_order: list[str],
) -> tuple[dict[str, Any], list[str], bool]:
    current_spec = candidate.get("constraint_spec") if isinstance(candidate, dict) else None
    if not _is_semantically_empty_constraint_spec(current_spec, feature_order=feature_order):
        return candidate, [], False

    recovered_spec, recovery_reasons = _extract_constraint_spec_from_user_text(
        user_text=user_text,
        feature_order=feature_order,
    )
    if not recovered_spec:
        return candidate, recovery_reasons, False

    updated = dict(candidate)
    updated["constraint_spec"] = recovered_spec
    return updated, [CONSTRAINT_SPEC_RECOVERED] + recovery_reasons, True


def _extract_constraint_spec_from_user_text(
    *,
    user_text: str,
    feature_order: list[str],
) -> tuple[dict[str, Any], list[str]]:
    text = " ".join(user_text.split())
    recovered: dict[str, Any] = {}
    reasons: list[str] = []

    disallowed_fields: list[str] = []
    found_do_not_change = False
    for field_name in feature_order:
        for alias in ordered_aliases_for_field(field_name=field_name):
            pattern = re.compile(
                rf"\b(?:do\s+not|don't|dont)\s+change\s+{build_alias_pattern(alias)}\b",
                re.IGNORECASE,
            )
            if pattern.search(text):
                disallowed_fields.append(field_name)
                found_do_not_change = True
                break
    if disallowed_fields:
        recovered["disallowed_changes"] = _ordered_feature_subset(disallowed_fields, feature_order=feature_order)
        reasons.append(CONSTRAINT_PHRASE_RECOVERED)
    elif _CONSTRAINT_VERB_RE.search(text):
        reasons.append(AMBIGUOUS_CONSTRAINT_PHRASE)

    numeric_bounds: dict[str, dict[str, float]] = {}
    bound_recovered = False
    bound_ambiguous = False
    for field_name in feature_order:
        if field_name not in {"Income", "CCAvg", "Mortgage"}:
            continue
        matches = []
        for alias in ordered_aliases_for_field(field_name=field_name):
            alias_pattern = build_alias_pattern(alias)
            patterns = (
                re.compile(
                    rf"\b{alias_pattern}\b\s+must\s+stay\s+(under|below|over|above)\s+({NUMBER_VALUE_PATTERN})\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    rf"\bkeep\s+{alias_pattern}\b\s+at\s+or\s+(below|above)\s+({NUMBER_VALUE_PATTERN})\b",
                    re.IGNORECASE,
                ),
            )
            for pattern in patterns:
                matches.extend(pattern.findall(text))
        if not matches:
            continue
        bounds: dict[str, float] = {}
        for direction, raw_value in matches:
            normalized_value = normalize_numeric_token(raw_value, integer_field=False)
            if normalized_value is None:
                bound_ambiguous = True
                continue
            bound_key = "max" if direction.lower() in {"under", "below"} else "min"
            value = float(normalized_value)
            if bound_key in bounds and bounds[bound_key] != value:
                bound_ambiguous = True
                continue
            bounds[bound_key] = value
        if bounds:
            numeric_bounds[field_name] = bounds
            bound_recovered = True
    if numeric_bounds:
        recovered["numeric_bounds"] = numeric_bounds
        reasons.append(CONSTRAINT_PHRASE_RECOVERED)
    elif _BOUND_PHRASE_RE.search(text):
        bound_ambiguous = True
    if bound_ambiguous:
        reasons.append(AMBIGUOUS_CONSTRAINT_PHRASE)

    if _MAX_CHANGE_RE.search(text):
        match = re.search(
            r"\bchange\s+at\s+most\s+(one|two|three|1|2|3)\s+things?\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            recovered["max_changed_features"] = _CHANGE_WORD_TO_INT[match.group(1).lower()]
            reasons.append(CONSTRAINT_PHRASE_RECOVERED)
        else:
            reasons.append(AMBIGUOUS_CONSTRAINT_PHRASE)

    preference_recovered = False
    if _PREFER_FEWER_RE.search(text) or _PREFER_SMALLER_RE.search(text):
        recovered["prefer_fewer_changes"] = True
        reasons.append(PREFERENCE_PHRASE_RECOVERED)
        preference_recovered = True
    if _PREFER_RE.search(text) and not preference_recovered:
        reasons.append(AMBIGUOUS_PREFERENCE_PHRASE)

    if recovered:
        normalized_constraint_spec, errors = validate_and_normalize_constraint_spec(
            recovered,
            feature_order=feature_order,
        )
        if not errors and normalized_constraint_spec is not None:
            recovered = dict(normalized_constraint_spec)
        else:
            recovered = {}

    if not recovered and found_do_not_change and CONSTRAINT_PHRASE_RECOVERED in reasons:
        reasons = [code for code in reasons if code != CONSTRAINT_PHRASE_RECOVERED]
        reasons.append(AMBIGUOUS_CONSTRAINT_PHRASE)

    return recovered, _dedupe_strings(reasons)


def _post_schema_normalize_candidate(
    candidate: dict[str, Any],
    *,
    feature_order: list[str],
) -> dict[str, Any]:
    updated = dict(candidate)
    cf_request = updated.get("cf_request")
    if isinstance(cf_request, dict):
        updated["cf_request"] = {
            field_name: cf_request[field_name]
            for field_name in feature_order
            if field_name in cf_request
        }
    missing_fields = updated.get("missing_fields")
    if isinstance(missing_fields, list):
        updated["missing_fields"] = _ordered_feature_subset(missing_fields, feature_order=feature_order)
    raw_constraint_spec = updated.get("constraint_spec")
    normalized_constraint_spec, errors = validate_and_normalize_constraint_spec(
        raw_constraint_spec,
        feature_order=feature_order,
    )
    if not errors:
        updated["constraint_spec"] = normalized_constraint_spec
        if normalized_constraint_spec in (None, {}):
            updated.pop("constraint_spec", None)
    return updated


def _derive_semantic_buckets(
    candidate: dict[str, Any],
    *,
    feature_order: list[str],
) -> dict[str, Any]:
    cf_request = candidate.get("cf_request")
    profile_facts = (
        {
            field_name: cf_request[field_name]
            for field_name in feature_order
            if isinstance(cf_request, dict) and field_name in cf_request
        }
        if isinstance(cf_request, dict)
        else {}
    )
    hard_constraints, soft_preferences = split_constraint_buckets(
        candidate.get("constraint_spec"),
        feature_order=feature_order,
    )
    return _stable_semantic_buckets(
        {
            "profile_facts": profile_facts,
            "hard_constraints": hard_constraints,
            "soft_preferences": soft_preferences,
        }
    )


def _canonical_field_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = " ".join(value.split()).strip().lower()
    if not compact:
        return None
    for field_name in BANK_REQUIRED_FIELD_ORDER:
        aliases = ordered_aliases_for_field(field_name=field_name)
        if compact in {" ".join(alias.split()).strip().lower() for alias in aliases}:
            return field_name
    return None


def _feature_order_for_benchmark(benchmark_spec) -> list[str]:
    benchmark_fields = [
        str(field.name)
        for field in getattr(benchmark_spec, "target_cf_fields", [])
        if isinstance(getattr(field, "name", None), str)
    ]
    ordered = [field_name for field_name in BANK_REQUIRED_FIELD_ORDER if field_name in benchmark_fields]
    ordered.extend(field_name for field_name in benchmark_fields if field_name not in ordered)
    return ordered


def _normalize_feature_name_list(values: list[Any], *, dedupe: bool) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = _canonical_field_name(value) or value
        if dedupe:
            key = str(normalized).strip()
            if key in seen:
                continue
            seen.add(key)
        ordered.append(str(normalized))
    return ordered


def _coerce_value_for_field(
    *,
    field_name: str,
    value: Any,
    field_types: dict[str, str],
) -> tuple[Any, bool, str]:
    field_type = field_types.get(field_name)
    if field_type == "binary":
        return _coerce_boolean_string(value)
    if field_type == "int":
        return _coerce_numeric_string(value, integer_only=True)
    if field_type == "float":
        return _coerce_numeric_string(value, integer_only=False)
    return value, False, ""


def _coerce_boolean_string(value: Any) -> tuple[Any, bool, str]:
    if not isinstance(value, str):
        return value, False, ""
    normalized = normalize_boolean_token(value)
    if normalized is None:
        return value, False, ""
    return normalized, True, BINARY_STRING_COERCED


def _coerce_numeric_string(value: Any, *, integer_only: bool = False) -> tuple[Any, bool, str]:
    if not isinstance(value, str):
        return value, False, ""
    normalized = normalize_numeric_token(value, integer_field=integer_only)
    if normalized is None:
        return value, False, ""
    return normalized, True, NUMERIC_STRING_COERCED


def _sync_profile_shape(candidate: dict[str, Any], *, feature_order: list[str]) -> dict[str, Any]:
    updated = dict(candidate)
    cf_request = updated.get("cf_request")
    if not isinstance(cf_request, dict):
        return updated
    conflicts = [item for item in updated.get("conflicts", []) if isinstance(item, str)]
    missing_fields = [field_name for field_name in feature_order if field_name not in cf_request]
    updated["missing_fields"] = missing_fields
    updated["status"] = _determine_candidate_status(
        conflicts=conflicts,
        missing_fields=missing_fields,
        prior_status=updated.get("status"),
    )
    return updated


def _determine_candidate_status(
    *,
    conflicts: list[str],
    missing_fields: list[str],
    prior_status: Any,
) -> str:
    if conflicts:
        return "conflict"
    if prior_status == "needs_clarification":
        return "needs_clarification"
    return "complete" if not missing_fields else "partial"


def _values_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _ordered_feature_subset(values: list[Any], *, feature_order: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    normalized_values = {
        str(item)
        for item in values
        if isinstance(item, str)
    }
    for field_name in feature_order:
        if field_name in normalized_values and field_name not in seen:
            ordered.append(field_name)
            seen.add(field_name)
    for item in values:
        if not isinstance(item, str):
            continue
        if item not in seen and item not in feature_order:
            ordered.append(item)
            seen.add(item)
    return ordered


def _is_semantically_empty_constraint_spec(
    raw_spec: Any,
    *,
    feature_order: list[str],
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if raw_spec is None:
        return True
    normalized_spec, errors = validate_and_normalize_constraint_spec(
        raw_spec,
        feature_order=feature_order,
        numeric_bound_fields=numeric_bound_fields,
    )
    if errors:
        return False
    if not isinstance(normalized_spec, dict):
        return True
    effective = {
        key: value
        for key, value in normalized_spec.items()
        if key in {"immutable", "disallowed_changes"} and value not in ([], None)
    }
    if isinstance(normalized_spec.get("numeric_bounds"), dict) and normalized_spec.get("numeric_bounds"):
        effective["numeric_bounds"] = normalized_spec["numeric_bounds"]
    if isinstance(normalized_spec.get("max_changed_features"), int):
        effective["max_changed_features"] = normalized_spec["max_changed_features"]
    if normalized_spec.get("prefer_fewer_changes") is True:
        effective["prefer_fewer_changes"] = True
    return not effective


def _prune_field_provenance(
    *,
    candidate: dict[str, Any] | None,
    field_provenance: dict[str, str] | None,
) -> dict[str, str]:
    if not isinstance(candidate, dict) or not isinstance(field_provenance, dict):
        return {}
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return {}
    retained_fields = set(cf_request)
    retained_fields.update(
        field_name
        for field_name, provenance in field_provenance.items()
        if provenance == FIELD_PROVENANCE_CONFLICT
    )
    return {
        str(field_name): str(value)
        for field_name, value in field_provenance.items()
        if field_name in retained_fields and isinstance(field_name, str) and isinstance(value, str)
    }


def _stable_semantic_buckets(value: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    return {
        "profile_facts": dict(payload.get("profile_facts") or {}),
        "hard_constraints": dict(payload.get("hard_constraints") or {}),
        "soft_preferences": dict(payload.get("soft_preferences") or {}),
    }


def _empty_semantic_buckets() -> dict[str, Any]:
    return _stable_semantic_buckets({})


def _empty_parser_quality_payload(
    *,
    canonical_pass_after_quality: bool = False,
    repair_invoked: bool = False,
) -> dict[str, Any]:
    return {
        "reason_codes": [],
        "flags": {
            "deterministic_recovery_applied": False,
            "post_quality_schema_valid": False,
            "canonical_pass_after_quality": bool(canonical_pass_after_quality),
            "repair_invoked": bool(repair_invoked),
            "still_failed_after_quality": not bool(canonical_pass_after_quality),
            "constraint_extraction_absent": False,
        },
        "semantic_buckets": _empty_semantic_buckets(),
    }


def _dedupe_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered
