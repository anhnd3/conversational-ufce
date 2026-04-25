from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from llm.src.conversation.bank_completeness_guard import collect_explicit_bank_boolean_matches
from llm.src.runtime.datasets.bank.metadata import BANK_ALIASES


FIELD_PROVENANCE_CONFLICT = "conflict"
FIELD_PROVENANCE_DETERMINISTIC = "deterministic_extractor"
FIELD_PROVENANCE_PARSER = "parser"
FIELD_PROVENANCE_PARSER_AGREE = "parser_and_extractor_agree"

DENSE_STRUCTURED_BANK_FIELD_THRESHOLD = 6
SUBTHRESHOLD_STRUCTURED_BANK_FIELD_THRESHOLD = DENSE_STRUCTURED_BANK_FIELD_THRESHOLD - 1
NUMBER_VALUE_PATTERN = r"[-+]?\d+(?:\.\d+)?"
VALUE_CONNECTOR_PATTERN = r"(?:=|:|is)?"
POSITIVE_VALUE_PATTERN = r"(?:yes|true|y|1)"
NEGATIVE_VALUE_PATTERN = r"(?:no|false|n|0)"
NEGATIVE_VERB_PATTERN = r"(?:do\s+not|don't|dont|not)"
POSITIVE_VERB_PATTERN = r"(?:want|use|have|need)"
ARTICLE_PATTERN = r"(?:(?:a|an)\s+)?"


@dataclass(frozen=True)
class ExplicitBankValueExtractionResult:
    values: dict[str, Any]
    conflicts: list[str]
    conflict_fields: list[str]
    labeled_fields: list[str]


@dataclass(frozen=True)
class DenseBankProfileRecoveryResult:
    candidate: dict[str, Any] | None
    field_provenance: dict[str, str]
    recovery_applied: bool
    dense_profile_detected: bool


@dataclass(frozen=True)
class ExplicitLabeledBankFieldRecoveryResult:
    candidate: dict[str, Any] | None
    field_provenance: dict[str, str]
    recovery_applied: bool
    recovered_fields: tuple[str, ...]


def recover_dense_bank_profile_candidate(
    *,
    user_input: str,
    candidate: dict[str, Any] | None,
    policy,
    required_fields: list[str],
) -> DenseBankProfileRecoveryResult:
    field_provenance = build_parser_field_provenance(candidate)
    if not isinstance(candidate, dict):
        return DenseBankProfileRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            dense_profile_detected=False,
        )
    if policy.dataset_name != "bank":
        return DenseBankProfileRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            dense_profile_detected=False,
        )

    extraction = extract_explicit_bank_values(
        user_input=user_input,
        policy=policy,
        target_fields=required_fields,
    )
    is_dense = looks_like_dense_structured_bank_profile(user_input=user_input, extraction=extraction)
    if not is_dense:
        return DenseBankProfileRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            dense_profile_detected=False,
        )

    merged_candidate, merged_field_provenance = merge_bank_candidate_with_explicit_values(
        candidate=candidate,
        extracted_values=extraction.values,
        extracted_conflicts=extraction.conflicts,
        conflict_fields=extraction.conflict_fields,
        required_fields=required_fields,
        base_field_provenance=field_provenance,
    )
    return DenseBankProfileRecoveryResult(
        candidate=merged_candidate,
        field_provenance=merged_field_provenance,
        recovery_applied=merged_candidate != candidate,
        dense_profile_detected=True,
    )


def recover_explicit_labeled_bank_fields(
    *,
    user_input: str,
    candidate: dict[str, Any] | None,
    policy,
    required_fields: list[str],
    target_fields: tuple[str, ...] = ("CCAvg",),
) -> ExplicitLabeledBankFieldRecoveryResult:
    field_provenance = build_parser_field_provenance(candidate)
    if not isinstance(candidate, dict):
        return ExplicitLabeledBankFieldRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            recovered_fields=(),
        )
    if policy.dataset_name != "bank":
        return ExplicitLabeledBankFieldRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            recovered_fields=(),
        )

    normalized_target_fields = tuple(
        field_name
        for field_name in required_fields
        if field_name in set(target_fields)
    )
    if not normalized_target_fields:
        return ExplicitLabeledBankFieldRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            recovered_fields=(),
        )

    extraction = extract_explicit_bank_values(
        user_input=user_input,
        policy=policy,
        target_fields=required_fields,
    )
    if not looks_like_structured_subthreshold_bank_profile(
        user_input=user_input,
        extraction=extraction,
    ):
        return ExplicitLabeledBankFieldRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            recovered_fields=(),
        )

    eligible_fields = tuple(
        field_name
        for field_name in normalized_target_fields
        if field_name in extraction.labeled_fields
        and (field_name in extraction.values or field_name in extraction.conflict_fields)
    )
    if not eligible_fields:
        return ExplicitLabeledBankFieldRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            recovered_fields=(),
        )

    extracted_values = {
        field_name: extraction.values[field_name]
        for field_name in eligible_fields
        if field_name in extraction.values
    }
    conflict_fields = [
        field_name
        for field_name in extraction.conflict_fields
        if field_name in eligible_fields
    ]
    extracted_conflicts = [
        conflict
        for conflict in extraction.conflicts
        if any(f"'{field_name}'" in conflict for field_name in eligible_fields)
    ]
    if not extracted_values and not conflict_fields:
        return ExplicitLabeledBankFieldRecoveryResult(
            candidate=candidate,
            field_provenance=field_provenance,
            recovery_applied=False,
            recovered_fields=(),
        )

    merged_candidate, merged_field_provenance = merge_bank_candidate_with_explicit_values(
        candidate=candidate,
        extracted_values=extracted_values,
        extracted_conflicts=extracted_conflicts,
        conflict_fields=conflict_fields,
        required_fields=required_fields,
        base_field_provenance=field_provenance,
    )
    recovered_fields = tuple(
        field_name for field_name in normalized_target_fields if field_name in extracted_values
    )
    return ExplicitLabeledBankFieldRecoveryResult(
        candidate=merged_candidate,
        field_provenance=merged_field_provenance,
        recovery_applied=merged_candidate != candidate,
        recovered_fields=recovered_fields,
    )


def extract_explicit_bank_values(
    *,
    user_input: str,
    policy,
    target_fields: list[str],
) -> ExplicitBankValueExtractionResult:
    text = " ".join(user_input.split())
    extracted_values: dict[str, Any] = {}
    conflicts: list[str] = []
    conflict_fields: list[str] = []
    labeled_fields: list[str] = []
    boolean_matches = {
        field_name: [
            _ValueMatch(value=match.value, start=match.start, end=match.end)
            for match in collect_explicit_bank_boolean_matches(
                text,
                target_fields=[field_name],
                field_aliases={
                    field_name: ordered_aliases_for_field(field_name=field_name, policy=policy),
                },
            )
        ]
        for field_name in target_fields
        if policy.feature_type_map.get(field_name) == "binary"
    }

    for field_name in target_fields:
        aliases = ordered_aliases_for_field(field_name=field_name, policy=policy)
        if policy.feature_type_map.get(field_name) == "binary":
            matches = list(boolean_matches.get(field_name, []))
        else:
            matches = collect_explicit_value_matches(
                text=text,
                aliases=aliases,
                integer_field=policy.feature_type_map.get(field_name) == "int",
                binary_field=False,
            )
        if has_explicit_label_reference(text=text, aliases=aliases):
            labeled_fields.append(field_name)
        if not matches:
            continue

        distinct_values = []
        for match in matches:
            if match.value not in distinct_values:
                distinct_values.append(match.value)
        if len(distinct_values) > 1:
            conflict_fields.append(field_name)
            conflicts.append(f"Explicit field '{field_name}' has conflicting values in the same turn.")
            continue
        extracted_values[field_name] = distinct_values[0]

    return ExplicitBankValueExtractionResult(
        values=extracted_values,
        conflicts=dedupe_strings(conflicts),
        conflict_fields=dedupe_strings(conflict_fields),
        labeled_fields=dedupe_strings(labeled_fields),
    )


@dataclass(frozen=True)
class _ValueMatch:
    value: Any
    start: int
    end: int


def collect_explicit_value_matches(
    *,
    text: str,
    aliases: list[str],
    integer_field: bool,
    binary_field: bool,
) -> list[_ValueMatch]:
    matches: list[_ValueMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_match(value: Any, start: int, end: int) -> None:
        span = (start, end)
        if span in seen_spans:
            return
        seen_spans.add(span)
        matches.append(_ValueMatch(value=value, start=start, end=end))

    for alias in aliases:
        alias_pattern = build_alias_pattern(alias)
        if binary_field:
            for match in re.finditer(
                rf"\b{alias_pattern}\b\s*{VALUE_CONNECTOR_PATTERN}\s*\b({POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b",
                text,
                flags=re.IGNORECASE,
            ):
                value = normalize_boolean_token(match.group(1))
                if value is not None:
                    add_match(value, match.start(), match.end())
            for match in re.finditer(
                rf"\b({POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b\s+\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                value = normalize_boolean_token(match.group(1))
                if value is not None:
                    add_match(value, match.start(), match.end())
            for match in re.finditer(
                rf"\b{NEGATIVE_VERB_PATTERN}\b\s+(?:want|use|have|need)\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(0, match.start(), match.end())
            for match in re.finditer(rf"\bno\s+{ARTICLE_PATTERN}\b{alias_pattern}\b", text, flags=re.IGNORECASE):
                add_match(0, match.start(), match.end())
            for match in re.finditer(
                rf"\bwithout\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(0, match.start(), match.end())
            for match in re.finditer(
                rf"\b{POSITIVE_VERB_PATTERN}\b\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(1, match.start(), match.end())
            continue

        for match in re.finditer(
            rf"\b{alias_pattern}\b\s*{VALUE_CONNECTOR_PATTERN}\s*({NUMBER_VALUE_PATTERN})\b",
            text,
            flags=re.IGNORECASE,
        ):
            value = normalize_numeric_token(match.group(1), integer_field=integer_field)
            if value is not None:
                add_match(value, match.start(), match.end())
        for match in re.finditer(
            rf"\b({NUMBER_VALUE_PATTERN})\b\s+\b{alias_pattern}\b",
            text,
            flags=re.IGNORECASE,
        ):
            value = normalize_numeric_token(match.group(1), integer_field=integer_field)
            if value is not None:
                add_match(value, match.start(), match.end())

    return sorted(matches, key=lambda item: (item.start, item.end))


def merge_bank_candidate_with_explicit_values(
    *,
    candidate: dict[str, Any],
    extracted_values: dict[str, Any],
    extracted_conflicts: list[str],
    conflict_fields: list[str],
    required_fields: list[str],
    base_field_provenance: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    merged_candidate = dict(candidate)
    parser_request = candidate.get("cf_request") if isinstance(candidate.get("cf_request"), dict) else {}
    merged_request = dict(parser_request)
    field_provenance = dict(base_field_provenance or {})
    conflicts = dedupe_strings(
        [
            str(item)
            for item in candidate.get("conflicts", [])
            if isinstance(item, str)
        ]
        + list(extracted_conflicts)
    )

    for field_name in required_fields:
        parser_has_value = field_name in parser_request
        extracted_has_value = field_name in extracted_values
        if field_name in conflict_fields:
            field_provenance[field_name] = FIELD_PROVENANCE_CONFLICT
            if not parser_has_value:
                merged_request.pop(field_name, None)
            continue
        if parser_has_value and not extracted_has_value:
            field_provenance.setdefault(field_name, FIELD_PROVENANCE_PARSER)
            continue
        if not parser_has_value and extracted_has_value:
            merged_request[field_name] = extracted_values[field_name]
            field_provenance[field_name] = FIELD_PROVENANCE_DETERMINISTIC
            continue
        if parser_has_value and extracted_has_value:
            if parser_request[field_name] == extracted_values[field_name]:
                field_provenance[field_name] = FIELD_PROVENANCE_PARSER_AGREE
                continue
            conflicts.append(
                f"Explicit field '{field_name}' disagrees between parser output and deterministic extraction."
            )
            field_provenance[field_name] = FIELD_PROVENANCE_CONFLICT

    missing_fields = [field for field in required_fields if field not in merged_request]
    merged_candidate["cf_request"] = merged_request
    merged_candidate["missing_fields"] = missing_fields
    merged_candidate["conflicts"] = dedupe_strings(conflicts)
    merged_candidate["status"] = determine_candidate_status(
        conflicts=merged_candidate["conflicts"],
        missing_fields=missing_fields,
    )
    return merged_candidate, field_provenance


def determine_candidate_status(*, conflicts: list[str], missing_fields: list[str]) -> str:
    if conflicts:
        return "conflict"
    return "complete" if not missing_fields else "partial"


def looks_like_dense_structured_bank_profile(
    *,
    user_input: str,
    extraction: ExplicitBankValueExtractionResult,
) -> bool:
    text = " ".join(user_input.split())
    if len(extraction.labeled_fields) < DENSE_STRUCTURED_BANK_FIELD_THRESHOLD:
        return False
    return "," in text or count_label_value_signals(text) >= DENSE_STRUCTURED_BANK_FIELD_THRESHOLD


def looks_like_structured_subthreshold_bank_profile(
    *,
    user_input: str,
    extraction: ExplicitBankValueExtractionResult,
) -> bool:
    labeled_field_count = len(extraction.labeled_fields)
    if labeled_field_count >= DENSE_STRUCTURED_BANK_FIELD_THRESHOLD:
        return False
    if labeled_field_count < SUBTHRESHOLD_STRUCTURED_BANK_FIELD_THRESHOLD:
        return False
    text = " ".join(user_input.split())
    return "," in text or count_label_value_signals(text) >= SUBTHRESHOLD_STRUCTURED_BANK_FIELD_THRESHOLD


def count_label_value_signals(text: str) -> int:
    signal_count = 0
    for aliases in BANK_ALIASES.values():
        if any(
            re.search(
                rf"\b{build_alias_pattern(alias)}\b\s*{VALUE_CONNECTOR_PATTERN}\s*(?:{NUMBER_VALUE_PATTERN}|{POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b",
                text,
                flags=re.IGNORECASE,
            )
            for alias in aliases
        ):
            signal_count += 1
    return signal_count


def has_explicit_label_reference(*, text: str, aliases: list[str]) -> bool:
    return any(
        re.search(
            rf"\b{build_alias_pattern(alias)}\b\s*{VALUE_CONNECTOR_PATTERN}\s*(?:{NUMBER_VALUE_PATTERN}|{POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b",
            text,
            flags=re.IGNORECASE,
        )
        for alias in aliases
    )


def ordered_aliases_for_field(*, field_name: str, policy=None) -> list[str]:
    if policy is not None:
        aliases = list(policy.conversation_aliases.get(field_name) or [])
    else:
        aliases = list(BANK_ALIASES.get(field_name) or [])
    if field_name not in aliases:
        aliases.insert(0, field_name)
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in sorted(aliases, key=lambda value: (-len(str(value)), str(value).lower())):
        clean = " ".join(str(alias).split()).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(clean)
    return ordered


def build_alias_pattern(alias: str) -> str:
    pieces = re.split(r"\s+", alias.strip())
    return r"\s+".join(re.escape(piece) for piece in pieces if piece)


def build_parser_field_provenance(candidate: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(candidate, dict):
        return {}
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return {}
    return {
        str(field_name): FIELD_PROVENANCE_PARSER
        for field_name in cf_request
        if isinstance(field_name, str)
    }


def normalize_boolean_token(value: str) -> int | None:
    token = value.strip().lower()
    if token in {"yes", "true", "y", "1"}:
        return 1
    if token in {"no", "false", "n", "0"}:
        return 0
    return None


def normalize_numeric_token(value: str, *, integer_field: bool) -> int | float | None:
    try:
        numeric_value = float(value)
    except ValueError:
        return None
    if integer_field:
        if numeric_value.is_integer():
            return int(numeric_value)
        return None
    return float(numeric_value)


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped
