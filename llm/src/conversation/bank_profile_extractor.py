from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import unicodedata

from llm.src.conversation.bank_completeness_guard import collect_explicit_bank_boolean_matches
from llm.src.runtime.datasets.bank.metadata import BANK_ALIASES


FIELD_PROVENANCE_CONFLICT = "conflict"
FIELD_PROVENANCE_DETERMINISTIC = "deterministic_extractor"
FIELD_PROVENANCE_PARSER = "parser"
FIELD_PROVENANCE_PARSER_AGREE = "parser_and_extractor_agree"

DENSE_STRUCTURED_BANK_FIELD_THRESHOLD = 6
SUBTHRESHOLD_STRUCTURED_BANK_FIELD_THRESHOLD = DENSE_STRUCTURED_BANK_FIELD_THRESHOLD - 1
NUMBER_VALUE_PATTERN = r"[-+]?\d+(?:\.\d+)?"
VALUE_CONNECTOR_PATTERN = (
    r"(?:=|:|is|to|of|about|around|approximately|approx\.?|should\s+be|must\s+be|will\s+be|la|lÃ )?"
)
POSITIVE_VALUE_PATTERN = r"(?:yes|true|y|1|co|cÃ³)"
NEGATIVE_VALUE_PATTERN = r"(?:no|false|n|0|khong|khÃ´ng)"
NEGATIVE_VERB_PATTERN = r"(?:do\s+not|don't|dont|not|khong|khÃ´ng|chua|chÆ°a)"
POSITIVE_VERB_PATTERN = r"(?:want|use|have|need|own|dung|dÃ¹ng|co|cÃ³|so\s+huu|sá»Ÿ\s+há»¯u)"
ARTICLE_PATTERN = r"(?:(?:a|an|mot|má»™t)\s+)?"

EVIDENCE_KIND_EXPLICIT_NUMERIC = "explicit_numeric"
EVIDENCE_KIND_EXPLICIT_BOOLEAN = "explicit_boolean"
EVIDENCE_KIND_UNIT_CONVERTED = "unit_converted"
EVIDENCE_KIND_TAXONOMY_MAPPED = "taxonomy_mapped"
EVIDENCE_KIND_CONFLICT = "conflict"
LANGUAGE_ENUM = {"en", "vi", "mixed", "unknown"}
VND_USD_FX_RATE = 25000.0
_UNIT_SIGNAL_RE = re.compile(
    r"\$|\b(?:usd|dollars?|vnd|dong|nghin|ngan|trieu|ty)\b|k\b",
    flags=re.IGNORECASE,
)
_UNIT_SUFFIX_RE = re.compile(
    r"^\s*(?:k\b|usd\b|dollars?\b|vnd\b|dong\b|nghin\b|ngan\b|trieu\b|ty\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitBankValueExtractionResult:
    values: dict[str, Any]
    conflicts: list[str]
    conflict_fields: list[str]
    labeled_fields: list[str]
    field_evidence: dict[str, dict[str, str]]


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
        extracted_field_evidence=extraction.field_evidence,
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
    extracted_field_evidence = {
        field_name: extraction.field_evidence[field_name]
        for field_name in eligible_fields
        if field_name in extraction.field_evidence
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
        extracted_field_evidence=extracted_field_evidence,
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
    field_evidence: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    conflict_fields: list[str] = []
    labeled_fields: list[str] = []
    boolean_matches = {
        field_name: [
            _ValueMatch(
                value=match.value,
                start=match.start,
                end=match.end,
                source_text=text[match.start:match.end].strip(),
                evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                normalized_from=text[match.start:match.end].strip(),
                language=detect_language_v3(text[match.start:match.end]),
            )
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
                field_name=field_name,
                aliases=aliases,
                integer_field=policy.feature_type_map.get(field_name) == "int",
                binary_field=False,
            )
        label_referenced = has_explicit_label_reference(text=text, aliases=aliases)
        if matches:
            label_referenced = True
        if label_referenced:
            labeled_fields.append(field_name)
        if not matches:
            continue

        distinct_values = []
        representative_matches: dict[Any, _ValueMatch] = {}
        for match in matches:
            if match.value not in distinct_values:
                distinct_values.append(match.value)
                representative_matches[match.value] = match
        if len(distinct_values) > 1:
            conflict_fields.append(field_name)
            conflicts.append(f"Explicit field '{field_name}' has conflicting values in the same turn.")
            continue
        value = distinct_values[0]
        extracted_values[field_name] = value
        representative = representative_matches.get(value) or matches[0]
        field_evidence[field_name] = {
            "source_text": representative.source_text,
            "evidence_kind": representative.evidence_kind,
            "normalized_from": representative.normalized_from,
            "language": representative.language if representative.language in LANGUAGE_ENUM else "unknown",
        }

    return ExplicitBankValueExtractionResult(
        values=extracted_values,
        conflicts=dedupe_strings(conflicts),
        conflict_fields=dedupe_strings(conflict_fields),
        labeled_fields=dedupe_strings(labeled_fields),
        field_evidence=field_evidence,
    )


@dataclass(frozen=True)
class _ValueMatch:
    value: Any
    start: int
    end: int
    source_text: str
    evidence_kind: str
    normalized_from: str
    language: str


def _fold_text(text: str) -> str:
    folded_chars: list[str] = []
    for char in unicodedata.normalize("NFKD", text or ""):
        if unicodedata.combining(char):
            continue
        if char in {"Ä‘", "Ä"}:
            folded_chars.append("d")
            continue
        folded_chars.append(char.lower())
    return "".join(folded_chars)


def _has_unit_suffix(*, text: str, end: int) -> bool:
    trailing_folded = _fold_text(text[end: min(len(text), end + 24)])
    return bool(_UNIT_SUFFIX_RE.match(trailing_folded))


def _contains_unit_signal(*, text: str) -> bool:
    return bool(_UNIT_SIGNAL_RE.search(_fold_text(text)))


def _number_group_has_token_suffix(*, text: str, match: re.Match[str], group_index: int) -> bool:
    _, group_end = match.span(group_index)
    if group_end < 0 or group_end >= len(text):
        return False
    next_char = text[group_end]
    if next_char == "." and group_end + 1 < len(text) and text[group_end + 1].isdigit():
        return True
    if next_char.isalpha():
        return True
    return next_char == "," and group_end + 1 < len(text) and text[group_end + 1].isdigit()


def collect_explicit_value_matches(
    *,
    text: str,
    field_name: str | None = None,
    aliases: list[str],
    integer_field: bool,
    binary_field: bool,
) -> list[_ValueMatch]:
    matches: list[_ValueMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_match(
        value: Any,
        start: int,
        end: int,
        *,
        evidence_kind: str,
        normalized_from: str,
    ) -> None:
        span = (start, end)
        if span in seen_spans:
            return
        seen_spans.add(span)
        source_text = text[start:end].strip()
        matches.append(
            _ValueMatch(
                value=value,
                start=start,
                end=end,
                source_text=source_text,
                evidence_kind=evidence_kind,
                normalized_from=(normalized_from or source_text),
                language=detect_language_v3(source_text),
            )
        )

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
                    add_match(
                        value,
                        match.start(),
                        match.end(),
                        evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                        normalized_from=match.group(1),
                    )
            for match in re.finditer(
                rf"\b({POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b\s+\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                value = normalize_boolean_token(match.group(1))
                if value is not None:
                    add_match(
                        value,
                        match.start(),
                        match.end(),
                        evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                        normalized_from=match.group(1),
                    )
            for match in re.finditer(
                rf"\b{NEGATIVE_VERB_PATTERN}\b\s+(?:want|use|have|need)\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    0,
                    match.start(),
                    match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                    normalized_from=match.group(0),
                )
            for match in re.finditer(rf"\bno\s+{ARTICLE_PATTERN}\b{alias_pattern}\b", text, flags=re.IGNORECASE):
                add_match(
                    0,
                    match.start(),
                    match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                    normalized_from=match.group(0),
                )
            for match in re.finditer(
                rf"\bwithout\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    0,
                    match.start(),
                    match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                    normalized_from=match.group(0),
                )
            for match in re.finditer(
                rf"\b{POSITIVE_VERB_PATTERN}\b\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(
                    1,
                    match.start(),
                    match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_BOOLEAN,
                    normalized_from=match.group(0),
                )
            continue

        for match in re.finditer(
            rf"\b{alias_pattern}\b\s*{VALUE_CONNECTOR_PATTERN}\s*({NUMBER_VALUE_PATTERN})\b",
            text,
            flags=re.IGNORECASE,
        ):
            if _number_group_has_token_suffix(text=text, match=match, group_index=1):
                continue
            if field_name in {"Income", "CCAvg", "Mortgage"} and _has_unit_suffix(text=text, end=match.end()):
                continue
            value = normalize_numeric_token(match.group(1), integer_field=integer_field)
            if value is not None:
                add_match(
                    value,
                    match.start(),
                    match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_NUMERIC,
                    normalized_from=match.group(1),
                )
        for match in re.finditer(
            rf"\b({NUMBER_VALUE_PATTERN})\b\s+\b{alias_pattern}\b",
            text,
            flags=re.IGNORECASE,
        ):
            if _number_group_has_token_suffix(text=text, match=match, group_index=1):
                continue
            if field_name in {"Income", "CCAvg", "Mortgage"} and _has_unit_suffix(text=text, end=match.end()):
                continue
            value = normalize_numeric_token(match.group(1), integer_field=integer_field)
            if value is not None:
                add_match(
                    value,
                    match.start(),
                    match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_NUMERIC,
                    normalized_from=match.group(1),
                )

    for match in collect_field_specific_numeric_matches(
        text=text,
        field_name=field_name,
        integer_field=integer_field,
    ):
        add_match(
            match.value,
            match.start,
            match.end,
            evidence_kind=match.evidence_kind,
            normalized_from=match.normalized_from,
        )

    return sorted(matches, key=lambda item: (item.start, item.end))


def collect_field_specific_numeric_matches(
    *,
    text: str,
    field_name: str | None,
    integer_field: bool,
) -> list[_ValueMatch]:
    if not field_name:
        return []
    matches: list[_ValueMatch] = []

    def add_from_group(
        pattern: str,
        group_index: int = 1,
        *,
        evidence_kind: str = EVIDENCE_KIND_EXPLICIT_NUMERIC,
    ) -> None:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matched_text = match.group(0)
            trailing_text = text[match.end(): min(len(text), match.end() + 24)]
            if evidence_kind == EVIDENCE_KIND_EXPLICIT_NUMERIC:
                if _number_group_has_token_suffix(text=text, match=match, group_index=group_index):
                    continue
                if _contains_unit_signal(text=matched_text) or _has_unit_suffix(text=trailing_text, end=0):
                    continue
            if evidence_kind == EVIDENCE_KIND_EXPLICIT_NUMERIC and re.search(
                r"\$|\b(?:usd|dollars?|vnd|dong|Ä‘á»“ng|trieu|triá»‡u|ty|tá»·|nghin|nghÃ¬n)\b|k\b",
                matched_text,
            ):
                continue
            if evidence_kind == EVIDENCE_KIND_EXPLICIT_NUMERIC and re.match(
                r"\s*(?:k\b|usd\b|dollars?\b|vnd\b|dong\b|Ä‘á»“ng\b|trieu\b|triá»‡u\b|ty\b|tá»·\b)",
                trailing_text,
            ):
                continue
            value = normalize_numeric_token(match.group(group_index), integer_field=integer_field)
            if value is not None:
                matches.append(
                    _build_value_match(
                        text=text,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        evidence_kind=evidence_kind,
                        normalized_from=match.group(group_index),
                    )
                )

    def add_taxonomy_match(pattern: str, value: int, normalized_from: str) -> None:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(
                _build_value_match(
                    text=text,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    evidence_kind=EVIDENCE_KIND_TAXONOMY_MAPPED,
                    normalized_from=normalized_from,
                )
            )

    if field_name == "Income":
        add_from_group(
            rf"\bI\s+(?:earn|make)\s+(?:about\s+|around\s+|approximately\s+|approx\.?\s+)?({NUMBER_VALUE_PATTERN})\b"
        )
        add_from_group(rf"\b(?:earning|salary)\s+(?:is\s+|of\s+|about\s+|around\s+)?({NUMBER_VALUE_PATTERN})\b")
        add_from_group(
            rf"\b(?:thu\s*nhap|thu\s*nháº­p|luong|lÆ°Æ¡ng)\s*(?:nam|nÄƒm)?\s*(?:la|lÃ |khoang|khoáº£ng|is|of)?\s*({NUMBER_VALUE_PATTERN})\b"
        )
    elif field_name == "Family":
        add_from_group(rf"\b(?:family|household)\s+of\s+({NUMBER_VALUE_PATTERN})\b")
        add_from_group(rf"\blive\s+in\s+a\s+(?:family|household)\s+of\s+({NUMBER_VALUE_PATTERN})\b")
        add_from_group(rf"\b(?:gia\s*dinh|gia\s*Ä‘Ã¬nh|ho\s*gia\s*dinh|há»™\s*gia\s*Ä‘Ã¬nh)\s*(?:co|cÃ³)?\s*({NUMBER_VALUE_PATTERN})\s*(?:nguoi|ngÆ°á»i)?\b")
    elif field_name == "CCAvg":
        add_from_group(
            rf"\bspend\s+(?:about\s+|around\s+|approximately\s+|approx\.?\s+)?({NUMBER_VALUE_PATTERN})\s+"
            r"(?:on|with)\s+(?:my\s+)?(?:credit\s+)?cards?\b"
        )
        add_from_group(
            rf"\b(?:credit\s+card|card)\s+(?:spend|spending|average|average\s+spending)\s+"
            rf"(?:is\s+|of\s+|about\s+|around\s+)?({NUMBER_VALUE_PATTERN})\b"
        )
        add_from_group(
            rf"\b(?:chi\s*tieu|chi\s*tiÃªu)\s*(?:the|tháº»)(?:\s*tin\s*dung|\s*tÃ­n\s*dá»¥ng)?\s*"
            rf"(?:trung\s*binh|trung\s*bÃ¬nh|moi\s*thang|má»—i\s*thÃ¡ng|hang\s*thang|hÃ ng\s*thÃ¡ng|la|lÃ |is|of)?\s*({NUMBER_VALUE_PATTERN})\b"
        )
    elif field_name == "Education":
        add_from_group(
            rf"\b(?:education|education\s+level|education\s+category|hoc\s*van|học\s*vấn|trinh\s*do\s*hoc\s*van|trình\s*độ\s*học\s*vấn)\s*"
            rf"(?:muc|mức|level|category|is|la|là|of)?\s*({NUMBER_VALUE_PATTERN})\b"
        )
        add_taxonomy_match(
            r"\b(?:undergraduate|undergrad|bachelor'?s?|college|university|cu\s*nhan|cá»­\s*nhÃ¢n|dai\s*hoc|Ä‘áº¡i\s*há»c)\b",
            1,
            "undergraduate",
        )
        add_taxonomy_match(
            r"\b(?:graduate|master'?s?|sau\s*dai\s*hoc|sau\s*Ä‘áº¡i\s*há»c|thac\s*si|tháº¡c\s*sÄ©)\b",
            2,
            "graduate",
        )
        add_taxonomy_match(
            r"\b(?:phd|doctoral|doctorate|advanced\s*professional|professional|tien\s*si|tiáº¿n\s*sÄ©|chuyen\s*mon\s*cao|chuyÃªn\s*mÃ´n\s*cao)\b",
            3,
            "advanced",
        )
        add_taxonomy_match(
            r"\b(?:c\u1eed\s*nh\u00e2n|\u0111\u1ea1i\s*h\u1ecdc)\b",
            1,
            "undergraduate",
        )
        add_taxonomy_match(
            r"\b(?:sau\s*\u0111\u1ea1i\s*h\u1ecdc|th\u1ea1c\s*s\u0129)\b",
            2,
            "graduate",
        )
        add_taxonomy_match(
            r"\b(?:ti\u1ebfn\s*s\u0129|chuy\u00ean\s*m\u00f4n\s*cao)\b",
            3,
            "advanced",
        )
    elif field_name == "Mortgage":
        add_from_group(
            rf"\bmortgage\s+(?:target|value|amount)\s+(?:is\s+|of\s+|about\s+|around\s+)?({NUMBER_VALUE_PATTERN})\b"
        )
        add_from_group(
            rf"\b(?:the\s*chap|tháº¿\s*cháº¥p|khoan\s*the\s*chap|khoáº£n\s*tháº¿\s*cháº¥p)\s*"
            rf"(?:la|lÃ |is|of|about|around)?\s*({NUMBER_VALUE_PATTERN})\b"
        )
        for match in re.finditer(
            r"\b(?:no|without)\s+(?:a\s+)?mortgage\b|\b(?:have|has)\s+no\s+mortgage\b|"
            r"\b(?:khong|khÃ´ng)\s+(?:co|cÃ³)?\s*(?:the\s*chap|tháº¿\s*cháº¥p)\b",
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(
                _build_value_match(
                    text=text,
                    value=0.0,
                    start=match.start(),
                    end=match.end(),
                    evidence_kind=EVIDENCE_KIND_EXPLICIT_NUMERIC,
                    normalized_from=match.group(0),
                )
            )

    if field_name in {"Income", "CCAvg", "Mortgage"}:
        matches.extend(collect_unit_converted_numeric_matches_v3(text=text, field_name=field_name))

    return sorted(matches, key=lambda item: (item.start, item.end))


def collect_unit_converted_numeric_matches(*, text: str, field_name: str) -> list[_ValueMatch]:
    return collect_unit_converted_numeric_matches_v3(text=text, field_name=field_name)


def collect_unit_converted_numeric_matches_v3(*, text: str, field_name: str) -> list[_ValueMatch]:
    matches: list[_ValueMatch] = []
    seen_spans: set[tuple[int, int]] = set()
    unit_pattern = (
        r"(?:\$\s*[-+]?\d[\d,]*(?:\.\d+)?\s*k?\b|"
        r"\b[-+]?\d[\d,]*(?:\.\d+)?\s*k(?:\s*(?:usd|dollars?|do|\u0111\u00f4|dola|do-la|do\s*la|\u0111\u00f4\s*la))?\b|"
        r"\b[-+]?\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?|do|\u0111\u00f4|dola|do-la|do\s*la|\u0111\u00f4\s*la)\b|"
        r"\b[-+]?\d[\d,]*(?:\.\d+)?\s*(?:nghin|ngh\u00ecn|ngan|ng\u00e0n)\s*(?:usd|do|\u0111\u00f4|dola|do-la|do\s*la|\u0111\u00f4\s*la)\b|"
        r"\b[-+]?\d[\d,]*(?:\.\d+)?\s*(?:trieu|tri\u1ec7u|ty|t\u1ef7|nghin|ngh\u00ecn|ngan|ng\u00e0n)"
        r"\s*(?:vnd|dong|\u0111\u1ed3ng)?(?:\s*/\s*(?:thang|th\u00e1ng|month))?\b)"
    )
    for match in re.finditer(unit_pattern, text, flags=re.IGNORECASE):
        span = (match.start(), match.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        raw_token = match.group(0).strip()
        if not _unit_token_relates_to_field_v3(text=text, field_name=field_name, start=match.start(), end=match.end()):
            continue
        normalized = parse_unit_token_to_dataset_value_v3(raw_token=raw_token, field_name=field_name)
        if normalized is None:
            continue
        if field_name in {"Income", "CCAvg"} and not has_required_timeframe_v3(
            text=text,
            field_name=field_name,
            token_start=match.start(),
            token_end=match.end(),
        ):
            continue
        matches.append(
            _build_value_match(
                text=text,
                value=normalized,
                start=match.start(),
                end=match.end(),
                evidence_kind=EVIDENCE_KIND_UNIT_CONVERTED,
                normalized_from=raw_token,
            )
        )
    return matches


def parse_unit_token_to_dataset_value_v3(*, raw_token: str, field_name: str) -> float | None:
    token = _fold_text(" ".join(raw_token.replace(",", "").split()))
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", token)
    if number_match is None:
        return None
    amount = float(number_match.group(0))

    has_usd_signal = bool(re.search(r"\$|\busd\b|\bdollars?\b|\bdo\b|\bdola\b|\bdo-la\b|\bdo\s*la\b", token))
    has_vnd_currency = bool(re.search(r"\b(?:vnd|dong)\b", token))
    has_vnd_magnitude = bool(re.search(r"\b(?:trieu|ty)\b", token))
    has_usd_thousand_word = bool(re.search(r"\b(?:nghin|ngan)\b", token))
    has_vnd_signal = has_vnd_currency or has_vnd_magnitude
    has_k_signal = bool(re.search(r"(?:\d\s*k\b|\dk\b)", token))
    usd_amount: float | None = None

    if has_vnd_signal:
        multiplier = 1.0
        if "ty" in token:
            multiplier = 1_000_000_000.0
        elif "trieu" in token:
            multiplier = 1_000_000.0
        elif has_vnd_currency and ("nghin" in token or "ngan" in token):
            multiplier = 1_000.0
        usd_amount = (amount * multiplier) / VND_USD_FX_RATE
    elif has_usd_signal or has_k_signal or has_usd_thousand_word:
        usd_amount = amount * 1000.0 if (has_k_signal or has_usd_thousand_word) else amount
    else:
        return None

    dataset_value = usd_amount / 1000.0
    return quantize_bank_numeric(value=dataset_value, field_name=field_name)


def has_required_timeframe_v3(*, text: str, field_name: str, token_start: int, token_end: int) -> bool:
    window = _fold_text(text[max(0, token_start - 96): min(len(text), token_end + 96)])
    if field_name == "Income":
        return bool(
            re.search(
                r"\b(?:annual|yearly|per\s+year|each\s+year|a\s+year|last\s+year|nam|hang\s*nam|moi\s*nam)\b",
                window,
            )
        )
    if field_name == "CCAvg":
        return bool(
            re.search(
                r"\b(?:monthly|per\s+month|each\s+month|a\s+month|month|thang|hang\s*thang|moi\s*thang)\b",
                window,
            )
        )
    return True


def _unit_token_relates_to_field_v3(*, text: str, field_name: str, start: int, end: int) -> bool:
    clause, clause_start = _extract_token_clause(text=text, start=start, end=end)
    clause_folded = _fold_text(clause)
    token_center = ((start + end) / 2.0) - float(clause_start)
    clause_signals = {
        "Income": bool(re.search(r"\b(?:income|salary|earn|earning|make|thu\s*nhap|luong)\b", clause_folded)),
        "CCAvg": bool(
            re.search(
                r"\b(?:spend|spending|credit\s*card|cards?|card|chi\s*tieu|ccavg)\b",
                clause_folded,
            )
        ),
        "Mortgage": bool(re.search(r"\b(?:mortgage|the\s*chap)\b", clause_folded)),
    }
    if not clause_signals.get(field_name, False):
        return False
    distances = {
        "Income": _nearest_keyword_distance(
            clause_folded,
            token_center=token_center,
            keywords=(r"income", r"salary", r"earn", r"earning", r"make", r"thu\s*nhap", r"luong"),
        ),
        "CCAvg": _nearest_keyword_distance(
            clause_folded,
            token_center=token_center,
            keywords=(r"spend", r"spending", r"credit\s*card", r"cards?", r"card", r"chi\s*tieu", r"ccavg"),
        ),
        "Mortgage": _nearest_keyword_distance(
            clause_folded,
            token_center=token_center,
            keywords=(r"mortgage", r"the\s*chap"),
        ),
    }
    target_distance = distances.get(field_name)
    if target_distance is None:
        return False
    other_distances = [value for name, value in distances.items() if name != field_name and value is not None]
    return target_distance <= (min(other_distances) if other_distances else target_distance)


def _nearest_keyword_distance(
    text: str,
    *,
    token_center: float,
    keywords: tuple[str, ...],
) -> float | None:
    nearest: float | None = None
    for keyword in keywords:
        for match in re.finditer(rf"\b{keyword}\b", text, flags=re.IGNORECASE):
            center = (match.start() + match.end()) / 2.0
            distance = abs(center - token_center)
            if nearest is None or distance < nearest:
                nearest = distance
    return nearest


def _extract_token_clause(*, text: str, start: int, end: int) -> tuple[str, int]:
    left = -1
    right = len(text)
    separator_chars = {";", ".", "!", "?", ","}
    for index in range(start - 1, -1, -1):
        char = text[index]
        if char not in separator_chars:
            continue
        if char == "," and index > 0 and index + 1 < len(text):
            if text[index - 1].isdigit() and text[index + 1].isdigit():
                continue
        left = index
        break
    for index in range(end, len(text)):
        char = text[index]
        if char not in separator_chars:
            continue
        if char == "," and index > 0 and index + 1 < len(text):
            if text[index - 1].isdigit() and text[index + 1].isdigit():
                continue
        right = index
        break
    return text[left + 1:right], left + 1


def quantize_bank_numeric(*, value: float, field_name: str) -> float:
    if field_name in {"Income", "Mortgage"}:
        return float(round(value))
    if field_name == "CCAvg":
        return round(float(value), 1)
    return float(value)


def _build_value_match(
    *,
    text: str,
    value: Any,
    start: int,
    end: int,
    evidence_kind: str,
    normalized_from: str,
) -> _ValueMatch:
    source_text = text[start:end].strip()
    return _ValueMatch(
        value=value,
        start=start,
        end=end,
        source_text=source_text,
        evidence_kind=evidence_kind,
        normalized_from=normalized_from,
        language=detect_language_v3(source_text),
    )


def detect_language(text: str) -> str:
    lowered = str(text or "").lower()
    has_vi_signal = bool(
        re.search(
            r"[Ã Ã¡áº¡áº£Ã£Ã¢áº§áº¥áº­áº©áº«Äƒáº±áº¯áº·áº³áºµÃ¨Ã©áº¹áº»áº½Ãªá»áº¿á»‡á»ƒá»…Ã¬Ã­á»‹á»‰Ä©Ã²Ã³á»á»ÃµÃ´á»“á»‘á»™á»•á»—Æ¡á»á»›á»£á»Ÿá»¡Ã¹Ãºá»¥á»§Å©Æ°á»«á»©á»±á»­á»¯á»³Ã½á»µá»·á»¹Ä‘]"
            r"|\b(?:thu\s*nhap|thu\s*nháº­p|luong|lÆ°Æ¡ng|thang|thÃ¡ng|nam|nÄƒm|khong|khÃ´ng|co|cÃ³|gia\s*dinh|gia\s*Ä‘Ã¬nh)\b",
            lowered,
        )
    )
    has_en_signal = bool(
        re.search(
            r"\b(?:income|salary|family|household|education|mortgage|credit|card|online|account|monthly|annual)\b",
            lowered,
        )
    )
    if has_vi_signal and has_en_signal:
        return "mixed"
    if has_vi_signal:
        return "vi"
    if has_en_signal:
        return "en"
    return "unknown"


def detect_language_v3(text: str) -> str:
    lowered = _fold_text(str(text or ""))
    has_vi_signal = bool(
        re.search(
            r"\b(?:thu\s*nhap|luong|thang|nam|khong|co|gia\s*dinh|the\s*chap|chi\s*tieu|thac\s*si|tien\s*si|dai\s*hoc|vnd|dong|trieu|ty|nghin|ngan)\b",
            lowered,
        )
    )
    has_en_signal = bool(
        re.search(
            r"\b(?:income|salary|family|household|education|mortgage|credit|card|online|account|monthly|annual)\b",
            lowered,
        )
    )
    if has_vi_signal and has_en_signal:
        return "mixed"
    if has_vi_signal:
        return "vi"
    if has_en_signal:
        return "en"
    return "unknown"


def merge_bank_candidate_with_explicit_values(
    *,
    candidate: dict[str, Any],
    extracted_values: dict[str, Any],
    extracted_field_evidence: dict[str, dict[str, str]] | None = None,
    extracted_conflicts: list[str],
    conflict_fields: list[str],
    required_fields: list[str],
    base_field_provenance: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    merged_candidate = dict(candidate)
    parser_request = candidate.get("cf_request") if isinstance(candidate.get("cf_request"), dict) else {}
    merged_request = dict(parser_request)
    field_provenance = dict(base_field_provenance or {})
    parser_field_evidence = (
        dict(candidate.get("field_evidence"))
        if isinstance(candidate.get("field_evidence"), dict)
        else {}
    )
    merged_field_evidence: dict[str, dict[str, str]] = {
        str(field_name): dict(value)
        for field_name, value in parser_field_evidence.items()
        if isinstance(field_name, str) and isinstance(value, dict)
    }
    extracted_field_evidence = dict(extracted_field_evidence or {})
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
            merged_field_evidence[field_name] = {
                "source_text": field_name,
                "evidence_kind": EVIDENCE_KIND_CONFLICT,
                "normalized_from": "",
                "language": "unknown",
            }
            if not parser_has_value:
                merged_request.pop(field_name, None)
            continue
        if parser_has_value and not extracted_has_value:
            field_provenance.setdefault(field_name, FIELD_PROVENANCE_PARSER)
            continue
        if not parser_has_value and extracted_has_value:
            merged_request[field_name] = extracted_values[field_name]
            field_provenance[field_name] = FIELD_PROVENANCE_DETERMINISTIC
            if field_name in extracted_field_evidence:
                merged_field_evidence[field_name] = dict(extracted_field_evidence[field_name])
            continue
        if parser_has_value and extracted_has_value:
            if parser_request[field_name] == extracted_values[field_name]:
                field_provenance[field_name] = FIELD_PROVENANCE_PARSER_AGREE
                if field_name in extracted_field_evidence and field_name not in merged_field_evidence:
                    merged_field_evidence[field_name] = dict(extracted_field_evidence[field_name])
                continue
            merged_request[field_name] = extracted_values[field_name]
            field_provenance[field_name] = FIELD_PROVENANCE_DETERMINISTIC
            if field_name in extracted_field_evidence:
                merged_field_evidence[field_name] = dict(extracted_field_evidence[field_name])
            continue

    missing_fields = [field for field in required_fields if field not in merged_request]
    merged_candidate["cf_request"] = merged_request
    merged_candidate["field_evidence"] = {
        field_name: value
        for field_name, value in merged_field_evidence.items()
        if field_name in merged_request or field_name in set(conflict_fields)
    }
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
