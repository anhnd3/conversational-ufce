from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


BANK_REQUIRED_FIELD_ORDER = (
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "SecuritiesAccount",
    "CDAccount",
    "Online",
    "CreditCard",
)

BANK_REQUIRED_BOOLEAN_FIELDS = (
    "SecuritiesAccount",
    "CDAccount",
    "Online",
    "CreditCard",
)

FIELD_ALIAS_PATTERNS: dict[str, tuple[str, ...]] = {
    "SecuritiesAccount": (
        "securities account",
        "security account",
        "securitiesaccount",
        "investment account",
    ),
    "CDAccount": (
        "cd account",
        "cdaccount",
        "certificate of deposit",
    ),
    "Online": (
        "online",
        "online banking",
    ),
    "CreditCard": (
        "credit card",
        "creditcard",
        "bank credit card",
    ),
}

POSITIVE_VALUE_PATTERN = r"(?:yes|true|1)"
NEGATIVE_VALUE_PATTERN = r"(?:no|false|0)"
NEGATIVE_VERB_PATTERN = r"(?:do\s+not|don't|dont|not)"
POSITIVE_VERB_PATTERN = r"(?:want|use|have|need|own)"
ARTICLE_PATTERN = r"(?:(?:a|an)\s+)?"
SEGMENT_BOUNDARY_PATTERN = re.compile(r"[,;.!?]|\bbut\b|\bhowever\b|\bthough\b|\bexcept\b", re.IGNORECASE)


@dataclass(frozen=True)
class BankBooleanFieldMatch:
    field_name: str
    value: int
    start: int
    end: int


@dataclass(frozen=True)
class BankCompletenessGuardResult:
    candidate: dict[str, Any] | None
    downgraded_fields: list[str]
    explicit_fields: list[str]


def apply_bank_boolean_completeness_guard(
    *,
    user_text: str,
    candidate: dict[str, Any] | None,
) -> BankCompletenessGuardResult:
    if not isinstance(candidate, dict):
        return BankCompletenessGuardResult(candidate=candidate, downgraded_fields=[], explicit_fields=[])

    if candidate.get("status") != "complete":
        return BankCompletenessGuardResult(candidate=candidate, downgraded_fields=[], explicit_fields=[])

    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return BankCompletenessGuardResult(candidate=candidate, downgraded_fields=[], explicit_fields=[])

    explicit_fields = detect_explicit_bank_boolean_fields(user_text)
    downgraded_fields = [
        field
        for field in BANK_REQUIRED_BOOLEAN_FIELDS
        if field in cf_request and field not in explicit_fields
    ]
    if not downgraded_fields:
        return BankCompletenessGuardResult(
            candidate=candidate,
            downgraded_fields=[],
            explicit_fields=list(explicit_fields),
        )

    adjusted_cf_request = {
        field_name: value
        for field_name, value in cf_request.items()
        if field_name not in downgraded_fields
    }
    existing_missing = [
        value
        for value in candidate.get("missing_fields", [])
        if isinstance(value, str)
    ]
    missing_union = set(existing_missing) | set(downgraded_fields)
    ordered_missing = [field for field in BANK_REQUIRED_FIELD_ORDER if field in missing_union]

    adjusted_candidate = dict(candidate)
    adjusted_candidate["status"] = "partial"
    adjusted_candidate["cf_request"] = adjusted_cf_request
    adjusted_candidate["missing_fields"] = ordered_missing

    return BankCompletenessGuardResult(
        candidate=adjusted_candidate,
        downgraded_fields=list(downgraded_fields),
        explicit_fields=list(explicit_fields),
    )


def detect_explicit_bank_boolean_fields(user_text: str) -> list[str]:
    matches = collect_explicit_bank_boolean_matches(user_text)
    explicit_fields = {match.field_name for match in matches}
    return [field_name for field_name in BANK_REQUIRED_BOOLEAN_FIELDS if field_name in explicit_fields]


def has_explicit_boolean_evidence(*, text: str, field_name: str) -> bool:
    return bool(
        collect_explicit_bank_boolean_matches(
            text,
            target_fields=[field_name],
        )
    )


def collect_explicit_bank_boolean_matches(
    user_text: str,
    *,
    target_fields: list[str] | tuple[str, ...] | None = None,
    field_aliases: dict[str, list[str] | tuple[str, ...]] | None = None,
) -> list[BankBooleanFieldMatch]:
    text = " ".join(user_text.split())
    active_fields = [
        field_name
        for field_name in (target_fields or BANK_REQUIRED_BOOLEAN_FIELDS)
        if field_name in BANK_REQUIRED_BOOLEAN_FIELDS
    ]
    active_aliases = {
        field_name: _normalize_aliases(
            (field_aliases or {}).get(field_name) or FIELD_ALIAS_PATTERNS[field_name]
        )
        for field_name in active_fields
    }
    matches: list[BankBooleanFieldMatch] = []
    seen: set[tuple[str, int, int, int]] = set()

    def add_match(field_name: str, value: int, start: int, end: int) -> None:
        key = (field_name, value, start, end)
        if key in seen:
            return
        seen.add(key)
        matches.append(BankBooleanFieldMatch(field_name=field_name, value=value, start=start, end=end))

    for field_name in active_fields:
        for alias in active_aliases[field_name]:
            alias_pattern = build_alias_pattern(alias)
            for match in re.finditer(
                rf"\b{alias_pattern}\b\s*(?:=|:|is)?\s*\b({POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b",
                text,
                flags=re.IGNORECASE,
            ):
                value = normalize_boolean_token(match.group(1))
                if value is not None:
                    add_match(field_name, value, match.start(), match.end())
            for match in re.finditer(
                rf"\b({POSITIVE_VALUE_PATTERN}|{NEGATIVE_VALUE_PATTERN})\b\s+\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                value = normalize_boolean_token(match.group(1))
                if value is not None:
                    add_match(field_name, value, match.start(), match.end())
            for match in re.finditer(
                rf"\b{NEGATIVE_VERB_PATTERN}\b\s+{POSITIVE_VERB_PATTERN}\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(field_name, 0, match.start(), match.end())
            for match in re.finditer(rf"\bno\s+{ARTICLE_PATTERN}\b{alias_pattern}\b", text, flags=re.IGNORECASE):
                add_match(field_name, 0, match.start(), match.end())
            for match in re.finditer(
                rf"\bwithout\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                add_match(field_name, 0, match.start(), match.end())
            for match in re.finditer(
                rf"\b{POSITIVE_VERB_PATTERN}\b\s+{ARTICLE_PATTERN}\b{alias_pattern}\b",
                text,
                flags=re.IGNORECASE,
            ):
                if _is_immediately_negated(text, match.start()):
                    continue
                add_match(field_name, 1, match.start(), match.end())

    for negative_phrase in re.finditer(
        rf"\b{NEGATIVE_VERB_PATTERN}\b\s+{POSITIVE_VERB_PATTERN}\b",
        text,
        flags=re.IGNORECASE,
    ):
        segment_start = negative_phrase.end()
        segment_end = _find_segment_end(text, segment_start)
        segment_text = text[segment_start:segment_end]
        for field_name in active_fields:
            for alias in active_aliases[field_name]:
                alias_pattern = build_alias_pattern(alias)
                for alias_match in re.finditer(
                    rf"{ARTICLE_PATTERN}\b{alias_pattern}\b",
                    segment_text,
                    flags=re.IGNORECASE,
                ):
                    add_match(
                        field_name,
                        0,
                        segment_start + alias_match.start(),
                        segment_start + alias_match.end(),
                    )

    return sorted(matches, key=lambda item: (item.start, item.end, item.field_name, item.value))


def build_alias_pattern(alias: str) -> str:
    pieces = re.split(r"\s+", alias.strip())
    return r"\s+".join(re.escape(piece) for piece in pieces if piece)


def normalize_boolean_token(value: str) -> int | None:
    token = value.strip().lower()
    if token in {"yes", "true", "1"}:
        return 1
    if token in {"no", "false", "0"}:
        return 0
    return None


def _normalize_aliases(value: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in value:
        clean = " ".join(str(alias).split()).strip()
        if not clean:
            continue
        lowered = clean.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(clean)
    return ordered


def _find_segment_end(text: str, segment_start: int) -> int:
    match = SEGMENT_BOUNDARY_PATTERN.search(text, segment_start)
    if match is None:
        return len(text)
    return match.start()


def _is_immediately_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 24):start]
    return re.search(rf"(?:{NEGATIVE_VERB_PATTERN})\s+$", prefix, flags=re.IGNORECASE) is not None
