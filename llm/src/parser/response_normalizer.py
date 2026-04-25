from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(?P<body>.*)\n```\s*$", re.DOTALL)


@dataclass(frozen=True)
class NormalizedParseResult:
    normalized_text: str
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    used_brace_extraction: bool


def normalize_and_parse(message_text: str) -> NormalizedParseResult:
    normalized_text = strip_markdown_fences(message_text.strip())
    parsed_json, parse_error = parse_json_object(normalized_text)
    used_brace_extraction = False

    if parsed_json is None:
        extracted = extract_first_json_object(normalized_text)
        if extracted and extracted != normalized_text:
            fallback_json, fallback_error = parse_json_object(extracted)
            if fallback_json is not None:
                normalized_text = extracted
                parsed_json = fallback_json
                parse_error = None
                used_brace_extraction = True
            elif parse_error is None:
                parse_error = fallback_error

    return NormalizedParseResult(
        normalized_text=normalized_text,
        parsed_json=parsed_json,
        parse_error=parse_error,
        used_brace_extraction=used_brace_extraction,
    )


def strip_markdown_fences(text: str) -> str:
    match = FENCE_RE.match(text)
    if match:
        return match.group("body").strip()
    return text.strip()


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
    if not isinstance(value, dict):
        return None, "Top-level JSON value must be an object."
    return value, None


def extract_first_json_object(text: str) -> str | None:
    start_index = text.find("{")
    if start_index == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    return None
