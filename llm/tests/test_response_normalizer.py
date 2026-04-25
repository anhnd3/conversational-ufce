from __future__ import annotations

from llm.src.parser.response_normalizer import (
    extract_first_json_object,
    normalize_and_parse,
    strip_markdown_fences,
)


def test_strip_markdown_fences_removes_outer_block():
    text = """```json
{"task":"extract_cf_request"}
```"""

    assert strip_markdown_fences(text) == '{"task":"extract_cf_request"}'


def test_normalize_and_parse_extracts_json_from_wrapped_text():
    text = 'Here is the result:\n{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}\nThanks.'

    result = normalize_and_parse(text)

    assert result.used_brace_extraction is True
    assert result.parsed_json["status"] == "partial"
    assert result.parse_error is None


def test_normalize_and_parse_reports_malformed_json():
    result = normalize_and_parse('{"task": "extract_cf_request", }')

    assert result.parsed_json is None
    assert result.parse_error is not None


def test_extract_first_json_object_balances_nested_braces():
    text = 'prefix {"a":{"b":1},"c":[1,2]} suffix'

    assert extract_first_json_object(text) == '{"a":{"b":1},"c":[1,2]}'


def test_normalize_and_parse_extracts_first_json_from_repeated_fenced_output():
    text = (
        'The answer is below.\n```json\n'
        '{"task":"extract_cf_request","status":"complete","cf_request":{"Income":65},'
        '"missing_fields":[],"conflicts":[],"notes":[]}\n```\n'
        'The answer is below again.\n```json\n'
        '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
        '"missing_fields":["Online"],"conflicts":[],"notes":[]}\n```'
    )

    result = normalize_and_parse(text)

    assert result.used_brace_extraction is True
    assert result.parsed_json is not None
    assert result.parsed_json["status"] == "complete"
    assert result.parsed_json["cf_request"] == {"Income": 65}
