from __future__ import annotations

from llm_eval.reporting import render_errors_markdown


def test_render_errors_markdown_includes_api_response_excerpt():
    rows = [
        {
            "case_id": "A01",
            "repeat_id": 1,
            "api_error": "HTTP 400: missing_required_parameter, param=model, message=Required",
            "parse_error": None,
            "validation_error_list": [],
            "schema_valid": False,
            "exact_match": False,
            "valid_json": False,
            "final_message_text": "",
            "full_api_response": {
                "error": {
                    "code": "missing_required_parameter",
                    "message": "Required",
                    "param": "model",
                }
            },
            "raw_response_text": None,
        }
    ]

    rendered = render_errors_markdown(rows)

    assert "api_error: HTTP 400: missing_required_parameter, param=model, message=Required" in rendered
    assert 'api_response_excerpt: {"error": {"code": "missing_required_parameter"' in rendered
    assert "parse_error: none" in rendered
    assert "validation_errors: none" in rendered
