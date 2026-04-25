from __future__ import annotations

from llm.src.adapters.lmstudio_client import (
    adapt_chat_completions_payload,
    call_lm_studio,
    extract_response_data,
)
from llm.src.conversation.parser_adapter import LiveLmStudioParserAdapter
from llm.src.orchestration.parse_then_validate import parse_then_validate
from llm.src.utils.hashing import strip_run_prefix


def test_parse_then_validate_skips_empty_parse_on_api_error(sample_benchmark):
    normalized, validation_result = parse_then_validate(
        message_text="",
        benchmark=sample_benchmark,
        api_error="HTTP 400: missing_required_parameter, param=model, message=Required",
    )

    assert normalized.normalized_text == ""
    assert normalized.parsed_json is None
    assert normalized.parse_error is None
    assert validation_result.is_valid is False
    assert validation_result.errors == ()


def test_extract_response_data_handles_output_items():
    response_json = {
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "text", "text": "thinking"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"task":"extract_cf_request"}'}],
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        "stats": {"time_to_first_token_ms": 12.0, "tokens_per_second": 34.5},
    }

    extracted = extract_response_data(response_json, 120.0)

    assert extracted["reasoning_text"] == "thinking"
    assert extracted["message_text"] == '{"task":"extract_cf_request"}'
    assert extracted["derived_metrics"]["prompt_tokens"] == 10
    assert extracted["derived_metrics"]["ttft_ms"] == 12.0


def test_call_lm_studio_formats_structured_http_errors(monkeypatch):
    class DummyResponse:
        ok = False
        status_code = 400
        text = (
            '{"error":{"message":"Required","type":"invalid_request",'
            '"code":"missing_required_parameter","param":"model"}}'
        )

        @staticmethod
        def json():
            return {
                "error": {
                    "message": "Required",
                    "type": "invalid_request",
                    "code": "missing_required_parameter",
                    "param": "model",
                }
            }

    def fake_post(url, json, timeout):
        assert url == "http://localhost:1234/v1/chat/completions"
        assert json["model"] == "qwen3.5-27b@q4_0"
        assert json["messages"] == [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
        ]
        return DummyResponse()

    monkeypatch.setattr("llm.src.adapters.lmstudio_client.requests.post", fake_post)

    result = call_lm_studio(
        "http://localhost:1234",
        {"model": "qwen3.5-27b@q4_0", "system_prompt": "x", "input": "y"},
        30.0,
    )

    assert result["ok"] is False
    assert result["status_code"] == 400
    assert result["error"] == (
        "HTTP 400: missing_required_parameter, param=model, "
        "type=invalid_request, message=Required"
    )


def test_adapt_chat_completions_payload_translates_legacy_fields():
    payload = adapt_chat_completions_payload(
        {
            "model": "qwen/qwen3-14b",
            "system_prompt": "system",
            "input": "user",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "stream": False,
            "response_format": {"type": "json_schema"},
        }
    )

    assert payload == {
        "model": "qwen/qwen3-14b",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "stream": False,
        "response_format": {"type": "json_schema"},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
    }


def test_strip_run_prefix_handles_python38_compatible_run_ids():
    assert strip_run_prefix("run_20260307_143848") == "20260307_143848"
    assert strip_run_prefix("20260307_143848") == "20260307_143848"


def test_parser_adapter_parse_uses_strict_schema_and_parse_budget(sample_benchmark, tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.txt"
    schema_path = tmp_path / "schema.json"
    prompt_path.write_text("system prompt", encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_call_lm_studio(api_base, payload, timeout_s):
        captured["api_base"] = api_base
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "response_json": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"task":"extract_cf_request","status":"partial","cf_request":{},'
                                    '"missing_fields":[],"conflicts":[],"notes":[]}'
                                ),
                            }
                        ],
                    }
                ]
            },
            "response_text": '{"ok":true}',
            "elapsed_ms": 12.0,
        }

    monkeypatch.setattr("llm.src.conversation.parser_adapter.call_lm_studio", fake_call_lm_studio)

    adapter = LiveLmStudioParserAdapter(
        system_prompt_path=prompt_path,
        schema_path=schema_path,
    )

    result = adapter.parse(user_text="bank parse", benchmark=sample_benchmark)

    payload = captured["payload"]
    assert captured["api_base"] == "http://localhost:1234"
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 512
    assert payload["stream"] is False
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert result.task_type == "parse"


def test_parser_adapter_repair_uses_repair_budget(sample_benchmark, tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.txt"
    schema_path = tmp_path / "schema.json"
    prompt_path.write_text("system prompt", encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_call_lm_studio(api_base, payload, timeout_s):
        del api_base
        del timeout_s
        captured["payload"] = payload
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "response_json": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"task":"extract_cf_request","status":"partial","cf_request":{},'
                                    '"missing_fields":[],"conflicts":[],"notes":[]}'
                                ),
                            }
                        ],
                    }
                ]
            },
            "response_text": '{"ok":true}',
            "elapsed_ms": 12.0,
        }

    monkeypatch.setattr("llm.src.conversation.parser_adapter.call_lm_studio", fake_call_lm_studio)

    adapter = LiveLmStudioParserAdapter(
        system_prompt_path=prompt_path,
        schema_path=schema_path,
    )

    result = adapter.repair(
        invalid_output='{"task":"extract_cf_request",}',
        errors=["Expecting property name enclosed in double quotes"],
        benchmark=sample_benchmark,
    )

    payload = captured["payload"]
    assert payload["max_tokens"] == 768
    assert result.task_type == "repair"


def test_parser_adapter_rejects_unsupported_structured_output(sample_benchmark, tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.txt"
    schema_path = tmp_path / "schema.json"
    prompt_path.write_text("system prompt", encoding="utf-8")
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    def fake_call_lm_studio(api_base, payload, timeout_s):
        del api_base
        del payload
        del timeout_s
        return {
            "ok": False,
            "status_code": 400,
            "error": "HTTP 400: param=response_format, type=invalid_request, message=Unsupported",
            "response_json": {"error": {"param": "response_format"}},
            "response_text": '{"error":{"param":"response_format"}}',
            "elapsed_ms": 5.0,
        }

    monkeypatch.setattr("llm.src.conversation.parser_adapter.call_lm_studio", fake_call_lm_studio)

    adapter = LiveLmStudioParserAdapter(
        system_prompt_path=prompt_path,
        schema_path=schema_path,
    )

    result = adapter.parse(user_text="bank parse", benchmark=sample_benchmark)

    assert result.failure_cause == "unsupported_structured_output"
    assert result.api_error is not None
    assert "Parser configuration error" in result.api_error
