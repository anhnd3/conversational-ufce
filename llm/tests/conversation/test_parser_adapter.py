from __future__ import annotations

import json

from llm.src.conversation.parser_adapter import LiveLmStudioParserAdapter


def test_parser_adapter_request_profiles_are_task_specific(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    schema_path = tmp_path / "schema.json"
    refinement_schema_path = tmp_path / "refinement_schema.json"
    prompt_path.write_text("system prompt", encoding="utf-8")
    schema_path.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    refinement_schema_path.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    adapter = LiveLmStudioParserAdapter(
        system_prompt_path=prompt_path,
        schema_path=schema_path,
        refinement_schema_path=refinement_schema_path,
    )

    parse_profile = adapter.describe_request_profile("parse")
    repair_profile = adapter.describe_request_profile("repair")
    refinement_parse_profile = adapter.describe_request_profile("parse_refinement")
    refinement_repair_profile = adapter.describe_request_profile("repair_refinement")
    token_policy = adapter.describe_token_policy()

    assert parse_profile["max_tokens"] == 512
    assert repair_profile["max_tokens"] == 768
    assert refinement_parse_profile["max_tokens"] == 512
    assert refinement_repair_profile["max_tokens"] == 768
    assert parse_profile["structured_output_mode"] == "json_schema_strict"
    assert refinement_parse_profile["response_schema_name"] == "ufce_bank_refinement_feedback_output_v1"
    assert token_policy["future_negotiation"]["min"] == 2048
