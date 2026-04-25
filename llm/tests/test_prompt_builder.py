from __future__ import annotations

from llm.src.parser.prompt_builder import (
    DEFAULT_RESPONSE_SCHEMA_NAME,
    build_live_refinement_user_prompt,
    build_live_user_prompt,
    build_repair_user_prompt,
    build_request_payload,
    build_user_prompt,
    load_system_prompt,
)


def test_build_user_prompt_includes_schema_and_case_payload(sample_benchmark):
    case = sample_benchmark.case_map["A01"]

    prompt = build_user_prompt(sample_benchmark, case)

    assert '"case_id": "A01"' in prompt
    assert '"Income": "number"' in prompt
    assert '"description": "Target income value"' in prompt
    assert '"allowed_status_values"' in prompt
    assert "expected_output" not in prompt


def test_build_live_user_prompt_includes_user_request(sample_benchmark):
    prompt = build_live_user_prompt(sample_benchmark, "Income 40 and Online yes.")

    assert '"input": "Income 40 and Online yes."' in prompt
    assert '"allowed_status_values"' in prompt
    assert "The first character of your response must be '{'" in prompt
    assert "do not omit explicitly labeled fields" in prompt
    assert "emit them in constraint_spec instead of dropping them" in prompt
    assert "Use disallowed_changes, numeric_bounds, or max_changed_features for hard constraints" in prompt
    assert '"label": "dense_complete_bank_profile"' in prompt
    assert '"label": "dense_complete_bank_profile_mixed_connectors"' in prompt
    assert '"label": "dense_complete_bank_profile_zero_boolean_values"' in prompt
    assert '"label": "compact_runtime_reject_profile"' in prompt
    assert '"label": "underspecified_clarification_profile"' in prompt
    assert '"label": "profile_plus_soft_preference"' in prompt
    assert '"label": "profile_plus_hard_constraint_and_soft_preference"' in prompt
    assert '"label": "clarification_style_compact_answer"' in prompt
    assert '"label": "correction_style_answer"' in prompt
    assert '"constraint_spec"' in prompt
    assert "Case payload" not in prompt


def test_build_live_refinement_user_prompt_includes_bank_clarification_exemplars(sample_benchmark):
    prompt = build_live_refinement_user_prompt(
        sample_benchmark,
        user_text="Make the bank result better without changing too much.",
        active_constraint_spec={},
        pending_refinement_clarification=None,
        dataset_id="bank",
        dataset_label="bank profile",
        numeric_bound_fields=["Income", "CCAvg", "Mortgage"],
    )

    assert '"input": "Make the bank result better without changing too much."' in prompt
    assert '"label": "vague_goal_needs_clarification"' in prompt
    assert '"label": "contradictory_same_turn"' in prompt
    assert 'return status "clarification_required"' in prompt
    assert "without changing too much" in prompt


def test_build_live_refinement_user_prompt_includes_grad_clarification_exemplars(sample_benchmark):
    prompt = build_live_refinement_user_prompt(
        sample_benchmark,
        user_text="Do not change CGPA, actually CGPA can change.",
        active_constraint_spec={},
        pending_refinement_clarification=None,
        dataset_id="grad",
        dataset_label="graduate admission profile",
        numeric_bound_fields=["CGPA"],
    )

    assert '"label": "grad_soft_preference"' in prompt
    assert '"label": "grad_vague_goal_needs_clarification"' in prompt
    assert '"label": "grad_contradictory_same_turn"' in prompt
    assert "Do not change CGPA, actually CGPA can change." in prompt


def test_build_repair_user_prompt_includes_invalid_output_and_errors(sample_benchmark):
    prompt = build_repair_user_prompt(
        sample_benchmark,
        invalid_output='{"task": "extract_cf_request", }',
        errors=["Expecting property name enclosed in double quotes"],
    )

    assert '"invalid_output"' in prompt
    assert '"validation_errors"' in prompt
    assert "Repair the invalid parser output" in prompt


def test_build_repair_user_prompt_strengthens_false_complete_guidance(sample_benchmark):
    prompt = build_repair_user_prompt(
        sample_benchmark,
        invalid_output='{"task":"extract_cf_request","status":"complete"}',
        errors=["status 'complete' requires all runtime-required bank fields."],
    )

    assert 'must not keep status "complete"' in prompt
    assert 'Change it to "partial"' in prompt


def test_build_request_payload_has_strict_json_schema_shape():
    payload = build_request_payload(
        model="qwen3.5-27b@q4_0",
        system_prompt="system text",
        user_prompt="user text",
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        response_schema={"type": "object"},
        stream=False,
    )

    assert payload["model"] == "qwen3.5-27b@q4_0"
    assert payload["system_prompt"] == "system text"
    assert payload["input"] == "user text"
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 512
    assert payload["stream"] is False
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": DEFAULT_RESPONSE_SCHEMA_NAME,
            "strict": True,
            "schema": {"type": "object"},
        },
    }
    assert "messages" not in payload


def test_default_response_schema_name_is_v2():
    assert DEFAULT_RESPONSE_SCHEMA_NAME == "ufce_bank_cf_parser_output_v2"


def test_load_system_prompt_preserves_file_contents(tmp_path):
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("\nSystem prompt artifact.\n", encoding="utf-8")

    loaded = load_system_prompt(prompt_path)

    assert loaded == "\nSystem prompt artifact.\n"
