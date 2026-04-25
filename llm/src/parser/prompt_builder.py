from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RESPONSE_SCHEMA_NAME = "ufce_bank_cf_parser_output_v2"
DEFAULT_REFINEMENT_SCHEMA_NAME = "ufce_bank_refinement_feedback_output_v1"


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_schema_reference(
    benchmark,
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    cf_field_schema: dict[str, str] = {}
    for field in benchmark.target_cf_fields:
        if field.type == "float":
            cf_field_schema[field.name] = "number"
        elif field.type == "int":
            cf_field_schema[field.name] = "integer"
        elif field.type == "binary":
            cf_field_schema[field.name] = "binary 0 or 1"
        else:
            cf_field_schema[field.name] = field.type
    return {
        "task": benchmark.output_contract.task,
        "status": list(benchmark.output_contract.status_enum),
        "cf_request": cf_field_schema,
        "constraint_spec": {
            "immutable": ["canonical field name"],
            "disallowed_changes": ["canonical field name"],
            "numeric_bounds": _build_numeric_bound_reference(numeric_bound_fields),
            "max_changed_features": "integer 1 to 3",
            "prefer_fewer_changes": "boolean",
        },
        "missing_fields": ["canonical field name"],
        "conflicts": ["brief string"],
        "notes": ["brief string"],
    }


def build_refinement_schema_reference(
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    return {
        "task": "extract_constraint_feedback",
        "status": [
            "apply",
            "clarification_required",
            "unsupported_feedback",
        ],
        "constraint_feedback_delta": {
            "add_blocked_fields": ["canonical field name"],
            "remove_blocked_fields": ["canonical field name"],
            "set_numeric_bounds": _build_numeric_bound_reference(numeric_bound_fields),
            "clear_numeric_bounds": {
                field_name: ["min", "max"]
                for field_name in _normalize_numeric_bound_fields(numeric_bound_fields)
            },
            "set_max_changed_features": "integer 1 to 3",
            "clear_max_changed_features": "boolean",
            "set_prefer_fewer_changes": "boolean",
            "clear_prefer_fewer_changes": "boolean",
        },
        "ambiguities": ["brief string"],
        "unsupported_feedback": ["brief string"],
        "notes": ["brief string"],
    }


def build_feature_dictionary(benchmark) -> dict[str, dict[str, str]]:
    feature_dictionary: dict[str, dict[str, str]] = {}
    for field in benchmark.target_cf_fields:
        feature_dictionary[field.name] = {
            "type": field.type,
            "description": field.description,
        }
    return feature_dictionary


def build_user_prompt(benchmark, case) -> str:
    payload = {
        "case_id": case.case_id,
        "input": case.input_text,
    }
    instructions = {
        "schema_reference": build_schema_reference(benchmark),
        "feature_dictionary": build_feature_dictionary(benchmark),
        "allowed_status_values": list(benchmark.output_contract.status_enum),
    }
    return (
        "Return exactly one JSON object that matches the schema reference.\n"
        "The first character of your response must be '{' and the last character must be '}'.\n"
        "Stop immediately after the closing '}'.\n"
        "Do not include markdown fences, explanation, or commentary.\n\n"
        "Schema and field contract:\n"
        f"{json.dumps(instructions, ensure_ascii=True, indent=2)}\n\n"
        "Case payload:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def build_live_user_prompt(
    benchmark,
    user_text: str,
    *,
    dataset_id: str = "bank",
    dataset_label: str = "bank profile",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> str:
    payload = {
        "input": user_text,
    }
    instructions = {
        "schema_reference": build_schema_reference(benchmark, numeric_bound_fields=numeric_bound_fields),
        "feature_dictionary": build_feature_dictionary(benchmark),
        "allowed_status_values": list(benchmark.output_contract.status_enum),
    }
    exemplars = build_live_parser_exemplars(dataset_id=dataset_id)
    dataset_specific_hints = build_dataset_prompt_hints(
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        numeric_bound_fields=numeric_bound_fields,
    )
    return (
        "Return exactly one JSON object that matches the schema reference.\n"
        "The first character of your response must be '{' and the last character must be '}'.\n"
        "Stop immediately after the closing '}'.\n"
        "Do not include markdown fences, explanation, or commentary.\n\n"
        f"{dataset_specific_hints}\n\n"
        "Schema and field contract:\n"
        f"{json.dumps(instructions, ensure_ascii=True, indent=2)}\n\n"
        "Fixed exemplars:\n"
        f"{json.dumps(exemplars, ensure_ascii=True, indent=2)}\n\n"
        "User request:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def build_live_refinement_user_prompt(
    benchmark,
    *,
    user_text: str,
    active_constraint_spec: dict[str, Any] | None,
    pending_refinement_clarification: dict[str, Any] | None,
    dataset_id: str = "bank",
    dataset_label: str = "bank profile",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> str:
    payload = {
        "input": user_text,
        "active_constraint_spec": active_constraint_spec or {},
        "pending_refinement_clarification": pending_refinement_clarification,
    }
    instructions = {
        "schema_reference": build_refinement_schema_reference(numeric_bound_fields=numeric_bound_fields),
        "feature_dictionary": build_feature_dictionary(benchmark),
        "supported_refinement_capabilities": [
            "add blocked fields",
            "remove blocked fields",
            "set or clear numeric bounds on " + ", ".join(_normalize_numeric_bound_fields(numeric_bound_fields)),
            "set or clear max_changed_features",
            "set or clear prefer_fewer_changes",
        ],
    }
    exemplars = build_live_refinement_exemplars(
        dataset_id=dataset_id,
        numeric_bound_fields=numeric_bound_fields,
    )
    return (
        "Return exactly one JSON object that matches the refinement schema reference.\n"
        "The first character of your response must be '{' and the last character must be '}'.\n"
        "Stop immediately after the closing '}'.\n"
        "Do not include markdown fences, explanation, or commentary.\n\n"
        f"Apply the refinement language to the active {dataset_label} only.\n"
        "If the same refinement turn contains contradictory instructions, return status \"clarification_required\".\n"
        "If the feedback only asks for a better result or says things like "
        "\"without changing too much\" without a concrete field, bound, or change-limit update, "
        "return status \"clarification_required\".\n"
        "If the request is outside the supported constraint language, return status \"unsupported_feedback\".\n\n"
        "Schema and field contract:\n"
        f"{json.dumps(instructions, ensure_ascii=True, indent=2)}\n\n"
        "Fixed exemplars:\n"
        f"{json.dumps(exemplars, ensure_ascii=True, indent=2)}\n\n"
        "Refinement request:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def build_repair_user_prompt(
    benchmark,
    *,
    invalid_output: str,
    errors: list[str],
    dataset_id: str = "bank",
    dataset_label: str = "bank profile",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> str:
    instructions = {
        "schema_reference": build_schema_reference(benchmark, numeric_bound_fields=numeric_bound_fields),
        "feature_dictionary": build_feature_dictionary(benchmark),
        "allowed_status_values": list(benchmark.output_contract.status_enum),
    }
    payload = {
        "invalid_output": invalid_output,
        "validation_errors": list(errors),
    }
    repair_rules = build_repair_rules(
        errors,
        dataset_label=dataset_label,
        numeric_bound_fields=numeric_bound_fields,
    )
    return (
        "Repair the invalid parser output into exactly one JSON object that matches the schema reference.\n"
        "Return JSON only.\n"
        "The first character of your response must be '{' and the last character must be '}'.\n"
        "Stop immediately after the closing '}'.\n"
        "Do not include markdown fences, explanation, or commentary.\n\n"
        "Repair rules:\n"
        f"{repair_rules}\n\n"
        "Schema and field contract:\n"
        f"{json.dumps(instructions, ensure_ascii=True, indent=2)}\n\n"
        "Repair payload:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def build_refinement_repair_user_prompt(
    benchmark,
    *,
    invalid_output: str,
    errors: list[str],
    active_constraint_spec: dict[str, Any] | None,
    pending_refinement_clarification: dict[str, Any] | None,
    dataset_id: str = "bank",
    dataset_label: str = "bank profile",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> str:
    instructions = {
        "schema_reference": build_refinement_schema_reference(numeric_bound_fields=numeric_bound_fields),
        "feature_dictionary": build_feature_dictionary(benchmark),
    }
    payload = {
        "invalid_output": invalid_output,
        "validation_errors": list(errors),
        "active_constraint_spec": active_constraint_spec or {},
        "pending_refinement_clarification": pending_refinement_clarification,
    }
    repair_rules = [
        "- Return one repaired JSON object only.",
        "- Use only the supported refinement delta keys.",
        "- If the same refinement turn still contains contradictory delta instructions, return status \"clarification_required\".",
        (
            "- If the feedback only asks for improvement or says things like "
            "\"without changing too much\" without a concrete field, bound, or change-limit update, "
            "return status \"clarification_required\"."
        ),
        "- If the request is outside the supported refinement language, return status \"unsupported_feedback\".",
    ]
    return (
        "Repair the invalid refinement parser output into exactly one JSON object that matches the schema reference.\n"
        "Return JSON only.\n"
        "The first character of your response must be '{' and the last character must be '}'.\n"
        "Stop immediately after the closing '}'.\n"
        "Do not include markdown fences, explanation, or commentary.\n\n"
        f"The refinement applies only to the active {dataset_label}.\n\n"
        "Repair rules:\n"
        f"{chr(10).join(repair_rules)}\n\n"
        "Schema and field contract:\n"
        f"{json.dumps(instructions, ensure_ascii=True, indent=2)}\n\n"
        "Repair payload:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def build_live_parser_exemplars(*, dataset_id: str = "bank") -> list[dict[str, Any]]:
    if dataset_id == "grad":
        return [
            {
                "label": "dense_complete_grad_profile",
                "input": (
                    "GRE Score 320, TOEFL Score 110, University Rating 4, SOP 4.5, "
                    "LOR 4.0, CGPA 8.9, Research yes."
                ),
                "output": {
                    "task": "extract_cf_request",
                    "status": "complete",
                    "cf_request": {
                        "GRE Score": 320,
                        "TOEFL Score": 110,
                        "University Rating": 4,
                        "SOP": 4.5,
                        "LOR": 4.0,
                        "CGPA": 8.9,
                        "Research": 1,
                    },
                    "missing_fields": [],
                    "conflicts": [],
                    "notes": [],
                },
            },
            {
                "label": "partial_grad_profile_with_bounds",
                "input": "GRE 315, TOEFL 108, CGPA 8.5, keep CGPA above 8.0.",
                "output": {
                    "task": "extract_cf_request",
                    "status": "partial",
                    "cf_request": {
                        "GRE Score": 315,
                        "TOEFL Score": 108,
                        "CGPA": 8.5,
                    },
                    "constraint_spec": {
                        "numeric_bounds": {
                            "CGPA": {"min": 8.0}
                        }
                    },
                    "missing_fields": [
                        "University Rating",
                        "SOP",
                        "LOR",
                        "Research",
                    ],
                    "conflicts": [],
                    "notes": [],
                },
            },
            {
                "label": "grad_profile_with_soft_preference",
                "input": (
                    "GRE 322, TOEFL 111, University Rating 4, SOP 4.5, LOR 4.5, CGPA 9.0, Research yes. "
                    "Prefer fewer changes."
                ),
                "output": {
                    "task": "extract_cf_request",
                    "status": "complete",
                    "cf_request": {
                        "GRE Score": 322,
                        "TOEFL Score": 111,
                        "University Rating": 4,
                        "SOP": 4.5,
                        "LOR": 4.5,
                        "CGPA": 9.0,
                        "Research": 1,
                    },
                    "constraint_spec": {
                        "prefer_fewer_changes": True
                    },
                    "missing_fields": [],
                    "conflicts": [],
                    "notes": [],
                },
            },
        ]
    return [
        {
            "label": "dense_complete_bank_profile",
            "input": (
                "Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200, "
                "CDAccount yes, Online no, SecuritiesAccount yes, CreditCard no."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 72,
                    "CCAvg": 4.8,
                    "Family": 1,
                    "Education": 2,
                    "Mortgage": 200,
                    "CDAccount": 1,
                    "Online": 0,
                    "SecuritiesAccount": 1,
                    "CreditCard": 0,
                },
                "constraint_spec": {
                    "immutable": [],
                    "disallowed_changes": [],
                    "numeric_bounds": {},
                    "prefer_fewer_changes": False,
                },
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "compact_runtime_reject_profile",
            "input": (
                "Income 40, CCAvg 1.5, Family 3, Education 2, Mortgage 80, "
                "CDAccount yes, Online yes, SecuritiesAccount yes, CreditCard yes."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 40,
                    "CCAvg": 1.5,
                    "Family": 3,
                    "Education": 2,
                    "Mortgage": 80,
                    "CDAccount": 1,
                    "Online": 1,
                    "SecuritiesAccount": 1,
                    "CreditCard": 1,
                },
                "constraint_spec": {
                    "disallowed_changes": ["CDAccount"],
                    "max_changed_features": 1,
                },
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "underspecified_clarification_profile",
            "input": "I want Income 40, CCAvg 1.5, Family 3, Education 2, and Mortgage 80.",
            "output": {
                "task": "extract_cf_request",
                "status": "partial",
                "cf_request": {
                    "Income": 40,
                    "CCAvg": 1.5,
                    "Family": 3,
                    "Education": 2,
                    "Mortgage": 80,
                },
                "constraint_spec": {"numeric_bounds": {"Income": {"min": 40}}},
                "missing_fields": [
                    "CDAccount",
                    "Online",
                    "SecuritiesAccount",
                    "CreditCard",
                ],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "dense_complete_bank_profile_mixed_connectors",
            "input": (
                "Income: 68, Family=1, CCAvg 1.5, Education is 2, Mortgage 0, "
                "SecuritiesAccount no, CDAccount no, Online no, CreditCard no."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 68,
                    "Family": 1,
                    "CCAvg": 1.5,
                    "Education": 2,
                    "Mortgage": 0,
                    "SecuritiesAccount": 0,
                    "CDAccount": 0,
                    "Online": 0,
                    "CreditCard": 0,
                },
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "dense_complete_bank_profile_zero_boolean_values",
            "input": (
                "Income 83, Family 1, CCAvg 2.8, Education 2, Mortgage 0, "
                "SecuritiesAccount 0, CDAccount 0, Online 1, CreditCard 1."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 83,
                    "Family": 1,
                    "CCAvg": 2.8,
                    "Education": 2,
                    "Mortgage": 0,
                    "SecuritiesAccount": 0,
                    "CDAccount": 0,
                    "Online": 1,
                    "CreditCard": 1,
                },
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "profile_plus_soft_preference",
            "input": (
                "Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200, "
                "CDAccount yes, Online no, SecuritiesAccount yes, CreditCard no. Prefer smaller edits."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 72,
                    "CCAvg": 4.8,
                    "Family": 1,
                    "Education": 2,
                    "Mortgage": 200,
                    "CDAccount": 1,
                    "Online": 0,
                    "SecuritiesAccount": 1,
                    "CreditCard": 0,
                },
                "constraint_spec": {"prefer_fewer_changes": True},
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "profile_plus_hard_constraint_and_soft_preference",
            "input": (
                "Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200, "
                "CDAccount yes, Online no, SecuritiesAccount yes, CreditCard no. "
                "Do not change Income. Prefer fewer changes."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 72,
                    "CCAvg": 4.8,
                    "Family": 1,
                    "Education": 2,
                    "Mortgage": 200,
                    "CDAccount": 1,
                    "Online": 0,
                    "SecuritiesAccount": 1,
                    "CreditCard": 0,
                },
                "constraint_spec": {"disallowed_changes": ["Income"], "prefer_fewer_changes": True},
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "clarification_style_compact_answer",
            "input": "Income 40, CCAvg 1.5, Family 3, Education 2, Mortgage 80. Need the remaining fields clarified.",
            "output": {
                "task": "extract_cf_request",
                "status": "partial",
                "cf_request": {
                    "Income": 40,
                    "CCAvg": 1.5,
                    "Family": 3,
                    "Education": 2,
                    "Mortgage": 80,
                },
                "missing_fields": [
                    "CDAccount",
                    "Online",
                    "SecuritiesAccount",
                    "CreditCard",
                ],
                "conflicts": [],
                "notes": [],
            },
        },
        {
            "label": "correction_style_answer",
            "input": (
                "Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200, "
                "CDAccount yes, Online yes, SecuritiesAccount yes, CreditCard no. "
                "Correction: Online should be no."
            ),
            "output": {
                "task": "extract_cf_request",
                "status": "complete",
                "cf_request": {
                    "Income": 72,
                    "CCAvg": 4.8,
                    "Family": 1,
                    "Education": 2,
                    "Mortgage": 200,
                    "CDAccount": 1,
                    "Online": 0,
                    "SecuritiesAccount": 1,
                    "CreditCard": 0,
                },
                "missing_fields": [],
                "conflicts": [],
                "notes": [],
            },
        },
    ]


def build_live_refinement_exemplars(
    *,
    dataset_id: str = "bank",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if dataset_id == "grad":
        active_numeric_bound_fields = _normalize_numeric_bound_fields(numeric_bound_fields)
        return [
            {
                "label": "tighten_grad_bounds",
                "input": "Keep CGPA above 8.5 and limit changes to one feature.",
                "output": {
                    "task": "extract_constraint_feedback",
                    "status": "apply",
                    "constraint_feedback_delta": {
                        "set_numeric_bounds": {
                            active_numeric_bound_fields[-1] if active_numeric_bound_fields else "CGPA": {"min": 8.5}
                        },
                        "set_max_changed_features": 1,
                    },
                    "ambiguities": [],
                    "unsupported_feedback": [],
                    "notes": [],
                },
            },
            {
                "label": "grad_soft_preference",
                "input": "Prefer fewer changes.",
                "output": {
                    "task": "extract_constraint_feedback",
                    "status": "apply",
                    "constraint_feedback_delta": {
                        "set_prefer_fewer_changes": True
                    },
                    "ambiguities": [],
                    "unsupported_feedback": [],
                    "notes": [],
                },
            },
            {
                "label": "grad_vague_goal_needs_clarification",
                "input": "Improve the graduate admission result without changing too much.",
                "output": {
                    "task": "extract_constraint_feedback",
                    "status": "clarification_required",
                    "constraint_feedback_delta": {},
                    "ambiguities": [
                        "The feedback asks for improvement without specifying allowed fields, blocked fields, bounds, or a change limit."
                    ],
                    "unsupported_feedback": [],
                    "notes": [],
                },
            },
            {
                "label": "grad_contradictory_same_turn",
                "input": "Do not change CGPA, actually CGPA can change.",
                "output": {
                    "task": "extract_constraint_feedback",
                    "status": "clarification_required",
                    "constraint_feedback_delta": {},
                    "ambiguities": [
                        "The feedback both blocks and unblocks CGPA in the same refinement turn."
                    ],
                    "unsupported_feedback": [],
                    "notes": [],
                },
            },
        ]
    return [
        {
            "label": "add_bounds_and_preference",
            "input": "Do not change Income. Keep Mortgage below 120. Prefer smaller edits.",
            "output": {
                "task": "extract_constraint_feedback",
                "status": "apply",
                "constraint_feedback_delta": {
                    "add_blocked_fields": ["Income"],
                    "set_numeric_bounds": {
                        "Mortgage": {"max": 120}
                    },
                    "set_prefer_fewer_changes": True,
                },
                "ambiguities": [],
                "unsupported_feedback": [],
                "notes": [],
            },
        },
        {
            "label": "remove_block_and_limit_changes",
            "input": "Income can change again, but allow at most one feature change.",
            "output": {
                "task": "extract_constraint_feedback",
                "status": "apply",
                "constraint_feedback_delta": {
                    "remove_blocked_fields": ["Income"],
                    "set_max_changed_features": 1,
                },
                "ambiguities": [],
                "unsupported_feedback": [],
                "notes": [],
            },
        },
        {
            "label": "vague_goal_needs_clarification",
            "input": "Make the bank result better without changing too much.",
            "output": {
                "task": "extract_constraint_feedback",
                "status": "clarification_required",
                "constraint_feedback_delta": {},
                "ambiguities": [
                    "The feedback asks for improvement without specifying allowed fields, blocked fields, bounds, or a change limit."
                ],
                "unsupported_feedback": [],
                "notes": [],
            },
        },
        {
            "label": "contradictory_same_turn",
            "input": "Do not change Income, actually Income can change.",
            "output": {
                "task": "extract_constraint_feedback",
                "status": "clarification_required",
                "constraint_feedback_delta": {},
                "ambiguities": [
                    "The feedback both blocks and unblocks Income in the same refinement turn."
                ],
                "unsupported_feedback": [],
                "notes": [],
            },
        },
        {
            "label": "unsupported_feedback",
            "input": "Show me all UFCE methods and rank them yourself.",
            "output": {
                "task": "extract_constraint_feedback",
                "status": "unsupported_feedback",
                "constraint_feedback_delta": {},
                "ambiguities": [],
                "unsupported_feedback": [
                    "Method-selection or free-form ranking requests are outside the supported refinement language."
                ],
                "notes": [],
            },
        },
    ]


def build_repair_rules(
    errors: list[str],
    *,
    dataset_label: str = "bank profile",
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> str:
    numeric_bound_text = ", ".join(_normalize_numeric_bound_fields(numeric_bound_fields)) or "supported numeric fields"
    rules = [
        "- Return one repaired JSON object only.",
        "- Keep only explicit canonical field values; do not invent missing values.",
        (
            "- constraint_spec is optional. If you include it, use only the supported keys, "
            f"canonical field names, and numeric-bound fields: {numeric_bound_text}."
        ),
        "- If required runtime fields are still missing but the request is otherwise usable, use status \"partial\".",
        "- Use status \"needs_clarification\" only when the remaining issue is ambiguity or insufficient information, not merely missing required fields.",
    ]
    error_text = " ".join(errors)
    if "status 'complete' requires all runtime-required" in error_text:
        rules.append(
            (
                "- If validation_errors say status \"complete\" requires all runtime-required fields, "
                f"you must not keep status \"complete\" for the active {dataset_label}. "
                "Change it to \"partial\" unless the remaining issue is genuine ambiguity."
            )
        )
    return "\n".join(rules)


def build_live_response_schema(
    benchmark,
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "task": {"type": "string", "const": benchmark.output_contract.task},
        "status": {"type": "string", "enum": list(benchmark.output_contract.status_enum)},
        "cf_request": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                field.name: _json_schema_for_field_type(field.type)
                for field in benchmark.target_cf_fields
            },
        },
        "missing_fields": {"type": "array", "items": {"type": "string"}},
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    }
    numeric_bounds_properties = {
        field_name: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "min": {"type": "number"},
                "max": {"type": "number"},
            },
        }
        for field_name in _normalize_numeric_bound_fields(numeric_bound_fields)
    }
    properties["constraint_spec"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "immutable": {"type": "array", "items": {"type": "string"}},
            "disallowed_changes": {"type": "array", "items": {"type": "string"}},
            "numeric_bounds": {
                "type": "object",
                "additionalProperties": False,
                "properties": numeric_bounds_properties,
            },
            "max_changed_features": {"type": "integer", "enum": [1, 2, 3]},
            "prefer_fewer_changes": {"type": "boolean"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "status", "cf_request", "missing_fields", "conflicts", "notes"],
        "properties": properties,
    }


def build_live_refinement_response_schema(
    *,
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    numeric_fields = _normalize_numeric_bound_fields(numeric_bound_fields)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "task",
            "status",
            "constraint_feedback_delta",
            "ambiguities",
            "unsupported_feedback",
            "notes",
        ],
        "properties": {
            "task": {"type": "string", "const": "extract_constraint_feedback"},
            "status": {"type": "string", "enum": ["apply", "clarification_required", "unsupported_feedback"]},
            "constraint_feedback_delta": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "add_blocked_fields": {"type": "array", "items": {"type": "string"}},
                    "remove_blocked_fields": {"type": "array", "items": {"type": "string"}},
                    "set_numeric_bounds": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            field_name: {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"},
                                },
                            }
                            for field_name in numeric_fields
                        },
                    },
                    "clear_numeric_bounds": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            field_name: {
                                "type": "array",
                                "items": {"type": "string", "enum": ["min", "max"]},
                            }
                            for field_name in numeric_fields
                        },
                    },
                    "set_max_changed_features": {"type": "integer", "enum": [1, 2, 3]},
                    "clear_max_changed_features": {"type": "boolean"},
                    "set_prefer_fewer_changes": {"type": "boolean"},
                    "clear_prefer_fewer_changes": {"type": "boolean"},
                },
            },
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "unsupported_feedback": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_dataset_prompt_hints(
    *,
    dataset_id: str,
    dataset_label: str,
    numeric_bound_fields: list[str] | tuple[str, ...] | None,
) -> str:
    numeric_bound_text = ", ".join(_normalize_numeric_bound_fields(numeric_bound_fields)) or "supported numeric fields"
    if dataset_id == "grad":
        return (
            f"For dense labeled {dataset_label}s, do not omit explicitly labeled fields, decimals, or binary research values.\n"
            "If the user states hard constraints or soft preferences explicitly, emit them in constraint_spec instead of dropping them.\n"
            f"Use numeric_bounds only for these fields: {numeric_bound_text}.\n"
            "Preserve labeled scores, rating scales, and GPA-style decimals exactly."
        )
    return (
        f"For dense labeled {dataset_label}s, do not omit explicitly labeled fields, including decimals and negative boolean values such as no or 0.\n"
        "If the user states hard constraints or soft preferences explicitly, emit them in constraint_spec instead of dropping them.\n"
        f"Use disallowed_changes, numeric_bounds, or max_changed_features for hard constraints, and use prefer_fewer_changes only for soft preferences. Numeric bounds are supported for: {numeric_bound_text}.\n"
        "Preserve labeled booleans, zeros, and decimals exactly."
    )


def _build_numeric_bound_reference(
    numeric_bound_fields: list[str] | tuple[str, ...] | None,
) -> dict[str, dict[str, str]]:
    return {
        field_name: {"min": "number", "max": "number"}
        for field_name in _normalize_numeric_bound_fields(numeric_bound_fields)
    }


def _normalize_numeric_bound_fields(
    numeric_bound_fields: list[str] | tuple[str, ...] | None,
) -> list[str]:
    default_fields = ["Income", "CCAvg", "Mortgage"]
    active = default_fields if numeric_bound_fields is None else list(numeric_bound_fields)
    seen: set[str] = set()
    normalized: list[str] = []
    for field_name in active:
        clean = str(field_name).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def _json_schema_for_field_type(field_type: str) -> dict[str, Any]:
    if field_type == "float":
        return {"type": "number"}
    if field_type == "int":
        return {"type": "integer"}
    if field_type == "binary":
        return {"type": "integer", "enum": [0, 1]}
    return {"type": "string"}


def build_request_payload(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = DEFAULT_RESPONSE_SCHEMA_NAME,
    stream: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "system_prompt": system_prompt,
        "input": user_prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
    }
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": dict(response_schema),
            },
        }
    return payload
