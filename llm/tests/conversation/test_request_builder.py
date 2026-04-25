from __future__ import annotations

from llm.src.conversation.canonical_validator import BankCanonicalValidator
from llm.src.conversation.request_builder import ConversationRequestBuilder
from llm.src.conversation.types import ConversationStage, PendingClarification
from llm.src.validation.schema_validator import ValidationResult, validate_prediction


def make_candidate(cf_request: dict, *, status: str) -> dict:
    required_fields = [
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    missing = [field for field in required_fields if field not in cf_request]
    return {
        "task": "extract_cf_request",
        "status": status,
        "cf_request": dict(cf_request),
        "missing_fields": missing,
        "conflicts": [],
        "notes": [],
    }


def test_builder_ready_result_includes_runtime_request(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    candidate = make_candidate(
        {
            "Income": 140,
            "Family": 2,
            "CCAvg": 7.7376709303,
            "Education": 2,
            "Mortgage": 32,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 1,
            "CreditCard": 0,
        },
        status="complete",
    )
    schema_validation = validate_prediction(candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="bank profile",
        normalized_candidate=candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.runtime_request == canonical_validation.runtime_request
    assert result.builder_reason_codes == ["ready_for_runtime"]
    assert result.policy_version == validator.context.policy.policy_version
    assert result.canonical_field_order == list(validator.required_fields)
    assert result.provenance["parser_status"] == "complete"
    assert result.provenance["pending_clarification_present"] is False
    assert result.provenance["pending_reset"] is False
    assert result.merge_applied is False
    assert result.carried_fields == []


def test_builder_merges_followup_and_keeps_non_ready_runtime_request_null(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 40,
            "Family": 3,
            "CCAvg": 1.5,
            "Education": 2,
            "Mortgage": 80,
        },
        prior_constraint_spec={},
        missing_fields=["CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate({"CDAccount": 1, "Online": 1}, status="partial")
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="CD account yes and online yes.",
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.NEEDS_CLARIFICATION
    assert result.runtime_request is None
    assert result.builder_reason_codes == ["missing_required_fields"]
    assert result.policy_version == validator.context.policy.policy_version
    assert result.canonical_field_order == list(validator.required_fields)
    assert result.provenance["parser_status"] == "partial"
    assert result.provenance["pending_clarification_present"] is True
    assert result.partial_profile_snapshot == {
        "Income": 40,
        "Family": 3,
        "CCAvg": 1.5,
        "Education": 2,
        "Mortgage": 80,
        "CDAccount": 1,
        "Online": 1,
    }
    assert result.merge_applied is True
    assert result.carried_fields == ["Income", "Family", "CCAvg", "Education", "Mortgage"]
    assert result.carried_constraint_keys == []
    assert result.partial_profile_snapshot["Online"] == 1


def test_builder_augments_empty_boolean_followup_from_explicit_text(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 140,
            "Family": 2,
            "CCAvg": 7.7376709303,
            "Education": 2,
            "Mortgage": 32,
        },
        prior_constraint_spec={},
        missing_fields=["SecuritiesAccount", "CDAccount", "Online", "CreditCard"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate({}, status="needs_clarification")
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="CD account no, online no, securities account no, and credit card no.",
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.runtime_request is not None
    assert result.merge_applied is True
    assert result.carried_fields == ["Income", "Family", "CCAvg", "Education", "Mortgage"]
    assert result.carried_constraint_keys == []
    assert result.partial_profile_snapshot == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 0,
        "CreditCard": 0,
    }
    assert result.provenance["pending_clarification_present"] is True
    assert result.provenance["pending_reset"] is False


def test_builder_rejects_cross_dataset_request_inside_bank_session(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    candidate = make_candidate(
        {
            "Income": 140,
            "Family": 2,
            "CCAvg": 7.7376709303,
            "Education": 2,
            "Mortgage": 32,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 1,
            "CreditCard": 0,
        },
        status="complete",
    )
    schema_validation = validate_prediction(candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="Switch this case to the graduate admission dataset and use GRE 320 instead.",
        normalized_candidate=candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        dataset_id="bank",
        supported_dataset_ids=["bank", "grad"],
    )

    assert result is not None
    assert result.builder_status == ConversationStage.UNSUPPORTED_REQUEST
    assert result.builder_reason_codes == ["unsupported_intent"]
    assert result.runtime_request is None
    assert result.provenance["unsupported_intent_type"] == "dataset_switch"
    assert result.provenance["requested_dataset_label"] == "graduate admission profile"


def test_builder_keeps_bank_session_authoritative_when_text_mentions_graduate(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    candidate = make_candidate(
        {
            "Income": 65000,
            "Family": 2,
            "CCAvg": 1.5,
            "Education": 2,
            "Mortgage": 0,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 1,
            "CreditCard": 1,
        },
        status="complete",
    )
    schema_validation = validate_prediction(candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input=(
            "I have an annual income of $65,000, a family of 2, and spend about $1.5k monthly on my credit cards. "
            "My education is level 2 (graduate). I have no mortgage, I do not have a securities account or a CD account, "
            "but I do use online banking and own a bank credit card."
        ),
        normalized_candidate=candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        dataset_id="bank",
        supported_dataset_ids=["bank", "grad"],
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.builder_reason_codes == ["ready_for_runtime"]


def test_builder_merges_followup_constraint_spec_with_deep_numeric_bounds(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 40,
            "Family": 3,
            "CCAvg": 1.5,
            "Education": 2,
            "Mortgage": 80,
        },
        prior_constraint_spec={
            "disallowed_changes": ["Income"],
            "numeric_bounds": {
                "Income": {"min": 35.0},
                "Mortgage": {"max": 80.0},
            },
        },
        missing_fields=["CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate(
        {
            "CDAccount": 1,
            "Online": 1,
            "SecuritiesAccount": 0,
            "CreditCard": 0,
        },
        status="partial",
    )
    followup_candidate["constraint_spec"] = {
        "numeric_bounds": {"Income": {"max": 65.0}},
        "prefer_fewer_changes": True,
    }
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="CD account yes, online yes, securities no, credit card no, keep income at or below 65.",
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.merge_applied is True
    assert result.carried_fields == ["Income", "Family", "CCAvg", "Education", "Mortgage"]
    assert result.carried_constraint_keys == [
        "disallowed_changes.Income",
        "numeric_bounds.Income.min",
        "numeric_bounds.Mortgage.max",
    ]
    assert result.carried_preference_keys == []
    assert result.provenance["followup_classification"] == "profile_completion"
    assert result.provenance["merge_provenance"]["soft_preferences"] == {
        "prefer_fewer_changes": "preference_added"
    }
    assert result.runtime_request is not None
    assert result.runtime_request["constraint_spec"] == {
        "disallowed_changes": ["Income"],
        "numeric_bounds": {
            "Income": {"min": 35.0, "max": 65.0},
            "Mortgage": {"max": 80.0},
        },
        "prefer_fewer_changes": True,
    }


def test_builder_augments_full_profile_restatement_before_pending_reset(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 75,
            "Family": 4,
            "Education": 2,
            "Mortgage": 182,
            "CDAccount": 0,
            "Online": 0,
            "SecuritiesAccount": 0,
            "CreditCard": 0,
        },
        prior_constraint_spec={"disallowed_changes": ["Income"]},
        missing_fields=["CCAvg"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate(
        {
            "Income": 75,
            "Family": 4,
            "Education": 2,
            "Mortgage": 182,
            "CDAccount": 0,
            "Online": 1,
            "SecuritiesAccount": 0,
            "CreditCard": 0,
        },
        status="partial",
    )
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input=(
            "Income 75, Family 4, CCAvg 0.1, Education 2, Mortgage 182, "
            "SecuritiesAccount no, CDAccount no, Online yes, CreditCard no. Do not change Income."
        ),
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.partial_profile_snapshot == {
        "Income": 75,
        "Family": 4,
        "CCAvg": 0.1,
        "Education": 2,
        "Mortgage": 182,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 1,
        "CreditCard": 0,
    }
    assert result.merge_applied is False
    assert result.pending_reset is True
    assert result.carried_fields == []


def test_builder_harmless_full_restatement_that_answers_missing_fields_merges(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 140,
            "Family": 2,
            "CCAvg": 7.7376709303,
            "Education": 2,
            "Mortgage": 32,
        },
        prior_constraint_spec={},
        missing_fields=["SecuritiesAccount", "CDAccount", "Online", "CreditCard"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate(
        {
            "Income": 140,
            "Family": 2,
            "CCAvg": 7.7376709303,
            "Education": 2,
            "Mortgage": 32,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 1,
            "CreditCard": 0,
        },
        status="complete",
    )
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount no, CDAccount no, Online yes, CreditCard no."
        ),
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.pending_reset is False
    assert result.merge_applied is True
    assert result.carried_fields == ["Income", "Family", "CCAvg", "Education", "Mortgage"]
    assert result.provenance["followup_classification"] == "profile_completion"
    assert result.provenance["merge_provenance"]["profile_fields"]["Income"] == "ignored_duplicate"
    assert result.provenance["merge_provenance"]["profile_fields"]["SecuritiesAccount"] == "filled_from_followup"


def test_builder_followup_with_explicit_correction_phrase_overwrites_prior_field(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 40,
            "Family": 3,
            "CCAvg": 1.5,
            "Education": 2,
            "Mortgage": 80,
        },
        prior_constraint_spec={},
        missing_fields=["CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate(
        {
            "Income": 60,
            "CDAccount": 1,
            "Online": 1,
            "SecuritiesAccount": 1,
            "CreditCard": 1,
        },
        status="partial",
    )
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="Actually, change Income to 60. CD account yes, online yes, securities yes, credit card yes.",
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.READY_FOR_RUNTIME
    assert result.partial_profile_snapshot == {
        "Income": 60,
        "Family": 3,
        "CCAvg": 1.5,
        "Education": 2,
        "Mortgage": 80,
        "SecuritiesAccount": 1,
        "CDAccount": 1,
        "Online": 1,
        "CreditCard": 1,
    }
    assert result.merge_applied is True
    assert result.provenance["followup_classification"] == "correction"
    assert result.provenance["merge_provenance"]["profile_fields"]["Income"] == "corrected_by_followup"


def test_builder_followup_with_restart_cue_clears_prior_state(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    pending = PendingClarification(
        prior_cf_request={
            "Income": 40,
            "Family": 3,
            "CCAvg": 1.5,
            "Education": 2,
            "Mortgage": 80,
        },
        prior_constraint_spec={"prefer_fewer_changes": True},
        missing_fields=["CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        required_field_order=list(validator.required_fields),
        originating_turn_id="run_parent",
    )
    followup_candidate = make_candidate({}, status="needs_clarification")
    schema_validation = validate_prediction(followup_candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=followup_candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="Start over.",
        normalized_candidate=followup_candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
        pending_clarification=pending,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.NEEDS_CLARIFICATION
    assert result.merge_applied is False
    assert result.pending_reset is True
    assert result.partial_profile_snapshot == {}
    assert result.provenance["followup_classification"] == "fresh_request"
    assert result.provenance["reset_decision"] == "fresh_request"


def test_builder_detects_unsupported_intent_even_without_runtime_ready(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    candidate = make_candidate({}, status="needs_clarification")
    schema_validation = ValidationResult(
        is_valid=True,
        errors=(),
        unexpected_top_level_keys=(),
        unexpected_cf_fields=(),
    )
    canonical_validation = validator.validate(candidate=candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="Give me general financial advice about how to optimize my finances.",
        normalized_candidate=candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.UNSUPPORTED_REQUEST
    assert result.runtime_request is None
    assert result.builder_reason_codes == ["unsupported_intent"]
    assert result.policy_version == validator.context.policy.policy_version
    assert result.canonical_field_order == list(validator.required_fields)
    assert result.provenance["pending_clarification_present"] is False
    assert result.provenance["pending_reset"] is False
    assert result.partial_profile_snapshot == {}


def test_builder_conflict_result_exposes_conflicts_and_null_runtime_request(sample_benchmark):
    validator = BankCanonicalValidator()
    builder = ConversationRequestBuilder(
        canonical_validator=validator,
        benchmark=sample_benchmark,
        policy=validator.context.policy,
    )
    candidate = {
        "task": "extract_cf_request",
        "status": "conflict",
        "cf_request": {"Income": 40},
        "missing_fields": [],
        "conflicts": ["Income cannot be both 40 and 60."],
        "notes": [],
    }
    schema_validation = validate_prediction(candidate, sample_benchmark)
    canonical_validation = validator.validate(candidate=candidate, schema_validation=schema_validation)

    result = builder.build(
        user_input="Income 40 and Income 60.",
        normalized_candidate=candidate,
        schema_validation=schema_validation,
        canonical_validation=canonical_validation,
    )

    assert result is not None
    assert result.builder_status == ConversationStage.CONFLICT
    assert result.builder_reason_codes == ["conflicting_values"]
    assert result.runtime_request is None
    assert result.partial_profile_snapshot == {"Income": 40}
    assert result.conflicts == ["Income cannot be both 40 and 60."]
    assert result.policy_version == validator.context.policy.policy_version
    assert result.canonical_field_order == list(validator.required_fields)
    assert result.provenance["parser_status"] == "conflict"
    assert result.provenance["pending_clarification_present"] is False
    assert result.provenance["pending_reset"] is False
