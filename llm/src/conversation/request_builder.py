from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from llm.src.conversation.bank_profile_extractor import (
    FIELD_PROVENANCE_PARSER,
    extract_explicit_bank_values,
    merge_bank_candidate_with_explicit_values,
)
from llm.src.conversation.canonical_session_state import (
    RESET_DECISION_FRESH_REQUEST,
    RESET_DECISION_RESET_NO_MERGE,
    combine_constraint_buckets,
    split_constraint_buckets,
)
from llm.src.conversation.types import (
    CanonicalValidationResult,
    ConversationStage,
    PendingClarification,
    RequestBuildResult,
    normalize_parser_quality_payload,
)
from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from llm.src.validation.schema_validator import ValidationResult, validate_prediction


MISSING_REQUIRED_FIELDS = "missing_required_fields"
CONFLICTING_VALUES = "conflicting_values"
UNSUPPORTED_INTENT = "unsupported_intent"
READY_FOR_RUNTIME_REASON = "ready_for_runtime"

FOLLOWUP_CLASSIFICATION_FRESH_REQUEST = "fresh_request"
FOLLOWUP_CLASSIFICATION_CORRECTION = "correction"
FOLLOWUP_CLASSIFICATION_PROFILE_COMPLETION = "profile_completion"
FOLLOWUP_CLASSIFICATION_CONSTRAINT_UPDATE = "constraint_update"
FOLLOWUP_CLASSIFICATION_PREFERENCE_UPDATE = "preference_update"
FOLLOWUP_CLASSIFICATION_AMBIGUOUS = "ambiguous_followup"

MERGE_PROVENANCE_CARRIED = "carried_from_previous_state"
MERGE_PROVENANCE_FILLED = "filled_from_followup"
MERGE_PROVENANCE_CORRECTED = "corrected_by_followup"
MERGE_PROVENANCE_CONSTRAINT = "constraint_added"
MERGE_PROVENANCE_PREFERENCE = "preference_added"
MERGE_PROVENANCE_CONFLICT = "merge_conflict"
MERGE_PROVENANCE_DUPLICATE = "ignored_duplicate"
MERGE_PROVENANCE_RESTART = "restart_reset"

UNSUPPORTED_DATASET_KEYWORDS = (
    "diabetes dataset",
    "adult dataset",
    "housing dataset",
    "german credit",
    "iris dataset",
    "mnist",
    "cifar",
)
DATASET_KEYWORD_MAP = {
    "bank": ("bank", "personal loan", "income", "mortgage", "ccavg"),
    "grad": (
        "graduate admission",
        "graduate admission dataset",
        "grad dataset",
        "gre",
        "toefl",
        "cgpa",
        "sop",
        "lor",
        "research",
        "university rating",
    ),
}
DATASET_LABEL_MAP = {
    "bank": "bank profile",
    "grad": "graduate admission profile",
}
OPTIMIZATION_KEYWORDS = (
    "optimize",
    "optimization",
    "maximize",
    "minimize",
    "best possible",
    "best strategy",
    "fewest changes",
    "lowest cost",
)
ADVICE_KEYWORDS = (
    "financial advice",
    "general advice",
    "what should i do",
    "what do you recommend",
    "tips",
    "guide me",
)
RESTART_CUES = (
    "new case",
    "another customer",
    "start over",
    "forget previous",
)
CORRECTION_CUES = (
    "actually",
    "correction",
    "i mean",
    "instead",
    "change it to",
)
EXPLICIT_DATASET_SWITCH_PATTERNS = (
    re.compile(r"\b(?:switch|change|move|convert)\s+(?:this\s+case|the\s+case)?\s*to\b", re.IGNORECASE),
    re.compile(r"\b(?:switch|change|move|convert)\s+to\b", re.IGNORECASE),
)
DATASET_SWITCH_VERBS = ("switch", "change", "move", "convert", "use")


@dataclass(frozen=True)
class FollowupMergeDecision:
    candidate: dict[str, Any]
    field_provenance: dict[str, str]
    followup_classification: str | None
    merge_provenance: dict[str, Any]
    carried_fields: list[str]
    carried_constraint_keys: list[str]
    carried_preference_keys: list[str]
    pending_reset: bool
    reset_decision: str
    merge_applied: bool


class ConversationRequestBuilder:
    def __init__(self, *, canonical_validator, benchmark, policy) -> None:
        self.canonical_validator = canonical_validator
        self.benchmark = benchmark
        self.policy = policy
        self.required_fields = list(canonical_validator.required_fields)

    def build(
        self,
        *,
        user_input: str,
        normalized_candidate: dict[str, Any] | None,
        schema_validation: ValidationResult,
        canonical_validation: CanonicalValidationResult,
        parser_quality: dict[str, Any] | None = None,
        pending_clarification: PendingClarification | None = None,
        canonical_session_state: dict[str, Any] | None = None,
        field_provenance: dict[str, str] | None = None,
        policy=None,
        required_fields: list[str] | None = None,
        dataset_id: str | None = None,
        supported_dataset_ids: list[str] | None = None,
    ) -> RequestBuildResult | None:
        active_policy = policy or self.policy
        active_required_fields = list(required_fields or self.required_fields)
        active_dataset_id = str(dataset_id or getattr(active_policy, "dataset_name", "bank")).strip().lower() or "bank"
        effective_candidate = normalized_candidate
        effective_schema = schema_validation
        effective_canonical = canonical_validation
        effective_field_provenance = dict(field_provenance or {})
        merge_applied = False
        carried_fields: list[str] = []
        carried_constraint_keys: list[str] = []
        carried_preference_keys: list[str] = []
        pending_reset = False
        followup_classification: str | None = None
        reset_decision = "none"
        merge_provenance: dict[str, Any] = {}

        if pending_clarification is not None and isinstance(effective_candidate, dict):
            decision = build_followup_merge_decision(
                user_input=user_input,
                candidate=effective_candidate,
                pending_clarification=pending_clarification,
                canonical_session_state=canonical_session_state,
                policy=active_policy,
                required_fields=active_required_fields,
                field_provenance=effective_field_provenance,
            )
            effective_candidate = decision.candidate
            effective_field_provenance = decision.field_provenance
            followup_classification = decision.followup_classification
            merge_provenance = decision.merge_provenance
            carried_fields = decision.carried_fields
            carried_constraint_keys = decision.carried_constraint_keys
            carried_preference_keys = decision.carried_preference_keys
            pending_reset = decision.pending_reset
            reset_decision = decision.reset_decision
            merge_applied = decision.merge_applied
            effective_schema = validate_prediction(
                effective_candidate,
                self.benchmark,
                numeric_bound_fields=getattr(self.canonical_validator, "numeric_bound_fields", None),
            )
            effective_canonical = self.canonical_validator.validate(
                candidate=effective_candidate,
                schema_validation=effective_schema,
                dataset_id=active_dataset_id,
            )

        provenance = build_builder_provenance(
            parser_status=extract_parser_status(effective_candidate),
            pending_clarification=pending_clarification,
            pending_reset=pending_reset,
            parser_quality=parser_quality,
            field_provenance=effective_field_provenance,
            followup_classification=followup_classification,
            reset_decision=reset_decision,
            merge_provenance=merge_provenance,
        )

        dataset_switch = detect_explicit_dataset_switch(
            user_input=user_input,
            dataset_id=active_dataset_id,
            supported_dataset_ids=supported_dataset_ids,
        )
        if dataset_switch is not None:
            provenance = dict(provenance)
            provenance.update(dataset_switch)
            provenance["unsupported_intent_type"] = "dataset_switch"
            return _build_unsupported_request_result(
                effective_candidate=effective_candidate,
                effective_canonical=effective_canonical,
                effective_field_provenance=effective_field_provenance,
                effective_schema=effective_schema,
                merge_applied=merge_applied,
                carried_fields=carried_fields,
                carried_constraint_keys=carried_constraint_keys,
                carried_preference_keys=carried_preference_keys,
                pending_reset=pending_reset,
                provenance=provenance,
                active_policy=active_policy,
                active_required_fields=active_required_fields,
            )

        if is_unsupported_intent(
            user_input=user_input,
            candidate=effective_candidate,
        ):
            provenance = dict(provenance)
            provenance["unsupported_intent_type"] = "generic"
            return _build_unsupported_request_result(
                effective_candidate=effective_candidate,
                effective_canonical=effective_canonical,
                effective_field_provenance=effective_field_provenance,
                effective_schema=effective_schema,
                merge_applied=merge_applied,
                carried_fields=carried_fields,
                carried_constraint_keys=carried_constraint_keys,
                carried_preference_keys=carried_preference_keys,
                pending_reset=pending_reset,
                provenance=provenance,
                active_policy=active_policy,
                active_required_fields=active_required_fields,
            )

        if effective_canonical.final_stage == ConversationStage.PARSER_FAILURE:
            return None

        if effective_canonical.final_stage == ConversationStage.CONFLICT:
            return RequestBuildResult(
                builder_status=ConversationStage.CONFLICT,
                builder_reason_codes=[CONFLICTING_VALUES],
                partial_profile_snapshot=build_partial_snapshot(
                    candidate=effective_candidate,
                    ordered_fields=active_required_fields,
                ),
                runtime_request=None,
                missing_fields=[],
                conflicts=list(effective_canonical.confirmed_conflicts),
                policy_version=active_policy.policy_version,
                canonical_field_order=list(active_required_fields),
                provenance=provenance,
                merge_applied=merge_applied,
                carried_fields=carried_fields,
                carried_constraint_keys=carried_constraint_keys,
                carried_preference_keys=carried_preference_keys,
                pending_reset=pending_reset,
                _normalized_candidate=effective_candidate,
                _schema_validation=effective_schema.to_dict(),
                _canonical_validation=effective_canonical,
            )

        if effective_canonical.final_stage == ConversationStage.NEEDS_CLARIFICATION:
            return RequestBuildResult(
                builder_status=ConversationStage.NEEDS_CLARIFICATION,
                builder_reason_codes=[MISSING_REQUIRED_FIELDS],
                partial_profile_snapshot=build_partial_snapshot(
                    candidate=effective_candidate,
                    ordered_fields=active_required_fields,
                ),
                runtime_request=None,
                missing_fields=list(effective_canonical.missing_runtime_fields),
                conflicts=[],
                policy_version=active_policy.policy_version,
                canonical_field_order=list(active_required_fields),
                provenance=provenance,
                merge_applied=merge_applied,
                carried_fields=carried_fields,
                carried_constraint_keys=carried_constraint_keys,
                carried_preference_keys=carried_preference_keys,
                pending_reset=pending_reset,
                _normalized_candidate=effective_candidate,
                _schema_validation=effective_schema.to_dict(),
                _canonical_validation=effective_canonical,
            )

        return RequestBuildResult(
            builder_status=ConversationStage.READY_FOR_RUNTIME,
            builder_reason_codes=[READY_FOR_RUNTIME_REASON],
            partial_profile_snapshot=build_partial_snapshot(
                candidate=effective_candidate,
                ordered_fields=active_required_fields,
            ),
            runtime_request=dict(effective_canonical.runtime_request or {}),
            missing_fields=[],
            conflicts=[],
            policy_version=active_policy.policy_version,
            canonical_field_order=list(active_required_fields),
            provenance=provenance,
            merge_applied=merge_applied,
            carried_fields=carried_fields,
            carried_constraint_keys=carried_constraint_keys,
            carried_preference_keys=carried_preference_keys,
            pending_reset=pending_reset,
            _normalized_candidate=effective_candidate,
            _schema_validation=effective_schema.to_dict(),
            _canonical_validation=effective_canonical,
        )


def build_followup_merge_decision(
    *,
    user_input: str,
    candidate: dict[str, Any],
    pending_clarification: PendingClarification,
    canonical_session_state: dict[str, Any] | None,
    policy,
    required_fields: list[str],
    field_provenance: dict[str, str] | None = None,
) -> FollowupMergeDecision:
    effective_candidate, effective_field_provenance = augment_followup_candidate_with_explicit_bank_values(
        user_input=user_input,
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
        field_provenance=field_provenance,
    )
    prior_profile, prior_hard_constraints, prior_soft_preferences, prior_field_provenance = build_merge_source(
        canonical_session_state=canonical_session_state,
        pending_clarification=pending_clarification,
        required_fields=required_fields,
    )
    correction_intent = has_any_cue(user_input=user_input, cues=CORRECTION_CUES)
    restart_intent = has_any_cue(user_input=user_input, cues=RESTART_CUES)
    followup_task = str(effective_candidate.get("task") or "extract_cf_request")
    followup_notes = _normalize_string_list(effective_candidate.get("notes"))
    followup_conflicts = _normalize_string_list(effective_candidate.get("conflicts"))
    followup_request = dict(effective_candidate.get("cf_request") or {})
    followup_hard_constraints, followup_soft_preferences = split_constraint_buckets(
        effective_candidate.get("constraint_spec"),
        feature_order=required_fields,
    )
    full_profile_restatement = has_full_profile_restatement(
        candidate=effective_candidate,
        required_fields=required_fields,
    )

    merge_provenance: dict[str, Any] = {
        "profile_fields": {},
        "hard_constraints": {},
        "soft_preferences": {},
        "reset_decision": "none",
    }
    if restart_intent:
        merge_provenance["session_state"] = MERGE_PROVENANCE_RESTART
        merge_provenance["reset_decision"] = RESET_DECISION_FRESH_REQUEST
        return FollowupMergeDecision(
            candidate=effective_candidate,
            field_provenance=effective_field_provenance,
            followup_classification=FOLLOWUP_CLASSIFICATION_FRESH_REQUEST,
            merge_provenance=merge_provenance,
            carried_fields=[],
            carried_constraint_keys=[],
            carried_preference_keys=[],
            pending_reset=True,
            reset_decision=RESET_DECISION_FRESH_REQUEST,
            merge_applied=False,
        )

    if full_profile_restatement and not correction_intent and not is_harmless_restatement_answer(
        prior_profile=prior_profile,
        followup_request=followup_request,
        pending_missing_fields=pending_clarification.missing_fields,
    ):
        merge_provenance["reset_decision"] = RESET_DECISION_RESET_NO_MERGE
        return FollowupMergeDecision(
            candidate=effective_candidate,
            field_provenance=effective_field_provenance,
            followup_classification=FOLLOWUP_CLASSIFICATION_AMBIGUOUS,
            merge_provenance=merge_provenance,
            carried_fields=[],
            carried_constraint_keys=[],
            carried_preference_keys=[],
            pending_reset=True,
            reset_decision=RESET_DECISION_RESET_NO_MERGE,
            merge_applied=False,
        )

    if is_empty_nonanswer_followup(
        candidate=effective_candidate,
        followup_hard_constraints=followup_hard_constraints,
        followup_soft_preferences=followup_soft_preferences,
    ):
        return FollowupMergeDecision(
            candidate=effective_candidate,
            field_provenance=effective_field_provenance,
            followup_classification=FOLLOWUP_CLASSIFICATION_AMBIGUOUS,
            merge_provenance=merge_provenance,
            carried_fields=[],
            carried_constraint_keys=[],
            carried_preference_keys=[],
            pending_reset=True,
            reset_decision="none",
            merge_applied=False,
        )

    (
        merged_profile,
        profile_provenance,
        carried_fields,
        profile_change_detected,
        profile_correction_detected,
        profile_fill_detected,
        profile_conflicts,
    ) = merge_profile_facts(
        prior_profile=prior_profile,
        followup_request=followup_request,
        followup_field_provenance=effective_field_provenance,
        prior_field_provenance=prior_field_provenance,
        required_fields=required_fields,
        pending_missing_fields=pending_clarification.missing_fields,
        correction_intent=correction_intent,
    )
    merge_provenance["profile_fields"] = profile_provenance

    (
        merged_hard_constraints,
        hard_constraint_provenance,
        carried_constraint_keys,
        hard_constraint_change_detected,
    ) = merge_hard_constraints(
        prior_hard_constraints=prior_hard_constraints,
        followup_hard_constraints=followup_hard_constraints,
        required_fields=required_fields,
        correction_intent=correction_intent,
    )
    merge_provenance["hard_constraints"] = hard_constraint_provenance

    (
        merged_soft_preferences,
        soft_preference_provenance,
        carried_preference_keys,
        soft_preference_change_detected,
    ) = merge_soft_preferences(
        prior_soft_preferences=prior_soft_preferences,
        followup_soft_preferences=followup_soft_preferences,
    )
    merge_provenance["soft_preferences"] = soft_preference_provenance

    merged_constraint_spec = combine_constraint_buckets(
        hard_constraints=merged_hard_constraints,
        soft_preferences=merged_soft_preferences,
    )
    normalized_constraint_spec, constraint_errors = validate_and_normalize_constraint_spec(
        merged_constraint_spec or None,
        feature_order=required_fields,
    )
    if constraint_errors:
        followup_conflicts.extend(constraint_errors)
        for error in constraint_errors:
            merge_provenance.setdefault("constraint_errors", []).append(str(error))
    if normalized_constraint_spec is None:
        merged_constraint_spec = {}
    else:
        merged_constraint_spec = dict(normalized_constraint_spec)
        merged_hard_constraints, merged_soft_preferences = split_constraint_buckets(
            merged_constraint_spec,
            feature_order=required_fields,
        )

    conflicts = _dedupe_strings(followup_conflicts + profile_conflicts)
    missing_fields = [field for field in required_fields if field not in merged_profile]
    merged_candidate = {
        "task": followup_task,
        "status": determine_candidate_status(conflicts=conflicts, missing_fields=missing_fields),
        "cf_request": merged_profile,
        "missing_fields": missing_fields,
        "conflicts": conflicts,
        "notes": followup_notes,
    }
    if merged_constraint_spec:
        merged_candidate["constraint_spec"] = merged_constraint_spec

    merged_field_provenance = {
        field_name: str(value)
        for field_name, value in prior_field_provenance.items()
        if field_name in merged_profile and isinstance(value, str)
    }
    for field_name, value in effective_field_provenance.items():
        if field_name in merged_profile and isinstance(value, str):
            merged_field_provenance[field_name] = value

    followup_classification = classify_followup(
        profile_correction_detected=profile_correction_detected,
        profile_fill_detected=profile_fill_detected,
        hard_constraint_change_detected=hard_constraint_change_detected,
        soft_preference_change_detected=soft_preference_change_detected,
    )
    merge_applied = determine_merge_applied(
        carried_fields=carried_fields,
        carried_constraint_keys=carried_constraint_keys,
        carried_preference_keys=carried_preference_keys,
        profile_change_detected=profile_change_detected,
        hard_constraint_change_detected=hard_constraint_change_detected,
        soft_preference_change_detected=soft_preference_change_detected,
    )

    return FollowupMergeDecision(
        candidate=merged_candidate,
        field_provenance=merged_field_provenance,
        followup_classification=followup_classification,
        merge_provenance=merge_provenance,
        carried_fields=carried_fields,
        carried_constraint_keys=carried_constraint_keys,
        carried_preference_keys=carried_preference_keys,
        pending_reset=False,
        reset_decision="none",
        merge_applied=merge_applied,
    )


def augment_followup_candidate_with_explicit_bank_values(
    *,
    user_input: str,
    candidate: dict[str, Any],
    policy,
    required_fields: list[str],
    field_provenance: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    effective_field_provenance = dict(field_provenance or {})
    if policy.dataset_name != "bank":
        return candidate, effective_field_provenance
    if candidate.get("status") == "conflict":
        return candidate, effective_field_provenance
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return candidate, effective_field_provenance

    extraction = extract_explicit_bank_values(
        user_input=user_input,
        policy=policy,
        target_fields=required_fields,
    )
    if not extraction.values and not extraction.conflicts:
        return candidate, effective_field_provenance
    augmented_candidate, effective_field_provenance = merge_bank_candidate_with_explicit_values(
        candidate=candidate,
        extracted_values=extraction.values,
        extracted_conflicts=extraction.conflicts,
        conflict_fields=extraction.conflict_fields,
        required_fields=required_fields,
        base_field_provenance=effective_field_provenance,
    )
    for field_name in cf_request:
        effective_field_provenance.setdefault(field_name, FIELD_PROVENANCE_PARSER)
    return augmented_candidate, effective_field_provenance


def build_merge_source(
    *,
    canonical_session_state: dict[str, Any] | None,
    pending_clarification: PendingClarification,
    required_fields: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    profile_facts = {}
    hard_constraints = {}
    soft_preferences = {}
    if isinstance(canonical_session_state, dict):
        profile_facts = {
            field_name: canonical_session_state.get("profile_facts", {}).get(field_name)
            for field_name in required_fields
            if field_name in dict(canonical_session_state.get("profile_facts") or {})
        }
        hard_constraints = {
            str(key): value
            for key, value in dict(canonical_session_state.get("hard_constraints") or {}).items()
            if isinstance(key, str)
        }
        soft_preferences = {
            str(key): value
            for key, value in dict(canonical_session_state.get("soft_preferences") or {}).items()
            if isinstance(key, str)
        }
    if not profile_facts:
        profile_facts = {
            field_name: pending_clarification.prior_cf_request[field_name]
            for field_name in required_fields
            if field_name in pending_clarification.prior_cf_request
        }
    if not hard_constraints and not soft_preferences:
        hard_constraints, soft_preferences = split_constraint_buckets(
            pending_clarification.prior_constraint_spec,
            feature_order=required_fields,
        )
    return (
        profile_facts,
        hard_constraints,
        soft_preferences,
        {
            str(field_name): str(value)
            for field_name, value in dict(pending_clarification.prior_field_provenance or {}).items()
            if isinstance(field_name, str) and isinstance(value, str)
        },
    )


def merge_profile_facts(
    *,
    prior_profile: dict[str, Any],
    followup_request: dict[str, Any],
    followup_field_provenance: dict[str, str],
    prior_field_provenance: dict[str, str],
    required_fields: list[str],
    pending_missing_fields: list[str],
    correction_intent: bool,
) -> tuple[dict[str, Any], dict[str, str], list[str], bool, bool, bool, list[str]]:
    merged_profile: dict[str, Any] = {}
    merge_provenance: dict[str, str] = {}
    carried_fields: list[str] = []
    profile_change_detected = False
    profile_correction_detected = False
    profile_fill_detected = False
    conflicts: list[str] = []
    pending_missing = set(pending_missing_fields)

    for field_name in required_fields:
        prior_has = field_name in prior_profile
        followup_has = field_name in followup_request
        if followup_has:
            followup_value = followup_request[field_name]
            if not prior_has:
                merged_profile[field_name] = followup_value
                merge_provenance[field_name] = MERGE_PROVENANCE_FILLED
                profile_change_detected = True
                profile_fill_detected = field_name in pending_missing or profile_fill_detected
                continue
            prior_value = prior_profile[field_name]
            if prior_value == followup_value:
                merged_profile[field_name] = prior_value
                merge_provenance[field_name] = (
                    MERGE_PROVENANCE_DUPLICATE
                    if field_name in followup_field_provenance
                    else MERGE_PROVENANCE_CARRIED
                )
                carried_fields.append(field_name)
                continue
            if correction_intent:
                merged_profile[field_name] = followup_value
                merge_provenance[field_name] = MERGE_PROVENANCE_CORRECTED
                profile_change_detected = True
                profile_correction_detected = True
                continue
            merged_profile[field_name] = prior_value
            merge_provenance[field_name] = MERGE_PROVENANCE_CONFLICT
            carried_fields.append(field_name)
            conflicts.append(
                "Follow-up field '{0}' conflicts with prior canonical state without explicit correction intent.".format(
                    field_name
                )
            )
            continue
        if prior_has:
            merged_profile[field_name] = prior_profile[field_name]
            merge_provenance[field_name] = MERGE_PROVENANCE_CARRIED
            carried_fields.append(field_name)

    for field_name, provenance in prior_field_provenance.items():
        if field_name in merged_profile and field_name not in followup_field_provenance:
            followup_field_provenance[field_name] = provenance

    return (
        merged_profile,
        merge_provenance,
        _dedupe_strings(carried_fields),
        profile_change_detected,
        profile_correction_detected,
        profile_fill_detected,
        _dedupe_strings(conflicts),
    )


def merge_hard_constraints(
    *,
    prior_hard_constraints: dict[str, Any],
    followup_hard_constraints: dict[str, Any],
    required_fields: list[str],
    correction_intent: bool,
) -> tuple[dict[str, Any], dict[str, str], list[str], bool]:
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    carried_keys: list[str] = []
    change_detected = False

    prior_blocked = _normalize_feature_list(prior_hard_constraints.get("disallowed_changes"))
    followup_blocked = _normalize_feature_list(followup_hard_constraints.get("disallowed_changes"))
    if followup_blocked:
        blocked_values = list(followup_blocked) if correction_intent else ordered_union(prior_blocked, followup_blocked)
        for value in blocked_values:
            key_path = "disallowed_changes.{0}".format(value)
            if value in followup_blocked and value not in prior_blocked:
                provenance[key_path] = MERGE_PROVENANCE_CONSTRAINT
                change_detected = True
            elif value in followup_blocked:
                provenance[key_path] = MERGE_PROVENANCE_DUPLICATE
            else:
                provenance[key_path] = MERGE_PROVENANCE_CARRIED
                carried_keys.append(key_path)
        if blocked_values:
            merged["disallowed_changes"] = blocked_values
    elif prior_blocked:
        merged["disallowed_changes"] = list(prior_blocked)
        for value in prior_blocked:
            key_path = "disallowed_changes.{0}".format(value)
            provenance[key_path] = MERGE_PROVENANCE_CARRIED
            carried_keys.append(key_path)

    prior_numeric_bounds = _normalize_numeric_bounds(prior_hard_constraints.get("numeric_bounds"))
    followup_numeric_bounds = _normalize_numeric_bounds(followup_hard_constraints.get("numeric_bounds"))
    merged_numeric_bounds: dict[str, dict[str, float]] = {}
    for field_name in sorted(set(prior_numeric_bounds) | set(followup_numeric_bounds)):
        base_bounds = {} if correction_intent and field_name in followup_numeric_bounds else dict(prior_numeric_bounds.get(field_name) or {})
        followup_bounds = dict(followup_numeric_bounds.get(field_name) or {})
        for bound_key, bound_value in followup_bounds.items():
            key_path = "numeric_bounds.{0}.{1}".format(field_name, bound_key)
            if base_bounds.get(bound_key) == bound_value:
                provenance[key_path] = MERGE_PROVENANCE_DUPLICATE
            else:
                provenance[key_path] = MERGE_PROVENANCE_CONSTRAINT
                change_detected = True
            base_bounds[bound_key] = bound_value
        for bound_key in sorted(base_bounds):
            key_path = "numeric_bounds.{0}.{1}".format(field_name, bound_key)
            provenance.setdefault(key_path, MERGE_PROVENANCE_CARRIED)
            if provenance[key_path] == MERGE_PROVENANCE_CARRIED:
                carried_keys.append(key_path)
        if base_bounds:
            merged_numeric_bounds[field_name] = base_bounds
    if merged_numeric_bounds:
        merged["numeric_bounds"] = merged_numeric_bounds

    prior_max_changes = prior_hard_constraints.get("max_changed_features")
    followup_max_changes = followup_hard_constraints.get("max_changed_features")
    if followup_max_changes is not None:
        merged["max_changed_features"] = followup_max_changes
        if followup_max_changes == prior_max_changes:
            provenance["max_changed_features"] = MERGE_PROVENANCE_DUPLICATE
        else:
            provenance["max_changed_features"] = MERGE_PROVENANCE_CONSTRAINT
            change_detected = True
    elif prior_max_changes is not None:
        merged["max_changed_features"] = prior_max_changes
        provenance["max_changed_features"] = MERGE_PROVENANCE_CARRIED
        carried_keys.append("max_changed_features")

    for key in ("immutable",):
        prior_list = _normalize_feature_list(prior_hard_constraints.get(key))
        followup_list = _normalize_feature_list(followup_hard_constraints.get(key))
        if followup_list:
            list_values = list(followup_list) if correction_intent else ordered_union(prior_list, followup_list)
            if list_values:
                merged[key] = list_values
            for value in list_values:
                key_path = "{0}.{1}".format(key, value)
                if value in followup_list and value not in prior_list:
                    provenance[key_path] = MERGE_PROVENANCE_CONSTRAINT
                    change_detected = True
                elif value in followup_list:
                    provenance[key_path] = MERGE_PROVENANCE_DUPLICATE
                else:
                    provenance[key_path] = MERGE_PROVENANCE_CARRIED
                    carried_keys.append(key_path)
        elif prior_list:
            merged[key] = list(prior_list)
            for value in prior_list:
                key_path = "{0}.{1}".format(key, value)
                provenance[key_path] = MERGE_PROVENANCE_CARRIED
                carried_keys.append(key_path)

    normalized_merged = {
        key: value
        for key, value in merged.items()
        if value not in (None, [], {})
    }
    return normalized_merged, provenance, _dedupe_strings(carried_keys), change_detected


def merge_soft_preferences(
    *,
    prior_soft_preferences: dict[str, Any],
    followup_soft_preferences: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], list[str], bool]:
    merged = dict(prior_soft_preferences)
    provenance: dict[str, str] = {}
    carried_keys: list[str] = []
    change_detected = False
    for key in sorted(set(prior_soft_preferences) | set(followup_soft_preferences)):
        prior_present = key in prior_soft_preferences
        followup_present = key in followup_soft_preferences
        if followup_present:
            followup_value = followup_soft_preferences[key]
            if prior_present and prior_soft_preferences[key] == followup_value:
                merged[key] = followup_value
                provenance[key] = MERGE_PROVENANCE_DUPLICATE
                carried_keys.append(key)
            else:
                merged[key] = followup_value
                provenance[key] = MERGE_PROVENANCE_PREFERENCE
                change_detected = True
            continue
        if prior_present:
            merged[key] = prior_soft_preferences[key]
            provenance[key] = MERGE_PROVENANCE_CARRIED
            carried_keys.append(key)
    return merged, provenance, _dedupe_strings(carried_keys), change_detected


def classify_followup(
    *,
    profile_correction_detected: bool,
    profile_fill_detected: bool,
    hard_constraint_change_detected: bool,
    soft_preference_change_detected: bool,
) -> str:
    if profile_correction_detected:
        return FOLLOWUP_CLASSIFICATION_CORRECTION
    if profile_fill_detected:
        return FOLLOWUP_CLASSIFICATION_PROFILE_COMPLETION
    if hard_constraint_change_detected:
        return FOLLOWUP_CLASSIFICATION_CONSTRAINT_UPDATE
    if soft_preference_change_detected:
        return FOLLOWUP_CLASSIFICATION_PREFERENCE_UPDATE
    return FOLLOWUP_CLASSIFICATION_AMBIGUOUS


def determine_merge_applied(
    *,
    carried_fields: list[str],
    carried_constraint_keys: list[str],
    carried_preference_keys: list[str],
    profile_change_detected: bool,
    hard_constraint_change_detected: bool,
    soft_preference_change_detected: bool,
) -> bool:
    prior_contributed = bool(carried_fields or carried_constraint_keys or carried_preference_keys)
    new_change_detected = bool(
        profile_change_detected or hard_constraint_change_detected or soft_preference_change_detected
    )
    return prior_contributed and new_change_detected


def has_full_profile_restatement(*, candidate: dict[str, Any], required_fields: list[str]) -> bool:
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return False
    return all(field_name in cf_request for field_name in required_fields)


def is_harmless_restatement_answer(
    *,
    prior_profile: dict[str, Any],
    followup_request: dict[str, Any],
    pending_missing_fields: list[str],
) -> bool:
    missing_fields = [field_name for field_name in pending_missing_fields if field_name not in followup_request]
    if missing_fields:
        return False
    pending_missing = set(pending_missing_fields)
    filled_any_missing = False
    for field_name, followup_value in followup_request.items():
        if field_name in pending_missing:
            filled_any_missing = True
            continue
        if field_name in prior_profile and prior_profile[field_name] != followup_value:
            return False
    return filled_any_missing


def is_empty_nonanswer_followup(
    *,
    candidate: dict[str, Any],
    followup_hard_constraints: dict[str, Any],
    followup_soft_preferences: dict[str, Any],
) -> bool:
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict) or cf_request:
        return False
    if followup_hard_constraints or followup_soft_preferences:
        return False
    conflicts = candidate.get("conflicts")
    notes = candidate.get("notes")
    if isinstance(conflicts, list) and conflicts:
        return False
    if isinstance(notes, list) and notes:
        return False
    return True


def has_any_cue(*, user_input: str, cues: tuple[str, ...]) -> bool:
    normalized_text = " ".join(user_input.lower().split())
    return any(cue in normalized_text for cue in cues)


def determine_candidate_status(*, conflicts: list[str], missing_fields: list[str]) -> str:
    if conflicts:
        return "conflict"
    return "complete" if not missing_fields else "partial"


def build_partial_snapshot(
    *,
    candidate: dict[str, Any] | None,
    ordered_fields: list[str],
) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return None
    return {field: cf_request[field] for field in ordered_fields if field in cf_request}


def build_builder_provenance(
    *,
    parser_status: str | None,
    pending_clarification: PendingClarification | None,
    pending_reset: bool,
    parser_quality: dict[str, Any] | None = None,
    field_provenance: dict[str, str] | None = None,
    followup_classification: str | None = None,
    reset_decision: str = "none",
    merge_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "parser_status": parser_status,
        "pending_clarification_present": pending_clarification is not None,
        "pending_reset": bool(pending_reset),
        "reset_decision": reset_decision,
    }
    if followup_classification is not None:
        provenance["followup_classification"] = followup_classification
    if merge_provenance:
        provenance["merge_provenance"] = dict(merge_provenance)
    provenance["parser_quality"] = normalize_parser_quality_payload(parser_quality)
    if field_provenance:
        provenance["field_provenance"] = {
            str(field_name): str(value)
            for field_name, value in field_provenance.items()
            if isinstance(field_name, str) and isinstance(value, str)
        }
    return provenance


def extract_parser_status(candidate: dict[str, Any] | None) -> str | None:
    if not isinstance(candidate, dict):
        return None
    status = candidate.get("status")
    return status if isinstance(status, str) else None


def is_unsupported_intent(
    *,
    user_input: str,
    candidate: dict[str, Any] | None,
) -> bool:
    lowered = " ".join(user_input.lower().split())
    has_profile_signal = False
    if isinstance(candidate, dict) and isinstance(candidate.get("cf_request"), dict):
        has_profile_signal = bool(candidate.get("cf_request"))
    if has_profile_signal:
        return False

    if any(keyword in lowered for keyword in OPTIMIZATION_KEYWORDS):
        return True
    if any(keyword in lowered for keyword in ADVICE_KEYWORDS):
        return True
    return False


def detect_explicit_dataset_switch(
    *,
    user_input: str,
    dataset_id: str = "bank",
    supported_dataset_ids: list[str] | None = None,
) -> dict[str, str] | None:
    lowered = " ".join(user_input.lower().split())
    if not _has_explicit_dataset_switch_cue(lowered):
        return None

    active_dataset_id = str(dataset_id or "bank").strip().lower() or "bank"
    active_supported_dataset_ids = list(supported_dataset_ids or DATASET_KEYWORD_MAP)
    for known_dataset_id in active_supported_dataset_ids:
        if known_dataset_id == active_dataset_id:
            continue
        if any(keyword in lowered for keyword in DATASET_KEYWORD_MAP.get(known_dataset_id, ())):
            return {
                "requested_dataset_id": known_dataset_id,
                "requested_dataset_label": DATASET_LABEL_MAP.get(known_dataset_id, known_dataset_id),
            }

    requested_unsupported_keyword = next(
        (keyword for keyword in UNSUPPORTED_DATASET_KEYWORDS if keyword in lowered),
        None,
    )
    if requested_unsupported_keyword is not None:
        return {
            "requested_dataset_id": "unsupported_dataset",
            "requested_dataset_label": requested_unsupported_keyword,
        }
    return None


def _has_explicit_dataset_switch_cue(lowered: str) -> bool:
    if any(pattern.search(lowered) for pattern in EXPLICIT_DATASET_SWITCH_PATTERNS):
        return True
    return "dataset" in lowered and any(verb in lowered for verb in DATASET_SWITCH_VERBS)


def _build_unsupported_request_result(
    *,
    effective_candidate: dict[str, Any] | None,
    effective_canonical: CanonicalValidationResult,
    effective_field_provenance: dict[str, str],
    effective_schema: ValidationResult,
    merge_applied: bool,
    carried_fields: list[str],
    carried_constraint_keys: list[str],
    carried_preference_keys: list[str],
    pending_reset: bool,
    provenance: dict[str, Any],
    active_policy,
    active_required_fields: list[str],
) -> RequestBuildResult:
    return RequestBuildResult(
        builder_status=ConversationStage.UNSUPPORTED_REQUEST,
        builder_reason_codes=[UNSUPPORTED_INTENT],
        partial_profile_snapshot=build_partial_snapshot(
            candidate=effective_candidate,
            ordered_fields=active_required_fields,
        ),
        runtime_request=None,
        missing_fields=[],
        conflicts=[],
        policy_version=active_policy.policy_version,
        canonical_field_order=list(active_required_fields),
        provenance=provenance,
        merge_applied=merge_applied,
        carried_fields=carried_fields,
        carried_constraint_keys=carried_constraint_keys,
        carried_preference_keys=carried_preference_keys,
        pending_reset=pending_reset,
        _normalized_candidate=effective_candidate,
        _schema_validation=effective_schema.to_dict(),
        _canonical_validation=effective_canonical,
    )


def ordered_union(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in list(left) + list(right):
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _normalize_feature_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_strings([str(item) for item in value if isinstance(item, str)])


def _normalize_numeric_bounds(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, float]] = {}
    for field_name, bounds in value.items():
        if not isinstance(field_name, str) or not isinstance(bounds, dict):
            continue
        normalized_bounds: dict[str, float] = {}
        for bound_key, bound_value in bounds.items():
            if bound_key not in {"min", "max"}:
                continue
            if isinstance(bound_value, bool) or not isinstance(bound_value, (int, float)):
                continue
            normalized_bounds[bound_key] = float(bound_value)
        if normalized_bounds:
            normalized[field_name] = normalized_bounds
    return normalized


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_strings([str(item) for item in value if isinstance(item, str)])


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        clean = " ".join(str(value).split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered
