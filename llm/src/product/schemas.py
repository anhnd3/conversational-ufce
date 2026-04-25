from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class ProductBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactRefBundle(ProductBaseModel):
    turn_id: str
    artifact_dir: Optional[str]
    files: List[str]
    download_urls: Dict[str, str]
    preview_urls: Dict[str, str]


class RuntimeSummary(ProductBaseModel):
    executed: bool
    controller_state: Optional[str]
    reason_codes: List[str]
    prediction_score: Optional[float]


class DebugSummary(ProductBaseModel):
    builder_status: Optional[str]
    builder_reason_codes: List[str]
    transition_reason: Optional[str]
    merge_applied: bool
    runtime_summary: RuntimeSummary
    invariant_validation_status: Optional[str]
    artifact_dir: Optional[str]
    timing_metrics: Optional[Dict[str, Any]] = None


class SessionSummary(ProductBaseModel):
    session_id: str
    dataset_key: str
    created_at: str
    updated_at: str
    current_public_state: Optional[str]
    last_turn_index: int
    has_pending_clarification: bool
    lifecycle_status: str
    archived_at: Optional[str]
    is_read_only: bool
    clarification_turns_used: int
    is_case_complete: bool
    case_completion_reason: Optional[str]
    restart_required: bool
    active_constraint_spec: Optional[Dict[str, Any]]
    refinement_revision_index: int
    refinement_rounds_used: int
    refinement_round_limit: int
    has_pending_refinement_clarification: bool
    refinement_allowed: bool
    latest_runtime_backed_turn_id: Optional[str]


class SessionDetail(SessionSummary):
    turn_count: int
    latest_turn_id: Optional[str]
    artifact_root: str
    ui_review: Optional["UiReviewPayload"] = None
    render_hints: Optional["SessionRenderHints"] = None


class UiReviewField(ProductBaseModel):
    field_name: str
    label: str
    value: Optional[Any]
    display_value: str
    feature_kind: str
    missing: bool
    editable: bool
    step: Optional[float] = None


class UiReviewConstraint(ProductBaseModel):
    key: str
    label: str
    value: Optional[Any]
    display_value: str
    kind: str
    field_name: Optional[str] = None
    editable: bool = False


class UiReviewPayload(ProductBaseModel):
    display_state: str
    profile_fields: List[UiReviewField]
    constraints: List[UiReviewConstraint]
    preferences: List[UiReviewConstraint]
    missing_fields: List[str]
    clarification_type: Optional[str]
    clarification_message: Optional[str]
    clarification_next_input: Optional[str]
    remaining_rounds: Optional[int]
    refinement_status: Optional[str]
    last_updated_turn_id: Optional[str]
    read_only: bool
    profile_editable: bool
    refinement_editable: bool


class ComposerContextPayload(ProductBaseModel):
    mode: str
    submit_target: Optional[str]
    mode_chip_text: Optional[str]
    title: Optional[str]
    help_text: Optional[str]
    placeholder: Optional[str]
    button_label: Optional[str]
    advanced_controls_relevant: bool
    hidden: bool
    disabled: bool


class TurnRenderHints(ProductBaseModel):
    primary_chat_text: str
    primary_action_type: str
    primary_action_items: List[str]
    supporting_detail_title: Optional[str]
    supporting_detail_body: Optional[str]
    supporting_detail_facts: List[str]
    state_marker_label: Optional[str]
    right_rail_anchor: Optional[str]


class SessionRenderHints(TurnRenderHints):
    page_state: str
    composer_mode: str
    composer_context: ComposerContextPayload


class TurnResponse(ProductBaseModel):
    session_id: str
    turn_id: str
    turn_index: int
    user_input: str
    assistant_text: str
    public_state: str
    clarification_payload: Optional[Dict[str, Any]]
    explanation_payload: Optional[Dict[str, Any]]
    artifact_refs: ArtifactRefBundle
    debug_summary: DebugSummary
    clarification_turns_used: int
    is_case_complete: bool
    case_completion_reason: Optional[str]
    restart_required: bool
    turn_kind: str
    refinement_status: Optional[str]
    refinement_revision_index: Optional[int]
    parent_terminal_turn_id: Optional[str]
    parent_refinement_revision_index: Optional[int]
    active_constraint_spec: Optional[Dict[str, Any]]
    constraint_feedback_delta: Optional[Dict[str, Any]]
    refinement_rounds_used: Optional[int]
    refinement_round_limit: Optional[int]
    ui_review: Optional[UiReviewPayload] = None
    render_hints: Optional[TurnRenderHints] = None


class ArtifactBundle(ProductBaseModel):
    turn_id: str
    public_state: str
    files: List[str]
    download_urls: Dict[str, str]
    preview_urls: Dict[str, str]


class HealthCheck(ProductBaseModel):
    ok: bool
    detail: str


class HealthChecks(ProductBaseModel):
    database: HealthCheck
    artifact_store: HealthCheck
    lm_studio: HealthCheck


class HealthResponse(ProductBaseModel):
    status: str
    checks: HealthChecks


class VersionResponse(ProductBaseModel):
    api_version: str
    app_version: str
    model_alias: str
    parser_schema_version: str
    bank_policy_version: str
    runtime_mode: str
    git_commit: Optional[str]


class MessageCreateRequest(ProductBaseModel):
    user_input: str


class SessionCreateRequest(ProductBaseModel):
    dataset_key: str = "bank"


class SessionBlockedResponse(ProductBaseModel):
    error_code: str
    detail: str
    current_public_state: Optional[str]
    case_completion_reason: Optional[str]
    restart_required: bool


class RefinementCreateRequest(ProductBaseModel):
    user_feedback: str


class RefinementBlockedResponse(ProductBaseModel):
    error_code: str
    detail: str
    current_public_state: Optional[str]
    case_completion_reason: Optional[str]
    active_constraint_spec: Dict[str, Any]
    refinement_revision_index: int
    refinement_rounds_used: int
    refinement_round_limit: int
    restart_required: bool
    refinement_status: Optional[str] = None


class DatasetFeatureGuide(ProductBaseModel):
    feature_name: str
    feature_kind: str
    changeable: bool
    step: Optional[Union[float, int]]
    aliases: List[str]
    definition: str
    check_guidance: str
    change_guidance: str


class DatasetCatalogEntry(ProductBaseModel):
    dataset_key: str
    display_name: str
    availability_status: str
    support_note: str
    artifact_version: str
    training_logic_version: str
    full_feature_list: List[str]
    f2change: List[str]
    outcome_label: str
    desired_outcome: float
    step_provenance: str
    feature_guides: List[DatasetFeatureGuide]
