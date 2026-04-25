from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llm.src.conversation.canonical_session_state import split_constraint_buckets
from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.conversation.parser_adapter import LiveLmStudioParserAdapter
from llm.src.conversation.session import SessionCaseCompleteError
from llm.src.product.config import ProductConfig
from llm.src.product.persistence import SessionRepository
from llm.src.product.schemas import (
    ArtifactBundle,
    ArtifactRefBundle,
    ComposerContextPayload,
    DatasetCatalogEntry,
    HealthChecks,
    HealthResponse,
    MessageCreateRequest,
    RefinementBlockedResponse,
    RefinementCreateRequest,
    SessionRenderHints,
    SessionCreateRequest,
    SessionDetail,
    SessionSummary,
    TurnResponse,
    TurnRenderHints,
    DebugSummary,
    RuntimeSummary,
    SessionBlockedResponse,
    UiReviewConstraint,
    UiReviewField,
    UiReviewPayload,
    VersionResponse,
)
from llm.src.product.service import (
    ProductSessionService,
    RefinementLimitReachedError,
    RefinementNotAllowedError,
    SessionArchivedError,
)
from llm.src.runtime.datasets.bank.metadata import (
    BANK_BOOLEAN_FIELDS,
    BANK_FEATURE_TYPES,
    BANK_REQUIRED_FIELD_ORDER,
    BANK_STEP,
)
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.reason_codes import INVALID_COUNTERFACTUAL_BLOCKED, REQUEST_CONSTRAINTS_BLOCKED


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNTIME_COMPLETE_STATES = frozenset({"RUNTIME_SUCCESS", "RUNTIME_REJECT"})
PRIMARY_ACTION_START_CASE = "start_case"
PRIMARY_ACTION_PROVIDE_MISSING_FIELDS = "provide_missing_fields"
PRIMARY_ACTION_NO_ACTION_REQUIRED = "no_action_required"
PRIMARY_ACTION_RELAX_CONSTRAINTS_OR_RESTART = "relax_constraints_or_restart"
PRIMARY_ACTION_CLARIFY_REFINEMENT = "clarify_refinement"
PRIMARY_ACTION_START_NEW_CASE = "start_new_case"
PRIMARY_ACTION_NONE = "none"
RIGHT_RAIL_REVIEW = "review-card"
RIGHT_RAIL_RESULT = "result-card"
RIGHT_RAIL_ADVANCED_CONTROLS = "advanced-refinement-controls"
RIGHT_RAIL_TECHNICAL = "technical-drawer"
COMPOSER_MODE_MESSAGE = "message"
COMPOSER_MODE_REFINEMENT = "refinement"
COMPOSER_MODE_DISABLED = "disabled"
SESSION_PAGE_STATE_LABELS = {
    "fresh": "fresh",
    "clarification": "needs clarification",
    "runtime_success": "runtime success",
    "runtime_reject": "runtime reject",
    "refinement_clarification": "refinement clarification",
    "restart_required": "restart required",
}


def create_app(
    config: ProductConfig | None = None,
    *,
    orchestrator: BankConversationOrchestrator | None = None,
    repository: SessionRepository | None = None,
    service: ProductSessionService | None = None,
) -> FastAPI:
    active_config = config or ProductConfig.load()
    active_repository = repository or SessionRepository(active_config.sqlite_path, app_version=active_config.app_version)

    if service is None:
        if orchestrator is None:
            parser_adapter = LiveLmStudioParserAdapter(
                model_alias=active_config.model_alias,
                api_base=active_config.lm_studio_api_base,
            )
            runtime_orchestrator = RuntimeOrchestrator(runtime_mode=active_config.product_mode)
            orchestrator = BankConversationOrchestrator(
                parser_adapter=parser_adapter,
                runtime_orchestrator=runtime_orchestrator,
                output_root=active_config.artifact_root,
                model_alias=active_config.model_alias,
            )
        service = ProductSessionService(
            orchestrator=orchestrator,
            repository=active_repository,
            config=active_config,
        )

    app = FastAPI(title="UFCE Agent Product MVP", version=active_config.app_version)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.state.config = active_config
    app.state.service = service

    api_prefix = f"/api/{active_config.api_version}"

    @app.get(f"{api_prefix}/sessions", response_model=List[SessionSummary])
    async def list_sessions():
        sessions = app.state.service.list_sessions()
        return [serialize_session_summary(item) for item in sessions]

    @app.post(f"{api_prefix}/sessions", response_model=SessionSummary)
    async def create_session(payload: Optional[SessionCreateRequest] = None):
        try:
            session = app.state.service.create_session(
                dataset_key="bank" if payload is None else payload.dataset_key
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return serialize_session_summary(session)

    @app.get(f"{api_prefix}/sessions/{{session_id}}", response_model=SessionDetail)
    async def get_session(session_id: str):
        try:
            session = app.state.service.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        turns = app.state.service.list_messages(session_id, order="desc")
        turn_payloads = [app.state.service.build_turn_response(turn) for turn in turns]
        latest_turn_payload = None if not turn_payloads else turn_payloads[0]
        return serialize_session_detail(
            session,
            turn_count=len(turn_payloads),
            artifact_root=app.state.config.artifact_root,
            latest_turn_payload=latest_turn_payload,
            turn_payloads=turn_payloads,
        )

    @app.get(f"{api_prefix}/sessions/{{session_id}}/messages", response_model=List[TurnResponse])
    async def list_messages(session_id: str, order: str = "desc"):
        try:
            turns = app.state.service.list_messages(session_id, order=order)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return [serialize_turn_response(app.state.service.build_turn_response(turn)) for turn in turns]

    @app.post(
        f"{api_prefix}/sessions/{{session_id}}/messages",
        response_model=TurnResponse,
        responses={409: {"model": SessionBlockedResponse}},
    )
    async def submit_message(session_id: str, payload: MessageCreateRequest):
        try:
            turn = app.state.service.submit_message(session_id, payload.user_input)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except SessionArchivedError as exc:
            session = app.state.service.get_session(session_id)
            return JSONResponse(
                status_code=409,
                content=SessionBlockedResponse(
                    error_code="session_archived",
                    detail=str(exc),
                    current_public_state=session.current_public_state,
                    case_completion_reason=session.case_completion_reason,
                    restart_required=session.restart_required,
                ).model_dump(),
            )
        except SessionCaseCompleteError as exc:
            session = app.state.service.get_session(session_id)
            return JSONResponse(
                status_code=409,
                content=SessionBlockedResponse(
                    error_code="case_complete",
                    detail=str(exc),
                    current_public_state=session.current_public_state,
                    case_completion_reason=session.case_completion_reason,
                    restart_required=session.restart_required,
                ).model_dump(),
            )
        return serialize_turn_response(app.state.service.build_turn_response(turn))

    @app.post(
        f"{api_prefix}/sessions/{{session_id}}/refinements",
        response_model=TurnResponse,
        responses={409: {"model": RefinementBlockedResponse}},
    )
    async def submit_refinement(session_id: str, payload: RefinementCreateRequest):
        try:
            turn = app.state.service.submit_refinement(session_id, payload.user_feedback)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except SessionArchivedError as exc:
            session = app.state.service.get_session(session_id)
            return JSONResponse(
                status_code=409,
                content=build_refinement_blocked_response(
                    session,
                    error_code="session_archived",
                    detail=str(exc),
                ).model_dump(),
            )
        except RefinementNotAllowedError as exc:
            session = app.state.service.get_session(session_id)
            return JSONResponse(
                status_code=409,
                content=build_refinement_blocked_response(
                    session,
                    error_code="refinement_not_allowed",
                    detail=str(exc),
                ).model_dump(),
            )
        except RefinementLimitReachedError as exc:
            session = app.state.service.get_session(session_id)
            return JSONResponse(
                status_code=409,
                content=build_refinement_blocked_response(
                    session,
                    error_code="refinement_limit_reached",
                    detail=str(exc),
                    refinement_status="limit_reached",
                    restart_required=True,
                ).model_dump(),
            )
        return serialize_turn_response(app.state.service.build_turn_response(turn))

    @app.post(f"{api_prefix}/sessions/{{session_id}}/archive", response_model=SessionSummary)
    async def archive_session(session_id: str):
        try:
            session = app.state.service.archive_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return serialize_session_summary(session)

    @app.get(f"{api_prefix}/sessions/{{session_id}}/artifacts", response_model=List[ArtifactBundle])
    async def list_artifacts(session_id: str):
        try:
            bundles = app.state.service.get_artifact_bundles(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return [ArtifactBundle(**bundle) for bundle in bundles]

    @app.get(f"{api_prefix}/sessions/{{session_id}}/artifacts/{{turn_id}}/{{filename}}")
    async def get_artifact_file(session_id: str, turn_id: str, filename: str):
        try:
            resolved = app.state.service.resolve_artifact_path(session_id, turn_id, filename)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return Response(
            content=resolved.read_bytes(),
            media_type="application/octet-stream",
            headers={"content-disposition": f'attachment; filename="{resolved.name}"'},
        )

    @app.get(f"{api_prefix}/sessions/{{session_id}}/artifacts/{{turn_id}}/{{filename}}/preview")
    async def get_artifact_preview(session_id: str, turn_id: str, filename: str):
        try:
            payload = app.state.service.get_artifact_preview(session_id, turn_id, filename)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return JSONResponse(content=payload)

    @app.get(f"{api_prefix}/health", response_model=HealthResponse)
    async def health():
        payload = app.state.service.health()
        return HealthResponse(
            status=payload["status"],
            checks=HealthChecks(**payload["checks"]),
        )

    @app.get(f"{api_prefix}/version", response_model=VersionResponse)
    async def version():
        return VersionResponse(**app.state.service.version())

    @app.get(f"{api_prefix}/catalog/datasets", response_model=List[DatasetCatalogEntry])
    async def list_dataset_catalog():
        entries = app.state.service.list_dataset_catalog()
        return [DatasetCatalogEntry(**entry) for entry in entries]

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        sessions = [serialize_session_summary(item).model_dump() for item in app.state.service.list_sessions()]
        datasets = [DatasetCatalogEntry(**item).model_dump() for item in app.state.service.list_dataset_catalog()]
        selected_dataset = _select_default_dataset(datasets)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "sessions": sessions[:6],
                "datasets": datasets,
                "selected_dataset": selected_dataset,
                "api_prefix": api_prefix,
            },
        )

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_page(request: Request, session_id: str):
        try:
            session = app.state.service.get_session(session_id)
            turns = app.state.service.list_messages(session_id, order="desc")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        turn_payloads = [app.state.service.build_turn_response(turn) for turn in turns]
        latest_turn_payload = None if not turn_payloads else turn_payloads[0]
        session_detail = serialize_session_detail(
            session,
            turn_count=len(turns),
            artifact_root=app.state.config.artifact_root,
            latest_turn_payload=latest_turn_payload,
            turn_payloads=turn_payloads,
        )
        serialized_turns = [serialize_turn_response(payload).model_dump() for payload in turn_payloads]
        latest_turn = None if not serialized_turns else serialized_turns[0]
        page_context = _build_session_page_render_context(
            session_detail,
            turns=serialized_turns,
            latest_turn=latest_turn,
        )
        return templates.TemplateResponse(
            request,
            "session.html",
            {
                "request": request,
                "session": session_detail.model_dump(),
                "turns": serialized_turns,
                "latest_turn": latest_turn,
                "api_prefix": api_prefix,
                **page_context,
            },
        )

    return app


def serialize_session_summary(session) -> SessionSummary:
    return SessionSummary(
        session_id=session.session_id,
        dataset_key=session.dataset_key,
        created_at=session.created_at,
        updated_at=session.updated_at,
        current_public_state=session.current_public_state,
        last_turn_index=session.last_turn_index,
        has_pending_clarification=session.pending_clarification_json is not None,
        lifecycle_status=session.lifecycle_status,
        archived_at=session.archived_at,
        is_read_only=session.lifecycle_status == "archived",
        clarification_turns_used=session.clarification_turns_used,
        is_case_complete=session.is_case_complete,
        case_completion_reason=session.case_completion_reason,
        restart_required=session.restart_required,
        active_constraint_spec=session.active_constraint_spec_json,
        refinement_revision_index=session.refinement_revision_index,
        refinement_rounds_used=session.refinement_rounds_used,
        refinement_round_limit=session.refinement_round_limit,
        has_pending_refinement_clarification=session.pending_refinement_clarification_json is not None,
        refinement_allowed=_serialize_refinement_allowed(session),
        latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
    )


def _select_default_dataset(datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not datasets:
        return None
    for item in datasets:
        if item.get("availability_status") == "active":
            return item
    return datasets[0]


def _show_refinement_surface(session: SessionSummary | SessionDetail) -> bool:
    if not session.refinement_allowed:
        return False
    if session.has_pending_refinement_clarification:
        return True
    return (
        session.latest_runtime_backed_turn_id is not None
        and session.current_public_state in RUNTIME_COMPLETE_STATES
    )


def serialize_session_detail(
    session,
    *,
    turn_count: int,
    artifact_root: Path,
    latest_turn_payload: dict[str, Any] | None = None,
    turn_payloads: list[dict[str, Any]] | None = None,
) -> SessionDetail:
    ui_review = _build_session_ui_review(session, latest_turn_payload=latest_turn_payload)
    latest_runtime_turn = _find_latest_runtime_turn_payload(
        turn_payloads=turn_payloads,
        latest_turn_payload=latest_turn_payload,
        latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
    )
    latest_runtime_summary = _build_latest_runtime_summary(latest_runtime_turn)
    page_state = _resolve_page_state_from_sources(
        session,
        latest_visible_turn=latest_turn_payload,
        latest_runtime_summary=latest_runtime_summary,
        turn_count=turn_count,
        ui_review=ui_review,
    )
    composer_mode = _resolve_composer_mode_from_page_state(session, page_state=page_state)
    session_render_hints = _build_session_render_hints(
        session=session,
        ui_review=ui_review,
        page_state=page_state,
        latest_turn_payload=latest_turn_payload,
        latest_runtime_summary=latest_runtime_summary,
        composer_mode=composer_mode,
    )
    return SessionDetail(
        session_id=session.session_id,
        dataset_key=session.dataset_key,
        created_at=session.created_at,
        updated_at=session.updated_at,
        current_public_state=session.current_public_state,
        last_turn_index=session.last_turn_index,
        has_pending_clarification=session.pending_clarification_json is not None,
        lifecycle_status=session.lifecycle_status,
        archived_at=session.archived_at,
        is_read_only=session.lifecycle_status == "archived",
        clarification_turns_used=session.clarification_turns_used,
        is_case_complete=session.is_case_complete,
        case_completion_reason=session.case_completion_reason,
        restart_required=session.restart_required,
        active_constraint_spec=session.active_constraint_spec_json,
        refinement_revision_index=session.refinement_revision_index,
        refinement_rounds_used=session.refinement_rounds_used,
        refinement_round_limit=session.refinement_round_limit,
        has_pending_refinement_clarification=session.pending_refinement_clarification_json is not None,
        refinement_allowed=_serialize_refinement_allowed(session),
        latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
        turn_count=turn_count,
        latest_turn_id=session.latest_turn_id,
        artifact_root=str(artifact_root),
        ui_review=ui_review,
        render_hints=session_render_hints,
    )


def serialize_turn_response(payload: dict) -> TurnResponse:
    debug_summary = payload["debug_summary"]
    runtime_summary = RuntimeSummary(**debug_summary["runtime_summary"])
    ui_review = _build_turn_ui_review(payload)
    render_hints = _build_turn_render_hints(
        payload,
        ui_review=ui_review,
    )
    return TurnResponse(
        session_id=payload["session_id"],
        turn_id=payload["turn_id"],
        turn_index=payload["turn_index"],
        user_input=payload["user_input"],
        assistant_text=payload["assistant_text"],
        public_state=payload["public_state"],
        clarification_payload=payload["clarification_payload"],
        explanation_payload=payload["explanation_payload"],
        artifact_refs=ArtifactRefBundle(**payload["artifact_refs"]),
        debug_summary=DebugSummary(
            builder_status=debug_summary["builder_status"],
            builder_reason_codes=list(debug_summary["builder_reason_codes"]),
            transition_reason=debug_summary["transition_reason"],
            merge_applied=bool(debug_summary["merge_applied"]),
            runtime_summary=runtime_summary,
            invariant_validation_status=debug_summary["invariant_validation_status"],
            artifact_dir=debug_summary["artifact_dir"],
            timing_metrics=debug_summary.get("timing_metrics"),
        ),
        clarification_turns_used=payload["clarification_turns_used"],
        is_case_complete=payload["is_case_complete"],
        case_completion_reason=payload["case_completion_reason"],
        restart_required=payload["restart_required"],
        turn_kind=payload["turn_kind"],
        refinement_status=payload["refinement_status"],
        refinement_revision_index=payload["refinement_revision_index"],
        parent_terminal_turn_id=payload["parent_terminal_turn_id"],
        parent_refinement_revision_index=payload["parent_refinement_revision_index"],
        active_constraint_spec=payload["active_constraint_spec"],
        constraint_feedback_delta=payload["constraint_feedback_delta"],
        refinement_rounds_used=payload["refinement_rounds_used"],
        refinement_round_limit=payload["refinement_round_limit"],
        ui_review=ui_review,
        render_hints=render_hints,
    )


def _build_session_page_render_context(
    session: SessionDetail,
    *,
    turns: list[dict[str, Any]],
    latest_turn: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_turns = [_decorate_turn_for_render(turn) for turn in turns]
    latest_visible_turn = _decorate_turn_for_render(latest_turn)
    latest_runtime_turn = _find_latest_runtime_turn(
        session,
        turns=normalized_turns,
        latest_visible_turn=latest_visible_turn,
    )
    latest_runtime_summary = _build_latest_runtime_summary(latest_runtime_turn)
    page_state = _resolve_page_state(
        session,
        latest_visible_turn=latest_visible_turn,
        latest_runtime_summary=latest_runtime_summary,
    )
    composer_mode = _resolve_composer_mode(session, page_state=page_state)
    session_render_hints = _build_session_render_hints(
        session=session,
        ui_review=session.ui_review,
        page_state=page_state,
        latest_turn_payload=latest_visible_turn,
        latest_runtime_summary=latest_runtime_summary,
        composer_mode=composer_mode,
    )
    review_summary = _build_review_summary(session.ui_review)
    next_action_summary = _build_next_action_summary(
        session,
        page_state=page_state,
        render_hints=session_render_hints,
    )
    show_advanced_controls_by_default = _show_advanced_controls_by_default(session_render_hints)
    return {
        "page_state": page_state,
        "session_render_hints": session_render_hints.model_dump(),
        "chat_header_summary": _build_chat_header_summary(
            session,
            page_state=page_state,
        ),
        "latest_runtime_summary": latest_runtime_summary,
        "latest_runtime_turn": latest_runtime_turn,
        "composer_mode": composer_mode,
        "composer_context": session_render_hints.composer_context.model_dump(),
        "transcript_items": _build_transcript_items(
            session,
            turns=normalized_turns,
            page_state=page_state,
        ),
        "review_summary": review_summary,
        "next_action_summary": next_action_summary,
        "context_sections": _build_context_sections(
            page_state=page_state,
            review_summary=review_summary,
            next_action_summary=next_action_summary,
        ),
        "show_advanced_controls_by_default": show_advanced_controls_by_default,
        "restart_helper_copy": _restart_helper_copy(session),
        "show_lifecycle_badge": _show_lifecycle_badge(session),
    }


def _decorate_turn_for_render(turn: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(turn, dict):
        return None
    enriched = dict(turn)
    ui_review_value = enriched.get("ui_review")
    if isinstance(ui_review_value, UiReviewPayload):
        ui_review = ui_review_value
    elif isinstance(ui_review_value, dict):
        ui_review = UiReviewPayload(**ui_review_value)
    else:
        ui_review = _build_turn_ui_review(enriched)
        enriched["ui_review"] = ui_review.model_dump()
    render_hints_value = enriched.get("render_hints")
    if isinstance(render_hints_value, TurnRenderHints):
        render_hints = render_hints_value
    elif isinstance(render_hints_value, dict):
        render_hints = TurnRenderHints(**render_hints_value)
    else:
        render_hints = _build_turn_render_hints(
            enriched,
            ui_review=ui_review,
        )
        enriched["render_hints"] = render_hints.model_dump()
    if "ui_review" not in enriched:
        enriched["ui_review"] = ui_review.model_dump()
    if "render_hints" not in enriched:
        enriched["render_hints"] = render_hints.model_dump()
    return enriched


def _resolve_page_state(
    session: SessionDetail,
    *,
    latest_visible_turn: dict[str, Any] | None,
    latest_runtime_summary: dict[str, Any] | None,
) -> str:
    return _resolve_page_state_from_sources(
        session,
        latest_visible_turn=latest_visible_turn,
        latest_runtime_summary=latest_runtime_summary,
        turn_count=session.turn_count,
        ui_review=session.ui_review,
    )


def _build_latest_runtime_summary(latest_runtime_turn: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(latest_runtime_turn, dict):
        return None
    explanation = latest_runtime_turn.get("explanation_payload")
    explanation_payload = explanation if isinstance(explanation, dict) else {}
    summary_type = _string_or_none(explanation_payload.get("summary_type"))
    changed_feature_count = _runtime_changed_feature_count(explanation_payload)
    if latest_runtime_turn.get("public_state") == "RUNTIME_SUCCESS":
        if summary_type == "no_recourse_needed":
            headline = "Current profile already qualifies"
            supporting_copy = "The current bank profile already reaches the desired outcome with no further changes."
        elif summary_type == "counterfactual_found":
            headline = "Validated counterfactual recommendation"
            supporting_copy = "A runtime-backed recommendation is available for the current request."
        else:
            headline = "Runtime-backed recommendation"
            supporting_copy = "Runtime completed with a recommendation for the current request."
        return {
            "kind": "success",
            "headline": headline,
            "supporting_copy": supporting_copy,
            "summary_type": summary_type,
            "changed_feature_count": changed_feature_count,
            "reject_mode": None,
        }

    reject_mode = _runtime_reject_mode(explanation_payload)
    if reject_mode == "constraints_blocked":
        headline = "Current constraints block a recommendation"
        supporting_copy = "No recommendation can be shown under the active request-specific constraints."
    elif reject_mode == "invalid_counterfactual_blocked":
        headline = "No safe recommendation available"
        supporting_copy = "A candidate was generated, but it could not be shown after validation."
    else:
        headline = "No recommendation available"
        supporting_copy = "Runtime completed without a feasible recommendation for the current request."
    return {
        "kind": "reject",
        "headline": headline,
        "supporting_copy": supporting_copy,
        "summary_type": summary_type,
        "changed_feature_count": changed_feature_count,
        "reject_mode": reject_mode,
    }


def _find_latest_runtime_turn_payload(
    *,
    turn_payloads: list[dict[str, Any]] | None,
    latest_turn_payload: dict[str, Any] | None,
    latest_runtime_backed_turn_id: str | None,
) -> dict[str, Any] | None:
    if _turn_carries_runtime_outcome(
        latest_turn_payload,
        latest_runtime_backed_turn_id=latest_runtime_backed_turn_id,
    ):
        return latest_turn_payload
    if not turn_payloads:
        return None
    if latest_runtime_backed_turn_id:
        for payload in turn_payloads:
            if payload.get("turn_id") != latest_runtime_backed_turn_id:
                continue
            if _turn_carries_runtime_outcome(
                payload,
                latest_runtime_backed_turn_id=latest_runtime_backed_turn_id,
            ):
                return payload
    for payload in turn_payloads:
        if _turn_carries_runtime_outcome(
            payload,
            latest_runtime_backed_turn_id=latest_runtime_backed_turn_id,
        ):
            return payload
    return None


def _build_turn_render_hints(
    payload: dict[str, Any],
    *,
    ui_review: UiReviewPayload | None,
) -> TurnRenderHints:
    clarification_payload = payload.get("clarification_payload")
    clarification_payload_dict = clarification_payload if isinstance(clarification_payload, dict) else None
    explanation_payload = payload.get("explanation_payload")
    explanation_payload_dict = explanation_payload if isinstance(explanation_payload, dict) else None
    assistant_text = str(payload.get("assistant_text") or "").strip()
    refinement_status = _string_or_none(payload.get("refinement_status"))
    clarification_type = _string_or_none(
        None if clarification_payload_dict is None else clarification_payload_dict.get("clarification_type")
    )
    runtime_summary = None
    if _turn_carries_runtime_outcome(
        payload,
        latest_runtime_backed_turn_id=None,
    ):
        runtime_summary = _build_latest_runtime_summary(payload)

    if clarification_type == "clarification_limit_reached":
        return _build_restart_required_render_hints(
            case_completion_reason=_string_or_none(payload.get("case_completion_reason")),
            assistant_text=assistant_text,
        )
    if (
        bool(payload.get("restart_required"))
        and payload.get("public_state") not in RUNTIME_COMPLETE_STATES
        and clarification_payload_dict is None
        and explanation_payload_dict is None
    ):
        return _build_restart_required_render_hints(
            case_completion_reason=_string_or_none(payload.get("case_completion_reason")),
            assistant_text=assistant_text,
        )
    if clarification_type == "refinement_clarification" or refinement_status == "clarification_required":
        return _build_refinement_clarification_render_hints(
            clarification_payload=clarification_payload_dict,
            ui_review=ui_review,
            assistant_text=assistant_text,
        )
    missing_fields = _ordered_fields(
        [] if clarification_payload_dict is None else clarification_payload_dict.get("missing_fields")
    )
    if missing_fields:
        return _build_original_clarification_render_hints(
            clarification_payload=clarification_payload_dict,
            ui_review=ui_review,
            assistant_text=assistant_text,
        )
    if runtime_summary is not None:
        explanation_fragments = _build_runtime_explanation_fragments(
            summary=runtime_summary,
            ui_review=ui_review,
            explanation_payload=explanation_payload_dict,
        )
        if runtime_summary["kind"] == "success":
            return _build_runtime_success_render_hints(explanation_fragments)
        return _build_runtime_reject_render_hints(explanation_fragments)
    return _build_fallback_render_hints(assistant_text=assistant_text)


def _build_session_render_hints(
    *,
    session,
    ui_review: UiReviewPayload | None,
    page_state: str,
    latest_turn_payload: dict[str, Any] | None,
    latest_runtime_summary: dict[str, Any] | None,
    composer_mode: str,
) -> SessionRenderHints:
    clarification_payload = None
    explanation_payload = None
    assistant_text = ""
    refinement_status = None
    if isinstance(latest_turn_payload, dict):
        clarification_payload = latest_turn_payload.get("clarification_payload")
        explanation_payload = latest_turn_payload.get("explanation_payload")
        assistant_text = str(latest_turn_payload.get("assistant_text") or "").strip()
        refinement_status = _string_or_none(latest_turn_payload.get("refinement_status"))
    clarification_payload_dict = clarification_payload if isinstance(clarification_payload, dict) else None
    explanation_payload_dict = explanation_payload if isinstance(explanation_payload, dict) else None

    if page_state == "fresh":
        hints = TurnRenderHints(
            primary_chat_text=_build_fresh_welcome_copy(dataset_key=str(session.dataset_key or "bank")),
            primary_action_type=PRIMARY_ACTION_START_CASE,
            primary_action_items=["Describe one bank profile in natural language."],
            supporting_detail_title=None,
            supporting_detail_body=None,
            supporting_detail_facts=[],
            state_marker_label=None,
            right_rail_anchor=None,
        )
    elif page_state == "restart_required":
        hints = _build_restart_required_render_hints(
            case_completion_reason=_string_or_none(session.case_completion_reason),
            assistant_text=assistant_text,
        )
    elif page_state == "clarification":
        payload = clarification_payload_dict or _build_session_level_clarification_payload(session)
        hints = _build_original_clarification_render_hints(
            clarification_payload=payload,
            ui_review=ui_review,
            assistant_text=assistant_text,
        )
    elif page_state == "refinement_clarification":
        payload = clarification_payload_dict or _build_session_level_clarification_payload(session)
        hints = _build_refinement_clarification_render_hints(
            clarification_payload=payload,
            ui_review=ui_review,
            assistant_text=assistant_text,
        )
    elif page_state in {"runtime_success", "runtime_reject"} and latest_runtime_summary is not None:
        explanation_fragments = _build_runtime_explanation_fragments(
            summary=latest_runtime_summary,
            ui_review=ui_review,
            explanation_payload=explanation_payload_dict,
        )
        if latest_runtime_summary["kind"] == "success":
            hints = _build_runtime_success_render_hints(explanation_fragments)
        else:
            hints = _build_runtime_reject_render_hints(explanation_fragments)
    else:
        hints = _build_fallback_render_hints(assistant_text=assistant_text)

    structured_controls_relevant = bool(
        page_state == "refinement_clarification"
        and hints.right_rail_anchor == RIGHT_RAIL_ADVANCED_CONTROLS
    )
    composer_context = _build_composer_context(
        session,
        composer_mode=composer_mode,
        page_state=page_state,
        advanced_controls_relevant=structured_controls_relevant,
    )
    return SessionRenderHints(
        primary_chat_text=hints.primary_chat_text,
        primary_action_type=hints.primary_action_type,
        primary_action_items=list(hints.primary_action_items),
        supporting_detail_title=hints.supporting_detail_title,
        supporting_detail_body=hints.supporting_detail_body,
        supporting_detail_facts=list(hints.supporting_detail_facts),
        state_marker_label=hints.state_marker_label,
        right_rail_anchor=hints.right_rail_anchor,
        page_state=page_state,
        composer_mode=composer_mode,
        composer_context=composer_context,
    )


def _build_original_clarification_render_hints(
    *,
    clarification_payload: dict[str, Any] | None,
    ui_review: UiReviewPayload | None,
    assistant_text: str,
) -> TurnRenderHints:
    missing_fields = _ordered_fields(
        [] if clarification_payload is None else clarification_payload.get("missing_fields")
    )
    carried_forward_fields = _ordered_fields(
        [] if clarification_payload is None else clarification_payload.get("carried_forward_fields")
    )
    next_required_input = _string_or_none(
        None if clarification_payload is None else clarification_payload.get("next_required_input")
    )
    primary_chat_text = _build_missing_fields_primary_text(
        missing_fields,
        carried_forward_fields=carried_forward_fields,
    )
    supporting_parts = [
        next_required_input or "Provide one corrected bank profile with the remaining required fields.",
        _build_resubmission_example(ui_review, missing_fields),
    ]
    if carried_forward_fields:
        supporting_parts.insert(
            0,
            f"I'll keep the values already provided for {_format_field_list(carried_forward_fields)}.",
        )
    supporting_detail_body = " ".join(part for part in supporting_parts if part)
    return TurnRenderHints(
        primary_chat_text=primary_chat_text if primary_chat_text else assistant_text,
        primary_action_type=PRIMARY_ACTION_PROVIDE_MISSING_FIELDS,
        primary_action_items=list(missing_fields),
        supporting_detail_title="Why Runtime Is Blocked",
        supporting_detail_body=supporting_detail_body,
        supporting_detail_facts=list(missing_fields[:3]),
        state_marker_label="Need more information",
        right_rail_anchor=RIGHT_RAIL_REVIEW,
    )


def _build_refinement_clarification_render_hints(
    *,
    clarification_payload: dict[str, Any] | None,
    ui_review: UiReviewPayload | None,
    assistant_text: str,
) -> TurnRenderHints:
    conflicts = [
        str(item)
        for item in list((clarification_payload or {}).get("conflicts") or [])
        if isinstance(item, str) and str(item).strip()
    ]
    next_required_input = _string_or_none(
        None if clarification_payload is None else clarification_payload.get("next_required_input")
    )
    if conflicts:
        primary_followup = conflicts[0].rstrip(".")
        primary_chat_text = f"I need a more specific refinement request. {primary_followup}. {next_required_input or 'Clarify the refinement intent.'}"
    else:
        primary_chat_text = (
            "I need a more specific refinement request. "
            f"{next_required_input or 'Clarify which fields should stay fixed or which bounds should change.'}"
        )
    action_items = list(conflicts[:2])
    if next_required_input and next_required_input not in action_items:
        action_items.append(next_required_input)
    supporting_detail_body = " ".join(
        part
        for part in [
            "The previous refinement could not be applied because it was still ambiguous.",
            next_required_input,
        ]
        if part
    )
    supporting_detail_facts = conflicts[:3]
    structured_relevant = _refinement_controls_are_relevant(
        clarification_payload=clarification_payload,
        ui_review=ui_review,
    )
    return TurnRenderHints(
        primary_chat_text=primary_chat_text if primary_chat_text else assistant_text,
        primary_action_type=PRIMARY_ACTION_CLARIFY_REFINEMENT,
        primary_action_items=action_items,
        supporting_detail_title="Clarify The Refinement",
        supporting_detail_body=supporting_detail_body or assistant_text or None,
        supporting_detail_facts=supporting_detail_facts,
        state_marker_label="Refinement needs clarification",
        right_rail_anchor=RIGHT_RAIL_ADVANCED_CONTROLS if structured_relevant else RIGHT_RAIL_RESULT,
    )


def _build_runtime_success_render_hints(explanation_fragments: dict[str, Any]) -> TurnRenderHints:
    return TurnRenderHints(
        primary_chat_text=str(explanation_fragments["visible_explanation_line"]),
        primary_action_type=PRIMARY_ACTION_NO_ACTION_REQUIRED,
        primary_action_items=[],
        supporting_detail_title="Result details",
        supporting_detail_body=_string_or_none(explanation_fragments.get("expanded_explanation_summary")),
        supporting_detail_facts=list(explanation_fragments.get("expanded_fact_rows") or []),
        state_marker_label="Recommendation found",
        right_rail_anchor=RIGHT_RAIL_RESULT,
    )


def _build_runtime_reject_render_hints(explanation_fragments: dict[str, Any]) -> TurnRenderHints:
    return TurnRenderHints(
        primary_chat_text=str(explanation_fragments["visible_explanation_line"]),
        primary_action_type=PRIMARY_ACTION_RELAX_CONSTRAINTS_OR_RESTART,
        primary_action_items=["Relax constraints", "Start a new case"],
        supporting_detail_title="Why no recommendation",
        supporting_detail_body=_string_or_none(explanation_fragments.get("expanded_explanation_summary")),
        supporting_detail_facts=list(explanation_fragments.get("expanded_fact_rows") or []),
        state_marker_label="No recommendation available",
        right_rail_anchor=RIGHT_RAIL_RESULT,
    )


def _build_restart_required_render_hints(
    *,
    case_completion_reason: str | None = None,
    assistant_text: str | None = None,
) -> TurnRenderHints:
    primary_chat_text = _string_or_none(assistant_text) or _build_restart_required_primary_chat_text(
        case_completion_reason
    )
    return TurnRenderHints(
        primary_chat_text=primary_chat_text
        or "This case has been completed. Start a new case whenever you're ready.",
        primary_action_type=PRIMARY_ACTION_START_NEW_CASE,
        primary_action_items=["Start a new case"],
        supporting_detail_title=None,
        supporting_detail_body=None,
        supporting_detail_facts=[],
        state_marker_label=None,
        right_rail_anchor=RIGHT_RAIL_RESULT,
    )


def _build_fallback_render_hints(*, assistant_text: str) -> TurnRenderHints:
    return TurnRenderHints(
        primary_chat_text=assistant_text,
        primary_action_type=PRIMARY_ACTION_NONE,
        primary_action_items=[],
        supporting_detail_title=None,
        supporting_detail_body=None,
        supporting_detail_facts=[],
        state_marker_label=None,
        right_rail_anchor=None,
    )


def _build_restart_required_primary_chat_text(case_completion_reason: str | None) -> str:
    if case_completion_reason == "runtime_success":
        return "This case is complete. A recommendation has been provided, so start a new case to check another profile."
    if case_completion_reason == "runtime_reject":
        return "This case is complete, but no viable recommendation was found under the current request. Start a new case to try a different profile or constraints."
    if case_completion_reason == "clarification_limit_reached":
        return "The clarification limit was reached for this case. Start a new case with one complete bank profile."
    if case_completion_reason == "conflict":
        return "This case ended because the request contained conflicting values. Start a new case with one corrected bank profile."
    if case_completion_reason == "unsupported_request":
        return "This request is outside the supported bank-profile flow. Start a new case with one complete bank profile."
    if case_completion_reason == "parser_failure":
        return "The request could not be safely interpreted. Start a new case and resubmit one complete bank profile."
    return "This case has been completed. Start a new case whenever you're ready."


def _build_runtime_explanation_fragments(
    *,
    summary: dict[str, Any],
    ui_review: UiReviewPayload | None,
    explanation_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    constraint_summary = _build_constraint_summary_line(
        [] if ui_review is None else ui_review.constraints,
        empty_text="",
    )
    changed_feature_count = summary.get("changed_feature_count")
    reject_mode = summary.get("reject_mode")
    summary_type = summary.get("summary_type")
    fact_rows: list[str] = []

    if summary["kind"] == "success":
        if summary_type == "no_recourse_needed":
            visible_line = (
                "Great news! Based on your current profile, your bank loan application would already be approved. "
                "No changes to your profile are needed."
            )
            expanded_summary = "The model predicts a positive outcome for your current profile without any modifications."
            fact_rows.append("Loan: Approved ✓")
        elif summary_type == "counterfactual_found":
            visible_change_text = _build_counterfactual_visible_change_text(explanation_payload)
            if visible_change_text:
                visible_line = (
                    "Your current profile would be rejected for the bank loan. "
                    f"However, I found a way to get approved — here are the recommended changes: {visible_change_text}."
                )
            else:
                visible_line = (
                    "Your current profile would be rejected for the bank loan, "
                    "but I found a recommendation that could help you get approved. "
                    "Check the details below to see what to change."
                )
            expanded_summary = "The system found a validated set of profile changes that would flip the prediction to approved."
            fact_rows.append("Loan: Rejected → Approved with changes")
        else:
            visible_line = (
                "A recommendation is available for your profile. "
                "Review the details below to understand what changes could improve your outcome."
            )
            expanded_summary = "Runtime completed with a recommendation for the current request."
        if isinstance(changed_feature_count, int):
            plural = "" if changed_feature_count == 1 else "s"
            fact_rows.append(f"{changed_feature_count} field{plural} to change")
        if constraint_summary:
            expanded_summary = f"{expanded_summary} Active constraints: {constraint_summary}."
        return {
            "visible_explanation_line": visible_line,
            "expanded_explanation_summary": expanded_summary,
            "expanded_fact_rows": fact_rows[:3],
        }

    if reject_mode == "constraints_blocked":
        visible_line = (
            "Your current profile would be rejected for the bank loan. "
            "Unfortunately, I couldn't find approved-path changes that also respect your constraints. "
            "Try relaxing some constraints (e.g. allow more fields to change) or start a new case."
        )
        expanded_summary = "All generated recommendations were blocked by your active constraints (e.g. blocked fields, bounds, or change limits)."
        fact_rows.append("Blocked by your constraints")
    elif reject_mode == "invalid_counterfactual_blocked":
        visible_line = (
            "Your current profile would be rejected for the bank loan. "
            "A potential recommendation was found but did not pass safety validation. "
            "Try adjusting your profile values or relaxing constraints."
        )
        expanded_summary = "A candidate was generated but failed post-generation safety checks."
        fact_rows.append("Safety check blocked")
    else:
        visible_line = (
            "Your current profile would be rejected for the bank loan. "
            "Unfortunately, the system could not find any feasible changes to get you approved. "
            "Try submitting a different profile or relaxing your constraints."
        )
        expanded_summary = "Runtime completed without finding a feasible path to approval for the current profile."
        fact_rows.append("No path to approval found")
    if constraint_summary:
        expanded_summary = f"{expanded_summary} Active constraints: {constraint_summary}."
        fact_rows.append(f"Constraints: {constraint_summary}")
    if isinstance(explanation_payload, dict):
        suggestion_types = [
            str(item).replace("_", " ")
            for item in list(explanation_payload.get("included_suggestion_types") or [])
            if isinstance(item, str)
        ]
        if suggestion_types:
            fact_rows.append(suggestion_types[0])
    return {
        "visible_explanation_line": visible_line,
        "expanded_explanation_summary": expanded_summary,
        "expanded_fact_rows": fact_rows[:3],
    }


def _build_counterfactual_visible_change_text(explanation_payload: dict[str, Any] | None) -> str | None:
    if not isinstance(explanation_payload, dict):
        return None
    counterfactual_summary = explanation_payload.get("counterfactual_summary")
    if not isinstance(counterfactual_summary, dict):
        return None
    profile_diff = counterfactual_summary.get("profile_diff")
    if not isinstance(profile_diff, dict) or not profile_diff:
        return None

    visible_changes: list[str] = []
    for field_name, values in profile_diff.items():
        if not isinstance(field_name, str) or not isinstance(values, dict):
            continue
        if "from" not in values or "to" not in values:
            continue
        from_value = values.get("from")
        to_value = values.get("to")
        from_display = _format_field_value(field_name, from_value, missing=from_value is None)
        to_display = _format_field_value(field_name, to_value, missing=to_value is None)
        visible_changes.append(f"{field_name}: {from_display} -> {to_display}")
    if not visible_changes:
        return None
    return ", ".join(visible_changes)


def _build_missing_fields_primary_text(
    missing_fields: list[str],
    *,
    carried_forward_fields: list[str] | None = None,
) -> str:
    if not missing_fields:
        return "Reply with the missing fields so I can continue."
    if len(missing_fields) == 1:
        primary_text = f"Reply with only the missing field: {missing_fields[0]}."
    else:
        primary_text = "Reply with only the missing fields: {fields}.".format(fields=_format_field_list(missing_fields))
    if carried_forward_fields:
        return (
            f"{primary_text} "
            f"I'll keep the values already provided for {_format_field_list(carried_forward_fields)}."
        )
    return primary_text


def _build_resubmission_example(ui_review: UiReviewPayload | None, missing_fields: list[str]) -> str:
    example_parts: list[str] = []
    fields_by_name = {}
    if ui_review is not None:
        fields_by_name = {field.field_name: field for field in ui_review.profile_fields}
    missing = set(missing_fields)
    for field_name in BANK_REQUIRED_FIELD_ORDER:
        field = fields_by_name.get(field_name)
        value = None if field is None else field.value
        if field_name in missing or value is None:
            display_value = "<value>"
        elif field_name in BANK_BOOLEAN_FIELDS:
            display_value = "yes" if bool(value) else "no"
        else:
            display_value = _format_field_value(field_name, value, missing=False)
        example_parts.append(f"{field_name} {display_value}")
    return f"Example resubmission: {', '.join(example_parts)}."


def _refinement_controls_are_relevant(
    *,
    clarification_payload: dict[str, Any] | None,
    ui_review: UiReviewPayload | None,
) -> bool:
    combined = " ".join(
        item
        for item in [
            None if ui_review is None else ui_review.clarification_message,
            " ".join(
                str(item)
                for item in list((clarification_payload or {}).get("conflicts") or [])
                if isinstance(item, str)
            ),
        ]
        if isinstance(item, str) and item
    ).lower()
    structured_keywords = (
        "max changed",
        "changed feature",
        "prefer fewer",
        "bound",
        "at most",
        "at least",
        "do not change",
        "allow ",
        "blocked",
        "fixed",
        "stay fixed",
        "may change",
        "can change",
    )
    return any(keyword in combined for keyword in structured_keywords)


def _build_fresh_welcome_copy(*, dataset_key: str) -> str:
    if dataset_key == "grad":
        return "Describe one graduate-admission profile to start a new case."
    return "Describe one bank profile in natural language to start a new case."


def _resolve_page_state_from_sources(
    session,
    *,
    latest_visible_turn: dict[str, Any] | None,
    latest_runtime_summary: dict[str, Any] | None,
    turn_count: int,
    ui_review: UiReviewPayload | None,
) -> str:
    same_case_continuation_allowed = bool(
        getattr(session, "refinement_allowed", _serialize_refinement_allowed(session))
    )
    if session.restart_required and not same_case_continuation_allowed:
        return "restart_required"
    if bool(
        getattr(
            session,
            "has_pending_refinement_clarification",
            getattr(session, "pending_refinement_clarification_json", None) is not None,
        )
    ):
        return "refinement_clarification"
    if _turn_is_original_clarification(latest_visible_turn):
        return "clarification"
    if latest_visible_turn is None and _has_pending_original_clarification_source(session, ui_review=ui_review):
        return "clarification"
    if _turn_carries_runtime_outcome(
        latest_visible_turn,
        latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
    ):
        if latest_visible_turn.get("public_state") == "RUNTIME_SUCCESS":
            return "runtime_success"
        if latest_visible_turn.get("public_state") == "RUNTIME_REJECT":
            return "runtime_reject"
    if latest_runtime_summary is not None:
        return "runtime_success" if latest_runtime_summary["kind"] == "success" else "runtime_reject"
    if turn_count == 0:
        return "fresh"
    if session.current_public_state == "RUNTIME_SUCCESS":
        return "runtime_success"
    if session.current_public_state == "RUNTIME_REJECT":
        return "runtime_reject"
    if session.current_public_state == "NEEDS_CLARIFICATION":
        return "clarification"
    return "restart_required"


def _resolve_composer_mode_from_page_state(session, *, page_state: str) -> str:
    if session.lifecycle_status == "archived" or page_state == "restart_required":
        return COMPOSER_MODE_DISABLED
    if page_state in {"runtime_success", "runtime_reject", "refinement_clarification"}:
        return COMPOSER_MODE_REFINEMENT
    return COMPOSER_MODE_MESSAGE


def _has_pending_original_clarification_source(
    session,
    *,
    ui_review: UiReviewPayload | None,
) -> bool:
    if ui_review is not None and ui_review.clarification_type == "refinement_clarification":
        return False
    return bool(
        getattr(
            session,
            "has_pending_clarification",
            getattr(session, "pending_clarification_json", None) is not None,
        )
        or session.current_public_state == "NEEDS_CLARIFICATION"
    )


def _runtime_changed_feature_count(explanation_payload: dict[str, Any]) -> int | None:
    if not isinstance(explanation_payload, dict):
        return None
    counterfactual_summary = explanation_payload.get("counterfactual_summary")
    if isinstance(counterfactual_summary, dict):
        profile_diff = counterfactual_summary.get("profile_diff")
        if isinstance(profile_diff, dict):
            return len(profile_diff)
    changed_fields = explanation_payload.get("changed_fields")
    if isinstance(changed_fields, list):
        return len(changed_fields)
    return None


def _runtime_reject_mode(explanation_payload: dict[str, Any]) -> str | None:
    if not isinstance(explanation_payload, dict):
        return None
    reason_codes = [str(item) for item in list(explanation_payload.get("reason_codes") or []) if isinstance(item, str)]
    if REQUEST_CONSTRAINTS_BLOCKED in reason_codes:
        return "constraints_blocked"
    if INVALID_COUNTERFACTUAL_BLOCKED in reason_codes:
        return "invalid_counterfactual_blocked"
    if reason_codes:
        return "runtime_reject"
    return None


def _find_latest_runtime_turn(
    session: SessionDetail,
    *,
    turns: list[dict[str, Any]],
    latest_visible_turn: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if _turn_carries_runtime_outcome(
        latest_visible_turn,
        latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
    ):
        return latest_visible_turn
    if session.latest_runtime_backed_turn_id:
        for turn in turns:
            if turn.get("turn_id") != session.latest_runtime_backed_turn_id:
                continue
            if _turn_carries_runtime_outcome(
                turn,
                latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
            ):
                return turn
    for turn in turns:
        if _turn_carries_runtime_outcome(
            turn,
            latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
        ):
            return turn
    return None


def _turn_carries_runtime_outcome(
    turn: dict[str, Any] | None,
    *,
    latest_runtime_backed_turn_id: str | None,
) -> bool:
    if not isinstance(turn, dict):
        return False
    if turn.get("public_state") not in RUNTIME_COMPLETE_STATES:
        return False
    if latest_runtime_backed_turn_id and turn.get("turn_id") == latest_runtime_backed_turn_id:
        return True
    if isinstance(turn.get("explanation_payload"), dict):
        return True
    if turn.get("refinement_status") == "applied":
        return True
    return turn.get("turn_kind") == "message"


def _turn_is_original_clarification(turn: dict[str, Any] | None) -> bool:
    if not isinstance(turn, dict):
        return False
    clarification_payload = turn.get("clarification_payload")
    clarification_type = None
    if isinstance(clarification_payload, dict):
        clarification_type = clarification_payload.get("clarification_type")
    if clarification_type == "refinement_clarification":
        return False
    if turn.get("turn_kind") == "refinement" and turn.get("refinement_status") == "clarification_required":
        return False
    return (
        turn.get("public_state") == "NEEDS_CLARIFICATION"
        or isinstance(clarification_payload, dict)
    )


def _has_pending_original_clarification(session: SessionDetail) -> bool:
    ui_review = session.ui_review
    if ui_review is not None and ui_review.clarification_type == "refinement_clarification":
        return False
    return bool(session.has_pending_clarification or session.current_public_state == "NEEDS_CLARIFICATION")


def _build_timeline_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        items.append(
            {
                "turn": turn,
                "is_latest": index == 0,
                "expanded": index == 0,
                "preview_text": _timeline_preview(turn),
            }
        )
    return items


def _timeline_preview(turn: dict[str, Any]) -> str:
    preview_source = turn.get("assistant_text") or turn.get("user_input") or ""
    preview = " ".join(str(preview_source).split())
    if len(preview) <= 120:
        return preview
    return f"{preview[:117].rstrip()}..."


def _build_page_banner(
    *,
    page_state: str,
    latest_runtime_summary: dict[str, Any] | None,
) -> dict[str, str] | None:
    if page_state == "clarification":
        return {
            "tone": "info",
            "headline": "Need more information",
            "copy": "Continue the same case by answering the clarification request below.",
        }
    if page_state == "runtime_success":
        return {
            "tone": "success",
            "headline": "Recommendation found",
            "copy": "A runtime-backed outcome is ready below, and same-case refinement remains available.",
        }
    if page_state == "runtime_reject":
        return {
            "tone": "warning",
            "headline": "No recommendation available",
            "copy": "The current request did not yield a recommendation, but same-case refinement remains available.",
        }
    if page_state == "refinement_clarification":
        return {
            "tone": "info",
            "headline": "Refinement needs clarification",
            "copy": "The prior runtime result still stands, but the latest refinement needs follow-up before it can be applied.",
        }
    if page_state == "restart_required":
        return {
            "tone": "warning",
            "headline": "This case is complete",
            "copy": "Same-case continuation is unavailable. Start a new case to continue.",
        }
    return None


def _resolve_composer_mode(session: SessionDetail, *, page_state: str) -> str:
    return _resolve_composer_mode_from_page_state(session, page_state=page_state)


def _build_composer_context(
    session,
    *,
    composer_mode: str,
    page_state: str,
    advanced_controls_relevant: bool = False,
) -> ComposerContextPayload:
    is_read_only = bool(
        getattr(session, "is_read_only", getattr(session, "lifecycle_status", None) == "archived")
    )
    refinement_allowed = bool(
        getattr(session, "refinement_allowed", _serialize_refinement_allowed(session))
    )
    if composer_mode == COMPOSER_MODE_DISABLED:
        return ComposerContextPayload(
            mode=COMPOSER_MODE_DISABLED,
            submit_target=None,
            mode_chip_text=None,
            title=None,
            help_text=_restart_helper_copy(session),
            placeholder=None,
            button_label=None,
            advanced_controls_relevant=False,
            hidden=True,
            disabled=True,
        )

    if composer_mode == COMPOSER_MODE_REFINEMENT:
        help_text = (
            "Continue the current case in natural language."
        )
        placeholder = (
            "Keep the current case but refine it naturally, for example: Do not change Income. "
            "Keep Mortgage below 120."
        )
        if page_state == "refinement_clarification":
            help_text = "Clarify the pending refinement so the current case can continue."
            placeholder = "Clarify the refinement intent naturally, for example: Keep max changed features at one."
        return ComposerContextPayload(
            mode=COMPOSER_MODE_REFINEMENT,
            submit_target="refinements",
            mode_chip_text="Continuing this case",
            title="Refine this case",
            help_text=help_text,
            placeholder=placeholder,
            button_label="Apply Refinement",
            advanced_controls_relevant=advanced_controls_relevant,
            hidden=False,
            disabled=bool(is_read_only or not refinement_allowed),
        )

    if page_state == "clarification":
        help_text = "Add the missing bank-profile details to continue the current case."
        placeholder = "Add the missing profile details to continue this case."
    else:
        help_text = "Describe the target bank profile naturally."
        placeholder = "Describe the target bank profile naturally."
    return ComposerContextPayload(
        mode=COMPOSER_MODE_MESSAGE,
        submit_target="messages",
        mode_chip_text=None,
        title="Message",
        help_text=help_text,
        placeholder=placeholder,
        button_label="Send Message",
        advanced_controls_relevant=False,
        hidden=False,
        disabled=is_read_only,
    )


def _restart_helper_copy(session: SessionDetail) -> str:
    if session.case_completion_reason == "runtime_success":
        return "A final runtime-backed recommendation is already available for this case. Start a new case to submit another request."
    if session.case_completion_reason == "runtime_reject":
        return "This case ended in a final runtime-backed reject outcome. Start a new case to try a different request."
    if session.case_completion_reason == "clarification_limit_reached":
        return "The clarification limit was reached for this case. Start a new case with one complete bank profile."
    if session.case_completion_reason == "conflict":
        return "This case ended in conflict. Start a new case with one corrected bank profile."
    if session.case_completion_reason == "unsupported_request":
        return "This request is outside the supported bank-profile flow. Start a new case with one complete bank profile."
    if session.case_completion_reason == "parser_failure":
        return "The request could not be safely interpreted. Start a new case and resubmit one complete bank profile."
    return "Same-case continuation is unavailable. Start a new case to continue."


def _show_lifecycle_badge(session: SessionDetail) -> bool:
    return session.lifecycle_status != "active"


def _build_chat_header_summary(
    session: SessionDetail,
    *,
    page_state: str,
) -> dict[str, Any]:
    meta_line = f"Session {session.session_id} · {session.turn_count} turns"
    if session.clarification_turns_used > 0:
        meta_line = f"{meta_line} · {session.clarification_turns_used} clarification turns"
    return {
        "dataset_label": f"Dataset {session.dataset_key}",
        "state_label": SESSION_PAGE_STATE_LABELS.get(page_state, page_state),
        "state_tone": page_state,
        "meta_line": meta_line,
    }


def _build_transcript_items(
    session: SessionDetail,
    *,
    turns: list[dict[str, Any]],
    page_state: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not turns:
        session_hints = session.render_hints
        welcome_text = (
            session_hints.primary_chat_text
            if session_hints is not None and session_hints.primary_chat_text
            else _build_fresh_welcome_message(session)
        )
        return [
            {
                "kind": "assistant_message",
                "id": "welcome-assistant",
                "turn_id": None,
                "text": welcome_text,
                "pending": False,
            }
        ]
    for turn in reversed(turns):
        items.extend(
            _build_turn_transcript_items(
                turn,
                latest_runtime_backed_turn_id=session.latest_runtime_backed_turn_id,
            )
        )
    if page_state == "restart_required":
        items.append(
            {
                "kind": "assistant_message",
                "id": "restart-terminal-note",
                "turn_id": None,
                "text": _restart_helper_copy(session),
                "pending": False,
            }
        )
    return items


def _build_turn_transcript_items(
    turn: dict[str, Any],
    *,
    latest_runtime_backed_turn_id: str | None,
) -> list[dict[str, Any]]:
    turn_id = _string_or_none(turn.get("turn_id")) or f"turn-{turn.get('turn_index', 'unknown')}"
    items: list[dict[str, Any]] = [
        {
            "kind": "user_message",
            "id": f"{turn_id}:user",
            "turn_id": turn.get("turn_id"),
            "text": str(turn.get("user_input") or ""),
            "pending": False,
            "failed": False,
        }
    ]
    marker = _build_stream_state_marker(turn)
    if marker is not None:
        items.append(
            {
                "kind": "system_marker",
                "id": f"{turn_id}:marker",
                "turn_id": turn.get("turn_id"),
                "tone": marker["tone"],
                "label": marker["label"],
            }
        )
    items.append(
        {
            "kind": "assistant_message",
            "id": f"{turn_id}:assistant",
            "turn_id": turn.get("turn_id"),
            "text": _build_transcript_visible_assistant_text(
                turn,
                latest_runtime_backed_turn_id=latest_runtime_backed_turn_id,
            ),
            "pending": False,
        }
    )
    detail_toggle = _build_inline_detail_toggle_item(
        turn,
        latest_runtime_backed_turn_id=latest_runtime_backed_turn_id,
    )
    if detail_toggle is not None:
        items.append(detail_toggle)
    return items


def _build_fresh_welcome_message(session: SessionDetail) -> str:
    return _build_fresh_welcome_copy(dataset_key=str(session.dataset_key or "bank"))


def _build_transcript_visible_assistant_text(
    turn: dict[str, Any] | None,
    *,
    latest_runtime_backed_turn_id: str | None,
) -> str:
    if not isinstance(turn, dict):
        return ""
    render_hints = turn.get("render_hints")
    if isinstance(render_hints, dict):
        primary_chat_text = _string_or_none(render_hints.get("primary_chat_text"))
        if primary_chat_text:
            return primary_chat_text
    return str(turn.get("assistant_text") or "")


def _build_inline_detail_toggle_item(
    turn: dict[str, Any] | None,
    *,
    latest_runtime_backed_turn_id: str | None,
) -> dict[str, Any] | None:
    if not isinstance(turn, dict):
        return None
    turn_id = _string_or_none(turn.get("turn_id")) or f"turn-{turn.get('turn_index', 'unknown')}"
    render_hints = turn.get("render_hints")
    if not isinstance(render_hints, dict):
        return None
    detail_title = _string_or_none(render_hints.get("supporting_detail_title"))
    detail_body = _string_or_none(render_hints.get("supporting_detail_body"))
    facts = [
        str(item)
        for item in list(render_hints.get("supporting_detail_facts") or [])
        if isinstance(item, str)
    ]
    if not detail_title and not detail_body and not facts:
        return None
    return {
        "kind": "inline_detail_toggle",
        "id": f"{turn_id}:detail",
        "turn_id": turn.get("turn_id"),
        "closed_label": "View details",
        "open_label": "Hide details",
        "detail_title": detail_title or "Details",
        "detail_body": detail_body or "",
        "facts": facts[:3],
    }
    return None


def _build_stream_state_marker(turn: dict[str, Any]) -> dict[str, str] | None:
    render_hints = turn.get("render_hints")
    if not isinstance(render_hints, dict):
        return None
    label = _string_or_none(render_hints.get("state_marker_label"))
    if not label:
        return None
    tone = _marker_tone_from_primary_action_type(
        _string_or_none(render_hints.get("primary_action_type")) or PRIMARY_ACTION_NONE
    )
    if tone is None:
        return None
    return {"tone": tone, "label": label}


def _marker_tone_from_primary_action_type(primary_action_type: str) -> str | None:
    if primary_action_type in {
        PRIMARY_ACTION_PROVIDE_MISSING_FIELDS,
        PRIMARY_ACTION_CLARIFY_REFINEMENT,
    }:
        return "info"
    if primary_action_type == PRIMARY_ACTION_NO_ACTION_REQUIRED:
        return "success"
    if primary_action_type in {
        PRIMARY_ACTION_RELAX_CONSTRAINTS_OR_RESTART,
        PRIMARY_ACTION_START_NEW_CASE,
    }:
        return "warning"
    return None


def _build_inline_outcome_payload(
    turn: dict[str, Any] | None,
    *,
    latest_runtime_backed_turn_id: str | None,
) -> dict[str, Any] | None:
    if not _turn_carries_runtime_outcome(
        turn,
        latest_runtime_backed_turn_id=latest_runtime_backed_turn_id,
    ):
        return None
    summary = _build_latest_runtime_summary(turn)
    if summary is None:
        return None
    return {
        "tone": "success" if summary["kind"] == "success" else "warning",
        "headline": summary["headline"],
        "facts": _build_inline_outcome_facts(summary),
    }


def _build_inline_outcome_facts(summary: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    summary_type = summary.get("summary_type")
    changed_feature_count = summary.get("changed_feature_count")
    reject_mode = summary.get("reject_mode")
    if summary_type == "no_recourse_needed":
        facts.append("No changes needed")
    elif summary_type == "counterfactual_found":
        facts.append("Runtime-validated result")
    elif isinstance(summary_type, str) and summary_type:
        facts.append(summary_type.replace("_", " "))
    if isinstance(changed_feature_count, int):
        plural = "" if changed_feature_count == 1 else "s"
        facts.append(f"{changed_feature_count} key change{plural}")
    elif reject_mode == "constraints_blocked":
        facts.append("Blocked by active constraints")
    elif reject_mode == "invalid_counterfactual_blocked":
        facts.append("Validation blocked exposure")
    return facts[:2]


def _build_review_summary(ui_review: UiReviewPayload | None) -> dict[str, Any]:
    if ui_review is None or not ui_review.profile_fields:
        return {
            "headline": "No interpreted profile yet",
            "summary_line": "Submit a natural-language bank request to populate the review.",
            "constraints_line": "No active hard constraints.",
            "preferences_line": "No active soft preferences.",
        }

    resolved_fields = [
        field
        for field in ui_review.profile_fields
        if not field.missing and field.value is not None and field.display_value != "Not provided"
    ]
    total_fields = len(ui_review.profile_fields)
    provided_fields = len(resolved_fields)
    preview_pairs = [f"{field.label} {field.display_value}" for field in resolved_fields[:3]]
    summary_line = " · ".join(preview_pairs) if preview_pairs else f"{provided_fields}/{total_fields} fields interpreted"
    if not preview_pairs:
        summary_line = f"{provided_fields}/{total_fields} fields interpreted"
    constraints_line = _build_constraint_summary_line(ui_review.constraints, empty_text="No active hard constraints.")
    preferences_line = _build_constraint_summary_line(ui_review.preferences, empty_text="No active soft preferences.")
    return {
        "headline": f"{provided_fields}/{total_fields} profile fields interpreted",
        "summary_line": summary_line,
        "constraints_line": constraints_line,
        "preferences_line": preferences_line,
    }


def _build_constraint_summary_line(items: list[UiReviewConstraint], *, empty_text: str) -> str:
    if not items:
        return empty_text
    labels = [str(item.label) for item in items[:2]]
    summary = " · ".join(labels)
    if len(items) > 2:
        summary = f"{summary} · +{len(items) - 2} more"
    return summary


def _build_next_action_summary(
    session: SessionDetail,
    *,
    page_state: str,
    render_hints: SessionRenderHints,
) -> dict[str, Any] | None:
    if page_state == "fresh":
        return None
    title = "Result" if render_hints.primary_action_type == PRIMARY_ACTION_NO_ACTION_REQUIRED else "Next Action"
    return {
        "title": title,
        "headline": render_hints.primary_chat_text,
        "summary_line": render_hints.primary_chat_text,
        "body": _next_action_body_from_render_hints(render_hints),
        "facts": list(render_hints.primary_action_items),
    }


def _next_action_body_from_render_hints(render_hints: SessionRenderHints) -> str:
    action_type = render_hints.primary_action_type
    if action_type == PRIMARY_ACTION_PROVIDE_MISSING_FIELDS:
        return "Reply with the missing fields to continue the current case."
    if action_type == PRIMARY_ACTION_NO_ACTION_REQUIRED:
        return "No further changes are required unless you want to refine the same case."
    if action_type == PRIMARY_ACTION_RELAX_CONSTRAINTS_OR_RESTART:
        return "Relax constraints or start a new case."
    if action_type == PRIMARY_ACTION_CLARIFY_REFINEMENT:
        return "Clarify the pending refinement so the current case can continue."
    if action_type == PRIMARY_ACTION_START_NEW_CASE:
        return "This case is complete. Start a new case to check another bank profile."
    if action_type == PRIMARY_ACTION_START_CASE:
        return "Describe one bank profile in natural language to start a new case."
    return render_hints.supporting_detail_body or render_hints.primary_chat_text


def _build_clarification_summary_body(
    ui_review: UiReviewPayload | None,
    latest_turn: dict[str, Any] | None,
) -> str:
    if ui_review is not None and ui_review.clarification_message:
        return ui_review.clarification_message
    if latest_turn is not None and isinstance(latest_turn.get("assistant_text"), str):
        return str(latest_turn["assistant_text"])
    return "More details are required before runtime can continue."


def _build_context_sections(
    *,
    page_state: str,
    review_summary: dict[str, Any],
    next_action_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    sections = [
        {
            "id": "review",
            "title": "Review",
            "summary_line": review_summary["summary_line"],
            "open": False,
        }
    ]
    if next_action_summary is not None:
        sections.append(
            {
                "id": "next-action",
                "title": next_action_summary["title"],
                "summary_line": next_action_summary["headline"],
                "open": False,
            }
        )
    sections.append(
        {
            "id": "technical",
            "title": "Technical Details",
            "summary_line": "Artifacts, payloads, and raw traces",
            "open": False,
        }
    )
    return sections


def _build_explanation_summary_line(latest_runtime_turn: dict[str, Any]) -> str:
    assistant_text = str(latest_runtime_turn.get("assistant_text") or "").strip()
    if not assistant_text:
        return "Short explanation available"
    collapsed = " ".join(assistant_text.split())
    if len(collapsed) <= 80:
        return collapsed
    return f"{collapsed[:77].rstrip()}..."


def _show_advanced_controls_by_default(render_hints: SessionRenderHints) -> bool:
    return bool(
        render_hints.page_state == "refinement_clarification"
        and render_hints.composer_context.advanced_controls_relevant
    )


def _build_session_ui_review(session, *, latest_turn_payload: dict[str, Any] | None = None) -> UiReviewPayload:
    public_state = session.current_public_state
    turn_kind = "message"
    refinement_status = "clarification_required" if session.pending_refinement_clarification_json is not None else None
    clarification_payload = _build_session_level_clarification_payload(session)
    canonical_state = session.canonical_session_state_json
    active_constraint_spec = session.active_constraint_spec_json
    last_updated_turn_id = session.latest_turn_id

    if isinstance(latest_turn_payload, dict):
        public_state = latest_turn_payload.get("public_state", public_state)
        turn_kind = latest_turn_payload.get("turn_kind", turn_kind)
        refinement_status = latest_turn_payload.get("refinement_status", refinement_status)
        clarification_payload = latest_turn_payload.get("clarification_payload", clarification_payload)
        canonical_state = latest_turn_payload.get("canonical_session_state") or canonical_state
        if latest_turn_payload.get("active_constraint_spec") is not None:
            active_constraint_spec = latest_turn_payload.get("active_constraint_spec")
        last_updated_turn_id = latest_turn_payload.get("turn_id", last_updated_turn_id)

    return _build_ui_review_payload(
        public_state=public_state,
        turn_kind=turn_kind,
        clarification_payload=clarification_payload,
        refinement_status=refinement_status,
        canonical_state=canonical_state,
        active_constraint_spec=active_constraint_spec,
        pending_profile=_pending_profile_from_session(session),
        pending_constraint_spec=_pending_constraint_spec_from_session(session),
        last_updated_turn_id=last_updated_turn_id,
        profile_editable=_session_profile_editable(session),
        refinement_editable=_serialize_refinement_allowed(session),
    )


def _build_turn_ui_review(payload: dict[str, Any]) -> UiReviewPayload:
    return _build_ui_review_payload(
        public_state=payload.get("public_state"),
        turn_kind=payload.get("turn_kind"),
        clarification_payload=payload.get("clarification_payload"),
        refinement_status=payload.get("refinement_status"),
        canonical_state=payload.get("canonical_session_state"),
        active_constraint_spec=payload.get("active_constraint_spec"),
        pending_profile=None,
        pending_constraint_spec=None,
        last_updated_turn_id=payload.get("turn_id"),
        profile_editable=_turn_profile_editable(payload),
        refinement_editable=_turn_refinement_editable(payload),
    )


def _build_ui_review_payload(
    *,
    public_state: str | None,
    turn_kind: str | None,
    clarification_payload: dict[str, Any] | None,
    refinement_status: str | None,
    canonical_state: dict[str, Any] | None,
    active_constraint_spec: dict[str, Any] | None,
    pending_profile: dict[str, Any] | None,
    pending_constraint_spec: dict[str, Any] | None,
    last_updated_turn_id: str | None,
    profile_editable: bool,
    refinement_editable: bool,
) -> UiReviewPayload:
    profile_facts = _extract_profile_facts(canonical_state, pending_profile)
    hard_constraints, soft_preferences = _extract_constraint_buckets(
        active_constraint_spec=active_constraint_spec,
        canonical_state=canonical_state,
        pending_constraint_spec=pending_constraint_spec,
    )
    missing_fields = _ordered_fields(
        [] if not isinstance(clarification_payload, dict) else clarification_payload.get("missing_fields")
    )
    constraints = _build_constraint_entries(
        hard_constraints,
        editable=bool(profile_editable or refinement_editable),
    )
    preferences = _build_preference_entries(
        soft_preferences,
        editable=bool(profile_editable or refinement_editable),
    )
    return UiReviewPayload(
        display_state=_resolve_ui_display_state(
            public_state=public_state,
            clarification_payload=clarification_payload,
            refinement_status=refinement_status,
            profile_facts=profile_facts,
            constraints=constraints,
            preferences=preferences,
        ),
        profile_fields=_build_profile_fields(
            profile_facts=profile_facts,
            missing_fields=missing_fields,
            editable=profile_editable,
        ),
        constraints=constraints,
        preferences=preferences,
        missing_fields=missing_fields,
        clarification_type=_string_or_none(
            None if not isinstance(clarification_payload, dict) else clarification_payload.get("clarification_type")
        ),
        clarification_message=_build_clarification_message(clarification_payload, refinement_status=refinement_status),
        clarification_next_input=_string_or_none(
            None if not isinstance(clarification_payload, dict) else clarification_payload.get("next_required_input")
        ),
        remaining_rounds=_as_optional_int(
            None if not isinstance(clarification_payload, dict) else clarification_payload.get("remaining_rounds")
        ),
        refinement_status=_string_or_none(refinement_status),
        last_updated_turn_id=_string_or_none(last_updated_turn_id),
        read_only=not profile_editable,
        profile_editable=bool(profile_editable),
        refinement_editable=bool(refinement_editable),
    )


def _build_session_level_clarification_payload(session) -> dict[str, Any] | None:
    if isinstance(session.pending_refinement_clarification_json, dict):
        payload = dict(session.pending_refinement_clarification_json)
        return {
            "clarification_type": "refinement_clarification",
            "missing_fields": [],
            "conflicts": list(payload.get("ambiguities") or []),
            "next_required_input": payload.get("next_required_input"),
            "remaining_rounds": max(session.refinement_round_limit - session.refinement_rounds_used, 0),
            "restart_required": False,
            "reply_strategy": "start_new_case",
            "carried_forward_fields": [],
        }
    if isinstance(session.pending_clarification_json, dict):
        payload = dict(session.pending_clarification_json)
        carried_forward_fields = _ordered_fields(list(dict(payload.get("prior_cf_request") or {}).keys()))
        return {
            "clarification_type": "missing_information",
            "missing_fields": _ordered_fields(payload.get("missing_fields")),
            "conflicts": [],
            "next_required_input": (
                "Reply with only the missing fields. "
                f"I'll keep the values already provided for {_format_field_list(carried_forward_fields)}."
                if carried_forward_fields
                else "Reply with only the missing fields."
            ),
            "remaining_rounds": max(3 - session.clarification_turns_used, 0),
            "restart_required": False,
            "reply_strategy": "missing_fields_only",
            "carried_forward_fields": carried_forward_fields,
        }
    if session.case_completion_reason == "clarification_limit_reached" and session.current_public_state == "NEEDS_CLARIFICATION":
        return {
            "clarification_type": "clarification_limit_reached",
            "missing_fields": [],
            "conflicts": [],
            "next_required_input": "Start a new case and submit one complete bank profile.",
            "remaining_rounds": 0,
            "restart_required": True,
            "reply_strategy": "start_new_case",
            "carried_forward_fields": [],
        }
    return None


def _pending_profile_from_session(session) -> dict[str, Any] | None:
    if not isinstance(session.pending_clarification_json, dict):
        return None
    prior_cf_request = session.pending_clarification_json.get("prior_cf_request")
    if not isinstance(prior_cf_request, dict):
        return None
    return dict(prior_cf_request)


def _pending_constraint_spec_from_session(session) -> dict[str, Any] | None:
    if not isinstance(session.pending_clarification_json, dict):
        return None
    prior_constraint_spec = session.pending_clarification_json.get("prior_constraint_spec")
    if not isinstance(prior_constraint_spec, dict):
        return None
    return dict(prior_constraint_spec)


def _extract_profile_facts(
    canonical_state: dict[str, Any] | None,
    pending_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(canonical_state, dict) and isinstance(canonical_state.get("profile_facts"), dict):
        return dict(canonical_state.get("profile_facts") or {})
    if isinstance(pending_profile, dict):
        return dict(pending_profile)
    return {}


def _extract_constraint_buckets(
    *,
    active_constraint_spec: dict[str, Any] | None,
    canonical_state: dict[str, Any] | None,
    pending_constraint_spec: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(active_constraint_spec, dict):
        return split_constraint_buckets(
            active_constraint_spec,
            feature_order=list(BANK_REQUIRED_FIELD_ORDER),
        )
    if isinstance(canonical_state, dict):
        return (
            dict(canonical_state.get("hard_constraints") or {}),
            dict(canonical_state.get("soft_preferences") or {}),
        )
    if isinstance(pending_constraint_spec, dict):
        return split_constraint_buckets(
            pending_constraint_spec,
            feature_order=list(BANK_REQUIRED_FIELD_ORDER),
        )
    return {}, {}


def _build_profile_fields(
    *,
    profile_facts: dict[str, Any],
    missing_fields: list[str],
    editable: bool,
) -> list[UiReviewField]:
    missing = set(missing_fields)
    fields: list[UiReviewField] = []
    for field_name in BANK_REQUIRED_FIELD_ORDER:
        value = profile_facts.get(field_name)
        field_type = BANK_FEATURE_TYPES[field_name]
        fields.append(
            UiReviewField(
                field_name=field_name,
                label=field_name,
                value=value,
                display_value=_format_field_value(field_name, value, missing=field_name in missing),
                feature_kind=field_type,
                missing=field_name in missing,
                editable=bool(editable),
                step=None if field_name in BANK_BOOLEAN_FIELDS else float(BANK_STEP[field_name]),
            )
        )
    return fields


def _build_constraint_entries(hard_constraints: dict[str, Any], *, editable: bool) -> list[UiReviewConstraint]:
    entries: list[UiReviewConstraint] = []
    blocked_fields = hard_constraints.get("disallowed_changes")
    for field_name in _ordered_fields(blocked_fields):
        entries.append(
            UiReviewConstraint(
                key=f"disallowed_changes:{field_name}",
                label=f"Do not change {field_name}",
                value=field_name,
                display_value=field_name,
                kind="disallowed_change",
                field_name=field_name,
                editable=bool(editable),
            )
        )
    numeric_bounds = hard_constraints.get("numeric_bounds")
    if isinstance(numeric_bounds, dict):
        for field_name in BANK_REQUIRED_FIELD_ORDER:
            bounds = numeric_bounds.get(field_name)
            if not isinstance(bounds, dict) or not bounds:
                continue
            entries.append(
                UiReviewConstraint(
                    key=f"numeric_bounds:{field_name}",
                    label=f"{field_name} bound",
                    value=dict(bounds),
                    display_value=_render_numeric_bounds(field_name, bounds),
                    kind="numeric_bounds",
                    field_name=field_name,
                    editable=bool(editable),
                )
            )
    max_changed_features = hard_constraints.get("max_changed_features")
    if isinstance(max_changed_features, int) and not isinstance(max_changed_features, bool):
        plural = "" if int(max_changed_features) == 1 else "s"
        entries.append(
            UiReviewConstraint(
                key="max_changed_features",
                label="Max changed features",
                value=int(max_changed_features),
                display_value=f"At most {int(max_changed_features)} changed feature{plural}",
                kind="max_changed_features",
                editable=bool(editable),
            )
        )
    return entries


def _build_preference_entries(soft_preferences: dict[str, Any], *, editable: bool) -> list[UiReviewConstraint]:
    entries: list[UiReviewConstraint] = []
    prefer_fewer_changes = soft_preferences.get("prefer_fewer_changes")
    if isinstance(prefer_fewer_changes, bool) and prefer_fewer_changes:
        entries.append(
            UiReviewConstraint(
                key="prefer_fewer_changes",
                label="Prefer fewer changes",
                value=True,
                display_value="Prefer fewer changes",
                kind="prefer_fewer_changes",
                editable=bool(editable),
            )
        )
    return entries


def _resolve_ui_display_state(
    *,
    public_state: str | None,
    clarification_payload: dict[str, Any] | None,
    refinement_status: str | None,
    profile_facts: dict[str, Any],
    constraints: list[UiReviewConstraint],
    preferences: list[UiReviewConstraint],
) -> str:
    if refinement_status == "clarification_required":
        return "refinement_clarification_view"
    if refinement_status == "applied":
        return "refinement_applied_view"
    if isinstance(clarification_payload, dict):
        return "needs_clarification_input"
    if public_state == "RUNTIME_SUCCESS":
        return "runtime_success_view"
    if public_state == "RUNTIME_REJECT":
        return "runtime_reject_view"
    if profile_facts or constraints or preferences:
        return "review_ready"
    return "draft"


def _build_clarification_message(
    clarification_payload: dict[str, Any] | None,
    *,
    refinement_status: str | None,
) -> str | None:
    if refinement_status == "clarification_required" and not isinstance(clarification_payload, dict):
        return "Clarification is required before the refinement can be applied."
    if not isinstance(clarification_payload, dict):
        return None
    clarification_type = str(clarification_payload.get("clarification_type") or "")
    reply_strategy = str(clarification_payload.get("reply_strategy") or "")
    conflicts = [str(item) for item in list(clarification_payload.get("conflicts") or []) if isinstance(item, str)]
    missing_fields = _ordered_fields(clarification_payload.get("missing_fields"))
    carried_forward_fields = _ordered_fields(clarification_payload.get("carried_forward_fields"))
    next_required_input = _string_or_none(clarification_payload.get("next_required_input"))
    if clarification_type == "clarification_limit_reached":
        return "The clarification limit was reached. Start a new case with one complete bank profile."
    if clarification_type == "refinement_clarification":
        if conflicts:
            return conflicts[0]
        return "Clarification is required before the refinement can be applied."
    if clarification_type == "conflict_resolution":
        if conflicts:
            conflict_text = "; ".join(conflicts)
            return f"Your request contains conflicting instructions: {conflict_text}. Start a new case and submit one corrected bank profile."
        return "Your request contains conflicting instructions. Start a new case and submit one corrected bank profile."
    if reply_strategy == "missing_fields_only" and missing_fields:
        missing_text = _format_field_list(missing_fields)
        if carried_forward_fields:
            return (
                f"Reply with only the missing fields: {missing_text}. "
                f"I'll keep the values already provided for {_format_field_list(carried_forward_fields)}."
            )
        return f"Reply with only the missing fields: {missing_text}."
    if reply_strategy == "start_new_case":
        return next_required_input or "Start a new case and submit one corrected bank profile."
    if missing_fields:
        missing_text = _format_field_list(missing_fields)
        if carried_forward_fields:
            return (
                f"Reply with only the missing fields: {missing_text}. "
                f"I'll keep the values already provided for {_format_field_list(carried_forward_fields)}."
            )
        return f"Reply with only the missing fields: {missing_text}."
    return next_required_input


def _format_field_list(fields: list[str]) -> str:
    ordered = [field for field in fields if isinstance(field, str) and field]
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


def _format_field_value(field_name: str, value: Any, *, missing: bool) -> str:
    if missing or value is None:
        return "Not provided"
    if field_name in BANK_BOOLEAN_FIELDS:
        return "Yes" if bool(value) else "No"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _render_numeric_bounds(field_name: str, bounds: dict[str, Any]) -> str:
    minimum = bounds.get("min")
    maximum = bounds.get("max")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        return f"{field_name}: {_format_number(float(minimum))} to {_format_number(float(maximum))}"
    if isinstance(minimum, (int, float)):
        return f"{field_name}: at least {_format_number(float(minimum))}"
    if isinstance(maximum, (int, float)):
        return f"{field_name}: at most {_format_number(float(maximum))}"
    return f"{field_name}: bound active"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _ordered_fields(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    valid = {field_name for field_name in BANK_REQUIRED_FIELD_ORDER}
    requested = [str(item) for item in values if isinstance(item, str) and str(item) in valid]
    requested_set = set(requested)
    return [field_name for field_name in BANK_REQUIRED_FIELD_ORDER if field_name in requested_set]


def _session_profile_editable(session) -> bool:
    if session.lifecycle_status == "archived":
        return False
    if session.is_case_complete:
        return False
    if session.pending_refinement_clarification_json is not None:
        return False
    return bool(session.pending_clarification_json is not None or session.current_public_state == "NEEDS_CLARIFICATION")


def _turn_profile_editable(payload: dict[str, Any]) -> bool:
    clarification_payload = payload.get("clarification_payload")
    clarification_type = None
    if isinstance(clarification_payload, dict):
        clarification_type = clarification_payload.get("clarification_type")
    return (
        payload.get("turn_kind") == "message"
        and payload.get("public_state") == "NEEDS_CLARIFICATION"
        and payload.get("is_case_complete") is False
        and clarification_type != "refinement_clarification"
    )


def _turn_refinement_editable(payload: dict[str, Any]) -> bool:
    if payload.get("refinement_status") == "clarification_required":
        return True
    return bool(
        payload.get("is_case_complete")
        and payload.get("public_state") in RUNTIME_COMPLETE_STATES
    )


def _string_or_none(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def build_refinement_blocked_response(
    session,
    *,
    error_code: str,
    detail: str,
    refinement_status: str | None = None,
    restart_required: bool | None = None,
) -> RefinementBlockedResponse:
    return RefinementBlockedResponse(
        error_code=error_code,
        detail=detail,
        current_public_state=session.current_public_state,
        case_completion_reason=session.case_completion_reason,
        active_constraint_spec=session.active_constraint_spec_json or {},
        refinement_revision_index=session.refinement_revision_index,
        refinement_rounds_used=session.refinement_rounds_used,
        refinement_round_limit=session.refinement_round_limit,
        restart_required=session.restart_required if restart_required is None else restart_required,
        refinement_status=refinement_status,
    )


def _serialize_refinement_allowed(session) -> bool:
    existing = getattr(session, "refinement_allowed", None)
    if isinstance(existing, bool):
        return existing
    if session.lifecycle_status == "archived":
        return False
    if session.refinement_rounds_used >= session.refinement_round_limit:
        return False
    pending_refinement = getattr(
        session,
        "pending_refinement_clarification_json",
        None if not getattr(session, "has_pending_refinement_clarification", False) else True,
    )
    if pending_refinement is not None:
        return True
    return (
        session.is_case_complete
        and session.current_public_state in {"RUNTIME_SUCCESS", "RUNTIME_REJECT"}
        and getattr(session, "last_runtime_request_json", {"_session_detail": True}) is not None
        and session.latest_runtime_backed_turn_id is not None
    )
