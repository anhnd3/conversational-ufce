from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from llm.src.conversation.session import (
    InteractiveSessionState,
    SessionCaseCompleteError,
    handle_session_turn,
)
from llm.src.conversation.canonical_session_state import (
    build_canonical_session_state_for_turn_result,
    split_constraint_buckets,
)
from llm.src.conversation.types import PendingClarification
from llm.src.product.catalog import build_dataset_catalog
from llm.src.product.config import ProductConfig, try_get_git_commit
from llm.src.product.persistence import SessionRepository, StoredSession, StoredTurn
from llm.src.refinement.orchestrator import ConstraintRefinementOrchestrator
from llm.src.refinement.delta import build_active_constraint_spec
from llm.src.refinement.types import (
    PendingRefinementClarification,
)
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.state.session_state_builder import build_session_state_from_turn
from llm.src.runtime.reason_codes import INVALID_COUNTERFACTUAL_BLOCKED


MESSAGE_ORDER_ASC = "asc"
MESSAGE_ORDER_DESC = "desc"
REFINEMENT_ROUND_LIMIT = 3
PREVIEWABLE_SUFFIXES = {
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}
MAX_PREVIEW_BYTES = 262144


class SessionArchivedError(RuntimeError):
    pass


class RefinementNotAllowedError(RuntimeError):
    pass


class RefinementLimitReachedError(RuntimeError):
    pass


class ProductSessionService:
    def __init__(self, *, orchestrator, repository: SessionRepository, config: ProductConfig) -> None:
        self.orchestrator = orchestrator
        self.repository = repository
        self.config = config
        self.dataset_registry = getattr(
            orchestrator,
            "dataset_registry",
            getattr(getattr(orchestrator, "runtime_orchestrator", None), "dataset_registry", None),
        )
        runtime_orchestrator = getattr(orchestrator, "runtime_orchestrator", None)
        if hasattr(runtime_orchestrator, "model_registry") and hasattr(runtime_orchestrator, "policy_registry"):
            self.catalog_runtime_orchestrator = runtime_orchestrator
        else:
            self.catalog_runtime_orchestrator = RuntimeOrchestrator(runtime_mode=config.product_mode)
        self.refinement_orchestrator = ConstraintRefinementOrchestrator(
            parser_adapter=self.orchestrator.parser_adapter,
            runtime_orchestrator=self.orchestrator.runtime_orchestrator,
            benchmark=self.orchestrator.benchmark,
            output_root=self.orchestrator.output_root,
            model_alias=self.orchestrator.model_alias,
        )

    def create_session(self, dataset_key: str = "bank") -> StoredSession:
        normalized_dataset_key = str(dataset_key or "bank").strip().lower() or "bank"
        if not _session_dataset_supported(normalized_dataset_key, dataset_registry=self.dataset_registry):
            raise ValueError(f"Unsupported session dataset: {dataset_key}")
        return self.repository.create_session(
            model_alias=self.config.model_alias,
            runtime_mode=self.config.product_mode,
            dataset_key=normalized_dataset_key,
        )

    def list_sessions(self) -> list[StoredSession]:
        return self.repository.list_sessions()

    def get_session(self, session_id: str) -> StoredSession:
        return self.repository.get_session(session_id)

    def submit_message(self, session_id: str, user_input: str):
        stored_session = self.repository.get_session(session_id)
        if stored_session.lifecycle_status == "archived":
            raise SessionArchivedError(f"Session {session_id} is archived and read-only.")
        if stored_session.is_case_complete:
            raise SessionCaseCompleteError(
                "This case is complete. Start a new case before sending another message."
            )
        state = InteractiveSessionState(
            session_id=stored_session.session_id,
            dataset_id=stored_session.dataset_key,
            turn_index=stored_session.last_turn_index,
            pending_clarification=_deserialize_pending(stored_session.pending_clarification_json),
            canonical_session_state=stored_session.canonical_session_state_json,
            clarification_turns_used=stored_session.clarification_turns_used,
            is_case_complete=stored_session.is_case_complete,
            case_completion_reason=stored_session.case_completion_reason,
            restart_required=stored_session.restart_required,
        )
        result = handle_session_turn(
            self.orchestrator,
            state,
            user_input=user_input,
            save_artifacts=True,
            scenario_slug=f"api_{session_id}_turn_{stored_session.last_turn_index + 1}",
            debug_trace_enabled=False,
            command=f"POST /api/{self.config.api_version}/sessions/{session_id}/messages",
            dataset_id=stored_session.dataset_key,
        )
        public_state = _extract_public_state(result)
        debug_summary = build_debug_summary(result)
        updated_runtime_request = _extract_runtime_request_for_session_update(result, stored_session)
        active_constraint_spec_session = _extract_active_constraint_spec_for_session_update(
            updated_runtime_request,
            result,
            stored_session,
        )
        canonical_session_state = _build_canonical_session_state_for_turn(
            result=result,
            stored_session=stored_session,
            runtime_request=updated_runtime_request,
            active_constraint_spec=active_constraint_spec_session,
            default_backend_id=getattr(self.orchestrator.runtime_orchestrator, "counterfactual_backend_name", "ufce"),
        )
        stored_turn = self.repository.save_turn(
            session_id=session_id,
            turn_index=state.turn_index,
            user_input=result.user_input,
            assistant_text=result.response_text,
            public_state=public_state,
            builder_status=None if result.builder_result is None else result.builder_result.builder_status,
            transition_reason=None
            if result.negotiation_transition is None
            else result.negotiation_transition.transition_reason,
            response_decision_json=None if result.response_decision is None else result.response_decision.to_dict(),
            clarification_payload_json=None
            if result.clarification_payload is None
            else result.clarification_payload.to_dict(),
            explanation_payload_json=None
            if result.explanation_payload is None
            else result.explanation_payload.to_dict(),
            debug_summary_json=debug_summary,
            artifact_dir=None if result.artifact_record is None else result.artifact_record.output_dir,
            parent_turn_id=None if result.artifact_record is None else result.artifact_record.parent_turn_id,
            merge_applied=False if result.artifact_record is None else result.artifact_record.merge_applied,
            pending_clarification_json=None if state.pending_clarification is None else state.pending_clarification.to_dict(),
            turn_id=result.turn_id,
            created_at=result.timestamp_utc,
            clarification_turns_used=result.clarification_turns_used,
            is_case_complete=result.is_case_complete,
            case_completion_reason=result.case_completion_reason,
            restart_required=result.restart_required,
            timing_metrics_json=result.timing_metrics,
            turn_kind=result.turn_kind,
            refinement_status=result.refinement_status,
            refinement_revision_index=result.refinement_revision_index,
            parent_terminal_turn_id=result.parent_terminal_turn_id,
            parent_refinement_revision_index=result.parent_refinement_revision_index,
            active_constraint_spec_json=result.active_constraint_spec,
            constraint_feedback_delta_json=result.constraint_feedback_delta,
            refinement_rounds_used=result.refinement_rounds_used or stored_session.refinement_rounds_used,
            refinement_round_limit=result.refinement_round_limit,
            active_constraint_spec_session_json=active_constraint_spec_session,
            last_runtime_request_json=updated_runtime_request,
            refinement_revision_index_session=stored_session.refinement_revision_index,
            refinement_rounds_used_session=stored_session.refinement_rounds_used,
            refinement_round_limit_session=stored_session.refinement_round_limit,
            pending_refinement_clarification_json=stored_session.pending_refinement_clarification_json,
            latest_runtime_backed_turn_id=_extract_latest_runtime_backed_turn_id(result, stored_session),
            canonical_runtime_result_json=_extract_canonical_runtime_result(result),
            verification_artifacts_json=_extract_verification_artifacts(result),
            canonical_session_state_json=canonical_session_state["state"],
            canonical_state_source=canonical_session_state["source"],
            canonical_mirror_ok=canonical_session_state["mirror_ok"],
        )
        return stored_turn

    def submit_refinement(self, session_id: str, user_feedback: str) -> StoredTurn:
        stored_session = self.repository.get_session(session_id)
        if stored_session.lifecycle_status == "archived":
            raise SessionArchivedError(f"Session {session_id} is archived and read-only.")
        if stored_session.refinement_rounds_used >= stored_session.refinement_round_limit:
            raise RefinementLimitReachedError(
                "The refinement round limit was reached. Start a new case to continue."
            )
        pending_refinement = _deserialize_pending_refinement(stored_session.pending_refinement_clarification_json)
        if not _refinement_allowed(stored_session):
            raise RefinementNotAllowedError(
                "Refinement is only available after a runtime-backed result or during a pending refinement clarification."
            )

        active_constraint_spec = stored_session.active_constraint_spec_json or {}
        parent_terminal_turn_id = stored_session.latest_runtime_backed_turn_id
        if not parent_terminal_turn_id:
            raise RefinementNotAllowedError("No runtime-backed turn is available for refinement.")
        last_runtime_request = stored_session.last_runtime_request_json
        if not isinstance(last_runtime_request, dict):
            raise RefinementNotAllowedError("The last canonical runtime request is unavailable for refinement.")

        refinement_revision_index = stored_session.refinement_revision_index + 1
        parent_refinement_revision_index = (
            None if stored_session.refinement_revision_index == 0 else stored_session.refinement_revision_index
        )
        refinement_rounds_used = stored_session.refinement_rounds_used + 1
        result, payload = self.refinement_orchestrator.run_turn(
            user_feedback=user_feedback,
            dataset_id=stored_session.dataset_key,
            active_constraint_spec=active_constraint_spec,
            last_runtime_request=last_runtime_request,
            parent_terminal_turn_id=parent_terminal_turn_id,
            parent_refinement_revision_index=parent_refinement_revision_index,
            refinement_revision_index=refinement_revision_index,
            refinement_rounds_used=refinement_rounds_used,
            refinement_round_limit=stored_session.refinement_round_limit,
            parent_public_state=stored_session.current_public_state or "RUNTIME_REJECT",
            parent_case_completion_reason=stored_session.case_completion_reason or "runtime_reject",
            pending_refinement_clarification=pending_refinement,
            save_artifacts=True,
            scenario_slug=f"api_{session_id}_refinement_{refinement_revision_index}",
            debug_trace_enabled=False,
            command=f"POST /api/{self.config.api_version}/sessions/{session_id}/refinements",
            session_trace={
                "session_id": stored_session.session_id,
                "dataset_id": stored_session.dataset_key,
                "turn_index": stored_session.last_turn_index + 1,
                "parent_turn_id": parent_terminal_turn_id,
                "merge_applied": False,
                "carried_fields": [],
                "carried_constraint_keys": [],
                "clarification_turns_used": stored_session.clarification_turns_used,
            },
        )
        public_state = payload["public_state"]
        debug_summary = build_debug_summary(result)
        canonical_session_state = _build_canonical_session_state_for_turn(
            result=result,
            stored_session=stored_session,
            runtime_request=payload["last_runtime_request"] or stored_session.last_runtime_request_json,
            active_constraint_spec=payload["active_constraint_spec"],
            default_backend_id=getattr(self.orchestrator.runtime_orchestrator, "counterfactual_backend_name", "ufce"),
        )
        stored_turn = self.repository.save_turn(
            session_id=session_id,
            turn_index=stored_session.last_turn_index + 1,
            user_input=result.user_input,
            assistant_text=result.response_text,
            public_state=public_state,
            builder_status=None,
            transition_reason=None,
            response_decision_json=None if result.response_decision is None else result.response_decision.to_dict(),
            clarification_payload_json=None
            if result.clarification_payload is None
            else result.clarification_payload.to_dict(),
            explanation_payload_json=None
            if result.explanation_payload is None
            else result.explanation_payload.to_dict(),
            debug_summary_json=debug_summary,
            artifact_dir=None if result.artifact_record is None else result.artifact_record.output_dir,
            parent_turn_id=parent_terminal_turn_id,
            merge_applied=False,
            pending_clarification_json=stored_session.pending_clarification_json,
            turn_id=result.turn_id,
            created_at=result.timestamp_utc,
            clarification_turns_used=stored_session.clarification_turns_used,
            is_case_complete=result.is_case_complete,
            case_completion_reason=result.case_completion_reason,
            restart_required=result.restart_required,
            timing_metrics_json=result.timing_metrics,
            turn_kind=result.turn_kind,
            refinement_status=result.refinement_status,
            refinement_revision_index=result.refinement_revision_index,
            parent_terminal_turn_id=result.parent_terminal_turn_id,
            parent_refinement_revision_index=result.parent_refinement_revision_index,
            active_constraint_spec_json=result.active_constraint_spec,
            constraint_feedback_delta_json=result.constraint_feedback_delta,
            refinement_rounds_used=result.refinement_rounds_used or refinement_rounds_used,
            refinement_round_limit=result.refinement_round_limit,
            active_constraint_spec_session_json=payload["active_constraint_spec"],
            last_runtime_request_json=payload["last_runtime_request"] or stored_session.last_runtime_request_json,
            refinement_revision_index_session=refinement_revision_index,
            refinement_rounds_used_session=refinement_rounds_used,
            refinement_round_limit_session=stored_session.refinement_round_limit,
            pending_refinement_clarification_json=payload["pending_refinement_clarification"],
            latest_runtime_backed_turn_id=payload["latest_runtime_backed_turn_id"],
            canonical_runtime_result_json=_extract_canonical_runtime_result(result),
            verification_artifacts_json=_extract_verification_artifacts(result),
            canonical_session_state_json=canonical_session_state["state"],
            canonical_state_source=canonical_session_state["source"],
            canonical_mirror_ok=canonical_session_state["mirror_ok"],
        )
        return stored_turn

    def list_messages(self, session_id: str, *, order: str = MESSAGE_ORDER_DESC) -> list[StoredTurn]:
        self.repository.get_session(session_id)
        normalized_order = normalize_message_order(order)
        return self.repository.list_turns(session_id, order=normalized_order)

    def archive_session(self, session_id: str) -> StoredSession:
        return self.repository.archive_session(session_id)

    def get_artifact_bundles(self, session_id: str) -> list[dict[str, Any]]:
        self.repository.get_session(session_id)
        turns = self.repository.list_turns(session_id, order=MESSAGE_ORDER_DESC)
        bundles: list[dict[str, Any]] = []
        for turn in turns:
            files = self._manifest_files(turn)
            bundles.append(
                {
                    "turn_id": turn.turn_id,
                    "public_state": turn.public_state,
                    "files": files,
                    "download_urls": {
                        filename: f"/api/{self.config.api_version}/sessions/{session_id}/artifacts/{turn.turn_id}/{filename}"
                        for filename in files
                    },
                    "preview_urls": {
                        filename: f"/api/{self.config.api_version}/sessions/{session_id}/artifacts/{turn.turn_id}/{filename}/preview"
                        for filename in files
                        if is_previewable_filename(filename)
                    },
                }
            )
        return bundles

    def resolve_artifact_path(self, session_id: str, turn_id: str, filename: str) -> Path:
        turn = self.repository.get_turn(session_id, turn_id)
        artifact_dir = _require_artifact_dir(turn)
        allowed = set(self._manifest_files(turn))
        if filename not in allowed:
            raise FileNotFoundError(f"Artifact file is not manifest-listed for turn {turn_id}: {filename}")
        requested = (artifact_dir / filename).resolve()
        artifact_root = artifact_dir.resolve()
        if not str(requested).startswith(str(artifact_root) + "/") and requested != artifact_root / filename:
            raise FileNotFoundError("Artifact path traversal is not allowed.")
        if not requested.exists():
            raise FileNotFoundError(f"Artifact file not found: {filename}")
        return requested

    def get_artifact_preview(self, session_id: str, turn_id: str, filename: str) -> dict[str, Any]:
        resolved = self.resolve_artifact_path(session_id, turn_id, filename)
        suffix = resolved.suffix.lower()
        if suffix not in PREVIEWABLE_SUFFIXES:
            raise FileNotFoundError(f"Artifact preview is not supported for {filename}")
        with resolved.open("rb") as handle:
            data = handle.read(MAX_PREVIEW_BYTES + 1)
        truncated = len(data) > MAX_PREVIEW_BYTES
        payload = data[:MAX_PREVIEW_BYTES]
        return {
            "filename": resolved.name,
            "content": payload.decode("utf-8", errors="replace"),
            "truncated": truncated,
            "byte_count": len(payload),
            "content_type": PREVIEWABLE_SUFFIXES[suffix],
        }

    def health(self) -> dict[str, Any]:
        checks = {
            "database": check_database(self.repository),
            "artifact_store": check_artifact_store(self.config.artifact_root),
            "lm_studio": check_lm_studio(self.config.lm_studio_api_base),
        }
        status = "healthy" if all(item["ok"] for item in checks.values()) else "unhealthy"
        return {"status": status, "checks": checks}

    def version(self) -> dict[str, Any]:
        return {
            "api_version": self.config.api_version,
            "app_version": self.config.app_version,
            "model_alias": self.config.model_alias,
            "parser_schema_version": self.config.parser_schema_version,
            "bank_policy_version": self.config.bank_policy_version,
            "runtime_mode": self.config.product_mode,
            "git_commit": try_get_git_commit(),
        }

    def list_dataset_catalog(self) -> list[dict[str, Any]]:
        return build_dataset_catalog(runtime_orchestrator=self.catalog_runtime_orchestrator)

    def build_turn_response(self, turn: StoredTurn) -> dict[str, Any]:
        artifact_files = self._manifest_files(turn)
        return {
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "turn_index": turn.turn_index,
            "user_input": turn.user_input,
            "assistant_text": turn.assistant_text,
            "public_state": turn.public_state,
            "clarification_payload": turn.clarification_payload_json,
            "explanation_payload": turn.explanation_payload_json,
            "artifact_refs": {
                "turn_id": turn.turn_id,
                "artifact_dir": turn.artifact_dir,
                "files": artifact_files,
                "download_urls": {
                    filename: f"/api/{self.config.api_version}/sessions/{turn.session_id}/artifacts/{turn.turn_id}/{filename}"
                    for filename in artifact_files
                },
                "preview_urls": {
                    filename: f"/api/{self.config.api_version}/sessions/{turn.session_id}/artifacts/{turn.turn_id}/{filename}/preview"
                    for filename in artifact_files
                    if is_previewable_filename(filename)
                },
            },
            "debug_summary": turn.debug_summary_json or build_empty_debug_summary(turn.artifact_dir),
            "clarification_turns_used": turn.clarification_turns_used,
            "is_case_complete": turn.is_case_complete,
            "case_completion_reason": turn.case_completion_reason,
            "restart_required": turn.restart_required,
            "turn_kind": turn.turn_kind,
            "refinement_status": turn.refinement_status,
            "refinement_revision_index": turn.refinement_revision_index,
            "parent_terminal_turn_id": turn.parent_terminal_turn_id,
            "parent_refinement_revision_index": turn.parent_refinement_revision_index,
            "active_constraint_spec": turn.active_constraint_spec_json,
            "constraint_feedback_delta": turn.constraint_feedback_delta_json,
            "refinement_rounds_used": turn.refinement_rounds_used,
            "refinement_round_limit": turn.refinement_round_limit,
            "canonical_session_state": turn.canonical_session_state_json,
        }

    def _manifest_files(self, turn: StoredTurn) -> list[str]:
        artifact_dir = turn.artifact_dir
        if not artifact_dir:
            return []
        manifest_path = Path(artifact_dir) / "artifact_manifest.json"
        if not manifest_path.exists():
            return []
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_files = payload.get("saved_files", [])
        if not isinstance(saved_files, list):
            return []
        return [str(item) for item in saved_files if isinstance(item, str)]


def build_debug_summary(result) -> dict[str, Any]:
    runtime_result = result.runtime_result or {}
    prediction = runtime_result.get("prediction") or {}
    invariant_validation = result.invariant_validation or {}
    builder_result = result.builder_result
    builder_provenance = (
        dict(getattr(builder_result, "provenance", {}) or {})
        if builder_result is not None
        else {}
    )
    return {
        "builder_status": None if builder_result is None else getattr(builder_result, "builder_status", None),
        "builder_reason_codes": []
        if builder_result is None
        else list(getattr(builder_result, "builder_reason_codes", []) or []),
        "transition_reason": None
        if result.negotiation_transition is None
        else result.negotiation_transition.transition_reason,
        "merge_applied": False if result.artifact_record is None else bool(result.artifact_record.merge_applied),
        "carried_fields": []
        if builder_result is None
        else list(getattr(builder_result, "carried_fields", []) or []),
        "carried_constraint_keys": []
        if builder_result is None
        else list(getattr(builder_result, "carried_constraint_keys", []) or []),
        "carried_preference_keys": []
        if builder_result is None
        else list(getattr(builder_result, "carried_preference_keys", []) or []),
        "followup_classification": None
        if builder_result is None
        else builder_provenance.get("followup_classification"),
        "reset_decision": None if builder_result is None else builder_provenance.get("reset_decision"),
        "merge_provenance": None if builder_result is None else builder_provenance.get("merge_provenance"),
        "runtime_summary": {
            "executed": result.runtime_result is not None,
            "controller_state": runtime_result.get("controller_state"),
            "reason_codes": list(runtime_result.get("reason_codes") or []),
            "prediction_score": prediction.get("predicted_proba"),
        },
        "invariant_validation_status": invariant_validation.get("status"),
        "artifact_dir": None if result.artifact_record is None else result.artifact_record.output_dir,
        "timing_metrics": None if result.timing_metrics is None else dict(result.timing_metrics),
    }


def build_empty_debug_summary(artifact_dir: str | None) -> dict[str, Any]:
    return {
        "builder_status": None,
        "builder_reason_codes": [],
        "transition_reason": None,
        "merge_applied": False,
        "carried_fields": [],
        "carried_constraint_keys": [],
        "carried_preference_keys": [],
        "followup_classification": None,
        "reset_decision": None,
        "merge_provenance": None,
        "runtime_summary": {
            "executed": False,
            "controller_state": None,
            "reason_codes": [],
            "prediction_score": None,
        },
        "invariant_validation_status": None,
        "artifact_dir": artifact_dir,
        "timing_metrics": None,
    }


def check_database(repository: SessionRepository) -> dict[str, Any]:
    try:
        metadata = repository.metadata()
        return {
            "ok": int(metadata["db_schema_version"]) >= 1,
            "detail": f"schema_version={metadata['db_schema_version']}",
        }
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def check_artifact_store(artifact_root: Path) -> dict[str, Any]:
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        writable = artifact_root.exists() and artifact_root.is_dir()
        return {"ok": writable, "detail": str(artifact_root)}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def check_lm_studio(api_base: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{api_base.rstrip('/')}/v1/models", timeout=5.0)
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}
    if response.status_code >= 400:
        return {"ok": False, "detail": f"http_{response.status_code}"}
    return {"ok": True, "detail": "reachable"}


def _deserialize_pending(payload: dict[str, Any] | None) -> PendingClarification | None:
    if not isinstance(payload, dict):
        return None
    return PendingClarification(
        prior_cf_request=dict(payload.get("prior_cf_request") or {}),
        prior_constraint_spec=dict(payload.get("prior_constraint_spec") or {}),
        missing_fields=list(payload.get("missing_fields") or []),
        required_field_order=list(payload.get("required_field_order") or []),
        originating_turn_id=str(payload.get("originating_turn_id")),
        prior_field_provenance={
            str(field_name): str(value)
            for field_name, value in dict(payload.get("prior_field_provenance") or {}).items()
            if isinstance(field_name, str) and isinstance(value, str)
        },
    )


def _deserialize_pending_refinement(payload: dict[str, Any] | None) -> PendingRefinementClarification | None:
    if not isinstance(payload, dict):
        return None
    originating_turn_id = payload.get("originating_turn_id")
    next_required_input = payload.get("next_required_input")
    parent_terminal_turn_id = payload.get("parent_terminal_turn_id")
    return PendingRefinementClarification(
        originating_turn_id=str(originating_turn_id) if isinstance(originating_turn_id, str) else "",
        ambiguities=list(payload.get("ambiguities") or []),
        next_required_input=str(next_required_input) if isinstance(next_required_input, str) else "",
        parent_terminal_turn_id=str(parent_terminal_turn_id) if isinstance(parent_terminal_turn_id, str) else "",
        parent_refinement_revision_index=payload.get("parent_refinement_revision_index"),
    )


def _extract_public_state(result) -> str:
    if result.response_decision is not None:
        return result.response_decision.final_public_state
    return result.stage


def _runtime_backed_public_state(public_state: str | None) -> bool:
    return public_state in {"RUNTIME_SUCCESS", "RUNTIME_REJECT"}


def _refinement_allowed(session: StoredSession) -> bool:
    if session.pending_refinement_clarification_json is not None:
        return True
    return (
        session.is_case_complete
        and _runtime_backed_public_state(session.current_public_state)
        and session.last_runtime_request_json is not None
        and session.latest_runtime_backed_turn_id is not None
    )


def _extract_runtime_request_for_session_update(result, stored_session: StoredSession) -> dict[str, Any] | None:
    runtime_request = None
    builder_result = getattr(result, "builder_result", None)
    builder_provenance = (
        dict(getattr(builder_result, "provenance", {}) or {})
        if builder_result is not None
        else {}
    )
    if builder_result is not None:
        runtime_request = result.builder_result.runtime_request
    if builder_provenance.get("reset_decision") == "fresh_request" and not isinstance(runtime_request, dict):
        return None
    if not isinstance(runtime_request, dict):
        return stored_session.last_runtime_request_json
    constraint_spec = runtime_request.get("constraint_spec")
    active_constraint_spec = build_active_constraint_spec(
        constraint_spec,
        feature_order=list(result.builder_result.canonical_field_order),
    )
    updated = dict(runtime_request)
    if active_constraint_spec:
        updated["constraint_spec"] = dict(active_constraint_spec)
    else:
        updated.pop("constraint_spec", None)
    return updated


def _extract_active_constraint_spec_for_session_update(
    runtime_request: dict[str, Any] | None,
    result,
    stored_session: StoredSession,
) -> dict[str, Any] | None:
    builder_result = result.builder_result
    builder_provenance = (
        dict(getattr(builder_result, "provenance", {}) or {})
        if builder_result is not None
        else {}
    )
    if (
        builder_result is not None
        and builder_result.builder_status in {"NEEDS_CLARIFICATION", "READY_FOR_RUNTIME"}
        and isinstance(result.normalized_parse, dict)
    ):
        constraint_spec = result.normalized_parse.get("constraint_spec")
        hard_constraints, soft_preferences = split_constraint_buckets(
            constraint_spec,
            feature_order=list(builder_result.canonical_field_order),
        )
        active_constraint_spec = dict(hard_constraints)
        active_constraint_spec.update(dict(soft_preferences))
        return active_constraint_spec or {}
    if builder_provenance.get("reset_decision") == "fresh_request":
        return {}
    if not isinstance(runtime_request, dict):
        return stored_session.active_constraint_spec_json
    if builder_result is None:
        return stored_session.active_constraint_spec_json
    return build_active_constraint_spec(
        runtime_request.get("constraint_spec"),
        feature_order=list(builder_result.canonical_field_order),
    )


def _extract_latest_runtime_backed_turn_id(result, stored_session: StoredSession) -> str | None:
    if result.stage in {"RUNTIME_SUCCESS", "RUNTIME_REJECT"} and result.runtime_result is not None:
        return result.turn_id
    return stored_session.latest_runtime_backed_turn_id


def _extract_canonical_runtime_result(result) -> dict[str, Any] | None:
    if not isinstance(result.runtime_result, dict):
        return None
    if "canonical_request" not in result.runtime_result and "verification_results" not in result.runtime_result:
        return None
    return dict(result.runtime_result)


def _extract_verification_artifacts(result) -> dict[str, Any] | None:
    if not isinstance(result.runtime_result, dict):
        return None
    verification_results = result.runtime_result.get("verification_results")
    backend_manifest = result.runtime_result.get("backend_manifest")
    backend_id = result.runtime_result.get("backend_id")
    reason_code_version = result.runtime_result.get("reason_code_version")
    if verification_results is None and backend_manifest is None and reason_code_version is None:
        return None
    return {
        "verification_results": list(verification_results or []),
        "backend_manifest": None if backend_manifest is None else dict(backend_manifest),
        "backend_id": backend_id,
        "reason_code_version": reason_code_version,
    }


def _build_canonical_session_state_for_turn(
    *,
    result,
    stored_session: StoredSession,
    runtime_request: dict[str, Any] | None,
    active_constraint_spec: dict[str, Any] | None,
    default_backend_id: str,
) -> dict[str, Any]:
    prior_state = stored_session.canonical_session_state_json
    runtime_result = result.runtime_result if isinstance(result.runtime_result, dict) else {}
    backend_id = runtime_result.get("backend_id") or default_backend_id
    if getattr(result, "turn_kind", "message") == "refinement" or result.builder_result is None:
        dataset_id = stored_session.dataset_key or "bank"
        if isinstance(runtime_request, dict) and isinstance(runtime_request.get("dataset"), str):
            dataset_id = str(runtime_request["dataset"]).strip().lower() or dataset_id
        elif isinstance(prior_state, dict) and isinstance(prior_state.get("dataset_id"), str):
            dataset_id = str(prior_state["dataset_id"]).strip().lower() or dataset_id
        if isinstance(prior_state, dict):
            prior_dataset = prior_state.get("dataset_id")
            if isinstance(prior_dataset, str) and prior_dataset.strip().lower() != dataset_id:
                prior_state = None
        state = build_session_state_from_turn(
            session_id=stored_session.session_id,
            dataset_id=dataset_id,
            backend_id=backend_id,
            runtime_request=runtime_request,
            active_constraint_spec=active_constraint_spec,
            reason_codes=list(runtime_result.get("reason_codes") or []),
            terminal_status=runtime_result.get("controller_state") or result.stage,
            prior_state=prior_state,
        )
        if isinstance(stored_session.canonical_session_state_json, dict):
            prior_backend_id = stored_session.canonical_session_state_json.get("backend_id")
            if isinstance(prior_backend_id, str) and prior_backend_id != backend_id:
                state.rejected_candidate_ids = []
                state.accepted_tradeoffs = []
        mirror_ok = _canonical_state_matches_legacy(
            state=state.to_dict(),
            runtime_request=runtime_request,
            active_constraint_spec=active_constraint_spec,
        )
        state.canonical_mirror_ok = mirror_ok
        return {
            "state": state.to_dict(),
            "source": "canonical_authoritative",
            "mirror_ok": mirror_ok,
        }

    canonical_state = build_canonical_session_state_for_turn_result(
        session_id=stored_session.session_id,
        prior_state=prior_state,
        result=result,
        backend_id=backend_id,
        feature_order=list(getattr(result.builder_result, "canonical_field_order", []) or []),
        dataset_id=stored_session.dataset_key,
    )
    state_payload = dict(canonical_state["state"])
    if isinstance(stored_session.canonical_session_state_json, dict):
        prior_backend_id = stored_session.canonical_session_state_json.get("backend_id")
        if isinstance(prior_backend_id, str) and prior_backend_id != backend_id:
            state_payload["rejected_candidate_ids"] = []
            state_payload["accepted_tradeoffs"] = []
            canonical_state["state"] = state_payload
    return canonical_state


def _canonical_state_matches_legacy(
    *,
    state: dict[str, Any],
    runtime_request: dict[str, Any] | None,
    active_constraint_spec: dict[str, Any] | None,
) -> bool:
    if not isinstance(state, dict):
        return False
    profile = None if not isinstance(runtime_request, dict) else runtime_request.get("profile")
    if isinstance(profile, dict) and dict(state.get("profile_facts") or {}) != dict(profile):
        return False
    if isinstance(active_constraint_spec, dict) and dict(state.get("hard_constraints") or {}) != dict(active_constraint_spec):
        return False
    return True


def _require_artifact_dir(turn: StoredTurn) -> Path:
    if not turn.artifact_dir:
        raise FileNotFoundError(f"Turn {turn.turn_id} has no artifact directory.")
    return Path(turn.artifact_dir)


def normalize_message_order(order: str) -> str:
    normalized = str(order).strip().lower()
    if normalized not in {MESSAGE_ORDER_ASC, MESSAGE_ORDER_DESC}:
        raise ValueError(f"Unsupported message order: {order}")
    return normalized


def is_previewable_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in PREVIEWABLE_SUFFIXES


def _session_dataset_supported(dataset_key: str, *, dataset_registry) -> bool:
    if dataset_registry is None:
        return dataset_key == "bank"
    return dataset_registry.has(dataset_key)
