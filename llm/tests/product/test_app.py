from __future__ import annotations

import asyncio
from html import unescape
import json
from pathlib import Path
import re

import httpx
import pytest

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.product.app import _build_session_page_render_context, create_app, serialize_session_detail
from llm.src.product.config import ProductConfig
from llm.src.product.persistence import SessionRepository, StoredSession
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.tests.conversation.test_conversation_orchestrator import (
    StubParserAdapter,
    StubResult,
    StubRuntimeOrchestrator,
)
from llm.tests.conversation.test_session import QueueParserAdapter


def build_config(tmp_path):
    return ProductConfig(
        lm_studio_api_base="http://localhost:1234",
        model_alias="stub-model",
        product_mode="stable_demo",
        artifact_root=(tmp_path / "artifacts").resolve(),
        sqlite_path=(tmp_path / "sessions.sqlite3").resolve(),
        api_version="v1",
        app_version="phase3_2_test",
        parser_schema_version="parser_schema_v1",
        bank_policy_version="bank_policy_v1",
        host="127.0.0.1",
        port=8000,
    )


def api_request(app, method: str, url: str, **kwargs):
    async def _request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(method, url, **kwargs)
            await response.aread()
            return response

    return asyncio.run(_request())


def element_text(html: str, element_id: str) -> str | None:
    pattern = rf'<[^>]+id="{re.escape(element_id)}"[^>]*>(.*?)</[^>]+>'
    match = re.search(pattern, html, flags=re.S)
    if not match:
        return None
    return unescape(re.sub(r"<[^>]+>", " ", match.group(1))).strip()


def assert_dom_order(html: str, element_ids: list[str]) -> None:
    positions = [html.index(f'id="{element_id}"') for element_id in element_ids]
    assert positions == sorted(positions)


REFINEMENT_BASE_PROFILE = {
    "Income": 72,
    "Family": 1,
    "CCAvg": 4.8,
    "Education": 2,
    "Mortgage": 200,
    "SecuritiesAccount": 1,
    "CDAccount": 1,
    "Online": 0,
    "CreditCard": 0,
}
REFINEMENT_BASE_USER_INPUT = (
    "Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200, "
    "CDAccount yes, Online no, SecuritiesAccount yes, CreditCard no."
)


def build_complete_message_result(*, constraint_spec=None) -> StubResult:
    payload = {
        "task": "extract_cf_request",
        "status": "complete",
        "cf_request": dict(REFINEMENT_BASE_PROFILE),
        "missing_fields": [],
        "conflicts": [],
        "notes": [],
    }
    if constraint_spec is not None:
        payload["constraint_spec"] = constraint_spec
    return StubResult(message_text=json.dumps(payload))


def build_refinement_result(
    *,
    status: str,
    delta: dict | None = None,
    ambiguities: list[str] | None = None,
    unsupported_feedback: list[str] | None = None,
) -> StubResult:
    return StubResult(
        message_text=json.dumps(
            {
                "task": "extract_constraint_feedback",
                "status": status,
                "constraint_feedback_delta": {} if delta is None else delta,
                "ambiguities": [] if ambiguities is None else ambiguities,
                "unsupported_feedback": [] if unsupported_feedback is None else unsupported_feedback,
                "notes": [],
            }
        )
    )


def build_runtime_backed_orchestrator(*, adapter, sample_benchmark, tmp_path):
    return BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=RuntimeOrchestrator(runtime_mode="stable_demo"),
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )


def build_stored_session(**overrides) -> StoredSession:
    payload = {
        "session_id": "session-test",
        "dataset_key": "bank",
        "created_at": "2026-04-03T10:00:00+00:00",
        "updated_at": "2026-04-03T10:05:00+00:00",
        "current_public_state": None,
        "pending_clarification_json": None,
        "latest_turn_id": None,
        "last_turn_index": 0,
        "model_alias": "stub-model",
        "runtime_mode": "stable_demo",
        "lifecycle_status": "active",
        "archived_at": None,
        "clarification_turns_used": 0,
        "is_case_complete": False,
        "case_completion_reason": None,
        "restart_required": False,
        "active_constraint_spec_json": {},
        "last_runtime_request_json": None,
        "refinement_revision_index": 0,
        "refinement_rounds_used": 0,
        "refinement_round_limit": 3,
        "pending_refinement_clarification_json": None,
        "latest_runtime_backed_turn_id": None,
        "canonical_session_state_json": {
            "profile_facts": {},
            "hard_constraints": {},
            "soft_preferences": {},
        },
        "canonical_state_source": "canonical_authoritative",
        "canonical_mirror_ok": True,
    }
    payload.update(overrides)
    return StoredSession(**payload)


def test_app_session_flow_persists_turn_and_artifacts(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "skipped_no_counterfactual",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "no_recourse_needed",
                "validated_changed_fields": [],
                "details": {"reason": "no_counterfactual_required"},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": [],
                "winning_path": None,
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_response = api_request(app, "POST", "/api/v1/sessions")
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    message_response = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
                "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
            )
        },
    )
    assert message_response.status_code == 200
    payload = message_response.json()
    assert payload["public_state"] == "RUNTIME_SUCCESS"
    assert payload["debug_summary"]["builder_status"] == "READY_FOR_RUNTIME"
    assert payload["debug_summary"]["invariant_validation_status"] == "skipped_no_counterfactual"
    assert payload["artifact_refs"]["files"]
    assert payload["ui_review"]["display_state"] == "runtime_success_view"
    assert payload["ui_review"]["profile_editable"] is False
    assert payload["ui_review"]["read_only"] is True
    assert payload["ui_review"]["refinement_editable"] is True
    assert payload["ui_review"]["missing_fields"] == []
    income_field = next(item for item in payload["ui_review"]["profile_fields"] if item["field_name"] == "Income")
    assert income_field["value"] == 140
    assert payload["render_hints"]["primary_chat_text"] == (
        "Great news! Based on your current profile, your bank loan application would already be approved. "
        "No changes to your profile are needed."
    )
    assert payload["render_hints"]["primary_action_type"] == "no_action_required"
    assert payload["render_hints"]["primary_action_items"] == []
    assert payload["render_hints"]["supporting_detail_title"] == "Result details"
    assert payload["render_hints"]["state_marker_label"] == "Recommendation found"
    assert payload["render_hints"]["right_rail_anchor"] == "result-card"

    messages_response = api_request(app, "GET", f"/api/v1/sessions/{session_id}/messages")
    assert messages_response.status_code == 200
    assert len(messages_response.json()) == 1

    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    assert session_detail.status_code == 200
    assert session_detail.json()["ui_review"]["display_state"] == "runtime_success_view"
    assert session_detail.json()["ui_review"]["last_updated_turn_id"] == payload["turn_id"]
    assert session_detail.json()["render_hints"]["primary_action_type"] == "no_action_required"
    assert session_detail.json()["render_hints"]["right_rail_anchor"] == "result-card"
    assert session_detail.json()["render_hints"]["composer_mode"] == "refinement"
    assert session_detail.json()["render_hints"]["composer_context"]["submit_target"] == "refinements"
    assert session_detail.json()["render_hints"]["composer_context"]["advanced_controls_relevant"] is False

    artifacts_response = api_request(app, "GET", f"/api/v1/sessions/{session_id}/artifacts")
    assert artifacts_response.status_code == 200
    bundles = artifacts_response.json()
    assert len(bundles) == 1
    assert "artifact_manifest.json" in bundles[0]["files"]

    download_response = api_request(
        app,
        "GET",
        f"/api/v1/sessions/{session_id}/artifacts/{payload['turn_id']}/artifact_manifest.json"
    )
    assert download_response.status_code == 200

    html_response = api_request(app, "GET", f"/sessions/{session_id}")
    assert html_response.status_code == 200
    assert session_id in html_response.text
    assert "Close Session" in html_response.text
    assert "Result" in html_response.text
    assert "Refine this case" in html_response.text
    assert "Technical Details" in html_response.text
    assert "Dataset bank" in html_response.text
    assert 'id="chat-pane"' in html_response.text
    assert 'id="context-pane"' in html_response.text
    assert 'id="chat-header-strip"' in html_response.text
    assert 'id="chat-transcript"' in html_response.text
    assert 'id="current-state-badge"' in html_response.text
    assert 'id="session-meta-line"' in html_response.text
    assert 'id="result-card"' in html_response.text
    assert 'id="advanced-refinement-controls"' not in html_response.text
    assert_dom_order(
        html_response.text,
        ["review-card", "result-card", "technical-drawer"],
    )
    assert '/static/css/core/base.css' in html_response.text
    assert '/static/css/pages/session.css' in html_response.text
    assert '/static/js/pages/session.js' in html_response.text
    page_markup = html_response.text.split("<script", 1)[0]
    assert 'id="chat-composer-bar"' in page_markup
    assert 'data-submit-target="refinements"' in page_markup
    assert re.search(r'<textarea id="composer-input"[^>]*disabled', page_markup) is None
    assert "Continuing this case" in html_response.text
    assert "Apply Refinement" in html_response.text
    assert (
        "Great news! Based on your current profile, your bank loan application would already be approved. "
        "No changes to your profile are needed."
    ) in html_response.text
    assert element_text(html_response.text, "result-summary-copy") == (
        "No further changes are required unless you want to refine the same case."
    )
    assert 'id="refinement-card"' not in html_response.text
    assert re.search(r'<details id="technical-drawer"[^>]*\sopen', html_response.text) is None
    assert "window.location.reload()" not in html_response.text


def test_app_runtime_success_counterfactual_shows_explicit_visible_solution(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":40,"Family":1,"CCAvg":2.5,"Education":2,"Mortgage":0,'
                '"SecuritiesAccount":0,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.21},
            "counterfactual": {
                "candidates": [
                    {
                        "rank": 1,
                        "method": "stub_method",
                        "profile": {
                            "Income": 72,
                            "Family": 1,
                            "CCAvg": 2.5,
                            "Education": 2,
                            "Mortgage": 0,
                            "SecuritiesAccount": 0,
                            "CDAccount": 1,
                            "Online": 0,
                            "CreditCard": 0,
                        },
                        "changed_features": ["Income", "CDAccount"],
                    }
                ]
            },
            "reason_codes": [],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "passed",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "counterfactual_found",
                "validated_changed_fields": ["Income", "CDAccount"],
                "details": {},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": ["stub_method"],
                "winning_path": "stub_method",
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    message_response = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 40, Family 1, CCAvg 2.5, Education 2, Mortgage 0, "
                "SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0."
            )
        },
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert message_response.status_code == 200
    payload = message_response.json()
    assert payload["public_state"] == "RUNTIME_SUCCESS"
    assert payload["render_hints"]["primary_chat_text"].startswith(
        "Your current profile would be rejected for the bank loan. However, I found a way to get approved — here are the recommended changes: "
    )
    assert "Income: 40 -> 72" in payload["render_hints"]["primary_chat_text"]
    assert "CDAccount: No -> Yes" in payload["render_hints"]["primary_chat_text"]
    assert payload["render_hints"]["primary_action_type"] == "no_action_required"
    assert payload["render_hints"]["supporting_detail_title"] == "Result details"
    assert "2 fields to change" in payload["render_hints"]["supporting_detail_facts"]

    assert session_detail.status_code == 200
    assert "Income: 40 -> 72" in session_detail.json()["render_hints"]["primary_chat_text"]
    assert "CDAccount: No -> Yes" in session_detail.json()["render_hints"]["primary_chat_text"]
    assert session_detail.json()["render_hints"]["composer_mode"] == "refinement"

    assert html_response.status_code == 200
    assert "Income: 40 -> 72" in element_text(html_response.text, "result-headline")
    assert "CDAccount: No -> Yes" in element_text(html_response.text, "result-headline")


def test_app_create_session_accepts_explicit_dataset_key(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_response = api_request(app, "POST", "/api/v1/sessions", json={"dataset_key": "grad"})
    assert session_response.status_code == 200
    payload = session_response.json()

    assert payload["dataset_key"] == "grad"

    detail_response = api_request(app, "GET", f"/api/v1/sessions/{payload['session_id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["dataset_key"] == "grad"


def test_app_create_session_rejects_unsupported_dataset(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    response = api_request(app, "POST", "/api/v1/sessions", json={"dataset_key": "movie"})
    assert response.status_code == 400
    assert "Unsupported session dataset" in response.json()["detail"]


def test_app_health_endpoint_reports_unhealthy_when_lm_studio_fails(sample_benchmark, tmp_path, monkeypatch):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)
    monkeypatch.setattr("llm.src.product.service.check_lm_studio", lambda api_base: {"ok": False, "detail": "down"})

    response = api_request(app, "GET", "/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "unhealthy"
    assert response.json()["checks"]["lm_studio"]["ok"] is False


def test_app_restores_pending_clarification_after_restart(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        )
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    app_one = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app_one, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app_one,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 40."},
    )
    assert first_turn.status_code == 200
    assert first_turn.json()["public_state"] == "NEEDS_CLARIFICATION"

    repository_two = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app_two = create_app(config=config, orchestrator=orchestrator, repository=repository_two)

    session_detail = api_request(app_two, "GET", f"/api/v1/sessions/{session_id}")
    assert session_detail.status_code == 200
    assert session_detail.json()["has_pending_clarification"] is True


def test_app_messages_default_to_desc_order_and_accept_asc(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "skipped_no_counterfactual",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "no_recourse_needed",
                "validated_changed_fields": [],
                "details": {"reason": "no_counterfactual_required"},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": [],
                "winning_path": None,
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32."},
    )
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32 again."},
    )

    desc_response = api_request(app, "GET", f"/api/v1/sessions/{session_id}/messages")
    asc_response = api_request(app, "GET", f"/api/v1/sessions/{session_id}/messages?order=asc")

    assert desc_response.status_code == 200
    assert asc_response.status_code == 200
    assert [item["turn_index"] for item in desc_response.json()] == [2, 1]
    assert [item["turn_index"] for item in asc_response.json()] == [1, 2]


def test_app_archive_session_becomes_read_only_and_clears_pending(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 40."},
    )
    assert first_turn.status_code == 200
    assert first_turn.json()["public_state"] == "NEEDS_CLARIFICATION"

    archive_response = api_request(app, "POST", f"/api/v1/sessions/{session_id}/archive")
    assert archive_response.status_code == 200
    assert archive_response.json()["lifecycle_status"] == "archived"
    assert archive_response.json()["is_read_only"] is True

    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    assert session_detail.status_code == 200
    assert session_detail.json()["has_pending_clarification"] is False
    assert session_detail.json()["is_read_only"] is True

    blocked_response = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Online yes."},
    )
    assert blocked_response.status_code == 409


def test_app_case_complete_followup_returns_structured_409_and_terminal_ui(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "skipped_no_counterfactual",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "no_recourse_needed",
                "validated_changed_fields": [],
                "details": {"reason": "no_counterfactual_required"},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": [],
                "winning_path": None,
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
                "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
            )
        },
    )
    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 120."},
    )
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert first_turn.status_code == 200
    assert first_turn.json()["is_case_complete"] is True
    assert first_turn.json()["case_completion_reason"] == "runtime_success"
    assert blocked.status_code == 409
    assert blocked.json() == {
        "error_code": "case_complete",
        "detail": "This case is complete. Start a new case before sending another message.",
        "current_public_state": "RUNTIME_SUCCESS",
        "case_completion_reason": "runtime_success",
        "restart_required": True,
    }
    assert html_response.status_code == 200
    assert "Recommendation found" in html_response.text
    assert "Start New Case" in html_response.text
    assert 'id="restart-helper-card"' not in html_response.text
    assert 'id="chat-composer-bar"' in html_response.text
    assert 'data-submit-target="refinements"' in html_response.text


def test_app_clarification_limit_marks_case_complete_and_blocks_followup(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5},'
                    '"missing_fields":["Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"Education":2},'
                    '"missing_fields":["Income","CCAvg","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Mortgage":80},'
                    '"missing_fields":["Income","CCAvg","Family","Education","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    turn1 = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 40 and CCAvg 1.5."},
    )
    turn2 = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Family 3 and Education 2."},
    )
    turn3 = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Mortgage 80."},
    )
    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Online yes."},
    )

    assert turn1.status_code == 200
    assert turn1.json()["clarification_turns_used"] == 1
    assert turn1.json()["is_case_complete"] is False
    assert turn2.status_code == 200
    assert turn2.json()["clarification_turns_used"] == 2
    assert turn2.json()["clarification_payload"]["remaining_rounds"] == 1
    assert turn3.status_code == 200
    assert turn3.json()["public_state"] == "NEEDS_CLARIFICATION"
    assert turn3.json()["clarification_turns_used"] == 3
    assert turn3.json()["is_case_complete"] is True
    assert turn3.json()["case_completion_reason"] == "clarification_limit_reached"
    assert turn3.json()["restart_required"] is True
    assert turn3.json()["clarification_payload"]["clarification_type"] == "clarification_limit_reached"
    assert turn3.json()["clarification_payload"]["remaining_rounds"] == 0
    assert turn3.json()["clarification_payload"]["restart_required"] is True
    assert blocked.status_code == 409
    assert blocked.json()["case_completion_reason"] == "clarification_limit_reached"
    assert blocked.json()["restart_required"] is True


def test_app_clarification_turn_exposes_ui_review_and_clarification_cards(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5},'
                '"missing_fields":["Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    turn_response = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 40 and CCAvg 1.5."},
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert turn_response.status_code == 200
    payload = turn_response.json()
    assert payload["public_state"] == "NEEDS_CLARIFICATION"
    assert payload["ui_review"]["display_state"] == "needs_clarification_input"
    assert payload["ui_review"]["profile_editable"] is True
    assert payload["ui_review"]["read_only"] is False
    assert payload["ui_review"]["refinement_editable"] is False
    assert payload["ui_review"]["missing_fields"] == [
        "Family",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    fields = {item["field_name"]: item for item in payload["ui_review"]["profile_fields"]}
    assert fields["Income"]["value"] == 40
    assert fields["Income"]["missing"] is False
    assert fields["Family"]["missing"] is True
    assert fields["Online"]["display_value"] == "Not provided"
    assert payload["render_hints"]["primary_chat_text"] == (
        "Reply with only the missing fields: Family, Education, Mortgage, "
        "SecuritiesAccount, CDAccount, Online, and CreditCard. "
        "I'll keep the values already provided for Income and CCAvg."
    )
    assert payload["render_hints"]["primary_action_type"] == "provide_missing_fields"
    assert payload["render_hints"]["primary_action_items"] == [
        "Family",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    assert payload["render_hints"]["supporting_detail_title"] == "Why Runtime Is Blocked"
    assert payload["render_hints"]["state_marker_label"] == "Need more information"
    assert payload["render_hints"]["right_rail_anchor"] == "review-card"

    assert session_detail.status_code == 200
    assert session_detail.json()["ui_review"]["display_state"] == "needs_clarification_input"
    assert session_detail.json()["ui_review"]["last_updated_turn_id"] == payload["turn_id"]
    assert session_detail.json()["render_hints"]["primary_action_type"] == "provide_missing_fields"
    assert session_detail.json()["render_hints"]["composer_mode"] == "message"
    assert session_detail.json()["render_hints"]["composer_context"]["submit_target"] == "messages"

    assert html_response.status_code == 200
    assert "Review" in html_response.text
    assert "Next Action" in html_response.text
    assert "Use Review Edits" in html_response.text
    assert "Family" in html_response.text
    assert "CreditCard" in html_response.text
    assert (
        "Reply with only the missing fields: Family, Education, Mortgage, "
        "SecuritiesAccount, CDAccount, Online, and CreditCard. "
        "I'll keep the values already provided for Income and CCAvg."
    ) in unescape(html_response.text)
    assert 'id="result-card"' in html_response.text
    assert 'id="explanation-card"' not in html_response.text
    assert 'id="refinement-card"' not in html_response.text
    assert 'id="advanced-refinement-controls"' not in html_response.text
    assert "Need more information" in html_response.text
    assert 'id="chat-composer-bar"' in html_response.text
    assert 'data-submit-target="messages"' in html_response.text
    assert "Continuing this case" not in html_response.text
    assert_dom_order(html_response.text, ["review-card", "result-card", "technical-drawer"])


def test_app_runtime_reject_exposes_ui_review_and_result_cards(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":49,"Family":4,"CCAvg":1.6,"Education":1,"Mortgage":0,'
                '"SecuritiesAccount":1,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    turn_response = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 49, Family 4, CCAvg 1.6, Education 1, Mortgage 0, "
                "SecuritiesAccount 1, CDAccount 0, Online 0, CreditCard 0."
            )
        },
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert turn_response.status_code == 200
    payload = turn_response.json()
    assert payload["public_state"] == "RUNTIME_REJECT"
    assert payload["ui_review"]["display_state"] == "runtime_reject_view"
    assert payload["ui_review"]["read_only"] is True
    assert payload["ui_review"]["refinement_editable"] is True
    assert payload["explanation_payload"]["summary_type"] == "runtime_reject"
    assert payload["render_hints"]["primary_chat_text"] == (
        "Your current profile would be rejected for the bank loan. "
        "Unfortunately, the system could not find any feasible changes to get you approved. "
        "Try submitting a different profile or relaxing your constraints."
    )
    assert payload["render_hints"]["primary_action_type"] == "relax_constraints_or_restart"
    assert payload["render_hints"]["primary_action_items"] == ["Relax constraints", "Start a new case"]
    assert payload["render_hints"]["supporting_detail_title"] == "Why no recommendation"
    assert payload["render_hints"]["state_marker_label"] == "No recommendation available"
    assert payload["render_hints"]["right_rail_anchor"] == "result-card"

    assert session_detail.status_code == 200
    assert session_detail.json()["ui_review"]["display_state"] == "runtime_reject_view"
    assert session_detail.json()["refinement_allowed"] is True
    assert session_detail.json()["render_hints"]["primary_action_type"] == "relax_constraints_or_restart"
    assert session_detail.json()["render_hints"]["composer_mode"] == "refinement"
    assert session_detail.json()["render_hints"]["composer_context"]["submit_target"] == "refinements"
    assert session_detail.json()["render_hints"]["composer_context"]["advanced_controls_relevant"] is False

    assert html_response.status_code == 200
    assert "Next Action" in html_response.text
    assert "No recommendation available" in html_response.text
    assert (
        "Your current profile would be rejected for the bank loan. "
        "Unfortunately, the system could not find any feasible changes to get you approved. "
        "Try submitting a different profile or relaxing your constraints."
    ) in html_response.text
    result_copy = element_text(html_response.text, "result-summary-copy")
    assert result_copy == "Relax constraints or start a new case."
    assert 'id="advanced-refinement-controls"' not in html_response.text
    assert_dom_order(html_response.text, ["review-card", "result-card", "technical-drawer"])
    assert 'id="chat-composer-bar"' in html_response.text
    assert 'data-submit-target="refinements"' in html_response.text
    assert "Continuing this case" in html_response.text
    assert 'id="refinement-card"' not in html_response.text


def test_runtime_complete_session_page_keeps_review_read_only_and_refinement_enabled(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":100,"Family":1,"CCAvg":2.7,"Education":2,"Mortgage":0,'
                '"SecuritiesAccount":0,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    turn_response = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, "
                "SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0."
            )
        },
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert turn_response.status_code == 200
    assert turn_response.json()["public_state"] == "RUNTIME_SUCCESS"
    assert session_detail.status_code == 200
    assert session_detail.json()["refinement_allowed"] is True
    assert session_detail.json()["ui_review"]["read_only"] is True
    assert session_detail.json()["ui_review"]["refinement_editable"] is True
    assert html_response.status_code == 200
    page_markup = html_response.text.split("<script", 1)[0]
    assert 'id="state-rail"' in html_response.text
    assert 'id="technical-drawer"' in html_response.text
    assert html_response.text.index('id="state-rail"') < html_response.text.index('id="technical-drawer"')
    assert "Result" in html_response.text
    assert 'id="advanced-refinement-controls"' not in html_response.text
    assert_dom_order(html_response.text, ["review-card", "result-card", "technical-drawer"])
    assert 'id="chat-composer-bar"' in page_markup
    assert 'data-submit-target="refinements"' in page_markup
    assert re.search(r'<textarea id="composer-input"[^>]*disabled', page_markup) is None
    assert re.search(r'id="apply-profile-edits"', page_markup) is None
    assert "Continuing this case" in html_response.text
    assert html_response.text.count('id="chat-composer-bar"') == 1
    assert len(re.findall(r'<details id="(?:review-card|result-card|technical-drawer)"[^>]*\sopen', html_response.text)) <= 1


def test_fresh_session_page_renders_compact_empty_review_and_hides_refinement_form(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert html_response.status_code == 200
    assert 'id="chat-header-strip"' in html_response.text
    assert 'id="chat-transcript"' in html_response.text
    assert 'id="chat-pane"' in html_response.text
    assert 'id="context-pane"' in html_response.text
    assert 'data-session-tab="chat"' in html_response.text
    assert 'data-session-tab="context"' in html_response.text
    assert 'id="state-rail"' in html_response.text
    assert 'id="technical-drawer"' in html_response.text
    assert 'id="chat-composer-bar"' in html_response.text
    assert html_response.text.count('id="chat-composer-bar"') == 1
    assert 'class="session-title"' in html_response.text
    assert 'class="composer-title"' in html_response.text
    assert 'data-page-state="fresh"' in html_response.text
    assert "Describe one bank profile in natural language to start a new case." in html_response.text
    assert 'id="chat-composer-bar"' in html_response.text
    assert 'data-submit-target="messages"' in html_response.text
    assert 'id="result-card"' not in html_response.text
    assert 'id="explanation-card"' not in html_response.text
    assert 'id="advanced-refinement-controls"' not in html_response.text
    assert "Continuing this case" not in html_response.text


def test_app_refinement_archived_session_returns_structured_409(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(parse_result=build_complete_message_result())
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    )
    archive_response = api_request(app, "POST", f"/api/v1/sessions/{session_id}/archive")
    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change."},
    )

    assert archive_response.status_code == 200
    assert blocked.status_code == 409
    assert blocked.json() == {
        "error_code": "session_archived",
        "detail": f"Session {session_id} is archived and read-only.",
        "current_public_state": "RUNTIME_SUCCESS",
        "case_completion_reason": "runtime_success",
        "active_constraint_spec": {},
        "refinement_revision_index": 0,
        "refinement_rounds_used": 0,
        "refinement_round_limit": 3,
        "restart_required": True,
        "refinement_status": None,
    }


@pytest.mark.parametrize(
    ("parse_result", "user_input", "expected_public_state", "expected_completion_reason"),
    [
        (
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"conflict","cf_request":{"Income":40},'
                    '"missing_fields":[],"conflicts":["Income cannot be both 40 and 60."],"notes":[]}'
                )
            ),
            "Income 40 and Income 60.",
            "CONFLICT",
            "conflict",
        ),
        (
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"needs_clarification","cf_request":{},'
                    '"missing_fields":["Income","Family","CCAvg","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            "Give me general financial advice about how to optimize my finances.",
            "UNSUPPORTED_REQUEST",
            "unsupported_request",
        ),
    ],
)
def test_app_refinement_not_allowed_returns_structured_409_for_non_runtime_terminal_outcomes(
    sample_benchmark,
    tmp_path,
    parse_result,
    user_input,
    expected_public_state,
    expected_completion_reason,
):
    adapter = StubParserAdapter(parse_result=parse_result)
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": user_input},
    )
    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change."},
    )

    assert first_turn.status_code == 200
    assert first_turn.json()["public_state"] == expected_public_state
    assert first_turn.json()["case_completion_reason"] == expected_completion_reason
    assert blocked.status_code == 409
    assert blocked.json() == {
        "error_code": "refinement_not_allowed",
        "detail": "Refinement is only available after a runtime-backed result or during a pending refinement clarification.",
        "current_public_state": expected_public_state,
        "case_completion_reason": expected_completion_reason,
        "active_constraint_spec": {},
        "refinement_revision_index": 0,
        "refinement_rounds_used": 0,
        "refinement_round_limit": 3,
        "restart_required": True,
        "refinement_status": None,
    }


def test_app_refinement_not_allowed_returns_structured_409_after_parser_failure(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(message_text='{"task": "extract_cf_request", }'),
        repair_result=StubResult(message_text='{"task": "extract_cf_request", }'),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "parser failure"},
    )
    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change."},
    )

    assert first_turn.status_code == 200
    assert first_turn.json()["public_state"] == "PARSER_FAILURE"
    assert first_turn.json()["case_completion_reason"] == "parser_failure"
    assert blocked.status_code == 409
    assert blocked.json() == {
        "error_code": "refinement_not_allowed",
        "detail": "Refinement is only available after a runtime-backed result or during a pending refinement clarification.",
        "current_public_state": "PARSER_FAILURE",
        "case_completion_reason": "parser_failure",
        "active_constraint_spec": {},
        "refinement_revision_index": 0,
        "refinement_rounds_used": 0,
        "refinement_round_limit": 3,
        "restart_required": True,
        "refinement_status": None,
    }


def test_app_refinement_not_allowed_returns_structured_409_after_clarification_limit(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5},'
                    '"missing_fields":["Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"Education":2},'
                    '"missing_fields":["Income","CCAvg","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Mortgage":80},'
                    '"missing_fields":["Income","CCAvg","Family","Education","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 40 and CCAvg 1.5."},
    )
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Family 3 and Education 2."},
    )
    final_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Mortgage 80."},
    )
    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change."},
    )

    assert final_turn.status_code == 200
    assert final_turn.json()["public_state"] == "NEEDS_CLARIFICATION"
    assert final_turn.json()["case_completion_reason"] == "clarification_limit_reached"
    assert blocked.status_code == 409
    assert blocked.json() == {
        "error_code": "refinement_not_allowed",
        "detail": "Refinement is only available after a runtime-backed result or during a pending refinement clarification.",
        "current_public_state": "NEEDS_CLARIFICATION",
        "case_completion_reason": "clarification_limit_reached",
        "active_constraint_spec": {},
        "refinement_revision_index": 0,
        "refinement_rounds_used": 0,
        "refinement_round_limit": 3,
        "restart_required": True,
        "refinement_status": None,
    }


def test_app_refinement_apply_updates_active_constraints_and_parent_refs(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=build_complete_message_result(),
        refinement_parse_result=build_refinement_result(
            status="apply",
            delta={"set_max_changed_features": 1},
        ),
    )
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    initial_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    )
    refinement_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change."},
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")

    assert initial_turn.status_code == 200
    assert initial_turn.json()["public_state"] == "RUNTIME_SUCCESS"
    assert refinement_turn.status_code == 200
    payload = refinement_turn.json()
    assert payload["turn_kind"] == "refinement"
    assert payload["public_state"] == "RUNTIME_SUCCESS"
    assert payload["refinement_status"] == "applied"
    assert payload["refinement_revision_index"] == 1
    assert payload["parent_terminal_turn_id"] == initial_turn.json()["turn_id"]
    assert payload["parent_refinement_revision_index"] is None
    assert payload["active_constraint_spec"] == {"max_changed_features": 1}
    assert payload["refinement_rounds_used"] == 1
    assert payload["debug_summary"]["runtime_summary"]["executed"] is True
    assert "refinement_result.json" in payload["artifact_refs"]["files"]
    assert session_detail.status_code == 200
    assert session_detail.json()["active_constraint_spec"] == {"max_changed_features": 1}
    assert session_detail.json()["refinement_revision_index"] == 1
    assert session_detail.json()["refinement_rounds_used"] == 1
    assert session_detail.json()["latest_runtime_backed_turn_id"] == payload["turn_id"]
    assert session_detail.json()["refinement_allowed"] is True


def test_app_refinement_clarification_followup_allows_multi_delta_resolution(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[build_complete_message_result()],
        refinement_results=[
            build_refinement_result(
                status="clarification_required",
                ambiguities=["The feedback both blocks and unblocks Income in the same refinement turn."],
            ),
            build_refinement_result(
                status="apply",
                delta={
                    "set_max_changed_features": 1,
                    "set_prefer_fewer_changes": True,
                },
            ),
        ],
    )
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    ).json()
    first_refinement = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Do not change Income, actually Income can change."},
    )
    detail_after_clarification = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    second_refinement = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change and prefer smaller edits."},
    )
    detail_after_apply = api_request(app, "GET", f"/api/v1/sessions/{session_id}")

    assert first_refinement.status_code == 200
    clarification_payload = first_refinement.json()
    assert clarification_payload["turn_kind"] == "refinement"
    assert clarification_payload["public_state"] == first_turn["public_state"]
    assert clarification_payload["refinement_status"] == "clarification_required"
    assert clarification_payload["refinement_revision_index"] == 1
    assert clarification_payload["clarification_payload"]["clarification_type"] == "refinement_clarification"
    assert clarification_payload["clarification_payload"]["restart_required"] is False
    assert clarification_payload["active_constraint_spec"] == {}
    assert detail_after_clarification.status_code == 200
    assert detail_after_clarification.json()["has_pending_refinement_clarification"] is True
    assert detail_after_clarification.json()["refinement_rounds_used"] == 1
    assert detail_after_clarification.json()["latest_runtime_backed_turn_id"] == first_turn["turn_id"]

    assert second_refinement.status_code == 200
    applied_payload = second_refinement.json()
    assert applied_payload["turn_kind"] == "refinement"
    assert applied_payload["refinement_status"] == "applied"
    assert applied_payload["refinement_revision_index"] == 2
    assert applied_payload["parent_terminal_turn_id"] == first_turn["turn_id"]
    assert applied_payload["parent_refinement_revision_index"] == 1
    assert applied_payload["active_constraint_spec"] == {
        "max_changed_features": 1,
        "prefer_fewer_changes": True,
    }
    assert detail_after_apply.status_code == 200
    assert detail_after_apply.json()["has_pending_refinement_clarification"] is False
    assert detail_after_apply.json()["refinement_revision_index"] == 2
    assert detail_after_apply.json()["latest_runtime_backed_turn_id"] == applied_payload["turn_id"]


def test_app_refinement_vague_apply_is_reclassified_to_clarification_and_artifacts_match(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=build_complete_message_result(),
        refinement_parse_result=build_refinement_result(
            status="apply",
            delta={"set_prefer_fewer_changes": True},
        ),
    )
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    initial_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    ).json()
    refinement_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Make the bank result better without changing too much."},
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")

    assert refinement_turn.status_code == 200
    payload = refinement_turn.json()
    assert payload["turn_kind"] == "refinement"
    assert payload["public_state"] == initial_turn["public_state"]
    assert payload["refinement_status"] == "clarification_required"
    assert payload["parent_terminal_turn_id"] == initial_turn["turn_id"]
    assert payload["active_constraint_spec"] == {}
    assert payload["ui_review"]["display_state"] == "refinement_clarification_view"
    assert payload["ui_review"]["profile_editable"] is False
    assert payload["ui_review"]["read_only"] is True
    assert payload["ui_review"]["refinement_editable"] is True
    assert payload["clarification_payload"]["clarification_type"] == "refinement_clarification"
    assert payload["explanation_payload"] is None
    assert payload["debug_summary"]["runtime_summary"]["executed"] is False
    assert "Clarification is required" in payload["assistant_text"]
    assert payload["render_hints"]["primary_chat_text"].startswith("I need a more specific refinement request.")
    assert payload["render_hints"]["primary_action_type"] == "clarify_refinement"
    assert payload["render_hints"]["right_rail_anchor"] == "advanced-refinement-controls"

    assert session_detail.status_code == 200
    assert session_detail.json()["active_constraint_spec"] == {}
    assert session_detail.json()["has_pending_refinement_clarification"] is True
    assert session_detail.json()["latest_runtime_backed_turn_id"] == initial_turn["turn_id"]
    assert session_detail.json()["ui_review"]["display_state"] == "refinement_clarification_view"
    assert session_detail.json()["ui_review"]["refinement_editable"] is True
    assert session_detail.json()["render_hints"]["composer_mode"] == "refinement"
    assert session_detail.json()["render_hints"]["composer_context"]["advanced_controls_relevant"] is True

    artifact_dir = Path(payload["artifact_refs"]["artifact_dir"])
    turn_result = json.loads((artifact_dir / "turn_result.json").read_text(encoding="utf-8"))
    clarification_payload = json.loads((artifact_dir / "clarification_payload.json").read_text(encoding="utf-8"))
    explanation_payload = json.loads((artifact_dir / "explanation_payload.json").read_text(encoding="utf-8"))
    refinement_result = json.loads((artifact_dir / "refinement_result.json").read_text(encoding="utf-8"))
    builder_result = json.loads((artifact_dir / "builder_result.json").read_text(encoding="utf-8"))

    assert turn_result["refinement_status"] == "clarification_required"
    assert turn_result["clarification_payload"] == payload["clarification_payload"]
    assert clarification_payload == payload["clarification_payload"]
    assert explanation_payload is None
    assert turn_result["explanation_payload"] is None
    assert refinement_result["refinement_status"] == "clarification_required"
    assert builder_result is None


def test_session_page_refinement_clarification_shows_prior_result_then_followup_then_refinement_input(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=build_complete_message_result(),
        refinement_parse_result=build_refinement_result(
            status="apply",
            delta={"set_prefer_fewer_changes": True},
        ),
    )
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    )
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Make the bank result better without changing too much."},
    )
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert html_response.status_code == 200
    assert 'data-page-state="refinement_clarification"' in html_response.text
    assert "Refinement needs clarification" in html_response.text
    assert 'id="explanation-card"' not in html_response.text
    assert_dom_order(
        html_response.text,
        ["review-card", "result-card", "advanced-refinement-controls", "technical-drawer"],
    )
    assert 'id="chat-composer-bar"' in html_response.text
    assert 'data-submit-target="refinements"' in html_response.text
    assert "Continuing this case" in html_response.text
    assert 'id="refinement-card"' not in html_response.text
    assert re.search(r'<details id="advanced-refinement-controls"[^>]*\sopen', html_response.text) is not None


def test_session_page_restart_required_shows_restart_helper_and_hides_same_case_continuation(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5},'
                    '"missing_fields":["Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"Education":2},'
                    '"missing_fields":["Income","CCAvg","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Mortgage":80},'
                    '"missing_fields":["Income","CCAvg","Family","Education","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 40 and CCAvg 1.5."},
    )
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Family 3 and Education 2."},
    )
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Mortgage 80."},
    )
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert html_response.status_code == 200
    assert 'data-page-state="restart_required"' in html_response.text
    assert 'id="refinement-card"' not in html_response.text
    assert 'id="advanced-refinement-controls"' not in html_response.text
    assert re.search(r'<section[^>]*id="chat-composer-bar"[^>]*hidden', html_response.text, flags=re.S) is not None
    assert "Continuing this case" not in html_response.text
    assert "Start New Case" in html_response.text
    assert_dom_order(html_response.text, ["review-card", "result-card", "technical-drawer"])


def test_session_page_transcript_keeps_turns_in_linear_chronological_order(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "skipped_no_counterfactual",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "no_recourse_needed",
                "validated_changed_fields": [],
                "details": {"reason": "no_counterfactual_required"},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": [],
                "winning_path": None,
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    first_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32."},
    ).json()
    second_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32 again."},
    ).json()
    html_response = api_request(app, "GET", f"/sessions/{session_id}")

    assert html_response.status_code == 200
    assert html_response.text.index(first_turn["turn_id"]) < html_response.text.index(second_turn["turn_id"])
    assert 'class="chat-transcript-items"' in html_response.text
    assert 'class="chat-message chat-message--user"' in html_response.text
    assert 'class="stream-turn"' not in html_response.text
    assert "Expand" not in html_response.text


def test_app_refinement_unsupported_feedback_keeps_active_constraints(sample_benchmark, tmp_path):
    initial_constraints = {"max_changed_features": 1}
    adapter = StubParserAdapter(
        parse_result=build_complete_message_result(constraint_spec=initial_constraints),
        refinement_parse_result=build_refinement_result(
            status="unsupported_feedback",
            unsupported_feedback=["Method-selection requests are outside the supported refinement language."],
        ),
    )
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    initial_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    )
    session_after_message = api_request(app, "GET", f"/api/v1/sessions/{session_id}")
    unsupported_turn = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Show me every UFCE method and rank them."},
    )
    session_after_refinement = api_request(app, "GET", f"/api/v1/sessions/{session_id}")

    assert initial_turn.status_code == 200
    assert session_after_message.json()["active_constraint_spec"] == initial_constraints
    assert unsupported_turn.status_code == 200
    payload = unsupported_turn.json()
    assert payload["refinement_status"] == "unsupported_feedback"
    assert payload["public_state"] == "RUNTIME_SUCCESS"
    assert payload["active_constraint_spec"] == initial_constraints
    assert "Active constraints were left unchanged." in payload["assistant_text"]
    assert session_after_refinement.status_code == 200
    assert session_after_refinement.json()["active_constraint_spec"] == initial_constraints
    assert session_after_refinement.json()["has_pending_refinement_clarification"] is False
    assert session_after_refinement.json()["refinement_rounds_used"] == 1


def test_app_refinement_limit_reached_returns_structured_409(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[build_complete_message_result()],
        refinement_results=[
            build_refinement_result(
                status="unsupported_feedback",
                unsupported_feedback=["Unsupported refinement feedback."],
            ),
            build_refinement_result(
                status="unsupported_feedback",
                unsupported_feedback=["Unsupported refinement feedback."],
            ),
            build_refinement_result(
                status="unsupported_feedback",
                unsupported_feedback=["Unsupported refinement feedback."],
            ),
        ],
    )
    orchestrator = build_runtime_backed_orchestrator(
        adapter=adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    )
    for _ in range(3):
        response = api_request(
            app,
            "POST",
            f"/api/v1/sessions/{session_id}/refinements",
            json={"user_feedback": "Unsupported refinement."},
        )
        assert response.status_code == 200

    blocked = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "One more refinement."},
    )
    session_detail = api_request(app, "GET", f"/api/v1/sessions/{session_id}")

    assert blocked.status_code == 409
    assert blocked.json() == {
        "error_code": "refinement_limit_reached",
        "detail": "The refinement round limit was reached. Start a new case to continue.",
        "current_public_state": "RUNTIME_SUCCESS",
        "case_completion_reason": "runtime_success",
        "active_constraint_spec": {},
        "refinement_revision_index": 3,
        "refinement_rounds_used": 3,
        "refinement_round_limit": 3,
        "restart_required": True,
        "refinement_status": "limit_reached",
    }
    assert session_detail.status_code == 200
    assert session_detail.json()["refinement_allowed"] is False
    assert session_detail.json()["refinement_rounds_used"] == 3
    assert session_detail.json()["refinement_revision_index"] == 3


def test_app_refinement_state_survives_restart(sample_benchmark, tmp_path):
    first_adapter = StubParserAdapter(
        parse_result=build_complete_message_result(),
        refinement_parse_result=build_refinement_result(
            status="apply",
            delta={"set_max_changed_features": 1},
        ),
    )
    config = build_config(tmp_path)
    repository_one = SessionRepository(config.sqlite_path, app_version=config.app_version)
    orchestrator_one = build_runtime_backed_orchestrator(
        adapter=first_adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    app_one = create_app(config=config, orchestrator=orchestrator_one, repository=repository_one)

    session_id = api_request(app_one, "POST", "/api/v1/sessions").json()["session_id"]
    api_request(
        app_one,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": REFINEMENT_BASE_USER_INPUT},
    )
    first_refinement = api_request(
        app_one,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Allow at most one feature change."},
    ).json()

    repository_two = SessionRepository(config.sqlite_path, app_version=config.app_version)
    second_adapter = StubParserAdapter(
        parse_result=build_complete_message_result(),
        refinement_parse_result=build_refinement_result(
            status="apply",
            delta={"set_prefer_fewer_changes": True},
        ),
    )
    orchestrator_two = build_runtime_backed_orchestrator(
        adapter=second_adapter,
        sample_benchmark=sample_benchmark,
        tmp_path=tmp_path,
    )
    app_two = create_app(config=config, orchestrator=orchestrator_two, repository=repository_two)

    session_detail = api_request(app_two, "GET", f"/api/v1/sessions/{session_id}")
    second_refinement = api_request(
        app_two,
        "POST",
        f"/api/v1/sessions/{session_id}/refinements",
        json={"user_feedback": "Prefer smaller edits."},
    )

    assert session_detail.status_code == 200
    assert session_detail.json()["active_constraint_spec"] == {"max_changed_features": 1}
    assert session_detail.json()["refinement_revision_index"] == 1
    assert session_detail.json()["latest_runtime_backed_turn_id"] == first_refinement["turn_id"]
    assert second_refinement.status_code == 200
    assert second_refinement.json()["refinement_revision_index"] == 2
    assert second_refinement.json()["parent_refinement_revision_index"] == 1
    assert second_refinement.json()["active_constraint_spec"] == {
        "max_changed_features": 1,
        "prefer_fewer_changes": True,
    }


def test_app_preview_endpoint_returns_manifest_content(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "skipped_no_counterfactual",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "no_recourse_needed",
                "validated_changed_fields": [],
                "details": {"reason": "no_counterfactual_required"},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": [],
                "winning_path": None,
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    turn_payload = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={"user_input": "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32."},
    ).json()

    preview = api_request(
        app,
        "GET",
        f"/api/v1/sessions/{session_id}/artifacts/{turn_payload['turn_id']}/artifact_manifest.json/preview",
    )

    assert preview.status_code == 200
    assert preview.json()["filename"] == "artifact_manifest.json"
    assert preview.json()["content_type"] == "application/json"
    assert '"turn_id"' in preview.json()["content"]


def test_app_catalog_endpoint_lists_bank_active_and_other_datasets_blocked(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    response = api_request(app, "GET", "/api/v1/catalog/datasets")

    assert response.status_code == 200
    payload = response.json()
    assert [item["dataset_key"] for item in payload[:5]] == ["bank", "grad", "bupa", "movie", "wine"]
    assert any(item["dataset_key"] == "bank" and item["availability_status"] == "active" for item in payload)
    assert any(item["dataset_key"] == "movie" and item["availability_status"] == "blocked" for item in payload)
    assert any(item["dataset_key"] == "grad" and item["feature_guides"] for item in payload)


def test_home_page_renders_dataset_catalog(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_response = api_request(app, "POST", "/api/v1/sessions")
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    response = api_request(app, "GET", "/")

    assert response.status_code == 200
    assert "Start New Bank Session" in response.text
    assert session_id in response.text
    assert "Resume Session" in response.text
    assert response.text.index("Persisted Sessions") < response.text.rindex("Dataset Bundles")
    assert "Dataset Bundles" in response.text
    assert "Bank Personal Loan" in response.text
    assert "Module 5 UI Fix" not in response.text
    assert "Bank-first live demo" not in response.text
    assert "Natural language first" not in response.text
    assert "Blocked In This MVP" in response.text
    assert "View Details" in response.text
    assert 'id="dataset-detail-panel"' in response.text
    assert 'id="recent-sessions-panel"' not in response.text
    assert "Selection Guidance" not in response.text
    assert "Key changeable features" in response.text
    assert "Locked / non-changeable features" in response.text
    assert "View Full Feature Guide" in response.text
    assert "Training logic" in response.text
    assert 'class="meta-value meta-value-wrap"' in response.text
    assert '/static/css/core/base.css' in response.text
    assert '/static/css/pages/home.css' in response.text
    assert '/static/js/pages/home.js' in response.text


def test_home_page_falls_back_to_first_sorted_dataset_when_no_dataset_is_active(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    catalog_entries = app.state.service.list_dataset_catalog()
    app.state.service.list_dataset_catalog = lambda: [
        {
            **item,
            "availability_status": "blocked",
            "support_note": "Dataset bundle is available locally for reference, but live conversational runtime is blocked in this MVP.",
        }
        for item in catalog_entries
    ]

    response = api_request(app, "GET", "/")

    assert response.status_code == 200
    assert 'id="dataset-detail-panel"' in response.text
    assert '<h3 id="dataset-detail-title">Bank Personal Loan</h3>' in response.text
    assert "Reference Only In This MVP" in response.text


def test_static_assets_are_served(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text='{"task":"extract_cf_request","status":"partial","cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    assert app.url_path_for("static", path="css/pages/home.css") == "/static/css/pages/home.css"
    assert app.url_path_for("static", path="js/pages/session.js") == "/static/js/pages/session.js"
    assert Path("llm/src/product/static/css/pages/home.css").is_file()
    assert Path("llm/src/product/static/js/pages/session.js").is_file()


def test_serialize_session_detail_ui_review_prefers_latest_visible_turn_state(tmp_path):
    session = StoredSession(
        session_id="session-precedence",
        dataset_key="bank",
        created_at="2026-04-03T10:00:00+00:00",
        updated_at="2026-04-03T10:05:00+00:00",
        current_public_state="NEEDS_CLARIFICATION",
        pending_clarification_json={
            "missing_fields": ["Income"],
            "prior_cf_request": {"Income": 40},
            "prior_constraint_spec": {"max_changed_features": 2},
        },
        latest_turn_id="session-turn",
        last_turn_index=1,
        model_alias="stub-model",
        runtime_mode="stable_demo",
        lifecycle_status="active",
        archived_at=None,
        clarification_turns_used=1,
        is_case_complete=False,
        case_completion_reason=None,
        restart_required=False,
        active_constraint_spec_json={"max_changed_features": 2},
        last_runtime_request_json=None,
        refinement_revision_index=0,
        refinement_rounds_used=0,
        refinement_round_limit=3,
        pending_refinement_clarification_json=None,
        latest_runtime_backed_turn_id=None,
        canonical_session_state_json={
            "profile_facts": {"Income": 40},
            "hard_constraints": {"max_changed_features": 2},
            "soft_preferences": {},
        },
        canonical_state_source="canonical_authoritative",
        canonical_mirror_ok=True,
    )
    latest_turn_payload = {
        "turn_id": "turn-latest",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "message",
        "clarification_payload": None,
        "refinement_status": None,
        "canonical_session_state": {
            "profile_facts": {"Income": 72},
            "hard_constraints": {},
            "soft_preferences": {"prefer_fewer_changes": True},
        },
        "active_constraint_spec": {"prefer_fewer_changes": True},
    }

    detail = serialize_session_detail(
        session,
        turn_count=2,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )

    assert detail.ui_review is not None
    assert detail.ui_review.display_state == "runtime_success_view"
    assert detail.ui_review.last_updated_turn_id == "turn-latest"
    income_field = next(item for item in detail.ui_review.profile_fields if item.field_name == "Income")
    assert income_field.value == 72
    assert detail.ui_review.constraints == []
    assert [item.key for item in detail.ui_review.preferences] == ["prefer_fewer_changes"]
    assert detail.ui_review.missing_fields == []


def test_session_page_context_runtime_success_outweighs_restart_required_when_refinement_is_allowed(tmp_path):
    session = build_stored_session(
        session_id="session-runtime-overlap",
        current_public_state="RUNTIME_SUCCESS",
        latest_turn_id="turn-runtime",
        last_turn_index=1,
        is_case_complete=True,
        case_completion_reason="runtime_success",
        restart_required=True,
        last_runtime_request_json={"Income": 72},
        latest_runtime_backed_turn_id="turn-runtime",
        canonical_session_state_json={
            "profile_facts": {"Income": 72},
            "hard_constraints": {},
            "soft_preferences": {},
        },
    )
    latest_turn_payload = {
        "turn_id": "turn-runtime",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "message",
        "assistant_text": "The current bank profile already reaches the desired outcome.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "no_recourse_needed",
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "changed_fields": [],
            "counterfactual_summary": None,
        },
        "refinement_status": None,
    }

    detail = serialize_session_detail(
        session,
        turn_count=1,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )
    context = _build_session_page_render_context(
        detail,
        turns=[latest_turn_payload],
        latest_turn=latest_turn_payload,
    )

    assert detail.refinement_allowed is True
    assert detail.render_hints is not None
    assert detail.render_hints.primary_chat_text == (
        "Great news! Based on your current profile, your bank loan application would already be approved. "
        "No changes to your profile are needed."
    )
    assert detail.render_hints.primary_action_type == "no_action_required"
    assert detail.render_hints.right_rail_anchor == "result-card"
    assert detail.render_hints.composer_mode == "refinement"
    assert context["page_state"] == "runtime_success"
    assert context["chat_header_summary"]["state_label"] == "runtime success"
    assert context["composer_mode"] == "refinement"
    assert context["composer_context"]["submit_target"] == "refinements"
    assert [item["kind"] for item in context["transcript_items"]] == [
        "user_message",
        "system_marker",
        "assistant_message",
        "inline_detail_toggle",
    ]
    assert context["transcript_items"][2]["text"] == (
        "Great news! Based on your current profile, your bank loan application would already be approved. "
        "No changes to your profile are needed."
    )


def test_session_page_context_prioritizes_pending_refinement_clarification_over_runtime_and_original_clarification(tmp_path):
    runtime_turn = {
        "turn_id": "turn-runtime",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "message",
        "assistant_text": "A feasible counterfactual was found.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "reason_codes": [],
            "changed_fields": ["Income"],
            "counterfactual_summary": {"profile_diff": {"Income": {"from": 40, "to": 72}}},
        },
        "refinement_status": None,
    }
    refinement_clarification_turn = {
        "turn_id": "turn-refinement-clarification",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "refinement",
        "assistant_text": "Clarification is required before the refinement can be applied.",
        "clarification_payload": {
            "clarification_type": "refinement_clarification",
            "missing_fields": [],
            "conflicts": ["Clarify whether Income may change."],
            "next_required_input": "State whether Income can change.",
            "remaining_rounds": 2,
            "restart_required": False,
            "reply_strategy": "start_new_case",
            "carried_forward_fields": [],
        },
        "explanation_payload": None,
        "refinement_status": "clarification_required",
    }
    session = build_stored_session(
        session_id="session-refinement-overlap",
        current_public_state="RUNTIME_SUCCESS",
        latest_turn_id="turn-refinement-clarification",
        last_turn_index=2,
        is_case_complete=True,
        case_completion_reason="runtime_success",
        restart_required=False,
        last_runtime_request_json={"Income": 72},
        latest_runtime_backed_turn_id="turn-runtime",
        pending_refinement_clarification_json={
            "ambiguities": ["Clarify whether Income may change."],
            "next_required_input": "State whether Income can change.",
        },
        canonical_session_state_json={
            "profile_facts": {"Income": 72},
            "hard_constraints": {},
            "soft_preferences": {},
        },
    )

    detail = serialize_session_detail(
        session,
        turn_count=2,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=refinement_clarification_turn,
    )
    context = _build_session_page_render_context(
        detail,
        turns=[refinement_clarification_turn, runtime_turn],
        latest_turn=refinement_clarification_turn,
    )
    expected_runtime_text = (
        "Your current profile would be rejected for the bank loan. "
        "However, I found a way to get approved — here are the recommended changes: Income: 40 -> 72."
    )

    assert context["page_state"] == "refinement_clarification"
    assert context["latest_runtime_summary"] is not None
    assert context["latest_runtime_summary"]["kind"] == "success"
    assert context["chat_header_summary"]["state_label"] == "refinement clarification"
    assert context["composer_mode"] == "refinement"
    assert detail.render_hints is not None
    assert detail.render_hints.primary_action_type == "clarify_refinement"
    assert detail.render_hints.right_rail_anchor == "advanced-refinement-controls"
    assert detail.render_hints.composer_context.advanced_controls_relevant is True
    assert context["show_advanced_controls_by_default"] is True
    assert [item["kind"] for item in context["transcript_items"]] == [
        "user_message",
        "system_marker",
        "assistant_message",
        "inline_detail_toggle",
        "user_message",
        "system_marker",
        "assistant_message",
        "inline_detail_toggle",
    ]
    assert context["transcript_items"][2]["text"] == expected_runtime_text
    assert context["transcript_items"][-2]["text"] == (
        "I need a more specific refinement request. Clarify whether Income may change. "
        "State whether Income can change."
    )


def test_session_page_context_turn_count_above_zero_never_resolves_fresh(tmp_path):
    latest_turn_payload = {
        "turn_id": "turn-conflict",
        "public_state": "CONFLICT",
        "turn_kind": "message",
        "assistant_text": "The request contains conflicting values.",
        "clarification_payload": {
            "clarification_type": "conflict_resolution",
            "missing_fields": [],
            "conflicts": ["Income cannot be both 40 and 60."],
            "next_required_input": "Submit one corrected bank profile.",
            "remaining_rounds": 2,
            "restart_required": True,
            "reply_strategy": "start_new_case",
            "carried_forward_fields": [],
        },
        "explanation_payload": None,
        "refinement_status": None,
    }
    session = build_stored_session(
        session_id="session-not-fresh",
        current_public_state="CONFLICT",
        latest_turn_id="turn-conflict",
        last_turn_index=1,
        is_case_complete=True,
        case_completion_reason="conflict",
        restart_required=True,
    )

    detail = serialize_session_detail(
        session,
        turn_count=1,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )
    context = _build_session_page_render_context(
        detail,
        turns=[latest_turn_payload],
        latest_turn=latest_turn_payload,
    )

    assert context["page_state"] != "fresh"
    assert context["page_state"] == "restart_required"
    assert context["chat_header_summary"]["state_label"] == "restart required"
    assert context["composer_mode"] == "disabled"
    assert context["composer_context"]["submit_target"] is None
    assert context["transcript_items"][-1]["id"] == "restart-terminal-note"


def test_session_page_context_prefers_latest_visible_runtime_turn_over_fallback_runtime_summary(tmp_path):
    latest_success_turn = {
        "turn_id": "turn-latest-success",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "message",
        "assistant_text": "The current bank profile already reaches the desired outcome.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "no_recourse_needed",
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "changed_fields": [],
            "counterfactual_summary": None,
        },
        "refinement_status": None,
    }
    older_reject_turn = {
        "turn_id": "turn-older-reject",
        "public_state": "RUNTIME_REJECT",
        "turn_kind": "message",
        "assistant_text": "Runtime completed without a feasible counterfactual.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "runtime_reject",
            "reason_codes": ["NO_FEASIBLE_CF_FOUND"],
            "changed_fields": [],
            "counterfactual_summary": None,
        },
        "refinement_status": None,
    }
    session = build_stored_session(
        session_id="session-visible-runtime-precedence",
        current_public_state="RUNTIME_SUCCESS",
        latest_turn_id="turn-latest-success",
        last_turn_index=2,
        is_case_complete=True,
        case_completion_reason="runtime_success",
        restart_required=False,
        last_runtime_request_json={"Income": 72},
        latest_runtime_backed_turn_id="turn-older-reject",
    )

    detail = serialize_session_detail(
        session,
        turn_count=2,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_success_turn,
    )
    context = _build_session_page_render_context(
        detail,
        turns=[latest_success_turn, older_reject_turn],
        latest_turn=latest_success_turn,
    )

    assert context["page_state"] == "runtime_success"
    assert context["latest_runtime_summary"] is not None
    assert context["latest_runtime_summary"]["kind"] == "success"
    assert context["chat_header_summary"]["state_label"] == "runtime success"
    assert context["transcript_items"][-2]["text"] == (
        "Great news! Based on your current profile, your bank loan application would already be approved. "
        "No changes to your profile are needed."
    )


def test_session_page_context_fresh_emits_welcome_transcript_and_header(tmp_path):
    session = build_stored_session(
        session_id="session-fresh",
        latest_turn_id=None,
        last_turn_index=0,
        current_public_state=None,
        is_case_complete=False,
        restart_required=False,
    )
    detail = serialize_session_detail(
        session,
        turn_count=0,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=None,
    )

    context = _build_session_page_render_context(
        detail,
        turns=[],
        latest_turn=None,
    )

    assert context["page_state"] == "fresh"
    assert context["chat_header_summary"]["state_label"] == "fresh"
    assert context["composer_mode"] == "message"
    assert context["composer_context"]["submit_target"] == "messages"
    assert detail.render_hints is not None
    assert detail.render_hints.primary_action_type == "start_case"
    assert detail.render_hints.composer_mode == "message"
    assert context["transcript_items"] == [
        {
            "kind": "assistant_message",
            "id": "welcome-assistant",
            "turn_id": None,
            "text": "Describe one bank profile in natural language to start a new case.",
            "pending": False,
        }
    ]


def test_session_page_context_clarification_emits_marker_and_detail_toggle(tmp_path):
    latest_turn_payload = {
        "turn_id": "turn-clarification",
        "public_state": "NEEDS_CLARIFICATION",
        "turn_kind": "message",
        "user_input": "Income 40.",
        "assistant_text": (
            "Reply with only the missing fields: Family, Education, and Mortgage. "
            "I'll keep the values already provided for Income."
        ),
        "clarification_payload": {
            "clarification_type": "missing_information",
            "missing_fields": ["Family", "Education", "Mortgage"],
            "conflicts": [],
            "next_required_input": "Reply with only the missing fields: Family, Education, and Mortgage. I'll keep the values already provided for Income.",
            "remaining_rounds": 2,
            "restart_required": False,
            "reply_strategy": "missing_fields_only",
            "carried_forward_fields": ["Income"],
        },
        "explanation_payload": None,
        "refinement_status": None,
    }
    session = build_stored_session(
        session_id="session-clarification",
        current_public_state="NEEDS_CLARIFICATION",
        latest_turn_id="turn-clarification",
        last_turn_index=1,
        pending_clarification_json={
            "missing_fields": ["Family", "Education", "Mortgage"],
            "prior_cf_request": {"Income": 40},
            "prior_constraint_spec": {},
        },
    )
    detail = serialize_session_detail(
        session,
        turn_count=1,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )

    context = _build_session_page_render_context(
        detail,
        turns=[latest_turn_payload],
        latest_turn=latest_turn_payload,
    )

    assert context["page_state"] == "clarification"
    assert context["chat_header_summary"]["state_label"] == "needs clarification"
    assert context["composer_mode"] == "message"
    assert detail.render_hints is not None
    assert detail.render_hints.primary_action_type == "provide_missing_fields"
    assert detail.render_hints.primary_action_items == ["Family", "Education", "Mortgage"]
    assert detail.render_hints.right_rail_anchor == "review-card"
    assert [item["kind"] for item in context["transcript_items"]] == [
        "user_message",
        "system_marker",
        "assistant_message",
        "inline_detail_toggle",
    ]
    assert context["transcript_items"][1]["label"] == "Need more information"
    assert context["transcript_items"][2]["text"] == (
        "Reply with only the missing fields: Family, Education, and Mortgage. "
        "I'll keep the values already provided for Income."
    )
    assert context["transcript_items"][3]["facts"] == ["Family", "Education", "Mortgage"]


def test_session_page_context_refinement_clarification_non_structured_keeps_advanced_controls_hidden(tmp_path):
    runtime_turn = {
        "turn_id": "turn-runtime",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "message",
        "assistant_text": "A feasible counterfactual was found.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "reason_codes": [],
            "changed_fields": ["Income"],
            "counterfactual_summary": {"profile_diff": {"Income": {"from": 40, "to": 72}}},
        },
        "refinement_status": None,
    }
    refinement_clarification_turn = {
        "turn_id": "turn-refinement-clarification",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "refinement",
        "assistant_text": "Clarification is required before the refinement can be applied.",
        "clarification_payload": {
            "clarification_type": "refinement_clarification",
            "missing_fields": [],
            "conflicts": ["Clarify what 'better' means for this case."],
            "next_required_input": "State which outcome should improve.",
            "remaining_rounds": 2,
            "restart_required": False,
            "reply_strategy": "start_new_case",
            "carried_forward_fields": [],
        },
        "explanation_payload": None,
        "refinement_status": "clarification_required",
    }
    session = build_stored_session(
        session_id="session-refinement-generic",
        current_public_state="RUNTIME_SUCCESS",
        latest_turn_id="turn-refinement-clarification",
        last_turn_index=2,
        is_case_complete=True,
        case_completion_reason="runtime_success",
        restart_required=False,
        last_runtime_request_json={"Income": 72},
        latest_runtime_backed_turn_id="turn-runtime",
        pending_refinement_clarification_json={
            "ambiguities": ["Clarify what 'better' means for this case."],
            "next_required_input": "State which outcome should improve.",
        },
        canonical_session_state_json={
            "profile_facts": {"Income": 72},
            "hard_constraints": {},
            "soft_preferences": {},
        },
    )

    detail = serialize_session_detail(
        session,
        turn_count=2,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=refinement_clarification_turn,
    )
    context = _build_session_page_render_context(
        detail,
        turns=[refinement_clarification_turn, runtime_turn],
        latest_turn=refinement_clarification_turn,
    )

    assert detail.render_hints is not None
    assert detail.render_hints.primary_action_type == "clarify_refinement"
    assert detail.render_hints.right_rail_anchor == "result-card"
    assert detail.render_hints.composer_context.advanced_controls_relevant is False
    assert context["page_state"] == "refinement_clarification"
    assert context["show_advanced_controls_by_default"] is False


def test_session_page_context_runtime_success_counterfactual_uses_explicit_visible_changes(tmp_path):
    latest_turn_payload = {
        "turn_id": "turn-runtime",
        "public_state": "RUNTIME_SUCCESS",
        "turn_kind": "message",
        "assistant_text": "A feasible counterfactual was found.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "reason_codes": [],
            "changed_fields": ["Income", "CDAccount"],
            "counterfactual_summary": {
                "profile_diff": {
                    "Income": {"from": 40, "to": 72},
                    "CDAccount": {"from": 0, "to": 1},
                }
            },
        },
        "refinement_status": None,
    }
    session = build_stored_session(
        session_id="session-runtime-visible-counterfactual",
        current_public_state="RUNTIME_SUCCESS",
        latest_turn_id="turn-runtime",
        last_turn_index=1,
        is_case_complete=True,
        case_completion_reason="runtime_success",
        restart_required=False,
        last_runtime_request_json={"Income": 72, "CDAccount": 1},
        latest_runtime_backed_turn_id="turn-runtime",
        canonical_session_state_json={
            "profile_facts": {
                "Income": 40,
                "Family": 1,
                "CCAvg": 2.5,
                "Education": 2,
                "Mortgage": 0,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 0,
                "CreditCard": 0,
            },
            "hard_constraints": {},
            "soft_preferences": {},
        },
    )

    detail = serialize_session_detail(
        session,
        turn_count=1,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )
    context = _build_session_page_render_context(
        detail,
        turns=[latest_turn_payload],
        latest_turn=latest_turn_payload,
    )
    expected_text = (
        "Your current profile would be rejected for the bank loan. "
        "However, I found a way to get approved — here are the recommended changes: Income: 40 -> 72, CDAccount: No -> Yes."
    )

    assert detail.render_hints is not None
    assert detail.render_hints.primary_chat_text == expected_text
    assert detail.render_hints.primary_action_type == "no_action_required"
    assert "2 fields to change" in detail.render_hints.supporting_detail_facts
    assert context["page_state"] == "runtime_success"
    assert context["transcript_items"][2]["text"] == expected_text


def test_session_page_context_runtime_reject_constraints_blocked_uses_constraint_specific_copy(tmp_path):
    latest_turn_payload = {
        "turn_id": "turn-reject",
        "public_state": "RUNTIME_REJECT",
        "turn_kind": "message",
        "user_input": "Income 49, Family 4, CCAvg 1.6.",
        "assistant_text": "Runtime completed without a feasible counterfactual.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "runtime_reject",
            "reason_codes": ["REQUEST_CONSTRAINTS_BLOCKED"],
            "changed_fields": [],
            "counterfactual_summary": None,
            "included_suggestion_types": ["relax_constraints"],
        },
        "active_constraint_spec": {"max_changed_features": 1},
        "refinement_status": None,
    }
    session = build_stored_session(
        session_id="session-reject-constraints",
        current_public_state="RUNTIME_REJECT",
        latest_turn_id="turn-reject",
        last_turn_index=1,
        is_case_complete=True,
        case_completion_reason="runtime_reject",
        last_runtime_request_json={"Income": 49},
        latest_runtime_backed_turn_id="turn-reject",
        active_constraint_spec_json={"max_changed_features": 1},
    )
    detail = serialize_session_detail(
        session,
        turn_count=1,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )

    context = _build_session_page_render_context(
        detail,
        turns=[latest_turn_payload],
        latest_turn=latest_turn_payload,
    )

    assert detail.render_hints is not None
    assert detail.render_hints.primary_chat_text == (
        "Your current profile would be rejected for the bank loan. "
        "Unfortunately, I couldn't find approved-path changes that also respect your constraints. "
        "Try relaxing some constraints (e.g. allow more fields to change) or start a new case."
    )
    assert detail.render_hints.primary_action_type == "relax_constraints_or_restart"
    assert detail.render_hints.supporting_detail_title == "Why no recommendation"
    assert "Blocked by your constraints" in detail.render_hints.supporting_detail_facts
    assert context["transcript_items"][2]["text"] == (
        "Your current profile would be rejected for the bank loan. "
        "Unfortunately, I couldn't find approved-path changes that also respect your constraints. "
        "Try relaxing some constraints (e.g. allow more fields to change) or start a new case."
    )


def test_session_page_context_runtime_reject_emits_runtime_summary_copy(tmp_path):
    latest_turn_payload = {
        "turn_id": "turn-reject",
        "public_state": "RUNTIME_REJECT",
        "turn_kind": "message",
        "user_input": "Income 49, Family 4, CCAvg 1.6.",
        "assistant_text": "Runtime completed without a feasible counterfactual.",
        "clarification_payload": None,
        "explanation_payload": {
            "summary_type": "runtime_reject",
            "reason_codes": ["NO_FEASIBLE_CF_FOUND"],
            "changed_fields": [],
            "counterfactual_summary": None,
        },
        "refinement_status": None,
    }
    session = build_stored_session(
        session_id="session-reject",
        current_public_state="RUNTIME_REJECT",
        latest_turn_id="turn-reject",
        last_turn_index=1,
        is_case_complete=True,
        case_completion_reason="runtime_reject",
        last_runtime_request_json={"Income": 49},
        latest_runtime_backed_turn_id="turn-reject",
    )
    detail = serialize_session_detail(
        session,
        turn_count=1,
        artifact_root=tmp_path / "artifacts",
        latest_turn_payload=latest_turn_payload,
    )

    context = _build_session_page_render_context(
        detail,
        turns=[latest_turn_payload],
        latest_turn=latest_turn_payload,
    )

    assert context["page_state"] == "runtime_reject"
    assert context["chat_header_summary"]["state_label"] == "runtime reject"
    assert context["composer_mode"] == "refinement"
    assert context["composer_context"]["submit_target"] == "refinements"
    assert detail.render_hints is not None
    assert detail.render_hints.primary_action_type == "relax_constraints_or_restart"
    assert detail.render_hints.primary_action_items == ["Relax constraints", "Start a new case"]
    assert detail.render_hints.right_rail_anchor == "result-card"
    assert [item["kind"] for item in context["transcript_items"]] == [
        "user_message",
        "system_marker",
        "assistant_message",
        "inline_detail_toggle",
    ]
    assert context["transcript_items"][1]["label"] == "No recommendation available"
    assert context["transcript_items"][2]["text"] == (
        "Your current profile would be rejected for the bank loan. "
        "Unfortunately, the system could not find any feasible changes to get you approved. "
        "Try submitting a different profile or relaxing your constraints."
    )


def test_artifact_endpoint_rejects_path_traversal(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
            "runtime_mode": "stable_demo",
            "invariant_validation": {
                "status": "skipped_no_counterfactual",
                "public_safe": True,
                "reason_codes": [],
                "validated_summary_type": "no_recourse_needed",
                "validated_changed_fields": [],
                "details": {"reason": "no_counterfactual_required"},
            },
            "debug_trace": {
                "runtime_mode": "stable_demo",
                "deterministic_seed": 1234,
                "policy_version": "bank_policy_v1",
                "mi_feature_pairs": [["CCAvg", "Income"]],
                "state_trace": ["READY_FOR_PREDICTION", "TERMINAL_SUCCESS"],
                "service_errors": [],
                "ufce_methods": [],
                "winning_path": None,
                "reject_path": None,
            },
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    config = build_config(tmp_path)
    repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
    app = create_app(config=config, orchestrator=orchestrator, repository=repository)

    session_id = api_request(app, "POST", "/api/v1/sessions").json()["session_id"]
    turn_payload = api_request(
        app,
        "POST",
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "user_input": (
                "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
                "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
            )
        },
    ).json()

    response = api_request(
        app,
        "GET",
        f"/api/v1/sessions/{session_id}/artifacts/{turn_payload['turn_id']}/../README.md"
    )

    assert response.status_code == 404
