from __future__ import annotations

import contextlib
import io
import sys
from types import SimpleNamespace

from llm.src.part2_eval.common import (
    apply_script_mismatch_validation,
    build_script_mismatch_summary,
    build_session_detail_payload,
    call_with_legacy_stdout_redirect,
    lm_studio_preflight,
    progress_iter,
    replay_scripted_session_case,
    recompute_and_validate_aggregates,
    write_optional_summary_outputs,
)


def test_recompute_and_validate_aggregates_reports_differences():
    validation = recompute_and_validate_aggregates(
        expected_blocks={"metrics": {"value": 1.0}},
        recomputed_blocks={"metrics": {"value": 0.5}},
    )

    assert validation["ok"] is False
    assert validation["difference_count"] == 1
    assert validation["differences"][0]["path"] == "metrics.value"


def test_build_session_detail_payload_exposes_active_constraint_state():
    payload = build_session_detail_payload(
        SimpleNamespace(
            session_id="session_1",
            current_public_state="RUNTIME_SUCCESS",
            clarification_turns_used=1,
            is_case_complete=True,
            case_completion_reason="runtime_success",
            restart_required=False,
            active_constraint_spec_json={"max_changed_features": 1},
            last_runtime_request_json={"dataset": "bank"},
            refinement_revision_index=2,
            refinement_rounds_used=2,
            refinement_round_limit=3,
            pending_refinement_clarification_json=None,
            latest_runtime_backed_turn_id="turn_1",
        )
    )

    assert payload["session_id"] == "session_1"
    assert payload["active_constraint_spec"] == {"max_changed_features": 1}
    assert payload["refinement_revision_index"] == 2


def test_replay_scripted_session_case_records_premature_completion():
    session_state = SimpleNamespace(
        session_id="session_1",
        current_public_state="RUNTIME_SUCCESS",
        clarification_turns_used=1,
        is_case_complete=True,
        case_completion_reason="runtime_success",
        restart_required=True,
        active_constraint_spec_json={},
        last_runtime_request_json={"dataset": "bank"},
        refinement_revision_index=0,
        refinement_rounds_used=0,
        refinement_round_limit=3,
        pending_refinement_clarification_json=None,
        latest_runtime_backed_turn_id="turn_1",
    )

    class FakeService:
        def create_session(self):
            return SimpleNamespace(session_id="session_1")

        def submit_message(self, session_id, turn_text):
            assert session_id == "session_1"
            assert turn_text == "turn one"
            return {"turn_id": "turn_1"}

        def build_turn_response(self, stored_turn):
            assert stored_turn == {"turn_id": "turn_1"}
            return {
                "public_state": "RUNTIME_SUCCESS",
                "is_case_complete": True,
                "case_completion_reason": "runtime_success",
            }

    handle = SimpleNamespace(
        service=FakeService(),
        repository=SimpleNamespace(get_session=lambda session_id: session_state),
    )

    replay = replay_scripted_session_case(
        handle=handle,
        case={"case_id": "case-1", "turns": ["turn one", "turn two"]},
    )

    assert replay["scripted_turn_count"] == 2
    assert replay["executed_turn_count"] == 1
    assert replay["script_execution_status"] == "script_mismatch"
    assert replay["failed_turn_index"] == 2
    assert replay["script_mismatch_reason"] == "premature_case_completion"
    assert replay["premature_terminal_state"] == "RUNTIME_SUCCESS"


def test_replay_scripted_session_case_records_non_terminal_script_exhaustion():
    session_state = SimpleNamespace(
        session_id="session_1",
        current_public_state="NEEDS_CLARIFICATION",
        clarification_turns_used=2,
        is_case_complete=False,
        case_completion_reason=None,
        restart_required=False,
        active_constraint_spec_json={},
        last_runtime_request_json={"dataset": "bank"},
        refinement_revision_index=0,
        refinement_rounds_used=0,
        refinement_round_limit=3,
        pending_refinement_clarification_json=None,
        latest_runtime_backed_turn_id=None,
    )

    class FakeService:
        def __init__(self):
            self._calls = 0

        def create_session(self):
            return SimpleNamespace(session_id="session_1")

        def submit_message(self, session_id, turn_text):
            assert session_id == "session_1"
            self._calls += 1
            return {"turn_id": f"turn_{self._calls}", "turn_text": turn_text}

        def build_turn_response(self, stored_turn):
            return {
                "public_state": "NEEDS_CLARIFICATION",
                "is_case_complete": False,
                "case_completion_reason": None,
                "turn_id": stored_turn["turn_id"],
            }

    handle = SimpleNamespace(
        service=FakeService(),
        repository=SimpleNamespace(get_session=lambda session_id: session_state),
    )

    replay = replay_scripted_session_case(
        handle=handle,
        case={"case_id": "case-1", "turns": ["turn one", "turn two"]},
    )

    assert replay["executed_turn_count"] == 2
    assert replay["script_execution_status"] == "script_mismatch"
    assert replay["failed_turn_index"] == 2
    assert replay["script_mismatch_reason"] == "script_exhausted_non_terminal"
    assert replay["premature_terminal_state"] == "NEEDS_CLARIFICATION"


def test_apply_script_mismatch_validation_forces_failed_validation():
    mismatch_summary = build_script_mismatch_summary(
        [
            {"case_id": "case-1", "script_execution_status": "completed"},
            {
                "case_id": "case-2",
                "script_execution_status": "script_mismatch",
                "script_mismatch_reason": "premature_case_completion",
            },
        ]
    )

    validation = apply_script_mismatch_validation(
        {"ok": True, "difference_count": 0, "differences": [], "validated_aggregates": {}},
        script_mismatch_summary=mismatch_summary,
    )

    assert mismatch_summary["count"] == 1
    assert mismatch_summary["case_identifiers"] == ["case-2"]
    assert validation["ok"] is False
    assert validation["difference_count"] == 1
    assert validation["differences"][0]["path"] == "script_mismatch_summary.count"


def test_lm_studio_preflight_checks_model_presence(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "model-a"}, {"id": "model-b"}]}

    monkeypatch.setattr("llm.src.part2_eval.common.requests.get", lambda url, timeout: Response())

    present = lm_studio_preflight(api_base="http://localhost:1234", model_alias="model-a")
    missing = lm_studio_preflight(api_base="http://localhost:1234", model_alias="model-c")

    assert present["ok"] is True
    assert present["model_alias_present"] is True
    assert missing["ok"] is False
    assert missing["detail"] == "model_alias_missing:model-c"


def test_progress_iter_can_be_disabled():
    values = list(progress_iter([1, 2, 3], enabled=False, desc="demo", unit="item"))

    assert values == [1, 2, 3]


def test_progress_iter_writes_to_stderr(monkeypatch):
    captured = {}

    def fake_tqdm(iterable, **kwargs):
        captured["kwargs"] = kwargs
        return iterable

    monkeypatch.setattr("llm.src.part2_eval.common.tqdm", fake_tqdm)

    values = list(progress_iter([1, 2], enabled=True, desc="demo", unit="case"))

    assert values == [1, 2]
    assert captured["kwargs"]["file"] is sys.stderr
    assert captured["kwargs"]["desc"] == "demo"


def test_call_with_legacy_stdout_redirect_routes_stdout_to_stderr():
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    def noisy():
        print("legacy-noise")
        return "ok"

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = call_with_legacy_stdout_redirect(noisy)

    assert result == "ok"
    assert stdout_buffer.getvalue() == ""
    assert "legacy-noise" in stderr_buffer.getvalue()


def test_write_optional_summary_outputs_writes_json_and_markdown(tmp_path):
    summary = {"ok": True}
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"

    write_optional_summary_outputs(
        summary=summary,
        summary_json_path=summary_json,
        summary_markdown_path=summary_md,
        markdown_text="# Demo\n",
    )

    assert summary_json.read_text(encoding="utf-8").strip().startswith("{")
    assert summary_md.read_text(encoding="utf-8") == "# Demo\n"
