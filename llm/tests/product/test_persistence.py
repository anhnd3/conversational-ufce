from __future__ import annotations

import sqlite3

from llm.src.product.persistence import (
    DB_SCHEMA_VERSION,
    SESSION_LIFECYCLE_ACTIVE,
    SESSION_LIFECYCLE_ARCHIVED,
    SessionRepository,
)


def test_repository_initializes_schema_metadata(tmp_path):
    repository = SessionRepository(tmp_path / "sessions.sqlite3", app_version="phase3_2_test")

    metadata = repository.metadata()

    assert metadata["db_schema_version"] == DB_SCHEMA_VERSION
    assert metadata["app_version"] == "phase3_2_test"


def test_repository_round_trip_session_and_turn(tmp_path):
    repository = SessionRepository(tmp_path / "sessions.sqlite3", app_version="phase3_2_test")
    session = repository.create_session(model_alias="stub-model", runtime_mode="stable_demo")
    stored_turn = repository.save_turn(
        session_id=session.session_id,
        turn_index=1,
        user_input="Income 100",
        assistant_text="Need more information.",
        public_state="NEEDS_CLARIFICATION",
        builder_status="NEEDS_CLARIFICATION",
        transition_reason="missing_required_fields",
        response_decision_json={"final_public_state": "NEEDS_CLARIFICATION"},
        clarification_payload_json={"clarification_type": "missing_information"},
        explanation_payload_json=None,
        debug_summary_json={"merge_applied": False},
        artifact_dir=str(tmp_path / "artifacts" / "turn1"),
        parent_turn_id=None,
        merge_applied=False,
        pending_clarification_json={
            "prior_cf_request": {"Income": 100},
            "missing_fields": ["CCAvg"],
            "required_field_order": ["Income", "CCAvg"],
            "originating_turn_id": "turn_1",
        },
        turn_id="turn_1",
        created_at="2026-03-22T00:00:00Z",
        timing_metrics_json={"end_to_end_latency_ms": 12.5},
        active_constraint_spec_session_json={"max_changed_features": 1},
        last_runtime_request_json={"dataset": "bank", "profile": {"Income": 100}},
        refinement_revision_index_session=1,
        refinement_rounds_used_session=1,
        refinement_round_limit_session=3,
        pending_refinement_clarification_json={
            "originating_turn_id": "turn_1",
            "ambiguities": ["Need one clearer blocked-field update."],
            "next_required_input": "State the blocked field update in one sentence.",
            "parent_terminal_turn_id": "turn_0",
            "parent_refinement_revision_index": None,
        },
        latest_runtime_backed_turn_id="turn_0",
        canonical_runtime_result_json={
            "dataset": "bank",
            "backend_id": "ufce",
            "canonical_request": {"dataset_id": "bank"},
        },
        verification_artifacts_json={
            "verification_results": [],
            "reason_code_version": "reason_codes_v1",
        },
        canonical_session_state_json={
            "session_id": session.session_id,
            "dataset_id": "bank",
            "backend_id": "ufce",
            "profile_facts": {"Income": 100},
            "hard_constraints": {"max_changed_features": 1},
            "canonical_mirror_ok": True,
        },
        canonical_state_source="canonical_authoritative",
        canonical_mirror_ok=True,
    )

    restored_session = repository.get_session(session.session_id)
    restored_turns = repository.list_turns(session.session_id)

    assert restored_session.last_turn_index == 1
    assert restored_session.pending_clarification_json is not None
    assert restored_session.lifecycle_status == SESSION_LIFECYCLE_ACTIVE
    assert restored_session.archived_at is None
    assert restored_session.active_constraint_spec_json == {"max_changed_features": 1}
    assert restored_session.last_runtime_request_json == {"dataset": "bank", "profile": {"Income": 100}}
    assert restored_session.refinement_revision_index == 1
    assert restored_session.refinement_rounds_used == 1
    assert restored_session.pending_refinement_clarification_json is not None
    assert restored_session.latest_runtime_backed_turn_id == "turn_0"
    assert restored_session.canonical_state_source == "canonical_authoritative"
    assert restored_session.canonical_mirror_ok is True
    assert restored_session.canonical_session_state_json is not None
    assert restored_turns[0].timing_metrics_json == {"end_to_end_latency_ms": 12.5}
    assert restored_turns[0].canonical_runtime_result_json is not None
    assert restored_turns[0].verification_artifacts_json == {
        "verification_results": [],
        "reason_code_version": "reason_codes_v1",
    }
    assert restored_turns[0].canonical_session_state_json is not None
    assert restored_turns == [stored_turn]


def test_repository_archive_session_marks_read_only_and_clears_pending(tmp_path):
    repository = SessionRepository(tmp_path / "sessions.sqlite3", app_version="phase3_2_test")
    session = repository.create_session(model_alias="stub-model", runtime_mode="stable_demo")
    repository.save_turn(
        session_id=session.session_id,
        turn_index=1,
        user_input="Income 100",
        assistant_text="Need more information.",
        public_state="NEEDS_CLARIFICATION",
        builder_status="NEEDS_CLARIFICATION",
        transition_reason="missing_required_fields",
        response_decision_json={"final_public_state": "NEEDS_CLARIFICATION"},
        clarification_payload_json={"clarification_type": "missing_information"},
        explanation_payload_json=None,
        debug_summary_json={"merge_applied": False},
        artifact_dir=str(tmp_path / "artifacts" / "turn1"),
        parent_turn_id=None,
        merge_applied=False,
        pending_clarification_json={
            "prior_cf_request": {"Income": 100},
            "missing_fields": ["CCAvg"],
            "required_field_order": ["Income", "CCAvg"],
            "originating_turn_id": "turn_1",
        },
        turn_id="turn_1",
        created_at="2026-03-22T00:00:00Z",
        pending_refinement_clarification_json={
            "originating_turn_id": "turn_1",
            "ambiguities": ["Need one clearer blocked-field update."],
            "next_required_input": "State the blocked field update in one sentence.",
            "parent_terminal_turn_id": "turn_0",
            "parent_refinement_revision_index": None,
        },
    )

    archived = repository.archive_session(session.session_id)

    assert archived.lifecycle_status == SESSION_LIFECYCLE_ARCHIVED
    assert archived.archived_at is not None
    assert archived.pending_clarification_json is None
    assert archived.pending_refinement_clarification_json is None


def test_repository_migrates_existing_v1_sessions_table(tmp_path):
    db_path = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            """
            CREATE TABLE app_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                db_schema_version INTEGER NOT NULL,
                app_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO app_metadata (singleton, db_schema_version, app_version, created_at, updated_at)
            VALUES (1, 1, 'phase3_2_test', '2026-03-22T00:00:00Z', '2026-03-22T00:00:00Z')
            """
        )
        connection.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                current_public_state TEXT,
                pending_clarification_json TEXT,
                latest_turn_id TEXT,
                last_turn_index INTEGER NOT NULL,
                model_alias TEXT NOT NULL,
                runtime_mode TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                public_state TEXT NOT NULL,
                builder_status TEXT,
                transition_reason TEXT,
                response_decision_json TEXT,
                clarification_payload_json TEXT,
                explanation_payload_json TEXT,
                debug_summary_json TEXT,
                artifact_dir TEXT,
                parent_turn_id TEXT,
                merge_applied INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, turn_index)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO sessions (
                session_id,
                created_at,
                updated_at,
                current_public_state,
                pending_clarification_json,
                latest_turn_id,
                last_turn_index,
                model_alias,
                runtime_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session_legacy",
                "2026-03-22T00:00:00Z",
                "2026-03-22T00:00:00Z",
                None,
                None,
                None,
                0,
                "stub-model",
                "stable_demo",
            ),
        )
        connection.commit()

    repository = SessionRepository(db_path, app_version="phase3_2_test")
    session = repository.get_session("session_legacy")
    metadata = repository.metadata()

    assert metadata["db_schema_version"] == DB_SCHEMA_VERSION
    assert session.lifecycle_status == SESSION_LIFECYCLE_ACTIVE
    assert session.archived_at is None
    assert session.active_constraint_spec_json is None
    assert session.last_runtime_request_json is None
    assert session.refinement_revision_index == 0
    assert session.refinement_rounds_used == 0
    assert session.refinement_round_limit == 3
    assert session.pending_refinement_clarification_json is None
    assert session.latest_runtime_backed_turn_id is None
    assert session.canonical_session_state_json is None
    assert session.canonical_state_source is None
    assert session.canonical_mirror_ok is True
