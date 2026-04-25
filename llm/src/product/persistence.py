from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm.src.utils.hashing import make_run_id
from llm.src.utils.time import local_now_iso


DB_SCHEMA_VERSION = 6
SESSION_LIFECYCLE_ACTIVE = "active"
SESSION_LIFECYCLE_ARCHIVED = "archived"
_UNSET = object()


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    dataset_key: str
    created_at: str
    updated_at: str
    current_public_state: str | None
    pending_clarification_json: dict[str, Any] | None
    latest_turn_id: str | None
    last_turn_index: int
    model_alias: str
    runtime_mode: str
    lifecycle_status: str
    archived_at: str | None
    clarification_turns_used: int
    is_case_complete: bool
    case_completion_reason: str | None
    restart_required: bool
    active_constraint_spec_json: dict[str, Any] | None
    last_runtime_request_json: dict[str, Any] | None
    refinement_revision_index: int
    refinement_rounds_used: int
    refinement_round_limit: int
    pending_refinement_clarification_json: dict[str, Any] | None
    latest_runtime_backed_turn_id: str | None
    canonical_session_state_json: dict[str, Any] | None
    canonical_state_source: str | None
    canonical_mirror_ok: bool


@dataclass(frozen=True)
class StoredTurn:
    turn_id: str
    session_id: str
    turn_index: int
    user_input: str
    assistant_text: str
    public_state: str
    builder_status: str | None
    transition_reason: str | None
    response_decision_json: dict[str, Any] | None
    clarification_payload_json: dict[str, Any] | None
    explanation_payload_json: dict[str, Any] | None
    debug_summary_json: dict[str, Any] | None
    artifact_dir: str | None
    parent_turn_id: str | None
    merge_applied: bool
    created_at: str
    clarification_turns_used: int
    is_case_complete: bool
    case_completion_reason: str | None
    restart_required: bool
    timing_metrics_json: dict[str, Any] | None
    turn_kind: str
    refinement_status: str | None
    refinement_revision_index: int | None
    parent_terminal_turn_id: str | None
    parent_refinement_revision_index: int | None
    active_constraint_spec_json: dict[str, Any] | None
    constraint_feedback_delta_json: dict[str, Any] | None
    refinement_rounds_used: int
    refinement_round_limit: int | None
    canonical_runtime_result_json: dict[str, Any] | None
    verification_artifacts_json: dict[str, Any] | None
    canonical_session_state_json: dict[str, Any] | None


class SessionRepository:
    def __init__(self, db_path: Path, *, app_version: str) -> None:
        self.db_path = Path(db_path)
        self.app_version = app_version
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        created_at = local_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_metadata (
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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    dataset_key TEXT NOT NULL DEFAULT 'bank',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    current_public_state TEXT,
                    pending_clarification_json TEXT,
                    latest_turn_id TEXT,
                    last_turn_index INTEGER NOT NULL,
                    model_alias TEXT NOT NULL,
                    runtime_mode TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL DEFAULT 'active',
                    archived_at TEXT,
                    clarification_turns_used INTEGER NOT NULL DEFAULT 0,
                    is_case_complete INTEGER NOT NULL DEFAULT 0,
                    case_completion_reason TEXT,
                    restart_required INTEGER NOT NULL DEFAULT 0,
                    active_constraint_spec_json TEXT,
                    last_runtime_request_json TEXT,
                    refinement_revision_index INTEGER NOT NULL DEFAULT 0,
                    refinement_rounds_used INTEGER NOT NULL DEFAULT 0,
                    refinement_round_limit INTEGER NOT NULL DEFAULT 3,
                    pending_refinement_clarification_json TEXT,
                    latest_runtime_backed_turn_id TEXT,
                    canonical_session_state_json TEXT,
                    canonical_state_source TEXT,
                    canonical_mirror_ok INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
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
                    clarification_turns_used INTEGER NOT NULL DEFAULT 0,
                    is_case_complete INTEGER NOT NULL DEFAULT 0,
                    case_completion_reason TEXT,
                    restart_required INTEGER NOT NULL DEFAULT 0,
                    timing_metrics_json TEXT,
                    turn_kind TEXT NOT NULL DEFAULT 'message',
                    refinement_status TEXT,
                    refinement_revision_index INTEGER,
                    parent_terminal_turn_id TEXT,
                    parent_refinement_revision_index INTEGER,
                    active_constraint_spec_json TEXT,
                    constraint_feedback_delta_json TEXT,
                    refinement_rounds_used INTEGER NOT NULL DEFAULT 0,
                    refinement_round_limit INTEGER,
                    canonical_runtime_result_json TEXT,
                    verification_artifacts_json TEXT,
                    canonical_session_state_json TEXT,
                    UNIQUE(session_id, turn_index),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            self._migrate_sessions_table(connection)
            self._migrate_turns_table(connection)
            row = connection.execute("SELECT singleton FROM app_metadata WHERE singleton = 1").fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO app_metadata (
                        singleton,
                        db_schema_version,
                        app_version,
                        created_at,
                        updated_at
                    ) VALUES (1, ?, ?, ?, ?)
                    """,
                    (DB_SCHEMA_VERSION, self.app_version, created_at, created_at),
                )
            else:
                connection.execute(
                    """
                    UPDATE app_metadata
                    SET db_schema_version = ?, app_version = ?, updated_at = ?
                    WHERE singleton = 1
                    """,
                    (DB_SCHEMA_VERSION, self.app_version, created_at),
                )
            connection.commit()

    def metadata(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM app_metadata WHERE singleton = 1").fetchone()
        if row is None:
            raise RuntimeError("app_metadata row is missing.")
        return dict(row)

    def create_session(self, *, model_alias: str, runtime_mode: str, dataset_key: str = "bank") -> StoredSession:
        created_at = local_now_iso()
        session_id = make_run_id().replace("run_", "session_")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id,
                    dataset_key,
                    created_at,
                    updated_at,
                    current_public_state,
                    pending_clarification_json,
                    latest_turn_id,
                    last_turn_index,
                    model_alias,
                    runtime_mode,
                    lifecycle_status,
                    archived_at,
                    clarification_turns_used,
                    is_case_complete,
                    case_completion_reason,
                    restart_required,
                    active_constraint_spec_json,
                    last_runtime_request_json,
                    refinement_revision_index,
                    refinement_rounds_used,
                    refinement_round_limit,
                    pending_refinement_clarification_json,
                    latest_runtime_backed_turn_id,
                    canonical_session_state_json,
                    canonical_state_source,
                    canonical_mirror_ok
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    dataset_key,
                    created_at,
                    created_at,
                    None,
                    None,
                    None,
                    0,
                    model_alias,
                    runtime_mode,
                    SESSION_LIFECYCLE_ACTIVE,
                    None,
                    0,
                    0,
                    None,
                    0,
                    None,
                    None,
                    0,
                    0,
                    3,
                    None,
                    None,
                    None,
                    "legacy_mirror",
                    1,
                ),
            )
            connection.commit()
        return self.get_session(session_id)

    def list_sessions(self) -> list[StoredSession]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM sessions").fetchall()
        sessions = [self._row_to_session(row) for row in rows]
        return sorted(
            sessions,
            key=lambda item: (
                0 if item.lifecycle_status == SESSION_LIFECYCLE_ACTIVE else 1,
                -parse_sort_timestamp(item.updated_at),
                -parse_sort_timestamp(item.created_at),
            ),
        )

    def get_session(self, session_id: str) -> StoredSession:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return self._row_to_session(row)

    def save_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        user_input: str,
        assistant_text: str,
        public_state: str,
        builder_status: str | None,
        transition_reason: str | None,
        response_decision_json: dict[str, Any] | None,
        clarification_payload_json: dict[str, Any] | None,
        explanation_payload_json: dict[str, Any] | None,
        debug_summary_json: dict[str, Any] | None,
        artifact_dir: str | None,
        parent_turn_id: str | None,
        merge_applied: bool,
        pending_clarification_json: dict[str, Any] | None,
        turn_id: str,
        created_at: str,
        clarification_turns_used: int = 0,
        is_case_complete: bool = False,
        case_completion_reason: str | None = None,
        restart_required: bool = False,
        timing_metrics_json: dict[str, Any] | None = None,
        turn_kind: str = "message",
        refinement_status: str | None = None,
        refinement_revision_index: int | None = None,
        parent_terminal_turn_id: str | None = None,
        parent_refinement_revision_index: int | None = None,
        active_constraint_spec_json: dict[str, Any] | None = None,
        constraint_feedback_delta_json: dict[str, Any] | None = None,
        refinement_rounds_used: int = 0,
        refinement_round_limit: int | None = None,
        active_constraint_spec_session_json: Any = _UNSET,
        last_runtime_request_json: Any = _UNSET,
        refinement_revision_index_session: Any = _UNSET,
        refinement_rounds_used_session: Any = _UNSET,
        refinement_round_limit_session: Any = _UNSET,
        pending_refinement_clarification_json: Any = _UNSET,
        latest_runtime_backed_turn_id: Any = _UNSET,
        canonical_runtime_result_json: dict[str, Any] | None = None,
        verification_artifacts_json: dict[str, Any] | None = None,
        canonical_session_state_json: Any = _UNSET,
        canonical_state_source: Any = _UNSET,
        canonical_mirror_ok: Any = _UNSET,
    ) -> StoredTurn:
        current_session = self.get_session(session_id)
        if refinement_round_limit_session is _UNSET:
            refinement_round_limit_session = current_session.refinement_round_limit
        if refinement_rounds_used_session is _UNSET:
            refinement_rounds_used_session = current_session.refinement_rounds_used
        if refinement_revision_index_session is _UNSET:
            refinement_revision_index_session = current_session.refinement_revision_index
        if active_constraint_spec_session_json is _UNSET:
            active_constraint_spec_session_json = current_session.active_constraint_spec_json
        if last_runtime_request_json is _UNSET:
            last_runtime_request_json = current_session.last_runtime_request_json
        if pending_refinement_clarification_json is _UNSET:
            pending_refinement_clarification_json = current_session.pending_refinement_clarification_json
        if latest_runtime_backed_turn_id is _UNSET:
            latest_runtime_backed_turn_id = current_session.latest_runtime_backed_turn_id
        if canonical_session_state_json is _UNSET:
            canonical_session_state_json = current_session.canonical_session_state_json
        if canonical_state_source is _UNSET:
            canonical_state_source = current_session.canonical_state_source
        if canonical_mirror_ok is _UNSET:
            canonical_mirror_ok = current_session.canonical_mirror_ok
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO turns (
                    turn_id,
                    session_id,
                    turn_index,
                    user_input,
                    assistant_text,
                    public_state,
                    builder_status,
                    transition_reason,
                    response_decision_json,
                    clarification_payload_json,
                    explanation_payload_json,
                    debug_summary_json,
                    artifact_dir,
                    parent_turn_id,
                    merge_applied,
                    created_at,
                    clarification_turns_used,
                    is_case_complete,
                    case_completion_reason,
                    restart_required,
                    timing_metrics_json,
                    turn_kind,
                    refinement_status,
                    refinement_revision_index,
                    parent_terminal_turn_id,
                    parent_refinement_revision_index,
                    active_constraint_spec_json,
                    constraint_feedback_delta_json,
                    refinement_rounds_used,
                    refinement_round_limit,
                    canonical_runtime_result_json,
                    verification_artifacts_json,
                    canonical_session_state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    turn_index,
                    user_input,
                    assistant_text,
                    public_state,
                    builder_status,
                    transition_reason,
                    _json_dump(response_decision_json),
                    _json_dump(clarification_payload_json),
                    _json_dump(explanation_payload_json),
                    _json_dump(debug_summary_json),
                    artifact_dir,
                    parent_turn_id,
                    1 if merge_applied else 0,
                    created_at,
                    int(clarification_turns_used),
                    1 if is_case_complete else 0,
                    case_completion_reason,
                    1 if restart_required else 0,
                    _json_dump(timing_metrics_json),
                    turn_kind,
                    refinement_status,
                    refinement_revision_index,
                    parent_terminal_turn_id,
                    parent_refinement_revision_index,
                    _json_dump(active_constraint_spec_json),
                    _json_dump(constraint_feedback_delta_json),
                    int(refinement_rounds_used),
                    refinement_round_limit,
                    _json_dump(canonical_runtime_result_json),
                    _json_dump(verification_artifacts_json),
                    _json_dump(canonical_session_state_json),
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?,
                    current_public_state = ?,
                    pending_clarification_json = ?,
                    latest_turn_id = ?,
                    last_turn_index = ?,
                    clarification_turns_used = ?,
                    is_case_complete = ?,
                    case_completion_reason = ?,
                    restart_required = ?,
                    active_constraint_spec_json = ?,
                    last_runtime_request_json = ?,
                    refinement_revision_index = ?,
                    refinement_rounds_used = ?,
                    refinement_round_limit = ?,
                    pending_refinement_clarification_json = ?,
                    latest_runtime_backed_turn_id = ?,
                    canonical_session_state_json = ?,
                    canonical_state_source = ?,
                    canonical_mirror_ok = ?
                WHERE session_id = ?
                """,
                (
                    created_at,
                    public_state,
                    _json_dump(pending_clarification_json),
                    turn_id,
                    turn_index,
                    int(clarification_turns_used),
                    1 if is_case_complete else 0,
                    case_completion_reason,
                    1 if restart_required else 0,
                    _json_dump(active_constraint_spec_session_json),
                    _json_dump(last_runtime_request_json),
                    int(refinement_revision_index_session or 0),
                    int(refinement_rounds_used_session or 0),
                    int(refinement_round_limit_session or 3),
                    _json_dump(pending_refinement_clarification_json),
                    latest_runtime_backed_turn_id,
                    _json_dump(canonical_session_state_json),
                    canonical_state_source,
                    1 if canonical_mirror_ok else 0,
                    session_id,
                ),
            )
            connection.commit()
        return self.get_turn(session_id, turn_id)

    def list_turns(self, session_id: str, *, order: str = "desc") -> list[StoredTurn]:
        direction = "DESC" if order.lower() == "desc" else "ASC"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index {direction}",
                (session_id,),
            ).fetchall()
        return [self._row_to_turn(row) for row in rows]

    def archive_session(self, session_id: str) -> StoredSession:
        archived_at = local_now_iso()
        already_archived = False
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT lifecycle_status FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            if existing["lifecycle_status"] == SESSION_LIFECYCLE_ARCHIVED:
                already_archived = True
            else:
                connection.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?,
                        lifecycle_status = ?,
                        archived_at = ?,
                        pending_clarification_json = NULL,
                        pending_refinement_clarification_json = NULL
                    WHERE session_id = ?
                    """,
                    (archived_at, SESSION_LIFECYCLE_ARCHIVED, archived_at, session_id),
                )
                connection.commit()
        if already_archived:
            return self.get_session(session_id)
        return self.get_session(session_id)

    def get_turn(self, session_id: str, turn_id: str) -> StoredTurn:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown turn_id: {turn_id}")
        return self._row_to_turn(row)

    def _row_to_session(self, row: sqlite3.Row) -> StoredSession:
        return StoredSession(
            session_id=str(row["session_id"]),
            dataset_key=str(row["dataset_key"] or "bank"),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            current_public_state=row["current_public_state"],
            pending_clarification_json=_json_load(row["pending_clarification_json"]),
            latest_turn_id=row["latest_turn_id"],
            last_turn_index=int(row["last_turn_index"]),
            model_alias=str(row["model_alias"]),
            runtime_mode=str(row["runtime_mode"]),
            lifecycle_status=str(row["lifecycle_status"] or SESSION_LIFECYCLE_ACTIVE),
            archived_at=row["archived_at"],
            clarification_turns_used=int(row["clarification_turns_used"] or 0),
            is_case_complete=bool(row["is_case_complete"]),
            case_completion_reason=row["case_completion_reason"],
            restart_required=bool(row["restart_required"]),
            active_constraint_spec_json=_json_load(row["active_constraint_spec_json"]),
            last_runtime_request_json=_json_load(row["last_runtime_request_json"]),
            refinement_revision_index=int(row["refinement_revision_index"] or 0),
            refinement_rounds_used=int(row["refinement_rounds_used"] or 0),
            refinement_round_limit=int(row["refinement_round_limit"] or 3),
            pending_refinement_clarification_json=_json_load(row["pending_refinement_clarification_json"]),
            latest_runtime_backed_turn_id=row["latest_runtime_backed_turn_id"],
            canonical_session_state_json=_json_load(row["canonical_session_state_json"]),
            canonical_state_source=row["canonical_state_source"],
            canonical_mirror_ok=bool(row["canonical_mirror_ok"]),
        )

    def _row_to_turn(self, row: sqlite3.Row) -> StoredTurn:
        return StoredTurn(
            turn_id=str(row["turn_id"]),
            session_id=str(row["session_id"]),
            turn_index=int(row["turn_index"]),
            user_input=str(row["user_input"]),
            assistant_text=str(row["assistant_text"]),
            public_state=str(row["public_state"]),
            builder_status=row["builder_status"],
            transition_reason=row["transition_reason"],
            response_decision_json=_json_load(row["response_decision_json"]),
            clarification_payload_json=_json_load(row["clarification_payload_json"]),
            explanation_payload_json=_json_load(row["explanation_payload_json"]),
            debug_summary_json=_json_load(row["debug_summary_json"]),
            artifact_dir=row["artifact_dir"],
            parent_turn_id=row["parent_turn_id"],
            merge_applied=bool(row["merge_applied"]),
            created_at=str(row["created_at"]),
            clarification_turns_used=int(row["clarification_turns_used"] or 0),
            is_case_complete=bool(row["is_case_complete"]),
            case_completion_reason=row["case_completion_reason"],
            restart_required=bool(row["restart_required"]),
            timing_metrics_json=_json_load(row["timing_metrics_json"]),
            turn_kind=str(row["turn_kind"] or "message"),
            refinement_status=row["refinement_status"],
            refinement_revision_index=None
            if row["refinement_revision_index"] is None
            else int(row["refinement_revision_index"]),
            parent_terminal_turn_id=row["parent_terminal_turn_id"],
            parent_refinement_revision_index=None
            if row["parent_refinement_revision_index"] is None
            else int(row["parent_refinement_revision_index"]),
            active_constraint_spec_json=_json_load(row["active_constraint_spec_json"]),
            constraint_feedback_delta_json=_json_load(row["constraint_feedback_delta_json"]),
            refinement_rounds_used=int(row["refinement_rounds_used"] or 0),
            refinement_round_limit=None
            if row["refinement_round_limit"] is None
            else int(row["refinement_round_limit"]),
            canonical_runtime_result_json=_json_load(row["canonical_runtime_result_json"]),
            verification_artifacts_json=_json_load(row["verification_artifacts_json"]),
            canonical_session_state_json=_json_load(row["canonical_session_state_json"]),
        )

    def _migrate_sessions_table(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"]): row
            for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "lifecycle_status" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN lifecycle_status TEXT NOT NULL DEFAULT 'active'"
            )
        if "dataset_key" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN dataset_key TEXT NOT NULL DEFAULT 'bank'")
        if "archived_at" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN archived_at TEXT")
        if "clarification_turns_used" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN clarification_turns_used INTEGER NOT NULL DEFAULT 0"
            )
        if "is_case_complete" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN is_case_complete INTEGER NOT NULL DEFAULT 0"
            )
        if "case_completion_reason" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN case_completion_reason TEXT")
        if "restart_required" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN restart_required INTEGER NOT NULL DEFAULT 0"
            )
        if "active_constraint_spec_json" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN active_constraint_spec_json TEXT")
        if "last_runtime_request_json" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN last_runtime_request_json TEXT")
        if "refinement_revision_index" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN refinement_revision_index INTEGER NOT NULL DEFAULT 0"
            )
        if "refinement_rounds_used" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN refinement_rounds_used INTEGER NOT NULL DEFAULT 0"
            )
        if "refinement_round_limit" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN refinement_round_limit INTEGER NOT NULL DEFAULT 3"
            )
        if "pending_refinement_clarification_json" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN pending_refinement_clarification_json TEXT")
        if "latest_runtime_backed_turn_id" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN latest_runtime_backed_turn_id TEXT")
        if "canonical_session_state_json" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN canonical_session_state_json TEXT")
        if "canonical_state_source" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN canonical_state_source TEXT")
        if "canonical_mirror_ok" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN canonical_mirror_ok INTEGER NOT NULL DEFAULT 1"
            )

    def _migrate_turns_table(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"]): row
            for row in connection.execute("PRAGMA table_info(turns)").fetchall()
        }
        if "clarification_turns_used" not in columns:
            connection.execute(
                "ALTER TABLE turns ADD COLUMN clarification_turns_used INTEGER NOT NULL DEFAULT 0"
            )
        if "is_case_complete" not in columns:
            connection.execute(
                "ALTER TABLE turns ADD COLUMN is_case_complete INTEGER NOT NULL DEFAULT 0"
            )
        if "case_completion_reason" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN case_completion_reason TEXT")
        if "restart_required" not in columns:
            connection.execute(
                "ALTER TABLE turns ADD COLUMN restart_required INTEGER NOT NULL DEFAULT 0"
            )
        if "timing_metrics_json" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN timing_metrics_json TEXT")
        if "turn_kind" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN turn_kind TEXT NOT NULL DEFAULT 'message'")
        if "refinement_status" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN refinement_status TEXT")
        if "refinement_revision_index" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN refinement_revision_index INTEGER")
        if "parent_terminal_turn_id" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN parent_terminal_turn_id TEXT")
        if "parent_refinement_revision_index" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN parent_refinement_revision_index INTEGER")
        if "active_constraint_spec_json" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN active_constraint_spec_json TEXT")
        if "constraint_feedback_delta_json" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN constraint_feedback_delta_json TEXT")
        if "refinement_rounds_used" not in columns:
            connection.execute(
                "ALTER TABLE turns ADD COLUMN refinement_rounds_used INTEGER NOT NULL DEFAULT 0"
            )
        if "refinement_round_limit" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN refinement_round_limit INTEGER")
        if "canonical_runtime_result_json" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN canonical_runtime_result_json TEXT")
        if "verification_artifacts_json" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN verification_artifacts_json TEXT")
        if "canonical_session_state_json" not in columns:
            connection.execute("ALTER TABLE turns ADD COLUMN canonical_session_state_json TEXT")


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _json_load(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def parse_sort_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
