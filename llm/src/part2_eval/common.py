from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm.auto import tqdm

from llm.src.utils.hashing import sha256_text
from llm.src.utils.io import write_json


ROOT = Path(__file__).resolve().parents[3]
REQUEST_TIMEOUT_S = 30.0


@dataclass
class IsolatedServiceHandle:
    process: subprocess.Popen[str]
    base_url: str
    service_command: str
    sqlite_path: Path
    artifact_root: Path
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any


@dataclass
class InProcessServiceHandle:
    config: Any
    repository: Any
    service: Any
    artifact_root: Path
    sqlite_path: Path
    benchmark_path: Path
    execution_mode: str = "in_process_conversational"
    service_command: str | None = None
    base_url: str | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    counterfactual_backend_name: str = "ufce"


def canonical_json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def sha256_json_payload(payload: Any) -> str:
    return sha256_text(canonical_json_text(payload))


def safe_mean(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return round(float(numerator) / float(denominator), 6)


def counter_dict(items) -> dict[str, int]:
    counts = Counter(str(item) for item in items)
    return dict(sorted(counts.items()))


def summarize_numeric_values(values: list[float | int | None]) -> dict[str, float | None]:
    numeric = sorted(float(item) for item in values if isinstance(item, (int, float)))
    if not numeric:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": round(sum(numeric) / len(numeric), 3),
        "p50": round(_percentile(numeric, 0.50), 3),
        "p95": round(_percentile(numeric, 0.95), 3),
        "max": round(numeric[-1], 3),
    }


def summarize_latency_ms(values: list[float | int | None]) -> dict[str, float | None]:
    return summarize_numeric_values(values)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, fraction)) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_runner_command(script_path: Path, argv: list[str] | None = None) -> str:
    command = [sys.executable, str(script_path.resolve()), *((argv or sys.argv[1:]))]
    return " ".join(shlex.quote(item) for item in command)


def render_env_prefixed_command(*, env_overrides: dict[str, str], command: list[str]) -> str:
    rendered_env = [f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())]
    rendered_command = [shlex.quote(item) for item in command]
    return " ".join(rendered_env + rendered_command)


def add_summary_output_args(parser) -> None:
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)


def write_optional_summary_outputs(
    *,
    summary: dict[str, Any],
    summary_json_path: Path | None,
    summary_markdown_path: Path | None,
    markdown_text: str | None = None,
) -> None:
    if summary_json_path is not None:
        summary_json_path = Path(summary_json_path).resolve()
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary_json_path, summary)
    if summary_markdown_path is not None:
        if markdown_text is None:
            raise ValueError("markdown_text is required when summary_markdown_path is provided")
        summary_markdown_path = Path(summary_markdown_path).resolve()
        summary_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        summary_markdown_path.write_text(markdown_text, encoding="utf-8")


def progress_iter(
    iterable,
    *,
    enabled: bool,
    desc: str,
    unit: str,
    total: int | None = None,
):
    if not enabled:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        ascii=True,
        dynamic_ncols=True,
        leave=False,
        file=sys.stderr,
    )


@contextmanager
def redirect_legacy_stdout_to_stderr():
    with redirect_stdout(sys.stderr):
        yield


def call_with_legacy_stdout_redirect(func, /, *args, **kwargs):
    with redirect_legacy_stdout_to_stderr():
        return func(*args, **kwargs)


def pick_free_port(*, host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def tail_lines(path: Path, *, limit: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-limit:]


def prepare_run_layout(*, out_dir: Path, run_id: str) -> dict[str, Path]:
    run_root = Path(out_dir) / run_id
    artifact_root = run_root / "isolated_product_artifacts"
    sqlite_path = run_root / "isolated_sessions.sqlite3"
    stdout_path = run_root / "service_stdout.log"
    stderr_path = run_root / "service_stderr.log"
    run_root.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=False)
    return {
        "run_root": run_root.resolve(),
        "artifact_root": artifact_root.resolve(),
        "sqlite_path": sqlite_path.resolve(),
        "stdout_path": stdout_path.resolve(),
        "stderr_path": stderr_path.resolve(),
    }


def launch_isolated_service(
    *,
    layout: dict[str, Path],
    lm_studio_api_base: str,
    model_alias: str,
    product_mode: str,
    api_version: str,
    app_version: str,
    service_script: Path,
    startup_timeout_s: float,
) -> IsolatedServiceHandle:
    host = "127.0.0.1"
    port = pick_free_port(host=host)
    base_url = f"http://{host}:{port}"
    env_overrides = {
        "LM_STUDIO_API_BASE": lm_studio_api_base.rstrip("/"),
        "MODEL_ALIAS": model_alias,
        "PRODUCT_MODE": product_mode,
        "ARTIFACT_ROOT": str(layout["artifact_root"]),
        "SQLITE_PATH": str(layout["sqlite_path"]),
        "API_VERSION": api_version,
        "APP_VERSION": app_version,
        "HOST": host,
        "PORT": str(port),
    }
    env = os.environ.copy()
    env.update(env_overrides)
    command = [sys.executable, str(Path(service_script).resolve())]
    service_command = render_env_prefixed_command(env_overrides=env_overrides, command=command)
    stdout_handle = layout["stdout_path"].open("w", encoding="utf-8")
    stderr_handle = layout["stderr_path"].open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    handle = IsolatedServiceHandle(
        process=process,
        base_url=base_url,
        service_command=service_command,
        sqlite_path=layout["sqlite_path"],
        artifact_root=layout["artifact_root"],
        stdout_path=layout["stdout_path"],
        stderr_path=layout["stderr_path"],
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )
    try:
        wait_for_service(
            base_url=base_url,
            api_version=api_version,
            process=process,
            stdout_path=layout["stdout_path"],
            stderr_path=layout["stderr_path"],
            timeout_s=startup_timeout_s,
        )
        return handle
    except Exception:
        stop_isolated_service(handle)
        raise


def stop_isolated_service(handle: IsolatedServiceHandle) -> None:
    try:
        if handle.process.poll() is None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=5.0)
    finally:
        handle.stdout_handle.close()
        handle.stderr_handle.close()


def wait_for_service(
    *,
    base_url: str,
    api_version: str,
    process: subprocess.Popen[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "Isolated product service exited before becoming healthy. "
                f"exit_code={return_code}; stdout_tail={tail_lines(stdout_path)}; stderr_tail={tail_lines(stderr_path)}"
            )
        try:
            response = requests.get(f"{base_url}/api/{api_version}/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"http_{response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(
        "Timed out waiting for isolated product service startup. "
        f"last_error={last_error}; stdout_tail={tail_lines(stdout_path)}; stderr_tail={tail_lines(stderr_path)}"
    )


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.request(method, url, json=json, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def get_dataset_entry(payload: list[dict[str, Any]], *, dataset_key: str) -> dict[str, Any]:
    for item in payload:
        if item.get("dataset_key") == dataset_key:
            return dict(item)
    raise KeyError(f"Missing dataset catalog entry: {dataset_key}")


def build_in_process_service(
    *,
    layout: dict[str, Path],
    lm_studio_api_base: str,
    model_alias: str,
    product_mode: str,
    api_version: str,
    app_version: str,
    benchmark_path: Path,
    execution_mode: str = "in_process_conversational",
    counterfactual_backend_name: str = "ufce",
) -> InProcessServiceHandle:
    from llm.src.conversation.orchestrator import BankConversationOrchestrator
    from llm.src.conversation.parser_adapter import DEFAULT_SCHEMA_PATH, LiveLmStudioParserAdapter
    from llm.src.product.config import ProductConfig
    from llm.src.product.persistence import SessionRepository
    from llm.src.product.service import ProductSessionService
    from llm.src.runtime.orchestrator import RuntimeOrchestrator
    from llm.src.runtime.policy_registry import BANK_POLICY_VERSION

    with redirect_legacy_stdout_to_stderr():
        parser_adapter = LiveLmStudioParserAdapter(
            model_alias=model_alias,
            api_base=lm_studio_api_base.rstrip("/"),
            benchmark_path=benchmark_path,
        )
        benchmark = parser_adapter.load_benchmark()
        config = ProductConfig(
            lm_studio_api_base=lm_studio_api_base.rstrip("/"),
            model_alias=model_alias,
            product_mode=product_mode,
            artifact_root=layout["artifact_root"],
            sqlite_path=layout["sqlite_path"],
            api_version=api_version,
            app_version=app_version,
            parser_schema_version=DEFAULT_SCHEMA_PATH.stem,
            bank_policy_version=BANK_POLICY_VERSION,
        )
        repository = SessionRepository(config.sqlite_path, app_version=config.app_version)
        orchestrator = BankConversationOrchestrator(
            parser_adapter=parser_adapter,
            runtime_orchestrator=RuntimeOrchestrator(
                runtime_mode=product_mode,
                counterfactual_backend_name=counterfactual_backend_name,
            ),
            benchmark=benchmark,
            benchmark_path=benchmark_path,
            output_root=config.artifact_root,
            model_alias=model_alias,
        )
        service = ProductSessionService(
            orchestrator=orchestrator,
            repository=repository,
            config=config,
        )
    return InProcessServiceHandle(
        config=config,
        repository=repository,
        service=service,
        artifact_root=layout["artifact_root"],
        sqlite_path=layout["sqlite_path"],
        benchmark_path=Path(benchmark_path).resolve(),
        execution_mode=execution_mode,
        counterfactual_backend_name=counterfactual_backend_name,
    )


def build_session_detail_payload(stored_session) -> dict[str, Any]:
    return {
        "session_id": stored_session.session_id,
        "current_public_state": stored_session.current_public_state,
        "clarification_turns_used": stored_session.clarification_turns_used,
        "is_case_complete": stored_session.is_case_complete,
        "case_completion_reason": stored_session.case_completion_reason,
        "restart_required": stored_session.restart_required,
        "active_constraint_spec": stored_session.active_constraint_spec_json or {},
        "last_runtime_request": stored_session.last_runtime_request_json,
        "refinement_revision_index": stored_session.refinement_revision_index,
        "refinement_rounds_used": stored_session.refinement_rounds_used,
        "refinement_round_limit": stored_session.refinement_round_limit,
        "pending_refinement_clarification": stored_session.pending_refinement_clarification_json,
        "latest_runtime_backed_turn_id": stored_session.latest_runtime_backed_turn_id,
    }


def replay_scripted_session_case(*, handle, case: dict[str, Any]) -> dict[str, Any]:
    from llm.src.product.service import SessionCaseCompleteError

    created = handle.service.create_session()
    session_id = str(created.session_id)
    turns = [str(item) for item in case.get("turns", [])]
    turn_payloads: list[dict[str, Any]] = []
    script_execution_status = "completed"
    failed_turn_index = None
    script_mismatch_reason = None
    premature_terminal_state = None
    premature_case_completion_reason = None

    for turn_index, turn_text in enumerate(turns, start=1):
        if turn_payloads and bool(turn_payloads[-1].get("is_case_complete")):
            script_execution_status = "script_mismatch"
            failed_turn_index = turn_index
            script_mismatch_reason = "premature_case_completion"
            premature_terminal_state = turn_payloads[-1].get("public_state")
            premature_case_completion_reason = turn_payloads[-1].get("case_completion_reason")
            break
        try:
            stored_turn = call_with_legacy_stdout_redirect(handle.service.submit_message, session_id, turn_text)
        except SessionCaseCompleteError as exc:
            session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
            script_execution_status = "script_mismatch"
            failed_turn_index = turn_index
            script_mismatch_reason = "session_case_complete_error"
            premature_terminal_state = session_detail.get("current_public_state")
            premature_case_completion_reason = session_detail.get("case_completion_reason") or str(exc)
            break
        turn_payload = handle.service.build_turn_response(stored_turn)
        turn_payloads.append(turn_payload)
        if bool(turn_payload.get("is_case_complete")) and turn_index < len(turns):
            script_execution_status = "script_mismatch"
            failed_turn_index = turn_index + 1
            script_mismatch_reason = "premature_case_completion"
            premature_terminal_state = turn_payload.get("public_state")
            premature_case_completion_reason = turn_payload.get("case_completion_reason")
            break

    if script_execution_status == "completed" and turn_payloads:
        final_turn = turn_payloads[-1]
        if not bool(final_turn.get("is_case_complete")):
            script_execution_status = "script_mismatch"
            failed_turn_index = len(turns)
            script_mismatch_reason = "script_exhausted_non_terminal"
            premature_terminal_state = final_turn.get("public_state")
            premature_case_completion_reason = final_turn.get("case_completion_reason")

    session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
    return {
        "session_id": session_id,
        "turn_payloads": turn_payloads,
        "session_detail": session_detail,
        "scripted_turn_count": len(turns),
        "executed_turn_count": len(turn_payloads),
        "script_execution_status": script_execution_status,
        "failed_turn_index": failed_turn_index,
        "script_mismatch_reason": script_mismatch_reason,
        "premature_terminal_state": premature_terminal_state,
        "premature_case_completion_reason": premature_case_completion_reason,
    }


def build_script_mismatch_summary(
    rows: list[dict[str, Any]],
    *,
    identifier_builder=None,
) -> dict[str, Any]:
    mismatches = [row for row in rows if row.get("script_execution_status", "completed") != "completed"]
    identifiers: list[str] = []
    for index, row in enumerate(mismatches, start=1):
        if identifier_builder is not None:
            identifier = identifier_builder(row)
        else:
            identifier = row.get("case_id") or row.get("session_id") or f"row_{index}"
        identifiers.append(str(identifier))
    return {
        "count": len(mismatches),
        "case_identifiers": identifiers,
        "status_counts": counter_dict(row.get("script_execution_status") for row in mismatches),
        "reason_counts": counter_dict(row.get("script_mismatch_reason") for row in mismatches if row.get("script_mismatch_reason")),
    }


def apply_script_mismatch_validation(
    validation: dict[str, Any],
    *,
    script_mismatch_summary: dict[str, Any],
    path: str = "script_mismatch_summary.count",
) -> dict[str, Any]:
    if int(script_mismatch_summary.get("count") or 0) <= 0:
        return validation
    updated = dict(validation)
    differences = list(updated.get("differences") or [])
    differences.append(
        {
            "path": path,
            "expected": 0,
            "recomputed": int(script_mismatch_summary.get("count") or 0),
        }
    )
    updated["differences"] = differences
    updated["difference_count"] = len(differences)
    updated["ok"] = False
    return updated


def lm_studio_preflight(*, api_base: str, model_alias: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/models"
    try:
        response = requests.get(url, timeout=10.0)
    except Exception as exc:
        return {
            "ok": False,
            "detail": str(exc),
            "api_base": api_base.rstrip("/"),
            "model_alias": model_alias,
            "model_alias_present": False,
            "available_models": [],
        }
    if response.status_code >= 400:
        return {
            "ok": False,
            "detail": f"http_{response.status_code}",
            "api_base": api_base.rstrip("/"),
            "model_alias": model_alias,
            "model_alias_present": False,
            "available_models": [],
        }
    try:
        payload = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "detail": f"invalid_json:{type(exc).__name__}",
            "api_base": api_base.rstrip("/"),
            "model_alias": model_alias,
            "model_alias_present": False,
            "available_models": [],
        }
    data = payload.get("data") if isinstance(payload, dict) else None
    available_models = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                available_models.append(str(item["id"]))
    model_alias_present = not available_models or model_alias in available_models
    return {
        "ok": model_alias_present,
        "detail": "reachable" if model_alias_present else f"model_alias_missing:{model_alias}",
        "api_base": api_base.rstrip("/"),
        "model_alias": model_alias,
        "model_alias_present": model_alias_present,
        "available_models": available_models,
    }


def recompute_and_validate_aggregates(
    *,
    expected_blocks: dict[str, Any],
    recomputed_blocks: dict[str, Any],
    max_differences: int = 100,
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    _collect_differences(
        expected=expected_blocks,
        recomputed=recomputed_blocks,
        path="",
        differences=differences,
        max_differences=max_differences,
    )
    return {
        "ok": not differences,
        "difference_count": len(differences),
        "differences": differences,
        "validated_aggregates": recomputed_blocks,
    }


def _collect_differences(
    *,
    expected: Any,
    recomputed: Any,
    path: str,
    differences: list[dict[str, Any]],
    max_differences: int,
) -> None:
    if len(differences) >= max_differences:
        return
    if isinstance(expected, dict) and isinstance(recomputed, dict):
        keys = sorted(set(expected.keys()) | set(recomputed.keys()))
        for key in keys:
            child_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append({"path": child_path, "expected": "__missing__", "recomputed": recomputed[key]})
                if len(differences) >= max_differences:
                    return
                continue
            if key not in recomputed:
                differences.append({"path": child_path, "expected": expected[key], "recomputed": "__missing__"})
                if len(differences) >= max_differences:
                    return
                continue
            _collect_differences(
                expected=expected[key],
                recomputed=recomputed[key],
                path=child_path,
                differences=differences,
                max_differences=max_differences,
            )
        return
    if isinstance(expected, list) and isinstance(recomputed, list):
        if expected != recomputed:
            differences.append({"path": path or "__root__", "expected": expected, "recomputed": recomputed})
        return
    if expected != recomputed:
        differences.append({"path": path or "__root__", "expected": expected, "recomputed": recomputed})
