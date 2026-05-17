from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = Path(__file__).resolve().parent
DATASETS = ("grad", "bank", "wine", "bupa", "movie")
DEFAULT_TAIL_LINES = 40


def ensure_repo_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def local_now() -> datetime:
    return datetime.now().astimezone()


def local_timestamp() -> str:
    return local_now().isoformat(timespec="seconds")


def local_timezone() -> str:
    now = local_now()
    return now.tzname() or str(now.utcoffset())


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{local_now().strftime('%Y%m%d_%H%M%S_%f')}"


def make_run_dir(out_dir: Path | str, prefix: str) -> tuple[str, Path]:
    run_id = make_run_id(prefix)
    run_root = Path(out_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    return run_id, run_root.resolve()


def render_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def current_command() -> str:
    return render_command([sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])


def capture_command_line() -> str:
    return current_command()


def repo_rel(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def capture_git_commit() -> str | None:
    return git_commit()


def git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return [f"git status failed: {tail(result.stderr, 1)[0] if result.stderr else 'unknown error'}"]
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def tail(text: str, limit: int = DEFAULT_TAIL_LINES) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()][-limit:]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_provenance_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_provenance_markdown(path: Path, markdown: str) -> None:
    write_text(path, markdown)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    env_path = ROOT / ".env" if path is None else Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    log_dir: Path | None = None,
    log_prefix: str = "command",
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = None
    stderr_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = (log_dir / f"{log_prefix}.stdout.log").resolve()
        stderr_path = (log_dir / f"{log_prefix}.stderr.log").resolve()
        stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    return {
        "command": render_command(command),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
        "stdout_path": None if stdout_path is None else str(stdout_path),
        "stderr_path": None if stderr_path is None else str(stderr_path),
    }


def expand_datasets(value: str) -> list[str]:
    cleaned = value.strip().lower()
    if cleaned == "all":
        return list(DATASETS)
    parts = [item.strip().lower() for item in cleaned.split(",") if item.strip()]
    unknown = [item for item in parts if item not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    return parts


def status_from_results(results: Iterable[dict[str, Any]], warnings: Sequence[str] | None = None) -> str:
    result_list = list(results)
    if any(not item.get("passed", False) for item in result_list):
        return "failed"
    if warnings:
        return "warning"
    return "passed"


def provenance(
    *,
    run_id: str,
    run_root: Path,
    title: str,
    input_paths: Sequence[Path | str] = (),
    commands: Sequence[str] = (),
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    commit = git_commit()
    return {
        "title": title,
        "run_id": run_id,
        "timestamp_local": local_timestamp(),
        "timezone": local_timezone(),
        "git_commit": commit,
        "git_status_short": git_status_short(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "final_script": repo_rel(Path(sys.argv[0])),
        "command_line": current_command(),
        "delegated_commands": list(commands),
        "input_paths": [str(Path(path)) for path in input_paths],
        "output_root": str(run_root),
        "summary_json_path": str((run_root / "summary.json").resolve()),
        "summary_markdown_path": str((run_root / "summary.md").resolve()),
        "warnings": list(warnings),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary.get('title', 'Final Wrapper Summary')}",
        "",
        "## Status",
        f"- status: `{summary.get('status')}`",
        f"- passed: `{summary.get('passed')}`",
        f"- run_id: `{summary.get('run_id')}`",
        f"- timestamp_local: `{summary.get('timestamp_local')}`",
        f"- timezone: `{summary.get('timezone')}`",
        f"- git_commit: `{summary.get('git_commit')}`",
        f"- output_root: `{summary.get('output_root')}`",
        "",
        "## Commands",
        f"- final: `{summary.get('command_line')}`",
    ]
    for command in summary.get("delegated_commands", []):
        lines.append(f"- delegated: `{command}`")
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    results = summary.get("results") or summary.get("stage_results") or []
    if results:
        lines.extend(["", "## Results"])
        for result in results:
            name = result.get("name") or result.get("dataset") or result.get("stage") or "command"
            lines.append(
                f"- {name}: passed=`{result.get('passed')}` exit_code=`{result.get('exit_code')}`"
            )
            if result.get("summary_json_path"):
                lines.append(f"  - summary_json_path: `{result.get('summary_json_path')}`")
            if result.get("summary_markdown_path"):
                lines.append(f"  - summary_markdown_path: `{result.get('summary_markdown_path')}`")
    extra = summary.get("markdown_notes") or []
    if extra:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {item}" for item in extra)
    return "\n".join(lines) + "\n"


def write_standard_outputs(
    *,
    run_root: Path,
    summary: dict[str, Any],
    markdown: str | None = None,
) -> None:
    commands = [summary.get("command_line", ""), *summary.get("delegated_commands", [])]
    command_text = "\n".join(command for command in commands if command) + "\n"
    write_text(run_root / "command.txt", command_text)
    write_text(run_root / "git_commit.txt", (summary.get("git_commit") or "unknown") + "\n")
    write_json(run_root / "summary.json", summary)
    write_text(run_root / "summary.md", markdown if markdown is not None else render_summary_markdown(summary))


def summarize_wrapper_run(
    *,
    title: str,
    run_id: str,
    run_root: Path,
    results: Sequence[dict[str, Any]],
    input_paths: Sequence[Path | str] = (),
    warnings: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commands = [result["command"] for result in results if "command" in result]
    summary = provenance(
        run_id=run_id,
        run_root=run_root,
        title=title,
        input_paths=input_paths,
        commands=commands,
        warnings=warnings,
    )
    status = status_from_results(results, warnings)
    summary.update(
        {
            "status": status,
            "passed": status == "passed",
            "results": list(results),
        }
    )
    if extra:
        summary.update(extra)
    write_standard_outputs(run_root=run_root, summary=summary)
    return summary

