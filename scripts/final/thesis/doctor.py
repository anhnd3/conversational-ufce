#!/usr/bin/env python3
"""
Environment doctor for thesis final runs.
Checks Python version, dependencies, LM Studio connectivity, data/model manifests, git state, and output writability.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_python_version() -> Dict[str, Any]:
    """Check Python version meets minimum requirements."""
    # R11: Python 3.8 compatibility (thesis requirement)
    min_major, min_minor = 3, 8

    actual_major = sys.version_info.major
    actual_minor = sys.version_info.minor

    passed = (actual_major > min_major or
              (actual_major == min_major and actual_minor >= min_minor))

    return {
        "name": "python_version",
        "expected": f">= {min_major}.{min_minor}",
        "actual": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "passed": passed,
    }


def check_requirements_txt() -> Dict[str, Any]:
    """Check requirements.txt exists."""
    path = ROOT / "requirements.txt"
    return {
        "name": "requirements_txt",
        "path": str(path.resolve()),
        "passed": path.exists(),
    }


def check_env_files() -> Dict[str, Any]:
    """Check .env or .env.example exists."""
    env_example = ROOT / ".env.example"
    env_actual = ROOT / ".env"

    passed = env_example.exists() or env_actual.exists()

    return {
        "name": "env_files",
        "example_path": str(env_example.resolve()),
        "actual_path": str(env_actual.resolve()),
        "has_example": env_example.exists(),
        "has_env": env_actual.exists(),
        "passed": passed,
    }


def check_lm_studio(base_url: str = None) -> Dict[str, Any]:
    """Check LM Studio API is reachable if --check-lm-studio is enabled."""
    url = base_url or "http://127.0.0.1:1234"

    try:
        # Try HTTP request to /v1/models (common endpoint)
        import urllib.request

        req = urllib.request.Request(
            f"{url}/v1/models",
            method="GET",
            headers={"Accept": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            models_available = len(body.get("data", [])) > 0

        return {
            "name": "lm_studio_connectivity",
            "url": url,
            "passed": True,
            "models_available": models_available,
            "model_list": [m.get("id") for m in body.get("data", [])],
        }
    except Exception as exc:
        return {
            "name": "lm_studio_connectivity",
            "url": url,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_model_alias() -> Dict[str, Any]:
    """Check configured model alias exists in LM Studio."""
    # Load from .env or default
    import os

    model_alias = os.getenv("MODEL_ALIAS", "qwen/qwen3-14b")

    # Try to fetch models and verify alias is present
    try:
        import urllib.request

        resp = urllib.request.urlopen(f"http://127.0.0.1:1234/v1/models", timeout=5.0)
        body = json.loads(resp.read().decode("utf-8"))

        available_ids = [m.get("id") for m in body.get("data", [])]
        passed = model_alias in available_ids

        return {
            "name": "model_alias_present",
            "expected_alias": model_alias,
            "available_models": available_ids[:5],  # First 5 only
            "passed": passed,
        }
    except Exception:
        return {
            "name": "model_alias_present",
            "expected_alias": model_alias,
            "passed": False,
            "error": "Could not connect to LM Studio or parse response",
        }


def check_required_directories() -> Dict[str, Any]:
    """Check required directories exist."""
    dirs_to_check = ["scripts", "ufce", "llm", "llm_eval", "docs"]

    missing = [d for d in dirs_to_check if not (ROOT / d).exists()]

    return {
        "name": "required_directories",
        "checked": dirs_to_check,
        "missing": missing,
        "passed": len(missing) == 0,
    }


def check_data_model_manifests() -> Dict[str, Any]:
    """Check data/model manifest files exist."""
    checks = {
        "ufce/data/bank.csv": (ROOT / "ufce" / "data" / "bank.csv").exists(),
        "ufce/model_bundles/": (ROOT / "ufce" / "model_bundles").is_dir() if (ROOT / "ufce" / "model_bundles").exists() else False,
    }

    missing = [k for k, v in checks.items() if not v]

    return {
        "name": "data_model_manifests",
        "checked_files": list(checks.keys()),
        "missing": missing,
        "passed": len(missing) == 0,
    }


def check_git_commit() -> Dict[str, Any]:
    """Detect current git commit."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            commit_hash = result.stdout.strip()

            # Also check working tree status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            working_tree_clean = not status_result.stdout.strip()

            return {
                "name": "git_commit",
                "commit_hash": commit_hash,
                "working_tree_clean": working_tree_clean,
                "passed": True,
            }
    except Exception as exc:
        return {
            "name": "git_commit",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_output_writable() -> Dict[str, Any]:
    """Check output root is writable."""
    test_path = ROOT / "outputs" / ".doctor_test_write"

    try:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text("test", encoding="utf-8")
        passed = True
    except Exception as exc:
        passed = False

    finally:
        # Cleanup test file
        if test_path.exists():
            try:
                test_path.unlink()
            except Exception:
                pass

    return {
        "name": "output_writable",
        "path": str((ROOT / "outputs").resolve()),
        "passed": passed,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Environment doctor for thesis final runs.")
    parser.add_argument("--check-lm-studio", action="store_true")

    args = parser.parse_args()

    checks = [
        check_python_version(),
        check_requirements_txt(),
        check_env_files(),
        check_required_directories(),
        check_data_model_manifests(),
        check_git_commit(),
        check_output_writable(),
    ]

    if args.check_lm_studio:
        lm_result = check_lm_studio()
        checks.append(lm_result)

        if lm_result["passed"]:
            alias_result = check_model_alias()
            checks.append(alias_result)

    all_passed = all(c["passed"] for c in checks)

    # Get local timezone safely (Python stdlib does not have timezone.local())
    try:
        tz_info = datetime.now().astimezone().tzinfo
        tz_str = str(tz_info) if tz_info is not None else "UTC"
    except Exception:
        tz_str = "unknown"

    summary = {
        "timestamp_local": datetime.now().isoformat(),
        "timezone": tz_str,
        "runner_scope": "doctor",
        "checks": checks,
        "all_passed": all_passed,
    }

    # Write to outputs/doctor/<run_id>/
    run_root = ROOT / "outputs" / "final" / "doctor" / f"doctor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    with open(run_root / "doctor_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    md_lines = ["# Environment Doctor Report", "",
                f"- **timestamp_local:** `{summary['timestamp_local']}`\n"]

    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        name = check.get("name", "unknown")
        md_lines.append(f"- **{name}**: `{status}`")

    md_lines.extend([
        "", "---", f"**Overall:** `{'PASS' if all_passed else 'FAIL'}`", ""
    ])

    with open(run_root / "doctor_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    # Print to stdout too
    print(json.dumps(summary, indent=2))

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
