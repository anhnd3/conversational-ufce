from __future__ import annotations

from pathlib import Path

from scripts.run_phase3_2_acceptance_report import build_acceptance_summary, render_markdown


def test_build_acceptance_summary_tracks_failed_gates(tmp_path):
    summary = build_acceptance_summary(
        run_id="phase3_2_acceptance_demo",
        base_url="http://127.0.0.1:8000",
        sqlite_path=tmp_path / "sessions.sqlite3",
        catalog_path=Path("docs/validation/catalogs/phase3_2_validation_catalog_v1.json"),
        smoke_suite={"passed": True, "summary": {"catalog_version": "phase3_2_validation_catalog_v1"}},
        repro_suite={"passed": False},
        metrics_suite={"passed": True, "summary": {"report_json_path": "/tmp/report.json", "report_markdown_path": "/tmp/report.md"}},
        product_checks={"passed": True},
        constraint_checks={"passed": True},
        docs_checks={
            "claim_to_evidence_map_exists": True,
            "thesis_alignment_notes_exists": True,
        },
        manual_evidence={"manual_evidence_recorded": False},
    )

    assert summary["acceptance_passed"] is False
    assert "reproducibility_probe_passed" in summary["failed_gates"]
    assert "manual_ui_evidence_recorded" not in summary["failed_gates"]
    assert summary["manual_evidence_policy"] == "informational_only"


def test_render_markdown_includes_acceptance_sections():
    summary = {
        "run_id": "phase3_2_acceptance_demo",
        "timestamp_local": "2026-03-22T18:00:00+07:00",
        "timezone": "UTC+07:00",
        "base_url": "http://127.0.0.1:8000",
        "sqlite_path": "/tmp/sessions.sqlite3",
        "catalog_path": "docs/validation/catalogs/phase3_2_validation_catalog_v1.json",
        "catalog_version": "phase3_2_validation_catalog_v1",
        "git_commit": None,
        "smoke_suite": {"command": "python smoke.py", "passed": True, "exit_code": 0},
        "repro_suite": {"command": "python repro.py", "passed": True, "exit_code": 0},
        "metrics_suite": {
            "command": "python metrics.py",
            "passed": True,
            "exit_code": 0,
            "summary": {
                "report_json_path": "/tmp/report.json",
                "report_markdown_path": "/tmp/report.md",
            },
        },
        "product_checks": {"checks": {"artifact_preview_ok": True}},
        "constraint_checks": {"checks": {"request_constraints_blocked_distinct": True}},
        "docs_checks": {
            "claim_to_evidence_map_exists": True,
            "thesis_alignment_notes_exists": True,
        },
        "manual_evidence": {
            "manual_session_id": "session_demo",
            "manual_session_present_in_db": True,
            "manual_evidence_recorded": True,
        },
        "manual_evidence_policy": "informational_only",
        "acceptance_gates": {
            "phase3_2_acceptance_report_formalized": True,
        },
        "acceptance_passed": True,
        "failed_gates": [],
    }

    markdown = render_markdown(summary)

    assert "# Phase 3.2 Acceptance Report" in markdown
    assert "## Product Smoke" in markdown
    assert "## Product Metrics" in markdown
    assert "## Constraint Verification" in markdown
    assert "manual_evidence_policy" in markdown
    assert "## Acceptance Verdict" in markdown
