from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from scripts.archieve import run_part2_closeout_bundle as runner


def build_args(tmp_path: Path) -> Namespace:
    return Namespace(
        out_dir=tmp_path,
        freeze_out_dir=tmp_path / "freeze",
        baseline_catalog=tmp_path / "catalog.json",
        benchmark=tmp_path / "benchmark.yaml",
        lm_studio_api_base="http://localhost:1234",
        model_alias="stub-model",
        product_mode="stable_demo",
        api_version="v1",
        app_version="phase3_2_test",
        tier_a_corpus=tmp_path / "tier_a.json",
        v1_tier_b_corpus=tmp_path / "tier_b_v1.json",
        v2_tier_b_corpus=tmp_path / "tier_b_v2.json",
        tier_c_corpus=tmp_path / "tier_c.json",
        tier_d_corpus=tmp_path / "tier_d.json",
        v2_g5_corpus=tmp_path / "g5_v2.json",
        g5_backends="ufce,dice,ar",
        g5_attempts_per_case=3,
        g5_case_limit=None,
        golden_parity_waiver=None,
        no_progress=True,
        summary_json=None,
        summary_md=None,
    )


def build_stage_result(*, name: str, passed: bool = True, payload: dict | None = None) -> dict:
    return {
        "name": name,
        "command": f"python {name}.py",
        "exit_code": 0 if passed else 1,
        "passed": passed,
        "payload": payload,
        "parse_error": None if passed else "failed",
        "stdout_tail": [],
        "stderr_tail": [],
        "summary_json_path": f"/tmp/{name}.json",
        "summary_markdown_path": f"/tmp/{name}.md",
    }


def write_report_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "corpus_path": str(path.with_suffix(".corpus.json")),
                "corpus_version": "demo_corpus_v1",
                "corpus_sha256": "demo_hash",
                "loaded_corpora": {"primary": {"corpus_path": str(path.with_suffix('.corpus.json'))}},
            }
        ),
        encoding="utf-8",
    )


def test_extract_archive_and_validate_harness_summaries_requires_all_required_reports(tmp_path):
    child_runs = []
    for stage_name in runner.REQUIRED_HARNESS_SUMMARIES:
        if stage_name == "agent_portability_report":
            continue
        summary_json = tmp_path / f"{stage_name}.json"
        summary_md = tmp_path / f"{stage_name}.md"
        write_report_summary(summary_json)
        summary_md.write_text("# demo\n", encoding="utf-8")
        child_runs.append(
            {
                "name": stage_name,
                "summary_json_path": str(summary_json),
                "summary_markdown_path": str(summary_md),
            }
        )

    extracted = runner.extract_archive_and_validate_harness_summaries(
        harness_payload={"child_runs": child_runs, "closeout_passed": True},
        archive_root=tmp_path / "archive",
    )

    assert extracted["ok"] is False
    assert extracted["summaries"]["portability"]["valid"] is False
    assert extracted["summaries"]["portability"]["error"] == "missing_stage_payload"
    assert extracted["corpus_version_hash_metadata"]["valid"] is False


def test_build_v2_harness_args_forces_full_tier_c(tmp_path):
    args = build_args(tmp_path)

    command_args = runner.build_v2_harness_args(args)

    assert "--tier-b-corpus" in command_args
    assert str(args.v2_tier_b_corpus) in command_args
    assert "--g5-corpus" in command_args
    assert str(args.v2_g5_corpus) in command_args
    assert "--full-tier-c" in command_args


def test_run_closeout_bundle_blocks_when_harness_green_but_required_summaries_invalid(monkeypatch, tmp_path):
    args = build_args(tmp_path)

    def fake_run_command(command, *, name, stream_stderr=False, summary_json_path=None):
        del command
        del stream_stderr
        del summary_json_path
        return build_stage_result(name=name, passed=True, payload=None)

    def fake_run_freeze_step(*, out_dir, summary_root):
        del out_dir
        del summary_root
        return build_stage_result(
            name="freeze_v2_corpora",
            passed=True,
            payload={"written_corpora": {}, "out_dir": str(tmp_path / "freeze")},
        )

    def fake_run_summary_script(*, name, script, out_dir, extra_args, summary_root, no_progress):
        del script
        del out_dir
        del extra_args
        del summary_root
        del no_progress
        if name == "v2_closeout_harness":
            return build_stage_result(
                name=name,
                passed=True,
                payload={"closeout_passed": True, "child_runs": []},
            )
        return build_stage_result(name=name, passed=True, payload={"aggregate_validation": {"ok": True}})

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    monkeypatch.setattr(runner, "run_freeze_step", fake_run_freeze_step)
    monkeypatch.setattr(runner, "run_summary_script", fake_run_summary_script)
    monkeypatch.setattr(
        runner,
        "extract_archive_and_validate_harness_summaries",
        lambda **kwargs: {
            "ok": False,
            "summaries": {},
            "corpus_version_hash_metadata": {"valid": False},
        },
    )
    monkeypatch.setattr(runner, "local_now_compact", lambda: "20260329_130000")
    monkeypatch.setattr(runner, "local_now_iso", lambda: "2026-03-29T13:00:00+07:00")

    summary = runner.run_closeout_bundle(
        args=args,
        command="python scripts/run_part2_closeout_bundle.py",
    )

    assert summary["supporting_evidence_passed"] is True
    assert summary["harness_gate_passed"] is True
    assert summary["closeout_passed"] is False
    assert "required_harness_summaries_invalid" in summary["blocked_reasons"]
