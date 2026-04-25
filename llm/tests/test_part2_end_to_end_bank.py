from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from llm.src.part2_eval.corpora import (
    G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH,
    TIER_B_SYNTH300_CORPUS_PATH,
)
from scripts import run_part2_end_to_end_bank as runner


def build_args(tmp_path: Path) -> Namespace:
    return Namespace(
        out_dir=tmp_path,
        baseline_catalog=tmp_path / "catalog.json",
        benchmark=tmp_path / "benchmark.yaml",
        lm_studio_api_base="http://localhost:1234",
        model_alias="stub-model",
        product_mode="stable_demo",
        api_version="v1",
        app_version="phase3_2_test",
        tier_a_corpus=tmp_path / "tier_a.json",
        tier_b_corpus=tmp_path / "tier_b.json",
        tier_c_corpus=tmp_path / "tier_c.json",
        tier_d_corpus=tmp_path / "tier_d.json",
        g5_corpus=tmp_path / "g5.json",
        g5_backends="ufce,dice,ar",
        g5_attempts_per_case=3,
        g5_case_limit=None,
        backend_seed_id="TIERC-001",
        backend_seed_index=None,
        full_tier_c=False,
        no_progress=False,
    )


def build_child_result(*, name: str, aggregate_ok: bool = True) -> dict:
    return {
        "name": name,
        "command": f"python {name}.py",
        "exit_code": 0,
        "passed": True,
        "payload": {
            "aggregate_validation": {"ok": aggregate_ok, "difference_count": 0},
            "corpus_path": f"/tmp/{name}.corpus.json",
            "report_json_path": f"/tmp/{name}.json",
            "report_markdown_path": f"/tmp/{name}.md",
            "corpus_version": f"{name}_corpus_v1",
            "corpus_sha256": f"{name}_hash",
            "loaded_corpora": {"primary": {"corpus_path": f"/tmp/{name}.corpus.json"}},
        },
        "parse_error": None,
        "stdout_tail": [],
        "stderr_tail": [],
    }


def test_run_closeout_executes_serverless_stages_in_locked_order(monkeypatch, tmp_path):
    call_order: list[str] = []

    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        del command
        del stream_stderr
        del summary_json_path
        call_order.append("targeted_pytest")
        return {
            "name": name,
            "command": "python -m pytest",
            "exit_code": 0,
            "passed": True,
            "payload": None,
            "parse_error": None,
            "stdout_tail": [],
            "stderr_tail": [],
        }

    def fake_preflight(*, api_base, model_alias):
        call_order.append("lm_studio_preflight")
        return {
            "ok": True,
            "detail": "reachable",
            "api_base": api_base,
            "model_alias": model_alias,
            "model_alias_present": True,
            "available_models": [model_alias],
        }

    def fake_run_script_step(*, name, script, out_dir, args):
        del script
        del out_dir
        del args
        call_order.append(name)
        return build_child_result(name=name)

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    monkeypatch.setattr(runner, "lm_studio_preflight", fake_preflight)
    monkeypatch.setattr(runner, "run_script_step", fake_run_script_step)
    monkeypatch.setattr(runner, "local_now_compact", lambda: "20260325_120000")
    monkeypatch.setattr(runner, "local_now_iso", lambda: "2026-03-25T12:00:00+07:00")

    summary = runner.run_closeout(args=build_args(tmp_path), command="python run_part2_end_to_end_bank.py")

    assert call_order == [
        "targeted_pytest",
        "lm_studio_preflight",
        "thesis_metrics_report",
        "refinement_metrics_report",
        "backend_comparison_report",
        "agent_portability_report",
        "replay_robustness_report",
    ]
    assert summary["closeout_passed"] is True
    assert [item["name"] for item in summary["report_validations"]] == [
        "thesis_metrics_report",
        "refinement_metrics_report",
        "backend_comparison_report",
        "agent_portability_report",
        "replay_robustness_report",
    ]
    assert (tmp_path / "part2_closeout_20260325_120000" / "part2_closeout_summary.json").exists()


def test_run_closeout_fails_when_report_aggregate_validation_fails(monkeypatch, tmp_path):
    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        del command
        del stream_stderr
        del summary_json_path
        return {
            "name": name,
            "command": "python -m pytest",
            "exit_code": 0,
            "passed": True,
            "payload": None,
            "parse_error": None,
            "stdout_tail": [],
            "stderr_tail": [],
        }

    def fake_preflight(*, api_base, model_alias):
        return {
            "ok": True,
            "detail": "reachable",
            "api_base": api_base,
            "model_alias": model_alias,
            "model_alias_present": True,
            "available_models": [model_alias],
        }

    def fake_run_script_step(*, name, script, out_dir, args):
        del script
        del out_dir
        del args
        return build_child_result(name=name, aggregate_ok=(name != "backend_comparison_report"))

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    monkeypatch.setattr(runner, "lm_studio_preflight", fake_preflight)
    monkeypatch.setattr(runner, "run_script_step", fake_run_script_step)
    monkeypatch.setattr(runner, "local_now_compact", lambda: "20260325_120100")
    monkeypatch.setattr(runner, "local_now_iso", lambda: "2026-03-25T12:01:00+07:00")

    summary = runner.run_closeout(args=build_args(tmp_path), command="python run_part2_end_to_end_bank.py")

    assert summary["closeout_passed"] is False
    failing_validation = next(item for item in summary["report_validations"] if item["name"] == "backend_comparison_report")
    assert failing_validation["aggregate_validation_ok"] is False


def test_run_command_loads_summary_json_file_while_streaming_stderr_progress(tmp_path):
    summary_json = tmp_path / "summary.json"
    summary_json.write_text('{"ok": true}', encoding="utf-8")
    result = runner.run_command(
        [
            runner.sys.executable,
            "-c",
            (
                "import sys;"
                "sys.stderr.write('progress 1/2\\n');"
                "print('noisy stdout');"
            ),
        ],
        name="demo",
        stream_stderr=True,
        summary_json_path=summary_json,
    )

    assert result["passed"] is True
    assert result["payload"] == {"ok": True}
    assert result["stderr_tail"] == []


def test_run_command_fails_when_summary_json_file_is_missing(tmp_path):
    result = runner.run_command(
        [runner.sys.executable, "-c", "print('done')"],
        name="demo",
        summary_json_path=tmp_path / "missing.json",
    )

    assert result["passed"] is False
    assert "No such file or directory" in result["parse_error"]


def test_run_script_step_defaults_backend_to_single_case_seed(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        captured["command"] = command
        captured["name"] = name
        captured["stream_stderr"] = stream_stderr
        captured["summary_json_path"] = summary_json_path
        return build_child_result(name=name or "backend_comparison_report")

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    runner.run_script_step(
        name="backend_comparison_report",
        script=runner.ROOT / "scripts" / "run_part2_backend_comparison_report.py",
        out_dir=tmp_path / "backend",
        args=build_args(tmp_path),
    )

    command = captured["command"]
    assert "--single-case" in command
    assert "--seed-id" in command
    assert "TIERC-001" in command
    assert "--summary-json" in command
    assert "--summary-md" in command
    assert str(captured["summary_json_path"]).endswith("backend_comparison_report_summary.json")


def test_run_script_step_passes_g5_portability_arguments(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    args = build_args(tmp_path)
    args.g5_case_limit = 12

    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        captured["command"] = command
        captured["name"] = name
        captured["stream_stderr"] = stream_stderr
        captured["summary_json_path"] = summary_json_path
        return build_child_result(name=name or "agent_portability_report")

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    runner.run_script_step(
        name="agent_portability_report",
        script=runner.ROOT / "scripts" / "run_part2_agent_portability_report.py",
        out_dir=tmp_path / "agent_portability",
        args=args,
    )

    command = captured["command"]
    assert "--g5-corpus" in command
    assert str(args.g5_corpus) in command
    assert "--backends" in command
    assert "ufce,dice,ar" in command
    assert "--attempts-per-case" in command
    assert "3" in command
    assert "--case-limit" in command
    assert "12" in command


def test_run_script_step_passes_custom_tier_b_override_to_thesis_and_refinement(monkeypatch, tmp_path):
    captured_commands: dict[str, list[str]] = {}
    args = build_args(tmp_path)
    args.tier_b_corpus = TIER_B_SYNTH300_CORPUS_PATH

    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        del stream_stderr
        del summary_json_path
        if name is None:
            raise AssertionError("expected named command")
        captured_commands[name] = command
        return build_child_result(name=name)

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    runner.run_script_step(
        name="thesis_metrics_report",
        script=runner.ROOT / "scripts" / "run_part2_thesis_metrics_report.py",
        out_dir=tmp_path / "thesis",
        args=args,
    )
    runner.run_script_step(
        name="refinement_metrics_report",
        script=runner.ROOT / "scripts" / "run_part2_refinement_metrics_report.py",
        out_dir=tmp_path / "refinement",
        args=args,
    )

    thesis_command = captured_commands["thesis_metrics_report"]
    refinement_command = captured_commands["refinement_metrics_report"]
    assert "--tier-b-corpus" in thesis_command
    assert str(TIER_B_SYNTH300_CORPUS_PATH) in thesis_command
    assert "--tier-b-corpus" in refinement_command
    assert str(TIER_B_SYNTH300_CORPUS_PATH) in refinement_command


def test_run_script_step_passes_custom_g5_override(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    args = build_args(tmp_path)
    args.g5_corpus = G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH

    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        captured["command"] = command
        captured["name"] = name
        captured["stream_stderr"] = stream_stderr
        captured["summary_json_path"] = summary_json_path
        return build_child_result(name=name or "agent_portability_report")

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    runner.run_script_step(
        name="agent_portability_report",
        script=runner.ROOT / "scripts" / "run_part2_agent_portability_report.py",
        out_dir=tmp_path / "agent_portability",
        args=args,
    )

    command = captured["command"]
    assert "--g5-corpus" in command
    assert str(G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH) in command


def test_run_closeout_aborts_after_failed_preflight(monkeypatch, tmp_path):
    call_order: list[str] = []

    def fake_run_command(command, *, name=None, stream_stderr=False, summary_json_path=None):
        del command
        del stream_stderr
        del summary_json_path
        call_order.append("targeted_pytest")
        return {
            "name": name,
            "command": "python -m pytest",
            "exit_code": 0,
            "passed": True,
            "payload": None,
            "parse_error": None,
            "stdout_tail": [],
            "stderr_tail": [],
        }

    def fake_preflight(*, api_base, model_alias):
        del api_base
        del model_alias
        call_order.append("lm_studio_preflight")
        return {
            "ok": False,
            "detail": "connection refused",
            "api_base": "http://localhost:1234",
            "model_alias": "stub-model",
            "model_alias_present": None,
            "available_models": [],
        }

    def fail_run_script_step(**kwargs):
        raise AssertionError(f"stage should not run after failed preflight: {kwargs['name']}")

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    monkeypatch.setattr(runner, "lm_studio_preflight", fake_preflight)
    monkeypatch.setattr(runner, "run_script_step", fail_run_script_step)
    monkeypatch.setattr(runner, "local_now_compact", lambda: "20260325_120200")
    monkeypatch.setattr(runner, "local_now_iso", lambda: "2026-03-25T12:02:00+07:00")

    summary = runner.run_closeout(args=build_args(tmp_path), command="python run_part2_end_to_end_bank.py")

    assert call_order == ["targeted_pytest", "lm_studio_preflight"]
    assert summary["closeout_passed"] is False
    assert summary["child_runs"][0]["name"] == "targeted_pytest"
