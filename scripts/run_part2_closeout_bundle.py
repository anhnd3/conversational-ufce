#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import (
    add_summary_output_args,
    build_runner_command,
    progress_iter,
    write_optional_summary_outputs,
)
from llm.src.part2_eval.corpora import (
    BANK_BOUNDARY_PROFILES_CORPUS_PATH,
    G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH,
    TIER_B_CORPUS_PATH,
    TIER_B_SYNTH300_CORPUS_PATH,
    TIER_C_CORPUS_PATH,
    TIER_D_CORPUS_PATH,
)
from llm.src.product.config import ProductConfig
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso
from scripts.freeze_part2_bank_synth_corpora import freeze_corpora as freeze_bank_synth_corpora


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_closeout_bundle"
RUNNER_SCOPE = "part2_closeout_bundle"
SCORER_VERSION = "part2_closeout_bundle_v1"
REQUIRED_HARNESS_SUMMARIES = {
    "thesis_metrics_report": "thesis metrics",
    "refinement_metrics_report": "refinement metrics",
    "backend_comparison_report": "backend comparison",
    "replay_robustness_report": "replay robustness",
    "agent_portability_report": "portability",
}


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Run the repo-aligned Part II closeout bundle.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze-out-dir", type=Path, default=TIER_B_SYNTH300_CORPUS_PATH.parent)
    parser.add_argument("--baseline-catalog", type=Path, default=ROOT / "docs" / "validation" / "catalogs" / "phase3_2_validation_catalog_v1.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml")
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--tier-a-corpus", type=Path, default=ROOT / "docs" / "validation" / "corpora" / "part2_tier_a_bank_annotations_v1.json")
    parser.add_argument("--v1-tier-b-corpus", type=Path, default=TIER_B_CORPUS_PATH)
    parser.add_argument("--v2-tier-b-corpus", type=Path, default=TIER_B_SYNTH300_CORPUS_PATH)
    parser.add_argument("--tier-c-corpus", type=Path, default=TIER_C_CORPUS_PATH)
    parser.add_argument("--tier-d-corpus", type=Path, default=TIER_D_CORPUS_PATH)
    parser.add_argument("--v2-g5-corpus", type=Path, default=G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH)
    parser.add_argument("--g5-backends", default="ufce,dice,ar")
    parser.add_argument("--g5-attempts-per-case", type=int, default=3)
    parser.add_argument("--g5-case-limit", type=int, default=None)
    parser.add_argument("--golden-parity-waiver", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_closeout_bundle(args=args, command=command)
    markdown = render_markdown(summary)
    write_optional_summary_outputs(
        summary=summary,
        summary_json_path=args.summary_json,
        summary_markdown_path=args.summary_md,
        markdown_text=markdown,
    )
    if args.summary_json is None:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print(f"summary_json_path={Path(args.summary_json).resolve()}")
    return 0 if summary["closeout_passed"] else 1


def run_closeout_bundle(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    run_id = "part2_closeout_bundle_" + local_now_compact()
    run_root = Path(args.out_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    stage_results: list[dict[str, Any]] = []
    stage_results.append(
        run_command(
            [sys.executable, "-m", "pytest", "llm/tests", "-q"],
            name="architecture_pytest_full",
        )
    )
    stage_results.append(
        run_command(
            [sys.executable, "-m", "pytest", "llm/tests/runtime", "llm/tests/conversation", "-q"],
            name="architecture_pytest_runtime_conversation",
        )
    )
    stage_results.append(
        run_freeze_step(
            out_dir=args.freeze_out_dir,
            summary_root=run_root,
        )
    )
    stage_results.append(
        run_summary_script(
            name="golden_parity_report",
            script=ROOT / "scripts" / "run_part2_golden_parity_report.py",
            out_dir=run_root / "golden_parity",
            extra_args=build_golden_parity_args(args),
            summary_root=run_root,
            no_progress=args.no_progress,
        )
    )
    stage_results.append(
        run_summary_script(
            name="v1_thesis_metrics_report",
            script=ROOT / "scripts" / "run_part2_thesis_metrics_report.py",
            out_dir=run_root / "v1_thesis_metrics",
            extra_args=build_v1_thesis_args(args),
            summary_root=run_root,
            no_progress=args.no_progress,
        )
    )
    stage_results.append(
        run_summary_script(
            name="v1_refinement_metrics_report",
            script=ROOT / "scripts" / "run_part2_refinement_metrics_report.py",
            out_dir=run_root / "v1_refinement_metrics",
            extra_args=build_v1_refinement_args(args),
            summary_root=run_root,
            no_progress=args.no_progress,
        )
    )
    stage_results.append(
        run_summary_script(
            name="v2_closeout_harness",
            script=ROOT / "scripts" / "run_part2_end_to_end_bank.py",
            out_dir=run_root / "v2_closeout_harness",
            extra_args=build_v2_harness_args(args),
            summary_root=run_root,
            no_progress=args.no_progress,
        )
    )

    harness_stage = next((item for item in stage_results if item["name"] == "v2_closeout_harness"), None)
    harness_payload = None if harness_stage is None else harness_stage.get("payload")
    extracted_harness_summaries = extract_archive_and_validate_harness_summaries(
        harness_payload=harness_payload,
        archive_root=run_root / "archived_harness_summaries",
    )

    supporting_stage_names = {
        "architecture_pytest_full",
        "architecture_pytest_runtime_conversation",
        "freeze_v2_corpora",
        "golden_parity_report",
        "v1_thesis_metrics_report",
        "v1_refinement_metrics_report",
    }
    supporting_evidence_passed = all(
        item["passed"]
        for item in stage_results
        if item["name"] in supporting_stage_names
    )
    harness_gate_passed = bool(harness_payload and harness_payload.get("closeout_passed") is True)
    blocked_reasons = build_blocked_reasons(
        stage_results=stage_results,
        harness_gate_passed=harness_gate_passed,
        extracted_harness_summaries=extracted_harness_summaries,
    )
    closeout_passed = supporting_evidence_passed and harness_gate_passed and extracted_harness_summaries["ok"]

    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "bundle_root": str(run_root.resolve()),
        "summary_note": (
            "The v2 harness is the final pass/fail gate. Architecture tests are required supporting evidence, not a replacement."
        ),
        "boundary_corpus_policy": {
            "path": str(BANK_BOUNDARY_PROFILES_CORPUS_PATH.resolve()),
            "included_in_closeout_metrics": False,
            "note": "Boundary corpus is diagnostic only and excluded from final closeout metrics.",
        },
        "stage_results": stage_results,
        "supporting_evidence_passed": supporting_evidence_passed,
        "harness_gate_passed": harness_gate_passed,
        "extracted_harness_summaries": extracted_harness_summaries,
        "blocked_reasons": blocked_reasons,
        "closeout_passed": closeout_passed,
    }
    write_json(run_root / "part2_closeout_bundle_summary.json", summary)
    (run_root / "part2_closeout_bundle_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_golden_parity_args(args: argparse.Namespace) -> list[str]:
    items = [
        "--backend",
        "ufce",
    ]
    if args.golden_parity_waiver is not None:
        items.extend(["--unexpected-regression-waiver", str(args.golden_parity_waiver)])
    return items


def build_v1_thesis_args(args: argparse.Namespace) -> list[str]:
    return [
        "--baseline-catalog",
        str(args.baseline_catalog),
        "--benchmark",
        str(args.benchmark),
        "--lm-studio-api-base",
        args.lm_studio_api_base,
        "--model-alias",
        args.model_alias,
        "--product-mode",
        args.product_mode,
        "--api-version",
        args.api_version,
        "--app-version",
        args.app_version,
        "--tier-a-corpus",
        str(args.tier_a_corpus),
        "--tier-b-corpus",
        str(args.v1_tier_b_corpus),
    ]


def build_v1_refinement_args(args: argparse.Namespace) -> list[str]:
    return [
        "--baseline-catalog",
        str(args.baseline_catalog),
        "--benchmark",
        str(args.benchmark),
        "--lm-studio-api-base",
        args.lm_studio_api_base,
        "--model-alias",
        args.model_alias,
        "--product-mode",
        args.product_mode,
        "--api-version",
        args.api_version,
        "--app-version",
        args.app_version,
        "--tier-a-corpus",
        str(args.tier_a_corpus),
        "--tier-b-corpus",
        str(args.v1_tier_b_corpus),
    ]


def build_v2_harness_args(args: argparse.Namespace) -> list[str]:
    items = [
        "--baseline-catalog",
        str(args.baseline_catalog),
        "--benchmark",
        str(args.benchmark),
        "--lm-studio-api-base",
        args.lm_studio_api_base,
        "--model-alias",
        args.model_alias,
        "--product-mode",
        args.product_mode,
        "--api-version",
        args.api_version,
        "--app-version",
        args.app_version,
        "--tier-a-corpus",
        str(args.tier_a_corpus),
        "--tier-b-corpus",
        str(args.v2_tier_b_corpus),
        "--tier-c-corpus",
        str(args.tier_c_corpus),
        "--tier-d-corpus",
        str(args.tier_d_corpus),
        "--g5-corpus",
        str(args.v2_g5_corpus),
        "--g5-backends",
        args.g5_backends,
        "--g5-attempts-per-case",
        str(args.g5_attempts_per_case),
        "--full-tier-c",
    ]
    if args.g5_case_limit is not None:
        items.extend(["--g5-case-limit", str(args.g5_case_limit)])
    return items


def run_summary_script(
    *,
    name: str,
    script: Path,
    out_dir: Path,
    extra_args: list[str],
    summary_root: Path,
    no_progress: bool,
) -> dict[str, Any]:
    summary_json_path = summary_root / f"{name}_summary.json"
    summary_markdown_path = summary_root / f"{name}_summary.md"
    command = [
        sys.executable,
        str(script.resolve()),
        "--out-dir",
        str(out_dir),
        *extra_args,
        "--summary-json",
        str(summary_json_path),
        "--summary-md",
        str(summary_markdown_path),
    ]
    if no_progress:
        command.append("--no-progress")
    result = run_command(
        command,
        name=name,
        stream_stderr=True,
        summary_json_path=summary_json_path,
    )
    result["summary_json_path"] = str(summary_json_path)
    result["summary_markdown_path"] = str(summary_markdown_path)
    return result


def run_freeze_step(*, out_dir: Path, summary_root: Path) -> dict[str, Any]:
    summary_json_path = summary_root / "freeze_v2_corpora_summary.json"
    summary_markdown_path = summary_root / "freeze_v2_corpora_summary.md"
    command = [sys.executable, str((ROOT / "scripts" / "freeze_part2_bank_synth_corpora.py").resolve()), "--out-dir", str(Path(out_dir).resolve())]
    try:
        payload = freeze_bank_synth_corpora(out_dir=Path(out_dir))
        passed = True
        parse_error = None
        stdout_tail = []
        stderr_tail = []
        write_json(summary_json_path, payload)
        summary_markdown_path.write_text(render_freeze_markdown(payload), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive for live runs
        payload = None
        passed = False
        parse_error = f"{type(exc).__name__}: {exc}"
        stdout_tail = []
        stderr_tail = []
    return {
        "name": "freeze_v2_corpora",
        "command": " ".join(shlex.quote(item) for item in command),
        "exit_code": 0 if passed else 1,
        "passed": passed,
        "payload": payload,
        "parse_error": parse_error,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "summary_json_path": str(summary_json_path),
        "summary_markdown_path": str(summary_markdown_path),
    }


def run_command(
    command: list[str],
    *,
    name: str,
    stream_stderr: bool = False,
    summary_json_path: Path | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None if stream_stderr else subprocess.PIPE,
    )
    payload = None
    parse_error = None
    if summary_json_path is not None and completed.returncode == 0:
        try:
            payload = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    passed = completed.returncode == 0 and (summary_json_path is None or payload is not None)
    return {
        "name": name,
        "command": " ".join(shlex.quote(item) for item in command),
        "exit_code": completed.returncode,
        "passed": passed,
        "payload": payload,
        "parse_error": parse_error,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": [] if stream_stderr else tail_text(completed.stderr or ""),
    }


def tail_text(text: str, *, line_limit: int = 20) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-line_limit:]


def extract_archive_and_validate_harness_summaries(
    *,
    harness_payload: dict[str, Any] | None,
    archive_root: Path,
) -> dict[str, Any]:
    archive_root.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Any] = {}
    metadata_reports: dict[str, Any] = {}
    child_runs = {}
    if isinstance(harness_payload, dict):
        child_runs = {str(item.get("name")): item for item in harness_payload.get("child_runs", [])}
        copy_if_exists(
            source_path=Path(harness_payload["report_json_path"]) if "report_json_path" in harness_payload else None,
            target_path=archive_root / "v2_harness_summary.json",
        )

    for stage_name, label in REQUIRED_HARNESS_SUMMARIES.items():
        stage_payload = child_runs.get(stage_name)
        extracted[label] = archive_single_summary(
            label=label,
            stage_name=stage_name,
            stage_payload=stage_payload,
            archive_root=archive_root,
        )
        if extracted[label]["valid"]:
            metadata_reports[label] = {
                "corpus_path": extracted[label]["corpus_path"],
                "corpus_version": extracted[label]["corpus_version"],
                "corpus_sha256": extracted[label]["corpus_sha256"],
                "loaded_corpora": extracted[label]["loaded_corpora"],
            }

    metadata_payload = {
        "reports": metadata_reports,
        "required_report_count": len(REQUIRED_HARNESS_SUMMARIES),
        "present_report_count": len(metadata_reports),
    }
    metadata_json_path = archive_root / "corpus_version_hash_metadata.json"
    metadata_md_path = archive_root / "corpus_version_hash_metadata.md"
    write_json(metadata_json_path, metadata_payload)
    metadata_md_path.write_text(render_metadata_markdown(metadata_payload), encoding="utf-8")
    metadata_valid = len(metadata_reports) == len(REQUIRED_HARNESS_SUMMARIES)

    ok = all(item["valid"] for item in extracted.values()) and metadata_valid
    return {
        "archive_root": str(archive_root.resolve()),
        "required_summaries": list(REQUIRED_HARNESS_SUMMARIES.values()),
        "summaries": extracted,
        "corpus_version_hash_metadata": {
            "valid": metadata_valid,
            "summary_json_path": str(metadata_json_path.resolve()),
            "summary_markdown_path": str(metadata_md_path.resolve()),
        },
        "ok": ok,
    }


def archive_single_summary(
    *,
    label: str,
    stage_name: str,
    stage_payload: dict[str, Any] | None,
    archive_root: Path,
) -> dict[str, Any]:
    result = {
        "label": label,
        "stage_name": stage_name,
        "valid": False,
        "summary_json_path": None,
        "summary_markdown_path": None,
        "archived_json_path": None,
        "archived_markdown_path": None,
        "corpus_path": None,
        "corpus_version": None,
        "corpus_sha256": None,
        "loaded_corpora": None,
        "error": None,
    }
    if not isinstance(stage_payload, dict):
        result["error"] = "missing_stage_payload"
        return result
    json_path_raw = stage_payload.get("summary_json_path")
    markdown_path_raw = stage_payload.get("summary_markdown_path")
    if not isinstance(json_path_raw, str):
        result["error"] = "missing_summary_json_path"
        return result
    json_path = Path(json_path_raw).resolve()
    markdown_path = None if not isinstance(markdown_path_raw, str) else Path(markdown_path_raw).resolve()
    if not json_path.exists():
        result["error"] = "summary_json_missing"
        result["summary_json_path"] = str(json_path)
        return result
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = f"summary_json_malformed:{type(exc).__name__}"
        result["summary_json_path"] = str(json_path)
        return result
    required_fields = ("corpus_path", "corpus_version", "corpus_sha256")
    missing_fields = [field_name for field_name in required_fields if payload.get(field_name) in {None, ""}]
    if missing_fields:
        result["error"] = f"missing_required_fields:{','.join(missing_fields)}"
        result["summary_json_path"] = str(json_path)
        return result

    archived_json_path = archive_root / f"{stage_name}.summary.json"
    archived_markdown_path = archive_root / f"{stage_name}.summary.md"
    shutil.copy2(json_path, archived_json_path)
    if markdown_path is not None and markdown_path.exists():
        shutil.copy2(markdown_path, archived_markdown_path)

    result.update(
        {
            "valid": True,
            "summary_json_path": str(json_path),
            "summary_markdown_path": None if markdown_path is None else str(markdown_path),
            "archived_json_path": str(archived_json_path.resolve()),
            "archived_markdown_path": str(archived_markdown_path.resolve()) if archived_markdown_path.exists() else None,
            "corpus_path": payload.get("corpus_path"),
            "corpus_version": payload.get("corpus_version"),
            "corpus_sha256": payload.get("corpus_sha256"),
            "loaded_corpora": payload.get("loaded_corpora"),
            "error": None,
        }
    )
    return result


def copy_if_exists(*, source_path: Path | None, target_path: Path) -> None:
    if source_path is None or not source_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def build_blocked_reasons(
    *,
    stage_results: list[dict[str, Any]],
    harness_gate_passed: bool,
    extracted_harness_summaries: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for stage in stage_results:
        if not stage["passed"]:
            reasons.append(f"{stage['name']}_failed")
    if not harness_gate_passed:
        reasons.append("v2_harness_gate_blocked")
    if not extracted_harness_summaries["ok"]:
        reasons.append("required_harness_summaries_invalid")
    deduped: list[str] = []
    for item in reasons:
        if item not in deduped:
            deduped.append(item)
    return deduped


def render_freeze_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Freeze V2 Corpora",
        "",
        f"- out_dir: `{payload['out_dir']}`",
        "",
        "## Written Corpora",
        "",
    ]
    for corpus_name, record in payload["written_corpora"].items():
        lines.append(
            f"- {corpus_name}: path=`{record['path']}` corpus_version=`{record['corpus_version']}` corpus_sha256=`{record['corpus_sha256']}`"
        )
    return "\n".join(lines)


def render_metadata_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Corpus Version Hash Metadata",
        "",
        f"- required_report_count: `{payload['required_report_count']}`",
        f"- present_report_count: `{payload['present_report_count']}`",
        "",
    ]
    for label, record in payload["reports"].items():
        lines.append(
            f"- {label}: corpus_path=`{record['corpus_path']}` corpus_version=`{record['corpus_version']}` corpus_sha256=`{record['corpus_sha256']}`"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Closeout Bundle Summary",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- scorer_version: `{summary['scorer_version']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- bundle_root: `{summary['bundle_root']}`",
        "",
        "## Closeout Policy",
        "",
        f"- summary_note: `{summary['summary_note']}`",
        f"- boundary_corpus_policy: `{summary['boundary_corpus_policy']}`",
        "",
        "## Stage Results",
        "",
    ]
    for stage in summary["stage_results"]:
        lines.extend(
            [
                f"- {stage['name']}: passed=`{stage['passed']}` exit_code=`{stage['exit_code']}`",
                f"  command: `{stage['command']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Extracted Harness Summaries",
            "",
            f"- ok: `{summary['extracted_harness_summaries']['ok']}`",
            f"- corpus_version_hash_metadata: `{summary['extracted_harness_summaries']['corpus_version_hash_metadata']}`",
            "",
            "## Verdict",
            "",
            f"- supporting_evidence_passed: `{summary['supporting_evidence_passed']}`",
            f"- harness_gate_passed: `{summary['harness_gate_passed']}`",
            f"- blocked_reasons: `{summary['blocked_reasons']}`",
            f"- closeout_passed: `{summary['closeout_passed']}`",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
