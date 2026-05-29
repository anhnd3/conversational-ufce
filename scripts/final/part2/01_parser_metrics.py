#!/usr/bin/env python3
"""
Final self-contained parser metrics benchmark for Part II thesis evidence.
Runs structured extraction quality evaluation using LLM Studio client against Bank corpora.
Default suite runs bilingual v3 benchmarks (EN + VI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_V2 = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v2.yaml"
BENCHMARK_V3_EN = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_en.yaml"
BENCHMARK_V3_VI = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_vi.yaml"
FX_RULE_VERSION = "fixed_vnd_usd_25000_v1"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL_ALIAS = "qwen3-14b"


def run_parser_metrics(*, out_dir: Path, benchmark_path: Path = None,
                       lm_studio_api_base: str = None, model_alias: str = None,
                       max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    """Run parser benchmark metrics."""
    
    # Load config
    from llm.src.parser.prompt_builder import DEFAULT_RESPONSE_SCHEMA_NAME
    from llm.src.utils.hashing import sha256_file
    
    api_base = lm_studio_api_base or os.getenv("LM_STUDIO_API_BASE", "http://127.0.0.1:1234")
    model_alias = (
        model_alias
        or os.getenv("LM_STUDIO_MODEL_ALIAS")
        or os.getenv("MODEL_ALIAS")
        or DEFAULT_MODEL_ALIAS
    )
    
    benchmark_path = benchmark_path or BENCHMARK_V3_EN
    system_prompt_path = ROOT / "llm" / "prompts" / "parser_system_prompt_v1.txt"
    schema_path = ROOT / "llm" / "config" / "parser_schema_v3.json"
    
    command = [
        sys.executable,
        "-m",
        "llm_eval.run_bank_cf_llm_eval",
        "--benchmark", str(benchmark_path),
        "--api_base", api_base,
        "--model_alias", model_alias,
        "--out_dir", str(out_dir),
        "--max_tokens", str(max_tokens),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env["PYTHONUNBUFFERED"] = "1"
    run_started_timestamp = datetime.now(timezone.utc).timestamp()

    print(
        f"[part2-parser] Running {benchmark_path.stem} "
        f"model={model_alias} api={api_base} max_tokens={max_tokens}",
        file=sys.stderr,
        flush=True,
    )
    returncode, subprocess_output = run_command_streaming(
        command,
        cwd=str(ROOT),
        env=env,
    )
    
    # Try to find the report JSON. The eval runner writes into a model/timestamp child dir.
    summary_json_path = None
    try:
        summary_candidates = [
            f for f in out_dir.rglob("summary.json")
            if f.stat().st_mtime >= run_started_timestamp - 1
        ]
        if not summary_candidates:
            summary_candidates = list(out_dir.rglob("summary.json"))
        if summary_candidates:
            latest = max(summary_candidates, key=lambda f: f.stat().st_mtime)
            summary_json_path = str(latest.resolve())
    except Exception:
        pass

    summary_metrics = load_summary_metrics(Path(summary_json_path)) if summary_json_path else {}
    failure_counts = summary_metrics.get("failure_counts", {})
    overall_metrics = summary_metrics.get("overall", {})
    evidence_passed = (
        returncode == 0
        and summary_json_path is not None
        and int(failure_counts.get("api_error_count", 0) or 0) == 0
        and float(overall_metrics.get("valid_json_rate", 0.0) or 0.0) == 1.0
        and float(overall_metrics.get("schema_valid_rate", 0.0) or 0.0) == 1.0
        and float(overall_metrics.get("avg_hallucination_count", 0.0) or 0.0) == 0.0
        and float(overall_metrics.get("field_accuracy_mean", 0.0) or 0.0) == 1.0
        and float(overall_metrics.get("status_accuracy", 0.0) or 0.0) == 1.0
        and float(overall_metrics.get("missing_fields_accuracy", 0.0) or 0.0) == 1.0
        and float(overall_metrics.get("conflict_accuracy", 0.0) or 0.0) == 1.0
    )
    
    return {
        "name": "parser_metrics",
        "command": " ".join(command),
        "exit_code": returncode,
        "passed": evidence_passed,
        "summary_json_path": summary_json_path,
        "failure_counts": failure_counts,
        "overall": overall_metrics,
        "provenance": {
            "benchmark_path": str(Path(benchmark_path).resolve()),
            "benchmark_sha256": sha256_file(Path(benchmark_path)),
            "parser_prompt_template_version": system_prompt_path.stem,
            "system_prompt_path": str(system_prompt_path.resolve()),
            "system_prompt_sha256": sha256_file(system_prompt_path),
            "response_schema_name": DEFAULT_RESPONSE_SCHEMA_NAME,
            "schema_path": str(schema_path.resolve()),
            "schema_sha256": sha256_file(schema_path),
            "parser_schema_version": schema_path.stem,
            "structured_output_mode": "json_schema_strict",
            "currency_conversion_mode": "allow_usd_and_vnd_heuristic",
            "vnd_usd_fx_rate": 25000,
            "fx_rule_version": FX_RULE_VERSION,
        },
        "stdout_tail": tail_text(subprocess_output),
    }


def run_command_streaming(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
) -> tuple[int, str]:
    import subprocess

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(1)
        if chunk:
            chunks.append(chunk)
            sys.stderr.write(chunk)
            sys.stderr.flush()
            continue
        if process.poll() is not None:
            break
    return process.wait(), "".join(chunks)


def load_summary_metrics(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tail_text(text: str, limit: int = 20) -> list[str]:
    """Get last N non-empty lines."""
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not looks_like_progress_line(line)
    ]
    return lines[-limit:]


def looks_like_progress_line(line: str) -> bool:
    return "%|" in line and "| " in line and "req" in line


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Final parser metrics benchmark.")
    parser.add_argument("--out-dir", type=Path, 
                       default=ROOT / "outputs" / "final" / "part2" / "parser_metrics")
    parser.add_argument(
        "--suite",
        choices=["v3-bilingual", "v3-en", "v3-vi", "v2", "custom"],
        default="v3-bilingual",
    )
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--lm-studio-api-base", default=None)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("PART2_PARSER_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
        help="Maximum response tokens for each parser call. v3 field_evidence needs a larger budget than v2.",
    )
    
    args = parser.parse_args()
    
    if args.suite == "custom":
        if args.benchmark is None:
            raise SystemExit("--benchmark is required when --suite custom")
        benchmarks = [Path(args.benchmark)]
    elif args.suite == "v2":
        benchmarks = [BENCHMARK_V2]
    elif args.suite == "v3-en":
        benchmarks = [BENCHMARK_V3_EN]
    elif args.suite == "v3-vi":
        benchmarks = [BENCHMARK_V3_VI]
    else:
        benchmarks = [BENCHMARK_V3_EN, BENCHMARK_V3_VI]

    runs = []
    for benchmark_path in benchmarks:
        run_out_dir = args.out_dir / benchmark_path.stem
        runs.append(
            run_parser_metrics(
                out_dir=run_out_dir,
                benchmark_path=benchmark_path,
                lm_studio_api_base=args.lm_studio_api_base,
                model_alias=args.model_alias,
                max_tokens=args.max_tokens,
            )
        )

    summary = {
        "suite": args.suite,
        "passed": all(run["passed"] for run in runs),
        "run_count": len(runs),
        "runs": runs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
