from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm.src.adapters.lmstudio_client import call_lm_studio, extract_response_data
from llm.src.orchestration.parse_then_validate import parse_then_validate
from llm.src.parser.response_normalizer import NormalizedParseResult
from llm.src.parser.prompt_builder import build_request_payload, build_user_prompt, load_system_prompt
from llm.src.utils.hashing import (
    make_run_id,
    sha256_file,
    sha256_text,
    slugify_model_alias,
    strip_run_prefix,
    utc_now_iso,
)
from llm.src.utils.io import write_case_scores_csv, write_json, write_jsonl
from llm.src.validation.schema_validator import ValidationResult

from llm_eval.config import load_benchmark, select_cases
from llm_eval.models import BenchmarkDefinition, EvalConfig
from llm_eval.reporting import (
    build_summary,
    render_errors_markdown,
    render_summary_markdown,
)
from llm_eval.scoring import attach_stability_scores, score_prediction


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    benchmark = load_benchmark(config.benchmark_path)
    cases = select_cases(benchmark, config)
    system_prompt = load_system_prompt(config.system_prompt_path)
    run_id = make_run_id()
    model_dir_name = slugify_model_alias(config.model_alias)
    run_suffix = strip_run_prefix(run_id)
    run_dir = config.out_dir / f"{model_dir_name}_{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    total_requests = len(cases) * config.repeats

    with tqdm(
        total=total_requests,
        desc="Preparing eval",
        unit="req",
        dynamic_ncols=True,
    ) as progress:
        for case in cases:
            user_prompt = build_user_prompt(benchmark, case)
            request_payload = build_request_payload(
                model=config.model_alias,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            for repeat_id in range(1, config.repeats + 1):
                progress.set_description(
                    format_progress_description(
                        case_id=case.case_id,
                        repeat_id=repeat_id,
                        total_repeats=config.repeats,
                    )
                )
                api_result = call_lm_studio(
                    api_base=config.api_base,
                    payload=request_payload,
                    timeout_s=config.timeout_s,
                )
                response_data = extract_response_data(
                    api_result.get("response_json"),
                    api_result.get("elapsed_ms"),
                )
                normalized, validation_result = evaluate_model_output(
                    api_result=api_result,
                    response_data=response_data,
                    benchmark=benchmark,
                )
                scoring = score_prediction(
                    benchmark=benchmark,
                    case=case,
                    parsed_json=normalized.parsed_json,
                    validation_result=validation_result,
                )

                raw_row = build_raw_row(
                    config=config,
                    case=case,
                    repeat_id=repeat_id,
                    request_payload=request_payload,
                    api_result=api_result,
                    response_data=response_data,
                )
                parsed_row = build_parsed_row(
                    config=config,
                    case=case,
                    repeat_id=repeat_id,
                    api_result=api_result,
                    normalized=normalized,
                    validation_result=validation_result,
                )
                score_row = build_score_row(
                    config=config,
                    case=case,
                    repeat_id=repeat_id,
                    response_data=response_data,
                    api_result=api_result,
                    normalized=normalized,
                    validation_result=validation_result,
                    scoring=scoring,
                )
                raw_rows.append(raw_row)
                parsed_rows.append(parsed_row)
                score_rows.append(score_row)

                progress.set_postfix_str(
                    format_progress_postfix(
                        api_result=api_result,
                        response_data=response_data,
                    ),
                    refresh=False,
                )
                progress.update(1)

    attach_stability_scores(score_rows)
    summary = build_summary(
        score_rows,
        benchmark_name=benchmark.benchmark_name,
        model_alias=config.model_alias,
        run_id=run_id,
        run_dir=run_dir,
    )
    write_outputs(
        run_dir=run_dir,
        config=config,
        benchmark=benchmark,
        cases=cases,
        system_prompt=system_prompt,
        run_id=run_id,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        score_rows=score_rows,
        summary=summary,
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "summary": summary,
    }


def build_raw_row(
    *,
    config: EvalConfig,
    case,
    repeat_id: int,
    request_payload: dict[str, Any],
    api_result: dict[str, Any],
    response_data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "model_alias": config.model_alias,
        "case_id": case.case_id,
        "group": case.group,
        "repeat_id": repeat_id,
        "request_payload": request_payload,
        "api_ok": api_result["ok"],
        "http_status_code": api_result["status_code"],
        "api_error": api_result["error"],
        "full_api_response": api_result["response_json"],
        "raw_response_text": api_result["response_text"],
        "reasoning_text": response_data["reasoning_text"],
        "final_message_text": response_data["message_text"],
        "usage": response_data["usage"],
        "stats": response_data["stats"],
        "derived_metrics": response_data["derived_metrics"],
    }


def build_parsed_row(
    *,
    config: EvalConfig,
    case,
    repeat_id: int,
    api_result: dict[str, Any],
    normalized,
    validation_result,
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "model_alias": config.model_alias,
        "case_id": case.case_id,
        "group": case.group,
        "repeat_id": repeat_id,
        "normalized_text": normalized.normalized_text,
        "used_brace_extraction": normalized.used_brace_extraction,
        "parsed_json": normalized.parsed_json,
        "parse_error": normalized.parse_error,
        "api_error": api_result["error"],
        "validator_result": validation_result.to_dict(),
    }


def build_score_row(
    *,
    config: EvalConfig,
    case,
    repeat_id: int,
    response_data: dict[str, Any],
    api_result: dict[str, Any],
    normalized,
    validation_result,
    scoring: dict[str, Any],
) -> dict[str, Any]:
    derived_metrics = response_data["derived_metrics"]
    return {
        "model_alias": config.model_alias,
        "case_id": case.case_id,
        "group": case.group,
        "repeat_id": repeat_id,
        "valid_json": scoring["valid_json"],
        "schema_valid": scoring["schema_valid"],
        "exact_match": scoring["exact_match"],
        "field_accuracy": scoring["field_accuracy"],
        "status_correct": scoring["status_correct"],
        "missing_fields_correct": scoring["missing_fields_correct"],
        "conflicts_correct": scoring["conflicts_correct"],
        "hallucination_count": scoring["hallucination_count"],
        "request_latency_ms": derived_metrics.get("request_latency_ms"),
        "ttft_ms": derived_metrics.get("ttft_ms"),
        "total_time_ms": derived_metrics.get("total_time_ms"),
        "tokens_per_second": derived_metrics.get("tokens_per_second"),
        "prompt_tokens": derived_metrics.get("prompt_tokens"),
        "completion_tokens": derived_metrics.get("completion_tokens"),
        "reasoning_output_tokens": derived_metrics.get("reasoning_output_tokens"),
        "total_tokens": derived_metrics.get("total_tokens"),
        "http_status_code": api_result["status_code"],
        "api_error": api_result["error"],
        "raw_response_text": api_result["response_text"],
        "full_api_response": api_result["response_json"],
        "final_message_text": response_data["message_text"],
        "parse_error": normalized.parse_error,
        "parsed_json": normalized.parsed_json,
        "validation_error_count": len(validation_result.errors),
        "validation_errors": " | ".join(validation_result.errors),
        "validation_error_list": list(validation_result.errors),
    }


def format_progress_description(case_id: str, repeat_id: int, total_repeats: int) -> str:
    return f"{case_id} repeat {repeat_id}/{total_repeats}"


def format_progress_postfix(
    *,
    api_result: dict[str, Any],
    response_data: dict[str, Any],
) -> str:
    parts: list[str] = []
    status_code = api_result.get("status_code")
    if status_code is not None:
        parts.append(f"http={status_code}")
    elif api_result.get("error"):
        parts.append("http=error")

    latency_ms = response_data.get("derived_metrics", {}).get("request_latency_ms")
    if isinstance(latency_ms, (int, float)):
        parts.append(f"latency_ms={latency_ms:.1f}")

    return ", ".join(parts) or "running"


def evaluate_model_output(
    *,
    api_result: dict[str, Any],
    response_data: dict[str, Any],
    benchmark: BenchmarkDefinition,
) -> tuple[NormalizedParseResult, ValidationResult]:
    return parse_then_validate(
        message_text=response_data["message_text"],
        benchmark=benchmark,
        api_error=api_result.get("error"),
    )


def should_skip_parse_for_api_error(api_result: dict[str, Any], message_text: str) -> bool:
    return bool(api_result.get("error")) and not message_text.strip()


def write_outputs(
    *,
    run_dir: Path,
    config: EvalConfig,
    benchmark: BenchmarkDefinition,
    cases,
    system_prompt: str,
    run_id: str,
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_jsonl(run_dir / "raw_outputs.jsonl", raw_rows)
    write_jsonl(run_dir / "parsed_outputs.jsonl", parsed_rows)
    write_case_scores_csv(run_dir / "case_scores.csv", score_rows)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    (run_dir / "errors.md").write_text(render_errors_markdown(score_rows), encoding="utf-8")
    write_json(
        run_dir / "config_snapshot.json",
        build_config_snapshot(
            config=config,
            benchmark=benchmark,
            cases=cases,
            system_prompt=system_prompt,
            run_id=run_id,
        ),
    )


def build_config_snapshot(
    *,
    config: EvalConfig,
    benchmark: BenchmarkDefinition,
    cases,
    system_prompt: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "benchmark_name": benchmark.benchmark_name,
        "benchmark_path": str(config.benchmark_path),
        "benchmark_sha256": sha256_file(config.benchmark_path),
        "system_prompt_path": str(config.system_prompt_path),
        "system_prompt_sha256": sha256_text(system_prompt),
        "system_prompt_version": config.system_prompt_path.stem,
        "model_alias": config.model_alias,
        "lm_studio_model": config.model_alias,
        "api_base": config.api_base,
        "repeats": config.repeats,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "timeout_s": config.timeout_s,
        "request_contract": {
            "endpoint": "/api/v1/chat",
            "payload_keys": ["model", "system_prompt", "input"],
            "generation_params_sent": [],
        },
        "selected_case_ids": [case.case_id for case in cases],
        "selected_group": config.group,
        "selected_limit": config.limit,
    }
