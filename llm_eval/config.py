from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from llm_eval.models import BenchmarkCase, BenchmarkDefinition, EvalConfig, OutputContract, TargetField


DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "llm" / "prompts" / "parser_system_prompt_v1.txt"
)
DEFAULT_TIMEOUT_S = 300.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bank CF parser benchmark against the active LM Studio model."
    )
    parser.add_argument("--benchmark", required=True, help="Path to the benchmark YAML file.")
    parser.add_argument(
        "--model_alias",
        required=True,
        help="Exact LM Studio model value to send as `model`, also used for reporting.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Base output directory where `<model_alias>_<timestamp>` will be created.",
    )
    parser.add_argument(
        "--api_base",
        default="http://localhost:1234",
        help="LM Studio base URL.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per case.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p value.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum response tokens.")
    parser.add_argument(
        "--case_ids",
        default="",
        help="Optional comma-separated case ids to run.",
    )
    parser.add_argument("--group", default=None, help="Optional group filter (A/B/C).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases after filtering.",
    )
    parser.add_argument(
        "--timeout_s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP timeout in seconds for each LM Studio request. Default is 300 seconds (5 minutes).",
    )
    parser.add_argument(
        "--system_prompt",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help="Path to the fixed system prompt artifact.",
    )
    return parser


def parse_case_ids(case_ids_arg: str) -> tuple[str, ...]:
    if not case_ids_arg:
        return ()
    return tuple(case_id.strip() for case_id in case_ids_arg.split(",") if case_id.strip())


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_tokens < 1:
        raise ValueError("--max_tokens must be >= 1")
    if args.timeout_s <= 0:
        raise ValueError("--timeout_s must be > 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    return EvalConfig(
        benchmark_path=Path(args.benchmark).resolve(),
        model_alias=args.model_alias.strip(),
        out_dir=Path(args.out_dir).resolve(),
        api_base=args.api_base.rstrip("/"),
        repeats=args.repeats,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        case_ids=parse_case_ids(args.case_ids),
        group=args.group,
        limit=args.limit,
        system_prompt_path=Path(args.system_prompt).resolve(),
    )


def load_benchmark(path: Path) -> BenchmarkDefinition:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load the benchmark YAML. Install `pyyaml` in the active environment."
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return benchmark_from_dict(payload)


def benchmark_from_dict(payload: dict[str, Any]) -> BenchmarkDefinition:
    target_fields = tuple(
        TargetField(
            name=item["name"],
            type=item["type"],
            description=item.get("description", ""),
        )
        for item in payload["target_cf_fields"]
    )
    output_contract = OutputContract(
        task=payload["output_contract"]["task"],
        status_enum=tuple(payload["output_contract"]["status_enum"]),
        rules=tuple(payload["output_contract"].get("rules", [])),
    )
    cases = tuple(
        BenchmarkCase(
            case_id=item["case_id"],
            group=item["group"],
            description=item.get("description", ""),
            input_text=item["input"],
            expected_output=item["expected_output"],
        )
        for item in payload["cases"]
    )
    return BenchmarkDefinition(
        benchmark_name=payload["benchmark_name"],
        description=payload.get("description", ""),
        target_cf_fields=target_fields,
        output_contract=output_contract,
        cases=cases,
    )


def select_cases(benchmark: BenchmarkDefinition, config: EvalConfig) -> list[BenchmarkCase]:
    selected = list(benchmark.cases)
    if config.group:
        selected = [case for case in selected if case.group == config.group]
    if config.case_ids:
        selected_map = {case.case_id: case for case in selected}
        missing = [case_id for case_id in config.case_ids if case_id not in selected_map]
        if missing:
            raise ValueError(f"Unknown case ids after filtering: {', '.join(missing)}")
        selected = [selected_map[case_id] for case_id in config.case_ids]
    if config.limit is not None:
        selected = selected[: config.limit]
    if not selected:
        raise ValueError("No benchmark cases selected.")
    return selected
