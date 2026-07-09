#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import shlex
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.conversation.types import serialize_normalized_parse_payload
from llm.src.parser.bank_profile_parse_v2 import (
    BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
    BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
    BANK_PROFILE_PARSE_V2_VERIFIER_STAGE,
    MAX_EXTRACTION_RETRIES,
    MAX_NORMALIZATION_RETRIES,
    MAX_VERIFIER_CORRECTIONS,
    classify_bank_profile_v2_failed_stage,
    parse_bank_profile_parse_v2_extraction_payload,
    parse_bank_profile_parse_v2_verifier_payload,
    run_bank_profile_parse_v2_quality,
    run_bank_profile_parse_v2_quality_from_candidate,
    validate_bank_profile_parse_v2_extraction_payload,
)
from llm.src.parser.parser_quality import finalize_parser_quality_metadata
from llm.src.runtime.datasets.bank.metadata import BANK_FEATURE_TYPES, BANK_REQUIRED_FIELD_ORDER
from llm.src.utils.hashing import sha256_file, sha256_text
from llm.src.utils.io import json_ready, write_json, write_jsonl
from llm.src.utils.time import local_now_compact, local_now_iso
from llm.src.validation.schema_validator import ValidationResult
from llm_eval.models import BenchmarkDefinition, OutputContract, TargetField
from ufce.ufce_ff import cfmethods as ufce_ff_cfmethods
from ufce.ufce_ff import evaluations as ufce_ff_evaluations
from ufce.ufce_ff.ufce import UFCE as UFCEFF

BANK_FEATURE_ORDER = list(BANK_REQUIRED_FIELD_ORDER)
INTEGER_BANK_FIELDS = [field for field, kind in BANK_FEATURE_TYPES.items() if kind != "float"]
NUMERIC_BANK_FIELDS = [field for field, kind in BANK_FEATURE_TYPES.items() if kind == "float"]
BOOLEAN_BANK_FIELDS = [field for field, kind in BANK_FEATURE_TYPES.items() if kind == "binary"]

PART1_RUNNER_PATH = ROOT / "scripts" / "final" / "part1" / "ufce_only_reproduction.py"
PART1_FF_PATH = ROOT / "scripts" / "final" / "part1" / "ufce_force_flip_experiment.py"
PART1_FF_CANONICAL_SCRIPT = ROOT / "scripts" / "final" / "part1" / "04_ufce_ff.py"
PART1_FF_CORE_PACKAGE = "ufce.ufce_ff"
BANK_TOTEST_ROOT = ROOT / "ufce" / "data" / "folds" / "bank" / "totest"
BANK_SOURCE_PATH = ROOT / "ufce" / "data" / "bank.csv"
BENCHMARK_EN_PATH = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_en.yaml"
BENCHMARK_VI_PATH = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_vi.yaml"
DEFAULT_OUT_PARENT = ROOT / "outputs" / "final" / "part2"
DEFAULT_OUT_DIR_PREFIX = "nl_bank_bridge_"
METHODS = ["UFCE1", "UFCE2", "UFCE3"]
DESIRED_OUTCOME = 1
EXPECTED_CASE_COUNT = 250
EXPECTED_FOLD_COUNT = 5
EXPECTED_ROWS_PER_FOLD = 50
BASELINE_PARSE_FAILURE_COUNT = 17
BASELINE_PARSE_MAX_ERRORS = BASELINE_PARSE_FAILURE_COUNT - 1
BASELINE_GROUP_MAX_ERRORS = {"G1": 1, "G2": 16, "G3": 0, "G4": 0, "G5": 0}
FIELD_VALUE_ERROR_RE = re.compile(r"fields\.([A-Za-z]+)\.value (.+)")
FIELD_EVIDENCE_ERROR_RE = re.compile(r"fields\.([A-Za-z]+)\.evidence_quote (.+)")
TEMPLATE_VERSION = "nl_bank_bridge_v2_word_numbers"
PARSER_BENCHMARK_PATHS = {
    "en": BENCHMARK_EN_PATH,
    "vi": BENCHMARK_VI_PATH,
}
GROUP_SPECS: list[dict[str, str]] = [
    {
        "group_id": "G1",
        "label": "vi_clear",
        "description": "Vietnamese explicit prose",
        "benchmark_key": "vi",
    },
    {
        "group_id": "G2",
        "label": "vi_number_words",
        "description": "Vietnamese reordered prose with selected numeric values written as words",
        "benchmark_key": "vi",
    },
    {
        "group_id": "G3",
        "label": "en_clear",
        "description": "English explicit prose",
        "benchmark_key": "en",
    },
    {
        "group_id": "G4",
        "label": "en_number_words",
        "description": "English reordered prose with selected numeric values written as words",
        "benchmark_key": "en",
    },
    {
        "group_id": "G5",
        "label": "mixed_explicit",
        "description": "Mixed bilingual labeled profile",
        "benchmark_key": "vi",
    },
]
PRIMARY_OUTPUTS = [
    "bank_250_queries.csv",
    "bank_250_nl_inputs.jsonl",
    "parsed_requests.jsonl",
    "parse_errors.jsonl",
    "parse_quality_progress.json",
    "record_reconstruction.csv",
    "evaluation_case_selection.csv",
    "direct_ufce_ff_results.csv",
    "nl_first_ufce_ff_results.csv",
    "outcome_match_by_variant.csv",
    "outcome_match_by_group.csv",
    "mismatch_cases.csv",
    "invalid_release_cases.csv",
    "table_main_bridge.csv",
    "table_language_group.csv",
    "table_mismatch_reasons.csv",
    "summary.json",
    "summary.md",
    "config_snapshot.json",
]
PARSE_CHECKPOINT_OUTPUTS = [
    "bank_250_queries.csv",
    "bank_250_nl_inputs.jsonl",
    "parsed_requests.jsonl",
    "parse_errors.jsonl",
    "parse_progress.json",
    "parse_quality_progress.json",
    "record_reconstruction.csv",
    "parse_acceptance.json",
    "config_snapshot.json",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Bank natural-language bridge experiment on the locked UFCE-FF path."
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "build_inputs", "parse", "eval_parse", "direct", "nl_first", "compare"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--parse-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional completed parse-only output directory to seed downstream "
            "eval_parse/direct/nl_first/compare stages into --out-dir."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases-per-group", type=int, default=10)
    parser.add_argument("--model-alias", default="qwen3-14b")
    parser.add_argument("--api-base", default="http://localhost:1234")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--config-profile", default="final_freeze", choices=["public_github_source", "final_freeze", "new_best_params"])
    parser.add_argument("--bundle-mode", default="table7_author_public", choices=["author_public", "table7_author_public", "domain_actionable"])
    parser.add_argument("--mi-k", default="5", help="MI pair budget passed into the frozen Part I runner. Use 'all' for every pair.")
    parser.add_argument("--no-cf", type=int, default=10)
    parser.add_argument("--contprox-metric", default="euclidean")
    parser.add_argument(
        "--parse-eval-scope",
        default="accepted",
        choices=["accepted", "ready", "all"],
        help=(
            "Select which parser outputs enter UFCE-FF evaluation. "
            "'accepted' keeps ready+exact cases with no parser/verifier/schema errors; "
            "'ready' keeps every runtime-ready case; 'all' keeps the full 250-case set."
        ),
    )
    parser.add_argument(
        "--mi-feature-scope",
        default="configured_actionable",
        choices=["configured_actionable", "uf_only", "all_features"],
    )
    parser.set_defaults(lmstudio_stream=env_flag("LMSTUDIO_STREAM", default=False))
    parser.add_argument(
        "--lmstudio-stream",
        dest="lmstudio_stream",
        action="store_true",
        help="Use OpenAI-compatible SSE streaming for LM Studio parser calls.",
    )
    parser.add_argument(
        "--no-lmstudio-stream",
        dest="lmstudio_stream",
        action="store_false",
        help="Use non-streaming LM Studio parser calls.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_runner_command(Path(__file__))
    progress_enabled = not bool(args.no_progress)

    if args.parse_source_dir is not None and args.stage in {"all", "build_inputs", "parse"}:
        raise ValueError("--parse-source-dir is only valid for eval_parse/direct/nl_first/compare stages.")
    if args.parse_source_dir is not None:
        copy_parse_checkpoint_artifacts(source_dir=Path(args.parse_source_dir), out_dir=out_dir)

    if args.stage in {"all", "build_inputs"}:
        query_df = build_query_frame(seed=int(args.seed), cases_per_group=int(args.cases_per_group))
        write_query_artifacts(out_dir=out_dir, query_df=query_df)
        write_config_snapshot(
            out_dir=out_dir,
            args=args,
            command=command,
        )
    else:
        query_df = load_query_frame(out_dir)
    if args.stage == "build_inputs":
        return 0

    if args.stage in {"all", "parse"}:
        parse_df = run_parse_stage(
            query_df=query_df,
            out_dir=out_dir,
            model_alias=str(args.model_alias),
            api_base=str(args.api_base),
            timeout_s=float(args.timeout_s),
            lmstudio_stream=bool(args.lmstudio_stream),
            progress_enabled=progress_enabled,
        )
    else:
        parse_df = load_parse_frame(out_dir)
    validate_parse_checkpoint_matches_query(query_df=query_df, parse_df=parse_df)
    if args.stage == "parse":
        return 0
    parse_acceptance = evaluate_parse_acceptance(
        parse_rows=parse_df.to_dict(orient="records"),
        total_cases=len(query_df),
        out_dir=out_dir,
    )
    eval_selection_df = build_eval_case_selection_frame(
        query_df=query_df,
        parse_df=parse_df,
        scope=str(args.parse_eval_scope),
    )
    write_eval_case_selection_artifact(out_dir=out_dir, selection_df=eval_selection_df)
    eval_query_df = filter_eval_query_frame(query_df=query_df, selection_df=eval_selection_df)
    if args.stage == "eval_parse":
        print(json.dumps(json_ready(parse_acceptance), ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if bool(parse_acceptance.get("acceptance_passed")) else 1
    if eval_query_df.empty:
        raise RuntimeError(
            "No parser cases qualified for UFCE-FF evaluation under "
            f"parse_eval_scope={args.parse_eval_scope!r}. "
            "See evaluation_case_selection.csv and parse_acceptance.json."
        )

    if args.stage in {"all", "direct"}:
        direct_df = run_direct_stage(
            query_df=eval_query_df,
            out_dir=out_dir,
            config_profile=str(args.config_profile),
            bundle_mode=str(args.bundle_mode),
            mi_k_token=str(args.mi_k),
            no_cf=int(args.no_cf),
            contprox_metric=str(args.contprox_metric),
            mi_feature_scope=str(args.mi_feature_scope),
            progress_enabled=progress_enabled,
        )
    else:
        direct_df = load_variant_frame(out_dir, "direct_ufce_ff_results.csv")
    if args.stage == "direct":
        return 0

    if args.stage in {"all", "nl_first"}:
        nl_df = run_nl_first_stage(
            query_df=eval_query_df,
            parse_df=parse_df,
            out_dir=out_dir,
            config_profile=str(args.config_profile),
            bundle_mode=str(args.bundle_mode),
            mi_k_token=str(args.mi_k),
            no_cf=int(args.no_cf),
            contprox_metric=str(args.contprox_metric),
            mi_feature_scope=str(args.mi_feature_scope),
            progress_enabled=progress_enabled,
        )
    else:
        nl_df = load_variant_frame(out_dir, "nl_first_ufce_ff_results.csv")
    if args.stage == "nl_first":
        return 0

    if args.stage in {"all", "compare"}:
        comparison_payload = build_bridge_artifacts(
            source_query_df=query_df,
            query_df=eval_query_df,
            parse_df=parse_df,
            direct_df=direct_df,
            nl_df=nl_df,
            out_dir=out_dir,
            args=args,
            command=command,
            eval_selection_df=eval_selection_df,
        )
        summary = comparison_payload["summary"]
        print(json.dumps(json_ready(summary), ensure_ascii=True, indent=2, sort_keys=True))

    ensure_primary_outputs(out_dir=out_dir, stage=str(args.stage))
    return 0


def resolve_out_dir(out_dir: Path | None) -> Path:
    if out_dir is not None:
        return Path(out_dir).resolve()
    return (DEFAULT_OUT_PARENT / (DEFAULT_OUT_DIR_PREFIX + local_now_compact())).resolve()


def copy_parse_checkpoint_artifacts(*, source_dir: Path, out_dir: Path) -> None:
    source_root = source_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Parse source directory does not exist: {source_root}")
    copied: list[str] = []
    missing_required: list[str] = []
    for name in PARSE_CHECKPOINT_OUTPUTS:
        source_path = source_root / name
        if not source_path.exists():
            if name in {"bank_250_queries.csv", "parsed_requests.jsonl"}:
                missing_required.append(name)
            continue
        shutil.copy2(source_path, out_dir / name)
        copied.append(name)
    if missing_required:
        raise FileNotFoundError(
            "Parse source directory is missing required checkpoint files: "
            + ", ".join(missing_required)
        )
    write_json(
        out_dir / "parse_source_snapshot.json",
        {
            "copied_at": local_now_iso(),
            "parse_source_dir": str(source_root),
            "output_dir": str(out_dir.resolve()),
            "copied_files": copied,
        },
    )


def env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_runner_command(script_path: Path, argv: list[str] | None = None) -> str:
    command = [sys.executable, str(script_path.resolve()), *((argv or sys.argv[1:]))]
    return " ".join(shlex.quote(item) for item in command)


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
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, total=total, ascii=True, dynamic_ncols=True, leave=False)


def create_parse_progress_bars(*, enabled: bool, total: int):
    if not enabled:
        return None, None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ModuleNotFoundError:
        return None, None
    parse_bar = tqdm(
        total=total,
        desc="NL parse",
        unit="case",
        ascii=True,
        dynamic_ncols=True,
        position=0,
        leave=True,
    )
    quality_bar = tqdm(
        total=total,
        desc="Parser quality",
        unit="case",
        ascii=True,
        dynamic_ncols=True,
        position=1,
        leave=True,
    )
    return parse_bar, quality_bar


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_ufce_ff_core(mod01b) -> None:
    """Use the split UFCE-FF core while retaining Part 1 runner helpers."""
    mod01b.cfmethods = ufce_ff_cfmethods
    mod01b.eval_module = ufce_ff_evaluations
    mod01b.UFCE = UFCEFF
    mod01b.ufc = UFCEFF()


def build_bridge_benchmark(*, benchmark_key: str) -> BenchmarkDefinition:
    benchmark_key = str(benchmark_key).strip().lower()
    if benchmark_key not in {"en", "vi"}:
        raise ValueError(f"Unsupported benchmark key: {benchmark_key}")
    descriptions = {
        "en": "Bridge benchmark for English NL bank parsing under parser schema v3.",
        "vi": "Bridge benchmark for Vietnamese/mixed NL bank parsing under parser schema v3.",
    }
    target_fields = tuple(
        TargetField(
            name=field_name,
            type=str(BANK_FEATURE_TYPES[field_name]),
            description=f"Canonical bank field {field_name}",
        )
        for field_name in BANK_FEATURE_ORDER
    )
    return BenchmarkDefinition(
        benchmark_name=f"ufce_bank_cf_parser_bridge_v3_{benchmark_key}",
        description=descriptions[benchmark_key],
        target_cf_fields=target_fields,
        output_contract=OutputContract(
            task="extract_cf_request",
            status_enum=("complete", "partial", "needs_clarification", "conflict"),
            rules=(
                "Return only fields explicitly inferable from the input.",
                "Do not invent missing values.",
                "Use canonical field names exactly as defined.",
                "Include field_evidence for every field in cf_request.",
            ),
        ),
        cases=(),
    )


def load_part1_modules():
    mod01b = load_module_from_path("part1_ufce_only_reproduction_for_bridge", PART1_RUNNER_PATH)
    ffmod = load_module_from_path("part1_ufce_ff_experiment_for_bridge", PART1_FF_PATH)
    install_ufce_ff_core(mod01b)
    return mod01b, ffmod


def parse_mi_k_token(token: str) -> int | None:
    value = str(token).strip().lower()
    if value == "all":
        return None
    return int(value)


def resolve_bridge_ufce_ff_config(mod01b, dataset: str, config_profile: str) -> dict[str, Any]:
    profile = str(config_profile).strip().lower()
    if profile == "public_github_source":
        cfg = dict(ffmod_public_github_source_config())
    elif profile == "final_freeze":
        cfg = dict(mod01b.FINAL_RUNTIME_CONFIG[dataset])
    elif profile == "new_best_params":
        cfg = dict(mod01b.NEW_BEST_PARAMS[dataset])
    else:
        raise ValueError(
            "Unsupported config_profile "
            f"'{config_profile}'. Allowed: public_github_source, final_freeze, new_best_params."
        )
    cfg.update(
        {
            "ufce_flip_filter": 1,
            "selection_policy": "ufce_ff",
            "validity_gate_stage": "pre_find_best_row",
            "core_variant": "ufce_ff",
        }
    )
    return {
        "radius": int(cfg["radius"]),
        "n_neighbors": int(cfg["n_neighbors"]),
        "min_act": int(cfg["min_act"]),
        "min_feas": int(cfg["min_feas"]),
        "ufce_flip_filter": int(cfg["ufce_flip_filter"]),
        "selection_policy": str(cfg["selection_policy"]),
        "validity_gate_stage": str(cfg["validity_gate_stage"]),
        "core_variant": str(cfg["core_variant"]),
    }


def ffmod_public_github_source_config() -> dict[str, int]:
    return {
        "radius": 500,
        "n_neighbors": 100,
        "min_act": 3,
        "min_feas": 2,
        "ufce_flip_filter": 0,
    }


def bank_float_value(field_name: str, value: Any) -> float:
    number = float(value)
    if field_name == "CCAvg":
        return round(number + 1e-12, 1)
    return float(int(round(number)))


def normalize_bank_value(field_name: str, value: Any) -> float | int:
    field_type = BANK_FEATURE_TYPES[field_name]
    if field_type == "float":
        return bank_float_value(field_name, value)
    return int(round(float(value)))


def normalize_bank_profile(raw_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        field_name: normalize_bank_value(field_name, raw_profile[field_name])
        for field_name in BANK_FEATURE_ORDER
    }


def bank_profile_signature(profile: dict[str, Any] | None) -> str | None:
    if not isinstance(profile, dict):
        return None
    normalized = normalize_bank_profile(profile)
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


def profile_json_text(profile: dict[str, Any] | None) -> str | None:
    if not isinstance(profile, dict):
        return None
    return json.dumps(normalize_bank_profile(profile), ensure_ascii=True, sort_keys=True)


def parse_json_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        return None
    return payload


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return []
    return list(payload) if isinstance(payload, list) else []


def bank_values_equal(field_name: str, left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if BANK_FEATURE_TYPES[field_name] == "float":
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    return int(left) == int(right)


def profiles_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    norm_left = normalize_bank_profile(left)
    norm_right = normalize_bank_profile(right)
    return all(bank_values_equal(field_name, norm_left[field_name], norm_right[field_name]) for field_name in BANK_FEATURE_ORDER)


def diff_profile_fields(left: dict[str, Any] | None, right: dict[str, Any] | None) -> list[str]:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return list(BANK_FEATURE_ORDER)
    norm_left = normalize_bank_profile(left)
    norm_right = normalize_bank_profile(right)
    return [
        field_name
        for field_name in BANK_FEATURE_ORDER
        if not bank_values_equal(field_name, norm_left[field_name], norm_right[field_name])
    ]


def bool_word_en(value: int) -> str:
    return "yes" if int(value) == 1 else "no"


def bool_word_vi(value: int) -> str:
    return "co" if int(value) == 1 else "khong"


VI_DIGITS = {
    0: "khong",
    1: "mot",
    2: "hai",
    3: "ba",
    4: "bon",
    5: "nam",
    6: "sau",
    7: "bay",
    8: "tam",
    9: "chin",
}
EN_DIGITS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def vi_integer_words(value: int) -> str:
    number = int(value)
    if number < 0 or number > 999:
        return str(number)
    if number < 10:
        return VI_DIGITS[number]
    if number < 20:
        if number == 10:
            return "muoi"
        if number == 15:
            return "muoi lam"
        return "muoi " + VI_DIGITS[number % 10]
    if number < 100:
        tens = number // 10
        ones = number % 10
        words = f"{VI_DIGITS[tens]} muoi"
        if ones == 0:
            return words
        if ones == 1:
            return words + " mot"
        if ones == 5:
            return words + " lam"
        return words + " " + VI_DIGITS[ones]
    hundreds = number // 100
    remainder = number % 100
    words = f"{VI_DIGITS[hundreds]} tram"
    if remainder == 0:
        return words
    if remainder < 10:
        return words + " linh " + VI_DIGITS[remainder]
    return words + " " + vi_integer_words(remainder)


def en_integer_words(value: int) -> str:
    number = int(value)
    if number < 0 or number > 999:
        return str(number)
    small = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
    }
    tens_words = {
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }
    if number < 20:
        return small[number]
    if number < 100:
        tens = (number // 10) * 10
        ones = number % 10
        if ones == 0:
            return tens_words[tens]
        return f"{tens_words[tens]} {small[ones]}"
    hundreds = number // 100
    remainder = number % 100
    words = f"{small[hundreds]} hundred"
    if remainder == 0:
        return words
    return words + " " + en_integer_words(remainder)


def vi_number_words(value: float) -> str:
    rounded = round(float(value) + 1e-12, 1)
    whole = int(rounded)
    decimal = int(round((rounded - whole) * 10))
    if decimal == 0:
        return vi_integer_words(whole)
    return f"{vi_integer_words(whole)} phay {VI_DIGITS[decimal]}"


def en_number_words(value: float) -> str:
    rounded = round(float(value) + 1e-12, 1)
    whole = int(rounded)
    decimal = int(round((rounded - whole) * 10))
    if decimal == 0:
        return en_integer_words(whole)
    return f"{en_integer_words(whole)} point {EN_DIGITS[decimal]}"


def vi_phrase(field_name: str, value: int, *, alt: bool = False) -> str:
    if field_name == "SecuritiesAccount":
        return (
            "toi co tai khoan chung khoan"
            if int(value) == 1 and not alt
            else "toi khong co tai khoan chung khoan"
            if int(value) == 0 and not alt
            else "hien tai co tai khoan chung khoan"
            if int(value) == 1
            else "hien tai khong co tai khoan chung khoan"
        )
    if field_name == "CDAccount":
        return (
            "toi co CD account"
            if int(value) == 1 and not alt
            else "toi khong co CD account"
            if int(value) == 0 and not alt
            else "hien tai co CD account"
            if int(value) == 1
            else "hien tai khong co CD account"
        )
    if field_name == "Online":
        return (
            "toi dung online banking"
            if int(value) == 1 and not alt
            else "toi khong dung online banking"
            if int(value) == 0 and not alt
            else "hien tai co online banking"
            if int(value) == 1
            else "hien tai khong co online banking"
        )
    if field_name == "CreditCard":
        return (
            "toi co the tin dung"
            if int(value) == 1 and not alt
            else "toi khong co the tin dung"
            if int(value) == 0 and not alt
            else "hien tai co the tin dung"
            if int(value) == 1
            else "hien tai khong co the tin dung"
        )
    raise KeyError(field_name)


def en_phrase(field_name: str, value: int, *, alt: bool = False) -> str:
    if field_name == "SecuritiesAccount":
        return (
            "I have a securities account"
            if int(value) == 1 and not alt
            else "I do not have a securities account"
            if int(value) == 0 and not alt
            else "currently securities account yes"
            if int(value) == 1
            else "currently securities account no"
        )
    if field_name == "CDAccount":
        return (
            "I have a CD account"
            if int(value) == 1 and not alt
            else "I do not have a CD account"
            if int(value) == 0 and not alt
            else "currently CD account yes"
            if int(value) == 1
            else "currently CD account no"
        )
    if field_name == "Online":
        return (
            "I use online banking"
            if int(value) == 1 and not alt
            else "I do not use online banking"
            if int(value) == 0 and not alt
            else "currently online banking yes"
            if int(value) == 1
            else "currently online banking no"
        )
    if field_name == "CreditCard":
        return (
            "I have a credit card"
            if int(value) == 1 and not alt
            else "I do not have a credit card"
            if int(value) == 0 and not alt
            else "currently credit card yes"
            if int(value) == 1
            else "currently credit card no"
        )
    raise KeyError(field_name)


def render_nl_input(profile: dict[str, Any], group_id: str) -> str:
    income = bank_float_value("Income", profile["Income"])
    ccavg = bank_float_value("CCAvg", profile["CCAvg"])
    mortgage = bank_float_value("Mortgage", profile["Mortgage"])
    family = int(profile["Family"])
    education = int(profile["Education"])
    if group_id == "G1":
        return (
            f"Toi co ho so ngan hang nhu sau: thu nhap nam {income:g}, gia dinh {family} nguoi, "
            f"chi tieu the {ccavg:.1f} moi thang, hoc van muc {education}, the chap {mortgage:g}. "
            f"{vi_phrase('SecuritiesAccount', profile['SecuritiesAccount'])}, "
            f"{vi_phrase('CDAccount', profile['CDAccount'])}, "
            f"{vi_phrase('Online', profile['Online'])}, "
            f"{vi_phrase('CreditCard', profile['CreditCard'])}."
        )
    if group_id == "G2":
        return (
            f"Ho so hien tai cua toi la: gia dinh {vi_integer_words(family)} nguoi; "
            f"hoc van muc {vi_integer_words(education)}; thu nhap nam {vi_integer_words(int(income))}; "
            f"the chap {vi_integer_words(int(mortgage))}; "
            f"chi tieu the trung binh {vi_number_words(ccavg)} moi thang. "
            f"{vi_phrase('Online', profile['Online'], alt=True)}, "
            f"{vi_phrase('CreditCard', profile['CreditCard'], alt=True)}, "
            f"{vi_phrase('CDAccount', profile['CDAccount'], alt=True)}, "
            f"{vi_phrase('SecuritiesAccount', profile['SecuritiesAccount'], alt=True)}."
        )
    if group_id == "G3":
        return (
            f"My bank profile is: annual income {income:g}, family {family}, monthly card spend {ccavg:.1f}, "
            f"education level {education}, mortgage {mortgage:g}. "
            f"{en_phrase('SecuritiesAccount', profile['SecuritiesAccount'])}. "
            f"{en_phrase('CDAccount', profile['CDAccount'])}. "
            f"{en_phrase('Online', profile['Online'])}. "
            f"{en_phrase('CreditCard', profile['CreditCard'])}."
        )
    if group_id == "G4":
        return (
            f"Current profile: family size {en_integer_words(family)}, education level {en_integer_words(education)}, "
            f"income {en_integer_words(int(income))} per year, mortgage {en_integer_words(int(mortgage))}, "
            f"and average credit-card spending {en_number_words(ccavg)} each month. "
            f"{en_phrase('Online', profile['Online'], alt=True)}, "
            f"{en_phrase('CreditCard', profile['CreditCard'], alt=True)}, "
            f"{en_phrase('CDAccount', profile['CDAccount'], alt=True)}, "
            f"{en_phrase('SecuritiesAccount', profile['SecuritiesAccount'], alt=True)}."
        )
    if group_id == "G5":
        return (
            f"Bank profile: Income {income:g}, gia dinh {family}, monthly CCAvg {ccavg:.1f}, hoc van {education}, "
            f"Mortgage {mortgage:g}. SecuritiesAccount {bool_word_en(profile['SecuritiesAccount'])}, "
            f"CDAccount {bool_word_vi(profile['CDAccount'])}, online banking {bool_word_en(profile['Online'])}, "
            f"the tin dung {bool_word_vi(profile['CreditCard'])}."
        )
    raise ValueError(f"Unsupported language group: {group_id}")


def load_bank_fold_frames() -> list[tuple[int, Path, pd.DataFrame]]:
    fold_paths = sorted(BANK_TOTEST_ROOT.glob("testfold_*_pred_0.csv"))
    if len(fold_paths) != EXPECTED_FOLD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FOLD_COUNT} bank pred_0 folds in {BANK_TOTEST_ROOT}, found {len(fold_paths)}."
        )
    payload: list[tuple[int, Path, pd.DataFrame]] = []
    for fold_index, fold_path in enumerate(fold_paths):
        frame = pd.read_csv(fold_path).loc[:, BANK_FEATURE_ORDER].copy()
        if len(frame) != EXPECTED_ROWS_PER_FOLD:
            raise ValueError(f"Expected {EXPECTED_ROWS_PER_FOLD} rows in {fold_path.name}, found {len(frame)}.")
        for field_name in BANK_FEATURE_ORDER:
            frame[field_name] = [normalize_bank_value(field_name, value) for value in frame[field_name]]
        payload.append((fold_index, fold_path, frame))
    return payload


def assign_group_labels(*, row_count: int, fold_index: int, seed: int, cases_per_group: int) -> list[str]:
    groups = [spec["group_id"] for spec in GROUP_SPECS]
    expected = len(groups) * int(cases_per_group)
    if row_count != expected:
        raise ValueError(
            f"Expected {expected} rows for even group allocation ({cases_per_group} per group), found {row_count}."
        )
    ordered_groups: list[str] = []
    for group_id in groups:
        ordered_groups.extend([group_id] * int(cases_per_group))
    rng = __import__("random").Random(int(seed) + int(fold_index))
    shuffled_positions = list(range(row_count))
    rng.shuffle(shuffled_positions)
    assignment = [""] * row_count
    for position, group_id in zip(shuffled_positions, ordered_groups):
        assignment[position] = group_id
    if any(not value for value in assignment):
        raise RuntimeError("Group assignment failed to cover every row.")
    return assignment


def build_query_frame(*, seed: int, cases_per_group: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_map = {spec["group_id"]: spec for spec in GROUP_SPECS}
    for fold_index, fold_path, frame in load_bank_fold_frames():
        assignments = assign_group_labels(
            row_count=len(frame),
            fold_index=fold_index,
            seed=int(seed),
            cases_per_group=int(cases_per_group),
        )
        for query_pos, record in enumerate(frame.to_dict(orient="records")):
            profile = normalize_bank_profile(record)
            group_id = assignments[query_pos]
            group_spec = group_map[group_id]
            user_text = render_nl_input(profile, group_id)
            case_id = f"bank_fold{fold_index:02d}_q{query_pos:02d}_{group_id.lower()}"
            numeric_style = "word_numbers" if group_id in {"G2", "G4"} else "digits"
            row = {
                "case_id": case_id,
                "dataset": "bank",
                "fold_index": int(fold_index),
                "fold_name": fold_path.name,
                "query_pos": int(query_pos),
                "language_group": group_id,
                "language_label": group_spec["label"],
                "language_description": group_spec["description"],
                "benchmark_key": group_spec["benchmark_key"],
                "numeric_style": numeric_style,
                "user_text": user_text,
                "user_text_sha256": sha256_text(user_text),
                "original_profile_json": profile_json_text(profile),
                "original_profile_signature": bank_profile_signature(profile),
                "source_prediction_label": 0,
                "template_version": TEMPLATE_VERSION,
                **profile,
            }
            rows.append(row)
    query_df = pd.DataFrame(rows)
    if len(query_df) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected {EXPECTED_CASE_COUNT} query rows, found {len(query_df)}.")
    return query_df.sort_values(by=["fold_index", "query_pos"], kind="mergesort").reset_index(drop=True)
def write_query_artifacts(*, out_dir: Path, query_df: pd.DataFrame) -> None:
    query_df.to_csv(out_dir / "bank_250_queries.csv", index=False)
    write_jsonl(
        out_dir / "bank_250_nl_inputs.jsonl",
        [
            {
                "case_id": str(row["case_id"]),
                "language_group": str(row["language_group"]),
                "language_label": str(row["language_label"]),
                "benchmark_key": str(row["benchmark_key"]),
                "numeric_style": str(row.get("numeric_style", "digits")),
                "template_version": str(row.get("template_version", TEMPLATE_VERSION)),
                "user_text": str(row["user_text"]),
            }
            for row in query_df.to_dict(orient="records")
        ],
    )


def load_query_frame(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "bank_250_queries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing query artifact: {path}")
    frame = pd.read_csv(path)
    for field_name in BANK_FEATURE_ORDER:
        frame[field_name] = [normalize_bank_value(field_name, value) for value in frame[field_name]]
    return frame


def run_parse_stage(
    *,
    query_df: pd.DataFrame,
    out_dir: Path,
    model_alias: str,
    api_base: str,
    timeout_s: float,
    lmstudio_stream: bool,
    progress_enabled: bool,
) -> pd.DataFrame:
    from llm.src.conversation.canonical_validator import BankCanonicalValidator
    from llm.src.conversation.parser_adapter import LiveLmStudioParserAdapter
    from llm.src.runtime.datasets.bank.package import BankDatasetPackage
    from llm.src.runtime.model_registry import ModelRegistry

    registry = ModelRegistry()
    dataset_package = BankDatasetPackage(registry)
    bundle = dataset_package.load_model_bundle()
    validator = BankCanonicalValidator(model_registry=registry)
    adapter = LiveLmStudioParserAdapter(
        model_alias=model_alias,
        api_base=api_base,
        timeout_s=float(timeout_s),
        parse_max_tokens=1280,
        repair_max_tokens=1792,
        stream=bool(lmstudio_stream),
    )
    benchmarks = {
        key: build_bridge_benchmark(benchmark_key=key)
        for key, path in PARSER_BENCHMARK_PATHS.items()
    }

    parse_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    partial_parse_path = out_dir / "parsed_requests.partial.jsonl"
    parse_progress_path = out_dir / "parse_progress.json"
    parse_quality_progress_path = out_dir / "parse_quality_progress.json"
    partial_parse_path.write_text("", encoding="utf-8")
    write_json(
        parse_progress_path,
        {
            "stage": "parse",
            "status": "running",
            "completed_cases": 0,
            "total_cases": int(len(query_df)),
            "ready_for_runtime_count": 0,
            "error_count": 0,
            "observed_error_count": 0,
            "observed_error_rate": 0.0,
            "missing_unrun_case_count": int(len(query_df)),
            "acceptance_error_count": int(len(query_df)),
            "updated_at": local_now_iso(),
        },
    )
    live_quality = build_parse_live_quality_metrics(parse_rows=parse_rows, total_cases=len(query_df))
    write_json(parse_quality_progress_path, live_quality)
    query_rows = query_df.to_dict(orient="records")
    parse_bar, quality_bar = create_parse_progress_bars(
        enabled=progress_enabled,
        total=len(query_rows),
    )
    fail_fast_reasons: list[str] = []
    try:
        for row in query_rows:
            record = evaluate_parse_case(
                query_row=row,
                adapter=adapter,
                benchmarks=benchmarks,
                dataset_package=dataset_package,
                validator=validator,
                model=bundle.lr,
            )
            parse_rows.append(record)
            with partial_parse_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            reconstruction_rows.append(
                {
                    "case_id": record["case_id"],
                    "fold_name": record["fold_name"],
                    "query_pos": record["query_pos"],
                    "language_group": record["language_group"],
                    "benchmark_name": record["benchmark_name"],
                    "parser_status": record["parser_status"],
                    "final_stage": record["final_stage"],
                    "ready_for_runtime": int(bool(record["ready_for_runtime"])),
                    "repair_invoked": int(bool(record["repair_invoked"])),
                    "llm_retry_count": int(record.get("llm_retry_count") or 0),
                    "deterministic_correction_count": int(record.get("deterministic_correction_count") or 0),
                    "deterministic_corrections_json": record.get("deterministic_corrections_json"),
                    "verifier_invoked": int(bool(record.get("verifier_invoked"))),
                    "verifier_verdict": record.get("verifier_verdict"),
                    "exact_reconstruction": int(bool(record["exact_reconstruction"])),
                    "reconstruction_mismatch_fields_json": record["reconstruction_mismatch_fields_json"],
                    "prediction_before_original": record["prediction_before_original"],
                    "prediction_before_reconstructed": record["prediction_before_reconstructed"],
                    "original_profile_json": record["original_profile_json"],
                    "reconstructed_profile_json": record["reconstructed_profile_json"],
                    "parser_failure_cause": record["parser_failure_cause"],
                    "parser_api_error": record["parser_api_error"],
                    "repair_api_error": record["repair_api_error"],
                    "verifier_api_error": record.get("verifier_api_error"),
                    "request_latency_ms": record["request_latency_ms"],
                    "missing_runtime_fields_json": record["missing_runtime_fields_json"],
                    "confirmed_conflicts_json": record["confirmed_conflicts_json"],
                }
            )
            if not is_parse_row_accepted(record):
                error_rows.append(record)
            live_quality = build_parse_live_quality_metrics(parse_rows=parse_rows, total_cases=len(query_df))
            fail_fast_reasons = build_parse_fail_fast_reasons(
                live_quality=live_quality,
                total_cases=len(query_df),
            )
            live_status = "failed_fast" if fail_fast_reasons else "running"
            live_quality_with_status = {
                **live_quality,
                "status": live_status,
                "fail_fast_triggered": bool(fail_fast_reasons),
                "fail_fast_reasons": fail_fast_reasons,
                "fail_fast_error_limit": int(BASELINE_PARSE_MAX_ERRORS),
                "baseline_failure_count": int(BASELINE_PARSE_FAILURE_COUNT),
            }
            write_json(parse_quality_progress_path, live_quality_with_status)
            write_json(
                parse_progress_path,
                {
                    "stage": "parse",
                    "status": live_status,
                    "completed_cases": int(len(parse_rows)),
                    "total_cases": int(len(query_df)),
                    "accepted_count": int(live_quality["accepted_count"]),
                    "ready_for_runtime_count": int(live_quality["ready_for_runtime_count"]),
                    "exact_reconstruction_count": int(live_quality["exact_reconstruction_count"]),
                    "error_count": int(live_quality["error_count"]),
                    "observed_error_count": int(live_quality["error_count"]),
                    "observed_error_rate": float(live_quality["error_rate"]),
                    "missing_unrun_case_count": int(live_quality["remaining_cases"]),
                    "acceptance_error_count": int(live_quality["error_count"]) + int(live_quality["remaining_cases"]),
                    "live_error_rate": float(live_quality["error_rate"]),
                    "parser_api_error_count": int(live_quality["parser_api_error_count"]),
                    "retry_api_error_count": int(live_quality["retry_api_error_count"]),
                    "verifier_api_error_count": int(live_quality["verifier_api_error_count"]),
                    "verifier_reported_error_count": int(live_quality["verifier_reported_error_count"]),
                    "schema_error_count": int(live_quality["schema_error_count"]),
                    "fail_fast_triggered": bool(fail_fast_reasons),
                    "fail_fast_reasons": fail_fast_reasons,
                    "fail_fast_error_limit": int(BASELINE_PARSE_MAX_ERRORS),
                    "baseline_failure_count": int(BASELINE_PARSE_FAILURE_COUNT),
                    "latest_case_id": str(record["case_id"]),
                    "latest_language_group": str(record["language_group"]),
                    "updated_at": local_now_iso(),
                },
            )
            if parse_bar is not None:
                parse_bar.set_postfix_str(
                    f"{record['case_id']} {record['language_group']} retries={record.get('llm_retry_count') or 0}",
                    refresh=False,
                )
                parse_bar.update(1)
            if quality_bar is not None:
                quality_bar.set_postfix_str(format_parse_live_quality_progress(live_quality), refresh=False)
                quality_bar.update(1)
            if fail_fast_reasons:
                break
    finally:
        if parse_bar is not None:
            parse_bar.close()
        if quality_bar is not None:
            quality_bar.close()

    parse_df = pd.DataFrame(parse_rows)
    write_jsonl(out_dir / "parsed_requests.jsonl", parse_rows)
    write_jsonl(out_dir / "parse_errors.jsonl", error_rows)
    pd.DataFrame(reconstruction_rows).to_csv(out_dir / "record_reconstruction.csv", index=False)
    acceptance = build_parse_acceptance_metrics(
        parse_rows=parse_rows,
        total_cases=len(query_df),
        fail_fast_reasons=fail_fast_reasons,
    )
    final_live_quality = build_parse_live_quality_metrics(parse_rows=parse_rows, total_cases=len(query_df))
    final_status = "failed_fast" if fail_fast_reasons else (
        "complete" if acceptance["acceptance_passed"] else "failed_acceptance"
    )
    write_json(
        parse_quality_progress_path,
        {
            **final_live_quality,
            "status": final_status,
            "acceptance_passed": bool(acceptance["acceptance_passed"]),
            "fail_fast_triggered": bool(fail_fast_reasons),
            "fail_fast_reasons": fail_fast_reasons,
            "fail_fast_error_limit": int(BASELINE_PARSE_MAX_ERRORS),
            "baseline_failure_count": int(BASELINE_PARSE_FAILURE_COUNT),
        },
    )
    write_json(out_dir / "parse_acceptance.json", acceptance)
    write_json(
        parse_progress_path,
        {
            "stage": "parse",
            "status": final_status,
            "completed_cases": int(len(parse_rows)),
            "total_cases": int(len(query_df)),
            "ready_for_runtime_count": int(acceptance["ready_for_runtime_count"]),
            "exact_reconstruction_count": int(acceptance["exact_reconstruction_count"]),
            "error_count": int(acceptance["error_count"]),
            "observed_error_count": int(acceptance["observed_error_count"]),
            "observed_error_rate": float(acceptance["observed_error_rate"]),
            "missing_unrun_case_count": int(acceptance["missing_unrun_case_count"]),
            "acceptance_error_count": int(acceptance["acceptance_error_count"]),
            "ready_for_runtime_rate": float(acceptance["ready_for_runtime_rate"]),
            "exact_reconstruction_rate": float(acceptance["exact_reconstruction_rate"]),
            "error_rate": float(acceptance["error_rate"]),
            "acceptance_passed": bool(acceptance["acceptance_passed"]),
            "fail_fast_triggered": bool(fail_fast_reasons),
            "fail_fast_reasons": fail_fast_reasons,
            "fail_fast_error_limit": int(BASELINE_PARSE_MAX_ERRORS),
            "baseline_failure_count": int(BASELINE_PARSE_FAILURE_COUNT),
            "updated_at": local_now_iso(),
        },
    )
    if not acceptance["acceptance_passed"]:
        raise RuntimeError(format_parse_acceptance_failure(acceptance))
    return parse_df


def validate_parse_acceptance_or_raise(
    *,
    parse_rows: list[dict[str, Any]],
    total_cases: int,
    out_dir: Path,
) -> dict[str, Any]:
    acceptance = evaluate_parse_acceptance(
        parse_rows=parse_rows,
        total_cases=total_cases,
        out_dir=out_dir,
    )
    if not acceptance["acceptance_passed"]:
        raise RuntimeError(format_parse_acceptance_failure(acceptance))
    return acceptance


def evaluate_parse_acceptance(
    *,
    parse_rows: list[dict[str, Any]],
    total_cases: int,
    out_dir: Path,
) -> dict[str, Any]:
    acceptance = build_parse_acceptance_metrics(parse_rows=parse_rows, total_cases=total_cases)
    write_json(out_dir / "parse_acceptance.json", acceptance)
    return acceptance


def build_parse_live_quality_metrics(
    *,
    parse_rows: list[dict[str, Any]],
    total_cases: int,
) -> dict[str, Any]:
    completed = int(len(parse_rows))
    ready_count = sum(1 for row in parse_rows if as_bool(row.get("ready_for_runtime")))
    exact_count = sum(1 for row in parse_rows if as_bool(row.get("exact_reconstruction")))
    accepted_count = sum(1 for row in parse_rows if is_parse_row_accepted(row))
    error_count = int(completed - accepted_count)
    parser_api_error_count = sum(1 for row in parse_rows if _has_text(row.get("parser_api_error")))
    retry_api_error_count = sum(1 for row in parse_rows if _has_text(row.get("repair_api_error")))
    verifier_api_error_count = sum(1 for row in parse_rows if _has_text(row.get("verifier_api_error")))
    verifier_reported_error_count = sum(1 for row in parse_rows if parse_json_list(row.get("verifier_errors_json")))
    schema_error_count = sum(1 for row in parse_rows if parse_json_list(row.get("schema_errors_json")))
    by_group = {
        group_name: build_parse_acceptance_group_metrics(group_rows)
        for group_name, group_rows in _group_parse_rows(parse_rows).items()
    }
    return {
        "stage": "parse_live_quality",
        "total_cases": int(total_cases),
        "completed_cases": completed,
        "remaining_cases": max(0, int(total_cases) - completed),
        "accepted_count": int(accepted_count),
        "ready_for_runtime_count": int(ready_count),
        "exact_reconstruction_count": int(exact_count),
        "error_count": int(error_count),
        "parser_api_error_count": int(parser_api_error_count),
        "retry_api_error_count": int(retry_api_error_count),
        "verifier_api_error_count": int(verifier_api_error_count),
        "verifier_reported_error_count": int(verifier_reported_error_count),
        "schema_error_count": int(schema_error_count),
        "accepted_rate": ratio(accepted_count, completed),
        "ready_for_runtime_rate": ratio(ready_count, completed),
        "exact_reconstruction_rate": ratio(exact_count, completed),
        "error_rate": ratio(error_count, completed),
        "completion_rate": ratio(completed, total_cases),
        "by_language_group": by_group,
        "failure_family_breakdown": build_parse_failure_family_breakdown(parse_rows),
        "generated_at": local_now_iso(),
    }


def format_parse_live_quality_progress(metrics: dict[str, Any]) -> str:
    completed = int(metrics.get("completed_cases") or 0)
    error_count = int(metrics.get("error_count") or 0)
    ready_count = int(metrics.get("ready_for_runtime_count") or 0)
    exact_count = int(metrics.get("exact_reconstruction_count") or 0)
    api_errors = (
        int(metrics.get("parser_api_error_count") or 0)
        + int(metrics.get("retry_api_error_count") or 0)
        + int(metrics.get("verifier_api_error_count") or 0)
    )
    schema_errors = int(metrics.get("schema_error_count") or 0)
    verifier_errors = int(metrics.get("verifier_reported_error_count") or 0)
    error_rate = 100.0 * float(metrics.get("error_rate") or 0.0)
    return (
        f"fail={error_count}/{completed} ({error_rate:.1f}%) "
        f"ready={ready_count}/{completed} exact={exact_count}/{completed} "
        f"api={api_errors} schema={schema_errors} verifier={verifier_errors}"
    )


def build_parse_acceptance_metrics(
    *,
    parse_rows: list[dict[str, Any]],
    total_cases: int,
    fail_fast_reasons: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    completed = int(len(parse_rows))
    ready_count = sum(1 for row in parse_rows if as_bool(row.get("ready_for_runtime")))
    exact_count = sum(1 for row in parse_rows if as_bool(row.get("exact_reconstruction")))
    parser_api_error_count = sum(1 for row in parse_rows if _has_text(row.get("parser_api_error")))
    retry_api_error_count = sum(1 for row in parse_rows if _has_text(row.get("repair_api_error")))
    verifier_api_error_count = sum(1 for row in parse_rows if _has_text(row.get("verifier_api_error")))
    verifier_reported_error_count = sum(1 for row in parse_rows if parse_json_list(row.get("verifier_errors_json")))
    schema_error_count = sum(1 for row in parse_rows if parse_json_list(row.get("schema_errors_json")))
    accepted_count = sum(1 for row in parse_rows if is_parse_row_accepted(row))
    missing_case_count = max(0, int(total_cases) - completed)
    observed_error_count = int(completed - accepted_count)
    error_count = int(observed_error_count + missing_case_count)
    by_group = {
        group_name: build_parse_acceptance_group_metrics(group_rows)
        for group_name, group_rows in _group_parse_rows(parse_rows).items()
    }
    strict_acceptance_passed = bool(
        completed == int(total_cases)
        and accepted_count == int(total_cases)
        and ready_count == int(total_cases)
        and exact_count == int(total_cases)
        and error_count == 0
    )
    fail_fast_reason_list = [str(reason) for reason in (fail_fast_reasons or [])]
    baseline_gate_applied = int(total_cases) == EXPECTED_CASE_COUNT
    baseline_failures = build_parse_baseline_gate_failures(
        completed=completed,
        total_cases=int(total_cases),
        error_count=error_count,
        by_group=by_group,
    ) if baseline_gate_applied else []
    baseline_acceptance_passed = bool(baseline_gate_applied and not baseline_failures and not fail_fast_reason_list)
    return {
        "stage": "parse_acceptance",
        "total_cases": int(total_cases),
        "completed_cases": completed,
        "accepted_count": int(accepted_count),
        "ready_for_runtime_count": int(ready_count),
        "exact_reconstruction_count": int(exact_count),
        "error_count": error_count,
        "missing_case_count": int(missing_case_count),
        "observed_error_count": int(observed_error_count),
        "observed_error_rate": ratio(observed_error_count, completed),
        "missing_unrun_case_count": int(missing_case_count),
        "acceptance_error_count": int(error_count),
        "parser_api_error_count": int(parser_api_error_count),
        "retry_api_error_count": int(retry_api_error_count),
        "verifier_api_error_count": int(verifier_api_error_count),
        "verifier_reported_error_count": int(verifier_reported_error_count),
        "schema_error_count": int(schema_error_count),
        "accepted_rate": ratio(accepted_count, total_cases),
        "ready_for_runtime_rate": ratio(ready_count, total_cases),
        "exact_reconstruction_rate": ratio(exact_count, total_cases),
        "error_rate": ratio(error_count, total_cases),
        "acceptance_passed": bool(baseline_acceptance_passed if baseline_gate_applied else strict_acceptance_passed),
        "strict_acceptance_passed": bool(strict_acceptance_passed),
        "fail_fast_triggered": bool(fail_fast_reason_list),
        "fail_fast_reasons": fail_fast_reason_list,
        "fail_fast_error_limit": int(BASELINE_PARSE_MAX_ERRORS),
        "baseline_failure_count": int(BASELINE_PARSE_FAILURE_COUNT),
        "baseline_gate_applied": bool(baseline_gate_applied),
        "baseline_acceptance_passed": bool(baseline_acceptance_passed),
        "baseline_total_error_limit": int(BASELINE_PARSE_MAX_ERRORS),
        "baseline_group_error_limits": dict(BASELINE_GROUP_MAX_ERRORS),
        "baseline_failures": baseline_failures,
        "by_language_group": by_group,
        "failure_family_breakdown": build_parse_failure_family_breakdown(parse_rows),
        "generated_at": local_now_iso(),
    }


def build_parse_fail_fast_reasons(
    *,
    live_quality: dict[str, Any],
    total_cases: int,
) -> list[str]:
    if int(total_cases) != EXPECTED_CASE_COUNT:
        return []
    reasons: list[str] = []
    error_count = int(live_quality.get("error_count") or 0)
    if error_count > BASELINE_PARSE_MAX_ERRORS:
        reasons.append(
            f"fail-fast: live parse errors {error_count} reached the old baseline count "
            f"{BASELINE_PARSE_FAILURE_COUNT}; target requires at most {BASELINE_PARSE_MAX_ERRORS} errors"
        )
    by_group = live_quality.get("by_language_group")
    if isinstance(by_group, dict):
        for group_name, max_errors in BASELINE_GROUP_MAX_ERRORS.items():
            group_metrics = by_group.get(group_name)
            if not isinstance(group_metrics, dict):
                continue
            group_errors = int(group_metrics.get("error_count") or 0)
            if group_errors > int(max_errors):
                reasons.append(
                    f"fail-fast: {group_name} live parse errors {group_errors} exceed baseline group target {max_errors}"
                )
    return reasons


def build_parse_baseline_gate_failures(
    *,
    completed: int,
    total_cases: int,
    error_count: int,
    by_group: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    if completed != total_cases:
        failures.append(f"completed_cases {completed} != total_cases {total_cases}")
    if error_count > BASELINE_PARSE_MAX_ERRORS:
        failures.append(
            f"overall parse errors {error_count} exceed baseline target {BASELINE_PARSE_MAX_ERRORS}"
        )
    for group_name, max_errors in BASELINE_GROUP_MAX_ERRORS.items():
        group_metrics = by_group.get(group_name)
        if not isinstance(group_metrics, dict):
            failures.append(f"missing language group {group_name} for baseline comparison")
            continue
        group_errors = int(group_metrics.get("error_count") or 0)
        if group_errors > int(max_errors):
            failures.append(
                f"{group_name} parse errors {group_errors} exceed baseline group target {max_errors}"
            )
    return failures


def build_parse_acceptance_group_metrics(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = int(len(group_rows))
    ready_count = sum(1 for row in group_rows if as_bool(row.get("ready_for_runtime")))
    exact_count = sum(1 for row in group_rows if as_bool(row.get("exact_reconstruction")))
    accepted_count = sum(1 for row in group_rows if is_parse_row_accepted(row))
    error_count = total - accepted_count
    return {
        "total_cases": total,
        "accepted_count": int(accepted_count),
        "ready_for_runtime_count": int(ready_count),
        "exact_reconstruction_count": int(exact_count),
        "error_count": int(error_count),
        "accepted_rate": ratio(accepted_count, total),
        "ready_for_runtime_rate": ratio(ready_count, total),
        "exact_reconstruction_rate": ratio(exact_count, total),
        "error_rate": ratio(error_count, total),
    }


def is_parse_row_accepted(row: dict[str, Any]) -> bool:
    if not as_bool(row.get("ready_for_runtime")):
        return False
    if not as_bool(row.get("exact_reconstruction")):
        return False
    if any(
        _has_text(row.get(field_name))
        for field_name in ("parser_api_error", "repair_api_error", "verifier_api_error")
    ):
        return False
    if parse_json_list(row.get("schema_errors_json")):
        return False
    if parse_json_list(row.get("verifier_errors_json")):
        return False
    return True


def format_parse_acceptance_failure(acceptance: dict[str, Any]) -> str:
    baseline_failures = acceptance.get("baseline_failures") or []
    fail_fast_reasons = acceptance.get("fail_fast_reasons") or []
    baseline_detail = ""
    if baseline_failures:
        baseline_detail = " baseline_failures=" + "; ".join(str(value) for value in baseline_failures) + "."
    if fail_fast_reasons:
        baseline_detail += " fail_fast_reasons=" + "; ".join(str(value) for value in fail_fast_reasons) + "."
    return (
        "Parse acceptance failed: "
        f"accepted={acceptance['accepted_count']}/{acceptance['total_cases']} "
        f"ready={acceptance['ready_for_runtime_count']}/{acceptance['total_cases']} "
        f"exact={acceptance['exact_reconstruction_count']}/{acceptance['total_cases']} "
        f"observed_errors={acceptance.get('observed_error_count', acceptance['error_count'])}/"
        f"{acceptance['completed_cases']} "
        f"observed_error_rate={acceptance.get('observed_error_rate', acceptance['error_rate']):.4f} "
        f"error_rate={acceptance['error_rate']:.4f}."
        f"{baseline_detail} "
        "See parse_acceptance.json, record_reconstruction.csv, and parse_errors.jsonl."
    )


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return int(value) != 0
    return bool(value)


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _group_parse_rows(parse_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in parse_rows:
        group_name = str(row.get("language_group") or "unknown")
        groups.setdefault(group_name, []).append(row)
    return groups


def evaluate_parse_case(
    *,
    query_row: dict[str, Any],
    adapter: LiveLmStudioParserAdapter,
    benchmarks: dict[str, Any],
    dataset_package: BankDatasetPackage,
    validator: BankCanonicalValidator,
    model,
) -> dict[str, Any]:
    benchmark_key = str(query_row["benchmark_key"])
    benchmark = benchmarks[benchmark_key]
    user_text = str(query_row["user_text"])
    numeric_bound_fields = dataset_package.numeric_bound_fields()

    extraction_result = adapter.parse_bank_profile_v2(
        user_text=user_text,
        benchmark=benchmark,
        dataset_package=dataset_package,
        stage=BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
    )
    extraction_retry_results: list[ParserAdapterResult] = []
    normalization_retry_results: list[ParserAdapterResult] = []
    extraction_hint_candidates: dict[str, list[dict[str, Any]]] = {}
    evidence_hint_applied_fields: list[str] = []
    evidence_hint_ambiguous_fields: list[str] = []

    current_extraction_result = extraction_result
    extraction_payload, extraction_parse_errors = parse_bank_profile_parse_v2_extraction_payload(
        extraction_result.message_text
    )
    extraction_errors = collect_bank_profile_v2_extraction_retry_errors(
        parser_result=extraction_result,
        extraction_payload=extraction_payload,
        extraction_parse_errors=extraction_parse_errors,
        benchmark=benchmark,
        user_text=user_text,
    )
    for _retry_index in range(MAX_EXTRACTION_RETRIES):
        if not should_attempt_bank_profile_v2_stage_retry(
            parser_result=current_extraction_result,
            errors=extraction_errors,
        ):
            break
        retry_missing_fields = (
            _missing_bank_profile_v2_extraction_fields(extraction_payload=extraction_payload, benchmark=benchmark)
            if isinstance(extraction_payload, dict)
            else []
        )
        retry_extraction_hints = build_bank_profile_v2_evidence_hint_candidates(
            user_text=user_text,
            field_names=retry_missing_fields,
        )
        extraction_hint_candidates = merge_bank_profile_v2_evidence_hints(
            extraction_hint_candidates,
            retry_extraction_hints,
        )
        retry_result = adapter.retry_bank_profile_v2(
            user_text=user_text,
            previous_output=current_extraction_result.message_text,
            errors=extraction_errors,
            benchmark=benchmark,
            dataset_package=dataset_package,
            stage=BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
            extraction_hints=retry_extraction_hints or None,
        )
        extraction_retry_results.append(retry_result)
        current_extraction_result = retry_result
        extraction_payload, extraction_parse_errors = parse_bank_profile_parse_v2_extraction_payload(
            retry_result.message_text
        )
        extraction_errors = collect_bank_profile_v2_extraction_retry_errors(
            parser_result=retry_result,
            extraction_payload=extraction_payload,
            extraction_parse_errors=extraction_parse_errors,
            benchmark=benchmark,
            user_text=user_text,
        )

    if isinstance(extraction_payload, dict):
        final_missing_fields = _missing_bank_profile_v2_extraction_fields(
            extraction_payload=extraction_payload,
            benchmark=benchmark,
        )
        final_extraction_hints = build_bank_profile_v2_evidence_hint_candidates(
            user_text=user_text,
            field_names=final_missing_fields,
        )
        extraction_hint_candidates = merge_bank_profile_v2_evidence_hints(
            extraction_hint_candidates,
            final_extraction_hints,
        )
        if extraction_errors:
            (
                extraction_payload,
                evidence_hint_applied_fields,
                evidence_hint_ambiguous_fields,
            ) = apply_bank_profile_v2_evidence_hint_fallback(
                extraction_payload=extraction_payload,
                benchmark=benchmark,
                user_text=user_text,
            )
            if evidence_hint_applied_fields:
                extraction_errors = collect_bank_profile_v2_extraction_retry_errors(
                    parser_result=current_extraction_result,
                    extraction_payload=extraction_payload,
                    extraction_parse_errors=extraction_parse_errors,
                    benchmark=benchmark,
                    user_text=user_text,
                )

    extraction_fields = None
    if isinstance(extraction_payload, dict) and isinstance(extraction_payload.get("fields"), dict):
        extraction_fields = dict(extraction_payload.get("fields") or {})

    current_normalization_result = None
    quality_result = None
    canonical_validation = None
    retry_errors = list(extraction_errors)
    verifier_result = None
    verifier_payload = None
    verifier_errors: list[str] = []
    verifier_verdict = None
    verifier_applied = False
    verifier_invocation_count = 0
    verifier_correction_count = 0
    failed_stage = None

    if extraction_errors or not extraction_fields:
        failed_stage = BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE
        extraction_failure_text = "; ".join(extraction_errors) or "Evidence extraction failed."
        quality_result = run_bank_profile_parse_v2_quality_from_candidate(
            candidate=None,
            benchmark_spec=benchmark,
            user_text=user_text,
            parse_error=extraction_failure_text,
            normalized_text=current_extraction_result.message_text,
            numeric_bound_fields=numeric_bound_fields,
        )
        canonical_validation = validator.validate(
            candidate=quality_result.normalized.parsed_json,
            schema_validation=quality_result.schema_validation,
        )
    else:
        current_normalization_result = adapter.parse_bank_profile_v2(
            user_text=user_text,
            benchmark=benchmark,
            dataset_package=dataset_package,
            stage=BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
            extracted_evidence=extraction_fields,
        )
        while True:
            quality_result = run_bank_profile_parse_v2_quality(
                message_text=current_normalization_result.message_text,
                benchmark_spec=benchmark,
                user_text=user_text,
                api_error=current_normalization_result.api_error,
                numeric_bound_fields=numeric_bound_fields,
            )
            canonical_validation = validator.validate(
                candidate=quality_result.normalized.parsed_json,
                schema_validation=quality_result.schema_validation,
            )
            retry_errors = collect_bank_profile_v2_normalization_retry_errors(
                parser_result=current_normalization_result,
                quality_result=quality_result,
                canonical_validation=canonical_validation,
            )
            verifier_errors = []
            verifier_payload = None
            verifier_result = None
            verifier_verdict = None
            if not retry_errors and isinstance(quality_result.v2_payload, dict) and not current_normalization_result.api_error:
                verifier_result = adapter.verify_bank_profile_v2(
                    user_text=user_text,
                    candidate=quality_result.v2_payload,
                    validation_errors=retry_errors,
                    benchmark=benchmark,
                    dataset_package=dataset_package,
                )
                verifier_invocation_count += 1
                verifier_payload, verifier_errors = parse_bank_profile_parse_v2_verifier_payload(
                    verifier_result.message_text
                )
                if verifier_result.api_error:
                    verifier_errors.append(str(verifier_result.api_error))
                if isinstance(verifier_payload, dict):
                    verifier_verdict = verifier_payload.get("verdict")
                    verifier_candidate = verifier_payload.get("candidate")
                    if verifier_verdict == "PASS":
                        verifier_errors = []
                    elif verifier_verdict == "CORRECTED":
                        if isinstance(verifier_candidate, dict):
                            verifier_quality = run_bank_profile_parse_v2_quality_from_candidate(
                                candidate=verifier_candidate,
                                benchmark_spec=benchmark,
                                user_text=user_text,
                                numeric_bound_fields=numeric_bound_fields,
                            )
                            verifier_canonical = validator.validate(
                                candidate=verifier_quality.normalized.parsed_json,
                                schema_validation=verifier_quality.schema_validation,
                            )
                            verifier_candidate_errors = collect_bank_profile_v2_normalization_retry_errors(
                                parser_result=verifier_result,
                                quality_result=verifier_quality,
                                canonical_validation=verifier_canonical,
                            )
                            if not verifier_candidate_errors and verifier_quality.schema_validation.is_valid:
                                quality_result = verifier_quality
                                canonical_validation = verifier_canonical
                                verifier_applied = True
                                verifier_correction_count = min(MAX_VERIFIER_CORRECTIONS, 1)
                                retry_errors = []
                                verifier_errors = []
                                break
                            verifier_errors.extend(verifier_candidate_errors)
                        else:
                            verifier_errors.append("Verifier returned CORRECTED without a candidate.")
                    else:
                        verifier_errors.append(f"Verifier verdict did not pass: {verifier_verdict}")
                if verifier_errors:
                    retry_errors = dedupe_preserve_order(list(retry_errors) + list(verifier_errors))
            if not retry_errors:
                break
            if len(normalization_retry_results) >= MAX_NORMALIZATION_RETRIES or not should_attempt_bank_profile_v2_stage_retry(
                parser_result=current_normalization_result,
                errors=retry_errors,
            ):
                if verifier_result is not None and verifier_errors:
                    quality_result = force_bank_profile_v2_failure(
                        quality_result=quality_result,
                        errors=verifier_errors,
                    )
                    canonical_validation = validator.validate(
                        candidate=quality_result.normalized.parsed_json,
                        schema_validation=quality_result.schema_validation,
                    )
                    failed_stage = BANK_PROFILE_PARSE_V2_VERIFIER_STAGE
                else:
                    failed_stage = (
                        classify_bank_profile_v2_failed_stage(retry_errors)
                        or BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE
                    )
                break
            retry_result = adapter.retry_bank_profile_v2(
                user_text=user_text,
                previous_output=current_normalization_result.message_text,
                errors=retry_errors,
                benchmark=benchmark,
                dataset_package=dataset_package,
                stage=BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
                extracted_evidence=extraction_fields,
            )
            normalization_retry_results.append(retry_result)
            current_normalization_result = retry_result
        if failed_stage is None and not bool(canonical_validation.ready_for_runtime):
            failed_stage = (
                classify_bank_profile_v2_failed_stage(retry_errors)
                or BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE
            )

    deterministic_corrections = list((quality_result.metadata or {}).get("deterministic_corrections") or [])
    parser_quality = finalize_parser_quality_metadata(
        quality_result.metadata,
        canonical_pass_after_quality=bool(canonical_validation.ready_for_runtime),
        repair_invoked=bool(
            extraction_retry_results
            or normalization_retry_results
            or verifier_applied
            or evidence_hint_applied_fields
            or deterministic_corrections
        ),
    )
    original_profile = normalize_bank_profile({field_name: query_row[field_name] for field_name in BANK_FEATURE_ORDER})
    reconstructed_profile = None
    if bool(canonical_validation.ready_for_runtime) and isinstance(canonical_validation.runtime_request, dict):
        runtime_profile = dict(canonical_validation.runtime_request.get("profile") or {})
        if runtime_profile:
            reconstructed_profile = normalize_bank_profile(runtime_profile)

    exact_reconstruction = profiles_equal(original_profile, reconstructed_profile)
    mismatch_fields = diff_profile_fields(original_profile, reconstructed_profile) if reconstructed_profile is not None else list(BANK_FEATURE_ORDER)
    prediction_before_original = predict_single_profile(model, original_profile)
    prediction_before_reconstructed = predict_single_profile(model, reconstructed_profile)
    last_retry_result = (normalization_retry_results or extraction_retry_results)[-1] if (normalization_retry_results or extraction_retry_results) else None
    candidate_profile = extract_bank_profile_candidate_values(quality_result.v2_payload)
    field_semantic_mismatch = build_bank_profile_v2_field_semantic_mismatch(
        original_profile=original_profile,
        reconstructed_profile=reconstructed_profile,
        candidate_profile=candidate_profile,
        payload=quality_result.v2_payload,
        errors=dedupe_preserve_order(list(retry_errors) + list(verifier_errors)),
    )
    numeric_domain_violations = build_bank_profile_v2_numeric_domain_violations(
        errors=list(quality_result.schema_validation.errors) + list(retry_errors) + list(verifier_errors),
        payload=quality_result.v2_payload,
    )
    parser_status = None
    if isinstance(quality_result.normalized.parsed_json, dict):
        parser_status = quality_result.normalized.parsed_json.get("status")
    elif isinstance(extraction_payload, dict):
        parser_status = extraction_payload.get("status")
    normalization_payload = quality_result.v2_payload if isinstance(quality_result.v2_payload, dict) else None
    extraction_missing_fields = (
        _missing_bank_profile_v2_extraction_fields(extraction_payload=extraction_payload, benchmark=benchmark)
        if isinstance(extraction_payload, dict)
        else []
    )
    return {
        "case_id": str(query_row["case_id"]),
        "dataset": "bank",
        "fold_index": int(query_row["fold_index"]),
        "fold_name": str(query_row["fold_name"]),
        "query_pos": int(query_row["query_pos"]),
        "language_group": str(query_row["language_group"]),
        "language_label": str(query_row.get("language_label", query_row["language_group"])),
        "benchmark_key": benchmark_key,
        "benchmark_name": str(getattr(benchmark, "benchmark_name", benchmark_key)),
        "numeric_style": str(query_row.get("numeric_style", "digits")),
        "template_version": str(query_row.get("template_version", TEMPLATE_VERSION)),
        "user_text": user_text,
        "parser_status": parser_status,
        "final_stage": canonical_validation.final_stage,
        "failed_stage": failed_stage,
        "ready_for_runtime": bool(canonical_validation.ready_for_runtime),
        "repair_invoked": bool(
            extraction_retry_results
            or normalization_retry_results
            or verifier_applied
            or evidence_hint_applied_fields
            or deterministic_corrections
        ),
        "llm_retry_count": int(len(extraction_retry_results) + len(normalization_retry_results)),
        "deterministic_correction_count": int(len(deterministic_corrections)),
        "deterministic_corrections_json": json.dumps(
            deterministic_corrections,
            ensure_ascii=True,
            sort_keys=True,
        ),
        "evidence_hint_fallback_applied": bool(evidence_hint_applied_fields),
        "evidence_hint_applied_fields_json": json.dumps(evidence_hint_applied_fields, ensure_ascii=True),
        "evidence_hint_ambiguous_fields_json": json.dumps(evidence_hint_ambiguous_fields, ensure_ascii=True),
        "evidence_hint_candidates_json": json.dumps(extraction_hint_candidates, ensure_ascii=True, sort_keys=True),
        "evidence_extraction_missing_fields_json": json.dumps(extraction_missing_fields, ensure_ascii=True),
        "evidence_extraction_retry_count": int(len(extraction_retry_results)),
        "normalization_retry_count": int(len(normalization_retry_results)),
        "verifier_invoked": bool(verifier_result is not None or verifier_invocation_count > 0),
        "verifier_invocation_count": int(verifier_invocation_count),
        "verifier_correction_count": int(verifier_correction_count),
        "verifier_applied": bool(verifier_applied),
        "verifier_verdict": verifier_verdict,
        "schema_valid": bool(quality_result.schema_validation.is_valid),
        "exact_reconstruction": bool(exact_reconstruction),
        "original_profile_json": profile_json_text(original_profile),
        "original_profile_signature": bank_profile_signature(original_profile),
        "reconstructed_profile_json": profile_json_text(reconstructed_profile),
        "reconstructed_profile_signature": bank_profile_signature(reconstructed_profile),
        "reconstruction_mismatch_fields_json": json.dumps(mismatch_fields, ensure_ascii=True),
        "prediction_before_original": prediction_before_original,
        "prediction_before_reconstructed": prediction_before_reconstructed,
        "normalized_parse_json": json.dumps(
            serialize_normalized_parse_payload(
                quality_result.normalized.parsed_json,
                quality_result.field_provenance,
                parser_quality,
            ),
            ensure_ascii=True,
            sort_keys=True,
        )
        if quality_result.normalized.parsed_json is not None
        else None,
        "runtime_request_json": json.dumps(canonical_validation.runtime_request, ensure_ascii=True, sort_keys=True)
        if canonical_validation.runtime_request is not None
        else None,
        "missing_runtime_fields_json": json.dumps(list(canonical_validation.missing_runtime_fields), ensure_ascii=True),
        "confirmed_conflicts_json": json.dumps(list(canonical_validation.confirmed_conflicts), ensure_ascii=True),
        "provided_fields_json": json.dumps(list(canonical_validation.provided_fields), ensure_ascii=True),
        "canonical_errors_json": json.dumps(list(canonical_validation.errors), ensure_ascii=True),
        "schema_errors_json": json.dumps(list(quality_result.schema_validation.errors), ensure_ascii=True),
        "retry_errors_json": json.dumps(list(retry_errors), ensure_ascii=True),
        "verifier_errors_json": json.dumps(list(verifier_errors), ensure_ascii=True),
        "semantic_verifier_errors_json": json.dumps(list(verifier_errors), ensure_ascii=True),
        "field_semantic_mismatch_json": json.dumps(field_semantic_mismatch, ensure_ascii=True, sort_keys=True),
        "numeric_domain_violation_json": json.dumps(numeric_domain_violations, ensure_ascii=True, sort_keys=True),
        "parser_quality_json": json.dumps(parser_quality, ensure_ascii=True, sort_keys=True),
        "parse_v2_payload_json": json.dumps(normalization_payload, ensure_ascii=True, sort_keys=True)
        if normalization_payload is not None
        else None,
        "evidence_extraction_payload_json": json.dumps(extraction_payload, ensure_ascii=True, sort_keys=True)
        if extraction_payload is not None
        else None,
        "normalization_payload_json": json.dumps(normalization_payload, ensure_ascii=True, sort_keys=True)
        if normalization_payload is not None
        else None,
        "semantic_verifier_payload_json": json.dumps(verifier_payload, ensure_ascii=True, sort_keys=True)
        if verifier_payload is not None
        else None,
        "verifier_payload_json": json.dumps(verifier_payload, ensure_ascii=True, sort_keys=True)
        if verifier_payload is not None
        else None,
        "parser_api_error": extraction_result.api_error,
        "evidence_extraction_api_error": extraction_result.api_error,
        "normalization_api_error": None if current_normalization_result is None else current_normalization_result.api_error,
        "repair_api_error": None if last_retry_result is None else last_retry_result.api_error,
        "verifier_api_error": None if verifier_result is None else verifier_result.api_error,
        "parser_http_status_code": extraction_result.http_status_code,
        "evidence_extraction_http_status_code": extraction_result.http_status_code,
        "normalization_http_status_code": None if current_normalization_result is None else current_normalization_result.http_status_code,
        "repair_http_status_code": None if last_retry_result is None else last_retry_result.http_status_code,
        "verifier_http_status_code": None if verifier_result is None else verifier_result.http_status_code,
        "parser_failure_cause": extraction_result.failure_cause,
        "evidence_extraction_failure_cause": extraction_result.failure_cause,
        "normalization_failure_cause": None if current_normalization_result is None else current_normalization_result.failure_cause,
        "repair_failure_cause": None if last_retry_result is None else last_retry_result.failure_cause,
        "verifier_failure_cause": None if verifier_result is None else verifier_result.failure_cause,
        "raw_parser_output": current_extraction_result.message_text if normalization_payload is None else current_normalization_result.message_text,
        "evidence_extraction_output": current_extraction_result.message_text,
        "normalization_output": None if current_normalization_result is None else current_normalization_result.message_text,
        "repaired_output": None if last_retry_result is None else last_retry_result.message_text,
        "verifier_output": None if verifier_result is None else verifier_result.message_text,
        "request_latency_ms": finite_float(
            (extraction_result.derived_metrics or {}).get("request_latency_ms")
        ),
        "evidence_extraction_latency_ms": finite_float(
            (current_extraction_result.derived_metrics or {}).get("request_latency_ms")
        ),
        "normalization_latency_ms": None
        if current_normalization_result is None
        else finite_float((current_normalization_result.derived_metrics or {}).get("request_latency_ms")),
        "repair_latency_ms": None
        if last_retry_result is None
        else finite_float((last_retry_result.derived_metrics or {}).get("request_latency_ms")),
        "verifier_latency_ms": None
        if verifier_result is None
        else finite_float((verifier_result.derived_metrics or {}).get("request_latency_ms")),
    }


def force_bank_profile_v2_failure(*, quality_result, errors: list[str]):
    merged_errors = dedupe_preserve_order(
        list(quality_result.schema_validation.errors) + list(errors)
    )
    failed_validation = ValidationResult(
        is_valid=False,
        errors=tuple(merged_errors),
        unexpected_top_level_keys=quality_result.schema_validation.unexpected_top_level_keys,
        unexpected_cf_fields=quality_result.schema_validation.unexpected_cf_fields,
    )
    metadata = dict(quality_result.metadata or {})
    reason_codes = list(metadata.get("reason_codes") or [])
    reason_codes.append("v2_verifier_failed")
    flags = dict(metadata.get("flags") or {})
    flags["post_quality_schema_valid"] = False
    flags["still_failed_after_quality"] = True
    metadata["reason_codes"] = dedupe_preserve_order(reason_codes)
    metadata["flags"] = flags
    return replace(
        quality_result,
        schema_validation=failed_validation,
        metadata=metadata,
    )


def collect_bank_profile_v2_extraction_retry_errors(
    *,
    parser_result,
    extraction_payload,
    extraction_parse_errors: list[str],
    benchmark,
    user_text: str,
) -> list[str]:
    errors: list[str] = []
    if parser_result.api_error:
        errors.append(str(parser_result.api_error))
    errors.extend(str(error) for error in extraction_parse_errors)
    if isinstance(extraction_payload, dict):
        errors.extend(
            validate_bank_profile_parse_v2_extraction_payload(
                extraction_payload,
                benchmark_spec=benchmark,
                user_text=user_text,
            )
        )
        missing_fields = _missing_bank_profile_v2_extraction_fields(
            extraction_payload=extraction_payload,
            benchmark=benchmark,
        )
        if missing_fields:
            errors.append(
                "Missing extracted fields after evidence extraction: "
                + ", ".join(str(field) for field in missing_fields)
            )
    elif not errors:
        errors.append("Evidence extraction did not produce a usable payload.")
    return dedupe_preserve_order(errors)


def _missing_bank_profile_v2_extraction_fields(*, extraction_payload: dict[str, Any], benchmark) -> list[str]:
    required_fields = _required_bank_profile_v2_fields(benchmark)
    fields = extraction_payload.get("fields")
    present_fields = set(fields) if isinstance(fields, dict) else set()
    declared_missing = extraction_payload.get("missing_fields")
    missing_fields = [field_name for field_name in required_fields if field_name not in present_fields]
    if isinstance(declared_missing, list):
        for field_name in declared_missing:
            field_name = str(field_name)
            if field_name in required_fields and field_name not in missing_fields:
                missing_fields.append(field_name)
    return missing_fields


def _required_bank_profile_v2_fields(benchmark) -> list[str]:
    benchmark_fields = [
        str(field.name)
        for field in getattr(benchmark, "target_cf_fields", ())
        if isinstance(getattr(field, "name", None), str)
    ]
    ordered = [field_name for field_name in BANK_FEATURE_ORDER if field_name in benchmark_fields]
    ordered.extend(field_name for field_name in benchmark_fields if field_name not in ordered)
    return ordered or list(BANK_FEATURE_ORDER)


def build_bank_profile_v2_evidence_hint_candidates(
    *,
    user_text: str,
    field_names: list[str] | tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    field_set = {str(field_name) for field_name in field_names}
    for field_name in BANK_FEATURE_ORDER:
        if field_name not in field_set:
            continue
        field_candidates = _dedupe_bank_profile_v2_hint_candidates(
            _bank_profile_v2_numeric_hint_candidates(user_text, field_name)
            + _bank_profile_v2_boolean_hint_candidates(user_text, field_name)
        )
        if field_candidates:
            candidates[field_name] = field_candidates
    return candidates


def merge_bank_profile_v2_evidence_hints(
    left: dict[str, list[dict[str, Any]]] | None,
    right: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for source in (left or {}, right or {}):
        for field_name, candidates in source.items():
            merged.setdefault(str(field_name), [])
            merged[str(field_name)].extend(candidate for candidate in candidates if isinstance(candidate, dict))
    return {
        field_name: _dedupe_bank_profile_v2_hint_candidates(candidates)
        for field_name, candidates in merged.items()
        if candidates
    }


def apply_bank_profile_v2_evidence_hint_fallback(
    *,
    extraction_payload: dict[str, Any],
    benchmark,
    user_text: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    payload = deepcopy(extraction_payload)
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return payload, [], []
    missing_fields = _missing_bank_profile_v2_extraction_fields(
        extraction_payload=payload,
        benchmark=benchmark,
    )
    hints = build_bank_profile_v2_evidence_hint_candidates(
        user_text=user_text,
        field_names=missing_fields,
    )
    applied_fields: list[str] = []
    ambiguous_fields: list[str] = []
    for field_name in missing_fields:
        candidates = list(hints.get(field_name) or [])
        if len(candidates) > 1:
            ambiguous_fields.append(field_name)
            continue
        if len(candidates) != 1:
            continue
        evidence_quote = str(candidates[0].get("evidence_quote") or "")
        if not evidence_quote or evidence_quote not in user_text:
            continue
        fields[field_name] = {
            "evidence_quote": evidence_quote,
            "confidence": 0.85,
            "reason": "exact-span evidence hint fallback after extraction retries",
        }
        applied_fields.append(field_name)
    if not applied_fields:
        return payload, [], ambiguous_fields
    payload["fields"] = fields
    remaining_missing = [
        field_name
        for field_name in _required_bank_profile_v2_fields(benchmark)
        if field_name not in fields
    ]
    payload["missing_fields"] = remaining_missing
    conflicts = payload.get("conflicts")
    if not remaining_missing and (not isinstance(conflicts, list) or not conflicts):
        payload["status"] = "complete"
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
    notes.append("Applied exact-span evidence hint fallback for: " + ", ".join(applied_fields))
    payload["notes"] = [str(note) for note in notes]
    return payload, applied_fields, ambiguous_fields


def _bank_profile_v2_numeric_hint_candidates(user_text: str, field_name: str) -> list[dict[str, Any]]:
    patterns = _BANK_PROFILE_V2_NUMERIC_HINT_PATTERNS.get(field_name, ())
    candidates: list[dict[str, Any]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, user_text, flags=re.IGNORECASE):
            evidence_quote = user_text[match.start():match.end()].strip()
            if not evidence_quote:
                continue
            candidates.append(
                {
                    "evidence_quote": evidence_quote,
                    "source": "alias_nearby_text",
                    "start": int(match.start()),
                    "end": int(match.end()),
                }
            )
    return candidates


def _bank_profile_v2_boolean_hint_candidates(user_text: str, field_name: str) -> list[dict[str, Any]]:
    phrases = _BANK_PROFILE_V2_BOOLEAN_HINT_PHRASES.get(field_name, ())
    candidates: list[dict[str, Any]] = []
    for phrase in phrases:
        for match in re.finditer(re.escape(phrase), user_text, flags=re.IGNORECASE):
            evidence_quote = user_text[match.start():match.end()].strip()
            if not evidence_quote:
                continue
            candidates.append(
                {
                    "evidence_quote": evidence_quote,
                    "source": "boolean_phrase",
                    "start": int(match.start()),
                    "end": int(match.end()),
                }
            )
    return candidates


def _dedupe_bank_profile_v2_hint_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: len(str(item.get("evidence_quote") or "")),
        reverse=True,
    ):
        quote = str(candidate.get("evidence_quote") or "").strip()
        if not quote:
            continue
        quote_key = quote.casefold()
        if quote_key in seen_quotes:
            continue
        if any(quote_key in str(existing.get("evidence_quote") or "").casefold() for existing in unique):
            continue
        seen_quotes.add(quote_key)
        unique.append(dict(candidate, evidence_quote=quote))
    return sorted(unique, key=lambda item: (int(item.get("start") or 0), str(item.get("evidence_quote") or "")))


_BANK_PROFILE_V2_NUMERIC_HINT_PATTERNS: dict[str, tuple[str, ...]] = {
    "Income": (
        r"\bthu nhap nam\s+[^;,.]+",
        r"\bannual income\s+[^;,.]+",
        r"\bincome\s+[^;,.]+",
    ),
    "Family": (
        r"\bgia dinh\s+[^;,.]+",
        r"\bfamily size\s+[^;,.]+",
        r"\bfamily\s+[^;,.]+",
    ),
    "CCAvg": (
        r"\bchi tieu the trung binh\s+[^;,.]+",
        r"\bchi tieu the\s+[^;,.]+",
        r"\baverage credit-card spending\s+[^;,.]+",
        r"\bmonthly card spend\s+[^;,.]+",
        r"\bmonthly CCAvg\s+[^;,.]+",
        r"\bCCAvg\s+[^;,.]+",
    ),
    "Education": (
        r"\bhoc van muc\s+[^;,.]+",
        r"\bhoc van\s+[^;,.]+",
        r"\beducation level\s+[^;,.]+",
    ),
    "Mortgage": (
        r"\bthe chap\s+[^;,.]+",
        r"\bmortgage\s+[^;,.]+",
    ),
}


_BANK_PROFILE_V2_BOOLEAN_HINT_PHRASES: dict[str, tuple[str, ...]] = {
    "SecuritiesAccount": (
        "toi co tai khoan chung khoan",
        "toi khong co tai khoan chung khoan",
        "hien tai co tai khoan chung khoan",
        "hien tai khong co tai khoan chung khoan",
        "I have a securities account",
        "I do not have a securities account",
        "currently securities account yes",
        "currently securities account no",
        "SecuritiesAccount yes",
        "SecuritiesAccount no",
    ),
    "CDAccount": (
        "toi co CD account",
        "toi khong co CD account",
        "hien tai co CD account",
        "hien tai khong co CD account",
        "I have a CD account",
        "I do not have a CD account",
        "currently CD account yes",
        "currently CD account no",
        "CDAccount co",
        "CDAccount khong",
        "CDAccount yes",
        "CDAccount no",
    ),
    "Online": (
        "toi dung online banking",
        "toi khong dung online banking",
        "hien tai co online banking",
        "hien tai khong co online banking",
        "I use online banking",
        "I do not use online banking",
        "currently online banking yes",
        "currently online banking no",
        "online banking yes",
        "online banking no",
    ),
    "CreditCard": (
        "toi co the tin dung",
        "toi khong co the tin dung",
        "hien tai co the tin dung",
        "hien tai khong co the tin dung",
        "I have a credit card",
        "I do not have a credit card",
        "currently credit card yes",
        "currently credit card no",
        "the tin dung co",
        "the tin dung khong",
        "credit card yes",
        "credit card no",
    ),
}


def collect_bank_profile_v2_normalization_retry_errors(
    *,
    parser_result,
    quality_result,
    canonical_validation,
) -> list[str]:
    errors: list[str] = []
    if parser_result.api_error:
        errors.append(str(parser_result.api_error))
    if quality_result.normalized.parse_error:
        errors.append(str(quality_result.normalized.parse_error))
    errors.extend(str(error) for error in quality_result.schema_validation.errors)
    errors.extend(str(error) for error in canonical_validation.errors)
    if canonical_validation.missing_runtime_fields:
        errors.append(
            "Missing runtime fields after semantic parse: "
            + ", ".join(str(field) for field in canonical_validation.missing_runtime_fields)
        )
    if not canonical_validation.ready_for_runtime and not errors:
        errors.append("Candidate is not ready for runtime handoff.")
    return dedupe_preserve_order(errors)


def should_attempt_bank_profile_v2_stage_retry(*, parser_result, errors: list[str]) -> bool:
    if not errors:
        return False
    if parser_result.api_error and not str(parser_result.message_text or "").strip():
        return False
    return bool(str(parser_result.message_text or "").strip())


def extract_bank_profile_candidate_values(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return None
    candidate: dict[str, Any] = {}
    for field_name, field_payload in fields.items():
        if not isinstance(field_payload, dict) or "value" not in field_payload:
            continue
        candidate[str(field_name)] = field_payload.get("value")
    return candidate or None


def build_bank_profile_v2_field_semantic_mismatch(
    *,
    original_profile: dict[str, Any],
    reconstructed_profile: dict[str, Any] | None,
    candidate_profile: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    errors: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    observed_profile = reconstructed_profile or candidate_profile or {}
    field_names = set(observed_profile) | {
        field_name for field_name in BANK_FEATURE_ORDER if _find_field_errors(errors, field_name)
    }
    for field_name in BANK_FEATURE_ORDER:
        if field_name not in field_names:
            continue
        expected_value = original_profile.get(field_name)
        observed_value = observed_profile.get(field_name)
        field_errors = _find_field_errors(errors, field_name)
        if observed_value is not None and bank_values_equal(field_name, expected_value, observed_value) and not field_errors:
            continue
        evidence_quote = _payload_quote_for_field(payload, field_name)
        records.append(
            {
                "field_name": field_name,
                "expected_value": expected_value,
                "observed_value": observed_value,
                "evidence_quote": evidence_quote,
                "failure_family": _infer_failure_family(
                    field_name=field_name,
                    expected_value=expected_value,
                    observed_value=observed_value,
                    evidence_quote=evidence_quote,
                    field_errors=field_errors,
                ),
                "errors": field_errors,
            }
        )
    return records


def build_bank_profile_v2_numeric_domain_violations(
    *,
    errors: list[str],
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for error in errors:
        match = FIELD_VALUE_ERROR_RE.search(str(error))
        if not match:
            continue
        field_name = str(match.group(1))
        detail = str(match.group(2))
        records.append(
            {
                "field_name": field_name,
                "evidence_quote": _payload_quote_for_field(payload, field_name),
                "error": str(error),
                "failure_family": _infer_failure_family(
                    field_name=field_name,
                    expected_value=None,
                    observed_value=None,
                    evidence_quote=_payload_quote_for_field(payload, field_name),
                    field_errors=[detail],
                ),
            }
        )
    return records


def build_parse_failure_family_breakdown(parse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in parse_rows:
        language_label = str(row.get("language_label") or row.get("language_group") or "unknown")
        for column_name in ("field_semantic_mismatch_json", "numeric_domain_violation_json"):
            for payload in parse_json_list(row.get(column_name)):
                if not isinstance(payload, dict):
                    continue
                field_name = str(payload.get("field_name") or "unknown")
                failure_family = str(payload.get("failure_family") or "unknown")
                key = (language_label, field_name, failure_family)
                counts[key] = counts.get(key, 0) + 1
    return [
        {
            "language_label": language_label,
            "field_name": field_name,
            "failure_family": failure_family,
            "count": int(count),
        }
        for (language_label, field_name, failure_family), count in sorted(counts.items())
    ]


def _payload_quote_for_field(payload: dict[str, Any] | None, field_name: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return None
    field_payload = fields.get(field_name)
    if not isinstance(field_payload, dict):
        return None
    evidence_quote = field_payload.get("evidence_quote")
    return str(evidence_quote) if isinstance(evidence_quote, str) and evidence_quote.strip() else None


def _find_field_errors(errors: list[str], field_name: str) -> list[str]:
    field_errors: list[str] = []
    lowered_field = field_name.lower()
    for error in errors:
        error_text = str(error)
        error_lower = error_text.lower()
        if f"fields.{field_name.lower()}" in error_lower:
            field_errors.append(error_text)
            continue
        if f" {lowered_field}" in error_lower and "missing runtime fields" in error_lower:
            field_errors.append(error_text)
    return field_errors


def _infer_failure_family(
    *,
    field_name: str,
    expected_value: Any,
    observed_value: Any,
    evidence_quote: str | None,
    field_errors: list[str],
) -> str:
    error_blob = " ".join(field_errors).lower()
    quote = str(evidence_quote or "").lower()
    if "evidence_quote" in error_blob or "exact substring" in error_blob or "overlap" in error_blob:
        return "quote_alignment_error"
    if "integer-valued" in error_blob or "non-negative integer" in error_blob:
        return "non_integral_integerlike_output"
    if field_name == "CCAvg" and ("decimal" in error_blob or "phay" in quote or "point" in quote):
        return "decimal_point_corruption"
    if field_name == "Income" and quote.startswith("thu nhap nam "):
        if expected_value is not None and observed_value is not None:
            try:
                if int(float(expected_value)) != int(float(observed_value)):
                    return "field_aware_number_prefix_collision"
            except (TypeError, ValueError):
                pass
        if "support" in error_blob:
            return "field_aware_number_prefix_collision"
    if quote and ("tram" in quote or "hundred" in quote):
        if expected_value is not None and observed_value is not None:
            try:
                if int(float(expected_value)) >= 100 and int(float(observed_value)) < 100:
                    return "hundreds_truncation"
            except (TypeError, ValueError):
                pass
        if "support" in error_blob:
            return "hundreds_truncation"
    if quote and any(token in quote for token in ["muoi", "teen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]):
        if expected_value is not None and observed_value is not None:
            try:
                if int(float(expected_value)) != int(float(observed_value)):
                    return "tens_teen_drift"
            except (TypeError, ValueError):
                pass
        if "support" in error_blob:
            return "tens_teen_drift"
    if expected_value is not None and observed_value is not None:
        try:
            expected_int = int(float(expected_value))
            observed_int = int(float(observed_value))
            if len(str(abs(observed_int))) > len(str(abs(expected_int))):
                return "token_concatenation_or_spill"
        except (TypeError, ValueError):
            pass
    if "support" in error_blob:
        return "token_concatenation_or_spill"
    return "quote_alignment_error" if field_errors else "token_concatenation_or_spill"


def dedupe_preserve_order(values: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def predict_single_profile(model, profile: dict[str, Any] | None) -> int | None:
    if not isinstance(profile, dict):
        return None
    frame = pd.DataFrame([normalize_bank_profile(profile)], columns=BANK_FEATURE_ORDER)
    pred = model.predict(frame)
    if len(pred) == 0:
        return None
    return int(pred[0])


def load_parse_frame(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "parsed_requests.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing parse artifact: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def validate_parse_checkpoint_matches_query(*, query_df: pd.DataFrame, parse_df: pd.DataFrame) -> None:
    if parse_df.empty:
        raise RuntimeError("Parse checkpoint is empty. Rerun --stage parse for this 250-case Bank set.")
    if "case_id" not in parse_df.columns:
        raise RuntimeError("Parse checkpoint is missing case_id. Rerun --stage parse for this 250-case Bank set.")
    query_case_ids = {str(value) for value in query_df["case_id"].tolist()}
    parse_case_ids = {str(value) for value in parse_df["case_id"].tolist()}
    missing = sorted(query_case_ids - parse_case_ids)
    extra = sorted(parse_case_ids - query_case_ids)
    if missing or extra:
        raise RuntimeError(
            "Parse checkpoint case_id set does not match bank_250_queries.csv. "
            f"missing={missing[:5]} extra={extra[:5]}. Rerun --stage parse for this exact 250-case set."
        )
    if "user_text" not in parse_df.columns or "user_text_sha256" not in query_df.columns:
        return
    parse_text_by_case = {
        str(row["case_id"]): str(row.get("user_text", ""))
        for row in parse_df.drop_duplicates(subset=["case_id"]).to_dict(orient="records")
    }
    stale_cases = []
    for row in query_df.to_dict(orient="records"):
        case_id = str(row["case_id"])
        if sha256_text(parse_text_by_case.get(case_id, "")) != str(row["user_text_sha256"]):
            stale_cases.append(case_id)
    if stale_cases:
        raise RuntimeError(
            "Parse checkpoint text does not match bank_250_queries.csv. "
            f"stale_cases={stale_cases[:5]}. Rerun --stage parse after changing the 250-case template."
        )


def parse_row_selected_for_eval(row: dict[str, Any] | None, *, scope: str) -> bool:
    scope_name = str(scope)
    if scope_name == "all":
        return True
    if row is None:
        return False
    if scope_name == "ready":
        return bool(as_bool(row.get("ready_for_runtime")))
    if scope_name == "accepted":
        return bool(is_parse_row_accepted(row))
    raise ValueError(f"Unsupported parse eval scope: {scope_name}")


def classify_parse_eval_drop_reason(row: dict[str, Any] | None, *, scope: str) -> str:
    scope_name = str(scope)
    if scope_name == "all":
        return ""
    if row is None:
        return "missing_parse_record"
    if scope_name == "ready":
        return "" if as_bool(row.get("ready_for_runtime")) else "not_ready_for_runtime"
    if scope_name != "accepted":
        raise ValueError(f"Unsupported parse eval scope: {scope_name}")
    if not as_bool(row.get("ready_for_runtime")):
        return "not_ready_for_runtime"
    if not as_bool(row.get("exact_reconstruction")):
        return "not_exact_reconstruction"
    if any(
        _has_text(row.get(field_name))
        for field_name in ("parser_api_error", "repair_api_error", "verifier_api_error")
    ):
        return "parser_api_error"
    if parse_json_list(row.get("schema_errors_json")):
        return "schema_error"
    if parse_json_list(row.get("verifier_errors_json")):
        return "verifier_error"
    return ""


def build_eval_case_selection_frame(*, query_df: pd.DataFrame, parse_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    parse_by_case = (
        {
            str(row["case_id"]): row
            for row in parse_df.to_dict(orient="records")
        }
        if not parse_df.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for record in query_df.to_dict(orient="records"):
        case_id = str(record["case_id"])
        parse_row = parse_by_case.get(case_id)
        included = parse_row_selected_for_eval(parse_row, scope=scope)
        drop_reason = "" if included else classify_parse_eval_drop_reason(parse_row, scope=scope)
        rows.append(
            {
                "case_id": case_id,
                "fold_index": int(record["fold_index"]),
                "fold_name": str(record["fold_name"]),
                "query_pos": int(record["query_pos"]),
                "language_group": str(record["language_group"]),
                "language_label": str(record["language_label"]),
                "benchmark_key": str(record["benchmark_key"]),
                "parse_eval_scope": str(scope),
                "eval_included": bool(included),
                "eval_drop_reason": str(drop_reason),
                "ready_for_runtime": False if parse_row is None else bool(as_bool(parse_row.get("ready_for_runtime"))),
                "exact_reconstruction": False if parse_row is None else bool(as_bool(parse_row.get("exact_reconstruction"))),
                "parser_status": None if parse_row is None else parse_row.get("parser_status"),
                "final_stage": None if parse_row is None else parse_row.get("final_stage"),
            }
        )
    return pd.DataFrame(rows)


def write_eval_case_selection_artifact(*, out_dir: Path, selection_df: pd.DataFrame) -> None:
    selection_df.to_csv(out_dir / "evaluation_case_selection.csv", index=False)


def filter_eval_query_frame(*, query_df: pd.DataFrame, selection_df: pd.DataFrame) -> pd.DataFrame:
    included_case_ids = {
        str(row["case_id"])
        for row in selection_df.loc[selection_df["eval_included"].fillna(False).astype(bool), ["case_id"]].to_dict(orient="records")
    }
    return query_df.loc[
        query_df["case_id"].astype(str).isin(included_case_ids)
    ].copy().sort_values(by=["fold_index", "query_pos"], kind="mergesort").reset_index(drop=True)


def prepare_ufce_context(
    *,
    config_profile: str,
    bundle_mode: str,
    mi_k_token: str,
    no_cf: int,
    contprox_metric: str,
    mi_feature_scope: str,
) -> dict[str, Any]:
    mod01b, ffmod = load_part1_modules()
    dataset = "bank"
    datasetdf = pd.read_csv(BANK_SOURCE_PATH)
    out = mod01b.classify_dataset_getModel(datasetdf, data_name=dataset)
    scaler = None
    if len(out) == 8:
        bb_model, _lr_mean, _lr_std, _xtest, xtrain, x_all, _y, datasetdf = out
    elif len(out) == 9:
        bb_model, _lr_mean, _lr_std, _xtest, xtrain, x_all, _y, datasetdf, scaler = out
    else:
        raise ValueError(f"Unexpected classify_dataset_getModel return length: {len(out)}")

    (
        features,
        catf,
        numf,
        author_uf,
        author_f2change,
        _outcome_label,
        desired_outcome,
        _nbr_features,
        protectf,
        _data_lab0,
        data_lab1,
    ) = mod01b.get_dataset_constraints(dataset, datasetdf)

    bundle = mod01b.resolve_bundle_config(
        dataset=dataset,
        args=SimpleNamespace(bundle_mode=str(bundle_mode)),
        author_uf=author_uf,
        author_f2change=author_f2change,
    )
    uf = dict(bundle.uf)
    f2change = list(bundle.f2change)
    step = dict(bundle.step)
    cfg = resolve_bridge_ufce_ff_config(mod01b, dataset, str(config_profile))

    mi_fp_raw = mod01b.normalize_mi_feature_pairs(mod01b.ufc.get_top_MI_features(x_all, features), features)
    mi_fp = ffmod.filter_mi_pairs_by_scope(
        mi_fp_raw,
        uf,
        step,
        f2change,
        str(mi_feature_scope),
    )
    mi_k = parse_mi_k_token(mi_k_token)
    mod01b.cfmethods.initUFCE(
        radius=cfg["radius"],
        n_neighbors=cfg["n_neighbors"],
        contprox_metric=str(contprox_metric),
        min_act=cfg["min_act"],
        min_feas=cfg["min_feas"],
        atol=1e-5,
    )
    mod01b.eval_module.ufc = mod01b.cfmethods.ufc
    active_ufc = mod01b._active_ufce_instance()
    bundle_meta = mod01b.effective_config_record(
        dataset=dataset,
        cfg=cfg,
        bundle=bundle,
        runtime_profile=str(config_profile),
    )
    return {
        "mod01b": mod01b,
        "ffmod": ffmod,
        "dataset": dataset,
        "bb_model": bb_model,
        "xtrain": xtrain,
        "x_all": x_all,
        "data_lab1": data_lab1,
        "features": list(features),
        "catf": list(catf),
        "numf": list(numf),
        "uf": uf,
        "f2change": f2change,
        "protectf": list(protectf),
        "desired_outcome": float(desired_outcome),
        "step": step,
        "cfg": dict(cfg),
        "bundle_mode": str(bundle.effective_bundle_mode),
        "bundle_cfg": dict(bundle.bundle_cfg),
        "bundle_meta": bundle_meta,
        "scaler": scaler,
        "mi_fp": [list(pair) for pair in mi_fp],
        "mi_k": mi_k,
        "no_cf": int(no_cf),
        "contprox_metric": str(contprox_metric),
        "mi_feature_scope": str(mi_feature_scope),
        "movie_distance_scaler": None,
        "active_ufc": active_ufc,
    }


def build_variant_fold_inputs(query_df: pd.DataFrame, parse_df: pd.DataFrame | None, variant: str) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    parse_by_case = {}
    if parse_df is not None and not parse_df.empty:
        parse_by_case = {
            str(row["case_id"]): row
            for row in parse_df.to_dict(orient="records")
        }

    for fold_name, fold_group in query_df.groupby("fold_name", sort=True):
        fold_rows = fold_group.sort_values(by="query_pos", kind="mergesort")
        local_records: list[dict[str, Any]] = []
        fold_payloads: list[dict[str, Any]] = []
        for row in fold_rows.to_dict(orient="records"):
            parse_record = parse_by_case.get(str(row["case_id"]))
            if variant == "nl_first":
                if not parse_record or not bool(parse_record.get("ready_for_runtime")):
                    continue
                runtime_profile = parse_json_dict(parse_record.get("reconstructed_profile_json"))
                if not isinstance(runtime_profile, dict):
                    continue
                profile = normalize_bank_profile(runtime_profile)
            else:
                profile = normalize_bank_profile({field_name: row[field_name] for field_name in BANK_FEATURE_ORDER})
            local_records.append({**row, **({"parse_record": parse_record} if parse_record else {})})
            fold_payloads.append(profile)

        if not fold_payloads:
            continue
        fold_df = pd.DataFrame(fold_payloads, columns=BANK_FEATURE_ORDER)
        inputs.append(
            {
                "fold_name": str(fold_name),
                "fold_index": int(fold_rows.iloc[0]["fold_index"]),
                "fold_df": fold_df,
                "local_records": local_records,
            }
        )
    return inputs


def run_part1_variant(
    *,
    context: dict[str, Any],
    fold_inputs: list[dict[str, Any]],
    variant: str,
    progress_enabled: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iterable = progress_iter(
        fold_inputs,
        enabled=progress_enabled,
        desc=f"{variant} UFCE-FF",
        unit="fold",
        total=len(fold_inputs),
    )
    for fold_input in iterable:
        fold_df = fold_input["fold_df"]
        fold_name = str(fold_input["fold_name"])
        local_records = list(fold_input["local_records"])
        fr = context["mod01b"].run_one_fold(
            dataset=str(context["dataset"]),
            fold_df=fold_df,
            x_all=context["x_all"],
            xtest=fold_df.copy(),
            xtrain=context["xtrain"],
            data_lab1=context["data_lab1"],
            features=list(context["features"]),
            catf=list(context["catf"]),
            numf=list(context["numf"]),
            uf=dict(context["uf"]),
            f2change=list(context["f2change"]),
            protectf=list(context["protectf"]),
            bb_model=context["bb_model"],
            desired_outcome=float(context["desired_outcome"]),
            mi_fp=[list(pair) for pair in context["mi_fp"]],
            no_cf=int(context["no_cf"]),
            step=dict(context["step"]),
            fold_name=fold_name,
            scaler=context["scaler"],
            flip_filter_enabled=bool(context["cfg"]["ufce_flip_filter"]),
            selection_policy=str(context["cfg"]["selection_policy"]),
            movie_distance_scaler=context["movie_distance_scaler"],
            fold_index=int(fold_input["fold_index"]),
            debug=0,
            prox_euc_contract_debug=0,
            contract_debug_fold=0,
            contract_debug_pos=0,
            contract_debug_method="UFCE2",
            diagnostics_enabled=True,
            diagnostics_top_k=max(100, int(context["no_cf"])),
            mi_top_k=context["mi_k"],
            cfg=dict(context["cfg"]),
            bundle_cfg=dict(context["bundle_cfg"]),
            bundle_meta=dict(context["bundle_meta"]),
        )
        rows.extend(
            extract_variant_rows(
                context=context,
                fold_name=fold_name,
                fold_df=fold_df,
                local_records=local_records,
                fold_result=fr,
                variant=variant,
            )
        )
    return rows


def extract_variant_rows(
    *,
    context: dict[str, Any],
    fold_name: str,
    fold_df: pd.DataFrame,
    local_records: list[dict[str, Any]],
    fold_result,
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ffmod = context["ffmod"]
    feature_order = list(context["features"])
    expected_pairs = {(int(record["query_pos"]), method): record for record in local_records for method in METHODS}
    seen_pairs: set[tuple[int, str]] = set()
    for method in METHODS:
        trace_rows = list((fold_result.trace_payload or {}).get(method, []))
        method_runtime_ms = float(ffmod.finite_float((fold_result.times or {}).get(method, 0.0)) or 0.0) * 1000.0
        for trace_row in trace_rows:
            local_pos = int(trace_row.get("instance_pos", 0))
            if local_pos < 0 or local_pos >= len(local_records):
                continue
            record = local_records[local_pos]
            seen_pairs.add((int(record["query_pos"]), method))
            input_profile = normalize_bank_profile(fold_df.iloc[local_pos].to_dict())
            input_frame = pd.DataFrame([input_profile], columns=feature_order)
            raw_pool_df = trace_row.get("generated_candidates_df")
            if not isinstance(raw_pool_df, pd.DataFrame):
                raw_pool_df = pd.DataFrame(columns=feature_order)
            else:
                raw_pool_df = raw_pool_df.loc[:, [column for column in feature_order if column in raw_pool_df.columns]].copy().reset_index(drop=True)
            selected_df = trace_row.get("selected_candidates_df")
            if not isinstance(selected_df, pd.DataFrame):
                selected_df = pd.DataFrame(columns=feature_order)
            else:
                selected_df = selected_df.loc[:, [column for column in feature_order if column in selected_df.columns]].copy().reset_index(drop=True)
            raw_count = int(len(raw_pool_df))
            unique_count = int(len(raw_pool_df.drop_duplicates())) if raw_count > 0 else 0
            flip_pool_df, predict_call_count, batch_predict_row_count, batch_predict_ms = ffmod.audit_flipping_pool(
                context["bb_model"],
                raw_pool_df,
                context["desired_outcome"],
                feature_order,
            )
            flip_count = int(len(flip_pool_df))
            prediction_before = ffmod.predict_label(context["bb_model"], input_frame, feature_order)
            selected_profile = None
            prediction_after = None
            selected_exists = not selected_df.empty
            selected_flip = False
            if selected_exists:
                selected_profile = normalize_bank_profile(selected_df.iloc[0].to_dict())
                prediction_after = ffmod.predict_label(
                    context["bb_model"],
                    pd.DataFrame([selected_profile], columns=feature_order),
                    feature_order,
                )
                selected_flip = bool(prediction_after == int(context["desired_outcome"]))
            metrics = ffmod._empty_metrics()
            if selected_exists and selected_flip and selected_profile is not None:
                metrics = ffmod.compute_candidate_metrics(
                    mod01b=context["mod01b"],
                    active_ufc=context["active_ufc"],
                    factual_row=input_frame,
                    candidate_row=pd.DataFrame([selected_profile], columns=feature_order),
                    xtrain=context["xtrain"],
                    features=feature_order,
                    numf=context["numf"],
                    catf=context["catf"],
                    f2change=context["f2change"],
                    uf=context["uf"],
                    bb_model=context["bb_model"],
                    desired_outcome=context["desired_outcome"],
                    movie_distance_scaler=context["movie_distance_scaler"],
                )
            failure_code, failure_detail = ffmod.classify_failure(
                raw_count=raw_count,
                flip_count=flip_count,
                public_selected_exists=selected_exists,
                public_selected_flip=selected_flip,
                flip_rank_rows=pd.DataFrame(),
            )
            if prediction_before == int(context["desired_outcome"]):
                status = "no_cf_needed"
            elif selected_exists and selected_flip:
                status = "released_cf"
            else:
                status = "no_valid_cf"
            parse_record = record.get("parse_record")
            rows.append(
                make_result_row(
                    case_record=record,
                    parse_record=parse_record,
                    variant=variant,
                    method=method,
                    status=status,
                    prediction_before=prediction_before,
                    prediction_after=prediction_after,
                    raw_candidate_count=raw_count,
                    unique_candidate_count=unique_count,
                    flip_candidate_count=flip_count,
                    failure_code=failure_code,
                    failure_detail=failure_detail,
                    selected_profile=selected_profile,
                    input_profile=input_profile,
                    metrics=metrics,
                    predict_call_count=int(predict_call_count),
                    batch_predict_row_count=int(batch_predict_row_count),
                    batch_predict_ms=float(batch_predict_ms),
                    method_runtime_ms=method_runtime_ms,
                )
            )
    for (query_pos, method), record in expected_pairs.items():
        if (query_pos, method) in seen_pairs:
            continue
        parse_record = record.get("parse_record")
        rows.append(
            make_result_row(
                case_record=record,
                parse_record=parse_record,
                variant=variant,
                method=method,
                status="error",
                prediction_before=None if variant == "nl_first" else int(record.get("source_prediction_label", 0)),
                prediction_after=None,
                raw_candidate_count=0,
                unique_candidate_count=0,
                flip_candidate_count=0,
                failure_code="TRACE_MISSING",
                failure_detail=f"Missing trace row for {fold_name}:{query_pos}:{method}",
                selected_profile=None,
                input_profile=None,
                metrics=context["ffmod"]._empty_metrics(),
                predict_call_count=0,
                batch_predict_row_count=0,
                batch_predict_ms=0.0,
                method_runtime_ms=0.0,
            )
        )
    return rows


def make_result_row(
    *,
    case_record: dict[str, Any],
    parse_record: dict[str, Any] | None,
    variant: str,
    method: str,
    status: str,
    prediction_before: int | None,
    prediction_after: int | None,
    raw_candidate_count: int,
    unique_candidate_count: int,
    flip_candidate_count: int,
    failure_code: str,
    failure_detail: str,
    selected_profile: dict[str, Any] | None,
    input_profile: dict[str, Any] | None,
    metrics: dict[str, Any],
    predict_call_count: int,
    batch_predict_row_count: int,
    batch_predict_ms: float,
    method_runtime_ms: float,
) -> dict[str, Any]:
    original_profile = normalize_bank_profile({field_name: case_record[field_name] for field_name in BANK_FEATURE_ORDER})
    active_input_profile = input_profile if isinstance(input_profile, dict) else (parse_json_dict(parse_record.get("reconstructed_profile_json")) if parse_record else original_profile if variant == "direct" else None)
    exact_reconstruction = True if variant == "direct" else bool(parse_record and parse_record.get("exact_reconstruction"))
    return {
        "case_id": str(case_record["case_id"]),
        "dataset": "bank",
        "fold_index": int(case_record["fold_index"]),
        "fold_name": str(case_record["fold_name"]),
        "query_pos": int(case_record["query_pos"]),
        "language_group": str(case_record["language_group"]),
        "language_label": str(case_record["language_label"]),
        "benchmark_key": str(case_record["benchmark_key"]),
        "variant": variant,
        "method": method,
        "status": status,
        "parse_ready": int(bool(parse_record and parse_record.get("ready_for_runtime"))) if variant == "nl_first" else 1,
        "exact_reconstruction": int(bool(exact_reconstruction)),
        "prediction_before": prediction_before,
        "prediction_after": prediction_after,
        "original_profile_json": profile_json_text(original_profile),
        "original_profile_signature": bank_profile_signature(original_profile),
        "input_profile_json": profile_json_text(active_input_profile),
        "input_profile_signature": bank_profile_signature(active_input_profile),
        "selected_profile_json": profile_json_text(selected_profile),
        "selected_profile_signature": bank_profile_signature(selected_profile),
        "selected_changed_fields_json": json.dumps(diff_profile_fields(active_input_profile, selected_profile), ensure_ascii=True)
        if isinstance(active_input_profile, dict) and isinstance(selected_profile, dict)
        else json.dumps([], ensure_ascii=True),
        "raw_candidate_count": int(raw_candidate_count),
        "unique_candidate_count": int(unique_candidate_count),
        "flip_candidate_count": int(flip_candidate_count),
        "failure_code": str(failure_code),
        "failure_detail": str(failure_detail),
        "selected_prox_jac": finite_float(metrics.get("prox_jac")),
        "selected_prox_euc": finite_float(metrics.get("prox_euc")),
        "selected_sparsity": finite_float(metrics.get("sparsity")),
        "selected_apf_score": finite_float(metrics.get("apf_score")),
        "selected_actionability_pass": finite_float(metrics.get("actionability_pass")),
        "selected_plausibility_pass": finite_float(metrics.get("plausibility_pass")),
        "selected_feasibility_pass": finite_float(metrics.get("feasibility_pass")),
        "predict_call_count": int(predict_call_count),
        "batch_predict_row_count": int(batch_predict_row_count),
        "batch_predict_ms": float(batch_predict_ms),
        "method_runtime_ms": float(method_runtime_ms),
        "runtime_ms": float(method_runtime_ms),
        "parser_status": None if parse_record is None else parse_record.get("parser_status"),
        "parse_final_stage": None if parse_record is None else parse_record.get("final_stage"),
        "parser_failure_cause": None if parse_record is None else parse_record.get("parser_failure_cause"),
    }


def complete_variant_results(
    *,
    query_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    runtime_rows: list[dict[str, Any]],
    variant: str,
) -> pd.DataFrame:
    runtime_map = {
        (str(row["case_id"]), str(row["method"])): row
        for row in runtime_rows
    }
    parse_map = {
        str(row["case_id"]): row
        for row in parse_df.to_dict(orient="records")
    } if not parse_df.empty else {}

    completed: list[dict[str, Any]] = []
    for case_row in query_df.to_dict(orient="records"):
        parse_record = parse_map.get(str(case_row["case_id"]))
        for method in METHODS:
            key = (str(case_row["case_id"]), method)
            if key in runtime_map:
                completed.append(runtime_map[key])
                continue
            if variant == "nl_first" and not bool((parse_record or {}).get("ready_for_runtime")):
                status = "parse_or_handoff_error"
                failure_code = "PARSE_NOT_READY"
                failure_detail = str((parse_record or {}).get("final_stage") or "parse_not_ready")
            else:
                status = "error"
                failure_code = "MISSING_RUNTIME_ROW"
                failure_detail = "Expected runtime result row is missing."
            completed.append(
                make_result_row(
                    case_record=case_row,
                    parse_record=parse_record,
                    variant=variant,
                    method=method,
                    status=status,
                    prediction_before=None if variant == "nl_first" else int(case_row.get("source_prediction_label", 0)),
                    prediction_after=None,
                    raw_candidate_count=0,
                    unique_candidate_count=0,
                    flip_candidate_count=0,
                    failure_code=failure_code,
                    failure_detail=failure_detail,
                    selected_profile=None,
                    input_profile=None,
                    metrics={
                        "prox_jac": None,
                        "prox_euc": None,
                        "sparsity": None,
                        "apf_score": None,
                        "actionability_pass": None,
                        "plausibility_pass": None,
                        "feasibility_pass": None,
                    },
                    predict_call_count=0,
                    batch_predict_row_count=0,
                    batch_predict_ms=0.0,
                    method_runtime_ms=0.0,
                )
            )
    result_df = pd.DataFrame(completed).sort_values(
        by=["fold_index", "query_pos", "method"],
        kind="mergesort",
    ).reset_index(drop=True)
    return result_df


def run_direct_stage(
    *,
    query_df: pd.DataFrame,
    out_dir: Path,
    config_profile: str,
    bundle_mode: str,
    mi_k_token: str,
    no_cf: int,
    contprox_metric: str,
    mi_feature_scope: str,
    progress_enabled: bool,
) -> pd.DataFrame:
    context = prepare_ufce_context(
        config_profile=config_profile,
        bundle_mode=bundle_mode,
        mi_k_token=mi_k_token,
        no_cf=no_cf,
        contprox_metric=contprox_metric,
        mi_feature_scope=mi_feature_scope,
    )
    fold_inputs = build_variant_fold_inputs(query_df=query_df, parse_df=None, variant="direct")
    runtime_rows = run_part1_variant(
        context=context,
        fold_inputs=fold_inputs,
        variant="direct",
        progress_enabled=progress_enabled,
    )
    result_df = complete_variant_results(
        query_df=query_df,
        parse_df=pd.DataFrame(),
        runtime_rows=runtime_rows,
        variant="direct",
    )
    result_df.to_csv(out_dir / "direct_ufce_ff_results.csv", index=False)
    return result_df


def run_nl_first_stage(
    *,
    query_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    out_dir: Path,
    config_profile: str,
    bundle_mode: str,
    mi_k_token: str,
    no_cf: int,
    contprox_metric: str,
    mi_feature_scope: str,
    progress_enabled: bool,
) -> pd.DataFrame:
    context = prepare_ufce_context(
        config_profile=config_profile,
        bundle_mode=bundle_mode,
        mi_k_token=mi_k_token,
        no_cf=no_cf,
        contprox_metric=contprox_metric,
        mi_feature_scope=mi_feature_scope,
    )
    fold_inputs = build_variant_fold_inputs(query_df=query_df, parse_df=parse_df, variant="nl_first")
    runtime_rows = run_part1_variant(
        context=context,
        fold_inputs=fold_inputs,
        variant="nl_first",
        progress_enabled=progress_enabled,
    )
    result_df = complete_variant_results(
        query_df=query_df,
        parse_df=parse_df,
        runtime_rows=runtime_rows,
        variant="nl_first",
    )
    result_df.to_csv(out_dir / "nl_first_ufce_ff_results.csv", index=False)
    return result_df


def load_variant_frame(out_dir: Path, filename: str) -> pd.DataFrame:
    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing variant artifact: {path}")
    return pd.read_csv(path)


def classify_match_reason(pair_row: dict[str, Any]) -> str:
    if not bool(pair_row.get("nl_parse_ready")):
        return "parse_or_handoff_error"
    direct_status = str(pair_row.get("direct_status"))
    nl_status = str(pair_row.get("nl_status"))
    if direct_status != nl_status:
        return f"status_mismatch:{direct_status}->{nl_status}"
    if direct_status == "released_cf":
        if bool(pair_row.get("selected_profile_match")):
            return "released_cf_exact_match"
        return "selected_profile_mismatch"
    if direct_status == "no_valid_cf":
        return "shared_no_valid_cf"
    if direct_status == "no_cf_needed":
        return "shared_no_cf_needed"
    if direct_status == "parse_or_handoff_error":
        return "shared_parse_error"
    if direct_status == "error":
        return "shared_runtime_error"
    return "other"


def build_bridge_artifacts(
    *,
    source_query_df: pd.DataFrame,
    query_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    nl_df: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
    command: str,
    eval_selection_df: pd.DataFrame,
) -> dict[str, Any]:
    pair_df = pair_variant_results(query_df=query_df, parse_df=parse_df, direct_df=direct_df, nl_df=nl_df)
    outcome_by_variant = summarize_bridge_pairs(pair_df=pair_df, group_cols=["method"])
    outcome_by_group = summarize_bridge_pairs(pair_df=pair_df, group_cols=["language_group", "method"])
    mismatch_df = pair_df.loc[
        (~pair_df["strict_outcome_match"]) | (~pair_df["nl_exact_reconstruction"]),
    ].copy()
    mismatch_reasons = build_mismatch_reason_table(mismatch_df)
    invalid_release_df = build_invalid_release_cases(
        direct_df=direct_df,
        nl_df=nl_df,
    )
    table_main_bridge = build_table_main_bridge(outcome_by_variant)
    table_language_group = build_table_language_group(outcome_by_group)
    summary = build_summary_payload(
        out_dir=out_dir,
        args=args,
        command=command,
        source_query_df=source_query_df,
        query_df=query_df,
        parse_df=parse_df,
        pair_df=pair_df,
        table_main_bridge=table_main_bridge,
        table_language_group=table_language_group,
        mismatch_reasons=mismatch_reasons,
        invalid_release_df=invalid_release_df,
        eval_selection_df=eval_selection_df,
    )

    outcome_by_variant.to_csv(out_dir / "outcome_match_by_variant.csv", index=False)
    outcome_by_group.to_csv(out_dir / "outcome_match_by_group.csv", index=False)
    mismatch_df.to_csv(out_dir / "mismatch_cases.csv", index=False)
    invalid_release_df.to_csv(out_dir / "invalid_release_cases.csv", index=False)
    table_main_bridge.to_csv(out_dir / "table_main_bridge.csv", index=False)
    table_language_group.to_csv(out_dir / "table_language_group.csv", index=False)
    mismatch_reasons.to_csv(out_dir / "table_mismatch_reasons.csv", index=False)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(
        render_summary_markdown(
            summary=summary,
            table_main_bridge=table_main_bridge,
            table_language_group=table_language_group,
            mismatch_reasons=mismatch_reasons,
        ),
        encoding="utf-8",
    )
    return {
        "pair_df": pair_df,
        "outcome_by_variant": outcome_by_variant,
        "outcome_by_group": outcome_by_group,
        "mismatch_df": mismatch_df,
        "invalid_release_df": invalid_release_df,
        "table_main_bridge": table_main_bridge,
        "table_language_group": table_language_group,
        "mismatch_reasons": mismatch_reasons,
        "summary": summary,
    }


def pair_variant_results(
    *,
    query_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    nl_df: pd.DataFrame,
) -> pd.DataFrame:
    query_cols = [
        "case_id",
        "fold_index",
        "fold_name",
        "query_pos",
        "language_group",
        "language_label",
        "benchmark_key",
    ]
    parse_cols = [
        "case_id",
        "ready_for_runtime",
        "exact_reconstruction",
        "parser_status",
        "final_stage",
        "prediction_before_reconstructed",
    ]
    query_view = ensure_columns(query_df, query_cols).loc[:, query_cols]
    parse_view = ensure_columns(parse_df, parse_cols).loc[:, parse_cols].rename(
        columns={
            "ready_for_runtime": "parse_ready_for_runtime",
            "exact_reconstruction": "parse_exact_reconstruction",
            "parser_status": "parse_parser_status",
            "final_stage": "parse_final_stage",
            "prediction_before_reconstructed": "parse_prediction_before_reconstructed",
        }
    )
    runtime_cols = ["case_id", "method", "status", "selected_profile_signature"]
    direct_view = ensure_columns(direct_df, runtime_cols).add_prefix("direct_")
    nl_view = ensure_columns(nl_df, runtime_cols).add_prefix("nl_")
    pair_df = query_view.merge(
        parse_view,
        on="case_id",
        how="left",
    )
    pair_df = pair_df.merge(
        direct_view,
        left_on=["case_id"],
        right_on=["direct_case_id"],
        how="left",
    )
    pair_df = pair_df.merge(
        nl_view,
        left_on=["case_id", "direct_method"],
        right_on=["nl_case_id", "nl_method"],
        how="left",
    )
    pair_df = pair_df.rename(
        columns={
            "direct_method": "method",
        }
    )
    pair_df["nl_parse_ready"] = coalesce_frame_columns(
        pair_df,
        "parse_ready_for_runtime",
        "nl_parse_ready",
        default=False,
    )
    pair_df["nl_exact_reconstruction"] = coalesce_frame_columns(
        pair_df,
        "parse_exact_reconstruction",
        "nl_exact_reconstruction",
        default=False,
    )
    pair_df["nl_parser_status"] = coalesce_frame_columns(
        pair_df,
        "parse_parser_status",
        "nl_parser_status",
        default=None,
    )
    pair_df["nl_final_stage"] = coalesce_frame_columns(
        pair_df,
        "parse_final_stage",
        "nl_final_stage",
        "nl_parse_final_stage",
        default=None,
    )
    pair_df["nl_prediction_before_reconstructed"] = coalesce_frame_columns(
        pair_df,
        "parse_prediction_before_reconstructed",
        "nl_prediction_before_reconstructed",
        default=None,
    )
    pair_df["status_match"] = pair_df["direct_status"].astype(str) == pair_df["nl_status"].astype(str)
    both_released_mask = pair_df["direct_status"].astype(str).eq("released_cf") & pair_df["nl_status"].astype(str).eq("released_cf")
    pair_df["selected_profile_match"] = both_released_mask & (
        pair_df["direct_selected_profile_signature"] == pair_df["nl_selected_profile_signature"]
    )
    pair_df["strict_outcome_match"] = pair_df["status_match"].fillna(False).astype(bool) & (
        ~pair_df["direct_status"].astype(str).eq("released_cf") | pair_df["selected_profile_match"].fillna(False).astype(bool)
    )
    pair_df["end_to_end_exact_match"] = pair_df["strict_outcome_match"].fillna(False).astype(bool) & pair_df[
        "nl_exact_reconstruction"
    ].fillna(False).astype(bool)
    pair_df["nl_parse_ready"] = pair_df["nl_parse_ready"].fillna(False).astype(bool)
    pair_df["nl_exact_reconstruction"] = pair_df["nl_exact_reconstruction"].fillna(False).astype(bool)
    pair_df["selected_profile_match"] = pair_df["selected_profile_match"].fillna(False).astype(bool)
    pair_df["status_match"] = pair_df["status_match"].fillna(False).astype(bool)
    pair_df["strict_outcome_match"] = pair_df["strict_outcome_match"].fillna(False).astype(bool)
    pair_df["end_to_end_exact_match"] = pair_df["end_to_end_exact_match"].fillna(False).astype(bool)
    pair_df = pair_df.loc[:, ~pair_df.columns.duplicated()].copy()
    pair_df["match_reason"] = [classify_match_reason(row) for row in pair_df.to_dict(orient="records")]
    return pair_df.sort_values(
        by=["fold_index", "query_pos", "method"],
        kind="mergesort",
    ).reset_index(drop=True)


def coalesce_frame_columns(frame: pd.DataFrame, *columns: str, default: Any = None) -> pd.Series:
    merged = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        current = frame.loc[:, column]
        if isinstance(current, pd.DataFrame):
            for idx in range(current.shape[1]):
                merged = merged.combine_first(current.iloc[:, idx])
        else:
            merged = merged.combine_first(current)
    if default is not None:
        merged = merged.fillna(default)
    return merged


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def summarize_bridge_pairs(*, pair_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_order = list(group_cols)
    for key, group in pair_df.groupby(group_order, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(_summarize_bridge_group(group=group, labels={column: key[index] for index, column in enumerate(group_order)}))
    summary_df = pd.DataFrame(rows)
    if group_cols == ["method"]:
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame([_summarize_bridge_group(group=pair_df, labels={"method": "ALL"})]),
            ],
            ignore_index=True,
        )
    return summary_df.sort_values(by=group_order, kind="mergesort").reset_index(drop=True)


def _summarize_bridge_group(*, group: pd.DataFrame, labels: dict[str, Any]) -> dict[str, Any]:
    total = int(len(group))
    parse_ready = int(group["nl_parse_ready"].fillna(False).astype(bool).sum())
    exact_reconstruction = int(group["nl_exact_reconstruction"].fillna(False).astype(bool).sum())
    status_match = int(group["status_match"].fillna(False).astype(bool).sum())
    strict_outcome_match = int(group["strict_outcome_match"].fillna(False).astype(bool).sum())
    end_to_end_exact = int(group["end_to_end_exact_match"].fillna(False).astype(bool).sum())
    direct_released = int((group["direct_status"] == "released_cf").sum())
    nl_released = int((group["nl_status"] == "released_cf").sum())
    both_released = int(((group["direct_status"] == "released_cf") & (group["nl_status"] == "released_cf")).sum())
    selected_profile_match = int(group["selected_profile_match"].fillna(False).astype(bool).sum())
    row = dict(labels)
    row.update(
        {
            "pair_count": total,
            "parse_ready_count": parse_ready,
            "parse_ready_rate": parse_ready / total if total else 0.0,
            "exact_reconstruction_count": exact_reconstruction,
            "exact_reconstruction_rate": exact_reconstruction / total if total else 0.0,
            "status_match_count": status_match,
            "status_match_rate": status_match / total if total else 0.0,
            "strict_outcome_match_count": strict_outcome_match,
            "strict_outcome_match_rate": strict_outcome_match / total if total else 0.0,
            "end_to_end_exact_match_count": end_to_end_exact,
            "end_to_end_exact_match_rate": end_to_end_exact / total if total else 0.0,
            "direct_released_cf_count": direct_released,
            "nl_released_cf_count": nl_released,
            "both_released_cf_count": both_released,
            "selected_profile_match_count": selected_profile_match,
            "selected_profile_match_rate_given_both_release": selected_profile_match / both_released if both_released else None,
            "main_match_reason": str(group["match_reason"].value_counts().idxmax()) if total else "",
        }
    )
    return row


def build_mismatch_reason_table(mismatch_df: pd.DataFrame) -> pd.DataFrame:
    if mismatch_df.empty:
        return pd.DataFrame(columns=["match_reason", "language_group", "method", "case_count"])
    rows = []
    for key, group in mismatch_df.groupby(["match_reason", "language_group", "method"], dropna=False):
        match_reason, language_group, method = key
        rows.append(
            {
                "match_reason": str(match_reason),
                "language_group": str(language_group),
                "method": str(method),
                "case_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["case_count", "match_reason", "language_group", "method"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_invalid_release_cases(*, direct_df: pd.DataFrame, nl_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["status", "selected_profile_signature", "prediction_after"]
    frames = [ensure_columns(direct_df, required_cols), ensure_columns(nl_df, required_cols)]
    merged = pd.concat(frames, ignore_index=True)
    prediction_after = pd.to_numeric(merged["prediction_after"], errors="coerce")
    mask = (
        (merged["status"].astype(str) == "released_cf")
        & (
            merged["selected_profile_signature"].isna()
            | (prediction_after.fillna(-1).astype(int) != DESIRED_OUTCOME)
        )
    )
    return merged.loc[mask].reset_index(drop=True)


def build_table_main_bridge(outcome_by_variant: pd.DataFrame) -> pd.DataFrame:
    if outcome_by_variant.empty:
        return outcome_by_variant
    keep_cols = [
        "method",
        "pair_count",
        "parse_ready_rate",
        "exact_reconstruction_rate",
        "status_match_rate",
        "strict_outcome_match_rate",
        "end_to_end_exact_match_rate",
        "direct_released_cf_count",
        "nl_released_cf_count",
        "selected_profile_match_rate_given_both_release",
        "main_match_reason",
    ]
    return outcome_by_variant.loc[:, keep_cols].copy()


def build_table_language_group(outcome_by_group: pd.DataFrame) -> pd.DataFrame:
    if outcome_by_group.empty:
        return outcome_by_group
    keep_cols = [
        "language_group",
        "method",
        "pair_count",
        "parse_ready_rate",
        "exact_reconstruction_rate",
        "strict_outcome_match_rate",
        "end_to_end_exact_match_rate",
        "main_match_reason",
    ]
    return outcome_by_group.loc[:, keep_cols].copy()


def build_summary_payload(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    command: str,
    source_query_df: pd.DataFrame,
    query_df: pd.DataFrame,
    parse_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    table_main_bridge: pd.DataFrame,
    table_language_group: pd.DataFrame,
    mismatch_reasons: pd.DataFrame,
    invalid_release_df: pd.DataFrame,
    eval_selection_df: pd.DataFrame,
) -> dict[str, Any]:
    overall_row = {}
    if not table_main_bridge.empty:
        overall_view = table_main_bridge.loc[table_main_bridge["method"] == "ALL"]
        if not overall_view.empty:
            overall_row = overall_view.iloc[0].to_dict()
    group_counts = (
        query_df.groupby("language_group").size().sort_index().to_dict()
        if not query_df.empty
        else {}
    )
    source_group_counts = (
        source_query_df.groupby("language_group").size().sort_index().to_dict()
        if not source_query_df.empty
        else {}
    )
    parse_ready = int(parse_df["ready_for_runtime"].fillna(False).astype(bool).sum()) if "ready_for_runtime" in parse_df.columns else 0
    exact_reconstruction = int(parse_df["exact_reconstruction"].fillna(False).astype(bool).sum()) if "exact_reconstruction" in parse_df.columns else 0
    accepted_count = sum(
        1
        for row in parse_df.to_dict(orient="records")
        if is_parse_row_accepted(row)
    ) if not parse_df.empty else 0
    source_total = int(len(source_query_df))
    eval_total = int(len(query_df))
    dropped_total = max(0, source_total - eval_total)
    eval_case_counts_by_fold = (
        query_df.groupby("fold_name").size().sort_index().to_dict()
        if not query_df.empty
        else {}
    )
    dropped_case_counts_by_fold = (
        eval_selection_df.loc[~eval_selection_df["eval_included"].fillna(False).astype(bool)]
        .groupby("fold_name")
        .size()
        .sort_index()
        .to_dict()
        if not eval_selection_df.empty
        else {}
    )
    effective_match_rate_over_source_total = None
    if source_total > 0 and overall_row:
        effective_match_rate_over_source_total = (
            float(overall_row.get("strict_outcome_match_rate") or 0.0) * float(eval_total) / float(source_total)
        )
    return {
        "runner_scope": "part2_nl_bank_bridge",
        "generated_at": local_now_iso(),
        "command": command,
        "output_root": str(out_dir),
        "dataset": "bank",
        "query_count": eval_total,
        "source_query_count": source_total,
        "language_group_counts": {str(key): int(value) for key, value in group_counts.items()},
        "source_language_group_counts": {str(key): int(value) for key, value in source_group_counts.items()},
        "parser": {
            "model_alias": str(args.model_alias),
            "api_base": str(args.api_base),
            "timeout_s": float(args.timeout_s),
            "benchmark_en_path": str(BENCHMARK_EN_PATH.resolve()),
            "benchmark_vi_path": str(BENCHMARK_VI_PATH.resolve()),
        },
        "ufce_ff_alignment": {
            "config_profile": str(args.config_profile),
            "bundle_mode": str(args.bundle_mode),
            "mi_k": str(args.mi_k),
            "no_cf": int(args.no_cf),
            "contprox_metric": str(args.contprox_metric),
            "mi_feature_scope": str(args.mi_feature_scope),
            "part1_run_path": str(PART1_RUNNER_PATH.resolve()),
            "part1_ff_canonical_script": str(PART1_FF_CANONICAL_SCRIPT.resolve()),
            "part1_ff_helper_path": str(PART1_FF_PATH.resolve()),
            "part1_ff_core_package": PART1_FF_CORE_PACKAGE,
            "bridge_note": "The bridge reuses Part I run_one_fold after installing the split ufce.ufce_ff core.",
            "thesis_alignment": {
                "published_public_anchor": "docs/THESIS_TABLE_TO_SCRIPT_MAP.md",
                "raw_reproduction_script": "scripts/final/part1/ufce_only_reproduction.py",
                "posthoc_label_audit_script": "scripts/final/part1/author_raw_posthoc_replay.py",
                "ufce_ff_script": "scripts/final/part1/04_ufce_ff.py",
                "ufce_ff_helper_script": "scripts/final/part1/ufce_force_flip_experiment.py",
            },
        },
        "evaluation": {
            "parse_eval_scope": str(args.parse_eval_scope),
            "evaluated_case_count": eval_total,
            "evaluated_case_rate": eval_total / source_total if source_total else 0.0,
            "dropped_case_count": dropped_total,
            "dropped_case_rate": dropped_total / source_total if source_total else 0.0,
            "evaluated_case_counts_by_fold": {str(key): int(value) for key, value in eval_case_counts_by_fold.items()},
            "dropped_case_counts_by_fold": {str(key): int(value) for key, value in dropped_case_counts_by_fold.items()},
            "effective_strict_outcome_match_rate_over_source_total": effective_match_rate_over_source_total,
        },
        "parse_summary": {
            "ready_for_runtime_count": parse_ready,
            "ready_for_runtime_rate": parse_ready / len(parse_df) if len(parse_df) else 0.0,
            "exact_reconstruction_count": exact_reconstruction,
            "exact_reconstruction_rate": exact_reconstruction / len(parse_df) if len(parse_df) else 0.0,
            "accepted_count": int(accepted_count),
            "accepted_rate": accepted_count / len(parse_df) if len(parse_df) else 0.0,
        },
        "bridge_overall": overall_row,
        "invalid_release_count": int(len(invalid_release_df)),
        "top_mismatch_reasons": mismatch_reasons.head(10).to_dict(orient="records"),
        "artifacts": {
            "evaluation_case_selection_csv": str((out_dir / "evaluation_case_selection.csv").resolve()),
            "table_main_bridge_csv": str((out_dir / "table_main_bridge.csv").resolve()),
            "table_language_group_csv": str((out_dir / "table_language_group.csv").resolve()),
            "mismatch_cases_csv": str((out_dir / "mismatch_cases.csv").resolve()),
            "summary_md": str((out_dir / "summary.md").resolve()),
            "summary_json": str((out_dir / "summary.json").resolve()),
        },
    }


def render_summary_markdown(
    *,
    summary: dict[str, Any],
    table_main_bridge: pd.DataFrame,
    table_language_group: pd.DataFrame,
    mismatch_reasons: pd.DataFrame,
) -> str:
    lines = [
        "# NL Bank Bridge Summary",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- dataset: `{summary['dataset']}`",
        f"- output_root: `{summary['output_root']}`",
        f"- query_count: `{summary['query_count']}`",
        f"- source_query_count: `{summary['source_query_count']}`",
        f"- parse_eval_scope: `{summary['evaluation']['parse_eval_scope']}`",
        f"- evaluated_case_rate: `{summary['evaluation']['evaluated_case_rate']:.4f}`",
        f"- dropped_case_count: `{summary['evaluation']['dropped_case_count']}`",
        f"- effective_strict_match_over_source_total: `{summary['evaluation']['effective_strict_outcome_match_rate_over_source_total']}`",
        f"- parser_model: `{summary['parser']['model_alias']}`",
        f"- parser_api_base: `{summary['parser']['api_base']}`",
        f"- ufce_config_profile: `{summary['ufce_ff_alignment']['config_profile']}`",
        f"- ufce_bundle_mode: `{summary['ufce_ff_alignment']['bundle_mode']}`",
        f"- ufce_mi_k: `{summary['ufce_ff_alignment']['mi_k']}`",
        f"- ufce_no_cf: `{summary['ufce_ff_alignment']['no_cf']}`",
        "",
        "## Evaluation Scope",
        "",
        f"- Evaluated parser cases: `{summary['evaluation']['evaluated_case_count']}` / `{summary['source_query_count']}`",
        f"- Dropped parser cases: `{summary['evaluation']['dropped_case_count']}`",
        f"- Parse accepted cases: `{summary['parse_summary']['accepted_count']}` / `{summary['source_query_count']}`",
        f"- Parse ready cases: `{summary['parse_summary']['ready_for_runtime_count']}` / `{summary['source_query_count']}`",
        "",
        "## Main Bridge Table",
        "",
        dataframe_to_markdown_fallback(table_main_bridge) if not table_main_bridge.empty else "_No bridge summary available._",
        "",
        "## Language Groups",
        "",
        dataframe_to_markdown_fallback(table_language_group) if not table_language_group.empty else "_No group summary available._",
        "",
        "## Top Mismatch Reasons",
        "",
        dataframe_to_markdown_fallback(mismatch_reasons.head(20)) if not mismatch_reasons.empty else "_No mismatch cases._",
        "",
        "## Part I Alignment",
        "",
        "- Published/public anchor: `docs/THESIS_TABLE_TO_SCRIPT_MAP.md`",
        "- Raw reproduction: `scripts/final/part1/ufce_only_reproduction.py`",
        "- Posthoc label audit: `scripts/final/part1/author_raw_posthoc_replay.py`",
        "- Locked UFCE-FF route used here: `scripts/final/part1/04_ufce_ff.py` / `ufce.ufce_ff`, with Part 2 trace helpers from `scripts/final/part1/ufce_force_flip_experiment.py`",
        "",
    ]
    return "\n".join(lines)


def dataframe_to_markdown_fallback(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def write_config_snapshot(*, out_dir: Path, args: argparse.Namespace, command: str) -> None:
    payload = {
        "generated_at": local_now_iso(),
        "command": command,
        "cwd": str(Path.cwd()),
        "stage": str(args.stage),
        "seed": int(args.seed),
        "cases_per_group": int(args.cases_per_group),
        "template_version": TEMPLATE_VERSION,
        "config_profile": str(args.config_profile),
        "bundle_mode": str(args.bundle_mode),
        "mi_k": str(args.mi_k),
        "no_cf": int(args.no_cf),
        "contprox_metric": str(args.contprox_metric),
        "parse_eval_scope": str(args.parse_eval_scope),
        "mi_feature_scope": str(args.mi_feature_scope),
        "parser_model_alias": str(args.model_alias),
        "parser_api_base": str(args.api_base),
        "parser_timeout_s": float(args.timeout_s),
        "dataset_contract": {
            "feature_order": list(BANK_FEATURE_ORDER),
            "feature_types": dict(BANK_FEATURE_TYPES),
            "bank_source_path": str(BANK_SOURCE_PATH.resolve()),
            "bank_source_sha256": sha256_file(BANK_SOURCE_PATH),
        },
        "benchmarks": {
            "en_path": str(BENCHMARK_EN_PATH.resolve()),
            "en_sha256": sha256_file(BENCHMARK_EN_PATH),
            "vi_path": str(BENCHMARK_VI_PATH.resolve()),
            "vi_sha256": sha256_file(BENCHMARK_VI_PATH),
        },
        "part1_paths": {
            "runner_path": str(PART1_RUNNER_PATH.resolve()),
            "runner_sha256": sha256_file(PART1_RUNNER_PATH),
            "ff_canonical_script": str(PART1_FF_CANONICAL_SCRIPT.resolve()),
            "ff_canonical_script_sha256": sha256_file(PART1_FF_CANONICAL_SCRIPT),
            "ff_helper_path": str(PART1_FF_PATH.resolve()),
            "ff_helper_sha256": sha256_file(PART1_FF_PATH),
            "ff_core_package": PART1_FF_CORE_PACKAGE,
        },
        "group_specs": list(GROUP_SPECS),
    }
    write_json(out_dir / "config_snapshot.json", payload)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def ensure_primary_outputs(*, out_dir: Path, stage: str) -> None:
    if stage not in {"all", "compare"}:
        return
    missing = [name for name in PRIMARY_OUTPUTS if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError("Missing required bridge outputs: " + ", ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())
