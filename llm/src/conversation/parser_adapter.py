from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.src.adapters.lmstudio_client import call_lm_studio, extract_response_data
from llm.src.conversation.types import ParserAdapterResult
from llm.src.parser.bank_profile_parse_v2 import (
    BANK_PROFILE_PARSE_V2_EXTRACTION_SCHEMA_NAME,
    BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
    BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
    BANK_PROFILE_PARSE_V2_SCHEMA_NAME,
    BANK_PROFILE_PARSE_V2_VERIFIER_SCHEMA_NAME,
    build_bank_profile_parse_v2_extraction_schema,
    build_bank_profile_parse_v2_prompt,
    build_bank_profile_parse_v2_schema,
    build_bank_profile_parse_v2_verifier_prompt,
    build_bank_profile_parse_v2_verifier_schema,
)
from llm.src.parser.prompt_builder import (
    DEFAULT_REFINEMENT_SCHEMA_NAME,
    build_live_refinement_response_schema,
    build_live_response_schema,
    build_live_user_prompt,
    build_live_refinement_user_prompt,
    build_repair_user_prompt,
    build_refinement_repair_user_prompt,
    build_request_payload,
    load_json_schema,
    load_system_prompt,
    normalize_parser_schema_version,
    response_schema_name_for_version,
)
from llm_eval.config import load_benchmark


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_PATH = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_en.yaml"
DEFAULT_SYSTEM_PROMPT_PATH = ROOT / "llm" / "prompts" / "parser_system_prompt_v1.txt"
DEFAULT_SCHEMA_PATH = ROOT / "llm" / "config" / "parser_schema_v3.json"
DEFAULT_REFINEMENT_SCHEMA_PATH = ROOT / "llm" / "config" / "refinement_parser_schema_v1.json"
DEFAULT_MODEL_ALIAS = "qwen3-14b"
DEFAULT_API_BASE = "http://localhost:1234"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_PARSE_MAX_TOKENS = 512
DEFAULT_REPAIR_MAX_TOKENS = 768
DEFAULT_FUTURE_EXPLANATION_TOKEN_BUDGET = (1536, 2048)
DEFAULT_FUTURE_NEGOTIATION_TOKEN_BUDGET = (2048, 3072)
DEFAULT_STRUCTURED_OUTPUT_MODE = "json_schema_strict"


class LiveLmStudioParserAdapter:
    def __init__(
        self,
        *,
        model_alias: str = DEFAULT_MODEL_ALIAS,
        api_base: str = DEFAULT_API_BASE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        system_prompt_path: Path | None = None,
        benchmark_path: Path | None = None,
        schema_path: Path | None = None,
        refinement_schema_path: Path | None = None,
        parse_max_tokens: int = DEFAULT_PARSE_MAX_TOKENS,
        repair_max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
        stream: bool = False,
    ) -> None:
        self.model_alias = model_alias
        self.api_base = api_base.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.system_prompt_path = Path(system_prompt_path or DEFAULT_SYSTEM_PROMPT_PATH)
        self.benchmark_path = Path(benchmark_path or DEFAULT_BENCHMARK_PATH)
        self.schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)
        self.refinement_schema_path = Path(refinement_schema_path or DEFAULT_REFINEMENT_SCHEMA_PATH)
        self.system_prompt = load_system_prompt(self.system_prompt_path)
        self.response_schema = load_json_schema(self.schema_path)
        self.refinement_response_schema = load_json_schema(self.refinement_schema_path)
        self.parser_schema_version = normalize_parser_schema_version(self.schema_path.stem)
        self.parse_max_tokens = int(parse_max_tokens)
        self.repair_max_tokens = int(repair_max_tokens)
        self.stream = bool(stream)
        self.structured_output_mode = DEFAULT_STRUCTURED_OUTPUT_MODE

    def load_benchmark(self):
        return load_benchmark(self.benchmark_path)

    def parse(self, *, user_text: str, benchmark=None, dataset_package=None) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        dataset_id = _dataset_id(dataset_package)
        schema_version = self._schema_version_for_benchmark(active_benchmark, dataset_id=dataset_id)
        user_prompt = build_live_user_prompt(
            active_benchmark,
            user_text,
            dataset_id=dataset_id,
            dataset_label=_primary_subject_label(dataset_package),
            numeric_bound_fields=_numeric_bound_fields(dataset_package),
            schema_version=schema_version,
        )
        return self._invoke(
            user_prompt=user_prompt,
            task_type="parse",
            max_tokens=self.parse_max_tokens,
            response_schema=self._response_schema_for_parse(
                active_benchmark,
                dataset_package=dataset_package,
                schema_version=schema_version,
            ),
            schema_name=_primary_schema_name(dataset_package, schema_version=schema_version),
            dataset_package=dataset_package,
        )

    def repair(
        self,
        *,
        invalid_output: str,
        errors: list[str],
        benchmark=None,
        dataset_package=None,
    ) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        schema_version = self._schema_version_for_benchmark(
            active_benchmark,
            dataset_id=_dataset_id(dataset_package),
        )
        user_prompt = build_repair_user_prompt(
            active_benchmark,
            invalid_output=invalid_output,
            errors=errors,
            dataset_id=_dataset_id(dataset_package),
            dataset_label=_primary_subject_label(dataset_package),
            numeric_bound_fields=_numeric_bound_fields(dataset_package),
            schema_version=schema_version,
        )
        return self._invoke(
            user_prompt=user_prompt,
            task_type="repair",
            max_tokens=self.repair_max_tokens,
            response_schema=self._response_schema_for_parse(
                active_benchmark,
                dataset_package=dataset_package,
                schema_version=schema_version,
            ),
            schema_name=_primary_schema_name(dataset_package, schema_version=schema_version),
            dataset_package=dataset_package,
        )

    def parse_bank_profile_v2(
        self,
        *,
        user_text: str,
        benchmark=None,
        dataset_package=None,
        stage: str = BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
        extracted_evidence: dict[str, Any] | None = None,
    ) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        if stage == BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE:
            user_prompt = build_bank_profile_parse_v2_prompt(
                active_benchmark,
                user_text=user_text,
                dataset_label=_primary_subject_label(dataset_package),
                stage=stage,
            )
            response_schema = build_bank_profile_parse_v2_extraction_schema(active_benchmark)
            schema_name = BANK_PROFILE_PARSE_V2_EXTRACTION_SCHEMA_NAME
            task_type = "bank_profile_parse_v2_extraction"
        else:
            user_prompt = build_bank_profile_parse_v2_prompt(
                active_benchmark,
                user_text=user_text,
                dataset_label=_primary_subject_label(dataset_package),
                extracted_evidence=extracted_evidence,
                stage=stage,
            )
            response_schema = build_bank_profile_parse_v2_schema(active_benchmark)
            schema_name = BANK_PROFILE_PARSE_V2_SCHEMA_NAME
            task_type = "bank_profile_parse_v2_normalization"
        return self._invoke(
            user_prompt=user_prompt,
            task_type=task_type,
            max_tokens=self.parse_max_tokens,
            response_schema=response_schema,
            schema_name=schema_name,
            dataset_package=dataset_package,
        )

    def retry_bank_profile_v2(
        self,
        *,
        user_text: str,
        previous_output: str,
        errors: list[str],
        benchmark=None,
        dataset_package=None,
        stage: str = BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        extracted_evidence: dict[str, Any] | None = None,
        extraction_hints: dict[str, Any] | None = None,
    ) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        if stage == BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE:
            user_prompt = build_bank_profile_parse_v2_prompt(
                active_benchmark,
                user_text=user_text,
                dataset_label=_primary_subject_label(dataset_package),
                retry_errors=errors,
                previous_output=previous_output,
                extraction_hints=extraction_hints,
                stage=stage,
            )
            response_schema = build_bank_profile_parse_v2_extraction_schema(active_benchmark)
            schema_name = BANK_PROFILE_PARSE_V2_EXTRACTION_SCHEMA_NAME
            task_type = "bank_profile_parse_v2_extraction_retry"
        else:
            user_prompt = build_bank_profile_parse_v2_prompt(
                active_benchmark,
                user_text=user_text,
                dataset_label=_primary_subject_label(dataset_package),
                extracted_evidence=extracted_evidence,
                retry_errors=errors,
                previous_output=previous_output,
                stage=stage,
            )
            response_schema = build_bank_profile_parse_v2_schema(active_benchmark)
            schema_name = BANK_PROFILE_PARSE_V2_SCHEMA_NAME
            task_type = "bank_profile_parse_v2_normalization_retry"
        return self._invoke(
            user_prompt=user_prompt,
            task_type=task_type,
            max_tokens=self.repair_max_tokens,
            response_schema=response_schema,
            schema_name=schema_name,
            dataset_package=dataset_package,
        )

    def verify_bank_profile_v2(
        self,
        *,
        user_text: str,
        candidate: dict[str, Any],
        validation_errors: list[str] | None = None,
        benchmark=None,
        dataset_package=None,
    ) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        user_prompt = build_bank_profile_parse_v2_verifier_prompt(
            active_benchmark,
            user_text=user_text,
            candidate=candidate,
            validation_errors=validation_errors or [],
        )
        return self._invoke(
            user_prompt=user_prompt,
            task_type="bank_profile_parse_v2_verifier",
            max_tokens=self.repair_max_tokens,
            response_schema=build_bank_profile_parse_v2_verifier_schema(active_benchmark),
            schema_name=BANK_PROFILE_PARSE_V2_VERIFIER_SCHEMA_NAME,
            dataset_package=dataset_package,
        )

    def parse_refinement(
        self,
        *,
        user_text: str,
        active_constraint_spec: dict[str, Any] | None,
        pending_refinement_clarification: dict[str, Any] | None,
        benchmark=None,
        dataset_package=None,
    ) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        user_prompt = build_live_refinement_user_prompt(
            active_benchmark,
            user_text=user_text,
            active_constraint_spec=active_constraint_spec,
            pending_refinement_clarification=pending_refinement_clarification,
            dataset_id=_dataset_id(dataset_package),
            dataset_label=_primary_subject_label(dataset_package),
            numeric_bound_fields=_numeric_bound_fields(dataset_package),
        )
        return self._invoke(
            user_prompt=user_prompt,
            task_type="parse_refinement",
            max_tokens=self.parse_max_tokens,
            response_schema=self._response_schema_for_refinement(dataset_package=dataset_package),
            schema_name=_refinement_schema_name(dataset_package),
            dataset_package=dataset_package,
        )

    def repair_refinement(
        self,
        *,
        invalid_output: str,
        errors: list[str],
        active_constraint_spec: dict[str, Any] | None,
        pending_refinement_clarification: dict[str, Any] | None,
        benchmark=None,
        dataset_package=None,
    ) -> ParserAdapterResult:
        active_benchmark = benchmark or self.load_benchmark()
        user_prompt = build_refinement_repair_user_prompt(
            active_benchmark,
            invalid_output=invalid_output,
            errors=errors,
            active_constraint_spec=active_constraint_spec,
            pending_refinement_clarification=pending_refinement_clarification,
            dataset_id=_dataset_id(dataset_package),
            dataset_label=_primary_subject_label(dataset_package),
            numeric_bound_fields=_numeric_bound_fields(dataset_package),
        )
        return self._invoke(
            user_prompt=user_prompt,
            task_type="repair_refinement",
            max_tokens=self.repair_max_tokens,
            response_schema=self._response_schema_for_refinement(dataset_package=dataset_package),
            schema_name=_refinement_schema_name(dataset_package),
            dataset_package=dataset_package,
        )

    def _invoke(
        self,
        *,
        user_prompt: str,
        task_type: str,
        max_tokens: int,
        response_schema: dict[str, Any],
        schema_name: str,
        dataset_package=None,
    ) -> ParserAdapterResult:
        request_payload = build_request_payload(
            model=self.model_alias,
            system_prompt=self._system_prompt_for_dataset(dataset_package),
            user_prompt=user_prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            response_schema=response_schema,
            schema_name=schema_name,
            stream=self.stream,
        )
        api_result = call_lm_studio(
            api_base=self.api_base,
            payload=request_payload,
            timeout_s=self.timeout_s,
        )
        response_data = extract_response_data(
            api_result.get("response_json"),
            api_result.get("elapsed_ms"),
        )
        raw_error = api_result.get("error")
        failure_cause = classify_parser_failure_cause(
            api_error=raw_error,
            message_text=response_data["message_text"],
        )
        api_error = normalize_parser_api_error(
            api_error=raw_error,
            failure_cause=failure_cause,
        )
        return ParserAdapterResult(
            message_text=response_data["message_text"],
            api_error=api_error,
            http_status_code=api_result.get("status_code"),
            raw_response_text=api_result.get("response_text"),
            response_json=api_result.get("response_json"),
            reasoning_text=response_data["reasoning_text"],
            usage=response_data["usage"],
            stats=response_data["stats"],
            derived_metrics=response_data["derived_metrics"],
            request_payload=request_payload,
            failure_cause=failure_cause,
            task_type=task_type,
        )

    def generate_conversational_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str | None:
        request_payload = {
            "model": self.model_alias,
            "temperature": 0.4,
            "top_p": 0.95,
            "stream": self.stream,
        }
        if max_tokens:
            request_payload["max_tokens"] = int(max_tokens)
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        if user_prompt and user_prompt.strip():
            messages.append({"role": "user", "content": user_prompt.strip()})
        
        request_payload["messages"] = messages

        api_result = call_lm_studio(
            api_base=self.api_base,
            payload=request_payload,
            timeout_s=self.timeout_s,
        )
        if not api_result.get("ok"):
            return None
            
        response_data = extract_response_data(
            api_result.get("response_json"),
            api_result.get("elapsed_ms"),
        )
        text = response_data.get("message_text")
        return text.strip() if text and text.strip() else None

    def describe_request_profile(self, task_type: str, *, dataset_package=None) -> dict[str, Any]:
        if task_type in {"repair", "repair_refinement"}:
            max_tokens = self.repair_max_tokens
        else:
            max_tokens = self.parse_max_tokens
        schema_name = (
            _refinement_schema_name(dataset_package)
            if task_type in {"parse_refinement", "repair_refinement"}
            else _primary_schema_name(dataset_package)
        )
        return {
            "task_type": task_type,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "stream": self.stream,
            "structured_output_mode": self.structured_output_mode,
            "response_schema_name": schema_name,
        }

    def describe_token_policy(self) -> dict[str, Any]:
        return {
            "parse": int(self.parse_max_tokens),
            "repair": int(self.repair_max_tokens),
            "future_explanation": {
                "min": DEFAULT_FUTURE_EXPLANATION_TOKEN_BUDGET[0],
                "max": DEFAULT_FUTURE_EXPLANATION_TOKEN_BUDGET[1],
            },
            "future_negotiation": {
                "min": DEFAULT_FUTURE_NEGOTIATION_TOKEN_BUDGET[0],
                "max": DEFAULT_FUTURE_NEGOTIATION_TOKEN_BUDGET[1],
            },
        }

    def _response_schema_for_parse(
        self,
        benchmark,
        *,
        dataset_package=None,
        schema_version: str | None = None,
    ) -> dict[str, Any]:
        active_schema_version = normalize_parser_schema_version(schema_version or self.parser_schema_version)
        if dataset_package is None:
            if active_schema_version == self.parser_schema_version:
                return self.response_schema
            return build_live_response_schema(
                benchmark,
                schema_version=active_schema_version,
            )
        return build_live_response_schema(
            benchmark,
            numeric_bound_fields=_numeric_bound_fields(dataset_package),
            schema_version=active_schema_version,
        )

    def _response_schema_for_refinement(self, *, dataset_package=None) -> dict[str, Any]:
        if dataset_package is None:
            return self.refinement_response_schema
        return build_live_refinement_response_schema(
            numeric_bound_fields=_numeric_bound_fields(dataset_package),
        )

    def _system_prompt_for_dataset(self, dataset_package) -> str:
        if dataset_package is None:
            return self.system_prompt
        subject_label = _primary_subject_label(dataset_package)
        dataset_name = _dataset_id(dataset_package)
        return (
            f"You are a structured extraction assistant for {subject_label} counterfactual requests.\n\n"
            + self.system_prompt.replace("for bank counterfactual requests", f"for {dataset_name} counterfactual requests")
        )

    def _schema_version_for_benchmark(self, benchmark, *, dataset_id: str = "bank") -> str:
        if dataset_id != "bank":
            return "v2"
        benchmark_name = str(getattr(benchmark, "benchmark_name", "") or "").lower()
        if "v2" in benchmark_name:
            return "v2"
        if "v3" in benchmark_name:
            return "v3"
        if "_v2" in str(self.benchmark_path).lower():
            return "v2"
        return self.parser_schema_version


def classify_parser_failure_cause(*, api_error: str | None, message_text: str) -> str | None:
    if message_text.strip():
        return None
    if not api_error:
        return None
    error_lower = api_error.lower()
    if "readtimeout" in error_lower or "timed out" in error_lower:
        return "timeout_no_body"
    if "response_format" in error_lower or "json_schema" in error_lower:
        return "unsupported_structured_output"
    return None


def normalize_parser_api_error(*, api_error: str | None, failure_cause: str | None) -> str | None:
    if api_error is None:
        return None
    if failure_cause == "unsupported_structured_output":
        return (
            "Parser configuration error: structured JSON output is unsupported by the current "
            f"LM Studio server or model. Original error: {api_error}"
        )
    return api_error


def _dataset_id(dataset_package) -> str:
    if dataset_package is None:
        return "bank"
    return str(getattr(dataset_package, "dataset_id", "bank"))


def _numeric_bound_fields(dataset_package) -> list[str] | None:
    numeric_bound_fields = getattr(dataset_package, "numeric_bound_fields", None)
    if callable(numeric_bound_fields):
        return list(numeric_bound_fields())
    return None


def _primary_subject_label(dataset_package) -> str:
    subject_label = getattr(dataset_package, "primary_subject_label", None)
    if callable(subject_label):
        return str(subject_label())
    return "bank profile"


def _primary_schema_name(dataset_package, *, schema_version: str | None = None) -> str:
    schema_name = getattr(dataset_package, "primary_response_schema_name", None)
    if callable(schema_name):
        return str(schema_name())
    return response_schema_name_for_version(schema_version)


def _refinement_schema_name(dataset_package) -> str:
    schema_name = getattr(dataset_package, "refinement_response_schema_name", None)
    if callable(schema_name):
        return str(schema_name())
    return DEFAULT_REFINEMENT_SCHEMA_NAME
