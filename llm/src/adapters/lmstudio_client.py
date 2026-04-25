from __future__ import annotations

import time
from typing import Any

import requests


def call_lm_studio(api_base: str, payload: dict[str, object], timeout_s: float) -> dict[str, Any]:
    url = f"{api_base}/v1/chat/completions"
    request_payload = adapt_chat_completions_payload(payload)
    started = time.perf_counter()
    try:
        response = requests.post(url, json=request_payload, timeout=timeout_s)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
            "response_json": None,
            "response_text": None,
            "elapsed_ms": elapsed_ms if "elapsed_ms" in locals() else None,
        }

    try:
        response_json = response.json()
    except ValueError:
        response_json = None

    error = None
    if not response.ok:
        error = format_http_error(response.status_code, response_json, response.text)
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "error": error,
        "response_json": response_json,
        "response_text": response.text,
        "elapsed_ms": elapsed_ms,
    }


def adapt_chat_completions_payload(payload: dict[str, object]) -> dict[str, object]:
    if "messages" in payload:
        return dict(payload)

    request_payload: dict[str, object] = {}
    for key in ("model", "temperature", "top_p", "max_tokens", "stream", "response_format"):
        if key in payload:
            request_payload[key] = payload[key]

    messages: list[dict[str, object]] = []
    system_prompt = payload.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    user_input = payload.get("input")
    if isinstance(user_input, str) and user_input.strip():
        messages.append({"role": "user", "content": user_input})
    elif isinstance(user_input, list):
        messages.append({"role": "user", "content": user_input})

    request_payload["messages"] = messages
    return request_payload


def extract_response_data(
    response_json: dict[str, Any] | None,
    request_latency_ms: float | None,
) -> dict[str, Any]:
    if response_json is None:
        return {
            "message_text": "",
            "reasoning_text": "",
            "usage": {},
            "stats": {},
            "derived_metrics": {"request_latency_ms": request_latency_ms},
        }

    message_parts: list[str] = []
    reasoning_parts: list[str] = []

    output_items = response_json.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "reasoning":
                text = extract_reasoning_text(item)
                if text:
                    reasoning_parts.append(text)
            elif item_type == "message":
                text = extract_message_text(item)
                if text:
                    message_parts.append(text)

    choices = response_json.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                text = flatten_text(message.get("content"))
                if text:
                    message_parts.append(text)
                reasoning = flatten_text(message.get("reasoning"))
                if reasoning:
                    reasoning_parts.append(reasoning)
            reasoning = flatten_text(choice.get("reasoning"))
            if reasoning:
                reasoning_parts.append(reasoning)

    top_level_message = response_json.get("message")
    if isinstance(top_level_message, dict):
        text = flatten_text(top_level_message.get("content"))
        if text:
            message_parts.append(text)

    if not message_parts:
        fallback = flatten_text(response_json.get("content"))
        if fallback:
            message_parts.append(fallback)

    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    stats = response_json.get("stats")
    if not isinstance(stats, dict):
        stats = {}
    timings = response_json.get("timings")
    if isinstance(timings, dict):
        stats = {**stats, **timings}

    return {
        "message_text": "\n\n".join(part for part in message_parts if part).strip(),
        "reasoning_text": "\n\n".join(part for part in reasoning_parts if part).strip(),
        "usage": usage,
        "stats": stats,
        "derived_metrics": derive_metrics(usage, stats, request_latency_ms),
    }


def extract_reasoning_text(item: dict[str, Any]) -> str:
    for key in ("summary", "content", "text"):
        text = flatten_text(item.get(key))
        if text:
            return text
    return ""


def extract_message_text(item: dict[str, Any]) -> str:
    return flatten_text(item.get("content") or item.get("text"))


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = flatten_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "summary"):
            text = flatten_text(value.get(key))
            if text:
                return text
        if value.get("type") in {"text", "output_text"}:
            text = flatten_text(value.get("text"))
            if text:
                return text
    return ""


def derive_metrics(
    usage: dict[str, Any],
    stats: dict[str, Any],
    request_latency_ms: float | None,
) -> dict[str, Any]:
    prompt_tokens = first_present(usage, ("prompt_tokens", "input_tokens"))
    if prompt_tokens is None:
        prompt_tokens = first_present(stats, ("input_tokens",))

    completion_tokens = first_present(usage, ("completion_tokens", "output_tokens"))
    if completion_tokens is None:
        completion_tokens = first_present(stats, ("total_output_tokens", "output_tokens"))

    reasoning_output_tokens = first_present(stats, ("reasoning_output_tokens",))

    total_tokens = first_present(usage, ("total_tokens",))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    ttft_ms = first_present(stats, ("time_to_first_token_ms", "ttft_ms", "time_to_first_token"))
    ttft_seconds = first_present(stats, ("time_to_first_token_seconds",))
    if ttft_ms is None and isinstance(ttft_seconds, (int, float)):
        ttft_ms = ttft_seconds * 1000.0

    total_time_ms = first_present(stats, ("total_time_ms", "generation_time_ms", "latency_ms"))
    tokens_per_second = first_present(stats, ("tokens_per_second", "tps"))
    return {
        "request_latency_ms": request_latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": ttft_ms,
        "total_time_ms": total_time_ms,
        "tokens_per_second": tokens_per_second,
    }


def first_present(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def format_http_error(
    status_code: int,
    response_json: dict[str, Any] | None,
    response_text: str | None,
) -> str:
    if isinstance(response_json, dict):
        error_payload = response_json.get("error")
        if isinstance(error_payload, dict):
            parts = [f"HTTP {status_code}"]
            code = error_payload.get("code")
            param = error_payload.get("param")
            message = error_payload.get("message")
            error_type = error_payload.get("type")
            if code:
                parts.append(str(code))
            if param:
                parts.append(f"param={param}")
            if error_type:
                parts.append(f"type={error_type}")
            if message:
                parts.append(f"message={message}")
            return ": ".join((parts[0], ", ".join(parts[1:]))) if len(parts) > 1 else parts[0]
    if response_text:
        return f"HTTP {status_code}: {response_text.strip()}"
    return f"HTTP {status_code}"
