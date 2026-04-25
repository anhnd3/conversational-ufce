from __future__ import annotations

from llm.src.orchestration.parse_then_validate import parse_then_validate


def parse_counterfactual_request(message_text: str, benchmark):
    return parse_then_validate(message_text=message_text, benchmark=benchmark)
