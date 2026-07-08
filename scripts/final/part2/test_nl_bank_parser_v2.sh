#!/usr/bin/env bash
set -euo pipefail

ROOT="${UFCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

cd "$ROOT"

echo "[test] python=$PYTHON"

"$PYTHON" -m py_compile \
  llm/src/parser/bank_profile_parse_v2.py \
  llm/src/conversation/parser_adapter.py \
  scripts/final/part2/nl_bank_bridge_experiment.py \
  llm/tests/test_bank_profile_parse_v2.py \
  llm/tests/test_parser_service.py \
  llm/tests/test_part2_nl_bank_bridge_experiment.py

if "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON" -m pytest \
    llm/tests/test_bank_profile_parse_v2.py \
    llm/tests/test_parser_service.py \
    llm/tests/test_part2_nl_bank_bridge_experiment.py \
    llm/tests/conversation/test_parser_adapter.py
else
  echo "[test] pytest not installed; running parser-v2 smoke checks instead."
  "$PYTHON" - <<'PY'
import json
from llm_eval.config import benchmark_from_dict
from llm.src.parser.bank_profile_parse_v2 import (
    BANK_PROFILE_PARSE_V2_TASK,
    build_bank_profile_parse_v2_prompt,
    parse_bank_profile_parse_v2_verifier_payload,
    run_bank_profile_parse_v2_quality,
)
from scripts.final.part2.nl_bank_bridge_experiment import (
    build_parse_acceptance_metrics,
    build_parse_live_quality_metrics,
    format_parse_acceptance_failure,
    format_parse_live_quality_progress,
)

benchmark = benchmark_from_dict({
    "benchmark_name": "ufce_bank_cf_parser_v3_en",
    "description": "Smoke benchmark.",
    "target_cf_fields": [
        {"name": "Income", "type": "float", "description": "Target income value"},
        {"name": "CCAvg", "type": "float", "description": "Average credit card spending"},
        {"name": "Family", "type": "int", "description": "Family size"},
        {"name": "Education", "type": "int", "description": "Education code"},
        {"name": "Mortgage", "type": "float", "description": "Mortgage value"},
        {"name": "CDAccount", "type": "binary", "description": "1=yes, 0=no"},
        {"name": "Online", "type": "binary", "description": "1=yes, 0=no"},
        {"name": "SecuritiesAccount", "type": "binary", "description": "1=yes, 0=no"},
        {"name": "CreditCard", "type": "binary", "description": "1=yes, 0=no"},
    ],
    "output_contract": {
        "task": "extract_cf_request",
        "status_enum": ["complete", "partial", "needs_clarification", "conflict"],
        "rules": [],
    },
    "cases": [],
})

def field(value, quote):
    return {
        "value": value,
        "evidence_quote": quote,
        "confidence": 1.0,
        "normalization_note": "semantic extraction",
    }

user_text = (
    "annual income seventy two; family size two; average card spend one point five; "
    "education level three; mortgage zero; securities account no; CD account yes; "
    "online banking yes; credit card no."
)
payload = {
    "task": BANK_PROFILE_PARSE_V2_TASK,
    "status": "complete",
    "fields": {
        "Income": field(72, "annual income seventy two"),
        "Family": field(2, "family size two"),
        "CCAvg": field(1.5, "average card spend one point five"),
        "Education": field(3, "education level three"),
        "Mortgage": field(0, "mortgage zero"),
        "SecuritiesAccount": field(0, "securities account no"),
        "CDAccount": field(1, "CD account yes"),
        "Online": field(1, "online banking yes"),
        "CreditCard": field(0, "credit card no"),
    },
    "missing_fields": [],
    "conflicts": [],
    "notes": [],
}
quality = run_bank_profile_parse_v2_quality(
    message_text=json.dumps(payload),
    benchmark_spec=benchmark,
    user_text=user_text,
    numeric_bound_fields=["Income", "CCAvg", "Mortgage"],
)
assert quality.schema_validation.is_valid, quality.schema_validation.errors
assert quality.normalized.parsed_json["cf_request"]["Income"] == 72

bad_verifier = {
    "verdict": "FAIL",
    "candidate": payload,
    "errors": ["not entailed"],
    "notes": [],
}
_, verifier_errors = parse_bank_profile_parse_v2_verifier_payload(json.dumps(bad_verifier))
assert verifier_errors == ["verifier reported: not entailed"], verifier_errors

prompt = build_bank_profile_parse_v2_prompt(
    benchmark,
    user_text="Income 68; Family mot; CCAvg zero point five.",
    retry_errors=["Missing runtime fields after semantic parse: Mortgage"],
    previous_output='{"task":"extract_bank_profile_v2"}',
)
assert "BankProfileParseV2" in prompt and "validation_errors" in prompt

metrics = build_parse_acceptance_metrics(
    parse_rows=[
        {"language_group": "G1", "ready_for_runtime": True, "exact_reconstruction": True, "schema_errors_json": "[]", "verifier_errors_json": "[]"},
        {"language_group": "G2", "ready_for_runtime": False, "exact_reconstruction": False, "schema_errors_json": '["x"]', "verifier_errors_json": '["y"]'},
    ],
    total_cases=3,
)
assert metrics["error_count"] == 2
assert abs(metrics["error_rate"] - (2 / 3)) < 1e-12
assert "error_rate=0.6667" in format_parse_acceptance_failure(metrics)
live_metrics = build_parse_live_quality_metrics(
    parse_rows=[
        {"language_group": "G1", "ready_for_runtime": True, "exact_reconstruction": True, "schema_errors_json": "[]", "verifier_errors_json": "[]"},
        {"language_group": "G2", "ready_for_runtime": True, "exact_reconstruction": False, "schema_errors_json": "[]", "verifier_errors_json": "[]"},
    ],
    total_cases=250,
)
assert live_metrics["error_rate"] == 0.5
assert "fail=1/2 (50.0%)" in format_parse_live_quality_progress(live_metrics)
print("[test] smoke_ok")
PY
fi

echo "[test] nl_bank_parser_v2_ok"
