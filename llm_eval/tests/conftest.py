from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_eval.config import benchmark_from_dict


@pytest.fixture
def sample_benchmark_payload() -> dict:
    return {
        "benchmark_name": "ufce_bank_cf_parser_v1",
        "description": "Sample benchmark payload for tests.",
        "target_cf_fields": [
            {"name": "Income", "type": "float", "description": "Target income value"},
            {"name": "CCAvg", "type": "float", "description": "Average credit card spending"},
            {"name": "Family", "type": "int", "description": "Family size"},
            {"name": "Education", "type": "int", "description": "Education code"},
            {"name": "Mortgage", "type": "float", "description": "Mortgage value"},
            {"name": "CDAccount", "type": "binary", "description": "1=yes, 0=no"},
            {"name": "Online", "type": "binary", "description": "1=yes, 0=no"},
            {
                "name": "SecuritiesAccount",
                "type": "binary",
                "description": "1=yes, 0=no",
            },
            {"name": "CreditCard", "type": "binary", "description": "1=yes, 0=no"},
        ],
        "output_contract": {
            "task": "extract_cf_request",
            "status_enum": ["complete", "partial", "needs_clarification", "conflict"],
            "rules": [
                "Return only fields explicitly inferable from the input.",
                "Do not invent missing values.",
            ],
        },
        "cases": [
            {
                "case_id": "A01",
                "group": "A",
                "description": "Complete case",
                "input": "Income 40, CCAvg 1.5, Family 3, Education 2, Mortgage 80, CDAccount yes, Online yes, SecuritiesAccount no, CreditCard yes.",
                "expected_output": {
                    "task": "extract_cf_request",
                    "status": "complete",
                    "cf_request": {
                        "Income": 40,
                        "CCAvg": 1.5,
                        "Family": 3,
                        "Education": 2,
                        "Mortgage": 80,
                        "CDAccount": 1,
                        "Online": 1,
                        "SecuritiesAccount": 0,
                        "CreditCard": 1,
                    },
                    "missing_fields": [],
                    "conflicts": [],
                    "notes": [],
                },
            },
            {
                "case_id": "B01",
                "group": "B",
                "description": "Partial case",
                "input": "Income 40 and Online yes.",
                "expected_output": {
                    "task": "extract_cf_request",
                    "status": "partial",
                    "cf_request": {
                        "Income": 40,
                        "Online": 1,
                    },
                    "missing_fields": [
                        "CCAvg",
                        "Family",
                        "Education",
                        "Mortgage",
                        "CDAccount",
                        "SecuritiesAccount",
                        "CreditCard",
                    ],
                    "conflicts": [],
                    "notes": [],
                },
            },
        ],
    }


@pytest.fixture
def sample_benchmark(sample_benchmark_payload):
    return benchmark_from_dict(sample_benchmark_payload)
