from __future__ import annotations

from pathlib import Path

from llm_eval.config import load_benchmark


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v2.yaml"


def test_bank_cf_parser_benchmark_v2_covers_explicit_nl_methodology():
    benchmark = load_benchmark(BENCHMARK_PATH)

    assert benchmark.benchmark_name == "ufce_bank_cf_parser_v2"
    assert len(benchmark.cases) == 12
    assert {case.group for case in benchmark.cases} == {"NL"}

    descriptions = {case.case_id: case.description.lower() for case in benchmark.cases}
    assert "explicit semantic values" in descriptions["NL01"]
    assert "coordinated boolean negation" in descriptions["NL02"]
    assert "qualitative numeric phrases" in descriptions["NL05"]
    assert "degree label" in descriptions["NL06"]
    assert "contradiction" in descriptions["NL07"]
    assert "synonym labels" in descriptions["NL09"]
    assert "not converted" in descriptions["NL10"]
    assert "boolean contradiction" in descriptions["NL11"]
    assert "missing numeric facts" in descriptions["NL12"]


def test_bank_cf_parser_benchmark_v2_preserves_integrity_cases():
    benchmark = load_benchmark(BENCHMARK_PATH)

    unit_case = benchmark.case_map["NL10"]
    assert unit_case.expected_output["status"] == "needs_clarification"
    assert "Income" not in unit_case.expected_output["cf_request"]
    assert "CCAvg" not in unit_case.expected_output["cf_request"]
    assert unit_case.expected_output["missing_fields"] == [
        "Income",
        "CCAvg",
        "SecuritiesAccount",
        "CDAccount",
        "CreditCard",
    ]

    conflict_case = benchmark.case_map["NL11"]
    assert conflict_case.expected_output["status"] == "conflict"
    assert "Online" not in conflict_case.expected_output["cf_request"]
    assert conflict_case.expected_output["conflicts"] == [
        "Explicit field 'Online' has conflicting values in the same turn."
    ]
