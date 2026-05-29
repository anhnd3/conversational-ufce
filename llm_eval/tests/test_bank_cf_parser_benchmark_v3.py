from __future__ import annotations

from pathlib import Path

from llm_eval.config import load_benchmark


ROOT = Path(__file__).resolve().parents[2]
EN_PATH = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_en.yaml"
VI_PATH = ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v3_vi.yaml"


def test_bank_cf_parser_benchmark_v3_en_contract():
    benchmark = load_benchmark(EN_PATH)

    assert benchmark.benchmark_name == "ufce_bank_cf_parser_v3_en"
    assert len(benchmark.cases) == 8
    assert {case.group for case in benchmark.cases} == {"NL_EN"}
    assert all("field_evidence" in case.expected_output for case in benchmark.cases)


def test_bank_cf_parser_benchmark_v3_vi_contract():
    benchmark = load_benchmark(VI_PATH)

    assert benchmark.benchmark_name == "ufce_bank_cf_parser_v3_vi"
    assert len(benchmark.cases) == 8
    assert {case.group for case in benchmark.cases} == {"NL_VI"}
    assert all("field_evidence" in case.expected_output for case in benchmark.cases)
