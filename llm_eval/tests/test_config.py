from __future__ import annotations

from argparse import Namespace

import pytest

from llm_eval.config import benchmark_from_dict, build_arg_parser, config_from_args, load_benchmark, select_cases


def test_benchmark_from_dict_builds_expected_contract(sample_benchmark_payload):
    benchmark = benchmark_from_dict(sample_benchmark_payload)

    assert benchmark.benchmark_name == "ufce_bank_cf_parser_v1"
    assert benchmark.allowed_field_names[0] == "Income"
    assert benchmark.field_type_map["CDAccount"] == "binary"
    assert benchmark.case_map["A01"].group == "A"


def test_select_cases_filters_group_and_case_ids(sample_benchmark):
    config = config_from_args(
        Namespace(
            benchmark="llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml",
            model_alias="test-model",
            out_dir="llm_eval/outputs",
            api_base="http://localhost:1234",
            repeats=3,
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            case_ids="B01",
            group="B",
            limit=None,
            timeout_s=120.0,
            system_prompt="llm/prompts/parser_system_prompt_v1.txt",
        )
    )

    selected = select_cases(sample_benchmark, config)

    assert [case.case_id for case in selected] == ["B01"]


def test_load_benchmark_from_yaml(tmp_path, sample_benchmark_payload):
    yaml = pytest.importorskip("yaml")
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.safe_dump(sample_benchmark_payload), encoding="utf-8")

    benchmark = load_benchmark(benchmark_path)

    assert benchmark.case_map["B01"].expected_output["status"] == "partial"


def test_build_arg_parser_uses_five_minute_timeout_default():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--benchmark",
            "llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml",
            "--model_alias",
            "test-model",
            "--out_dir",
            "llm_eval/outputs",
        ]
    )

    assert args.timeout_s == 300.0
