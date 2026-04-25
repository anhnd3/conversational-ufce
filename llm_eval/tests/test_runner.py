from __future__ import annotations

from pathlib import Path

from llm_eval.models import EvalConfig
from llm_eval.runner import evaluate_model_output, run_evaluation


def test_evaluate_model_output_skips_empty_parse_on_api_error(sample_benchmark):
    normalized, validation_result = evaluate_model_output(
        api_result={"error": "HTTP 400: missing_required_parameter, param=model, message=Required"},
        response_data={"message_text": ""},
        benchmark=sample_benchmark,
    )

    assert normalized.normalized_text == ""
    assert normalized.parsed_json is None
    assert normalized.parse_error is None
    assert validation_result.is_valid is False
    assert validation_result.errors == ()


def test_run_evaluation_tracks_total_requests_with_tqdm(monkeypatch, tmp_path, sample_benchmark):
    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, total, desc, unit, dynamic_ncols):
            self.total = total
            self.initial_desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.descriptions: list[str] = [desc]
            self.postfixes: list[str] = []
            self.updates: list[int] = []
            self.closed = False
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True

        def set_description(self, desc, refresh=True):
            del refresh
            self.descriptions.append(desc)

        def set_postfix_str(self, text, refresh=True):
            del refresh
            self.postfixes.append(text)

        def update(self, n=1):
            self.updates.append(n)

    config = EvalConfig(
        benchmark_path=tmp_path / "benchmark.yaml",
        model_alias="qwen3.5-27b@q4_0",
        out_dir=tmp_path / "outputs",
        api_base="http://localhost:1234",
        repeats=2,
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        timeout_s=120.0,
        case_ids=(),
        group=None,
        limit=None,
        system_prompt_path=tmp_path / "system_prompt.txt",
    )
    config.out_dir.mkdir()
    cases = [sample_benchmark.case_map["A01"], sample_benchmark.case_map["B01"]]

    monkeypatch.setattr("llm_eval.runner.tqdm", FakeTqdm)
    monkeypatch.setattr("llm_eval.runner.load_benchmark", lambda path: sample_benchmark)
    monkeypatch.setattr("llm_eval.runner.select_cases", lambda benchmark, cfg: cases)
    monkeypatch.setattr("llm_eval.runner.load_system_prompt", lambda path: "system prompt")
    monkeypatch.setattr("llm_eval.runner.make_run_id", lambda: "run_20260308_010203")
    monkeypatch.setattr(
        "llm_eval.runner.call_lm_studio",
        lambda api_base, payload, timeout_s: {
            "ok": True,
            "status_code": 200,
            "error": None,
            "response_json": {"output": []},
            "response_text": None,
            "elapsed_ms": 123.4,
        },
    )
    monkeypatch.setattr(
        "llm_eval.runner.extract_response_data",
        lambda response_json, elapsed_ms: {
            "message_text": (
                '{"task":"extract_cf_request","status":"complete",'
                '"cf_request":{},"missing_fields":[],"conflicts":[],"notes":[]}'
            ),
            "reasoning_text": "",
            "usage": {},
            "stats": {},
            "derived_metrics": {"request_latency_ms": elapsed_ms},
        },
    )
    monkeypatch.setattr("llm_eval.runner.write_outputs", lambda **kwargs: None)

    result = run_evaluation(config)

    progress = FakeTqdm.instances[0]
    assert progress.total == len(cases) * config.repeats
    assert progress.unit == "req"
    assert progress.dynamic_ncols is True
    assert progress.updates == [1, 1, 1, 1]
    assert "A01 repeat 1/2" in progress.descriptions
    assert "B01 repeat 2/2" in progress.descriptions
    assert progress.postfixes == [
        "http=200, latency_ms=123.4",
        "http=200, latency_ms=123.4",
        "http=200, latency_ms=123.4",
        "http=200, latency_ms=123.4",
    ]
    assert progress.closed is True
    assert result["run_dir"] == Path(config.out_dir) / "qwen3.5-27b_q4_0_20260308_010203"
