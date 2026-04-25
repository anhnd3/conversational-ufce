from __future__ import annotations

from dataclasses import dataclass

from llm.src.part2_eval.annotation_scoring import score_initial_constraint_cases


@dataclass(frozen=True)
class _StubParserResult:
    message_text: str
    api_error: str | None = None
    derived_metrics: dict | None = None


class _StubParserAdapter:
    def __init__(self, result: _StubParserResult) -> None:
        self.result = result
        self.repair_calls = 0

    def parse(self, *, user_text: str, benchmark=None):
        del user_text
        del benchmark
        return self.result

    def repair(self, *, invalid_output: str, errors: list[str], benchmark=None):
        del invalid_output
        del errors
        del benchmark
        self.repair_calls += 1
        raise AssertionError("repair should not be called in this scorer-path regression")


def test_score_initial_constraint_cases_recovers_missing_constraint_spec_from_user_text(sample_benchmark):
    adapter = _StubParserAdapter(
        _StubParserResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            ),
            derived_metrics={"request_latency_ms": 12.0},
        )
    )

    summary = score_initial_constraint_cases(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        cases=[
            {
                "case_id": "tier_a_constraint_recovery",
                "annotation_type": "initial_constraint",
                "input_text": (
                    "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
                    "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no. Do not change Income."
                ),
                "expected_constraint_spec": {
                    "disallowed_changes": ["Income"]
                },
            }
        ],
        progress_enabled=False,
    )

    case_result = summary["per_case_results"][0]

    assert summary["M1_json_validity_rate"]["mean"] == 1.0
    assert summary["M2_schema_compliance_rate"]["mean"] == 1.0
    assert summary["M3_canonical_validation_pass_rate"]["mean"] == 1.0
    assert summary["M4_repair_rate"]["mean"] == 0.0
    assert summary["M6_constraint_extraction_fidelity"]["mean"] == 1.0
    assert case_result["predicted_constraint_spec"] == {"disallowed_changes": ["Income"]}
    assert case_result["parser_quality"]["flags"]["canonical_pass_after_quality"] is True
    assert "constraint_spec_recovered" in case_result["parser_quality"]["reason_codes"]
