from __future__ import annotations

import pandas as pd

from llm.src.runtime.counterfactual_service import CounterfactualService
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.reason_codes import NO_FEASIBLE_CF_FOUND, UFCE_EXECUTION_ERROR
from llm.src.runtime.types import RuntimeDebugTrace
from llm.src.runtime.ufce_request_builder import UFCERequestBuilder


def build_request():
    registry = ModelRegistry()
    context = PolicyRegistry(registry).get_runtime_context("bank")
    builder = UFCERequestBuilder()
    canonical_profile = pd.DataFrame(
        [
            {
                "Income": 100.0,
                "Family": 1,
                "CCAvg": 2.7,
                "Education": 2,
                "Mortgage": 0.0,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 0,
                "CreditCard": 0,
            }
        ],
        columns=context.bundle.feature_order,
    )
    return builder.build("bank", canonical_profile, context)


def test_counterfactual_service_init_is_silent(capsys):
    CounterfactualService()

    captured = capsys.readouterr()

    assert captured.out == ""


def test_counterfactual_service_maps_empty_outputs_to_no_feasible(monkeypatch):
    service = CounterfactualService()
    request = build_request()

    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.sfexp",
        lambda *args, **kwargs: (pd.DataFrame(), 0.0, []),
    )
    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.dfexp",
        lambda *args, **kwargs: (pd.DataFrame(), 0.0, []),
    )
    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.tfexp",
        lambda *args, **kwargs: (pd.DataFrame(), 0.0, []),
    )

    result = service.generate(request, RuntimeDebugTrace())

    assert result.feasible is False
    assert result.reason_codes == [NO_FEASIBLE_CF_FOUND]


def test_counterfactual_service_maps_raised_errors_to_execution_error(monkeypatch):
    service = CounterfactualService()
    request = build_request()
    debug_trace = RuntimeDebugTrace()

    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.sfexp",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.dfexp",
        lambda *args, **kwargs: (pd.DataFrame(), 0.0, []),
    )
    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.tfexp",
        lambda *args, **kwargs: (pd.DataFrame(), 0.0, []),
    )

    result = service.generate(request, debug_trace)

    assert result.feasible is False
    assert result.reason_codes == [UFCE_EXECUTION_ERROR]
    assert debug_trace.service_errors[0]["service"] == "sfexp"


def test_counterfactual_service_hides_execution_error_when_other_method_succeeds(monkeypatch):
    service = CounterfactualService()
    request = build_request()
    debug_trace = RuntimeDebugTrace()

    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.sfexp",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.dfexp",
        lambda *args, **kwargs: (
            pd.DataFrame(
                [
                    {
                        "Income": 100.0,
                        "Family": 1,
                        "CCAvg": 2.7,
                        "Education": 2,
                        "Mortgage": 0.0,
                        "SecuritiesAccount": 0,
                        "CDAccount": 1,
                        "Online": 0,
                        "CreditCard": 0,
                    }
                ]
            ),
            0.0,
            [0],
        ),
    )
    monkeypatch.setattr(
        "llm.src.runtime.counterfactual_service.tfexp",
        lambda *args, **kwargs: (pd.DataFrame(), 0.0, []),
    )

    result = service.generate(request, debug_trace)

    assert result.feasible is True
    assert result.reason_codes == []
    assert result.candidates[0].changed_features == ["CDAccount"]
