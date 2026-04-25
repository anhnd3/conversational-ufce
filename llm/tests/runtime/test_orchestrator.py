from __future__ import annotations

import pytest

from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult
from llm.src.runtime.reason_codes import (
    INVALID_DATASET,
    NO_FEASIBLE_CF_FOUND,
    NO_RECOURSE_NEEDED,
    POLICY_NOT_FOUND,
    UFCE_EXECUTION_ERROR,
)

pytestmark = pytest.mark.filterwarnings("ignore:tostring\\(\\) is deprecated:DeprecationWarning")


def test_orchestrator_rejects_unknown_dataset():
    orchestrator = RuntimeOrchestrator()

    result = orchestrator.handle({"dataset": "unknown", "profile": {}}, include_debug_trace=True)

    assert result.controller_state == "TERMINAL_REJECT"
    assert result.reason_codes == [INVALID_DATASET]
    assert result.debug_trace is not None
    assert result.debug_trace.state_trace == ["TERMINAL_REJECT"]


def test_orchestrator_rejects_known_but_disabled_dataset():
    orchestrator = RuntimeOrchestrator()

    result = orchestrator.handle({"dataset": "movie", "profile": {}}, include_debug_trace=True)

    assert result.controller_state == "TERMINAL_REJECT"
    assert result.reason_codes == [POLICY_NOT_FOUND]


def test_orchestrator_maps_service_exception_to_terminal_failure(monkeypatch):
    orchestrator = RuntimeOrchestrator()

    def boom(*args, **kwargs):
        raise RuntimeServiceError((UFCE_EXECUTION_ERROR,), "injected failure")

    monkeypatch.setattr(orchestrator.counterfactual_service, "generate", boom)
    request = {
        "dataset": "bank",
        "profile": {
            "Income": 100,
            "Family": 1,
            "CCAvg": 2.7,
            "Education": 2,
            "Mortgage": 0,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 0,
        },
    }

    result = orchestrator.handle(request, include_debug_trace=True)

    assert result.controller_state == "TERMINAL_REJECT"
    assert result.reason_codes == [UFCE_EXECUTION_ERROR]
    assert result.debug_trace is not None
    assert result.debug_trace.state_trace[-1] == "TERMINAL_REJECT"


def test_orchestrator_positive_fixture_returns_no_recourse_needed():
    orchestrator = RuntimeOrchestrator()
    request = {
        "dataset": "bank",
        "profile": {
            "Income": 140,
            "Family": 2,
            "CCAvg": 7.7376709303,
            "Education": 2,
            "Mortgage": 32,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 1,
            "CreditCard": 0,
        },
    }

    result = orchestrator.handle(request)

    assert result.controller_state == "TERMINAL_SUCCESS"
    assert result.counterfactual is None
    assert result.reason_codes == [NO_RECOURSE_NEEDED]
    assert result.prediction is not None
    assert result.prediction.predicted_label == 1


def test_orchestrator_feasible_negative_fixture_returns_candidate():
    orchestrator = RuntimeOrchestrator()
    request = {
        "dataset": "bank",
        "profile": {
            "Income": 100,
            "Family": 1,
            "CCAvg": 2.7,
            "Education": 2,
            "Mortgage": 0,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 0,
        },
    }

    result = orchestrator.handle(request)

    assert result.controller_state == "TERMINAL_SUCCESS"
    assert result.reason_codes == []
    assert result.counterfactual is not None
    assert result.counterfactual.feasible is True
    assert any(
        candidate.method == "sfexp"
        and candidate.profile["CDAccount"] == 1
        for candidate in result.counterfactual.candidates
    )


def test_orchestrator_infeasible_negative_fixture_returns_reject():
    orchestrator = RuntimeOrchestrator()
    request = {
        "dataset": "bank",
        "profile": {
            "Income": 49,
            "Family": 4,
            "CCAvg": 1.6,
            "Education": 1,
            "Mortgage": 0,
            "SecuritiesAccount": 1,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 0,
        },
    }

    result = orchestrator.handle(request)

    assert result.controller_state == "TERMINAL_REJECT"
    assert result.counterfactual is not None
    assert result.counterfactual.feasible is False
    assert result.reason_codes == [NO_FEASIBLE_CF_FOUND]


def test_orchestrator_supports_injected_counterfactual_backend():
    class StubBackend:
        backend_name = "dice"

        def generate(self, request, context, *, deterministic_seed_value=None, debug_trace=None):
            del request
            del context
            del deterministic_seed_value
            del debug_trace
            return CounterfactualResult(
                feasible=True,
                candidates=[
                    CounterfactualCandidate(
                        method="DiCE",
                        rank=1,
                        profile={
                            "Income": 100.0,
                            "Family": 1,
                            "CCAvg": 2.7,
                            "Education": 2,
                            "Mortgage": 0.0,
                            "SecuritiesAccount": 0,
                            "CDAccount": 1,
                            "Online": 0,
                            "CreditCard": 0,
                        },
                        changed_features=["CDAccount"],
                    )
                ],
                reason_codes=[],
            )

    orchestrator = RuntimeOrchestrator(counterfactual_backend=StubBackend())
    request = {
        "dataset": "bank",
        "profile": {
            "Income": 100,
            "Family": 1,
            "CCAvg": 2.7,
            "Education": 2,
            "Mortgage": 0,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 0,
        },
    }

    result = orchestrator.handle(request, include_debug_trace=True)

    assert result.controller_state == "TERMINAL_SUCCESS"
    assert result.counterfactual is not None
    assert result.counterfactual.candidates[0].method == "DiCE"
    assert result.debug_trace is not None
    assert result.debug_trace.backend_name == "dice"
    assert result.debug_trace.generation_stats == {
        "backend_name": "dice",
        "generated_candidate_count": 1,
        "counterfactual_feasible": True,
        "reason_codes": [],
    }
