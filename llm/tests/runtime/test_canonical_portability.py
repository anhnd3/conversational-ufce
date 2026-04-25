from __future__ import annotations

from llm.src.conversation.canonical_validator import DatasetRoutedCanonicalValidator
from llm.src.runtime.datasets import BankDatasetPackage, GradDatasetPackage
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.reason_codes import NO_FLIP
from llm.src.runtime.registries.backend_registry import BackendRegistry
from llm.src.runtime.registries.dataset_registry import DatasetRegistry
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult
from llm.src.validation.schema_validator import ValidationResult


def test_dataset_registry_and_validator_delegate_to_bank_by_default():
    validator = DatasetRoutedCanonicalValidator()
    registry = DatasetRegistry(
        {
            "bank": BankDatasetPackage(),
            "grad": GradDatasetPackage(),
        }
    )

    result = validator.validate(
        candidate={
            "status": "partial",
            "cf_request": {"Income": 40},
            "missing_fields": ["Family"],
            "conflicts": [],
            "notes": [],
        },
        schema_validation=ValidationResult(
            is_valid=True,
            errors=(),
            unexpected_top_level_keys=(),
            unexpected_cf_fields=(),
        ),
    )

    assert registry.has("bank") is True
    assert registry.has("grad") is True
    assert validator.dataset_id == "bank"
    assert result.final_stage == "NEEDS_CLARIFICATION"
    assert "Income" in validator.required_fields


def test_backend_registry_loads_canonical_backend_manifest():
    registry = BackendRegistry(
        backends={
            "ufce": object(),
            "dice": object(),
            "ar": object(),
        }
    )

    manifest = registry.manifest("ufce")

    assert manifest.backend_id == "ufce"
    assert manifest.request_contract_version == "canonical_request_v1"
    assert manifest.candidate_contract_version == "canonical_candidate_v1"
    assert manifest.capabilities["supports_multi_candidate"] is True


def test_runtime_orchestrator_uses_verified_fallback_candidate():
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
                        method="stub_invalid",
                        rank=1,
                        profile={
                            "Income": 100.0,
                            "Family": 1,
                            "CCAvg": 2.7,
                            "Education": 2,
                            "Mortgage": 0.0,
                            "SecuritiesAccount": 0,
                            "CDAccount": 0,
                            "Online": 0,
                            "CreditCard": 0,
                        },
                        changed_features=[],
                    ),
                    CounterfactualCandidate(
                        method="stub_valid",
                        rank=2,
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
                    ),
                ],
                reason_codes=[],
            )

    orchestrator = RuntimeOrchestrator(counterfactual_backend=StubBackend())
    result = orchestrator.handle(
        {
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
    )

    assert result.controller_state == "TERMINAL_SUCCESS"
    assert result.counterfactual is not None
    assert [candidate.method for candidate in result.counterfactual.candidates] == ["stub_valid"]
    assert result.backend_id == "dice"
    assert result.reason_code_version == "reason_codes_v1"
    assert result.canonical_request is not None
    assert len(result.canonical_candidates) == 2
    assert any(
        verification["reason_codes"] == [NO_FLIP]
        for verification in result.verification_results
    )
