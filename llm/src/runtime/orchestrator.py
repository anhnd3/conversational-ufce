from __future__ import annotations

from typing import Any

import pandas as pd

from llm.src.runtime.backends import CounterfactualBackend, resolve_counterfactual_backend
from llm.src.runtime.backend_packages.base import BackendExecutionResult
from llm.src.runtime.backend_packages.registry_defaults import build_default_backends
from llm.src.runtime.backend_packages.ufce.adapter import UFCECanonicalBackend
from llm.src.runtime.constraint_spec import apply_constraint_spec_to_candidates
from llm.src.runtime.contracts import (
    REASON_CODE_VERSION,
    CanonicalCandidate,
    CanonicalProfile,
    VerificationResult,
    canonical_candidates_to_legacy_result,
    canonical_request_from_legacy_request,
    legacy_candidate_to_canonical_candidate,
)
from llm.src.runtime.datasets import BankDatasetPackage, GradDatasetPackage
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.invariant_validator import RuntimeInvariantValidator
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.negotiation_controller import (
    PREDICTION_COMPLETE,
    READY_FOR_PREDICTION,
    READY_FOR_UFCE,
    TERMINAL_REJECT,
    TERMINAL_SUCCESS,
    UFCE_INFEASIBLE,
    UFCE_SUCCESS,
    NegotiationController,
)
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.prediction_service import PredictionService
from llm.src.runtime.profile_service import ProfileService
from llm.src.runtime.ranking.scorer import DefaultCandidateRanker
from llm.src.runtime.reason_codes import (
    INVALID_DATASET,
    NO_FEASIBLE_CF_FOUND,
    NO_RECOURSE_NEEDED,
    POLICY_NOT_FOUND,
    REQUEST_CONSTRAINTS_BLOCKED,
    UFCE_EXECUTION_ERROR,
)
from llm.src.runtime.registries.backend_registry import BackendRegistry
from llm.src.runtime.registries.dataset_registry import DatasetRegistry
from llm.src.runtime.reproducibility import (
    RUNTIME_MODE_STABLE_DEMO,
    build_deterministic_seed,
    sort_counterfactual_candidates,
)
from llm.src.runtime.types import (
    CounterfactualCandidate,
    CounterfactualResult,
    RuntimeDebugTrace,
    RuntimeResult,
)
from llm.src.runtime.ufce_request_builder import UFCERequestBuilder
from llm.src.runtime.verification.checks import (
    ConsistencyCheck,
    DatasetActionabilityCheck,
    DatasetDomainCheck,
    FlipCheck,
    HardConstraintCheck,
)
from llm.src.runtime.verification.verifier import CompositeCandidateVerifier


UFCE_METHOD_PRIORITY = {"sfexp": 0, "dfexp": 1, "tfexp": 2}


class RuntimeOrchestrator:
    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        policy_registry: PolicyRegistry | None = None,
        profile_service: ProfileService | None = None,
        prediction_service: PredictionService | None = None,
        ufce_request_builder: UFCERequestBuilder | None = None,
        counterfactual_service=None,
        counterfactual_backend: CounterfactualBackend | Any | None = None,
        counterfactual_backend_name: str = "ufce",
        invariant_validator: RuntimeInvariantValidator | None = None,
        runtime_mode: str = RUNTIME_MODE_STABLE_DEMO,
        dataset_registry: DatasetRegistry | None = None,
        backend_registry: BackendRegistry | None = None,
        candidate_verifier: CompositeCandidateVerifier | None = None,
        candidate_ranker: DefaultCandidateRanker | None = None,
    ) -> None:
        self.model_registry = model_registry or ModelRegistry()
        self.policy_registry = policy_registry or PolicyRegistry(self.model_registry)
        self.profile_service = profile_service or ProfileService()
        self.prediction_service = prediction_service or PredictionService()
        self.ufce_request_builder = ufce_request_builder or UFCERequestBuilder()
        self.dataset_registry = dataset_registry or DatasetRegistry(
            {
                "bank": BankDatasetPackage(self.model_registry),
                "grad": GradDatasetPackage(self.model_registry),
            }
        )
        self.backend_registry = backend_registry or BackendRegistry(backends=build_default_backends())
        self.candidate_verifier = candidate_verifier or CompositeCandidateVerifier(
            [
                FlipCheck(self.prediction_service),
                HardConstraintCheck(),
                ConsistencyCheck(),
                DatasetDomainCheck(),
                DatasetActionabilityCheck(),
            ]
        )
        self.candidate_ranker = candidate_ranker or DefaultCandidateRanker()
        self.runtime_mode = runtime_mode

        self._legacy_backend: CounterfactualBackend | None = None
        if counterfactual_backend is not None:
            self.counterfactual_backend = counterfactual_backend
            self.counterfactual_backend_name = getattr(
                counterfactual_backend,
                "backend_id",
                getattr(counterfactual_backend, "backend_name", counterfactual_backend_name),
            )
            if hasattr(counterfactual_backend, "backend_id"):
                self._canonical_backend = counterfactual_backend
                self._legacy_backend = None
            else:
                self._canonical_backend = None
                self._legacy_backend = counterfactual_backend
        elif counterfactual_service is not None:
            self._canonical_backend = UFCECanonicalBackend(
                mapper=None,
                normalizer=None,
                service=counterfactual_service,
            )
            self.counterfactual_backend = self._canonical_backend
            self.counterfactual_backend_name = self._canonical_backend.backend_id
        else:
            self._canonical_backend = self.backend_registry.get(counterfactual_backend_name)
            self.counterfactual_backend = self._canonical_backend
            self.counterfactual_backend_name = getattr(self._canonical_backend, "backend_id", counterfactual_backend_name)
        self.counterfactual_service = getattr(self.counterfactual_backend, "service", self.counterfactual_backend)
        self.invariant_validator = invariant_validator or RuntimeInvariantValidator(self.prediction_service)

    def handle(self, request: dict[str, Any], include_debug_trace: bool = False) -> RuntimeResult:
        debug_trace = RuntimeDebugTrace()
        controller = NegotiationController()
        dataset_name = self._extract_dataset_name(request)
        try:
            dataset_package = self._resolve_dataset_package(dataset_name)
            context = dataset_package.runtime_context()
            runtime_request = self.profile_service.parse_request(
                request,
                dataset_name,
                feature_order=list(context.bundle.feature_order),
            )
            canonical_profile = dataset_package.normalize_profile(runtime_request.profile)
            canonical_profile_dict = dict(canonical_profile.values)
            canonical_request = canonical_request_from_legacy_request(
                legacy_request=runtime_request,
                desired_outcome=context.policy.desired_outcome,
                canonical_profile=canonical_profile,
            )
            backend_manifest = self._load_backend_manifest()
            deterministic_seed_value = build_deterministic_seed(
                dataset_name=dataset_name,
                canonical_profile=canonical_profile_dict,
                feature_order=list(context.bundle.feature_order),
                policy_version=context.policy.policy_version,
            )
            debug_trace.runtime_mode = self.runtime_mode
            debug_trace.backend_name = self.counterfactual_backend_name
            debug_trace.deterministic_seed = deterministic_seed_value
            debug_trace.policy_version = context.policy.policy_version
            debug_trace.mi_feature_pairs = [list(pair) for pair in context.mi_feature_pairs]

            controller.transition(READY_FOR_PREDICTION)
            prediction = self.prediction_service.predict(
                dataset_name,
                pd.DataFrame([canonical_profile_dict], columns=context.bundle.feature_order),
                context,
            )
            controller.transition(PREDICTION_COMPLETE)

            if prediction.predicted_label == context.policy.desired_outcome:
                controller.transition(TERMINAL_SUCCESS)
                debug_trace.state_trace = list(controller.state_trace)
                result = RuntimeResult(
                    dataset=dataset_name,
                    controller_state=TERMINAL_SUCCESS,
                    prediction=prediction,
                    counterfactual=None,
                    reason_codes=[NO_RECOURSE_NEEDED],
                    runtime_mode=self.runtime_mode,
                    invariant_validation=self.invariant_validator.validate(
                        result=RuntimeResult(
                            dataset=dataset_name,
                            controller_state=TERMINAL_SUCCESS,
                            prediction=prediction,
                            counterfactual=None,
                            reason_codes=[NO_RECOURSE_NEEDED],
                            runtime_mode=self.runtime_mode,
                            canonical_request=canonical_request.to_dict(),
                            canonical_candidates=[],
                            verification_results=[],
                            backend_manifest=backend_manifest,
                            backend_id=self.counterfactual_backend_name,
                            reason_code_version=REASON_CODE_VERSION,
                        ),
                        current_profile=canonical_profile_dict,
                        context=context,
                    ),
                    debug_trace=debug_trace,
                    canonical_request=canonical_request.to_dict(),
                    canonical_candidates=[],
                    verification_results=[],
                    backend_manifest=backend_manifest,
                    backend_id=self.counterfactual_backend_name,
                    reason_code_version=REASON_CODE_VERSION,
                )
                return self._finalize_result(result, include_debug_trace)

            controller.transition(READY_FOR_UFCE)
            backend_result = self._generate_backend_result(
                canonical_request=canonical_request,
                canonical_profile=canonical_profile,
                dataset_package=dataset_package,
                context=context,
                deterministic_seed_value=deterministic_seed_value,
                debug_trace=debug_trace,
            )
            generated_candidates = list(backend_result.candidates)
            filtered_candidates, filter_reason_codes, constraint_filter = self._apply_request_constraints(
                candidates=generated_candidates,
                factual_profile=canonical_profile_dict,
                constraint_spec=runtime_request.constraint_spec,
                feature_order=list(context.bundle.feature_order),
            )
            if constraint_filter is not None:
                debug_trace.constraint_filter = constraint_filter
            verification_results: list[VerificationResult] = []
            ranked_candidates: list[CanonicalCandidate] = []
            reason_codes = list(filter_reason_codes or backend_result.reason_codes)
            if filtered_candidates:
                verified_set = self.candidate_verifier.verify_all(
                    filtered_candidates,
                    canonical_request,
                    dataset_package,
                    context,
                )
                verification_results = list(verified_set.verification_results)
                if verified_set.valid_candidates:
                    ranked_candidates = self.candidate_ranker.rank(
                        verified_set.valid_candidates,
                        canonical_request,
                        dataset_package,
                    )
                    reason_codes = []
                else:
                    reason_codes = list(reason_codes or [NO_FEASIBLE_CF_FOUND])

            legacy_counterfactual = canonical_candidates_to_legacy_result(
                candidates=ranked_candidates,
                failure_reason_codes=reason_codes,
            )
            debug_trace.generation_stats = {
                "backend_name": self.counterfactual_backend_name,
                "generated_candidate_count": len(generated_candidates),
                "counterfactual_feasible": bool(legacy_counterfactual.feasible),
                "reason_codes": list(reason_codes),
            }

            if legacy_counterfactual.feasible and legacy_counterfactual.candidates:
                winner = legacy_counterfactual.candidates[0]
                debug_trace.winning_path = {
                    "method": winner.method,
                    "rank": winner.rank,
                    "changed_features": list(winner.changed_features),
                }
                debug_trace.reject_path = None
                controller.transition(UFCE_SUCCESS)
                controller.transition(TERMINAL_SUCCESS)
                controller_state = TERMINAL_SUCCESS
            else:
                debug_trace.winning_path = None
                debug_trace.reject_path = {"reason_codes": list(reason_codes)}
                controller.transition(UFCE_INFEASIBLE)
                controller.transition(TERMINAL_REJECT)
                controller_state = TERMINAL_REJECT

            debug_trace.state_trace = list(controller.state_trace)
            result = RuntimeResult(
                dataset=dataset_name,
                controller_state=controller_state,
                prediction=prediction,
                counterfactual=legacy_counterfactual,
                reason_codes=list(reason_codes),
                runtime_mode=self.runtime_mode,
                invariant_validation=None,
                debug_trace=debug_trace,
                canonical_request=canonical_request.to_dict(),
                canonical_candidates=[candidate.to_dict() for candidate in generated_candidates],
                verification_results=[item.to_dict() for item in verification_results],
                backend_manifest=backend_manifest,
                backend_id=self.counterfactual_backend_name,
                reason_code_version=REASON_CODE_VERSION,
            )
            invariant_validation = self.invariant_validator.validate(
                result=result,
                current_profile=canonical_profile_dict,
                context=context,
            )
            result = RuntimeResult(
                dataset=result.dataset,
                controller_state=result.controller_state,
                prediction=result.prediction,
                counterfactual=result.counterfactual,
                reason_codes=list(result.reason_codes),
                runtime_mode=result.runtime_mode,
                invariant_validation=invariant_validation,
                debug_trace=result.debug_trace,
                canonical_request=result.canonical_request,
                canonical_candidates=list(result.canonical_candidates),
                verification_results=list(result.verification_results),
                backend_manifest=result.backend_manifest,
                backend_id=result.backend_id,
                reason_code_version=result.reason_code_version,
            )
            return self._finalize_result(result, include_debug_trace)
        except RuntimeServiceError as exc:
            return self._failure_result(
                dataset_name=dataset_name,
                controller=controller,
                debug_trace=debug_trace,
                reason_codes=list(exc.reason_codes),
                error_message=exc.message,
                include_debug_trace=include_debug_trace,
            )
        except Exception as exc:  # pragma: no cover - exercised in tests with service monkeypatching
            return self._failure_result(
                dataset_name=dataset_name,
                controller=controller,
                debug_trace=debug_trace,
                reason_codes=[UFCE_EXECUTION_ERROR],
                error_message=str(exc),
                include_debug_trace=include_debug_trace,
            )

    def _resolve_dataset_package(self, dataset_name: str | None) -> DatasetPackage:
        if not dataset_name:
            raise RuntimeServiceError((INVALID_DATASET,), "Request is missing dataset.")
        if not self.model_registry.has_dataset(dataset_name):
            raise RuntimeServiceError(
                (INVALID_DATASET,),
                "Unsupported dataset '{0}'.".format(dataset_name),
            )
        if not self.dataset_registry.has(dataset_name):
            raise RuntimeServiceError(
                (POLICY_NOT_FOUND,),
                "No enabled runtime dataset package for dataset '{0}'.".format(dataset_name),
            )
        dataset_package = self.dataset_registry.get(dataset_name)
        manifest = dataset_package.compatibility_manifest()
        if not manifest.live_runtime_enabled or not self.policy_registry.has_enabled_policy(dataset_name):
            raise RuntimeServiceError(
                (POLICY_NOT_FOUND,),
                "No enabled runtime policy for dataset '{0}'.".format(dataset_name),
            )
        if self.counterfactual_backend_name not in manifest.supported_backends:
            raise RuntimeServiceError(
                (POLICY_NOT_FOUND,),
                "Backend '{0}' is not supported for dataset '{1}'.".format(
                    self.counterfactual_backend_name,
                    dataset_name,
                ),
            )
        return dataset_package

    def _generate_backend_result(
        self,
        *,
        canonical_request,
        canonical_profile: CanonicalProfile,
        dataset_package: DatasetPackage,
        context,
        deterministic_seed_value: int,
        debug_trace: RuntimeDebugTrace,
    ) -> BackendExecutionResult:
        if self._legacy_backend is not None:
            legacy_request = self.ufce_request_builder.build(
                canonical_request.dataset_id,
                pd.DataFrame([canonical_profile.values], columns=context.bundle.feature_order),
                context,
            )
            legacy_result = self._legacy_backend.generate(
                legacy_request,
                context,
                deterministic_seed_value=deterministic_seed_value,
                debug_trace=debug_trace,
            )
            return BackendExecutionResult(
                backend_id=self.counterfactual_backend_name,
                candidates=_canonical_candidates_from_legacy_candidates(
                    candidates=legacy_result.candidates,
                    factual_profile=dict(canonical_profile.values),
                    backend_id=self.counterfactual_backend_name,
                ),
                reason_codes=list(legacy_result.reason_codes),
                failure_metadata={"feasible": bool(legacy_result.feasible)},
            )
        return self._canonical_backend.generate(
            canonical_request,
            dataset_package,
            context,
            deterministic_seed_value=deterministic_seed_value,
            debug_trace=debug_trace,
        )

    def _apply_request_constraints(
        self,
        *,
        candidates: list[CanonicalCandidate],
        factual_profile: dict[str, Any],
        constraint_spec: dict[str, Any] | None,
        feature_order: list[str],
    ) -> tuple[list[CanonicalCandidate], list[str], dict[str, Any] | None]:
        if not isinstance(constraint_spec, dict) or not candidates:
            return candidates, [], None
        legacy_candidates = [
            CounterfactualCandidate(
                method=str(candidate.backend_metadata.get("legacy_method", candidate.backend_id)),
                rank=int(candidate.backend_metadata.get("legacy_rank", index + 1)),
                profile=dict(candidate.new_values),
                changed_features=list(candidate.changed_features),
            )
            for index, candidate in enumerate(candidates)
        ]
        filtered, debug_summary = apply_constraint_spec_to_candidates(
            result=CounterfactualResult(
                feasible=True,
                candidates=legacy_candidates,
                reason_codes=[],
            ),
            constraint_spec=constraint_spec,
            feature_order=feature_order,
            sort_candidates=sort_counterfactual_candidates,
            request_constraints_blocked_code=REQUEST_CONSTRAINTS_BLOCKED,
        )
        if not filtered.feasible:
            return [], list(filtered.reason_codes), debug_summary
        original_by_profile = {
            tuple(sorted(candidate.new_values.items())): candidate
            for candidate in candidates
        }
        restored: list[CanonicalCandidate] = []
        for index, legacy_candidate in enumerate(filtered.candidates):
            key = tuple(sorted(dict(legacy_candidate.profile).items()))
            original = original_by_profile.get(key)
            if original is None:
                original = legacy_candidate_to_canonical_candidate(
                    backend_id=self.counterfactual_backend_name,
                    candidate=legacy_candidate,
                    factual_profile=factual_profile,
                )
            metadata = dict(original.backend_metadata)
            metadata["rank_hint"] = [index]
            restored.append(
                CanonicalCandidate(
                    backend_id=original.backend_id,
                    candidate_id=original.candidate_id,
                    changed_features=list(original.changed_features),
                    original_values=dict(original.original_values),
                    new_values=dict(original.new_values),
                    delta_summary=[dict(item) for item in original.delta_summary],
                    predicted_outcome=original.predicted_outcome,
                    raw_backend_score=original.raw_backend_score,
                    backend_metadata=metadata,
                )
            )
        return restored, [], debug_summary

    def _extract_dataset_name(self, request: Any) -> str | None:
        if not isinstance(request, dict):
            return None
        dataset_name = request.get("dataset")
        if not isinstance(dataset_name, str):
            return None
        normalized = dataset_name.strip().lower()
        return normalized or None

    def _load_backend_manifest(self) -> dict[str, Any] | None:
        try:
            return self.backend_registry.manifest(self.counterfactual_backend_name).to_dict()
        except Exception:
            return None

    def _failure_result(
        self,
        dataset_name: str | None,
        controller: NegotiationController,
        debug_trace: RuntimeDebugTrace,
        reason_codes: list[str],
        error_message: str,
        include_debug_trace: bool,
    ) -> RuntimeResult:
        if controller.current_state != TERMINAL_REJECT:
            controller.transition(TERMINAL_REJECT)
        debug_trace.state_trace = list(controller.state_trace)
        debug_trace.add_service_error("orchestrator", error_message)
        result = RuntimeResult(
            dataset=dataset_name,
            controller_state=TERMINAL_REJECT,
            prediction=None,
            counterfactual=None,
            reason_codes=reason_codes,
            runtime_mode=self.runtime_mode,
            invariant_validation=None,
            debug_trace=debug_trace,
            backend_manifest=self._load_backend_manifest(),
            backend_id=self.counterfactual_backend_name,
            reason_code_version=REASON_CODE_VERSION,
        )
        return self._finalize_result(result, include_debug_trace)

    def _finalize_result(self, result: RuntimeResult, include_debug_trace: bool) -> RuntimeResult:
        del include_debug_trace
        return result


def _canonical_candidates_from_legacy_candidates(
    *,
    candidates: list[CounterfactualCandidate],
    factual_profile: dict[str, Any],
    backend_id: str,
) -> list[CanonicalCandidate]:
    normalized: list[CanonicalCandidate] = []
    for index, candidate in enumerate(candidates):
        payload = legacy_candidate_to_canonical_candidate(
            backend_id=backend_id,
            candidate=candidate,
            factual_profile=factual_profile,
        )
        metadata = dict(payload.backend_metadata)
        if backend_id == "ufce":
            metadata["rank_hint"] = [
                UFCE_METHOD_PRIORITY.get(str(candidate.method), 99),
                int(candidate.rank),
            ]
        else:
            metadata["rank_hint"] = [int(candidate.rank), index]
        normalized.append(
            CanonicalCandidate(
                backend_id=payload.backend_id,
                candidate_id=payload.candidate_id,
                changed_features=list(payload.changed_features),
                original_values=dict(payload.original_values),
                new_values=dict(payload.new_values),
                delta_summary=[dict(item) for item in payload.delta_summary],
                predicted_outcome=payload.predicted_outcome,
                raw_backend_score=payload.raw_backend_score,
                backend_metadata=metadata,
            )
        )
    return normalized
