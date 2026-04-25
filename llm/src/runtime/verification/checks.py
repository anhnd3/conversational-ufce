from __future__ import annotations

from typing import Any

import pandas as pd

from llm.src.runtime.contracts import CanonicalCandidate, CanonicalRecourseRequest, VerificationResult, build_delta_summary
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.prediction_service import PredictionService
from llm.src.runtime.reason_codes import (
    HARD_CONSTRAINT_VIOLATION,
    INCONSISTENT_DELTA_SUMMARY,
    NO_FLIP,
)
from llm.src.runtime.types import RuntimeContext


class FlipCheck:
    def __init__(self, prediction_service: PredictionService | None = None) -> None:
        self.prediction_service = prediction_service or PredictionService()

    def verify(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerificationResult:
        del dataset
        candidate_frame = pd.DataFrame([candidate.new_values], columns=context.bundle.feature_order)
        prediction = self.prediction_service.predict(request.dataset_id, candidate_frame, context)
        if prediction.predicted_label == context.policy.desired_outcome:
            return VerificationResult(
                is_valid=True,
                candidate_id=candidate.candidate_id,
                evidence={"predicted_label": prediction.predicted_label},
            )
        return VerificationResult(
            is_valid=False,
            candidate_id=candidate.candidate_id,
            reason_codes=[NO_FLIP],
            evidence={"predicted_label": prediction.predicted_label},
        )


class HardConstraintCheck:
    def verify(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerificationResult:
        del dataset
        del context
        reasons: list[str] = []
        hard_constraints = dict(request.hard_constraints)
        immutable = {
            item
            for item in hard_constraints.get("immutable", [])
            if isinstance(item, str)
        }
        disallowed = {
            item
            for item in hard_constraints.get("disallowed_changes", [])
            if isinstance(item, str)
        }
        blocked = immutable | disallowed | set(request.forbidden_features)
        if any(feature in blocked for feature in candidate.changed_features):
            reasons.append(HARD_CONSTRAINT_VIOLATION)
        if isinstance(request.max_changes, int) and len(candidate.changed_features) > request.max_changes:
            reasons.append(HARD_CONSTRAINT_VIOLATION)
        numeric_bounds = hard_constraints.get("numeric_bounds")
        if isinstance(numeric_bounds, dict):
            for field_name, bounds in numeric_bounds.items():
                if field_name not in candidate.new_values or not isinstance(bounds, dict):
                    continue
                value = float(candidate.new_values[field_name])
                if "min" in bounds and value < float(bounds["min"]):
                    reasons.append(HARD_CONSTRAINT_VIOLATION)
                if "max" in bounds and value > float(bounds["max"]):
                    reasons.append(HARD_CONSTRAINT_VIOLATION)
        return VerificationResult(
            is_valid=not reasons,
            candidate_id=candidate.candidate_id,
            reason_codes=reasons,
            evidence={"changed_features": list(candidate.changed_features)},
        )


class ConsistencyCheck:
    def verify(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerificationResult:
        del request
        del dataset
        del context
        expected_delta = build_delta_summary(
            original_values=candidate.original_values,
            new_values=candidate.new_values,
            changed_features=list(candidate.changed_features),
        )
        if expected_delta == candidate.delta_summary:
            return VerificationResult(is_valid=True, candidate_id=candidate.candidate_id)
        return VerificationResult(
            is_valid=False,
            candidate_id=candidate.candidate_id,
            reason_codes=[INCONSISTENT_DELTA_SUMMARY],
            evidence={"expected_delta_summary": expected_delta},
        )


class DatasetDomainCheck:
    def verify(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerificationResult:
        del context
        return dataset.legality_check(candidate, request)


class DatasetActionabilityCheck:
    def verify(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerificationResult:
        del context
        return dataset.actionability_check(candidate, request)
