from __future__ import annotations

from typing import Any

import pandas as pd

from llm.src.runtime.prediction_service import PredictionService
from llm.src.runtime.reason_codes import INVALID_COUNTERFACTUAL_BLOCKED, NO_RECOURSE_NEEDED
from llm.src.runtime.types import InvariantValidationResult, RuntimeContext, RuntimeResult


class RuntimeInvariantValidator:
    def __init__(self, prediction_service: PredictionService | None = None) -> None:
        self.prediction_service = prediction_service or PredictionService()

    def validate(
        self,
        *,
        result: RuntimeResult,
        current_profile: dict[str, Any],
        context: RuntimeContext,
    ) -> InvariantValidationResult:
        if result.reason_codes == [NO_RECOURSE_NEEDED]:
            return InvariantValidationResult(
                status="skipped_no_counterfactual",
                public_safe=True,
                reason_codes=[],
                validated_summary_type="no_recourse_needed",
                validated_changed_fields=[],
                details={"reason": "no_counterfactual_required"},
            )

        if result.counterfactual is None or not result.counterfactual.feasible:
            return InvariantValidationResult(
                status="skipped_no_counterfactual",
                public_safe=True,
                reason_codes=[],
                validated_summary_type="runtime_reject",
                validated_changed_fields=[],
                details={"reason": "no_counterfactual_candidate"},
            )

        if not result.counterfactual.candidates:
            return self._failed(["missing_counterfactual_candidate"])

        candidate = result.counterfactual.candidates[0]
        violations: list[str] = []
        candidate_profile = dict(candidate.profile)
        feature_order = list(context.bundle.feature_order)
        allowed_changes = set(context.policy.f2change)
        protected = set(context.policy.protected_features)

        missing_fields = [feature for feature in feature_order if feature not in candidate_profile]
        if missing_fields:
            violations.append("missing_candidate_fields")

        actual_changed_fields = [
            feature
            for feature in feature_order
            if current_profile.get(feature) != candidate_profile.get(feature)
        ]
        if list(candidate.changed_features) != actual_changed_fields:
            violations.append("changed_fields_mismatch")
        if any(feature not in allowed_changes for feature in actual_changed_fields):
            violations.append("changed_outside_allowed_fields")
        if any(feature in protected for feature in actual_changed_fields):
            violations.append("changed_protected_fields")

        dataset_df = context.bundle.dataset_df
        for feature in feature_order:
            if feature not in candidate_profile:
                continue
            raw_value = candidate_profile[feature]
            feature_type = context.policy.feature_type_map[feature]
            if feature_type == "binary":
                if raw_value not in {0, 1}:
                    violations.append(f"invalid_binary_value:{feature}")
                    continue
            elif feature_type == "int":
                if not isinstance(raw_value, int):
                    violations.append(f"invalid_int_value:{feature}")
                    continue
            elif feature_type == "float":
                if not isinstance(raw_value, (int, float)):
                    violations.append(f"invalid_float_value:{feature}")
                    continue
            feature_series = dataset_df[feature]
            min_value = float(feature_series.min())
            max_value = float(feature_series.max())
            numeric_value = float(raw_value)
            if numeric_value < min_value or numeric_value > max_value:
                violations.append(f"out_of_bounds:{feature}")

        candidate_frame = pd.DataFrame(
            [{feature: candidate_profile[feature] for feature in feature_order}],
            columns=feature_order,
        )
        prediction = self.prediction_service.predict(context.dataset_name, candidate_frame, context)
        if prediction.predicted_label != context.policy.desired_outcome:
            violations.append("candidate_does_not_flip")

        if violations:
            return self._failed(
                violations,
                details={
                    "candidate_method": candidate.method,
                    "candidate_rank": candidate.rank,
                    "candidate_changed_fields": list(candidate.changed_features),
                },
            )

        return InvariantValidationResult(
            status="passed",
            public_safe=True,
            reason_codes=[],
            validated_summary_type="counterfactual_found",
            validated_changed_fields=actual_changed_fields,
            details={
                "candidate_method": candidate.method,
                "candidate_rank": candidate.rank,
            },
        )

    def _failed(
        self,
        violations: list[str],
        *,
        details: dict[str, Any] | None = None,
    ) -> InvariantValidationResult:
        payload = dict(details or {})
        payload["violations"] = list(violations)
        return InvariantValidationResult(
            status="failed",
            public_safe=False,
            reason_codes=[INVALID_COUNTERFACTUAL_BLOCKED],
            validated_summary_type="runtime_reject",
            validated_changed_fields=[],
            details=payload,
        )
