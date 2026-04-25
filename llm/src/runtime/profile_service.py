from __future__ import annotations

from typing import Any

import pandas as pd

from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.reason_codes import (
    INVALID_FIELD_TYPE,
    MISSING_REQUIRED_FEATURES,
    UNKNOWN_PROFILE_FIELDS,
)
from llm.src.runtime.types import RuntimeContext, RuntimeRequest


ALLOWED_TOP_LEVEL_KEYS = frozenset({"dataset", "profile", "constraint_spec"})


class ProfileService:
    def parse_request(
        self,
        request: dict[str, Any],
        dataset_name: str,
        *,
        feature_order: list[str] | None = None,
    ) -> RuntimeRequest:
        if not isinstance(request, dict):
            raise RuntimeServiceError(
                (MISSING_REQUIRED_FEATURES,),
                "Runtime request must be an object.",
            )
        profile = request.get("profile")
        if not isinstance(profile, dict):
            raise RuntimeServiceError(
                (MISSING_REQUIRED_FEATURES,),
                "Runtime request must include a profile object.",
            )
        constraint_spec = request.get("constraint_spec")
        if feature_order is not None:
            normalized_constraint_spec, errors = validate_and_normalize_constraint_spec(
                constraint_spec,
                feature_order=feature_order,
            )
            if errors:
                raise RuntimeServiceError(
                    (INVALID_FIELD_TYPE,),
                    "; ".join(errors),
                )
            constraint_spec = normalized_constraint_spec
        elif constraint_spec is not None and not isinstance(constraint_spec, dict):
            raise RuntimeServiceError(
                (INVALID_FIELD_TYPE,),
                "constraint_spec must be an object.",
            )
        return RuntimeRequest(
            dataset=dataset_name,
            profile=dict(profile),
            constraint_spec=None if constraint_spec is None else dict(constraint_spec),
        )

    def canonicalize(self, request: RuntimeRequest, context: RuntimeContext, raw_request: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(raw_request, dict):
            raise RuntimeServiceError(
                (MISSING_REQUIRED_FEATURES,),
                "Runtime request must be an object.",
            )

        unknown_top_level = sorted(key for key in raw_request if key not in ALLOWED_TOP_LEVEL_KEYS)
        if unknown_top_level:
            # This slice intentionally collapses unknown top-level keys and unknown profile keys
            # into the same deterministic reason code for a smaller public error taxonomy.
            raise RuntimeServiceError(
                (UNKNOWN_PROFILE_FIELDS,),
                "Unknown top-level request fields: {0}".format(", ".join(unknown_top_level)),
            )

        profile = request.profile
        unknown_profile_fields = sorted(
            key for key in profile if key not in context.bundle.feature_order
        )
        if unknown_profile_fields:
            raise RuntimeServiceError(
                (UNKNOWN_PROFILE_FIELDS,),
                "Unknown profile fields: {0}".format(", ".join(unknown_profile_fields)),
            )

        missing_fields = [
            feature for feature in context.bundle.feature_order if feature not in profile
        ]
        if missing_fields:
            raise RuntimeServiceError(
                (MISSING_REQUIRED_FEATURES,),
                "Missing required profile fields: {0}".format(", ".join(missing_fields)),
            )

        canonical_profile: dict[str, Any] = {}
        for feature_name in context.bundle.feature_order:
            canonical_profile[feature_name] = self._coerce_value(
                feature_name,
                profile[feature_name],
                context.policy.feature_type_map[feature_name],
            )

        return pd.DataFrame([canonical_profile], columns=context.bundle.feature_order)

    def _coerce_value(self, feature_name: str, value: Any, expected_type: str) -> Any:
        if isinstance(value, bool):
            raise RuntimeServiceError(
                (INVALID_FIELD_TYPE,),
                "{0} cannot be a bool.".format(feature_name),
            )
        if expected_type == "float":
            if not isinstance(value, (int, float)):
                raise RuntimeServiceError(
                    (INVALID_FIELD_TYPE,),
                    "{0} must be numeric.".format(feature_name),
                )
            return float(value)
        if expected_type == "int":
            if not isinstance(value, int):
                raise RuntimeServiceError(
                    (INVALID_FIELD_TYPE,),
                    "{0} must be an integer.".format(feature_name),
                )
            return int(value)
        if expected_type == "binary":
            if not isinstance(value, int) or value not in (0, 1):
                raise RuntimeServiceError(
                    (INVALID_FIELD_TYPE,),
                    "{0} must be binary 0 or 1.".format(feature_name),
                )
            return int(value)
        raise RuntimeServiceError(
            (INVALID_FIELD_TYPE,),
            "Unsupported type contract for {0}: {1}".format(feature_name, expected_type),
        )
