from __future__ import annotations

from typing import Any

from llm.src.runtime.contracts.canonical_candidate import CanonicalCandidate
from llm.src.runtime.contracts.canonical_profile import CanonicalProfile
from llm.src.runtime.contracts.canonical_request import CanonicalRecourseRequest
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult, RuntimeRequest


def build_delta_summary(
    *,
    original_values: dict[str, Any],
    new_values: dict[str, Any],
    changed_features: list[str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for feature_name in changed_features:
        summary.append(
            {
                "feature": feature_name,
                "old_value": original_values.get(feature_name),
                "new_value": new_values.get(feature_name),
            }
        )
    return summary


def canonical_request_from_legacy_request(
    *,
    legacy_request: RuntimeRequest,
    desired_outcome: int | str,
    canonical_profile: CanonicalProfile,
) -> CanonicalRecourseRequest:
    raw_constraints = dict(legacy_request.constraint_spec or {})
    forbidden = []
    for key in ("immutable", "disallowed_changes"):
        value = raw_constraints.get(key)
        if isinstance(value, list):
            forbidden.extend(item for item in value if isinstance(item, str))
    soft_preferences: dict[str, Any] = {}
    if "prefer_fewer_changes" in raw_constraints:
        soft_preferences["prefer_fewer_changes"] = bool(raw_constraints["prefer_fewer_changes"])
    return CanonicalRecourseRequest(
        dataset_id=legacy_request.dataset,
        desired_outcome=desired_outcome,
        profile=canonical_profile,
        hard_constraints=raw_constraints,
        soft_preferences=soft_preferences,
        forbidden_features=_dedupe_strings(forbidden),
        max_changes=raw_constraints.get("max_changed_features")
        if isinstance(raw_constraints.get("max_changed_features"), int)
        else None,
        session_context={
            "constraint_spec_present": bool(raw_constraints),
            "constraint_spec_version": "constraint_spec_v1" if raw_constraints else None,
        },
    )


def legacy_candidate_to_canonical_candidate(
    *,
    backend_id: str,
    candidate: CounterfactualCandidate,
    factual_profile: dict[str, Any],
) -> CanonicalCandidate:
    candidate_id = "{0}:{1}:{2}".format(backend_id, candidate.method, candidate.rank)
    return CanonicalCandidate(
        backend_id=backend_id,
        candidate_id=candidate_id,
        changed_features=list(candidate.changed_features),
        original_values=dict(factual_profile),
        new_values=dict(candidate.profile),
        delta_summary=build_delta_summary(
            original_values=factual_profile,
            new_values=candidate.profile,
            changed_features=list(candidate.changed_features),
        ),
        predicted_outcome=None,
        raw_backend_score=None,
        backend_metadata={
            "legacy_method": candidate.method,
            "legacy_rank": int(candidate.rank),
        },
    )


def canonical_candidates_to_legacy_result(
    *,
    candidates: list[CanonicalCandidate],
    failure_reason_codes: list[str] | None = None,
) -> CounterfactualResult:
    if not candidates:
        return CounterfactualResult(
            feasible=False,
            candidates=[],
            reason_codes=list(failure_reason_codes or []),
        )
    legacy_candidates = [
        CounterfactualCandidate(
            method=str(candidate.backend_metadata.get("legacy_method", candidate.backend_id)),
            rank=int(candidate.backend_metadata.get("legacy_rank", index + 1)),
            profile=dict(candidate.new_values),
            changed_features=list(candidate.changed_features),
        )
        for index, candidate in enumerate(candidates)
    ]
    return CounterfactualResult(
        feasible=True,
        candidates=legacy_candidates,
        reason_codes=[],
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if not isinstance(item, str) or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
