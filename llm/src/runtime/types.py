from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ufce.model_bundles import DatasetModelBundle


@dataclass(frozen=True)
class RuntimeRequest:
    dataset: str
    profile: dict[str, Any]
    constraint_spec: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "profile": dict(self.profile),
        }
        if self.constraint_spec is not None:
            payload["constraint_spec"] = dict(self.constraint_spec)
        return payload


@dataclass(frozen=True)
class PredictionResult:
    dataset: str
    predicted_label: int
    predicted_proba: float
    feature_order_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "predicted_label": int(self.predicted_label),
            "predicted_proba": float(self.predicted_proba),
            "feature_order_used": list(self.feature_order_used),
        }


@dataclass(frozen=True)
class CounterfactualCandidate:
    method: str
    rank: int
    profile: dict[str, Any]
    changed_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "rank": int(self.rank),
            "profile": dict(self.profile),
            "changed_features": list(self.changed_features),
        }


@dataclass(frozen=True)
class CounterfactualResult:
    feasible: bool
    candidates: list[CounterfactualCandidate]
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class RuntimeDebugTrace:
    runtime_mode: str | None = None
    backend_name: str | None = None
    deterministic_seed: int | None = None
    policy_version: str | None = None
    mi_feature_pairs: list[list[str]] = field(default_factory=list)
    state_trace: list[str] = field(default_factory=list)
    service_errors: list[dict[str, str]] = field(default_factory=list)
    ufce_methods: list[dict[str, Any]] = field(default_factory=list)
    winning_path: dict[str, Any] | None = None
    reject_path: dict[str, Any] | None = None
    constraint_filter: dict[str, Any] | None = None
    generation_stats: dict[str, Any] | None = None

    def add_service_error(self, service: str, error: str) -> None:
        self.service_errors.append(
            {
                "service": str(service),
                "error": str(error),
            }
        )

    def add_ufce_method(self, method: str, status: str, candidate_count: int, error: str | None = None) -> None:
        record = {
            "method": str(method),
            "status": str(status),
            "candidate_count": int(candidate_count),
        }
        if error is not None:
            record["error"] = str(error)
        self.ufce_methods.append(record)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.runtime_mode,
            "backend_name": self.backend_name,
            "deterministic_seed": self.deterministic_seed,
            "policy_version": self.policy_version,
            "mi_feature_pairs": [list(pair) for pair in self.mi_feature_pairs],
            "state_trace": list(self.state_trace),
            "service_errors": [dict(item) for item in self.service_errors],
            "ufce_methods": [dict(item) for item in self.ufce_methods],
            "winning_path": None if self.winning_path is None else dict(self.winning_path),
            "reject_path": None if self.reject_path is None else dict(self.reject_path),
            "constraint_filter": None if self.constraint_filter is None else dict(self.constraint_filter),
            "generation_stats": None if self.generation_stats is None else dict(self.generation_stats),
        }


@dataclass(frozen=True)
class InvariantValidationResult:
    status: str
    public_safe: bool
    reason_codes: list[str]
    validated_summary_type: str | None
    validated_changed_fields: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_safe": bool(self.public_safe),
            "reason_codes": list(self.reason_codes),
            "validated_summary_type": self.validated_summary_type,
            "validated_changed_fields": list(self.validated_changed_fields),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RuntimeResult:
    dataset: str | None
    controller_state: str
    prediction: PredictionResult | None
    counterfactual: CounterfactualResult | None
    reason_codes: list[str]
    runtime_mode: str | None = None
    invariant_validation: InvariantValidationResult | None = None
    debug_trace: RuntimeDebugTrace | None = None
    canonical_request: dict[str, Any] | None = None
    canonical_candidates: list[dict[str, Any]] = field(default_factory=list)
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    backend_manifest: dict[str, Any] | None = None
    backend_id: str | None = None
    reason_code_version: str | None = None

    def to_dict(self, include_debug_trace: bool = False) -> dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "controller_state": self.controller_state,
            "prediction": None if self.prediction is None else self.prediction.to_dict(),
            "counterfactual": None if self.counterfactual is None else self.counterfactual.to_dict(),
            "reason_codes": list(self.reason_codes),
            "runtime_mode": self.runtime_mode,
            "invariant_validation": None
            if self.invariant_validation is None
            else self.invariant_validation.to_dict(),
            "canonical_request": None if self.canonical_request is None else dict(self.canonical_request),
            "canonical_candidates": [dict(candidate) for candidate in self.canonical_candidates],
            "verification_results": [dict(result) for result in self.verification_results],
            "backend_manifest": None if self.backend_manifest is None else dict(self.backend_manifest),
            "backend_id": self.backend_id,
            "reason_code_version": self.reason_code_version,
        }
        if include_debug_trace and self.debug_trace is not None:
            payload["debug_trace"] = self.debug_trace.to_dict()
        return payload


@dataclass(frozen=True)
class DatasetPolicy:
    dataset_name: str
    policy_version: str
    label_col: str
    feature_type_map: dict[str, str]
    numeric_features: list[str]
    categorical_features: list[str]
    f2change: list[str]
    conversation_aliases: dict[str, list[str]]
    uf: dict[str, Any]
    step: dict[str, Any]
    desired_outcome: int
    protected_features: list[str]
    runtime_enabled: bool
    expected_feature_order: list[str]
    step_provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "policy_version": self.policy_version,
            "label_col": self.label_col,
            "feature_type_map": dict(self.feature_type_map),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "f2change": list(self.f2change),
            "conversation_aliases": {key: list(values) for key, values in self.conversation_aliases.items()},
            "uf": dict(self.uf),
            "step": dict(self.step),
            "desired_outcome": int(self.desired_outcome),
            "protected_features": list(self.protected_features),
            "runtime_enabled": bool(self.runtime_enabled),
            "expected_feature_order": list(self.expected_feature_order),
            "step_provenance": self.step_provenance,
        }


@dataclass(frozen=True)
class RuntimeContext:
    dataset_name: str
    bundle: DatasetModelBundle
    policy: DatasetPolicy
    mi_feature_pairs: list[list[str]]


@dataclass(frozen=True)
class UFCERequest:
    dataset: str
    query_row: pd.DataFrame
    feature_matrix: pd.DataFrame
    positive_class_pool: pd.DataFrame
    bundle: DatasetModelBundle
    policy: DatasetPolicy
    mi_feature_pairs: list[list[str]]
