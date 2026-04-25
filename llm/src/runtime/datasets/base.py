from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from llm.src.runtime.contracts import CanonicalCandidate, CanonicalProfile, CanonicalRecourseRequest, VerificationResult
from llm.src.runtime.types import RuntimeContext
from llm_eval.models import BenchmarkDefinition


@dataclass(frozen=True)
class DatasetValidationResult:
    ok: bool
    missing_fields: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "missing_fields": list(self.missing_fields),
            "conflicts": list(self.conflicts),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class DatasetCompatibilityManifest:
    dataset_id: str
    schema_version: str
    policy_version: str
    supported_backends: list[str]
    model_bundle_version: str
    live_runtime_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "supported_backends": list(self.supported_backends),
            "model_bundle_version": self.model_bundle_version,
            "live_runtime_enabled": bool(self.live_runtime_enabled),
        }


class DatasetPackage(Protocol):
    dataset_id: str

    def compatibility_manifest(self) -> DatasetCompatibilityManifest:
        ...

    def feature_schema(self) -> dict[str, Any]:
        ...

    def profile_schema(self) -> dict[str, Any]:
        ...

    def policy(self) -> dict[str, Any]:
        ...

    def aliases(self) -> dict[str, list[str]]:
        ...

    def normalize_profile(self, raw_profile: dict[str, Any]) -> CanonicalProfile:
        ...

    def validate_profile(
        self,
        profile: CanonicalProfile,
        hard_constraints: dict[str, Any] | None = None,
    ) -> DatasetValidationResult:
        ...

    def explanation_templates(self) -> dict[str, str]:
        ...

    def load_model_bundle(self) -> Any:
        ...

    def runtime_context(self) -> RuntimeContext:
        ...

    def legality_check(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
    ) -> VerificationResult:
        ...

    def actionability_check(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
    ) -> VerificationResult:
        ...

    def golden_parity_corpus_path(self) -> Path:
        ...

    def display_name(self) -> str:
        ...

    def primary_subject_label(self) -> str:
        ...

    def refinement_subject_label(self) -> str:
        ...

    def numeric_bound_fields(self) -> list[str]:
        ...

    def primary_response_schema_name(self) -> str:
        ...

    def refinement_response_schema_name(self) -> str:
        ...

    def live_primary_benchmark(self) -> BenchmarkDefinition:
        ...
