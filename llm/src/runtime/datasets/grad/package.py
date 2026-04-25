from __future__ import annotations

from pathlib import Path
from typing import Any

from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from llm.src.runtime.contracts import CanonicalCandidate, CanonicalProfile, CanonicalRecourseRequest, VerificationResult
from llm.src.runtime.datasets.base import DatasetCompatibilityManifest, DatasetPackage, DatasetValidationResult
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.profile_service import ProfileService
from llm.src.runtime.reason_codes import NON_ACTIONABLE_DELTA, OUT_OF_DOMAIN_RANGE, PROTECTED_FIELD_CHANGED
from llm.src.runtime.types import DatasetPolicy, RuntimeContext
from llm_eval.models import BenchmarkDefinition, OutputContract, TargetField
from ufce import UFCE
from ufce.core.data_processing import get_grad_user_constraints


GRAD_SCHEMA_VERSION = "grad_schema_v1"
GRAD_POLICY_VERSION = "grad_policy_v1"
GRAD_STEP = {
    "GRE Score": 1,
    "TOEFL Score": 1,
    "University Rating": 1,
    "SOP": 1,
    "LOR": 1,
    "CGPA": 0.1,
    "Research": 1,
}
GRAD_STEP_PROVENANCE = (
    "Copied from legacy UFCE grad source assumptions and behavior in ufce/core/data_processing.py; "
    "not derived from user input and not learned online."
)
GRAD_ALIASES = {
    "GRE Score": ["gre", "gre score"],
    "TOEFL Score": ["toefl", "toefl score"],
    "University Rating": ["university rating", "rating"],
    "SOP": ["sop", "statement of purpose"],
    "LOR": ["lor", "letter of recommendation"],
    "CGPA": ["cgpa", "gpa"],
    "Research": ["research", "research experience"],
}
GRAD_FEATURE_TYPES = {
    "GRE Score": "int",
    "TOEFL Score": "int",
    "University Rating": "int",
    "SOP": "float",
    "LOR": "float",
    "CGPA": "float",
    "Research": "binary",
}
ROOT = Path(__file__).resolve().parents[5]
GOLDEN_CORPUS_PATH = ROOT / "docs" / "validation" / "corpora" / "part2_grad_golden_parity_v1.json"


class GradDatasetPackage(DatasetPackage):
    dataset_id = "grad"

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()
        self.profile_service = ProfileService()
        self.bundle = self.model_registry.get_bundle(self.dataset_id)
        self._policy = self._build_policy()
        self._context = RuntimeContext(
            dataset_name=self.dataset_id,
            bundle=self.bundle,
            policy=self._policy,
            mi_feature_pairs=self._build_mi_feature_pairs(),
        )

    def compatibility_manifest(self) -> DatasetCompatibilityManifest:
        entry = self.model_registry.get_manifest_entry(self.dataset_id)
        return DatasetCompatibilityManifest(
            dataset_id=self.dataset_id,
            schema_version=str(entry["schema_version"]),
            policy_version=str(entry["policy_version"]),
            supported_backends=[str(item) for item in entry["supported_backends"]],
            model_bundle_version=str(entry["model_bundle_version"]),
            live_runtime_enabled=bool(entry.get("live_runtime_enabled", False)),
        )

    def feature_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        for feature_name in self.bundle.feature_order:
            series = self.bundle.dataset_df[feature_name]
            schema[feature_name] = {
                "type": GRAD_FEATURE_TYPES[feature_name],
                "aliases": list(GRAD_ALIASES.get(feature_name, [])),
                "min": float(series.min()),
                "max": float(series.max()),
                "changeable": feature_name in self._policy.f2change,
                "protected": feature_name in self._policy.protected_features,
            }
        return schema

    def profile_schema(self) -> dict[str, Any]:
        return {
            "required_fields": list(self.bundle.feature_order),
            "field_order": list(self.bundle.feature_order),
            "dataset_id": self.dataset_id,
        }

    def policy(self) -> dict[str, Any]:
        payload = self._policy.to_dict()
        payload["ranking_weights"] = {"rank_hint": 1.0, "sparsity": 0.1}
        payload["clarification_semantics"] = "full_profile_required"
        payload["infeasibility_wording"] = "No valid graduate-admission recourse was found under the current limits."
        return payload

    def aliases(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in GRAD_ALIASES.items()}

    def normalize_profile(self, raw_profile: dict[str, Any]) -> CanonicalProfile:
        raw_request = {"dataset": self.dataset_id, "profile": dict(raw_profile)}
        parsed = self.profile_service.parse_request(
            raw_request,
            self.dataset_id,
            feature_order=list(self.bundle.feature_order),
        )
        canonical_df = self.profile_service.canonicalize(parsed, self._context, raw_request)
        return CanonicalProfile(
            dataset_id=self.dataset_id,
            values=canonical_df.iloc[0].to_dict(),
        )

    def validate_profile(
        self,
        profile: CanonicalProfile,
        hard_constraints: dict[str, Any] | None = None,
    ) -> DatasetValidationResult:
        missing_fields = [field for field in self.bundle.feature_order if field not in profile.values]
        if missing_fields:
            return DatasetValidationResult(ok=False, missing_fields=missing_fields)
        try:
            self.normalize_profile(profile.values)
        except Exception as exc:
            return DatasetValidationResult(ok=False, errors=[str(exc)])
        if hard_constraints:
            _normalized, errors = validate_and_normalize_constraint_spec(
                hard_constraints,
                feature_order=list(self.bundle.feature_order),
                numeric_bound_fields=self.numeric_bound_fields(),
            )
            if errors:
                return DatasetValidationResult(ok=False, errors=errors)
        return DatasetValidationResult(ok=True)

    def explanation_templates(self) -> dict[str, str]:
        return {
            "summary_no_recourse": "The graduate admission profile already meets the desired outcome.",
            "summary_counterfactual": "The graduate admission profile can be improved with validated changes.",
            "summary_reject": "The graduate admission profile could not produce a valid counterfactual under current constraints.",
        }

    def load_model_bundle(self):
        return self.bundle

    def runtime_context(self) -> RuntimeContext:
        return self._context

    def legality_check(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
    ) -> VerificationResult:
        del request
        violations: list[str] = []
        for feature_name, descriptor in self.feature_schema().items():
            if feature_name not in candidate.new_values:
                continue
            value = candidate.new_values[feature_name]
            if descriptor["type"] == "binary" and value not in {0, 1}:
                violations.append(f"{OUT_OF_DOMAIN_RANGE}:{feature_name}")
                continue
            numeric_value = float(value)
            if numeric_value < float(descriptor["min"]) or numeric_value > float(descriptor["max"]):
                violations.append(f"{OUT_OF_DOMAIN_RANGE}:{feature_name}")
        return VerificationResult(
            is_valid=not violations,
            reason_codes=violations,
            evidence={"dataset_id": self.dataset_id},
            candidate_id=candidate.candidate_id,
        )

    def actionability_check(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
    ) -> VerificationResult:
        violations: list[str] = []
        protected = set(self._policy.protected_features)
        changeable = set(self._policy.f2change)
        forbidden = set(request.forbidden_features)
        if any(feature in protected for feature in candidate.changed_features):
            violations.append(PROTECTED_FIELD_CHANGED)
        if any(feature not in changeable for feature in candidate.changed_features):
            violations.append(NON_ACTIONABLE_DELTA)
        if any(feature in forbidden for feature in candidate.changed_features):
            violations.append(NON_ACTIONABLE_DELTA)
        return VerificationResult(
            is_valid=not violations,
            reason_codes=violations,
            evidence={"dataset_id": self.dataset_id, "changed_features": list(candidate.changed_features)},
            candidate_id=candidate.candidate_id,
        )

    def golden_parity_corpus_path(self) -> Path:
        return GOLDEN_CORPUS_PATH

    def display_name(self) -> str:
        return "Graduate Admission"

    def primary_subject_label(self) -> str:
        return "graduate admission profile"

    def refinement_subject_label(self) -> str:
        return "graduate admission constraints"

    def numeric_bound_fields(self) -> list[str]:
        return [
            field_name
            for field_name in self.bundle.feature_order
            if GRAD_FEATURE_TYPES.get(field_name) in {"int", "float"}
        ]

    def primary_response_schema_name(self) -> str:
        return "ufce_grad_cf_parser_output_v1"

    def refinement_response_schema_name(self) -> str:
        return "ufce_grad_refinement_feedback_output_v1"

    def live_primary_benchmark(self) -> BenchmarkDefinition:
        feature_schema = self.feature_schema()
        return BenchmarkDefinition(
            benchmark_name="ufce_grad_cf_parser_live_v1",
            description="Live primary parser contract for the graduate-admission dataset.",
            target_cf_fields=tuple(
                TargetField(
                    name=field_name,
                    type=str(feature_schema[field_name]["type"]),
                    description=f"Canonical graduate admission field {field_name}",
                )
                for field_name in self.bundle.feature_order
            ),
            output_contract=OutputContract(
                task="extract_cf_request",
                status_enum=("complete", "partial", "needs_clarification", "conflict"),
                rules=(
                    "Return only fields explicitly inferable from the input.",
                    "Do not invent missing values.",
                    "Use canonical field names exactly as defined.",
                ),
            ),
            cases=(),
        )

    def _build_policy(self) -> DatasetPolicy:
        (
            features,
            categorical_features,
            numeric_features,
            uf,
            f2change,
            label_col,
            desired_outcome,
            _nbr_features,
            protected_features,
            _data_lab0,
            _data_lab1,
        ) = get_grad_user_constraints(self.bundle.dataset_df.copy())
        return DatasetPolicy(
            dataset_name=self.dataset_id,
            policy_version=GRAD_POLICY_VERSION,
            label_col=str(label_col),
            feature_type_map=dict(GRAD_FEATURE_TYPES),
            numeric_features=[str(feature) for feature in numeric_features],
            categorical_features=[str(feature) for feature in categorical_features],
            f2change=[str(feature) for feature in f2change],
            conversation_aliases={key: list(values) for key, values in GRAD_ALIASES.items()},
            uf=dict(uf),
            step=dict(GRAD_STEP),
            desired_outcome=int(desired_outcome),
            protected_features=[str(feature) for feature in protected_features],
            runtime_enabled=True,
            expected_feature_order=[str(feature) for feature in features],
            step_provenance=GRAD_STEP_PROVENANCE,
        )

    def _build_mi_feature_pairs(self) -> list[list[str]]:
        ufc = UFCE()
        pairs = ufc.get_top_MI_features(
            self.bundle.X.loc[:, self.bundle.feature_order].copy(),
            list(self.bundle.feature_order),
        )
        return [list(item) for item in pairs[:5]]
