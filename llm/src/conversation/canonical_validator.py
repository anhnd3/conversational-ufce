from __future__ import annotations

from typing import Any

from llm.src.conversation.types import CanonicalValidationResult, ConversationStage
from llm.src.runtime.datasets import BankDatasetPackage, GradDatasetPackage
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from llm.src.runtime.profile_service import ProfileService
from llm.src.runtime.registries.dataset_registry import DatasetRegistry
from llm.src.runtime.types import RuntimeContext
from llm.src.validation.schema_validator import ValidationResult


class PackageCanonicalValidator:
    def __init__(
        self,
        *,
        dataset_package: DatasetPackage,
        model_registry: ModelRegistry | None = None,
        policy_registry: PolicyRegistry | None = None,
        profile_service: ProfileService | None = None,
    ) -> None:
        self.dataset_package = dataset_package
        self.model_registry = model_registry or ModelRegistry()
        self.policy_registry = policy_registry or PolicyRegistry(self.model_registry)
        self.profile_service = profile_service or ProfileService()
        self.context: RuntimeContext = self.dataset_package.runtime_context()
        self.required_fields = tuple(self.dataset_package.profile_schema()["field_order"])
        self.numeric_bound_fields = tuple(self.dataset_package.numeric_bound_fields())

    def validate(
        self,
        *,
        candidate: dict[str, Any] | None,
        schema_validation: ValidationResult,
        dataset_id: str | None = None,
    ) -> CanonicalValidationResult:
        if not schema_validation.is_valid or not isinstance(candidate, dict):
            return CanonicalValidationResult(
                parser_status=None if not isinstance(candidate, dict) else candidate.get("status"),
                final_stage=ConversationStage.PARSER_FAILURE,
                is_usable=False,
                ready_for_runtime=False,
                missing_runtime_fields=[],
                confirmed_conflicts=[],
                provided_fields=[],
                errors=list(schema_validation.errors) or ["Candidate is unavailable for canonical validation."],
                runtime_request=None,
            )

        status = candidate.get("status")
        cf_request = candidate.get("cf_request")
        raw_constraint_spec = candidate.get("constraint_spec")
        conflicts = candidate.get("conflicts")
        if not isinstance(cf_request, dict):
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.PARSER_FAILURE,
                is_usable=False,
                ready_for_runtime=False,
                missing_runtime_fields=[],
                confirmed_conflicts=[],
                provided_fields=[],
                errors=["cf_request must remain an object after schema validation."],
                runtime_request=None,
            )

        provided_fields = [field for field in self.required_fields if field in cf_request]
        missing_runtime_fields = [field for field in self.required_fields if field not in cf_request]
        confirmed_conflicts = normalize_conflicts(conflicts)
        errors: list[str] = []

        if confirmed_conflicts:
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.CONFLICT,
                is_usable=True,
                ready_for_runtime=False,
                missing_runtime_fields=[],
                confirmed_conflicts=confirmed_conflicts,
                provided_fields=provided_fields,
                errors=[],
                runtime_request=None,
            )

        if status == "conflict":
            errors.append("status 'conflict' requires at least one confirmed conflict entry.")
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.PARSER_FAILURE,
                is_usable=False,
                ready_for_runtime=False,
                missing_runtime_fields=missing_runtime_fields,
                confirmed_conflicts=[],
                provided_fields=provided_fields,
                errors=errors,
                runtime_request=None,
            )

        if status in ("partial", "needs_clarification"):
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.NEEDS_CLARIFICATION,
                is_usable=True,
                ready_for_runtime=False,
                missing_runtime_fields=missing_runtime_fields,
                confirmed_conflicts=[],
                provided_fields=provided_fields,
                errors=[],
                runtime_request=None,
            )

        if status != "complete":
            errors.append("Unrecognized canonical parser status.")
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.PARSER_FAILURE,
                is_usable=False,
                ready_for_runtime=False,
                missing_runtime_fields=missing_runtime_fields,
                confirmed_conflicts=[],
                provided_fields=provided_fields,
                errors=errors,
                runtime_request=None,
            )

        if missing_runtime_fields:
            errors.append(
                "status 'complete' requires all runtime-required "
                f"{self.dataset_package.dataset_id} fields."
            )
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.NEEDS_CLARIFICATION,
                is_usable=True,
                ready_for_runtime=False,
                missing_runtime_fields=missing_runtime_fields,
                confirmed_conflicts=[],
                provided_fields=provided_fields,
                errors=errors,
                runtime_request=None,
            )

        normalized_constraint_spec, constraint_errors = validate_and_normalize_constraint_spec(
            raw_constraint_spec,
            feature_order=list(self.required_fields),
            numeric_bound_fields=list(self.numeric_bound_fields),
        )
        if constraint_errors:
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.PARSER_FAILURE,
                is_usable=False,
                ready_for_runtime=False,
                missing_runtime_fields=missing_runtime_fields,
                confirmed_conflicts=[],
                provided_fields=provided_fields,
                errors=constraint_errors,
                runtime_request=None,
            )

        runtime_request = {"dataset": self.dataset_package.dataset_id, "profile": dict(cf_request)}
        if normalized_constraint_spec is not None:
            runtime_request["constraint_spec"] = dict(normalized_constraint_spec)
        try:
            parsed_request = self.profile_service.parse_request(
                runtime_request,
                self.dataset_package.dataset_id,
                feature_order=list(self.required_fields),
            )
            canonical_profile = self.profile_service.canonicalize(parsed_request, self.context, runtime_request)
        except Exception as exc:
            return CanonicalValidationResult(
                parser_status=status,
                final_stage=ConversationStage.PARSER_FAILURE,
                is_usable=False,
                ready_for_runtime=False,
                missing_runtime_fields=missing_runtime_fields,
                confirmed_conflicts=[],
                provided_fields=provided_fields,
                errors=[str(exc)],
                runtime_request=None,
            )

        serialized_profile = serialize_profile_row(
            canonical_profile.iloc[0].to_dict(),
            self.context.policy.feature_type_map,
            self.required_fields,
        )
        return CanonicalValidationResult(
            parser_status=status,
            final_stage=ConversationStage.READY_FOR_RUNTIME,
            is_usable=True,
            ready_for_runtime=True,
            missing_runtime_fields=[],
            confirmed_conflicts=[],
            provided_fields=provided_fields,
            errors=[],
            runtime_request=build_runtime_request_payload(parsed_request.to_dict(), serialized_profile),
        )


class BankCanonicalValidator(PackageCanonicalValidator):
    def __init__(
        self,
        *,
        model_registry: ModelRegistry | None = None,
        policy_registry: PolicyRegistry | None = None,
        profile_service: ProfileService | None = None,
    ) -> None:
        active_model_registry = model_registry or ModelRegistry()
        super().__init__(
            dataset_package=BankDatasetPackage(active_model_registry),
            model_registry=active_model_registry,
            policy_registry=policy_registry,
            profile_service=profile_service,
        )


class DatasetRoutedCanonicalValidator:
    def __init__(
        self,
        *,
        default_dataset: str = "bank",
        model_registry: ModelRegistry | None = None,
        policy_registry: PolicyRegistry | None = None,
        profile_service: ProfileService | None = None,
        dataset_registry: DatasetRegistry | None = None,
    ) -> None:
        self.model_registry = model_registry or ModelRegistry()
        self.policy_registry = policy_registry or PolicyRegistry(self.model_registry)
        self.profile_service = profile_service or ProfileService()
        self.dataset_registry = dataset_registry or DatasetRegistry(
            {
                "bank": BankDatasetPackage(self.model_registry),
                "grad": GradDatasetPackage(self.model_registry),
            }
        )
        self.validators: dict[str, Any] = {
            dataset_id: PackageCanonicalValidator(
                dataset_package=self.dataset_registry.get(dataset_id),
                model_registry=self.model_registry,
                policy_registry=self.policy_registry,
                profile_service=self.profile_service,
            )
            for dataset_id in self.dataset_registry.keys()
        }
        self.default_dataset = str(default_dataset).strip().lower() or "bank"
        self.dataset_id = self.default_dataset
        active = self.validators[self.default_dataset]
        self.context = active.context
        self.required_fields = active.required_fields
        self.numeric_bound_fields = active.numeric_bound_fields

    def validate(
        self,
        *,
        candidate: dict[str, Any] | None,
        schema_validation: ValidationResult,
        dataset_id: str | None = None,
    ) -> CanonicalValidationResult:
        active_dataset = str(dataset_id or self.default_dataset).strip().lower() or self.default_dataset
        validator = self.validators.get(active_dataset)
        if validator is None:
            validator = self.validators[self.default_dataset]
            active_dataset = self.default_dataset
        self.dataset_id = active_dataset
        self.context = validator.context
        self.required_fields = validator.required_fields
        self.numeric_bound_fields = validator.numeric_bound_fields
        return validator.validate(
            candidate=candidate,
            schema_validation=schema_validation,
        )


def normalize_conflicts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        clean = " ".join(item.split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def serialize_profile_row(
    row: dict[str, Any],
    feature_type_map: dict[str, str],
    ordered_fields: tuple[str, ...],
) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for field_name in ordered_fields:
        value = row[field_name]
        field_type = feature_type_map[field_name]
        if field_type == "float":
            profile[field_name] = float(value)
        else:
            profile[field_name] = int(value)
    return profile


def build_runtime_request_payload(payload: dict[str, Any], serialized_profile: dict[str, Any]) -> dict[str, Any]:
    runtime_request = dict(payload)
    runtime_request["profile"] = dict(serialized_profile)
    return runtime_request
