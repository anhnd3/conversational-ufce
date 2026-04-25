from __future__ import annotations

from dataclasses import replace

from llm.src.runtime.datasets import BankDatasetPackage, GradDatasetPackage
from llm.src.runtime.datasets.bank.metadata import (
    BANK_FROZEN_MI_FEATURE_PAIRS,
    BANK_POLICY_VERSION,
    BANK_STEP,
    BANK_STEP_PROVENANCE,
)
from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.reason_codes import POLICY_NOT_FOUND
from llm.src.runtime.registries.dataset_registry import DatasetRegistry
from llm.src.runtime.types import DatasetPolicy, RuntimeContext


class PolicyRegistry:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry
        self.dataset_registry = DatasetRegistry(
            {
                "bank": BankDatasetPackage(self.model_registry),
                "grad": GradDatasetPackage(self.model_registry),
            }
        )
        self._policies: dict[str, DatasetPolicy] = {}
        self._runtime_contexts: dict[str, RuntimeContext] = {}
        self._register_policies()

    def has_enabled_policy(self, dataset_name: str) -> bool:
        return dataset_name in self._runtime_contexts

    def get_policy(self, dataset_name: str) -> DatasetPolicy:
        if dataset_name not in self._policies or not self._policies[dataset_name].runtime_enabled:
            raise RuntimeServiceError(
                (POLICY_NOT_FOUND,),
                "Runtime policy not found for dataset '{0}'.".format(dataset_name),
            )
        return self._policies[dataset_name]

    def get_runtime_context(self, dataset_name: str) -> RuntimeContext:
        if dataset_name not in self._runtime_contexts:
            raise RuntimeServiceError(
                (POLICY_NOT_FOUND,),
                "Runtime context not found for dataset '{0}'.".format(dataset_name),
            )
        return self._runtime_contexts[dataset_name]

    def _register_policies(self) -> None:
        for dataset_name in self.dataset_registry.keys():
            package = self.dataset_registry.get(dataset_name)
            manifest = package.compatibility_manifest()
            runtime_enabled = bool(package.policy().get("runtime_enabled", False)) and bool(manifest.live_runtime_enabled)
            policy = replace(package.runtime_context().policy, runtime_enabled=runtime_enabled)
            self._policies[dataset_name] = policy
            if runtime_enabled:
                self._runtime_contexts[dataset_name] = replace(
                    package.runtime_context(),
                    policy=policy,
                )
