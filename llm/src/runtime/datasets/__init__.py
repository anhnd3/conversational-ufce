from __future__ import annotations

import importlib

__all__ = [
    "BankDatasetPackage",
    "DatasetCompatibilityManifest",
    "DatasetPackage",
    "DatasetValidationResult",
    "GradDatasetPackage",
]


def __getattr__(name: str):
    if name in {"DatasetCompatibilityManifest", "DatasetPackage", "DatasetValidationResult"}:
        base = importlib.import_module("llm.src.runtime.datasets.base")

        return getattr(base, name)
    if name == "BankDatasetPackage":
        from llm.src.runtime.datasets.bank.package import BankDatasetPackage

        return BankDatasetPackage
    if name == "GradDatasetPackage":
        from llm.src.runtime.datasets.grad.package import GradDatasetPackage

        return GradDatasetPackage
    raise AttributeError(name)
