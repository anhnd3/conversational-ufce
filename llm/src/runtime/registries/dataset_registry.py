from __future__ import annotations

from llm.src.runtime.datasets.base import DatasetPackage


class DatasetRegistry:
    def __init__(self, packages: dict[str, DatasetPackage]) -> None:
        self._packages = {
            str(dataset_id).strip().lower(): package
            for dataset_id, package in packages.items()
        }

    def keys(self) -> list[str]:
        return sorted(self._packages)

    def has(self, dataset_id: str) -> bool:
        return str(dataset_id).strip().lower() in self._packages

    def get(self, dataset_id: str) -> DatasetPackage:
        normalized = str(dataset_id).strip().lower()
        if normalized not in self._packages:
            raise KeyError(f"Unsupported dataset: {dataset_id}")
        return self._packages[normalized]
