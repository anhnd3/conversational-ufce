from __future__ import annotations

__all__ = ["GradDatasetPackage"]


def __getattr__(name: str):
    if name == "GradDatasetPackage":
        from llm.src.runtime.datasets.grad.package import GradDatasetPackage

        return GradDatasetPackage
    raise AttributeError(name)
