from __future__ import annotations

__all__ = ["BankDatasetPackage"]


def __getattr__(name: str):
    if name == "BankDatasetPackage":
        from llm.src.runtime.datasets.bank.package import BankDatasetPackage

        return BankDatasetPackage
    raise AttributeError(name)
