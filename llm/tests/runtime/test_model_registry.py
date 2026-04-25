from __future__ import annotations

import json

import pytest

from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.reason_codes import BUNDLE_LOAD_FAILED, POLICY_NOT_FOUND


def test_model_registry_loads_manifest_and_bank_bundle():
    registry = ModelRegistry()

    assert registry.has_dataset("bank") is True
    assert registry.get_bundle("bank").dataset_name == "bank"
    assert registry.get_manifest_entry("bank")["source_data_hash"] == registry.get_bundle("bank").source_data_hash


def test_model_registry_rejects_duplicate_dataset_entries(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset_name": "bank",
                        "artifact_version": "v1",
                        "source_data_hash": "x",
                        "has_scaler": False,
                        "has_movie_distance_scaler": False,
                    },
                    {
                        "dataset_name": "bank",
                        "artifact_version": "v1",
                        "source_data_hash": "x",
                        "has_scaler": False,
                        "has_movie_distance_scaler": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeServiceError) as exc_info:
        ModelRegistry(manifest_path=manifest_path)

    assert exc_info.value.reason_codes == (BUNDLE_LOAD_FAILED,)


def test_policy_registry_creates_bank_runtime_context_once():
    registry = ModelRegistry()
    policy_registry = PolicyRegistry(registry)

    first = policy_registry.get_runtime_context("bank")
    second = policy_registry.get_runtime_context("bank")

    assert first is second


def test_policy_registry_rejects_known_but_disabled_dataset():
    registry = ModelRegistry()
    policy_registry = PolicyRegistry(registry)

    with pytest.raises(RuntimeServiceError) as exc_info:
        policy_registry.get_runtime_context("movie")

    assert exc_info.value.reason_codes == (POLICY_NOT_FOUND,)
