from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.reason_codes import BUNDLE_LOAD_FAILED
from ufce.model_bundles import DatasetModelBundle, load_dataset_model_bundle


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = ROOT / "llm" / "models"
DEFAULT_DATA_ROOT = ROOT / "ufce" / "data"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_ROOT / "manifest.json"


class ModelRegistry:
    def __init__(
        self,
        manifest_path: Path | None = None,
        artifact_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST_PATH)
        self.artifact_root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT)
        self.data_root = Path(data_root or DEFAULT_DATA_ROOT)
        self._manifest_entries: dict[str, dict[str, Any]] = {}
        self._bundles: dict[str, DatasetModelBundle] = {}
        self._load()

    def datasets(self) -> list[str]:
        return sorted(self._bundles)

    def has_dataset(self, dataset_name: str) -> bool:
        return dataset_name in self._bundles

    def get_bundle(self, dataset_name: str) -> DatasetModelBundle:
        return self._bundles[dataset_name]

    def get_manifest_entry(self, dataset_name: str) -> dict[str, Any]:
        return dict(self._manifest_entries[dataset_name])

    def _load(self) -> None:
        payload = self._load_manifest_payload()
        datasets = payload.get("datasets")
        if not isinstance(datasets, list):
            raise RuntimeServiceError((BUNDLE_LOAD_FAILED,), "Manifest 'datasets' must be a list.")

        # The manifest contract for this slice is exactly one entry per dataset name.
        seen: set[str] = set()
        for raw_entry in datasets:
            if not isinstance(raw_entry, dict):
                raise RuntimeServiceError((BUNDLE_LOAD_FAILED,), "Manifest dataset entries must be objects.")
            dataset_name = str(raw_entry.get("dataset_name", "")).strip().lower()
            if not dataset_name:
                raise RuntimeServiceError((BUNDLE_LOAD_FAILED,), "Manifest entry is missing dataset_name.")
            if dataset_name in seen:
                raise RuntimeServiceError(
                    (BUNDLE_LOAD_FAILED,),
                    "Manifest contains duplicate dataset entries for '{0}'.".format(dataset_name),
                )
            seen.add(dataset_name)
            self._manifest_entries[dataset_name] = dict(raw_entry)

        for dataset_name in sorted(self._manifest_entries):
            raw_data_path = self.data_root / "{0}.csv".format(dataset_name)
            if not raw_data_path.exists():
                raise RuntimeServiceError(
                    (BUNDLE_LOAD_FAILED,),
                    "Dataset CSV not found for '{0}': {1}".format(dataset_name, raw_data_path),
                )
            try:
                bundle = load_dataset_model_bundle(
                    data_name=dataset_name,
                    raw_data_path=raw_data_path,
                    artifact_root=self.artifact_root,
                )
            except Exception as exc:  # pragma: no cover - exercised through tests via raised error
                raise RuntimeServiceError(
                    (BUNDLE_LOAD_FAILED,),
                    "Failed to load bundle for '{0}': {1}".format(dataset_name, exc),
                ) from exc
            self._cross_check_manifest_entry(self._manifest_entries[dataset_name], bundle)
            self._bundles[dataset_name] = bundle

    def _load_manifest_payload(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise RuntimeServiceError(
                (BUNDLE_LOAD_FAILED,),
                "Manifest not found: {0}".format(self.manifest_path),
            )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeServiceError((BUNDLE_LOAD_FAILED,), "Manifest payload must be an object.")
        return payload

    def _cross_check_manifest_entry(self, entry: dict[str, Any], bundle: DatasetModelBundle) -> None:
        checks = (
            ("dataset_name", bundle.dataset_name),
            ("artifact_version", bundle.artifact_version),
            ("source_data_hash", bundle.source_data_hash),
            ("has_scaler", bool(bundle.has_scaler)),
            ("has_movie_distance_scaler", bool(bundle.has_movie_distance_scaler)),
        )
        for key, expected in checks:
            if key not in entry:
                raise RuntimeServiceError(
                    (BUNDLE_LOAD_FAILED,),
                    "Manifest entry for '{0}' is missing {1}.".format(bundle.dataset_name, key),
                )
            if entry[key] != expected:
                raise RuntimeServiceError(
                    (BUNDLE_LOAD_FAILED,),
                    "Manifest mismatch for '{0}' field '{1}': expected {2!r}, found {3!r}.".format(
                        bundle.dataset_name,
                        key,
                        expected,
                        entry[key],
                    ),
                )
