from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BACKEND_MANIFEST_PATH = ROOT / "llm" / "config" / "backend_manifest.json"


@dataclass(frozen=True)
class BackendCompatibilityManifest:
    backend_id: str
    request_contract_version: str
    candidate_contract_version: str
    capabilities: dict[str, Any]
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "request_contract_version": self.request_contract_version,
            "candidate_contract_version": self.candidate_contract_version,
            "capabilities": dict(self.capabilities),
            "enabled": bool(self.enabled),
        }


class BackendRegistry:
    def __init__(
        self,
        *,
        backends: dict[str, Any],
        manifest_path: Path | None = None,
    ) -> None:
        self._backends = {
            str(backend_id).strip().lower(): backend
            for backend_id, backend in backends.items()
        }
        self.manifest_path = Path(manifest_path or DEFAULT_BACKEND_MANIFEST_PATH)
        self._manifest_entries = self._load_manifest()
        self._validate_backends()

    def has(self, backend_id: str) -> bool:
        normalized = str(backend_id).strip().lower()
        return normalized in self._backends and normalized in self._manifest_entries and self._manifest_entries[
            normalized
        ].enabled

    def get(self, backend_id: str):
        normalized = str(backend_id).strip().lower()
        if normalized not in self._backends:
            raise KeyError(f"Unsupported backend: {backend_id}")
        manifest = self._manifest_entries.get(normalized)
        if manifest is None or not manifest.enabled:
            raise KeyError(f"Disabled backend: {backend_id}")
        return self._backends[normalized]

    def manifest(self, backend_id: str) -> BackendCompatibilityManifest:
        normalized = str(backend_id).strip().lower()
        if normalized not in self._manifest_entries:
            raise KeyError(f"Unsupported backend manifest: {backend_id}")
        return self._manifest_entries[normalized]

    def _load_manifest(self) -> dict[str, BackendCompatibilityManifest]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("backends")
        if not isinstance(entries, list):
            raise ValueError("Backend manifest 'backends' must be a list.")
        manifests: dict[str, BackendCompatibilityManifest] = {}
        for entry in entries:
            backend_id = str(entry["backend_id"]).strip().lower()
            manifests[backend_id] = BackendCompatibilityManifest(
                backend_id=backend_id,
                request_contract_version=str(entry["request_contract_version"]),
                candidate_contract_version=str(entry["candidate_contract_version"]),
                capabilities=dict(entry.get("capabilities") or {}),
                enabled=bool(entry.get("enabled", True)),
            )
        return manifests

    def _validate_backends(self) -> None:
        missing = sorted(key for key in self._backends if key not in self._manifest_entries)
        if missing:
            raise ValueError(f"Backend manifest entries missing for: {', '.join(missing)}")
