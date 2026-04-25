from __future__ import annotations

from llm.src.runtime.backend_packages.ar.adapter import ARCanonicalBackend
from llm.src.runtime.backend_packages.dice.adapter import DiCECanonicalBackend
from llm.src.runtime.backend_packages.ufce.adapter import UFCECanonicalBackend


def build_default_backends() -> dict[str, object]:
    return {
        "ufce": UFCECanonicalBackend(),
        "dice": DiCECanonicalBackend(),
        "ar": ARCanonicalBackend(),
    }
