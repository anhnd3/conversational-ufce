from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

__all__ = ["RuntimeOrchestrator"]


def __getattr__(name: str):
    if name == "RuntimeOrchestrator":
        from llm.src.runtime.orchestrator import RuntimeOrchestrator

        return RuntimeOrchestrator
    raise AttributeError(name)
