"""UFCE external binary evaluation workflow.

This package is intentionally isolated from the thesis reproduction runners.  It
implements the frozen UPV-2025 Phase-1 contract and writes artifacts only to the
caller supplied output directory.
"""

__all__ = [
    "config",
    "constraints",
    "dataset",
    "model",
    "verification",
    "adapters",
    "metrics",
]

