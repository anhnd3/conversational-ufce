from __future__ import annotations


def detect_conflicts(candidate: dict | None) -> tuple[str, ...]:
    if not isinstance(candidate, dict):
        return ()
    conflicts = candidate.get("conflicts", ())
    if isinstance(conflicts, list):
        return tuple(item for item in conflicts if isinstance(item, str))
    return ()
