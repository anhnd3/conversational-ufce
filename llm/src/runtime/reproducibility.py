from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from llm.src.runtime.types import CounterfactualCandidate


RUNTIME_MODE_STABLE_DEMO = "stable_demo"


def build_deterministic_seed(
    *,
    dataset_name: str,
    canonical_profile: dict[str, Any],
    feature_order: list[str],
    policy_version: str,
) -> int:
    ordered_profile = {feature: canonical_profile.get(feature) for feature in feature_order}
    payload = {
        "dataset": dataset_name,
        "policy_version": policy_version,
        "profile": ordered_profile,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return int(digest[:8], 16)


@contextmanager
def deterministic_seed(seed: int) -> Iterator[None]:
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def sort_counterfactual_candidates(
    *,
    candidates: list[CounterfactualCandidate],
    feature_order: list[str],
    prefer_fewer_changes: bool = False,
) -> list[CounterfactualCandidate]:
    method_priority = {"sfexp": 0, "dfexp": 1, "tfexp": 2}

    def key(candidate: CounterfactualCandidate):
        profile_key = tuple(candidate.profile.get(feature) for feature in feature_order)
        if prefer_fewer_changes:
            return (
                len(candidate.changed_features),
                method_priority.get(candidate.method, 99),
                int(candidate.rank),
                tuple(candidate.changed_features),
                profile_key,
            )
        return (
            method_priority.get(candidate.method, 99),
            int(candidate.rank),
            len(candidate.changed_features),
            tuple(candidate.changed_features),
            profile_key,
        )

    return sorted(candidates, key=key)
