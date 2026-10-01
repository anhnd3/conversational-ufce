import itertools
import json

import numpy as np
import pandas as pd
import pytest

from . import mi_cache
from .config import ACTIONABLE_FEATURES, MI_TOP_K


def _encoded_train():
    rng = np.random.RandomState(7)
    return pd.DataFrame({
        name: rng.randint(0, 8, size=40).astype(float)
        for name in ACTIONABLE_FEATURES
    })


def test_cache_reuses_ranked_pairs_without_recomputing(tmp_path, monkeypatch):
    train = _encoded_train()
    row_ids = np.arange(len(train), dtype=np.int64)
    expected_pairs = [
        list(pair) for pair in itertools.islice(
            itertools.combinations(ACTIONABLE_FEATURES, 2), MI_TOP_K
        )
    ]
    calls = []

    def fake_compute(frame, seed):
        calls.append((len(frame), seed))
        return expected_pairs

    monkeypatch.setattr(mi_cache, "compute_mi_pairs", fake_compute)
    path = tmp_path / "ranked-mi.json"
    first, reused = mi_cache.prepare_mi_cache(path, train, row_ids, "dataset-hash")
    assert not reused
    assert first["mi_pairs"] == expected_pairs
    assert first["scope"] == "train_only"
    second, reused = mi_cache.prepare_mi_cache(path, train, row_ids, "dataset-hash")
    assert reused
    assert second == first
    assert calls == [(len(train), 42)]


def test_cache_rejects_changed_reference_or_dataset(tmp_path):
    train = _encoded_train()
    row_ids = np.arange(len(train), dtype=np.int64)
    path = tmp_path / "ranked-mi.json"
    mi_cache.prepare_mi_cache(path, train, row_ids, "dataset-hash")

    changed = train.copy()
    changed.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="train_encoded_sha256"):
        mi_cache.load_mi_cache(path, changed, row_ids, "dataset-hash")
    with pytest.raises(ValueError, match="train_row_ids_sha256"):
        mi_cache.load_mi_cache(path, train, row_ids[::-1], "dataset-hash")
    with pytest.raises(ValueError, match="source_dataset_sha256"):
        mi_cache.load_mi_cache(path, train, row_ids, "different-dataset")
    with pytest.raises(FileNotFoundError):
        mi_cache.prepare_mi_cache(
            tmp_path / "missing.json", train, row_ids, "dataset-hash",
            require_existing=True,
        )

    artifact = json.loads(path.read_text())
    used = {tuple(sorted(pair)) for pair in artifact["mi_pairs"]}
    replacement = next(
        list(pair) for pair in itertools.combinations(ACTIONABLE_FEATURES, 2)
        if tuple(sorted(pair)) not in used
    )
    artifact["mi_pairs"][0] = replacement
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="pair checksum"):
        mi_cache.load_mi_cache(path, train, row_ids, "dataset-hash")
