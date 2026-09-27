"""Precompute and validate train-only MI pairs independently of CF queries."""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np

from .adapters import compute_mi_pairs
from .config import ACTIONABLE_FEATURES, MI_TOP_K, SPLIT_SEED
from .dataset import load_dataset, split_dataset, write_json
from .space import EncodedSpace


CACHE_VERSION = 1
DEFAULT_DATA_PATH = Path("data/native_multiclass_eval/uci_students_697.zip")
DEFAULT_CACHE_PATH = Path("outputs/native_multiclass_eval/student_outcomes_train_mi.json")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _identity(train_encoded, train_row_ids, dataset_sha256, seed):
    names = list(train_encoded.columns)
    if not set(ACTIONABLE_FEATURES).issubset(names):
        raise ValueError("Training frame is missing actionable MI features")
    values = np.ascontiguousarray(train_encoded.loc[:, names].to_numpy(dtype=np.float64))
    row_ids = np.ascontiguousarray(np.asarray(train_row_ids, dtype=np.int64))
    if len(values) != len(row_ids) or not np.isfinite(values).all():
        raise ValueError("Training rows or encoded values are invalid for MI")
    return {
        "cache_version": CACHE_VERSION,
        "scope": "train_only",
        "source_dataset_sha256": str(dataset_sha256),
        "split_seed": SPLIT_SEED,
        "mi_seed": int(seed),
        "top_k": MI_TOP_K,
        "actionable_features": list(ACTIONABLE_FEATURES),
        "encoded_feature_names": names,
        "train_row_count": int(len(values)),
        "train_row_ids_sha256": _sha256_bytes(row_ids.tobytes()),
        "train_encoded_sha256": _sha256_bytes(values.tobytes()),
        "mi_implementation_sha256": _sha256_bytes(inspect.getsource(compute_mi_pairs).encode("utf-8")),
    }


def load_mi_cache(path, train_encoded, train_row_ids, dataset_sha256, seed=SPLIT_SEED):
    """Fail closed when the saved ranking does not match the frozen reference."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = _identity(train_encoded, train_row_ids, dataset_sha256, seed)
    mismatches = [key for key, value in expected.items() if artifact.get(key) != value]
    if mismatches:
        raise ValueError("MI cache does not match training reference: %s" % ", ".join(mismatches))
    pairs = artifact.get("mi_pairs")
    if not isinstance(pairs, list) or len(pairs) != MI_TOP_K:
        raise ValueError("MI cache has an invalid ranked pair list")
    keys = []
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]
                or any(feature not in ACTIONABLE_FEATURES for feature in pair)):
            raise ValueError("MI cache contains an invalid feature pair")
        keys.append(tuple(sorted(pair)))
    if len(set(keys)) != len(keys):
        raise ValueError("MI cache contains duplicate feature pairs")
    pair_hash = _sha256_bytes(json.dumps(pairs, separators=(",", ":")).encode("utf-8"))
    if artifact.get("mi_pairs_sha256") != pair_hash:
        raise ValueError("MI cache pair checksum differs from saved ranking")
    elapsed = artifact.get("precompute_seconds")
    if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("MI cache has invalid precompute time")
    return artifact


def prepare_mi_cache(path, train_encoded, train_row_ids, dataset_sha256,
                     seed=SPLIT_SEED, require_existing=False):
    """Load an existing MI ranking, or compute and persist it once."""
    path = Path(path)
    if path.exists():
        return load_mi_cache(path, train_encoded, train_row_ids, dataset_sha256, seed), True
    if require_existing:
        raise FileNotFoundError("Precomputed MI cache does not exist: %s" % path)
    identity = _identity(train_encoded, train_row_ids, dataset_sha256, seed)
    started = time.perf_counter()
    pairs = compute_mi_pairs(train_encoded, seed=seed)
    elapsed = time.perf_counter() - started
    artifact = dict(identity)
    artifact.update({
        "mi_pairs": pairs,
        "mi_pairs_sha256": _sha256_bytes(
            json.dumps(pairs, separators=(",", ":")).encode("utf-8")
        ),
        "precompute_seconds": float(elapsed),
    })
    write_json(path, artifact)
    return load_mi_cache(path, train_encoded, train_row_ids, dataset_sha256, seed), False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--out", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--verify-freeze", help="Optional existing freeze_manifest.json to compare MI pairs")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    data_path = Path(args.data_path)
    cache_path = Path(args.out)
    if not data_path.is_absolute():
        data_path = root / data_path
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    features, labels, schema, summary = load_dataset(data_path, auto_download=False)
    train_row_ids = split_dataset(features, labels, SPLIT_SEED)["train"]
    train_raw = features.iloc[train_row_ids].reset_index(drop=True)
    train_encoded = EncodedSpace(train_raw, schema).encode(train_raw)
    artifact, reused = prepare_mi_cache(
        cache_path, train_encoded, train_row_ids, summary["source_sha256"]
    )
    if args.verify_freeze:
        freeze_path = Path(args.verify_freeze)
        if not freeze_path.is_absolute():
            freeze_path = root / freeze_path
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))["configuration"]
        if frozen["dataset"]["source_sha256"] != summary["source_sha256"]:
            raise ValueError("Freeze manifest refers to another dataset")
        if artifact["mi_pairs"] != frozen["mi_pairs"]:
            raise ValueError("Precomputed MI pairs differ from the frozen experiment")
    print(json.dumps({
        "path": str(cache_path), "reused": reused,
        "train_rows": artifact["train_row_count"],
        "precompute_seconds": artifact["precompute_seconds"],
        "mi_pairs": artifact["mi_pairs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
