#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufce.model_bundles.lr_bundle import (
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_ARTIFACT_VERSION,
    DEFAULT_DATASETS,
    DEFAULT_MOVIE_SCALER_LOGIC_VERSION,
    DEFAULT_N_TRIES,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAINING_LOGIC_VERSION,
    METADATA_FILENAME,
    ROOT_MANIFEST_FILENAME,
    bundle_to_legacy_tuple,
    load_dataset_model_bundle,
    prepare_dataset_for_training,
    save_dataset_model_bundle,
    train_dataset_model_bundle,
    validate_bundle_against_metadata,
)


def _parse_datasets(raw_value: str) -> List[str]:
    value = str(raw_value).strip()
    if not value or value.lower() == "all":
        return list(DEFAULT_DATASETS)
    datasets = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not datasets:
        raise ValueError("No datasets were provided.")
    unknown = sorted(set(datasets).difference(DEFAULT_DATASETS))
    if unknown:
        raise ValueError("Unsupported datasets: {0}".format(", ".join(unknown)))
    return datasets


def _ensure_output_dir(target_dir: Path, overwrite: bool) -> None:
    if not target_dir.exists():
        return
    if not overwrite and any(target_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite existing artifacts in {0}. Use --overwrite to replace them.".format(
                target_dir
            )
        )


def _write_root_manifest(out_root: Path, records: Iterable[Dict[str, object]]) -> None:
    payload = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": list(records),
    }
    with (out_root / ROOT_MANIFEST_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_round_trip(dataset_name: str, bundle, loaded_bundle) -> None:
    validate_bundle_against_metadata(bundle)
    validate_bundle_against_metadata(loaded_bundle)

    expected_preds = np.asarray(bundle.lr.predict(bundle.Xtest))
    loaded_preds = np.asarray(loaded_bundle.lr.predict(loaded_bundle.Xtest))
    if not np.array_equal(expected_preds, loaded_preds):
        raise ValueError("Prediction mismatch after round-trip for dataset '{0}'.".format(dataset_name))
    if not math.isclose(bundle.lr_mean, loaded_bundle.lr_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("lr_mean mismatch after round-trip for dataset '{0}'.".format(dataset_name))
    if not math.isclose(bundle.lr_std, loaded_bundle.lr_std, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("lr_std mismatch after round-trip for dataset '{0}'.".format(dataset_name))
    if len(bundle_to_legacy_tuple(loaded_bundle)) != 9:
        raise ValueError("Legacy tuple adapter did not return 9 items for dataset '{0}'.".format(dataset_name))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export deterministic LR model bundles for UFCE datasets.")
    parser.add_argument("--data_dir", default="ufce/data", help="Directory containing source CSV files.")
    parser.add_argument("--out_root", default="llm/models", help="Artifact root directory.")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset list or 'all'. Default: all",
    )
    parser.add_argument("--artifact_version", default=DEFAULT_ARTIFACT_VERSION)
    parser.add_argument("--n_tries", type=int, default=DEFAULT_N_TRIES)
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset artifact directories.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_root).resolve()
    datasets = _parse_datasets(args.datasets)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, object]] = []
    print(
        "[EXPORT] data_dir={0} out_root={1} datasets={2} artifact_version={3}".format(
            data_dir,
            out_root,
            ",".join(datasets),
            args.artifact_version,
        )
    )

    for dataset_name in datasets:
        raw_data_path = data_dir / "{0}.csv".format(dataset_name)
        dataset_out_dir = out_root / dataset_name
        if not raw_data_path.exists():
            raise FileNotFoundError("Dataset CSV not found: {0}".format(raw_data_path))
        _ensure_output_dir(dataset_out_dir, overwrite=args.overwrite)

        raw_df = pd.read_csv(raw_data_path)
        cleaned_dataset_df, X, y, label_col, dropped_columns = prepare_dataset_for_training(
            raw_df,
            dataset_name,
        )
        bundle = train_dataset_model_bundle(
            cleaned_dataset_df=cleaned_dataset_df,
            X=X,
            y=y,
            data_name=dataset_name,
            label_col=label_col,
            dropped_columns=dropped_columns,
            source_data_path=raw_data_path,
            artifact_version=args.artifact_version,
            n_tries=args.n_tries,
            test_size=args.test_size,
            bundle_schema_version=BUNDLE_SCHEMA_VERSION,
            training_logic_version=DEFAULT_TRAINING_LOGIC_VERSION,
            movie_scaler_logic_version=DEFAULT_MOVIE_SCALER_LOGIC_VERSION,
        )
        save_dataset_model_bundle(bundle, dataset_out_dir)
        loaded_bundle = load_dataset_model_bundle(
            data_name=dataset_name,
            raw_data_path=raw_data_path,
            artifact_root=out_root,
        )
        _validate_round_trip(dataset_name, bundle, loaded_bundle)

        metadata_path = dataset_out_dir / METADATA_FILENAME
        print(
            "[OK] dataset={0} split_random_state={1} train_shape={2} test_shape={3} "
            "lr_mean={4:.12f} lr_std={5:.12f} metadata={6}".format(
                dataset_name,
                bundle.random_state,
                bundle.Xtrain.shape,
                bundle.Xtest.shape,
                bundle.lr_mean,
                bundle.lr_std,
                metadata_path,
            )
        )
        manifest_records.append(
            {
                "dataset_name": dataset_name,
                "artifact_version": bundle.artifact_version,
                "bundle_schema_version": bundle.bundle_schema_version,
                "has_scaler": bool(bundle.has_scaler),
                "has_movie_distance_scaler": bool(bundle.has_movie_distance_scaler),
                "training_logic_version": bundle.training_logic_version,
                "movie_scaler_logic_version": bundle.movie_scaler_logic_version,
                "source_data_hash": bundle.source_data_hash,
                "random_state": bundle.random_state,
            }
        )

    _write_root_manifest(out_root, manifest_records)
    print("[DONE] Wrote manifest: {0}".format(out_root / ROOT_MANIFEST_FILENAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
