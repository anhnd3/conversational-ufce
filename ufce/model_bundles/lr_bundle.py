from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

from ufce.core.data_processing import get_movie_user_constraints


DEFAULT_DATASETS: Tuple[str, ...] = ("bank", "grad", "wine", "bupa", "movie")
DEFAULT_ARTIFACT_VERSION = "v1"
BUNDLE_SCHEMA_VERSION = "1.0"
DEFAULT_TRAINING_LOGIC_VERSION = "legacy_classify_dataset_getModel_v1"
DEFAULT_MOVIE_SCALER_LOGIC_VERSION = "reproduce_full_table7_result_v1"
DEFAULT_NON_MOVIE_SCALER_LOGIC_VERSION = "not_applicable"
DEFAULT_N_TRIES = 10
DEFAULT_TEST_SIZE = 0.3
DEFAULT_MAX_ITER = 1000
DEFAULT_SPLIT_STRATEGY = "min_max_feature_ks"
DEFAULT_MODEL_TYPE = "logistic_regression"

MODEL_FILENAME = "model.joblib"
SCALER_FILENAME = "scaler.joblib"
MOVIE_DISTANCE_SCALER_FILENAME = "movie_distance_scaler.joblib"
METADATA_FILENAME = "metadata.json"
ROOT_MANIFEST_FILENAME = "manifest.json"

DATASET_LABELS = {
    "bank": "Personal Loan",
    "grad": "Chance of Admit",
    "wine": "quality",
    "movie": "Start_Tech_Oscar",
    "bupa": "Selector",
}

DATASET_DROP_COLUMNS = {
    "bank": ["Unnamed: 0", "age", "Experience"],
    "grad": ["Unnamed: 0"],
    "wine": ["Unnamed: 0"],
    "movie": ["Unnamed: 0"],
    "bupa": [],
}


@dataclass
class DatasetModelBundle:
    dataset_name: str
    artifact_version: str
    bundle_schema_version: str
    training_logic_version: str
    movie_scaler_logic_version: str
    model_type: str
    max_iter: int
    lr: Any
    lr_mean: float
    lr_std: float
    Xtrain: pd.DataFrame
    Xtest: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    dataset_df: pd.DataFrame
    scaler: Optional[Any]
    movie_distance_scaler: Optional[Any]
    random_state: int
    split_strategy: str
    n_tries: int
    test_size: float
    feature_order: List[str]
    label_col: str
    dropped_columns: List[str]
    source_data_path: str
    source_data_hash: str
    train_indices: List[int]
    test_indices: List[int]
    split_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    has_scaler: bool = False
    has_movie_distance_scaler: bool = False


def _coerce_dataset_name(data_name: str) -> str:
    name = str(data_name).strip().lower()
    if name not in DATASET_LABELS:
        raise ValueError("Unsupported dataset: {0}".format(data_name))
    return name


def _normalize_index_list(indices: Sequence[int]) -> List[int]:
    return [int(idx) for idx in indices]


def _resolve_movie_scaler_logic_version(
    dataset_name: str,
    movie_scaler_logic_version: Optional[str],
) -> str:
    if _coerce_dataset_name(dataset_name) == "movie":
        if movie_scaler_logic_version is None:
            return DEFAULT_MOVIE_SCALER_LOGIC_VERSION
        return str(movie_scaler_logic_version)
    return DEFAULT_NON_MOVIE_SCALER_LOGIC_VERSION


def _hash_file(path: Union[str, Path]) -> str:
    file_path = Path(path)
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, pd.Index):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _split_indices_to_frames(
    X: pd.DataFrame,
    y: pd.Series,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_idx = _normalize_index_list(train_indices)
    test_idx = _normalize_index_list(test_indices)
    Xtrain = X.iloc[train_idx].copy()
    Xtest = X.iloc[test_idx].copy()
    ytrain = y.iloc[train_idx].copy()
    ytest = y.iloc[test_idx].copy()
    return Xtrain, Xtest, ytrain, ytest


def prepare_dataset_for_training(
    dataset_df: pd.DataFrame,
    data_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, str, List[str]]:
    dataset_name = _coerce_dataset_name(data_name)
    label_col = DATASET_LABELS[dataset_name]
    dropped_columns = list(DATASET_DROP_COLUMNS[dataset_name])

    cleaned_dataset_df = dataset_df.copy()
    cleaned_dataset_df.reset_index(drop=True, inplace=True)

    for column in dropped_columns:
        del cleaned_dataset_df[column]

    X = cleaned_dataset_df.loc[:, cleaned_dataset_df.columns != label_col].copy()
    y = cleaned_dataset_df[label_col].copy()
    return cleaned_dataset_df, X, y, label_col, dropped_columns


def select_best_split(
    X: pd.DataFrame,
    y: pd.Series,
    n_tries: int = DEFAULT_N_TRIES,
    test_size: float = DEFAULT_TEST_SIZE,
) -> Tuple[int, List[int], List[int], List[Dict[str, Any]]]:
    if int(n_tries) <= 0:
        raise ValueError("n_tries must be greater than 0.")
    if float(test_size) <= 0.0 or float(test_size) >= 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    split_diagnostics: List[Dict[str, Any]] = []
    n_features = X.shape[1]
    if n_features == 0:
        raise ValueError("Cannot select a split for an empty feature matrix.")

    for random_state in range(int(n_tries)):
        X_train, X_test, _, _ = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        feature_ks = {}
        for feature_idx, feature_name in enumerate(X.columns):
            feature_ks[str(feature_name)] = float(
                stats.ks_2samp(X_train.iloc[:, feature_idx], X_test.iloc[:, feature_idx]).statistic
            )
        split_diagnostics.append(
            {
                "random_state": int(random_state),
                "max_ks_distance": float(max(feature_ks.values())),
                "feature_ks": feature_ks,
            }
        )

    best_diag = min(split_diagnostics, key=lambda item: (item["max_ks_distance"], item["random_state"]))
    random_state = int(best_diag["random_state"])
    Xtrain, Xtest, _, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return (
        random_state,
        _normalize_index_list(Xtrain.index.tolist()),
        _normalize_index_list(Xtest.index.tolist()),
        split_diagnostics,
    )


def build_movie_scalers(
    dataset_df: pd.DataFrame,
    Xtrain: pd.DataFrame,
    X: pd.DataFrame,
    features: Sequence[str],
    numf: Sequence[str],
    outcome_label: str,
) -> Tuple[Optional[Any], Optional[Any]]:
    del Xtrain
    del X

    cols = [col for col in numf if col in features and col in dataset_df.columns and col != outcome_label]
    if not cols:
        raise ValueError("Movie distance scaler could not find numeric columns to scale.")

    base = dataset_df.loc[:, list(features)].copy()
    mins = base.loc[:, cols].min()
    maxs = base.loc[:, cols].max()
    ranges = maxs - mins
    constant_cols = [col for col in cols if float(ranges[col]) == 0.0]

    mads = (ranges / 100.0).replace(0.0, 1.0)
    medians = mins
    movie_distance_scaler = {
        "kind": "movie_minmax_0_100",
        "scale_cols": list(cols),
        "medians": medians.astype(float),
        "mads": mads.astype(float),
        "mins": mins.astype(float),
        "maxs": maxs.astype(float),
        "constant_cols": list(constant_cols),
    }
    return None, movie_distance_scaler


def train_dataset_model_bundle(
    cleaned_dataset_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    data_name: str,
    label_col: str,
    dropped_columns: Sequence[str],
    source_data_path: Union[str, Path],
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
    n_tries: int = DEFAULT_N_TRIES,
    test_size: float = DEFAULT_TEST_SIZE,
    bundle_schema_version: str = BUNDLE_SCHEMA_VERSION,
    training_logic_version: str = DEFAULT_TRAINING_LOGIC_VERSION,
    movie_scaler_logic_version: str = DEFAULT_MOVIE_SCALER_LOGIC_VERSION,
) -> DatasetModelBundle:
    dataset_name = _coerce_dataset_name(data_name)
    source_path = Path(source_data_path).resolve()
    random_state, train_indices, test_indices, split_diagnostics = select_best_split(
        X=X,
        y=y,
        n_tries=n_tries,
        test_size=test_size,
    )
    Xtrain, Xtest, ytrain, _ = _split_indices_to_frames(X, y, train_indices, test_indices)

    lr = LogisticRegression(max_iter=DEFAULT_MAX_ITER)
    lr.fit(Xtrain, ytrain)
    scores = cross_val_score(lr, X=Xtrain, y=ytrain, cv=10, n_jobs=1)
    lr_mean = float(np.mean(scores))
    lr_std = float(np.std(scores))

    scaler = None
    movie_distance_scaler = None
    if dataset_name == "movie":
        (
            features,
            _catf,
            numf,
            _uf,
            _f2change,
            outcome_label,
            _desired_outcome,
            _nbr_features,
            _protectf,
            _data_lab0,
            _data_lab1,
        ) = get_movie_user_constraints(cleaned_dataset_df.copy())
        scaler, movie_distance_scaler = build_movie_scalers(
            dataset_df=cleaned_dataset_df,
            Xtrain=Xtrain,
            X=X,
            features=features,
            numf=numf,
            outcome_label=outcome_label,
        )

    bundle = DatasetModelBundle(
        dataset_name=dataset_name,
        artifact_version=str(artifact_version),
        bundle_schema_version=str(bundle_schema_version),
        training_logic_version=str(training_logic_version),
        movie_scaler_logic_version=_resolve_movie_scaler_logic_version(
            dataset_name,
            movie_scaler_logic_version,
        ),
        model_type=DEFAULT_MODEL_TYPE,
        max_iter=int(DEFAULT_MAX_ITER),
        lr=lr,
        lr_mean=lr_mean,
        lr_std=lr_std,
        Xtrain=Xtrain,
        Xtest=Xtest,
        X=X.copy(),
        y=y.copy(),
        dataset_df=cleaned_dataset_df.copy(),
        scaler=scaler,
        movie_distance_scaler=movie_distance_scaler,
        random_state=int(random_state),
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        n_tries=int(n_tries),
        test_size=float(test_size),
        feature_order=[str(column) for column in X.columns.tolist()],
        label_col=str(label_col),
        dropped_columns=[str(column) for column in dropped_columns],
        source_data_path=str(source_path),
        source_data_hash=_hash_file(source_path),
        train_indices=train_indices,
        test_indices=test_indices,
        split_diagnostics=list(split_diagnostics),
        has_scaler=scaler is not None,
        has_movie_distance_scaler=movie_distance_scaler is not None,
    )
    validate_bundle_against_metadata(bundle)
    return bundle


def _bundle_metadata(bundle: DatasetModelBundle) -> Dict[str, Any]:
    return {
        "dataset_name": bundle.dataset_name,
        "artifact_version": bundle.artifact_version,
        "bundle_schema_version": bundle.bundle_schema_version,
        "training_logic_version": bundle.training_logic_version,
        "movie_scaler_logic_version": bundle.movie_scaler_logic_version,
        "model_type": bundle.model_type,
        "max_iter": int(bundle.max_iter),
        "lr_mean": float(bundle.lr_mean),
        "lr_std": float(bundle.lr_std),
        "split_strategy": bundle.split_strategy,
        "n_tries": int(bundle.n_tries),
        "test_size": float(bundle.test_size),
        "random_state": int(bundle.random_state),
        "feature_order": [str(column) for column in bundle.feature_order],
        "label_col": bundle.label_col,
        "dropped_columns": [str(column) for column in bundle.dropped_columns],
        "train_indices": [int(idx) for idx in bundle.train_indices],
        "test_indices": [int(idx) for idx in bundle.test_indices],
        "split_diagnostics": _json_ready(bundle.split_diagnostics),
        "source_data_path": bundle.source_data_path,
        "source_data_hash": bundle.source_data_hash,
        "has_scaler": bool(bundle.has_scaler),
        "has_movie_distance_scaler": bool(bundle.has_movie_distance_scaler),
    }


def save_dataset_model_bundle(bundle: DatasetModelBundle, out_dir: Union[str, Path]) -> None:
    validate_bundle_against_metadata(bundle)
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.lr, target_dir / MODEL_FILENAME)
    if bundle.scaler is not None:
        joblib.dump(bundle.scaler, target_dir / SCALER_FILENAME)
    if bundle.movie_distance_scaler is not None:
        joblib.dump(bundle.movie_distance_scaler, target_dir / MOVIE_DISTANCE_SCALER_FILENAME)

    metadata = _bundle_metadata(bundle)
    _write_metadata(target_dir / METADATA_FILENAME, metadata)


def load_dataset_model_bundle(
    data_name: str,
    raw_data_path: Union[str, Path],
    artifact_root: Union[str, Path],
) -> DatasetModelBundle:
    dataset_name = _coerce_dataset_name(data_name)
    raw_path = Path(raw_data_path).resolve()
    dataset_dir = Path(artifact_root) / dataset_name
    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError("Bundle metadata not found: {0}".format(metadata_path))

    metadata = _load_metadata(metadata_path)
    if metadata.get("dataset_name") != dataset_name:
        raise ValueError(
            "Bundle dataset mismatch: expected {0}, found {1}".format(
                dataset_name,
                metadata.get("dataset_name"),
            )
        )

    source_hash = _hash_file(raw_path)
    if source_hash != metadata.get("source_data_hash"):
        raise ValueError(
            "Source data hash mismatch for dataset '{0}': expected {1}, found {2}".format(
                dataset_name,
                metadata.get("source_data_hash"),
                source_hash,
            )
        )

    raw_df = pd.read_csv(raw_path)
    cleaned_dataset_df, X, y, label_col, dropped_columns = prepare_dataset_for_training(raw_df, dataset_name)

    if label_col != metadata.get("label_col"):
        raise ValueError(
            "Label column mismatch for dataset '{0}': expected {1}, found {2}".format(
                dataset_name,
                metadata.get("label_col"),
                label_col,
            )
        )
    if list(dropped_columns) != list(metadata.get("dropped_columns", [])):
        raise ValueError(
            "Dropped column mismatch for dataset '{0}': expected {1}, found {2}".format(
                dataset_name,
                metadata.get("dropped_columns", []),
                dropped_columns,
            )
        )
    if list(X.columns) != list(metadata.get("feature_order", [])):
        raise ValueError(
            "Feature order mismatch for dataset '{0}': expected {1}, found {2}".format(
                dataset_name,
                metadata.get("feature_order", []),
                list(X.columns),
            )
        )

    train_indices = _normalize_index_list(metadata.get("train_indices", []))
    test_indices = _normalize_index_list(metadata.get("test_indices", []))
    Xtrain, Xtest, _, _ = _split_indices_to_frames(X, y, train_indices, test_indices)

    model_path = dataset_dir / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError("Bundle model not found: {0}".format(model_path))
    lr = joblib.load(model_path)

    has_scaler = bool(metadata.get("has_scaler", False))
    scaler = None
    if has_scaler:
        scaler_path = dataset_dir / SCALER_FILENAME
        if not scaler_path.exists():
            raise FileNotFoundError("Bundle scaler not found: {0}".format(scaler_path))
        scaler = joblib.load(scaler_path)

    has_movie_distance_scaler = bool(metadata.get("has_movie_distance_scaler", False))
    movie_distance_scaler = None
    if has_movie_distance_scaler:
        movie_scaler_path = dataset_dir / MOVIE_DISTANCE_SCALER_FILENAME
        if not movie_scaler_path.exists():
            raise FileNotFoundError(
                "Bundle movie distance scaler not found: {0}".format(movie_scaler_path)
            )
        movie_distance_scaler = joblib.load(movie_scaler_path)

    bundle = DatasetModelBundle(
        dataset_name=dataset_name,
        artifact_version=str(metadata.get("artifact_version", DEFAULT_ARTIFACT_VERSION)),
        bundle_schema_version=str(metadata.get("bundle_schema_version", BUNDLE_SCHEMA_VERSION)),
        training_logic_version=str(
            metadata.get("training_logic_version", DEFAULT_TRAINING_LOGIC_VERSION)
        ),
        movie_scaler_logic_version=_resolve_movie_scaler_logic_version(
            dataset_name,
            metadata.get("movie_scaler_logic_version"),
        ),
        model_type=str(metadata.get("model_type", DEFAULT_MODEL_TYPE)),
        max_iter=int(metadata.get("max_iter", DEFAULT_MAX_ITER)),
        lr=lr,
        lr_mean=float(metadata["lr_mean"]),
        lr_std=float(metadata["lr_std"]),
        Xtrain=Xtrain,
        Xtest=Xtest,
        X=X.copy(),
        y=y.copy(),
        dataset_df=cleaned_dataset_df.copy(),
        scaler=scaler,
        movie_distance_scaler=movie_distance_scaler,
        random_state=int(metadata["random_state"]),
        split_strategy=str(metadata["split_strategy"]),
        n_tries=int(metadata["n_tries"]),
        test_size=float(metadata["test_size"]),
        feature_order=[str(column) for column in metadata.get("feature_order", [])],
        label_col=str(metadata["label_col"]),
        dropped_columns=[str(column) for column in metadata.get("dropped_columns", [])],
        source_data_path=str(metadata.get("source_data_path", raw_path)),
        source_data_hash=str(source_hash),
        train_indices=train_indices,
        test_indices=test_indices,
        split_diagnostics=list(metadata.get("split_diagnostics", [])),
        has_scaler=has_scaler,
        has_movie_distance_scaler=has_movie_distance_scaler,
    )
    validate_bundle_against_metadata(bundle)
    return bundle


def validate_bundle_against_metadata(bundle: DatasetModelBundle) -> None:
    dataset_name = _coerce_dataset_name(bundle.dataset_name)
    expected_label = DATASET_LABELS[dataset_name]
    if bundle.label_col != expected_label:
        raise ValueError(
            "Bundle label mismatch for dataset '{0}': expected {1}, found {2}".format(
                dataset_name,
                expected_label,
                bundle.label_col,
            )
        )

    if bundle.label_col not in bundle.dataset_df.columns:
        raise ValueError(
            "Bundle dataset_df is missing the label column '{0}'.".format(bundle.label_col)
        )

    feature_order = list(bundle.feature_order)
    if list(bundle.X.columns) != feature_order:
        raise ValueError(
            "Bundle feature order mismatch: expected {0}, found {1}".format(
                feature_order,
                list(bundle.X.columns),
            )
        )

    dataset_feature_order = list(bundle.dataset_df.drop(columns=[bundle.label_col]).columns)
    if dataset_feature_order != feature_order:
        raise ValueError(
            "Bundle dataset_df feature order mismatch: expected {0}, found {1}".format(
                feature_order,
                dataset_feature_order,
            )
        )

    n_rows = len(bundle.X)
    if len(bundle.y) != n_rows or len(bundle.dataset_df) != n_rows:
        raise ValueError("Bundle X, y, and dataset_df lengths must match.")

    train_indices = _normalize_index_list(bundle.train_indices)
    test_indices = _normalize_index_list(bundle.test_indices)
    if not train_indices or not test_indices:
        raise ValueError("Bundle train/test indices must both be non-empty.")
    if len(train_indices) != len(set(train_indices)):
        raise ValueError("Bundle train_indices must be unique.")
    if len(test_indices) != len(set(test_indices)):
        raise ValueError("Bundle test_indices must be unique.")
    if set(train_indices).intersection(test_indices):
        raise ValueError("Bundle train/test indices must be disjoint.")
    if min(train_indices + test_indices) < 0 or max(train_indices + test_indices) >= n_rows:
        raise ValueError("Bundle train/test indices are out of bounds.")
    if set(train_indices).union(test_indices) != set(range(n_rows)):
        raise ValueError("Bundle train/test indices must partition all rows.")

    if bundle.Xtrain.index.tolist() != train_indices:
        raise ValueError("Bundle Xtrain indices do not match saved train_indices.")
    if bundle.Xtest.index.tolist() != test_indices:
        raise ValueError("Bundle Xtest indices do not match saved test_indices.")
    if list(bundle.Xtrain.columns) != feature_order:
        raise ValueError("Bundle Xtrain feature order does not match bundle feature_order.")
    if list(bundle.Xtest.columns) != feature_order:
        raise ValueError("Bundle Xtest feature order does not match bundle feature_order.")

    has_scaler = bundle.scaler is not None
    if bool(bundle.has_scaler) != has_scaler:
        raise ValueError("Bundle has_scaler flag does not match scaler presence.")

    has_movie_distance_scaler = bundle.movie_distance_scaler is not None
    if bool(bundle.has_movie_distance_scaler) != has_movie_distance_scaler:
        raise ValueError(
            "Bundle has_movie_distance_scaler flag does not match movie scaler presence."
        )

    if dataset_name == "movie":
        if not has_movie_distance_scaler:
            raise ValueError("Movie bundle must include movie_distance_scaler.")
        required_keys = {
            "kind",
            "scale_cols",
            "medians",
            "mads",
            "mins",
            "maxs",
            "constant_cols",
        }
        missing_keys = required_keys.difference(bundle.movie_distance_scaler.keys())
        if missing_keys:
            raise ValueError(
                "Movie distance scaler is missing keys: {0}".format(sorted(missing_keys))
            )


def bundle_to_legacy_tuple(bundle: DatasetModelBundle) -> Tuple[Any, ...]:
    return (
        bundle.lr,
        bundle.lr_mean,
        bundle.lr_std,
        bundle.Xtest,
        bundle.Xtrain,
        bundle.X,
        bundle.y,
        bundle.dataset_df,
        bundle.scaler,
    )


def load_all_dataset_model_bundles(
    data_dir: Union[str, Path],
    artifact_root: Union[str, Path],
    datasets: Optional[Iterable[str]] = None,
) -> Dict[str, DatasetModelBundle]:
    if datasets is None:
        selected_datasets = DEFAULT_DATASETS
    elif isinstance(datasets, str):
        if datasets.strip().lower() == "all":
            selected_datasets = DEFAULT_DATASETS
        else:
            selected_datasets = tuple(item.strip() for item in datasets.split(",") if item.strip())
    else:
        selected_datasets = tuple(datasets)
    registry: Dict[str, DatasetModelBundle] = {}
    base_data_dir = Path(data_dir)
    for dataset_name in selected_datasets:
        normalized_name = _coerce_dataset_name(dataset_name)
        raw_data_path = base_data_dir / "{0}.csv".format(normalized_name)
        registry[normalized_name] = load_dataset_model_bundle(
            data_name=normalized_name,
            raw_data_path=raw_data_path,
            artifact_root=artifact_root,
        )
    return registry
