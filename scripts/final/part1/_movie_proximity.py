from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


MOVIE_PROX_EUC_CONTRACT = "RMS Euclidean over movie_minmax_0_100 distance space"


def apply_affine_distance_scaler(
    df: pd.DataFrame,
    scaler: Optional[Dict[str, object]],
) -> pd.DataFrame:
    if scaler is None:
        return df.copy()
    out = df.copy()
    scale_cols = scaler.get("scale_cols", None)
    if scale_cols is None:
        scale_cols = list(getattr(scaler.get("medians", pd.Series(dtype=float)), "index", []))
    cols = [col for col in scale_cols if col in out.columns]
    if not cols:
        return out
    out.loc[:, cols] = (out.loc[:, cols] - scaler["medians"].reindex(cols)) / scaler["mads"].reindex(cols)
    if scaler.get("kind") == "movie_minmax_0_100":
        out.loc[:, cols] = out.loc[:, cols].clip(lower=0.0, upper=100.0)
    constant_cols = [col for col in scaler.get("constant_cols", []) if col in cols]
    if constant_cols:
        out.loc[:, constant_cols] = 0.0
    return out


def shared_numeric_features(numf: Sequence[str], *dfs: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in numf:
        if all(isinstance(df, pd.DataFrame) and col in df.columns for df in dfs):
            cols.append(str(col))
    return cols


def normalized_l2_0_100(
    factual: pd.DataFrame,
    candidate: pd.DataFrame,
    numf: Sequence[str],
    distance_scaler: Optional[Dict[str, object]],
) -> float:
    if not isinstance(factual, pd.DataFrame) or factual.empty:
        return float("nan")
    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return float("nan")
    cols = shared_numeric_features(numf, factual, candidate)
    if not cols:
        return float("nan")

    factual_dist = apply_affine_distance_scaler(factual.loc[:, cols], distance_scaler)
    candidate_dist = apply_affine_distance_scaler(candidate.loc[:, cols], distance_scaler)
    delta = candidate_dist.iloc[0].to_numpy(dtype=float) - factual_dist.iloc[0].to_numpy(dtype=float)
    if delta.size == 0:
        return float("nan")
    return float(np.linalg.norm(delta) / np.sqrt(float(delta.size)))


def pairwise_normalized_l2_0_100_values(
    factual_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    numf: Sequence[str],
    distance_scaler: Optional[Dict[str, object]],
) -> List[float]:
    n_pairs = min(len(factual_df), len(candidate_df))
    values: List[float] = []
    for i in range(n_pairs):
        value = normalized_l2_0_100(
            factual_df.iloc[i : i + 1],
            candidate_df.iloc[i : i + 1],
            numf,
            distance_scaler,
        )
        if not np.isnan(value):
            values.append(float(value))
    return values


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr) / np.sqrt(float(arr.size)))
