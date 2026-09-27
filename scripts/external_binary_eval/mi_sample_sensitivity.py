"""Offline sensitivity check for UPV MI rankings across train reference sizes.

Run from the repository root with .venv/bin/python -m
scripts.external_binary_eval.mi_sample_sensitivity. Outputs are diagnostic only;
the frozen experiment is not modified.
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from .adapters import EncodedSpace
from .config import RANDOM_SEED, REGIMES, STUDENT_ID
from .constraints import actionable_features
from .dataset import group_split, load_upv, write_json
from .run_ten_percent_eval import _student_reference_subset


_FRAME = None
_NAMES = None


def _score_pair(pair):
    left_index, right_index = pair
    left, right = _NAMES[left_index], _NAMES[right_index]
    started = time.perf_counter()
    try:
        score = float(mutual_info_regression(
            _FRAME[[left]], _FRAME[right], random_state=0
        )[0])
        error = None
    except Exception as exc:
        score, error = 0.0, "%s: %s" % (type(exc).__name__, exc)
    return left_index, right_index, score, time.perf_counter() - started, error


def _rank(frame, names, workers):
    global _FRAME, _NAMES
    _FRAME, _NAMES = frame, names
    pairs = list(itertools.combinations(range(len(names)), 2))
    started = time.perf_counter()
    if workers == 1:
        results = list(map(_score_pair, pairs))
    else:
        # fork shares the immutable encoded frame without serializing 325k rows
        # for each pair. Each worker still calls the original sklearn estimator.
        with mp.get_context("fork").Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_score_pair, pairs, chunksize=1))
    elapsed = time.perf_counter() - started
    scores = {(names[i], names[j]): score for i, j, score, _, _ in results}
    ranked = sorted(scores, key=lambda pair: (-scores[pair], pair[0], pair[1]))
    failures = [
        {"left": names[i], "right": names[j], "error": error}
        for i, j, _, _, error in results if error
    ]
    return scores, ranked, {
        "rows": len(frame), "calls": len(results), "wall_seconds": elapsed,
        "sum_call_seconds": sum(result[3] for result in results),
        "median_call_seconds": float(np.median([result[3] for result in results])),
        "max_call_seconds": max(result[3] for result in results),
        "failures": failures,
    }


def _compare(base, candidate, numeric):
    base_scores, base_rank = base
    scores, ranked = candidate
    numeric = set(numeric)
    numeric_base = [pair for pair in base_rank if set(pair) <= numeric]
    numeric_new = [pair for pair in ranked if set(pair) <= numeric]
    dedication_base = [pair for pair in base_rank if "dedicacion" in pair]
    dedication_new = [pair for pair in ranked if "dedicacion" in pair]
    ordered_pairs = sorted(base_scores)
    base_positions = {pair: i for i, pair in enumerate(base_rank)}
    new_positions = {pair: i for i, pair in enumerate(ranked)}
    return {
        "numeric_top100_overlap": len(set(numeric_base[:100]) & set(numeric_new[:100])),
        "numeric_top20_overlap": len(set(numeric_base[:20]) & set(numeric_new[:20])),
        "numeric_top5_overlap": len(set(numeric_base[:5]) & set(numeric_new[:5])),
        "dedicacion_top5_overlap": len(set(dedication_base[:5]) & set(dedication_new[:5])),
        "combined_105_overlap": len(set(numeric_base[:100] + dedication_base[:5]) & set(numeric_new[:100] + dedication_new[:5])),
        "spearman_406_ranks": float(spearmanr(
            [base_positions[pair] for pair in ordered_pairs],
            [new_positions[pair] for pair in ordered_pairs],
        )[0]),
        "numeric_top5_base": [list(pair) for pair in numeric_base[:5]],
        "numeric_top5_new": [list(pair) for pair in numeric_new[:5]],
        "dedicacion_top5_base": [list(pair) for pair in dedication_base[:5]],
        "dedicacion_top5_new": [list(pair) for pair in dedication_new[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/upv_2025")
    parser.add_argument("--out-dir", default="outputs/external_binary_eval/upv_2025_mi_sensitivity_20260927")
    parser.add_argument("--sizes", nargs="+", default=["10000", "100000", "full_train"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.sizes or args.sizes[0] != "10000":
        parser.error("The first size must be 10000 to validate the saved baseline")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw, schema, _ = load_upv(args.data_dir, auto_download=False)
    partition, _ = group_split(raw, RANDOM_SEED)
    train = raw.loc[partition == "train"].copy()
    train["student_id"] = train[STUDENT_ID].astype(str)
    if len(train) != 325483:
        raise ValueError("Unexpected train size: %d" % len(train))
    space = EncodedSpace(train, schema)
    numeric = actionable_features(schema["feature_names"], next(r for r in REGIMES if r.name == "MODERATE"))
    names = [name for name in schema["feature_names"] if name in set(numeric) or name == "dedicacion"]
    if len(names) != 29:
        raise ValueError("Expected 29 mutable features, got %d" % len(names))
    saved = json.loads(Path("outputs/external_binary_eval/upv_2025_author_alignment_v1/mi_pair_sets.json").read_text())
    results = {}
    rows = []
    summaries = []
    for label in args.sizes:
        reference = train if label == "full_train" else _student_reference_subset(train, int(label), RANDOM_SEED)
        if label == "10000":
            complete = space.encode(reference)
            digest = hashlib.sha256(complete.to_csv(index=False).replace("\r\n", "\n").encode()).hexdigest()
            if digest != saved["mi_reference_sha256"]:
                raise ValueError("10k encoded reference differs from saved MI artifact")
            encoded = complete.loc[:, names]
        else:
            encoded = space.encode(reference).loc[:, names]
        scores, ranked, metrics = _rank(encoded, names, args.workers)
        if metrics["failures"]:
            raise RuntimeError("Estimator failures: %s" % metrics["failures"][:3])
        if label == "10000":
            saved_rank = pd.read_csv("outputs/external_binary_eval/upv_2025_author_alignment_v1/mi_pair_rankings.csv")
            saved_rank = saved_rank.loc[saved_rank.mi_ranking == "DIGITAL_PLUS_CATEGORY_SOURCE"]
            saved_pairs = list(zip(saved_rank.left_feature, saved_rank.right_feature))
            if ranked != saved_pairs or not np.allclose(
                [scores[pair] for pair in ranked], saved_rank.score.to_numpy(), rtol=0, atol=1e-12
            ):
                raise ValueError("10k MI ranking differs from the saved artifact")
        results[label] = (scores, ranked)
        for rank, pair in enumerate(ranked, start=1):
            rows.append({"sample": label, "n_rows": len(reference), "rank": rank,
                         "score": scores[pair], "left_feature": pair[0], "right_feature": pair[1]})
        summary = {"sample": label, **metrics}
        if label != args.sizes[0]:
            summary["versus_10k"] = _compare(results[args.sizes[0]], results[label], numeric)
        summaries.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "failures"}), flush=True)
        write_json(out / "summary.json", {"settings": vars(args), "sklearn_version": sklearn.__version__, "runs": summaries})
        pd.DataFrame(rows).to_csv(out / "rankings.csv", index=False)


if __name__ == "__main__":
    main()
