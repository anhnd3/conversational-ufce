from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = ROOT / "scripts" / "final" / "part1" / "06_public_forceflip_newbest.py"
    spec = importlib.util.spec_from_file_location("public_forceflip_newbest_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metric_direction_contract_is_complete() -> None:
    runner = load_runner()

    assert runner.METRIC_DIRECTIONS == {
        "Prox-Jac": "lower_is_better",
        "Prox-Euc": "lower_is_better",
        "Sparsity": "lower_is_better",
        "Actionability": "higher_is_better",
        "Plausibility": "higher_is_better",
        "Feasibility": "higher_is_better",
    }


def test_directional_delta_respects_metric_direction() -> None:
    runner = load_runner()

    assert runner.directional_delta(10, 7, "lower_is_better") == 3
    assert runner.directional_delta(10, 7, "higher_is_better") == -3
    assert runner.comparison_verdict(3, positive="improved", negative="regressed") == "improved"
    assert runner.comparison_verdict(-3, positive="improved", negative="regressed") == "regressed"
    assert runner.comparison_verdict(0, positive="improved", negative="regressed") == "same"
    assert runner.comparison_verdict(None, positive="improved", negative="regressed") == "na"


def test_final_comparison_uses_directional_verdicts() -> None:
    runner = load_runner()
    datasets = ["bank"]

    def rows(source, values):
        return pd.DataFrame(
            [
                {"dataset": "bank", "method": "UFCE1", "metric": metric, "value": value, "source": source}
                for metric, value in values.items()
            ]
        )

    published = rows(
        "published",
        {
            "Prox-Jac": 0.5,
            "Prox-Euc": 10,
            "Sparsity": 2,
            "Actionability": 20,
            "Plausibility": 20,
            "Feasibility": 20,
        },
    )
    public_raw = published.copy()
    public_forceflip = rows(
        "public_forceflip",
        {
            "Prox-Jac": 0.4,
            "Prox-Euc": 12,
            "Sparsity": 2,
            "Actionability": 10,
            "Plausibility": 8,
            "Feasibility": 6,
        },
    )
    new_best = rows(
        "new_best",
        {
            "Prox-Jac": 0.3,
            "Prox-Euc": 8,
            "Sparsity": 3,
            "Actionability": 15,
            "Plausibility": 9,
            "Feasibility": 5,
        },
    )

    df = runner.build_final_comparison(
        published_df=published,
        public_raw_df=public_raw,
        public_forceflip_df=public_forceflip,
        new_best_df=new_best,
        datasets=datasets,
    )

    by_metric = {row["metric"]: row for row in df[df["method"] == "UFCE1"].to_dict("records")}

    assert by_metric["Prox-Jac"]["forceflip_vs_published_verdict"] == "better"
    assert by_metric["Prox-Euc"]["forceflip_vs_published_verdict"] == "worse"
    assert by_metric["Sparsity"]["forceflip_vs_published_verdict"] == "same"
    assert by_metric["Actionability"]["forceflip_vs_published_verdict"] == "worse"
    assert by_metric["Plausibility"]["forceflip_vs_published_verdict"] == "worse"
    assert by_metric["Feasibility"]["forceflip_vs_published_verdict"] == "worse"

    assert by_metric["Prox-Jac"]["new_best_vs_forceflip_verdict"] == "improved"
    assert by_metric["Prox-Euc"]["new_best_vs_forceflip_verdict"] == "improved"
    assert by_metric["Sparsity"]["new_best_vs_forceflip_verdict"] == "regressed"
    assert by_metric["Actionability"]["new_best_vs_forceflip_verdict"] == "improved"
    assert by_metric["Plausibility"]["new_best_vs_forceflip_verdict"] == "improved"
    assert by_metric["Feasibility"]["new_best_vs_forceflip_verdict"] == "regressed"
