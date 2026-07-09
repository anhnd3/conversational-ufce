from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = ROOT / "scripts" / "final" / "part1" / "table7_full_reproduction.py"
    spec = importlib.util.spec_from_file_location("table7_full_final_contract_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_final_contract() -> None:
    runner = load_runner()

    args = runner.build_arg_parser().parse_args(["--dataset", "bank", "--methods", "dice,ar"])

    assert args.runtime_profile == "final_freeze"
    assert args.bundle_mode == "table7_author_public"
    assert args.radius is None
    assert args.n_neighbors is None
    assert args.min_act is None
    assert args.min_feas is None
    assert args.ufce_flip_filter is None


def test_shared_final_runtime_profile_is_used_for_bank_and_movie() -> None:
    runner = load_runner()

    bank_args = runner.build_arg_parser().parse_args(["--dataset", "bank"])
    movie_args = runner.build_arg_parser().parse_args(["--dataset", "movie"])

    bank_cfg = runner.final_cfg.resolve_effective_cfg("bank", bank_args)
    movie_cfg = runner.final_cfg.resolve_effective_cfg("movie", movie_args)

    assert {key: bank_cfg[key] for key in ["radius", "n_neighbors", "min_act", "min_feas", "ufce_flip_filter"]} == {
        "radius": 500,
        "n_neighbors": 1000,
        "min_act": 0,
        "min_feas": 0,
        "ufce_flip_filter": 0,
    }
    assert {key: movie_cfg[key] for key in ["radius", "n_neighbors", "min_act", "min_feas", "ufce_flip_filter"]} == {
        "radius": 80,
        "n_neighbors": 50,
        "min_act": 1,
        "min_feas": 1,
        "ufce_flip_filter": 0,
    }


def test_build_bank_table48_uses_forceflip_reference_rows(tmp_path: Path) -> None:
    runner = load_runner()

    reference_df = pd.DataFrame(
        [
            {
                "dataset": "bank",
                "method": "UFCE1",
                "metric_scope": "author_pool_forceflip",
                "Prox-Jac": 0.6660568086883876,
                "Prox-Euc": 0.017333333333333333,
                "Sparsity": 1.0,
                "Actionability": 16.0,
                "Plausibility": 12.0,
                "Feasibility": 12.0,
            },
            {
                "dataset": "bank",
                "method": "UFCE2",
                "metric_scope": "author_pool_forceflip",
                "Prox-Jac": 0.2687253487253487,
                "Prox-Euc": 20.376825275669194,
                "Sparsity": 1.7189321789321788,
                "Actionability": 8.8,
                "Plausibility": 8.8,
                "Feasibility": 8.8,
            },
            {
                "dataset": "bank",
                "method": "UFCE3",
                "metric_scope": "author_pool_forceflip",
                "Prox-Jac": 0.06190476190476191,
                "Prox-Euc": 28.26627136701392,
                "Sparsity": 1.5285714285714287,
                "Actionability": 5.4,
                "Plausibility": 5.4,
                "Feasibility": 5.4,
            },
        ]
    )
    reference_csv = tmp_path / "author_pool_metric_summary.csv"
    reference_df.to_csv(reference_csv, index=False)

    metric_rows = []
    dice_values = {
        "Prox-Jac": 0.69,
        "Prox-Euc": 127.8,
        "Sparsity": 5.92,
        "Actionability": 25.4,
        "Plausibility": 50.0,
        "Feasibility": 26.2,
    }
    ar_values = {
        "Prox-Jac": 0.66,
        "Prox-Euc": 131.29,
        "Sparsity": 5.46,
        "Actionability": 17.4,
        "Plausibility": 50.0,
        "Feasibility": 21.4,
    }
    for metric_name, reproduced_value in dice_values.items():
        metric_rows.append(
            {
                "dataset": "bank",
                "method": "DiCE",
                "metric_name": metric_name,
                "reproduced_value": reproduced_value,
            }
        )
    for metric_name, reproduced_value in ar_values.items():
        metric_rows.append(
            {
                "dataset": "bank",
                "method": "AR",
                "metric_name": metric_name,
                "reproduced_value": reproduced_value,
            }
        )

    table48_df = runner.build_bank_table48(
        str(reference_csv),
        "author_pool_forceflip",
        metric_rows,
    )

    assert table48_df is not None
    assert list(table48_df.columns) == ["Metric", "UFCE-FF1", "UFCE-FF2", "UFCE-FF3", "DiCE", "AR"]

    actionability_row = table48_df.loc[table48_df["Metric"] == "Actionability"].iloc[0]
    plausibility_row = table48_df.loc[table48_df["Metric"] == "Plausibility"].iloc[0]
    feasibility_row = table48_df.loc[table48_df["Metric"] == "Feasibility"].iloc[0]

    assert float(actionability_row["UFCE-FF1"]) == 16.0
    assert float(plausibility_row["UFCE-FF1"]) == 12.0
    assert float(feasibility_row["UFCE-FF1"]) == 12.0
    assert float(actionability_row["DiCE"]) == 25.4
    assert float(feasibility_row["AR"]) == 21.4
