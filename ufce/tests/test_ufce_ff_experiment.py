from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    path = ROOT / "scripts" / "final" / "part1" / "07_ufce_ff_experiment.py"
    spec = importlib.util.spec_from_file_location("ufce_ff_experiment_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_and_mode_parsing() -> None:
    runner = load_runner()
    args = runner.build_arg_parser().parse_args([])

    assert runner.METHODS == ["UFCE1", "UFCE2", "UFCE3"]
    assert args.dataset == "all"
    assert args.mode == "all"
    assert args.mi_k_list == "5,10,20,all"
    assert args.config_profile == "public_github_source"
    assert args.bundle_mode == "table7_author_public"
    assert runner.mode_list_from_arg("all") == [
        "public_posthoc",
        "public_forceflip",
        "pool_expansion_forceflip",
    ]
    assert runner.mode_list_from_arg("final_comparison") == [
        "public_posthoc",
        "public_forceflip",
    ]


def test_classify_failure_priority() -> None:
    runner = load_runner()
    empty = pd.DataFrame()

    code, _detail = runner.classify_failure(
        raw_count=0,
        flip_count=0,
        public_selected_exists=False,
        public_selected_flip=False,
        flip_rank_rows=empty,
    )
    assert code == "F0"

    code, detail = runner.classify_failure(
        raw_count=3,
        flip_count=0,
        public_selected_exists=True,
        public_selected_flip=False,
        flip_rank_rows=empty,
    )
    assert code == "F1"
    assert detail == "F5_not_observable_label_only"

    code, _detail = runner.classify_failure(
        raw_count=3,
        flip_count=2,
        public_selected_exists=True,
        public_selected_flip=False,
        flip_rank_rows=empty,
    )
    assert code == "F2"


def test_summary_uses_public_selected_flip_for_forceflip() -> None:
    runner = load_runner()
    rows = []
    for selected_flip in [1, 0, 1]:
        rows.append(
            {
                "dataset": "bank",
                "method": "UFCE2",
                "mode": "public_forceflip",
                "config_profile": "public_github_source",
                "bundle_mode": "table7_author_public",
                "mi_k": "5",
                "top_n": 0,
                "effective_radius": 500,
                "effective_n_neighbors": 100,
                "effective_min_act": 3,
                "effective_min_feas": 2,
                "effective_force_flip": 1,
                "fold_id": "fold0.csv",
                "raw_candidate_count": 2,
                "unique_candidate_count": 2,
                "flip_candidate_count": 1,
                "public_selected_flip": selected_flip,
                "verified_selected_flip": selected_flip,
                "strict_valid_selected": selected_flip,
                "failure_code": "OK" if selected_flip else "F1",
                "selected_prox_euc": 1.0 if selected_flip else None,
                "selected_sparsity": 1.0 if selected_flip else None,
                "selected_apf_score": 3.0 if selected_flip else None,
                "selected_actionability_pass": 1.0 if selected_flip else None,
                "selected_plausibility_pass": 1.0 if selected_flip else None,
                "selected_feasibility_pass": 1.0 if selected_flip else None,
                "predict_call_count": 1,
                "batch_predict_row_count": 2,
                "batch_predict_ms": 1.0,
                "runtime_ms": 10.0,
            }
        )

    summary = runner.summarize_queries(pd.DataFrame(rows))
    assert len(summary) == 1
    assert abs(float(summary.iloc[0]["strict_valid_rate"]) - (2 / 3)) < 1e-12


def test_build_stage_comparison_table_merges_posthoc_forceflip() -> None:
    runner = load_runner()
    base = {
        "dataset": "bank",
        "method": "UFCE2",
        "config_profile": "public_github_source",
        "bundle_mode": "table7_author_public",
        "mi_k": "5",
        "top_n": 0,
        "effective_radius": 500,
        "effective_n_neighbors": 100,
        "effective_min_act": 3,
        "effective_min_feas": 2,
        "query_count": 10,
        "raw_candidate_count": 10,
        "unique_candidate_count": 10,
        "flip_candidate_count": 5,
        "public_selected_flip_count": 0,
        "verified_selected_flip_count": 0,
        "main_failure_reduced": "F2",
        "mean_prox_euc_valid": 1.0,
        "mean_sparsity_valid": 1.0,
        "mean_apf_valid": 2.0,
        "predict_call_count": 10,
        "batch_predict_row_count": 10,
        "mean_batch_predict_ms": 1.0,
        "runtime_ratio": None,
    }
    rows = [
        {**base, "mode": "public_posthoc", "effective_force_flip": 0, "strict_valid_rate": 0.2, "mean_runtime_ms": 100.0},
        {**base, "mode": "public_forceflip", "effective_force_flip": 1, "strict_valid_rate": 0.5, "mean_runtime_ms": 110.0},
    ]

    stage = runner.build_stage_comparison_table(pd.DataFrame(rows))
    assert len(stage) == 1
    row = stage.iloc[0]
    assert abs(float(row["forceflip_vs_posthoc_delta"]) - 0.3) < 1e-12
    assert row["forceflip_vs_posthoc_verdict"] == "improved"
