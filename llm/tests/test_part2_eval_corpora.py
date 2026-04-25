from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from llm.src.part2_eval.corpora import (
    BANK_BOUNDARY_PROFILES_CORPUS_PATH,
    G5_AGENT_PORTABILITY_CORPUS_PATH,
    G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH,
    TIER_A_CORPUS_PATH,
    TIER_A_ANNOTATION_SCHEMA_VERSION,
    TIER_A_SCORER_OUTPUT_SCHEMA_VERSION,
    TIER_B_CORPUS_PATH,
    TIER_B_SYNTH300_CORPUS_PATH,
    TIER_C_CORPUS_PATH,
    TIER_C_CORPUS_V1_PATH,
    TIER_D_CORPUS_PATH,
    generate_tier_a_annotation_corpus,
    generate_tier_b_bank_corpus,
    generate_tier_c_bank_backend_corpus,
    generate_tier_d_bank_replay_corpus,
    generate_g5_agent_portability_corpus,
    load_frozen_corpus,
    load_g5_agent_portability_corpus,
    load_tier_a_annotation_corpus,
    load_tier_b_bank_corpus,
    load_tier_c_bank_backend_corpus,
    load_tier_d_bank_replay_corpus,
)
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry


def build_fake_bank_bundle(row_count: int = 260):
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "Income": 40.0 + index,
                "Family": 1 + (index % 4),
                "CCAvg": 1.0 + (index % 5) * 0.5,
                "Education": 1 + (index % 3),
                "Mortgage": 20.0 + index,
                "SecuritiesAccount": index % 2,
                "CDAccount": (index + 1) % 2,
                "Online": index % 2,
                "CreditCard": (index // 2) % 2,
                "Personal Loan": 0,
            }
        )
    return SimpleNamespace(
        dataset_df=pd.DataFrame(rows),
        label_col="Personal Loan",
    )


def load_real_bank_bundle():
    model_registry = ModelRegistry()
    policy_registry = PolicyRegistry(model_registry)
    return model_registry.get_bundle("bank"), policy_registry.get_policy("bank").desired_outcome


def test_load_tier_a_annotation_corpus_freezes_versions_and_case_split():
    payload = load_tier_a_annotation_corpus()

    assert payload["annotation_schema_version"] == TIER_A_ANNOTATION_SCHEMA_VERSION
    assert payload["scorer_output_schema_version"] == TIER_A_SCORER_OUTPUT_SCHEMA_VERSION
    assert payload["case_count"] == 50
    assert len(payload["cases"]) == 50
    assert len([case for case in payload["cases"] if case["annotation_type"] == "initial_constraint_spec"]) == 25
    assert len([case for case in payload["cases"] if case["annotation_type"] == "refinement_delta"]) == 25
    assert payload["corpus_sha256"]


def test_load_tier_b_bank_corpus_freezes_group_mix():
    payload = load_tier_b_bank_corpus()

    assert payload["case_count"] == 250
    assert payload["group_counts"] == {"G1": 100, "G2": 100, "REFINEMENT": 50}
    assert len([case for case in payload["cases"] if case["group"] == "G1"]) == 100
    assert len([case for case in payload["cases"] if case["group"] == "G2"]) == 100
    assert len([case for case in payload["cases"] if case["group"] == "REFINEMENT"]) == 50
    assert payload["corpus_sha256"]


def test_load_tier_c_bank_backend_corpus_freezes_seed_counts():
    payload = load_tier_c_bank_backend_corpus()

    assert payload["seed_count"] == 200
    assert payload["constrained_subset_count"] == 50
    assert payload["selection_rule"] == "predictor_rejected_factual_seeds"
    assert payload["selection_space"] == "raw_feature_space"
    assert len(payload["seeds"]) == 200
    assert len(payload["constrained_subset"]) == 50
    assert payload["corpus_sha256"]


def test_load_tier_c_bank_backend_corpus_uses_predictor_rejected_rows():
    payload = load_tier_c_bank_backend_corpus()
    bundle, _desired_outcome = load_real_bank_bundle()
    profiles = pd.DataFrame([seed["profile"] for seed in payload["seeds"]], columns=bundle.feature_order)
    predictions = bundle.lr.predict(profiles)

    assert set(int(item) for item in predictions.tolist()) == {int(payload["rejected_label"])}


def test_load_tier_d_bank_replay_corpus_freezes_replay_counts():
    payload = load_tier_d_bank_replay_corpus()

    assert payload["replay_request_count"] == 1000
    assert payload["source_request_count"] == 290
    assert payload["source_case_count"] == 250
    assert len(payload["replay_requests"]) == 1000
    assert payload["replay_requests"][0]["replay_id"] == "TIERD-0001"
    assert payload["corpus_sha256"]


def test_load_g5_agent_portability_corpus_freezes_selection_mix():
    payload = load_g5_agent_portability_corpus()

    assert payload["case_count"] == 100
    assert payload["selection_counts"] == {
        "clarification_followup": 40,
        "g1_single_turn": 30,
        "g2_single_turn": 30,
        "refinement_excluded": True,
    }
    assert len([case for case in payload["cases"] if case["session_shape"] == "clarification_followup"]) == 40
    assert len([case for case in payload["cases"] if case["group"] == "G1"]) == 50
    assert len([case for case in payload["cases"] if case["group"] == "G2"]) == 50
    assert len([case for case in payload["cases"] if case["group"] == "REFINEMENT"]) == 0
    assert payload["corpus_sha256"]


def test_generated_corpora_match_repo_tracked_snapshots():
    bundle, desired_outcome = load_real_bank_bundle()

    assert generate_tier_a_annotation_corpus() == load_tier_a_annotation_corpus()
    assert generate_tier_b_bank_corpus(bundle=bundle, desired_outcome=desired_outcome) == load_tier_b_bank_corpus()
    assert generate_tier_c_bank_backend_corpus(bundle=bundle, desired_outcome=desired_outcome) == load_tier_c_bank_backend_corpus()
    assert generate_tier_d_bank_replay_corpus(bundle=bundle, desired_outcome=desired_outcome) == load_tier_d_bank_replay_corpus()
    assert generate_g5_agent_portability_corpus() == load_g5_agent_portability_corpus()


def test_load_frozen_corpus_fails_for_missing_or_malformed_files(tmp_path):
    missing_path = tmp_path / "missing.json"

    try:
        load_frozen_corpus(missing_path, expected_version="demo")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError for missing corpus")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{invalid", encoding="utf-8")
    try:
        load_frozen_corpus(invalid_path, expected_version="demo")
    except ValueError as exc:
        assert "not valid JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for malformed corpus JSON")


def test_snapshot_files_exist():
    assert TIER_A_CORPUS_PATH.exists()
    assert TIER_B_CORPUS_PATH.exists()
    assert TIER_B_SYNTH300_CORPUS_PATH.exists()
    assert TIER_C_CORPUS_PATH.exists()
    assert TIER_C_CORPUS_V1_PATH.exists()
    assert TIER_D_CORPUS_PATH.exists()
    assert G5_AGENT_PORTABILITY_CORPUS_PATH.exists()
    assert G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH.exists()
    assert BANK_BOUNDARY_PROFILES_CORPUS_PATH.exists()
