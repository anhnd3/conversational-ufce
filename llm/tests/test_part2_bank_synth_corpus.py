from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm.src.part2_eval.bank_synth_corpus import (
    COMMON_CORE_RANGES,
    NUMERIC_FIELDS,
    TARGET_MAIN_PROFILE_COUNT,
    TARGET_SOURCE_PROFILE_COUNT,
    TARGET_SYNTH_PROFILE_COUNT,
    TEMPLATE_FIELDS,
    build_synth_group_profile_map,
    generate_bank_boundary_profiles_corpus,
    generate_bank_synth_profile_pool,
    load_bank_source_analysis,
    select_prioritized_rare_seed_rows,
)
from llm.src.part2_eval.corpora import (
    BANK_BOUNDARY_PROFILES_CORPUS_PATH,
    G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH,
    TIER_B_SYNTH300_CORPUS_PATH,
    TIER_B_SYNTH300_CORPUS_VERSION,
    build_active_constraint_pattern,
    generate_bank_boundary_profiles_snapshot,
    generate_g5_agent_portability_synth300_corpus,
    generate_tier_b_bank_synth300_corpus,
    load_bank_boundary_profiles_corpus,
    load_g5_agent_portability_corpus,
    load_g5_agent_portability_synth300_corpus,
    load_tier_b_bank_corpus,
    load_tier_b_bank_synth300_corpus,
)
from ufce.core.data_processing import get_bank_user_constraints
from ufce.model_bundles import load_dataset_model_bundle


ROOT = Path(__file__).resolve().parents[2]


def load_real_bank_bundle():
    bundle = load_dataset_model_bundle(
        "bank",
        ROOT / "ufce" / "data" / "bank.csv",
        ROOT / "llm" / "models",
    )
    (
        _features,
        _categorical_features,
        _numeric_features,
        _uf,
        _f2change,
        _label_col,
        desired_outcome,
        _nbr_features,
        _protected_features,
        _data_lab0,
        _data_lab1,
    ) = get_bank_user_constraints(bundle.dataset_df.copy())
    return bundle, desired_outcome


def profile_signature(profile: dict) -> str:
    return json.dumps(profile, ensure_ascii=True, sort_keys=True)


def test_load_bank_source_analysis_matches_bank_totest_baseline():
    analysis = load_bank_source_analysis()

    template_counts = {
        tuple(item["template_key"]): int(item["count"])
        for item in analysis["templates"]
    }

    assert analysis["merged_row_count"] == 250
    assert analysis["unique_row_count"] == TARGET_SOURCE_PROFILE_COUNT
    assert analysis["duplicate_row_count"] == 1
    assert analysis["common_core_ranges"] == COMMON_CORE_RANGES
    assert analysis["full_ranges"] == {
        "Income": {"min": 10.0, "max": 198.0},
        "CCAvg": {"min": 0.0, "max": 8.0},
        "Mortgage": {"min": 0.0, "max": 509.0},
    }
    assert template_counts[(1, 1, 0, 0, 0, 0)] == 18
    assert template_counts[(2, 1, 0, 0, 1, 0)] == 16
    assert template_counts[(4, 2, 0, 0, 1, 0)] == 11
    assert template_counts[(1, 3, 0, 0, 1, 0)] == 11
    assert template_counts[(3, 3, 0, 0, 1, 0)] == 10


def test_select_prioritized_rare_seed_rows_prioritizes_cd_then_securities():
    prioritized = select_prioritized_rare_seed_rows(load_bank_source_analysis())

    assert [row["profile"]["CDAccount"] for row in prioritized[:4]] == [1, 1, 1, 1]
    assert prioritized[4]["profile"]["CDAccount"] == 0
    assert prioritized[4]["profile"]["SecuritiesAccount"] == 1
    assert prioritized[5]["template_count"] <= prioritized[-1]["template_count"]


def test_generate_tier_b_bank_synth300_corpus_matches_locked_snapshot():
    bundle, desired_outcome = load_real_bank_bundle()
    profile_pool = generate_bank_synth_profile_pool(bundle=bundle, desired_outcome=desired_outcome)
    grouped_profiles = build_synth_group_profile_map(profile_pool)
    payload = generate_tier_b_bank_synth300_corpus(bundle=bundle, desired_outcome=desired_outcome)
    loaded = load_tier_b_bank_synth300_corpus()
    generic_loaded = load_tier_b_bank_corpus(TIER_B_SYNTH300_CORPUS_PATH)

    template_map = profile_pool["source_analysis"]["template_map"]
    case_profiles = [case["seed_profile"] for case in payload["cases"]]

    assert payload == loaded
    assert payload == generic_loaded
    assert payload["corpus_version"] == TIER_B_SYNTH300_CORPUS_VERSION
    assert payload["case_count"] == TARGET_MAIN_PROFILE_COUNT
    assert payload["group_counts"] == {"G1": 120, "G2": 120, "REFINEMENT": 60}
    assert Counter(case["session_shape"] for case in payload["cases"]) == {
        "clarification_followup": 48,
        "single_turn": 192,
        "refinement": 60,
    }
    assert len({profile_signature(profile) for profile in case_profiles}) == TARGET_MAIN_PROFILE_COUNT
    assert all(case["active_constraint_spec_expected"] is None for case in payload["cases"] if case["group"] == "G1")
    assert all(case["active_constraint_spec_expected"] is not None for case in payload["cases"] if case["group"] == "G2")
    assert {key: len(value) for key, value in grouped_profiles.items()} == {"G1": 120, "G2": 120, "REFINEMENT": 60}
    assert len(profile_pool["source_profiles"]) == TARGET_SOURCE_PROFILE_COUNT
    assert len(profile_pool["synth_profiles"]) == TARGET_SYNTH_PROFILE_COUNT
    assert sum(profile_pool["stage_counts"].values()) == TARGET_SYNTH_PROFILE_COUNT
    assert profile_pool["stage_counts"]["rare_template_mutation"] == 15
    assert profile_pool["stage_counts"]["common_template_expansion"] >= 35

    g1_clarification_cases = [
        case for case in payload["cases"] if case["group"] == "G1" and case["session_shape"] == "clarification_followup"
    ]
    g2_clarification_cases = [
        case for case in payload["cases"] if case["group"] == "G2" and case["session_shape"] == "clarification_followup"
    ]
    assert len(g1_clarification_cases) == 24
    assert len(g2_clarification_cases) == 24
    for case in g1_clarification_cases:
        assert len(case["turns"]) == 2
        assert "Income " in case["turns"][1]
        assert "CCAvg " in case["turns"][1]
        assert "Mortgage " in case["turns"][1]
        assert "SecuritiesAccount " in case["turns"][1]
        assert "CreditCard " in case["turns"][1]
    for case_index, case in enumerate(g2_clarification_cases, start=1):
        expected_sentence, _expected_spec = build_active_constraint_pattern(case_index, case["seed_profile"])
        assert len(case["turns"]) == 2
        assert expected_sentence in case["turns"][0]
        assert expected_sentence in case["turns"][1]
        assert "Income " in case["turns"][1]
        assert "CCAvg " in case["turns"][1]
        assert "Mortgage " in case["turns"][1]
        assert "SecuritiesAccount " in case["turns"][1]
        assert "CreditCard " in case["turns"][1]

    for profile in case_profiles:
        frame = __import__("pandas").DataFrame([profile], columns=list(bundle.feature_order))
        prediction = int(bundle.lr.predict(frame)[0])
        assert prediction != int(desired_outcome)

    for record in profile_pool["synth_records"]:
        template_key = tuple(int(record["source_template"][field_name]) for field_name in TEMPLATE_FIELDS)
        numeric_stats = template_map[template_key]["numeric_stats"]
        for field_name in NUMERIC_FIELDS:
            assert float(numeric_stats[field_name]["min"]) <= float(record["profile"][field_name]) <= float(
                numeric_stats[field_name]["max"]
            )
        if int(numeric_stats["Mortgage"]["nonzero_count"]) == 0:
            assert float(record["profile"]["Mortgage"]) == 0.0


def test_generate_g5_agent_portability_synth300_matches_locked_snapshot():
    bundle, desired_outcome = load_real_bank_bundle()
    tier_b_payload = generate_tier_b_bank_synth300_corpus(bundle=bundle, desired_outcome=desired_outcome)
    payload = generate_g5_agent_portability_synth300_corpus(tier_b_corpus=tier_b_payload)
    loaded = load_g5_agent_portability_synth300_corpus()
    generic_loaded = load_g5_agent_portability_corpus(G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH)

    assert payload == loaded
    assert payload == generic_loaded
    assert payload["case_count"] == 100
    assert payload["selection_counts"] == {
        "clarification_followup": 40,
        "g1_single_turn": 30,
        "g2_single_turn": 30,
        "refinement_excluded": True,
    }
    assert payload["source_corpus_version"] == tier_b_payload["corpus_version"]
    assert payload["source_corpus_sha256"] == tier_b_payload["corpus_sha256"]
    assert Counter(case["session_shape"] for case in payload["cases"]) == {
        "clarification_followup": 40,
        "single_turn": 60,
    }
    assert Counter(case["group"] for case in payload["cases"]) == {"G1": 54, "G2": 46}


def test_generate_bank_boundary_profiles_snapshot_matches_locked_snapshot():
    analysis = load_bank_source_analysis()
    payload = generate_bank_boundary_profiles_snapshot()
    direct_payload = generate_bank_boundary_profiles_corpus()
    loaded = load_bank_boundary_profiles_corpus()
    template_map = analysis["template_map"]
    full_ranges = analysis["full_ranges"]

    assert payload == direct_payload
    assert payload == loaded
    assert payload["case_count"] == 60
    assert payload["bucket_counts"] == {
        "core_boundary": 20,
        "full_boundary": 20,
        "rare_adversarial": 10,
        "negative_out_of_range": 10,
    }

    for case in payload["cases"]:
        profile = case["seed_profile"]
        template_key = tuple(int(case["source_template"][field_name]) for field_name in TEMPLATE_FIELDS)
        numeric_stats = template_map[template_key]["numeric_stats"]
        if case["bucket"] == "core_boundary":
            for field_name in NUMERIC_FIELDS:
                assert COMMON_CORE_RANGES[field_name]["min"] <= float(profile[field_name]) <= COMMON_CORE_RANGES[field_name]["max"]
                assert float(numeric_stats[field_name]["min"]) <= float(profile[field_name]) <= float(numeric_stats[field_name]["max"])
            assert case["expected_validity"] == "valid"
        elif case["bucket"] in {"full_boundary", "rare_adversarial"}:
            for field_name in NUMERIC_FIELDS:
                assert full_ranges[field_name]["min"] <= float(profile[field_name]) <= full_ranges[field_name]["max"]
                assert float(numeric_stats[field_name]["min"]) <= float(profile[field_name]) <= float(numeric_stats[field_name]["max"])
            assert case["expected_validity"] == "valid"
        else:
            outside_fields = [
                field_name
                for field_name in NUMERIC_FIELDS
                if not (full_ranges[field_name]["min"] <= float(profile[field_name]) <= full_ranges[field_name]["max"])
            ]
            assert case["expected_validity"] == "reject"
            assert len(outside_fields) == 1


def test_synth_snapshot_files_exist():
    assert TIER_B_SYNTH300_CORPUS_PATH.exists()
    assert G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH.exists()
    assert BANK_BOUNDARY_PROFILES_CORPUS_PATH.exists()
