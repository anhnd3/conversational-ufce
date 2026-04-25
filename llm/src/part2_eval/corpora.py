from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm.src.part2_eval.bank_synth_corpus import (
    build_synth_generation_metadata,
    build_synth_group_profile_map,
    generate_bank_boundary_profiles_corpus,
    generate_bank_synth_profile_pool,
)
from llm.src.part2_eval.common import sha256_json_payload
from llm.src.refinement.delta import build_active_constraint_spec
from llm.src.refinement.validation import validate_refinement_prediction
from llm.src.runtime.constraint_spec import validate_and_normalize_constraint_spec
from ufce.core.data_processing import get_bank_user_constraints


BANK_FEATURE_ORDER = [
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "SecuritiesAccount",
    "CDAccount",
    "Online",
    "CreditCard",
]
BANK_BOOLEAN_FIELDS = ("SecuritiesAccount", "CDAccount", "Online", "CreditCard")
ROOT = Path(__file__).resolve().parents[3]
TIER_A_CORPUS_VERSION = "part2_tier_a_bank_annotations_v1"
TIER_B_CORPUS_VERSION = "part2_tier_b_bank_sessions_v1"
TIER_B_SYNTH300_CORPUS_VERSION = "part2_tier_b_bank_sessions_v2_synth300"
TIER_C_CORPUS_V1_VERSION = "part2_tier_c_bank_backend_v1"
TIER_C_CORPUS_VERSION = "part2_tier_c_bank_backend_v2"
TIER_D_CORPUS_VERSION = "part2_tier_d_bank_replay_v1"
G5_AGENT_PORTABILITY_CORPUS_VERSION = "part2_g5_agent_portability_bank_v1"
G5_AGENT_PORTABILITY_SYNTH300_CORPUS_VERSION = "part2_g5_agent_portability_bank_v2_synth300"
BANK_BOUNDARY_PROFILES_CORPUS_VERSION = "part2_bank_boundary_profiles_v1"
TIER_A_ANNOTATION_SCHEMA_VERSION = "part2_tier_a_annotation_schema_v1"
TIER_A_SCORER_OUTPUT_SCHEMA_VERSION = "part2_tier_a_scorer_output_schema_v1"
TIER_A_ANNOTATION_SCHEMA_PATH = ROOT / "docs" / "validation" / "schemas" / "part2_tier_a_annotation_schema_v1.json"
TIER_A_SCORER_OUTPUT_SCHEMA_PATH = ROOT / "docs" / "validation" / "schemas" / "part2_tier_a_scorer_output_schema_v1.json"
VALIDATION_CORPORA_ROOT = ROOT / "docs" / "validation" / "corpora"
TIER_A_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_tier_a_bank_annotations_v1.json"
TIER_B_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_tier_b_bank_sessions_v1.json"
TIER_B_SYNTH300_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_tier_b_bank_sessions_v2_synth300.json"
TIER_C_CORPUS_V1_PATH = VALIDATION_CORPORA_ROOT / "part2_tier_c_bank_backend_v1.json"
TIER_C_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_tier_c_bank_backend_v2.json"
TIER_D_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_tier_d_bank_replay_v1.json"
G5_AGENT_PORTABILITY_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_g5_agent_portability_bank_v1.json"
G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_g5_agent_portability_bank_v2_synth300.json"
BANK_BOUNDARY_PROFILES_CORPUS_PATH = VALIDATION_CORPORA_ROOT / "part2_bank_boundary_profiles_v1.json"


def build_tier_a_annotation_corpus() -> dict[str, Any]:
    return load_tier_a_annotation_corpus()


def load_tier_a_annotation_corpus(path: Path | None = None) -> dict[str, Any]:
    return load_frozen_corpus(path or TIER_A_CORPUS_PATH, expected_version=TIER_A_CORPUS_VERSION)


def load_tier_b_bank_corpus(path: Path | None = None) -> dict[str, Any]:
    source_path = Path(path or TIER_B_CORPUS_PATH).resolve()
    return load_frozen_corpus(source_path, expected_version=infer_tier_b_expected_version(source_path))


def load_tier_b_bank_synth300_corpus(path: Path | None = None) -> dict[str, Any]:
    source_path = Path(path or TIER_B_SYNTH300_CORPUS_PATH).resolve()
    return load_frozen_corpus(source_path, expected_version=TIER_B_SYNTH300_CORPUS_VERSION)


def load_tier_c_bank_backend_corpus(path: Path | None = None) -> dict[str, Any]:
    source_path = Path(path or TIER_C_CORPUS_PATH).resolve()
    expected_version = infer_tier_c_expected_version(source_path)
    return load_frozen_corpus(source_path, expected_version=expected_version)


def load_tier_d_bank_replay_corpus(path: Path | None = None) -> dict[str, Any]:
    return load_frozen_corpus(path or TIER_D_CORPUS_PATH, expected_version=TIER_D_CORPUS_VERSION)


def load_g5_agent_portability_corpus(path: Path | None = None) -> dict[str, Any]:
    source_path = Path(path or G5_AGENT_PORTABILITY_CORPUS_PATH).resolve()
    return load_frozen_corpus(source_path, expected_version=infer_g5_expected_version(source_path))


def load_g5_agent_portability_synth300_corpus(path: Path | None = None) -> dict[str, Any]:
    source_path = Path(path or G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH).resolve()
    return load_frozen_corpus(source_path, expected_version=G5_AGENT_PORTABILITY_SYNTH300_CORPUS_VERSION)


def load_bank_boundary_profiles_corpus(path: Path | None = None) -> dict[str, Any]:
    source_path = Path(path or BANK_BOUNDARY_PROFILES_CORPUS_PATH).resolve()
    return load_frozen_corpus(source_path, expected_version=BANK_BOUNDARY_PROFILES_CORPUS_VERSION)


def build_tier_b_bank_corpus(*, bundle=None, desired_outcome: int | None = None) -> dict[str, Any]:
    del bundle
    del desired_outcome
    return load_tier_b_bank_corpus()


def build_tier_c_bank_backend_corpus(*, bundle=None, desired_outcome: int | None = None) -> dict[str, Any]:
    del bundle
    del desired_outcome
    return load_tier_c_bank_backend_corpus()


def build_tier_d_bank_replay_corpus(
    *,
    bundle=None,
    desired_outcome: int | None = None,
    replay_request_count: int = 1000,
) -> dict[str, Any]:
    del bundle
    del desired_outcome
    payload = load_tier_d_bank_replay_corpus()
    if int(payload["replay_request_count"]) != int(replay_request_count):
        raise ValueError(
            f"Frozen Tier D replay corpus count mismatch: expected {replay_request_count}, "
            f"snapshot contains {payload['replay_request_count']}"
        )
    return payload


def build_g5_agent_portability_corpus(*, bundle=None, desired_outcome: int | None = None) -> dict[str, Any]:
    del bundle
    del desired_outcome
    return load_g5_agent_portability_corpus()


def build_tier_b_bank_synth300_corpus(*, bundle=None, desired_outcome: int | None = None) -> dict[str, Any]:
    del bundle
    del desired_outcome
    return load_tier_b_bank_synth300_corpus()


def build_g5_agent_portability_synth300_corpus(*, bundle=None, desired_outcome: int | None = None) -> dict[str, Any]:
    del bundle
    del desired_outcome
    return load_g5_agent_portability_synth300_corpus()


def build_bank_boundary_profiles_corpus(*, bundle=None, desired_outcome: int | None = None) -> dict[str, Any]:
    del bundle
    del desired_outcome
    return load_bank_boundary_profiles_corpus()


def generate_tier_a_annotation_corpus() -> dict[str, Any]:
    profiles = [
        {
            "Income": 72,
            "Family": 1,
            "CCAvg": 4.8,
            "Education": 2,
            "Mortgage": 200,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 0,
            "CreditCard": 0,
        },
        {
            "Income": 68,
            "Family": 2,
            "CCAvg": 3.5,
            "Education": 1,
            "Mortgage": 90,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 1,
            "CreditCard": 1,
        },
        {
            "Income": 55,
            "Family": 4,
            "CCAvg": 2.1,
            "Education": 3,
            "Mortgage": 150,
            "SecuritiesAccount": 0,
            "CDAccount": 1,
            "Online": 1,
            "CreditCard": 0,
        },
        {
            "Income": 101,
            "Family": 1,
            "CCAvg": 5.2,
            "Education": 2,
            "Mortgage": 20,
            "SecuritiesAccount": 1,
            "CDAccount": 0,
            "Online": 1,
            "CreditCard": 1,
        },
        {
            "Income": 43,
            "Family": 3,
            "CCAvg": 1.3,
            "Education": 2,
            "Mortgage": 240,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 1,
        },
    ]
    initial_patterns = [
        ("Do not change Income.", {"disallowed_changes": ["Income"]}),
        ("Do not change Mortgage.", {"disallowed_changes": ["Mortgage"]}),
        ("Keep Mortgage at or below 120.", {"numeric_bounds": {"Mortgage": {"max": 120.0}}}),
        ("Change at most one thing.", {"max_changed_features": 1}),
        ("Prefer smaller edits.", {"prefer_fewer_changes": True}),
    ]
    refinement_patterns = [
        (
            "Do not change Income.",
            {},
            {"add_blocked_fields": ["Income"]},
        ),
        (
            "Income can change again.",
            {"disallowed_changes": ["Income"]},
            {"remove_blocked_fields": ["Income"]},
        ),
        (
            "Change at most one thing.",
            {},
            {"set_max_changed_features": 1},
        ),
        (
            "Prefer smaller edits.",
            {},
            {"set_prefer_fewer_changes": True},
        ),
        (
            "Mortgage must stay under 120.",
            {},
            {"set_numeric_bounds": {"Mortgage": {"max": 120.0}}},
        ),
    ]

    cases: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles, start=1):
        profile_prompt = render_bank_profile_prompt(profile)
        for pattern_index, (feedback_text, raw_spec) in enumerate(initial_patterns, start=1):
            expected_spec, errors = validate_and_normalize_constraint_spec(
                raw_spec,
                feature_order=BANK_FEATURE_ORDER,
            )
            if errors:
                raise ValueError(f"Invalid frozen initial constraint pattern: {errors}")
            cases.append(
                {
                    "case_id": f"TIERA-INIT-{profile_index:02d}-{pattern_index:02d}",
                    "annotation_type": "initial_constraint_spec",
                    "group": "M6",
                    "input_text": f"{profile_prompt} {feedback_text}",
                    "active_constraint_spec": None,
                    "pending_refinement_clarification": None,
                    "expected_constraint_spec": expected_spec,
                    "expected_delta": None,
                    "tags": ["bank", "initial_constraint_spec"],
                }
            )
        for pattern_index, (feedback_text, active_spec_raw, raw_delta) in enumerate(refinement_patterns, start=1):
            active_spec = build_active_constraint_spec(active_spec_raw, feature_order=BANK_FEATURE_ORDER)
            prediction = {
                "task": "extract_constraint_feedback",
                "status": "apply",
                "constraint_feedback_delta": raw_delta,
                "ambiguities": [],
                "unsupported_feedback": [],
                "notes": [],
            }
            validation = validate_refinement_prediction(prediction, feature_order=BANK_FEATURE_ORDER)
            if not validation.is_valid or validation.normalized_delta is None:
                raise ValueError(f"Invalid frozen refinement pattern: {validation.errors}")
            cases.append(
                {
                    "case_id": f"TIERA-REF-{profile_index:02d}-{pattern_index:02d}",
                    "annotation_type": "refinement_delta",
                    "group": "M36",
                    "input_text": feedback_text,
                    "active_constraint_spec": active_spec,
                    "pending_refinement_clarification": None,
                    "expected_constraint_spec": None,
                    "expected_delta": validation.normalized_delta,
                    "tags": ["bank", "refinement_delta"],
                }
            )
    payload = {
        "corpus_version": TIER_A_CORPUS_VERSION,
        "annotation_schema_version": TIER_A_ANNOTATION_SCHEMA_VERSION,
        "scorer_output_schema_version": TIER_A_SCORER_OUTPUT_SCHEMA_VERSION,
        "case_count": len(cases),
        "cases": cases,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_tier_b_bank_corpus(*, bundle, desired_outcome: int) -> dict[str, Any]:
    negative_rows = bundle.dataset_df.loc[bundle.dataset_df[bundle.label_col] != desired_outcome].reset_index(drop=True)
    required = 250
    if len(negative_rows) < required:
        raise ValueError(f"Bank corpus requires at least {required} negative rows, found {len(negative_rows)}")

    cases: list[dict[str, Any]] = []
    seed_index = 0

    for case_index in range(1, 101):
        row = negative_rows.iloc[seed_index]
        seed_index += 1
        profile = row_to_profile(row.to_dict())
        case_id = f"TIERB-G1-{case_index:03d}"
        if case_index <= 20:
            initial_prompt = render_bank_profile_prompt(profile, include_booleans=False)
            followup_prompt = render_bank_boolean_followup_prompt(profile)
            turns = [initial_prompt, followup_prompt]
            session_shape = "clarification_followup"
        else:
            turns = [render_bank_profile_prompt(profile)]
            session_shape = "single_turn"
        cases.append(
            {
                "case_id": case_id,
                "group": "G1",
                "dataset": "bank",
                "seed_profile": profile,
                "turns": turns,
                "session_shape": session_shape,
                "active_constraint_spec_expected": None,
            }
        )

    for case_index in range(1, 101):
        row = negative_rows.iloc[seed_index]
        seed_index += 1
        profile = row_to_profile(row.to_dict())
        constraint_sentence, expected_spec = build_active_constraint_pattern(case_index, profile)
        case_id = f"TIERB-G2-{case_index:03d}"
        if case_index <= 20:
            initial_prompt = render_bank_profile_prompt(
                profile,
                include_booleans=False,
                trailing_sentences=[constraint_sentence],
            )
            followup_prompt = render_bank_boolean_followup_prompt(profile)
            turns = [initial_prompt, followup_prompt]
            session_shape = "clarification_followup"
        else:
            turns = [render_bank_profile_prompt(profile, trailing_sentences=[constraint_sentence])]
            session_shape = "single_turn"
        cases.append(
            {
                "case_id": case_id,
                "group": "G2",
                "dataset": "bank",
                "seed_profile": profile,
                "turns": turns,
                "session_shape": session_shape,
                "active_constraint_spec_expected": expected_spec,
            }
        )

    for case_index in range(1, 51):
        row = negative_rows.iloc[seed_index]
        seed_index += 1
        profile = row_to_profile(row.to_dict())
        feedback_text = build_refinement_feedback_pattern(case_index, profile)
        cases.append(
            {
                "case_id": f"TIERB-REF-{case_index:03d}",
                "group": "REFINEMENT",
                "dataset": "bank",
                "seed_profile": profile,
                "turns": [render_bank_profile_prompt(profile)],
                "session_shape": "refinement",
                "active_constraint_spec_expected": None,
                "refinement_feedback": [feedback_text],
            }
        )

    payload = {
        "corpus_version": TIER_B_CORPUS_VERSION,
        "dataset": "bank",
        "case_count": len(cases),
        "group_counts": {
            "G1": 100,
            "G2": 100,
            "REFINEMENT": 50,
        },
        "cases": cases,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_tier_b_bank_synth300_corpus(*, bundle, desired_outcome: int) -> dict[str, Any]:
    profile_pool = generate_bank_synth_profile_pool(bundle=bundle, desired_outcome=desired_outcome)
    grouped_profiles = build_synth_group_profile_map(profile_pool)
    cases: list[dict[str, Any]] = []

    for case_index, profile in enumerate(grouped_profiles["G1"], start=1):
        case_id = f"TIERB2-G1-{case_index:03d}"
        if case_index <= 24:
            turns = [
                render_bank_profile_prompt(profile, include_booleans=False),
                render_bank_clarification_followup_prompt(profile),
            ]
            session_shape = "clarification_followup"
        else:
            turns = [render_bank_profile_prompt(profile)]
            session_shape = "single_turn"
        cases.append(
            {
                "case_id": case_id,
                "group": "G1",
                "dataset": "bank",
                "seed_profile": profile,
                "turns": turns,
                "session_shape": session_shape,
                "active_constraint_spec_expected": None,
            }
        )

    for case_index, profile in enumerate(grouped_profiles["G2"], start=1):
        constraint_sentence, expected_spec = build_active_constraint_pattern(case_index, profile)
        case_id = f"TIERB2-G2-{case_index:03d}"
        if case_index <= 24:
            turns = [
                render_bank_profile_prompt(
                    profile,
                    include_booleans=False,
                    trailing_sentences=[constraint_sentence],
                ),
                render_bank_clarification_followup_prompt(
                    profile,
                    trailing_sentences=[constraint_sentence],
                ),
            ]
            session_shape = "clarification_followup"
        else:
            turns = [render_bank_profile_prompt(profile, trailing_sentences=[constraint_sentence])]
            session_shape = "single_turn"
        cases.append(
            {
                "case_id": case_id,
                "group": "G2",
                "dataset": "bank",
                "seed_profile": profile,
                "turns": turns,
                "session_shape": session_shape,
                "active_constraint_spec_expected": expected_spec,
            }
        )

    for case_index, profile in enumerate(grouped_profiles["REFINEMENT"], start=1):
        feedback_text = build_refinement_feedback_pattern(case_index, profile)
        cases.append(
            {
                "case_id": f"TIERB2-REF-{case_index:03d}",
                "group": "REFINEMENT",
                "dataset": "bank",
                "seed_profile": profile,
                "turns": [render_bank_profile_prompt(profile)],
                "session_shape": "refinement",
                "active_constraint_spec_expected": None,
                "refinement_feedback": [feedback_text],
            }
        )

    payload = {
        "corpus_version": TIER_B_SYNTH300_CORPUS_VERSION,
        "dataset": "bank",
        "case_count": len(cases),
        "group_counts": {"G1": 120, "G2": 120, "REFINEMENT": 60},
        "generation_metadata": build_synth_generation_metadata(profile_pool),
        "cases": cases,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_tier_c_bank_backend_corpus(*, bundle, desired_outcome: int) -> dict[str, Any]:
    (
        features,
        _catf,
        _numf,
        _uf,
        _f2change,
        _outcome_label,
        helper_desired_outcome,
        _nbr_features,
        _protectf,
        data_lab0,
        _data_lab1,
    ) = get_bank_user_constraints(bundle.dataset_df.copy())
    feature_frame = data_lab0.loc[:, features].copy()
    predictions = np.asarray(bundle.lr.predict(feature_frame)).reshape(-1)
    unique_vals, counts = np.unique(predictions, return_counts=True)
    if len(unique_vals) == 0:
        raise ValueError("Bank backend corpus selection produced no predictions on data_lab0")
    rejected_label = int(unique_vals[np.argmax(counts)])
    rejected_rows = data_lab0.loc[predictions == rejected_label].reset_index(drop=True)
    if len(rejected_rows) < 200:
        raise ValueError(
            f"Bank backend corpus requires at least 200 predictor-rejected rows, found {len(rejected_rows)}"
        )

    seeds: list[dict[str, Any]] = []
    constrained_subset: list[dict[str, Any]] = []
    for index in range(200):
        row = rejected_rows.iloc[index]
        profile = row_to_profile(row.to_dict())
        seeds.append(
            {
                "seed_id": f"TIERC-{index + 1:03d}",
                "dataset": "bank",
                "profile": profile,
            }
        )
        if index < 50:
            _constraint_text, constraint_spec = build_active_constraint_pattern(index + 1, profile)
            constrained_subset.append(
                {
                    "seed_id": f"TIERC-C-{index + 1:03d}",
                    "dataset": "bank",
                    "profile": profile,
                    "constraint_spec": constraint_spec,
                }
            )

    payload = {
        "corpus_version": TIER_C_CORPUS_VERSION,
        "dataset": "bank",
        "selection_rule": "predictor_rejected_factual_seeds",
        "selection_space": "raw_feature_space",
        "desired_outcome": int(desired_outcome),
        "helper_desired_outcome": int(helper_desired_outcome),
        "rejected_label": rejected_label,
        "seed_count": len(seeds),
        "constrained_subset_count": len(constrained_subset),
        "seeds": seeds,
        "constrained_subset": constrained_subset,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_tier_d_bank_replay_corpus(
    *,
    bundle,
    desired_outcome: int,
    replay_request_count: int = 1000,
) -> dict[str, Any]:
    if replay_request_count <= 0:
        raise ValueError("replay_request_count must be positive")
    tier_b_corpus = generate_tier_b_bank_corpus(bundle=bundle, desired_outcome=desired_outcome)
    source_requests: list[dict[str, Any]] = []
    for case in tier_b_corpus["cases"]:
        for turn_index, turn_text in enumerate(case["turns"], start=1):
            source_requests.append(
                {
                    "source_case_id": case["case_id"],
                    "source_group": case["group"],
                    "source_session_shape": case["session_shape"],
                    "source_turn_index": turn_index,
                    "user_input": turn_text,
                }
            )
    if not source_requests:
        raise ValueError("Tier D replay corpus requires at least one source request")

    replay_requests: list[dict[str, Any]] = []
    for replay_index in range(replay_request_count):
        source = source_requests[replay_index % len(source_requests)]
        replay_requests.append(
            {
                "replay_id": f"TIERD-{replay_index + 1:04d}",
                "cycle_index": replay_index // len(source_requests),
                **source,
            }
        )

    payload = {
        "corpus_version": TIER_D_CORPUS_VERSION,
        "dataset": "bank",
        "replay_request_count": len(replay_requests),
        "source_request_count": len(source_requests),
        "source_case_count": len(tier_b_corpus["cases"]),
        "replay_requests": replay_requests,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_g5_agent_portability_corpus(*, tier_b_corpus: dict[str, Any] | None = None) -> dict[str, Any]:
    source = dict(tier_b_corpus or load_tier_b_bank_corpus())
    source_cases = [dict(item) for item in source.get("cases", []) if isinstance(item, dict)]
    clarification_cases = [
        dict(case)
        for case in source_cases
        if case.get("session_shape") == "clarification_followup"
    ]
    g1_single_turn = [
        dict(case)
        for case in source_cases
        if case.get("group") == "G1" and case.get("session_shape") == "single_turn"
    ]
    g2_single_turn = [
        dict(case)
        for case in source_cases
        if case.get("group") == "G2" and case.get("session_shape") == "single_turn"
    ]
    if len(clarification_cases) < 40:
        raise ValueError(f"G5 portability corpus requires 40 clarification cases, found {len(clarification_cases)}")
    if len(g1_single_turn) < 30:
        raise ValueError(f"G5 portability corpus requires 30 G1 single-turn cases, found {len(g1_single_turn)}")
    if len(g2_single_turn) < 30:
        raise ValueError(f"G5 portability corpus requires 30 G2 single-turn cases, found {len(g2_single_turn)}")
    selected_cases = clarification_cases[:40] + g1_single_turn[:30] + g2_single_turn[:30]
    payload = {
        "corpus_version": G5_AGENT_PORTABILITY_CORPUS_VERSION,
        "dataset": "bank",
        "source_corpus_version": source.get("corpus_version"),
        "source_corpus_sha256": source.get("corpus_sha256"),
        "selection_rule": (
            "all_clarification_followup_plus_first_30_G1_single_turn_plus_first_30_G2_single_turn_from_tier_b"
        ),
        "case_count": len(selected_cases),
        "selection_counts": {
            "clarification_followup": 40,
            "g1_single_turn": 30,
            "g2_single_turn": 30,
            "refinement_excluded": True,
        },
        "cases": selected_cases,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_g5_agent_portability_synth300_corpus(*, tier_b_corpus: dict[str, Any] | None = None) -> dict[str, Any]:
    source = dict(tier_b_corpus or load_tier_b_bank_synth300_corpus())
    source_cases = [dict(item) for item in source.get("cases", []) if isinstance(item, dict)]
    clarification_cases = [
        dict(case)
        for case in source_cases
        if case.get("session_shape") == "clarification_followup"
    ]
    g1_single_turn = [
        dict(case)
        for case in source_cases
        if case.get("group") == "G1" and case.get("session_shape") == "single_turn"
    ]
    g2_single_turn = [
        dict(case)
        for case in source_cases
        if case.get("group") == "G2" and case.get("session_shape") == "single_turn"
    ]
    if len(clarification_cases) < 40:
        raise ValueError(f"G5 portability synth300 corpus requires 40 clarification cases, found {len(clarification_cases)}")
    if len(g1_single_turn) < 30:
        raise ValueError(f"G5 portability synth300 corpus requires 30 G1 single-turn cases, found {len(g1_single_turn)}")
    if len(g2_single_turn) < 30:
        raise ValueError(f"G5 portability synth300 corpus requires 30 G2 single-turn cases, found {len(g2_single_turn)}")
    selected_cases = clarification_cases[:40] + g1_single_turn[:30] + g2_single_turn[:30]
    payload = {
        "corpus_version": G5_AGENT_PORTABILITY_SYNTH300_CORPUS_VERSION,
        "dataset": "bank",
        "source_corpus_version": source.get("corpus_version"),
        "source_corpus_sha256": source.get("corpus_sha256"),
        "selection_rule": (
            "all_clarification_followup_plus_first_30_G1_single_turn_plus_first_30_G2_single_turn_from_tier_b"
        ),
        "case_count": len(selected_cases),
        "selection_counts": {
            "clarification_followup": 40,
            "g1_single_turn": 30,
            "g2_single_turn": 30,
            "refinement_excluded": True,
        },
        "cases": selected_cases,
    }
    payload["corpus_sha256"] = sha256_json_payload(payload_without_hash(payload))
    return payload


def generate_bank_boundary_profiles_snapshot() -> dict[str, Any]:
    return generate_bank_boundary_profiles_corpus()


def row_to_profile(row: dict[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for field_name in BANK_FEATURE_ORDER:
        value = row[field_name]
        if field_name in BANK_BOOLEAN_FIELDS or field_name in {"Family", "Education"}:
            profile[field_name] = int(value)
        else:
            profile[field_name] = float(value)
    return profile


def render_bank_profile_prompt(
    profile: dict[str, Any],
    *,
    include_booleans: bool = True,
    trailing_sentences: list[str] | None = None,
) -> str:
    parts = [
        f"Income {render_number(profile['Income'])}",
        f"Family {int(profile['Family'])}",
        f"CCAvg {render_number(profile['CCAvg'])}",
        f"Education {int(profile['Education'])}",
        f"Mortgage {render_number(profile['Mortgage'])}",
    ]
    if include_booleans:
        for field_name in ("SecuritiesAccount", "CDAccount", "Online", "CreditCard"):
            parts.append(f"{field_name} {'yes' if int(profile[field_name]) == 1 else 'no'}")
    sentences = [", ".join(parts) + "."]
    if trailing_sentences:
        sentences.extend(str(item).strip() for item in trailing_sentences if str(item).strip())
    return " ".join(sentences).strip()


def render_bank_boolean_followup_prompt(profile: dict[str, Any]) -> str:
    parts = []
    for field_name in ("SecuritiesAccount", "CDAccount", "Online", "CreditCard"):
        parts.append(f"{field_name} {'yes' if int(profile[field_name]) == 1 else 'no'}")
    return ", ".join(parts) + "."


def render_bank_clarification_followup_prompt(
    profile: dict[str, Any],
    *,
    trailing_sentences: list[str] | None = None,
) -> str:
    return render_bank_profile_prompt(
        profile,
        include_booleans=True,
        trailing_sentences=trailing_sentences,
    )


def build_active_constraint_pattern(index: int, profile: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    patterns = [
        ("Do not change Income.", {"disallowed_changes": ["Income"]}),
        ("Do not change Mortgage.", {"disallowed_changes": ["Mortgage"]}),
        (
            f"Keep Mortgage at or below {render_number(max(0.0, float(profile['Mortgage'])))}.",
            {"numeric_bounds": {"Mortgage": {"max": float(profile["Mortgage"])}}},
        ),
        ("Change at most one thing.", {"max_changed_features": 1}),
        ("Prefer smaller edits.", {"prefer_fewer_changes": True}),
        (
            "Do not change Income and change at most one thing.",
            {"disallowed_changes": ["Income"], "max_changed_features": 1},
        ),
        (
            "Do not change Income and prefer smaller edits.",
            {"disallowed_changes": ["Income"], "prefer_fewer_changes": True},
        ),
        (
            f"Keep Income at or above {render_number(max(0.0, float(profile['Income']) - 5.0))}.",
            {"numeric_bounds": {"Income": {"min": float(profile["Income"]) - 5.0}}},
        ),
    ]
    text, raw_spec = patterns[(index - 1) % len(patterns)]
    normalized, errors = validate_and_normalize_constraint_spec(raw_spec, feature_order=BANK_FEATURE_ORDER)
    if errors:
        raise ValueError(f"Invalid active constraint pattern: {errors}")
    return text, normalized


def build_refinement_feedback_pattern(index: int, profile: dict[str, Any]) -> str:
    patterns = [
        "Do not change Income.",
        "Do not change Mortgage.",
        "Change at most one thing.",
        "Prefer smaller edits.",
        f"Mortgage must stay under {render_number(max(0.0, float(profile['Mortgage']) - 5.0))}.",
    ]
    return patterns[(index - 1) % len(patterns)]


def render_number(value: float) -> str:
    return format(float(value), ".15g")


def infer_tier_c_expected_version(path: Path) -> str | None:
    filename = Path(path).name
    if filename == TIER_C_CORPUS_PATH.name:
        return TIER_C_CORPUS_VERSION
    if filename == TIER_C_CORPUS_V1_PATH.name:
        return TIER_C_CORPUS_V1_VERSION
    return None


def infer_tier_b_expected_version(path: Path) -> str | None:
    filename = Path(path).name
    if filename == TIER_B_CORPUS_PATH.name:
        return TIER_B_CORPUS_VERSION
    if filename == TIER_B_SYNTH300_CORPUS_PATH.name:
        return TIER_B_SYNTH300_CORPUS_VERSION
    return None


def infer_g5_expected_version(path: Path) -> str | None:
    filename = Path(path).name
    if filename == G5_AGENT_PORTABILITY_CORPUS_PATH.name:
        return G5_AGENT_PORTABILITY_CORPUS_VERSION
    if filename == G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH.name:
        return G5_AGENT_PORTABILITY_SYNTH300_CORPUS_VERSION
    return None


def load_frozen_corpus(path: Path, *, expected_version: str | None) -> dict[str, Any]:
    source_path = Path(path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Frozen corpus file not found: {source_path}")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Frozen corpus file is not valid JSON: {source_path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen corpus payload must be a JSON object: {source_path}")
    version = payload.get("corpus_version")
    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"Frozen corpus version mismatch for {source_path}: expected {expected_version}, found {version}"
        )
    expected_hash = payload.get("corpus_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"Frozen corpus is missing corpus_sha256: {source_path}")
    computed_hash = sha256_json_payload(payload_without_hash(payload))
    if computed_hash != expected_hash:
        raise ValueError(
            f"Frozen corpus hash mismatch for {source_path}: expected {expected_hash}, computed {computed_hash}"
        )
    return dict(payload)


def payload_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "corpus_sha256"}
