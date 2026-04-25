from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from llm.src.part2_eval.common import sha256_json_payload
from llm.src.utils.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[3]
SOURCE_TOTEST_ROOT = ROOT / "ufce" / "data" / "folds" / "bank" / "totest"
TIER_B_V1_CORPUS_PATH = ROOT / "docs" / "validation" / "corpora" / "part2_tier_b_bank_sessions_v1.json"

BANK_SYNTH_FEATURE_ORDER = [
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
TEMPLATE_FIELDS = [
    "Family",
    "Education",
    "SecuritiesAccount",
    "CDAccount",
    "Online",
    "CreditCard",
]
NUMERIC_FIELDS = ["Income", "CCAvg", "Mortgage"]
INTEGER_FIELDS = {"Family", "Education", "SecuritiesAccount", "CDAccount", "Online", "CreditCard"}
NUMERIC_STEPS = {"Income": 1.0, "CCAvg": 0.1, "Mortgage": 1.0}
COMMON_CORE_RANGES = {
    "Income": {"min": 15.0, "max": 153.0},
    "CCAvg": {"min": 0.0, "max": 6.3},
    "Mortgage": {"min": 0.0, "max": 323.0},
}
TARGET_SOURCE_PROFILE_COUNT = 249
TARGET_SYNTH_PROFILE_COUNT = 51
TARGET_MAIN_PROFILE_COUNT = 300


def _quantize_numeric(field_name: str, value: float | int) -> float:
    numeric_value = float(value)
    if field_name == "CCAvg":
        return round(numeric_value + 1e-12, 1)
    return float(int(round(numeric_value)))


def _normalize_profile(raw_profile: dict[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for field_name in BANK_SYNTH_FEATURE_ORDER:
        value = raw_profile[field_name]
        if field_name in INTEGER_FIELDS:
            profile[field_name] = int(value)
        else:
            profile[field_name] = _quantize_numeric(field_name, value)
    return profile


def _profile_signature(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=True, sort_keys=True)


def _template_key_from_profile(profile: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(profile[field_name]) for field_name in TEMPLATE_FIELDS)


def _template_dict_from_key(template_key: tuple[int, ...]) -> dict[str, int]:
    return {field_name: int(value) for field_name, value in zip(TEMPLATE_FIELDS, template_key)}


def _ordered_unique(values: list[float]) -> list[float]:
    seen: set[float] = set()
    ordered: list[float] = []
    for value in values:
        normalized = float(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _compute_numeric_stats(frame: pd.DataFrame, field_name: str) -> dict[str, Any]:
    series = frame[field_name].astype(float)
    stats = {
        "min": _quantize_numeric(field_name, float(series.min())),
        "q25": _quantize_numeric(field_name, float(series.quantile(0.25))),
        "median": _quantize_numeric(field_name, float(series.quantile(0.50))),
        "q75": _quantize_numeric(field_name, float(series.quantile(0.75))),
        "max": _quantize_numeric(field_name, float(series.max())),
    }
    stats["anchor_values"] = _ordered_unique(
        [
            stats["min"],
            stats["q25"],
            stats["median"],
            stats["q75"],
            stats["max"],
        ]
    )
    if field_name == "Mortgage":
        nonzero = series.loc[series > 0.0]
        stats["zero_count"] = int((series == 0.0).sum())
        stats["nonzero_count"] = int(nonzero.shape[0])
        if nonzero.empty:
            stats["q25_nonzero"] = None
            stats["median_nonzero"] = None
            stats["q75_nonzero"] = None
            stats["max_nonzero"] = None
            stats["nonzero_anchor_values"] = [0.0]
        else:
            stats["q25_nonzero"] = _quantize_numeric(field_name, float(nonzero.quantile(0.25)))
            stats["median_nonzero"] = _quantize_numeric(field_name, float(nonzero.quantile(0.50)))
            stats["q75_nonzero"] = _quantize_numeric(field_name, float(nonzero.quantile(0.75)))
            stats["max_nonzero"] = _quantize_numeric(field_name, float(nonzero.max()))
            stats["nonzero_anchor_values"] = _ordered_unique(
                [
                    stats["q25_nonzero"],
                    stats["median_nonzero"],
                    stats["q75_nonzero"],
                    stats["max_nonzero"],
                ]
            )
    return stats


def _build_template_summary(unique_frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], dict[str, Any]]]:
    templates: list[dict[str, Any]] = []
    template_map: dict[tuple[int, ...], dict[str, Any]] = {}
    grouped = unique_frame.groupby(TEMPLATE_FIELDS, sort=False, dropna=False)
    for raw_key, group_frame in grouped:
        template_key = tuple(int(value) for value in raw_key)
        numeric_stats = {
            field_name: _compute_numeric_stats(group_frame, field_name)
            for field_name in NUMERIC_FIELDS
        }
        template_record = {
            "template_key": template_key,
            "template": _template_dict_from_key(template_key),
            "count": int(group_frame.shape[0]),
            "numeric_stats": numeric_stats,
        }
        templates.append(template_record)
        template_map[template_key] = template_record
    templates.sort(key=lambda item: (-int(item["count"]), tuple(item["template_key"])))
    return templates, template_map


def _compute_source_file_ranges(source_frames: list[pd.DataFrame]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for field_name in NUMERIC_FIELDS:
        mins = [_quantize_numeric(field_name, float(frame[field_name].min())) for frame in source_frames]
        maxs = [_quantize_numeric(field_name, float(frame[field_name].max())) for frame in source_frames]
        ranges[field_name] = {
            "min": float(max(mins)),
            "max": float(min(maxs)),
        }
    return ranges


def _compute_full_ranges(unique_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        field_name: {
            "min": _quantize_numeric(field_name, float(unique_frame[field_name].min())),
            "max": _quantize_numeric(field_name, float(unique_frame[field_name].max())),
        }
        for field_name in NUMERIC_FIELDS
    }


def load_bank_source_analysis() -> dict[str, Any]:
    source_paths = sorted(SOURCE_TOTEST_ROOT.glob("testfold_*_pred_0.csv"))
    if len(source_paths) != 5:
        raise ValueError(f"Expected 5 bank totest source files, found {len(source_paths)} in {SOURCE_TOTEST_ROOT}")

    source_frames: list[pd.DataFrame] = []
    source_files: list[dict[str, Any]] = []
    for source_path in source_paths:
        frame = pd.read_csv(source_path).loc[:, BANK_SYNTH_FEATURE_ORDER].copy()
        for field_name in BANK_SYNTH_FEATURE_ORDER:
            if field_name in INTEGER_FIELDS:
                frame[field_name] = frame[field_name].astype(int)
            else:
                frame[field_name] = frame[field_name].astype(float)
        source_frames.append(frame)
        source_files.append(
            {
                "path": str(source_path.relative_to(ROOT)),
                "sha256": sha256_file(source_path),
                "row_count": int(frame.shape[0]),
            }
        )

    merged_frame = pd.concat(source_frames, ignore_index=True)
    unique_frame = merged_frame.drop_duplicates(subset=BANK_SYNTH_FEATURE_ORDER, keep="first").reset_index(drop=True)
    source_profiles = [
        _normalize_profile(row)
        for row in unique_frame.to_dict(orient="records")
    ]
    if len(source_profiles) != TARGET_SOURCE_PROFILE_COUNT:
        raise ValueError(
            "Unexpected deduped source pool size: expected "
            f"{TARGET_SOURCE_PROFILE_COUNT}, found {len(source_profiles)}"
        )

    templates, template_map = _build_template_summary(unique_frame)
    template_summary_payload = [
        {
            "template": item["template"],
            "count": item["count"],
            "numeric_stats": item["numeric_stats"],
        }
        for item in templates
    ]
    return {
        "source_files": source_files,
        "merged_row_count": int(merged_frame.shape[0]),
        "unique_row_count": int(unique_frame.shape[0]),
        "duplicate_row_count": int(merged_frame.shape[0] - unique_frame.shape[0]),
        "source_profiles": source_profiles,
        "source_signatures": {_profile_signature(profile) for profile in source_profiles},
        "templates": templates,
        "template_map": template_map,
        "template_summary_payload": template_summary_payload,
        "template_summary_sha256": sha256_json_payload(template_summary_payload),
        "common_core_ranges": _compute_source_file_ranges(source_frames),
        "full_ranges": _compute_full_ranges(unique_frame),
    }


def _load_v1_seed_profile_signatures() -> set[str]:
    payload = json.loads(TIER_B_V1_CORPUS_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    signatures = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        profile = case.get("seed_profile")
        if not isinstance(profile, dict):
            continue
        signatures.add(_profile_signature(_normalize_profile(profile)))
    return signatures


def _profile_prediction_rejected(*, bundle, desired_outcome: int | float, profile: dict[str, Any]) -> bool:
    frame = pd.DataFrame([profile], columns=list(bundle.feature_order))
    prediction = bundle.lr.predict(frame)[0]
    return int(prediction) != int(desired_outcome)


def _mortgage_center_value(stats: dict[str, Any]) -> float:
    if int(stats["nonzero_count"]) == 0:
        return 0.0
    if int(stats["zero_count"]) >= int(stats["nonzero_count"]):
        return 0.0
    return float(stats["median_nonzero"])


def _mortgage_low_value(stats: dict[str, Any]) -> float:
    if int(stats["zero_count"]) > 0:
        return 0.0
    if stats["q25_nonzero"] is not None:
        return float(stats["q25_nonzero"])
    return float(stats["min"])


def _mortgage_high_value(stats: dict[str, Any]) -> float:
    if int(stats["nonzero_count"]) == 0:
        return 0.0
    if stats["q75_nonzero"] is not None:
        return float(stats["q75_nonzero"])
    if stats["max_nonzero"] is not None:
        return float(stats["max_nonzero"])
    return float(stats["max"])


def _build_common_template_candidates(source_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for template_record in source_analysis["templates"][:12]:
        numeric_stats = template_record["numeric_stats"]
        numeric_variants = [
            (
                "center",
                {
                    "Income": float(numeric_stats["Income"]["median"]),
                    "CCAvg": float(numeric_stats["CCAvg"]["median"]),
                    "Mortgage": _mortgage_center_value(numeric_stats["Mortgage"]),
                },
            ),
            (
                "low",
                {
                    "Income": float(numeric_stats["Income"]["q25"]),
                    "CCAvg": float(numeric_stats["CCAvg"]["q25"]),
                    "Mortgage": _mortgage_low_value(numeric_stats["Mortgage"]),
                },
            ),
            (
                "high",
                {
                    "Income": float(numeric_stats["Income"]["q75"]),
                    "CCAvg": float(numeric_stats["CCAvg"]["q75"]),
                    "Mortgage": _mortgage_high_value(numeric_stats["Mortgage"]),
                },
            ),
        ]
        for variant_name, numeric_values in numeric_variants:
            profile = dict(template_record["template"])
            for field_name, value in numeric_values.items():
                profile[field_name] = _quantize_numeric(field_name, value)
            normalized_profile = _normalize_profile(profile)
            candidates.append(
                {
                    "stage": "common_template_expansion",
                    "generation_rule": f"common_template_{variant_name}",
                    "source_template": dict(template_record["template"]),
                    "profile": normalized_profile,
                }
            )
    return candidates


def select_prioritized_rare_seed_rows(source_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    template_counts = {
        tuple(item["template_key"]): int(item["count"])
        for item in source_analysis["templates"]
    }
    prioritized: list[dict[str, Any]] = []
    for source_index, profile in enumerate(source_analysis["source_profiles"]):
        template_key = _template_key_from_profile(profile)
        prioritized.append(
            {
                "profile": dict(profile),
                "source_index": int(source_index),
                "template_key": template_key,
                "source_template": _template_dict_from_key(template_key),
                "template_count": int(template_counts[template_key]),
            }
        )
    prioritized.sort(
        key=lambda item: (
            -int(item["profile"]["CDAccount"]),
            -int(item["profile"]["SecuritiesAccount"]),
            int(item["template_count"]),
            int(item["source_index"]),
        )
    )
    return prioritized


def _ordered_anchor_candidates(field_name: str, current_value: float, stats: dict[str, Any]) -> list[float]:
    anchor_values = list(stats["anchor_values"])
    if field_name == "Mortgage" and int(stats["nonzero_count"]) > 0:
        anchor_values.extend(stats["nonzero_anchor_values"])
    deduped = _ordered_unique([float(value) for value in anchor_values])
    alternatives = [value for value in deduped if abs(float(value) - float(current_value)) > 1e-9]
    priority = {value: index for index, value in enumerate(deduped)}
    alternatives.sort(
        key=lambda value: (
            abs(float(value) - float(current_value)),
            int(priority[value]),
            float(value),
        )
    )
    return alternatives


def _build_single_numeric_mutation(
    *,
    source_row: dict[str, Any],
    template_record: dict[str, Any],
    start_field_index: int,
) -> dict[str, Any] | None:
    for offset in range(len(NUMERIC_FIELDS)):
        field_name = NUMERIC_FIELDS[(start_field_index + offset) % len(NUMERIC_FIELDS)]
        current_value = float(source_row["profile"][field_name])
        stats = template_record["numeric_stats"][field_name]
        for candidate_value in _ordered_anchor_candidates(field_name, current_value, stats):
            if field_name == "Mortgage" and int(template_record["numeric_stats"]["Mortgage"]["nonzero_count"]) == 0:
                if candidate_value != 0.0:
                    continue
            profile = dict(source_row["profile"])
            profile[field_name] = _quantize_numeric(field_name, candidate_value)
            if profile == source_row["profile"]:
                continue
            return {
                "stage": "rare_template_mutation",
                "generation_rule": f"rare_seed_nearest_{field_name.lower()}",
                "source_template": dict(source_row["source_template"]),
                "profile": _normalize_profile(profile),
                "source_index": int(source_row["source_index"]),
            }
    return None


def _build_rare_template_candidates(source_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    prioritized = select_prioritized_rare_seed_rows(source_analysis)
    for seed_index, source_row in enumerate(prioritized):
        template_record = source_analysis["template_map"][tuple(source_row["template_key"])]
        candidate = _build_single_numeric_mutation(
            source_row=source_row,
            template_record=template_record,
            start_field_index=seed_index % len(NUMERIC_FIELDS),
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        if len(candidates) == 15:
            break
    return candidates


def _build_fallback_candidates(source_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    prioritized = select_prioritized_rare_seed_rows(source_analysis)
    for source_row in prioritized:
        template_record = source_analysis["template_map"][tuple(source_row["template_key"])]
        for start_field_index in range(len(NUMERIC_FIELDS)):
            for offset in range(len(NUMERIC_FIELDS)):
                field_name = NUMERIC_FIELDS[(start_field_index + offset) % len(NUMERIC_FIELDS)]
                current_value = float(source_row["profile"][field_name])
                for candidate_value in _ordered_anchor_candidates(
                    field_name,
                    current_value,
                    template_record["numeric_stats"][field_name],
                ):
                    if (
                        field_name == "Mortgage"
                        and int(template_record["numeric_stats"]["Mortgage"]["nonzero_count"]) == 0
                        and candidate_value != 0.0
                    ):
                        continue
                    profile = dict(source_row["profile"])
                    profile[field_name] = _quantize_numeric(field_name, candidate_value)
                    if profile == source_row["profile"]:
                        continue
                    candidates.append(
                        {
                            "stage": "fallback_mutation",
                            "generation_rule": f"fallback_mutation_{field_name.lower()}",
                            "source_template": dict(source_row["source_template"]),
                            "profile": _normalize_profile(profile),
                            "source_index": int(source_row["source_index"]),
                        }
                    )
    return candidates


def generate_bank_synth_profile_pool(*, bundle, desired_outcome: int | float) -> dict[str, Any]:
    source_analysis = load_bank_source_analysis()
    source_profiles = [dict(profile) for profile in source_analysis["source_profiles"]]
    source_signatures = set(source_analysis["source_signatures"])
    v1_signatures = _load_v1_seed_profile_signatures()

    for profile in source_profiles:
        if not _profile_prediction_rejected(bundle=bundle, desired_outcome=desired_outcome, profile=profile):
            raise ValueError("Source profile pool contains a non-rejected bank profile under the current predictor.")

    selected_signatures = set(source_signatures)
    synth_records: list[dict[str, Any]] = []
    accepted_stage_counts = {
        "common_template_expansion": 0,
        "rare_template_mutation": 0,
        "fallback_mutation": 0,
    }

    candidate_stages = [
        _build_common_template_candidates(source_analysis),
        _build_rare_template_candidates(source_analysis),
        _build_fallback_candidates(source_analysis),
    ]
    for stage_candidates in candidate_stages:
        for candidate in stage_candidates:
            signature = _profile_signature(candidate["profile"])
            if signature in selected_signatures:
                continue
            if signature in v1_signatures:
                continue
            if not _profile_prediction_rejected(
                bundle=bundle,
                desired_outcome=desired_outcome,
                profile=candidate["profile"],
            ):
                continue
            synth_records.append(candidate)
            selected_signatures.add(signature)
            accepted_stage_counts[str(candidate["stage"])] += 1
            if len(synth_records) == TARGET_SYNTH_PROFILE_COUNT:
                break
        if len(synth_records) == TARGET_SYNTH_PROFILE_COUNT:
            break

    if len(synth_records) != TARGET_SYNTH_PROFILE_COUNT:
        raise ValueError(
            "Unable to generate the required synthetic bank profile count: "
            f"expected {TARGET_SYNTH_PROFILE_COUNT}, found {len(synth_records)}"
        )

    synth_profiles = [dict(item["profile"]) for item in synth_records]
    profile_pool = source_profiles + synth_profiles
    if len(profile_pool) != TARGET_MAIN_PROFILE_COUNT:
        raise ValueError(
            f"Unexpected combined profile count: expected {TARGET_MAIN_PROFILE_COUNT}, found {len(profile_pool)}"
        )
    return {
        "source_analysis": source_analysis,
        "source_profiles": source_profiles,
        "synth_records": synth_records,
        "synth_profiles": synth_profiles,
        "profile_pool": profile_pool,
        "stage_counts": accepted_stage_counts,
    }


def _interleave_profiles(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not secondary:
        return [dict(profile) for profile in primary]
    combined: list[dict[str, Any]] = []
    primary_index = 0
    secondary_index = 0
    total = len(primary) + len(secondary)
    for position in range(total):
        target_secondary = round((position + 1) * len(secondary) / total)
        if secondary_index < target_secondary and secondary_index < len(secondary):
            combined.append(dict(secondary[secondary_index]))
            secondary_index += 1
        elif primary_index < len(primary):
            combined.append(dict(primary[primary_index]))
            primary_index += 1
        else:
            combined.append(dict(secondary[secondary_index]))
            secondary_index += 1
    return combined


def build_synth_group_profile_map(profile_pool: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_profiles = list(profile_pool["source_profiles"])
    synth_profiles = list(profile_pool["synth_profiles"])
    source_splits = {"G1": 100, "G2": 100, "REFINEMENT": 49}
    synth_splits = {"G1": 20, "G2": 20, "REFINEMENT": 11}

    source_index = 0
    synth_index = 0
    grouped: dict[str, list[dict[str, Any]]] = {}
    for group_name in ("G1", "G2", "REFINEMENT"):
        source_chunk = source_profiles[source_index : source_index + source_splits[group_name]]
        synth_chunk = synth_profiles[synth_index : synth_index + synth_splits[group_name]]
        source_index += source_splits[group_name]
        synth_index += synth_splits[group_name]
        grouped[group_name] = _interleave_profiles(source_chunk, synth_chunk)

    if source_index != len(source_profiles) or synth_index != len(synth_profiles):
        raise ValueError("Synthetic group split did not consume the full source and synth profile pools.")
    return grouped


def build_synth_generation_metadata(profile_pool: dict[str, Any]) -> dict[str, Any]:
    source_analysis = profile_pool["source_analysis"]
    return {
        "source_files": list(source_analysis["source_files"]),
        "source_unique_count": int(source_analysis["unique_row_count"]),
        "template_summary_sha256": source_analysis["template_summary_sha256"],
        "generation_policy": {
            "source_profile_count": TARGET_SOURCE_PROFILE_COUNT,
            "synth_profile_count": TARGET_SYNTH_PROFILE_COUNT,
            "total_profile_count": TARGET_MAIN_PROFILE_COUNT,
            "common_template_expansion_target": 36,
            "rare_template_mutation_target": 15,
            "dedupe_generated_against_v1": True,
            "common_core_ranges": dict(COMMON_CORE_RANGES),
            "full_observed_ranges": dict(source_analysis["full_ranges"]),
            "zero_inflated_mortgage_guard": True,
        },
        "generated_stage_counts": dict(profile_pool["stage_counts"]),
    }


def _append_boundary_case(
    *,
    cases: list[dict[str, Any]],
    seen: set[str],
    case_id: str,
    bucket: str,
    profile: dict[str, Any],
    source_template: dict[str, int],
    generation_rule: str,
    expected_validity: str,
    notes: list[str],
) -> bool:
    normalized = _normalize_profile(profile)
    signature = _profile_signature(normalized)
    if signature in seen:
        return False
    seen.add(signature)
    cases.append(
        {
            "case_id": case_id,
            "bucket": bucket,
            "seed_profile": normalized,
            "source_template": dict(source_template),
            "generation_rule": generation_rule,
            "expected_validity": expected_validity,
            "notes": list(notes),
        }
    )
    return True


def _clamp_to_range(field_name: str, value: float, bounds: dict[str, float]) -> float:
    return _quantize_numeric(field_name, max(float(bounds["min"]), min(float(bounds["max"]), float(value))))


def _intersection_bounds(local_bounds: dict[str, Any], target_bounds: dict[str, float]) -> dict[str, float] | None:
    lower = max(float(local_bounds["min"]), float(target_bounds["min"]))
    upper = min(float(local_bounds["max"]), float(target_bounds["max"]))
    if lower > upper:
        return None
    return {"min": lower, "max": upper}


def generate_bank_boundary_profiles_corpus() -> dict[str, Any]:
    source_analysis = load_bank_source_analysis()
    templates = source_analysis["templates"]
    common_core = source_analysis["common_core_ranges"]
    full_ranges = source_analysis["full_ranges"]
    prioritized_rare_rows = select_prioritized_rare_seed_rows(source_analysis)

    cases: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    bucket_counts = Counter()

    full_templates = templates[10:20]
    if len(full_templates) < 10:
        full_templates = templates[:10]

    case_index = 1
    for template_record in templates:
        if bucket_counts["core_boundary"] >= 20:
            break
        low_profile = dict(template_record["template"])
        high_profile = dict(template_record["template"])
        core_bounds: dict[str, dict[str, float]] = {}
        for field_name in NUMERIC_FIELDS:
            bounds = _intersection_bounds(template_record["numeric_stats"][field_name], common_core[field_name])
            if bounds is None:
                core_bounds = {}
                break
            core_bounds[field_name] = bounds
        if not core_bounds:
            continue

        low_profile["Income"] = _quantize_numeric("Income", core_bounds["Income"]["min"])
        low_profile["CCAvg"] = _quantize_numeric("CCAvg", core_bounds["CCAvg"]["min"])
        low_profile["Mortgage"] = _quantize_numeric("Mortgage", core_bounds["Mortgage"]["min"])
        if _append_boundary_case(
            cases=cases,
            seen=seen_profiles,
            case_id=f"BOUNDARY-CORE-{case_index:03d}",
            bucket="core_boundary",
            profile=low_profile,
            source_template=template_record["template"],
            generation_rule="common_core_lower_clamped",
            expected_validity="valid",
            notes=["inside common-core bounds", "clamped to template-local range"],
        ):
            bucket_counts["core_boundary"] += 1
            case_index += 1

        high_profile["Income"] = _quantize_numeric("Income", core_bounds["Income"]["max"])
        high_profile["CCAvg"] = _quantize_numeric("CCAvg", core_bounds["CCAvg"]["max"])
        high_profile["Mortgage"] = _quantize_numeric("Mortgage", core_bounds["Mortgage"]["max"])
        if _append_boundary_case(
            cases=cases,
            seen=seen_profiles,
            case_id=f"BOUNDARY-CORE-{case_index:03d}",
            bucket="core_boundary",
            profile=high_profile,
            source_template=template_record["template"],
            generation_rule="common_core_upper_clamped",
            expected_validity="valid",
            notes=["inside common-core bounds", "clamped to template-local range"],
        ):
            bucket_counts["core_boundary"] += 1
            case_index += 1

    case_index = 1
    for template_record in full_templates:
        low_profile = dict(template_record["template"])
        high_profile = dict(template_record["template"])
        for field_name in ("Income", "CCAvg", "Mortgage"):
            low_profile[field_name] = float(template_record["numeric_stats"][field_name]["min"])
            high_profile[field_name] = float(template_record["numeric_stats"][field_name]["max"])
        if _append_boundary_case(
            cases=cases,
            seen=seen_profiles,
            case_id=f"BOUNDARY-FULL-{case_index:03d}",
            bucket="full_boundary",
            profile=low_profile,
            source_template=template_record["template"],
            generation_rule="full_observed_lower_template_min",
            expected_validity="valid",
            notes=["inside full observed bounds", "template-local minimum profile"],
        ):
            bucket_counts["full_boundary"] += 1
            case_index += 1
        if _append_boundary_case(
            cases=cases,
            seen=seen_profiles,
            case_id=f"BOUNDARY-FULL-{case_index:03d}",
            bucket="full_boundary",
            profile=high_profile,
            source_template=template_record["template"],
            generation_rule="full_observed_upper_template_max",
            expected_validity="valid",
            notes=["inside full observed bounds", "template-local maximum profile"],
        ):
            bucket_counts["full_boundary"] += 1
            case_index += 1

    case_index = 1
    for rare_row in prioritized_rare_rows:
        if bucket_counts["rare_adversarial"] >= 10:
            break
        template_record = source_analysis["template_map"][tuple(rare_row["template_key"])]
        candidate_profile = dict(rare_row["profile"])
        chosen = False
        for field_name in ("Mortgage", "CCAvg", "Income"):
            alternatives = _ordered_anchor_candidates(
                field_name,
                float(candidate_profile[field_name]),
                template_record["numeric_stats"][field_name],
            )
            if not alternatives:
                continue
            candidate_profile[field_name] = _quantize_numeric(field_name, alternatives[-1])
            chosen = True
            break
        if not chosen:
            continue
        if _append_boundary_case(
            cases=cases,
            seen=seen_profiles,
            case_id=f"BOUNDARY-RARE-{case_index:03d}",
            bucket="rare_adversarial",
            profile=candidate_profile,
            source_template=rare_row["source_template"],
            generation_rule="rare_seed_extreme_local_anchor",
            expected_validity="valid",
            notes=["rare observed discrete pattern", "numeric stress stays inside template-local range"],
        ):
            bucket_counts["rare_adversarial"] += 1
            case_index += 1

    negative_specs = [
        ("Income", float(full_ranges["Income"]["min"]) - 1.0, "below_full_min"),
        ("Income", float(full_ranges["Income"]["max"]) + 1.0, "above_full_max"),
        ("CCAvg", float(full_ranges["CCAvg"]["min"]) - 0.1, "below_full_min"),
        ("CCAvg", float(full_ranges["CCAvg"]["max"]) + 0.1, "above_full_max"),
        ("Mortgage", float(full_ranges["Mortgage"]["min"]) - 1.0, "below_full_min"),
        ("Mortgage", float(full_ranges["Mortgage"]["max"]) + 1.0, "above_full_max"),
        ("Income", float(full_ranges["Income"]["min"]) - 1.0, "below_full_min_repeat"),
        ("CCAvg", float(full_ranges["CCAvg"]["max"]) + 0.1, "above_full_max_repeat"),
        ("Mortgage", float(full_ranges["Mortgage"]["max"]) + 1.0, "above_full_max_repeat"),
        ("Income", float(full_ranges["Income"]["max"]) + 1.0, "above_full_max_repeat"),
    ]
    case_index = 1
    for spec_index, (field_name, invalid_value, label) in enumerate(negative_specs):
        base_row = prioritized_rare_rows[spec_index]
        profile = dict(base_row["profile"])
        profile[field_name] = _quantize_numeric(field_name, invalid_value)
        if _append_boundary_case(
            cases=cases,
            seen=seen_profiles,
            case_id=f"BOUNDARY-NEG-{case_index:03d}",
            bucket="negative_out_of_range",
            profile=profile,
            source_template=base_row["source_template"],
            generation_rule=f"negative_out_of_range_{field_name.lower()}_{label}",
            expected_validity="reject",
            notes=["one numeric feature moved just outside full observed range", "discrete fields remain observed-valid"],
        ):
            bucket_counts["negative_out_of_range"] += 1
            case_index += 1

    expected_counts = {
        "core_boundary": 20,
        "full_boundary": 20,
        "rare_adversarial": 10,
        "negative_out_of_range": 10,
    }
    if dict(bucket_counts) != expected_counts:
        raise ValueError(f"Unexpected boundary bucket counts: expected {expected_counts}, found {dict(bucket_counts)}")

    payload = {
        "corpus_version": "part2_bank_boundary_profiles_v1",
        "dataset": "bank",
        "case_count": len(cases),
        "bucket_counts": dict(bucket_counts),
        "cases": cases,
    }
    payload["corpus_sha256"] = sha256_json_payload({key: value for key, value in payload.items() if key != "corpus_sha256"})
    return payload
