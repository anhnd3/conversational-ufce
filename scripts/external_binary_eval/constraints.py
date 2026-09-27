"""Train-derived raw-space constraint policies."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, MONTHS, REGIMES


def feature_family(feature):
    for month in MONTHS:
        suffix = "_" + month
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return None


def actionable_features(feature_names, regime):
    selected = [
        name for name in feature_names
        if feature_family(name) in set(regime.families)
    ]
    return selected


def _is_integer_like(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return bool(len(values) == 0 or np.all(np.isclose(values, np.round(values))))


def build_constraint_policy(train_frame, feature_schema):
    policies = {}
    numeric = set(feature_schema["numeric_features"])
    for regime in REGIMES:
        features = actionable_features(feature_schema["feature_names"], regime)
        feature_rules = {}
        for feature in features:
            values = pd.to_numeric(train_frame[feature], errors="coerce")
            q25, q75 = values.quantile([0.25, 0.75])
            iqr = float(q75 - q25) if np.isfinite(q75 - q25) else 0.0
            cap = float(values.quantile(regime.cap_quantile)) if values.notna().any() else 0.0
            feature_rules[feature] = {
                "family": feature_family(feature),
                "delta": float(regime.delta_iqr * iqr),
                "train_iqr": float(iqr),
                "train_cap": float(cap),
                "direction": "increase_only",
                "step": 1 if _is_integer_like(values) else None,
                "integer": _is_integer_like(values),
            }
        policies[regime.name] = {
            "name": regime.name,
            "actionable_features": features,
            "n_actionable_features": len(features),
            "delta_iqr": regime.delta_iqr,
            "cap_quantile": regime.cap_quantile,
            "all_non_digital_immutable": True,
            "rules": feature_rules,
        }
    if [policies[name]["n_actionable_features"] for name in ("STRICT", "MODERATE", "FLEXIBLE")] != [16, 28, 36]:
        raise AssertionError("Regime actionable feature counts must be 16/28/36")
    return policies


def policy_for_query(train_policy, raw_query, train_frame, feature_schema):
    """Materialize per-query lower/upper bounds in canonical raw space."""
    result = {
        "regime": train_policy["name"],
        "actionable_features": list(train_policy["actionable_features"]),
        "features": {},
    }
    numeric = set(feature_schema["numeric_features"])
    query = raw_query.iloc[0] if isinstance(raw_query, pd.DataFrame) else pd.Series(raw_query)
    for feature in feature_schema["feature_names"]:
        value = query.get(feature, np.nan)
        is_missing = pd.isna(value)
        if feature not in train_policy["rules"] or is_missing or feature not in numeric:
            result["features"][feature] = {
                "immutable": True,
                "missing_query": bool(is_missing),
                "direction": "immutable",
            }
            continue
        value = float(value)
        rule = train_policy["rules"][feature]
        requested_upper = min(float(rule["train_cap"]), value + float(rule["delta"]))
        # A factual row can lie above the train quantile cap.  In that case the
        # exact intersection of [factual, +inf) with the cap is the singleton
        # factual value, i.e. no movement is allowed for this query.
        cap_below_factual = requested_upper < value
        upper = max(value, requested_upper)
        if rule["integer"]:
            upper = float(np.floor(upper))
        result["features"][feature] = {
            "immutable": False,
            "lower": value,
            "upper": max(value, upper),
            "cap_below_factual": bool(cap_below_factual),
            "direction": "increase_only",
            "step": rule["step"],
            "integer": bool(rule["integer"]),
        }
    # Record raw categorical domains from train, rather than accepting arbitrary
    # one-hot/ordinal values produced by a generator.
    result["categorical_domains"] = {
        feature: sorted(train_frame[feature].dropna().astype(str).unique().tolist())
        for feature in feature_schema["categorical_features"]
    }
    return result


def write_constraint_policy(path, policies):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(policies, handle, indent=2, sort_keys=True, default=str)
