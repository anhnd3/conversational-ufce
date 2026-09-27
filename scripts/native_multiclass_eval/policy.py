"""Frozen query bounds and train-audited academic invariants."""

import math

import numpy as np
import pandas as pd

from .config import ACTIONABLE_FEATURES, INVARIANT_CANDIDATES


def audit_invariants(train_frame):
    findings = []
    for name, left_key, operator, right_key, rationale in INVARIANT_CANDIDATES:
        # All registered relations are evaluated separately in each semester.
        for sem in ("1st sem", "2nd sem"):
            approved = "Curricular units %s (approved)" % sem
            without = "Curricular units %s (without evaluations)" % sem
            enrolled = "Curricular units %s (enrolled)" % sem
            evaluations = "Curricular units %s (evaluations)" % sem
            if left_key == "approved":
                left = pd.to_numeric(train_frame[approved], errors="coerce")
            elif left_key == "without_evaluations":
                left = pd.to_numeric(train_frame[without], errors="coerce")
            elif left_key == "approved_plus_without_evaluations":
                left = pd.to_numeric(train_frame[approved], errors="coerce") + pd.to_numeric(train_frame[without], errors="coerce")
            else:
                raise ValueError("Unsupported audited invariant operand: %s" % left_key)
            relation_name = "%s_%s" % (name, sem.replace(" ", "_"))
            if right_key == "enrolled":
                right = pd.to_numeric(train_frame[enrolled], errors="coerce")
            elif right_key == "evaluations":
                right = pd.to_numeric(train_frame[evaluations], errors="coerce")
            else:
                raise ValueError("Unsupported audited invariant right operand: %s" % right_key)
            if operator == "le":
                violations = int((left > right).sum())
            elif operator == "ge":
                violations = int((left < right).sum())
            else:
                raise ValueError("Unsupported audited invariant operator: %s" % operator)
            findings.append({
                "name": relation_name,
                "features": [approved, without, enrolled, evaluations],
                "semantic_rationale": rationale,
                "train_rows": int(len(train_frame)),
                "train_violations": violations,
                "train_violation_rate": float(violations / len(train_frame)) if len(train_frame) else None,
                "enforced": bool(violations == 0),
            })
    return findings


def build_train_policy(train_frame, schema):
    invariants = audit_invariants(train_frame)
    numeric = set(schema["numeric_features"])
    if any(name not in numeric for name in ACTIONABLE_FEATURES):
        raise ValueError("Every actionable academic field must be numeric")
    rules = {}
    for feature in ACTIONABLE_FEATURES:
        values = pd.to_numeric(train_frame[feature], errors="raise").astype(float)
        q25, q75 = values.quantile([0.25, 0.75])
        rules[feature] = {
            "train_min": float(values.min()),
            "train_max": float(values.max()),
            "train_iqr": float(q75 - q25),
            "delta": float(0.5 * (q75 - q25)),
            "integer": bool(np.all(np.isclose(values.to_numpy(), np.round(values.to_numpy())))),
            "intrinsic_min": 0.0,
            "intrinsic_max": 20.0 if feature.endswith("(grade)") else None,
        }
    return {
        "actionable_features": list(ACTIONABLE_FEATURES),
        "rules": rules,
        "invariants": invariants,
        "invariants_enforced": [item["name"] for item in invariants if item["enforced"]],
        "immutable_features": [name for name in schema["feature_names"] if name not in ACTIONABLE_FEATURES],
        "categorical_domains": {
            name: sorted(train_frame[name].astype(str).unique().tolist())
            for name in schema["categorical_features"]
        },
    }


def policy_for_query(train_policy, factual):
    query = factual.iloc[0] if isinstance(factual, pd.DataFrame) else pd.Series(factual)
    features = {}
    for name, rule in train_policy["rules"].items():
        value = float(query[name])
        delta = float(rule["delta"])
        lower_domain = max(float(rule["train_min"]), float(rule["intrinsic_min"]))
        upper_domain = float(rule["train_max"])
        if rule["intrinsic_max"] is not None:
            upper_domain = min(upper_domain, float(rule["intrinsic_max"]))
        # Keep an observed factual feasible even when a held-out value lies
        # just outside the extrema seen during training.
        lower = max(lower_domain, value - delta)
        upper = min(upper_domain, value + delta)
        lower = min(lower, value)
        upper = max(upper, value)
        if rule["integer"]:
            lower = float(math.ceil(lower - 1e-12))
            upper = float(math.floor(upper + 1e-12))
            lower = min(lower, value)
            upper = max(upper, value)
        features[name] = {
            "lower": float(lower),
            "upper": float(upper),
            "integer": bool(rule["integer"]),
            "immutable": False,
        }
    return {
        "features": features,
        "immutable_features": list(train_policy["immutable_features"]),
        "categorical_domains": train_policy["categorical_domains"],
        "invariants_enforced": list(train_policy["invariants_enforced"]),
    }
