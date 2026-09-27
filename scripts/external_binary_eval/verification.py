"""Independent canonical raw-space candidate verification."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class CandidateVerification:
    raw_reconstruction_ok: bool
    categorical_domain_valid: bool
    immutable_valid: bool
    bounds_valid: bool
    direction_valid: bool
    target_valid: bool
    valid: bool
    constraint_feasible: bool
    violations: list
    raw_candidate: dict

    def as_dict(self):
        return asdict(self)


def reconstruct_raw(candidate, feature_schema):
    names = list(feature_schema["feature_names"])
    if isinstance(candidate, pd.Series):
        candidate = candidate.to_frame().T
    if isinstance(candidate, dict):
        candidate = pd.DataFrame([candidate])
    if not isinstance(candidate, pd.DataFrame) or len(candidate) == 0:
        return None, ["schema_violation"]
    missing = [name for name in names if name not in candidate.columns]
    if missing:
        return None, ["schema_violation"]
    row = candidate.iloc[0].loc[names].to_dict()
    return pd.DataFrame([row], columns=names), []


def verify_candidate(raw_query, candidate, bundle, feature_schema, query_policy):
    raw_candidate, violations = reconstruct_raw(candidate, feature_schema)
    if raw_candidate is None:
        return CandidateVerification(False, False, False, False, False, False, False, False, violations, {})
    categorical = set(feature_schema["categorical_features"])
    numeric = set(feature_schema["numeric_features"])
    query = raw_query.iloc[0] if isinstance(raw_query, pd.DataFrame) else pd.Series(raw_query)
    row = raw_candidate.iloc[0]
    domains = query_policy.get("categorical_domains", {})
    categorical_valid = True
    for feature in categorical:
        if pd.isna(row[feature]) and pd.isna(query[feature]):
            continue
        value = str(row[feature])
        if value not in set(domains.get(feature, [])):
            categorical_valid = False
            violations.append("categorical_domain_violation:%s" % feature)
    immutable_valid = True
    bounds_valid = True
    direction_valid = True
    for feature, rule in query_policy.get("features", {}).items():
        if feature not in row:
            immutable_valid = False
            violations.append("schema_violation:%s" % feature)
            continue
        if feature in categorical:
            if rule.get("immutable", True):
                query_missing = pd.isna(query[feature])
                candidate_missing = pd.isna(row[feature])
                if query_missing:
                    if not candidate_missing:
                        immutable_valid = False
                        violations.append("immutable_violation:%s" % feature)
                elif str(row[feature]) != str(query[feature]):
                    immutable_valid = False
                    violations.append("immutable_violation:%s" % feature)
            else:
                allowed = set(rule.get("allowed_values", domains.get(feature, [])))
                if pd.isna(row[feature]) or str(row[feature]) not in allowed:
                    categorical_valid = False
                    violations.append("categorical_domain_violation:%s" % feature)
            continue
        candidate_value = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
        query_value = pd.to_numeric(pd.Series([query[feature]]), errors="coerce").iloc[0]
        if pd.isna(candidate_value):
            if rule.get("immutable", True) and pd.isna(query_value):
                continue
            bounds_valid = False
            violations.append("numeric_bound_violation:%s" % feature)
            continue
        if rule.get("immutable", True):
            if pd.isna(query_value):
                if not pd.isna(candidate_value):
                    immutable_valid = False
                    violations.append("immutable_violation:%s" % feature)
            elif pd.isna(candidate_value) or not np.isclose(float(candidate_value), float(query_value), atol=1e-9, rtol=0):
                immutable_valid = False
                violations.append("immutable_violation:%s" % feature)
        else:
            lower = float(rule["lower"])
            upper = float(rule["upper"])
            if float(candidate_value) < lower - 1e-9 or float(candidate_value) > upper + 1e-9:
                bounds_valid = False
                violations.append("numeric_bound_violation:%s" % feature)
            if rule.get("direction") == "increase_only" and float(candidate_value) < float(query_value) - 1e-9:
                direction_valid = False
                violations.append("direction_violation:%s" % feature)
            if rule.get("integer") and not np.isclose(float(candidate_value), round(float(candidate_value)), atol=1e-9):
                bounds_valid = False
                violations.append("numeric_step_violation:%s" % feature)
    try:
        prediction = int(bundle.predict_raw(raw_candidate)[0])
        target_valid = prediction == 1
    except Exception:
        target_valid = False
        violations.append("schema_violation:model_input")
    valid = bool(target_valid)
    feasible = bool(valid and categorical_valid and immutable_valid and bounds_valid and direction_valid)
    return CandidateVerification(
        raw_reconstruction_ok=True,
        categorical_domain_valid=categorical_valid,
        immutable_valid=immutable_valid,
        bounds_valid=bounds_valid,
        direction_valid=direction_valid,
        target_valid=target_valid,
        valid=valid,
        constraint_feasible=feasible,
        violations=sorted(set(violations)),
        raw_candidate=row.to_dict(),
    )
