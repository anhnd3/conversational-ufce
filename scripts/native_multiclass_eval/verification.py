"""Independent raw-space candidate and constraint validation."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Verification:
    schema_valid: bool
    target_valid: bool
    categorical_domain_valid: bool
    immutable_valid: bool
    bounds_valid: bool
    invariant_valid: bool
    constraint_feasible: bool
    prediction: object
    violations: list
    raw_candidate: dict

    def as_dict(self):
        return asdict(self)


def _raw_frame(candidate, schema):
    names = schema["feature_names"]
    if isinstance(candidate, pd.Series):
        candidate = candidate.to_frame().T
    if isinstance(candidate, dict):
        candidate = pd.DataFrame([candidate])
    if not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return None
    if any(name not in candidate.columns for name in names):
        return None
    return candidate.loc[:, names].iloc[[0]].copy()


def verify_candidate(factual, candidate, bundle, schema, query_policy, source_class, target_class):
    names = schema["feature_names"]
    raw = _raw_frame(candidate, schema)
    if raw is None:
        return Verification(False, False, False, False, False, False, False, None, ["schema_violation"], {})
    factual_row = factual.iloc[0] if isinstance(factual, pd.DataFrame) else pd.Series(factual)
    candidate_row = raw.iloc[0]
    violations = []
    try:
        prediction = int(bundle.predict_raw(raw)[0])
        target_valid = prediction == int(target_class)
        if not target_valid:
            violations.append("target_class_violation")
    except Exception:
        prediction = None
        target_valid = False
        violations.append("schema_violation:model_input")

    categorical = set(schema["categorical_features"])
    categorical_valid = True
    for feature in categorical:
        allowed = set(query_policy["categorical_domains"].get(feature, []))
        value = str(candidate_row[feature])
        if value not in allowed:
            categorical_valid = False
            violations.append("categorical_domain_violation:%s" % feature)

    immutable_valid = True
    for feature in query_policy["immutable_features"]:
        left, right = candidate_row[feature], factual_row[feature]
        if feature in categorical:
            same = str(left) == str(right)
        else:
            try:
                same = bool(np.isclose(float(left), float(right), atol=1e-9, rtol=0))
            except (TypeError, ValueError):
                same = False
        if not same:
            immutable_valid = False
            violations.append("immutable_violation:%s" % feature)

    bounds_valid = True
    for feature, rule in query_policy["features"].items():
        try:
            value = float(candidate_row[feature])
        except (TypeError, ValueError):
            bounds_valid = False
            violations.append("numeric_value_violation:%s" % feature)
            continue
        if value < float(rule["lower"]) - 1e-9 or value > float(rule["upper"]) + 1e-9:
            bounds_valid = False
            violations.append("numeric_bound_violation:%s" % feature)
        if rule["integer"] and not np.isclose(value, round(value), atol=1e-9, rtol=0):
            bounds_valid = False
            violations.append("numeric_integer_violation:%s" % feature)
        if value < 0:
            bounds_valid = False
            violations.append("numeric_domain_violation:%s" % feature)
        if feature.endswith("(grade)") and not (0 <= value <= 20):
            bounds_valid = False
            violations.append("grade_domain_violation:%s" % feature)

    invariant_valid = True
    for invariant_name in query_policy["invariants_enforced"]:
        parts = invariant_name.split("_")
        semester = "1st sem" if "1st" in parts else "2nd sem"
        approved = float(candidate_row["Curricular units %s (approved)" % semester])
        without = float(candidate_row["Curricular units %s (without evaluations)" % semester])
        enrolled = float(candidate_row["Curricular units %s (enrolled)" % semester])
        if invariant_name.startswith("approved_plus_without_eval"):
            passed = approved + without <= enrolled
        elif invariant_name.startswith("approved_le_enrolled"):
            passed = approved <= enrolled
        elif invariant_name.startswith("approved_le_evaluations"):
            evaluations = float(candidate_row["Curricular units %s (evaluations)" % semester])
            passed = approved <= evaluations
        elif invariant_name.startswith("without_eval_le_enrolled"):
            passed = without <= enrolled
        else:
            passed = False
        if not passed:
            invariant_valid = False
            violations.append("invariant_violation:%s" % invariant_name)

    feasible = bool(
        target_valid and categorical_valid and immutable_valid and bounds_valid and invariant_valid
    )
    return Verification(
        schema_valid=True,
        target_valid=target_valid,
        categorical_domain_valid=categorical_valid,
        immutable_valid=immutable_valid,
        bounds_valid=bounds_valid,
        invariant_valid=invariant_valid,
        constraint_feasible=feasible,
        prediction=prediction,
        violations=sorted(set(violations)),
        raw_candidate=candidate_row.to_dict(),
    )
