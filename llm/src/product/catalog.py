from __future__ import annotations

from typing import Any, Callable

from ufce.core.data_processing import (
    get_bank_user_constraints,
    get_bupa_user_constraints,
    get_grad_user_constraints,
    get_movie_user_constraints,
    get_wine_user_constraints,
)


DATASET_DISPLAY_NAMES = {
    "bank": "Bank Personal Loan",
    "grad": "Graduate Admission",
    "wine": "Wine Quality",
    "bupa": "BUPA Liver Disorders",
    "movie": "Movie Success",
}

DATASET_SUPPORT_NOTES = {
    "bank": "Live conversational runtime is enabled for the bank dataset in this MVP.",
    "grad": "Live conversational runtime is enabled for the graduate-admission dataset in this MVP.",
    "wine": "Dataset bundle is available locally for reference, but live conversational runtime is blocked in this MVP.",
    "bupa": "Dataset bundle is available locally for reference, but live conversational runtime is blocked in this MVP.",
    "movie": "Dataset bundle is available locally for reference, but live conversational runtime is blocked in this MVP.",
}

BANK_BOOLEAN_FIELDS = {"SecuritiesAccount", "CDAccount", "Online", "CreditCard"}

CONSTRAINT_LOADERS: dict[str, Callable[[Any], tuple[Any, ...]]] = {
    "bank": get_bank_user_constraints,
    "grad": get_grad_user_constraints,
    "wine": get_wine_user_constraints,
    "bupa": get_bupa_user_constraints,
    "movie": get_movie_user_constraints,
}

HOME_DATASET_ORDER = {
    "bank": 0,
    "grad": 1,
    "bupa": 2,
    "movie": 3,
    "wine": 4,
}


def build_dataset_catalog(*, runtime_orchestrator) -> list[dict[str, Any]]:
    model_registry = runtime_orchestrator.model_registry
    policy_registry = runtime_orchestrator.policy_registry
    entries: list[dict[str, Any]] = []
    for dataset_name in model_registry.datasets():
        bundle = model_registry.get_bundle(dataset_name)
        manifest_entry = model_registry.get_manifest_entry(dataset_name)
        policy = policy_registry.get_policy(dataset_name) if policy_registry.has_enabled_policy(dataset_name) else None
        feature_metadata = load_feature_metadata(dataset_name=dataset_name, dataset_df=bundle.dataset_df.copy())
        availability_status = "active" if policy is not None and policy.runtime_enabled else "blocked"
        entries.append(
            {
                "dataset_key": dataset_name,
                "display_name": DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.title()),
                "availability_status": availability_status,
                "support_note": DATASET_SUPPORT_NOTES.get(
                    dataset_name,
                    "Dataset bundle is available locally for reference only in this MVP.",
                ),
                "artifact_version": str(manifest_entry.get("artifact_version", "")),
                "training_logic_version": str(manifest_entry.get("training_logic_version", "")),
                "full_feature_list": list(feature_metadata["features"]),
                "f2change": list(feature_metadata["f2change"]),
                "outcome_label": str(feature_metadata["outcome_label"]),
                "desired_outcome": float(feature_metadata["desired_outcome"]),
                "step_provenance": (
                    policy.step_provenance
                    if policy is not None
                    else "Copied from legacy UFCE core assumptions in ufce/core/data_processing.py."
                ),
                "feature_guides": build_feature_guides(
                    dataset_name=dataset_name,
                    features=list(feature_metadata["features"]),
                    categorical_features=set(feature_metadata["categorical_features"]),
                    f2change=set(feature_metadata["f2change"]),
                    step_map=dict(feature_metadata["step"]),
                    aliases={}
                    if policy is None
                    else {key: list(values) for key, values in policy.conversation_aliases.items()},
                    dataset_df=bundle.dataset_df,
                    availability_status=availability_status,
                ),
            }
        )
        if availability_status != "active":
            entries[-1]["support_note"] = (
                "Dataset bundle is available locally for reference, but live conversational runtime is blocked in this MVP."
            )
    return sorted(
        entries,
        key=lambda item: (
            item["availability_status"] != "active",
            HOME_DATASET_ORDER.get(item["dataset_key"], 999),
            item["dataset_key"],
        ),
    )


def load_feature_metadata(*, dataset_name: str, dataset_df) -> dict[str, Any]:
    if dataset_name not in CONSTRAINT_LOADERS:
        raise KeyError(f"Unsupported dataset for catalog metadata: {dataset_name}")
    (
        features,
        categorical_features,
        _numeric_features,
        _uf,
        f2change,
        outcome_label,
        desired_outcome,
        _nbr_features,
        _protected_features,
        _data_lab0,
        _data_lab1,
    ) = CONSTRAINT_LOADERS[dataset_name](dataset_df)
    step_map = load_step_map(dataset_name=dataset_name)
    return {
        "features": [str(feature) for feature in features],
        "categorical_features": [str(feature) for feature in categorical_features],
        "f2change": [str(feature) for feature in f2change],
        "outcome_label": str(outcome_label),
        "desired_outcome": float(desired_outcome),
        "step": step_map,
    }


def load_step_map(*, dataset_name: str) -> dict[str, Any]:
    # Step maps are copied from the legacy UFCE helpers for catalog display.
    if dataset_name == "bank":
        return {
            "Income": 1,
            "CCAvg": 0.1,
            "Family": 1,
            "Education": 1,
            "Mortgage": 1,
            "CDAccount": 1,
            "Online": 1,
            "SecuritiesAccount": 1,
            "CreditCard": 1,
        }
    if dataset_name == "grad":
        return {
            "GRE Score": 1,
            "TOEFL Score": 1,
            "University Rating": 1,
            "SOP": 1,
            "LOR": 1,
            "CGPA": 0.1,
            "Research": 1,
        }
    if dataset_name == "wine":
        return {
            "fixed acidity": 0.5,
            "residual sugar": 0.5,
            "free sulfur dioxide": 1.0,
            "total sulfur dioxide": 1.0,
            "pH": 0.5,
            "alcohol": 0.5,
            "density": 0.1,
            "volatile acidity": 0.1,
            "citric acid": 0.1,
        }
    if dataset_name == "bupa":
        return {
            "Mcv": 1,
            "Alkphos": 1,
            "Sgpt": 1,
            "Sgot": 1,
            "Gammagt": 1,
            "Drinks": 1,
        }
    if dataset_name == "movie":
        return {
            "Production_expense": 5,
            "Num_multiplex": 5,
            "Multiplex_coverage": 0.2,
            "Movie_length": 10,
            "Lead_Actor_Rating": 1.0,
            "Lead_Actress_rating": 1.0,
            "Director_rating": 1.0,
            "Producer_rating": 1.0,
            "Genre": 1,
            "Collection": 5000,
        }
    return {}


def build_feature_guides(
    *,
    dataset_name: str,
    features: list[str],
    categorical_features: set[str],
    f2change: set[str],
    step_map: dict[str, Any],
    aliases: dict[str, list[str]],
    dataset_df,
    availability_status: str,
) -> list[dict[str, Any]]:
    guides: list[dict[str, Any]] = []
    for feature_name in features:
        feature_kind = infer_feature_kind(
            dataset_name=dataset_name,
            feature_name=feature_name,
            categorical_features=categorical_features,
            dataset_df=dataset_df,
        )
        changeable = feature_name in f2change
        step_value = step_map.get(feature_name)
        alias_values = list(aliases.get(feature_name) or [])
        guides.append(
            {
                "feature_name": feature_name,
                "feature_kind": feature_kind,
                "changeable": changeable,
                "step": step_value,
                "aliases": alias_values,
                "definition": build_feature_definition(
                    dataset_name=dataset_name,
                    feature_name=feature_name,
                    feature_kind=feature_kind,
                ),
                "check_guidance": build_check_guidance(
                    feature_name=feature_name,
                    feature_kind=feature_kind,
                    aliases=alias_values,
                    availability_status=availability_status,
                ),
                "change_guidance": build_change_guidance(
                    feature_name=feature_name,
                    changeable=changeable,
                    step_value=step_value,
                    availability_status=availability_status,
                ),
            }
        )
    return guides


def infer_feature_kind(
    *,
    dataset_name: str,
    feature_name: str,
    categorical_features: set[str],
    dataset_df,
) -> str:
    if dataset_name == "bank" and feature_name in BANK_BOOLEAN_FIELDS:
        return "binary"
    if feature_name in categorical_features:
        values = set(dataset_df[feature_name].dropna().astype(int).tolist())
        if values and values.issubset({0, 1}):
            return "binary"
        return "categorical"
    return "numeric"


def build_feature_definition(*, dataset_name: str, feature_name: str, feature_kind: str) -> str:
    if dataset_name == "bank":
        bank_definitions = {
            "Income": "Target annual income value used by the bank recourse model.",
            "Family": "Target household size used by the bank recourse model.",
            "CCAvg": "Target average credit card spending used by the bank recourse model.",
            "Education": "Target education level code used by the bank recourse model.",
            "Mortgage": "Target mortgage value used by the bank recourse model.",
            "SecuritiesAccount": "Whether the target profile includes a securities account.",
            "CDAccount": "Whether the target profile includes a certificate-of-deposit account.",
            "Online": "Whether the target profile uses online banking.",
            "CreditCard": "Whether the target profile includes a credit card.",
        }
        return bank_definitions.get(feature_name, f"Bank feature '{feature_name}' used by the legacy UFCE model.")
    return (
        f"Dataset feature '{feature_name}' from the legacy UFCE {dataset_name} bundle. "
        f"Current catalog type: {feature_kind}."
    )


def build_check_guidance(
    *,
    feature_name: str,
    feature_kind: str,
    aliases: list[str],
    availability_status: str,
) -> str:
    alias_note = ""
    if aliases:
        alias_note = " Recognized bank aliases: " + ", ".join(aliases) + "."
    if availability_status != "active":
        return (
            f"Reference only in this MVP. To inspect it, review the feature name '{feature_name}' in the catalog."
            + alias_note
        )
    if feature_kind == "binary":
        return (
            f"State the feature explicitly with yes/no, for example '{feature_name} yes' or '{feature_name} no'."
            + alias_note
        )
    return (
        f"State the feature explicitly with a value, for example '{feature_name} 55'."
        + alias_note
    )


def build_change_guidance(
    *,
    feature_name: str,
    changeable: bool,
    step_value: Any,
    availability_status: str,
) -> str:
    if availability_status != "active":
        return "Live changes for this dataset are blocked in this MVP. This metadata is informational only."
    if not changeable:
        return f"Current UFCE core does not mark '{feature_name}' as directly changeable."
    if step_value is None:
        return f"Ask to set '{feature_name}' to a new target value. No explicit UFCE step value is recorded."
    return (
        f"Ask to set '{feature_name}' to a new target value. Current UFCE step guidance is {step_value} per change."
    )
