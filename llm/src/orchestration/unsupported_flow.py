from __future__ import annotations


def render_unsupported_request_text(
    *,
    required_fields: list[str],
    dataset_label: str = "bank profile",
    dataset_id: str = "bank",
    unsupported_intent_type: str | None = None,
    requested_dataset_label: str | None = None,
) -> str:
    if unsupported_intent_type == "dataset_switch":
        if requested_dataset_label:
            return (
                f"This session is locked to the active {dataset_label}. "
                f"To use the {requested_dataset_label}, start a new session and choose that dataset first."
            )
        return (
            f"This session is locked to the active {dataset_label}. "
            "Start a new session and choose the desired dataset first."
        )

    ordered_fields = ", ".join(required_fields)
    return (
        "This conversational MVP only supports requests that provide or clarify a target "
        f"{dataset_label} using the active {dataset_id} fields. "
        "Unsupported requests include other datasets, open-ended advice, or optimization asks "
        f"outside the current {dataset_label} contract. "
        f"Please start a new case and submit one complete {dataset_label} using these fields: "
        f"{ordered_fields}."
    )
