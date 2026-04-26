from __future__ import annotations

from typing import Any

from llm.src.orchestration.user_response_payload import ChangeItem, NextAction, UserResponsePayload


def render_deterministic_user_response_text(
    payload: UserResponsePayload,
    *,
    dataset_label: str = "bank profile",
) -> str:
    kind = payload.response_kind

    if kind == "no_recourse_needed":
        return (
            f"Good news - your current {dataset_label} already reaches the desired outcome.\n\n"
            "No changes are needed right now. The runtime confirmed this profile already satisfies the target result."
        )

    if kind == "counterfactual_found":
        lines = ["A valid improvement path was found."]
        change_lines = _render_changed_items(payload.changed_items)
        if change_lines:
            lines.append("")
            lines.append("To reach the desired outcome, the checked recommendation changes:")
            lines.extend(change_lines)
        lines.append("")
        lines.append("These changes passed validation and stayed within the active constraints.")
        return "\n".join(lines)

    if kind == "runtime_reject_constraints_blocked":
        lines = [
            "I found candidate options, but none can be shown under your current constraints.",
            "",
            "The active constraints blocked the generated options. Review the fields you marked as fixed or the bounds you added.",
        ]
        blocked_fields = _blocked_fields(payload)
        if blocked_fields:
            lines.append("")
            lines.append("Blocked fields:")
            lines.extend(f"* {field}" for field in blocked_fields)
        lines.extend(_render_next_step_lines(payload.next_actions))
        return "\n".join(lines)

    if kind == "runtime_reject_invalid_counterfactual_blocked":
        lines = [
            "I could not safely show a recommendation for this request.",
            "",
            "A candidate was generated, but it failed post-generation validation checks, so it was blocked instead of being shown as advice.",
        ]
        lines.extend(_render_next_step_lines(payload.next_actions))
        return "\n".join(lines)

    if kind == "runtime_reject_no_feasible_cf":
        lines = [
            "I could not find a valid recommendation under the current profile and policy.",
            "",
            "No checked candidate reached the desired outcome while satisfying the current runtime rules.",
        ]
        lines.extend(_render_next_step_lines(payload.next_actions))
        return "\n".join(lines)

    if kind in {"clarification_required", "refinement_clarification"}:
        missing_fields = _string_list(payload.technical_facts.get("missing_fields"))
        carried_fields = _string_list(payload.technical_facts.get("carried_forward_fields"))
        conflict_items = _string_list(payload.technical_facts.get("conflicts"))
        if conflict_items:
            lines = [
                "I found conflicting instructions in this case.",
                "",
                "Conflicts:",
            ]
            lines.extend(f"* {item}" for item in conflict_items)
            lines.extend(
                [
                    "",
                    f"To avoid merging the wrong {dataset_label}, start a new corrected case with one clear value for each required field.",
                ]
            )
            return "\n".join(lines)

        lines = [f"I still need a few fields before I can run UFCE safely for this {dataset_label}."]
        if missing_fields:
            lines.extend(["", "Missing fields:"])
            lines.extend(f"* {field}" for field in missing_fields)
            lines.extend(
                [
                    "",
                    "You do not need to repeat values already provided. Reply only with the missing values, for example:",
                    _example_assignment_line(missing_fields),
                ]
            )
        if carried_fields:
            lines.extend(
                [
                    "",
                    "I will keep the values already provided for " + _format_list_with_and(carried_fields) + ".",
                ]
            )
        return "\n".join(lines)

    if kind == "clarification_limit_reached":
        return (
            "I could not complete this case within the allowed clarification rounds.\n\n"
            f"To avoid using an incomplete or inconsistent {dataset_label}, start a new case with one complete corrected profile."
        )

    if kind == "conflict":
        return (
            "I found conflicting instructions in this case.\n\n"
            f"Please start a new corrected case with one clear value for each required field in the {dataset_label}."
        )

    if kind == "unsupported_request":
        return (
            "I could not process this request in the current workflow.\n\n"
            f"Start a new case and provide one complete {dataset_label}."
        )

    if kind == "parser_failure":
        return (
            "I could not safely parse this request.\n\n"
            f"Start a new case and provide one complete {dataset_label} so I can continue."
        )

    summary = payload.short_summary.strip()
    if summary:
        return f"{payload.headline.strip()}\n\n{summary}"
    return payload.headline.strip()


def _render_changed_items(changed_items: list[ChangeItem]) -> list[str]:
    rows: list[str] = []
    for item in changed_items:
        if item.user_facing_text:
            rows.append(f"* {item.user_facing_text}")
            continue
        rows.append(
            "* {name}: {before} -> {after}".format(
                name=item.display_name,
                before=_format_value(item.before),
                after=_format_value(item.after),
            )
        )
    return rows


def _render_next_step_lines(next_actions: list[NextAction]) -> list[str]:
    if not next_actions:
        return []
    primary = [action for action in next_actions if action.primary]
    selected = primary[0] if primary else next_actions[0]
    return ["", f"Next step: {selected.detail}"]


def _blocked_fields(payload: UserResponsePayload) -> list[str]:
    values: list[str] = []
    for reason in payload.blocked_reasons:
        values.extend(_string_list(reason.fields))
    for effect in payload.constraint_effects:
        values.extend(_string_list(effect.affected_fields))
    deduped: list[str] = []
    seen: set[str] = set()
    for field in values:
        if field in seen:
            continue
        seen.add(field)
        deduped.append(field)
    return deduped


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and str(item).strip()]


def _format_list_with_and(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _example_assignment_line(missing_fields: list[str]) -> str:
    seed_values = ["42", "80", "3", "2", "120", "1", "0", "1", "0"]
    fragments: list[str] = []
    for index, field in enumerate(missing_fields[:3]):
        sample_value = seed_values[index] if index < len(seed_values) else "<value>"
        fragments.append(f"{field} = {sample_value}")
    return ", ".join(fragments) if fragments else "Field = value"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return ("{0:.6f}".format(value)).rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)
