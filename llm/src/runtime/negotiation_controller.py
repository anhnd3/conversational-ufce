from __future__ import annotations


READY_FOR_PREDICTION = "READY_FOR_PREDICTION"
PREDICTION_COMPLETE = "PREDICTION_COMPLETE"
READY_FOR_UFCE = "READY_FOR_UFCE"
UFCE_SUCCESS = "UFCE_SUCCESS"
UFCE_INFEASIBLE = "UFCE_INFEASIBLE"
TERMINAL_SUCCESS = "TERMINAL_SUCCESS"
TERMINAL_REJECT = "TERMINAL_REJECT"


ALLOWED_TRANSITIONS = {
    None: {READY_FOR_PREDICTION, TERMINAL_REJECT},
    READY_FOR_PREDICTION: {PREDICTION_COMPLETE, TERMINAL_REJECT},
    PREDICTION_COMPLETE: {READY_FOR_UFCE, TERMINAL_SUCCESS, TERMINAL_REJECT},
    READY_FOR_UFCE: {UFCE_SUCCESS, UFCE_INFEASIBLE, TERMINAL_REJECT},
    UFCE_SUCCESS: {TERMINAL_SUCCESS},
    UFCE_INFEASIBLE: {TERMINAL_REJECT},
    TERMINAL_SUCCESS: set(),
    TERMINAL_REJECT: set(),
}


class NegotiationController:
    def __init__(self) -> None:
        self.current_state: str | None = None
        self.state_trace: list[str] = []

    def transition(self, next_state: str) -> None:
        allowed = ALLOWED_TRANSITIONS.get(self.current_state, set())
        if next_state not in allowed:
            raise ValueError(
                "Invalid controller transition from {0!r} to {1!r}.".format(
                    self.current_state,
                    next_state,
                )
            )
        self.current_state = next_state
        self.state_trace.append(next_state)
