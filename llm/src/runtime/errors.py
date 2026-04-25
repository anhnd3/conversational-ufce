from __future__ import annotations


class RuntimeServiceError(Exception):
    def __init__(self, reason_codes: tuple[str, ...], message: str) -> None:
        super().__init__(message)
        self.reason_codes = tuple(reason_codes)
        self.message = message
