from __future__ import annotations

from datetime import datetime, timedelta, timezone


UTC_PLUS_7 = timezone(timedelta(hours=7), name="UTC+07:00")


def local_now_iso() -> str:
    return datetime.now(UTC_PLUS_7).replace(microsecond=0).isoformat()


def local_now_compact() -> str:
    return datetime.now(UTC_PLUS_7).strftime("%Y%m%d_%H%M%S_%f")


def local_now_folder_timestamp() -> str:
    return datetime.now(UTC_PLUS_7).strftime("%Y-%m-%dT%H-%M-%S")
