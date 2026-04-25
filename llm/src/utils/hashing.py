from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm.src.utils.time import local_now_compact


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id() -> str:
    return "run_" + local_now_compact()


def strip_run_prefix(run_id: str) -> str:
    if run_id.startswith("run_"):
        return run_id[4:]
    return run_id


def slugify_model_alias(model_alias: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_alias.strip())
    return slug.strip("_") or "model"


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)
