from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from llm.src.conversation.parser_adapter import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SCHEMA_PATH,
)
from llm.src.runtime.policy_registry import BANK_POLICY_VERSION
from llm.src.runtime.reproducibility import RUNTIME_MODE_STABLE_DEMO


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase3_2_product"
DEFAULT_ARTIFACT_ROOT = DEFAULT_OUTPUT_ROOT / "artifacts"
DEFAULT_SQLITE_PATH = DEFAULT_OUTPUT_ROOT / "sessions.sqlite3"
DEFAULT_API_VERSION = "v1"
DEFAULT_APP_VERSION = "phase3_2_mvp_v1"


@dataclass(frozen=True)
class ProductConfig:
    lm_studio_api_base: str
    model_alias: str
    product_mode: str
    artifact_root: Path
    sqlite_path: Path
    api_version: str
    app_version: str
    parser_schema_version: str
    bank_policy_version: str
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def load(cls) -> "ProductConfig":
        load_dotenv()
        artifact_root = Path(os.getenv("ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))).expanduser().resolve()
        sqlite_path = Path(os.getenv("SQLITE_PATH", str(DEFAULT_SQLITE_PATH))).expanduser().resolve()
        return cls(
            lm_studio_api_base=os.getenv("LM_STUDIO_API_BASE", DEFAULT_API_BASE).rstrip("/"),
            model_alias=os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
            product_mode=os.getenv("PRODUCT_MODE", RUNTIME_MODE_STABLE_DEMO),
            artifact_root=artifact_root,
            sqlite_path=sqlite_path,
            api_version=os.getenv("API_VERSION", DEFAULT_API_VERSION),
            app_version=os.getenv("APP_VERSION", DEFAULT_APP_VERSION),
            parser_schema_version=DEFAULT_SCHEMA_PATH.stem,
            bank_policy_version=BANK_POLICY_VERSION,
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")),
        )


def try_get_git_commit(root: Path = ROOT) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None
