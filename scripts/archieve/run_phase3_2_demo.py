#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.product.app import create_app
from llm.src.product.config import ProductConfig


def main() -> int:
    config = ProductConfig.load()
    app = create_app(config=config)
    health = app.state.service.health()
    if health["status"] != "healthy":
        print(json.dumps({"startup_status": "unhealthy", "health": health}, indent=2))
        return 1
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
