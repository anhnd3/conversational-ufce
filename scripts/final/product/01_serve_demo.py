#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Serve final product demo API.")
    parser.add_argument("--host", type=str, default=None, help="Override host from ProductConfig")
    parser.add_argument("--port", type=int, default=None, help="Override port from ProductConfig")
    args = parser.parse_args()

    import uvicorn
    from llm.src.product.app import create_app
    from llm.src.product.config import ProductConfig

    config = ProductConfig.load()
    if args.host:
        config.host = args.host
    if args.port is not None:
        config.port = int(args.port)

    app = create_app(config=config)
    health = app.state.service.health()
    if health["status"] != "healthy":
        print(json.dumps({"startup_status": "unhealthy", "health": health}, indent=2))
        return 1
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
