#!/usr/bin/env python3
"""
Final self-contained parser metrics benchmark for Part II thesis evidence.
Runs structured extraction quality evaluation using LLM Studio client against Bank corpus.

Current sources: llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_parser_metrics(*, out_dir: Path, benchmark_path: Path = None,
                       lm_studio_api_base: str = None, model_alias: str = None) -> Dict[str, Any]:
    """Run parser benchmark metrics."""
    
    # Load config
    from llm.src.product.config import ProductConfig
    cfg = ProductConfig.load()
    
    api_base = lm_studio_api_base or os.getenv("LM_STUDIO_API_BASE", "http://127.0.0.1:1234")
    model_alias = model_alias or cfg.model_alias
    
    benchmark_path = benchmark_path or ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml"
    
    # Run LLM evaluation runner
    import subprocess
    
    command = [
        sys.executable,
        str(ROOT / "llm_eval/run_bank_cf_llm_eval.py"),
        "--benchmark", str(benchmark_path),
        "--lm-studio-api-base", api_base,
        "--model-alias", model_alias,
        "--out-dir", str(out_dir),
    ]
    
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    
    # Try to find the report JSON
    summary_json_path = None
    try:
        for f in sorted(out_dir.glob("*.json")):
            if "report" in str(f).lower() or "summary" in str(f).lower():
                summary_json_path = str(f.resolve())
                break
    except Exception:
        pass
    
    return {
        "name": "parser_metrics",
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "summary_json_path": summary_json_path,
        "stdout_tail": tail_text(completed.stdout),
    }


def tail_text(text: str, limit: int = 20) -> list[str]:
    """Get last N non-empty lines."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Final parser metrics benchmark.")
    parser.add_argument("--out-dir", type=Path, 
                       default=ROOT / "outputs" / "final" / "part2" / "parser_metrics")
    parser.add_argument("--benchmark", type=Path,
                       default=ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml")
    parser.add_argument("--lm-studio-api-base", 
                       default=os.getenv("LM_STUDIO_API_BASE", "http://127.0.0.1:1234"))
    parser.add_argument("--model-alias", default=None)
    
    args = parser.parse_args()
    
    result = run_parser_metrics(
        out_dir=args.out_dir,
        benchmark_path=args.benchmark,
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
    )
    
    print(json.dumps(result, indent=2))
    
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
