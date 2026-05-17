#!/usr/bin/env python3
"""
Thesis-level final evidence closeout orchestrator.
Runs Part I closeout, Part II closeout, and optionally product validation.

This is the single-command entrypoint for generating complete thesis evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def local_timezone() -> str:
    now = datetime.now().astimezone()
    return now.tzname() or str(now.utcoffset())


def run_thesis_closeout(*, out_dir: Path, skip_product: bool = False) -> Dict[str, Any]:
    """Run complete thesis evidence closeout."""
    
    run_id = "all_closeout_" + datetime.now().strftime("%Y%m%d%H%M%S")
    run_root = out_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    
    part1_out = run_root / "part1"
    part2_out = run_root / "part2"
    product_out = run_root / "product" if not skip_product else None
    
    stage_results: list[dict[str, Any]] = []
    
    # Part I Closeout
    print("[THESIS_CLOSEOUT] Running Part I closeout...")
    part1_cmd = [
        sys.executable,
        str(ROOT / "scripts/final/part1/99_part1_closeout.py"),
        "--out-dir", str(part1_out),
    ]
    
    completed = subprocess.run(
        part1_cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    
    # Try to parse Part I summary if it exists
    part1_summary_path = list(part1_out.glob("*/part1_closeout_summary.json"))[0] if any(part1_out.glob("*")) else None
    
    stage_results.append({
        "name": "part1_closeout",
        "command": " ".join(subprocess.list2cmdline(part1_cmd)),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "summary_json_path": str(part1_summary_path.resolve()) if part1_summary_path else None,
        "stdout_tail": tail_text(completed.stdout),
    })
    
    # Part II Closeout  
    print("[THESIS_CLOSEOUT] Running Part II closeout...")
    part2_cmd = [
        sys.executable,
        str(ROOT / "scripts/final/part2/99_part2_closeout.py"),
        "--out-dir", str(part2_out),
    ]
    
    completed = subprocess.run(
        part2_cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    
    # Try to parse Part II summary if it exists  
    part2_summary_path = list(part2_out.glob("*/part2_closeout_bundle_summary.json"))[0] if any(part2_out.glob("*")) else None
    
    stage_results.append({
        "name": "part2_closeout",
        "command": " ".join(subprocess.list2cmdline(part2_cmd)),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "summary_json_path": str(part2_summary_path.resolve()) if part2_summary_path else None,
        "stdout_tail": tail_text(completed.stdout),
    })
    
    # Product validation (optional)
    if not skip_product:
        print("[THESIS_CLOSEOUT] Running product acceptance...")
        
        # Check if server is already running or can be started
        import socket
        
        def check_port(port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port)) == 0
            sock.close()
            return result
            
        server_running = check_port(8000)
        
        if not server_running:
            print("[THESIS_CLOSEOUT] Warning: Product server not running on port 8000. Skipping product acceptance.")
            stage_results.append({
                "name": "product_acceptance",
                "passed": False,
                "skipped": True,
                "note": "Product server not available on port 8000",
            })
        else:
            # Run smoke tests
            product_cmd = [
                sys.executable,
                str(ROOT / "scripts/final/product/02_product_smoke.py"),
                "--base-url", "http://127.0.0.1:8000",
                "--out-dir", str(product_out),
            ]
            
            completed = subprocess.run(
                product_cmd,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            
            stage_results.append({
                "name": "product_smoke",
                "command": " ".join(subprocess.list2cmdline(product_cmd)),
                "exit_code": completed.returncode,
                "passed": completed.returncode == 0,
                "stdout_tail": tail_text(completed.stdout),
            })
            
    # Determine overall pass/fail
    critical_passed = all(item["passed"] for item in stage_results if not item.get("skipped"))
    
    timestamp = datetime.now().isoformat()
    
    summary = {
        "run_id": run_id,
        "timestamp_local": timestamp,
        "timezone": local_timezone(),
        "runner_scope": "all_closeout",
        "bundle_root": str(run_root.resolve()),
        "part1_output_root": str(part1_out.resolve()),
        "part2_output_root": str(part2_out.resolve()),
        "product_skipped": skip_product or not server_running,
        "stage_results": stage_results,
        "closeout_passed": critical_passed,
    }
    
    # Write summary JSON
    with open(run_root / "all_closeout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Write markdown
    md_lines = [
        "# Thesis Final Evidence Closeout", "",
        f"- **run_id:** `{summary['run_id']}`",
        f"- **timestamp_local:** `{summary['timestamp_local']}`",
        "", "## Part I Results", ""
    ]
    
    for item in stage_results:
        status = "PASS" if item.get("passed") else ("SKIP" if item.get("skipped") else "FAIL")
        md_lines.append(f"- **{item['name']}**: `{status}` (exit_code=`{item.get('exit_code', 'N/A')}`)")
        
    md_lines.extend([
        "", "## Verdict", "",
        f"- **closeout_passed:** `{critical_passed}`",
        ""
    ])
    
    with open(run_root / "all_closeout_summary.md", "w") as f:
        f.write("\n".join(md_lines))
        
    return summary


def tail_text(text: str, limit: int = 20) -> list[str]:
    """Get last N non-empty lines."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Thesis-level final evidence closeout.")
    parser.add_argument("--out-dir", type=Path, 
                       default=ROOT / "outputs" / "final" / "all_closeout")
    parser.add_argument("--skip-product", action="store_true")
    
    args = parser.parse_args()
    
    print(f"[THESIS_CLOSEOUT] Starting full thesis closeout...")
    summary = run_thesis_closeout(out_dir=args.out_dir, skip_product=args.skip_product)
    
    print(json.dumps(summary, indent=2))
    
    return 0 if summary["closeout_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
