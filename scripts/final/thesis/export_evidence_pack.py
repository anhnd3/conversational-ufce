#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import sys

FINAL_ROOT = Path(__file__).resolve().parents[1]
if str(FINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FINAL_ROOT))

from _common import ROOT, make_run_id, provenance, sha256_file, write_json, write_standard_outputs, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight final thesis evidence pack.")
    parser.add_argument("--part1-closeout", type=Path, required=True)
    parser.add_argument("--part2-closeout", type=Path, required=True)
    parser.add_argument("--product-closeout", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "final" / "evidence_pack")
    parser.add_argument("--include-raw", action="store_true")
    return parser.parse_args()


def selected_files(root: Path, *, include_raw: bool) -> Iterable[Path]:
    if include_raw:
        yield from (path for path in root.rglob("*") if path.is_file())
        return
    names = {
        "summary.json",
        "summary.md",
        "evidence_index.json",
        "evidence_index.md",
        "part2_closeout_bundle_summary.json",
        "part2_closeout_bundle_summary.md",
        "part1_closeout_summary.json",
        "part1_closeout_summary.md",
    }
    for path in root.rglob("*"):
        if path.is_file() and (path.name in names or path.suffix.lower() in {".md", ".json"} and "logs" not in path.parts):
            yield path


def copy_group(label: str, source_root: Path, selected_root: Path, *, include_raw: bool) -> list[dict[str, str]]:
    copied = []
    target_root = selected_root / label
    for source in selected_files(source_root, include_raw=include_raw):
        relative = source.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"source": str(source.resolve()), "target": str(target.resolve()), "sha256": sha256_file(target)})
    return copied


def main() -> int:
    args = parse_args()
    run_id = make_run_id("evidence_pack")
    pack_root = (args.out_dir / run_id).resolve()
    selected_root = pack_root / "selected_reports"
    selected_root.mkdir(parents=True, exist_ok=False)
    copied = []
    copied.extend(copy_group("part1", args.part1_closeout.resolve(), selected_root, include_raw=args.include_raw))
    copied.extend(copy_group("part2", args.part2_closeout.resolve(), selected_root, include_raw=args.include_raw))
    if args.product_closeout is not None:
        copied.extend(copy_group("product", args.product_closeout.resolve(), selected_root, include_raw=args.include_raw))
    hashes = {item["target"]: item["sha256"] for item in copied}
    write_json(pack_root / "hashes.json", hashes)
    manifest = {
        "run_id": run_id,
        "pack_root": str(pack_root),
        "include_raw": args.include_raw,
        "part1_closeout": str(args.part1_closeout.resolve()),
        "part2_closeout": str(args.part2_closeout.resolve()),
        "product_closeout": None if args.product_closeout is None else str(args.product_closeout.resolve()),
        "copied_files": copied,
    }
    write_json(pack_root / "MANIFEST.json", manifest)
    write_text(pack_root / "README.md", "# Final Thesis Evidence Pack\n\nThis pack contains selected lightweight summaries and hashes from final closeout outputs.\n")
    write_text(pack_root / "CLAIM_TO_EVIDENCE.md", "See `docs/FINAL_CLAIM_TO_EVIDENCE_MAP.md` in the repository root.\n")
    write_text(pack_root / "PART1_SUMMARY.md", "See `selected_reports/part1`.\n")
    write_text(pack_root / "PART2_SUMMARY.md", "See `selected_reports/part2`.\n")
    if args.product_closeout is not None:
        write_text(pack_root / "PRODUCT_SUMMARY.md", "See `selected_reports/product`.\n")
    summary = provenance(
        run_id=run_id,
        run_root=pack_root,
        title="Final Thesis Evidence Pack Export",
        input_paths=[args.part1_closeout, args.part2_closeout, *( [args.product_closeout] if args.product_closeout is not None else [] )],
        warnings=[] if args.include_raw else ["Raw/heavy outputs were excluded. Re-run with --include-raw to copy all files."],
    )
    summary.update({"status": "passed", "passed": True, "manifest_path": str((pack_root / "MANIFEST.json").resolve()), "hashes_path": str((pack_root / "hashes.json").resolve()), "copied_file_count": len(copied), "results": []})
    write_standard_outputs(run_root=pack_root, summary=summary)
    print(pack_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

