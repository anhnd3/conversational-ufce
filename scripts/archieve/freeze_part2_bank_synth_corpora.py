#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.corpora import (
    BANK_BOUNDARY_PROFILES_CORPUS_PATH,
    BANK_BOUNDARY_PROFILES_CORPUS_VERSION,
    G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH,
    G5_AGENT_PORTABILITY_SYNTH300_CORPUS_VERSION,
    TIER_B_SYNTH300_CORPUS_PATH,
    TIER_B_SYNTH300_CORPUS_VERSION,
    generate_bank_boundary_profiles_snapshot,
    generate_g5_agent_portability_synth300_corpus,
    generate_tier_b_bank_synth300_corpus,
    load_frozen_corpus,
)
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the repo-tracked Part II synthetic bank corpus snapshots.")
    parser.add_argument("--out-dir", type=Path, default=TIER_B_SYNTH300_CORPUS_PATH.parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = freeze_corpora(out_dir=args.out_dir)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def freeze_corpora(*, out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_registry = ModelRegistry()
    policy_registry = PolicyRegistry(model_registry)
    bank_bundle = model_registry.get_bundle("bank")
    desired_outcome = policy_registry.get_policy("bank").desired_outcome

    tier_b_payload = generate_tier_b_bank_synth300_corpus(bundle=bank_bundle, desired_outcome=desired_outcome)
    g5_payload = generate_g5_agent_portability_synth300_corpus(tier_b_corpus=tier_b_payload)
    boundary_payload = generate_bank_boundary_profiles_snapshot()

    targets = [
        (
            "tier_b_synth300",
            out_dir / TIER_B_SYNTH300_CORPUS_PATH.name,
            TIER_B_SYNTH300_CORPUS_VERSION,
            tier_b_payload,
        ),
        (
            "g5_synth300",
            out_dir / G5_AGENT_PORTABILITY_SYNTH300_CORPUS_PATH.name,
            G5_AGENT_PORTABILITY_SYNTH300_CORPUS_VERSION,
            g5_payload,
        ),
        (
            "boundary_profiles",
            out_dir / BANK_BOUNDARY_PROFILES_CORPUS_PATH.name,
            BANK_BOUNDARY_PROFILES_CORPUS_VERSION,
            boundary_payload,
        ),
    ]

    written: dict[str, Any] = {}
    for corpus_name, target_path, expected_version, payload in targets:
        write_json(target_path, payload)
        loaded = load_frozen_corpus(target_path, expected_version=expected_version)
        if loaded["corpus_sha256"] != payload["corpus_sha256"]:
            raise RuntimeError(
                f"Frozen corpus verification failed for {corpus_name}: "
                f"expected {payload['corpus_sha256']} got {loaded['corpus_sha256']}"
            )
        written[corpus_name] = {
            "path": str(target_path),
            "corpus_version": loaded["corpus_version"],
            "corpus_sha256": loaded["corpus_sha256"],
        }

    return {
        "out_dir": str(out_dir),
        "written_corpora": written,
    }


if __name__ == "__main__":
    raise SystemExit(main())
