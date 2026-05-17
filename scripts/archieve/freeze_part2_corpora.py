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
    G5_AGENT_PORTABILITY_CORPUS_PATH,
    G5_AGENT_PORTABILITY_CORPUS_VERSION,
    TIER_A_CORPUS_PATH,
    TIER_A_CORPUS_VERSION,
    TIER_B_CORPUS_PATH,
    TIER_B_CORPUS_VERSION,
    TIER_C_CORPUS_PATH,
    TIER_C_CORPUS_VERSION,
    TIER_D_CORPUS_PATH,
    TIER_D_CORPUS_VERSION,
    VALIDATION_CORPORA_ROOT,
    generate_tier_a_annotation_corpus,
    generate_tier_b_bank_corpus,
    generate_tier_c_bank_backend_corpus,
    generate_tier_d_bank_replay_corpus,
    generate_g5_agent_portability_corpus,
    load_frozen_corpus,
)
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the repo-tracked Part II corpus snapshots.")
    parser.add_argument("--out-dir", type=Path, default=VALIDATION_CORPORA_ROOT)
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
    tier_b_payload = generate_tier_b_bank_corpus(bundle=bank_bundle, desired_outcome=desired_outcome)

    targets = [
        ("tier_a", out_dir / TIER_A_CORPUS_PATH.name, TIER_A_CORPUS_VERSION, generate_tier_a_annotation_corpus()),
        (
            "tier_b",
            out_dir / TIER_B_CORPUS_PATH.name,
            TIER_B_CORPUS_VERSION,
            tier_b_payload,
        ),
        (
            "tier_c",
            out_dir / TIER_C_CORPUS_PATH.name,
            TIER_C_CORPUS_VERSION,
            generate_tier_c_bank_backend_corpus(bundle=bank_bundle, desired_outcome=desired_outcome),
        ),
        (
            "tier_d",
            out_dir / TIER_D_CORPUS_PATH.name,
            TIER_D_CORPUS_VERSION,
            generate_tier_d_bank_replay_corpus(bundle=bank_bundle, desired_outcome=desired_outcome),
        ),
        (
            "g5_agent_portability",
            out_dir / G5_AGENT_PORTABILITY_CORPUS_PATH.name,
            G5_AGENT_PORTABILITY_CORPUS_VERSION,
            generate_g5_agent_portability_corpus(tier_b_corpus=tier_b_payload),
        ),
    ]

    written: dict[str, Any] = {}
    for tier_name, target_path, expected_version, payload in targets:
        write_json(target_path, payload)
        loaded = load_frozen_corpus(target_path, expected_version=expected_version)
        if loaded["corpus_sha256"] != payload["corpus_sha256"]:
            raise RuntimeError(
                f"Frozen corpus verification failed for {tier_name}: "
                f"expected {payload['corpus_sha256']} got {loaded['corpus_sha256']}"
            )
        written[tier_name] = {
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
