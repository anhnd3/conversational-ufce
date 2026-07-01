# Final Thesis Runbook

This runbook lists the GitHub-facing workflows kept for thesis methodology and
experiments. Numbered process scripts have been replaced by semantic script
names; historical/debug scripts remain local-only.

## Setup

```bash
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

For Part II parser runs, LM Studio must expose the configured model alias:

```bash
LM_STUDIO_API_BASE=http://127.0.0.1:1234
MODEL_ALIAS=qwen/qwen3-14b
```

## Part I: UFCE Reproduction And UFCE-FF

Full Table 7 reproduction with DiCE, DiCE-UF, AR, and UFCE baselines:

```bash
python scripts/final/part1/table7_full_reproduction.py --dataset all
```

UFCE-only locked reproduction under the thesis final-freeze profile:

```bash
python scripts/final/part1/ufce_only_reproduction.py --dataset all --runtime_profile final_freeze --bundle-mode table7_author_public --out-dir outputs/final/part1/ufce_only_final_freeze
```

Parameter tuning provenance for the final-freeze/tuned settings:

```bash
python scripts/final/part1/ufce_parameter_tuning.py --dataset all --run_stage all --out-dir outputs/final/part1/ufce_parameter_tuning
```

Original author-pool replay for raw, post-hoc-valid-only, and UFCE-FF selector views:

```bash
python scripts/final/part1/author_pool_selector_audit.py --dataset all --config-profile final_freeze --bundle-mode table7_author_public --mi-k 5 --no-cf 10 --emit-pairs --out-dir outputs/final/part1/author_pool_selector_audit
```

Standalone author raw/post-hoc replay:

```bash
python scripts/final/part1/author_raw_posthoc_replay.py --dataset all --bundle-mode table7_author_public --out-dir outputs/final/part1/author_raw_posthoc_replay
```

Red Wine proximity-space sensitivity diagnostic:

```bash
python scripts/final/part1/red_wine_proximity_sensitivity.py --out-dir outputs/final/part1/red_wine_proximity_sensitivity
```

UFCE-FF experiment runner for public-source, final-freeze, and new-best profile checks:

```bash
python scripts/final/part1/ufce_force_flip_experiment.py --dataset all --mode final_comparison --config-profile final_freeze --bundle-mode table7_author_public --out-dir outputs/final/part1/ufce_force_flip_experiment
```

## Part II: Natural-Language Feedback Prototype

Parser/model benchmark metrics for the local LLM comparison:

```bash
python scripts/final/part2/parser_benchmark_metrics.py --out-dir outputs/final/part2/parser_benchmark_metrics
```

Conversation quality metrics for Bank Loan sessions:

```bash
python scripts/final/part2/conversation_quality_metrics.py --out-dir outputs/final/part2/conversation_quality_metrics
```

No-valid-counterfactual diagnostics for safe-stop cases:

```bash
python scripts/final/part2/no_valid_counterfactual_diagnostics.py --out-dir outputs/final/part2/no_valid_counterfactual_diagnostics
```

Policy-control evaluation for constraint behavior:

```bash
python scripts/final/part2/policy_control_evaluation.py --out-dir outputs/final/part2/policy_control_evaluation
```

## Evidence Map

Use `docs/THESIS_TABLE_TO_SCRIPT_MAP.md` for the table-by-table mapping from
Chapter 4 numbers to scripts, configs, inputs, and expected artifacts.
