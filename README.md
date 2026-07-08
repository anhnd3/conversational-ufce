# UFCE Agent

Code and experiment runners for a thesis on UFCE reproduction and a bank-loan natural-language interface to UFCE.

The repository has two thesis-facing experiment tracks:

- Route 1: UFCE reproduction, post-hoc audit, and UFCE-FF under the locked final configuration.
- Route 2: bank-profile natural-language parsing and the NL -> UFCE-FF bridge evaluation on the 250-case Bank Loan set derived from the 5 original test folds.

The deterministic UFCE backend remains the authority. The LLM is used only for structured parsing.

Step-by-step commands are in `docs/FINAL_RUNBOOK.md`.
Claim-level provenance is in `docs/FINAL_CLAIM_TO_EVIDENCE_MAP.md`.

## Scope

- Live conversational runtime: bank only
- Recommended runtime: Linux or WSL
- Python version: `3.8.20`
- Part II parser endpoint: `http://127.0.0.1:1234`
- Default Part II parser model alias: `qwen3-14b`

## Repository Layout

- `scripts/final/part1/`: Route 1 runners
- `scripts/final/part2/`: Route 2 runners
- `scripts/final/thesis/`: thesis utility scripts
- `scripts/final/product/`: optional local demo
- `llm/`: parser, validation, runtime, and product code
- `llm_eval/`: parser benchmark support
- `ufce/`: UFCE cores and datasets
- `outputs/`: generated artifacts, ignored and local-only
- `docs/`: runbook, evidence map, and thesis-facing notes

## Publication Boundary

The GitHub-facing repo keeps source, small frozen inputs, tests, and compact
thesis-facing documentation. Generated outputs, logs, temporary workspaces,
archived process notes, and exploratory thesis drafts stay local-only through
`.gitignore`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/final/thesis/doctor.py --out-dir outputs/final/doctor
```

## Canonical Pipelines

### Route 1: UFCE -> post-hoc -> UFCE-FF

Recommended order:

1. Author raw reproduction from `ufce/core_author`
2. Thesis raw/final-freeze reproduction from `ufce/core`
3. Post-hoc valid-only aggregation over the raw selected outputs
4. UFCE-FF validity-gated reproduction from `ufce/ufce_ff`

Representative commands:

```bash
source .venv/bin/activate

python scripts/final/part1/run_part1_final_bundle.py \
  --dataset all \
  --runtime-profile final_freeze \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/final_ufce_bundle

python scripts/final/part1/01_author_raw_reproduction.py \
  --dataset all \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/01_author_raw_reproduction

python scripts/final/part1/02_ufce_core_final_freeze.py \
  --dataset all \
  --runtime-profile final_freeze \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/02_ufce_core_final_freeze

python scripts/final/part1/03_ufce_post_hoc.py \
  --dataset all \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/03_ufce_post_hoc

python scripts/final/part1/04_ufce_ff.py \
  --dataset all \
  --runtime-profile final_freeze \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/04_ufce_ff
```

### Route 2: parser-only -> gate -> NL + UFCE-FF

Recommended order:

1. Parser smoke or unit check
2. Parser-only evaluation on all 250 NL Bank cases
3. Gate on parser quality and choose the evaluation scope
4. NL bridge or accepted-subset diagnostic against UFCE-FF

Parser smoke or unit check:

```bash
bash scripts/final/part2/test_nl_bank_parser_v2.sh
```

Parser-only evaluation on the 250-case Bank Loan set:

```bash
MODEL_ALIAS=qwen3-14b \
API_BASE=http://127.0.0.1:1234 \
TIMEOUT_S=600 \
OUT_DIR=outputs/final/part2/nl_bank_bridge_<run_id> \
bash scripts/final/part2/run_nl_bank_parse_only_v2.sh
```

This runner writes live progress to:

- `parse_progress.json`
- `parse_quality_progress.json`
- `run.log`

Parser quality gate before running downstream UFCE-FF stages:

Proceed only after a completed parser run has `error_rate < 0.10` and no missing
cases, API errors, schema errors, or verifier errors.

The strict all-case target remains `ready_for_runtime_count = 250`,
`exact_reconstruction_count = 250`, and `error_count = 0`. If that strict target
is not yet met, use `--parse-eval-scope accepted` and report the dropped parser
cases explicitly.

Strict bridge command against frozen UFCE-FF:

```bash
source .venv/bin/activate
python scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage all \
  --seed 42 \
  --cases-per-group 10 \
  --out-dir outputs/final/part2/nl_bank_bridge_<run_id>_strict \
  --model-alias qwen3-14b \
  --api-base http://127.0.0.1:1234 \
  --timeout-s 600 \
  --config-profile final_freeze \
  --bundle-mode table7_author_public \
  --mi-k 5 \
  --no-cf 10 \
  --contprox-metric euclidean \
  --parse-eval-scope accepted \
  --mi-feature-scope configured_actionable
```

Accepted-subset NL -> UFCE-FF evaluation after parser-only evaluation:

```bash
source .venv/bin/activate
PARSE_DIR=outputs/final/part2/nl_bank_bridge_<run_id>
ACCEPTED_DIR=outputs/final/part2/nl_bank_bridge_<run_id>_accepted
FULL250_DIR=outputs/final/part2/nl_bank_bridge_<run_id>_full250

python scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage direct \
  --parse-source-dir "$PARSE_DIR" \
  --out-dir "$ACCEPTED_DIR" \
  --config-profile final_freeze \
  --bundle-mode table7_author_public \
  --mi-k 5 \
  --no-cf 10 \
  --contprox-metric euclidean \
  --parse-eval-scope accepted \
  --mi-feature-scope configured_actionable \
  --model-alias qwen3-14b \
  --api-base http://127.0.0.1:1234 \
  --timeout-s 600

python scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage nl_first \
  --out-dir "$ACCEPTED_DIR" \
  --config-profile final_freeze \
  --bundle-mode table7_author_public \
  --mi-k 5 \
  --no-cf 10 \
  --contprox-metric euclidean \
  --parse-eval-scope accepted \
  --mi-feature-scope configured_actionable \
  --model-alias qwen3-14b \
  --api-base http://127.0.0.1:1234 \
  --timeout-s 600

python scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage compare \
  --out-dir "$ACCEPTED_DIR" \
  --config-profile final_freeze \
  --bundle-mode table7_author_public \
  --mi-k 5 \
  --no-cf 10 \
  --contprox-metric euclidean \
  --parse-eval-scope accepted \
  --mi-feature-scope configured_actionable \
  --model-alias qwen3-14b \
  --api-base http://127.0.0.1:1234 \
  --timeout-s 600

python scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage direct \
  --parse-source-dir "$PARSE_DIR" \
  --out-dir "$FULL250_DIR" \
  --config-profile final_freeze \
  --bundle-mode table7_author_public \
  --mi-k 5 \
  --no-cf 10 \
  --contprox-metric euclidean \
  --parse-eval-scope all \
  --mi-feature-scope configured_actionable \
  --model-alias qwen3-14b \
  --api-base http://127.0.0.1:1234 \
  --timeout-s 600

python scripts/final/part2/failed_cases_impact_check.py \
  --parse-out-dir "$PARSE_DIR" \
  --full250-out-dir "$FULL250_DIR" \
  --accepted-out-dir "$ACCEPTED_DIR"
```

## Reading Results

Route 1:

- raw or post-hoc or UFCE-FF comparisons: `outputs/final/part1/**/summary.json`
- selector-level comparisons: `author_pool_selector_summary.csv`

Route 2 parser-only:

- gate metric: `parse_acceptance.json`
- live progress metric: `parse_quality_progress.json`
- case-level status: `record_reconstruction.csv`
- failures: `parse_errors.jsonl`

Route 2 bridge:

- overall summary: `summary.json`
- headline table: `table_main_bridge.csv`
- group breakdown: `table_language_group.csv`
- mismatches: `mismatch_cases.csv`
- final-vs-full250 table: `impact_check_7_failed_cases/route2_final_vs_full250.md`
- failed-case impact: `impact_check_7_failed_cases/failed_cases_impact_summary.md`

## Result Snapshots

### Route 1 snapshot

Representative local Part I snapshot:

- Path: `outputs/final/part1/09_author_pool_finalfreeze_20260624`
- Dataset: Bank Loan
- Query count: `250`
- Config: `final_freeze`
- Bundle mode: `table7_author_public`

| Method | Raw selected | Post-hoc valid | UFCE-FF valid |
| --- | ---: | ---: | ---: |
| UFCE1 | 80 | 80 | 80 |
| UFCE2 | 250 | 20 | 44 |
| UFCE3 | 237 | 19 | 27 |

Source artifacts:

- `author_pool_selector_summary.csv`
- `author_pool_metric_summary.csv`

### Route 2 snapshot

Historical local bridge baseline:

- Path: `outputs/final/part2/nl_bank_bridge_v2_wordnum_20260702_201114`
- Query count: `250`
- Group layout: `G1..G5`, `50` cases each
- Parse-ready rate: `0.40`
- Exact reconstruction rate: `0.40`
- Strict outcome match rate: `0.40`
- Invalid release count: `0`

Group-level summary:

| Group | Parse-ready | End-to-end exact |
| --- | ---: | ---: |
| G1 | 1.00 | 1.00 |
| G2 | 0.00 | 0.00 |
| G3 | 1.00 | 1.00 |
| G4 | 0.00 | 0.00 |
| G5 | 0.00 | 0.00 |

This snapshot is useful as the pre-improvement baseline for the parser bottleneck.

Current completed parser snapshot:

- Path: `outputs/final/part2/nl_bank_bridge_20260704_200630`
- Completed cases: `250`
- Runtime-ready count: `247`
- Exact reconstruction count: `243`
- Parser error count: `7`
- Parser error rate: `0.028`
- Strict all-case target met: `no`
- Operational `< 0.10` gate met: `yes`

Final accepted-subset NL -> UFCE-FF snapshot:

- Path: `outputs/final/part2/nl_bank_bridge_20260704_200630_latest_ufceff_accepted_20260707`
- Evaluation scope: parser-accepted cases
- Included cases: `243`
- Dropped parser cases: `7`
- Accepted direct equals NL-first: `true`
- Accepted-scope released CF counts: `UFCE1 = 79`, `UFCE2 = 43`, `UFCE3 = 27`
- Effective strict match over the original 250 source cases: `0.972`
- Invalid release count: `0`

Full-250 direct UFCE-FF comparator:

- Path: `outputs/final/part2/nl_bank_bridge_20260704_200630_latest_ufceff_all250_20260707`
- Direct released CF counts: `UFCE1 = 80`, `UFCE2 = 43`, `UFCE3 = 27`

Failed-case impact check:

- Path: `outputs/final/part2/nl_bank_bridge_20260704_200630_latest_ufceff_accepted_20260707/impact_check_7_failed_cases`
- Lost released CF count from the 7 dropped parser cases: `UFCE1 = 1`, `UFCE2 = 0`, `UFCE3 = 0`
- Final table artifact: `route2_final_vs_full250.md`

Do not cite an active or partial parser rerun as final until it writes
`parse_acceptance.json` for all 250 cases.

## Optional Demo

```bash
source .venv/bin/activate
python scripts/final/product/01_serve_demo.py
```

Optional checks:

```bash
python scripts/final/product/02_product_smoke.py --base-url http://127.0.0.1:8000
python scripts/final/product/03_product_acceptance.py --base-url http://127.0.0.1:8000
```

## Further Documentation

- `docs/FINAL_RUNBOOK.md`
- `docs/THESIS_TABLE_TO_SCRIPT_MAP.md`
- `docs/FINAL_CLAIM_TO_EVIDENCE_MAP.md`
- `docs/FINAL_EVIDENCE_POLICY.md`

## Usage Note

This repository is published for thesis transparency, supervisor review, and academic reference.

No open-source license is attached. Third-party code, datasets, and derived materials remain subject to their original licenses.
