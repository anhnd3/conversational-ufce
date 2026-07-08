# Final Thesis Runbook

This runbook documents the thesis-facing workflows that should be used to regenerate evidence from a clean clone.

Use Linux or WSL for the final runners.

Use `README.md` for the top-level route overview.
Use `docs/FINAL_CLAIM_TO_EVIDENCE_MAP.md` for claim-level provenance.

Generated artifacts under `outputs/**` are local-only. Regenerate them from the
commands below instead of treating output directories as part of the published
source tree.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/final/thesis/doctor.py --out-dir outputs/final/doctor
```

For Part II runs, LM Studio must be available:

```bash
API_BASE=http://127.0.0.1:1234
MODEL_ALIAS=qwen3-14b
TIMEOUT_S=600
```

## Route 1: UFCE Reproduction And UFCE-FF

Canonical order:

1. Regenerate raw author-code replay from `ufce/core_author`.
2. Regenerate thesis raw/final-freeze replay from `ufce/core`.
3. Regenerate post-hoc valid-only aggregation over the raw selected outputs.
4. Regenerate UFCE-FF from `ufce/ufce_ff`.

### Step 0: tuning provenance

```bash
python scripts/final/part1/ufce_parameter_tuning.py \
  --dataset all \
  --run_stage all \
  --out-dir outputs/final/part1/ufce_parameter_tuning
```

Read:

- tuning summaries under `outputs/final/part1/ufce_parameter_tuning`

### Step 1: author raw reproduction

```bash
python scripts/final/part1/01_author_raw_reproduction.py \
  --dataset all \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/01_author_raw_reproduction
```

Read:

- `manifest.json`
- `author_raw_posthoc_metric_summary.csv`

### Step 2: thesis raw/final-freeze reproduction

```bash
python scripts/final/part1/02_ufce_core_final_freeze.py \
  --dataset all \
  --runtime-profile final_freeze \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/02_ufce_core_final_freeze
```

Read:

- run-level `summary.json`
- dataset-level summary CSVs

### Step 3: post-hoc valid-only aggregation

```bash
python scripts/final/part1/03_ufce_post_hoc.py \
  --dataset all \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/03_ufce_post_hoc
```

Read:

- `author_raw_vs_posthoc_metric_comparison.csv`
- `author_raw_posthoc_metric_summary.csv`

### Step 4: UFCE-FF

```bash
python scripts/final/part1/04_ufce_ff.py \
  --dataset all \
  --runtime-profile final_freeze \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/04_ufce_ff
```

Read:

- `summary.json`
- dataset-level summary CSVs

Aggregate command for smoke or final local regeneration:

```bash
python scripts/final/part1/run_part1_final_bundle.py \
  --dataset all \
  --runtime-profile final_freeze \
  --bundle-mode table7_author_public \
  --out-dir outputs/final/part1/final_ufce_bundle
```

## Route 2: Bank NLP Parser And NL -> UFCE-FF Bridge

Canonical order:

1. Run parser smoke or unit checks.
2. Run parser-only evaluation on the 250 NL Bank cases.
3. Monitor live parser quality.
4. Choose either the strict all-case bridge or the accepted-subset diagnostic.

### Step 1: parser smoke or unit checks

```bash
bash scripts/final/part2/test_nl_bank_parser_v2.sh
```

### Step 2: parser-only evaluation on the 250-case Bank set

This runner rebuilds the single 250-case evaluation set from the 5 original Bank Loan test folds and then runs parser-only evaluation.

```bash
MODEL_ALIAS=qwen3-14b \
API_BASE=http://127.0.0.1:1234 \
TIMEOUT_S=600 \
OUT_DIR=outputs/final/part2/nl_bank_bridge_<run_id> \
bash scripts/final/part2/run_nl_bank_parse_only_v2.sh
```

Primary artifacts:

- `bank_250_queries.csv`
- `parsed_requests.jsonl`
- `parse_errors.jsonl`
- `record_reconstruction.csv`
- `parse_progress.json`
- `parse_quality_progress.json`
- `parse_acceptance.json`
- `run.log`

### Step 3: parser gate

Use `parse_quality_progress.json` during the run and `parse_acceptance.json` after the run.

Operational gate before any downstream UFCE-FF stage:

Continue only after a completed parser run has `error_rate < 0.10`, no missing
cases, and no API/schema/verifier failures.

Final parser acceptance target:

`ready_for_runtime_count = 250`, `exact_reconstruction_count = 250`, and `error_count = 0`.

Interpretation:

- The `< 0.10` gate is the operational threshold for downstream diagnostics.
- The `250/250` target is the strict all-case claim threshold.
- If strict acceptance is not met, use `--parse-eval-scope accepted` and report
  the excluded parser cases.

Use these files for diagnosis:

- `parse_quality_progress.json`: live `failed/completed`, ready, exact, API, schema, verifier counts
- `record_reconstruction.csv`: case-level reconstruction status
- `parse_errors.jsonl`: raw failure cases

### Step 4A: strict NL -> UFCE-FF bridge

Run this as the final all-case bridge only after strict parser acceptance is
met. With `--parse-eval-scope accepted`, this still evaluates all 250 source
cases when the parser accepted count is 250.

```bash
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

Primary artifacts:

- `summary.json`
- `summary.md`
- `table_main_bridge.csv`
- `table_language_group.csv`
- `mismatch_cases.csv`
- `invalid_release_cases.csv`

Interpretation:

- `table_main_bridge.csv`: overall headline numbers
- `table_language_group.csv`: `G1..G5` breakdown
- `summary.json`: machine-readable final bundle

### Step 4B: accepted-subset NL -> UFCE-FF diagnostic

Use this when parser `error_rate < 0.10` but the strict `250/250` target is not
yet met. This measures the locked UFCE-FF backend only on parser-accepted cases
and keeps parser loss visible through `evaluation_case_selection.csv`.

```bash
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

Primary artifacts:

- `evaluation_case_selection.csv`
- `direct_ufce_ff_results.csv`
- `nl_first_ufce_ff_results.csv`
- `summary.json`
- `config_snapshot.json`
- `impact_check_7_failed_cases/route2_final_vs_full250.md`
- `impact_check_7_failed_cases/failed_cases_impact_summary.md`

## Current Local Snapshots

### Route 1 snapshot

Representative Route 1 directory:

- `outputs/final/part1/09_author_pool_finalfreeze_20260624`

Useful files there:

- `author_pool_selector_summary.csv`
- `author_pool_metric_summary.csv`
- `summary.json`

### Route 2 historical bridge baseline

Representative Route 2 directory:

- `outputs/final/part2/nl_bank_bridge_v2_wordnum_20260702_201114`

This is a useful baseline because it exposes the pre-improvement parser bottleneck:

- overall parse-ready rate: `0.40`
- overall exact reconstruction rate: `0.40`
- overall strict outcome match rate: `0.40`
- invalid release count: `0`

### Route 2 completed parser rerun

Representative completed parser directory:

- `outputs/final/part2/nl_bank_bridge_20260704_200630`

Summary from `parse_acceptance.json`:

- completed cases: `250`
- runtime-ready count: `247`
- exact reconstruction count: `243`
- parser error count: `7`
- parser error rate: `0.028`
- strict all-case target met: `no`
- operational `< 0.10` gate met: `yes`

### Route 2 accepted-subset final diagnostic

Representative accepted-subset directory:

- `outputs/final/part2/nl_bank_bridge_20260704_200630_latest_ufceff_accepted_20260707`

Useful files there:

- `evaluation_case_selection.csv`
- `direct_ufce_ff_results.csv`
- `nl_first_ufce_ff_results.csv`
- `summary.json`
- `parse_acceptance.json`
- `impact_check_7_failed_cases/route2_final_vs_full250.md`
- `impact_check_7_failed_cases/failed_cases_impact_summary.md`

Scope:

- included parser-accepted cases: `243`
- dropped parser cases: `7`
- accepted direct equals NL-first: `true`
- accepted-scope released CF counts: `UFCE1 = 79`, `UFCE2 = 43`, `UFCE3 = 27`
- effective strict match over the original 250 source cases: `0.972`
- full-250 direct comparator released CF counts: `UFCE1 = 80`, `UFCE2 = 43`, `UFCE3 = 27`
- lost released CF count from the 7 dropped parser cases: `UFCE1 = 1`, `UFCE2 = 0`, `UFCE3 = 0`

### Route 2 active or partial reruns

Monitor active output directories with:

- `parse_quality_progress.json`
- `record_reconstruction.csv`
- `parse_errors.jsonl`

Do not treat a partial parser rerun as the final thesis snapshot until the full
250-case run writes `parse_acceptance.json`.

## Optional Local Demo

Optional local demo:

```bash
python scripts/final/product/01_serve_demo.py
python scripts/final/product/02_product_smoke.py --base-url http://127.0.0.1:8000
python scripts/final/product/03_product_acceptance.py --base-url http://127.0.0.1:8000
```

## Related Docs

- `docs/THESIS_TABLE_TO_SCRIPT_MAP.md`
- `docs/FINAL_CLAIM_TO_EVIDENCE_MAP.md`
- `docs/FINAL_EVIDENCE_POLICY.md`
