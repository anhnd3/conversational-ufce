# Final Thesis Runbook

This document provides clean, final-facing commands for reproducing all thesis evidence.

For full script inventory and dependency details, see `docs/FINAL_SCRIPT_INVENTORY.md`.

## 1. Setup

```bash
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Edit `.env` if your LM Studio instance uses a non-default port or model alias:

- `LM_STUDIO_API_BASE=http://127.0.0.1:1234` (change if needed)
- `MODEL_ALIAS=qwen/qwen3-14b` (must match what LM Studio is serving)

## 2. Environment Doctor

Before running any evidence reproduction, validate the environment:

```bash
python scripts/final/thesis/doctor.py
python scripts/final/thesis/doctor.py --check-lm-studio    # also checks LM Studio connectivity
```

Output goes to `outputs/final/doctor/<run_id>/` with both JSON and Markdown summaries.

## 3. Part I Evidence Reproduction

### Quick Start (All-in-One)

Run the complete Part I closeout in one command:

```bash
python scripts/final/part1/99_part1_closeout.py --out-dir outputs/final/part1_closeout
```

This orchestrates all required evidence groups sequentially. Outputs go to `outputs/final/part1_closeout/<run_id>/`.

### Individual Reproduction Steps (Optional)

If you need to rerun only a specific Part I section:

#### 3.1 Table 7 Full Reproduction

```bash
python scripts/final/part1/01_reproduce_full_table7.py --dataset all --out-dir outputs/final/part1/table7
python scripts/final/part1/01_reproduce_full_table7.py --dataset bank --out-dir outputs/final/part1/table7_bank
```

#### 3.2 UFCE-Only Reproduction

```bash
python scripts/final/part1/01b_reproduce_ufce_only.py --dataset all --out-dir outputs/final/part1/ufce_only
```

#### 3.3 Parameter Tuning

```bash
python scripts/final/part1/02_tune_final_parameters.py --dataset all --stage all --out-dir outputs/final/part1/hypertune
python scripts/final/part1/02_tune_final_parameters.py --dataset bank --stage run2 --out-dir outputs/final/part1/hypertune_bank
```

Supported `--stage`: `run1`, `run2`, or `all`.

#### 3.4 Force-Flip Audit

```bash
python scripts/final/part1/03_force_flip_audit.py --dataset all --out-dir outputs/final/part1/force_flip
```

Validates that raw UFCE output does not produce strict valid recourse in force-flip cases.

**Note:** The current trace harness may not expose `raw_candidate_count` and `flip_valid_candidate_count` directly; these fields will be `null` with a warning if the underlying script doesn't provide them. This is documented as a limitation.

#### 3.5 Black-Box Regression Audit

```bash
python scripts/final/part1/03b_blackbox_regression_audit.py --dataset all --out-dir outputs/final/part1/blackbox_regression
```

Validates classifier/model bundle assumptions used by force-flip audit. Does not retrain models unless the underlying reproduction script already does so.

#### 3.6 Parameter Bundle Ablation (uf / f2change / step)

```bash
python scripts/final/part1/04_parameter_bundle_ablation.py --dataset all --out-dir outputs/final/part1/parameter_bundle
```

Supports thesis claim that public UFCE behavior is best understood as a coupled heuristic parameter bundle.

#### 3.7 Trace Harness (Per-Query Debug)

```bash
python scripts/final/part1/05_trace_harness.py --dataset bank --out-dir outputs/final/part1/traces_bank
```

## 4. Part II Conversational Evidence Validation

### Quick Start (All-in-One)

```bash
python scripts/final/part2/99_part2_closeout.py --out-dir outputs/final/part2_closeout
```

**Requirements:** LM Studio must be running at the configured `LM_STUDIO_API_BASE` URL. If unavailable, the closeout will record a clear failure reason without faking a pass.

### Individual Steps (Optional)

#### 4.1 Parser Metrics (Requires LM Studio)

```bash
python scripts/final/part2/01_parser_metrics.py --out-dir outputs/final/part2/parser_metrics
```

#### 4.2 Conversation Metrics

```bash
python scripts/final/part2/02_conversation_metrics.py --out-dir outputs/final/part2/conversation_metrics
```

#### 4.3 Refinement Metrics

```bash
python scripts/final/part2/03_refinement_metrics.py --out-dir outputs/final/part2/refinement_metrics
```

#### 4.4 Backend Comparison

```bash
python scripts/final/part2/04_backend_comparison.py --out-dir outputs/final/part2/backend_comparison
```

#### 4.5 Agent Portability

```bash
python scripts/final/part2/05_agent_portability.py --out-dir outputs/final/part2/agent_portability
```

#### 4.6 Replay Robustness

```bash
python scripts/final/part2/06_replay_robustness.py --out-dir outputs/final/part2/replay_robustness
```

## 5. Product Demo Validation

### Start Local Demo Server

```bash
# Terminal A: start the product demo server
python scripts/final/product/01_serve_demo.py
```

The server reads config from `.env` and prints resolved configuration on startup (host, port, LM Studio base, model alias).

### Run Smoke Test Against Running Server

```bash
# Terminal B: run smoke tests against a running server
python scripts/final/product/02_product_smoke.py --base-url http://127.0.0.1:8000 --out-dir outputs/final/product_smoke
```

### Run Acceptance Report

```bash
# Optional: include manual session ID for final validation
python scripts/final/product/03_product_acceptance.py \
  --base-url http://127.0.0.1:8000 \
  --out-dir outputs/final/product_acceptance
```

## 6. Export Final Evidence Pack

After running closeouts, create a final evidence pack without rerunning experiments:

```bash
python scripts/final/thesis/export_evidence_pack.py \
  --part1-closeout outputs/final/part1_closeout/<run_id> \
  --part2-closeout outputs/final/part2_closeout/<run_id> \
  --out-dir outputs/final/evidence_pack
```

Output contains `README.md`, `MANIFEST.json`, claim-to-evidence mapping, and selected reports. Heavy raw outputs are excluded unless `--include-raw` is passed.

## 7. Troubleshooting

### LM Studio Not Reachable

If Part II closeout fails with a preflight error:
1. Ensure LM Studio is running at the configured URL (default `http://127.0.0.1:1234`).
2. Run `python scripts/final/thesis/doctor.py --check-lm-studio` to diagnose connectivity.
3. Verify model alias in `.env` matches what LM Studio is actually serving.

### Part II Closeout Fails at Preflight (LM Studio Unavailable)

The closeout records a clear failure reason and does not fake a pass. The doctor script explains how to start LM Studio. Do not attempt to bypass the preflight check.

### Product Tests Require Server

Product smoke and acceptance tests require a running server in another terminal. If you get connection errors, verify the server is running with `python scripts/final/product/01_serve_demo.py` before launching tests.

### Missing Data Files

If reproduction scripts fail with missing data:
- Verify `ufce/data/` contains CSV files (bank.csv, wine.csv, etc.)
- Verify `ufce/data/folds/` exists and has fold definitions for each dataset.