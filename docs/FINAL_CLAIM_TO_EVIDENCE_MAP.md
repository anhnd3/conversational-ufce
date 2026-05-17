# Claim-to-Evidence Map

Maps thesis claims to their reproducing scripts and expected outputs.

## Part I — UFCE Core Reproduction & Audit

| Thesis Section | Claim | Evidence Script | Expected Output | Notes |
|---|---|---|---|---|
| 4.2.1 | Table 7 reproduction can be rerun under frozen configs | `scripts/final/part1/01_reproduce_full_table7.py` | `summary.json`, `summary.md`, dataset reports CSVs | Reproduces DiCE, DiCE-UF, UFCE comparison for all datasets (bank, grad, wine, bupa, movie) and combined report. |
| 4.2.1b | UFCE-only behavior is separable from baselines | `scripts/final/part1/01b_reproduce_ufce_only.py` | Per-dataset CSVs with UFCE metrics only | Uses `reproduce_results_v3.py`; provides baseline-free UFCE behavior evidence. |
| 4.2.2 (hyper-tuning) | Hyper-tuned parameters yield optimal per-dataset results | `scripts/final/part1/02_tune_final_parameters.py` | Tuning summary with stage-level pass/fail | Validates both gate-style tuning and radius/n_neighbors sweep for each dataset. |
| 4.2.3 (force-flip) | Raw UFCE output must not be conflated with strict valid recourse in force-flip cases | `scripts/final/part1/03_force_flip_audit.py` | Force-flip summary JSON with per-dataset breakdown; raw vs flip-valid counts where available | Uses trace harness with `--strict --flip-filter 1`. Warnings document unexposed fields as null. |
| 4.2.4 (black-box regression) | Flip validity is checked against the target prediction model's actual behavior, not a heuristic proxy | `scripts/final/part1/03b_blackbox_regression_audit.py` | Model bundle validation report; classifier accuracy/reproduction check summary | Validates that classifier/model bundles match reproduction expectations. Does not retrain models. |
| 4.2.5 (parameter bundle) | Public UFCE behavior is best understood as a coupled heuristic parameter bundle (uf, f2change, step), not one transparent semantic variable | `scripts/final/part1/04_parameter_bundle_ablation.py` | Parameter bundle ablation report per dataset | Uses existing hypothesis ablation script; supports thesis argument that individual parameters cannot be disentangled from each other. |
| 4.2.6 (trace harness) | Per-query trace diagnostics can isolate suspicious UFCE behavior for manual review | `scripts/final/part1/05_trace_harness.py` | Trace JSONL files, ENV.md, and markdown summaries per dataset | Debug tool used during thesis investigation; still usable by reviewers. |

## Part II — Conversational UFCE Evaluation (Bank-Only)

| Thesis Section | Claim | Evidence Script | Expected Output | Notes |
|---|---|---|---|---|
| 4.3 (parser quality) | The structured parser achieves acceptable extraction quality under LM Studio | `scripts/final/part2/01_parser_metrics.py` | Parser benchmark metrics JSON and Markdown report | Requires LM Studio; uses frozen benchmark YAML configuration. |
| 4.3 (conversation integrity) | Conversational system preserves safe states, validated outputs, and deterministic responses for Bank dataset | `scripts/final/part2/02_conversation_metrics.py` | Conversation quality metrics summary with pass/fail gate | Core Part II thesis evidence; validates all conversation turns against canonical validator outcomes. |
| 4.3 (refinement) | Continue-same-profile refinement behavior is correct and does not corrupt runtime state | `scripts/final/part2/03_refinement_metrics.py` | Refinement metrics report with merge/validation pass rate | Uses separate refinement corpus to test repeated profile modifications. |
| 4.3 (backend comparison) | Backend deterministic validation produces identical results for golden-parity cases across runs | `scripts/final/part2/04_backend_comparison.py` | Backend comparison report showing exact-match or documented deviation | Compares current runtime against frozen golden traces from earlier validated sessions. |
| 4.3 (agent portability) | Agent output is portable when using different parser backends with identical canonical validation rules | `scripts/final/part2/05_agent_portability.py` | Portability report comparing outputs from two backend configurations | Tests that the same bank profile yields equivalent runtime results regardless of parser layer differences. |
| 4.3 (replay robustness) | Replaying a frozen corpus against the current system produces identical outcomes when deterministic seeding is used | `scripts/final/part2/06_replay_robustness.py` | Replay robustness report with pass rate per scenario category | Ensures stable_demo mode seeding yields bit-identical counterfactuals. |
| 4.3 (closeout) | All Part II evaluation gates pass collectively, forming a cohesive body of evidence | `scripts/final/part2/99_part2_closeout.py` | Combined closeout summary JSON + Markdown; evidence index with per-report verdicts | Orchestrates all above individual checks plus targeted pytest suites. Requires LM Studio preflight. |

## Product Demo Validation

| Thesis Section | Claim | Evidence Script | Expected Output | Notes |
|---|---|---|---|---|
| Product demo (API/UI) | Local API and server-rendered UI can demonstrate the complete system end-to-end | `scripts/final/product/01_serve_demo.py` | Running server on configured host:port; prints resolved config to stdout | Starts FastAPI-based product with LM Studio parser, SQLite persistence. |
| Product smoke (automated) | All critical API endpoints respond correctly and produce valid session state transitions | `scripts/final/product/02_product_smoke.py` | Smoke test JSON + Markdown report with pass/fail per endpoint group | Requires running server; tests health, version, sessions CRUD, catalog, artifact access. |
| Product acceptance (MVP) | The product meets MVP compliance criteria including manual validation session | `scripts/final/product/03_product_acceptance.py` | Acceptance report with verdict: all scenarios pass or documented failures | Combines automated smoke results and optional manual session replay for final human-checked evidence. |

## Part I Closeout Gate Summary

| Thesis Section | Claim | Evidence Script | Expected Output | Notes |
|---|---|---|---|---|
| 4.x (Part I collective) | All required Part I evidence groups are present and pass their gates | `scripts/final/part1/99_part1_closeout.py` | `part1_closeout_summary.json`, Markdown summary, evidence index JSON + Markdown | Runs all Part I wrappers above in sequence; fails if any required group is missing or failing. |
