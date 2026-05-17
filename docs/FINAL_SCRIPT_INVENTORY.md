# Final Script Inventory

Generated for thesis final evidence repository consolidation (v1/final).

## Legend

- **FINAL_WRAPPER**: Script is actively called by a `scripts/final/` wrapper. Must not be removed or renamed without updating the wrapper.
- **INTERNAL_SUBGATE**: Called internally by another script that is itself FINAL_WRAPPER, but has no direct final wrapper of its own (e.g., utility scripts).
- **HISTORICAL_KEEP**: No longer called by any active final wrapper or test; kept for traceability and audit trail.
- **ARCHIVE_LATER**: Not yet safe to archive due to unresolved dependency check. Requires manual review before archiving.

## Part I — UFCE Reproduction & Audit Scripts

| Current Path | Final Wrapper(s) | Thesis Claim | Inputs | Outputs | Decision | Notes |
|---|---|---|---|---|---|---|
| `scripts/reproduce_full_table7_result.py` | `part1/01_reproduce_full_table7.py`, `format_part1_table7_dataset_tables.py`, `ufce_hypothesis_ablation.py`, tests | Table 7 reproduction can be rerun under frozen configs | UFCE data, folds | Chapter table CSVs | FINAL_WRAPPER | Imported by test files and ablation script; do not rename without updating all callers. |
| `scripts/reproduce_results_v3.py` | `part1/01b_reproduce_ufce_only.py`, `hypertune_ufce_p7.py`, `consolidate_final_table7_freeze.py`, tests | UFCE-only behavior separate from baselines | Same data as above | Per-dataset CSVs, scalers | FINAL_WRAPPER | Used in LR bundle test; has movie distance scaler function imported by `test_lr_bundle`. |
| `scripts/hypertune_ufce_p7.py` | `part1/02_tune_final_parameters.py` | Hyper-tuning final parameter validation | Bank data (or all datasets) | Tuned parameters per dataset/stage | FINAL_WRAPPER | Imports from reproduce_results_v3. Supports --stage run1/run2/all. |
| `scripts/run_ufce_trace_harness.py` | `part1/03_force_flip_audit.py`, `part1/05_trace_harness.py` | Force-flip validation; raw candidates vs strict valid recourse | Dataset CSV, folds | Trace harness JSONL + ENV.md per dataset | FINAL_WRAPPER | Supports --strict and --flip-filter flags for force-flip audit. |
| `scripts/run_ufce_smoke.py` | Not directly wrapped in final (used by old phase scripts) | Smoke test for UFCE sanity check | Bank data only | stdout pass/fail | HISTORICAL_KEEP | Not called by any final wrapper or active test; kept for historical traceability. |
| `scripts/consolidate_final_table7_freeze.py` | None directly, but used internally by hypertune | Consolidation of frozen Table 7 results | Outputs from reproduce scripts | Freeze report | INTERNAL_SUBGATE | Called indirectly via hypertune workflow. Not a standalone final entrypoint. |
| `scripts/format_part1_table7_dataset_tables.py` | None (legacy formatting) | Format output CSVs for thesis tables | Reproduction outputs | Formatted CSVs ready for paper | HISTORICAL_KEEP | No longer called by any active test or wrapper; legacy thesis table formatter. |
| `scripts/ufce_hypothesis_ablation.py` | Called indirectly via `part1/04_parameter_bundle_ablation.py` (INTERNAL_SUBGATE) | uf/f2change/step operational analysis | Same data as above | Ablation report | INTERNAL_SUBGATE | Imported by Part I parameter-bundle ablation wrapper; do not remove without updating that path. |
| `scripts/export_lr_model_bundles.py` | None (model export) | Export logistic regression model bundles for runtime | Trained models | Serialized model bundles in llm/models/ | HISTORICAL_KEEP | Model export tool; not called by any active final path but needed to regenerate model bundles. |
| `scripts/rerun_movie_run2_minmax100.sh` | None (bash helper) | Legacy bash script for movie re-run | Shell environment | stdout log | ARCHIVE_LATER | Bash script, no Python callers found; requires manual review before archive as it may be referenced in historical docs. |

## Part II — Conversational UFCE Evaluation Scripts

| Current Path | Final Wrapper(s) | Thesis Claim | Inputs | Outputs | Decision | Notes |
|---|---|---|---|---|---|---|
| `scripts/run_part2_thesis_metrics_report.py` | `part2/02_conversation_metrics.py` | Part II thesis metrics report | Corpus, parser outputs | Metrics report | FINAL_WRAPPER | Primary conversation quality evidence. |
| `scripts/run_part2_refinement_metrics_report.py` | `part2/03_refinement_metrics.py` | Refinement behavior validation | Same corpus + refinement cases | Refinement report | FINAL_WRAPPER | Validates continue-same-profile behavior. |
| `scripts/run_part2_backend_comparison_report.py` | `part2/04_backend_comparison.py` | Backend comparison (golden parity) | Corpus, golden traces | Comparison report | FINAL_WRAPPER | |
| `scripts/run_part2_agent_portability_report.py` | `part2/05_agent_portability.py` | Agent portability validation | Portability corpus | Report | FINAL_WRAPPER | |
| `scripts/run_part2_replay_robustness_report.py` | `part2/06_replay_robustness.py` | Replay robustness validation | Replay corpus | Report | FINAL_WRAPPER | |
| `scripts/run_part2_closeout_bundle.py` | `part2/99_part2_closeout.py` | One final Part II closeout entrypoint | All corpora, benchmark configs | Combined closeout summary + evidence index | FINAL_WRAPPER | Orchestrates all Part II reports. Used directly by final wrapper. |
| `scripts/run_part2_end_to_end_bank.py` | None (higher-level orchestrator) | End-to-end bank pipeline runner with optional synthetic corpora | Same as above, plus optional synth300 corpora paths | Full e2e outputs | INTERNAL_SUBGATE | Can override corpus paths for synthetic runs; used by run_part2_closeout_bundle internally. Not a direct final wrapper target. |
| `scripts/run_part2_bank_evidence_pack.py` | None (legacy evidence pack) | Legacy Part II evidence packing | Corpus, reports | Evidence pack JSON/MD | HISTORICAL_KEEP | Superseded by new closeout approach but kept for traceability. |
| `scripts/export_part2_case_studies.py` | Not wrapped; used manually | Export conversation case studies from raw output folders | outputs/conversations/* | Case study files per turn | HISTORICAL_KEEP | Used in manual workflows; no automated test depends on it. |
| `scripts/extract_part2_expected_distribution.py` | None (analysis helper) | Extract expected distributions from corpus | Corpus JSONL | Distribution file | ARCHIVE_LATER | Requires dependency check against closeout_bundle internal paths before archiving. |
| `scripts/extract_part2_trace_cases.py` | None (debug helper) | Extract trace cases for debugging | Corpus, artifacts | Trace case files | ARCHIVE_LATER | Debug tool; no test coverage found but may be used manually during development. |

## Product / Demo Scripts

| Current Path | Final Wrapper(s) | Thesis Claim | Inputs | Outputs | Decision | Notes |
|---|---|---|---|---|---|---|
| `scripts/run_phase3_2_demo.py` | `product/01_serve_demo.py` | Local product demo server works end-to-end | .env config, LM Studio connection | Running FastAPI server on port 8000 | FINAL_WRAPPER | Start local demo. Configurable via environment variables. |
| `scripts/run_phase3_2_product_smoke.py` | `product/02_product_smoke.py` | Product smoke against running server passes all scenarios | Running server base URL | Smoke test results JSON + markdown | FINAL_WRAPPER | Requires a running server; sends requests and validates responses. |
| `scripts/run_phase3_2_acceptance_report.py` | `product/03_product_acceptance.py` | Acceptance report shows MVP compliance | Running server, optional manual session ID | Acceptance report with pass/fail verdicts | FINAL_WRAPPER | Uses both automated smoke tests and an optional manual session for final validation. |
| `scripts/run_phase3_2_metrics_report.py` | Not wrapped; legacy metrics | Phase 3.2 metrics report (internal) | Product outputs | Metrics JSON/MD | HISTORICAL_KEEP | No final wrapper calls this directly but it produces internal product metrics. Kept as historical record. |

## Phase Closeout / Orchestration Scripts

| Current Path | Final Wrapper(s) | Thesis Claim | Inputs | Outputs | Decision | Notes |
|---|---|---|---|---|---|---|
| `scripts/run_phase1_closeout_suite.py` | Not directly wrapped (historical) | Phase 1 formal closeout | Various Part I outputs | Phase 1 closeout summary | HISTORICAL_KEEP | Historical phase closeout; superseded by new final wrappers but kept for traceability. |
| `scripts/run_phase3_1_closeout_suite.py` | Not directly wrapped (historical) | Phase 3.1 formal closeout | Conversation artifacts, test results | Closeout summary with pass/fail per scenario | HISTORICAL_KEEP | Historical phase closeout; not called by any final wrapper but kept for the record. |

## Other / Chapter Assembly Scripts

| Current Path | Final Wrapper(s) | Thesis Claim | Inputs | Outputs | Decision | Notes |
|---|---|---|---|---|---|---|
| `scripts/ufce_final_blindspot_audit.py` | Not directly wrapped (high-level audit harness) | Blind-spot audit of uf/f2change/step interactions across datasets | Phase 1 + confirm artifacts | CSVs, rankings, sidecars, notes | INTERNAL_SUBGATE | Used indirectly via Part I parameter-bundle ablation wrapper; not a standalone final entrypoint. |
| `scripts/build_chapter4_evidence_index.py` | None (chapter helper) | Build evidence index for Chapter 4 materials | Various outputs | Evidence index JSON/MD | HISTORICAL_KEEP | Not called by any final wrapper; kept for chapter assembly traceability. |
| `scripts/build_chapter4_part2_core_evidence.py` | None (chapter helper) | Build core evidence bundle for Part II in Chapter 4 | Corpus + reports | Evidence bundle JSON/MD | HISTORICAL_KEEP | Not called by any final wrapper; kept for chapter assembly traceability. |
| `scripts/build_phase2_chapter_pack.py` | None (legacy packer) | Phase 2 chapter-pack builder | Reports, outputs | Chapter pack artifacts | HISTORICAL_KEEP | Legacy pack script; superseded by final wrappers but retained for reference. |
| `scripts/probe_phase2_reject_candidates.py` | None (analysis helper) | Probe reject candidates in Phase 2 artifacts | Corpus subset, outputs | Diagnostic output files | ARCHIVE_LATER | Requires dependency check before archiving; used historically during investigation. |
| `scripts/probe_phase3_2_reproducibility.py` | None (repro probe) | Probe reproducibility of specific product scenarios | Product outputs | Reproducibility report JSON/MD | HISTORICAL_KEEP | Diagnostic-only; not called by any final wrapper. |
| `scripts/repro_author.py` | None (authoring helper) | Authoring-time reproduction utility | Mixed artifacts | stdout + optional files | HISTORICAL_KEEP | Not integrated in final wrappers but useful for one-off reruns. |
| `scripts/freeze_part2_bank_synth_corpora.py` | Called indirectly via Part II closeout bundle | Freeze Bank-synthetic corpora into canonical paths | Raw corpus outputs | Frozen JSONL + hash metadata | INTERNAL_SUBGATE | Used within run_part2_closeout_bundle to stabilize corpora. Do not remove without updating that path. |
| `scripts/freeze_part2_corpora.py` | Called indirectly via Part II closeout bundle | Freeze main Part II corpora into canonical paths | Raw corpus outputs | Frozen JSONL + hash metadata | INTERNAL_SUBGATE | Used within run_part2_closeout_bundle to stabilize corpora. Do not remove without updating that path. |

## Final Wrapper Scripts (`scripts/final/`)

All scripts in `scripts/final/` are **FINAL_WRAPPER** entrypoints. They delegate to legacy implementations without modifying algorithm behavior. The full list:

```
scripts/final/_common.py                           # Shared utility module (not a wrapper)
scripts/final/part1/01_reproduce_full_table7.py    # → scripts/reproduce_full_table7_result.py
scripts/final/part1/01b_reproduce_ufce_only.py     # → scripts/reproduce_results_v3.py
scripts/final/part1/02_tune_final_parameters.py    # → scripts/hypertune_ufce_p7.py
scripts/final/part1/03_force_flip_audit.py         # → scripts/run_ufce_trace_harness.py --strict --flip-filter 1
scripts/final/part1/03b_blackbox_regression_audit.py # → pytest ufce/tests (model bundle checks)
scripts/final/part1/04_parameter_bundle_ablation.py # → scripts/ufce_hypothesis_ablation.py per dataset
scripts/final/part1/05_trace_harness.py            # → scripts/run_ufce_trace_harness.py
scripts/final/part1/99_part1_closeout.py           # Orchestrates all Part I wrappers above
scripts/final/part2/01_parser_metrics.py           # → llm_eval parser benchmark (LM Studio)
scripts/final/part2/02_conversation_metrics.py     # → scripts/run_part2_thesis_metrics_report.py
scripts/final/part2/03_refinement_metrics.py       # → scripts/run_part2_refinement_metrics_report.py
scripts/final/part2/04_backend_comparison.py       # → scripts/run_part2_backend_comparison_report.py
scripts/final/part2/05_agent_portability.py        # → scripts/run_part2_agent_portability_report.py
scripts/final/part2/06_replay_robustness.py        # → scripts/run_part2_replay_robustness_report.py
scripts/final/part2/99_part2_closeout.py           # → scripts/run_part2_closeout_bundle.py
scripts/final/product/01_serve_demo.py             # → scripts/run_phase3_2_demo.py
scripts/final/product/02_product_smoke.py          # → scripts/run_phase3_2_product_smoke.py
scripts/final/product/03_product_acceptance.py     # → scripts/run_phase3_2_acceptance_report.py
scripts/final/thesis/doctor.py                     # Environment check (self-contained)
scripts/final/thesis/all_closeout.py               # Orchestrates Part I + Part II closeouts
scripts/final/thesis/export_evidence_pack.py       # Evidence pack exporter from existing outputs
```

## Dependency Warnings

These scripts are **heavily imported** by other parts of the codebase and must never be removed or renamed without updating all callers:

- `reproduce_full_table7_result` — used in test files, ablation, format script.
- `reproduce_results_v3` — used in hypertune, consolidation, LR bundle tests, model bundles metadata.
- `run_ufce_trace_harness` — used by two different final wrappers with different flag combinations.