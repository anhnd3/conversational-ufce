# Claim-To-Evidence Map

This file maps thesis claims to the GitHub-facing scripts retained after the
cleanup. Detailed table-level provenance is in `docs/THESIS_TABLE_TO_SCRIPT_MAP.md`.

Target thesis version: Final_v7.6 / Final.
PDF export timestamp: 2026-07-12T16:49:19+07:00.
Committed headline summary: `outputs/final/thesis_canonical/summary.json`.

## Part I: UFCE Reproduction And UFCE-FF

| Thesis Area | Claim | Evidence Script | Notes |
|---|---|---|---|
| Chapter 4 UFCE reproduction | Published UFCE paper metrics can be compared against a rerunnable local reproduction. | `scripts/final/part1/table7_full_reproduction.py` | Includes DiCE, DiCE-UF, AR, and UFCE baselines. |
| Chapter 4 UFCE-only audit | UFCE behavior can be isolated from external baselines under locked profiles. | `scripts/final/part1/02_ufce_core_final_freeze.py` | Uses `ufce/core` with `FINAL_RUNTIME_CONFIG` and no validity gate. |
| Chapter 4 tuning provenance | Thesis raw/final-freeze reproduction uses tuned parameters, not the author's default runtime settings. | `scripts/final/part1/ufce_parameter_tuning.py` | Keeps gate tuning plus radius/n_neighbors sweep provenance. |
| Tables 4.2-4.6 | Raw author replay, thesis raw replay, post-hoc-valid-only aggregation, and UFCE-FF are separated by semantic path. | `scripts/final/part1/run_part1_final_bundle.py` | Calls `01_author_raw_reproduction.py`, `02_ufce_core_final_freeze.py`, `03_ufce_post_hoc.py`, and `04_ufce_ff.py`; post-hoc aggregates raw selected outputs rather than regenerating a new pool. |
| Tables 4.2-4.6 support | Standalone author-core raw replay remains available for raw/post-hoc audit checks. | `scripts/final/part1/01_author_raw_reproduction.py` | Loads `ufce/core_author` and records upstream provenance. |
| Red Wine proximity discussion | Red Wine Prox-Euc differences depend on distance-space conventions. | `scripts/final/part1/red_wine_proximity_sensitivity.py` | Consumes emitted factual/CF pairs from author-pool selector audit. |
| UFCE-FF method audit | Candidate validity is enforced before UFCE-FF selection instead of being toggled inside the raw core. | `scripts/final/part1/04_ufce_ff.py` | Loads `ufce/ufce_ff`; old mode-based runners are non-canonical compatibility tools. |

## Part II: Natural-Language Feedback Prototype

| Thesis Area | Claim | Evidence Script | Notes |
|---|---|---|---|
| Table 4.12 | UFCE-FF valid-solution coverage is reported separately from conditional metric quality. | `scripts/final/part1/04_ufce_ff.py` | Final Bank Loan headline coverage is locked in `outputs/final/thesis_canonical/summary.json`: direct `80/44/26`. |
| Table 4.13 | Local parser/model selection can be measured by schema validity, field accuracy, status accuracy, stability, and latency. | `llm_eval/scripts/run_bank_cf_llm_eval.py` and `llm_eval/reporting.py` | Uses frozen `llm_eval/benchmarks`; the metric contract is `22 cases x 3 repeats = 66` requests/model and field accuracy over 9 fields. |
| Figure 4.5 | Bank natural-language requests can be evaluated on runtime readiness and exact reconstruction before UFCE-FF execution. | `scripts/final/part2/run_nl_bank_parse_only_v2.sh` and `scripts/final/part2/nl_bank_bridge_experiment.py --stage eval_parse` | Uses the 250-case Bank set derived from the original 5 test folds; final parser gate is `247` runtime-ready and `243` accepted. |
| Figure 4.6 | Parser errors are retained as handoff-gate loss rather than hidden by downstream UFCE-FF evaluation. | `scripts/final/part2/run_nl_bank_parse_only_v2.sh` and `scripts/final/part2/nl_bank_bridge_experiment.py --stage eval_parse` | The seven excluded case IDs are listed in the committed canonical summary. |
| Figure 4.7 | Parser-accepted Bank natural-language requests can be handed to the locked UFCE-FF backend without changing backend authority. | `scripts/final/part2/nl_bank_bridge_experiment.py` | Final headline counts are direct `80/44/26` and after handoff `79/44/26`. |
| Table 4.14 | The parser handoff gate has a measured cost relative to the full-250 UFCE-FF direct comparator. | `scripts/final/part2/failed_cases_impact_check.py` | Final released-count loss is one UFCE-FF1 solution and zero UFCE-FF2/UFCE-FF3 solutions; detailed rerun artifacts stay local-only. |
| Policy controls | Constraint policy behavior can be evaluated deterministically. | `scripts/final/part2/policy_control_evaluation.py` | Supports the policy-control side of Part II methodology. |

## Local-Only Evidence

Generated raw outputs, historical closeouts, archived scripts, and forensic
audit notes are kept on disk but removed from Git tracking. They are not required
for a fresh GitHub clone to understand or rerun the thesis-facing workflows.
