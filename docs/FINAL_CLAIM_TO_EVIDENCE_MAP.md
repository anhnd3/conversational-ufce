# Claim-To-Evidence Map

This file maps thesis claims to the GitHub-facing scripts retained after the
cleanup. Detailed table-level provenance is in `docs/THESIS_TABLE_TO_SCRIPT_MAP.md`.

## Part I: UFCE Reproduction And UFCE-FF

| Thesis Area | Claim | Evidence Script | Notes |
|---|---|---|---|
| Chapter 4 UFCE reproduction | Published UFCE paper metrics can be compared against a rerunnable local reproduction. | `scripts/final/part1/table7_full_reproduction.py` | Includes DiCE, DiCE-UF, AR, and UFCE baselines. |
| Chapter 4 UFCE-only audit | UFCE behavior can be isolated from external baselines under locked profiles. | `scripts/final/part1/ufce_only_reproduction.py` | Defines `TUNED_RUN2`, `FINAL_RUNTIME_CONFIG`, and `NEW_BEST_PARAMS`. |
| Chapter 4 tuning provenance | Thesis raw/final-freeze reproduction uses tuned parameters, not the author's default runtime settings. | `scripts/final/part1/ufce_parameter_tuning.py` | Keeps gate tuning plus radius/n_neighbors sweep provenance. |
| Tables 4.2-4.6 | Raw, post-hoc-valid-only, and UFCE-FF values are evaluated on paired author candidate pools. | `scripts/final/part1/author_pool_selector_audit.py` | Use `--config-profile final_freeze --bundle-mode table7_author_public --emit-pairs`. |
| Tables 4.2-4.6 support | Standalone author-core raw replay remains available for raw/post-hoc audit checks. | `scripts/final/part1/author_raw_posthoc_replay.py` | Loads `ufce/core_author` instead of patched `ufce.core`. |
| Red Wine proximity discussion | Red Wine Prox-Euc differences depend on distance-space conventions. | `scripts/final/part1/red_wine_proximity_sensitivity.py` | Consumes emitted factual/CF pairs from author-pool selector audit. |
| UFCE-FF method audit | Force-flip selection should be evaluated against desired model output, not only generated-pool existence. | `scripts/final/part1/ufce_force_flip_experiment.py` | Retained for UFCE-FF experiment checks and profile comparisons. |

## Part II: Natural-Language Feedback Prototype

| Thesis Area | Claim | Evidence Script | Notes |
|---|---|---|---|
| Table 4.12 | Local parser/model selection can be measured by schema validity, field accuracy, status accuracy, stability, and latency. | `scripts/final/part2/parser_benchmark_metrics.py` | Uses frozen `llm_eval/benchmarks` inputs and lightweight reports. |
| Tables 4.13-4.14 | Bank Loan conversations preserve validated outputs and safe terminal states. | `scripts/final/part2/conversation_quality_metrics.py` | Primary 200-session conversation evidence. |
| Table 4.14 diagnostics | No-valid-CF/safe-stop cases can be separated from valid CF presentations. | `scripts/final/part2/no_valid_counterfactual_diagnostics.py` | Supports the safe-stop and no-recourse counts. |
| Policy controls | Constraint policy behavior can be evaluated deterministically. | `scripts/final/part2/policy_control_evaluation.py` | Supports the policy-control side of Part II methodology. |

## Local-Only Evidence

Generated raw outputs, historical closeouts, archived scripts, and forensic
audit notes are kept on disk but removed from Git tracking. They are not required
for a fresh GitHub clone to understand or rerun the thesis-facing workflows.
