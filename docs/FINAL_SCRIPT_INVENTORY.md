# Final Script Inventory

This inventory lists the scripts intentionally kept in Git after cleanup. The
old numeric prefixes were process bookkeeping only and are not part of the
public workflow contract.

## Part I

| Script | Role |
|---|---|
| `scripts/final/part1/table7_full_reproduction.py` | Full Table 7 reproduction with external baselines and UFCE variants. |
| `scripts/final/part1/ufce_only_reproduction.py` | UFCE-only reproduction plus locked runtime/config profile definitions. |
| `scripts/final/part1/ufce_parameter_tuning.py` | Hyper-parameter tuning provenance for final-freeze/tuned settings. |
| `scripts/final/part1/author_raw_posthoc_replay.py` | Original author-core raw replay with post-hoc validity checks. |
| `scripts/final/part1/author_pool_selector_audit.py` | Paired raw/post-hoc/UFCE-FF selector audit on identical author pools. |
| `scripts/final/part1/red_wine_proximity_sensitivity.py` | Red Wine Prox-Euc distance-space sensitivity diagnostic. |
| `scripts/final/part1/ufce_force_flip_experiment.py` | UFCE-FF experiment runner for profile and force-flip checks. |
| `scripts/final/part1/_movie_proximity.py` | Shared Movie proximity helper used by retained Part I scripts. |

## Part II

| Script | Role |
|---|---|
| `scripts/final/part2/parser_benchmark_metrics.py` | Parser/model benchmark metrics for Table 4.12. |
| `scripts/final/part2/conversation_quality_metrics.py` | Main Bank Loan conversation metrics for Tables 4.13-4.14. |
| `scripts/final/part2/no_valid_counterfactual_diagnostics.py` | Diagnostics for no-valid-counterfactual and safe-stop outcomes. |
| `scripts/final/part2/policy_control_evaluation.py` | Policy-control evaluation for constraint behavior. |

## Product And Utilities

Product demo scripts and thesis utility scripts remain tracked when they support
the prototype or environment checks, but they are not the source of Chapter 4
experimental numbers. See `docs/THESIS_TABLE_TO_SCRIPT_MAP.md` for the table
source of truth.
