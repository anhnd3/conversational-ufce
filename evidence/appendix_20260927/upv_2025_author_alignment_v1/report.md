# UPV-2025 UFCE-FF categorical author-alignment evaluation

The run uses the accepted frozen LogisticRegression snapshot and the saved 207-query, five-fold test cohort. The model and split are reused. The 10,007-row train reference is compared with the full 325,483-row train reference; MI pair rankings for both arms are computed only from the 10,007-row train sample.

Only the raw categorical predictor dedicacion may switch between TC and TP when present. The other five categorical predictors remain immutable. The category switch is an algorithmic capability test, not a validated educational intervention.

## Primary query results

| Reference | MI arm | Method | Valid availability | Constraint-feasible availability | Exposed validity | Constraint compliance | Median ms | P95 ms | ECCF ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10k | ALL_FEATURE_TOP5 | UFCE-FF2 | 0.019 (4/207) | 0.019 (4/207) | 1.000 (4/4) | 1.000 (4/4) | 124.503 | 629.405 | 11273.394 |
| 10k | ALL_FEATURE_TOP5 | UFCE-FF3 | 0.019 (4/207) | 0.019 (4/207) | 1.000 (4/4) | 1.000 (4/4) | 125.574 | 595.601 | 10349.678 |
| 10k | DIGITAL_PAIRS_PLUS_CATEGORY | UFCE-FF2 | 0.029 (6/207) | 0.024 (5/207) | 1.000 (6/6) | 0.833 (5/6) | 134.055 | 14833.438 | 124630.729 |
| 10k | DIGITAL_PAIRS_PLUS_CATEGORY | UFCE-FF3 | 0.029 (6/207) | 0.024 (5/207) | 1.000 (6/6) | 0.833 (5/6) | 133.220 | 8203.370 | 70482.260 |
| 10k | SHARED_FF1 | UFCE-FF1 | 0.053 (11/207) | 0.029 (6/207) | 1.000 (11/11) | 0.545 (6/11) | 153.403 | 2980.685 | 25896.366 |
| full_train | ALL_FEATURE_TOP5 | UFCE-FF2 | 0.019 (4/207) | 0.019 (4/207) | 1.000 (4/4) | 1.000 (4/4) | 64.309 | 573.477 | 8367.090 |
| full_train | ALL_FEATURE_TOP5 | UFCE-FF3 | 0.014 (3/207) | 0.014 (3/207) | 1.000 (3/3) | 1.000 (3/3) | 64.070 | 514.166 | 9822.728 |
| full_train | DIGITAL_PAIRS_PLUS_CATEGORY | UFCE-FF2 | 0.029 (6/207) | 0.024 (5/207) | 1.000 (6/6) | 0.833 (5/6) | 66.408 | 13688.546 | 116495.308 |
| full_train | DIGITAL_PAIRS_PLUS_CATEGORY | UFCE-FF3 | 0.024 (5/207) | 0.019 (4/207) | 1.000 (5/5) | 0.800 (4/5) | 64.740 | 8001.022 | 80885.496 |
| full_train | SHARED_FF1 | UFCE-FF1 | 0.053 (11/207) | 0.029 (6/207) | 1.000 (11/11) | 0.545 (6/11) | 87.207 | 2824.060 | 22264.384 |

## Paired comparisons

See pairwise_comparison.csv for paired percentage-point differences, 2,000-resample bootstrap intervals with seed 42, McNemar results, and paired median latency differences. Categorical 10k versus saved DIGITAL_ONLY estimates the policy-arm difference. Full versus 10k within an MI arm estimates the reference-size difference.

## Categorical transitions

See categorical_transition_summary.csv. Valid/feasible categorical evidence requires an actual TC-to-TP or TP-to-TC change that passes raw-domain, constraint and frozen-target checks. The supplemental TP cohort is shown separately from the paired 207-query estimates.

## Compatibility

DiCE and AR statuses are preserved in adapter_compatibility.csv. They are not included in the new branch when the existing compatibility pilot recorded a timeout or an unsupported raw-constraint adapter.

## Six Table 7 metrics

table7_descriptive_summary.csv contains adapted descriptive Prox-Jac, Prox-Euc, sparsity, actionability, LOF plausibility and feasibility-style values. They characterize this dataset and are not magnitude benchmarks against the published Table 7 values.

Setup, MI, query and verification wall time are separated in adapter_setup.csv and query_results.csv. Full-reference rows are never capped: a conservative feature-box distance bound returns all radius neighbors exactly when it proves the configured radius contains the entire desired-class pool; otherwise the KD-tree path is used. Repeatable auxiliary model fits are cached.
