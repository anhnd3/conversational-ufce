# Thesis Table To Script Map

This map records the source workflow for the Chapter 4 numbers in thesis v3.8.
The script names are semantic; old numeric prefixes were only process-order
bookkeeping.

## Part I: UFCE Methodology And Experiments

| Thesis Table | Evidence | Script And Config | Inputs | Expected Artifact |
|---|---|---|---|---|
| Tables 4.2-4.6 | Published UFCE paper values | `AUTHOR_TABLE7` constants in `table7_full_reproduction.py` and `02_ufce_core_final_freeze.py` via the shared final-freeze runner | UFCE public datasets/folds | Summary CSV/JSON with author reference columns |
| Tables 4.2-4.6 | Raw author replay | `01_author_raw_reproduction.py --dataset all --bundle-mode table7_author_public` | Original author core under `ufce/core_author` | raw author metric summaries |
| Tables 4.2-4.6 | Thesis raw/final-freeze replay | `02_ufce_core_final_freeze.py --dataset all --runtime-profile final_freeze --bundle-mode table7_author_public` | `ufce/core`, tuned final-freeze profiles, thesis folds | raw final-freeze summary CSV/JSON |
| Tables 4.2-4.6 | Post-hoc-valid-only aggregation | `03_ufce_post_hoc.py --dataset all --bundle-mode table7_author_public` | raw selected UFCE outputs from `ufce/core_author`, thesis folds | post-hoc metric summaries and raw-vs-posthoc comparison |
| Tables 4.2-4.6 | UFCE-FF validity-gated reproduction | `04_ufce_ff.py --dataset all --runtime-profile final_freeze --bundle-mode table7_author_public` | `ufce/ufce_ff`, final-freeze bundle, thesis folds | UFCE-FF summary CSV/JSON |
| Tables 4.2-4.6 | Tuned/final-freeze config provenance | `ufce_parameter_tuning.py --dataset all --run_stage all` plus profiles consumed by `02_ufce_core_final_freeze.py` and `04_ufce_ff.py` | UFCE datasets/folds | tuning summaries and locked profile records |
| Tables 4.7-4.11 | DiCE, AR, and UFCE comparator values | `table7_full_reproduction.py --dataset all` | UFCE datasets/folds | dataset summaries for UFCE, DiCE, DiCE-UF, AR |
| Red Wine discussion | Prox-Euc sensitivity by distance space | `red_wine_proximity_sensitivity.py` | emitted pairs from `author_pool_selector_audit.py`, Wine metadata/data | distance-space sensitivity CSV/Markdown |

Important: the thesis raw/final-freeze reproduction uses tuned profiles. Do not
replace these commands with the author's default runtime settings.

## Part II: Natural-Language Feedback Prototype

| Thesis Table | Evidence | Script And Config | Inputs | Expected Artifact |
|---|---|---|---|---|
| Table 4.12 | Local LLM/parser model selection | `parser_benchmark_metrics.py` | `llm_eval/benchmarks`, prompt docs, model config | parser benchmark metrics JSON/Markdown |
| Table 4.13 | Presented CF invariant checks | `conversation_quality_metrics.py` | frozen Bank Loan conversation corpora and runtime validators | conversation metrics summary |
| Table 4.14 | 200-session outcomes, safe stops, no-recourse counts | `conversation_quality_metrics.py` plus `no_valid_counterfactual_diagnostics.py` | Tier-B Bank sessions and runtime diagnostics | session outcome summaries and diagnostic reports |
| Table 4.14 support | Constraint/policy behavior | `policy_control_evaluation.py` | Bank runtime context and policy registry | policy-control evaluation summary |

## Appendix-Oriented Route 2 Additions

| Appendix Evidence | Evidence | Script And Config | Inputs | Expected Artifact |
|---|---|---|---|---|
| Bank NLP parser quality on the single 250-case test set | parser-only exact reconstruction and runtime-readiness evaluation | `run_nl_bank_parse_only_v2.sh` or `nl_bank_bridge_experiment.py --stage parse` and `--stage eval_parse` | 250 NL cases derived from the 5 original Bank Loan test folds, LM Studio parser, Bank runtime schema | `parse_acceptance.json`, `parse_quality_progress.json`, `record_reconstruction.csv`, `parse_errors.jsonl` |
| Bank NLP -> UFCE-FF strict bridge comparison | direct UFCE-FF versus NL -> parser -> UFCE-FF under the locked final configuration | `nl_bank_bridge_experiment.py --stage all --parse-eval-scope accepted --config-profile final_freeze --bundle-mode table7_author_public --mi-k 5 --no-cf 10 --contprox-metric euclidean --mi-feature-scope configured_actionable` | Parser-accepted NL cases; this is all 250 cases only when strict parser acceptance is 250/250 | `summary.json`, `table_main_bridge.csv`, `table_language_group.csv`, `mismatch_cases.csv` |
| Bank NLP -> UFCE-FF accepted-subset diagnostic | direct and NL-first UFCE-FF on parser-accepted cases without hiding parser loss | run `nl_bank_bridge_experiment.py` stages `direct`, `nl_first`, and `compare` with `--parse-source-dir <parse_dir> --parse-eval-scope accepted --config-profile final_freeze --bundle-mode table7_author_public --mi-k 5 --no-cf 10 --contprox-metric euclidean --mi-feature-scope configured_actionable` | Completed parser-only artifact, accepted case selection, frozen Part I backend path | `evaluation_case_selection.csv`, `direct_ufce_ff_results.csv`, `nl_first_ufce_ff_results.csv`, `summary.json` |
| Bank NLP accepted subset vs full-250 UFCE-FF comparator | full-250 direct comparator and parser-accepted NL-first result share the same locked UFCE-FF backend while reporting parser drop cost | `failed_cases_impact_check.py --parse-out-dir <parse_dir> --full250-out-dir <full250_dir> --accepted-out-dir <accepted_dir>` | Completed parser artifact, full-250 direct UFCE-FF output, accepted-subset NL-first output | `impact_check_7_failed_cases/route2_final_vs_full250.md`, `failed_cases_impact_summary.md` |

Operational note for the appendix workflow:

- Run downstream diagnostics only after a completed parser-only rerun shows
  `error_rate < 0.10` and no missing/API/schema/verifier failures.
- Reserve all-case bridge claims for runs where `ready_for_runtime_count = 250`,
  `exact_reconstruction_count = 250`, and `error_count = 0`.

Raw run outputs under `outputs/**` and `llm_eval/outputs/**` are local-only. A
fresh GitHub clone should rely on the retained scripts, small frozen inputs, and
this map rather than committed raw output dumps.
