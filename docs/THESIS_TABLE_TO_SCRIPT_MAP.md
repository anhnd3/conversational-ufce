# Thesis Table To Script Map

Target thesis version: Final_v7.6 / Final.
PDF export timestamp: 2026-07-12T16:49:19+07:00.

This map records the source workflow for the Chapter 4 numbers in the final
thesis. The committed headline summary is
`outputs/final/thesis_canonical/summary.json`.
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
| Figure 4.1 | Selected-output flip validity check | `01_author_raw_reproduction.py`, `03_ufce_post_hoc.py`, `04_ufce_ff.py` | UFCE public folds and final-freeze Bank/Graduate/Wine/BUPA/Movie configs | raw/post-hoc/UFCE-FF summaries |
| Red Wine discussion | Prox-Euc sensitivity by distance space | `red_wine_proximity_sensitivity.py` | emitted pairs from `author_pool_selector_audit.py`, Wine metadata/data | distance-space sensitivity CSV/Markdown |

Important: the thesis raw/final-freeze reproduction uses tuned profiles. Do not
replace these commands with the author's default runtime settings.

## Part II: Natural-Language Feedback Prototype

| Thesis Table | Evidence | Script And Config | Inputs | Expected Artifact |
|---|---|---|---|---|
| Table 4.12 | UFCE-FF valid-solution coverage over five datasets | `scripts/final/part1/04_ufce_ff.py` plus the release summary for thesis-final Bank counts | UFCE folds, `final_freeze`, `table7_author_public` | UFCE-FF coverage summary; Bank headline in `outputs/final/thesis_canonical/summary.json` |
| Table 4.13 | Local LLM parser configuration benchmark | `llm_eval/scripts/run_bank_cf_llm_eval.py` and `llm_eval/reporting.py` | `llm_eval/benchmarks`, prompt docs, model config | `llm_eval/reports/part2_phase1_local_llm_evaluation_report.md` plus `llm_eval/outputs/*/{summary.json,config_snapshot.json}` |
| Figure 4.5 | Parser handoff gate on the 250-case Bank Loan set | `scripts/final/part2/run_nl_bank_parse_only_v2.sh` or `nl_bank_bridge_experiment.py --stage eval_parse` | 250 NL Bank cases derived from the 5 original Bank Loan test folds | `parse_acceptance.json`, `parse_quality_progress.json`, `record_reconstruction.csv` |
| Figure 4.6 | Parser error distribution after the handoff gate | `scripts/final/part2/run_nl_bank_parse_only_v2.sh` or `nl_bank_bridge_experiment.py --stage eval_parse` | Completed parser-only artifact | `parse_acceptance.json`, `parse_errors.jsonl`, `record_reconstruction.csv` |
| Figure 4.7 | Bank Loan released-solution counts before and after NL handoff | `scripts/final/part2/nl_bank_bridge_experiment.py`, `scripts/final/part2/failed_cases_impact_check.py` | Completed parser artifact and locked UFCE-FF backend | Canonical counts in `outputs/final/thesis_canonical/summary.json` |
| Table 4.14 | UFCE-FF metric delta after handoff versus the direct full-250 comparator | `scripts/final/part2/nl_bank_bridge_experiment.py`, `scripts/final/part2/failed_cases_impact_check.py` | Accepted 243-case parser subset and direct 250-case comparator | Canonical final-vs-full250 counts in `outputs/final/thesis_canonical/summary.json`; detailed rerun artifacts stay local-only |

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
