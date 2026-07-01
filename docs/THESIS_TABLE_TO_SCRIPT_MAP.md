# Thesis Table To Script Map

This map records the source workflow for the Chapter 4 numbers in thesis v3.8.
The script names are semantic; old numeric prefixes were only process-order
bookkeeping.

## Part I: UFCE Methodology And Experiments

| Thesis Table | Evidence | Script And Config | Inputs | Expected Artifact |
|---|---|---|---|---|
| Tables 4.2-4.6 | Published UFCE paper values | `AUTHOR_TABLE7` constants in `table7_full_reproduction.py` and `ufce_only_reproduction.py` | UFCE public datasets/folds | Summary CSV/JSON with author reference columns |
| Tables 4.2-4.6 | Raw reproduction, post-hoc-valid-only, UFCE-FF | `author_pool_selector_audit.py --dataset all --config-profile final_freeze --bundle-mode table7_author_public --mi-k 5 --no-cf 10 --emit-pairs` | `ufce/data`, `ufce/data/folds`, `ufce/core_author` | `author_pool_selector_*` CSVs and emitted factual/CF pairs |
| Tables 4.2-4.6 | Standalone raw author replay | `author_raw_posthoc_replay.py --dataset all --bundle-mode table7_author_public` | Original author core under `ufce/core_author` | raw/post-hoc metric summaries |
| Tables 4.2-4.6 | Tuned/final-freeze config provenance | `ufce_parameter_tuning.py --dataset all --run_stage all` plus profiles in `ufce_only_reproduction.py` | UFCE datasets/folds | tuning summaries and locked profile records |
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

Raw run outputs under `outputs/**` and `llm_eval/outputs/**` are local-only. A
fresh GitHub clone should rely on the retained scripts, small frozen inputs, and
this map rather than committed raw output dumps.
