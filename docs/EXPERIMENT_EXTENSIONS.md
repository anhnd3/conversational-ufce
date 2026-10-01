# Evaluation Extensions After the Final Thesis Snapshot

The final thesis routes and Chapter 4 claims remain tied to Final_v7.6 and
`outputs/final/thesis_canonical/summary.json`. The experiments below extend the
evaluation to a native three-class task and a substantially larger external
binary dataset. They use separate runners and ignored output directories; their
results are not replacements for the five-dataset reproduction or the Bank Loan
natural-language bridge. Curated, de-identified evidence for the 2026-09-27
defense appendix is committed under `evidence/appendix_20260927/` and mapped
in `docs/APPENDIX_QUICK_EXPERIMENTS.md`.

## Native multiclass: UCI Student Outcomes

- Source: UCI dataset 697, *Predict Students' Dropout and Academic Success*;
  the loader downloads the source archive and checks its 4,424 rows, 36 input
  features, class labels, and schema. The local archive SHA-256 for the
  completed run is `e90e55fd65ec462ae283ebeb2cca409319e3460ed898d8754f62fb35cc83a65d`.
- Model and split: one train-only fitted multinomial logistic classifier;
  stratified 70/15/15 train/dev/test split. The final test contains 664 rows
  and 1,328 ordered factual-to-target pairs across six transitions.
- Methods: UFCE-FF1/2/3 and DiCE random search. The same frozen model, eight
  actionable first-/second-semester fields, raw-domain checks, and 60-second
  per-query budget apply to all methods. Target-class verification is explicit:
  a third-class prediction is off-target, not a valid flip.
- Gates: a 20-pair dev pilot must pass before final-test generation. Train-only
  MI pairs are computed once for FF2/FF3, recorded with a reference fingerprint,
  and can be reused only with the matching frozen reference.
- Local completed baseline:
  `outputs/native_multiclass_eval/student_outcomes_20260924_resume1/` has
  `run_status.json: COMPLETE`, a passing model/pilot gate, 1,328 pairs per
  method, zero recorded runtime/schema errors, and zero exposed target or
  constraint failures. The counts of queries with a constraint-feasible CF
  were FF1 `113/1328`, FF2 `110/1328`, FF3 `217/1328`, DiCE `382/1328`.
  These are availability counts, not numbers of exposed CF rows or evidence
  that the methods have matched internal search budgets.
- The saved baseline's source-hash manifest differs from the current tree for
  `config.py`, `runner.py`, and shared `ufce/ufce_ff/ufce.py`. The figures above
  describe that frozen local run; a fresh run of this branch needs a new output
  directory and may produce different values.
- The DiCE genetic/KD-tree backend supplement is post-hoc. Its local
  `student_outcomes_dice_variants_20260925_retry2` directory has a partial
  final-test checkpoint (`1440/2656` method-query rows), so no complete
  six-transition supplement claim follows. Both backends did complete the
  546-pair improvement-direction slice used in the committed appendix ledger.
  Order ablations are also post-hoc diagnostics rather than part of the frozen
  baseline.

Run and resume commands are in `scripts/native_multiclass_eval/README.md` and
`docs/FINAL_RUNBOOK.md`.

## Larger external binary evaluation: UPV-2025

- Source: three UPV archives (2018, 2021, 2022) from the Zenodo record pinned
  in `scripts/external_binary_eval/config.py`. The loader validates published
  MD5 values, 464,739 rows, 39,364 students, and the 81-feature schema
  (75 numeric, six categorical). The student-group train/test split keeps
  records from one student together.
- Frozen model: a LogisticRegression snapshot fitted on the 325,483-row train
  partition. Its recorded automatic quality gate is `STOP` because training
  reached `max_iter` without convergence. The later exploratory CF runs use
  that same saved checkpoint under an explicitly recorded acceptance override;
  they must not be described as having passed the original automatic gate.
  The model checkpoint and raw generated manifests stay local and must be
  regenerated for a clean clone before the downstream scripts can run. Curated
  aggregate manifests and reports are committed in the defense appendix.
- Query design: 207 unique held-out students predicted as source class 0,
  sampled at 10% and split into five query folds. Those folds partition fixed
  queries; they do not refit the predictive model. The sampled train reference
  has 10,007 rows, and the full train reference has 325,483 rows.
- Baseline and author-alignment: `run_ten_percent_eval.py` creates the saved
  binary baseline. `author_alignment.py` then tests the `dedicacion` TC/TP
  category change and paired reference/MI variants on the same 207 IDs. In the
  local completed primary MODERATE cohort, the 10k `SHARED_FF1` arm has
  `6/207` feasible queries; `DIGITAL_PAIRS_PLUS_CATEGORY` gives FF2 and FF3
  `5/207` each. The category switch is an algorithmic capability test, not a
  validated educational intervention. Only this paired primary cohort should
  be used for its reference/MI comparisons.
- Comparator recovery: the later raw-space AR adapter and DiCE wrapper are
  separate from the original compatibility pilot where AR was unsupported and
  DiCE timed out. The recovery report covers a primary paired 30-query cohort
  and a separate 207-query 10k extension. In that extension AR exposed feasible
  CFs for `20/207`; DiCE exposed none and timed out on `17/207`. DiCE used
  120 seconds/query while saved UFCE and AR used 60, and the report records
  shared-host timing contention. These are descriptive results, not a matched
  budget or isolated-runtime performance ranking.
- The original 10% primary matrix did not complete all planned robustness
  seeds or 50k/100k scaling points. Do not infer a full scaling curve from the
  completed 10k and targeted full-reference comparisons.
- The author-alignment manifest's saved `adapters.py` hash differs from the
  current source tree. Its numerical rows are historical local evidence; a
  clean rerun must keep a new manifest and should not assume identical counts.

Run order, checkpoint prerequisites, and output locations are in
`scripts/external_binary_eval/README.md` and `docs/FINAL_RUNBOOK.md`.

## Evidence and publication boundary

The values above summarize local generated reports and the curated appendix
evidence. Raw outputs and third-party archives under `outputs/**` and `data/**`
are ignored; `evidence/appendix_20260927/` contains de-identified ledgers,
aggregate reports, and provenance manifests without raw student records or
model bundles. A fresh clone can obtain the UCI archive automatically and the
UPV archives through the pinned Zenodo loader, but UPV
follow-on runs also require the regenerated model checkpoint and their preceding
stage manifests. Exact local run timestamps and absolute paths are diagnostic
only; the code and frozen manifests define each rerun's provenance.
