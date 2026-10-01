# UPV-2025 external binary evaluation

This extension is separate from the frozen thesis reproduction. It uses three
Zenodo archives, a student-group split, 75 numeric and six categorical raw
predictors, and the source-class-0 to target-class-1 task. The loader checks
published MD5 values and the expected 464,739 records from 39,364 students.
Downloaded archives and all run outputs stay local under ignored `data/` and
`outputs/` paths. Curated aggregate evidence and provenance manifests for the
defense appendix are committed under `evidence/appendix_20260927/`.

## First stage: source and model audit

```bash
./.venv/bin/python -m scripts.external_binary_eval.runner \
  --data-dir data/upv_2025 \
  --out-dir outputs/external_binary_eval/upv_2025 \
  --stage audit
```

The audit downloads missing archives by default. Use `--no-download` only when
all three are present. It writes `model_bundle_checkpoint.joblib`, model
metrics, schema/split manifests, and `model_quality_gate.json` before stopping
if the model gate fails. The local saved LogisticRegression reached `max_iter`;
the automatic gate is `STOP/CONVERGENCE_WARNING`. The initial runner therefore
does **not** proceed to CF generation on that run.

The later exploratory scripts require this exact local checkpoint at
`outputs/external_binary_eval/upv_2025/model_bundle_checkpoint.joblib`.
Their manifests record an acceptance override for its predictive performance;
they do not turn the failed automatic gate into a pass. A clean clone must
regenerate and inspect the checkpoint and gate before continuing.

The optional training diagnostic warm-starts a separate trace from the saved
coefficients without replacing the primary model:

```bash
./.venv/bin/python -m scripts.external_binary_eval.training_diagnostic \
  --data-dir data/upv_2025 \
  --out-dir outputs/external_binary_eval/upv_2025
```

It logs validation loss and dropout-risk AUC in 100-iteration chunks, stopping
after three chunks without a `1e-5` loss improvement or 2,000 additional
iterations. Scikit-learn 0.24 does not retain SAGA gradient memory in this
checkpoint, so each chunk restarts that optimizer state from saved coefficients.

## Fixed 10% query sample and author alignment

The resumable baseline samples 207 unique held-out students predicted as
class 0, assigns five stratified query folds, and keeps the predictive model
fixed. Run the compatibility pilot, then the primary matrix:

```bash
./.venv/bin/python -m scripts.external_binary_eval.run_ten_percent_eval \
  --stage pilot --data-dir data/upv_2025 \
  --out-dir outputs/external_binary_eval/upv_2025_10pct_5fold

./.venv/bin/python -m scripts.external_binary_eval.run_ten_percent_eval \
  --stage primary --data-dir data/upv_2025 \
  --out-dir outputs/external_binary_eval/upv_2025_10pct_5fold
```

The primary output includes query IDs, status and method summaries, paired
comparisons, and setup/scaling records. The completed local main matrix is
seed 0 on a 10,007-row reference; robustness seeds and the planned 50k/100k
scaling points were not completed. Its original AR adapter reports
`UNSUPPORTED`, and DiCE timed out in the compatibility pilot. Treat these as
adapter/runtime outcomes.

The categorical author-alignment path reuses the same checkpoint and fixed
207 query IDs. Only `dedicacion` may change between TC and TP; the other five
raw categorical predictors remain immutable. Run its pilot and paired primary
stage in order:

```bash
./.venv/bin/python -m scripts.external_binary_eval.author_alignment \
  --stage pilot --data-dir data/upv_2025 \
  --baseline-dir outputs/external_binary_eval/upv_2025_10pct_5fold/primary \
  --out-dir outputs/external_binary_eval/upv_2025_author_alignment_v1

./.venv/bin/python -m scripts.external_binary_eval.author_alignment \
  --stage primary --data-dir data/upv_2025 \
  --baseline-dir outputs/external_binary_eval/upv_2025_10pct_5fold/primary \
  --out-dir outputs/external_binary_eval/upv_2025_author_alignment_v1
```

`report.md`, `method_summary.csv`, MI pair sets, reference-ID manifests, and
categorical-transition tables identify the paired arms and denominators. The
10k MODERATE primary arm has feasible availability `6/207` for FF1 and
`5/207` for each FF2/FF3 digital-plus-category arm. The category switch tests
algorithm behavior; it is not a validated educational intervention. The saved
author-alignment manifest has an `adapters.py` hash that differs from the
current source; reruns should use fresh output directories and manifests.

## Separate comparator recovery

The later recovery runner implements raw-space AR and a DiCE wrapper around
the frozen pipeline. It reads the author-alignment query and reference
manifests, then runs a paired 30-query cohort and a separately labelled
207-query extension. It uses 60 seconds/query for AR and saved UFCE results
and 120 seconds/query for DiCE; timings are not a matched-budget comparison.

```bash
./.venv/bin/python -m scripts.external_binary_eval.comparator_recovery \
  --data-dir data/upv_2025 \
  --baseline-dir outputs/external_binary_eval/upv_2025_author_alignment_v1 \
  --out-dir outputs/external_binary_eval/upv_2025_comparator_recovery_v1

./.venv/bin/python -m scripts.external_binary_eval.postprocess_comparator_recovery \
  --baseline-dir outputs/external_binary_eval/upv_2025_author_alignment_v1 \
  --out-dir outputs/external_binary_eval/upv_2025_comparator_recovery_v1
```

The local 207-query 10k extension records AR `20/207` feasible queries and
DiCE `0/207` with 17 timeouts. The report also notes concurrent CPU load, so
its wall-clock times are descriptive. Read `report.md` and
`expanded_207_report.md` for cohort sizes, budgets, and limitations.

The source/evidence boundary is summarized in
`docs/EXPERIMENT_EXTENSIONS.md`; the clean-clone order is in
`docs/FINAL_RUNBOOK.md`.

## Defense appendix: scale and MI sensitivity

`docs/APPENDIX_QUICK_EXPERIMENTS.md` maps slides 24–25 to the committed
aggregate reports and manifests in `evidence/appendix_20260927/`.
The separate MI sensitivity diagnostic ranks 406 pairs at 10k, 100k, and
full-train reference sizes with eight workers:

```bash
./.venv/bin/python -m scripts.external_binary_eval.mi_sample_sensitivity --workers 8
python3 scripts/appendix_evidence/summarize.py
```

The first command regenerates local diagnostic output; the second recomputes
published aggregates from the de-identified committed evidence. Raw student
profiles, candidate ledgers, and model bundles remain local. Their checksums
are recorded in `evidence/appendix_20260927/LOCAL_RAW_SHA256SUMS.txt`.
