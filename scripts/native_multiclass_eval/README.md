# Native multiclass UFCE-FF evaluation

This isolated pipeline evaluates UFCE-FF1/2/3 and DiCE on UCI dataset 697 with
one native multinomial logistic classifier. It keeps the binary reproduction
and the external binary evaluation in separate files and output directories.

Run the gated experiment with:

```bash
./.venv/bin/python -m scripts.native_multiclass_eval.runner \
  --data-path data/native_multiclass_eval/uci_students_697.zip \
  --out-dir outputs/native_multiclass_eval/student_outcomes
```

The runner loads the UCI archive, creates a stratified 70/15/15 split, fits the
preprocessor and model on train only, then runs the 20-pair pilot on dev. It
freezes the source/config/model hashes after the pilot gate passes and proceeds
directly to every eligible final-test `(factual, target)` pair. A failed
sanity or pilot gate stops before the final test is read by a generator.

The canonical labels are `Dropout=0`, `Enrolled=1`, and `Graduate=2`. Only the
eight first- and second-semester `evaluations`, `approved`, `grade`, and
`without evaluations` fields can change. All other fields are immutable,
including binary and multi-valued categorical features. Numeric bounds are
query-specific ±0.5 train IQR, clipped to the observed train domain and known
grade/count domains. Academic cross-field invariants are enforced only when
their semantics are explicit and the relation has zero violations in train.
Train-unseen immutable categories keep a reversible encoding extension and
are passed to the frozen one-hot model as unknown categories; they never enter
the train reference space or get changed by either generator.

## Reusable train-only MI ranking

UFCE-FF2 and UFCE-FF3 use the same ranked MI feature pairs. MI is computed
once on all 3,096 **train** rows before the pilot, outside both methods'
per-query timers. Dev/test rows are never used to rank features. The pairs are
saved in mi_cache.json with a fingerprint of the frozen encoded train
reference, row IDs, dataset checksum, feature policy and MI implementation.
A changed training reference fails validation instead of silently reusing
stale ranks.

To prepare a cache once for future CF batches using the same train reference:

```bash
./.venv/bin/python -m scripts.native_multiclass_eval.mi_cache \
  --data-path data/native_multiclass_eval/uci_students_697.zip \
  --out outputs/native_multiclass_eval/student_outcomes_train_mi.json \
  --verify-freeze outputs/native_multiclass_eval/student_outcomes_20260924_resume1/freeze_manifest.json
```

A new run can use it with
`--mi-cache outputs/native_multiclass_eval/student_outcomes_train_mi.json`.
Without that option, the runner computes the same MI ranking once and saves a
local cache before the pilot. The cache is copied into each new run directory
and its checksum is frozen; resumed runs reuse the frozen pairs. The historical
completed experiment and its reported timings are unchanged. The recorded
`precompute_seconds` is a one-time offline cost, not part of FF2 or FF3
per-query latency. A serving path using the same frozen train reference
can call load_mi_cache and pass its mi_pairs to UFCEAdapter for incoming
factual rows. Adding or retraining reference rows requires a new cache.

The output directory includes split and query manifests, model/policy/freeze
manifests, pilot and final per-query/candidate/exposed-CF tables, transition
summaries, paired bootstrap comparisons, and the final run status. Transition
availability has numerator/denominator and Wilson 95% intervals. Paired
UFCE-FF versus DiCE differences use 2,000 bootstrap resamples with seed 42.
All methods request at most five exposed CFs; DiCE's fixed random-search pool
size (1,000 draws) and the shared 60-second per-query timeout are recorded in
the freeze manifest. A transition with no exposed CF reports validity and
constraint satisfaction as N/A.

The run records predicted proposals in a deduplicated ledger. Predictions in
the source class, requested target class, and third class are counted
separately; third-class proposals contribute to `OFF_TARGET`. Exposed rows are
rechecked in raw feature space and fail closed if target or constraints do not
hold.

During a full run, the runner atomically refreshes partial result tables every
10 completed queries and records the last transition/method/query in
`final_test_progress.json`. A stopped run can continue in a new output
folder from a preserved checkpoint, reusing only completed method/query rows:

```bash
./.venv/bin/python -m scripts.native_multiclass_eval.runner \
  --data-path data/native_multiclass_eval/uci_students_697.zip \
  --no-download \
  --resume-from outputs/native_multiclass_eval/<checkpoint-folder> \
  --out-dir outputs/native_multiclass_eval/<new-resume-folder>
```

Resume validates the saved split, model, policy, freeze manifest and completed
query/candidate ledgers before continuing. The source checkpoint remains
unchanged; pilot, model fitting and completed final-test queries are reused.

## Post-hoc DiCE backend comparison

After the canonical native-multiclass run completes, the supplementary runner
can evaluate DiCE's genetic and KD-tree methods on the same frozen split,
model, feature policy, and final-test query-target manifest. It writes to a
separate output folder and does not rerun or modify UFCE or DiCE-random rows.
The run first checks a 20-pair dev pilot, then automatically evaluates all
1,328 final-test query-target pairs per backend if the pilot has no runtime or
schema errors.

```bash
PYTHONPATH=. ./.venv/bin/python -m scripts.native_multiclass_eval.dice_variants \
  --baseline-dir outputs/native_multiclass_eval/student_outcomes_20260924_resume1 \
  --out-dir outputs/native_multiclass_eval/student_outcomes_dice_variants_20260925_retry2
```

The report labels this as a post-hoc analysis because the canonical final test
has already been inspected. All backends return at most five CFs and share a
60-second per-query timeout; their internal search budgets differ. The genetic
backend uses DiCE's 500-iteration default and KD-tree initialization, while
KD-tree searches the frozen training reference.

## Defense appendix: completed improvement slice

`docs/APPENDIX_QUICK_EXPERIMENTS.md` maps slide 26–27 to a de-identified paired
ledger in `evidence/appendix_20260927/`. The baseline completed all 1,328
final-test pairs; the post-hoc DiCE variant run was interrupted after both
backends completed the 546 improvement-direction pairs. The local file name `final_test_query_results.partial.csv` describes the
*whole* variant run; the 546-pair slice in the public ledger is complete
for each backend. `scripts/appendix_evidence/sanitize.py` records the
transformation from local per-query logs to the released ledger.

```bash
python3 scripts/appendix_evidence/summarize.py
```

The script derives F, X, runtime, and timeout counts from the committed
de-identified paired ledger. For exact historical timing, preserve the source SHA-256
values in the baseline freeze manifest; three baseline source files have
subsequently changed, as documented in the appendix guide.
