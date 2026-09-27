# UFCE External Binary Evaluation v1.2

This workflow is isolated from the thesis reproduction runners and writes only
to the directory passed through `--out-dir`.

```bash
# Audit/download, schema-check and fit the frozen predictive model
./.venv/bin/python -m scripts.external_binary_eval.runner \
  --data-dir /path/to/upv-data \
  --out-dir outputs/external_binary_eval/upv_2025 \
  --stage audit

# 20-query compatibility pilot
./.venv/bin/python -m scripts.external_binary_eval.runner \
  --data-dir /path/to/upv-data \
  --out-dir outputs/external_binary_eval/upv_2025_pilot \
  --stage pilot

# Primary seed-0 matrix, robustness seeds, or the full matrix plus scaling
./.venv/bin/python -m scripts.external_binary_eval.runner \
  --data-dir /path/to/upv-data \
  --out-dir outputs/external_binary_eval/upv_2025_primary \
  --stage primary
```

The loader accepts the three Zenodo archives directly and validates their
published MD5 checksums.  `--no-download` requires archives already present in
`--data-dir`.  A failed predictive-model quality gate stops before CF
generation, as required by the frozen plan.

To diagnose the existing fitted LogisticRegression checkpoint without
re-fitting preprocessing or replacing the primary model, run:

```bash
./.venv/bin/python -m scripts.external_binary_eval.training_diagnostic \
  --data-dir data/upv_2025 \
  --out-dir outputs/external_binary_eval/upv_2025
```

This warm-starts coefficients from `model_bundle_checkpoint.joblib`, logs
training/validation losses and dropout-risk validation AUCs per 100-iteration
chunk, and stops after three chunks without a validation-loss improvement of
at least `1e-5` (or after 2,000 additional iterations). Since scikit-learn
0.24 does not save SAGA's gradient memory in the estimator checkpoint, each
chunk starts fresh SAGA optimizer memory from the previous coefficients. The
trace and diagnostic snapshots are kept separately from the primary model.

`AR` is deliberately fail-closed as `UNSUPPORTED` in the default adapter: the
installed ActionSet API cannot express the frozen one-hot preprocessor together
with raw 81-feature constraints without silently changing the model contract.

## Defense appendix: UPV scale and MI sensitivity

The published evidence and limitations are in
`docs/APPENDIX_QUICK_EXPERIMENTS.md`; public aggregate evidence is under
`evidence/appendix_20260927/upv_2025_author_alignment_v1/` and
`evidence/appendix_20260927/upv_2025_mi_sensitivity_20260927/`.
The latter reruns only the 406 MI pair rankings with 8 workers:

```bash
./.venv/bin/python -m scripts.external_binary_eval.mi_sample_sensitivity --workers 8
```

The former uses the saved 207-query cohort and the author-alignment runner.
Its model-quality artifact records `STOP` for a convergence warning; treat
its CF rows as exploratory diagnostics. The public package retains aggregate reports, policy, model provenance, and
MI rankings. Per-query profiles and candidate ledgers remain local; their
checksums are recorded in `evidence/appendix_20260927/LOCAL_RAW_SHA256SUMS.txt`.
