# Final Evidence Policy

The repository keeps thesis-facing source, extension runners, small frozen
inputs, lightweight reports, and focused tests. Large or identifying raw
artifacts remain local-only. The curated,
de-identified defense appendix in `evidence/appendix_20260927/` is a deliberate
exception for aggregate results and provenance manifests.

## Tracked In Git

Tracked evidence must satisfy at least one of these conditions:

- It is a script needed to reproduce or audit Chapter 4 tables.
- It defines a locked configuration used by the thesis, including tuned/final-freeze UFCE parameters.
- It is a small frozen input corpus, benchmark, schema, prompt, or model manifest required by those scripts.
- It is a lightweight report that records a final model/config selection.
- It is a focused test for a retained thesis workflow.
- It is a runner or focused test for the separately labelled native-multiclass
  or UPV-2025 evaluation extension.
- It is a compact evidence note that states an extension run's status,
  denominators, and limits without publishing raw records.
- It is a curated defense-appendix ledger, aggregate report, or manifest that
  has passed the appendix publication boundary documented in
  `docs/APPENDIX_QUICK_EXPERIMENTS.md`.

## Local-Only

These files stay on the machine but are removed from Git tracking:

- generated output folders such as `outputs/**` and `llm_eval/outputs/**`
- original author result plots under `ufce/core_author/results/**`
- Windows `desktop.ini` files
- temporary workspaces such as `tmp/**`, `.tmp/**`, and `experiments/**`
- historical `scripts/archieve/**` workflows after their active references are replaced
- numbered audit/closeout/checkpoint scripts in `scripts/final/part1` and `scripts/final/part2` that are not part of the thesis-facing workflow list
- forensic/development notes not cited by the final thesis evidence map
- thesis draft folders such as `docs/thesis/**` and generated report folders such as `docs/reports/**`
- downloaded UCI and UPV source archives under `data/**`
- generated native-multiclass and UPV output trees under `outputs/**`, including
  model checkpoints, query-level rows, and partial-run checkpoints
- patch backup/reject files in the external evaluation source directory

## Reproducibility Rule

The GitHub-facing repo must preserve the parameter provenance behind the thesis
numbers. In particular, Part I raw/final-freeze reproduction is not treated as
the author's default configuration. The thesis-facing Part I scripts are split
by semantic path: `ufce/core_author` raw replay, `ufce/core` final-freeze replay,
post-hoc valid-only aggregation over raw selected outputs, and `ufce/ufce_ff`.
`ufce_parameter_tuning.py` and the canonical scripts under
`scripts/final/part1/0*.py` remain tracked because they document and expose the
tuned profiles and final core variants used by the experiments.

## Output Policy

Scripts should write new artifacts under `outputs/final/...` or another ignored
output directory. If a result must be cited in the thesis, keep a compact report
or table map in `docs/`, not a full raw output dump.

The later evaluation extensions write to `outputs/native_multiclass_eval/` and
`outputs/external_binary_eval/`. Their code, tests, and compact provenance note
are retained. The curated appendix is also retained, while raw third-party
archives, identifying query/candidate tables, learned model bundles, and
interim checkpoints remain local. A model gate that failed, a partial
supplement, or a post-hoc comparison must keep that status in any summary.
