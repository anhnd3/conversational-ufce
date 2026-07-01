# Final Evidence Policy

The repository keeps only thesis-facing source, small frozen inputs, lightweight
reports, and tests needed to support the methodology and experiments in the
thesis. Large/generated artifacts remain local-only.

## Tracked In Git

Tracked evidence must satisfy at least one of these conditions:

- It is a script needed to reproduce or audit Chapter 4 tables.
- It defines a locked configuration used by the thesis, including tuned/final-freeze UFCE parameters.
- It is a small frozen input corpus, benchmark, schema, prompt, or model manifest required by those scripts.
- It is a lightweight report that records a final model/config selection.
- It is a focused test for a retained thesis workflow.

## Local-Only

These files stay on the machine but are removed from Git tracking:

- generated output folders such as `outputs/**` and `llm_eval/outputs/**`
- original author result plots under `ufce/core_author/results/**`
- Windows `desktop.ini` files
- temporary workspaces such as `tmp/**` and `experiments/**`
- historical `scripts/archieve/**` workflows after their active references are replaced
- numbered audit/closeout/checkpoint scripts in `scripts/final/part1` and `scripts/final/part2` that are not part of the thesis-facing workflow list
- forensic/development notes not cited by the final thesis evidence map

## Reproducibility Rule

The GitHub-facing repo must preserve the parameter provenance behind the thesis
numbers. In particular, Part I raw/final-freeze reproduction is not treated as
the author's default configuration; `ufce_parameter_tuning.py` and
`ufce_only_reproduction.py` remain tracked because they document and expose the
tuned profiles used by the experiments.

## Output Policy

Scripts should write new artifacts under `outputs/final/...` or another ignored
output directory. If a result must be cited in the thesis, keep a compact report
or table map in `docs/`, not a full raw output dump.
