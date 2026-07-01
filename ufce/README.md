# UFCE Package In This Repository

This folder contains the vendored UFCE implementation used by the thesis codebase. It is not maintained here as a standalone published package. The current repository uses it in two ways:

- Part I reproduction and hyper-tuning of the original UFCE algorithms.
- Part II bank-only conversational evaluation, where UFCE is one backend inside a deterministic runtime and validation shell.

The original UFCE method is described in:

- Muhammad Suffian, Jose M. Alonso-Moral, Alessandro Bogliolo, "Introducing User Feedback-Based Counterfactual Explanations (UFCE)", *International Journal of Computational Intelligence Systems*, 2024.

## What Is In This Folder

| Path | Purpose |
| --- | --- |
| `ufce/core` | active vendored UFCE core used by the current repository |
| `ufce/core_bk2` | archival snapshot of older upstream-style material kept for reference only |
| `ufce/data` | bundled tabular datasets used by reproduction scripts |
| `ufce/model_bundles` | local model-bundle helpers used by the thesis runtime |
| `ufce/wrappers` | thin convenience wrappers |
| `ufce/tests` | UFCE-specific tests maintained in this repo |

## Current Status

The current maintained entry point is:

```python
from ufce import UFCE
```

The active package version marker is:

```python
import ufce
print(ufce.__version__)  # vendored
```

Important practical notes:

- the active implementation lives under `ufce/core`, not `ufce/core_bk2`
- dataset helpers in `ufce/core/data_processing.py` are bank/grad/wine/bupa/movie-specific
- the current helper pipeline trains a Logistic Regression model and returns the fitted model, cross-validation summary, train/test splits, dataset frame, and an optional scaler payload
- movie reproduction uses extra distance-scaling logic in the root reproduction scripts rather than inside the package README examples

## Installation

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The thesis runs in this repository have been executed primarily with Python 3.8 and the pinned dependencies in `requirements.txt`.

## Minimal Import Example

This is the smallest current import/use pattern that matches the repository code:

```python
import pandas as pd

from ufce import UFCE
from ufce.core.data_processing import classify_dataset_getModel, get_bank_user_constraints

df = pd.read_csv("ufce/data/bank.csv")

lr, lr_mean, lr_std, xtest, xtrain, x_all, y, dataset_df, scaler = classify_dataset_getModel(
    df,
    data_name="bank",
)

(
    features,
    catf,
    numf,
    uf,
    f2change,
    outcome_label,
    desired_outcome,
    nbr_features,
    protectf,
    data_lab0,
    data_lab1,
) = get_bank_user_constraints(df)

ufc = UFCE()
mi_pairs = ufc.get_top_MI_features(x_all, features)
print(mi_pairs[:5])
```

## Supported Thesis Workflows

The current repo does not treat `ufce/` as a standalone app. The maintained
workflows live in the root `scripts/final/` directory and use semantic names.

### 1. UFCE-only reproduction across the five datasets

```bash
./.venv/bin/python scripts/final/part1/ufce_only_reproduction.py --dataset all --runtime_profile final_freeze --bundle-mode table7_author_public --out-dir outputs/final/part1/ufce_only_final_freeze
```

What it does:

- runs UFCE1, UFCE2, and UFCE3 only
- uses the tuned/final-freeze profiles used by the thesis
- writes fold-aggregated CSV outputs

### 2. Full Table 7 style comparison

```bash
./.venv/bin/python scripts/final/part1/table7_full_reproduction.py --dataset all
```

What it does:

- runs UFCE, DiCE, DiCE-UF, and AR
- compares reproduced values against author-style reference targets

### 3. Parameter tuning provenance

```bash
./.venv/bin/python scripts/final/part1/ufce_parameter_tuning.py --dataset all --run_stage all --out-dir outputs/final/part1/ufce_parameter_tuning
```

What it does:

- `run1`: tunes gate-style parameters
- `run2`: sweeps `radius` and `n_neighbors`
- documents why thesis reproduction uses tuned/final-freeze settings rather than author defaults

### 4. Bank conversational Part II evaluation

Representative maintained scripts:

- `scripts/final/part2/parser_benchmark_metrics.py`
- `scripts/final/part2/conversation_quality_metrics.py`
- `scripts/final/part2/no_valid_counterfactual_diagnostics.py`
- `scripts/final/part2/policy_control_evaluation.py`

These scripts evaluate UFCE as part of the larger Part II bank-only conversational system rather than as a standalone package.

Detailed table-level provenance is maintained in `docs/THESIS_TABLE_TO_SCRIPT_MAP.md`.

## Testing

Package-level regression check:

```bash
./.venv/bin/pytest ufce/tests/test_lr_bundle.py -q
```

Broader Part II evaluation checks live under `llm/tests/` and can be run with
the retained Part II metric scripts listed above.

## Notes On Legacy Material

`ufce/core_bk2` is intentionally kept as archival material. It should not be treated as the active implementation target for new work.

## Citation

If you use UFCE in your research, cite the original paper:

```bibtex
@article{suffian2024introducing,
  title={Introducing User Feedback-Based Counterfactual Explanations (UFCE)},
  author={Suffian, Muhammad and Alonso-Moral, Jose M and Bogliolo, Alessandro},
  journal={International Journal of Computational Intelligence Systems},
  volume={17},
  number={1},
  pages={123},
  year={2024},
  publisher={Springer}
}
```
