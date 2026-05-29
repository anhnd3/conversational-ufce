# Core Author Fixed Design

This document records the implementation plan for `ufce/core_author_fixed`, a corrected fork of the frozen author snapshot in `ufce/core_author`.

## Goal

`core_author_fixed` keeps the author code shape recognizable while making the counterfactual generation contract explicit:

- A returned candidate must be a valid counterfactual, i.e. `model.predict(candidate) == desired_outcome`.
- A returned candidate must be plausible under the LOF plausibility check used by the paper.
- Generation must stay inside user-actionable feature subspaces and must not modify protected features.
- Feasibility is evaluated as validity, plausibility, and an actionability threshold.
- Runtime configuration should expose theory-relevant knobs only.

The original `ufce/core_author` directory remains immutable and can still be used as the upstream reference snapshot.

## Paper-To-Code Gaps Being Addressed

| Area | Paper expectation | Author snapshot gap | Fixed design |
| --- | --- | --- | --- |
| `Single_F` numeric search | Traverse candidate values in the feature interval. | The midpoint index is written as the feature value. | Assign the actual candidate value from the search grid. |
| `Single_F` categorical search | Binary categorical features are reversed. | The categorical branch effectively writes `1.0`. | Assign `1 - current` for binary categories. |
| `Double_F` / `Triple_F` validity | Append only candidates satisfying the desired outcome. | Some branches append candidates even when prediction does not flip. | Candidate appends are guarded by `model.predict(candidate) == desired_outcome`. |
| User feedback intervals | Perturbation map represents feasible bounds. | Current code mostly treats user feedback as one-sided additive deltas. | Support both explicit `[lower, upper]` bounds and legacy delta values. |
| Protected/actionable features | Suggested changes should remain in user-actionable subspace. | Membership/protected checks are inconsistent in several branches. | Normalize checks and keep non-actionable changes out of fixed candidates. |
| Generator plausibility | Algorithms 3/4 require `f(z)=t AND z is plausible`. | Plausibility was mostly measured after generation. | Strict generation filters selected candidates by LOF before returning them. |
| Best CF selection | Choose `z*` by mixed distance `prox_Jac + lambda * prox_Euc`. | Wrapper selection uses numeric Euclidean distance only. | Use Jaccard categorical distance plus normalized MAD-scaled Euclidean distance. |
| Feasibility | `valid AND plausible AND actionable >= threshold`. | Thresholds are hard-coded as integer counts in places. | Use `actionability_threshold` as a ratio, default `0.30`. |

## Runtime Configuration

`core_author_fixed` intentionally exposes only theory-relevant dynamic config:

- `radius`: KDTree neighborhood radius.
- `n_neighbors`: LOF neighbor count.
- `actionability_threshold`: ratio threshold used by feasibility, default `0.30`.
- `contprox_metric`: continuous proximity metric, default `euclidean`.
- `atol`: numeric equality tolerance.

The fixed package does not expose `force_flip`, `ufce_flip_filter`, `min_act`, `min_feas`, or relaxed strictness toggles.

Reason: validity, plausibility, and user-actionable subspace membership are not policy options in the strict paper implementation. If a selected candidate violates any of these constraints, that is a bug rather than a configuration choice.

## Package Layout

`ufce/core_author_fixed` is a package-level fork of the author implementation:

- `ufce.py`: fixed `UFCE` class and metrics.
- `cfmethods.py`: UFCE1/2/3 wrappers and mixed-distance selection.
- `goodness.py`, `evaluations.py`, `data_processing.py`, `datasets.py`, `models.py`, `generate_text_explanations.py`: package-relative clones needed for reproduction.
- `__init__.py`: exports `UFCE`.

The fixed package does not duplicate the author `data/`, `folds/`, or `results/` artifacts. Reproduction scripts use the canonical `ufce/data` and `ufce/data/folds` paths.

## Reproduction Script

`scripts/final/part1/01c_reproduce_core_author_fixed_ufce_only.py` is the fixed-core counterpart of `01b_reproduce_ufce_only.py`.

Default behavior:

- Runs UFCE1, UFCE2, and UFCE3 only.
- Uses `ufce.core_author_fixed`.
- Uses strict paper semantics: selected CFs must flip, pass LOF plausibility, and only modify user-actionable, non-protected features.
- Uses MAD distance scaling for mixed-distance selection and reported continuous proximity.
- Uses Table 7 UFCE-only author/public values embedded in the runner.
- Writes per-dataset summaries, plots when matplotlib is available, and Table 7 delta rows.

Example smoke run:

```bash
python scripts/final/part1/01c_reproduce_core_author_fixed_ufce_only.py --dataset bank --max_folds 1
```

Example full run:

```bash
python scripts/final/part1/01c_reproduce_core_author_fixed_ufce_only.py --dataset all
```

## Expected Verification

Static checks:

```bash
python -m py_compile ufce/core_author_fixed/*.py scripts/final/part1/01c_reproduce_core_author_fixed_ufce_only.py
python -c "from ufce.core_author_fixed import UFCE; print(UFCE)"
```

Algorithm smoke checks:

- UFCE1 selected rows change at most one feature and always flip.
- UFCE2 selected rows change at most two features and always flip.
- UFCE3 selected rows change at most three features and always flip.
- UFCE1/2/3 selected rows pass LOF plausibility before being returned.
- UFCE1/2/3 selected rows only modify features in the user-actionable list and never modify protected features.
- Feasibility rejects non-flipping candidates.

## Claim Boundary

`core_author_fixed` is a strict theory-corrected fork for audit and reproduction analysis. It is not intended to preserve bugs from `core_author` for historical parity. Deviations from public Table 7, including lower coverage after LOF/actionability filtering, should be interpreted as evidence about implementation contracts and hidden assumptions, not as tuning failures.
