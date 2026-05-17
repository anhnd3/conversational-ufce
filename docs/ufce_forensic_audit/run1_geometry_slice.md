# Run 1 - Geometry Slice

Coverage:

- A. Metric Contract and Distance Geometry
- B. Neighborhood Construction
- D. Perturbation Search Logic

Primary files:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/goodness.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py`

## Key Findings

### UF-AUD-01
Algorithm 3 describes midpoint traversal in feature-value space, but the active single-feature path writes the midpoint index to the feature column instead of the candidate feature value. `Single_F` builds `f1_space`, computes `mid`, and then calls `pred_for_binsearch`, which assigns `tempdf.loc[:, feature] = mid` rather than `f1_space[mid]`. This breaks the intended geometry of the search and can move the instance to values that do not belong to the admissible subspace at all.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:328-340`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:367-377`
- Paper: Algorithm 3, PDF p. 10

### UF-AUD-02
The categorical branch of `Single_F` does not implement the paper's reverse-value perturbation. The code writes `1.0` in both branches of the conditional expression, so the operation is not symmetric and does not match the paper's `1 - end` rule.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:385-390`
- Paper: Algorithm 3, PDF p. 10

### UF-AUD-03
The paper defines user feedback as lower and upper feasible bounds and validates both bounds against the neighborhood. The code instead represents `uf` as a single positive delta per feature, sets the lower bound to the factual value, and only clips the upper bound against `nn.max()`. Lower-bound validation against `nn.min()` is absent from both interval constructors.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/data_processing.py:90-228`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:239-325`
- Paper: Sect. 4.1, Sect. 4.4, Algorithm 2, PDF pp. 7-9

### UF-AUD-04
Neighborhood construction is performed with a raw KDTree over the desired-space features and a hard-coded radius of `500` in all three UFCE wrappers. No feature scaling is applied before KDTree construction, and no fallback exists when the neighborhood is empty beyond silently skipping the instance.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:199-211`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py:197-324`
- Paper: Sect. 4.3, PDF p. 7

### UF-AUD-08
Algorithm 5 selects the suitable counterfactual with a mixed categorical-numerical distance, but the active path chooses the nearest row using Euclidean distance over continuous features only. This happens in the shared `find_best_row` helper, so the mismatch affects UFCE1, UFCE2, UFCE3, DiCE, DiCE-UF, and AR whenever multiple candidates are available.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py:20-31`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py:57-195`
- Paper: Algorithm 5, PDF p. 11

### UF-AUD-09
The active distance helpers do not implement the paper's published geometry. The paper states that numerical proximity is MAD-weighted Euclidean distance normalized to `[0, 1]`, while the repository defaults to raw Euclidean distance and offers an optional `mad` mode that is actually MAD-scaled cityblock distance. The mixed helper also substitutes feature-count weighting for the paper's explicit `lambda` term.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/goodness.py:140-261`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:938-958`
- Paper: Eq. (2)-(4), PDF pp. 5-6

## Non-Findings And Notes

- No direct label leakage into the KDTree input was found in the default dataset helpers. Each `get_*_user_constraints` helper drops the outcome column before returning `data_lab1`, so the desired-space matrix excludes the label in the active defaults.
- `get_top_MI_features` is used in the active path, but no standalone discrepancy is recorded here because the paper only requires MI-ordered feature selection, not a stricter estimator contract.
