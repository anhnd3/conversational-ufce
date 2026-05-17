# Run 2 - Constraints Slice

Coverage:

- C. Counterfactual Validity Enforcement
- E. Constraint Handling
- F. Multi-feature Dependency Modeling
- I. UF Structure

Primary files:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/data_processing.py`

## Key Findings

### UF-AUD-05
`Double_F` does not preserve the paper's validity contract in its numeric-numeric branch. Even when `pred != desired_outcome`, the candidate is still appended to `cfdf` because the concatenation on line 494 sits outside the prediction guard. Since `dfexp` later chooses the nearest row from `cc2`, non-flipping outputs can be surfaced as returned counterfactuals.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:457-495`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py:241-278`
- Paper: Algorithm 4, PDF pp. 10-11

### UF-AUD-06
`Triple_F` repeats the same validity failure pattern. Several branches append `temptempdf` to `cfdf` even when the candidate has not been verified as target-flipping, and none of the search branches invoke LOF before acceptance. The numeric-numeric path is the clearest example because the same candidate is concatenated once conditionally and once unconditionally.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:654-665`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:678-683`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:744-864`
- Paper: Sect. 4.4 and Algorithm 1, PDF pp. 8-11

### UF-AUD-07
The paper's validity-and-plausibility gating is not part of the active search path. `Single_F`, `Double_F`, and `Triple_F` never call `lofn`, and the standalone helper `get_cfs_validated` exists but is not used by the observable orchestration path. Plausibility is deferred to later metric code rather than enforced during generation.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:213-225`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:343-867`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/experiments.py:163-275`
- Paper: Algorithm 3-4, PDF pp. 10-11

### UF-AUD-11
Protected-feature handling is structurally present in the algorithm signatures but omitted by the default dataset presets. All `get_*_user_constraints` helpers set `protectf = []`, so the repository snapshot does not supply any immutable-feature protection even though the paper presents protected features as a core part of the UFCE input contract and gives domain examples such as `Family`.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/data_processing.py:90-228`
- Paper: Sect. 4.5, PDF p. 8; qualitative discussion in Sect. 1 and Fig. 1

### UF-AUD-12
The actionability and feasibility contracts used in the active metric path diverge from the paper. The paper defines actionability as the average number or percentage of changes drawn from the user-specified feature list and defines feasibility as validity plus plausibility plus an actionability threshold. The active code instead uses hard-coded count thresholds (`>= 3` for actionability, `>= 2` for feasibility), does not compute the paper's ratio-based threshold in the observable path, and does not re-check model validity inside feasibility.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:1044-1105`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:1143-1182`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py:217-320`
- Paper: Sect. 3.3, PDF p. 7; Sect. 5.2, PDF p. 13

## Dependency-Model Observation

Area F is only partially problematic. The repository does implement per-target dependency models via `LinearRegression` and `LogisticRegression` trained on all non-target features:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:394-427`

No standalone discrepancy is recorded here because the paper allows feature-value prediction from the available dataset. However:

- the `f_independent` argument is ignored by both model builders
- holdout quality scores are computed and never used to gate the search

This weakens traceability, but the paper does not specify a stricter acceptance rule for `h` and `g`, so the issue remains an observation rather than a separate matrix entry.
