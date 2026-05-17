# Run 3 - Ecosystem Slice

Coverage:

- G. Plausibility (LOF)
- H. Baseline Fairness
- J. Hidden Evaluation Logic

Primary files:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/experiments.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/data_processing.py`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py`

## Key Findings

### UF-AUD-09
The repository exposes multiple distance helpers, but the active reporting path does not reproduce the paper's metric contract. Numerical proximity defaults to raw Euclidean distance, the optional MAD path is cityblock-based, and the mixed helper replaces the paper's explicit `lambda` re-scaling factor with feature-count proportions.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/goodness.py:140-261`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py:109-176`
- Paper: Eq. (2)-(4), PDF pp. 5-6

### UF-AUD-10
The paper's joint-distance selection contract is not exercised by the observable evaluation script. `Joint_proximity` exists and `jproxidf` is allocated, but `experiments.py` never calls `Joint_proximity` before printing the joint-proximity table. The active reporting path therefore separates categorical and continuous proximity and never operationalizes the published `delta` metric.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py:16-57`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/experiments.py:129-155`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/experiments.py:218-285`
- Paper: Algorithm 5, PDF p. 11

### UF-AUD-12
The observable metric aggregation rules are materially different from the paper's definitions. `Actionability` reports row counts instead of the paper's average number or percentage of actionable feature changes. `Feasibility` counts rows that pass LOF and a hard-coded actionable-feature count without re-verifying `f(z) = t`. Because invalid candidates can already enter UFCE2 and UFCE3, the feasibility table can absorb non-flipping rows.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py:217-320`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:1068-1182`
- Paper: Sect. 3.3, PDF p. 7; Sect. 5.2, PDF p. 13

### UF-AUD-13
Baseline constraint handling is asymmetric across DiCE, DiCE-UF, and AR. Basic DiCE passes `features_to_vary` but no permitted ranges. DiCE-UF hardcodes Graduate-only feature names in its `permitted_range` map. AR receives `uf`, `f2change`, `scaler`, and `X_train` as parameters but ignores them internally, so its constraints are post-hoc only in the observable path.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py:57-141`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py:143-195`
- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py:217-320`
- Paper: Sect. 5.1.3 and Sect. 5.2, PDF pp. 12-13

### UF-AUD-14
The observable experiment script in the repository snapshot is configured for a narrower scope than the paper reports. The active defaults restrict the evaluation to the Graduate dataset and LR only. This does not prove the paper tables are incorrect, but it does mean the visible orchestration path in the snapshot is not, by default, the same multi-dataset setup described in the article.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/experiments.py:61-62`
- Paper: Abstract; Sect. 5.1.1; Sect. 5.4, PDF pp. 1, 12, 16

### UF-AUD-15
The published evaluation protocol states fivefold cross-validation for Table 3, while the active training helper uses `cv=10`. This is a direct static mismatch between the reported CV protocol and the code visible in the snapshot.

Evidence:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/data_processing.py:65-84`
- Paper: Sect. 5.1.2 and Table 3, PDF p. 12

## Secondary-Path Note

The repository contains two different LOF helper implementations:

- active path: `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py:960-1001` uses `n_neighbors=100`
- secondary helper: `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/goodness.py:442-453` uses `n_neighbors=3`

No standalone discrepancy is recorded from this difference because the paper does not publish a fixed LOF hyperparameter, but the duplication weakens internal traceability and should be treated as a secondary-path ambiguity rather than part of the active contract.
