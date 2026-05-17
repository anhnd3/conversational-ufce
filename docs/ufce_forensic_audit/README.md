# UFCE Static Forensic Audit

This directory contains a static paper-vs-code audit of the published UFCE journal paper against the author code snapshot in `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author`.

Audit scope:

- Paper target: `/mnt/d/Workspace/MasterAI/Thesis/2024_Introducing User Feedback-Based Counterfactual Explanations (UFCE).pdf`
- Paper identity: DOI `10.1007/s44196-024-00508-6`
- Audit date: `2026-04-10`
- Audit type: static code-and-theory consistency audit

Out of scope:

- `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core` is a modified local fork and is not the audited implementation.
- No experiment reruns
- No reproduction audit
- No execution-based metric verification
- No refactoring

Primary evidence path:

- Generation stack:
  - `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/ufce.py`
  - `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/cfmethods.py`
  - `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/goodness.py`
- Evaluation stack:
  - `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/evaluations.py`
  - `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/experiments.py`
  - `/mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent/ufce/core_author/data_processing.py`

Evidence discipline:

- README files, notebooks, and precomputed result text are corroboration only.
- No finding in this package relies solely on README or notebook commentary.

Artifacts:

- `claim_ledger.md`
- `run1_geometry_slice.md`
- `run2_constraints_slice.md`
- `run3_ecosystem_slice.md`
- `discrepancy_matrix.md`
- `critical_issue_summary.md`
- `thesis_subsection_draft.md`
- `snapshot_manifest.sha256`
