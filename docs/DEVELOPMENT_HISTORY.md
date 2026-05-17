# Development History

This document explains why historical scripts remain in the repository, organized by phase. This is a reference for understanding the evolution of the codebase — it is **not** an operational runbook. For final evidence reproduction commands, use `docs/FINAL_RUNBOOK.md`.

## Phase 1: UFCE Reproduction & Audit (Part I)

### Objective
Reproduce and audit the original author's Table 7 results from the paper, using frozen dataset configurations.

### Key Scripts Created
- `scripts/reproduce_full_table7_result.py` — Full reproduction of Table 7 with DiCE, DiCE-UF, UFCE comparisons.
- `scripts/reproduce_results_v3.py` — UFCE-only version without baseline methods (for isolating UFCE behavior).
- `scripts/hypertune_ufce_p7.py` — Hyper-tuning pipeline for per-dataset parameter optimization.

### Discovery Outcomes
- Confirmed Table 7 numbers match under frozen configs.
- Discovered force-flip discrepancy: raw UFCE output does not always produce strict valid recourse when the target classifier's prediction changes, but the candidate fails to actually flip the decision (thesis claim about raw vs. validated counterfactuals).

### Artifacts Remaining
All Phase 1 scripts remain because they are **actively used by final wrappers** (`scripts/final/part1/`). They are not obsolete — they implement the core algorithms whose behavior we want to preserve exactly as-is for reproducibility.

## Phase 2: Conversational UFCE MVP & Evidence Pack (Part II)

### Objective
Build a bank-only conversational system that uses a structured LLM parser, validates inputs, runs deterministic runtime orchestration, and provides full artifact logging.

### Key Scripts Created
- Parser benchmark (`llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml`)
- Case study exporter for conversation artifacts
- Phase 2 chapter pack tooling (catalogs, generated materials)

### Discovery Outcomes
- Established parser quality at acceptable levels with local LM Studio model.
- Defined canonical validator as the boundary between LLM output and runtime execution.

## Phase 3.1: Conversation Builder & Negotiation Controller Closeout

### Objective
Implement explicit conversation request builder, negotiation controller state machine, and formal closeout suite for the conversational system.

### Key Scripts Created
- `scripts/run_phase3_1_closeout_suite.py` — Formal closeout with live scenarios covering all public states (NEEDS_CLARIFICATION, CONFLICT, UNSUPPORTED_REQUEST, etc.).

### Discovery Outcomes
- 9/9 live scenarios passed; ready to move to Phase 3.2 product MVP.
- Established bounded clarification loop with restart-required case handling.

### Historical Scripts Still Present
`scripts/run_phase3_1_closeout_suite.py` is kept for traceability but **not called by any final wrapper**. It represents the historical closeout process before we moved to a more automated Part II evidence pipeline.

## Phase 3.2: Local Product MVP (API/UI/SQLite)

### Objective
Turn the conversational system into a local, server-rendered UI product with FastAPI backend, SQLite persistence, and artifact browsing.

### Key Scripts Created
- `scripts/run_phase3_2_demo.py` — Start local demo server
- `scripts/run_phase3_2_product_smoke.py` — Automated smoke against running server
- `scripts/run_phase3_2_acceptance_report.py` — Final acceptance combining automated + manual validation
- `scripts/probe_phase3_2_reproducibility.py` — Reproducibility verification with repeated runs

### Current Status
**Complete.** The product MVP is functional and passes all smoke tests. A post-change manual session should be recorded for the final authoritative acceptance verdict, but this does not block the evidence consolidation effort.

## Why Old Scripts Remain Unchanged

1. **Traceability**: Each script represents a specific discovery step in the thesis journey. Removing them would lose historical context about how we arrived at current conclusions.
2. **Active Dependencies**: Some Phase 1 scripts (`reproduce_full_table7_result.py`, `reproduce_results_v3.py`) are imported by test files and utility scripts, not just final wrappers.
3. **No Algorithm Changes**: The goal was consolidation and reproducibility, not refactoring. Changing any algorithm behavior would require re-validation of all previous evidence.

## Future Archiving (After Thesis Submission)

Once the thesis is submitted and reviewed:
- Scripts marked `HISTORICAL_KEEP` could be moved to a dedicated `archive/` folder if desired.
- Scripts with active test dependencies must remain in place regardless of final wrapper status.
