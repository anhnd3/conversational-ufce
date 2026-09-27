# UFCE-FF2/FF3 order check (post-hoc, 2026-09-27)

Scope: the same 546 frozen final-test query-target pairs in Dropout→Enrolled,
Dropout→Graduate, and Enrolled→Graduate. The frozen dataset, train split, model,
feature policy, per-query seed, timeout and MI pair ranking were reused. Each
method was run on every pair once in each new schedule, with a new adapter per
transition-method block. Output from the completed baseline was not changed.

| Method | Historical FF2→FF3 total | New FF3→FF2 total | New FF2→FF3 total | New FF3→FF2 median / P95 | New FF2→FF3 median / P95 |
|---|---:|---:|---:|---:|---:|
| UFCE-FF2 | 860.166 s | 319.519 s (second) | 326.495 s (first) | 0.554 / 0.835 s | 0.568 / 0.858 s |
| UFCE-FF3 | 865.150 s | 251.414 s (first) | 257.898 s (second) | 0.456 / 0.709 s | 0.467 / 0.733 s |

Both methods were about 2–3% slower in the second *run*, irrespective of
whether they ran first or second within each transition. FF3 was about 21%
faster than FF2 in both new schedules. This does not support a consistent
speed advantage from running after the other method.

The new schedules produced identical target-valid and constraint-feasible
availability, proposal counts, exposed-CF counts, status, and schema-error
counts on all 1,092 matched method-query rows. All 145 exposed CF rows had
identical values and verification results between the two new schedules and
the historical run. FF2 found feasible CFs on 77/546 pairs; FF3 on 68/546.

**Timing qualification:** the frozen `ufce/ufce_ff/ufce.py` source checksum
differs from the current source. The current file includes reference-index
caching added after the historical run, among other changes. The new total
latencies therefore cannot replace the historical slide timings or isolate
how much that code change contributed to the large absolute speedup. The
within-current-code order comparison is the valid order-effect check. Query
latencies exclude adapter setup and the shared offline MI precomputation, as
in the historical runner. The MI ranking was read from the freeze manifest,
not computed by FF2 and passed on to FF3.

Artifacts:
- `student_outcomes_order_ff3_before_ff2_20260927/{provenance.json,query_results.csv,candidate_summaries.csv,exposed_candidates.csv,comparison.json}`
- `student_outcomes_order_ff2_before_ff3_20260927/{provenance.json,query_results.csv,candidate_summaries.csv,exposed_candidates.csv,comparison.json}`
- Runner: `scripts/native_multiclass_eval/order_ablation.py`
