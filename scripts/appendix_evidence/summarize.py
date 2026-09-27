"""Derive the Appendix 2026-09-27 numbers from public, de-identified evidence."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "appendix_20260927"


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    ledger = rows(EVIDENCE / "multiclass_improvement_ledger.csv")
    by_method = defaultdict(list)
    for item in ledger:
        by_method[item["method"]].append(item)
    methods = {}
    for name, selected in sorted(by_method.items()):
        assert len(selected) == 546, (name, len(selected))
        found = sum(item["feasible_found"] == "True" for item in selected)
        exposed = sum(item["exposed"] == "True" for item in selected)
        runtime_s = sum(float(item["runtime_ms"]) for item in selected) / 1000
        methods[name] = {
            "pairs": len(selected),
            "feasible_found": found,
            "feasible_percent": round(100 * found / len(selected), 1),
            "exposed": exposed,
            "exposed_percent": round(100 * exposed / len(selected), 1),
            "runtime_seconds": round(runtime_s, 1),
            "seconds_per_feasible_found": round(runtime_s / found, 1) if found else None,
            "status_counts": dict(Counter(item["status"] for item in selected)),
        }
    assert len(methods) == 6
    by_pair = defaultdict(set)
    for item in ledger:
        if item["feasible_found"] == "True":
            by_pair[item["pair_index"]].add(item["method"])
    mi = json.loads((EVIDENCE / "upv_2025_mi_sensitivity_20260927" / "summary.json").read_text())
    output = {
        "multiclass": {
            "unique_improvement_pairs": len({item["pair_index"] for item in ledger}),
            "any_method_feasible": len(by_pair),
            "methods": methods,
        },
        "upv_mi": {
            "reference_rows": {run["sample"]: run["rows"] for run in mi["runs"]},
            "wall_seconds": {run["sample"]: round(run["wall_seconds"], 2) for run in mi["runs"]},
            "numeric_top100_overlap_vs_10k": {
                run["sample"]: run["versus_10k"]["numeric_top100_overlap"]
                for run in mi["runs"] if "versus_10k" in run
            },
            "all_406_pair_order_matches_saved": mi["baseline_validation"]["all_406_pair_order_matches_saved"],
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
