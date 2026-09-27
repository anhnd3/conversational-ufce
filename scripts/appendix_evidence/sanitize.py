"""Export non-identifying paired indicators from local multiclass query logs."""

import argparse
import csv
import secrets
from collections import defaultdict
from pathlib import Path

IMPROVEMENT = {"dropout_to_enrolled", "dropout_to_graduate", "enrolled_to_graduate"}
FIELDS = ["pair_index", "method", "feasible_found", "exposed", "runtime_ms", "status"]


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = list(read(args.baseline / "final_test_query_results.csv"))
    source += list(read(args.supplement / "final_test_query_results.partial.csv"))
    by_pair = defaultdict(dict)
    for item in source:
        if item["transition"] in IMPROVEMENT:
            by_pair[item["query_id"]][item["method"]] = item
    assert len(by_pair) == 546
    pairs = list(by_pair.values())
    assert all(len(pair) == 6 for pair in pairs)
    secrets.SystemRandom().shuffle(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, pair in enumerate(pairs, 1):
            for name, item in sorted(pair.items()):
                writer.writerow({
                    "pair_index": index,
                    "method": name,
                    "feasible_found": item["constraint_feasible_availability"] == "True",
                    "exposed": int(item["exposed_cf_count"]) > 0,
                    "runtime_ms": item["runtime_ms"],
                    "status": item["status"],
                })


if __name__ == "__main__":
    main()
