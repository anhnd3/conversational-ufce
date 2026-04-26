#!/bin/bash
RUN_ID=20260425_165918
ROOT_OUT="outputs/final_table7_freeze/${RUN_ID}"
SNAPSHOT_DIR="$ROOT_OUT/snapshot"
mkdir -p "$SNAPSHOT_DIR"

git rev-parse HEAD > "$SNAPSHOT_DIR/git_commit.txt"
git status --short > "$SNAPSHOT_DIR/git_status_short.txt"
git diff --stat > "$SNAPSHOT_DIR/git_diff_stat.txt"
git diff > "$SNAPSHOT_DIR/git_diff.patch"
git log -1 --stat > "$SNAPSHOT_DIR/git_last_commit_stat.txt"

cat > "$SNAPSHOT_DIR/reproduction_commands.sh" <<EOF
.venv/bin/python scripts/reproduce_results_v3.py \\
  --dataset all \\
  --runtime_profile final_freeze \\
  --bundle_mode final_blindspot_best \\
  --ufce_flip_filter 0 \\
  --out_dir outputs/final_table7_freeze/${RUN_ID}/raw

.venv/bin/python scripts/reproduce_results_v3.py \\
  --dataset all \\
  --runtime_profile final_freeze \\
  --bundle_mode final_blindspot_best \\
  --ufce_flip_filter 1 \\
  --out_dir outputs/final_table7_freeze/${RUN_ID}/strict_validity
EOF

.venv/bin/python - <<'PY' > "$SNAPSHOT_DIR/source_sha256_manifest.json"
import hashlib
import json
from pathlib import Path

paths = [
    "scripts/reproduce_results_v3.py",
    "ufce/core/cfmethods.py",
    "ufce/core/ufce.py",
    "ufce/core/evaluations.py",
    "ufce/core/data_processing.py",
]

out = {}
for p in paths:
    path = Path(p)
    out[p] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
print(json.dumps(out, indent=2))
PY
