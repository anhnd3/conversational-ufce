#!/usr/bin/env bash
set -euo pipefail

# Re-run Movie run2 grid after ranking-logic changes.
# Usage:
#   bash scripts/rerun_movie_run2_minmax100.sh
#   bash scripts/rerun_movie_run2_minmax100.sh <out_dir>
#
# Default out_dir is timestamped to avoid mixing with old leaderboard rows.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR_DEFAULT="${ROOT_DIR}/archive/part1_old_runs/hypertune_out/movie_run2_minmax100_flip0_rerun_${TS}"
OUT_DIR="${1:-${OUT_DIR_DEFAULT}}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

echo "Running Movie run2 rerun..."
echo "Output dir: ${OUT_DIR}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/hypertune_ufce_p7.py \
  --dataset movie \
  --run_stage run2 \
  --out_dir "${OUT_DIR}" \
  --radius_grid 5,10,15,20,30,40,60,80,120 \
  --n_neighbors_grid 50,100,200,400 \
  --min_act 1 \
  --min_feas 1 \
  --ufce_flip_filter 0 \
  --contprox_metric euclidean \
  --skip_existing 0 \
  --print_top_k 3

echo
echo "Done."
echo "Leaderboard: ${OUT_DIR}/leaderboard_run2.csv"
echo "Best config: ${OUT_DIR}/best_run2.json"
echo "Comparison (all): ${OUT_DIR}/comparison_run2_all_configs.csv"
echo "Comparison (best): ${OUT_DIR}/comparison_run2_best.csv"
