#!/bin/bash
set -e

RUN_ID=$(date +%Y%m%d_%H%M%S)
ROOT_OUT="outputs/final_table7_freeze/${RUN_ID}"
mkdir -p "$ROOT_OUT"

echo "Running raw..."
.venv/bin/python scripts/reproduce_results_v3.py \
  --dataset all \
  --runtime_profile final_freeze \
  --bundle_mode final_blindspot_best \
  --ufce_flip_filter 0 \
  --out_dir "$ROOT_OUT/raw"

echo "Running strict..."
.venv/bin/python scripts/reproduce_results_v3.py \
  --dataset all \
  --runtime_profile final_freeze \
  --bundle_mode final_blindspot_best \
  --ufce_flip_filter 1 \
  --out_dir "$ROOT_OUT/strict_validity"

echo "${RUN_ID}" > /tmp/run_id.txt
echo "Done"
