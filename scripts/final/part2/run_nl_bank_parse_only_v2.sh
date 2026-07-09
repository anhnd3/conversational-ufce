#!/usr/bin/env bash
set -euo pipefail

ROOT="${UFCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

MODEL_ALIAS="${MODEL_ALIAS:-qwen3-14b}"
API_BASE="${API_BASE:-http://127.0.0.1:1234}"
TIMEOUT_S="${TIMEOUT_S:-600}"
SEED="${SEED:-42}"
CASES_PER_GROUP="${CASES_PER_GROUP:-10}"
OUT_DIR="${OUT_DIR:-outputs/final/part2/nl_bank_bridge_v2_parse_only_$(date +%Y%m%d_%H%M%S)}"
LMSTUDIO_STREAM="${LMSTUDIO_STREAM:-1}"
PROGRESS_ARG=()
STREAM_ARG=()

if [[ "${NO_PROGRESS:-0}" == "1" ]]; then
  PROGRESS_ARG=(--no-progress)
fi

if [[ "$LMSTUDIO_STREAM" == "1" || "$LMSTUDIO_STREAM" == "true" || "$LMSTUDIO_STREAM" == "yes" || "$LMSTUDIO_STREAM" == "on" ]]; then
  STREAM_ARG=(--lmstudio-stream)
else
  STREAM_ARG=(--no-lmstudio-stream)
fi

cd "$ROOT"
mkdir -p "$OUT_DIR"
LOG_PATH="${LOG_PATH:-$OUT_DIR/run.log}"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[parse-only] python=$PYTHON"
echo "[parse-only] out_dir=$OUT_DIR"
echo "[parse-only] model=$MODEL_ALIAS api_base=$API_BASE timeout_s=$TIMEOUT_S"
echo "[parse-only] lmstudio_stream=$LMSTUDIO_STREAM"
echo "[parse-only] log=$LOG_PATH"

"$PYTHON" scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage build_inputs \
  --seed "$SEED" \
  --cases-per-group "$CASES_PER_GROUP" \
  --out-dir "$OUT_DIR" \
  --model-alias "$MODEL_ALIAS" \
  --api-base "$API_BASE" \
  --timeout-s "$TIMEOUT_S" \
  "${STREAM_ARG[@]}" \
  "${PROGRESS_ARG[@]}"

set +e
"$PYTHON" scripts/final/part2/nl_bank_bridge_experiment.py \
  --stage parse \
  --seed "$SEED" \
  --cases-per-group "$CASES_PER_GROUP" \
  --out-dir "$OUT_DIR" \
  --model-alias "$MODEL_ALIAS" \
  --api-base "$API_BASE" \
  --timeout-s "$TIMEOUT_S" \
  "${STREAM_ARG[@]}" \
  "${PROGRESS_ARG[@]}"
PARSE_STATUS=$?
set -e

if [[ -f "$OUT_DIR/parsed_requests.jsonl" ]]; then
  set +e
  "$PYTHON" scripts/final/part2/nl_bank_bridge_experiment.py \
    --stage eval_parse \
    --out-dir "$OUT_DIR" \
    --model-alias "$MODEL_ALIAS" \
    --api-base "$API_BASE" \
    --timeout-s "$TIMEOUT_S" \
    "${STREAM_ARG[@]}"
  EVAL_STATUS=$?
  set -e
else
  echo "[parse-only] parsed_requests.jsonl was not produced; skipping eval_parse."
  EVAL_STATUS=1
fi

echo "[parse-only] artifacts:"
echo "  $OUT_DIR/bank_250_queries.csv"
echo "  $OUT_DIR/parsed_requests.jsonl"
echo "  $OUT_DIR/parse_errors.jsonl"
echo "  $OUT_DIR/parse_quality_progress.json"
echo "  $OUT_DIR/record_reconstruction.csv"
echo "  $OUT_DIR/parse_acceptance.json"
echo "  $LOG_PATH"

if [[ "$PARSE_STATUS" -ne 0 ]]; then
  exit "$PARSE_STATUS"
fi
exit "$EVAL_STATUS"
