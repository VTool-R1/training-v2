#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/VTool-7B}"
DATA_FILE="${DATA_FILE:-/verifier-agent/refocus_chart/test.parquet}"
SERVER_BASE_URL="http://0.0.0.0:8000/v1"
SERVER_API_KEY="${SERVER_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
SERVER_MODEL="${SERVER_MODEL:-${OPENAI_MODEL:-}}"

if [[ -z "${SERVER_BASE_URL}" ]]; then
  echo "Missing endpoint. Set OPENAI_BASE_URL or SERVER_BASE_URL to your OpenAI-compatible host, e.g. http://HOST:PORT/v1" >&2
  exit 2
fi

ARGS=(
  --model "${MODEL_PATH}"
  --server-base-url "${SERVER_BASE_URL}"
  --server-api-key "${SERVER_API_KEY}"
  --data-file "${DATA_FILE}"
  --preview-images
  --preview-image-sample-rate 1.0
  --preview-image-output-root "${SCRIPT_DIR}/preview"
)

if [[ -n "${SERVER_MODEL}" ]]; then
  ARGS+=(--server-model "${SERVER_MODEL}")
fi

python3 "${SCRIPT_DIR}/run_eval.py" \
  "${ARGS[@]}" \
  "$@"
