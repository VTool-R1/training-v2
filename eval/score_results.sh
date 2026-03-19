#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export SCORER_OPENAI_BASE_URL="${SCORER_OPENAI_BASE_URL:-http://localhost:8000/v1}"
export SCORER_OPENAI_API_KEY="${SCORER_OPENAI_API_KEY:-EMPTY}"
export SCORER_MAX_TOKENS="${SCORER_MAX_TOKENS:-8192}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/score_responses.py" --write-scored "$@"

