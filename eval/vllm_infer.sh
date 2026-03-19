#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME="${HF_HOME:-/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/.cache/huggingface}"

MODEL_PATH="/vtool/checkpoints/VTool-7B"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

apptainer run --nv \
    --bind : \
    --bind /.cache/huggingface:/.cache/huggingface \
    /vllm-openai_v0.17.1.sif "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    "$@"
