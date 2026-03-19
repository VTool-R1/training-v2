#!/usr/bin/env bash

set -xeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VTOOL_JUDGE_SCHEME=${VTOOL_JUDGE_SCHEME:-http}
export VTOOL_JUDGE_HOST=${VTOOL_JUDGE_HOST:-localhost}
export VTOOL_JUDGE_PORT=${VTOOL_JUDGE_PORT:-8000}
export VTOOL_JUDGE_API_BASE=${VTOOL_JUDGE_API_BASE:-${VTOOL_JUDGE_SCHEME}://${VTOOL_JUDGE_HOST}:${VTOOL_JUDGE_PORT}/v1}
export VTOOL_JUDGE_USE_ENDPOINT_DEFAULT=${VTOOL_JUDGE_USE_ENDPOINT_DEFAULT:-1}

MODEL_PATH=${MODEL_PATH:-models/Qwen2.5-VL-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/verifier-agent/refocus_chart/test.parquet}
VAL_FILE=${VAL_FILE:-/verifier-agent/refocus_chart/test.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
ROLLOUT_N=${ROLLOUT_N:-8}
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-24576}
INFER_MAX_TOKEN_LEN_PER_GPU=${INFER_MAX_TOKEN_LEN_PER_GPU:-24576}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-2}
ROLLOUT_SKIP_TOKENIZER_INIT=${ROLLOUT_SKIP_TOKENIZER_INIT:-false}
VLLM_COMPILATION_CUDAGRAPH_CAPTURE_SIZES=${VLLM_COMPILATION_CUDAGRAPH_CAPTURE_SIZES:-[1,2,4,8,16]}
ROLLOUT_CORRECTION_IS=${ROLLOUT_CORRECTION_IS:-token}
ROLLOUT_CORRECTION_IS_THRESHOLD=${ROLLOUT_CORRECTION_IS_THRESHOLD:-2.0}
ROLLOUT_CORRECTION_IS_BATCH_NORMALIZE=${ROLLOUT_CORRECTION_IS_BATCH_NORMALIZE:-false}
ROLLOUT_CORRECTION_RS=${ROLLOUT_CORRECTION_RS:-null}
ROLLOUT_CORRECTION_RS_THRESHOLD=${ROLLOUT_CORRECTION_RS_THRESHOLD:-null}
ROLLOUT_CORRECTION_BYPASS_MODE=${ROLLOUT_CORRECTION_BYPASS_MODE:-false}
ROLLOUT_CORRECTION_LOSS_TYPE=${ROLLOUT_CORRECTION_LOSS_TYPE:-ppo_clip}
USE_LEGACY_WORKER_IMPL=${USE_LEGACY_WORKER_IMPL:-auto}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-40}

python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/recipe/vtool" \
    --config-name=refocus_multiturn_grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.validation_shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$INFER_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.skip_tokenizer_init=${ROLLOUT_SKIP_TOKENIZER_INIT} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.quantization=fp8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="$VLLM_COMPILATION_CUDAGRAPH_CAPTURE_SIZES" \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_CORRECTION_IS} \
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_CORRECTION_IS_THRESHOLD} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_CORRECTION_IS_BATCH_NORMALIZE} \
    algorithm.rollout_correction.rollout_rs=${ROLLOUT_CORRECTION_RS} \
    algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_CORRECTION_RS_THRESHOLD} \
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS_MODE} \
    algorithm.rollout_correction.loss_type=${ROLLOUT_CORRECTION_LOSS_TYPE} \
    trainer.ray_wait_register_center_timeout=2000 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=vtool \
    trainer.experiment_name=3b_chart_fp8 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL} \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS" \
    "$@"
