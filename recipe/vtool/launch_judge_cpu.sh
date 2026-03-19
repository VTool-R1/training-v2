apptainer run --ipc --cleanenv \
  --bind /Qwen3-4B:/Qwen3-4B \
  --env VLLM_TARGET_DEVICE=cpu \
  --env VLLM_CPU_OMP_THREADS_BIND=auto \
  --env VLLM_CPU_NUM_OF_RESERVED_CPU=1 \
  --env CPU_VISIBLE_MEMORY_NODES=0,1,2,3 \
  --env VLLM_CPU_KVCACHE_SPACE=16 \
  /verifier-agent/vllm_cpu.sif \
  --model /Qwen3-4B \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --block-size 128 \
  --async-scheduling \
  --port 8001