apptainer run --nv \
    --bind /.cache/huggingface:/.cache/huggingface \
    /vllm-openai_v0.17.1.sif \
    --model /Qwen3-4B \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.05 \
    --cpu-offload-gb 80 \
    --async-scheduling \
    --port 8002