export HF_HOME='/.cache/huggingface'
export HUGGINGFACE_HUB_CACHE='/.cache/huggingface'
export TIKTOKEN_ENCODINGS_BASE='/verifier-agent/harmony'

apptainer run --nv \
    --bind : \
    --bind /.cache/huggingface:/.cache/huggingface \
    /vllm-openai_v0.17.1.sif \
    --model openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --reasoning-parser openai_gptoss \
    --async-scheduling