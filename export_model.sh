python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/vtool/3b_chart_fp16/global_step_140/actor \
    --target_dir checkpoints/VTool-3B-200