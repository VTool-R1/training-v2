# VTool

Port of the older VTool-style multimodal agent loop onto current `verl` extension points.

## What this recipe adds

- A custom `vtool_agent` loop that:
  - treats the first assistant turn as a Python refocus action,
  - executes one image-editing round against ChartQA/TableVQA-style bounding box metadata,
  - feeds the edited image back as a second user observation,
  - optimizes only the final assistant answer.
- A custom reward manager that scores only the trainable final assistant segment instead of the full mixed trajectory.
- A function reward compatible with an OpenAI-style judge endpoint.

## Expected dataset contract

This recipe assumes parquet rows already look like the older VTool data:

- `agent_name` is `vtool_agent`
- `prompt` contains multimodal chat messages with `<image>` placeholders
- `images` contains the source image(s)
- `extra_info.tools_kwargs.metadata` contains JSON with refocus bounding boxes
- `reward_model.ground_truth` contains the target answer

The old `refocus_chart/*.parquet` files should work without dataset code changes.

## Judge endpoint

The reward function talks to an OpenAI-compatible endpoint. Configure it with:

```bash
export OPENAI_API_KEY=EMPTY
export VTOOL_JUDGE_HOST=localhost
export VTOOL_JUDGE_PORT=8000
export VTOOL_JUDGE_USE_ENDPOINT_DEFAULT=1
```

You can also set `VTOOL_JUDGE_API_BASE=http://host:port/v1` directly. Resolution order is:
`api_base` argument, then `VTOOL_JUDGE_API_BASE`, then `OPENAI_API_BASE` / `OPENAI_BASE_URL`, then `VTOOL_JUDGE_SCHEME` + `VTOOL_JUDGE_HOST` + `VTOOL_JUDGE_PORT`.

If `VTOOL_JUDGE_USE_ENDPOINT_DEFAULT=1`, the recipe ignores `VTOOL_JUDGE_MODEL` / `OPENAI_MODEL`, queries `/v1/models`, and uses the server's default or first model.

## Launch

```bash
bash recipe/vtool/run_qwen2_5_vl_3b_chart.sh
```

Override `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, or pass extra Hydra overrides on the command line.
