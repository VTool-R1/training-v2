# new_eval

`new_eval` is a standalone evaluation pipeline for
`verl.experimental.agent_loop.vtool_agent_loop.VToolAgentLoop`.

It does not depend on the older eval folders. It runs the VTool eval loop
against an externally hosted OpenAI-compatible vLLM server, writes JSONL
outputs, and provides a separate scorer for OpenAI-compatible judge endpoints.

## What It Evaluates

Use this folder when you want evaluation to follow the same loop structure as
`VToolAgentLoop`:

- initial assistant pass
- parse refocus tool code from the assistant output
- execute at most one tool round
- append observation as a follow-up user turn
- final assistant pass

## Dataset Requirements

For parity with the VTool training data, evaluate the processed parquet written
by [refocus_tool_use_processing.py](/verifier-agent/refocus_tool_use_processing.py),
not the raw ReFocus dataset.

That processed parquet contains:

- `prompt`
- `images`
- `extra_info.answer`
- `extra_info.question`
- `extra_info.tools_kwargs.metadata`

This is the same dataset source used by
[train_vtool_3b_chart_full.sh](/verifier-agent/train_vtool_3b_chart_full.sh#L18):

```bash
/verifier-agent/refocus_chart/test.parquet
```

## Environment

Generation uses an external OpenAI-compatible server, typically started with
[vllm_infer.sh](/vtool/eval/vllm_infer.sh). The eval
runtime needs:

- `datasets`
- `transformers`
- `Pillow`
- the local model files or HF access for `--model` so the tokenizer/processor
  can render prompts and compute token counts

Scoring is separate and uses the OpenAI client against either OpenAI or an
OpenAI-compatible local server.

## Basic Evaluation

Start the serving endpoint first:

```bash
bash /vtool/eval/vllm_infer.sh
```

Then run evaluation:

```bash
bash /vtool/eval/run_eval.sh \
  --model /path/to/model-or-checkpoint \
  --server-base-url http://127.0.0.1:8000/v1 \
  --data-file /verifier-agent/refocus_chart/test.parquet
```

The script prints the output JSONL path at the end.

## Match The Training Setup

If you want evaluation settings close to
[train_vtool_3b_chart_full.sh](/verifier-agent/train_vtool_3b_chart_full.sh),
use the same processed parquet and explicitly match the loop and length limits:

```bash
bash /vtool/eval/run_eval.sh \
  --model /path/to/model-or-checkpoint \
  --server-base-url http://127.0.0.1:8000/v1 \
  --data-file /verifier-agent/refocus_chart/test.parquet \
  --prompt-key prompt \
  --image-key images \
  --prompt-length 8192 \
  --response-length 8192 \
  --max-model-len 16384
```

Notes:

- `train_vtool_3b_chart_full.sh` sets `data.max_prompt_length=8192` and
  `data.max_response_length=8192`.
- The training config at
  [refocus_multiturn_grpo_w_verification.yaml](/verifier-agent/examples/self_verify/config/refocus_multiturn_grpo_w_verification.yaml#L36)
  uses `max_user_turns=3` and `max_assistant_turns=3`.
- `--max-model-len` must be large enough to hold prompt plus generation. If it
  is too small, evaluation will fail with negative `max_tokens`.

## Chat Template

`VToolAgentLoop` builds prompts with `apply_chat_template(..., tools=...)`, so
the model chat template matters.

If the model already carries the correct template, you do not need to pass one.
If it was trained with a custom template, pass it explicitly:

```bash
bash /vtool/eval/run_eval.sh \
  --model /path/to/model-or-checkpoint \
  --server-base-url http://127.0.0.1:8000/v1 \
  --data-file /verifier-agent/refocus_chart/test.parquet \
  --chat-template-path /path/to/template.jinja
```

## Useful Options

- `--output /path/to/results.jsonl`: write to a fixed output path
- `--resume`: skip indices already present in the output file
- `--max-concurrency N`: control concurrent sample evaluation
- `--preview-images`: save a small sample of tool-edited images under `/tmp`
- `--preview-image-sample-rate`: preview sampling rate
- `--server-base-url http://host:port/v1`: point eval at an external vLLM host
- `--server-model model-id`: override the served model id if `/v1/models`
  exposes a different name than `--model`
- `--request-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`:
  pass extra request fields through to the endpoint

## Output Format

Each output row includes:

- `index`
- `query`
- `ground_truth`
- `prompt_messages`
- `prompt_text`
- `response_text`
- `generated_text`
- `model_response`
- `response_mask`
- `num_turns`
- `metrics.tool_use_attempted`
- `metrics.tool_use_success`

`generated_text` is the decoded assistant-generated portion where
`response_mask == 1`. `masked_context_text` is the non-optimized part of the
trajectory, including the intermediate tool-turn context.

## Scoring

Score a result file with the bundled judge script:

```bash
bash /verifier-agent/eval/new_eval/score_results.sh \
  /path/to/results.jsonl
```

By default the scorer uses:

- `SCORER_OPENAI_BASE_URL=http://localhost:8000/v1`
- `SCORER_OPENAI_API_KEY=EMPTY`
- `SCORER_MAX_TOKENS=2048`

Override them as needed:

```bash
SCORER_OPENAI_BASE_URL=http://localhost:8001/v1 \
SCORER_OPENAI_API_KEY=EMPTY \
SCORER_MODEL=your-judge-model \
SCORER_MAX_TOKENS=2048 \
bash /verifier-agent/eval/new_eval/score_results.sh \
  /path/to/results.jsonl
```

The scored output is written next to the input as `*_scored.jsonl`.

## Common Failure Modes

`Could not infer served model name from endpoint`

- Pass `--server-model` explicitly.
- Check `curl http://host:port/v1/models` and make sure the served model is up.

`Empty scorer response`

- The judge endpoint is reachable but not returning usable final content.
- Try a different judge model, increase `SCORER_MAX_TOKENS`, or fix the local
  OpenAI-compatible server response shape.

Tool execution failure with no edited image

- Check that the input parquet includes `extra_info.tools_kwargs.metadata`.
- Check that the model is actually emitting a fenced Python block that the
  refocus parser can execute.
