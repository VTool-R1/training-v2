from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from eval.config import DEFAULT_TOOL_CONFIG_PATH, load_chat_template, parse_json_dict
    from eval.dataset import PreparedExample, load_source_dataset, prepare_dataset_examples
    from eval.result_io import append_jsonl, load_completed_indices
    from eval.server import OpenAIChatServerManager
    from eval.vtool_eval_loop import EvalLoopOutput, run_vtool_eval_loop
else:
    from .config import DEFAULT_TOOL_CONFIG_PATH, load_chat_template, parse_json_dict
    from .dataset import PreparedExample, load_source_dataset, prepare_dataset_examples
    from .result_io import append_jsonl, load_completed_indices
    from .server import OpenAIChatServerManager
    from .vtool_eval_loop import EvalLoopOutput, run_vtool_eval_loop


DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:8000/v1"


@dataclass
class EvalStats:
    completed: int = 0
    tool_attempted: int = 0
    tool_success: int = 0
    tool_failure_reasons: Counter[str] = field(default_factory=Counter)
    tool_recovery_modes: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone VTool evaluation runner against an OpenAI endpoint.")

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-file", nargs="+", help="Parquet file(s) to evaluate.")
    data_group.add_argument("--dataset-name", help="Hugging Face dataset name.")

    parser.add_argument("--dataset-subset", default=None, help="Dataset subset/config name.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split for --dataset-name.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--model", required=True, help="Local model path or HF id used for tokenizer/processor.")
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--server-api-key", default="EMPTY")
    parser.add_argument("--server-model", default=None, help="Served model id exposed by the OpenAI endpoint.")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-extra-body", default="{}")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template-path", default=None)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--apply-chat-template-kwargs", default="{}")

    parser.add_argument("--tool-config-path", default=str(DEFAULT_TOOL_CONFIG_PATH))
    parser.add_argument("--send-tools", action="store_true", help="Forward tool schemas to the chat completion API.")

    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview-images", action="store_true")
    parser.add_argument("--preview-image-sample-rate", type=float, default=0.01)
    parser.add_argument("--preview-image-output-root", default="/tmp/vtool_agent_loop_preview")

    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))

    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()

    model_short = Path(args.model.rstrip("/")).name
    if args.data_file:
        dataset_short = Path(args.data_file[0]).stem
    else:
        dataset_short = args.dataset_name.replace("/", "_")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{model_short}_{dataset_short}_{timestamp}.jsonl"
    return Path(args.output_dir).expanduser().resolve() / output_name


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "stop": args.stop or None,
        "max_tokens": args.response_length,
        "seed": args.seed,
    }
    return {key: value for key, value in params.items() if value is not None}


def _normalize_token_ids(token_ids: Any) -> list[int]:
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _apply_chat_template_string(
    *,
    tokenizer: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Any],
    tools: list[dict[str, Any]] | None,
    apply_chat_template_kwargs: dict[str, Any],
) -> str:
    target = processor if processor is not None and hasattr(processor, "apply_chat_template") else tokenizer
    if target is None:
        return json.dumps(messages, ensure_ascii=False)

    kwargs = dict(apply_chat_template_kwargs)
    try:
        return target.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
    except TypeError:
        return target.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )


def render_prompt(
    *,
    tokenizer: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Any],
    tools: list[dict[str, Any]] | None,
    apply_chat_template_kwargs: dict[str, Any],
) -> tuple[str, list[int]]:
    prompt_text = _apply_chat_template_string(
        tokenizer=tokenizer,
        processor=processor,
        messages=messages,
        images=images,
        tools=tools,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    if processor is not None:
        try:
            model_inputs = processor(text=[prompt_text], images=images or None, return_tensors="pt")
            prompt_ids = _normalize_token_ids(model_inputs.get("input_ids"))
            return prompt_text, prompt_ids
        except Exception:
            pass

    if tokenizer is not None:
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
        except TypeError:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
        return prompt_text, _normalize_token_ids(prompt_ids)

    return prompt_text, []


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    if tokenizer is None:
        return []
    return _normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


def extract_tool_code(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        return code or None
    return None


def extract_images(multi_modal_data: dict[str, Any] | None) -> list[Any]:
    multi_modal_data = multi_modal_data or {}
    images = multi_modal_data.get("images")
    if images is None:
        images = multi_modal_data.get("image", [])
    return list(images or [])


def maybe_save_preview_artifacts(
    *,
    example: PreparedExample,
    output: EvalLoopOutput,
    preview_images: bool,
    preview_image_sample_rate: float,
    preview_image_output_root: str,
) -> str | None:
    if not preview_images:
        return None
    if random.random() >= min(max(preview_image_sample_rate, 0.0), 1.0):
        return None
    if not bool(output.metrics.get("tool_use_success")):
        return None

    input_images = extract_images(example.multi_modal_data)
    output_images = extract_images(output.multi_modal_data)
    if len(output_images) <= len(input_images):
        return None

    edited_image = output_images[-1]
    tool_code = extract_tool_code(output.masked_context_text)

    preview_root = Path(preview_image_output_root).expanduser()
    if not preview_root.is_absolute():
        preview_root = Path("/tmp") / preview_root
    preview_dir = preview_root / f"{example.index}_{example.uid}_tool_exec"
    preview_dir.mkdir(parents=True, exist_ok=True)

    image_path = preview_dir / "modified_image.png"
    edited_image.save(image_path)

    code_path = None
    if tool_code:
        code_path = preview_dir / "tool_code.py"
        code_path.write_text(tool_code, encoding="utf-8")

    masked_context_path = preview_dir / "masked_context_text.txt"
    masked_context_path.write_text(output.masked_context_text, encoding="utf-8")

    preview_info = {
        "index": example.index,
        "uid": str(example.uid),
        "query": example.query,
        "ground_truth": example.ground_truth,
        "saved_image_path": str(image_path),
        "saved_code_path": str(code_path) if code_path is not None else None,
        "saved_masked_context_path": str(masked_context_path),
        "preview_sample_rate": preview_image_sample_rate,
    }
    (preview_dir / "preview_info.json").write_text(json.dumps(preview_info, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(preview_dir)


def save_failure_artifacts(
    *,
    example: PreparedExample,
    result: dict[str, Any],
    output_path: Path,
) -> str | None:
    failure_reason = result.get("tool_failure_reason")
    if not failure_reason:
        return None

    failure_root = output_path.parent / f"{output_path.stem}_faults"
    sample_dir = failure_root / str(failure_reason) / f"{example.index}_{example.uid}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    initial_response_text = str(result.get("initial_response_text") or "")
    masked_context_text = str(result.get("masked_context_text") or "")
    generated_text = str(result.get("generated_text") or "")
    tool_code = str(result.get("executed_tool_code") or "") or extract_tool_code(initial_response_text) or extract_tool_code(masked_context_text)

    info = {
        "index": example.index,
        "uid": str(example.uid),
        "query": example.query,
        "ground_truth": example.ground_truth,
        "failure_reason": failure_reason,
        "failure_detail": result.get("tool_failure_detail"),
        "tool_parse_status": result.get("tool_parse_status"),
        "tool_parse_message": result.get("tool_parse_message"),
        "tool_parse_error_code": result.get("tool_parse_error_code"),
        "tool_recovery_mode": result.get("tool_recovery_mode"),
        "tools_kwargs": example.tools_kwargs,
        "saved_initial_response_path": str(sample_dir / "initial_response.txt"),
        "saved_masked_context_path": str(sample_dir / "masked_context_text.txt"),
        "saved_generated_text_path": str(sample_dir / "generated_text.txt"),
        "saved_tool_code_path": str(sample_dir / "tool_code.py") if tool_code else None,
    }
    (sample_dir / "fault_info.json").write_text(json.dumps(info, ensure_ascii=True, indent=2), encoding="utf-8")

    (sample_dir / "initial_response.txt").write_text(initial_response_text, encoding="utf-8")
    (sample_dir / "masked_context_text.txt").write_text(masked_context_text, encoding="utf-8")
    (sample_dir / "generated_text.txt").write_text(generated_text, encoding="utf-8")
    if tool_code:
        (sample_dir / "tool_code.py").write_text(tool_code, encoding="utf-8")

    input_images = extract_images(example.multi_modal_data)
    if input_images:
        first_image = input_images[0]
        if isinstance(first_image, Image.Image):
            input_image_path = sample_dir / "input_image.png"
            first_image.save(input_image_path)
            info["saved_input_image_path"] = str(input_image_path)
            (sample_dir / "fault_info.json").write_text(json.dumps(info, ensure_ascii=True, indent=2), encoding="utf-8")

    return str(sample_dir)


def build_result_record(
    example: PreparedExample,
    output: EvalLoopOutput,
    *,
    tokenizer: Any,
    processor: Any,
    model_name: str,
    tool_schemas: list[dict[str, Any]] | None,
    apply_chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    input_images = extract_images(example.multi_modal_data)
    prompt_text, prompt_ids = render_prompt(
        tokenizer=tokenizer,
        processor=processor,
        messages=example.raw_prompt,
        images=input_images,
        tools=tool_schemas,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    masked_context_text = output.masked_context_text
    generated_text = output.generated_text
    masked_ids = tokenize_text(tokenizer, masked_context_text)
    generated_ids = tokenize_text(tokenizer, generated_text)
    response_mask = [0] * len(masked_ids) + [1] * len(generated_ids)

    output_images = extract_images(output.multi_modal_data)

    return {
        "index": example.index,
        "uid": example.uid,
        "model": model_name,
        "query": example.query,
        "ground_truth": example.ground_truth,
        "prompt_messages": example.raw_prompt,
        "prompt_text": prompt_text,
        "response_text": output.response_text,
        "generated_text": generated_text,
        "model_response": generated_text,
        "masked_context_text": masked_context_text,
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_mask),
        "generated_token_count": len(generated_ids),
        "response_mask": response_mask,
        "num_turns": output.num_turns,
        "num_images": len(output_images),
        "metrics": dict(output.metrics),
        "tools_kwargs": example.tools_kwargs,
        "initial_response_text": output.extra_fields.get("vtool_initial_response_text"),
        "executed_tool_code": output.extra_fields.get("vtool_executed_tool_code"),
        "tool_failure_reason": output.extra_fields.get("vtool_tool_failure_reason"),
        "tool_failure_detail": output.extra_fields.get("vtool_tool_failure_detail"),
        "tool_recovery_mode": output.extra_fields.get("vtool_tool_recovery_mode"),
        "tool_parse_status": output.extra_fields.get("vtool_tool_parse_status"),
        "tool_parse_message": output.extra_fields.get("vtool_tool_parse_message"),
        "tool_parse_error_code": output.extra_fields.get("vtool_tool_parse_error_code"),
    }


def _format_top_failure_reasons(counter: Counter[str], limit: int = 3) -> str:
    items = [(reason, count) for reason, count in counter.most_common(limit) if reason]
    return ", ".join(f"{reason}={count}" for reason, count in items)


def write_summary_json(output_path: Path, stats: EvalStats) -> Path:
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "completed": stats.completed,
        "tool_attempted": stats.tool_attempted,
        "tool_success": stats.tool_success,
        "tool_failed": stats.tool_attempted - stats.tool_success,
        "tool_failure_reasons": dict(stats.tool_failure_reasons),
        "tool_recovery_modes": dict(stats.tool_recovery_modes),
        "failure_artifacts_root": str(output_path.parent / f"{output_path.stem}_faults"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary_path


async def evaluate_example(
    example: PreparedExample,
    *,
    server_manager: OpenAIChatServerManager,
    tokenizer: Any,
    processor: Any,
    sampling_params: dict[str, Any],
    model_name: str,
    preview_images: bool,
    preview_image_sample_rate: float,
    preview_image_output_root: str,
    tool_schemas: list[dict[str, Any]] | None,
    send_tools: bool,
    request_extra_body: dict[str, Any],
    apply_chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    output = await run_vtool_eval_loop(
        raw_prompt=example.raw_prompt,
        multi_modal_data=example.multi_modal_data,
        tools_kwargs=example.tools_kwargs,
        sampling_params=sampling_params,
        server_manager=server_manager,
        tool_schemas=tool_schemas,
        send_tools=send_tools,
        request_extra_body=request_extra_body,
    )
    result = build_result_record(
        example,
        output,
        tokenizer=tokenizer,
        processor=processor,
        model_name=model_name,
        tool_schemas=tool_schemas if send_tools else None,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
    preview_dir = maybe_save_preview_artifacts(
        example=example,
        output=output,
        preview_images=preview_images,
        preview_image_sample_rate=preview_image_sample_rate,
        preview_image_output_root=preview_image_output_root,
    )
    if preview_dir is not None:
        result["preview_dir"] = preview_dir
    return result


async def async_main(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    chat_template = load_chat_template(chat_template=args.chat_template, chat_template_path=args.chat_template_path)
    apply_chat_template_kwargs = parse_json_dict(args.apply_chat_template_kwargs)
    request_extra_body = parse_json_dict(args.request_extra_body)

    output_path = resolve_output_path(args)
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from verl.utils.fs import copy_to_local
    from verl.utils.tokenizer import hf_processor, hf_tokenizer

    local_model_path = copy_to_local(args.model)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=args.trust_remote_code)
    processor = hf_processor(local_model_path, trust_remote_code=args.trust_remote_code, use_fast=True)

    if chat_template is not None:
        if processor is not None:
            processor.chat_template = chat_template
        tokenizer.chat_template = chat_template

    tool_schemas: list[dict[str, Any]] | None = None
    if args.send_tools:
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        tool_list = initialize_tools_from_config(args.tool_config_path) if args.tool_config_path else []
        tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    server_manager = OpenAIChatServerManager.create(
        base_url=args.server_base_url,
        api_key=args.server_api_key,
        model=args.server_model,
        request_timeout=args.request_timeout,
        default_extra_body=request_extra_body,
    )
    served_model_name = await server_manager.resolve_model(args.server_model or args.model)

    dataset = load_source_dataset(
        data_files=args.data_file,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
    )
    examples = prepare_dataset_examples(
        dataset,
        prompt_key=args.prompt_key,
        image_key=args.image_key,
        offset=args.offset,
        limit=args.limit,
    )

    completed_indices = load_completed_indices(output_path) if args.resume else set()
    sampling_params = build_sampling_params(args)
    stats = EvalStats()
    writer_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def run_one(example: PreparedExample) -> None:
        nonlocal stats
        if example.index in completed_indices:
            return

        async with semaphore:
            result = await evaluate_example(
                example,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=processor,
                sampling_params=sampling_params,
                model_name=served_model_name,
                preview_images=args.preview_images,
                preview_image_sample_rate=args.preview_image_sample_rate,
                preview_image_output_root=args.preview_image_output_root,
                tool_schemas=tool_schemas,
                send_tools=args.send_tools,
                request_extra_body=request_extra_body,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )

        async with writer_lock:
            failure_dir = save_failure_artifacts(example=example, result=result, output_path=output_path)
            if failure_dir is not None:
                result["failure_dir"] = failure_dir
            append_jsonl(output_path, result)
            stats.completed += 1
            attempted = int(bool(result["metrics"].get("tool_use_attempted")))
            success = int(bool(result["metrics"].get("tool_use_success")))
            stats.tool_attempted += attempted
            stats.tool_success += success
            recovery_mode = result.get("tool_recovery_mode")
            if recovery_mode:
                stats.tool_recovery_modes[str(recovery_mode)] += 1
            if attempted and not success:
                failure_reason = str(result.get("tool_failure_reason") or "unknown_tool_failure")
                stats.tool_failure_reasons[failure_reason] += 1
            if stats.completed % 10 == 0 or stats.completed == 1:
                top_failures = _format_top_failure_reasons(stats.tool_failure_reasons)
                failure_suffix = f" top_failures=[{top_failures}]" if top_failures else ""
                print(
                    f"[progress] completed={stats.completed} "
                    f"tool_attempted={stats.tool_attempted} tool_success={stats.tool_success}"
                    f"{failure_suffix}"
                )

    try:
        tasks = [asyncio.create_task(run_one(example)) for example in examples]
        for task in asyncio.as_completed(tasks):
            await task
    finally:
        await server_manager.shutdown()

    summary_path = write_summary_json(output_path, stats)
    top_failures = _format_top_failure_reasons(stats.tool_failure_reasons, limit=10)
    top_recoveries = _format_top_failure_reasons(stats.tool_recovery_modes, limit=10)
    if top_failures:
        print(f"[summary] tool_failures={top_failures}")
    if top_recoveries:
        print(f"[summary] tool_recoveries={top_recoveries}")
    print(output_path)
    print(summary_path)
    return output_path


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
