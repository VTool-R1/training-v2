from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class PreparedExample:
    index: int
    uid: Any
    query: str
    ground_truth: str
    raw_prompt: list[dict[str, Any]]
    multi_modal_data: dict[str, Any]
    tools_kwargs: dict[str, Any]
    original_record: dict[str, Any]


def load_source_dataset(
    *,
    data_files: list[str] | None,
    dataset_name: str | None,
    dataset_subset: str | None,
    dataset_split: str,
):
    from datasets import load_dataset

    if data_files:
        return load_dataset("parquet", data_files=data_files)["train"]

    if not dataset_name:
        raise ValueError("Either data_files or dataset_name must be provided.")

    kwargs = {"split": dataset_split}
    if dataset_subset:
        kwargs["name"] = dataset_subset
    return load_dataset(dataset_name, **kwargs)


def prepare_dataset_examples(
    dataset,
    *,
    prompt_key: str = "prompt",
    image_key: str = "images",
    offset: int = 0,
    limit: int | None = None,
) -> list[PreparedExample]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    total = len(dataset)
    start = min(offset, total)
    stop = total if limit is None else min(total, start + limit)

    examples: list[PreparedExample] = []
    for dataset_index in range(start, stop):
        record = dataset[dataset_index]
        examples.append(_prepare_example(record, record_index=dataset_index, prompt_key=prompt_key, image_key=image_key))
    return examples


def _prepare_example(record: dict[str, Any], *, record_index: int, prompt_key: str, image_key: str) -> PreparedExample:
    row = copy.deepcopy(record)
    extra_info = row.get("extra_info") or {}

    images = _load_images(row.get(image_key))
    raw_prompt = _extract_messages(row, prompt_key=prompt_key, image_count=len(images))
    query = _extract_query(row, raw_prompt)
    ground_truth = _extract_ground_truth(row)
    tools_kwargs = _extract_tools_kwargs(row, extra_info)
    uid = row.get("uid", extra_info.get("uid", record_index))

    multi_modal_data = {"image": images} if images else {}

    return PreparedExample(
        index=record_index,
        uid=uid,
        query=query,
        ground_truth=ground_truth,
        raw_prompt=raw_prompt,
        multi_modal_data=multi_modal_data,
        tools_kwargs=tools_kwargs,
        original_record=row,
    )


def _load_images(value: Any) -> list[Image.Image]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [_load_single_image(item) for item in value]


def _load_single_image(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")

    if isinstance(item, dict):
        if "bytes" in item:
            return Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        if "path" in item:
            return Image.open(item["path"]).convert("RGB")
        if "array" in item:
            return Image.fromarray(item["array"]).convert("RGB")

    if isinstance(item, str):
        return Image.open(item).convert("RGB")

    if hasattr(item, "__array__"):
        import numpy as np

        return Image.fromarray(np.asarray(item)).convert("RGB")

    raise TypeError(f"Unsupported image payload type: {type(item).__name__}")


def _extract_messages(row: dict[str, Any], *, prompt_key: str, image_count: int) -> list[dict[str, Any]]:
    messages = row.get(prompt_key) or row.get("messages")
    if messages is None:
        question = row.get("query") or row.get("question") or row.get("prompt")
        if question is None:
            raise KeyError(f"Could not find prompt/messages field in record with keys: {sorted(row.keys())}")
        messages = [{"role": "user", "content": question}]

    normalized = [copy.deepcopy(message) for message in messages]
    has_image_marker = False

    for message in normalized:
        content = message.get("content", "")
        if isinstance(content, str):
            if "<image>" in content or "<video>" in content:
                converted = _convert_string_content(content)
                message["content"] = converted
                has_image_marker = has_image_marker or any(
                    isinstance(item, dict) and item.get("type") == "image" for item in converted
                )
        elif isinstance(content, list):
            has_image_marker = has_image_marker or any(
                isinstance(item, dict) and item.get("type") == "image" for item in content
            )
        else:
            message["content"] = [{"type": "text", "text": str(content)}]

    if image_count and not has_image_marker:
        target_message = next((m for m in normalized if m.get("role") == "user"), normalized[0])
        content = target_message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        target_message["content"] = [{"type": "image"} for _ in range(image_count)] + list(content)

    return normalized


def _convert_string_content(content: str) -> list[dict[str, Any]]:
    if "<image>" not in content and "<video>" not in content:
        return [{"type": "text", "text": content}]

    parts = []
    tokens = content.replace("<video>", "<image>").split("<image>")
    for index, segment in enumerate(tokens):
        if segment:
            parts.append({"type": "text", "text": segment})
        if index < len(tokens) - 1:
            parts.append({"type": "image"})
    return parts


def _extract_query(row: dict[str, Any], raw_prompt: list[dict[str, Any]]) -> str:
    extra_info = row.get("extra_info") or {}
    query = row.get("query") or row.get("question") or extra_info.get("question")
    if query:
        return str(query)

    for message in reversed(raw_prompt):
        if message.get("role") != "user":
            continue
        text = _extract_text_content(message.get("content"))
        if text:
            return text

    return ""


def _extract_ground_truth(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info") or {}
    candidates = [
        row.get("ground_truth"),
        row.get("answer"),
        extra_info.get("answer"),
        extra_info.get("ground_truth"),
    ]
    for candidate in candidates:
        if candidate is not None:
            return str(candidate)
    return ""


def _extract_tools_kwargs(row: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, Any]:
    tools_kwargs = row.get("tools_kwargs") or extra_info.get("tools_kwargs") or {}
    if not isinstance(tools_kwargs, dict):
        return {}

    normalized = copy.deepcopy(tools_kwargs)
    metadata = normalized.get("metadata", extra_info.get("metadata"))
    if isinstance(metadata, dict):
        normalized["metadata"] = json.dumps(metadata, ensure_ascii=False)
    elif metadata is not None:
        normalized["metadata"] = metadata

    return normalized


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return str(content or "")
