from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from recipe.vtool.refocus_tools import RefocusCodeParser, build_refocus_context


logger = logging.getLogger(__name__)

SUCCESS_OBSERVATION = (
    "OBSERVATION: Execution success. The output is as follows:\n"
    "<the image outputs of the code is added as the second image>"
)
FAILURE_OBSERVATION = (
    "OBSERVATION: Execution failed. "
    "The code did not produce a valid edited image. "
    "Please regenerate your final answer."
)


@dataclass
class EvalLoopOutput:
    response_text: str
    generated_text: str
    masked_context_text: str
    num_turns: int
    metrics: dict[str, Any]
    multi_modal_data: dict[str, Any]
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    image: Image.Image | None
    success: bool
    failure_reason: str | None = None
    failure_detail: str | None = None
    recovery_mode: str | None = None
    executed_code: str | None = None


def _parse_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str) and metadata:
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return {}


def _encode_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_endpoint_messages(messages: list[dict[str, Any]], images: list[Image.Image]) -> list[dict[str, Any]]:
    endpoint_messages: list[dict[str, Any]] = []
    image_index = 0

    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            endpoint_messages.append({"role": message["role"], "content": content})
            continue

        if not isinstance(content, list):
            endpoint_messages.append({"role": message["role"], "content": str(content)})
            continue

        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                parts.append({"type": "text", "text": str(item)})
                continue

            item_type = item.get("type")
            if item_type == "text" or "text" in item:
                parts.append({"type": "text", "text": str(item.get("text", ""))})
                continue
            if item_type == "image_url" and "image_url" in item:
                parts.append(item)
                continue
            if item_type == "image" or "image" in item:
                if image_index >= len(images):
                    raise ValueError("Prompt contains more image placeholders than image payloads.")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image_to_data_url(images[image_index])},
                    }
                )
                image_index += 1
                continue

            parts.append({"type": "text", "text": str(item)})

        endpoint_messages.append({"role": message["role"], "content": parts})

    return endpoint_messages


def _categorize_parse_failure(parse_result: Any) -> tuple[str, str]:
    message = str(getattr(parse_result, "message", "") or "")
    lowered = message.lower()

    if "syntaxerror" in lowered or "invalid syntax" in lowered:
        return "parse_failure_invalid_syntax", message
    if "enclosed in ```python```" in lowered:
        return "parse_failure_unclosed_code_block", message
    if "empty" in lowered:
        return "parse_failure_empty_code", message

    error_code = str(getattr(parse_result, "error_code", "") or "unknown").lower()
    return f"parse_failure_{error_code}", message


async def _execute_tool_code(
    *,
    parser: RefocusCodeParser,
    code: str,
    images: list[Image.Image],
    tools_kwargs: dict[str, Any],
    tool_timeout_seconds: float,
) -> ToolExecutionResult:
    if not images:
        return ToolExecutionResult(
            image=None,
            success=False,
            failure_reason="missing_input_image",
            failure_detail="Tool execution requires at least one input image.",
        )

    metadata = _parse_metadata((tools_kwargs or {}).get("metadata"))
    tool_output: Any = None

    def _display(image: Any) -> None:
        nonlocal tool_output
        tool_output = image

    context = build_refocus_context(
        display_callback=_display,
        image=images[0],
        metadata=metadata,
    )
    initial_context_keys = set(context)
    executable_code = parser.ensure_display_call(code)

    def _exec_tool() -> None:
        exec(executable_code, context)

    try:
        await asyncio.wait_for(asyncio.to_thread(_exec_tool), timeout=tool_timeout_seconds)
    except asyncio.TimeoutError:
        detail = f"Tool execution timed out after {tool_timeout_seconds:.1f}s."
        logger.warning("VTool tool execution failed: %s", detail)
        return ToolExecutionResult(
            image=None,
            success=False,
            failure_reason="tool_execution_timeout",
            failure_detail=detail,
            executed_code=executable_code,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VTool tool execution failed: %s", exc)
        return ToolExecutionResult(
            image=None,
            success=False,
            failure_reason=f"tool_execution_{type(exc).__name__}",
            failure_detail=str(exc) or repr(exc),
            executed_code=executable_code,
        )

    if isinstance(tool_output, Image.Image):
        if tool_output.size[0] > 0 and tool_output.size[1] > 0:
            recovery_mode = "auto_append_display" if executable_code != code else None
            return ToolExecutionResult(
                image=tool_output,
                success=True,
                recovery_mode=recovery_mode,
                executed_code=executable_code,
            )
        return ToolExecutionResult(
            image=None,
            success=False,
            failure_reason="invalid_output_empty_image",
            failure_detail=f"Tool returned an empty PIL image with size={tool_output.size!r}.",
            executed_code=executable_code,
        )

    for key, value in reversed(list(context.items())):
        if key in initial_context_keys:
            continue
        if isinstance(value, Image.Image) and value.size[0] > 0 and value.size[1] > 0:
            logger.info("Recovered VTool output from variable '%s' without an explicit display(...) call.", key)
            return ToolExecutionResult(
                image=value,
                success=True,
                recovery_mode="implicit_last_image_variable" if executable_code == code else "auto_append_display",
                executed_code=executable_code,
            )

    if tool_output is None:
        return ToolExecutionResult(
            image=None,
            success=False,
            failure_reason="invalid_output_missing_display",
            failure_detail="Tool execution completed without calling display(...).",
            executed_code=executable_code,
        )

    return ToolExecutionResult(
        image=None,
        success=False,
        failure_reason="invalid_output_non_image",
        failure_detail=f"Tool passed a non-image object to display(...): {type(tool_output).__name__}.",
        executed_code=executable_code,
    )


async def run_vtool_eval_loop(
    *,
    raw_prompt: list[dict[str, Any]],
    multi_modal_data: dict[str, Any] | None,
    tools_kwargs: dict[str, Any] | None,
    sampling_params: dict[str, Any],
    server_manager: Any,
    tool_schemas: list[dict[str, Any]] | None = None,
    send_tools: bool = False,
    request_extra_body: dict[str, Any] | None = None,
    tool_timeout_seconds: float = 10.0,
) -> EvalLoopOutput:
    parser = RefocusCodeParser()
    messages = copy.deepcopy(list(raw_prompt))
    images = list((multi_modal_data or {}).get("image") or (multi_modal_data or {}).get("images") or [])

    metrics = {
        "tool_use_attempted": 0.0,
        "tool_use_success": 0.0,
    }

    initial_output = await server_manager.generate(
        messages=_to_endpoint_messages(messages, images),
        sampling_params=sampling_params,
        tools=tool_schemas if send_tools else None,
        extra_body=request_extra_body,
    )
    initial_text = initial_output.text or ""
    parse_result = parser.parse(initial_text)

    tool_attempted = False
    tool_success = False
    masked_context_text = ""
    final_text = initial_text
    tool_failure_reason: str | None = None
    tool_failure_detail: str | None = None
    tool_recovery_mode: str | None = None
    executed_tool_code: str | None = None

    assistant_turns = 1
    user_turns = 0

    if parse_result.error_code != "NOTOOL":
        tool_attempted = True
        metrics["tool_use_attempted"] = 1.0

        messages.append({"role": "assistant", "content": initial_text})
        if parse_result.status and parse_result.code.strip():
            tool_result = await _execute_tool_code(
                parser=parser,
                code=parse_result.code,
                images=images,
                tools_kwargs=tools_kwargs or {},
                tool_timeout_seconds=tool_timeout_seconds,
            )
        else:
            logger.debug("Skipping VTool execution due to parse failure: %s", parse_result.message)
            failure_reason, failure_detail = _categorize_parse_failure(parse_result)
            tool_result = ToolExecutionResult(
                image=None,
                success=False,
                failure_reason=failure_reason,
                failure_detail=failure_detail,
            )
        edited_image = tool_result.image
        tool_success = tool_result.success
        tool_failure_reason = tool_result.failure_reason
        tool_failure_detail = tool_result.failure_detail
        tool_recovery_mode = tool_result.recovery_mode
        executed_tool_code = tool_result.executed_code
        metrics["tool_use_success"] = float(tool_success)

        if tool_success and edited_image is not None:
            images.append(edited_image)
            observation_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SUCCESS_OBSERVATION},
                ],
            }
            observation_text = SUCCESS_OBSERVATION
        else:
            observation_message = {"role": "user", "content": FAILURE_OBSERVATION}
            observation_text = FAILURE_OBSERVATION

        messages.append(observation_message)
        masked_context_text = f"{initial_text}\n\n{observation_text}\n\n"

        final_output = await server_manager.generate(
            messages=_to_endpoint_messages(messages, images),
            sampling_params=sampling_params,
            tools=tool_schemas if send_tools else None,
            extra_body=request_extra_body,
        )
        final_text = final_output.text or ""
        assistant_turns += 1
        user_turns += 1

    response_text = final_text if not masked_context_text else f"{masked_context_text}{final_text}"
    output_multi_modal_data: dict[str, Any] = {}
    if images:
        output_multi_modal_data["images"] = images

    extra_fields = {
        "vtool_tool_attempted": tool_attempted,
        "vtool_tool_success": tool_success,
        "vtool_tool_failure_reason": tool_failure_reason,
        "vtool_tool_failure_detail": tool_failure_detail,
        "vtool_tool_recovery_mode": tool_recovery_mode,
        "vtool_executed_tool_code": executed_tool_code,
        "vtool_tool_parse_status": parse_result.status,
        "vtool_tool_parse_message": parse_result.message,
        "vtool_tool_parse_error_code": parse_result.error_code,
        "vtool_initial_response_text": initial_text,
        "vtool_final_response_text": final_text,
    }

    return EvalLoopOutput(
        response_text=response_text,
        generated_text=final_text,
        masked_context_text=masked_context_text,
        num_turns=user_turns + assistant_turns + 1,
        metrics=metrics,
        multi_modal_data=output_multi_modal_data,
        extra_fields=extra_fields,
    )
