from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from recipe.vtool.refocus_tools import RefocusCodeParser
from recipe.vtool.reward_utils import compute_score as judge_compute_score
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SUCCESS_OBSERVATION = (
    "OBSERVATION: Execution success. The output is as follows:\n"
    "<the image outputs of the code is added as the second image>"
)
FAILURE_OBSERVATION = (
    "OBSERVATION: Execution failed. "
    "The code did not produce a valid edited image. "
    "Please regenerate your final answer."
)


def compute_score(*args, **kwargs):
    return judge_compute_score(*args, **kwargs)


def _normalize_object_field(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0]
    return value


class VToolRewardManager(RewardManagerBase):
    """Reward only the final assistant segment that remains trainable after the tool round."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or judge_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        tool_extra_fields = _normalize_object_field(data_item.non_tensor_batch.get("tool_extra_fields", {}))
        if not isinstance(tool_extra_fields, dict):
            tool_extra_fields = {}

        response_str = tool_extra_fields.get("vtool_final_response_text")
        if response_str is None:
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            if "response_mask" in data_item.batch.keys():
                valid_response_mask = data_item.batch["response_mask"][:valid_response_length].bool()
                final_response_ids = valid_response_ids[valid_response_mask]
                if final_response_ids.numel() == 0:
                    final_response_ids = valid_response_ids
            else:
                final_response_ids = valid_response_ids

            response_str = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(final_response_ids.tolist(), skip_special_tokens=True),
            )

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["tool_extra_fields"] = tool_extra_fields

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            reward_score = float(result["score"])
            reward_extra_info.update(result)
        else:
            reward_score = float(result)
            reward_extra_info["acc"] = reward_score

        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }


class VToolAgentLoop(AgentLoopBase):
    """One-turn image refocus loop ported from the older verl 0.5 based VTool fork."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.tool_timeout_seconds = 10.0
        self.code_parser = RefocusCodeParser()
        self._tool_output: Image.Image | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = list(multi_modal_data.get("images") or [])
        videos = list(multi_modal_data.get("videos") or [])
        tools_kwargs = kwargs.get("tools_kwargs") or {}
        metrics: dict[str, Any] = {
            "tool_use_attempted": 0.0,
            "tool_use_success": 0.0,
        }
        request_id = uuid4().hex

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images or None,
            videos=videos or None,
        )

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        user_turns = 0
        waiting_final_after_tool = False
        tool_attempted = False
        tool_success = False

        while True:
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images or None,
                    video_data=videos or None,
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            current_resp_start = len(response_mask)
            response_ids_chunk = output.token_ids
            prompt_ids += response_ids_chunk
            response_mask += [1] * len(response_ids_chunk)
            response_logprobs = self._extend_logprobs(
                response_logprobs,
                current_response_len=current_resp_start,
                chunk_len=len(response_ids_chunk),
                chunk_logprobs=output.log_probs,
            )
            assistant_turns += 1

            if not waiting_final_after_tool and assistant_turns == 1:
                raw_response = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(response_ids_chunk, skip_special_tokens=False),
                )
                parse_result = self.code_parser.parse(raw_response)

                if parse_result.error_code == "NOTOOL":
                    break

                tool_attempted = True
                metrics["tool_use_attempted"] = 1.0
                for index in range(current_resp_start, current_resp_start + len(response_ids_chunk)):
                    response_mask[index] = 0
                if response_logprobs:
                    for index in range(current_resp_start, current_resp_start + len(response_ids_chunk)):
                        response_logprobs[index] = 0.0

                with simple_timer("tool_calls", metrics):
                    observation_ids, edited_image, tool_success = await self._run_tool_round(
                        parse_result=parse_result,
                        images=images,
                        tools_kwargs=tools_kwargs,
                    )
                metrics["tool_use_success"] = float(tool_success)

                if len(response_mask) + len(observation_ids) >= self.response_length:
                    break

                prompt_ids += observation_ids
                response_mask += [0] * len(observation_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(observation_ids)

                if edited_image is not None:
                    images.append(edited_image)

                user_turns += 1
                waiting_final_after_tool = True
                continue

            if waiting_final_after_tool:
                break

            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

        if response_mask:
            response_ids = prompt_ids[-len(response_mask) :]
            prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        else:
            response_ids = []

        output_multi_modal_data: dict[str, Any] = {}
        if images:
            output_multi_modal_data["images"] = images
        if videos:
            output_multi_modal_data["videos"] = videos

        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "vtool_tool_attempted": tool_attempted,
            "vtool_tool_success": tool_success,
        }

        final_response_ids = [token for token, keep in zip(response_ids, response_mask, strict=False) if keep]
        if final_response_ids:
            extra_fields["vtool_final_response_text"] = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(final_response_ids, skip_special_tokens=True),
            )
        else:
            extra_fields["vtool_final_response_text"] = ""

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=output_multi_modal_data,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )

    def _display(self, image: Image.Image) -> None:
        self._tool_output = image

    @staticmethod
    def _extend_logprobs(
        existing_logprobs: list[float],
        *,
        current_response_len: int,
        chunk_len: int,
        chunk_logprobs: list[float] | None,
    ) -> list[float]:
        if chunk_logprobs:
            if len(existing_logprobs) < current_response_len:
                existing_logprobs.extend([0.0] * (current_response_len - len(existing_logprobs)))
            existing_logprobs.extend(chunk_logprobs)
        elif existing_logprobs:
            existing_logprobs.extend([0.0] * chunk_len)
        return existing_logprobs

    async def _run_tool_round(
        self,
        *,
        parse_result,
        images: list[Image.Image],
        tools_kwargs: dict[str, Any],
    ) -> tuple[list[int], Image.Image | None, bool]:
        edited_image = None
        success = False

        if parse_result.status and images:
            metadata = self._parse_metadata(tools_kwargs.get("metadata"))
            context = self.code_parser.get_tool_context(self._display)
            source = metadata.get("source", "")
            x_bbox = metadata.get("x_values_bbox")
            y_bbox = metadata.get("y_values_bbox")

            if source == "chartqa_v_bar":
                bbox_mapping = x_bbox
            elif source == "chartqa_h_bar":
                bbox_mapping = y_bbox
            else:
                bbox_mapping = x_bbox

            if "tablevqa" in source:
                context["columns_bbox"] = x_bbox
                context["rows_bbox"] = y_bbox
            else:
                context["columns_bbox"] = bbox_mapping
                context["rows_bbox"] = bbox_mapping

            context["display"] = self._display
            context["image_1"] = images[0]
            initial_context_keys = set(context)
            self._tool_output = None
            executable_code = self.code_parser.ensure_display_call(parse_result.code)

            def _exec_tool():
                exec(executable_code, context)

            try:
                await asyncio.wait_for(
                    self.loop.run_in_executor(None, _exec_tool),
                    timeout=self.tool_timeout_seconds,
                )
                if (
                    isinstance(self._tool_output, Image.Image)
                    and self._tool_output.size[0] > 0
                    and self._tool_output.size[1] > 0
                ):
                    edited_image = self._tool_output
                    success = True
                else:
                    for key, value in reversed(list(context.items())):
                        if key in initial_context_keys:
                            continue
                        if isinstance(value, Image.Image) and value.size[0] > 0 and value.size[1] > 0:
                            logger.info(
                                "Recovered VTool output from variable '%s' without an explicit display(...) call.",
                                key,
                            )
                            edited_image = value
                            success = True
                            break
            except Exception as exc:
                logger.warning("VTool code execution failed: %s", exc)

        observation_ids = await self._build_observation_ids(edited_image if success else None)
        return observation_ids, edited_image, success

    async def _build_observation_ids(self, edited_image: Image.Image | None) -> list[int]:
        if edited_image is not None and self.processor is not None:
            return await self.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": SUCCESS_OBSERVATION},
                        ],
                    }
                ],
                images=[edited_image],
                remove_system_prompt=True,
            )

        if edited_image is not None:
            observation = (
                "OBSERVATION: Execution success. The tool produced an edited image. "
                "Please regenerate your final answer based on this observation."
            )
        else:
            observation = FAILURE_OBSERVATION

        return await self.apply_chat_template(
            [{"role": "user", "content": observation}],
            remove_system_prompt=True,
        )

    @staticmethod
    def _parse_metadata(metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str) and metadata:
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        return {}
