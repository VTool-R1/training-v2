from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_TOOL_CONFIG_PATH = Path(__file__).resolve().with_name("image_zoom_in_tool_config.yaml")


def load_chat_template(chat_template: str | None = None, chat_template_path: str | None = None) -> str | None:
    if chat_template and chat_template_path:
        raise ValueError("Specify only one of chat_template or chat_template_path.")
    if chat_template_path:
        return Path(chat_template_path).read_text(encoding="utf-8")
    return chat_template


def parse_json_dict(raw_value: str | None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if raw_value is None or raw_value.strip() == "":
        return {} if default is None else dict(default)

    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def build_vtool_eval_config(
    *,
    model_path: str,
    trust_remote_code: bool,
    custom_chat_template: str | None,
    apply_chat_template_kwargs: dict[str, Any],
    prompt_length: int,
    response_length: int,
    max_model_len: int | None,
    tool_config_path: str,
    tool_parser_format: str,
    max_user_turns: int,
    max_assistant_turns: int,
    max_parallel_calls: int,
    max_tool_response_length: int,
    tool_response_truncate_side: str,
):
    from omegaconf import OmegaConf

    resolved_max_model_len = max_model_len or (prompt_length + response_length)

    return OmegaConf.create(
        {
            "data": {
                "apply_chat_template_kwargs": dict(apply_chat_template_kwargs),
            },
            "actor_rollout_ref": {
                "model": {
                    "path": model_path,
                    "trust_remote_code": trust_remote_code,
                    "custom_chat_template": custom_chat_template,
                },
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "max_model_len": resolved_max_model_len,
                    "multi_turn": {
                        "max_user_turns": max_user_turns,
                        "max_assistant_turns": max_assistant_turns,
                        "max_parallel_calls": max_parallel_calls,
                        "max_tool_response_length": max_tool_response_length,
                        "tool_response_truncate_side": tool_response_truncate_side,
                        "tool_config_path": tool_config_path,
                        "format": tool_parser_format,
                    },
                },
            },
        }
    )

