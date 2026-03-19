from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache
from typing import Any

GRADING_QUERY = """
You are given a prediction for the question. You need to rate it given the correct answer. Disregard the format, and only rate based on the content. If you think the prediction is correct, i.e. same as the correct answer, then return 1, otherwise return 0. Return 0 or 1 only.

# Example
# Question: What is the height of the tower?
# Prediction: ANSWER: The tower is built in China from 200 years ago. The total hight of the tower is 180 Meters. TERMINATE
# Correct Answer: 80 Meters
# Your Response: 0

# Example
# Question: What is the difference?
# Prediction: ANSWER: 69%. TERMINATE
# Correct Answer: 69
# Your Response: 1

# Example
# Question: What is the increase?
# Prediction: ANSWER: 3%. TERMINATE
# Correct Answer: 0.03
# Your Response: 1

# Example
# Question: What is the ratio?
# Prediction: ANSWER: 1.541. TERMINATE
# Correct Answer: 1.54
# Your Response: 1

# Prediction: <answer>
# Correct Answer: <gt>
# Your Response:
""".strip()


def normalize_api_base(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/v1"):
        return api_base
    return f"{api_base}/v1"


def resolve_api_base(
    api_base: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str:
    env = os.environ if env is None else env

    if api_base:
        return normalize_api_base(api_base)

    explicit_api_base = env.get("VTOOL_JUDGE_API_BASE") or env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL")
    if explicit_api_base:
        return normalize_api_base(explicit_api_base)

    scheme = env.get("VTOOL_JUDGE_SCHEME", "http")
    host = env.get("VTOOL_JUDGE_HOST", "localhost")
    port = env.get("VTOOL_JUDGE_PORT", "8000")
    return normalize_api_base(f"{scheme}://{host}:{port}")


def env_flag(name: str, *, default: bool = False, env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_extract_answer(pred: str) -> str:
    pred = pred.replace("||", "|")

    result = re.search(r"FINAL ANSWER:\s*(.*?)\s*TERMINATE", pred, re.DOTALL)
    if result:
        return result.group(1).strip()

    result = re.search(r"FINAL ANSWER:\s*(.*?)(?=\s*(?:\||$))", pred, re.DOTALL)
    if result:
        return result.group(1).strip()

    result = re.search(r"ANSWER:\s*(.*?)(?=\s*(?:\||$))", pred, re.DOTALL)
    if result:
        return result.group(1).strip()

    return pred.strip()


def _request_json(url: str, *, api_key: str, payload: dict[str, Any] | None, timeout_seconds: float) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, headers=headers, data=data)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


@lru_cache(maxsize=16)
def pick_default_model_id(api_base: str, api_key: str, timeout_seconds: float) -> str:
    response = _request_json(
        f"{normalize_api_base(api_base)}/models",
        api_key=api_key,
        payload=None,
        timeout_seconds=timeout_seconds,
    )
    models = response.get("data", [])
    if not models:
        raise RuntimeError(f"No models returned by server at {api_base}")

    for model in models:
        model_id = str(model.get("id", ""))
        if model_id.lower() in {"default", "gpt", "chat", "auto"}:
            return model_id

    return str(models[0]["id"])


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 300.0,
    temperature: float = 0.0,
    max_tokens: int = 10,
    **_: Any,
) -> float:
    del data_source, extra_info

    try:
        final_answer = parse_extract_answer(solution_str)
        resolved_api_base = resolve_api_base(api_base)
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        use_endpoint_default_model = env_flag("VTOOL_JUDGE_USE_ENDPOINT_DEFAULT", default=False)
        resolved_model = model
        if not resolved_model and not use_endpoint_default_model:
            resolved_model = os.environ.get("VTOOL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL")
        if not resolved_model:
            resolved_model = pick_default_model_id(resolved_api_base, resolved_api_key, timeout_seconds)

        prompt = GRADING_QUERY.replace("<gt>", ground_truth).replace("<answer>", final_answer)
        response = _request_json(
            f"{resolved_api_base}/chat/completions",
            api_key=resolved_api_key,
            payload={
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout_seconds=timeout_seconds,
        )
        output_text = str(response["choices"][0]["message"]["content"]).strip()
        return 1.0 if output_text.startswith("1") else 0.0
    except Exception:
        return 0.0
