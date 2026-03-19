from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Any

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from eval.result_io import extract_answer_span
else:
    from .result_io import extract_answer_span


DEFAULT_LOCAL_SCORER_BASE_URL = "http://localhost:8000/v1"
DEFAULT_REMOTE_SCORER_MODEL = "gpt-4o"


@lru_cache(maxsize=2)
def _get_scorer_client(local: bool = False):
    from openai import OpenAI

    base_url = os.environ.get("SCORER_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if local and not base_url:
        base_url = DEFAULT_LOCAL_SCORER_BASE_URL

    api_key = os.environ.get("SCORER_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if base_url and not api_key:
        api_key = "EMPTY"

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    model = os.environ.get("SCORER_MODEL")
    if not model:
        if base_url:
            models = client.models.list()
            if not models.data:
                raise RuntimeError(f"No models found at scorer endpoint {base_url}")
            model = models.data[0].id
        else:
            model = DEFAULT_REMOTE_SCORER_MODEL

    return client, model, base_url


def get_scorer_config(local: bool = False) -> dict[str, Any]:
    _, model, base_url = _get_scorer_client(local=local)
    return {"model": model, "base_url": base_url}


def _get_scorer_max_tokens() -> int:
    raw_value = os.environ.get("SCORER_MAX_TOKENS", "2048").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 2048


def _disable_thinking_for_local_server() -> bool:
    raw_value = os.environ.get("SCORER_DISABLE_THINKING", "1").strip().lower()
    return raw_value not in {"0", "false", "no"}


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


def _extract_completion_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is not None:
        content = _extract_text_from_content(getattr(message, "content", None))
        if content.strip():
            return content.strip()

    choice_text = getattr(choice, "text", None)
    if isinstance(choice_text, str) and choice_text.strip():
        return choice_text.strip()

    return ""


def _parse_binary_score(score_text: str) -> int:
    score_text = score_text.strip()
    if score_text in {"0", "1"}:
        return int(score_text)

    match = re.search(r"\b([01])\b", score_text)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not parse binary score from {score_text!r}")


def run_scorer_prompt(prompt: str, *, local: bool = False, temperature: float = 0.0, max_retries: int = 5) -> str:
    client, model, base_url = _get_scorer_client(local=local)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
        "max_tokens": _get_scorer_max_tokens(),
    }
    if base_url and _disable_thinking_for_local_server():
        request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(**request_kwargs)
            output_text = _extract_completion_text(response)
            if output_text:
                return output_text
            last_error = ValueError("Empty scorer response")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(exc)
            time.sleep(1)

    print(f"Scorer request failed after retry limit: {last_error}")
    return "0"


def match_score(prompt: str, *, local: bool = False) -> int:
    return _parse_binary_score(run_scorer_prompt(prompt, local=local, temperature=0.0))


def compute_acc_from_raw_answer(question: str, solution: str, pred: str, *, local: bool = False) -> tuple[int, str]:
    prediction = extract_answer_span((pred or "").replace("||", "|"))
    grading_query = f"""
You are given a prediction for the question. Rate it against the correct answer.
Ignore formatting differences. If the prediction is correct in meaning, return 1.
Otherwise return 0. Return 0 or 1 only.

# Example
# Question: What is the difference?
# Prediction: 69%.
# Correct Answer: 69
# Your Response: 1

# Example
# Question: What is the increase?
# Prediction: 3%.
# Correct Answer: 0.03
# Your Response: 1

# Example
# Question: What is the ratio?
# Prediction: 1.541
# Correct Answer: 1.54
# Your Response: 1

# Question: {question}
# Prediction: {prediction}
# Correct Answer: {solution}
# Your Response:"""
    return match_score(grading_query, local=local), prediction
