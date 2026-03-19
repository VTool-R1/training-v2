from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


class OpenAIEndpointError(RuntimeError):
    pass


@dataclass
class ChatCompletionOutput:
    text: str
    finish_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("server_base_url must be non-empty")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
                continue
            if "text" in item:
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


class OpenAIChatServerManager:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str | None,
        request_timeout: float,
        default_extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.request_timeout = float(request_timeout)
        self.default_extra_body = dict(default_extra_body or {})

    @classmethod
    def create(
        cls,
        *,
        base_url: str,
        api_key: str,
        model: str | None,
        request_timeout: float,
        default_extra_body: dict[str, Any] | None = None,
    ) -> "OpenAIChatServerManager":
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            request_timeout=request_timeout,
            default_extra_body=default_extra_body,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_json_sync(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.request_timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OpenAIEndpointError(f"{method} {path} failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise OpenAIEndpointError(
                f"{method} {path} failed: {exc.reason}. "
                "Point --server-base-url or OPENAI_BASE_URL at a reachable OpenAI-compatible endpoint."
            ) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenAIEndpointError(f"{method} {path} returned invalid JSON: {raw[:400]}") from exc

    def _resolve_model_sync(self, model_hint: str | None) -> str:
        if self.model:
            return self.model

        data = self._request_json_sync(method="GET", path="/models")
        model_entries = data.get("data") or []
        model_ids = [entry.get("id") for entry in model_entries if isinstance(entry, dict) and entry.get("id")]
        if not model_ids:
            raise OpenAIEndpointError(f"No models were returned by {self.base_url}/models")

        candidates = [model_hint, str(Path(model_hint).name) if model_hint else None]
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in model_ids:
                self.model = candidate
                return candidate
            for model_id in model_ids:
                if model_id == candidate or model_id.endswith(candidate) or candidate.endswith(model_id):
                    self.model = model_id
                    return model_id

        if len(model_ids) == 1:
            self.model = model_ids[0]
            return self.model

        raise OpenAIEndpointError(
            "Could not infer served model name from endpoint. "
            f"Pass --server-model explicitly. Available models: {model_ids}"
        )

    async def resolve_model(self, model_hint: str | None) -> str:
        return await asyncio.to_thread(self._resolve_model_sync, model_hint)

    def _chat_completion_sync(self, payload: dict[str, Any]) -> ChatCompletionOutput:
        data = self._request_json_sync(method="POST", path="/chat/completions", payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise OpenAIEndpointError(f"Endpoint returned no choices: {data}")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        text = _extract_text_content(message.get("content"))
        return ChatCompletionOutput(
            text=text,
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionOutput:
        if not self.model:
            raise OpenAIEndpointError("Server model has not been resolved.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        for key, value in sampling_params.items():
            if value is not None:
                payload[key] = value
        if tools:
            payload["tools"] = tools

        merged_extra_body = dict(self.default_extra_body)
        if extra_body:
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            payload["extra_body"] = merged_extra_body

        return await asyncio.to_thread(self._chat_completion_sync, payload)

    async def shutdown(self) -> None:
        return None
