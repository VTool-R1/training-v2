from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from eval.vtool_eval_loop import run_vtool_eval_loop


class DummyServerManager:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    async def generate(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(text=next(self._responses))


def test_malformed_python_block_is_not_executed() -> None:
    server_manager = DummyServerManager(
        [
            (
                "THOUGHT 0: I can answer directly.\n"
                "ACTION 1: No action needed.\n"
                "```python\n"
                "FINAL ANSWER: 62. TERMINATE\n"
                "```"
            ),
            "FINAL ANSWER: 62. TERMINATE",
        ]
    )

    async def _unexpected_execute_tool_code(**_: object) -> tuple[None, bool]:
        raise AssertionError("Malformed parser output should not be executed.")

    with patch("eval.vtool_eval_loop._execute_tool_code", new=_unexpected_execute_tool_code):
        output = asyncio.run(
            run_vtool_eval_loop(
                raw_prompt=[{"role": "user", "content": "What is the value?"}],
                multi_modal_data={"images": [Image.new("RGB", (4, 4), "white")]},
                tools_kwargs={"metadata": "{}"},
                sampling_params={},
                server_manager=server_manager,
            )
        )

    assert output.metrics["tool_use_attempted"] == 1.0
    assert output.metrics["tool_use_success"] == 0.0
    assert "OBSERVATION: Execution failed." in output.masked_context_text
    assert output.generated_text == "FINAL ANSWER: 62. TERMINATE"
