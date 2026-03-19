from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_completed_indices(path: str | Path) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    completed = set()
    for record in iter_jsonl(path):
        index = record.get("index")
        if isinstance(index, int):
            completed.add(index)
    return completed


def extract_prediction_text(record: dict[str, Any]) -> str:
    candidates = [
        record.get("model_response"),
        record.get("generated_text"),
        record.get("final_answer_text"),
        record.get("response_text"),
        record.get("prediction"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def extract_answer_span(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    final_match = re.search(r"FINAL ANSWER:\s*(.*?)\s*(?:TERMINATE|$)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()

    answer_prefix_match = re.search(r"ANSWER:\s*(.*?)\s*(?:TERMINATE|$)", text, re.DOTALL | re.IGNORECASE)
    if answer_prefix_match:
        return answer_prefix_match.group(1).strip()

    boxed_match = re.search(r"\\boxed\{(.+?)\}", text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    return text

