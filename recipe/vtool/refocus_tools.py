from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any, Callable

from PIL import Image, ImageDraw

PAD_MASK = 1
PAD_DRAW = 2
PAD_HIGHLIGHT = 2
DRAW_WIDTH = 3
HIGHLIGHT_FILL = (255, 0, 0, 50)


def expand_bbox(bbox, image, pad):
    w, h = image.size
    x1 = max(0, int(bbox["x1"] - pad))
    y1 = max(0, int(bbox["y1"] - pad))
    x2 = min(w, int(bbox["x2"] + pad))
    y2 = min(h, int(bbox["y2"] + pad))
    return x1, y1, x2, y2


def _resolve_keys(requested_keys, all_bboxes, fuzzy=False):
    resolved = []
    for key in requested_keys:
        if key in all_bboxes:
            resolved.append(key)
            continue

        if fuzzy:
            match = None
            for candidate in all_bboxes:
                if key == candidate or key in candidate or candidate in key:
                    match = candidate
                    break
            if match is not None:
                resolved.append(match)

    return list(dict.fromkeys(resolved))


def _apply_to_regions(
    image,
    keys_to_use,
    all_bboxes,
    mode,
    *,
    pad,
    fill="white",
    outline="red",
    width=DRAW_WIDTH,
    fuzzy=False,
):
    if not all_bboxes or not keys_to_use:
        return image

    resolved_keys = _resolve_keys(keys_to_use, all_bboxes, fuzzy=fuzzy)
    if not resolved_keys:
        return image

    if mode == "highlight":
        target = image.convert("RGBA").copy()
        draw = ImageDraw.Draw(target, "RGBA")
    else:
        target = image
        draw = ImageDraw.Draw(target, "RGBA")

    for key in resolved_keys:
        bbox = all_bboxes[key]
        x1, y1, x2, y2 = expand_bbox(bbox, image, pad=pad)

        if mode == "mask":
            draw.rectangle(((x1, y1), (x2, y2)), fill=fill)
        elif mode == "draw":
            draw.rectangle(((x1, y1), (x2, y2)), outline=outline, width=width)
        elif mode == "highlight":
            draw.rectangle(((x1, y1), (x2, y2)), fill=fill)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if mode == "highlight":
        return Image.alpha_composite(image.convert("RGBA"), target)
    return target


def _mask_except(image, keys_to_focus_on, all_bboxes, *, pad, keep_first=False, fuzzy=False):
    if not all_bboxes or not keys_to_focus_on:
        return image

    all_keys = list(all_bboxes.keys())
    focus_keys = set(_resolve_keys(keys_to_focus_on, all_bboxes, fuzzy=fuzzy))

    if keep_first and all_keys:
        candidate_keys = all_keys[1:]
    else:
        candidate_keys = all_keys

    keys_to_mask = [k for k in candidate_keys if k not in focus_keys]

    if not keys_to_mask:
        return image

    if keep_first and len(keys_to_mask) == max(0, len(all_keys) - 1):
        return image
    if not keep_first and len(keys_to_mask) == len(all_keys):
        return image

    return _apply_to_regions(
        image,
        keys_to_mask,
        all_bboxes,
        "mask",
        pad=pad,
        fill="white",
        fuzzy=fuzzy,
    )


def focus_on_columns_with_mask(image, columns_to_focus_on, all_columns_bounding_boxes):
    return _mask_except(
        image,
        columns_to_focus_on,
        all_columns_bounding_boxes,
        pad=PAD_MASK,
        keep_first=False,
        fuzzy=False,
    )


def focus_on_rows_with_mask(image, rows_to_focus_on, all_rows_bounding_boxes):
    return _mask_except(
        image,
        rows_to_focus_on,
        all_rows_bounding_boxes,
        pad=PAD_MASK,
        keep_first=True,
        fuzzy=False,
    )


def focus_on_columns_with_draw(image, columns_to_focus_on, all_columns_bounding_boxes):
    return _apply_to_regions(
        image,
        columns_to_focus_on,
        all_columns_bounding_boxes,
        "draw",
        pad=PAD_DRAW,
        outline="red",
        width=DRAW_WIDTH,
        fuzzy=False,
    )


def focus_on_rows_with_draw(image, rows_to_focus_on, all_rows_bounding_boxes):
    return _apply_to_regions(
        image,
        rows_to_focus_on,
        all_rows_bounding_boxes,
        "draw",
        pad=PAD_DRAW,
        outline="red",
        width=DRAW_WIDTH,
        fuzzy=False,
    )


def focus_on_columns_with_highlight(image, columns_to_focus_on, all_columns_bounding_boxes):
    return _apply_to_regions(
        image,
        columns_to_focus_on,
        all_columns_bounding_boxes,
        "highlight",
        pad=PAD_HIGHLIGHT,
        fill=HIGHLIGHT_FILL,
        fuzzy=False,
    )


def focus_on_rows_with_highlight(image, rows_to_focus_on, all_rows_bounding_boxes):
    return _apply_to_regions(
        image,
        rows_to_focus_on,
        all_rows_bounding_boxes,
        "highlight",
        pad=PAD_HIGHLIGHT,
        fill=HIGHLIGHT_FILL,
        fuzzy=False,
    )


def focus_on_x_values_with_mask(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    return _mask_except(
        image,
        x_values_to_focus_on,
        all_x_values_bounding_boxes,
        pad=PAD_MASK,
        keep_first=False,
        fuzzy=False,
    )


def focus_on_y_values_with_mask(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    return _mask_except(
        image,
        y_values_to_focus_on,
        all_y_values_bounding_boxes,
        pad=PAD_MASK,
        keep_first=False,
        fuzzy=False,
    )


def focus_on_x_values_with_draw(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    return _apply_to_regions(
        image,
        x_values_to_focus_on,
        all_x_values_bounding_boxes,
        "draw",
        pad=PAD_DRAW,
        outline="red",
        width=DRAW_WIDTH,
        fuzzy=False,
    )


def focus_on_y_values_with_draw(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    return _apply_to_regions(
        image,
        y_values_to_focus_on,
        all_y_values_bounding_boxes,
        "draw",
        pad=PAD_DRAW,
        outline="red",
        width=DRAW_WIDTH,
        fuzzy=True,
    )


def focus_on_x_values_with_highlight(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    return _apply_to_regions(
        image,
        x_values_to_focus_on,
        all_x_values_bounding_boxes,
        "highlight",
        pad=PAD_HIGHLIGHT,
        fill=HIGHLIGHT_FILL,
        fuzzy=False,
    )


def focus_on_y_values_with_highlight(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    return _apply_to_regions(
        image,
        y_values_to_focus_on,
        all_y_values_bounding_boxes,
        "highlight",
        pad=PAD_HIGHLIGHT,
        fill=HIGHLIGHT_FILL,
        fuzzy=False,
    )


def build_refocus_context(
    *,
    display_callback: Callable[[Image.Image], None],
    image: Image.Image,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    parser = RefocusCodeParser()
    context = parser.get_tool_context(display_callback)

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

    context["display"] = display_callback
    context["image_1"] = image
    return context


@dataclass(frozen=True)
class ParseResult:
    status: bool
    code: str
    message: str
    error_code: str


class RefocusCodeParser:
    _TOOL_HINT_PATTERNS = (
        "focus_on_",
        "display(",
        "image_1",
        "columns_bbox",
        "rows_bbox",
    )

    def get_tool_context(self, display):
        return {
            "display": display,
            "focus_on_columns_with_mask": focus_on_columns_with_mask,
            "focus_on_rows_with_mask": focus_on_rows_with_mask,
            "focus_on_columns_with_draw": focus_on_columns_with_draw,
            "focus_on_rows_with_draw": focus_on_rows_with_draw,
            "focus_on_columns_with_highlight": focus_on_columns_with_highlight,
            "focus_on_rows_with_highlight": focus_on_rows_with_highlight,
            "focus_on_x_values_with_mask": focus_on_x_values_with_mask,
            "focus_on_y_values_with_mask": focus_on_y_values_with_mask,
            "focus_on_x_values_with_draw": focus_on_x_values_with_draw,
            "focus_on_y_values_with_draw": focus_on_y_values_with_draw,
            "focus_on_x_values_with_highlight": focus_on_x_values_with_highlight,
            "focus_on_y_values_with_highlight": focus_on_y_values_with_highlight,
        }

    def _looks_like_vtool_code(self, content: str) -> bool:
        text = content or ""
        return any(pattern in text for pattern in self._TOOL_HINT_PATTERNS)

    def _looks_like_direct_answer(self, content: str) -> bool:
        return bool(re.search(r"\b(?:FINAL ANSWER|ANSWER)\s*:", content, re.IGNORECASE))

    @staticmethod
    def _get_call_name(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _has_display_call(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if self._get_call_name(node) == "display":
                return True
        return False

    def ensure_display_call(self, code: str) -> str:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            return code

        if self._has_display_call(tree):
            return code

        candidate_name: str | None = None
        candidate_expr: str | None = None

        for stmt in tree.body:
            value: ast.AST | None = None
            target_name: str | None = None

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                value = stmt.value
                target_name = stmt.targets[0].id
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                value = stmt.value
                target_name = stmt.target.id
            elif isinstance(stmt, ast.Expr):
                value = stmt.value

            if self._get_call_name(value) and self._get_call_name(value).startswith("focus_on_"):
                if target_name:
                    candidate_name = target_name
                    candidate_expr = None
                else:
                    candidate_expr = ast.get_source_segment(code, value)
                    candidate_name = None

        if candidate_name:
            return f"{code.rstrip()}\n\ndisplay({candidate_name})\n"
        if candidate_expr:
            return f"{code.rstrip()}\n\ndisplay({candidate_expr})\n"
        return code

    def parse(self, response: Any) -> ParseResult:
        if isinstance(response, dict) and "content" in response:
            response = response["content"]

        original_content = str(response).replace("\\_", "_")
        content = original_content.replace("\\", "")

        try:
            start_pos = content.find("```python")
            if start_pos != -1:
                content = content[start_pos + len("```python") :]

            end_pos = content.find("```")
            if end_pos != -1:
                content = content[:end_pos]

            if start_pos == -1:
                return ParseResult(False, content, "No tool call", "NOTOOL")

            stripped_content = content.strip()
            if stripped_content and not self._looks_like_vtool_code(content):
                message = "Python block does not appear to be a VTool call; treating it as a direct answer."
                return ParseResult(False, content, message, "NOTOOL")

            if end_pos == -1:
                if self._looks_like_direct_answer(content):
                    message = "Direct answer found inside an unclosed python fence; treating it as a no-tool response."
                    return ParseResult(False, content, message, "NOTOOL")
                return ParseResult(
                    False,
                    content,
                    "Program is NOT enclosed in ```python``` properly.",
                    "unknown",
                )

            if stripped_content:
                compile(content, "prog.py", "exec")
                return ParseResult(True, content, "Parsing succeeded.", "")

            return ParseResult(
                False,
                content,
                "The content is empty, or it failed to parse the content correctly.",
                "unknown",
            )

        except Exception as err:
            return ParseResult(False, content, f"Unexpected {type(err)}: {err}.", "unknown")

    def trim_to_action_end(self, text):
        last_code_block_start = text.rfind("```")
        if last_code_block_start == -1:
            return text
        preceding_code_block_start = text.rfind("```", 0, last_code_block_start)
        if preceding_code_block_start == -1:
            return text
        return text[: last_code_block_start + 3]
