from __future__ import annotations

import unittest

from PIL import Image

from recipe.vtool.refocus_tools import RefocusCodeParser, build_refocus_context
from recipe.vtool.reward_utils import env_flag, parse_extract_answer, resolve_api_base


class VToolHelperTests(unittest.TestCase):
    def test_parse_extract_answer_prefers_final_answer(self):
        text = "THOUGHT ... ANSWER: 41 FINAL ANSWER: 42. TERMINATE"
        self.assertEqual(parse_extract_answer(text), "42.")

    def test_refocus_parser_detects_missing_tool(self):
        result = RefocusCodeParser().parse("FINAL ANSWER: 1. TERMINATE")
        self.assertFalse(result.status)
        self.assertEqual(result.error_code, "NOTOOL")

    def test_refocus_parser_extracts_python_block(self):
        result = RefocusCodeParser().parse(
            "ACTION:\n```python\nimage_with_focus = focus_on_columns_with_highlight(image_1, ['A'], columns_bbox)\n"
            "display(image_with_focus)\n```"
        )
        self.assertTrue(result.status)
        self.assertIn("focus_on_columns_with_highlight", result.code)

    def test_refocus_parser_treats_arithmetic_python_as_no_tool(self):
        result = RefocusCodeParser().parse("ACTION:\n```python\nx = 1 + 2\n```")
        self.assertFalse(result.status)
        self.assertEqual(result.error_code, "NOTOOL")

    def test_refocus_parser_treats_answer_inside_python_fence_as_no_tool(self):
        result = RefocusCodeParser().parse("ACTION:\n```python\nFINAL ANSWER: 42. TERMINATE\n```")
        self.assertFalse(result.status)
        self.assertEqual(result.error_code, "NOTOOL")

    def test_ensure_display_call_appends_display_for_last_focus_image(self):
        parser = RefocusCodeParser()
        code = (
            "image_with_focused_columns = focus_on_columns_with_highlight(image_1, ['A'], columns_bbox)\n"
            "image_with_focused_rows = focus_on_rows_with_draw(image_with_focused_columns, ['B'], rows_bbox)"
        )
        rewritten = parser.ensure_display_call(code)
        self.assertIn("display(image_with_focused_rows)", rewritten)

    def test_ensure_display_call_leaves_existing_display_unchanged(self):
        parser = RefocusCodeParser()
        code = (
            "image_with_focus = focus_on_columns_with_highlight(image_1, ['A'], columns_bbox)\n"
            "display(image_with_focus)\n"
        )
        rewritten = parser.ensure_display_call(code)
        self.assertEqual(rewritten, code)

    def test_build_refocus_context_uses_y_bbox_for_horizontal_bar(self):
        image = Image.new("RGB", (16, 16), color="white")
        metadata = {
            "source": "chartqa_h_bar",
            "x_values_bbox": {"foo": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}},
            "y_values_bbox": {"bar": {"x1": 3, "y1": 3, "x2": 4, "y2": 4}},
        }
        context = build_refocus_context(display_callback=lambda _: None, image=image, metadata=metadata)
        self.assertEqual(context["columns_bbox"], metadata["y_values_bbox"])
        self.assertEqual(context["rows_bbox"], metadata["y_values_bbox"])

    def test_build_refocus_context_exposes_legacy_aliases(self):
        image = Image.new("RGB", (16, 16), color="white")
        metadata = {
            "source": "tablevqa",
            "x_values_bbox": {"col": {"x1": 1, "y1": 1, "x2": 6, "y2": 6}},
            "y_values_bbox": {"row": {"x1": 2, "y1": 2, "x2": 8, "y2": 8}},
        }
        context = build_refocus_context(display_callback=lambda _: None, image=image, metadata=metadata)
        self.assertIn("image_1", context)
        self.assertIn("focus_on_columns_with_mask", context)
        self.assertNotIn("image", context)
        self.assertEqual(context["rows_bbox"], metadata["y_values_bbox"])

    def test_context_uses_raw_pil_images(self):
        image = Image.new("RGB", (16, 16), color="white")
        rendered = []
        metadata = {
            "source": "tablevqa",
            "x_values_bbox": {"col": {"x1": 1, "y1": 1, "x2": 6, "y2": 6}},
            "y_values_bbox": {"row": {"x1": 2, "y1": 2, "x2": 8, "y2": 8}},
        }
        context = build_refocus_context(display_callback=rendered.append, image=image, metadata=metadata)
        self.assertIs(context["image_1"], image)
        edited = context["focus_on_rows_with_draw"](context["image_1"], ["row"], context["rows_bbox"])
        context["display"](edited)
        self.assertTrue(rendered)
        self.assertIsInstance(rendered[0], Image.Image)

    def test_resolve_api_base_supports_host_and_port(self):
        api_base = resolve_api_base(env={"VTOOL_JUDGE_HOST": "10.0.0.9", "VTOOL_JUDGE_PORT": "9001"})
        self.assertEqual(api_base, "http://10.0.0.9:9001/v1")

    def test_resolve_api_base_prefers_explicit_api_base(self):
        api_base = resolve_api_base(
            env={
                "VTOOL_JUDGE_API_BASE": "http://judge-box:7000/v1",
                "VTOOL_JUDGE_HOST": "10.0.0.9",
                "VTOOL_JUDGE_PORT": "9001",
            }
        )
        self.assertEqual(api_base, "http://judge-box:7000/v1")

    def test_env_flag_parses_truthy_values(self):
        self.assertTrue(env_flag("VTOOL_JUDGE_USE_ENDPOINT_DEFAULT", env={"VTOOL_JUDGE_USE_ENDPOINT_DEFAULT": "1"}))
        self.assertTrue(env_flag("VTOOL_JUDGE_USE_ENDPOINT_DEFAULT", env={"VTOOL_JUDGE_USE_ENDPOINT_DEFAULT": "true"}))
        self.assertFalse(env_flag("VTOOL_JUDGE_USE_ENDPOINT_DEFAULT", env={"VTOOL_JUDGE_USE_ENDPOINT_DEFAULT": "0"}))


if __name__ == "__main__":
    unittest.main()
