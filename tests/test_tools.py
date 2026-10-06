"""工具层测试：Schema 生成、参数校验、安全边界、执行结果。

工具是"模型的手"，手能不能被信任，完全取决于这里的每一条约束有没有被测到。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List, Optional

from codeassist.tools import (
    ToolRegistry,
    build_default_registry,
    build_parameters_schema,
    cleanup_dir,
    make_scratch_dir,
    tool,
)


class TestSchemaGeneration(unittest.TestCase):
    def test_types_required_and_defaults(self):
        @tool(description="测试用工具", param_docs={"name": "名字"})
        def sample(name: str, count: int = 3, flag: bool = False, items: Optional[List[str]] = None) -> str:
            return name

        schema = sample.parameters
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["name"])
        self.assertEqual(schema["properties"]["name"], {"type": "string", "description": "名字"})
        self.assertEqual(schema["properties"]["count"]["type"], "integer")
        self.assertEqual(schema["properties"]["count"]["default"], 3)
        self.assertEqual(schema["properties"]["flag"]["type"], "boolean")
        self.assertEqual(schema["properties"]["items"]["type"], "array")
        self.assertEqual(schema["properties"]["items"]["items"]["type"], "string")
        self.assertFalse(schema["additionalProperties"])

    def test_openai_protocol_shape(self):
        @tool(description="读取点东西")
        def read(path: str) -> str:
            return path

        payload = read.to_openai_schema()
        self.assertEqual(payload["type"], "function")
        self.assertEqual(payload["function"]["name"], "read")
        self.assertEqual(payload["function"]["description"], "读取点东西")
        self.assertIn("parameters", payload["function"])

    def test_unknown_annotation_degrades_to_string(self):
        class Custom:
            pass

        def func(thing: Custom) -> str:
            return ""

        schema = build_parameters_schema(func)
        # 让模型面对一个它无法构造的类型是最糟的设计，降级为 string 更实用
        self.assertEqual(schema["properties"]["thing"]["type"], "string")

    def test_docstring_used_as_description(self):
        @tool()
        def with_doc(path: str) -> str:
            """第一行才是描述。

            后面是详细说明。
            """
            return path

        self.assertEqual(with_doc.description, "第一行才是描述。")


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        @self.registry.tool(description="把两个数相加", param_docs={"a": "左操作数", "b": "右操作数"})
        def add(a: int, b: int = 1) -> str:
            return str(a + b)

        @self.registry.tool(description="总是失败")
        def boom(reason: str) -> str:
            raise RuntimeError(f"炸了：{reason}")

    def test_names_and_subset(self):
        self.assertEqual(self.registry.names(), ["add", "boom"])
        subset = self.registry.subset(["add"])
        self.assertEqual(subset.names(), ["add"])
        with self.assertRaises(KeyError):
            self.registry.subset(["nope"])

    def test_call_from_json_string(self):
        """协议规定 arguments 是 JSON 字符串，这是最常规的路径。"""
        result = self.registry.call("add", '{"a": 2, "b": 5}')
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "7")

    def test_call_from_dict(self):
        self.assertTrue(self.registry.call("add", {"a": 4}).ok)

    def test_default_argument_applies(self):
        self.assertEqual(self.registry.call("add", {"a": 9}).content, "10")

    def test_unknown_tool_lists_alternatives(self):
        result = self.registry.call("missing", {})
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error)
        self.assertIn("add", result.error)  # 错误信息里给出可用工具，模型才能自我纠正

    def test_bad_json_arguments(self):
        result = self.registry.call("add", "{不是 json")
        self.assertFalse(result.ok)
        self.assertIn("JSON", result.error)

    def test_unknown_parameter_is_rejected_with_guidance(self):
        """"自创参数"是模型高频错误，必须拦住并告诉它正确参数。"""
        result = self.registry.call("add", {"a": 1, "c": 2})
        self.assertFalse(result.ok)
        self.assertIn("参数名错误", result.error)
        self.assertIn("add", result.error)

    def test_tool_exception_becomes_result_not_crash(self):
        """工具失败必须变成结果，而不是让 Agent 崩掉 —— 这是自愈的前提。"""
        result = self.registry.call("boom", {"reason": "测试"})
        self.assertFalse(result.ok)
        self.assertIn("炸了", result.error)
        self.assertIn("[工具执行失败]", result.as_message_content())

    def test_call_log_records_everything(self):
        self.registry.call("add", {"a": 1})
        self.registry.call("missing", {})
        self.assertEqual(len(self.registry.call_log), 2)
        self.assertTrue(self.registry.call_log[0]["ok"])
        self.assertFalse(self.registry.call_log[1]["ok"])

    def test_duplicate_registration_rejected(self):
        existing = self.registry.get("add")
        with self.assertRaises(ValueError):
            self.registry.register_tool(existing)

    def test_describe_mentions_usage(self):
        text = self.registry.describe()
        self.assertIn("add", text)
        self.assertIn("用途", text)


class TestDefaultRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 用项目自身作为工作区，保证测试跑的路径都是真实存在的
        cls.root = Path(__file__).resolve().parent.parent
        cls.registry = build_default_registry(cls.root, allow_execute=True)

    def test_expected_tools_present(self):
        self.assertEqual(
            set(self.registry.names()),
            {"list_dir", "read_file", "write_file", "search_code", "check_python_syntax", "run_python"},
        )

    def test_no_execute_flag_removes_dangerous_tool(self):
        registry = build_default_registry(self.root, allow_execute=False)
        self.assertNotIn("run_python", registry.names())

    def test_list_dir(self):
        result = self.registry.call("list_dir", {"path": "codeassist"})
        self.assertTrue(result.ok)
        self.assertIn("config.py", result.content)

    def test_read_file_with_line_numbers(self):
        result = self.registry.call("read_file", {"path": "codeassist/__init__.py", "start_line": 1, "end_line": 3})
        self.assertTrue(result.ok)
        self.assertIn("1 |", result.content)
        self.assertIn("共", result.content)

    def test_read_out_of_range_line(self):
        result = self.registry.call("read_file", {"path": "codeassist/__init__.py", "start_line": 99999})
        self.assertFalse(result.ok)
        self.assertIn("超出", result.error)

    def test_missing_file_lists_siblings(self):
        """报错要给出"下一步怎么改"，模型才能自己纠正路径。"""
        result = self.registry.call("read_file", {"path": "codeassist/confg.py"})
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error)
        self.assertIn("config.py", result.error)

    def test_path_escape_is_blocked(self):
        """安全边界第一原则：永远不要相信模型给的路径。"""
        for evil in ("../../secret.txt", "../../../Windows/win.ini", "codeassist/../../outside.txt"):
            result = self.registry.call("read_file", {"path": evil})
            self.assertFalse(result.ok, f"{evil} 竟然通过了路径校验")
            self.assertIn("越界", result.error)

    def test_search_code(self):
        result = self.registry.call("search_code", {"pattern": "def extract_json", "file_glob": "*.py"})
        self.assertTrue(result.ok)
        self.assertIn("parsers.py", result.content)

    def test_search_bad_regex_gives_actionable_error(self):
        result = self.registry.call("search_code", {"pattern": "([unclosed"})
        self.assertFalse(result.ok)
        self.assertIn("正则", result.error)

    def test_check_syntax(self):
        ok = self.registry.call("check_python_syntax", {"code": "def f():\n    return 1\n"})
        self.assertTrue(ok.ok)
        self.assertIn("通过", ok.content)
        bad = self.registry.call("check_python_syntax", {"code": "def f(:\n"})
        self.assertIn("未通过", bad.content)

    def test_write_file_roundtrip(self):
        """写入 → 读回，验证工作区内写入链路可用。"""
        target = "codeassist/.test_artifact.txt"
        written = self.registry.call("write_file", {"path": target, "content": "hello\n世界\n"})
        self.assertTrue(written.ok, written.error)
        read_back = self.registry.call("read_file", {"path": target})
        self.assertTrue(read_back.ok)
        self.assertIn("世界", read_back.content)
        Path(self.root / target).unlink(missing_ok=True)

    def test_read_file_strips_utf8_bom(self):
        """带 BOM 的源码文件不能让首行出现 \ufeff 这种鬼字符。"""
        target = Path(self.root) / "codeassist" / ".test_bom.py"
        target.write_bytes(b"\xef\xbb\xbfdef f():\n    return 1\n")
        try:
            result = self.registry.call("read_file", {"path": "codeassist/.test_bom.py"})
            self.assertTrue(result.ok, result.error)
            self.assertNotIn("\ufeff", result.content)
            self.assertIn("def f():", result.content)
        finally:
            target.unlink(missing_ok=True)

    def test_run_python_executes_and_reports_output(self):
        result = self.registry.call("run_python", {"code": "print(6 * 7)"})
        self.assertTrue(result.ok, result.error)
        self.assertIn("42", result.content)
        self.assertIn("退出码：0", result.content)

    def test_run_python_reports_traceback(self):
        result = self.registry.call("run_python", {"code": "raise ValueError('boom')"})
        self.assertTrue(result.ok)  # 执行本身成功，只是脚本退出码非 0
        self.assertIn("ValueError", result.content)
        self.assertIn("退出码：1", result.content)

    def test_run_python_rejects_empty(self):
        self.assertFalse(self.registry.call("run_python", {"code": "   "}).ok)


class TestScratchDir(unittest.TestCase):
    def test_returns_writable_directory(self):
        """核心断言：返回的目录**确实可写**（这是本项目踩过的真实坑）。"""
        base = Path(__file__).resolve().parent.parent
        scratch = make_scratch_dir(prefix="test-scratch-", fallback_parent=base)
        try:
            probe = scratch / "file.txt"
            probe.write_text("ok", encoding="utf-8")
            self.assertEqual(probe.read_text(encoding="utf-8"), "ok")
        finally:
            cleanup_dir(scratch)

    def test_cleanup_is_best_effort_and_never_raises(self):
        cleanup_dir(Path("完全不存在的路径/whatever"))  # 不抛异常即通过


if __name__ == "__main__":
    unittest.main()
