"""解析层测试：模型输出的"脏数据"能不能被稳稳接住。

这一层是 Agent 稳定性的地基，因此测试要专门覆盖各种畸形输出。
"""

from __future__ import annotations

import unittest

from codeassist.parsers import (
    PLACEHOLDER_PATTERNS,
    detect_language,
    extract_code_blocks,
    extract_json,
    extract_symbol,
    find_placeholders,
    first_code_block,
    list_symbols,
    normalize_language,
    split_prose_and_code,
    strip_code_fence,
    validate_generated_code,
    validate_python,
)


class TestCodeBlocks(unittest.TestCase):
    def test_basic_fence(self):
        blocks = extract_code_blocks("说明文字\n```python\nprint(1)\n```\n后面还有话")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].language, "python")
        self.assertEqual(blocks[0].code, "print(1)")
        self.assertTrue(blocks[0].closed)

    def test_multiple_blocks_and_prose(self):
        text = "先这样：\n```python\na = 1\n```\n然后：\n```javascript\nlet a = 1;\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual([b.language for b in blocks], ["python", "javascript"])
        prose, code = split_prose_and_code(text)
        self.assertIn("先这样", prose)
        self.assertIn("然后", prose)
        self.assertEqual(code, "a = 1")

    def test_unclosed_fence_takes_rest_of_text(self):
        """模型被 max_tokens 截断时围栏不会闭合，这是高频故障。"""
        blocks = extract_code_blocks("```python\ndef f():\n    return 1")
        self.assertEqual(len(blocks), 1)
        self.assertFalse(blocks[0].closed)
        self.assertIn("def f():", blocks[0].code)

    def test_tilde_fence_and_longer_fence(self):
        self.assertEqual(extract_code_blocks("~~~py\nx = 1\n~~~")[0].language, "python")
        # 代码里含 ``` 时，模型会用更长的围栏包裹
        blocks = extract_code_blocks("````markdown\n```python\nx = 1\n```\n````")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].fence, "````")
        self.assertIn("```python", blocks[0].code)

    def test_language_alias_and_extra_info(self):
        self.assertEqual(extract_code_blocks("```js\nvar a;\n```")[0].language, "javascript")
        self.assertEqual(extract_code_blocks('```python title="a.py"\nx = 1\n```')[0].language, "python")

    def test_no_block_returns_empty(self):
        self.assertEqual(extract_code_blocks("纯文字，没有代码"), [])

    def test_strip_code_fence_only_for_pure_code(self):
        self.assertEqual(strip_code_fence("```python\nx = 1\n```"), "x = 1")
        # 带解释的回复不应该被"剥壳"后误当成纯代码
        mixed = "好的，代码如下：\n```python\nx = 1\n```\n以上。"
        self.assertIn("好的", strip_code_fence(mixed))

    def test_first_code_block_default_language(self):
        block = first_code_block("```\nx = 1\n```", default_language="python")
        self.assertIsNotNone(block)
        self.assertEqual(block.language, "python")


class TestJsonExtraction(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json("```\n[1, 2]\n```"), [1, 2])

    def test_json_with_leading_prose(self):
        """最典型的真实场景：模型先说一句"好的，以下是结果"。"""
        text = '好的，以下是分析结果：\n{"summary": "做了一件事", "n": 2}\n希望有帮助！'
        self.assertEqual(extract_json(text)["summary"], "做了一件事")

    def test_braces_inside_strings(self):
        """括号配平扫描必须理解字符串，否则 '}' 会提前结束匹配。"""
        payload = extract_json('前缀 {"code": "if x: print(\\"}\\" )", "ok": true} 后缀')
        self.assertEqual(payload["ok"], True)
        self.assertIn("}", payload["code"])

    def test_trailing_comma_repair(self):
        self.assertEqual(extract_json('{"a": 1, "b": [1, 2,],}'), {"a": 1, "b": [1, 2]})

    def test_single_quote_repair(self):
        self.assertEqual(extract_json("{'a': 1}"), {"a": 1})

    def test_top_level_array(self):
        self.assertEqual(extract_json("结果如下 [1, 2, 3] 完毕"), [1, 2, 3])

    def test_failure_returns_default_not_exception(self):
        """解析失败必须返回默认值，绝不能抛异常打断整条业务流程。"""
        self.assertIsNone(extract_json("完全不是 JSON"))
        self.assertEqual(extract_json("完全不是 JSON", default={}), {})
        self.assertEqual(extract_json("", default={"fallback": True}), {"fallback": True})


class TestLanguageDetection(unittest.TestCase):
    def test_by_filename(self):
        self.assertEqual(detect_language("", filename="main.py"), "python")
        self.assertEqual(detect_language("", filename="App.tsx"), "typescript")

    def test_by_keywords(self):
        self.assertEqual(detect_language("def f():\n    return None"), "python")
        self.assertEqual(detect_language("const a = () => console.log(1)"), "javascript")
        self.assertEqual(detect_language("SELECT * FROM users"), "sql")

    def test_unknown(self):
        self.assertEqual(detect_language("随便写点什么"), "")

    def test_aliases(self):
        self.assertEqual(normalize_language("PY"), "python")
        self.assertEqual(normalize_language("C++"), "cpp")
        self.assertEqual(normalize_language(""), "")
        self.assertEqual(normalize_language(None), "")


class TestValidation(unittest.TestCase):
    def test_valid_python(self):
        self.assertTrue(validate_python("def f():\n    return 1\n"))
        self.assertTrue(validate_python("x = 1"))

    def test_syntax_error_reports_line_and_text(self):
        result = validate_python("def f()\n    return 1")
        self.assertFalse(result.ok)
        self.assertEqual(result.line, 1)
        self.assertIn("第 1 行", result.error)
        self.assertIn("def f()", result.error)

    def test_empty_code(self):
        self.assertFalse(validate_python("   ").ok)

    def test_bom_is_tolerated(self):
        """Windows 上带 BOM 的文件极常见，BOM 不能让校验报"非法字符"。"""
        self.assertTrue(validate_python("\ufeffdef f():\n    return 1\n").ok)

    def test_placeholder_detection(self):
        self.assertIn("...", find_placeholders("def f():\n    ...\n"))
        self.assertIn("TODO: implement", find_placeholders("# TODO: implement"))
        # 正常的 Ellipsis 用法（如类型存根或切片）不应误报
        self.assertEqual(find_placeholders("items = data[1:...]"), [])

    def test_validate_generated_code_multi_language(self):
        problems = validate_generated_code("def f():\n    ...\n", "python")
        self.assertTrue(any("占位符" in p for p in problems))
        # 非 Python 语言不做 AST 校验，只查占位符
        self.assertEqual(validate_generated_code("const a = 1;", "javascript"), [])
        self.assertTrue(validate_generated_code("   ", "python"))

    def test_all_placeholder_patterns_are_reachable(self):
        """防止有人加了模式却写了个永远不会命中的分支。"""
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern == "...":
                sample = "def f():\n    ...\n"
            else:
                sample = f"{pattern}\n"
            self.assertIn(pattern, find_placeholders(sample), f"模式 {pattern!r} 未被检出")


class TestSymbolExtraction(unittest.TestCase):
    """按函数名抽取上下文 —— 让"问某个函数"不必把整个文件塞进提示词。"""

    SAMPLE = '''
import os


def helper():
    return 1


@decorator
def target(a, b=2):
    """这是目标函数。"""
    if a > b:
        return a
    return b


class Service:
    """一个服务类。"""

    def start(self):
        return "started"

    def stop(self):
        return "stopped"


class Other:
    def stop(self):
        return "other stopped"
'''

    def test_extract_top_level_function(self):
        symbol = extract_symbol(self.SAMPLE, "target")
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.kind, "function")
        self.assertIn("def target(a, b=2):", symbol.code)
        self.assertIn("这是目标函数", symbol.code)
        self.assertNotIn("def helper", symbol.code)

    def test_decorator_is_included(self):
        """装饰器属于这个函数，不能被切掉。"""
        symbol = extract_symbol(self.SAMPLE, "target")
        self.assertIn("@decorator", symbol.code)
        self.assertLess(symbol.code.index("@decorator"), symbol.code.index("def target"))

    def test_extract_class_whole(self):
        symbol = extract_symbol(self.SAMPLE, "Service")
        self.assertEqual(symbol.kind, "class")
        self.assertIn("def start", symbol.code)
        self.assertIn("def stop", symbol.code)
        self.assertNotIn("class Other", symbol.code)

    def test_extract_method_with_qualified_name(self):
        symbol = extract_symbol(self.SAMPLE, "Service.start")
        self.assertEqual(symbol.name, "Service.start")
        self.assertIn('return "started"', symbol.code)
        self.assertNotIn("def stop", symbol.code)

    def test_unique_short_name_resolves(self):
        """短名唯一时允许省略类名前缀。"""
        symbol = extract_symbol(self.SAMPLE, "start")
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.name, "Service.start")

    def test_ambiguous_short_name_is_rejected(self):
        """有歧义时必须报错，绝不能瞎选一个 —— 否则用户会拿到错误的答案。"""
        with self.assertRaises(ValueError) as ctx:
            extract_symbol(self.SAMPLE, "stop")
        self.assertIn("不唯一", str(ctx.exception))
        self.assertIn("Service.stop", str(ctx.exception))

    def test_missing_symbol_returns_none(self):
        self.assertIsNone(extract_symbol(self.SAMPLE, "does_not_exist"))

    def test_empty_inputs(self):
        self.assertIsNone(extract_symbol("", "target"))
        self.assertIsNone(extract_symbol(self.SAMPLE, "  "))

    def test_unparsable_code_returns_none(self):
        self.assertIsNone(extract_symbol("def broken(:\n", "broken"))

    def test_line_numbers_match_source(self):
        symbol = extract_symbol(self.SAMPLE, "target")
        lines = self.SAMPLE.split("\n")
        # start_line 指向装饰器（装饰器属于这个函数），因此用"区间包含"来断言
        snippet = "\n".join(lines[symbol.start_line - 1 : symbol.end_line])
        self.assertIn("@decorator", snippet)
        self.assertIn("def target(a, b=2):", snippet)
        self.assertEqual(lines[symbol.end_line - 1].strip(), "return b")

    def test_list_symbols_and_dunder_filtering(self):
        names = list_symbols(self.SAMPLE)
        self.assertIn("target", names)
        self.assertIn("Service.start", names)
        self.assertIn("Service.stop", names)
        self.assertNotIn("Service.__str__", names)
        self.assertEqual(list_symbols("def broken(:"), [])


if __name__ == "__main__":
    unittest.main()
