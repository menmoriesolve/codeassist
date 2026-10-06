"""业务能力层与配置层测试。

覆盖五大能力各自的"契约"：输入什么、返回什么结构、失败时如何降级。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from codeassist.assistant import CodeAssistant
from codeassist.config import PRESETS, Config, load_config
from codeassist.llm import Message, MockLLM
from codeassist.mock import scripted_handler
from codeassist.tools import cleanup_dir, make_scratch_dir

ROOT = Path(__file__).resolve().parent.parent
SIMPLE_CODE = "def add(a, b):\n    return a + b\n"


class AssistantTestCase(unittest.TestCase):
    def make(self, llm=None, **overrides) -> CodeAssistant:
        return CodeAssistant(llm or MockLLM(), Config(provider="mock", **overrides))


class TestGenerate(AssistantTestCase):
    def test_returns_runnable_code(self):
        result = self.make().generate("实现一个判断质数的函数")
        self.assertTrue(result.ok)
        self.assertEqual(result.language, "python")
        self.assertIn("def ", result.code)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.repaired)

    def test_language_normalization(self):
        result = self.make().generate("写个函数", language="PY")
        self.assertEqual(result.language, "python")

    def test_constraints_reach_the_prompt(self):
        assistant = self.make()
        assistant.generate("写个函数", constraints=["不要使用第三方库", "必须带类型标注"])
        prompt = assistant.llm.request_log[-1]["payload"]["messages"][-1]["content"]
        self.assertIn("不要使用第三方库", prompt)
        self.assertIn("必须带类型标注", prompt)

    def test_few_shot_can_be_disabled(self):
        with_few = self.make()
        with_few.generate("写个函数", few_shot=True)
        without_few = self.make()
        without_few.generate("写个函数", few_shot=False)
        self.assertGreater(
            len(with_few.llm.request_log[-1]["payload"]["messages"]),
            len(without_few.llm.request_log[-1]["payload"]["messages"]),
        )

    def test_markdown_rendering(self):
        self.assertIn("```python", self.make().generate("写个判断质数的函数").to_markdown())


class TestSelfRepair(AssistantTestCase):
    """自修复闭环 —— 本项目最重要的工程模式，必须被测试锁住。"""

    BROKEN = "```python\ndef f()\n    return 1\n```"
    FIXED = "```python\ndef f() -> int:\n    return 1\n```"

    def test_repair_loop_fixes_syntax_error(self):
        llm = MockLLM(scripted_handler([self.BROKEN, self.FIXED]))
        result = self.make(llm, repair_attempts=1).generate("写一个返回 1 的函数")
        self.assertTrue(result.repaired)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.ok, result.problems)
        self.assertIn("return 1", result.code)

    def test_repair_prompt_contains_the_actual_error(self):
        """错误信息越具体，模型修得越准 —— 校验器必须把「第几行、什么错」说清楚。"""
        llm = MockLLM(scripted_handler([self.BROKEN, self.FIXED]))
        self.make(llm, repair_attempts=1).generate("写一个返回 1 的函数")
        repair_prompt = llm.request_log[-1]["payload"]["messages"][-1]["content"]
        self.assertIn("语法错误", repair_prompt)
        self.assertIn("第 1 行", repair_prompt)
        self.assertIn("def f()", repair_prompt)

    def test_repair_disabled_reports_problems_instead(self):
        llm = MockLLM(scripted_handler([self.BROKEN]))
        result = self.make(llm, repair_attempts=1).generate("写一个函数", repair=False)
        self.assertFalse(result.repaired)
        self.assertEqual(len(llm.request_log), 1)
        self.assertFalse(result.ok)
        self.assertTrue(result.problems)

    def test_gives_up_after_max_attempts_without_hanging(self):
        """模型修不好时必须有明确终点，不能无限重试烧钱。"""
        llm = MockLLM(scripted_handler([self.BROKEN]))
        result = self.make(llm, repair_attempts=2).generate("写一个函数")
        self.assertEqual(result.attempts, 3)  # 1 次首次 + 2 次修复
        self.assertEqual(len(llm.request_log), 3)
        self.assertFalse(result.ok)

    def test_placeholder_is_treated_as_a_defect(self):
        stubbed = "```python\ndef f():\n    ...\n```"
        llm = MockLLM(scripted_handler([stubbed, self.FIXED]))
        result = self.make(llm, repair_attempts=1).generate("实现 f")
        self.assertTrue(result.repaired)
        self.assertIn("占位符", " ".join(p for p in result.problems) or "占位符")


class TestComplete(AssistantTestCase):
    def test_only_returns_the_increment(self):
        result = self.make().complete("def add(a, b):\n", "", language="python")
        self.assertTrue(result.completion.strip())
        self.assertNotIn("def add(a, b):", result.completion)  # 不能把前缀重抄一遍

    def test_prefix_and_suffix_are_both_passed(self):
        assistant = self.make()
        assistant.complete("def f():\n", "\nprint(f())\n")
        prompt = assistant.llm.request_log[-1]["payload"]["messages"][-1]["content"]
        self.assertIn("<prefix>", prompt)
        self.assertIn("<suffix>", prompt)
        self.assertIn("print(f())", prompt)


class TestExplain(AssistantTestCase):
    def test_structured_fields_are_parsed(self):
        result = self.make().explain(SIMPLE_CODE, language="python")
        self.assertTrue(result.structured)
        self.assertTrue(result.summary)
        self.assertTrue(result.purpose)
        self.assertTrue(result.logic_steps)
        self.assertTrue(result.complexity.get("time"))

    def test_markdown_report_renders_sections(self):
        markdown = self.make().explain(SIMPLE_CODE).to_markdown()
        self.assertIn("一句话概括", markdown)
        self.assertIn("复杂度", markdown)

    def test_detects_real_anti_patterns(self):
        """解释能力要抓得住真实问题：可变默认参数、裸 except。"""
        result = self.make().explain(
            "def load(path, cache={}):\n    try:\n        return cache[path]\n    except:\n        return None\n"
        )
        descriptions = " ".join(issue.get("description", "") for issue in result.issues)
        self.assertIn("可变默认参数", descriptions)
        self.assertIn("裸 except", descriptions)

    def test_falls_back_to_text_when_model_returns_prose(self):
        """模型不按契约输出 JSON 时必须优雅降级，而不是抛异常。"""
        llm = MockLLM(scripted_handler(["这段代码就是简单地把两个数加起来，没有别的。"]))
        result = self.make(llm).explain(SIMPLE_CODE)
        self.assertFalse(result.structured)
        self.assertIn("两个数加起来", result.text)
        self.assertIn("两个数加起来", result.to_markdown())

    def test_fenced_json_is_accepted(self):
        payload = {"summary": "一句话", "purpose": "目的", "logic_steps": [], "key_points": ["k"],
                   "edge_cases": ["e"], "issues": [], "complexity": {"time": "O(1)", "space": "O(1)"}}
        llm = MockLLM(scripted_handler([f"好的：\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"]))
        result = self.make(llm).explain(SIMPLE_CODE)
        self.assertTrue(result.structured)
        self.assertEqual(result.summary, "一句话")

    def test_complexity_as_string_is_normalized(self):
        payload = {"summary": "s", "purpose": "p", "logic_steps": [], "key_points": [],
                   "edge_cases": [], "issues": [], "complexity": "O(n)"}
        llm = MockLLM(scripted_handler([json.dumps(payload)]))
        result = self.make(llm).explain(SIMPLE_CODE)
        self.assertEqual(result.complexity["time"], "O(n)")


class TestComment(AssistantTestCase):
    def test_returns_commented_code(self):
        result = self.make().comment(SIMPLE_CODE, language="python")
        self.assertTrue(result.code.strip())
        self.assertIn("def add", result.code)

    def test_prompt_states_why_not_what(self):
        assistant = self.make()
        assistant.comment(SIMPLE_CODE)
        system = assistant.llm.request_log[-1]["payload"]["messages"][0]["content"]
        self.assertIn("Why", system)


class TestAskAndChat(AssistantTestCase):
    def test_answer_with_context(self):
        result = self.make().ask("这段代码的复杂度是多少？", code=SIMPLE_CODE)
        self.assertTrue(result.answer.strip())
        self.assertTrue(result.used_context)

    def test_context_marked_as_not_used_when_absent(self):
        self.assertFalse(self.make().ask("Python 是什么？").used_context)

    def test_context_appears_in_prompt(self):
        assistant = self.make()
        assistant.ask("问题", code="MARKER_TOKEN = 1")
        prompt = assistant.llm.request_log[-1]["payload"]["messages"][-1]["content"]
        self.assertIn("MARKER_TOKEN", prompt)

    def test_multi_turn_history_accumulates(self):
        assistant = self.make()
        assistant.chat("第一个问题")
        assistant.chat("第二个问题")
        self.assertEqual(len(assistant.history), 4)
        # 第二轮请求必须带上第一轮的历史
        second_round = assistant.llm.request_log[-1]["payload"]["messages"]
        self.assertTrue(any("第一个问题" in (m.get("content") or "") for m in second_round))

    def test_history_is_trimmed(self):
        assistant = self.make()
        assistant.history = [Message.user(f"第{i}条") for i in range(50)]
        assistant.chat("新问题")
        self.assertLessEqual(len(assistant.history), 20)


class TestUsageTracking(AssistantTestCase):
    def test_usage_accumulates_across_calls(self):
        assistant = self.make()
        assistant.explain(SIMPLE_CODE)
        assistant.ask("问题")
        self.assertGreater(assistant.total_usage.total_tokens, 0)
        self.assertGreater(assistant.total_usage.prompt_tokens, 0)


class TestConfigLayer(unittest.TestCase):
    """配置文件相关用例。

    注意：这里刻意不用 tempfile.TemporaryDirectory —— 受限沙箱里系统临时区可能
    不可写，而本项目的 make_scratch_dir 会用"实际写一次"的方式找到真正可写的位置。
    """

    def _scratch(self) -> Path:
        return make_scratch_dir(prefix="cfg-test-", fallback_parent=ROOT)

    def test_defaults(self):
        config = load_config(env={})
        self.assertEqual(config.provider, "mock")
        self.assertEqual(config.max_steps, 8)

    def test_env_overrides_and_type_coercion(self):
        config = load_config(
            env={
                "CODEASSIST_PROVIDER": "openai",
                "CODEASSIST_TEMPERATURE": "0.7",
                "CODEASSIST_MAX_TOKENS": "4096",
                "CODEASSIST_ALLOW_EXECUTE": "false",
                "CODEASSIST_VERBOSE": "1",
            }
        )
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 4096)
        self.assertFalse(config.allow_execute)
        self.assertTrue(config.verbose)

    def test_explicit_argument_beats_env(self):
        config = load_config(env={"CODEASSIST_PROVIDER": "openai"}, provider="deepseek")
        self.assertEqual(config.provider, "deepseek")

    def test_config_file_is_loaded_and_env_wins(self):
        scratch = self._scratch()
        try:
            path = scratch / "cfg.json"
            path.write_text(json.dumps({"provider": "ollama", "temperature": 0.9}), encoding="utf-8")
            config = load_config(config_file=path, env={})
            self.assertEqual(config.provider, "ollama")
            self.assertEqual(config.temperature, 0.9)

            overridden = load_config(config_file=path, env={"CODEASSIST_TEMPERATURE": "0.1"})
            self.assertEqual(overridden.temperature, 0.1)
        finally:
            cleanup_dir(scratch)

    def test_auto_detect_provider_from_env_key(self):
        """用户设了厂商密钥却没指定 provider 时，自动选对厂商，减少"第一次跑不通"。"""
        config = load_config(env={"DEEPSEEK_API_KEY": "sk-xxx"})
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.resolved_api_key, "sk-xxx")

    def test_bad_json_config_raises_with_path(self):
        scratch = self._scratch()
        try:
            path = scratch / "bad.json"
            path.write_text("{不是 json", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(config_file=path, env={})
            self.assertIn("bad.json", str(ctx.exception))
        finally:
            cleanup_dir(scratch)

    def test_config_file_with_utf8_bom_is_accepted(self):
        """Windows 的 PowerShell / 记事本默认会写 UTF-8 BOM，必须能吃下。"""
        scratch = self._scratch()
        try:
            path = scratch / "bom.json"
            path.write_bytes(
                b"\xef\xbb\xbf" + json.dumps({"provider": "ollama", "model": "qwen2.5-coder:7b"}).encode("utf-8")
            )
            config = load_config(config_file=path, env={})
            self.assertEqual(config.provider, "ollama")
            self.assertEqual(config.resolved_model, "qwen2.5-coder:7b")
        finally:
            cleanup_dir(scratch)

    def test_missing_config_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config(config_file="C:/definitely/not/here.json", env={})

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            Config(provider="not-a-provider").validate()
        self.assertIn("可选", str(ctx.exception))

    def test_child_creates_modified_copy(self):
        base = Config(provider="mock")
        child = base.child(temperature=0.9)
        self.assertEqual(base.temperature, 0.2)  # 原对象不受影响
        self.assertEqual(child.temperature, 0.9)

    def test_all_presets_are_complete(self):
        for name, preset in PRESETS.items():
            self.assertEqual(name, preset.name)
            self.assertTrue(preset.base_url)
            self.assertTrue(preset.default_model)
            self.assertTrue(preset.api_key_env)

    def test_provider_preset_resolution(self):
        config = Config(provider="deepseek")
        self.assertEqual(config.resolved_base_url, "https://api.deepseek.com/v1")
        self.assertEqual(config.resolved_model, "deepseek-chat")
        # 显式值优先于预设
        self.assertEqual(config.child(model="x").resolved_model, "x")


if __name__ == "__main__":
    unittest.main()
