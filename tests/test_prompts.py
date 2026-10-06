"""Prompt 层测试：模板渲染的严格性、消息装配顺序、模板库完整性。

Prompt 是代码资产，那就该像代码一样被测试。
"""

from __future__ import annotations

import unittest

from codeassist.llm import Message
from codeassist.prompts import (
    CODE_GEN_FEWSHOT,
    TEMPLATES,
    MissingVariable,
    PromptTemplate,
    build_messages,
    render,
)


class TestTemplateEngine(unittest.TestCase):
    def test_simple_render(self):
        template = PromptTemplate("你好 {{name}}，你是 {{role}}。", name="t")
        self.assertEqual(template.render(name="小明", role="工程师"), "你好 小明，你是 工程师。")
        self.assertEqual(template.variables, ["name", "role"])

    def test_missing_variable_fails_loudly(self):
        """缺失变量必须报错：悄悄渲染成空字符串会让模型输出莫名其妙的结果。"""
        template = PromptTemplate("{{a}} 和 {{b}}", name="t")
        with self.assertRaises(MissingVariable) as ctx:
            template.render(a="1")
        self.assertIn("b", str(ctx.exception))
        self.assertIn("['a', 'b']", str(ctx.exception))

    def test_lenient_mode_keeps_placeholder(self):
        template = PromptTemplate("{{a}}", name="t")
        self.assertEqual(template.render(strict=False), "{{a}}")

    def test_braces_in_code_are_not_special(self):
        """代码里的 {} 不能触发格式化 —— 这正是弃用 str.format 的原因。"""
        template = PromptTemplate("代码：{{code}}", name="t")
        rendered = template.render(code="if x:\n    d = {'a': 1}")
        self.assertEqual(rendered, "代码：if x:\n    d = {'a': 1}")

    def test_list_values_are_joined(self):
        template = PromptTemplate("{{items}}", name="t")
        self.assertEqual(template.render(items=["a", "b"]), "a\nb")

    def test_none_becomes_empty(self):
        self.assertEqual(PromptTemplate("x{{v}}y", name="t").render(v=None), "xy")

    def test_template_variables_are_declared(self):
        self.assertEqual(PromptTemplate("{{ x }}{{y}}", name="t").variables, ["x", "y"])


class TestTemplateLibrary(unittest.TestCase):
    def test_all_templates_registered_and_described(self):
        self.assertGreaterEqual(len(TEMPLATES), 13)
        for name, template in TEMPLATES.items():
            self.assertTrue(template.description, f"{name} 缺少用途说明")
            self.assertEqual(name, template.name)

    def test_render_injects_shared_rules(self):
        """可复用的规则片段应自动注入，避免在每个模板里复制粘贴。"""
        system = render("code_gen.system", language="python")
        self.assertIn("只输出代码", system)  # 来自 CODE_ONLY_RULES
        self.assertIn("python", system)

    def test_render_unknown_template(self):
        with self.assertRaises(KeyError):
            render("not.exist")

    def test_every_template_can_render_with_dummy_values(self):
        """防止模板里写了变量却忘了在调用处传 —— 这类 bug 只有渲染时才暴露。"""
        for name, template in TEMPLATES.items():
            values = {variable: "X" for variable in template.variables}
            rendered = render(name, **values)
            self.assertTrue(rendered.strip(), f"{name} 渲染结果为空")

    def test_explain_contract_declares_required_fields(self):
        """结构化输出的契约字段必须写在提示词里，否则模型只能靠猜。"""
        system = render("code_explain.system")
        for field in ("summary", "purpose", "logic_steps", "key_points", "edge_cases", "issues", "complexity"):
            self.assertIn(field, system, f"解释模板缺少字段 {field}")

    def test_comment_template_states_why_not_what(self):
        system = render("code_comment.system", language="python", comment_language="中文", extra_note="")
        self.assertIn("Why", system)
        self.assertIn("What", system)

    def test_complete_template_isolates_prefix_and_suffix(self):
        user = render("code_complete.user", language="python", file_context="", prefix="def f():", suffix="")
        self.assertIn("<prefix>", user)
        self.assertIn("<suffix>", user)


class TestMessageAssembly(unittest.TestCase):
    def test_order_is_system_fewshot_history_user(self):
        """顺序很重要：system 定基调，few-shot 定格式，问题放最后（注意力最强）。"""
        messages = build_messages(
            system="S",
            few_shot=[Message.user("few"), Message.assistant("shot")],
            history=[Message.user("旧问题"), Message.assistant("旧回答")],
            user="现在的问题",
        )
        roles = [m.role for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[0].content, "S")
        self.assertEqual(messages[-1].content, "现在的问题")

    def test_optional_parts_omitted(self):
        self.assertEqual([m.role for m in build_messages(user="只有问题")], ["user"])
        self.assertEqual(build_messages(), [])

    def test_few_shot_examples_are_well_formed(self):
        """Few-shot 示例必须成对出现且非空，否则会教坏模型。"""
        self.assertGreaterEqual(len(CODE_GEN_FEWSHOT), 2)
        self.assertEqual(CODE_GEN_FEWSHOT[0].role, "user")
        self.assertEqual(CODE_GEN_FEWSHOT[1].role, "assistant")
        for message in CODE_GEN_FEWSHOT:
            self.assertTrue((message.content or "").strip())

    def test_few_shot_fixes_output_format(self):
        """示例的价值在于"演示格式"：assistant 示例必须是带语言标识的代码块。"""
        self.assertIn("```python", CODE_GEN_FEWSHOT[1].content or "")


if __name__ == "__main__":
    unittest.main()
