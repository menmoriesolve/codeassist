"""Agent 循环测试：ReAct 的每个分支都要有覆盖。

Agent 的复杂度几乎全在"异常路径"上：工具失败、模型原地打转、步数耗尽、模型报错。
这些分支不测，就等于没写。
"""

from __future__ import annotations

import unittest

from codeassist.agent import Agent, AgentSession, AgentTrace, TraceStep
from codeassist.config import Config
from codeassist.llm import LLMError, Message, MockLLM, ToolCall, Usage
from codeassist.mock import scripted_handler
from codeassist.tools import ToolRegistry, tool


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(description="读取一个文件的内容", param_docs={"path": "文件路径"})
    def read_file(path: str) -> str:
        if path == "missing.py":
            raise FileNotFoundError(f"文件不存在：{path}")
        return f"内容({path}): print('hello')"

    @registry.tool(description="执行一段代码", param_docs={"code": "代码"})
    def run_python(code: str) -> str:
        return "退出码：0\n--- stdout ---\n42"

    return registry


class RecordingLLM(MockLLM):
    """记录每次调用收到的 messages，用来断言"历史是否被正确回灌"。"""

    def __init__(self, handler):
        super().__init__(handler)
        self.seen_messages = []

    def chat(self, messages, **kwargs):
        self.seen_messages.append(list(messages))
        return super().chat(messages, **kwargs)


class TestAgentLoop(unittest.TestCase):
    def test_direct_answer_without_tools(self):
        """不需要工具时应该直接回答 —— 为用工具而用工具是坏 Agent 的典型症状。"""
        llm = MockLLM(scripted_handler(["直接回答：这是一个纯知识问题"]))
        agent = Agent(llm, build_registry())
        result = agent.run("Python 的 GIL 是什么？")
        self.assertEqual(result.stop_reason, "final")
        self.assertIn("直接回答", result.answer)
        self.assertEqual(len(result.trace.tool_calls), 0)
        self.assertEqual(result.steps_used, 1)

    def test_single_tool_round_trip(self):
        """完整 ReAct：模型决定用工具 → 我们执行 → 结果回灌 → 模型给最终答案。"""
        calls = [[ToolCall("c1", "read_file", '{"path": "app.py"}')], []]
        llm = RecordingLLM(scripted_handler(["app.py 里定义了一个 hello 函数"], tool_calls=calls))
        agent = Agent(llm, build_registry())
        result = agent.run("读取 app.py 并说明它做了什么")

        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(len(result.trace.tool_calls), 1)
        self.assertEqual(result.trace.tool_calls[0].tool_name, "read_file")
        self.assertIn("hello", result.answer)

        # 第二次调用必须能看到工具的真实返回（这是 Agent 与"一次性问答"的分水岭）
        second_call = llm.seen_messages[1]
        tool_messages = [m for m in second_call if m.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("print('hello')", tool_messages[0].content or "")
        self.assertEqual(tool_messages[0].tool_call_id, "c1")

    def test_multiple_tool_calls_in_one_round(self):
        """一轮里并行调用多个工具（比如同时读两个文件）必须都执行。"""
        round_one = [
            ToolCall("c1", "read_file", '{"path": "a.py"}'),
            ToolCall("c2", "read_file", '{"path": "b.py"}'),
        ]
        llm = MockLLM(scripted_handler(["两个文件都读完了"], tool_calls=[round_one, []]))
        result = Agent(llm, build_registry()).run("对比 a.py 和 b.py")
        self.assertEqual(len(result.trace.tool_calls), 2)
        tool_messages = [m for m in result.messages if m.role == "tool"]
        self.assertEqual({m.tool_call_id for m in tool_messages}, {"c1", "c2"})

    def test_tool_failure_is_fed_back_and_agent_recovers(self):
        """工具失败 → 错误信息回灌 → 模型换方案 → 成功。这是 Agent 鲁棒性的核心。"""
        calls = [
            [ToolCall("c1", "read_file", '{"path": "missing.py"}')],
            [ToolCall("c2", "read_file", '{"path": "app.py"}')],
            [],
        ]
        llm = RecordingLLM(scripted_handler(["已经改用正确路径并拿到内容"], tool_calls=calls))
        result = Agent(llm, build_registry()).run("读取 app.py")

        self.assertEqual(result.stop_reason, "final")
        self.assertEqual(len(result.trace.tool_calls), 2)
        self.assertFalse(result.trace.tool_calls[0].ok)
        self.assertTrue(result.trace.tool_calls[1].ok)
        # 失败原因必须出现在给模型的 tool 消息里
        failed = [m for m in llm.seen_messages[1] if m.role == "tool"][0]
        self.assertIn("[工具执行失败]", failed.content or "")
        self.assertIn("文件不存在", failed.content or "")

    def test_unknown_tool_error_is_recoverable(self):
        calls = [[ToolCall("c1", "not_a_tool", "{}")], []]
        llm = MockLLM(scripted_handler(["放弃该工具，直接回答"], tool_calls=calls))
        result = Agent(llm, build_registry()).run("做点什么")
        self.assertFalse(result.trace.tool_calls[0].ok)
        self.assertIn("不存在", result.trace.tool_calls[0].detail)

    def test_repeat_call_detection_injects_correction(self):
        """模型反复调用同一工具同一参数 = 死循环，必须主动打断。"""
        same = [ToolCall("c1", "read_file", '{"path": "app.py"}')]
        llm = MockLLM(scripted_handler(["最终答案"], tool_calls=[same, same, same, same, []]))
        result = Agent(llm, build_registry(), max_repeat_calls=2).run("读取 app.py")

        nudges = [s for s in result.trace.steps if s.kind == "nudge"]
        self.assertEqual(len(nudges), 1, "重复调用应当被检测到并注入纠偏提示")
        nudge_messages = [m for m in result.messages if m.role == "user" and "重复调用" in (m.content or "")]
        self.assertEqual(len(nudge_messages), 1)

    def test_max_steps_forces_a_final_answer(self):
        """步数耗尽不能空手而归：必须基于已有信息强制总结。"""
        looping = [ToolCall("c1", "run_python", '{"code": "print(1)"}')]
        # max_steps=2：前两轮调用工具，第三轮（强制总结）脚本给出纯文本
        llm = RecordingLLM(
            scripted_handler(["基于工具结果的总结"], tool_calls=[looping, looping, []])
        )
        agent = Agent(llm, build_registry(), max_steps=2)
        result = agent.run("无限折腾吧")

        self.assertEqual(result.stop_reason, "max_steps")
        self.assertIn("总结", result.answer)
        self.assertEqual(len(result.trace.tool_calls), 2)
        # 最后一次调用必须**不带工具**，否则模型还会继续调用
        last_call_kwargs = llm.request_log[-1]["payload"]
        self.assertNotIn("tools", last_call_kwargs)
        self.assertIn("不要再调用任何工具", last_call_kwargs["messages"][-1]["content"])

    def test_llm_error_is_reported_not_raised(self):
        def failing_handler(messages, tools):
            raise LLMError("模拟服务不可用")

        agent = Agent(MockLLM(failing_handler), build_registry())
        result = agent.run("随便问点什么")
        self.assertEqual(result.stop_reason, "error")
        self.assertFalse(result.ok)
        self.assertIn("模拟服务不可用", result.answer)
        self.assertTrue(any(s.kind == "error" for s in result.trace.steps))

    def test_system_prompt_lists_available_tools(self):
        """系统提示里必须写明可用工具，否则模型不知道能调用什么。"""
        llm = RecordingLLM(scripted_handler(["好的"]))
        Agent(llm, build_registry()).run("你好")
        system = llm.seen_messages[0][0].content or ""
        self.assertIn("read_file", system)
        self.assertIn("run_python", system)

    def test_task_and_extra_context_both_reach_model(self):
        llm = RecordingLLM(scripted_handler(["好的"]))
        Agent(llm, build_registry()).run("解释这段代码", extra_context="def f(): pass")
        user_message = [m for m in llm.seen_messages[0] if m.role == "user"][-1]
        self.assertIn("解释这段代码", user_message.content or "")
        self.assertIn("def f(): pass", user_message.content or "")

    def test_history_is_prepended(self):
        llm = RecordingLLM(scripted_handler(["好的"]))
        history = [Message.user("上一轮问题"), Message.assistant("上一轮回答")]
        Agent(llm, build_registry()).run("接着说", history=history)
        roles = [m.role for m in llm.seen_messages[0]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class TestTrace(unittest.TestCase):
    def test_trace_serialization(self):
        trace = AgentTrace()
        trace.add(TraceStep(index=1, kind="tool", title="调用", tool_name="read_file", duration_ms=12.5))
        trace.usage = Usage(10, 5, 15)
        payload = trace.to_dict()
        self.assertEqual(payload["tool_call_count"], 1)
        self.assertEqual(payload["usage"]["total_tokens"], 15)
        self.assertEqual(len(trace.tool_calls), 1)

    def test_trace_markdown_is_human_readable(self):
        trace = AgentTrace()
        trace.add(TraceStep(index=1, kind="final", title="给出最终答案", detail="答案是 42"))
        markdown = trace.to_markdown()
        self.assertIn("执行轨迹", markdown)
        self.assertIn("答案是 42", markdown)

    def test_event_callback_receives_lifecycle(self):
        events = []
        llm = MockLLM(
            scripted_handler(
                ["完成"], tool_calls=[[ToolCall("c1", "read_file", '{"path": "app.py"}')], []]
            )
        )
        agent = Agent(llm, build_registry(), on_event=lambda name, payload: events.append(name))
        agent.run("读文件")
        for expected in ("start", "llm_end", "tool_start", "tool_end", "final"):
            self.assertIn(expected, events, f"缺少事件 {expected}")

    def test_callback_exception_does_not_break_run(self):
        def bad_callback(name, payload):
            raise RuntimeError("回调自己炸了")

        llm = MockLLM(scripted_handler(["仍然成功"]))
        result = Agent(llm, build_registry(), on_event=bad_callback).run("你好")
        self.assertEqual(result.stop_reason, "final")


class TestAgentSession(unittest.TestCase):
    """多轮会话：连续追问要能记住上一轮读过什么。"""

    def test_history_carries_across_turns(self):
        calls = [
            [ToolCall("c1", "read_file", '{"path": "app.py"}')], [],  # 第一轮：调工具后回答
            [],                                                       # 第二轮：直接回答
        ]
        llm = RecordingLLM(scripted_handler(["第一轮答案", "第二轮答案"], tool_calls=calls))
        session = AgentSession(llm, build_registry())

        first = session.ask("读一下 app.py")
        second = session.ask("那它返回什么？")

        self.assertEqual(first.answer, "第一轮答案")
        self.assertEqual(second.answer, "第二轮答案")
        self.assertEqual(session.turns, 2)

        # 第二轮必须能看到第一轮读到的文件内容 —— 这正是"连续追问"的价值
        second_call = llm.seen_messages[-1]
        self.assertTrue(any("print('hello')" in (m.content or "") for m in second_call))
        self.assertTrue(any("那它返回什么？" in (m.content or "") for m in second_call))

    def test_system_message_is_not_carried_forward(self):
        llm = RecordingLLM(scripted_handler(["好的"]))
        session = AgentSession(llm, build_registry())
        session.ask("第一问")
        session.ask("第二问")

        carried = [m for m in session.history if m.role != "system"]
        self.assertEqual(len(carried), len(session.history))
        # 但发给模型的请求里必须**恰好有一条** system
        systems = [m for m in llm.seen_messages[-1] if m.role == "system"]
        self.assertEqual(len(systems), 1)

    def test_reset_clears_history(self):
        llm = MockLLM(scripted_handler(["好的"]))
        session = AgentSession(llm, build_registry())
        session.ask("问题一")
        session.ask("问题二")
        self.assertGreater(session.turns, 0)
        session.reset()
        self.assertEqual(session.history, [])
        self.assertEqual(session.turns, 0)

    def test_history_is_trimmed(self):
        llm = MockLLM(scripted_handler(["好的"]))
        session = AgentSession(llm, build_registry(), max_history=4)
        for index in range(6):
            session.ask(f"问题 {index}")
        self.assertLessEqual(len(session.history), 4)

    def test_trimming_never_orphans_tool_messages(self):
        """裁剪历史时若把起点切在 tool 消息上，API 会因找不到对应的 tool_calls 而报错。

        这是多轮 Agent 最容易踩、且只在长会话里才暴露的坑，必须锁死。
        """
        llm = MockLLM(scripted_handler(["好的"]))
        session = AgentSession(llm, build_registry(), max_history=5)
        # 手工构造一段"以 tool 消息开头"的历史：5 条里前 2 条是孤儿 tool 消息
        session.history = [
            Message.tool_result("c1", "结果一", name="read_file"),
            Message.tool_result("c2", "结果二", name="read_file"),
            Message.user("问题"),
            Message.assistant("回答"),
            Message.user("问题二"),
            Message.assistant("回答二"),
        ]
        trimmed = session._trim(session.history)
        self.assertNotEqual(trimmed[0].role, "tool", "裁剪后不能以孤儿 tool 消息开头")

    def test_history_summary(self):
        llm = MockLLM(scripted_handler(["好的"]))
        session = AgentSession(llm, build_registry())
        session.ask("问题")
        summary = session.history_summary()
        self.assertIn("1 轮", summary)
        self.assertIn("上下文中保留", summary)

    def test_tool_names_exposed(self):
        session = AgentSession(MockLLM(scripted_handler(["好的"])), build_registry())
        self.assertEqual(session.tool_names, ["read_file", "run_python"])


class TestUsageAccounting(unittest.TestCase):
    def test_agent_run_usage_can_be_recorded(self):
        """Agent 直连 LLM，不经过 CodeAssistant._chat()，用量必须手动记账，
        否则命令行结尾会打印误导性的 total=0。"""
        from codeassist.assistant import CodeAssistant
        from codeassist.config import Config

        assistant = CodeAssistant(MockLLM(scripted_handler(["好的"])), Config(provider="mock"))
        self.assertEqual(assistant.total_usage.total_tokens, 0)
        result = assistant.agent_run("随便问点什么", tools=build_registry())
        assistant.record_usage(result.trace.usage)
        self.assertGreater(assistant.total_usage.total_tokens, 0)


class TestConfigIntegration(unittest.TestCase):
    def test_max_steps_from_config_is_respected(self):
        looping = [ToolCall("c1", "run_python", '{"code": "print(1)"}')]
        llm = MockLLM(scripted_handler(["总结"], tool_calls=[looping] * 5))
        agent = Agent(llm, build_registry(), max_steps=Config(provider="mock", max_steps=3).max_steps)
        result = agent.run("折腾")
        self.assertEqual(len(result.trace.tool_calls), 3)

    def test_agent_repr(self):
        text = repr(Agent(MockLLM(), build_registry()))
        self.assertIn("tools=2", text)


if __name__ == "__main__":
    unittest.main()
