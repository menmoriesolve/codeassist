"""LLM 调用层测试。

真实网络调用无法在离线环境验证，因此这里测的是**协议层与工程保障**：
请求体结构、错误翻译、重试策略、缓存指纹、Mock 的协议一致性。
这些恰恰是"换掉 SDK 自己写客户端"时最容易出错的部分。
"""

from __future__ import annotations

import unittest

from codeassist.config import Config
from codeassist.llm import (
    BaseLLM,
    CachingLLM,
    ChatResponse,
    LLMError,
    Message,
    MockLLM,
    OpenAICompatibleLLM,
    ToolCall,
    Usage,
    get_llm,
)
from codeassist.mock import heuristic_handler, scripted_handler, scripted_llm


class TestMessageProtocol(unittest.TestCase):
    def test_basic_serialization(self):
        self.assertEqual(Message.system("s").to_dict(), {"role": "system", "content": "s"})
        self.assertEqual(Message.user("u").to_dict(), {"role": "user", "content": "u"})

    def test_tool_result_requires_tool_call_id(self):
        payload = Message.tool_result("call_1", "结果", name="read_file").to_dict()
        self.assertEqual(payload["role"], "tool")
        self.assertEqual(payload["tool_call_id"], "call_1")
        self.assertEqual(payload["name"], "read_file")

    def test_assistant_with_tool_calls_keeps_null_content(self):
        """协议细节：带 tool_calls 的 assistant 消息 content 允许为 null，但字段要在。"""
        call = ToolCall(id="c1", name="read_file", arguments='{"path": "a.py"}')
        payload = Message.assistant("", [call]).to_dict()
        self.assertIn("content", payload)
        self.assertEqual(payload["tool_calls"][0]["type"], "function")
        self.assertEqual(payload["tool_calls"][0]["function"]["name"], "read_file")

    def test_tool_call_argument_parsing_and_fault_tolerance(self):
        self.assertEqual(ToolCall("c", "f", '{"a": 1}').parsed_arguments(), {"a": 1})
        self.assertEqual(ToolCall("c", "f", "").parsed_arguments(), {})
        # 模型偶尔会给出非 JSON 的参数，必须兜住而不是崩
        self.assertIn("_raw", ToolCall("c", "f", "不是json").parsed_arguments())
        # 被围栏包裹的 JSON
        self.assertEqual(ToolCall("c", "f", '```json\n{"a": 2}\n```').parsed_arguments(), {"a": 2})

    def test_usage_addition(self):
        total = Usage(1, 2, 3) + Usage(10, 20, 30)
        self.assertEqual((total.prompt_tokens, total.completion_tokens, total.total_tokens), (11, 22, 33))


class TestPayloadBuilding(unittest.TestCase):
    def test_payload_contains_all_parts(self):
        llm = MockLLM()
        tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}]
        payload = llm.build_payload(
            [Message.system("s"), Message.user("u")], tools=tools, temperature=0.3, max_tokens=100
        )
        self.assertEqual(payload["model"], llm.model)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["tool_choice"], "auto")  # 有工具时必须显式声明选择策略

    def test_payload_omits_tools_when_absent(self):
        payload = MockLLM().build_payload([Message.user("u")])
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)


class TestHttpClientBehaviour(unittest.TestCase):
    """不需要真实网络就能验证的关键工程行为。"""

    def test_endpoint_join(self):
        config = Config(provider="deepseek", api_key="sk-test")
        client = OpenAICompatibleLLM(config)
        self.assertEqual(client.endpoint, "https://api.deepseek.com/v1/chat/completions")
        # base_url 末尾多余斜杠不应产生双斜杠
        client2 = OpenAICompatibleLLM(config.child(base_url="https://example.com/v1/"))
        self.assertEqual(client2.endpoint, "https://example.com/v1/chat/completions")

    def test_missing_key_fails_before_network(self):
        """配置错误必须在发请求之前就报错，并给出可执行的修复建议。"""
        with self.assertRaises(ValueError) as ctx:
            OpenAICompatibleLLM(Config(provider="deepseek", api_key=""))
        message = str(ctx.exception)
        self.assertIn("DEEPSEEK_API_KEY", message)
        self.assertIn("--mock", message)

    def test_local_provider_needs_no_key(self):
        client = OpenAICompatibleLLM(Config(provider="ollama"))
        self.assertEqual(client.api_key, "")

    def test_http_errors_are_translated_to_actions(self):
        cases = {
            401: "认证失败",
            404: "/v1",
            429: "限流",
            500: "重试",
        }
        for status, keyword in cases.items():
            text = OpenAICompatibleLLM._explain_http_error(status, "body")
            self.assertIn(keyword, text, f"HTTP {status} 的提示缺少关键词 {keyword}")

    def test_retryable_status_set(self):
        self.assertIn(429, OpenAICompatibleLLM.RETRYABLE_STATUS)
        self.assertIn(503, OpenAICompatibleLLM.RETRYABLE_STATUS)
        self.assertNotIn(401, OpenAICompatibleLLM.RETRYABLE_STATUS)
        self.assertNotIn(400, OpenAICompatibleLLM.RETRYABLE_STATUS)

    def test_error_type_carries_status_and_body(self):
        error = LLMError("boom", status=429, body="rate limited")
        self.assertEqual(error.status, 429)
        self.assertEqual(error.body, "rate limited")


class TestCaching(unittest.TestCase):
    def test_cache_hit_and_fingerprint_sensitivity(self):
        inner = MockLLM()
        cached = CachingLLM(inner)
        messages = [Message.system("你是工程师"), Message.user("需求描述：\n写个函数\n")]
        first = cached.chat(messages, temperature=0.0)
        second = cached.chat(messages, temperature=0.0)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(cached.hits, 1)
        # 内容变了就不该命中缓存 —— 指纹必须覆盖全部 messages
        cached.chat([Message.system("你是工程师"), Message.user("需求描述：\n写个别的\n")], temperature=0.0)
        self.assertEqual(cached.misses, 2)
        # 采样参数变了也不该命中
        cached.chat(messages, temperature=0.9)
        self.assertEqual(cached.misses, 3)


class TestMockLlmContract(unittest.TestCase):
    def test_mock_returns_tool_calls_when_task_needs_files(self):
        """Mock 必须能返回 tool_calls，否则 Agent 循环就测不到。"""
        from codeassist.prompts import render

        llm = MockLLM()
        system = render("agent.system", tool_descriptions="- read_file", answer_language="中文")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读文件",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        response = llm.chat(
            [Message.system(system), Message.user("读取 app.py 并解释它")], tools=tools
        )
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_mock_fills_usage_for_offline_stats(self):
        response = MockLLM().chat([Message.user("你好")])
        self.assertGreater(response.usage.total_tokens, 0)

    def test_mock_never_returns_code_placeholders(self):
        """回归测试：Mock 生成的代码必须能通过本项目自己的校验器。"""
        from codeassist.parsers import validate_generated_code
        from codeassist.assistant import CodeAssistant

        assistant = CodeAssistant(MockLLM(), Config(provider="mock"))
        for description in (
            "实现一个判断质数的函数",
            "写一个快速排序",
            "二分查找",
            "把列表去重",
            "最大子数组和",
            "实现一个 LRU 缓存",
            "统计文本词频",
            "反转字符串",
            "写一个通用的数据处理函数",  # 命中兜底模板，同样必须是合法可运行代码
        ):
            result = assistant.generate(description, language="python")
            self.assertTrue(result.code.strip(), description)
            self.assertEqual(
                validate_generated_code(result.code, "python"), [], f"{description} 生成的代码未通过校验"
            )

    def test_scripted_llm_follows_the_script(self):
        llm = scripted_llm(["第一条", "第二条"])
        self.assertEqual(llm.chat([Message.user("x")]).content, "第一条")
        self.assertEqual(llm.chat([Message.user("x")]).content, "第二条")
        # 默认最后一条会重复，避免测试因为多跑一轮而莫名失败
        self.assertEqual(llm.chat([Message.user("x")]).content, "第二条")

    def test_scripted_llm_can_raise_when_exhausted(self):
        """需要严格断言"只跑 N 轮"时，可以让剧本用尽即报错。"""
        llm = MockLLM(scripted_handler(["唯一一条"], repeat_last=False))
        self.assertEqual(llm.chat([Message.user("x")]).content, "唯一一条")
        with self.assertRaises(AssertionError):
            llm.chat([Message.user("x")])

    def test_scripted_tool_calls(self):
        calls = [[ToolCall("c1", "read_file", '{"path": "a.py"}')], []]
        llm = MockLLM(scripted_handler(["最终答案"], tool_calls=calls))
        first = llm.chat([Message.user("x")])
        self.assertTrue(first.has_tool_calls)
        second = llm.chat([Message.user("x")])
        self.assertEqual(second.content, "最终答案")

    def test_mock_stream_chunks_reassemble(self):
        llm = MockLLM(heuristic_handler)
        text = "".join(llm.stream([Message.user("你好")]))
        self.assertEqual(text, llm.chat([Message.user("你好")]).content)


class TestFactory(unittest.TestCase):
    def test_mock_flag_overrides_provider(self):
        llm = get_llm(Config(provider="deepseek", api_key="sk-x"), mock=True)
        self.assertIsInstance(llm, MockLLM)

    def test_mock_provider_directly(self):
        self.assertIsInstance(get_llm(Config(provider="mock")), MockLLM)

    def test_real_provider_constructs_http_client(self):
        llm = get_llm(Config(provider="openai", api_key="sk-x", model="gpt-4o-mini"))
        self.assertIsInstance(llm, OpenAICompatibleLLM)
        self.assertEqual(llm.model, "gpt-4o-mini")

    def test_cache_wrapper(self):
        self.assertIsInstance(get_llm(Config(provider="mock"), cache=True), CachingLLM)

    def test_base_llm_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            BaseLLM().chat([Message.user("x")])


class TestChatResponse(unittest.TestCase):
    def test_as_message_roundtrip(self):
        response = ChatResponse(content="hi")
        message = response.as_message()
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "hi")

    def test_has_tool_calls(self):
        self.assertFalse(ChatResponse(content="x").has_tool_calls)
        self.assertTrue(ChatResponse(tool_calls=[ToolCall("c", "f", "{}")]).has_tool_calls)


if __name__ == "__main__":
    unittest.main()
