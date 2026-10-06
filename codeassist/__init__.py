"""codeassist —— 一个极简但完整的「代码助手 Agent」参考实现。

设计目标（也是本项目的教学主线）：
    1. 零第三方依赖：只用 Python 标准库，方便阅读和移植，把所有"魔法"都摊开。
    2. 分层清晰：LLM 调用 / Prompt 工程 / 解析 / 工具 / Agent 循环 / 业务能力，各层可单独替换。
    3. 离线可跑：内置 MockProvider，没有 API Key、没有网络也能完整演示与测试。

对外主入口：
    CodeAssistant  —— 五大业务能力（生成 / 补全 / 解释 / 注释 / 问答）
    Agent          —— 带工具调用的通用 Agent 循环
    get_llm        —— 按配置拿到一个 LLM 客户端
"""

from .config import Config, ProviderPreset, load_config
from .llm import ChatResponse, LLMError, Message, ToolCall, get_llm
from .agent import Agent, AgentResult, AgentSession, AgentTrace, TraceStep
from .assistant import (
    CodeAssistant,
    CompletionResult,
    ExplanationResult,
    GeneratedCode,
    QAAnswer,
)
from .tools import Tool, ToolRegistry, tool

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ProviderPreset",
    "load_config",
    "Message",
    "ChatResponse",
    "ToolCall",
    "LLMError",
    "get_llm",
    "Agent",
    "AgentResult",
    "AgentSession",
    "AgentTrace",
    "TraceStep",
    "CodeAssistant",
    "GeneratedCode",
    "CompletionResult",
    "ExplanationResult",
    "QAAnswer",
    "Tool",
    "ToolRegistry",
    "tool",
    "__version__",
]
