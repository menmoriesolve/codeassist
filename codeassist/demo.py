"""离线全流程演示：不需要 API Key、不需要网络，把六大能力一次跑完。

运行：
    py -m codeassist demo

演示内容：
    1. 代码生成（Prompt Engineering + Few-shot）
    2. **自修复闭环**：故意让第一次生成出错，看 Agent 如何自我纠正
    3. 代码补全（只输出增量）
    4. 代码解释（结构化 JSON → Markdown 报告）
    5. 注释生成（基于 AST 改写，语法绝不会被破坏）
    6. 代码问答
    7. Agent 工具循环（真实读写文件、执行代码，ReAct 轨迹可见）
    8. 用量与可观测性汇总

为什么演示要用 Mock 模型？
    因为"流程正确"和"模型聪明"是两件独立的事。本项目的价值在于流程与工程骨架，
    用 Mock 演示能让你看清骨架，而不是被模型输出的随机性干扰。
    换成真实模型只需一条命令：py -m codeassist --provider deepseek generate "..."
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from .assistant import CodeAssistant
from .config import Config
from .llm import MockLLM
from .mock import scripted_handler
from .tools import build_default_registry, cleanup_dir, make_scratch_dir

# 演示用的示例代码：故意包含几个典型的"可优化点"，让解释与注释功能有东西可说
SAMPLE_CODE = '''\
import json


def load_config(path, default={}):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except:
        return default


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def summarize(items):
    total = 0
    for item in items:
        for field in item.get("values", []):
            total += field
    return total
'''

# 一段"有语法错误"的代码，用来触发自修复闭环
BROKEN_CODE = '''\
def average(numbers)
    total = sum(numbers)
    return total / len(numbers)
'''

FIXED_CODE = '''\
def average(numbers: list[float]) -> float:
    """计算平均值。

    参数:
        numbers: 数值列表。

    返回:
        平均值。

    异常:
        ValueError: 当列表为空时抛出 —— 空列表没有平均值，
            静默返回 0 会掩盖上游的数据问题。
    """
    if not numbers:
        raise ValueError("numbers 不能为空列表")
    return sum(numbers) / len(numbers)
'''


def _title(text: str) -> None:
    print()
    print("=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _sub(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, 66 - len(text)))


def _show_prompt(llm: Any, note: str = "实际发给模型的消息（节选）") -> None:
    """打印最近一次请求的 messages —— 这是理解 Prompt 工程最快的方式。"""
    if not getattr(llm, "request_log", None):
        return
    payload = llm.request_log[-1]["payload"]
    messages = payload.get("messages", [])
    print(f"\n[教学] {note}：共 {len(messages)} 条")
    for message in messages:
        content = (message.get("content") or "").replace("\n", " ⏎ ")
        role = message.get("role")
        preview = content[:150] + ("…" if len(content) > 150 else "")
        print(f"   {role:<9} | {preview}")
    if payload.get("tools"):
        print(f"   tools     | {[t['function']['name'] for t in payload['tools']]}")


def demo_generate() -> None:
    _title("1. 代码生成：自然语言 → 可运行代码（Prompt Engineering + Few-shot）")
    llm = MockLLM()
    assistant = CodeAssistant(llm, Config(provider="mock"))
    result = assistant.generate("实现一个判断质数的函数，并给出 30 以内的所有质数", language="python")
    print(result.to_markdown())
    print(f"\n本地校验：{'通过 ✔' if result.ok else '未通过 ✘ ' + str(result.problems)}")
    _show_prompt(llm)
    print(
        "\n[要点] 系统提示负责'角色 + 硬性约束 + 输出格式'，few-shot 负责'用示例固定风格'，\n"
        "       两者叠加才能稳定拿到可直接使用的代码，而不是一段带解释的文本。"
    )


def demo_self_repair() -> None:
    _title("2. 自修复闭环：本地校验发现语法错误 → 自动让模型重写（Agent 思想）")
    print("场景：模型第一次返回的代码缺少冒号，本地语法校验抓到了它。")
    handler = scripted_handler(
        [
            f"```python\n{BROKEN_CODE}```",
            f"```python\n{FIXED_CODE}```",
            "修复说明：补上了函数定义缺失的冒号，并增加了空列表校验与类型标注。",
        ]
    )
    llm = MockLLM(handler)
    config = Config(provider="mock", repair_attempts=2)
    assistant = CodeAssistant(llm, config)

    _sub("第一次生成的内容（有 bug）")
    print(textwrap.indent(BROKEN_CODE.strip(), "    "))

    result = assistant.generate("写一个计算平均值的函数", language="python")

    _sub("Agent 的自动修复过程")
    print(f"  共调用模型 {len(llm.request_log)} 次（1 次生成 + {len(llm.request_log) - 1} 次修复）")
    print(f"  是否触发修复：{'是' if result.repaired else '否'}")
    print(f"  最终校验：{'通过 ✔' if result.ok else '仍有问题 ✘ ' + str(result.problems)}")

    _sub("最终交付的代码")
    print(result.code)

    _sub("回灌给模型的错误信息（模型正是靠它自我纠正）")
    repair_payload = llm.request_log[-1]["payload"]["messages"][-1]["content"]
    print(textwrap.indent(repair_payload[:600], "    "))
    print(
        "\n[要点] 这是本项目最重要的工程模式：**不要指望模型一次就对，\n"
        "       而要给它装一个'发现错误并重试'的闭环**。校验器可以是语法检查、单元测试、\n"
        "       lint、类型检查 —— 只要能给出可读的错误信息，模型就有很大概率自己修好。"
    )


def demo_complete() -> None:
    _title("3. 代码补全：只输出缺失的片段（Fill-in-the-Middle）")
    llm = MockLLM()
    assistant = CodeAssistant(llm, Config(provider="mock"))
    prefix = "def calculate_total(items: list[dict]) -> float:\n"
    print("光标前的代码：")
    print(textwrap.indent(prefix.rstrip(), "    "))
    result = assistant.complete(prefix, "", language="python")
    print("\n补全结果（只含新增部分）：")
    print(textwrap.indent(result.completion.rstrip(), "    "))
    print(
        "\n[要点] 补全的关键是把 prefix / suffix 用标签明确分隔，并**严格约束只输出增量**。\n"
        "       否则模型会把整段代码重抄一遍，你还得自己 diff。"
    )


def demo_explain() -> None:
    _title("4. 代码解释：代码理解 + 结构化输出")
    llm = MockLLM()
    assistant = CodeAssistant(llm, Config(provider="mock"))
    result = assistant.explain(SAMPLE_CODE, language="python", level="detailed")
    print(result.to_markdown())
    print(f"\n结构化解析：{'成功（字段完整，可被程序消费）' if result.structured else '失败，已降级为纯文本'}")
    _sub("同样的数据也能被程序直接消费（JSON）")
    print(f"   summary      = {result.summary!r}")
    print(f"   issues 数量  = {len(result.issues)}")
    print(f"   complexity   = {result.complexity}")


def demo_comment() -> None:
    _title("5. 注释生成：代码理解 + 自然语言生成")
    llm = MockLLM()
    assistant = CodeAssistant(llm, Config(provider="mock"))
    result = assistant.comment(SAMPLE_CODE, language="python")
    print(result.to_markdown())
    print(
        "\n[要点] 注释的目标是解释 **Why**，而不是复述代码在做什么。\n"
        "       提示词里必须明确这一点，否则模型会生成大量 'i += 1  # i 加一' 式的噪音。"
    )


def demo_qa() -> None:
    _title("6. 代码问答：把代码作为上下文（RAG 的最小形态）")
    llm = MockLLM()
    assistant = CodeAssistant(llm, Config(provider="mock"))
    answer = assistant.ask("summarize 函数里嵌套循环的复杂度是多少？如何优化？", code=SAMPLE_CODE)
    print(answer.answer)
    _sub("多轮对话：历史会自动带上")
    assistant.chat("那如果 items 很大，应该怎么改？", code=SAMPLE_CODE)
    print(f"  当前会话历史长度：{len(assistant.history)} 条消息（user/assistant 成对累积）")
    print(
        "\n[要点] 问答与 Agent 的分界线：**如果答案取决于你没提供的信息，就需要工具去取**\n"
        "       （见下一节的 Agent 模式），而不是让模型猜。"
    )


def demo_agent() -> int:
    _title("7. Agent 工具循环：真实的读写文件与执行代码（ReAct）")

    workspace = make_scratch_dir(prefix="codeassist-demo-", fallback_parent=Path.cwd())
    try:
        (workspace / "README.md").write_text(
            "# 示例项目\n\n这是一个用于演示 Agent 工具调用的最小项目。\n"
            "它包含一个计算模块和一个数据处理脚本。\n",
            encoding="utf-8",
        )
        (workspace / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n",
            encoding="utf-8",
        )
        (workspace / "data").mkdir()
        (workspace / "data" / "sample.json").write_text('{"values": [1, 2, 3]}', encoding="utf-8")

        print(f"临时工作区：{workspace}")
        print("目录内容：README.md、calc.py、data/sample.json\n")

        registry = build_default_registry(workspace, allow_execute=True)
        print(f"装配工具：{registry.names()}\n")
        _sub("工具声明（模型看到的 JSON Schema 片段）")
        schema = registry.to_openai_schema()
        first = schema[0]["function"]
        print(f"   name        : {first['name']}")
        print(f"   description : {first['description'][:80]}…")
        print(f"   parameters  : {list(first['parameters'].get('properties', {}))}")

        config = Config(provider="mock", workspace=str(workspace), max_steps=6)
        llm = MockLLM()
        assistant = CodeAssistant(llm, config, tools=registry)

        task = "列出目录结构，然后读取 README.md 的前 40 行，总结这个项目的用途"
        _sub(f"任务：{task}")

        def on_event(event: str, payload: dict) -> None:
            if event == "llm_end" and payload.get("tool_calls"):
                names = "、".join(call["name"] for call in payload["tool_calls"])
                print(f"  [第 {payload['step']} 轮] 模型决定调用：{names}")
            elif event == "tool_end":
                status = "成功" if payload["ok"] else f"失败（{payload['error'][:80]}）"
                print(f"          工具 {payload['name']} 执行{status}")

        result = assistant.build_agent(registry, on_event=on_event).run(task)
        print()
        print(result.answer)
        _sub("执行轨迹（可观测性：出问题时能回答「Agent 到底干了什么」）")
        for step in result.trace.steps:
            detail = (step.detail or "").strip().split("\n")[0][:80]
            print(f"  [{step.index}] {step.kind:<7} {step.title[:46]:<48} {detail}")
    finally:
        # 沙箱环境下可能删不掉临时目录，清理失败不影响演示
        cleanup_dir(workspace)

    print(
        "\n[要点] 工具集是 Agent 的能力边界：模型再聪明，没有工具也只能'说'。\n"
        "       而工具的安全性（路径围栏、执行超时、输出截断）必须由**代码**保证，\n"
        "       绝不能指望模型自觉。"
    )
    return 0


def run_demo() -> int:
    print()
    print("╔" + "═" * 76 + "╗")
    print("║" + " 代码助手 Agent —— 离线全流程演示 ".ljust(66) + "║")
    print("║" + " 使用内置 Mock 模型：无需 API Key、无需网络 ".ljust(60) + "║")
    print("╚" + "═" * 76 + "╝")

    demo_generate()
    demo_self_repair()
    demo_complete()
    demo_explain()
    demo_comment()
    demo_qa()
    demo_agent()

    _title("8. 演示结束：下一步怎么用真实模型")
    print(
        textwrap.dedent(
            """
            设置密钥后，把上面的命令换成真实调用即可（代码一行都不用改）：

              # Windows PowerShell
              $env:DEEPSEEK_API_KEY = "sk-xxxx"
              py -m codeassist --provider deepseek generate "实现一个线程安全的 LRU 缓存"

              # 或者用配置文件（推荐，密钥不进命令行历史）
              # .codeassist.json
              # { "provider": "deepseek", "model": "deepseek-chat", "temperature": 0.2 }

            查看自己每次到底发了什么给模型（最好的学习方式）：

              py -m codeassist --show-prompt --provider deepseek explain --file codeassist/parsers.py

            只想看请求、不花钱：

              py -m codeassist --dry-run generate "写一个二分查找"
            """
        ).strip()
    )
    print()
    print("文档：README.md（使用方法）、Design.md（架构与设计决策）")
    return 0
