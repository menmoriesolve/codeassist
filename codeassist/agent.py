"""Agent 循环层：把 LLM、Prompt、工具三者串成一个能"自主行动"的整体。

**Agent 与"调用一次大模型"的本质区别，就在于这个循环。**

    while 还没拿到最终答案:
        1. 把「任务 + 历史 + 工具清单」发给模型
        2. 模型要么直接回答（结束），要么要求调用某个工具
        3. 我们执行工具，把结果追加回历史
        4. 回到第 1 步 —— 模型这一次能看到工具的真实返回

这个模式叫 ReAct（Reason + Act）。所有"自主 Agent"框架，剥开外壳都是这个循环，
区别只在于：有没有规划、有没有反思、有没有多 Agent 协作、有没有记忆持久化。

本模块额外实现了工程上真正重要的几件事：
    * 最大步数熔断（防止模型原地打转烧钱）
    * 重复调用检测（同样的工具 + 同样的参数再来一次，几乎一定是死循环）
    * 工具失败自愈（错误信息回灌，让模型自己换方案）
    * 全程 trace（可观测性：Agent 出错时你必须能回答"它刚才到底干了什么"）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm import BaseLLM, ChatResponse, LLMError, Message, Usage
from .prompts import render
from .tools import ToolRegistry, ToolResult

#: 事件回调签名：(事件名, 数据)。用于 CLI 实时打印、日志落盘、UI 进度条。
EventHandler = Callable[[str, Dict[str, Any]], None]


# --------------------------------------------------------------------------------------
# 可观测性：一次运行的完整轨迹
# --------------------------------------------------------------------------------------
@dataclass
class TraceStep:
    """轨迹中的一步。"""

    index: int
    kind: str  # llm | tool | nudge | error | final
    title: str = ""
    detail: str = ""
    tool_name: str = ""
    arguments: Any = None
    ok: bool = True
    duration_ms: float = 0.0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 2),
            "tokens": self.tokens,
        }


@dataclass
class AgentTrace:
    steps: List[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.started_at) * 1000

    @property
    def tool_calls(self) -> List[TraceStep]:
        return [s for s in self.steps if s.kind == "tool"]

    def add(self, step: TraceStep) -> TraceStep:
        self.steps.append(step)
        return step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "elapsed_ms": round(self.elapsed_ms, 2),
            "tool_call_count": len(self.tool_calls),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """把轨迹渲染成人类可读的时间线 —— 用来回答"Agent 刚才做了什么"。"""
        lines = ["## 执行轨迹", ""]
        for step in self.steps:
            head = f"**[{step.index}] {step.kind.upper()}** {step.title}"
            if step.duration_ms:
                head += f" · {step.duration_ms:.0f}ms"
            lines.append(head)
            if step.detail:
                quoted = "\n".join(f"  > {ln}" for ln in step.detail.strip().split("\n")[:12])
                lines.append(quoted)
            lines.append("")
        lines.append(
            f"合计 {len(self.tool_calls)} 次工具调用，"
            f"token 用量 {self.usage.total_tokens}，耗时 {self.elapsed_ms:.0f}ms"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 运行结果
# --------------------------------------------------------------------------------------
@dataclass
class AgentResult:
    answer: str = ""
    trace: AgentTrace = field(default_factory=AgentTrace)
    messages: List[Message] = field(default_factory=list)
    stop_reason: str = "final"  # final | max_steps | error
    steps_used: int = 0

    @property
    def ok(self) -> bool:
        return self.stop_reason != "error"

    def __str__(self) -> str:  # pragma: no cover
        return self.answer


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
class Agent:
    """ReAct 风格的通用工具调用 Agent。

    它不关心"代码助手"这个业务 —— 只负责"想 → 做 → 看 → 再想"。
    业务能力（生成/解释/注释）由 CodeAssistant 在上层组装，这就是关注点分离。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[ToolRegistry] = None,
        *,
        system_prompt: Optional[str] = None,
        max_steps: int = 8,
        answer_language: str = "中文",
        on_event: Optional[EventHandler] = None,
        max_repeat_calls: int = 2,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_steps = max(1, max_steps)
        self.answer_language = answer_language
        self.on_event = on_event
        self.max_repeat_calls = max_repeat_calls
        self.system_prompt = system_prompt or render(
            "agent.system",
            tool_descriptions=self.tools.describe(),
            answer_language=answer_language,
        )

    # ---------------------------------------------------------------- 内部工具
    def _emit(self, event: str, **payload: Any) -> None:
        if self.on_event:
            try:
                self.on_event(event, payload)
            except Exception:  # pragma: no cover - 回调不该影响主流程
                pass

    def _run_tool_calls(
        self,
        response: ChatResponse,
        trace: AgentTrace,
        step_index: int,
        call_counts: Dict[str, int],
    ) -> List[Message]:
        """执行模型请求的全部工具调用，返回要追加到历史里的 tool 消息。"""
        results: List[Message] = []
        for call in response.tool_calls:
            signature = f"{call.name}:{call.arguments}"
            call_counts[signature] = call_counts.get(signature, 0) + 1

            self._emit("tool_start", name=call.name, arguments=call.arguments)
            result: ToolResult = self.tools.call(call.name, call.arguments)

            trace.add(
                TraceStep(
                    index=step_index,
                    kind="tool",
                    title=f"调用工具 {call.name}",
                    detail=result.content if result.ok else f"失败：{result.error}",
                    tool_name=call.name,
                    arguments=call.parsed_arguments(),
                    ok=result.ok,
                    duration_ms=result.duration_ms,
                )
            )
            self._emit(
                "tool_end",
                name=call.name,
                ok=result.ok,
                output=result.content,
                error=result.error,
            )
            results.append(
                Message.tool_result(call.id, result.as_message_content(), name=call.name)
            )
        return results

    def _nudge_for_repeats(
        self,
        call_counts: Dict[str, int],
        trace: AgentTrace,
        step_index: int,
        nudged: set,
    ) -> Optional[Message]:
        """检测并打断"原地打转"。

        真实场景里非常常见：模型反复读同一个文件、反复跑同一段代码。
        与其让它把步数耗尽，不如主动提醒它换策略。

        注意 nudged 集合的作用：同一个重复签名**只提醒一次**。
        如果每轮都提醒，纠偏提示本身就会把上下文塞满，反而让模型更混乱。
        """
        repeated = [
            key for key, count in call_counts.items() if count > self.max_repeat_calls and key not in nudged
        ]
        if not repeated:
            return None
        nudged.update(repeated)
        names = sorted({item.split(":", 1)[0] for item in repeated})
        note = (
            f"注意：你已经多次重复调用 {names}，且参数完全相同，结果不会改变。"
            "请换一个思路：要么使用不同的参数/工具获取新信息，"
            "要么基于现有信息直接给出最终答案。"
        )
        trace.add(
            TraceStep(
                index=step_index,
                kind="nudge",
                title="检测到重复工具调用，已注入纠偏提示",
                detail=note,
            )
        )
        self._emit("nudge", message=note)
        return Message.user(note)

    # ---------------------------------------------------------------- 主循环
    def run(
        self,
        task: str,
        *,
        history: Optional[Sequence[Message]] = None,
        extra_context: str = "",
    ) -> AgentResult:
        """执行一次完整任务。

        参数:
            task:          用户任务描述
            history:       之前的对话消息（多轮场景）
            extra_context: 额外上下文（如用户粘贴的代码），会拼进用户消息
        """
        trace = AgentTrace()
        tool_schemas = self.tools.to_openai_schema()

        user_content = task if not extra_context else f"{task}\n\n---\n{extra_context}"
        messages: List[Message] = [Message.system(self.system_prompt)]
        messages.extend(history or [])
        messages.append(Message.user(user_content))

        self._emit("start", task=task, tools=self.tools.names())
        call_counts: Dict[str, int] = {}
        nudged: set = set()
        step_index = 0
        stop_reason = "max_steps"
        answer = ""

        for step_index in range(1, self.max_steps + 1):
            self._emit("llm_start", step=step_index, messages=len(messages))
            started = time.perf_counter()
            try:
                response = self.llm.chat(messages, tools=tool_schemas)
            except LLMError as exc:
                trace.add(
                    TraceStep(index=step_index, kind="error", title="模型调用失败", detail=str(exc), ok=False)
                )
                self._emit("error", message=str(exc))
                return AgentResult(
                    answer=f"抱歉，调用模型时出错：{exc}",
                    trace=trace,
                    messages=messages,
                    stop_reason="error",
                    steps_used=step_index,
                )
            duration = (time.perf_counter() - started) * 1000
            trace.usage = trace.usage + response.usage

            # --- 情况 A：模型不再调用工具 → 这就是最终答案 ---
            if not response.has_tool_calls:
                answer = response.content.strip()
                trace.add(
                    TraceStep(
                        index=step_index,
                        kind="final",
                        title="模型给出最终答案",
                        detail=answer,
                        duration_ms=duration,
                        tokens=response.usage.total_tokens,
                    )
                )
                messages.append(response.as_message())
                stop_reason = "final"
                self._emit("final", answer=answer, step=step_index)
                break

            # --- 情况 B：模型要求调用工具 ---
            trace.add(
                TraceStep(
                    index=step_index,
                    kind="llm",
                    title=f"模型决定调用 {len(response.tool_calls)} 个工具：" 
                    + "、".join(c.name for c in response.tool_calls),
                    detail=(response.content or "").strip(),
                    duration_ms=duration,
                    tokens=response.usage.total_tokens,
                )
            )
            self._emit(
                "llm_end",
                step=step_index,
                content=response.content,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                usage=response.usage.total_tokens,
            )
            messages.append(response.as_message())
            messages.extend(self._run_tool_calls(response, trace, step_index, call_counts))

            nudge = self._nudge_for_repeats(call_counts, trace, step_index, nudged)
            if nudge is not None:
                messages.append(nudge)
        else:
            # --- 情况 C：步数耗尽 → 强制收口 ---
            # 这一步很关键：宁可让模型基于已有信息总结，也不要空手而归。
            self._emit("max_steps", steps=self.max_steps)
            messages.append(
                Message.user(
                    "已达到工具调用步数上限。请**不要再调用任何工具**，"
                    "立即基于上面已经获得的信息给出最终答案；"
                    "如果信息不足以完成任务，请说明已经查到什么、还缺什么。"
                )
            )
            try:
                final = self.llm.chat(messages, tools=None)
                answer = final.content.strip()
                trace.usage = trace.usage + final.usage
                trace.add(
                    TraceStep(
                        index=step_index + 1,
                        kind="final",
                        title="步数用尽，基于已有信息强制总结",
                        detail=answer,
                    )
                )
            except LLMError as exc:  # pragma: no cover - 极少见
                answer = f"步数已用尽，且总结时调用模型失败：{exc}"
                trace.add(TraceStep(index=step_index + 1, kind="error", title="强制总结失败", ok=False))

        if not answer:
            answer = "（Agent 未能产出最终答案）"
            stop_reason = stop_reason if stop_reason != "final" else "error"

        return AgentResult(
            answer=answer,
            trace=trace,
            messages=messages,
            stop_reason=stop_reason,
            steps_used=step_index,
        )

    # ---------------------------------------------------------------- 便捷入口
    def chat(self, history: Sequence[Message], user_input: str) -> AgentResult:
        """多轮对话：把已有历史带进来继续跑。"""
        return self.run(user_input, history=history)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Agent model={self.llm.model} tools={len(self.tools)} max_steps={self.max_steps}>"


# --------------------------------------------------------------------------------------
# 多轮会话：把"一问一答"串成"能连续追问"
# --------------------------------------------------------------------------------------
class AgentSession:
    """可连续追问的 Agent 会话。

    与 `Agent.run()` 的区别：
        Agent.run()   —— 一问一答，跑完即弃。适合脚本、CI、单次任务。
        AgentSession  —— 跨轮保留历史**和工具上下文**。适合人坐在终端前连续追问。

    为什么这件事需要单独一个类，而不是在 CLI 里拼一个 while 循环？
        因为"跨轮保留上下文"有三个必须处理好的细节，放在 CLI 里必然写漏：
          1. 历史里 assistant(tool_calls) 与 tool 消息必须**成对保留**，
             在中间截断会让 API 直接报错（tool_call_id 找不到对应的 tool_calls）。
          2. 历史不能无限增长，否则每一轮都在为陈年旧事付费。
          3. 用户可能随时想清空重来（`/clear`）。
        这三件事都有明确的正确性要求，属于领域逻辑，不属于界面逻辑。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[ToolRegistry] = None,
        *,
        max_steps: int = 8,
        max_history: int = 40,
        answer_language: str = "中文",
        on_event: Optional[EventHandler] = None,
    ):
        self.agent = Agent(
            llm,
            tools,
            max_steps=max_steps,
            answer_language=answer_language,
            on_event=on_event,
        )
        self.max_history = max(4, max_history)
        self.history: List[Message] = []
        self.turns = 0

    # ---------------------------------------------------------------- 会话状态
    def reset(self) -> None:
        """清空上下文，相当于开启一段全新对话。"""
        self.history = []
        self.turns = 0

    @property
    def tool_names(self) -> List[str]:
        return self.agent.tools.names()

    def history_summary(self) -> str:
        """给用户看的会话概况（不打印全部内容，避免刷屏）。"""
        user_turns = sum(1 for m in self.history if m.role == "user")
        return (
            f"已完成 {self.turns} 轮对话，上下文中保留 {len(self.history)} 条消息"
            f"（其中用户提问 {user_turns} 条）"
        )

    # ---------------------------------------------------------------- 历史裁剪
    def _trim(self, messages: List[Message]) -> List[Message]:
        """裁剪历史，且**绝不破坏 assistant(tool_calls) 与 tool 的配对关系**。

        规则：从尾部往前保留 max_history 条，然后向后移动起点，
        直到起点不是一条 `tool` 消息（孤儿 tool 消息会让 API 报参数错误）。
        """
        if len(messages) <= self.max_history:
            return list(messages)
        trimmed = list(messages[-self.max_history :])
        start = 0
        while start < len(trimmed) and trimmed[start].role == "tool":
            start += 1
        if start:
            trimmed = trimmed[start:]
        return trimmed

    # ---------------------------------------------------------------- 一轮对话
    def ask(self, question: str, *, extra_context: str = "") -> AgentResult:
        """提一个问题，完整跑一轮 ReAct，然后把本轮产生的上下文并入历史。"""
        result = self.agent.run(question, history=self.history, extra_context=extra_context)
        self.turns += 1
        # agent.run 返回的 messages = [system, *旧历史, 本轮 user, ...本轮全部中间消息]
        # 去掉 system（下一轮会重新生成，且它已包含最新的工具清单），其余全部继承下来。
        # 保留中间的工具消息是有意为之：这样用户追问"那第 20 行呢"时，
        # 模型还记得刚才读过哪个文件、看到了什么 —— 这正是"连续追问"的价值所在。
        carried = [m for m in result.messages if m.role != "system"]
        self.history = self._trim(carried)
        return result

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentSession turns={self.turns} history={len(self.history)} tools={len(self.agent.tools)}>"
