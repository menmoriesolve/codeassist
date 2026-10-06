"""业务能力层：把底层零件组装成「代码助手」的五大核心功能。

    1. generate()  自然语言 → 代码    （Prompt Engineering + Few-shot）
    2. complete()  上下文   → 补全    （Fill-in-the-Middle）
    3. explain()   代码     → 结构化解释（代码理解 + 结构化输出）
    4. comment()   代码     → 带注释代码（代码理解 + 自然语言生成）
    5. ask()       问题     → 回答    （代码问答 / RAG 的最小形态）

外加一个 generate 的"自修复闭环"：生成 → 校验 → 把错误喂回去 → 再生成。
这是本项目中最能体现 Agent 思想的工程模式：**让程序替你发现错误，并驱动模型自我修正**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .agent import Agent, AgentResult, AgentSession
from .config import Config
from .llm import BaseLLM, ChatResponse, Message, Usage
from .parsers import (
    extract_code_blocks,
    extract_json,
    first_code_block,
    normalize_language,
    validate_generated_code,
)
from .prompts import (
    CODE_COMMENT_FEWSHOT,
    CODE_GEN_FEWSHOT,
    build_messages,
    render,
)
from .tools import ToolRegistry

MAX_HISTORY_MESSAGES = 20


# ======================================================================================
# 结果对象：结构化的返回值比裸字符串好用得多
# ======================================================================================
@dataclass
class GeneratedCode:
    code: str = ""
    language: str = "python"
    explanation: str = ""
    problems: List[str] = field(default_factory=list)
    attempts: int = 1
    repaired: bool = False
    usage: Usage = field(default_factory=Usage)
    raw: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.code.strip()) and not self.problems

    def to_markdown(self) -> str:
        parts = [f"```{self.language}", self.code, "```"]
        if self.explanation:
            parts += ["", "**说明**", "", self.explanation]
        if self.problems:
            parts += ["", "**校验未通过的问题**", ""] + [f"- {p}" for p in self.problems]
        if self.repaired:
            parts += ["", f"> 已自动修复（共尝试 {self.attempts} 次生成）。"]
        return "\n".join(parts)


@dataclass
class CompletionResult:
    completion: str = ""
    language: str = "python"
    usage: Usage = field(default_factory=Usage)
    raw: str = ""

    def to_markdown(self) -> str:
        return f"```{self.language}\n{self.completion}\n```"


@dataclass
class ExplanationResult:
    summary: str = ""
    purpose: str = ""
    logic_steps: List[Dict[str, Any]] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    complexity: Dict[str, str] = field(default_factory=dict)
    structured: bool = False
    text: str = ""
    usage: Usage = field(default_factory=Usage)

    def to_markdown(self) -> str:
        """把结构化结果渲染成可读的报告。

        结构化输出的最大价值就在这里：同一份数据既能给程序消费（做 CI 卡点、
        生成文档、写入知识库），又能渲染成人看的报告。裸文本做不到这一点。
        """
        if not self.structured:
            return self.text.strip() or "（模型未返回可解析的结构化结果）"

        lines: List[str] = []
        if self.summary:
            lines += ["### 一句话概括", "", self.summary, ""]
        if self.purpose:
            lines += ["### 业务目的", "", self.purpose, ""]
        if self.logic_steps:
            lines += ["### 执行流程", ""]
            for step in self.logic_steps:
                number = step.get("step", "-")
                title = step.get("title", "")
                detail = step.get("detail", "")
                lines.append(f"{number}. **{title}** —— {detail}")
            lines.append("")
        if self.key_points:
            lines += ["### 关键设计点", ""] + [f"- {item}" for item in self.key_points] + [""]
        if self.edge_cases:
            lines += ["### 边界与陷阱", ""] + [f"- {item}" for item in self.edge_cases] + [""]
        if self.issues:
            lines += ["### 潜在问题", ""]
            for issue in self.issues:
                severity = str(issue.get("severity", "?")).upper()
                lines.append(f"- **[{severity}]** {issue.get('description', '')}")
                if issue.get("suggestion"):
                    lines.append(f"  - 建议：{issue['suggestion']}")
            lines.append("")
        if self.complexity:
            lines += ["### 复杂度", ""]
            if self.complexity.get("time"):
                lines.append(f"- 时间：{self.complexity['time']}")
            if self.complexity.get("space"):
                lines.append(f"- 空间：{self.complexity['space']}")
        return "\n".join(lines).strip()


@dataclass
class CommentedCode:
    code: str = ""
    notes: str = ""
    language: str = "python"
    usage: Usage = field(default_factory=Usage)

    def to_markdown(self) -> str:
        parts = [f"```{self.language}", self.code, "```"]
        if self.notes:
            parts += ["", "**改动说明**", "", self.notes]
        return "\n".join(parts)


@dataclass
class QAAnswer:
    answer: str = ""
    used_context: bool = False
    usage: Usage = field(default_factory=Usage)

    def __str__(self) -> str:  # pragma: no cover
        return self.answer


# ======================================================================================
# CodeAssistant
# ======================================================================================
class CodeAssistant:
    """代码助手：面向具体任务的封装。

    与 Agent 的分工：
        Agent          —— 通用循环，解决"我需要先看看文件才知道怎么答"这类问题；
        CodeAssistant  —— 面向确定任务的"单人回合"，流程固定、成本可控、输出结构化。
    真实产品里两者常常并存：简单请求走固定流程（快、便宜、可控），
    复杂请求才升级到 Agent 循环（灵活、但贵且慢）。这个"路由"本身就是重要设计。
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        config: Optional[Config] = None,
        *,
        tools: Optional[ToolRegistry] = None,
    ):
        self.config = config or Config()
        if llm is None:
            from .llm import get_llm

            llm = get_llm(self.config)
        self.llm = llm
        self.tools = tools
        #: 会话累计用量，方便在 CLI 退出时打印"这次一共花了多少 token"
        self.total_usage = Usage()
        self.history: List[Message] = []

    # ---------------------------------------------------------------- 用量记账
    def record_usage(self, usage: Usage) -> Usage:
        """把一次"绕过本类直连 LLM"的调用用量记进会话总账。

        Agent / AgentSession 是直接持有 llm 的，不经过 CodeAssistant._chat()，
        所以它们的 token 用量不会自动计入 total_usage。
        如果不显式记账，命令行结尾就会打印出 "total=0" 这种误导性的数字 ——
        成本可见性是 Agent 工程的基本要求，宁可多一行代码也要记准。
        """
        self.total_usage = self.total_usage + usage
        return self.total_usage

    # ---------------------------------------------------------------- 内部
    def _chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        response = self.llm.chat(
            messages,
            tools=tools,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens,
        )
        self.total_usage = self.total_usage + response.usage
        return response

    def _trim_history(self, history: Sequence[Message]) -> List[Message]:
        """只保留最近若干条消息。

        上下文是有限资源：越长的历史越贵、越慢，而且模型对中间部分的注意力会衰减
        （"lost in the middle"）。生产系统应在此处做摘要压缩，这里用最简单的截断。
        """
        items = list(history)
        if len(items) <= MAX_HISTORY_MESSAGES:
            return items
        return items[-MAX_HISTORY_MESSAGES:]

    # ---------------------------------------------------------------- 1. 生成
    def generate(
        self,
        description: str,
        *,
        language: str = "python",
        constraints: Optional[Sequence[str]] = None,
        few_shot: bool = True,
        temperature: Optional[float] = None,
        repair: bool = True,
    ) -> GeneratedCode:
        """根据自然语言描述生成代码。

        这里体现了三个 Prompt 工程要点：
            * 系统提示定义"角色 + 硬性约束 + 输出格式"；
            * few-shot 用示例固定代码风格与结构；
            * 生成后立刻做**本地校验**，不合格就进自修复闭环。
        """
        language = normalize_language(language) or "python"
        extra = ""
        if constraints:
            extra = "附加要求：\n" + "\n".join(f"- {item}" for item in constraints)

        system = render("code_gen.system", language=language)
        user = render(
            "code_gen.user", description=description, language=language, extra_constraints=extra
        )
        messages = build_messages(
            system=system,
            few_shot=CODE_GEN_FEWSHOT if few_shot else None,
            user=user,
        )

        response = self._chat(messages, temperature=temperature)
        block = first_code_block(response.content, default_language=language)
        code = block.code if block else response.content.strip()
        prose = self._prose_outside_code(response.content)

        attempts = 1
        repaired = False
        # 校验永远执行：它很便宜，而且能让用户明确知道"这份代码没通过本地检查"。
        # repair 开关只决定"要不要把错误回灌给模型重写"（那一步才花钱、花时间）。
        problems = validate_generated_code(code, language)

        # ---- 自修复闭环：把校验器的错误回灌给模型，让它自己改 ----
        for _ in range(max(0, self.config.repair_attempts) if repair else 0):
            if not problems:
                break
            attempts += 1
            repaired = True
            repair_messages = build_messages(
                system=render("repair.system"),
                user=render(
                    "repair.user",
                    description=description,
                    language=language,
                    code=code,
                    errors="\n".join(f"- {item}" for item in problems),
                ),
            )
            repair_response = self._chat(repair_messages, temperature=temperature)
            repaired_block = first_code_block(repair_response.content, default_language=language)
            if repaired_block and repaired_block.code.strip():
                code = repaired_block.code
                prose = self._prose_outside_code(repair_response.content) or prose
            problems = validate_generated_code(code, language)

        return GeneratedCode(
            code=code,
            language=language,
            explanation=prose,
            problems=problems,
            attempts=attempts,
            repaired=repaired,
            usage=self.total_usage,
            raw=response.content,
        )

    @staticmethod
    def _prose_outside_code(text: str) -> str:
        """抽取代码块之外的说明文字（用于"生成说明"）。"""
        from .parsers import split_prose_and_code

        prose, _ = split_prose_and_code(text)
        return prose.strip()

    # ---------------------------------------------------------------- 2. 补全
    def complete(
        self,
        prefix: str,
        suffix: str = "",
        *,
        language: str = "python",
        file_context: str = "",
        temperature: Optional[float] = None,
    ) -> CompletionResult:
        """代码补全。

        关键设计：**把"补全的部分"和"已有的部分"用标签明确隔开**。
        如果不告诉模型"你只需要输出中间那段"，它就会把整段代码重抄一遍，
        下游还得写代码去 diff —— 这就是补全类任务最典型的坑。
        """
        language = normalize_language(language) or "python"
        context_block = f"文件背景：{file_context}\n" if file_context else ""
        messages = build_messages(
            system=render("code_complete.system"),
            user=render(
                "code_complete.user",
                language=language,
                file_context=context_block,
                prefix=prefix,
                suffix=suffix,
            ),
        )
        response = self._chat(messages, temperature=0.0 if temperature is None else temperature)

        completion = response.content
        # 容错：如果模型仍然用了代码块，剥掉围栏
        blocks = extract_code_blocks(completion)
        if blocks and not completion.strip().split("\n")[0].strip().startswith((" ", "\t")):
            completion = blocks[0].code

        return CompletionResult(
            completion=completion.rstrip("\n") + "\n",
            language=language,
            usage=self.total_usage,
            raw=response.content,
        )

    # ---------------------------------------------------------------- 3. 解释
    def explain(
        self,
        code: str,
        *,
        language: str = "",
        level: str = "normal",
        temperature: Optional[float] = None,
    ) -> ExplanationResult:
        """解释代码逻辑，返回结构化结果。

        这里演示了"结构化输出"的完整落地方式：
            1. 在系统提示里给出精确的 JSON Schema（字段名、类型、语义）；
            2. 用宽容解析器接住模型的格式漂移；
            3. 解析失败时优雅降级为纯文本，而不是抛异常。
        """
        language = normalize_language(language) or self._guess_language(code)
        messages = build_messages(
            system=render("code_explain.system"),
            user=render("code_explain.user", language=language, level=level, code=code),
        )
        response = self._chat(messages, temperature=temperature)
        payload = extract_json(response.content, default=None)

        if not isinstance(payload, dict):
            return ExplanationResult(
                structured=False,
                text=response.content,
                summary=response.content.strip().split("\n")[0][:120],
                usage=self.total_usage,
            )

        complexity = payload.get("complexity") or {}
        if not isinstance(complexity, dict):
            complexity = {"time": str(complexity), "space": ""}

        return ExplanationResult(
            summary=str(payload.get("summary") or ""),
            purpose=str(payload.get("purpose") or ""),
            logic_steps=[s for s in (payload.get("logic_steps") or []) if isinstance(s, dict)],
            key_points=[str(s) for s in (payload.get("key_points") or [])],
            edge_cases=[str(s) for s in (payload.get("edge_cases") or [])],
            issues=[s for s in (payload.get("issues") or []) if isinstance(s, dict)],
            complexity={str(k): str(v) for k, v in complexity.items()},
            structured=True,
            text=response.content,
            usage=self.total_usage,
        )

    @staticmethod
    def _guess_language(code: str) -> str:
        from .parsers import detect_language

        return detect_language(code) or "python"

    # ---------------------------------------------------------------- 4. 注释
    def comment(
        self,
        code: str,
        *,
        language: str = "",
        comment_language: str = "中文",
        few_shot: bool = True,
        extra_note: str = "",
        temperature: Optional[float] = None,
    ) -> CommentedCode:
        """为代码补充注释 / 文档字符串（保持逻辑不变）。"""
        language = normalize_language(language) or self._guess_language(code)
        note = extra_note or "- 若原代码已有有价值的注释，请保留其含义。"
        messages = build_messages(
            system=render(
                "code_comment.system",
                language=language,
                comment_language=comment_language,
                extra_note=note,
            ),
            few_shot=CODE_COMMENT_FEWSHOT if few_shot else None,
            user=render("code_comment.user", language=language, code=code),
        )
        response = self._chat(messages, temperature=temperature)

        block = first_code_block(response.content, default_language=language)
        commented = block.code if block else code
        prose = self._prose_outside_code(response.content)
        # 模型有时会把"修复说明"之类的段落也带进来，去掉纯代码块部分后剩下的即说明
        prose = prose.replace("修复说明", "").strip()

        return CommentedCode(
            code=commented,
            notes=prose,
            language=language,
            usage=self.total_usage,
        )

    # ---------------------------------------------------------------- 5. 问答
    def ask(
        self,
        question: str,
        *,
        code: str = "",
        language: str = "",
        history: Optional[Sequence[Message]] = None,
        answer_language: str = "中文",
        temperature: Optional[float] = None,
    ) -> QAAnswer:
        """回答关于代码的问题。

        带 code 参数时，这就是 RAG（检索增强生成）的最小形态：
        把"检索到的上下文"塞进提示词，让模型的回答有据可依。
        生产环境把 code 换成"向量检索出来的若干代码片段"即可。
        """
        context_block = ""
        if code.strip():
            tag = normalize_language(language) or self._guess_language(code)
            context_block = f"以下是相关的代码上下文：\n```{tag}\n{code}\n```\n"

        messages = build_messages(
            system=render("code_qa.system", answer_language=answer_language),
            history=self._trim_history(history or []),
            user=render("code_qa.user", context_block=context_block, question=question),
        )
        response = self._chat(messages, temperature=temperature)
        return QAAnswer(
            answer=response.content.strip(),
            used_context=bool(context_block),
            usage=self.total_usage,
        )

    # ---------------------------------------------------------------- 6. 多轮对话
    def chat(
        self,
        user_input: str,
        *,
        code: str = "",
        answer_language: str = "中文",
        temperature: Optional[float] = None,
    ) -> QAAnswer:
        """多轮问答：把历史带进上下文，实现"接着上一句继续聊"。"""
        answer = self.ask(
            user_input,
            code=code,
            history=self.history,
            answer_language=answer_language,
            temperature=temperature,
        )
        self.history.append(Message.user(user_input))
        self.history.append(Message.assistant(answer.answer))
        self.history = self._trim_history(self.history)
        return answer

    # ---------------------------------------------------------------- 7. Agent
    def build_agent(
        self,
        tools: Optional[ToolRegistry] = None,
        *,
        on_event: Any = None,
        answer_language: str = "中文",
    ) -> Agent:
        """构造一个带工具能力的 Agent（用于需要"先看代码再回答"的任务）。"""
        registry = tools or self.tools or ToolRegistry()
        return Agent(
            self.llm,
            registry,
            max_steps=self.config.max_steps,
            answer_language=answer_language,
            on_event=on_event,
        )

    def build_session(
        self,
        tools: Optional[ToolRegistry] = None,
        *,
        on_event: Any = None,
        answer_language: str = "中文",
        max_history: int = 40,
    ) -> AgentSession:
        """构造一个可连续追问的 Agent 会话（终端交互模式用）。"""
        registry = tools or self.tools or ToolRegistry()
        return AgentSession(
            self.llm,
            registry,
            max_steps=self.config.max_steps,
            max_history=max_history,
            answer_language=answer_language,
            on_event=on_event,
        )

    def agent_run(self, task: str, *, tools: Optional[ToolRegistry] = None, on_event: Any = None) -> AgentResult:
        return self.build_agent(tools, on_event=on_event).run(task)
