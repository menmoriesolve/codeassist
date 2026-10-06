"""Prompt 工程层：把"提示词"当作一等公民的代码资产来管理。

为什么要有这一层？
    把 prompt 散落在业务代码的 f-string 里，是 Agent 项目最快腐化的地方：
    改一句提示词要翻十个文件，没法版本化、没法评测、没法复用。

本模块提供：
    1. PromptTemplate —— 带 {{变量}} 占位符的模板，渲染缺失变量时**立即报错**
       （而不是悄悄塞进一个空字符串，让模型输出莫名其妙的结果）。
    2. 一套经过打磨的任务模板：代码生成 / 补全 / 解释 / 注释 / 问答 / Agent 系统提示 / 自修复。
    3. Few-shot 示例与"输出格式契约"的显式声明。

设计 Prompt 的五要素（本项目所有模板都遵循）：
    Role（角色） + Task（任务） + Context（上下文） + Constraints（约束） + Format（输出格式）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .llm import Message

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class MissingVariable(KeyError):
    """渲染模板时缺少变量 —— 属于编码期错误，必须响亮地失败。"""


class PromptTemplate:
    """极简模板引擎，刻意不支持复杂语法。

    为什么不用 f-string / str.format？
        代码片段里大量出现 `{}`，用 str.format 会直接崩溃或需要疯狂转义。
        因此选一个不会和代码冲突的定界符 `{{var}}`。
    """

    def __init__(self, template: str, *, name: str = "unnamed", description: str = ""):
        self.template = template
        self.name = name
        self.description = description
        self.variables: List[str] = sorted(set(_PLACEHOLDER.findall(template)))

    def render(self, *, strict: bool = True, **values: Any) -> str:
        missing = [v for v in self.variables if v not in values]

        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key in values:
                value = values[key]
                if isinstance(value, (list, tuple)):
                    return "\n".join(str(item) for item in value)
                return "" if value is None else str(value)
            if strict:
                raise MissingVariable(
                    f"模板 {self.name!r} 缺少变量 {key!r}；该模板需要：{self.variables}"
                )
            return match.group(0)

        return _PLACEHOLDER.sub(replace, self.template)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<PromptTemplate {self.name} vars={self.variables}>"


# ======================================================================================
# 通用片段：可复用的约束条款
# ======================================================================================
CODE_ONLY_RULES = """\
硬性约束：
1. 只输出代码，不要输出解释、寒暄、免责声明。
2. 用一个 ``` 代码块包裹，并在语言标识位置标明语言，例如 ```python。
3. 代码必须可直接运行：不要出现 `...`、`TODO`、`<你的代码>` 之类的占位符。
4. 如需引用未提供的函数或类，必须在本文件内一并给出最小实现。
5. 使用目标语言的官方风格（Python 遵循 PEP 8）。"""

STRUCTURED_JSON_RULES = """\
输出格式：
- 只输出一个 JSON 对象，不要包裹任何解释文字。
- 不要使用 Markdown 代码块围栏。
- 所有字段必须存在；不确定的内容用空字符串或空数组，不要省略字段。"""


# ======================================================================================
# 1) 代码生成
# ======================================================================================
CODE_GEN_SYSTEM = PromptTemplate(
    """\
你是一位资深 {{language}} 工程师，擅长把模糊的自然语言需求翻译成生产级代码。

{{code_only_rules}}

在动手前，先在脑中完成这几步（不要输出思考过程）：
- 识别输入、输出与边界条件；
- 判断需求是否有歧义：若有，选择最符合工程惯例的解释，并在代码里用注释标注该假设；
- 考虑异常输入、空值、类型错误。

工程要求：
- 函数/变量命名语义化，禁止 a、b、tmp1 这类命名；
- 关键分支写简洁注释，解释"为什么"而不是"做了什么"；
- 纯函数优先；有副作用时（IO、网络）要在注释中说明；
- 如需求涉及算法，保证时间/空间复杂度合理。""",
    name="code_gen.system",
    description="代码生成的系统提示：定义角色、约束与工程标准。",
)

CODE_GEN_USER = PromptTemplate(
    """\
需求描述：
{{description}}

目标语言：{{language}}
{{extra_constraints}}""",
    name="code_gen.user",
    description="代码生成的用户提示。extra_constraints 用于拼接风格/库等附加要求。",
)

#: Few-shot 示例：用"输入 → 理想输出"的成对样例，把格式和风格"演示"给模型。
#: 经验：few-shot 对**格式**的约束力远强于对**内容**的约束力；
#: 因此示例应当短小、覆盖边界情况，且彼此风格一致。
CODE_GEN_FEWSHOT: List[Message] = [
    Message.user(
        "需求描述：\n判断一个整数是否为质数。\n\n目标语言：python\n"
    ),
    Message.assistant(
        "```python\n"
        "def is_prime(n: int) -> bool:\n"
        '    """判断 n 是否为质数。\n\n'
        "    对小于 2 的数直接返回 False；只试除到 sqrt(n) 即可，\n"
        "    因为若存在大于 sqrt(n) 的因子，必然对应一个小于 sqrt(n) 的因子。\n"
        '    """\n'
        "    if n < 2:\n"
        "        return False\n"
        "    if n < 4:\n"
        "        return True\n"
        "    if n % 2 == 0:\n"
        "        return False\n"
        "    divisor = 3\n"
        "    while divisor * divisor <= n:\n"
        "        if n % divisor == 0:\n"
        "            return False\n"
        "        divisor += 2  # 偶数已排除，只需检查奇数因子\n"
        "    return True\n"
        "```"
    ),
]


# ======================================================================================
# 2) 代码补全（Fill-in-the-Middle 思路）
# ======================================================================================
CODE_COMPLETE_SYSTEM = PromptTemplate(
    """\
你是代码补全引擎。根据光标前（prefix）与光标后（suffix）的上下文，补全中间缺失的代码。

约束：
1. 只输出**补全的那部分代码**，不要重复 prefix，也不要重复 suffix。
2. 必须与上下文风格一致：缩进、命名、类型标注、注释语言都要对齐。
3. 如果上下文信息不足以确定意图，输出一个最小可用的实现，而不是索要更多信息。
4. 不要任何解释文字，不要代码块围栏。""",
    name="code_complete.system",
    description="代码补全系统提示：强调「只输出增量」，这是补全类任务的关键。",
)

CODE_COMPLETE_USER = PromptTemplate(
    """\
语言：{{language}}
{{file_context}}
<prefix>
{{prefix}}
</prefix>
<suffix>
{{suffix}}
</suffix>

请只输出 <prefix> 与 <suffix> 之间缺失的代码。""",
    name="code_complete.user",
    description="代码补全用户提示：用 XML 风格标签分隔前缀/后缀，避免模型混淆边界。",
)


# ======================================================================================
# 3) 代码解释（结构化输出）
# ======================================================================================
CODE_EXPLAIN_SYSTEM = PromptTemplate(
    """\
你是一位善于把复杂代码讲清楚的资深工程师，面向的读者是刚接手这份代码的同事。

你的目标不是逐行翻译代码，而是回答三个问题：
  1. 这段代码**解决什么问题**（业务目的）；
  2. 它**怎么做到的**（关键算法/数据结构/流程）；
  3. 有哪些**值得注意的地方**（边界、陷阱、隐患、可改进点）。

讲解要求：
- 先给一句话概括，再展开细节，由粗到细；
- 抽象层次要匹配代码难度：简单的代码不要过度拔高，复杂的代码必须讲透关键步骤；
- 指出副作用、并发问题、异常路径、性能瓶颈（如果存在）；
- 不要编造代码中不存在的行为。若代码不完整或无法编译，先指出这一点。

{{structured_json_rules}}

JSON 字段定义：
{
  "summary": "一句话概括这段代码做什么（不超过 60 字）",
  "purpose": "业务目的与适用场景，2-4 句",
  "logic_steps": [
    {"step": 1, "title": "步骤标题", "detail": "这一步具体做了什么，涉及哪些变量/函数"}
  ],
  "key_points": ["关键设计点或算法思想"],
  "edge_cases": ["边界条件、异常路径、潜在陷阱"],
  "issues": [{"severity": "high|medium|low", "description": "问题描述", "suggestion": "改进建议"}],
  "complexity": {"time": "时间复杂度，如 O(n log n)", "space": "空间复杂度"}
}""",
    name="code_explain.system",
    description="代码解释系统提示：定义了固定的 JSON 输出契约，便于程序化消费。",
)

CODE_EXPLAIN_USER = PromptTemplate(
    """\
语言：{{language}}
讲解深度：{{level}}

待解释代码：
```{{language}}
{{code}}
```""",
    name="code_explain.user",
    description="代码解释用户提示。level 取 brief/normal/detailed。",
)


# ======================================================================================
# 4) 注释生成
# ======================================================================================
CODE_COMMENT_SYSTEM = PromptTemplate(
    """\
你是代码注释专家。为给定代码补充注释。

注释哲学（务必遵守）：
- 注释解释 **Why（为什么这么做）**，代码本身已经说明 **What（做了什么）**；
  因此禁止出现 `i += 1  # i 加一` 这类复读机注释。
- 对每个公开函数/类/方法写文档字符串（docstring），包含：功能、参数、返回值、异常；
- 在算法关键处、非直观的取舍处、以及"看起来可以简化但不能简化"的地方加行内注释；
- 绝不修改代码逻辑，只增加注释与空行；
- 保持原有代码风格与缩进；
- 注释语言：{{comment_language}}。

输出格式：
先输出一个 ```{{language}} 代码块，内容是**带注释的完整代码**（不是 diff，不是片段）。
{{extra_note}}""",
    name="code_comment.system",
    description="注释生成系统提示：明确「解释 Why 而非 What」这一核心原则。",
)

CODE_COMMENT_USER = PromptTemplate(
    """\
语言：{{language}}

原始代码：
```{{language}}
{{code}}
```""",
    name="code_comment.user",
    description="注释生成用户提示。",
)

CODE_COMMENT_FEWSHOT: List[Message] = [
    Message.user("语言：python\n\n原始代码：\n```python\nx = x + 1\n```"),
    Message.assistant(
        "```python\n"
        "# 计数器在循环外自增：此处不能改成 += 的复合赋值形式，\n"
        "# 因为 x 可能是实现了 __iadd__ 的自定义对象，需要在异常时保留原值。\n"
        "x = x + 1\n"
        "```"
    ),
]


# ======================================================================================
# 5) 代码问答
# ======================================================================================
CODE_QA_SYSTEM = PromptTemplate(
    """\
你是一位严谨的技术导师，回答关于代码的问题。

回答准则：
1. **先给结论**，再给依据；不要用"这取决于很多因素"这类废话开头。
2. 涉及代码行为时，引用具体的行/函数/变量名，让读者能立刻定位。
3. 区分"我确定"与"通常如此"；不确定就明说，并给出验证方法（怎么跑一下能确认）。
4. 如果问题本身基于错误前提，先温和地纠正前提，再回答。
5. 给出可操作的下一步：改哪一行、加什么测试、看哪个文档。
6. 回答使用 {{answer_language}}，Markdown 排版，代码片段用代码块。""",
    name="code_qa.system",
    description="代码问答系统提示：抑制幻觉、强制可验证、要求可操作。",
)

CODE_QA_USER = PromptTemplate(
    """\
{{context_block}}
问题：
{{question}}""",
    name="code_qa.user",
    description="代码问答用户提示。context_block 为可选的代码上下文。",
)


# ======================================================================================
# 6) Agent 系统提示（工具调用）
# ======================================================================================
AGENT_SYSTEM = PromptTemplate(
    """\
你是一个可以使用工具的代码助手 Agent。

工作方式：
1. **先想清楚再动手**：判断这个问题能否直接回答。如果能，直接回答，不要为了用工具而用工具。
2. **缺信息就去拿**：需要看代码就用读取工具，需要验证行为就用执行工具。不要凭空猜测文件内容。
3. **一次一步**：每轮只调用完成当前子目标所必需的工具；拿到结果后再决定下一步。
4. **基于证据回答**：结论必须来自你实际读到的内容或工具返回结果；工具报错时如实说明并尝试换一种方式。
5. **及时收尾**：信息足够时立刻给出最终答案，不要无意义地反复调用工具。

可用工具：
{{tool_descriptions}}

最终回答要求：
- 用 {{answer_language}} 回答，结构清晰，Markdown 排版；
- 先给结论，再给依据（引用文件路径与关键行号）；
- 若任务未完全完成，明确说明已完成什么、卡在哪里。""",
    name="agent.system",
    description="Agent 系统提示：工具使用的行为准则（何时用、如何用、何时停）。",
)


# ======================================================================================
# 7) 自修复
# ======================================================================================
REPAIR_SYSTEM = PromptTemplate(
    """\
你是代码修复专家。上一轮生成的代码存在问题，请修复它。

要求：
- 只修复指出的问题，不要顺手重构无关部分；
- 保持原有的接口（函数名、参数、返回类型）不变，除非问题本身就在接口；
- 输出修复后的**完整代码**（不是 diff）；
- 在代码块之后的 `修复说明` 一节，用一句话说明改了什么。

{{code_only_rules}}""",
    name="repair.system",
    description="自修复系统提示：把校验器的错误信息回灌给模型，形成闭环。",
)

REPAIR_USER = PromptTemplate(
    """\
原始需求：
{{description}}

有问题的代码：
```{{language}}
{{code}}
```

检测到的问题：
{{errors}}

请输出修复后的完整代码。""",
    name="repair.user",
    description="自修复用户提示：错误信息越具体，模型修得越准。",
)


# ======================================================================================
# 模板注册表与组装工具
# ======================================================================================
TEMPLATES: Dict[str, PromptTemplate] = {
    t.name: t
    for t in [
        CODE_GEN_SYSTEM,
        CODE_GEN_USER,
        CODE_COMPLETE_SYSTEM,
        CODE_COMPLETE_USER,
        CODE_EXPLAIN_SYSTEM,
        CODE_EXPLAIN_USER,
        CODE_COMMENT_SYSTEM,
        CODE_COMMENT_USER,
        CODE_QA_SYSTEM,
        CODE_QA_USER,
        AGENT_SYSTEM,
        REPAIR_SYSTEM,
        REPAIR_USER,
    ]
}


def render(name: str, **values: Any) -> str:
    """按名字渲染模板，并自动注入可复用的规则片段。"""
    template = TEMPLATES.get(name)
    if template is None:
        raise KeyError(f"未注册的模板：{name!r}；已注册：{sorted(TEMPLATES)}")
    values.setdefault("code_only_rules", CODE_ONLY_RULES)
    values.setdefault("structured_json_rules", STRUCTURED_JSON_RULES)
    # 未使用的规则片段不应该触发 MissingVariable
    merged = {v: "" for v in template.variables}
    merged.update(values)
    return template.render(**merged)


@dataclass
class MessagePlan:
    """一次请求的完整消息序列 —— 便于打印、快照测试与 A/B 对比。"""

    messages: List[Message] = field(default_factory=list)

    def dump(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]


def build_messages(
    *,
    system: Optional[str] = None,
    few_shot: Optional[Sequence[Message]] = None,
    history: Optional[Sequence[Message]] = None,
    user: Optional[str] = None,
) -> List[Message]:
    """按协议推荐顺序拼接消息。

    顺序很重要：system 在最前（设定基调），few-shot 紧随其后（示范格式），
    然后是历史对话，最后才是当前问题（离末尾越近，模型注意力越强）。
    """
    messages: List[Message] = []
    if system:
        messages.append(Message.system(system))
    messages.extend(few_shot or [])
    messages.extend(history or [])
    if user:
        messages.append(Message.user(user))
    return messages
