"""工具集成层：让模型从"会说"变成"能做"。

一个 Agent 的能力上限，往往不取决于模型多聪明，而取决于你给了它什么工具、
以及工具的参数设计是否"模型友好"。

本模块解决四件事：
    1. **声明**：`@tool` 装饰器从类型注解自动生成 JSON Schema —— 模型靠它决定怎么调。
    2. **协议**：`to_openai_schema()` 输出标准的 tools 数组，与 Function Calling 协议对齐。
    3. **执行**：`ToolRegistry.call()` 统一处理参数解析、异常捕获、结果截断。
    4. **安全**：工作区路径围栏、执行超时、输出长度限制。

核心设计原则（写工具时请反复默念）：
    * **描述就是文档**：模型只能看到 name / description / schema，看不到你的实现。
      描述里必须写清"什么时候用我"和"什么时候别用我"。
    * **参数少而明确**：参数越多，模型填错的概率指数上升。宁可拆成两个工具。
    * **错误要可自愈**：工具报错信息要能让模型知道"下一步怎么改"，
      比如「文件不存在，当前目录下有：a.py、b.py」，而不是一句 NullPointerException。
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, get_args, get_origin

MAX_OUTPUT_CHARS = 8000


# --------------------------------------------------------------------------------------
# 临时目录：优先用系统临时区，失败则回退到工作区
# --------------------------------------------------------------------------------------
def _mktemp_dir(parent: Optional[Path], prefix: str) -> Path:
    """创建一个唯一命名的临时目录（不用 tempfile.mkdtemp）。

    为什么不用标准库的 mkdtemp？
        它内部以 mode=0o700 创建目录。在某些受限沙箱 / 定制 ACL 环境下，
        "以 0o700 创建的目录"会变成**不可写**的（本项目开发环境正是如此），
        于是会出现 mkdtemp 成功、紧接着 write 就 PermissionError 的诡异现象。
        这里改为：用默认权限建目录，成功后再尽力把权限收紧到 0o700（POSIX 下）。
        功能正确性优先于权限美观 —— 建一个私有但写不进去的目录毫无意义。
    """
    base_parent = Path(parent) if parent is not None else Path(tempfile.gettempdir())
    base_parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for _ in range(8):
        candidate = base_parent / f"{prefix}{secrets.token_hex(4)}"
        try:
            candidate.mkdir()
        except FileExistsError as exc:  # 撞名了，换个随机后缀重试
            last_error = exc
            continue
        except OSError as exc:
            raise OSError(f"无法在 {base_parent} 下创建目录：{exc}") from exc
        if os.name != "nt":
            try:
                os.chmod(candidate, 0o700)
            except OSError:
                pass
        return candidate
    raise OSError(f"临时目录命名连续冲突：{last_error}")


def make_scratch_dir(*, prefix: str = "codeassist-", fallback_parent: Optional[Path] = None) -> Path:
    """找一个**确实可写**的临时工作目录。

    为什么要封装这一层，还要做"写入探测"？
        "目录创建成功"不等于"可以往里写文件"。受限沙箱、只读挂载、容器只读层、
        企业安全软件都会让建目录成功而 write 失败。
        只靠捕获 mkdtemp 的异常，就会得到"创建成功、一用就崩"的诡异故障。
        因此策略是：**候选位置逐个真写一次，第一个真正能写的胜出**。

    参数:
        prefix:          临时目录名前缀，便于在系统里辨识来源。
        fallback_parent: 系统临时区不可写时的回退父目录（通常是工作区）。
    """
    candidates: List[Optional[Path]] = [None]  # None 表示使用系统临时区
    if fallback_parent is not None:
        candidates.append(Path(fallback_parent) / ".codeassist_scratch")

    problems: List[str] = []
    for parent in candidates:
        label = str(parent) if parent else "系统临时目录"
        try:
            base = _mktemp_dir(parent, prefix)
        except OSError as exc:
            problems.append(f"{label}：创建失败（{exc}）")
            continue
        # 实际写一次，确认可用
        probe = base / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
        except OSError as exc:
            problems.append(f"{label}：不可写（{type(exc).__name__}）")
            cleanup_dir(base)
            continue
        try:
            probe.unlink()
        except OSError:
            pass  # 删不掉不影响使用
        return base

    raise ToolError("找不到可写的临时目录，已尝试：" + "；".join(problems))


def cleanup_dir(path: Path) -> None:
    """**尽力而为**地删除临时目录。

    关键点：清理失败绝不能影响主流程。
    在受限沙箱里"删不掉文件"是很常见的（本项目的开发环境就是如此），
    如果让清理异常往外冒，用户会看到"代码明明执行成功了，工具却报失败"。
    这类"把辅助动作当成主流程"的错误在工程里非常常见，务必避免。
    """
    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------------------
# 结果与异常
# --------------------------------------------------------------------------------------
class ToolError(Exception):
    """工具自身执行失败（参数非法、路径越界等）。"""


@dataclass
class ToolResult:
    """工具执行结果。

    注意：**工具失败不是异常，而是一种正常结果**。
    它会被序列化后塞回对话历史，让模型看到"刚才那步错了、错在哪"，从而自我纠正。
    这是 Agent 能"爬起来"的关键 —— 如果工具一报错就把整个流程抛崩，Agent 就没有鲁棒性可言。
    """

    ok: bool
    content: str = ""
    tool: str = ""
    error: str = ""
    duration_ms: float = 0.0
    truncated: bool = False

    def as_message_content(self) -> str:
        """转成要回灌给模型的那段文本。"""
        if self.ok:
            body = self.content or "(工具执行成功，无输出)"
        else:
            body = f"[工具执行失败] {self.error}"
        if self.truncated:
            body += "\n\n(输出过长已截断，如需完整内容请缩小查询范围)"
        return body


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple:
    """超长输出做"保头留尾"截断：开头往往是关键信息，结尾常有报错堆栈。"""
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n...\n[已截断 {len(text) - limit} 字符]\n...\n{text[-tail:]}", True


# --------------------------------------------------------------------------------------
# Schema 生成：从 Python 类型注解推导 JSON Schema
# --------------------------------------------------------------------------------------
_SIMPLE_TYPES: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    """把 Python 类型注解翻译成 JSON Schema 片段。

    只支持模型真正能理解的类型。遇到复杂类型（自定义类、Callable 等）
    一律降级为 string，并在描述里要求模型传 JSON 字符串 ——
    因为**让模型面对一个它无法构造的参数，是最糟糕的接口设计**。
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / Union[X, None] → 取非 None 的那一支
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"type": "string"}

    if origin in (list, List, Sequence, typing.Sequence, set, tuple):
        item = _annotation_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin in (dict, Dict, typing.Dict, typing.Mapping):
        return {"type": "object"}

    if annotation in _SIMPLE_TYPES:
        schema: Dict[str, Any] = {"type": _SIMPLE_TYPES[annotation]}
        return schema

    return {"type": "string"}


def build_parameters_schema(
    func: Callable[..., Any],
    param_docs: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """从函数签名构造 `parameters` 字段（JSON Schema 对象）。"""
    param_docs = param_docs or {}
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # pragma: no cover - 注解里有无法求值的字符串时兜底
        hints = {}

    signature = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        schema = _annotation_to_schema(hints.get(name, parameter.annotation))
        if name in param_docs:
            schema["description"] = param_docs[name]
        elif parameter.default is inspect.Parameter.empty:
            # 没有描述的参数是"坏接口"，至少给个占位提示，方便在日志里发现
            schema.setdefault("description", f"参数 {name}")
        if parameter.default is not inspect.Parameter.empty:
            schema["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = schema

    result: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    # 明确禁止额外参数，能让模型少犯"自创参数"的错
    result["additionalProperties"] = False
    return result


def _first_doc_line(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    for line in doc.split("\n"):
        if line.strip():
            return line.strip()
    return ""


# --------------------------------------------------------------------------------------
# Tool 与注册表
# --------------------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, Any]
    dangerous: bool = False
    returns_text: bool = True

    def to_openai_schema(self) -> Dict[str, Any]:
        """Function Calling 协议要求的单个工具声明。

        结构是固定的三段式：type=function → function.name / function.description
        → function.parameters（标准 JSON Schema）。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def signature_preview(self) -> str:
        """生成人类可读的签名（给系统提示和 /tools 命令用）。

        格式刻意做成两行：签名一行、参数说明一行。
        参数说明内联在签名里（`path: string  # 说明, next: int  # 说明`）在参数多的时候
        会折行折断、难以阅读 —— 而这段文本既要给模型看，也要给人看。
        """
        params = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        if not params:
            return f"{self.name}()"

        signature_parts: List[str] = []
        doc_parts: List[str] = []
        for key, schema in params.items():
            piece = f"{key}: {schema.get('type', 'string')}"
            if key not in required:
                piece += " = 可选"
            signature_parts.append(piece)
            if schema.get("description"):
                doc_parts.append(f"{key}={schema['description']}")

        lines = [f"{self.name}({', '.join(signature_parts)})"]
        if doc_parts:
            lines.append("    参数：" + "；".join(doc_parts))
        return "\n".join(lines)


class ToolRegistry:
    """工具集合。Agent 通过它把工具声明交给模型，并通过它执行模型的选择。"""

    def __init__(self, tools: Optional[Sequence[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for item in tools or []:
            self.register_tool(item)
        #: 调用审计：每次工具调用都留痕，便于排查"Agent 到底干了什么"
        self.call_log: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- 注册
    def register_tool(self, item: Tool) -> Tool:
        if item.name in self._tools:
            raise ValueError(f"工具重名：{item.name!r}")
        self._tools[item.name] = item
        return item

    def tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        param_docs: Optional[Dict[str, str]] = None,
        dangerous: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """在注册表上直接注册一个函数的装饰器。"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            wrapped = tool(
                name=name, description=description, param_docs=param_docs, dangerous=dangerous
            )(func)
            self.register_tool(wrapped)
            return func

        return decorator

    # ---------------------------------------------------------------- 查询
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def subset(self, names: Sequence[str]) -> "ToolRegistry":
        """只保留部分工具。

        真实系统里用处很大：给不同角色的子 Agent 装配不同的能力集，
        既降低模型选错工具的概率，也是最小权限原则的落地。
        """
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"工具不存在：{missing}；可用：{self.names()}")
        return ToolRegistry([self._tools[n] for n in names])

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        return [item.to_openai_schema() for item in self._tools.values()]

    def describe(self) -> str:
        """给系统提示用的"人类可读"工具清单。

        为什么不用 JSON Schema？因为 schema 是给机器解析的，塞进提示词里
        既浪费 token 又难读。人读的清单 + 机器读的 schema 双轨，效果最好。
        """
        lines = []
        for item in self._tools.values():
            lines.append(f"- {item.signature_preview()}\n    用途：{item.description}")
        return "\n".join(lines) if lines else "（当前没有可用工具）"

    # ---------------------------------------------------------------- 执行
    def call(self, name: str, arguments: Union[str, Dict[str, Any], None] = None) -> ToolResult:
        """执行一次工具调用。**永不抛异常**，所有失败都变成 ToolResult(ok=False)。"""
        started = time.perf_counter()
        item = self._tools.get(name)
        if item is None:
            available = "、".join(self.names()) or "无"
            result = ToolResult(
                ok=False,
                tool=name,
                error=f"工具 {name!r} 不存在。可用工具：{available}。请从可用工具中选择。",
            )
            self._log(name, arguments, result, started)
            return result

        # 参数可能是 JSON 字符串（协议规定）或已解析的 dict
        if arguments is None:
            kwargs: Dict[str, Any] = {}
        elif isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                kwargs = {}
            else:
                from .parsers import extract_json

                parsed = extract_json(text, default=None)
                if not isinstance(parsed, dict):
                    result = ToolResult(
                        ok=False,
                        tool=name,
                        error=(
                            f"参数不是合法 JSON 对象，收到的是：{text[:200]!r}。"
                            '请严格按 schema 输出，例如 {"path": "a.py"}'
                        ),
                    )
                    self._log(name, arguments, result, started)
                    return result
                kwargs = parsed
        elif isinstance(arguments, dict):
            kwargs = arguments
        else:
            result = ToolResult(ok=False, tool=name, error=f"不支持的参数类型：{type(arguments)}")
            self._log(name, arguments, result, started)
            return result

        # 未知参数直接拦掉并告诉模型正确参数 —— 比让 Python 抛 TypeError 更有指导性
        allowed = set(item.parameters.get("properties", {}))
        unexpected = set(kwargs) - allowed
        if unexpected:
            result = ToolResult(
                ok=False,
                tool=name,
                error=(
                    f"参数名错误：{sorted(unexpected)}。{name} 只接受 {sorted(allowed)}。"
                    "请修正参数名后重试。"
                ),
            )
            self._log(name, arguments, result, started)
            return result

        try:
            output = item.func(**kwargs)
        except ToolError as exc:
            result = ToolResult(ok=False, tool=name, error=str(exc))
        except TypeError as exc:
            result = ToolResult(
                ok=False, tool=name, error=f"参数调用失败：{exc}。参数 schema：{json.dumps(item.parameters, ensure_ascii=False)}"
            )
        except Exception as exc:  # noqa: BLE001 - 工具内部的任何异常都不能炸掉 Agent
            result = ToolResult(ok=False, tool=name, error=f"{type(exc).__name__}: {exc}")
        else:
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
            text, truncated = _truncate(text)
            result = ToolResult(ok=True, tool=name, content=text, truncated=truncated)

        self._log(name, arguments, result, started)
        return result

    def _log(self, name: str, arguments: Any, result: ToolResult, started: float) -> None:
        result.duration_ms = (time.perf_counter() - started) * 1000
        self.call_log.append(
            {
                "tool": name,
                "arguments": arguments,
                "ok": result.ok,
                "duration_ms": round(result.duration_ms, 2),
                "output_preview": result.content[:200] if result.ok else result.error[:200],
            }
        )


def tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_docs: Optional[Dict[str, str]] = None,
    dangerous: bool = False,
) -> Callable[[Callable[..., Any]], Tool]:
    """把一个普通 Python 函数变成工具。

    用法：
        @tool(description="读取文件内容", param_docs={"path": "相对工作区的路径"})
        def read_file(path: str, max_bytes: int = 20000) -> str: ...

    关键点：description **必须**包含"什么时候该用它"。这是模型唯一的判断依据。
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or _first_doc_line(func) or f"执行 {tool_name}"
        return Tool(
            name=tool_name,
            description=tool_description,
            func=func,
            parameters=build_parameters_schema(func, param_docs),
            dangerous=dangerous,
        )

    return decorator


# --------------------------------------------------------------------------------------
# 内置工具：代码助手真正需要的那几个
# --------------------------------------------------------------------------------------
def _resolve_within(root: Path, raw_path: str, *, must_exist: bool = True) -> Path:
    """把用户/模型给的路径解析到工作区之内，越界直接拒绝。

    安全边界的第一原则：**永远不要相信模型给的路径**。
    `../../.ssh/id_rsa` 这种输入在真实对抗场景里必然出现。
    """
    if raw_path is None or str(raw_path).strip() == "":
        raise ToolError("path 参数不能为空")
    candidate = Path(str(raw_path).strip())
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover
        raise ToolError(f"路径无法解析：{raw_path}（{exc}）") from exc

    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ToolError(
            f"路径越界：{raw_path} 不在工作区内。为安全起见，只允许访问 {root_resolved} 及其子目录。"
        )
    if must_exist and not resolved.exists():
        # 报错时顺便列出同级目录，帮助模型立刻纠正路径
        siblings = []
        parent = resolved.parent
        if parent.is_dir():
            siblings = sorted(p.name + ("/" if p.is_dir() else "") for p in list(parent.iterdir())[:25])
        hint = f"同目录下现有：{siblings}" if siblings else "父目录不存在"
        raise ToolError(f"路径不存在：{raw_path}。{hint}")
    return resolved


def build_default_registry(
    workspace: str | Path = ".",
    *,
    allow_execute: bool = True,
    max_read_bytes: int = 200_000,
    exec_timeout: float = 15.0,
) -> ToolRegistry:
    """装配默认工具集：文件读写 + 目录浏览 + 代码搜索 + 语法检查 +（可选）执行。"""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry()

    # ---------------------------------------------------------------- 列出目录
    @registry.tool(
        description=(
            "列出工作区内某个目录的文件与子目录。当你不确定项目结构、"
            "或需要找文件但不知道确切路径时，先用这个工具探路。"
        ),
        param_docs={"path": "目录路径，相对于工作区；留空表示工作区根目录"},
    )
    def list_dir(path: str = ".") -> str:
        directory = _resolve_within(root, path)
        if not directory.is_dir():
            raise ToolError(f"{path} 不是目录。")
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return "（空目录）"
        lines = []
        for entry in entries[:200]:
            if entry.is_dir():
                lines.append(f"[DIR]  {entry.name}/")
            else:
                lines.append(f"[FILE] {entry.name}  ({entry.stat().st_size} bytes)")
        if len(entries) > 200:
            lines.append(f"...（还有 {len(entries) - 200} 项未显示）")
        return "\n".join(lines)

    # ---------------------------------------------------------------- 读取文件
    @registry.tool(
        description=(
            "读取工作区内一个文本文件的内容，返回带行号的文本。"
            "需要了解某段代码的实现、排查 bug、回答关于代码的问题时使用。"
            "文件很大时请先用 start_line/end_line 指定范围，避免浪费上下文。"
        ),
        param_docs={
            "path": "文件路径，相对于工作区",
            "start_line": "起始行号（从 1 开始，包含）；留空表示从头开始",
            "end_line": "结束行号（包含）；留空表示读到文件末尾",
        },
    )
    def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
        file_path = _resolve_within(root, path)
        if not file_path.is_file():
            raise ToolError(f"{path} 不是文件。")
        size = file_path.stat().st_size
        if size > max_read_bytes:
            raise ToolError(
                f"文件过大（{size} 字节，上限 {max_read_bytes}）。请用 start_line/end_line 分段读取。"
            )
        try:
            # utf-8-sig：Windows 上带 BOM 的源码文件很常见，BOM 会污染第一行
            text = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise ToolError(f"{path} 不是 UTF-8 文本文件（可能是二进制），无法读取。") from None
        lines = text.split("\n")
        start = max(1, int(start_line) or 1)
        end = int(end_line) or len(lines)
        end = min(end, len(lines))
        if start > len(lines):
            raise ToolError(f"start_line={start} 超出文件总行数 {len(lines)}。")
        chunk = lines[start - 1 : end]
        # 带行号返回：模型引用行号，最终答案才有可验证的依据
        width = len(str(end))
        body = "\n".join(f"{start + i:>{width}} | {line}" for i, line in enumerate(chunk))
        header = f"{path}（共 {len(lines)} 行，显示第 {start}-{end} 行）\n"
        return header + body

    # ---------------------------------------------------------------- 写入文件
    @registry.tool(
        description=(
            "把内容写入工作区内的文件（覆盖写）。仅在用户明确要求把代码保存成文件时使用；"
            "如果用户只是想要代码内容，直接在回答里给出代码块即可，不要调用本工具。"
        ),
        param_docs={"path": "目标文件路径，相对于工作区", "content": "要写入的完整文本内容"},
        dangerous=True,
    )
    def write_file(path: str, content: str) -> str:
        file_path = _resolve_within(root, path, must_exist=False)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        existed = file_path.exists()
        file_path.write_text(content, encoding="utf-8")
        return f"已{'覆盖' if existed else '创建'}文件 {path}（{len(content)} 字符，{content.count(chr(10)) + 1} 行）"

    # ---------------------------------------------------------------- 搜索代码
    @registry.tool(
        description=(
            "在工作区的代码文件中按关键词或正则搜索，返回匹配的文件、行号与命中行。"
            "用于定位函数定义、查找某个变量的使用位置、了解某个功能的实现分布。"
        ),
        param_docs={
            "pattern": "要搜索的关键词或正则表达式，例如 'def login' 或 'class \\w+Service'",
            "path": "搜索起始目录，相对于工作区，默认为整个工作区",
            "file_glob": "只搜索匹配的文件名模式，例如 '*.py'；留空搜索常见代码文件",
        },
    )
    def search_code(pattern: str, path: str = ".", file_glob: str = "") -> str:
        import fnmatch
        import re as _re

        base = _resolve_within(root, path)
        try:
            regex = _re.compile(pattern)
        except _re.error as exc:
            raise ToolError(f"正则表达式不合法：{pattern}（{exc}）。请修正后重试。") from exc

        code_suffixes = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php",
            ".c", ".h", ".cpp", ".hpp", ".cs", ".kt", ".swift", ".sh", ".sql", ".md", ".json", ".yaml", ".yml", ".toml",
        }
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea"}
        hits: List[str] = []
        files_scanned = 0

        for file_path in sorted(base.rglob("*")):
            if len(hits) >= 100:
                break
            if not file_path.is_file():
                continue
            if any(part in skip_dirs for part in file_path.parts):
                continue
            if file_glob and not fnmatch.fnmatch(file_path.name, file_glob):
                continue
            if not file_glob and file_path.suffix.lower() not in code_suffixes:
                continue
            files_scanned += 1
            try:
                for number, line in enumerate(file_path.read_text(encoding="utf-8-sig", errors="ignore").split("\n"), 1):
                    if regex.search(line):
                        rel = file_path.relative_to(root)
                        hits.append(f"{rel}:{number}: {line.strip()[:160]}")
                        if len(hits) >= 100:
                            break
            except OSError:
                continue

        if not hits:
            return f"未找到匹配 {pattern!r} 的内容（已扫描 {files_scanned} 个文件）。可以换个关键词或去掉 file_glob 再试。"
        header = f"匹配 {pattern!r} 共 {len(hits)} 条（已扫描 {files_scanned} 个文件）：\n"
        return header + "\n".join(hits)

    # ---------------------------------------------------------------- 语法检查
    @registry.tool(
        description=(
            "检查一段 Python 代码的语法是否正确。在给出代码前自检、或用户报告语法错误时使用。"
            "注意：本工具只做语法检查，不执行代码。"
        ),
        param_docs={"code": "要检查的 Python 代码文本"},
    )
    def check_python_syntax(code: str) -> str:
        from .parsers import validate_python

        result = validate_python(code)
        if result.ok:
            line_count = code.count("\n") + 1
            return f"语法检查通过（{line_count} 行，可被 Python 解析器正确解析）。"
        return f"语法检查未通过：{result.error}"

    # ---------------------------------------------------------------- 执行代码
    if allow_execute:

        @registry.tool(
            description=(
                "在隔离的临时目录中运行一段 Python 代码并返回输出（stdout/stderr/退出码）。"
                "用于验证算法是否正确、复现 bug、确认某个函数的真实行为。"
                "代码有超时限制，不能访问网络，请只写短小的验证脚本；"
                "不要在代码里写 input()，因为没有交互输入。"
            ),
            param_docs={"code": "要执行的完整 Python 代码，可用 print 输出结果"},
            dangerous=True,
        )
        def run_python(code: str) -> str:
            if not code.strip():
                raise ToolError("code 不能为空")
            workdir = make_scratch_dir(prefix="codeassist-run-", fallback_parent=root)
            try:
                script = Path(workdir) / "snippet.py"
                script.write_text(code, encoding="utf-8")
                try:
                    completed = subprocess.run(
                        [sys.executable, "-I", "-X", "utf8", str(script)],
                        cwd=str(workdir),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=exec_timeout,
                        # 极简环境变量：不把宿主的密钥、代理等敏感信息泄漏给被执行的代码
                        env={"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                except subprocess.TimeoutExpired:
                    raise ToolError(
                        f"执行超时（>{exec_timeout}s）。请检查是否有死循环，或改为更小的输入规模。"
                    ) from None
            finally:
                # 清理失败不能影响结果 —— 沙箱里常常删不掉临时文件
                cleanup_dir(Path(workdir))
            parts = [f"退出码：{completed.returncode}"]
            if completed.stdout.strip():
                parts.append(f"--- stdout ---\n{completed.stdout.rstrip()}")
            else:
                parts.append("--- stdout ---\n（无输出）")
            if completed.stderr.strip():
                parts.append(f"--- stderr ---\n{completed.stderr.rstrip()}")
            if completed.returncode != 0:
                parts.append("提示：退出码非 0 说明代码抛出了异常，请根据 stderr 的 traceback 定位问题。")
            return "\n".join(parts)

    return registry
