"""解析层：把模型的自由文本，变成程序能用的结构化数据。

**这是最容易被低估、却最影响 Agent 稳定性的模块。**

新手常犯的错：`json.loads(response.content)` 直接一把梭，线上遇到模型多打了一句
"好的，以下是结果："就崩。真实世界的模型输出永远是"大体合规、局部跑偏"，
所以解析器必须遵循 **Be liberal in what you accept（对输入宽容）** 原则。

本模块提供：
    * extract_code_blocks   从 Markdown 回复里挖出代码块（含未闭合容错）
    * extract_json          从废话里抠出 JSON（含围栏、前后缀、括号配平容错）
    * validate_python       生成的代码到底能不能跑？（语法级校验）
    * detect_language       从代码内容/文件扩展名猜语言
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# 语言识别
# --------------------------------------------------------------------------------------
_EXTENSION_MAP = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".swift": "swift", ".m": "objectivec", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".ps1": "powershell", ".sql": "sql", ".r": "r", ".lua": "lua", ".dart": "dart",
    ".html": "html", ".css": "css", ".scss": "scss", ".vue": "vue", ".svelte": "svelte",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".xml": "xml",
    ".md": "markdown", ".dockerfile": "dockerfile", ".tf": "hcl",
}

#: 模型写代码块时爱用的别名，统一收敛，避免下游判断语言时漏网
_LANGUAGE_ALIASES = {
    "py": "python", "python3": "python", "python2": "python",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "ts": "typescript", "golang": "go", "rs": "rust",
    "c++": "cpp", "cxx": "cpp", "cc": "cpp",
    "c#": "csharp", "cs": "csharp", "sh": "bash", "shell": "bash",
    "console": "bash", "powershell": "powershell", "ps1": "powershell",
    "yml": "yaml", "rb": "ruby", "kt": "kotlin",
    "plaintext": "", "text": "", "txt": "", "": "",
}

_KEYWORD_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("python", ("def ", "import ", "self.", "elif ", "print(", "__init__", "None", "lambda ")),
    ("javascript", ("const ", "let ", "=>", "function ", "console.log", "require(", "export ")),
    ("typescript", ("interface ", ": string", ": number", "type ", "as const")),
    ("java", ("public class", "private ", "System.out.println", "void main")),
    ("go", ("func ", "package main", ":= ", "fmt.")),
    ("rust", ("fn main", "let mut", "impl ", "println!", "pub fn")),
    ("sql", ("SELECT ", "FROM ", "WHERE ", "INSERT INTO", "CREATE TABLE")),
    ("bash", ("#!/bin/", "echo $", "fi\n", "sudo ", "$(")),
    ("csharp", ("using System", "public static void", "Console.WriteLine")),
    ("cpp", ("#include <", "std::", "int main(")),
]


def detect_language(text: str, *, filename: Optional[str] = None) -> str:
    """猜测语言。优先用文件名（最可靠），其次用关键字打分。"""
    if filename:
        lowered = filename.lower()
        if lowered.endswith("dockerfile"):
            return "dockerfile"
        for ext, language in _EXTENSION_MAP.items():
            if lowered.endswith(ext):
                return language
    if not text.strip():
        return ""

    scores: Dict[str, int] = {}
    for language, keywords in _KEYWORD_HINTS:
        score = sum(1 for kw in keywords if kw in text)
        if score:
            scores[language] = score
    # typescript 常被 javascript 关键字命中，做一次消歧
    if "typescript" in scores and "javascript" in scores:
        if scores["typescript"] >= scores["javascript"] - 1:
            scores["javascript"] = 0
    if not scores:
        return ""
    return max(scores.items(), key=lambda kv: kv[1])[0]


def normalize_language(name: Optional[str]) -> str:
    if not name:
        return ""
    return _LANGUAGE_ALIASES.get(name.strip().lower(), name.strip().lower())


# --------------------------------------------------------------------------------------
# 代码块抽取
# --------------------------------------------------------------------------------------
@dataclass
class CodeBlock:
    language: str
    code: str
    fence: str = "```"
    closed: bool = True

    def __str__(self) -> str:  # pragma: no cover
        return self.code


_FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`~]*)[^\n]*$", re.MULTILINE)


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """从 Markdown 文本中抽取所有代码块。

    容错点（都来自真实踩坑）：
      * 支持 ``` 与 ~~~ 两种围栏；
      * 支持 4 个及以上反引号（当代码里含 ``` 时模型会加长围栏）；
      * **未闭合的围栏**：按"到文本结尾"处理（模型被 max_tokens 截断时极常见）；
      * 语言标识带多余信息（```python title="x.py"）时只取第一个词。
    """
    if not text:
        return []
    blocks: List[CodeBlock] = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        match = _FENCE_OPEN.match(lines[index])
        if not match:
            index += 1
            continue
        fence, raw_language = match.group(1), match.group(2)
        language = normalize_language(raw_language.split()[0] if raw_language else "")
        body: List[str] = []
        index += 1
        closed = False
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith(fence[0] * 3) and stripped.rstrip() == fence.rstrip():
                closed = True
                index += 1
                break
            body.append(lines[index])
            index += 1
        blocks.append(
            CodeBlock(language=language, code="\n".join(body).strip("\n"), fence=fence, closed=closed)
        )
    return blocks


def first_code_block(text: str, *, default_language: str = "") -> Optional[CodeBlock]:
    blocks = extract_code_blocks(text)
    if not blocks:
        return None
    block = blocks[0]
    if not block.language and default_language:
        block = CodeBlock(default_language, block.code, block.fence, block.closed)
    return block


def strip_code_fence(text: str) -> str:
    """去掉包裹代码的围栏；没有围栏就原样返回（补全场景经常没有围栏）。"""
    block = first_code_block(text)
    if block is not None and block.code.strip():
        # 只有在"整个回复就是一个代码块"时才认为它是纯代码，避免误伤带解释的回复
        if text.strip().startswith(block.fence) and len(extract_code_blocks(text)) == 1:
            return block.code
    return text.strip()


def split_prose_and_code(text: str) -> Tuple[str, str]:
    """把回复切成（围栏外的说明文字, 第一个代码块的内容）。"""
    blocks = extract_code_blocks(text)
    if not blocks:
        return text.strip(), ""
    code = blocks[0].code
    prose = re.sub(r"(`{3,}|~{3,})[^\n]*\n.*?(?:\n\1|\Z)", "", text, flags=re.DOTALL).strip()
    return prose, code


# --------------------------------------------------------------------------------------
# JSON 抽取
# --------------------------------------------------------------------------------------
def _scan_balanced(text: str, start: int) -> Optional[str]:
    """从 start 位置开始做括号配平扫描，返回完整 JSON 片段。

    必须处理字符串字面量，否则 `{"a": "}"}` 这种会让配平提前结束。
    """
    opening = text[start]
    closing = {"{": "}", "[": "]"}[opening]
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None  # 没配平（被截断）


def extract_json(text: str, *, default: Any = None) -> Any:
    """尽最大努力从模型回复中解析出 JSON。

    依次尝试：
      1. 整体就是合法 JSON（最理想情况）
      2. 剥掉 ```json 围栏后再解析
      3. 从第一个 { 或 [ 开始做括号配平扫描
      4. 修复常见毛病（单引号、尾随逗号）后重试
    全失败时返回 default（默认 None），把决策权交回调用方 —— 绝不抛异常打断流程。
    """
    if not text or not text.strip():
        return default
    candidate = text.strip()

    # 1) 直接解析（最理想：模型老实输出了纯 JSON）
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2) 剥掉 ```json 围栏后再试
    fenced = re.search(r"(`{3,}|~{3,})\s*(?:json|JSON)?\s*\n(.*?)(?:\n\1|\Z)", candidate, re.DOTALL)
    if fenced:
        inner = fenced.group(2).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            candidate = inner  # 后续扫描在"围栏内"进行，避免被围栏外的废话干扰

    # 3) 括号配平扫描：先找对象，再找数组
    for opener in ("{", "["):
        position = candidate.find(opener)
        while position != -1:
            snippet = _scan_balanced(candidate, position)
            if snippet:
                for attempt in (snippet, _repair_json(snippet)):
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        continue
            position = candidate.find(opener, position + 1)

    return default


def _repair_json(snippet: str) -> str:
    """修复模型最常犯的两种 JSON 毛病：尾随逗号、单引号当作字符串定界符。"""
    repaired = re.sub(r",\s*([}\]])", r"\1", snippet)
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


# --------------------------------------------------------------------------------------
# 代码校验
# --------------------------------------------------------------------------------------
@dataclass
class ValidationResult:
    ok: bool
    error: str = ""
    line: Optional[int] = None

    def __bool__(self) -> bool:
        return self.ok


def validate_python(code: str) -> ValidationResult:
    """语法级校验：能否被解析。

    注意这不是"能跑"的保证，但能拦掉 80% 的低级错误（缩进错、括号不配对、
    模型写了 `...` 占位符）。把这类错误在本地拦住、回灌给模型修复，
    比让用户拿到跑不了的代码体验好得多 —— 这就是"自修复闭环"的触发点。
    """
    if not code.strip():
        return ValidationResult(False, "代码为空")
    # 纵深防御：无论代码是从哪个路径来的（文件读取、模型输出、用户粘贴），
    # 都要先摘掉可能存在的 BOM，否则 ast.parse 会报"非法字符"这种毫无指向性的错。
    if code.startswith("\ufeff"):
        code = code.lstrip("\ufeff")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        detail = f"{exc.msg}"
        if exc.lineno:
            detail += f"（第 {exc.lineno} 行）"
            lines = code.split("\n")
            if 1 <= exc.lineno <= len(lines):
                detail += f"：{lines[exc.lineno - 1].strip()}"
        return ValidationResult(False, f"Python 语法错误：{detail}", exc.lineno)
    return ValidationResult(True)


#: 一眼看出是"占位符"而不是真代码的特征，生成结果里出现即视为不合格
PLACEHOLDER_PATTERNS = (
    "...",  # 省略号占位
    "<your code",
    "<code here",
    "TODO: implement",
    "// your code",
    "# your code",
    "pass  # TODO",
)


def find_placeholders(code: str) -> List[str]:
    """检测模型"偷懒"留下的占位符。这是代码生成最影响体验的失败模式。"""
    found = []
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in code:
            # `...` 在 Python 里是合法的 Ellipsis，但作为函数体几乎总是占位符
            if pattern == "..." and re.search(r"^\s*\.\.\.\s*(#.*)?$", code, re.MULTILINE) is None:
                continue
            found.append(pattern)
    return found


def validate_generated_code(code: str, language: str) -> List[str]:
    """多语言统一校验入口，返回问题列表（空列表 = 通过）。"""
    problems: List[str] = []
    language = normalize_language(language)
    if not code.strip():
        return ["生成结果为空"]
    for placeholder in find_placeholders(code):
        problems.append(f"代码中包含占位符 {placeholder!r}，未给出完整实现")
    if language in {"python", ""}:
        result = validate_python(code)
        if not result.ok:
            problems.append(result.error)
    return problems


# --------------------------------------------------------------------------------------
# 符号抽取：只把"要问的那个函数"放进上下文
# --------------------------------------------------------------------------------------
@dataclass
class SymbolSlice:
    """从大文件里切出来的一个函数/类片段。"""

    name: str
    code: str
    start_line: int
    end_line: int
    kind: str = "function"  # function | class

    def __str__(self) -> str:  # pragma: no cover
        return self.code


def _collect_definitions(body: Sequence[ast.stmt], prefix: str = "") -> Dict[str, ast.AST]:
    """递归收集模块/类/函数体内的定义，键为限定名（如 `LRUCache.get`）。"""
    found: Dict[str, ast.AST] = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualified = f"{prefix}{node.name}"
            found[qualified] = node
            # 继续往里找：类的成员方法（Class.method）是最常被问到的东西
            found.update(_collect_definitions(getattr(node, "body", []), prefix=f"{qualified}."))
    return found


def list_symbols(code: str) -> List[str]:
    """列出代码里所有可被抽取的符号名。

    用来在"符号没找到"时给用户一个可执行的提示 ——
    只报"没找到"是坏体验，列出"有哪些"才能让人立刻改对。
    会过滤掉 `__str__` 这类魔法方法：它们几乎不会是用户想问的东西，只会干扰列表。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = _collect_definitions(tree.body)
    return sorted(name for name in names if not name.split(".")[-1].startswith("__"))


def _node_span(node: ast.AST) -> Tuple[int, int]:
    """取节点的行范围，并把装饰器一起包含进来（@decorator 也属于这个函数）。"""
    start = getattr(node, "lineno", 1)
    for decorator in getattr(node, "decorator_list", []) or []:
        start = min(start, getattr(decorator, "lineno", start))
    end = getattr(node, "end_lineno", None) or start
    return start, end


def extract_symbol(code: str, name: str) -> Optional[SymbolSlice]:
    """按名字抽取函数/类的源码片段。

    支持三种写法（按优先级）：
        1. 完全限定名：`LRUCache.get`
        2. 精确的顶层名：`extract_json`
        3. 唯一的短名：`get`（若多个类都有 get，则拒绝并提示用限定名）

    **为什么要做这件事？**
        上下文是稀缺资源。问"extract_json 怎么处理括号"时把整个 400 行文件塞进去，
        既费钱又会稀释模型对关键代码的注意力（lost in the middle）。
        精准给一段 30 行的函数，答案质量和成本都会好一个档次。
    """
    if not code.strip() or not name.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    definitions = _collect_definitions(tree.body)
    target = name.strip()
    node = definitions.get(target)

    if node is None:
        # 退化为短名匹配：`Class.method` 里的 `method`
        short_matches = [
            key for key in definitions if key == target or key.split(".")[-1] == target
        ]
        if len(short_matches) == 1:
            target = short_matches[0]
            node = definitions[target]
        elif len(short_matches) > 1:
            # 有歧义时必须报错，不能瞎选一个 —— 否则用户拿到的是错的答案
            raise ValueError(
                f"符号名 {name!r} 不唯一，匹配到多个：{sorted(short_matches)}。"
                "请使用限定名（例如 Object.method）。"
            )
    if node is None:
        return None

    start, end = _node_span(node)
    lines = code.split("\n")
    if start < 1 or end > len(lines):
        return None
    snippet = "\n".join(lines[start - 1 : end])
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    return SymbolSlice(name=target, code=snippet, start_line=start, end_line=end, kind=kind)
