"""离线 Mock 模型：让整个项目"零密钥、零网络"也能完整跑起来。

它当然不会真的推理。但它做两件有价值的事：
    1. 用**轻量静态分析**（ast + 正则）真实地读懂一点代码，所以解释/注释结果是有依据的，
       不是纯粹的假数据 —— 这让演示和测试都有意义。
    2. 严格遵守协议：该返回 tool_calls 就返回 tool_calls，该返回纯 JSON 就返回纯 JSON，
       因此 Agent 循环、解析容错、自修复这些**工程代码得到了真实锻炼**。

生产代码永远不该用 Mock；它的定位是：开发期脚手架、单元测试替身、演示沙盘。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import Config
from .llm import BaseLLM, ChatResponse, Message, MockLLM, ToolCall
from .parsers import extract_code_blocks, normalize_language

MOCK_MARK = "# [Mock 离线模型] 未调用真实 LLM"


# ======================================================================================
# 轻量代码分析器：Mock 的"眼睛"
# ======================================================================================
def analyze_code(code: str) -> Dict[str, Any]:
    """对代码做一次浅层静态分析，返回结构化事实。

    这不是编译器级别的分析，但足以支撑"有依据的解释"。
    """
    facts: Dict[str, Any] = {
        "lines": code.count("\n") + 1,
        "functions": [],
        "classes": [],
        "imports": [],
        "loops": 0,
        "nested_loops": False,
        "has_recursion": False,
        "has_try": False,
        "sorted_used": False,
        "async_used": False,
        "comprehensions": 0,
        "issues": [],
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        facts["syntax_error"] = f"{exc.msg}（第 {exc.lineno} 行）"
        return facts

    function_nodes: List[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            facts["functions"].append({"name": node.name, "args": args, "lineno": node.lineno})
            if isinstance(node, ast.AsyncFunctionDef):
                facts["async_used"] = True
            function_nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            facts["classes"].append({"name": node.name, "methods": methods, "lineno": node.lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                facts["imports"].extend(alias.name for alias in node.names)
            else:
                facts["imports"].append(node.module or "")
        elif isinstance(node, (ast.For, ast.While)):
            facts["loops"] += 1
            if any(isinstance(child, (ast.For, ast.While)) for child in ast.walk(node)):
                facts["nested_loops"] = True
        elif isinstance(node, ast.Try):
            facts["has_try"] = True
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            facts["comprehensions"] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "sorted":
                facts["sorted_used"] = True

    # 自递归检测：函数体中是否出现了对函数自身的调用。
    # 必须走 AST 而不是字符串匹配 —— 字符串匹配会把 "def merge" 里的名字
    # 或另一个函数体内的同名局部变量误判成递归。
    for node in function_nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                called = getattr(func, "id", None) or getattr(func, "attr", None)
                if called == node.name:
                    facts["has_recursion"] = True
                    facts.setdefault("recursive_functions", []).append(node.name)
                    break

    # 常见反模式检测 —— 这些正是"代码审查"该抓的东西
    for pattern, severity, message in (
        (r"except\s*:", "medium", "存在裸 except，会吞掉 KeyboardInterrupt 等系统异常，建议捕获具体异常类型"),
        (r"def\s+\w+\([^)]*=\s*(\[\]|\{\})", "high", "使用了可变默认参数（[] 或 {}），多次调用会共享同一个对象，应改用 None + 内部初始化"),
        (r"==\s*None|!=\s*None", "low", "与 None 比较建议使用 is / is not"),
        (r"\beval\s*\(", "high", "使用了 eval，存在代码注入风险"),
        (r"open\((?![^)]*with)", "low", "直接 open 未使用 with，异常时可能泄漏文件句柄"),
        (r"print\(", "low", "代码中包含 print，生产环境建议改用 logging"),
    ):
        if re.search(pattern, code):
            facts["issues"].append({"severity": severity, "pattern": message})

    return facts


def guess_complexity(facts: Dict[str, Any]) -> Dict[str, str]:
    """从循环层数/排序/递归粗略推断复杂度。够用，且不会瞎吹。

    注意这是**整段代码的量级估算**，不是逐函数的严格证明 ——
    在给用户的报告里也要如实表达这一点，宁可保守，不要编造。
    """
    if facts.get("nested_loops"):
        time_desc = "O(n²) —— 存在嵌套循环"
    elif facts.get("sorted_used"):
        time_desc = "O(n log n) —— 主要开销来自排序"
    elif facts.get("loops"):
        time_desc = f"O(n) —— 单层循环（共 {facts['loops']} 处）"
    else:
        time_desc = "O(1) —— 无循环结构"

    if facts.get("has_recursion"):
        recursive = "、".join(facts.get("recursive_functions") or []) or "部分函数"
        time_desc += f"；{recursive} 为递归实现，另有递归深度开销（分治型通常为 O(log n) 层）"
        space_desc = "递归调用栈 O(depth)，外加循环内的临时变量"
    else:
        space_desc = "O(1) 或 O(n)，取决于是否使用额外容器"
    return {"time": time_desc, "space": space_desc}


# ======================================================================================
# 代码生成：常见需求的真实实现（Mock 的"知识库"）
# ======================================================================================
def _codegen_python(description: str) -> str:
    """按关键词匹配一小批常见需求，返回可直接运行的实现。"""
    text = description.lower()

    if any(k in description for k in ("斐波那契", "fib")) and any(k in text for k in ("fib", "斐波那契")):
        return MOCK_MARK + '''
from functools import lru_cache


@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """返回斐波那契数列第 n 项（从 0 开始计数）。

    使用记忆化递归：朴素递归是 O(2^n)，加上缓存后每次子问题只算一次，
    时间复杂度降到 O(n)，代价是 O(n) 的缓存空间。
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_sequence(count: int) -> list[int]:
    """返回前 count 项斐波那契数列（迭代实现，便于处理大 count）。"""
    if count <= 0:
        return []
    sequence = [0, 1][:count]
    while len(sequence) < count:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence


if __name__ == "__main__":
    print(fibonacci_sequence(10))
'''

    if ("排序" in description or "sort" in text) and any(k in description for k in ("快速", "快排", "quick")):
        return MOCK_MARK + '''
def quicksort(items: list[int]) -> list[int]:
    """快速排序（返回新列表，不修改入参）。

    平均时间 O(n log n)，最坏 O(n²)（已排序输入 + 取首个元素为基准时）。
    这里用"随机基准 + 三路划分"来规避最坏情况：
    三路划分把等于基准的元素集中到中间，重复元素很多时性能尤其好。
    """
    import random

    if len(items) <= 1:
        return list(items)
    pivot = random.choice(items)
    less = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items if x > pivot]
    return quicksort(less) + equal + quicksort(greater)


if __name__ == "__main__":
    print(quicksort([3, 1, 4, 1, 5, 9, 2, 6]))
'''

    if "二分" in description or "binary search" in text or "bisect" in text:
        return MOCK_MARK + '''
def binary_search(sorted_items: list[int], target: int) -> int:
    """在升序列表中查找 target，返回下标；找不到返回 -1。

    使用左闭右开区间 [low, high)：这样循环条件、区间收缩、终止状态三者
    形式统一，是最不容易写错边界的写法。
    时间复杂度 O(log n)，空间 O(1)。
    """
    low, high = 0, len(sorted_items)
    while low < high:
        mid = (low + high) // 2
        if sorted_items[mid] == target:
            return mid
        if sorted_items[mid] < target:
            low = mid + 1
        else:
            high = mid
    return -1


if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    print(binary_search(data, 7), binary_search(data, 4))
'''

    if "质数" in description or "prime" in text:
        return MOCK_MARK + '''
def is_prime(n: int) -> bool:
    """判断 n 是否为质数。只试除到 sqrt(n)，复杂度 O(√n)。"""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2
    return True


def primes_up_to(limit: int) -> list[int]:
    """埃拉托斯特尼筛法，返回 limit 以内的所有质数，复杂度 O(n log log n)。"""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\\x00\\x00"
    for number in range(2, int(limit ** 0.5) + 1):
        if sieve[number]:
            sieve[number * number :: number] = bytearray(len(sieve[number * number :: number]))
    return [index for index, flag in enumerate(sieve) if flag]


if __name__ == "__main__":
    print(primes_up_to(30))
'''

    if "去重" in description or "dedup" in text or "unique" in text:
        return MOCK_MARK + '''
def dedupe(items: list) -> list:
    """去重并保持首次出现顺序。

    不能用 set(items)：集合会打乱顺序；也不能用 list(set(...)) 取巧。
    这里用 dict 的插入有序特性，时间 O(n)、空间 O(n)。
    要求元素可哈希（不可哈希的元素会在 seen.add 处抛 TypeError）。
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def dedupe_unhashable(items: list) -> list:
    """支持不可哈希元素（如 dict/list）的去重，退化为 O(n²)。"""
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
'''

    if "反转" in description and ("字符串" in description or "string" in text):
        return MOCK_MARK + '''
def reverse_string(text: str) -> str:
    """反转字符串。

    用切片 text[::-1]：C 层实现，比 Python 层循环快一个数量级。
    注意：这反转的是 Unicode 码点，遇到组合字符（如 emoji、带音标的字母）会拆散字形，
    如需按"字素簇"反转要借助 grapheme 库。
    """
    return text[::-1]


def is_palindrome(text: str) -> bool:
    """判断回文（忽略大小写与非字母数字字符）。"""
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    return cleaned == cleaned[::-1]
'''

    if "词频" in description or "word count" in text or ("统计" in description and "词" in description):
        return MOCK_MARK + '''
import re
from collections import Counter


def word_frequency(text: str, top_n: int = 10) -> list[tuple[str, int]]:
    """统计英文文本词频，返回出现次数最多的 top_n 个 (单词, 次数)。

    用 Counter.most_common 而不是自己排序：它内部用 heapq.nlargest，
    在 top_n 远小于词表规模时是 O(n log k) 而不是 O(n log n)。
    """
    words = re.findall(r"[A-Za-z']+", text.lower())
    return Counter(words).most_common(top_n)


if __name__ == "__main__":
    sample = "the quick brown fox jumps over the lazy dog the fox"
    print(word_frequency(sample, 3))
'''

    if "最大子" in description or "kadane" in text or "max subarray" in text:
        return MOCK_MARK + '''
def max_subarray_sum(numbers: list[int]) -> int:
    """最大子数组和（Kadane 算法）。

    核心思想：以当前位置结尾的最大子数组和，要么是"只取当前元素"，
    要么是"前一个位置的最优解 + 当前元素"。因此只需保留一个状态变量。
    时间 O(n)，空间 O(1)。列表中全为负数时返回最大的那个负数。
    """
    if not numbers:
        raise ValueError("列表不能为空")
    best = current = numbers[0]
    for value in numbers[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best


if __name__ == "__main__":
    print(max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
'''

    if "lru" in text or "缓存" in description:
        return MOCK_MARK + '''
from collections import OrderedDict


class LRUCache:
    """最近最少使用缓存。

    用 OrderedDict 维护访问顺序：访问或写入时把 key 移到末尾，
    容量超限时淘汰链表头部（最久未使用）。所有操作均摊 O(1)。
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self.capacity = capacity
        self._data: "OrderedDict[object, object]" = OrderedDict()

    def get(self, key: object, default: object = None) -> object:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: object, value: object) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
'''

    # ---- 兜底：生成一个结构完整、可直接运行的函数（不含任何占位符）----
    summary = description.strip().split("\n")[0][:60]
    template = '''
def solve(data: list) -> dict:
    """按需求实现：__REQUIREMENT__

    由于离线 Mock 模型不具备真实语言理解能力，这里给出一个结构完整、
    可直接运行的骨架（真实的 LLM 会在此处生成针对需求的实现）。

    参数:
        data: 待处理的输入序列。

    返回:
        dict: 包含 "count"（元素个数）、"unique"（去重后元素）、
              "total"（若元素为数值则为求和结果，否则为 None）。

    异常:
        TypeError: 当 data 不是可迭代对象时抛出。
    """
    try:
        items = list(data)
    except TypeError as exc:
        raise TypeError(f"data 必须是可迭代对象，收到 {type(data).__name__}") from exc

    numeric = [item for item in items if isinstance(item, (int, float)) and not isinstance(item, bool)]
    return {
        "count": len(items),
        "unique": sorted(set(items), key=str),
        "total": sum(numeric) if numeric else None,
    }


if __name__ == "__main__":
    print(solve([3, 1, 4, 1, 5]))
'''
    # 用占位符替换而不是 f-string：模板里的代码包含大量 {}，
    # 一旦用 f-string 就必须全部转义，漏一个就会在生成时被执行 —— 极易埋雷。
    return MOCK_MARK + template.replace("__REQUIREMENT__", summary)


# ======================================================================================
# 意图识别：从 system/user 消息里判断"现在该做什么"
# ======================================================================================
def _detect_task(messages: Sequence[Message], system_text: str) -> str:
    if "代码补全引擎" in system_text:
        return "complete"
    if "注释专家" in system_text:
        return "comment"
    if "修复专家" in system_text:
        return "repair"
    if "代码助手 Agent" in system_text:
        return "agent"
    if "善于把复杂代码讲清楚" in system_text:
        return "explain"
    if "工程师" in system_text and "自然语言需求" in system_text:
        return "generate"
    if "技术导师" in system_text:
        return "qa"
    return "generic"


def _original_task(messages: Sequence[Message]) -> str:
    """取出本次任务的原始用户输入（跳过纠偏提示等系统注入的 user 消息）。"""
    for message in messages:
        if message.role != "user" or not message.content:
            continue
        text = message.content
        if text.startswith("注意：你已经多次重复调用") or text.startswith("已达到工具调用步数上限"):
            continue
        return text
    return ""


def _field(text: str, label: str) -> str:
    """从形如 `标签：值` 的提示词里取值。"""
    match = re.search(rf"{label}\s*[：:]\s*(.+)", text)
    return match.group(1).strip() if match else ""


# ======================================================================================
# 各任务的响应构造
# ======================================================================================
def _respond_generate(user_text: str) -> str:
    description = _field(user_text, "需求描述") or user_text
    language = normalize_language(_field(user_text, "目标语言")) or "python"
    if language != "python":
        return (
            f"{MOCK_MARK}\n```{language}\n"
            f"// 目标语言：{language}（离线 Mock 仅内置 Python 生成模板）\n"
            f"// 需求：{description[:80]}\n"
            "```\n"
        )
    return f"```python\n{_codegen_python(description)}```\n"


def _respond_complete(user_text: str) -> str:
    """补全：只输出缺失的片段。"""
    prefix_match = re.search(r"<prefix>\n?(.*?)</prefix>", user_text, re.DOTALL)
    prefix = prefix_match.group(1) if prefix_match else user_text
    last_line = prefix.rstrip().split("\n")[-1] if prefix.strip() else ""
    match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)\s*(->[^:]+)?\s*:?\s*$", last_line)
    if match:
        name, params, _ = match.group(1), match.group(2), match.group(3)
        arg_names = [p.split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()]
        if name in {"add", "sum_two"} and len(arg_names) == 2:
            return f"\n    return {arg_names[0]} + {arg_names[1]}\n"
        if name.startswith("is_") and len(arg_names) == 1:
            return f'\n    """判断 {arg_names[0]} 是否满足 {name[3:]} 条件。"""\n    return bool({arg_names[0]})\n'
        if len(arg_names) == 1:
            return f'\n    """处理 {arg_names[0]} 并返回结果。"""\n    return {arg_names[0]}\n'
        return "\n    return None\n"
    if last_line.strip().endswith(("+", "=", "return", ",")):
        return "  # 补全：根据上文推断的表达式\n"
    return "    # 补全：上下文不足以确定意图，此处给出最小可用实现\n    return None\n"


def _respond_explain(user_text: str) -> str:
    """解释：输出契约规定的 JSON，字段由静态分析支撑。"""
    blocks = extract_code_blocks(user_text)
    code = blocks[0].code if blocks else user_text
    language = normalize_language(_field(user_text, "语言")) or "python"
    facts = analyze_code(code) if language in ("python", "") else {}

    functions = facts.get("functions", [])
    classes = facts.get("classes", [])
    loops = facts.get("loops", 0)

    if classes:
        summary = f"定义类 {classes[0]['name']}，包含 {len(classes[0]['methods'])} 个方法"
    elif functions:
        names = "、".join(f["name"] for f in functions[:3])
        summary = f"定义函数 {names}，共 {len(functions)} 个函数、{facts.get('lines', 0)} 行代码"
    else:
        summary = f"一段 {facts.get('lines', '若干')} 行的脚本逻辑"

    steps: List[Dict[str, Any]] = []
    order = 1
    for imported in facts.get("imports", [])[:3]:
        steps.append({"step": order, "title": "导入依赖", "detail": f"引入 {imported}，用于后续逻辑"})
        order += 1
    for function in functions[:5]:
        args = "、".join(function["args"]) or "无"
        steps.append(
            {
                "step": order,
                "title": f"定义函数 {function['name']}",
                "detail": f"接收参数：{args}；在第 {function['lineno']} 行定义。",
            }
        )
        order += 1
    if classes:
        for klass in classes[:2]:
            steps.append(
                {
                    "step": order,
                    "title": f"定义类 {klass['name']}",
                    "detail": f"包含方法：{'、'.join(klass['methods']) or '无'}。",
                }
            )
            order += 1
    if loops:
        steps.append(
            {
                "step": order,
                "title": "循环处理",
                "detail": f"包含 {loops} 处循环" + ("，且存在嵌套循环。" if facts.get("nested_loops") else "，为单层遍历。"),
            }
        )

    key_points = []
    if facts.get("has_recursion"):
        key_points.append("使用递归实现，注意递归深度与重复子问题")
    if facts.get("sorted_used"):
        key_points.append("借助排序简化问题，主要开销在 O(n log n) 的排序上")
    if facts.get("comprehensions"):
        key_points.append(f"使用 {facts['comprehensions']} 处推导式，写法紧凑且性能优于显式循环")
    if facts.get("has_try"):
        key_points.append("使用异常处理保证健壮性")
    if facts.get("async_used"):
        key_points.append("包含 async 定义，运行在事件循环中")
    if not key_points:
        key_points.append("逻辑直白，无特殊算法技巧")

    edge_cases = []
    if not functions and not classes:
        edge_cases.append("脚本式代码，无函数封装，复用与测试成本较高")
    if loops:
        edge_cases.append("空输入时循环不执行，需确认这是否符合预期")
    if not facts.get("has_try"):
        edge_cases.append("未做异常处理，非法输入会直接抛出异常")
    if facts.get("syntax_error"):
        edge_cases.append(f"代码无法解析：{facts['syntax_error']}")
    if not edge_cases:
        edge_cases.append("未发现明显的边界风险")

    issues = [
        {"severity": item["severity"], "description": item["pattern"], "suggestion": "按描述调整实现"}
        for item in facts.get("issues", [])
    ]

    payload = {
        "summary": summary,
        "purpose": (
            "该代码用于完成一项独立的计算/处理任务，可作为函数被其他模块调用；"
            f"整体规模 {facts.get('lines', 0)} 行，包含 {len(functions)} 个函数、{len(classes)} 个类。"
        ),
        "logic_steps": steps or [{"step": 1, "title": "顺序执行", "detail": "按行顺序执行表达式语句。"}],
        "key_points": key_points,
        "edge_cases": edge_cases,
        "issues": issues,
        "complexity": guess_complexity(facts) if facts else {"time": "无法判断", "space": "无法判断"},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _respond_comment(user_text: str) -> str:
    """注释：用 ast 重写代码，为每个函数/类补上 docstring。"""
    blocks = extract_code_blocks(user_text)
    code = blocks[0].code if blocks else ""
    language = normalize_language(_field(user_text, "语言")) or "python"
    if not code.strip():
        return f"```{language or 'python'}\n{MOCK_MARK}\n# 未检测到待注释代码\n```"

    if language not in ("python", ""):
        return f"```{language}\n// {MOCK_MARK}\n{code}\n```"

    annotated = _annotate_python(code)
    return f"```python\n{annotated}\n```\n"


def _annotate_python(code: str) -> str:
    """基于 AST 为缺失 docstring 的函数/类补写文档注释。

    用真实 AST 改写而不是字符串拼接，好处是**绝不会破坏原代码的语法**。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"{MOCK_MARK}\n# （原代码无法解析，未做 AST 改写）\n{code}"

    changed = False

    class Annotator(ast.NodeTransformer):
        def _annotate(self, node: ast.AST, kind: str) -> ast.AST:
            nonlocal changed
            body = getattr(node, "body", None)
            if not body:
                return node
            existing = body[0]
            if isinstance(existing, ast.Expr) and isinstance(existing.value, ast.Constant) and isinstance(existing.value.value, str):
                return node  # 已有 docstring，不覆盖

            if kind == "class":
                doc = (
                    f"{node.name}：负责本模块中与 {node.name.lower()} 相关的职责。\n\n"
                    "维护该类型的不变式（invariant），对外暴露稳定的方法接口。"
                )
            else:
                args = [a.arg for a in node.args.args if a.arg not in {"self", "cls"}]
                arg_text = "、".join(f"``{a}``" for a in args) if args else "无"
                doc = (
                    f"{node.name}：执行 {node.name.replace('_', ' ')} 相关逻辑。\n\n"
                    f"参数：{arg_text}。\n"
                    "返回：处理结果；具体类型见函数签名。"
                )
            docstring = ast.Expr(value=ast.Constant(value=doc))
            ast.copy_location(docstring, existing)
            node.body = [docstring, *body]
            changed = True
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            self.generic_visit(node)
            if node.name.startswith("_") and not node.name.endswith("__"):
                return node  # 私有函数不写文档，减少噪音
            return self._annotate(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            self.generic_visit(node)
            return self._annotate(node, "function")

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            self.generic_visit(node)
            return self._annotate(node, "class")

    tree = Annotator().visit(tree)
    ast.fix_missing_locations(tree)
    if not changed:
        return f"{MOCK_MARK}\n# （原代码已包含完整文档字符串，未新增注释）\n{code}"

    try:
        regenerated = ast.unparse(tree)
    except Exception:  # pragma: no cover
        return f"{MOCK_MARK}\n{code}"
    header = (
        f"{MOCK_MARK}\n"
        "# 说明：注释由 AST 改写生成，因此原有行内注释会丢失；\n"
        "#      真实模型会在保留原注释的前提下，按 Why-而非-What 的原则补充说明。"
    )
    return f"{header}\n\n{regenerated}"


def _respond_qa(user_text: str) -> str:
    question = _field(user_text, "问题").split("\n")[0] or user_text[:60]
    return (
        f"**结论**：针对「{question}」，需要结合具体代码上下文才能给出确定答案；"
        "以下为离线 Mock 模型能提供的一般性说明。\n\n"
        "**依据**：\n"
        "1. 判断代码行为最可靠的方式是实际运行它 —— 可以让我调用 `run_python` 工具验证。\n"
        "2. 若涉及具体文件，可以让我用 `read_file` / `search_code` 定位实现。\n\n"
        "**下一步**：把相关代码贴出来，或直接说「读取 xxx.py 并解释第 20-40 行」。\n\n"
        "> 提示：当前使用的是内置 Mock 离线模型，接入真实 API（`--provider deepseek`）后"
        "会得到真正基于代码的回答。"
    )


# ======================================================================================
# Agent 工具调用：离线也能演示完整 ReAct 循环
# ======================================================================================
def _plan_tools(task: str, available: set) -> List[tuple]:
    """根据任务描述规划一组工具调用（模拟模型的"决策"）。"""
    plan: List[tuple] = []

    if "list_dir" in available and any(k in task for k in ("目录", "结构", "项目里", "有哪些文件", "文件列表")):
        plan.append(("list_dir", {"path": "."}))

    file_match = re.search(r"([\w./\\-]+\.\w{1,5})", task)
    if "read_file" in available and file_match and any(
        k in task for k in ("读取", "打开", "查看", "分析", "解释", "看看", "内容", "注释", "优化")
    ):
        plan.append(("read_file", {"path": file_match.group(1).replace("\\", "/"), "start_line": 0, "end_line": 0}))

    if "search_code" in available and any(k in task for k in ("搜索", "查找", "哪里定义", "在哪里", "定位", "都用到了")):
        keyword_match = re.search(r"[「\"']([^」\"']{2,30})[」\"']", task)
        keyword = keyword_match.group(1) if keyword_match else "def "
        plan.append(("search_code", {"pattern": keyword, "path": ".", "file_glob": ""}))

    if "run_python" in available and any(k in task for k in ("运行", "执行", "验证", "测试一下", "跑一下", "对不对", "是否正确")):
        plan.append(("run_python", {"code": "print('hello from sandbox')\nprint(sum(range(10)))\n"}))

    if "check_python_syntax" in available and file_match and "语法" in task:
        plan.append(("check_python_syntax", {"code": "def ok():\n    return 1\n"}))

    return plan


def _summarize_tool_results(messages: Sequence[Message], task: str) -> str:
    """把工具调用结果汇总成最终答案。

    只汇总**最后一轮提问之后**产生的工具消息：
    多轮对话时，历史里还留着上一轮的工具结果，把它们也算进来会让回答变成
    "把之前查过的所有东西再复述一遍"，答非所问。
    """
    last_user_index = 0
    for index, message in enumerate(messages):
        if message.role == "user":
            last_user_index = index
    tool_messages = [m for m in messages[last_user_index:] if m.role == "tool"]

    if not tool_messages:
        return "未能获取到任何工具结果。"

    lines = ["**结论**", ""]
    failed = [m for m in tool_messages if (m.content or "").startswith("[工具执行失败]")]
    succeeded = [m for m in tool_messages if m not in failed]

    if not succeeded and failed:
        lines.append("所有工具调用都失败了，因此无法完成任务。失败详情：")
        for message in failed:
            lines.append(f"- {message.content}")
        lines.append("")
        lines.append("**下一步建议**：请确认文件路径是否正确，或调整任务描述。")
        return "\n".join(lines)

    lines.append(f"我调用了 {len(tool_messages)} 次工具（成功 {len(succeeded)} 次）来完成任务："
                 f"「{task.strip().splitlines()[0][:60]}」。")
    lines.append("")

    for index, message in enumerate(tool_messages, 1):
        name = message.name or "tool"
        body = (message.content or "").strip()
        preview = "\n".join(body.split("\n")[:12])
        lines.append(f"**步骤 {index} · `{name}` 返回：**")
        lines.append("")
        lines.append("```")
        lines.append(preview)
        lines.append("```")
        lines.append("")

    if failed:
        lines.append(f"其中 {len(failed)} 次调用失败，已在上面列出原因。")
        lines.append("")
    lines.append(
        "> 以上为内置 Mock 离线模型的汇总行为：它会真实地调用工具并回填结果，"
        "但不会做深度推理。接入真实 LLM 后，这一步会变成基于证据的分析。"
    )
    return "\n".join(lines)


# ======================================================================================
# 主入口：heuristic_handler
# ======================================================================================
def heuristic_handler(
    messages: Sequence[Message], tools: Optional[List[Dict[str, Any]]] = None
) -> ChatResponse:
    """MockLLM 的默认处理器：按规则决定"回答"还是"调用工具"。"""
    system_text = next((m.content or "" for m in messages if m.role == "system"), "")
    task = _detect_task(messages, system_text)
    user_text = next((m.content or "" for m in reversed(messages) if m.role == "user"), "")

    # ---------- 情况 1：Agent 循环中，上一条是工具结果 ----------
    if messages and messages[-1].role == "tool":
        original = _original_task(messages)
        available = {t["function"]["name"] for t in (tools or [])}
        plan = _plan_tools(original, available)
        called = {m.name for m in messages if m.role == "tool" and m.name}
        remaining = [item for item in plan if item[0] not in called]
        if remaining:
            name, args = remaining[0]
            index = len([m for m in messages if m.role == "tool"])
            return ChatResponse(
                content=f"继续执行计划的第 {index + 1} 步：调用 {name}。",
                tool_calls=[
                    ToolCall(id=f"mock_call_{index + 1}", name=name, arguments=json.dumps(args, ensure_ascii=False))
                ],
                finish_reason="tool_calls",
            )
        return ChatResponse(content=_summarize_tool_results(messages, original), finish_reason="stop")

    # ---------- 情况 2：指定了工具，且任务确实需要动手 ----------
    if task == "agent" and tools:
        available = {t["function"]["name"] for t in tools}
        plan = _plan_tools(user_text, available)
        if plan:
            name, args = plan[0]
            return ChatResponse(
                content=f"这个任务需要先获取实际信息，我打算调用 {name}。",
                tool_calls=[ToolCall(id="mock_call_1", name=name, arguments=json.dumps(args, ensure_ascii=False))],
                finish_reason="tool_calls",
            )

    # ---------- 情况 3：纯文本任务（生成 / 补全 / 解释 / 注释 / 问答）----------
    if task == "generate":
        content = _respond_generate(user_text)
    elif task == "complete":
        content = _respond_complete(user_text)
    elif task == "explain":
        content = _respond_explain(user_text)
    elif task == "comment":
        content = _respond_comment(user_text)
    elif task == "repair":
        blocks = extract_code_blocks(user_text)
        code = blocks[0].code if blocks else ""
        content = f"```python\n{MOCK_MARK}\n# 已按报错信息修正（Mock 直接回显原始代码）\n{code}\n```\n"
    elif task == "qa":
        content = _respond_qa(user_text)
    else:
        content = (
            "这是内置 Mock 离线模型的通用回复。请使用具体命令体验各项能力，例如：\n\n"
            "```bash\n"
            "py -m codeassist generate \"实现一个判断质数的函数\"\n"
            "py -m codeassist explain --file codeassist/parsers.py\n"
            "py -m codeassist agent \"列出目录结构并读取 README.md\"\n"
            "```"
        )
    return ChatResponse(content=content, finish_reason="stop")


# ======================================================================================
# 脚本化模型：单元测试专用
# ======================================================================================
def scripted_handler(
    responses: Sequence[str], *, repeat_last: bool = True, tool_calls: Optional[Sequence[List[ToolCall]]] = None
) -> Callable[..., ChatResponse]:
    """按剧本依次返回预设响应。

    两个剧本维度**彼此独立**，这点很关键：
        responses[i]   —— 第 i 次「纯文本回复」的内容，按顺序消费
        tool_calls[i]  —— 第 i 次「整体 LLM 调用」是否返回工具调用

    早期版本让文本响应也按"全局调用序号"取值，结果只要中间夹了工具调用，
    文本就会错位（第 1 轮的回答取到了第 2 轮的文本）。这个 bug 是写
    AgentSession 的连续追问测试时才暴露出来的。

    测试的关键原则：**测试要确定性的**。
    用真模型写断言 = 用随机数写断言。ScriptedLLM 让"模型返回了坏 JSON""模型连续
    两次调用同一个工具"这类边界场景可以被精确构造出来。
    """
    queue = list(responses)
    calls = list(tool_calls or [])
    state = {"call": 0, "text": 0}

    def handler(messages: Sequence[Message], tools: Any) -> ChatResponse:
        call_index = state["call"]
        state["call"] = call_index + 1

        if call_index < len(calls) and calls[call_index]:
            return ChatResponse(content="", tool_calls=list(calls[call_index]), finish_reason="tool_calls")

        text_index = state["text"]
        state["text"] = text_index + 1
        if text_index < len(queue):
            return ChatResponse(content=queue[text_index], finish_reason="stop")
        if repeat_last and queue:
            return ChatResponse(content=queue[-1], finish_reason="stop")
        raise AssertionError(
            f"ScriptedLLM 剧本已用尽：已请求第 {text_index + 1} 条文本响应，但只提供了 {len(queue)} 条。"
            "（这通常说明 Agent 循环比预期多跑了几轮）"
        )

    return handler


def scripted_llm(responses: Sequence[str], **kwargs: Any) -> BaseLLM:
    """构造一个脚本化的 MockLLM，供测试直接使用。"""
    return MockLLM(scripted_handler(responses, **kwargs))


def make_mock_llm(config: Optional[Config] = None) -> BaseLLM:
    return MockLLM(model=(config or Config()).resolved_model)
