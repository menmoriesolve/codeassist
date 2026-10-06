"""命令行入口：把五大能力暴露成可组合的命令。

    py -m codeassist generate "实现一个 LRU 缓存" --lang python
    py -m codeassist complete --file demo.py --line 12
    py -m codeassist explain --file codeassist/parsers.py
    py -m codeassist comment --file demo.py -o demo_commented.py
    py -m codeassist ask "这段代码的时间复杂度是多少？" --file demo.py
    py -m codeassist agent "列出目录结构，然后读取 README.md 的前 40 行"
    py -m codeassist demo          # 离线全流程演示（无需 API Key）
    py -m codeassist providers     # 查看可用的模型服务商

设计上刻意保留的"教学开关"：
    --dry-run      只打印将要发给模型的原始请求体，不真的调用
    --show-prompt  打印每次调用的完整 messages
    --mock         强制使用离线模型
这些开关让你能在不花钱的情况下，把"模型到底收到了什么"看得一清二楚。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .assistant import CodeAssistant
from .config import PRESETS, Config, load_config
from .llm import BaseLLM, ChatResponse, get_llm
from .parsers import detect_language
from .tools import build_default_registry


# ======================================================================================
# 终端输出小工具（不引入任何第三方库）
# ======================================================================================
class C:
    """ANSI 颜色。若终端不支持或设置了 NO_COLOR，则自动降级为空字符串。"""

    enabled = False
    RESET = BOLD = DIM = CYAN = GREEN = YELLOW = RED = MAGENTA = ""

    @classmethod
    def setup(cls, force: Optional[bool] = None) -> None:
        supported = force
        if supported is None:
            supported = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        if not supported:
            return
        if os.name == "nt":  # 让 Windows 终端识别 ANSI 转义序列
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                return
        cls.enabled = True
        cls.RESET, cls.BOLD, cls.DIM = "\033[0m", "\033[1m", "\033[2m"
        cls.CYAN, cls.GREEN, cls.YELLOW = "\033[36m", "\033[32m", "\033[33m"
        cls.RED, cls.MAGENTA = "\033[31m", "\033[35m"


def _rule(title: str = "") -> None:
    line = "─" * 72
    if title:
        print(f"\n{C.CYAN}{C.BOLD}▍{title}{C.RESET}")
    else:
        print(f"{C.DIM}{line}{C.RESET}")


def _info(message: str) -> None:
    print(f"{C.DIM}{message}{C.RESET}")


def _warn(message: str) -> None:
    print(f"{C.YELLOW}⚠ {message}{C.RESET}")


def _error(message: str) -> None:
    print(f"{C.RED}✖ {message}{C.RESET}", file=sys.stderr)


# ======================================================================================
# 观测包装：把"模型收到了什么、返回了什么"显式打印出来
# ======================================================================================
class TracingLLM(BaseLLM):
    """装饰器模式的又一次应用：给任意 LLM 套上"打印进出内容"的能力。

    为什么要把这个放在客户端而不是业务代码里？
        因为业务代码不该关心"怎么观测"。用包装器实现横切关注点（AOP 思想），
        业务层就能保持干净，而观测能力可以对任意 Provider 生效。
    """

    def __init__(self, inner: BaseLLM, *, show_prompt: bool = False, dry_run: bool = False):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.show_prompt = show_prompt
        self.dry_run = dry_run
        self.call_index = 0

    def chat(self, messages, *, tools=None, temperature=None, **kwargs) -> ChatResponse:
        self.call_index += 1
        payload = self.inner.build_payload(
            messages, tools=tools, temperature=temperature, max_tokens=kwargs.get("max_tokens")
        )
        _rule(f"LLM 调用 #{self.call_index} · provider={self.name} · model={self.model}")
        if self.show_prompt or self.dry_run:
            print(f"{C.MAGENTA}▸ 请求 messages（{len(messages)} 条）：{C.RESET}")
            for index, message in enumerate(payload["messages"], 1):
                role = message.get("role")
                color = {"system": C.YELLOW, "user": C.CYAN, "assistant": C.GREEN, "tool": C.MAGENTA}.get(role, "")
                print(f"{color}[{index}] {role}{C.RESET}")
                content = message.get("content") or ""
                for line in str(content).split("\n"):
                    print(f"    {line}")
                if message.get("tool_calls"):
                    print(f"    {C.DIM}tool_calls: {json.dumps(message['tool_calls'], ensure_ascii=False)}{C.RESET}")
            if tools:
                names = [t["function"]["name"] for t in tools]
                print(f"{C.MAGENTA}▸ 声明工具：{names}{C.RESET}")
        if self.dry_run:
            print(f"{C.YELLOW}（--dry-run：以上请求已打印，未真正发送）{C.RESET}")
            return ChatResponse(content="```python\n# dry-run：未调用模型\n```", finish_reason="stop")

        response = self.inner.chat(messages, tools=tools, temperature=temperature, **kwargs)
        stats = (
            f"{C.DIM}← {response.latency_ms:.0f}ms · tokens={response.usage.total_tokens} "
            f"· finish={response.finish_reason}{C.RESET}"
        )
        print(stats)
        if response.tool_calls:
            for call in response.tool_calls:
                print(f"  {C.GREEN}→ 请求调用工具 {call.name}({call.arguments}){C.RESET}")
        elif response.content:
            preview = response.content.strip().split("\n")[0][:100]
            print(f"  {C.GREEN}→ {preview}{C.RESET}")
        return response

    def stream(self, messages, **kwargs):
        return self.inner.stream(messages, **kwargs)


# ======================================================================================
# 参数解析
# ======================================================================================
#: 全局参数清单。定义一次，注册到主解析器和每个子解析器上。
#: 为什么两边都要注册？因为用户既会写 `codeassist --mock generate "..."`，
#: 也会写 `codeassist generate "..." --mock` —— 只接受其中一种的 CLI 就是坏体验。
#: 关键技巧：子解析器上用 default=argparse.SUPPRESS，
#: 这样"子命令里没写这个参数"时不会往 namespace 里塞 None，从而不会覆盖主解析器解析出的值。
GLOBAL_FLAGS: List[tuple] = [
    (("-v", "--verbose"), {"action": "store_true", "help": "打印执行轨迹等细节"}),
    (("--provider",), {"help": f"模型服务商，可选：{', '.join(sorted(PRESETS))}"}),
    (("--model",), {"help": "模型名，如 deepseek-chat / gpt-4o-mini"}),
    (("--api-key",), {"help": "API Key（也可用环境变量）"}),
    (("--base-url",), {"help": "自定义 API 基地址（接入自建/私有化部署时使用）"}),
    (("--mock",), {"action": "store_true", "help": "强制使用内置离线模型（无需密钥与网络）"}),
    (("--temperature",), {"type": float, "help": "采样温度，代码任务建议 0~0.3"}),
    (("--max-tokens",), {"type": int, "help": "单次生成的最大 token 数"}),
    (("--timeout",), {"type": float, "help": "单次请求超时秒数"}),
    (("--config",), {"help": "配置文件路径（默认自动查找 .codeassist.json）"}),
    (("--workspace",), {"help": "工作区根目录，工具的文件访问被限制在此目录内"}),
    (("--max-steps",), {"type": int, "help": "Agent 最大工具调用轮数"}),
    (("--no-execute",), {"action": "store_true", "help": "禁用 run_python 工具（更安全）"}),
    (("--show-prompt",), {"action": "store_true", "help": "打印每次调用的完整提示词"}),
    (("--dry-run",), {"action": "store_true", "help": "只组装并打印请求，不真正调用模型"}),
    (("--json",), {"action": "store_true", "help": "以 JSON 输出结果（便于脚本消费）"}),
    (("--no-color",), {"action": "store_true", "help": "关闭彩色输出"}),
]


def add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """把全局参数注册到某个解析器上。

    suppress=True 时用 argparse.SUPPRESS 作为默认值：参数没出现就**不写入** namespace，
    因此不会用 None 覆盖主解析器已经解析到的值。
    这是 argparse 子命令架构里最经典的坑之一：子解析器的默认值会静默覆盖父级的解析结果。
    """
    group = parser if suppress else parser.add_argument_group("全局参数")
    for names, kwargs in GLOBAL_FLAGS:
        options = dict(kwargs)
        if suppress:
            options["default"] = argparse.SUPPRESS
        group.add_argument(*names, **options)


def _add_context_slicing(parser: argparse.ArgumentParser, function_help: str) -> None:
    """给子命令加上"只取文件的一部分当上下文"的三个参数。

    这是代码类 Agent 最实用的一个开关：绝大多数问题只针对某一个函数，
    把整个文件塞进上下文既费钱又会稀释模型的注意力。
    """
    parser.add_argument("--start", type=int, default=0, help="起始行（1 起始）")
    parser.add_argument("--end", type=int, default=0, help="结束行")
    parser.add_argument("--function", help=function_help)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeassist",
        description="代码助手 Agent：生成 / 补全 / 解释 / 注释 / 问答 / 工具调用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  py -m codeassist demo\n"
            "  py -m codeassist generate \"写一个二分查找\" --lang python\n"
            "  py -m codeassist explain --file main.py --level detailed\n"
            "  py -m codeassist agent \"读取 README.md 并总结项目用途\"\n"
            "\n全局参数既可写在子命令之前，也可写在子命令之后。\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"codeassist {__version__}")
    add_global_flags(parser, suppress=False)

    sub = parser.add_subparsers(dest="command", metavar="<命令>")

    def add_subparser(*names: str, **kwargs: Any) -> argparse.ArgumentParser:
        """统一为每个子命令补上全局参数，使其也能写在子命令之后。"""
        child = sub.add_parser(*names, **kwargs)
        add_global_flags(child, suppress=True)
        return child


    # ---- generate ----
    p_gen = add_subparser("generate", aliases=["gen", "g"], help="自然语言 → 代码")
    p_gen.add_argument("description", help="用自然语言描述你的需求")
    p_gen.add_argument("--lang", default="python", help="目标语言，默认 python")
    p_gen.add_argument("-c", "--constraint", action="append", default=[], help="附加约束，可重复")
    p_gen.add_argument("--no-fewshot", action="store_true", help="关闭 few-shot 示例（用于对比效果）")
    p_gen.add_argument("--no-repair", action="store_true", help="关闭生成后的自修复校验")
    p_gen.add_argument("-o", "--output", help="把生成的代码写入文件")

    # ---- complete ----
    p_cmp = add_subparser("complete", aliases=["cmp"], help="代码补全")
    p_cmp.add_argument("--prefix", help="光标前的代码文本")
    p_cmp.add_argument("--suffix", default="", help="光标后的代码文本")
    p_cmp.add_argument("--file", help="从文件读取上下文")
    p_cmp.add_argument("--line", type=int, default=0, help="光标所在行号（1 起始），默认文件末尾")
    p_cmp.add_argument("--lang", default="python", help="语言，默认 python")

    # ---- explain ----
    p_exp = add_subparser("explain", aliases=["exp", "e"], help="解释代码逻辑")
    p_exp.add_argument("--file", help="代码文件路径")
    p_exp.add_argument("--code", help="直接给定代码字符串")
    p_exp.add_argument("--lang", default="", help="语言，留空自动识别")
    p_exp.add_argument("--level", default="normal", choices=["brief", "normal", "detailed"], help="讲解深度")
    p_exp.add_argument("--start", type=int, default=0, help="起始行（1 起始）")
    p_exp.add_argument("--end", type=int, default=0, help="结束行")
    p_exp.add_argument("--function", help="只解释该函数/类（可用 Object.method 限定）")
    p_exp.add_argument("--raw", action="store_true", help="输出模型的原始回复（不渲染 Markdown 报告）")

    # ---- comment ----
    p_cmt = add_subparser("comment", aliases=["cmt", "c"], help="生成注释 / 文档字符串")
    p_cmt.add_argument("--file", help="代码文件路径")
    p_cmt.add_argument("--code", help="直接给定代码字符串")
    p_cmt.add_argument("--lang", default="", help="语言，留空自动识别")
    p_cmt.add_argument("--comment-lang", default="中文", help="注释语言，默认中文")
    p_cmt.add_argument("-o", "--output", help="把带注释的代码写入文件")
    _add_context_slicing(p_cmt, "只给该函数/类加注释（可用 Object.method 限定）")

    # ---- ask ----
    p_ask = add_subparser("ask", aliases=["qa"], help="代码问答")
    p_ask.add_argument("question", nargs="?", help="你的问题；配合 --chat 时可省略，进入对话后再输入")
    p_ask.add_argument("--file", help="相关代码文件（作为上下文一起提交）")
    p_ask.add_argument("--code", help="直接给定相关代码")
    p_ask.add_argument("--lang", default="", help="上下文代码的语言")
    p_ask.add_argument("--chat", action="store_true", help="进入多轮对话模式")
    _add_context_slicing(p_ask, "只把该函数/类作为上下文（强烈推荐：比整个文件省 token 且答得更准）")

    # ---- agent ----
    p_agt = add_subparser("agent", aliases=["a"], help="带工具调用的 Agent（可读写文件、执行代码）")
    p_agt.add_argument("task", nargs="?", help="任务描述；省略或加 --chat 则进入连续追问模式")
    p_agt.add_argument("--tools", help="只启用指定工具，逗号分隔，如 read_file,list_dir")
    p_agt.add_argument("--trace", action="store_true", help="打印完整执行轨迹（Markdown）")
    p_agt.add_argument(
        "-i", "--chat", action="store_true",
        help="进入连续追问模式（可多轮对话，跨轮记住读过哪些文件）",
    )

    # ---- chat：agent --chat 的快捷方式，便于发现和使用 ----
    p_chat = add_subparser("chat", aliases=["ch"], help="与 Agent 连续对话（它可以自己读文件、跑代码）")
    p_chat.add_argument("task", nargs="?", help="可选：进入对话后先问的第一个问题")
    p_chat.add_argument("--tools", help="只启用指定工具，逗号分隔")
    p_chat.add_argument("--trace", action="store_true", help="每轮都打印执行轨迹")

    # ---- 其他 ----
    add_subparser("demo", help="离线全流程演示（无需 API Key）")
    add_subparser("providers", help="列出内置的模型服务商预设")
    add_subparser("prompts", help="列出所有 Prompt 模板及其变量")

    return parser


# ======================================================================================
# 辅助函数
# ======================================================================================
def _resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(
        config_file=args.config,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        workspace=args.workspace,
        max_steps=args.max_steps,
        verbose=True if args.verbose else None,
    )
    if args.mock:
        config = config.child(provider="mock")
    if args.no_execute:
        config = config.child(allow_execute=False)
    return config


def _make_assistant(config: Config, args: argparse.Namespace) -> CodeAssistant:
    if args.dry_run and config.preset.requires_key and not config.resolved_api_key:
        # --dry-run 的整个意义就是"还没配好密钥时，先看看请求长什么样"。
        # 它不会真的发请求，所以不该被密钥校验挡住 —— 用一个占位密钥放行。
        config = config.child(api_key="<DRY-RUN>")
    llm: BaseLLM = get_llm(config)
    if args.show_prompt or args.dry_run or args.verbose:
        llm = TracingLLM(llm, show_prompt=args.show_prompt, dry_run=args.dry_run)
    tools = build_default_registry(config.workspace, allow_execute=config.allow_execute)
    return CodeAssistant(llm, config, tools=tools)


def _read_code(args: argparse.Namespace) -> tuple:
    """从 --code / --file 读取代码，返回 (代码, 语言, 来源描述)。

    支持三种"缩小上下文"的方式，优先级从高到低：
        --function NAME   按符号名精确抽取一个函数/类（最推荐）
        --start/--end     按行范围截取
        都不给            整个文件
    """
    if getattr(args, "code", None):
        code = args.code
        return code, detect_language(code), "<命令行参数>"
    file_path = getattr(args, "file", None)
    if not file_path:
        raise SystemExit("请通过 --file 或 --code 提供代码")
    path = Path(file_path)
    if not path.is_file():
        raise SystemExit(f"文件不存在：{path}")
    # utf-8-sig：带 BOM 的文件若按 utf-8 读，首字符会是 \ufeff，
    # 直接丢给 ast.parse 会报"非法字符"，让用户一头雾水。
    text = path.read_text(encoding="utf-8-sig")
    total_lines = text.count("\n") + 1
    language = getattr(args, "lang", "") or detect_language(text, filename=str(path))

    # --- 方式一：按函数/类名精确抽取 ---
    symbol_name = getattr(args, "function", None)
    if symbol_name:
        from .parsers import extract_symbol, list_symbols

        try:
            symbol = extract_symbol(text, symbol_name)
        except ValueError as exc:  # 有歧义的短名
            raise SystemExit(f"✖ {exc}")
        if symbol is None:
            available = list_symbols(text)
            hint = (
                "文件里可用的符号：\n  " + "\n  ".join(available)
                if available
                else "（该文件里没有解析到任何函数或类定义）"
            )
            raise SystemExit(
                f"✖ 在 {path} 里没找到符号 {symbol_name!r}。\n{hint}\n"
                "提示：可用 --function 指定函数名，或用 --start/--end 指定行范围。"
            )
        header = f"# 摘自 {path} 第 {symbol.start_line}-{symbol.end_line} 行（{symbol.kind}: {symbol.name}）"
        return f"{header}\n{symbol.code}", language, f"{path} :: {symbol.name}"

    # --- 方式二：按行范围截取 ---
    lines = text.split("\n")
    start = getattr(args, "start", 0) or 1
    end = getattr(args, "end", 0) or len(lines)
    if start > 1 or end < len(lines):
        if start > len(lines):
            raise SystemExit(f"✖ --start {start} 超出文件总行数 {len(lines)}")
        text = "\n".join(lines[start - 1 : end])
        return text, language, f"{path} 第 {start}-{end} 行（共 {total_lines} 行）"
    return text, language, str(path)


def _dump_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _usage_line(config: Config, assistant: CodeAssistant) -> None:
    usage = assistant.total_usage
    _info(
        f"—— provider={config.provider} model={config.resolved_model} "
        f"| tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
        f"total={usage.total_tokens}"
    )


# ======================================================================================
# 各子命令实现
# ======================================================================================
def cmd_generate(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    result = assistant.generate(
        args.description,
        language=args.lang,
        constraints=args.constraint,
        few_shot=not args.no_fewshot,
        repair=not args.no_repair,
    )
    if args.json:
        _dump_json(
            {
                "code": result.code,
                "language": result.language,
                "explanation": result.explanation,
                "problems": result.problems,
                "attempts": result.attempts,
                "repaired": result.repaired,
                "usage": {"total_tokens": result.usage.total_tokens},
            }
        )
    else:
        _rule(f"生成结果 · {result.language}")
        print(result.to_markdown())
        if result.ok:
            _info(f"本地校验通过（尝试 {result.attempts} 次）")
        else:
            _warn(f"本地校验仍有问题：{'；'.join(result.problems)}")

    if args.output and result.code.strip() and not args.dry_run:
        Path(args.output).write_text(result.code + "\n", encoding="utf-8")
        _info(f"已写入 {args.output}")
    _usage_line(config, assistant)
    return 0 if result.ok else 2


def cmd_complete(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    if args.file and not args.prefix:
        path = Path(args.file)
        if not path.is_file():
            raise SystemExit(f"文件不存在：{path}")
        lines = path.read_text(encoding="utf-8").split("\n")
        cursor = args.line or len(lines)
        prefix = "\n".join(lines[:cursor])
        suffix = "\n".join(lines[cursor:])
    else:
        prefix = args.prefix or ""
        suffix = args.suffix
    if not prefix.strip():
        raise SystemExit("请通过 --prefix 或 --file 提供光标前的代码")

    result = assistant.complete(prefix, suffix, language=args.lang)
    if args.json:
        _dump_json({"completion": result.completion, "language": result.language})
    else:
        _rule("补全结果（仅新增部分）")
        print(result.completion)
    _usage_line(config, assistant)
    return 0


def cmd_explain(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    code, language, source = _read_code(args)
    result = assistant.explain(code, language=language, level=args.level)
    if args.json:
        _dump_json(
            {
                "summary": result.summary,
                "purpose": result.purpose,
                "logic_steps": result.logic_steps,
                "key_points": result.key_points,
                "edge_cases": result.edge_cases,
                "issues": result.issues,
                "complexity": result.complexity,
                "structured": result.structured,
            }
        )
    else:
        _rule(f"代码解释 · {source} · 语言={language or '未知'} · 深度={args.level}")
        print(result.text if args.raw else result.to_markdown())
        if not result.structured and not args.raw:
            _warn("模型未返回结构化 JSON，已降级为原始文本展示。")
    _usage_line(config, assistant)
    return 0


def cmd_comment(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    code, language, source = _read_code(args)
    result = assistant.comment(code, language=language, comment_language=args.comment_lang)
    if args.json:
        _dump_json({"code": result.code, "notes": result.notes, "language": result.language})
    else:
        _rule(f"注释结果 · {source}")
        print(result.to_markdown())
    if args.output and not args.dry_run:
        Path(args.output).write_text(result.code + "\n", encoding="utf-8")
        _info(f"已写入 {args.output}")
    _usage_line(config, assistant)
    return 0


def cmd_ask(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    code, language = "", ""
    if args.file or args.code:
        code, language, _ = _read_code(args)

    if not args.chat:
        if not args.question:
            raise SystemExit(
                '请提供问题，例如：codeassist ask "这段代码做什么" --file app.py\n'
                "（想连续追问请加 --chat，或直接用 codeassist chat）"
            )
        answer = assistant.ask(args.question, code=code, language=language)
        if args.json:
            _dump_json({"answer": answer.answer, "used_context": answer.used_context})
        else:
            _rule("回答")
            print(answer.answer)
        _usage_line(config, assistant)
        return 0

    # 多轮对话模式：问题可以写在命令行上，也可以进去再打字
    _rule("多轮问答模式（纯问答、无工具）")
    _info(f"上下文代码：{len(code)} 字符" if code else "未加载代码上下文（可用 --file 指定）")
    _info("输入 exit / quit 退出，Ctrl+C 中断。需要它自己读文件请改用 codeassist chat")
    print()
    try:
        if args.question:
            first = assistant.chat(args.question, code=code)
            print(f"{C.GREEN}助手：{C.RESET}{first.answer}\n")
        while True:
            try:
                user_input = input(f"{C.CYAN}你：{C.RESET}").strip()
            except EOFError:
                break
            if user_input.lower() in {"exit", "quit", ":q"}:
                break
            if not user_input:
                continue
            reply = assistant.chat(user_input)
            print(f"\n{C.GREEN}助手：{C.RESET}{reply.answer}\n")
    except KeyboardInterrupt:
        print()
    _usage_line(config, assistant)
    return 0


def _agent_event_printer(args: argparse.Namespace) -> Any:
    """构造一个把 Agent 动作实时打印到终端的事件回调。

    可观测性不应该是"事后看日志"，而应该是"过程可见" ——
    用户能亲眼看到它在读哪个文件、为什么失败，信任感完全不同。
    """

    def on_event(event: str, payload: Dict[str, Any]) -> None:
        if args.json:
            return
        if event == "tool_start":
            print(f"  {C.GREEN}→ {payload['name']}{C.RESET} {C.DIM}{payload['arguments']}{C.RESET}")
        elif event == "tool_end":
            if payload["ok"]:
                preview = (payload.get("output") or "").strip().split("\n")[:4]
                print(f"    {C.DIM}✓ {len(payload.get('output') or '')} 字符{C.RESET}")
                for line in preview:
                    print(f"      {C.DIM}{line[:100]}{C.RESET}")
            else:
                print(f"    {C.YELLOW}✗ {payload['error'][:160]}{C.RESET}")
        elif event == "nudge":
            _warn(payload["message"])
        elif event == "max_steps":
            _warn(f"达到最大步数 {payload['steps']}，将强制总结")

    return on_event


def _build_tool_registry(assistant: CodeAssistant, args: argparse.Namespace):
    registry = assistant.tools
    if getattr(args, "tools", None):
        registry = registry.subset([name.strip() for name in args.tools.split(",") if name.strip()])
    return registry


def cmd_agent(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    # 没给任务、或显式要求，就进入连续追问模式
    if args.chat or not args.task:
        return cmd_chat(args, config, assistant)

    registry = _build_tool_registry(assistant, args)

    if not args.json:
        _rule(f"Agent 任务 · 工作区={Path(config.workspace).resolve()}")
        print(f"{C.BOLD}{args.task}{C.RESET}\n")
        _info(f"可用工具：{registry.names()}")

    on_event = None if args.json else (
        _agent_event_printer(args) if (args.verbose or args.show_prompt) else None
    )
    agent = assistant.build_agent(registry, on_event=on_event)
    result = agent.run(args.task)
    assistant.record_usage(result.trace.usage)  # Agent 直连 LLM，需手动记账

    if args.json:
        _dump_json({"answer": result.answer, "stop_reason": result.stop_reason, "trace": result.trace.to_dict()})
    else:
        _rule("最终回答")
        print(result.answer)
        if args.trace:
            print()
            print(result.trace.to_markdown())
        else:
            _info(
                f"（用了 {result.steps_used} 轮、{len(result.trace.tool_calls)} 次工具调用、"
                f"{result.trace.usage.total_tokens} tokens；加 --trace 查看完整轨迹）"
            )
    _usage_line(config, assistant)
    return 0 if result.ok else 1


CHAT_HELP = """\
可用命令（以 / 开头）：
  /help            显示本帮助
  /tools           列出当前可用工具
  /context         显示会话上下文概况
  /trace           切换"每轮显示执行轨迹"
  /clear           清空上下文，开始全新对话
  /exit            退出（也可用 exit、quit、Ctrl+D）

提问示例：
  这个项目是做什么的？先列目录再读 README 总结
  config.py 里默认温度是多少？
  读取 parsers.py 的 extract_json，解释它怎么处理字符串里的括号
  写一段代码验证 0.1 + 0.2 是否等于 0.3
"""


def cmd_chat(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    """与 Agent 连续对话。

    与 `ask --chat` 的本质区别：
        ask --chat   —— 没有工具。你贴什么它看什么，答不上来就只能说"请提供代码"。
        chat（这里） —— 有工具。它会自己列目录、读文件、搜代码、跑代码来回答你的问题。
    """
    registry = _build_tool_registry(assistant, args)

    _rule("与 Agent 对话 · 它可以自己读文件、搜代码、执行代码")
    _info(f"工作区：{Path(config.workspace).resolve()}")
    _info(f"模型：{config.provider} / {config.resolved_model}")
    _info(f"可用工具：{registry.names()}")
    _info(f"最大步数：{config.max_steps}（每轮最多调用这么多次工具）")
    print(f"{C.DIM}输入 /help 查看命令，/exit 退出。{C.RESET}\n")

    show_trace = bool(getattr(args, "trace", False))
    session = assistant.build_session(registry, on_event=None if args.json else _agent_event_printer(args))

    def answer(question: str) -> None:
        nonlocal session
        print()
        try:
            result = session.ask(question)
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}⚠ 本轮已中断（上下文保留，可以继续问）{C.RESET}\n")
            return
        assistant.record_usage(result.trace.usage)  # Agent 直连 LLM，需手动记账
        print(f"\n{C.GREEN}助手 ▸{C.RESET} {result.answer}\n")
        meta = (
            f"{C.DIM}—— {result.steps_used} 轮 / {len(result.trace.tool_calls)} 次工具调用 / "
            f"{result.trace.usage.total_tokens} tokens / {result.trace.elapsed_ms:.0f}ms"
            f"{' · 步数耗尽后强制总结' if result.stop_reason == 'max_steps' else ''}{C.RESET}"
        )
        print(meta)
        if show_trace:
            print()
            print(result.trace.to_markdown())
        print()

    # 允许直接把第一个问题写在命令行上：codeassist chat "这个项目做什么"
    first = getattr(args, "task", None)
    if first:
        print(f"{C.CYAN}你 ▸{C.RESET} {first}")
        answer(first)
    elif sys.stdin.isatty():
        pass  # 交互式终端，等下进入 input 循环
    else:
        _warn("检测到非交互式输入。可以把问题写在命令里：codeassist chat \"你的问题\"")

    while True:
        try:
            line = input(f"{C.CYAN}你 ▸{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # ---- 会话内命令 ----
        if line.startswith("/") or line.lower() in {"exit", "quit"}:
            command = line.lower()
            if command in {"/exit", "exit", "quit", ":q", "/q"}:
                break
            if command in {"/help", "/h", "/?"}:
                print()
                print(CHAT_HELP)
                continue
            if command == "/tools":
                print()
                print(f"{C.BOLD}当前可用工具：{C.RESET}")
                print(registry.describe())
                print()
                continue
            if command == "/context":
                print(f"\n{C.DIM}{session.history_summary()}{C.RESET}\n")
                continue
            if command == "/trace":
                show_trace = not show_trace
                _info(f"执行轨迹显示：{'开' if show_trace else '关'}\n")
                continue
            if command == "/clear":
                session.reset()
                _info("上下文已清空，开始全新对话。\n")
                continue
            _warn(f"未知命令 {line}；输入 /help 查看可用命令。")
            continue

        answer(line)

    _rule("会话结束")
    _info(session.history_summary())
    _usage_line(config, assistant)
    return 0


def cmd_demo(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    from .demo import run_demo

    return run_demo()


def cmd_providers(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    _rule("内置服务商预设")
    print(f"{'名称':<12}{'默认模型':<26}{'密钥环境变量':<24}说明")
    for name, preset in PRESETS.items():
        key_env = preset.api_key_env if preset.requires_key else "（无需密钥）"
        print(f"{name:<12}{preset.default_model:<26}{key_env:<24}{preset.note}")
    print()
    _info("设置 CODEASSIST_PROVIDER 与对应密钥环境变量即可切换；用 --base-url 可接入任意 OpenAI 兼容服务。")
    return 0


def cmd_prompts(args: argparse.Namespace, config: Config, assistant: CodeAssistant) -> int:
    from .prompts import TEMPLATES

    _rule("Prompt 模板清单")
    for name, template in sorted(TEMPLATES.items()):
        print(f"{C.BOLD}{name}{C.RESET}")
        print(f"  用途：{template.description}")
        print(f"  变量：{template.variables or '（无）'}")
        print(f"  长度：{len(template.template)} 字符")
        print()
    return 0


# ======================================================================================
# main
# ======================================================================================
COMMANDS = {
    "generate": cmd_generate, "gen": cmd_generate, "g": cmd_generate,
    "complete": cmd_complete, "cmp": cmd_complete,
    "explain": cmd_explain, "exp": cmd_explain, "e": cmd_explain,
    "comment": cmd_comment, "cmt": cmd_comment, "c": cmd_comment,
    "ask": cmd_ask, "qa": cmd_ask,
    "agent": cmd_agent, "a": cmd_agent,
    "chat": cmd_chat, "ch": cmd_chat,
    "demo": cmd_demo,
    "providers": cmd_providers,
    "prompts": cmd_prompts,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows 控制台默认可能不是 UTF-8，中文与符号会直接抛 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    C.setup(force=False if args.no_color else None)

    if not args.command:
        parser.print_help()
        return 0

    try:
        config = _resolve_config(args)
    except ValueError as exc:
        _error(str(exc))
        return 1

    # demo 命令永远用离线模型，保证"零配置上手"
    if args.command == "demo":
        config = config.child(provider="mock")

    try:
        assistant = _make_assistant(config, args)
    except ValueError as exc:
        _error(str(exc))
        return 1

    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover
        parser.print_help()
        return 1

    try:
        return handler(args, config, assistant)
    except KeyboardInterrupt:
        print()
        _warn("已中断")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底，避免用户看到裸 traceback
        _error(f"{type(exc).__name__}: {exc}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        else:
            _info("加 -v 查看完整堆栈。")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
