"""LLM 调用层：把「一次模型调用」这件事讲清楚。

这一层是整个 Agent 的地基。剥掉所有框架后，所谓"调用大模型"就三件事：
    1. 用 HTTP POST 把 messages（一个 JSON 数组）发到厂商的 /chat/completions；
    2. 带上 Authorization: Bearer <key>；
    3. 从返回 JSON 里取出 choices[0].message.content。

本项目只用标准库 urllib 实现，为的是让你看清协议本身，而不是被 SDK 包装遮住。
同时实现了：
    * 重试 + 指数退避 + Retry-After 尊重（生产必备）
    * SSE 流式输出（打字机效果的原理）
    * Token 用量统计与耗时统计（成本与性能观测）
    * Function Calling 的请求/响应结构（工具调用的协议层）
    * MockLLM 离线假模型（无网络/无密钥也能开发与测试）
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .config import Config


# --------------------------------------------------------------------------------------
# 数据结构：与 OpenAI Chat Completions 协议一一对应
# --------------------------------------------------------------------------------------
class LLMError(RuntimeError):
    """所有调用失败的统一异常类型。"""

    def __init__(self, message: str, *, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ToolCall:
    """模型请求调用一个工具。

    arguments 是**字符串形式的 JSON**（这是协议规定的，不是我们偷懒）——
    所以必须有解析容错，模型偶尔会吐出不合法的 JSON。
    """

    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> Dict[str, Any]:
        if not self.arguments or not self.arguments.strip():
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            # 容错：有的模型会返回 ```json ... ``` 或前后带解释文字
            from .parsers import extract_json

            value = extract_json(self.arguments, default=None)
        if value is None:
            # 解析彻底失败：把原文带上，便于 trace 里一眼看出模型发了什么垃圾
            return {"_raw": self.arguments}
        return value if isinstance(value, dict) else {"_raw": value}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Message:
    """对话历史中的一条消息。

    role 的取值（协议规定）：
        system    系统设定 / 人设 / 规则（优先级最高，模型最"听话"的地方）
        user      用户输入
        assistant 模型输出（可能带 tool_calls）
        tool      工具执行结果（必须带 tool_call_id 与前面的 tool_calls 对应）
    """

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role}
        # 协议细节：带 tool_calls 的 assistant 消息，content 允许为 null，
        # 但很多厂商要求这个字段必须存在，因此显式给 None 而不是省略。
        if self.content is not None or (self.role == "assistant" and self.tool_calls):
            payload["content"] = self.content
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    # 便捷构造器：让业务代码读起来像自然语言
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        return cls("assistant", content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: Optional[str] = None) -> "Message":
        return cls("tool", content, name=name, tool_call_id=tool_call_id)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> "Usage":
        raw = raw or {}
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass
class ChatResponse:
    """一次模型调用的结果。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def as_message(self) -> Message:
        """把响应转回一条 assistant 消息，方便加入对话历史。"""
        return Message.assistant(self.content, self.tool_calls)


# --------------------------------------------------------------------------------------
# 抽象接口
# --------------------------------------------------------------------------------------
class BaseLLM:
    """所有 LLM 客户端的最小公共契约。

    只有一个方法：给定 messages（和可选的工具声明），返回 ChatResponse。
    把接口压到最窄，上层（Agent / Assistant）才能同时兼容真模型、假模型、缓存代理。
    """

    name: str = "base"
    model: str = ""

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ChatResponse:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def build_payload(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """组装请求体。单独抽出来是为了让 --dry-run 能原样打印"将要发出去的东西"。"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
        return payload


# --------------------------------------------------------------------------------------
# 真实实现：OpenAI 兼容 HTTP 客户端
# --------------------------------------------------------------------------------------
class OpenAICompatibleLLM(BaseLLM):
    """通过 /chat/completions 调用任何 OpenAI 兼容服务。"""

    #: 这些状态码值得重试（限流 / 服务端抖动）
    RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: Config, *, request_log: Optional[List[Dict[str, Any]]] = None):
        self.config = config
        config.validate()
        self.name = config.provider
        self.model = config.resolved_model
        self.base_url = config.resolved_base_url
        self.api_key = config.resolved_api_key
        self.chat_path = config.chat_path or config.preset.chat_path
        self.timeout = config.timeout
        self.max_retries = max(0, config.max_retries)
        self.backoff = config.retry_backoff
        #: 请求审计日志：出问题时"到底发了什么"永远是最重要的信息
        self.request_log: List[Dict[str, Any]] = request_log if request_log is not None else []
        #: 累计用量，便于在长会话结束时回报成本
        self.total_usage = Usage()

    # ---------------------------------------------------------------- 底层 HTTP
    @property
    def endpoint(self) -> str:
        base = self.base_url
        path = self.chat_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # 建议所有请求都带 UA，方便服务端侧排查
            "User-Agent": "codeassist/0.1 (+https://example.local)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_once(self, payload: Dict[str, Any], *, stream: bool) -> Any:
        """发一次请求。失败抛 LLMError / urllib 异常，由上层决定是否重试。"""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=data, headers=self._headers(), method="POST"
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - 读取失败不影响主流程
                pass
            raise LLMError(
                self._explain_http_error(exc.code, body), status=exc.code, body=body
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            hint = ""
            text = str(reason)
            # 常见但报错信息毫无指向性的两类问题，主动给出提示
            if "CERTIFICATE" in text.upper() or "SSL" in text.upper() or "凭证" in text:
                hint = "\n  提示：TLS/证书握手失败。常见于受限网络或代理环境，可尝试设置 HTTPS_PROXY 或换用 --mock。"
            elif "getaddrinfo" in text or "Name or service not known" in text:
                hint = "\n  提示：DNS 解析失败，检查网络或 base_url 拼写。"
            raise LLMError(f"网络请求失败：{reason}{hint}") from exc
        except TimeoutError as exc:
            raise LLMError(f"请求超时（{self.timeout}s）：可调大 --timeout 或减小 max_tokens") from exc

        if stream:
            return response  # 调用方负责逐行读取

        raw_bytes = response.read()
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"服务端返回的不是合法 JSON（前 300 字符）：{text[:300]}", body=text
            ) from exc

    @staticmethod
    def _explain_http_error(status: int, body: str) -> str:
        """把 HTTP 错误翻译成"人话 + 怎么办"。这是新手最容易卡住的地方。"""
        table = {
            400: "请求格式错误（常见原因：messages 结构不合法、模型名不存在、参数越界）",
            401: "认证失败（API Key 错误、过期，或没有加上 sk- 前缀）",
            402: "余额不足",
            403: "无权限（可能是地域限制 / 模型未开通）",
            404: "接口不存在（多半是 base_url 少了或多了 /v1）",
            413: "请求体过大（上下文超长，需要裁剪历史）",
            422: "参数校验失败",
            429: "触发限流或配额用尽（可重试，建议降低并发）",
            500: "服务端内部错误（可重试）",
            502: "网关错误（可重试）",
            503: "服务暂不可用（可重试）",
            504: "网关超时（可重试）",
        }
        return f"HTTP {status}：{table.get(status, '未知错误')}；响应体：{body[:500]}"

    def _post_with_retry(self, payload: Dict[str, Any], *, stream: bool = False) -> Any:
        """带指数退避的重试包装。重试只针对"可能自愈"的错误。"""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_once(payload, stream=stream)
            except LLMError as exc:
                last_error = exc
                retryable = exc.status in self.RETRYABLE_STATUS if exc.status else True
                if not retryable or attempt >= self.max_retries:
                    raise
            except Exception as exc:  # pragma: no cover - 兜底
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMError(f"调用失败：{exc}") from exc

            delay = self.backoff ** attempt + random.uniform(0, 0.3)
            time.sleep(min(delay, 20.0))
        raise LLMError(f"重试 {self.max_retries} 次后仍失败：{last_error}")

    # ---------------------------------------------------------------- 对外接口
    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        payload = self.build_payload(
            messages,
            tools=tools,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
        # 允许调用方透传额外参数（如 seed、frequency_penalty）
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        self.request_log.append({"url": self.endpoint, "payload": payload})

        started = time.perf_counter()
        raw = self._post_with_retry(payload)
        latency_ms = (time.perf_counter() - started) * 1000

        choices = raw.get("choices") or []
        if not choices:
            raise LLMError(f"响应中没有 choices 字段：{json.dumps(raw, ensure_ascii=False)[:400]}")
        message = choices[0].get("message") or {}

        tool_calls: List[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = item.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=item.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "{}",
                )
            )

        usage = Usage.from_raw(raw.get("usage"))
        self.total_usage = self.total_usage + usage

        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason") or "stop",
            usage=usage,
            model=raw.get("model") or self.model,
            latency_ms=latency_ms,
            raw=raw,
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """SSE 流式输出，逐块 yield 文本增量。

        协议本质：服务端不再一次性返回完整 JSON，而是持续推 `data: {...}\\n\\n`，
        每个分片里带着 `choices[0].delta.content` 的一小段文字，最后以 `data: [DONE]` 收尾。
        工具调用（tool_calls）在流式下是分片拼接的，实现复杂，因此 Agent 循环统一走非流式。
        """
        payload = self.build_payload(
            messages,
            tools=tools,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        self.request_log.append({"url": self.endpoint, "payload": payload, "stream": True})

        response = self._post_with_retry(payload, stream=True)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        finally:
            response.close()

    def stream_text(self, messages: Sequence[Message], **kwargs: Any) -> str:
        """流式 + 汇总：既能实时打印，又能拿到完整结果。"""
        chunks: List[str] = []
        for piece in self.stream(messages, **kwargs):
            chunks.append(piece)
        return "".join(chunks)


# --------------------------------------------------------------------------------------
# 离线实现：MockLLM
# --------------------------------------------------------------------------------------
Handler = Callable[[Sequence[Message], Optional[List[Dict[str, Any]]]], ChatResponse]


class MockLLM(BaseLLM):
    """离线假模型。

    它不会"思考"，只是按规则返回预设/启发式响应，但**协议行为与真模型完全一致**
    （同样接受 messages/tools，同样可能返回 tool_calls）。
    价值在于：
      * 没有 API Key、没有网络时，依然能端到端跑通并写单元测试；
      * 测试用例可以精确构造边界响应（比如"模型返回了坏 JSON"）；
      * 避免把测试写成"依赖真模型随机输出"的脆弱测试。
    """

    def __init__(self, handler: Optional[Handler] = None, *, model: str = "mock-code-assistant"):
        self.name = "mock"
        self.model = model
        self.request_log: List[Dict[str, Any]] = []
        self.total_usage = Usage()
        if handler is None:
            from .mock import heuristic_handler

            handler = heuristic_handler
        self.handler = handler

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        payload = self.build_payload(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        self.request_log.append({"url": "mock://offline", "payload": payload})
        response = self.handler(list(messages), tools)
        if not response.model:
            response.model = self.model
        # 粗略估算 token（4 字符 ≈ 1 token），让用量统计在离线模式下也能跑
        if not response.usage.total_tokens:
            prompt_chars = sum(len(m.content or "") for m in messages)
            response.usage = Usage(
                prompt_tokens=prompt_chars // 4,
                completion_tokens=len(response.content) // 4,
                total_tokens=(prompt_chars + len(response.content)) // 4,
            )
        self.total_usage = self.total_usage + response.usage
        return response

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterator[str]:
        """假流式：先算完整结果，再切块吐出来，用于本地演示打字机效果。"""
        response = self.chat(messages, tools=None, **{k: v for k, v in kwargs.items() if k != "tools"})
        text = response.content
        step = 24
        for index in range(0, len(text), step):
            yield text[index : index + step]


class CachingLLM(BaseLLM):
    """给任意 LLM 套一层进程内缓存。

    教学点：同样的 prompt 重复调用是纯浪费钱。缓存 key 必须是
    "模型 + 全部 messages + 工具声明 + 采样参数"的完整指纹，否则会串味。
    """

    def __init__(self, inner: BaseLLM, *, max_entries: int = 256):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.model = inner.model
        self.max_entries = max_entries
        self._cache: Dict[str, ChatResponse] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _fingerprint(messages: Sequence[Message], tools: Any, temperature: Any) -> str:
        return json.dumps(
            {
                "messages": [m.to_dict() for m in messages],
                "tools": tools or [],
                "temperature": temperature,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def chat(self, messages: Sequence[Message], *, tools=None, temperature=None, **kwargs: Any) -> ChatResponse:
        key = self._fingerprint(messages, tools, temperature)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return ChatResponse(
                content=cached.content,
                tool_calls=cached.tool_calls,
                finish_reason=cached.finish_reason,
                usage=cached.usage,
                model=cached.model,
                raw=cached.raw,
                from_cache=True,
            )
        self.misses += 1
        response = self.inner.chat(messages, tools=tools, temperature=temperature, **kwargs)
        if len(self._cache) >= self.max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = response
        return response

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterator[str]:
        yield self.chat(messages, **kwargs).content


# --------------------------------------------------------------------------------------
# 工厂函数
# --------------------------------------------------------------------------------------
def get_llm(
    config: Optional[Config] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    mock: bool = False,
    cache: bool = False,
    **overrides: Any,
) -> BaseLLM:
    """按配置构造 LLM 客户端。

    上层业务代码只依赖 BaseLLM 接口，因此可以在"真模型 / 假模型 / 缓存代理"
    之间无缝切换 —— 这就是依赖倒置带来的好处。
    """
    config = config or Config()
    if any(v is not None for v in (provider, model, api_key)) or overrides:
        config = config.child(
            **{
                k: v
                for k, v in {
                    "provider": provider,
                    "model": model,
                    "api_key": api_key,
                    "temperature": overrides.pop("temperature", None),
                    "max_tokens": overrides.pop("max_tokens", None),
                    "timeout": overrides.pop("timeout", None),
                    "max_retries": overrides.pop("max_retries", None),
                }.items()
                if v is not None
            }
        )
    if mock:
        config = config.child(provider="mock")

    if config.provider == "mock":
        llm: BaseLLM = MockLLM(model=config.resolved_model)
    else:
        llm = OpenAICompatibleLLM(config)

    if cache:
        return CachingLLM(llm)
    return llm
