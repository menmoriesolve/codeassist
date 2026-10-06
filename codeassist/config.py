"""配置层：Provider 预设、环境变量加载、密钥管理。

教学要点：
  * 密钥永远不进代码库 —— 只从环境变量 / 本地配置文件读取。
  * "Provider 预设"把 base_url / 默认模型 / 密钥环境变量名打包，让上层不关心厂商差异。
  * 所有厂商只要兼容 OpenAI 的 /chat/completions 协议，就能用同一个客户端调用。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------------------
# Provider 预设
# --------------------------------------------------------------------------------------
# 说明：这里只声明"协议层面的坐标"，不含任何密钥。
# 想接新厂商？在 PRESETS 里加一条即可 —— 这就是「适配器模式」的最小形态。
PRESETS: Dict[str, "ProviderPreset"] = {}


@dataclass(frozen=True)
class ProviderPreset:
    """一个 OpenAI 兼容服务端的接入坐标。"""

    name: str
    base_url: str
    default_model: str
    api_key_env: str
    # 有些厂商（如本地 Ollama）不需要密钥
    requires_key: bool = True
    # 部分兼容端点路径不是 /v1，这里允许整体覆盖
    chat_path: str = "/chat/completions"
    note: str = ""


def _register(preset: ProviderPreset) -> None:
    PRESETS[preset.name] = preset


_register(
    ProviderPreset(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        note="中文与代码能力均衡，价格低，适合作为默认教学 Provider。",
    )
)
_register(
    ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        note="事实标准。函数调用（tools）协议即由此定义。",
    )
)
_register(
    ProviderPreset(
        name="moonshot",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
        note="月之暗面 Kimi，长上下文。",
    )
)
_register(
    ProviderPreset(
        name="dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        note="阿里通义千问的 OpenAI 兼容模式。",
    )
)
_register(
    ProviderPreset(
        name="zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
        note="智谱 GLM，flash 系列有免费额度。",
    )
)
_register(
    ProviderPreset(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5-coder:7b",
        api_key_env="OLLAMA_API_KEY",
        requires_key=False,
        note="本地推理，零成本、数据不出机，适合练手。",
    )
)
_register(
    ProviderPreset(
        name="mock",
        base_url="mock://offline",
        default_model="mock-code-assistant",
        api_key_env="MOCK_API_KEY",
        requires_key=False,
        note="内置离线假模型：不需要网络和密钥，用于演示与单元测试。",
    )
)


# --------------------------------------------------------------------------------------
# 运行时配置
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG_FILENAMES = (".codeassist.json", "codeassist.config.json")


@dataclass
class Config:
    """运行期配置。所有字段都能被环境变量或配置文件覆盖。"""

    provider: str = "mock"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    chat_path: str = "/chat/completions"

    # 生成参数（LLM 的"旋钮"）
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 2048

    # 稳健性参数
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 1.5

    # Agent 行为
    max_steps: int = 8
    repair_attempts: int = 1
    allow_execute: bool = True
    workspace: str = "."
    verbose: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- 解析
    @property
    def preset(self) -> ProviderPreset:
        if self.provider not in PRESETS:
            known = ", ".join(sorted(PRESETS))
            raise ValueError(f"未知 provider: {self.provider!r}；可选：{known}")
        return PRESETS[self.provider]

    @property
    def resolved_base_url(self) -> str:
        return (self.base_url or self.preset.base_url).rstrip("/")

    @property
    def resolved_model(self) -> str:
        return self.model or self.preset.default_model

    @property
    def resolved_api_key(self) -> str:
        """密钥优先级：显式配置 > 预设声明的环境变量。"""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.preset.api_key_env, "")

    def validate(self) -> None:
        """提前失败：把"缺密钥"这种错误拦在发请求之前，并给可执行的修复建议。"""
        self.preset  # 触发 provider 校验
        if self.preset.requires_key and not self.resolved_api_key:
            raise ValueError(
                f"provider={self.provider!r} 需要 API Key，但没有找到。\n"
                f"  修复方式（任选其一）：\n"
                f"    1) 设置环境变量：{self.preset.api_key_env}=sk-xxxx\n"
                f"    2) 命令行传入：  --api-key sk-xxxx\n"
                f"    3) 想先跑通流程：加 --mock，或设置 CODEASSIST_PROVIDER=mock"
            )

    def child(self, **overrides: Any) -> "Config":
        """返回一个覆盖了部分字段的新配置（不可变风格，避免隐式副作用）。"""
        return replace(self, **overrides)


# --------------------------------------------------------------------------------------
# 加载：默认值 -> 配置文件 -> 环境变量 -> 显式覆盖
# --------------------------------------------------------------------------------------
_ENV_MAP: Dict[str, Any] = {
    "CODEASSIST_PROVIDER": ("provider", str),
    "CODEASSIST_MODEL": ("model", str),
    "CODEASSIST_BASE_URL": ("base_url", str),
    "CODEASSIST_API_KEY": ("api_key", str),
    "CODEASSIST_TEMPERATURE": ("temperature", float),
    "CODEASSIST_TOP_P": ("top_p", float),
    "CODEASSIST_MAX_TOKENS": ("max_tokens", int),
    "CODEASSIST_TIMEOUT": ("timeout", float),
    "CODEASSIST_MAX_RETRIES": ("max_retries", int),
    "CODEASSIST_MAX_STEPS": ("max_steps", int),
    "CODEASSIST_REPAIR_ATTEMPTS": ("repair_attempts", int),
    "CODEASSIST_WORKSPACE": ("workspace", str),
    # 布尔变量：出现即代表真值语义，交给 _coerce_bool 解析
    "CODEASSIST_ALLOW_EXECUTE": ("allow_execute", bool),
    "CODEASSIST_VERBOSE": ("verbose", bool),
}

_TRUE_WORDS = {"1", "true", "yes", "on", "y", "t"}


def _coerce_bool(value: str) -> bool:
    return str(value).strip().lower() in _TRUE_WORDS


def _coerce(value: str, caster: Any) -> Any:
    if caster is bool:
        return _coerce_bool(value)
    if caster is str:
        return value
    return caster(value)


def find_config_file(start: Optional[Path] = None) -> Optional[Path]:
    """在给定目录及其父目录中寻找配置文件，最后回退到用户主目录。"""
    here = Path(start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        for filename in DEFAULT_CONFIG_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    home = Path.home()
    for filename in DEFAULT_CONFIG_FILENAMES:
        candidate = home / filename
        if candidate.is_file():
            return candidate
    return None


def load_config(
    *,
    config_file: Optional[str | Path] = None,
    env: Optional[Dict[str, str]] = None,
    **overrides: Any,
) -> Config:
    """把四层来源合并成一个 Config。

    优先级（后者覆盖前者）：默认值 < 配置文件 < 环境变量 < 函数参数。
    """
    env = dict(os.environ if env is None else env)
    values: Dict[str, Any] = {}

    # --- 第 1 层：配置文件 ---
    path: Optional[Path]
    if config_file:
        path = Path(config_file)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在：{path}")
    else:
        path = find_config_file()

    if path is not None:
        try:
            # 用 utf-8-sig 读取：它会自动吃掉可能存在的 UTF-8 BOM。
            # Windows 上太容易踩这个坑了 —— PowerShell 的 `Set-Content -Encoding utf8`
            # 和旧版记事本默认都会写 BOM，而带 BOM 的 JSON 会让 json.loads 直接报错。
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"配置文件不是合法 JSON：{path} —— {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"配置文件顶层必须是对象：{path}")
        values.update({k: v for k, v in raw.items() if not k.startswith("_")})

    # --- 第 2 层：环境变量 ---
    for env_name, (field_name, caster) in _ENV_MAP.items():
        if env_name in env and env[env_name] != "":
            values[field_name] = _coerce(env[env_name], caster)

    # --- 第 3 层：显式覆盖（None 表示"没传"，跳过，避免把默认值冲掉）---
    values.update({k: v for k, v in overrides.items() if v is not None})

    config = Config(**values)

    # 兼容常见厂商的开箱即用写法：如果用户设了某厂商的 key 但没指定 provider，
    # 就自动选用该厂商，降低"第一次跑不起来"的挫败感。
    if "provider" not in values and config.provider == "mock":
        for name, preset in PRESETS.items():
            if preset.requires_key and env.get(preset.api_key_env):
                config.provider = name
                # 顺手把密钥也固定下来。否则 resolved_api_key 会去读真实 os.environ，
                # 一旦调用方传入的是自定义 env（测试、多环境切换），就会出现
                # "provider 认出来了、密钥却是空的"这种自相矛盾的状态。
                config.api_key = env[preset.api_key_env]
                break

    return config
