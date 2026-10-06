# codeassist —— 「代码助手 Agent」

**零第三方依赖**（纯 Python 标准库）的代码助手 Agent 参考实现，把 Agent 开发的各项基础能力用可读、可跑、可测的代码实现一遍。

| 核心功能 | 用到的能力 | 实现 |
| --- | --- | --- |
| ① 自然语言 → 代码 | Prompt Engineering、Few-shot、自修复闭环 | `assistant.generate()` |
| ② 代码补全 | Fill-in-the-Middle、增量输出约束 | `assistant.complete()` |
| ③ 解释代码逻辑 | 代码理解、结构化输出 | `assistant.explain()` |
| ④ 生成注释 | 代码理解、自然语言生成 | `assistant.comment()` |
| ⑤ 代码问答 | 上下文注入（RAG 最小形态）、多轮对话 | `assistant.ask()` |
| ⑥ 工具调用 Agent | Function Calling、ReAct 循环、可观测性 | `agent.Agent` |
| ⑦ 连续对话 | 多轮上下文管理、跨轮保留工具上下文 | `agent.AgentSession` |

## 文档

| 文档 | 内容 |
| --- | --- |
| **README.md**（本文） | 项目说明、使用方法、需求对照 |
| [Design.md](Design.md) | 架构设计、模块设计、关键决策与取舍 |

## 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | **3.10+**（用到 `ast.unparse` 与运行时求值的 `X \| Y` 注解；开发验证环境为 3.13） |
| 第三方依赖 | **无**，不需要 `pip install` |
| 网络 / API Key | 均为可选，用 `--mock` 可完全离线运行 |
| 系统 | Windows / macOS / Linux（已在 Windows + Python 3.13 验证） |

> **Windows 用户注意**：如果 `python` 命令执行后没有任何输出，说明它是 Microsoft Store 的存根程序，请改用 `py`。

## 快速开始

### 1. 离线跑通全部能力（不需要密钥、不联网）

```bash
py -m codeassist demo
```

内置 Mock 模型会真实跑完七项能力，重点看两个环节：**自修复闭环**（生成出错 → 本地校验抓到 → 自动重写）和 **Agent 工具循环**（真实列目录、真实读文件）。

### 2. 接入真实模型

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxx"
py -m codeassist generate "实现一个线程安全的 LRU 缓存"

# macOS / Linux
export DEEPSEEK_API_KEY=sk-xxxx
python -m codeassist generate "实现一个线程安全的 LRU 缓存"
```

只设密钥即可，**不用指定 `--provider`**（自动识别厂商）。支持 6 家服务商 + 本地 Ollama，用 `py -m codeassist providers` 查看。

想长期使用就写配置文件 `.codeassist.json`（密钥仍用环境变量提供）：

```json
{ "provider": "deepseek", "model": "deepseek-chat", "temperature": 0.2, "max_tokens": 2048 }
```

### 3. 验证配置（不花钱）

```bash
py -m codeassist --dry-run generate "写个二分查找"     # 只打印请求，不真正调用
```

## 使用方法

```bash
# 生成代码
py -m codeassist generate "写一个二分查找" --lang python
py -m codeassist generate "解析 CSV 并聚合" -c "只用标准库" -c "必须带类型标注" -o csv_utils.py

# 代码补全（只输出新增片段）
py -m codeassist complete --file app.py --line 12

# 解释代码（推荐 --function 精准限定：实测比整个文件省 11 倍 token）
py -m codeassist explain --file parsers.py --function extract_json --level detailed
py -m codeassist explain --file app.py --json           # 结构化输出

# 生成注释（逻辑不变）
py -m codeassist comment --file app.py --function process -o app_commented.py

# 代码问答（无工具，你贴代码它回答）
py -m codeassist ask "这段代码复杂度多少？" --file app.py --function process

# 与 Agent 连续对话（有工具，它自己读文件、搜代码、执行代码）
py -m codeassist chat
py -m codeassist chat "这个项目做什么？先列目录再读 README 总结"

# 工具调用 Agent（一次性任务）
py -m codeassist agent "读取 README.md 并总结项目用途" --trace

# 调试开关
py -m codeassist --show-prompt explain --file app.py   # 打印完整提示词
py -m codeassist prompts                               # 查看所有 Prompt 模板
```

**怎么选**：答案在你已提供的代码里 → `ask`；需要它自己去查 → `chat`（连续追问）或 `agent`（一次问完）。

`chat` 会**跨轮记住它读过哪些文件**，所以能顺着追问"那第 20 行呢"。会话内命令：`/help` `/tools` `/context` `/trace` `/clear` `/exit`。

**Agent 的 6 个工具**：`list_dir` `read_file` `search_code` `check_python_syntax`（只读）、`write_file` `run_python`（有副作用）。
用 `--no-execute` 禁用代码执行，用 `--tools read_file,search_code` 只给只读工具，用 `--workspace <目录>` 限定文件访问范围。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--provider` / `--model` | 服务商 / 模型（留空自动识别） |
| `--temperature` | 采样温度，代码任务建议 0~0.3 |
| `--workspace` | 工作区根目录，工具的文件访问被限制在此目录内 |
| `--max-steps` | Agent 每轮最多调用多少次工具（默认 8） |
| `--tools` / `--no-execute` | 裁剪工具集 / 禁用代码执行 |
| `--dry-run` / `--show-prompt` / `--json` / `--trace` / `-v` | 调试与脚本集成 |

全局参数写在子命令前后都可以。`py -m codeassist <命令> --help` 查看完整参数。

### 排错速查

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| 需要 API Key | 密钥没配 | 设环境变量，或加 `--api-key` |
| HTTP 401 | 密钥错 / 过期 / 带了空格 | 重新复制 |
| HTTP 404 | **base_url 多了或少了 `/v1`**（最高频） | 用 `--base-url` 纠正 |
| HTTP 429 | 限流 | 程序自动重试 |
| 请求超时 | 网络慢或输出太长 | `--timeout 120` 或调小 `--max-tokens` |
| 符号名不唯一 / 没找到 | `--function` 写错 | 提示里会列出所有可用符号 |
| 路径越界 | 工具访问了工作区外的文件 | 安全拦截；用 `--workspace` 调整 |
| `python` 无任何输出 | 那是 Store 存根程序 | 改用 `py` |

还是不行就三步定位：`--dry-run` 确认配置 → `--show-prompt` 看提示词 → `-v` 看完整堆栈。

## 需求对照

| 原始要求 | 实现 | 验证 |
| --- | --- | --- |
| 自然语言生成代码片段 | `assistant.generate()` | `py -m codeassist generate "..."` |
| Prompt Engineering + Few-shot | `prompts.py`（模板引擎 + 13 套模板 + Few-shot 示例） | `py -m codeassist prompts`；`generate --no-fewshot` 对比 |
| 代码补全 | `assistant.complete()` | `complete --file app.py --line 12` |
| 解释代码逻辑 | `assistant.explain()` | `explain --file x.py --level detailed` |
| 生成注释 | `assistant.comment()` | `comment --file app.py -o out.py` |
| 回答代码问题 / 代码理解 | `assistant.ask()`、`assistant.chat()`、`chat` 模式 | `ask "..." --file app.py` |
| 自然语言生成 | explain / comment / ask 均为自然语言输出 | 见上 |
| Agent 基础能力：LLM 调用 | `config.py`、`llm.py` | `tests/test_llm.py`（28 用例） |
| Agent 基础能力：Prompt 设计 | `prompts.py`、`parsers.py` | `tests/test_prompts.py`（18 用例） |
| Agent 基础能力：工具集成 | `tools.py` | `tests/test_tools.py`（32 用例） |

## 项目结构

```
├── README.md / Design.md           项目说明 / 设计文档
├── codeassist/                     源码约 5700 行（含注释），12 个模块
│   ├── llm.py         LLM 调用：HTTP 客户端、流式、重试、Mock 模型、缓存
│   ├── tools.py       工具集成：装饰器、Schema 生成、注册表、6 个内置工具
│   ├── assistant.py   业务能力：生成/补全/解释/注释/问答 + 自修复
│   ├── agent.py       Agent 循环：ReAct、熔断、纠偏、trace、多轮会话
│   ├── parsers.py     解析校验：代码块/JSON 宽容解析、语法校验、符号抽取
│   ├── prompts.py     Prompt 工程：模板引擎 + 13 套模板 + Few-shot
│   ├── mock.py        离线 Mock 模型与脚本化测试替身
│   ├── cli.py         命令行入口（10 个子命令）
│   ├── demo.py        离线全流程演示
│   └── config.py      配置与密钥管理
├── tests/             179 个单元测试，全部离线可跑
```

## 架构

```
  用户 / CLI
      │
      ▼
  assistant.py  业务能力层 ── 固定流程、单次调用、成本可预测
      │
      ▼
  agent.py      Agent 循环层 ── ReAct：想 → 做 → 看 → 再想
      │                    │
      ▼                    ▼
  prompts.py           tools.py
      └────────┬───────────┘
               ▼
           llm.py   只依赖 BaseLLM 抽象 → 真模型 / Mock / 缓存 / 追踪 可互换
               │
               ▼
      config.py + parsers.py
```

每层都能被单独替换：换厂商只动 `config`/`llm`，换提示词只动 `prompts`，加工具只动 `tools`。详见 [Design.md](Design.md) §2。

## 测试

```bash
py -m unittest discover -s tests -t .
# Ran 179 tests in 0.2s — OK
```

179 个用例，**不联网、不需要密钥、不依赖真模型**。用 `MockLLM` / `ScriptedLLM` 替换真实模型，让"模型返回坏 JSON""工具报错""模型原地打转""步数耗尽"这类边界场景可被精确构造——用真模型写断言等于用随机数写断言。

## 已知限制

- **`run_python` 不是生产级沙箱**：只做了进程隔离（`python -I`）、超时、极简环境变量、路径围栏、输出截断。生产需叠加容器 / 微虚拟机 / seccomp，或用 `--no-execute` 关闭。
- **上下文管理是截断而非摘要**：`AgentSession` 保留最近 40 条消息，未做摘要压缩。
- **无持久化、无并发工具执行、无评测脚本**：见 [Design.md](Design.md) §9。
- **Mock 模型不做真实推理**：它只按规则汇总，用于验证流程正确性，不能代替真实模型效果。
- **受限沙箱下的临时目录**：若系统临时区不可写，`run_python` 会用"实际写一次"探测并回退到 `<工作区>/.codeassist_scratch/`（已加入 `.gitignore`）。

## 安全提醒

- **代码执行**：`run_python` 会真的执行模型生成的代码。建议在专用目录运行、用 `--workspace` 限定范围。
- **文件访问**：所有文件操作都被限制在 `--workspace` 内，越界路径（如 `../../.ssh/id_rsa`）会被代码级拒绝。
- **密钥**：只放环境变量，不写进配置文件或代码；不要用 `--api-key` 长期传参（会留在命令历史）。
- **数据合规**：你的代码会发送给模型服务商。敏感代码请用本地模型（`--provider ollama`）。


