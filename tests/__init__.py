"""测试包的公共辅助。

本项目所有测试都**不联网、不需要 API Key、不依赖真模型**：
    用 MockLLM / ScriptedLLM 替换真实模型，用 unittest 标准库跑。
这不是将就，而是正确做法 —— 测试必须确定性、快、可离线复现。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 让 `py -m unittest discover -s tests` 能直接 import 到 codeassist
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["unittest"]
