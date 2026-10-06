"""`python -m codeassist` 的入口。

放在独立文件里是 Python 的惯例：`__main__.py` 只做转发，不承载业务逻辑，
这样 `python -m codeassist` 和 `codeassist` 控制台脚本能走同一条代码路径。
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
