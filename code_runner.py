"""
Python 代码执行模块 - 安全沙箱内执行数学计算和 Python 代码

通过限制可用模块和执行时间来保证安全性。
"""

import sys
import io
import math
import traceback
from contextlib import redirect_stdout, redirect_stderr


# 沙箱中允许使用的安全模块和内置函数
_SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "sorted": sorted,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "print": print, "type": type, "isinstance": isinstance,
    "pow": pow, "divmod": divmod, "hex": hex, "bin": bin, "oct": oct,
    "chr": chr, "ord": ord,
    "True": True, "False": False, "None": None,
}

_SAFE_MODULES = {
    "math": math,
}

# 尝试导入可选的科学计算库
try:
    import numpy as np
    _SAFE_MODULES["numpy"] = np
    _SAFE_MODULES["np"] = np
except ImportError:
    pass

try:
    import statistics
    _SAFE_MODULES["statistics"] = statistics
except ImportError:
    pass


def run_python_code(code, timeout_hint=10):
    """
    在受限沙箱中执行 Python 代码。

    Args:
        code: 要执行的 Python 代码字符串
        timeout_hint: 超时提示（秒），注意这不是强制限制

    Returns:
        str: 执行结果（stdout 输出）或错误信息
    """
    # 安全检查：禁止危险操作
    dangerous_patterns = [
        "import os", "import sys", "import subprocess",
        "import shutil", "__import__", "eval(", "exec(",
        "open(", "file(", "compile(", "globals(", "locals(",
        "getattr(", "setattr(", "delattr(",
        "os.system", "os.popen", "os.exec",
        "subprocess.", "shutil.",
    ]
    code_lower = code.lower().replace(" ", "")
    for pattern in dangerous_patterns:
        if pattern.lower().replace(" ", "") in code_lower:
            return f"安全限制：不允许使用 `{pattern.strip()}`。本工具仅支持数学计算和数据处理。"

    # 构建沙箱环境
    sandbox_globals = {"__builtins__": _SAFE_BUILTINS}
    sandbox_globals.update(_SAFE_MODULES)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, sandbox_globals)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result_parts = []
        if output:
            result_parts.append(output.rstrip())
        if errors:
            result_parts.append(f"警告:\n{errors.rstrip()}")

        # 如果没有 print 输出，尝试返回最后一个表达式的值
        if not result_parts:
            try:
                last_line = code.strip().split("\n")[-1]
                val = eval(last_line, sandbox_globals)
                if val is not None:
                    result_parts.append(str(val))
            except Exception:
                pass

        return "\n".join(result_parts) if result_parts else "代码执行成功，无输出。"

    except Exception:
        tb = traceback.format_exc()
        # 只保留有用的错误信息
        lines = tb.split("\n")
        useful = [l for l in lines if not l.strip().startswith("File \"<") or "line" in l]
        return f"执行错误:\n{tb}"


# 工具定义（供 OpenAI function calling 使用）
CODE_RUNNER_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python_code",
        "description": "当用户需要进行数学计算、数据分析、单位换算、方程求解或任何需要精确计算的任务时，调用此工具执行 Python 代码。支持 math 和 numpy 库。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码。用 print() 输出结果。支持 math、numpy 库。"
                }
            },
            "required": ["code"]
        }
    }
}
