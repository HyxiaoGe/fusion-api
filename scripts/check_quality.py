"""
代码质量检查器（轻量版）

检查项：
  1. 超长函数（>50 行）
  2. TODO 过期检查（列出所有 TODO）

用法：
  python scripts/check_quality.py
"""

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

MAX_FUNCTION_LINES = 50


def check_long_functions(py_files: list[Path]) -> list[str]:
    """检查超过 50 行的函数"""
    warnings = []
    for filepath in py_files:
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 计算函数行数（从 def 到最后一个子节点）
                if not node.body:
                    continue
                last_line = max(getattr(n, "end_lineno", n.lineno) for n in ast.walk(node) if hasattr(n, "lineno"))
                func_lines = last_line - node.lineno + 1
                if func_lines > MAX_FUNCTION_LINES:
                    rel = filepath.relative_to(PROJECT_ROOT)
                    warnings.append(f"  {rel}:{node.lineno} — {node.name}() 共 {func_lines} 行（上限 {MAX_FUNCTION_LINES}）")

    return warnings


def check_todos(py_files: list[Path]) -> list[str]:
    """列出所有 TODO 标记"""
    todos = []
    todo_pattern = re.compile(r"#\s*TODO[:\s](.+)", re.IGNORECASE)

    for filepath in py_files:
        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for lineno, line in enumerate(lines, 1):
            match = todo_pattern.search(line)
            if match:
                rel = filepath.relative_to(PROJECT_ROOT)
                todos.append(f"  {rel}:{lineno} — {match.group(1).strip()}")

    return todos


def main():
    py_files = sorted(APP_DIR.rglob("*.py"))

    # 超长函数检查
    long_funcs = check_long_functions(py_files)
    if long_funcs:
        print(f"⚠️  超长函数（>{MAX_FUNCTION_LINES} 行）：\n")
        for line in long_funcs:
            print(line)

    # TODO 检查
    todos = check_todos(py_files)
    if todos:
        print(f"\n📝 TODO 清单（共 {len(todos)} 项）：\n")
        for line in todos:
            print(line)

    if not long_funcs and not todos:
        print("✅ 质量检查通过，无超长函数和 TODO")

    # 质量检查只做提醒，不阻断
    sys.exit(0)


if __name__ == "__main__":
    main()
