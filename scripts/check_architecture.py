"""
架构约束检查器

基于 AST 分析检查分层依赖规则：
  API(app/api/) → Service(app/services/) → AI(app/ai/) → Data(app/db/)
  依赖只能向下，不能反向。

用法：
  python scripts/check_architecture.py          # 检查全部
  python scripts/check_architecture.py --verbose # 显示每条规则的检查详情
"""

import ast
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

# 分层依赖规则：每层禁止 import 的模块
# 格式：{层路径: [(禁止 import 的模块前缀, 规则说明)]}
LAYER_RULES = {
    "app/db": [
        ("app.services", "数据层禁止 import 服务层"),
        ("app.api", "数据层禁止 import API 层"),
        ("app.ai", "数据层禁止 import AI 层"),
    ],
    "app/ai": [
        ("app.services", "AI 层禁止 import 服务层"),
        ("app.api", "AI 层禁止 import API 层"),
    ],
    "app/services": [
        ("app.api", "服务层禁止 import API 层"),
    ],
    "app/core": [
        ("app.services", "核心层禁止 import 服务层"),
        ("app.api", "核心层禁止 import API 层"),
        ("app.ai", "核心层禁止 import AI 层"),
    ],
    "app/schemas": [
        ("app.services", "Schema 层禁止 import 服务层"),
        ("app.api", "Schema 层禁止 import API 层"),
        ("app.ai", "Schema 层禁止 import AI 层"),
        ("app.db", "Schema 层禁止 import 数据层"),
        ("app.core", "Schema 层禁止 import 核心层"),
    ],
}

# API 层禁止直接操作数据库的方法调用
DB_DIRECT_OPS = {"query", "execute", "add", "add_all", "delete", "commit", "rollback", "flush"}


def get_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """从 AST 中提取所有 import 的模块名和行号"""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))
    return imports


def check_db_operations_in_api(tree: ast.AST, filepath: Path) -> list[str]:
    """检查 API 层是否直接调用数据库操作（db.query() 等）"""
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # 检查 db.query()、db.execute() 等模式
            if isinstance(node.func, ast.Attribute) and node.func.attr in DB_DIRECT_OPS:
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "db":
                    rel = filepath.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"  {rel}:{node.lineno} — API 层直接调用 db.{node.func.attr}()，应通过 Service/Repository 操作"
                    )
    return violations


def check_file(filepath: Path, verbose: bool = False) -> list[str]:
    """检查单个文件的架构违规"""
    violations = []
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        return [f"  {filepath.relative_to(PROJECT_ROOT)} — 解析失败: {e}"]

    rel_path = str(filepath.relative_to(PROJECT_ROOT))

    # 检查分层 import 规则
    for layer_prefix, rules in LAYER_RULES.items():
        if rel_path.startswith(layer_prefix):
            imports = get_imports(tree)
            for lineno, module in imports:
                for forbidden_prefix, desc in rules:
                    if module.startswith(forbidden_prefix):
                        rel = filepath.relative_to(PROJECT_ROOT)
                        violations.append(f"  {rel}:{lineno} — {desc}（import {module}）")

    # API 层特殊检查：禁止直接数据库操作
    if rel_path.startswith("app/api"):
        violations.extend(check_db_operations_in_api(tree, filepath))

    return violations


def check_new_api_has_test() -> list[str]:
    """检查每个 API 模块是否有对应测试文件"""
    violations = []
    api_dir = APP_DIR / "api"
    test_dir = PROJECT_ROOT / "test"

    for api_file in sorted(api_dir.glob("*.py")):
        if api_file.name.startswith("_"):
            continue
        module_name = api_file.stem
        # 检查是否有对应测试（test_<module>.py 或 test_<module>_*.py）
        test_files = list(test_dir.glob(f"test_{module_name}*.py"))
        if not test_files:
            violations.append(f"  app/api/{api_file.name} — 缺少对应的测试文件 test/test_{module_name}.py")

    return violations


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    has_errors = False

    # 收集所有 Python 文件
    py_files = sorted(APP_DIR.rglob("*.py"))

    if verbose:
        print(f"扫描 {len(py_files)} 个 Python 文件...\n")

    # 检查分层依赖（错误级别，阻断部署）
    layer_violations = []
    for f in py_files:
        layer_violations.extend(check_file(f, verbose))

    if layer_violations:
        has_errors = True
        print("❌ 分层依赖违规：\n")
        for line in layer_violations:
            print(line)

    # 检查 API 测试覆盖（警告级别，不阻断）
    test_warnings = check_new_api_has_test()
    if test_warnings:
        print("\n⚠️  测试覆盖缺失（警告）：\n")
        for line in test_warnings:
            print(line)

    # 输出结果
    if has_errors:
        error_count = len(layer_violations)
        print(f"\n共 {error_count} 处违规，架构检查未通过")
        sys.exit(1)
    else:
        print("✅ 架构检查通过")
        if verbose:
            print(f"   已检查 {len(py_files)} 个文件，无分层违规")
        sys.exit(0)


if __name__ == "__main__":
    main()
