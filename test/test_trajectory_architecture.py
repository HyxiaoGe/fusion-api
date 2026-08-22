"""轨迹读取读侧的依赖方向约束。"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


def _is_service_module(module: str) -> bool:
    return module == "app.services" or module.startswith("app.services.")


def _absolute_from_import(node: ast.ImportFrom, package: tuple[str, ...]) -> tuple[str, ...]:
    if node.level == 0:
        base = ()
    else:
        levels_up = node.level - 1
        base = package[: len(package) - levels_up] if levels_up < len(package) else ()
    module_parts = tuple(node.module.split(".")) if node.module else ()
    return base + module_parts


def _forbidden_service_imports(tree: ast.AST, package: tuple[str, ...]) -> list[str]:
    """返回当前模块 AST 中指向 app.services 的导入。"""
    detected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            detected.update(alias.name for alias in node.names if _is_service_module(alias.name))
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = ".".join(_absolute_from_import(node, package))
        if _is_service_module(module):
            detected.add(module)
            continue
        detected.update(
            f"{module}.{alias.name}" if module else alias.name
            for alias in node.names
            if _is_service_module(f"{module}.{alias.name}" if module else alias.name)
        )
    return sorted(detected)


def _package_for_source(source: Path, data_dir: Path) -> tuple[str, ...]:
    return ("app", "db", *source.relative_to(data_dir).parent.parts)


class TrajectoryArchitectureTests(unittest.TestCase):
    def test_import_guard_detects_absolute_and_relative_service_import_spellings(self):
        """若守卫遗漏 Import、相对 ImportFrom 或 from app import services，反向依赖会假绿。"""
        cases = {
            "import app.services.agent": ["app.services.agent"],
            "from ..services import agent": ["app.services"],
            "from app import services": ["app.services"],
            "from app.schemas import trajectory": [],
            "from app import services_like": [],
            "import app.services_like": [],
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_forbidden_service_imports(ast.parse(source), ("app", "db")), expected)

    def test_data_layer_does_not_depend_on_service_layer(self):
        """若 app/db 任一模块重新导入 app.services，四层依赖方向将被破坏。"""
        data_dir = Path(__file__).resolve().parents[1] / "app" / "db"
        service_imports = {
            source.relative_to(data_dir).as_posix(): _forbidden_service_imports(
                ast.parse(source.read_text(encoding="utf-8")),
                _package_for_source(source, data_dir),
            )
            for source in data_dir.rglob("*.py")
        }
        service_imports = {source: imports for source, imports in service_imports.items() if imports}

        self.assertEqual(service_imports, {})
