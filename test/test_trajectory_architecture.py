"""轨迹读取读侧的依赖方向约束。"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class TrajectoryArchitectureTests(unittest.TestCase):
    def test_data_layer_does_not_depend_on_service_layer(self):
        """若 app/db 任一模块重新导入 app.services，四层依赖方向将被破坏。"""
        data_dir = Path(__file__).resolve().parents[1] / "app" / "db"
        service_imports = {
            source.relative_to(data_dir).as_posix(): [
                node.module
                for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app.services")
            ]
            for source in data_dir.rglob("*.py")
        }
        service_imports = {source: imports for source, imports in service_imports.items() if imports}

        self.assertEqual(service_imports, {})
