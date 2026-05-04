"""导出 FastAPI OpenAPI schema。

用法：
    python scripts/export_openapi.py > openapi.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402


def main() -> None:
    print(json.dumps(app.openapi(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
