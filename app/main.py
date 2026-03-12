"""
Compatibility entrypoint.

The canonical FastAPI app lives in the repository root at `main.py`.
Keep this shim only to avoid breaking legacy imports such as `app.main:app`.
"""

from main import app

