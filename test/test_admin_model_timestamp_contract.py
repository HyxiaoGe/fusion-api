import os
import subprocess
import sys
import unittest
from datetime import datetime, timezone

from app.db.admin_audit_repository import AdminAuditRepository
from app.db.models import AgentSession


class AdminModelTimestampContractTests(unittest.TestCase):
    def test_latest_datetime_normalizes_legacy_naive_utc_value(self):
        aware_value = datetime(2026, 7, 14, 5, 34, tzinfo=timezone.utc)
        legacy_naive_utc_value = datetime(2026, 7, 14, 5, 35)

        latest = AdminAuditRepository._latest_datetime(aware_value, legacy_naive_utc_value)

        self.assertEqual(latest, datetime(2026, 7, 14, 5, 35, tzinfo=timezone.utc))

    def test_agent_session_created_at_uses_timezone_aware_utc_column(self):
        created_at = AgentSession.__table__.c.created_at

        self.assertTrue(created_at.type.timezone)
        self.assertEqual(str(created_at.server_default.arg), "now()")

    def test_migration_interprets_legacy_agent_session_timestamp_as_utc(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        env = {
            **os.environ,
            "DATABASE_URL": "postgresql://user:pass@localhost/fusion",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "upgrade",
                "c4f8a2d1e6b9:d7e4a9c2f1b6",
                "--sql",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ALTER TABLE agent_sessions ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE "
            "USING created_at AT TIME ZONE 'UTC';",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
