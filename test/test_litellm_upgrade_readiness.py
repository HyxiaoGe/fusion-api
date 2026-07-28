from __future__ import annotations

import gzip
import json
import tempfile
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts import check_litellm_upgrade_readiness as readiness


class LiteLLMUpgradeReadinessTest(unittest.TestCase):
    def test_evaluate_requires_all_gates(self) -> None:
        report = readiness.evaluate_readiness(
            liveliness_ok=True,
            readiness_ok=True,
            database_connected=True,
            model_count=14,
            primary_backup=readiness.FileGate(path="/backup/db.sql.gz", ok=True, age_seconds=60),
            secondary_backup=readiness.FileGate(path="/backup/restic.ok", ok=False, reason="标记过期"),
            cost_sync_scheduled=False,
        )

        self.assertFalse(report.ready_for_upgrade)
        self.assertIn("secondary_backup", report.blockers)
        self.assertIn("cost_sync_not_scheduled", report.warnings)

    def test_empty_model_catalog_blocks_upgrade(self) -> None:
        report = readiness.evaluate_readiness(
            liveliness_ok=True,
            readiness_ok=True,
            database_connected=True,
            model_count=0,
            primary_backup=readiness.FileGate(path="/backup/db.sql.gz", ok=True),
            secondary_backup=readiness.FileGate(path="/backup/restic.ok", ok=True),
            cost_sync_scheduled=True,
        )

        self.assertEqual(report.blockers, ["model_catalog_empty"])

    def test_check_file_gate_rejects_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "restic.ok"
            marker.write_text("ok", encoding="utf-8")
            old = time.time() - 7200
            marker.touch()
            marker.chmod(0o600)
            import os

            os.utime(marker, (old, old))

            gate = readiness.check_file_gate(marker, max_age_seconds=3600)

        self.assertFalse(gate.ok)
        self.assertIn("超过", gate.reason)

    def test_primary_backup_rejects_invalid_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup = Path(tmp) / "db.sql.gz"
            backup.write_text("not-gzip", encoding="utf-8")

            gate = readiness.check_gzip_file_gate(backup, max_age_seconds=3600)

        self.assertFalse(gate.ok)
        self.assertIn("gzip", gate.reason)

    def test_primary_backup_accepts_valid_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backup = Path(tmp) / "db.sql.gz"
            with gzip.open(backup, "wb") as stream:
                stream.write(b"SELECT 1;")

            gate = readiness.check_gzip_file_gate(backup, max_age_seconds=3600)

        self.assertTrue(gate.ok)

    def test_secondary_marker_requires_success_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "restic.json"
            marker.write_text("{}", encoding="utf-8")

            gate = readiness.check_restic_marker(marker, max_age_seconds=3600)

        self.assertFalse(gate.ok)
        self.assertIn("success", gate.reason)

    def test_secondary_marker_uses_snapshot_time_not_file_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "restic.json"
            marker.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "snapshot_id": "snapshot-id",
                        "completed_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
                        "tag": "daily",
                    }
                ),
                encoding="utf-8",
            )
            marker.touch()

            gate = readiness.check_restic_marker(marker, max_age_seconds=3600)

        self.assertFalse(gate.ok)
        self.assertIn("snapshot", gate.reason)

    def test_serialized_report_never_contains_master_key(self) -> None:
        report = readiness.evaluate_readiness(
            liveliness_ok=True,
            readiness_ok=True,
            database_connected=True,
            model_count=1,
            primary_backup=readiness.FileGate(path="/backup/db.sql.gz", ok=True),
            secondary_backup=readiness.FileGate(path="/backup/restic.ok", ok=True),
            cost_sync_scheduled=True,
        )

        serialized = json.dumps(readiness.serialize_report(report), ensure_ascii=False)

        self.assertNotIn("master", serialized.lower())
        self.assertNotIn("token", serialized.lower())


if __name__ == "__main__":
    unittest.main()
