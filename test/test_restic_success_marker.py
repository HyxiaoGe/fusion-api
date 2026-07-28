from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts import write_restic_success_marker as marker

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def snapshot_payload(*, snapshot_id: str = "snapshot-full-id", completed_at: datetime | None = None) -> list[dict]:
    return [
        {
            "id": snapshot_id,
            "short_id": snapshot_id[:8],
            "time": (completed_at or (NOW - timedelta(minutes=5))).isoformat(),
            "tags": ["daily"],
        }
    ]


class ResticSuccessMarkerTests(unittest.TestCase):
    def test_build_marker_accepts_fresh_snapshot(self) -> None:
        result = marker.build_success_marker(
            snapshot_payload(),
            now=NOW,
            max_age_seconds=3600,
        )

        self.assertEqual(
            result,
            {
                "status": "success",
                "snapshot_id": "snapshot-full-id",
                "completed_at": "2026-07-28T11:55:00+00:00",
                "tag": "daily",
            },
        )

    def test_build_marker_rejects_missing_snapshot_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "snapshot id"):
            marker.build_success_marker(
                snapshot_payload(snapshot_id=""),
                now=NOW,
                max_age_seconds=3600,
            )

    def test_build_marker_rejects_snapshot_without_daily_tag(self) -> None:
        payload = snapshot_payload()
        payload[0]["tags"] = ["manual"]

        with self.assertRaisesRegex(ValueError, "required tag"):
            marker.build_success_marker(payload, now=NOW, max_age_seconds=3600)

    def test_build_marker_rejects_stale_snapshot(self) -> None:
        with self.assertRaisesRegex(ValueError, "过期"):
            marker.build_success_marker(
                snapshot_payload(completed_at=NOW - timedelta(hours=2)),
                now=NOW,
                max_age_seconds=3600,
            )

    def test_build_marker_rejects_future_snapshot(self) -> None:
        with self.assertRaisesRegex(ValueError, "未来"):
            marker.build_success_marker(
                snapshot_payload(completed_at=NOW + timedelta(minutes=10)),
                now=NOW,
                max_age_seconds=3600,
            )

    def test_main_reads_stdin_and_writes_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "restic-success.json"
            stdin = io.StringIO(json.dumps(snapshot_payload()))

            exit_code = marker.main(
                [
                    "--snapshots-json",
                    "-",
                    "--output",
                    str(output),
                    "--max-age-seconds",
                    str(10**9),
                ],
                stdin=stdin,
                now=NOW,
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["snapshot_id"], "snapshot-full-id")

    def test_failure_does_not_overwrite_existing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "restic-success.json"
            original = '{"status":"success","snapshot_id":"old","completed_at":"2026-07-27T00:00:00+00:00"}\n'
            output.write_text(original, encoding="utf-8")
            source = Path(tmp) / "snapshots.json"
            source.write_text(json.dumps(snapshot_payload(snapshot_id="")), encoding="utf-8")

            exit_code = marker.main(
                [
                    "--snapshots-json",
                    str(source),
                    "--output",
                    str(output),
                    "--max-age-seconds",
                    str(10**9),
                ],
                now=NOW,
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_atomic_replace_failure_preserves_existing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "restic-success.json"
            original = '{"status":"success","snapshot_id":"old","completed_at":"2026-07-27T00:00:00+00:00"}\n'
            output.write_text(original, encoding="utf-8")
            stdin = io.StringIO(json.dumps(snapshot_payload()))

            with patch.object(marker.os, "replace", side_effect=OSError("replace failed")):
                exit_code = marker.main(
                    [
                        "--snapshots-json",
                        "-",
                        "--output",
                        str(output),
                        "--max-age-seconds",
                        str(10**9),
                    ],
                    stdin=stdin,
                    now=NOW,
                )

            self.assertEqual(exit_code, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), original)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
