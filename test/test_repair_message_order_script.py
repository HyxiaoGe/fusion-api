import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import repair_message_order


def make_entry() -> repair_message_order.RepairEntry:
    return repair_message_order.RepairEntry(
        conversation_id="conv-1",
        user_message_id="user-1",
        assistant_message_id="assistant-1",
        expected_conversation_created_at=datetime(2026, 7, 13, 23, 17, 16, tzinfo=timezone.utc),
        corrected_conversation_created_at=datetime(2026, 7, 13, 15, 17, 16, tzinfo=timezone.utc),
        expected_conversation_updated_at=datetime(2026, 7, 13, 23, 17, 18, tzinfo=timezone.utc),
        corrected_conversation_updated_at=datetime(2026, 7, 13, 15, 17, 18, tzinfo=timezone.utc),
        expected_user_created_at=datetime(2026, 7, 13, 23, 17, 16, tzinfo=timezone.utc),
        corrected_user_created_at=datetime(2026, 7, 13, 15, 17, 16, tzinfo=timezone.utc),
        expected_assistant_created_at=datetime(2026, 7, 13, 15, 17, 17, tzinfo=timezone.utc),
    )


class MappingResult:
    def __init__(self, *, one=None, all_rows=None):
        self._one = one
        self._all = all_rows

    def mappings(self):
        return self

    def one(self):
        return self._one

    def all(self):
        return self._all


class RepairManifestTests(unittest.TestCase):
    def _write_manifest(self, payload: dict) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "repairs.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_manifest_rejects_duplicate_conversations(self):
        item = {
            "conversation_id": "conv-1",
            "user_message_id": "user-1",
            "assistant_message_id": "assistant-1",
            "expected_conversation_created_at": "2026-07-13T23:17:16+00:00",
            "corrected_conversation_created_at": "2026-07-13T15:17:16+00:00",
            "expected_conversation_updated_at": "2026-07-13T23:17:18+00:00",
            "corrected_conversation_updated_at": "2026-07-13T15:17:18+00:00",
            "expected_user_created_at": "2026-07-13T23:17:16+00:00",
            "corrected_user_created_at": "2026-07-13T15:17:16+00:00",
            "expected_assistant_created_at": "2026-07-13T15:17:17+00:00",
        }
        path = self._write_manifest({"repairs": [item, {**item, "user_message_id": "user-2"}]})

        with self.assertRaisesRegex(ValueError, "conversation_id 不得重复"):
            repair_message_order._load_manifest(path)

    def test_manifest_loads_explicit_expected_and_corrected_values(self):
        path = self._write_manifest(
            {
                "repairs": [
                    {
                        "conversation_id": "conv-1",
                        "user_message_id": "user-1",
                        "assistant_message_id": "assistant-1",
                        "expected_conversation_created_at": "2026-07-13T23:17:16+00:00",
                        "corrected_conversation_created_at": "2026-07-13T15:17:16+00:00",
                        "expected_conversation_updated_at": "2026-07-13T23:17:18+00:00",
                        "corrected_conversation_updated_at": "2026-07-13T15:17:18+00:00",
                        "expected_user_created_at": "2026-07-13T23:17:16+00:00",
                        "corrected_user_created_at": "2026-07-13T15:17:16+00:00",
                        "expected_assistant_created_at": "2026-07-13T15:17:17+00:00",
                    }
                ]
            }
        )

        entry = repair_message_order._load_manifest(path)[0]

        self.assertEqual(entry.corrected_user_created_at.hour, 15)
        self.assertEqual(entry.corrected_conversation_created_at.hour, 15)
        self.assertEqual(entry.corrected_conversation_updated_at.hour, 15)

    def test_manifest_rejects_timestamp_without_timezone(self):
        path = self._write_manifest(
            {
                "repairs": [
                    {
                        "conversation_id": "conv-1",
                        "user_message_id": "user-1",
                        "assistant_message_id": "assistant-1",
                        "expected_conversation_created_at": "2026-07-13T23:17:16",
                        "corrected_conversation_created_at": "2026-07-13T15:17:16+00:00",
                        "expected_conversation_updated_at": "2026-07-13T23:17:18+00:00",
                        "corrected_conversation_updated_at": "2026-07-13T15:17:18+00:00",
                        "expected_user_created_at": "2026-07-13T23:17:16+00:00",
                        "corrected_user_created_at": "2026-07-13T15:17:16+00:00",
                        "expected_assistant_created_at": "2026-07-13T15:17:17+00:00",
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "必须包含明确时区"):
            repair_message_order._load_manifest(path)


class RepairExecutionTests(unittest.TestCase):
    def test_validation_locks_conversation_and_messages(self):
        entry = make_entry()
        connection = MagicMock()
        connection.execute.side_effect = [
            MappingResult(
                one={
                    "id": entry.conversation_id,
                    "created_at": entry.expected_conversation_created_at,
                    "updated_at": entry.expected_conversation_updated_at,
                }
            ),
            MappingResult(
                all_rows=[
                    {
                        "id": entry.user_message_id,
                        "role": "user",
                        "created_at": entry.expected_user_created_at,
                        "sequence": None,
                    },
                    {
                        "id": entry.assistant_message_id,
                        "role": "assistant",
                        "created_at": entry.expected_assistant_created_at,
                        "sequence": None,
                    },
                ]
            ),
        ]

        repair_message_order._validate_entry(connection, entry)

        statements = [str(call.args[0]).upper() for call in connection.execute.call_args_list]
        self.assertTrue(all("FOR UPDATE" in statement for statement in statements))
        self.assertIn("UPDATED_AT", statements[0])

    def test_validation_rejects_changed_conversation_updated_at(self):
        entry = make_entry()
        connection = MagicMock()
        connection.execute.side_effect = [
            MappingResult(
                one={
                    "id": entry.conversation_id,
                    "created_at": entry.expected_conversation_created_at,
                    "updated_at": entry.expected_conversation_updated_at.replace(second=19),
                }
            ),
            MappingResult(
                all_rows=[
                    {
                        "id": entry.user_message_id,
                        "role": "user",
                        "created_at": entry.expected_user_created_at,
                        "sequence": None,
                    },
                    {
                        "id": entry.assistant_message_id,
                        "role": "assistant",
                        "created_at": entry.expected_assistant_created_at,
                        "sequence": None,
                    },
                ]
            ),
        ]

        with self.assertRaisesRegex(ValueError, "conversation.updated_at"):
            repair_message_order._validate_entry(connection, entry)

    def test_validation_rejects_assistant_with_existing_sequence(self):
        entry = make_entry()
        connection = MagicMock()
        connection.execute.side_effect = [
            MappingResult(
                one={
                    "id": entry.conversation_id,
                    "created_at": entry.expected_conversation_created_at,
                    "updated_at": entry.expected_conversation_updated_at,
                }
            ),
            MappingResult(
                all_rows=[
                    {
                        "id": entry.user_message_id,
                        "role": "user",
                        "created_at": entry.expected_user_created_at,
                        "sequence": None,
                    },
                    {
                        "id": entry.assistant_message_id,
                        "role": "assistant",
                        "created_at": entry.expected_assistant_created_at,
                        "sequence": 2,
                    },
                ]
            ),
        ]

        with self.assertRaisesRegex(ValueError, "已分配 sequence"):
            repair_message_order._validate_entry(connection, entry)

    def test_default_dry_run_never_calls_apply_and_rolls_back(self):
        engine = MagicMock()
        connection = engine.connect.return_value.__enter__.return_value
        transaction = connection.begin.return_value
        entry = make_entry()

        with (
            patch.object(repair_message_order, "_validate_entry") as validate,
            patch.object(repair_message_order, "_apply_entry") as apply_entry,
        ):
            repair_message_order.run_repairs(engine, [entry], apply=False)

        validate.assert_called_once_with(connection, entry)
        apply_entry.assert_not_called()
        transaction.rollback.assert_called_once()
        transaction.commit.assert_not_called()

    def test_validation_failure_rolls_back_without_apply(self):
        engine = MagicMock()
        connection = engine.connect.return_value.__enter__.return_value
        transaction = connection.begin.return_value

        with (
            patch.object(repair_message_order, "_validate_entry", side_effect=ValueError("不匹配")),
            patch.object(repair_message_order, "_apply_entry") as apply_entry,
            self.assertRaisesRegex(ValueError, "不匹配"),
        ):
            repair_message_order.run_repairs(engine, [make_entry()], apply=True)

        apply_entry.assert_not_called()
        transaction.rollback.assert_called_once()
        transaction.commit.assert_not_called()

    def test_apply_assigns_reserved_pair_and_checks_all_rowcounts(self):
        connection = MagicMock()
        connection.execute.side_effect = [
            SimpleNamespace(scalar_one=lambda: 101),
            SimpleNamespace(rowcount=1),
            SimpleNamespace(rowcount=1),
            SimpleNamespace(rowcount=1),
        ]

        sequences = repair_message_order._apply_entry(connection, make_entry())

        self.assertEqual(sequences, (101, 102))
        parameters = [call.args[1] for call in connection.execute.call_args_list[1:3]]
        self.assertEqual([params["sequence"] for params in parameters], [101, 102])
        self.assertEqual(parameters[0]["corrected_created_at"], make_entry().corrected_user_created_at)
        conversation_parameters = connection.execute.call_args_list[3].args[1]
        self.assertEqual(
            conversation_parameters["corrected_created_at"],
            make_entry().corrected_conversation_created_at,
        )
        self.assertEqual(
            conversation_parameters["corrected_updated_at"],
            make_entry().corrected_conversation_updated_at,
        )
        statements = [str(call.args[0]) for call in connection.execute.call_args_list]
        self.assertTrue(all("INTERVAL '8 hours'" not in statement for statement in statements))

    def test_apply_rejects_partial_update(self):
        connection = MagicMock()
        connection.execute.side_effect = [
            SimpleNamespace(scalar_one=lambda: 101),
            SimpleNamespace(rowcount=1),
            SimpleNamespace(rowcount=0),
            SimpleNamespace(rowcount=1),
        ]

        with self.assertRaisesRegex(RuntimeError, "更新行数不是 1"):
            repair_message_order._apply_entry(connection, make_entry())

    def test_apply_rejects_missing_conversation_update(self):
        connection = MagicMock()
        connection.execute.side_effect = [
            SimpleNamespace(scalar_one=lambda: 101),
            SimpleNamespace(rowcount=1),
            SimpleNamespace(rowcount=1),
            SimpleNamespace(rowcount=0),
        ]

        with self.assertRaisesRegex(RuntimeError, "conversation 更新行数不是 1"):
            repair_message_order._apply_entry(connection, make_entry())

    def test_script_help_works_from_file_entrypoint(self):
        result = subprocess.run(
            [sys.executable, "scripts/repair_message_order.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--apply", result.stdout)


if __name__ == "__main__":
    unittest.main()
