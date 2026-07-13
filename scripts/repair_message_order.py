"""按显式清单修复已确认的消息时间与顺序。

脚本默认 dry-run。只有清单逐项通过以下校验后，``--apply`` 才会写入：
1. 会话恰好两条消息；
2. user/assistant ID、角色、会话 created_at/updated_at 和消息预期时间完全匹配；
3. 两条消息尚未分配 sequence。

清单格式：
{
  "repairs": [
    {
      "conversation_id": "...",
      "user_message_id": "...",
      "assistant_message_id": "...",
      "expected_conversation_created_at": "2026-07-13T23:17:16.000000+00:00",
      "corrected_conversation_created_at": "2026-07-13T15:17:16.000000+00:00",
      "expected_conversation_updated_at": "2026-07-13T23:17:18.000000+00:00",
      "corrected_conversation_updated_at": "2026-07-13T15:17:18.000000+00:00",
      "expected_user_created_at": "2026-07-13T23:17:16.000000+00:00",
      "corrected_user_created_at": "2026-07-13T15:17:16.000000+00:00",
      "expected_assistant_created_at": "2026-07-13T15:17:17.000000+00:00"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402


@dataclass(frozen=True)
class RepairEntry:
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    expected_conversation_created_at: datetime
    corrected_conversation_created_at: datetime
    expected_conversation_updated_at: datetime
    corrected_conversation_updated_at: datetime
    expected_user_created_at: datetime
    corrected_user_created_at: datetime
    expected_assistant_created_at: datetime


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("清单时间必须包含明确时区")
    return parsed


def _load_manifest(path: Path) -> list[RepairEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("repairs")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("清单 repairs 必须是非空数组")
    entries = [
        RepairEntry(
            conversation_id=str(item["conversation_id"]),
            user_message_id=str(item["user_message_id"]),
            assistant_message_id=str(item["assistant_message_id"]),
            expected_conversation_created_at=_parse_timestamp(item["expected_conversation_created_at"]),
            corrected_conversation_created_at=_parse_timestamp(item["corrected_conversation_created_at"]),
            expected_conversation_updated_at=_parse_timestamp(item["expected_conversation_updated_at"]),
            corrected_conversation_updated_at=_parse_timestamp(item["corrected_conversation_updated_at"]),
            expected_user_created_at=_parse_timestamp(item["expected_user_created_at"]),
            corrected_user_created_at=_parse_timestamp(item["corrected_user_created_at"]),
            expected_assistant_created_at=_parse_timestamp(item["expected_assistant_created_at"]),
        )
        for item in raw_entries
    ]
    if len({entry.conversation_id for entry in entries}) != len(entries):
        raise ValueError("清单 conversation_id 不得重复")
    message_ids = [
        message_id for entry in entries for message_id in (entry.user_message_id, entry.assistant_message_id)
    ]
    if len(set(message_ids)) != len(message_ids):
        raise ValueError("清单 message_id 不得重复")
    return entries


def _validate_entry(connection, entry: RepairEntry) -> None:
    conversation = (
        connection.execute(
            text("SELECT id, created_at, updated_at FROM conversations WHERE id = :id FOR UPDATE"),
            {"id": entry.conversation_id},
        )
        .mappings()
        .one()
    )
    messages = (
        connection.execute(
            text(
                "SELECT id, role, created_at, sequence FROM messages "
                "WHERE conversation_id = :conversation_id ORDER BY id FOR UPDATE"
            ),
            {"conversation_id": entry.conversation_id},
        )
        .mappings()
        .all()
    )
    if len(messages) != 2:
        raise ValueError(f"{entry.conversation_id}: 消息数不是 2")
    by_id = {str(message["id"]): message for message in messages}
    user = by_id.get(entry.user_message_id)
    assistant = by_id.get(entry.assistant_message_id)
    if user is None or user["role"] != "user":
        raise ValueError(f"{entry.conversation_id}: user 消息白名单不匹配")
    if assistant is None or assistant["role"] != "assistant":
        raise ValueError(f"{entry.conversation_id}: assistant 消息白名单不匹配")
    if user["sequence"] is not None or assistant["sequence"] is not None:
        raise ValueError(f"{entry.conversation_id}: 消息已分配 sequence，拒绝重复修复")
    expected_values = (
        ("conversation.created_at", conversation["created_at"], entry.expected_conversation_created_at),
        ("conversation.updated_at", conversation["updated_at"], entry.expected_conversation_updated_at),
        ("user.created_at", user["created_at"], entry.expected_user_created_at),
        ("assistant.created_at", assistant["created_at"], entry.expected_assistant_created_at),
    )
    for label, actual, expected in expected_values:
        if actual != expected:
            raise ValueError(f"{entry.conversation_id}: {label} 与清单预期不一致")


def _apply_entry(connection, entry: RepairEntry) -> tuple[int, int]:
    user_sequence = int(connection.execute(text("SELECT nextval('message_order_sequence')")).scalar_one())
    assistant_sequence = user_sequence + 1
    user_update = connection.execute(
        text(
            "UPDATE messages SET created_at = :corrected_created_at, sequence = :sequence "
            "WHERE id = :id AND conversation_id = :conversation_id AND role = 'user'"
        ),
        {
            "id": entry.user_message_id,
            "conversation_id": entry.conversation_id,
            "corrected_created_at": entry.corrected_user_created_at,
            "sequence": user_sequence,
        },
    )
    assistant_update = connection.execute(
        text(
            "UPDATE messages SET sequence = :sequence "
            "WHERE id = :id AND conversation_id = :conversation_id AND role = 'assistant'"
        ),
        {
            "id": entry.assistant_message_id,
            "conversation_id": entry.conversation_id,
            "sequence": assistant_sequence,
        },
    )
    conversation_update = connection.execute(
        text(
            "UPDATE conversations "
            "SET created_at = :corrected_created_at, updated_at = :corrected_updated_at "
            "WHERE id = :id"
        ),
        {
            "id": entry.conversation_id,
            "corrected_created_at": entry.corrected_conversation_created_at,
            "corrected_updated_at": entry.corrected_conversation_updated_at,
        },
    )
    rowcounts = (
        ("user 消息", user_update.rowcount),
        ("assistant 消息", assistant_update.rowcount),
        ("conversation", conversation_update.rowcount),
    )
    for label, rowcount in rowcounts:
        if rowcount != 1:
            raise RuntimeError(f"{entry.conversation_id}: {label} 更新行数不是 1")
    return user_sequence, assistant_sequence


def run_repairs(engine, entries: list[RepairEntry], *, apply: bool) -> None:
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            for entry in entries:
                _validate_entry(connection, entry)
            if apply:
                for entry in entries:
                    user_sequence, assistant_sequence = _apply_entry(connection, entry)
                    print(
                        f"APPLY {entry.conversation_id}: "
                        f"user_sequence={user_sequence}, assistant_sequence={assistant_sequence}"
                    )
                transaction.commit()
            else:
                for entry in entries:
                    print(f"DRY-RUN {entry.conversation_id}: 校验通过，未写入")
                transaction.rollback()
        except Exception:
            transaction.rollback()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="修复显式白名单中的消息时区错位与顺序")
    parser.add_argument("--manifest", type=Path, required=True, help="显式修复清单 JSON")
    parser.add_argument("--apply", action="store_true", help="实际提交；缺省为 dry-run")
    args = parser.parse_args()

    entries = _load_manifest(args.manifest)
    engine = create_engine(settings.DATABASE_URL)
    run_repairs(engine, entries, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
