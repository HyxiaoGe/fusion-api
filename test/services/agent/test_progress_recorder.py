from unittest.mock import Mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentProgressSnapshot
from app.services.agent.progress_recorder import AgentProgressRecorder


def test_recorder_ignores_non_agent_event_chunks():
    db = Mock()
    recorder = AgentProgressRecorder(
        db=db,
        run_id="r1",
        conversation_id="c1",
        message_id="m1",
        user_id="u1",
    )

    recorder.record_chunk("c1", "answering", {"delta": "x"})

    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_recorder_upserts_snapshot_for_v2_event():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        recorder = AgentProgressRecorder(
            db=db,
            run_id="r1",
            conversation_id="c1",
            message_id="m1",
            user_id="u1",
        )

        recorder.record_chunk(
            "c1",
            "agent_event",
            {
                "type": "run_progress_updated",
                "protocol_version": 2,
                "phase": "planning",
                "label": "正在理解问题",
            },
        )
        recorder.record_chunk(
            "c1",
            "agent_event",
            {
                "type": "plan_snapshot",
                "protocol_version": 2,
                "plan_id": "plan-r1",
                "revision": 1,
                "items": [
                    {
                        "id": "understand",
                        "title": "理解问题",
                        "status": "running",
                        "kind": "reasoning",
                        "tool_names": [],
                        "evidence_item_ids": [],
                    }
                ],
            },
        )

        row = db.query(AgentProgressSnapshot).filter_by(run_id="r1").one()
        assert row.conversation_id == "c1"
        assert row.message_id == "m1"
        assert row.user_id == "u1"
        assert row.protocol_version == 2
        assert row.state["progress"]["label"] == "正在理解问题"
        assert row.state["plan"]["plan_id"] == "plan-r1"
    finally:
        db.close()
        engine.dispose()


def test_recorder_rolls_back_and_swallows_db_failure():
    db = Mock()
    db.query.side_effect = RuntimeError("db down")
    recorder = AgentProgressRecorder(
        db=db,
        run_id="r1",
        conversation_id="c1",
        message_id="m1",
        user_id="u1",
    )

    recorder.record_chunk(
        "c1",
        "agent_event",
        {
            "type": "run_progress_updated",
            "protocol_version": 2,
            "phase": "planning",
            "label": "正在理解问题",
        },
    )

    db.rollback.assert_called_once()
