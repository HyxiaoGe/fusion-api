import unittest
from pathlib import Path

from app.db.models import AdminAuditEvent, AgentStep, PerformanceRun


class AdminAuditModelTests(unittest.TestCase):
    def test_agent_step_model_declares_cascading_session_fk(self):
        foreign_keys = list(AgentStep.__table__.c.trace_id.foreign_keys)

        self.assertEqual(len(foreign_keys), 1)
        self.assertEqual(foreign_keys[0].target_fullname, "agent_sessions.id")
        self.assertEqual(foreign_keys[0].ondelete, "CASCADE")

    def test_audit_and_performance_models_only_store_safe_summary_fields(self):
        self.assertIn("admin_snapshot", AdminAuditEvent.__table__.c)
        self.assertIn("metadata", AdminAuditEvent.__table__.c)
        self.assertNotIn("content", AdminAuditEvent.__table__.c)
        self.assertIn("safe_summary", PerformanceRun.__table__.c)
        self.assertNotIn("conversation_ids", PerformanceRun.__table__.c)

    def test_migration_uses_non_destructive_not_valid_fk(self):
        migration = Path("alembic/versions/b8d4f7a1c2e6_add_admin_audit_center.py").read_text()

        self.assertIn("FOREIGN KEY (trace_id) REFERENCES agent_sessions(id) ON DELETE CASCADE NOT VALID", migration)
        self.assertNotIn("DELETE FROM agent_steps", migration)


if __name__ == "__main__":
    unittest.main()
