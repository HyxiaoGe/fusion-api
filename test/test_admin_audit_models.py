import unittest
from pathlib import Path

from app.db.models import (
    AdminAuditEvent,
    AgentStep,
    ModelAdmissionOperation,
    ModelCatalogControl,
    PerformanceRun,
)


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

    def test_model_catalog_control_is_visibility_only_and_migration_never_deletes_models(self):
        self.assertTrue(ModelCatalogControl.__table__.c.model_id.unique)
        self.assertIn("selectable", ModelCatalogControl.__table__.c)
        self.assertIn("routable", ModelCatalogControl.__table__.c)
        self.assertIn("revision", ModelCatalogControl.__table__.c)
        self.assertIn("lease_token_hash", ModelAdmissionOperation.__table__.c)
        self.assertNotIn("lease_token", ModelAdmissionOperation.__table__.c)
        self.assertIn("catalog_invalidated_at", ModelAdmissionOperation.__table__.c)

        migrations = list(Path("alembic/versions").glob("*_add_model_catalog_controls.py"))
        self.assertEqual(len(migrations), 1)
        migration = migrations[0].read_text()
        self.assertIn("model_catalog_controls", migration)
        self.assertIn("model_admission_operations", migration)
        self.assertIn("uq_model_admission_operations_single_running", migration)
        self.assertIn("CHECK", migration.upper())
        self.assertNotIn("/model/delete", migration)


if __name__ == "__main__":
    unittest.main()
