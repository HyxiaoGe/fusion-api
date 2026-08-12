import unittest
from pathlib import Path


class KnowledgeDeployConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
        cls.milvus_compose = Path("ops/knowledge/milvus-compose.yml").read_text(encoding="utf-8")
        cls.fusion_override = Path("ops/knowledge/fusion-compose.override.yml").read_text(encoding="utf-8")

    def test_real_acceptance_stack_pins_private_authenticated_milvus(self):
        self.assertIn("milvusdb/milvus:v2.6.21", self.milvus_compose)
        self.assertIn('COMMON_SECURITY_AUTHORIZATIONENABLED: "true"', self.milvus_compose)
        self.assertIn('"127.0.0.1:19530:19530"', self.milvus_compose)
        self.assertIn("name: fusion_knowledge_milvus", self.milvus_compose)

    def test_worker_is_independent_and_shares_no_port(self):
        worker = self.fusion_override.split("knowledge-worker:", 1)[1]
        self.assertIn('["python", "-m", "scripts.run_knowledge_worker"]', worker)
        self.assertIn("./storage/files:/app/storage/files", worker)
        self.assertIn("fusion-knowledge-milvus", worker)
        self.assertNotIn("ports:", worker)

    def test_deploy_tracks_worker_identity_health_and_rollback(self):
        required = (
            "fusion-knowledge-worker",
            "knowledge_worker_image_id",
            "knowledge_worker_supported",
            "ROLLBACK_KNOWLEDGE_WORKER_EXISTED",
            "--profile knowledge-worker up -d",
            "--health-max-age-seconds 120",
            "DEPLOY_MILVUS_PASSWORD: ${{ secrets.MILVUS_PASSWORD }}",
            "DEPLOY_KNOWLEDGE_BASE_ENABLED: ${{ vars.KNOWLEDGE_BASE_ENABLED || 'false' }}",
        )
        for marker in required:
            self.assertIn(marker, self.workflow)
        self.assertNotIn("MILVUS_BOOTSTRAP_PASSWORD", self.workflow)


if __name__ == "__main__":
    unittest.main()
