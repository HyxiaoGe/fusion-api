import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.repositories import ModelSourceRepository


class ModelSourceRepositoryTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()

    def tearDown(self):
        self.session.close()

    def test_create_inherits_provider_templates_and_priority(self):
        repo = ModelSourceRepository(self.session)

        repo.create(
            {
                "modelId": "qwen-template",
                "name": "Qwen Template",
                "provider": "qwen",
                "knowledgeCutoff": "2025-01",
                "capabilities": {"deepThinking": False, "fileSupport": True},
                "pricing": {"input": 0.001, "output": 0.002, "unit": "USD"},
                "auth_config": {
                    "fields": [
                        {
                            "name": "api_key",
                            "display_name": "API Key",
                            "type": "password",
                            "required": True,
                        }
                    ],
                    "auth_type": "api_key",
                },
                "model_configuration": {
                    "params": [
                        {
                            "name": "temperature",
                            "display_name": "温度",
                            "type": "number",
                            "default": 0.7,
                            "min": 0,
                            "max": 2,
                        }
                    ]
                },
                "priority": 1,
                "enabled": True,
                "description": "template",
            }
        )

        created = repo.create(
            {
                "modelId": "qwen-custom",
                "name": "Qwen Custom",
                "provider": "qwen",
                "knowledgeCutoff": "2026-03",
                "capabilities": {"deepThinking": True, "fileSupport": False},
                "pricing": {"input": 0.0, "output": 0.0, "unit": "USD"},
                "priority": 10,
                "enabled": True,
                "description": "custom",
            }
        )

        self.assertEqual(created.priority, 10)
        self.assertIsNotNone(created.auth_config)
        self.assertIsNotNone(created.model_configuration)

        schema = repo.to_full_schema(created)
        self.assertEqual(schema.priority, 10)
        self.assertEqual(schema.auth_config.auth_type, "api_key")
        self.assertEqual(schema.auth_config.fields[0].name, "api_key")
        self.assertEqual(schema.model_configuration.params[0].name, "temperature")


if __name__ == "__main__":
    unittest.main()
