import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.repositories import ModelSourceRepository, ProviderRepository


class ProviderRepositoryTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()

    def tearDown(self):
        self.session.close()

    def test_create_and_get_provider(self):
        repo = ProviderRepository(self.session)
        provider = repo.create(
            {
                "id": "qwen",
                "name": "通义千问",
                "auth_config": {
                    "fields": [
                        {"name": "api_key", "display_name": "API Key", "type": "password", "required": True}
                    ],
                    "auth_type": "api_key",
                },
                "litellm_prefix": "openai",
                "custom_base_url": True,
                "priority": 10,
            }
        )
        self.assertEqual(provider.id, "qwen")
        self.assertEqual(provider.name, "通义千问")
        self.assertTrue(provider.custom_base_url)

        fetched = repo.get_by_id("qwen")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.litellm_prefix, "openai")


class ModelSourceRepositoryTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()

        # 先创建 provider
        provider_repo = ProviderRepository(self.session)
        provider_repo.create(
            {
                "id": "qwen",
                "name": "通义千问",
                "auth_config": {
                    "fields": [
                        {"name": "api_key", "display_name": "API Key", "type": "password", "required": True}
                    ],
                    "auth_type": "api_key",
                },
                "litellm_prefix": "openai",
                "custom_base_url": True,
            }
        )

    def tearDown(self):
        self.session.close()

    def test_create_model_and_get_auth_from_provider(self):
        repo = ModelSourceRepository(self.session)

        created = repo.create(
            {
                "modelId": "qwen-max",
                "name": "Qwen Max",
                "provider": "qwen",
                "capabilities": {"deepThinking": True},
                "pricing": {"input": 0.001, "output": 0.002, "unit": "USD"},
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
                "description": "test",
            }
        )

        self.assertEqual(created.provider, "qwen")
        self.assertIsNotNone(created.provider_rel)

        schema = repo.to_full_schema(created)
        self.assertEqual(schema.auth_config.auth_type, "api_key")
        self.assertEqual(schema.auth_config.fields[0].name, "api_key")
        self.assertEqual(schema.model_configuration.params[0].name, "temperature")


if __name__ == "__main__":
    unittest.main()
