import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")
os.environ.setdefault("SERVER_HOST", "http://dev.example:8002")
os.environ.setdefault("FRONTEND_URL", "http://dev.example:3004")
os.environ.setdefault("AUTH_SERVICE_BASE_URL", "http://auth.example:8100")
os.environ.setdefault("AUTH_SERVICE_CLIENT_ID", "fusion-client")
os.environ.setdefault("AUTH_SERVICE_JWKS_URL", "http://auth.example:8100/.well-known/jwks.json")

from app.api.deps import (
    get_chat_service,
    get_file_service,
    get_model_credential_repo,
    get_model_source_repo,
    get_provider_repo,
    get_user_memory_service,
)
from app.db.repositories import (
    ModelCredentialRepository,
    ModelSourceRepository,
    ProviderRepository,
)
from app.services.chat_service import ChatService
from app.services.file_service import FileService
from app.services.user_memory_service import UserMemoryService


class TestDepsFactories(unittest.TestCase):
    """deps 工厂函数返回正确类型的实例"""

    def setUp(self):
        self.mock_db = MagicMock()

    def test_get_chat_service(self):
        svc = get_chat_service(db=self.mock_db)
        self.assertIsInstance(svc, ChatService)
        self.assertIs(svc.db, self.mock_db)

    @patch("app.services.file_service.get_storage", return_value=MagicMock())
    def test_get_file_service(self, _mock_storage):
        svc = get_file_service(db=self.mock_db)
        self.assertIsInstance(svc, FileService)
        self.assertIs(svc.db, self.mock_db)

    def test_get_user_memory_service(self):
        svc = get_user_memory_service(db=self.mock_db)
        self.assertIsInstance(svc, UserMemoryService)
        self.assertIs(svc.db, self.mock_db)

    def test_get_provider_repo(self):
        repo = get_provider_repo(db=self.mock_db)
        self.assertIsInstance(repo, ProviderRepository)

    def test_get_model_source_repo(self):
        repo = get_model_source_repo(db=self.mock_db)
        self.assertIsInstance(repo, ModelSourceRepository)

    def test_get_model_credential_repo(self):
        repo = get_model_credential_repo(db=self.mock_db)
        self.assertIsInstance(repo, ModelCredentialRepository)
