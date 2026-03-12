import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import User
from app.db.repositories import FileRepository


class FileRepositoryTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()

        self.session.add(
            User(
                id="user-123",
                username="tester",
                email="tester@example.com",
            )
        )
        self.session.commit()

    def tearDown(self):
        self.session.close()

    def test_create_file_persists_user_id(self):
        repo = FileRepository(self.session)

        saved = repo.create_file(
            {
                "id": "file-123",
                "user_id": "user-123",
                "filename": "file-123_note.txt",
                "original_filename": "note.txt",
                "mimetype": "text/plain",
                "size": 12,
                "path": "/tmp/file-123_note.txt",
                "status": "parsing",
                "processing_result": None,
            }
        )

        self.assertEqual(saved.id, "file-123")
        self.assertEqual(saved.user_id, "user-123")


if __name__ == "__main__":
    unittest.main()
