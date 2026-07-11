import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CIContainerContractTest(unittest.TestCase):
    def test_ci_dependencies_are_separate_from_production(self) -> None:
        production = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        ci = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8")

        self.assertNotIn("fakeredis", production)
        self.assertIn("-r requirements.txt", ci)
        self.assertIn("fakeredis[lua]", ci)
        self.assertIn("ruff==", ci)

    def test_dockerfile_exposes_ci_and_production_targets(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("AS dependencies", dockerfile)
        self.assertIn("AS ci", dockerfile)
        self.assertIn("AS production", dockerfile)
        self.assertIn("COPY requirements-ci.txt", dockerfile)
        self.assertNotIn("python -m unittest discover", dockerfile)

        production = dockerfile[dockerfile.index("AS production") :]
        self.assertNotIn("build-essential", production)
        self.assertNotIn("gcc", production)
        self.assertNotIn("ruff", production)

    def test_windows_workflow_builds_ci_before_production(self) -> None:
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        build_job = workflow[workflow.index("  build:") : workflow.index("  deploy-dev:")]

        self.assertIn('"--target", "ci"', build_job)
        self.assertIn('"--target", "production"', build_job)
        self.assertIn('"--no-cache-filter", "ci"', build_job)
        self.assertNotIn("pip install --default-timeout=30 --no-cache-dir ruff", build_job)
        self.assertIn("python scripts/check_architecture.py", build_job)
        self.assertIn("ruff check .", build_job)
        self.assertIn("python -u -m unittest discover -s test -t . -v", build_job)


if __name__ == "__main__":
    unittest.main()
