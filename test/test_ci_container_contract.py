import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CIContainerContractTest(unittest.TestCase):
    def test_development_dependencies_cover_runtime_and_ci(self) -> None:
        development = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

        self.assertIn("-r requirements.txt", development)
        self.assertIn("-r requirements-ci.txt", development)
        self.assertIn("pytest==9.1.1", development)

    def test_ci_dependencies_are_separate_from_production(self) -> None:
        production = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        ci = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8")

        self.assertNotIn("fakeredis", production)
        self.assertNotIn("-r requirements.txt", ci)
        self.assertIn("fakeredis[lua]", ci)
        self.assertIn("ruff==", ci)

    def test_auth_client_uses_fixed_pypi_release(self) -> None:
        production = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("seanfield-auth-client[fastapi]==0.3.1", production)
        self.assertNotIn("git+https://github.com/HyxiaoGe/auth-service", production)
        self.assertNotIn("#subdirectory=auth-client", production)

    def test_dockerfile_exposes_dependencies_and_production_targets(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("AS dependencies", dockerfile)
        self.assertIn("AS production", dockerfile)
        self.assertIn("COPY requirements-ci.txt", dockerfile)
        self.assertNotIn("python -m unittest discover", dockerfile)

        production = dockerfile[dockerfile.index("AS production") :]
        self.assertNotIn("build-essential", production)
        self.assertNotIn("gcc", production)
        self.assertNotIn("ruff", production)
        self.assertNotRegex(dockerfile, r"(?m)^\s*git\s*\\?\s*$")

    def test_windows_workflow_tests_ephemeral_production_container(self) -> None:
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        build_job = workflow[workflow.index("  build:") : workflow.index("  deploy-dev:")]
        script = (ROOT / "scripts/ci/run_windows_container_ci.ps1").read_text(encoding="utf-8")

        self.assertIn("run_windows_container_ci.ps1", build_job)
        self.assertIn('"build", "--target", "production"', script)
        self.assertIn("pip install --default-timeout=30 --no-cache-dir -r requirements-ci.txt", script)
        self.assertIn("python scripts/check_architecture.py", script)
        self.assertIn("ruff check .", script)
        self.assertIn("python -u -m unittest discover -s test -t . -v", script)

    def test_windows_ci_script_records_stages_and_cleans_container(self) -> None:
        script = (ROOT / "scripts/ci/run_windows_container_ci.ps1").read_text(encoding="utf-8")

        for stage in ("docker-build", "ci-dependencies", "architecture", "ruff", "unit-tests"):
            self.assertIn(stage, script)
        self.assertIn("ConvertTo-Json", script)
        self.assertIn("-Encoding utf8", script)
        self.assertIn("finally", script)
        self.assertIn('$ErrorActionPreference = "Continue"', script)
        self.assertIn("docker container inspect", script)
        self.assertIn("docker rm -f", script)
        self.assertIn("exit $failureExitCode", script)

    def test_windows_workflow_publishes_summary_and_failure_logs(self) -> None:
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        build_job = workflow[workflow.index("  build:") : workflow.index("  deploy-dev:")]

        self.assertIn("scripts\\ci\\run_windows_container_ci.ps1", build_job)
        self.assertIn("image-push", build_job)
        self.assertIn("GITHUB_STEP_SUMMARY", build_job)
        self.assertIn("[System.Collections.ArrayList]::new()", build_job)
        self.assertIn("foreach ($stage in $parsedStages)", build_job)
        self.assertIn("('Runner: `{0}`' -f '${{ runner.name }}')", build_job)
        self.assertIn("if: always()", build_job)
        self.assertIn("actions/upload-artifact@v4", build_job)
        self.assertIn("retention-days: 7", build_job)
        self.assertIn("_ci-logs", build_job)


if __name__ == "__main__":
    unittest.main()
