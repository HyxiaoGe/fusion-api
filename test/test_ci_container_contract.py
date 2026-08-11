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

    def test_pr_and_release_workflows_run_equivalent_container_tests(self) -> None:
        release_workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        pr_workflow = (ROOT / ".github/workflows/pr-ci.yml").read_text(encoding="utf-8")
        windows_build_script = (ROOT / ".github/scripts/windows-build-and-test.ps1").read_text(encoding="utf-8")
        linux_build_script = (ROOT / ".github/scripts/linux-build-and-test.sh").read_text(encoding="utf-8")

        self.assertIn(".github/scripts/windows-build-and-test.ps1", release_workflow)
        self.assertIn(".github/scripts/linux-build-and-test.sh", pr_workflow)
        for build_script in (windows_build_script, linux_build_script):
            self.assertIn("docker build --target production", build_script)
            self.assertIn(
                "pip install --default-timeout=30 --no-cache-dir -r requirements-ci.txt",
                build_script,
            )
            self.assertIn("python scripts/check_architecture.py", build_script)
            self.assertIn("ruff check .", build_script)
            self.assertIn("python -u -m unittest discover -s test -t . -v", build_script)

        self.assertIn(
            '--mount "type=bind,source=${PWD}/README.md,target=/app/README.md,readonly"',
            linux_build_script,
        )
        self.assertIn(
            '--mount "type=bind,source=$((Get-Location).Path)\\README.md,target=/app/README.md,readonly"',
            windows_build_script,
        )

    def test_windows_release_build_disables_registry_incompatible_attestations(self) -> None:
        windows_build_script = (ROOT / ".github/scripts/windows-build-and-test.ps1").read_text(encoding="utf-8")
        build_commands = [
            line.strip()
            for line in windows_build_script.splitlines()
            if line.strip().startswith("docker build ")
        ]

        self.assertEqual(3, len(build_commands))
        for command in build_commands:
            self.assertIn("--provenance=false", command)


if __name__ == "__main__":
    unittest.main()
