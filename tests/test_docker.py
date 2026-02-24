"""Tests for Docker configuration files."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


class TestDockerfile:
    """Dockerfile should be well-formed for CLI usage."""

    def test_exists(self) -> None:
        assert (ROOT / "Dockerfile").exists()

    def test_uses_python_slim(self) -> None:
        content = (ROOT / "Dockerfile").read_text()
        assert "python:3.11-slim" in content

    def test_sets_entrypoint(self) -> None:
        content = (ROOT / "Dockerfile").read_text()
        assert "ENTRYPOINT" in content
        assert "src.cli" in content

    def test_sets_pythonpath(self) -> None:
        content = (ROOT / "Dockerfile").read_text()
        assert "PYTHONPATH=/app" in content

    def test_copies_source(self) -> None:
        content = (ROOT / "Dockerfile").read_text()
        assert "COPY src/ src/" in content
        assert "COPY configs/ configs/" in content

    def test_creates_data_dirs(self) -> None:
        content = (ROOT / "Dockerfile").read_text()
        assert "/app/data" in content
        assert "/app/prompts" in content


class TestDockerfileDev:
    """Dockerfile.dev should include dev tools."""

    def test_exists(self) -> None:
        assert (ROOT / "Dockerfile.dev").exists()

    def test_includes_git(self) -> None:
        content = (ROOT / "Dockerfile.dev").read_text()
        assert "git" in content

    def test_sets_pythonpath(self) -> None:
        content = (ROOT / "Dockerfile.dev").read_text()
        assert "PYTHONPATH=/app" in content


class TestDockerCompose:
    """docker-compose.yml should define expected services."""

    def test_exists(self) -> None:
        assert (ROOT / "docker-compose.yml").exists()

    def test_valid_yaml(self) -> None:
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert isinstance(data, dict)

    def test_has_prompteval_service(self) -> None:
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert "prompteval" in data["services"]

    def test_has_dev_service(self) -> None:
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert "dev" in data["services"]

    def test_prompteval_volumes(self) -> None:
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        svc = data["services"]["prompteval"]
        assert "volumes" in svc
        volume_strs = [str(v) for v in svc["volumes"]]
        assert any("/app/data" in v for v in volume_strs)

    def test_environment_vars(self) -> None:
        content = (ROOT / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        svc = data["services"]["prompteval"]
        env = svc.get("environment", [])
        env_str = str(env)
        assert "OPENAI_API_KEY" in env_str
        assert "ANTHROPIC_API_KEY" in env_str


class TestDockerignore:
    """Dockerignore should exclude dev artifacts."""

    def test_exists(self) -> None:
        assert (ROOT / ".dockerignore").exists()

    def test_excludes_git(self) -> None:
        content = (ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_excludes_env(self) -> None:
        content = (ROOT / ".dockerignore").read_text()
        assert ".env" in content

    def test_excludes_venv(self) -> None:
        content = (ROOT / ".dockerignore").read_text()
        assert "venv" in content

    def test_excludes_pycache(self) -> None:
        content = (ROOT / ".dockerignore").read_text()
        assert "__pycache__" in content


class TestMakefile:
    """Makefile should have docker targets."""

    def test_exists(self) -> None:
        assert (ROOT / "Makefile").exists()

    def test_has_docker_build(self) -> None:
        content = (ROOT / "Makefile").read_text()
        assert "docker-build" in content

    def test_has_docker_run(self) -> None:
        content = (ROOT / "Makefile").read_text()
        assert "docker-run" in content

    def test_has_docker_dev(self) -> None:
        content = (ROOT / "Makefile").read_text()
        assert "docker-dev" in content

    def test_has_test_target(self) -> None:
        content = (ROOT / "Makefile").read_text()
        assert "test:" in content

    def test_has_lint_target(self) -> None:
        content = (ROOT / "Makefile").read_text()
        assert "lint:" in content
