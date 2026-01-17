"""Tests for execution tools module."""

import os
import tempfile
import json
import pytest

from orchestrator.tools.execution import (
    ExecutionTools,
    ExecutionConfig,
    ExecutionMode,
    ExecutionError,
    CommandDeniedError,
)


class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    def test_default_denied_commands(self):
        """Default config should deny dangerous commands."""
        config = ExecutionConfig(working_dir="/tmp")
        assert "rm -rf /" in config.denied_commands
        assert "sudo" in config.denied_commands

    def test_default_mode_is_standard(self):
        """Default mode should be STANDARD."""
        config = ExecutionConfig(working_dir="/tmp")
        assert config.mode == ExecutionMode.STANDARD


class TestExecutionTools:
    """Tests for ExecutionTools."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def exec_tools(self, temp_dir):
        """Create ExecutionTools with temp directory."""
        config = ExecutionConfig(
            working_dir=temp_dir,
            mode=ExecutionMode.STANDARD,
            timeout_seconds=30,
        )
        return ExecutionTools(config)

    @pytest.fixture
    def restricted_tools(self, temp_dir):
        """Create ExecutionTools in restricted mode."""
        config = ExecutionConfig(
            working_dir=temp_dir,
            mode=ExecutionMode.RESTRICTED,
            timeout_seconds=30,
        )
        return ExecutionTools(config)

    def test_run_simple_command(self, exec_tools):
        """Should run simple allowed commands."""
        result = exec_tools.run_command("echo 'hello world'")
        assert result["success"] is True
        assert "hello world" in result["stdout"]

    def test_run_ls_command(self, exec_tools, temp_dir):
        """Should run ls command."""
        # Create a test file
        open(os.path.join(temp_dir, "test.txt"), "w").close()

        result = exec_tools.run_command("ls")
        assert result["success"] is True
        assert "test.txt" in result["stdout"]

    def test_dangerous_command_blocked(self, exec_tools):
        """Should block dangerous commands."""
        with pytest.raises(CommandDeniedError):
            exec_tools.run_command("rm -rf /")

    def test_sudo_blocked(self, exec_tools):
        """Should block sudo commands."""
        with pytest.raises(CommandDeniedError):
            exec_tools.run_command("sudo ls")

    def test_command_timeout(self, temp_dir):
        """Should timeout long-running commands."""
        # Note: 'sleep' is not in STANDARD_COMMANDS, so we use allowed_commands
        config = ExecutionConfig(
            working_dir=temp_dir,
            mode=ExecutionMode.RESTRICTED,
            timeout_seconds=1,
            allowed_commands=["sleep"],
        )
        exec_tools = ExecutionTools(config)

        result = exec_tools.run_command("sleep 10")
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_working_directory_respected(self, exec_tools, temp_dir):
        """Should run commands in the configured working directory."""
        result = exec_tools.run_command("pwd")
        assert result["success"] is True
        assert temp_dir in result["stdout"]

    def test_custom_working_directory(self, exec_tools, temp_dir):
        """Should allow custom working directory within bounds."""
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)

        result = exec_tools.run_command("pwd", working_dir=subdir)
        assert result["success"] is True
        assert subdir in result["stdout"]

    def test_restricted_mode_limits_commands(self, restricted_tools):
        """Restricted mode should only allow pre-approved commands."""
        # In restricted mode, echo is not in the allowlist by default
        # So this should raise CommandDeniedError
        with pytest.raises(CommandDeniedError):
            restricted_tools.run_command("echo test")


class TestTestFrameworkDetection:
    """Tests for test framework detection."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_detect_pytest(self, temp_dir):
        """Should detect pytest from pytest.ini."""
        open(os.path.join(temp_dir, "pytest.ini"), "w").close()

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "pytest"

    def test_detect_pytest_from_pyproject(self, temp_dir):
        """Should detect pytest from pyproject.toml."""
        with open(os.path.join(temp_dir, "pyproject.toml"), "w") as f:
            f.write("[tool.pytest]\n")

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "pytest"

    def test_detect_jest(self, temp_dir):
        """Should detect jest from package.json."""
        package = {
            "devDependencies": {"jest": "^29.0.0"}
        }
        with open(os.path.join(temp_dir, "package.json"), "w") as f:
            json.dump(package, f)

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "jest"

    def test_detect_vitest(self, temp_dir):
        """Should detect vitest from package.json."""
        package = {
            "devDependencies": {"vitest": "^1.0.0"}
        }
        with open(os.path.join(temp_dir, "package.json"), "w") as f:
            json.dump(package, f)

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "vitest"

    def test_detect_cucumber_js(self, temp_dir):
        """Should detect cucumber-js from package.json."""
        package = {
            "devDependencies": {"@cucumber/cucumber": "^9.0.0"}
        }
        with open(os.path.join(temp_dir, "package.json"), "w") as f:
            json.dump(package, f)

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "cucumber-js"

    def test_detect_behave(self, temp_dir):
        """Should detect behave from behave.ini."""
        open(os.path.join(temp_dir, "behave.ini"), "w").close()

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "behave"

    def test_detect_cargo(self, temp_dir):
        """Should detect cargo from Cargo.toml."""
        with open(os.path.join(temp_dir, "Cargo.toml"), "w") as f:
            f.write("[package]\nname = 'test'\n")

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        framework = exec_tools._detect_test_framework()
        assert framework == "cargo"


class TestCommandBuilders:
    """Tests for test command builders."""

    @pytest.fixture
    def exec_tools(self):
        config = ExecutionConfig(working_dir="/tmp")
        return ExecutionTools(config)

    def test_build_pytest_command(self, exec_tools):
        """Should build correct pytest command."""
        cmd = exec_tools._build_pytest_command(
            path="tests/",
            verbose=True,
            coverage=True,
            filter_pattern="test_user"
        )
        assert "pytest" in cmd
        assert "-v" in cmd
        assert "--cov" in cmd
        assert "-k test_user" in cmd
        assert "tests/" in cmd

    def test_build_jest_command(self, exec_tools):
        """Should build correct jest command."""
        cmd = exec_tools._build_jest_command(
            path="src/__tests__",
            verbose=True,
            coverage=True,
            filter_pattern="user"
        )
        assert "jest" in cmd
        assert "--verbose" in cmd
        assert "--coverage" in cmd
        assert "--testNamePattern" in cmd

    def test_build_cucumber_command(self, exec_tools):
        """Should build correct cucumber command."""
        cmd = exec_tools._build_cucumber_command(
            path=None,
            verbose=True,
            coverage=False,
            filter_pattern="@smoke"
        )
        assert "cucumber" in cmd
        assert "--verbose" in cmd
        assert "--tags @smoke" in cmd
        assert "features/" in cmd

    def test_build_behave_command(self, exec_tools):
        """Should build correct behave command."""
        cmd = exec_tools._build_behave_command(
            path="features/login.feature",
            verbose=True,
            coverage=False,
            filter_pattern="@wip"
        )
        assert "behave" in cmd
        assert "--verbose" in cmd
        assert "--tags @wip" in cmd
        assert "features/login.feature" in cmd


class TestOutputParsing:
    """Tests for test output parsing."""

    @pytest.fixture
    def exec_tools(self):
        config = ExecutionConfig(working_dir="/tmp")
        return ExecutionTools(config)

    def test_parse_pytest_output(self, exec_tools):
        """Should parse pytest output."""
        output = """
        ===== 10 passed, 2 failed, 1 error, 3 skipped in 5.23s =====
        """
        result = exec_tools._parse_test_output("pytest", output, "")

        assert result["passed"] == 10
        assert result["failed"] == 2
        assert result["errors"] == 1
        assert result["skipped"] == 3

    def test_parse_jest_output(self, exec_tools):
        """Should parse jest output."""
        output = """
        Tests:  15 passed, 3 failed, 18 total
        """
        result = exec_tools._parse_test_output("jest", output, "")

        assert result["passed"] == 15
        assert result["failed"] == 3

    def test_parse_cucumber_output(self, exec_tools):
        """Should parse cucumber output."""
        output = """
        5 scenarios (4 passed, 1 failed)
        20 steps (18 passed, 1 failed, 1 pending)
        """
        result = exec_tools._parse_test_output("cucumber", output, "")

        assert result["scenarios"] == 5
        assert result["passed"] == 4
        assert result["failed"] == 1
        assert result["steps"] == 20

    def test_parse_behave_output(self, exec_tools):
        """Should parse behave output."""
        # The parser looks for "X scenarios passed" and "X scenarios failed" separately
        # Real behave output format: "X features passed, Y failed, Z skipped"
        output = """
        2 features passed, 0 features failed
        10 scenarios passed, 2 scenarios failed
        50 steps passed, 5 steps failed
        """
        result = exec_tools._parse_test_output("behave", output, "")

        assert result["features_passed"] == 2
        assert result["scenarios_passed"] == 10
        assert result["scenarios_failed"] == 2


class TestLinterDetection:
    """Tests for linter detection."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_detect_ruff(self, temp_dir):
        """Should detect ruff from ruff.toml."""
        open(os.path.join(temp_dir, "ruff.toml"), "w").close()

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        linter = exec_tools._detect_linter()
        assert linter == "ruff"

    def test_detect_eslint(self, temp_dir):
        """Should detect eslint from .eslintrc."""
        open(os.path.join(temp_dir, ".eslintrc.js"), "w").close()

        config = ExecutionConfig(working_dir=temp_dir)
        exec_tools = ExecutionTools(config)

        linter = exec_tools._detect_linter()
        assert linter == "eslint"
