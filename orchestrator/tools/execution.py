"""Execution tools for running commands, tests, and linters with sandboxing."""

import os
import subprocess
import shutil
import json
import re
import signal
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import tempfile
import threading


class ExecutionMode(str, Enum):
    """Execution modes with different permission levels."""
    RESTRICTED = "restricted"  # Only pre-approved commands
    STANDARD = "standard"  # Common dev tools allowed
    ELEVATED = "elevated"  # More tools, still sandboxed
    UNRESTRICTED = "unrestricted"  # Full access (use with caution)


@dataclass
class ExecutionConfig:
    """Configuration for command execution."""
    working_dir: str
    mode: ExecutionMode = ExecutionMode.STANDARD
    timeout_seconds: int = 300  # 5 minutes default
    max_output_bytes: int = 1024 * 1024  # 1MB
    allowed_commands: List[str] = field(default_factory=list)
    denied_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        "sudo",
        "su ",
        "chmod 777",
        "curl | sh",
        "wget | sh",
        "eval",
        "exec",
        "> /dev/sd",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",  # Fork bomb
    ])
    environment_allowlist: List[str] = field(default_factory=lambda: [
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TERM",
        "SHELL",
        "NODE_ENV",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "GOPATH",
        "CARGO_HOME",
        "RUSTUP_HOME",
    ])
    inject_env: Dict[str, str] = field(default_factory=dict)


class ExecutionError(Exception):
    """Raised when command execution fails or is blocked."""
    pass


class CommandDeniedError(ExecutionError):
    """Raised when a command is explicitly denied."""
    pass


class ExecutionTools:
    """
    Sandboxed command execution for agent use.

    Provides controlled execution of shell commands, test runners,
    and linters with configurable safety controls.
    """

    # Pre-approved commands for STANDARD mode
    STANDARD_COMMANDS = {
        # Package managers
        "npm": ["install", "test", "run", "build", "lint", "ci", "audit", "outdated", "ls"],
        "yarn": ["install", "test", "run", "build", "lint", "audit"],
        "pnpm": ["install", "test", "run", "build", "lint"],
        "pip": ["install", "list", "show", "check", "freeze"],
        "pip3": ["install", "list", "show", "check", "freeze"],
        "poetry": ["install", "build", "run", "check", "show"],
        "cargo": ["build", "test", "run", "check", "clippy", "fmt", "doc"],
        "go": ["build", "test", "run", "mod", "fmt", "vet", "generate"],

        # Test runners
        "pytest": None,  # All subcommands allowed
        "jest": None,
        "mocha": None,
        "vitest": None,
        "cargo-test": None,
        "go-test": None,

        # BDD/Cucumber test frameworks
        "cucumber": None,
        "cucumber-js": None,
        "behave": None,  # Python BDD
        "lettuce": None,  # Python BDD
        "radish": None,  # Python BDD
        "gauge": None,  # Thoughtworks Gauge

        # Linters & formatters
        "eslint": None,
        "prettier": None,
        "black": None,
        "ruff": None,
        "flake8": None,
        "mypy": None,
        "pylint": None,
        "rustfmt": None,
        "gofmt": None,
        "rubocop": None,
        "tsc": None,  # TypeScript compiler

        # Build tools
        "make": None,
        "cmake": None,
        "gradle": ["build", "test", "check", "assemble"],
        "mvn": ["compile", "test", "package", "verify"],

        # Utilities
        "cat": None,
        "head": None,
        "tail": None,
        "grep": None,
        "find": None,
        "ls": None,
        "pwd": None,
        "echo": None,
        "wc": None,
        "sort": None,
        "uniq": None,
        "diff": None,
        "which": None,
        "env": None,
        "printenv": None,
    }

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self._operation_log: List[Dict[str, Any]] = []
        self._active_processes: Dict[str, subprocess.Popen] = {}

    def _is_command_allowed(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is allowed to run.

        Returns:
            Tuple of (allowed, reason)
        """
        # Check denied patterns first
        for denied in self.config.denied_commands:
            if denied in command:
                return False, f"Command contains denied pattern: {denied}"

        # In UNRESTRICTED mode, allow everything not explicitly denied
        if self.config.mode == ExecutionMode.UNRESTRICTED:
            return True, "Unrestricted mode"

        # Parse the command to get the base executable
        parts = command.split()
        if not parts:
            return False, "Empty command"

        base_cmd = os.path.basename(parts[0])

        # Check explicit allowlist
        if base_cmd in self.config.allowed_commands:
            return True, "Explicitly allowed"

        # In RESTRICTED mode, only explicit allowlist
        if self.config.mode == ExecutionMode.RESTRICTED:
            if base_cmd not in self.config.allowed_commands:
                return False, f"Command '{base_cmd}' not in allowlist (restricted mode)"
            return True, "In allowlist"

        # STANDARD and ELEVATED modes check pre-approved commands
        if base_cmd in self.STANDARD_COMMANDS:
            allowed_subcommands = self.STANDARD_COMMANDS[base_cmd]

            # None means all subcommands allowed
            if allowed_subcommands is None:
                return True, f"Command '{base_cmd}' is pre-approved"

            # Check if subcommand is allowed
            if len(parts) > 1:
                subcommand = parts[1]
                if subcommand in allowed_subcommands:
                    return True, f"Subcommand '{subcommand}' is pre-approved for '{base_cmd}'"
                else:
                    return False, f"Subcommand '{subcommand}' not allowed for '{base_cmd}'"

            return True, f"Command '{base_cmd}' is pre-approved"

        # ELEVATED mode allows more commands
        if self.config.mode == ExecutionMode.ELEVATED:
            # Allow common development commands
            elevated_commands = [
                "node", "python", "python3", "ruby", "php",
                "java", "javac", "kotlin", "scala",
                "docker", "docker-compose",
                "kubectl", "helm",
                "terraform", "ansible",
                "curl", "wget", "http",
            ]
            if base_cmd in elevated_commands:
                return True, f"Command '{base_cmd}' allowed in elevated mode"

        return False, f"Command '{base_cmd}' not allowed in {self.config.mode.value} mode"

    def _prepare_environment(self) -> Dict[str, str]:
        """Prepare sanitized environment variables."""
        env = {}

        # Copy allowed environment variables
        for key in self.config.environment_allowlist:
            if key in os.environ:
                env[key] = os.environ[key]

        # Inject custom environment variables
        env.update(self.config.inject_env)

        return env

    def _log_operation(
        self,
        operation: str,
        command: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        """Log an execution operation."""
        self._operation_log.append({
            "operation": operation,
            "command": command,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result,
            "error": error,
        })

    def run_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        shell: bool = True,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            working_dir: Working directory (default: config.working_dir)
            timeout: Timeout in seconds (default: config.timeout_seconds)
            capture_output: Capture stdout/stderr
            shell: Run through shell
            env_vars: Additional environment variables

        Returns:
            Dict with exit code, stdout, stderr, and execution info
        """
        # Check if command is allowed
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            self._log_operation("run_command_denied", command, error=reason)
            raise CommandDeniedError(f"Command denied: {reason}")

        cwd = working_dir or self.config.working_dir
        if not os.path.isdir(cwd):
            return {
                "success": False,
                "error": f"Working directory does not exist: {cwd}",
            }

        timeout_secs = timeout or self.config.timeout_seconds
        env = self._prepare_environment()
        if env_vars:
            env.update(env_vars)

        start_time = datetime.utcnow()

        try:
            result = subprocess.run(
                command,
                shell=shell,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout_secs,
                env=env,
            )

            duration = (datetime.utcnow() - start_time).total_seconds()

            # Truncate output if too large
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[:self.config.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"

            if len(stderr) > self.config.max_output_bytes:
                stderr = stderr[:self.config.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"

            response = {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": command,
                "working_dir": cwd,
                "duration_seconds": duration,
            }

            self._log_operation("run_command", command, result=response)
            return response

        except subprocess.TimeoutExpired:
            self._log_operation("run_command_timeout", command, error="Timeout")
            return {
                "success": False,
                "error": f"Command timed out after {timeout_secs} seconds",
                "command": command,
                "timeout": timeout_secs,
            }
        except Exception as e:
            self._log_operation("run_command_error", command, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "command": command,
            }

    def run_tests(
        self,
        test_command: Optional[str] = None,
        test_path: Optional[str] = None,
        framework: Optional[str] = None,
        verbose: bool = False,
        coverage: bool = False,
        filter_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run tests with auto-detected or specified framework.

        Args:
            test_command: Explicit test command to run
            test_path: Path to test file/directory
            framework: Test framework (pytest, jest, etc.)
            verbose: Verbose output
            coverage: Enable coverage reporting
            filter_pattern: Pattern to filter tests

        Returns:
            Dict with test results
        """
        if test_command:
            return self.run_command(test_command)

        # Auto-detect test framework if not specified
        if not framework:
            framework = self._detect_test_framework()
            if not framework:
                return {
                    "success": False,
                    "error": "Could not detect test framework. Specify framework or test_command.",
                }

        # Build command based on framework
        cmd = self._build_test_command(
            framework=framework,
            test_path=test_path,
            verbose=verbose,
            coverage=coverage,
            filter_pattern=filter_pattern,
        )

        if not cmd:
            return {
                "success": False,
                "error": f"Unknown test framework: {framework}",
            }

        result = self.run_command(cmd)

        # Parse test results if possible
        if result["success"] or result.get("exit_code") == 1:  # Tests may fail with exit code 1
            parsed = self._parse_test_output(framework, result.get("stdout", ""), result.get("stderr", ""))
            result["test_summary"] = parsed

        return result

    def _detect_test_framework(self) -> Optional[str]:
        """Auto-detect test framework from project files."""
        cwd = self.config.working_dir

        # Check for BDD/Cucumber frameworks first (they may coexist with unit test frameworks)
        # Python BDD - Behave
        if os.path.exists(os.path.join(cwd, "behave.ini")) or \
           os.path.exists(os.path.join(cwd, "features")) and \
           any(f.endswith(".feature") for f in os.listdir(os.path.join(cwd, "features"))
               if os.path.isfile(os.path.join(cwd, "features", f))):
            # Check if behave is installed vs other Python BDD
            if os.path.exists(os.path.join(cwd, ".behaverc")) or \
               os.path.exists(os.path.join(cwd, "behave.ini")):
                return "behave"

        # JavaScript/TypeScript - check for Cucumber-js
        package_json = os.path.join(cwd, "package.json")
        if os.path.exists(package_json):
            try:
                with open(package_json) as f:
                    pkg = json.load(f)
                    deps = {**pkg.get("devDependencies", {}), **pkg.get("dependencies", {})}

                    # Cucumber-js detection
                    if "@cucumber/cucumber" in deps or "cucumber" in deps:
                        return "cucumber-js"
            except Exception:
                pass

        # Ruby Cucumber
        if os.path.exists(os.path.join(cwd, "cucumber.yml")) or \
           os.path.exists(os.path.join(cwd, "Gemfile")):
            gemfile_path = os.path.join(cwd, "Gemfile")
            if os.path.exists(gemfile_path):
                try:
                    with open(gemfile_path) as f:
                        content = f.read()
                        if "cucumber" in content.lower():
                            return "cucumber"
                except Exception:
                    pass

        # Gauge (Thoughtworks)
        if os.path.exists(os.path.join(cwd, "manifest.json")):
            try:
                with open(os.path.join(cwd, "manifest.json")) as f:
                    manifest = json.load(f)
                    if "Language" in manifest:  # Gauge manifest format
                        return "gauge"
            except Exception:
                pass

        # Python unit test frameworks
        if os.path.exists(os.path.join(cwd, "pytest.ini")) or \
           os.path.exists(os.path.join(cwd, "pyproject.toml")):
            return "pytest"

        # JavaScript/TypeScript unit test frameworks
        if os.path.exists(package_json):
            try:
                with open(package_json) as f:
                    pkg = json.load(f)
                    scripts = pkg.get("scripts", {})
                    deps = {**pkg.get("devDependencies", {}), **pkg.get("dependencies", {})}

                    if "vitest" in deps:
                        return "vitest"
                    if "jest" in deps or "test" in scripts and "jest" in scripts.get("test", ""):
                        return "jest"
                    if "mocha" in deps:
                        return "mocha"
            except Exception:
                pass

        # Rust
        if os.path.exists(os.path.join(cwd, "Cargo.toml")):
            return "cargo"

        # Go
        if any(f.endswith(".go") for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))):
            return "go"

        return None

    def _build_test_command(
        self,
        framework: str,
        test_path: Optional[str],
        verbose: bool,
        coverage: bool,
        filter_pattern: Optional[str],
    ) -> Optional[str]:
        """Build test command for a framework."""
        commands = {
            # Unit test frameworks
            "pytest": self._build_pytest_command,
            "jest": self._build_jest_command,
            "vitest": self._build_vitest_command,
            "mocha": self._build_mocha_command,
            "cargo": self._build_cargo_test_command,
            "go": self._build_go_test_command,
            # BDD/Cucumber frameworks
            "cucumber": self._build_cucumber_command,
            "cucumber-js": self._build_cucumber_js_command,
            "behave": self._build_behave_command,
            "gauge": self._build_gauge_command,
        }

        builder = commands.get(framework)
        if builder:
            return builder(test_path, verbose, coverage, filter_pattern)
        return None

    def _build_pytest_command(self, path, verbose, coverage, filter_pattern):
        cmd = ["pytest"]
        if verbose:
            cmd.append("-v")
        if coverage:
            cmd.extend(["--cov", "--cov-report=term-missing"])
        if filter_pattern:
            cmd.extend(["-k", filter_pattern])
        if path:
            cmd.append(path)
        return " ".join(cmd)

    def _build_jest_command(self, path, verbose, coverage, filter_pattern):
        cmd = ["npx", "jest"]
        if verbose:
            cmd.append("--verbose")
        if coverage:
            cmd.append("--coverage")
        if filter_pattern:
            cmd.extend(["--testNamePattern", filter_pattern])
        if path:
            cmd.append(path)
        return " ".join(cmd)

    def _build_vitest_command(self, path, verbose, coverage, filter_pattern):
        cmd = ["npx", "vitest", "run"]
        if coverage:
            cmd.append("--coverage")
        if filter_pattern:
            cmd.extend(["--testNamePattern", filter_pattern])
        if path:
            cmd.append(path)
        return " ".join(cmd)

    def _build_mocha_command(self, path, verbose, coverage, filter_pattern):
        cmd = ["npx", "mocha"]
        if filter_pattern:
            cmd.extend(["--grep", filter_pattern])
        if path:
            cmd.append(path)
        return " ".join(cmd)

    def _build_cargo_test_command(self, path, verbose, coverage, filter_pattern):
        cmd = ["cargo", "test"]
        if verbose:
            cmd.append("--verbose")
        if filter_pattern:
            cmd.append(filter_pattern)
        cmd.append("--")
        if verbose:
            cmd.append("--nocapture")
        return " ".join(cmd)

    def _build_go_test_command(self, path, verbose, coverage, filter_pattern):
        cmd = ["go", "test"]
        if verbose:
            cmd.append("-v")
        if coverage:
            cmd.append("-cover")
        if filter_pattern:
            cmd.extend(["-run", filter_pattern])
        cmd.append(path or "./...")
        return " ".join(cmd)

    def _build_cucumber_command(self, path, verbose, coverage, filter_pattern):
        """Build Ruby Cucumber command."""
        cmd = ["bundle", "exec", "cucumber"]
        if verbose:
            cmd.append("--verbose")
        if filter_pattern:
            cmd.extend(["--tags", filter_pattern])
        if path:
            cmd.append(path)
        else:
            cmd.append("features/")
        return " ".join(cmd)

    def _build_cucumber_js_command(self, path, verbose, coverage, filter_pattern):
        """Build Cucumber-js command."""
        cmd = ["npx", "cucumber-js"]
        if filter_pattern:
            cmd.extend(["--tags", filter_pattern])
        if path:
            cmd.append(path)
        else:
            cmd.append("features/")
        # Add format for better output
        cmd.extend(["--format", "progress-bar"])
        return " ".join(cmd)

    def _build_behave_command(self, path, verbose, coverage, filter_pattern):
        """Build Python Behave command."""
        cmd = ["behave"]
        if verbose:
            cmd.append("--verbose")
        if filter_pattern:
            cmd.extend(["--tags", filter_pattern])
        if path:
            cmd.append(path)
        # Add summary format
        cmd.extend(["--format", "progress"])
        return " ".join(cmd)

    def _build_gauge_command(self, path, verbose, coverage, filter_pattern):
        """Build Gauge command."""
        cmd = ["gauge", "run"]
        if verbose:
            cmd.append("--verbose")
        if filter_pattern:
            cmd.extend(["--tags", filter_pattern])
        if path:
            cmd.append(path)
        else:
            cmd.append("specs/")
        return " ".join(cmd)

    def _parse_test_output(self, framework: str, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse test output to extract summary."""
        output = stdout + stderr

        if framework == "pytest":
            # Look for pytest summary line
            match = re.search(r"(\d+) passed", output)
            passed = int(match.group(1)) if match else 0

            match = re.search(r"(\d+) failed", output)
            failed = int(match.group(1)) if match else 0

            match = re.search(r"(\d+) error", output)
            errors = int(match.group(1)) if match else 0

            match = re.search(r"(\d+) skipped", output)
            skipped = int(match.group(1)) if match else 0

            return {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "total": passed + failed + errors + skipped,
            }

        elif framework in ["jest", "vitest"]:
            # Jest/Vitest summary
            match = re.search(r"Tests:\s+(\d+) passed", output)
            passed = int(match.group(1)) if match else 0

            match = re.search(r"(\d+) failed", output)
            failed = int(match.group(1)) if match else 0

            return {
                "passed": passed,
                "failed": failed,
                "total": passed + failed,
            }

        elif framework in ["cucumber", "cucumber-js"]:
            # Cucumber output: "X scenarios (Y passed, Z failed)"
            scenarios_match = re.search(r"(\d+) scenarios?", output)
            scenarios = int(scenarios_match.group(1)) if scenarios_match else 0

            passed_match = re.search(r"(\d+) passed", output)
            passed = int(passed_match.group(1)) if passed_match else 0

            failed_match = re.search(r"(\d+) failed", output)
            failed = int(failed_match.group(1)) if failed_match else 0

            pending_match = re.search(r"(\d+) pending", output)
            pending = int(pending_match.group(1)) if pending_match else 0

            skipped_match = re.search(r"(\d+) skipped", output)
            skipped = int(skipped_match.group(1)) if skipped_match else 0

            # Also capture step counts
            steps_match = re.search(r"(\d+) steps?", output)
            steps = int(steps_match.group(1)) if steps_match else 0

            return {
                "scenarios": scenarios,
                "steps": steps,
                "passed": passed,
                "failed": failed,
                "pending": pending,
                "skipped": skipped,
                "total": scenarios,
            }

        elif framework == "behave":
            # Behave output: "X features passed, Y failed, Z skipped"
            # "X scenarios passed, Y failed, Z skipped"
            # "X steps passed, Y failed, Z skipped, W undefined"
            features_match = re.search(r"(\d+) features? passed", output)
            features_passed = int(features_match.group(1)) if features_match else 0

            scenarios_match = re.search(r"(\d+) scenarios? passed", output)
            scenarios_passed = int(scenarios_match.group(1)) if scenarios_match else 0

            scenarios_failed_match = re.search(r"(\d+) scenarios? failed", output)
            scenarios_failed = int(scenarios_failed_match.group(1)) if scenarios_failed_match else 0

            steps_passed_match = re.search(r"(\d+) steps? passed", output)
            steps_passed = int(steps_passed_match.group(1)) if steps_passed_match else 0

            steps_failed_match = re.search(r"(\d+) steps? failed", output)
            steps_failed = int(steps_failed_match.group(1)) if steps_failed_match else 0

            return {
                "features_passed": features_passed,
                "scenarios_passed": scenarios_passed,
                "scenarios_failed": scenarios_failed,
                "steps_passed": steps_passed,
                "steps_failed": steps_failed,
                "passed": scenarios_passed,
                "failed": scenarios_failed,
                "total": scenarios_passed + scenarios_failed,
            }

        elif framework == "gauge":
            # Gauge output parsing
            specs_match = re.search(r"Specifications:\s+(\d+)", output)
            specs = int(specs_match.group(1)) if specs_match else 0

            scenarios_match = re.search(r"Scenarios:\s+(\d+)", output)
            scenarios = int(scenarios_match.group(1)) if scenarios_match else 0

            passed_match = re.search(r"Passed:\s+(\d+)", output)
            passed = int(passed_match.group(1)) if passed_match else 0

            failed_match = re.search(r"Failed:\s+(\d+)", output)
            failed = int(failed_match.group(1)) if failed_match else 0

            return {
                "specifications": specs,
                "scenarios": scenarios,
                "passed": passed,
                "failed": failed,
                "total": specs,
            }

        return {"raw": True}

    def run_linter(
        self,
        linter: Optional[str] = None,
        path: Optional[str] = None,
        fix: bool = False,
        config_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a linter/formatter.

        Args:
            linter: Linter to run (auto-detected if not specified)
            path: Path to lint (default: current directory)
            fix: Auto-fix issues if supported
            config_file: Custom config file

        Returns:
            Dict with linting results
        """
        if not linter:
            linter = self._detect_linter()
            if not linter:
                return {
                    "success": False,
                    "error": "Could not detect linter. Specify linter explicitly.",
                }

        cmd = self._build_linter_command(linter, path, fix, config_file)
        if not cmd:
            return {
                "success": False,
                "error": f"Unknown linter: {linter}",
            }

        result = self.run_command(cmd)

        # Parse linter output
        if result.get("stdout") or result.get("stderr"):
            parsed = self._parse_linter_output(linter, result.get("stdout", ""), result.get("stderr", ""))
            result["issues"] = parsed

        return result

    def _detect_linter(self) -> Optional[str]:
        """Auto-detect linter from project files."""
        cwd = self.config.working_dir

        # Python
        if os.path.exists(os.path.join(cwd, "ruff.toml")) or \
           os.path.exists(os.path.join(cwd, ".ruff.toml")):
            return "ruff"

        if os.path.exists(os.path.join(cwd, "pyproject.toml")):
            # Check for ruff or black in pyproject.toml
            return "ruff"  # Default to ruff for Python

        # JavaScript/TypeScript
        if os.path.exists(os.path.join(cwd, ".eslintrc.js")) or \
           os.path.exists(os.path.join(cwd, ".eslintrc.json")) or \
           os.path.exists(os.path.join(cwd, "eslint.config.js")):
            return "eslint"

        # Rust
        if os.path.exists(os.path.join(cwd, "Cargo.toml")):
            return "clippy"

        # Go
        if any(f.endswith(".go") for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))):
            return "go-vet"

        return None

    def _build_linter_command(
        self,
        linter: str,
        path: Optional[str],
        fix: bool,
        config_file: Optional[str],
    ) -> Optional[str]:
        """Build linter command."""
        path = path or "."

        commands = {
            "ruff": f"ruff check {'--fix' if fix else ''} {f'--config {config_file}' if config_file else ''} {path}",
            "black": f"black {'--check' if not fix else ''} {path}",
            "flake8": f"flake8 {f'--config {config_file}' if config_file else ''} {path}",
            "mypy": f"mypy {f'--config-file {config_file}' if config_file else ''} {path}",
            "pylint": f"pylint {f'--rcfile {config_file}' if config_file else ''} {path}",
            "eslint": f"npx eslint {'--fix' if fix else ''} {f'--config {config_file}' if config_file else ''} {path}",
            "prettier": f"npx prettier {'--write' if fix else '--check'} {path}",
            "clippy": "cargo clippy -- -D warnings",
            "rustfmt": f"cargo fmt {'--' if not fix else ''} {'--check' if not fix else ''}",
            "go-vet": "go vet ./...",
            "gofmt": f"gofmt {'-w' if fix else '-l'} {path}",
        }

        return commands.get(linter)

    def _parse_linter_output(self, linter: str, stdout: str, stderr: str) -> List[Dict[str, Any]]:
        """Parse linter output to extract issues."""
        output = stdout + stderr
        issues = []

        if linter in ["ruff", "flake8", "pylint"]:
            # Format: file:line:col: code message
            for line in output.split("\n"):
                match = re.match(r"(.+):(\d+):(\d+): (\w+) (.+)", line)
                if match:
                    issues.append({
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "column": int(match.group(3)),
                        "code": match.group(4),
                        "message": match.group(5),
                    })

        elif linter == "eslint":
            # ESLint output format
            for line in output.split("\n"):
                match = re.match(r"\s+(\d+):(\d+)\s+(error|warning)\s+(.+?)\s+(\S+)$", line)
                if match:
                    issues.append({
                        "line": int(match.group(1)),
                        "column": int(match.group(2)),
                        "severity": match.group(3),
                        "message": match.group(4),
                        "rule": match.group(5),
                    })

        elif linter == "mypy":
            # mypy format: file:line: error: message
            for line in output.split("\n"):
                match = re.match(r"(.+):(\d+): (error|warning|note): (.+)", line)
                if match:
                    issues.append({
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "severity": match.group(3),
                        "message": match.group(4),
                    })

        return issues

    def run_build(
        self,
        build_command: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a build command.

        Args:
            build_command: Explicit build command
            target: Build target (for make, gradle, etc.)

        Returns:
            Dict with build results
        """
        if build_command:
            return self.run_command(build_command)

        # Auto-detect build system
        cwd = self.config.working_dir

        if os.path.exists(os.path.join(cwd, "package.json")):
            return self.run_command("npm run build")

        if os.path.exists(os.path.join(cwd, "Cargo.toml")):
            cmd = f"cargo build {'--release' if target == 'release' else ''}"
            return self.run_command(cmd)

        if os.path.exists(os.path.join(cwd, "Makefile")):
            cmd = f"make {target or ''}"
            return self.run_command(cmd)

        if os.path.exists(os.path.join(cwd, "build.gradle")) or \
           os.path.exists(os.path.join(cwd, "build.gradle.kts")):
            return self.run_command("./gradlew build")

        if os.path.exists(os.path.join(cwd, "pom.xml")):
            return self.run_command("mvn package")

        if os.path.exists(os.path.join(cwd, "go.mod")):
            return self.run_command("go build ./...")

        return {
            "success": False,
            "error": "Could not detect build system. Specify build_command explicitly.",
        }

    def run_type_check(
        self,
        checker: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run type checking.

        Args:
            checker: Type checker (mypy, tsc, etc.)
            path: Path to check

        Returns:
            Dict with type check results
        """
        cwd = self.config.working_dir
        path = path or "."

        if not checker:
            # Auto-detect
            if os.path.exists(os.path.join(cwd, "tsconfig.json")):
                checker = "tsc"
            elif os.path.exists(os.path.join(cwd, "pyproject.toml")) or \
                 os.path.exists(os.path.join(cwd, "mypy.ini")):
                checker = "mypy"

        if checker == "tsc":
            return self.run_command("npx tsc --noEmit")
        elif checker == "mypy":
            return self.run_command(f"mypy {path}")
        elif checker == "pyright":
            return self.run_command(f"pyright {path}")

        return {
            "success": False,
            "error": "Could not detect type checker. Specify checker explicitly.",
        }

    def install_dependencies(
        self,
        manager: Optional[str] = None,
        packages: Optional[List[str]] = None,
        dev: bool = False,
    ) -> Dict[str, Any]:
        """
        Install project dependencies.

        Args:
            manager: Package manager (npm, pip, cargo, etc.)
            packages: Specific packages to install
            dev: Install as dev dependency

        Returns:
            Dict with installation results
        """
        cwd = self.config.working_dir

        if not manager:
            # Auto-detect
            if os.path.exists(os.path.join(cwd, "package.json")):
                if os.path.exists(os.path.join(cwd, "pnpm-lock.yaml")):
                    manager = "pnpm"
                elif os.path.exists(os.path.join(cwd, "yarn.lock")):
                    manager = "yarn"
                else:
                    manager = "npm"
            elif os.path.exists(os.path.join(cwd, "requirements.txt")) or \
                 os.path.exists(os.path.join(cwd, "pyproject.toml")):
                manager = "pip"
            elif os.path.exists(os.path.join(cwd, "Cargo.toml")):
                manager = "cargo"
            elif os.path.exists(os.path.join(cwd, "go.mod")):
                manager = "go"

        if not manager:
            return {
                "success": False,
                "error": "Could not detect package manager.",
            }

        # Build install command
        if packages:
            pkg_str = " ".join(packages)
            commands = {
                "npm": f"npm install {'-D' if dev else ''} {pkg_str}",
                "yarn": f"yarn add {'-D' if dev else ''} {pkg_str}",
                "pnpm": f"pnpm add {'-D' if dev else ''} {pkg_str}",
                "pip": f"pip install {pkg_str}",
                "cargo": f"cargo add {pkg_str}",
                "go": f"go get {pkg_str}",
            }
        else:
            commands = {
                "npm": "npm install",
                "yarn": "yarn install",
                "pnpm": "pnpm install",
                "pip": "pip install -r requirements.txt" if os.path.exists(os.path.join(cwd, "requirements.txt")) else "pip install -e .",
                "cargo": "cargo build",
                "go": "go mod download",
            }

        cmd = commands.get(manager)
        if not cmd:
            return {
                "success": False,
                "error": f"Unknown package manager: {manager}",
            }

        return self.run_command(cmd)

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the log of all execution operations."""
        return self._operation_log.copy()


# =============================================================================
# Factory and Singleton
# =============================================================================

_execution_tools: Optional[ExecutionTools] = None


def get_execution_tools(working_dir: Optional[str] = None) -> ExecutionTools:
    """Get or create execution tools instance."""
    global _execution_tools

    if _execution_tools is None or working_dir:
        cwd = working_dir or os.getcwd()
        config = ExecutionConfig(working_dir=cwd)
        _execution_tools = ExecutionTools(config)

    return _execution_tools


def configure_execution_tools(config: ExecutionConfig) -> ExecutionTools:
    """Configure execution tools with custom settings."""
    global _execution_tools
    _execution_tools = ExecutionTools(config)
    return _execution_tools
