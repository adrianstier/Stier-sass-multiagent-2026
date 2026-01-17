"""Tests for git tools module."""

import os
import tempfile
import subprocess
import pytest

from orchestrator.tools.git_tools import (
    GitTools,
    GitConfig,
    GitOperationType,
    GitOperationError,
)


class TestGitConfig:
    """Tests for GitConfig."""

    def test_default_protected_branches(self):
        """Default config should protect main branches."""
        config = GitConfig(
            repo_path="/tmp/repo",
            allowed_operations=[GitOperationType.READ]
        )
        assert "main" in config.protected_branches
        assert "master" in config.protected_branches

    def test_custom_protected_branches(self):
        """Custom protected branches should be used."""
        config = GitConfig(
            repo_path="/tmp/repo",
            allowed_operations=[GitOperationType.READ],
            protected_branches=["develop", "release"]
        )
        assert "develop" in config.protected_branches
        assert "release" in config.protected_branches


class TestGitOperationTypes:
    """Tests for operation type permissions."""

    def test_read_operations(self):
        """READ type should be available."""
        read_ops = GitOperationType.READ
        assert read_ops == "read"

    def test_write_operations(self):
        """WRITE type should be available."""
        write_ops = GitOperationType.WRITE
        assert write_ops == "write"

    def test_dangerous_operations(self):
        """DANGEROUS type should exist for risky operations."""
        dangerous_ops = GitOperationType.DANGEROUS
        assert dangerous_ops == "dangerous"


class TestGitTools:
    """Tests for GitTools."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary git repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=tmpdir, capture_output=True
            )

            # Create initial commit
            test_file = os.path.join(tmpdir, "README.md")
            with open(test_file, "w") as f:
                f.write("# Test Repo\n")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=tmpdir, capture_output=True
            )

            yield tmpdir

    @pytest.fixture
    def git_tools_read_only(self, git_repo):
        """Create GitTools with read-only access."""
        config = GitConfig(
            repo_path=git_repo,
            allowed_operations=[GitOperationType.READ]
        )
        return GitTools(config)

    @pytest.fixture
    def git_tools_full(self, git_repo):
        """Create GitTools with full access."""
        config = GitConfig(
            repo_path=git_repo,
            allowed_operations=[
                GitOperationType.READ,
                GitOperationType.WRITE,
                GitOperationType.BRANCH,
            ]
        )
        return GitTools(config)

    def test_status_success(self, git_tools_read_only, git_repo):
        """Should return git status."""
        result = git_tools_read_only.status()
        assert result["success"] is True

    def test_log_success(self, git_tools_read_only):
        """Should return git log."""
        result = git_tools_read_only.log(count=5)
        assert result["success"] is True

    def test_diff_success(self, git_tools_read_only, git_repo):
        """Should return git diff."""
        # Modify a file
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nNew line")

        result = git_tools_read_only.diff()
        assert result["success"] is True

    def test_show_commit(self, git_tools_read_only):
        """Should show commit details."""
        result = git_tools_read_only.show("HEAD")
        assert result["success"] is True

    def test_write_requires_permission(self, git_tools_read_only, git_repo):
        """Should raise error for write operations without permission."""
        # Create a new file
        with open(os.path.join(git_repo, "new_file.txt"), "w") as f:
            f.write("test")

        with pytest.raises(GitOperationError):
            git_tools_read_only.add("new_file.txt")

    def test_blame(self, git_tools_read_only, git_repo):
        """Should return blame information."""
        result = git_tools_read_only.blame("README.md")
        assert result["success"] is True
