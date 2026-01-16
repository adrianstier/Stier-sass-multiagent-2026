"""Tests for the tools module."""

import pytest
from orchestrator.tools.executor import ToolExecutor, ToolResult
from orchestrator.tools.registry import (
    get_tools_for_role, get_tool_definitions_for_role,
    ROLE_ALLOWLISTS, TOOL_REGISTRY
)


class TestToolRegistry:
    """Tests for tool registry functionality."""

    def test_role_allowlists_exist(self):
        """Test that all expected roles have allowlists."""
        expected_roles = [
            "orchestrator", "business_analyst", "project_manager",
            "ux_engineer", "tech_lead", "database_engineer",
            "backend_engineer", "frontend_engineer",
            "code_reviewer", "security_reviewer"
        ]
        for role in expected_roles:
            assert role in ROLE_ALLOWLISTS
            assert isinstance(ROLE_ALLOWLISTS[role], list)

    def test_get_tools_for_role(self):
        """Test getting tools for a specific role."""
        ba_tools = get_tools_for_role("business_analyst")
        assert "create_artifact" in ba_tools
        assert "read_artifact" in ba_tools
        assert "list_artifacts" in ba_tools

    def test_reviewer_has_gate_tools(self):
        """Test that reviewers have gate approval tools."""
        cr_tools = get_tools_for_role("code_reviewer")
        sec_tools = get_tools_for_role("security_reviewer")

        assert "approve_gate" in cr_tools
        assert "reject_gate" in cr_tools
        assert "approve_gate" in sec_tools
        assert "reject_gate" in sec_tools

    def test_non_reviewer_lacks_gate_tools(self):
        """Test that non-reviewers don't have gate tools."""
        ba_tools = get_tools_for_role("business_analyst")
        be_tools = get_tools_for_role("backend_engineer")

        assert "approve_gate" not in ba_tools
        assert "approve_gate" not in be_tools

    def test_get_tool_definitions(self):
        """Test getting tool definitions for LLM context."""
        definitions = get_tool_definitions_for_role("business_analyst")

        assert isinstance(definitions, list)
        for defn in definitions:
            assert "name" in defn
            assert "description" in defn
            assert "parameters" in defn


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    def test_can_execute_allowed_tool(self):
        """Test that allowed tools can be executed."""
        executor = ToolExecutor("business_analyst", "run-1", "task-1")
        assert executor.can_execute("create_artifact") is True
        assert executor.can_execute("read_artifact") is True

    def test_cannot_execute_disallowed_tool(self):
        """Test that disallowed tools are blocked."""
        executor = ToolExecutor("business_analyst", "run-1", "task-1")
        assert executor.can_execute("approve_gate") is False

    def test_execute_unauthorized_tool(self):
        """Test executing an unauthorized tool returns error."""
        executor = ToolExecutor("business_analyst", "run-1", "task-1")
        result = executor.execute("approve_gate", run_id="run-1", gate_type="code_review")

        assert result.success is False
        assert "not allowed" in result.error

    def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool returns error."""
        executor = ToolExecutor("orchestrator", "run-1", "task-1")
        # First add it to allowlist for test
        ROLE_ALLOWLISTS["orchestrator"].append("fake_tool")

        result = executor.execute("fake_tool")

        assert result.success is False
        assert "not found" in result.error

        # Clean up
        ROLE_ALLOWLISTS["orchestrator"].remove("fake_tool")

    def test_execution_log(self):
        """Test that executions are logged."""
        executor = ToolExecutor("business_analyst", "run-1", "task-1")

        # Attempt a disallowed tool (will fail but still logged)
        executor.execute("approve_gate", run_id="run-1", gate_type="code_review")

        log = executor.get_execution_log()
        assert len(log) == 1
        assert log[0]["tool_name"] == "approve_gate"
        assert log[0]["error"] is not None

    def test_sensitive_data_redaction(self):
        """Test that sensitive data is redacted in logs."""
        executor = ToolExecutor("business_analyst", "run-1", "task-1")

        # The executor should redact sensitive keys
        redacted = executor._redact_sensitive({
            "username": "test",
            "password": "secret123",
            "api_key": "sk-test",
        })

        assert redacted["username"] == "test"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
