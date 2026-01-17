"""Tool executor with role-based access control."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import logging
import traceback

from .registry import get_tools_for_role, TOOL_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: int = 0


class ToolExecutor:
    """Execute tools with role-based access control and logging."""

    def __init__(self, role: str, run_id: str, task_id: str):
        self.role = role
        self.run_id = run_id
        self.task_id = task_id
        self.allowed_tools = get_tools_for_role(role)
        self.execution_log: list[Dict[str, Any]] = []

    def can_execute(self, tool_name: str) -> bool:
        """Check if the role can execute this tool."""
        return tool_name in self.allowed_tools

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool if allowed."""
        start_time = datetime.utcnow()

        # Check authorization
        if not self.can_execute(tool_name):
            error_msg = f"Tool '{tool_name}' not allowed for role '{self.role}'"
            logger.warning(
                f"Role '{self.role}' attempted unauthorized tool '{tool_name}'"
            )
            # Log the unauthorized attempt
            self._log_execution(tool_name, kwargs, None, error_msg, 0)
            return ToolResult(
                success=False,
                result=None,
                error=error_msg,
            )

        # Get tool function
        tool_info = TOOL_REGISTRY.get(tool_name)
        if not tool_info:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        tool_fn = tool_info["function"]

        # Execute tool
        try:
            result = tool_fn(**kwargs)
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Log execution
            self._log_execution(tool_name, kwargs, result, None, execution_time)

            return ToolResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            error_msg = f"{type(e).__name__}: {str(e)}"

            logger.error(f"Tool '{tool_name}' failed: {error_msg}")
            self._log_execution(tool_name, kwargs, None, error_msg, execution_time)

            return ToolResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time_ms=execution_time,
            )

    def _log_execution(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        result: Any,
        error: Optional[str],
        execution_time_ms: int,
    ) -> None:
        """Log tool execution for traceability."""
        self.execution_log.append({
            "tool_name": tool_name,
            "inputs": self._redact_sensitive(inputs),
            "result": self._redact_sensitive(result) if result else None,
            "error": error,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _redact_sensitive(self, data: Any) -> Any:
        """Redact sensitive information from data."""
        if isinstance(data, dict):
            return {
                k: "[REDACTED]" if self._is_sensitive_key(k) else self._redact_sensitive(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._redact_sensitive(item) for item in data]
        elif isinstance(data, str) and len(data) > 1000:
            return data[:500] + "... [TRUNCATED]"
        return data

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key name suggests sensitive data."""
        sensitive_patterns = [
            "password", "secret", "token", "key", "credential",
            "auth", "api_key", "private",
        ]
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)

    def get_execution_log(self) -> list[Dict[str, Any]]:
        """Get the execution log for this session."""
        return self.execution_log
