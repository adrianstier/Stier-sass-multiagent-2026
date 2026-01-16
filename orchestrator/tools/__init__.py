"""Tool abstractions for agent actions.

No agent performs privileged actions directly. All actions go through
the tool executor with role-based allowlists.
"""

from .executor import ToolExecutor, ToolResult
from .registry import get_tools_for_role, register_tool, TOOL_REGISTRY

__all__ = [
    "ToolExecutor",
    "ToolResult",
    "get_tools_for_role",
    "register_tool",
    "TOOL_REGISTRY",
]
