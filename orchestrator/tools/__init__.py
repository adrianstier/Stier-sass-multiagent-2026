"""Tool abstractions for agent actions.

No agent performs privileged actions directly. All actions go through
the tool executor with role-based allowlists.
"""

from .executor import ToolExecutor, ToolResult
from .registry import get_tools_for_role, register_tool, TOOL_REGISTRY

# New comprehensive tool modules
from .filesystem import (
    FilesystemTools,
    SandboxConfig,
    SandboxViolationError,
    get_filesystem_tools,
    configure_filesystem_sandbox,
)
from .git_tools import (
    GitTools,
    GitConfig,
    GitOperationType,
    GitOperationError,
    get_git_tools,
    configure_git_tools,
)
from .execution import (
    ExecutionTools,
    ExecutionConfig,
    ExecutionMode,
    ExecutionError,
    CommandDeniedError,
    get_execution_tools,
    configure_execution_tools,
)
from .code_analysis import (
    CodeAnalyzer,
    CodeAnalysisConfig,
    Symbol,
    SymbolKind,
    Reference,
    get_code_analyzer,
    configure_code_analyzer,
)

__all__ = [
    # Core
    "ToolExecutor",
    "ToolResult",
    "get_tools_for_role",
    "register_tool",
    "TOOL_REGISTRY",
    # Filesystem
    "FilesystemTools",
    "SandboxConfig",
    "SandboxViolationError",
    "get_filesystem_tools",
    "configure_filesystem_sandbox",
    # Git
    "GitTools",
    "GitConfig",
    "GitOperationType",
    "GitOperationError",
    "get_git_tools",
    "configure_git_tools",
    # Execution
    "ExecutionTools",
    "ExecutionConfig",
    "ExecutionMode",
    "ExecutionError",
    "CommandDeniedError",
    "get_execution_tools",
    "configure_execution_tools",
    # Code Analysis
    "CodeAnalyzer",
    "CodeAnalysisConfig",
    "Symbol",
    "SymbolKind",
    "Reference",
    "get_code_analyzer",
    "configure_code_analyzer",
]
