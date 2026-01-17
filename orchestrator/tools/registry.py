"""Tool registry with role-based allowlists."""

from typing import Dict, List, Any, Callable, Optional


# Global tool registry
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Role-based tool allowlists
ROLE_ALLOWLISTS: Dict[str, List[str]] = {
    "orchestrator": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "update_run_status",
        "send_notification",
        # Filesystem (read-only for orchestrator)
        "read_file",
        "list_directory",
        "search_files",
        "get_file_info",
        "get_project_structure",
        # Git (read-only)
        "git_status",
        "git_log",
        "git_diff",
    ],
    "business_analyst": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "search_documents",
        # Filesystem (read-only)
        "read_file",
        "list_directory",
        "search_files",
        "get_project_structure",
        # Code analysis
        "search_code",
        "get_symbol_outline",
    ],
    "project_manager": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "create_timeline",
        "estimate_effort",
        # Filesystem (read-only)
        "read_file",
        "list_directory",
        "get_project_structure",
        # Git (read-only)
        "git_status",
        "git_log",
    ],
    "ux_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "create_wireframe",
        "validate_accessibility",
        # Filesystem
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "create_directory",
        # Code analysis
        "search_code",
        "get_symbol_outline",
    ],
    "tech_lead": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "analyze_architecture",
        "evaluate_tech_stack",
        # Full filesystem access
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "get_file_info",
        "create_directory",
        # Full git access
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "git_blame",
        "git_branch",
        # Full code analysis
        "extract_symbols",
        "find_references",
        "find_definition",
        "search_code",
        "get_symbol_outline",
        "get_file_dependencies",
        "get_project_structure",
        # Execution (read-only validation)
        "run_linter",
        "run_type_check",
    ],
    "database_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "validate_schema",
        "generate_migration",
        # Filesystem
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "create_directory",
        # Git
        "git_status",
        "git_diff",
        "git_add",
        "git_commit",
        # Code analysis
        "extract_symbols",
        "find_references",
        "search_code",
        "get_file_dependencies",
        # Execution
        "run_command",
        "run_tests",
    ],
    "backend_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        # Full filesystem access
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "get_file_info",
        "copy_file",
        "move_file",
        "create_directory",
        # Full git access
        "git_status",
        "git_diff",
        "git_log",
        "git_add",
        "git_commit",
        "git_branch",
        "git_checkout",
        "git_stash",
        "git_stash_pop",
        "git_show",
        "git_blame",
        # Full code analysis
        "extract_symbols",
        "find_references",
        "find_definition",
        "search_code",
        "get_symbol_outline",
        "get_file_dependencies",
        "get_project_structure",
        # Full execution
        "run_command",
        "run_tests",
        "run_linter",
        "run_build",
        "run_type_check",
        "install_dependencies",
    ],
    "frontend_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        # Full filesystem access
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "get_file_info",
        "copy_file",
        "move_file",
        "create_directory",
        # Full git access
        "git_status",
        "git_diff",
        "git_log",
        "git_add",
        "git_commit",
        "git_branch",
        "git_checkout",
        "git_stash",
        "git_stash_pop",
        "git_show",
        "git_blame",
        # Full code analysis
        "extract_symbols",
        "find_references",
        "find_definition",
        "search_code",
        "get_symbol_outline",
        "get_file_dependencies",
        "get_project_structure",
        # Full execution
        "run_command",
        "run_tests",
        "run_linter",
        "run_build",
        "run_type_check",
        "install_dependencies",
    ],
    "code_reviewer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "approve_gate",
        "reject_gate",
        # Filesystem (read-only)
        "read_file",
        "list_directory",
        "search_files",
        "get_file_info",
        # Git (read-only)
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "git_blame",
        # Full code analysis
        "extract_symbols",
        "find_references",
        "find_definition",
        "search_code",
        "get_symbol_outline",
        "get_file_dependencies",
        "get_project_structure",
        # Execution (read-only validation)
        "run_tests",
        "run_linter",
        "run_type_check",
    ],
    "security_reviewer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "scan_vulnerabilities",
        "check_compliance",
        "approve_gate",
        "reject_gate",
        # Filesystem (read-only)
        "read_file",
        "list_directory",
        "search_files",
        "get_file_info",
        # Git (read-only)
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "git_blame",
        # Full code analysis
        "extract_symbols",
        "find_references",
        "find_definition",
        "search_code",
        "get_symbol_outline",
        "get_file_dependencies",
        "get_project_structure",
        # Execution (security scanning)
        "run_command",
        "run_tests",
    ],
    "cleanup_agent": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "scan_repository",
        "find_unused_imports",
        "find_debug_statements",
        "find_todo_comments",
        "generate_cleanup_report",
        # Filesystem (full access for cleanup)
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "get_file_info",
        "copy_file",
        "move_file",
        "create_directory",
        # Git
        "git_status",
        "git_diff",
        "git_add",
        "git_commit",
        # Code analysis
        "extract_symbols",
        "find_references",
        "search_code",
        "get_file_dependencies",
        "get_project_structure",
        # Execution
        "run_linter",
    ],
    "data_scientist": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "analyze_data",
        "create_pipeline",
        "train_model",
        "evaluate_model",
        # Filesystem
        "read_file",
        "write_file",
        "list_directory",
        "search_files",
        "create_directory",
        # Git
        "git_status",
        "git_diff",
        "git_add",
        "git_commit",
        # Code analysis
        "extract_symbols",
        "search_code",
        "get_symbol_outline",
        # Execution
        "run_command",
        "run_tests",
        "install_dependencies",
    ],
    "design_reviewer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "check_design_system",
        "validate_accessibility",
        "check_responsive",
        "approve_gate",
        "reject_gate",
        # Filesystem (read-only)
        "read_file",
        "list_directory",
        "search_files",
        # Code analysis
        "search_code",
        "get_symbol_outline",
        "get_project_structure",
    ],
}


def register_tool(
    name: str,
    function: Callable,
    description: str,
    parameters: Dict[str, Any],
) -> None:
    """Register a tool in the registry."""
    TOOL_REGISTRY[name] = {
        "name": name,
        "function": function,
        "description": description,
        "parameters": parameters,
    }


def get_tools_for_role(role: str) -> List[str]:
    """Get the list of tools allowed for a role."""
    return ROLE_ALLOWLISTS.get(role, [])


def get_tool_definitions_for_role(role: str) -> List[Dict[str, Any]]:
    """Get tool definitions (for LLM context) for a role."""
    allowed = get_tools_for_role(role)
    return [
        {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["parameters"],
        }
        for name, tool_info in TOOL_REGISTRY.items()
        if name in allowed
    ]


# ============================================================
# Built-in Tool Implementations
# ============================================================

def _create_artifact(
    run_id: str,
    task_id: str,
    artifact_type: str,
    name: str,
    content: str,
    content_type: str = "text/markdown",
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a new artifact."""
    from orchestrator.core.database import get_db
    from orchestrator.core.models import Artifact
    import uuid

    with get_db() as db:
        artifact = Artifact(
            id=uuid.uuid4(),
            run_id=run_id,
            task_id=task_id,
            artifact_type=artifact_type,
            name=name,
            content=content,
            content_type=content_type,
            metadata=metadata or {},
            produced_by="system",  # Will be set by caller
        )
        db.add(artifact)
        db.commit()
        return {
            "id": str(artifact.id),
            "artifact_type": artifact_type,
            "name": name,
        }


def _read_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Read an artifact by ID."""
    from orchestrator.core.database import get_db
    from orchestrator.core.models import Artifact
    import uuid

    with get_db() as db:
        artifact = db.query(Artifact).filter(
            Artifact.id == uuid.UUID(artifact_id)
        ).first()

        if not artifact:
            return None

        return {
            "id": str(artifact.id),
            "artifact_type": artifact.artifact_type,
            "name": artifact.name,
            "content": artifact.content,
            "content_type": artifact.content_type,
            "metadata": artifact.artifact_metadata,
            "created_at": artifact.created_at.isoformat(),
        }


def _list_artifacts(
    run_id: str,
    artifact_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List artifacts for a run."""
    from orchestrator.core.database import get_db
    from orchestrator.core.models import Artifact
    import uuid

    with get_db() as db:
        query = db.query(Artifact).filter(Artifact.run_id == uuid.UUID(run_id))

        if artifact_type:
            query = query.filter(Artifact.artifact_type == artifact_type)

        artifacts = query.order_by(Artifact.created_at.desc()).all()

        return [
            {
                "id": str(a.id),
                "artifact_type": a.artifact_type,
                "name": a.name,
                "content_type": a.content_type,
                "created_at": a.created_at.isoformat(),
            }
            for a in artifacts
        ]


def _approve_gate(run_id: str, gate_type: str, notes: str = "") -> Dict[str, Any]:
    """Approve a quality gate."""
    from orchestrator.core.database import get_db
    from orchestrator.core.models import Run, GateStatus
    import uuid

    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            return {"success": False, "error": "Run not found"}

        if gate_type == "code_review":
            run.code_review_status = GateStatus.PASSED
        elif gate_type == "security_review":
            run.security_review_status = GateStatus.PASSED
        else:
            return {"success": False, "error": f"Unknown gate type: {gate_type}"}

        db.commit()
        return {"success": True, "gate": gate_type, "status": "passed", "notes": notes}


def _reject_gate(run_id: str, gate_type: str, reason: str) -> Dict[str, Any]:
    """Reject a quality gate."""
    from orchestrator.core.database import get_db
    from orchestrator.core.models import Run, GateStatus
    import uuid

    with get_db() as db:
        run = db.query(Run).filter(Run.id == uuid.UUID(run_id)).first()
        if not run:
            return {"success": False, "error": "Run not found"}

        if gate_type == "code_review":
            run.code_review_status = GateStatus.FAILED
        elif gate_type == "security_review":
            run.security_review_status = GateStatus.FAILED
        else:
            return {"success": False, "error": f"Unknown gate type: {gate_type}"}

        db.commit()
        return {"success": True, "gate": gate_type, "status": "failed", "reason": reason}


# Register built-in tools
register_tool(
    "create_artifact",
    _create_artifact,
    "Create a new artifact (document, code, spec, etc.)",
    {
        "type": "object",
        "properties": {
            "artifact_type": {"type": "string", "description": "Type of artifact"},
            "name": {"type": "string", "description": "Name of the artifact"},
            "content": {"type": "string", "description": "Content of the artifact"},
            "content_type": {"type": "string", "description": "MIME type", "default": "text/markdown"},
        },
        "required": ["artifact_type", "name", "content"],
    },
)

register_tool(
    "read_artifact",
    _read_artifact,
    "Read an artifact by ID",
    {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "UUID of the artifact"},
        },
        "required": ["artifact_id"],
    },
)

register_tool(
    "list_artifacts",
    _list_artifacts,
    "List artifacts for a run",
    {
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "UUID of the run"},
            "artifact_type": {"type": "string", "description": "Filter by artifact type"},
        },
        "required": ["run_id"],
    },
)

register_tool(
    "approve_gate",
    _approve_gate,
    "Approve a quality gate (code_review or security_review)",
    {
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "UUID of the run"},
            "gate_type": {"type": "string", "enum": ["code_review", "security_review"]},
            "notes": {"type": "string", "description": "Approval notes"},
        },
        "required": ["run_id", "gate_type"],
    },
)

register_tool(
    "reject_gate",
    _reject_gate,
    "Reject a quality gate with reason",
    {
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "description": "UUID of the run"},
            "gate_type": {"type": "string", "enum": ["code_review", "security_review"]},
            "reason": {"type": "string", "description": "Reason for rejection"},
        },
        "required": ["run_id", "gate_type", "reason"],
    },
)


# ============================================================
# Cleanup Agent Tools
# ============================================================

def _scan_repository(path: str, patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Scan repository for cleanup issues."""
    import os
    from pathlib import Path

    scan_path = Path(path)
    if not scan_path.exists():
        return {"error": f"Path does not exist: {path}"}

    # Default patterns for AI artifacts
    default_patterns = [
        "*.bak", "*.old", "*_backup.*", "*_copy.*", "*~",
        ".DS_Store", "Thumbs.db", "*.pyc", "__pycache__",
        "*.orig", "*.swp", "*.swo",
    ]
    patterns = patterns or default_patterns

    findings = {
        "backup_files": [],
        "system_files": [],
        "cache_files": [],
        "empty_directories": [],
        "total_files_scanned": 0,
    }

    for root, dirs, files in os.walk(scan_path):
        # Skip hidden directories and common ignore paths
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '.git', '__pycache__']]

        for file in files:
            findings["total_files_scanned"] += 1
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, scan_path)

            # Check for backup patterns
            if any(file.endswith(ext) for ext in ['.bak', '.old', '.orig', '~']):
                findings["backup_files"].append(rel_path)
            elif '_backup' in file or '_copy' in file or '_old' in file:
                findings["backup_files"].append(rel_path)

            # System files
            if file in ['.DS_Store', 'Thumbs.db']:
                findings["system_files"].append(rel_path)

            # Cache files
            if file.endswith('.pyc') or file == '__pycache__':
                findings["cache_files"].append(rel_path)

        # Check for empty directories
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                findings["empty_directories"].append(os.path.relpath(dir_path, scan_path))

    return findings


def _find_unused_imports(file_path: str) -> Dict[str, Any]:
    """Find unused imports in a Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Simple heuristic - this would be better with AST parsing
        import re
        imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+(\S+)', content, re.MULTILINE)

        unused = []
        for imp in imports:
            # Check if import is used (simple check)
            imp_name = imp.split('.')[0].split(' as ')[-1]
            # Count occurrences (excluding the import line itself)
            occurrences = len(re.findall(rf'\b{imp_name}\b', content))
            if occurrences <= 1:  # Only the import itself
                unused.append(imp)

        return {"file": file_path, "unused_imports": unused}
    except Exception as e:
        return {"file": file_path, "error": str(e)}


def _find_debug_statements(file_path: str) -> Dict[str, Any]:
    """Find debug statements in a file."""
    debug_patterns = [
        (r'console\.log\(', 'console.log'),
        (r'console\.debug\(', 'console.debug'),
        (r'print\(.*DEBUG', 'debug print'),
        (r'print\([\'"]test', 'test print'),
        (r'debugger;?', 'debugger statement'),
        (r'# DEBUG', 'debug comment'),
        (r'// DEBUG', 'debug comment'),
    ]

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        import re
        findings = []
        for i, line in enumerate(lines, 1):
            for pattern, desc in debug_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "line": i,
                        "type": desc,
                        "content": line.strip()[:100],
                    })

        return {"file": file_path, "debug_statements": findings}
    except Exception as e:
        return {"file": file_path, "error": str(e)}


def _find_todo_comments(file_path: str) -> Dict[str, Any]:
    """Find TODO/FIXME comments in a file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        import re
        todos = []
        for i, line in enumerate(lines, 1):
            if re.search(r'\b(TODO|FIXME|XXX|HACK|BUG)\b', line, re.IGNORECASE):
                todos.append({
                    "line": i,
                    "content": line.strip()[:100],
                })

        return {"file": file_path, "todos": todos}
    except Exception as e:
        return {"file": file_path, "error": str(e)}


def _generate_cleanup_report(findings: Dict[str, Any]) -> str:
    """Generate a markdown cleanup report."""
    report = ["# Repository Cleanup Report\n"]

    report.append("## Summary\n")
    report.append(f"- Files scanned: {findings.get('total_files_scanned', 'N/A')}\n")
    report.append(f"- Backup files found: {len(findings.get('backup_files', []))}\n")
    report.append(f"- System files found: {len(findings.get('system_files', []))}\n")
    report.append(f"- Empty directories: {len(findings.get('empty_directories', []))}\n")

    if findings.get('backup_files'):
        report.append("\n## Backup/Duplicate Files\n")
        for f in findings['backup_files']:
            report.append(f"- `{f}`\n")

    if findings.get('system_files'):
        report.append("\n## System Files (Safe to Remove)\n")
        for f in findings['system_files']:
            report.append(f"- `{f}`\n")

    if findings.get('empty_directories'):
        report.append("\n## Empty Directories\n")
        for d in findings['empty_directories']:
            report.append(f"- `{d}/`\n")

    report.append("\n## Recommendations\n")
    report.append("- Add backup patterns to `.gitignore`\n")
    report.append("- Configure pre-commit hooks to prevent committing debug code\n")
    report.append("- Run cleanup before code review\n")

    return "".join(report)


# Register cleanup tools
register_tool(
    "scan_repository",
    _scan_repository,
    "Scan repository for cleanup issues (backup files, system files, etc.)",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to scan"},
            "patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to look for"},
        },
        "required": ["path"],
    },
)

register_tool(
    "find_unused_imports",
    _find_unused_imports,
    "Find unused imports in a Python file",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["file_path"],
    },
)

register_tool(
    "find_debug_statements",
    _find_debug_statements,
    "Find debug statements (console.log, print DEBUG, etc.) in a file",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["file_path"],
    },
)

register_tool(
    "find_todo_comments",
    _find_todo_comments,
    "Find TODO/FIXME comments in a file",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["file_path"],
    },
)

register_tool(
    "generate_cleanup_report",
    _generate_cleanup_report,
    "Generate a markdown cleanup report from scan findings",
    {
        "type": "object",
        "properties": {
            "findings": {"type": "object", "description": "Findings from scan_repository"},
        },
        "required": ["findings"],
    },
)


# ============================================================
# Filesystem Tools Registration
# ============================================================

from .filesystem import get_filesystem_tools

def _read_file(path: str, max_lines: Optional[int] = None, start_line: int = 0) -> Dict[str, Any]:
    """Read file contents."""
    fs = get_filesystem_tools()
    return fs.read_file(path, max_lines=max_lines, start_line=start_line)

def _write_file(path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
    """Write content to a file."""
    fs = get_filesystem_tools()
    return fs.write_file(path, content, create_dirs=create_dirs)

def _list_directory(path: str, recursive: bool = False, pattern: Optional[str] = None) -> Dict[str, Any]:
    """List directory contents."""
    fs = get_filesystem_tools()
    return fs.list_directory(path, recursive=recursive, pattern=pattern)

def _search_files(path: str, pattern: str, content_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Search for files by name and optionally content."""
    fs = get_filesystem_tools()
    return fs.search_files(path, pattern, content_pattern=content_pattern)

def _get_file_info(path: str) -> Dict[str, Any]:
    """Get file information."""
    fs = get_filesystem_tools()
    return fs.get_file_info(path)

def _copy_file(source: str, destination: str) -> Dict[str, Any]:
    """Copy a file."""
    fs = get_filesystem_tools()
    return fs.copy_file(source, destination)

def _move_file(source: str, destination: str) -> Dict[str, Any]:
    """Move/rename a file."""
    fs = get_filesystem_tools()
    return fs.move_file(source, destination)

def _create_directory(path: str) -> Dict[str, Any]:
    """Create a directory."""
    fs = get_filesystem_tools()
    return fs.create_directory(path)

register_tool(
    "read_file",
    _read_file,
    "Read the contents of a file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "max_lines": {"type": "integer", "description": "Maximum lines to read"},
            "start_line": {"type": "integer", "description": "Line to start from (0-indexed)"},
        },
        "required": ["path"],
    },
)

register_tool(
    "write_file",
    _write_file,
    "Write content to a file (creates if doesn't exist)",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to"},
            "content": {"type": "string", "description": "Content to write"},
            "create_dirs": {"type": "boolean", "description": "Create parent directories"},
        },
        "required": ["path", "content"],
    },
)

register_tool(
    "list_directory",
    _list_directory,
    "List files and directories in a path",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path"},
            "recursive": {"type": "boolean", "description": "List recursively"},
            "pattern": {"type": "string", "description": "Filter by glob pattern"},
        },
        "required": ["path"],
    },
)

register_tool(
    "search_files",
    _search_files,
    "Search for files by name pattern and optionally content",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to search in"},
            "pattern": {"type": "string", "description": "Filename glob pattern"},
            "content_pattern": {"type": "string", "description": "Regex to search in file contents"},
        },
        "required": ["path", "pattern"],
    },
)

register_tool(
    "get_file_info",
    _get_file_info,
    "Get detailed information about a file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["path"],
    },
)

register_tool(
    "copy_file",
    _copy_file,
    "Copy a file to a new location",
    {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Source file path"},
            "destination": {"type": "string", "description": "Destination path"},
        },
        "required": ["source", "destination"],
    },
)

register_tool(
    "move_file",
    _move_file,
    "Move or rename a file",
    {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Source file path"},
            "destination": {"type": "string", "description": "Destination path"},
        },
        "required": ["source", "destination"],
    },
)

register_tool(
    "create_directory",
    _create_directory,
    "Create a directory (and parent directories if needed)",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to create"},
        },
        "required": ["path"],
    },
)


# ============================================================
# Git Tools Registration
# ============================================================

from .git_tools import get_git_tools

def _git_status() -> Dict[str, Any]:
    """Get git repository status."""
    git = get_git_tools()
    return git.status()

def _git_diff(path: Optional[str] = None, staged: bool = False, stat_only: bool = False) -> Dict[str, Any]:
    """Get diff of changes."""
    git = get_git_tools()
    return git.diff(path=path, staged=staged, stat_only=stat_only)

def _git_log(count: int = 10, path: Optional[str] = None, oneline: bool = False) -> Dict[str, Any]:
    """Get commit history."""
    git = get_git_tools()
    return git.log(count=count, path=path, oneline=oneline)

def _git_add(paths: List[str], all: bool = False) -> Dict[str, Any]:
    """Stage files for commit."""
    git = get_git_tools()
    return git.add(paths, all=all)

def _git_commit(message: str, files: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a commit."""
    git = get_git_tools()
    return git.commit(message, files=files)

def _git_branch(name: Optional[str] = None, delete: bool = False, list_all: bool = False) -> Dict[str, Any]:
    """Manage git branches."""
    git = get_git_tools()
    return git.branch(name=name, delete=delete, list_all=list_all)

def _git_checkout(target: str, create: bool = False) -> Dict[str, Any]:
    """Checkout a branch or commit."""
    git = get_git_tools()
    return git.checkout(target, create=create)

def _git_stash(message: Optional[str] = None) -> Dict[str, Any]:
    """Stash current changes."""
    git = get_git_tools()
    return git.stash(message=message)

def _git_stash_pop(index: int = 0) -> Dict[str, Any]:
    """Pop a stash."""
    git = get_git_tools()
    return git.stash_pop(index)

def _git_show(commit: str = "HEAD", stat_only: bool = False) -> Dict[str, Any]:
    """Show commit details."""
    git = get_git_tools()
    return git.show(commit, stat_only=stat_only)

def _git_blame(path: str, line_start: Optional[int] = None, line_end: Optional[int] = None) -> Dict[str, Any]:
    """Show line-by-line blame."""
    git = get_git_tools()
    return git.blame(path, line_start=line_start, line_end=line_end)

register_tool(
    "git_status",
    _git_status,
    "Get repository status (staged, unstaged, untracked files)",
    {"type": "object", "properties": {}},
)

register_tool(
    "git_diff",
    _git_diff,
    "Show diff of changes",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Specific file to diff"},
            "staged": {"type": "boolean", "description": "Show staged changes"},
            "stat_only": {"type": "boolean", "description": "Show only statistics"},
        },
    },
)

register_tool(
    "git_log",
    _git_log,
    "Show commit history",
    {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "Number of commits"},
            "path": {"type": "string", "description": "Filter by path"},
            "oneline": {"type": "boolean", "description": "Compact format"},
        },
    },
)

register_tool(
    "git_add",
    _git_add,
    "Stage files for commit",
    {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Files to stage"},
            "all": {"type": "boolean", "description": "Stage all changes"},
        },
        "required": ["paths"],
    },
)

register_tool(
    "git_commit",
    _git_commit,
    "Create a commit",
    {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Commit message"},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Specific files to commit"},
        },
        "required": ["message"],
    },
)

register_tool(
    "git_branch",
    _git_branch,
    "List, create, or delete branches",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Branch name"},
            "delete": {"type": "boolean", "description": "Delete the branch"},
            "list_all": {"type": "boolean", "description": "List all branches including remote"},
        },
    },
)

register_tool(
    "git_checkout",
    _git_checkout,
    "Checkout a branch or commit",
    {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Branch or commit to checkout"},
            "create": {"type": "boolean", "description": "Create new branch"},
        },
        "required": ["target"],
    },
)

register_tool(
    "git_stash",
    _git_stash,
    "Stash current changes",
    {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Stash message"},
        },
    },
)

register_tool(
    "git_stash_pop",
    _git_stash_pop,
    "Pop a stash",
    {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "Stash index (default: 0)"},
        },
    },
)

register_tool(
    "git_show",
    _git_show,
    "Show commit details and diff",
    {
        "type": "object",
        "properties": {
            "commit": {"type": "string", "description": "Commit hash or reference"},
            "stat_only": {"type": "boolean", "description": "Show only statistics"},
        },
    },
)

register_tool(
    "git_blame",
    _git_blame,
    "Show line-by-line blame for a file",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "line_start": {"type": "integer", "description": "Start line"},
            "line_end": {"type": "integer", "description": "End line"},
        },
        "required": ["path"],
    },
)


# ============================================================
# Execution Tools Registration
# ============================================================

from .execution import get_execution_tools

def _run_command(command: str, working_dir: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Run a shell command."""
    exec_tools = get_execution_tools()
    return exec_tools.run_command(command, working_dir=working_dir, timeout=timeout)

def _run_tests(test_command: Optional[str] = None, framework: Optional[str] = None, verbose: bool = False, coverage: bool = False) -> Dict[str, Any]:
    """Run tests with auto-detected or specified framework."""
    exec_tools = get_execution_tools()
    return exec_tools.run_tests(test_command=test_command, framework=framework, verbose=verbose, coverage=coverage)

def _run_linter(linter: Optional[str] = None, path: Optional[str] = None, fix: bool = False) -> Dict[str, Any]:
    """Run linter/formatter."""
    exec_tools = get_execution_tools()
    return exec_tools.run_linter(linter=linter, path=path, fix=fix)

def _run_build(build_command: Optional[str] = None, target: Optional[str] = None) -> Dict[str, Any]:
    """Run build command."""
    exec_tools = get_execution_tools()
    return exec_tools.run_build(build_command=build_command, target=target)

def _run_type_check(checker: Optional[str] = None, path: Optional[str] = None) -> Dict[str, Any]:
    """Run type checking."""
    exec_tools = get_execution_tools()
    return exec_tools.run_type_check(checker=checker, path=path)

def _install_dependencies(manager: Optional[str] = None, packages: Optional[List[str]] = None, dev: bool = False) -> Dict[str, Any]:
    """Install dependencies."""
    exec_tools = get_execution_tools()
    return exec_tools.install_dependencies(manager=manager, packages=packages, dev=dev)

register_tool(
    "run_command",
    _run_command,
    "Run a shell command (sandboxed)",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to run"},
            "working_dir": {"type": "string", "description": "Working directory"},
            "timeout": {"type": "integer", "description": "Timeout in seconds"},
        },
        "required": ["command"],
    },
)

register_tool(
    "run_tests",
    _run_tests,
    "Run tests (auto-detects framework)",
    {
        "type": "object",
        "properties": {
            "test_command": {"type": "string", "description": "Explicit test command"},
            "framework": {"type": "string", "description": "Test framework (pytest, jest, etc.)"},
            "verbose": {"type": "boolean", "description": "Verbose output"},
            "coverage": {"type": "boolean", "description": "Enable coverage"},
        },
    },
)

register_tool(
    "run_linter",
    _run_linter,
    "Run linter (auto-detects linter)",
    {
        "type": "object",
        "properties": {
            "linter": {"type": "string", "description": "Linter to use"},
            "path": {"type": "string", "description": "Path to lint"},
            "fix": {"type": "boolean", "description": "Auto-fix issues"},
        },
    },
)

register_tool(
    "run_build",
    _run_build,
    "Run build command (auto-detects build system)",
    {
        "type": "object",
        "properties": {
            "build_command": {"type": "string", "description": "Explicit build command"},
            "target": {"type": "string", "description": "Build target"},
        },
    },
)

register_tool(
    "run_type_check",
    _run_type_check,
    "Run type checking (mypy, tsc, etc.)",
    {
        "type": "object",
        "properties": {
            "checker": {"type": "string", "description": "Type checker to use"},
            "path": {"type": "string", "description": "Path to check"},
        },
    },
)

register_tool(
    "install_dependencies",
    _install_dependencies,
    "Install project dependencies",
    {
        "type": "object",
        "properties": {
            "manager": {"type": "string", "description": "Package manager (npm, pip, etc.)"},
            "packages": {"type": "array", "items": {"type": "string"}, "description": "Specific packages"},
            "dev": {"type": "boolean", "description": "Install as dev dependency"},
        },
    },
)


# ============================================================
# Code Analysis Tools Registration
# ============================================================

from .code_analysis import get_code_analyzer

def _extract_symbols(file_path: str) -> Dict[str, Any]:
    """Extract symbols from a source file."""
    analyzer = get_code_analyzer()
    symbols = analyzer.extract_symbols(file_path)
    return {
        "success": True,
        "file": file_path,
        "symbols": [
            {
                "name": s.name,
                "kind": s.kind.value,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "signature": s.signature,
                "docstring": s.docstring[:200] if s.docstring else None,
                "parent": s.parent,
            }
            for s in symbols
        ],
        "count": len(symbols),
    }

def _find_references(symbol_name: str, search_path: Optional[str] = None) -> Dict[str, Any]:
    """Find all references to a symbol."""
    analyzer = get_code_analyzer()
    refs = analyzer.find_references(symbol_name, search_path=search_path)
    return {
        "success": True,
        "symbol": symbol_name,
        "references": [
            {
                "file": r.file_path,
                "line": r.line,
                "column": r.column,
                "context": r.context,
                "type": r.ref_type,
            }
            for r in refs
        ],
        "count": len(refs),
    }

def _find_definition(symbol_name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Find the definition of a symbol."""
    analyzer = get_code_analyzer()
    symbol = analyzer.find_definition(symbol_name, file_path=file_path)
    if symbol:
        return {
            "success": True,
            "found": True,
            "name": symbol.name,
            "kind": symbol.kind.value,
            "file": symbol.file_path,
            "line_start": symbol.line_start,
            "line_end": symbol.line_end,
            "signature": symbol.signature,
            "docstring": symbol.docstring,
        }
    return {"success": True, "found": False, "symbol": symbol_name}

def _search_code(pattern: str, is_regex: bool = False, file_extensions: Optional[List[str]] = None, max_results: int = 50) -> Dict[str, Any]:
    """Search for code patterns."""
    analyzer = get_code_analyzer()
    results = analyzer.search_code(pattern, is_regex=is_regex, file_extensions=file_extensions, max_results=max_results)
    return {"success": True, "pattern": pattern, "results": results, "count": len(results)}

def _get_symbol_outline(file_path: str) -> Dict[str, Any]:
    """Get a hierarchical outline of symbols in a file."""
    analyzer = get_code_analyzer()
    return analyzer.get_symbol_outline(file_path)

def _get_file_dependencies(file_path: str) -> Dict[str, Any]:
    """Analyze imports/dependencies in a file."""
    analyzer = get_code_analyzer()
    return analyzer.get_file_dependencies(file_path)

def _get_project_structure(max_depth: int = 5) -> Dict[str, Any]:
    """Get project directory structure."""
    analyzer = get_code_analyzer()
    return analyzer.get_project_structure(max_depth=max_depth)

register_tool(
    "extract_symbols",
    _extract_symbols,
    "Extract code symbols (classes, functions, etc.) from a file",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the source file"},
        },
        "required": ["file_path"],
    },
)

register_tool(
    "find_references",
    _find_references,
    "Find all references to a symbol across the codebase",
    {
        "type": "object",
        "properties": {
            "symbol_name": {"type": "string", "description": "Name of the symbol"},
            "search_path": {"type": "string", "description": "Path to search in"},
        },
        "required": ["symbol_name"],
    },
)

register_tool(
    "find_definition",
    _find_definition,
    "Find the definition of a symbol",
    {
        "type": "object",
        "properties": {
            "symbol_name": {"type": "string", "description": "Name of the symbol"},
            "file_path": {"type": "string", "description": "Start search from this file"},
        },
        "required": ["symbol_name"],
    },
)

register_tool(
    "search_code",
    _search_code,
    "Search for code patterns across the codebase",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern"},
            "is_regex": {"type": "boolean", "description": "Treat as regex"},
            "file_extensions": {"type": "array", "items": {"type": "string"}, "description": "File extensions to search"},
            "max_results": {"type": "integer", "description": "Maximum results"},
        },
        "required": ["pattern"],
    },
)

register_tool(
    "get_symbol_outline",
    _get_symbol_outline,
    "Get hierarchical outline of symbols in a file",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["file_path"],
    },
)

register_tool(
    "get_file_dependencies",
    _get_file_dependencies,
    "Analyze imports and dependencies in a file",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["file_path"],
    },
)

register_tool(
    "get_project_structure",
    _get_project_structure,
    "Get project directory structure",
    {
        "type": "object",
        "properties": {
            "max_depth": {"type": "integer", "description": "Maximum directory depth"},
        },
    },
)
