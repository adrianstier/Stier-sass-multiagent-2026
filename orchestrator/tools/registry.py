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
    ],
    "business_analyst": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "search_documents",
    ],
    "project_manager": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "create_timeline",
        "estimate_effort",
    ],
    "ux_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "create_wireframe",
        "validate_accessibility",
    ],
    "tech_lead": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "analyze_architecture",
        "evaluate_tech_stack",
    ],
    "database_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "validate_schema",
        "generate_migration",
    ],
    "backend_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "write_code",
        "run_tests",
        "validate_api",
    ],
    "frontend_engineer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "write_code",
        "run_tests",
        "check_accessibility",
    ],
    "code_reviewer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "analyze_code",
        "check_standards",
        "approve_gate",
        "reject_gate",
    ],
    "security_reviewer": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "scan_vulnerabilities",
        "check_compliance",
        "approve_gate",
        "reject_gate",
    ],
    "cleanup_agent": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "scan_repository",
        "list_files",
        "delete_file",
        "find_duplicates",
        "find_unused_imports",
        "find_dead_code",
        "find_debug_statements",
        "find_todo_comments",
        "generate_cleanup_report",
    ],
    "data_scientist": [
        "create_artifact",
        "read_artifact",
        "list_artifacts",
        "analyze_data",
        "create_pipeline",
        "train_model",
        "evaluate_model",
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
