"""Filesystem tools for repository access with sandboxing."""

import os
import shutil
import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import mimetypes


@dataclass
class SandboxConfig:
    """Configuration for filesystem sandboxing."""
    allowed_paths: List[str]  # List of allowed root paths
    denied_patterns: List[str] = None  # Glob patterns to deny
    max_file_size_mb: float = 10.0  # Max file size to read/write
    allow_delete: bool = False  # Whether delete operations are allowed
    allow_write_outside_workspace: bool = False

    def __post_init__(self):
        if self.denied_patterns is None:
            self.denied_patterns = [
                "**/.git/objects/*",
                "**/.env",
                "**/.env.*",
                "**/secrets.*",
                "**/credentials.*",
                "**/*_secret*",
                "**/node_modules/*",
                "**/__pycache__/*",
                "**/.venv/*",
                "**/venv/*",
            ]


class SandboxViolationError(Exception):
    """Raised when an operation violates sandbox rules."""
    pass


class FilesystemTools:
    """
    Sandboxed filesystem operations for agent use.

    All operations are restricted to allowed paths and respect
    denied patterns to prevent access to sensitive files.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._operation_log: List[Dict[str, Any]] = []

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        abs_path = os.path.abspath(path)

        # Check if path is under any allowed path
        for allowed in self.config.allowed_paths:
            allowed_abs = os.path.abspath(allowed)
            if abs_path.startswith(allowed_abs):
                return True
        return False

    def _is_path_denied(self, path: str) -> bool:
        """Check if a path matches any denied pattern."""
        for pattern in self.config.denied_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _validate_path(self, path: str, operation: str) -> str:
        """Validate and normalize a path for an operation."""
        abs_path = os.path.abspath(path)

        if not self._is_path_allowed(abs_path):
            raise SandboxViolationError(
                f"Path '{path}' is outside allowed directories. "
                f"Allowed: {self.config.allowed_paths}"
            )

        if self._is_path_denied(abs_path):
            raise SandboxViolationError(
                f"Path '{path}' matches a denied pattern and cannot be accessed."
            )

        self._log_operation(operation, abs_path)
        return abs_path

    def _log_operation(self, operation: str, path: str, details: Dict = None):
        """Log a filesystem operation for audit."""
        self._operation_log.append({
            "operation": operation,
            "path": path,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        })

    def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        max_lines: Optional[int] = None,
        start_line: int = 0,
    ) -> Dict[str, Any]:
        """
        Read file contents with optional line limits.

        Args:
            path: Path to the file
            encoding: File encoding (default: utf-8)
            max_lines: Maximum number of lines to read
            start_line: Line number to start from (0-indexed)

        Returns:
            Dict with content, metadata, and line info
        """
        abs_path = self._validate_path(path, "read_file")

        if not os.path.exists(abs_path):
            return {"error": f"File not found: {path}", "success": False}

        if not os.path.isfile(abs_path):
            return {"error": f"Path is not a file: {path}", "success": False}

        # Check file size
        file_size = os.path.getsize(abs_path)
        max_size = self.config.max_file_size_mb * 1024 * 1024

        if file_size > max_size:
            return {
                "error": f"File too large ({file_size / 1024 / 1024:.2f}MB). "
                        f"Max allowed: {self.config.max_file_size_mb}MB",
                "success": False,
                "file_size_bytes": file_size,
            }

        try:
            with open(abs_path, "r", encoding=encoding) as f:
                if max_lines is not None or start_line > 0:
                    lines = f.readlines()
                    total_lines = len(lines)

                    if start_line >= total_lines:
                        return {
                            "error": f"Start line {start_line} exceeds file length ({total_lines} lines)",
                            "success": False,
                        }

                    end_line = start_line + max_lines if max_lines else total_lines
                    selected_lines = lines[start_line:end_line]
                    content = "".join(selected_lines)

                    return {
                        "success": True,
                        "content": content,
                        "path": path,
                        "total_lines": total_lines,
                        "start_line": start_line,
                        "end_line": min(end_line, total_lines),
                        "lines_returned": len(selected_lines),
                        "truncated": end_line < total_lines,
                    }
                else:
                    content = f.read()
                    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

                    return {
                        "success": True,
                        "content": content,
                        "path": path,
                        "total_lines": line_count,
                        "file_size_bytes": file_size,
                    }

        except UnicodeDecodeError:
            return {
                "error": f"Cannot decode file with encoding '{encoding}'. "
                        "Try a different encoding or the file may be binary.",
                "success": False,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        backup: bool = True,
    ) -> Dict[str, Any]:
        """
        Write content to a file.

        Args:
            path: Path to write to
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if needed
            backup: Create backup of existing file

        Returns:
            Dict with operation result
        """
        abs_path = self._validate_path(path, "write_file")

        # Check content size
        content_size = len(content.encode(encoding))
        max_size = self.config.max_file_size_mb * 1024 * 1024

        if content_size > max_size:
            return {
                "error": f"Content too large ({content_size / 1024 / 1024:.2f}MB). "
                        f"Max allowed: {self.config.max_file_size_mb}MB",
                "success": False,
            }

        try:
            # Create parent directories
            parent_dir = os.path.dirname(abs_path)
            if create_dirs and parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # Backup existing file
            backup_path = None
            if backup and os.path.exists(abs_path):
                backup_path = f"{abs_path}.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(abs_path, backup_path)

            # Write file
            existed = os.path.exists(abs_path)
            with open(abs_path, "w", encoding=encoding) as f:
                f.write(content)

            return {
                "success": True,
                "path": path,
                "bytes_written": content_size,
                "created": not existed,
                "backup_path": backup_path,
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def append_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """Append content to a file."""
        abs_path = self._validate_path(path, "append_file")

        if not os.path.exists(abs_path):
            return {"error": f"File not found: {path}", "success": False}

        try:
            with open(abs_path, "a", encoding=encoding) as f:
                f.write(content)

            return {
                "success": True,
                "path": path,
                "bytes_appended": len(content.encode(encoding)),
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file (if allowed by sandbox config)."""
        if not self.config.allow_delete:
            return {
                "error": "Delete operations are not allowed in this sandbox",
                "success": False,
            }

        abs_path = self._validate_path(path, "delete_file")

        if not os.path.exists(abs_path):
            return {"error": f"File not found: {path}", "success": False}

        if not os.path.isfile(abs_path):
            return {"error": f"Path is not a file: {path}", "success": False}

        try:
            os.remove(abs_path)
            return {"success": True, "path": path, "deleted": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def list_directory(
        self,
        path: str,
        recursive: bool = False,
        include_hidden: bool = False,
        pattern: Optional[str] = None,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            path: Directory path
            recursive: Whether to list recursively
            include_hidden: Include hidden files (starting with .)
            pattern: Glob pattern to filter files
            max_depth: Maximum recursion depth

        Returns:
            Dict with files and directories lists
        """
        abs_path = self._validate_path(path, "list_directory")

        if not os.path.exists(abs_path):
            return {"error": f"Directory not found: {path}", "success": False}

        if not os.path.isdir(abs_path):
            return {"error": f"Path is not a directory: {path}", "success": False}

        try:
            files = []
            directories = []

            def should_include(name: str, full_path: str) -> bool:
                if not include_hidden and name.startswith("."):
                    return False
                if self._is_path_denied(full_path):
                    return False
                if pattern and not fnmatch.fnmatch(name, pattern):
                    return False
                return True

            def scan_dir(dir_path: str, current_depth: int = 0):
                if current_depth > max_depth:
                    return

                try:
                    for entry in os.scandir(dir_path):
                        rel_path = os.path.relpath(entry.path, abs_path)

                        if not should_include(entry.name, entry.path):
                            continue

                        if entry.is_file():
                            stat = entry.stat()
                            files.append({
                                "name": entry.name,
                                "path": rel_path,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            })
                        elif entry.is_dir():
                            directories.append({
                                "name": entry.name,
                                "path": rel_path,
                            })
                            if recursive:
                                scan_dir(entry.path, current_depth + 1)
                except PermissionError:
                    pass  # Skip directories we can't access

            scan_dir(abs_path)

            return {
                "success": True,
                "path": path,
                "files": sorted(files, key=lambda x: x["path"]),
                "directories": sorted(directories, key=lambda x: x["path"]),
                "total_files": len(files),
                "total_directories": len(directories),
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def create_directory(
        self,
        path: str,
        parents: bool = True,
    ) -> Dict[str, Any]:
        """Create a directory."""
        abs_path = self._validate_path(path, "create_directory")

        if os.path.exists(abs_path):
            if os.path.isdir(abs_path):
                return {"success": True, "path": path, "already_exists": True}
            else:
                return {"error": f"Path exists but is not a directory: {path}", "success": False}

        try:
            if parents:
                os.makedirs(abs_path, exist_ok=True)
            else:
                os.mkdir(abs_path)

            return {"success": True, "path": path, "created": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def delete_directory(
        self,
        path: str,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """Delete a directory (if allowed by sandbox config)."""
        if not self.config.allow_delete:
            return {
                "error": "Delete operations are not allowed in this sandbox",
                "success": False,
            }

        abs_path = self._validate_path(path, "delete_directory")

        if not os.path.exists(abs_path):
            return {"error": f"Directory not found: {path}", "success": False}

        if not os.path.isdir(abs_path):
            return {"error": f"Path is not a directory: {path}", "success": False}

        try:
            if recursive:
                shutil.rmtree(abs_path)
            else:
                os.rmdir(abs_path)

            return {"success": True, "path": path, "deleted": True}
        except OSError as e:
            if "not empty" in str(e).lower():
                return {
                    "error": "Directory not empty. Use recursive=True to delete non-empty directories.",
                    "success": False,
                }
            return {"error": str(e), "success": False}

    def copy_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Copy a file."""
        src_path = self._validate_path(source, "copy_file_source")
        dst_path = self._validate_path(destination, "copy_file_destination")

        if not os.path.exists(src_path):
            return {"error": f"Source file not found: {source}", "success": False}

        if not os.path.isfile(src_path):
            return {"error": f"Source is not a file: {source}", "success": False}

        if os.path.exists(dst_path) and not overwrite:
            return {
                "error": f"Destination already exists: {destination}. Use overwrite=True to replace.",
                "success": False,
            }

        try:
            # Create parent directory if needed
            parent_dir = os.path.dirname(dst_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            shutil.copy2(src_path, dst_path)

            return {
                "success": True,
                "source": source,
                "destination": destination,
                "bytes_copied": os.path.getsize(dst_path),
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def move_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Move/rename a file."""
        src_path = self._validate_path(source, "move_file_source")
        dst_path = self._validate_path(destination, "move_file_destination")

        if not os.path.exists(src_path):
            return {"error": f"Source not found: {source}", "success": False}

        if os.path.exists(dst_path) and not overwrite:
            return {
                "error": f"Destination already exists: {destination}. Use overwrite=True to replace.",
                "success": False,
            }

        try:
            # Create parent directory if needed
            parent_dir = os.path.dirname(dst_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            shutil.move(src_path, dst_path)

            return {
                "success": True,
                "source": source,
                "destination": destination,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed file information."""
        abs_path = self._validate_path(path, "get_file_info")

        if not os.path.exists(abs_path):
            return {"error": f"Path not found: {path}", "success": False}

        try:
            stat = os.stat(abs_path)
            is_file = os.path.isfile(abs_path)

            info = {
                "success": True,
                "path": path,
                "absolute_path": abs_path,
                "exists": True,
                "is_file": is_file,
                "is_directory": os.path.isdir(abs_path),
                "is_symlink": os.path.islink(abs_path),
                "size_bytes": stat.st_size if is_file else None,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
            }

            if is_file:
                info["mime_type"] = mimetypes.guess_type(abs_path)[0]
                info["extension"] = Path(abs_path).suffix

                # Calculate hash for small files
                if stat.st_size < 1024 * 1024:  # < 1MB
                    with open(abs_path, "rb") as f:
                        info["md5"] = hashlib.md5(f.read()).hexdigest()

            return info

        except Exception as e:
            return {"error": str(e), "success": False}

    def search_files(
        self,
        path: str,
        pattern: str,
        content_pattern: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 100,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for files by name pattern and optionally content.

        Args:
            path: Directory to search in
            pattern: Glob pattern for file names
            content_pattern: Regex pattern to search in file contents
            file_types: List of extensions to include (e.g., [".py", ".js"])
            max_results: Maximum number of results
            include_hidden: Include hidden files

        Returns:
            Dict with matching files
        """
        import re

        abs_path = self._validate_path(path, "search_files")

        if not os.path.isdir(abs_path):
            return {"error": f"Path is not a directory: {path}", "success": False}

        try:
            content_regex = re.compile(content_pattern) if content_pattern else None
            results = []
            files_searched = 0

            for root, dirs, files in os.walk(abs_path):
                # Filter hidden directories
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                for filename in files:
                    if len(results) >= max_results:
                        break

                    if not include_hidden and filename.startswith("."):
                        continue

                    # Check name pattern
                    if not fnmatch.fnmatch(filename, pattern):
                        continue

                    # Check file type
                    if file_types:
                        ext = Path(filename).suffix.lower()
                        if ext not in file_types:
                            continue

                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, abs_path)

                    # Check if path is denied
                    if self._is_path_denied(full_path):
                        continue

                    # Content search if pattern provided
                    matches = []
                    if content_regex:
                        files_searched += 1
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                for line_num, line in enumerate(f, 1):
                                    if content_regex.search(line):
                                        matches.append({
                                            "line": line_num,
                                            "content": line.strip()[:200],
                                        })
                                        if len(matches) >= 10:  # Limit matches per file
                                            break
                        except Exception:
                            continue

                        if not matches:
                            continue

                    result = {
                        "path": rel_path,
                        "name": filename,
                        "size": os.path.getsize(full_path),
                    }

                    if matches:
                        result["matches"] = matches

                    results.append(result)

            return {
                "success": True,
                "search_path": path,
                "pattern": pattern,
                "content_pattern": content_pattern,
                "results": results,
                "total_matches": len(results),
                "files_searched": files_searched if content_pattern else None,
                "truncated": len(results) >= max_results,
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the log of all filesystem operations."""
        return self._operation_log.copy()


# =============================================================================
# Factory and Singleton
# =============================================================================

_filesystem_tools: Optional[FilesystemTools] = None


def get_filesystem_tools(workspace_path: Optional[str] = None) -> FilesystemTools:
    """Get or create filesystem tools instance."""
    global _filesystem_tools

    if _filesystem_tools is None or workspace_path:
        allowed_paths = [workspace_path] if workspace_path else [os.getcwd()]
        config = SandboxConfig(
            allowed_paths=allowed_paths,
            allow_delete=True,  # Enable for agent use
        )
        _filesystem_tools = FilesystemTools(config)

    return _filesystem_tools


def configure_filesystem_sandbox(config: SandboxConfig) -> FilesystemTools:
    """Configure filesystem tools with custom sandbox settings."""
    global _filesystem_tools
    _filesystem_tools = FilesystemTools(config)
    return _filesystem_tools
