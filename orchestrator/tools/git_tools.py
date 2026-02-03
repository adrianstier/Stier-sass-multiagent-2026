"""Git tools for repository operations with safety controls."""

import os
import subprocess
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class GitOperationType(str, Enum):
    """Types of git operations for access control."""
    READ = "read"  # status, log, diff, show
    WRITE = "write"  # add, commit, tag
    BRANCH = "branch"  # branch, checkout, merge
    REMOTE = "remote"  # push, pull, fetch
    DANGEROUS = "dangerous"  # reset --hard, force push, rebase


@dataclass
class GitConfig:
    """Configuration for git operations."""
    repo_path: str
    allowed_operations: List[GitOperationType] = field(default_factory=lambda: [
        GitOperationType.READ,
        GitOperationType.WRITE,
        GitOperationType.BRANCH,
    ])
    protected_branches: List[str] = field(default_factory=lambda: ["main", "master", "production"])
    max_commit_files: int = 50  # Max files in a single commit
    require_commit_message: bool = True
    auto_stage: bool = False  # Whether to auto-stage all changes on commit
    sign_commits: bool = False
    commit_author: Optional[str] = None  # Override author for commits


class GitOperationError(Exception):
    """Raised when a git operation fails or is not allowed."""
    pass


class GitTools:
    """
    Git operations with safety controls for agent use.

    Provides a safe interface for common git operations while
    preventing dangerous actions on protected branches.
    """

    def __init__(self, config: GitConfig):
        self.config = config
        self._validate_repo()
        self._operation_log: List[Dict[str, Any]] = []

    def _validate_repo(self):
        """Validate that the repo path is a git repository."""
        git_dir = os.path.join(self.config.repo_path, ".git")
        if not os.path.isdir(git_dir):
            raise GitOperationError(
                f"Path is not a git repository: {self.config.repo_path}"
            )

    def _check_operation_allowed(self, op_type: GitOperationType):
        """Check if an operation type is allowed."""
        if op_type not in self.config.allowed_operations:
            raise GitOperationError(
                f"Operation type '{op_type.value}' is not allowed. "
                f"Allowed: {[o.value for o in self.config.allowed_operations]}"
            )

    def _run_git(
        self,
        args: List[str],
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git", "-C", self.config.repo_path] + args

        self._log_operation("git_command", {"args": args})

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=check,
                timeout=60,  # 1 minute timeout
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitOperationError(
                f"Git command failed: {' '.join(args)}\n"
                f"Error: {e.stderr or e.stdout or str(e)}"
            )
        except subprocess.TimeoutExpired:
            raise GitOperationError(f"Git command timed out: {' '.join(args)}")

    def _log_operation(self, operation: str, details: Dict = None):
        """Log a git operation for audit."""
        self._operation_log.append({
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        })

    def _get_current_branch(self) -> str:
        """Get the current branch name."""
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def _is_protected_branch(self, branch: str) -> bool:
        """Check if a branch is protected."""
        return branch in self.config.protected_branches

    # =========================================================================
    # Read Operations
    # =========================================================================

    def status(self, short: bool = False) -> Dict[str, Any]:
        """
        Get repository status.

        Returns:
            Dict with staged, unstaged, and untracked files
        """
        self._check_operation_allowed(GitOperationType.READ)

        args = ["status", "--porcelain"]
        if not short:
            args.append("-b")

        result = self._run_git(args)

        staged = []
        unstaged = []
        untracked = []
        branch = None

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            if line.startswith("## "):
                # Branch info
                branch_info = line[3:]
                if "..." in branch_info:
                    branch = branch_info.split("...")[0]
                else:
                    branch = branch_info.split()[0] if branch_info else None
                continue

            if len(line) < 3:
                continue

            index_status = line[0]
            worktree_status = line[1]
            filename = line[3:]

            # Handle renames (format: "old_name -> new_name")
            if " -> " in filename:
                parts = filename.split(" -> ")
                filename = parts[1] if len(parts) > 1 else parts[0]

            if index_status == "?":
                untracked.append(filename)
            else:
                if index_status != " ":
                    staged.append({
                        "file": filename,
                        "status": self._status_code_to_name(index_status),
                    })
                if worktree_status != " ":
                    unstaged.append({
                        "file": filename,
                        "status": self._status_code_to_name(worktree_status),
                    })

        return {
            "success": True,
            "branch": branch or self._get_current_branch(),
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "clean": not (staged or unstaged or untracked),
            "staged_count": len(staged),
            "unstaged_count": len(unstaged),
            "untracked_count": len(untracked),
        }

    def _status_code_to_name(self, code: str) -> str:
        """Convert git status code to readable name."""
        codes = {
            "M": "modified",
            "A": "added",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
            "U": "unmerged",
            "?": "untracked",
            "!": "ignored",
        }
        return codes.get(code, f"unknown({code})")

    def diff(
        self,
        path: Optional[str] = None,
        staged: bool = False,
        commit: Optional[str] = None,
        stat_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Get diff of changes.

        Args:
            path: Specific file/directory to diff
            staged: Show staged changes (--cached)
            commit: Compare against specific commit
            stat_only: Only show statistics, not full diff

        Returns:
            Dict with diff content or stats
        """
        self._check_operation_allowed(GitOperationType.READ)

        args = ["diff"]

        if staged:
            args.append("--cached")

        if commit:
            args.append(commit)

        if stat_only:
            args.append("--stat")

        if path:
            args.extend(["--", path])

        result = self._run_git(args)

        if stat_only:
            # Parse stat output
            lines = result.stdout.strip().split("\n")
            files_changed = []
            summary = None

            for line in lines:
                if "|" in line:
                    parts = line.split("|")
                    filename = parts[0].strip()
                    changes = parts[1].strip() if len(parts) > 1 else ""
                    files_changed.append({"file": filename, "changes": changes})
                elif "file" in line and "changed" in line:
                    summary = line.strip()

            return {
                "success": True,
                "files_changed": files_changed,
                "summary": summary,
                "stat_only": True,
            }

        return {
            "success": True,
            "diff": result.stdout,
            "staged": staged,
            "path": path,
            "commit": commit,
        }

    def log(
        self,
        count: int = 10,
        path: Optional[str] = None,
        author: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        oneline: bool = False,
    ) -> Dict[str, Any]:
        """
        Get commit history.

        Args:
            count: Number of commits to retrieve
            path: Filter by file/directory
            author: Filter by author
            since: Filter commits since date (e.g., "2024-01-01")
            until: Filter commits until date
            oneline: Compact format

        Returns:
            Dict with list of commits
        """
        self._check_operation_allowed(GitOperationType.READ)

        if oneline:
            args = ["log", f"-{count}", "--oneline"]
            if path:
                args.extend(["--", path])

            result = self._run_git(args)

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(" ", 1)
                    commits.append({
                        "hash": parts[0],
                        "message": parts[1] if len(parts) > 1 else "",
                    })

            return {"success": True, "commits": commits, "count": len(commits)}

        # Full format
        format_str = "%H|%h|%an|%ae|%at|%s"
        args = ["log", f"-{count}", f"--format={format_str}"]

        if author:
            args.append(f"--author={author}")
        if since:
            args.append(f"--since={since}")
        if until:
            args.append(f"--until={until}")
        if path:
            args.extend(["--", path])

        result = self._run_git(args)

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 6:
                commits.append({
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "author_name": parts[2],
                    "author_email": parts[3],
                    "timestamp": datetime.fromtimestamp(int(parts[4])).isoformat(),
                    "message": parts[5],
                })

        return {"success": True, "commits": commits, "count": len(commits)}

    def show(
        self,
        commit: str = "HEAD",
        path: Optional[str] = None,
        stat_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Show commit details.

        Args:
            commit: Commit hash or reference
            path: Specific file to show
            stat_only: Only show file stats

        Returns:
            Dict with commit details and diff
        """
        self._check_operation_allowed(GitOperationType.READ)

        args = ["show", commit]

        if stat_only:
            args.append("--stat")

        if path:
            args.extend(["--", path])

        result = self._run_git(args)

        return {
            "success": True,
            "commit": commit,
            "content": result.stdout,
            "path": path,
            "stat_only": stat_only,
        }

    def blame(
        self,
        path: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Show line-by-line blame for a file.

        Args:
            path: File path
            line_start: Start line number
            line_end: End line number

        Returns:
            Dict with blame information
        """
        self._check_operation_allowed(GitOperationType.READ)

        args = ["blame", "--porcelain"]

        if line_start and line_end:
            args.extend([f"-L{line_start},{line_end}"])

        args.append(path)

        try:
            result = self._run_git(args)
        except GitOperationError as e:
            if "no such path" in str(e).lower():
                return {"error": f"File not found: {path}", "success": False}
            raise

        # Parse porcelain blame output
        lines = []
        current_commit = None
        commit_info = {}

        for line in result.stdout.split("\n"):
            if not line:
                continue

            # New commit header
            if len(line) == 40 or (len(line) > 40 and line[40] == " "):
                parts = line.split()
                current_commit = parts[0]
                if current_commit not in commit_info:
                    commit_info[current_commit] = {}
            elif line.startswith("author "):
                if current_commit:
                    commit_info[current_commit]["author"] = line[7:]
            elif line.startswith("author-time "):
                if current_commit:
                    commit_info[current_commit]["time"] = int(line[12:])
            elif line.startswith("summary "):
                if current_commit:
                    commit_info[current_commit]["summary"] = line[8:]
            elif line.startswith("\t"):
                # This is the actual code line
                if current_commit:
                    info = commit_info.get(current_commit, {})
                    lines.append({
                        "commit": current_commit[:8],
                        "author": info.get("author", ""),
                        "content": line[1:],
                    })

        return {
            "success": True,
            "path": path,
            "lines": lines,
            "line_count": len(lines),
        }

    # =========================================================================
    # Write Operations
    # =========================================================================

    def add(
        self,
        paths: List[str],
        all: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage files for commit.

        Args:
            paths: List of file paths to stage
            all: Stage all changes (use with caution)

        Returns:
            Dict with staged files
        """
        self._check_operation_allowed(GitOperationType.WRITE)

        if all:
            args = ["add", "-A"]
        else:
            if not paths:
                return {"error": "No paths specified", "success": False}

            if len(paths) > self.config.max_commit_files:
                return {
                    "error": f"Too many files ({len(paths)}). Maximum: {self.config.max_commit_files}",
                    "success": False,
                }

            args = ["add", "--"] + paths

        self._run_git(args)

        return {
            "success": True,
            "staged_paths": paths if not all else ["(all changes)"],
            "all": all,
        }

    def reset(
        self,
        paths: Optional[List[str]] = None,
        soft: bool = True,
    ) -> Dict[str, Any]:
        """
        Unstage files or reset to a state.

        Args:
            paths: Files to unstage (None = all staged)
            soft: Soft reset (keep changes in working tree)

        Returns:
            Dict with reset result
        """
        self._check_operation_allowed(GitOperationType.WRITE)

        if not soft:
            # Hard reset is dangerous
            self._check_operation_allowed(GitOperationType.DANGEROUS)

        if paths:
            args = ["reset", "HEAD", "--"] + paths
        else:
            args = ["reset", "--soft" if soft else "--hard", "HEAD"]

        self._run_git(args)

        return {
            "success": True,
            "paths": paths or ["(all staged)"],
            "soft": soft,
        }

    def commit(
        self,
        message: str,
        files: Optional[List[str]] = None,
        amend: bool = False,
        allow_empty: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a commit.

        Args:
            message: Commit message
            files: Specific files to commit (None = staged files)
            amend: Amend the last commit
            allow_empty: Allow empty commits

        Returns:
            Dict with commit details
        """
        self._check_operation_allowed(GitOperationType.WRITE)

        if self.config.require_commit_message and not message.strip():
            return {"error": "Commit message is required", "success": False}

        # Check if we're on a protected branch and amending
        current_branch = self._get_current_branch()
        if amend and self._is_protected_branch(current_branch):
            return {
                "error": f"Cannot amend commits on protected branch: {current_branch}",
                "success": False,
            }

        # Stage specific files if provided
        if files:
            if len(files) > self.config.max_commit_files:
                return {
                    "error": f"Too many files ({len(files)}). Maximum: {self.config.max_commit_files}",
                    "success": False,
                }
            self.add(files)

        args = ["commit", "-m", message]

        if amend:
            args.append("--amend")

        if allow_empty:
            args.append("--allow-empty")

        if self.config.commit_author:
            args.extend(["--author", self.config.commit_author])

        try:
            result = self._run_git(args)
        except GitOperationError as e:
            if "nothing to commit" in str(e).lower():
                return {
                    "error": "Nothing to commit. Stage changes first with git add.",
                    "success": False,
                }
            raise

        # Get the commit hash
        hash_result = self._run_git(["rev-parse", "HEAD"])
        commit_hash = hash_result.stdout.strip()

        return {
            "success": True,
            "commit_hash": commit_hash,
            "message": message,
            "branch": current_branch,
            "amended": amend,
        }

    def stash(
        self,
        message: Optional[str] = None,
        include_untracked: bool = False,
    ) -> Dict[str, Any]:
        """
        Stash current changes.

        Args:
            message: Stash message
            include_untracked: Include untracked files

        Returns:
            Dict with stash result
        """
        self._check_operation_allowed(GitOperationType.WRITE)

        args = ["stash", "push"]

        if message:
            args.extend(["-m", message])

        if include_untracked:
            args.append("-u")

        result = self._run_git(args)

        return {
            "success": True,
            "message": message,
            "include_untracked": include_untracked,
            "output": result.stdout.strip(),
        }

    def stash_pop(self, index: int = 0) -> Dict[str, Any]:
        """Pop a stash."""
        self._check_operation_allowed(GitOperationType.WRITE)

        args = ["stash", "pop", f"stash@{{{index}}}"]

        try:
            result = self._run_git(args)
            return {"success": True, "output": result.stdout.strip()}
        except GitOperationError as e:
            if "no stash" in str(e).lower():
                return {"error": "No stash entries found", "success": False}
            raise

    def stash_list(self) -> Dict[str, Any]:
        """List all stashes."""
        self._check_operation_allowed(GitOperationType.READ)

        result = self._run_git(["stash", "list"])

        stashes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                # Parse: stash@{0}: WIP on branch: message
                match = re.match(r"stash@\{(\d+)\}: (.+)", line)
                if match:
                    stashes.append({
                        "index": int(match.group(1)),
                        "description": match.group(2),
                    })

        return {"success": True, "stashes": stashes, "count": len(stashes)}

    # =========================================================================
    # Branch Operations
    # =========================================================================

    def branch(
        self,
        name: Optional[str] = None,
        delete: bool = False,
        list_all: bool = False,
        list_remote: bool = False,
    ) -> Dict[str, Any]:
        """
        Manage branches.

        Args:
            name: Branch name (for create/delete)
            delete: Delete the branch
            list_all: List all branches including remote
            list_remote: List only remote branches

        Returns:
            Dict with branch operation result
        """
        self._check_operation_allowed(GitOperationType.BRANCH)

        if list_all or list_remote or name is None:
            # List branches
            args = ["branch"]
            if list_all:
                args.append("-a")
            elif list_remote:
                args.append("-r")

            result = self._run_git(args)

            branches = []
            current = None
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                is_current = line.startswith("* ")
                branch_name = line[2:].strip()

                if is_current:
                    current = branch_name

                branches.append({
                    "name": branch_name,
                    "current": is_current,
                    "remote": branch_name.startswith("remotes/"),
                })

            return {
                "success": True,
                "branches": branches,
                "current": current,
                "count": len(branches),
            }

        if delete:
            # Delete branch
            if self._is_protected_branch(name):
                return {
                    "error": f"Cannot delete protected branch: {name}",
                    "success": False,
                }

            current = self._get_current_branch()
            if name == current:
                return {
                    "error": "Cannot delete the current branch",
                    "success": False,
                }

            self._run_git(["branch", "-d", name])
            return {"success": True, "deleted": name}

        else:
            # Create branch
            self._run_git(["branch", name])
            return {"success": True, "created": name}

    def checkout(
        self,
        target: str,
        create: bool = False,
        files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Checkout a branch or files.

        Args:
            target: Branch name or commit hash
            create: Create the branch if it doesn't exist
            files: Checkout specific files only

        Returns:
            Dict with checkout result
        """
        self._check_operation_allowed(GitOperationType.BRANCH)

        if files:
            # Checkout specific files
            args = ["checkout", target, "--"] + files
            self._run_git(args)
            return {"success": True, "files": files, "from": target}

        # Check for uncommitted changes
        status = self.status()
        if not status["clean"]:
            return {
                "error": "Cannot checkout with uncommitted changes. Commit or stash first.",
                "success": False,
                "staged": status["staged_count"],
                "unstaged": status["unstaged_count"],
            }

        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(target)

        self._run_git(args)

        return {
            "success": True,
            "branch": target,
            "created": create,
        }

    def merge(
        self,
        branch: str,
        message: Optional[str] = None,
        no_ff: bool = False,
        abort: bool = False,
    ) -> Dict[str, Any]:
        """
        Merge a branch.

        Args:
            branch: Branch to merge
            message: Merge commit message
            no_ff: Force merge commit even if fast-forward possible
            abort: Abort an in-progress merge

        Returns:
            Dict with merge result
        """
        self._check_operation_allowed(GitOperationType.BRANCH)

        if abort:
            self._run_git(["merge", "--abort"])
            return {"success": True, "aborted": True}

        current = self._get_current_branch()

        # Check if merging into protected branch
        if self._is_protected_branch(current):
            return {
                "error": f"Cannot merge into protected branch: {current}. "
                        "Create a pull request instead.",
                "success": False,
            }

        args = ["merge", branch]

        if no_ff:
            args.append("--no-ff")

        if message:
            args.extend(["-m", message])

        try:
            result = self._run_git(args)
            return {
                "success": True,
                "merged": branch,
                "into": current,
                "output": result.stdout.strip(),
            }
        except GitOperationError as e:
            if "conflict" in str(e).lower():
                return {
                    "error": "Merge conflict detected. Resolve conflicts and commit, or use abort=True.",
                    "success": False,
                    "conflicts": True,
                }
            raise

    # =========================================================================
    # Remote Operations
    # =========================================================================

    def fetch(
        self,
        remote: str = "origin",
        prune: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch from remote.

        Args:
            remote: Remote name
            prune: Prune deleted remote branches

        Returns:
            Dict with fetch result
        """
        self._check_operation_allowed(GitOperationType.REMOTE)

        args = ["fetch", remote]
        if prune:
            args.append("--prune")

        result = self._run_git(args)

        return {
            "success": True,
            "remote": remote,
            "output": result.stdout.strip() or result.stderr.strip(),
        }

    def pull(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        rebase: bool = False,
    ) -> Dict[str, Any]:
        """
        Pull from remote.

        Args:
            remote: Remote name
            branch: Branch to pull (default: current)
            rebase: Use rebase instead of merge

        Returns:
            Dict with pull result
        """
        self._check_operation_allowed(GitOperationType.REMOTE)

        current = self._get_current_branch()

        # Check for uncommitted changes
        status = self.status()
        if not status["clean"]:
            return {
                "error": "Cannot pull with uncommitted changes. Commit or stash first.",
                "success": False,
            }

        args = ["pull"]
        if rebase:
            args.append("--rebase")
        args.append(remote)
        if branch:
            args.append(branch)

        try:
            result = self._run_git(args)
            return {
                "success": True,
                "remote": remote,
                "branch": branch or current,
                "output": result.stdout.strip(),
            }
        except GitOperationError as e:
            if "conflict" in str(e).lower():
                return {
                    "error": "Pull resulted in conflicts. Resolve and commit.",
                    "success": False,
                    "conflicts": True,
                }
            raise

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        set_upstream: bool = False,
        force: bool = False,
        tags: bool = False,
    ) -> Dict[str, Any]:
        """
        Push to remote.

        Args:
            remote: Remote name
            branch: Branch to push (default: current)
            set_upstream: Set upstream tracking
            force: Force push (DANGEROUS)
            tags: Push tags

        Returns:
            Dict with push result
        """
        self._check_operation_allowed(GitOperationType.REMOTE)

        if force:
            self._check_operation_allowed(GitOperationType.DANGEROUS)

        current = self._get_current_branch()
        target_branch = branch or current

        # Prevent force push to protected branches
        if force and self._is_protected_branch(target_branch):
            return {
                "error": f"Cannot force push to protected branch: {target_branch}",
                "success": False,
            }

        args = ["push"]

        if set_upstream:
            args.extend(["-u", remote, target_branch])
        else:
            args.append(remote)
            if branch:
                args.append(branch)

        if force:
            args.append("--force")

        if tags:
            args.append("--tags")

        result = self._run_git(args)

        return {
            "success": True,
            "remote": remote,
            "branch": target_branch,
            "force": force,
            "output": result.stdout.strip() or result.stderr.strip(),
        }

    def remote_list(self) -> Dict[str, Any]:
        """List configured remotes."""
        self._check_operation_allowed(GitOperationType.READ)

        result = self._run_git(["remote", "-v"])

        remotes = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                op_type = parts[2].strip("()") if len(parts) > 2 else "fetch"

                if name not in remotes:
                    remotes[name] = {}
                remotes[name][op_type] = url

        return {
            "success": True,
            "remotes": [
                {"name": name, "urls": urls}
                for name, urls in remotes.items()
            ],
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_repo_info(self) -> Dict[str, Any]:
        """Get repository information."""
        self._check_operation_allowed(GitOperationType.READ)

        current_branch = self._get_current_branch()

        # Get origin URL
        try:
            origin_result = self._run_git(["config", "--get", "remote.origin.url"])
            origin_url = origin_result.stdout.strip()
        except GitOperationError:
            origin_url = None

        # Get latest commit
        try:
            commit_result = self._run_git(["rev-parse", "HEAD"])
            head_commit = commit_result.stdout.strip()
        except GitOperationError:
            head_commit = None

        # Count commits
        try:
            count_result = self._run_git(["rev-list", "--count", "HEAD"])
            commit_count = int(count_result.stdout.strip())
        except GitOperationError:
            commit_count = 0

        return {
            "success": True,
            "path": self.config.repo_path,
            "current_branch": current_branch,
            "origin_url": origin_url,
            "head_commit": head_commit,
            "commit_count": commit_count,
            "protected_branches": self.config.protected_branches,
        }

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the log of all git operations."""
        return self._operation_log.copy()


# =============================================================================
# Factory and Singleton
# =============================================================================

_git_tools: Optional[GitTools] = None


def get_git_tools(repo_path: Optional[str] = None) -> GitTools:
    """Get or create git tools instance."""
    global _git_tools

    if _git_tools is None or repo_path:
        path = repo_path or os.getcwd()
        config = GitConfig(repo_path=path)
        _git_tools = GitTools(config)

    return _git_tools


def configure_git_tools(config: GitConfig) -> GitTools:
    """Configure git tools with custom settings."""
    global _git_tools
    _git_tools = GitTools(config)
    return _git_tools
