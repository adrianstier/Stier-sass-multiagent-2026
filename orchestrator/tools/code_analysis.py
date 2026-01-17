"""Code analysis tools for parsing, searching, and understanding codebases."""

import os
import re
import ast
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from enum import Enum


class SymbolKind(str, Enum):
    """Types of code symbols."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    INTERFACE = "interface"
    TYPE = "type"
    ENUM = "enum"


@dataclass
class Symbol:
    """Represents a code symbol (class, function, variable, etc.)."""
    name: str
    kind: SymbolKind
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent: Optional[str] = None  # Parent symbol name (for nested definitions)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_exported: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reference:
    """Represents a reference to a symbol."""
    symbol_name: str
    file_path: str
    line: int
    column: int
    context: str  # Line content
    ref_type: str  # "usage", "import", "definition", "assignment"


@dataclass
class CodeAnalysisConfig:
    """Configuration for code analysis."""
    root_path: str
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/venv/**",
        "**/.venv/**",
        "**/dist/**",
        "**/build/**",
        "**/*.min.js",
        "**/*.bundle.js",
    ])
    include_patterns: List[str] = field(default_factory=list)
    max_file_size_mb: float = 5.0
    parse_docstrings: bool = True


class CodeAnalyzer:
    """
    Multi-language code analysis for understanding codebases.

    Provides symbol extraction, reference finding, and code search
    capabilities across Python, JavaScript, TypeScript, and more.
    """

    def __init__(self, config: CodeAnalysisConfig):
        self.config = config
        self._symbol_cache: Dict[str, List[Symbol]] = {}
        self._file_hashes: Dict[str, str] = {}

    def _should_include_file(self, file_path: str) -> bool:
        """Check if a file should be included in analysis."""
        rel_path = os.path.relpath(file_path, self.config.root_path)

        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if self._match_glob(rel_path, pattern):
                return False

        # Check include patterns (if specified)
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if self._match_glob(rel_path, pattern):
                    return True
            return False

        return True

    def _match_glob(self, path: str, pattern: str) -> bool:
        """Simple glob matching."""
        import fnmatch
        # Handle ** patterns
        if "**" in pattern:
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                return path.startswith(prefix.rstrip("/")) and path.endswith(suffix.lstrip("/"))
        return fnmatch.fnmatch(path, pattern)

    def _get_file_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".java": "java",
            ".kt": "kotlin",
            ".scala": "scala",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".swift": "swift",
            ".php": "php",
        }
        return language_map.get(ext)

    # =========================================================================
    # Symbol Extraction
    # =========================================================================

    def extract_symbols(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> List[Symbol]:
        """
        Extract symbols from a source file.

        Args:
            file_path: Path to the source file
            language: Programming language (auto-detected if not specified)

        Returns:
            List of Symbol objects
        """
        if not os.path.isfile(file_path):
            return []

        if not self._should_include_file(file_path):
            return []

        lang = language or self._get_file_language(file_path)
        if not lang:
            return []

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.config.max_file_size_mb * 1024 * 1024:
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if lang == "python":
                return self._extract_python_symbols(file_path, content)
            elif lang in ["javascript", "typescript"]:
                return self._extract_js_ts_symbols(file_path, content, lang)
            else:
                # Fallback to regex-based extraction
                return self._extract_symbols_regex(file_path, content, lang)

        except Exception as e:
            return []

    def _extract_python_symbols(self, file_path: str, content: str) -> List[Symbol]:
        """Extract symbols from Python code using AST."""
        symbols = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        lines = content.split("\n")

        for node in ast.walk(tree):
            symbol = None

            if isinstance(node, ast.ClassDef):
                symbol = Symbol(
                    name=node.name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    column_start=node.col_offset,
                    docstring=ast.get_docstring(node) if self.config.parse_docstrings else None,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                )

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Determine if it's a method or function
                parent_name = None
                for potential_parent in ast.walk(tree):
                    if isinstance(potential_parent, ast.ClassDef):
                        for child in ast.iter_child_nodes(potential_parent):
                            if child is node:
                                parent_name = potential_parent.name
                                break

                kind = SymbolKind.METHOD if parent_name else SymbolKind.FUNCTION

                # Build signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)

                return_annotation = ""
                if node.returns:
                    return_annotation = f" -> {ast.unparse(node.returns)}"

                signature = f"def {node.name}({', '.join(args)}){return_annotation}"

                symbol = Symbol(
                    name=node.name,
                    kind=kind,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    column_start=node.col_offset,
                    docstring=ast.get_docstring(node) if self.config.parse_docstrings else None,
                    signature=signature,
                    parent=parent_name,
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )

            elif isinstance(node, ast.Assign):
                # Module-level variables
                if node.col_offset == 0:  # Top-level
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Check if it's a constant (ALL_CAPS)
                            is_constant = target.id.isupper()
                            symbol = Symbol(
                                name=target.id,
                                kind=SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE,
                                file_path=file_path,
                                line_start=node.lineno,
                                line_end=node.end_lineno or node.lineno,
                                column_start=node.col_offset,
                            )
                            symbols.append(symbol)
                            symbol = None  # Already added

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    symbol = Symbol(
                        name=alias.asname or alias.name,
                        kind=SymbolKind.IMPORT,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=node.lineno,
                        column_start=node.col_offset,
                        metadata={"module": alias.name},
                    )
                    symbols.append(symbol)
                    symbol = None

            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    symbol = Symbol(
                        name=alias.asname or alias.name,
                        kind=SymbolKind.IMPORT,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=node.lineno,
                        column_start=node.col_offset,
                        metadata={"module": node.module or "", "name": alias.name},
                    )
                    symbols.append(symbol)
                    symbol = None

            if symbol:
                symbols.append(symbol)

        return symbols

    def _get_decorator_name(self, node) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_decorator_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return str(node)

    def _extract_js_ts_symbols(self, file_path: str, content: str, lang: str) -> List[Symbol]:
        """Extract symbols from JavaScript/TypeScript using regex."""
        symbols = []
        lines = content.split("\n")

        # Class definitions
        class_pattern = r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{"
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(
                name=match.group(1),
                kind=SymbolKind.CLASS,
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,  # Would need proper parsing for end
                column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                is_exported="export" in match.group(0),
            ))

        # Function declarations
        func_pattern = r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)"
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(
                name=match.group(1),
                kind=SymbolKind.FUNCTION,
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,
                column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                is_async="async" in match.group(0),
                is_exported="export" in match.group(0),
            ))

        # Arrow functions (const name = () => or const name = async () =>)
        arrow_pattern = r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+(?:<[^>]+>)?)?\s*=>"
        for match in re.finditer(arrow_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(
                name=match.group(1),
                kind=SymbolKind.FUNCTION,
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,
                column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                is_async="async" in match.group(0),
                is_exported="export" in match.group(0),
            ))

        # Interface definitions (TypeScript)
        if lang == "typescript":
            interface_pattern = r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s<>]+)?\s*\{"
            for match in re.finditer(interface_pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                symbols.append(Symbol(
                    name=match.group(1),
                    kind=SymbolKind.INTERFACE,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                    is_exported="export" in match.group(0),
                ))

            # Type definitions
            type_pattern = r"(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*="
            for match in re.finditer(type_pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                symbols.append(Symbol(
                    name=match.group(1),
                    kind=SymbolKind.TYPE,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                    is_exported="export" in match.group(0),
                ))

        # Enum definitions
        enum_pattern = r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)\s*\{"
        for match in re.finditer(enum_pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            symbols.append(Symbol(
                name=match.group(1),
                kind=SymbolKind.ENUM,
                file_path=file_path,
                line_start=line_num,
                line_end=line_num,
                column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                is_exported="export" in match.group(0),
            ))

        return symbols

    def _extract_symbols_regex(self, file_path: str, content: str, lang: str) -> List[Symbol]:
        """Fallback regex-based symbol extraction for other languages."""
        symbols = []

        # Generic patterns that work across many languages
        patterns = {
            "class": r"(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?class\s+(\w+)",
            "function": r"(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:async\s+)?(?:fn|func|function|def)\s+(\w+)",
            "interface": r"interface\s+(\w+)",
            "struct": r"struct\s+(\w+)",
            "enum": r"enum\s+(\w+)",
        }

        for kind_name, pattern in patterns.items():
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                symbols.append(Symbol(
                    name=match.group(1),
                    kind=SymbolKind.CLASS if kind_name in ["class", "struct"] else
                         SymbolKind.FUNCTION if kind_name == "function" else
                         SymbolKind.INTERFACE if kind_name == "interface" else
                         SymbolKind.ENUM,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    column_start=match.start() - content.rfind("\n", 0, match.start()) - 1,
                ))

        return symbols

    # =========================================================================
    # Reference Finding
    # =========================================================================

    def find_references(
        self,
        symbol_name: str,
        search_path: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
    ) -> List[Reference]:
        """
        Find all references to a symbol.

        Args:
            symbol_name: Name of the symbol to find
            search_path: Path to search in (default: config.root_path)
            file_extensions: File extensions to search (auto-detected if not specified)

        Returns:
            List of Reference objects
        """
        references = []
        search_root = search_path or self.config.root_path

        # Build file list
        files_to_search = []
        for root, dirs, files in os.walk(search_root):
            # Filter directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "venv", "__pycache__"]]

            for file in files:
                file_path = os.path.join(root, file)
                if not self._should_include_file(file_path):
                    continue

                if file_extensions:
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue

                files_to_search.append(file_path)

        # Search each file
        # Build regex pattern - match whole word
        pattern = re.compile(rf"\b{re.escape(symbol_name)}\b")

        for file_path in files_to_search:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    for match in pattern.finditer(line):
                        # Determine reference type
                        ref_type = self._classify_reference(line, match.start(), symbol_name)

                        references.append(Reference(
                            symbol_name=symbol_name,
                            file_path=file_path,
                            line=line_num,
                            column=match.start(),
                            context=line.strip(),
                            ref_type=ref_type,
                        ))

            except Exception:
                continue

        return references

    def _classify_reference(self, line: str, column: int, symbol_name: str) -> str:
        """Classify the type of reference."""
        before = line[:column].strip()
        after = line[column + len(symbol_name):].strip()

        # Import patterns
        if re.match(r"^(from|import)\s+", line.strip()):
            return "import"

        # Definition patterns
        if re.match(r"^(class|def|function|func|fn|interface|type|struct|enum)\s+$", before):
            return "definition"

        # Assignment patterns
        if after.startswith("=") and not after.startswith("=="):
            return "assignment"

        # Function call
        if after.startswith("("):
            return "call"

        # Attribute access
        if before.endswith("."):
            return "attribute"

        return "usage"

    def find_definition(
        self,
        symbol_name: str,
        file_path: Optional[str] = None,
    ) -> Optional[Symbol]:
        """
        Find the definition of a symbol.

        Args:
            symbol_name: Name of the symbol
            file_path: Start search from this file (searches imports)

        Returns:
            Symbol object if found, None otherwise
        """
        # Search in specified file first
        if file_path and os.path.isfile(file_path):
            symbols = self.extract_symbols(file_path)
            for symbol in symbols:
                if symbol.name == symbol_name and symbol.kind != SymbolKind.IMPORT:
                    return symbol

        # Search entire project
        for root, dirs, files in os.walk(self.config.root_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "venv", "__pycache__"]]

            for file in files:
                full_path = os.path.join(root, file)
                if not self._should_include_file(full_path):
                    continue

                symbols = self.extract_symbols(full_path)
                for symbol in symbols:
                    if symbol.name == symbol_name and symbol.kind != SymbolKind.IMPORT:
                        return symbol

        return None

    # =========================================================================
    # Code Search
    # =========================================================================

    def search_code(
        self,
        pattern: str,
        is_regex: bool = False,
        case_sensitive: bool = True,
        file_extensions: Optional[List[str]] = None,
        max_results: int = 100,
        context_lines: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Search for code patterns across the codebase.

        Args:
            pattern: Search pattern (string or regex)
            is_regex: Treat pattern as regex
            case_sensitive: Case-sensitive search
            file_extensions: File extensions to search
            max_results: Maximum number of results
            context_lines: Lines of context before/after match

        Returns:
            List of search results with context
        """
        results = []

        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return [{"error": f"Invalid regex: {e}"}]
        else:
            if not case_sensitive:
                pattern = pattern.lower()

        for root, dirs, files in os.walk(self.config.root_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "venv", "__pycache__"]]

            if len(results) >= max_results:
                break

            for file in files:
                if len(results) >= max_results:
                    break

                file_path = os.path.join(root, file)
                if not self._should_include_file(file_path):
                    continue

                if file_extensions:
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        search_line = line if case_sensitive else line.lower()

                        if is_regex:
                            match = regex.search(line)
                            if match:
                                # Get context
                                start = max(0, line_num - context_lines - 1)
                                end = min(len(lines), line_num + context_lines)
                                context = lines[start:end]

                                results.append({
                                    "file": os.path.relpath(file_path, self.config.root_path),
                                    "line": line_num,
                                    "column": match.start(),
                                    "match": match.group(),
                                    "content": line.rstrip(),
                                    "context": [l.rstrip() for l in context],
                                    "context_start_line": start + 1,
                                })
                        else:
                            if pattern in search_line:
                                col = search_line.find(pattern)
                                start = max(0, line_num - context_lines - 1)
                                end = min(len(lines), line_num + context_lines)
                                context = lines[start:end]

                                results.append({
                                    "file": os.path.relpath(file_path, self.config.root_path),
                                    "line": line_num,
                                    "column": col,
                                    "match": line[col:col + len(pattern)],
                                    "content": line.rstrip(),
                                    "context": [l.rstrip() for l in context],
                                    "context_start_line": start + 1,
                                })

                except Exception:
                    continue

        return results

    # =========================================================================
    # Project Structure Analysis
    # =========================================================================

    def get_project_structure(
        self,
        max_depth: int = 5,
        include_file_counts: bool = True,
    ) -> Dict[str, Any]:
        """
        Get project directory structure.

        Args:
            max_depth: Maximum directory depth
            include_file_counts: Include file counts per directory

        Returns:
            Dict representing project structure
        """
        def build_tree(path: str, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"truncated": True}

            result = {
                "name": os.path.basename(path) or path,
                "type": "directory",
                "children": [],
            }

            if include_file_counts:
                result["file_count"] = 0

            try:
                entries = sorted(os.listdir(path))
            except PermissionError:
                return result

            for entry in entries:
                if entry.startswith("."):
                    continue

                full_path = os.path.join(path, entry)

                if os.path.isdir(full_path):
                    if entry in ["node_modules", "venv", "__pycache__", ".git", "dist", "build"]:
                        continue
                    child = build_tree(full_path, depth + 1)
                    result["children"].append(child)
                    if include_file_counts:
                        result["file_count"] += child.get("file_count", 0)
                else:
                    if self._should_include_file(full_path):
                        result["children"].append({
                            "name": entry,
                            "type": "file",
                            "extension": Path(entry).suffix,
                        })
                        if include_file_counts:
                            result["file_count"] += 1

            return result

        return build_tree(self.config.root_path)

    def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze dependencies/imports in a file.

        Args:
            file_path: Path to analyze

        Returns:
            Dict with import information
        """
        if not os.path.isfile(file_path):
            return {"error": f"File not found: {file_path}"}

        lang = self._get_file_language(file_path)
        if not lang:
            return {"error": "Unsupported file type"}

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return {"error": str(e)}

        imports = {
            "internal": [],  # Project imports
            "external": [],  # Package imports
            "standard": [],  # Standard library
        }

        if lang == "python":
            return self._analyze_python_imports(content, file_path, imports)
        elif lang in ["javascript", "typescript"]:
            return self._analyze_js_imports(content, file_path, imports)

        return imports

    def _analyze_python_imports(self, content: str, file_path: str, imports: Dict) -> Dict[str, Any]:
        """Analyze Python imports."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"error": "Syntax error in file"}

        standard_lib = {
            "os", "sys", "re", "json", "typing", "collections", "itertools",
            "functools", "datetime", "pathlib", "asyncio", "logging", "unittest",
            "abc", "dataclasses", "enum", "uuid", "hashlib", "base64", "io",
            "subprocess", "threading", "multiprocessing", "contextlib", "copy",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in standard_lib:
                        imports["standard"].append(alias.name)
                    elif self._is_internal_import(alias.name, file_path):
                        imports["internal"].append(alias.name)
                    else:
                        imports["external"].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                base_module = module.split(".")[0]

                if base_module in standard_lib:
                    imports["standard"].append(f"{module}.{node.names[0].name}")
                elif self._is_internal_import(module, file_path):
                    imports["internal"].append(f"{module}.{node.names[0].name}")
                else:
                    imports["external"].append(f"{module}.{node.names[0].name}")

        return {
            "file": file_path,
            "language": "python",
            "imports": imports,
            "total_imports": sum(len(v) for v in imports.values()),
        }

    def _analyze_js_imports(self, content: str, file_path: str, imports: Dict) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript imports."""
        # ES6 imports
        import_pattern = r"import\s+(?:{[^}]+}|\w+|\*\s+as\s+\w+)\s+from\s+['\"]([^'\"]+)['\"]"
        require_pattern = r"require\s*\(['\"]([^'\"]+)['\"]\)"

        all_imports = re.findall(import_pattern, content) + re.findall(require_pattern, content)

        for imp in all_imports:
            if imp.startswith(".") or imp.startswith("/"):
                imports["internal"].append(imp)
            elif imp.startswith("@") or "/" not in imp:
                imports["external"].append(imp)
            else:
                imports["external"].append(imp)

        return {
            "file": file_path,
            "language": "javascript/typescript",
            "imports": imports,
            "total_imports": sum(len(v) for v in imports.values()),
        }

    def _is_internal_import(self, module: str, file_path: str) -> bool:
        """Check if an import is project-internal."""
        # Check if module path exists relative to project
        module_path = module.replace(".", os.sep)
        possible_paths = [
            os.path.join(self.config.root_path, module_path + ".py"),
            os.path.join(self.config.root_path, module_path, "__init__.py"),
        ]
        return any(os.path.exists(p) for p in possible_paths)

    def get_symbol_outline(
        self,
        file_path: str,
        include_private: bool = False,
    ) -> Dict[str, Any]:
        """
        Get a hierarchical outline of symbols in a file.

        Args:
            file_path: Path to the file
            include_private: Include private symbols (starting with _)

        Returns:
            Dict with hierarchical symbol outline
        """
        symbols = self.extract_symbols(file_path)

        if not include_private:
            symbols = [s for s in symbols if not s.name.startswith("_") or s.name.startswith("__")]

        # Build hierarchy
        outline = {
            "file": file_path,
            "classes": [],
            "functions": [],
            "variables": [],
            "imports": [],
        }

        class_symbols = {s.name: s for s in symbols if s.kind == SymbolKind.CLASS}
        method_symbols = [s for s in symbols if s.kind == SymbolKind.METHOD]

        for class_sym in class_symbols.values():
            class_outline = {
                "name": class_sym.name,
                "line": class_sym.line_start,
                "docstring": class_sym.docstring,
                "decorators": class_sym.decorators,
                "methods": [],
            }

            # Find methods belonging to this class
            for method in method_symbols:
                if method.parent == class_sym.name:
                    class_outline["methods"].append({
                        "name": method.name,
                        "line": method.line_start,
                        "signature": method.signature,
                        "is_async": method.is_async,
                        "decorators": method.decorators,
                    })

            outline["classes"].append(class_outline)

        # Top-level functions
        for sym in symbols:
            if sym.kind == SymbolKind.FUNCTION:
                outline["functions"].append({
                    "name": sym.name,
                    "line": sym.line_start,
                    "signature": sym.signature,
                    "is_async": sym.is_async,
                    "docstring": sym.docstring,
                })

        # Variables
        for sym in symbols:
            if sym.kind in [SymbolKind.VARIABLE, SymbolKind.CONSTANT]:
                outline["variables"].append({
                    "name": sym.name,
                    "line": sym.line_start,
                    "kind": sym.kind.value,
                })

        # Imports
        for sym in symbols:
            if sym.kind == SymbolKind.IMPORT:
                outline["imports"].append({
                    "name": sym.name,
                    "line": sym.line_start,
                    "module": sym.metadata.get("module"),
                })

        return outline


# =============================================================================
# Factory and Singleton
# =============================================================================

_code_analyzer: Optional[CodeAnalyzer] = None


def get_code_analyzer(root_path: Optional[str] = None) -> CodeAnalyzer:
    """Get or create code analyzer instance."""
    global _code_analyzer

    if _code_analyzer is None or root_path:
        path = root_path or os.getcwd()
        config = CodeAnalysisConfig(root_path=path)
        _code_analyzer = CodeAnalyzer(config)

    return _code_analyzer


def configure_code_analyzer(config: CodeAnalysisConfig) -> CodeAnalyzer:
    """Configure code analyzer with custom settings."""
    global _code_analyzer
    _code_analyzer = CodeAnalyzer(config)
    return _code_analyzer
