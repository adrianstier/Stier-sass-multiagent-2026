"""Tests for code analysis tools module."""

import os
import tempfile
import pytest

from orchestrator.tools.code_analysis import (
    CodeAnalyzer,
    CodeAnalysisConfig,
    Symbol,
    SymbolKind,
)


class TestSymbol:
    """Tests for Symbol dataclass."""

    def test_symbol_creation(self):
        """Should create symbol with required fields."""
        symbol = Symbol(
            name="test_function",
            kind=SymbolKind.FUNCTION,
            file_path="/test/file.py",
            line_start=10,
            line_end=20,
        )
        assert symbol.name == "test_function"
        assert symbol.kind == SymbolKind.FUNCTION
        assert symbol.line_start == 10

    def test_symbol_with_parent(self):
        """Should create symbol with parent reference."""
        symbol = Symbol(
            name="method",
            kind=SymbolKind.METHOD,
            file_path="/test/file.py",
            line_start=15,
            line_end=25,
            parent="MyClass",
        )
        assert symbol.parent == "MyClass"


class TestSymbolKind:
    """Tests for SymbolKind enum."""

    def test_all_kinds_exist(self):
        """Should have all expected symbol kinds."""
        assert SymbolKind.MODULE
        assert SymbolKind.CLASS
        assert SymbolKind.FUNCTION
        assert SymbolKind.METHOD
        assert SymbolKind.VARIABLE
        assert SymbolKind.CONSTANT

    def test_kind_values(self):
        """Symbol kinds should have string values."""
        assert SymbolKind.CLASS.value == "class"
        assert SymbolKind.FUNCTION.value == "function"
        assert SymbolKind.METHOD.value == "method"


class TestCodeAnalysisConfig:
    """Tests for CodeAnalysisConfig."""

    def test_config_creation(self):
        """Should create config with root_path."""
        config = CodeAnalysisConfig(root_path="/tmp/project")
        assert config.root_path == "/tmp/project"

    def test_default_exclude_patterns(self):
        """Default config should exclude common directories."""
        config = CodeAnalysisConfig(root_path="/tmp")
        assert any("node_modules" in p for p in config.exclude_patterns)
        assert any(".git" in p for p in config.exclude_patterns)
        assert any("__pycache__" in p for p in config.exclude_patterns)


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python file
            py_file = os.path.join(tmpdir, "example.py")
            with open(py_file, "w") as f:
                f.write('''"""Example module."""

class MyClass:
    """A sample class."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        """Return the value."""
        return self.value


def standalone_function(x, y):
    """Add two numbers."""
    return x + y


CONSTANT = 42
''')

            # Create JavaScript file
            js_file = os.path.join(tmpdir, "example.js")
            with open(js_file, "w") as f:
                f.write('''// Example JavaScript file

class User {
    constructor(name) {
        this.name = name;
    }

    getName() {
        return this.name;
    }
}

function greet(user) {
    return `Hello, ${user.getName()}!`;
}
''')

            yield tmpdir

    @pytest.fixture
    def analyzer(self, temp_project):
        """Create CodeAnalyzer for temp project."""
        config = CodeAnalysisConfig(root_path=temp_project)
        return CodeAnalyzer(config)

    def test_extract_python_symbols(self, analyzer, temp_project):
        """Should extract symbols from Python file."""
        # Returns List[Symbol], not a dict
        symbols = analyzer.extract_symbols(
            os.path.join(temp_project, "example.py")
        )

        assert isinstance(symbols, list)
        assert len(symbols) > 0

        # Check class was found
        class_names = [s.name for s in symbols if s.kind == SymbolKind.CLASS]
        assert "MyClass" in class_names

        # Check functions were found
        func_names = [s.name for s in symbols if s.kind == SymbolKind.FUNCTION]
        assert "standalone_function" in func_names

    def test_search_code(self, analyzer, temp_project):
        """Should search for patterns in code."""
        # Returns List[Dict], not a dict with "success" key
        results = analyzer.search_code("def.*function", is_regex=True)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_project_structure(self, analyzer):
        """Should return project structure."""
        result = analyzer.get_project_structure()

        # Returns a dict with project structure
        assert isinstance(result, dict)
        assert "name" in result or "children" in result

    def test_nonexistent_file(self, analyzer):
        """Should handle non-existent files gracefully."""
        # Returns empty list for non-existent files
        symbols = analyzer.extract_symbols("/nonexistent/file.py")

        assert isinstance(symbols, list)
        assert len(symbols) == 0


class TestLanguageDetection:
    """Tests for language detection."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def analyzer(self, temp_dir):
        config = CodeAnalysisConfig(root_path=temp_dir)
        return CodeAnalyzer(config)

    def test_detect_python(self, analyzer, temp_dir):
        """Should detect Python files."""
        py_file = os.path.join(temp_dir, "test.py")
        open(py_file, "w").close()

        # Method is _get_file_language, not _detect_language
        lang = analyzer._get_file_language(py_file)
        assert lang == "python"

    def test_detect_javascript(self, analyzer, temp_dir):
        """Should detect JavaScript files."""
        js_file = os.path.join(temp_dir, "test.js")
        open(js_file, "w").close()

        lang = analyzer._get_file_language(js_file)
        assert lang == "javascript"

    def test_detect_typescript(self, analyzer, temp_dir):
        """Should detect TypeScript files."""
        ts_file = os.path.join(temp_dir, "test.ts")
        open(ts_file, "w").close()

        lang = analyzer._get_file_language(ts_file)
        assert lang == "typescript"
