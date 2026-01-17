"""
Project State Analyzer

Analyzes existing codebases to understand current state before orchestration.
Supports both greenfield projects and ongoing development.

Key capabilities:
- Detect project type, language, and framework
- Analyze existing architecture and patterns
- Identify technical debt and improvement areas
- Map dependencies and their health
- Assess test coverage and code quality
- Generate context for agents working on existing code
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Column, DateTime, String, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Session

from orchestrator.core.models import Base, Event
from orchestrator.core.database import get_db

logger = logging.getLogger(__name__)


class ProjectType(str, Enum):
    """Types of projects the orchestrator can work with."""
    GREENFIELD = "greenfield"           # New project from scratch
    EXISTING_ACTIVE = "existing_active"  # Ongoing development
    LEGACY_MAINTENANCE = "legacy_maintenance"  # Legacy codebase
    FEATURE_BRANCH = "feature_branch"    # Working on specific feature
    BUG_FIX = "bug_fix"                  # Fixing specific issues
    REFACTOR = "refactor"                # Improving existing code


class FrameworkType(str, Enum):
    """Detected framework types."""
    FASTAPI = "fastapi"
    DJANGO = "django"
    FLASK = "flask"
    EXPRESS = "express"
    NEXTJS = "nextjs"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    RAILS = "rails"
    SPRING = "spring"
    DOTNET = "dotnet"
    UNKNOWN = "unknown"


@dataclass
class DependencyHealth:
    """Health status of a dependency."""
    name: str
    current_version: str
    latest_version: Optional[str] = None
    is_outdated: bool = False
    has_vulnerabilities: bool = False
    vulnerability_count: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class CodeQualityMetrics:
    """Code quality assessment."""
    total_files: int = 0
    total_lines: int = 0
    test_files: int = 0
    test_coverage_percent: Optional[float] = None
    lint_errors: int = 0
    lint_warnings: int = 0
    type_coverage_percent: Optional[float] = None
    complexity_score: Optional[float] = None
    duplication_percent: Optional[float] = None
    documentation_coverage: Optional[float] = None


@dataclass
class ArchitecturePattern:
    """Detected architecture pattern."""
    pattern_name: str  # e.g., "MVC", "microservices", "monolith"
    confidence: float  # 0-1
    evidence: list[str] = field(default_factory=list)


@dataclass
class TechnicalDebtItem:
    """Identified technical debt."""
    category: str  # "dependency", "architecture", "code_quality", "security", "testing"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    estimated_effort: Optional[str] = None  # "hours", "days", "weeks"


@dataclass
class SecurityIssue:
    """Identified security vulnerability or concern."""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "dependency", "code", "configuration", "secrets", "permissions"
    description: str
    location: Optional[str] = None
    cve_id: Optional[str] = None
    affected_component: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class ProjectState:
    """Complete state assessment of a project."""
    project_id: uuid.UUID
    project_path: str
    project_type: ProjectType
    analyzed_at: datetime

    # Basic info
    name: str
    description: Optional[str] = None
    primary_language: Optional[str] = None
    languages: dict[str, int] = field(default_factory=dict)  # language -> line count
    framework: FrameworkType = FrameworkType.UNKNOWN

    # Structure
    directory_structure: dict = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)

    # Dependencies
    dependencies: list[DependencyHealth] = field(default_factory=list)
    dev_dependencies: list[DependencyHealth] = field(default_factory=list)

    # Quality
    quality_metrics: CodeQualityMetrics = field(default_factory=CodeQualityMetrics)

    # Architecture
    architecture_patterns: list[ArchitecturePattern] = field(default_factory=list)
    api_endpoints: list[dict] = field(default_factory=list)
    database_models: list[str] = field(default_factory=list)

    # Issues
    technical_debt: list[TechnicalDebtItem] = field(default_factory=list)
    security_issues: list[dict] = field(default_factory=list)

    # Git info
    git_info: dict = field(default_factory=dict)
    recent_commits: list[dict] = field(default_factory=list)
    active_branches: list[str] = field(default_factory=list)
    contributors: list[str] = field(default_factory=list)

    # Context for agents
    agent_context: dict = field(default_factory=dict)


# Database model for persisting project analysis
class ProjectAnalysis(Base):
    """Persisted project analysis."""
    __tablename__ = "project_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_path = Column(String(1000), nullable=False, index=True)
    project_type = Column(String(50), nullable=False)
    analysis_data = Column(JSONB, nullable=False)
    is_current = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProjectAnalyzer:
    """
    Analyzes project state to provide context for orchestration.

    Supports:
    - Greenfield projects (minimal or no existing code)
    - Existing active development
    - Legacy codebases
    - Feature branches
    - Bug fixes
    - Refactoring efforts
    """

    # File patterns for language detection
    LANGUAGE_PATTERNS = {
        "python": ["*.py"],
        "javascript": ["*.js", "*.jsx"],
        "typescript": ["*.ts", "*.tsx"],
        "java": ["*.java"],
        "go": ["*.go"],
        "rust": ["*.rs"],
        "ruby": ["*.rb"],
        "php": ["*.php"],
        "csharp": ["*.cs"],
        "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp"],
        "c": ["*.c", "*.h"],
    }

    # Framework detection patterns
    FRAMEWORK_INDICATORS = {
        FrameworkType.FASTAPI: ["fastapi", "from fastapi import"],
        FrameworkType.DJANGO: ["django", "DJANGO_SETTINGS_MODULE"],
        FrameworkType.FLASK: ["from flask import", "Flask(__name__)"],
        FrameworkType.EXPRESS: ["express()", "require('express')"],
        FrameworkType.NEXTJS: ["next.config", "getServerSideProps", "getStaticProps"],
        FrameworkType.REACT: ["react-dom", "ReactDOM", "useState", "useEffect"],
        FrameworkType.VUE: ["vue.config", "createApp", "Vue.component"],
        FrameworkType.ANGULAR: ["@angular/core", "NgModule"],
        FrameworkType.RAILS: ["Rails.application", "ActiveRecord"],
        FrameworkType.SPRING: ["@SpringBootApplication", "springframework"],
        FrameworkType.DOTNET: ["Microsoft.AspNetCore", ".csproj"],
    }

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.state: Optional[ProjectState] = None

    async def analyze(self, deep_scan: bool = True) -> ProjectState:
        """
        Perform comprehensive project analysis.

        Args:
            deep_scan: If True, performs detailed analysis (slower but more thorough)
        """
        logger.info(f"Analyzing project at {self.project_path}")

        self.state = ProjectState(
            project_id=uuid.uuid4(),
            project_path=str(self.project_path),
            project_type=ProjectType.GREENFIELD,  # Will be updated
            analyzed_at=datetime.utcnow(),
            name=self.project_path.name,
        )

        # Run analysis tasks
        await asyncio.gather(
            self._analyze_structure(),
            self._detect_languages(),
            self._detect_framework(),
            self._analyze_git(),
            self._analyze_dependencies(),
        )

        if deep_scan:
            await asyncio.gather(
                self._analyze_code_quality(),
                self._detect_architecture(),
                self._find_technical_debt(),
                self._scan_security(),
            )

        # Determine project type based on analysis
        self._determine_project_type()

        # Generate agent context
        self._generate_agent_context()

        return self.state

    async def _analyze_structure(self):
        """Analyze directory structure."""
        structure = {}
        config_files = []
        entry_points = []

        common_configs = [
            "package.json", "pyproject.toml", "setup.py", "Cargo.toml",
            "go.mod", "pom.xml", "build.gradle", "Gemfile", "composer.json",
            ".env.example", "docker-compose.yml", "Dockerfile", "Makefile",
            "tsconfig.json", "webpack.config.js", "vite.config.js",
        ]

        common_entry_points = [
            "main.py", "app.py", "index.js", "index.ts", "main.go",
            "Main.java", "Program.cs", "main.rs", "application.rb",
        ]

        for root, dirs, files in os.walk(self.project_path):
            # Skip hidden and common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
                'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build',
                'target', 'vendor', '.git'
            ]]

            rel_root = os.path.relpath(root, self.project_path)
            if rel_root == '.':
                rel_root = ''

            for f in files:
                rel_path = os.path.join(rel_root, f) if rel_root else f

                if f in common_configs:
                    config_files.append(rel_path)
                if f in common_entry_points:
                    entry_points.append(rel_path)

                # Build structure (limit depth)
                parts = rel_path.split(os.sep)
                if len(parts) <= 4:
                    current = structure
                    for part in parts[:-1]:
                        current = current.setdefault(part, {})
                    current[parts[-1]] = None

        self.state.directory_structure = structure
        self.state.config_files = config_files
        self.state.entry_points = entry_points

    async def _detect_languages(self):
        """Detect programming languages used."""
        language_counts = {}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            count = 0
            for pattern in patterns:
                for f in self.project_path.rglob(pattern):
                    if self._should_include_file(f):
                        try:
                            count += sum(1 for _ in open(f, 'r', errors='ignore'))
                        except:
                            pass
            if count > 0:
                language_counts[lang] = count

        self.state.languages = language_counts
        if language_counts:
            self.state.primary_language = max(language_counts, key=language_counts.get)

    async def _detect_framework(self):
        """Detect the primary framework being used."""
        framework_scores = {f: 0 for f in FrameworkType}

        # Check config files
        for config in self.state.config_files:
            config_path = self.project_path / config
            if config_path.exists():
                try:
                    content = config_path.read_text(errors='ignore')
                    for framework, indicators in self.FRAMEWORK_INDICATORS.items():
                        for indicator in indicators:
                            if indicator.lower() in content.lower():
                                framework_scores[framework] += 1
                except:
                    pass

        # Check source files (sample)
        source_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.rb']
        sample_files = []
        for ext in source_extensions:
            sample_files.extend(list(self.project_path.rglob(f'*{ext}'))[:20])

        for f in sample_files:
            if self._should_include_file(f):
                try:
                    content = f.read_text(errors='ignore')[:5000]  # First 5KB
                    for framework, indicators in self.FRAMEWORK_INDICATORS.items():
                        for indicator in indicators:
                            if indicator in content:
                                framework_scores[framework] += 1
                except:
                    pass

        # Find highest scoring framework
        best_framework = max(framework_scores, key=framework_scores.get)
        if framework_scores[best_framework] > 0:
            self.state.framework = best_framework

    async def _analyze_git(self):
        """Analyze git repository information."""
        if not (self.project_path / '.git').exists():
            return

        try:
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_path, capture_output=True, text=True
            )
            current_branch = result.stdout.strip()

            # Get all branches
            result = subprocess.run(
                ['git', 'branch', '-a'],
                cwd=self.project_path, capture_output=True, text=True
            )
            branches = [b.strip().replace('* ', '') for b in result.stdout.split('\n') if b.strip()]

            # Get recent commits
            result = subprocess.run(
                ['git', 'log', '--oneline', '-20', '--format=%H|%s|%an|%ad'],
                cwd=self.project_path, capture_output=True, text=True
            )
            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        commits.append({
                            'hash': parts[0],
                            'message': parts[1],
                            'author': parts[2],
                            'date': parts[3],
                        })

            # Get contributors
            result = subprocess.run(
                ['git', 'shortlog', '-sn', '--all'],
                cwd=self.project_path, capture_output=True, text=True
            )
            contributors = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        contributors.append(parts[1])

            # Get uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_path, capture_output=True, text=True
            )
            uncommitted_changes = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

            self.state.git_info = {
                'current_branch': current_branch,
                'uncommitted_changes': uncommitted_changes,
                'total_commits': len(commits),
            }
            self.state.active_branches = branches[:20]  # Limit
            self.state.recent_commits = commits
            self.state.contributors = contributors[:20]  # Limit

        except Exception as e:
            logger.warning(f"Git analysis failed: {e}")

    async def _analyze_dependencies(self):
        """Analyze project dependencies."""
        deps = []
        dev_deps = []

        # Python dependencies
        req_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
        for req_file in req_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                content = req_path.read_text(errors='ignore')
                if req_file == 'pyproject.toml':
                    # Parse TOML dependencies
                    dep_match = re.findall(r'^\s*"?([a-zA-Z0-9_-]+)"?\s*[=<>~]', content, re.MULTILINE)
                    for dep in dep_match:
                        deps.append(DependencyHealth(name=dep, current_version="unknown"))
                else:
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                            if match:
                                is_dev = 'dev' in req_file.lower()
                                target = dev_deps if is_dev else deps
                                target.append(DependencyHealth(name=match.group(1), current_version="unknown"))

        # Node.js dependencies
        package_json = self.project_path / 'package.json'
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text())
                for name, version in pkg.get('dependencies', {}).items():
                    deps.append(DependencyHealth(name=name, current_version=version))
                for name, version in pkg.get('devDependencies', {}).items():
                    dev_deps.append(DependencyHealth(name=name, current_version=version))
            except:
                pass

        self.state.dependencies = deps
        self.state.dev_dependencies = dev_deps

    async def _analyze_code_quality(self):
        """Analyze code quality metrics."""
        metrics = CodeQualityMetrics()

        # Count files and lines
        for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb']:
            for f in self.project_path.rglob(f'*{ext}'):
                if self._should_include_file(f):
                    metrics.total_files += 1
                    try:
                        metrics.total_lines += sum(1 for _ in open(f, 'r', errors='ignore'))
                    except:
                        pass

        # Count test files
        test_patterns = ['test_*.py', '*_test.py', '*.test.js', '*.test.ts', '*.spec.js', '*.spec.ts']
        for pattern in test_patterns:
            metrics.test_files += len(list(self.project_path.rglob(pattern)))

        # Try to get coverage if available
        coverage_files = ['.coverage', 'coverage.xml', 'coverage/lcov.info']
        for cov_file in coverage_files:
            cov_path = self.project_path / cov_file
            if cov_path.exists():
                try:
                    content = cov_path.read_text(errors='ignore')
                    # Try to extract coverage percentage
                    match = re.search(r'line-rate="([0-9.]+)"', content)
                    if match:
                        metrics.test_coverage_percent = float(match.group(1)) * 100
                        break
                except:
                    pass

        self.state.quality_metrics = metrics

    async def _detect_architecture(self):
        """Detect architecture patterns."""
        patterns = []

        # Check for microservices indicators
        docker_compose = self.project_path / 'docker-compose.yml'
        if docker_compose.exists():
            content = docker_compose.read_text(errors='ignore')
            services = content.count('image:') + content.count('build:')
            if services > 3:
                patterns.append(ArchitecturePattern(
                    pattern_name="microservices",
                    confidence=0.7,
                    evidence=[f"Found {services} services in docker-compose.yml"]
                ))

        # Check for MVC pattern
        mvc_dirs = ['models', 'views', 'controllers', 'templates']
        mvc_found = sum(1 for d in mvc_dirs if (self.project_path / d).exists())
        if mvc_found >= 2:
            patterns.append(ArchitecturePattern(
                pattern_name="MVC",
                confidence=mvc_found / 4,
                evidence=[f"Found {mvc_found}/4 MVC directories"]
            ))

        # Check for clean architecture
        clean_dirs = ['domain', 'application', 'infrastructure', 'presentation']
        clean_found = sum(1 for d in clean_dirs if (self.project_path / d).exists())
        if clean_found >= 2:
            patterns.append(ArchitecturePattern(
                pattern_name="clean_architecture",
                confidence=clean_found / 4,
                evidence=[f"Found {clean_found}/4 clean architecture layers"]
            ))

        # Detect API endpoints (FastAPI/Flask)
        api_endpoints = []
        for py_file in self.project_path.rglob('*.py'):
            if self._should_include_file(py_file):
                try:
                    content = py_file.read_text(errors='ignore')
                    # FastAPI routes
                    routes = re.findall(r'@(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)', content)
                    for method, path in routes:
                        api_endpoints.append({'method': method.upper(), 'path': path, 'file': str(py_file.relative_to(self.project_path))})
                except:
                    pass

        self.state.architecture_patterns = patterns
        self.state.api_endpoints = api_endpoints[:100]  # Limit

    async def _find_technical_debt(self):
        """Identify technical debt items."""
        debt_items = []

        # Check for TODO/FIXME comments
        for ext in ['.py', '.js', '.ts', '.java', '.go']:
            for f in self.project_path.rglob(f'*{ext}'):
                if self._should_include_file(f):
                    try:
                        content = f.read_text(errors='ignore')
                        todos = re.findall(r'#\s*(TODO|FIXME|HACK|XXX)[\s:]+(.+)', content, re.IGNORECASE)
                        for tag, desc in todos[:5]:  # Limit per file
                            debt_items.append(TechnicalDebtItem(
                                category="code_quality",
                                severity="low" if tag.upper() == "TODO" else "medium",
                                description=desc.strip()[:100],
                                location=str(f.relative_to(self.project_path)),
                            ))
                    except:
                        pass

        # Check for outdated dependencies (basic check)
        if self.state.dependencies:
            old_deps = [d for d in self.state.dependencies if d.is_outdated]
            if old_deps:
                debt_items.append(TechnicalDebtItem(
                    category="dependency",
                    severity="medium",
                    description=f"{len(old_deps)} outdated dependencies",
                    suggested_fix="Run dependency update",
                ))

        # Check for missing tests
        if self.state.quality_metrics.test_files == 0 and self.state.quality_metrics.total_files > 5:
            debt_items.append(TechnicalDebtItem(
                category="testing",
                severity="high",
                description="No test files detected",
                suggested_fix="Add unit tests for core functionality",
                estimated_effort="days",
            ))

        self.state.technical_debt = debt_items[:50]  # Limit

    async def _scan_security(self):
        """Scan for security issues."""
        issues = []

        # Check for hardcoded secrets patterns
        secret_patterns = [
            r'(?i)(api[_-]?key|apikey|secret|password|passwd|token)\s*[=:]\s*["\'][^"\']{8,}["\']',
            r'(?i)aws[_-]?(access[_-]?key|secret)',
            r'-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----',
        ]

        for ext in ['.py', '.js', '.ts', '.java', '.env', '.yml', '.yaml', '.json']:
            for f in self.project_path.rglob(f'*{ext}'):
                if self._should_include_file(f) and '.example' not in f.name:
                    try:
                        content = f.read_text(errors='ignore')
                        for pattern in secret_patterns:
                            if re.search(pattern, content):
                                issues.append({
                                    'type': 'potential_secret',
                                    'severity': 'high',
                                    'file': str(f.relative_to(self.project_path)),
                                    'description': 'Potential hardcoded secret detected',
                                })
                                break  # One issue per file
                    except:
                        pass

        self.state.security_issues = issues[:20]  # Limit

    def _determine_project_type(self):
        """Determine the type of project based on analysis."""
        # Greenfield: Very few files, no git history, minimal structure
        if self.state.quality_metrics.total_files < 5:
            self.state.project_type = ProjectType.GREENFIELD
            return

        # Check git info
        git_info = self.state.git_info
        if git_info:
            # Feature branch: On a branch that's not main/master
            branch = git_info.get('current_branch', '')
            if branch and branch not in ['main', 'master', 'develop']:
                if 'feature' in branch.lower():
                    self.state.project_type = ProjectType.FEATURE_BRANCH
                    return
                if 'fix' in branch.lower() or 'bug' in branch.lower():
                    self.state.project_type = ProjectType.BUG_FIX
                    return
                if 'refactor' in branch.lower():
                    self.state.project_type = ProjectType.REFACTOR
                    return

        # Legacy: Old dependencies, lots of technical debt, no recent commits
        if len(self.state.technical_debt) > 20:
            self.state.project_type = ProjectType.LEGACY_MAINTENANCE
            return

        # Default: Existing active development
        self.state.project_type = ProjectType.EXISTING_ACTIVE

    def _generate_agent_context(self):
        """Generate context summaries for each agent type."""
        self.state.agent_context = {
            "business_analyst": {
                "project_summary": f"{'New' if self.state.project_type == ProjectType.GREENFIELD else 'Existing'} {self.state.primary_language or 'multi-language'} project using {self.state.framework.value}",
                "existing_requirements": self._find_requirements_docs(),
                "current_features": self._infer_features_from_code(),
            },
            "tech_lead": {
                "architecture": [p.pattern_name for p in self.state.architecture_patterns],
                "tech_stack": {
                    "language": self.state.primary_language,
                    "framework": self.state.framework.value,
                    "dependencies": len(self.state.dependencies),
                },
                "technical_debt_summary": self._summarize_tech_debt(),
                "api_endpoints_count": len(self.state.api_endpoints),
            },
            "backend_engineer": {
                "framework": self.state.framework.value,
                "api_endpoints": self.state.api_endpoints[:20],
                "database_models": self.state.database_models,
                "entry_points": self.state.entry_points,
            },
            "frontend_engineer": {
                "framework": self.state.framework.value if self.state.framework in [
                    FrameworkType.REACT, FrameworkType.VUE, FrameworkType.ANGULAR, FrameworkType.NEXTJS
                ] else None,
                "component_patterns": self._detect_component_patterns(),
            },
            "code_reviewer": {
                "quality_metrics": {
                    "total_files": self.state.quality_metrics.total_files,
                    "test_coverage": self.state.quality_metrics.test_coverage_percent,
                    "lint_errors": self.state.quality_metrics.lint_errors,
                },
                "common_issues": [d.description for d in self.state.technical_debt[:10]],
            },
            "security_reviewer": {
                "security_issues": self.state.security_issues,
                "sensitive_files": self._find_sensitive_files(),
                "auth_patterns": self._detect_auth_patterns(),
            },
        }

    def _should_include_file(self, path: Path) -> bool:
        """Check if file should be included in analysis."""
        exclude_patterns = [
            'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build',
            '.git', '.idea', '.vscode', 'target', 'vendor', '.mypy_cache',
            '.pytest_cache', 'coverage', '.tox', 'eggs', '*.egg-info'
        ]
        path_str = str(path)
        return not any(excl in path_str for excl in exclude_patterns)

    def _find_requirements_docs(self) -> list[str]:
        """Find existing requirements documentation."""
        req_files = []
        patterns = ['*requirements*.md', '*spec*.md', 'PRD*.md', 'README.md']
        for pattern in patterns:
            for f in self.project_path.rglob(pattern):
                if self._should_include_file(f):
                    req_files.append(str(f.relative_to(self.project_path)))
        return req_files[:10]

    def _infer_features_from_code(self) -> list[str]:
        """Infer features from API endpoints and code structure."""
        features = set()
        for endpoint in self.state.api_endpoints:
            path_parts = endpoint['path'].strip('/').split('/')
            if path_parts:
                features.add(path_parts[0])
        return list(features)[:20]

    def _summarize_tech_debt(self) -> dict:
        """Summarize technical debt by category."""
        summary = {}
        for item in self.state.technical_debt:
            cat = item.category
            summary[cat] = summary.get(cat, 0) + 1
        return summary

    def _detect_component_patterns(self) -> list[str]:
        """Detect frontend component patterns."""
        patterns = []
        components_dir = self.project_path / 'components'
        if components_dir.exists():
            patterns.append("components directory")
        src_components = self.project_path / 'src' / 'components'
        if src_components.exists():
            patterns.append("src/components structure")
        return patterns

    def _find_sensitive_files(self) -> list[str]:
        """Find potentially sensitive files."""
        sensitive = []
        patterns = ['.env', '*.pem', '*.key', 'credentials*', 'secrets*']
        for pattern in patterns:
            for f in self.project_path.rglob(pattern):
                if self._should_include_file(f):
                    sensitive.append(str(f.relative_to(self.project_path)))
        return sensitive[:20]

    def _detect_auth_patterns(self) -> list[str]:
        """Detect authentication patterns."""
        patterns = []
        auth_indicators = {
            'jwt': ['jwt', 'jsonwebtoken', 'PyJWT'],
            'oauth': ['oauth', 'OAuth2'],
            'session': ['session', 'cookie'],
            'api_key': ['api_key', 'x-api-key'],
        }
        for auth_type, indicators in auth_indicators.items():
            for dep in self.state.dependencies:
                if any(ind.lower() in dep.name.lower() for ind in indicators):
                    patterns.append(auth_type)
                    break
        return patterns

    async def save_analysis(self, db: Session) -> ProjectAnalysis:
        """Save analysis to database."""
        if not self.state:
            raise ValueError("Must run analyze() first")

        # Mark previous analyses as not current
        db.query(ProjectAnalysis).filter(
            ProjectAnalysis.project_path == str(self.project_path),
            ProjectAnalysis.is_current == True
        ).update({"is_current": False})

        # Create new analysis
        analysis = ProjectAnalysis(
            project_path=str(self.project_path),
            project_type=self.state.project_type.value,
            analysis_data=self._state_to_dict(),
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        # Record event
        event = Event(
            run_id=None,
            event_type="project_analyzed",
            payload={
                "project_path": str(self.project_path),
                "project_type": self.state.project_type.value,
                "analysis_id": str(analysis.id),
            }
        )
        db.add(event)
        db.commit()

        return analysis

    def _state_to_dict(self) -> dict:
        """Convert state to dictionary for JSON storage."""
        return {
            "project_id": str(self.state.project_id),
            "project_path": self.state.project_path,
            "project_type": self.state.project_type.value,
            "analyzed_at": self.state.analyzed_at.isoformat(),
            "name": self.state.name,
            "primary_language": self.state.primary_language,
            "languages": self.state.languages,
            "framework": self.state.framework.value,
            "directory_structure": self.state.directory_structure,
            "config_files": self.state.config_files,
            "entry_points": self.state.entry_points,
            "quality_metrics": {
                "total_files": self.state.quality_metrics.total_files,
                "total_lines": self.state.quality_metrics.total_lines,
                "test_files": self.state.quality_metrics.test_files,
                "test_coverage_percent": self.state.quality_metrics.test_coverage_percent,
            },
            "architecture_patterns": [
                {"name": p.pattern_name, "confidence": p.confidence}
                for p in self.state.architecture_patterns
            ],
            "api_endpoints": self.state.api_endpoints,
            "technical_debt_count": len(self.state.technical_debt),
            "security_issues_count": len(self.state.security_issues),
            "git_info": self.state.git_info,
            "agent_context": self.state.agent_context,
        }


# Singleton instance
_analyzer_cache: dict[str, ProjectAnalyzer] = {}


def get_project_analyzer(project_path: str) -> ProjectAnalyzer:
    """Get or create a project analyzer for the given path."""
    if project_path not in _analyzer_cache:
        _analyzer_cache[project_path] = ProjectAnalyzer(project_path)
    return _analyzer_cache[project_path]


async def analyze_project(project_path: str, db: Optional[Session] = None) -> ProjectState:
    """Convenience function to analyze a project."""
    analyzer = get_project_analyzer(project_path)
    state = await analyzer.analyze()
    if db:
        await analyzer.save_analysis(db)
    return state
