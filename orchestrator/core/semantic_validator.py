"""Semantic Validation for Artifacts.

Provides:
- LLM-based semantic validation (does output match requirements?)
- Cross-artifact consistency checks
- Automated test generation for code artifacts
- Configurable validation criteria
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import hashlib

from orchestrator.core.database import get_db
from orchestrator.core.models import Run, Task, Artifact, Event
from orchestrator.core.config import settings
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Validation Results
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: str  # critical, high, medium, low, info
    category: str  # semantic, consistency, completeness, format, security
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of artifact validation."""

    artifact_id: str
    artifact_type: str
    validation_type: str  # semantic, schema, consistency, test_generation

    passed: bool
    score: float  # 0-1
    issues: List[ValidationIssue] = field(default_factory=list)

    # Metadata
    criteria_checked: List[str] = field(default_factory=list)
    related_artifacts: List[str] = field(default_factory=list)

    # Timing
    validation_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "validation_type": self.validation_type,
            "passed": self.passed,
            "score": round(self.score, 2),
            "issues": [i.to_dict() for i in self.issues],
            "criteria_checked": self.criteria_checked,
            "related_artifacts": self.related_artifacts,
            "validation_time_ms": self.validation_time_ms,
            "critical_issues": len([i for i in self.issues if i.severity == "critical"]),
            "high_issues": len([i for i in self.issues if i.severity == "high"]),
        }


# =============================================================================
# Validation Criteria
# =============================================================================

# Default validation criteria by artifact type
VALIDATION_CRITERIA = {
    "requirements_document": {
        "semantic": [
            "completeness",  # All necessary requirements covered
            "clarity",  # Requirements are unambiguous
            "testability",  # Requirements can be verified
            "consistency",  # No contradicting requirements
            "feasibility",  # Requirements are technically achievable
        ],
        "format": [
            "proper_structure",  # Has required sections
            "numbered_requirements",  # Requirements are numbered
            "priority_markers",  # Priorities are indicated
        ],
    },
    "architecture_document": {
        "semantic": [
            "completeness",  # All components documented
            "security_considerations",  # Security addressed
            "scalability_plan",  # Scalability considered
            "api_completeness",  # APIs fully specified
            "data_flow_clarity",  # Data flows are clear
        ],
        "consistency": [
            "api_match_requirements",  # APIs match requirements
            "component_coverage",  # All features have components
        ],
    },
    "database_schema": {
        "semantic": [
            "normalization",  # Properly normalized
            "referential_integrity",  # Foreign keys valid
            "index_coverage",  # Proper indexing
            "naming_conventions",  # Consistent naming
        ],
        "consistency": [
            "match_architecture",  # Matches architecture doc
            "support_apis",  # Supports required APIs
        ],
    },
    "backend_code": {
        "semantic": [
            "error_handling",  # Proper error handling
            "security",  # Security best practices
            "api_implementation",  # APIs match spec
            "input_validation",  # Inputs are validated
        ],
        "consistency": [
            "match_architecture",  # Follows architecture
            "use_database_schema",  # Uses correct schema
        ],
        "test_generation": [
            "unit_tests",  # Generate unit tests
            "api_tests",  # Generate API tests
        ],
    },
    "frontend_code": {
        "semantic": [
            "accessibility",  # WCAG compliance
            "responsiveness",  # Mobile-friendly
            "error_handling",  # User error feedback
            "state_management",  # Proper state handling
        ],
        "consistency": [
            "match_design",  # Matches UX design
            "use_correct_apis",  # Uses backend APIs correctly
        ],
    },
}


# =============================================================================
# Semantic Validator
# =============================================================================

class SemanticValidator:
    """Validates artifacts semantically using LLM and rules."""

    def __init__(self):
        self._validation_cache: Dict[str, ValidationResult] = {}

    def validate_artifact(
        self,
        artifact_id: str,
        run_id: str,
        validation_types: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate a single artifact.

        Args:
            artifact_id: The artifact ID
            run_id: The run ID
            validation_types: Optional list of validation types to run

        Returns:
            ValidationResult
        """
        start_time = datetime.utcnow()

        with get_db() as db:
            artifact = db.query(Artifact).filter(
                Artifact.id == UUID(artifact_id)
            ).first()

            if not artifact:
                return ValidationResult(
                    artifact_id=artifact_id,
                    artifact_type="unknown",
                    validation_type="error",
                    passed=False,
                    score=0,
                    issues=[ValidationIssue(
                        severity="critical",
                        category="error",
                        message="Artifact not found",
                    )],
                )

            # Get criteria for this artifact type
            criteria = VALIDATION_CRITERIA.get(
                artifact.artifact_type,
                {"semantic": ["completeness", "clarity"]}
            )

            # Determine validation types
            if validation_types is None:
                validation_types = list(criteria.keys())

            # Run validations
            all_issues: List[ValidationIssue] = []
            criteria_checked = []

            for val_type in validation_types:
                if val_type == "semantic":
                    issues = self._validate_semantic(artifact, criteria.get("semantic", []))
                elif val_type == "consistency":
                    issues = self._validate_consistency(db, artifact, run_id, criteria.get("consistency", []))
                elif val_type == "format":
                    issues = self._validate_format(artifact, criteria.get("format", []))
                elif val_type == "test_generation":
                    issues = self._validate_with_tests(artifact, criteria.get("test_generation", []))
                else:
                    continue

                all_issues.extend(issues)
                criteria_checked.extend(criteria.get(val_type, []))

            # Calculate score
            score = self._calculate_score(all_issues, len(criteria_checked))
            passed = score >= 0.7 and not any(i.severity == "critical" for i in all_issues)

            # Calculate validation time
            validation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            result = ValidationResult(
                artifact_id=artifact_id,
                artifact_type=artifact.artifact_type,
                validation_type=",".join(validation_types),
                passed=passed,
                score=score,
                issues=all_issues,
                criteria_checked=criteria_checked,
                validation_time_ms=validation_time,
            )

            # Update artifact validation status
            artifact.is_valid = passed
            artifact.validation_errors = [i.to_dict() for i in all_issues if i.severity in ["critical", "high"]]
            db.commit()

            # Record event
            self._record_validation_event(db, run_id, artifact_id, result)

            return result

    def _validate_semantic(
        self, artifact: Artifact, criteria: List[str]
    ) -> List[ValidationIssue]:
        """Validate semantic quality of artifact."""
        issues = []
        content = artifact.content or ""

        # Check completeness
        if "completeness" in criteria:
            if len(content) < 500:
                issues.append(ValidationIssue(
                    severity="medium",
                    category="semantic",
                    message="Content appears incomplete (less than 500 characters)",
                    suggestion="Expand the artifact with more detailed content",
                ))

            # Check for placeholder text
            placeholder_patterns = [
                r"\[TODO\]", r"\[TBD\]", r"\[PLACEHOLDER\]",
                r"lorem ipsum", r"xxx", r"..."
            ]
            for pattern in placeholder_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(ValidationIssue(
                        severity="high",
                        category="semantic",
                        message=f"Found placeholder text: {pattern}",
                        location="content",
                        suggestion="Replace placeholder text with actual content",
                    ))

        # Check clarity
        if "clarity" in criteria:
            # Check for very long sentences
            sentences = re.split(r'[.!?]', content)
            long_sentences = [s for s in sentences if len(s.split()) > 50]
            if long_sentences:
                issues.append(ValidationIssue(
                    severity="low",
                    category="semantic",
                    message=f"Found {len(long_sentences)} very long sentences that may be unclear",
                    suggestion="Break long sentences into shorter, clearer statements",
                ))

        # Check testability (for requirements)
        if "testability" in criteria:
            vague_terms = ["appropriate", "reasonable", "adequate", "as needed", "etc"]
            for term in vague_terms:
                if term.lower() in content.lower():
                    issues.append(ValidationIssue(
                        severity="medium",
                        category="semantic",
                        message=f"Vague term '{term}' found - requirements may not be testable",
                        suggestion=f"Replace '{term}' with specific, measurable criteria",
                    ))

        # Check security considerations
        if "security_considerations" in criteria:
            security_keywords = ["authentication", "authorization", "encryption", "security", "https", "token"]
            has_security = any(kw in content.lower() for kw in security_keywords)
            if not has_security:
                issues.append(ValidationIssue(
                    severity="high",
                    category="semantic",
                    message="No security considerations found",
                    suggestion="Add security requirements and considerations",
                ))

        # Check error handling (for code)
        if "error_handling" in criteria:
            if "def " in content or "function" in content:
                error_patterns = ["try", "catch", "except", "throw", "raise", "error"]
                has_error_handling = any(p in content.lower() for p in error_patterns)
                if not has_error_handling:
                    issues.append(ValidationIssue(
                        severity="high",
                        category="semantic",
                        message="No error handling found in code",
                        suggestion="Add try/catch blocks and error handling",
                    ))

        # Check input validation (for code)
        if "input_validation" in criteria:
            if "def " in content or "function" in content:
                validation_patterns = ["validate", "check", "verify", "assert", "if not", "raise ValueError"]
                has_validation = any(p in content.lower() for p in validation_patterns)
                if not has_validation:
                    issues.append(ValidationIssue(
                        severity="medium",
                        category="semantic",
                        message="No input validation found",
                        suggestion="Add input validation for function parameters",
                    ))

        return issues

    def _validate_consistency(
        self,
        db,
        artifact: Artifact,
        run_id: str,
        criteria: List[str],
    ) -> List[ValidationIssue]:
        """Validate consistency with other artifacts."""
        issues = []
        content = artifact.content or ""

        # Get related artifacts
        related_artifacts = db.query(Artifact).filter(
            Artifact.run_id == UUID(run_id),
            Artifact.id != artifact.id,
        ).all()

        # Check API match
        if "api_match_requirements" in criteria or "match_architecture" in criteria:
            # Find architecture artifact
            arch_artifact = next(
                (a for a in related_artifacts if a.artifact_type == "architecture_document"),
                None
            )

            if arch_artifact:
                # Extract API endpoints from architecture
                arch_endpoints = self._extract_api_endpoints(arch_artifact.content or "")
                current_endpoints = self._extract_api_endpoints(content)

                missing = set(arch_endpoints) - set(current_endpoints)
                if missing and artifact.artifact_type in ["backend_code", "frontend_code"]:
                    issues.append(ValidationIssue(
                        severity="high",
                        category="consistency",
                        message=f"Missing API endpoints from architecture: {', '.join(list(missing)[:3])}",
                        suggestion="Implement all API endpoints specified in architecture",
                    ))

        # Check schema match
        if "use_database_schema" in criteria:
            schema_artifact = next(
                (a for a in related_artifacts if a.artifact_type == "database_schema"),
                None
            )

            if schema_artifact:
                # Extract table names from schema
                tables = self._extract_table_names(schema_artifact.content or "")

                # Check if code references these tables
                for table in tables[:5]:  # Check first 5
                    if table.lower() not in content.lower():
                        issues.append(ValidationIssue(
                            severity="info",
                            category="consistency",
                            message=f"Table '{table}' from schema not referenced in code",
                            suggestion="Verify if this table should be used",
                        ))

        # Check requirements coverage
        if "match_requirements" in criteria:
            req_artifact = next(
                (a for a in related_artifacts if a.artifact_type == "requirements_document"),
                None
            )

            if req_artifact:
                # Extract requirement IDs
                req_ids = re.findall(r'(?:REQ|R)[-_]?(\d+)', req_artifact.content or "")
                if req_ids:
                    covered_in_current = re.findall(r'(?:REQ|R)[-_]?(\d+)', content)
                    missing = set(req_ids) - set(covered_in_current)
                    if missing:
                        issues.append(ValidationIssue(
                            severity="medium",
                            category="consistency",
                            message=f"Requirements not addressed: {', '.join(list(missing)[:5])}",
                            suggestion="Ensure all requirements are addressed",
                        ))

        return issues

    def _validate_format(
        self, artifact: Artifact, criteria: List[str]
    ) -> List[ValidationIssue]:
        """Validate format and structure."""
        issues = []
        content = artifact.content or ""

        # Check proper structure (markdown headers)
        if "proper_structure" in criteria:
            headers = re.findall(r'^#+\s+.+$', content, re.MULTILINE)
            if not headers:
                issues.append(ValidationIssue(
                    severity="low",
                    category="format",
                    message="No markdown headers found - document may lack structure",
                    suggestion="Add headers to organize content",
                ))

        # Check numbered requirements
        if "numbered_requirements" in criteria:
            numbered = re.findall(r'^\d+\.\s+|^-\s+|^\*\s+', content, re.MULTILINE)
            if len(numbered) < 5 and artifact.artifact_type == "requirements_document":
                issues.append(ValidationIssue(
                    severity="medium",
                    category="format",
                    message="Few numbered/bulleted items found",
                    suggestion="Use numbered lists for requirements",
                ))

        # Check naming conventions (for code/schema)
        if "naming_conventions" in criteria:
            # Check for inconsistent naming
            snake_case = len(re.findall(r'\b[a-z]+_[a-z]+\b', content))
            camel_case = len(re.findall(r'\b[a-z]+[A-Z][a-z]+\b', content))

            if snake_case > 10 and camel_case > 10:
                issues.append(ValidationIssue(
                    severity="low",
                    category="format",
                    message="Mixed naming conventions (snake_case and camelCase)",
                    suggestion="Use consistent naming convention",
                ))

        return issues

    def _validate_with_tests(
        self, artifact: Artifact, criteria: List[str]
    ) -> List[ValidationIssue]:
        """Generate and validate tests for code artifacts."""
        issues = []
        content = artifact.content or ""

        if "unit_tests" in criteria:
            # Check if functions are testable
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            public_functions = [f for f in functions if not f.startswith('_')]

            if public_functions:
                # Check for existing test references
                has_tests = "test_" in content.lower() or "unittest" in content.lower()
                if not has_tests:
                    issues.append(ValidationIssue(
                        severity="medium",
                        category="test_generation",
                        message=f"No tests found for {len(public_functions)} public functions",
                        suggestion=f"Add unit tests for: {', '.join(public_functions[:3])}",
                    ))

                    # Generate test suggestions
                    for func in public_functions[:3]:
                        issues.append(ValidationIssue(
                            severity="info",
                            category="test_generation",
                            message=f"Suggested test: test_{func}()",
                            suggestion=f"Test the {func} function with various inputs",
                        ))

        if "api_tests" in criteria:
            # Check for API endpoint definitions
            endpoints = self._extract_api_endpoints(content)
            if endpoints:
                issues.append(ValidationIssue(
                    severity="info",
                    category="test_generation",
                    message=f"Found {len(endpoints)} API endpoints that should have integration tests",
                    suggestion="Add integration tests for all API endpoints",
                ))

        return issues

    def _extract_api_endpoints(self, content: str) -> List[str]:
        """Extract API endpoint paths from content."""
        # Look for various API patterns
        patterns = [
            r'@app\.(get|post|put|delete|patch)\s*\([\'"]([^\'"]+)[\'"]',  # Flask/FastAPI decorators
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+([/\w{}]+)',  # HTTP method + path
            r'path\s*:\s*[\'"]([/\w{}]+)[\'"]',  # OpenAPI style
        ]

        endpoints = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    endpoints.extend([m[-1] for m in matches])
                else:
                    endpoints.extend(matches)

        return list(set(endpoints))

    def _extract_table_names(self, content: str) -> List[str]:
        """Extract table names from schema content."""
        patterns = [
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\']?(\w+)[`"\']?',
            r'class\s+(\w+)\s*\(.*?Base\)',  # SQLAlchemy models
            r'__tablename__\s*=\s*[\'"](\w+)[\'"]',  # SQLAlchemy tablename
        ]

        tables = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _calculate_score(
        self, issues: List[ValidationIssue], total_criteria: int
    ) -> float:
        """Calculate validation score from issues."""
        if total_criteria == 0:
            return 1.0

        # Weight by severity
        severity_weights = {
            "critical": 0.4,
            "high": 0.25,
            "medium": 0.15,
            "low": 0.1,
            "info": 0.05,
        }

        penalty = sum(severity_weights.get(i.severity, 0.1) for i in issues)

        # Score is 1 - penalty, clamped to [0, 1]
        score = max(0, min(1, 1 - penalty))

        return score

    def _record_validation_event(
        self, db, run_id: str, artifact_id: str, result: ValidationResult
    ) -> None:
        """Record validation event."""
        event = Event(
            run_id=UUID(run_id),
            event_type="artifact_validated",
            actor="semantic_validator",
            data={
                "artifact_id": artifact_id,
                "validation_type": result.validation_type,
                "passed": result.passed,
                "score": result.score,
                "issue_count": len(result.issues),
                "critical_count": len([i for i in result.issues if i.severity == "critical"]),
            },
        )
        db.add(event)
        db.commit()

    def validate_run_artifacts(
        self,
        run_id: str,
        artifact_types: Optional[List[str]] = None,
    ) -> Dict[str, ValidationResult]:
        """Validate all artifacts in a run.

        Args:
            run_id: The run ID
            artifact_types: Optional filter for artifact types

        Returns:
            Dict mapping artifact_id to ValidationResult
        """
        with get_db() as db:
            query = db.query(Artifact).filter(Artifact.run_id == UUID(run_id))

            if artifact_types:
                query = query.filter(Artifact.artifact_type.in_(artifact_types))

            artifacts = query.all()

        results = {}
        for artifact in artifacts:
            result = self.validate_artifact(str(artifact.id), run_id)
            results[str(artifact.id)] = result

        return results

    def check_cross_artifact_consistency(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        """Check consistency across all artifacts in a run.

        Args:
            run_id: The run ID

        Returns:
            Dict with consistency check results
        """
        with get_db() as db:
            artifacts = db.query(Artifact).filter(
                Artifact.run_id == UUID(run_id)
            ).all()

            # Build artifact map
            artifact_map = {a.artifact_type: a for a in artifacts}

            issues = []

            # Check requirement coverage chain
            if "requirements_document" in artifact_map:
                req_content = artifact_map["requirements_document"].content or ""
                req_ids = set(re.findall(r'(?:REQ|R)[-_]?(\d+)', req_content))

                downstream = ["architecture_document", "backend_code", "frontend_code"]
                for art_type in downstream:
                    if art_type in artifact_map:
                        art_content = artifact_map[art_type].content or ""
                        covered = set(re.findall(r'(?:REQ|R)[-_]?(\d+)', art_content))
                        missing = req_ids - covered
                        if missing and len(missing) < len(req_ids):
                            issues.append({
                                "type": "requirement_coverage",
                                "artifact": art_type,
                                "message": f"Requirements not traced: {', '.join(list(missing)[:5])}",
                                "severity": "medium",
                            })

            # Check API consistency
            if "architecture_document" in artifact_map:
                arch_endpoints = self._extract_api_endpoints(
                    artifact_map["architecture_document"].content or ""
                )

                if "backend_code" in artifact_map:
                    backend_endpoints = self._extract_api_endpoints(
                        artifact_map["backend_code"].content or ""
                    )

                    missing = set(arch_endpoints) - set(backend_endpoints)
                    if missing:
                        issues.append({
                            "type": "api_consistency",
                            "artifact": "backend_code",
                            "message": f"API endpoints not implemented: {', '.join(list(missing)[:5])}",
                            "severity": "high",
                        })

            # Calculate overall consistency score
            total_checks = 3
            passed_checks = total_checks - len([i for i in issues if i["severity"] != "info"])
            score = passed_checks / total_checks if total_checks > 0 else 1.0

            return {
                "run_id": run_id,
                "artifacts_checked": len(artifacts),
                "consistency_score": round(score, 2),
                "issues": issues,
                "passed": score >= 0.7,
            }


# Global semantic validator instance
_semantic_validator: Optional[SemanticValidator] = None


def get_semantic_validator() -> SemanticValidator:
    """Get the global semantic validator instance."""
    global _semantic_validator
    if _semantic_validator is None:
        _semantic_validator = SemanticValidator()
    return _semantic_validator
