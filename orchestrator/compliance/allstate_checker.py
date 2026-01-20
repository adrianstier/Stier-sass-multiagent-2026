#!/usr/bin/env python3
"""
Allstate Compliance Checker - Static Analysis for ISSAS Compliance.

This module provides automated compliance checking for InsurTech applications
deployed within the Allstate ecosystem.

Standards covered:
- ISSAS (Information Security Standards for Allstate Suppliers)
- NIST SP 800-88 (Data Destruction)
- NIST AI RMF (AI Risk Management Framework)
- NAIC Model Bulletin (Insurance AI Governance)
- CPRA ADMT (Automated Decision-Making Technology)
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceCategory(Enum):
    ISSAS_DATA = "ISSAS/Data Destruction"
    ISSAS_CRYPTO = "ISSAS/Encryption"
    PRIVACY = "Privacy/PII"
    ACCESS_CONTROL = "Access Control"
    AI_GOVERNANCE = "AI/ADMT Governance"
    AGENCY_OPS = "Agency Operations"
    DATA_EXCHANGE = "Data Exchange/ACORD"


@dataclass
class ComplianceViolation:
    """Represents a single compliance violation."""
    code: str
    category: ComplianceCategory
    severity: Severity
    file_path: str
    line_number: int
    description: str
    remediation: str
    matched_text: str = ""
    standard_reference: str = ""


@dataclass
class ComplianceReport:
    """Full compliance report for a codebase."""
    total_files_scanned: int = 0
    violations: List[ComplianceViolation] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        return not any(v.severity in [Severity.CRITICAL, Severity.HIGH] for v in self.violations)

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_scanned": self.total_files_scanned,
            "is_compliant": self.is_compliant,
            "violation_counts": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": sum(1 for v in self.violations if v.severity == Severity.MEDIUM),
                "low": sum(1 for v in self.violations if v.severity == Severity.LOW),
            },
            "violations": [
                {
                    "code": v.code,
                    "category": v.category.value,
                    "severity": v.severity.value,
                    "file": v.file_path,
                    "line": v.line_number,
                    "description": v.description,
                    "remediation": v.remediation,
                    "standard": v.standard_reference,
                }
                for v in self.violations
            ],
            "passed_checks": self.passed_checks,
        }


# =============================================================================
# Compliance Rules
# =============================================================================

COMPLIANCE_RULES = [
    # SEC-01: Soft Delete Detection
    {
        "code": "SEC-01",
        "category": ComplianceCategory.ISSAS_DATA,
        "severity": Severity.CRITICAL,
        "pattern": r"(is_deleted\s*=|deleted_at|soft_delete|\.delete\(\s*soft\s*=|is_active\s*=\s*False)",
        "description": "Logical/Soft deletion of PII detected. ISSAS requires crypto-shredding.",
        "remediation": "Implement Crypto-Shredding: 1) Fetch tenant_id, 2) DELETE FROM tenant_keys WHERE id=?, 3) Verify key destruction.",
        "standard": "ISSAS Section 4: Data Destruction / NIST SP 800-88",
        "file_types": [".py", ".js", ".ts", ".java", ".rb", ".go"],
    },
    # SEC-02: Weak Encryption
    {
        "code": "SEC-02",
        "category": ComplianceCategory.ISSAS_CRYPTO,
        "severity": Severity.CRITICAL,
        "pattern": r"(AES[-_]?128|DES|3DES|RC4|RC2|MD5|SHA1\b|sha1\(|md5\(|hashlib\.md5|hashlib\.sha1)",
        "description": "Weak or deprecated encryption/hashing algorithm detected.",
        "remediation": "Use AES-256-GCM for encryption, SHA-256/SHA-384/SHA-512 for hashing. Ensure FIPS 140-2/3 validated modules.",
        "standard": "ISSAS Encryption Standards / FIPS 140-3",
        "file_types": [".py", ".js", ".ts", ".java", ".go", ".rs"],
    },
    # SEC-03: PII Logging
    {
        "code": "SEC-03",
        "category": ComplianceCategory.PRIVACY,
        "severity": Severity.HIGH,
        "pattern": r"(console\.log\s*\(\s*(user|customer|client|person|ssn|dob|email|phone|address)|logger\.(info|debug|warn|error)\s*\(.*(user|customer|password|ssn|email|phone|address)|print\s*\(\s*f?['\"].*\{(user|password|ssn|email))",
        "description": "Potential PII being logged. This violates privacy requirements.",
        "remediation": "Implement a PII Scrubber middleware. Replace PII values with <REDACTED> before writing to logs.",
        "standard": "ISSAS Privacy / CPRA",
        "file_types": [".py", ".js", ".ts", ".java"],
    },
    # SEC-04: Missing MFA
    {
        "code": "SEC-04",
        "category": ComplianceCategory.ACCESS_CONTROL,
        "severity": Severity.HIGH,
        "pattern": r"(mfa_skip|skip_mfa|bypass_mfa|mfa\s*=\s*False|trust_device.*>\s*12|rememberMe.*>\s*720)",
        "description": "MFA bypass or excessive device trust period detected.",
        "remediation": "Enforce MFA for all external access. Max device trust: 12 hours. if (!user.mfa_verified) return redirect('/mfa-challenge');",
        "standard": "ISSAS Access Control",
        "file_types": [".py", ".js", ".ts", ".java", ".go"],
    },
    # AI-01: Data Shadowing
    {
        "code": "AI-01",
        "category": ComplianceCategory.AI_GOVERNANCE,
        "severity": Severity.CRITICAL,
        "pattern": r"(chat_history|prompt_log|save.*prompt|store.*completion|INSERT.*INTO.*(chat|prompt|message)|\.create\(.*prompt.*\)|conversation_history\.append)",
        "description": "LLM prompts/responses being persisted to database. Violates Zero Data Retention.",
        "remediation": "Use Ephemeral Storage (Redis/RAM) only. Do not persist chat logs to disk. Set store=False in API calls.",
        "standard": "NAIC Model Bulletin / CPRA ADMT / ZDR Protocol",
        "file_types": [".py", ".js", ".ts"],
    },
    # AI-02: Automated Decisioning
    {
        "code": "AI-02",
        "category": ComplianceCategory.AI_GOVERNANCE,
        "severity": Severity.CRITICAL,
        "pattern": r"(decision\s*=\s*['\"]?(deny|approve|reject|decline)|auto_underwrite|auto_decision|\.update\(.*status\s*=\s*['\"]?(denied|approved))",
        "description": "AI making automated consumer outcome decisions without human review.",
        "remediation": "Wrap output in review_required object. status: 'pending_review' until human action for Underwriting/Claims/Pricing decisions.",
        "standard": "NAIC Model Bulletin / CPRA ADMT / Human-in-the-Loop",
        "file_types": [".py", ".js", ".ts"],
    },
    # INT-01: Blocked Installer
    {
        "code": "INT-01",
        "category": ComplianceCategory.AGENCY_OPS,
        "severity": Severity.HIGH,
        "pattern": r"(electron-builder|\.exe['\"]|\.msi['\"]|nsis|pkg\.targets|electronPackagerConfig|electron-packager)",
        "description": "Desktop installer/packaging detected. Exclusive Agents cannot install software on Allstate RASC computers.",
        "remediation": "Switch to PWA (Progressive Web App). Remove Electron/installer dependencies. Use web-only architecture.",
        "standard": "Allstate Agency Ops / Exclusive Agent Policy",
        "file_types": [".json", ".js", ".ts", ".yml", ".yaml", ".xml"],
    },
    # INT-02: Non-ACORD Export
    {
        "code": "INT-02",
        "category": ComplianceCategory.DATA_EXCHANGE,
        "severity": Severity.MEDIUM,
        "pattern": r"(export.*\.csv|download.*csv|to_csv\(|writeFile.*\.csv|\.json.*export|exportData.*json)",
        "description": "CSV/JSON export for policy data detected. Independent Agents require ACORD AL3/XML.",
        "remediation": "Implement ACORD AL3 Parser/Generator for batch operations, ACORD XML for real-time Ivans integration.",
        "standard": "ACORD Standards / Ivans Integration",
        "file_types": [".py", ".js", ".ts", ".java"],
    },
    # Additional: Hardcoded Secrets
    {
        "code": "SEC-05",
        "category": ComplianceCategory.PRIVACY,
        "severity": Severity.CRITICAL,
        "pattern": r"(api_key\s*=\s*['\"][a-zA-Z0-9]{20,}|password\s*=\s*['\"][^'\"]+['\"]|secret\s*=\s*['\"][^'\"]+['\"]|ANTHROPIC_API_KEY\s*=\s*['\"]sk-)",
        "description": "Hardcoded secret/API key detected in source code.",
        "remediation": "Move secrets to environment variables or a secrets manager. Never commit secrets to version control.",
        "standard": "ISSAS / OWASP",
        "file_types": [".py", ".js", ".ts", ".java", ".go", ".env"],
    },
    # Session Timeout
    {
        "code": "SEC-06",
        "category": ComplianceCategory.ACCESS_CONTROL,
        "severity": Severity.MEDIUM,
        "pattern": r"(session.*timeout.*>\s*900|maxAge.*>\s*900000|SESSION_TIMEOUT.*=\s*[0-9]{4,})",
        "description": "Session timeout exceeds 15 minutes (900 seconds).",
        "remediation": "Set session absolute timeout to 15 minutes maximum for ISSAS compliance.",
        "standard": "ISSAS Access Control",
        "file_types": [".py", ".js", ".ts", ".java", ".yml", ".yaml", ".json"],
    },
]


class AllstateComplianceChecker:
    """Static analysis checker for Allstate/ISSAS compliance."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.report = ComplianceReport()

    def scan(self, exclude_dirs: Optional[List[str]] = None) -> ComplianceReport:
        """Scan the codebase for compliance violations."""
        exclude_dirs = exclude_dirs or [
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".nuxt", "coverage"
        ]

        files_scanned = 0

        for root, dirs, files in os.walk(self.root_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                # Check each rule against applicable files
                for rule in COMPLIANCE_RULES:
                    if ext in rule["file_types"]:
                        try:
                            self._check_file(file_path, rule)
                            files_scanned += 1
                        except Exception:
                            pass  # Skip files that can't be read

        self.report.total_files_scanned = files_scanned
        self._add_passed_checks()

        return self.report

    def _check_file(self, file_path: Path, rule: dict) -> None:
        """Check a single file against a rule."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        pattern = re.compile(rule["pattern"], re.IGNORECASE | re.MULTILINE)

        for line_num, line in enumerate(content.split("\n"), 1):
            match = pattern.search(line)
            if match:
                violation = ComplianceViolation(
                    code=rule["code"],
                    category=rule["category"],
                    severity=rule["severity"],
                    file_path=str(file_path.relative_to(self.root_path)),
                    line_number=line_num,
                    description=rule["description"],
                    remediation=rule["remediation"],
                    matched_text=match.group(0)[:50],
                    standard_reference=rule.get("standard", ""),
                )
                self.report.violations.append(violation)

    def _add_passed_checks(self) -> None:
        """Add passed checks based on what wasn't found."""
        violation_codes = {v.code for v in self.report.violations}

        all_codes = {r["code"] for r in COMPLIANCE_RULES}
        passed_codes = all_codes - violation_codes

        code_to_name = {
            "SEC-01": "Data Destruction (Crypto-Shredding)",
            "SEC-02": "Encryption Standards (AES-256-GCM)",
            "SEC-03": "PII Logging Protection",
            "SEC-04": "MFA Enforcement",
            "SEC-05": "Secret Management",
            "SEC-06": "Session Timeout Policy",
            "AI-01": "Zero Data Retention (ZDR)",
            "AI-02": "Human-in-the-Loop for Decisions",
            "INT-01": "Web-Native Architecture",
            "INT-02": "ACORD Data Exchange",
        }

        for code in sorted(passed_codes):
            name = code_to_name.get(code, code)
            self.report.passed_checks.append(f"{code}: {name} - No violations found")


def check_compliance(path: str, exclude_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run compliance check on a directory.

    Args:
        path: Path to the directory to scan
        exclude_dirs: Directories to exclude from scanning

    Returns:
        Compliance report as a dictionary
    """
    checker = AllstateComplianceChecker(path)
    report = checker.scan(exclude_dirs)
    return report.to_dict()


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python allstate_checker.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    result = check_compliance(path)
    print(json.dumps(result, indent=2))
