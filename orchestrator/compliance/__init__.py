"""Allstate Compliance Module for InsurTech applications."""

from .allstate_checker import (
    AllstateComplianceChecker,
    ComplianceReport,
    ComplianceViolation,
    Severity,
    ComplianceCategory,
    check_compliance,
    COMPLIANCE_RULES,
)

__all__ = [
    "AllstateComplianceChecker",
    "ComplianceReport",
    "ComplianceViolation",
    "Severity",
    "ComplianceCategory",
    "check_compliance",
    "COMPLIANCE_RULES",
]
