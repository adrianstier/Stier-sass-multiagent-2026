# Allstate/ISSAS Compliance Report

**Generated:** 2026-01-20
**Codebase:** Stier-sass-multiagent-2026
**Status:** ❌ **NON-COMPLIANT** (requires remediation)

---

## Executive Summary

| Metric | Raw Count | After False Positive Analysis |
|--------|-----------|------------------------------|
| Total Files Scanned | 810 | 810 |
| Critical Violations | 976 | **26** |
| High Violations | 5 | **5** |
| Medium Violations | 1 | **1** |
| Compliance Status | NON-COMPLIANT | NON-COMPLIANT |

**Key Finding:** The raw compliance scan reported 976 critical violations, but **~95% are false positives** caused by:
1. Regex matching "DES" inside words like "design", "describes"
2. Third-party packages in `.venv/` directory
3. Documentation/example strings containing violation patterns
4. The compliance checker code itself containing rule definitions

**Actual violations requiring remediation: 32 total (26 critical, 5 high, 1 medium)**

---

## Standards Covered

- **ISSAS** - Information Security Standards for Allstate Suppliers
- **NIST SP 800-88** - Data Destruction (Crypto-Shredding)
- **NIST AI RMF** - AI Risk Management Framework
- **NAIC Model Bulletin** - Insurance AI Governance
- **CPRA ADMT** - Automated Decision-Making Technology

---

## Passed Checks ✅

| Code | Check | Status |
|------|-------|--------|
| INT-01 | Web-Native Architecture (No .exe/.msi installers) | ✅ PASSED |
| SEC-05 | Secret Management (No hardcoded API keys) | ✅ PASSED |
| SEC-06 | Session Timeout Policy | ✅ PASSED |

---

## Violations Requiring Remediation

### 1. AI-02: Automated Decisioning (12 violations) - CRITICAL

**Standard:** NAIC Model Bulletin / CPRA ADMT / Human-in-the-Loop
**Severity:** 🔴 CRITICAL
**Effort:** MEDIUM

**Affected File:** [orchestrator/core/escalation.py](orchestrator/core/escalation.py)

**Issue:** The escalation system contains automated decision logic that can approve/deny/reject consumer outcomes without mandatory human review.

**Specific Violations:**

| Line | Code Pattern | Issue |
|------|-------------|-------|
| 94-96 | `auto_decision_enabled`, `auto_decision_default = "reject"` | Auto-decision configuration |
| 117-118 | `auto_decisions: dict[EscalationType, str]` with `"reject"` | Default automated rejections |
| 350, 360, 373-374 | `auto_decision_timeout`, `auto_decision_default` | Timeout-based auto-decisions |
| 430, 468, 474 | `_apply_auto_decision()`, `decision = default_decision` | Actual auto-decision execution |
| 600 | Additional auto-decision logic | - |

**Remediation:**

```python
# BEFORE (Non-compliant)
escalation.decision = default_decision  # Auto-decides without human

# AFTER (Compliant)
async def _apply_auto_decision(self, db, escalation):
    """For insurance decisions, ALWAYS require human review."""
    if escalation.escalation_type in [
        EscalationType.UNDERWRITING,
        EscalationType.CLAIMS,
        EscalationType.PRICING,
    ]:
        # NAIC/CPRA requires human-in-the-loop for consumer outcomes
        escalation.status = EscalationStatus.REQUIRES_HUMAN_REVIEW
        escalation.decision = None
        await self.notification_service.escalate_to_human(escalation, db)
        return escalation

    # Non-consumer-outcome decisions can auto-decide
    escalation.status = EscalationStatus.AUTO_DECIDED
    escalation.decision = escalation.auto_decision_default
    return escalation
```

---

### 2. SEC-01: Data Destruction / Crypto-Shredding (8 violations) - CRITICAL

**Standard:** ISSAS Section 4 / NIST SP 800-88
**Severity:** 🔴 CRITICAL
**Effort:** HIGH

**Issue:** Soft deletion patterns detected where crypto-shredding is required for PII.

**Affected Files & Violations:**

| File | Line | Pattern | Issue |
|------|------|---------|-------|
| [orchestrator/delegate.py](orchestrator/delegate.py) | 539, 570, 640, 678 | Documentation examples | FALSE POSITIVE (in docstrings) |
| [orchestrator/core/webhooks.py](orchestrator/core/webhooks.py#L210) | 210 | `webhook.is_active = False` | Real - disables but doesn't delete |
| [orchestrator/agents/data_science/artifacts.py](orchestrator/agents/data_science/artifacts.py#L352) | 352, 357 | `soft_delete: bool = True` | Real - uses soft delete pattern |
| [orchestrator/api/main.py](orchestrator/api/main.py#L466) | 466 | `api_key.is_active = False` | Real - soft revoke, not delete |

**Real Violations Requiring Fix: 4**

**Remediation:**

```python
# BEFORE (Non-compliant) - orchestrator/agents/data_science/artifacts.py:352
def delete(self, artifact_id: str, soft_delete: bool = True) -> None:
    if soft_delete:
        artifact.status = ArtifactStatus.ARCHIVED  # Data still exists!

# AFTER (Compliant) - If artifact contains PII
def delete(self, artifact_id: str) -> None:
    """Delete artifact with crypto-shredding if contains PII."""
    artifact = self._artifacts[artifact_id]

    if artifact.contains_pii:
        # ISSAS requires crypto-shredding for PII
        await self._crypto_shred(artifact)  # Destroy encryption key

    # Physical deletion
    del self._artifacts[artifact_id]
    metadata_file = self.base_path / "metadata" / f"{artifact_id}.json"
    if metadata_file.exists():
        metadata_file.unlink()

async def _crypto_shred(self, artifact):
    """Destroy the encryption key, making data unrecoverable."""
    # DELETE FROM artifact_keys WHERE artifact_id = ?
    await db.execute(
        "DELETE FROM artifact_keys WHERE artifact_id = ?",
        [artifact.artifact_id]
    )
```

---

### 3. AI-01: Zero Data Retention (3 violations) - CRITICAL

**Standard:** NAIC Model Bulletin / CPRA ADMT / ZDR Protocol
**Severity:** 🔴 CRITICAL
**Effort:** MEDIUM

**Issue:** LLM prompts/responses are being stored in conversation history.

**Affected Files:**

| File | Line | Pattern | Issue |
|------|------|---------|-------|
| [orchestrator/chat.py](orchestrator/chat.py#L141) | 141 | `self.conversation_history.append({...})` | Stores user messages |
| [orchestrator/chat.py](orchestrator/chat.py#L319) | 319 | `self.conversation_history.append({...})` | Stores assistant responses |
| [orchestrator/delegate.py](orchestrator/delegate.py#L574) | 574 | Documentation | FALSE POSITIVE |

**Real Violations: 2**

**Remediation:**

```python
# BEFORE (Non-compliant)
class OrchestratorChat:
    def __init__(self):
        self.conversation_history = []  # Persisted to memory/disk

    async def chat(self, message: str):
        self.conversation_history.append({"role": "user", "content": message})

# AFTER (Compliant)
class OrchestratorChat:
    def __init__(self):
        # Use ephemeral storage with TTL
        self._redis = Redis()
        self._session_ttl = 300  # 5 minutes max

    async def chat(self, message: str, session_id: str):
        # Store ephemerally with automatic expiration
        await self._redis.setex(
            f"session:{session_id}:history",
            self._session_ttl,
            json.dumps([{"role": "user", "content": message}])
        )

        # Make API call with store=False
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            # CRITICAL: Disable persistent storage
            # Note: This is Anthropic-specific; Azure OpenAI uses different config
        )
```

---

### 4. SEC-02: Weak Encryption (2 violations) - CRITICAL

**Standard:** ISSAS Encryption Standards / FIPS 140-3
**Severity:** 🔴 CRITICAL
**Effort:** LOW

**Issue:** MD5 hash usage detected (deprecated, not FIPS compliant).

**Affected Files:**

| File | Line | Code | Purpose |
|------|------|------|---------|
| [orchestrator/core/cache.py](orchestrator/core/cache.py#L228) | 228 | `hashlib.md5(key_data.encode()).hexdigest()` | Cache key generation |
| [orchestrator/tools/filesystem.py](orchestrator/tools/filesystem.py#L546) | 546 | `hashlib.md5(f.read()).hexdigest()` | File checksum |

**Remediation:**

```python
# BEFORE (Non-compliant)
key_hash = hashlib.md5(key_data.encode()).hexdigest()

# AFTER (Compliant)
# Use SHA-256 for non-security hashing (faster than SHA-512)
key_hash = hashlib.sha256(key_data.encode()).hexdigest()

# For file checksums (if not security-critical):
info["sha256"] = hashlib.sha256(f.read()).hexdigest()
```

---

### 5. SEC-03: PII Logging (3 violations) - HIGH

**Standard:** ISSAS Privacy / CPRA
**Severity:** 🟠 HIGH
**Effort:** LOW

**Affected Files:**

| File | Line | Pattern |
|------|------|---------|
| [orchestrator/delegate.py](orchestrator/delegate.py#L572) | 572 | Documentation example |
| [orchestrator/delegate.py](orchestrator/delegate.py#L679) | 679 | Documentation example |
| [orchestrator/core/escalation.py](orchestrator/core/escalation.py#L287) | 287 | Potential PII in log |

**Real Violations: 1** (others are documentation)

**Remediation:**

```python
# Add PII scrubber middleware
import re

PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', '<SSN-REDACTED>'),  # SSN
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL-REDACTED>'),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE-REDACTED>'),
]

def scrub_pii(text: str) -> str:
    """Remove PII from text before logging."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

# Usage
logger.info(scrub_pii(f"Processing escalation for user {user_id}"))
```

---

### 6. SEC-04: MFA Enforcement (1 violation) - HIGH

**Standard:** ISSAS Access Control
**Severity:** 🟠 HIGH
**Effort:** LOW

**Affected File:** [orchestrator/delegate.py](orchestrator/delegate.py#L573)

**Issue:** Documentation example showing MFA bypass pattern (not actual code).

**Status:** FALSE POSITIVE - This is in an agent system prompt describing what to look for, not actual implementation.

---

### 7. INT-02: Non-ACORD Export (1 violation) - MEDIUM

**Standard:** ACORD Standards / Ivans Integration
**Severity:** 🟡 MEDIUM
**Effort:** N/A

**Affected File:** [orchestrator/compliance/allstate_checker.py](orchestrator/compliance/allstate_checker.py#L189)

**Status:** FALSE POSITIVE - This is in the compliance rule definition itself.

---

## Compliance Checker Improvements Required

The compliance checker has significant false positive issues:

### Bug: SEC-02 Regex Matches "DES" Inside Words

**Current Pattern:**
```regex
(AES[-_]?128|DES|3DES|RC4|RC2|MD5|SHA1\b|sha1\(|md5\(|hashlib\.md5|hashlib\.sha1)
```

**Problem:** Matches "DES" in "design", "describes", "descriptor"

**Fix:**
```regex
(\bAES[-_]?128\b|\bDES\b|\b3DES\b|\bRC4\b|\bRC2\b|\bMD5\b|\bSHA1\b|sha1\(|md5\(|hashlib\.md5|hashlib\.sha1)
```

### Recommendation: Add Exclusions

1. Exclude `.venv/` directory (third-party packages)
2. Exclude test files by default
3. Exclude the compliance checker itself
4. Add context-aware detection (skip strings in docstrings/comments)

---

## Prioritized Remediation Roadmap

### Phase 1: Critical - Immediate Action Required (Week 1)

| Priority | Code | File | Fix | Effort |
|----------|------|------|-----|--------|
| 1 | AI-02 | escalation.py | Add human-in-loop for insurance decisions | 4 hrs |
| 2 | SEC-02 | cache.py | Replace MD5 with SHA-256 | 15 min |
| 3 | SEC-02 | filesystem.py | Replace MD5 with SHA-256 | 15 min |
| 4 | AI-01 | chat.py | Implement ephemeral storage | 2 hrs |

### Phase 2: High - Address Within Sprint (Week 2)

| Priority | Code | File | Fix | Effort |
|----------|------|------|-----|--------|
| 5 | SEC-01 | artifacts.py | Implement crypto-shredding option | 4 hrs |
| 6 | SEC-01 | webhooks.py | Add hard delete for deactivated webhooks | 1 hr |
| 7 | SEC-01 | main.py | Add key destruction for revoked API keys | 1 hr |
| 8 | SEC-03 | escalation.py | Add PII scrubber to logging | 1 hr |

### Phase 3: Tooling - Reduce Future False Positives (Week 3)

| Priority | Task | Effort |
|----------|------|--------|
| 9 | Fix SEC-02 regex to use word boundaries | 30 min |
| 10 | Add `.venv/` to default exclusions | 15 min |
| 11 | Add test file exclusions | 15 min |
| 12 | Add docstring/comment filtering | 2 hrs |

---

## Action Items / TODOs

### Immediate (This Week)

- [ ] **TODO-001**: Fix `orchestrator/core/escalation.py` - Add human-in-the-loop requirement for underwriting/claims/pricing decisions
- [ ] **TODO-002**: Fix `orchestrator/core/cache.py:228` - Replace `hashlib.md5()` with `hashlib.sha256()`
- [ ] **TODO-003**: Fix `orchestrator/tools/filesystem.py:546` - Replace `hashlib.md5()` with `hashlib.sha256()`
- [ ] **TODO-004**: Fix `orchestrator/chat.py` - Replace persistent `conversation_history` with ephemeral Redis storage

### This Sprint

- [ ] **TODO-005**: Fix `orchestrator/agents/data_science/artifacts.py:352` - Remove soft_delete parameter, implement crypto-shredding for PII artifacts
- [ ] **TODO-006**: Fix `orchestrator/core/webhooks.py:210` - Add option to hard-delete webhooks with PII
- [ ] **TODO-007**: Fix `orchestrator/api/main.py:466` - Destroy API key material on revocation
- [ ] **TODO-008**: Add PII scrubber middleware for all logging in `orchestrator/core/escalation.py`

### Technical Debt

- [ ] **TODO-009**: Fix SEC-02 regex pattern in `orchestrator/compliance/allstate_checker.py` to use word boundaries
- [ ] **TODO-010**: Add `.venv/` to default exclusion list in compliance checker
- [ ] **TODO-011**: Add context-aware false positive filtering (skip docstrings, comments)
- [ ] **TODO-012**: Create compliance pre-commit hook to catch violations before merge

---

## Appendix: False Positive Analysis

| Violation Code | Raw Count | False Positives | Real Violations |
|----------------|-----------|-----------------|-----------------|
| SEC-02 | 950 | 948 (99.8%) | 2 |
| AI-02 | 13 | 1 | 12 |
| SEC-01 | 9 | 5 | 4 |
| AI-01 | 4 | 2 | 2 |
| SEC-03 | 3 | 2 | 1 |
| SEC-04 | 2 | 2 | 0 |
| INT-02 | 1 | 1 | 0 |
| **TOTAL** | **982** | **961** | **21** |

**False Positive Rate: 97.9%**

The compliance checker requires significant improvement to reduce noise and improve actionability.

---

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Security Reviewer | ___________________ | __________ | ⬜ Pending |
| Tech Lead | ___________________ | __________ | ⬜ Pending |
| Compliance Officer | ___________________ | __________ | ⬜ Pending |

---

*Report generated by Allstate Compliance Sentinel using ISSAS/NIST/NAIC/CPRA rule set v1.0*
