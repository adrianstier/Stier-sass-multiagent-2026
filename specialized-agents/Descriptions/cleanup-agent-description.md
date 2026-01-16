# Cleanup Agent

## Role Overview

The **Cleanup Agent** is a specialized utility agent responsible for maintaining repository hygiene and fixing the common messes left behind by AI coding assistants like Claude Code. It ensures codebases remain clean, organized, and maintainable throughout the development lifecycle.

## Primary Purpose

AI coding assistants are powerful but imperfect. They often leave behind:
- Backup and duplicate files
- Orphaned imports and unused code
- Debug statements and console logs
- Scattered TODO comments
- Inconsistent naming and organization
- Partial implementations

The Cleanup Agent systematically identifies and resolves these issues, ensuring the repository stays healthy for both human developers and future AI interactions.

## When to Use This Agent

### Trigger Points
1. **Post-Development Phase**: After Backend/Frontend engineers complete their work
2. **Pre-Code Review**: To ensure reviewers see clean code
3. **Post-Review Feedback**: To implement cleanup suggestions
4. **Pre-Deployment**: Final hygiene check before release
5. **On-Demand**: When repository quality degrades

### Signs You Need Cleanup
- Build times increasing without new features
- Reviewers commenting on formatting/dead code
- Duplicate functionality discovered
- Import errors or circular dependencies
- Growing bundle/package sizes
- Confusion about which file is "the right one"

## Core Capabilities

### File System Cleanup
- Detect and remove backup files (`.bak`, `.old`, `_backup`)
- Delete system files (`.DS_Store`, `Thumbs.db`)
- Remove orphaned test fixtures
- Identify misplaced files
- Clean empty directories

### Code Cleanup
- Remove unused imports
- Delete dead code branches
- Strip debug statements
- Resolve or flag TODO comments
- Remove commented-out code
- Fix formatting inconsistencies

### Dependency Cleanup
- Identify unused dependencies
- Flag outdated packages
- Detect duplicate dependencies
- Validate lock file consistency

### Configuration Cleanup
- Remove duplicate config entries
- Validate environment files
- Clean package manifests
- Identify conflicting settings

## Safety Protocols

The Cleanup Agent follows strict safety rules:

### Auto-Safe Operations
- Removing `.DS_Store`, `Thumbs.db`, `*.pyc`
- Fixing trailing whitespace
- Sorting imports
- Removing clearly unused imports

### Requires Confirmation
- Deleting source files (`.js`, `.ts`, `.py`)
- Removing TODO comments
- Deleting test files
- Restructuring directories

### Never Touches
- Config files (without explicit permission)
- Environment files (`.env*`)
- Lock files (`package-lock.json`, etc.)
- `.git` directory contents
- Files matching `.gitignore` patterns

## Interaction Model

### Inputs
- Repository path to scan
- Scope (full repo, specific directories, specific file types)
- Safety level (conservative, moderate, aggressive)
- Previous cleanup reports (for delta analysis)

### Outputs
- Comprehensive cleanup report (Markdown)
- List of changes made
- List of items flagged for human review
- Verification checklist
- Prevention recommendations

## Integration Points

### With Code Reviewer
```
Cleanup Agent → Clean Code → Code Reviewer
                    ↑
Code Reviewer Feedback → Cleanup Agent (fixes)
```

### With CI/CD Pipeline
```yaml
# Example workflow
- name: Run Cleanup Check
  run: orchestrator run-agent cleanup --check-only

- name: Apply Safe Cleanups
  run: orchestrator run-agent cleanup --auto-fix --safe-only
```

### With Pre-Commit Hooks
The Cleanup Agent can generate configurations for:
- `pre-commit` framework
- Husky (JavaScript)
- Git hooks (shell scripts)

## Output Report Structure

```markdown
# Repository Cleanup Report

## Executive Summary
Brief overview of findings and actions

## Scan Results
- File system issues
- Code quality issues
- Organization issues

## Actions Taken
- Files removed (with reasons)
- Code changes (with locations)
- Configurations updated

## Flagged for Review
Items requiring human decision

## Verification Checklist
Steps to confirm cleanup success

## Prevention Recommendations
Suggestions for maintaining cleanliness
```

## Claude Code-Specific Intelligence

The agent is specifically trained to recognize Claude Code patterns:

### Common Artifacts
- `*_backup.*`, `*_old.*`, `*_copy.*`
- Merge conflict markers left in code
- "Claude" mentioned in TODO comments
- Duplicate function implementations
- Abandoned refactoring attempts

### Preventive Measures
Recommends `.gitignore` patterns:
```gitignore
# Claude Code artifacts
*_backup.*
*_old.*
*_copy.*
*.bak
*~
.claude/commands/*_draft*
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Duplicate files | 0 |
| Unused imports | 0 |
| Debug statements | 0 in production paths |
| TODO age | < 2 sprints |
| Build warnings | No increase |
| Bundle size | No unnecessary growth |

## Limitations

The Cleanup Agent does NOT:
- Refactor working code for style preferences
- Make architectural decisions
- Implement new features
- Modify business logic
- Override team conventions

When architectural issues are discovered, they are flagged for the Tech Lead.

## Example Usage

### CLI
```bash
# Full repository scan
orchestrator run-agent cleanup

# Check only (no changes)
orchestrator run-agent cleanup --check-only

# Auto-fix safe issues only
orchestrator run-agent cleanup --auto-fix --safe-only

# Specific directory
orchestrator run-agent cleanup --path src/components

# Generate prevention config
orchestrator run-agent cleanup --generate-config
```

### API
```json
POST /runs
{
  "goal": "Clean up repository after feature development",
  "context": {
    "scope": "full",
    "safety_level": "moderate",
    "auto_fix": true
  }
}
```

## Best Practices

1. **Run Regularly**: Schedule cleanup as part of sprint hygiene
2. **Pre-Review Always**: Clean before code review to reduce noise
3. **Document Exceptions**: If something should NOT be cleaned, document why
4. **Learn Patterns**: Use reports to identify recurring issues
5. **Automate Prevention**: Implement suggested pre-commit hooks

## Related Agents

- **Code Reviewer**: Reviews code quality (complementary to cleanup)
- **Tech Lead**: Escalation path for architectural issues
- **Security Reviewer**: May find issues during cleanup (e.g., exposed secrets)

---

*The Cleanup Agent ensures your repository stays healthy, making all other agents more effective and keeping human developers happy.*
