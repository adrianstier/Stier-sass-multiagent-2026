# Multi-Agent Orchestration Platform

A production-grade multi-agent orchestration system that coordinates specialized AI agents through the software development lifecycle. Built with FastAPI, Celery, PostgreSQL, and Redis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This platform enables fully autonomous software development workflows by orchestrating multiple specialized AI agents. Each agent has a specific role, tools, and domain expertise, working together through a coordinated task DAG to deliver complete solutions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Goal                                       │
│                    "Build a task management application"                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                       │
│           Decomposes goals → Creates task DAG → Coordinates agents           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ Requirements  │           │   Planning    │           │   Design      │
│  Business     │    ──►    │   Project     │    ──►    │   UX          │
│  Analyst      │           │   Manager     │           │   Engineer    │
└───────────────┘           └───────────────┘           └───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ Architecture  │           │   Database    │           │ Implementation│
│   Tech Lead   │    ──►    │   Engineer    │    ──►    │ Backend/      │
│               │           │               │           │ Frontend      │
└───────────────┘           └───────────────┘           └───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │  Code Review  │       ──►     │   Security    │
            │    GATE       │               │  Review GATE  │
            └───────────────┘               └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │   COMPLETE    │
                                            └───────────────┘
```

## Philosophy: Get Shit Done

This orchestrator embodies a **pragmatic, results-driven approach** to AI-powered development:

### Principles

1. **Ship First, Perfect Later**
   - Working code beats perfect plans
   - Iterate rapidly, validate continuously
   - 80% solution now > 100% solution never

2. **Minimal Viable Process**
   - Only the steps that add value
   - No ceremony for ceremony's sake
   - Agents do real work, not busywork

3. **Fail Fast, Recover Faster**
   - Dead Letter Queue catches failures
   - Checkpoints enable recovery
   - Escalation gets humans involved only when needed

4. **Parallel Everything**
   - Backend and Frontend run concurrently
   - Multiple agents work simultaneously
   - Don't serialize what can parallelize

5. **Quality Gates, Not Quality Theater**
   - Code review catches real issues
   - Security review blocks actual vulnerabilities
   - Skip the rubber stamps

### The Anti-Patterns We Avoid

| Anti-Pattern | Our Approach |
|--------------|--------------|
| Endless planning meetings | Orchestrator creates plan, starts executing |
| Waterfall handoffs | Agents collaborate via channels in real-time |
| Manual status updates | Event log tracks everything automatically |
| Approval bottlenecks | Tiered escalation with timeout auto-decisions |
| Scope creep | Task DSL locks in deliverables upfront |

---

## The Ralph Wiggum Loop

> *"I'm helping!"* - Ralph Wiggum

The **Ralph Wiggum Loop** is our iterative validation pattern that ensures agents don't just *think* they're done—they actually *are* done. Every agent output goes through a self-correction cycle until it meets objective criteria.

> **Credit**: Inspired by [Geoffrey Huntley's Ralph Wiggum Technique](https://github.com/ghuntley/how-to-ralph-wiggum) - "the AI development methodology that reduces software costs to less than a fast food worker's wage."

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE RALPH WIGGUM LOOP                              │
│                                                                              │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│    │  Agent   │ ───► │ Validate │ ───► │  Pass?   │ ───► │   Done   │      │
│    │  Output  │      │  Output  │      │          │      │          │      │
│    └──────────┘      └──────────┘      └────┬─────┘      └──────────┘      │
│                                             │ NO                            │
│                                             ▼                               │
│                                      ┌──────────┐                           │
│                                      │ Critique │                           │
│                                      │ & Revise │ ◄─────────┐               │
│                                      └────┬─────┘           │               │
│                                           │                 │               │
│                                           ▼                 │               │
│                                      ┌──────────┐           │               │
│                                      │  Still   │ ── NO ───┘               │
│                                      │  Wrong?  │                           │
│                                      └────┬─────┘                           │
│                                           │ YES (max retries)               │
│                                           ▼                                 │
│                                      ┌──────────┐                           │
│                                      │ Escalate │                           │
│                                      │ to Human │                           │
│                                      └──────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Critical Difference: Fresh Context vs Bloated Context

Most people get Ralph Wiggum wrong. The key insight from the community:

> *"The original bash loop starts a fresh context window each iteration. That's a fundamental difference... as tasks pile up, your context gets more bloated, more hallucinations."*

**Our orchestrator solves this by design:**

| Problem | Single-Context Loop | Our Multi-Agent Approach |
|---------|--------------------|-----------------------|
| Context bloat | Everything in one window | Each agent gets fresh context per task |
| Hallucination accumulation | Errors compound | Checkpoints isolate failures |
| No compaction control | Manual intervention | Automatic context management |
| Ambiguous decisions | Agent guesses | `plan.md` + Task DSL locks deliverables |

### The Five Pillars of Running Ralph Right

Based on real-world experience from the AI coding community:

#### 1. Safety: Sandbox Everything
```
Don't let agents nuke your system. Our approach:
├── Tool executor with role-based allowlists
├── No agent performs privileged actions directly
├── Git worktrees for isolated execution
└── Full audit trail of every tool usage
```

#### 2. Efficiency: Plan Before You Loop
```
You don't want Ralph making ambiguous decisions:
├── plan.md     → Clear task breakdown upfront
├── activity.md → Running log of what's happening
├── Task DSL    → Dependencies and expected artifacts defined
└── USE GIT     → Every iteration is recoverable
```

#### 3. Cost: Set Max Iterations
```python
# Don't let it run forever
RALPH_CONFIG = {
    "max_iterations": 10,        # Start here, adjust based on task
    "cost_ceiling": 5.00,        # Dollar limit per task
    "checkpoint_interval": 3,    # Save state every N iterations
}
```

#### 4. Validation: Objective Completion Criteria
```python
# Don't trust "I think I'm done" - verify objectively
async def verify_completion(task, output):
    checks = [
        run_test_suite(),           # Tests pass?
        lint_check(),               # Code quality?
        type_check(),               # Types valid?
        security_scan(),            # No vulnerabilities?
        matches_requirements(),     # Meets spec?
    ]
    return all(await asyncio.gather(*checks))
```

#### 5. Feedback: Let Agents See Their Work
```
Give agents tools to verify their own output:
├── Playwright/browser → Screenshot and verify UI
├── Test runners       → Execute and see results
├── Console logs       → Debug output visible
└── Semantic validator → LLM checks artifact quality
```

### How It Works in Our System

The loop is embedded at every critical junction:

| Stage | Ralph Loop Application | Module |
|-------|----------------------|--------|
| **Task Execution** | Agent validates own output before marking complete | `base.py` |
| **Quality Gates** | Code/Security reviewers loop until issues fixed | `tasks.py` |
| **Supervision** | Lead agents critique subordinate work | `supervision.py` |
| **Semantic Check** | LLM validates artifacts meet requirements | `semantic_validator.py` |
| **Final Review** | Cross-artifact consistency validation | `semantic_validator.py` |

### The Three Exit Conditions

Every Ralph Loop terminates via one of three paths:

1. **PASS** - Output meets all validation criteria (objective, not vibes)
2. **MAX_RETRIES** - Hit iteration limit, escalate to supervisor/human
3. **BUDGET_EXCEEDED** - Cost threshold reached, checkpoint and pause

### Implementation Pattern

```python
# The orchestrator's Ralph Loop implementation
class BaseAgent:
    async def execute_with_ralph_loop(self, task, max_iterations=10):
        for i in range(max_iterations):
            # Fresh context management - avoid bloat
            context = await self.context_manager.prepare_context(task)

            # 1. Generate output
            output = await self.generate(task, context)

            # 2. OBJECTIVE validation - not "do you think you're done?"
            validation = await self.verify_completion(output, task)

            if validation.passed:
                await self.checkpoint_manager.save(task, output)
                return output  # Actually done!

            # 3. Critique with specific, actionable feedback
            critique = await self.get_critique(output, validation)

            # 4. Log for observability
            self.log_event("ralph_loop_iteration", {
                "iteration": i + 1,
                "issues": validation.issues,
                "cost_so_far": self.cost_tracker.total,
            })

            # 5. Check cost ceiling
            if self.cost_tracker.total > task.cost_ceiling:
                await self.checkpoint_manager.save(task, output)
                raise BudgetExceededError(f"Hit ${task.cost_ceiling} limit")

        # Max retries - escalate, don't just fail
        await self.escalate_to_supervisor(task, output, validation)

    async def verify_completion(self, output, task):
        """Objective verification - the key to Ralph working right"""
        results = {
            "tests_pass": await self.run_tests(output),
            "lint_clean": await self.lint_check(output),
            "types_valid": await self.type_check(output),
            "security_ok": await self.security_scan(output),
            "meets_spec": await self.semantic_validator.validate(output, task.requirements),
        }
        return ValidationResult(
            passed=all(results.values()),
            issues=[k for k, v in results.items() if not v]
        )
```

### Supervision Hierarchy + Ralph Loop

The hierarchical supervision system applies Ralph Loops at each level:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARCHITECT LEVEL                           │
│              (Tech Lead, Security Reviewer)                      │
│                                                                  │
│   Ralph Loop: Strategic validation, architecture compliance      │
│   Escalates to: Human (via escalation.py with Slack/email)      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ critique
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LEAD LEVEL                               │
│              (Code Reviewer, Project Manager)                    │
│                                                                  │
│   Ralph Loop: Quality validation, standards compliance           │
│   Escalates to: Architect level                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │ critique
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EXECUTION LEVEL                            │
│         (Backend, Frontend, Database, UX Engineers)              │
│                                                                  │
│   Ralph Loop: Self-validation, unit tests, lint checks           │
│   Escalates to: Lead level                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Ralph Loop Metrics

The observability module tracks loop efficiency:

```python
# Metrics collected for each Ralph Loop
metrics.record_ralph_loop(
    agent_role="backend_engineer",
    iterations=3,           # How many times before passing
    total_duration=45.2,    # Seconds spent in loop
    cost_incurred=0.42,     # Dollars spent
    escalated=False,        # Did it need supervisor help?
    issues_found=["missing error handling", "no input validation"],
    issues_fixed=["missing error handling", "no input validation"],
    exit_condition="PASS"   # PASS | MAX_RETRIES | BUDGET_EXCEEDED
)
```

### Common Pitfalls (and How We Avoid Them)

| Pitfall | What Goes Wrong | Our Solution |
|---------|----------------|--------------|
| **Single context window** | Bloat, hallucinations compound | Fresh context per task, checkpoints |
| **No objective validation** | "I think I'm done" ≠ done | Test suites, lint, semantic validation |
| **Unlimited iterations** | Runaway costs | Max iterations + cost ceiling |
| **No plan upfront** | Ambiguous decisions | Task DSL + plan.md |
| **Can't verify own work** | Blind to mistakes | Playwright, test runners, logs |
| **Mega PRs** | Unreviewable changes | Atomic tasks, incremental commits |
| **Early mistakes compound** | Wrong foundation | Quality gates block bad code |

### Why "Ralph Wiggum"?

Because AI agents, like Ralph, are enthusiastic helpers who genuinely believe they're contributing—but without validation, they might be "helping" in ways that don't actually help. The loop ensures that:

- **Confidence ≠ Correctness** - Agents validate, not just generate
- **Iteration beats perfection** - Multiple passes produce better results
- **Escalation is okay** - Some problems need a smarter brain
- **No false positives** - "Done" means actually done
- **Fresh context matters** - Don't let bloat kill your loop

---

## Key Features

### Core Orchestration
- **Asynchronous Autonomy** - Runs end-to-end without human intervention
- **Durable State Model** - Append-only event log with full traceability
- **Role-Based Queues** - Dedicated Celery queue per agent role
- **Quality Gates** - Code review and security review gates with blocking logic
- **Task DSL** - Declarative workflow specification with dependencies

### Production Features
- **Authentication** - JWT tokens and API key authentication
- **Multi-tenancy** - Organization and team-based isolation
- **Rate Limiting** - Token budget enforcement per tenant
- **Webhooks** - Event notifications for external integrations
- **Caching** - Redis-based response caching

### Advanced Orchestration

| Feature | Description |
|---------|-------------|
| **Dead Letter Queue** | Failed task persistence with replay capability |
| **Context Window Management** | Token estimation and intelligent summarization |
| **Dynamic Workflow Modification** | Runtime task add/remove/modify with branching |
| **Agent Collaboration Channels** | Redis pub/sub messaging between agents |
| **Checkpoint/Resume** | Workflow state persistence and recovery |
| **Cost Prediction** | Historical-based cost estimation and optimization |
| **Hierarchical Supervision** | Agent → Lead → Architect critique loops |
| **Semantic Validation** | LLM-based artifact quality validation |
| **Expanded Observability** | Histograms, gauges, Prometheus export |
| **Human-in-the-Loop Escalation** | Tiered escalation with Slack/email notifications |
| **Project Analysis** | Assess existing codebases before starting work |
| **Project Modes** | Adaptive workflows for greenfield vs existing projects |

### Agent Tools

The orchestrator provides comprehensive tools that enable agents to work on real repositories:

| Category | Tools | Description |
|----------|-------|-------------|
| **Filesystem** | `read_file`, `write_file`, `list_directory`, `search_files`, `copy_file`, `move_file`, `create_directory`, `get_file_info` | Sandboxed file operations with path validation and size limits |
| **Git** | `git_status`, `git_diff`, `git_log`, `git_add`, `git_commit`, `git_branch`, `git_checkout`, `git_merge`, `git_stash`, `git_fetch`, `git_pull`, `git_push`, `git_show`, `git_blame` | Full git workflow with protected branch enforcement |
| **Execution** | `run_command`, `run_tests`, `run_linter`, `run_build`, `run_type_check`, `install_dependencies` | Sandboxed command execution with allowlists and auto-detection |
| **Code Analysis** | `extract_symbols`, `find_references`, `find_definition`, `search_code`, `get_symbol_outline`, `get_file_dependencies`, `get_project_structure` | Multi-language AST parsing and semantic analysis |

#### Tool Security Model

All tools operate within a comprehensive security sandbox:

```python
# Filesystem sandboxing
SandboxConfig(
    allowed_paths=["/project/src", "/project/tests"],
    denied_patterns=["*.env", "*.pem", "*secret*", "node_modules/"],
    max_file_size_mb=10.0,
    allow_delete=False  # Configurable per role
)

# Git safety controls
GitConfig(
    protected_branches=["main", "master", "production"],
    allowed_operations=[GitOperationType.READ, GitOperationType.WRITE],
    max_commit_files=50  # Prevent mega-commits
)

# Execution sandboxing
ExecutionConfig(
    mode=ExecutionMode.STANDARD,  # RESTRICTED | STANDARD | ELEVATED
    timeout_seconds=300,
    max_output_size=1_000_000
)
```

#### Role-Based Tool Access

Each agent role has carefully scoped tool permissions:

| Role | Filesystem | Git | Execution | Code Analysis |
|------|------------|-----|-----------|---------------|
| **Backend/Frontend Engineer** | Full | Full (except push) | Full | Full |
| **Code Reviewer** | Read-only | Read-only | Tests + Lint | Full |
| **Security Reviewer** | Read-only | Read-only | Security scans | Full |
| **Tech Lead** | Full | Full | Full | Full |
| **Business Analyst** | Read-only | Read-only | None | Structure only |

#### Auto-Detection

The execution tools automatically detect project tooling:

```
Detected test framework: pytest (found pytest.ini)
Detected linter: ruff (found ruff.toml)
Detected package manager: poetry (found pyproject.toml)
Detected type checker: mypy (found mypy.ini)
```

---

## Project Modes: Greenfield vs Existing Projects

The orchestrator intelligently adapts its workflow based on the type of project—whether you're building from scratch or working within an existing codebase.

### Project Types

| Type | Description | Workflow Adaptation |
|------|-------------|---------------------|
| `GREENFIELD` | New project from scratch | Full planning cycle, all agents engaged |
| `EXISTING_ACTIVE` | Active codebase with recent commits | Code analysis first, match existing patterns |
| `LEGACY_MAINTENANCE` | Older codebase, maintenance mode | Backwards compatibility required, minimal changes |
| `FEATURE_BRANCH` | Adding feature to existing project | Scope to specific area, integration focus |
| `BUG_FIX` | Fix specific issue | Minimal changes, root cause analysis |
| `REFACTOR` | Improve existing code | Preserve behavior, incremental changes |

### Automatic Project Analysis

Before any work begins, the `ProjectAnalyzer` assesses the current state:

```python
from orchestrator.core import get_project_analyzer

analyzer = get_project_analyzer()
project_state = await analyzer.analyze("/path/to/project")

# What you get:
print(project_state.project_type)      # EXISTING_ACTIVE
print(project_state.languages)         # ["python", "typescript"]
print(project_state.frameworks)        # ["fastapi", "react"]
print(project_state.architecture)      # "monolith" | "microservices" | "modular"
print(project_state.technical_debt)    # List of debt items
print(project_state.security_issues)   # Potential vulnerabilities
```

### Analysis Captures

| Category | What's Detected |
|----------|-----------------|
| **Languages** | Python, TypeScript, Go, Rust, Java, etc. |
| **Frameworks** | FastAPI, Django, React, Vue, Express, etc. |
| **Architecture** | Monolith, microservices, modular, serverless |
| **Testing** | Test framework, coverage, test patterns |
| **CI/CD** | GitHub Actions, Jenkins, CircleCI configs |
| **Code Quality** | Linting, type hints, documentation coverage |
| **Dependencies** | Outdated packages, security vulnerabilities |
| **Technical Debt** | Code smells, complexity hotspots, TODOs |

### Workflow Adaptation

Each project type gets a tailored workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GREENFIELD PROJECT                                   │
│                                                                              │
│   [BA] → [PM] → [UX] → [Tech Lead] → [DB] → [Backend] → [Frontend]         │
│                                              ↓           ↓                   │
│                                         [Code Review] → [Security Review]    │
│                                                              ↓               │
│   Full planning cycle, all agents, complete architecture                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXISTING PROJECT                                      │
│                                                                              │
│   [Project Analyzer] → [Tech Lead] → [Relevant Engineers Only]              │
│          ↓                                    ↓                              │
│   Detect patterns,           [Code Review] → [Security Review]               │
│   match style,                                    ↓                          │
│   identify scope             Backwards-compatible, minimal changes           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            BUG FIX                                            │
│                                                                              │
│   [Project Analyzer] → [Relevant Engineer] → [Code Review]                   │
│          ↓                     ↓                   ↓                         │
│   Root cause          Minimal fix only      Verify no regressions            │
│   analysis                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Context Injection

Each agent receives project-specific context:

```python
# Backend Engineer working on EXISTING_ACTIVE project gets:
{
    "project_type": "EXISTING_ACTIVE",
    "existing_patterns": {
        "orm": "SQLAlchemy with async sessions",
        "api_style": "REST with Pydantic models",
        "error_handling": "Custom HTTPException subclasses",
        "testing": "pytest with fixtures in conftest.py"
    },
    "code_style": {
        "formatter": "black",
        "linter": "ruff",
        "type_hints": "required",
        "docstring_style": "Google"
    },
    "constraints": [
        "Must maintain backwards compatibility",
        "Follow existing naming conventions",
        "Add tests for any new functionality"
    ]
}
```

### Using Project Modes

```python
from orchestrator.core import (
    get_project_analyzer,
    get_project_mode_handler,
    get_workflow_config,
)

# 1. Analyze the project
analyzer = get_project_analyzer()
project_state = await analyzer.analyze("/path/to/existing/project")

# 2. Get appropriate workflow configuration
workflow_config = get_workflow_config(project_state.project_type)

# 3. Or explicitly set the mode
mode_handler = get_project_mode_handler()
workflow = await mode_handler.create_adaptive_workflow(
    goal="Add user authentication",
    project_state=project_state,
    db=db
)

# 4. The workflow automatically:
#    - Skips unnecessary planning phases for bug fixes
#    - Requires backwards compatibility for legacy projects
#    - Matches existing code patterns and style
#    - Adjusts quality gate strictness
```

### CLI Support

```bash
# Analyze project before starting work
orchestrator analyze /path/to/project

# Start with explicit mode
orchestrator run "Add OAuth support" --mode existing_active --project /path/to/project

# Auto-detect mode from project
orchestrator run "Fix login bug" --project /path/to/project
# → Automatically detects EXISTING_ACTIVE or BUG_FIX based on goal
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Anthropic API key

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/adrianstier/Stier-sass-multiagent-2026.git
cd Stier-sass-multiagent-2026/orchestrator

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Start all services
docker-compose up -d

# Initialize database
docker-compose exec api python -c "from orchestrator.core.database import init_db; init_db()"

# View logs
docker-compose logs -f
```

### Using CLI

```bash
# Install the package
cd orchestrator
pip install -e ".[dev]"

# Start a new workflow run
orchestrator run "Build a simple CRUD application for task management"

# Monitor progress
orchestrator status <run_id>

# View generated artifacts
orchestrator artifacts <run_id> --content
```

### Using API

```bash
# Create a new run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{"goal": "Build a task management API with user authentication"}'

# Check status
curl http://localhost:8000/runs/<run_id>/status

# View artifacts
curl http://localhost:8000/runs/<run_id>/artifacts
```

## Architecture

### Agent Roles

| Agent | Role | Artifacts |
|-------|------|-----------|
| Business Analyst | Requirements gathering | Requirements document |
| Project Manager | Planning and coordination | Project plan, timeline |
| UX Engineer | User experience design | Wireframes, user flows |
| Tech Lead | Technical architecture | Architecture document |
| Database Engineer | Database design | Schema, migrations |
| Backend Engineer | Server-side implementation | API code, tests |
| Frontend Engineer | Client-side implementation | UI code, components |
| Code Reviewer | Quality gate (code) | Review report |
| Security Reviewer | Quality gate (security) | Security audit |
| Data Scientist | Data analysis (optional) | Analysis, models |
| Cleanup Agent | Post-workflow cleanup | Cleanup report |

### Technology Stack

- **API Framework**: FastAPI with async support
- **Task Queue**: Celery with Redis broker
- **Database**: PostgreSQL with SQLAlchemy ORM
- **LLM Integration**: Anthropic Claude API
- **Monitoring**: Flower (Celery), Prometheus metrics
- **Containerization**: Docker and Docker Compose

## Project Structure

```
├── orchestrator/                 # Core orchestration system
│   ├── agents/                   # Agent implementations
│   │   ├── base.py              # Base agent class with LLM integration
│   │   ├── orchestrator.py      # Main orchestrator logic
│   │   ├── specialists.py       # All specialist agents
│   │   └── tasks.py             # Celery task definitions
│   ├── api/                      # REST API
│   │   └── main.py              # FastAPI application
│   ├── core/                     # Core modules
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── celery_app.py        # Celery configuration
│   │   ├── dlq.py               # Dead Letter Queue
│   │   ├── checkpoint.py        # Checkpoint/Resume
│   │   ├── escalation.py        # Human-in-the-loop
│   │   ├── supervision.py       # Hierarchical supervision
│   │   ├── cost_predictor.py    # Cost prediction
│   │   ├── observability.py     # Metrics and monitoring
│   │   ├── project_analyzer.py  # Existing codebase analysis
│   │   ├── project_modes.py     # Adaptive workflow configuration
│   │   └── ...                  # Other core modules
│   ├── tools/                    # Tool registry and execution
│   ├── tests/                    # Test suite
│   └── README.md                 # Detailed documentation
├── specialized-agents/           # Agent definitions
│   ├── Descriptions/            # Role descriptions
│   └── system-prompts/          # System prompts per agent
├── docs/                         # Additional documentation
│   └── claude-code-guide.md     # Claude Code usage guide
├── mcp-servers/                  # MCP server documentation
└── Images/                       # Documentation assets
```

## Documentation

- **[Orchestrator Guide](orchestrator/README.md)** - Detailed system documentation
- **[Advanced Features](docs/advanced-features.md)** - 10 orchestration enhancements
- **[Claude Code Guide](docs/claude-code-guide.md)** - Claude Code best practices
- **[MCP Servers](mcp-servers/README.md)** - MCP server setup guides
- **[API Documentation](http://localhost:8000/docs)** - Swagger UI (when running)

## Monitoring

| Service | URL | Description |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | Swagger UI |
| Flower | http://localhost:5555 | Celery monitoring |
| Metrics | http://localhost:8000/metrics | Prometheus metrics |

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://...` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `LLM_MODEL` | Claude model | `claude-sonnet-4-20250514` |
| `MAX_ITERATIONS` | Max orchestrator iterations | `50` |
| `REQUIRE_CODE_REVIEW` | Enforce code review gate | `true` |
| `REQUIRE_SECURITY_REVIEW` | Enforce security gate | `true` |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/)
- Task queue powered by [Celery](https://docs.celeryq.dev/)
- API framework by [FastAPI](https://fastapi.tiangolo.com/)
