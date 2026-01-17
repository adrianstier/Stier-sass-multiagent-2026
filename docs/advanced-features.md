# Advanced Orchestration Features

This document describes the 10 advanced orchestration enhancements that extend the core multi-agent system with production-grade capabilities.

## Table of Contents

1. [Dead Letter Queue (DLQ)](#1-dead-letter-queue-dlq)
2. [Context Window Management](#2-context-window-management)
3. [Dynamic Workflow Modification](#3-dynamic-workflow-modification)
4. [Agent Collaboration Channels](#4-agent-collaboration-channels)
5. [Checkpoint/Resume](#5-checkpointresume)
6. [Cost Prediction](#6-cost-prediction)
7. [Hierarchical Supervision](#7-hierarchical-supervision)
8. [Semantic Validation](#8-semantic-validation)
9. [Observability Dashboard](#9-observability-dashboard)
10. [Human-in-the-Loop Escalation](#10-human-in-the-loop-escalation)

---

## 1. Dead Letter Queue (DLQ)

**Module:** `orchestrator/core/dlq.py`

Handles permanently failed tasks with replay capability for debugging and recovery.

### Features
- Stores failed tasks with full execution context
- Tracks failure reasons and stack traces
- Supports task replay after fixing issues
- Manual resolution and discard options

### Usage

```python
from orchestrator.core import get_dlq_manager

dlq = get_dlq_manager()

# Send a failed task to DLQ
await dlq.send_to_dlq(
    db=db,
    task_id=task.id,
    error_message="API rate limit exceeded",
    stack_trace=traceback.format_exc(),
    execution_context={"attempt": 3, "last_response": response}
)

# Replay a task from DLQ
await dlq.replay_task(db, dlq_entry_id)

# Mark as manually resolved
await dlq.resolve_entry(db, dlq_entry_id, resolution_notes="Fixed API key")
```

### Database Model

```sql
CREATE TABLE dead_letter_tasks (
    id UUID PRIMARY KEY,
    task_id UUID REFERENCES tasks(id),
    run_id UUID REFERENCES runs(id),
    error_message TEXT,
    stack_trace TEXT,
    execution_context JSONB,
    status VARCHAR(50),  -- pending, replayed, resolved, discarded
    created_at TIMESTAMP,
    resolved_at TIMESTAMP
);
```

---

## 2. Context Window Management

**Module:** `orchestrator/core/context_manager.py`

Intelligent token management to prevent context overflow during long-running agent tasks.

### Features
- Token estimation for messages and artifacts
- Automatic summarization of large content
- Sliding window management
- Semantic code chunking for large files

### Usage

```python
from orchestrator.core import get_context_manager

ctx = get_context_manager(max_tokens=100000)

# Add content to context window
ctx.add_message(role="user", content=user_input)
ctx.add_message(role="assistant", content=response)

# Check if context needs compaction
if ctx.needs_compaction():
    summary = await ctx.compact(keep_recent=10)

# Get current context for LLM call
messages = ctx.get_messages()
```

### Summarization Strategies

| Content Type | Strategy |
|-------------|----------|
| Code | Extract signatures, docstrings, key logic |
| Requirements | Bullet point extraction |
| Conversation | Key decisions and outcomes |
| Large files | Semantic chunking with overlap |

---

## 3. Dynamic Workflow Modification

**Module:** `orchestrator/core/workflow_modifier.py`

Runtime modification of workflow DAGs without restarting runs.

### Features
- Add/remove/modify tasks at runtime
- Insert conditional branches
- Create parallel execution paths
- Modify dependencies dynamically

### Usage

```python
from orchestrator.core import get_workflow_modifier

modifier = get_workflow_modifier()

# Add a new task mid-workflow
await modifier.add_task(
    db=db,
    run_id=run_id,
    role="security_reviewer",
    prompt="Review the authentication implementation",
    depends_on=[backend_task_id]
)

# Create a conditional branch
await modifier.create_branch(
    db=db,
    run_id=run_id,
    condition="artifact.test_coverage < 80",
    if_true_tasks=[create_tests_task],
    if_false_tasks=[]
)

# Insert a quality gate
await modifier.insert_gate(
    db=db,
    run_id=run_id,
    after_task_id=implementation_task_id,
    gate_type="code_review"
)
```

### Workflow Branch Model

```python
@dataclass
class WorkflowBranch:
    branch_id: UUID
    condition: str
    if_true_tasks: list[UUID]
    if_false_tasks: list[UUID]
    evaluated: bool = False
    result: Optional[bool] = None
```

---

## 4. Agent Collaboration Channels

**Module:** `orchestrator/core/channels.py`

Real-time communication between agents using Redis pub/sub.

### Features
- Direct agent-to-agent messaging
- Handoff requests with context transfer
- Multi-agent meetings for complex decisions
- Message persistence and replay

### Usage

```python
from orchestrator.core import get_channel_manager, AgentMessage

channels = get_channel_manager()

# Send a message to another agent
await channels.send_message(AgentMessage(
    from_agent="backend_engineer",
    to_agent="database_engineer",
    message_type="question",
    content="Should we use UUID or auto-increment for primary keys?",
    context={"table": "users"}
))

# Request a handoff
await channels.request_handoff(
    from_agent="frontend_engineer",
    to_agent="ux_engineer",
    reason="Need design clarification for the dashboard layout",
    artifacts=["wireframe_v1.png"]
)

# Start a multi-agent meeting
meeting = await channels.create_meeting(
    topic="API Design Review",
    participants=["tech_lead", "backend_engineer", "frontend_engineer"],
    agenda=["REST vs GraphQL", "Authentication strategy"]
)
```

### Message Types

| Type | Purpose |
|------|---------|
| `question` | Request information from another agent |
| `answer` | Response to a question |
| `handoff` | Transfer task ownership |
| `broadcast` | Notify all agents |
| `meeting_invite` | Start collaborative session |

---

## 5. Checkpoint/Resume

**Module:** `orchestrator/core/checkpoint.py`

Workflow state persistence for failure recovery and long-running operations.

### Features
- Automatic checkpointing at configurable intervals
- Full workflow state capture
- Agent conversation history persistence
- Resume from any checkpoint

### Usage

```python
from orchestrator.core import get_checkpoint_manager

checkpoints = get_checkpoint_manager()

# Create a checkpoint
checkpoint_id = await checkpoints.create_checkpoint(
    db=db,
    run_id=run_id,
    label="pre_security_review"
)

# List available checkpoints
available = await checkpoints.list_checkpoints(db, run_id)

# Resume from checkpoint
await checkpoints.resume_from_checkpoint(
    db=db,
    checkpoint_id=checkpoint_id
)
```

### Checkpoint Contents

```python
@dataclass
class WorkflowCheckpoint:
    id: UUID
    run_id: UUID
    label: str
    run_state: dict           # Full run status and metadata
    task_states: list[dict]   # All task statuses
    artifacts: list[dict]     # Generated artifacts
    agent_contexts: dict      # Conversation histories (compressed)
    created_at: datetime
```

---

## 6. Cost Prediction

**Module:** `orchestrator/core/cost_predictor.py`

Predict and optimize LLM costs based on historical execution data.

### Features
- Per-task cost estimation
- Full run cost forecasting
- Model tier recommendations
- Budget enforcement with warnings

### Usage

```python
from orchestrator.core import get_cost_predictor

predictor = get_cost_predictor()

# Estimate cost for a single task
estimate = await predictor.estimate_task_cost(
    role="backend_engineer",
    prompt_length=5000,
    expected_iterations=3
)
print(f"Estimated: ${estimate.total_cost:.4f}")

# Forecast full run cost
forecast = await predictor.forecast_run_cost(
    goal="Build a REST API with authentication",
    db=db
)
print(f"Run forecast: ${forecast.total_estimated:.2f}")

# Check budget before execution
can_proceed = await predictor.check_budget_before_execution(
    db=db,
    run_id=run_id,
    estimated_cost=estimate.total_cost
)
```

### Cost Breakdown

```python
@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    confidence: float  # 0-1 based on historical data availability
```

---

## 7. Hierarchical Supervision

**Module:** `orchestrator/core/supervision.py`

Multi-level agent supervision with critique loops.

### Features
- Supervision hierarchy (Agent → Lead → Architect)
- Critique-and-revise loops within tasks
- Approval workflows for critical decisions
- Escalation paths for unresolved issues

### Supervision Hierarchy

```
┌─────────────────────────────────────────┐
│            ARCHITECT AGENTS              │
│         (Final approval authority)       │
│    tech_lead, security_reviewer          │
└─────────────────┬───────────────────────┘
                  │ supervises
                  ▼
┌─────────────────────────────────────────┐
│            LEAD AGENTS                   │
│      (Review and guide work)            │
│    code_reviewer, project_manager        │
└─────────────────┬───────────────────────┘
                  │ supervises
                  ▼
┌─────────────────────────────────────────┐
│          EXECUTION AGENTS                │
│        (Produce artifacts)              │
│  backend, frontend, database, ux         │
└─────────────────────────────────────────┘
```

### Usage

```python
from orchestrator.core import get_supervision_manager, SupervisedTaskExecutor

supervisor = get_supervision_manager()
executor = SupervisedTaskExecutor(supervisor)

# Execute with supervision
result = await executor.execute_with_supervision(
    db=db,
    task=task,
    agent=backend_agent,
    max_critique_rounds=3
)

# Request explicit approval
approval = await supervisor.request_approval(
    db=db,
    task_id=task.id,
    artifact=generated_code,
    approver_role="tech_lead"
)
```

---

## 8. Semantic Validation

**Module:** `orchestrator/core/semantic_validator.py`

LLM-powered validation of artifact quality and consistency.

### Features
- Per-artifact-type validation criteria
- Cross-artifact consistency checks
- Severity-based issue reporting
- Actionable fix suggestions

### Usage

```python
from orchestrator.core import get_semantic_validator

validator = get_semantic_validator()

# Validate a single artifact
result = await validator.validate_artifact(
    artifact_type="code",
    content=generated_code,
    context={"language": "python", "requirements": requirements_doc}
)

for issue in result.issues:
    print(f"[{issue.severity}] {issue.message}")
    print(f"  Suggestion: {issue.suggestion}")

# Check cross-artifact consistency
consistency = await validator.check_cross_artifact_consistency(
    db=db,
    run_id=run_id
)
```

### Validation Criteria

| Artifact Type | Validation Checks |
|--------------|-------------------|
| Requirements | Completeness, clarity, testability |
| Architecture | Consistency, scalability, security |
| Code | Style, correctness, error handling |
| Tests | Coverage, edge cases, assertions |
| Security | OWASP top 10, authentication, authorization |

---

## 9. Observability Dashboard

**Module:** `orchestrator/core/observability.py`

Comprehensive metrics collection with Prometheus export.

### Features
- Histogram metrics for latencies
- Gauge metrics for queue depths
- Counter metrics for throughput
- Bottleneck analysis
- Prometheus-compatible export

### Usage

```python
from orchestrator.core import get_metrics_collector, DashboardDataProvider

metrics = get_metrics_collector()

# Record task execution
metrics.record_task_duration("backend_engineer", 45.2)
metrics.record_llm_call("claude-sonnet-4", 1500, 800, 0.023)

# Get dashboard data
dashboard = DashboardDataProvider(metrics)
data = await dashboard.get_dashboard_data(db)

# Export for Prometheus
prometheus_output = dashboard.export_prometheus()
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `task_duration_seconds` | Histogram | Task execution time by role |
| `llm_tokens_total` | Counter | Total tokens used |
| `llm_cost_dollars` | Counter | Total LLM spend |
| `queue_depth` | Gauge | Tasks waiting per queue |
| `active_runs` | Gauge | Currently executing runs |
| `gate_pass_rate` | Gauge | Quality gate pass percentage |

---

## 10. Human-in-the-Loop Escalation

**Module:** `orchestrator/core/escalation.py`

Tiered escalation system with non-blocking human input.

### Features
- Three-tier escalation (Agent → Lead Agent → Human)
- Timeout-based auto-decisions with conservative defaults
- Partial approvals with conditions
- Slack and email notifications
- Non-blocking escalation (work continues)

### Escalation Flow

```
┌──────────────┐     timeout      ┌──────────────┐     timeout      ┌──────────────┐
│    AGENT     │ ──────────────►  │  LEAD AGENT  │ ──────────────►  │    HUMAN     │
│   (5 min)    │                  │   (30 min)   │                  │   (1 hour)   │
└──────────────┘                  └──────────────┘                  └──────────────┘
       │                                 │                                 │
       ▼                                 ▼                                 ▼
   Can approve                      Can approve                      Can approve
   or escalate                      or escalate                    or auto-decide
```

### Usage

```python
from orchestrator.core import (
    get_escalation_manager,
    request_gate_approval,
    request_budget_override,
    EscalationPriority
)

escalation_mgr = get_escalation_manager()

# Request gate approval
escalation = await request_gate_approval(
    db=db,
    run_id=run_id,
    task_id=task_id,
    gate_type="security_review",
    gate_result={"issues": security_findings},
    priority=EscalationPriority.HIGH
)

# Request budget override
escalation = await request_budget_override(
    db=db,
    run_id=run_id,
    current_spend=45.00,
    requested_amount=20.00,
    budget_limit=50.00,
    reason="Complex implementation requires additional iterations"
)

# Decide an escalation (from webhook/UI)
decision = await escalation_mgr.decide_escalation(
    db=db,
    escalation_id=escalation.id,
    status=EscalationStatus.APPROVED_WITH_CONDITIONS,
    decision="approved",
    decided_by="admin@company.com",
    conditions=["Add rate limiting", "Review again after changes"]
)
```

### Escalation Types

| Type | Description | Default Auto-Decision |
|------|-------------|----------------------|
| `gate_approval` | Quality gate failed | Reject |
| `budget_override` | Cost limit exceeded | Reject |
| `security_review` | Security findings | Reject |
| `error_resolution` | Execution error | Retry |
| `ambiguous_requirement` | Unclear instructions | Pause |
| `quality_concern` | Output quality issue | Reject |
| `sensitive_data` | PII/secrets detected | Reject |

### Notification Configuration

```python
from orchestrator.core.escalation import EscalationConfig

config = EscalationConfig(
    slack_webhook_url="https://hooks.slack.com/...",
    slack_channel="#orchestrator-alerts",
    email_recipients=["team@company.com"],
    smtp_host="smtp.gmail.com",
    critical_always_human=True,
    auto_escalate_on_timeout=True
)
```

---

## Integration Example

Here's how these features work together in a production workflow:

```python
from orchestrator.core import (
    get_checkpoint_manager,
    get_cost_predictor,
    get_escalation_manager,
    get_supervision_manager,
    get_semantic_validator,
    get_metrics_collector,
)

async def execute_production_workflow(run_id: UUID, goal: str):
    # 1. Predict costs before starting
    predictor = get_cost_predictor()
    forecast = await predictor.forecast_run_cost(goal, db)

    if forecast.total_estimated > budget_limit:
        await request_budget_override(db, run_id, ...)

    # 2. Execute with supervision and checkpoints
    checkpoints = get_checkpoint_manager()
    supervisor = get_supervision_manager()
    metrics = get_metrics_collector()

    for task in workflow_tasks:
        # Create checkpoint before major phases
        if task.is_phase_start:
            await checkpoints.create_checkpoint(db, run_id, task.phase)

        # Execute with supervision
        start_time = time.time()
        result = await supervisor.execute_with_supervision(db, task)
        metrics.record_task_duration(task.role, time.time() - start_time)

        # Validate artifacts
        validator = get_semantic_validator()
        validation = await validator.validate_artifact(
            task.artifact_type,
            result.artifact
        )

        if validation.has_critical_issues:
            await get_escalation_manager().create_escalation(
                db, run_id,
                EscalationType.QUALITY_CONCERN,
                title=f"Quality issues in {task.role} output",
                description=str(validation.issues),
                priority=EscalationPriority.HIGH
            )
```

---

## Configuration

All features can be configured via environment variables or the settings module:

```python
# orchestrator/core/config.py

class Settings:
    # DLQ
    DLQ_RETENTION_DAYS: int = 30
    DLQ_AUTO_REPLAY_ENABLED: bool = False

    # Context Management
    MAX_CONTEXT_TOKENS: int = 100000
    CONTEXT_COMPACTION_THRESHOLD: float = 0.8

    # Checkpoints
    AUTO_CHECKPOINT_INTERVAL: int = 300  # seconds
    CHECKPOINT_COMPRESSION: bool = True

    # Cost Prediction
    COST_WARNING_THRESHOLD: float = 0.8
    COST_BLOCK_THRESHOLD: float = 1.0

    # Supervision
    MAX_CRITIQUE_ROUNDS: int = 3
    REQUIRE_LEAD_APPROVAL: bool = True

    # Escalation
    ESCALATION_SLACK_ENABLED: bool = True
    ESCALATION_EMAIL_ENABLED: bool = False
    CRITICAL_AUTO_ESCALATE_TO_HUMAN: bool = True
```
