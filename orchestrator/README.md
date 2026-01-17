# Multi-Agent Orchestration System

A production-grade multi-agent workflow orchestration system that coordinates specialized AI agents through the software development lifecycle.

## Overview

This system implements a functional multi-agent workflow using Celery, where an Orchestrator decomposes user goals into role-based task DAGs and coordinates specialist agents:

- **Business Analyst** - Requirements gathering and stakeholder analysis
- **Project Manager** - Planning and coordination
- **UX Engineer** - User experience design
- **Tech Lead** - Technical architecture and guidance
- **Database Engineer** - Database design and optimization
- **Backend Engineer** - Server-side implementation
- **Frontend Engineer** - Client-side implementation
- **Code Reviewer** - Quality gate for code quality
- **Security Reviewer** - Final quality gate for security

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI / CLI                            │
│                      (Submit Goals, Check Status)                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PostgreSQL Database                         │
│              (Runs, Tasks, Events, Artifacts)                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Redis Broker                             │
│                    (Celery Task Queues)                         │
└─────────────────────────────────────────────────────────────────┘
          │         │         │         │         │
          ▼         ▼         ▼         ▼         ▼
       ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
       │q_ba │  │q_pm │  │q_tl │  │q_be │  │q_sec│
       │     │  │     │  │     │  │     │  │     │
       │ BA  │  │ PM  │  │ TL  │  │ BE  │  │ SEC │
       │Worker│ │Worker│ │Worker│ │Worker│ │Worker│
       └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
```

## Key Features

### 1. Asynchronous Autonomy
- Runs end-to-end without human intervention
- Persists state and is fully resumable
- Halts in `NEEDS_INPUT` state when blocked

### 2. Durable State Model
- Append-only event log for full traceability
- Tables: `runs`, `tasks`, `events`, `artifacts`
- All agent prompts/responses logged

### 3. Role-Based Queue Architecture
- Separate Celery queue per role (q_ba, q_pm, q_ux, etc.)
- Each role has dedicated worker(s)
- Orchestrator dispatches tasks to appropriate queues

### 4. Tool Boundaries
- No agent performs privileged actions directly
- Tool executor abstraction with role-based allowlists
- Full audit trail of tool usage

### 5. Quality Gates
- Code Review must pass before Security Review
- Security Review must pass before completion
- Gates can be PASSED, FAILED, or WAIVED

### 6. Task DSL
- Lightweight DSL for task specification
- Dependencies, expected artifacts, validation methods
- Idempotent tasks with dedupe keys

### 7. Advanced Features (NEW)

The system includes 10 advanced orchestration capabilities:

| Feature | Module | Description |
|---------|--------|-------------|
| Dead Letter Queue | `core/dlq.py` | Failed task persistence with replay |
| Context Management | `core/context_manager.py` | Token estimation and summarization |
| Dynamic Workflows | `core/workflow_modifier.py` | Runtime DAG modification |
| Agent Channels | `core/channels.py` | Redis pub/sub messaging |
| Checkpoints | `core/checkpoint.py` | State persistence and recovery |
| Cost Prediction | `core/cost_predictor.py` | Historical cost estimation |
| Supervision | `core/supervision.py` | Hierarchical agent oversight |
| Semantic Validation | `core/semantic_validator.py` | LLM-based quality checks |
| Observability | `core/observability.py` | Metrics and Prometheus export |
| Escalation | `core/escalation.py` | Human-in-the-loop with notifications |

See [Advanced Features Documentation](../docs/advanced-features.md) for detailed usage.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Anthropic API key (for LLM calls)

### Option 1: Docker Compose (Recommended)

```bash
# Clone and navigate to orchestrator directory
cd orchestrator

# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key
nano .env

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Initialize database
docker-compose exec api python -c "from orchestrator.core.database import init_db; init_db()"
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start PostgreSQL and Redis (using Docker)
docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15-alpine
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
python -c "from orchestrator.core.database import init_db; init_db()"

# Start workers (in separate terminals)
python -m orchestrator.workers.entrypoint all

# Start API
uvicorn orchestrator.api.main:app --reload
```

## Usage

### CLI

```bash
# Initialize database
orchestrator init-db

# Start a new run
orchestrator run "Build a simple CRUD application for task management"

# Check status
orchestrator status <run_id>

# List all runs
orchestrator list

# View events
orchestrator events <run_id>

# View artifacts
orchestrator artifacts <run_id> --content

# Manually trigger orchestrator tick
orchestrator tick <run_id>
```

### API

```bash
# Create a new run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"goal": "Build a simple CRUD application for task management"}'

# Get run status
curl http://localhost:8000/runs/<run_id>/status

# List tasks
curl http://localhost:8000/runs/<run_id>/tasks

# View events
curl http://localhost:8000/runs/<run_id>/events

# View artifacts
curl http://localhost:8000/runs/<run_id>/artifacts

# Health check
curl http://localhost:8000/health
```

### Monitoring

- **Flower (Celery monitoring)**: http://localhost:5555
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

## Demo: "Hello, Multi-Agent"

Run the built-in demo:

```bash
# Using CLI
orchestrator run "Build a small CRUD app for managing TODO items"

# Watch the workflow progress
watch orchestrator status <run_id>

# View generated artifacts
orchestrator artifacts <run_id> --content
```

The workflow will:
1. **BA** analyzes requirements, creates requirements document
2. **PM** creates project plan and timeline
3. **UX** designs user flows and wireframes
4. **Tech Lead** defines architecture and tech stack
5. **DB Engineer** designs database schema
6. **Backend/Frontend** implement the solution (in parallel)
7. **Code Reviewer** validates code quality (GATE)
8. **Security Reviewer** validates security (FINAL GATE)
9. Run completes only if both gates pass

## Project Structure

```
orchestrator/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py           # Base agent class
│   ├── orchestrator.py   # Orchestrator logic
│   ├── specialists.py    # All specialist agents
│   └── tasks.py          # Celery task definitions
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI application
├── cli/
│   ├── __init__.py
│   └── main.py           # CLI commands
├── core/
│   ├── __init__.py
│   ├── celery_app.py     # Celery configuration
│   ├── config.py         # Settings
│   ├── database.py       # Database session
│   ├── models.py         # SQLAlchemy models
│   ├── task_dsl.py       # Task DSL definitions
│   ├── dlq.py            # Dead Letter Queue
│   ├── context_manager.py # Context window management
│   ├── workflow_modifier.py # Dynamic workflow modification
│   ├── channels.py       # Agent collaboration channels
│   ├── checkpoint.py     # Checkpoint/resume system
│   ├── cost_predictor.py # Cost prediction
│   ├── supervision.py    # Hierarchical supervision
│   ├── semantic_validator.py # Semantic validation
│   ├── observability.py  # Metrics and monitoring
│   └── escalation.py     # Human-in-the-loop escalation
├── tools/
│   ├── __init__.py
│   ├── executor.py       # Tool execution engine
│   └── registry.py       # Tool registry & allowlists
├── workers/
│   ├── __init__.py
│   └── entrypoint.py     # Worker entrypoints
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Data Model

### Run
- Represents a complete orchestration execution
- Contains goal, context, status, quality gate states
- Has success/acceptance criteria

### Task
- Unit of work assigned to a specific agent role
- Contains dependencies, expected artifacts, validation
- Supports idempotency and retries

### Event
- Append-only log for full traceability
- Records: task lifecycle, LLM calls, tool usage, state changes

### Artifact
- Outputs produced by agents
- Types: requirements, plans, designs, schemas, code, reviews
- Versioned with metadata

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/orchestrator` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `LLM_MODEL` | Claude model to use | `claude-sonnet-4-20250514` |
| `MAX_ITERATIONS` | Maximum orchestrator iterations | `50` |
| `TASK_TIMEOUT_SECONDS` | Task execution timeout | `300` |
| `REQUIRE_CODE_REVIEW` | Enforce code review gate | `true` |
| `REQUIRE_SECURITY_REVIEW` | Enforce security review gate | `true` |

## Extending the System

### Adding a New Agent Role

1. Create agent class in `agents/specialists.py`:
```python
class MyNewAgent(BaseAgent):
    role = "my_new_role"

    def get_system_prompt(self) -> str:
        return "You are..."
```

2. Register in `AGENT_REGISTRY`

3. Add queue mapping in `core/config.py`:
```python
ROLE_QUEUES = {
    ...
    "my_new_role": "q_mynew",
}
```

4. Add tool allowlist in `tools/registry.py`

5. Add worker in `docker-compose.yml`

### Adding New Tools

```python
from orchestrator.tools.registry import register_tool

def my_tool(param1: str, param2: int) -> dict:
    # Implementation
    return {"result": "..."}

register_tool(
    name="my_tool",
    function=my_tool,
    description="What this tool does",
    parameters={...}
)

# Add to role allowlists
ROLE_ALLOWLISTS["my_role"].append("my_tool")
```

## Troubleshooting

### Workers not picking up tasks
```bash
# Check worker status
docker-compose logs worker-ba

# Verify queues
docker-compose exec redis redis-cli KEYS "celery*"
```

### Database connection issues
```bash
# Check PostgreSQL
docker-compose exec postgres psql -U postgres -c "SELECT 1"

# Reinitialize
docker-compose exec api python -c "from orchestrator.core.database import init_db; init_db()"
```

### Run stuck in NEEDS_INPUT
```bash
# Check run status
orchestrator status <run_id>

# Check events for blocked reason
orchestrator events <run_id> -v

# Manually trigger tick
orchestrator tick <run_id>
```

## License

MIT License - See LICENSE file for details.
