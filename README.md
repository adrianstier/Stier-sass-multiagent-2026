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
