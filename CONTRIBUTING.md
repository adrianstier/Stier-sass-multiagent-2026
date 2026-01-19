# Contributing to Multi-Agent Orchestration Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/adrianstier/Stier-sass-multiagent-2026.git
   cd Stier-sass-multiagent-2026
   ```

2. **Install dependencies**
   ```bash
   cd orchestrator
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Start infrastructure services**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d postgres redis
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your ANTHROPIC_API_KEY
   ```

6. **Initialize database**
   ```bash
   orchestrator init-db
   ```

## Code Style

We use the following tools to maintain code quality:

- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **bandit** - Security vulnerability scanning

Run all checks:
```bash
# Lint
ruff check orchestrator/

# Format
ruff format orchestrator/

# Type check
mypy orchestrator/ --ignore-missing-imports

# Security scan
bandit -r orchestrator/ -ll
```

## Testing

### Running Tests

```bash
cd orchestrator
python -m pytest tests/ -v
```

### With Coverage

```bash
python -m pytest tests/ -v --cov=orchestrator --cov-report=html
```

### Test Categories

- `tests/test_api.py` - API endpoint tests
- `tests/test_auth.py` - Authentication tests
- `tests/test_tools.py` - Tool registry tests
- `tests/test_task_dsl.py` - Task DSL tests
- `tests/test_ralph_wiggum.py` - Ralph Wiggum loop tests

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linting**
   ```bash
   python -m pytest tests/ -v
   ruff check orchestrator/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: description of your change"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation only
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `test:` - Adding tests
   - `chore:` - Maintenance tasks

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Describe your changes** in the PR description:
   - What problem does this solve?
   - How was it tested?
   - Any breaking changes?

## Adding New Agents

1. Create the agent class in `agents/specialists.py`:
   ```python
   class MyNewAgent(BaseAgent):
       role = "my_new_role"

       def get_system_prompt(self) -> str:
           return "You are..."
   ```

2. Register in `AGENT_REGISTRY`

3. Add queue mapping in `core/config.py`

4. Add tool allowlist in `tools/registry.py`

5. Add tests in `tests/`

## Adding New Tools

1. Define the tool function:
   ```python
   def my_tool(param1: str, param2: int) -> dict:
       # Implementation
       return {"result": "..."}
   ```

2. Register with `register_tool()` in `tools/registry.py`

3. Add to appropriate role allowlists

4. Add tests

## Reporting Issues

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Questions?

Feel free to open an issue for questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
