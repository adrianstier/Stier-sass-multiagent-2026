.PHONY: help install dev-install docker-up docker-down init-db run-api run-workers test lint clean demo

help:
	@echo "Multi-Agent Orchestrator - Available Commands:"
	@echo ""
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install development dependencies"
	@echo "  make docker-up      Start all Docker services"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make init-db        Initialize the database"
	@echo "  make run-api        Start the FastAPI server"
	@echo "  make run-workers    Start all Celery workers"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linters"
	@echo "  make clean          Clean up temporary files"
	@echo "  make demo           Run the demo workflow"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@echo "Services started. API: http://localhost:8000, Flower: http://localhost:5555"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

init-db:
	python -c "from orchestrator.core.database import init_db; init_db()"
	@echo "Database initialized successfully!"

run-api:
	uvicorn orchestrator.api.main:app --reload --host 0.0.0.0 --port 8000

run-workers:
	python -m orchestrator.workers.entrypoint all

run-worker-%:
	python -m orchestrator.workers.entrypoint $*

test:
	pytest tests/ -v --cov=orchestrator --cov-report=term-missing

lint:
	ruff check orchestrator/
	mypy orchestrator/ --ignore-missing-imports

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo:
	@echo "Running demo workflow..."
	python -m orchestrator.cli.main run "Build a simple CRUD app for managing TODO items with REST API"
