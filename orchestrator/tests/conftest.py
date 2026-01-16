"""Pytest configuration and fixtures."""

import os
import pytest
import uuid
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

# Set test environment before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["LOG_FORMAT"] = "text"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from orchestrator.core.models import Base
from orchestrator.core.database import engine, SessionLocal
from orchestrator.core.auth import AuthContext, create_access_token


@pytest.fixture(scope="session")
def db_engine():
    """Create test database engine."""
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a fresh database session for each test."""
    connection = db_engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_goal():
    """Sample goal for testing."""
    return "Build a simple CRUD application for task management"


@pytest.fixture(scope="session")
def test_org_id() -> str:
    """Generate a test organization ID."""
    return str(uuid.uuid4())


@pytest.fixture(scope="session")
def test_user_id() -> str:
    """Generate a test user ID."""
    return str(uuid.uuid4())


@pytest.fixture
def auth_context(test_org_id: str, test_user_id: str) -> AuthContext:
    """Create an auth context for testing."""
    return AuthContext(
        user_id=test_user_id,
        organization_id=test_org_id,
        role="admin",
        scopes=["*"],
        is_api_key=False,
    )


@pytest.fixture
def auth_headers(test_org_id: str, test_user_id: str) -> dict:
    """Create authorization headers for testing."""
    token = create_access_token(test_user_id, test_org_id, "admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key_headers() -> dict:
    """Create API key headers for testing."""
    return {"X-API-Key": "orch_live_test_key_for_testing"}


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    from orchestrator.api.main import app
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    mock_session = MagicMock()
    mock_session.query.return_value = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    mock_session.execute = MagicMock()
    return mock_session


@pytest.fixture
def mock_run(test_org_id: str) -> MagicMock:
    """Create a mock run object."""
    from orchestrator.core.models import RunStatus, GateStatus

    run = MagicMock()
    run.id = uuid.uuid4()
    run.organization_id = uuid.UUID(test_org_id)
    run.goal = "Test goal"
    run.status = RunStatus.RUNNING
    run.current_phase = "planning"
    run.current_iteration = 1
    run.max_iterations = 50
    run.tokens_used = 1000
    run.budget_tokens = 100000
    run.total_cost_usd = "0.05"
    run.code_review_status = GateStatus.PENDING
    run.security_review_status = GateStatus.PENDING
    run.created_at = datetime.utcnow()
    run.started_at = datetime.utcnow()
    run.completed_at = None
    run.blocked_reason = None
    return run


@pytest.fixture
def mock_task() -> MagicMock:
    """Create a mock task object."""
    from orchestrator.core.models import TaskStatus

    task = MagicMock()
    task.id = uuid.uuid4()
    task.run_id = uuid.uuid4()
    task.task_type = "requirements_analysis"
    task.assigned_role = "business_analyst"
    task.description = "Analyze requirements"
    task.status = TaskStatus.PENDING
    task.priority = 10
    task.retry_count = 0
    task.created_at = datetime.utcnow()
    task.started_at = None
    task.completed_at = None
    task.error_message = None
    return task


@pytest.fixture
def mock_artifact() -> MagicMock:
    """Create a mock artifact object."""
    artifact = MagicMock()
    artifact.id = uuid.uuid4()
    artifact.run_id = uuid.uuid4()
    artifact.artifact_type = "requirements_document"
    artifact.name = "test_artifact"
    artifact.content_type = "text/markdown"
    artifact.content = "# Test Content"
    artifact.artifact_metadata = {}
    artifact.produced_by = "business_analyst"
    artifact.created_at = datetime.utcnow()
    return artifact


@pytest.fixture
def mock_event() -> MagicMock:
    """Create a mock event object."""
    event = MagicMock()
    event.id = uuid.uuid4()
    event.run_id = uuid.uuid4()
    event.event_type = "task_started"
    event.actor = "business_analyst"
    event.data = {"input": {}}
    event.timestamp = datetime.utcnow()
    return event


@pytest.fixture
def mock_celery_task():
    """Mock Celery task result."""
    mock_result = MagicMock()
    mock_result.id = str(uuid.uuid4())
    mock_result.status = "PENDING"
    return mock_result


# Helper function for patching database
def patch_db(mock_session):
    """Context manager to patch database session."""
    return patch(
        "orchestrator.core.database.get_db",
        return_value=MagicMock(
            __enter__=MagicMock(return_value=mock_session),
            __exit__=MagicMock(return_value=None),
        )
    )


# Helper function for patching authentication
def patch_auth(auth_context: AuthContext):
    """Context manager to patch authentication."""
    return patch(
        "orchestrator.api.main.get_current_auth",
        return_value=auth_context
    )
