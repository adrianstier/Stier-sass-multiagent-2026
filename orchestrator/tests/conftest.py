"""Pytest fixtures for orchestrator tests."""

import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from orchestrator.core.models import Base
from orchestrator.core.database import engine, SessionLocal


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
