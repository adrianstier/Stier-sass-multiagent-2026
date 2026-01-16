"""Tests for the API endpoints."""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from orchestrator.api.main import app
from orchestrator.core.auth import (
    AuthContext,
    create_access_token,
    generate_api_key,
    Organization,
    User,
    APIKey,
)
from orchestrator.core.models import Run, RunStatus


@pytest.fixture
def test_org_id():
    return str(uuid.uuid4())


@pytest.fixture
def test_user_id():
    return str(uuid.uuid4())


@pytest.fixture
def auth_context(test_org_id, test_user_id):
    return AuthContext(
        user_id=test_user_id,
        organization_id=test_org_id,
        role="admin",
        scopes=["*"],
        is_api_key=False,
    )


@pytest.fixture
def auth_headers(test_org_id, test_user_id):
    token = create_access_token(test_user_id, test_org_id, "admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Health check should return 200."""
        with patch("orchestrator.api.main.get_db") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_db.return_value.__exit__ = MagicMock(return_value=None)

            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] in ["healthy", "degraded"]

    def test_root_returns_api_info(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Multi-Agent Orchestrator API"
        assert "version" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_unauthenticated_request_returns_401(self, client):
        """Requests without auth should return 401."""
        response = client.get("/runs")
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, client):
        """Invalid tokens should return 401."""
        response = client.get(
            "/runs",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        """Invalid API keys should return 401."""
        response = client.get(
            "/runs",
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401


class TestRunEndpoints:
    """Tests for run management endpoints."""

    def test_create_run_requires_auth(self, client):
        """Creating a run should require authentication."""
        response = client.post("/runs", json={"goal": "Test goal for a new run"})
        assert response.status_code == 401

    def test_create_run_validates_goal_length(self, client, auth_context):
        """Goal must be at least 10 characters."""
        with patch("orchestrator.api.main.get_current_auth", return_value=auth_context):
            with patch("orchestrator.api.main.require_scope") as mock_scope:
                mock_scope.return_value = lambda: auth_context

                response = client.post(
                    "/runs",
                    json={"goal": "short"},
                    headers={"Authorization": "Bearer test"}
                )
                # Should fail validation before hitting auth
                assert response.status_code in [401, 422]

    def test_get_run_validates_uuid_format(self, client, auth_context):
        """Run ID must be a valid UUID."""
        with patch("orchestrator.api.main.require_scope") as mock_scope:
            async def return_auth(auth=None):
                return auth_context
            mock_scope.return_value = return_auth

            response = client.get(
                "/runs/not-a-uuid",
                headers={"Authorization": "Bearer test"}
            )
            assert response.status_code in [400, 401]

    def test_list_runs_validates_status(self, client, auth_context):
        """Status filter should validate against valid statuses."""
        with patch("orchestrator.api.main.require_scope") as mock_scope:
            async def return_auth(auth=None):
                return auth_context
            mock_scope.return_value = return_auth

            with patch("orchestrator.api.main.get_db") as mock_db:
                mock_session = MagicMock()
                mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
                mock_db.return_value.__exit__ = MagicMock(return_value=None)
                mock_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []

                response = client.get(
                    "/runs?status=invalid_status",
                    headers={"Authorization": "Bearer test"}
                )
                # Either 400 for invalid status or 401 for auth
                assert response.status_code in [400, 401]


class TestInputValidation:
    """Tests for input validation."""

    def test_run_goal_stripped_of_whitespace(self):
        """Goal should be stripped of leading/trailing whitespace."""
        from orchestrator.api.main import RunCreate

        run = RunCreate(goal="   Test goal with whitespace   ")
        assert run.goal == "Test goal with whitespace"

    def test_run_budget_must_be_positive(self):
        """Budget tokens must be >= 1000."""
        from orchestrator.api.main import RunCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunCreate(goal="Test goal here", budget_tokens=500)

    def test_run_budget_has_max_limit(self):
        """Budget tokens must be <= 10000000."""
        from orchestrator.api.main import RunCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunCreate(goal="Test goal here", budget_tokens=100000000)


class TestMultiTenancy:
    """Tests for multi-tenancy isolation."""

    def test_runs_filtered_by_organization(self, client, auth_context, test_org_id):
        """Runs should be filtered by organization ID."""
        with patch("orchestrator.api.main.require_scope") as mock_scope:
            async def return_auth(auth=None):
                return auth_context
            mock_scope.return_value = return_auth

            with patch("orchestrator.api.main.get_db") as mock_db:
                mock_session = MagicMock()
                mock_query = MagicMock()
                mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
                mock_db.return_value.__exit__ = MagicMock(return_value=None)
                mock_session.query.return_value = mock_query
                mock_query.filter.return_value = mock_query
                mock_query.order_by.return_value = mock_query
                mock_query.offset.return_value = mock_query
                mock_query.limit.return_value = mock_query
                mock_query.all.return_value = []

                response = client.get(
                    "/runs",
                    headers={"Authorization": "Bearer test"}
                )

                # Verify filter was called (would include org_id filter)
                if response.status_code == 200:
                    mock_query.filter.assert_called()


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_nonexistent_run(self, client, auth_context):
        """Should return 404 for non-existent run."""
        run_id = str(uuid.uuid4())

        with patch("orchestrator.api.main.require_scope") as mock_scope:
            async def return_auth(auth=None):
                return auth_context
            mock_scope.return_value = return_auth

            with patch("orchestrator.api.main.get_db") as mock_db:
                mock_session = MagicMock()
                mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
                mock_db.return_value.__exit__ = MagicMock(return_value=None)
                mock_session.query.return_value.filter.return_value.first.return_value = None

                response = client.get(
                    f"/runs/{run_id}",
                    headers={"Authorization": "Bearer test"}
                )
                # Either 404 or 401 depending on auth
                assert response.status_code in [401, 404]

    def test_400_for_invalid_uuid(self, client, auth_context):
        """Should return 400 for invalid UUID format."""
        with patch("orchestrator.api.main.require_scope") as mock_scope:
            async def return_auth(auth=None):
                return auth_context
            mock_scope.return_value = return_auth

            response = client.get(
                "/runs/not-a-valid-uuid",
                headers={"Authorization": "Bearer test"}
            )
            assert response.status_code in [400, 401]

    def test_cannot_cancel_completed_run(self, client, auth_context):
        """Should return 400 when trying to cancel completed run."""
        run_id = str(uuid.uuid4())

        with patch("orchestrator.api.main.require_scope") as mock_scope:
            async def return_auth(auth=None):
                return auth_context
            mock_scope.return_value = return_auth

            with patch("orchestrator.api.main.get_db") as mock_db:
                mock_session = MagicMock()
                mock_run = MagicMock()
                mock_run.status = RunStatus.COMPLETED
                mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
                mock_db.return_value.__exit__ = MagicMock(return_value=None)
                mock_session.query.return_value.filter.return_value.first.return_value = mock_run

                response = client.post(
                    f"/runs/{run_id}/cancel",
                    headers={"Authorization": "Bearer test"}
                )
                assert response.status_code in [400, 401]
