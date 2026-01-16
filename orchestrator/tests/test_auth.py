"""Tests for authentication module."""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from orchestrator.core.auth import (
    AuthContext,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    verify_api_key,
    TokenPayload,
)


class TestJWTTokens:
    """Tests for JWT token functions."""

    def test_create_access_token_returns_string(self):
        """Access token should be a string."""
        token = create_access_token("user-123", "org-456", "admin")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_access_token(self):
        """Should be able to decode a valid access token."""
        user_id = "user-123"
        org_id = "org-456"
        role = "admin"

        token = create_access_token(user_id, org_id, role)
        payload = decode_token(token)

        assert payload.sub == user_id
        assert payload.org_id == org_id
        assert payload.role == role
        assert payload.type == "access"

    def test_create_refresh_token(self):
        """Refresh token should be decodable."""
        user_id = "user-123"
        org_id = "org-456"

        token = create_refresh_token(user_id, org_id)
        payload = decode_token(token)

        assert payload.sub == user_id
        assert payload.org_id == org_id
        assert payload.type == "refresh"

    def test_decode_invalid_token_raises_exception(self):
        """Invalid tokens should raise HTTPException."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid-token")

        assert exc_info.value.status_code == 401


class TestAPIKeys:
    """Tests for API key functions."""

    def test_generate_api_key_returns_tuple(self):
        """Generate should return (full_key, prefix, hash)."""
        full_key, prefix, key_hash = generate_api_key()

        assert isinstance(full_key, str)
        assert isinstance(prefix, str)
        assert isinstance(key_hash, str)

    def test_api_key_starts_with_prefix(self):
        """API key should start with 'orch_live'."""
        full_key, prefix, _ = generate_api_key()

        assert full_key.startswith("orch_live_")
        assert prefix == "orch_live"

    def test_verify_api_key_valid(self):
        """Valid API key should verify successfully."""
        full_key, _, key_hash = generate_api_key()

        assert verify_api_key(full_key, key_hash) is True

    def test_verify_api_key_invalid(self):
        """Invalid API key should not verify."""
        _, _, key_hash = generate_api_key()

        assert verify_api_key("wrong_key", key_hash) is False

    def test_api_keys_are_unique(self):
        """Each generated key should be unique."""
        keys = [generate_api_key()[0] for _ in range(10)]
        assert len(set(keys)) == 10


class TestAuthContext:
    """Tests for AuthContext class."""

    def test_has_scope_with_wildcard(self):
        """Wildcard scope should match any scope."""
        ctx = AuthContext(
            organization_id="org-123",
            scopes=["*"],
        )

        assert ctx.has_scope("runs:read") is True
        assert ctx.has_scope("anything:else") is True

    def test_has_scope_specific(self):
        """Should only match specific scopes."""
        ctx = AuthContext(
            organization_id="org-123",
            scopes=["runs:read", "artifacts:read"],
        )

        assert ctx.has_scope("runs:read") is True
        assert ctx.has_scope("artifacts:read") is True
        assert ctx.has_scope("runs:write") is False

    def test_auth_context_defaults(self):
        """Default values should be set correctly."""
        ctx = AuthContext(organization_id="org-123")

        assert ctx.user_id is None
        assert ctx.role == "member"
        assert ctx.scopes == ["*"]
        assert ctx.is_api_key is False


class TestTokenExpiration:
    """Tests for token expiration."""

    def test_access_token_has_expiration(self):
        """Access token should have expiration time."""
        token = create_access_token("user", "org", "admin")
        payload = decode_token(token)

        assert payload.exp is not None
        assert payload.exp > datetime.utcnow()

    def test_refresh_token_expires_later(self):
        """Refresh token should expire later than access token."""
        access = create_access_token("user", "org", "admin")
        refresh = create_refresh_token("user", "org")

        access_payload = decode_token(access)
        refresh_payload = decode_token(refresh)

        assert refresh_payload.exp > access_payload.exp
