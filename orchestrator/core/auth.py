"""Authentication and authorization module.

Implements:
- API key authentication
- JWT token authentication
- Multi-tenant organization context
- Role-based access control
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import UUID, uuid4

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from orchestrator.core.config import settings
from orchestrator.core.models import Base


# =============================================================================
# Auth Models
# =============================================================================

class Organization(Base):
    """Organization/tenant for multi-tenancy."""

    __tablename__ = "organizations"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)

    # Billing/Limits
    plan = Column(String(50), default="free")  # free, starter, pro, enterprise
    monthly_token_limit = Column(Integer, default=100000)
    tokens_used_this_month = Column(Integer, default=0)
    billing_cycle_start = Column(DateTime, default=datetime.utcnow)

    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="organization")
    api_keys = relationship("APIKey", back_populates="organization")


class User(Base):
    """User account within an organization."""

    __tablename__ = "users"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)

    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)  # Null for API-only users

    # Role within organization
    role = Column(String(50), default="member")  # owner, admin, member, viewer

    # Status
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="users")


class APIKey(Base):
    """API key for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    organization_id = Column(PGUUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by_user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    name = Column(String(255), nullable=False)  # e.g., "Production API Key"
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for identification
    key_hash = Column(String(255), nullable=False)  # SHA-256 hash of full key

    # Permissions
    scopes = Column(Text, default="*")  # Comma-separated: "runs:read,runs:write,artifacts:read"

    # Rate limits (per minute)
    rate_limit = Column(Integer, default=60)

    # Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="api_keys")


# =============================================================================
# Pydantic Models
# =============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # User ID
    org_id: str  # Organization ID
    role: Optional[str] = None  # Optional for refresh tokens
    exp: datetime
    type: str = "access"


class AuthContext(BaseModel):
    """Current authentication context passed to endpoints."""
    user_id: Optional[str] = None
    organization_id: str
    role: str = "member"
    scopes: list[str] = ["*"]
    is_api_key: bool = False

    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        if "*" in self.scopes:
            return True
        return scope in self.scopes


# =============================================================================
# JWT Functions
# =============================================================================

JWT_SECRET = settings.jwt_secret_key
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_access_token(user_id: str, org_id: str, role: str) -> str:
    """Create a JWT access token."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "org_id": org_id,
        "role": role,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, org_id: str) -> str:
    """Create a JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "org_id": org_id,
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# API Key Functions
# =============================================================================

def generate_api_key() -> Tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        Tuple of (full_key, prefix, hash)
    """
    # Generate a secure random key: orch_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    prefix = "orch_live"
    random_part = secrets.token_urlsafe(32)
    full_key = f"{prefix}_{random_part}"

    # Create hash for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    return full_key, prefix, key_hash


def verify_api_key(key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash."""
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return secrets.compare_digest(key_hash, stored_hash)


# =============================================================================
# FastAPI Security Dependencies
# =============================================================================

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[AuthContext]:
    """Validate API key and return auth context."""
    if not api_key:
        return None

    # Check format
    if not api_key.startswith("orch_"):
        return None

    from orchestrator.core.database import get_db

    with get_db() as db:
        # Find key by prefix (first 9 chars: "orch_live")
        prefix = api_key[:9]
        key_records = db.query(APIKey).filter(
            APIKey.key_prefix == prefix,
            APIKey.is_active == True,
        ).all()

        for key_record in key_records:
            if verify_api_key(api_key, key_record.key_hash):
                # Check expiration
                if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                    continue

                # Update last used
                key_record.last_used = datetime.utcnow()
                db.commit()

                # Parse scopes
                scopes = key_record.scopes.split(",") if key_record.scopes else ["*"]

                return AuthContext(
                    user_id=str(key_record.created_by_user_id) if key_record.created_by_user_id else None,
                    organization_id=str(key_record.organization_id),
                    role="api_key",
                    scopes=scopes,
                    is_api_key=True,
                )

    return None


async def get_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[AuthContext]:
    """Validate JWT token and return auth context."""
    if not credentials:
        return None

    token_payload = decode_token(credentials.credentials)

    if token_payload.type != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    return AuthContext(
        user_id=token_payload.sub,
        organization_id=token_payload.org_id,
        role=token_payload.role,
        scopes=["*"],  # JWT users have full scopes based on role
        is_api_key=False,
    )


async def get_current_auth(
    api_key_auth: Optional[AuthContext] = Depends(get_api_key),
    jwt_auth: Optional[AuthContext] = Depends(get_jwt_token),
) -> AuthContext:
    """Get current authentication context.

    Tries API key first, then JWT token.
    """
    auth = api_key_auth or jwt_auth

    if not auth:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify organization is active
    from orchestrator.core.database import get_db

    with get_db() as db:
        org = db.query(Organization).filter(
            Organization.id == UUID(auth.organization_id),
            Organization.is_active == True,
        ).first()

        if not org:
            raise HTTPException(status_code=403, detail="Organization is inactive or not found")

    return auth


async def get_optional_auth(
    api_key_auth: Optional[AuthContext] = Depends(get_api_key),
    jwt_auth: Optional[AuthContext] = Depends(get_jwt_token),
) -> Optional[AuthContext]:
    """Get optional authentication context (for public endpoints)."""
    return api_key_auth or jwt_auth


def require_scope(scope: str):
    """Dependency factory to require a specific scope."""
    async def check_scope(auth: AuthContext = Depends(get_current_auth)) -> AuthContext:
        if not auth.has_scope(scope):
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scope: {scope}",
            )
        return auth
    return check_scope


def require_role(allowed_roles: list[str]):
    """Dependency factory to require specific roles."""
    async def check_role(auth: AuthContext = Depends(get_current_auth)) -> AuthContext:
        if auth.role not in allowed_roles and auth.role != "owner":
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required roles: {allowed_roles}",
            )
        return auth
    return check_role
