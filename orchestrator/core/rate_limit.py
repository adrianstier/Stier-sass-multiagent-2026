"""Rate limiting middleware and utilities.

Implements:
- Per-endpoint rate limiting
- Per-organization rate limiting
- Token budget enforcement
- Cost tracking
"""

import time
from datetime import datetime, timedelta
from typing import Callable, Optional
import logging

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# In-Memory Rate Limiter (for single-instance)
# For production, use Redis-based implementation below
# =============================================================================

class InMemoryRateLimiter:
    """Simple in-memory rate limiter for development/testing."""

    def __init__(self):
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        """Check if request is allowed under rate limit.

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = time.time()
        window_start = now - window_seconds

        # Clean old entries
        if key in self.requests:
            self.requests[key] = [ts for ts in self.requests[key] if ts > window_start]
        else:
            self.requests[key] = []

        # Check limit
        current_count = len(self.requests[key])
        if current_count >= limit:
            return False, 0

        # Record request
        self.requests[key].append(now)
        return True, limit - current_count - 1


# =============================================================================
# Redis Rate Limiter (for production)
# =============================================================================

class RedisRateLimiter:
    """Redis-based rate limiter for distributed deployments."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            import redis
            self._client = redis.from_url(self.redis_url)
        return self._client

    # Lua script for atomic rate limiting - checks BEFORE adding
    RATE_LIMIT_SCRIPT = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    -- Remove old entries
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

    -- Get current count
    local count = redis.call('ZCARD', key)

    -- Check if under limit BEFORE adding
    if count < limit then
        -- Add this request
        redis.call('ZADD', key, now, tostring(now) .. ':' .. tostring(math.random()))
        redis.call('EXPIRE', key, window)
        return {1, limit - count - 1}
    else
        -- Over limit, don't add
        redis.call('EXPIRE', key, window)
        return {0, 0}
    end
    """

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        """Check if request is allowed using sliding window (atomic via Lua script)."""
        now = time.time()
        window_key = f"ratelimit:{key}"

        # Use Lua script for atomic check-then-add
        result = self.client.eval(
            self.RATE_LIMIT_SCRIPT,
            1,  # Number of keys
            window_key,
            limit,
            window_seconds,
            now
        )

        allowed = bool(result[0])
        remaining = int(result[1])

        return allowed, remaining

    def get_token_usage(self, org_id: str) -> int:
        """Get current month's token usage for an organization."""
        key = f"tokens:{org_id}:{datetime.utcnow().strftime('%Y-%m')}"
        usage = self.client.get(key)
        return int(usage) if usage else 0

    def increment_token_usage(self, org_id: str, tokens: int) -> int:
        """Increment token usage and return new total."""
        key = f"tokens:{org_id}:{datetime.utcnow().strftime('%Y-%m')}"
        # Set expiry to 35 days to cover billing cycle overlap
        new_total = self.client.incrby(key, tokens)
        self.client.expire(key, 35 * 24 * 60 * 60)
        return new_total


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


def get_rate_limiter() -> RedisRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter()
    return _rate_limiter


# =============================================================================
# Rate Limit Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""

    # Default limits per endpoint pattern
    DEFAULT_LIMITS = {
        "POST /runs": 10,  # 10 new runs per minute
        "GET /runs": 60,  # 60 list requests per minute
        "POST /runs/*/tick": 30,  # 30 ticks per minute
        "DEFAULT": 120,  # 120 requests per minute default
    }

    def __init__(self, app, use_redis: bool = True):
        super().__init__(app)
        self.use_redis = use_redis
        self.memory_limiter = InMemoryRateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Get rate limit key
        key = self._get_rate_key(request)
        limit = self._get_limit(request)

        # Check rate limit
        if self.use_redis:
            try:
                limiter = get_rate_limiter()
                is_allowed, remaining = limiter.is_allowed(key, limit)
            except Exception as e:
                logger.warning(f"Redis rate limiter failed, using memory: {e}")
                is_allowed, remaining = self.memory_limiter.is_allowed(key, limit)
        else:
            is_allowed, remaining = self.memory_limiter.is_allowed(key, limit)

        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down.",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_rate_key(self, request: Request) -> str:
        """Generate rate limit key from request."""
        # Try to get org ID from auth context
        org_id = getattr(request.state, "organization_id", None)
        if org_id:
            return f"org:{org_id}:{request.method}:{request.url.path}"

        # Fall back to IP-based limiting
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}:{request.method}:{request.url.path}"

    def _get_limit(self, request: Request) -> int:
        """Get rate limit for the endpoint."""
        # Build pattern
        path = request.url.path
        # Normalize path (replace UUIDs with *)
        import re
        normalized = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '*',
            path,
        )
        pattern = f"{request.method} {normalized}"

        return self.DEFAULT_LIMITS.get(pattern, self.DEFAULT_LIMITS["DEFAULT"])


# =============================================================================
# Token Budget Enforcement
# =============================================================================

class TokenBudgetError(Exception):
    """Raised when token budget is exceeded."""
    def __init__(self, message: str, used: int, limit: int):
        super().__init__(message)
        self.used = used
        self.limit = limit


class TokenBudgetEnforcer:
    """Enforces token budgets at run and organization levels."""

    def __init__(self):
        self.limiter = get_rate_limiter()

    def check_budget(self, org_id: str, run_budget: Optional[int], run_used: int, tokens_needed: int) -> None:
        """Check if token usage is within budget.

        Args:
            org_id: Organization ID
            run_budget: Optional per-run budget
            run_used: Tokens already used in this run
            tokens_needed: Estimated tokens for this request

        Raises:
            TokenBudgetError: If budget would be exceeded
        """
        # Check run-level budget
        if run_budget:
            projected = run_used + tokens_needed
            if projected > run_budget:
                raise TokenBudgetError(
                    f"Run token budget exceeded. Used: {run_used}, Limit: {run_budget}",
                    used=run_used,
                    limit=run_budget,
                )

        # Check organization-level budget
        org_usage = self.limiter.get_token_usage(org_id)
        org_limit = self._get_org_limit(org_id)

        if org_usage + tokens_needed > org_limit:
            raise TokenBudgetError(
                f"Organization monthly token limit exceeded. Used: {org_usage}, Limit: {org_limit}",
                used=org_usage,
                limit=org_limit,
            )

    def record_usage(self, org_id: str, tokens: int) -> int:
        """Record token usage and return new total."""
        return self.limiter.increment_token_usage(org_id, tokens)

    def _get_org_limit(self, org_id: str) -> int:
        """Get token limit for organization based on plan."""
        from orchestrator.core.database import get_db
        from orchestrator.core.auth import Organization
        from uuid import UUID

        with get_db() as db:
            org = db.query(Organization).filter(Organization.id == UUID(org_id)).first()
            if org:
                return org.monthly_token_limit

        # Default limit for unknown orgs
        return 100000


# Global enforcer instance
_budget_enforcer: Optional[TokenBudgetEnforcer] = None


def get_budget_enforcer() -> TokenBudgetEnforcer:
    """Get the global token budget enforcer."""
    global _budget_enforcer
    if _budget_enforcer is None:
        _budget_enforcer = TokenBudgetEnforcer()
    return _budget_enforcer


# =============================================================================
# Cost Calculation
# =============================================================================

# Pricing per 1M tokens (as of 2024)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for token usage."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-20250514"])

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return round(input_cost + output_cost, 6)
