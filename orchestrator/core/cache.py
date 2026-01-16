"""Redis caching layer for artifacts and frequently accessed data.

Provides:
- Artifact content caching with TTL
- Run status caching for quick lookups
- Cache invalidation helpers
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List

import redis

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Cache TTL settings (in seconds)
ARTIFACT_CACHE_TTL = 3600  # 1 hour
RUN_STATUS_CACHE_TTL = 30  # 30 seconds
TASK_STATUS_CACHE_TTL = 30  # 30 seconds

# Cache key prefixes
ARTIFACT_PREFIX = "cache:artifact:"
RUN_STATUS_PREFIX = "cache:run_status:"
TASK_STATUS_PREFIX = "cache:task_status:"
ORG_USAGE_PREFIX = "cache:org_usage:"


class RedisCache:
    """Redis cache client with high-level caching operations."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """Lazy-initialize Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        return self._client

    def is_available(self) -> bool:
        """Check if Redis is available."""
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    # =========================================================================
    # Generic Cache Operations
    # =========================================================================

    def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        try:
            return self.client.get(key)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("cache_get_error", key=key, error=str(e))
            return None

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set a value in cache with optional TTL."""
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("cache_set_error", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            self.client.delete(key)
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("cache_delete_error", key=key, error=str(e))
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        try:
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                return self.client.delete(*keys)
            return 0
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("cache_delete_pattern_error", pattern=pattern, error=str(e))
            return 0

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a JSON value from cache."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value: Dict[str, Any], ttl: int = None) -> bool:
        """Set a JSON value in cache."""
        try:
            return self.set(key, json.dumps(value, default=str), ttl)
        except (TypeError, ValueError) as e:
            logger.warning("cache_set_json_error", key=key, error=str(e))
            return False

    # =========================================================================
    # Artifact Caching
    # =========================================================================

    def _artifact_key(self, artifact_id: str) -> str:
        """Generate cache key for an artifact."""
        return f"{ARTIFACT_PREFIX}{artifact_id}"

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get an artifact from cache."""
        key = self._artifact_key(artifact_id)
        return self.get_json(key)

    def set_artifact(self, artifact_id: str, artifact_data: Dict[str, Any]) -> bool:
        """Cache an artifact."""
        key = self._artifact_key(artifact_id)
        return self.set_json(key, artifact_data, ARTIFACT_CACHE_TTL)

    def invalidate_artifact(self, artifact_id: str) -> bool:
        """Remove an artifact from cache."""
        key = self._artifact_key(artifact_id)
        return self.delete(key)

    def invalidate_run_artifacts(self, run_id: str) -> int:
        """Invalidate all artifacts for a run."""
        # We don't have run_id in the artifact key, so we need to track differently
        # For now, this is a placeholder - in production you might use a Redis set
        # to track artifact IDs per run
        return 0

    # =========================================================================
    # Run Status Caching
    # =========================================================================

    def _run_status_key(self, run_id: str) -> str:
        """Generate cache key for run status."""
        return f"{RUN_STATUS_PREFIX}{run_id}"

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run status from cache."""
        key = self._run_status_key(run_id)
        return self.get_json(key)

    def set_run_status(self, run_id: str, status_data: Dict[str, Any]) -> bool:
        """Cache run status."""
        key = self._run_status_key(run_id)
        return self.set_json(key, status_data, RUN_STATUS_CACHE_TTL)

    def invalidate_run_status(self, run_id: str) -> bool:
        """Remove run status from cache."""
        key = self._run_status_key(run_id)
        return self.delete(key)

    # =========================================================================
    # Organization Usage Caching
    # =========================================================================

    def _org_usage_key(self, org_id: str) -> str:
        """Generate cache key for organization usage."""
        return f"{ORG_USAGE_PREFIX}{org_id}"

    def get_org_usage(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get organization usage from cache."""
        key = self._org_usage_key(org_id)
        return self.get_json(key)

    def set_org_usage(self, org_id: str, usage_data: Dict[str, Any], ttl: int = 60) -> bool:
        """Cache organization usage (short TTL as it changes frequently)."""
        key = self._org_usage_key(org_id)
        return self.set_json(key, usage_data, ttl)

    def increment_org_tokens(self, org_id: str, tokens: int) -> Optional[int]:
        """Increment token usage for an organization (atomic operation)."""
        key = f"{ORG_USAGE_PREFIX}{org_id}:tokens"
        try:
            return self.client.incrby(key, tokens)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("cache_increment_error", key=key, error=str(e))
            return None


# Global cache instance
cache = RedisCache()


# =========================================================================
# Decorator for caching function results
# =========================================================================

def cached(prefix: str, ttl: int = 300, key_func=None):
    """
    Decorator to cache function results.

    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key from args/kwargs
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"{prefix}:{key_func(*args, **kwargs)}"
            else:
                # Default: hash all arguments
                key_data = f"{args}:{sorted(kwargs.items())}"
                key_hash = hashlib.md5(key_data.encode()).hexdigest()
                cache_key = f"{prefix}:{key_hash}"

            # Try to get from cache
            cached_result = cache.get_json(cache_key)
            if cached_result is not None:
                logger.debug("cache_hit", key=cache_key)
                return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set_json(cache_key, result, ttl)
                logger.debug("cache_miss_stored", key=cache_key)

            return result

        return wrapper
    return decorator


# =========================================================================
# Cache warming utilities
# =========================================================================

def warm_run_cache(run_id: str):
    """Pre-populate cache for a run."""
    from .models import Run, Task, Artifact
    from .database import get_db

    with get_db() as db:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            return

        # Cache run status
        status_data = {
            "run_id": str(run.id),
            "status": run.status.value,
            "current_phase": run.current_phase,
            "iteration": run.current_iteration,
            "tokens_used": run.tokens_used,
            "cached_at": datetime.utcnow().isoformat(),
        }
        cache.set_run_status(str(run.id), status_data)

        # Cache recent artifacts
        artifacts = db.query(Artifact).filter(
            Artifact.run_id == run_id
        ).order_by(Artifact.created_at.desc()).limit(10).all()

        for artifact in artifacts:
            artifact_data = {
                "id": str(artifact.id),
                "artifact_type": artifact.artifact_type,
                "name": artifact.name,
                "content_type": artifact.content_type,
                "content": artifact.content,
                "produced_by": artifact.produced_by,
                "created_at": artifact.created_at.isoformat(),
            }
            cache.set_artifact(str(artifact.id), artifact_data)

    logger.info("cache_warmed", run_id=run_id)
