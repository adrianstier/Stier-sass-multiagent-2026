"""Structured logging configuration.

Provides:
- JSON-formatted logs for production
- Human-readable logs for development
- Request ID tracking
- Sensitive data redaction
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional
import structlog

from orchestrator.core.config import settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
org_id_var: ContextVar[Optional[str]] = ContextVar("org_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


# =============================================================================
# Sensitive Data Redaction
# =============================================================================

SENSITIVE_KEYS = {
    "password", "secret", "token", "key", "credential", "auth",
    "api_key", "apikey", "access_token", "refresh_token", "jwt",
    "authorization", "cookie", "session",
}

SENSITIVE_PATTERNS = [
    "sk-",  # Anthropic API keys
    "orch_live_",  # Our API keys
    "Bearer ",
]


def redact_sensitive(data: Any, depth: int = 0) -> Any:
    """Recursively redact sensitive data from logs."""
    if depth > 10:
        return "[MAX_DEPTH]"

    if isinstance(data, dict):
        return {
            k: "[REDACTED]" if _is_sensitive_key(k) else redact_sensitive(v, depth + 1)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [redact_sensitive(item, depth + 1) for item in data]
    elif isinstance(data, str):
        return _redact_string(data)
    return data


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name indicates sensitive data."""
    key_lower = key.lower()
    return any(sensitive in key_lower for sensitive in SENSITIVE_KEYS)


def _redact_string(value: str) -> str:
    """Redact sensitive patterns from strings."""
    for pattern in SENSITIVE_PATTERNS:
        if pattern in value:
            return "[REDACTED]"
    return value


# =============================================================================
# Structlog Processors
# =============================================================================

def add_request_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add request context to log entries."""
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id

    org_id = org_id_var.get()
    if org_id:
        event_dict["organization_id"] = org_id

    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id

    return event_dict


def redact_processor(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Redact sensitive data from log entries."""
    if settings.redact_sensitive_data:
        return redact_sensitive(event_dict)
    return event_dict


def add_service_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service metadata to log entries."""
    event_dict["service"] = "orchestrator"
    event_dict["version"] = "1.0.0"
    return event_dict


# =============================================================================
# Logger Configuration
# =============================================================================

def configure_logging() -> None:
    """Configure structured logging for the application."""
    # Determine output format
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_request_context,
            add_service_info,
            redact_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware:
    """Middleware to add request ID to all requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate or extract request ID
            headers = dict(scope.get("headers", []))
            request_id = headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())

            # Set context variable
            token = request_id_var.set(request_id)

            # Add to response headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", request_id.encode()))
                    message["headers"] = headers
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                request_id_var.reset(token)
        else:
            await self.app(scope, receive, send)


# =============================================================================
# Logging Decorators
# =============================================================================

def log_execution(func):
    """Decorator to log function execution with timing."""
    import functools
    import time

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.time()

        logger.info(
            "function_started",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )

        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(
                "function_completed",
                function=func.__name__,
                duration_ms=round(duration * 1000, 2),
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                "function_failed",
                function=func.__name__,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
                exc_info=True,
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.time()

        logger.info(
            "function_started",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(
                "function_completed",
                function=func.__name__,
                duration_ms=round(duration * 1000, 2),
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                "function_failed",
                function=func.__name__,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
                exc_info=True,
            )
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
