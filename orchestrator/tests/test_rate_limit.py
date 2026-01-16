"""Tests for rate limiting module."""

import pytest
import time
from unittest.mock import patch, MagicMock

from orchestrator.core.rate_limit import (
    InMemoryRateLimiter,
    TokenBudgetError,
    calculate_cost,
)


class TestInMemoryRateLimiter:
    """Tests for in-memory rate limiter."""

    def test_allows_requests_under_limit(self):
        """Requests under limit should be allowed."""
        limiter = InMemoryRateLimiter()

        for i in range(5):
            allowed, remaining = limiter.is_allowed("test_key", 10)
            assert allowed is True
            assert remaining == 10 - i - 1

    def test_blocks_requests_over_limit(self):
        """Requests over limit should be blocked."""
        limiter = InMemoryRateLimiter()

        # Use up the limit
        for _ in range(10):
            limiter.is_allowed("test_key", 10)

        # Next request should be blocked
        allowed, remaining = limiter.is_allowed("test_key", 10)
        assert allowed is False
        assert remaining == 0

    def test_different_keys_have_separate_limits(self):
        """Different keys should have independent limits."""
        limiter = InMemoryRateLimiter()

        # Use up limit for key1
        for _ in range(10):
            limiter.is_allowed("key1", 10)

        # key2 should still have quota
        allowed, remaining = limiter.is_allowed("key2", 10)
        assert allowed is True

    def test_window_expiration(self):
        """Old requests should expire from the window."""
        limiter = InMemoryRateLimiter()

        # Make requests
        for _ in range(5):
            limiter.is_allowed("test_key", 10, window_seconds=1)

        # Wait for window to expire
        time.sleep(1.1)

        # Should have full quota again
        allowed, remaining = limiter.is_allowed("test_key", 10, window_seconds=1)
        assert allowed is True
        assert remaining == 9


class TestTokenBudgetError:
    """Tests for TokenBudgetError."""

    def test_error_contains_usage_info(self):
        """Error should contain usage information."""
        error = TokenBudgetError("Budget exceeded", used=50000, limit=100000)

        assert error.used == 50000
        assert error.limit == 100000
        assert "Budget exceeded" in str(error)


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_calculate_cost_sonnet(self):
        """Should calculate cost for Sonnet model."""
        # 1000 input tokens, 1000 output tokens
        cost = calculate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=1000,
        )

        # Input: 1000/1M * $3 = $0.003
        # Output: 1000/1M * $15 = $0.015
        # Total: $0.018
        assert cost == pytest.approx(0.018, abs=0.001)

    def test_calculate_cost_haiku(self):
        """Should calculate cost for Haiku model."""
        cost = calculate_cost(
            "claude-3-haiku-20240307",
            input_tokens=1000,
            output_tokens=1000,
        )

        # Input: 1000/1M * $0.25 = $0.00025
        # Output: 1000/1M * $1.25 = $0.00125
        # Total: $0.0015
        assert cost == pytest.approx(0.0015, abs=0.0001)

    def test_calculate_cost_opus(self):
        """Should calculate cost for Opus model."""
        cost = calculate_cost(
            "claude-3-opus-20240229",
            input_tokens=1000,
            output_tokens=1000,
        )

        # Input: 1000/1M * $15 = $0.015
        # Output: 1000/1M * $75 = $0.075
        # Total: $0.09
        assert cost == pytest.approx(0.09, abs=0.001)

    def test_calculate_cost_unknown_model_uses_default(self):
        """Unknown models should use default (Sonnet) pricing."""
        cost = calculate_cost(
            "unknown-model",
            input_tokens=1000,
            output_tokens=1000,
        )

        # Should use Sonnet pricing
        assert cost == pytest.approx(0.018, abs=0.001)

    def test_calculate_cost_zero_tokens(self):
        """Zero tokens should return zero cost."""
        cost = calculate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=0,
            output_tokens=0,
        )

        assert cost == 0.0

    def test_calculate_cost_large_usage(self):
        """Should handle large token counts."""
        cost = calculate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000000,  # 1M input
            output_tokens=100000,   # 100k output
        )

        # Input: 1M/1M * $3 = $3
        # Output: 100k/1M * $15 = $1.5
        # Total: $4.5
        assert cost == pytest.approx(4.5, abs=0.01)
