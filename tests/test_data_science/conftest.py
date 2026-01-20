"""Pytest configuration for data science tests.

This conftest configures the test environment to allow isolated testing
of the data science modules without requiring the full orchestrator
infrastructure (database, etc.).
"""

import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_configure(config):
    """Configure pytest before tests run."""
    # We need to prevent the orchestrator.agents.__init__.py from being evaluated
    # because it imports modules with heavy dependencies (database, etc.)

    # Create mock for settings
    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql://test:test@localhost/test"
    mock_settings.llm_provider = "anthropic"
    mock_settings.claude_model = "claude-sonnet-4-20250514"

    # Pre-populate sys.modules with mocks for problematic imports
    sys.modules['orchestrator.core.database'] = MagicMock()
    sys.modules['orchestrator.core.config'] = MagicMock()
    sys.modules['orchestrator.core.config'].settings = mock_settings
