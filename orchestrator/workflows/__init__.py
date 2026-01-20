"""
Workflow definitions for the orchestrator.

Available workflows:
- frontend_review: Multi-agent frontend design and code review
"""

from .frontend_review import (
    ReviewConfig,
    ReviewPriority,
    ReviewVerdict,
    WORKFLOW_DEFINITION,
    get_frontend_review_prompts,
    get_execution_instructions,
)

__all__ = [
    "ReviewConfig",
    "ReviewPriority",
    "ReviewVerdict",
    "WORKFLOW_DEFINITION",
    "get_frontend_review_prompts",
    "get_execution_instructions",
]
