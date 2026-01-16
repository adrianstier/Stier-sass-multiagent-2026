#!/usr/bin/env python3
"""Worker entrypoint script.

Usage:
    python -m orchestrator.workers.entrypoint <role>

Where <role> is one of:
    orchestrator, business_analyst, project_manager, ux_engineer,
    tech_lead, database_engineer, backend_engineer, frontend_engineer,
    code_reviewer, security_reviewer, cleanup_agent, data_scientist,
    design_reviewer, all

Example:
    python -m orchestrator.workers.entrypoint backend_engineer
    python -m orchestrator.workers.entrypoint cleanup_agent
    python -m orchestrator.workers.entrypoint all  # Start all queues
"""

import sys
import logging
from orchestrator.core.config import ROLE_QUEUES, ALL_QUEUES
from orchestrator.core.celery_app import celery_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def get_queues_for_role(role: str) -> list[str]:
    """Get the queue(s) to listen on for a given role."""
    if role == "all":
        return ALL_QUEUES
    elif role in ROLE_QUEUES:
        return [ROLE_QUEUES[role]]
    else:
        raise ValueError(f"Unknown role: {role}. Valid roles: {list(ROLE_QUEUES.keys()) + ['all']}")


def main():
    """Main entrypoint for worker."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    role = sys.argv[1]

    try:
        queues = get_queues_for_role(role)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    queue_str = ",".join(queues)
    logger.info(f"Starting worker for role '{role}' on queues: {queue_str}")

    # Start Celery worker
    celery_app.worker_main([
        "worker",
        f"--queues={queue_str}",
        "--loglevel=INFO",
        "--concurrency=1",
        f"--hostname={role}@%h",
    ])


if __name__ == "__main__":
    main()
