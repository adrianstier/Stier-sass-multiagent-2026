"""Celery application configuration with role-based queues and beat scheduler."""

from celery import Celery
from celery.schedules import crontab
from kombu import Queue

from .config import settings, ROLE_QUEUES, ALL_QUEUES

# Create Celery app
celery_app = Celery(
    "orchestrator",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
    task_time_limit=settings.task_timeout_seconds + 60,
    task_soft_time_limit=settings.task_timeout_seconds,

    # Result settings
    result_expires=86400,  # 24 hours

    # Worker settings
    worker_prefetch_multiplier=1,  # Don't prefetch, one task at a time
    worker_concurrency=1,  # Single concurrent task per worker

    # Retry settings
    task_default_retry_delay=30,
    task_max_retries=settings.max_retries,

    # Queue configuration
    task_queues=[
        Queue(queue_name, routing_key=queue_name)
        for queue_name in ALL_QUEUES
    ],
    task_default_queue="q_orch",

    # Task routes - map tasks to queues
    task_routes={
        "orchestrator.agents.tasks.execute_agent_task": {"queue": "q_orch"},
        "orchestrator.agents.tasks.orchestrator_tick": {"queue": "q_orch"},
        "orchestrator.agents.tasks.start_run": {"queue": "q_orch"},
        "orchestrator.agents.tasks.monitor_active_runs": {"queue": "q_orch"},
        "orchestrator.agents.orchestrator.*": {"queue": "q_orch"},
        "orchestrator.agents.business_analyst.*": {"queue": "q_ba"},
        "orchestrator.agents.project_manager.*": {"queue": "q_pm"},
        "orchestrator.agents.ux_engineer.*": {"queue": "q_ux"},
        "orchestrator.agents.tech_lead.*": {"queue": "q_tl"},
        "orchestrator.agents.database_engineer.*": {"queue": "q_db"},
        "orchestrator.agents.backend_engineer.*": {"queue": "q_be"},
        "orchestrator.agents.frontend_engineer.*": {"queue": "q_fe"},
        "orchestrator.agents.code_reviewer.*": {"queue": "q_cr"},
        "orchestrator.agents.security_reviewer.*": {"queue": "q_sec"},
        # Design & Creativity Cluster
        "orchestrator.agents.creative_director.*": {"queue": "q_cd"},
        "orchestrator.agents.visual_designer.*": {"queue": "q_vd"},
        "orchestrator.agents.motion_designer.*": {"queue": "q_md"},
        "orchestrator.agents.brand_strategist.*": {"queue": "q_bs"},
        "orchestrator.agents.design_systems_architect.*": {"queue": "q_dsa"},
        "orchestrator.agents.content_designer.*": {"queue": "q_cont"},
        "orchestrator.agents.illustration_specialist.*": {"queue": "q_illus"},
    },

    # ==========================================================================
    # Celery Beat Schedule - Automatic orchestration
    # ==========================================================================
    beat_schedule={
        # Monitor all active runs every 30 seconds
        "monitor-active-runs": {
            "task": "orchestrator.agents.tasks.monitor_active_runs",
            "schedule": 30.0,  # Every 30 seconds
            "options": {"queue": "q_orch"},
        },
        # Cleanup stale tasks every 5 minutes
        "cleanup-stale-tasks": {
            "task": "orchestrator.agents.tasks.cleanup_stale_tasks",
            "schedule": 300.0,  # Every 5 minutes
            "options": {"queue": "q_orch"},
        },
    },
)

# Auto-discover tasks from agents module
celery_app.autodiscover_tasks(["orchestrator.agents"])


def get_queue_for_role(role: str) -> str:
    """Get the Celery queue name for a given role."""
    return ROLE_QUEUES.get(role, "q_orch")
