"""Database models for the orchestration system.

Implements append-only event sourcing for full traceability.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, DateTime, Enum, ForeignKey,
    JSON, Integer, Boolean, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


class RunStatus(str, enum.Enum):
    """Status of a run."""
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, enum.Enum):
    """Status of a task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_DEPENDENCY = "waiting_dependency"
    NEEDS_INPUT = "needs_input"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateStatus(str, enum.Enum):
    """Status of a quality gate."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WAIVED = "waived"


class Run(Base):
    """A run represents a complete orchestration execution for a goal."""

    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Multi-tenancy - required for SaaS
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), nullable=True)

    # Goal and context
    goal = Column(Text, nullable=False)
    context = Column(JSON, default=dict)  # Additional context/parameters

    # Status tracking
    status = Column(Enum(RunStatus), default=RunStatus.PENDING, nullable=False)
    current_phase = Column(String(100), nullable=True)

    # Success criteria (defined by BA/PM/TL early in the run)
    success_criteria = Column(JSON, default=list)
    acceptance_criteria = Column(JSON, default=list)

    # Budget and limits
    max_iterations = Column(Integer, default=50)
    current_iteration = Column(Integer, default=0)
    budget_tokens = Column(Integer, nullable=True)
    tokens_used = Column(Integer, default=0)

    # Cost tracking
    total_cost_usd = Column(String(20), default="0.00")  # Store as string for precision

    # Quality gates
    code_review_status = Column(Enum(GateStatus), default=GateStatus.PENDING)
    security_review_status = Column(Enum(GateStatus), default=GateStatus.PENDING)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)
    blocked_reason = Column(Text, nullable=True)

    # Relationships
    tasks = relationship("Task", back_populates="run", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_runs_status", "status"),
        Index("ix_runs_created_at", "created_at"),
        Index("ix_runs_organization_id", "organization_id"),
    )


class Task(Base):
    """A task represents a unit of work assigned to a specific agent role."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)

    # Task definition
    task_type = Column(String(100), nullable=False)  # e.g., "requirements_analysis"
    assigned_role = Column(String(50), nullable=False)  # e.g., "business_analyst"

    # Task specification (from DSL)
    description = Column(Text, nullable=False)
    input_data = Column(JSON, default=dict)
    expected_artifacts = Column(JSON, default=list)  # List of artifact types expected
    validation_method = Column(String(100), nullable=True)  # e.g., "schema_validation"

    # Dependencies
    dependencies = Column(JSON, default=list)  # List of task IDs this depends on

    # Execution
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    priority = Column(Integer, default=0)  # Higher = more important

    # Idempotency
    idempotency_key = Column(String(255), nullable=False)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Additional task metadata (for gate_blocks, is_gate flag, etc.)
    task_metadata = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    queued_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(255), nullable=True)

    # Relationships
    run = relationship("Run", back_populates="tasks")
    events = relationship("Event", back_populates="task", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_tasks_run_id", "run_id"),
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_assigned_role", "assigned_role"),
        UniqueConstraint("run_id", "idempotency_key", name="uq_task_idempotency"),
    )


class Event(Base):
    """Append-only event log for full traceability.

    Records all agent prompts, responses, tool calls, and state changes.
    """

    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)

    # Event type
    event_type = Column(String(100), nullable=False)
    # Types: task_created, task_queued, task_started, task_completed, task_failed,
    #        llm_request, llm_response, tool_call, tool_result,
    #        gate_check, state_change, error, user_input

    # Event data
    actor = Column(String(100), nullable=False)  # Role or "system" or "user"
    data = Column(JSON, nullable=False)  # Event-specific payload

    # For LLM events - may be redacted
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    tokens_used = Column(Integer, nullable=True)

    # Timestamps (append-only, never updated)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    run = relationship("Run", back_populates="events")
    task = relationship("Task", back_populates="events")

    __table_args__ = (
        Index("ix_events_run_id", "run_id"),
        Index("ix_events_task_id", "task_id"),
        Index("ix_events_timestamp", "timestamp"),
        Index("ix_events_event_type", "event_type"),
    )


class Artifact(Base):
    """Artifacts are the outputs produced by agents.

    Can be Markdown documents, JSON schemas, code files, etc.
    """

    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True)

    # Artifact identification
    artifact_type = Column(String(100), nullable=False)
    # Types: requirements_doc, project_plan, ux_design, architecture_spec,
    #        database_schema, api_spec, frontend_code, backend_code,
    #        code_review_report, security_review_report

    name = Column(String(255), nullable=False)
    version = Column(Integer, default=1)

    # Content
    content_type = Column(String(100), default="text/markdown")  # MIME type
    content = Column(Text, nullable=False)
    artifact_metadata = Column(JSON, default=dict)  # Additional artifact metadata

    # Validation
    is_valid = Column(Boolean, nullable=True)
    validation_errors = Column(JSON, default=list)

    # Produced by
    produced_by = Column(String(50), nullable=False)  # Role that created this

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    run = relationship("Run", back_populates="artifacts")
    task = relationship("Task", back_populates="artifacts")

    __table_args__ = (
        Index("ix_artifacts_run_id", "run_id"),
        Index("ix_artifacts_artifact_type", "artifact_type"),
        Index("ix_artifacts_produced_by", "produced_by"),
    )


class WebhookEventType(str, enum.Enum):
    """Types of events that can trigger webhooks."""
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_PAUSED = "run.paused"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    GATE_PASSED = "gate.passed"
    GATE_FAILED = "gate.failed"
    GATE_WAIVED = "gate.waived"
    ARTIFACT_CREATED = "artifact.created"


class Webhook(Base):
    """Webhook registration for real-time event notifications."""

    __tablename__ = "webhooks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), nullable=True)

    # Webhook configuration
    name = Column(String(255), nullable=False)
    url = Column(String(2048), nullable=False)
    secret = Column(String(255), nullable=True)  # For HMAC signature verification

    # Event subscriptions (JSON array of event types)
    events = Column(JSON, default=list)  # List of WebhookEventType values

    # Optional filtering
    run_id_filter = Column(UUID(as_uuid=True), nullable=True)  # Only for specific run

    # Status
    is_active = Column(Boolean, default=True)
    failure_count = Column(Integer, default=0)
    last_failure_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_webhooks_organization_id", "organization_id"),
        Index("ix_webhooks_is_active", "is_active"),
    )


class WebhookDelivery(Base):
    """Log of webhook delivery attempts."""

    __tablename__ = "webhook_deliveries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(UUID(as_uuid=True), ForeignKey("webhooks.id"), nullable=False)

    # Event details
    event_type = Column(String(100), nullable=False)
    event_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to Event if applicable
    payload = Column(JSON, nullable=False)

    # Delivery status
    status = Column(String(50), default="pending")  # pending, success, failed
    http_status_code = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    delivered_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_webhook_deliveries_webhook_id", "webhook_id"),
        Index("ix_webhook_deliveries_created_at", "created_at"),
    )
