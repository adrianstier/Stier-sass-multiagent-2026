"""Dead Letter Queue (DLQ) for failed tasks.

Provides a mechanism to:
- Route permanently failed tasks to a DLQ for manual inspection
- Store full execution context for debugging
- Enable task replay from the DLQ
- Track DLQ metrics and alerts
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
import logging

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from orchestrator.core.database import get_db, Base
from orchestrator.core.models import Task, Run, Event, TaskStatus, RunStatus
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# DLQ Model
# =============================================================================

class DeadLetterTask(Base):
    """Dead Letter Queue entry for permanently failed tasks."""

    __tablename__ = "dead_letter_tasks"

    id = Column(PGUUID(as_uuid=True), primary_key=True)

    # Reference to original task
    original_task_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    run_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    organization_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)

    # Task metadata
    task_type = Column(String(100), nullable=False)
    assigned_role = Column(String(50), nullable=False)

    # Failure details
    failure_reason = Column(Text, nullable=False)
    failure_count = Column(Integer, default=1)
    last_error = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    # Full execution context for replay
    execution_context = Column(JSON, nullable=False)
    # Contains: goal, description, dependencies_output, task_input, run_context

    # Agent execution log
    agent_log = Column(JSON, default=dict)
    # Contains: tool_calls, llm_requests, artifacts_attempted

    # DLQ status
    status = Column(String(50), default="pending")  # pending, replayed, discarded, resolved
    resolution_notes = Column(Text, nullable=True)
    resolved_by = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # Replay tracking
    replay_count = Column(Integer, default=0)
    last_replay_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_dlq_status", "status"),
        Index("ix_dlq_created_at", "created_at"),
        Index("ix_dlq_task_type", "task_type"),
    )


# =============================================================================
# DLQ Manager
# =============================================================================

class DLQManager:
    """Manager for Dead Letter Queue operations."""

    def __init__(self):
        self.max_replay_attempts = 3

    def send_to_dlq(
        self,
        task: Task,
        run: Run,
        failure_reason: str,
        error_message: str,
        traceback: Optional[str] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        agent_log: Optional[Dict[str, Any]] = None,
    ) -> DeadLetterTask:
        """Send a failed task to the Dead Letter Queue.

        Args:
            task: The failed Task object
            run: The associated Run object
            failure_reason: Human-readable failure reason
            error_message: The actual error message
            traceback: Optional error traceback
            execution_context: Full context needed to replay the task
            agent_log: Log of agent actions before failure

        Returns:
            The created DeadLetterTask entry
        """
        import uuid

        with get_db() as db:
            # Build execution context if not provided
            if execution_context is None:
                execution_context = self._build_execution_context(db, task, run)

            dlq_entry = DeadLetterTask(
                id=uuid.uuid4(),
                original_task_id=task.id,
                run_id=run.id,
                organization_id=run.organization_id,
                task_type=task.task_type,
                assigned_role=task.assigned_role,
                failure_reason=failure_reason,
                failure_count=task.retry_count + 1,
                last_error=error_message,
                error_traceback=traceback,
                execution_context=execution_context,
                agent_log=agent_log or {},
                status="pending",
            )

            db.add(dlq_entry)

            # Record event
            event = Event(
                run_id=run.id,
                task_id=task.id,
                event_type="task_sent_to_dlq",
                actor="system",
                data={
                    "dlq_id": str(dlq_entry.id),
                    "failure_reason": failure_reason,
                    "retry_count": task.retry_count,
                },
            )
            db.add(event)
            db.commit()

            logger.warning(
                "task_sent_to_dlq",
                task_id=str(task.id),
                run_id=str(run.id),
                dlq_id=str(dlq_entry.id),
                task_type=task.task_type,
                failure_reason=failure_reason,
            )

            return dlq_entry

    def _build_execution_context(
        self, db, task: Task, run: Run
    ) -> Dict[str, Any]:
        """Build the full execution context for a task."""
        from orchestrator.core.models import Artifact

        # Gather dependency outputs
        dependencies_output = {}
        for dep_id in task.dependencies or []:
            dep_task = db.query(Task).filter(Task.id == UUID(dep_id)).first()
            if dep_task and dep_task.status == TaskStatus.COMPLETED:
                artifacts = db.query(Artifact).filter(
                    Artifact.task_id == UUID(dep_id)
                ).all()
                if artifacts:
                    dependencies_output[dep_task.task_type] = {
                        "artifacts": [
                            {
                                "id": str(a.id),
                                "type": a.artifact_type,
                                "name": a.name,
                                "content": a.content[:5000] if a.content else "",  # Truncate for storage
                            }
                            for a in artifacts
                        ]
                    }

        return {
            "goal": run.goal,
            "task_description": task.description,
            "task_type": task.task_type,
            "assigned_role": task.assigned_role,
            "dependencies": task.dependencies,
            "dependencies_output": dependencies_output,
            "run_context": run.context,
            "task_input_data": task.input_data,
            "expected_artifacts": task.expected_artifacts,
        }

    def replay_task(
        self,
        dlq_id: str,
        replayed_by: str,
        modified_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replay a task from the DLQ.

        Args:
            dlq_id: The DLQ entry ID
            replayed_by: User ID who initiated the replay
            modified_context: Optional modifications to the execution context

        Returns:
            Dict with replay status and new task ID
        """
        from orchestrator.agents.tasks import execute_agent_task
        import uuid

        with get_db() as db:
            dlq_entry = db.query(DeadLetterTask).filter(
                DeadLetterTask.id == UUID(dlq_id)
            ).first()

            if not dlq_entry:
                return {"success": False, "error": "DLQ entry not found"}

            if dlq_entry.status != "pending":
                return {"success": False, "error": f"DLQ entry already {dlq_entry.status}"}

            if dlq_entry.replay_count >= self.max_replay_attempts:
                return {
                    "success": False,
                    "error": f"Max replay attempts ({self.max_replay_attempts}) exceeded"
                }

            # Get original task
            original_task = db.query(Task).filter(
                Task.id == dlq_entry.original_task_id
            ).first()

            if not original_task:
                return {"success": False, "error": "Original task not found"}

            # Merge modified context if provided
            execution_context = dlq_entry.execution_context
            if modified_context:
                execution_context = {**execution_context, **modified_context}

            # Reset original task for retry
            original_task.status = TaskStatus.PENDING
            original_task.error_message = None
            original_task.started_at = None
            original_task.completed_at = None
            original_task.retry_count = 0  # Reset retry count for DLQ replay

            # Update DLQ entry
            dlq_entry.replay_count += 1
            dlq_entry.last_replay_at = datetime.utcnow()
            dlq_entry.status = "replayed"
            dlq_entry.resolution_notes = f"Replayed by {replayed_by}"
            dlq_entry.resolved_by = replayed_by

            # Record event
            event = Event(
                run_id=dlq_entry.run_id,
                task_id=dlq_entry.original_task_id,
                event_type="dlq_task_replayed",
                actor=replayed_by,
                data={
                    "dlq_id": dlq_id,
                    "replay_count": dlq_entry.replay_count,
                    "modified_context": bool(modified_context),
                },
            )
            db.add(event)
            db.commit()

            # Dispatch the task
            run_id = str(dlq_entry.run_id)
            task_id = str(original_task.id)

        # Execute outside of DB session
        from orchestrator.agents.tasks import orchestrator_tick
        orchestrator_tick.apply_async(args=[run_id], countdown=2)

        logger.info(
            "dlq_task_replayed",
            dlq_id=dlq_id,
            task_id=task_id,
            run_id=run_id,
            replayed_by=replayed_by,
        )

        return {
            "success": True,
            "task_id": task_id,
            "run_id": run_id,
            "replay_count": dlq_entry.replay_count,
        }

    def discard_entry(
        self,
        dlq_id: str,
        discarded_by: str,
        reason: str,
    ) -> bool:
        """Discard a DLQ entry (mark as not worth replaying).

        Args:
            dlq_id: The DLQ entry ID
            discarded_by: User ID who discarded
            reason: Reason for discarding

        Returns:
            True if successful
        """
        with get_db() as db:
            dlq_entry = db.query(DeadLetterTask).filter(
                DeadLetterTask.id == UUID(dlq_id)
            ).first()

            if not dlq_entry:
                return False

            dlq_entry.status = "discarded"
            dlq_entry.resolution_notes = reason
            dlq_entry.resolved_by = discarded_by
            dlq_entry.resolved_at = datetime.utcnow()

            # Record event
            event = Event(
                run_id=dlq_entry.run_id,
                task_id=dlq_entry.original_task_id,
                event_type="dlq_entry_discarded",
                actor=discarded_by,
                data={
                    "dlq_id": dlq_id,
                    "reason": reason,
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "dlq_entry_discarded",
                dlq_id=dlq_id,
                discarded_by=discarded_by,
                reason=reason,
            )

            return True

    def resolve_entry(
        self,
        dlq_id: str,
        resolved_by: str,
        notes: str,
    ) -> bool:
        """Mark a DLQ entry as resolved (fixed externally).

        Args:
            dlq_id: The DLQ entry ID
            resolved_by: User ID who resolved
            notes: Resolution notes

        Returns:
            True if successful
        """
        with get_db() as db:
            dlq_entry = db.query(DeadLetterTask).filter(
                DeadLetterTask.id == UUID(dlq_id)
            ).first()

            if not dlq_entry:
                return False

            dlq_entry.status = "resolved"
            dlq_entry.resolution_notes = notes
            dlq_entry.resolved_by = resolved_by
            dlq_entry.resolved_at = datetime.utcnow()

            db.commit()

            return True

    def list_pending(
        self,
        organization_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DeadLetterTask]:
        """List pending DLQ entries for an organization."""
        with get_db() as db:
            return db.query(DeadLetterTask).filter(
                DeadLetterTask.organization_id == UUID(organization_id),
                DeadLetterTask.status == "pending",
            ).order_by(
                DeadLetterTask.created_at.desc()
            ).offset(offset).limit(limit).all()

    def get_stats(self, organization_id: str) -> Dict[str, Any]:
        """Get DLQ statistics for an organization."""
        from sqlalchemy import func

        with get_db() as db:
            stats = db.query(
                DeadLetterTask.status,
                func.count(DeadLetterTask.id).label("count"),
            ).filter(
                DeadLetterTask.organization_id == UUID(organization_id)
            ).group_by(DeadLetterTask.status).all()

            by_task_type = db.query(
                DeadLetterTask.task_type,
                func.count(DeadLetterTask.id).label("count"),
            ).filter(
                DeadLetterTask.organization_id == UUID(organization_id),
                DeadLetterTask.status == "pending",
            ).group_by(DeadLetterTask.task_type).all()

            return {
                "by_status": {s.status: s.count for s in stats},
                "pending_by_task_type": {t.task_type: t.count for t in by_task_type},
                "total_pending": sum(s.count for s in stats if s.status == "pending"),
            }


# Global DLQ manager instance
_dlq_manager: Optional[DLQManager] = None


def get_dlq_manager() -> DLQManager:
    """Get the global DLQ manager instance."""
    global _dlq_manager
    if _dlq_manager is None:
        _dlq_manager = DLQManager()
    return _dlq_manager
