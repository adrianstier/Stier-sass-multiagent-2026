"""Checkpoint and Resume for Long-Running Workflows.

Provides:
- Periodic workflow state serialization
- Agent conversation history persistence
- Resume from any completed task checkpoint
- Crash recovery mechanisms
"""

import json
import pickle
import base64
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
import uuid as uuid_module

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer, LargeBinary, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from orchestrator.core.database import get_db, Base
from orchestrator.core.models import Run, Task, Event, Artifact, TaskStatus, RunStatus
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Checkpoint Model
# =============================================================================

class WorkflowCheckpoint(Base):
    """Checkpoint for workflow state."""

    __tablename__ = "workflow_checkpoints"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    run_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    organization_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)

    # Checkpoint metadata
    checkpoint_type = Column(String(50), nullable=False)  # auto, manual, task_complete, phase_complete
    checkpoint_name = Column(String(255), nullable=True)

    # State snapshot
    run_state = Column(JSON, nullable=False)  # Run status, iterations, tokens, etc.
    task_states = Column(JSON, nullable=False)  # All task statuses and results
    gate_states = Column(JSON, nullable=False)  # Gate statuses

    # Artifact references (not full content)
    artifact_refs = Column(JSON, nullable=False)  # List of artifact IDs

    # Agent conversation histories (compressed)
    agent_histories = Column(LargeBinary, nullable=True)  # Compressed pickle of conversation histories

    # Metrics at checkpoint time
    metrics = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Checkpoint version for compatibility
    version = Column(Integer, default=1)

    __table_args__ = (
        Index("ix_checkpoint_run_created", "run_id", "created_at"),
        Index("ix_checkpoint_type", "checkpoint_type"),
    )


# =============================================================================
# Checkpoint Manager
# =============================================================================

@dataclass
class CheckpointData:
    """Data structure for checkpoint contents."""

    run_id: str
    checkpoint_id: str
    checkpoint_type: str
    created_at: datetime

    # Run state
    run_status: str
    current_phase: str
    current_iteration: int
    tokens_used: int
    total_cost: str

    # Task states
    tasks: List[Dict[str, Any]]

    # Gate states
    gates: Dict[str, str]

    # Artifact references
    artifacts: List[str]

    # Metrics
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type,
            "created_at": self.created_at.isoformat(),
            "run_status": self.run_status,
            "current_phase": self.current_phase,
            "current_iteration": self.current_iteration,
            "tokens_used": self.tokens_used,
            "total_cost": self.total_cost,
            "tasks": self.tasks,
            "gates": self.gates,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }


class CheckpointManager:
    """Manages workflow checkpoints and recovery."""

    # Auto-checkpoint intervals
    AUTO_CHECKPOINT_INTERVAL_TASKS = 3  # Checkpoint every N completed tasks
    AUTO_CHECKPOINT_INTERVAL_TOKENS = 50000  # Checkpoint every N tokens used
    MAX_CHECKPOINTS_PER_RUN = 20  # Keep only most recent N checkpoints

    def __init__(self):
        self._agent_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._last_checkpoint_tokens: Dict[str, int] = {}
        self._last_checkpoint_tasks: Dict[str, int] = {}

    def create_checkpoint(
        self,
        run_id: str,
        checkpoint_type: str = "manual",
        checkpoint_name: Optional[str] = None,
        include_agent_histories: bool = True,
    ) -> Optional[WorkflowCheckpoint]:
        """Create a checkpoint of the current workflow state.

        Args:
            run_id: The run ID
            checkpoint_type: Type of checkpoint (auto, manual, task_complete, phase_complete)
            checkpoint_name: Optional human-readable name
            include_agent_histories: Whether to include agent conversation histories

        Returns:
            The created WorkflowCheckpoint or None if failed
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                logger.error("checkpoint_failed", run_id=run_id, reason="run_not_found")
                return None

            # Gather task states
            tasks = db.query(Task).filter(Task.run_id == UUID(run_id)).all()
            task_states = []
            for task in tasks:
                task_states.append({
                    "id": str(task.id),
                    "task_type": task.task_type,
                    "assigned_role": task.assigned_role,
                    "status": task.status.value,
                    "retry_count": task.retry_count,
                    "dependencies": task.dependencies,
                    "result_summary": self._summarize_result(task.result),
                    "error_message": task.error_message,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                })

            # Gather gate states
            gate_states = {
                "code_review": run.code_review_status.value,
                "security_review": run.security_review_status.value,
            }

            # Gather artifact references
            artifacts = db.query(Artifact).filter(Artifact.run_id == UUID(run_id)).all()
            artifact_refs = [str(a.id) for a in artifacts]

            # Compress agent histories if included
            agent_histories_data = None
            if include_agent_histories and run_id in self._agent_histories:
                try:
                    pickled = pickle.dumps(self._agent_histories[run_id])
                    agent_histories_data = zlib.compress(pickled)
                except Exception as e:
                    logger.warning("agent_history_compression_failed", error=str(e))

            # Calculate metrics
            completed_count = len([t for t in task_states if t["status"] == "completed"])
            failed_count = len([t for t in task_states if t["status"] == "failed"])
            pending_count = len([t for t in task_states if t["status"] == "pending"])

            metrics = {
                "tasks_completed": completed_count,
                "tasks_failed": failed_count,
                "tasks_pending": pending_count,
                "artifacts_created": len(artifact_refs),
                "checkpoint_size_bytes": len(json.dumps(task_states)),
            }

            # Create checkpoint
            checkpoint = WorkflowCheckpoint(
                id=uuid_module.uuid4(),
                run_id=UUID(run_id),
                organization_id=run.organization_id,
                checkpoint_type=checkpoint_type,
                checkpoint_name=checkpoint_name,
                run_state={
                    "status": run.status.value,
                    "current_phase": run.current_phase,
                    "current_iteration": run.current_iteration,
                    "max_iterations": run.max_iterations,
                    "tokens_used": run.tokens_used,
                    "budget_tokens": run.budget_tokens,
                    "total_cost_usd": run.total_cost_usd,
                    "blocked_reason": run.blocked_reason,
                    "success_criteria": run.success_criteria,
                    "acceptance_criteria": run.acceptance_criteria,
                },
                task_states=task_states,
                gate_states=gate_states,
                artifact_refs=artifact_refs,
                agent_histories=agent_histories_data,
                metrics=metrics,
            )

            db.add(checkpoint)

            # Clean up old checkpoints
            self._cleanup_old_checkpoints(db, run_id)

            # Record event
            event = Event(
                run_id=UUID(run_id),
                event_type="checkpoint_created",
                actor="checkpoint_manager",
                data={
                    "checkpoint_id": str(checkpoint.id),
                    "checkpoint_type": checkpoint_type,
                    "checkpoint_name": checkpoint_name,
                    "metrics": metrics,
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "checkpoint_created",
                run_id=run_id,
                checkpoint_id=str(checkpoint.id),
                checkpoint_type=checkpoint_type,
                tasks_completed=completed_count,
            )

            return checkpoint

    def _summarize_result(self, result: Optional[Dict]) -> Optional[Dict]:
        """Create a summary of task result for checkpoint."""
        if not result:
            return None

        return {
            "success": result.get("success"),
            "artifacts_count": len(result.get("artifacts", [])),
            "tool_calls_count": len(result.get("tool_calls", [])),
            "has_clarifications": bool(result.get("clarifications")),
        }

    def _cleanup_old_checkpoints(self, db, run_id: str) -> None:
        """Remove old checkpoints beyond the limit."""
        checkpoints = db.query(WorkflowCheckpoint).filter(
            WorkflowCheckpoint.run_id == UUID(run_id)
        ).order_by(WorkflowCheckpoint.created_at.desc()).all()

        if len(checkpoints) > self.MAX_CHECKPOINTS_PER_RUN:
            # Keep manual checkpoints and most recent auto checkpoints
            manual_checkpoints = [c for c in checkpoints if c.checkpoint_type == "manual"]
            auto_checkpoints = [c for c in checkpoints if c.checkpoint_type != "manual"]

            # Delete oldest auto checkpoints beyond limit
            keep_auto = self.MAX_CHECKPOINTS_PER_RUN - len(manual_checkpoints)
            to_delete = auto_checkpoints[keep_auto:]

            for checkpoint in to_delete:
                db.delete(checkpoint)

    def should_auto_checkpoint(self, run_id: str) -> bool:
        """Check if an auto-checkpoint should be created.

        Args:
            run_id: The run ID

        Returns:
            True if auto-checkpoint is due
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                return False

            # Check token usage
            last_tokens = self._last_checkpoint_tokens.get(run_id, 0)
            if run.tokens_used - last_tokens >= self.AUTO_CHECKPOINT_INTERVAL_TOKENS:
                return True

            # Check completed tasks
            completed_count = db.query(Task).filter(
                Task.run_id == UUID(run_id),
                Task.status == TaskStatus.COMPLETED,
            ).count()

            last_tasks = self._last_checkpoint_tasks.get(run_id, 0)
            if completed_count - last_tasks >= self.AUTO_CHECKPOINT_INTERVAL_TASKS:
                return True

        return False

    def auto_checkpoint_if_needed(self, run_id: str) -> Optional[WorkflowCheckpoint]:
        """Create an auto-checkpoint if needed."""
        if self.should_auto_checkpoint(run_id):
            checkpoint = self.create_checkpoint(run_id, "auto")
            if checkpoint:
                # Update tracking
                with get_db() as db:
                    run = db.query(Run).filter(Run.id == UUID(run_id)).first()
                    if run:
                        self._last_checkpoint_tokens[run_id] = run.tokens_used

                    completed_count = db.query(Task).filter(
                        Task.run_id == UUID(run_id),
                        Task.status == TaskStatus.COMPLETED,
                    ).count()
                    self._last_checkpoint_tasks[run_id] = completed_count

            return checkpoint
        return None

    def list_checkpoints(
        self,
        run_id: str,
        limit: int = 10,
    ) -> List[CheckpointData]:
        """List checkpoints for a run.

        Args:
            run_id: The run ID
            limit: Maximum checkpoints to return

        Returns:
            List of CheckpointData
        """
        with get_db() as db:
            checkpoints = db.query(WorkflowCheckpoint).filter(
                WorkflowCheckpoint.run_id == UUID(run_id)
            ).order_by(WorkflowCheckpoint.created_at.desc()).limit(limit).all()

            return [
                CheckpointData(
                    run_id=str(c.run_id),
                    checkpoint_id=str(c.id),
                    checkpoint_type=c.checkpoint_type,
                    created_at=c.created_at,
                    run_status=c.run_state.get("status"),
                    current_phase=c.run_state.get("current_phase"),
                    current_iteration=c.run_state.get("current_iteration"),
                    tokens_used=c.run_state.get("tokens_used"),
                    total_cost=c.run_state.get("total_cost_usd"),
                    tasks=c.task_states,
                    gates=c.gate_states,
                    artifacts=c.artifact_refs,
                    metrics=c.metrics,
                )
                for c in checkpoints
            ]

    def restore_from_checkpoint(
        self,
        checkpoint_id: str,
        reset_failed_tasks: bool = True,
        restored_by: str = "system",
    ) -> Dict[str, Any]:
        """Restore workflow state from a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID to restore from
            reset_failed_tasks: Whether to reset failed tasks to pending
            restored_by: Who initiated the restore

        Returns:
            Dict with restore status and details
        """
        with get_db() as db:
            checkpoint = db.query(WorkflowCheckpoint).filter(
                WorkflowCheckpoint.id == UUID(checkpoint_id)
            ).first()

            if not checkpoint:
                return {"success": False, "error": "Checkpoint not found"}

            run = db.query(Run).filter(Run.id == checkpoint.run_id).first()
            if not run:
                return {"success": False, "error": "Run not found"}

            # Restore run state
            run_state = checkpoint.run_state
            run.status = RunStatus(run_state.get("status", "running"))
            run.current_phase = run_state.get("current_phase")
            run.current_iteration = run_state.get("current_iteration", 0)
            run.blocked_reason = None  # Clear any blocked reason

            # Restore gate states
            gate_states = checkpoint.gate_states
            from orchestrator.core.models import GateStatus
            run.code_review_status = GateStatus(gate_states.get("code_review", "pending"))
            run.security_review_status = GateStatus(gate_states.get("security_review", "pending"))

            # Restore task states
            tasks_restored = 0
            tasks_reset = 0

            for task_state in checkpoint.task_states:
                task = db.query(Task).filter(Task.id == UUID(task_state["id"])).first()
                if task:
                    original_status = task_state["status"]

                    # Reset failed tasks if requested
                    if reset_failed_tasks and original_status == "failed":
                        task.status = TaskStatus.PENDING
                        task.error_message = None
                        task.retry_count = 0
                        task.started_at = None
                        task.completed_at = None
                        tasks_reset += 1
                    else:
                        task.status = TaskStatus(original_status)

                    tasks_restored += 1

            # Restore agent histories if present
            if checkpoint.agent_histories:
                try:
                    decompressed = zlib.decompress(checkpoint.agent_histories)
                    self._agent_histories[str(run.id)] = pickle.loads(decompressed)
                except Exception as e:
                    logger.warning("agent_history_restore_failed", error=str(e))

            # Record event
            event = Event(
                run_id=checkpoint.run_id,
                event_type="checkpoint_restored",
                actor=restored_by,
                data={
                    "checkpoint_id": checkpoint_id,
                    "tasks_restored": tasks_restored,
                    "tasks_reset": tasks_reset,
                    "checkpoint_created_at": checkpoint.created_at.isoformat(),
                },
            )
            db.add(event)
            db.commit()

            logger.info(
                "checkpoint_restored",
                run_id=str(run.id),
                checkpoint_id=checkpoint_id,
                tasks_restored=tasks_restored,
                tasks_reset=tasks_reset,
                restored_by=restored_by,
            )

            return {
                "success": True,
                "run_id": str(run.id),
                "checkpoint_id": checkpoint_id,
                "tasks_restored": tasks_restored,
                "tasks_reset": tasks_reset,
                "run_status": run.status.value,
            }

    def store_agent_history(
        self,
        run_id: str,
        task_id: str,
        role: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Store agent conversation history for checkpointing.

        Args:
            run_id: The run ID
            task_id: The task ID
            role: Agent role
            messages: Conversation messages
        """
        if run_id not in self._agent_histories:
            self._agent_histories[run_id] = []

        self._agent_histories[run_id].append({
            "task_id": task_id,
            "role": role,
            "messages": messages,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_agent_history(
        self,
        run_id: str,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get agent conversation history.

        Args:
            run_id: The run ID
            task_id: Optional filter by task ID

        Returns:
            List of conversation records
        """
        histories = self._agent_histories.get(run_id, [])

        if task_id:
            return [h for h in histories if h.get("task_id") == task_id]

        return histories

    def resume_run(
        self,
        run_id: str,
        from_checkpoint_id: Optional[str] = None,
        resumed_by: str = "system",
    ) -> Dict[str, Any]:
        """Resume a run, optionally from a specific checkpoint.

        Args:
            run_id: The run ID
            from_checkpoint_id: Optional checkpoint to restore from
            resumed_by: Who initiated the resume

        Returns:
            Dict with resume status
        """
        # Restore from checkpoint if specified
        if from_checkpoint_id:
            restore_result = self.restore_from_checkpoint(
                from_checkpoint_id,
                reset_failed_tasks=True,
                restored_by=resumed_by,
            )
            if not restore_result.get("success"):
                return restore_result

        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                return {"success": False, "error": "Run not found"}

            # Update run status to running
            if run.status in [RunStatus.NEEDS_INPUT, RunStatus.PAUSED]:
                run.status = RunStatus.RUNNING
                run.blocked_reason = None
                db.commit()

        # Trigger orchestrator tick to continue
        from orchestrator.agents.tasks import orchestrator_tick
        result = orchestrator_tick.apply_async(args=[run_id], countdown=2)

        logger.info(
            "run_resumed",
            run_id=run_id,
            from_checkpoint=from_checkpoint_id,
            resumed_by=resumed_by,
            celery_task_id=result.id,
        )

        return {
            "success": True,
            "run_id": run_id,
            "from_checkpoint": from_checkpoint_id,
            "celery_task_id": result.id,
        }


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
