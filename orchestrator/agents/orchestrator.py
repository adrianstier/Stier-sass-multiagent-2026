"""Orchestrator agent that coordinates the multi-agent workflow."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from uuid import UUID

from orchestrator.core.database import get_db
from orchestrator.core.models import (
    Run, Task, Event, Artifact,
    RunStatus, TaskStatus, GateStatus
)
from orchestrator.core.task_dsl import (
    WorkflowPlan, TaskSpec, create_standard_workflow,
    ValidationMethod
)
from orchestrator.core.config import settings

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Orchestrates the multi-agent workflow.

    Responsibilities:
    - Decompose goal into task DAG
    - Dispatch tasks to role queues
    - Monitor task completion
    - Validate quality gates
    - Enforce gate_blocks (tasks blocked by gate failures)
    - Handle errors and retries
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._task_spec_cache: Dict[str, TaskSpec] = {}

    def create_plan(self, goal: str, context: Dict[str, Any] = None) -> WorkflowPlan:
        """Create a workflow plan for the given goal."""
        # For now, use the standard workflow
        # In production, this could use LLM to customize the plan
        plan = create_standard_workflow()

        # Cache task specs for gate_blocks lookup
        for spec in plan.tasks:
            self._task_spec_cache[spec.task_type] = spec

        # Set success/acceptance criteria based on goal
        plan.success_criteria = [
            "All required artifacts produced",
            "Code review gate passed",
            "Security review gate passed",
        ]
        plan.acceptance_criteria = [
            f"Goal achieved: {goal}",
            "All quality standards met",
        ]

        return plan

    def initialize_run(self, goal: str, context: Dict[str, Any] = None) -> Run:
        """Initialize a new run with the given goal."""
        plan = self.create_plan(goal, context)

        # Validate the plan
        errors = plan.validate()
        if errors:
            raise ValueError(f"Invalid workflow plan: {errors}")

        with get_db() as db:
            # Get the run
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if not run:
                raise ValueError(f"Run {self.run_id} not found")

            # Update run with plan details
            run.status = RunStatus.PLANNING
            run.success_criteria = plan.success_criteria
            run.acceptance_criteria = plan.acceptance_criteria
            db.commit()

            # Create tasks from plan, including gate_blocks metadata
            for task_spec in plan.tasks:
                idempotency_key = task_spec.generate_idempotency_key(self.run_id)

                # Check for existing task with same idempotency key
                existing = db.query(Task).filter(
                    Task.run_id == UUID(self.run_id),
                    Task.idempotency_key == idempotency_key
                ).first()

                if existing:
                    logger.info(f"Task {task_spec.task_type} already exists, skipping")
                    continue

                task = Task(
                    run_id=UUID(self.run_id),
                    task_type=task_spec.task_type,
                    assigned_role=task_spec.assigned_role,
                    description=task_spec.description,
                    dependencies=[],  # Will be resolved to task IDs
                    expected_artifacts=task_spec.expected_artifacts,
                    validation_method=task_spec.validation_method.value,
                    priority=task_spec.priority,
                    idempotency_key=idempotency_key,
                    max_retries=task_spec.max_retries,
                    # Store gate metadata in task for later enforcement
                    metadata={
                        "is_gate": task_spec.is_gate,
                        "gate_blocks": task_spec.gate_blocks,
                    },
                )
                db.add(task)

            db.commit()

            # Now resolve dependencies to task IDs
            tasks = db.query(Task).filter(Task.run_id == UUID(self.run_id)).all()
            task_id_map = {t.task_type: str(t.id) for t in tasks}

            for task_spec in plan.tasks:
                task = next(t for t in tasks if t.task_type == task_spec.task_type)
                task.dependencies = [
                    task_id_map[dep] for dep in task_spec.dependencies
                    if dep in task_id_map
                ]

            db.commit()

            # Record planning complete event
            self._record_event(db, "planning_complete", {
                "task_count": len(plan.tasks),
                "plan": plan.to_dict(),
            })

            # Update run status
            run.status = RunStatus.RUNNING
            run.started_at = datetime.utcnow()
            db.commit()

            return run

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute.

        A task is ready when:
        1. It's in PENDING status
        2. All its dependencies are completed
        3. It's not blocked by a failed gate (gate_blocks enforcement)
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()

            # Get all tasks for this run
            all_tasks = db.query(Task).filter(
                Task.run_id == UUID(self.run_id)
            ).all()

            # Get completed task IDs
            completed_ids = {
                str(t.id) for t in all_tasks
                if t.status == TaskStatus.COMPLETED
            }

            # Build task type to task mapping
            task_by_type = {t.task_type: t for t in all_tasks}

            # Get blocked task types based on gate failures
            blocked_by_gates = self._get_gate_blocked_tasks(run, all_tasks)

            # Find tasks where all dependencies are met and not blocked
            ready = []
            for task in all_tasks:
                if task.status != TaskStatus.PENDING:
                    continue

                # Check if blocked by a failed gate
                if task.task_type in blocked_by_gates:
                    logger.debug(
                        f"Task {task.task_type} blocked by failed gate: "
                        f"{blocked_by_gates[task.task_type]}"
                    )
                    continue

                # Check dependencies
                deps_met = all(dep in completed_ids for dep in task.dependencies)
                if deps_met:
                    ready.append(task)

            # Sort by priority (higher first)
            ready.sort(key=lambda t: t.priority, reverse=True)
            return ready

    def _get_gate_blocked_tasks(
        self, run: Run, all_tasks: List[Task]
    ) -> Dict[str, str]:
        """Get tasks that are blocked by failed gates.

        Returns a dict of {task_type: blocking_gate_name}
        """
        blocked = {}

        # Find all gate tasks and their status
        for task in all_tasks:
            metadata = task.metadata or {}
            if not metadata.get("is_gate"):
                continue

            gate_blocks = metadata.get("gate_blocks", [])
            if not gate_blocks:
                continue

            # Check if this gate has failed
            gate_failed = False
            if task.assigned_role == "code_reviewer":
                gate_failed = run.code_review_status == GateStatus.FAILED
            elif task.assigned_role == "security_reviewer":
                gate_failed = run.security_review_status == GateStatus.FAILED
            elif task.assigned_role == "design_reviewer":
                # Design review gate (if we add it to the model)
                gate_failed = task.status == TaskStatus.FAILED

            if gate_failed:
                # Block all tasks in gate_blocks
                for blocked_type in gate_blocks:
                    blocked[blocked_type] = task.task_type
                    logger.info(
                        f"Task '{blocked_type}' blocked by failed gate '{task.task_type}'"
                    )

        return blocked

    def dispatch_task(self, task: Task) -> str:
        """Dispatch a task to the appropriate worker queue."""
        from orchestrator.agents.tasks import execute_agent_task
        from orchestrator.core.celery_app import get_queue_for_role

        queue = get_queue_for_role(task.assigned_role)

        with get_db() as db:
            # Update task status
            db_task = db.query(Task).filter(Task.id == task.id).first()
            db_task.status = TaskStatus.QUEUED
            db_task.queued_at = datetime.utcnow()
            db.commit()

            # Record dispatch event
            self._record_event(db, "task_dispatched", {
                "task_id": str(task.id),
                "task_type": task.task_type,
                "assigned_role": task.assigned_role,
                "queue": queue,
            })

        # Dispatch to Celery
        result = execute_agent_task.apply_async(
            args=[self.run_id, str(task.id)],
            queue=queue,
            task_id=f"{self.run_id}_{task.id}",
        )

        # Update with Celery task ID
        with get_db() as db:
            db_task = db.query(Task).filter(Task.id == task.id).first()
            db_task.celery_task_id = result.id
            db.commit()

        return result.id

    def check_gates(self) -> Dict[str, GateStatus]:
        """Check the status of quality gates."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            return {
                "code_review": run.code_review_status,
                "security_review": run.security_review_status,
            }

    def is_complete(self) -> bool:
        """Check if the run is complete."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()

            # Check if all tasks are completed (or skipped)
            pending_tasks = db.query(Task).filter(
                Task.run_id == UUID(self.run_id),
                Task.status.not_in([TaskStatus.COMPLETED, TaskStatus.SKIPPED])
            ).count()

            if pending_tasks > 0:
                return False

            # Check quality gates (waived gates also count as passed)
            if settings.require_code_review:
                if run.code_review_status not in [GateStatus.PASSED, GateStatus.WAIVED]:
                    return False
            if settings.require_security_review:
                if run.security_review_status not in [GateStatus.PASSED, GateStatus.WAIVED]:
                    return False

            return True

    def mark_complete(self, success: bool = True) -> None:
        """Mark the run as complete."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            run.status = RunStatus.COMPLETED if success else RunStatus.FAILED
            run.completed_at = datetime.utcnow()
            db.commit()

            self._record_event(db, "run_complete", {
                "success": success,
                "final_status": run.status.value,
            })

    def mark_needs_input(self, reason: str) -> None:
        """Mark the run as needing input."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            run.status = RunStatus.NEEDS_INPUT
            run.blocked_reason = reason
            db.commit()

            self._record_event(db, "run_blocked", {"reason": reason})

    def handle_task_failure(self, task_id: str, error: str) -> bool:
        """Handle a failed task. Returns True if should retry."""
        with get_db() as db:
            task = db.query(Task).filter(Task.id == UUID(task_id)).first()

            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error_message = error
                db.commit()

                self._record_event(db, "task_retry", {
                    "task_id": task_id,
                    "retry_count": task.retry_count,
                    "error": error,
                })
                return True
            else:
                task.status = TaskStatus.FAILED
                task.error_message = error
                db.commit()

                self._record_event(db, "task_failed_final", {
                    "task_id": task_id,
                    "error": error,
                })
                return False

    def skip_blocked_tasks(self) -> List[str]:
        """Skip tasks that are blocked by failed gates and cannot proceed.

        Returns list of skipped task IDs.
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            all_tasks = db.query(Task).filter(Task.run_id == UUID(self.run_id)).all()

            blocked = self._get_gate_blocked_tasks(run, all_tasks)
            skipped = []

            for task in all_tasks:
                if task.status == TaskStatus.PENDING and task.task_type in blocked:
                    task.status = TaskStatus.SKIPPED
                    task.error_message = f"Skipped due to failed gate: {blocked[task.task_type]}"
                    skipped.append(str(task.id))

                    self._record_event(db, "task_skipped", {
                        "task_id": str(task.id),
                        "reason": f"Blocked by gate: {blocked[task.task_type]}",
                    })

            db.commit()
            return skipped

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the run."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()

            tasks = db.query(Task).filter(Task.run_id == UUID(self.run_id)).all()
            task_summary = {}
            for status in TaskStatus:
                task_summary[status.value] = len([t for t in tasks if t.status == status])

            artifacts = db.query(Artifact).filter(Artifact.run_id == UUID(self.run_id)).all()

            # Get blocked tasks info
            blocked_by_gates = self._get_gate_blocked_tasks(run, tasks)

            return {
                "run_id": self.run_id,
                "status": run.status.value,
                "current_phase": run.current_phase,
                "iteration": run.current_iteration,
                "max_iterations": run.max_iterations,
                "gates": {
                    "code_review": run.code_review_status.value,
                    "security_review": run.security_review_status.value,
                },
                "tasks": task_summary,
                "blocked_tasks": list(blocked_by_gates.keys()),
                "artifacts_count": len(artifacts),
                "blocked_reason": run.blocked_reason,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "tokens_used": run.tokens_used,
                "total_cost_usd": run.total_cost_usd,
            }

    def _record_event(self, db, event_type: str, data: Dict[str, Any]) -> None:
        """Record an orchestrator event."""
        event = Event(
            run_id=UUID(self.run_id),
            event_type=event_type,
            actor="orchestrator",
            data=data,
        )
        db.add(event)
        db.commit()
