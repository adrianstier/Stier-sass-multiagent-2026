"""Celery tasks for agent execution."""

import logging
from datetime import datetime
from typing import Dict, Any
from uuid import UUID

from celery import shared_task

from orchestrator.core.database import get_db
from orchestrator.core.models import Task, Run, Artifact, TaskStatus, GateStatus
from orchestrator.agents.specialists import get_agent_class

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
    acks_late=True,
)
def execute_agent_task(self, run_id: str, task_id: str) -> Dict[str, Any]:
    """Execute a single agent task.

    This is the main Celery task that:
    1. Loads the task from DB
    2. Instantiates the appropriate agent
    3. Gathers input from dependencies
    4. Executes the agent
    5. Saves results and artifacts
    """
    logger.info(f"Executing task {task_id} for run {run_id}")

    with get_db() as db:
        # Load task
        task = db.query(Task).filter(Task.id == UUID(task_id)).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Load run
        run = db.query(Run).filter(Run.id == UUID(run_id)).first()
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        db.commit()

        # Get agent class
        agent_class = get_agent_class(task.assigned_role)
        if not agent_class:
            raise ValueError(f"Unknown role: {task.assigned_role}")

        # Gather input from dependencies
        dependencies_output = {}
        for dep_id in task.dependencies:
            dep_task = db.query(Task).filter(Task.id == UUID(dep_id)).first()
            if dep_task and dep_task.status == TaskStatus.COMPLETED:
                # Get the artifact from this dependency
                dep_artifacts = db.query(Artifact).filter(
                    Artifact.task_id == UUID(dep_id)
                ).all()
                if dep_artifacts:
                    dependencies_output[dep_task.task_type] = dep_artifacts[0].content

        # Prepare task input
        task_input = {
            "goal": run.goal,
            "description": task.description,
            "task_type": task.task_type,
            "dependencies_output": dependencies_output,
            "context": run.context or {},
        }

    # Execute agent (outside DB session to avoid long transactions)
    try:
        agent = agent_class(run_id, task_id)
        result = agent.execute(task_input)
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        with get_db() as db:
            task = db.query(Task).filter(Task.id == UUID(task_id)).first()
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            db.commit()
        raise

    # Save results
    with get_db() as db:
        task = db.query(Task).filter(Task.id == UUID(task_id)).first()

        if result.get("success"):
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow()

            # Handle quality gates
            if task.assigned_role == "code_reviewer":
                _handle_code_review_gate(db, run_id, result)
            elif task.assigned_role == "security_reviewer":
                _handle_security_review_gate(db, run_id, result)

        else:
            task.status = TaskStatus.FAILED
            task.error_message = result.get("error", "Unknown error")
            task.completed_at = datetime.utcnow()

        db.commit()

    logger.info(f"Task {task_id} completed with status: {task.status}")
    return result


def _handle_code_review_gate(db, run_id: str, result: Dict[str, Any]) -> None:
    """Handle the code review quality gate."""
    run = db.query(Run).filter(Run.id == UUID(run_id)).first()

    # Check if the response contains APPROVED or REJECTED
    response = result.get("response", "")
    if "APPROVED" in response.upper():
        run.code_review_status = GateStatus.PASSED
        logger.info(f"Code review gate PASSED for run {run_id}")
    elif "REJECTED" in response.upper():
        run.code_review_status = GateStatus.FAILED
        logger.info(f"Code review gate FAILED for run {run_id}")
    else:
        # Default to pending if unclear
        logger.warning(f"Code review gate unclear for run {run_id}")

    db.commit()


def _handle_security_review_gate(db, run_id: str, result: Dict[str, Any]) -> None:
    """Handle the security review quality gate."""
    run = db.query(Run).filter(Run.id == UUID(run_id)).first()

    # Check if the response contains APPROVED or REJECTED
    response = result.get("response", "")
    if "APPROVED" in response.upper():
        run.security_review_status = GateStatus.PASSED
        logger.info(f"Security review gate PASSED for run {run_id}")
    elif "REJECTED" in response.upper():
        run.security_review_status = GateStatus.FAILED
        logger.info(f"Security review gate FAILED for run {run_id}")
    else:
        # Default to pending if unclear
        logger.warning(f"Security review gate unclear for run {run_id}")

    db.commit()


@shared_task(bind=True)
def orchestrator_tick(self, run_id: str) -> Dict[str, Any]:
    """Orchestrator tick - dispatch ready tasks and check completion.

    This task runs periodically to:
    1. Find tasks ready to execute
    2. Dispatch them to worker queues
    3. Check if the run is complete
    """
    from orchestrator.agents.orchestrator import OrchestratorAgent

    logger.info(f"Orchestrator tick for run {run_id}")

    orchestrator = OrchestratorAgent(run_id)

    # Check if already complete
    if orchestrator.is_complete():
        orchestrator.mark_complete(success=True)
        return {"status": "complete"}

    # Get and dispatch ready tasks
    ready_tasks = orchestrator.get_ready_tasks()
    dispatched = []

    for task in ready_tasks:
        try:
            celery_id = orchestrator.dispatch_task(task)
            dispatched.append({
                "task_id": str(task.id),
                "task_type": task.task_type,
                "celery_id": celery_id,
            })
        except Exception as e:
            logger.error(f"Failed to dispatch task {task.id}: {e}")

    # Check iteration limit
    with get_db() as db:
        run = db.query(Run).filter(Run.id == UUID(run_id)).first()
        run.current_iteration += 1

        if run.current_iteration >= run.max_iterations:
            run.status = "needs_input"
            run.blocked_reason = "Maximum iterations reached"
            db.commit()
            return {"status": "iteration_limit", "iteration": run.current_iteration}

        db.commit()

    return {
        "status": "running",
        "dispatched": dispatched,
        "ready_count": len(ready_tasks),
    }


@shared_task(bind=True)
def start_run(self, run_id: str, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Start a new orchestration run.

    Creates the workflow plan and initializes all tasks.
    """
    from orchestrator.agents.orchestrator import OrchestratorAgent

    logger.info(f"Starting run {run_id} with goal: {goal}")

    orchestrator = OrchestratorAgent(run_id)

    try:
        run = orchestrator.initialize_run(goal, context)

        # Schedule first tick
        orchestrator_tick.apply_async(
            args=[run_id],
            countdown=1,  # Start after 1 second
        )

        return {
            "status": "started",
            "run_id": run_id,
            "task_count": len(run.tasks),
        }

    except Exception as e:
        logger.error(f"Failed to start run {run_id}: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }
