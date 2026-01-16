"""Celery tasks for agent execution."""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from celery import shared_task

from orchestrator.core.database import get_db
from orchestrator.core.models import Task, Run, Artifact, TaskStatus, GateStatus, RunStatus
from orchestrator.agents.specialists import get_agent_class

logger = logging.getLogger(__name__)


# =============================================================================
# Gate Decision Parsing
# =============================================================================

class GateDecision:
    """Structured gate decision from reviewer agents."""

    def __init__(
        self,
        status: str,  # "APPROVED" or "REJECTED"
        summary: str,
        issues: List[Dict[str, Any]] = None,
        recommendations: List[str] = None,
        confidence: float = 1.0,
    ):
        self.status = status.upper()
        self.summary = summary
        self.issues = issues or []
        self.recommendations = recommendations or []
        self.confidence = confidence

    @classmethod
    def parse_from_response(cls, response: str) -> Optional["GateDecision"]:
        """Parse a gate decision from LLM response.

        Supports multiple formats:
        1. JSON block: ```json { "gate_decision": {...} } ```
        2. Structured markers: ## DECISION: APPROVED/REJECTED
        3. Fallback: String matching for APPROVED/REJECTED keywords
        """
        # Try JSON format first (most reliable)
        json_decision = cls._parse_json_format(response)
        if json_decision:
            return json_decision

        # Try structured marker format
        marker_decision = cls._parse_marker_format(response)
        if marker_decision:
            return marker_decision

        # Fallback to keyword detection (least reliable)
        return cls._parse_keyword_format(response)

    @classmethod
    def _parse_json_format(cls, response: str) -> Optional["GateDecision"]:
        """Parse JSON gate decision block."""
        # Look for JSON code block with gate_decision
        json_pattern = r'```(?:json)?\s*(\{[^`]*"gate_decision"[^`]*\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)

        if match:
            try:
                data = json.loads(match.group(1))
                decision_data = data.get("gate_decision", data)
                return cls(
                    status=decision_data.get("status", "PENDING"),
                    summary=decision_data.get("summary", ""),
                    issues=decision_data.get("issues", []),
                    recommendations=decision_data.get("recommendations", []),
                    confidence=decision_data.get("confidence", 1.0),
                )
            except json.JSONDecodeError:
                pass

        # Try inline JSON object
        inline_pattern = r'\{[^{}]*"gate_decision"[^{}]*\{[^{}]*\}[^{}]*\}'
        match = re.search(inline_pattern, response)
        if match:
            try:
                data = json.loads(match.group())
                decision_data = data.get("gate_decision", data)
                return cls(
                    status=decision_data.get("status", "PENDING"),
                    summary=decision_data.get("summary", ""),
                    issues=decision_data.get("issues", []),
                    recommendations=decision_data.get("recommendations", []),
                    confidence=decision_data.get("confidence", 1.0),
                )
            except json.JSONDecodeError:
                pass

        return None

    @classmethod
    def _parse_marker_format(cls, response: str) -> Optional["GateDecision"]:
        """Parse structured marker format like ## DECISION: APPROVED"""
        # Look for explicit decision markers
        patterns = [
            r'##\s*(?:FINAL\s+)?DECISION\s*:\s*(APPROVED|REJECTED)',
            r'\*\*(?:FINAL\s+)?DECISION\*\*\s*:\s*(APPROVED|REJECTED)',
            r'GATE\s+(?:STATUS|DECISION)\s*:\s*(APPROVED|REJECTED)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                status = match.group(1).upper()
                # Try to extract summary from nearby text
                summary = cls._extract_summary_near_decision(response, match.end())
                return cls(
                    status=status,
                    summary=summary,
                    confidence=0.9,  # High confidence with explicit marker
                )

        return None

    @classmethod
    def _parse_keyword_format(cls, response: str) -> Optional["GateDecision"]:
        """Fallback keyword-based parsing."""
        response_upper = response.upper()

        # Count occurrences and context
        approved_count = response_upper.count("APPROVED")
        rejected_count = response_upper.count("REJECTED")

        # Look for conclusive statements
        conclusive_approved = any([
            "I APPROVE" in response_upper,
            "THIS IS APPROVED" in response_upper,
            "REVIEW: APPROVED" in response_upper,
            "VERDICT: APPROVED" in response_upper,
            "STATUS: APPROVED" in response_upper,
        ])
        conclusive_rejected = any([
            "I REJECT" in response_upper,
            "THIS IS REJECTED" in response_upper,
            "REVIEW: REJECTED" in response_upper,
            "VERDICT: REJECTED" in response_upper,
            "STATUS: REJECTED" in response_upper,
        ])

        if conclusive_approved and not conclusive_rejected:
            return cls(status="APPROVED", summary="Keyword match: conclusive approval", confidence=0.8)
        elif conclusive_rejected and not conclusive_approved:
            return cls(status="REJECTED", summary="Keyword match: conclusive rejection", confidence=0.8)
        elif approved_count > rejected_count and approved_count > 0:
            return cls(status="APPROVED", summary="Keyword match: more approvals", confidence=0.5)
        elif rejected_count > approved_count and rejected_count > 0:
            return cls(status="REJECTED", summary="Keyword match: more rejections", confidence=0.5)

        return None

    @staticmethod
    def _extract_summary_near_decision(response: str, position: int) -> str:
        """Extract summary text near the decision marker."""
        # Get text after the decision (up to 200 chars or next section)
        remaining = response[position:position + 500]
        # Stop at next heading or end of paragraph
        match = re.match(r'[:\s]*([^\n#]+)', remaining)
        if match:
            return match.group(1).strip()[:200]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "status": self.status,
            "summary": self.summary,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }


# =============================================================================
# Task Execution
# =============================================================================

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
    3. Gathers input from dependencies (ALL artifacts, not just first)
    4. Executes the agent
    5. Saves results and artifacts
    6. Schedules next orchestrator tick
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

        # Gather input from dependencies - ALL artifacts, not just first
        dependencies_output = _gather_dependency_artifacts(db, task.dependencies)

        # Get accumulated run context
        run_context = _get_accumulated_context(db, run_id)

        # Prepare task input
        task_input = {
            "goal": run.goal,
            "description": task.description,
            "task_type": task.task_type,
            "dependencies_output": dependencies_output,
            "context": {**(run.context or {}), **run_context},
        }

    # Execute agent (outside DB session to avoid long transactions)
    try:
        agent = agent_class(run_id, task_id)
        result = agent.execute(task_input)
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        _handle_task_failure_with_retry(run_id, task_id, str(e))
        raise

    # Save results
    with get_db() as db:
        task = db.query(Task).filter(Task.id == UUID(task_id)).first()

        if result.get("success"):
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow()

            # Handle quality gates with improved parsing
            if task.assigned_role == "code_reviewer":
                _handle_gate_decision(db, run_id, result, "code_review")
            elif task.assigned_role == "security_reviewer":
                _handle_gate_decision(db, run_id, result, "security_review")
            elif task.assigned_role == "design_reviewer":
                _handle_gate_decision(db, run_id, result, "design_review")

        else:
            task.status = TaskStatus.FAILED
            task.error_message = result.get("error", "Unknown error")
            task.completed_at = datetime.utcnow()

        db.commit()

    # Schedule next orchestrator tick to continue the workflow
    orchestrator_tick.apply_async(
        args=[run_id],
        countdown=2,  # Small delay to let DB settle
    )

    logger.info(f"Task {task_id} completed with status: {task.status}")
    return result


def _gather_dependency_artifacts(db, dependencies: List[str]) -> Dict[str, Any]:
    """Gather ALL artifacts from dependency tasks, not just the first one.

    Returns a structured dict with all artifacts organized by task type.
    """
    dependencies_output = {}

    for dep_id in dependencies:
        dep_task = db.query(Task).filter(Task.id == UUID(dep_id)).first()
        if dep_task and dep_task.status == TaskStatus.COMPLETED:
            # Get ALL artifacts from this dependency
            dep_artifacts = db.query(Artifact).filter(
                Artifact.task_id == UUID(dep_id)
            ).order_by(Artifact.created_at).all()

            if dep_artifacts:
                if len(dep_artifacts) == 1:
                    # Single artifact - just use content directly
                    dependencies_output[dep_task.task_type] = dep_artifacts[0].content
                else:
                    # Multiple artifacts - include all with metadata
                    dependencies_output[dep_task.task_type] = {
                        "primary": dep_artifacts[0].content,
                        "artifacts": [
                            {
                                "name": a.name,
                                "type": a.artifact_type,
                                "content": a.content,
                            }
                            for a in dep_artifacts
                        ],
                    }

    return dependencies_output


def _get_accumulated_context(db, run_id: str) -> Dict[str, Any]:
    """Get accumulated context from all completed phases.

    This provides agents with a summary of what's been done so far.
    """
    completed_tasks = db.query(Task).filter(
        Task.run_id == UUID(run_id),
        Task.status == TaskStatus.COMPLETED,
    ).order_by(Task.completed_at).all()

    context = {
        "completed_phases": [t.task_type for t in completed_tasks],
        "phase_summaries": {},
    }

    # Add brief summaries from each phase
    for task in completed_tasks:
        artifact = db.query(Artifact).filter(
            Artifact.task_id == task.id
        ).first()
        if artifact:
            # Extract first 500 chars as summary
            context["phase_summaries"][task.task_type] = {
                "role": task.assigned_role,
                "summary": artifact.content[:500] if artifact.content else "",
            }

    return context


def _handle_task_failure_with_retry(run_id: str, task_id: str, error: str) -> None:
    """Handle task failure with retry logic."""
    from orchestrator.agents.orchestrator import OrchestratorAgent

    orchestrator = OrchestratorAgent(run_id)
    should_retry = orchestrator.handle_task_failure(task_id, error)

    if should_retry:
        # Re-queue the task for retry
        with get_db() as db:
            task = db.query(Task).filter(Task.id == UUID(task_id)).first()
            logger.info(f"Task {task_id} scheduled for retry (attempt {task.retry_count})")

        # Schedule retry via orchestrator tick
        orchestrator_tick.apply_async(
            args=[run_id],
            countdown=30,  # Delay before retry
        )


def _handle_gate_decision(
    db, run_id: str, result: Dict[str, Any], gate_type: str
) -> None:
    """Handle quality gate decision with improved parsing."""
    run = db.query(Run).filter(Run.id == UUID(run_id)).first()
    response = result.get("response", "")

    # Parse the gate decision using structured parser
    decision = GateDecision.parse_from_response(response)

    if decision:
        # Store decision details in result for audit
        result["gate_decision"] = decision.to_dict()

        if decision.status == "APPROVED":
            if gate_type == "code_review":
                run.code_review_status = GateStatus.PASSED
            elif gate_type == "security_review":
                run.security_review_status = GateStatus.PASSED
            logger.info(
                f"{gate_type} gate PASSED for run {run_id} "
                f"(confidence: {decision.confidence})"
            )
        elif decision.status == "REJECTED":
            if gate_type == "code_review":
                run.code_review_status = GateStatus.FAILED
            elif gate_type == "security_review":
                run.security_review_status = GateStatus.FAILED
            logger.info(
                f"{gate_type} gate FAILED for run {run_id}: {decision.summary} "
                f"(confidence: {decision.confidence})"
            )

            # If low confidence rejection, log warning
            if decision.confidence < 0.7:
                logger.warning(
                    f"Low confidence gate decision for {run_id} - "
                    "consider manual review"
                )
    else:
        # No clear decision found - mark as needs input
        logger.warning(
            f"{gate_type} gate decision unclear for run {run_id} - "
            "requires manual review"
        )
        run.blocked_reason = f"{gate_type} decision unclear - manual review required"

    db.commit()


# =============================================================================
# Gate Override/Waiver API
# =============================================================================

def waive_gate(run_id: str, gate_type: str, reason: str, waived_by: str) -> bool:
    """Manually waive a quality gate (requires admin/owner permission).

    Args:
        run_id: The run ID
        gate_type: "code_review" or "security_review"
        reason: Reason for waiving the gate
        waived_by: User ID who waived

    Returns:
        True if gate was waived successfully
    """
    with get_db() as db:
        run = db.query(Run).filter(Run.id == UUID(run_id)).first()
        if not run:
            return False

        from orchestrator.core.models import Event

        if gate_type == "code_review":
            run.code_review_status = GateStatus.WAIVED
        elif gate_type == "security_review":
            run.security_review_status = GateStatus.WAIVED
        else:
            return False

        # Clear any blocked reason
        if run.blocked_reason and gate_type in run.blocked_reason:
            run.blocked_reason = None

        # Record waiver event for audit trail
        event = Event(
            run_id=UUID(run_id),
            event_type="gate_waived",
            actor=waived_by,
            data={
                "gate_type": gate_type,
                "reason": reason,
                "waived_by": waived_by,
            },
        )
        db.add(event)
        db.commit()

        logger.info(f"Gate {gate_type} waived for run {run_id} by {waived_by}: {reason}")

        # Trigger orchestrator to continue if blocked
        orchestrator_tick.apply_async(args=[run_id], countdown=1)

        return True


def override_gate(
    run_id: str, gate_type: str, decision: str, reason: str, overridden_by: str
) -> bool:
    """Manually override a gate decision (set to PASSED or FAILED).

    Args:
        run_id: The run ID
        gate_type: "code_review" or "security_review"
        decision: "PASSED" or "FAILED"
        reason: Reason for override
        overridden_by: User ID who overrode

    Returns:
        True if gate was overridden successfully
    """
    with get_db() as db:
        run = db.query(Run).filter(Run.id == UUID(run_id)).first()
        if not run:
            return False

        from orchestrator.core.models import Event

        new_status = GateStatus.PASSED if decision.upper() == "PASSED" else GateStatus.FAILED

        if gate_type == "code_review":
            run.code_review_status = new_status
        elif gate_type == "security_review":
            run.security_review_status = new_status
        else:
            return False

        # Clear blocked reason if approving
        if new_status == GateStatus.PASSED and run.blocked_reason:
            run.blocked_reason = None

        # Record override event for audit trail
        event = Event(
            run_id=UUID(run_id),
            event_type="gate_overridden",
            actor=overridden_by,
            data={
                "gate_type": gate_type,
                "decision": decision.upper(),
                "reason": reason,
                "overridden_by": overridden_by,
            },
        )
        db.add(event)
        db.commit()

        logger.info(
            f"Gate {gate_type} overridden to {decision} for run {run_id} "
            f"by {overridden_by}: {reason}"
        )

        # Trigger orchestrator to continue
        orchestrator_tick.apply_async(args=[run_id], countdown=1)

        return True


# =============================================================================
# Orchestrator Tasks
# =============================================================================

@shared_task(bind=True)
def orchestrator_tick(self, run_id: str) -> Dict[str, Any]:
    """Orchestrator tick - dispatch ready tasks and check completion.

    This task runs periodically to:
    1. Find tasks ready to execute
    2. Dispatch them to worker queues
    3. Check if the run is complete
    4. Schedule next tick if work remains
    """
    from orchestrator.agents.orchestrator import OrchestratorAgent

    logger.info(f"Orchestrator tick for run {run_id}")

    orchestrator = OrchestratorAgent(run_id)

    # Check if run is blocked or paused
    with get_db() as db:
        run = db.query(Run).filter(Run.id == UUID(run_id)).first()
        if run.status in [RunStatus.NEEDS_INPUT, RunStatus.CANCELLED]:
            return {"status": run.status.value, "message": run.blocked_reason}

    # Check if already complete
    if orchestrator.is_complete():
        orchestrator.mark_complete(success=True)
        return {"status": "complete"}

    # Check for failed gates that block progress
    gates = orchestrator.check_gates()
    if gates.get("code_review") == GateStatus.FAILED:
        orchestrator.mark_needs_input("Code review failed - requires fixes")
        return {"status": "blocked", "reason": "code_review_failed"}
    if gates.get("security_review") == GateStatus.FAILED:
        orchestrator.mark_needs_input("Security review failed - requires fixes")
        return {"status": "blocked", "reason": "security_review_failed"}

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
            run.status = RunStatus.NEEDS_INPUT
            run.blocked_reason = "Maximum iterations reached"
            db.commit()
            return {"status": "iteration_limit", "iteration": run.current_iteration}

        db.commit()

    # If tasks are running but none dispatched, schedule another tick
    with get_db() as db:
        running_count = db.query(Task).filter(
            Task.run_id == UUID(run_id),
            Task.status.in_([TaskStatus.RUNNING, TaskStatus.QUEUED]),
        ).count()

        if running_count > 0 and len(dispatched) == 0:
            # Tasks still running, schedule follow-up tick
            orchestrator_tick.apply_async(
                args=[run_id],
                countdown=10,  # Check again in 10 seconds
            )

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


# =============================================================================
# Celery Beat Tasks - Automatic Monitoring
# =============================================================================

@shared_task(bind=True)
def monitor_active_runs(self) -> Dict[str, Any]:
    """Monitor all active runs and trigger ticks as needed.

    This task runs periodically via Celery Beat to ensure all
    active runs make progress even if individual tick scheduling fails.
    """
    logger.info("Monitoring active runs")

    with get_db() as db:
        # Find all runs that should be active
        active_runs = db.query(Run).filter(
            Run.status == RunStatus.RUNNING
        ).all()

        triggered = []
        for run in active_runs:
            run_id = str(run.id)

            # Check if run has pending tasks that could be dispatched
            pending_count = db.query(Task).filter(
                Task.run_id == run.id,
                Task.status == TaskStatus.PENDING,
            ).count()

            running_count = db.query(Task).filter(
                Task.run_id == run.id,
                Task.status.in_([TaskStatus.RUNNING, TaskStatus.QUEUED]),
            ).count()

            # Trigger tick if there are pending tasks but none running
            # (indicates the workflow might be stuck)
            if pending_count > 0 and running_count == 0:
                orchestrator_tick.apply_async(args=[run_id], countdown=1)
                triggered.append(run_id)
                logger.info(f"Triggered tick for stuck run {run_id}")

    return {
        "active_runs": len(active_runs) if active_runs else 0,
        "triggered_ticks": triggered,
    }


@shared_task(bind=True)
def cleanup_stale_tasks(self) -> Dict[str, Any]:
    """Clean up tasks that have been running too long.

    Marks tasks as failed if they've exceeded the timeout.
    """
    from datetime import timedelta
    from orchestrator.core.config import settings

    logger.info("Cleaning up stale tasks")

    timeout = timedelta(seconds=settings.task_timeout_seconds * 2)  # 2x safety margin
    cutoff = datetime.utcnow() - timeout

    cleaned = []
    affected_runs = set()
    with get_db() as db:
        # Find tasks stuck in RUNNING state
        stale_tasks = db.query(Task).filter(
            Task.status == TaskStatus.RUNNING,
            Task.started_at < cutoff,
        ).all()

        for task in stale_tasks:
            task.status = TaskStatus.FAILED
            task.error_message = "Task timed out (exceeded maximum execution time)"
            task.completed_at = datetime.utcnow()
            cleaned.append(str(task.id))
            affected_runs.add(str(task.run_id))

            # Record event
            from orchestrator.core.models import Event
            event = Event(
                run_id=task.run_id,
                task_id=task.id,
                event_type="task_timeout",
                actor="cleanup",
                data={
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "timeout_seconds": settings.task_timeout_seconds * 2,
                },
            )
            db.add(event)

        db.commit()

        # Trigger ticks for affected runs to continue workflow
        for run_id in affected_runs:
            orchestrator_tick.apply_async(args=[run_id], countdown=5)

    logger.info(f"Cleaned up {len(cleaned)} stale tasks")
    return {
        "cleaned_tasks": cleaned,
        "affected_runs": list(affected_runs),
    }
