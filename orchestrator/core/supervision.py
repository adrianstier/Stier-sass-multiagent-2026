"""Hierarchical Agent Supervision.

Provides:
- Supervisor agents for real-time oversight
- Critique and revise loops within task execution
- Parent agents spawning and managing child agents
- Quality oversight during execution
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from uuid import UUID
import uuid as uuid_module

from orchestrator.core.database import get_db
from orchestrator.core.models import Run, Task, Event, Artifact, TaskStatus
from orchestrator.core.config import settings
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Supervision Hierarchy
# =============================================================================

# Define supervision relationships
SUPERVISION_HIERARCHY = {
    # Supervisor -> List of supervised roles
    "tech_lead": ["backend_engineer", "frontend_engineer", "database_engineer"],
    "project_manager": ["business_analyst", "ux_engineer"],
    "code_reviewer": ["backend_engineer", "frontend_engineer"],
    "security_reviewer": ["backend_engineer", "tech_lead"],
}

# Inverse mapping for quick lookup
SUPERVISED_BY = {}
for supervisor, supervised_list in SUPERVISION_HIERARCHY.items():
    for supervised in supervised_list:
        if supervised not in SUPERVISED_BY:
            SUPERVISED_BY[supervised] = []
        SUPERVISED_BY[supervised].append(supervisor)


# =============================================================================
# Supervision Request/Response
# =============================================================================

@dataclass
class SupervisionRequest:
    """A request for supervision/review during task execution."""

    id: str
    run_id: str
    task_id: str
    agent_role: str
    supervisor_role: str

    request_type: str  # review, approval, guidance, critique
    context: Dict[str, Any]
    artifact_draft: Optional[str] = None
    specific_questions: List[str] = field(default_factory=list)

    status: str = "pending"  # pending, in_progress, completed, timeout
    response: Optional[Dict[str, Any]] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    responded_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "supervisor_role": self.supervisor_role,
            "request_type": self.request_type,
            "context": self.context,
            "artifact_draft": self.artifact_draft[:500] if self.artifact_draft else None,
            "specific_questions": self.specific_questions,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SupervisionResponse:
    """Response from a supervisor agent."""

    request_id: str
    supervisor_role: str

    decision: str  # approve, revise, escalate, reject
    feedback: str
    suggestions: List[str] = field(default_factory=list)
    required_changes: List[Dict[str, str]] = field(default_factory=list)
    answers: Dict[str, str] = field(default_factory=dict)  # Answers to specific questions

    confidence: float = 0.8
    escalate_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "supervisor_role": self.supervisor_role,
            "decision": self.decision,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "required_changes": self.required_changes,
            "answers": self.answers,
            "confidence": self.confidence,
            "escalate_to": self.escalate_to,
        }


# =============================================================================
# Critique and Revise Loop
# =============================================================================

@dataclass
class CritiqueResult:
    """Result of a critique evaluation."""

    passed: bool
    score: float  # 0-1
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    iteration: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "iteration": self.iteration,
        }


class CritiqueLoop:
    """Implements critique-and-revise loops within task execution."""

    def __init__(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        max_iterations: int = 3,
        pass_threshold: float = 0.8,
    ):
        self.run_id = run_id
        self.task_id = task_id
        self.agent_role = agent_role
        self.max_iterations = max_iterations
        self.pass_threshold = pass_threshold
        self.iteration = 0
        self.history: List[CritiqueResult] = []

    def critique(
        self,
        artifact_content: str,
        artifact_type: str,
        criteria: Optional[List[str]] = None,
    ) -> CritiqueResult:
        """Critique an artifact against quality criteria.

        Args:
            artifact_content: The content to critique
            artifact_type: Type of artifact
            criteria: Optional specific criteria to check

        Returns:
            CritiqueResult with evaluation
        """
        self.iteration += 1

        # Default criteria by artifact type
        if criteria is None:
            criteria = self._get_default_criteria(artifact_type)

        # Perform critique (in production, this would use LLM)
        issues = self._evaluate_criteria(artifact_content, criteria)

        # Calculate score
        passed_criteria = len([i for i in issues if i.get("severity") != "critical"])
        total_criteria = len(criteria)
        score = passed_criteria / total_criteria if total_criteria > 0 else 0

        passed = score >= self.pass_threshold and not any(
            i.get("severity") == "critical" for i in issues
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(issues)

        result = CritiqueResult(
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            iteration=self.iteration,
        )

        self.history.append(result)

        # Record event
        self._record_critique_event(result)

        return result

    def _get_default_criteria(self, artifact_type: str) -> List[str]:
        """Get default criteria for artifact type."""
        criteria_map = {
            "requirements_document": [
                "completeness",
                "clarity",
                "testability",
                "consistency",
                "stakeholder_coverage",
            ],
            "architecture_document": [
                "scalability",
                "security",
                "maintainability",
                "completeness",
                "diagram_clarity",
            ],
            "backend_code": [
                "functionality",
                "error_handling",
                "security",
                "performance",
                "code_style",
                "documentation",
            ],
            "frontend_code": [
                "functionality",
                "accessibility",
                "responsiveness",
                "performance",
                "code_style",
            ],
            "database_schema": [
                "normalization",
                "indexing",
                "constraints",
                "naming_conventions",
                "scalability",
            ],
            "default": [
                "completeness",
                "clarity",
                "correctness",
                "consistency",
            ],
        }
        return criteria_map.get(artifact_type, criteria_map["default"])

    def _evaluate_criteria(
        self, content: str, criteria: List[str]
    ) -> List[Dict[str, Any]]:
        """Evaluate content against criteria.

        In production, this would use LLM for evaluation.
        """
        issues = []

        # Simple heuristic checks (in production, use LLM)
        if "completeness" in criteria:
            if len(content) < 500:
                issues.append({
                    "criterion": "completeness",
                    "severity": "medium",
                    "description": "Content appears too brief for comprehensive coverage",
                    "location": "general",
                })

        if "documentation" in criteria:
            if "```" in content and "#" not in content:
                issues.append({
                    "criterion": "documentation",
                    "severity": "low",
                    "description": "Code blocks may lack comments or documentation",
                    "location": "code_blocks",
                })

        if "error_handling" in criteria:
            if "try" not in content.lower() and "except" not in content.lower():
                if "def " in content or "function" in content:
                    issues.append({
                        "criterion": "error_handling",
                        "severity": "medium",
                        "description": "No visible error handling in code",
                        "location": "functions",
                    })

        return issues

    def _generate_suggestions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions from issues."""
        suggestions = []

        for issue in issues:
            if issue["severity"] == "critical":
                suggestions.append(f"CRITICAL: Address {issue['criterion']} - {issue['description']}")
            elif issue["severity"] == "medium":
                suggestions.append(f"Improve {issue['criterion']}: {issue['description']}")
            else:
                suggestions.append(f"Consider: {issue['description']}")

        return suggestions

    def _record_critique_event(self, result: CritiqueResult) -> None:
        """Record critique event for audit."""
        with get_db() as db:
            event = Event(
                run_id=UUID(self.run_id),
                task_id=UUID(self.task_id),
                event_type="critique_evaluation",
                actor=self.agent_role,
                data={
                    "iteration": result.iteration,
                    "passed": result.passed,
                    "score": result.score,
                    "issue_count": len(result.issues),
                    "max_iterations": self.max_iterations,
                },
            )
            db.add(event)
            db.commit()

    def should_continue(self) -> bool:
        """Check if loop should continue."""
        if self.iteration >= self.max_iterations:
            return False
        if self.history and self.history[-1].passed:
            return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of critique loop."""
        return {
            "iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "final_passed": self.history[-1].passed if self.history else None,
            "final_score": self.history[-1].score if self.history else None,
            "total_issues": sum(len(r.issues) for r in self.history),
            "improvement_trend": self._calculate_trend(),
        }

    def _calculate_trend(self) -> str:
        """Calculate score improvement trend."""
        if len(self.history) < 2:
            return "insufficient_data"

        scores = [r.score for r in self.history]
        if scores[-1] > scores[0]:
            return "improving"
        elif scores[-1] < scores[0]:
            return "declining"
        return "stable"


# =============================================================================
# Supervision Manager
# =============================================================================

class SupervisionManager:
    """Manages hierarchical supervision between agents."""

    def __init__(self):
        self._pending_requests: Dict[str, SupervisionRequest] = {}
        self._critique_loops: Dict[str, CritiqueLoop] = {}

    def get_supervisor(self, role: str) -> Optional[str]:
        """Get the supervisor role for a given role."""
        supervisors = SUPERVISED_BY.get(role, [])
        return supervisors[0] if supervisors else None

    def get_all_supervisors(self, role: str) -> List[str]:
        """Get all supervisor roles for a given role."""
        return SUPERVISED_BY.get(role, [])

    def get_supervised_roles(self, supervisor_role: str) -> List[str]:
        """Get all roles supervised by a given supervisor."""
        return SUPERVISION_HIERARCHY.get(supervisor_role, [])

    def request_supervision(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        request_type: str,
        context: Dict[str, Any],
        artifact_draft: Optional[str] = None,
        questions: Optional[List[str]] = None,
        supervisor_role: Optional[str] = None,
    ) -> SupervisionRequest:
        """Request supervision from a supervisor agent.

        Args:
            run_id: The run ID
            task_id: The task ID
            agent_role: The requesting agent's role
            request_type: Type of supervision requested
            context: Context for the request
            artifact_draft: Optional draft artifact to review
            questions: Specific questions for supervisor
            supervisor_role: Optional specific supervisor

        Returns:
            SupervisionRequest
        """
        # Determine supervisor
        if supervisor_role is None:
            supervisor_role = self.get_supervisor(agent_role)

        if supervisor_role is None:
            raise ValueError(f"No supervisor found for role: {agent_role}")

        request = SupervisionRequest(
            id=str(uuid_module.uuid4()),
            run_id=run_id,
            task_id=task_id,
            agent_role=agent_role,
            supervisor_role=supervisor_role,
            request_type=request_type,
            context=context,
            artifact_draft=artifact_draft,
            specific_questions=questions or [],
        )

        self._pending_requests[request.id] = request

        # Record event
        self._record_event(run_id, task_id, "supervision_requested", request.to_dict())

        logger.info(
            "supervision_requested",
            run_id=run_id,
            task_id=task_id,
            agent=agent_role,
            supervisor=supervisor_role,
            request_type=request_type,
        )

        return request

    def respond_to_supervision(
        self,
        request_id: str,
        decision: str,
        feedback: str,
        suggestions: Optional[List[str]] = None,
        required_changes: Optional[List[Dict[str, str]]] = None,
        answers: Optional[Dict[str, str]] = None,
        escalate_to: Optional[str] = None,
    ) -> SupervisionResponse:
        """Respond to a supervision request.

        Args:
            request_id: The request ID
            decision: Decision (approve, revise, escalate, reject)
            feedback: Feedback text
            suggestions: Optional suggestions
            required_changes: Optional required changes
            answers: Answers to specific questions
            escalate_to: Optional role to escalate to

        Returns:
            SupervisionResponse
        """
        request = self._pending_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        response = SupervisionResponse(
            request_id=request_id,
            supervisor_role=request.supervisor_role,
            decision=decision,
            feedback=feedback,
            suggestions=suggestions or [],
            required_changes=required_changes or [],
            answers=answers or {},
            escalate_to=escalate_to,
        )

        # Update request
        request.status = "completed"
        request.response = response.to_dict()
        request.responded_at = datetime.utcnow()

        # Record event
        self._record_event(
            request.run_id,
            request.task_id,
            "supervision_responded",
            response.to_dict(),
        )

        logger.info(
            "supervision_responded",
            request_id=request_id,
            decision=decision,
            supervisor=request.supervisor_role,
        )

        # Handle escalation
        if decision == "escalate" and escalate_to:
            self._escalate_supervision(request, escalate_to)

        return response

    def _escalate_supervision(
        self, original_request: SupervisionRequest, escalate_to: str
    ) -> SupervisionRequest:
        """Escalate a supervision request to a higher authority."""
        return self.request_supervision(
            run_id=original_request.run_id,
            task_id=original_request.task_id,
            agent_role=original_request.agent_role,
            request_type="escalation",
            context={
                "original_request_id": original_request.id,
                "original_supervisor": original_request.supervisor_role,
                **original_request.context,
            },
            artifact_draft=original_request.artifact_draft,
            questions=original_request.specific_questions,
            supervisor_role=escalate_to,
        )

    def create_critique_loop(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        max_iterations: int = 3,
        pass_threshold: float = 0.8,
    ) -> CritiqueLoop:
        """Create a critique loop for a task.

        Args:
            run_id: The run ID
            task_id: The task ID
            agent_role: The agent role
            max_iterations: Maximum revision iterations
            pass_threshold: Score threshold to pass

        Returns:
            CritiqueLoop instance
        """
        loop = CritiqueLoop(
            run_id=run_id,
            task_id=task_id,
            agent_role=agent_role,
            max_iterations=max_iterations,
            pass_threshold=pass_threshold,
        )

        loop_key = f"{run_id}:{task_id}"
        self._critique_loops[loop_key] = loop

        return loop

    def get_critique_loop(self, run_id: str, task_id: str) -> Optional[CritiqueLoop]:
        """Get an existing critique loop."""
        loop_key = f"{run_id}:{task_id}"
        return self._critique_loops.get(loop_key)

    def get_pending_requests(
        self,
        supervisor_role: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[SupervisionRequest]:
        """Get pending supervision requests."""
        requests = list(self._pending_requests.values())

        if supervisor_role:
            requests = [r for r in requests if r.supervisor_role == supervisor_role]

        if run_id:
            requests = [r for r in requests if r.run_id == run_id]

        return [r for r in requests if r.status == "pending"]

    def _record_event(
        self, run_id: str, task_id: str, event_type: str, data: Dict[str, Any]
    ) -> None:
        """Record a supervision event."""
        with get_db() as db:
            event = Event(
                run_id=UUID(run_id),
                task_id=UUID(task_id),
                event_type=event_type,
                actor="supervision_manager",
                data=data,
            )
            db.add(event)
            db.commit()


# =============================================================================
# Supervised Task Execution
# =============================================================================

class SupervisedTaskExecutor:
    """Executes tasks with supervision support."""

    def __init__(self, supervision_manager: SupervisionManager):
        self.supervision_manager = supervision_manager

    def execute_with_supervision(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        execute_fn: Callable[[], Dict[str, Any]],
        require_approval: bool = False,
        critique_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Execute a task with supervision.

        Args:
            run_id: The run ID
            task_id: The task ID
            agent_role: The agent role
            execute_fn: Function that performs the actual task
            require_approval: Whether supervisor approval is required
            critique_enabled: Whether to enable critique loop

        Returns:
            Execution result with supervision details
        """
        result = {
            "success": False,
            "supervision_applied": False,
            "critique_iterations": 0,
            "supervisor_approval": None,
        }

        try:
            # Initial execution
            execution_result = execute_fn()

            if not execution_result.get("success"):
                return execution_result

            # Apply critique loop if enabled
            if critique_enabled:
                artifact = execution_result.get("artifacts", [{}])[0]
                if artifact:
                    critique_result = self._apply_critique_loop(
                        run_id, task_id, agent_role, artifact
                    )
                    result["critique_iterations"] = critique_result.get("iterations", 0)
                    result["critique_passed"] = critique_result.get("passed", True)

            # Request supervisor approval if required
            if require_approval:
                supervisor = self.supervision_manager.get_supervisor(agent_role)
                if supervisor:
                    approval_result = self._request_approval(
                        run_id, task_id, agent_role, execution_result
                    )
                    result["supervisor_approval"] = approval_result
                    result["supervision_applied"] = True

                    if not approval_result.get("approved"):
                        result["success"] = False
                        result["revision_required"] = True
                        result["feedback"] = approval_result.get("feedback")
                        return result

            result["success"] = True
            result.update(execution_result)
            return result

        except Exception as e:
            logger.error("supervised_execution_failed", error=str(e))
            result["error"] = str(e)
            return result

    def _apply_critique_loop(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        artifact: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply critique loop to an artifact."""
        loop = self.supervision_manager.create_critique_loop(
            run_id, task_id, agent_role
        )

        content = artifact.get("content", "")
        artifact_type = artifact.get("type", "default")

        while loop.should_continue():
            critique = loop.critique(content, artifact_type)

            if critique.passed:
                break

            # In production, would revise content based on feedback
            # For now, just track iterations
            logger.debug(
                "critique_iteration",
                iteration=critique.iteration,
                passed=critique.passed,
                score=critique.score,
            )

        return loop.get_summary()

    def _request_approval(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Request supervisor approval."""
        request = self.supervision_manager.request_supervision(
            run_id=run_id,
            task_id=task_id,
            agent_role=agent_role,
            request_type="approval",
            context={
                "artifacts_count": len(execution_result.get("artifacts", [])),
                "tool_calls": len(execution_result.get("tool_calls", [])),
            },
            artifact_draft=execution_result.get("response"),
        )

        # In production, would wait for or trigger supervisor agent
        # For now, return pending status
        return {
            "request_id": request.id,
            "supervisor": request.supervisor_role,
            "status": "pending",
            "approved": None,  # Would be filled by supervisor response
        }


# Global supervision manager instance
_supervision_manager: Optional[SupervisionManager] = None


def get_supervision_manager() -> SupervisionManager:
    """Get the global supervision manager instance."""
    global _supervision_manager
    if _supervision_manager is None:
        _supervision_manager = SupervisionManager()
    return _supervision_manager
