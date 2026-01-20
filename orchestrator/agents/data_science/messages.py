"""Inter-agent communication protocols for Data Science Multi-Agent Framework.

This module provides message classes and utilities for:
- Agent-to-agent communication
- Task delegation and completion tracking
- Clarification requests and responses
- Artifact sharing and handoffs
- Quality gate notifications
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid
import json


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK_DELEGATION = "task_delegation"
    TASK_COMPLETION = "task_completion"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    HANDOFF = "handoff"
    FEEDBACK = "feedback"
    QUALITY_GATE_ALERT = "quality_gate_alert"
    ARTIFACT_NOTIFICATION = "artifact_notification"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"


@dataclass
class MessageMetadata:
    """Metadata for all inter-agent messages."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: Optional[str] = None  # Links related messages
    session_id: Optional[str] = None  # Links to overall workflow session
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: Optional[int] = None  # Time to live
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BaseMessage:
    """Base class for all inter-agent messages."""
    message_type: MessageType = field(default=MessageType.TASK_DELEGATION)
    sender_agent: str = ""
    recipient_agent: str = ""
    metadata: MessageMetadata = field(default_factory=MessageMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_type": self.message_type.value,
            "sender_agent": self.sender_agent,
            "recipient_agent": self.recipient_agent,
            "metadata": {
                "message_id": self.metadata.message_id,
                "timestamp": self.metadata.timestamp,
                "correlation_id": self.metadata.correlation_id,
                "session_id": self.metadata.session_id,
                "priority": self.metadata.priority.value,
                "ttl_seconds": self.metadata.ttl_seconds,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
            }
        }

    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TaskDelegationMessage(BaseMessage):
    """Message for delegating tasks between agents."""
    message_type: MessageType = MessageType.TASK_DELEGATION

    # Task details
    task_id: str = ""
    task_description: str = ""
    task_type: str = ""  # e.g., "data_cleaning", "model_training"

    # Context from previous agents
    context: Dict[str, Any] = field(default_factory=dict)
    predecessor_outputs: List[str] = field(default_factory=list)

    # Requirements
    required_inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)

    # Constraints
    deadline: Optional[str] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "task_description": self.task_description,
            "task_type": self.task_type,
            "context": self.context,
            "predecessor_outputs": self.predecessor_outputs,
            "required_inputs": self.required_inputs,
            "expected_outputs": self.expected_outputs,
            "quality_requirements": self.quality_requirements,
            "deadline": self.deadline,
            "resource_limits": self.resource_limits,
        })
        return base


@dataclass
class TaskCompletionMessage(BaseMessage):
    """Message indicating task completion."""
    message_type: MessageType = MessageType.TASK_COMPLETION

    # Reference to original task
    task_id: str = ""
    original_delegation_id: str = ""

    # Completion status
    status: str = "completed"  # completed, partial, failed

    # Results
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts_produced: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Quality assessment
    quality_score: Optional[float] = None
    quality_details: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Execution details
    execution_time_seconds: Optional[float] = None
    resources_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "original_delegation_id": self.original_delegation_id,
            "status": self.status,
            "outputs": self.outputs,
            "artifacts_produced": self.artifacts_produced,
            "metrics": self.metrics,
            "quality_score": self.quality_score,
            "quality_details": self.quality_details,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "execution_time_seconds": self.execution_time_seconds,
            "resources_used": self.resources_used,
        })
        return base


@dataclass
class ClarificationRequestMessage(BaseMessage):
    """Message requesting clarification from another agent."""
    message_type: MessageType = MessageType.CLARIFICATION_REQUEST

    # Reference
    task_id: str = ""

    # Clarification details
    question: str = ""
    question_type: str = ""  # requirement, technical, data, methodology
    options: List[str] = field(default_factory=list)  # If multiple choice
    default_option: Optional[str] = None

    # Context
    context: str = ""
    blocking: bool = True  # Whether this blocks progress

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "question": self.question,
            "question_type": self.question_type,
            "options": self.options,
            "default_option": self.default_option,
            "context": self.context,
            "blocking": self.blocking,
        })
        return base


@dataclass
class ClarificationResponseMessage(BaseMessage):
    """Response to a clarification request."""
    message_type: MessageType = MessageType.CLARIFICATION_RESPONSE

    # Reference to original request
    original_request_id: str = ""
    task_id: str = ""

    # Response
    response: str = ""
    selected_option: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "original_request_id": self.original_request_id,
            "task_id": self.task_id,
            "response": self.response,
            "selected_option": self.selected_option,
            "additional_context": self.additional_context,
        })
        return base


@dataclass
class HandoffMessage(BaseMessage):
    """Message for handing off work between agents."""
    message_type: MessageType = MessageType.HANDOFF

    # Task reference
    task_id: str = ""

    # Handoff details
    reason: str = ""  # Why handoff is happening
    handoff_type: str = ""  # sequential, parallel, escalation, specialization

    # State transfer
    current_state: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    context_summary: str = ""

    # Recommendations for receiving agent
    recommendations: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "reason": self.reason,
            "handoff_type": self.handoff_type,
            "current_state": self.current_state,
            "artifacts": self.artifacts,
            "context_summary": self.context_summary,
            "recommendations": self.recommendations,
            "known_issues": self.known_issues,
        })
        return base


@dataclass
class FeedbackMessage(BaseMessage):
    """Feedback on another agent's work."""
    message_type: MessageType = MessageType.FEEDBACK

    # Reference
    task_id: str = ""
    completion_message_id: str = ""

    # Feedback details
    feedback_type: str = ""  # approval, revision_request, quality_concern
    rating: Optional[float] = None  # 0-1 scale

    # Content
    summary: str = ""
    positive_points: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)

    # Next steps
    action_required: str = ""  # none, revise, escalate

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "completion_message_id": self.completion_message_id,
            "feedback_type": self.feedback_type,
            "rating": self.rating,
            "summary": self.summary,
            "positive_points": self.positive_points,
            "issues": self.issues,
            "required_changes": self.required_changes,
            "action_required": self.action_required,
        })
        return base


@dataclass
class QualityGateAlertMessage(BaseMessage):
    """Alert when quality gates are triggered."""
    message_type: MessageType = MessageType.QUALITY_GATE_ALERT

    # Reference
    task_id: str = ""
    agent_name: str = ""

    # Gate details
    gate_name: str = ""
    gate_type: str = ""  # performance, fairness, data_quality, stability

    # Status
    passed: bool = False
    threshold: float = 0.0
    actual_value: float = 0.0

    # Details
    description: str = ""
    impact: str = ""  # low, medium, high, critical
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "gate_name": self.gate_name,
            "gate_type": self.gate_type,
            "passed": self.passed,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "description": self.description,
            "impact": self.impact,
            "recommendations": self.recommendations,
        })
        return base


@dataclass
class ArtifactNotificationMessage(BaseMessage):
    """Notification about artifact creation/update."""
    message_type: MessageType = MessageType.ARTIFACT_NOTIFICATION

    # Artifact details
    artifact_id: str = ""
    artifact_type: str = ""  # dataset, model, report, visualization, etc.
    artifact_path: str = ""

    # Metadata
    version: str = ""
    size_bytes: Optional[int] = None
    format: str = ""

    # Context
    task_id: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Lineage
    parent_artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "artifact_path": self.artifact_path,
            "version": self.version,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "task_id": self.task_id,
            "description": self.description,
            "tags": self.tags,
            "parent_artifacts": self.parent_artifacts,
        })
        return base


@dataclass
class StatusUpdateMessage(BaseMessage):
    """Status update during long-running tasks."""
    message_type: MessageType = MessageType.STATUS_UPDATE

    # Reference
    task_id: str = ""

    # Progress
    status: str = ""  # started, in_progress, near_completion, blocked
    progress_percent: Optional[float] = None
    current_step: str = ""

    # Details
    message: str = ""
    estimated_completion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "message": self.message,
            "estimated_completion": self.estimated_completion,
        })
        return base


@dataclass
class ErrorReportMessage(BaseMessage):
    """Report of errors during task execution."""
    message_type: MessageType = MessageType.ERROR_REPORT

    # Reference
    task_id: str = ""

    # Error details
    error_type: str = ""  # validation, execution, resource, timeout, etc.
    error_code: str = ""
    error_message: str = ""

    # Context
    stack_trace: Optional[str] = None
    failed_step: str = ""
    partial_results: Dict[str, Any] = field(default_factory=dict)

    # Recovery
    recoverable: bool = False
    recovery_suggestions: List[str] = field(default_factory=list)
    retry_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "task_id": self.task_id,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "failed_step": self.failed_step,
            "partial_results": self.partial_results,
            "recoverable": self.recoverable,
            "recovery_suggestions": self.recovery_suggestions,
            "retry_recommended": self.retry_recommended,
        })
        return base


# Message Bus Implementation
class MessageBus:
    """Simple in-memory message bus for agent communication."""

    def __init__(self):
        self._queues: Dict[str, List[BaseMessage]] = {}
        self._handlers: Dict[str, List[callable]] = {}
        self._message_history: List[BaseMessage] = []

    def subscribe(self, agent_name: str, handler: callable = None) -> None:
        """Subscribe an agent to receive messages."""
        if agent_name not in self._queues:
            self._queues[agent_name] = []
        if handler and agent_name not in self._handlers:
            self._handlers[agent_name] = []
        if handler:
            self._handlers[agent_name].append(handler)

    def publish(self, message: BaseMessage) -> None:
        """Publish a message to the recipient's queue."""
        recipient = message.recipient_agent

        # Ensure recipient queue exists
        if recipient not in self._queues:
            self._queues[recipient] = []

        # Add to queue
        self._queues[recipient].append(message)

        # Add to history
        self._message_history.append(message)

        # Call handlers
        if recipient in self._handlers:
            for handler in self._handlers[recipient]:
                handler(message)

    def get_messages(
        self,
        agent_name: str,
        message_type: Optional[MessageType] = None,
        clear: bool = False
    ) -> List[BaseMessage]:
        """Get messages for an agent, optionally filtering by type."""
        if agent_name not in self._queues:
            return []

        messages = self._queues[agent_name]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        if clear:
            if message_type:
                # Only clear matching messages
                self._queues[agent_name] = [
                    m for m in self._queues[agent_name]
                    if m.message_type != message_type
                ]
            else:
                self._queues[agent_name] = []

        return messages

    def get_message_by_id(self, message_id: str) -> Optional[BaseMessage]:
        """Retrieve a specific message by ID."""
        for message in self._message_history:
            if message.metadata.message_id == message_id:
                return message
        return None

    def get_conversation_history(
        self,
        correlation_id: str
    ) -> List[BaseMessage]:
        """Get all messages with a specific correlation ID."""
        return [
            m for m in self._message_history
            if m.metadata.correlation_id == correlation_id
        ]

    def clear_queue(self, agent_name: str) -> None:
        """Clear all messages for an agent."""
        if agent_name in self._queues:
            self._queues[agent_name] = []


# Factory functions for creating messages
def create_task_delegation(
    sender: str,
    recipient: str,
    task_id: str,
    task_description: str,
    task_type: str,
    **kwargs
) -> TaskDelegationMessage:
    """Factory function to create a task delegation message."""
    return TaskDelegationMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        task_id=task_id,
        task_description=task_description,
        task_type=task_type,
        **kwargs
    )


def create_task_completion(
    sender: str,
    recipient: str,
    task_id: str,
    original_delegation_id: str,
    status: str = "completed",
    **kwargs
) -> TaskCompletionMessage:
    """Factory function to create a task completion message."""
    return TaskCompletionMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        task_id=task_id,
        original_delegation_id=original_delegation_id,
        status=status,
        **kwargs
    )


def create_clarification_request(
    sender: str,
    recipient: str,
    task_id: str,
    question: str,
    question_type: str,
    **kwargs
) -> ClarificationRequestMessage:
    """Factory function to create a clarification request."""
    return ClarificationRequestMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        task_id=task_id,
        question=question,
        question_type=question_type,
        **kwargs
    )


def create_handoff(
    sender: str,
    recipient: str,
    task_id: str,
    reason: str,
    handoff_type: str,
    **kwargs
) -> HandoffMessage:
    """Factory function to create a handoff message."""
    return HandoffMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        task_id=task_id,
        reason=reason,
        handoff_type=handoff_type,
        **kwargs
    )


def create_quality_gate_alert(
    sender: str,
    recipient: str,
    task_id: str,
    gate_name: str,
    gate_type: str,
    passed: bool,
    threshold: float,
    actual_value: float,
    **kwargs
) -> QualityGateAlertMessage:
    """Factory function to create a quality gate alert."""
    return QualityGateAlertMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        task_id=task_id,
        agent_name=sender,
        gate_name=gate_name,
        gate_type=gate_type,
        passed=passed,
        threshold=threshold,
        actual_value=actual_value,
        **kwargs
    )


def create_artifact_notification(
    sender: str,
    recipient: str,
    artifact_id: str,
    artifact_type: str,
    artifact_path: str,
    task_id: str,
    **kwargs
) -> ArtifactNotificationMessage:
    """Factory function to create an artifact notification."""
    return ArtifactNotificationMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        artifact_path=artifact_path,
        task_id=task_id,
        **kwargs
    )


def create_error_report(
    sender: str,
    recipient: str,
    task_id: str,
    error_type: str,
    error_code: str,
    error_message: str,
    **kwargs
) -> ErrorReportMessage:
    """Factory function to create an error report."""
    return ErrorReportMessage(
        sender_agent=sender,
        recipient_agent=recipient,
        task_id=task_id,
        error_type=error_type,
        error_code=error_code,
        error_message=error_message,
        **kwargs
    )


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


def reset_message_bus() -> None:
    """Reset the global message bus (useful for testing)."""
    global _message_bus
    _message_bus = MessageBus()
