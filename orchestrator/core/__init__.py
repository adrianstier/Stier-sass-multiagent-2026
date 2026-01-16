"""Core module for multi-agent orchestration system."""

from .models import Run, Task, Event, Artifact, RunStatus, TaskStatus
from .config import settings
from .database import get_db, init_db

# New orchestration enhancements
from .dlq import DLQManager, DeadLetterTask, get_dlq_manager
from .context_manager import ContextWindowManager, ContentSummarizer, get_context_manager
from .workflow_modifier import WorkflowModifier, ConditionalBranchEvaluator, get_workflow_modifier
from .channels import ChannelManager, AgentMessage, get_channel_manager
from .checkpoint import CheckpointManager, WorkflowCheckpoint, get_checkpoint_manager
from .cost_predictor import CostPredictor, CostEstimate, get_cost_predictor
from .supervision import SupervisionManager, SupervisedTaskExecutor, get_supervision_manager
from .semantic_validator import SemanticValidator, ValidationResult, get_semantic_validator
from .observability import MetricsCollector, DashboardDataProvider, get_metrics_collector
from .escalation import (
    EscalationManager,
    EscalationRequest,
    EscalationDecision,
    EscalationType,
    EscalationStatus,
    EscalationPriority,
    EscalationTier,
    NonBlockingEscalation,
    get_escalation_manager,
    request_gate_approval,
    request_budget_override,
    request_security_review,
)

__all__ = [
    # Models
    "Run",
    "Task",
    "Event",
    "Artifact",
    "RunStatus",
    "TaskStatus",
    "settings",
    "get_db",
    "init_db",
    # DLQ
    "DLQManager",
    "DeadLetterTask",
    "get_dlq_manager",
    # Context Management
    "ContextWindowManager",
    "ContentSummarizer",
    "get_context_manager",
    # Workflow Modification
    "WorkflowModifier",
    "ConditionalBranchEvaluator",
    "get_workflow_modifier",
    # Agent Channels
    "ChannelManager",
    "AgentMessage",
    "get_channel_manager",
    # Checkpoint/Resume
    "CheckpointManager",
    "WorkflowCheckpoint",
    "get_checkpoint_manager",
    # Cost Prediction
    "CostPredictor",
    "CostEstimate",
    "get_cost_predictor",
    # Supervision
    "SupervisionManager",
    "SupervisedTaskExecutor",
    "get_supervision_manager",
    # Semantic Validation
    "SemanticValidator",
    "ValidationResult",
    "get_semantic_validator",
    # Observability
    "MetricsCollector",
    "DashboardDataProvider",
    "get_metrics_collector",
    # Escalation
    "EscalationManager",
    "EscalationRequest",
    "EscalationDecision",
    "EscalationType",
    "EscalationStatus",
    "EscalationPriority",
    "EscalationTier",
    "NonBlockingEscalation",
    "get_escalation_manager",
    "request_gate_approval",
    "request_budget_override",
    "request_security_review",
]
