"""Tests for inter-agent messaging system."""

import pytest
from orchestrator.agents.data_science.messages import (
    # Message types
    MessageType,
    MessagePriority,
    MessageMetadata,
    BaseMessage,
    TaskDelegationMessage,
    TaskCompletionMessage,
    ClarificationRequestMessage,
    ClarificationResponseMessage,
    HandoffMessage,
    FeedbackMessage,
    QualityGateAlertMessage,
    ArtifactNotificationMessage,
    StatusUpdateMessage,
    ErrorReportMessage,

    # Message bus
    MessageBus,
    get_message_bus,
    reset_message_bus,

    # Factory functions
    create_task_delegation,
    create_task_completion,
    create_clarification_request,
    create_handoff,
    create_quality_gate_alert,
    create_artifact_notification,
    create_error_report,
)


class TestMessageMetadata:
    """Test message metadata."""

    def test_metadata_auto_generates_id(self):
        metadata = MessageMetadata()
        assert metadata.message_id is not None
        assert len(metadata.message_id) > 0

    def test_metadata_auto_generates_timestamp(self):
        metadata = MessageMetadata()
        assert metadata.timestamp is not None

    def test_metadata_defaults(self):
        metadata = MessageMetadata()
        assert metadata.priority == MessagePriority.NORMAL
        assert metadata.retry_count == 0
        assert metadata.max_retries == 3


class TestTaskDelegationMessage:
    """Test task delegation messages."""

    def test_create_delegation_message(self):
        msg = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Clean the customer dataset",
            task_type="data_cleaning",
        )

        assert msg.message_type == MessageType.TASK_DELEGATION
        assert msg.sender_agent == "ds_orchestrator"
        assert msg.recipient_agent == "data_engineer"

    def test_delegation_with_context(self):
        msg = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="feature_engineer",
            task_id="task-002",
            task_description="Create features",
            task_type="feature_engineering",
            context={"target_variable": "churn", "problem_type": "classification"},
            predecessor_outputs=["data/cleaned/dataset.parquet"],
        )

        assert msg.context["target_variable"] == "churn"
        assert len(msg.predecessor_outputs) == 1

    def test_delegation_to_dict(self):
        msg = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Test",
            task_type="test",
        )

        data = msg.to_dict()
        assert data["message_type"] == "task_delegation"
        assert data["task_id"] == "task-001"
        assert "metadata" in data


class TestTaskCompletionMessage:
    """Test task completion messages."""

    def test_create_completion_message(self):
        msg = TaskCompletionMessage(
            sender_agent="data_engineer",
            recipient_agent="ds_orchestrator",
            task_id="task-001",
            original_delegation_id="msg-123",
            status="completed",
            outputs={"cleaned_data": "data/cleaned/dataset.parquet"},
        )

        assert msg.message_type == MessageType.TASK_COMPLETION
        assert msg.status == "completed"

    def test_completion_with_metrics(self):
        msg = TaskCompletionMessage(
            sender_agent="modeler",
            recipient_agent="ds_orchestrator",
            task_id="task-003",
            original_delegation_id="msg-456",
            status="completed",
            artifacts_produced=["models/model.pkl"],
            metrics={"auc": 0.85, "f1": 0.78},
            quality_score=0.85,
        )

        assert msg.metrics["auc"] == 0.85
        assert msg.quality_score == 0.85


class TestClarificationMessages:
    """Test clarification request/response messages."""

    def test_clarification_request(self):
        msg = ClarificationRequestMessage(
            sender_agent="modeler",
            recipient_agent="ds_orchestrator",
            task_id="task-003",
            question="Should I optimize for precision or recall?",
            question_type="methodology",
            options=["precision", "recall", "balanced"],
            blocking=True,
        )

        assert msg.message_type == MessageType.CLARIFICATION_REQUEST
        assert len(msg.options) == 3
        assert msg.blocking == True

    def test_clarification_response(self):
        msg = ClarificationResponseMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="modeler",
            original_request_id="msg-789",
            task_id="task-003",
            response="Optimize for recall - we want to minimize false negatives",
            selected_option="recall",
        )

        assert msg.message_type == MessageType.CLARIFICATION_RESPONSE
        assert msg.selected_option == "recall"


class TestHandoffMessage:
    """Test handoff messages."""

    def test_sequential_handoff(self):
        msg = HandoffMessage(
            sender_agent="eda_agent",
            recipient_agent="feature_engineer",
            task_id="task-002",
            reason="EDA complete, ready for feature engineering",
            handoff_type="sequential",
            artifacts=["reports/eda/report.html"],
            context_summary="Found 3 high-correlation features",
        )

        assert msg.message_type == MessageType.HANDOFF
        assert msg.handoff_type == "sequential"

    def test_escalation_handoff(self):
        msg = HandoffMessage(
            sender_agent="modeler",
            recipient_agent="ds_orchestrator",
            task_id="task-003",
            reason="Model performance below threshold",
            handoff_type="escalation",
            known_issues=["AUC only 0.65", "Possible data leakage"],
        )

        assert msg.handoff_type == "escalation"
        assert len(msg.known_issues) == 2


class TestQualityGateAlertMessage:
    """Test quality gate alert messages."""

    def test_quality_gate_failure_alert(self):
        msg = QualityGateAlertMessage(
            sender_agent="evaluator",
            recipient_agent="ds_orchestrator",
            task_id="task-004",
            agent_name="evaluator",
            gate_name="min_auc",
            gate_type="performance",
            passed=False,
            threshold=0.7,
            actual_value=0.65,
            impact="high",
            recommendations=["Collect more data", "Try different features"],
        )

        assert msg.message_type == MessageType.QUALITY_GATE_ALERT
        assert msg.passed == False
        assert msg.actual_value < msg.threshold


class TestMessageBus:
    """Test message bus functionality."""

    def setup_method(self):
        """Reset message bus before each test."""
        reset_message_bus()

    def test_subscribe_and_publish(self):
        bus = get_message_bus()
        bus.subscribe("data_engineer")

        msg = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Test",
            task_type="test",
        )

        bus.publish(msg)

        messages = bus.get_messages("data_engineer")
        assert len(messages) == 1
        assert messages[0].task_id == "task-001"

    def test_get_messages_with_type_filter(self):
        bus = get_message_bus()
        bus.subscribe("ds_orchestrator")

        # Publish different message types
        bus.publish(TaskCompletionMessage(
            sender_agent="data_engineer",
            recipient_agent="ds_orchestrator",
            task_id="task-001",
            original_delegation_id="msg-1",
            status="completed",
        ))

        bus.publish(ClarificationRequestMessage(
            sender_agent="modeler",
            recipient_agent="ds_orchestrator",
            task_id="task-002",
            question="Test question",
            question_type="methodology",
        ))

        # Filter by type
        completions = bus.get_messages(
            "ds_orchestrator",
            message_type=MessageType.TASK_COMPLETION
        )
        assert len(completions) == 1

        clarifications = bus.get_messages(
            "ds_orchestrator",
            message_type=MessageType.CLARIFICATION_REQUEST
        )
        assert len(clarifications) == 1

    def test_clear_queue(self):
        bus = get_message_bus()
        bus.subscribe("data_engineer")

        bus.publish(TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Test",
            task_type="test",
        ))

        bus.clear_queue("data_engineer")
        messages = bus.get_messages("data_engineer")
        assert len(messages) == 0

    def test_message_handler(self):
        bus = get_message_bus()
        received = []

        def handler(msg):
            received.append(msg)

        bus.subscribe("data_engineer", handler)

        msg = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Test",
            task_type="test",
        )

        bus.publish(msg)
        assert len(received) == 1

    def test_get_message_by_id(self):
        bus = get_message_bus()
        bus.subscribe("data_engineer")

        msg = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Test",
            task_type="test",
        )

        bus.publish(msg)
        retrieved = bus.get_message_by_id(msg.metadata.message_id)

        assert retrieved is not None
        assert retrieved.task_id == "task-001"

    def test_conversation_history(self):
        bus = get_message_bus()
        correlation_id = "conv-123"

        # Create messages with same correlation ID
        msg1 = TaskDelegationMessage(
            sender_agent="ds_orchestrator",
            recipient_agent="data_engineer",
            task_id="task-001",
            task_description="Test 1",
            task_type="test",
        )
        msg1.metadata.correlation_id = correlation_id

        msg2 = TaskCompletionMessage(
            sender_agent="data_engineer",
            recipient_agent="ds_orchestrator",
            task_id="task-001",
            original_delegation_id=msg1.metadata.message_id,
            status="completed",
        )
        msg2.metadata.correlation_id = correlation_id

        bus.publish(msg1)
        bus.publish(msg2)

        history = bus.get_conversation_history(correlation_id)
        assert len(history) == 2


class TestFactoryFunctions:
    """Test message factory functions."""

    def test_create_task_delegation(self):
        msg = create_task_delegation(
            sender="ds_orchestrator",
            recipient="data_engineer",
            task_id="task-001",
            task_description="Clean data",
            task_type="data_cleaning",
        )

        assert isinstance(msg, TaskDelegationMessage)
        assert msg.task_id == "task-001"

    def test_create_task_completion(self):
        msg = create_task_completion(
            sender="data_engineer",
            recipient="ds_orchestrator",
            task_id="task-001",
            original_delegation_id="msg-123",
            status="completed",
            outputs={"data": "path/to/data.parquet"},
        )

        assert isinstance(msg, TaskCompletionMessage)
        assert msg.outputs["data"] == "path/to/data.parquet"

    def test_create_quality_gate_alert(self):
        msg = create_quality_gate_alert(
            sender="evaluator",
            recipient="ds_orchestrator",
            task_id="task-004",
            gate_name="min_auc",
            gate_type="performance",
            passed=False,
            threshold=0.7,
            actual_value=0.65,
        )

        assert isinstance(msg, QualityGateAlertMessage)
        assert msg.passed == False
