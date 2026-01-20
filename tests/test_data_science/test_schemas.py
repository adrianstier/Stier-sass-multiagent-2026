"""Tests for data science schemas."""

import pytest
from orchestrator.agents.data_science.schemas import (
    # Enums
    MessageType,
    TaskStatus,
    CompletionStatus,
    DeploymentRecommendation,
    ProblemType,
    AnalysisType,
    HandoffType,

    # Communication schemas
    TaskDelegationMessage,
    TaskCompletionMessage,
    ClarificationRequest,
    HandoffMessage,
    AgentFeedback,

    # Input schemas
    DataEngineerInput,
    EDAInput,
    FeatureEngineerInput,
    ModelerInput,
    EvaluatorInput,
    VisualizerInput,
    StatisticianInput,
    MLOpsInput,

    # Output schemas
    DataEngineerOutput,
    EDAOutput,
    FeatureEngineerOutput,
    ModelerOutput,
    EvaluatorOutput,
    VisualizerOutput,
    StatisticianOutput,
    MLOpsOutput,

    # Error schemas
    GracefulFailureReport,
    DataQualityAlert,
    QualityGateFailure,
)


class TestEnums:
    """Test enum definitions."""

    def test_message_type_values(self):
        assert MessageType.TASK_DELEGATION.value == "task_delegation"
        assert MessageType.TASK_COMPLETION.value == "task_completion"
        assert MessageType.CLARIFICATION_REQUEST.value == "clarification_request"

    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"

    def test_problem_type_values(self):
        assert ProblemType.CLASSIFICATION_BINARY.value == "classification_binary"
        assert ProblemType.CLASSIFICATION_MULTICLASS.value == "classification_multiclass"
        assert ProblemType.REGRESSION.value == "regression"
        assert ProblemType.CLUSTERING.value == "clustering"

    def test_handoff_type_values(self):
        assert HandoffType.SEQUENTIAL.value == "sequential"
        assert HandoffType.PARALLEL.value == "parallel"


class TestCommunicationSchemas:
    """Test communication schema creation and serialization."""

    def test_task_delegation_message_creation(self):
        msg = TaskDelegationMessage(
            from_agent="ds_orchestrator",
            to_agent="data_engineer",
        )

        assert msg.from_agent == "ds_orchestrator"
        assert msg.to_agent == "data_engineer"

    def test_task_delegation_to_dict(self):
        msg = TaskDelegationMessage(
            from_agent="ds_orchestrator",
            to_agent="data_engineer",
        )

        data = msg.to_dict()
        assert "from" in data or "from_agent" in data

    def test_task_completion_message(self):
        msg = TaskCompletionMessage(
            from_agent="data_engineer",
            to_agent="ds_orchestrator",
            status=CompletionStatus.SUCCESS,
        )

        assert msg.status == CompletionStatus.SUCCESS

    def test_clarification_request(self):
        req = ClarificationRequest(
            from_agent="modeler",
            to_agent="ds_orchestrator",
            question="Should I optimize for precision or recall?",
        )

        assert req.question == "Should I optimize for precision or recall?"

    def test_handoff_message(self):
        msg = HandoffMessage(
            from_agent="eda_agent",
            to_agent="feature_engineer",
            handoff_type=HandoffType.SEQUENTIAL,
            context_summary="EDA completed, found 3 high-correlation features",
            artifacts=["reports/eda/eda_report.html"],
        )

        assert msg.handoff_type == HandoffType.SEQUENTIAL
        assert len(msg.artifacts) == 1


class TestInputSchemas:
    """Test input schema classes exist and can be imported."""

    def test_data_engineer_input_exists(self):
        assert DataEngineerInput is not None

    def test_eda_input_exists(self):
        assert EDAInput is not None

    def test_modeler_input_exists(self):
        assert ModelerInput is not None

    def test_evaluator_input_exists(self):
        assert EvaluatorInput is not None


class TestOutputSchemas:
    """Test output schema classes exist and can be imported."""

    def test_data_engineer_output_exists(self):
        assert DataEngineerOutput is not None

    def test_eda_output_exists(self):
        assert EDAOutput is not None

    def test_modeler_output_exists(self):
        assert ModelerOutput is not None

    def test_evaluator_output_exists(self):
        assert EvaluatorOutput is not None


class TestErrorSchemas:
    """Test error handling schemas exist and can be imported."""

    def test_graceful_failure_report_exists(self):
        assert GracefulFailureReport is not None

    def test_quality_gate_failure_exists(self):
        assert QualityGateFailure is not None

    def test_data_quality_alert_exists(self):
        assert DataQualityAlert is not None
