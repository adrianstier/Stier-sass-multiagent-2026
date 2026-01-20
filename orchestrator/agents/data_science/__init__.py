"""Data Science Multi-Agent Framework.

This module provides a comprehensive suite of specialized agents for end-to-end
data science workflows, including:

- DataScienceOrchestratorAgent: Central coordinator for task decomposition
- DataEngineerAgent: Data ingestion, cleaning, transformation
- EDAAgent: Exploratory data analysis and pattern discovery
- FeatureEngineerAgent: Feature creation, selection, encoding
- ModelerAgent: Model selection, training, hyperparameter tuning
- EvaluatorAgent: Performance assessment, fairness audit
- VisualizerAgent: Charts, dashboards, publication graphics
- StatisticianAgent: Hypothesis testing, experimental design
- MLOpsAgent: Deployment, monitoring, versioning
"""

from .schemas import (
    # Base schemas
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
)

from .agents import (
    DataScienceOrchestratorAgent,
    DataEngineerAgent,
    EDAAgent,
    FeatureEngineerAgent,
    ModelerAgent,
    EvaluatorAgent,
    VisualizerAgent,
    StatisticianAgent,
    MLOpsAgent,

    # Registry
    DATA_SCIENCE_AGENT_REGISTRY,
    get_data_science_agent_class,
    list_data_science_agents,
)

from .workflows import (
    create_ml_project_workflow,
    create_statistical_analysis_workflow,
    create_reporting_workflow,
    create_ab_test_workflow,
    create_model_iteration_workflow,
)

from .config import DataScienceConfig, AgentConfig

from .messages import (
    MessageBus,
    get_message_bus,
    reset_message_bus,
    create_task_delegation,
    create_task_completion,
    create_clarification_request,
    create_handoff,
    create_quality_gate_alert,
    create_artifact_notification,
    create_error_report,
)

from .artifacts import (
    ArtifactRegistry,
    Artifact,
    ArtifactType,
    ArtifactStatus,
    get_artifact_registry,
    reset_artifact_registry,
    create_artifact_from_file,
)

from .quality_gates import (
    QualityGate,
    QualityGateSet,
    QualityGateManager,
    get_quality_gate_manager,
    GracefulFailureHandler,
    ErrorHandler,
    create_default_error_handler,
    with_retry,
    DATA_QUALITY_GATES,
    CLASSIFICATION_PERFORMANCE_GATES,
    REGRESSION_PERFORMANCE_GATES,
    FAIRNESS_GATES,
    STABILITY_GATES,
)

from .prompt_templates import (
    PromptTemplate,
    get_template,
    render_template,
    list_templates,
    PROMPT_TEMPLATE_REGISTRY,
)

from .workflows import create_data_quality_workflow, create_data_science_workflow

__all__ = [
    # Communication schemas
    "TaskDelegationMessage",
    "TaskCompletionMessage",
    "ClarificationRequest",
    "HandoffMessage",
    "AgentFeedback",

    # Input schemas
    "DataEngineerInput",
    "EDAInput",
    "FeatureEngineerInput",
    "ModelerInput",
    "EvaluatorInput",
    "VisualizerInput",
    "StatisticianInput",
    "MLOpsInput",

    # Output schemas
    "DataEngineerOutput",
    "EDAOutput",
    "FeatureEngineerOutput",
    "ModelerOutput",
    "EvaluatorOutput",
    "VisualizerOutput",
    "StatisticianOutput",
    "MLOpsOutput",

    # Agents
    "DataScienceOrchestratorAgent",
    "DataEngineerAgent",
    "EDAAgent",
    "FeatureEngineerAgent",
    "ModelerAgent",
    "EvaluatorAgent",
    "VisualizerAgent",
    "StatisticianAgent",
    "MLOpsAgent",

    # Registry
    "DATA_SCIENCE_AGENT_REGISTRY",
    "get_data_science_agent_class",
    "list_data_science_agents",

    # Workflows
    "create_ml_project_workflow",
    "create_statistical_analysis_workflow",
    "create_reporting_workflow",
    "create_ab_test_workflow",
    "create_model_iteration_workflow",
    "create_data_quality_workflow",
    "create_data_science_workflow",

    # Config
    "DataScienceConfig",
    "AgentConfig",

    # Messages
    "MessageBus",
    "get_message_bus",
    "reset_message_bus",
    "create_task_delegation",
    "create_task_completion",
    "create_clarification_request",
    "create_handoff",
    "create_quality_gate_alert",
    "create_artifact_notification",
    "create_error_report",

    # Artifacts
    "ArtifactRegistry",
    "Artifact",
    "ArtifactType",
    "ArtifactStatus",
    "get_artifact_registry",
    "reset_artifact_registry",
    "create_artifact_from_file",

    # Quality Gates
    "QualityGate",
    "QualityGateSet",
    "QualityGateManager",
    "get_quality_gate_manager",
    "GracefulFailureHandler",
    "ErrorHandler",
    "create_default_error_handler",
    "with_retry",
    "DATA_QUALITY_GATES",
    "CLASSIFICATION_PERFORMANCE_GATES",
    "REGRESSION_PERFORMANCE_GATES",
    "FAIRNESS_GATES",
    "STABILITY_GATES",

    # Prompt Templates
    "PromptTemplate",
    "get_template",
    "render_template",
    "list_templates",
    "PROMPT_TEMPLATE_REGISTRY",
]
