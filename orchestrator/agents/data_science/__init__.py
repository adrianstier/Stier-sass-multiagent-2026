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

    # Config
    "DataScienceConfig",
    "AgentConfig",
]
