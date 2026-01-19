"""Configuration system for Data Science Multi-Agent Framework.

This module provides configuration management with:
- Global framework settings
- Agent-specific overrides
- Quality gate thresholds
- Artifact storage configuration
- Notification settings
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import os
from pathlib import Path


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class NotificationChannel(str, Enum):
    """Notification channels."""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class RetryConfig:
    """Retry configuration for agents."""
    max_attempts: int = 3
    backoff: str = "exponential"  # exponential, linear, fixed
    base_delay_seconds: int = 10
    max_delay_seconds: int = 300
    retryable_errors: List[str] = field(default_factory=lambda: [
        "timeout",
        "resource_unavailable",
        "transient_failure",
        "rate_limited",
    ])
    non_retryable_errors: List[str] = field(default_factory=lambda: [
        "invalid_input",
        "data_not_found",
        "permission_denied",
        "schema_validation_error",
    ])


@dataclass
class QualityGateConfig:
    """Quality gate thresholds."""
    # Performance thresholds
    min_classification_auc: float = 0.7
    min_regression_r2: float = 0.5
    max_rmse_threshold: Optional[float] = None  # Context-dependent

    # Fairness thresholds
    max_fairness_disparity: float = 0.1
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "demographic_parity",
        "equalized_odds",
    ])

    # Data quality thresholds
    min_data_quality_score: float = 0.8
    max_missing_rate: float = 0.2
    max_duplicate_rate: float = 0.01

    # Model stability
    max_cv_variance: float = 0.1
    min_sample_size: int = 100


@dataclass
class NotificationConfig:
    """Notification configuration."""
    on_failure: bool = True
    on_completion: bool = True
    on_quality_gate_failure: bool = True
    on_clarification_needed: bool = True
    channels: List[NotificationChannel] = field(default_factory=lambda: [
        NotificationChannel.CONSOLE
    ])
    slack_webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)


@dataclass
class ArtifactConfig:
    """Artifact storage configuration."""
    storage_path: str = "./artifacts"
    structure: Dict[str, str] = field(default_factory=lambda: {
        "data": "data",
        "raw": "data/raw",
        "cleaned": "data/cleaned",
        "features": "data/features",
        "models": "models",
        "experiments": "models/experiments",
        "production": "models/production",
        "reports": "reports",
        "eda": "reports/eda",
        "evaluation": "reports/evaluation",
        "statistical": "reports/statistical",
        "visualizations": "visualizations",
        "pipelines": "pipelines",
    })
    max_artifact_size_mb: int = 500
    compression: bool = True
    versioning: bool = True


@dataclass
class AgentConfig:
    """Configuration for a specific agent."""
    timeout_minutes: int = 60
    max_retries: int = 3
    max_parallel_tasks: int = 1
    priority: int = 0
    enabled: bool = True

    # Agent-specific settings (override in subclasses)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataEngineerConfig(AgentConfig):
    """DataEngineer-specific configuration."""
    max_file_size_gb: float = 10.0
    supported_formats: List[str] = field(default_factory=lambda: [
        "csv", "parquet", "json", "jsonl", "xlsx", "sql"
    ])
    default_output_format: str = "parquet"
    enable_data_profiling: bool = True
    profiling_sample_size: int = 10000


@dataclass
class EDAConfig(AgentConfig):
    """EDA-specific configuration."""
    default_depth: str = "standard"  # quick, standard, comprehensive
    max_columns_for_full_analysis: int = 100
    correlation_threshold: float = 0.7
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    generate_visualizations: bool = True
    max_categories_for_detailed_analysis: int = 20


@dataclass
class FeatureEngineerConfig(AgentConfig):
    """FeatureEngineer-specific configuration."""
    max_features: int = 500
    selection_method: str = "mutual_information"
    handle_missing: str = "impute"  # impute, drop, flag
    encoding_cardinality_threshold: int = 10
    create_interactions: bool = True
    max_interaction_order: int = 2


@dataclass
class ModelerConfig(AgentConfig):
    """Modeler-specific configuration."""
    timeout_minutes: int = 180  # Override: longer for training
    max_parallel_experiments: int = 4
    default_cv_folds: int = 5
    default_optimization_trials: int = 50
    enable_early_stopping: bool = True
    baseline_models: List[str] = field(default_factory=lambda: [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ])
    track_experiments: bool = True
    experiment_tracking_backend: str = "mlflow"


@dataclass
class EvaluatorConfig(AgentConfig):
    """Evaluator-specific configuration."""
    fairness_audit_required: bool = True
    robustness_tests: List[str] = field(default_factory=lambda: [
        "temporal_stability",
        "feature_sensitivity",
        "missing_data_robustness",
    ])
    interpretability_method: str = "shap"
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


@dataclass
class VisualizerConfig(AgentConfig):
    """Visualizer-specific configuration."""
    default_style: str = "professional"  # minimal, professional, publication
    default_palette: str = "viridis"
    output_formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    interactive_enabled: bool = True
    max_points_for_scatter: int = 10000
    figure_dpi: int = 150


@dataclass
class StatisticianConfig(AgentConfig):
    """Statistician-specific configuration."""
    default_alpha: float = 0.05
    default_power: float = 0.8
    multiple_comparison_method: str = "benjamini_hochberg"
    effect_size_reporting: bool = True
    bayesian_methods_enabled: bool = False


@dataclass
class MLOpsConfig(AgentConfig):
    """MLOps-specific configuration."""
    deployment_approval_required: bool = True
    default_deployment_pattern: str = "realtime"
    monitoring_enabled: bool = True
    drift_detection_enabled: bool = True
    canary_deployment_percentage: float = 0.1
    rollback_threshold: float = 0.05  # Performance drop threshold
    container_registry: str = ""
    kubernetes_namespace: str = "ml-models"


@dataclass
class DataScienceConfig:
    """Main configuration for the Data Science Multi-Agent Framework."""

    # Framework metadata
    name: str = "DataScienceMultiAgent"
    version: str = "1.0.0"

    # Global defaults
    timeout_minutes: int = 60
    max_retries: int = 3
    logging_level: LogLevel = LogLevel.INFO

    # Sub-configurations
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    quality_gates: QualityGateConfig = field(default_factory=QualityGateConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Agent-specific configurations
    agents: Dict[str, AgentConfig] = field(default_factory=lambda: {
        "ds_orchestrator": AgentConfig(),
        "data_engineer": DataEngineerConfig(),
        "eda_agent": EDAConfig(),
        "feature_engineer": FeatureEngineerConfig(),
        "modeler": ModelerConfig(),
        "evaluator": EvaluatorConfig(),
        "visualizer": VisualizerConfig(),
        "statistician": StatisticianConfig(),
        "mlops": MLOpsConfig(),
    })

    def get_agent_config(self, role: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        return self.agents.get(role, AgentConfig())

    def ensure_artifact_directories(self) -> None:
        """Create artifact directory structure if it doesn't exist."""
        base_path = Path(self.artifacts.storage_path)
        for subdir in self.artifacts.structure.values():
            dir_path = base_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "DataScienceConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override from environment
        if os.environ.get("DS_ARTIFACT_PATH"):
            config.artifacts.storage_path = os.environ["DS_ARTIFACT_PATH"]

        if os.environ.get("DS_LOG_LEVEL"):
            config.logging_level = LogLevel(os.environ["DS_LOG_LEVEL"])

        if os.environ.get("DS_SLACK_WEBHOOK"):
            config.notifications.slack_webhook_url = os.environ["DS_SLACK_WEBHOOK"]
            if NotificationChannel.SLACK not in config.notifications.channels:
                config.notifications.channels.append(NotificationChannel.SLACK)

        # Quality gate overrides
        if os.environ.get("DS_MIN_AUC"):
            config.quality_gates.min_classification_auc = float(os.environ["DS_MIN_AUC"])

        if os.environ.get("DS_MAX_FAIRNESS_DISPARITY"):
            config.quality_gates.max_fairness_disparity = float(
                os.environ["DS_MAX_FAIRNESS_DISPARITY"]
            )

        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataScienceConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Update top-level settings
        if "name" in data:
            config.name = data["name"]
        if "version" in data:
            config.version = data["version"]
        if "timeout_minutes" in data:
            config.timeout_minutes = data["timeout_minutes"]
        if "max_retries" in data:
            config.max_retries = data["max_retries"]
        if "logging_level" in data:
            config.logging_level = LogLevel(data["logging_level"])

        # Update artifacts config
        if "artifacts" in data:
            for key, value in data["artifacts"].items():
                if hasattr(config.artifacts, key):
                    setattr(config.artifacts, key, value)

        # Update quality gates
        if "quality_gates" in data:
            for key, value in data["quality_gates"].items():
                if hasattr(config.quality_gates, key):
                    setattr(config.quality_gates, key, value)

        # Update agent configs
        if "agents" in data:
            for agent_role, agent_config in data["agents"].items():
                if agent_role in config.agents:
                    for key, value in agent_config.items():
                        if hasattr(config.agents[agent_role], key):
                            setattr(config.agents[agent_role], key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "timeout_minutes": self.timeout_minutes,
            "max_retries": self.max_retries,
            "logging_level": self.logging_level.value,
            "artifacts": {
                "storage_path": self.artifacts.storage_path,
                "structure": self.artifacts.structure,
                "max_artifact_size_mb": self.artifacts.max_artifact_size_mb,
                "compression": self.artifacts.compression,
                "versioning": self.artifacts.versioning,
            },
            "quality_gates": {
                "min_classification_auc": self.quality_gates.min_classification_auc,
                "min_regression_r2": self.quality_gates.min_regression_r2,
                "max_fairness_disparity": self.quality_gates.max_fairness_disparity,
                "min_data_quality_score": self.quality_gates.min_data_quality_score,
            },
            "agents": {
                role: {
                    "timeout_minutes": cfg.timeout_minutes,
                    "max_retries": cfg.max_retries,
                    "enabled": cfg.enabled,
                }
                for role, cfg in self.agents.items()
            },
        }


# Global configuration instance
_config: Optional[DataScienceConfig] = None


def get_ds_config() -> DataScienceConfig:
    """Get the global data science configuration."""
    global _config
    if _config is None:
        _config = DataScienceConfig.from_env()
    return _config


def set_ds_config(config: DataScienceConfig) -> None:
    """Set the global data science configuration."""
    global _config
    _config = config
