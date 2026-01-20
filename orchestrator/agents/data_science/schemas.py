"""Schema definitions for Data Science Multi-Agent Framework.

This module defines all input/output schemas and inter-agent communication
protocols following the framework specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


# =============================================================================
# Enums for type safety
# =============================================================================

class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK_DELEGATION = "task_delegation"
    TASK_COMPLETION = "task_completion"
    CLARIFICATION_REQUEST = "clarification_request"
    PROGRESS_UPDATE = "progress_update"
    HANDOFF = "handoff"
    FEEDBACK = "feedback"
    ERROR_REPORT = "error_report"


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_CLARIFICATION = "needs_clarification"


class CompletionStatus(str, Enum):
    """Task completion status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class DeploymentRecommendation(str, Enum):
    """Model deployment recommendations."""
    DEPLOY = "deploy"
    CONDITIONAL_DEPLOY = "conditional_deploy"
    DO_NOT_DEPLOY = "do_not_deploy"


class ProblemType(str, Enum):
    """ML problem types."""
    CLASSIFICATION_BINARY = "classification_binary"
    CLASSIFICATION_MULTICLASS = "classification_multiclass"
    REGRESSION = "regression"
    RANKING = "ranking"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class AnalysisType(str, Enum):
    """Statistical analysis types."""
    HYPOTHESIS_TEST = "hypothesis_test"
    EXPERIMENTAL_DESIGN = "experimental_design"
    CAUSAL_INFERENCE = "causal_inference"
    POWER_ANALYSIS = "power_analysis"
    AB_TEST = "ab_test"


class DeploymentPattern(str, Enum):
    """Model deployment patterns."""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"


class EDADepth(str, Enum):
    """Depth of exploratory analysis."""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Priority(str, Enum):
    """Task priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HandoffType(str, Enum):
    """Types of handoffs between agents."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"


# =============================================================================
# Inter-Agent Communication Protocols
# =============================================================================

@dataclass
class TaskDelegationMessage:
    """Message for delegating tasks from orchestrator to agents.

    This is the standard format for the orchestrator to assign work to
    specialist agents.
    """
    message_type: MessageType = field(default=MessageType.TASK_DELEGATION)
    from_agent: str = "orchestrator"
    to_agent: str = ""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Payload
    objective: str = ""
    inputs: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: str = ""
    expected_outputs: List[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    deadline: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": {
                "objective": self.objective,
                "inputs": self.inputs,
                "constraints": self.constraints,
                "context": self.context,
                "expected_outputs": self.expected_outputs,
                "priority": self.priority.value,
                "deadline": self.deadline.isoformat() if self.deadline else None,
            }
        }


@dataclass
class TaskCompletionMessage:
    """Message for reporting task completion from agent to orchestrator."""
    message_type: MessageType = field(default=MessageType.TASK_COMPLETION)
    from_agent: str = ""
    to_agent: str = "orchestrator"
    task_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Payload
    status: CompletionStatus = CompletionStatus.SUCCESS
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": {
                "status": self.status.value,
                "outputs": self.outputs,
                "artifacts": self.artifacts,
                "notes": self.notes,
                "issues": self.issues,
                "recommendations": self.recommendations,
                "duration_seconds": self.duration_seconds,
            }
        }


@dataclass
class ClarificationRequest:
    """Message for requesting clarification from orchestrator."""
    message_type: MessageType = field(default=MessageType.CLARIFICATION_REQUEST)
    from_agent: str = ""
    to_agent: str = "orchestrator"
    task_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Payload
    question: str = ""
    options: List[str] = field(default_factory=list)
    impact: str = ""
    is_blocking: bool = True
    default_choice: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": {
                "question": self.question,
                "options": self.options,
                "impact": self.impact,
                "is_blocking": self.is_blocking,
                "default_choice": self.default_choice,
            }
        }


@dataclass
class HandoffMessage:
    """Message for handing off work between agents."""
    message_type: MessageType = field(default=MessageType.HANDOFF)
    from_agent: str = ""
    to_agent: str = ""
    task_id: str = ""
    handoff_type: HandoffType = HandoffType.SEQUENTIAL
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Payload
    context_summary: str = ""  # Alias for context
    context: str = ""
    artifacts: List[str] = field(default_factory=list)  # Simple list of artifact paths
    artifacts_provided: List[Dict[str, str]] = field(default_factory=list)
    important_notes: List[str] = field(default_factory=list)
    assumptions_made: List[str] = field(default_factory=list)
    questions_deferred: List[str] = field(default_factory=list)
    suggested_approach: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": {
                "context": self.context,
                "artifacts_provided": self.artifacts_provided,
                "important_notes": self.important_notes,
                "assumptions_made": self.assumptions_made,
                "questions_deferred": self.questions_deferred,
                "suggested_approach": self.suggested_approach,
            }
        }


@dataclass
class AgentFeedback:
    """Feedback message between agents."""
    message_type: MessageType = field(default=MessageType.FEEDBACK)
    from_agent: str = ""
    to_agent: str = ""
    task_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Payload
    regarding: str = ""  # artifact or decision being reviewed
    what_worked_well: List[str] = field(default_factory=list)
    concerns: List[Dict[str, str]] = field(default_factory=list)  # {concern, impact, suggestion}
    request: str = ""
    urgency: str = "informational"  # blocking, important, informational

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": {
                "regarding": self.regarding,
                "what_worked_well": self.what_worked_well,
                "concerns": self.concerns,
                "request": self.request,
                "urgency": self.urgency,
            }
        }


# =============================================================================
# Data Source Definitions
# =============================================================================

@dataclass
class DataSource:
    """Definition of a data source."""
    source_type: str  # csv, parquet, database, api, cloud_storage
    location: str
    credentials: Optional[str] = None
    query: Optional[str] = None
    schema: Optional[Dict[str, str]] = None


@dataclass
class CleaningRule:
    """Data cleaning rule specification."""
    column: str
    operation: str  # drop_nulls, fill_nulls, coerce_type, deduplicate, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DataEngineer Agent Schemas
# =============================================================================

@dataclass
class DataEngineerInput:
    """Input schema for DataEngineer agent."""
    objective: str
    data_sources: List[DataSource]
    cleaning_rules: Optional[List[CleaningRule]] = None
    output_format: str = "parquet"  # parquet, csv, database
    output_location: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataIssue:
    """Data quality issue found during processing."""
    severity: Severity
    column: Optional[str]
    description: str
    affected_rows: int = 0
    recommended_action: str = ""


@dataclass
class Transformation:
    """Record of a transformation applied to data."""
    operation: str
    columns: List[str]
    parameters: Dict[str, Any]
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataEngineerOutput:
    """Output schema for DataEngineer agent."""
    status: CompletionStatus
    output_path: str
    profile: Dict[str, Any]  # rows, columns, memory_mb, quality_score
    issues_found: List[DataIssue]
    issues_resolved: List[DataIssue]
    issues_flagged: List[DataIssue]
    transformations_applied: List[Transformation]
    data_dictionary: Dict[str, Any]
    lineage: List[str]  # transformation log


# =============================================================================
# EDA Agent Schemas
# =============================================================================

@dataclass
class EDAInput:
    """Input schema for EDA agent."""
    objective: str
    dataset_path: str
    target_variable: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    depth: EDADepth = EDADepth.STANDARD
    hypotheses_to_test: Optional[List[str]] = None


@dataclass
class UnivariateAnalysis:
    """Univariate analysis for a single column."""
    column: str
    dtype: str
    statistics: Dict[str, Any]
    distribution: Optional[str] = None
    notable_features: List[str] = field(default_factory=list)
    visualization_path: str = ""


@dataclass
class RelationshipAnalysis:
    """Analysis of relationship between variables."""
    variables: List[str]
    relationship_type: str  # correlation, association, etc.
    strength: float
    interpretation: str
    visualization_path: str = ""


@dataclass
class AnomalyReport:
    """Report of detected anomaly."""
    anomaly_type: str
    location: str  # column or row identifiers
    severity: Severity
    description: str
    recommended_action: str


@dataclass
class Hypothesis:
    """Generated hypothesis for investigation."""
    observation: str
    hypothesis: str
    suggested_test: str
    priority: Priority


@dataclass
class EDAOutput:
    """Output schema for EDA agent."""
    summary: Dict[str, Any]  # key_findings, data_quality_score, recommended_actions
    univariate: List[UnivariateAnalysis]
    relationships: List[RelationshipAnalysis]
    anomalies: List[AnomalyReport]
    hypotheses: List[Hypothesis]
    report_path: str
    visualizations: List[str]


# =============================================================================
# FeatureEngineer Agent Schemas
# =============================================================================

@dataclass
class FeatureEngineerInput:
    """Input schema for FeatureEngineer agent."""
    objective: str
    dataset_path: str
    target_variable: str
    problem_type: ProblemType
    constraints: Dict[str, Any] = field(default_factory=dict)
    # max_features, interpretability_required, real_time_inference
    guidance: Dict[str, Any] = field(default_factory=dict)
    # domain_knowledge, features_to_try, features_to_avoid
    eda_insights_path: Optional[str] = None


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    name: str
    source_columns: List[str]
    transformation: str
    dtype: str
    rationale: str
    leakage_risk: str = "none"  # none, low, medium, high


@dataclass
class DropReason:
    """Reason for dropping a feature."""
    feature: str
    reason: str
    importance_score: Optional[float] = None


@dataclass
class FeatureEngineerOutput:
    """Output schema for FeatureEngineer agent."""
    status: CompletionStatus
    feature_set: Dict[str, Any]  # version, num_features, feature_definitions
    pipeline_path: str
    selection_report: Dict[str, Any]  # method, original_count, final_count
    leakage_audit: Dict[str, Any]  # high_risk, mitigations_applied
    documentation_path: str
    feature_definitions: List[FeatureDefinition]


# =============================================================================
# Modeler Agent Schemas
# =============================================================================

@dataclass
class DataPaths:
    """Paths to model training/validation data."""
    train_path: str
    validation_path: Optional[str] = None
    target_column: str = ""
    feature_columns: Union[List[str], str] = "all"


@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration."""
    strategy: str = "random"  # grid, random, bayesian, none
    n_trials: int = 50
    cv_folds: int = 5
    timeout_seconds: Optional[int] = None


@dataclass
class ModelerInput:
    """Input schema for Modeler agent."""
    objective: str
    problem_type: ProblemType
    data: DataPaths
    constraints: Dict[str, Any] = field(default_factory=dict)
    # primary_metric, secondary_metrics, max_training_time_hours,
    # max_inference_latency_ms, interpretability
    guidance: Dict[str, Any] = field(default_factory=dict)
    # models_to_try, models_to_avoid, class_weights
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


@dataclass
class ExperimentLog:
    """Log of a single experiment."""
    experiment_id: str
    model_type: str
    hyperparameters: Dict[str, Any]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelerOutput:
    """Output schema for Modeler agent."""
    status: CompletionStatus
    best_model: Dict[str, Any]  # type, artifact_path, config_path, cv_score
    experiments: Dict[str, Any]  # total_trials, experiment_log_path
    ensemble: Optional[Dict[str, Any]] = None  # models_included, ensemble_score
    feature_importances: Dict[str, float] = field(default_factory=dict)
    model_card_path: str = ""
    recommendations: Dict[str, List[str]] = field(default_factory=dict)


# =============================================================================
# Evaluator Agent Schemas
# =============================================================================

@dataclass
class EvaluatorInput:
    """Input schema for Evaluator agent."""
    objective: str
    model_artifact_path: str
    data: Dict[str, Any]  # test_path, target_column, protected_attributes, temporal_column
    thresholds: Dict[str, Any]  # minimum_performance, maximum_fairness_disparity
    requirements: Dict[str, Any]  # interpretability_depth, fairness_audit, robustness_tests


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    value: float
    ci_lower: float
    ci_upper: float
    method: str = ""


@dataclass
class ErrorCategory:
    """Category of prediction errors."""
    category: str
    count: int
    percentage: float
    description: str
    examples: List[Dict[str, Any]]


@dataclass
class FairnessDisparity:
    """Fairness disparity between groups."""
    metric: str
    group_a: str
    group_b: str
    value_a: float
    value_b: float
    disparity: float
    significance: str


@dataclass
class LocalExplanation:
    """Local explanation for a single prediction."""
    sample_id: str
    prediction: float
    actual: Optional[float]
    top_features: Dict[str, float]
    explanation_path: str


@dataclass
class EvaluatorOutput:
    """Output schema for Evaluator agent."""
    recommendation: DeploymentRecommendation
    confidence: str  # high, medium, low
    performance: Dict[str, MetricResult]
    error_analysis: Dict[str, Any]  # total_errors, error_taxonomy, worst_segments
    fairness: Dict[str, Any]  # audit_performed, groups_assessed, disparities
    robustness: Dict[str, Any]  # temporal_stability, shift_sensitivity, edge_case_failures
    calibration: Dict[str, Any]  # ece, reliability_diagram_path
    interpretability: Dict[str, Any]  # global_importance, pdp_paths, local_explanations
    report_path: str
    deployment_notes: Dict[str, List[str]]


# =============================================================================
# Visualizer Agent Schemas
# =============================================================================

@dataclass
class VariableSpec:
    """Specification for a variable to visualize."""
    name: str
    role: str = "data"  # data, x, y, color, size, facet
    dtype: Optional[str] = None


@dataclass
class Annotation:
    """Annotation to add to visualization."""
    annotation_type: str  # text, line, region, arrow
    location: Dict[str, Any]
    content: str
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizerInput:
    """Input schema for Visualizer agent."""
    objective: str
    data: Dict[str, Any]  # source_path, variables, filters
    requirements: Dict[str, Any]  # chart_type, output_formats, dimensions, style, interactive
    context: Dict[str, Any]  # audience, medium, color_scheme
    annotations: Dict[str, Any] = field(default_factory=dict)  # title, subtitle, caption, highlights


@dataclass
class VisualizationOutput:
    """Output for a single visualization."""
    title: str
    insight: str
    chart_type: str
    files: List[Dict[str, Any]]  # format, path, size_kb
    accessibility: Dict[str, Any]  # colorblind_safe, contrast_ratio, alt_text


@dataclass
class VisualizerOutput:
    """Output schema for Visualizer agent."""
    status: CompletionStatus
    visualizations: List[VisualizationOutput]
    code_path: str
    design_decisions: List[Dict[str, str]]  # decision, rationale
    alternatives_considered: List[Dict[str, str]]  # chart_type, why_not


# =============================================================================
# Statistician Agent Schemas
# =============================================================================

@dataclass
class StatisticianInput:
    """Input schema for Statistician agent."""
    objective: str
    analysis_type: AnalysisType
    data: Dict[str, Any]  # source_path, sample_size, variables
    hypotheses: Dict[str, str] = field(default_factory=dict)  # null, alternative
    design: Dict[str, Any] = field(default_factory=dict)  # type, randomization, temporal
    requirements: Dict[str, Any] = field(default_factory=dict)  # alpha, power, effect_size_of_interest


@dataclass
class AssumptionCheck:
    """Result of checking a statistical assumption."""
    assumption: str
    test: str
    result: str  # pass, warning, fail
    details: str
    p_value: Optional[float] = None


@dataclass
class ExperimentalDesign:
    """Experimental design specification."""
    sample_size_per_arm: int
    minimum_detectable_effect: float
    recommended_duration: str
    randomization_procedure: str
    stopping_rules: str
    power_analysis: Dict[str, float]


@dataclass
class CausalAnalysis:
    """Causal analysis results."""
    dag_path: str
    identification_strategy: str
    assumptions: List[str]
    sensitivity_analysis: str
    effect_estimate: float
    confidence_interval: List[float]


@dataclass
class StatisticianOutput:
    """Output schema for Statistician agent."""
    status: CompletionStatus
    test_results: Dict[str, Any]  # test_name, statistic, p_value, effect_size, power
    assumptions: List[AssumptionCheck]
    interpretation: Dict[str, str]  # summary, practical_significance, limitations
    experimental_design: Optional[ExperimentalDesign] = None
    causal_analysis: Optional[CausalAnalysis] = None
    report_path: str = ""


# =============================================================================
# MLOps Agent Schemas
# =============================================================================

@dataclass
class ModelSpec:
    """Model specification for deployment."""
    artifact_path: str
    model_type: str  # sklearn, tensorflow, pytorch, xgboost, custom
    preprocessing_pipeline_path: Optional[str] = None
    version: str = ""


@dataclass
class ResourceSpec:
    """Resource specification for deployment."""
    cpu: str
    memory: str
    gpu: Optional[str] = None
    replicas: int = 1


@dataclass
class MLOpsInput:
    """Input schema for MLOps agent."""
    objective: str
    action: str  # package, deploy, monitor, rollback, update
    model: ModelSpec
    deployment: Dict[str, Any]  # pattern, target, replicas, resources
    requirements: Dict[str, Any]  # latency_sla_ms, throughput_rps, availability_sla
    monitoring: Dict[str, Any] = field(default_factory=dict)  # metrics, alert_channels, drift_detection


@dataclass
class PackagingResult:
    """Result of model packaging."""
    container_image: str
    artifact_registry: str
    size_mb: float
    dockerfile_path: str


@dataclass
class DeploymentResult:
    """Result of model deployment."""
    endpoint: str
    version: str
    replicas_running: int
    health_status: str  # healthy, degraded, unhealthy
    rollback_version: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    dashboard_url: str
    alerts_configured: List[str]
    baseline_metrics: Dict[str, float]


@dataclass
class MLOpsOutput:
    """Output schema for MLOps agent."""
    status: CompletionStatus
    packaging: Optional[PackagingResult] = None
    deployment: Optional[DeploymentResult] = None
    monitoring: Optional[MonitoringConfig] = None
    infrastructure: Dict[str, Any] = field(default_factory=dict)
    documentation: Dict[str, str] = field(default_factory=dict)
    next_steps: List[Dict[str, str]] = field(default_factory=list)


# =============================================================================
# Error Handling Schemas
# =============================================================================

@dataclass
class GracefulFailureReport:
    """Report for graceful failure handling."""
    task_id: str
    agent: str
    failure_type: str
    description: str
    root_cause: str
    attempts: List[Dict[str, str]]  # attempt description and result
    partial_results: Dict[str, Any]
    recovery_options: List[Dict[str, Any]]  # option, pros, cons
    recommended_option: str
    prevention_notes: str


@dataclass
class DataQualityAlert:
    """Alert for data quality issues."""
    severity: Severity
    dataset: str
    agent: str
    issue_description: str
    evidence: List[str]
    impact: str
    proposed_handling: List[Dict[str, str]]
    decision_needed: bool


@dataclass
class QualityGateFailure:
    """Report for quality gate failure."""
    model_id: str
    gate_type: str  # performance, fairness, robustness
    threshold_failures: List[Dict[str, Any]]  # metric, required, actual, gap
    critical_issues: List[Dict[str, Any]]  # issue, evidence, impact
    recommended_actions: List[Dict[str, str]]  # action, assigned_to
    estimated_remediation: str
