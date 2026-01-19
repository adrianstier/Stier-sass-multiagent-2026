"""Data Science workflow patterns.

This module provides pre-defined workflow patterns for common data science
tasks following the framework specification:

1. ML Project Flow - Full end-to-end machine learning workflow
2. Statistical Analysis Flow - Hypothesis testing and experimental analysis
3. Reporting Flow - Data aggregation and visualization
4. A/B Test Flow - Experimental design and analysis
5. Model Iteration Flow - Rapid model improvement cycles
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from orchestrator.core.task_dsl import (
    WorkflowPlan, TaskSpec, ValidationMethod
)


class DataScienceWorkflowType(str, Enum):
    """Types of data science workflows."""
    ML_PROJECT = "ml_project"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    REPORTING = "reporting"
    AB_TEST = "ab_test"
    MODEL_ITERATION = "model_iteration"
    DATA_QUALITY = "data_quality"
    FEATURE_EXPLORATION = "feature_exploration"


# =============================================================================
# ML Project Workflow
# =============================================================================

def create_ml_project_workflow(
    include_deployment: bool = True,
    include_fairness_audit: bool = True,
    include_visualization: bool = True,
) -> WorkflowPlan:
    """Create a full end-to-end ML project workflow.

    This is the standard workflow for building and deploying ML models:

    1. DataEngineer: Ingest and clean raw data
    2. EDA: Explore data, identify patterns
    3. FeatureEngineer: Create and select features
    4. Modeler: Train and tune models
    5. Evaluator: Assess model quality (quality gate)
    6. Visualizer: Create results visualizations (optional)
    7. MLOps: Deploy to production (optional)

    Args:
        include_deployment: Whether to include MLOps deployment step
        include_fairness_audit: Whether Evaluator should include fairness audit
        include_visualization: Whether to include Visualizer step

    Returns:
        WorkflowPlan with all tasks configured
    """
    plan = WorkflowPlan()

    # Phase 1: Data Engineering
    plan.add_task(TaskSpec(
        task_type="data_ingestion_cleaning",
        assigned_role="data_engineer",
        description="Ingest raw data, profile quality, clean and transform into analysis-ready dataset",
        expected_artifacts=["cleaned_dataset", "data_dictionary", "profiling_report", "transformation_log"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Exploratory Data Analysis
    plan.add_task(TaskSpec(
        task_type="exploratory_analysis",
        assigned_role="eda_agent",
        description="Explore data distributions, relationships, anomalies. Generate hypotheses for modeling",
        dependencies=["data_ingestion_cleaning"],
        expected_artifacts=["eda_report", "hypothesis_list", "visualization_gallery"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 3: Feature Engineering
    plan.add_task(TaskSpec(
        task_type="feature_engineering",
        assigned_role="feature_engineer",
        description="Create features, handle encoding, prevent leakage, select final feature set",
        dependencies=["data_ingestion_cleaning", "exploratory_analysis"],
        expected_artifacts=["feature_set", "preprocessing_pipeline", "leakage_audit", "feature_documentation"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    # Phase 4: Model Training
    plan.add_task(TaskSpec(
        task_type="model_training",
        assigned_role="modeler",
        description="Train baseline and advanced models, tune hyperparameters, create ensemble if beneficial",
        dependencies=["feature_engineering"],
        expected_artifacts=["trained_model", "experiment_log", "model_card", "cv_results"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=70,
    ))

    # Phase 5: Model Evaluation (Quality Gate)
    eval_artifacts = ["evaluation_report", "deployment_recommendation"]
    if include_fairness_audit:
        eval_artifacts.extend(["fairness_audit", "bias_analysis"])

    gate_blocks = []
    if include_deployment:
        gate_blocks.append("model_deployment")
    if include_visualization:
        gate_blocks.append("results_visualization")

    plan.add_task(TaskSpec(
        task_type="model_evaluation",
        assigned_role="evaluator",
        description="Assess model performance, fairness, robustness. Provide deployment recommendation",
        dependencies=["model_training"],
        expected_artifacts=eval_artifacts,
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        gate_blocks=gate_blocks,
        priority=60,
    ))

    # Phase 6: Results Visualization (optional)
    if include_visualization:
        plan.add_task(TaskSpec(
            task_type="results_visualization",
            assigned_role="visualizer",
            description="Create publication-quality visualizations of model results and insights",
            dependencies=["model_evaluation"],
            expected_artifacts=["results_dashboard", "presentation_figures", "interactive_report"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=50,
        ))

    # Phase 7: Deployment (optional)
    if include_deployment:
        deploy_deps = ["model_evaluation"]
        if include_visualization:
            deploy_deps.append("results_visualization")

        plan.add_task(TaskSpec(
            task_type="model_deployment",
            assigned_role="mlops",
            description="Package model, deploy to production, configure monitoring and alerts",
            dependencies=deploy_deps,
            expected_artifacts=["deployment_config", "endpoint_url", "monitoring_dashboard", "runbook"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=40,
        ))

    plan.success_criteria = [
        "Model meets performance thresholds",
        "No critical fairness issues detected",
        "Evaluator approval obtained",
    ]
    if include_deployment:
        plan.success_criteria.append("Model successfully deployed")

    plan.acceptance_criteria = [
        "All required artifacts produced",
        "Quality gates passed",
        "Documentation complete",
    ]

    return plan


# =============================================================================
# Statistical Analysis Workflow
# =============================================================================

def create_statistical_analysis_workflow(
    analysis_type: str = "hypothesis_test",
    include_visualization: bool = True,
) -> WorkflowPlan:
    """Create a statistical analysis workflow.

    This workflow is for rigorous statistical analysis:

    1. DataEngineer: Prepare analysis dataset
    2. Statistician: Design experiment or analysis plan
    3. EDA: Verify assumptions, explore distributions
    4. Statistician: Conduct hypothesis tests
    5. Visualizer: Create publication figures (optional)

    Args:
        analysis_type: Type of analysis (hypothesis_test, experimental_design, causal_inference)
        include_visualization: Whether to include visualization step

    Returns:
        WorkflowPlan for statistical analysis
    """
    plan = WorkflowPlan()

    # Phase 1: Data Preparation
    plan.add_task(TaskSpec(
        task_type="analysis_data_prep",
        assigned_role="data_engineer",
        description="Prepare clean dataset for statistical analysis with proper formatting",
        expected_artifacts=["analysis_dataset", "data_dictionary", "sample_summary"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Analysis Design
    plan.add_task(TaskSpec(
        task_type="analysis_design",
        assigned_role="statistician",
        description=f"Design {analysis_type} plan: specify hypotheses, tests, power requirements",
        dependencies=["analysis_data_prep"],
        expected_artifacts=["analysis_plan", "power_analysis", "assumptions_checklist"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 3: Assumption Verification
    plan.add_task(TaskSpec(
        task_type="assumption_verification",
        assigned_role="eda_agent",
        description="Verify statistical assumptions, explore distributions, check for violations",
        dependencies=["analysis_design"],
        expected_artifacts=["assumption_report", "distribution_analysis", "diagnostic_plots"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    # Phase 4: Statistical Analysis (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="statistical_analysis",
        assigned_role="statistician",
        description="Conduct planned analyses, apply corrections, interpret results rigorously",
        dependencies=["assumption_verification"],
        expected_artifacts=["statistical_report", "effect_sizes", "confidence_intervals", "sensitivity_analysis"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        gate_blocks=["publication_figures"] if include_visualization else [],
        priority=70,
    ))

    # Phase 5: Publication Figures (optional)
    if include_visualization:
        plan.add_task(TaskSpec(
            task_type="publication_figures",
            assigned_role="visualizer",
            description="Create publication-quality figures for statistical results",
            dependencies=["statistical_analysis"],
            expected_artifacts=["figure_set", "figure_captions", "supplementary_figures"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=60,
        ))

    plan.success_criteria = [
        "Statistical assumptions verified or addressed",
        "Appropriate tests applied with corrections",
        "Effect sizes and confidence intervals reported",
        "Results interpretation is rigorous",
    ]

    plan.acceptance_criteria = [
        "Analysis plan followed",
        "All required statistics reported",
        "Limitations documented",
    ]

    return plan


# =============================================================================
# Reporting Workflow
# =============================================================================

def create_reporting_workflow(
    output_type: str = "dashboard",  # dashboard, report, presentation
) -> WorkflowPlan:
    """Create a reporting and visualization workflow.

    This workflow is for data aggregation and visualization:

    1. DataEngineer: Aggregate data sources
    2. EDA: Generate insights
    3. Visualizer: Create dashboard/report

    Args:
        output_type: Type of output (dashboard, report, presentation)

    Returns:
        WorkflowPlan for reporting
    """
    plan = WorkflowPlan()

    # Phase 1: Data Aggregation
    plan.add_task(TaskSpec(
        task_type="data_aggregation",
        assigned_role="data_engineer",
        description="Aggregate multiple data sources, compute metrics, prepare for visualization",
        expected_artifacts=["aggregated_data", "metrics_summary", "data_refresh_pipeline"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Insight Generation
    plan.add_task(TaskSpec(
        task_type="insight_generation",
        assigned_role="eda_agent",
        description="Analyze aggregated data, identify trends, anomalies, and key insights",
        dependencies=["data_aggregation"],
        expected_artifacts=["insights_summary", "trend_analysis", "key_findings"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 3: Dashboard/Report Creation
    output_artifacts = {
        "dashboard": ["interactive_dashboard", "dashboard_documentation", "refresh_instructions"],
        "report": ["written_report", "executive_summary", "appendix"],
        "presentation": ["slide_deck", "speaker_notes", "backup_slides"],
    }

    plan.add_task(TaskSpec(
        task_type=f"{output_type}_creation",
        assigned_role="visualizer",
        description=f"Create {output_type} with key insights, KPIs, and drill-down capabilities",
        dependencies=["insight_generation"],
        expected_artifacts=output_artifacts.get(output_type, ["output_artifact"]),
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    plan.success_criteria = [
        "All KPIs clearly visualized",
        "Insights actionable and clear",
        f"{output_type.title()} meets stakeholder requirements",
    ]

    return plan


# =============================================================================
# A/B Test Workflow
# =============================================================================

def create_ab_test_workflow(
    phase: str = "full",  # design, analysis, full
) -> WorkflowPlan:
    """Create an A/B testing workflow.

    This workflow handles experimental design and analysis:

    Design phase:
    1. Statistician: Design experiment (sample size, power, duration)
    2. DataEngineer: Set up data collection pipeline

    Analysis phase:
    1. DataEngineer: Collect and validate experiment data
    2. Statistician: Analyze results
    3. Visualizer: Create results visualization

    Args:
        phase: Which phase to include (design, analysis, full)

    Returns:
        WorkflowPlan for A/B testing
    """
    plan = WorkflowPlan()

    if phase in ("design", "full"):
        # Experimental Design Phase
        plan.add_task(TaskSpec(
            task_type="experiment_design",
            assigned_role="statistician",
            description="Design A/B test: hypotheses, sample size, power, duration, guardrails",
            expected_artifacts=["experiment_design", "power_analysis", "randomization_plan", "guardrail_metrics"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=100,
        ))

        plan.add_task(TaskSpec(
            task_type="data_collection_setup",
            assigned_role="data_engineer",
            description="Set up data collection pipeline for experiment metrics",
            dependencies=["experiment_design"],
            expected_artifacts=["collection_pipeline", "validation_rules", "monitoring_dashboard"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=90,
        ))

    if phase in ("analysis", "full"):
        analysis_deps = []
        if phase == "full":
            analysis_deps.append("data_collection_setup")

        # Data Validation
        plan.add_task(TaskSpec(
            task_type="experiment_data_validation",
            assigned_role="data_engineer",
            description="Validate experiment data: check randomization, sample ratio, data quality",
            dependencies=analysis_deps,
            expected_artifacts=["validated_data", "srm_check", "data_quality_report"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=80,
        ))

        # Statistical Analysis
        plan.add_task(TaskSpec(
            task_type="ab_test_analysis",
            assigned_role="statistician",
            description="Analyze A/B test: primary metrics, guardrails, heterogeneous effects",
            dependencies=["experiment_data_validation"],
            expected_artifacts=["analysis_report", "effect_estimates", "guardrail_results", "segment_analysis"],
            validation_method=ValidationMethod.GATE_APPROVAL,
            is_gate=True,
            gate_blocks=["results_communication"],
            priority=70,
        ))

        # Results Communication
        plan.add_task(TaskSpec(
            task_type="results_communication",
            assigned_role="visualizer",
            description="Create visualizations and summary for experiment results",
            dependencies=["ab_test_analysis"],
            expected_artifacts=["results_dashboard", "executive_summary", "detailed_report"],
            validation_method=ValidationMethod.ARTIFACT_EXISTS,
            priority=60,
        ))

    plan.success_criteria = [
        "Proper randomization verified",
        "No sample ratio mismatch",
        "Statistical significance properly assessed",
        "Practical significance evaluated",
    ]

    return plan


# =============================================================================
# Model Iteration Workflow
# =============================================================================

def create_model_iteration_workflow() -> WorkflowPlan:
    """Create a rapid model iteration workflow.

    This workflow is for improving an existing model:

    1. Evaluator: Diagnose current model issues
    2. EDA: Analyze error patterns
    3. FeatureEngineer: Create targeted features
    4. Modeler: Retrain with improvements
    5. Evaluator: Validate improvement (quality gate)

    Returns:
        WorkflowPlan for model iteration
    """
    plan = WorkflowPlan()

    # Phase 1: Model Diagnostics
    plan.add_task(TaskSpec(
        task_type="model_diagnostics",
        assigned_role="evaluator",
        description="Diagnose current model: identify failure modes, segments, systematic errors",
        expected_artifacts=["diagnostic_report", "error_taxonomy", "improvement_opportunities"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Error Analysis
    plan.add_task(TaskSpec(
        task_type="error_pattern_analysis",
        assigned_role="eda_agent",
        description="Deep dive into error patterns, find commonalities, identify data gaps",
        dependencies=["model_diagnostics"],
        expected_artifacts=["error_analysis", "pattern_report", "data_recommendations"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 3: Feature Improvement
    plan.add_task(TaskSpec(
        task_type="targeted_feature_engineering",
        assigned_role="feature_engineer",
        description="Create features targeting identified weaknesses and error patterns",
        dependencies=["error_pattern_analysis"],
        expected_artifacts=["new_features", "updated_pipeline", "feature_rationale"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    # Phase 4: Model Retraining
    plan.add_task(TaskSpec(
        task_type="model_retraining",
        assigned_role="modeler",
        description="Retrain model with new features, compare against baseline",
        dependencies=["targeted_feature_engineering"],
        expected_artifacts=["improved_model", "comparison_report", "ablation_study"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=70,
    ))

    # Phase 5: Improvement Validation (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="improvement_validation",
        assigned_role="evaluator",
        description="Validate improvement: compare to baseline, check for regressions",
        dependencies=["model_retraining"],
        expected_artifacts=["improvement_report", "regression_check", "deployment_decision"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        priority=60,
    ))

    plan.success_criteria = [
        "Improvement over baseline demonstrated",
        "No regressions on key segments",
        "Error patterns addressed",
    ]

    return plan


# =============================================================================
# Data Quality Workflow
# =============================================================================

def create_data_quality_workflow() -> WorkflowPlan:
    """Create a data quality assessment workflow.

    This workflow is for comprehensive data quality analysis:

    1. DataEngineer: Profile and validate data
    2. EDA: Identify quality issues and anomalies
    3. Statistician: Statistical quality assessment
    4. Visualizer: Quality dashboard

    Returns:
        WorkflowPlan for data quality assessment
    """
    plan = WorkflowPlan()

    # Phase 1: Data Profiling
    plan.add_task(TaskSpec(
        task_type="data_profiling",
        assigned_role="data_engineer",
        description="Comprehensive data profiling: schema, types, cardinality, nulls, patterns",
        expected_artifacts=["profile_report", "schema_documentation", "sample_data"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Quality Issue Detection
    plan.add_task(TaskSpec(
        task_type="quality_issue_detection",
        assigned_role="eda_agent",
        description="Detect quality issues: outliers, duplicates, inconsistencies, impossible values",
        dependencies=["data_profiling"],
        expected_artifacts=["issue_report", "anomaly_catalog", "consistency_check"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 3: Statistical Validation
    plan.add_task(TaskSpec(
        task_type="statistical_validation",
        assigned_role="statistician",
        description="Statistical quality checks: distribution tests, temporal stability, drift detection",
        dependencies=["data_profiling"],
        expected_artifacts=["statistical_report", "drift_analysis", "stability_metrics"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 4: Quality Dashboard
    plan.add_task(TaskSpec(
        task_type="quality_dashboard",
        assigned_role="visualizer",
        description="Create data quality monitoring dashboard with alerts and trends",
        dependencies=["quality_issue_detection", "statistical_validation"],
        expected_artifacts=["quality_dashboard", "alert_configuration", "trend_visualizations"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    plan.success_criteria = [
        "All quality dimensions assessed",
        "Issues documented with severity",
        "Monitoring dashboard operational",
    ]

    return plan


# =============================================================================
# Workflow Factory
# =============================================================================

def create_data_science_workflow(
    workflow_type: DataScienceWorkflowType,
    **kwargs,
) -> WorkflowPlan:
    """Factory function to create data science workflows.

    Args:
        workflow_type: Type of workflow to create
        **kwargs: Additional arguments for specific workflow types

    Returns:
        WorkflowPlan for the requested workflow type
    """
    workflow_creators = {
        DataScienceWorkflowType.ML_PROJECT: create_ml_project_workflow,
        DataScienceWorkflowType.STATISTICAL_ANALYSIS: create_statistical_analysis_workflow,
        DataScienceWorkflowType.REPORTING: create_reporting_workflow,
        DataScienceWorkflowType.AB_TEST: create_ab_test_workflow,
        DataScienceWorkflowType.MODEL_ITERATION: create_model_iteration_workflow,
        DataScienceWorkflowType.DATA_QUALITY: create_data_quality_workflow,
    }

    creator = workflow_creators.get(workflow_type)
    if creator is None:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    return creator(**kwargs)
