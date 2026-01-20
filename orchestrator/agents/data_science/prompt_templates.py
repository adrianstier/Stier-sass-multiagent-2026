"""Prompt Templates for Data Science Multi-Agent Framework.

This module provides prompt templates for:
- Task delegation between agents
- Clarification requests and responses
- Handoff communications
- Quality gate reports
- Error escalations
- Progress updates
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from string import Template


# Base template class
@dataclass
class PromptTemplate:
    """Base class for prompt templates."""
    name: str
    description: str
    template: str
    required_vars: List[str]
    optional_vars: List[str] = None

    def __post_init__(self):
        if self.optional_vars is None:
            self.optional_vars = []

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        # Check required variables
        missing = [v for v in self.required_vars if v not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Set defaults for optional variables
        for var in self.optional_vars:
            if var not in kwargs:
                kwargs[var] = ""

        return Template(self.template).safe_substitute(**kwargs)


# =============================================================================
# Task Delegation Templates
# =============================================================================

TASK_DELEGATION_TEMPLATE = PromptTemplate(
    name="task_delegation",
    description="Template for delegating tasks between agents",
    required_vars=["task_id", "task_description", "from_agent", "to_agent", "task_type"],
    optional_vars=["context", "requirements", "constraints", "deadline"],
    template="""
## Task Delegation

**Task ID:** $task_id
**From:** $from_agent
**To:** $to_agent
**Type:** $task_type

### Description
$task_description

### Context
$context

### Requirements
$requirements

### Constraints
$constraints

### Deadline
$deadline

---
Please proceed with this task and report back upon completion or if clarification is needed.
"""
)

DATA_INGESTION_TASK = PromptTemplate(
    name="data_ingestion_task",
    description="Template for data ingestion tasks",
    required_vars=["data_source", "expected_format", "target_location"],
    optional_vars=["validation_rules", "schema_requirements", "sampling_strategy"],
    template="""
## Data Ingestion Task

### Data Source
$data_source

### Expected Format
$expected_format

### Target Location
$target_location

### Validation Rules
$validation_rules

### Schema Requirements
$schema_requirements

### Sampling Strategy
$sampling_strategy

### Deliverables
1. Loaded data in specified format
2. Data quality report
3. Schema documentation
4. Any issues or anomalies detected
"""
)

EDA_TASK = PromptTemplate(
    name="eda_task",
    description="Template for exploratory data analysis tasks",
    required_vars=["dataset_path", "analysis_depth", "target_variable"],
    optional_vars=["specific_questions", "focus_areas", "time_constraints"],
    template="""
## Exploratory Data Analysis Task

### Dataset
$dataset_path

### Analysis Depth
$analysis_depth

### Target Variable
$target_variable

### Specific Questions
$specific_questions

### Focus Areas
$focus_areas

### Time Constraints
$time_constraints

### Expected Outputs
1. Summary statistics for all variables
2. Distribution analysis
3. Correlation analysis
4. Missing value assessment
5. Outlier detection
6. Key insights and recommendations
7. Visualizations supporting findings
"""
)

FEATURE_ENGINEERING_TASK = PromptTemplate(
    name="feature_engineering_task",
    description="Template for feature engineering tasks",
    required_vars=["input_data", "target_variable", "problem_type"],
    optional_vars=["eda_insights", "domain_knowledge", "feature_constraints"],
    template="""
## Feature Engineering Task

### Input Data
$input_data

### Target Variable
$target_variable

### Problem Type
$problem_type

### EDA Insights
$eda_insights

### Domain Knowledge
$domain_knowledge

### Feature Constraints
$feature_constraints

### Expected Outputs
1. Engineered feature set
2. Feature documentation (names, descriptions, transformations)
3. Feature importance ranking
4. Preprocessing pipeline
5. Leakage assessment
"""
)

MODEL_TRAINING_TASK = PromptTemplate(
    name="model_training_task",
    description="Template for model training tasks",
    required_vars=["features_path", "target_variable", "problem_type", "evaluation_metric"],
    optional_vars=["baseline_models", "hyperparameter_budget", "cross_validation_strategy"],
    template="""
## Model Training Task

### Features
$features_path

### Target Variable
$target_variable

### Problem Type
$problem_type

### Primary Evaluation Metric
$evaluation_metric

### Baseline Models to Try
$baseline_models

### Hyperparameter Budget
$hyperparameter_budget

### Cross-Validation Strategy
$cross_validation_strategy

### Expected Outputs
1. Trained model(s)
2. Performance metrics on validation set
3. Hyperparameter search results
4. Learning curves
5. Feature importance from model
6. Model artifacts for deployment
"""
)

EVALUATION_TASK = PromptTemplate(
    name="evaluation_task",
    description="Template for model evaluation tasks",
    required_vars=["model_path", "test_data", "evaluation_metrics"],
    optional_vars=["fairness_groups", "robustness_tests", "confidence_level"],
    template="""
## Model Evaluation Task

### Model
$model_path

### Test Data
$test_data

### Evaluation Metrics
$evaluation_metrics

### Fairness Assessment Groups
$fairness_groups

### Robustness Tests
$robustness_tests

### Confidence Level
$confidence_level

### Expected Outputs
1. Performance metrics with confidence intervals
2. Confusion matrix / residual analysis
3. Error analysis by segment
4. Fairness audit results
5. Robustness test results
6. Deployment recommendation
7. Identified risks and limitations
"""
)

VISUALIZATION_TASK = PromptTemplate(
    name="visualization_task",
    description="Template for visualization tasks",
    required_vars=["data_source", "visualization_purpose", "audience"],
    optional_vars=["style_guide", "specific_charts", "interactivity_requirements"],
    template="""
## Visualization Task

### Data Source
$data_source

### Purpose
$visualization_purpose

### Target Audience
$audience

### Style Guide
$style_guide

### Specific Charts Requested
$specific_charts

### Interactivity Requirements
$interactivity_requirements

### Expected Outputs
1. Requested visualizations
2. Supporting documentation
3. Interactive versions if applicable
4. Export in requested formats
"""
)

STATISTICAL_ANALYSIS_TASK = PromptTemplate(
    name="statistical_analysis_task",
    description="Template for statistical analysis tasks",
    required_vars=["research_question", "data_path", "analysis_type"],
    optional_vars=["alpha_level", "power_requirements", "multiple_comparison_correction"],
    template="""
## Statistical Analysis Task

### Research Question
$research_question

### Data
$data_path

### Analysis Type
$analysis_type

### Significance Level (α)
$alpha_level

### Power Requirements
$power_requirements

### Multiple Comparison Correction
$multiple_comparison_correction

### Expected Outputs
1. Statistical test results
2. Effect sizes with confidence intervals
3. Assumption checks
4. Visualizations of results
5. Plain-language interpretation
6. Limitations and caveats
"""
)

DEPLOYMENT_TASK = PromptTemplate(
    name="deployment_task",
    description="Template for model deployment tasks",
    required_vars=["model_path", "deployment_target", "serving_pattern"],
    optional_vars=["scaling_requirements", "monitoring_config", "rollback_criteria"],
    template="""
## Model Deployment Task

### Model
$model_path

### Deployment Target
$deployment_target

### Serving Pattern
$serving_pattern

### Scaling Requirements
$scaling_requirements

### Monitoring Configuration
$monitoring_config

### Rollback Criteria
$rollback_criteria

### Expected Outputs
1. Deployed model endpoint
2. Deployment configuration
3. Monitoring dashboard
4. Runbook documentation
5. Rollback procedure verification
"""
)


# =============================================================================
# Clarification Templates
# =============================================================================

CLARIFICATION_REQUEST_TEMPLATE = PromptTemplate(
    name="clarification_request",
    description="Template for requesting clarification",
    required_vars=["from_agent", "question", "context"],
    optional_vars=["options", "default_recommendation", "blocking"],
    template="""
## Clarification Request

**From:** $from_agent
**Blocking:** $blocking

### Question
$question

### Context
$context

### Options (if applicable)
$options

### Recommended Default
$default_recommendation

---
Please provide clarification to proceed with the task.
"""
)

CLARIFICATION_RESPONSE_TEMPLATE = PromptTemplate(
    name="clarification_response",
    description="Template for responding to clarification requests",
    required_vars=["to_agent", "response", "original_question"],
    optional_vars=["additional_context", "revised_requirements"],
    template="""
## Clarification Response

**To:** $to_agent

### Original Question
$original_question

### Response
$response

### Additional Context
$additional_context

### Revised Requirements
$revised_requirements

---
Please proceed with the task using this clarification.
"""
)


# =============================================================================
# Handoff Templates
# =============================================================================

HANDOFF_TEMPLATE = PromptTemplate(
    name="handoff",
    description="Template for handing off work between agents",
    required_vars=["from_agent", "to_agent", "task_summary", "current_state"],
    optional_vars=["artifacts", "recommendations", "known_issues"],
    template="""
## Task Handoff

**From:** $from_agent
**To:** $to_agent

### Task Summary
$task_summary

### Current State
$current_state

### Artifacts Produced
$artifacts

### Recommendations for Next Steps
$recommendations

### Known Issues
$known_issues

---
Please continue from this point. All relevant artifacts are available in the shared registry.
"""
)

SEQUENTIAL_HANDOFF_TEMPLATE = PromptTemplate(
    name="sequential_handoff",
    description="Template for sequential pipeline handoffs",
    required_vars=["previous_agent", "current_agent", "next_agent", "completed_work", "next_task"],
    optional_vars=["pipeline_stage", "quality_metrics"],
    template="""
## Sequential Pipeline Handoff

**Pipeline Stage:** $pipeline_stage
**Previous:** $previous_agent → **Current:** $current_agent → **Next:** $next_agent

### Completed Work (from $previous_agent)
$completed_work

### Your Task
$next_task

### Quality Metrics from Previous Stage
$quality_metrics

---
Complete your stage and hand off to $next_agent.
"""
)


# =============================================================================
# Quality Gate Templates
# =============================================================================

QUALITY_GATE_REPORT_TEMPLATE = PromptTemplate(
    name="quality_gate_report",
    description="Template for quality gate assessment reports",
    required_vars=["gate_set", "overall_status", "gate_results"],
    optional_vars=["recommendations", "blocking_failures", "next_steps"],
    template="""
## Quality Gate Report

### Gate Set: $gate_set
### Overall Status: $overall_status

### Individual Gate Results
$gate_results

### Blocking Failures
$blocking_failures

### Recommendations
$recommendations

### Next Steps
$next_steps
"""
)

QUALITY_GATE_FAILURE_TEMPLATE = PromptTemplate(
    name="quality_gate_failure",
    description="Template for quality gate failure notifications",
    required_vars=["gate_name", "threshold", "actual_value", "impact"],
    optional_vars=["remediation_options", "escalation_path"],
    template="""
## ⚠️ Quality Gate Failure

### Gate: $gate_name
### Threshold: $threshold
### Actual Value: $actual_value
### Impact: $impact

### Remediation Options
$remediation_options

### Escalation Path
$escalation_path

---
This failure requires attention before proceeding.
"""
)


# =============================================================================
# Error and Escalation Templates
# =============================================================================

ERROR_REPORT_TEMPLATE = PromptTemplate(
    name="error_report",
    description="Template for error reports",
    required_vars=["error_type", "error_message", "agent_name", "task_id"],
    optional_vars=["traceback", "partial_results", "recovery_suggestions"],
    template="""
## Error Report

**Agent:** $agent_name
**Task ID:** $task_id

### Error Type
$error_type

### Error Message
$error_message

### Stack Trace
```
$traceback
```

### Partial Results (if any)
$partial_results

### Recovery Suggestions
$recovery_suggestions
"""
)

ESCALATION_TEMPLATE = PromptTemplate(
    name="escalation",
    description="Template for escalating issues",
    required_vars=["issue_summary", "severity", "from_agent", "attempted_resolutions"],
    optional_vars=["impact_assessment", "recommended_action", "time_sensitivity"],
    template="""
## Escalation Notice

**Severity:** $severity
**From:** $from_agent
**Time Sensitivity:** $time_sensitivity

### Issue Summary
$issue_summary

### Impact Assessment
$impact_assessment

### Attempted Resolutions
$attempted_resolutions

### Recommended Action
$recommended_action

---
This issue requires intervention beyond the current agent's capabilities.
"""
)


# =============================================================================
# Progress and Status Templates
# =============================================================================

PROGRESS_UPDATE_TEMPLATE = PromptTemplate(
    name="progress_update",
    description="Template for progress updates",
    required_vars=["agent_name", "task_id", "current_step", "progress_percent"],
    optional_vars=["completed_steps", "remaining_steps", "blockers"],
    template="""
## Progress Update

**Agent:** $agent_name
**Task ID:** $task_id
**Progress:** $progress_percent%

### Current Step
$current_step

### Completed Steps
$completed_steps

### Remaining Steps
$remaining_steps

### Blockers
$blockers
"""
)

TASK_COMPLETION_TEMPLATE = PromptTemplate(
    name="task_completion",
    description="Template for task completion reports",
    required_vars=["agent_name", "task_id", "status", "summary"],
    optional_vars=["artifacts", "metrics", "recommendations", "execution_time"],
    template="""
## Task Completion Report

**Agent:** $agent_name
**Task ID:** $task_id
**Status:** $status
**Execution Time:** $execution_time

### Summary
$summary

### Artifacts Produced
$artifacts

### Key Metrics
$metrics

### Recommendations
$recommendations
"""
)


# =============================================================================
# Collaboration Templates
# =============================================================================

CONSULTATION_REQUEST_TEMPLATE = PromptTemplate(
    name="consultation_request",
    description="Template for requesting consultation from another agent",
    required_vars=["from_agent", "to_agent", "topic", "question"],
    optional_vars=["context", "urgency", "specific_expertise_needed"],
    template="""
## Consultation Request

**From:** $from_agent
**To:** $to_agent
**Urgency:** $urgency

### Topic
$topic

### Question
$question

### Context
$context

### Specific Expertise Needed
$specific_expertise_needed

---
Your input would be valuable for completing this task effectively.
"""
)

FEEDBACK_TEMPLATE = PromptTemplate(
    name="feedback",
    description="Template for providing feedback on work",
    required_vars=["from_agent", "to_agent", "work_reviewed", "feedback_type"],
    optional_vars=["positive_points", "issues", "required_changes", "rating"],
    template="""
## Work Feedback

**From:** $from_agent
**To:** $to_agent
**Feedback Type:** $feedback_type
**Rating:** $rating

### Work Reviewed
$work_reviewed

### Positive Points
$positive_points

### Issues Identified
$issues

### Required Changes
$required_changes
"""
)


# =============================================================================
# Template Registry
# =============================================================================

PROMPT_TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    # Task delegation
    "task_delegation": TASK_DELEGATION_TEMPLATE,
    "data_ingestion_task": DATA_INGESTION_TASK,
    "eda_task": EDA_TASK,
    "feature_engineering_task": FEATURE_ENGINEERING_TASK,
    "model_training_task": MODEL_TRAINING_TASK,
    "evaluation_task": EVALUATION_TASK,
    "visualization_task": VISUALIZATION_TASK,
    "statistical_analysis_task": STATISTICAL_ANALYSIS_TASK,
    "deployment_task": DEPLOYMENT_TASK,

    # Clarification
    "clarification_request": CLARIFICATION_REQUEST_TEMPLATE,
    "clarification_response": CLARIFICATION_RESPONSE_TEMPLATE,

    # Handoff
    "handoff": HANDOFF_TEMPLATE,
    "sequential_handoff": SEQUENTIAL_HANDOFF_TEMPLATE,

    # Quality gates
    "quality_gate_report": QUALITY_GATE_REPORT_TEMPLATE,
    "quality_gate_failure": QUALITY_GATE_FAILURE_TEMPLATE,

    # Errors and escalation
    "error_report": ERROR_REPORT_TEMPLATE,
    "escalation": ESCALATION_TEMPLATE,

    # Progress and status
    "progress_update": PROGRESS_UPDATE_TEMPLATE,
    "task_completion": TASK_COMPLETION_TEMPLATE,

    # Collaboration
    "consultation_request": CONSULTATION_REQUEST_TEMPLATE,
    "feedback": FEEDBACK_TEMPLATE,
}


def get_template(name: str) -> PromptTemplate:
    """Get a template by name."""
    if name not in PROMPT_TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template: {name}")
    return PROMPT_TEMPLATE_REGISTRY[name]


def render_template(name: str, **kwargs) -> str:
    """Render a template by name with provided variables."""
    template = get_template(name)
    return template.render(**kwargs)


def list_templates() -> List[str]:
    """List all available template names."""
    return list(PROMPT_TEMPLATE_REGISTRY.keys())


def get_template_info(name: str) -> Dict[str, Any]:
    """Get information about a template."""
    template = get_template(name)
    return {
        "name": template.name,
        "description": template.description,
        "required_vars": template.required_vars,
        "optional_vars": template.optional_vars,
    }
