"""MCP Tools for Data Science Multi-Agent Framework.

This module provides MCP tool definitions and handlers for:
- Data science workflow orchestration
- Agent delegation and coordination
- Quality gate checks
- Artifact management
- Progress monitoring
"""

from typing import Any, Dict, List, Optional
import os
import json


# Tool Definitions for MCP
DATA_SCIENCE_TOOLS = [
    {
        "name": "ds_workflow_plan",
        "description": "[USES CLAUDE MAX] Get a data science workflow plan for Claude Code execution. Supports ML projects, statistical analysis, A/B testing, and more. No separate API credits needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the data science task (e.g., 'build a churn prediction model', 'analyze A/B test results')"
                },
                "workflow_type": {
                    "type": "string",
                    "enum": ["ml_project", "statistical_analysis", "reporting", "ab_test", "model_iteration", "data_quality"],
                    "description": "Type of data science workflow"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Path to the project directory"
                },
                "data_path": {
                    "type": "string",
                    "description": "Path to the data file or directory"
                },
                "target_variable": {
                    "type": "string",
                    "description": "Name of the target variable for ML tasks"
                },
                "problem_type": {
                    "type": "string",
                    "enum": ["classification", "regression", "clustering", "time_series", "nlp", "recommendation"],
                    "description": "Type of ML problem"
                }
            },
            "required": ["task"]
        }
    },
    {
        "name": "ds_analyze_data",
        "description": "[USES CLAUDE MAX] Perform exploratory data analysis on a dataset. Returns analysis plan with code templates for Claude Code to execute.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to the data file (CSV, Parquet, JSON)"
                },
                "analysis_depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "comprehensive"],
                    "default": "standard",
                    "description": "Depth of analysis to perform"
                },
                "target_variable": {
                    "type": "string",
                    "description": "Optional target variable for supervised learning context"
                }
            },
            "required": ["data_path"]
        }
    },
    {
        "name": "ds_quality_check",
        "description": "[USES CLAUDE MAX] Run quality gates on data or model metrics. Validates against configurable thresholds for data quality, model performance, fairness, and stability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gate_set": {
                    "type": "string",
                    "enum": ["data_quality", "classification", "regression", "fairness", "stability"],
                    "description": "Which quality gate set to run"
                },
                "metrics": {
                    "type": "object",
                    "description": "Dictionary of metric names to values (e.g., {'auc_roc': 0.85, 'f1_score': 0.78})"
                }
            },
            "required": ["gate_set", "metrics"]
        }
    },
    {
        "name": "ds_list_agents",
        "description": "[USES CLAUDE MAX] List available data science agents and their capabilities.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "ds_artifact_status",
        "description": "[USES CLAUDE MAX] Get status of artifacts in the data science workflow. Lists registered datasets, models, reports, and their lineage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "enum": ["all", "data", "model", "report", "visualization"],
                    "default": "all",
                    "description": "Filter artifacts by type"
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional task ID to filter artifacts"
                }
            }
        }
    }
]


# Data Science Agent Definitions
DS_AGENTS = {
    "ds_orchestrator": {
        "name": "Data Science Orchestrator",
        "system_prompt": """You are the Data Science Orchestrator Agent, the central coordinator for end-to-end data science workflows.

## Core Responsibilities
- Understand business problems and translate to technical approaches
- Decompose complex tasks into subtasks for specialist agents
- Coordinate workflow execution and manage dependencies
- Synthesize results and ensure quality

## Agent Coordination
You coordinate these specialist agents:
1. **DataEngineer** - Data ingestion, cleaning, transformation
2. **EDA** - Exploratory analysis and pattern discovery
3. **FeatureEngineer** - Feature creation and selection
4. **Modeler** - Model training and optimization
5. **Evaluator** - Model assessment and fairness audit
6. **Visualizer** - Charts and dashboards
7. **Statistician** - Hypothesis testing and experimental design
8. **MLOps** - Deployment and monitoring

## Decision Framework
- Start with understanding the business context
- Choose workflow based on task type (ML, statistics, reporting)
- Delegate to appropriate specialists in the right sequence
- Monitor quality gates at each stage
- Aggregate results into actionable insights""",
        "tools": ["Read", "Glob", "Grep", "Bash", "Write", "Edit", "Task"]
    },
    "data_engineer": {
        "name": "Data Engineer",
        "system_prompt": """You are the Data Engineer Agent, responsible for data ingestion, quality, and transformation.

## Core Responsibilities
- Load data from various sources (files, databases, APIs)
- Assess and improve data quality
- Transform data for downstream analysis
- Create reproducible data pipelines

## Data Quality Checklist
1. **Completeness** - Missing value patterns
2. **Uniqueness** - Duplicates and IDs
3. **Consistency** - Format standardization
4. **Validity** - Domain constraints
5. **Timeliness** - Data freshness

## Output Standards
- Provide data quality report with every dataset
- Document all transformations applied
- Flag any data quality concerns
- Use parquet format for large datasets""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "eda_agent": {
        "name": "EDA Analyst",
        "system_prompt": """You are the Exploratory Data Analysis Agent, discovering patterns and generating insights.

## Analysis Framework
1. **Univariate** - Distribution of each variable
2. **Bivariate** - Relationships between pairs
3. **Multivariate** - Complex interactions
4. **Target Analysis** - Target variable behavior

## Key Outputs
- Summary statistics (mean, median, std, quartiles)
- Distribution visualizations
- Correlation analysis
- Missing value patterns
- Outlier detection
- Key insights and hypotheses

## Report Structure
- Executive Summary (key findings)
- Data Overview (shape, types)
- Variable Analysis
- Relationships
- Recommendations for modeling""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "feature_engineer": {
        "name": "Feature Engineer",
        "system_prompt": """You are the Feature Engineer Agent, creating and selecting features for machine learning.

## Feature Engineering Toolkit
- **Encoding**: One-hot, target, ordinal
- **Scaling**: Standard, MinMax, robust
- **Interactions**: Polynomial, ratios
- **Time Features**: Lags, windows, seasonality
- **Text Features**: TF-IDF, embeddings

## Critical Rule: Prevent Leakage
- Never use future information
- Split before any target-dependent transformations
- Validate temporal consistency

## Selection Methods
- Correlation analysis
- Mutual information
- Feature importance from models
- Recursive feature elimination""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "modeler": {
        "name": "ML Modeler",
        "system_prompt": """You are the Modeler Agent, responsible for model training and optimization.

## Training Protocol
1. Establish baseline (simple model)
2. Try multiple algorithms
3. Optimize hyperparameters
4. Validate with cross-validation
5. Select best model

## Model Selection Matrix
| Problem | Start With | Advanced |
|---------|-----------|----------|
| Classification | Logistic, RF | XGBoost, LightGBM |
| Regression | Linear, RF | XGBoost, Neural Net |
| Time Series | ARIMA | Prophet, LSTM |

## Output Requirements
- Trained model artifacts
- Performance metrics
- Learning curves
- Feature importance
- Hyperparameter search results""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "evaluator": {
        "name": "Model Evaluator",
        "system_prompt": """You are the Evaluator Agent, assessing model performance and fairness.

## Evaluation Framework
1. **Performance Metrics** - Task-appropriate metrics
2. **Error Analysis** - Where does the model fail?
3. **Fairness Audit** - Bias across groups
4. **Robustness** - Stability under perturbation
5. **Interpretability** - SHAP, feature importance

## Deployment Recommendations
- **DEPLOY**: All gates passed
- **CONDITIONAL_DEPLOY**: Minor concerns, monitor closely
- **DO_NOT_DEPLOY**: Significant issues found

## Key Metrics
| Type | Metrics |
|------|---------|
| Classification | AUC, F1, Precision, Recall |
| Regression | R², RMSE, MAE, MAPE |
| Fairness | Demographic parity, Equalized odds |""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "visualizer": {
        "name": "Data Visualizer",
        "system_prompt": """You are the Visualizer Agent, creating clear and effective data visualizations.

## Chart Selection
| Data Type | Comparison | Distribution | Relationship |
|-----------|------------|--------------|--------------|
| Numeric | Bar | Histogram | Scatter |
| Time | Line | - | Line |
| Category | Bar | Bar | Heatmap |

## Design Principles
1. Title explains the insight
2. Axis labels with units
3. Colorblind-safe palettes
4. Minimal chart junk
5. Data-ink ratio optimization

## Output Formats
- PNG for reports (150+ DPI)
- SVG for scalable graphics
- Interactive HTML when appropriate""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "statistician": {
        "name": "Statistician",
        "system_prompt": """You are the Statistician Agent, providing rigorous statistical analysis.

## Analysis Types
1. **Hypothesis Testing** - t-tests, ANOVA, chi-square
2. **Effect Size** - Cohen's d, odds ratios
3. **Confidence Intervals** - Bootstrap when needed
4. **Power Analysis** - Sample size calculations
5. **A/B Testing** - Experimental design

## Reporting Standards
- Always report effect sizes with CIs
- State assumptions and check them
- Use multiple comparison corrections
- Provide plain-language interpretation

## Key Principles
- Statistical significance ≠ practical significance
- Report both p-value and effect size
- Be explicit about assumptions
- Acknowledge limitations""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    },
    "mlops": {
        "name": "MLOps Engineer",
        "system_prompt": """You are the MLOps Agent, ensuring reliable model deployment and monitoring.

## Deployment Pipeline
1. **Package** - Model + preprocessing + config
2. **Containerize** - Docker for reproducibility
3. **Deploy** - REST API or batch pipeline
4. **Monitor** - Metrics, drift, errors
5. **Maintain** - Versioning, rollback

## Monitoring Essentials
- Prediction latency (p50, p95, p99)
- Error rates
- Feature drift (PSI)
- Model performance decay

## Rollback Triggers
- Performance drop > 5%
- Error rate > 5%
- Significant drift detected""",
        "tools": ["Read", "Write", "Bash", "Glob"]
    }
}


async def execute_ds_tool(tool_name: str, args: dict) -> Any:
    """Execute a data science tool."""

    if tool_name == "ds_workflow_plan":
        return await get_ds_workflow_plan(args)

    elif tool_name == "ds_analyze_data":
        return await analyze_data(args)

    elif tool_name == "ds_quality_check":
        return await run_quality_check(args)

    elif tool_name == "ds_list_agents":
        return list_ds_agents()

    elif tool_name == "ds_artifact_status":
        return await get_artifact_status(args)

    else:
        raise ValueError(f"Unknown data science tool: {tool_name}")


async def get_ds_workflow_plan(args: dict) -> Dict[str, Any]:
    """Generate a data science workflow plan."""
    task = args["task"]
    workflow_type = args.get("workflow_type", "ml_project")
    working_dir = args.get("working_directory", os.getcwd())
    data_path = args.get("data_path", "")
    target_variable = args.get("target_variable", "")
    problem_type = args.get("problem_type", "classification")

    # Define workflow sequences based on type
    workflow_sequences = {
        "ml_project": [
            ["ds_orchestrator"],
            ["data_engineer"],
            ["eda_agent"],
            ["feature_engineer"],
            ["modeler"],
            ["evaluator"],
            ["visualizer"],
        ],
        "statistical_analysis": [
            ["ds_orchestrator"],
            ["data_engineer"],
            ["statistician"],
            ["visualizer"],
        ],
        "reporting": [
            ["ds_orchestrator"],
            ["eda_agent"],
            ["visualizer"],
        ],
        "ab_test": [
            ["ds_orchestrator"],
            ["data_engineer"],
            ["statistician"],
            ["evaluator"],
            ["visualizer"],
        ],
        "model_iteration": [
            ["feature_engineer"],
            ["modeler"],
            ["evaluator"],
        ],
        "data_quality": [
            ["data_engineer"],
            ["eda_agent"],
        ],
    }

    sequence = workflow_sequences.get(workflow_type, workflow_sequences["ml_project"])

    # Build workflow steps
    workflow_steps = []
    step_num = 1

    for phase in sequence:
        phase_steps = []
        for agent_name in phase:
            if agent_name not in DS_AGENTS:
                continue

            agent_config = DS_AGENTS[agent_name]

            # Build context-aware task prompt
            task_context = f"""## Task
{task}

## Working Directory
{working_dir}

## Data Path
{data_path if data_path else 'To be determined'}

## Target Variable
{target_variable if target_variable else 'To be determined'}

## Problem Type
{problem_type}

## Instructions
You are now acting as the {agent_config['name']}. Complete your assigned task using the available tools.

When working with data:
1. Read and understand the data structure first
2. Apply your specialized expertise
3. Document all findings and decisions
4. Save outputs to appropriate locations
5. Provide a summary of what was accomplished

Coordinate with other agents by:
- Reading artifacts from previous agents
- Writing clear documentation for downstream agents
- Flagging any concerns or blockers
"""

            phase_steps.append({
                "step": step_num,
                "agent": agent_name,
                "agent_name": agent_config["name"],
                "system_prompt": agent_config["system_prompt"],
                "task_prompt": task_context,
                "available_tools": agent_config["tools"],
            })
            step_num += 1

        if phase_steps:
            workflow_steps.append({
                "phase": len(workflow_steps) + 1,
                "parallel": len(phase_steps) > 1,
                "agents": phase_steps
            })

    return {
        "task": task,
        "workflow_type": workflow_type,
        "working_directory": working_dir,
        "data_path": data_path,
        "target_variable": target_variable,
        "problem_type": problem_type,
        "total_agents": sum(len(phase["agents"]) for phase in workflow_steps),
        "phases": len(workflow_steps),
        "workflow": workflow_steps,
        "execution_instructions": """
## How to Execute This Data Science Workflow

Claude Code should execute each phase in order. For each agent step:

1. **Adopt the agent's persona** by following its system_prompt
2. **Execute the task_prompt** using available tools
3. **Save artifacts** (data, models, reports) for downstream agents
4. **Document findings** clearly for the next agent

### Typical Flow
```
Orchestrator → DataEngineer → EDA → FeatureEngineer → Modeler → Evaluator → Visualizer
```

### Quality Gates
At each stage, validate:
- Data quality (completeness, consistency)
- Model performance (meets thresholds)
- Fairness (no bias issues)

### Artifact Locations
- Data: ./artifacts/data/
- Models: ./artifacts/models/
- Reports: ./artifacts/reports/
- Visualizations: ./artifacts/visualizations/

This workflow uses YOUR Claude Max subscription - no separate API calls needed!
""",
        "quality_gates": {
            "data_quality": {
                "missing_rate": "< 0.2",
                "duplicate_rate": "< 0.01"
            },
            "model_performance": {
                "auc_roc": "> 0.7 for classification",
                "r2": "> 0.5 for regression"
            },
            "fairness": {
                "demographic_parity_ratio": "> 0.8"
            }
        },
        "available_agents": list(DS_AGENTS.keys()),
    }


async def analyze_data(args: dict) -> Dict[str, Any]:
    """Perform exploratory data analysis."""
    data_path = args["data_path"]
    analysis_depth = args.get("analysis_depth", "standard")
    target_variable = args.get("target_variable")

    # This returns a structured plan for Claude Code to execute
    return {
        "data_path": data_path,
        "analysis_depth": analysis_depth,
        "target_variable": target_variable,
        "analysis_plan": {
            "steps": [
                {
                    "name": "Load Data",
                    "description": "Read the data file and determine its structure",
                    "code_template": f"""
import pandas as pd

# Load data
df = pd.read_csv('{data_path}')  # Adjust based on file format
print(f"Shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
print(df.dtypes)
"""
                },
                {
                    "name": "Summary Statistics",
                    "description": "Calculate basic statistics for all variables",
                    "code_template": """
print("\\n=== Summary Statistics ===")
print(df.describe(include='all'))
"""
                },
                {
                    "name": "Missing Values",
                    "description": "Analyze missing value patterns",
                    "code_template": """
print("\\n=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'count': missing, 'percent': missing_pct})[missing > 0])
"""
                },
                {
                    "name": "Distribution Analysis",
                    "description": "Examine distributions of key variables",
                    "code_template": """
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include='number').columns[:6]  # First 6 numeric
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    ax = axes[i // 3, i % 3]
    df[col].hist(ax=ax, bins=30)
    ax.set_title(col)
plt.tight_layout()
plt.savefig('distribution_analysis.png')
"""
                },
                {
                    "name": "Correlation Analysis",
                    "description": "Calculate correlations between numeric variables",
                    "code_template": """
print("\\n=== Correlation Matrix ===")
corr = df.select_dtypes(include='number').corr()
print(corr)

# High correlations
high_corr = corr.abs().unstack()
high_corr = high_corr[high_corr < 1.0]  # Remove self-correlations
print("\\nHigh correlations (>0.7):")
print(high_corr[high_corr > 0.7].sort_values(ascending=False).drop_duplicates())
"""
                }
            ],
            "depth_specific": {
                "quick": ["Load Data", "Summary Statistics", "Missing Values"],
                "standard": ["Load Data", "Summary Statistics", "Missing Values", "Distribution Analysis", "Correlation Analysis"],
                "comprehensive": "All steps plus outlier detection, target analysis, and automated insights"
            }
        },
        "execution_note": "Claude Code should execute these analysis steps using the Bash tool with Python"
    }


async def run_quality_check(args: dict) -> Dict[str, Any]:
    """Run quality gate checks."""
    from .quality_gates import get_quality_gate_manager

    gate_set = args["gate_set"]
    metrics = args["metrics"]

    manager = get_quality_gate_manager()
    results = manager.run_gates(gate_set, metrics, agent_name="mcp_tools")

    return results


def list_ds_agents() -> Dict[str, Any]:
    """List available data science agents."""
    agents = []
    for agent_id, config in DS_AGENTS.items():
        agents.append({
            "id": agent_id,
            "name": config["name"],
            "description": config["system_prompt"].split("\n")[0].replace("You are the ", "").replace(".", ""),
            "tools": config["tools"]
        })

    return {
        "total_agents": len(agents),
        "agents": agents,
        "workflow_types": [
            {"type": "ml_project", "description": "End-to-end machine learning project"},
            {"type": "statistical_analysis", "description": "Hypothesis testing and statistical inference"},
            {"type": "reporting", "description": "Data exploration and visualization"},
            {"type": "ab_test", "description": "A/B test design and analysis"},
            {"type": "model_iteration", "description": "Rapid model improvement cycle"},
            {"type": "data_quality", "description": "Data quality assessment"},
        ]
    }


async def get_artifact_status(args: dict) -> Dict[str, Any]:
    """Get artifact status from the registry."""
    from .artifacts import get_artifact_registry, ArtifactType

    artifact_type_filter = args.get("artifact_type", "all")
    task_id = args.get("task_id")

    registry = get_artifact_registry()
    stats = registry.get_statistics()

    # Get filtered artifacts
    if artifact_type_filter == "all":
        artifacts = list(registry._artifacts.values())
    elif artifact_type_filter == "data":
        artifacts = []
        for at in [ArtifactType.RAW_DATA, ArtifactType.CLEANED_DATA,
                   ArtifactType.FEATURE_SET, ArtifactType.TRAIN_SET, ArtifactType.TEST_SET]:
            artifacts.extend(registry.find_by_type(at))
    elif artifact_type_filter == "model":
        artifacts = registry.find_by_type(ArtifactType.MODEL)
    elif artifact_type_filter == "report":
        artifacts = []
        for at in [ArtifactType.EDA_REPORT, ArtifactType.EVALUATION_REPORT,
                   ArtifactType.STATISTICAL_REPORT]:
            artifacts.extend(registry.find_by_type(at))
    elif artifact_type_filter == "visualization":
        artifacts = []
        for at in [ArtifactType.PLOT, ArtifactType.CHART, ArtifactType.DASHBOARD]:
            artifacts.extend(registry.find_by_type(at))
    else:
        artifacts = []

    if task_id:
        artifacts = [a for a in artifacts if a.lineage.source_task == task_id]

    return {
        "filter": artifact_type_filter,
        "task_id": task_id,
        "total_artifacts": len(artifacts),
        "statistics": stats,
        "artifacts": [
            {
                "id": a.artifact_id,
                "type": a.artifact_type.value,
                "name": a.metadata.name,
                "status": a.status.value,
                "version": a.version_info.version,
                "agent": a.lineage.source_agent,
                "created_at": a.created_at,
            }
            for a in artifacts[:20]  # Limit to 20 for readability
        ]
    }
