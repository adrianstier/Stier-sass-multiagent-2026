"""Cost Prediction and Optimization.

Provides:
- Estimate token cost before agent execution based on historical data
- Warn when approaching budget with remaining work estimate
- Suggest cheaper model alternatives for non-critical tasks
- Cost breakdown by phase/agent
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from orchestrator.core.config import settings
from orchestrator.core.database import get_db
from orchestrator.core.models import Run, Task, Event, TaskStatus
from orchestrator.core.rate_limit import MODEL_PRICING, calculate_cost
from orchestrator.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Model Recommendations
# =============================================================================

# Model capabilities and costs
MODEL_TIERS = {
    "opus": {
        "model": "claude-3-opus-20240229",
        "input_cost_per_m": 15.00,
        "output_cost_per_m": 75.00,
        "capabilities": ["complex_reasoning", "creative_writing", "code_review", "architecture"],
        "recommended_for": ["tech_lead", "security_reviewer"],
    },
    "sonnet": {
        "model": "claude-sonnet-4-20250514",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
        "capabilities": ["general", "code_generation", "analysis", "planning"],
        "recommended_for": ["business_analyst", "project_manager", "code_reviewer", "backend_engineer", "frontend_engineer"],
    },
    "haiku": {
        "model": "claude-3-haiku-20240307",
        "input_cost_per_m": 0.25,
        "output_cost_per_m": 1.25,
        "capabilities": ["simple_tasks", "formatting", "extraction", "cleanup"],
        "recommended_for": ["cleanup_agent", "design_reviewer"],
    },
}

# Historical token usage averages by task type (from analysis)
HISTORICAL_AVERAGES = {
    "requirements_analysis": {"input": 3000, "output": 2000, "variance": 0.3},
    "project_planning": {"input": 4000, "output": 3000, "variance": 0.25},
    "ux_design": {"input": 3500, "output": 4000, "variance": 0.35},
    "technical_architecture": {"input": 5000, "output": 5000, "variance": 0.3},
    "database_design": {"input": 3000, "output": 2500, "variance": 0.2},
    "backend_development": {"input": 6000, "output": 8000, "variance": 0.4},
    "frontend_development": {"input": 5000, "output": 7000, "variance": 0.4},
    "code_review": {"input": 8000, "output": 3000, "variance": 0.35},
    "security_review": {"input": 6000, "output": 2500, "variance": 0.3},
    "default": {"input": 4000, "output": 3000, "variance": 0.3},
}


# =============================================================================
# Cost Prediction
# =============================================================================

@dataclass
class CostEstimate:
    """Estimated cost for a task or run."""

    task_type: Optional[str] = None
    model: str = ""

    # Token estimates
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    total_estimated_tokens: int = 0

    # Cost estimates
    estimated_cost_usd: float = 0.0
    cost_lower_bound: float = 0.0
    cost_upper_bound: float = 0.0

    # Confidence
    confidence: float = 0.8  # Based on historical data availability
    variance: float = 0.3

    # Recommendations
    recommended_model: Optional[str] = None
    potential_savings_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "model": self.model,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "total_estimated_tokens": self.total_estimated_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cost_lower_bound": round(self.cost_lower_bound, 6),
            "cost_upper_bound": round(self.cost_upper_bound, 6),
            "confidence": self.confidence,
            "recommended_model": self.recommended_model,
            "potential_savings_usd": round(self.potential_savings_usd, 6),
        }


@dataclass
class RunCostForecast:
    """Full cost forecast for a run."""

    run_id: str
    current_cost_usd: float
    tokens_used: int
    budget_tokens: Optional[int]

    # Remaining work estimates
    remaining_tasks: int
    estimated_remaining_cost: float
    estimated_remaining_tokens: int

    # Totals
    estimated_total_cost: float
    estimated_total_tokens: int

    # Budget status
    budget_status: str  # healthy, warning, critical, exceeded
    budget_usage_percentage: float
    estimated_final_usage_percentage: float

    # Task-level estimates
    task_estimates: List[CostEstimate]

    # Recommendations
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "current_cost_usd": round(self.current_cost_usd, 6),
            "tokens_used": self.tokens_used,
            "budget_tokens": self.budget_tokens,
            "remaining_tasks": self.remaining_tasks,
            "estimated_remaining_cost": round(self.estimated_remaining_cost, 6),
            "estimated_remaining_tokens": self.estimated_remaining_tokens,
            "estimated_total_cost": round(self.estimated_total_cost, 6),
            "estimated_total_tokens": self.estimated_total_tokens,
            "budget_status": self.budget_status,
            "budget_usage_percentage": round(self.budget_usage_percentage, 1),
            "estimated_final_usage_percentage": round(self.estimated_final_usage_percentage, 1),
            "task_estimates": [e.to_dict() for e in self.task_estimates],
            "recommendations": self.recommendations,
        }


class CostPredictor:
    """Predicts and optimizes costs for agent executions."""

    def __init__(self):
        self._historical_cache: Dict[str, Dict[str, Any]] = {}

    def estimate_task_cost(
        self,
        task_type: str,
        assigned_role: str,
        model: Optional[str] = None,
        context_tokens: int = 0,
    ) -> CostEstimate:
        """Estimate the cost for executing a single task.

        Args:
            task_type: The task type
            assigned_role: The assigned agent role
            model: Optional specific model (uses default if not specified)
            context_tokens: Additional tokens from context/dependencies

        Returns:
            CostEstimate with predictions
        """
        model = model or settings.llm_model

        # Get historical averages
        averages = HISTORICAL_AVERAGES.get(task_type, HISTORICAL_AVERAGES["default"])

        # Adjust for context
        estimated_input = averages["input"] + context_tokens
        estimated_output = averages["output"]
        variance = averages["variance"]

        # Calculate costs
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-20250514"])
        estimated_cost = calculate_cost(model, estimated_input, estimated_output)

        lower_bound = estimated_cost * (1 - variance)
        upper_bound = estimated_cost * (1 + variance)

        # Check for cheaper model recommendations
        recommended_model = None
        potential_savings = 0.0

        cheaper_model = self._get_cheaper_model_recommendation(task_type, assigned_role)
        if cheaper_model and cheaper_model != model:
            cheaper_pricing = MODEL_PRICING.get(cheaper_model)
            if cheaper_pricing:
                cheaper_cost = calculate_cost(cheaper_model, estimated_input, estimated_output)
                if cheaper_cost < estimated_cost:
                    recommended_model = cheaper_model
                    potential_savings = estimated_cost - cheaper_cost

        return CostEstimate(
            task_type=task_type,
            model=model,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            total_estimated_tokens=estimated_input + estimated_output,
            estimated_cost_usd=estimated_cost,
            cost_lower_bound=lower_bound,
            cost_upper_bound=upper_bound,
            variance=variance,
            recommended_model=recommended_model,
            potential_savings_usd=potential_savings,
        )

    def _get_cheaper_model_recommendation(
        self, task_type: str, assigned_role: str
    ) -> Optional[str]:
        """Get a cheaper model recommendation if appropriate."""
        # Check if task/role can use a cheaper model
        for tier_name, tier_info in MODEL_TIERS.items():
            if assigned_role in tier_info["recommended_for"]:
                return tier_info["model"]

        # Default recommendations based on task criticality
        non_critical_tasks = ["cleanup_report", "design_review"]
        if task_type in non_critical_tasks:
            return MODEL_TIERS["haiku"]["model"]

        return None

    def forecast_run_cost(
        self,
        run_id: str,
        model: Optional[str] = None,
    ) -> RunCostForecast:
        """Forecast the total cost for a run.

        Args:
            run_id: The run ID
            model: Optional model override

        Returns:
            RunCostForecast with full predictions
        """
        model = model or settings.llm_model

        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                raise ValueError(f"Run not found: {run_id}")

            tasks = db.query(Task).filter(Task.run_id == UUID(run_id)).all()

            # Current state
            current_cost = float(run.total_cost_usd or "0")
            tokens_used = run.tokens_used or 0
            budget_tokens = run.budget_tokens

            # Estimate remaining tasks
            remaining_tasks = [t for t in tasks if t.status in [TaskStatus.PENDING, TaskStatus.QUEUED]]
            task_estimates = []
            total_remaining_tokens = 0
            total_remaining_cost = 0.0

            for task in remaining_tasks:
                # Estimate context size from dependencies
                context_tokens = 0
                for dep_id in (task.dependencies or []):
                    dep_task = db.query(Task).filter(Task.id == UUID(dep_id)).first()
                    if dep_task and dep_task.status == TaskStatus.COMPLETED:
                        context_tokens += 2000  # Approximate context per dependency

                estimate = self.estimate_task_cost(
                    task.task_type,
                    task.assigned_role,
                    model,
                    context_tokens,
                )
                task_estimates.append(estimate)
                total_remaining_tokens += estimate.total_estimated_tokens
                total_remaining_cost += estimate.estimated_cost_usd

            # Calculate totals
            estimated_total_cost = current_cost + total_remaining_cost
            estimated_total_tokens = tokens_used + total_remaining_tokens

            # Budget status
            budget_usage = 0.0
            estimated_final_usage = 0.0

            if budget_tokens:
                budget_usage = (tokens_used / budget_tokens) * 100
                estimated_final_usage = (estimated_total_tokens / budget_tokens) * 100

            if estimated_final_usage >= 100:
                budget_status = "exceeded"
            elif estimated_final_usage >= 90:
                budget_status = "critical"
            elif estimated_final_usage >= 70:
                budget_status = "warning"
            else:
                budget_status = "healthy"

            # Generate recommendations
            recommendations = self._generate_recommendations(
                budget_status,
                task_estimates,
                estimated_final_usage,
                budget_tokens,
            )

            return RunCostForecast(
                run_id=run_id,
                current_cost_usd=current_cost,
                tokens_used=tokens_used,
                budget_tokens=budget_tokens,
                remaining_tasks=len(remaining_tasks),
                estimated_remaining_cost=total_remaining_cost,
                estimated_remaining_tokens=total_remaining_tokens,
                estimated_total_cost=estimated_total_cost,
                estimated_total_tokens=estimated_total_tokens,
                budget_status=budget_status,
                budget_usage_percentage=budget_usage,
                estimated_final_usage_percentage=estimated_final_usage,
                task_estimates=task_estimates,
                recommendations=recommendations,
            )

    def _generate_recommendations(
        self,
        budget_status: str,
        task_estimates: List[CostEstimate],
        estimated_final_usage: float,
        budget_tokens: Optional[int],
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Budget warnings
        if budget_status == "exceeded":
            recommendations.append(
                "⚠️ CRITICAL: Estimated token usage exceeds budget. "
                "Consider increasing budget or reducing scope."
            )
        elif budget_status == "critical":
            recommendations.append(
                "⚠️ WARNING: Approaching budget limit. "
                f"Estimated {estimated_final_usage:.0f}% usage."
            )

        # Model recommendations
        total_potential_savings = sum(e.potential_savings_usd for e in task_estimates)
        if total_potential_savings > 0.10:
            recommendations.append(
                f"💡 Use cheaper models for non-critical tasks to save ~${total_potential_savings:.2f}"
            )

        # Specific model suggestions
        for estimate in task_estimates:
            if estimate.recommended_model and estimate.potential_savings_usd > 0.05:
                recommendations.append(
                    f"→ {estimate.task_type}: Use {estimate.recommended_model} "
                    f"(save ${estimate.potential_savings_usd:.2f})"
                )

        # General tips
        if not recommendations:
            recommendations.append("✅ Cost usage looks healthy. No optimizations needed.")

        return recommendations

    def get_cost_breakdown(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        """Get a detailed cost breakdown for a completed or in-progress run.

        Args:
            run_id: The run ID

        Returns:
            Dict with cost breakdown by phase and agent
        """
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                raise ValueError(f"Run not found: {run_id}")

            # Get all LLM events
            events = db.query(Event).filter(
                Event.run_id == UUID(run_id),
                Event.event_type == "llm_response",
            ).all()

            # Aggregate by role/task
            by_role: Dict[str, Dict[str, Any]] = {}
            by_task_type: Dict[str, Dict[str, Any]] = {}

            for event in events:
                data = event.data or {}
                tokens = data.get("tokens_used", 0)
                cost = data.get("cost_usd", 0)
                role = event.actor

                # Get task info
                task = None
                if event.task_id:
                    task = db.query(Task).filter(Task.id == event.task_id).first()

                # Aggregate by role
                if role not in by_role:
                    by_role[role] = {"tokens": 0, "cost": 0, "requests": 0}
                by_role[role]["tokens"] += tokens
                by_role[role]["cost"] += cost
                by_role[role]["requests"] += 1

                # Aggregate by task type
                if task:
                    task_type = task.task_type
                    if task_type not in by_task_type:
                        by_task_type[task_type] = {"tokens": 0, "cost": 0, "requests": 0}
                    by_task_type[task_type]["tokens"] += tokens
                    by_task_type[task_type]["cost"] += cost
                    by_task_type[task_type]["requests"] += 1

            total_cost = float(run.total_cost_usd or "0")
            total_tokens = run.tokens_used or 0

            return {
                "run_id": run_id,
                "total_cost_usd": round(total_cost, 6),
                "total_tokens": total_tokens,
                "by_role": {
                    role: {
                        "tokens": data["tokens"],
                        "cost_usd": round(data["cost"], 6),
                        "requests": data["requests"],
                        "percentage": round((data["cost"] / total_cost) * 100, 1) if total_cost > 0 else 0,
                    }
                    for role, data in by_role.items()
                },
                "by_task_type": {
                    task_type: {
                        "tokens": data["tokens"],
                        "cost_usd": round(data["cost"], 6),
                        "requests": data["requests"],
                        "percentage": round((data["cost"] / total_cost) * 100, 1) if total_cost > 0 else 0,
                    }
                    for task_type, data in by_task_type.items()
                },
            }

    def check_budget_before_execution(
        self,
        run_id: str,
        task_type: str,
        assigned_role: str,
    ) -> Tuple[bool, Optional[str], CostEstimate]:
        """Check if execution should proceed based on budget.

        Args:
            run_id: The run ID
            task_type: The task type
            assigned_role: The assigned role

        Returns:
            Tuple of (should_proceed, warning_message, cost_estimate)
        """
        estimate = self.estimate_task_cost(task_type, assigned_role)

        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(run_id)).first()
            if not run:
                return False, "Run not found", estimate

            budget = run.budget_tokens
            used = run.tokens_used or 0

            if not budget:
                return True, None, estimate

            projected = used + estimate.total_estimated_tokens

            if projected > budget:
                # Would exceed budget
                return False, (
                    f"Estimated execution would exceed budget. "
                    f"Used: {used}, Estimated: {estimate.total_estimated_tokens}, "
                    f"Budget: {budget}"
                ), estimate

            usage_after = (projected / budget) * 100

            if usage_after >= 90:
                return True, (
                    f"⚠️ After this task, budget usage will be ~{usage_after:.0f}%. "
                    f"Consider increasing budget for remaining work."
                ), estimate

            if usage_after >= 70:
                return True, (
                    f"Budget usage will be ~{usage_after:.0f}% after this task."
                ), estimate

            return True, None, estimate


# Global cost predictor instance
_cost_predictor: Optional[CostPredictor] = None


def get_cost_predictor() -> CostPredictor:
    """Get the global cost predictor instance."""
    global _cost_predictor
    if _cost_predictor is None:
        _cost_predictor = CostPredictor()
    return _cost_predictor
