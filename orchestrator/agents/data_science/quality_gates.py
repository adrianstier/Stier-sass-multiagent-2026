"""Quality Gates and Error Handling for Data Science Multi-Agent Framework.

This module provides:
- Quality gate definitions and validation
- Error handling and recovery strategies
- Graceful failure handling
- Retry logic with exponential backoff
- Quality metrics tracking
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
import time
import functools
import logging
import traceback

logger = logging.getLogger(__name__)


class QualityGateStatus(str, Enum):
    """Status of a quality gate check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class GateCategory(str, Enum):
    """Categories of quality gates."""
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"
    FAIRNESS = "fairness"
    STABILITY = "stability"
    BUSINESS_RULES = "business_rules"
    SECURITY = "security"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Strategies for error recovery."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    category: GateCategory
    status: QualityGateStatus

    # Metrics
    threshold: float
    actual_value: float
    warning_threshold: Optional[float] = None

    # Details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checked_by: str = ""  # Agent that performed the check

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "category": self.category.value,
            "status": self.status.value,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "warning_threshold": self.warning_threshold,
            "message": self.message,
            "details": self.details,
            "recommendations": self.recommendations,
            "checked_at": self.checked_at,
            "checked_by": self.checked_by,
        }


@dataclass
class QualityGate:
    """Definition of a quality gate."""
    name: str
    category: GateCategory
    description: str

    # Thresholds
    threshold: float
    warning_threshold: Optional[float] = None
    comparison: str = "gte"  # gte, gt, lte, lt, eq

    # Behavior
    blocking: bool = True  # If True, failure stops the pipeline
    enabled: bool = True

    # Custom validation function
    validator: Optional[Callable[[Any], float]] = None

    def check(
        self,
        value: Any,
        agent_name: str = ""
    ) -> QualityGateResult:
        """Check if a value passes this quality gate."""
        if not self.enabled:
            return QualityGateResult(
                gate_name=self.name,
                category=self.category,
                status=QualityGateStatus.SKIPPED,
                threshold=self.threshold,
                actual_value=0.0,
                message="Gate is disabled",
                checked_by=agent_name,
            )

        try:
            # Get actual value
            if self.validator:
                actual_value = self.validator(value)
            else:
                actual_value = float(value)

            # Determine status
            passed = self._compare(actual_value, self.threshold)
            warning = False

            if self.warning_threshold is not None:
                warning = not passed and self._compare(
                    actual_value, self.warning_threshold
                )

            if passed:
                status = QualityGateStatus.PASSED
                message = f"Gate passed: {actual_value:.4f} meets threshold {self.threshold:.4f}"
            elif warning:
                status = QualityGateStatus.WARNING
                message = f"Warning: {actual_value:.4f} is below threshold {self.threshold:.4f} but above warning level {self.warning_threshold:.4f}"
            else:
                status = QualityGateStatus.FAILED
                message = f"Gate failed: {actual_value:.4f} does not meet threshold {self.threshold:.4f}"

            return QualityGateResult(
                gate_name=self.name,
                category=self.category,
                status=status,
                threshold=self.threshold,
                actual_value=actual_value,
                warning_threshold=self.warning_threshold,
                message=message,
                checked_by=agent_name,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                category=self.category,
                status=QualityGateStatus.ERROR,
                threshold=self.threshold,
                actual_value=0.0,
                message=f"Error during gate check: {str(e)}",
                checked_by=agent_name,
            )

    def _compare(self, actual: float, threshold: float) -> bool:
        """Compare actual value against threshold."""
        if self.comparison == "gte":
            return actual >= threshold
        elif self.comparison == "gt":
            return actual > threshold
        elif self.comparison == "lte":
            return actual <= threshold
        elif self.comparison == "lt":
            return actual < threshold
        elif self.comparison == "eq":
            return abs(actual - threshold) < 1e-9
        else:
            raise ValueError(f"Unknown comparison: {self.comparison}")


@dataclass
class QualityGateSet:
    """Collection of quality gates for a specific context."""
    name: str
    gates: List[QualityGate] = field(default_factory=list)
    require_all: bool = True  # If False, passes if any gate passes

    def check_all(
        self,
        values: Dict[str, Any],
        agent_name: str = ""
    ) -> Dict[str, QualityGateResult]:
        """Check all gates against provided values."""
        results = {}

        for gate in self.gates:
            if gate.name in values:
                results[gate.name] = gate.check(values[gate.name], agent_name)
            else:
                results[gate.name] = QualityGateResult(
                    gate_name=gate.name,
                    category=gate.category,
                    status=QualityGateStatus.SKIPPED,
                    threshold=gate.threshold,
                    actual_value=0.0,
                    message=f"No value provided for gate '{gate.name}'",
                    checked_by=agent_name,
                )

        return results

    def overall_status(
        self,
        results: Dict[str, QualityGateResult]
    ) -> QualityGateStatus:
        """Determine overall status from individual gate results."""
        statuses = [r.status for r in results.values()]

        if QualityGateStatus.ERROR in statuses:
            return QualityGateStatus.ERROR

        # Check blocking gates
        for gate in self.gates:
            if gate.blocking and gate.name in results:
                if results[gate.name].status == QualityGateStatus.FAILED:
                    return QualityGateStatus.FAILED

        if self.require_all:
            if all(s in [QualityGateStatus.PASSED, QualityGateStatus.SKIPPED]
                   for s in statuses):
                return QualityGateStatus.PASSED
            elif QualityGateStatus.WARNING in statuses:
                return QualityGateStatus.WARNING
            else:
                return QualityGateStatus.FAILED
        else:
            if any(s == QualityGateStatus.PASSED for s in statuses):
                return QualityGateStatus.PASSED
            else:
                return QualityGateStatus.FAILED


# Predefined Quality Gate Sets
DATA_QUALITY_GATES = QualityGateSet(
    name="data_quality",
    gates=[
        QualityGate(
            name="missing_rate",
            category=GateCategory.DATA_QUALITY,
            description="Maximum allowed missing value rate",
            threshold=0.2,
            warning_threshold=0.1,
            comparison="lte",
        ),
        QualityGate(
            name="duplicate_rate",
            category=GateCategory.DATA_QUALITY,
            description="Maximum allowed duplicate rate",
            threshold=0.01,
            warning_threshold=0.005,
            comparison="lte",
        ),
        QualityGate(
            name="data_quality_score",
            category=GateCategory.DATA_QUALITY,
            description="Minimum overall data quality score",
            threshold=0.8,
            warning_threshold=0.9,
            comparison="gte",
        ),
    ]
)

CLASSIFICATION_PERFORMANCE_GATES = QualityGateSet(
    name="classification_performance",
    gates=[
        QualityGate(
            name="auc_roc",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum AUC-ROC score",
            threshold=0.7,
            warning_threshold=0.8,
            comparison="gte",
        ),
        QualityGate(
            name="precision",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum precision",
            threshold=0.6,
            warning_threshold=0.7,
            comparison="gte",
            blocking=False,
        ),
        QualityGate(
            name="recall",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum recall",
            threshold=0.6,
            warning_threshold=0.7,
            comparison="gte",
            blocking=False,
        ),
        QualityGate(
            name="f1_score",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum F1 score",
            threshold=0.6,
            warning_threshold=0.7,
            comparison="gte",
        ),
    ]
)

REGRESSION_PERFORMANCE_GATES = QualityGateSet(
    name="regression_performance",
    gates=[
        QualityGate(
            name="r2_score",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum R² score",
            threshold=0.5,
            warning_threshold=0.7,
            comparison="gte",
        ),
        QualityGate(
            name="mape",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Maximum Mean Absolute Percentage Error",
            threshold=0.2,
            warning_threshold=0.1,
            comparison="lte",
        ),
    ]
)

FAIRNESS_GATES = QualityGateSet(
    name="fairness",
    gates=[
        QualityGate(
            name="demographic_parity_ratio",
            category=GateCategory.FAIRNESS,
            description="Demographic parity ratio (should be close to 1)",
            threshold=0.8,
            comparison="gte",
        ),
        QualityGate(
            name="equalized_odds_ratio",
            category=GateCategory.FAIRNESS,
            description="Equalized odds ratio",
            threshold=0.8,
            comparison="gte",
        ),
        QualityGate(
            name="max_group_disparity",
            category=GateCategory.FAIRNESS,
            description="Maximum disparity between groups",
            threshold=0.1,
            comparison="lte",
        ),
    ]
)

STABILITY_GATES = QualityGateSet(
    name="stability",
    gates=[
        QualityGate(
            name="cv_variance",
            category=GateCategory.STABILITY,
            description="Cross-validation variance",
            threshold=0.1,
            comparison="lte",
        ),
        QualityGate(
            name="feature_drift_psi",
            category=GateCategory.STABILITY,
            description="Population Stability Index for feature drift",
            threshold=0.2,
            warning_threshold=0.1,
            comparison="lte",
        ),
        QualityGate(
            name="prediction_drift_psi",
            category=GateCategory.STABILITY,
            description="Population Stability Index for prediction drift",
            threshold=0.2,
            warning_threshold=0.1,
            comparison="lte",
        ),
    ]
)


# Error Handling
@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    traceback: Optional[str] = None

    # Context
    agent_name: str = ""
    task_id: str = ""
    step_name: str = ""

    # State
    partial_results: Dict[str, Any] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "traceback": self.traceback,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "step_name": self.step_name,
            "partial_results": self.partial_results,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
        }


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    strategy: RecoveryStrategy
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # For retry strategy
    delay_seconds: float = 0
    max_retries: int = 3

    # For fallback strategy
    fallback_value: Any = None
    fallback_agent: Optional[str] = None


class ErrorHandler:
    """Handler for errors with recovery strategies."""

    def __init__(self):
        self._error_mappings: Dict[str, RecoveryAction] = {}
        self._default_strategy = RecoveryAction(
            strategy=RecoveryStrategy.ESCALATE,
            reason="No specific handler found"
        )

    def register_handler(
        self,
        error_pattern: str,
        action: RecoveryAction
    ) -> None:
        """Register a recovery action for an error pattern."""
        self._error_mappings[error_pattern] = action

    def get_recovery_action(self, error_context: ErrorContext) -> RecoveryAction:
        """Get appropriate recovery action for an error."""
        error_key = error_context.error_type

        # Check for exact match
        if error_key in self._error_mappings:
            return self._error_mappings[error_key]

        # Check for partial match
        for pattern, action in self._error_mappings.items():
            if pattern in error_key or pattern in error_context.error_message:
                return action

        # Return default
        return self._default_strategy

    def set_default_strategy(self, action: RecoveryAction) -> None:
        """Set the default recovery strategy."""
        self._default_strategy = action


# Default error handler with common mappings
def create_default_error_handler() -> ErrorHandler:
    """Create an error handler with default mappings."""
    handler = ErrorHandler()

    # Retry-able errors
    handler.register_handler(
        "timeout",
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            reason="Transient timeout, retry recommended",
            delay_seconds=10,
            max_retries=3,
        )
    )
    handler.register_handler(
        "resource_unavailable",
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            reason="Resource temporarily unavailable",
            delay_seconds=30,
            max_retries=3,
        )
    )
    handler.register_handler(
        "rate_limited",
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            reason="Rate limit hit, backing off",
            delay_seconds=60,
            max_retries=5,
        )
    )

    # Skip-able errors
    handler.register_handler(
        "optional_step_failed",
        RecoveryAction(
            strategy=RecoveryStrategy.SKIP,
            reason="Optional step failed, continuing without it",
        )
    )

    # Fallback errors
    handler.register_handler(
        "model_training_failed",
        RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            reason="Model training failed, using fallback model",
            fallback_value="baseline_model",
        )
    )

    # Critical errors
    handler.register_handler(
        "data_not_found",
        RecoveryAction(
            strategy=RecoveryStrategy.ABORT,
            reason="Required data not found",
        )
    )
    handler.register_handler(
        "permission_denied",
        RecoveryAction(
            strategy=RecoveryStrategy.ABORT,
            reason="Permission denied",
        )
    )
    handler.register_handler(
        "invalid_input",
        RecoveryAction(
            strategy=RecoveryStrategy.ESCALATE,
            reason="Invalid input, human review needed",
        )
    )

    return handler


# Retry decorator with exponential backoff
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


# Graceful failure handling
@dataclass
class GracefulFailureReport:
    """Report for graceful failure handling."""
    failed_step: str
    error_context: ErrorContext
    recovery_action: RecoveryAction

    # Results
    partial_results: Dict[str, Any] = field(default_factory=dict)
    completed_steps: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    estimated_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_step": self.failed_step,
            "error_context": self.error_context.to_dict(),
            "recovery_action": {
                "strategy": self.recovery_action.strategy.value,
                "reason": self.recovery_action.reason,
            },
            "partial_results": self.partial_results,
            "completed_steps": self.completed_steps,
            "recommendations": self.recommendations,
            "estimated_impact": self.estimated_impact,
        }


class GracefulFailureHandler:
    """Handler for graceful failure scenarios."""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or create_default_error_handler()
        self._completed_steps: List[str] = []
        self._partial_results: Dict[str, Any] = {}

    def mark_step_complete(
        self,
        step_name: str,
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a step as completed with optional results."""
        self._completed_steps.append(step_name)
        if results:
            self._partial_results.update(results)

    def handle_failure(
        self,
        step_name: str,
        error: Exception,
        agent_name: str = "",
        task_id: str = "",
    ) -> GracefulFailureReport:
        """Handle a failure gracefully."""
        # Create error context
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._determine_severity(error),
            traceback=traceback.format_exc(),
            agent_name=agent_name,
            task_id=task_id,
            step_name=step_name,
            partial_results=self._partial_results.copy(),
        )

        # Get recovery action
        recovery_action = self.error_handler.get_recovery_action(error_context)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            error_context, recovery_action
        )

        # Create report
        report = GracefulFailureReport(
            failed_step=step_name,
            error_context=error_context,
            recovery_action=recovery_action,
            partial_results=self._partial_results.copy(),
            completed_steps=self._completed_steps.copy(),
            recommendations=recommendations,
            estimated_impact=self._estimate_impact(step_name),
        )

        return report

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine severity based on error type."""
        error_type = type(error).__name__

        critical_errors = ["SystemError", "MemoryError", "RecursionError"]
        high_errors = ["ValueError", "TypeError", "KeyError", "AttributeError"]
        medium_errors = ["TimeoutError", "ConnectionError", "IOError"]

        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _generate_recommendations(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction
    ) -> List[str]:
        """Generate recommendations based on error and recovery action."""
        recommendations = []

        if recovery_action.strategy == RecoveryStrategy.RETRY:
            recommendations.append(
                f"Retry the operation with a delay of {recovery_action.delay_seconds}s"
            )
            recommendations.append(
                f"Maximum {recovery_action.max_retries} retries recommended"
            )

        elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
            recommendations.append(
                f"Use fallback: {recovery_action.fallback_value}"
            )
            if recovery_action.fallback_agent:
                recommendations.append(
                    f"Consider delegating to {recovery_action.fallback_agent}"
                )

        elif recovery_action.strategy == RecoveryStrategy.ESCALATE:
            recommendations.append("Escalate to human review")
            recommendations.append("Provide detailed error context for debugging")

        elif recovery_action.strategy == RecoveryStrategy.SKIP:
            recommendations.append(f"Skip step '{error_context.step_name}' and continue")
            recommendations.append("Document skipped step in final report")

        elif recovery_action.strategy == RecoveryStrategy.ABORT:
            recommendations.append("Abort the current task")
            recommendations.append("Save partial results before terminating")

        return recommendations

    def _estimate_impact(self, failed_step: str) -> str:
        """Estimate the impact of failure at a specific step."""
        # This is a simplified estimation
        total_steps = len(self._completed_steps) + 1
        completion_rate = len(self._completed_steps) / total_steps

        if completion_rate >= 0.8:
            return "Low - Most work completed"
        elif completion_rate >= 0.5:
            return "Medium - Significant work lost"
        else:
            return "High - Most work needs to be redone"

    def reset(self) -> None:
        """Reset the handler state."""
        self._completed_steps = []
        self._partial_results = {}


# Quality Gate Manager
class QualityGateManager:
    """Manager for running quality gate checks."""

    def __init__(self):
        self._gate_sets: Dict[str, QualityGateSet] = {
            "data_quality": DATA_QUALITY_GATES,
            "classification": CLASSIFICATION_PERFORMANCE_GATES,
            "regression": REGRESSION_PERFORMANCE_GATES,
            "fairness": FAIRNESS_GATES,
            "stability": STABILITY_GATES,
        }
        self._history: List[Dict[str, QualityGateResult]] = []

    def register_gate_set(self, name: str, gate_set: QualityGateSet) -> None:
        """Register a custom gate set."""
        self._gate_sets[name] = gate_set

    def run_gates(
        self,
        gate_set_name: str,
        values: Dict[str, Any],
        agent_name: str = ""
    ) -> Dict[str, Any]:
        """Run a gate set and return results."""
        if gate_set_name not in self._gate_sets:
            raise ValueError(f"Unknown gate set: {gate_set_name}")

        gate_set = self._gate_sets[gate_set_name]
        results = gate_set.check_all(values, agent_name)
        overall_status = gate_set.overall_status(results)

        # Store in history
        self._history.append(results)

        return {
            "gate_set": gate_set_name,
            "overall_status": overall_status.value,
            "gates": {k: v.to_dict() for k, v in results.items()},
            "summary": self._create_summary(results),
        }

    def _create_summary(
        self,
        results: Dict[str, QualityGateResult]
    ) -> Dict[str, Any]:
        """Create a summary of gate results."""
        passed = sum(1 for r in results.values()
                     if r.status == QualityGateStatus.PASSED)
        warnings = sum(1 for r in results.values()
                       if r.status == QualityGateStatus.WARNING)
        failed = sum(1 for r in results.values()
                     if r.status == QualityGateStatus.FAILED)
        skipped = sum(1 for r in results.values()
                      if r.status == QualityGateStatus.SKIPPED)

        return {
            "total": len(results),
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": passed / len(results) if results else 0,
        }

    def get_blocking_failures(
        self,
        results: Dict[str, QualityGateResult],
        gate_set_name: str
    ) -> List[QualityGateResult]:
        """Get list of blocking gate failures."""
        if gate_set_name not in self._gate_sets:
            return []

        gate_set = self._gate_sets[gate_set_name]
        blocking_gates = {g.name for g in gate_set.gates if g.blocking}

        return [
            r for name, r in results.items()
            if name in blocking_gates and r.status == QualityGateStatus.FAILED
        ]


# Global instances
_error_handler: Optional[ErrorHandler] = None
_quality_gate_manager: Optional[QualityGateManager] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = create_default_error_handler()
    return _error_handler


def get_quality_gate_manager() -> QualityGateManager:
    """Get the global quality gate manager."""
    global _quality_gate_manager
    if _quality_gate_manager is None:
        _quality_gate_manager = QualityGateManager()
    return _quality_gate_manager
