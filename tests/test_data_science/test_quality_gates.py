"""Tests for quality gates and error handling."""

import pytest
from orchestrator.agents.data_science.quality_gates import (
    # Enums
    QualityGateStatus,
    GateCategory,
    ErrorSeverity,
    RecoveryStrategy,

    # Quality gates
    QualityGate,
    QualityGateResult,
    QualityGateSet,

    # Predefined gate sets
    DATA_QUALITY_GATES,
    CLASSIFICATION_PERFORMANCE_GATES,
    REGRESSION_PERFORMANCE_GATES,
    FAIRNESS_GATES,
    STABILITY_GATES,

    # Error handling
    ErrorContext,
    RecoveryAction,
    ErrorHandler,
    create_default_error_handler,

    # Graceful failure
    GracefulFailureReport,
    GracefulFailureHandler,

    # Quality gate manager
    QualityGateManager,
    get_quality_gate_manager,

    # Retry decorator
    with_retry,
)


class TestQualityGate:
    """Test individual quality gates."""

    def test_gate_passes(self):
        gate = QualityGate(
            name="min_auc",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum AUC score",
            threshold=0.7,
            comparison="gte",
        )

        result = gate.check(0.85, agent_name="test")
        assert result.status == QualityGateStatus.PASSED
        assert result.actual_value == 0.85

    def test_gate_fails(self):
        gate = QualityGate(
            name="min_auc",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum AUC score",
            threshold=0.7,
            comparison="gte",
        )

        result = gate.check(0.65, agent_name="test")
        assert result.status == QualityGateStatus.FAILED
        assert result.actual_value == 0.65

    def test_gate_warning(self):
        gate = QualityGate(
            name="min_auc",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum AUC score",
            threshold=0.8,
            warning_threshold=0.7,
            comparison="gte",
        )

        result = gate.check(0.75, agent_name="test")
        assert result.status == QualityGateStatus.WARNING

    def test_gate_disabled(self):
        gate = QualityGate(
            name="min_auc",
            category=GateCategory.MODEL_PERFORMANCE,
            description="Minimum AUC score",
            threshold=0.7,
            enabled=False,
        )

        result = gate.check(0.5, agent_name="test")
        assert result.status == QualityGateStatus.SKIPPED

    def test_gate_lte_comparison(self):
        gate = QualityGate(
            name="max_missing_rate",
            category=GateCategory.DATA_QUALITY,
            description="Maximum missing rate",
            threshold=0.2,
            comparison="lte",
        )

        # 0.1 <= 0.2, should pass
        result = gate.check(0.1, agent_name="test")
        assert result.status == QualityGateStatus.PASSED

        # 0.3 > 0.2, should fail
        result = gate.check(0.3, agent_name="test")
        assert result.status == QualityGateStatus.FAILED

    def test_gate_with_custom_validator(self):
        def custom_validator(data):
            # Calculate completeness from data dict
            return data.get("completeness", 0)

        gate = QualityGate(
            name="data_completeness",
            category=GateCategory.DATA_QUALITY,
            description="Data completeness check",
            threshold=0.9,
            comparison="gte",
            validator=custom_validator,
        )

        result = gate.check({"completeness": 0.95}, agent_name="test")
        assert result.status == QualityGateStatus.PASSED


class TestQualityGateSet:
    """Test quality gate sets."""

    def test_check_all_passes(self):
        gate_set = QualityGateSet(
            name="test_gates",
            gates=[
                QualityGate(
                    name="gate1",
                    category=GateCategory.MODEL_PERFORMANCE,
                    description="Test gate 1",
                    threshold=0.5,
                ),
                QualityGate(
                    name="gate2",
                    category=GateCategory.MODEL_PERFORMANCE,
                    description="Test gate 2",
                    threshold=0.6,
                ),
            ]
        )

        results = gate_set.check_all({"gate1": 0.7, "gate2": 0.8})

        assert results["gate1"].status == QualityGateStatus.PASSED
        assert results["gate2"].status == QualityGateStatus.PASSED

    def test_overall_status_passed(self):
        gate_set = QualityGateSet(
            name="test_gates",
            gates=[
                QualityGate(
                    name="gate1",
                    category=GateCategory.MODEL_PERFORMANCE,
                    description="Test",
                    threshold=0.5,
                ),
            ]
        )

        results = gate_set.check_all({"gate1": 0.7})
        status = gate_set.overall_status(results)

        assert status == QualityGateStatus.PASSED

    def test_overall_status_failed_blocking(self):
        gate_set = QualityGateSet(
            name="test_gates",
            gates=[
                QualityGate(
                    name="blocking_gate",
                    category=GateCategory.MODEL_PERFORMANCE,
                    description="Test",
                    threshold=0.7,
                    blocking=True,
                ),
                QualityGate(
                    name="non_blocking_gate",
                    category=GateCategory.MODEL_PERFORMANCE,
                    description="Test",
                    threshold=0.7,
                    blocking=False,
                ),
            ]
        )

        results = gate_set.check_all({
            "blocking_gate": 0.5,  # Fails
            "non_blocking_gate": 0.8,  # Passes
        })
        status = gate_set.overall_status(results)

        assert status == QualityGateStatus.FAILED

    def test_missing_value_skipped(self):
        gate_set = QualityGateSet(
            name="test_gates",
            gates=[
                QualityGate(
                    name="gate1",
                    category=GateCategory.MODEL_PERFORMANCE,
                    description="Test",
                    threshold=0.5,
                ),
            ]
        )

        results = gate_set.check_all({})  # No values provided
        assert results["gate1"].status == QualityGateStatus.SKIPPED


class TestPredefinedGateSets:
    """Test predefined quality gate sets."""

    def test_data_quality_gates(self):
        results = DATA_QUALITY_GATES.check_all({
            "missing_rate": 0.05,
            "duplicate_rate": 0.001,
            "data_quality_score": 0.95,
        })

        assert all(r.status == QualityGateStatus.PASSED for r in results.values())

    def test_classification_performance_gates(self):
        results = CLASSIFICATION_PERFORMANCE_GATES.check_all({
            "auc_roc": 0.85,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        })

        status = CLASSIFICATION_PERFORMANCE_GATES.overall_status(results)
        assert status == QualityGateStatus.PASSED

    def test_fairness_gates_failure(self):
        results = FAIRNESS_GATES.check_all({
            "demographic_parity_ratio": 0.6,  # Below 0.8 threshold
            "equalized_odds_ratio": 0.9,
            "max_group_disparity": 0.05,
        })

        # Should have at least one failure
        assert any(r.status == QualityGateStatus.FAILED for r in results.values())


class TestErrorHandler:
    """Test error handler."""

    def test_default_handler(self):
        handler = create_default_error_handler()

        context = ErrorContext(
            error_type="timeout",
            error_message="Request timed out",
            severity=ErrorSeverity.MEDIUM,
        )

        action = handler.get_recovery_action(context)
        assert action.strategy == RecoveryStrategy.RETRY

    def test_critical_error_aborts(self):
        handler = create_default_error_handler()

        context = ErrorContext(
            error_type="data_not_found",
            error_message="Required data file not found",
            severity=ErrorSeverity.CRITICAL,
        )

        action = handler.get_recovery_action(context)
        assert action.strategy == RecoveryStrategy.ABORT

    def test_custom_handler_registration(self):
        handler = ErrorHandler()

        handler.register_handler(
            "custom_error",
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                reason="Custom fallback",
                fallback_value="default",
            )
        )

        context = ErrorContext(
            error_type="custom_error",
            error_message="Custom error occurred",
            severity=ErrorSeverity.LOW,
        )

        action = handler.get_recovery_action(context)
        assert action.strategy == RecoveryStrategy.FALLBACK
        assert action.fallback_value == "default"


class TestGracefulFailureHandler:
    """Test graceful failure handling."""

    def test_mark_step_complete(self):
        handler = GracefulFailureHandler()

        handler.mark_step_complete("data_loading", {"rows": 1000})
        handler.mark_step_complete("data_cleaning")

        assert "data_loading" in handler._completed_steps
        assert "data_cleaning" in handler._completed_steps
        assert handler._partial_results["rows"] == 1000

    def test_handle_failure(self):
        handler = GracefulFailureHandler()

        handler.mark_step_complete("data_loading", {"rows": 1000})
        handler.mark_step_complete("feature_engineering")

        try:
            raise ValueError("Model training failed due to memory")
        except ValueError as e:
            report = handler.handle_failure(
                step_name="model_training",
                error=e,
                agent_name="modeler",
                task_id="task-001",
            )

        assert report.failed_step == "model_training"
        assert len(report.completed_steps) == 2
        assert "rows" in report.partial_results

    def test_impact_estimation(self):
        handler = GracefulFailureHandler()

        # Complete most steps
        for i in range(8):
            handler.mark_step_complete(f"step_{i}")

        try:
            raise RuntimeError("Late failure")
        except RuntimeError as e:
            report = handler.handle_failure("step_9", e)

        # Should have low impact since most work done
        assert "Low" in report.estimated_impact


class TestQualityGateManager:
    """Test quality gate manager."""

    def test_run_gates(self):
        manager = QualityGateManager()

        results = manager.run_gates(
            "classification",
            {"auc_roc": 0.85, "f1_score": 0.78}
        )

        assert "overall_status" in results
        assert "gates" in results
        assert "summary" in results

    def test_custom_gate_set_registration(self):
        manager = QualityGateManager()

        custom_set = QualityGateSet(
            name="custom",
            gates=[
                QualityGate(
                    name="custom_metric",
                    category=GateCategory.BUSINESS_RULES,
                    description="Custom business rule",
                    threshold=100,
                )
            ]
        )

        manager.register_gate_set("custom", custom_set)

        results = manager.run_gates("custom", {"custom_metric": 150})
        assert results["overall_status"] == "passed"

    def test_get_blocking_failures(self):
        manager = QualityGateManager()

        results_data = manager.run_gates(
            "classification",
            {"auc_roc": 0.5, "f1_score": 0.3}  # Below thresholds
        )

        # Get result objects
        results = {
            name: QualityGateResult(
                gate_name=name,
                category=GateCategory.MODEL_PERFORMANCE,
                status=QualityGateStatus(data["status"]),
                threshold=data["threshold"],
                actual_value=data["actual_value"],
            )
            for name, data in results_data["gates"].items()
        }

        blocking = manager.get_blocking_failures(results, "classification")
        # Should have blocking failures since values are below thresholds
        assert len(blocking) > 0


class TestRetryDecorator:
    """Test retry decorator."""

    def test_retry_success_first_attempt(self):
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_all_attempts_fail(self):
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries
