"""Tests for the Ralph Wiggum Loop."""

import pytest
import asyncio
from orchestrator.core.ralph_wiggum import (
    RalphWiggumLoop,
    RalphConfig,
    RalphLoopResult,
    TerminationPath,
    ValidationCriteria,
    ValidationResult,
    LoopIteration,
    create_custom_validator,
    execute_with_ralph_loop,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_validation_result(self):
        """Should create validation result with all fields."""
        result = ValidationResult(
            criteria=ValidationCriteria.TESTS_PASS,
            passed=True,
            message="All tests passed",
            details={"count": 10}
        )

        assert result.criteria == ValidationCriteria.TESTS_PASS
        assert result.passed is True
        assert result.message == "All tests passed"
        assert result.details["count"] == 10
        assert result.timestamp is not None

    def test_failed_validation_result(self):
        """Should create failed validation result."""
        result = ValidationResult(
            criteria=ValidationCriteria.LINT_CLEAN,
            passed=False,
            message="5 lint errors found"
        )

        assert result.passed is False
        assert "5 lint errors" in result.message


class TestRalphConfig:
    """Tests for RalphConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = RalphConfig()

        assert config.max_iterations == 10
        assert config.require_all_pass is True
        assert config.escalation_threshold == 5
        assert config.timeout_seconds == 300.0

    def test_custom_config(self):
        """Should accept custom values."""
        config = RalphConfig(
            max_iterations=5,
            escalation_threshold=3,
            timeout_seconds=60.0,
            context={"project": "test"}
        )

        assert config.max_iterations == 5
        assert config.escalation_threshold == 3
        assert config.context["project"] == "test"


class TestRalphWiggumLoop:
    """Tests for RalphWiggumLoop execution."""

    @pytest.fixture
    def always_pass_validator(self):
        """Validator that always passes."""
        async def validator(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,
                message="Always passes"
            )
        return validator

    @pytest.fixture
    def always_fail_validator(self):
        """Validator that always fails."""
        async def validator(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=False,
                message="Always fails"
            )
        return validator

    @pytest.fixture
    def pass_after_n_validator(self):
        """Validator that passes after N attempts."""
        attempt_count = {"count": 0}

        async def validator(output, context):
            attempt_count["count"] += 1
            n = context.get("pass_after", 3)
            passed = attempt_count["count"] >= n
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=passed,
                message=f"Attempt {attempt_count['count']}, pass_after={n}"
            )
        return validator

    @pytest.mark.asyncio
    async def test_immediate_success(self, always_pass_validator):
        """Should terminate immediately when validation passes."""
        config = RalphConfig(
            max_iterations=10,
            validators=[always_pass_validator]
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return "output"

        result = await loop.execute(agent_fn, "input")

        assert result.success is True
        assert result.termination_path == TerminationPath.SUCCESS
        assert result.total_iterations == 1

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, always_fail_validator):
        """Should stop at max iterations."""
        config = RalphConfig(
            max_iterations=3,
            validators=[always_fail_validator],
            escalation_threshold=10  # Don't escalate
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return "output"

        result = await loop.execute(agent_fn, "input")

        assert result.success is False
        assert result.termination_path == TerminationPath.MAX_ITERATIONS
        assert result.total_iterations == 3
        assert "Max iterations" in result.error

    @pytest.mark.asyncio
    async def test_escalation_after_failures(self, always_fail_validator):
        """Should escalate after consecutive failures."""
        config = RalphConfig(
            max_iterations=10,
            validators=[always_fail_validator],
            escalation_threshold=3
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return "output"

        result = await loop.execute(agent_fn, "input")

        assert result.success is False
        assert result.termination_path == TerminationPath.ESCALATION
        assert result.total_iterations == 3
        assert "Escalating" in result.error

    @pytest.mark.asyncio
    async def test_success_after_retries(self, pass_after_n_validator):
        """Should succeed after corrections."""
        config = RalphConfig(
            max_iterations=10,
            validators=[pass_after_n_validator],
            context={"pass_after": 3}
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return f"attempt with feedback: {feedback}"

        result = await loop.execute(agent_fn, "input")

        assert result.success is True
        assert result.termination_path == TerminationPath.SUCCESS
        assert result.total_iterations == 3

    @pytest.mark.asyncio
    async def test_iteration_callback(self, pass_after_n_validator):
        """Should call iteration callback."""
        config = RalphConfig(
            max_iterations=10,
            validators=[pass_after_n_validator],
            context={"pass_after": 2}
        )
        loop = RalphWiggumLoop(config)

        iterations_seen = []

        def on_iteration(iteration: LoopIteration):
            iterations_seen.append(iteration)

        async def agent_fn(input, feedback):
            return "output"

        await loop.execute(agent_fn, "input", on_iteration=on_iteration)

        assert len(iterations_seen) == 2
        assert iterations_seen[0].iteration == 1
        assert iterations_seen[1].iteration == 2

    @pytest.mark.asyncio
    async def test_sync_agent_function(self, always_pass_validator):
        """Should work with sync agent functions."""
        config = RalphConfig(
            max_iterations=10,
            validators=[always_pass_validator]
        )
        loop = RalphWiggumLoop(config)

        def sync_agent_fn(input, feedback):
            return "sync output"

        result = await loop.execute(sync_agent_fn, "input")

        assert result.success is True
        assert "sync output" in result.final_output

    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Should handle agent errors gracefully."""
        async def failing_validator(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,
                message="Would pass"
            )

        config = RalphConfig(
            max_iterations=10,
            validators=[failing_validator]
        )
        loop = RalphWiggumLoop(config)

        async def failing_agent(input, feedback):
            raise ValueError("Agent crashed!")

        result = await loop.execute(failing_agent, "input")

        assert result.success is False
        assert "Agent error" in result.error

    @pytest.mark.asyncio
    async def test_validator_error_handling(self):
        """Should handle validator errors."""
        async def crashing_validator(output, context):
            raise RuntimeError("Validator crashed!")

        config = RalphConfig(
            max_iterations=3,
            validators=[crashing_validator],
            escalation_threshold=10
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return "output"

        result = await loop.execute(agent_fn, "input")

        # Validator error counts as failure
        assert result.success is False
        assert result.total_iterations == 3

    @pytest.mark.asyncio
    async def test_metrics_collection(self, pass_after_n_validator):
        """Should collect metrics correctly."""
        config = RalphConfig(
            max_iterations=10,
            validators=[pass_after_n_validator],
            context={"pass_after": 2}
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return "output"

        result = await loop.execute(agent_fn, "input")
        metrics = result.metrics

        assert metrics["termination_path"] == "success"
        assert metrics["total_iterations"] == 2
        assert metrics["success"] is True
        assert metrics["validation_attempts"] == 2
        assert metrics["total_duration_ms"] > 0


class TestCustomValidator:
    """Tests for custom validator creation."""

    @pytest.mark.asyncio
    async def test_create_custom_validator_passing(self):
        """Should create validator from check function."""
        validator = create_custom_validator(
            check_fn=lambda x: x > 5,
            criteria_name="greater_than_5",
            success_message="Value is greater than 5",
            failure_message="Value is not greater than 5"
        )

        result = await validator(10, {})

        assert result.passed is True
        assert result.message == "Value is greater than 5"

    @pytest.mark.asyncio
    async def test_create_custom_validator_failing(self):
        """Should fail when check returns False."""
        validator = create_custom_validator(
            check_fn=lambda x: x > 5,
            criteria_name="greater_than_5",
            failure_message="Too small!"
        )

        result = await validator(3, {})

        assert result.passed is False
        assert result.message == "Too small!"

    @pytest.mark.asyncio
    async def test_create_custom_validator_with_error(self):
        """Should handle check function errors."""
        def bad_check(x):
            raise ValueError("Check failed!")

        validator = create_custom_validator(
            check_fn=bad_check,
            criteria_name="bad_check"
        )

        result = await validator("anything", {})

        assert result.passed is False
        assert "Validation error" in result.message


class TestExecuteWithRalphLoop:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_execute_convenience_function(self):
        """Should execute with convenience function."""
        async def always_pass(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,
                message="Pass"
            )

        async def agent_fn(input, feedback):
            return "done"

        result = await execute_with_ralph_loop(
            agent_fn=agent_fn,
            task="do something",
            validators=[always_pass],
            max_iterations=5
        )

        assert result.success is True
        assert result.total_iterations == 1


class TestMultipleValidators:
    """Tests for multiple validators."""

    @pytest.mark.asyncio
    async def test_all_validators_must_pass(self):
        """All validators must pass for success."""
        async def pass_validator(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.TESTS_PASS,
                passed=True,
                message="Tests pass"
            )

        async def fail_validator(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.LINT_CLEAN,
                passed=False,
                message="Lint fails"
            )

        config = RalphConfig(
            max_iterations=2,
            validators=[pass_validator, fail_validator],
            escalation_threshold=10
        )
        loop = RalphWiggumLoop(config)

        async def agent_fn(input, feedback):
            return "output"

        result = await loop.execute(agent_fn, "input")

        # Should fail because one validator fails
        assert result.success is False
        assert result.total_iterations == 2

    @pytest.mark.asyncio
    async def test_feedback_includes_all_failures(self):
        """Feedback should include all failed validations."""
        failures = []

        async def fail_tests(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.TESTS_PASS,
                passed=False,
                message="3 tests failed"
            )

        async def fail_lint(output, context):
            return ValidationResult(
                criteria=ValidationCriteria.LINT_CLEAN,
                passed=False,
                message="10 lint errors"
            )

        config = RalphConfig(
            max_iterations=2,
            validators=[fail_tests, fail_lint],
            escalation_threshold=10
        )
        loop = RalphWiggumLoop(config)

        received_feedback = []

        async def agent_fn(input, feedback):
            if feedback:
                received_feedback.append(feedback)
            return "output"

        await loop.execute(agent_fn, "input")

        # Check that feedback mentioned both failures
        assert len(received_feedback) == 1
        feedback = received_feedback[0]
        assert "tests_pass" in feedback
        assert "lint_clean" in feedback
