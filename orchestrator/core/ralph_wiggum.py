"""Ralph Wiggum Loop - Iterative Validation Pattern.

"I'm helping!" - Ralph Wiggum

The Ralph Wiggum Loop ensures agents don't just *think* they're done—they actually
*are* done. Every agent output goes through a self-correction cycle until it meets
objective criteria.

Credit: Inspired by Geoffrey Huntley's Ralph Wiggum Technique
https://github.com/ghuntley/how-to-ralph-wiggum

Key insight: The loop terminates via OBJECTIVE verification, not agent self-assessment.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar
from datetime import datetime


class TerminationPath(Enum):
    """How the Ralph Loop can terminate."""
    SUCCESS = "success"           # All validation criteria passed
    MAX_ITERATIONS = "max_iter"   # Hit iteration limit
    ESCALATION = "escalation"     # Needs human/supervisor intervention


class ValidationCriteria(Enum):
    """Types of objective validation criteria."""
    TESTS_PASS = "tests_pass"
    LINT_CLEAN = "lint_clean"
    TYPE_CHECK = "type_check"
    SECURITY_SCAN = "security_scan"
    BUILD_SUCCESS = "build_success"
    COVERAGE_THRESHOLD = "coverage_threshold"
    CUSTOM = "custom"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    criteria: ValidationCriteria
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoopIteration:
    """Record of a single loop iteration."""
    iteration: int
    action_taken: str
    validations: List[ValidationResult]
    all_passed: bool
    duration_ms: float
    agent_output: Optional[str] = None


@dataclass
class RalphLoopResult:
    """Final result of the Ralph Wiggum Loop."""
    termination_path: TerminationPath
    iterations: List[LoopIteration]
    total_iterations: int
    total_duration_ms: float
    final_output: Any
    success: bool
    error: Optional[str] = None

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return metrics for observability."""
        return {
            "termination_path": self.termination_path.value,
            "total_iterations": self.total_iterations,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "validation_attempts": sum(
                len(it.validations) for it in self.iterations
            ),
            "avg_iteration_ms": (
                self.total_duration_ms / self.total_iterations
                if self.total_iterations > 0 else 0
            ),
        }


class Validator(Protocol):
    """Protocol for validation functions."""
    async def __call__(self, output: Any, context: Dict[str, Any]) -> ValidationResult:
        ...


@dataclass
class RalphConfig:
    """Configuration for the Ralph Wiggum Loop.

    You don't want Ralph making ambiguous decisions.
    Give clear, objective criteria.
    """
    max_iterations: int = 10
    validators: List[Callable] = field(default_factory=list)
    require_all_pass: bool = True  # All validators must pass
    escalation_threshold: int = 5  # Escalate after this many failures
    timeout_seconds: float = 300.0  # 5 minute timeout

    # Context for validators
    context: Dict[str, Any] = field(default_factory=dict)


class RalphWiggumLoop:
    """The Ralph Wiggum Loop - iterative validation until objective criteria met.

    Usage:
        ```python
        async def test_validator(output, context):
            result = run_tests()
            return ValidationResult(
                criteria=ValidationCriteria.TESTS_PASS,
                passed=result.all_passed,
                message=f"{result.passed}/{result.total} tests passed"
            )

        config = RalphConfig(
            max_iterations=10,
            validators=[test_validator],
        )

        loop = RalphWiggumLoop(config)
        result = await loop.execute(agent_function, initial_input)
        ```
    """

    def __init__(self, config: RalphConfig):
        self.config = config
        self.iterations: List[LoopIteration] = []
        self._start_time: Optional[float] = None

    async def execute(
        self,
        agent_fn: Callable,
        initial_input: Any,
        on_iteration: Optional[Callable[[LoopIteration], None]] = None,
    ) -> RalphLoopResult:
        """Execute the Ralph Wiggum Loop.

        Args:
            agent_fn: The agent function to call. Should accept (input, feedback)
                     and return output to validate.
            initial_input: Initial input to the agent.
            on_iteration: Optional callback after each iteration.

        Returns:
            RalphLoopResult with termination path and all iterations.
        """
        self._start_time = time.time()
        self.iterations = []

        current_input = initial_input
        feedback = None
        consecutive_failures = 0

        for iteration in range(1, self.config.max_iterations + 1):
            # Check timeout
            elapsed = time.time() - self._start_time
            if elapsed > self.config.timeout_seconds:
                return self._build_result(
                    TerminationPath.MAX_ITERATIONS,
                    current_input,
                    error=f"Timeout after {elapsed:.1f}s"
                )

            iteration_start = time.time()

            # Run the agent
            try:
                if asyncio.iscoroutinefunction(agent_fn):
                    output = await agent_fn(current_input, feedback)
                else:
                    output = agent_fn(current_input, feedback)
            except Exception as e:
                return self._build_result(
                    TerminationPath.MAX_ITERATIONS,
                    current_input,
                    error=f"Agent error: {str(e)}"
                )

            # Run all validators
            validations = await self._run_validators(output)
            all_passed = all(v.passed for v in validations)

            # Record iteration
            loop_iteration = LoopIteration(
                iteration=iteration,
                action_taken=self._describe_action(feedback),
                validations=validations,
                all_passed=all_passed,
                duration_ms=(time.time() - iteration_start) * 1000,
                agent_output=str(output)[:500] if output else None,
            )
            self.iterations.append(loop_iteration)

            if on_iteration:
                on_iteration(loop_iteration)

            # Check termination conditions
            if all_passed:
                return self._build_result(
                    TerminationPath.SUCCESS,
                    output,
                    success=True
                )

            # Build feedback for next iteration
            failed_validations = [v for v in validations if not v.passed]
            feedback = self._build_feedback(failed_validations)

            # Track consecutive failures for escalation
            consecutive_failures += 1
            if consecutive_failures >= self.config.escalation_threshold:
                return self._build_result(
                    TerminationPath.ESCALATION,
                    output,
                    error=f"Escalating after {consecutive_failures} consecutive failures"
                )

            # Update input for next iteration (output becomes new input)
            current_input = output

        # Hit max iterations
        return self._build_result(
            TerminationPath.MAX_ITERATIONS,
            current_input,
            error=f"Max iterations ({self.config.max_iterations}) reached"
        )

    async def _run_validators(self, output: Any) -> List[ValidationResult]:
        """Run all configured validators."""
        results = []
        for validator in self.config.validators:
            try:
                if asyncio.iscoroutinefunction(validator):
                    result = await validator(output, self.config.context)
                else:
                    result = validator(output, self.config.context)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    criteria=ValidationCriteria.CUSTOM,
                    passed=False,
                    message=f"Validator error: {str(e)}",
                ))
        return results

    def _build_feedback(self, failed_validations: List[ValidationResult]) -> str:
        """Build feedback message from failed validations."""
        lines = ["The following validation criteria were not met:"]
        for v in failed_validations:
            lines.append(f"- {v.criteria.value}: {v.message}")
            if v.details:
                for key, value in v.details.items():
                    lines.append(f"    {key}: {value}")
        lines.append("\nPlease fix these issues and try again.")
        return "\n".join(lines)

    def _describe_action(self, feedback: Optional[str]) -> str:
        """Describe what action led to this iteration."""
        if feedback is None:
            return "Initial execution"
        return "Correction based on validation feedback"

    def _build_result(
        self,
        termination_path: TerminationPath,
        final_output: Any,
        success: bool = False,
        error: Optional[str] = None,
    ) -> RalphLoopResult:
        """Build the final result."""
        total_duration = (time.time() - self._start_time) * 1000 if self._start_time else 0

        return RalphLoopResult(
            termination_path=termination_path,
            iterations=self.iterations,
            total_iterations=len(self.iterations),
            total_duration_ms=total_duration,
            final_output=final_output,
            success=success,
            error=error,
        )


# =============================================================================
# Built-in Validators
# =============================================================================

def create_test_validator(
    test_command: str = "pytest",
    working_dir: Optional[str] = None,
) -> Callable:
    """Create a validator that runs tests."""
    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        import subprocess

        cwd = working_dir or context.get("working_dir", ".")
        try:
            result = subprocess.run(
                test_command.split(),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            passed = result.returncode == 0
            return ValidationResult(
                criteria=ValidationCriteria.TESTS_PASS,
                passed=passed,
                message="All tests passed" if passed else "Tests failed",
                details={
                    "stdout": result.stdout[-500:] if result.stdout else "",
                    "stderr": result.stderr[-500:] if result.stderr else "",
                    "returncode": result.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                criteria=ValidationCriteria.TESTS_PASS,
                passed=False,
                message="Test execution timed out",
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.TESTS_PASS,
                passed=False,
                message=f"Test execution error: {str(e)}",
            )

    return validator


def create_lint_validator(
    lint_command: str = "ruff check .",
    working_dir: Optional[str] = None,
) -> Callable:
    """Create a validator that runs linting."""
    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        import subprocess

        cwd = working_dir or context.get("working_dir", ".")
        try:
            result = subprocess.run(
                lint_command.split(),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            passed = result.returncode == 0
            return ValidationResult(
                criteria=ValidationCriteria.LINT_CLEAN,
                passed=passed,
                message="No lint errors" if passed else "Lint errors found",
                details={
                    "output": result.stdout[-500:] if result.stdout else "",
                    "errors": result.stderr[-500:] if result.stderr else "",
                }
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.LINT_CLEAN,
                passed=False,
                message=f"Lint execution error: {str(e)}",
            )

    return validator


def create_build_validator(
    build_command: str = "npm run build",
    working_dir: Optional[str] = None,
) -> Callable:
    """Create a validator that runs build."""
    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        import subprocess

        cwd = working_dir or context.get("working_dir", ".")
        try:
            result = subprocess.run(
                build_command.split(),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            passed = result.returncode == 0
            return ValidationResult(
                criteria=ValidationCriteria.BUILD_SUCCESS,
                passed=passed,
                message="Build successful" if passed else "Build failed",
                details={
                    "output": result.stdout[-500:] if result.stdout else "",
                    "errors": result.stderr[-500:] if result.stderr else "",
                }
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.BUILD_SUCCESS,
                passed=False,
                message=f"Build execution error: {str(e)}",
            )

    return validator


def create_custom_validator(
    check_fn: Callable[[Any], bool],
    criteria_name: str,
    success_message: str = "Validation passed",
    failure_message: str = "Validation failed",
) -> Callable:
    """Create a custom validator from a simple check function."""
    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        try:
            passed = check_fn(output)
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=passed,
                message=success_message if passed else failure_message,
                details={"criteria_name": criteria_name}
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=False,
                message=f"Validation error: {str(e)}",
            )

    return validator


# =============================================================================
# Integration with Orchestrator
# =============================================================================

async def execute_with_ralph_loop(
    agent_fn: Callable,
    task: Any,
    validators: List[Callable],
    max_iterations: int = 10,
    context: Optional[Dict[str, Any]] = None,
    on_iteration: Optional[Callable[[LoopIteration], None]] = None,
) -> RalphLoopResult:
    """Convenience function to execute an agent with Ralph Wiggum Loop.

    Args:
        agent_fn: Agent function to execute.
        task: Initial task/input.
        validators: List of validator functions.
        max_iterations: Maximum iterations before giving up.
        context: Additional context for validators.
        on_iteration: Callback after each iteration.

    Returns:
        RalphLoopResult with full execution details.
    """
    config = RalphConfig(
        max_iterations=max_iterations,
        validators=validators,
        context=context or {},
    )

    loop = RalphWiggumLoop(config)
    return await loop.execute(agent_fn, task, on_iteration)
