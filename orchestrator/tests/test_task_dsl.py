"""Tests for the task DSL module."""

import pytest
from orchestrator.core.task_dsl import (
    TaskSpec, WorkflowPlan, ValidationMethod,
    create_standard_workflow, create_minimal_workflow,
    create_frontend_workflow, create_design_iteration_workflow
)


class TestTaskSpec:
    """Tests for TaskSpec dataclass."""

    def test_create_basic_task(self):
        """Test creating a basic task specification."""
        task = TaskSpec(
            task_type="requirements_analysis",
            assigned_role="business_analyst",
            description="Analyze business requirements",
        )
        assert task.task_type == "requirements_analysis"
        assert task.assigned_role == "business_analyst"
        assert task.dependencies == []
        assert task.expected_artifacts == []

    def test_create_task_with_dependencies(self):
        """Test creating a task with dependencies."""
        task = TaskSpec(
            task_type="project_planning",
            assigned_role="project_manager",
            description="Create project plan",
            dependencies=["requirements_analysis"],
            expected_artifacts=["project_plan"],
        )
        assert task.dependencies == ["requirements_analysis"]
        assert task.expected_artifacts == ["project_plan"]

    def test_idempotency_key_generation(self):
        """Test that idempotency keys are unique."""
        task = TaskSpec(
            task_type="test_task",
            assigned_role="test_role",
            description="Test",
        )
        key1 = task.generate_idempotency_key("run-1")
        key2 = task.generate_idempotency_key("run-2")
        key3 = task.generate_idempotency_key("run-1")

        assert key1 != key2
        assert key1 == key3  # Same run should produce same key


class TestWorkflowPlan:
    """Tests for WorkflowPlan class."""

    def test_create_empty_plan(self):
        """Test creating an empty workflow plan."""
        plan = WorkflowPlan()
        assert plan.tasks == []
        assert plan.success_criteria == []

    def test_add_tasks(self):
        """Test adding tasks to a plan."""
        plan = WorkflowPlan()
        plan.add_task(TaskSpec(
            task_type="task1",
            assigned_role="role1",
            description="Task 1",
        ))
        plan.add_task(TaskSpec(
            task_type="task2",
            assigned_role="role2",
            description="Task 2",
            dependencies=["task1"],
        ))
        assert len(plan.tasks) == 2

    def test_get_ready_tasks_no_dependencies(self):
        """Test getting ready tasks when no dependencies."""
        plan = WorkflowPlan()
        plan.add_task(TaskSpec(
            task_type="task1",
            assigned_role="role1",
            description="Task 1",
        ))
        plan.add_task(TaskSpec(
            task_type="task2",
            assigned_role="role2",
            description="Task 2",
        ))

        ready = plan.get_ready_tasks([])
        assert len(ready) == 2

    def test_get_ready_tasks_with_dependencies(self):
        """Test getting ready tasks with dependencies."""
        plan = WorkflowPlan()
        plan.add_task(TaskSpec(
            task_type="task1",
            assigned_role="role1",
            description="Task 1",
        ))
        plan.add_task(TaskSpec(
            task_type="task2",
            assigned_role="role2",
            description="Task 2",
            dependencies=["task1"],
        ))

        # Initially only task1 is ready
        ready = plan.get_ready_tasks([])
        assert len(ready) == 1
        assert ready[0].task_type == "task1"

        # After task1 completes, task2 is ready
        ready = plan.get_ready_tasks(["task1"])
        assert len(ready) == 1
        assert ready[0].task_type == "task2"

    def test_validate_valid_plan(self):
        """Test validation of a valid plan."""
        plan = WorkflowPlan()
        plan.add_task(TaskSpec(
            task_type="task1",
            assigned_role="role1",
            description="Task 1",
        ))
        plan.add_task(TaskSpec(
            task_type="task2",
            assigned_role="role2",
            description="Task 2",
            dependencies=["task1"],
        ))

        errors = plan.validate()
        assert errors == []

    def test_validate_missing_dependency(self):
        """Test validation catches missing dependencies."""
        plan = WorkflowPlan()
        plan.add_task(TaskSpec(
            task_type="task1",
            assigned_role="role1",
            description="Task 1",
            dependencies=["nonexistent"],
        ))

        errors = plan.validate()
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        plan = WorkflowPlan()
        plan.add_task(TaskSpec(
            task_type="task1",
            assigned_role="role1",
            description="Task 1",
        ))
        plan.success_criteria = ["All tests pass"]

        data = plan.to_dict()
        assert "tasks" in data
        assert len(data["tasks"]) == 1
        assert data["success_criteria"] == ["All tests pass"]


class TestStandardWorkflows:
    """Tests for pre-built workflow templates."""

    def test_standard_workflow_structure(self):
        """Test the standard workflow has all required phases."""
        plan = create_standard_workflow()

        roles = {t.assigned_role for t in plan.tasks}
        expected_roles = {
            "business_analyst", "project_manager", "ux_engineer",
            "tech_lead", "database_engineer", "backend_engineer",
            "frontend_engineer", "code_reviewer", "security_reviewer",
            "design_reviewer", "graphic_designer"  # Added design review agents
        }
        assert roles == expected_roles

    def test_standard_workflow_dependencies(self):
        """Test the standard workflow has correct dependency chain."""
        plan = create_standard_workflow()

        # BA should have no dependencies
        ba_task = plan.get_task_by_type("requirements_analysis")
        assert ba_task.dependencies == []

        # Security review should depend on code review
        sec_task = plan.get_task_by_type("security_review")
        assert "code_review" in sec_task.dependencies

    def test_standard_workflow_quality_gates(self):
        """Test quality gates are properly marked."""
        plan = create_standard_workflow()

        cr_task = plan.get_task_by_type("code_review")
        sec_task = plan.get_task_by_type("security_review")

        assert cr_task.is_gate is True
        assert sec_task.is_gate is True

    def test_minimal_workflow(self):
        """Test the minimal workflow structure."""
        plan = create_minimal_workflow()

        assert len(plan.tasks) == 3
        roles = {t.assigned_role for t in plan.tasks}
        assert "business_analyst" in roles
        assert "code_reviewer" in roles

    def test_frontend_workflow_structure(self):
        """Test the frontend workflow includes design reviews."""
        plan = create_frontend_workflow()

        roles = {t.assigned_role for t in plan.tasks}
        assert "ux_engineer" in roles
        assert "frontend_engineer" in roles
        assert "design_reviewer" in roles
        assert "graphic_designer" in roles
        assert "code_reviewer" in roles

    def test_frontend_workflow_design_gate(self):
        """Test the frontend workflow has visual review as a quality gate."""
        plan = create_frontend_workflow()

        final_review = plan.get_task_by_type("final_visual_review")
        assert final_review is not None
        assert final_review.is_gate is True
        assert final_review.assigned_role == "graphic_designer"

    def test_frontend_workflow_parallel_reviews(self):
        """Test that design_review and visual_review run in parallel."""
        plan = create_frontend_workflow()

        design_review = plan.get_task_by_type("design_review")
        visual_review = plan.get_task_by_type("visual_review")

        # Both should depend only on frontend_development
        assert design_review.dependencies == ["frontend_development"]
        assert visual_review.dependencies == ["frontend_development"]

        # Both should have same priority (run in parallel)
        assert design_review.priority == visual_review.priority

    def test_design_iteration_workflow(self):
        """Test the design iteration workflow for improving existing frontends."""
        plan = create_design_iteration_workflow()

        assert len(plan.tasks) == 4
        roles = {t.assigned_role for t in plan.tasks}
        assert "graphic_designer" in roles
        assert "design_reviewer" in roles
        assert "frontend_engineer" in roles

        # Final beauty check should be a gate
        final_check = plan.get_task_by_type("final_beauty_check")
        assert final_check.is_gate is True
