"""Task DSL for specifying agent tasks with dependencies and validation.

Provides a lightweight DSL for defining:
- Task type and assigned role
- Dependencies on other tasks
- Expected artifacts
- Validation methods
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import hashlib
import json


class ValidationMethod(str, Enum):
    """Built-in validation methods for task outputs."""
    NONE = "none"
    SCHEMA = "schema"  # JSON schema validation
    ARTIFACT_EXISTS = "artifact_exists"  # Just check artifact was produced
    CUSTOM = "custom"  # Custom validation function
    GATE_APPROVAL = "gate_approval"  # Requires explicit gate approval


@dataclass
class TaskSpec:
    """Specification for a single task in the workflow."""

    task_type: str
    assigned_role: str
    description: str

    # Dependencies - list of task_type names this task depends on
    dependencies: List[str] = field(default_factory=list)

    # Expected outputs
    expected_artifacts: List[str] = field(default_factory=list)

    # Input requirements
    required_inputs: List[str] = field(default_factory=list)

    # Validation
    validation_method: ValidationMethod = ValidationMethod.ARTIFACT_EXISTS
    validation_schema: Optional[Dict[str, Any]] = None

    # Execution settings
    priority: int = 0
    timeout_seconds: int = 300
    max_retries: int = 3

    # Is this a quality gate?
    is_gate: bool = False
    gate_blocks: List[str] = field(default_factory=list)  # Tasks blocked by this gate

    def generate_idempotency_key(self, run_id: str, input_hash: str = "") -> str:
        """Generate a unique idempotency key for this task."""
        key_data = f"{run_id}:{self.task_type}:{self.assigned_role}:{input_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]


@dataclass
class WorkflowPlan:
    """A complete workflow plan with ordered tasks."""

    tasks: List[TaskSpec] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)

    def add_task(self, task: TaskSpec) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)

    def get_task_by_type(self, task_type: str) -> Optional[TaskSpec]:
        """Get a task by its type."""
        for task in self.tasks:
            if task.task_type == task_type:
                return task
        return None

    def get_ready_tasks(self, completed_tasks: List[str]) -> List[TaskSpec]:
        """Get tasks that are ready to execute (all dependencies met)."""
        ready = []
        for task in self.tasks:
            if task.task_type in completed_tasks:
                continue
            if all(dep in completed_tasks for dep in task.dependencies):
                ready.append(task)
        return ready

    def validate(self) -> List[str]:
        """Validate the workflow plan. Returns list of errors."""
        errors = []

        # Check for missing dependencies
        task_types = {t.task_type for t in self.tasks}
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in task_types:
                    errors.append(f"Task '{task.task_type}' depends on unknown task '{dep}'")

        # Check for circular dependencies
        def has_cycle(task_type: str, visited: set, path: set) -> bool:
            if task_type in path:
                return True
            if task_type in visited:
                return False
            visited.add(task_type)
            path.add(task_type)
            task = self.get_task_by_type(task_type)
            if task:
                for dep in task.dependencies:
                    if has_cycle(dep, visited, path):
                        return True
            path.remove(task_type)
            return False

        for task in self.tasks:
            if has_cycle(task.task_type, set(), set()):
                errors.append(f"Circular dependency detected involving '{task.task_type}'")
                break

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tasks": [
                {
                    "task_type": t.task_type,
                    "assigned_role": t.assigned_role,
                    "description": t.description,
                    "dependencies": t.dependencies,
                    "expected_artifacts": t.expected_artifacts,
                    "validation_method": t.validation_method.value,
                    "priority": t.priority,
                    "is_gate": t.is_gate,
                }
                for t in self.tasks
            ],
            "success_criteria": self.success_criteria,
            "acceptance_criteria": self.acceptance_criteria,
        }


# Standard workflow templates

def create_standard_workflow() -> WorkflowPlan:
    """Create the standard 9-phase development workflow."""
    plan = WorkflowPlan()

    # Phase 1: Business Analysis
    plan.add_task(TaskSpec(
        task_type="requirements_analysis",
        assigned_role="business_analyst",
        description="Analyze business requirements, identify stakeholders, define success criteria",
        expected_artifacts=["requirements_document", "stakeholder_analysis"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Project Planning
    plan.add_task(TaskSpec(
        task_type="project_planning",
        assigned_role="project_manager",
        description="Create project plan, timeline, resource allocation, and risk assessment",
        dependencies=["requirements_analysis"],
        expected_artifacts=["project_plan", "risk_assessment"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=90,
    ))

    # Phase 3: UX Design
    plan.add_task(TaskSpec(
        task_type="ux_design",
        assigned_role="ux_engineer",
        description="Create user flows, wireframes, and design specifications",
        dependencies=["project_planning"],
        expected_artifacts=["user_flows", "wireframes", "design_spec"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    # Phase 4: Technical Architecture
    plan.add_task(TaskSpec(
        task_type="technical_architecture",
        assigned_role="tech_lead",
        description="Design system architecture, tech stack, and implementation guidelines",
        dependencies=["ux_design"],
        expected_artifacts=["architecture_document", "tech_stack_spec", "implementation_guidelines"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=70,
    ))

    # Phase 5: Database Design
    plan.add_task(TaskSpec(
        task_type="database_design",
        assigned_role="database_engineer",
        description="Design database schema, queries, and data access patterns",
        dependencies=["technical_architecture"],
        expected_artifacts=["database_schema", "migration_scripts"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=60,
    ))

    # Phase 6a: Backend Development
    plan.add_task(TaskSpec(
        task_type="backend_development",
        assigned_role="backend_engineer",
        description="Implement API endpoints, business logic, and database integration",
        dependencies=["technical_architecture", "database_design"],
        expected_artifacts=["api_implementation", "business_logic"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=50,
    ))

    # Phase 6b: Frontend Development
    plan.add_task(TaskSpec(
        task_type="frontend_development",
        assigned_role="frontend_engineer",
        description="Implement UI components, state management, and API integration",
        dependencies=["technical_architecture", "ux_design"],
        expected_artifacts=["frontend_components", "ui_implementation"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=50,
    ))

    # Phase 7a: Design Pattern Review
    plan.add_task(TaskSpec(
        task_type="design_review",
        assigned_role="design_reviewer",
        description="Review frontend implementation for design patterns, typography, color, motion, and distinctiveness",
        dependencies=["frontend_development"],
        expected_artifacts=["design_review_report"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=45,
    ))

    # Phase 7b: Visual Beauty Assessment
    plan.add_task(TaskSpec(
        task_type="visual_review",
        assigned_role="graphic_designer",
        description="Evaluate visual beauty, emotional impact, and aesthetic quality of the frontend implementation",
        dependencies=["frontend_development"],
        expected_artifacts=["visual_review_report", "beauty_score"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=45,
    ))

    # Phase 8: Code Review (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="code_review",
        assigned_role="code_reviewer",
        description="Review all code for quality, standards compliance, and best practices",
        dependencies=["backend_development", "frontend_development", "design_review", "visual_review"],
        expected_artifacts=["code_review_report"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        gate_blocks=["security_review"],
        priority=40,
    ))

    # Phase 9: Security Review (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="security_review",
        assigned_role="security_reviewer",
        description="Conduct security assessment, vulnerability testing, and compliance validation",
        dependencies=["code_review"],
        expected_artifacts=["security_review_report"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        priority=30,
    ))

    return plan


def create_minimal_workflow() -> WorkflowPlan:
    """Create a minimal workflow for simple tasks."""
    plan = WorkflowPlan()

    plan.add_task(TaskSpec(
        task_type="quick_analysis",
        assigned_role="business_analyst",
        description="Quick requirements analysis",
        expected_artifacts=["requirements_summary"],
        priority=100,
    ))

    plan.add_task(TaskSpec(
        task_type="implementation",
        assigned_role="backend_engineer",
        description="Implement the solution",
        dependencies=["quick_analysis"],
        expected_artifacts=["implementation"],
        priority=50,
    ))

    plan.add_task(TaskSpec(
        task_type="review",
        assigned_role="code_reviewer",
        description="Review the implementation",
        dependencies=["implementation"],
        expected_artifacts=["review_report"],
        is_gate=True,
        priority=30,
    ))

    return plan


def create_frontend_workflow() -> WorkflowPlan:
    """Create a frontend-focused workflow with design quality emphasis.

    This workflow prioritizes visual excellence with multiple design review stages:
    1. UX/wireframe design
    2. Frontend implementation
    3. Design pattern review (technical design patterns)
    4. Visual beauty assessment (aesthetic quality)
    5. Iteration based on feedback

    The graphic_designer and design_reviewer run in parallel after frontend
    development, with their feedback feeding into potential iterations.
    """
    plan = WorkflowPlan()

    # Phase 1: UX Design
    plan.add_task(TaskSpec(
        task_type="ux_design",
        assigned_role="ux_engineer",
        description="Create user flows, wireframes, interaction patterns, and design specifications",
        expected_artifacts=["user_flows", "wireframes", "design_spec", "accessibility_requirements"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Frontend Implementation
    plan.add_task(TaskSpec(
        task_type="frontend_development",
        assigned_role="frontend_engineer",
        description="Implement distinctive UI with bold aesthetics, curated typography, and purposeful motion",
        dependencies=["ux_design"],
        expected_artifacts=["frontend_components", "ui_implementation", "style_system"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    # Phase 3a: Design Pattern Review (runs in parallel with 3b)
    plan.add_task(TaskSpec(
        task_type="design_review",
        assigned_role="design_reviewer",
        description="Review implementation for design patterns, typography choices, color consistency, motion, and distinctiveness",
        dependencies=["frontend_development"],
        expected_artifacts=["design_review_report"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=60,
    ))

    # Phase 3b: Visual Beauty Assessment (runs in parallel with 3a)
    plan.add_task(TaskSpec(
        task_type="visual_review",
        assigned_role="graphic_designer",
        description="Evaluate visual beauty, emotional impact, typography refinement, color harmony, and overall aesthetic quality",
        dependencies=["frontend_development"],
        expected_artifacts=["visual_review_report", "beauty_score", "improvement_recommendations"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=60,
    ))

    # Phase 4: Design Iteration (if needed based on reviews)
    plan.add_task(TaskSpec(
        task_type="design_iteration",
        assigned_role="frontend_engineer",
        description="Implement design improvements based on design_review and visual_review feedback",
        dependencies=["design_review", "visual_review"],
        expected_artifacts=["updated_frontend_components", "iteration_changelog"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=50,
    ))

    # Phase 5: Final Beauty Check (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="final_visual_review",
        assigned_role="graphic_designer",
        description="Final beauty assessment - verify improvements meet aesthetic standards (Beauty Score >= 7/10)",
        dependencies=["design_iteration"],
        expected_artifacts=["final_beauty_report", "final_beauty_score"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        gate_blocks=["code_review"],
        priority=45,
    ))

    # Phase 6: Code Review (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="code_review",
        assigned_role="code_reviewer",
        description="Review code quality, accessibility, performance, and best practices",
        dependencies=["final_visual_review"],
        expected_artifacts=["code_review_report"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        priority=40,
    ))

    return plan


def create_design_iteration_workflow() -> WorkflowPlan:
    """Create a quick design iteration workflow for improving existing frontends.

    Use this when the frontend already exists but needs aesthetic improvements.
    Focuses on: graphic_designer assessment → frontend fixes → re-assessment.
    """
    plan = WorkflowPlan()

    # Phase 1: Initial Beauty Assessment
    plan.add_task(TaskSpec(
        task_type="initial_visual_review",
        assigned_role="graphic_designer",
        description="Evaluate current frontend for beauty, identify specific improvements needed",
        expected_artifacts=["visual_review_report", "beauty_score", "prioritized_improvements"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 2: Design Pattern Analysis
    plan.add_task(TaskSpec(
        task_type="design_analysis",
        assigned_role="design_reviewer",
        description="Analyze design patterns, typography, color usage, and motion implementation",
        expected_artifacts=["design_analysis_report", "pattern_recommendations"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=100,
    ))

    # Phase 3: Implement Improvements
    plan.add_task(TaskSpec(
        task_type="design_improvements",
        assigned_role="frontend_engineer",
        description="Implement prioritized design improvements from visual and pattern reviews",
        dependencies=["initial_visual_review", "design_analysis"],
        expected_artifacts=["updated_components", "improvement_changelog"],
        validation_method=ValidationMethod.ARTIFACT_EXISTS,
        priority=80,
    ))

    # Phase 4: Final Beauty Check (Quality Gate)
    plan.add_task(TaskSpec(
        task_type="final_beauty_check",
        assigned_role="graphic_designer",
        description="Final beauty assessment - verify Beauty Score >= 7/10, emotional impact achieved",
        dependencies=["design_improvements"],
        expected_artifacts=["final_beauty_report", "final_beauty_score"],
        validation_method=ValidationMethod.GATE_APPROVAL,
        is_gate=True,
        priority=60,
    ))

    return plan
