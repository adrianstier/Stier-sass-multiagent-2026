"""
Project Modes Handler

Adapts orchestration strategy based on project type:
- Greenfield: Full SDLC from requirements to deployment
- Existing Active: Feature additions with respect to existing patterns
- Legacy Maintenance: Careful changes with extensive testing
- Feature Branch: Focused work on specific functionality
- Bug Fix: Targeted investigation and minimal changes
- Refactor: Structural improvements while maintaining behavior
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from orchestrator.core.project_analyzer import (
    ProjectAnalyzer,
    ProjectState,
    ProjectType,
    analyze_project,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent in a specific mode."""
    enabled: bool = True
    priority: int = 1  # Lower = higher priority
    max_iterations: int = 10
    context_emphasis: list[str] = field(default_factory=list)
    skip_conditions: list[str] = field(default_factory=list)
    additional_instructions: str = ""


@dataclass
class WorkflowConfig:
    """Workflow configuration for a project mode."""
    mode: ProjectType
    description: str

    # Agent configurations
    agent_configs: dict[str, AgentConfig] = field(default_factory=dict)

    # Workflow settings
    require_full_analysis: bool = True
    require_planning_phase: bool = True
    require_design_phase: bool = True
    parallel_implementation: bool = True
    require_code_review: bool = True
    require_security_review: bool = True

    # Quality gates
    min_test_coverage: Optional[float] = None
    require_tests_pass: bool = True
    require_lint_pass: bool = True
    require_type_check: bool = False

    # Change scope
    max_files_changed: Optional[int] = None
    prefer_minimal_changes: bool = False
    require_backwards_compatibility: bool = False

    # Context settings
    analyze_existing_patterns: bool = False
    respect_existing_architecture: bool = False
    include_dependency_context: bool = False


# Pre-defined workflow configurations for each project mode
WORKFLOW_CONFIGS: dict[ProjectType, WorkflowConfig] = {
    ProjectType.GREENFIELD: WorkflowConfig(
        mode=ProjectType.GREENFIELD,
        description="New project from scratch - full SDLC workflow",
        require_full_analysis=False,  # Nothing to analyze yet
        require_planning_phase=True,
        require_design_phase=True,
        parallel_implementation=True,
        require_code_review=True,
        require_security_review=True,
        min_test_coverage=80.0,
        require_tests_pass=True,
        require_lint_pass=True,
        require_type_check=True,
        analyze_existing_patterns=False,
        respect_existing_architecture=False,
        agent_configs={
            "business_analyst": AgentConfig(
                enabled=True,
                priority=1,
                additional_instructions="Create comprehensive requirements from scratch. No existing codebase to consider."
            ),
            "project_manager": AgentConfig(
                enabled=True,
                priority=2,
                additional_instructions="Create full project plan including all phases."
            ),
            "ux_engineer": AgentConfig(
                enabled=True,
                priority=3,
                additional_instructions="Design user experience from scratch. Consider modern UX patterns."
            ),
            "tech_lead": AgentConfig(
                enabled=True,
                priority=4,
                additional_instructions="Design architecture from scratch. Choose appropriate patterns and technologies."
            ),
            "database_engineer": AgentConfig(
                enabled=True,
                priority=5,
                additional_instructions="Design database schema from requirements. No existing schema to consider."
            ),
            "backend_engineer": AgentConfig(
                enabled=True,
                priority=6,
                additional_instructions="Implement backend from architecture design. Establish coding patterns."
            ),
            "frontend_engineer": AgentConfig(
                enabled=True,
                priority=6,  # Same as backend - parallel
                additional_instructions="Implement frontend from UX designs. Establish component patterns."
            ),
            "code_reviewer": AgentConfig(
                enabled=True,
                priority=7,
                additional_instructions="Review for code quality and pattern consistency."
            ),
            "security_reviewer": AgentConfig(
                enabled=True,
                priority=8,
                additional_instructions="Full security review - establish security baseline."
            ),
        }
    ),

    ProjectType.EXISTING_ACTIVE: WorkflowConfig(
        mode=ProjectType.EXISTING_ACTIVE,
        description="Ongoing development - respect existing patterns",
        require_full_analysis=True,
        require_planning_phase=True,
        require_design_phase=True,
        parallel_implementation=True,
        require_code_review=True,
        require_security_review=True,
        require_tests_pass=True,
        require_lint_pass=True,
        analyze_existing_patterns=True,
        respect_existing_architecture=True,
        include_dependency_context=True,
        agent_configs={
            "business_analyst": AgentConfig(
                enabled=True,
                priority=1,
                context_emphasis=["existing_requirements", "current_features"],
                additional_instructions="Analyze how new requirements fit with existing features. Identify integration points."
            ),
            "project_manager": AgentConfig(
                enabled=True,
                priority=2,
                additional_instructions="Plan work considering existing codebase. Identify areas that need modification."
            ),
            "ux_engineer": AgentConfig(
                enabled=True,
                priority=3,
                additional_instructions="Design consistent with existing UI patterns. Maintain design system coherence."
            ),
            "tech_lead": AgentConfig(
                enabled=True,
                priority=4,
                context_emphasis=["architecture", "tech_stack", "api_endpoints_count"],
                additional_instructions="IMPORTANT: Analyze existing architecture first. Propose changes that fit existing patterns. Avoid unnecessary refactoring."
            ),
            "database_engineer": AgentConfig(
                enabled=True,
                priority=5,
                context_emphasis=["database_models"],
                additional_instructions="Review existing schema. Design migrations that preserve data. Maintain backwards compatibility."
            ),
            "backend_engineer": AgentConfig(
                enabled=True,
                priority=6,
                context_emphasis=["api_endpoints", "entry_points", "framework"],
                additional_instructions="FOLLOW existing code patterns. Use established utilities and helpers. Match existing style."
            ),
            "frontend_engineer": AgentConfig(
                enabled=True,
                priority=6,
                context_emphasis=["component_patterns", "framework"],
                additional_instructions="Use existing component library. Match established patterns. Maintain style consistency."
            ),
            "code_reviewer": AgentConfig(
                enabled=True,
                priority=7,
                context_emphasis=["quality_metrics", "common_issues"],
                additional_instructions="Check consistency with existing codebase. Flag deviations from established patterns."
            ),
            "security_reviewer": AgentConfig(
                enabled=True,
                priority=8,
                context_emphasis=["security_issues", "auth_patterns"],
                additional_instructions="Review in context of existing security model. Ensure changes don't weaken security."
            ),
        }
    ),

    ProjectType.LEGACY_MAINTENANCE: WorkflowConfig(
        mode=ProjectType.LEGACY_MAINTENANCE,
        description="Legacy codebase - careful, well-tested changes",
        require_full_analysis=True,
        require_planning_phase=True,
        require_design_phase=False,  # Usually not changing design
        parallel_implementation=False,  # Sequential for safety
        require_code_review=True,
        require_security_review=True,
        require_tests_pass=True,
        require_lint_pass=True,
        prefer_minimal_changes=True,
        require_backwards_compatibility=True,
        max_files_changed=20,
        analyze_existing_patterns=True,
        respect_existing_architecture=True,
        include_dependency_context=True,
        agent_configs={
            "business_analyst": AgentConfig(
                enabled=True,
                priority=1,
                additional_instructions="Document current behavior before proposing changes. Identify regression risks."
            ),
            "project_manager": AgentConfig(
                enabled=True,
                priority=2,
                additional_instructions="Plan conservative changes. Include extensive testing phases. Plan for rollback."
            ),
            "ux_engineer": AgentConfig(
                enabled=False,  # Usually not changing UX in maintenance
                skip_conditions=["no_ui_changes"]
            ),
            "tech_lead": AgentConfig(
                enabled=True,
                priority=3,
                additional_instructions="MINIMIZE architectural changes. Document why changes are necessary. Prefer refactoring in isolation."
            ),
            "database_engineer": AgentConfig(
                enabled=True,
                priority=4,
                additional_instructions="EXTREME CAUTION with schema changes. Always reversible migrations. Backup strategies required."
            ),
            "backend_engineer": AgentConfig(
                enabled=True,
                priority=5,
                max_iterations=15,  # More iterations for careful work
                additional_instructions="Make MINIMAL changes. Add tests BEFORE changing code. Preserve all existing behavior unless explicitly changing."
            ),
            "frontend_engineer": AgentConfig(
                enabled=True,
                priority=5,
                additional_instructions="Minimal UI changes. Test extensively in existing context."
            ),
            "code_reviewer": AgentConfig(
                enabled=True,
                priority=6,
                additional_instructions="STRICT review. Check for regressions. Verify backwards compatibility. Question any large changes."
            ),
            "security_reviewer": AgentConfig(
                enabled=True,
                priority=7,
                additional_instructions="Review for security regressions. Check that fixes don't introduce new vulnerabilities."
            ),
        }
    ),

    ProjectType.FEATURE_BRANCH: WorkflowConfig(
        mode=ProjectType.FEATURE_BRANCH,
        description="Feature development on a branch",
        require_full_analysis=True,
        require_planning_phase=True,
        require_design_phase=True,
        parallel_implementation=True,
        require_code_review=True,
        require_security_review=True,
        require_tests_pass=True,
        analyze_existing_patterns=True,
        respect_existing_architecture=True,
        agent_configs={
            "business_analyst": AgentConfig(
                enabled=True,
                priority=1,
                additional_instructions="Focus on the specific feature requirements. Understand feature boundaries."
            ),
            "project_manager": AgentConfig(
                enabled=True,
                priority=2,
                additional_instructions="Plan for feature scope. Identify dependencies on other features."
            ),
            "ux_engineer": AgentConfig(
                enabled=True,
                priority=3,
                additional_instructions="Design for the specific feature. Ensure consistency with existing UX."
            ),
            "tech_lead": AgentConfig(
                enabled=True,
                priority=4,
                additional_instructions="Design feature architecture within existing structure. Define integration points."
            ),
            "database_engineer": AgentConfig(
                enabled=True,
                priority=5,
                additional_instructions="Design schema additions for feature. Plan feature-specific migrations."
            ),
            "backend_engineer": AgentConfig(
                enabled=True,
                priority=6,
                additional_instructions="Implement feature following existing patterns. Add comprehensive tests."
            ),
            "frontend_engineer": AgentConfig(
                enabled=True,
                priority=6,
                additional_instructions="Implement feature UI. Integrate with existing components."
            ),
            "code_reviewer": AgentConfig(
                enabled=True,
                priority=7,
                additional_instructions="Review feature implementation. Check integration with existing code."
            ),
            "security_reviewer": AgentConfig(
                enabled=True,
                priority=8,
                additional_instructions="Review feature for security. Check for feature-specific vulnerabilities."
            ),
        }
    ),

    ProjectType.BUG_FIX: WorkflowConfig(
        mode=ProjectType.BUG_FIX,
        description="Bug investigation and fix",
        require_full_analysis=True,
        require_planning_phase=False,  # Jump to investigation
        require_design_phase=False,
        parallel_implementation=False,
        require_code_review=True,
        require_security_review=False,  # Unless security bug
        require_tests_pass=True,
        prefer_minimal_changes=True,
        max_files_changed=10,
        analyze_existing_patterns=True,
        agent_configs={
            "business_analyst": AgentConfig(
                enabled=True,
                priority=1,
                additional_instructions="Document the bug: symptoms, reproduction steps, expected vs actual behavior."
            ),
            "project_manager": AgentConfig(
                enabled=False,  # Skip planning for bug fixes
            ),
            "ux_engineer": AgentConfig(
                enabled=False,  # Usually not needed
            ),
            "tech_lead": AgentConfig(
                enabled=True,
                priority=2,
                additional_instructions="Investigate root cause. Identify the minimal fix. Consider side effects."
            ),
            "database_engineer": AgentConfig(
                enabled=False,  # Enable if data bug
                skip_conditions=["not_data_related"]
            ),
            "backend_engineer": AgentConfig(
                enabled=True,
                priority=3,
                max_iterations=15,
                additional_instructions="Fix the bug with MINIMAL changes. Add regression test FIRST. Verify fix doesn't break other functionality."
            ),
            "frontend_engineer": AgentConfig(
                enabled=True,
                priority=3,
                additional_instructions="Fix UI bugs minimally. Add test to prevent regression."
            ),
            "code_reviewer": AgentConfig(
                enabled=True,
                priority=4,
                additional_instructions="Verify fix addresses root cause. Check for side effects. Ensure regression test covers the bug."
            ),
            "security_reviewer": AgentConfig(
                enabled=False,  # Enable for security bugs
                skip_conditions=["not_security_bug"]
            ),
        }
    ),

    ProjectType.REFACTOR: WorkflowConfig(
        mode=ProjectType.REFACTOR,
        description="Code improvement without behavior change",
        require_full_analysis=True,
        require_planning_phase=True,
        require_design_phase=True,
        parallel_implementation=False,  # Sequential for safety
        require_code_review=True,
        require_security_review=True,
        require_tests_pass=True,
        require_lint_pass=True,
        require_type_check=True,
        require_backwards_compatibility=True,
        analyze_existing_patterns=True,
        respect_existing_architecture=True,
        agent_configs={
            "business_analyst": AgentConfig(
                enabled=True,
                priority=1,
                additional_instructions="Document current behavior that MUST be preserved. Define refactor scope."
            ),
            "project_manager": AgentConfig(
                enabled=True,
                priority=2,
                additional_instructions="Plan incremental refactoring steps. Each step must leave code working."
            ),
            "ux_engineer": AgentConfig(
                enabled=False,  # No UX changes
            ),
            "tech_lead": AgentConfig(
                enabled=True,
                priority=3,
                additional_instructions="Design target architecture. Plan migration path. Define success criteria."
            ),
            "database_engineer": AgentConfig(
                enabled=True,
                priority=4,
                additional_instructions="Plan schema refactoring if needed. Zero data loss tolerance."
            ),
            "backend_engineer": AgentConfig(
                enabled=True,
                priority=5,
                max_iterations=20,
                additional_instructions="Refactor in small steps. Run tests after EACH change. Preserve all behavior."
            ),
            "frontend_engineer": AgentConfig(
                enabled=True,
                priority=5,
                additional_instructions="Refactor components. Maintain identical behavior. Improve code quality."
            ),
            "code_reviewer": AgentConfig(
                enabled=True,
                priority=6,
                additional_instructions="Verify behavior preservation. Check code quality improvements. Ensure no regressions."
            ),
            "security_reviewer": AgentConfig(
                enabled=True,
                priority=7,
                additional_instructions="Verify refactoring doesn't introduce security issues. Check for exposed internals."
            ),
        }
    ),
}


class ProjectModeHandler:
    """
    Handles project mode detection and workflow configuration.

    Provides context-aware orchestration based on whether we're:
    - Starting from scratch (greenfield)
    - Working on an existing active project
    - Maintaining legacy code
    - Developing a specific feature
    - Fixing bugs
    - Refactoring
    """

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.analyzer: Optional[ProjectAnalyzer] = None
        self.project_state: Optional[ProjectState] = None
        self.workflow_config: Optional[WorkflowConfig] = None

    async def initialize(self, force_mode: Optional[ProjectType] = None) -> WorkflowConfig:
        """
        Initialize the handler by analyzing the project.

        Args:
            force_mode: Optionally force a specific project mode
        """
        # Analyze the project
        self.project_state = await analyze_project(self.project_path)

        # Use forced mode or detected mode
        mode = force_mode or self.project_state.project_type

        # Get workflow config for this mode
        self.workflow_config = WORKFLOW_CONFIGS.get(mode, WORKFLOW_CONFIGS[ProjectType.EXISTING_ACTIVE])

        logger.info(f"Project mode: {mode.value} - {self.workflow_config.description}")

        return self.workflow_config

    def get_agent_instructions(self, agent_role: str) -> dict[str, Any]:
        """
        Get instructions for a specific agent based on project mode.

        Returns context and instructions tailored to the project state.
        """
        if not self.workflow_config or not self.project_state:
            raise ValueError("Must call initialize() first")

        config = self.workflow_config.agent_configs.get(agent_role, AgentConfig())

        # Build context from project state
        context = {}
        if self.project_state.agent_context.get(agent_role):
            agent_context = self.project_state.agent_context[agent_role]

            # Add emphasized context fields
            for field in config.context_emphasis:
                if field in agent_context:
                    context[field] = agent_context[field]

            # Add all context if no specific emphasis
            if not config.context_emphasis:
                context = agent_context

        return {
            "enabled": config.enabled,
            "priority": config.priority,
            "max_iterations": config.max_iterations,
            "additional_instructions": config.additional_instructions,
            "project_context": context,
            "project_type": self.project_state.project_type.value,
            "workflow_settings": {
                "respect_existing_architecture": self.workflow_config.respect_existing_architecture,
                "prefer_minimal_changes": self.workflow_config.prefer_minimal_changes,
                "require_backwards_compatibility": self.workflow_config.require_backwards_compatibility,
            },
        }

    def get_enabled_agents(self) -> list[str]:
        """Get list of enabled agents for this project mode."""
        if not self.workflow_config:
            raise ValueError("Must call initialize() first")

        enabled = []
        for role, config in self.workflow_config.agent_configs.items():
            if config.enabled:
                enabled.append(role)

        return sorted(enabled, key=lambda r: self.workflow_config.agent_configs[r].priority)

    def get_quality_requirements(self) -> dict[str, Any]:
        """Get quality gate requirements for this project mode."""
        if not self.workflow_config:
            raise ValueError("Must call initialize() first")

        return {
            "require_code_review": self.workflow_config.require_code_review,
            "require_security_review": self.workflow_config.require_security_review,
            "require_tests_pass": self.workflow_config.require_tests_pass,
            "require_lint_pass": self.workflow_config.require_lint_pass,
            "require_type_check": self.workflow_config.require_type_check,
            "min_test_coverage": self.workflow_config.min_test_coverage,
            "max_files_changed": self.workflow_config.max_files_changed,
        }

    def should_skip_agent(self, agent_role: str, task_context: dict) -> tuple[bool, str]:
        """
        Check if an agent should be skipped based on skip conditions.

        Returns (should_skip, reason).
        """
        if not self.workflow_config:
            return False, ""

        config = self.workflow_config.agent_configs.get(agent_role)
        if not config:
            return False, ""

        if not config.enabled:
            return True, "Agent disabled for this project mode"

        for condition in config.skip_conditions:
            if self._evaluate_skip_condition(condition, task_context):
                return True, f"Skip condition met: {condition}"

        return False, ""

    def _evaluate_skip_condition(self, condition: str, context: dict) -> bool:
        """Evaluate a skip condition against the context."""
        # Simple condition evaluation
        if condition == "no_ui_changes":
            return context.get("ui_changes", True) is False
        if condition == "not_data_related":
            return context.get("data_related", True) is False
        if condition == "not_security_bug":
            return context.get("security_related", False) is False
        return False

    def generate_system_prompt_additions(self, agent_role: str) -> str:
        """
        Generate additions to the agent's system prompt based on project state.
        """
        if not self.project_state:
            return ""

        additions = []

        # Add project type context
        additions.append(f"\n## Project Context")
        additions.append(f"Project Type: {self.project_state.project_type.value}")
        additions.append(f"Primary Language: {self.project_state.primary_language or 'Unknown'}")
        additions.append(f"Framework: {self.project_state.framework.value}")

        # Add mode-specific instructions
        config = self.workflow_config.agent_configs.get(agent_role) if self.workflow_config else None
        if config and config.additional_instructions:
            additions.append(f"\n## Mode-Specific Instructions")
            additions.append(config.additional_instructions)

        # Add existing patterns context
        if self.workflow_config and self.workflow_config.respect_existing_architecture:
            additions.append(f"\n## Existing Architecture")
            patterns = [p.pattern_name for p in self.project_state.architecture_patterns]
            if patterns:
                additions.append(f"Detected patterns: {', '.join(patterns)}")
            additions.append("You MUST respect existing architectural patterns unless explicitly told otherwise.")

        # Add constraint context
        if self.workflow_config and self.workflow_config.prefer_minimal_changes:
            additions.append(f"\n## Change Constraints")
            additions.append("Prefer MINIMAL changes. Only modify what is strictly necessary.")
            if self.workflow_config.max_files_changed:
                additions.append(f"Maximum files to change: {self.workflow_config.max_files_changed}")

        return "\n".join(additions)


# Convenience functions
async def get_project_mode_handler(project_path: str) -> ProjectModeHandler:
    """Get an initialized project mode handler."""
    handler = ProjectModeHandler(project_path)
    await handler.initialize()
    return handler


def get_workflow_config(mode: ProjectType) -> WorkflowConfig:
    """Get workflow configuration for a specific mode."""
    return WORKFLOW_CONFIGS.get(mode, WORKFLOW_CONFIGS[ProjectType.EXISTING_ACTIVE])
