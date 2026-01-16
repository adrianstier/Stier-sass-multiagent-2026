"""Base agent class with LLM integration and tool execution."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from anthropic import Anthropic

from orchestrator.core.config import settings
from orchestrator.core.database import get_db
from orchestrator.core.models import Run, Task, Event, Artifact, TaskStatus
from orchestrator.core.logging import get_logger
from orchestrator.core.rate_limit import (
    get_budget_enforcer,
    TokenBudgetError,
    calculate_cost,
)
from orchestrator.tools import ToolExecutor, ToolResult

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all specialized agents."""

    role: str = "base"
    role_description: str = "Base agent"

    def __init__(self, run_id: str, task_id: str):
        self.run_id = run_id
        self.task_id = task_id
        self.tool_executor = ToolExecutor(self.role, run_id, task_id)
        self.client = Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent role."""
        pass

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task."""
        logger.info(f"Agent '{self.role}' executing task {self.task_id}")

        # Record task start
        self._record_event("task_started", {"input": task_input})

        try:
            # Build the prompt
            system_prompt = self.get_system_prompt()
            user_prompt = self._build_user_prompt(task_input)

            # Get context from previous artifacts
            context = self._get_context_artifacts()

            # Execute LLM call
            response = self._call_llm(system_prompt, user_prompt, context)

            # Parse and execute any tool calls
            artifacts = self._process_response(response)

            # Record completion
            self._record_event("task_completed", {
                "artifacts_created": len(artifacts),
                "response_length": len(response),
            })

            return {
                "success": True,
                "response": response,
                "artifacts": artifacts,
                "tool_log": self.tool_executor.get_execution_log(),
            }

        except Exception as e:
            logger.error(f"Agent '{self.role}' failed: {e}")
            self._record_event("task_failed", {"error": str(e)})
            return {
                "success": False,
                "error": str(e),
                "tool_log": self.tool_executor.get_execution_log(),
            }

    def _build_user_prompt(self, task_input: Dict[str, Any]) -> str:
        """Build the user prompt from task input."""
        goal = task_input.get("goal", "")
        description = task_input.get("description", "")
        dependencies_output = task_input.get("dependencies_output", {})

        prompt_parts = [
            f"## Goal\n{goal}",
            f"\n## Task Description\n{description}",
        ]

        if dependencies_output:
            prompt_parts.append("\n## Input from Previous Phases")
            for dep_name, dep_output in dependencies_output.items():
                prompt_parts.append(f"\n### {dep_name}\n{dep_output}")

        prompt_parts.append(
            "\n## Instructions\n"
            "Complete your assigned task and create the required artifacts. "
            "Be thorough but concise. Structure your output clearly."
        )

        return "\n".join(prompt_parts)

    def _get_context_artifacts(self) -> List[Dict[str, Any]]:
        """Get relevant artifacts from previous tasks for context."""
        with get_db() as db:
            artifacts = db.query(Artifact).filter(
                Artifact.run_id == UUID(self.run_id)
            ).order_by(Artifact.created_at.desc()).limit(10).all()

            return [
                {
                    "type": a.artifact_type,
                    "name": a.name,
                    "summary": a.content[:500] if a.content else "",
                }
                for a in artifacts
            ]

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        context: List[Dict[str, Any]],
    ) -> str:
        """Call the LLM with the given prompts."""
        if not self.client:
            # Mock response for testing without API key
            logger.warning("No Anthropic API key - returning mock response")
            return self._generate_mock_response(user_prompt)

        # Build context message
        context_str = ""
        if context:
            context_str = "\n\n## Previous Artifacts (for reference)\n"
            for ctx in context:
                context_str += f"- {ctx['type']}: {ctx['name']}\n"

        full_user_prompt = user_prompt + context_str

        # Check token budget BEFORE making LLM call
        estimated_tokens = self._estimate_tokens(system_prompt, full_user_prompt)
        self._check_budget(estimated_tokens)

        # Record the LLM request
        self._record_event("llm_request", {
            "model": settings.llm_model,
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(full_user_prompt),
            "estimated_tokens": estimated_tokens,
        })

        # Make the API call
        message = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": full_user_prompt}
            ],
        )

        response_text = message.content[0].text
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = calculate_cost(settings.llm_model, input_tokens, output_tokens)

        # Update run token usage and cost
        self._update_usage(total_tokens, cost)

        # Record the response
        self._record_event("llm_response", {
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "response_length": len(response_text),
        })

        return response_text

    def _estimate_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Estimate tokens for a request (rough approximation)."""
        # Rough estimate: ~4 chars per token for English text
        total_chars = len(system_prompt) + len(user_prompt)
        estimated_input = total_chars // 4
        estimated_output = settings.llm_max_tokens  # Assume worst case
        return estimated_input + estimated_output

    def _check_budget(self, estimated_tokens: int) -> None:
        """Check if request is within budget limits."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if not run:
                raise ValueError(f"Run not found: {self.run_id}")

            try:
                enforcer = get_budget_enforcer()
                enforcer.check_budget(
                    org_id=str(run.organization_id),
                    run_budget=run.budget_tokens,
                    run_used=run.tokens_used,
                    tokens_needed=estimated_tokens,
                )
            except TokenBudgetError as e:
                logger.error(
                    "token_budget_exceeded",
                    run_id=self.run_id,
                    used=e.used,
                    limit=e.limit,
                )
                self._record_event("budget_exceeded", {
                    "used": e.used,
                    "limit": e.limit,
                    "error": str(e),
                })
                raise

    def _update_usage(self, tokens: int, cost: float) -> None:
        """Update run's token usage and cost."""
        with get_db() as db:
            run = db.query(Run).filter(Run.id == UUID(self.run_id)).first()
            if run:
                run.tokens_used = (run.tokens_used or 0) + tokens
                current_cost = float(run.total_cost_usd or "0")
                run.total_cost_usd = str(round(current_cost + cost, 6))
                db.commit()

                # Also update organization usage
                try:
                    enforcer = get_budget_enforcer()
                    enforcer.record_usage(str(run.organization_id), tokens)
                except Exception as e:
                    logger.warning(f"Failed to record org usage: {e}")

    def _generate_mock_response(self, user_prompt: str) -> str:
        """Generate a mock response for testing."""
        return f"""# {self.role.replace('_', ' ').title()} Output

## Analysis Complete

This is a mock response for testing purposes.

### Summary
- Task received and analyzed
- Mock artifacts will be created
- Ready for next phase

### Deliverables
The required artifacts have been prepared based on the input requirements.
"""

    def _process_response(self, response: str) -> List[Dict[str, Any]]:
        """Process the LLM response and create artifacts."""
        artifacts = []

        # Create the main output artifact
        artifact_type = self._get_artifact_type()
        artifact_name = f"{self.role}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        result = self.tool_executor.execute(
            "create_artifact",
            run_id=self.run_id,
            task_id=self.task_id,
            artifact_type=artifact_type,
            name=artifact_name,
            content=response,
            content_type="text/markdown",
            metadata={"role": self.role},
        )

        if result.success:
            artifacts.append(result.result)

        return artifacts

    def _get_artifact_type(self) -> str:
        """Get the artifact type for this agent's output."""
        artifact_types = {
            "business_analyst": "requirements_document",
            "project_manager": "project_plan",
            "ux_engineer": "design_spec",
            "tech_lead": "architecture_document",
            "database_engineer": "database_schema",
            "backend_engineer": "backend_code",
            "frontend_engineer": "frontend_code",
            "code_reviewer": "code_review_report",
            "security_reviewer": "security_review_report",
            "cleanup_agent": "cleanup_report",
            "data_scientist": "data_analysis",
            "design_reviewer": "design_review_report",
        }
        return artifact_types.get(self.role, "general_artifact")

    def _record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an event for traceability."""
        with get_db() as db:
            event = Event(
                run_id=UUID(self.run_id),
                task_id=UUID(self.task_id),
                event_type=event_type,
                actor=self.role,
                data=data,
            )
            db.add(event)
            db.commit()
