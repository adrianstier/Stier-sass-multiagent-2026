"""Base agent class with LLM integration and tool execution."""

import json
import re
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


# =============================================================================
# Tool Definitions for Claude API
# =============================================================================

AGENT_TOOLS = [
    {
        "name": "create_artifact",
        "description": "Create a work artifact (document, code, schema, etc.) that will be stored and passed to dependent tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "description": "Type of artifact: requirements_document, project_plan, design_spec, architecture_document, database_schema, backend_code, frontend_code, code_review_report, security_review_report, etc."
                },
                "name": {
                    "type": "string",
                    "description": "Short descriptive name for the artifact"
                },
                "content": {
                    "type": "string",
                    "description": "The full content of the artifact"
                },
                "content_type": {
                    "type": "string",
                    "description": "MIME type of content (default: text/markdown)",
                    "default": "text/markdown"
                },
            },
            "required": ["artifact_type", "name", "content"]
        }
    },
    {
        "name": "read_artifact",
        "description": "Read the full content of a previously created artifact by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "UUID of the artifact to read"
                }
            },
            "required": ["artifact_id"]
        }
    },
    {
        "name": "list_artifacts",
        "description": "List all artifacts created in this run, optionally filtered by type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "description": "Optional: filter by artifact type"
                }
            }
        }
    },
    {
        "name": "request_clarification",
        "description": "Request clarification or additional input from the user/orchestrator when requirements are ambiguous.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarification question to ask"
                },
                "context": {
                    "type": "string",
                    "description": "Context about why this clarification is needed"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: suggested options/choices"
                }
            },
            "required": ["question"]
        }
    }
]


class BaseAgent(ABC):
    """Base class for all specialized agents.

    Supports:
    - LLM calls with tool use (Claude API native tools)
    - Agentic loops with multiple tool calls
    - Token budget enforcement
    - Artifact creation and management
    - Event recording for full traceability
    """

    role: str = "base"
    role_description: str = "Base agent"
    max_tool_iterations: int = 10  # Max agentic loop iterations

    def __init__(self, run_id: str, task_id: str):
        self.run_id = run_id
        self.task_id = task_id
        self.tool_executor = ToolExecutor(self.role, run_id, task_id)
        self.client = Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
        self._tool_calls: List[Dict[str, Any]] = []
        self._clarifications_requested: List[Dict[str, Any]] = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent role."""
        pass

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to this agent.

        Override in subclasses to customize available tools.
        """
        return AGENT_TOOLS

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task with full tool support."""
        logger.info(f"Agent '{self.role}' executing task {self.task_id}")

        # Record task start
        self._record_event("task_started", {"input": task_input})

        try:
            # Build the prompt
            system_prompt = self.get_system_prompt()
            user_prompt = self._build_user_prompt(task_input)

            # Get context from previous artifacts
            context = self._get_context_artifacts()

            # Execute LLM call with tool use
            response, artifacts = self._call_llm_with_tools(
                system_prompt, user_prompt, context
            )

            # Record completion
            self._record_event("task_completed", {
                "artifacts_created": len(artifacts),
                "response_length": len(response),
                "tool_calls": len(self._tool_calls),
            })

            return {
                "success": True,
                "response": response,
                "artifacts": artifacts,
                "tool_log": self.tool_executor.get_execution_log(),
                "tool_calls": self._tool_calls,
                "clarifications": self._clarifications_requested,
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
        context = task_input.get("context", {})

        prompt_parts = [
            f"## Goal\n{goal}",
            f"\n## Task Description\n{description}",
        ]

        if dependencies_output:
            prompt_parts.append("\n## Input from Previous Phases")
            for dep_name, dep_output in dependencies_output.items():
                if isinstance(dep_output, dict):
                    # Multiple artifacts from this dependency
                    prompt_parts.append(f"\n### {dep_name}")
                    prompt_parts.append(dep_output.get("primary", ""))
                    if dep_output.get("artifacts"):
                        prompt_parts.append("\n**Additional artifacts:**")
                        for art in dep_output["artifacts"][1:]:  # Skip primary
                            prompt_parts.append(f"- {art['name']} ({art['type']})")
                else:
                    prompt_parts.append(f"\n### {dep_name}\n{dep_output}")

        # Add accumulated context summary
        if context.get("completed_phases"):
            prompt_parts.append("\n## Project Progress")
            prompt_parts.append(
                f"Completed phases: {', '.join(context['completed_phases'])}"
            )

        prompt_parts.append(
            "\n## Instructions\n"
            "Complete your assigned task. Use the available tools to:\n"
            "1. Create artifacts for your deliverables\n"
            "2. Read previous artifacts if you need more detail\n"
            "3. Request clarification if requirements are ambiguous\n\n"
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
                    "id": str(a.id),
                    "type": a.artifact_type,
                    "name": a.name,
                    "summary": a.content[:500] if a.content else "",
                }
                for a in artifacts
            ]

    def _call_llm_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        context: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Call LLM with tool support and handle agentic loop."""
        if not self.client:
            # Mock response for testing without API key
            logger.warning("No Anthropic API key - returning mock response")
            return self._generate_mock_response(user_prompt), self._create_default_artifact()

        # Build context message
        context_str = ""
        if context:
            context_str = "\n\n## Available Artifacts (use read_artifact to get full content)\n"
            for ctx in context:
                context_str += f"- [{ctx['id'][:8]}...] {ctx['type']}: {ctx['name']}\n"

        full_user_prompt = user_prompt + context_str

        # Initialize conversation
        messages = [{"role": "user", "content": full_user_prompt}]
        tools = self.get_available_tools()

        # Agentic loop
        artifacts_created = []
        final_response = ""
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            # Check token budget BEFORE making LLM call
            estimated_tokens = self._estimate_tokens(system_prompt, str(messages))
            self._check_budget(estimated_tokens)

            # Record the LLM request
            self._record_event("llm_request", {
                "model": settings.llm_model,
                "iteration": iteration,
                "message_count": len(messages),
            })

            # Make the API call with tools
            message = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            # Track token usage
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            cost = calculate_cost(settings.llm_model, input_tokens, output_tokens)
            self._update_usage(total_tokens, cost)

            # Record the response
            self._record_event("llm_response", {
                "iteration": iteration,
                "tokens_used": total_tokens,
                "cost_usd": cost,
                "stop_reason": message.stop_reason,
                "content_blocks": len(message.content),
            })

            # Process response content
            tool_use_blocks = []
            text_blocks = []

            for block in message.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # Collect text response
            if text_blocks:
                final_response = "\n".join(text_blocks)

            # If no tool use, we're done
            if message.stop_reason == "end_turn" or not tool_use_blocks:
                break

            # Process tool calls
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input

                # Record tool call
                self._tool_calls.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "iteration": iteration,
                })

                # Execute the tool
                result = self._execute_tool(tool_name, tool_input)

                # Track artifacts created
                if tool_name == "create_artifact" and result.get("success"):
                    artifacts_created.append(result.get("artifact"))

                # Track clarifications
                if tool_name == "request_clarification":
                    self._clarifications_requested.append(tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result),
                })

            # Add assistant message and tool results to conversation
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": tool_results})

        # If no artifacts were created via tools, create a default one
        if not artifacts_created and final_response:
            artifacts_created = self._create_default_artifact(final_response)

        return final_response, artifacts_created

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        logger.info(f"Executing tool '{tool_name}' with input: {tool_input}")

        try:
            if tool_name == "create_artifact":
                result = self.tool_executor.execute(
                    "create_artifact",
                    run_id=self.run_id,
                    task_id=self.task_id,
                    artifact_type=tool_input.get("artifact_type"),
                    name=tool_input.get("name"),
                    content=tool_input.get("content"),
                    content_type=tool_input.get("content_type", "text/markdown"),
                    metadata={"role": self.role, "created_via": "tool_use"},
                )
                if result.success:
                    return {"success": True, "artifact": result.result, "message": "Artifact created successfully"}
                return {"success": False, "error": result.error}

            elif tool_name == "read_artifact":
                result = self.tool_executor.execute(
                    "read_artifact",
                    artifact_id=tool_input.get("artifact_id"),
                )
                if result.success:
                    return {"success": True, "content": result.result}
                return {"success": False, "error": result.error}

            elif tool_name == "list_artifacts":
                result = self.tool_executor.execute(
                    "list_artifacts",
                    run_id=self.run_id,
                    artifact_type=tool_input.get("artifact_type"),
                )
                if result.success:
                    return {"success": True, "artifacts": result.result}
                return {"success": False, "error": result.error}

            elif tool_name == "request_clarification":
                # Store the clarification request - will be handled by orchestrator
                self._record_event("clarification_requested", tool_input)
                return {
                    "success": True,
                    "message": "Clarification request recorded. Proceeding with best interpretation.",
                    "question": tool_input.get("question"),
                }

            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_default_artifact(self, response: str = None) -> List[Dict[str, Any]]:
        """Create a default artifact from the response if none were created via tools."""
        if response is None:
            response = "Mock artifact content for testing."

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
            metadata={"role": self.role, "created_via": "default"},
        )

        if result.success:
            return [result.result]
        return []

    def _estimate_tokens(self, system_prompt: str, content: str) -> int:
        """Estimate tokens for a request (rough approximation)."""
        # Rough estimate: ~4 chars per token for English text
        total_chars = len(system_prompt) + len(content)
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
