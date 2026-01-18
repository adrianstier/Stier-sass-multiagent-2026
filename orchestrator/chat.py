#!/usr/bin/env python3
"""
Chat interface to the Multi-Agent Orchestrator.

This provides a single interface where you talk to ONE orchestrator,
and it coordinates multiple specialized agents to complete your tasks.

Usage:
    # Interactive mode
    python -m orchestrator.chat

    # Single command
    python -m orchestrator.chat "Build a user authentication system"

    # From Python
    from orchestrator.chat import Orchestrator
    orch = Orchestrator(project_dir="/path/to/project")
    response = await orch.chat("Add JWT authentication to the API")
"""

import asyncio
import os
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from orchestrator.delegate import delegate, AGENTS, DelegationResult


@dataclass
class AgentTask:
    """A task assigned to an agent."""
    agent: str
    task: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[DelegationResult] = None


@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""
    message: str
    tasks_dispatched: List[AgentTask] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    needs_input: bool = False
    question: Optional[str] = None


class Orchestrator:
    """
    Main orchestrator that coordinates specialized agents.

    You talk to this, and it decides which agents to invoke
    to complete your tasks.
    """

    SYSTEM_PROMPT = """You are a software development orchestrator. Users describe what they want built,
and you coordinate specialized agents to make it happen.

## Your Role
- Understand user requirements
- Break down complex tasks into agent-appropriate subtasks
- Dispatch tasks to the RIGHT specialist agent
- Synthesize results and report back to the user
- Ask clarifying questions when requirements are unclear

## Available Agents
You can delegate to these specialists by calling the delegate_to_agent tool:

1. **analyst** - Business Analyst: Requirements gathering, user stories, acceptance criteria, documentation
2. **project_manager** - Project Manager: Project plans, timelines, coordination, risk management
3. **ux_engineer** - UX Engineer: User research, wireframes, design systems, accessibility
4. **tech_lead** - Tech Lead: Architecture, design decisions, API contracts, tech stack
5. **database** - Database Engineer: Schema design, migrations, query optimization
6. **backend** - Backend Engineer: REST APIs, authentication, business logic, Python/FastAPI
7. **frontend** - Frontend Engineer: React, TypeScript, distinctive UI design, bold aesthetics
8. **reviewer** - Code Reviewer: Code quality, bugs, best practices, test coverage
9. **security** - Security Reviewer: Vulnerabilities, OWASP, security scans, compliance
10. **devops** - DevOps Engineer: Docker, CI/CD, deployment, infrastructure
11. **data_scientist** - Data Scientist: EDA, ML models, feature engineering, data pipelines
12. **design_reviewer** - Design Reviewer: UI aesthetics, typography, color, motion, distinctiveness

## How to Work
1. When a user makes a request, analyze what needs to be done
2. If the request is clear, delegate to appropriate agent(s)
3. If unclear, ask ONE clarifying question
4. For complex features, break into steps and use multiple agents in sequence
5. After agents complete, summarize what was done

## Example Workflow
User: "Add user authentication"

You think: This needs backend work (JWT, endpoints) and maybe frontend (login form).
Let me start with backend, then handle frontend.

Action: delegate_to_agent("backend", "Implement JWT authentication with login/register endpoints")
[wait for result]
Action: delegate_to_agent("frontend", "Create login and registration forms that call the auth API")
[wait for result]

Then summarize: "I've added user authentication. The backend now has /auth/login and /auth/register
endpoints using JWT tokens. The frontend has LoginForm and RegisterForm components..."

## Important Rules
- ALWAYS use delegate_to_agent to do actual work - you cannot modify code directly
- One agent at a time for sequential dependencies
- Ask questions BEFORE delegating if requirements are unclear
- After delegation, report what was accomplished"""

    def __init__(
        self,
        project_dir: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.project_dir = project_dir or os.getcwd()
        self.model = model
        self.conversation_history: List[Dict] = []
        self.tasks_completed: List[AgentTask] = []

    async def chat(self, message: str) -> OrchestratorResponse:
        """
        Send a message to the orchestrator and get a response.

        The orchestrator will analyze your request and dispatch
        appropriate agents to complete it.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return OrchestratorResponse(
                message="Error: ANTHROPIC_API_KEY environment variable not set",
                needs_input=False
            )

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Define the delegation tool
        tools = [
            {
                "name": "delegate_to_agent",
                "description": "Delegate a task to a specialized agent. The agent will use tools (filesystem, git, etc.) to complete the task and return results.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": list(AGENTS.keys()),
                            "description": "Which specialist to delegate to"
                        },
                        "task": {
                            "type": "string",
                            "description": "Detailed description of what the agent should do"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context (existing code, constraints, etc.)"
                        }
                    },
                    "required": ["agent", "task"]
                }
            },
            {
                "name": "ask_user",
                "description": "Ask the user a clarifying question before proceeding",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask"
                        }
                    },
                    "required": ["question"]
                }
            }
        ]

        tasks_dispatched = []
        files_modified = []
        final_message = ""
        needs_input = False
        question = None

        # Agentic loop
        messages = self.conversation_history.copy()
        max_iterations = 10

        for iteration in range(max_iterations):
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )

            # Process response
            assistant_content = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    final_message = block.text
                    assistant_content.append({"type": "text", "text": block.text})

                elif block.type == "tool_use":
                    has_tool_use = True
                    tool_name = block.name
                    tool_input = block.input

                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": tool_name,
                        "input": tool_input
                    })

                    if tool_name == "delegate_to_agent":
                        # Actually delegate to the agent
                        agent_name = tool_input["agent"]
                        agent_task = tool_input["task"]
                        context = tool_input.get("context")

                        print(f"\n→ Delegating to {agent_name}: {agent_task[:80]}...")

                        task_record = AgentTask(
                            agent=agent_name,
                            task=agent_task,
                            status="running"
                        )
                        tasks_dispatched.append(task_record)

                        # Run the agent
                        result = await delegate(
                            agent=agent_name,
                            task=agent_task,
                            context=context,
                            working_dir=self.project_dir,
                            model=self.model,
                        )

                        task_record.status = "completed" if result.success else "failed"
                        task_record.result = result

                        if result.files_modified:
                            files_modified.extend(result.files_modified)

                        # Format result for orchestrator
                        tool_result = f"""Agent: {agent_name}
Success: {result.success}
Tool calls made: {len(result.tool_calls)}
Files modified: {result.files_modified}

Agent's response:
{result.output}"""

                        if result.error:
                            tool_result += f"\n\nError: {result.error}"

                        print(f"  ✓ {agent_name} completed ({len(result.tool_calls)} tool calls)")

                    elif tool_name == "ask_user":
                        # Orchestrator wants to ask a question
                        needs_input = True
                        question = tool_input["question"]
                        tool_result = "Question sent to user. Waiting for response."

                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    # Add to messages
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result
                        }]
                    })
                    assistant_content = []

                    # If asking user, break out to get their response
                    if needs_input:
                        break

            # If no tool use or asking user, we're done
            if not has_tool_use or needs_input:
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                break

            if response.stop_reason == "end_turn":
                break

        # Update conversation history with final assistant message
        if final_message:
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })

        self.tasks_completed.extend(tasks_dispatched)

        return OrchestratorResponse(
            message=final_message,
            tasks_dispatched=tasks_dispatched,
            files_modified=list(set(files_modified)),
            needs_input=needs_input,
            question=question
        )

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.tasks_completed = []


async def interactive_chat():
    """Run an interactive chat session with the orchestrator."""
    print("=" * 60)
    print("Multi-Agent Orchestrator")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print("\nI coordinate specialized agents to build software.")
    print("Tell me what you want to build, and I'll make it happen.")
    print("\nCommands:")
    print("  /agents  - List available agents")
    print("  /status  - Show tasks completed this session")
    print("  /reset   - Clear conversation history")
    print("  /quit    - Exit")
    print("=" * 60)

    orch = Orchestrator()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd == "/quit" or cmd == "/exit":
                print("Goodbye!")
                break

            elif cmd == "/agents":
                print("\nAvailable Agents:")
                for name, config in AGENTS.items():
                    print(f"  {name}: {config['name']}")
                continue

            elif cmd == "/status":
                print(f"\nTasks completed: {len(orch.tasks_completed)}")
                for task in orch.tasks_completed:
                    status_icon = "✓" if task.status == "completed" else "✗"
                    print(f"  {status_icon} [{task.agent}] {task.task[:60]}...")
                continue

            elif cmd == "/reset":
                orch.reset()
                print("Conversation reset.")
                continue

            else:
                print(f"Unknown command: {cmd}")
                continue

        # Chat with orchestrator
        print("\nOrchestrator: Thinking...")

        response = await orch.chat(user_input)

        print(f"\nOrchestrator: {response.message}")

        if response.files_modified:
            print(f"\n📁 Files modified: {', '.join(response.files_modified)}")

        if response.needs_input and response.question:
            # Orchestrator is asking a question - it will be the next message
            pass


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) > 1:
        # Single command mode
        task = " ".join(sys.argv[1:])
        orch = Orchestrator()
        response = asyncio.run(orch.chat(task))
        print(response.message)
        if response.files_modified:
            print(f"\nFiles modified: {response.files_modified}")
    else:
        # Interactive mode
        asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()
