#!/usr/bin/env python3
"""MCP Server for Multi-Agent Orchestrator.

This exposes the orchestrator as an MCP tool that Claude Code can use.

Usage:
    1. Add to ~/.claude/claude_desktop_config.json:
       {
         "mcpServers": {
           "orchestrator": {
             "command": "python",
             "args": ["/path/to/orchestrator/mcp_server.py"]
           }
         }
       }

    2. Or run directly: python mcp_server.py
"""

import asyncio
import json
import sys
from typing import Any

# MCP protocol uses JSON-RPC over stdio
async def handle_request(request: dict) -> dict:
    """Handle incoming MCP requests."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "multi-agent-orchestrator",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "orchestrate_task",
                        "description": "Run a multi-agent workflow to complete a complex software task. Agents include: Backend Engineer, Frontend Engineer, Code Reviewer, Security Reviewer, Tech Lead, and Business Analyst.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Natural language description of what to build or do"
                                },
                                "project_type": {
                                    "type": "string",
                                    "enum": ["greenfield", "existing"],
                                    "default": "greenfield",
                                    "description": "Whether this is a new project or existing codebase"
                                },
                                "working_directory": {
                                    "type": "string",
                                    "description": "Path to the project directory"
                                }
                            },
                            "required": ["task"]
                        }
                    },
                    {
                        "name": "analyze_project",
                        "description": "Analyze an existing codebase to understand its structure, dependencies, and patterns before making changes.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to the project to analyze"
                                }
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "run_tests",
                        "description": "Run tests in the project using auto-detected test framework (pytest, jest, vitest, cucumber, behave, etc.)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to run tests in"
                                },
                                "filter": {
                                    "type": "string",
                                    "description": "Test filter pattern"
                                }
                            }
                        }
                    },
                    {
                        "name": "build_prompt",
                        "description": "Translate natural language into an optimized orchestrator prompt. Takes casual descriptions like 'make buttons look better' and generates structured prompts that leverage the multi-agent framework effectively.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "request": {
                                    "type": "string",
                                    "description": "Natural language description of what you want to change (e.g., 'the login page is ugly and slow')"
                                },
                                "project_path": {
                                    "type": "string",
                                    "description": "Optional path to the project for context"
                                }
                            },
                            "required": ["request"]
                        }
                    },
                    {
                        "name": "get_workflow_plan",
                        "description": "Get a detailed multi-agent workflow plan with prompts that Claude Code can execute directly. Returns agent system prompts and tasks - Claude Code then becomes each agent in sequence. No separate API calls needed - uses your Claude Max subscription.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Natural language description of what to build or do"
                                },
                                "working_directory": {
                                    "type": "string",
                                    "description": "Path to the project directory"
                                },
                                "agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional: specific agents to use (e.g., ['backend', 'reviewer']). If not provided, auto-selects based on task."
                                }
                            },
                            "required": ["task"]
                        }
                    }
                ]
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            result = await execute_tool(tool_name, args)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ],
                    "isError": True
                }
            }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}"
        }
    }


async def execute_tool(tool_name: str, args: dict) -> Any:
    """Execute an orchestrator tool."""
    import os

    # Tools that don't require API key (they don't make their own API calls)
    no_api_key_tools = ["get_workflow_plan", "build_prompt", "analyze_project", "run_tests"]

    # Check for API key only for tools that need it
    if tool_name not in no_api_key_tools:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Ensure the MCP server config includes the 'env' block with the API key. "
                "Or use 'get_workflow_plan' which uses Claude Code's connection instead."
            )

    if tool_name == "orchestrate_task":
        # Use lightweight chat-based orchestrator (no Celery/Redis needed)
        from orchestrator.chat import Orchestrator

        working_dir = args.get("working_directory")
        orch = Orchestrator(project_dir=working_dir)

        response = await orch.chat(args["task"])

        return {
            "success": not response.needs_input,
            "message": response.message,
            "tasks_dispatched": [
                {"agent": t.agent, "task": t.task, "status": t.status}
                for t in response.tasks_dispatched
            ],
            "files_modified": response.files_modified,
            "question": response.question if response.needs_input else None
        }

    elif tool_name == "analyze_project":
        # Use lightweight code analysis (no heavy infrastructure needed)
        from orchestrator.tools.code_analysis import CodeAnalyzer, CodeAnalysisConfig
        import os

        path = args["path"]
        # Handle relative paths
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        config = CodeAnalysisConfig(root_path=path)
        analyzer = CodeAnalyzer(config)

        # Get project structure and basic analysis
        structure = analyzer.get_project_structure(max_depth=3)

        # Detect languages and frameworks
        analysis = {
            "path": path,
            "structure": structure,
            "languages": [],
            "frameworks": [],
            "files_analyzed": 0
        }

        # Scan for common patterns
        for ext, lang in [(".py", "Python"), (".js", "JavaScript"), (".ts", "TypeScript"),
                          (".jsx", "React"), (".tsx", "React/TypeScript"), (".go", "Go"),
                          (".rs", "Rust"), (".java", "Java")]:
            try:
                results = analyzer.search_code(f".*{ext}$", is_regex=True, file_extensions=[ext])
                if results:
                    analysis["languages"].append(lang)
                    analysis["files_analyzed"] += len(results)
            except Exception:
                pass

        # Detect frameworks from config files
        framework_indicators = {
            "package.json": "Node.js",
            "requirements.txt": "Python",
            "pyproject.toml": "Python (modern)",
            "Cargo.toml": "Rust",
            "go.mod": "Go",
            "pom.xml": "Java/Maven",
            "build.gradle": "Java/Gradle",
        }

        for file, framework in framework_indicators.items():
            if os.path.exists(os.path.join(path, file)):
                analysis["frameworks"].append(framework)

        return analysis

    elif tool_name == "run_tests":
        from orchestrator.tools.execution import ExecutionTools, ExecutionConfig

        config = ExecutionConfig(working_dir=args.get("path", "."))
        exec_tools = ExecutionTools(config)

        result = exec_tools.run_tests(
            path=args.get("path"),
            filter_pattern=args.get("filter")
        )
        return result

    elif tool_name == "build_prompt":
        from orchestrator.prompt_builder import analyze_request

        result = analyze_request(
            args["request"],
            project_dir=args.get("project_path")
        )

        return {
            "original_input": result.original_input,
            "optimized_prompt": result.optimized_prompt,
            "workflow": result.workflow_recommendation.value,
            "agents": result.agents_involved,
            "agent_sequence": result.agent_sequence,
            "categories": [c.value for c in result.categories],
            "intensity": result.intensity,
            "scope": result.scope,
            "explanation": result.explanation,
            "usage_hint": "Copy the 'optimized_prompt' and use it with orchestrate_task"
        }

    elif tool_name == "get_workflow_plan":
        # Import agent definitions
        from orchestrator.delegate import AGENTS
        from orchestrator.prompt_builder import analyze_request

        task = args["task"]
        working_dir = args.get("working_directory", os.getcwd())

        # Analyze the task to determine which agents to use
        analysis = analyze_request(task, project_dir=working_dir)

        # Get requested agents or use auto-detected ones
        requested_agents = args.get("agents")
        if requested_agents:
            agent_sequence = [[a] for a in requested_agents]
        else:
            agent_sequence = analysis.agent_sequence

        # Build the workflow plan with full agent prompts
        workflow_steps = []
        step_num = 1

        for phase in agent_sequence:
            phase_steps = []
            for agent_name in phase:
                if agent_name not in AGENTS:
                    continue

                agent_config = AGENTS[agent_name]

                phase_steps.append({
                    "step": step_num,
                    "agent": agent_name,
                    "agent_name": agent_config["name"],
                    "system_prompt": agent_config["system_prompt"],
                    "task_prompt": f"""## Task
{task}

## Working Directory
{working_dir}

## Instructions
You are now acting as the {agent_config['name']}. Complete your assigned task using the available tools.
When you're done, provide a summary of what was accomplished.""",
                    "available_tools": agent_config["tools"],
                })
                step_num += 1

            if phase_steps:
                workflow_steps.append({
                    "phase": len(workflow_steps) + 1,
                    "parallel": len(phase_steps) > 1,
                    "agents": phase_steps
                })

        return {
            "task": task,
            "working_directory": working_dir,
            "workflow_type": analysis.workflow_recommendation.value,
            "total_agents": sum(len(phase["agents"]) for phase in workflow_steps),
            "phases": len(workflow_steps),
            "workflow": workflow_steps,
            "execution_instructions": """
## How to Execute This Workflow

Claude Code should execute each phase in order. For each agent step:

1. **Adopt the agent's persona** by following its system_prompt
2. **Execute the task_prompt** using available tools (read files, write code, run tests, etc.)
3. **Pass results to the next agent** - each agent builds on previous work

### Parallel Phases
When `parallel: true`, multiple agents can work simultaneously on independent tasks.

### Sequential Phases
When `parallel: false`, complete the agent's work before moving to the next.

### Example Execution
For each step, Claude Code should:
```
I am now acting as the [Agent Name].

[System prompt context]

[Execute the task using tools...]

[Summarize what was accomplished]
```

This workflow uses YOUR Claude Max subscription - no separate API calls needed!
""",
            "available_agents": list(AGENTS.keys()),
        }

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


async def main():
    """Run the MCP server."""
    # Read from stdin, write to stdout
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break

            request = json.loads(line)
            response = await handle_request(request)

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
