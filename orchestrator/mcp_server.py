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
import uuid
from datetime import datetime
from typing import Any, Dict, List

# In-memory workflow state (persists for session)
WORKFLOW_STATE: Dict[str, dict] = {}

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
        # Import data science tools
        from orchestrator.agents.data_science.mcp_tools import DATA_SCIENCE_TOOLS

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "orchestrate_task",
                        "description": "[USES CLAUDE MAX] Orchestrate a multi-agent workflow. Returns a complete execution plan with Task-ready prompts for Claude Code to run. Agents include: Backend Engineer, Frontend Engineer, Code Reviewer, Security Reviewer, Tech Lead, Business Analyst, and more. No separate API credits needed.",
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
                        "description": "[USES CLAUDE MAX] Get a multi-agent workflow plan that Claude Code executes directly. No separate API credits needed. Returns agent prompts for: Backend Engineer, Frontend Engineer, Code Reviewer, Security Reviewer, Tech Lead, Business Analyst, and more.",
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
                    },
                    {
                        "name": "execute_agent",
                        "description": "[USES CLAUDE MAX] Execute a single specialized agent as a Claude Code Task subagent. Returns a Task-ready prompt. The agent will use Claude Code's tools (Read, Write, Edit, Bash, etc.) to complete its work.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "agent": {
                                    "type": "string",
                                    "description": "Agent type: backend, frontend, reviewer, security, devops, tech_lead, analyst, database, project_manager, ux_engineer, data_scientist"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "The specific task for this agent to complete"
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Optional: context from previous agents (e.g., files created, decisions made)"
                                },
                                "working_directory": {
                                    "type": "string",
                                    "description": "Path to the project directory"
                                }
                            },
                            "required": ["agent", "task"]
                        }
                    },
                    {
                        "name": "list_agents",
                        "description": "List all available specialized agents and their capabilities.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "start_workflow",
                        "description": "Start a new orchestrated workflow. Returns a workflow_id to track progress across agents.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "The main task/goal for this workflow"
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
                        "name": "update_workflow",
                        "description": "Update workflow state after an agent completes. Track progress, artifacts, and pass context to next agent.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "workflow_id": {
                                    "type": "string",
                                    "description": "The workflow ID from start_workflow"
                                },
                                "agent": {
                                    "type": "string",
                                    "description": "The agent that just completed"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["completed", "failed", "blocked"],
                                    "description": "Agent completion status"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Summary of what the agent accomplished"
                                },
                                "files_modified": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of files created/modified by this agent"
                                },
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Any issues or blockers encountered"
                                }
                            },
                            "required": ["workflow_id", "agent", "status", "summary"]
                        }
                    },
                    {
                        "name": "get_workflow_status",
                        "description": "Get current status of a workflow including all agent progress and accumulated context.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "workflow_id": {
                                    "type": "string",
                                    "description": "The workflow ID to check"
                                }
                            },
                            "required": ["workflow_id"]
                        }
                    },
                    {
                        "name": "get_workflow_context",
                        "description": "Get accumulated context from all completed agents to pass to the next agent.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "workflow_id": {
                                    "type": "string",
                                    "description": "The workflow ID"
                                }
                            },
                            "required": ["workflow_id"]
                        }
                    },
                    {
                        "name": "check_allstate_compliance",
                        "description": "Run Allstate/ISSAS compliance checks on the codebase. Scans for violations of ISSAS, NIST SP 800-88, NAIC AI Governance, and CPRA ADMT requirements. Returns detailed report with violation codes (SEC-01 through INT-02) and remediation guidance.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to the directory to scan for compliance violations"
                                },
                                "exclude_dirs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Directories to exclude from scanning (default: node_modules, .git, __pycache__, etc.)"
                                }
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "get_compliance_rules",
                        "description": "Get the full list of Allstate/ISSAS compliance rules and their descriptions. Useful for understanding what the compliance checker looks for.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "frontend_review",
                        "description": "[USES CLAUDE MAX] Run a comprehensive frontend review workflow with multiple specialized agents: Graphic Designer (visual beauty), UX Engineer (accessibility/usability), Frontend Engineer (code quality), and Design Reviewer (live Playwright testing). Returns Task-ready prompts for each agent.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "Live URL to test (e.g., http://localhost:3000)"
                                },
                                "codebase_path": {
                                    "type": "string",
                                    "description": "Path to the frontend codebase"
                                },
                                "wcag_level": {
                                    "type": "string",
                                    "enum": ["A", "AA", "AAA"],
                                    "default": "AA",
                                    "description": "WCAG compliance level to check"
                                },
                                "figma_url": {
                                    "type": "string",
                                    "description": "Optional Figma design file URL for comparison"
                                }
                            },
                            "required": ["url", "codebase_path"]
                        }
                    },
                    # Add data science tools
                    *DATA_SCIENCE_TOOLS
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

    # Data science tools (handled separately)
    ds_tools = ["ds_workflow_plan", "ds_analyze_data", "ds_quality_check", "ds_list_agents", "ds_artifact_status"]

    if tool_name in ds_tools:
        from orchestrator.agents.data_science.mcp_tools import execute_ds_tool
        return await execute_ds_tool(tool_name, args)

    # All tools now use Claude Max subscription - no separate API key needed

    if tool_name == "orchestrate_task":
        # Returns a complete orchestration plan for Claude Code to execute
        # No separate API calls needed - uses your Claude Max subscription
        from orchestrator.delegate import AGENTS
        from orchestrator.prompt_builder import analyze_request

        task = args["task"]
        project_type = args.get("project_type", "greenfield")
        working_dir = args.get("working_directory", os.getcwd())

        # Analyze the task to determine which agents to use
        analysis = analyze_request(task, project_dir=working_dir)

        # Build orchestrator system prompt
        orchestrator_prompt = f"""You are a Software Development Orchestrator coordinating specialized agents.

## Your Task
{task}

## Working Directory
{working_dir}

## Project Type
{project_type}

## Workflow Type
{analysis.workflow_recommendation.value}

## Available Agents
{chr(10).join(f"- **{name}**: {config['name']}" for name, config in AGENTS.items())}

## Recommended Agent Sequence
{analysis.agent_sequence}

## Instructions
As the orchestrator, you will:

1. **Analyze** the task requirements
2. **Delegate** to specialized agents using the Task tool
3. **Coordinate** information flow between agents
4. **Synthesize** results into a final summary

For each agent in the sequence, spawn a Task subagent:
```
Task(
    subagent_type="general-purpose",
    description="<Agent Name>: <Brief task>",
    prompt=<agent's full prompt from the workflow below>
)
```

## Workflow Execution Plan
"""

        # Build the detailed workflow with agent prompts
        workflow_steps = []
        step_num = 1

        for phase in analysis.agent_sequence:
            phase_steps = []
            for agent_name in phase:
                if agent_name not in AGENTS:
                    continue

                agent_config = AGENTS[agent_name]

                # Build complete Task-ready prompt
                full_prompt = f"""{agent_config['system_prompt']}

---

## Your Task
{task}

## Working Directory
{working_dir}

## Instructions
You are now acting as the {agent_config['name']}. Complete your assigned task using the available tools:
- Use Read/Glob/Grep to explore the codebase
- Use Edit/Write to make changes
- Use Bash to run commands (tests, builds, etc.)

When you're done, provide a clear summary of:
1. What you accomplished
2. Files created/modified
3. Any issues encountered
4. Recommendations for next steps"""

                phase_steps.append({
                    "step": step_num,
                    "agent": agent_name,
                    "agent_name": agent_config["name"],
                    "task_prompt": full_prompt,
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
            "project_type": project_type,
            "workflow_type": analysis.workflow_recommendation.value,
            "total_agents": sum(len(phase["agents"]) for phase in workflow_steps),
            "phases": len(workflow_steps),
            "orchestrator_prompt": orchestrator_prompt,
            "workflow": workflow_steps,
            "execution_mode": "claude_max",
            "api_credits_required": False,
            "execution_instructions": """
## How to Execute This Workflow

This orchestration plan uses YOUR Claude Max subscription - no separate API credits needed!

### Option 1: Full Orchestration (Recommended)
Use the orchestrator_prompt with a Task agent to coordinate the entire workflow:

```
Task(
    subagent_type="general-purpose",
    description="Orchestrator: {task[:40]}...",
    prompt=orchestrator_prompt
)
```

The orchestrator will spawn specialist agents as needed.

### Option 2: Direct Agent Execution
Execute each agent phase manually:

```python
# Phase 1
Task(subagent_type="general-purpose", description="Tech Lead", prompt=workflow["phases"][0]["agents"][0]["task_prompt"])

# Phase 2 (parallel if multiple agents)
Task(subagent_type="general-purpose", description="Backend", prompt=backend_prompt)
Task(subagent_type="general-purpose", description="Frontend", prompt=frontend_prompt)

# Continue with remaining phases...
```

### Context Passing
Pass results from previous agents to the next:
```
prompt = next_agent_prompt + f"\\n\\n## Previous Agent Results\\n{previous_results}"
```
""",
            "available_agents": list(AGENTS.keys()),
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
            "usage_hint": "Use 'get_workflow_plan' to execute this with Claude Code (recommended - uses your Claude Max subscription, no API credits needed)"
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

        # Frontend agents that should use Playwright
        frontend_agents = ["frontend", "ux_engineer", "design_reviewer", "graphic_designer"]

        for phase in agent_sequence:
            phase_steps = []
            for agent_name in phase:
                if agent_name not in AGENTS:
                    continue

                agent_config = AGENTS[agent_name]
                is_frontend_agent = agent_name in frontend_agents

                # Playwright instructions for frontend agents
                playwright_section = """
## Visual Verification with Playwright (REQUIRED)

You have access to Playwright browser tools. **USE THEM** to verify your work visually.

### Available Playwright Tools
- `mcp__playwright__browser_navigate(url)` - Navigate to a URL
- `mcp__playwright__browser_take_screenshot(filename)` - Capture screenshots
- `mcp__playwright__browser_resize(width, height)` - Test responsive breakpoints
- `mcp__playwright__browser_snapshot()` - Get accessibility tree
- `mcp__playwright__browser_click(element, ref)` - Click interactive elements
- `mcp__playwright__browser_type(element, ref, text)` - Type in inputs
- `mcp__playwright__browser_press_key(key)` - Test keyboard navigation
- `mcp__playwright__browser_console_messages(level)` - Check for errors

### Verification Workflow
1. Navigate to the live preview URL
2. Take screenshots at multiple viewports (1440x900, 768x1024, 375x812)
3. Use browser_snapshot for accessibility verification
4. Test all interactive elements
5. Check console for errors

**Do NOT mark your work complete without visual verification.**
""" if is_frontend_agent else ""

                # Build complete Task-ready prompt
                full_prompt = f"""{agent_config['system_prompt']}

---

## Your Task
{task}

## Working Directory
{working_dir}
{playwright_section}
## Instructions
You are now acting as the {agent_config['name']}. Complete your assigned task using the available tools:
- Use Read/Glob/Grep to explore the codebase
- Use Edit/Write to make changes
- Use Bash to run commands (tests, builds, etc.)
{'''- Use Playwright browser tools to visually verify your implementation''' if is_frontend_agent else ''}

When you're done, provide a clear summary of:
1. What you accomplished
2. Files created/modified
3. Any issues encountered
{'''4. Screenshots captured for verification
5. Recommendations for next steps''' if is_frontend_agent else '''4. Recommendations for next steps'''}"""

                phase_steps.append({
                    "step": step_num,
                    "agent": agent_name,
                    "agent_name": agent_config["name"],
                    "task_prompt": full_prompt,
                    "available_tools": agent_config["tools"] + (["playwright"] if is_frontend_agent else []),
                    "uses_playwright": is_frontend_agent,
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
## How to Execute This Workflow in Claude Code

Use the **Task tool** to spawn each agent as a subagent. This leverages your Claude Max subscription.

### For Sequential Phases (parallel: false)
Execute agents one at a time, passing context forward:

```python
# Phase 1: Tech Lead
result_1 = Task(
    subagent_type="general-purpose",
    description="Tech Lead: Architecture",
    prompt=workflow["phases"][0]["agents"][0]["task_prompt"]
)

# Phase 2: Backend (with context from Phase 1)
result_2 = Task(
    subagent_type="general-purpose",
    description="Backend: Implementation",
    prompt=workflow["phases"][1]["agents"][0]["task_prompt"] + f"\\n\\n## Context from Tech Lead\\n{result_1}"
)
```

### For Parallel Phases (parallel: true)
Launch multiple Task agents simultaneously:

```python
# Launch backend and frontend in parallel
Task(subagent_type="general-purpose", description="Backend Engineer", prompt=backend_prompt)
Task(subagent_type="general-purpose", description="Frontend Engineer", prompt=frontend_prompt)
```

### Workflow Pattern
1. **Planning** → Tech Lead / Analyst (sequential)
2. **Implementation** → Backend + Frontend (parallel)
3. **Review** → Code Reviewer (sequential)
4. **Security** → Security Reviewer (sequential)

Each agent has access to Claude Code's full toolset (Read, Write, Edit, Bash, Glob, Grep).

### Playwright Visual Verification (Frontend Agents)
Frontend-related agents (frontend, ux_engineer, design_reviewer, graphic_designer) are instructed to use Playwright for visual verification. **Ensure your app is running locally before executing these agents.**

They will:
- Navigate to the live URL
- Take screenshots at multiple viewports
- Test keyboard navigation and accessibility
- Verify interactive states
- Check for console errors
""",
            "available_agents": list(AGENTS.keys()),
            "claude_code_integration": {
                "recommended_approach": "Use Task tool with subagent_type='general-purpose'",
                "parallel_execution": "Send multiple Task calls in same message for parallel phases",
                "context_passing": "Append previous agent summaries to next agent's prompt",
                "model_hint": "Use 'haiku' for simple agents like reviewer, 'sonnet' for complex agents like backend"
            }
        }

    elif tool_name == "execute_agent":
        # Return a prompt that Claude Code can use with the Task tool
        from orchestrator.delegate import AGENTS

        agent_type = args["agent"]
        task = args["task"]
        context = args.get("context", "")
        working_dir = args.get("working_directory", os.getcwd())

        if agent_type not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_type}. Available: {list(AGENTS.keys())}")

        agent_config = AGENTS[agent_type]

        # Determine if this is a frontend-related agent that should use Playwright
        frontend_agents = ["frontend", "ux_engineer", "design_reviewer", "graphic_designer"]
        is_frontend_agent = agent_type in frontend_agents

        # Playwright instructions for frontend agents
        playwright_instructions = """
## Visual Verification with Playwright (REQUIRED)

You have access to Playwright browser tools. **USE THEM** to verify your work visually.

### Available Playwright Tools
- `mcp__playwright__browser_navigate(url)` - Navigate to a URL
- `mcp__playwright__browser_take_screenshot(filename)` - Capture screenshots
- `mcp__playwright__browser_resize(width, height)` - Test responsive breakpoints
- `mcp__playwright__browser_snapshot()` - Get accessibility tree (critical for a11y)
- `mcp__playwright__browser_click(element, ref)` - Click interactive elements
- `mcp__playwright__browser_type(element, ref, text)` - Type in inputs
- `mcp__playwright__browser_press_key(key)` - Test keyboard navigation (Tab, Enter, Escape)
- `mcp__playwright__browser_console_messages(level)` - Check for JS errors
- `mcp__playwright__browser_hover(element, ref)` - Test hover states

### Verification Workflow
1. Navigate to the live preview URL
2. Take screenshots at multiple viewports (1440x900, 768x1024, 375x812)
3. Use browser_snapshot to verify accessibility tree
4. Click through all interactive elements
5. Check console for errors

### Iteration Loop
```
Code change → Visual verification → Fix issues → Re-verify → Complete
```

**Do NOT mark your work complete without visual verification.**
""" if is_frontend_agent else ""

        # Build a complete prompt for Claude Code's Task tool
        task_prompt = f"""{agent_config['system_prompt']}

---

## Your Task
{task}

## Working Directory
{working_dir}

{f'''## Context from Previous Agents
{context}
''' if context else ''}
{playwright_instructions}
## Instructions
You are now acting as the {agent_config['name']}. Complete your assigned task using the available tools:
- Use Read/Glob/Grep to explore the codebase
- Use Edit/Write to make changes
- Use Bash to run commands (tests, builds, etc.)
{'''- Use Playwright browser tools to visually verify your implementation''' if is_frontend_agent else ''}

When you're done, provide a clear summary of:
1. What you accomplished
2. Files created/modified
3. Any issues encountered
{'''4. Screenshots captured for verification''' if is_frontend_agent else ''}
{'''5.''' if is_frontend_agent else '4.'} Recommendations for next steps"""

        return {
            "agent": agent_type,
            "agent_name": agent_config["name"],
            "task_prompt": task_prompt,
            "tools_available": agent_config["tools"] + (["playwright"] if is_frontend_agent else []),
            "uses_playwright": is_frontend_agent,
            "execution_hint": f"""
## How to Execute in Claude Code

Use the Task tool to spawn this agent:

```
Task(
    subagent_type="general-purpose",
    description="{agent_config['name']}: {task[:50]}...",
    prompt=<the task_prompt above>
)
```

The agent will use Claude Code's built-in tools to complete the work.
{'''
**Note:** This agent will use Playwright for visual verification. Ensure the app is running locally before execution.
''' if is_frontend_agent else ''}"""
        }

    elif tool_name == "list_agents":
        from orchestrator.delegate import AGENTS

        agents_list = []
        for agent_id, config in AGENTS.items():
            agents_list.append({
                "id": agent_id,
                "name": config["name"],
                "tools": config["tools"],
                "description": config["system_prompt"].split("\n")[0]  # First line as summary
            })

        return {
            "agents": agents_list,
            "usage": """
## Using Agents in Claude Code

1. **Start a workflow**: Use `start_workflow` to begin tracking
2. **Get a workflow plan**: Use `get_workflow_plan` with your task description
3. **Execute agents**: Use Claude Code's Task tool with prompts from the plan
4. **Update progress**: Use `update_workflow` after each agent completes
5. **Get context**: Use `get_workflow_context` to pass to next agent

## Recommended Workflow

```
# Step 1: Start tracking
workflow = start_workflow(task="Build user auth with JWT")

# Step 2: Get the execution plan
plan = get_workflow_plan(task="Build user auth with JWT")

# Step 3: For each agent in the plan, use Claude Code's Task tool
# Claude Code will spawn the agent and execute it

# Step 4: After each agent completes, update the workflow
update_workflow(
    workflow_id=workflow.id,
    agent="tech_lead",
    status="completed",
    summary="Designed JWT architecture with refresh tokens..."
)

# Step 5: Get context for next agent
context = get_workflow_context(workflow_id=workflow.id)
```
"""
        }

    elif tool_name == "start_workflow":
        from orchestrator.delegate import AGENTS
        from orchestrator.prompt_builder import analyze_request

        task = args["task"]
        working_dir = args.get("working_directory", os.getcwd())

        # Generate workflow ID
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        # Analyze task to determine agents
        analysis = analyze_request(task, project_dir=working_dir)

        # Initialize workflow state
        WORKFLOW_STATE[workflow_id] = {
            "id": workflow_id,
            "task": task,
            "working_directory": working_dir,
            "status": "in_progress",
            "created_at": datetime.now().isoformat(),
            "workflow_type": analysis.workflow_recommendation.value,
            "planned_agents": [agent for phase in analysis.agent_sequence for agent in phase],
            "agent_results": [],
            "files_modified": [],
            "total_issues": []
        }

        return {
            "workflow_id": workflow_id,
            "task": task,
            "working_directory": working_dir,
            "planned_agents": WORKFLOW_STATE[workflow_id]["planned_agents"],
            "workflow_type": analysis.workflow_recommendation.value,
            "next_step": f"Use get_workflow_plan(task='{task}') to get agent prompts, then execute with Claude Code's Task tool"
        }

    elif tool_name == "update_workflow":
        workflow_id = args["workflow_id"]

        if workflow_id not in WORKFLOW_STATE:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        workflow = WORKFLOW_STATE[workflow_id]

        # Record agent result
        agent_result = {
            "agent": args["agent"],
            "status": args["status"],
            "summary": args["summary"],
            "files_modified": args.get("files_modified", []),
            "issues": args.get("issues", []),
            "completed_at": datetime.now().isoformat()
        }

        workflow["agent_results"].append(agent_result)
        workflow["files_modified"].extend(args.get("files_modified", []))
        workflow["total_issues"].extend(args.get("issues", []))

        # Determine next agent
        completed_agents = [r["agent"] for r in workflow["agent_results"]]
        remaining_agents = [a for a in workflow["planned_agents"] if a not in completed_agents]

        # Update workflow status
        if args["status"] == "failed":
            workflow["status"] = "failed"
        elif not remaining_agents:
            workflow["status"] = "completed"

        return {
            "workflow_id": workflow_id,
            "agent": args["agent"],
            "status": args["status"],
            "workflow_status": workflow["status"],
            "completed_agents": completed_agents,
            "remaining_agents": remaining_agents,
            "next_agent": remaining_agents[0] if remaining_agents else None,
            "total_files_modified": len(set(workflow["files_modified"])),
            "total_issues": len(workflow["total_issues"])
        }

    elif tool_name == "get_workflow_status":
        workflow_id = args["workflow_id"]

        if workflow_id not in WORKFLOW_STATE:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        workflow = WORKFLOW_STATE[workflow_id]
        completed_agents = [r["agent"] for r in workflow["agent_results"]]
        remaining_agents = [a for a in workflow["planned_agents"] if a not in completed_agents]

        return {
            "workflow_id": workflow_id,
            "task": workflow["task"],
            "status": workflow["status"],
            "created_at": workflow["created_at"],
            "workflow_type": workflow["workflow_type"],
            "progress": {
                "completed": len(completed_agents),
                "total": len(workflow["planned_agents"]),
                "percentage": round(len(completed_agents) / len(workflow["planned_agents"]) * 100) if workflow["planned_agents"] else 0
            },
            "completed_agents": completed_agents,
            "remaining_agents": remaining_agents,
            "files_modified": list(set(workflow["files_modified"])),
            "issues": workflow["total_issues"],
            "agent_results": workflow["agent_results"]
        }

    elif tool_name == "get_workflow_context":
        workflow_id = args["workflow_id"]

        if workflow_id not in WORKFLOW_STATE:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        workflow = WORKFLOW_STATE[workflow_id]

        # Build context from all completed agents
        context_parts = [f"## Workflow: {workflow['task']}\n"]

        for result in workflow["agent_results"]:
            context_parts.append(f"""
### {result['agent'].replace('_', ' ').title()} - {result['status'].upper()}
{result['summary']}

Files modified: {', '.join(result['files_modified']) if result['files_modified'] else 'None'}
Issues: {', '.join(result['issues']) if result['issues'] else 'None'}
""")

        return {
            "workflow_id": workflow_id,
            "context": "\n".join(context_parts),
            "files_modified": list(set(workflow["files_modified"])),
            "completed_agents": [r["agent"] for r in workflow["agent_results"]],
            "usage": "Pass this context to the next agent's prompt to maintain continuity"
        }

    elif tool_name == "check_allstate_compliance":
        from orchestrator.compliance import check_compliance

        path = args["path"]
        # Handle relative paths
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        exclude_dirs = args.get("exclude_dirs")

        report = check_compliance(path, exclude_dirs)

        # Add summary
        report["summary"] = f"""
## Allstate/ISSAS Compliance Report

**Status**: {"✅ COMPLIANT" if report["is_compliant"] else "❌ NON-COMPLIANT"}
**Files Scanned**: {report["total_files_scanned"]}

### Violation Counts
- 🔴 Critical: {report["violation_counts"]["critical"]}
- 🟠 High: {report["violation_counts"]["high"]}
- 🟡 Medium: {report["violation_counts"]["medium"]}
- 🔵 Low: {report["violation_counts"]["low"]}

### Standards Covered
- ISSAS (Information Security Standards for Allstate Suppliers)
- NIST SP 800-88 (Data Destruction)
- NIST AI RMF (AI Risk Management Framework)
- NAIC Model Bulletin (Insurance AI Governance)
- CPRA ADMT (Automated Decision-Making Technology)
"""
        return report

    elif tool_name == "frontend_review":
        # Import the frontend review workflow
        from orchestrator.workflows.frontend_review import get_frontend_review_prompts, WORKFLOW_DEFINITION

        url = args["url"]
        codebase_path = args["codebase_path"]
        wcag_level = args.get("wcag_level", "AA")
        figma_url = args.get("figma_url")

        # Handle relative paths
        if not os.path.isabs(codebase_path):
            codebase_path = os.path.abspath(codebase_path)

        # Get all agent prompts
        prompts = get_frontend_review_prompts(
            url=url,
            codebase_path=codebase_path,
            wcag_level=wcag_level,
            figma_url=figma_url
        )

        return {
            "workflow": "Frontend Review",
            "url": url,
            "codebase_path": codebase_path,
            "wcag_level": wcag_level,
            "figma_url": figma_url,
            "phases": [
                {
                    "phase": 1,
                    "name": "Visual & UX Assessment",
                    "parallel": True,
                    "description": "Graphic Designer and UX Engineer review in parallel",
                    "agents": [
                        {
                            "agent": "graphic_designer",
                            "name": "Graphic Designer",
                            "task_prompt": prompts["graphic_designer"],
                            "model": "sonnet",
                            "focus": "Visual beauty, typography, color, emotional impact"
                        },
                        {
                            "agent": "ux_engineer",
                            "name": "UX Engineer",
                            "task_prompt": prompts["ux_engineer"],
                            "model": "sonnet",
                            "focus": "Accessibility, usability, user flows"
                        }
                    ]
                },
                {
                    "phase": 2,
                    "name": "Code Review",
                    "parallel": False,
                    "description": "Frontend Engineer reviews code quality",
                    "agents": [
                        {
                            "agent": "frontend_engineer",
                            "name": "Frontend Engineer",
                            "task_prompt": prompts["frontend_engineer"],
                            "model": "sonnet",
                            "focus": "Component architecture, performance, patterns"
                        }
                    ]
                },
                {
                    "phase": 3,
                    "name": "Live Testing & Synthesis",
                    "parallel": False,
                    "description": "Design Reviewer performs live Playwright testing",
                    "agents": [
                        {
                            "agent": "design_reviewer",
                            "name": "Design Reviewer",
                            "task_prompt": prompts["design_reviewer"],
                            "model": "sonnet",
                            "focus": "Responsive testing, interactions, final report"
                        }
                    ]
                }
            ],
            "execution_instructions": """
## Frontend Review Workflow - Execution Guide

### Quick Start
Execute this 3-phase review using Claude Code's Task tool:

### Phase 1: Visual & UX Assessment (PARALLEL)
Launch both agents in a single message:

```python
Task(
    subagent_type="general-purpose",
    description="Graphic Designer: Visual beauty review",
    prompt=phases[0]["agents"][0]["task_prompt"],
    model="sonnet"
)

Task(
    subagent_type="general-purpose",
    description="UX Engineer: Accessibility review",
    prompt=phases[0]["agents"][1]["task_prompt"],
    model="sonnet"
)
```

### Phase 2: Code Review (after Phase 1)
```python
Task(
    subagent_type="general-purpose",
    description="Frontend Engineer: Code review",
    prompt=phases[1]["agents"][0]["task_prompt"] + "\\n\\n## Visual Review Findings\\n" + phase1_results,
    model="sonnet"
)
```

### Phase 3: Live Testing (after Phase 2)
```python
Task(
    subagent_type="general-purpose",
    description="Design Reviewer: Live testing",
    prompt=phases[2]["agents"][0]["task_prompt"] + "\\n\\n## All Previous Findings\\n" + all_findings,
    model="sonnet"
)
```

### Review Outputs
Each agent produces:
- Detailed findings with priorities (Blocker/High/Medium/Nitpick)
- Screenshots (from Playwright)
- Specific recommendations
- Verdict (APPROVED/NEEDS_CHANGES/BLOCKED)

### Final Report Structure
```
# Frontend Review Report

## Summary
- URL: {url}
- Codebase: {codebase_path}
- Date: [timestamp]

## Beauty Score: X/10 (Graphic Designer)
## Accessibility: PASS/FAIL (UX Engineer)
## Code Quality: APPROVED/CHANGES_REQUESTED (Frontend Engineer)
## Live Testing: APPROVED/BLOCKED (Design Reviewer)

## Issues by Priority
### Blockers (must fix)
### High Priority (fix before merge)
### Medium Priority (follow-up)
### Nitpicks (optional)

## Screenshots
[Attached from Playwright testing]

## Final Verdict
SHIP IT / NEEDS WORK / BLOCKED
```
""",
            "triage_matrix": {
                "BLOCKER": "Critical issues that prevent release - broken functionality, security issues, major accessibility failures",
                "HIGH": "Significant issues to fix before merge - poor UX, performance problems, code quality concerns",
                "MEDIUM": "Should fix in follow-up PR - minor UX improvements, code refactoring opportunities",
                "NITPICK": "Optional polish - minor visual tweaks, preference-based suggestions"
            }
        }

    elif tool_name == "get_compliance_rules":
        from orchestrator.compliance import COMPLIANCE_RULES

        rules_formatted = []
        for rule in COMPLIANCE_RULES:
            rules_formatted.append({
                "code": rule["code"],
                "category": rule["category"].value,
                "severity": rule["severity"].value,
                "description": rule["description"],
                "remediation": rule["remediation"],
                "standard": rule.get("standard", ""),
                "file_types": rule["file_types"],
            })

        return {
            "rules": rules_formatted,
            "total_rules": len(rules_formatted),
            "categories": [
                "ISSAS/Data Destruction",
                "ISSAS/Encryption",
                "Privacy/PII",
                "Access Control",
                "AI/ADMT Governance",
                "Agency Operations",
                "Data Exchange/ACORD",
            ],
            "usage": """
## Allstate/ISSAS Compliance Rules

These rules are automatically checked by the `check_allstate_compliance` tool.
For InsurTech projects in the Allstate ecosystem, run compliance checks before deployment.

### Critical Rules (Must Fix)
- SEC-01: No soft deletes for PII (use crypto-shredding)
- SEC-02: Use AES-256-GCM, FIPS 140-2/3 validated encryption
- SEC-05: No hardcoded secrets in source code
- AI-01: Zero Data Retention for LLM calls
- AI-02: Human-in-the-loop for automated decisions

### Available Agents
Use the `allstate_compliance` agent for full compliance review, or
`insurance_backend` agent for compliance-aware backend development.
"""
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
