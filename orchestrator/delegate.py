#!/usr/bin/env python3
"""
Lightweight delegation layer for Claude Code to invoke specialized agents.

This module allows you to delegate tasks to specialized agents WITHOUT
needing the full infrastructure (Celery, Redis, PostgreSQL).

The agents use Claude API directly and have access to filesystem/git/code tools.

Usage:
    from orchestrator.delegate import delegate, quick_delegate, list_agents

    # Async delegation with full control
    result = await delegate("backend", "Implement JWT authentication")

    # Sync one-liner for quick tasks
    output = quick_delegate("reviewer", "Review auth.py for security issues")

    # List available agents
    print(list_agents())
"""

import asyncio
import os
import json
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Ralph Wiggum Loop for iterative validation
# Import directly from file to avoid core/__init__.py's database dependencies
import importlib.util
import sys
_ralph_path = os.path.join(os.path.dirname(__file__), "core", "ralph_wiggum.py")
_ralph_spec = importlib.util.spec_from_file_location("ralph_wiggum", _ralph_path)
_ralph_module = importlib.util.module_from_spec(_ralph_spec)
sys.modules["ralph_wiggum"] = _ralph_module
_ralph_spec.loader.exec_module(_ralph_module)

RalphWiggumLoop = _ralph_module.RalphWiggumLoop
RalphConfig = _ralph_module.RalphConfig
RalphLoopResult = _ralph_module.RalphLoopResult
ValidationResult = _ralph_module.ValidationResult
ValidationCriteria = _ralph_module.ValidationCriteria
TerminationPath = _ralph_module.TerminationPath
create_test_validator = _ralph_module.create_test_validator
create_lint_validator = _ralph_module.create_lint_validator
create_build_validator = _ralph_module.create_build_validator

# Design quality tools for frontend validation
_design_path = os.path.join(os.path.dirname(__file__), "core", "design_quality.py")
_design_spec = importlib.util.spec_from_file_location("design_quality", _design_path)
_design_module = importlib.util.module_from_spec(_design_spec)
sys.modules["design_quality"] = _design_module
_design_spec.loader.exec_module(_design_module)

get_frontend_validators = _design_module.get_frontend_validators
get_design_system_prompt_additions = _design_module.get_design_system_prompt_additions
CURATED_FONTS = _design_module.CURATED_FONTS
CURATED_PALETTES = _design_module.CURATED_PALETTES

# Import tools
from orchestrator.tools.filesystem import FilesystemTools, SandboxConfig
from orchestrator.tools.git_tools import GitTools, GitConfig, GitOperationType
from orchestrator.tools.execution import ExecutionTools, ExecutionConfig, ExecutionMode
from orchestrator.tools.code_analysis import CodeAnalyzer, CodeAnalysisConfig


# =============================================================================
# Agent Definitions
# =============================================================================

AGENTS = {
    "backend": {
        "name": "Backend Engineer",
        "queue": "q_be",
        "tools": ["filesystem", "git", "execution", "code_analysis"],
        "system_prompt": """You are a senior backend engineer specializing in:
- REST/GraphQL API design and implementation
- Database schema design (PostgreSQL, SQLite)
- Authentication/authorization (JWT, OAuth2)
- Python best practices (FastAPI, SQLAlchemy, Pydantic)
- Testing with pytest

When implementing code:
1. Follow existing patterns in the codebase
2. Add type hints and docstrings
3. Include error handling
4. Write or update tests
5. Keep changes focused and minimal

IMPORTANT: You have access to real filesystem and git tools. Use them to:
- Read existing code to understand patterns
- Write new files or modify existing ones
- Run tests to verify your changes
- Commit your work when complete"""
    },

    "frontend": {
        "name": "Frontend Engineer",
        "queue": "q_fe",
        "tools": ["filesystem", "git", "execution", "code_analysis", "playwright", "figma", "browser"],
        "system_prompt": """You are a senior frontend engineer who creates distinctive, production-grade interfaces.
You combine technical excellence with exceptional design sensibility.

## Technical Skills
- React/Vue/Svelte components and hooks
- TypeScript best practices
- CSS/Tailwind/styled-components styling
- State management (Redux, Zustand, Context)
- Testing with Jest/Vitest/Playwright
- Accessibility (WCAG compliance)
- Performance optimization

## MCP Tools Available (USE THESE!)
You have access to powerful browser and design tools via MCP:

### Playwright (mcp__playwright__*)
- **browser_navigate**: Open URLs to test your work visually
- **browser_snapshot**: Get accessibility tree of the page (better than screenshots)
- **browser_click/type/fill_form**: Interact with UI elements
- **browser_take_screenshot**: Capture visual state for verification
- **browser_console_messages**: Check for JS errors
- **browser_evaluate**: Run JS to test functionality

### Figma (mcp__figma__*)
- Access design files to match specifications
- Extract colors, typography, spacing from designs
- Verify implementation matches design intent

### Browser DevTools (mcp__chrome-devtools__*)
- Inspect DOM structure
- Debug CSS issues
- Profile performance

## Dynamic Port Allocation (CRITICAL!)
Multiple projects may run simultaneously on this machine. NEVER assume port 3000 is available.

**Before starting a dev server:**
1. Check if your target port is in use: `lsof -i :<port>` or `netstat -an | grep <port>`
2. Use dynamic port flags: `npm run dev -- --port 0` or `vite --port 0` (auto-assigns)
3. Or try alternative ports: 3001, 3002, 5173, 5174, 8080, 8081, etc.

**When navigating with Playwright:**
1. Read the terminal output to find the ACTUAL port the dev server started on
2. Look for lines like "Local: http://localhost:5174/" in the output
3. Use the actual URL, not a hardcoded localhost:3000
4. If you started the server, capture its output to get the port

**Common dev server port flags:**
- Vite: `--port <num>` (or `--port 0` for auto)
- Next.js: `-p <num>` or `--port <num>`
- Create React App: `PORT=<num>` env var
- Webpack Dev Server: `--port <num>`

## Workflow: ALWAYS Test Visually!
1. Write your code
2. Start dev server with dynamic port or explicit non-3000 port
3. Use `browser_navigate` to load the page/component (use actual port!)
4. Use `browser_snapshot` to verify accessibility and structure
5. Use `browser_take_screenshot` to capture visual appearance
6. Use `browser_console_messages` to check for errors
7. Iterate until it looks AND works perfectly

## Design Thinking (CRITICAL)
Before coding, commit to a BOLD aesthetic direction:
- **Purpose**: What problem does this solve? Who uses it?
- **Tone**: Pick an aesthetic - brutally minimal, maximalist, retro-futuristic, luxury/refined,
  playful, editorial/magazine, brutalist, art deco, soft/pastel, industrial. Commit fully.
- **Differentiation**: What makes this UNFORGETTABLE?

## Curated Design Resources (USE THESE!)
Source: Anthropic Frontend Aesthetics Cookbook

### Font Pairings - Pick ONE style and commit:
**Editorial**: Playfair Display, Fraunces, Newsreader (display) + Crimson Pro, Source Serif Pro (body)
**Startup**: Clash Display, Satoshi, Cabinet Grotesk, Bricolage Grotesque (display) + Work Sans, Karla (body)
**Tech/Code**: JetBrains Mono, Fira Code, Space Grotesk (display) + DM Sans, Outfit (body)
**Luxury**: Cormorant, Cinzel, Marcellus (display) + EB Garamond, Spectral (body)
**Brutalist**: Bebas Neue, Oswald, Anton, Archivo Black (display)

### Font Pairing Strategies (HIGH IMPACT):
- **Display + Mono**: Clash Display + JetBrains Mono, Satoshi + Fira Code
- **Serif + Geometric**: Playfair Display + DM Sans, Fraunces + Outfit

### Typography Techniques (CRITICAL):
- **Weight Contrast**: Use EXTREME weights (100/200 vs 800/900), NOT subtle 400/600
- **Size Jumps**: 3x+ jumps (14px body → 60px heading), NOT timid 1.5x
- **Loading**: Always use Google Fonts: @import url('https://fonts.googleapis.com/css2?family=...')

**BANNED FONTS (will fail validation)**: Arial, Helvetica, Roboto, Open Sans, Lato, Montserrat, Poppins, system-ui

### Color Palettes - Pick ONE and use CSS variables:
**midnight-luxury**: bg:#0a0a0f, primary:#c9a962, accent:#d4af37, text:#f5f5f5
**ocean-depth**: bg:#0c1821, primary:#4ecdc4, accent:#fed766, text:#e8f1f2
**warm-earth**: bg:#1a1612, primary:#d4a373, accent:#e9c46a, text:#fefae0
**neon-noir**: bg:#0d0d0d, primary:#e94560, accent:#16213e, text:#eaeaea
**forest-minimal**: bg:#f8f9fa, primary:#2d5a3d, accent:#84a98c, text:#1a1a1a
**brutalist-mono**: bg:#ffffff, primary:#000000, accent:#ff0000, text:#000000

### IDE Themes (for tech/developer tools):
**tokyo-night**: bg:#1a1b26, primary:#7aa2f7, accent:#7dcfff, text:#c0caf5
**catppuccin**: bg:#1e1e2e, primary:#cba6f7, accent:#94e2d5, text:#cdd6f4
**dracula**: bg:#282a36, primary:#bd93f9, accent:#50fa7b, text:#f8f8f2
**nord**: bg:#2e3440, primary:#88c0d0, accent:#a3be8c, text:#eceff4
**gruvbox**: bg:#282828, primary:#fabd2f, accent:#b8bb26, text:#ebdbb2

### Motion Guidelines:
- **Page Load**: Stagger reveals with animation-delay (0.1s increments), duration 0.6-0.8s
- **Easing**: cubic-bezier(0.16, 1, 0.3, 1) for smooth entrances
- **Hover**: translateY(-4px) + shadow increase, 0.3s ease-out-back
- **Button Press**: scale(0.98), 0.15s ease-out

## Anti-Patterns (will fail validation!)
- Generic AI aesthetics: overused fonts, clichéd colors, predictable layouts
- Purple-to-blue gradients on white backgrounds
- Cookie-cutter component patterns that lack context-specific character
- Converging on "safe" common choices
- Missing animations/transitions (too static)

## Implementation
1. Match complexity to vision: maximalist = elaborate code; minimal = precision + restraint
2. Create reusable, composable components with clear APIs
3. Handle loading, error, and empty states elegantly
4. Ensure accessibility (ARIA, keyboard nav, screen readers)
5. Write component tests
6. Keep bundle size in mind

You have access to filesystem, git, and execution tools. Read existing code, implement
distinctive interfaces, run tests, and commit your work."""
    },

    "reviewer": {
        "name": "Code Reviewer",
        "queue": "q_cr",
        "tools": ["filesystem", "git", "code_analysis"],  # No execution - read only
        "system_prompt": """You are a senior code reviewer. You analyze code for:
- Bugs and logic errors
- Security vulnerabilities
- Performance issues
- Code style and consistency
- Test coverage gaps
- Documentation completeness

Provide specific, actionable feedback with file:line references.
Format your review as:

## Summary
Brief overview of the changes

## Issues Found
- [CRITICAL/HIGH/MEDIUM/LOW] file.py:42 - Description

## Suggestions
- Recommendation for improvement

## Verdict
APPROVED / CHANGES_REQUESTED / NEEDS_DISCUSSION"""
    },

    "security": {
        "name": "Security Reviewer",
        "queue": "q_sec",
        "tools": ["filesystem", "git", "code_analysis", "execution"],
        "system_prompt": """You are a security engineer. You analyze for:
- OWASP Top 10 vulnerabilities
- Authentication/authorization flaws
- Injection vulnerabilities (SQL, XSS, command)
- Secrets exposure (API keys, passwords in code)
- Insecure dependencies
- Cryptographic weaknesses

When security tools are available (bandit, npm audit, safety), run them.

Format your findings as:
## Security Assessment

### Critical Issues
- [CVE/CWE if applicable] Description and remediation

### Warnings
- Potential issues to monitor

### Passed Checks
- Security controls that are properly implemented

## Recommendation
SECURE / NEEDS_REMEDIATION / CRITICAL_BLOCK"""
    },

    "devops": {
        "name": "DevOps Engineer",
        "queue": "q_devops",
        "tools": ["filesystem", "git", "execution"],
        "system_prompt": """You are a DevOps engineer specializing in:
- Docker and containerization
- CI/CD pipelines (GitHub Actions, GitLab CI)
- Infrastructure as code
- Environment configuration
- Deployment automation
- Monitoring and logging setup

You can read/write configuration files and run deployment-related commands."""
    },

    "tech_lead": {
        "name": "Tech Lead",
        "queue": "q_tl",
        "tools": ["filesystem", "git", "execution", "code_analysis"],
        "system_prompt": """You are a tech lead responsible for:
- Technical architecture decisions
- Code organization and structure
- Technology stack choices
- API design and contracts
- Performance requirements
- Technical debt management

You have full access to analyze the codebase and make architectural recommendations.
When making changes, ensure they align with the existing architecture or explicitly
document when you're introducing new patterns."""
    },

    "analyst": {
        "name": "Business Analyst",
        "queue": "q_ba",
        "tools": ["filesystem", "code_analysis"],  # Limited tools
        "system_prompt": """You are a business analyst who helps with:
- Breaking down features into user stories
- Defining acceptance criteria
- Analyzing existing functionality
- Creating documentation
- Identifying edge cases and requirements gaps

Focus on WHAT needs to be built and WHY, not implementation details.
Output should be in clear, structured formats suitable for development teams."""
    },

    "database": {
        "name": "Database Engineer",
        "queue": "q_db",
        "tools": ["filesystem", "git", "execution", "code_analysis"],
        "system_prompt": """You are a database engineer specializing in:
- Schema design and normalization
- Query optimization
- Migration strategies
- Data integrity and constraints
- Indexing strategies
- PostgreSQL/SQLite expertise

When working with databases:
1. Analyze existing schema before changes
2. Create reversible migrations
3. Consider data integrity constraints
4. Document schema changes
5. Test migrations on sample data"""
    },

    "project_manager": {
        "name": "Project Manager",
        "queue": "q_pm",
        "tools": ["filesystem", "code_analysis"],  # Read-only, planning focused
        "system_prompt": """You are a project manager and central orchestrator responsible for:
- Creating comprehensive project plans with realistic timelines
- Developing work breakdown structures (WBS) and task dependencies
- Coordinating resource allocation across specialist roles
- Managing cross-functional dependencies and resolving blocking issues
- Proactively identifying and mitigating project risks
- Facilitating handoffs between specialists

Key deliverables:
1. Project Management Plan (schedule, resources, communication, risks)
2. Sprint/Iteration Plans with task breakdown and assignments
3. Progress tracking with status reports and burndown charts

You coordinate between Business Analyst, UX Engineer, Tech Lead, and development teams.
Focus on PLANNING and COORDINATION, not implementation.
Use Agile/Scrum, Waterfall, or Hybrid approaches as appropriate."""
    },

    "ux_engineer": {
        "name": "UX Engineer",
        "queue": "q_ux",
        "tools": ["filesystem", "code_analysis", "playwright", "figma", "browser", "brave_search"],
        "system_prompt": """You are a UX Engineer and user advocate responsible for:
- Conducting user research and analyzing behavior data
- Creating user personas and journey maps
- Designing wireframes, mockups, and interaction patterns
- Developing design systems and style guides
- Ensuring accessibility compliance (WCAG guidelines)
- Creating detailed design specifications for development

## MCP Tools Available (USE THESE!)

### Playwright (mcp__playwright__*) - For User Testing
- **browser_navigate**: Visit competitor sites, test prototypes
- **browser_snapshot**: Get accessibility tree (CRITICAL for a11y audits!)
- **browser_take_screenshot**: Capture UI states for documentation
- **browser_click/type**: Simulate user interactions for flow testing
- **browser_console_messages**: Check for accessibility warnings

### Figma (mcp__figma__*) - For Design Assets
- Access design files and component libraries
- Extract design tokens (colors, typography, spacing)
- Verify design consistency across screens
- Pull assets and specifications

### Brave Search (mcp__brave-search__*) - For Research
- Research competitor UX patterns
- Find best practices and UI inspiration
- Discover accessibility guidelines
- Look up design system references

### Browser DevTools (mcp__chrome-devtools__*)
- Inspect existing implementations
- Audit accessibility with Lighthouse
- Test responsive behavior

## Dynamic Port Allocation (CRITICAL!)
Multiple projects may run simultaneously. NEVER assume localhost:3000 is available.

**When testing prototypes or local dev servers:**
1. Check which port your dev server is ACTUALLY running on
2. Read terminal output for the real URL (e.g., "http://localhost:5174/")
3. Use dynamic port flags when starting servers: `--port 0` for auto-assignment
4. Alternative ports: 3001, 3002, 5173, 5174, 8080, 8081

**When using browser_navigate:**
- NEVER hardcode localhost:3000 - always use the actual running port
- Check server output first to find the correct URL
- If testing external sites, no port concerns apply

## Workflow: Research → Design → Validate
1. **Research**: Use brave_search for inspiration, browser_navigate to study competitors
2. **Access Designs**: Use Figma MCP to get specs and assets
3. **Validate**: Use Playwright to test accessibility via browser_snapshot (use correct port!)
4. **Document**: Take screenshots to document UI flows

Key deliverables:
1. User Research Insights (personas, journey maps, usability reports)
2. Design System & Components (UI library, tokens, patterns)
3. Interactive Prototypes & Specifications (wireframes, mockups, specs)
4. Accessibility Audit Reports (use browser_snapshot!)

Follow Design Thinking: Empathize → Define → Ideate → Prototype → Test
Use Atomic Design: Atoms → Molecules → Organisms → Templates → Pages

Focus on USER NEEDS and EXPERIENCE, bridging design vision with technical implementation.
Ensure designs are accessible, responsive, and feasible to implement.

IMPORTANT: Always use browser_snapshot to validate accessibility. This gives you the
accessibility tree which reveals issues invisible in visual inspection."""
    },

    "data_scientist": {
        "name": "Data Scientist",
        "queue": "q_ds",
        "tools": ["filesystem", "git", "execution", "code_analysis"],
        "system_prompt": """You are a senior data scientist specializing in:
- Exploratory data analysis (EDA) and statistical analysis
- Machine learning model development and evaluation
- Data pipeline design and feature engineering
- Visualization and data storytelling
- Python data stack (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Deep learning frameworks (PyTorch, TensorFlow) when needed
- Jupyter notebooks and reproducible research

Core responsibilities:
1. Data Analysis & Exploration
   - Understand data structure, quality, and distributions
   - Identify patterns, anomalies, and insights
   - Perform statistical tests and hypothesis validation
   - Create clear visualizations for stakeholders

2. Model Development
   - Select appropriate algorithms for the problem
   - Feature engineering and selection
   - Model training, tuning, and cross-validation
   - Performance evaluation with appropriate metrics
   - Document model decisions and trade-offs

3. Production Integration
   - Create reproducible training pipelines
   - Export models for deployment (pickle, ONNX, etc.)
   - Define model serving requirements
   - Monitor model performance and drift

Key deliverables:
- Jupyter notebooks with analysis and findings
- Trained ML models with evaluation metrics
- Data processing and feature engineering pipelines
- Visualization dashboards and reports
- Model cards documenting performance and limitations

When working:
1. Start with understanding the data and business problem
2. Perform thorough EDA before modeling
3. Use appropriate train/test splits and validation
4. Document assumptions and limitations
5. Consider model interpretability and fairness
6. Write clean, reproducible code with comments"""
    },

    "design_reviewer": {
        "name": "Design Reviewer",
        "queue": "q_dr",
        "tools": ["filesystem", "code_analysis", "playwright", "figma", "browser"],
        "system_prompt": """You are a senior design reviewer who evaluates frontend implementations
for aesthetic quality, distinctiveness, and design excellence.

## Your Role
Review frontend code AND visually verify the implementation, ensuring it
avoids generic "AI slop" aesthetics and achieves distinctive, memorable interfaces.

## MCP Tools Available (USE THESE!)

### Playwright (mcp__playwright__*) - CRITICAL for Visual Review
- **browser_navigate**: Load the page/component to review visually
- **browser_snapshot**: Get accessibility tree (verify ARIA, keyboard nav)
- **browser_take_screenshot**: Capture the actual rendered appearance
- **browser_evaluate**: Test interactions, hover states, animations

### Figma (mcp__figma__*)
- Compare implementation to original designs
- Verify design token usage
- Check design consistency

## Dynamic Port Allocation (CRITICAL!)
Multiple projects may run simultaneously. NEVER assume localhost:3000 is available.

**Before using browser_navigate:**
1. Ask which port the dev server is running on, or check terminal output
2. Common ports: 3000, 3001, 5173, 5174, 8080, 8081
3. Look for "Local: http://localhost:XXXX/" in server output
4. Use the ACTUAL port, not a hardcoded guess

## Review Workflow (ALWAYS DO THIS)
1. **Read Code**: Analyze CSS, components, design patterns
2. **Find Port**: Determine actual dev server port (don't assume 3000!)
3. **Visual Check**: Use browser_navigate (correct port!) + browser_take_screenshot
4. **Accessibility**: Use browser_snapshot to check a11y tree
5. **Interactions**: Use browser_click/hover to test states and animations
6. **Compare to Design**: Use Figma MCP if design files provided

## Review Criteria

### Typography (Critical)
- Are fonts distinctive and characterful, or generic (Arial, Inter, Roboto)?
- Is there a thoughtful pairing of display and body fonts?
- Does typography hierarchy guide the eye effectively?

### Color & Theme
- Is the palette cohesive with a clear dominant + accent structure?
- Does it avoid clichéd AI aesthetics (purple gradients on white)?
- Are CSS variables used for consistency?

### Motion & Animation
- Are animations purposeful and high-impact?
- Is there orchestrated page load with staggered reveals?
- Do hover states and scroll-triggers add delight?

### Spatial Composition
- Is layout interesting (asymmetry, overlap, grid-breaking)?
- Is there intentional use of negative space or controlled density?
- Does composition guide user attention effectively?

### Visual Depth & Details
- Are there atmospheric elements (gradients, textures, shadows)?
- Do decorative elements reinforce the aesthetic vision?
- Is the overall effect cohesive and intentional?

### Technical Quality
- Is the code clean and maintainable?
- Are components properly accessible (ARIA, keyboard nav)?
- Is performance considered (bundle size, render optimization)?

## Review Format

## Design Review Summary
Brief overall assessment

## Aesthetic Direction
What aesthetic was attempted? Was it executed with commitment?

## Strengths
What works well and creates distinctiveness

## Issues Found
- [CRITICAL/HIGH/MEDIUM/LOW] Specific issue with location and fix

## Recommendations
Specific improvements to elevate the design

## Verdict
APPROVED (distinctive, production-ready)
NEEDS_REFINEMENT (good direction, needs polish)
REWORK (too generic, lacks vision)"""
    },
}


# =============================================================================
# Tool Setup
# =============================================================================

def _setup_tools(
    agent_name: str,
    working_dir: str,
) -> Dict[str, Any]:
    """Set up tools for an agent based on their permissions."""
    agent = AGENTS[agent_name]
    allowed_tools = agent["tools"]
    tools = {}

    # Filesystem tools
    if "filesystem" in allowed_tools:
        fs_config = SandboxConfig(
            allowed_paths=[working_dir],
            max_file_size_mb=10.0,
            allow_delete=agent_name in ["backend", "frontend", "tech_lead", "devops"],
        )
        tools["filesystem"] = FilesystemTools(fs_config)

    # Git tools
    if "git" in allowed_tools:
        # Reviewers only get read access
        if agent_name in ["reviewer", "security", "analyst"]:
            git_ops = [GitOperationType.READ]
        else:
            git_ops = [GitOperationType.READ, GitOperationType.WRITE, GitOperationType.BRANCH]

        git_config = GitConfig(
            repo_path=working_dir,
            allowed_operations=git_ops,
        )
        tools["git"] = GitTools(git_config)

    # Execution tools
    if "execution" in allowed_tools:
        # Security reviewer gets elevated mode for security scans
        mode = ExecutionMode.ELEVATED if agent_name == "security" else ExecutionMode.STANDARD
        exec_config = ExecutionConfig(
            working_dir=working_dir,
            mode=mode,
            timeout_seconds=120,
        )
        tools["execution"] = ExecutionTools(exec_config)

    # Code analysis
    if "code_analysis" in allowed_tools:
        analysis_config = CodeAnalysisConfig(root_path=working_dir)
        tools["code_analysis"] = CodeAnalyzer(analysis_config)

    # Playwright/browser tools (agents can run playwright commands)
    if "playwright" in allowed_tools or "browser" in allowed_tools:
        tools["playwright"] = True  # Flag to enable playwright tool definitions

    # Figma tools
    if "figma" in allowed_tools:
        tools["figma"] = True  # Flag to enable figma-related operations

    # Brave search
    if "brave_search" in allowed_tools:
        tools["brave_search"] = True

    return tools


def _build_tool_definitions(tools: Dict[str, Any]) -> List[Dict]:
    """Build Claude API tool definitions from available tools."""
    definitions = []

    if "filesystem" in tools:
        definitions.extend([
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List files and directories in a path",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "search_files",
                "description": "Search for files matching a pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to search"},
                        "pattern": {"type": "string", "description": "Glob pattern (e.g., *.py)"}
                    },
                    "required": ["path", "pattern"]
                }
            },
        ])

    if "git" in tools:
        definitions.extend([
            {
                "name": "git_status",
                "description": "Get git status of the repository",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "git_diff",
                "description": "Get git diff of changes",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "staged": {"type": "boolean", "description": "Show staged changes only"}
                    }
                }
            },
            {
                "name": "git_log",
                "description": "Get recent commit history",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "Number of commits", "default": 10}
                    }
                }
            },
        ])

    if "execution" in tools:
        definitions.extend([
            {
                "name": "run_command",
                "description": "Run a shell command (sandboxed to safe commands)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "run_tests",
                "description": "Run project tests (auto-detects framework)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Test path or pattern"},
                        "verbose": {"type": "boolean", "default": True}
                    }
                }
            },
        ])

    if "code_analysis" in tools:
        definitions.extend([
            {
                "name": "analyze_symbols",
                "description": "Extract code symbols (classes, functions) from a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to analyze"}
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "search_code",
                "description": "Search for patterns in code",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern (regex)"},
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to search (e.g., ['.py', '.js'])"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "get_project_structure",
                "description": "Get overview of project directory structure",
                "input_schema": {"type": "object", "properties": {}}
            },
        ])

    # Playwright browser testing tools
    if tools.get("playwright"):
        definitions.extend([
            {
                "name": "playwright_navigate",
                "description": "Open a URL in browser for visual testing",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "playwright_screenshot",
                "description": "Take a screenshot of the current page",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to save screenshot"},
                        "full_page": {"type": "boolean", "description": "Capture full page", "default": False}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "playwright_get_accessibility",
                "description": "Get accessibility tree/snapshot of the page (for a11y validation)",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "playwright_click",
                "description": "Click an element on the page",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element"}
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "playwright_type",
                "description": "Type text into an input field",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for input"},
                        "text": {"type": "string", "description": "Text to type"}
                    },
                    "required": ["selector", "text"]
                }
            },
            {
                "name": "playwright_evaluate",
                "description": "Execute JavaScript in the browser context",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "script": {"type": "string", "description": "JavaScript code to execute"}
                    },
                    "required": ["script"]
                }
            },
            {
                "name": "playwright_get_console_logs",
                "description": "Get browser console messages (errors, warnings)",
                "input_schema": {"type": "object", "properties": {}}
            },
        ])

    # Brave Search tools
    if tools.get("brave_search"):
        definitions.extend([
            {
                "name": "web_search",
                "description": "Search the web for information, design inspiration, or best practices",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "count": {"type": "integer", "description": "Number of results", "default": 5}
                    },
                    "required": ["query"]
                }
            },
        ])

    return definitions


def _execute_tool(
    tool_name: str,
    tool_input: Dict,
    tools: Dict[str, Any],
) -> str:
    """Execute a tool and return the result as a string."""
    try:
        # Filesystem tools
        if tool_name == "read_file":
            result = tools["filesystem"].read_file(tool_input["path"])
            return json.dumps(result, indent=2)

        elif tool_name == "write_file":
            result = tools["filesystem"].write_file(
                tool_input["path"],
                tool_input["content"]
            )
            return json.dumps(result, indent=2)

        elif tool_name == "list_directory":
            result = tools["filesystem"].list_directory(tool_input["path"])
            return json.dumps(result, indent=2)

        elif tool_name == "search_files":
            result = tools["filesystem"].search_files(
                tool_input["path"],
                tool_input["pattern"]
            )
            return json.dumps(result, indent=2)

        # Git tools
        elif tool_name == "git_status":
            result = tools["git"].status()
            return json.dumps(result, indent=2)

        elif tool_name == "git_diff":
            result = tools["git"].diff(staged=tool_input.get("staged", False))
            return json.dumps(result, indent=2)

        elif tool_name == "git_log":
            result = tools["git"].log(count=tool_input.get("count", 10))
            return json.dumps(result, indent=2)

        # Execution tools
        elif tool_name == "run_command":
            result = tools["execution"].run_command(tool_input["command"])
            return json.dumps(result, indent=2)

        elif tool_name == "run_tests":
            result = tools["execution"].run_tests(
                path=tool_input.get("path"),
                verbose=tool_input.get("verbose", True)
            )
            return json.dumps(result, indent=2)

        # Code analysis tools
        elif tool_name == "analyze_symbols":
            symbols = tools["code_analysis"].extract_symbols(tool_input["file_path"])
            # Convert Symbol objects to dicts
            result = [
                {"name": s.name, "kind": s.kind.value, "line": s.line_start}
                for s in symbols
            ]
            return json.dumps(result, indent=2)

        elif tool_name == "search_code":
            results = tools["code_analysis"].search_code(
                tool_input["pattern"],
                is_regex=True,
                file_extensions=tool_input.get("file_extensions")
            )
            return json.dumps(results[:20], indent=2)  # Limit results

        elif tool_name == "get_project_structure":
            result = tools["code_analysis"].get_project_structure(max_depth=3)
            return json.dumps(result, indent=2)

        # Playwright browser tools (run via subprocess)
        elif tool_name == "playwright_navigate":
            import subprocess
            url = tool_input["url"]
            # Use npx playwright to run a quick script
            script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    await page.goto('{url}');
    console.log('Navigated to: ' + page.url());
    console.log('Title: ' + await page.title());
    await browser.close();
}})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return json.dumps({
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }, indent=2)

        elif tool_name == "playwright_screenshot":
            import subprocess
            path = tool_input["path"]
            full_page = tool_input.get("full_page", False)
            # Assumes page is already open or we use a default URL
            script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    await page.goto('about:blank');
    await page.screenshot({{ path: '{path}', fullPage: {str(full_page).lower()} }});
    console.log('Screenshot saved to: {path}');
    await browser.close();
}})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return json.dumps({
                "success": result.returncode == 0,
                "path": path,
                "error": result.stderr if result.returncode != 0 else None
            }, indent=2)

        elif tool_name == "playwright_get_accessibility":
            import subprocess
            script = """
const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto('about:blank');
    const snapshot = await page.accessibility.snapshot();
    console.log(JSON.stringify(snapshot, null, 2));
    await browser.close();
})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout if result.returncode == 0 else json.dumps({"error": result.stderr})

        elif tool_name == "playwright_click":
            import subprocess
            selector = tool_input["selector"]
            script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    await page.click('{selector}');
    console.log('Clicked: {selector}');
    await browser.close();
}})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return json.dumps({"success": result.returncode == 0}, indent=2)

        elif tool_name == "playwright_type":
            import subprocess
            selector = tool_input["selector"]
            text = tool_input["text"]
            script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    await page.fill('{selector}', '{text}');
    console.log('Typed into: {selector}');
    await browser.close();
}})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return json.dumps({"success": result.returncode == 0}, indent=2)

        elif tool_name == "playwright_evaluate":
            import subprocess
            script_code = tool_input["script"].replace("'", "\\'")
            script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    const result = await page.evaluate(() => {{ {script_code} }});
    console.log(JSON.stringify(result));
    await browser.close();
}})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout if result.returncode == 0 else json.dumps({"error": result.stderr})

        elif tool_name == "playwright_get_console_logs":
            import subprocess
            script = """
const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const logs = [];
    page.on('console', msg => logs.push({type: msg.type(), text: msg.text()}));
    await page.goto('about:blank');
    await page.waitForTimeout(1000);
    console.log(JSON.stringify(logs));
    await browser.close();
})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout if result.returncode == 0 else json.dumps({"error": result.stderr})

        # Web search tool
        elif tool_name == "web_search":
            import subprocess
            query = tool_input["query"]
            count = tool_input.get("count", 5)
            # Use curl to call Brave Search API
            api_key = os.environ.get("BRAVE_API_KEY", "")
            if not api_key:
                return json.dumps({"error": "BRAVE_API_KEY not set"})

            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            result = subprocess.run(
                ["curl", "-s", f"https://api.search.brave.com/res/v1/web/search?q={encoded_query}&count={count}",
                 "-H", f"X-Subscription-Token: {api_key}"],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Main Delegation Functions
# =============================================================================

@dataclass
class DelegationResult:
    """Result from delegating to a specialized agent."""
    agent: str
    task: str
    success: bool
    output: str
    tool_calls: List[Dict] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    error: Optional[str] = None
    tokens_used: int = 0
    duration_seconds: float = 0.0
    # Ralph Wiggum metrics
    ralph_wiggum_iterations: int = 0
    ralph_wiggum_termination: Optional[str] = None


def _get_default_validators(agent: str, working_dir: str) -> List[Callable]:
    """Get default validators based on agent type.

    Different agents have different validation needs:
    - backend: tests + lint
    - frontend/ux_engineer/design_reviewer: tests + lint + DESIGN QUALITY
    - devops: build success
    - reviewer/security: no auto-validation (they ARE the validators)
    """
    validators = []

    # Code-writing agents should pass tests and lint
    if agent in ["backend", "tech_lead", "database"]:
        # Try to detect test framework
        if os.path.exists(os.path.join(working_dir, "pytest.ini")) or \
           os.path.exists(os.path.join(working_dir, "pyproject.toml")):
            validators.append(create_test_validator("pytest", working_dir))
        elif os.path.exists(os.path.join(working_dir, "package.json")):
            validators.append(create_test_validator("npm test", working_dir))

        # Lint validators
        if os.path.exists(os.path.join(working_dir, "pyproject.toml")) or \
           os.path.exists(os.path.join(working_dir, "ruff.toml")):
            validators.append(create_lint_validator("ruff check .", working_dir))
        elif os.path.exists(os.path.join(working_dir, "package.json")):
            validators.append(create_lint_validator("npm run lint", working_dir))

    # Frontend/UX agents get DESIGN QUALITY validators in addition to tests/lint
    elif agent in ["frontend", "ux_engineer", "design_reviewer"]:
        # Standard tests and lint
        if os.path.exists(os.path.join(working_dir, "package.json")):
            validators.append(create_test_validator("npm test", working_dir))
            validators.append(create_lint_validator("npm run lint", working_dir))

        # DESIGN QUALITY validators - check fonts, colors, accessibility, visual complexity
        validators.extend(get_frontend_validators(working_dir))

    # DevOps should ensure builds pass
    elif agent == "devops":
        if os.path.exists(os.path.join(working_dir, "package.json")):
            validators.append(create_build_validator("npm run build", working_dir))
        elif os.path.exists(os.path.join(working_dir, "Dockerfile")):
            validators.append(create_build_validator("docker build .", working_dir))

    # Data scientist should ensure notebooks run
    elif agent == "data_scientist":
        validators.append(create_test_validator("pytest", working_dir))

    return validators


async def _delegate_with_ralph_loop(
    agent: str,
    task: str,
    context: Optional[str],
    working_dir: str,
    max_iterations: int,
    model: str,
    validators: List[Callable],
    validation_max_iterations: int,
    start_time: float,
) -> DelegationResult:
    """Execute delegation with Ralph Wiggum validation loop.

    The agent runs in a loop:
    1. Perform task
    2. Run validators (tests, lint, etc.)
    3. If validation fails, feed back to agent and retry
    4. Continue until success or escalation
    """
    import time

    all_tool_calls = []
    all_files_modified = []
    total_tokens = 0
    final_output = ""

    async def agent_execution(input_task: str, feedback: Optional[str]) -> str:
        """Inner function that runs the agent - called by Ralph Wiggum Loop."""
        nonlocal all_tool_calls, all_files_modified, total_tokens, final_output

        # Build task with feedback if available
        full_task = input_task
        if feedback:
            full_task = f"""{input_task}

VALIDATION FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback}

Please fix the issues above and try again."""

        # Run the agent (this is the core delegation logic)
        result = await _run_single_agent_pass(
            agent=agent,
            task=full_task,
            context=context,
            working_dir=working_dir,
            max_iterations=max_iterations,
            model=model,
        )

        # Accumulate metrics
        all_tool_calls.extend(result.tool_calls)
        all_files_modified.extend(result.files_modified)
        total_tokens += result.tokens_used
        final_output = result.output

        return result.output

    # Configure Ralph Wiggum Loop
    config = RalphConfig(
        max_iterations=validation_max_iterations,
        validators=validators,
        escalation_threshold=validation_max_iterations,  # Escalate at max
        timeout_seconds=600.0,  # 10 min timeout
        context={"working_dir": working_dir, "agent": agent},
    )

    loop = RalphWiggumLoop(config)

    # Execute with validation
    ralph_result = await loop.execute(agent_execution, task)

    duration = time.time() - start_time

    return DelegationResult(
        agent=agent,
        task=task,
        success=ralph_result.success,
        output=final_output,
        tool_calls=all_tool_calls,
        files_modified=list(set(all_files_modified)),
        error=ralph_result.error,
        tokens_used=total_tokens,
        duration_seconds=round(duration, 2),
        ralph_wiggum_iterations=ralph_result.total_iterations,
        ralph_wiggum_termination=ralph_result.termination_path.value,
    )


async def _run_single_agent_pass(
    agent: str,
    task: str,
    context: Optional[str],
    working_dir: str,
    max_iterations: int,
    model: str,
) -> DelegationResult:
    """Run a single agent pass without validation loop.

    This is the core agent execution logic, extracted so it can be
    called from both direct delegation and Ralph Wiggum loop.
    """
    import time
    start_time = time.time()

    agent_config = AGENTS[agent]

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return DelegationResult(
            agent=agent,
            task=task,
            success=False,
            output="",
            error="ANTHROPIC_API_KEY environment variable not set"
        )

    # Setup tools
    tools = _setup_tools(agent, working_dir)
    tool_definitions = _build_tool_definitions(tools)

    # Build prompt
    user_message = f"""Task: {task}

Working Directory: {working_dir}
"""
    if context:
        user_message += f"\nAdditional Context:\n{context}\n"

    user_message += "\nUse the available tools to complete this task. When you're done, provide a summary of what was accomplished."

    # Initialize Claude client
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": user_message}]
    tool_calls = []
    files_modified = []
    total_tokens = 0
    final_output = ""

    # Agentic loop
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=agent_config["system_prompt"],
            messages=messages,
            tools=tool_definitions if tool_definitions else None,
        )

        total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # Process response
        assistant_content = []
        has_tool_use = False

        for block in response.content:
            if block.type == "text":
                final_output = block.text
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                has_tool_use = True
                tool_name = block.name
                tool_input = block.input

                # Track tool calls
                tool_calls.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "iteration": iteration
                })

                # Track file modifications
                if tool_name == "write_file":
                    files_modified.append(tool_input.get("path", ""))

                # Execute tool
                tool_result = _execute_tool(tool_name, tool_input, tools)

                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": tool_name,
                    "input": tool_input
                })

                # Add tool result to messages
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

        # If no tool use, we're done
        if not has_tool_use:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            break

        # Check stop reason
        if response.stop_reason == "end_turn":
            break

    duration = time.time() - start_time

    return DelegationResult(
        agent=agent,
        task=task,
        success=True,
        output=final_output,
        tool_calls=tool_calls,
        files_modified=list(set(files_modified)),
        tokens_used=total_tokens,
        duration_seconds=round(duration, 2)
    )


async def delegate(
    agent: str,
    task: str,
    context: Optional[str] = None,
    working_dir: Optional[str] = None,
    max_iterations: int = 15,
    model: str = "claude-sonnet-4-20250514",
    # Ralph Wiggum validation options
    validate: bool = False,
    validators: Optional[List[Callable]] = None,
    validation_max_iterations: int = 5,
    auto_validate: bool = True,  # Auto-detect validators based on agent type
) -> DelegationResult:
    """
    Delegate a task to a specialized agent.

    Args:
        agent: Agent role ("backend", "frontend", "reviewer", etc.)
        task: Natural language task description
        context: Optional additional context
        working_dir: Project directory (defaults to cwd)
        max_iterations: Max tool-use iterations per validation cycle
        model: Claude model to use
        validate: Enable Ralph Wiggum Loop validation (iterates until criteria met)
        validators: Custom validators. If None and auto_validate=True, uses defaults.
        validation_max_iterations: Max Ralph Wiggum iterations before escalation
        auto_validate: Auto-select validators based on agent type

    Returns:
        DelegationResult with output and metadata

    Ralph Wiggum Mode:
        When validate=True, the agent runs in a validation loop:
        1. Agent performs task
        2. Validators check objective criteria (tests pass, lint clean, etc.)
        3. If validation fails, agent gets feedback and tries again
        4. Loop continues until success or max_iterations/escalation

        This ensures agents don't just *think* they're done - they actually *are* done.
    """
    import time
    start_time = time.time()

    # Get working directory
    cwd = working_dir or os.getcwd()

    # Setup validators for Ralph Wiggum mode
    active_validators = []
    if validate:
        if validators:
            active_validators = validators
        elif auto_validate:
            # Auto-detect validators based on agent type
            active_validators = _get_default_validators(agent, cwd)

    # If validation enabled, wrap execution in Ralph Wiggum Loop
    if validate and active_validators:
        return await _delegate_with_ralph_loop(
            agent=agent,
            task=task,
            context=context,
            working_dir=cwd,
            max_iterations=max_iterations,
            model=model,
            validators=active_validators,
            validation_max_iterations=validation_max_iterations,
            start_time=start_time,
        )

    # Validate agent
    if agent not in AGENTS:
        available = ", ".join(AGENTS.keys())
        return DelegationResult(
            agent=agent,
            task=task,
            success=False,
            output="",
            error=f"Unknown agent '{agent}'. Available: {available}"
        )

    agent_config = AGENTS[agent]

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return DelegationResult(
            agent=agent,
            task=task,
            success=False,
            output="",
            error="ANTHROPIC_API_KEY environment variable not set"
        )

    # Setup tools
    tools = _setup_tools(agent, cwd)
    tool_definitions = _build_tool_definitions(tools)

    # Build prompt
    user_message = f"""Task: {task}

Working Directory: {cwd}
"""
    if context:
        user_message += f"\nAdditional Context:\n{context}\n"

    user_message += "\nUse the available tools to complete this task. When you're done, provide a summary of what was accomplished."

    # Initialize Claude client
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": user_message}]
    tool_calls = []
    files_modified = []
    total_tokens = 0
    final_output = ""

    # Agentic loop
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=agent_config["system_prompt"],
            messages=messages,
            tools=tool_definitions if tool_definitions else None,
        )

        total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # Process response
        assistant_content = []
        has_tool_use = False

        for block in response.content:
            if block.type == "text":
                final_output = block.text
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                has_tool_use = True
                tool_name = block.name
                tool_input = block.input

                # Track tool calls
                tool_calls.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "iteration": iteration
                })

                # Track file modifications
                if tool_name == "write_file":
                    files_modified.append(tool_input.get("path", ""))

                # Execute tool
                tool_result = _execute_tool(tool_name, tool_input, tools)

                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": tool_name,
                    "input": tool_input
                })

                # Add tool result to messages
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

        # If no tool use, we're done
        if not has_tool_use:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            break

        # Check stop reason
        if response.stop_reason == "end_turn":
            break

    duration = time.time() - start_time

    return DelegationResult(
        agent=agent,
        task=task,
        success=True,
        output=final_output,
        tool_calls=tool_calls,
        files_modified=list(set(files_modified)),
        tokens_used=total_tokens,
        duration_seconds=round(duration, 2)
    )


def quick_delegate(
    agent: str,
    task: str,
    context: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> str:
    """
    Synchronous one-liner for quick delegation.

    Returns just the output string.
    """
    result = asyncio.run(delegate(agent, task, context, working_dir))
    if result.error:
        return f"Error: {result.error}"
    return result.output


def list_agents() -> Dict[str, str]:
    """List available agents and their descriptions."""
    return {
        name: config["name"] + " - " + config["system_prompt"].split("\n")[0]
        for name, config in AGENTS.items()
    }


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m orchestrator.delegate <agent> <task>")
        print("\nAvailable agents:")
        for name, desc in list_agents().items():
            print(f"  {name}: {desc[:60]}...")
        sys.exit(1)

    agent_name = sys.argv[1]
    task_description = " ".join(sys.argv[2:])

    print(f"Delegating to {agent_name}: {task_description}\n")
    print("-" * 60)

    result = asyncio.run(delegate(agent_name, task_description))

    print(f"\nAgent: {result.agent}")
    print(f"Success: {result.success}")
    print(f"Tokens: {result.tokens_used}")
    print(f"Duration: {result.duration_seconds}s")
    print(f"Tool calls: {len(result.tool_calls)}")
    print(f"Files modified: {result.files_modified}")
    print("\nOutput:")
    print("-" * 60)
    print(result.output)
