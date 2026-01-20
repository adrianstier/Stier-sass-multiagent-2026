"""
Frontend Review Workflow - Multi-Agent Design & Code Review

This workflow coordinates multiple specialized agents to perform a comprehensive
frontend review including visual design, UX, accessibility, code quality, and
live browser testing.

Usage:
    from orchestrator.workflows.frontend_review import FrontendReviewWorkflow

    # Execute via MCP
    workflow = FrontendReviewWorkflow(
        url="http://localhost:3000",
        codebase_path="/path/to/frontend",
        figma_url="https://figma.com/file/..."  # optional
    )
    results = await workflow.execute()
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ReviewPriority(Enum):
    """Issue priority levels for triage."""
    BLOCKER = "blocker"      # Critical - blocks release
    HIGH = "high"            # Must fix before merge
    MEDIUM = "medium"        # Should fix in follow-up
    NITPICK = "nitpick"      # Nice to have


class ReviewVerdict(Enum):
    """Final review verdicts."""
    APPROVED = "approved"
    NEEDS_CHANGES = "needs_changes"
    BLOCKED = "blocked"


@dataclass
class ReviewConfig:
    """Configuration for frontend review workflow."""
    # Required
    url: str                                    # Live URL to test
    codebase_path: str                          # Path to frontend code

    # Optional
    figma_url: Optional[str] = None             # Figma design file URL
    pr_number: Optional[int] = None             # GitHub PR number

    # Viewport configurations
    viewports: list[dict] = field(default_factory=lambda: [
        {"name": "desktop", "width": 1440, "height": 900},
        {"name": "tablet", "width": 768, "height": 1024},
        {"name": "mobile", "width": 375, "height": 812},
    ])

    # Review thresholds
    min_beauty_score: float = 7.0               # Minimum acceptable beauty score
    wcag_level: str = "AA"                      # WCAG compliance level
    lighthouse_threshold: int = 90              # Minimum Lighthouse score


# =============================================================================
# AGENT PROMPTS - Task-ready prompts for each reviewer role
# =============================================================================

GRAPHIC_DESIGNER_PROMPT = """
# Graphic Designer Review - Visual Beauty Assessment

You are a world-class graphic designer evaluating the aesthetic quality of a frontend implementation.

## Your Task
Review the frontend at `{url}` for visual beauty, typography, color harmony, and emotional impact.

## Review Process

### 1. Visual Inspection (USE PLAYWRIGHT!)
```
1. browser_navigate to {url}
2. browser_take_screenshot for desktop view
3. browser_resize to 768x1024, take tablet screenshot
4. browser_resize to 375x812, take mobile screenshot
4. browser_hover on interactive elements to see states
5. browser_evaluate to test animations
```

### 2. Beauty Criteria (Score each 1-10)

**Emotional Impact (25%)**
- Does it evoke an emotional response?
- Is it memorable or forgettable?
- Does it feel crafted or templated?

**Typography (20%)**
- Font selection quality (NO generic fonts: Arial, Helvetica, Roboto, Inter, system-ui)
- Hierarchy clarity (extreme weight contrast: 200 vs 800, not 400 vs 600)
- Size jumps (3x+ difference between body and display)
- Proper pairing of display + body fonts

**Color & Harmony (20%)**
- Palette coherence and sophistication
- Contrast for readability
- Mood alignment with purpose
- Avoids AI clichés (purple gradients on white)

**Composition (15%)**
- Visual balance (symmetric or dynamic asymmetric)
- Intentional whitespace usage
- Clear visual hierarchy and flow

**Motion & Animation (10%)**
- Page load animations (staggered reveals)
- Hover/interaction states
- Transition smoothness and purpose

**Polish & Details (10%)**
- Shadow/depth subtlety
- Icon/imagery consistency
- Overall cohesion

### 3. Output Format

```markdown
# Graphic Design Review

## First Impression
[Honest gut reaction - be specific]

## Beauty Score: X/10

## Category Scores
- Emotional Impact: X/10
- Typography: X/10
- Color & Harmony: X/10
- Composition: X/10
- Motion: X/10
- Polish: X/10

## Screenshots
[Reference the screenshots you captured]

## What's Working
- [Specific elements to preserve]

## Issues Found
- [{priority}] Issue description with file:line if applicable

## Verdict
BEAUTIFUL / GOOD / MEDIOCRE / NEEDS_WORK

## Path to Beautiful
[3 specific actionable improvements if not BEAUTIFUL]
```

## Working Directory: {codebase_path}
"""


UX_ENGINEER_PROMPT = """
# UX Engineer Review - User Experience Assessment

You are a senior UX Engineer assessing the user experience quality of a frontend implementation.

## Your Task
Review the frontend at `{url}` for usability, accessibility, and user flow quality.

## Review Process

### 1. User Flow Testing (USE PLAYWRIGHT!)
```
1. browser_navigate to {url}
2. browser_snapshot to get accessibility tree
3. Execute primary user flows by clicking through
4. Test all interactive states (hover, active, disabled, focus)
5. Verify destructive action confirmations
```

### 2. Accessibility Testing (WCAG 2.1 {wcag_level})

**Keyboard Navigation**
- Tab through all interactive elements
- Verify logical tab order
- Check visible focus states on EVERY element
- Test Enter/Space activation
- Test Escape for modals/dialogs

**Screen Reader Compatibility**
- Semantic HTML usage (proper headings, landmarks)
- ARIA labels where needed
- Form label associations
- Image alt text
- Live region announcements

**Visual Accessibility**
- Color contrast (4.5:1 minimum for text)
- No color-only information
- Text resizing support
- Reduced motion support

### 3. Usability Patterns

**Feedback & Affordances**
- Clear interactive element indicators
- Loading states for async operations
- Error state handling and messaging
- Success confirmations
- Progress indicators for multi-step flows

**Information Architecture**
- Logical content grouping
- Clear navigation patterns
- Breadcrumbs/location awareness
- Search functionality (if applicable)

**Form Usability**
- Clear labels and placeholders
- Inline validation with helpful messages
- Logical field ordering
- Appropriate input types

### 4. Output Format

```markdown
# UX Review

## Accessibility Compliance: {wcag_level}
- [PASS/FAIL] Keyboard Navigation
- [PASS/FAIL] Screen Reader Compatibility
- [PASS/FAIL] Color Contrast
- [PASS/FAIL] Focus Management

## User Flow Assessment
[Description of tested flows and findings]

## Issues Found
- [{priority}] Issue description
  - Location: [where in the UI]
  - Impact: [who is affected]
  - Recommendation: [how to fix]

## Usability Strengths
- [What works well]

## Verdict
ACCESSIBLE / NEEDS_REMEDIATION / BLOCKED

## Priority Fixes
[Top 3 issues to address immediately]
```

## Working Directory: {codebase_path}
"""


FRONTEND_ENGINEER_PROMPT = """
# Frontend Engineer Review - Code Quality Assessment

You are a senior Frontend Engineer reviewing the code quality of a frontend implementation.

## Your Task
Review the frontend code at `{codebase_path}` for architecture, patterns, and technical excellence.

## Review Process

### 1. Code Exploration
```
1. Glob for component files: **/*.{tsx,jsx,vue,svelte}
2. Glob for style files: **/*.{css,scss,styled.ts}
3. Read key components to understand architecture
4. Check for consistent patterns
```

### 2. Architecture Review

**Component Structure**
- Clear component hierarchy
- Appropriate component granularity
- Proper separation of concerns
- Consistent file organization

**State Management**
- Appropriate state location (local vs global)
- Clean data flow patterns
- Proper side effect handling
- Memoization where needed

**Styling Approach**
- Design token/CSS variable usage
- Responsive design implementation
- Style organization and consistency
- No magic numbers

### 3. Code Quality

**TypeScript/Type Safety**
- Proper type definitions
- No `any` types without justification
- Interface documentation

**Performance Patterns**
- Code splitting/lazy loading
- Image optimization
- Bundle size awareness
- Render optimization (memo, useMemo, useCallback)

**Testing**
- Component test coverage
- Interaction testing
- Accessibility testing
- Visual regression tests

**Error Handling**
- Error boundaries
- Loading states
- Empty states
- Graceful degradation

### 4. Anti-Patterns to Flag

- Prop drilling (use context/state management)
- Massive components (should be split)
- Inline styles without system
- Hardcoded strings (should be constants/i18n)
- Console.log statements
- Commented-out code
- TODO comments without tickets

### 5. Output Format

```markdown
# Frontend Code Review

## Architecture Overview
[Brief description of the codebase structure]

## Code Quality Score: X/10

## Strengths
- [Well-implemented patterns]

## Issues Found
- [{priority}] file:line - Issue description
  - Current: [what's wrong]
  - Recommended: [how to fix]

## Performance Concerns
- [Bundle size, render performance, etc.]

## Test Coverage Assessment
- [Current state and gaps]

## Verdict
APPROVED / CHANGES_REQUESTED / NEEDS_DISCUSSION

## Recommended Refactors
[Priority list of improvements]
```

## Working Directory: {codebase_path}
"""


DESIGN_REVIEWER_PROMPT = """
# Design Reviewer - Live Testing & Integration

You are the Design Reviewer conducting systematic live testing of the frontend.

## Your Task
Perform comprehensive live testing of `{url}` using Playwright, synthesizing findings from all reviewers.

## Review Process

### Phase 0: Setup
```
browser_navigate("{url}")
```

### Phase 1: Responsive Testing
```
# Desktop (1440x900)
browser_resize(1440, 900)
browser_take_screenshot("desktop.png")
browser_snapshot()  # Check accessibility tree

# Tablet (768x1024)
browser_resize(768, 1024)
browser_take_screenshot("tablet.png")

# Mobile (375x812)
browser_resize(375, 812)
browser_take_screenshot("mobile.png")
```

### Phase 2: Interaction Testing
```
# Test all interactive elements
browser_click on buttons, links, form controls
browser_type in input fields
browser_select_option in dropdowns
browser_hover on elements with hover states
```

### Phase 3: State Testing
```
# Loading states - trigger async actions
# Error states - submit invalid data
# Empty states - clear data if possible
# Success states - complete actions
```

### Phase 4: Accessibility Verification
```
browser_snapshot()  # Full accessibility tree
browser_press_key("Tab") repeatedly - verify focus order
browser_console_messages() - check for errors/warnings
```

### Phase 5: Console & Network
```
browser_console_messages(level="error")
browser_network_requests()
```

## Output Format

```markdown
# Design Review - Live Testing Report

## Test Environment
- URL: {url}
- Date: [timestamp]
- Browser: Chromium (Playwright)

## Viewport Screenshots
- Desktop (1440x900): [screenshot reference]
- Tablet (768x1024): [screenshot reference]
- Mobile (375x812): [screenshot reference]

## Responsive Behavior
- [PASS/FAIL] No horizontal scrolling
- [PASS/FAIL] No element overlap
- [PASS/FAIL] Touch targets adequate on mobile
- [PASS/FAIL] Text readable without zooming

## Interaction States
- [PASS/FAIL] Hover states present
- [PASS/FAIL] Active/pressed states
- [PASS/FAIL] Focus states visible
- [PASS/FAIL] Disabled states clear

## Accessibility Tree
[Key findings from browser_snapshot]

## Console Errors
[List any errors found]

## Issues Found

### Blockers
- [Issue + screenshot reference]

### High Priority
- [Issue + screenshot reference]

### Medium Priority
- [Issue]

### Nitpicks
- Nit: [Issue]

## Final Verdict
APPROVED / NEEDS_CHANGES / BLOCKED

## Summary
[Overall assessment integrating all review perspectives]
```

## Working Directory: {codebase_path}
"""


# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================

WORKFLOW_DEFINITION = {
    "name": "Frontend Review",
    "description": "Comprehensive multi-agent frontend review workflow",
    "phases": [
        {
            "phase": 1,
            "name": "Visual & UX Assessment",
            "parallel": True,
            "agents": [
                {
                    "agent": "graphic_designer",
                    "prompt_template": GRAPHIC_DESIGNER_PROMPT,
                    "model": "sonnet",  # Needs visual reasoning
                    "tools": ["playwright", "filesystem", "figma"]
                },
                {
                    "agent": "ux_engineer",
                    "prompt_template": UX_ENGINEER_PROMPT,
                    "model": "sonnet",
                    "tools": ["playwright", "filesystem"]
                }
            ]
        },
        {
            "phase": 2,
            "name": "Code Review",
            "parallel": False,
            "agents": [
                {
                    "agent": "frontend",
                    "prompt_template": FRONTEND_ENGINEER_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem", "code_analysis", "execution"]
                }
            ]
        },
        {
            "phase": 3,
            "name": "Live Testing & Synthesis",
            "parallel": False,
            "agents": [
                {
                    "agent": "design_reviewer",
                    "prompt_template": DESIGN_REVIEWER_PROMPT,
                    "model": "sonnet",
                    "tools": ["playwright", "filesystem"]
                }
            ]
        }
    ]
}


def get_frontend_review_prompts(
    url: str,
    codebase_path: str,
    wcag_level: str = "AA",
    figma_url: Optional[str] = None
) -> dict:
    """
    Generate all agent prompts for a frontend review.

    Returns a dict with prompt for each agent, ready for Task tool execution.
    """
    format_vars = {
        "url": url,
        "codebase_path": codebase_path,
        "wcag_level": wcag_level,
        "figma_url": figma_url or "Not provided"
    }

    return {
        "graphic_designer": GRAPHIC_DESIGNER_PROMPT.format(**format_vars),
        "ux_engineer": UX_ENGINEER_PROMPT.format(**format_vars),
        "frontend_engineer": FRONTEND_ENGINEER_PROMPT.format(**format_vars),
        "design_reviewer": DESIGN_REVIEWER_PROMPT.format(**format_vars),
    }


def get_execution_instructions() -> str:
    """Get instructions for executing this workflow in Claude Code."""
    return """
## Frontend Review Workflow - Execution Instructions

### Quick Start
To run a frontend review, execute agents in this order:

### Phase 1: Visual & UX Assessment (PARALLEL)
Launch these two agents simultaneously:

```python
# In Claude Code, send a single message with both Task calls:

Task(
    subagent_type="general-purpose",
    description="Graphic Designer Review",
    prompt=graphic_designer_prompt,
    model="sonnet"
)

Task(
    subagent_type="general-purpose",
    description="UX Engineer Review",
    prompt=ux_engineer_prompt,
    model="sonnet"
)
```

### Phase 2: Code Review (SEQUENTIAL)
After Phase 1 completes:

```python
Task(
    subagent_type="general-purpose",
    description="Frontend Code Review",
    prompt=frontend_engineer_prompt + f"\\n\\n## Context from Visual Review\\n{phase1_results}",
    model="sonnet"
)
```

### Phase 3: Live Testing & Synthesis (SEQUENTIAL)
After Phase 2 completes:

```python
Task(
    subagent_type="general-purpose",
    description="Design Review - Live Testing",
    prompt=design_reviewer_prompt + f"\\n\\n## Previous Review Findings\\n{all_previous_results}",
    model="sonnet"
)
```

### Using the Orchestrator MCP

Alternatively, use the orchestrator directly:

```python
# Start workflow tracking
mcp__orchestrator__start_workflow(
    task="Frontend review for {url}",
    working_directory="{codebase_path}"
)

# Execute each agent
mcp__orchestrator__execute_agent(
    agent="graphic_designer",
    task="Review visual design at {url}",
    working_directory="{codebase_path}"
)

# Update workflow after each agent
mcp__orchestrator__update_workflow(
    workflow_id=workflow_id,
    agent="graphic_designer",
    status="completed",
    summary="..."
)
```
"""
