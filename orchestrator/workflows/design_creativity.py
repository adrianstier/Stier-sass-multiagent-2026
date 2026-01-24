"""
Design & Creativity Workflow - Multi-Agent Beauty Enhancement

This workflow coordinates the full design creativity cluster to improve the
visual beauty, brand coherence, and emotional impact of SaaS products.

The cluster operates as an integrated creative team:
1. Brand Strategist establishes the identity foundation
2. Visual Designer + Illustration Specialist develop the visual language (parallel)
3. Motion Designer adds the kinetic layer
4. Content Designer crafts the product voice
5. Design Systems Architect codifies everything into tokens/components
6. Creative Director conducts final beauty assessment (quality gate)

Usage:
    from orchestrator.workflows.design_creativity import (
        get_creativity_workflow_prompts,
        CREATIVITY_WORKFLOW_DEFINITION,
    )

    # Get prompts for each agent
    prompts = get_creativity_workflow_prompts(
        product_name="MyApp",
        product_description="A project management tool for small teams",
        target_audience="Startup founders and small team leads",
        existing_url="http://localhost:3000",  # optional
        codebase_path="/path/to/frontend",    # optional
    )
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CreativityConfig:
    """Configuration for the design creativity workflow."""
    # Required
    product_name: str
    product_description: str
    target_audience: str

    # Optional - existing product context
    existing_url: Optional[str] = None
    codebase_path: Optional[str] = None
    competitor_urls: list[str] = field(default_factory=list)

    # Optional - constraints
    existing_brand_guidelines: Optional[str] = None
    tech_stack: Optional[str] = None
    timeline_pressure: str = "normal"  # relaxed, normal, tight

    # Quality thresholds
    min_beauty_score: float = 7.5
    require_dark_mode: bool = True
    require_motion: bool = True
    require_illustration: bool = True


# =============================================================================
# AGENT PROMPTS - Context-aware prompts for each creative role
# =============================================================================

BRAND_STRATEGIST_TASK_PROMPT = """
# Brand Strategy Task

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Target Audience**: {target_audience}
- **Competitors**: {competitors}
- **Existing Brand Guidelines**: {existing_guidelines}

## Your Mission
Establish the brand foundation that will guide all creative work for this SaaS product.
Your output will directly inform the Visual Designer, Motion Designer, Content Designer,
and Illustration Specialist.

## Deliverables
Using your system prompt framework, produce:
1. Brand Purpose & Positioning
2. Brand Personality (archetype + 3-5 traits)
3. Brand Voice Framework (dimensions, context adjustments, vocabulary)
4. Experience Principles (3-5 principles with design implications)
5. Competitive Differentiation Strategy
6. Creative Direction briefs for each downstream agent

## Critical Requirements
- The positioning must be SPECIFIC enough to exclude people
- Personality traits must be ACTIONABLE (not aspirational platitudes)
- Voice framework must include EXAMPLES for each context
- Creative direction must be DETAILED enough for each agent to work independently

Create the artifact as a comprehensive brand strategy document.
"""

VISUAL_DESIGNER_TASK_PROMPT = """
# Visual Design Task

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Target Audience**: {target_audience}
- **Tech Stack**: {tech_stack}
- **Dark Mode Required**: {require_dark_mode}

## Brand Direction (from Brand Strategist)
Use the brand strategy artifact to inform your visual decisions.
Translate the brand personality into a visual language.

## Your Mission
Create a complete visual design system specification that establishes the product's
distinctive visual identity. Focus on typography, color, spacing, shadows, and layout
that could ONLY belong to this product.

## Deliverables
1. Typography System (display + body fonts, full type scale, pairing rationale)
2. Color System (light + dark mode, semantic colors, tinted neutrals)
3. Spacing System (8-point grid, section spacing)
4. Shadow & Depth System (elevation levels for both modes)
5. Border & Radius System
6. Layout Principles (grid, breakpoints, compositions)

## Critical Requirements
- NO generic fonts (Inter, Arial, Helvetica, Roboto for display)
- NO pure white backgrounds (#FFFFFF)
- NO default Tailwind color palettes
- Dark mode must be a FIRST-CLASS design, not an inversion
- Typography weight contrast must be DRAMATIC (200 vs 800)
- The visual system must be DISTINCTIVE (recognizable without the logo)

{existing_context}

Create the artifact as a detailed visual design specification.
"""

MOTION_DESIGNER_TASK_PROMPT = """
# Motion Design Task

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Target Audience**: {target_audience}
- **Tech Stack**: {tech_stack}

## Visual Direction (from Visual Designer)
Use the visual design specification to ensure motion complements the visual language.

## Brand Direction (from Brand Strategist)
Use the brand personality to determine the motion personality
(snappy/fluid/playful/dramatic).

## Your Mission
Create a complete motion design specification that makes the interface feel alive,
responsive, and emotionally engaging -- without being distracting or busy.

## Deliverables
1. Motion Personality (which personality from the spectrum, with rationale)
2. Motion Tokens (durations, easings, stagger values as CSS custom properties)
3. Animation Inventory:
   - Page-level transitions
   - Component-level micro-interactions (buttons, cards, inputs, toggles)
   - Scroll-driven animations
   - Loading & progress states
   - Success/celebration moments
4. Stagger Sequences (which elements stagger, order, delay)
5. Reduced Motion Fallbacks (alternatives for EVERY animation)
6. Implementation Notes (CSS vs JS, performance guidelines)
7. Code Samples (ready for frontend engineer)

## Critical Requirements
- EVERY animation must serve a purpose (orient, focus, connect, feedback, delight)
- ONLY animate GPU-accelerated properties (transform, opacity)
- Reduced motion fallbacks for ALL animations
- Motion personality must be CONSISTENT throughout
- Stagger sequences max 7 elements
- No animation should feel sluggish OR jarring

Create the artifact as a motion design specification with code samples.
"""

CONTENT_DESIGNER_TASK_PROMPT = """
# Content Design Task

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Target Audience**: {target_audience}

## Brand Direction (from Brand Strategist)
Use the brand voice framework to inform all content decisions.
Your copy should sound like this brand's personality talking.

## Visual Direction (from Visual Designer)
Respect the visual hierarchy -- your copy lengths must work within the
established type scale and layout constraints.

## Your Mission
Craft every word that will appear inside this product. Write copy that guides
users clearly, reduces anxiety, and occasionally creates moments of delight.
The words ARE the interface.

## Deliverables
1. Voice Summary (product-specific alignment with brand voice)
2. Navigation & Wayfinding labels
3. Page Titles & Descriptions (every key screen)
4. Button & Action Labels (with rationale for each choice)
5. Empty States (heading, body, action for each context)
6. Error Messages (every scenario with recovery action)
7. Success Messages (confirmation + next step)
8. Loading States (contextual messages by duration threshold)
9. Tooltips & Help text
10. Confirmation Dialogs (title, body, buttons for destructive actions)
11. Onboarding Copy (first-run experience)
12. Content Patterns (reusable templates for the design system)
13. Glossary (product-specific terms)

## Critical Requirements
- NO "Submit", "Click here", "Error", "Invalid input"
- Error messages ALWAYS provide a path forward
- Button labels describe OUTCOMES, not processes
- Empty states are USEFUL and encouraging, not depressing
- Tone matches the EMOTIONAL CONTEXT of each moment
- All copy works for NON-NATIVE English speakers
- Confirmation dialogs name CONSEQUENCES clearly

Create the artifact as a comprehensive content design specification.
"""

ILLUSTRATION_SPECIALIST_TASK_PROMPT = """
# Illustration & Iconography Task

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Target Audience**: {target_audience}

## Brand Direction (from Brand Strategist)
Use the brand personality to determine illustration style parameters
(abstract vs representational, minimal vs detailed, geometric vs organic).

## Visual Direction (from Visual Designer)
Use ONLY colors from the approved brand color palette.
Match the visual language's corner radius and overall geometry.

## Your Mission
Create a custom illustration system and icon set that gives this product
its unique visual fingerprint. No generic libraries -- everything custom.

## Deliverables
1. Visual Language Definition:
   - Style parameters (realism, complexity, geometry, warmth, humor)
   - Construction rules (stroke weight, corner radius, perspective)
   - Color subset from brand palette
   - Texture/grain treatment
2. Icon System:
   - Grid & construction specs (24px canvas, keylines)
   - Complete icon inventory by category (nav, actions, objects, status, etc.)
   - Icon states (default, hover, active, disabled, selected)
   - Size variants (16, 20, 24, 32, 48px)
3. Spot Illustrations:
   - Empty state illustrations (for each product context)
   - Onboarding step illustrations
   - Error/success moment illustrations
   - Loading state illustrations
4. Hero Illustrations (if applicable):
   - Landing page graphics
   - Feature announcement visuals
5. Animation-Ready Notes:
   - Layer structure for animated illustrations
   - Motion-ready SVG specifications
6. Asset Delivery:
   - File format recommendations
   - Naming conventions
   - File organization structure
7. Accessibility:
   - Alt text for every meaningful illustration
   - Grayscale readability verification

## Critical Requirements
- NO Blush/unDraw/generic illustration library styles
- Consistent stroke weight across ALL icons
- Every icon uses the same grid and construction rules
- Style is DISTINCTIVE (couldn't belong to another product)
- Empty states are ENCOURAGING, not depressing
- All illustrations have ACCESSIBILITY text
- SVGs are OPTIMIZED and properly structured

Create the artifact as an illustration system specification.
"""

DESIGN_SYSTEMS_ARCHITECT_TASK_PROMPT = """
# Design Systems Architecture Task

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Tech Stack**: {tech_stack}
- **Dark Mode Required**: {require_dark_mode}

## Input Artifacts (from upstream agents)
You will receive specifications from:
- Visual Designer: Typography, color, spacing, shadows, layout
- Motion Designer: Duration, easing, stagger tokens
- Content Designer: Content patterns and constraints
- Illustration Specialist: Icon system, asset organization

## Your Mission
Synthesize all creative specifications into a scalable, maintainable design system
with three-tier tokens, component specifications, and theming architecture.

## Deliverables
1. Token Architecture:
   - Primitive Tokens (raw values: every color, size, space value)
   - Alias Tokens (semantic meaning: purpose-mapped to primitives)
   - Component Tokens (component-scoped: referencing aliases)
2. Theme Implementation:
   - Light mode complete token set
   - Dark mode complete token set (overrides only)
   - Theme contract (TypeScript interface)
   - Custom theme creation guide
3. Component Library Specifications:
   - Primitives: Box, Flex, Grid, Text, Icon, Spacer
   - Forms: Input, Select, Checkbox, Radio, Switch, Textarea, Slider
   - Actions: Button, IconButton, Link, DropdownMenu
   - Feedback: Toast, Alert, Badge, Progress, Skeleton
   - Layout: Container, Stack, Divider, Card, Panel
   - Navigation: Navbar, Sidebar, Tabs, Breadcrumb, Pagination
   - Overlay: Modal, Drawer, Popover, Tooltip, Command Palette
   - Data: Table, List, Avatar, Tag, Stat
4. For each component:
   - Purpose & anatomy (ASCII diagram)
   - Variants (size, color, emphasis)
   - All states (default, hover, active, focus, disabled, loading)
   - Tokens used (exact token names and values)
   - Accessibility (role, keyboard, focus management)
   - Content patterns (from Content Designer)
   - Motion specs (from Motion Designer)
5. Composition Patterns:
   - Common page layouts
   - Form patterns
   - Navigation patterns
   - Dashboard patterns

## Critical Requirements
- THREE-TIER token hierarchy maintained (no shortcuts)
- EVERY value references a token (no magic numbers)
- Dark mode is COMPLETE (no missing overrides)
- Motion tokens ALIGN with Motion Designer's spec
- Color tokens ALIGN with Visual Designer's palette
- Accessibility BUILT INTO every component
- Component specs include ALL states
- Documentation sufficient for a new engineer to implement without questions

Create the artifact as a comprehensive design system specification.
"""

CREATIVE_DIRECTOR_TASK_PROMPT = """
# Creative Director Review

## Product Context
- **Product**: {product_name}
- **Description**: {product_description}
- **Target Audience**: {target_audience}
- **Minimum Beauty Score**: {min_beauty_score}/10

## Your Mission
As the final creative authority, review ALL creative artifacts produced by the team:
1. Brand Strategy (from Brand Strategist)
2. Visual Design Spec (from Visual Designer)
3. Motion Design Spec (from Motion Designer)
4. Content Design Spec (from Content Designer)
5. Illustration System (from Illustration Specialist)
6. Design System Spec (from Design Systems Architect)

Assess whether the combined creative output achieves the beauty standard required
for this SaaS product to be distinctive, emotionally resonant, and craft-excellent.

## Assessment Criteria (Weighted)
| Dimension | Weight | Minimum Score |
|-----------|--------|--------------|
| Distinctiveness | 20% | 7/10 |
| Emotional Resonance | 20% | 7/10 |
| Visual Craft | 20% | 8/10 |
| Systemic Coherence | 15% | 7/10 |
| Motion & Life | 10% | 7/10 |
| Content & Voice | 10% | 7/10 |
| Innovation | 5% | 6/10 |

## Deliverables
1. Creative Vision Assessment (first impression, story the design tells)
2. Weighted Score with dimension breakdown
3. What's Working (preserve these)
4. What Needs Work (ranked by impact)
5. The Path to Beautiful (specific, actionable creative direction)
6. Creative References (2-3 products pointing the right direction)
7. Gate Decision: APPROVED or REJECTED

## Critical Requirements
- Be BRUTALLY HONEST about the first impression
- Score against REAL-WORLD premium SaaS (Stripe, Linear, Vercel), not against average
- If REJECTING, provide SPECIFIC actionable direction for improvement
- Ensure all creative work is COHERENT (brand → visual → motion → content → illustration)
- Verify the system could ONLY belong to this product (the logo test)
- Check for SaaS sameness traps (purple gradients, generic illustrations, safe choices)

## CRITICAL: Gate Decision
You MUST explicitly state:
- **APPROVED** - Creative work meets beauty standards and can proceed to implementation
- **REJECTED** - Creative work needs revision (always provide the Path to Beautiful)

Create the artifact as a creative director review with gate decision.
"""


# =============================================================================
# WORKFLOW DEFINITION
# =============================================================================

CREATIVITY_WORKFLOW_DEFINITION = {
    "name": "Design & Creativity Enhancement",
    "description": "Multi-agent creative team for improving SaaS product beauty",
    "phases": [
        {
            "phase": 1,
            "name": "Brand Foundation",
            "parallel": False,
            "agents": [
                {
                    "agent": "brand_strategist",
                    "prompt_template": BRAND_STRATEGIST_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem"],
                    "artifacts": ["brand_strategy"]
                }
            ]
        },
        {
            "phase": 2,
            "name": "Visual Language Development",
            "parallel": True,
            "agents": [
                {
                    "agent": "visual_designer",
                    "prompt_template": VISUAL_DESIGNER_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem", "playwright"],
                    "artifacts": ["visual_design_spec"]
                },
                {
                    "agent": "illustration_specialist",
                    "prompt_template": ILLUSTRATION_SPECIALIST_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem"],
                    "artifacts": ["illustration_system"]
                }
            ]
        },
        {
            "phase": 3,
            "name": "Kinetic & Verbal Layer",
            "parallel": True,
            "agents": [
                {
                    "agent": "motion_designer",
                    "prompt_template": MOTION_DESIGNER_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem"],
                    "artifacts": ["motion_design_spec"]
                },
                {
                    "agent": "content_designer",
                    "prompt_template": CONTENT_DESIGNER_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem"],
                    "artifacts": ["content_design_spec"]
                }
            ]
        },
        {
            "phase": 4,
            "name": "System Codification",
            "parallel": False,
            "agents": [
                {
                    "agent": "design_systems_architect",
                    "prompt_template": DESIGN_SYSTEMS_ARCHITECT_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem"],
                    "artifacts": ["design_system_spec"]
                }
            ]
        },
        {
            "phase": 5,
            "name": "Creative Review (Quality Gate)",
            "parallel": False,
            "agents": [
                {
                    "agent": "creative_director",
                    "prompt_template": CREATIVE_DIRECTOR_TASK_PROMPT,
                    "model": "sonnet",
                    "tools": ["filesystem", "playwright"],
                    "artifacts": ["creative_review"],
                    "is_gate": True,
                    "min_score": 7.5
                }
            ]
        }
    ]
}


def get_creativity_workflow_prompts(
    product_name: str,
    product_description: str,
    target_audience: str,
    existing_url: Optional[str] = None,
    codebase_path: Optional[str] = None,
    competitors: Optional[list[str]] = None,
    existing_guidelines: Optional[str] = None,
    tech_stack: Optional[str] = None,
    require_dark_mode: bool = True,
    min_beauty_score: float = 7.5,
) -> dict:
    """
    Generate all agent prompts for the design creativity workflow.

    Returns a dict with prompt for each agent, ready for Task tool execution.
    """
    existing_context = ""
    if existing_url:
        existing_context = f"""
## Existing Product
URL: {existing_url}
Codebase: {codebase_path or 'Not provided'}

USE PLAYWRIGHT to review the current state before designing improvements:
1. browser_navigate to {existing_url}
2. browser_take_screenshot for current state assessment
3. Identify what to KEEP vs what to REDESIGN
"""

    format_vars = {
        "product_name": product_name,
        "product_description": product_description,
        "target_audience": target_audience,
        "competitors": ", ".join(competitors) if competitors else "Not specified",
        "existing_guidelines": existing_guidelines or "None - creating from scratch",
        "tech_stack": tech_stack or "React/TypeScript (modern SaaS stack)",
        "require_dark_mode": str(require_dark_mode),
        "existing_context": existing_context,
        "min_beauty_score": str(min_beauty_score),
    }

    return {
        "brand_strategist": BRAND_STRATEGIST_TASK_PROMPT.format(**format_vars),
        "visual_designer": VISUAL_DESIGNER_TASK_PROMPT.format(**format_vars),
        "motion_designer": MOTION_DESIGNER_TASK_PROMPT.format(**format_vars),
        "content_designer": CONTENT_DESIGNER_TASK_PROMPT.format(**format_vars),
        "illustration_specialist": ILLUSTRATION_SPECIALIST_TASK_PROMPT.format(**format_vars),
        "design_systems_architect": DESIGN_SYSTEMS_ARCHITECT_TASK_PROMPT.format(**format_vars),
        "creative_director": CREATIVE_DIRECTOR_TASK_PROMPT.format(**format_vars),
    }


def get_execution_instructions() -> str:
    """Get instructions for executing this workflow in Claude Code."""
    return """
## Design & Creativity Workflow - Execution Instructions

### Quick Start
To run the full creative enhancement workflow:

### Phase 1: Brand Foundation (SEQUENTIAL - Must complete first)
```python
Task(
    subagent_type="general-purpose",
    description="Brand Strategy Foundation",
    prompt=brand_strategist_prompt,
    model="sonnet"
)
```

### Phase 2: Visual Language Development (PARALLEL)
Launch both simultaneously after Phase 1 completes:
```python
Task(
    subagent_type="general-purpose",
    description="Visual Design System",
    prompt=visual_designer_prompt + f"\\n\\n## Brand Strategy\\n{phase1_result}",
    model="sonnet"
)

Task(
    subagent_type="general-purpose",
    description="Illustration System",
    prompt=illustration_specialist_prompt + f"\\n\\n## Brand Strategy\\n{phase1_result}",
    model="sonnet"
)
```

### Phase 3: Kinetic & Verbal Layer (PARALLEL)
Launch both after Phase 2 completes:
```python
Task(
    subagent_type="general-purpose",
    description="Motion Design",
    prompt=motion_designer_prompt + f"\\n\\n## Context\\n{phase1_and_2_results}",
    model="sonnet"
)

Task(
    subagent_type="general-purpose",
    description="Content Design",
    prompt=content_designer_prompt + f"\\n\\n## Context\\n{phase1_result}",
    model="sonnet"
)
```

### Phase 4: System Codification (SEQUENTIAL)
After all creative specs are complete:
```python
Task(
    subagent_type="general-purpose",
    description="Design System Architecture",
    prompt=design_systems_architect_prompt + f"\\n\\n## All Specs\\n{all_creative_results}",
    model="sonnet"
)
```

### Phase 5: Creative Review - Quality Gate (SEQUENTIAL)
After design system is complete:
```python
Task(
    subagent_type="general-purpose",
    description="Creative Director Review",
    prompt=creative_director_prompt + f"\\n\\n## All Creative Work\\n{all_results}",
    model="sonnet"
)
```

### If REJECTED by Creative Director:
The rejection will include specific "Path to Beautiful" direction.
Re-run the relevant agents with the feedback incorporated, then re-submit
for Creative Director review.

### Using the Orchestrator MCP
```python
mcp__orchestrator__start_workflow(
    task="Design creativity enhancement for {product_name}",
    working_directory="{codebase_path}"
)

# Execute each phase...
mcp__orchestrator__execute_agent(
    agent="brand_strategist",
    task="Establish brand foundation for {product_name}",
    working_directory="{codebase_path}"
)
```
"""
