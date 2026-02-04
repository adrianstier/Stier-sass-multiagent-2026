#!/usr/bin/env python3
"""
Prompt Builder: Translates natural language requests into optimized orchestrator prompts.

Takes casual user descriptions of what they want to change and generates structured
prompts that leverage the multi-agent framework's interconnected agents effectively.

Usage:
    from orchestrator.prompt_builder import PromptBuilder, build_prompt

    # Quick usage
    prompt = build_prompt("make the buttons look better and more modern")

    # Full builder with customization
    builder = PromptBuilder()
    result = builder.analyze("fix the login page, it's ugly and slow")
    print(result.optimized_prompt)
    print(result.workflow_recommendation)
    print(result.agents_involved)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum


class ChangeCategory(str, Enum):
    """Categories of changes users might request."""
    VISUAL_DESIGN = "visual_design"          # Colors, fonts, spacing, aesthetics
    UX_INTERACTION = "ux_interaction"         # User flows, interactions, usability
    PERFORMANCE = "performance"               # Speed, loading, optimization
    FUNCTIONALITY = "functionality"           # Features, bugs, behavior
    ACCESSIBILITY = "accessibility"           # A11y, screen readers, keyboard nav
    RESPONSIVE = "responsive"                 # Mobile, tablet, different screens
    ANIMATION = "animation"                   # Motion, transitions, micro-interactions
    CONTENT = "content"                       # Text, copy, messaging
    ARCHITECTURE = "architecture"             # Code structure, refactoring
    SECURITY = "security"                     # Auth, vulnerabilities, data protection
    TESTING = "testing"                       # Tests, coverage, QA
    DEPLOYMENT = "deployment"                 # CI/CD, infrastructure


class WorkflowType(str, Enum):
    """Pre-defined workflow patterns."""
    FRONTEND_DESIGN = "frontend_design"       # Visual/UX focus
    FULL_STACK = "full_stack"                 # End-to-end feature
    DESIGN_ITERATION = "design_iteration"     # Quick design refinement
    DESIGN_CREATIVITY = "design_creativity"   # Full creative overhaul (7 agents)
    BUG_FIX = "bug_fix"                       # Targeted fix
    PERFORMANCE = "performance"                # Optimization focus
    SECURITY_REVIEW = "security_review"       # Security audit
    REFACTOR = "refactor"                     # Code quality improvement


@dataclass
class AgentCapability:
    """Describes what an agent can do."""
    name: str
    role_id: str
    strengths: List[str]
    categories: List[ChangeCategory]
    depends_on: List[str] = field(default_factory=list)  # Agents whose output helps this one
    enhances: List[str] = field(default_factory=list)    # Agents this one's output helps


# Agent interconnection map - who benefits from whose work
AGENT_GRAPH = {
    "business_analyst": AgentCapability(
        name="Business Analyst",
        role_id="business_analyst",
        strengths=["requirements", "user stories", "acceptance criteria", "stakeholder needs"],
        categories=[ChangeCategory.FUNCTIONALITY, ChangeCategory.UX_INTERACTION],
        depends_on=[],
        enhances=["architect", "ux_designer", "frontend", "backend"]
    ),
    "architect": AgentCapability(
        name="Tech Lead / Architect",
        role_id="architect",
        strengths=["system design", "architecture", "technical decisions", "patterns"],
        categories=[ChangeCategory.ARCHITECTURE, ChangeCategory.PERFORMANCE],
        depends_on=["business_analyst"],
        enhances=["backend", "frontend", "devops"]
    ),
    "ux_designer": AgentCapability(
        name="UX Designer",
        role_id="ux_designer",
        strengths=["wireframes", "user flows", "information architecture", "usability"],
        categories=[ChangeCategory.UX_INTERACTION, ChangeCategory.ACCESSIBILITY, ChangeCategory.RESPONSIVE],
        depends_on=["business_analyst"],
        enhances=["frontend", "graphic_designer"]
    ),
    "graphic_designer": AgentCapability(
        name="Graphic Designer",
        role_id="graphic_designer",
        strengths=["visual beauty", "color harmony", "typography", "emotional impact", "aesthetics"],
        categories=[ChangeCategory.VISUAL_DESIGN, ChangeCategory.ANIMATION],
        depends_on=["ux_designer", "frontend"],
        enhances=["frontend", "design_reviewer"]
    ),
    "design_reviewer": AgentCapability(
        name="Design Reviewer",
        role_id="design_reviewer",
        strengths=["design patterns", "consistency", "distinctiveness", "anti-patterns"],
        categories=[ChangeCategory.VISUAL_DESIGN, ChangeCategory.UX_INTERACTION],
        depends_on=["frontend", "graphic_designer"],
        enhances=["frontend"]
    ),
    "frontend": AgentCapability(
        name="Frontend Engineer",
        role_id="frontend",
        strengths=["React/Vue", "CSS/Tailwind", "components", "state management", "browser APIs"],
        categories=[ChangeCategory.VISUAL_DESIGN, ChangeCategory.UX_INTERACTION,
                   ChangeCategory.ANIMATION, ChangeCategory.RESPONSIVE, ChangeCategory.ACCESSIBILITY],
        depends_on=["ux_designer", "architect"],
        enhances=["graphic_designer", "design_reviewer", "reviewer"]
    ),
    "backend": AgentCapability(
        name="Backend Engineer",
        role_id="backend",
        strengths=["APIs", "databases", "auth", "server logic", "data models"],
        categories=[ChangeCategory.FUNCTIONALITY, ChangeCategory.PERFORMANCE, ChangeCategory.SECURITY],
        depends_on=["architect", "business_analyst"],
        enhances=["frontend", "security", "reviewer"]
    ),
    "reviewer": AgentCapability(
        name="Code Reviewer",
        role_id="reviewer",
        strengths=["code quality", "bugs", "best practices", "test coverage"],
        categories=[ChangeCategory.ARCHITECTURE, ChangeCategory.TESTING],
        depends_on=["frontend", "backend"],
        enhances=["frontend", "backend"]
    ),
    "security": AgentCapability(
        name="Security Reviewer",
        role_id="security",
        strengths=["vulnerabilities", "OWASP", "auth flaws", "injection", "secrets"],
        categories=[ChangeCategory.SECURITY],
        depends_on=["backend", "frontend"],
        enhances=["backend", "frontend"]
    ),
    "devops": AgentCapability(
        name="DevOps Engineer",
        role_id="devops",
        strengths=["Docker", "CI/CD", "deployment", "infrastructure", "monitoring"],
        categories=[ChangeCategory.DEPLOYMENT, ChangeCategory.PERFORMANCE],
        depends_on=["architect"],
        enhances=[]
    ),
    "qa": AgentCapability(
        name="QA Engineer",
        role_id="qa",
        strengths=["test plans", "E2E tests", "edge cases", "regression"],
        categories=[ChangeCategory.TESTING, ChangeCategory.FUNCTIONALITY],
        depends_on=["frontend", "backend"],
        enhances=["frontend", "backend"]
    ),
    "docs": AgentCapability(
        name="Documentation Writer",
        role_id="docs",
        strengths=["API docs", "user guides", "README", "code comments"],
        categories=[ChangeCategory.CONTENT],
        depends_on=["backend", "frontend"],
        enhances=[]
    ),
    # Design & Creativity Cluster
    "brand_strategist": AgentCapability(
        name="Brand Strategist",
        role_id="brand_strategist",
        strengths=["brand identity", "positioning", "personality", "voice framework", "experience principles"],
        categories=[ChangeCategory.VISUAL_DESIGN, ChangeCategory.CONTENT],
        depends_on=[],
        enhances=["visual_designer", "motion_designer", "content_designer", "illustration_specialist"]
    ),
    "visual_designer": AgentCapability(
        name="Visual Designer",
        role_id="visual_designer",
        strengths=["typography", "color systems", "spacing", "shadows", "visual hierarchy", "dark mode"],
        categories=[ChangeCategory.VISUAL_DESIGN],
        depends_on=["brand_strategist"],
        enhances=["motion_designer", "design_systems_architect", "frontend", "creative_director"]
    ),
    "motion_designer": AgentCapability(
        name="Motion Designer",
        role_id="motion_designer",
        strengths=["animations", "transitions", "micro-interactions", "motion tokens", "reduced-motion"],
        categories=[ChangeCategory.ANIMATION],
        depends_on=["visual_designer", "brand_strategist"],
        enhances=["design_systems_architect", "frontend", "creative_director"]
    ),
    "content_designer": AgentCapability(
        name="Content Designer",
        role_id="content_designer",
        strengths=["microcopy", "error messages", "empty states", "button labels", "UX writing", "tone"],
        categories=[ChangeCategory.CONTENT, ChangeCategory.UX_INTERACTION],
        depends_on=["brand_strategist"],
        enhances=["design_systems_architect", "frontend", "creative_director"]
    ),
    "illustration_specialist": AgentCapability(
        name="Illustration Specialist",
        role_id="illustration_specialist",
        strengths=["icons", "spot illustrations", "visual assets", "SVG", "icon grid", "custom graphics"],
        categories=[ChangeCategory.VISUAL_DESIGN],
        depends_on=["brand_strategist", "visual_designer"],
        enhances=["design_systems_architect", "frontend", "creative_director"]
    ),
    "design_systems_architect": AgentCapability(
        name="Design Systems Architect",
        role_id="design_systems_architect",
        strengths=["design tokens", "component specs", "theming", "dark mode", "token architecture"],
        categories=[ChangeCategory.VISUAL_DESIGN, ChangeCategory.ARCHITECTURE],
        depends_on=["visual_designer", "motion_designer", "content_designer", "illustration_specialist"],
        enhances=["frontend", "creative_director"]
    ),
    "creative_director": AgentCapability(
        name="Creative Director",
        role_id="creative_director",
        strengths=["beauty assessment", "distinctiveness", "brand coherence", "creative vision", "quality gate"],
        categories=[ChangeCategory.VISUAL_DESIGN, ChangeCategory.ANIMATION, ChangeCategory.CONTENT],
        depends_on=["visual_designer", "motion_designer", "content_designer", "illustration_specialist", "design_systems_architect"],
        enhances=["visual_designer", "motion_designer", "frontend"]
    ),
    # Data Science & R Cluster
    "tidyverse_r": AgentCapability(
        name="Tidyverse/R Expert",
        role_id="tidyverse_r",
        strengths=["R programming", "tidyverse", "dplyr", "ggplot2", "purrr", "tidymodels", "statistical computing", "data wrangling"],
        categories=[ChangeCategory.FUNCTIONALITY, ChangeCategory.ARCHITECTURE],
        depends_on=["data_scientist"],
        enhances=["nature_figures", "visualizer", "statistician"]
    ),
    "nature_figures": AgentCapability(
        name="Nature Figures Specialist",
        role_id="nature_figures",
        strengths=["publication figures", "scientific visualization", "Nature standards", "ggplot2", "colorblind palettes", "vector graphics"],
        categories=[ChangeCategory.VISUAL_DESIGN],
        depends_on=["tidyverse_r", "data_scientist", "statistician"],
        enhances=[]
    ),
    # Security & Authorization
    "authorization": AgentCapability(
        name="Authorization Expert",
        role_id="authorization",
        strengths=["RBAC", "ABAC", "ReBAC", "OAuth", "OIDC", "JWT", "RLS", "access control", "identity", "permissions"],
        categories=[ChangeCategory.SECURITY, ChangeCategory.ARCHITECTURE],
        depends_on=["architect", "backend"],
        enhances=["backend", "frontend", "security"]
    ),
}


# Keyword patterns for detecting intent
CATEGORY_PATTERNS = {
    ChangeCategory.VISUAL_DESIGN: [
        r"\b(ugly|beautiful|pretty|look|looks|color|colour|font|typography|spacing|padding|margin)\b",
        r"\b(style|styled|styling|design|designed|aesthetic|visual|appearance|theme|dark mode|light mode)\b",
        r"\b(icon|icons|image|images|logo|gradient|shadow|border|rounded|corners)\b",
    ],
    ChangeCategory.UX_INTERACTION: [
        r"\b(confusing|intuitive|flow|journey|experience|usability|user flow|navigation)\b",
        r"\b(click|tap|hover|interact|interaction|button|form|input|dropdown|modal)\b",
        r"\b(feedback|error message|loading|spinner|progress|toast|notification)\b",
    ],
    ChangeCategory.PERFORMANCE: [
        r"\b(slow|fast|speed|performance|loading|load time|optimize|lag|laggy)\b",
        r"\b(render|rendering|bundle|chunk|lazy|cache|cached|memory)\b",
    ],
    ChangeCategory.FUNCTIONALITY: [
        r"\b(broken|bug|fix|work|working|doesn't work|not working|feature|add|remove)\b",
        r"\b(implement|create|build|make it|should|needs to|supposed to)\b",
    ],
    ChangeCategory.ACCESSIBILITY: [
        r"\b(accessibility|a11y|screen reader|keyboard|focus|aria|wcag|contrast)\b",
        r"\b(accessible|blind|deaf|disability|impaired)\b",
    ],
    ChangeCategory.RESPONSIVE: [
        r"\b(mobile|tablet|desktop|responsive|breakpoint|screen size|viewport)\b",
        r"\b(phone|iphone|android|small screen|large screen)\b",
    ],
    ChangeCategory.ANIMATION: [
        r"\b(animation|animate|animated|transition|motion|move|moving|smooth)\b",
        r"\b(fade|slide|bounce|ease|timing|keyframe|micro-interaction)\b",
    ],
    ChangeCategory.CONTENT: [
        r"\b(text|copy|wording|message|label|title|heading|description)\b",
        r"\b(write|rewrite|change the text|update the copy)\b",
    ],
    ChangeCategory.ARCHITECTURE: [
        r"\b(refactor|restructure|reorganize|clean up|technical debt|architecture)\b",
        r"\b(component|module|service|pattern|abstract|decouple)\b",
    ],
    ChangeCategory.SECURITY: [
        r"\b(security|secure|auth|authentication|authorization|password|token)\b",
        r"\b(vulnerability|exploit|injection|xss|csrf|sql injection)\b",
    ],
    ChangeCategory.TESTING: [
        r"\b(test|tests|testing|coverage|unit test|e2e|integration test)\b",
        r"\b(qa|quality|regression|automated)\b",
    ],
    ChangeCategory.DEPLOYMENT: [
        r"\b(deploy|deployment|ci/cd|pipeline|docker|kubernetes|hosting)\b",
        r"\b(production|staging|environment|server)\b",
    ],
}


# Intensity/urgency patterns
INTENSITY_PATTERNS = {
    "critical": [r"\b(urgent|critical|asap|immediately|broken|down|outage)\b"],
    "high": [r"\b(important|priority|soon|needs|must|should)\b"],
    "medium": [r"\b(would be nice|could|might|consider|improve)\b"],
    "low": [r"\b(eventually|someday|minor|small|tweak)\b"],
}


# Scope patterns
SCOPE_PATTERNS = {
    "specific_component": [r"\b(the\s+\w+\s+(page|component|button|form|modal|header|footer|sidebar|nav))\b"],
    "specific_page": [r"\b(the\s+\w+\s+page|on\s+\w+\s+page|homepage|home page|landing page|login page|dashboard)\b"],
    "whole_site": [r"\b(entire|whole|all|everywhere|site-wide|global|throughout)\b"],
}


@dataclass
class AnalysisResult:
    """Result of analyzing a user's natural language request."""
    original_input: str
    categories: List[ChangeCategory]
    primary_category: ChangeCategory
    agents_involved: List[str]
    agent_sequence: List[List[str]]  # Parallelizable groups in order
    workflow_recommendation: WorkflowType
    intensity: str
    scope: str
    optimized_prompt: str
    orchestrator_command: str
    explanation: str


class PromptBuilder:
    """Analyzes natural language and builds optimized orchestrator prompts."""

    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = project_dir
        self.agents = AGENT_GRAPH

    def _detect_categories(self, text: str) -> List[Tuple[ChangeCategory, int]]:
        """Detect change categories from text with confidence scores."""
        text_lower = text.lower()
        scores = []

        for category, patterns in CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches)
            if score > 0:
                scores.append((category, score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])
        return scores

    def _detect_intensity(self, text: str) -> str:
        """Detect urgency/intensity level."""
        text_lower = text.lower()

        for intensity, patterns in INTENSITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intensity

        return "medium"  # Default

    def _detect_scope(self, text: str) -> str:
        """Detect scope of changes."""
        text_lower = text.lower()

        for scope, patterns in SCOPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return scope

        return "unspecified"

    def _select_agents(self, categories: List[ChangeCategory]) -> List[str]:
        """Select relevant agents based on detected categories."""
        selected = set()

        for category in categories:
            for agent_id, agent in self.agents.items():
                if category in agent.categories:
                    selected.add(agent_id)

        # Add supporting agents based on dependencies
        to_add = set()
        for agent_id in selected:
            agent = self.agents[agent_id]
            # Add agents that enhance selected ones
            for other_id, other in self.agents.items():
                if agent_id in other.enhances and other_id not in selected:
                    to_add.add(other_id)

        selected.update(to_add)
        return list(selected)

    def _order_agents(self, agent_ids: List[str]) -> List[List[str]]:
        """Order agents into parallelizable execution groups."""
        if not agent_ids:
            return []

        remaining = set(agent_ids)
        ordered = []
        completed = set()

        while remaining:
            # Find agents whose dependencies are all completed or not in our list
            ready = []
            for agent_id in remaining:
                agent = self.agents[agent_id]
                deps_met = all(
                    dep not in remaining or dep in completed
                    for dep in agent.depends_on
                )
                if deps_met:
                    ready.append(agent_id)

            if not ready:
                # Cycle or all remaining have unmet deps - just add them
                ready = list(remaining)

            ordered.append(sorted(ready))  # Sort for determinism
            completed.update(ready)
            remaining -= set(ready)

        return ordered

    def _select_workflow(self, categories: List[ChangeCategory], agents: List[str]) -> WorkflowType:
        """Select the best workflow type."""
        cat_set = set(categories)
        agent_set = set(agents)

        # Creativity cluster detection - if creative agents are selected
        creativity_agents = {"brand_strategist", "visual_designer", "motion_designer",
                           "content_designer", "illustration_specialist", "design_systems_architect",
                           "creative_director"}
        if agent_set & creativity_agents:
            if len(agent_set & creativity_agents) >= 3:
                return WorkflowType.DESIGN_CREATIVITY
            # Fewer creative agents = design iteration
            return WorkflowType.DESIGN_ITERATION

        # Security takes priority
        if ChangeCategory.SECURITY in cat_set:
            return WorkflowType.SECURITY_REVIEW

        # Performance focus
        if ChangeCategory.PERFORMANCE in cat_set and len(cat_set) <= 2:
            return WorkflowType.PERFORMANCE

        # Pure visual/design work
        if cat_set <= {ChangeCategory.VISUAL_DESIGN, ChangeCategory.ANIMATION, ChangeCategory.UX_INTERACTION}:
            if "graphic_designer" in agents or "design_reviewer" in agents:
                return WorkflowType.DESIGN_ITERATION if len(agents) <= 3 else WorkflowType.FRONTEND_DESIGN

        # Architecture/refactoring
        if ChangeCategory.ARCHITECTURE in cat_set:
            return WorkflowType.REFACTOR

        # Bug fixes
        if ChangeCategory.FUNCTIONALITY in cat_set and len(cat_set) == 1:
            return WorkflowType.BUG_FIX

        # Default to full stack for complex requests
        if "backend" in agents and "frontend" in agents:
            return WorkflowType.FULL_STACK

        return WorkflowType.FRONTEND_DESIGN

    def _build_optimized_prompt(
        self,
        original: str,
        categories: List[ChangeCategory],
        agents: List[str],
        agent_sequence: List[List[str]],
        workflow: WorkflowType,
        intensity: str,
        scope: str,
    ) -> str:
        """Build an optimized prompt for the orchestrator."""

        # Build the structured prompt
        lines = []

        # Header with workflow context
        if workflow == WorkflowType.FRONTEND_DESIGN:
            lines.append("Run the frontend workflow with design quality focus:")
        elif workflow == WorkflowType.DESIGN_ITERATION:
            lines.append("Run a quick design iteration workflow:")
        elif workflow == WorkflowType.FULL_STACK:
            lines.append("Run the standard development workflow:")
        elif workflow == WorkflowType.SECURITY_REVIEW:
            lines.append("Run a security-focused review workflow:")
        elif workflow == WorkflowType.PERFORMANCE:
            lines.append("Run a performance optimization workflow:")
        elif workflow == WorkflowType.BUG_FIX:
            lines.append("Run a targeted bug fix workflow:")
        else:
            lines.append("Run the following workflow:")

        lines.append("")

        # Main request
        lines.append(f"## Request")
        lines.append(original)
        lines.append("")

        # Agent sequence
        lines.append(f"## Agent Workflow")
        for i, group in enumerate(agent_sequence, 1):
            agent_names = [self.agents[a].name for a in group]
            if len(group) > 1:
                lines.append(f"{i}. **Parallel**: {', '.join(agent_names)}")
            else:
                lines.append(f"{i}. {agent_names[0]}")
        lines.append("")

        # Quality criteria based on categories
        lines.append("## Quality Criteria")
        if ChangeCategory.VISUAL_DESIGN in categories:
            lines.append("- Visual beauty score ≥ 7/10 (assessed by graphic_designer)")
            lines.append("- No generic fonts (Roboto, Poppins, etc.) - use distinctive typography")
            lines.append("- Consistent color palette with CSS variables")
        if ChangeCategory.ANIMATION in categories:
            lines.append("- Smooth, purposeful animations (no jarring transitions)")
            lines.append("- Appropriate easing curves")
        if ChangeCategory.UX_INTERACTION in categories:
            lines.append("- Clear user feedback for all interactions")
            lines.append("- Intuitive navigation and flow")
        if ChangeCategory.PERFORMANCE in categories:
            lines.append("- No performance regressions")
            lines.append("- Lazy loading where appropriate")
        if ChangeCategory.ACCESSIBILITY in categories:
            lines.append("- WCAG 2.1 AA compliance")
            lines.append("- Keyboard navigable")
            lines.append("- Screen reader friendly")
        if ChangeCategory.SECURITY in categories:
            lines.append("- No security vulnerabilities introduced")
            lines.append("- OWASP Top 10 addressed")
        lines.append("")

        # Specific focus areas
        if scope != "unspecified":
            lines.append(f"## Scope")
            lines.append(f"Focus on: {scope.replace('_', ' ')}")
            lines.append("")

        # Intensity note
        if intensity in ["critical", "high"]:
            lines.append(f"## Priority: {intensity.upper()}")
            lines.append("")

        return "\n".join(lines)

    def _build_orchestrator_command(self, prompt: str, workflow: WorkflowType) -> str:
        """Build the actual orchestrator MCP command."""
        # Escape for JSON
        escaped = prompt.replace('"', '\\"').replace('\n', '\\n')

        cmd = f'orchestrate_task: {prompt[:200]}...' if len(prompt) > 200 else f'orchestrate_task: {prompt}'

        return cmd

    def analyze(self, user_input: str) -> AnalysisResult:
        """Analyze user input and produce optimized orchestrator prompt."""

        # Detect categories
        category_scores = self._detect_categories(user_input)
        categories = [cat for cat, _ in category_scores]

        if not categories:
            # Default to general functionality
            categories = [ChangeCategory.FUNCTIONALITY]

        primary_category = categories[0]

        # Select and order agents
        agents = self._select_agents(categories)
        agent_sequence = self._order_agents(agents)

        # Determine workflow
        workflow = self._select_workflow(categories, agents)

        # Detect intensity and scope
        intensity = self._detect_intensity(user_input)
        scope = self._detect_scope(user_input)

        # Build optimized prompt
        optimized = self._build_optimized_prompt(
            user_input, categories, agents, agent_sequence, workflow, intensity, scope
        )

        # Build orchestrator command
        command = self._build_orchestrator_command(optimized, workflow)

        # Build explanation
        agent_names = [self.agents[a].name for a in agents]
        explanation = (
            f"Detected {len(categories)} change categories: {', '.join(c.value for c in categories[:3])}. "
            f"Recommending {workflow.value} workflow with {len(agents)} agents: {', '.join(agent_names[:4])}..."
        )

        return AnalysisResult(
            original_input=user_input,
            categories=categories,
            primary_category=primary_category,
            agents_involved=agents,
            agent_sequence=agent_sequence,
            workflow_recommendation=workflow,
            intensity=intensity,
            scope=scope,
            optimized_prompt=optimized,
            orchestrator_command=command,
            explanation=explanation,
        )


def build_prompt(user_input: str, project_dir: Optional[str] = None) -> str:
    """Quick helper to build an optimized prompt from natural language."""
    builder = PromptBuilder(project_dir)
    result = builder.analyze(user_input)
    return result.optimized_prompt


def analyze_request(user_input: str, project_dir: Optional[str] = None) -> AnalysisResult:
    """Analyze a request and return full analysis result."""
    builder = PromptBuilder(project_dir)
    return builder.analyze(user_input)


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prompt_builder.py 'your natural language request'")
        print()
        print("Examples:")
        print("  python prompt_builder.py 'make the buttons look more modern and add hover effects'")
        print("  python prompt_builder.py 'the login page is ugly and slow, fix it'")
        print("  python prompt_builder.py 'add dark mode to the whole site'")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    result = analyze_request(user_input)

    print("=" * 60)
    print("PROMPT BUILDER ANALYSIS")
    print("=" * 60)
    print()
    print(f"Original: {result.original_input}")
    print()
    print(f"Categories: {', '.join(c.value for c in result.categories)}")
    print(f"Workflow: {result.workflow_recommendation.value}")
    print(f"Intensity: {result.intensity}")
    print(f"Scope: {result.scope}")
    print()
    print(f"Agents ({len(result.agents_involved)}):")
    for i, group in enumerate(result.agent_sequence, 1):
        print(f"  Phase {i}: {', '.join(group)}")
    print()
    print("-" * 60)
    print("OPTIMIZED PROMPT:")
    print("-" * 60)
    print(result.optimized_prompt)
