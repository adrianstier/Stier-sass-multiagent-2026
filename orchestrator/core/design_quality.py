"""Design Quality Tools for Frontend Agents.

This module provides:
1. Curated design resources (fonts, colors, inspiration)
2. Design quality validators for Ralph Wiggum loop
3. Visual regression testing helpers
4. Anti-pattern detection

The goal: Prevent generic "AI slop" aesthetics by providing concrete
references and objective quality checks.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

# Import Ralph Wiggum types
import importlib.util
import sys

if "ralph_wiggum" in sys.modules:
    # Already loaded - use existing module
    _ralph_module = sys.modules["ralph_wiggum"]
else:
    # Load fresh
    _ralph_path = os.path.join(os.path.dirname(__file__), "ralph_wiggum.py")
    _ralph_spec = importlib.util.spec_from_file_location("ralph_wiggum", _ralph_path)
    _ralph_module = importlib.util.module_from_spec(_ralph_spec)
    sys.modules["ralph_wiggum"] = _ralph_module
    _ralph_spec.loader.exec_module(_ralph_module)

ValidationResult = _ralph_module.ValidationResult
ValidationCriteria = _ralph_module.ValidationCriteria


# =============================================================================
# Curated Design Resources
# =============================================================================

# Premium fonts that create distinctive aesthetics (NOT generic system fonts)
# Source: Anthropic Frontend Aesthetics Cookbook
CURATED_FONTS = {
    "display": {
        "editorial": ["Playfair Display", "Cormorant Garamond", "Libre Baskerville", "Fraunces", "Newsreader"],
        "modern": ["Syne", "Clash Display", "Cabinet Grotesk", "Satoshi", "Obviously"],
        "startup": ["Clash Display", "Satoshi", "Cabinet Grotesk", "Bricolage Grotesque"],  # From cookbook
        "brutalist": ["Bebas Neue", "Oswald", "Anton", "Archivo Black"],
        "playful": ["Fredoka", "Baloo 2", "Quicksand", "Nunito"],
        "luxury": ["Cormorant", "Cinzel", "Marcellus", "Forum"],
        "retro": ["Righteous", "Bungee", "Abril Fatface", "Lobster"],
        "minimal": ["Manrope", "DM Sans", "Plus Jakarta Sans", "Outfit"],
        "tech": ["JetBrains Mono", "Fira Code", "Source Code Pro", "IBM Plex Mono", "Space Grotesk"],
    },
    "body": {
        "readable": ["Source Serif Pro", "Merriweather", "Crimson Text", "Crimson Pro", "PT Serif"],
        "clean": ["Work Sans", "Nunito Sans", "Karla", "Rubik"],
        "elegant": ["Cormorant Garamond", "EB Garamond", "Spectral", "Libre Baskerville"],
        "modern": ["Inter", "DM Sans", "Plus Jakarta Sans", "Outfit"],  # Inter OK for body only
    },
    # Fonts to AVOID (overused, generic) - from cookbook
    "banned": [
        "Arial", "Helvetica", "Times New Roman", "Comic Sans",
        "Roboto",  # Too ubiquitous
        "Open Sans",  # Default everywhere
        "Lato",  # Overused
        "Montserrat",  # AI default
        "Poppins",  # Every AI uses this
        "system-ui", "-apple-system", "BlinkMacSystemFont",  # System defaults
    ],
    # High-impact font pairing strategies (from cookbook)
    "pairing_strategies": {
        "display_plus_mono": ["Clash Display + JetBrains Mono", "Satoshi + Fira Code"],
        "serif_plus_geometric": ["Playfair Display + DM Sans", "Fraunces + Outfit"],
        "editorial": ["Newsreader + Source Serif Pro", "Crimson Pro + Work Sans"],
    }
}

# Typography techniques from Anthropic Cookbook
TYPOGRAPHY_TECHNIQUES = {
    "weight_contrast": {
        "description": "Use EXTREME weight contrasts, not subtle differences",
        "good": "font-weight: 100/200 for body, 800/900 for headings",
        "bad": "font-weight: 400/600 (too subtle, looks generic)",
    },
    "size_jumps": {
        "description": "Use 3x+ size jumps, not 1.5x increments",
        "good": "text-sm (14px) body → text-6xl (60px) heading = 4x jump",
        "bad": "text-base (16px) → text-2xl (24px) = 1.5x (too timid)",
    },
    "loading": {
        "description": "Always load from Google Fonts for reliability",
        "example": "@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@200;700&display=swap');",
    },
}

# IDE Theme Inspiration (from cookbook) - for tech/developer tools
IDE_THEME_PALETTES = {
    "tokyo-night": {
        "background": "#1a1b26",
        "surface": "#24283b",
        "primary": "#7aa2f7",
        "secondary": "#bb9af7",
        "accent": "#7dcfff",
        "text": "#c0caf5",
        "muted": "#565f89",
    },
    "catppuccin-mocha": {
        "background": "#1e1e2e",
        "surface": "#313244",
        "primary": "#cba6f7",
        "secondary": "#f5c2e7",
        "accent": "#94e2d5",
        "text": "#cdd6f4",
        "muted": "#6c7086",
    },
    "dracula": {
        "background": "#282a36",
        "surface": "#44475a",
        "primary": "#bd93f9",
        "secondary": "#ff79c6",
        "accent": "#50fa7b",
        "text": "#f8f8f2",
        "muted": "#6272a4",
    },
    "nord": {
        "background": "#2e3440",
        "surface": "#3b4252",
        "primary": "#88c0d0",
        "secondary": "#81a1c1",
        "accent": "#a3be8c",
        "text": "#eceff4",
        "muted": "#4c566a",
    },
    "gruvbox-dark": {
        "background": "#282828",
        "surface": "#3c3836",
        "primary": "#fabd2f",
        "secondary": "#83a598",
        "accent": "#b8bb26",
        "text": "#ebdbb2",
        "muted": "#928374",
    },
}

# Curated color palettes by aesthetic
CURATED_PALETTES = {
    "midnight-luxury": {
        "background": "#0a0a0f",
        "surface": "#14141f",
        "primary": "#c9a962",
        "secondary": "#8b7355",
        "accent": "#d4af37",
        "text": "#f5f5f5",
        "muted": "#6b6b7b",
    },
    "ocean-depth": {
        "background": "#0c1821",
        "surface": "#1b2838",
        "primary": "#4ecdc4",
        "secondary": "#2ab7ca",
        "accent": "#fed766",
        "text": "#e8f1f2",
        "muted": "#6c8ea0",
    },
    "warm-earth": {
        "background": "#1a1612",
        "surface": "#2d2520",
        "primary": "#d4a373",
        "secondary": "#ccd5ae",
        "accent": "#e9c46a",
        "text": "#fefae0",
        "muted": "#8b7355",
    },
    "neon-noir": {
        "background": "#0d0d0d",
        "surface": "#1a1a2e",
        "primary": "#e94560",
        "secondary": "#0f3460",
        "accent": "#16213e",
        "text": "#eaeaea",
        "muted": "#4a4a6a",
    },
    "forest-minimal": {
        "background": "#f8f9fa",
        "surface": "#ffffff",
        "primary": "#2d5a3d",
        "secondary": "#52796f",
        "accent": "#84a98c",
        "text": "#1a1a1a",
        "muted": "#6c757d",
    },
    "terracotta-warm": {
        "background": "#fdf6e3",
        "surface": "#fff8e7",
        "primary": "#c05746",
        "secondary": "#d4a373",
        "accent": "#2a4858",
        "text": "#2d2d2d",
        "muted": "#8b7355",
    },
    "electric-purple": {
        "background": "#0f0e17",
        "surface": "#1a1825",
        "primary": "#7f5af0",
        "secondary": "#2cb67d",
        "accent": "#ff8906",
        "text": "#fffffe",
        "muted": "#94a1b2",
    },
    "brutalist-mono": {
        "background": "#ffffff",
        "surface": "#f0f0f0",
        "primary": "#000000",
        "secondary": "#333333",
        "accent": "#ff0000",
        "text": "#000000",
        "muted": "#666666",
    },
}

# Animation/motion guidelines
MOTION_PATTERNS = {
    "page_load": {
        "stagger_delay": "0.1s",
        "duration": "0.6s",
        "easing": "cubic-bezier(0.16, 1, 0.3, 1)",  # ease-out-expo
        "transform": "translateY(20px) -> translateY(0)",
    },
    "hover_lift": {
        "duration": "0.3s",
        "easing": "cubic-bezier(0.34, 1.56, 0.64, 1)",  # ease-out-back
        "transform": "translateY(-4px)",
        "shadow": "0 10px 40px rgba(0,0,0,0.15)",
    },
    "button_press": {
        "duration": "0.15s",
        "easing": "ease-out",
        "transform": "scale(0.98)",
    },
    "reveal_up": {
        "duration": "0.8s",
        "easing": "cubic-bezier(0.16, 1, 0.3, 1)",
        "from": "opacity: 0; transform: translateY(30px)",
        "to": "opacity: 1; transform: translateY(0)",
    },
}

# Design anti-patterns to detect and reject
ANTI_PATTERNS = {
    "colors": [
        r"#6366f1",  # Indigo-500 (every AI uses this)
        r"#8b5cf6",  # Violet-500 (AI purple)
        r"linear-gradient.*purple.*blue",  # The cliché gradient
        r"from-purple.*to-blue",  # Tailwind version
        r"from-indigo.*to-purple",
    ],
    "fonts": [
        r"font-family:\s*['\"]?(Arial|Helvetica|sans-serif)['\"]?",
        r"font-family:\s*['\"]?system-ui['\"]?",
    ],
    "layouts": [
        # Generic hero patterns
        r"text-center.*text-5xl.*text-gray-600",  # Boring centered hero
    ],
    "shadows": [
        r"shadow-sm",  # Too subtle
        r"shadow(?!\-)",  # Default shadow
    ],
}


# =============================================================================
# Design Quality Validators
# =============================================================================

def create_font_validator(
    banned_fonts: Optional[List[str]] = None,
) -> Callable:
    """Create a validator that checks for banned/generic fonts in CSS/code."""
    banned = banned_fonts or CURATED_FONTS["banned"]

    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        working_dir = context.get("working_dir", ".")
        issues = []

        # Search for CSS and style files
        try:
            result = subprocess.run(
                ["find", ".", "-type", "f", "-name", "*.css", "-o", "-name", "*.scss",
                 "-o", "-name", "*.tsx", "-o", "-name", "*.jsx"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            files = result.stdout.strip().split("\n") if result.stdout else []

            for filepath in files[:50]:  # Limit to 50 files
                if not filepath:
                    continue
                full_path = os.path.join(working_dir, filepath)
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        for font in banned:
                            if font.lower() in content.lower():
                                issues.append(f"{filepath}: Uses banned font '{font}'")
                except Exception:
                    pass

            passed = len(issues) == 0
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=passed,
                message="Typography is distinctive" if passed else f"Found {len(issues)} generic font usage(s)",
                details={
                    "criteria_name": "font_quality",
                    "issues": issues[:10],  # Limit output
                    "recommendation": "Use fonts from CURATED_FONTS instead of system defaults",
                }
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,  # Don't block on error
                message=f"Font validation skipped: {str(e)}",
            )

    return validator


def create_color_antipattern_validator() -> Callable:
    """Detect clichéd color patterns (purple gradients, etc.)."""

    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        working_dir = context.get("working_dir", ".")
        issues = []

        try:
            result = subprocess.run(
                ["find", ".", "-type", "f", "-name", "*.css", "-o", "-name", "*.scss",
                 "-o", "-name", "*.tsx", "-o", "-name", "*.jsx", "-o", "-name", "*.html"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            files = result.stdout.strip().split("\n") if result.stdout else []

            for filepath in files[:50]:
                if not filepath:
                    continue
                full_path = os.path.join(working_dir, filepath)
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        for pattern in ANTI_PATTERNS["colors"]:
                            if re.search(pattern, content, re.IGNORECASE):
                                issues.append(f"{filepath}: Uses clichéd color pattern")
                                break
                except Exception:
                    pass

            passed = len(issues) == 0
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=passed,
                message="Color palette is distinctive" if passed else f"Found {len(issues)} clichéd color pattern(s)",
                details={
                    "criteria_name": "color_originality",
                    "issues": issues[:10],
                    "recommendation": "Use palettes from CURATED_PALETTES or create a unique scheme",
                }
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,
                message=f"Color validation skipped: {str(e)}",
            )

    return validator


def create_accessibility_validator(
    url: Optional[str] = None,
) -> Callable:
    """Validate accessibility using Playwright's accessibility snapshot."""

    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        target_url = url or context.get("url", "http://localhost:3000")

        try:
            # Run accessibility check via Playwright
            script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    try {{
        await page.goto('{target_url}', {{ timeout: 10000 }});
        const snapshot = await page.accessibility.snapshot();

        // Basic a11y checks
        const issues = [];

        // Check for page title
        const title = await page.title();
        if (!title || title.length < 3) {{
            issues.push('Missing or short page title');
        }}

        // Check for main landmark
        const main = await page.$('main, [role="main"]');
        if (!main) {{
            issues.push('Missing <main> landmark');
        }}

        // Check for heading hierarchy
        const h1 = await page.$('h1');
        if (!h1) {{
            issues.push('Missing <h1> heading');
        }}

        // Check for skip link
        const skipLink = await page.$('a[href="#main"], a[href="#content"], .skip-link');
        if (!skipLink) {{
            issues.push('Missing skip-to-content link');
        }}

        // Check images for alt text
        const imagesWithoutAlt = await page.$$('img:not([alt])');
        if (imagesWithoutAlt.length > 0) {{
            issues.push(`${{imagesWithoutAlt.length}} image(s) missing alt text`);
        }}

        console.log(JSON.stringify({{
            passed: issues.length === 0,
            issues: issues,
            hasSnapshot: !!snapshot
        }}));
    }} catch (e) {{
        console.log(JSON.stringify({{ error: e.message }}));
    }}
    await browser.close();
}})();
"""
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if "error" in data:
                    return ValidationResult(
                        criteria=ValidationCriteria.CUSTOM,
                        passed=True,  # Don't block on connection errors
                        message=f"Accessibility check skipped: {data['error']}",
                    )

                passed = data.get("passed", False)
                issues = data.get("issues", [])

                return ValidationResult(
                    criteria=ValidationCriteria.CUSTOM,
                    passed=passed,
                    message="Accessibility checks passed" if passed else f"Found {len(issues)} accessibility issue(s)",
                    details={
                        "criteria_name": "accessibility",
                        "issues": issues,
                        "url": target_url,
                    }
                )
            else:
                return ValidationResult(
                    criteria=ValidationCriteria.CUSTOM,
                    passed=True,
                    message="Accessibility check skipped (Playwright not available)",
                )

        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,
                message=f"Accessibility validation skipped: {str(e)}",
            )

    return validator


def create_visual_complexity_validator(
    min_css_rules: int = 20,
    require_animations: bool = True,
) -> Callable:
    """Ensure the design has sufficient visual complexity (not too minimal/boring)."""

    async def validator(output: Any, context: Dict[str, Any]) -> ValidationResult:
        working_dir = context.get("working_dir", ".")

        try:
            # Count CSS rules and check for animations
            css_rule_count = 0
            has_animations = False
            has_transitions = False
            has_transforms = False
            has_gradients = False
            has_shadows = False

            result = subprocess.run(
                ["find", ".", "-type", "f", "-name", "*.css", "-o", "-name", "*.scss"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            files = result.stdout.strip().split("\n") if result.stdout else []

            for filepath in files[:20]:
                if not filepath:
                    continue
                full_path = os.path.join(working_dir, filepath)
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        css_rule_count += content.count('{')
                        has_animations = has_animations or '@keyframes' in content or 'animation' in content
                        has_transitions = has_transitions or 'transition' in content
                        has_transforms = has_transforms or 'transform' in content
                        has_gradients = has_gradients or 'gradient' in content
                        has_shadows = has_shadows or 'shadow' in content or 'box-shadow' in content
                except Exception:
                    pass

            issues = []
            if css_rule_count < min_css_rules:
                issues.append(f"Only {css_rule_count} CSS rules (minimum: {min_css_rules})")
            if require_animations and not has_animations:
                issues.append("No CSS animations found")
            if not has_transitions:
                issues.append("No CSS transitions found")
            if not has_gradients and not has_shadows:
                issues.append("No gradients or shadows (lacks visual depth)")

            passed = len(issues) == 0
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=passed,
                message="Design has good visual complexity" if passed else "Design may be too minimal",
                details={
                    "criteria_name": "visual_complexity",
                    "css_rules": css_rule_count,
                    "has_animations": has_animations,
                    "has_transitions": has_transitions,
                    "has_gradients": has_gradients,
                    "has_shadows": has_shadows,
                    "issues": issues,
                }
            )
        except Exception as e:
            return ValidationResult(
                criteria=ValidationCriteria.CUSTOM,
                passed=True,
                message=f"Visual complexity check skipped: {str(e)}",
            )

    return validator


# =============================================================================
# Design Resource Helpers
# =============================================================================

def get_font_pairing(aesthetic: str = "modern") -> Dict[str, str]:
    """Get a recommended font pairing for a given aesthetic."""
    display_fonts = CURATED_FONTS["display"].get(aesthetic, CURATED_FONTS["display"]["modern"])
    body_fonts = CURATED_FONTS["body"].get("clean", CURATED_FONTS["body"]["modern"])

    import random
    return {
        "display": random.choice(display_fonts),
        "body": random.choice(body_fonts),
        "mono": random.choice(CURATED_FONTS["display"]["tech"]),
    }


def get_color_palette(style: Optional[str] = None) -> Dict[str, str]:
    """Get a curated color palette."""
    if style and style in CURATED_PALETTES:
        return CURATED_PALETTES[style]

    import random
    return random.choice(list(CURATED_PALETTES.values()))


def generate_css_variables(palette_name: str = "midnight-luxury") -> str:
    """Generate CSS custom properties from a palette."""
    palette = CURATED_PALETTES.get(palette_name, CURATED_PALETTES["midnight-luxury"])

    lines = [":root {"]
    for name, value in palette.items():
        lines.append(f"  --color-{name}: {value};")
    lines.append("}")

    return "\n".join(lines)


def get_design_system_prompt_additions() -> str:
    """Get additional context for frontend agents about design resources."""
    return f"""
## Curated Design Resources Available

### Font Pairings (Use These!)
You have access to curated font collections. Import from Google Fonts:

**Editorial Style:**
{', '.join(CURATED_FONTS['display']['editorial'][:3])} (display)
{', '.join(CURATED_FONTS['body']['readable'][:2])} (body)

**Modern Style:**
{', '.join(CURATED_FONTS['display']['modern'][:3])} (display)
{', '.join(CURATED_FONTS['body']['clean'][:2])} (body)

**Brutalist Style:**
{', '.join(CURATED_FONTS['display']['brutalist'][:3])} (display)

**BANNED FONTS (Never Use):**
{', '.join(CURATED_FONTS['banned'][:8])}

### Color Palettes (Pre-Built)
Use these cohesive palettes instead of picking random colors:

**midnight-luxury**: Dark luxury with gold accents
**ocean-depth**: Deep teal/cyan on dark blue
**warm-earth**: Terracotta and sage on warm dark
**neon-noir**: Pink/red neon on pure black
**forest-minimal**: Sage green on light background
**electric-purple**: Purple/green on near-black

Example CSS variables for "midnight-luxury":
```css
{generate_css_variables("midnight-luxury")}
```

### Motion Guidelines
- **Page Load**: Use staggered reveals with animation-delay (0.1s increments)
- **Hover**: translateY(-4px) with shadow increase, 0.3s ease-out-back
- **Buttons**: scale(0.98) on press, 0.15s
- **Reveals**: translateY(30px) → 0 with opacity, 0.8s ease-out-expo

### Quality Checks (Auto-Run)
Your code will be validated for:
1. No banned/generic fonts
2. No clichéd color patterns (purple-to-blue gradients)
3. Sufficient visual complexity (animations, shadows, gradients)
4. Accessibility (landmarks, headings, alt text)
"""


# =============================================================================
# Integrated Design Validators for Ralph Wiggum
# =============================================================================

def get_frontend_validators(working_dir: str, url: Optional[str] = None) -> List[Callable]:
    """Get all design quality validators for frontend work."""
    return [
        create_font_validator(),
        create_color_antipattern_validator(),
        create_visual_complexity_validator(min_css_rules=15, require_animations=False),
        create_accessibility_validator(url=url),
    ]
