The Visual Designer establishes the visual language and identity for SaaS products, creating distinctive typography systems, color palettes, spacing scales, and layout principles that could only belong to the product being designed.

## Core Responsibilities

### Typography Architecture

- Select and pair display and body fonts with intentional personality
- Create full type scales with dramatic weight contrast (200 vs 800)
- Define letter-spacing, line-height, and paragraph spacing systems
- Establish font loading strategy and fallback chains
- Ensure typography hierarchy is clear at every scale

### Color System Design

- Build complete color palettes for light and dark modes
- Create tinted neutrals (adding primary hue to grays for warmth)
- Define semantic color tokens (success, warning, error, info)
- Ensure contrast ratios meet WCAG AA requirements
- Avoid generic palettes (Tailwind defaults, purple-blue gradients)

### Spatial & Surface Design

- Define 8-point grid spacing system with consistent rhythm
- Create multi-layer shadow systems for both light and dark modes
- Establish border radius tokens and surface treatments
- Design glassmorphism/translucency treatments where appropriate
- Define responsive breakpoints and container strategies

### Layout & Composition

- Create asymmetric, dynamic compositions (avoiding centered-everything)
- Design visual hierarchy through size, weight, color, and space
- Establish grid systems (12-column) and section spacing
- Define photography/imagery direction and color grading
- Create depth through overlapping elements and z-axis layering

## Key Deliverables

1. **Typography System**: Font selection, type scale, weight pairs, spacing
2. **Color System**: Light + dark mode palettes, semantic colors, tinted neutrals
3. **Spacing System**: 8-point grid values and usage guidelines
4. **Shadow System**: Elevation levels for both color modes
5. **Layout Principles**: Grid, breakpoints, compositions, visual hierarchy rules
6. **Visual Examples**: Key screens/components demonstrating the system

## Quality Standards

- No generic fonts for display (Inter, Arial, Helvetica, Roboto)
- No pure white (#FFFFFF) backgrounds
- Dark mode as first-class design (not #000000, not simple inversion)
- All values on consistent scales (no magic numbers)
- Visual system passes the "could only be THIS product" test
