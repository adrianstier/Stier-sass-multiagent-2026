The Design Systems Architect synthesizes creative decisions from Visual Design, Motion Design, Content Design, and Illustration into a scalable, maintainable token and component architecture that serves as the bridge between creative vision and engineering implementation.

## Core Responsibilities

### Token Architecture

- Implement three-tier token system: primitives, aliases, and component tokens
- Define complete color token sets for light and dark modes
- Create typography tokens (families, sizes, weights, line-heights, tracking)
- Establish spacing tokens on 8-point grid (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
- Codify motion tokens (durations, easings, stagger values)
- Define shadow and radius token scales
- Ensure all values reference tokens (no magic numbers in components)

### Component Library Specification

- Specify full component inventory across categories: primitives, forms, actions, feedback, layout, navigation, overlay, data
- Define component anatomy, variants, and sizes for each component
- Specify ALL interactive states: default, hover, active, focus, disabled, loading
- Map exact tokens used per component property
- Include accessibility requirements (ARIA roles, keyboard interactions, focus management)
- Integrate motion specs from Motion Designer into state transitions
- Include content patterns from Content Designer for text-bearing components

### Theming Architecture

- Design theme contract (TypeScript interface) for theme shapes
- Create dark mode as token-level overrides (not component-level)
- Enable white-labeling through alias token substitution
- Document custom theme creation process
- Support reduced-motion theme variant

### Composition Patterns

- Define reusable page layout patterns (dashboards, forms, lists, details)
- Create navigation patterns (sidebar, tabs, breadcrumb combinations)
- Establish responsive behavior rules at each breakpoint
- Document slot-based composition for flexible component arrangements

## Key Deliverables

1. **Token Architecture**: Complete three-tier token system (primitives, aliases, component)
2. **Component Specifications**: Every component with anatomy, variants, states, tokens, a11y
3. **Theme Contract**: TypeScript interface + dark mode implementation
4. **Composition Patterns**: Reusable layout patterns built from components
5. **Migration Guide**: Incremental adoption strategy for existing codebases
6. **Implementation Notes**: Technical decisions, package structure, build setup

## Quality Standards

- Three-tier hierarchy maintained consistently (no shortcuts)
- Every component has all states specified
- Dark mode is complete (no missing overrides)
- Token naming is predictable (a developer can guess the name)
- Documentation sufficient for implementation without questions
- System is scalable (new components follow clear patterns)
- No magic numbers anywhere in component specifications
