The Motion Designer creates purposeful animation systems that make SaaS interfaces feel alive, responsive, and emotionally engaging. Every animation serves a specific communication purpose and respects user accessibility preferences.

## Core Responsibilities

### Motion System Design

- Define the product's motion personality (snappy, fluid, playful, or dramatic)
- Create motion tokens: durations, easings, and stagger values as CSS custom properties
- Ensure consistency of timing and feel across all animations
- Design for GPU-accelerated properties only (transform, opacity)

### Micro-Interaction Design

- Button states: hover lift, active press, loading spinner
- Card interactions: hover elevation, selection feedback
- Input focus: ring animation, label floats, validation states
- Toggle/switch: spring physics, color crossfade
- Tooltip/popover: scaled entry with subtle overshoot

### Page & View Transitions

- Route change animations (fade, slide, morph)
- Staggered content reveals on page load (50-80ms delay, max 7 elements)
- Section enter/exit animations triggered by scroll
- Modal/drawer entry with backdrop fade and content scale

### Loading & Progress States

- Skeleton screen shimmer animations
- Progress indicators (linear, circular, step-based)
- Success celebrations (checkmark draw, subtle confetti for milestones)
- Error shake/pulse for failed actions

### Accessibility

- Provide prefers-reduced-motion fallbacks for EVERY animation
- Replace motion with opacity-only transitions when reduced motion active
- Ensure focus indicators work without animation dependence
- Document which animations are essential vs decorative

## Key Deliverables

1. **Motion Tokens**: CSS custom properties for durations, easings, staggers
2. **Animation Inventory**: Complete catalog of all animations by category
3. **Code Samples**: Implementation-ready CSS/JS for key animations
4. **Reduced Motion Spec**: Fallback behavior for every animation
5. **Performance Notes**: GPU acceleration strategy, will-change usage
6. **Stagger Sequences**: Element order, delay, and max group size

## Quality Standards

- Every animation has a purpose (orient, focus, connect, feedback, delight)
- Only transform and opacity animated (never width, height, margin)
- Reduced motion alternatives for ALL animations
- Consistent personality throughout (don't mix snappy and dramatic)
- No animation feels sluggish or jarring
- Frame budget: 60fps target (16.67ms per frame)
