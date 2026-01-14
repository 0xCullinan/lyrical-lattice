# /simplify - Simplify Recent Code Changes

Review and simplify code that was just written.

## Instructions

1. Identify files changed in the current session
2. For each file, look for opportunities to:
   - Remove unnecessary abstractions
   - Eliminate dead code
   - Combine similar functions
   - Simplify complex conditionals
   - Remove over-engineering

## Simplification Checklist

### Remove Over-Engineering
- [ ] No helper functions used only once
- [ ] No premature abstractions
- [ ] No unnecessary configuration
- [ ] No feature flags for hypothetical futures

### Reduce Complexity
- [ ] Functions do one thing
- [ ] No deeply nested conditionals (max 3 levels)
- [ ] No god objects/classes
- [ ] Clear variable names (no abbreviations)

### Clean Up
- [ ] No commented-out code
- [ ] No TODO comments for things we're not doing
- [ ] No unused imports
- [ ] No debug print statements

## Philosophy

> "Three similar lines of code is better than a premature abstraction."

> "The right amount of complexity is the minimum needed for the current task."

## After Simplifying

Run `/test` to ensure nothing broke.
