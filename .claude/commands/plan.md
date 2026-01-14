# /plan - Plan a New Feature

Enter plan mode to design implementation before coding.

## Instructions

1. Press `Shift + Tab` twice to enter Plan mode
2. Explore the codebase to understand:
   - Where the feature fits in the architecture
   - What existing code can be reused
   - What new files/changes are needed
3. Write a clear plan with:
   - Files to create/modify
   - Key implementation steps
   - Testing approach
4. Get approval before implementing

## Plan Template

```markdown
## Feature: [Name]

### Goal
[What this feature does]

### Files to Modify
- `src/...` - [what changes]
- `tests/...` - [what tests]

### Implementation Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Testing
- [ ] Unit tests for [component]
- [ ] Integration test for [flow]
- [ ] Manual verification with /verify

### Risks
- [Any potential issues]
```

## Tips

- A good plan is everything - invest time here
- Check CLAUDE.md for conventions
- Look at similar existing code first
- Consider edge cases upfront
