# /pr - Create a Pull Request

Create a GitHub pull request for the current branch.

## Instructions

1. Check current state:
   ```bash
   git status
   git branch -vv
   git log origin/main..HEAD --oneline
   ```

2. Ensure branch is pushed:
   ```bash
   git push -u origin HEAD
   ```

3. Review all changes in the PR:
   ```bash
   git diff main...HEAD
   ```

4. Create PR with gh CLI:
   ```bash
   gh pr create --title "type: Short description" --body "$(cat <<'EOF'
   ## Summary
   - Brief description of changes
   - Why this change is needed

   ## Changes
   - List of specific changes made

   ## Test Plan
   - [ ] Tests pass (`pytest`)
   - [ ] Manual verification (`/verify`)
   - [ ] No regressions

   ---
   Generated with [Claude Code](https://claude.ai/code)
   EOF
   )"
   ```

## PR Title Format

Use same types as commits:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring
- `docs:` Documentation
- `test:` Test changes

## After Creating

1. Return the PR URL to the user
2. Suggest running `/verify` before merging
