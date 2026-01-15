# /commit - Create a Git Commit

Create a well-formatted git commit for the current changes.

## Instructions

1. Check current state:
   ```bash
   git status
   git diff --stat
   ```

2. Review changes in detail:
   ```bash
   git diff
   ```

3. Check recent commit style:
   ```bash
   git log -5 --oneline
   ```

4. Stage relevant files:
   ```bash
   git add <files>
   ```
   Or stage all: `git add -A`

5. Create commit with descriptive message:
   ```bash
   git commit -m "$(cat <<'EOF'
   type: Short description (50 chars max)

   Longer explanation if needed. Wrap at 72 chars.
   - Bullet points for multiple changes
   - Focus on WHY not WHAT

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
   EOF
   )"
   ```

## Commit Types

- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring
- `docs:` Documentation only
- `test:` Adding tests
- `chore:` Maintenance tasks

## Rules

- DO NOT commit secrets, .env files, or credentials
- DO NOT use `git commit --amend` unless explicitly asked
- DO NOT force push to main
- ALWAYS include the Co-Authored-By trailer
