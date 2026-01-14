# /lint - Format and Lint Code

Format Python code with black and isort, then run type checks.

## Instructions

1. Format with black:
   ```bash
   cd /Users/macbook/.gemini/antigravity/scratch/oronym-assistant && black src/ tests/ --line-length 100
   ```

2. Sort imports with isort:
   ```bash
   isort src/ tests/ --profile black
   ```

3. Type check with mypy (optional):
   ```bash
   mypy src/ --ignore-missing-imports
   ```

## Notes

- black uses 100 char line length (project convention)
- isort uses black profile for compatibility
- Fix any type errors reported by mypy
