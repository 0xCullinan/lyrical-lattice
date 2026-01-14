# /test - Run Test Suite

Run the project test suite and fix any failures.

## Instructions

1. Run pytest with verbose output:
   ```bash
   cd /Users/macbook/.gemini/antigravity/scratch/oronym-assistant && pytest -v
   ```

2. If tests fail:
   - Read the failing test file
   - Understand what's being tested
   - Fix the issue in the source code (not the test, unless the test is wrong)
   - Re-run tests to verify

3. Report summary:
   - Total tests run
   - Passed/Failed count
   - Any fixes made

## Options

- `--cov` - Run with coverage report
- `-k <pattern>` - Run tests matching pattern
- `<path>` - Run specific test file
