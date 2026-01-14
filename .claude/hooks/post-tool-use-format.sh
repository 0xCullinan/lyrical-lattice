#!/bin/bash
# PostToolUse hook - Auto-format Python files after edits
#
# This hook runs after Claude writes/edits Python files.
# It ensures consistent formatting without CI failures.

# Get the file path from the hook context
FILE_PATH="$1"

# Only format Python files
if [[ "$FILE_PATH" == *.py ]]; then
    # Check if black is available
    if command -v black &> /dev/null; then
        black "$FILE_PATH" --quiet --line-length 100 2>/dev/null || true
    fi

    # Check if isort is available
    if command -v isort &> /dev/null; then
        isort "$FILE_PATH" --quiet --profile black 2>/dev/null || true
    fi
fi
