#!/bin/bash
# Agent Stop Hook - Verification Reminder
#
# This hook runs when Claude finishes a task.
# It reminds you to verify changes before considering the task complete.

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    exit 0
fi

# Check for uncommitted changes
if ! git diff --quiet HEAD 2>/dev/null; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Changes detected - consider running /verify"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi
