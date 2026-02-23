#!/bin/bash
# Hook: Stop (runs before Claude Code session ends)
# Runs pytest on modified backend modules

set -e

CHANGED_FILES=$(git diff --name-only --diff-filter=ACMR HEAD -- 'backend/**/*.py' 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
    echo "✓ No backend files changed — skipping tests"
    exit 0
fi

echo "🧪 Running tests on modified modules..."
pytest tests/ -x --tb=short -q
echo "✓ All tests passed"
