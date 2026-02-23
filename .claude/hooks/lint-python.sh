#!/bin/bash
# Hook: PostEdit on backend/**/*.py
# Runs ruff lint + format check on edited Python files

set -e

# Get list of changed Python files from git
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMR HEAD -- 'backend/**/*.py' 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
    echo "✓ No Python files changed"
    exit 0
fi

echo "🔍 Linting Python files..."
ruff check $CHANGED_FILES
echo "🔍 Checking format..."
ruff format --check $CHANGED_FILES
echo "✓ Python lint passed"
