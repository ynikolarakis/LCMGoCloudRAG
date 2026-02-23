#!/bin/bash
# Hook: PreCommit
# Scans staged files for hardcoded secrets using gitleaks

set -e

echo "🔒 Scanning for secrets..."
gitleaks detect --source . --no-banner --verbose 2>&1

if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Secrets detected in code! Remove them before committing."
    echo "   Use .env for dev secrets, AWS Secrets Manager for prod."
    exit 1
fi

echo "✓ No secrets detected"
