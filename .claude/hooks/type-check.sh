#!/bin/bash
# Hook: PostEdit on frontend/**/*.ts,tsx
# Runs TypeScript type checking

set -e

echo "🔍 Type checking frontend..."
cd frontend && npx tsc --noEmit
echo "✓ TypeScript check passed"
