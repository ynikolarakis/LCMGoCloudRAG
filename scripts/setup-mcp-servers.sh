#!/bin/bash
# ============================================================================
# LCM DocIntel — Claude Code MCP Server Setup
# ============================================================================
# Run this script from your project root to configure all MCP servers.
# Prerequisites: Docker, Node.js (npx), Python (uvx), GitHub PAT
#
# Usage:
#   chmod +x scripts/setup-mcp-servers.sh
#   ./scripts/setup-mcp-servers.sh
# ============================================================================

set -e

echo "============================================"
echo "  LCM DocIntel — MCP Server Setup"
echo "============================================"
echo ""

# ── Check prerequisites ──────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found. Install Docker Desktop first."; exit 1; }
command -v npx >/dev/null 2>&1 || { echo "❌ npx not found. Install Node.js first."; exit 1; }
command -v uvx >/dev/null 2>&1 || { echo "⚠️  uvx not found. Install with: pip install uv"; }

echo "✓ Prerequisites check passed"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────
# Replace these with your actual values before running
GITHUB_TOKEN="${GITHUB_TOKEN:-ghp_YOUR_GITHUB_PAT_HERE}"
AWS_PROFILE="${AWS_PROFILE:-lcmgocloud}"
POSTGRES_URL="${POSTGRES_URL:-postgresql://ragadmin:devpassword@localhost:5432/docintel}"

# ============================================================================
# ESSENTIAL MCP SERVERS (Install Day 1)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installing ESSENTIAL MCP Servers..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. GitHub — PRs, issues, code search, CI/CD
echo "📦 [1/7] GitHub MCP Server..."
claude mcp add github -s user \
  -- docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN="$GITHUB_TOKEN" \
  ghcr.io/github/github-mcp-server
echo "   ✓ GitHub configured"
echo ""

# 2. Terraform (HashiCorp Official) — Provider schemas, module registry
echo "📦 [2/7] Terraform MCP Server (HashiCorp)..."
claude mcp add terraform -s project \
  -- docker run -i --rm hashicorp/terraform-mcp-server
echo "   ✓ Terraform (HashiCorp) configured"
echo ""

# 3. AWS Terraform (AWS Labs) — AWS-specific TF guidance + Checkov
echo "📦 [3/7] AWS Terraform MCP Server..."
claude mcp add awslabs-terraform -s project \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.terraform-mcp-server@latest
echo "   ✓ AWS Terraform configured"
echo ""

# 4. AWS Documentation — Real-time AWS docs and API references
echo "📦 [4/7] AWS Documentation MCP Server..."
claude mcp add aws-docs -s user \
  -e AWS_DOCUMENTATION_PARTITION=aws \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-documentation-mcp-server@latest
echo "   ✓ AWS Docs configured"
echo ""

# 5. Docker — Container management, image builds
echo "📦 [5/7] Docker MCP Server..."
claude mcp add docker -s user \
  -- npx -y docker-mcp-server
echo "   ✓ Docker configured"
echo ""

# 6. PostgreSQL — Schema design, queries, data inspection
echo "📦 [6/7] PostgreSQL MCP Server..."
claude mcp add postgres -s project \
  -- npx -y @modelcontextprotocol/server-postgres "$POSTGRES_URL"
echo "   ✓ PostgreSQL configured"
echo ""

# 7. Context7 — Latest library docs (Haystack, Qdrant, FastAPI, etc.)
echo "📦 [7/7] Context7 MCP Server..."
claude mcp add context7 -s user \
  -- npx -y @upstash/context7-mcp@latest
echo "   ✓ Context7 configured"
echo ""

# ============================================================================
# RECOMMENDED MCP SERVERS (Install Phase 1-2)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installing RECOMMENDED MCP Servers..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 8. Sequential Thinking — Structured reasoning for architecture decisions
echo "📦 [8/12] Sequential Thinking MCP Server..."
claude mcp add thinking -s user \
  -- npx -y @modelcontextprotocol/server-sequential-thinking
echo "   ✓ Sequential Thinking configured"
echo ""

# 9. AWS Core — Orchestrates all AWS MCP servers
echo "📦 [9/12] AWS Core MCP Server..."
claude mcp add aws-core -s project \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.core-mcp-server@latest
echo "   ✓ AWS Core configured"
echo ""

# 10. AWS Cost Analysis — Pre-deployment cost estimation
echo "📦 [10/12] AWS Cost Analysis MCP Server..."
claude mcp add aws-costs -s project \
  -e AWS_PROFILE="$AWS_PROFILE" \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cost-analysis-mcp-server@latest
echo "   ✓ AWS Cost Analysis configured"
echo ""

# 11. Playwright — Browser testing for React frontend
echo "📦 [11/12] Playwright MCP Server..."
claude mcp add playwright -s project \
  -- npx -y @playwright/mcp@latest
echo "   ✓ Playwright configured"
echo ""

# 12. Memory — Persistent knowledge graph across sessions
echo "📦 [12/12] Memory MCP Server..."
claude mcp add memory -s user \
  -- npx -y @modelcontextprotocol/server-memory
echo "   ✓ Memory configured"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================"
echo "  ✅ MCP Server Setup Complete!"
echo "============================================"
echo ""
echo "  Essential (7):  GitHub, Terraform, AWS Terraform,"
echo "                  AWS Docs, Docker, PostgreSQL, Context7"
echo ""
echo "  Recommended (5): Sequential Thinking, AWS Core,"
echo "                   AWS Cost Analysis, Playwright, Memory"
echo ""
echo "  Verify with:    claude /mcp"
echo "  Disable/enable: claude /mcp (toggle per session)"
echo ""
echo "  ⚠️  Remember to update:"
echo "     - GITHUB_TOKEN with your actual GitHub PAT"
echo "     - AWS_PROFILE with your AWS CLI profile name"
echo "     - POSTGRES_URL if your local DB differs"
echo "============================================"
