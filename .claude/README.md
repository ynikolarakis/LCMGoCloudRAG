# .claude/ Directory — Claude Code Configuration

This directory contains all configuration for Claude Code's AI-assisted development environment.

## Structure

```
.claude/
├── agents/                    # Specialized AI subagents
│   ├── terraform-infra.md     # AWS Terraform modules (Sonnet)
│   ├── python-backend.md      # FastAPI + Haystack + Celery (Sonnet)
│   ├── rag-specialist.md      # RAG pipeline design + evaluation (Opus)
│   ├── security-reviewer.md   # Security audit — READ-ONLY (Sonnet)
│   ├── frontend-engineer.md   # React/Next.js chat UI (Sonnet)
│   └── devops.md              # Docker, CI/CD, ECS deployment (Sonnet)
├── rules/                     # Scoped instruction files (auto-loaded by glob)
│   ├── terraform.md           # → terraform/**
│   ├── python.md              # → backend/**/*.py
│   ├── security.md            # → **/* (all files)
│   ├── frontend.md            # → frontend/**
│   └── rag-pipeline.md        # → backend/app/pipeline/**, backend/app/services/rag/**
├── hooks/                     # Deterministic scripts (run outside AI loop)
│   ├── lint-python.sh         # PostEdit: ruff check + format
│   ├── validate-terraform.sh  # PostEdit: terraform fmt + validate
│   ├── check-secrets.sh       # PreCommit: gitleaks scan
│   ├── run-tests.sh           # Stop: pytest on modified modules
│   └── type-check.sh          # PostEdit: tsc --noEmit
└── settings.json              # Claude Code project settings (auto-generated)
```

## How It Works

1. **CLAUDE.md** (project root) is read first every session — contains universal project context
2. **Rules** are loaded automatically when you edit files matching their glob pattern
3. **Agents** are delegated to by the main Claude Code agent for domain-specific tasks
4. **Hooks** run deterministic scripts on specific events (edit, commit, session end)

## MCP Servers

Run `scripts/setup-mcp-servers.sh` to configure all MCP servers.
Check status with `/mcp` in Claude Code.

## First Session

See `docs/FIRST_PROMPT.md` for the bootstrap prompt to start building.
