---
globs: **/*
---

# Security Rules — Apply to ALL Files

## Secrets Management
- **NEVER** hardcode passwords, API keys, tokens, connection strings, or credentials
- Dev: `.env` file (MUST be in `.gitignore`)
- Prod: AWS Secrets Manager (encrypted with per-client KMS CMK)
- Terraform: use `data "aws_secretsmanager_secret_version"` — never inline secrets
- Docker: never put secrets in Dockerfile or docker-compose.yml — use env_file or runtime injection
- If you see ANY hardcoded secret, flag it as **CRITICAL** immediately and replace with env var

## Encryption
- At rest: AES-256 via AWS KMS CMK (per-client key) — S3, RDS, EBS, Qdrant backups
- In transit: TLS 1.3 on ALL connections (external AND internal service-to-service)
- Qdrant: API key authentication + TLS between API server and Qdrant
- PostgreSQL: `sslmode=require` on all connection strings
- Redis: TLS enabled in ElastiCache, password auth in dev

## GDPR Compliance
- PII detection via LLM Guard BEFORE indexing into Qdrant (mask or flag)
- Right-to-erasure: document delete MUST cascade through S3 → Qdrant → PostgreSQL → audit entry
- Audit log entries are IMMUTABLE — no DELETE permission on audit table (only INSERT and SELECT)
- Data residency: data stays in configured AWS region (no cross-region replication without explicit consent)
- User consent: tracked in database with timestamp and scope
- Data export: admin endpoint to export all user data for right-to-access requests
- Data retention: configurable per client, automated cleanup via Celery scheduled tasks

## Authentication & Authorization
- Keycloak: per-client realm, OIDC/SAML integration
- JWT tokens: 15-minute access tokens, 7-day refresh tokens
- JWT storage: httpOnly secure cookies only (NEVER localStorage)
- RBAC: Admin, Manager, User, Viewer roles with document-level ACLs
- API: validate JWT on every request via middleware
- CORS: explicit allowed origins (never wildcard `*` in production)
- CSRF: token validation on all state-changing requests
- Rate limiting: on auth endpoints (login, refresh, register)

## Input Validation
- All user inputs validated server-side (Pydantic models)
- File uploads: validate type (whitelist), size (configurable max), scan for malware
- SQL: parameterized queries only (SQLAlchemy handles this — never raw string formatting)
- XSS: sanitize all user-generated content before rendering
- Prompt injection: NeMo Guardrails + LLM Guard on ALL user queries before LLM processing

## Network
- No 0.0.0.0/0 ingress on security groups (except ALB port 443)
- Backend services in private subnets only
- VPC endpoints for S3, KMS, Secrets Manager (no public internet for data tier)
- VPC flow logs enabled and stored in S3
- WAF on ALB with OWASP Core Rule Set

## Audit Logging
Every significant action must be logged to the immutable audit table:
- User login/logout
- Document upload, view, delete
- Query submitted (full text)
- LLM response generated (full text + citations)
- Guardrail triggers (blocked queries/responses)
- Admin actions (user management, config changes, permission changes)
- Retention: 90 days hot (PostgreSQL), 7 years cold (S3 Glacier)
