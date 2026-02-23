---
globs: terraform/**
---

# Terraform Rules for LCM DocIntel

## Module Structure
Every Terraform module must contain: `main.tf`, `variables.tf`, `outputs.tf`, `versions.tf`
Root structure: `terraform/modules/` with subdirectories per concern (vpc, compute, database, storage, security, monitoring).
Environments: `terraform/environments/{env}/main.tf` that calls modules.

## Naming Convention
- Resources: `{client_id}-{env}-{resource}` (e.g., `acme-prod-vpc`, `acme-prod-gpu-sg`)
- Variables: `snake_case`, always include `description` and `type`
- Outputs: `snake_case`, always include `description`
- Locals: `snake_case`, use for computed values and tag maps

## Required Tags on ALL Resources
```hcl
locals {
  common_tags = {
    client_id   = var.client_id
    environment = var.environment
    project     = "docintel"
    managed_by  = "terraform"
  }
}
```
Merge with resource-specific tags: `tags = merge(local.common_tags, { Name = "..." })`

## Security Non-Negotiables
- S3: `block_public_access` all four settings = true, SSE-KMS with per-client CMK, versioning enabled
- RDS: `storage_encrypted = true`, `publicly_accessible = false`, SSL enforced via parameter group
- EBS: `encrypted = true` with `kms_key_id` pointing to client CMK
- Security groups: NEVER allow `0.0.0.0/0` ingress (except ALB on port 443)
- VPC flow logs: enabled, stored in S3 with lifecycle policy
- State backend: S3 with encryption + DynamoDB locking, never committed to git
- IAM policies: least-privilege, no wildcard `*` on Resource

## State Backend
```hcl
terraform {
  backend "s3" {
    bucket         = "docintel-terraform-state"
    key            = "clients/{client_id}/{env}/terraform.tfstate"
    region         = "eu-central-1"
    encrypt        = true
    dynamodb_table = "docintel-terraform-locks"
  }
}
```

## Provider Version Pinning
Always pin provider versions with `~>` constraint:
```hcl
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}
```

## Before Suggesting Apply
Always run in order:
1. `terraform fmt -recursive`
2. `terraform validate`
3. `terraform plan` — show the full plan to user
4. Only apply after user confirms
