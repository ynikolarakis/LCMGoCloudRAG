#!/bin/bash
# Hook: PostEdit on terraform/**
# Runs terraform fmt check + validate on Terraform files

set -e

echo "🔍 Checking Terraform formatting..."
terraform fmt -check -recursive terraform/
echo "🔍 Validating Terraform configuration..."
cd terraform && terraform validate
echo "✓ Terraform validation passed"
