# Dev environment - local development only
# Real AWS infrastructure provisioned in client-specific environments

terraform {
  required_version = ">= 1.7"
}

locals {
  client_id   = "dev"
  environment = "dev"
}
