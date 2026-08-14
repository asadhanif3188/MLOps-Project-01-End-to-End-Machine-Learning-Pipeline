# Root module — foundation only (Sprint 6, PR 1).
#
# This PR establishes a professional Terraform project WITHOUT provisioning any
# billable AWS resources. It declares no VPC, IAM, or EKS — those arrive in
# later Sprint 6 PRs (network → IAM → EKS). What it does establish is the shared
# vocabulary every later resource will lean on: a canonical name prefix and a
# single common tag set, plus provider-context lookups consumed by outputs.
#
# Keeping the foundation resource-free is deliberate: the configuration is valid
# and `terraform validate`-clean, yet an accidental `apply` creates nothing and
# costs nothing.

locals {
  # Canonical name prefix, e.g. "mlops-pipeline-dev". Later PRs derive resource
  # names from this so naming is consistent and environment-scoped by construction.
  name_prefix = "${var.project_name}-${var.environment}"

  # Common tags merged onto every resource via the provider's `default_tags`.
  # `additional_tags` is applied first so it can extend the set but cannot
  # override the reserved keys below (later entries in merge() win).
  common_tags = merge(
    var.additional_tags,
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
      Repository  = var.repository_url
    },
  )
}

# Provider-context lookups. These are inert during `terraform validate` (no API
# call) and resolve at plan/apply time using the caller's AWS credentials. They
# back the outputs below, which confirm *which* account/region Terraform is
# pointed at before any resource-creating PR runs.
data "aws_caller_identity" "current" {}

data "aws_region" "current" {}
