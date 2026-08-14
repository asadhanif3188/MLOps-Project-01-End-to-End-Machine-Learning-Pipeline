# Outputs.
#
# In this foundation PR the outputs describe the *context* Terraform is
# operating in (region, account, naming, tags) rather than provisioned
# infrastructure, since no resources exist yet. Later PRs add outputs for the
# VPC, subnets, IAM roles, and EKS connection details.
#
# The account ID is marked `sensitive` so it is not printed to the console or CI
# logs; it is still available to tooling via `terraform output -raw`.

output "aws_region" {
  description = "AWS region Terraform is operating in (resolved from the active provider configuration)."
  value       = data.aws_region.current.name
}

output "aws_account_id" {
  description = "AWS account ID resolved from the active credentials. Marked sensitive to keep the account identifier out of logs."
  value       = data.aws_caller_identity.current.account_id
  sensitive   = true
}

output "name_prefix" {
  description = "Canonical \"<project>-<environment>\" prefix that later PRs apply to resource names."
  value       = local.name_prefix
}

output "common_tags" {
  description = "The common tag set applied to every resource via the provider's default_tags."
  value       = local.common_tags
}
