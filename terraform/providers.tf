# AWS provider configuration.
#
# Authentication is intentionally NOT expressed here. No access keys, profiles,
# or account IDs are hard-coded; the provider resolves credentials from the
# standard AWS chain (environment variables, `~/.aws` config/credentials, or an
# assumed IAM role) at plan/apply time. This keeps the repository safe to
# publish and free of committed secrets (see terraform/README.md § Security).
#
# `default_tags` applies the project's common tag set (defined in main.tf) to
# every taggable resource created by this provider, so tagging is consistent by
# construction rather than remembered per-resource in later PRs.

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}
