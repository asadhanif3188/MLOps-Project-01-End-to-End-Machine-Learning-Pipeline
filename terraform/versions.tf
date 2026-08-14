# Terraform and provider version constraints.
#
# Pinning these is the first line of reproducibility for infrastructure code:
# two engineers (or CI and a laptop) resolve the same Terraform core and the
# same AWS provider, so a `plan` means the same thing everywhere.
#
#   - required_version pins the Terraform CLI to the 1.x line. The lower bound
#     guarantees features we rely on (native `default_tags`, input validation);
#     the `< 2.0.0` upper bound keeps a future breaking major from silently
#     changing behaviour.
#   - The AWS provider is pinned with a pessimistic `~> 5.x` constraint so we
#     get patch/minor fixes but never an unreviewed major bump.

terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}
