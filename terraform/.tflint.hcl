# TFLint configuration for the terraform/ module.
#
# TFLint is a STATIC linter: it inspects the Terraform source and never contacts
# AWS. It runs in CI (see .github/workflows/ci.yml § terraform-validate) and can
# be run locally the same way — no AWS credentials are read or required.
#
# Two rulesets are enabled:
#   * terraform (bundled) — the language best-practices preset: naming
#     conventions, unused declarations, missing `type`/`description` on variables,
#     deprecated syntax, provider version pinning, etc.
#   * aws (plugin) — AWS-specific correctness: invalid instance types, malformed
#     ARNs/values, deprecated resource arguments. `tflint --init` downloads this
#     pinned plugin from GitHub; it is inert linting only and reads no cloud state.
#
# Versions are pinned so a green lint today is a green lint tomorrow; bump
# deliberately.

config {
  # Analyse module source only. `terraform init` is not required for linting and
  # no remote modules are used here.
  call_module_type = "local"
}

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

plugin "aws" {
  enabled = true
  version = "0.37.0"
  source  = "github.com/terraform-linters/tflint-ruleset-aws"
}
