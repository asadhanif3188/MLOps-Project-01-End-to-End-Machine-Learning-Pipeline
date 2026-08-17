# Terraform contract tests for the EKS API-access security posture (Sprint 7, PR 2).
#
# These are OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"`
# replaces the real provider so no AWS API is ever called, and every run uses
# `command = plan`, so nothing is created. The suite pins the SECURE-BY-DEFAULT
# contract of the EKS control-plane endpoint (closing Sprint 6 finding H-02): the
# API server is private by default, public access is an explicit opt-in, and an
# unrestricted 0.0.0.0/0 CIDR can never be configured. It is executable proof that
# the secure posture is enforced by the configuration itself — not merely
# documented — in the same no-AWS spirit as the rest of the Terraform CI gate
# (ADR-019, ADR-022). It complements the static fmt/validate/tflint/trivy gates.
#
# Run locally with:  terraform test   (from terraform/)

# Pin the provider-context data sources to realistic values. The default random
# mocks would otherwise feed an invalid partition into the IAM policy ARNs and an
# empty AZ list into the network slice(), failing the plan on unrelated resources.
mock_provider "aws" {
  mock_data "aws_partition" {
    defaults = {
      partition = "aws"
    }
  }

  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-1a", "us-east-1b", "us-east-1c"]
    }
  }
}

# Default configuration: the API server is PRIVATE-ONLY. Public access is off and
# private access is on — the regression guard for H-02: if a future change
# re-enables public access by default, this run fails. The public CIDR default is
# empty (no exposure), but that value is intentionally NOT asserted here because
# `terraform test` auto-loads an operator's git-ignored terraform.tfvars, which by
# convention records their own opt-in IP; the security-critical property is that
# the public endpoint itself is OFF by default, which the assertions below pin.
run "eks_api_is_private_by_default" {
  command = plan

  assert {
    condition     = var.cluster_endpoint_public_access == false
    error_message = "SECURITY REGRESSION (H-02): the EKS public API endpoint must default to OFF (private-by-default)."
  }

  assert {
    condition     = var.cluster_endpoint_private_access == true
    error_message = "The EKS private API endpoint must default to ON so the cluster is reachable from within the VPC."
  }

  assert {
    condition     = aws_eks_cluster.this.vpc_config[0].endpoint_public_access == false
    error_message = "The cluster's vpc_config must carry the private-by-default posture through to the resource (endpoint_public_access = false)."
  }

  assert {
    condition     = aws_eks_cluster.this.vpc_config[0].endpoint_private_access == true
    error_message = "The cluster's vpc_config must enable private endpoint access."
  }
}

# An unrestricted 0.0.0.0/0 API CIDR is rejected before any plan — the hard
# "never allow 0.0.0.0/0" rule, enforced by variable validation.
run "eks_rejects_unrestricted_ipv4_cidr" {
  command = plan

  variables {
    cluster_endpoint_public_access       = true
    cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]
  }

  expect_failures = [
    var.cluster_endpoint_public_access_cidrs,
  ]
}

# Any /0 (not just the canonical 0.0.0.0/0) is unrestricted and rejected.
run "eks_rejects_any_slash_zero_cidr" {
  command = plan

  variables {
    cluster_endpoint_public_access       = true
    cluster_endpoint_public_access_cidrs = ["10.0.0.0/0"]
  }

  expect_failures = [
    var.cluster_endpoint_public_access_cidrs,
  ]
}

# A syntactically invalid CIDR is rejected by the validity check.
run "eks_rejects_invalid_cidr" {
  command = plan

  variables {
    cluster_endpoint_public_access       = true
    cluster_endpoint_public_access_cidrs = ["not-a-cidr"]
  }

  expect_failures = [
    var.cluster_endpoint_public_access_cidrs,
  ]
}

# Opting into public access WITHOUT a CIDR allow-list is rejected by the cluster
# precondition: EKS would otherwise fall back to 0.0.0.0/0 for an empty list, so
# "public on, list empty" is treated as the insecure configuration it is.
run "eks_public_access_requires_explicit_cidrs" {
  command = plan

  variables {
    cluster_endpoint_public_access       = true
    cluster_endpoint_public_access_cidrs = []
  }

  expect_failures = [
    aws_eks_cluster.this,
  ]
}

# Disabling BOTH endpoints is rejected: the API server would be unreachable.
run "eks_rejects_both_endpoints_disabled" {
  command = plan

  variables {
    cluster_endpoint_public_access  = false
    cluster_endpoint_private_access = false
  }

  expect_failures = [
    aws_eks_cluster.this,
  ]
}

# The supported secure opt-in: public access enabled AND scoped to a specific
# operator /32. This must plan cleanly and carry the scoped CIDR to the resource,
# proving public access remains *configurable* (requirement 3) without ever being
# unrestricted.
run "eks_public_access_optin_with_scoped_cidr" {
  command = plan

  variables {
    cluster_endpoint_public_access       = true
    cluster_endpoint_public_access_cidrs = ["203.0.113.10/32"]
  }

  assert {
    condition     = aws_eks_cluster.this.vpc_config[0].endpoint_public_access == true
    error_message = "A scoped public-access opt-in should plan successfully with the public endpoint enabled."
  }

  assert {
    condition     = contains(aws_eks_cluster.this.vpc_config[0].public_access_cidrs, "203.0.113.10/32")
    error_message = "The scoped operator CIDR should flow through to the cluster's public_access_cidrs."
  }
}
