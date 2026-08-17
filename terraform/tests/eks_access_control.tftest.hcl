# Terraform contract tests for the EKS access model (Sprint 7, PR 3).
#
# These are OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"`
# replaces the real provider so no AWS API is ever called, and every run uses
# `command = plan`, so nothing is created. The suite pins the EXPLICIT-ACCESS
# contract that closes Sprint 6 finding H-03: the cluster no longer grants the
# creating principal implicit cluster-admin, authentication uses EKS access
# entries, the old insecure bootstrap setting is rejected, and access entries can
# only reference valid principals and scoped AWS-managed EKS access policies. It
# is executable proof that the secure access model is enforced by the
# configuration itself — not merely documented — in the same no-AWS spirit as the
# rest of the Terraform CI gate (ADR-019, ADR-023). It complements
# eks_api_security.tftest.hcl (H-02) and the static fmt/validate/tflint/trivy gates.
#
# Run locally with:  terraform test   (from terraform/)

# Pin the provider-context data sources to realistic values. The default random
# mocks would otherwise feed an invalid partition into the IAM/EKS policy ARNs and
# an empty AZ list into the network slice(), failing the plan on unrelated resources.
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

# Default configuration: creator-admin bootstrap is OFF and authentication is
# access-entries-only. The regression guard for H-03 — if a future change
# re-enables the creator-admin bootstrap or reverts to a ConfigMap-only auth mode
# by default, this run fails.
run "eks_access_is_explicit_by_default" {
  command = plan

  assert {
    condition     = var.cluster_bootstrap_creator_admin_permissions == false
    error_message = "SECURITY REGRESSION (H-03): cluster_bootstrap_creator_admin_permissions must default to false — the cluster creator must not receive automatic cluster-admin."
  }

  assert {
    condition     = var.cluster_authentication_mode == "API"
    error_message = "The default authentication mode must be \"API\" (access entries only), the explicit access model that replaced the aws-auth/creator-admin bootstrap."
  }

  assert {
    condition     = aws_eks_cluster.this.access_config[0].bootstrap_cluster_creator_admin_permissions == false
    error_message = "The cluster's access_config must carry bootstrap_cluster_creator_admin_permissions = false through to the resource."
  }

  assert {
    condition     = aws_eks_cluster.this.access_config[0].authentication_mode == "API"
    error_message = "The cluster's access_config must carry authentication_mode = API through to the resource."
  }
}

# The old insecure bootstrap setting is rejected by variable validation — the
# tripwire that stops creator-admin from being reintroduced, even deliberately.
run "eks_rejects_creator_admin_bootstrap" {
  command = plan

  variables {
    cluster_bootstrap_creator_admin_permissions = true
  }

  expect_failures = [
    var.cluster_bootstrap_creator_admin_permissions,
  ]
}

# A ConfigMap-only authentication mode (aws-auth only, access entries ignored) is
# rejected — it would bypass the access-entry model H-03 mandates.
run "eks_rejects_configmap_only_auth_mode" {
  command = plan

  variables {
    cluster_authentication_mode = "CONFIG_MAP"
  }

  expect_failures = [
    var.cluster_authentication_mode,
  ]
}

# A non-IAM / malformed principal ARN is rejected by validation.
run "eks_rejects_invalid_principal_arn" {
  command = plan

  variables {
    cluster_access_entries = {
      bad = {
        principal_arn = "not-an-arn"
        policy        = "AmazonEKSAdminPolicy"
      }
    }
  }

  expect_failures = [
    var.cluster_access_entries,
  ]
}

# An access policy outside the AWS-managed EKS access-policy set is rejected.
run "eks_rejects_unknown_access_policy" {
  command = plan

  variables {
    cluster_access_entries = {
      operator = {
        principal_arn = "arn:aws:iam::123456789012:role/operator"
        policy        = "AdministratorAccess"
      }
    }
  }

  expect_failures = [
    var.cluster_access_entries,
  ]
}

# A namespace-scoped entry with no namespaces is rejected — it would grant nothing
# usefully scoped and is almost certainly a mistake.
run "eks_rejects_namespace_scope_without_namespaces" {
  command = plan

  variables {
    cluster_access_entries = {
      operator = {
        principal_arn = "arn:aws:iam::123456789012:role/operator"
        policy        = "AmazonEKSEditPolicy"
        access_scope  = "namespace"
        namespaces    = []
      }
    }
  }

  expect_failures = [
    var.cluster_access_entries,
  ]
}

# The supported shape: an explicit operator entry with a scoped managed policy.
# This must plan cleanly and wire an access entry + a policy association carrying
# the correctly-constructed EKS cluster-access-policy ARN — proving access is
# grantable explicitly (requirement 3) via a narrow policy (requirement 4).
run "eks_explicit_access_entry_plans_cleanly" {
  command = plan

  variables {
    cluster_access_entries = {
      operator = {
        principal_arn = "arn:aws:iam::123456789012:role/mlops-operator"
        policy        = "AmazonEKSAdminPolicy"
        access_scope  = "cluster"
      }
    }
  }

  assert {
    condition     = aws_eks_access_entry.this["operator"].principal_arn == "arn:aws:iam::123456789012:role/mlops-operator"
    error_message = "The operator principal ARN should flow through to the access entry unchanged."
  }

  assert {
    condition     = aws_eks_access_entry.this["operator"].type == "STANDARD"
    error_message = "Human/automation access entries should be of type STANDARD."
  }

  assert {
    condition     = aws_eks_access_policy_association.this["operator"].policy_arn == "arn:aws:eks::aws:cluster-access-policy/AmazonEKSAdminPolicy"
    error_message = "The scoped managed policy should be associated as an EKS cluster-access-policy ARN built from the resolved partition and the policy short name."
  }

  assert {
    condition     = aws_eks_access_policy_association.this["operator"].access_scope[0].type == "cluster"
    error_message = "A cluster-scoped entry should associate the policy at cluster scope."
  }
}

# A namespace-scoped, read-only entry is also supported and narrows access — this
# is the least-privilege path (requirements 4 & 5): view-only, single namespace.
run "eks_namespace_scoped_view_entry_plans_cleanly" {
  command = plan

  variables {
    cluster_access_entries = {
      observer = {
        principal_arn = "arn:aws:iam::123456789012:role/mlops-observer"
        policy        = "AmazonEKSViewPolicy"
        access_scope  = "namespace"
        namespaces    = ["mlops"]
      }
    }
  }

  assert {
    condition     = aws_eks_access_policy_association.this["observer"].policy_arn == "arn:aws:eks::aws:cluster-access-policy/AmazonEKSViewPolicy"
    error_message = "A view-only entry should associate AmazonEKSViewPolicy."
  }

  assert {
    condition     = aws_eks_access_policy_association.this["observer"].access_scope[0].type == "namespace"
    error_message = "A namespace-scoped entry should associate the policy at namespace scope."
  }

  assert {
    condition     = contains(aws_eks_access_policy_association.this["observer"].access_scope[0].namespaces, "mlops")
    error_message = "The requested namespace should flow through to the policy association's access scope."
  }
}
