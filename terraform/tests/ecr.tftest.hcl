# Terraform contract tests for the ECR configuration (Sprint 7, PR 1).
#
# These are OFFLINE, credential-free `terraform test` runs: `mock_provider "aws"`
# replaces the real provider so no AWS API is ever called (data sources such as
# aws_caller_identity/aws_region/aws_partition return mocked values). Every run
# uses `command = plan`, so nothing is created. The suite pins the security- and
# lifecycle-relevant contract of the ECR repository so a regression — losing
# immutable tags, turning off scan-on-push, dropping encryption, or breaking the
# retention policy — fails fast, in the same no-AWS spirit as the rest of the
# Terraform CI gate (ADR-019). It complements, and does not replace, the static
# fmt/validate/tflint/trivy gates.
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

# Default configuration: name derives from project_name, tags immutable, scan on
# push, encrypted at rest, force_delete on for the ephemeral registry, and the
# retention policy present with the default count.
run "ecr_defaults_preserve_security_contract" {
  command = plan

  assert {
    condition     = aws_ecr_repository.this.name == "mlops-pipeline"
    error_message = "ECR repository name should default to project_name (\"mlops-pipeline\") to match the committed image reference in k8s/overlays/aws."
  }

  assert {
    condition     = aws_ecr_repository.this.image_tag_mutability == "IMMUTABLE"
    error_message = "ECR image tags must be IMMUTABLE so a version tag cannot be repointed (reproducible, pinned pulls)."
  }

  assert {
    condition     = aws_ecr_repository.this.image_scanning_configuration[0].scan_on_push == true
    error_message = "scan_on_push must stay enabled — image vulnerability scanning is a security feature and must not be disabled."
  }

  assert {
    condition     = aws_ecr_repository.this.encryption_configuration[0].encryption_type == "AES256"
    error_message = "ECR must be encrypted at rest (AES256, the AWS-managed default)."
  }

  assert {
    condition     = aws_ecr_lifecycle_policy.this.repository == aws_ecr_repository.this.name
    error_message = "A lifecycle policy must be attached to the ECR repository to cap image accumulation."
  }

  assert {
    condition     = strcontains(aws_ecr_lifecycle_policy.this.policy, "\"countNumber\":10")
    error_message = "The default lifecycle policy should retain 10 images (ecr_max_image_count default)."
  }

  assert {
    condition     = strcontains(aws_ecr_lifecycle_policy.this.policy, "\"type\":\"expire\"")
    error_message = "The lifecycle policy must expire images beyond the retention count."
  }
}

# The MLflow server registry (Sprint 7, PR 6) exists and carries the SAME hardened
# contract as the pipeline registry: fixed name matching the AWS overlay, immutable
# tags, scan-on-push, encryption, and a retention policy.
run "mlflow_server_ecr_preserves_security_contract" {
  command = plan

  assert {
    condition     = aws_ecr_repository.mlflow_server.name == "mlflow-server"
    error_message = "The MLflow server ECR repository must be named \"mlflow-server\" to match the committed image reference in k8s/overlays/aws."
  }

  assert {
    condition     = aws_ecr_repository.mlflow_server.image_tag_mutability == "IMMUTABLE"
    error_message = "MLflow server ECR image tags must be IMMUTABLE (reproducible, pinned pulls)."
  }

  assert {
    condition     = aws_ecr_repository.mlflow_server.image_scanning_configuration[0].scan_on_push == true
    error_message = "scan_on_push must stay enabled on the MLflow server registry."
  }

  assert {
    condition     = aws_ecr_repository.mlflow_server.encryption_configuration[0].encryption_type == "AES256"
    error_message = "The MLflow server registry must be encrypted at rest (AES256)."
  }

  assert {
    condition     = aws_ecr_lifecycle_policy.mlflow_server.repository == aws_ecr_repository.mlflow_server.name
    error_message = "A lifecycle policy must be attached to the MLflow server registry to cap image accumulation."
  }
}

# The repository name is overridable without touching resource definitions.
run "ecr_repository_name_override" {
  command = plan

  variables {
    ecr_repository_name = "custom-registry"
  }

  assert {
    condition     = aws_ecr_repository.this.name == "custom-registry"
    error_message = "ecr_repository_name should override the default repository name."
  }
}

# The retention count is variable-driven and flows into the lifecycle policy JSON.
run "ecr_retention_count_override" {
  command = plan

  variables {
    ecr_max_image_count = 25
  }

  assert {
    condition     = strcontains(aws_ecr_lifecycle_policy.this.policy, "\"countNumber\":25")
    error_message = "ecr_max_image_count should drive the lifecycle policy's retained image count."
  }
}

# Input validation: an out-of-range retention count is rejected before any plan.
run "ecr_retention_count_rejects_zero" {
  command = plan

  variables {
    ecr_max_image_count = 0
  }

  expect_failures = [
    var.ecr_max_image_count,
  ]
}
