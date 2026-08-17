# Amazon ECR container registry (Sprint 7, PR 1 — closes Sprint 6 finding H-01).
#
# The private registry that stores the MLOps workload image the EKS node group
# pulls at run time. Through Sprint 6 this repository was created out-of-band with
# `aws ecr create-repository` and deleted out-of-band with `aws ecr
# delete-repository --force`, which left one live AWS resource *outside* Terraform
# state (finding H-01): it was not provisioned, tagged, versioned, or torn down by
# `terraform apply`/`destroy` like everything else. This file brings it fully under
# Terraform management so the registry has the same lifecycle, tagging, and
# reproducibility as the VPC, IAM, and EKS resources.
#
# Design rationale (naming, immutability, scanning, retention, encryption, and the
# teardown change) lives in ADR-021 and terraform/README.md § Container registry,
# not repeated in these comments. Common Project/Environment/Owner tags are applied
# to every resource automatically via the provider's default_tags (providers.tf);
# only the resource-specific Name tag is set below.
#
# Security posture (explicitly preserved, never weakened to simplify):
#   - PRIVATE registry — no repository policy granting public/cross-account access
#     is authored here, so the repository is reachable only by the account's own
#     principals (the node role's AmazonEC2ContainerRegistryReadOnly from ADR-016).
#   - scan_on_push stays ON (image vulnerability scanning is a security feature).
#   - image_tag_mutability = IMMUTABLE so a pushed tag cannot be overwritten — this
#     matches the "explicit, immutable version tag, never :latest" convention the
#     AWS overlay already relies on (k8s/overlays/aws) for reproducible pulls.
#   - Encrypted at rest with the AWS-managed key (AES256, the ECR default). A
#     customer-managed KMS CMK is the documented hardening follow-up (tracked with
#     the EKS-secrets KMS work, finding M-02); it is deliberately not bundled here
#     to keep this PR to the H-01 boundary.

# The repository name is NOT environment-scoped via local.name_prefix like the EKS
# resources are: the image registry is a per-project artifact store, and the name
# must stay in lock-step with the image reference already committed in the AWS
# overlay (k8s/overlays/aws/kustomization.yaml uses ".../mlops-pipeline"). It
# therefore defaults to var.project_name; ecr_repository_name overrides it only if
# a different registry name is genuinely needed.
locals {
  ecr_repository_name = coalesce(var.ecr_repository_name, var.project_name)
}

resource "aws_ecr_repository" "this" {
  name = local.ecr_repository_name

  # Immutable tags: a version tag (e.g. "1.3.1") can never be repointed at a new
  # image, so a deployed digest is reproducible and the overlay's static
  # image-pinning contract holds.
  image_tag_mutability = "IMMUTABLE"

  # Vulnerability scan every pushed image. This is a security feature and stays on.
  image_scanning_configuration {
    scan_on_push = true
  }

  # Encryption at rest with the AWS-managed key (ECR's default). A customer-managed
  # KMS CMK is the documented follow-up (see ADR-021 / M-02), not a regression.
  encryption_configuration {
    encryption_type = "AES256"
  }

  # Let `terraform destroy` remove the repository even if it still holds images.
  # This is what replaces the manual `aws ecr delete-repository --force` teardown
  # step and makes the ephemeral environment (ADR-020) fully Terraform-torn-down.
  # Appropriate for a short-lived, single-operator validation registry; a
  # persistent/production registry would set this false to prevent accidental loss.
  force_delete = true

  tags = {
    Name = local.ecr_repository_name
  }
}

# Lifecycle policy — caps image accumulation so storage cannot grow unbounded as
# repeated validation runs push new tags. Keeps only the most recent
# ecr_max_image_count images and expires older ones. This is a cost/hygiene
# control, complementary to the ephemeral-environment teardown (ADR-020): even
# within a single environment's life, old images are reaped automatically.
resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire all but the most recent ${var.ecr_max_image_count} images to cap registry storage."
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.ecr_max_image_count
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
