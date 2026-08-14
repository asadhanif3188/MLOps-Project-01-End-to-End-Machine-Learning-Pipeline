# Input variables for the AWS foundation.
#
# Every variable has a description (surfaced by `terraform` tooling and docs) and
# a safe, non-secret default so the configuration is valid out of the box. None
# of these carry credentials — secrets never travel through Terraform variables
# in this project (see terraform/README.md § Security).

variable "aws_region" {
  description = "AWS region in which infrastructure is provisioned (e.g. \"us-east-1\"). Drives the provider and all regional resources added in later PRs."
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}(-[a-z]+)+-[0-9]$", var.aws_region))
    error_message = "aws_region must be a valid AWS region identifier, e.g. \"us-east-1\" or \"eu-west-2\"."
  }
}

variable "project_name" {
  description = "Short project identifier used as the naming/tagging prefix for all resources. Lowercase letters, digits and hyphens only."
  type        = string
  default     = "mlops-pipeline"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}[a-z0-9]$", var.project_name))
    error_message = "project_name must be 3-32 chars: lowercase alphanumeric and hyphens, starting with a letter and not ending in a hyphen."
  }
}

variable "environment" {
  description = "Deployment environment name. Participates in resource naming and tagging so dev/staging/prod resources are unambiguous."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner/steward recorded in the \"Owner\" tag for cost attribution and accountability. Not a secret."
  type        = string
  default     = "asadhanif3188"
}

variable "repository_url" {
  description = "Source repository URL recorded in the \"Repository\" tag so provisioned resources trace back to the code that created them."
  type        = string
  default     = "https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline"
}

variable "additional_tags" {
  description = "Optional extra tags merged into (and able to extend, not override the reserved keys of) the common tag set."
  type        = map(string)
  default     = {}
}

# --- Network (Sprint 6, PR 2) -------------------------------------------------
# The VPC that hosts the EKS cluster added in a later PR. Everything below is
# variable-driven so the same configuration works for a small dev VPC or a
# larger footprint without editing resource definitions. See ADR-015 and
# terraform/README.md § Network architecture for the design rationale.

variable "vpc_cidr" {
  description = "IPv4 CIDR block for the VPC. A /16 gives ample room for the /24 per-AZ subnets this configuration derives; the prefix is capped at /20 so derived subnets stay usefully sized for EKS ENIs/pods."
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0)) && tonumber(split("/", var.vpc_cidr)[1]) >= 16 && tonumber(split("/", var.vpc_cidr)[1]) <= 20
    error_message = "vpc_cidr must be a valid IPv4 CIDR with a prefix between /16 and /20 (e.g. \"10.0.0.0/16\")."
  }
}

variable "az_count" {
  description = "Number of Availability Zones to spread subnets across. EKS requires subnets in at least two AZs; AZ names are discovered at plan time, never hard-coded. Kept small (2) by default for cost; 3 is supported for broader spread."
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 3
    error_message = "az_count must be 2 or 3 (EKS needs at least two AZs; three is the practical upper bound for this portfolio scope)."
  }
}

variable "enable_nat_gateway" {
  description = "Whether to create NAT gateway(s) so private-subnet nodes have outbound internet egress (image pulls, dataset fetch, pip, package indexes). Required for the EKS worker nodes, which live in the private subnets. Disable only if nodes are placed in public subnets instead."
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "When NAT is enabled, use one shared NAT gateway for all AZs (true, cost-optimized for this short-lived portfolio environment) instead of one per AZ (false, AZ-fault-tolerant but ~1x NAT cost per additional AZ)."
  type        = bool
  default     = true
}
