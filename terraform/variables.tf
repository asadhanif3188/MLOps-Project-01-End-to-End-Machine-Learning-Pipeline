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
