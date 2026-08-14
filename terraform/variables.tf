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

# --- EKS platform (Sprint 6, PR 4) --------------------------------------------
# The managed Kubernetes cluster and node group that run the MLOps workload.
# Everything is variable-driven so the cluster stays small and cost-conscious by
# default but can be resized without editing resource definitions. See ADR-017
# and terraform/README.md § EKS platform for the design rationale.

variable "kubernetes_version" {
  description = "Kubernetes minor version for the EKS control plane and managed node group, pinned explicitly for reproducibility (e.g. \"1.35\"). Must be a version AWS EKS currently supports in the target region; EKS manages the patch version."
  type        = string
  default     = "1.35"

  validation {
    condition     = can(regex("^1\\.(2[5-9]|3[0-9])$", var.kubernetes_version))
    error_message = "kubernetes_version must be a supported EKS minor version of the form \"1.NN\" (e.g. \"1.35\")."
  }
}

variable "node_instance_types" {
  description = "EC2 instance types for the managed node group. A single small general-purpose type (t3.medium: 2 vCPU / 4 GiB) is the cost-conscious default — enough for the batch Job plus EKS system pods, no GPUs."
  type        = list(string)
  default     = ["t3.medium"]

  validation {
    condition     = length(var.node_instance_types) > 0
    error_message = "node_instance_types must list at least one EC2 instance type."
  }
}

variable "node_capacity_type" {
  description = "Managed node group purchasing option. ON_DEMAND (default) gives predictable capacity for a short validation run; SPOT is cheaper but can be interrupted."
  type        = string
  default     = "ON_DEMAND"

  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.node_capacity_type)
    error_message = "node_capacity_type must be either \"ON_DEMAND\" or \"SPOT\"."
  }
}

variable "node_desired_size" {
  description = "Desired number of worker nodes. Two small nodes comfortably host the single batch Job plus EKS system pods (CoreDNS, kube-proxy, VPC CNI) with room to schedule."
  type        = number
  default     = 2

  validation {
    condition     = var.node_desired_size >= 1 && var.node_desired_size <= 4
    error_message = "node_desired_size must be between 1 and 4 for this cost-conscious validation cluster."
  }
}

variable "node_min_size" {
  description = "Minimum node count. Equal to the desired size by default: there is no Cluster Autoscaler in this PR, so the group is effectively fixed-size (no autoscaling)."
  type        = number
  default     = 2

  validation {
    condition     = var.node_min_size >= 1 && var.node_min_size <= 4
    error_message = "node_min_size must be between 1 and 4."
  }
}

variable "node_max_size" {
  description = "Maximum node count. Kept equal to the desired size so the group does not scale on its own (no autoscaler is installed); raise only if a future workload needs headroom for rolling updates."
  type        = number
  default     = 2

  validation {
    condition     = var.node_max_size >= 1 && var.node_max_size <= 4
    error_message = "node_max_size must be between 1 and 4."
  }
}

variable "node_disk_size" {
  description = "EBS root volume size (GiB) per worker node. 20 GiB is enough for the container images and ephemeral pipeline data this validation workload uses."
  type        = number
  default     = 20

  validation {
    condition     = var.node_disk_size >= 20 && var.node_disk_size <= 100
    error_message = "node_disk_size must be between 20 and 100 GiB for this validation cluster."
  }
}

variable "cluster_endpoint_public_access" {
  description = "Whether the EKS API server is reachable from the public internet. Enabled so an operator can run kubectl from a workstation to validate the cluster; pair with cluster_endpoint_public_access_cidrs to restrict the source range."
  type        = bool
  default     = true
}

variable "cluster_endpoint_private_access" {
  description = "Whether the EKS API server is reachable privately from within the VPC. Enabled so in-VPC nodes/tools reach the API over private networking rather than traversing the public endpoint."
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks allowed to reach the public EKS API endpoint. Defaults to open (0.0.0.0/0) for first-run validation from any workstation; SET THIS to your operator IP/CIDR for a real environment — it is the primary control-plane exposure knob."
  type        = list(string)
  default     = ["0.0.0.0/0"]

  validation {
    condition     = length(var.cluster_endpoint_public_access_cidrs) > 0
    error_message = "cluster_endpoint_public_access_cidrs must contain at least one CIDR block."
  }
}

variable "cluster_enabled_log_types" {
  description = "EKS control-plane log types shipped to CloudWatch Logs. Defaults to the security-relevant subset (api, audit, authenticator) for auditability; set to [] to eliminate the small CloudWatch Logs cost entirely."
  type        = list(string)
  default     = ["api", "audit", "authenticator"]

  validation {
    condition = alltrue([
      for t in var.cluster_enabled_log_types :
      contains(["api", "audit", "authenticator", "controllerManager", "scheduler"], t)
    ])
    error_message = "cluster_enabled_log_types entries must be from: api, audit, authenticator, controllerManager, scheduler."
  }
}
