# AWS network foundation for EKS (Sprint 6, PR 2).
#
# This is the minimum, EKS-ready network: a single VPC spread across two (or
# three) Availability Zones, with a public subnet per AZ (for the internet
# gateway, NAT, and any future public load balancers) and a private subnet per
# AZ (where the EKS worker nodes will run). Nodes reach the internet outbound
# through NAT — the workload is a batch Job with no inbound surface, so there is
# no public ingress requirement, only egress for image/dataset/package pulls.
#
# Nothing here provisions EKS itself; it provisions only the network the EKS PR
# depends on. Design rationale (public/private split, AZ strategy, single-NAT
# cost trade-off) lives in ADR-015 and terraform/README.md, not in these
# comments. Common Project/Environment/Owner tags are applied to every resource
# automatically via the provider's default_tags (providers.tf); only
# resource-specific tags (Name, Tier, EKS role tags) are set below.

# Availability Zones are discovered at plan time rather than hard-coded, so the
# same configuration is correct in any region. Local/Wavelength zones (which
# require opt-in and do not support EKS node groups) are filtered out.
data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

locals {
  # The first `az_count` standard AZs in the active region.
  azs = slice(data.aws_availability_zones.available.names, 0, var.az_count)

  # Per-AZ subnet CIDRs derived from the VPC CIDR with a fixed 8-bit split
  # (e.g. /16 -> /24). Public subnets take the low indices; private subnets are
  # offset by 8 so the two ranges never overlap and stay readable:
  #   public : 10.0.0.0/24, 10.0.1.0/24, ...
  #   private: 10.0.8.0/24, 10.0.9.0/24, ...
  public_subnet_cidrs  = [for i in range(var.az_count) : cidrsubnet(var.vpc_cidr, 8, i)]
  private_subnet_cidrs = [for i in range(var.az_count) : cidrsubnet(var.vpc_cidr, 8, i + 8)]

  # 0 when NAT is disabled, 1 when a single shared NAT gateway is used, or one
  # per AZ for AZ-fault-tolerant egress.
  nat_gateway_count = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : var.az_count) : 0
}

# --- VPC ----------------------------------------------------------------------
# DNS support and hostnames are both enabled: EKS requires them for private
# cluster endpoint resolution and for nodes/pods to resolve service names.
resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${local.name_prefix}-vpc"
  }
}

# --- Internet Gateway ---------------------------------------------------------
# The single egress/ingress point for the public subnets.
resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${local.name_prefix}-igw"
  }
}

# --- Public subnets -----------------------------------------------------------
# One per AZ. Host the NAT gateway(s) and are tagged for public load balancers
# so the AWS Load Balancer Controller / in-tree provisioner can auto-discover
# them (`kubernetes.io/role/elb`). map_public_ip_on_launch is on because NAT and
# public LBs live here; the EKS nodes do NOT.
resource "aws_subnet" "public" {
  count = var.az_count

  vpc_id                  = aws_vpc.this.id
  cidr_block              = local.public_subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                     = "${local.name_prefix}-public-${local.azs[count.index]}"
    Tier                     = "public"
    "kubernetes.io/role/elb" = "1"
  }
}

# --- Private subnets ----------------------------------------------------------
# One per AZ. The EKS worker nodes run here with no public IPs; outbound traffic
# is routed through NAT. Tagged for internal load balancers
# (`kubernetes.io/role/internal-elb`).
resource "aws_subnet" "private" {
  count = var.az_count

  vpc_id            = aws_vpc.this.id
  cidr_block        = local.private_subnet_cidrs[count.index]
  availability_zone = local.azs[count.index]

  tags = {
    Name                              = "${local.name_prefix}-private-${local.azs[count.index]}"
    Tier                              = "private"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# --- NAT gateways -------------------------------------------------------------
# Give private-subnet nodes outbound internet without inbound exposure. One
# Elastic IP per NAT gateway. Single (shared) NAT by default keeps this
# short-lived portfolio environment cheap; flip single_nat_gateway to false for
# per-AZ fault tolerance. NAT gateways and their EIPs are the main hourly cost
# in this PR (see terraform/README.md § Cost considerations).
resource "aws_eip" "nat" {
  count = local.nat_gateway_count

  domain     = "vpc"
  depends_on = [aws_internet_gateway.this]

  tags = {
    Name = "${local.name_prefix}-nat-eip-${count.index}"
  }
}

resource "aws_nat_gateway" "this" {
  count = local.nat_gateway_count

  allocation_id = aws_eip.nat[count.index].id
  # A single NAT lives in the first public subnet; per-AZ NAT lives in the
  # matching AZ's public subnet.
  subnet_id  = aws_subnet.public[count.index].id
  depends_on = [aws_internet_gateway.this]

  tags = {
    Name = "${local.name_prefix}-nat-${count.index}"
  }
}

# --- Public routing -----------------------------------------------------------
# A single public route table shared by all public subnets: default route to the
# internet gateway.
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${local.name_prefix}-public-rt"
  }
}

resource "aws_route" "public_internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.this.id
}

resource "aws_route_table_association" "public" {
  count = var.az_count

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# --- Private routing ----------------------------------------------------------
# One private route table per AZ so each AZ can point at its own NAT gateway when
# single_nat_gateway is false. The default route is added only when NAT is
# enabled; with a single NAT every AZ routes through nat[0], otherwise through
# the AZ-local NAT.
resource "aws_route_table" "private" {
  count = var.az_count

  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${local.name_prefix}-private-rt-${local.azs[count.index]}"
  }
}

resource "aws_route" "private_nat" {
  count = var.enable_nat_gateway ? var.az_count : 0

  route_table_id         = aws_route_table.private[count.index].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = var.single_nat_gateway ? aws_nat_gateway.this[0].id : aws_nat_gateway.this[count.index].id
}

resource "aws_route_table_association" "private" {
  count = var.az_count

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
