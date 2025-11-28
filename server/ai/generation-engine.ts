import { AnalysisResult, DetectedService, DetectedDatabase, DetectedCache, DetectedQueue } from './engine';
import { knowledgeBase, TERRAFORM_PATTERNS, KUBERNETES_PATTERNS, DOCKER_PATTERNS } from './knowledge-base';

export interface GenerationOptions {
  cloudProvider: 'aws' | 'gcp' | 'azure' | 'multi-cloud';
  environment: 'development' | 'staging' | 'production';
  highAvailability: boolean;
  autoScaling: boolean;
  monitoring: boolean;
  security: 'basic' | 'standard' | 'enterprise';
  costOptimization: boolean;
}

export interface GeneratedInfrastructure {
  terraform: TerraformOutput;
  kubernetes: KubernetesOutput;
  docker: DockerOutput;
  cicd: CICDOutput;
  monitoring: MonitoringOutput;
  readme: string;
}

export interface TerraformOutput {
  mainTf: string;
  variablesTf: string;
  outputsTf: string;
  providersTf: string;
  modules: { name: string; content: string }[];
  tfvars: { environment: string; content: string }[];
}

export interface KubernetesOutput {
  deployments: { name: string; content: string }[];
  services: { name: string; content: string }[];
  configMaps: { name: string; content: string }[];
  secrets: { name: string; content: string }[];
  ingress: string;
  hpa: { name: string; content: string }[];
  networkPolicies: { name: string; content: string }[];
  rbac: string;
}

export interface DockerOutput {
  dockerfile: string;
  dockerCompose: string;
  dockerignore: string;
}

export interface CICDOutput {
  githubActions: string;
  gitlabCI: string;
}

export interface MonitoringOutput {
  prometheus: string;
  grafanaDashboards: string[];
  alertRules: string;
}

export class InfrastructureGenerator {
  private options: GenerationOptions;

  constructor(options: Partial<GenerationOptions> = {}) {
    this.options = {
      cloudProvider: options.cloudProvider || 'aws',
      environment: options.environment || 'production',
      highAvailability: options.highAvailability ?? true,
      autoScaling: options.autoScaling ?? true,
      monitoring: options.monitoring ?? true,
      security: options.security || 'standard',
      costOptimization: options.costOptimization ?? false,
    };
  }

  generate(analysis: AnalysisResult): GeneratedInfrastructure {
    return {
      terraform: this.generateTerraform(analysis),
      kubernetes: this.generateKubernetes(analysis),
      docker: this.generateDocker(analysis),
      cicd: this.generateCICD(analysis),
      monitoring: this.generateMonitoring(analysis),
      readme: this.generateReadme(analysis),
    };
  }

  private generateTerraform(analysis: AnalysisResult): TerraformOutput {
    const projectName = 'platform-app';
    const region = 'us-west-2';

    const providersTf = `
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "s3" {
    bucket         = "\${var.project_name}-terraform-state"
    key            = "\${var.environment}/terraform.tfstate"
    region         = "\${var.aws_region}"
    dynamodb_table = "\${var.project_name}-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}
`;

    const variablesTf = `
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "${projectName}"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "${region}"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["${region}a", "${region}b", "${region}c"]
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = ${analysis.services.length > 3 ? 3 : 2}
}

variable "node_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = ${Math.max(20, analysis.services.length * 5)}
}

variable "node_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = ${analysis.services.length > 3 ? 5 : 3}
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "${this.options.environment === 'production' ? 'db.r6g.large' : 'db.t3.medium'}"
}

variable "db_allocated_storage" {
  description = "Initial database storage in GB"
  type        = number
  default     = ${this.options.environment === 'production' ? 100 : 20}
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "${this.options.environment === 'production' ? 'cache.r6g.large' : 'cache.t3.medium'}"
}

variable "enable_monitoring" {
  description = "Enable enhanced monitoring"
  type        = bool
  default     = ${this.options.monitoring}
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {}
}
`;

    const mainTf = `
locals {
  name = "\${var.project_name}-\${var.environment}"
  
  common_tags = merge(var.common_tags, {
    Project     = var.project_name
    Environment = var.environment
  })
}

# VPC
module "vpc" {
  source = "./modules/vpc"

  name               = local.name
  cidr               = var.vpc_cidr
  availability_zones = var.availability_zones
  environment        = var.environment
  
  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
  
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true

  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"

  cluster_name    = local.name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = var.environment != "production"
  cluster_endpoint_private_access = true

  # Node groups
  eks_managed_node_groups = {
    main = {
      instance_types = var.node_instance_types
      capacity_type  = var.environment == "production" ? "ON_DEMAND" : "SPOT"

      min_size     = var.node_min_size
      max_size     = var.node_max_size
      desired_size = var.node_desired_size

      labels = {
        Environment = var.environment
      }
    }
  }

  # Enable IRSA
  enable_irsa = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
  }

  tags = local.common_tags
}

${analysis.databases.map((db, i) => `
# RDS ${db.type} Database
module "rds_${i}" {
  source = "./modules/rds"

  identifier = "\${local.name}-${db.type}"
  
  engine         = "${db.type}"
  engine_version = "${db.type === 'postgresql' ? '15.4' : '8.0'}"
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_allocated_storage * 5
  storage_encrypted     = true

  db_name  = "${projectName.replace(/-/g, '_')}"
  username = "admin"
  port     = ${db.type === 'postgresql' ? 5432 : 3306}

  multi_az               = ${this.options.highAvailability}
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [module.rds_security_group.security_group_id]

  backup_retention_period = ${this.options.environment === 'production' ? 7 : 1}
  skip_final_snapshot     = ${this.options.environment !== 'production'}
  deletion_protection     = ${this.options.environment === 'production'}

  performance_insights_enabled = var.enable_monitoring
  monitoring_interval          = var.enable_monitoring ? 60 : 0

  tags = local.common_tags
}
`).join('\n')}

${analysis.caches.map((cache, i) => `
# ElastiCache ${cache.type}
module "elasticache_${i}" {
  source = "./modules/elasticache"

  cluster_id      = "\${local.name}-${cache.type}"
  engine          = "${cache.type}"
  node_type       = var.redis_node_type
  num_cache_nodes = ${this.options.highAvailability ? 2 : 1}
  
  parameter_group_family = "${cache.type === 'redis' ? 'redis7' : 'memcached1.6'}"
  
  subnet_group_name  = module.vpc.elasticache_subnet_group_name
  security_group_ids = [module.cache_security_group.security_group_id]

  automatic_failover_enabled = ${this.options.highAvailability}
  multi_az_enabled           = ${this.options.highAvailability}

  snapshot_retention_limit = ${this.options.environment === 'production' ? 7 : 0}

  tags = local.common_tags
}
`).join('\n')}

# Security Groups
module "rds_security_group" {
  source = "terraform-aws-modules/security-group/aws"

  name        = "\${local.name}-rds"
  description = "Security group for RDS"
  vpc_id      = module.vpc.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                = 5432
      to_port                  = 5432
      protocol                 = "tcp"
      source_security_group_id = module.eks.node_security_group_id
    }
  ]

  tags = local.common_tags
}

module "cache_security_group" {
  source = "terraform-aws-modules/security-group/aws"

  name        = "\${local.name}-cache"
  description = "Security group for ElastiCache"
  vpc_id      = module.vpc.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                = 6379
      to_port                  = 6379
      protocol                 = "tcp"
      source_security_group_id = module.eks.node_security_group_id
    }
  ]

  tags = local.common_tags
}

# ECR Repository
resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
`;

    const outputsTf = `
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

${analysis.databases.map((db, i) => `
output "rds_${i}_endpoint" {
  description = "RDS endpoint for ${db.type}"
  value       = module.rds_${i}.db_instance_endpoint
  sensitive   = true
}
`).join('\n')}

${analysis.caches.map((cache, i) => `
output "elasticache_${i}_endpoint" {
  description = "ElastiCache endpoint for ${cache.type}"
  value       = module.elasticache_${i}.primary_endpoint_address
}
`).join('\n')}

output "configure_kubectl" {
  description = "Configure kubectl command"
  value       = "aws eks update-kubeconfig --region \${var.aws_region} --name \${module.eks.cluster_name}"
}
`;

    return {
      mainTf,
      variablesTf,
      outputsTf,
      providersTf,
      modules: [
        { name: 'vpc', content: this.generateVPCModule() },
        { name: 'eks', content: this.generateEKSModule() },
        { name: 'rds', content: this.generateRDSModule() },
      ],
      tfvars: [
        { environment: 'development', content: this.generateTfvars('development') },
        { environment: 'staging', content: this.generateTfvars('staging') },
        { environment: 'production', content: this.generateTfvars('production') },
      ],
    };
  }

  private generateVPCModule(): string {
    return `
# VPC Module
# This is a simplified version - in production use terraform-aws-modules/vpc/aws

resource "aws_vpc" "this" {
  cidr_block           = var.cidr
  enable_dns_hostnames = var.enable_dns_hostnames
  enable_dns_support   = var.enable_dns_support

  tags = merge(var.tags, {
    Name = var.name
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.cidr, 4, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name                     = "\${var.name}-public-\${count.index + 1}"
    "kubernetes.io/role/elb" = "1"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.cidr, 4, count.index + length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.tags, {
    Name                              = "\${var.name}-private-\${count.index + 1}"
    "kubernetes.io/role/internal-elb" = "1"
  })
}

# Database Subnets
resource "aws_subnet" "database" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.cidr, 4, count.index + 2 * length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.tags, {
    Name = "\${var.name}-database-\${count.index + 1}"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = merge(var.tags, {
    Name = var.name
  })
}

# NAT Gateway
resource "aws_eip" "nat" {
  count  = var.single_nat_gateway ? 1 : length(var.availability_zones)
  domain = "vpc"

  tags = merge(var.tags, {
    Name = "\${var.name}-nat-\${count.index + 1}"
  })
}

resource "aws_nat_gateway" "this" {
  count = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.availability_zones)) : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(var.tags, {
    Name = "\${var.name}-nat-\${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.this]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = merge(var.tags, {
    Name = "\${var.name}-public"
  })
}

resource "aws_route_table" "private" {
  count  = var.single_nat_gateway ? 1 : length(var.availability_zones)
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this[var.single_nat_gateway ? 0 : count.index].id
  }

  tags = merge(var.tags, {
    Name = "\${var.name}-private-\${count.index + 1}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(var.availability_zones)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(var.availability_zones)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[var.single_nat_gateway ? 0 : count.index].id
}

# DB Subnet Group
resource "aws_db_subnet_group" "database" {
  name       = "\${var.name}-database"
  subnet_ids = aws_subnet.database[*].id

  tags = merge(var.tags, {
    Name = "\${var.name}-database"
  })
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "cache" {
  name       = "\${var.name}-cache"
  subnet_ids = aws_subnet.private[*].id

  tags = var.tags
}

# VPC Flow Logs
resource "aws_flow_log" "this" {
  count = var.enable_flow_log ? 1 : 0

  log_destination_type = "cloud-watch-logs"
  log_destination      = aws_cloudwatch_log_group.flow_log[0].arn
  traffic_type         = "ALL"
  vpc_id               = aws_vpc.this.id
  iam_role_arn         = aws_iam_role.flow_log[0].arn

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "flow_log" {
  count = var.enable_flow_log && var.create_flow_log_cloudwatch_log_group ? 1 : 0

  name              = "/aws/vpc-flow-log/\${var.name}"
  retention_in_days = 30

  tags = var.tags
}

resource "aws_iam_role" "flow_log" {
  count = var.enable_flow_log && var.create_flow_log_cloudwatch_iam_role ? 1 : 0

  name = "\${var.name}-flow-log"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "vpc-flow-logs.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "flow_log" {
  count = var.enable_flow_log && var.create_flow_log_cloudwatch_iam_role ? 1 : 0

  name = "\${var.name}-flow-log"
  role = aws_iam_role.flow_log[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ]
      Effect   = "Allow"
      Resource = "*"
    }]
  })
}

# Outputs
output "vpc_id" {
  value = aws_vpc.this.id
}

output "public_subnets" {
  value = aws_subnet.public[*].id
}

output "private_subnets" {
  value = aws_subnet.private[*].id
}

output "database_subnets" {
  value = aws_subnet.database[*].id
}

output "database_subnet_group_name" {
  value = aws_db_subnet_group.database.name
}

output "elasticache_subnet_group_name" {
  value = aws_elasticache_subnet_group.cache.name
}
`;
  }

  private generateEKSModule(): string {
    return `
# EKS Module - Simplified version
# In production, use terraform-aws-modules/eks/aws

resource "aws_eks_cluster" "this" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = var.cluster_version

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = var.cluster_endpoint_private_access
    endpoint_public_access  = var.cluster_endpoint_public_access
    security_group_ids      = [aws_security_group.cluster.id]
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]

  tags = var.tags

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_cloudwatch_log_group.this
  ]
}

# KMS Key for EKS
resource "aws_kms_key" "eks" {
  description             = "KMS key for EKS secrets encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = var.tags
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/eks/\${var.cluster_name}/cluster"
  retention_in_days = 30

  tags = var.tags
}

# IAM Role for Cluster
resource "aws_iam_role" "cluster" {
  name = "\${var.cluster_name}-cluster"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}

# Security Group for Cluster
resource "aws_security_group" "cluster" {
  name        = "\${var.cluster_name}-cluster"
  description = "Security group for EKS cluster"
  vpc_id      = var.vpc_id

  tags = merge(var.tags, {
    Name = "\${var.cluster_name}-cluster"
  })
}

resource "aws_security_group_rule" "cluster_egress" {
  security_group_id = aws_security_group.cluster.id
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

# Managed Node Group
resource "aws_eks_node_group" "main" {
  for_each = var.eks_managed_node_groups

  cluster_name    = aws_eks_cluster.this.name
  node_group_name = each.key
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids

  capacity_type  = each.value.capacity_type
  instance_types = each.value.instance_types

  scaling_config {
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
    min_size     = each.value.min_size
  }

  update_config {
    max_unavailable_percentage = 25
  }

  labels = each.value.labels

  tags = var.tags

  depends_on = [
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly
  ]
}

# IAM Role for Nodes
resource "aws_iam_role" "node" {
  name = "\${var.cluster_name}-node"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node.name
}

# OIDC Provider for IRSA
data "tls_certificate" "this" {
  url = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "this" {
  count = var.enable_irsa ? 1 : 0

  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.this.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.this.identity[0].oidc[0].issuer

  tags = var.tags
}

# Node Security Group
resource "aws_security_group" "node" {
  name        = "\${var.cluster_name}-node"
  description = "Security group for EKS nodes"
  vpc_id      = var.vpc_id

  tags = merge(var.tags, {
    Name = "\${var.cluster_name}-node"
  })
}

# Outputs
output "cluster_endpoint" {
  value = aws_eks_cluster.this.endpoint
}

output "cluster_name" {
  value = aws_eks_cluster.this.name
}

output "cluster_certificate_authority_data" {
  value = aws_eks_cluster.this.certificate_authority[0].data
}

output "node_security_group_id" {
  value = aws_security_group.node.id
}

output "oidc_provider_arn" {
  value = var.enable_irsa ? aws_iam_openid_connect_provider.this[0].arn : null
}
`;
  }

  private generateRDSModule(): string {
    return `
# RDS Module
resource "random_password" "master_password" {
  length  = 32
  special = false
}

resource "aws_db_instance" "this" {
  identifier = var.identifier

  engine         = var.engine
  engine_version = var.engine_version
  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = var.storage_encrypted

  db_name  = var.db_name
  username = var.username
  password = random_password.master_password.result
  port     = var.port

  multi_az               = var.multi_az
  db_subnet_group_name   = var.db_subnet_group_name
  vpc_security_group_ids = var.vpc_security_group_ids
  publicly_accessible    = false

  backup_retention_period = var.backup_retention_period
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  auto_minor_version_upgrade = true
  deletion_protection        = var.deletion_protection
  skip_final_snapshot        = var.skip_final_snapshot

  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_enabled ? 7 : null
  monitoring_interval                   = var.monitoring_interval
  monitoring_role_arn                   = var.monitoring_interval > 0 ? aws_iam_role.rds_monitoring[0].arn : null

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  tags = var.tags
}

resource "aws_iam_role" "rds_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  name = "\${var.identifier}-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "monitoring.rds.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  role       = aws_iam_role.rds_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Store credentials in Secrets Manager
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "\${var.identifier}-credentials"
  description = "Database credentials for \${var.identifier}"

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.username
    password = random_password.master_password.result
    host     = aws_db_instance.this.address
    port     = aws_db_instance.this.port
    database = var.db_name
  })
}

output "db_instance_endpoint" {
  value = aws_db_instance.this.endpoint
}

output "db_instance_id" {
  value = aws_db_instance.this.id
}

output "secret_arn" {
  value = aws_secretsmanager_secret.db_credentials.arn
}
`;
  }

  private generateTfvars(environment: string): string {
    const config = {
      development: {
        nodeMinSize: 1,
        nodeMaxSize: 5,
        nodeDesiredSize: 2,
        dbInstance: 'db.t3.small',
        dbStorage: 20,
        redisNode: 'cache.t3.micro',
      },
      staging: {
        nodeMinSize: 2,
        nodeMaxSize: 10,
        nodeDesiredSize: 3,
        dbInstance: 'db.t3.medium',
        dbStorage: 50,
        redisNode: 'cache.t3.small',
      },
      production: {
        nodeMinSize: 3,
        nodeMaxSize: 50,
        nodeDesiredSize: 5,
        dbInstance: 'db.r6g.large',
        dbStorage: 100,
        redisNode: 'cache.r6g.large',
      },
    };

    const c = config[environment as keyof typeof config];

    return `
# ${environment.charAt(0).toUpperCase() + environment.slice(1)} Environment Configuration

environment = "${environment}"

# EKS Configuration
node_instance_types = ${environment === 'production' ? '["m5.large", "m5.xlarge"]' : '["t3.medium", "t3.large"]'}
node_min_size       = ${c.nodeMinSize}
node_max_size       = ${c.nodeMaxSize}
node_desired_size   = ${c.nodeDesiredSize}

# Database Configuration
db_instance_class    = "${c.dbInstance}"
db_allocated_storage = ${c.dbStorage}

# Cache Configuration
redis_node_type = "${c.redisNode}"

# Monitoring
enable_monitoring = ${environment === 'production'}

# Tags
common_tags = {
  Environment = "${environment}"
  Team        = "platform"
  CostCenter  = "${environment}-infra"
}
`;
  }

  private generateKubernetes(analysis: AnalysisResult): KubernetesOutput {
    const namespace = 'platform';
    const appName = 'platform-app';

    const deployments = analysis.services.map(service => ({
      name: service.name,
      content: this.generateDeployment(service, namespace),
    }));

    const services = analysis.services.map(service => ({
      name: service.name,
      content: this.generateService(service, namespace),
    }));

    const hpa = analysis.services.filter(s => s.type !== 'worker').map(service => ({
      name: service.name,
      content: this.generateHPA(service, namespace),
    }));

    const networkPolicies = analysis.services.map(service => ({
      name: service.name,
      content: this.generateNetworkPolicy(service, namespace, analysis.services),
    }));

    return {
      deployments,
      services,
      configMaps: [
        { name: 'app-config', content: this.generateConfigMap(namespace) },
      ],
      secrets: [
        { name: 'app-secrets', content: this.generateSecretTemplate(namespace) },
      ],
      ingress: this.generateIngress(analysis.services, namespace),
      hpa,
      networkPolicies,
      rbac: this.generateRBAC(namespace),
    };
  }

  private generateDeployment(service: DetectedService, namespace: string): string {
    return `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${service.name}
  namespace: ${namespace}
  labels:
    app: ${service.name}
    tier: ${service.type}
spec:
  replicas: ${service.type === 'worker' ? 2 : 3}
  selector:
    matchLabels:
      app: ${service.name}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
  template:
    metadata:
      labels:
        app: ${service.name}
        tier: ${service.type}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: ${service.name}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: ${service.name}
          image: \${ECR_REPOSITORY}/${service.name}:\${VERSION}
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: ${service.port || 3000}
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP
          envFrom:
            - configMapRef:
                name: app-config
            - secretRef:
                name: app-secrets
          resources:
            requests:
              memory: "${service.resources.memory}"
              cpu: "${service.resources.cpu}"
            limits:
              memory: "${this.increaseResource(service.resources.memory, 2)}"
              cpu: "${this.increaseResource(service.resources.cpu, 2)}"
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /var/cache
      volumes:
        - name: tmp
          emptyDir: {}
        - name: cache
          emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: ${service.name}
                topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: ${service.name}
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ${service.name}
  namespace: ${namespace}
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: ${service.name}
`;
  }

  private increaseResource(resource: string, factor: number): string {
    const match = resource.match(/^(\d+)(m|Mi|Gi)?$/);
    if (!match) return resource;
    const value = parseInt(match[1]) * factor;
    return `${value}${match[2] || ''}`;
  }

  private generateService(service: DetectedService, namespace: string): string {
    return `
apiVersion: v1
kind: Service
metadata:
  name: ${service.name}
  namespace: ${namespace}
  labels:
    app: ${service.name}
spec:
  type: ClusterIP
  ports:
    - port: ${service.port || 80}
      targetPort: http
      protocol: TCP
      name: http
    - port: 9090
      targetPort: metrics
      protocol: TCP
      name: metrics
  selector:
    app: ${service.name}
`;
  }

  private generateHPA(service: DetectedService, namespace: string): string {
    return `
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${service.name}
  namespace: ${namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${service.name}
  minReplicas: ${this.options.environment === 'production' ? 3 : 2}
  maxReplicas: ${this.options.environment === 'production' ? 50 : 10}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
`;
  }

  private generateNetworkPolicy(service: DetectedService, namespace: string, allServices: DetectedService[]): string {
    const dependencies = service.dependencies || [];
    
    return `
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ${service.name}
  namespace: ${namespace}
spec:
  podSelector:
    matchLabels:
      app: ${service.name}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: ${service.port || 3000}
    # Allow from other services that depend on this
    ${allServices.filter(s => s.dependencies?.includes(service.name)).map(s => `
    - from:
        - podSelector:
            matchLabels:
              app: ${s.name}
      ports:
        - protocol: TCP
          port: ${service.port || 3000}
    `).join('')}
    # Allow Prometheus scraping
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    ${dependencies.map(dep => `
    # Allow to ${dep}
    - to:
        - podSelector:
            matchLabels:
              app: ${dep}
    `).join('')}
    # Allow external HTTPS
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 10.0.0.0/8
              - 172.16.0.0/12
              - 192.168.0.0/16
      ports:
        - protocol: TCP
          port: 443
`;
  }

  private generateIngress(services: DetectedService[], namespace: string): string {
    const gatewayService = services.find(s => s.type === 'gateway') || services[0];
    
    return `
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: main-ingress
  namespace: ${namespace}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
    - hosts:
        - \${DOMAIN}
        - api.\${DOMAIN}
      secretName: tls-secret
  rules:
    - host: \${DOMAIN}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ${gatewayService?.name || 'api-gateway'}
                port:
                  number: 80
    - host: api.\${DOMAIN}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ${gatewayService?.name || 'api-gateway'}
                port:
                  number: 80
`;
  }

  private generateConfigMap(namespace: string): string {
    return `
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: ${namespace}
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  METRICS_ENABLED: "true"
  METRICS_PORT: "9090"
`;
  }

  private generateSecretTemplate(namespace: string): string {
    return `
# This is a template - actual secrets should be managed via:
# - AWS Secrets Manager with external-secrets operator
# - HashiCorp Vault
# - Sealed Secrets
#
# DO NOT commit actual secrets to version control

apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: ${namespace}
type: Opaque
stringData:
  DATABASE_URL: "\${DATABASE_URL}"
  REDIS_URL: "\${REDIS_URL}"
  JWT_SECRET: "\${JWT_SECRET}"
`;
  }

  private generateRBAC(namespace: string): string {
    return `
apiVersion: v1
kind: ServiceAccount
metadata:
  name: platform-app
  namespace: ${namespace}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: platform-app
  namespace: ${namespace}
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: platform-app
  namespace: ${namespace}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: platform-app
subjects:
  - kind: ServiceAccount
    name: platform-app
    namespace: ${namespace}
`;
  }

  private generateDocker(analysis: AnalysisResult): DockerOutput {
    const language = analysis.language;
    const dockerPattern = DOCKER_PATTERNS.find(p => p.language === language);
    
    return {
      dockerfile: dockerPattern?.dockerfile || this.generateGenericDockerfile(analysis),
      dockerCompose: this.generateDockerCompose(analysis),
      dockerignore: this.generateDockerignore(language),
    };
  }

  private generateGenericDockerfile(analysis: AnalysisResult): string {
    return `
# syntax=docker/dockerfile:1.4

FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine AS production

RUN addgroup -g 1001 -S app && adduser -S app -u 1001

WORKDIR /app

COPY --from=builder --chown=app:app /app/dist ./dist
COPY --from=builder --chown=app:app /app/node_modules ./node_modules
COPY --from=builder --chown=app:app /app/package.json ./

USER app

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

EXPOSE 3000

ENV NODE_ENV=production

CMD ["node", "dist/server.js"]
`;
  }

  private generateDockerCompose(analysis: AnalysisResult): string {
    return `
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "node", "-e", "require('http').get('http://localhost:3000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=app
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
`;
  }

  private generateDockerignore(language: string): string {
    const common = `
# Dependencies
node_modules/
vendor/
.venv/

# Build outputs
dist/
build/
*.pyc
__pycache__/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Git
.git/
.gitignore

# Docker
Dockerfile*
docker-compose*
.docker/

# Environment
.env*
*.local

# Tests
coverage/
.nyc_output/
*.test.*
*.spec.*

# Documentation
docs/
*.md
LICENSE

# Misc
.DS_Store
Thumbs.db
*.log
`;

    return common;
  }

  private generateCICD(analysis: AnalysisResult): CICDOutput {
    return {
      githubActions: this.generateGitHubActions(analysis),
      gitlabCI: this.generateGitLabCI(analysis),
    };
  }

  private generateGitHubActions(analysis: AnalysisResult): string {
    return `
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-west-2
  ECR_REPOSITORY: platform-app
  EKS_CLUSTER: platform-app-production

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run linting
        run: npm run lint
      
      - name: Run tests
        run: npm test -- --coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          ignore-unfixed: true
          severity: 'CRITICAL,HIGH'

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    outputs:
      image-tag: \${{ steps.build-image.outputs.image-tag }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: \${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: \${{ env.AWS_REGION }}
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build, tag, and push image
        id: build-image
        env:
          ECR_REGISTRY: \${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: \${{ github.sha }}
        run: |
          docker build -t \$ECR_REGISTRY/\$ECR_REPOSITORY:\$IMAGE_TAG .
          docker push \$ECR_REGISTRY/\$ECR_REPOSITORY:\$IMAGE_TAG
          echo "image-tag=\$IMAGE_TAG" >> \$GITHUB_OUTPUT
      
      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: \${{ steps.login-ecr.outputs.registry }}/\${{ env.ECR_REPOSITORY }}:\${{ github.sha }}
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: \${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: \${{ env.AWS_REGION }}
      
      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name \${{ env.EKS_CLUSTER }}-staging
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/platform-app \\
            platform-app=\${{ secrets.ECR_REGISTRY }}/\${{ env.ECR_REPOSITORY }}:\${{ needs.build.outputs.image-tag }} \\
            -n staging
          kubectl rollout status deployment/platform-app -n staging

  deploy-production:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: \${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: \${{ env.AWS_REGION }}
      
      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name \${{ env.EKS_CLUSTER }}
      
      - name: Deploy canary (10%)
        run: |
          kubectl set image deployment/platform-app-canary \\
            platform-app=\${{ secrets.ECR_REGISTRY }}/\${{ env.ECR_REPOSITORY }}:\${{ needs.build.outputs.image-tag }} \\
            -n production
          kubectl rollout status deployment/platform-app-canary -n production
      
      - name: Wait for canary validation
        run: sleep 300
      
      - name: Deploy production
        run: |
          kubectl set image deployment/platform-app \\
            platform-app=\${{ secrets.ECR_REGISTRY }}/\${{ env.ECR_REPOSITORY }}:\${{ needs.build.outputs.image-tag }} \\
            -n production
          kubectl rollout status deployment/platform-app -n production
`;
  }

  private generateGitLabCI(analysis: AnalysisResult): string {
    return `
stages:
  - test
  - build
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  AWS_REGION: us-west-2
  ECR_REPOSITORY: platform-app

.aws-configure: &aws-configure
  before_script:
    - aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
    - aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
    - aws configure set region $AWS_REGION

test:
  stage: test
  image: node:20-alpine
  script:
    - npm ci
    - npm run lint
    - npm test -- --coverage
  coverage: '/All files[^|]*\\|[^|]*\\s+([\\d\\.]+)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  <<: *aws-configure
  script:
    - aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
    - docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$CI_COMMIT_SHA .
    - docker push $ECR_REGISTRY/$ECR_REPOSITORY:$CI_COMMIT_SHA
  only:
    - main

security-scan:
  stage: security
  image:
    name: aquasec/trivy:latest
    entrypoint: [""]
  script:
    - trivy image --exit-code 1 --severity CRITICAL,HIGH $ECR_REGISTRY/$ECR_REPOSITORY:$CI_COMMIT_SHA
  only:
    - main

deploy-staging:
  stage: deploy
  image: alpine/k8s:1.28.0
  <<: *aws-configure
  script:
    - aws eks update-kubeconfig --name platform-app-staging
    - kubectl set image deployment/platform-app platform-app=$ECR_REGISTRY/$ECR_REPOSITORY:$CI_COMMIT_SHA -n staging
    - kubectl rollout status deployment/platform-app -n staging
  environment:
    name: staging
  only:
    - main

deploy-production:
  stage: deploy
  image: alpine/k8s:1.28.0
  <<: *aws-configure
  script:
    - aws eks update-kubeconfig --name platform-app-production
    - kubectl set image deployment/platform-app platform-app=$ECR_REGISTRY/$ECR_REPOSITORY:$CI_COMMIT_SHA -n production
    - kubectl rollout status deployment/platform-app -n production
  environment:
    name: production
  when: manual
  only:
    - main
`;
  }

  private generateMonitoring(analysis: AnalysisResult): MonitoringOutput {
    return {
      prometheus: this.generatePrometheusConfig(analysis),
      grafanaDashboards: [this.generateGrafanaDashboard(analysis)],
      alertRules: this.generateAlertRules(analysis),
    };
  }

  private generatePrometheusConfig(analysis: AnalysisResult): string {
    return `
# Prometheus Operator ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: platform-app
  namespace: monitoring
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: platform-app
  namespaceSelector:
    matchNames:
      - production
      - staging
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
`;
  }

  private generateGrafanaDashboard(analysis: AnalysisResult): string {
    return JSON.stringify({
      title: 'Platform Application Dashboard',
      uid: 'platform-app',
      panels: [
        {
          title: 'Request Rate',
          type: 'graph',
          targets: [
            { expr: 'rate(http_requests_total[5m])' }
          ]
        },
        {
          title: 'Response Time (P95)',
          type: 'graph',
          targets: [
            { expr: 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))' }
          ]
        },
        {
          title: 'Error Rate',
          type: 'graph',
          targets: [
            { expr: 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])' }
          ]
        },
        {
          title: 'CPU Usage',
          type: 'graph',
          targets: [
            { expr: 'rate(container_cpu_usage_seconds_total{container="platform-app"}[5m])' }
          ]
        },
        {
          title: 'Memory Usage',
          type: 'graph',
          targets: [
            { expr: 'container_memory_usage_bytes{container="platform-app"}' }
          ]
        }
      ]
    }, null, 2);
  }

  private generateAlertRules(analysis: AnalysisResult): string {
    return `
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: platform-app-alerts
  namespace: monitoring
spec:
  groups:
    - name: platform-app
      rules:
        - alert: HighErrorRate
          expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: High error rate detected
            description: Error rate is above 5% for 5 minutes

        - alert: HighLatency
          expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: High latency detected
            description: P95 latency is above 1 second

        - alert: HighCPU
          expr: rate(container_cpu_usage_seconds_total{container="platform-app"}[5m]) > 0.8
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: High CPU usage
            description: CPU usage is above 80% for 10 minutes

        - alert: HighMemory
          expr: container_memory_usage_bytes{container="platform-app"} / container_spec_memory_limit_bytes > 0.85
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: High memory usage
            description: Memory usage is above 85%

        - alert: PodNotReady
          expr: kube_pod_status_ready{condition="false",pod=~"platform-app.*"} == 1
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: Pod not ready
            description: Pod {{ $labels.pod }} is not ready
`;
  }

  private generateReadme(analysis: AnalysisResult): string {
    return `
# Infrastructure Documentation

## Overview

This infrastructure configuration provides a production-ready deployment for a ${analysis.architecture} architecture application.

## Components

### Cloud Infrastructure (Terraform)
- **VPC**: Multi-AZ networking with public/private subnets
- **EKS**: Managed Kubernetes cluster
- **RDS**: ${analysis.databases.map(d => d.type).join(', ')} database(s)
- **ElastiCache**: ${analysis.caches.map(c => c.type).join(', ') || 'Redis'} caching layer

### Kubernetes Resources
- Deployments with health checks and resource limits
- Horizontal Pod Autoscalers
- Network Policies for zero-trust networking
- RBAC configuration

### CI/CD
- GitHub Actions workflow
- GitLab CI configuration
- Multi-environment deployment (staging → production)

## Quick Start

### Prerequisites
- AWS CLI configured
- Terraform >= 1.5
- kubectl
- Docker

### Deploy Infrastructure

\`\`\`bash
# Initialize Terraform
cd infrastructure
terraform init

# Plan deployment
terraform plan -var-file=environments/production.tfvars

# Apply
terraform apply -var-file=environments/production.tfvars
\`\`\`

### Configure kubectl

\`\`\`bash
aws eks update-kubeconfig --name platform-app-production --region us-west-2
\`\`\`

### Deploy Application

\`\`\`bash
# Apply Kubernetes manifests
kubectl apply -k kubernetes/overlays/production
\`\`\`

## Security Considerations

- All secrets stored in AWS Secrets Manager
- Network policies enforce zero-trust
- Container images scanned for vulnerabilities
- TLS encryption for all traffic
- RBAC configured with least-privilege

## Monitoring

- Prometheus for metrics collection
- Grafana dashboards for visualization
- AlertManager for alerting
- CloudWatch for AWS-level monitoring

## Cost Optimization

- Spot instances for non-critical workloads
- Right-sizing recommendations
- Auto-scaling to match demand
- Reserved instances for baseline capacity
`;
  }
}

export const generator = new InfrastructureGenerator();
