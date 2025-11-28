export interface TerraformPattern {
  name: string;
  description: string;
  category: 'compute' | 'network' | 'database' | 'security' | 'storage' | 'monitoring';
  provider: 'aws' | 'gcp' | 'azure' | 'multi-cloud';
  complexity: 'basic' | 'intermediate' | 'advanced';
  template: string;
  variables: VariableDefinition[];
  outputs: OutputDefinition[];
  bestPractices: string[];
  securityConsiderations: string[];
  costFactors: string[];
}

export interface VariableDefinition {
  name: string;
  type: string;
  description: string;
  default?: string;
  validation?: string;
}

export interface OutputDefinition {
  name: string;
  description: string;
  sensitive: boolean;
}

export interface KubernetesPattern {
  name: string;
  description: string;
  category: 'workload' | 'service' | 'config' | 'storage' | 'security' | 'scaling';
  manifest: string;
  requirements: string[];
  bestPractices: string[];
}

export interface DockerPattern {
  name: string;
  language: string;
  framework?: string;
  dockerfile: string;
  optimizations: string[];
  securityMeasures: string[];
}

export interface ArchitecturePattern {
  name: string;
  description: string;
  useCases: string[];
  components: string[];
  tradeoffs: { advantages: string[]; disadvantages: string[] };
  scalingStrategy: string;
  dataFlow: string;
}

export const TERRAFORM_PATTERNS: TerraformPattern[] = [
  {
    name: 'Production VPC',
    description: 'Multi-AZ VPC with public and private subnets, NAT gateways, and flow logs',
    category: 'network',
    provider: 'aws',
    complexity: 'intermediate',
    template: `
# Production-Ready VPC Configuration
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-vpc"
  })
}

# Public Subnets (for Load Balancers, NAT Gateways)
resource "aws_subnet" "public" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-public-\${count.index + 1}"
    Tier = "Public"
  })
}

# Private Subnets (for Application Workloads)
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-private-\${count.index + 1}"
    Tier = "Private"
  })
}

# Database Subnets (isolated, no internet access)
resource "aws_subnet" "database" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index + 2 * length(var.availability_zones))
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-database-\${count.index + 1}"
    Tier = "Database"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-igw"
  })
}

# NAT Gateways (one per AZ for high availability)
resource "aws_eip" "nat" {
  count  = length(var.availability_zones)
  domain = "vpc"

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-nat-eip-\${count.index + 1}"
  })
}

resource "aws_nat_gateway" "main" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-nat-\${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-private-rt-\${count.index + 1}"
  })
}

# VPC Flow Logs
resource "aws_flow_log" "main" {
  log_destination_type = "cloud-watch-logs"
  log_destination      = aws_cloudwatch_log_group.flow_logs.arn
  traffic_type         = "ALL"
  vpc_id               = aws_vpc.main.id
  iam_role_arn         = aws_iam_role.flow_logs.arn

  tags = merge(var.common_tags, {
    Name = "\${var.environment}-flow-logs"
  })
}
`,
    variables: [
      { name: 'vpc_cidr', type: 'string', description: 'CIDR block for VPC', default: '10.0.0.0/16' },
      { name: 'availability_zones', type: 'list(string)', description: 'List of AZs to use' },
      { name: 'environment', type: 'string', description: 'Environment name (dev/staging/prod)' },
      { name: 'common_tags', type: 'map(string)', description: 'Common tags for all resources' },
    ],
    outputs: [
      { name: 'vpc_id', description: 'VPC ID', sensitive: false },
      { name: 'public_subnet_ids', description: 'List of public subnet IDs', sensitive: false },
      { name: 'private_subnet_ids', description: 'List of private subnet IDs', sensitive: false },
      { name: 'database_subnet_ids', description: 'List of database subnet IDs', sensitive: false },
    ],
    bestPractices: [
      'Use /16 CIDR for flexibility in subnet allocation',
      'Deploy NAT Gateways in each AZ for high availability',
      'Enable VPC Flow Logs for security monitoring',
      'Use consistent tagging strategy for cost allocation',
      'Isolate database subnets with no internet access',
    ],
    securityConsiderations: [
      'Restrict default security group rules',
      'Enable VPC Flow Logs for audit trails',
      'Use VPC endpoints for AWS services',
      'Implement network ACLs as additional layer',
    ],
    costFactors: [
      'NAT Gateway: ~$32/month per gateway + data processing',
      'VPC Flow Logs: CloudWatch Logs pricing',
      'Consider NAT instances for dev environments',
    ],
  },
  {
    name: 'EKS Cluster',
    description: 'Production EKS cluster with managed node groups and IRSA',
    category: 'compute',
    provider: 'aws',
    complexity: 'advanced',
    template: `
# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "\${var.cluster_name}-\${var.environment}"
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = var.private_subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = var.enable_public_access
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

  tags = var.common_tags

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
    aws_cloudwatch_log_group.eks
  ]
}

# Managed Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "\${var.cluster_name}-\${var.environment}-nodes"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.private_subnet_ids

  capacity_type  = var.use_spot_instances ? "SPOT" : "ON_DEMAND"
  instance_types = var.instance_types

  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }

  update_config {
    max_unavailable_percentage = 25
  }

  labels = {
    Environment = var.environment
    NodeGroup   = "main"
  }

  tags = var.common_tags

  depends_on = [
    aws_iam_role_policy_attachment.node_policy,
    aws_iam_role_policy_attachment.cni_policy,
    aws_iam_role_policy_attachment.ecr_policy
  ]
}

# OIDC Provider for IRSA
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer

  tags = var.common_tags
}

# Cluster Autoscaler IAM Role
resource "aws_iam_role" "cluster_autoscaler" {
  name = "\${var.cluster_name}-\${var.environment}-cluster-autoscaler"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRoleWithWebIdentity"
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Condition = {
        StringEquals = {
          "\${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub" = "system:serviceaccount:kube-system:cluster-autoscaler"
        }
      }
    }]
  })

  tags = var.common_tags
}

# EKS Add-ons
resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "vpc-cni"
  
  resolve_conflicts_on_update = "OVERWRITE"
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "coredns"
  
  resolve_conflicts_on_update = "OVERWRITE"

  depends_on = [aws_eks_node_group.main]
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "kube-proxy"
  
  resolve_conflicts_on_update = "OVERWRITE"
}
`,
    variables: [
      { name: 'cluster_name', type: 'string', description: 'Name of the EKS cluster' },
      { name: 'kubernetes_version', type: 'string', description: 'Kubernetes version', default: '1.28' },
      { name: 'private_subnet_ids', type: 'list(string)', description: 'Private subnet IDs for nodes' },
      { name: 'instance_types', type: 'list(string)', description: 'Instance types for nodes', default: '["m5.large", "m5.xlarge"]' },
      { name: 'node_min_size', type: 'number', description: 'Minimum number of nodes', default: '2' },
      { name: 'node_max_size', type: 'number', description: 'Maximum number of nodes', default: '20' },
      { name: 'node_desired_size', type: 'number', description: 'Desired number of nodes', default: '3' },
      { name: 'use_spot_instances', type: 'bool', description: 'Use Spot instances', default: 'false' },
    ],
    outputs: [
      { name: 'cluster_endpoint', description: 'EKS cluster endpoint', sensitive: false },
      { name: 'cluster_security_group_id', description: 'Cluster security group ID', sensitive: false },
      { name: 'cluster_certificate_authority_data', description: 'Cluster CA certificate', sensitive: true },
      { name: 'oidc_provider_arn', description: 'OIDC provider ARN for IRSA', sensitive: false },
    ],
    bestPractices: [
      'Enable secrets encryption with KMS',
      'Use IRSA for pod-level IAM permissions',
      'Enable control plane logging',
      'Use managed node groups for easier operations',
      'Implement cluster autoscaler for dynamic scaling',
      'Use Spot instances for non-critical workloads',
    ],
    securityConsiderations: [
      'Restrict public endpoint access',
      'Enable audit logging',
      'Use private subnets for worker nodes',
      'Implement pod security standards',
      'Enable network policies',
    ],
    costFactors: [
      'EKS Control Plane: $0.10/hour (~$73/month)',
      'Worker nodes: EC2 pricing based on instance types',
      'Consider Savings Plans or Reserved Instances',
      'Spot instances can save 60-90%',
    ],
  },
  {
    name: 'RDS PostgreSQL',
    description: 'Production RDS PostgreSQL with Multi-AZ and automated backups',
    category: 'database',
    provider: 'aws',
    complexity: 'intermediate',
    template: `
# RDS PostgreSQL Instance
resource "aws_db_instance" "main" {
  identifier = "\${var.name}-\${var.environment}"

  engine         = "postgres"
  engine_version = var.engine_version
  instance_class = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds.arn

  db_name  = var.database_name
  username = var.master_username
  password = random_password.master.result

  multi_az               = var.multi_az
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  backup_retention_period = var.backup_retention_period
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  auto_minor_version_upgrade = true
  deletion_protection        = var.environment == "prod"
  skip_final_snapshot        = var.environment != "prod"
  final_snapshot_identifier  = var.environment == "prod" ? "\${var.name}-\${var.environment}-final" : null

  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  monitoring_interval                   = 60
  monitoring_role_arn                   = aws_iam_role.rds_monitoring.arn

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  parameter_group_name = aws_db_parameter_group.main.name

  tags = var.common_tags
}

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name        = "\${var.name}-\${var.environment}"
  description = "Database subnet group for \${var.name}"
  subnet_ids  = var.database_subnet_ids

  tags = var.common_tags
}

# Parameter Group with optimized settings
resource "aws_db_parameter_group" "main" {
  name   = "\${var.name}-\${var.environment}-pg15"
  family = "postgres15"

  parameter {
    name  = "log_statement"
    value = "ddl"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "pg_stat_statements.track"
    value = "all"
  }

  tags = var.common_tags
}

# Read Replica (optional)
resource "aws_db_instance" "replica" {
  count = var.create_read_replica ? 1 : 0

  identifier = "\${var.name}-\${var.environment}-replica"

  replicate_source_db = aws_db_instance.main.identifier
  instance_class      = var.replica_instance_class
  
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn

  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  monitoring_interval                   = 60
  monitoring_role_arn                   = aws_iam_role.rds_monitoring.arn

  tags = merge(var.common_tags, {
    Name = "\${var.name}-\${var.environment}-replica"
  })
}

# Store password in Secrets Manager
resource "aws_secretsmanager_secret" "db_password" {
  name        = "\${var.name}-\${var.environment}-db-password"
  description = "RDS master password for \${var.name}"

  tags = var.common_tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = var.master_username
    password = random_password.master.result
    host     = aws_db_instance.main.address
    port     = aws_db_instance.main.port
    database = var.database_name
  })
}

resource "random_password" "master" {
  length  = 32
  special = false
}
`,
    variables: [
      { name: 'name', type: 'string', description: 'Database identifier name' },
      { name: 'engine_version', type: 'string', description: 'PostgreSQL version', default: '15.4' },
      { name: 'instance_class', type: 'string', description: 'RDS instance type', default: 'db.t3.medium' },
      { name: 'allocated_storage', type: 'number', description: 'Initial storage in GB', default: '20' },
      { name: 'max_allocated_storage', type: 'number', description: 'Max autoscaling storage', default: '100' },
      { name: 'multi_az', type: 'bool', description: 'Enable Multi-AZ', default: 'true' },
      { name: 'backup_retention_period', type: 'number', description: 'Backup retention days', default: '7' },
      { name: 'create_read_replica', type: 'bool', description: 'Create read replica', default: 'false' },
    ],
    outputs: [
      { name: 'endpoint', description: 'RDS endpoint', sensitive: false },
      { name: 'port', description: 'RDS port', sensitive: false },
      { name: 'secret_arn', description: 'Secrets Manager ARN for credentials', sensitive: false },
    ],
    bestPractices: [
      'Enable Multi-AZ for production',
      'Use Secrets Manager for credentials',
      'Enable Performance Insights',
      'Configure automated backups',
      'Use parameter groups for tuning',
      'Enable encryption at rest',
    ],
    securityConsiderations: [
      'Never expose publicly',
      'Use security groups to restrict access',
      'Enable SSL for connections',
      'Rotate passwords regularly',
      'Enable audit logging',
    ],
    costFactors: [
      'Instance pricing varies by size',
      'Multi-AZ doubles instance cost',
      'Storage: gp3 is most cost-effective',
      'Consider Reserved Instances for production',
    ],
  },
];

export const KUBERNETES_PATTERNS: KubernetesPattern[] = [
  {
    name: 'Production Deployment',
    description: 'Deployment with resource limits, probes, and pod disruption budget',
    category: 'workload',
    manifest: `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: \${APP_NAME}
  namespace: \${NAMESPACE}
  labels:
    app: \${APP_NAME}
    version: \${VERSION}
spec:
  replicas: \${REPLICAS}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
  selector:
    matchLabels:
      app: \${APP_NAME}
  template:
    metadata:
      labels:
        app: \${APP_NAME}
        version: \${VERSION}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "\${METRICS_PORT}"
    spec:
      serviceAccountName: \${APP_NAME}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: \${APP_NAME}
          image: \${IMAGE}:\${VERSION}
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: \${PORT}
              protocol: TCP
            - name: metrics
              containerPort: \${METRICS_PORT}
              protocol: TCP
          env:
            - name: NODE_ENV
              value: "production"
            - name: PORT
              value: "\${PORT}"
          envFrom:
            - configMapRef:
                name: \${APP_NAME}-config
            - secretRef:
                name: \${APP_NAME}-secrets
          resources:
            requests:
              memory: "\${MEMORY_REQUEST}"
              cpu: "\${CPU_REQUEST}"
            limits:
              memory: "\${MEMORY_LIMIT}"
              cpu: "\${CPU_LIMIT}"
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
      volumes:
        - name: tmp
          emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: \${APP_NAME}
                topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: \${APP_NAME}
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: \${APP_NAME}
  namespace: \${NAMESPACE}
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: \${APP_NAME}
`,
    requirements: [
      'Kubernetes 1.21+',
      'Container registry access',
      'Namespace created',
    ],
    bestPractices: [
      'Always set resource requests and limits',
      'Use liveness and readiness probes',
      'Run as non-root user',
      'Use read-only root filesystem',
      'Implement pod anti-affinity for HA',
      'Add topology spread constraints',
      'Configure PodDisruptionBudget',
    ],
  },
  {
    name: 'Horizontal Pod Autoscaler',
    description: 'HPA with CPU, memory, and custom metrics scaling',
    category: 'scaling',
    manifest: `
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: \${APP_NAME}
  namespace: \${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: \${APP_NAME}
  minReplicas: \${MIN_REPLICAS}
  maxReplicas: \${MAX_REPLICAS}
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
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
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
`,
    requirements: [
      'Metrics Server installed',
      'Prometheus Adapter for custom metrics',
    ],
    bestPractices: [
      'Set appropriate stabilization windows',
      'Use multiple metrics for better scaling',
      'Configure scale-down policies to prevent thrashing',
      'Test scaling behavior under load',
    ],
  },
  {
    name: 'Network Policy',
    description: 'Zero-trust network policy for microservices',
    category: 'security',
    manifest: `
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: \${APP_NAME}-network-policy
  namespace: \${NAMESPACE}
spec:
  podSelector:
    matchLabels:
      app: \${APP_NAME}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
          podSelector:
            matchLabels:
              app.kubernetes.io/name: ingress-nginx
      ports:
        - protocol: TCP
          port: \${PORT}
    # Allow traffic from other services in same namespace
    - from:
        - podSelector:
            matchLabels:
              app: api-gateway
      ports:
        - protocol: TCP
          port: \${PORT}
    # Allow Prometheus scraping
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
          podSelector:
            matchLabels:
              app: prometheus
      ports:
        - protocol: TCP
          port: \${METRICS_PORT}
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
    # Allow database access
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432
    # Allow Redis access
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    # Allow external HTTPS (for external APIs)
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
`,
    requirements: [
      'CNI plugin supporting NetworkPolicy (Calico, Cilium)',
    ],
    bestPractices: [
      'Default deny all traffic',
      'Explicitly allow only required connections',
      'Use namespace selectors for cross-namespace access',
      'Allow egress to DNS',
      'Document all allowed flows',
    ],
  },
];

export const DOCKER_PATTERNS: DockerPattern[] = [
  {
    name: 'Node.js Production',
    language: 'nodejs',
    framework: 'express',
    dockerfile: `
# syntax=docker/dockerfile:1.4

# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source and build
COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine AS production

# Security: run as non-root
RUN addgroup -g 1001 -S nodejs && \\
    adduser -S nodejs -u 1001

WORKDIR /app

# Copy only production artifacts
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./

# Security hardening
USER nodejs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

EXPOSE 3000

ENV NODE_ENV=production

CMD ["node", "dist/server.js"]
`,
    optimizations: [
      'Multi-stage build reduces image size by 60-80%',
      'Alpine base for minimal footprint',
      'Dependency caching for faster builds',
      'Production-only dependencies',
    ],
    securityMeasures: [
      'Non-root user execution',
      'Minimal base image',
      'No development dependencies',
      'Health check for orchestration',
    ],
  },
  {
    name: 'Python Production',
    language: 'python',
    framework: 'fastapi',
    dockerfile: `
# syntax=docker/dockerfile:1.4

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Security: create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
`,
    optimizations: [
      'Multi-stage build separates build and runtime',
      'Slim base image for smaller footprint',
      'Virtual environment isolation',
      'No build tools in production image',
    ],
    securityMeasures: [
      'Non-root user execution',
      'Minimal dependencies',
      'No cache in pip install',
      'Health check for orchestration',
    ],
  },
  {
    name: 'Go Production',
    language: 'go',
    dockerfile: `
# syntax=docker/dockerfile:1.4

# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install certificates and timezone data
RUN apk add --no-cache ca-certificates tzdata

# Download dependencies first (better caching)
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \\
    -ldflags="-w -s -X main.version=\${VERSION}" \\
    -o /app/server ./cmd/server

# Production stage - using scratch for minimal image
FROM scratch AS production

# Copy certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy binary
COPY --from=builder /app/server /server

# Run as non-root (numeric UID)
USER 65534

EXPOSE 8080

ENTRYPOINT ["/server"]
`,
    optimizations: [
      'Scratch base for smallest possible image (~5-10MB)',
      'Static binary with CGO disabled',
      'Stripped binary with ldflags',
      'Dependency caching',
    ],
    securityMeasures: [
      'No shell, no OS in scratch image',
      'Static binary eliminates runtime deps',
      'Non-root UID',
      'Minimal attack surface',
    ],
  },
];

export const ARCHITECTURE_PATTERNS: ArchitecturePattern[] = [
  {
    name: 'Microservices',
    description: 'Distributed architecture with independently deployable services',
    useCases: [
      'Large teams working on different features',
      'Services with different scaling requirements',
      'Polyglot development environments',
      'Frequent, independent deployments',
    ],
    components: [
      'API Gateway',
      'Service Discovery',
      'Load Balancers',
      'Message Queues',
      'Distributed Tracing',
      'Centralized Logging',
    ],
    tradeoffs: {
      advantages: [
        'Independent scaling and deployment',
        'Technology flexibility per service',
        'Fault isolation',
        'Easier to understand individual services',
        'Team autonomy',
      ],
      disadvantages: [
        'Distributed system complexity',
        'Network latency between services',
        'Data consistency challenges',
        'Operational overhead',
        'Testing complexity',
      ],
    },
    scalingStrategy: 'Horizontal scaling per service based on individual metrics. Use HPA for Kubernetes workloads.',
    dataFlow: 'Synchronous REST/gRPC for queries, async messaging for events. Saga pattern for distributed transactions.',
  },
  {
    name: 'Modular Monolith',
    description: 'Single deployable unit with well-defined internal boundaries',
    useCases: [
      'Small to medium teams',
      'Early-stage products',
      'Simpler operational requirements',
      'Unclear domain boundaries',
    ],
    components: [
      'Load Balancer',
      'Application Server',
      'Database',
      'Cache',
      'Background Workers',
    ],
    tradeoffs: {
      advantages: [
        'Simpler deployment and operations',
        'Easier debugging and testing',
        'No network overhead between modules',
        'Strong consistency within transactions',
        'Lower infrastructure costs',
      ],
      disadvantages: [
        'Single point of failure',
        'Scaling is all-or-nothing',
        'Technology stack constraints',
        'Risk of becoming a big ball of mud',
        'Longer build/deploy times as it grows',
      ],
    },
    scalingStrategy: 'Horizontal scaling of the entire application. Vertical scaling for database.',
    dataFlow: 'In-process function calls between modules. Single database with logical separation.',
  },
  {
    name: 'Event-Driven',
    description: 'Loosely coupled services communicating via events',
    useCases: [
      'Real-time data processing',
      'Complex workflows across services',
      'Audit logging requirements',
      'Eventually consistent systems',
    ],
    components: [
      'Event Bus (Kafka/RabbitMQ)',
      'Event Store',
      'Event Processors',
      'Projections',
      'Saga Orchestrator',
    ],
    tradeoffs: {
      advantages: [
        'Loose coupling between services',
        'Better resilience (async processing)',
        'Event sourcing for audit trails',
        'Easy to add new consumers',
        'Natural fit for CQRS',
      ],
      disadvantages: [
        'Eventually consistent (complex for users)',
        'Debugging distributed events is hard',
        'Message ordering challenges',
        'Requires event schema management',
        'Infrastructure complexity',
      ],
    },
    scalingStrategy: 'Scale consumers based on message backlog. Partition for parallelism.',
    dataFlow: 'Publishers emit events to topics. Consumers subscribe and process asynchronously.',
  },
];

export class InfrastructureKnowledgeBase {
  getTerraformPattern(name: string): TerraformPattern | undefined {
    return TERRAFORM_PATTERNS.find(p => p.name.toLowerCase().includes(name.toLowerCase()));
  }

  getKubernetesPattern(name: string): KubernetesPattern | undefined {
    return KUBERNETES_PATTERNS.find(p => p.name.toLowerCase().includes(name.toLowerCase()));
  }

  getDockerPattern(language: string): DockerPattern | undefined {
    return DOCKER_PATTERNS.find(p => p.language.toLowerCase() === language.toLowerCase());
  }

  getArchitecturePattern(name: string): ArchitecturePattern | undefined {
    return ARCHITECTURE_PATTERNS.find(p => p.name.toLowerCase().includes(name.toLowerCase()));
  }

  generateTerraformConfig(analysis: any): string {
    const configs: string[] = [];
    
    // VPC
    const vpcPattern = this.getTerraformPattern('vpc');
    if (vpcPattern) {
      configs.push(vpcPattern.template);
    }

    // EKS if microservices
    if (analysis.architecture === 'Microservices') {
      const eksPattern = this.getTerraformPattern('eks');
      if (eksPattern) {
        configs.push(eksPattern.template);
      }
    }

    // RDS for databases
    if (analysis.databases?.length > 0) {
      const rdsPattern = this.getTerraformPattern('rds');
      if (rdsPattern) {
        configs.push(rdsPattern.template);
      }
    }

    return configs.join('\n\n');
  }

  generateKubernetesManifests(analysis: any): string {
    const manifests: string[] = [];

    // Deployment
    const deployPattern = this.getKubernetesPattern('deployment');
    if (deployPattern) {
      manifests.push(deployPattern.manifest);
    }

    // HPA
    const hpaPattern = this.getKubernetesPattern('autoscaler');
    if (hpaPattern) {
      manifests.push(hpaPattern.manifest);
    }

    // Network Policy
    const netPolPattern = this.getKubernetesPattern('network');
    if (netPolPattern) {
      manifests.push(netPolPattern.manifest);
    }

    return manifests.join('\n---\n');
  }

  generateDockerfile(language: string): string {
    const pattern = this.getDockerPattern(language);
    return pattern?.dockerfile || '';
  }

  getBestPractices(component: string): string[] {
    const tfPattern = this.getTerraformPattern(component);
    if (tfPattern) return tfPattern.bestPractices;

    const k8sPattern = this.getKubernetesPattern(component);
    if (k8sPattern) return k8sPattern.bestPractices;

    const dockerPattern = this.getDockerPattern(component);
    if (dockerPattern) return [...dockerPattern.optimizations, ...dockerPattern.securityMeasures];

    return [];
  }

  getSecurityRecommendations(component: string): string[] {
    const tfPattern = this.getTerraformPattern(component);
    if (tfPattern) return tfPattern.securityConsiderations;

    const k8sPattern = this.getKubernetesPattern(component);
    if (k8sPattern) return k8sPattern.bestPractices.filter(bp => 
      bp.toLowerCase().includes('security') || 
      bp.toLowerCase().includes('policy') ||
      bp.toLowerCase().includes('non-root')
    );

    const dockerPattern = this.getDockerPattern(component);
    if (dockerPattern) return dockerPattern.securityMeasures;

    return [];
  }
}

export const knowledgeBase = new InfrastructureKnowledgeBase();
