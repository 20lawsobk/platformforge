import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useSearch, Link } from "wouter";
import Editor from "@monaco-editor/react";
import { 
  Terminal, 
  Play, 
  Share2, 
  FileCode, 
  Folder, 
  ChevronRight, 
  ChevronDown, 
  Box, 
  BrainCircuit,
  Server,
  Globe,
  Shield,
  Download,
  Database,
  Plus,
  X,
  Search,
  GitBranch,
  Settings,
  LayoutDashboard,
  Cpu,
  RefreshCw,
  Copy,
  Check,
  ExternalLink,
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeft,
  TerminalSquare,
  FileJson,
  FileType,
  Layers,
  Users,
  MessageSquare,
  Key,
  Package,
  GitCommit,
  History,
  Trash2,
  Edit3,
  FolderPlus,
  FilePlus,
  MoreVertical,
  Save,
  Undo,
  Redo,
  Command,
  Zap,
  HardDrive,
  Lock,
  Eye,
  EyeOff,
  RotateCcw,
  ChevronUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import JSZip from "jszip";
import { saveAs } from "file-saver";

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  content?: string;
  language?: string;
  children?: FileNode[];
  isNew?: boolean;
  isEditing?: boolean;
}

interface EnvVariable {
  key: string;
  value: string;
  isSecret: boolean;
}

const COMMAND_RESPONSES: Record<string, string[]> = {
  'ls': ['infrastructure/', 'kubernetes/', 'docker/', 'scripts/', 'README.md'],
  'pwd': ['/home/user/project'],
  'whoami': ['platform-architect'],
  'date': [new Date().toString()],
  'echo': [''],
  'npm': ['npm 10.2.0', 'Usage: npm <command>'],
  'node': ['v20.10.0'],
  'terraform': ['Terraform v1.6.0', 'Usage: terraform [global options] <subcommand> [args]'],
  'kubectl': ['Client Version: v1.28.0', 'Server Version: v1.28.0'],
  'docker': ['Docker version 24.0.0, build xxxxx'],
  'git': ['git version 2.42.0'],
  'cat': [''],
  'help': [
    'Available commands:',
    '  ls        - List files',
    '  pwd       - Print working directory',
    '  cat       - Display file contents',
    '  npm       - Node package manager',
    '  terraform - Infrastructure as code',
    '  kubectl   - Kubernetes CLI',
    '  docker    - Container management',
    '  git       - Version control',
    '  clear     - Clear terminal',
    '  help      - Show this help',
  ],
};

export default function Builder() {
  const search = useSearch();
  const params = new URLSearchParams(search);
  const projectId = params.get("id");
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const terminalEndRef = useRef<HTMLDivElement>(null);
  const terminalInputRef = useRef<HTMLInputElement>(null);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState<'architecture' | 'collab' | 'database' | 'env' | 'packages' | 'git'>('architecture');
  const [terminalMaximized, setTerminalMaximized] = useState(false);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [openTabs, setOpenTabs] = useState<string[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['/', '/infrastructure', '/kubernetes', '/docker', '/scripts']));
  const [copied, setCopied] = useState(false);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState<{type: 'input' | 'output', text: string}[]>([
    { type: 'output', text: '$ PlatformArchitect Terminal v1.0' },
    { type: 'output', text: '$ Type "help" for available commands' },
  ]);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [fileContents, setFileContents] = useState<Record<string, string>>({});
  const [unsavedChanges, setUnsavedChanges] = useState<Set<string>>(new Set());
  const [newItemDialog, setNewItemDialog] = useState<{open: boolean, type: 'file' | 'folder', parentPath: string}>({open: false, type: 'file', parentPath: '/'});
  const [newItemName, setNewItemName] = useState('');
  const [envVariables, setEnvVariables] = useState<EnvVariable[]>([
    { key: 'DATABASE_URL', value: 'postgresql://localhost:5432/db', isSecret: true },
    { key: 'REDIS_URL', value: 'redis://localhost:6379', isSecret: true },
    { key: 'NODE_ENV', value: 'production', isSecret: false },
    { key: 'API_PORT', value: '3000', isSecret: false },
  ]);
  const [showSecrets, setShowSecrets] = useState(false);
  const [packages, setPackages] = useState([
    { name: 'express', version: '^4.18.2', type: 'dependency' },
    { name: 'pg', version: '^8.11.0', type: 'dependency' },
    { name: 'redis', version: '^4.6.0', type: 'dependency' },
    { name: 'typescript', version: '^5.0.0', type: 'devDependency' },
    { name: 'jest', version: '^29.5.0', type: 'devDependency' },
  ]);
  const [gitHistory, setGitHistory] = useState([
    { hash: 'a1b2c3d', message: 'Initial infrastructure generation', author: 'AI Agent', time: '2 minutes ago' },
    { hash: 'e4f5g6h', message: 'Add Kubernetes manifests', author: 'AI Agent', time: '5 minutes ago' },
    { hash: 'i7j8k9l', message: 'Configure auto-scaling', author: 'AI Agent', time: '8 minutes ago' },
  ]);
  const [collaborators, setCollaborators] = useState([
    { name: 'You', avatar: 'Y', color: 'bg-green-500', status: 'online', cursor: null },
    { name: 'AI Agent', avatar: 'AI', color: 'bg-purple-500', status: 'online', cursor: 'infrastructure/main.tf:42' },
  ]);

  const { data: project } = useQuery({
    queryKey: ['project', projectId],
    queryFn: async () => {
      const res = await fetch(`/api/projects/${projectId}`);
      if (!res.ok) throw new Error('Failed to fetch project');
      return res.json();
    },
    enabled: !!projectId,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === 'complete' || data?.status === 'failed' ? false : 1000;
    },
  });

  const { data: infrastructure } = useQuery({
    queryKey: ['infrastructure', projectId],
    queryFn: async () => {
      const res = await fetch(`/api/projects/${projectId}/infrastructure`);
      if (!res.ok) throw new Error('Failed to fetch infrastructure');
      return res.json();
    },
    enabled: !!projectId && project?.status === 'complete',
  });

  const { data: logs = [] } = useQuery({
    queryKey: ['logs', projectId],
    queryFn: async () => {
      const res = await fetch(`/api/projects/${projectId}/logs`);
      if (!res.ok) throw new Error('Failed to fetch logs');
      return res.json();
    },
    enabled: !!projectId,
    refetchInterval: 1000,
  });

  const fileSystem = useMemo<FileNode[]>(() => {
    if (!infrastructure) {
      return [
        { name: 'Generating...', path: '/generating', type: 'file', content: '// Infrastructure generation in progress...\n// Please wait while we analyze your project.', language: 'typescript' }
      ];
    }

    const projectName = project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'app';

    return [
      {
        name: 'infrastructure',
        path: '/infrastructure',
        type: 'folder',
        children: [
          { name: 'main.tf', path: '/infrastructure/main.tf', type: 'file', content: infrastructure.terraformConfig || '# Terraform configuration', language: 'hcl' },
          { name: 'variables.tf', path: '/infrastructure/variables.tf', type: 'file', content: `# Variables for ${projectName}\n\nvariable "environment" {\n  description = "Deployment environment"\n  type        = string\n  default     = "production"\n}\n\nvariable "region" {\n  description = "AWS region"\n  type        = string\n  default     = "us-east-1"\n}\n\nvariable "db_password" {\n  description = "Database password"\n  type        = string\n  sensitive   = true\n}\n\nvariable "min_capacity" {\n  description = "Minimum number of instances"\n  type        = number\n  default     = ${infrastructure.minInstances || 2}\n}\n\nvariable "max_capacity" {\n  description = "Maximum number of instances"\n  type        = number\n  default     = ${infrastructure.maxInstances || 20}\n}`, language: 'hcl' },
          { name: 'outputs.tf', path: '/infrastructure/outputs.tf', type: 'file', content: `# Outputs for ${projectName}\n\noutput "cluster_endpoint" {\n  description = "EKS cluster endpoint"\n  value       = aws_eks_cluster.main.endpoint\n}\n\noutput "cluster_name" {\n  description = "EKS cluster name"\n  value       = aws_eks_cluster.main.name\n}\n\noutput "database_endpoint" {\n  description = "RDS instance endpoint"\n  value       = aws_db_instance.main.endpoint\n}\n\noutput "redis_endpoint" {\n  description = "ElastiCache endpoint"\n  value       = aws_elasticache_cluster.main.cache_nodes[0].address\n}\n\noutput "load_balancer_dns" {\n  description = "Application Load Balancer DNS"\n  value       = aws_lb.main.dns_name\n}`, language: 'hcl' },
          { name: 'providers.tf', path: '/infrastructure/providers.tf', type: 'file', content: `# Provider configuration\n\nterraform {\n  required_version = ">= 1.0"\n\n  required_providers {\n    aws = {\n      source  = "hashicorp/aws"\n      version = "~> 5.0"\n    }\n    kubernetes = {\n      source  = "hashicorp/kubernetes"\n      version = "~> 2.23"\n    }\n    helm = {\n      source  = "hashicorp/helm"\n      version = "~> 2.11"\n    }\n  }\n\n  backend "s3" {\n    bucket = "${projectName}-terraform-state"\n    key    = "infrastructure/terraform.tfstate"\n    region = "us-east-1"\n  }\n}\n\nprovider "aws" {\n  region = var.region\n\n  default_tags {\n    tags = {\n      Project     = "${projectName}"\n      Environment = var.environment\n      ManagedBy   = "Terraform"\n    }\n  }\n}`, language: 'hcl' },
        ]
      },
      {
        name: 'kubernetes',
        path: '/kubernetes',
        type: 'folder',
        children: [
          { name: 'deployment.yaml', path: '/kubernetes/deployment.yaml', type: 'file', content: infrastructure.kubernetesConfig || '# Kubernetes deployment', language: 'yaml' },
          { name: 'service.yaml', path: '/kubernetes/service.yaml', type: 'file', content: `apiVersion: v1\nkind: Service\nmetadata:\n  name: ${projectName}-service\n  labels:\n    app: ${projectName}\nspec:\n  type: LoadBalancer\n  ports:\n    - port: 80\n      targetPort: 3000\n      protocol: TCP\n      name: http\n  selector:\n    app: ${projectName}`, language: 'yaml' },
          { name: 'ingress.yaml', path: '/kubernetes/ingress.yaml', type: 'file', content: `apiVersion: networking.k8s.io/v1\nkind: Ingress\nmetadata:\n  name: ${projectName}-ingress\n  annotations:\n    kubernetes.io/ingress.class: nginx\n    cert-manager.io/cluster-issuer: letsencrypt-prod\n    nginx.ingress.kubernetes.io/ssl-redirect: "true"\nspec:\n  tls:\n    - hosts:\n        - ${projectName}.example.com\n      secretName: ${projectName}-tls\n  rules:\n    - host: ${projectName}.example.com\n      http:\n        paths:\n          - path: /\n            pathType: Prefix\n            backend:\n              service:\n                name: ${projectName}-service\n                port:\n                  number: 80`, language: 'yaml' },
          { name: 'hpa.yaml', path: '/kubernetes/hpa.yaml', type: 'file', content: `apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\nmetadata:\n  name: ${projectName}-hpa\nspec:\n  scaleTargetRef:\n    apiVersion: apps/v1\n    kind: Deployment\n    name: ${projectName}-deployment\n  minReplicas: ${infrastructure.minInstances || 2}\n  maxReplicas: ${infrastructure.maxInstances || 20}\n  metrics:\n    - type: Resource\n      resource:\n        name: cpu\n        target:\n          type: Utilization\n          averageUtilization: 70\n    - type: Resource\n      resource:\n        name: memory\n        target:\n          type: Utilization\n          averageUtilization: 80`, language: 'yaml' },
          { name: 'configmap.yaml', path: '/kubernetes/configmap.yaml', type: 'file', content: `apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: ${projectName}-config\ndata:\n  NODE_ENV: "production"\n  LOG_LEVEL: "info"\n  API_PORT: "3000"`, language: 'yaml' },
          { name: 'secrets.yaml', path: '/kubernetes/secrets.yaml', type: 'file', content: `apiVersion: v1\nkind: Secret\nmetadata:\n  name: ${projectName}-secrets\ntype: Opaque\nstringData:\n  DATABASE_URL: "\${DATABASE_URL}"\n  REDIS_URL: "\${REDIS_URL}"\n  API_KEY: "\${API_KEY}"`, language: 'yaml' },
        ]
      },
      {
        name: 'docker',
        path: '/docker',
        type: 'folder',
        children: [
          { name: 'Dockerfile', path: '/docker/Dockerfile', type: 'file', content: infrastructure.dockerfileContent || '# Dockerfile', language: 'dockerfile' },
          { name: 'Dockerfile.dev', path: '/docker/Dockerfile.dev', type: 'file', content: `# Development Dockerfile\nFROM node:20-alpine\n\nWORKDIR /app\n\n# Install development dependencies\nCOPY package*.json ./\nRUN npm install\n\n# Copy source code\nCOPY . .\n\n# Expose development port\nEXPOSE 3000\n\n# Start development server with hot reload\nCMD ["npm", "run", "dev"]`, language: 'dockerfile' },
          { name: 'docker-compose.yml', path: '/docker/docker-compose.yml', type: 'file', content: `version: '3.8'\n\nservices:\n  app:\n    build:\n      context: .\n      dockerfile: docker/Dockerfile\n    ports:\n      - "3000:3000"\n    environment:\n      - NODE_ENV=production\n      - DATABASE_URL=\${DATABASE_URL}\n      - REDIS_URL=\${REDIS_URL}\n    depends_on:\n      postgres:\n        condition: service_healthy\n      redis:\n        condition: service_started\n    restart: unless-stopped\n    networks:\n      - ${projectName}-network\n\n  postgres:\n    image: postgres:15-alpine\n    environment:\n      POSTGRES_USER: \${DB_USER:-app}\n      POSTGRES_PASSWORD: \${DB_PASSWORD}\n      POSTGRES_DB: \${DB_NAME:-${projectName}}\n    volumes:\n      - postgres_data:/var/lib/postgresql/data\n    healthcheck:\n      test: ["CMD-SHELL", "pg_isready -U app"]\n      interval: 10s\n      timeout: 5s\n      retries: 5\n    restart: unless-stopped\n    networks:\n      - ${projectName}-network\n\n  redis:\n    image: redis:7-alpine\n    command: redis-server --appendonly yes\n    volumes:\n      - redis_data:/data\n    restart: unless-stopped\n    networks:\n      - ${projectName}-network\n\nvolumes:\n  postgres_data:\n  redis_data:\n\nnetworks:\n  ${projectName}-network:\n    driver: bridge`, language: 'yaml' },
          { name: 'docker-compose.dev.yml', path: '/docker/docker-compose.dev.yml', type: 'file', content: `version: '3.8'\n\nservices:\n  app:\n    build:\n      context: .\n      dockerfile: docker/Dockerfile.dev\n    ports:\n      - "3000:3000"\n    volumes:\n      - .:/app\n      - /app/node_modules\n    environment:\n      - NODE_ENV=development\n    depends_on:\n      - postgres\n      - redis\n\n  postgres:\n    image: postgres:15-alpine\n    ports:\n      - "5432:5432"\n    environment:\n      POSTGRES_USER: dev\n      POSTGRES_PASSWORD: devpassword\n      POSTGRES_DB: ${projectName}_dev\n\n  redis:\n    image: redis:7-alpine\n    ports:\n      - "6379:6379"`, language: 'yaml' },
          { name: '.dockerignore', path: '/docker/.dockerignore', type: 'file', content: `node_modules\nnpm-debug.log*\n.git\n.gitignore\n.env\n.env.*\n*.md\n!README.md\n.DS_Store\ncoverage\n.nyc_output\ndist\n.cache\n*.log\nDockerfile*\ndocker-compose*\n.dockerignore`, language: 'plaintext' },
        ]
      },
      {
        name: 'scripts',
        path: '/scripts',
        type: 'folder',
        children: [
          { name: 'deploy.sh', path: '/scripts/deploy.sh', type: 'file', content: `#!/bin/bash\nset -euo pipefail\n\n# ${projectName} Deployment Script\n# Generated by PlatformArchitect\n\necho "🚀 Starting deployment for ${projectName}..."\n\n# Configuration\nREGISTRY="\${REGISTRY:-your-registry.io}"\nIMAGE_TAG="\${IMAGE_TAG:-latest}"\nNAMESPACE="\${NAMESPACE:-production}"\n\n# Build Docker image\necho "📦 Building Docker image..."\ndocker build -t $REGISTRY/${projectName}:$IMAGE_TAG -f docker/Dockerfile .\n\n# Push to registry\necho "📤 Pushing to container registry..."\ndocker push $REGISTRY/${projectName}:$IMAGE_TAG\n\n# Update Kubernetes deployment\necho "☸️ Updating Kubernetes deployment..."\nkubectl set image deployment/${projectName}-deployment \\\n  ${projectName}=$REGISTRY/${projectName}:$IMAGE_TAG \\\n  -n $NAMESPACE\n\n# Wait for rollout\necho "⏳ Waiting for rollout to complete..."\nkubectl rollout status deployment/${projectName}-deployment -n $NAMESPACE\n\necho "✅ Deployment complete!"\necho "🌐 Application is live at: https://${projectName}.example.com"`, language: 'bash' },
          { name: 'setup.sh', path: '/scripts/setup.sh', type: 'file', content: `#!/bin/bash\nset -euo pipefail\n\n# ${projectName} Infrastructure Setup\n# Generated by PlatformArchitect\n\necho "🏗️ Setting up infrastructure for ${projectName}..."\n\ncd infrastructure\n\n# Initialize Terraform\necho "📋 Initializing Terraform..."\nterraform init\n\n# Validate configuration\necho "✔️ Validating Terraform configuration..."\nterraform validate\n\n# Plan changes\necho "📝 Planning infrastructure changes..."\nterraform plan -out=tfplan\n\necho ""\necho "Review the plan above. To apply, run:"\necho "  terraform apply tfplan"`, language: 'bash' },
          { name: 'rollback.sh', path: '/scripts/rollback.sh', type: 'file', content: `#!/bin/bash\nset -euo pipefail\n\n# ${projectName} Rollback Script\n\nNAMESPACE="\${NAMESPACE:-production}"\nREVISION="\${1:-}"\n\nif [ -z "$REVISION" ]; then\n  echo "Usage: ./rollback.sh <revision-number>"\n  echo ""\n  echo "Available revisions:"\n  kubectl rollout history deployment/${projectName}-deployment -n $NAMESPACE\n  exit 1\nfi\n\necho "⏪ Rolling back to revision $REVISION..."\nkubectl rollout undo deployment/${projectName}-deployment \\\n  --to-revision=$REVISION \\\n  -n $NAMESPACE\n\necho "⏳ Waiting for rollback to complete..."\nkubectl rollout status deployment/${projectName}-deployment -n $NAMESPACE\n\necho "✅ Rollback complete!"`, language: 'bash' },
          { name: 'healthcheck.sh', path: '/scripts/healthcheck.sh', type: 'file', content: `#!/bin/bash\n\n# ${projectName} Health Check Script\n\nENDPOINT="\${1:-http://localhost:3000/health}"\nTIMEOUT=5\nRETRIES=3\n\nfor i in $(seq 1 $RETRIES); do\n  echo "Checking health (attempt $i/$RETRIES)..."\n  \n  if curl -sf --max-time $TIMEOUT "$ENDPOINT" > /dev/null; then\n    echo "✅ Health check passed!"\n    exit 0\n  fi\n  \n  sleep 2\ndone\n\necho "❌ Health check failed after $RETRIES attempts"\nexit 1`, language: 'bash' },
        ]
      },
      {
        name: '.github',
        path: '/.github',
        type: 'folder',
        children: [
          {
            name: 'workflows',
            path: '/.github/workflows',
            type: 'folder',
            children: [
              { name: 'ci.yml', path: '/.github/workflows/ci.yml', type: 'file', content: `name: CI/CD Pipeline\n\non:\n  push:\n    branches: [main, develop]\n  pull_request:\n    branches: [main]\n\nenv:\n  REGISTRY: ghcr.io\n  IMAGE_NAME: \${{ github.repository }}\n\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-node@v4\n        with:\n          node-version: '20'\n          cache: 'npm'\n      - run: npm ci\n      - run: npm run lint\n      - run: npm test\n\n  build:\n    needs: test\n    runs-on: ubuntu-latest\n    permissions:\n      contents: read\n      packages: write\n    steps:\n      - uses: actions/checkout@v4\n      - uses: docker/login-action@v3\n        with:\n          registry: \${{ env.REGISTRY }}\n          username: \${{ github.actor }}\n          password: \${{ secrets.GITHUB_TOKEN }}\n      - uses: docker/build-push-action@v5\n        with:\n          context: .\n          file: docker/Dockerfile\n          push: \${{ github.event_name != 'pull_request' }}\n          tags: \${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}:latest\n\n  deploy:\n    if: github.ref == 'refs/heads/main'\n    needs: build\n    runs-on: ubuntu-latest\n    environment: production\n    steps:\n      - uses: actions/checkout@v4\n      - uses: azure/k8s-set-context@v3\n        with:\n          kubeconfig: \${{ secrets.KUBE_CONFIG }}\n      - run: kubectl apply -f kubernetes/`, language: 'yaml' },
            ]
          }
        ]
      },
      { name: 'README.md', path: '/README.md', type: 'file', content: `# ${project?.name || 'Project'}\n\nGenerated by **PlatformArchitect AI**\n\n## Overview\n\nThis repository contains production-ready infrastructure for deploying ${project?.name || 'your application'} at scale.\n\n## Architecture\n\n- **Compute**: AWS EKS (Kubernetes) with auto-scaling (${infrastructure?.minInstances || 2}-${infrastructure?.maxInstances || 20} nodes)\n- **Database**: PostgreSQL on AWS RDS with read replicas\n- **Cache**: Redis on AWS ElastiCache\n- **CDN**: CloudFront for static assets\n- **Monitoring**: Prometheus + Grafana\n\n## Quick Start\n\n### Prerequisites\n\n- Terraform >= 1.0\n- kubectl >= 1.28\n- Docker >= 24.0\n- AWS CLI configured with appropriate permissions\n\n### Deploy Infrastructure\n\n\`\`\`bash\n# 1. Initialize and apply Terraform\ncd infrastructure\nterraform init\nterraform apply\n\n# 2. Build and push Docker image\ndocker build -t your-registry/${projectName}:latest -f docker/Dockerfile .\ndocker push your-registry/${projectName}:latest\n\n# 3. Deploy to Kubernetes\nkubectl apply -f kubernetes/\n\`\`\`\n\n### Development\n\n\`\`\`bash\n# Start development environment\ndocker-compose -f docker/docker-compose.dev.yml up\n\`\`\`\n\n## Configuration\n\nSee \`infrastructure/variables.tf\` for all configurable options.\n\n## Monitoring\n\nAccess Grafana dashboard at: https://grafana.${projectName}.example.com\n\n## Support\n\nGenerated by PlatformArchitect - Enterprise Infrastructure Platform`, language: 'markdown' },
      { name: 'package.json', path: '/package.json', type: 'file', content: JSON.stringify({ name: projectName, version: "1.0.0", scripts: { start: "node dist/index.js", dev: "tsx watch src/index.ts", build: "tsc", test: "jest", lint: "eslint src/" }, dependencies: { express: "^4.18.2", pg: "^8.11.0", redis: "^4.6.0", dotenv: "^16.0.0" }, devDependencies: { typescript: "^5.0.0", "@types/node": "^20.0.0", "@types/express": "^4.17.0", jest: "^29.5.0", eslint: "^8.0.0", tsx: "^4.0.0" } }, null, 2), language: 'json' },
      { name: '.env.example', path: '/.env.example', type: 'file', content: `# Environment Variables for ${projectName}\n# Copy this file to .env and fill in the values\n\n# Database\nDATABASE_URL=postgresql://user:password@localhost:5432/${projectName}\nDB_USER=app\nDB_PASSWORD=\nDB_NAME=${projectName}\n\n# Redis\nREDIS_URL=redis://localhost:6379\n\n# Application\nNODE_ENV=development\nPORT=3000\nAPI_KEY=\n\n# AWS (for Terraform)\nAWS_ACCESS_KEY_ID=\nAWS_SECRET_ACCESS_KEY=\nAWS_REGION=us-east-1`, language: 'plaintext' },
    ];
  }, [infrastructure, project]);

  useEffect(() => {
    if (infrastructure && !activeFile && fileSystem.length > 0) {
      const defaultFile = '/infrastructure/main.tf';
      setActiveFile(defaultFile);
      setOpenTabs([defaultFile]);
    }
  }, [infrastructure, activeFile, fileSystem]);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [terminalHistory]);

  useEffect(() => {
    if (project?.status === 'complete') {
      toast({
        title: "Infrastructure Generated",
        description: "Your production infrastructure is ready. Review and deploy.",
      });
    }
  }, [project?.status, toast]);

  const findFile = useCallback((path: string, nodes: FileNode[] = fileSystem): FileNode | null => {
    for (const node of nodes) {
      if (node.path === path) return node;
      if (node.children) {
        const found = findFile(path, node.children);
        if (found) return found;
      }
    }
    return null;
  }, [fileSystem]);

  const getFileContent = useCallback((path: string): string => {
    if (fileContents[path] !== undefined) return fileContents[path];
    const file = findFile(path);
    return file?.content || '';
  }, [fileContents, findFile]);

  const activeFileData = activeFile ? findFile(activeFile) : null;
  const activeFileContent = activeFile ? getFileContent(activeFile) : '';

  const handleFileClick = (file: FileNode) => {
    if (file.type === 'folder') {
      setExpandedFolders(prev => {
        const next = new Set(prev);
        if (next.has(file.path)) {
          next.delete(file.path);
        } else {
          next.add(file.path);
        }
        return next;
      });
    } else {
      setActiveFile(file.path);
      if (!openTabs.includes(file.path)) {
        setOpenTabs([...openTabs, file.path]);
      }
    }
  };

  const handleCloseTab = (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (unsavedChanges.has(path)) {
      if (!confirm('You have unsaved changes. Close anyway?')) return;
    }
    const newTabs = openTabs.filter(t => t !== path);
    setOpenTabs(newTabs);
    setUnsavedChanges(prev => {
      const next = new Set(prev);
      next.delete(path);
      return next;
    });
    if (activeFile === path) {
      setActiveFile(newTabs[newTabs.length - 1] || null);
    }
  };

  const handleEditorChange = (value: string | undefined) => {
    if (!activeFile || value === undefined) return;
    setFileContents(prev => ({ ...prev, [activeFile]: value }));
    setUnsavedChanges(prev => new Set(Array.from(prev).concat(activeFile)));
  };

  const handleSaveFile = () => {
    if (!activeFile) return;
    setUnsavedChanges(prev => {
      const next = new Set(prev);
      next.delete(activeFile);
      return next;
    });
    toast({
      title: "File Saved",
      description: `${activeFile.split('/').pop()} has been saved.`,
    });
  };

  const handleCopyCode = () => {
    if (activeFileContent) {
      navigator.clipboard.writeText(activeFileContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({ title: "Copied to clipboard" });
    }
  };

  const handleTerminalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!terminalInput.trim()) return;

    const cmd = terminalInput.trim();
    const parts = cmd.split(' ');
    const command = parts[0].toLowerCase();
    const args = parts.slice(1);

    setTerminalHistory(prev => [...prev, { type: 'input', text: `$ ${cmd}` }]);
    setCommandHistory(prev => [...prev, cmd]);
    setHistoryIndex(-1);

    if (command === 'clear') {
      setTerminalHistory([]);
    } else if (command === 'cat' && args.length > 0) {
      const filePath = args[0].startsWith('/') ? args[0] : '/' + args[0];
      const file = findFile(filePath);
      if (file && file.content) {
        file.content.split('\n').forEach(line => {
          setTerminalHistory(prev => [...prev, { type: 'output', text: line }]);
        });
      } else {
        setTerminalHistory(prev => [...prev, { type: 'output', text: `cat: ${args[0]}: No such file or directory` }]);
      }
    } else if (command === 'echo') {
      setTerminalHistory(prev => [...prev, { type: 'output', text: args.join(' ') }]);
    } else if (COMMAND_RESPONSES[command]) {
      COMMAND_RESPONSES[command].forEach(line => {
        setTerminalHistory(prev => [...prev, { type: 'output', text: line }]);
      });
    } else {
      setTerminalHistory(prev => [...prev, { type: 'output', text: `${command}: command not found. Type 'help' for available commands.` }]);
    }

    setTerminalInput('');
  };

  const handleTerminalKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        setTerminalInput(commandHistory[commandHistory.length - 1 - newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setTerminalInput(commandHistory[commandHistory.length - 1 - newIndex]);
      } else {
        setHistoryIndex(-1);
        setTerminalInput('');
      }
    }
  };

  const handleDownloadArtifacts = async () => {
    const zip = new JSZip();
    
    const addToZip = (nodes: FileNode[], parentPath: string = '') => {
      nodes.forEach(node => {
        const path = parentPath ? `${parentPath}/${node.name}` : node.name;
        if (node.type === 'file') {
          const content = getFileContent(node.path);
          zip.file(path, content);
        } else if (node.children) {
          addToZip(node.children, path);
        }
      });
    };

    addToZip(fileSystem);
    const content = await zip.generateAsync({ type: "blob" });
    saveAs(content, `${project?.name || 'infrastructure'}-bundle.zip`);
    toast({ title: "Downloaded", description: "Complete infrastructure bundle saved." });
  };

  const getLanguage = (fileName: string): string => {
    if (fileName.endsWith('.tf')) return 'hcl';
    if (fileName.endsWith('.yaml') || fileName.endsWith('.yml')) return 'yaml';
    if (fileName.endsWith('.json')) return 'json';
    if (fileName.endsWith('.md')) return 'markdown';
    if (fileName.endsWith('.sh')) return 'shell';
    if (fileName.endsWith('.ts') || fileName.endsWith('.tsx')) return 'typescript';
    if (fileName.endsWith('.js') || fileName.endsWith('.jsx')) return 'javascript';
    if (fileName === 'Dockerfile' || fileName.startsWith('Dockerfile')) return 'dockerfile';
    return 'plaintext';
  };

  const getFileIcon = (name: string) => {
    if (name.endsWith('.tf')) return <FileCode className="h-4 w-4 text-purple-400" />;
    if (name.endsWith('.yaml') || name.endsWith('.yml')) return <FileJson className="h-4 w-4 text-yellow-400" />;
    if (name.endsWith('.md')) return <FileType className="h-4 w-4 text-blue-400" />;
    if (name.endsWith('.sh')) return <TerminalSquare className="h-4 w-4 text-green-400" />;
    if (name.endsWith('.json')) return <FileJson className="h-4 w-4 text-orange-400" />;
    if (name === 'Dockerfile' || name.startsWith('Dockerfile') || name === '.dockerignore') return <Layers className="h-4 w-4 text-cyan-400" />;
    return <FileCode className="h-4 w-4 text-muted-foreground" />;
  };

  const isBuilding = project?.status === 'analyzing' || project?.status === 'generating' || project?.status === 'pending';
  const isComplete = project?.status === 'complete';

  return (
    <div className="h-screen flex flex-col bg-[#0d1117] text-foreground overflow-hidden" data-testid="builder-container">
      <header className="h-12 border-b border-[#30363d] bg-[#161b22] flex items-center justify-between px-3 shrink-0">
        <div className="flex items-center gap-3">
          <Link href="/">
            <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" data-testid="link-home">
              <Cpu className="h-5 w-5 text-primary" />
              <span className="font-mono font-bold text-sm">Platform<span className="text-primary">Architect</span></span>
            </div>
          </Link>
          
          <div className="h-4 w-px bg-[#30363d]" />
          
          <div className="flex items-center gap-2">
            <Box className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium truncate max-w-[200px]" data-testid="text-project-name">
              {project?.name || 'Loading...'}
            </span>
            <Badge 
              variant="outline" 
              className={`text-[10px] ${isBuilding ? 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30 animate-pulse' : 'bg-green-500/10 text-green-500 border-green-500/30'}`}
              data-testid="badge-status"
            >
              {isBuilding ? "Generating..." : isComplete ? "Ready" : project?.status}
            </Badge>
          </div>

          <div className="h-4 w-px bg-[#30363d]" />

          <div className="flex items-center gap-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleSaveFile} disabled={!unsavedChanges.has(activeFile || '')} data-testid="button-save">
                  <Save className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Save (Ctrl+S)</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8" data-testid="button-undo">
                  <Undo className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Undo</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8" data-testid="button-redo">
                  <Redo className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Redo</TooltipContent>
            </Tooltip>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 mr-2">
            {collaborators.map((c, i) => (
              <Tooltip key={i}>
                <TooltipTrigger>
                  <div className={`h-7 w-7 rounded-full ${c.color} flex items-center justify-center text-[10px] font-bold text-white border-2 border-[#0d1117] -ml-2 first:ml-0`}>
                    {c.avatar}
                  </div>
                </TooltipTrigger>
                <TooltipContent>{c.name} - {c.status}</TooltipContent>
              </Tooltip>
            ))}
          </div>

          <Button variant="ghost" size="sm" className="h-8 text-xs" data-testid="button-share">
            <Share2 className="h-3.5 w-3.5 mr-1.5" /> Share
          </Button>
          
          {isComplete && (
            <Button 
              variant="outline" 
              size="sm"
              className="h-8 text-xs border-primary/50 text-primary hover:bg-primary/10"
              onClick={handleDownloadArtifacts}
              data-testid="button-download"
            >
              <Download className="h-3.5 w-3.5 mr-1.5" /> Export
            </Button>
          )}

          <Button 
            size="sm" 
            className="h-8 text-xs bg-green-600 hover:bg-green-700 text-white font-medium" 
            disabled={!isComplete}
            data-testid="button-deploy"
          >
            <Play className="h-3.5 w-3.5 mr-1.5 fill-current" /> Deploy
          </Button>

          <Link href="/dashboard">
            <Button variant="ghost" size="icon" className="h-8 w-8" data-testid="button-dashboard">
              <LayoutDashboard className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {sidebarOpen && (
          <aside className="w-60 border-r border-[#30363d] bg-[#0d1117] flex flex-col shrink-0">
            <div className="h-10 flex items-center justify-between px-3 border-b border-[#30363d]">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Files</span>
              <div className="flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-new-file" onClick={() => setNewItemDialog({open: true, type: 'file', parentPath: '/'})}>
                      <FilePlus className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>New File</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-new-folder" onClick={() => setNewItemDialog({open: true, type: 'folder', parentPath: '/'})}>
                      <FolderPlus className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>New Folder</TooltipContent>
                </Tooltip>
              </div>
            </div>
            
            <div className="px-2 py-2">
              <div className="relative">
                <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input 
                  placeholder="Search files..." 
                  className="h-7 pl-7 text-xs bg-[#161b22] border-[#30363d]"
                  data-testid="input-search-files"
                />
              </div>
            </div>

            <ScrollArea className="flex-1">
              <div className="px-1 pb-4">
                {fileSystem.map((node) => (
                  <FileTreeNode 
                    key={node.path} 
                    node={node} 
                    level={0}
                    activeFile={activeFile}
                    expandedFolders={expandedFolders}
                    onFileClick={handleFileClick}
                    getFileIcon={getFileIcon}
                    unsavedChanges={unsavedChanges}
                  />
                ))}
              </div>
            </ScrollArea>
          </aside>
        )}

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex h-9 bg-[#161b22] border-b border-[#30363d] items-center shrink-0">
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 rounded-none border-r border-[#30363d] shrink-0"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              data-testid="button-toggle-sidebar"
            >
              {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
            </Button>

            <div className="flex-1 overflow-x-auto">
              <div className="flex">
                {openTabs.map(tabPath => {
                  const fileName = tabPath.split('/').pop() || '';
                  const hasChanges = unsavedChanges.has(tabPath);
                  return (
                    <div
                      key={tabPath}
                      className={`flex items-center gap-2 px-3 h-9 border-r border-[#30363d] cursor-pointer text-xs shrink-0 ${
                        activeFile === tabPath 
                          ? 'bg-[#0d1117] text-foreground border-t-2 border-t-primary' 
                          : 'text-muted-foreground hover:text-foreground hover:bg-[#0d1117]/50'
                      }`}
                      onClick={() => setActiveFile(tabPath)}
                      data-testid={`tab-${fileName}`}
                    >
                      {getFileIcon(fileName)}
                      <span className={hasChanges ? 'italic' : ''}>{fileName}</span>
                      {hasChanges && <span className="h-2 w-2 rounded-full bg-primary" />}
                      <button 
                        onClick={(e) => handleCloseTab(tabPath, e)}
                        className="ml-1 hover:bg-[#30363d] rounded p-0.5"
                        data-testid={`button-close-tab-${fileName}`}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>

            {activeFile && (
              <div className="flex items-center gap-1 px-2 border-l border-[#30363d] shrink-0">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleCopyCode} data-testid="button-copy-code">
                      {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Copy</TooltipContent>
                </Tooltip>
              </div>
            )}
          </div>

          <div className={`flex-1 flex flex-col ${terminalMaximized ? 'hidden' : ''} overflow-hidden`}>
            {activeFileData ? (
              <div className="flex-1 overflow-hidden">
                <Editor
                  height="100%"
                  language={getLanguage(activeFileData.name)}
                  value={activeFileContent}
                  onChange={handleEditorChange}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: true },
                    fontSize: 13,
                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                    lineNumbers: 'on',
                    renderLineHighlight: 'all',
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    tabSize: 2,
                    wordWrap: 'on',
                    padding: { top: 10 },
                  }}
                />
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-muted-foreground bg-[#0d1117]">
                <div className="text-center">
                  <FileCode className="h-16 w-16 mx-auto mb-4 opacity-20" />
                  <p className="text-sm mb-2">Select a file to edit</p>
                  <p className="text-xs text-muted-foreground">Use the file explorer or press Ctrl+P</p>
                </div>
              </div>
            )}
          </div>

          <div className={`${terminalMaximized ? 'flex-1' : 'h-48'} border-t border-[#30363d] bg-[#0d1117] flex flex-col shrink-0`}>
            <div className="flex items-center justify-between h-9 px-3 bg-[#161b22] border-b border-[#30363d] shrink-0">
              <Tabs defaultValue="terminal" className="h-full">
                <TabsList className="h-full bg-transparent gap-2">
                  <TabsTrigger value="terminal" className="h-full px-2 text-xs data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none" data-testid="tab-terminal">
                    <TerminalSquare className="h-3.5 w-3.5 mr-1.5" /> Terminal
                  </TabsTrigger>
                  <TabsTrigger value="output" className="h-full px-2 text-xs data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none" data-testid="tab-output">
                    <BrainCircuit className="h-3.5 w-3.5 mr-1.5" /> AI Output
                  </TabsTrigger>
                </TabsList>
              </Tabs>
              <div className="flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setTerminalHistory([])} data-testid="button-clear-terminal">
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Clear</TooltipContent>
                </Tooltip>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={() => setTerminalMaximized(!terminalMaximized)}
                  data-testid="button-toggle-terminal"
                >
                  {terminalMaximized ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
                </Button>
              </div>
            </div>

            <ScrollArea className="flex-1 p-3">
              <div className="font-mono text-xs space-y-0.5">
                {terminalHistory.map((entry, i) => (
                  <div key={i} className={entry.type === 'input' ? 'text-green-400' : 'text-[#8b949e]'} data-testid={`terminal-line-${i}`}>
                    {entry.text}
                  </div>
                ))}
                {isBuilding && logs.slice(-3).map((log: any, i: number) => (
                  <div key={`log-${i}`} className="text-purple-400 flex items-center gap-2">
                    <BrainCircuit className="h-3 w-3" /> {log.message}
                  </div>
                ))}
                <div ref={terminalEndRef} />
              </div>
            </ScrollArea>

            <form onSubmit={handleTerminalSubmit} className="px-3 pb-3 shrink-0">
              <div className="flex items-center gap-2 bg-[#161b22] rounded border border-[#30363d] px-3 py-1.5">
                <span className="text-green-400 text-xs">$</span>
                <input
                  ref={terminalInputRef}
                  type="text"
                  value={terminalInput}
                  onChange={(e) => setTerminalInput(e.target.value)}
                  onKeyDown={handleTerminalKeyDown}
                  placeholder="Type a command..."
                  className="flex-1 bg-transparent text-xs outline-none text-foreground placeholder:text-muted-foreground font-mono"
                  data-testid="input-terminal"
                />
              </div>
            </form>
          </div>
        </div>

        {rightPanelOpen && (
          <aside className="w-72 border-l border-[#30363d] bg-[#0d1117] flex flex-col shrink-0">
            <div className="flex border-b border-[#30363d] bg-[#161b22]">
              {[
                { id: 'architecture', icon: Server, label: 'Arch' },
                { id: 'database', icon: Database, label: 'DB' },
                { id: 'env', icon: Key, label: 'Env' },
                { id: 'packages', icon: Package, label: 'Pkg' },
                { id: 'git', icon: GitBranch, label: 'Git' },
                { id: 'collab', icon: Users, label: 'Team' },
              ].map(tab => (
                <Tooltip key={tab.id}>
                  <TooltipTrigger asChild>
                    <button
                      className={`flex-1 h-10 flex items-center justify-center transition-colors ${
                        rightPanelTab === tab.id ? 'bg-[#0d1117] text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground'
                      }`}
                      onClick={() => setRightPanelTab(tab.id as any)}
                      data-testid={`tab-panel-${tab.id}`}
                    >
                      <tab.icon className="h-4 w-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>{tab.label}</TooltipContent>
                </Tooltip>
              ))}
            </div>

            <ScrollArea className="flex-1">
              {rightPanelTab === 'architecture' && (
                <div className="p-4">
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Infrastructure Map</h3>
                  <div className="space-y-4">
                    <ArchNode icon={<Globe className="h-5 w-5 text-blue-400" />} label="CDN / Load Balancer" color="blue" visible={logs.length > 2} />
                    <div className={`h-6 w-px bg-gradient-to-b from-blue-500/50 to-purple-500/50 mx-auto transition-opacity ${logs.length > 3 ? 'opacity-100' : 'opacity-0'}`} />
                    <ArchNode icon={<Server className="h-6 w-6 text-purple-400" />} label="Kubernetes Cluster" color="purple" badge={`${infrastructure?.minInstances || 2}-${infrastructure?.maxInstances || 20}`} visible={logs.length > 5} large />
                    <div className={`h-6 w-px bg-gradient-to-b from-purple-500/50 to-orange-500/50 mx-auto transition-opacity ${logs.length > 6 ? 'opacity-100' : 'opacity-0'}`} />
                    <div className={`flex justify-center gap-4 transition-all ${logs.length > 8 ? 'opacity-100' : 'opacity-0'}`}>
                      <ArchNode icon={<Database className="h-4 w-4 text-orange-400" />} label="PostgreSQL" color="orange" visible={true} small />
                      <ArchNode icon={<Zap className="h-4 w-4 text-red-400" />} label="Redis" color="red" visible={true} small />
                    </div>
                  </div>
                  {isComplete && (
                    <div className="mt-6 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                      <div className="flex items-center gap-2 text-green-400 text-xs font-medium mb-2">
                        <Check className="h-4 w-4" /> Ready to Deploy
                      </div>
                      <p className="text-[10px] text-muted-foreground">All infrastructure configured.</p>
                    </div>
                  )}
                </div>
              )}

              {rightPanelTab === 'database' && (
                <div className="p-4">
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Database</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
                      <div className="flex items-center gap-2 mb-2">
                        <Database className="h-4 w-4 text-orange-400" />
                        <span className="text-sm font-medium">PostgreSQL</span>
                        <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-400 border-green-500/20 ml-auto">Connected</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground space-y-1">
                        <div className="flex justify-between"><span>Host:</span><span className="font-mono">db.example.com</span></div>
                        <div className="flex justify-between"><span>Database:</span><span className="font-mono">{project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '_') || 'app'}</span></div>
                        <div className="flex justify-between"><span>Tables:</span><span>12</span></div>
                      </div>
                    </div>
                    <div className="p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
                      <div className="flex items-center gap-2 mb-2">
                        <Zap className="h-4 w-4 text-red-400" />
                        <span className="text-sm font-medium">Redis</span>
                        <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-400 border-green-500/20 ml-auto">Connected</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground space-y-1">
                        <div className="flex justify-between"><span>Host:</span><span className="font-mono">redis.example.com</span></div>
                        <div className="flex justify-between"><span>Keys:</span><span>1,234</span></div>
                        <div className="flex justify-between"><span>Memory:</span><span>128 MB</span></div>
                      </div>
                    </div>
                    <Button variant="outline" size="sm" className="w-full border-[#30363d]" data-testid="button-open-console">
                      <Terminal className="h-3.5 w-3.5 mr-2" /> Open Console
                    </Button>
                  </div>
                </div>
              )}

              {rightPanelTab === 'env' && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Environment</h3>
                    <div className="flex items-center gap-2">
                      <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setShowSecrets(!showSecrets)} data-testid="button-toggle-secrets">
                        {showSecrets ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </Button>
                      <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-add-env">
                        <Plus className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {envVariables.map((env, i) => (
                      <div key={i} className="p-2 bg-[#161b22] rounded border border-[#30363d] text-xs" data-testid={`env-${env.key}`}>
                        <div className="flex items-center gap-2 mb-1">
                          {env.isSecret ? <Lock className="h-3 w-3 text-yellow-400" /> : <Key className="h-3 w-3 text-muted-foreground" />}
                          <span className="font-mono font-medium">{env.key}</span>
                        </div>
                        <div className="font-mono text-muted-foreground pl-5">
                          {env.isSecret && !showSecrets ? '••••••••••••' : env.value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {rightPanelTab === 'packages' && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Packages</h3>
                    <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-add-package">
                      <Plus className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <div className="space-y-1">
                    {packages.map((pkg, i) => (
                      <div key={i} className="flex items-center justify-between p-2 hover:bg-[#161b22] rounded text-xs" data-testid={`package-${pkg.name}`}>
                        <div className="flex items-center gap-2">
                          <Package className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="font-mono">{pkg.name}</span>
                        </div>
                        <span className="text-muted-foreground font-mono">{pkg.version}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {rightPanelTab === 'git' && (
                <div className="p-4">
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Version Control</h3>
                  <div className="flex items-center gap-2 p-2 bg-[#161b22] rounded border border-[#30363d] mb-4">
                    <GitBranch className="h-4 w-4 text-green-400" />
                    <span className="text-sm font-medium">main</span>
                    <Badge variant="outline" className="text-[10px] ml-auto">default</Badge>
                  </div>
                  <h4 className="text-xs text-muted-foreground mb-2">Recent Commits</h4>
                  <div className="space-y-2">
                    {gitHistory.map((commit, i) => (
                      <div key={i} className="p-2 hover:bg-[#161b22] rounded text-xs" data-testid={`commit-${commit.hash}`}>
                        <div className="flex items-center gap-2 mb-1">
                          <GitCommit className="h-3 w-3 text-primary" />
                          <span className="font-mono text-primary">{commit.hash}</span>
                          <span className="text-muted-foreground ml-auto">{commit.time}</span>
                        </div>
                        <p className="text-foreground pl-5">{commit.message}</p>
                        <p className="text-muted-foreground pl-5">{commit.author}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {rightPanelTab === 'collab' && (
                <div className="p-4">
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Collaborators</h3>
                  <div className="space-y-2">
                    {collaborators.map((user, i) => (
                      <div key={i} className="flex items-center gap-3 p-2 hover:bg-[#161b22] rounded" data-testid={`collaborator-${user.name}`}>
                        <div className={`h-8 w-8 rounded-full ${user.color} flex items-center justify-center text-xs font-bold text-white`}>
                          {user.avatar}
                        </div>
                        <div className="flex-1">
                          <div className="text-sm font-medium flex items-center gap-2">
                            {user.name}
                            <span className={`h-2 w-2 rounded-full ${user.status === 'online' ? 'bg-green-500' : 'bg-gray-500'}`} />
                          </div>
                          {user.cursor && <div className="text-xs text-muted-foreground font-mono">{user.cursor}</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                  <Button variant="outline" size="sm" className="w-full mt-4 border-[#30363d]" data-testid="button-invite">
                    <Users className="h-3.5 w-3.5 mr-2" /> Invite
                  </Button>
                </div>
              )}
            </ScrollArea>
          </aside>
        )}
      </div>

      <Dialog open={newItemDialog.open} onOpenChange={(open) => setNewItemDialog({...newItemDialog, open})}>
        <DialogContent className="bg-[#161b22] border-[#30363d]">
          <DialogHeader>
            <DialogTitle>Create New {newItemDialog.type === 'file' ? 'File' : 'Folder'}</DialogTitle>
            <DialogDescription>Enter a name for the new {newItemDialog.type}.</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              value={newItemName}
              onChange={(e) => setNewItemName(e.target.value)}
              placeholder={newItemDialog.type === 'file' ? 'filename.ts' : 'folder-name'}
              className="bg-[#0d1117] border-[#30363d]"
              data-testid="input-new-item-name"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setNewItemDialog({...newItemDialog, open: false})} className="border-[#30363d]">Cancel</Button>
            <Button onClick={() => { toast({ title: `${newItemDialog.type} created` }); setNewItemDialog({...newItemDialog, open: false}); setNewItemName(''); }}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function FileTreeNode({ 
  node, 
  level, 
  activeFile, 
  expandedFolders, 
  onFileClick,
  getFileIcon,
  unsavedChanges
}: { 
  node: FileNode; 
  level: number;
  activeFile: string | null;
  expandedFolders: Set<string>;
  onFileClick: (file: FileNode) => void;
  getFileIcon: (name: string) => React.ReactNode;
  unsavedChanges: Set<string>;
}) {
  const isExpanded = expandedFolders.has(node.path);
  const isActive = node.path === activeFile;
  const hasChanges = unsavedChanges.has(node.path);

  return (
    <ContextMenu>
      <ContextMenuTrigger>
        <div>
          <div
            className={`flex items-center gap-1.5 py-1 px-2 rounded cursor-pointer text-xs group ${
              isActive 
                ? 'bg-primary/20 text-primary' 
                : 'text-[#c9d1d9] hover:bg-[#161b22]'
            }`}
            style={{ paddingLeft: `${level * 12 + 8}px` }}
            onClick={() => onFileClick(node)}
            data-testid={`file-${node.name}`}
          >
            {node.type === 'folder' ? (
              <>
                {isExpanded ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
                <Folder className={`h-3.5 w-3.5 shrink-0 ${isExpanded ? 'text-blue-400' : 'text-[#8b949e]'}`} />
              </>
            ) : (
              <>
                <span className="w-3 shrink-0" />
                {getFileIcon(node.name)}
              </>
            )}
            <span className={`truncate ${hasChanges ? 'italic' : ''}`}>{node.name}</span>
            {hasChanges && <span className="h-1.5 w-1.5 rounded-full bg-primary shrink-0 ml-auto" />}
          </div>
          {node.type === 'folder' && isExpanded && node.children && (
            <div>
              {node.children.map(child => (
                <FileTreeNode
                  key={child.path}
                  node={child}
                  level={level + 1}
                  activeFile={activeFile}
                  expandedFolders={expandedFolders}
                  onFileClick={onFileClick}
                  getFileIcon={getFileIcon}
                  unsavedChanges={unsavedChanges}
                />
              ))}
            </div>
          )}
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent className="w-48">
        {node.type === 'folder' && (
          <>
            <ContextMenuItem><FilePlus className="h-4 w-4 mr-2" /> New File</ContextMenuItem>
            <ContextMenuItem><FolderPlus className="h-4 w-4 mr-2" /> New Folder</ContextMenuItem>
            <ContextMenuSeparator />
          </>
        )}
        <ContextMenuItem><Edit3 className="h-4 w-4 mr-2" /> Rename</ContextMenuItem>
        <ContextMenuItem><Copy className="h-4 w-4 mr-2" /> Duplicate</ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem className="text-red-400"><Trash2 className="h-4 w-4 mr-2" /> Delete</ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}

function ArchNode({ 
  icon, 
  label, 
  color, 
  badge, 
  visible, 
  large, 
  small 
}: { 
  icon: React.ReactNode;
  label: string;
  color: string;
  badge?: string;
  visible: boolean;
  large?: boolean;
  small?: boolean;
}) {
  const sizeClasses = large ? 'h-14 w-14' : small ? 'h-9 w-9' : 'h-11 w-11';

  return (
    <div className={`flex flex-col items-center transition-all duration-500 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
      <div className={`${sizeClasses} rounded-xl bg-${color}-500/20 border border-${color}-500/50 flex items-center justify-center relative`}
           style={{ backgroundColor: `rgba(var(--${color}-500), 0.2)` }}>
        {badge && (
          <div className="absolute -top-1.5 -right-1.5 bg-green-500 text-black text-[8px] font-bold px-1.5 rounded-full">
            {badge}
          </div>
        )}
        {icon}
      </div>
      <span className={`${small ? 'text-[9px]' : 'text-[10px]'} font-mono mt-1.5 text-muted-foreground`}>{label}</span>
    </div>
  );
}