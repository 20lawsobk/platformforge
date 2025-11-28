import { useState, useEffect, useRef, useMemo } from "react";
import { useSearch, Link } from "wouter";
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
  Layers
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { useQuery } from "@tanstack/react-query";
import JSZip from "jszip";
import { saveAs } from "file-saver";

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  content?: string;
  language?: string;
  children?: FileNode[];
}

export default function Builder() {
  const search = useSearch();
  const params = new URLSearchParams(search);
  const projectId = params.get("id");
  const { toast } = useToast();
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [terminalMaximized, setTerminalMaximized] = useState(false);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [openTabs, setOpenTabs] = useState<string[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['/', '/infrastructure', '/src']));
  const [copied, setCopied] = useState(false);
  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState<string[]>([]);

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

    return [
      {
        name: 'infrastructure',
        path: '/infrastructure',
        type: 'folder',
        children: [
          { 
            name: 'main.tf', 
            path: '/infrastructure/main.tf', 
            type: 'file', 
            content: infrastructure.terraformConfig || '# Terraform configuration', 
            language: 'hcl' 
          },
          { 
            name: 'variables.tf', 
            path: '/infrastructure/variables.tf', 
            type: 'file', 
            content: `variable "db_password" {\n  description = "Database password"\n  type        = string\n  sensitive   = true\n}\n\nvariable "environment" {\n  description = "Deployment environment"\n  type        = string\n  default     = "production"\n}`, 
            language: 'hcl' 
          },
          { 
            name: 'outputs.tf', 
            path: '/infrastructure/outputs.tf', 
            type: 'file', 
            content: `output "cluster_endpoint" {\n  value = aws_eks_cluster.main.endpoint\n}\n\noutput "database_endpoint" {\n  value = aws_db_instance.default.endpoint\n}`, 
            language: 'hcl' 
          },
        ]
      },
      {
        name: 'kubernetes',
        path: '/kubernetes',
        type: 'folder',
        children: [
          { 
            name: 'deployment.yaml', 
            path: '/kubernetes/deployment.yaml', 
            type: 'file', 
            content: infrastructure.kubernetesConfig || '# Kubernetes manifests', 
            language: 'yaml' 
          },
          { 
            name: 'service.yaml', 
            path: '/kubernetes/service.yaml', 
            type: 'file', 
            content: `apiVersion: v1\nkind: Service\nmetadata:\n  name: ${project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'app'}-service\nspec:\n  selector:\n    app: main\n  ports:\n    - protocol: TCP\n      port: 80\n      targetPort: 3000\n  type: LoadBalancer`, 
            language: 'yaml' 
          },
          { 
            name: 'ingress.yaml', 
            path: '/kubernetes/ingress.yaml', 
            type: 'file', 
            content: `apiVersion: networking.k8s.io/v1\nkind: Ingress\nmetadata:\n  name: main-ingress\n  annotations:\n    kubernetes.io/ingress.class: nginx\n    cert-manager.io/cluster-issuer: letsencrypt-prod\nspec:\n  tls:\n    - hosts:\n        - api.example.com\n      secretName: tls-secret\n  rules:\n    - host: api.example.com\n      http:\n        paths:\n          - path: /\n            pathType: Prefix\n            backend:\n              service:\n                name: ${project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'app'}-service\n                port:\n                  number: 80`, 
            language: 'yaml' 
          },
        ]
      },
      {
        name: 'docker',
        path: '/docker',
        type: 'folder',
        children: [
          { 
            name: 'Dockerfile', 
            path: '/docker/Dockerfile', 
            type: 'file', 
            content: infrastructure.dockerfileContent || '# Dockerfile', 
            language: 'dockerfile' 
          },
          { 
            name: 'docker-compose.yml', 
            path: '/docker/docker-compose.yml', 
            type: 'file', 
            content: `version: '3.8'\n\nservices:\n  app:\n    build: .\n    ports:\n      - "3000:3000"\n    environment:\n      - NODE_ENV=production\n      - DATABASE_URL=\${DATABASE_URL}\n      - REDIS_URL=\${REDIS_URL}\n    depends_on:\n      - postgres\n      - redis\n    restart: unless-stopped\n\n  postgres:\n    image: postgres:15-alpine\n    environment:\n      POSTGRES_USER: app\n      POSTGRES_PASSWORD: \${DB_PASSWORD}\n      POSTGRES_DB: platform\n    volumes:\n      - postgres_data:/var/lib/postgresql/data\n    restart: unless-stopped\n\n  redis:\n    image: redis:7-alpine\n    command: redis-server --appendonly yes\n    volumes:\n      - redis_data:/data\n    restart: unless-stopped\n\nvolumes:\n  postgres_data:\n  redis_data:`, 
            language: 'yaml' 
          },
          { 
            name: '.dockerignore', 
            path: '/docker/.dockerignore', 
            type: 'file', 
            content: `node_modules\nnpm-debug.log\n.git\n.env\n.env.local\n*.md\n.DS_Store\ncoverage\n.nyc_output\ndist`, 
            language: 'plaintext' 
          },
        ]
      },
      {
        name: 'scripts',
        path: '/scripts',
        type: 'folder',
        children: [
          { 
            name: 'deploy.sh', 
            path: '/scripts/deploy.sh', 
            type: 'file', 
            content: `#!/bin/bash\nset -e\n\necho "🚀 Starting deployment..."\n\n# Build Docker image\necho "📦 Building Docker image..."\ndocker build -t ${project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'app'}:latest .\n\n# Push to registry\necho "📤 Pushing to container registry..."\ndocker push \${REGISTRY}/${project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'app'}:latest\n\n# Apply Kubernetes manifests\necho "☸️ Applying Kubernetes manifests..."\nkubectl apply -f kubernetes/\n\n# Wait for rollout\necho "⏳ Waiting for rollout..."\nkubectl rollout status deployment/${project?.name?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'app'}-app\n\necho "✅ Deployment complete!"`, 
            language: 'bash' 
          },
          { 
            name: 'setup-infra.sh', 
            path: '/scripts/setup-infra.sh', 
            type: 'file', 
            content: `#!/bin/bash\nset -e\n\necho "🏗️ Setting up infrastructure..."\n\ncd infrastructure\n\n# Initialize Terraform\necho "📋 Initializing Terraform..."\nterraform init\n\n# Plan changes\necho "📝 Planning infrastructure changes..."\nterraform plan -out=tfplan\n\n# Apply (with confirmation)\nread -p "Apply these changes? (y/n) " -n 1 -r\necho\nif [[ $REPLY =~ ^[Yy]$ ]]; then\n  terraform apply tfplan\n  echo "✅ Infrastructure provisioned!"\nfi`, 
            language: 'bash' 
          },
        ]
      },
      { 
        name: 'README.md', 
        path: '/README.md', 
        type: 'file', 
        content: `# ${project?.name || 'Project'} - Infrastructure\n\nGenerated by PlatformArchitect AI\n\n## Quick Start\n\n### Prerequisites\n- Terraform >= 1.0\n- kubectl configured\n- Docker\n- AWS CLI configured\n\n### Deploy Infrastructure\n\n\`\`\`bash\n# 1. Set up cloud resources\ncd infrastructure\nterraform init\nterraform apply\n\n# 2. Build and push Docker image\ndocker build -t your-registry/app:latest .\ndocker push your-registry/app:latest\n\n# 3. Deploy to Kubernetes\nkubectl apply -f kubernetes/\n\`\`\`\n\n## Architecture\n\n- **Compute**: AWS EKS (Kubernetes)\n- **Database**: PostgreSQL on RDS\n- **Cache**: Redis on ElastiCache\n- **Scaling**: HPA (2-20 replicas)\n\n## Configuration\n\nSee \`infrastructure/variables.tf\` for configurable options.`, 
        language: 'markdown' 
      },
    ];
  }, [infrastructure, project]);

  useEffect(() => {
    if (infrastructure && !activeFile) {
      const defaultFile = '/infrastructure/main.tf';
      setActiveFile(defaultFile);
      setOpenTabs([defaultFile]);
    }
  }, [infrastructure, activeFile]);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    if (project?.status === 'complete') {
      toast({
        title: "Infrastructure Generated",
        description: "Your production infrastructure is ready to deploy.",
      });
    }
  }, [project?.status, toast]);

  const findFile = (path: string, nodes: FileNode[] = fileSystem): FileNode | null => {
    for (const node of nodes) {
      if (node.path === path) return node;
      if (node.children) {
        const found = findFile(path, node.children);
        if (found) return found;
      }
    }
    return null;
  };

  const activeFileContent = activeFile ? findFile(activeFile)?.content : null;
  const activeFileName = activeFile?.split('/').pop() || '';

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
    const newTabs = openTabs.filter(t => t !== path);
    setOpenTabs(newTabs);
    if (activeFile === path) {
      setActiveFile(newTabs[newTabs.length - 1] || null);
    }
  };

  const handleCopyCode = () => {
    if (activeFileContent) {
      navigator.clipboard.writeText(activeFileContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleTerminalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (terminalInput.trim()) {
      setTerminalHistory([...terminalHistory, `$ ${terminalInput}`, 'Command simulation not available in preview mode.']);
      setTerminalInput('');
    }
  };

  const handleDownloadArtifacts = async () => {
    if (!infrastructure) return;

    const zip = new JSZip();
    
    const addFolder = (nodes: FileNode[], folder: JSZip) => {
      nodes.forEach(node => {
        if (node.type === 'file' && node.content) {
          folder.file(node.name, node.content);
        } else if (node.type === 'folder' && node.children) {
          const subFolder = folder.folder(node.name);
          if (subFolder) addFolder(node.children, subFolder);
        }
      });
    };

    addFolder(fileSystem, zip);

    const content = await zip.generateAsync({ type: "blob" });
    saveAs(content, `${project?.name || 'infrastructure'}-bundle.zip`);
    
    toast({
      title: "Artifacts Downloaded",
      description: "Complete infrastructure bundle saved.",
    });
  };

  const isBuilding = project?.status === 'analyzing' || project?.status === 'generating' || project?.status === 'pending';
  const isComplete = project?.status === 'complete';

  const getFileIcon = (name: string) => {
    if (name.endsWith('.tf')) return <FileCode className="h-4 w-4 text-purple-400" />;
    if (name.endsWith('.yaml') || name.endsWith('.yml')) return <FileJson className="h-4 w-4 text-yellow-400" />;
    if (name.endsWith('.md')) return <FileType className="h-4 w-4 text-blue-400" />;
    if (name.endsWith('.sh')) return <TerminalSquare className="h-4 w-4 text-green-400" />;
    if (name === 'Dockerfile' || name === '.dockerignore') return <Layers className="h-4 w-4 text-cyan-400" />;
    return <FileCode className="h-4 w-4 text-muted-foreground" />;
  };

  return (
    <div className="h-screen flex flex-col bg-[#0d1117] text-foreground overflow-hidden" data-testid="builder-container">
      <header className="h-12 border-b border-[#30363d] bg-[#161b22] flex items-center justify-between px-3">
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
              className={`text-[10px] ${isBuilding ? 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30' : 'bg-green-500/10 text-green-500 border-green-500/30'}`}
              data-testid="badge-status"
            >
              {isBuilding ? "Generating..." : isComplete ? "Ready" : project?.status}
            </Badge>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" className="h-8 text-xs" data-testid="button-branch">
            <GitBranch className="h-3.5 w-3.5 mr-1.5" /> main
          </Button>

          <div className="h-4 w-px bg-[#30363d]" />
          
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

          <Button variant="ghost" size="icon" className="h-8 w-8" data-testid="button-settings">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {sidebarOpen && (
          <aside className="w-60 border-r border-[#30363d] bg-[#0d1117] flex flex-col">
            <div className="h-10 flex items-center justify-between px-3 border-b border-[#30363d]">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Explorer</span>
              <div className="flex items-center gap-1">
                <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-new-file">
                  <Plus className="h-3.5 w-3.5" />
                </Button>
                <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-refresh-files">
                  <RefreshCw className="h-3.5 w-3.5" />
                </Button>
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
                  />
                ))}
              </div>
            </ScrollArea>
          </aside>
        )}

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex h-9 bg-[#161b22] border-b border-[#30363d] items-center">
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 rounded-none border-r border-[#30363d]"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              data-testid="button-toggle-sidebar"
            >
              {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
            </Button>

            <div className="flex-1 flex overflow-x-auto">
              {openTabs.map(tabPath => {
                const file = findFile(tabPath);
                const fileName = tabPath.split('/').pop() || '';
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
                    <span>{fileName}</span>
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

            {activeFile && (
              <div className="flex items-center gap-1 px-2 border-l border-[#30363d]">
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-7 w-7"
                  onClick={handleCopyCode}
                  data-testid="button-copy-code"
                >
                  {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
                </Button>
                <Button variant="ghost" size="icon" className="h-7 w-7" data-testid="button-external-link">
                  <ExternalLink className="h-3.5 w-3.5" />
                </Button>
              </div>
            )}
          </div>

          <div className={`flex-1 flex flex-col ${terminalMaximized ? 'hidden' : ''}`}>
            {activeFileContent ? (
              <ScrollArea className="flex-1 bg-[#0d1117]">
                <pre className="p-4 font-mono text-sm leading-relaxed">
                  <code className="text-[#c9d1d9]">
                    {activeFileContent.split('\n').map((line, i) => (
                      <div key={i} className="flex">
                        <span className="w-12 text-right pr-4 text-[#484f58] select-none text-xs">{i + 1}</span>
                        <span className="flex-1">{highlightSyntax(line, activeFileName)}</span>
                      </div>
                    ))}
                  </code>
                </pre>
              </ScrollArea>
            ) : (
              <div className="flex-1 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <FileCode className="h-12 w-12 mx-auto mb-4 opacity-20" />
                  <p className="text-sm">Select a file to view its contents</p>
                </div>
              </div>
            )}
          </div>

          <div className={`${terminalMaximized ? 'flex-1' : 'h-56'} border-t border-[#30363d] bg-[#0d1117] flex flex-col`}>
            <div className="flex items-center justify-between h-9 px-3 bg-[#161b22] border-b border-[#30363d]">
              <Tabs defaultValue="output" className="h-full">
                <TabsList className="h-full bg-transparent gap-4">
                  <TabsTrigger value="output" className="h-full px-0 text-xs data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none" data-testid="tab-output">
                    <BrainCircuit className="h-3.5 w-3.5 mr-1.5" /> AI Output
                  </TabsTrigger>
                  <TabsTrigger value="terminal" className="h-full px-0 text-xs data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none" data-testid="tab-terminal">
                    <Terminal className="h-3.5 w-3.5 mr-1.5" /> Terminal
                  </TabsTrigger>
                </TabsList>
              </Tabs>
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

            <ScrollArea className="flex-1 p-3">
              <div className="font-mono text-xs space-y-1">
                {logs.map((log: any, i: number) => (
                  <div key={i} className="flex gap-2" data-testid={`log-entry-${i}`}>
                    <span className="text-[#484f58] w-5 text-right shrink-0">{i + 1}</span>
                    <LogEntry log={log} />
                  </div>
                ))}
                {terminalHistory.map((line, i) => (
                  <div key={`hist-${i}`} className="flex gap-2">
                    <span className="text-[#484f58] w-5 text-right shrink-0">{logs.length + i + 1}</span>
                    <span className="text-[#8b949e]">{line}</span>
                  </div>
                ))}
                {isBuilding && (
                  <div className="flex gap-2 items-center text-primary animate-pulse pl-7">
                    <span className="h-1.5 w-1.5 bg-primary rounded-full animate-bounce" />
                    <span className="h-1.5 w-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <span className="h-1.5 w-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                )}
                <div ref={terminalEndRef} />
              </div>
            </ScrollArea>

            <form onSubmit={handleTerminalSubmit} className="px-3 pb-3">
              <div className="flex items-center gap-2 bg-[#161b22] rounded border border-[#30363d] px-3 py-1.5">
                <span className="text-primary text-xs">$</span>
                <input
                  type="text"
                  value={terminalInput}
                  onChange={(e) => setTerminalInput(e.target.value)}
                  placeholder="Enter command..."
                  className="flex-1 bg-transparent text-xs outline-none text-foreground placeholder:text-muted-foreground"
                  data-testid="input-terminal"
                />
              </div>
            </form>
          </div>
        </div>

        <aside className="w-72 border-l border-[#30363d] bg-[#0d1117] flex flex-col">
          <div className="h-9 flex items-center px-3 border-b border-[#30363d]">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Architecture</span>
          </div>

          <div className="flex-1 p-4">
            <div className="relative h-full flex flex-col items-center justify-center gap-4">
              <ArchNode 
                icon={<Globe className="h-5 w-5 text-blue-400" />}
                label="CDN / Edge"
                color="blue"
                visible={logs.length > 2}
              />
              
              <div className={`h-6 w-px bg-gradient-to-b from-blue-500/50 to-purple-500/50 transition-opacity duration-500 ${logs.length > 3 ? 'opacity-100' : 'opacity-0'}`} />
              
              <ArchNode 
                icon={<Server className="h-6 w-6 text-purple-400" />}
                label="K8s Cluster"
                color="purple"
                badge={`${infrastructure?.minInstances || 2}-${infrastructure?.maxInstances || 20}`}
                visible={logs.length > 5}
                large
              />
              
              <div className={`h-6 w-px bg-gradient-to-b from-purple-500/50 to-orange-500/50 transition-opacity duration-500 ${logs.length > 6 ? 'opacity-100' : 'opacity-0'}`} />
              
              <div className={`flex gap-3 transition-all duration-500 ${logs.length > 8 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                <ArchNode 
                  icon={<Database className="h-4 w-4 text-orange-400" />}
                  label={infrastructure?.requiresDatabase || "Postgres"}
                  color="orange"
                  visible={true}
                  small
                />
                <ArchNode 
                  icon={<Shield className="h-4 w-4 text-red-400" />}
                  label={infrastructure?.requiresCache || "Redis"}
                  color="red"
                  visible={true}
                  small
                />
              </div>
            </div>
          </div>

          {isComplete && (
            <div className="p-3 border-t border-[#30363d]">
              <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                <div className="flex items-center gap-2 text-green-500 text-xs font-medium mb-2">
                  <Check className="h-4 w-4" /> Ready to Deploy
                </div>
                <p className="text-[10px] text-muted-foreground mb-3">
                  Infrastructure configured for {infrastructure?.architecture || 'production'} deployment.
                </p>
                <Button size="sm" className="w-full h-7 text-xs bg-green-600 hover:bg-green-700" data-testid="button-deploy-now">
                  <Play className="h-3 w-3 mr-1.5 fill-current" /> Deploy Now
                </Button>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function FileTreeNode({ 
  node, 
  level, 
  activeFile, 
  expandedFolders, 
  onFileClick,
  getFileIcon
}: { 
  node: FileNode; 
  level: number;
  activeFile: string | null;
  expandedFolders: Set<string>;
  onFileClick: (file: FileNode) => void;
  getFileIcon: (name: string) => React.ReactNode;
}) {
  const isExpanded = expandedFolders.has(node.path);
  const isActive = node.path === activeFile;

  return (
    <div>
      <div
        className={`flex items-center gap-1.5 py-1 px-2 rounded cursor-pointer text-xs ${
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
            {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            <Folder className={`h-3.5 w-3.5 ${isExpanded ? 'text-blue-400' : 'text-[#8b949e]'}`} />
          </>
        ) : (
          <>
            <span className="w-3" />
            {getFileIcon(node.name)}
          </>
        )}
        <span className="truncate">{node.name}</span>
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
            />
          ))}
        </div>
      )}
    </div>
  );
}

function LogEntry({ log }: { log: any }) {
  const icons: Record<string, React.ReactNode> = {
    ai: <BrainCircuit className="h-3 w-3 text-purple-400" />,
    action: <Server className="h-3 w-3 text-orange-400" />,
    success: <Check className="h-3 w-3 text-green-500" />,
  };

  const colors: Record<string, string> = {
    ai: 'text-purple-400',
    system: 'text-[#58a6ff]',
    action: 'text-orange-400',
    cmd: 'text-[#8b949e]',
    info: 'text-[#58a6ff]',
    success: 'text-green-400',
    error: 'text-red-400',
  };

  return (
    <span className={`flex items-center gap-1.5 ${colors[log.logLevel] || 'text-[#c9d1d9]'}`}>
      {icons[log.logLevel]}
      {log.message}
    </span>
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
  const bgColor = `bg-${color}-500/20`;
  const borderColor = `border-${color}-500/50`;

  return (
    <div className={`flex flex-col items-center transition-all duration-500 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
      <div className={`${sizeClasses} rounded-xl ${bgColor} border ${borderColor} flex items-center justify-center relative`}>
        {badge && (
          <div className="absolute -top-1.5 -right-1.5 bg-green-500 text-black text-[8px] font-bold px-1.5 rounded-full">
            {badge}
          </div>
        )}
        {icon}
      </div>
      <span className={`${small ? 'text-[9px]' : 'text-[10px]'} font-mono mt-1.5 text-${color}-300`}>{label}</span>
    </div>
  );
}

function highlightSyntax(line: string, fileName: string): React.ReactNode {
  if (fileName.endsWith('.tf')) {
    return line
      .replace(/(resource|variable|output|provider|data|module|locals|terraform)\s/g, '<span class="text-purple-400">$1</span> ')
      .replace(/"([^"]+)"/g, '<span class="text-green-400">"$1"</span>')
      .replace(/(\{|\})/g, '<span class="text-yellow-400">$1</span>')
      .replace(/(=)/g, '<span class="text-cyan-400">$1</span>')
      .replace(/(#.*)/g, '<span class="text-[#6a737d]">$1</span>');
  }
  
  if (fileName.endsWith('.yaml') || fileName.endsWith('.yml')) {
    if (line.trim().startsWith('#')) {
      return <span className="text-[#6a737d]">{line}</span>;
    }
    const parts = line.split(':');
    if (parts.length > 1) {
      return (
        <>
          <span className="text-cyan-400">{parts[0]}</span>
          <span>:</span>
          <span className="text-green-400">{parts.slice(1).join(':')}</span>
        </>
      );
    }
  }

  if (fileName.endsWith('.sh')) {
    if (line.trim().startsWith('#')) {
      return <span className="text-[#6a737d]">{line}</span>;
    }
    return line
      .replace(/^(\s*)(echo|cd|docker|kubectl|terraform|read|if|then|fi|set)\b/g, '$1<span class="text-purple-400">$2</span>')
      .replace(/"([^"]+)"/g, '<span class="text-green-400">"$1"</span>');
  }

  return line;
}