import { useState, useEffect, useRef } from "react";
import { useLocation, useSearch } from "wouter";
import { 
  Terminal, 
  Play, 
  Settings, 
  Share2, 
  FileCode, 
  Folder, 
  ChevronRight, 
  ChevronDown, 
  Box, 
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  BrainCircuit,
  Database,
  Server,
  Globe,
  Shield
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";

// Mock File System Data - Enhanced for "Full Scale Platform"
const mockFileSystem = [
  {
    name: "infrastructure",
    type: "folder",
    isOpen: true,
    children: [
      { name: "main.tf", type: "file", active: false },
      { name: "variables.tf", type: "file" },
      { name: "k8s-cluster.yaml", type: "file" },
    ]
  },
  {
    name: "src",
    type: "folder",
    isOpen: true,
    children: [
      { name: "api", type: "folder", isOpen: false, children: [] },
      { name: "workers", type: "folder", isOpen: false, children: [] },
      { name: "server.ts", type: "file", active: true },
    ]
  },
  { name: "docker-compose.prod.yml", type: "file" },
  { name: "Dockerfile", type: "file" },
];

// AI Architect Logs - Simulating "What I do but for platforms"
const aiLogs = [
  { type: "ai", text: "Analyzing repository structure and dependencies..." },
  { type: "ai", text: "Detected high-throughput Node.js API pattern." },
  { type: "system", text: ">> Architecting microservices solution..." },
  { type: "action", text: "PROVISIONING: Managed PostgreSQL (v15) for persistent data" },
  { type: "action", text: "PROVISIONING: Redis Cluster for session caching & rate limiting" },
  { type: "action", text: "CONFIGURING: Auto-scaling group (min: 2, max: 20 instances)" },
  { type: "action", text: "SECURITY: Setting up VPC peering and private subnets" },
  { type: "info", text: "Generating Terraform configuration..." },
  { type: "cmd", text: "> terraform init && terraform apply -auto-approve" },
  { type: "success", text: "Infrastructure provisioned in us-east-1, eu-west-1" },
  { type: "ai", text: "Optimizing CDN rules for global content delivery..." },
  { type: "success", text: "Platform deployed. Health checks passing." },
];

export default function Builder() {
  const search = useSearch();
  const params = new URLSearchParams(search);
  const source = params.get("source") || "Untitled Project";
  
  const [isBuilding, setIsBuilding] = useState(false);
  const [logs, setLogs] = useState<typeof aiLogs>([]);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Simulate AI Build Process
  useEffect(() => {
    if (source) {
      setLogs([]);
      let currentStep = 0;
      setIsBuilding(true);

      const interval = setInterval(() => {
        if (currentStep < aiLogs.length) {
          setLogs(prev => [...prev, aiLogs[currentStep]]);
          currentStep++;
        } else {
          setIsBuilding(false);
          clearInterval(interval);
        }
      }, 1200); // Slower to let the user read the "AI Thinking"

      return () => clearInterval(interval);
    }
  }, [source]);

  // Auto-scroll terminal
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      {/* Builder Header */}
      <header className="h-14 border-b border-border bg-card/50 flex items-center justify-between px-4">
        <div className="flex items-center gap-4">
          <div className="font-mono font-bold text-lg tracking-tight flex items-center gap-2">
            <BrainCircuit className="h-5 w-5 text-primary" />
            Platform<span className="text-primary">Architect</span>
          </div>
          <div className="h-6 w-px bg-border" />
          <div className="flex items-center gap-2 text-sm">
             <Box className="h-4 w-4 text-muted-foreground" />
             <span className="font-medium">{source}</span>
             <Badge variant="outline" className="ml-2 bg-primary/10 text-primary border-primary/20">
                {isBuilding ? "Architecting..." : "Live"}
             </Badge>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm"><Share2 className="h-4 w-4 mr-2" /> Share</Button>
          <Button variant="outline" size="sm"><Settings className="h-4 w-4 mr-2" /> Config</Button>
          <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white font-bold">
            <Play className="h-4 w-4 mr-2 fill-current" /> View Live
          </Button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 border-r border-border bg-card/20 flex flex-col">
          <div className="p-3 text-xs font-bold text-muted-foreground uppercase tracking-wider">Generated Infrastructure</div>
          <ScrollArea className="flex-1">
            <div className="px-2 space-y-1">
              {mockFileSystem.map((item, idx) => (
                <FileSystemItem key={idx} item={item} />
              ))}
            </div>
          </ScrollArea>
        </aside>

        {/* Editor Area */}
        <div className="flex-1 flex flex-col min-w-0 bg-background">
          {/* Editor Tabs */}
          <div className="flex h-10 border-b border-border bg-card/20">
            <div className="px-4 flex items-center gap-2 bg-background border-r border-border text-sm font-medium text-primary border-t-2 border-t-primary">
              <FileCode className="h-4 w-4" />
              server.ts
            </div>
            <div className="px-4 flex items-center gap-2 text-muted-foreground text-sm border-r border-border/50 hover:bg-card/40 cursor-pointer transition-colors">
              main.tf
            </div>
          </div>

          {/* Mock Editor Content - Showing Backend Code */}
          <div className="flex-1 p-4 font-mono text-sm overflow-auto text-muted-foreground leading-relaxed">
            <div><span className="text-gray-500">// AI GENERATED: High-availability server entry point</span></div>
            <div><span className="text-purple-400">import</span> <span className="text-blue-400">{`{ Cluster }`}</span> <span className="text-purple-400">from</span> <span className="text-green-400">"ioredis"</span>;</div>
            <div><span className="text-purple-400">import</span> <span className="text-blue-400">{`{ Pool }`}</span> <span className="text-purple-400">from</span> <span className="text-green-400">"pg"</span>;</div>
            <br />
            <div><span className="text-gray-500">// Initialize connection pool with read replicas</span></div>
            <div><span className="text-purple-400">const</span> db = <span className="text-purple-400">new</span> Pool({`{`}</div>
            <div>&nbsp;&nbsp;host: <span className="text-green-400">process.env.DB_PRIMARY_HOST</span>,</div>
            <div>&nbsp;&nbsp;max: <span className="text-orange-400">20</span>,</div>
            <div>&nbsp;&nbsp;idleTimeoutMillis: <span className="text-orange-400">30000</span></div>
            <div>{`}`});</div>
            <br />
            <div><span className="text-gray-500">// Configure distributed caching layer</span></div>
            <div><span className="text-purple-400">const</span> cache = <span className="text-purple-400">new</span> Cluster([<span className="text-green-400">process.env.REDIS_CLUSTER_URL</span>]);</div>
            <br />
            <div><span className="text-purple-400">export const</span> handler = <span className="text-blue-400">async</span> (req) ={`>`} {`{`}</div>
            <div>&nbsp;&nbsp;<span className="text-gray-500">// Automatic request deduplication</span></div>
            <div>&nbsp;&nbsp;<span className="text-purple-400">const</span> cached = <span className="text-blue-400">await</span> cache.get(req.id);</div>
            <div>&nbsp;&nbsp;<span className="text-purple-400">if</span> (cached) <span className="text-purple-400">return</span> JSON.parse(cached);</div>
            <div>{`}`}</div>
          </div>

          {/* Terminal Panel - The "AI Brain" */}
          <div className="h-64 border-t border-border bg-black/40 flex flex-col">
             <div className="flex items-center justify-between px-4 py-2 bg-card/20 border-b border-border/50">
                <div className="flex items-center gap-2 text-xs font-bold uppercase text-primary animate-pulse">
                  <BrainCircuit className="h-3 w-3" /> AI Architect Agent
                </div>
                <div className="flex gap-2">
                   <div className="h-2 w-2 rounded-full bg-red-500/50" />
                   <div className="h-2 w-2 rounded-full bg-yellow-500/50" />
                   <div className="h-2 w-2 rounded-full bg-green-500/50" />
                </div>
             </div>
             <ScrollArea className="flex-1 p-4 font-mono text-xs">
                <div className="space-y-1.5">
                  {logs.map((log, i) => (
                    <div key={i} className="flex gap-3 items-baseline">
                      <span className="text-muted-foreground select-none w-6 text-right opacity-30 text-[10px]">{i + 1}</span>
                      
                      {log.type === "ai" && (
                        <span className="text-purple-400 font-bold flex items-center gap-2">
                          <BrainCircuit className="h-3 w-3" /> {log.text}
                        </span>
                      )}
                      {log.type === "system" && <span className="text-blue-300 italic">{log.text}</span>}
                      {log.type === "action" && (
                        <span className="text-orange-300 flex items-center gap-2">
                          <Server className="h-3 w-3" /> {log.text}
                        </span>
                      )}
                      {log.type === "cmd" && <span className="text-muted-foreground">{log.text}</span>}
                      {log.type === "success" && <span className="text-green-400 font-bold">{log.text}</span>}
                    </div>
                  ))}
                  {isBuilding && (
                    <div className="flex gap-2 items-center text-primary animate-pulse mt-2 pl-9">
                      <span className="h-2 w-2 bg-primary rounded-full animate-bounce" />
                      <span className="h-2 w-2 bg-primary rounded-full animate-bounce delay-75" />
                      <span className="h-2 w-2 bg-primary rounded-full animate-bounce delay-150" />
                    </div>
                  )}
                  <div ref={terminalEndRef} />
                </div>
             </ScrollArea>
          </div>
        </div>

        {/* Preview Panel (Right Sidebar) */}
        <div className="w-80 border-l border-border bg-card/10 flex flex-col">
          <div className="p-3 border-b border-border text-xs font-bold text-muted-foreground uppercase tracking-wider flex justify-between items-center">
            <span>Infrastructure Map</span>
          </div>
          <div className="flex-1 flex items-center justify-center bg-black/20 p-6">
             {/* Visualizing the architecture being built */}
             <div className="relative w-full h-full flex flex-col items-center justify-center gap-6">
                {/* Load Balancer */}
                <div className={`flex flex-col items-center transition-all duration-500 ${logs.length > 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                   <div className="h-12 w-12 rounded-full bg-blue-500/20 border border-blue-500/50 flex items-center justify-center">
                      <Globe className="h-6 w-6 text-blue-400" />
                   </div>
                   <span className="text-[10px] font-mono mt-2 text-blue-300">Global CDN</span>
                </div>

                {/* Connecting Line */}
                <div className={`h-8 w-px bg-gradient-to-b from-blue-500/50 to-purple-500/50 transition-all duration-500 delay-200 ${logs.length > 3 ? 'opacity-100' : 'opacity-0'}`} />

                {/* Compute Cluster */}
                <div className={`flex flex-col items-center transition-all duration-500 delay-300 ${logs.length > 5 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                   <div className="h-16 w-16 rounded-xl bg-purple-500/20 border border-purple-500/50 flex items-center justify-center relative">
                      <div className="absolute -top-2 -right-2 bg-green-500 text-black text-[9px] font-bold px-1.5 rounded-full">x20</div>
                      <Server className="h-8 w-8 text-purple-400" />
                   </div>
                   <span className="text-[10px] font-mono mt-2 text-purple-300">K8s Cluster</span>
                </div>

                {/* Connecting Line */}
                <div className={`h-8 w-px bg-gradient-to-b from-purple-500/50 to-orange-500/50 transition-all duration-500 delay-500 ${logs.length > 6 ? 'opacity-100' : 'opacity-0'}`} />

                {/* Database Layer */}
                <div className={`flex gap-4 transition-all duration-500 delay-700 ${logs.length > 8 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                   <div className="flex flex-col items-center">
                      <div className="h-10 w-10 rounded-lg bg-orange-500/20 border border-orange-500/50 flex items-center justify-center">
                         <Database className="h-5 w-5 text-orange-400" />
                      </div>
                      <span className="text-[10px] font-mono mt-2 text-orange-300">Postgres</span>
                   </div>
                   <div className="flex flex-col items-center">
                      <div className="h-10 w-10 rounded-lg bg-red-500/20 border border-red-500/50 flex items-center justify-center">
                         <Shield className="h-5 w-5 text-red-400" />
                      </div>
                      <span className="text-[10px] font-mono mt-2 text-red-300">Redis</span>
                   </div>
                </div>

             </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function FileSystemItem({ item, level = 0 }: { item: any, level?: number }) {
  return (
    <div className="select-none">
      <div 
        className={`
          flex items-center gap-1.5 py-1 px-2 rounded-sm cursor-pointer text-sm
          ${item.active ? 'bg-primary/20 text-primary font-medium' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'}
        `}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
      >
        {item.type === 'folder' ? (
           <>
             {item.isOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
             <Folder className={`h-3.5 w-3.5 ${item.isOpen ? 'fill-blue-400/20 text-blue-400' : ''}`} />
           </>
        ) : (
           <FileCode className="h-3.5 w-3.5" />
        )}
        <span className="truncate">{item.name}</span>
      </div>
      {item.type === 'folder' && item.isOpen && item.children && (
        <div>
          {item.children.map((child: any, idx: number) => (
            <FileSystemItem key={idx} item={child} level={level + 1} />
          ))}
        </div>
      )}
    </div>
  )
}