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
  AlertCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

// Mock File System Data
const mockFileSystem = [
  {
    name: "src",
    type: "folder",
    isOpen: true,
    children: [
      { name: "index.ts", type: "file", active: true },
      { name: "routes.ts", type: "file" },
      { name: "config.json", type: "file" },
    ]
  },
  { name: "package.json", type: "file" },
  { name: "README.md", type: "file" },
  { name: ".env", type: "file" },
];

// Mock Terminal Logs
const mockLogs = [
  { type: "info", text: "Initializing build environment..." },
  { type: "info", text: "Cloning repository..." },
  { type: "success", text: "Repository cloned successfully." },
  { type: "info", text: "Detecting language..." },
  { type: "info", text: "Detected TypeScript / Node.js environment." },
  { type: "info", text: "Installing dependencies..." },
  { type: "cmd", text: "> npm install" },
  { type: "info", text: "added 842 packages in 3s" },
  { type: "cmd", text: "> npm run build" },
  { type: "success", text: "Build completed in 1.2s" },
  { type: "info", text: "Deploying to Edge Network..." },
  { type: "success", text: "Deployment Active: https://fancy-platform-83x.pb.dev" },
];

export default function Builder() {
  const search = useSearch();
  const params = new URLSearchParams(search);
  const source = params.get("source") || "Untitled Project";
  
  const [isBuilding, setIsBuilding] = useState(false);
  const [logs, setLogs] = useState<typeof mockLogs>([]);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Simulate build process
  useEffect(() => {
    if (source) {
      setLogs([]);
      let currentStep = 0;
      setIsBuilding(true);

      const interval = setInterval(() => {
        if (currentStep < mockLogs.length) {
          setLogs(prev => [...prev, mockLogs[currentStep]]);
          currentStep++;
        } else {
          setIsBuilding(false);
          clearInterval(interval);
        }
      }, 800);

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
          <div className="font-mono font-bold text-lg tracking-tight">
            Platform<span className="text-primary">Builder</span>
          </div>
          <div className="h-6 w-px bg-border" />
          <div className="flex items-center gap-2 text-sm">
             <Box className="h-4 w-4 text-muted-foreground" />
             <span className="font-medium">{source}</span>
             <Badge variant="outline" className="ml-2 bg-primary/10 text-primary border-primary/20">
                {isBuilding ? "Building..." : "Ready"}
             </Badge>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm"><Share2 className="h-4 w-4 mr-2" /> Share</Button>
          <Button variant="outline" size="sm"><Settings className="h-4 w-4 mr-2" /> Settings</Button>
          <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white font-bold">
            <Play className="h-4 w-4 mr-2 fill-current" /> Deploy
          </Button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 border-r border-border bg-card/20 flex flex-col">
          <div className="p-3 text-xs font-bold text-muted-foreground uppercase tracking-wider">Explorer</div>
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
              index.ts
            </div>
            <div className="px-4 flex items-center gap-2 text-muted-foreground text-sm border-r border-border/50 hover:bg-card/40 cursor-pointer transition-colors">
              config.json
            </div>
          </div>

          {/* Mock Editor Content */}
          <div className="flex-1 p-4 font-mono text-sm overflow-auto text-muted-foreground leading-relaxed">
            <div><span className="text-purple-400">import</span> <span className="text-blue-400">{`{ createServer }`}</span> <span className="text-purple-400">from</span> <span className="text-green-400">"http"</span>;</div>
            <div><span className="text-purple-400">import</span> <span className="text-blue-400">{`{ platform }`}</span> <span className="text-purple-400">from</span> <span className="text-green-400">"@platform/core"</span>;</div>
            <br />
            <div><span className="text-gray-500">// Initialize the platform builder instance</span></div>
            <div><span className="text-purple-400">const</span> app = platform.create({`{`}</div>
            <div>&nbsp;&nbsp;region: <span className="text-green-400">"global"</span>,</div>
            <div>&nbsp;&nbsp;scaling: <span className="text-blue-400">true</span>,</div>
            <div>{`}`});</div>
            <br />
            <div>app.on(<span className="text-green-400">"request"</span>, (req, res) ={`>`} {`{`}</div>
            <div>&nbsp;&nbsp;<span className="text-gray-500">// Your logic here handles 10M+ requests automatically</span></div>
            <div>&nbsp;&nbsp;res.json({`{`} message: <span className="text-green-400">"Hello from the edge!"</span> {`}`});</div>
            <div>{`}`});</div>
            <br />
            <div>app.listen(<span className="text-orange-400">3000</span>);</div>
          </div>

          {/* Terminal Panel */}
          <div className="h-64 border-t border-border bg-black/40 flex flex-col">
             <div className="flex items-center justify-between px-4 py-2 bg-card/20 border-b border-border/50">
                <div className="flex items-center gap-2 text-xs font-bold uppercase text-muted-foreground">
                  <Terminal className="h-3 w-3" /> Terminal Output
                </div>
                <div className="flex gap-2">
                   <div className="h-2 w-2 rounded-full bg-red-500/50" />
                   <div className="h-2 w-2 rounded-full bg-yellow-500/50" />
                   <div className="h-2 w-2 rounded-full bg-green-500/50" />
                </div>
             </div>
             <ScrollArea className="flex-1 p-4 font-mono text-xs">
                <div className="space-y-1">
                  {logs.map((log, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-muted-foreground select-none w-6 text-right opacity-50">{i + 1}</span>
                      {log.type === "cmd" && <span className="text-yellow-400 font-bold">{log.text}</span>}
                      {log.type === "info" && <span className="text-blue-300">{log.text}</span>}
                      {log.type === "success" && <span className="text-green-400">{log.text}</span>}
                      {log.type === "error" && <span className="text-red-400">{log.text}</span>}
                    </div>
                  ))}
                  {isBuilding && (
                    <div className="flex gap-2 items-center text-primary animate-pulse mt-2">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Processing...
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
            <span>Preview</span>
            <div className="flex gap-1">
              <div className="h-2 w-2 rounded-full bg-muted-foreground/30" />
              <div className="h-2 w-2 rounded-full bg-muted-foreground/30" />
            </div>
          </div>
          <div className="flex-1 flex items-center justify-center bg-white/5 p-4">
             <div className="w-full h-full bg-background border border-border rounded-lg shadow-2xl overflow-hidden flex flex-col">
                <div className="h-6 bg-muted/50 border-b border-border flex items-center px-2 gap-1">
                   <div className="h-1.5 w-1.5 rounded-full bg-red-400" />
                   <div className="h-1.5 w-1.5 rounded-full bg-yellow-400" />
                   <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                   <div className="ml-2 h-3 w-32 bg-muted rounded-sm" />
                </div>
                <div className="flex-1 flex items-center justify-center p-4 text-center">
                   {logs.length > 5 ? (
                     <div className="animate-in fade-in zoom-in duration-500">
                        <div className="text-2xl font-bold mb-2">Hello from the edge!</div>
                        <div className="text-xs text-muted-foreground">Served from region: <span className="text-primary">us-east-1</span></div>
                     </div>
                   ) : (
                     <div className="text-xs text-muted-foreground flex flex-col items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Waiting for deployment...
                     </div>
                   )}
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