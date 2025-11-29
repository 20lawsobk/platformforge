import { useState, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { ArrowRight, Github, Code2, Zap, Globe, Shield, Upload, FileCode, X, File, FolderUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { useMutation } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import bgImage from "@assets/generated_images/cybernetic_schematic_background.png";

const ACCEPTED_FILE_TYPES = [
  '.js', '.jsx', '.ts', '.tsx', '.py', '.go', '.rs', '.rb', '.java', '.c', '.cpp', '.h',
  '.css', '.scss', '.html', '.json', '.yaml', '.yml', '.md', '.txt', '.sh', '.bash',
  '.sql', '.graphql', '.vue', '.svelte', '.php', '.swift', '.kt', '.scala'
];

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function Home() {
  const [input, setInput] = useState("");
  const [activeTab, setActiveTab] = useState<'github' | 'upload'>('github');
  const [files, setFiles] = useState<File[]>([]);
  const [projectName, setProjectName] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const createProject = useMutation({
    mutationFn: async (sourceUrl: string) => {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: sourceUrl.split('/').pop() || 'New Project',
          sourceUrl,
          sourceType: sourceUrl.includes('github') ? 'github' : 'script',
          status: 'pending',
        }),
      });
      if (!res.ok) throw new Error('Failed to create project');
      return res.json();
    },
    onSuccess: (data) => {
      setLocation(`/builder?id=${data.id}`);
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to start infrastructure generation",
        variant: "destructive",
      });
    },
  });

  const uploadProject = useMutation({
    mutationFn: async ({ files, name }: { files: File[]; name: string }) => {
      const formData = new FormData();
      formData.append('name', name);
      files.forEach((file) => {
        formData.append('files', file);
      });
      
      const res = await fetch('/api/projects/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const error = await res.json().catch(() => ({ message: 'Failed to upload files' }));
        throw new Error(error.message || 'Failed to upload files');
      }
      return res.json();
    },
    onSuccess: (data) => {
      setLocation(`/builder?id=${data.id}`);
    },
    onError: (error: Error) => {
      toast({
        title: "Upload Error",
        description: error.message || "Failed to upload files",
        variant: "destructive",
      });
    },
  });

  const handleIgniteGithub = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      createProject.mutate(input);
    }
  };

  const handleIgniteUpload = (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload",
        variant: "destructive",
      });
      return;
    }
    if (!projectName.trim()) {
      toast({
        title: "Project name required",
        description: "Please enter a name for your project",
        variant: "destructive",
      });
      return;
    }
    uploadProject.mutate({ files, name: projectName });
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files).filter(file => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      return ACCEPTED_FILE_TYPES.includes(ext);
    });
    
    if (droppedFiles.length > 0) {
      setFiles(prev => [...prev, ...droppedFiles]);
    } else {
      toast({
        title: "Invalid file types",
        description: "Please upload code files (.js, .ts, .py, .go, etc.)",
        variant: "destructive",
      });
    }
  }, [toast]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...selectedFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const isLoading = createProject.isPending || uploadProject.isPending;

  return (
    <Layout>
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden bg-background">
        <div className="absolute inset-0 z-0">
          <div className="absolute inset-0 bg-grid-pattern opacity-[0.03] z-10 pointer-events-none" />
          <img 
            src={bgImage} 
            alt="Schematic Background" 
            className="w-full h-full object-cover opacity-20 mix-blend-screen"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-background/90 via-background/40 to-background" />
        </div>

        <div className="container relative z-10 px-4 text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-mono font-bold mb-8 shadow-[0_0_15px_-3px_var(--color-primary)]"
            >
              <Zap className="h-3 w-3" />
              <span>V2.0 NOW LIVE: MULTI-REGION DEPLOYMENT</span>
            </motion.div>
            
            <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
              AI-Architected <br className="hidden md:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-cyan-400 to-blue-500">Platforms.</span>
            </h1>
            
            <p className="text-lg md:text-xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
              Don't just build apps. Generate enterprise-grade infrastructure.
              From scripts to auto-scaling Kubernetes clusters in one click.
            </p>

            <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'github' | 'upload')} className="max-w-xl mx-auto">
              <TabsList className="grid w-full grid-cols-2 mb-4 bg-white/5 border border-white/10">
                <TabsTrigger 
                  value="github" 
                  data-testid="tab-github"
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary gap-2"
                >
                  <Github className="h-4 w-4" />
                  GitHub URL
                </TabsTrigger>
                <TabsTrigger 
                  value="upload" 
                  data-testid="tab-upload"
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary gap-2"
                >
                  <FolderUp className="h-4 w-4" />
                  Upload Files
                </TabsTrigger>
              </TabsList>

              <TabsContent value="github">
                <form onSubmit={handleIgniteGithub} className="flex flex-col sm:flex-row gap-4 p-2 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm shadow-2xl">
                  <div className="relative flex-1">
                    <Code2 className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input 
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="https://github.com/user/repo or ./script.js" 
                      className="pl-10 h-12 bg-transparent border-none focus-visible:ring-0 text-base font-mono placeholder:text-muted-foreground/50"
                      disabled={isLoading}
                      data-testid="input-github-url"
                    />
                  </div>
                  <Button 
                    type="submit" 
                    size="lg" 
                    className="h-12 px-8 bg-primary text-primary-foreground font-bold transition-all"
                    disabled={isLoading || !input.trim()}
                    data-testid="button-ignite"
                  >
                    {createProject.isPending ? (
                      <>Processing...</>
                    ) : (
                      <>Ignite <ArrowRight className="ml-2 h-4 w-4" /></>
                    )}
                  </Button>
                </form>
              </TabsContent>

              <TabsContent value="upload">
                <form onSubmit={handleIgniteUpload} className="flex flex-col gap-4 p-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm shadow-2xl">
                  <div className="relative">
                    <FileCode className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                    <Input 
                      value={projectName}
                      onChange={(e) => setProjectName(e.target.value)}
                      placeholder="Project Name" 
                      className="pl-10 h-12 bg-transparent border border-white/10 focus-visible:ring-1 focus-visible:ring-primary text-base font-mono placeholder:text-muted-foreground/50"
                      disabled={isLoading}
                      data-testid="input-project-name"
                    />
                  </div>

                  <div
                    data-testid="dropzone-files"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`
                      relative flex flex-col items-center justify-center gap-3 p-8 
                      border-2 border-dashed rounded-lg cursor-pointer transition-all duration-200
                      ${isDragging 
                        ? 'border-primary bg-primary/10 shadow-[0_0_20px_-5px_var(--color-primary)]' 
                        : 'border-white/20 hover:border-primary/50 hover:bg-white/5'
                      }
                    `}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept={ACCEPTED_FILE_TYPES.join(',')}
                      onChange={handleFileSelect}
                      className="hidden"
                      disabled={isLoading}
                    />
                    <div className={`p-3 rounded-full ${isDragging ? 'bg-primary/20' : 'bg-white/5'} transition-colors`}>
                      <Upload className={`h-6 w-6 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-foreground">
                        {isDragging ? 'Drop files here' : 'Drag & drop files or click to browse'}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Supports .js, .ts, .py, .go, .rs, .rb, .java, and more
                      </p>
                    </div>
                  </div>

                  {files.length > 0 && (
                    <div className="flex flex-col gap-2 max-h-48 overflow-y-auto">
                      {files.map((file, index) => (
                        <div 
                          key={`${file.name}-${index}`}
                          data-testid={`file-item-${index}`}
                          className="flex items-center justify-between gap-3 p-2 rounded-md bg-white/5 border border-white/10"
                        >
                          <div className="flex items-center gap-2 min-w-0 flex-1">
                            <File className="h-4 w-4 text-primary flex-shrink-0" />
                            <span className="text-sm font-mono truncate">{file.name}</span>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <span className="text-xs text-muted-foreground">{formatFileSize(file.size)}</span>
                            <Button
                              type="button"
                              size="icon"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                removeFile(index);
                              }}
                              className="h-6 w-6"
                              disabled={isLoading}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <Button 
                    type="submit" 
                    size="lg" 
                    className="h-12 px-8 bg-primary text-primary-foreground font-bold transition-all w-full"
                    disabled={isLoading || files.length === 0 || !projectName.trim()}
                    data-testid="button-ignite"
                  >
                    {uploadProject.isPending ? (
                      <>Uploading...</>
                    ) : (
                      <>Ignite <ArrowRight className="ml-2 h-4 w-4" /></>
                    )}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>

            <div className="mt-10 flex items-center justify-center gap-8 text-sm text-muted-foreground/60 font-mono flex-wrap">
              <span className="flex items-center gap-2 hover:text-primary transition-colors cursor-default"><Github className="h-4 w-4" /> GitHub Supported</span>
              <span className="flex items-center gap-2 hover:text-primary transition-colors cursor-default"><Code2 className="h-4 w-4" /> Python / Node / Go</span>
              <span className="flex items-center gap-2 hover:text-primary transition-colors cursor-default"><Globe className="h-4 w-4" /> Edge Ready</span>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="py-24 bg-secondary/30 border-t border-white/5 relative overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern opacity-[0.02] pointer-events-none" />
        <div className="container px-4 md:px-8 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="p-6 bg-card/50 border-white/5 backdrop-blur-sm hover:border-primary/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <Globe className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2 font-sans">Global Edge Network</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Your platforms are automatically replicated across 35+ regions globally. sub-50ms latency for everyone.
              </p>
            </Card>

            <Card className="p-6 bg-card/50 border-white/5 backdrop-blur-sm hover:border-purple-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="h-12 w-12 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4 group-hover:bg-purple-500/20 transition-colors">
                <Zap className="h-6 w-6 text-purple-400" />
              </div>
              <h3 className="text-xl font-bold mb-2 font-sans">Instant Scaling</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                From 0 to 10 million requests. Our serverless infrastructure handles the load so you don't have to.
              </p>
            </Card>

            <Card className="p-6 bg-card/50 border-white/5 backdrop-blur-sm hover:border-green-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="h-12 w-12 rounded-lg bg-green-500/10 flex items-center justify-center mb-4 group-hover:bg-green-500/20 transition-colors">
                <Shield className="h-6 w-6 text-green-400" />
              </div>
              <h3 className="text-xl font-bold mb-2 font-sans">Enterprise Grade</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                SOC2 Type II compliant. Automated DDoS protection, WAF, and encrypted secrets management built-in.
              </p>
            </Card>
          </div>
        </div>
      </section>
    </Layout>
  );
}
