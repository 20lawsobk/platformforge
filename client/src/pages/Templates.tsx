import { useState } from "react";
import { useLocation } from "wouter";
import { 
  Search, 
  ArrowRight,
  Globe,
  Server,
  Database,
  Brain,
  Gamepad2,
  Smartphone,
  Boxes,
  Zap,
  Flame,
  Code2,
  FileCode,
  Terminal,
  Cpu,
  Layers,
  Box
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import Layout from "@/components/Layout";

interface Template {
  id: string;
  name: string;
  description: string;
  icon: typeof Globe;
  iconColor: string;
  iconBg: string;
  languages: string[];
  category: 'web' | 'api' | 'data' | 'games' | 'mobile';
}

const templates: Template[] = [
  {
    id: "react-vite",
    name: "React + Vite",
    description: "Modern React app with Vite bundler for lightning-fast HMR and optimized builds",
    icon: Zap,
    iconColor: "text-cyan-400",
    iconBg: "bg-cyan-400/10",
    languages: ["React", "TypeScript", "Vite"],
    category: "web"
  },
  {
    id: "nextjs",
    name: "Next.js",
    description: "Full-stack React framework with server-side rendering and API routes",
    icon: Boxes,
    iconColor: "text-white",
    iconBg: "bg-white/10",
    languages: ["React", "TypeScript", "Next.js"],
    category: "web"
  },
  {
    id: "express",
    name: "Express.js",
    description: "Fast, unopinionated Node.js web framework for building APIs",
    icon: Server,
    iconColor: "text-green-400",
    iconBg: "bg-green-400/10",
    languages: ["Node.js", "JavaScript"],
    category: "api"
  },
  {
    id: "fastapi",
    name: "FastAPI",
    description: "Modern, high-performance Python API framework with automatic docs",
    icon: Flame,
    iconColor: "text-emerald-400",
    iconBg: "bg-emerald-400/10",
    languages: ["Python", "FastAPI"],
    category: "api"
  },
  {
    id: "django",
    name: "Django",
    description: "High-level Python web framework that encourages rapid development",
    icon: Layers,
    iconColor: "text-green-500",
    iconBg: "bg-green-500/10",
    languages: ["Python", "Django"],
    category: "web"
  },
  {
    id: "flask",
    name: "Flask",
    description: "Lightweight Python microframework for building web applications",
    icon: Terminal,
    iconColor: "text-gray-400",
    iconBg: "bg-gray-400/10",
    languages: ["Python", "Flask"],
    category: "api"
  },
  {
    id: "go-http",
    name: "Go HTTP Server",
    description: "Lightweight and efficient HTTP server built with Go's standard library",
    icon: Box,
    iconColor: "text-cyan-500",
    iconBg: "bg-cyan-500/10",
    languages: ["Go"],
    category: "api"
  },
  {
    id: "rust-actix",
    name: "Rust Actix Web",
    description: "Blazingly fast Rust web framework for building reliable services",
    icon: Cpu,
    iconColor: "text-orange-400",
    iconBg: "bg-orange-400/10",
    languages: ["Rust", "Actix"],
    category: "api"
  },
  {
    id: "tensorflow",
    name: "TensorFlow",
    description: "Machine learning platform for building and training neural networks",
    icon: Brain,
    iconColor: "text-orange-500",
    iconBg: "bg-orange-500/10",
    languages: ["Python", "TensorFlow"],
    category: "data"
  },
  {
    id: "pytorch",
    name: "PyTorch",
    description: "Flexible deep learning framework for research and production",
    icon: Database,
    iconColor: "text-red-400",
    iconBg: "bg-red-400/10",
    languages: ["Python", "PyTorch"],
    category: "data"
  },
  {
    id: "phaser",
    name: "Phaser 3",
    description: "Fast HTML5 game framework for building 2D browser games",
    icon: Gamepad2,
    iconColor: "text-purple-400",
    iconBg: "bg-purple-400/10",
    languages: ["JavaScript", "Phaser"],
    category: "games"
  },
  {
    id: "react-native",
    name: "React Native",
    description: "Build native mobile apps for iOS and Android with React",
    icon: Smartphone,
    iconColor: "text-blue-400",
    iconBg: "bg-blue-400/10",
    languages: ["React Native", "TypeScript"],
    category: "mobile"
  },
  {
    id: "vue",
    name: "Vue.js",
    description: "Progressive JavaScript framework for building user interfaces",
    icon: Code2,
    iconColor: "text-emerald-400",
    iconBg: "bg-emerald-400/10",
    languages: ["Vue", "TypeScript"],
    category: "web"
  },
  {
    id: "svelte",
    name: "SvelteKit",
    description: "Cybernetically enhanced web apps with Svelte's compiler approach",
    icon: FileCode,
    iconColor: "text-orange-500",
    iconBg: "bg-orange-500/10",
    languages: ["Svelte", "TypeScript"],
    category: "web"
  },
  {
    id: "flutter",
    name: "Flutter",
    description: "Google's UI toolkit for building natively compiled mobile apps",
    icon: Smartphone,
    iconColor: "text-sky-400",
    iconBg: "bg-sky-400/10",
    languages: ["Dart", "Flutter"],
    category: "mobile"
  },
  {
    id: "unity-webgl",
    name: "Unity WebGL",
    description: "Create immersive 3D games that run directly in browsers",
    icon: Gamepad2,
    iconColor: "text-gray-300",
    iconBg: "bg-gray-300/10",
    languages: ["C#", "Unity"],
    category: "games"
  }
];

const categories = [
  { id: "all", label: "All Templates", icon: Globe },
  { id: "web", label: "Web Applications", icon: Globe },
  { id: "api", label: "APIs & Backends", icon: Server },
  { id: "data", label: "Data & AI", icon: Brain },
  { id: "games", label: "Games", icon: Gamepad2 },
  { id: "mobile", label: "Mobile", icon: Smartphone },
];

export default function Templates() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [projectName, setProjectName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [, setLocation] = useLocation();

  const filteredTemplates = templates.filter((template) => {
    const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.languages.some(lang => lang.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = activeCategory === "all" || template.category === activeCategory;
    
    return matchesSearch && matchesCategory;
  });

  const handleUseTemplate = (template: Template) => {
    setSelectedTemplate(template);
    setProjectName(template.name.toLowerCase().replace(/[^a-z0-9]/g, '-') + "-project");
  };

  const handleCreateProject = async () => {
    if (!projectName.trim()) return;
    
    setIsCreating(true);
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setIsCreating(false);
    setSelectedTemplate(null);
    setLocation("/dashboard");
  };

  const handleCloseDialog = () => {
    if (!isCreating) {
      setSelectedTemplate(null);
      setProjectName("");
    }
  };

  return (
    <Layout>
      <div className="min-h-screen bg-background">
        <div className="container px-4 md:px-8 py-12">
          <div className="text-center mb-10">
            <h1 
              className="text-4xl md:text-5xl font-bold tracking-tight mb-4"
              data-testid="text-page-title"
            >
              Start a new project
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Choose a template to get started quickly with a pre-configured development environment
            </p>
          </div>

          <div className="max-w-md mx-auto mb-8">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 h-11 bg-secondary/30 border-white/10"
                data-testid="input-search-templates"
              />
            </div>
          </div>

          <Tabs value={activeCategory} onValueChange={setActiveCategory} className="mb-8">
            <TabsList className="w-full flex-wrap h-auto gap-1 bg-secondary/30 p-1">
              {categories.map((category) => {
                const Icon = category.icon;
                return (
                  <TabsTrigger
                    key={category.id}
                    value={category.id}
                    className="flex items-center gap-2 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
                    data-testid={`tab-category-${category.id}`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{category.label}</span>
                    <span className="sm:hidden">{category.label.split(' ')[0]}</span>
                  </TabsTrigger>
                );
              })}
            </TabsList>

            <TabsContent value={activeCategory} className="mt-6">
              {filteredTemplates.length === 0 ? (
                <div className="text-center py-16" data-testid="text-no-results">
                  <p className="text-muted-foreground text-lg">No templates found matching your search.</p>
                  <Button 
                    variant="ghost" 
                    className="mt-4"
                    onClick={() => { setSearchQuery(""); setActiveCategory("all"); }}
                    data-testid="button-clear-filters"
                  >
                    Clear filters
                  </Button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {filteredTemplates.map((template) => {
                    const Icon = template.icon;
                    return (
                      <Card
                        key={template.id}
                        className="bg-card/50 border-white/5 hover-elevate transition-all duration-300 flex flex-col"
                        data-testid={`card-template-${template.id}`}
                      >
                        <CardHeader className="pb-3">
                          <div className="flex items-start gap-3">
                            <div className={`h-10 w-10 rounded-lg ${template.iconBg} flex items-center justify-center flex-shrink-0`}>
                              <Icon className={`h-5 w-5 ${template.iconColor}`} />
                            </div>
                            <div className="min-w-0 flex-1">
                              <h3 className="font-semibold text-base leading-tight" data-testid={`text-template-name-${template.id}`}>
                                {template.name}
                              </h3>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="pb-3 flex-1">
                          <p className="text-sm text-muted-foreground line-clamp-2" data-testid={`text-template-description-${template.id}`}>
                            {template.description}
                          </p>
                          <div className="flex flex-wrap gap-1.5 mt-3">
                            {template.languages.map((lang) => (
                              <Badge 
                                key={lang} 
                                variant="outline" 
                                className="text-xs"
                                data-testid={`badge-language-${template.id}-${lang.toLowerCase().replace(/\s+/g, '-')}`}
                              >
                                {lang}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                        <CardFooter className="pt-0">
                          <Button
                            variant="secondary"
                            className="w-full"
                            onClick={() => handleUseTemplate(template)}
                            data-testid={`button-use-template-${template.id}`}
                          >
                            Use Template
                            <ArrowRight className="ml-2 h-4 w-4" />
                          </Button>
                        </CardFooter>
                      </Card>
                    );
                  })}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <Dialog open={!!selectedTemplate} onOpenChange={(open) => !open && handleCloseDialog()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3" data-testid="text-dialog-title">
              {selectedTemplate && (
                <>
                  <div className={`h-8 w-8 rounded-lg ${selectedTemplate.iconBg} flex items-center justify-center`}>
                    <selectedTemplate.icon className={`h-4 w-4 ${selectedTemplate.iconColor}`} />
                  </div>
                  Create {selectedTemplate.name} Project
                </>
              )}
            </DialogTitle>
            <DialogDescription data-testid="text-dialog-description">
              Enter a name for your new project. This will be used as the project identifier.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              placeholder="my-awesome-project"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              disabled={isCreating}
              className="bg-secondary/30"
              data-testid="input-project-name"
              autoFocus
            />
          </div>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              variant="ghost"
              onClick={handleCloseDialog}
              disabled={isCreating}
              data-testid="button-dialog-cancel"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateProject}
              disabled={!projectName.trim() || isCreating}
              data-testid="button-dialog-create"
            >
              {isCreating ? (
                <>
                  <div className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                  Creating...
                </>
              ) : (
                <>
                  Create Project
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Layout>
  );
}
