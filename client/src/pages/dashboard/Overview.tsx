import { useState } from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { 
  Cpu, 
  Database, 
  Globe, 
  Clock, 
  GitBranch,
  ArrowUpRight, 
  CheckCircle2,
  Terminal,
  Play,
  Plus,
  Folder,
  MoreVertical,
  Trash2,
  Edit,
  Box,
  Code,
  Cloud,
  Shield,
  HardDrive,
  Eye,
  Github,
  Rocket
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
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
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Tooltip as TooltipWrapper,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import DashboardLayout from '@/components/DashboardLayout';
import { useQuery } from '@tanstack/react-query';
import { Link, useLocation } from 'wouter';
import type { Project } from '@shared/schema';

const metricsData = [
  { time: '00:00', reqs: 400, cpu: 24 },
  { time: '04:00', reqs: 300, cpu: 18 },
  { time: '08:00', reqs: 2000, cpu: 65 },
  { time: '12:00', reqs: 4500, cpu: 88 },
  { time: '16:00', reqs: 3800, cpu: 75 },
  { time: '20:00', reqs: 1200, cpu: 45 },
  { time: '23:59', reqs: 800, cpu: 30 },
];

export default function Overview() {
  const [, setLocation] = useLocation();
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [newProjectName, setNewProjectName] = useState('');

  const { data: projects = [], isLoading } = useQuery<Project[]>({
    queryKey: ['/api/user/projects'],
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'analyzing':
      case 'generating': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'failed': return 'bg-red-500/10 text-red-400 border-red-500/20';
      default: return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'complete': return 'Ready';
      case 'analyzing': return 'Analyzing';
      case 'generating': return 'Building';
      case 'failed': return 'Failed';
      case 'pending': return 'Pending';
      default: return status;
    }
  };

  const getDeploymentStatusColor = (status: string) => {
    switch (status) {
      case 'complete': return 'bg-green-500';
      case 'analyzing':
      case 'generating': return 'bg-yellow-500';
      default: return 'bg-gray-400';
    }
  };

  const handleOpenRenameDialog = (project: Project) => {
    setSelectedProject(project);
    setNewProjectName(project.name);
    setRenameDialogOpen(true);
  };

  const handleOpenDeleteDialog = (project: Project) => {
    setSelectedProject(project);
    setDeleteDialogOpen(true);
  };

  const handleRename = () => {
    setRenameDialogOpen(false);
    setSelectedProject(null);
    setNewProjectName('');
  };

  const handleDelete = () => {
    setDeleteDialogOpen(false);
    setSelectedProject(null);
  };

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Your Projects</h2>
            <p className="text-muted-foreground">Manage your infrastructure projects and deployments</p>
          </div>
          <Link href="/">
            <Button className="bg-primary hover:bg-primary/90" data-testid="button-new-project">
              <Plus className="h-4 w-4 mr-2" /> New Project
            </Button>
          </Link>
        </div>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="bg-card/50 border-white/5 animate-pulse">
                <CardContent className="p-6">
                  <div className="h-6 bg-white/10 rounded mb-4 w-3/4" />
                  <div className="h-4 bg-white/10 rounded mb-2 w-1/2" />
                  <div className="h-4 bg-white/10 rounded w-1/3" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : projects.length === 0 ? (
          <Card className="bg-card/50 border-white/5 border-dashed" data-testid="empty-state">
            <CardContent className="flex flex-col items-center justify-center py-20">
              <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                <Rocket className="h-10 w-10 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Get started with your first project</h3>
              <p className="text-muted-foreground text-sm mb-8 text-center max-w-md">
                Create production-ready infrastructure in minutes. Start from a template or import your existing code from GitHub.
              </p>
              <div className="flex flex-wrap gap-3 justify-center">
                <Link href="/templates">
                  <Button data-testid="button-create-from-template">
                    <Box className="h-4 w-4 mr-2" /> Create from Template
                  </Button>
                </Link>
                <Button variant="outline" data-testid="button-import-github">
                  <Github className="h-4 w-4 mr-2" /> Import from GitHub
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((project: Project) => (
              <Card 
                key={project.id} 
                className="bg-card/50 border-white/5 hover:border-primary/30 transition-colors group"
                data-testid={`card-project-${project.id}`}
              >
                <CardContent className="p-5">
                  <div className="flex items-start justify-between gap-2 mb-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <Box className="h-5 w-5 text-primary" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="font-semibold truncate" data-testid={`text-project-name-${project.id}`}>
                          {project.name}
                        </h3>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <GitBranch className="h-3 w-3" />
                          <span>main</span>
                        </div>
                      </div>
                    </div>
                    
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" data-testid={`button-project-menu-${project.id}`}>
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-48">
                        <DropdownMenuItem 
                          onClick={() => setLocation(`/ide/${project.id}`)}
                          disabled={project.status !== 'complete'}
                          data-testid={`menu-open-ide-${project.id}`}
                        >
                          <Code className="h-4 w-4 mr-2" /> Open in IDE
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => setLocation(`/builder?id=${project.id}`)}
                          data-testid={`menu-view-details-${project.id}`}
                        >
                          <Eye className="h-4 w-4 mr-2" /> View Details
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => setLocation('/dashboard/deployments')}
                          data-testid={`menu-deploy-${project.id}`}
                        >
                          <Cloud className="h-4 w-4 mr-2" /> Deploy
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => setLocation('/dashboard/security')}
                          data-testid={`menu-security-${project.id}`}
                        >
                          <Shield className="h-4 w-4 mr-2" /> Run Security Scan
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => setLocation('/dashboard/storage')}
                          data-testid={`menu-storage-${project.id}`}
                        >
                          <HardDrive className="h-4 w-4 mr-2" /> View Storage
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem 
                          onClick={() => handleOpenRenameDialog(project)}
                          data-testid={`menu-rename-${project.id}`}
                        >
                          <Edit className="h-4 w-4 mr-2" /> Rename
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="text-red-400 focus:text-red-400"
                          onClick={() => handleOpenDeleteDialog(project)}
                          data-testid={`menu-delete-${project.id}`}
                        >
                          <Trash2 className="h-4 w-4 mr-2" /> Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  <div className="flex items-center gap-2 mb-3">
                    <Badge variant="outline" className={getStatusColor(project.status)} data-testid={`badge-status-${project.id}`}>
                      {project.status === 'complete' && <CheckCircle2 className="h-3 w-3 mr-1" />}
                      {(project.status === 'analyzing' || project.status === 'generating') && (
                        <div className="h-2 w-2 mr-1 bg-current rounded-full animate-pulse" />
                      )}
                      {getStatusLabel(project.status)}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {project.createdAt ? new Date(project.createdAt).toLocaleDateString() : 'Recently'}
                    </span>
                  </div>

                  <div className="flex items-center gap-4 text-xs text-muted-foreground mb-4" data-testid={`stats-${project.id}`}>
                    <div className="flex items-center gap-1" data-testid={`stat-files-${project.id}`}>
                      <Folder className="h-3 w-3" />
                      <span>12 files</span>
                    </div>
                    <div className="flex items-center gap-1" data-testid={`stat-runtime-${project.id}`}>
                      <Clock className="h-3 w-3" />
                      <span>2m ago</span>
                    </div>
                    <div className="flex items-center gap-1" data-testid={`stat-deployment-${project.id}`}>
                      <div className="relative">
                        <Cloud className="h-3 w-3" />
                        <div className={`absolute -top-0.5 -right-0.5 h-1.5 w-1.5 rounded-full ${getDeploymentStatusColor(project.status)}`} />
                      </div>
                      <span>{project.status === 'complete' ? 'Live' : 'Pending'}</span>
                    </div>
                  </div>

                  {project.sourceUrl && (
                    <p className="text-xs text-muted-foreground mb-4 truncate">
                      <Globe className="h-3 w-3 inline mr-1" />
                      {project.sourceUrl.replace('https://github.com/', '')}
                    </p>
                  )}

                  <div className="flex gap-2">
                    {project.status === 'complete' ? (
                      <Link href={`/ide/${project.id}`} className="flex-1">
                        <Button 
                          variant="default"
                          size="sm" 
                          className="w-full"
                          data-testid={`button-open-ide-${project.id}`}
                        >
                          <Code className="h-3 w-3 mr-1" /> Open in IDE
                        </Button>
                      </Link>
                    ) : (
                      <TooltipWrapper>
                        <TooltipTrigger asChild>
                          <span className="flex-1">
                            <Button 
                              variant="outline"
                              size="sm" 
                              className="w-full border-white/10"
                              disabled
                              data-testid={`button-open-ide-disabled-${project.id}`}
                            >
                              <Code className="h-3 w-3 mr-1" /> Open in IDE
                            </Button>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Project not ready</p>
                        </TooltipContent>
                      </TooltipWrapper>
                    )}
                    {project.status === 'complete' && (
                      <Button 
                        size="sm" 
                        className="bg-green-600 hover:bg-green-700"
                        data-testid={`button-deploy-${project.id}`}
                      >
                        <Play className="h-3 w-3 mr-1 fill-current" /> Deploy
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Projects</CardTitle>
              <Folder className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-total-projects">{projects.length}</div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                <span className="text-green-400 flex items-center mr-1"><ArrowUpRight className="h-3 w-3" /> Active</span>
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Deployments</CardTitle>
              <Globe className="h-4 w-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-deployments">{projects.filter((p: Project) => p.status === 'complete').length}</div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                Ready to deploy
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Build Status</CardTitle>
              <Cpu className="h-4 w-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-build-status">
                {projects.filter((p: Project) => p.status === 'analyzing' || p.status === 'generating').length > 0 ? 'Active' : 'Idle'}
              </div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                {projects.filter((p: Project) => p.status === 'analyzing' || p.status === 'generating').length} in progress
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Success Rate</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-success-rate">
                {projects.length > 0 
                  ? Math.round((projects.filter((p: Project) => p.status === 'complete').length / projects.length) * 100) 
                  : 0}%
              </div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                Infrastructure generation
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <Card className="col-span-2 bg-card/50 border-white/5">
            <CardHeader>
              <CardTitle>Activity Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metricsData}>
                    <defs>
                      <linearGradient id="colorReqs" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis 
                      dataKey="time" 
                      stroke="hsl(var(--muted-foreground))" 
                      fontSize={12} 
                      tickLine={false} 
                      axisLine={false} 
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))" 
                      fontSize={12} 
                      tickLine={false} 
                      axisLine={false}
                      tickFormatter={(value) => `${value}`}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
                      itemStyle={{ color: 'hsl(var(--foreground))' }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="reqs" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorReqs)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-white/5">
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[280px]">
                <div className="space-y-4">
                  {projects.slice(0, 5).map((project: Project, i: number) => (
                    <div key={i} className="flex gap-3">
                      <div className={`mt-1 h-2 w-2 rounded-full shrink-0 ${
                        project.status === 'complete' ? 'bg-green-400' :
                        project.status === 'failed' ? 'bg-red-400' :
                        project.status === 'analyzing' || project.status === 'generating' ? 'bg-yellow-400' :
                        'bg-blue-400'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium leading-none mb-1 truncate">{project.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {getStatusLabel(project.status)} - {project.createdAt ? new Date(project.createdAt).toLocaleDateString() : 'Recently'}
                        </p>
                      </div>
                    </div>
                  ))}
                  {projects.length === 0 && (
                    <p className="text-sm text-muted-foreground text-center py-8">No recent activity</p>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-card/50 border-white/5 overflow-hidden">
          <div className="flex flex-col md:flex-row">
            <div className="p-6 flex-1 border-b md:border-b-0 md:border-r border-white/5">
              <h3 className="text-sm font-medium text-muted-foreground mb-4">Quick Start</h3>
              <div className="flex items-start gap-4">
                <div className="h-10 w-10 rounded bg-primary/10 flex items-center justify-center text-primary shrink-0">
                  <Terminal className="h-5 w-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <span className="font-bold">Get started in minutes</span>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    Paste a GitHub URL or upload your code to generate production infrastructure.
                  </p>
                  <div className="flex items-center gap-4 flex-wrap text-xs text-muted-foreground">
                    <span className="flex items-center gap-1"><Database className="h-3 w-3" /> PostgreSQL</span>
                    <span className="flex items-center gap-1"><Globe className="h-3 w-3" /> Kubernetes</span>
                    <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> Auto-scaling</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="p-6 w-full md:w-80 bg-white/2">
              <h3 className="text-sm font-medium text-muted-foreground mb-4">Actions</h3>
              <div className="space-y-2">
                <Link href="/">
                  <Button variant="outline" className="w-full justify-start border-white/10 hover:bg-white/5">
                    <Plus className="h-4 w-4 mr-2" /> New Project
                  </Button>
                </Link>
                <Link href="/templates">
                  <Button variant="outline" className="w-full justify-start border-white/10 hover:bg-white/5">
                    <Database className="h-4 w-4 mr-2" /> Browse Templates
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
        <DialogContent data-testid="dialog-rename">
          <DialogHeader>
            <DialogTitle>Rename Project</DialogTitle>
            <DialogDescription>
              Enter a new name for your project.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="Project name"
              data-testid="input-project-name"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameDialogOpen(false)} data-testid="button-cancel-rename">
              Cancel
            </Button>
            <Button onClick={handleRename} data-testid="button-confirm-rename">
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent data-testid="dialog-delete">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{selectedProject?.name}"? This action cannot be undone and will permanently remove all associated infrastructure and deployments.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-delete">Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleDelete} 
              className="bg-red-600 hover:bg-red-700"
              data-testid="button-confirm-delete"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </DashboardLayout>
  );
}
