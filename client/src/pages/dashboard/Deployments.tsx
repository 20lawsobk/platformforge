import { useState } from "react";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger,
  DialogFooter
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import DashboardLayout from "@/components/DashboardLayout";
import { 
  ExternalLink, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  Rocket, 
  Globe,
  Zap,
  ArrowRight,
  RefreshCw,
  Plus,
  Trash2,
  Server,
  Play
} from "lucide-react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import type { 
  DeploymentTarget, 
  Deployment, 
  DeploymentRun, 
  Project 
} from "@shared/schema";

const cloudProviders = [
  { id: "aws", name: "Amazon Web Services", icon: Server, description: "Deploy to AWS EKS with full infrastructure" },
  { id: "gcp", name: "Google Cloud Platform", icon: Server, description: "Deploy to GKE with managed services" },
  { id: "azure", name: "Microsoft Azure", icon: Server, description: "Deploy to AKS with enterprise features" },
  { id: "do", name: "DigitalOcean", icon: Server, description: "Simple Kubernetes on DOKS" },
];

const regions: Record<string, { value: string; label: string }[]> = {
  aws: [
    { value: "us-east-1", label: "US East (N. Virginia)" },
    { value: "us-west-2", label: "US West (Oregon)" },
    { value: "eu-west-1", label: "EU (Ireland)" },
    { value: "ap-southeast-1", label: "Asia Pacific (Singapore)" },
  ],
  gcp: [
    { value: "us-central1", label: "Iowa (us-central1)" },
    { value: "us-east1", label: "South Carolina (us-east1)" },
    { value: "europe-west1", label: "Belgium (europe-west1)" },
    { value: "asia-east1", label: "Taiwan (asia-east1)" },
  ],
  azure: [
    { value: "eastus", label: "East US" },
    { value: "westus2", label: "West US 2" },
    { value: "westeurope", label: "West Europe" },
    { value: "southeastasia", label: "Southeast Asia" },
  ],
  do: [
    { value: "nyc1", label: "New York 1" },
    { value: "sfo3", label: "San Francisco 3" },
    { value: "ams3", label: "Amsterdam 3" },
    { value: "sgp1", label: "Singapore 1" },
  ],
};

function getProviderName(providerId: string): string {
  return cloudProviders.find(p => p.id === providerId)?.name || providerId;
}

function getRegionLabel(provider: string, region: string | null): string {
  if (!region) return "N/A";
  const providerRegions = regions[provider];
  if (!providerRegions) return region;
  return providerRegions.find(r => r.value === region)?.label || region;
}

function formatTimeAgo(date: Date | string | null): string {
  if (!date) return "N/A";
  const d = new Date(date);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
}

export default function Deployments() {
  const { toast } = useToast();
  const [deployDialogOpen, setDeployDialogOpen] = useState(false);
  const [targetDialogOpen, setTargetDialogOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("aws");
  const [selectedRegion, setSelectedRegion] = useState("");
  const [selectedProject, setSelectedProject] = useState("");
  const [selectedTarget, setSelectedTarget] = useState("");
  const [deploymentName, setDeploymentName] = useState("");
  const [deployStep, setDeployStep] = useState(1);
  const [targetName, setTargetName] = useState("");
  const [expandedDeploymentId, setExpandedDeploymentId] = useState<string | null>(null);

  const { data: deploymentTargets = [], isLoading: targetsLoading } = useQuery<DeploymentTarget[]>({
    queryKey: ['/api/deployment-targets'],
  });

  const { data: projects = [], isLoading: projectsLoading } = useQuery<Project[]>({
    queryKey: ['/api/user/projects'],
  });

  const readyProjects = projects.filter((p) => p.status === 'complete');

  const allDeploymentsQueries = readyProjects.map(project => ({
    projectId: project.id,
    projectName: project.name,
  }));

  const { data: deploymentsData = [], isLoading: deploymentsLoading, refetch: refetchDeployments } = useQuery<
    { deployment: Deployment; projectName: string }[]
  >({
    queryKey: ['/api/all-deployments', readyProjects.map(p => p.id)],
    queryFn: async () => {
      if (readyProjects.length === 0) return [];
      
      const results = await Promise.all(
        readyProjects.map(async (project) => {
          try {
            const res = await fetch(`/api/projects/${project.id}/deployments`, { credentials: 'include' });
            if (!res.ok) return [];
            const deployments: Deployment[] = await res.json();
            return deployments.map(d => ({ deployment: d, projectName: project.name }));
          } catch {
            return [];
          }
        })
      );
      return results.flat();
    },
    enabled: readyProjects.length > 0,
  });

  const createTargetMutation = useMutation({
    mutationFn: async (data: { name: string; provider: string; region: string; isDefault?: boolean }) => {
      const res = await apiRequest('POST', '/api/deployment-targets', data);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/deployment-targets'] });
      setTargetDialogOpen(false);
      setTargetName("");
      setSelectedProvider("aws");
      setSelectedRegion("");
      toast({ title: "Target created", description: "Deployment target has been created successfully." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const deleteTargetMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest('DELETE', `/api/deployment-targets/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/deployment-targets'] });
      toast({ title: "Target deleted", description: "Deployment target has been deleted." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const createDeploymentMutation = useMutation({
    mutationFn: async (data: { projectId: string; name: string; targetId: string; deploymentType: string }) => {
      const res = await apiRequest('POST', `/api/projects/${data.projectId}/deployments`, {
        name: data.name,
        targetId: data.targetId,
        deploymentType: data.deploymentType,
        status: 'pending',
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/all-deployments'] });
      refetchDeployments();
      setDeployDialogOpen(false);
      resetDeployDialog();
      toast({ title: "Deployment created", description: "Deployment has been created. You can now trigger it." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const triggerDeploymentMutation = useMutation({
    mutationFn: async (deploymentId: string) => {
      const res = await apiRequest('POST', `/api/deployments/${deploymentId}/trigger`, {});
      return res.json();
    },
    onSuccess: (data, deploymentId) => {
      queryClient.invalidateQueries({ queryKey: ['/api/all-deployments'] });
      queryClient.invalidateQueries({ queryKey: ['/api/deployments', deploymentId, 'runs'] });
      refetchDeployments();
      toast({ title: "Deployment triggered", description: "Deployment run has been started." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const resetDeployDialog = () => {
    setDeployStep(1);
    setSelectedProject("");
    setSelectedTarget("");
    setDeploymentName("");
  };

  const handleCreateTarget = () => {
    if (!targetName || !selectedRegion) return;
    createTargetMutation.mutate({
      name: targetName,
      provider: selectedProvider,
      region: selectedRegion,
      isDefault: deploymentTargets.length === 0,
    });
  };

  const handleCreateDeployment = () => {
    if (!selectedProject || !selectedTarget || !deploymentName) return;
    const target = deploymentTargets.find(t => t.id === selectedTarget);
    createDeploymentMutation.mutate({
      projectId: selectedProject,
      name: deploymentName,
      targetId: selectedTarget,
      deploymentType: target?.provider || 'kubernetes',
    });
  };

  const activeDeployments = deploymentsData.filter(d => d.deployment.status === 'deployed').length;
  const totalDeploys = deploymentsData.length;

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Deployments</h2>
            <p className="text-muted-foreground">Deploy and manage your infrastructure across cloud providers</p>
          </div>
          
          <div className="flex items-center gap-2 flex-wrap">
            <Dialog open={targetDialogOpen} onOpenChange={setTargetDialogOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" data-testid="button-new-target">
                  <Plus className="h-4 w-4 mr-2" /> New Target
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[500px]">
                <DialogHeader>
                  <DialogTitle>Create Deployment Target</DialogTitle>
                  <DialogDescription>
                    Configure a cloud provider target for your deployments
                  </DialogDescription>
                </DialogHeader>

                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="target-name">Target Name</Label>
                    <Input 
                      id="target-name"
                      placeholder="e.g., Production AWS" 
                      value={targetName}
                      onChange={(e) => setTargetName(e.target.value)}
                      data-testid="input-target-name"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Cloud Provider</Label>
                    <RadioGroup value={selectedProvider} onValueChange={(v) => { setSelectedProvider(v); setSelectedRegion(""); }} className="grid grid-cols-2 gap-3">
                      {cloudProviders.map((provider) => (
                        <div key={provider.id}>
                          <RadioGroupItem value={provider.id} id={`target-${provider.id}`} className="peer sr-only" />
                          <Label
                            htmlFor={`target-${provider.id}`}
                            className="flex flex-col items-start gap-2 rounded-lg border-2 border-border bg-card p-3 hover:border-muted-foreground/50 peer-data-[state=checked]:border-primary cursor-pointer transition-colors"
                            data-testid={`target-provider-${provider.id}`}
                          >
                            <div className="flex items-center gap-2">
                              <provider.icon className="h-4 w-4" />
                              <span className="font-medium text-sm">{provider.name}</span>
                            </div>
                          </Label>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  <div className="space-y-2">
                    <Label>Region</Label>
                    <Select value={selectedRegion} onValueChange={setSelectedRegion}>
                      <SelectTrigger data-testid="select-target-region">
                        <SelectValue placeholder="Select region" />
                      </SelectTrigger>
                      <SelectContent>
                        {(regions[selectedProvider] || []).map((region) => (
                          <SelectItem key={region.value} value={region.value}>
                            {region.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <DialogFooter>
                  <Button 
                    onClick={handleCreateTarget} 
                    disabled={!targetName || !selectedRegion || createTargetMutation.isPending}
                    data-testid="button-create-target"
                  >
                    {createTargetMutation.isPending ? (
                      <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Creating...</>
                    ) : (
                      <><Plus className="h-4 w-4 mr-2" /> Create Target</>
                    )}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={deployDialogOpen} onOpenChange={(open) => { setDeployDialogOpen(open); if (!open) resetDeployDialog(); }}>
              <DialogTrigger asChild>
                <Button variant="default" disabled={readyProjects.length === 0 || deploymentTargets.length === 0} data-testid="button-new-deployment">
                  <Rocket className="h-4 w-4 mr-2" /> New Deployment
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[600px]">
                <DialogHeader>
                  <DialogTitle>Create Deployment</DialogTitle>
                  <DialogDescription>
                    Configure and deploy your project to a target
                  </DialogDescription>
                </DialogHeader>

                {deployStep === 1 && (
                  <div className="space-y-4 py-4">
                    <div className="space-y-2">
                      <Label>Deployment Name</Label>
                      <Input 
                        placeholder="e.g., Production Release v1.0" 
                        value={deploymentName}
                        onChange={(e) => setDeploymentName(e.target.value)}
                        data-testid="input-deployment-name"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Select Project</Label>
                      <Select value={selectedProject} onValueChange={setSelectedProject}>
                        <SelectTrigger data-testid="select-project">
                          <SelectValue placeholder="Choose a project to deploy" />
                        </SelectTrigger>
                        <SelectContent>
                          {readyProjects.map((project) => (
                            <SelectItem key={project.id} value={project.id}>
                              {project.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Select Target</Label>
                      <Select value={selectedTarget} onValueChange={setSelectedTarget}>
                        <SelectTrigger data-testid="select-target">
                          <SelectValue placeholder="Choose deployment target" />
                        </SelectTrigger>
                        <SelectContent>
                          {deploymentTargets.map((target) => (
                            <SelectItem key={target.id} value={target.id}>
                              {target.name} ({getProviderName(target.provider)})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex justify-end pt-4">
                      <Button 
                        onClick={() => setDeployStep(2)} 
                        disabled={!selectedProject || !selectedTarget || !deploymentName}
                        data-testid="button-next-step"
                      >
                        Continue <ArrowRight className="h-4 w-4 ml-2" />
                      </Button>
                    </div>
                  </div>
                )}

                {deployStep === 2 && (
                  <div className="space-y-4 py-4">
                    <Card>
                      <CardContent className="p-4">
                        <h4 className="text-sm font-medium mb-3">Deployment Summary</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">Name</span>
                            <span className="truncate max-w-[200px]">{deploymentName}</span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">Project</span>
                            <span>{readyProjects.find(p => p.id === selectedProject)?.name}</span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">Target</span>
                            <span>{deploymentTargets.find(t => t.id === selectedTarget)?.name}</span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span className="text-muted-foreground">Provider</span>
                            <span>{getProviderName(deploymentTargets.find(t => t.id === selectedTarget)?.provider || '')}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <div className="flex justify-between pt-4 gap-2">
                      <Button variant="outline" onClick={() => setDeployStep(1)} data-testid="button-back">
                        Back
                      </Button>
                      <Button 
                        onClick={handleCreateDeployment} 
                        disabled={createDeploymentMutation.isPending}
                        data-testid="button-create-deployment"
                      >
                        {createDeploymentMutation.isPending ? (
                          <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Creating...</>
                        ) : (
                          <><Rocket className="h-4 w-4 mr-2" /> Create Deployment</>
                        )}
                      </Button>
                    </div>
                  </div>
                )}
              </DialogContent>
            </Dialog>
          </div>
        </div>

        {deploymentTargets.length === 0 && !targetsLoading && (
          <Card>
            <CardContent className="p-6 text-center">
              <Server className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Deployment Targets</h3>
              <p className="text-muted-foreground mb-4">Create a deployment target to start deploying your projects.</p>
              <Button onClick={() => setTargetDialogOpen(true)} data-testid="button-create-first-target">
                <Plus className="h-4 w-4 mr-2" /> Create Your First Target
              </Button>
            </CardContent>
          </Card>
        )}

        {(targetsLoading || deploymentTargets.length > 0) && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-4">
              <CardTitle>Deployment Targets</CardTitle>
            </CardHeader>
            <CardContent>
              {targetsLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-16 w-full" />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {deploymentTargets.map((target) => (
                    <Card key={target.id} className="relative" data-testid={`card-target-${target.id}`}>
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                              <h4 className="font-medium truncate">{target.name}</h4>
                              {target.isDefault && (
                                <Badge variant="secondary" className="text-xs">Default</Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground">{getProviderName(target.provider)}</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              {getRegionLabel(target.provider, target.region)}
                            </p>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => deleteTargetMutation.mutate(target.id)}
                            disabled={deleteTargetMutation.isPending}
                            data-testid={`button-delete-target-${target.id}`}
                          >
                            <Trash2 className="h-4 w-4 text-muted-foreground" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Active Deployments</CardTitle>
              <Globe className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-active-deployments">
                {deploymentsLoading ? <Skeleton className="h-8 w-12" /> : activeDeployments}
              </div>
              <p className="text-xs text-muted-foreground">Running in production</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Deployments</CardTitle>
              <Rocket className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-total-deploys">
                {deploymentsLoading ? <Skeleton className="h-8 w-12" /> : totalDeploys}
              </div>
              <p className="text-xs text-muted-foreground">All time</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Targets</CardTitle>
              <Server className="h-4 w-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-targets-count">
                {targetsLoading ? <Skeleton className="h-8 w-12" /> : deploymentTargets.length}
              </div>
              <p className="text-xs text-muted-foreground">Cloud targets configured</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Success Rate</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-deploy-success-rate">
                {deploymentsLoading ? <Skeleton className="h-8 w-12" /> : totalDeploys > 0 ? `${Math.round((activeDeployments / totalDeploys) * 100)}%` : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">Deployed successfully</p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <CardTitle>Deployments</CardTitle>
              <Button variant="outline" size="sm" onClick={() => refetchDeployments()} data-testid="button-refresh-deployments">
                <RefreshCw className="h-4 w-4 mr-2" /> Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {deploymentsLoading || projectsLoading ? (
              <div className="p-4 space-y-4">
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            ) : deploymentsData.length === 0 ? (
              <div className="p-8 text-center">
                <Rocket className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No Deployments Yet</h3>
                <p className="text-muted-foreground mb-4">
                  {readyProjects.length === 0 
                    ? "Complete a project first, then create a deployment."
                    : deploymentTargets.length === 0
                    ? "Create a deployment target first."
                    : "Create your first deployment to get started."}
                </p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[300px]">Deployment</TableHead>
                    <TableHead>Project</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Target</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {deploymentsData.map(({ deployment, projectName }) => (
                    <DeploymentRow 
                      key={deployment.id} 
                      deployment={deployment} 
                      projectName={projectName}
                      targets={deploymentTargets}
                      onTrigger={() => triggerDeploymentMutation.mutate(deployment.id)}
                      isPending={triggerDeploymentMutation.isPending}
                      isExpanded={expandedDeploymentId === deployment.id}
                      onToggleExpand={() => setExpandedDeploymentId(
                        expandedDeploymentId === deployment.id ? null : deployment.id
                      )}
                    />
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}

function DeploymentRow({ 
  deployment, 
  projectName, 
  targets,
  onTrigger,
  isPending,
  isExpanded,
  onToggleExpand
}: { 
  deployment: Deployment; 
  projectName: string;
  targets: DeploymentTarget[];
  onTrigger: () => void;
  isPending: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const target = targets.find(t => t.id === deployment.targetId);

  const { data: runs = [], isLoading: runsLoading } = useQuery<DeploymentRun[]>({
    queryKey: ['/api/deployments', deployment.id, 'runs'],
    queryFn: async () => {
      const res = await fetch(`/api/deployments/${deployment.id}/runs`, { credentials: 'include' });
      if (!res.ok) return [];
      return res.json();
    },
    enabled: isExpanded,
    refetchInterval: isExpanded && deployment.status === 'deploying' ? 2000 : false,
  });

  const latestRun = runs[0];

  return (
    <>
      <TableRow data-testid={`row-deployment-${deployment.id}`}>
        <TableCell>
          <div className="flex flex-col gap-1">
            <div className="font-medium flex items-center gap-2 flex-wrap">
              <span className="font-mono text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                #{deployment.id.slice(0, 8)}
              </span>
              <span className="truncate max-w-[200px]">{deployment.name}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" /> {formatTimeAgo(deployment.createdAt)}
              </span>
              {deployment.url && (
                <a 
                  href={deployment.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-primary hover:underline"
                >
                  <ExternalLink className="h-3 w-3" /> View
                </a>
              )}
            </div>
          </div>
        </TableCell>
        <TableCell>
          <Badge variant="secondary" className="font-mono text-xs">{projectName}</Badge>
        </TableCell>
        <TableCell>
          <StatusBadge status={deployment.status} />
        </TableCell>
        <TableCell>
          {target ? (
            <div className="text-sm">
              <div>{target.name}</div>
              <div className="text-xs text-muted-foreground">{getProviderName(target.provider)}</div>
            </div>
          ) : (
            <span className="text-muted-foreground">No target</span>
          )}
        </TableCell>
        <TableCell className="text-right">
          <div className="flex items-center justify-end gap-1 flex-wrap">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={onToggleExpand}
              data-testid={`button-view-runs-${deployment.id}`}
            >
              {isExpanded ? 'Hide' : 'Runs'}
            </Button>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={onTrigger}
              disabled={isPending || deployment.status === 'deploying'}
              data-testid={`button-deploy-${deployment.id}`}
            >
              {deployment.status === 'deploying' ? (
                <><Loader2 className="h-4 w-4 mr-1 animate-spin" /> Deploying</>
              ) : (
                <><Play className="h-4 w-4 mr-1" /> Deploy</>
              )}
            </Button>
          </div>
        </TableCell>
      </TableRow>
      {isExpanded && (
        <TableRow>
          <TableCell colSpan={5} className="bg-muted/30 p-4">
            <div className="space-y-4">
              <h4 className="font-medium text-sm">Deployment Runs</h4>
              {runsLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : runs.length === 0 ? (
                <p className="text-sm text-muted-foreground">No runs yet. Click Deploy to start a deployment.</p>
              ) : (
                <div className="space-y-3">
                  {runs.slice(0, 5).map((run) => (
                    <RunCard key={run.id} run={run} />
                  ))}
                </div>
              )}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function RunCard({ run }: { run: DeploymentRun }) {
  const [showLogs, setShowLogs] = useState(false);

  return (
    <Card data-testid={`card-run-${run.id}`}>
      <CardContent className="p-3">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-3 flex-wrap">
            <StatusBadge status={run.status} />
            <span className="font-mono text-xs text-muted-foreground">{run.version}</span>
            <span className="text-xs text-muted-foreground">
              {formatTimeAgo(run.startedAt)}
            </span>
            {run.completedAt && (
              <span className="text-xs text-muted-foreground">
                Duration: {formatDuration(run.startedAt, run.completedAt)}
              </span>
            )}
          </div>
          {run.logs && (
            <Button variant="ghost" size="sm" onClick={() => setShowLogs(!showLogs)}>
              {showLogs ? 'Hide Logs' : 'Show Logs'}
            </Button>
          )}
        </div>
        {run.errorMessage && (
          <div className="mt-2 text-sm text-destructive">
            Error: {run.errorMessage}
          </div>
        )}
        {showLogs && run.logs && (
          <pre className="mt-3 p-3 bg-muted rounded-md text-xs font-mono overflow-x-auto max-h-48 overflow-y-auto">
            {run.logs}
          </pre>
        )}
      </CardContent>
    </Card>
  );
}

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case 'deployed':
    case 'success':
    case 'ready':
      return (
        <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20 gap-1">
          <CheckCircle2 className="h-3 w-3" /> {status === 'deployed' ? 'Deployed' : 'Success'}
        </Badge>
      );
    case 'failed':
    case 'error':
      return (
        <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 gap-1">
          <XCircle className="h-3 w-3" /> Failed
        </Badge>
      );
    case 'deploying':
    case 'running':
    case 'pending':
      return (
        <Badge variant="outline" className="bg-blue-500/10 text-blue-500 border-blue-500/20 gap-1">
          <Loader2 className="h-3 w-3 animate-spin" /> {status === 'deploying' ? 'Deploying' : status === 'running' ? 'Running' : 'Pending'}
        </Badge>
      );
    default:
      return (
        <Badge variant="outline" className="gap-1">
          {status}
        </Badge>
      );
  }
}

function formatDuration(start: Date | string, end: Date | string): string {
  const startDate = new Date(start);
  const endDate = new Date(end);
  const diffMs = endDate.getTime() - startDate.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  
  if (diffSecs < 60) return `${diffSecs}s`;
  const mins = Math.floor(diffSecs / 60);
  const secs = diffSecs % 60;
  return `${mins}m ${secs}s`;
}
