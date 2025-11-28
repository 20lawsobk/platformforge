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
  DialogTrigger 
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
import DashboardLayout from "@/components/DashboardLayout";
import { 
  GitCommit, 
  ExternalLink, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  Rocket, 
  Cloud, 
  Server,
  Globe,
  Shield,
  Database,
  Zap,
  ArrowRight,
  RefreshCw
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";

const mockDeployments = [
  { id: "dep_8f2a9c1", commit: "8f2a9c1", message: "Updated landing page copy", branch: "main", status: "ready", duration: "45s", age: "2m ago", creator: "alex_dev" },
  { id: "dep_7b1x2y3", commit: "7b1x2y3", message: "Fix navigation bug on mobile", branch: "main", status: "ready", duration: "52s", age: "4h ago", creator: "alex_dev" },
  { id: "dep_3c4v5b6", commit: "3c4v5b6", message: "Add analytics tracking", branch: "feature/analytics", status: "error", duration: "1m 12s", age: "1d ago", creator: "sarah_lead" },
  { id: "dep_9n8m7k6", commit: "9n8m7k6", message: "Initial project setup", branch: "main", status: "ready", duration: "2m 05s", age: "3d ago", creator: "alex_dev" },
];

const cloudProviders = [
  { id: "aws", name: "Amazon Web Services", icon: "🔶", description: "Deploy to AWS EKS with full infrastructure" },
  { id: "gcp", name: "Google Cloud Platform", icon: "🔵", description: "Deploy to GKE with managed services" },
  { id: "azure", name: "Microsoft Azure", icon: "🔷", description: "Deploy to AKS with enterprise features" },
  { id: "do", name: "DigitalOcean", icon: "💧", description: "Simple Kubernetes on DOKS" },
];

const regions = {
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

export default function Deployments() {
  const [deployDialogOpen, setDeployDialogOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState("aws");
  const [selectedRegion, setSelectedRegion] = useState("");
  const [deployStep, setDeployStep] = useState(1);
  const [deploying, setDeploying] = useState(false);

  const { data: projects = [] } = useQuery({
    queryKey: ['projects'],
    queryFn: async () => {
      const res = await fetch('/api/projects');
      if (!res.ok) throw new Error('Failed to fetch projects');
      return res.json();
    },
  });

  const readyProjects = projects.filter((p: any) => p.status === 'complete');

  const handleDeploy = async () => {
    setDeploying(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    setDeploying(false);
    setDeployDialogOpen(false);
    setDeployStep(1);
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Deployments</h2>
            <p className="text-muted-foreground">Deploy and manage your infrastructure across cloud providers</p>
          </div>
          
          <Dialog open={deployDialogOpen} onOpenChange={setDeployDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-green-600 hover:bg-green-700" disabled={readyProjects.length === 0} data-testid="button-new-deployment">
                <Rocket className="h-4 w-4 mr-2" /> New Deployment
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[600px] bg-[#161b22] border-[#30363d]">
              <DialogHeader>
                <DialogTitle>Deploy Infrastructure</DialogTitle>
                <DialogDescription>
                  Configure your deployment target and settings
                </DialogDescription>
              </DialogHeader>

              {deployStep === 1 && (
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label>Select Project</Label>
                    <Select>
                      <SelectTrigger className="bg-[#0d1117] border-[#30363d]" data-testid="select-project">
                        <SelectValue placeholder="Choose a project to deploy" />
                      </SelectTrigger>
                      <SelectContent>
                        {readyProjects.map((project: any) => (
                          <SelectItem key={project.id} value={project.id}>
                            {project.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Cloud Provider</Label>
                    <RadioGroup value={selectedProvider} onValueChange={setSelectedProvider} className="grid grid-cols-2 gap-3">
                      {cloudProviders.map((provider) => (
                        <div key={provider.id}>
                          <RadioGroupItem value={provider.id} id={provider.id} className="peer sr-only" />
                          <Label
                            htmlFor={provider.id}
                            className="flex flex-col items-start gap-2 rounded-lg border-2 border-[#30363d] bg-[#0d1117] p-4 hover:border-[#484f58] peer-data-[state=checked]:border-primary cursor-pointer transition-colors"
                            data-testid={`provider-${provider.id}`}
                          >
                            <div className="flex items-center gap-2">
                              <span className="text-xl">{provider.icon}</span>
                              <span className="font-medium">{provider.name}</span>
                            </div>
                            <span className="text-xs text-muted-foreground">{provider.description}</span>
                          </Label>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  <div className="flex justify-end pt-4">
                    <Button onClick={() => setDeployStep(2)} data-testid="button-next-step">
                      Continue <ArrowRight className="h-4 w-4 ml-2" />
                    </Button>
                  </div>
                </div>
              )}

              {deployStep === 2 && (
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label>Region</Label>
                    <Select value={selectedRegion} onValueChange={setSelectedRegion}>
                      <SelectTrigger className="bg-[#0d1117] border-[#30363d]" data-testid="select-region">
                        <SelectValue placeholder="Select deployment region" />
                      </SelectTrigger>
                      <SelectContent>
                        {(regions[selectedProvider as keyof typeof regions] || []).map((region) => (
                          <SelectItem key={region.value} value={region.value}>
                            {region.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Instance Configuration</Label>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">Min Instances</Label>
                        <Input type="number" defaultValue="2" className="bg-[#0d1117] border-[#30363d]" data-testid="input-min-instances" />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">Max Instances</Label>
                        <Input type="number" defaultValue="10" className="bg-[#0d1117] border-[#30363d]" data-testid="input-max-instances" />
                      </div>
                    </div>
                  </div>

                  <Card className="bg-[#0d1117] border-[#30363d]">
                    <CardContent className="p-4">
                      <h4 className="text-sm font-medium mb-3">Deployment Summary</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Provider</span>
                          <span>{cloudProviders.find(p => p.id === selectedProvider)?.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Region</span>
                          <span>{selectedRegion || 'Not selected'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Estimated Cost</span>
                          <span className="text-green-400">~$45/month</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="flex justify-between pt-4">
                    <Button variant="outline" onClick={() => setDeployStep(1)} className="border-[#30363d]" data-testid="button-back">
                      Back
                    </Button>
                    <Button 
                      onClick={handleDeploy} 
                      disabled={!selectedRegion || deploying}
                      className="bg-green-600 hover:bg-green-700"
                      data-testid="button-deploy"
                    >
                      {deploying ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" /> Deploying...
                        </>
                      ) : (
                        <>
                          <Rocket className="h-4 w-4 mr-2" /> Deploy Now
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-card/50 border-[#30363d]">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Active Deployments</CardTitle>
              <Globe className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-active-deployments">3</div>
              <p className="text-xs text-muted-foreground">Running in production</p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-[#30363d]">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Deploys</CardTitle>
              <Rocket className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-total-deploys">47</div>
              <p className="text-xs text-muted-foreground">This month</p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-[#30363d]">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Avg Deploy Time</CardTitle>
              <Zap className="h-4 w-4 text-yellow-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-avg-deploy-time">52s</div>
              <p className="text-xs text-muted-foreground">-8s from last month</p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-[#30363d]">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Success Rate</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="text-deploy-success-rate">98.2%</div>
              <p className="text-xs text-muted-foreground">Last 30 days</p>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-card/50 border-[#30363d]">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Deployment History</CardTitle>
              <Button variant="outline" size="sm" className="border-[#30363d]" data-testid="button-refresh-deployments">
                <RefreshCw className="h-4 w-4 mr-2" /> Refresh
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader className="bg-[#161b22]">
                <TableRow className="hover:bg-transparent border-[#30363d]">
                  <TableHead className="w-[300px]">Deployment</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Environment</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {mockDeployments.map((dep) => (
                  <TableRow key={dep.id} className="border-[#30363d] hover:bg-[#161b22] transition-colors" data-testid={`row-deployment-${dep.id}`}>
                    <TableCell>
                      <div className="flex flex-col gap-1">
                        <div className="font-medium flex items-center gap-2">
                          <span className="font-mono text-xs text-muted-foreground bg-[#161b22] px-1.5 py-0.5 rounded">#{dep.id.slice(4)}</span>
                          <span className="truncate max-w-[200px]">{dep.message}</span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1 font-mono"><GitCommit className="h-3 w-3" /> {dep.commit}</span>
                          <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> {dep.age}</span>
                          <span>by {dep.creator}</span>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      {dep.status === 'ready' && (
                        <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20 gap-1">
                          <CheckCircle2 className="h-3 w-3" /> Ready
                        </Badge>
                      )}
                      {dep.status === 'error' && (
                        <Badge variant="outline" className="bg-red-500/10 text-red-400 border-red-500/20 gap-1">
                          <XCircle className="h-3 w-3" /> Error
                        </Badge>
                      )}
                      {dep.status === 'building' && (
                        <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 gap-1">
                          <Loader2 className="h-3 w-3 animate-spin" /> Building
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="font-mono text-xs bg-[#161b22]">{dep.branch}</Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {dep.duration}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0" data-testid={`button-view-${dep.id}`}>
                          <ExternalLink className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="sm" className="h-8 px-2 text-xs" data-testid={`button-redeploy-${dep.id}`}>
                          Redeploy
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}