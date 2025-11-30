import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  Shield, 
  Play, 
  AlertTriangle, 
  AlertCircle, 
  Info, 
  CheckCircle2,
  Clock,
  FileCode,
  Filter,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Folder
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import DashboardLayout from '@/components/DashboardLayout';
import { apiRequest, queryClient } from '@/lib/queryClient';
import type { Project, SecurityScan, SecurityFinding } from '@shared/schema';

type SeverityLevel = 'critical' | 'high' | 'medium' | 'low';
type FindingStatus = 'open' | 'resolved' | 'ignored';

const severityColors: Record<SeverityLevel, string> = {
  critical: 'bg-red-500/10 text-red-400 border-red-500/20',
  high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  low: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
};

const statusColors: Record<string, string> = {
  open: 'bg-red-500/10 text-red-400 border-red-500/20',
  resolved: 'bg-green-500/10 text-green-400 border-green-500/20',
  ignored: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
  running: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  completed: 'bg-green-500/10 text-green-400 border-green-500/20',
  pending: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  failed: 'bg-red-500/10 text-red-400 border-red-500/20',
};

function SeverityBadge({ severity }: { severity: string }) {
  const level = severity.toLowerCase() as SeverityLevel;
  const colorClass = severityColors[level] || severityColors.low;
  
  return (
    <Badge variant="outline" className={colorClass} data-testid={`badge-severity-${severity}`}>
      {severity === 'critical' && <AlertCircle className="h-3 w-3 mr-1" />}
      {severity === 'high' && <AlertTriangle className="h-3 w-3 mr-1" />}
      {severity === 'medium' && <Info className="h-3 w-3 mr-1" />}
      {severity === 'low' && <Info className="h-3 w-3 mr-1" />}
      {severity.charAt(0).toUpperCase() + severity.slice(1)}
    </Badge>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colorClass = statusColors[status.toLowerCase()] || statusColors.open;
  
  return (
    <Badge variant="outline" className={colorClass} data-testid={`badge-status-${status}`}>
      {status === 'running' && <RefreshCw className="h-3 w-3 mr-1 animate-spin" />}
      {status === 'completed' && <CheckCircle2 className="h-3 w-3 mr-1" />}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
}

function ProjectScansRow({ project, onSelectProject, isSelected }: { 
  project: Project; 
  onSelectProject: (projectId: string) => void;
  isSelected: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  
  const { data: scans = [], isLoading: scansLoading } = useQuery<SecurityScan[]>({
    queryKey: ['/api/projects', project.id, 'security', 'scans'],
  });

  const scanMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/security/scan', { projectId: project.id });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/projects', project.id, 'security', 'scans'] });
    },
  });

  const latestScan = scans.length > 0 ? scans[0] : null;
  const isScanning = latestScan?.status === 'running' || scanMutation.isPending;

  return (
    <Card className="bg-card/50 border-white/5" data-testid={`card-project-scan-${project.id}`}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2 cursor-pointer">
            <div className="flex items-center gap-3">
              {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              <div className="h-8 w-8 rounded bg-primary/10 flex items-center justify-center">
                <Folder className="h-4 w-4 text-primary" />
              </div>
              <div>
                <CardTitle className="text-sm font-medium" data-testid={`text-project-name-${project.id}`}>
                  {project.name}
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  {scans.length} scan{scans.length !== 1 ? 's' : ''}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {latestScan && (
                <div className="flex items-center gap-2 text-xs">
                  <StatusBadge status={latestScan.status} />
                  {latestScan.status === 'completed' && (
                    <div className="flex items-center gap-1">
                      {(latestScan.criticalCount ?? 0) > 0 && (
                        <span className="text-red-400">{latestScan.criticalCount} critical</span>
                      )}
                      {(latestScan.highCount ?? 0) > 0 && (
                        <span className="text-orange-400">{latestScan.highCount} high</span>
                      )}
                    </div>
                  )}
                </div>
              )}
              <Button
                size="sm"
                variant="outline"
                onClick={(e) => {
                  e.stopPropagation();
                  scanMutation.mutate();
                }}
                disabled={isScanning}
                data-testid={`button-scan-${project.id}`}
              >
                {isScanning ? (
                  <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                ) : (
                  <Play className="h-3 w-3 mr-1 fill-current" />
                )}
                {isScanning ? 'Scanning...' : 'Scan'}
              </Button>
              <Button
                size="sm"
                variant={isSelected ? "default" : "ghost"}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectProject(project.id);
                }}
                data-testid={`button-view-findings-${project.id}`}
              >
                View Findings
              </Button>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent>
            {scansLoading ? (
              <div className="flex items-center justify-center py-4">
                <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : scans.length === 0 ? (
              <div className="text-center py-4 text-sm text-muted-foreground">
                No scans yet. Run a security scan to check for vulnerabilities.
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="border-white/5">
                    <TableHead className="text-muted-foreground">Status</TableHead>
                    <TableHead className="text-muted-foreground">Started</TableHead>
                    <TableHead className="text-muted-foreground text-center">Critical</TableHead>
                    <TableHead className="text-muted-foreground text-center">High</TableHead>
                    <TableHead className="text-muted-foreground text-center">Medium</TableHead>
                    <TableHead className="text-muted-foreground text-center">Low</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {scans.map((scan) => (
                    <TableRow key={scan.id} className="border-white/5" data-testid={`row-scan-${scan.id}`}>
                      <TableCell>
                        <StatusBadge status={scan.status} />
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {scan.startedAt ? new Date(scan.startedAt).toLocaleString() : '-'}
                        </div>
                      </TableCell>
                      <TableCell className="text-center">
                        <span className={`text-sm ${(scan.criticalCount ?? 0) > 0 ? 'text-red-400 font-medium' : 'text-muted-foreground'}`}>
                          {scan.criticalCount ?? 0}
                        </span>
                      </TableCell>
                      <TableCell className="text-center">
                        <span className={`text-sm ${(scan.highCount ?? 0) > 0 ? 'text-orange-400 font-medium' : 'text-muted-foreground'}`}>
                          {scan.highCount ?? 0}
                        </span>
                      </TableCell>
                      <TableCell className="text-center">
                        <span className={`text-sm ${(scan.mediumCount ?? 0) > 0 ? 'text-yellow-400 font-medium' : 'text-muted-foreground'}`}>
                          {scan.mediumCount ?? 0}
                        </span>
                      </TableCell>
                      <TableCell className="text-center">
                        <span className={`text-sm ${(scan.lowCount ?? 0) > 0 ? 'text-blue-400 font-medium' : 'text-muted-foreground'}`}>
                          {scan.lowCount ?? 0}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

function FindingsSection({ projectId, projectName }: { projectId: string; projectName: string }) {
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const { data: findings = [], isLoading } = useQuery<SecurityFinding[]>({
    queryKey: ['/api/projects', projectId, 'security', 'findings'],
    enabled: !!projectId,
  });

  const updateStatusMutation = useMutation({
    mutationFn: async ({ findingId, status }: { findingId: string; status: string }) => {
      const response = await apiRequest('PATCH', `/api/security/findings/${findingId}/status`, { status });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/projects', projectId, 'security', 'findings'] });
    },
  });

  const filteredFindings = findings.filter((finding) => {
    if (severityFilter !== 'all' && finding.severity.toLowerCase() !== severityFilter) {
      return false;
    }
    if (statusFilter !== 'all' && finding.status !== statusFilter) {
      return false;
    }
    return true;
  });

  const severityCounts = {
    critical: findings.filter(f => f.severity.toLowerCase() === 'critical').length,
    high: findings.filter(f => f.severity.toLowerCase() === 'high').length,
    medium: findings.filter(f => f.severity.toLowerCase() === 'medium').length,
    low: findings.filter(f => f.severity.toLowerCase() === 'low').length,
  };

  return (
    <Card className="bg-card/50 border-white/5" data-testid="card-findings-section">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              Security Findings
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1" data-testid="text-findings-project">
              {projectName}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-xs">
              <Badge variant="outline" className={severityColors.critical}>
                {severityCounts.critical}
              </Badge>
              <Badge variant="outline" className={severityColors.high}>
                {severityCounts.high}
              </Badge>
              <Badge variant="outline" className={severityColors.medium}>
                {severityCounts.medium}
              </Badge>
              <Badge variant="outline" className={severityColors.low}>
                {severityCounts.low}
              </Badge>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-4">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="w-32 h-9 bg-[#0d1117] border-[#30363d]" data-testid="select-severity-filter">
              <SelectValue placeholder="Severity" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severity</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-32 h-9 bg-[#0d1117] border-[#30363d]" data-testid="select-status-filter">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="open">Open</SelectItem>
              <SelectItem value="resolved">Resolved</SelectItem>
              <SelectItem value="ignored">Ignored</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : filteredFindings.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center" data-testid="empty-findings">
            <Shield className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">
              {findings.length === 0 ? 'No Security Findings' : 'No Matching Findings'}
            </h3>
            <p className="text-sm text-muted-foreground max-w-sm">
              {findings.length === 0 
                ? 'Run a security scan to check for vulnerabilities in this project.'
                : 'Try adjusting your filters to see more findings.'}
            </p>
          </div>
        ) : (
          <ScrollArea className="h-[500px]">
            <div className="space-y-4">
              {filteredFindings.map((finding) => (
                <Card 
                  key={finding.id} 
                  className="bg-[#161b22] border-[#30363d]"
                  data-testid={`card-finding-${finding.id}`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <div className="flex items-center gap-2 flex-wrap">
                        <SeverityBadge severity={finding.severity} />
                        <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20">
                          {finding.category}
                        </Badge>
                      </div>
                      <Select 
                        value={finding.status || 'open'} 
                        onValueChange={(status) => updateStatusMutation.mutate({ findingId: finding.id, status })}
                        disabled={updateStatusMutation.isPending}
                      >
                        <SelectTrigger 
                          className="w-28 h-8 bg-[#0d1117] border-[#30363d]"
                          data-testid={`select-finding-status-${finding.id}`}
                        >
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="open">Open</SelectItem>
                          <SelectItem value="resolved">Resolved</SelectItem>
                          <SelectItem value="ignored">Ignored</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <h4 className="font-medium mb-2" data-testid={`text-finding-title-${finding.id}`}>
                      {finding.title}
                    </h4>
                    
                    <p className="text-sm text-muted-foreground mb-3" data-testid={`text-finding-description-${finding.id}`}>
                      {finding.description}
                    </p>
                    
                    {(finding.filePath || finding.lineNumber) && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
                        <FileCode className="h-3 w-3" />
                        <code className="bg-[#0d1117] px-2 py-1 rounded" data-testid={`text-finding-location-${finding.id}`}>
                          {finding.filePath}{finding.lineNumber ? `:${finding.lineNumber}` : ''}
                        </code>
                      </div>
                    )}
                    
                    {finding.recommendation && (
                      <div className="bg-[#0d1117] border border-[#30363d] rounded-md p-3 mt-3">
                        <div className="flex items-center gap-2 text-xs text-primary mb-1">
                          <CheckCircle2 className="h-3 w-3" />
                          Recommendation
                        </div>
                        <p className="text-sm text-muted-foreground" data-testid={`text-finding-recommendation-${finding.id}`}>
                          {finding.recommendation}
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}

export default function Security() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  const { data: projects = [], isLoading: projectsLoading } = useQuery<Project[]>({
    queryKey: ['/api/user/projects'],
  });

  const selectedProject = projects.find(p => p.id === selectedProjectId);

  return (
    <DashboardLayout>
      <div className="space-y-8" data-testid="security-page">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Shield className="h-6 w-6 text-primary" />
            Security Scanner
          </h2>
          <p className="text-muted-foreground">
            Scan your projects for security vulnerabilities and manage findings
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Projects & Scan History</h3>
            {projectsLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <Card key={i} className="bg-card/50 border-white/5 animate-pulse">
                    <CardContent className="p-6">
                      <div className="h-6 bg-white/10 rounded mb-4 w-3/4" />
                      <div className="h-4 bg-white/10 rounded w-1/2" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : projects.length === 0 ? (
              <Card className="bg-card/50 border-white/5 border-dashed" data-testid="empty-projects">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Folder className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">No Projects</h3>
                  <p className="text-muted-foreground text-sm text-center max-w-sm">
                    Create a project first to run security scans
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {projects.map((project) => (
                  <ProjectScansRow
                    key={project.id}
                    project={project}
                    onSelectProject={setSelectedProjectId}
                    isSelected={selectedProjectId === project.id}
                  />
                ))}
              </div>
            )}
          </div>

          <div>
            {selectedProject ? (
              <FindingsSection 
                projectId={selectedProject.id} 
                projectName={selectedProject.name}
              />
            ) : (
              <Card className="bg-card/50 border-white/5 border-dashed h-full min-h-[400px]" data-testid="no-project-selected">
                <CardContent className="flex flex-col items-center justify-center h-full py-12">
                  <Shield className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">Select a Project</h3>
                  <p className="text-muted-foreground text-sm text-center max-w-sm">
                    Click "View Findings" on a project to see its security findings
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
