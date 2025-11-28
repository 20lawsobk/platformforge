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
import DashboardLayout from "@/components/DashboardLayout";
import { GitCommit, ExternalLink, Clock, CheckCircle2, XCircle, Loader2 } from "lucide-react";

const deployments = [
  { id: "dep_8f2a9c1", commit: "8f2a9c1", message: "Updated landing page copy", branch: "main", status: "ready", duration: "45s", age: "2m ago", creator: "alex_dev" },
  { id: "dep_7b1x2y3", commit: "7b1x2y3", message: "Fix navigation bug on mobile", branch: "main", status: "ready", duration: "52s", age: "4h ago", creator: "alex_dev" },
  { id: "dep_3c4v5b6", commit: "3c4v5b6", message: "Add analytics tracking", branch: "feature/analytics", status: "error", duration: "1m 12s", age: "1d ago", creator: "sarah_lead" },
  { id: "dep_9n8m7k6", commit: "9n8m7k6", message: "Initial project setup", branch: "main", status: "ready", duration: "2m 05s", age: "3d ago", creator: "alex_dev" },
];

export default function Deployments() {
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
           <div>
             <h2 className="text-2xl font-bold tracking-tight">Deployments</h2>
             <p className="text-muted-foreground">Manage and view your project build history.</p>
           </div>
           <Button>Redeploy</Button>
        </div>

        <div className="rounded-md border border-white/5 bg-card/50 backdrop-blur-sm overflow-hidden">
          <Table>
            <TableHeader className="bg-white/5">
              <TableRow className="hover:bg-transparent border-white/5">
                <TableHead className="w-[250px]">Deployment</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Environment</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {deployments.map((dep) => (
                <TableRow key={dep.id} className="border-white/5 hover:bg-white/5 transition-colors">
                  <TableCell>
                    <div className="flex flex-col gap-1">
                      <div className="font-medium flex items-center gap-2">
                        <span className="font-mono text-xs text-muted-foreground">#{dep.id.slice(4)}</span>
                        <span>{dep.message}</span>
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
                    <Badge variant="secondary" className="font-mono text-xs">{dep.branch}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {dep.duration}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </DashboardLayout>
  );
}