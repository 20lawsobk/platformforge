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
  GitCommit, 
  GitBranch,
  ArrowUpRight, 
  CheckCircle2,
  AlertCircle,
  Terminal
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import DashboardLayout from '@/components/DashboardLayout';

const data = [
  { time: '00:00', reqs: 400, cpu: 24 },
  { time: '04:00', reqs: 300, cpu: 18 },
  { time: '08:00', reqs: 2000, cpu: 65 },
  { time: '12:00', reqs: 4500, cpu: 88 },
  { time: '16:00', reqs: 3800, cpu: 75 },
  { time: '20:00', reqs: 1200, cpu: 45 },
  { time: '23:59', reqs: 800, cpu: 30 },
];

export default function Overview() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Requests</CardTitle>
              <Globe className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">2.4M</div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                <span className="text-green-400 flex items-center mr-1"><ArrowUpRight className="h-3 w-3" /> +12.5%</span> from last week
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Avg. Latency</CardTitle>
              <Clock className="h-4 w-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">42ms</div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                <span className="text-green-400 flex items-center mr-1"><ArrowUpRight className="h-3 w-3" /> -14ms</span> global average
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">CPU Usage</CardTitle>
              <Cpu className="h-4 w-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">65%</div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                <span className="text-yellow-400 flex items-center mr-1"><AlertCircle className="h-3 w-3" /> High load</span> peak hour
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-white/5">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Success Rate</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">99.99%</div>
              <p className="text-xs text-muted-foreground flex items-center mt-1">
                <span className="text-muted-foreground">Last 24 hours</span>
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <Card className="col-span-2 bg-card/50 border-white/5">
            <CardHeader>
              <CardTitle>Traffic Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data}>
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

          {/* Recent Activity */}
          <Card className="bg-card/50 border-white/5">
            <CardHeader>
              <CardTitle>Latest Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {[
                  { action: "Deployment to production", user: "alex_dev", time: "2m ago", status: "success" },
                  { action: "Env variable updated", user: "sarah_lead", time: "1h ago", status: "info" },
                  { action: "Build failed: main", user: "system", time: "3h ago", status: "error" },
                  { action: "Scale up: us-east-1", user: "autoscaler", time: "5h ago", status: "warning" },
                ].map((item, i) => (
                  <div key={i} className="flex gap-4">
                     <div className={`mt-1 h-2 w-2 rounded-full shrink-0
                        ${item.status === 'success' ? 'bg-green-400' : ''}
                        ${item.status === 'error' ? 'bg-red-400' : ''}
                        ${item.status === 'warning' ? 'bg-yellow-400' : ''}
                        ${item.status === 'info' ? 'bg-blue-400' : ''}
                     `} />
                     <div>
                        <p className="text-sm font-medium leading-none mb-1">{item.action}</p>
                        <p className="text-xs text-muted-foreground">
                           by <span className="text-foreground">{item.user}</span> • {item.time}
                        </p>
                     </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Deployment Info */}
        <Card className="bg-card/50 border-white/5 overflow-hidden">
           <div className="flex flex-col md:flex-row">
              <div className="p-6 flex-1 border-b md:border-b-0 md:border-r border-white/5">
                 <h3 className="text-sm font-medium text-muted-foreground mb-4">Active Deployment</h3>
                 <div className="flex items-start gap-4">
                    <div className="h-10 w-10 rounded bg-green-500/10 flex items-center justify-center text-green-400 shrink-0">
                       <CheckCircle2 className="h-5 w-5" />
                    </div>
                    <div>
                       <div className="flex items-center gap-2 mb-1">
                          <span className="font-mono font-bold text-lg">8f2a9c1</span>
                          <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20">Live</Badge>
                       </div>
                       <p className="text-sm text-muted-foreground mb-2">Updated landing page copy and fixed header responsiveness.</p>
                       <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1"><GitBranch className="h-3 w-3" /> main</span>
                          <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> 2m ago</span>
                       </div>
                    </div>
                 </div>
              </div>
              <div className="p-6 w-full md:w-80 bg-white/2">
                 <h3 className="text-sm font-medium text-muted-foreground mb-4">Quick Actions</h3>
                 <div className="space-y-2">
                    <Button variant="outline" className="w-full justify-start border-white/10 hover:bg-white/5">
                       <Terminal className="h-4 w-4 mr-2" /> View Logs
                    </Button>
                    <Button variant="outline" className="w-full justify-start border-white/10 hover:bg-white/5">
                       <Database className="h-4 w-4 mr-2" /> Connect Database
                    </Button>
                 </div>
              </div>
           </div>
        </Card>
      </div>
    </DashboardLayout>
  );
}