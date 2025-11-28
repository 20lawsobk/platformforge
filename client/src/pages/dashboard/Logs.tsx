import DashboardLayout from "@/components/DashboardLayout";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, Download, Clock, Filter } from "lucide-react";

export default function Logs() {
  const logs = Array.from({ length: 50 }).map((_, i) => ({
    id: i,
    timestamp: new Date(Date.now() - i * 1000 * 5).toISOString(),
    level: i % 10 === 0 ? 'error' : i % 5 === 0 ? 'warn' : 'info',
    message: i % 10 === 0 
      ? "Failed to connect to database pool" 
      : i % 5 === 0 
      ? "Memory usage high: 85%" 
      : `Request processed: GET /api/v1/users/${1000 + i} 200 OK`
  }));

  return (
    <DashboardLayout>
      <div className="space-y-6 h-[calc(100vh-10rem)] flex flex-col">
        <div className="flex items-center justify-between">
           <div>
             <h2 className="text-2xl font-bold tracking-tight">Runtime Logs</h2>
             <p className="text-muted-foreground">Live stream of your application logs.</p>
           </div>
           <div className="flex gap-2">
              <Button variant="outline"><Download className="h-4 w-4 mr-2" /> Export</Button>
              <Button variant="secondary">Live Mode <span className="ml-2 h-2 w-2 rounded-full bg-green-500 animate-pulse" /></Button>
           </div>
        </div>

        <div className="flex gap-2">
          <div className="relative flex-1">
             <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
             <Input placeholder="Search logs..." className="pl-9 bg-card/50 border-white/10" />
          </div>
          <Button variant="outline"><Filter className="h-4 w-4 mr-2" /> Filter</Button>
        </div>

        <div className="flex-1 rounded-md border border-white/5 bg-black/50 font-mono text-xs overflow-hidden flex flex-col">
           <div className="flex items-center px-4 py-2 border-b border-white/5 bg-white/5 text-muted-foreground">
              <div className="w-40">Timestamp</div>
              <div className="w-16">Level</div>
              <div className="flex-1">Message</div>
           </div>
           <ScrollArea className="flex-1">
              <div className="p-2">
                 {logs.map((log) => (
                   <div key={log.id} className="flex items-start py-1 px-2 hover:bg-white/5 rounded-sm group">
                      <div className="w-40 text-muted-foreground shrink-0 select-none">{log.timestamp.split('T')[1].slice(0, -1)}</div>
                      <div className="w-16 shrink-0">
                         <span className={`
                           uppercase font-bold
                           ${log.level === 'error' ? 'text-red-500' : ''}
                           ${log.level === 'warn' ? 'text-yellow-500' : ''}
                           ${log.level === 'info' ? 'text-blue-500' : ''}
                         `}>{log.level}</span>
                      </div>
                      <div className="flex-1 break-all text-foreground/90">
                         {log.message}
                      </div>
                   </div>
                 ))}
              </div>
           </ScrollArea>
        </div>
      </div>
    </DashboardLayout>
  );
}