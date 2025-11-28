import { Link, useLocation } from "wouter";
import { 
  LayoutDashboard, 
  Activity, 
  Settings, 
  Terminal, 
  GitBranch, 
  Box, 
  LogOut,
  Cpu,
  Layers
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  const navItems = [
    { icon: LayoutDashboard, label: "Overview", href: "/dashboard" },
    { icon: Activity, label: "Deployments", href: "/dashboard/deployments" },
    { icon: Terminal, label: "Logs", href: "/dashboard/logs" },
    { icon: Settings, label: "Settings", href: "/dashboard/settings" },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground font-sans flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-border bg-card/20 flex flex-col">
        <div className="h-16 flex items-center px-6 border-b border-border/50">
          <div className="flex items-center gap-2 text-primary">
             <Cpu className="h-6 w-6" />
             <span className="font-bold font-mono tracking-tight">Platform<span className="text-white">Builder</span></span>
          </div>
        </div>

        <div className="p-4">
          <div className="mb-4 px-2 text-xs font-bold text-muted-foreground uppercase tracking-wider">
            Project
          </div>
          <div className="flex items-center gap-3 p-2 rounded-lg bg-white/5 border border-white/5 mb-6">
            <div className="h-8 w-8 rounded bg-primary/20 flex items-center justify-center text-primary">
              <Box className="h-4 w-4" />
            </div>
            <div className="overflow-hidden">
              <div className="font-bold text-sm truncate">fintech-core-api</div>
              <div className="text-xs text-muted-foreground flex items-center gap-1">
                <GitBranch className="h-3 w-3" /> main
              </div>
            </div>
          </div>

          <nav className="space-y-1">
            {navItems.map((item) => {
              const isActive = location === item.href;
              return (
                <Link key={item.href} href={item.href}>
                  <div className={`
                    flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors cursor-pointer
                    ${isActive 
                      ? "bg-primary/10 text-primary border border-primary/20" 
                      : "text-muted-foreground hover:text-foreground hover:bg-white/5"}
                  `}>
                    <item.icon className="h-4 w-4" />
                    {item.label}
                  </div>
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="mt-auto p-4 border-t border-border/50">
           <div className="flex items-center gap-3 px-2">
              <Avatar className="h-8 w-8 border border-border">
                <AvatarImage src="https://github.com/shadcn.png" />
                <AvatarFallback>US</AvatarFallback>
              </Avatar>
              <div className="flex-1 overflow-hidden">
                <div className="text-sm font-medium truncate">Demo User</div>
                <div className="text-xs text-muted-foreground truncate">user@example.com</div>
              </div>
              <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
                <LogOut className="h-4 w-4" />
              </Button>
           </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
         {/* Top Bar */}
         <header className="h-16 border-b border-border/50 bg-background/50 backdrop-blur-sm flex items-center justify-between px-8">
            <div className="flex items-center gap-4">
               <h1 className="text-lg font-semibold">
                 {navItems.find(i => i.href === location)?.label || "Dashboard"}
               </h1>
               {location === "/dashboard" && (
                 <div className="flex items-center gap-2 px-2 py-1 rounded-full bg-green-500/10 text-green-400 text-xs border border-green-500/20">
                   <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                   Operational
                 </div>
               )}
            </div>
            <div className="flex items-center gap-4">
               <Button variant="outline" size="sm" className="font-mono text-xs border-white/10 bg-white/5 hover:bg-white/10">
                  Feedback
               </Button>
               <Button size="sm" className="font-bold bg-primary text-primary-foreground hover:bg-primary/90">
                  Visit Deployment
               </Button>
            </div>
         </header>

         <div className="flex-1 overflow-auto p-8">
           <div className="max-w-6xl mx-auto">
             {children}
           </div>
         </div>
      </main>
    </div>
  );
}