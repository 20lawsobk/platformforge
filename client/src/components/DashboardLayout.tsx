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
  Plus,
  ChevronDown,
  Search,
  Bell,
  HelpCircle,
  Database,
  Shield,
  Key
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  const navItems = [
    { icon: LayoutDashboard, label: "Overview", href: "/dashboard" },
    { icon: Activity, label: "Deployments", href: "/dashboard/deployments" },
    { icon: Database, label: "Storage", href: "/dashboard/storage" },
    { icon: Key, label: "Secrets", href: "/dashboard/secrets" },
    { icon: Shield, label: "Security", href: "/dashboard/security" },
    { icon: Terminal, label: "Logs", href: "/dashboard/logs" },
    { icon: Settings, label: "Settings", href: "/dashboard/settings" },
  ];

  return (
    <div className="min-h-screen bg-[#0d1117] text-foreground font-sans flex" data-testid="dashboard-layout">
      <aside className="w-60 border-r border-[#30363d] bg-[#161b22] flex flex-col">
        <div className="h-14 flex items-center px-4 border-b border-[#30363d]">
          <Link href="/">
            <div className="flex items-center gap-2 text-primary cursor-pointer hover:opacity-80 transition-opacity" data-testid="link-logo">
              <Cpu className="h-5 w-5" />
              <span className="font-bold font-mono text-sm tracking-tight">Platform<span className="text-white">Architect</span></span>
            </div>
          </Link>
        </div>

        <div className="p-3">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="w-full flex items-center gap-3 p-2 rounded-md bg-[#0d1117] border border-[#30363d] hover:border-[#484f58] transition-colors cursor-pointer" data-testid="dropdown-project-selector">
                <div className="h-8 w-8 rounded bg-primary/20 flex items-center justify-center text-primary">
                  <Box className="h-4 w-4" />
                </div>
                <div className="flex-1 text-left overflow-hidden">
                  <div className="font-medium text-sm truncate">My Projects</div>
                  <div className="text-xs text-muted-foreground flex items-center gap-1">
                    <GitBranch className="h-3 w-3" /> workspace
                  </div>
                </div>
                <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuItem>
                <Box className="h-4 w-4 mr-2" /> My Projects
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Box className="h-4 w-4 mr-2" /> Team Projects
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Plus className="h-4 w-4 mr-2" /> Create Workspace
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {navItems.map((item) => {
            const isActive = location === item.href;
            return (
              <Link key={item.href} href={item.href}>
                <div 
                  className={`
                    flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors cursor-pointer
                    ${isActive 
                      ? "bg-primary/10 text-primary" 
                      : "text-[#8b949e] hover:text-[#c9d1d9] hover:bg-[#21262d]"}
                  `}
                  data-testid={`nav-${item.label.toLowerCase()}`}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </div>
              </Link>
            );
          })}
        </nav>

        <div className="p-3 border-t border-[#30363d]">
          <Link href="/">
            <Button className="w-full bg-green-600 hover:bg-green-700 text-white font-medium" size="sm" data-testid="button-new-project-sidebar">
              <Plus className="h-4 w-4 mr-2" /> New Project
            </Button>
          </Link>
        </div>

        <div className="p-3 border-t border-[#30363d]">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="w-full flex items-center gap-3 px-2 py-1.5 rounded-md hover:bg-[#21262d] transition-colors cursor-pointer" data-testid="dropdown-user-menu">
                <Avatar className="h-7 w-7 border border-[#30363d]">
                  <AvatarImage src="https://github.com/shadcn.png" />
                  <AvatarFallback>PA</AvatarFallback>
                </Avatar>
                <div className="flex-1 overflow-hidden text-left">
                  <div className="text-sm font-medium truncate">Demo User</div>
                </div>
                <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuItem>
                <Settings className="h-4 w-4 mr-2" /> Account Settings
              </DropdownMenuItem>
              <DropdownMenuItem>
                <HelpCircle className="h-4 w-4 mr-2" /> Help & Support
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-red-400 focus:text-red-400">
                <LogOut className="h-4 w-4 mr-2" /> Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-14 border-b border-[#30363d] bg-[#161b22] flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="relative w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input 
                placeholder="Search projects..." 
                className="pl-9 h-9 bg-[#0d1117] border-[#30363d] focus:border-primary"
                data-testid="input-search"
              />
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="h-9 w-9 text-[#8b949e] hover:text-[#c9d1d9]" data-testid="button-notifications">
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-9 w-9 text-[#8b949e] hover:text-[#c9d1d9]" data-testid="button-help">
              <HelpCircle className="h-4 w-4" />
            </Button>
            <div className="h-6 w-px bg-[#30363d]" />
            <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white font-medium" data-testid="button-upgrade">
              Upgrade
            </Button>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6 bg-[#0d1117]">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
}