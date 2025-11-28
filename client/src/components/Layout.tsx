import { Link, useLocation } from "wouter";
import { Terminal, Box, Cpu, Layers, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { 
  Sheet, 
  SheetContent, 
  SheetTrigger 
} from "@/components/ui/sheet";

export default function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  return (
    <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20 selection:text-primary">
      <nav className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="container flex h-16 items-center justify-between px-4 md:px-8">
          <div className="flex items-center gap-2">
            <Link href="/">
              <div className="flex items-center gap-2 cursor-pointer group">
                <div className="bg-primary/10 p-2 rounded-md group-hover:bg-primary/20 transition-colors">
                  <Cpu className="h-6 w-6 text-primary" />
                </div>
                <span className="text-xl font-bold tracking-tight font-mono">
                  Platform<span className="text-primary">Builder</span>
                </span>
              </div>
            </Link>
          </div>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-6 text-sm font-medium">
            <Link href="/features" className="text-muted-foreground hover:text-foreground transition-colors">Features</Link>
            <Link href="/enterprise" className="text-muted-foreground hover:text-foreground transition-colors">Enterprise</Link>
            <Link href="/docs" className="text-muted-foreground hover:text-foreground transition-colors">Docs</Link>
            <div className="h-4 w-px bg-border/50 mx-2" />
            <Button variant="ghost" className="font-mono text-xs">Login</Button>
            <Button variant="default" className="font-mono text-xs font-bold bg-primary text-primary-foreground hover:bg-primary/90">
              Start Building
            </Button>
          </div>

          {/* Mobile Nav */}
          <div className="md:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent>
                <div className="flex flex-col gap-4 mt-8">
                  <Link href="/features" className="text-lg font-medium">Features</Link>
                  <Link href="/enterprise" className="text-lg font-medium">Enterprise</Link>
                  <Link href="/docs" className="text-lg font-medium">Docs</Link>
                  <Button className="w-full mt-4">Start Building</Button>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </nav>

      <main>
        {children}
      </main>

      <footer className="border-t border-border/50 bg-background py-12 mt-20">
        <div className="container px-4 md:px-8 grid grid-cols-2 md:grid-cols-4 gap-8">
          <div>
            <h3 className="font-mono font-bold mb-4 text-primary">Platform</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><a href="#" className="hover:text-foreground">Deployments</a></li>
              <li><a href="#" className="hover:text-foreground">Edge Network</a></li>
              <li><a href="#" className="hover:text-foreground">Integrations</a></li>
            </ul>
          </div>
          <div>
            <h3 className="font-mono font-bold mb-4 text-primary">Resources</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><a href="#" className="hover:text-foreground">Documentation</a></li>
              <li><a href="#" className="hover:text-foreground">API Reference</a></li>
              <li><a href="#" className="hover:text-foreground">Community</a></li>
            </ul>
          </div>
          <div>
            <h3 className="font-mono font-bold mb-4 text-primary">Company</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><a href="#" className="hover:text-foreground">About</a></li>
              <li><a href="#" className="hover:text-foreground">Blog</a></li>
              <li><a href="#" className="hover:text-foreground">Careers</a></li>
            </ul>
          </div>
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <Terminal className="h-5 w-5 text-muted-foreground" />
              <span className="font-mono font-bold text-muted-foreground">System Status: <span className="text-green-500">Operational</span></span>
            </div>
            <p className="text-xs text-muted-foreground">
              © 2025 PlatformBuilder Inc.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}