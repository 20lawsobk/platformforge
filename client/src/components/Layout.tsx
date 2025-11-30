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
            <Link href="/compare" className="text-muted-foreground hover:text-foreground transition-colors" data-testid="link-nav-compare">Compare</Link>
            <Link href="/features" className="text-muted-foreground hover:text-foreground transition-colors" data-testid="link-nav-features">Features</Link>
            <Link href="/docs" className="text-muted-foreground hover:text-foreground transition-colors" data-testid="link-nav-docs">Docs</Link>
            <div className="h-4 w-px bg-border/50 mx-2" />
            <a href="/api/login">
              <Button variant="ghost" className="font-mono text-xs" data-testid="button-nav-login">Login</Button>
            </a>
            <a href="/api/login">
              <Button variant="default" className="font-mono text-xs font-bold bg-primary text-primary-foreground hover:bg-primary/90" data-testid="button-nav-start">
                Start Building
              </Button>
            </a>
          </div>

          {/* Mobile Nav */}
          <div className="md:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" data-testid="button-mobile-menu">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent>
                <div className="flex flex-col gap-4 mt-8">
                  <Link href="/compare" className="text-lg font-medium" data-testid="link-mobile-compare">Compare</Link>
                  <Link href="/features" className="text-lg font-medium" data-testid="link-mobile-features">Features</Link>
                  <Link href="/docs" className="text-lg font-medium" data-testid="link-mobile-docs">Docs</Link>
                  <a href="/api/login" className="text-lg font-medium" data-testid="link-mobile-login">Login</a>
                  <a href="/api/login">
                    <Button className="w-full mt-4" data-testid="button-mobile-start">Start Building</Button>
                  </a>
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
              <li><Link href="/deployments" className="hover:text-foreground" data-testid="link-footer-deployments">Deployments</Link></li>
              <li><Link href="/edge-network" className="hover:text-foreground" data-testid="link-footer-edge-network">Edge Network</Link></li>
              <li><Link href="/integrations" className="hover:text-foreground" data-testid="link-footer-integrations">Integrations</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="font-mono font-bold mb-4 text-primary">Resources</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link href="/docs" className="hover:text-foreground" data-testid="link-footer-docs">Documentation</Link></li>
              <li><Link href="/api-reference" className="hover:text-foreground" data-testid="link-footer-api-reference">API Reference</Link></li>
              <li><Link href="/community" className="hover:text-foreground" data-testid="link-footer-community">Community</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="font-mono font-bold mb-4 text-primary">Company</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link href="/about" className="hover:text-foreground" data-testid="link-footer-about">About</Link></li>
              <li><Link href="/blog" className="hover:text-foreground" data-testid="link-footer-blog">Blog</Link></li>
              <li><Link href="/careers" className="hover:text-foreground" data-testid="link-footer-careers">Careers</Link></li>
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