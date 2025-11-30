import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useAuth } from "@/hooks/useAuth";
import NotFound from "@/pages/not-found";
import Home from "@/pages/Home";
import Builder from "@/pages/Builder";
import Compare from "@/pages/Compare";
import Features from "@/pages/Features";
import Docs from "@/pages/Docs";
import Onboarding from "@/pages/Onboarding";
import Overview from "@/pages/dashboard/Overview";
import DashboardDeployments from "@/pages/dashboard/Deployments";
import Settings from "@/pages/dashboard/Settings";
import Logs from "@/pages/dashboard/Logs";
import Storage from "@/pages/dashboard/Storage";
import Security from "@/pages/dashboard/Security";
import Secrets from "@/pages/dashboard/Secrets";
import DeploymentsPage from "@/pages/Deployments";
import EdgeNetwork from "@/pages/EdgeNetwork";
import Integrations from "@/pages/Integrations";
import ApiReference from "@/pages/ApiReference";
import Community from "@/pages/Community";
import About from "@/pages/About";
import Blog from "@/pages/Blog";
import Careers from "@/pages/Careers";
import IDE from "@/pages/IDE";
import Templates from "@/pages/Templates";

function Router() {
  const { user, isLoading, isAuthenticated } = useAuth();

  return (
    <Switch>
      {/* Public pages */}
      <Route path="/" component={Home} />
      <Route path="/compare" component={Compare} />
      <Route path="/features" component={Features} />
      <Route path="/templates" component={Templates} />
      <Route path="/docs" component={Docs} />
      
      {/* Platform Pages */}
      <Route path="/deployments" component={DeploymentsPage} />
      <Route path="/edge-network" component={EdgeNetwork} />
      <Route path="/integrations" component={Integrations} />
      
      {/* Resource Pages */}
      <Route path="/api-reference" component={ApiReference} />
      <Route path="/community" component={Community} />
      
      {/* Company Pages */}
      <Route path="/about" component={About} />
      <Route path="/blog" component={Blog} />
      <Route path="/careers" component={Careers} />
      
      {/* Auth-protected routes */}
      <Route path="/onboarding">
        {isAuthenticated ? <Onboarding /> : <Home />}
      </Route>
      <Route path="/builder">
        {isAuthenticated ? <Builder /> : <Home />}
      </Route>
      
      {/* Dashboard Routes - protected */}
      <Route path="/dashboard">
        {isAuthenticated ? (
          user?.onboardingCompleted ? <Overview /> : <Onboarding />
        ) : <Home />}
      </Route>
      <Route path="/dashboard/deployments">
        {isAuthenticated ? <DashboardDeployments /> : <Home />}
      </Route>
      <Route path="/dashboard/settings">
        {isAuthenticated ? <Settings /> : <Home />}
      </Route>
      <Route path="/dashboard/logs">
        {isAuthenticated ? <Logs /> : <Home />}
      </Route>
      <Route path="/dashboard/storage">
        {isAuthenticated ? <Storage /> : <Home />}
      </Route>
      <Route path="/dashboard/security">
        {isAuthenticated ? <Security /> : <Home />}
      </Route>
      <Route path="/dashboard/secrets">
        {isAuthenticated ? <Secrets /> : <Home />}
      </Route>
      
      {/* IDE Route - protected */}
      <Route path="/ide/:projectId">
        {isAuthenticated ? <IDE /> : <Home />}
      </Route>
      
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
