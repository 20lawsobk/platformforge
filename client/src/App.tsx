import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import Home from "@/pages/Home";
import Builder from "@/pages/Builder";
import Overview from "@/pages/dashboard/Overview";
import Deployments from "@/pages/dashboard/Deployments";
import Settings from "@/pages/dashboard/Settings";
import Logs from "@/pages/dashboard/Logs";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/builder" component={Builder} />
      
      {/* Dashboard Routes */}
      <Route path="/dashboard" component={Overview} />
      <Route path="/dashboard/deployments" component={Deployments} />
      <Route path="/dashboard/settings" component={Settings} />
      <Route path="/dashboard/logs" component={Logs} />
      
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