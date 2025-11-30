import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import Home from "@/pages/Home";
import Builder from "@/pages/Builder";
import Compare from "@/pages/Compare";
import Features from "@/pages/Features";
import Docs from "@/pages/Docs";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import Overview from "@/pages/dashboard/Overview";
import Deployments from "@/pages/dashboard/Deployments";
import Settings from "@/pages/dashboard/Settings";
import Logs from "@/pages/dashboard/Logs";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/builder" component={Builder} />
      <Route path="/compare" component={Compare} />
      <Route path="/features" component={Features} />
      <Route path="/docs" component={Docs} />
      <Route path="/login" component={Login} />
      <Route path="/register" component={Register} />
      
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