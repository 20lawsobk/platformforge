import { motion } from "framer-motion";
import { 
  Code,
  FileText,
  Server,
  Shield,
  Webhook,
  Folder,
  Rocket,
  Key,
  Copy,
  ExternalLink,
  ChevronRight,
  Terminal,
  Zap
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";

interface Endpoint {
  method: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  path: string;
  description: string;
}

interface ApiSection {
  title: string;
  description: string;
  icon: React.ElementType;
  endpoints: Endpoint[];
}

const apiSections: ApiSection[] = [
  {
    title: "Projects",
    description: "Create, manage, and configure projects",
    icon: Folder,
    endpoints: [
      { method: "GET", path: "/api/v1/projects", description: "List all projects for the authenticated user" },
      { method: "POST", path: "/api/v1/projects", description: "Create a new project with specified configuration" },
      { method: "GET", path: "/api/v1/projects/:id", description: "Retrieve details of a specific project" },
      { method: "PUT", path: "/api/v1/projects/:id", description: "Update project settings and configuration" },
      { method: "DELETE", path: "/api/v1/projects/:id", description: "Delete a project and all associated resources" }
    ]
  },
  {
    title: "Deployments",
    description: "Deploy and manage application instances",
    icon: Rocket,
    endpoints: [
      { method: "GET", path: "/api/v1/deployments", description: "List all deployments for a project" },
      { method: "POST", path: "/api/v1/deployments", description: "Trigger a new deployment" },
      { method: "GET", path: "/api/v1/deployments/:id", description: "Get deployment status and details" },
      { method: "POST", path: "/api/v1/deployments/:id/rollback", description: "Rollback to a previous deployment" },
      { method: "DELETE", path: "/api/v1/deployments/:id", description: "Cancel or remove a deployment" }
    ]
  },
  {
    title: "Infrastructure",
    description: "Manage infrastructure resources and scaling",
    icon: Server,
    endpoints: [
      { method: "GET", path: "/api/v1/infrastructure", description: "List infrastructure resources" },
      { method: "POST", path: "/api/v1/infrastructure/scale", description: "Scale infrastructure up or down" },
      { method: "GET", path: "/api/v1/infrastructure/metrics", description: "Retrieve infrastructure metrics" },
      { method: "PUT", path: "/api/v1/infrastructure/config", description: "Update infrastructure configuration" }
    ]
  },
  {
    title: "Auth",
    description: "Authentication and authorization endpoints",
    icon: Shield,
    endpoints: [
      { method: "POST", path: "/api/v1/auth/token", description: "Generate a new API access token" },
      { method: "POST", path: "/api/v1/auth/refresh", description: "Refresh an existing access token" },
      { method: "DELETE", path: "/api/v1/auth/token/:id", description: "Revoke an API token" },
      { method: "GET", path: "/api/v1/auth/permissions", description: "List permissions for current token" }
    ]
  },
  {
    title: "Webhooks",
    description: "Configure and manage webhook integrations",
    icon: Webhook,
    endpoints: [
      { method: "GET", path: "/api/v1/webhooks", description: "List all configured webhooks" },
      { method: "POST", path: "/api/v1/webhooks", description: "Create a new webhook endpoint" },
      { method: "PUT", path: "/api/v1/webhooks/:id", description: "Update webhook configuration" },
      { method: "DELETE", path: "/api/v1/webhooks/:id", description: "Remove a webhook" },
      { method: "POST", path: "/api/v1/webhooks/:id/test", description: "Send a test event to webhook" }
    ]
  }
];

const methodColors: Record<string, string> = {
  GET: "bg-green-500/10 text-green-500 border-green-500/30",
  POST: "bg-blue-500/10 text-blue-500 border-blue-500/30",
  PUT: "bg-yellow-500/10 text-yellow-500 border-yellow-500/30",
  DELETE: "bg-red-500/10 text-red-500 border-red-500/30",
  PATCH: "bg-purple-500/10 text-purple-500 border-purple-500/30"
};

const sampleRequest = `curl -X POST https://api.platformforge.dev/v1/projects \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "my-project",
    "framework": "nextjs",
    "region": "us-east-1"
  }'`;

const sampleResponse = `{
  "id": "proj_a1b2c3d4e5",
  "name": "my-project",
  "framework": "nextjs",
  "region": "us-east-1",
  "status": "initializing",
  "created_at": "2025-01-15T10:30:00Z",
  "url": "https://my-project.platformforge.app"
}`;

export default function ApiReference() {
  return (
    <Layout>
      <div className="min-h-screen bg-background">
        <section className="py-20 border-b border-border relative overflow-hidden">
          <div className="absolute inset-0 bg-grid-pattern opacity-[0.02]" />
          <div className="container px-4 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center justify-center gap-3 mb-6">
                <Badge variant="outline" className="text-primary border-primary/30" data-testid="badge-api-reference">
                  <Code className="h-3 w-3 mr-1" /> API Reference
                </Badge>
                <Badge variant="secondary" className="font-mono" data-testid="badge-api-version">
                  v1.0.0
                </Badge>
              </div>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-api-title">
                API <span className="text-primary">Reference</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-10" data-testid="text-api-subtitle">
                Complete documentation for the Platform Forge REST API. 
                Build integrations, automate deployments, and manage your infrastructure programmatically.
              </p>

              <div className="flex flex-wrap justify-center gap-4">
                <Button variant="default" data-testid="button-get-api-key">
                  <Key className="h-4 w-4 mr-2" />
                  Get API Key
                </Button>
                <Button variant="outline" data-testid="button-view-examples">
                  <Terminal className="h-4 w-4 mr-2" />
                  View Examples
                </Button>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <div className="flex flex-col lg:flex-row gap-8">
              <motion.aside
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className="lg:w-64 shrink-0"
              >
                <div className="sticky top-24">
                  <h3 className="font-semibold mb-4 text-sm text-muted-foreground uppercase tracking-wider" data-testid="text-sidebar-title">
                    Endpoints
                  </h3>
                  <nav className="space-y-1">
                    {apiSections.map((section, index) => (
                      <a
                        key={section.title}
                        href={`#${section.title.toLowerCase()}`}
                        className="flex items-center gap-2 px-3 py-2 rounded-md text-sm hover-elevate cursor-pointer group"
                        data-testid={`link-sidebar-${section.title.toLowerCase()}`}
                      >
                        <section.icon className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                        <span className="group-hover:text-foreground transition-colors">{section.title}</span>
                        <ChevronRight className="h-3 w-3 ml-auto text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </a>
                    ))}
                  </nav>

                  <div className="mt-8 p-4 rounded-lg bg-card border border-border">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium">Base URL</span>
                    </div>
                    <code className="text-xs font-mono text-muted-foreground block bg-muted px-2 py-1 rounded" data-testid="text-base-url">
                      https://api.platformforge.dev
                    </code>
                  </div>
                </div>
              </motion.aside>

              <div className="flex-1 min-w-0">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="mb-12"
                >
                  <Card className="p-6 bg-card border-border" data-testid="card-code-example">
                    <div className="flex items-center justify-between gap-4 mb-4 flex-wrap">
                      <div className="flex items-center gap-2">
                        <Terminal className="h-5 w-5 text-primary" />
                        <h3 className="font-semibold" data-testid="text-example-title">Quick Example</h3>
                      </div>
                      <Button variant="ghost" size="sm" data-testid="button-copy-example">
                        <Copy className="h-4 w-4 mr-2" />
                        Copy
                      </Button>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Request</p>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-xs font-mono" data-testid="code-request">
                          <code>{sampleRequest}</code>
                        </pre>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Response</p>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-xs font-mono" data-testid="code-response">
                          <code>{sampleResponse}</code>
                        </pre>
                      </div>
                    </div>
                  </Card>
                </motion.div>

                <div className="space-y-16">
                  {apiSections.map((section, sectionIndex) => (
                    <motion.div
                      key={section.title}
                      id={section.title.toLowerCase()}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.1 }}
                      data-testid={`section-${section.title.toLowerCase()}`}
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <section.icon className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h2 className="text-xl font-bold" data-testid={`text-section-title-${sectionIndex}`}>
                            {section.title}
                          </h2>
                          <p className="text-sm text-muted-foreground" data-testid={`text-section-desc-${sectionIndex}`}>
                            {section.description}
                          </p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        {section.endpoints.map((endpoint, endpointIndex) => (
                          <Card 
                            key={`${endpoint.method}-${endpoint.path}`}
                            className="p-4 bg-card border-border hover-elevate cursor-pointer group"
                            data-testid={`card-endpoint-${sectionIndex}-${endpointIndex}`}
                          >
                            <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                              <Badge 
                                variant="outline" 
                                className={`${methodColors[endpoint.method]} font-mono text-xs shrink-0 w-fit`}
                                data-testid={`badge-method-${sectionIndex}-${endpointIndex}`}
                              >
                                {endpoint.method}
                              </Badge>
                              <code className="font-mono text-sm flex-1 truncate" data-testid={`text-path-${sectionIndex}-${endpointIndex}`}>
                                {endpoint.path}
                              </code>
                              <ExternalLink className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                            </div>
                            <p className="text-sm text-muted-foreground mt-2" data-testid={`text-endpoint-desc-${sectionIndex}-${endpointIndex}`}>
                              {endpoint.description}
                            </p>
                          </Card>
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 bg-card/30 border-t border-border">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center max-w-2xl mx-auto"
            >
              <div className="h-14 w-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <FileText className="h-7 w-7 text-primary" />
              </div>
              <h2 className="text-2xl font-bold mb-4" data-testid="text-help-title">Need More Help?</h2>
              <p className="text-muted-foreground mb-8" data-testid="text-help-subtitle">
                Explore our comprehensive guides, join the community, or reach out to our developer support team.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <Button variant="default" data-testid="button-view-guides">
                  <FileText className="h-4 w-4 mr-2" />
                  View Guides
                </Button>
                <Button variant="outline" data-testid="button-join-discord">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Join Discord
                </Button>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
