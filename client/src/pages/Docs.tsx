import { motion } from "framer-motion";
import { 
  Search,
  Book,
  FileText,
  Code,
  Database,
  Shield,
  Plug,
  Rocket,
  Github,
  Upload,
  Cpu,
  Server,
  Key,
  Lock,
  Users,
  Globe,
  Layers,
  Terminal,
  Cloud,
  HardDrive,
  ArrowRight,
  Zap
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface QuickStartCard {
  title: string;
  description: string;
  icon: React.ElementType;
  href: string;
  badge?: string;
}

interface DocCategory {
  title: string;
  description: string;
  icon: React.ElementType;
  items: DocItem[];
}

interface DocItem {
  title: string;
  description: string;
  href: string;
}

const quickStartCards: QuickStartCard[] = [
  {
    title: "Getting Started",
    description: "Learn the basics of Platform Forge and create your first project in minutes.",
    icon: Rocket,
    href: "/",
    badge: "Start Here"
  },
  {
    title: "GitHub Integration",
    description: "Connect your GitHub repositories and import existing projects seamlessly.",
    icon: Github,
    href: "#github-integration"
  },
  {
    title: "File Upload",
    description: "Upload your local code files directly to generate infrastructure.",
    icon: Upload,
    href: "#file-upload"
  },
  {
    title: "Infrastructure Generation",
    description: "Understand how AI analyzes your code and generates production-ready infrastructure.",
    icon: Cpu,
    href: "#infrastructure-generation"
  }
];

const docCategories: DocCategory[] = [
  {
    title: "API Reference",
    description: "Complete API documentation for programmatic access",
    icon: Code,
    items: [
      { title: "REST API Overview", description: "Authentication, endpoints, and rate limits", href: "#rest-api" },
      { title: "Project Endpoints", description: "Create, update, and manage projects", href: "#project-endpoints" },
      { title: "Deployment API", description: "Trigger and monitor deployments", href: "#deployment-api" },
      { title: "Webhooks", description: "Real-time event notifications", href: "#webhooks" }
    ]
  },
  {
    title: "Deployment Guides",
    description: "Step-by-step guides for deploying to any cloud",
    icon: Cloud,
    items: [
      { title: "AWS Deployment", description: "Deploy to Amazon Web Services with Terraform", href: "#aws-deployment" },
      { title: "GCP Deployment", description: "Google Cloud Platform integration guide", href: "#gcp-deployment" },
      { title: "Kubernetes Setup", description: "Configure and deploy to K8s clusters", href: "#kubernetes" },
      { title: "Multi-Region Deployments", description: "Scale globally with edge computing", href: "#multi-region" }
    ]
  },
  {
    title: "Database & Storage",
    description: "Manage your data layer and storage solutions",
    icon: Database,
    items: [
      { title: "PostgreSQL Setup", description: "Managed database configuration", href: "#postgresql" },
      { title: "Redis & Caching", description: "In-memory data store integration", href: "#redis" },
      { title: "Object Storage", description: "S3-compatible file storage", href: "#object-storage" },
      { title: "Database Migrations", description: "Schema versioning and rollbacks", href: "#migrations" }
    ]
  },
  {
    title: "Security & Auth",
    description: "Secure your applications and manage access",
    icon: Shield,
    items: [
      { title: "Authentication", description: "OAuth, JWT, and session management", href: "#authentication" },
      { title: "Secrets Management", description: "Secure environment variables", href: "#secrets" },
      { title: "IAM & Permissions", description: "Role-based access control", href: "#iam" },
      { title: "SSL/TLS Certificates", description: "HTTPS and certificate management", href: "#ssl" }
    ]
  },
  {
    title: "Integrations",
    description: "Connect with your favorite tools and services",
    icon: Plug,
    items: [
      { title: "CI/CD Pipelines", description: "GitHub Actions, GitLab CI, Jenkins", href: "#cicd" },
      { title: "Monitoring Tools", description: "Datadog, New Relic, Prometheus", href: "#monitoring" },
      { title: "Logging Services", description: "Centralized log aggregation", href: "#logging" },
      { title: "Third-Party APIs", description: "Stripe, Twilio, SendGrid, and more", href: "#third-party" }
    ]
  }
];

export default function Docs() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-docs">
                <Book className="h-3 w-3 mr-1" /> Documentation
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-docs-title">
                Platform Forge<br />
                <span className="text-primary">Documentation</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-10" data-testid="text-docs-subtitle">
                Everything you need to build, deploy, and scale enterprise-grade infrastructure.
                From quick starts to advanced configurations.
              </p>

              <div className="max-w-xl mx-auto">
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                  <Input
                    placeholder="Search documentation..."
                    className="pl-12 h-14 bg-card border-border text-lg"
                    data-testid="input-docs-search"
                  />
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <kbd className="pointer-events-none hidden sm:inline-flex h-7 select-none items-center gap-1 rounded border border-border bg-muted px-2 font-mono text-xs text-muted-foreground">
                      <span className="text-sm">Ctrl</span>K
                    </kbd>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-16 bg-card/30">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="mb-10"
            >
              <div className="flex items-center gap-2 mb-2">
                <Zap className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-quickstart-title">Quick Start</h2>
              </div>
              <p className="text-muted-foreground">Get up and running in minutes with these guides</p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {quickStartCards.map((card, index) => (
                <motion.div
                  key={card.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + index * 0.05 }}
                >
                  <Link href={card.href}>
                    <Card 
                      className="p-6 bg-card border-border h-full hover-elevate cursor-pointer group"
                      data-testid={`card-quickstart-${index}`}
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <card.icon className="h-5 w-5 text-primary" />
                        </div>
                        {card.badge && (
                          <Badge variant="secondary" className="text-xs" data-testid={`badge-quickstart-${index}`}>
                            {card.badge}
                          </Badge>
                        )}
                      </div>
                      <h3 className="font-semibold mb-2 group-hover:text-primary transition-colors" data-testid={`text-quickstart-title-${index}`}>
                        {card.title}
                      </h3>
                      <p className="text-sm text-muted-foreground" data-testid={`text-quickstart-desc-${index}`}>
                        {card.description}
                      </p>
                      <div className="mt-4 flex items-center text-sm text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                        <span>Read more</span>
                        <ArrowRight className="h-4 w-4 ml-1" />
                      </div>
                    </Card>
                  </Link>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-10"
            >
              <div className="flex items-center gap-2 mb-2">
                <Layers className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-categories-title">Documentation Categories</h2>
              </div>
              <p className="text-muted-foreground">Explore our comprehensive documentation by topic</p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {docCategories.map((category, catIndex) => (
                <motion.div
                  key={category.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + catIndex * 0.05 }}
                >
                  <Card 
                    className="p-6 bg-card border-border h-full"
                    data-testid={`card-category-${catIndex}`}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <category.icon className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold" data-testid={`text-category-title-${catIndex}`}>{category.title}</h3>
                        <p className="text-xs text-muted-foreground">{category.description}</p>
                      </div>
                    </div>

                    <div className="space-y-3">
                      {category.items.map((item, itemIndex) => (
                        <a
                          key={item.title}
                          href={item.href}
                          className="block p-3 rounded-md bg-background/50 hover-elevate cursor-pointer group"
                          data-testid={`link-doc-${catIndex}-${itemIndex}`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="min-w-0 flex-1">
                              <h4 className="text-sm font-medium group-hover:text-primary transition-colors truncate" data-testid={`text-doc-title-${catIndex}-${itemIndex}`}>
                                {item.title}
                              </h4>
                              <p className="text-xs text-muted-foreground truncate" data-testid={`text-doc-desc-${catIndex}-${itemIndex}`}>
                                {item.description}
                              </p>
                            </div>
                            <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary flex-shrink-0 ml-2 opacity-0 group-hover:opacity-100 transition-opacity" />
                          </div>
                        </a>
                      ))}
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 bg-card/30 border-t border-border">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-center max-w-2xl mx-auto"
            >
              <div className="h-14 w-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Terminal className="h-7 w-7 text-primary" />
              </div>
              <h2 className="text-2xl font-bold mb-4" data-testid="text-help-title">Need Help?</h2>
              <p className="text-muted-foreground mb-8" data-testid="text-help-subtitle">
                Can't find what you're looking for? Our team is here to help you build and scale your infrastructure.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <Button variant="default" data-testid="button-community">
                  <Users className="h-4 w-4 mr-2" />
                  Join Community
                </Button>
                <Button variant="outline" data-testid="button-support">
                  <FileText className="h-4 w-4 mr-2" />
                  Contact Support
                </Button>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
