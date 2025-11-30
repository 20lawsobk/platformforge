import { motion } from "framer-motion";
import {
  UserCheck,
  CreditCard,
  Database,
  HardDrive,
  Brain,
  Bot,
  Activity,
  ArrowRight,
  Sparkles,
  Check,
  Lock,
  Webhook,
  Mail,
  MessageSquare
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface IntegrationService {
  name: string;
}

interface Integration {
  icon: React.ElementType;
  title: string;
  description: string;
  services: IntegrationService[];
  color: string;
  badge?: string;
}

const integrations: Integration[] = [
  {
    icon: UserCheck,
    title: "Authentication",
    description: "Secure user authentication with multiple providers. OAuth, SSO, and social login out of the box.",
    services: [
      { name: "Google OAuth" },
      { name: "GitHub OAuth" },
      { name: "SAML SSO" },
      { name: "Magic Links" },
      { name: "Passkeys" }
    ],
    color: "from-blue-500 to-cyan-600"
  },
  {
    icon: CreditCard,
    title: "Payments (Stripe)",
    description: "Accept payments, manage subscriptions, and handle billing with Stripe integration.",
    services: [
      { name: "Stripe Checkout" },
      { name: "Subscriptions" },
      { name: "Invoicing" },
      { name: "Payment Links" },
      { name: "Customer Portal" }
    ],
    color: "from-purple-500 to-violet-600",
    badge: "Popular"
  },
  {
    icon: Database,
    title: "Databases",
    description: "Managed database solutions for every use case, from relational to NoSQL.",
    services: [
      { name: "PostgreSQL" },
      { name: "MySQL" },
      { name: "MongoDB" },
      { name: "Redis" },
      { name: "SQLite" }
    ],
    color: "from-green-500 to-emerald-600"
  },
  {
    icon: HardDrive,
    title: "Storage",
    description: "Object storage and file management for your applications with global CDN delivery.",
    services: [
      { name: "S3-Compatible Storage" },
      { name: "File Uploads" },
      { name: "Image Optimization" },
      { name: "CDN Delivery" },
      { name: "Backup & Restore" }
    ],
    color: "from-orange-500 to-amber-600"
  },
  {
    icon: Brain,
    title: "AI / LLM",
    description: "Integrate AI capabilities with popular language models and machine learning APIs.",
    services: [
      { name: "OpenAI GPT" },
      { name: "Anthropic Claude" },
      { name: "Google Gemini" },
      { name: "Replicate" },
      { name: "Hugging Face" }
    ],
    color: "from-pink-500 to-rose-600",
    badge: "New"
  },
  {
    icon: Bot,
    title: "Bots & Automation",
    description: "Build chatbots, Discord bots, Slack apps, and automated workflows.",
    services: [
      { name: "Discord Bot" },
      { name: "Slack App" },
      { name: "Telegram Bot" },
      { name: "Webhooks" },
      { name: "Scheduled Jobs" }
    ],
    color: "from-cyan-500 to-teal-600"
  },
  {
    icon: Activity,
    title: "Monitoring",
    description: "Application performance monitoring, logging, and alerting for production systems.",
    services: [
      { name: "Real-time Logs" },
      { name: "Error Tracking" },
      { name: "Performance Metrics" },
      { name: "Custom Alerts" },
      { name: "Uptime Monitoring" }
    ],
    color: "from-red-500 to-orange-600"
  },
  {
    icon: Mail,
    title: "Email & Notifications",
    description: "Transactional emails, push notifications, and SMS messaging services.",
    services: [
      { name: "SendGrid" },
      { name: "Resend" },
      { name: "Twilio SMS" },
      { name: "Push Notifications" },
      { name: "In-App Messages" }
    ],
    color: "from-indigo-500 to-purple-600"
  }
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5
    }
  }
};

export default function Integrations() {
  const totalServices = integrations.reduce((acc, int) => acc + int.services.length, 0);

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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-integrations">
                <Sparkles className="h-3 w-3 mr-1" /> Connect Everything
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-integrations-title">
                Powerful <span className="text-primary">Integrations</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-integrations-subtitle">
                Connect your applications with the services you already use. Authentication, payments,
                databases, AI, and more - all ready to integrate in minutes.
              </p>

              <div className="flex justify-center gap-8 mt-10 flex-wrap">
                <div className="text-center" data-testid="stat-categories">
                  <div className="text-4xl font-bold text-primary">{integrations.length}</div>
                  <div className="text-sm text-muted-foreground">Categories</div>
                </div>
                <div className="text-center" data-testid="stat-services">
                  <div className="text-4xl font-bold text-primary">{totalServices}+</div>
                  <div className="text-sm text-muted-foreground">Services</div>
                </div>
                <div className="text-center" data-testid="stat-setup">
                  <div className="text-4xl font-bold text-primary">5 min</div>
                  <div className="text-sm text-muted-foreground">Avg. Setup Time</div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-20">
          <div className="container px-4">
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
            >
              {integrations.map((integration, index) => (
                <motion.div
                  key={integration.title}
                  variants={itemVariants}
                  data-testid={`card-integration-${integration.title.toLowerCase().replace(/\s+/g, '-').replace(/[()\/]/g, '')}`}
                >
                  <Card className="p-6 bg-card border-border h-full hover-elevate relative overflow-visible">
                    {integration.badge && (
                      <Badge 
                        className="absolute -top-2 right-4 bg-primary text-primary-foreground"
                        data-testid={`badge-integration-${integration.title.toLowerCase()}`}
                      >
                        {integration.badge}
                      </Badge>
                    )}
                    <div className={`h-12 w-12 rounded-xl bg-gradient-to-br ${integration.color} flex items-center justify-center mb-4`}>
                      <integration.icon className="h-6 w-6 text-white" />
                    </div>
                    <h3 className="text-xl font-bold mb-2" data-testid={`text-integration-title-${index}`}>
                      {integration.title}
                    </h3>
                    <p className="text-sm text-muted-foreground mb-4" data-testid={`text-integration-description-${index}`}>
                      {integration.description}
                    </p>
                    <div className="space-y-1.5">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                        Connects to:
                      </p>
                      {integration.services.map((service, serviceIndex) => (
                        <div 
                          key={serviceIndex}
                          className="flex items-center gap-2 text-sm"
                          data-testid={`text-integration-service-${index}-${serviceIndex}`}
                        >
                          <Check className="h-3.5 w-3.5 text-green-500 shrink-0" />
                          <span className="text-muted-foreground">{service.name}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        <section className="py-16 bg-card/30">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-12"
            >
              <h2 className="text-3xl font-bold mb-4" data-testid="text-benefits-title">
                Why Use Our Integrations?
              </h2>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-benefits-subtitle">
                Pre-built, tested, and optimized for production use
              </p>
            </motion.div>

            <div className="grid md:grid-cols-4 gap-6 max-w-5xl mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <Card className="p-6 text-center" data-testid="card-benefit-setup">
                  <Sparkles className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">One-Click Setup</h3>
                  <p className="text-sm text-muted-foreground">Add integrations instantly without complex configuration</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <Card className="p-6 text-center" data-testid="card-benefit-secrets">
                  <Lock className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Secure Secrets</h3>
                  <p className="text-sm text-muted-foreground">API keys and credentials are encrypted and managed for you</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
              >
                <Card className="p-6 text-center" data-testid="card-benefit-webhooks">
                  <Webhook className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Webhooks Ready</h3>
                  <p className="text-sm text-muted-foreground">Event handling and webhooks configured automatically</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 }}
              >
                <Card className="p-6 text-center" data-testid="card-benefit-docs">
                  <MessageSquare className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Full Documentation</h3>
                  <p className="text-sm text-muted-foreground">Comprehensive guides and examples for every integration</p>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>

        <section className="py-20 border-t border-border relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
          <div className="container px-4 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-cta-title">
                Start Integrating
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Add powerful integrations to your application in minutes, not hours.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/">
                  <Button size="lg" className="font-bold" data-testid="button-start-building">
                    Start Building <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/docs">
                  <Button size="lg" variant="outline" data-testid="button-view-docs">
                    View Documentation
                  </Button>
                </Link>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
