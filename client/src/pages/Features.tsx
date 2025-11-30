import { motion } from "framer-motion";
import {
  Brain,
  Code2,
  Languages,
  Database,
  HardDrive,
  Key,
  Lock,
  Cloud,
  Server,
  Globe,
  Zap,
  Activity,
  Shield,
  DollarSign,
  History,
  FileText,
  UserCheck,
  CreditCard,
  Bot,
  Search,
  ImageIcon,
  Sparkles,
  ArrowRight,
  Layers,
  Cpu
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface Feature {
  icon: React.ElementType;
  title: string;
  description: string;
}

interface FeatureCategory {
  title: string;
  description: string;
  icon: React.ElementType;
  color: string;
  features: Feature[];
}

const featureCategories: FeatureCategory[] = [
  {
    title: "AI Capabilities",
    description: "Intelligent code analysis and infrastructure generation powered by advanced AI",
    icon: Brain,
    color: "from-purple-500 to-violet-600",
    features: [
      {
        icon: Code2,
        title: "Code Analysis",
        description: "Deep semantic analysis of your codebase to understand architecture, dependencies, and patterns"
      },
      {
        icon: Layers,
        title: "Infrastructure Generation",
        description: "Automatically generate production-ready Terraform, Kubernetes, and Docker configurations"
      },
      {
        icon: Languages,
        title: "Multi-Language Support",
        description: "Support for JavaScript, TypeScript, Python, Go, Rust, Java, and more"
      }
    ]
  },
  {
    title: "Storage & Data",
    description: "Enterprise-grade data storage solutions with built-in redundancy and scaling",
    icon: Database,
    color: "from-cyan-500 to-blue-600",
    features: [
      {
        icon: Database,
        title: "PostgreSQL",
        description: "Managed PostgreSQL databases with automatic backups, scaling, and read replicas"
      },
      {
        icon: Key,
        title: "Key-Value Store",
        description: "High-performance Redis-compatible key-value storage for caching and sessions"
      },
      {
        icon: HardDrive,
        title: "Object Storage",
        description: "S3-compatible object storage for files, assets, and large binary data"
      },
      {
        icon: Lock,
        title: "Secrets Manager",
        description: "Secure vault for API keys, credentials, and sensitive configuration"
      }
    ]
  },
  {
    title: "Deployment",
    description: "Flexible deployment options from serverless to dedicated infrastructure",
    icon: Cloud,
    color: "from-green-500 to-emerald-600",
    features: [
      {
        icon: Activity,
        title: "Autoscale",
        description: "Automatically scale based on traffic with zero configuration required"
      },
      {
        icon: Server,
        title: "Reserved VM",
        description: "Dedicated virtual machines for consistent performance and full control"
      },
      {
        icon: Globe,
        title: "Static Hosting",
        description: "Lightning-fast static site hosting with global CDN distribution"
      },
      {
        icon: Zap,
        title: "Edge Deployments",
        description: "Deploy to the edge for ultra-low latency in 300+ locations worldwide"
      },
      {
        icon: Cpu,
        title: "Serverless Functions",
        description: "Event-driven serverless computing that scales to zero when idle"
      }
    ]
  },
  {
    title: "Security",
    description: "Enterprise security features to protect your applications and data",
    icon: Shield,
    color: "from-orange-500 to-red-600",
    features: [
      {
        icon: Shield,
        title: "Safety Guards",
        description: "Automated security scanning and vulnerability detection for your code"
      },
      {
        icon: DollarSign,
        title: "Cost Estimator",
        description: "Real-time cost estimation and budget alerts to prevent unexpected charges"
      },
      {
        icon: History,
        title: "Checkpoints",
        description: "Point-in-time recovery and rollback for databases and deployments"
      },
      {
        icon: FileText,
        title: "Audit Logging",
        description: "Comprehensive audit trails for compliance and security monitoring"
      }
    ]
  },
  {
    title: "Integrations",
    description: "Pre-built integrations with popular services and platforms",
    icon: Sparkles,
    color: "from-pink-500 to-rose-600",
    features: [
      {
        icon: UserCheck,
        title: "Auth System",
        description: "Built-in authentication with OAuth, SSO, and role-based access control"
      },
      {
        icon: CreditCard,
        title: "Stripe Payments",
        description: "Seamless Stripe integration for subscriptions, payments, and invoicing"
      },
      {
        icon: Bot,
        title: "Bots & Automations",
        description: "Create Discord bots, Slack apps, and scheduled automation workflows"
      },
      {
        icon: Search,
        title: "Web Search",
        description: "Integrated web search API for building AI-powered search experiences"
      },
      {
        icon: ImageIcon,
        title: "Image Generation",
        description: "AI-powered image generation for dynamic content creation"
      }
    ]
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

export default function Features() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-features">
                <Sparkles className="h-3 w-3 mr-1" /> Platform Capabilities
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-features-title">
                Platform <span className="text-primary">Features</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-features-subtitle">
                Everything you need to build, deploy, and scale enterprise-grade applications.
                From AI-powered code analysis to global edge deployments.
              </p>

              <div className="flex justify-center gap-8 mt-10 flex-wrap">
                <div className="text-center" data-testid="stat-categories">
                  <div className="text-4xl font-bold text-primary">{featureCategories.length}</div>
                  <div className="text-sm text-muted-foreground">Categories</div>
                </div>
                <div className="text-center" data-testid="stat-features">
                  <div className="text-4xl font-bold text-primary">
                    {featureCategories.reduce((acc, cat) => acc + cat.features.length, 0)}
                  </div>
                  <div className="text-sm text-muted-foreground">Features</div>
                </div>
                <div className="text-center" data-testid="stat-languages">
                  <div className="text-4xl font-bold text-primary">20+</div>
                  <div className="text-sm text-muted-foreground">Languages</div>
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
              className="space-y-24"
            >
              {featureCategories.map((category, categoryIndex) => (
                <motion.div
                  key={category.title}
                  variants={itemVariants}
                  className="relative"
                  data-testid={`category-${category.title.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <div className="flex flex-col md:flex-row items-start md:items-center gap-4 mb-8">
                    <div className={`h-14 w-14 rounded-xl bg-gradient-to-br ${category.color} flex items-center justify-center shrink-0`}>
                      <category.icon className="h-7 w-7 text-white" />
                    </div>
                    <div>
                      <h2 className="text-2xl md:text-3xl font-bold" data-testid={`text-category-title-${categoryIndex}`}>
                        {category.title}
                      </h2>
                      <p className="text-muted-foreground mt-1" data-testid={`text-category-description-${categoryIndex}`}>
                        {category.description}
                      </p>
                    </div>
                  </div>

                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {category.features.map((feature, featureIndex) => (
                      <motion.div
                        key={feature.title}
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: featureIndex * 0.1 }}
                      >
                        <Card
                          className="p-6 bg-card border-border h-full hover-elevate"
                          data-testid={`card-feature-${feature.title.toLowerCase().replace(/\s+/g, '-')}`}
                        >
                          <div className="flex items-start gap-4">
                            <div className={`h-10 w-10 rounded-lg bg-gradient-to-br ${category.color} bg-opacity-10 flex items-center justify-center shrink-0`}>
                              <feature.icon className="h-5 w-5 text-primary" />
                            </div>
                            <div>
                              <h3 className="font-semibold text-lg mb-2" data-testid={`text-feature-title-${categoryIndex}-${featureIndex}`}>
                                {feature.title}
                              </h3>
                              <p className="text-sm text-muted-foreground leading-relaxed" data-testid={`text-feature-description-${categoryIndex}-${featureIndex}`}>
                                {feature.description}
                              </p>
                            </div>
                          </div>
                        </Card>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </motion.div>
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
                Ready to Build?
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Start generating enterprise-grade infrastructure from your code today.
                No credit card required.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/">
                  <Button size="lg" className="font-bold" data-testid="button-start-building">
                    Start Building <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/compare">
                  <Button size="lg" variant="outline" data-testid="button-compare">
                    Compare Plans
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
