import { motion } from "framer-motion";
import {
  Activity,
  Server,
  Globe,
  Clock,
  Zap,
  Cloud,
  ArrowRight,
  Sparkles,
  Check,
  Cpu,
  Timer,
  Gauge
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface DeploymentFeature {
  text: string;
}

interface DeploymentType {
  icon: React.ElementType;
  title: string;
  description: string;
  features: DeploymentFeature[];
  color: string;
  badge?: string;
}

const deploymentTypes: DeploymentType[] = [
  {
    icon: Activity,
    title: "Autoscale",
    description: "Automatically scale your application based on traffic demand. Pay only for what you use with zero configuration.",
    features: [
      { text: "Scale from 0 to thousands of instances" },
      { text: "Automatic load balancing" },
      { text: "Pay-per-request pricing" },
      { text: "Zero cold starts option" }
    ],
    color: "from-green-500 to-emerald-600",
    badge: "Most Popular"
  },
  {
    icon: Server,
    title: "Reserved VM",
    description: "Dedicated virtual machines for consistent performance. Full control over your runtime environment.",
    features: [
      { text: "Guaranteed CPU and memory" },
      { text: "Persistent storage" },
      { text: "SSH access available" },
      { text: "Custom runtime configurations" }
    ],
    color: "from-blue-500 to-cyan-600"
  },
  {
    icon: Globe,
    title: "Static Hosting",
    description: "Lightning-fast static site hosting with global CDN distribution for your frontend applications.",
    features: [
      { text: "Global CDN with 300+ PoPs" },
      { text: "Automatic SSL certificates" },
      { text: "Instant cache invalidation" },
      { text: "Custom domain support" }
    ],
    color: "from-purple-500 to-violet-600"
  },
  {
    icon: Clock,
    title: "Scheduled",
    description: "Run jobs on a schedule with cron-like syntax. Perfect for background tasks and recurring operations.",
    features: [
      { text: "Cron expression support" },
      { text: "Timezone-aware scheduling" },
      { text: "Retry policies" },
      { text: "Execution logs and history" }
    ],
    color: "from-orange-500 to-amber-600"
  },
  {
    icon: Zap,
    title: "Edge",
    description: "Deploy to the edge for ultra-low latency. Run your code in 300+ locations worldwide.",
    features: [
      { text: "Sub-millisecond cold starts" },
      { text: "Global distribution" },
      { text: "Edge caching built-in" },
      { text: "WebSocket support" }
    ],
    color: "from-pink-500 to-rose-600",
    badge: "Beta"
  },
  {
    icon: Cloud,
    title: "Serverless",
    description: "Event-driven serverless functions that scale to zero. Ideal for APIs and microservices.",
    features: [
      { text: "Scale to zero when idle" },
      { text: "Event triggers (HTTP, queue, schedule)" },
      { text: "Automatic scaling" },
      { text: "Integrated monitoring" }
    ],
    color: "from-cyan-500 to-teal-600"
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

export default function Deployments() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-deployments">
                <Sparkles className="h-3 w-3 mr-1" /> Deployment Options
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-deployments-title">
                Deploy Your Way
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-deployments-subtitle">
                Choose the deployment strategy that fits your needs. From serverless functions
                to dedicated VMs, we have the infrastructure to power your applications.
              </p>

              <div className="flex justify-center gap-8 mt-10 flex-wrap">
                <div className="text-center" data-testid="stat-deployment-types">
                  <div className="text-4xl font-bold text-primary">{deploymentTypes.length}</div>
                  <div className="text-sm text-muted-foreground">Deployment Types</div>
                </div>
                <div className="text-center" data-testid="stat-regions">
                  <div className="text-4xl font-bold text-primary">300+</div>
                  <div className="text-sm text-muted-foreground">Edge Locations</div>
                </div>
                <div className="text-center" data-testid="stat-uptime">
                  <div className="text-4xl font-bold text-primary">99.99%</div>
                  <div className="text-sm text-muted-foreground">Uptime SLA</div>
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
              className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {deploymentTypes.map((deployment, index) => (
                <motion.div
                  key={deployment.title}
                  variants={itemVariants}
                  data-testid={`card-deployment-${deployment.title.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <Card className="p-6 bg-card border-border h-full hover-elevate relative overflow-visible">
                    {deployment.badge && (
                      <Badge 
                        className="absolute -top-2 right-4 bg-primary text-primary-foreground"
                        data-testid={`badge-deployment-${deployment.title.toLowerCase()}`}
                      >
                        {deployment.badge}
                      </Badge>
                    )}
                    <div className={`h-12 w-12 rounded-xl bg-gradient-to-br ${deployment.color} flex items-center justify-center mb-4`}>
                      <deployment.icon className="h-6 w-6 text-white" />
                    </div>
                    <h3 className="text-xl font-bold mb-2" data-testid={`text-deployment-title-${index}`}>
                      {deployment.title}
                    </h3>
                    <p className="text-muted-foreground mb-4" data-testid={`text-deployment-description-${index}`}>
                      {deployment.description}
                    </p>
                    <ul className="space-y-2">
                      {deployment.features.map((feature, featureIndex) => (
                        <li 
                          key={featureIndex} 
                          className="flex items-center gap-2 text-sm"
                          data-testid={`text-deployment-feature-${index}-${featureIndex}`}
                        >
                          <Check className="h-4 w-4 text-green-500 shrink-0" />
                          <span className="text-muted-foreground">{feature.text}</span>
                        </li>
                      ))}
                    </ul>
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
              <h2 className="text-3xl font-bold mb-4" data-testid="text-comparison-title">
                Quick Comparison
              </h2>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-comparison-subtitle">
                Choose the right deployment type for your use case
              </p>
            </motion.div>

            <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <Card className="p-6 text-center" data-testid="card-comparison-speed">
                  <Gauge className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Fastest Cold Start</h3>
                  <p className="text-2xl font-bold text-primary">Edge</p>
                  <p className="text-sm text-muted-foreground">Sub-millisecond</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <Card className="p-6 text-center" data-testid="card-comparison-cost">
                  <Timer className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Most Cost Effective</h3>
                  <p className="text-2xl font-bold text-primary">Serverless</p>
                  <p className="text-sm text-muted-foreground">Scale to zero</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
              >
                <Card className="p-6 text-center" data-testid="card-comparison-control">
                  <Cpu className="h-8 w-8 text-primary mx-auto mb-4" />
                  <h3 className="font-bold mb-2">Most Control</h3>
                  <p className="text-2xl font-bold text-primary">Reserved VM</p>
                  <p className="text-sm text-muted-foreground">Full access</p>
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
                Ready to Deploy?
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Get started with any deployment type in minutes. No credit card required for the free tier.
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
