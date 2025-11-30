import { motion } from "framer-motion";
import {
  Globe,
  Zap,
  Shield,
  Server,
  ArrowRight,
  Sparkles,
  Activity,
  Clock,
  Network,
  MapPin
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface Region {
  name: string;
  locations: number;
  latency: string;
  icon: React.ElementType;
}

interface NetworkFeature {
  icon: React.ElementType;
  title: string;
  description: string;
  stat: string;
  statLabel: string;
}

const regions: Region[] = [
  { name: "North America", locations: 85, latency: "<20ms", icon: MapPin },
  { name: "Europe", locations: 72, latency: "<25ms", icon: MapPin },
  { name: "Asia Pacific", locations: 68, latency: "<30ms", icon: MapPin },
  { name: "South America", locations: 28, latency: "<35ms", icon: MapPin },
  { name: "Middle East", locations: 18, latency: "<30ms", icon: MapPin },
  { name: "Africa", locations: 24, latency: "<40ms", icon: MapPin },
  { name: "Oceania", locations: 15, latency: "<25ms", icon: MapPin }
];

const features: NetworkFeature[] = [
  {
    icon: Zap,
    title: "Low Latency",
    description: "Serve content from the edge location nearest to your users for lightning-fast response times.",
    stat: "<50ms",
    statLabel: "Average Global Latency"
  },
  {
    icon: Globe,
    title: "Global CDN",
    description: "Content delivery network spanning 300+ points of presence across 100+ countries worldwide.",
    stat: "300+",
    statLabel: "Edge Locations"
  },
  {
    icon: Server,
    title: "Edge Caching",
    description: "Intelligent caching at the edge reduces origin load and improves performance for repeat visitors.",
    stat: "95%",
    statLabel: "Cache Hit Ratio"
  },
  {
    icon: Shield,
    title: "DDoS Protection",
    description: "Built-in DDoS mitigation absorbs attacks at the edge before they reach your infrastructure.",
    stat: "10Tbps+",
    statLabel: "Attack Mitigation"
  }
];

const stats = [
  { value: "300+", label: "Edge Locations" },
  { value: "100+", label: "Countries" },
  { value: "99.99%", label: "Uptime" },
  { value: "<50ms", label: "Global P95 Latency" }
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

export default function EdgeNetwork() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-edge-network">
                <Sparkles className="h-3 w-3 mr-1" /> Global Infrastructure
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-edge-title">
                Global <span className="text-primary">Edge Network</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-edge-subtitle">
                Deploy your applications to the edge and serve users from the nearest location.
                Ultra-low latency, global distribution, built-in protection.
              </p>

              <div className="flex justify-center gap-8 mt-10 flex-wrap">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center" data-testid={`stat-${index}`}>
                    <div className="text-4xl font-bold text-primary">{stat.value}</div>
                    <div className="text-sm text-muted-foreground">{stat.label}</div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-20">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-12"
            >
              <h2 className="text-3xl font-bold mb-4" data-testid="text-regions-title">
                Global Coverage
              </h2>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-regions-subtitle">
                Our edge network spans across all continents, ensuring low-latency access for your users worldwide
              </p>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              className="grid md:grid-cols-2 lg:grid-cols-4 gap-4"
            >
              {regions.map((region, index) => (
                <motion.div
                  key={region.name}
                  variants={itemVariants}
                  data-testid={`card-region-${region.name.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <Card className="p-4 bg-card border-border h-full hover-elevate">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Globe className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold" data-testid={`text-region-name-${index}`}>
                          {region.name}
                        </h3>
                      </div>
                    </div>
                    <div className="flex justify-between text-sm">
                      <div>
                        <span className="text-muted-foreground">Locations: </span>
                        <span className="font-medium" data-testid={`text-region-locations-${index}`}>{region.locations}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Latency: </span>
                        <span className="font-medium text-green-500" data-testid={`text-region-latency-${index}`}>{region.latency}</span>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="mt-8"
            >
              <Card className="p-6 bg-gradient-to-br from-primary/5 to-purple-500/5 border-primary/20" data-testid="card-total-coverage">
                <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="h-14 w-14 rounded-xl bg-primary/20 flex items-center justify-center">
                      <Network className="h-7 w-7 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold" data-testid="text-total-locations">Total: 310 Edge Locations</h3>
                      <p className="text-muted-foreground">Across 100+ countries and territories</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-green-500" />
                    <span className="text-green-500 font-medium" data-testid="text-network-status">All Systems Operational</span>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>
        </section>

        <section className="py-20 bg-card/30">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-12"
            >
              <h2 className="text-3xl font-bold mb-4" data-testid="text-features-title">
                Edge Features
              </h2>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-features-subtitle">
                Built-in capabilities that power high-performance applications
              </p>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid md:grid-cols-2 gap-6"
            >
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  variants={itemVariants}
                  data-testid={`card-feature-${feature.title.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <Card className="p-6 bg-card border-border h-full hover-elevate">
                    <div className="flex items-start gap-4">
                      <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                        <feature.icon className="h-6 w-6 text-primary" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-bold mb-2" data-testid={`text-feature-title-${index}`}>
                          {feature.title}
                        </h3>
                        <p className="text-muted-foreground mb-4" data-testid={`text-feature-description-${index}`}>
                          {feature.description}
                        </p>
                        <div className="flex items-baseline gap-2">
                          <span className="text-3xl font-bold text-primary" data-testid={`text-feature-stat-${index}`}>
                            {feature.stat}
                          </span>
                          <span className="text-sm text-muted-foreground">{feature.statLabel}</span>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center mb-12"
            >
              <h2 className="text-3xl font-bold mb-4" data-testid="text-performance-title">
                Performance Stats
              </h2>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-performance-subtitle">
                Real-time performance metrics from our global network
              </p>
            </motion.div>

            <div className="grid md:grid-cols-4 gap-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <Card className="p-6 text-center" data-testid="card-stat-requests">
                  <Clock className="h-8 w-8 text-primary mx-auto mb-4" />
                  <div className="text-3xl font-bold text-primary mb-1">1.2B+</div>
                  <p className="text-sm text-muted-foreground">Daily Requests</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <Card className="p-6 text-center" data-testid="card-stat-bandwidth">
                  <Activity className="h-8 w-8 text-primary mx-auto mb-4" />
                  <div className="text-3xl font-bold text-primary mb-1">50 PB</div>
                  <p className="text-sm text-muted-foreground">Monthly Bandwidth</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
              >
                <Card className="p-6 text-center" data-testid="card-stat-ttfb">
                  <Zap className="h-8 w-8 text-primary mx-auto mb-4" />
                  <div className="text-3xl font-bold text-primary mb-1">25ms</div>
                  <p className="text-sm text-muted-foreground">Median TTFB</p>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 }}
              >
                <Card className="p-6 text-center" data-testid="card-stat-uptime">
                  <Shield className="h-8 w-8 text-primary mx-auto mb-4" />
                  <div className="text-3xl font-bold text-primary mb-1">99.99%</div>
                  <p className="text-sm text-muted-foreground">Uptime SLA</p>
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
                Go Global Today
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Deploy to our edge network and reach users worldwide with minimal latency.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/">
                  <Button size="lg" className="font-bold" data-testid="button-start-building">
                    Start Building <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/features">
                  <Button size="lg" variant="outline" data-testid="button-view-features">
                    View All Features
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
