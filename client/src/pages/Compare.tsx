import { motion } from "framer-motion";
import { 
  Check, 
  X, 
  Zap, 
  Server, 
  Database, 
  Globe, 
  Shield, 
  Layers, 
  GitBranch,
  Container,
  Cloud,
  Activity,
  Network,
  Cpu,
  HardDrive,
  ArrowRight,
  Sparkles
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import { useLocation } from "wouter";

interface FeatureRow {
  feature: string;
  category: string;
  replit: boolean | string;
  platformArchitect: boolean | string;
  highlight?: boolean;
}

const comparisonData: FeatureRow[] = [
  { category: "Development Environment", feature: "Browser-based IDE", replit: true, platformArchitect: true },
  { category: "Development Environment", feature: "Git Integration", replit: true, platformArchitect: true },
  { category: "Development Environment", feature: "Real-time Collaboration", replit: true, platformArchitect: true },
  { category: "Development Environment", feature: "Instant Preview", replit: true, platformArchitect: true },
  { category: "Development Environment", feature: "AI Code Assistance", replit: true, platformArchitect: true },
  { category: "Development Environment", feature: "Multi-language Support", replit: true, platformArchitect: true },
  
  { category: "Deployment", feature: "One-Click Deploy", replit: true, platformArchitect: true },
  { category: "Deployment", feature: "Custom Domains", replit: true, platformArchitect: true },
  { category: "Deployment", feature: "SSL/TLS Certificates", replit: true, platformArchitect: true },
  { category: "Deployment", feature: "Autoscale Deployments", replit: true, platformArchitect: true },
  { category: "Deployment", feature: "Static Hosting", replit: true, platformArchitect: true },
  { category: "Deployment", feature: "Reserved VMs", replit: true, platformArchitect: true },
  
  { category: "Database & Storage", feature: "Managed PostgreSQL", replit: true, platformArchitect: true },
  { category: "Database & Storage", feature: "Object Storage", replit: true, platformArchitect: true },
  { category: "Database & Storage", feature: "Key-Value Store", replit: true, platformArchitect: true },
  { category: "Database & Storage", feature: "Read Replicas", replit: false, platformArchitect: true, highlight: true },
  { category: "Database & Storage", feature: "Redis/Memcached Clusters", replit: false, platformArchitect: true, highlight: true },
  { category: "Database & Storage", feature: "Message Queues (RabbitMQ/Kafka)", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Infrastructure Complexity", feature: "Simple Web Apps", replit: true, platformArchitect: true },
  { category: "Infrastructure Complexity", feature: "APIs & Backends", replit: true, platformArchitect: true },
  { category: "Infrastructure Complexity", feature: "Microservices Architecture", replit: "Limited", platformArchitect: true, highlight: true },
  { category: "Infrastructure Complexity", feature: "Multi-Service Orchestration", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure Complexity", feature: "Event-Driven Systems", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure Complexity", feature: "Data Pipelines", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Infrastructure-as-Code", feature: "Terraform Generation", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure-as-Code", feature: "Kubernetes Manifests", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure-as-Code", feature: "Docker Compose (Production)", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure-as-Code", feature: "Helm Charts", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure-as-Code", feature: "Exportable Artifacts", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Scaling & Performance", feature: "Basic Auto-scaling", replit: true, platformArchitect: true },
  { category: "Scaling & Performance", feature: "Horizontal Pod Autoscaler", replit: false, platformArchitect: "2-1000 pods", highlight: true },
  { category: "Scaling & Performance", feature: "Custom Scaling Rules", replit: false, platformArchitect: true, highlight: true },
  { category: "Scaling & Performance", feature: "Load Balancer Configuration", replit: "Managed", platformArchitect: "Custom ALB/NLB" },
  { category: "Scaling & Performance", feature: "CDN Configuration", replit: true, platformArchitect: "Multi-region" },
  
  { category: "Cloud & Networking", feature: "Replit Cloud", replit: true, platformArchitect: true },
  { category: "Cloud & Networking", feature: "AWS Deployment", replit: false, platformArchitect: true, highlight: true },
  { category: "Cloud & Networking", feature: "GCP Deployment", replit: false, platformArchitect: true, highlight: true },
  { category: "Cloud & Networking", feature: "Azure Deployment", replit: false, platformArchitect: true, highlight: true },
  { category: "Cloud & Networking", feature: "VPC Configuration", replit: false, platformArchitect: true, highlight: true },
  { category: "Cloud & Networking", feature: "Private Subnets", replit: false, platformArchitect: true, highlight: true },
  { category: "Cloud & Networking", feature: "VPC Peering", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Security & Compliance", feature: "Secrets Management", replit: true, platformArchitect: true },
  { category: "Security & Compliance", feature: "Environment Variables", replit: true, platformArchitect: true },
  { category: "Security & Compliance", feature: "DDoS Protection", replit: "Basic", platformArchitect: "AWS Shield / Cloudflare" },
  { category: "Security & Compliance", feature: "WAF Rules", replit: false, platformArchitect: true, highlight: true },
  { category: "Security & Compliance", feature: "IAM Policies", replit: false, platformArchitect: true, highlight: true },
  { category: "Security & Compliance", feature: "Audit Logging", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Observability", feature: "Application Logs", replit: true, platformArchitect: true },
  { category: "Observability", feature: "Basic Metrics", replit: true, platformArchitect: true },
  { category: "Observability", feature: "Prometheus/Grafana Stack", replit: false, platformArchitect: true, highlight: true },
  { category: "Observability", feature: "Distributed Tracing", replit: false, platformArchitect: true, highlight: true },
  { category: "Observability", feature: "Custom Alerts", replit: false, platformArchitect: true, highlight: true },
  { category: "Observability", feature: "APM Integration", replit: false, platformArchitect: true, highlight: true },
];

const categories = Array.from(new Set(comparisonData.map(d => d.category)));

export default function Compare() {
  const [, setLocation] = useLocation();

  const replitFeatures = comparisonData.filter(d => d.replit === true || typeof d.replit === 'string').length;
  const platformFeatures = comparisonData.filter(d => d.platformArchitect === true || typeof d.platformArchitect === 'string').length;
  const exclusiveFeatures = comparisonData.filter(d => d.highlight).length;

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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-comparison">
                <Sparkles className="h-3 w-3 mr-1" /> Full Comparison
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-title">
                Everything Replit Has.<br />
                <span className="text-primary">Plus Enterprise Scale.</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-subtitle">
                PlatformArchitect is a production-grade version of Replit designed to handle 
                complex, enterprise-scale projects with real infrastructure-as-code output.
              </p>

              <div className="flex justify-center gap-8 mt-10">
                <div className="text-center">
                  <div className="text-4xl font-bold text-muted-foreground">{replitFeatures}</div>
                  <div className="text-sm text-muted-foreground">Replit Features</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">{platformFeatures}</div>
                  <div className="text-sm text-primary">PlatformArchitect Features</div>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-green-500">+{exclusiveFeatures}</div>
                  <div className="text-sm text-green-500">Enterprise Exclusives</div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-16 bg-card/30">
          <div className="container px-4">
            <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="p-8 bg-card border-border h-full" data-testid="card-replit">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="h-12 w-12 rounded-xl bg-orange-500/10 flex items-center justify-center">
                      <Zap className="h-6 w-6 text-orange-500" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold">Replit</h3>
                      <p className="text-sm text-muted-foreground">Great for prototypes & small apps</p>
                    </div>
                  </div>
                  
                  <div className="space-y-4 mb-8">
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Full Browser IDE</p>
                        <p className="text-sm text-muted-foreground">Write, run, and debug code instantly</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">One-Click Deployments</p>
                        <p className="text-sm text-muted-foreground">Managed hosting infrastructure</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Managed Database</p>
                        <p className="text-sm text-muted-foreground">PostgreSQL included</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <X className="h-5 w-5 text-muted-foreground/50 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium text-muted-foreground">Limited to Simple Architectures</p>
                        <p className="text-sm text-muted-foreground">Not designed for microservices</p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-border">
                    <p className="text-sm font-medium text-muted-foreground mb-3">Best For:</p>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="secondary" data-testid="badge-replit-usecase-1">Prototypes</Badge>
                      <Badge variant="secondary" data-testid="badge-replit-usecase-2">Side Projects</Badge>
                      <Badge variant="secondary" data-testid="badge-replit-usecase-3">Learning</Badge>
                      <Badge variant="secondary" data-testid="badge-replit-usecase-4">Simple Apps</Badge>
                    </div>
                  </div>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="p-8 bg-gradient-to-br from-primary/10 to-purple-500/10 border-primary/50 h-full relative overflow-hidden" data-testid="card-platform-architect">
                  <div className="absolute top-4 right-4">
                    <Badge className="bg-green-500 text-white" data-testid="badge-superset">Superset</Badge>
                  </div>
                  
                  <div className="flex items-center gap-3 mb-6">
                    <div className="h-12 w-12 rounded-xl bg-primary/20 flex items-center justify-center">
                      <Layers className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold">PlatformArchitect</h3>
                      <p className="text-sm text-muted-foreground">Production-grade Replit</p>
                    </div>
                  </div>
                  
                  <div className="space-y-4 mb-8">
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Everything Replit Offers</p>
                        <p className="text-sm text-muted-foreground">IDE, deployments, databases, collaboration</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium text-green-400">+ Complex Architectures</p>
                        <p className="text-sm text-muted-foreground">Microservices, event-driven, data pipelines</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium text-green-400">+ Infrastructure-as-Code</p>
                        <p className="text-sm text-muted-foreground">Terraform, Kubernetes, Helm, Docker</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium text-green-400">+ Multi-Cloud Deployment</p>
                        <p className="text-sm text-muted-foreground">AWS, GCP, Azure with VPC configuration</p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-primary/30">
                    <p className="text-sm font-medium text-muted-foreground mb-3">Best For:</p>
                    <div className="flex flex-wrap gap-2">
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-1">Enterprise</Badge>
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-2">Microservices</Badge>
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-3">High Scale</Badge>
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-4">Multi-Cloud</Badge>
                    </div>
                  </div>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4" data-testid="text-feature-comparison-title">Complete Feature Breakdown</h2>
              <p className="text-muted-foreground">
                <span className="text-green-500 font-medium">Green highlights</span> = Features unique to PlatformArchitect
              </p>
            </div>

            <div className="max-w-5xl mx-auto">
              {categories.map((category, catIdx) => (
                <motion.div
                  key={category}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: catIdx * 0.05 }}
                  className="mb-6"
                  data-testid={`section-category-${category.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <div className="flex items-center gap-2 mb-3">
                    {category === "Development Environment" && <GitBranch className="h-5 w-5 text-blue-400" />}
                    {category === "Deployment" && <Cloud className="h-5 w-5 text-green-400" />}
                    {category === "Database & Storage" && <Database className="h-5 w-5 text-orange-400" />}
                    {category === "Infrastructure Complexity" && <Layers className="h-5 w-5 text-purple-400" />}
                    {category === "Infrastructure-as-Code" && <Container className="h-5 w-5 text-cyan-400" />}
                    {category === "Scaling & Performance" && <Activity className="h-5 w-5 text-yellow-400" />}
                    {category === "Cloud & Networking" && <Network className="h-5 w-5 text-pink-400" />}
                    {category === "Security & Compliance" && <Shield className="h-5 w-5 text-red-400" />}
                    {category === "Observability" && <Cpu className="h-5 w-5 text-indigo-400" />}
                    <h3 className="text-lg font-bold">{category}</h3>
                  </div>

                  <Card className="overflow-hidden">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-border bg-muted/30">
                          <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Feature</th>
                          <th className="text-center py-3 px-4 text-sm font-medium text-orange-400 w-36">Replit</th>
                          <th className="text-center py-3 px-4 text-sm font-medium text-primary w-36">PlatformArchitect</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonData
                          .filter(row => row.category === category)
                          .map((row, idx) => (
                            <tr 
                              key={idx} 
                              className={`border-b border-border/50 last:border-0 ${row.highlight ? 'bg-green-500/5' : ''}`}
                              data-testid={`row-feature-${row.feature.toLowerCase().replace(/\s+/g, '-')}`}
                            >
                              <td className="py-3 px-4 text-sm">
                                {row.feature}
                                {row.highlight && (
                                  <Badge variant="outline" className="ml-2 text-[10px] py-0 text-green-500 border-green-500/30">
                                    Enterprise Only
                                  </Badge>
                                )}
                              </td>
                              <td className="text-center py-3 px-4">
                                {typeof row.replit === 'boolean' ? (
                                  row.replit ? (
                                    <Check className="h-5 w-5 text-green-500 mx-auto" />
                                  ) : (
                                    <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                                  )
                                ) : (
                                  <span className="text-xs text-muted-foreground">{row.replit}</span>
                                )}
                              </td>
                              <td className="text-center py-3 px-4">
                                {typeof row.platformArchitect === 'boolean' ? (
                                  row.platformArchitect ? (
                                    <Check className="h-5 w-5 text-primary mx-auto" />
                                  ) : (
                                    <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                                  )
                                ) : (
                                  <span className="text-xs text-primary font-medium">{row.platformArchitect}</span>
                                )}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 bg-card/30 border-t border-border">
          <div className="container px-4">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4" data-testid="text-output-title">The Difference: Exportable Infrastructure</h2>
              <p className="text-muted-foreground">Deploy anywhere, not just on our cloud</p>
            </div>

            <div className="max-w-5xl mx-auto grid md:grid-cols-4 gap-4">
              <Card className="p-5 bg-card border-orange-500/20" data-testid="card-output-terraform">
                <div className="h-10 w-10 rounded-lg bg-orange-500/10 flex items-center justify-center mb-3">
                  <HardDrive className="h-5 w-5 text-orange-400" />
                </div>
                <h3 className="font-bold mb-1">Terraform</h3>
                <p className="text-xs text-muted-foreground mb-3">
                  EKS, RDS, ElastiCache, VPC
                </p>
                <div className="bg-black/40 rounded p-2 font-mono text-[9px] text-orange-300">
                  resource "aws_eks_cluster"<br />
                  resource "aws_db_instance"
                </div>
              </Card>

              <Card className="p-5 bg-card border-blue-500/20" data-testid="card-output-kubernetes">
                <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center mb-3">
                  <Container className="h-5 w-5 text-blue-400" />
                </div>
                <h3 className="font-bold mb-1">Kubernetes</h3>
                <p className="text-xs text-muted-foreground mb-3">
                  Deployments, HPA, Ingress
                </p>
                <div className="bg-black/40 rounded p-2 font-mono text-[9px] text-blue-300">
                  kind: Deployment<br />
                  kind: HorizontalPodAutoscaler
                </div>
              </Card>

              <Card className="p-5 bg-card border-purple-500/20" data-testid="card-output-docker">
                <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center mb-3">
                  <Cloud className="h-5 w-5 text-purple-400" />
                </div>
                <h3 className="font-bold mb-1">Docker</h3>
                <p className="text-xs text-muted-foreground mb-3">
                  Optimized multi-stage builds
                </p>
                <div className="bg-black/40 rounded p-2 font-mono text-[9px] text-purple-300">
                  FROM node:18-alpine<br />
                  RUN npm ci --prod
                </div>
              </Card>

              <Card className="p-5 bg-card border-green-500/20" data-testid="card-output-helm">
                <div className="h-10 w-10 rounded-lg bg-green-500/10 flex items-center justify-center mb-3">
                  <Server className="h-5 w-5 text-green-400" />
                </div>
                <h3 className="font-bold mb-1">Helm Charts</h3>
                <p className="text-xs text-muted-foreground mb-3">
                  Parameterized deployments
                </p>
                <div className="bg-black/40 rounded p-2 font-mono text-[9px] text-green-300">
                  Chart.yaml<br />
                  values.yaml
                </div>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-20 border-t border-border">
          <div className="container px-4 text-center">
            <h2 className="text-3xl font-bold mb-4" data-testid="text-cta-title">Ready to Build Production Systems?</h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Same Replit experience you love. Enterprise infrastructure you need.
            </p>
            <Button 
              size="lg" 
              className="bg-primary hover:bg-primary/90"
              onClick={() => setLocation('/')}
              data-testid="button-start-building"
            >
              Start Building <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </section>
      </div>
    </Layout>
  );
}