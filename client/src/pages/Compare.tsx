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
  Clock
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
  planned?: boolean;
}

const comparisonData: FeatureRow[] = [
  { category: "Development", feature: "Browser-based IDE", replit: true, platformArchitect: false },
  { category: "Development", feature: "Git Integration", replit: true, platformArchitect: "Import from GitHub" },
  { category: "Development", feature: "Real-time Collaboration", replit: true, platformArchitect: false, planned: true },
  { category: "Development", feature: "Instant Preview", replit: true, platformArchitect: false },
  
  { category: "Code Generation", feature: "Infrastructure-as-Code Output", replit: false, platformArchitect: true, highlight: true },
  { category: "Code Generation", feature: "Terraform Configurations", replit: false, platformArchitect: true, highlight: true },
  { category: "Code Generation", feature: "Kubernetes Manifests", replit: false, platformArchitect: true, highlight: true },
  { category: "Code Generation", feature: "Dockerfile Generation", replit: false, platformArchitect: true, highlight: true },
  { category: "Code Generation", feature: "Downloadable Artifacts", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Deployment", feature: "One-Click Deploy", replit: true, platformArchitect: false, planned: true },
  { category: "Deployment", feature: "Autoscale Deployments", replit: true, platformArchitect: "Config Generated" },
  { category: "Deployment", feature: "Static Hosting", replit: true, platformArchitect: false },
  { category: "Deployment", feature: "Reserved VMs", replit: true, platformArchitect: "Config Generated" },
  
  { category: "Infrastructure Targets", feature: "AWS (EKS, RDS, ElastiCache)", replit: false, platformArchitect: true, highlight: true },
  { category: "Infrastructure Targets", feature: "GCP Support", replit: false, platformArchitect: true, planned: true },
  { category: "Infrastructure Targets", feature: "Azure Support", replit: false, platformArchitect: true, planned: true },
  { category: "Infrastructure Targets", feature: "Multi-Region Config", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Database Layer", feature: "Managed PostgreSQL", replit: true, platformArchitect: "Config Generated" },
  { category: "Database Layer", feature: "Redis Cache Config", replit: false, platformArchitect: true, highlight: true },
  { category: "Database Layer", feature: "Connection Pooling", replit: false, platformArchitect: true },
  
  { category: "Scaling", feature: "Auto-scaling Rules", replit: "Managed", platformArchitect: "HPA Config (2-1000)" },
  { category: "Scaling", feature: "Load Balancer Config", replit: "Managed", platformArchitect: "ALB/Ingress YAML" },
  { category: "Scaling", feature: "Horizontal Pod Autoscaler", replit: false, platformArchitect: true, highlight: true },
  
  { category: "Security", feature: "Secrets Management", replit: true, platformArchitect: "K8s Secrets Config" },
  { category: "Security", feature: "VPC Configuration", replit: false, platformArchitect: true, highlight: true },
  { category: "Security", feature: "Private Subnet Config", replit: false, platformArchitect: true, highlight: true },
];

const categories = Array.from(new Set(comparisonData.map(d => d.category)));

export default function Compare() {
  const [, setLocation] = useLocation();

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
                Platform Comparison
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-title">
                Replit vs <span className="text-primary">PlatformArchitect</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-subtitle">
                <span className="text-foreground font-medium">Different tools for different jobs.</span>{" "}
                Replit excels at rapid prototyping. We generate the infrastructure code needed for production scale.
              </p>
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
                      <p className="text-sm text-muted-foreground">Complete development platform</p>
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
                        <p className="text-sm text-muted-foreground">Managed hosting with auto-scaling</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Real-time Collaboration</p>
                        <p className="text-sm text-muted-foreground">Code together like Google Docs</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Managed Database</p>
                        <p className="text-sm text-muted-foreground">PostgreSQL included and managed</p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-border">
                    <p className="text-sm font-medium text-muted-foreground mb-3">Best For:</p>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="secondary" data-testid="badge-replit-usecase-1">Prototypes</Badge>
                      <Badge variant="secondary" data-testid="badge-replit-usecase-2">Side Projects</Badge>
                      <Badge variant="secondary" data-testid="badge-replit-usecase-3">Learning</Badge>
                      <Badge variant="secondary" data-testid="badge-replit-usecase-4">Small Apps</Badge>
                    </div>
                  </div>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="p-8 bg-gradient-to-br from-primary/5 to-purple-500/5 border-primary/30 h-full relative overflow-hidden" data-testid="card-platform-architect">
                  <div className="absolute top-4 right-4">
                    <Badge className="bg-primary text-primary-foreground" data-testid="badge-infrastructure">Infrastructure</Badge>
                  </div>
                  
                  <div className="flex items-center gap-3 mb-6">
                    <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center">
                      <Layers className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold">PlatformArchitect</h3>
                      <p className="text-sm text-muted-foreground">AI infrastructure code generator</p>
                    </div>
                  </div>
                  
                  <div className="space-y-4 mb-8">
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Terraform Generation</p>
                        <p className="text-sm text-muted-foreground">Complete AWS infrastructure code</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Kubernetes Manifests</p>
                        <p className="text-sm text-muted-foreground">Deployments, services, HPA configs</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Production-Ready Dockerfiles</p>
                        <p className="text-sm text-muted-foreground">Optimized for your stack</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <div>
                        <p className="font-medium">Downloadable Artifacts</p>
                        <p className="text-sm text-muted-foreground">Take configs to your own cloud</p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-primary/20">
                    <p className="text-sm font-medium text-muted-foreground mb-3">Best For:</p>
                    <div className="flex flex-wrap gap-2">
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-1">Enterprise Infra</Badge>
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-2">IaC Generation</Badge>
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-3">K8s Configs</Badge>
                      <Badge className="bg-primary/20 text-primary border-0" data-testid="badge-pa-usecase-4">Cloud Migration</Badge>
                    </div>
                  </div>
                </Card>
              </motion.div>
            </div>

            <div className="max-w-5xl mx-auto mt-8 p-4 bg-muted/30 rounded-lg border border-border">
              <p className="text-sm text-muted-foreground text-center">
                <strong className="text-foreground">Note:</strong> These platforms serve different purposes. 
                Replit is a complete development environment. PlatformArchitect generates infrastructure configurations 
                that you deploy to your own cloud accounts.
              </p>
            </div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold mb-4" data-testid="text-feature-comparison-title">Feature Comparison</h2>
              <p className="text-muted-foreground">Detailed breakdown of capabilities</p>
            </div>

            <div className="max-w-4xl mx-auto">
              {categories.map((category, catIdx) => (
                <motion.div
                  key={category}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: catIdx * 0.1 }}
                  className="mb-8"
                  data-testid={`section-category-${category.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <div className="flex items-center gap-2 mb-4">
                    {category === "Development" && <GitBranch className="h-5 w-5 text-blue-400" />}
                    {category === "Code Generation" && <Container className="h-5 w-5 text-purple-400" />}
                    {category === "Deployment" && <Cloud className="h-5 w-5 text-green-400" />}
                    {category === "Infrastructure Targets" && <Server className="h-5 w-5 text-orange-400" />}
                    {category === "Database Layer" && <Database className="h-5 w-5 text-cyan-400" />}
                    {category === "Scaling" && <Activity className="h-5 w-5 text-yellow-400" />}
                    {category === "Security" && <Shield className="h-5 w-5 text-red-400" />}
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
                              className={`border-b border-border/50 last:border-0 ${row.highlight ? 'bg-primary/5' : ''}`}
                              data-testid={`row-feature-${row.feature.toLowerCase().replace(/\s+/g, '-')}`}
                            >
                              <td className="py-3 px-4 text-sm">
                                {row.feature}
                                {row.highlight && (
                                  <Badge variant="outline" className="ml-2 text-[10px] py-0 text-primary border-primary/30">
                                    Key Diff
                                  </Badge>
                                )}
                                {row.planned && (
                                  <Badge variant="outline" className="ml-2 text-[10px] py-0 text-yellow-500 border-yellow-500/30">
                                    <Clock className="h-2 w-2 mr-1" /> Planned
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
              <h2 className="text-3xl font-bold mb-4" data-testid="text-output-title">What Gets Generated</h2>
              <p className="text-muted-foreground">Export production-ready infrastructure code</p>
            </div>

            <div className="max-w-4xl mx-auto grid md:grid-cols-3 gap-6">
              <Card className="p-6 bg-card border-orange-500/20" data-testid="card-output-terraform">
                <div className="h-10 w-10 rounded-lg bg-orange-500/10 flex items-center justify-center mb-4">
                  <HardDrive className="h-5 w-5 text-orange-400" />
                </div>
                <h3 className="font-bold mb-2">Terraform</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  AWS infrastructure: EKS, RDS, ElastiCache, VPC
                </p>
                <div className="bg-black/40 rounded-md p-3 font-mono text-[10px] text-orange-300">
                  resource "aws_eks_cluster" {`{`}<br />
                  &nbsp;&nbsp;name = "prod-cluster"<br />
                  &nbsp;&nbsp;role_arn = aws_iam...<br />
                  {`}`}
                </div>
              </Card>

              <Card className="p-6 bg-card border-blue-500/20" data-testid="card-output-kubernetes">
                <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center mb-4">
                  <Container className="h-5 w-5 text-blue-400" />
                </div>
                <h3 className="font-bold mb-2">Kubernetes</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Deployments, services, HPA, ingress configs
                </p>
                <div className="bg-black/40 rounded-md p-3 font-mono text-[10px] text-blue-300">
                  apiVersion: apps/v1<br />
                  kind: Deployment<br />
                  spec:<br />
                  &nbsp;&nbsp;replicas: 3
                </div>
              </Card>

              <Card className="p-6 bg-card border-purple-500/20" data-testid="card-output-docker">
                <div className="h-10 w-10 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4">
                  <Cloud className="h-5 w-5 text-purple-400" />
                </div>
                <h3 className="font-bold mb-2">Docker</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Optimized Dockerfiles for your language/framework
                </p>
                <div className="bg-black/40 rounded-md p-3 font-mono text-[10px] text-purple-300">
                  FROM node:18-alpine<br />
                  WORKDIR /app<br />
                  RUN npm ci --prod<br />
                  CMD ["npm", "start"]
                </div>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-20 border-t border-border">
          <div className="container px-4 text-center">
            <h2 className="text-3xl font-bold mb-4" data-testid="text-cta-title">Ready to Generate Infrastructure?</h2>
            <p className="text-muted-foreground mb-8 max-w-xl mx-auto">
              Start with a GitHub repo. Get Terraform, Kubernetes, and Docker configurations in seconds.
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